"""Laptop-safe GLiNER fine-tuning helpers."""

from __future__ import annotations

import dataclasses
import inspect
import json
import random
import shutil
from pathlib import Path
from typing import Any

import yaml
from transformers import TrainerCallback

from .dataset import read_examples
from .gliner_data import to_gliner_dataset
from .schema import ALLOWED_LABELS, Example


class ListDataset:
    """Small torch-compatible dataset without requiring datasets/arrow for local runs."""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


class FixedLabelCollator:
    """Pin GLiNER's per-batch entity vocabulary to MEDliNER's closed label schema.

    GLiNER otherwise derives the batch's entity types from the gold labels that happen to be
    present, plus sampled negatives. A batch containing only no-entity examples then has zero
    entity types and the loss fails on ``scores.view(BS, -1, CL)`` with ``CL == 0`` -- which at
    ``per_device_train_batch_size: 1`` means the first reviewed empty example kills the run.
    Empty examples are deliberate training signal here, and evaluation always queries all three
    labels, so fixing the vocabulary matches inference as well as keeping the loss well-defined.
    """

    def __init__(self, collator: Any, labels: list[str]) -> None:
        self.collator = collator
        self.labels = labels

    def __call__(self, features: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        # A list-of-lists is required: the flat form leaves `classes_to_id` as a single dict
        # and GLiNER's `create_labels` then indexes it positionally.
        return self.collator(features, entity_types=[self.labels for _ in features], **kwargs)


def load_config(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"training config {path} must be a mapping")
    return value


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _device() -> str:
    """Use CUDA only when the installed wheel contains kernels for this GPU."""
    try:
        import torch

        if not torch.cuda.is_available():
            return "cpu"
        major, minor = torch.cuda.get_device_capability()
        arch = f"sm_{major}{minor}"
        if arch not in set(torch.cuda.get_arch_list()):
            return "cpu"
        return "cuda"
    except (ImportError, RuntimeError):
        return "cpu"


def _precision(config: dict[str, Any], device: str) -> tuple[bool, bool]:
    fp16 = bool(config.get("fp16", False))
    bf16 = bool(config.get("bf16", False))
    if fp16 and bf16:
        raise ValueError("training config sets both fp16 and bf16; choose one")
    fp16 = fp16 and device == "cuda"
    bf16 = bf16 and device == "cuda"
    if bf16:
        try:
            import torch

            bf16 = bool(torch.cuda.is_bf16_supported())
        except (ImportError, AttributeError):
            bf16 = False
    return fp16, bf16


def _latest_checkpoint(output_dir: Path) -> str | None:
    """Newest `checkpoint-<step>` directory, ignoring `final/` and any other sibling output."""
    numbered = [
        (int(path.name.rsplit("-", 1)[-1]), path)
        for path in output_dir.glob("checkpoint-*")
        if path.is_dir() and path.name.rsplit("-", 1)[-1].isdigit()
    ]
    return str(max(numbered)[1]) if numbered else None


def _eval_strategy_field() -> str:
    """`evaluation_strategy` was renamed to `eval_strategy` and dropped in transformers 5."""
    from gliner.training import TrainingArguments

    names = {field.name for field in dataclasses.fields(TrainingArguments)}
    return "eval_strategy" if "eval_strategy" in names else "evaluation_strategy"


def _training_arguments(model: Any, config: dict[str, Any], output_dir: Path, device: str) -> Any:
    fp16, bf16 = _precision(config, device)
    eval_steps = int(config.get("eval_steps", 25))
    save_steps = int(config.get("save_steps", 25))
    if eval_steps <= 0 or save_steps <= 0:
        raise ValueError(f"eval_steps ({eval_steps}) and save_steps ({save_steps}) must be positive")
    if save_steps % eval_steps:
        # `best_model_checkpoint` is only recorded when an evaluation lands on a saved step, so a
        # non-multiple would leave `_export_best_checkpoint` with nothing better than the last step.
        raise ValueError(f"save_steps ({save_steps}) must be a multiple of eval_steps ({eval_steps})")
    common = {
        "output_dir": str(output_dir),
        "learning_rate": float(config.get("learning_rate", 5e-5)),
        "weight_decay": float(config.get("weight_decay", 0.01)),
        "per_device_train_batch_size": int(config.get("per_device_train_batch_size", 1)),
        "per_device_eval_batch_size": int(config.get("per_device_eval_batch_size", 1)),
        "gradient_accumulation_steps": int(config.get("gradient_accumulation_steps", 8)),
        "num_train_epochs": float(config.get("num_train_epochs", 3)),
        "max_steps": int(config.get("max_steps", -1)),
        "max_grad_norm": float(config.get("max_grad_norm", 1.0)),
        "lr_scheduler_type": str(config.get("lr_scheduler_type", "linear")),
        "warmup_ratio": float(config.get("warmup_ratio", 0.1)),
        "logging_steps": int(config.get("logging_steps", 1)),
        "eval_steps": eval_steps,
        "save_steps": save_steps,
        "save_total_limit": int(config.get("save_total_limit", 2)),
        "dataloader_num_workers": int(config.get("dataloader_num_workers", 0)),
        "seed": int(config.get("seed", 2026)),
        "fp16": fp16,
        "bf16": bf16,
        "report_to": [],
        "use_cpu": device != "cuda",
        "remove_unused_columns": False,
        # `load_best_model_at_end` cannot work here: GLiNER writes the *inner* module's
        # state dict (`token_rep_layer.*`) while transformers reloads it into the GLiNER
        # wrapper (`model.token_rep_layer.*`), so every key mismatches and the reload is a
        # silent no-op that leaves the last step's weights in place. `metric_for_best_model`
        # still records `best_model_checkpoint`, and `_export_best_checkpoint` copies it.
        "load_best_model_at_end": False,
        "metric_for_best_model": "eval_strict_f1",
        "greater_is_better": True,
        "save_strategy": "steps",
        _eval_strategy_field(): "steps",
    }
    return model.create_training_args(**common)


# Trainer bookkeeping, not model weights; a published checkpoint should not carry it.
_TRAINER_STATE_FILES = frozenset(
    {"optimizer.pt", "scheduler.pt", "rng_state.pth", "scaler.pt", "trainer_state.json", "training_args.bin"}
)


def _export_best_checkpoint(trainer: Any, final_dir: Path) -> str | None:
    """Materialize the best-validation checkpoint as `final/`, falling back to the last step."""
    best = getattr(trainer.state, "best_model_checkpoint", None)
    source = Path(best) if best else None
    if source is None or not source.is_dir():
        trainer.save_model(str(final_dir))
        return None
    final_dir.mkdir(parents=True, exist_ok=True)
    for item in sorted(source.iterdir()):
        if item.is_file() and item.name not in _TRAINER_STATE_FILES:
            shutil.copy2(item, final_dir / item.name)
    return str(source)


def load_model(model_id: str, *, device: str, max_length: int | None = None) -> Any:
    from gliner import GLiNER

    kwargs: dict[str, Any] = {"map_location": device}
    if max_length is not None:
        kwargs["max_length"] = max_length
    return GLiNER.from_pretrained(model_id, **kwargs)


def _enable_memory_saving(model: Any, config: dict[str, Any]) -> None:
    if bool(config.get("freeze_text_encoder", False)) and hasattr(model, "freeze_component"):
        model.freeze_component("text_encoder")
    if not bool(config.get("gradient_checkpointing", True)):
        return
    encoder = getattr(getattr(getattr(model, "model", None), "token_rep_layer", None), "bert_layer", None)
    encoder_model = getattr(encoder, "model", None)
    enable = getattr(encoder_model, "gradient_checkpointing_enable", None)
    if callable(enable):
        enable()


class ValidationF1Callback(TrainerCallback):
    """Stop on reviewed validation strict F1 and expose it to Trainer model selection."""

    def __init__(
        self, examples: list[Example], *, labels: list[str], threshold: float = 0.3, patience: int = 2
    ) -> None:
        self.examples = examples
        self.labels = labels
        self.threshold = threshold
        self.patience = patience
        self.best_f1 = -1.0
        self.bad_evaluations = 0

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        if model is None or metrics is None:
            return control
        from .evaluation import score_examples

        was_training = bool(model.training)
        model.eval()

        def predict(text: str) -> list[dict[str, Any]]:
            return model.predict_entities(text, self.labels, threshold=self.threshold)

        try:
            report = score_examples(predict, self.examples)
        finally:
            if was_training:
                model.train()
        f1 = float(report["overall"]["strict"]["f1"])
        metrics["eval_strict_f1"] = f1
        if f1 > self.best_f1:
            self.best_f1 = f1
            self.bad_evaluations = 0
        else:
            self.bad_evaluations += 1
            if self.bad_evaluations >= self.patience:
                control.should_training_stop = True
        return control


def _make_trainer(
    model: Any,
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
    eval_examples: list[Example],
    args: Any,
    config: dict[str, Any],
) -> Any:
    from gliner.training import Trainer

    labels = list(config.get("labels") or ALLOWED_LABELS)
    callback = ValidationF1Callback(
        eval_examples,
        labels=labels,
        threshold=float(config.get("evaluation_threshold", 0.3)),
        patience=int(config.get("early_stopping_patience", 2)),
    )
    kwargs: dict[str, Any] = {
        "model": model,
        "args": args,
        "train_dataset": ListDataset(train_records),
        "eval_dataset": ListDataset(eval_records),
        "data_collator": FixedLabelCollator(model._create_data_collator(), labels),
        "callbacks": [callback],
    }
    signature = inspect.signature(Trainer).parameters
    if "processing_class" in signature:
        kwargs["processing_class"] = model.data_processor.transformer_tokenizer
    else:
        kwargs["tokenizer"] = model.data_processor.transformer_tokenizer
    return Trainer(**kwargs)


def train_from_split_directory(
    split_dir: str | Path,
    output_dir: str | Path,
    *,
    config_path: str | Path = "configs/train-small.yaml",
    resume_from_checkpoint: str | None = None,
    smoke_test: bool = False,
) -> Path:
    """Train and save a checkpoint. A one-step smoke test uses the same code path."""
    config = load_config(config_path)
    if smoke_test:
        config["max_steps"] = 1
        config["num_train_epochs"] = 1
        config["save_steps"] = 1
        config["eval_steps"] = 1
        # A smoke run is a fresh one-step sanity check. Auto-discovering the previous run's
        # checkpoint would append a step each time it is re-run instead of repeating the check.
        config["resume"] = False
    seed = int(config.get("seed", 2026))
    _seed_everything(seed)
    split_dir = Path(split_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _device()
    model_id = str(config.get("model_id", "urchade/gliner_small-v2.1"))
    model = load_model(model_id, device=device, max_length=int(config.get("max_length", 384)))
    _enable_memory_saving(model, config)
    train_examples = read_examples(split_dir / "train.jsonl")
    eval_examples = read_examples(split_dir / "validation.jsonl")
    if not train_examples or not eval_examples:
        raise ValueError("training requires non-empty train and validation splits")
    train_records = to_gliner_dataset(train_examples, model=model)
    eval_records = to_gliner_dataset(eval_examples, model=model)
    args = _training_arguments(model, config, output_dir, device)
    trainer = _make_trainer(model, train_records, eval_records, eval_examples, args, config)
    resume = resume_from_checkpoint or (_latest_checkpoint(output_dir) if bool(config.get("resume", True)) else None)
    trainer.train(resume_from_checkpoint=resume)
    final_dir = output_dir / "final"
    selected_checkpoint = _export_best_checkpoint(trainer, final_dir)
    metadata = {
        "model_id": model_id,
        "labels": list(config.get("labels") or ALLOWED_LABELS),
        "device": device,
        "seed": seed,
        "smoke_test": smoke_test,
        "config": config,
        "train_examples": len(train_examples),
        "validation_examples": len(eval_examples),
        "resume_from_checkpoint": resume,
        "selected_checkpoint": selected_checkpoint,
        "best_validation_strict_f1": next(
            (
                callback.best_f1
                for callback in trainer.callback_handler.callbacks
                if isinstance(callback, ValidationF1Callback)
            ),
            None,
        ),
    }
    (final_dir / "medliner-training.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return final_dir


__all__ = [
    "FixedLabelCollator",
    "ListDataset",
    "ValidationF1Callback",
    "load_config",
    "load_model",
    "train_from_split_directory",
]
