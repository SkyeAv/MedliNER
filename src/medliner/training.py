"""Laptop-safe GLiNER fine-tuning helpers."""

from __future__ import annotations

import dataclasses
import inspect
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any

import yaml
from gliner.training import Trainer as GLiNERTrainer
from transformers import TrainerCallback

from .dataset import hash_file, read_examples
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
    """Pin GLiNER's per-batch entity vocabulary to MedliNER's closed label schema.

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


class WeightedCollator:
    """``FixedLabelCollator`` plus a per-batch ``sample_weight`` tensor for gold/synthetic mixes.

    GLiNER reduces the whole batch to a single loss scalar, so a batch can only carry one
    weight: gold (1.0) and synthetic examples must never share a batch, because their gradients
    could not be scaled apart afterwards. Mixed-weight batches — only possible with
    ``per_device_train_batch_size > 1`` — fail loudly here instead of silently averaging the
    two populations. The trainer pops the tensor before the model call, so the model itself
    never sees it.
    """

    def __init__(self, collator: Any, labels: list[str]) -> None:
        self.fixed = FixedLabelCollator(collator, labels)

    def __call__(self, features: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        import torch

        weights = [float(feature.get("weight", 1.0)) for feature in features]
        distinct = sorted(set(weights))
        if len(distinct) > 1:
            raise ValueError(
                f"batch mixes sample weights {distinct[0]} and {distinct[-1]} "
                f"({len(features)} examples; requires per_device_train_batch_size > 1); the batch "
                "loss is one scalar, so gold (1.0) and synthetic examples cannot share a batch — "
                "set per_device_train_batch_size to 1 or group examples by weight"
            )
        batch = self.fixed(features, **kwargs)
        batch["sample_weight"] = torch.tensor(weights, dtype=torch.float32)
        return batch


class WeightedTrainer(GLiNERTrainer):
    """GLiNER trainer that scales the batch loss by its ``sample_weight``.

    Only ``compute_loss`` is overridden: the inherited ``training_step`` keeps the CUDA-OOM
    skip and the gradient-accumulation division untouched, so a weighted micro-batch is scaled
    *before* accumulation — exactly the standard weighted-sum objective. The popped tensor is
    consumed here and never forwarded to the model; weight 1.0 everywhere therefore reproduces
    the unweighted loss bit-for-bit.
    """

    def compute_loss(
        self,
        model: Any,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ) -> Any:
        weights = inputs.pop("sample_weight", None)
        if weights is not None and bool((weights != weights[0]).any()):
            # Defense in depth: WeightedCollator already rejects mixed batches at collate time;
            # fail here too, before the forward pass spends work on an unscaleable batch.
            raise ValueError(
                f"sample_weight must be uniform within a batch, got {round(float(weights.min()), 6)} and "
                f"{round(float(weights.max()), 6)}; scale batches with WeightedCollator"
            )
        result = super().compute_loss(
            model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
        )
        if weights is None:
            return result
        loss, outputs = result if return_outputs else (result, None)
        scaled = loss * weights[0]
        return (scaled, outputs) if return_outputs else scaled


def load_config(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"training config {path} must be a mapping")
    if "synthetic_weight" in value:
        # The gold weight is fixed at 1.0, so a synthetic weight above 1.0 would up-weight the
        # noisier population and 0/negative would silently drop it from the loss entirely.
        raw = value["synthetic_weight"]
        try:
            synthetic_weight = float(raw)
        except (TypeError, ValueError):
            synthetic_weight = float("nan")
        if not 0 < synthetic_weight <= 1:
            raise ValueError(f"training config {path}: synthetic_weight must be a number in (0, 1], got {raw!r}")
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
    *,
    weighted: bool = False,
) -> Any:
    labels = list(config.get("labels") or ALLOWED_LABELS)
    callback = ValidationF1Callback(
        eval_examples,
        labels=labels,
        threshold=float(config.get("evaluation_threshold", 0.3)),
        patience=int(config.get("early_stopping_patience", 2)),
    )
    # Weight 1.0 everywhere must stay bit-equivalent to the historical gold-only path, so the
    # plain FixedLabelCollator + GLiNER Trainer pair is kept verbatim whenever no synthetic
    # records are mixed in; the weighted pair only ever handles a real mix.
    collator: Any
    trainer_cls: Any
    if weighted:
        collator = WeightedCollator(model._create_data_collator(), labels)
        trainer_cls = WeightedTrainer
    else:
        collator = FixedLabelCollator(model._create_data_collator(), labels)
        trainer_cls = GLiNERTrainer
    kwargs: dict[str, Any] = {
        "model": model,
        "args": args,
        "train_dataset": ListDataset(train_records),
        "eval_dataset": ListDataset(eval_records),
        "data_collator": collator,
        "callbacks": [callback],
    }
    signature = inspect.signature(trainer_cls).parameters
    if "processing_class" in signature:
        kwargs["processing_class"] = model.data_processor.transformer_tokenizer
    else:
        kwargs["tokenizer"] = model.data_processor.transformer_tokenizer
    return trainer_cls(**kwargs)


def _synthetic_pool_path() -> Path:
    """Where `medliner synthesize` materializes the pool; matches ``cli.workdir()`` resolution."""
    return Path(os.environ.get("MEDLINER_WORKDIR", "data/materialized")) / "synthetic" / "examples.jsonl"


def _load_synthetic_examples(config: dict[str, Any], *, no_synthetic: bool) -> tuple[list[Example], str | None]:
    """Load the synthetic pool for down-weighted mixing into the train split.

    A configured ``synthetic_weight`` with no pool is a hard error, not a silent gold-only run:
    the operator asked for a semi-supervised mix and must either generate the pool or opt out
    explicitly with ``--no-synthetic``.
    """
    if no_synthetic:
        return [], None
    path = _synthetic_pool_path()
    if not path.exists():
        if "synthetic_weight" in config:
            raise ValueError(
                f"config sets synthetic_weight={config['synthetic_weight']!r} but the synthetic pool "
                f"is missing at {path}; generate it ('medliner synthesize') or train gold-only "
                "with --no-synthetic"
            )
        return [], None
    return read_examples(path), hash_file(path)


def _assert_no_synthetic_in_held_out(synthetic: list[Example], held_out: list[Example]) -> None:
    """Held-out splits must measure gold performance only; synthetic ids never enter them."""
    synthetic_ids = {example.id for example in synthetic}
    overlap = sorted(synthetic_ids & {example.id for example in held_out})
    assert not overlap, f"synthetic examples leaked into validation/test: {overlap[:5]} (of {len(overlap)})"


def train_from_split_directory(
    split_dir: str | Path,
    output_dir: str | Path,
    *,
    config_path: str | Path = "configs/train-small.yaml",
    resume_from_checkpoint: str | None = None,
    smoke_test: bool = False,
    no_synthetic: bool = False,
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
    test_path = split_dir / "test.jsonl"
    held_out_examples = eval_examples + (read_examples(test_path) if test_path.exists() else [])
    synthetic_weight = float(config.get("synthetic_weight", 1.0))
    synthetic_examples, synthetic_dataset_hash = _load_synthetic_examples(config, no_synthetic=no_synthetic)
    _assert_no_synthetic_in_held_out(synthetic_examples, held_out_examples)
    train_records = to_gliner_dataset(train_examples, model=model)
    synthetic_records = (
        to_gliner_dataset(synthetic_examples, model=model, weight=synthetic_weight) if synthetic_examples else []
    )
    eval_records = to_gliner_dataset(eval_examples, model=model)
    args = _training_arguments(model, config, output_dir, device)
    trainer = _make_trainer(
        model,
        [*train_records, *synthetic_records],
        eval_records,
        eval_examples,
        args,
        config,
        weighted=bool(synthetic_records),
    )
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
        "train_examples": len(train_examples) + len(synthetic_examples),
        "gold_train_examples": len(train_examples),
        "synthetic_examples": len(synthetic_examples),
        "synthetic_weight": synthetic_weight,
        "synthetic_dataset_hash": synthetic_dataset_hash,
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
    "WeightedCollator",
    "WeightedTrainer",
    "load_config",
    "load_model",
    "train_from_split_directory",
]
