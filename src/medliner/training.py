"""Laptop-safe GLiNER fine-tuning helpers."""

from __future__ import annotations

import inspect
import json
import random
from pathlib import Path
from typing import Any

import yaml
from transformers import TrainerCallback

from .dataset import read_examples
from .gliner_data import to_gliner_dataset


class ListDataset:
    """Small torch-compatible dataset without requiring datasets/arrow for local runs."""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


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
    fp16 = bool(config.get("fp16", False)) and device == "cuda"
    bf16 = bool(config.get("bf16", False)) and device == "cuda"
    if bf16:
        try:
            import torch

            bf16 = bool(torch.cuda.is_bf16_supported())
        except (ImportError, AttributeError):
            bf16 = False
    return fp16, bf16


def _latest_checkpoint(output_dir: Path) -> str | None:
    checkpoints = sorted(output_dir.glob("checkpoint-*"), key=lambda path: int(path.name.rsplit("-", 1)[-1]))
    return str(checkpoints[-1]) if checkpoints else None


def _training_arguments(model: Any, config: dict[str, Any], output_dir: Path, device: str) -> Any:
    fp16, bf16 = _precision(config, device)
    common = {
        "output_dir": str(output_dir),
        "learning_rate": float(config.get("learning_rate", 5e-5)),
        "weight_decay": float(config.get("weight_decay", 0.01)),
        "per_device_train_batch_size": int(config.get("per_device_train_batch_size", 1)),
        "per_device_eval_batch_size": int(config.get("per_device_eval_batch_size", 1)),
        "gradient_accumulation_steps": int(config.get("gradient_accumulation_steps", 8)),
        "num_train_epochs": float(config.get("num_train_epochs", 3)),
        "max_steps": int(config.get("max_steps", -1)),
        "warmup_ratio": float(config.get("warmup_ratio", 0.1)),
        "logging_steps": int(config.get("logging_steps", 1)),
        "eval_steps": int(config.get("eval_steps", 25)),
        "save_steps": int(config.get("save_steps", 25)),
        "save_total_limit": int(config.get("save_total_limit", 2)),
        "fp16": fp16,
        "bf16": bf16,
        "report_to": [],
        "use_cpu": device != "cuda",
        "remove_unused_columns": False,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_strict_f1",
        "greater_is_better": True,
        "evaluation_strategy": "steps",
        "save_strategy": "steps",
    }
    try:
        return model.create_training_args(**common)
    except TypeError:
        common["eval_strategy"] = common.pop("evaluation_strategy")
        return model.create_training_args(**common)


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

    def __init__(self, examples: list[Any], *, labels: list[str], threshold: float = 0.3, patience: int = 2) -> None:
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

        report = score_examples(predict, self.examples)
        f1 = float(report["overall"]["strict"]["f1"])
        metrics["eval_strict_f1"] = f1
        if f1 > self.best_f1:
            self.best_f1 = f1
            self.bad_evaluations = 0
        else:
            self.bad_evaluations += 1
            if self.bad_evaluations >= self.patience:
                control.should_training_stop = True
        if was_training:
            model.train()
        return control


def _make_trainer(
    model: Any,
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
    eval_examples: list[Any],
    args: Any,
    config: dict[str, Any],
) -> Any:
    from gliner.training import Trainer

    callback = ValidationF1Callback(
        eval_examples,
        labels=["disease", "phenotype", "drug"],
        threshold=float(config.get("evaluation_threshold", 0.3)),
        patience=int(config.get("early_stopping_patience", 2)),
    )
    kwargs: dict[str, Any] = {
        "model": model,
        "args": args,
        "train_dataset": ListDataset(train_records),
        "eval_dataset": ListDataset(eval_records),
        "data_collator": model._create_data_collator(),
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
    seed = int(config.get("seed", 2026))
    _seed_everything(seed)
    split_dir = Path(split_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _device()
    model = load_model(
        str(config.get("model_id", "urchade/gliner_small-v2.1")),
        device=device,
        max_length=int(config.get("max_length", 384)),
    )
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
    trainer.save_model(str(final_dir))
    metadata = {
        "model_id": config.get("model_id", "urchade/gliner_small-v2.1"),
        "device": device,
        "seed": seed,
        "smoke_test": smoke_test,
        "config": config,
        "train_examples": len(train_examples),
        "validation_examples": len(eval_examples),
        "resume_from_checkpoint": resume,
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


__all__ = ["ListDataset", "load_config", "load_model", "train_from_split_directory"]
