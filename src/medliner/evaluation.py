"""Strict and lenient medical NER evaluation."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dataset import read_examples
from .gliner_data import split_words
from .schema import ALLOWED_LABELS, Annotation, Example

Predictor = Callable[[str], list[dict[str, Any]]]


def benchmark_path() -> Path:
    """Path of the gold benchmark: `$MEDLINER_BENCHMARK`, else the workdir default.

    Resolution matches ``cli.workdir()`` (``$MEDLINER_WORKDIR`` or ``data/materialized``).
    No existence guarantee: callers decide how to report a missing file.
    """
    override = os.environ.get("MEDLINER_BENCHMARK")
    if override:
        return Path(override)
    workdir = Path(os.environ.get("MEDLINER_WORKDIR", "data/materialized"))
    return workdir / "ingested" / "ner_gold.json"


@dataclass(frozen=True)
class Counts:
    tp: int
    fp: int
    fn: int

    def __add__(self, other: Counts) -> Counts:
        return Counts(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


@dataclass(frozen=True)
class ExampleScore:
    """Per-example counts, so each example is predicted exactly once per report."""

    example: Example
    strict: Counts
    boundary: Counts
    predicted_any: bool

    @property
    def is_negative(self) -> bool:
        return not self.example.annotations


def _score(predictions: set[tuple[Any, ...]], gold: set[tuple[Any, ...]]) -> Counts:
    return Counts(tp=len(predictions & gold), fp=len(predictions - gold), fn=len(gold - predictions))


def _normalize_prediction(item: dict[str, Any]) -> tuple[int, int, str] | None:
    start = int(item["start"])
    end = int(item["end"])
    if start >= end:
        return None
    return start, end, str(item.get("label", item.get("type", ""))).strip().lower()


def score_example(predictor: Predictor, example: Example) -> ExampleScore:
    predicted = {span for span in (_normalize_prediction(item) for item in predictor(example.text)) if span is not None}
    gold_strict = {(annotation.start, annotation.end, annotation.label) for annotation in example.annotations}
    predicted_boundary = {(start, end) for start, end, _label in predicted}
    gold_boundary = {(annotation.start, annotation.end) for annotation in example.annotations}
    return ExampleScore(
        example=example,
        strict=_score(predicted, gold_strict),
        boundary=_score(predicted_boundary, gold_boundary),
        predicted_any=bool(predicted),
    )


def _aggregate(scores: Iterable[ExampleScore]) -> dict[str, Any]:
    strict = Counts(0, 0, 0)
    boundary = Counts(0, 0, 0)
    count = 0
    for score in scores:
        strict += score.strict
        boundary += score.boundary
        count += 1
    return {"strict": strict.as_dict(), "boundary_only": boundary.as_dict(), "examples": count}


def _truncation_report(values: list[Example], max_words: int | None) -> dict[str, Any]:
    """GLiNER truncates over-budget inputs with only a warning, which silently depresses recall."""
    if not max_words:
        return {"max_words": None, "checked": False}
    over = [example.id for example in values if len(split_words(example.text)) > max_words]
    return {"max_words": max_words, "checked": True, "over_budget_examples": len(over), "example_ids": over[:20]}


def score_examples(
    predictor: Predictor, examples: Iterable[Example], *, max_words: int | None = None
) -> dict[str, Any]:
    """Score once per example, then slice the same counts by task and source family."""
    values = list(examples)
    if max_words is None:
        max_words = getattr(predictor, "max_words", None)
    scores = [score_example(predictor, example) for example in values]
    by_task: dict[str, list[ExampleScore]] = defaultdict(list)
    by_source: dict[str, list[ExampleScore]] = defaultdict(list)
    for score in scores:
        by_task[score.example.task].append(score)
        by_source[score.example.source.family].append(score)
    negatives = [score for score in scores if score.is_negative]
    false_positives = [score for score in negatives if score.predicted_any]
    overall = _aggregate(scores)
    return {
        "overall": {"strict": overall["strict"], "boundary_only": overall["boundary_only"]},
        "by_task": {key: _aggregate(items) for key, items in sorted(by_task.items())},
        "by_source": {key: _aggregate(items) for key, items in sorted(by_source.items())},
        "no_entity": {
            "examples": len(negatives),
            "false_positive_examples": len(false_positives),
            "false_positive_rate": len(false_positives) / len(negatives) if negatives else 0.0,
        },
        "truncation": _truncation_report(values, max_words),
        "examples": len(values),
    }


class GLiNERPredictor:
    """Callable wrapper that also advertises the model's word budget to the scorer."""

    def __init__(self, model: Any, *, threshold: float, labels: Iterable[str] = ALLOWED_LABELS) -> None:
        self.model = model
        self.threshold = threshold
        self.labels = list(labels)
        max_len = getattr(getattr(model, "config", None), "max_len", None)
        self.max_words: int | None = max_len if isinstance(max_len, int) else None

    def __call__(self, text: str) -> list[dict[str, Any]]:
        result = self.model.predict_entities(text, self.labels, threshold=self.threshold)
        return [
            {
                "start": int(item["start"]),
                "end": int(item["end"]),
                "label": str(item.get("label", item.get("type", ""))).lower(),
                "text": str(item.get("text", text[int(item["start"]) : int(item["end"])])),
                "score": float(item.get("score", 0.0)),
            }
            for item in result
        ]


def make_gliner_predictor(checkpoint: str | Path, *, threshold: float = 0.3) -> Predictor:
    from gliner import GLiNER

    model = GLiNER.from_pretrained(str(checkpoint), map_location="cuda" if _cuda_available() else "cpu")
    model.eval()
    return GLiNERPredictor(model, threshold=threshold)


def _cuda_available() -> bool:
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        major, minor = torch.cuda.get_device_capability()
        return f"sm_{major}{minor}" in set(torch.cuda.get_arch_list())
    except (ImportError, RuntimeError):
        return False


def load_gold_benchmark(path: str | Path) -> list[Example]:
    """Load the ingested DAKP gold benchmark without adding it to training."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    examples: list[Example] = []
    for item in payload["cases"]:
        text = str(item["text"])
        annotations: list[Annotation] = []
        for index, mention in enumerate(item["mentions"]):
            surface = str(mention["surface"])
            start = mention.get("start")
            if start is None:
                if text.count(surface) != 1:
                    raise ValueError(
                        f"benchmark case {item['id']!r} surface {surface!r} occurs {text.count(surface)} times; "
                        "add an explicit 'start' offset to disambiguate it"
                    )
                start = text.index(surface)
            start = int(start)
            if text[start : start + len(surface)] != surface:
                raise ValueError(f"benchmark case {item['id']!r} offset {start} does not contain {surface!r}")
            annotations.append(
                Annotation(
                    id=f"{item['id']}-{index}",
                    start=start,
                    end=start + len(surface),
                    label=str(mention["type"]),
                    text=surface,
                )
            )
        source = str(item.get("source", "unknown"))
        examples.append(
            Example(
                id=str(item["id"]),
                text=text,
                task="contraindication" if source == "dailymed" else "indication",
                source={"family": source, "document_id": str(item["id"])},
                annotations=annotations,
                annotation_status="adjudicated",
            )
        )
    return examples


def evaluate_checkpoint(
    checkpoint: str | Path,
    split_dir: str | Path,
    output_path: str | Path,
    *,
    include_baselines: bool = True,
    threshold: float = 0.3,
) -> dict[str, Any]:
    """Evaluate systems after validating gold, so an actionable ingest error wins model failures."""
    # Validate and load gold before model or split work: a missing benchmark is actionable, while
    # model loading can fail for unrelated reasons and would otherwise mask the ingest hint.
    gold_path = benchmark_path()
    if not gold_path.exists():
        raise RuntimeError(
            f"gold benchmark not found at {gold_path}; point $MEDLINER_BENCHMARK at an existing "
            "ner_gold.json (or run `medliner ingest` with the older bundle layout)"
        )
    benchmark = load_gold_benchmark(gold_path)

    tuned_predictor = make_gliner_predictor(checkpoint, threshold=threshold)
    split_dir = Path(split_dir)
    test_path = split_dir / "test.jsonl"
    evaluated_split = "test" if test_path.exists() else "validation"
    values = read_examples(test_path if test_path.exists() else split_dir / "validation.jsonl")
    result: dict[str, Any] = {
        "tuned": score_examples(tuned_predictor, values),
        "checkpoint": str(checkpoint),
        "evaluated_split": evaluated_split,
        "threshold": threshold,
    }
    result["tuned_gold_regression"] = score_examples(tuned_predictor, benchmark)
    if include_baselines:
        result["baselines"] = {}
        metadata_path = Path(checkpoint) / "medliner-training.json"
        base_model_id = None
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            base_model_id = metadata.get("model_id")
        if base_model_id:
            try:
                untuned = make_gliner_predictor(str(base_model_id), threshold=threshold)
                result["baselines"]["untuned_gliner"] = score_examples(untuned, values)
                result["baselines"]["untuned_gliner_gold_regression"] = score_examples(untuned, benchmark)
            except Exception as exc:  # noqa: BLE001 - a missing baseline must not void the tuned report
                result["baselines"]["untuned_gliner_error"] = f"{type(exc).__name__}: {exc}"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


__all__ = [
    "Counts",
    "ExampleScore",
    "GLiNERPredictor",
    "benchmark_path",
    "evaluate_checkpoint",
    "load_gold_benchmark",
    "make_gliner_predictor",
    "score_example",
    "score_examples",
]
