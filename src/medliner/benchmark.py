"""Gold-benchmark loading and strict/boundary span scoring for the pre-labeler."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schema import Annotation, Example, canonical_label

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
    return start, end, canonical_label(str(item.get("label", item.get("type", "")))) or ""


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


def score_examples(predictor: Predictor, examples: Iterable[Example]) -> dict[str, Any]:
    """Score once per example, then slice the same counts by task and source family."""
    values = list(examples)
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
        "examples": len(values),
    }


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


__all__ = ["Counts", "ExampleScore", "benchmark_path", "load_gold_benchmark", "score_example", "score_examples"]
