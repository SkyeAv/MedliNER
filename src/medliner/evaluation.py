"""Strict and lenient medical NER evaluation."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dataset import read_examples
from .schema import ALLOWED_LABELS, Annotation, Example

Predictor = Callable[[str], list[dict[str, Any]]]


@dataclass(frozen=True)
class Counts:
    tp: int
    fp: int
    fn: int

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


def _prediction_tuple(item: dict[str, Any], *, typed: bool) -> tuple[Any, ...]:
    start = int(item["start"])
    end = int(item["end"])
    if typed:
        label = str(item.get("label", item.get("type", ""))).strip().lower()
        return start, end, label
    return start, end


def _score(predictions: set[tuple[Any, ...]], gold: set[tuple[Any, ...]]) -> Counts:
    return Counts(tp=len(predictions & gold), fp=len(predictions - gold), fn=len(gold - predictions))


def _overall(predictor: Predictor, values: list[Example]) -> tuple[Counts, Counts, int, int]:
    strict_total = Counts(0, 0, 0)
    boundary_total = Counts(0, 0, 0)
    negative_count = 0
    negative_false_positive = 0
    for example in values:
        raw = predictor(example.text)
        predicted = {
            (int(item["start"]), int(item["end"]), str(item.get("label", item.get("type", ""))).lower())
            for item in raw
            if int(item["start"]) < int(item["end"])
        }
        gold_strict = {(annotation.start, annotation.end, annotation.label) for annotation in example.annotations}
        predicted_boundary = {(start, end) for start, end, _label in predicted}
        gold_boundary = {(annotation.start, annotation.end) for annotation in example.annotations}
        strict = _score(predicted, gold_strict)
        boundary = _score(predicted_boundary, gold_boundary)
        strict_total = Counts(strict_total.tp + strict.tp, strict_total.fp + strict.fp, strict_total.fn + strict.fn)
        boundary_total = Counts(
            boundary_total.tp + boundary.tp, boundary_total.fp + boundary.fp, boundary_total.fn + boundary.fn
        )
        if not gold_strict:
            negative_count += 1
            negative_false_positive += bool(predicted)
    return strict_total, boundary_total, negative_count, negative_false_positive


def score_examples(predictor: Predictor, examples: Iterable[Example]) -> dict[str, Any]:
    values = list(examples)
    strict_total, boundary_total, negative_count, negative_false_positive = _overall(predictor, values)
    by_task: dict[str, list[Example]] = defaultdict(list)
    by_source: dict[str, list[Example]] = defaultdict(list)
    for example in values:
        by_task[example.task].append(example)
        by_source[example.source.family].append(example)

    def grouped(items: list[Example]) -> dict[str, Any]:
        strict, boundary, _negative, _false_positive = _overall(predictor, items)
        return {"strict": strict.as_dict(), "boundary_only": boundary.as_dict()}

    return {
        "overall": {"strict": strict_total.as_dict(), "boundary_only": boundary_total.as_dict()},
        "by_task": {key: grouped(items) for key, items in sorted(by_task.items())},
        "by_source": {key: grouped(items) for key, items in sorted(by_source.items())},
        "no_entity": {
            "examples": negative_count,
            "false_positive_examples": negative_false_positive,
            "false_positive_rate": negative_false_positive / negative_count if negative_count else 0.0,
        },
        "examples": len(values),
    }


def make_gliner_predictor(checkpoint: str | Path, *, threshold: float = 0.3) -> Predictor:
    from gliner import GLiNER

    model = GLiNER.from_pretrained(str(checkpoint), map_location="cuda" if _cuda_available() else "cpu")

    def predict(text: str) -> list[dict[str, Any]]:
        result = model.predict_entities(text, list(ALLOWED_LABELS), threshold=threshold)
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

    return predict


def make_dakp_gazetteer_predictor(dakp_root: str | Path = "../DAKP") -> Predictor:
    """Load DAKP's deterministic offline gazetteer when the sibling checkout is available."""
    root = Path(dakp_root).resolve()
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from dakp_pipeline.ner.ner import DiseaseNER

    ner = DiseaseNER(offline=True)

    def predict(text: str) -> list[dict[str, Any]]:
        return [
            {"start": mention.start, "end": mention.end, "label": mention.type, "text": mention.text}
            for mention in ner.extract(text)
        ]

    return predict


def _cuda_available() -> bool:
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        major, minor = torch.cuda.get_device_capability()
        return f"sm_{major}{minor}" in set(torch.cuda.get_arch_list())
    except (ImportError, RuntimeError):
        return False


def load_dakp_benchmark(path: str | Path = "../DAKP/tests/eval/ner_gold.json") -> list[Example]:
    """Load DAKP's committed regression fixture without adding it to training."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    examples: list[Example] = []
    for item in payload["cases"]:
        text = str(item["text"])
        annotations: list[Annotation] = []
        for index, mention in enumerate(item["mentions"]):
            surface = str(mention["surface"])
            if text.count(surface) != 1:
                raise ValueError(f"benchmark surface is not unique: {surface!r}")
            start = text.index(surface)
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
) -> dict[str, Any]:
    """Evaluate tuned, untuned, and available gazetteer systems."""
    tuned_predictor = make_gliner_predictor(checkpoint)
    split_dir = Path(split_dir)
    test_path = split_dir / "test.jsonl"
    values = read_examples(test_path if test_path.exists() else split_dir / "validation.jsonl")
    result: dict[str, Any] = {"tuned": score_examples(tuned_predictor, values), "checkpoint": str(checkpoint)}
    benchmark_path = Path("../DAKP/tests/eval/ner_gold.json")
    if benchmark_path.exists():
        result["tuned_dakp_regression"] = score_examples(tuned_predictor, load_dakp_benchmark(benchmark_path))
    if include_baselines:
        result["baselines"] = {}
        metadata_path = Path(checkpoint) / "medliner-training.json"
        base_model_id = None
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            base_model_id = metadata.get("model_id")
        if base_model_id:
            try:
                untuned = make_gliner_predictor(str(base_model_id))
                result["baselines"]["untuned_gliner"] = score_examples(untuned, values)
                if benchmark_path.exists():
                    result["baselines"]["untuned_gliner_dakp_regression"] = score_examples(
                        untuned, load_dakp_benchmark(benchmark_path)
                    )
            except Exception as exc:
                result["baselines"]["untuned_gliner_error"] = str(exc)
        try:
            gazetteer = make_dakp_gazetteer_predictor()
            result["baselines"]["dakp_gazetteer"] = score_examples(gazetteer, values)
            if benchmark_path.exists():
                result["baselines"]["dakp_gazetteer_regression"] = score_examples(
                    gazetteer, load_dakp_benchmark(benchmark_path)
                )
        except Exception as exc:
            result["baselines"]["dakp_gazetteer_error"] = str(exc)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


__all__ = ["Counts", "evaluate_checkpoint", "load_dakp_benchmark", "make_gliner_predictor", "score_examples"]
