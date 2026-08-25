"""Conversion from canonical character spans to GLiNER training records."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from .schema import Example

# Matches GLiNER's documented WhitespaceTokenSplitter in 0.2.x. A real model splitter is preferred.
GLINER_TOKEN = re.compile(r"\w+(?:[-_]\w+)*|\S")


@dataclass(frozen=True)
class WordToken:
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class ModelLimits:
    """The two GLiNER budgets that silently discard supervision when exceeded.

    ``max_len`` truncates the token sequence (``UserWarning`` only), and ``max_width`` bounds
    the enumerated span candidates, so a gold span wider than ``max_width`` is never assigned a
    label at all. MedliNER refuses to build such a record instead of training on silent holes.
    """

    max_len: int | None = None
    max_width: int | None = None


def model_limits(model: Any | None) -> ModelLimits:
    config = getattr(model, "config", None)
    max_len = getattr(config, "max_len", None)
    max_width = getattr(config, "max_width", None)
    return ModelLimits(
        max_len=max_len if isinstance(max_len, int) else None,
        max_width=max_width if isinstance(max_width, int) else None,
    )


def split_words(text: str, model: Any | None = None) -> list[WordToken]:
    """Return model-compatible word tokens and character offsets."""
    if model is not None:
        splitter = getattr(getattr(model, "data_processor", None), "words_splitter", None)
        if splitter is not None:
            return [WordToken(str(token), int(start), int(end)) for token, start, end in splitter(text)]
    return [WordToken(match.group(), match.start(), match.end()) for match in GLINER_TOKEN.finditer(text)]


def char_span_to_token_span(text: str, start: int, end: int, tokens: Sequence[WordToken]) -> tuple[int, int]:
    """Map a Label Studio half-open character span to GLiNER's inclusive token indexes."""
    if start < 0 or end <= start or end > len(text):
        raise ValueError(f"invalid character span [{start}, {end})")
    covered = [index for index, token in enumerate(tokens) if token.start >= start and token.end <= end]
    if not covered:
        raise ValueError(f"character span [{start}, {end}) ({text[start:end]!r}) has no complete model tokens")
    first, last = covered[0], covered[-1]
    if tokens[first].start != start or tokens[last].end != end:
        raise ValueError(
            f"character span [{start}, {end}) ({text[start:end]!r}) is not aligned to model tokens; "
            f"the nearest whole-token span is [{tokens[first].start}, {tokens[last].end}) "
            f"({text[tokens[first].start : tokens[last].end]!r})"
        )
    if any(tokens[index].start < start or tokens[index].end > end for index in range(first, last + 1)):
        raise ValueError("character span crosses an unexpected token boundary")
    return first, last


def to_gliner_record(
    example: Example, model: Any | None = None, *, limits: ModelLimits | None = None, weight: float = 1.0
) -> dict[str, Any]:
    """Convert one canonical example to GLiNER 0.2.x's training record shape.

    ``weight`` is a per-record loss multiplier for semi-supervised mixes (e.g. down-weighting
    synthetic examples). It is omitted from the record at its default so the existing record
    shape — and every consumer of it — stays byte-for-byte stable.
    """
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError(f"weight must be a positive finite number, got {weight!r}")
    limits = limits if limits is not None else model_limits(model)
    tokens = split_words(example.text, model=model)
    if limits.max_len is not None and len(tokens) > limits.max_len:
        raise ValueError(
            f"example {example.id!r} has {len(tokens)} word tokens, exceeding GLiNER max_len={limits.max_len}; "
            "GLiNER would silently truncate it. Split the text upstream or raise max_length."
        )
    ner: list[tuple[int, int, str]] = []
    for annotation in example.annotations:
        start, end = char_span_to_token_span(example.text, annotation.start, annotation.end, tokens)
        width = end - start + 1
        if limits.max_width is not None and width > limits.max_width:
            raise ValueError(
                f"example {example.id!r} annotation {annotation.text!r} spans {width} word tokens, "
                f"exceeding GLiNER max_width={limits.max_width}; the span would never be enumerated "
                "as a candidate and its supervision would be silently dropped."
            )
        ner.append((start, end, annotation.label))
    return {
        "id": example.id,
        "tokenized_text": [token.text for token in tokens],
        "ner": ner,
        "text": example.text,
        "task": example.task,
        "source": example.source.model_dump(mode="json"),
        "char_annotations": [annotation.model_dump(mode="json") for annotation in example.annotations],
        **({"weight": weight} if weight != 1.0 else {}),
    }


def to_gliner_dataset(
    examples: Iterable[Example], model: Any | None = None, *, weight: float = 1.0
) -> list[dict[str, Any]]:
    limits = model_limits(model)
    return [to_gliner_record(example, model=model, limits=limits, weight=weight) for example in examples]


__all__ = [
    "GLINER_TOKEN",
    "ModelLimits",
    "WordToken",
    "char_span_to_token_span",
    "model_limits",
    "split_words",
    "to_gliner_dataset",
    "to_gliner_record",
]
