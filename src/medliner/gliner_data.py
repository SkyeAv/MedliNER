"""Conversion from canonical character spans to GLiNER training records."""

from __future__ import annotations

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
        raise ValueError(f"character span [{start}, {end}) has no complete model tokens")
    first, last = covered[0], covered[-1]
    if tokens[first].start != start or tokens[last].end != end:
        raise ValueError(
            f"character span [{start}, {end}) is not aligned to model tokens; "
            f"covered [{tokens[first].start}, {tokens[last].end})"
        )
    if any(tokens[index].start < start or tokens[index].end > end for index in range(first, last + 1)):
        raise ValueError("character span crosses an unexpected token boundary")
    return first, last


def to_gliner_record(example: Example, model: Any | None = None) -> dict[str, Any]:
    """Convert one canonical example to GLiNER 0.2.x's training record shape."""
    tokens = split_words(example.text, model=model)
    ner: list[tuple[int, int, str]] = []
    for annotation in example.annotations:
        start, end = char_span_to_token_span(example.text, annotation.start, annotation.end, tokens)
        ner.append((start, end, annotation.label))
    record: dict[str, Any] = {
        "id": example.id,
        "tokenized_text": [token.text for token in tokens],
        "ner": ner,
        "text": example.text,
        "task": example.task,
        "source": example.source.model_dump(mode="json"),
        "char_annotations": [annotation.model_dump(mode="json") for annotation in example.annotations],
    }
    max_len = getattr(getattr(model, "config", None), "max_len", None)
    if isinstance(max_len, int) and len(tokens) > max_len:
        raise ValueError(f"example {example.id!r} has {len(tokens)} tokens, exceeding GLiNER max_len={max_len}")
    return record


def to_gliner_dataset(examples: Iterable[Example], model: Any | None = None) -> list[dict[str, Any]]:
    return [to_gliner_record(example, model=model) for example in examples]


__all__ = [
    "GLINER_TOKEN",
    "WordToken",
    "char_span_to_token_span",
    "split_words",
    "to_gliner_dataset",
    "to_gliner_record",
]
