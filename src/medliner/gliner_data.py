"""Conversion from canonical character spans to GLiNER training records."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from .schema import Annotation, Example

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


def sliding_windows(
    example: Example,
    *,
    max_len: int,
    max_width: int | None = None,
    model: Any | None = None,
) -> list[Example]:
    """Tile a long example into overlapping exact-substring model windows.

    This mirrors DAKP's production NER windowing: every model input stays within GLiNER's word
    budget, while overlap is wide enough that every annotation (including one crossing a hard
    boundary) is wholly present in at least one window. Character offsets are remapped into each
    window and retained in the canonical annotation payload; no text or span is truncated.
    """
    if max_len < 1:
        raise ValueError(f"max_len must be positive, got {max_len!r}")
    tokens = split_words(example.text, model=model)
    if len(tokens) <= max_len:
        return [example]
    overlap = min(max_width or 0, max_len - 1)
    stride = max_len - overlap
    starts = list(range(0, len(tokens) - max_len + 1, stride))
    final_start = len(tokens) - max_len
    if not starts or starts[-1] != final_start:
        starts.append(final_start)
    windows: list[Example] = []
    seen_annotations: set[str] = set()
    for index, token_start in enumerate(starts):
        token_end = min(token_start + max_len, len(tokens))
        char_start = 0 if token_start == 0 else tokens[token_start].start
        char_end = len(example.text) if token_end == len(tokens) else tokens[token_end - 1].end
        text = example.text[char_start:char_end]
        annotations: list[Annotation] = []
        for annotation in example.annotations:
            if annotation.start < char_start or annotation.end > char_end:
                continue
            key = annotation.id or f"{annotation.start}:{annotation.end}:{annotation.label}"
            seen_annotations.add(key)
            local_start = annotation.start - char_start
            local_end = annotation.end - char_start
            annotations.append(
                annotation.model_copy(
                    update={
                        "start": local_start,
                        "end": local_end,
                        "text": text[local_start:local_end],
                    }
                )
            )
        payload = example.model_dump(mode="python")
        payload.update(
            id=f"{example.id}::window-{index:04d}",
            text=text,
            annotations=annotations,
            metadata={
                **example.metadata,
                "window_index": index,
                "window_count": len(starts),
                "window_char_start": char_start,
                "window_char_end": char_end,
            },
        )
        windows.append(Example.model_validate(payload))
    expected = {
        annotation.id or f"{annotation.start}:{annotation.end}:{annotation.label}" for annotation in example.annotations
    }
    missing = expected - seen_annotations
    if missing:
        raise ValueError(
            f"long example {example.id!r} has annotation(s) not contained in any sliding window: {sorted(missing)[:3]}"
        )
    return windows


def sliding_window_examples(
    examples: Iterable[Example], model: Any, *, max_len: int | None = None, max_width: int | None = None
) -> list[Example]:
    """Window every example using the loaded model's budgets, preserving short examples as-is."""
    limits = model_limits(model)
    budget = max_len if max_len is not None else limits.max_len
    if budget is None:
        return list(examples)
    width = max_width if max_width is not None else limits.max_width
    return [
        window
        for example in examples
        for window in sliding_windows(example, max_len=budget, max_width=width, model=model)
    ]


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
    "sliding_window_examples",
    "sliding_windows",
    "to_gliner_dataset",
    "to_gliner_record",
]
