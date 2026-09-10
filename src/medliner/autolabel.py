"""9router-only pseudo-labeling for unlabeled candidate texts.

The model returns mention strings, never character offsets. This module maps those strings
with a deterministic exact-match scan and keeps the result auditable as ``model_suggestion``
provenance. It is intentionally separate from the reviewed export: autolabel output is a
weak-supervision pool and must not be mistaken for human gold.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from . import llm
from .candidates import CandidateText
from .gliner_data import ModelLimits, to_gliner_record
from .schema import ALLOWED_LABELS, Annotation, AnnotationStatus, Example, SourceMetadata, canonical_label
from .synthesis import map_mentions

AUTOLABEL_PROMPT = """Identify every disease or phenotype mention in the medical text.
Return ONLY a JSON array, with no markdown or explanation. Each item must be an object with
exactly these fields: {{\"text\": the exact contiguous mention copied from the input,
\"label\": either \"disease\" or \"phenotype\"}}. Return [] when there are no mentions.
Do not return offsets. Do not normalize, expand, or paraphrase mentions.

Text:
{text}"""


class AutolabelError(RuntimeError):
    """Raised when a model reply cannot become a safe canonical example."""


@dataclass(frozen=True)
class AutolabelResult:
    example: Example | None
    reason: str | None = None
    detail: str | None = None


def candidate_id(candidate: CandidateText) -> str:
    digest = hashlib.sha256(f"{candidate.task}\n{candidate.text}".encode()).hexdigest()[:16]
    return f"medliner-autolabel-{digest}"


def _parse_reply(reply: str) -> list[dict[str, str]]:
    try:
        value = json.loads(reply)
    except json.JSONDecodeError as exc:
        raise AutolabelError(f"reply is not JSON: {exc}") from exc
    if not isinstance(value, list):
        raise AutolabelError("reply must be a JSON array")
    mentions: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"text", "label"}:
            raise AutolabelError("each item must contain exactly text and label")
        text, label = item["text"], item["label"]
        if not isinstance(text, str) or not text:
            raise AutolabelError("mention text must be a non-empty string")
        canonical = canonical_label(label) if isinstance(label, str) else None
        if canonical is None:
            raise AutolabelError(f"unsupported label {label!r}; expected one of {ALLOWED_LABELS}")
        mentions.append({"text": text, "label": canonical})
    return mentions


def label_candidate(
    candidate: CandidateText,
    *,
    endpoint: llm.Endpoint,
    limits: ModelLimits | None = None,
    max_tokens: int = 1024,
    timeout: float = 120.0,
) -> AutolabelResult:
    """Turn one candidate into a draft model-suggestion example or a rejection result."""
    try:
        reply = llm.chat(
            [{"role": "user", "content": AUTOLABEL_PROMPT.format(text=candidate.text)}],
            endpoint=endpoint,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        mentions = _parse_reply(reply)
    except (llm.LLMError, AutolabelError) as exc:
        return AutolabelResult(None, reason="invalid_reply", detail=str(exc))

    spans = map_mentions(candidate.text, [item["text"] for item in mentions])
    if spans is None:
        return AutolabelResult(
            None, reason="unmappable_mention", detail="a returned mention was absent or out of order"
        )
    annotations = [
        Annotation(
            start=start,
            end=end,
            label=item["label"],
            text=item["text"],
            status=AnnotationStatus.DRAFT,
            provenance="model_suggestion",
            origin="prediction",
        )
        for item, (start, end) in zip(mentions, spans, strict=True)
    ]
    source = SourceMetadata(
        family="autolabel",
        document_id=candidate.source_document_id,
        record_id=candidate.source_record_id,
        section=candidate.section,
        source_uri=candidate.source_uri,
        source_hash=candidate.source_hash,
        generator=endpoint.name,
    )
    example = Example(
        id=candidate_id(candidate),
        text=candidate.text,
        task=candidate.task,
        source=source,
        annotations=annotations,
        annotation_status=AnnotationStatus.DRAFT,
        metadata={"autolabel_generator": endpoint.name, "autolabel_negative": not annotations},
    )
    try:
        to_gliner_record(example, limits=limits)
    except ValueError as exc:
        return AutolabelResult(None, reason="budget_exceeded", detail=str(exc))
    return AutolabelResult(example)


def result_manifest(results: list[AutolabelResult]) -> dict[str, Any]:
    accepted = [result.example for result in results if result.example is not None]
    counts: dict[str, int] = {}
    for result in results:
        if result.example is not None:
            key = "negative" if not result.example.annotations else "positive"
        else:
            key = result.reason or "rejected"
        counts[key] = counts.get(key, 0) + 1
    return {
        "accepted": len(accepted),
        "positive": sum(bool(example.annotations) for example in accepted),
        "negative": sum(not example.annotations for example in accepted),
        "rejected": len(results) - len(accepted),
        "counts": dict(sorted(counts.items())),
    }


__all__ = [
    "AUTOLABEL_PROMPT",
    "AutolabelError",
    "AutolabelResult",
    "candidate_id",
    "label_candidate",
    "result_manifest",
]
