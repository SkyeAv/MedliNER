"""Label Studio export adapter.

Label Studio records character offsets; GLiNER training uses model-token indexes. This module
only accepts completed annotation exports and converts them into MedliNER's canonical schema.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .schema import (
    ALLOWED_LABELS,
    ALLOWED_TASKS,
    SPAN_ORIGINS,
    Annotation,
    AnnotationStatus,
    Example,
    SourceMetadata,
    canonical_label,
)


class LabelStudioExportError(ValueError):
    """Raised when an export cannot be safely converted to training data."""


def read_tasks(path: str | Path) -> list[dict[str, Any]]:
    """Read Label Studio JSON array or JSONL export."""
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        return []
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        tasks: list[dict[str, Any]] = []
        for line_number, line in enumerate(raw.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LabelStudioExportError(f"invalid JSONL at line {line_number}: {exc}") from exc
            if not isinstance(item, dict):
                raise LabelStudioExportError(f"JSONL line {line_number} is not an object") from None
            tasks.append(item)
        return tasks
    if isinstance(value, list):
        return [require_object(item, "export task") for item in value]
    if isinstance(value, dict):
        return [value]
    raise LabelStudioExportError("Label Studio export must be a JSON object, array, or JSONL")


def require_object(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise LabelStudioExportError(f"{context} must be an object")
    return value


def _data(task: dict[str, Any]) -> dict[str, Any]:
    return require_object(task.get("data", task), "task data")


def _text(task: dict[str, Any]) -> str:
    data = _data(task)
    text = data.get("text")
    if not isinstance(text, str):
        # A common alternate export uses the configured Text name as the key.
        text = data.get("raw_text")
    if not isinstance(text, str):
        raise LabelStudioExportError(f"task {task.get('id', '<unknown>')!r} has no string data.text")
    return text


def _task_kind(task: dict[str, Any]) -> str:
    data = _data(task)
    value = data.get("task") or data.get("task_kind") or data.get("context")
    if not isinstance(value, str) or value.strip().lower() not in ALLOWED_TASKS:
        raise LabelStudioExportError(
            f"task {task.get('id', '<unknown>')!r} has invalid task metadata {value!r}; expected {ALLOWED_TASKS}"
        )
    return value.strip().lower()


def _source(task: dict[str, Any]) -> SourceMetadata:
    data = _data(task)
    source_value = data.get("source")
    source = source_value if isinstance(source_value, dict) else {}
    family = (
        source.get("family")
        or data.get("source_family")
        or (source_value if isinstance(source_value, str) else None)
        or "unknown"
    )
    document_id = source.get("document_id") or data.get("source_document_id") or data.get("document_id")
    record_id = source.get("record_id") or data.get("source_record_id") or data.get("record_id")
    section = source.get("section") or data.get("section") or data.get("section_code")
    source_uri = source.get("source_uri") or data.get("source_uri")
    source_hash = source.get("source_hash") or data.get("source_hash")
    return SourceMetadata(
        family=str(family),
        document_id=str(document_id) if document_id is not None else None,
        record_id=str(record_id) if record_id is not None else None,
        section=str(section) if section is not None else None,
        source_uri=str(source_uri) if source_uri is not None else None,
        source_hash=str(source_hash) if source_hash is not None else None,
    )


def _annotation_sets(task: dict[str, Any]) -> list[dict[str, Any]]:
    sets = task.get("annotations", [])
    if sets is None:
        return []
    if not isinstance(sets, list) or any(not isinstance(item, dict) for item in sets):
        raise LabelStudioExportError(f"task {task.get('id', '<unknown>')!r} has malformed annotations")
    live = [item for item in sets if not item.get("was_cancelled", False)]
    if sets and not live:
        # Every annotation was skipped/cancelled in the UI. Treating that as a reviewed
        # no-entity example would teach the model that this text contains nothing.
        raise LabelStudioExportError(
            f"task {task.get('id', '<unknown>')!r} has only cancelled/skipped annotations; "
            "resolve or drop it before training"
        )
    return live


def _select_annotation_set(task: dict[str, Any]) -> dict[str, Any] | None:
    sets = _annotation_sets(task)
    if not sets:
        return None
    if len(sets) == 1:
        return sets[0]
    data = _data(task)
    selected_id = data.get("final_annotation_id") or task.get("final_annotation_id")
    if selected_id is not None:
        for item in sets:
            if str(item.get("id")) == str(selected_id):
                return item
    status = str(data.get("annotation_status", "")).lower()
    adjudicated = [item for item in sets if item.get("ground_truth") is True or item.get("is_adjudication") is True]
    if status == AnnotationStatus.ADJUDICATED.value and len(adjudicated) == 1:
        return adjudicated[0]
    raise LabelStudioExportError(
        f"task {task.get('id', '<unknown>')!r} has {len(sets)} annotation sets; "
        "provide final_annotation_id or adjudication metadata"
    )


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Drop whitespace a click-drag selection picked up at either edge of the phrase.

    Highlighting by dragging routinely captures the trailing space after a word. Left in place
    it becomes a token-alignment failure much later, during GLiNER conversion, so the offsets
    are tightened here where the source text is still available to re-verify them.
    """
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _result_annotations(task: dict[str, Any], annotation_set: dict[str, Any] | None) -> list[Annotation]:
    if annotation_set is None:
        return []
    results = annotation_set.get("result", [])
    if not isinstance(results, list):
        raise LabelStudioExportError(f"task {task.get('id', '<unknown>')!r} annotation result is not a list")
    text = _text(task)
    annotator = annotation_set.get("created_username") or annotation_set.get("completed_by")
    status = AnnotationStatus.ADJUDICATED if annotation_set.get("ground_truth") else AnnotationStatus.REVIEWED
    parsed: dict[tuple[int, int], Annotation] = {}
    for result in results:
        if not isinstance(result, dict) or result.get("type") != "labels":
            continue
        value = require_object(result.get("value"), "Label Studio label result value")
        try:
            start = int(value["start"])
            end = int(value["end"])
            surface = value["text"]
            labels = value["labels"]
        except (KeyError, TypeError, ValueError) as exc:
            raise LabelStudioExportError(f"task {task.get('id', '<unknown>')} has malformed span result") from exc
        if (
            not isinstance(surface, str)
            or not isinstance(labels, list)
            or len(labels) != 1
            or not isinstance(labels[0], str)
        ):
            raise LabelStudioExportError(
                f"task {task.get('id', '<unknown>')} span must contain exactly one string label"
            )
        label = canonical_label(labels[0]) or labels[0].strip()
        if label not in ALLOWED_LABELS:
            raise LabelStudioExportError(f"unsupported Label Studio label {label!r}; expected {ALLOWED_LABELS}")
        if start < 0 or end <= start or end > len(text):
            raise LabelStudioExportError(
                f"invalid character span [{start}, {end}) for task {task.get('id', '<unknown>')!r}"
            )
        if text[start:end] != surface:
            raise LabelStudioExportError(
                f"Label Studio span text mismatch at [{start}, {end}): export={surface!r}, source={text[start:end]!r}"
            )
        start, end = _trim_span(text, start, end)
        if start >= end:
            raise LabelStudioExportError(
                f"task {task.get('id', '<unknown>')!r} has a whitespace-only span labelled {label!r}"
            )
        annotation = _build_annotation(
            task,
            id=str(result.get("id")) if result.get("id") is not None else None,
            start=start,
            end=end,
            label=label,
            text=text[start:end],
            annotator=str(annotator) if annotator is not None else None,
            status=status,
            origin=_span_origin(task, result),
        )
        key = (start, end)
        # With one canonical label, same-offset spans are always exact duplicates; they are
        # harmless export duplication and are collapsed.
        parsed[key] = annotation
    return sorted(parsed.values(), key=lambda item: (item.start, item.end, item.label))


def _span_origin(task: dict[str, Any], result: dict[str, Any]) -> str | None:
    """Where a submitted region came from, when Label Studio recorded it.

    Pre-labeled projects copy the model's regions into the annotator's draft, and Label Studio
    marks each one ``prediction`` (submitted untouched), ``prediction-changed`` (corrected), or
    ``manual`` (drawn by hand). Older exports and non-pre-labeled projects omit the field, which
    stays ``None`` rather than being guessed at. An unrecognized value is an export problem, not
    something to silently normalize away.
    """
    origin = result.get("origin")
    if origin is None:
        return None
    if not isinstance(origin, str) or origin not in SPAN_ORIGINS:
        raise LabelStudioExportError(
            f"task {task.get('id', '<unknown>')!r} has unsupported span origin {origin!r}; expected {SPAN_ORIGINS}"
        )
    return origin


def _build_annotation(task: dict[str, Any], *, status: AnnotationStatus, **fields: Any) -> Annotation:
    try:
        return Annotation(
            status=status, provenance="adjudicated" if status == AnnotationStatus.ADJUDICATED else "human", **fields
        )
    except ValidationError as exc:
        raise LabelStudioExportError(f"task {task.get('id', '<unknown>')!r} has an invalid span: {exc}") from exc


def normalize_task(task: dict[str, Any], *, export_id: str | None = None) -> Example:
    """Convert one Label Studio task to the canonical reviewed-example contract."""
    task_id = task.get("id")
    if task_id is None:
        raise LabelStudioExportError("task has no id")
    annotation_set = _select_annotation_set(task)
    data = _data(task)
    status_value = str(data.get("annotation_status", "reviewed")).lower()
    if status_value not in {item.value for item in AnnotationStatus}:
        raise LabelStudioExportError(f"invalid annotation_status {status_value!r}")
    if annotation_set is not None and annotation_set.get("ground_truth") is True:
        status_value = AnnotationStatus.ADJUDICATED.value
    try:
        return Example(
            id=str(task_id),
            text=_text(task),
            task=_task_kind(task),
            source=_source(task),
            annotations=_result_annotations(task, annotation_set),
            annotation_status=status_value,
            annotation_set_id=str(annotation_set.get("id"))
            if annotation_set and annotation_set.get("id") is not None
            else None,
            export_id=export_id,
            metadata={
                "label_studio_task_id": task_id,
                "annotation_count": len(annotation_set.get("result", [])) if annotation_set else 0,
            },
        )
    except ValidationError as exc:
        # Overlap, offset, and policy violations are export problems, not internal errors.
        raise LabelStudioExportError(f"task {task_id!r} is not a valid MedliNER example: {exc}") from exc


def normalize_export(path: str | Path, *, export_id: str | None = None, require_reviewed: bool = True) -> list[Example]:
    """Normalize an export; only reviewed/adjudicated examples are training candidates."""
    examples = [normalize_task(task, export_id=export_id or Path(path).name) for task in read_tasks(path)]
    duplicates = sorted(item for item, count in Counter(item.id for item in examples).items() if count > 1)
    if duplicates:
        raise LabelStudioExportError(f"export contains duplicate task ids: {duplicates[:5]}")
    if require_reviewed:
        unreviewed = [
            item.id
            for item in examples
            if item.annotation_status not in {AnnotationStatus.REVIEWED, AnnotationStatus.ADJUDICATED}
        ]
        if unreviewed:
            raise LabelStudioExportError(f"unreviewed tasks cannot enter training: {unreviewed[:5]}")
    return examples


__all__ = ["LabelStudioExportError", "normalize_export", "normalize_task", "read_tasks"]
