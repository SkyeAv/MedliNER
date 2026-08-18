"""Candidate task generation for Label Studio imports.

The user authors a small JSONL file of raw candidate texts — typically derived from
intermediate DAKP inputs such as DailyMed section text or FAERS indication strings. This
module validates that file, deduplicates it, and emits the plain-text Label Studio import
shape documented in ``docs/LABEL_STUDIO.md``. Pre-annotations are intentionally not
generated here.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from blake3 import blake3
from pydantic import BaseModel, ValidationError, field_validator

from .schema import ALLOWED_TASKS

GENERATOR_VERSION = "medliner.candidates.v1"


class CandidateInputError(ValueError):
    """Raised when a raw candidates file cannot be converted to import tasks."""


class CandidateText(BaseModel):
    """One raw candidate text row authored from upstream (e.g. DAKP intermediate) inputs."""

    text: str
    task: str
    source_family: str = "unknown"
    source_document_id: str | None = None
    source_record_id: str | None = None
    section: str | None = None
    source_uri: str | None = None
    source_hash: str | None = None

    @field_validator("text")
    @classmethod
    def text_is_nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("candidate text must be non-empty")
        return value

    @field_validator("task")
    @classmethod
    def task_is_known(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in ALLOWED_TASKS:
            raise ValueError(f"unsupported task {value!r}; expected one of {ALLOWED_TASKS}")
        return value


def read_candidates(path: str | Path) -> list[CandidateText]:
    """Read a JSONL file of raw candidate rows, with line-numbered errors."""
    candidates: list[CandidateText] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            candidates.append(CandidateText.model_validate_json(line))
        except ValidationError as exc:
            raise CandidateInputError(f"invalid candidate at line {line_number}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise CandidateInputError(f"invalid JSONL at line {line_number}: {exc}") from exc
    return candidates


def hash_candidates_file(path: str | Path) -> str:
    digest = blake3()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized(text: str) -> str:
    return " ".join(text.split()).lower()


def _task_id(candidate: CandidateText) -> str:
    digest = blake3(f"{candidate.task}\n{_normalized(candidate.text)}".encode()).hexdigest()
    return f"medliner-{digest[:16]}"


def build_import_tasks(
    candidates: list[CandidateText], *, generated_at: datetime | None = None
) -> list[dict[str, Any]]:
    """Convert validated candidates to plain-text Label Studio import tasks.

    Tasks are deduplicated on normalized text + task. The first occurrence wins and records
    how many duplicates were merged. Task ids are deterministic so re-running the generator
    over the same input reproduces the same import file.
    """
    stamp = (generated_at or datetime.now(UTC)).isoformat()
    merged: dict[str, dict[str, Any]] = {}
    duplicates: Counter[str] = Counter()
    for candidate in candidates:
        task_id = _task_id(candidate)
        if task_id in merged:
            duplicates[task_id] += 1
            continue
        data: dict[str, Any] = {
            "text": candidate.text,
            "task": candidate.task,
            "source_family": candidate.source_family,
            "generator_version": GENERATOR_VERSION,
            "generated_at": stamp,
        }
        for field in ("source_document_id", "source_record_id", "section", "source_uri", "source_hash"):
            value = getattr(candidate, field)
            if value is not None:
                data[field] = value
        merged[task_id] = {"id": task_id, "data": data}
    for task_id, count in duplicates.items():
        merged[task_id]["data"]["duplicate_count"] = count + 1
    return list(merged.values())


def write_import_file(tasks: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tasks, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def import_manifest(tasks: list[dict[str, Any]], *, input_path: str | Path) -> dict[str, Any]:
    return {
        "schema_version": "medliner.candidates.manifest.v1",
        "generator_version": GENERATOR_VERSION,
        "input_path": str(input_path),
        "input_hash": hash_candidates_file(input_path),
        "task_count": len(tasks),
        "task_counts": dict(sorted(Counter(task["data"]["task"] for task in tasks).items())),
        "family_counts": dict(sorted(Counter(task["data"]["source_family"] for task in tasks).items())),
        "duplicates_merged": sum(task["data"].get("duplicate_count", 1) - 1 for task in tasks),
    }


__all__ = [
    "GENERATOR_VERSION",
    "CandidateInputError",
    "CandidateText",
    "build_import_tasks",
    "hash_candidates_file",
    "import_manifest",
    "read_candidates",
    "write_import_file",
]
