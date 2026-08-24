"""Candidate task generation for Label Studio imports.

The user authors a small NDJSON file of raw candidate texts — typically derived from
intermediate DAKP inputs such as DailyMed section text or FAERS indication strings. This
module validates that file, deduplicates it, and emits the plain-text Label Studio import
shape documented in ``docs/LABEL_STUDIO.md``. Pre-annotations are intentionally not
generated here.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from blake3 import blake3
from pydantic import BaseModel, ValidationError, field_validator

from .schema import ALLOWED_TASKS

GENERATOR_VERSION = "medliner.candidates.v1"
WARMUP_SOURCE_FAMILY = "gold-warmup"


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
    """Read an NDJSON file of raw candidate rows, with line-numbered errors."""
    candidates: list[CandidateText] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            candidates.append(CandidateText.model_validate_json(line))
        except ValidationError as exc:
            raise CandidateInputError(f"invalid candidate at line {line_number}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise CandidateInputError(f"invalid NDJSON at line {line_number}: {exc}") from exc
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


def _word_count(text: str) -> int:
    return len(text.split())


def _sampling_rank(seed: int, task_id: str) -> str:
    """Deterministic selection rank for a task id under a sampling seed."""
    return blake3(f"{seed}:{task_id}".encode()).hexdigest()


def _largest_remainder(counts: dict[str, int], total: int) -> dict[str, int]:
    """Split ``total`` across keys proportionally to ``counts`` (largest-remainder rounding)."""
    if total <= 0:
        return {key: 0 for key in counts}
    pool = sum(counts.values())
    if total >= pool:
        return dict(counts)
    quotas = {key: total * value / pool for key, value in counts.items()}
    allocation = {key: int(quota) for key, quota in quotas.items()}
    by_remainder = sorted(counts, key=lambda key: (-(quotas[key] - allocation[key]), key))
    for key in by_remainder[: total - sum(allocation.values())]:
        allocation[key] += 1
    return allocation


def sample_tasks(
    tasks: list[dict[str, Any]],
    targets: dict[str, int],
    *,
    seed: int = 2026,
    max_words: int | None = None,
) -> list[dict[str, Any]]:
    """Deterministically sample a family-stratified subset of deduplicated import tasks.

    ``targets`` caps how many tasks to keep per task name; task names absent from it are
    dropped, and an empty mapping disables sampling (the input is returned unchanged). Within
    a task, the selection is stratified across ``source_family`` in proportion to each
    family's share of the eligible pool, and members are chosen by ``blake3(seed:task_id)``
    rank so the same input plus configuration always reproduce the same subset. ``max_words``
    drops texts longer than that many whitespace-separated words: GLiNER conversion refuses
    such texts at training time (``max_length`` in ``configs/train-small.yaml``), so
    annotating longer passages is wasted effort.
    """
    for name, target in targets.items():
        if name not in ALLOWED_TASKS:
            raise ValueError(f"unknown sampling task {name!r}; expected one of {sorted(ALLOWED_TASKS)}")
        if int(target) < 0:
            raise ValueError(f"sampling target for {name!r} must be non-negative, got {target}")
    if not targets:
        return list(tasks)
    selected: list[dict[str, Any]] = []
    for name in sorted(targets):
        limit = int(targets[name])
        if limit <= 0:
            continue
        pool = [
            task
            for task in tasks
            if task["data"]["task"] == name and (max_words is None or _word_count(task["data"]["text"]) <= max_words)
        ]
        if not pool:
            continue
        by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for task in pool:
            by_family[task["data"]["source_family"]].append(task)
        allocation = _largest_remainder(
            {family: len(members) for family, members in sorted(by_family.items())}, min(limit, len(pool))
        )
        for family in sorted(allocation):
            members = sorted(by_family[family], key=lambda task: _sampling_rank(seed, task["id"]))
            selected.extend(members[: allocation[family]])
    return selected


def stagger_tasks(tasks: list[dict[str, Any]], *, max_run: int = 3, seed: int = 2026) -> list[dict[str, Any]]:
    """Reorder tasks so labelers working top-to-bottom see interleaved task types.

    Tasks are consumed from per-``(task, source_family)`` strata, each in deterministic
    ``blake3(seed:task_id)`` order. Both levels — the task value, then the family within it —
    pick the eligible stratum with the largest remaining fraction (ties continue the current
    run), so task types and source families drain at the same relative pace. While more than
    one task type remains, no more than ``max_run`` consecutive tasks share a task value. When
    the ratio between task types exceeds ``max_run`` the cap cannot hold to the end, so a tail
    of the majority task type is unavoidable once the others are exhausted.
    """
    if max_run < 1:
        raise ValueError("max_run must be at least 1")
    if len(tasks) <= 1:
        return list(tasks)
    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        strata[(task["data"]["task"], task["data"]["source_family"])].append(task)
    for members in strata.values():
        members.sort(key=lambda task: _sampling_rank(seed, task["id"]))
    totals = {key: len(members) for key, members in strata.items()}
    remaining = dict(totals)
    value_totals: dict[str, int] = defaultdict(int)
    for (task_value, _family), count in totals.items():
        value_totals[task_value] += count
    value_remaining = dict(value_totals)
    ordered: list[dict[str, Any]] = []
    run_task: str | None = None
    run_length = 0
    run_family: str | None = None
    while any(value_remaining.values()):
        values = [value for value in sorted(value_totals) if value_remaining[value]]
        eligible = [value for value in values if value != run_task or run_length < max_run] or values
        # Continue the current run on fraction ties so minority switches are not wasted early.
        value = min(eligible, key=lambda item: (-value_remaining[item] / value_totals[item], item != run_task, item))
        families = sorted(
            family for (task_value, family) in strata if task_value == value and remaining[(value, family)]
        )
        chosen = (
            value,
            min(
                families,
                key=lambda family: (
                    -remaining[(value, family)] / totals[(value, family)],
                    family != run_family,
                    family,
                ),
            ),
        )
        ordered.append(strata[chosen].pop(0))
        remaining[chosen] -= 1
        value_remaining[value] -= 1
        if value == run_task:
            run_length += 1
        else:
            run_task, run_length = value, 1
        run_family = chosen[1]
    return ordered


def import_file_name(*, input_hash: str, sampling: str | None = None) -> str:
    """Basename for an import file; the unsampled legacy name depends only on the input hash."""
    if not sampling:
        return f"import-{input_hash[:16]}.json"
    digest = blake3(f"{input_hash}\n{sampling}".encode()).hexdigest()
    return f"import-{digest[:16]}.json"


def import_manifest(
    tasks: list[dict[str, Any]], *, input_path: str | Path, sampling: dict[str, Any] | None = None
) -> dict[str, Any]:
    manifest = {
        "schema_version": "medliner.candidates.manifest.v1",
        "generator_version": GENERATOR_VERSION,
        "input_path": str(input_path),
        "input_hash": hash_candidates_file(input_path),
        "task_count": len(tasks),
        "task_counts": dict(sorted(Counter(task["data"]["task"] for task in tasks).items())),
        "family_counts": dict(sorted(Counter(task["data"]["source_family"] for task in tasks).items())),
        "duplicates_merged": sum(task["data"].get("duplicate_count", 1) - 1 for task in tasks),
    }
    if sampling is not None:
        manifest["sampling"] = sampling
    return manifest


def build_warmup_tasks(gold_path: str | Path, *, limit: int = 10) -> list[dict[str, Any]]:
    """Build demo tasks from the ingested gold benchmark for a separate warm-up project.

    Warm-up tasks never enter the main candidate queue, so gold cases cannot leak into
    training data; they give present-day annotators instant feedback against known answers.
    Each task carries its resolved ``gold_mentions`` in ``data`` so a presenter can compare
    an annotation against truth in the Data Manager. Task mapping mirrors the benchmark
    loader: DailyMed-sourced cases are contraindications, everything else indications.
    Raises :class:`CandidateInputError` on a malformed or empty benchmark.
    """
    try:
        payload = json.loads(Path(gold_path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CandidateInputError(f"cannot read gold benchmark {gold_path}: {exc}") from exc
    cases = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(cases, list) or not cases:
        raise CandidateInputError(f"{gold_path}: 'cases' must be a non-empty list")
    if limit < 1:
        raise ValueError("warm-up limit must be at least 1")
    stamp = datetime.now(UTC).isoformat()
    tasks: list[dict[str, Any]] = []
    for case in cases:
        if len(tasks) >= limit:
            break
        case_id = str(case.get("id") or "").strip()
        text = str(case.get("text") or "").strip()
        mentions = case.get("mentions")
        if not case_id or not text or not isinstance(mentions, list):
            raise CandidateInputError(f"{gold_path}: each case needs non-blank id/text and a mentions list")
        resolved: list[dict[str, Any]] = []
        for mention in mentions:
            surface = str(mention.get("surface") or "").strip()
            if not surface:
                raise CandidateInputError(f"{gold_path}: case {case_id!r} has a blank mention surface")
            start = mention.get("start")
            try:
                start = text.index(surface) if start is None else int(start)
            except ValueError as exc:
                raise CandidateInputError(
                    f"{gold_path}: case {case_id!r} surface {surface!r} is not present in its text"
                ) from exc
            if text[start : start + len(surface)] != surface:
                raise CandidateInputError(f"{gold_path}: case {case_id!r} offset {start} does not contain {surface!r}")
            resolved.append(
                {"start": start, "end": start + len(surface), "label": str(mention.get("type") or ""), "text": surface}
            )
        source = str(case.get("source") or "unknown")
        digest = blake3(f"warmup\n{case_id}".encode()).hexdigest()
        tasks.append(
            {
                "id": f"warmup-{digest[:16]}",
                "data": {
                    "text": text,
                    "task": "contraindication" if source == "dailymed" else "indication",
                    "source_family": WARMUP_SOURCE_FAMILY,
                    "source_document_id": case_id,
                    "generator_version": GENERATOR_VERSION,
                    "generated_at": stamp,
                    "warmup": True,
                    "gold_mentions": resolved,
                },
            }
        )
    return tasks


__all__ = [
    "GENERATOR_VERSION",
    "WARMUP_SOURCE_FAMILY",
    "CandidateInputError",
    "CandidateText",
    "build_import_tasks",
    "build_warmup_tasks",
    "hash_candidates_file",
    "import_file_name",
    "import_manifest",
    "read_candidates",
    "sample_tasks",
    "stagger_tasks",
    "write_import_file",
]
