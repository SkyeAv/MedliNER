"""Ingest of the DAKP training-data export bundle (``dakp.medliner.export.v1``).

The bundle is produced by ``dakp export-medliner`` and is self-describing: a manifest with
per-file BLAKE3 hashes, candidate rows in MedliNER's raw-candidate shape, and the NER gold
benchmark. This module verifies the bundle on disk — no DAKP checkout required — enforces
the exact per-family wire contract of ``dakp.medliner.export.v1`` (which is stricter than
:class:`~medliner.candidates.CandidateText`, since Pydantic ignores unknown fields and
provenance is optional for manual candidates), and materializes the payload under
``$MEDLINER_WORKDIR/ingested/`` so the existing ``candidates`` → Label Studio flow consumes
it unchanged. Every failure raises :class:`ExportIngestError` with a file/line-specific
message.
"""

from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .benchmark import load_gold_benchmark
from .candidates import CandidateInputError, hash_candidates_file, read_candidates

EXPORT_SCHEMA_VERSION = "dakp.medliner.export.v1"
CANDIDATE_SCHEMA_VERSION = "medliner.candidates.v1"
BENCHMARK_SCHEMA_VERSION = "dakp.ner.gold.v1"
INGEST_SCHEMA_VERSION = "medliner.ingest.v1"

MANIFEST_FILENAME = "manifest.json"
CANDIDATES_FILENAME = "candidates.ndjson"
GOLD_FILENAME = "ner_gold.json"
PAYLOAD_FILES = (CANDIDATES_FILENAME, GOLD_FILENAME)

# Exact wire keys per source family (dakp.medliner.export.v1). Pydantic ignores unknown
# fields by default, so CandidateText alone cannot enforce the export contract.
_DAILYMED_ROW_KEYS = frozenset({"text", "task", "source_family", "source_document_id", "section", "source_uri"})
_FAERS_ROW_KEYS = frozenset({"text", "task", "source_family", "source_record_id", "source_uri"})
_ROW_KEYS_BY_FAMILY = {"dailymed": _DAILYMED_ROW_KEYS, "faers": _FAERS_ROW_KEYS}


class ExportIngestError(ValueError):
    """Raised for any export-bundle verification or ingest failure (file/line specific)."""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExportIngestError(f"cannot read {path}: {exc}") from exc


def _read_candidates_for_verification(path: Path) -> list[Any]:
    try:
        return read_candidates(path)
    except CandidateInputError as exc:
        raise ExportIngestError(f"{path}: {exc}") from exc
    except (OSError, UnicodeError) as exc:
        raise ExportIngestError(f"cannot read {path}: {exc}") from exc


def _hash_payload(path: Path) -> str:
    try:
        return hash_candidates_file(path)
    except (OSError, UnicodeError) as exc:
        raise ExportIngestError(f"cannot hash {path}: {exc}") from exc


def _validated_count_mapping(
    manifest_path: Path, manifest: dict[str, Any], name: str, keys: tuple[str, ...]
) -> dict[str, int]:
    value = manifest.get(name)
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ExportIngestError(f"{manifest_path}: {name} must contain exactly {keys}")
    counts: dict[str, int] = {}
    for key in keys:
        count = value[key]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ExportIngestError(f"{manifest_path}: {name}.{key} must be a non-negative integer")
        counts[key] = count
    return counts


def _validate_bundle_rows(path: Path) -> None:
    """Enforce the exact per-family row contract of ``dakp.medliner.export.v1``.

    :class:`~medliner.candidates.CandidateText` ignores unknown fields and treats
    provenance as optional (manual candidates may omit it), so the export bundle needs a
    stricter pass: exact key set per source family, every value a non-blank string.
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ExportIngestError(f"cannot read {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ExportIngestError(f"{path}: invalid NDJSON at line {line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ExportIngestError(f"{path}: line {line_number} must be a JSON object")
        family = row.get("source_family")
        if not isinstance(family, str) or family not in _ROW_KEYS_BY_FAMILY:
            raise ExportIngestError(
                f"{path}: line {line_number} source_family {family!r} must be one of {tuple(_ROW_KEYS_BY_FAMILY)}"
            )
        expected = _ROW_KEYS_BY_FAMILY[family]
        if set(row) != expected:
            missing = sorted(expected - set(row))
            unexpected = sorted(set(row) - expected)
            raise ExportIngestError(
                f"{path}: line {line_number} {family} rows must contain exactly {sorted(expected)} "
                f"(missing {missing}, unexpected {unexpected})"
            )
        for key in sorted(expected):
            value = row[key]
            if not isinstance(value, str) or not value.strip():
                raise ExportIngestError(f"{path}: line {line_number} field {key!r} must be a non-blank string")


def _validated_generated_at(manifest_path: Path, value: Any) -> None:
    """Require an ISO-8601 timezone-aware UTC timestamp (the exporter emits ``...+00:00``)."""
    if not isinstance(value, str) or not value.strip():
        raise ExportIngestError(f"{manifest_path}: generated_at must be an ISO-8601 UTC timestamp string")
    try:
        stamp = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ExportIngestError(f"{manifest_path}: generated_at {value!r} is not a valid ISO-8601 timestamp") from exc
    offset = stamp.utcoffset()
    if offset is None:
        raise ExportIngestError(f"{manifest_path}: generated_at {value!r} must be timezone-aware UTC (e.g. ...+00:00)")
    if offset.total_seconds() != 0:
        raise ExportIngestError(f"{manifest_path}: generated_at {value!r} must be UTC, not offset {offset}")


def _validated_inputs(manifest_path: Path, value: Any) -> None:
    """Require the consumed interim tables as a sorted list of non-blank artifact ids.

    The exporter fails loudly when either interim table is missing, so a valid bundle
    always records at least one input; ids may appear bare or ``b3:``-prefixed.
    """
    if not isinstance(value, list) or not value:
        raise ExportIngestError(f"{manifest_path}: inputs must be a non-empty list of artifact ids")
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ExportIngestError(f"{manifest_path}: inputs entries must be non-blank strings")
    if value != sorted(value):
        raise ExportIngestError(f"{manifest_path}: inputs must be sorted in deterministic order")


def _expected_hash_and_count(manifest_path: Path, files: dict[str, Any], name: str, count_key: str) -> tuple[str, int]:
    entry = files.get(name)
    if not isinstance(entry, dict) or set(entry) != {"blake3", count_key}:
        raise ExportIngestError(f"{manifest_path}: files.{name} must contain exactly blake3 and {count_key}")
    recorded = entry["blake3"]
    if not isinstance(recorded, str) or not recorded:
        raise ExportIngestError(f"{manifest_path}: files.{name}.blake3 hash is missing")
    count = entry[count_key]
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ExportIngestError(f"{manifest_path}: files.{name}.{count_key} must be a non-negative integer")
    # The exporter writes ``b3:<hex>`` ids; accept the bare hex form too.
    return recorded.removeprefix("b3:"), count


def verify_bundle(bundle_dir: str | Path) -> dict[str, Any]:
    """Verify the manifest and reconcile every declared payload count with its contents."""
    bundle_dir = Path(bundle_dir)
    manifest_path = bundle_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        raise ExportIngestError(f"bundle {bundle_dir} has no {MANIFEST_FILENAME}")
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ExportIngestError(f"{manifest_path}: manifest must be a JSON object")
    if manifest.get("schema_version") != EXPORT_SCHEMA_VERSION:
        raise ExportIngestError(
            f"{manifest_path}: schema_version {manifest.get('schema_version')!r} does not match expected {EXPORT_SCHEMA_VERSION!r}"
        )
    if manifest.get("candidate_schema") != CANDIDATE_SCHEMA_VERSION:
        raise ExportIngestError(
            f"{manifest_path}: candidate_schema {manifest.get('candidate_schema')!r} does not match expected {CANDIDATE_SCHEMA_VERSION!r}"
        )
    if manifest.get("benchmark_schema") != BENCHMARK_SCHEMA_VERSION:
        raise ExportIngestError(
            f"{manifest_path}: benchmark_schema {manifest.get('benchmark_schema')!r} does not match expected {BENCHMARK_SCHEMA_VERSION!r}"
        )
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(PAYLOAD_FILES):
        raise ExportIngestError(f"{manifest_path}: files must contain exactly {PAYLOAD_FILES}")
    candidate_hash, declared_rows = _expected_hash_and_count(manifest_path, files, CANDIDATES_FILENAME, "rows")
    gold_hash, declared_cases = _expected_hash_and_count(manifest_path, files, GOLD_FILENAME, "cases")
    task_counts = _validated_count_mapping(manifest_path, manifest, "task_counts", ("contraindication", "indication"))
    family_counts = _validated_count_mapping(manifest_path, manifest, "family_counts", ("dailymed", "faers"))
    _validated_generated_at(manifest_path, manifest.get("generated_at"))
    _validated_inputs(manifest_path, manifest.get("inputs"))

    candidate_path = bundle_dir / CANDIDATES_FILENAME
    gold_path = bundle_dir / GOLD_FILENAME
    for name, path in ((CANDIDATES_FILENAME, candidate_path), (GOLD_FILENAME, gold_path)):
        if not path.exists():
            raise ExportIngestError(f"bundle {bundle_dir} is missing payload file {name}")
    actual_candidate_hash = _hash_payload(candidate_path)
    if actual_candidate_hash != candidate_hash:
        raise ExportIngestError(
            f"{candidate_path}: blake3 mismatch (manifest {candidate_hash}, actual {actual_candidate_hash})"
        )
    actual_gold_hash = _hash_payload(gold_path)
    if actual_gold_hash != gold_hash:
        raise ExportIngestError(f"{gold_path}: blake3 mismatch (manifest {gold_hash}, actual {actual_gold_hash})")

    _validate_bundle_rows(candidate_path)
    candidates = _read_candidates_for_verification(candidate_path)
    actual_task_counts = Counter(candidate.task for candidate in candidates)
    actual_family_counts = Counter(candidate.source_family for candidate in candidates)
    actual_tasks = {key: actual_task_counts.get(key, 0) for key in task_counts}
    actual_families = {key: actual_family_counts.get(key, 0) for key in family_counts}
    if len(candidates) != declared_rows:
        raise ExportIngestError(
            f"{candidate_path}: row count mismatch (manifest {declared_rows}, actual {len(candidates)})"
        )
    if actual_tasks != task_counts:
        raise ExportIngestError(
            f"{candidate_path}: task_counts mismatch (manifest {task_counts}, actual {actual_tasks})"
        )
    if actual_families != family_counts:
        raise ExportIngestError(
            f"{candidate_path}: family_counts mismatch (manifest {family_counts}, actual {actual_families})"
        )
    if any(candidate.source_family not in family_counts for candidate in candidates):
        raise ExportIngestError(f"{candidate_path}: source_family must be one of {tuple(family_counts)}")

    gold = _read_json(gold_path)
    if not isinstance(gold, dict):
        raise ExportIngestError(f"{gold_path}: gold benchmark must be a JSON object")
    if gold.get("schema_version") != BENCHMARK_SCHEMA_VERSION:
        raise ExportIngestError(
            f"{gold_path}: schema_version {gold.get('schema_version')!r} does not match expected {BENCHMARK_SCHEMA_VERSION!r}"
        )
    if not isinstance(gold.get("annotation_policy"), str) or not gold["annotation_policy"].strip():
        raise ExportIngestError(f"{gold_path}: required annotation_policy is missing")
    cases = gold.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ExportIngestError(f"{gold_path}: 'cases' must be a non-empty list")
    if len(cases) != declared_cases:
        raise ExportIngestError(f"{gold_path}: case count mismatch (manifest {declared_cases}, actual {len(cases)})")
    return manifest


def _default_workdir() -> Path:
    """Same resolution as ``cli.workdir()``; kept local so ``cli`` can import this module."""
    return Path(os.environ.get("MEDLINER_WORKDIR", "data/materialized"))


def ingest_export(bundle_dir: str | Path, *, workdir: str | Path | None = None) -> dict[str, Any]:
    """Verify the bundle, validate it with MedliNER's own rules, materialize it under ``workdir``.

    Writes ``<workdir>/ingested/{candidates.ndjson, ner_gold.json, ingest-manifest.json}``
    (byte copies of the verified payload) and returns the paths plus the manifest counts.
    """
    bundle_dir = Path(bundle_dir)
    manifest = verify_bundle(bundle_dir)
    candidates_path = bundle_dir / CANDIDATES_FILENAME
    gold_path = bundle_dir / GOLD_FILENAME
    try:
        candidates = read_candidates(candidates_path)
    except CandidateInputError as exc:
        raise ExportIngestError(f"{candidates_path}: {exc}") from exc
    except (OSError, UnicodeError) as exc:
        raise ExportIngestError(f"cannot read {candidates_path}: {exc}") from exc
    try:
        benchmark = load_gold_benchmark(gold_path)
    except (OSError, UnicodeError, KeyError, TypeError, ValueError, OverflowError) as exc:
        # OverflowError: e.g. a mention offset of 1e999 parses to inf and overflows int().
        raise ExportIngestError(f"{gold_path}: malformed gold case: {exc}") from exc

    output_dir = Path(workdir) / "ingested" if workdir is not None else _default_workdir() / "ingested"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ExportIngestError(f"cannot create {output_dir}: {exc}") from exc
    materialized_candidates = output_dir / CANDIDATES_FILENAME
    materialized_gold = output_dir / GOLD_FILENAME
    try:
        shutil.copyfile(candidates_path, materialized_candidates)
    except OSError as exc:
        raise ExportIngestError(f"cannot write {materialized_candidates} from {candidates_path}: {exc}") from exc
    try:
        shutil.copyfile(gold_path, materialized_gold)
    except OSError as exc:
        raise ExportIngestError(f"cannot write {materialized_gold} from {gold_path}: {exc}") from exc
    ingest_manifest = {
        "schema_version": INGEST_SCHEMA_VERSION,
        "bundle_path": str(bundle_dir),
        "export_schema_version": manifest["schema_version"],
        "candidates_blake3": _hash_payload(materialized_candidates),
        "gold_blake3": _hash_payload(materialized_gold),
        "task_counts": manifest["task_counts"],
        "family_counts": manifest["family_counts"],
        "ingested_at": datetime.now(UTC).isoformat(),
    }
    materialized_manifest = output_dir / "ingest-manifest.json"
    try:
        materialized_manifest.write_text(json.dumps(ingest_manifest, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        raise ExportIngestError(f"cannot write {materialized_manifest}: {exc}") from exc
    return {
        "candidates_path": materialized_candidates,
        "gold_path": materialized_gold,
        "manifest_path": materialized_manifest,
        "candidate_rows": len(candidates),
        "gold_cases": len(benchmark),
        "task_counts": ingest_manifest["task_counts"],
        "family_counts": ingest_manifest["family_counts"],
    }


__all__ = [
    "BENCHMARK_SCHEMA_VERSION",
    "CANDIDATES_FILENAME",
    "EXPORT_SCHEMA_VERSION",
    "GOLD_FILENAME",
    "INGEST_SCHEMA_VERSION",
    "MANIFEST_FILENAME",
    "PAYLOAD_FILES",
    "ExportIngestError",
    "ingest_export",
    "verify_bundle",
]
