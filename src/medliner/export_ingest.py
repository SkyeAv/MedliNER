"""Ingest of the DAKP training-data export bundle (``dakp.medliner.export.v1``).

The bundle is produced by ``dakp export-medliner`` and is self-describing: a manifest with
per-file BLAKE3 hashes, candidate rows in MEDliNER's raw-candidate shape, and the NER gold
benchmark. This module verifies the bundle on disk — no DAKP checkout required — validates
the rows through MEDliNER's own :class:`~medliner.candidates.CandidateText` rules, and
materializes the payload under ``$MEDLINER_WORKDIR/ingested/`` so the existing
``candidates`` → Label Studio flow consumes it unchanged. Every failure raises
:class:`ExportIngestError` with a file/line-specific message.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .candidates import CandidateInputError, hash_candidates_file, read_candidates
from .evaluation import load_dakp_benchmark

EXPORT_SCHEMA_VERSION = "dakp.medliner.export.v1"
BENCHMARK_SCHEMA_VERSION = "dakp.ner.gold.v1"
INGEST_SCHEMA_VERSION = "medliner.ingest.v1"

MANIFEST_FILENAME = "manifest.json"
CANDIDATES_FILENAME = "candidates.jsonl"
GOLD_FILENAME = "ner_gold.json"
PAYLOAD_FILES = (CANDIDATES_FILENAME, GOLD_FILENAME)


class ExportIngestError(ValueError):
    """Raised for any export-bundle verification or ingest failure (file/line specific)."""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExportIngestError(f"cannot read {path}: {exc}") from exc


def _expected_hash(manifest_path: Path, files: dict[str, Any], name: str) -> str:
    entry = files.get(name)
    if not isinstance(entry, dict):
        raise ExportIngestError(f"{manifest_path}: files.{name} entry is missing")
    recorded = entry.get("blake3")
    if not isinstance(recorded, str) or not recorded:
        raise ExportIngestError(f"{manifest_path}: files.{name}.blake3 hash is missing")
    # The exporter writes ``b3:<hex>`` ids; accept the bare hex form too.
    return recorded.removeprefix("b3:")


def verify_bundle(bundle_dir: str | Path) -> dict[str, Any]:
    """Verify the manifest, payload hashes, and gold schema; returns the manifest."""
    bundle_dir = Path(bundle_dir)
    manifest_path = bundle_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        raise ExportIngestError(f"bundle {bundle_dir} has no {MANIFEST_FILENAME}")
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ExportIngestError(f"{manifest_path}: manifest must be a JSON object")
    schema_version = manifest.get("schema_version")
    if schema_version != EXPORT_SCHEMA_VERSION:
        raise ExportIngestError(
            f"{manifest_path}: schema_version {schema_version!r} does not match expected {EXPORT_SCHEMA_VERSION!r}"
        )
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ExportIngestError(f"{manifest_path}: 'files' object is missing")
    # Extra files in the bundle are ignored on purpose (forward compatibility).
    for name in PAYLOAD_FILES:
        expected = _expected_hash(manifest_path, files, name)
        path = bundle_dir / name
        if not path.exists():
            raise ExportIngestError(f"bundle {bundle_dir} is missing payload file {name}")
        actual = hash_candidates_file(path)
        if actual != expected:
            raise ExportIngestError(f"{path}: blake3 mismatch (manifest {expected}, actual {actual})")
    for key in ("task_counts", "family_counts"):
        if not isinstance(manifest.get(key), dict):
            raise ExportIngestError(f"{manifest_path}: '{key}' object is missing")
    gold = _read_json(bundle_dir / GOLD_FILENAME)
    if not isinstance(gold, dict):
        raise ExportIngestError(f"{bundle_dir / GOLD_FILENAME}: gold benchmark must be a JSON object")
    gold_schema = gold.get("schema_version")
    if gold_schema != BENCHMARK_SCHEMA_VERSION:
        raise ExportIngestError(
            f"{bundle_dir / GOLD_FILENAME}: schema_version {gold_schema!r} "
            f"does not match expected {BENCHMARK_SCHEMA_VERSION!r}"
        )
    if not isinstance(gold.get("cases"), list) or not gold["cases"]:
        raise ExportIngestError(f"{bundle_dir / GOLD_FILENAME}: 'cases' must be a non-empty list")
    return manifest


def _default_workdir() -> Path:
    """Same resolution as ``cli.workdir()``; kept local so ``cli`` can import this module."""
    return Path(os.environ.get("MEDLINER_WORKDIR", "data/materialized"))


def ingest_export(bundle_dir: str | Path, *, workdir: str | Path | None = None) -> dict[str, Any]:
    """Verify the bundle, validate it with MEDliNER's own rules, materialize it under ``workdir``.

    Writes ``<workdir>/ingested/{candidates.jsonl, ner_gold.json, ingest-manifest.json}``
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
    try:
        benchmark = load_dakp_benchmark(gold_path)
    except (KeyError, TypeError, ValueError) as exc:
        raise ExportIngestError(f"{gold_path}: malformed gold case: {exc}") from exc

    output_dir = Path(workdir) / "ingested" if workdir is not None else _default_workdir() / "ingested"
    output_dir.mkdir(parents=True, exist_ok=True)
    materialized_candidates = output_dir / CANDIDATES_FILENAME
    materialized_gold = output_dir / GOLD_FILENAME
    shutil.copyfile(candidates_path, materialized_candidates)
    shutil.copyfile(gold_path, materialized_gold)
    ingest_manifest = {
        "schema_version": INGEST_SCHEMA_VERSION,
        "bundle_path": str(bundle_dir),
        "export_schema_version": manifest["schema_version"],
        "candidates_blake3": hash_candidates_file(materialized_candidates),
        "gold_blake3": hash_candidates_file(materialized_gold),
        "task_counts": manifest["task_counts"],
        "family_counts": manifest["family_counts"],
        "ingested_at": datetime.now(UTC).isoformat(),
    }
    materialized_manifest = output_dir / "ingest-manifest.json"
    materialized_manifest.write_text(json.dumps(ingest_manifest, indent=2) + "\n", encoding="utf-8")
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
