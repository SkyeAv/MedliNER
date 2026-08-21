"""Ingest of the committed DAKP export fixture: the end-to-end contract proof.

The fixture under ``tests/fixtures/dakp_export`` is a REAL bundle produced by DAKP's
exporter (``dakp export-medliner --fixtures``), so these tests prove the on-disk contract
between the two repos, not just our own writer.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from medliner.candidates import build_import_tasks, hash_candidates_file, read_candidates
from medliner.evaluation import load_gold_benchmark, score_examples
from medliner.export_ingest import (
    BENCHMARK_SCHEMA_VERSION,
    EXPORT_SCHEMA_VERSION,
    INGEST_SCHEMA_VERSION,
    ExportIngestError,
    ingest_export,
    verify_bundle,
)
from medliner.schema import Example

FIXTURE = Path(__file__).parent / "fixtures" / "dakp_export"


def _copy_bundle(target: Path) -> Path:
    """Work on a private copy so negative tests can corrupt the bundle freely."""
    shutil.copytree(FIXTURE, target / "bundle")
    return target / "bundle"


def _rehash(bundle: Path, *names: str) -> None:
    """Rewrite manifest hashes after deliberately mutating payload files."""
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name in names:
        manifest["files"][name]["blake3"] = f"b3:{hash_candidates_file(bundle / name)}"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_end_to_end_ingested_bundle_becomes_label_studio_import(tmp_path):
    """Prove the cross-repo export→ingest contract entirely offline with the committed fixture."""
    result = ingest_export(FIXTURE, workdir=tmp_path)
    bundle_manifest = json.loads((FIXTURE / "manifest.json").read_text(encoding="utf-8"))

    ingested = tmp_path / "ingested"
    # Payload lands as byte copies; the ingest manifest is the only new file.
    assert (ingested / "candidates.jsonl").read_bytes() == (FIXTURE / "candidates.jsonl").read_bytes()
    assert (ingested / "ner_gold.json").read_bytes() == (FIXTURE / "ner_gold.json").read_bytes()

    # Rows validate through MEDliNER's own CandidateText rules and feed Label Studio directly.
    candidates = read_candidates(ingested / "candidates.jsonl")
    assert len(candidates) == bundle_manifest["files"]["candidates.jsonl"]["rows"]
    tasks = build_import_tasks(candidates)
    assert len(tasks) == len(candidates)  # the exporter already deduped
    assert all(task["id"].startswith("medliner-") for task in tasks)
    assert all({"text", "task", "source_family", "generator_version"} <= task["data"].keys() for task in tasks)
    assert sum(task["data"].get("duplicate_count", 1) - 1 for task in tasks) == 0

    # The gold file parses into canonical Examples through the evaluation parser and can be
    # scored without a model or network access, proving the benchmark side of the contract too.
    examples = load_gold_benchmark(ingested / "ner_gold.json")
    assert len(examples) == bundle_manifest["files"]["ner_gold.json"]["cases"]
    assert all(isinstance(example, Example) for example in examples)
    report = score_examples(lambda _text: [], examples)
    assert report["examples"] == len(examples)
    assert report["overall"]["strict"]["tp"] == 0
    assert report["overall"]["strict"]["fn"] > 0

    # Counts and hashes round-trip through the ingest manifest.
    ingest_manifest = json.loads((ingested / "ingest-manifest.json").read_text(encoding="utf-8"))
    assert ingest_manifest["schema_version"] == INGEST_SCHEMA_VERSION
    assert ingest_manifest["export_schema_version"] == EXPORT_SCHEMA_VERSION
    assert ingest_manifest["task_counts"] == bundle_manifest["task_counts"]
    assert ingest_manifest["family_counts"] == bundle_manifest["family_counts"]
    assert sum(ingest_manifest["task_counts"].values()) == len(candidates)
    assert ingest_manifest["candidates_blake3"] == hash_candidates_file(ingested / "candidates.jsonl")
    assert ingest_manifest["gold_blake3"] == hash_candidates_file(ingested / "ner_gold.json")
    assert result["candidate_rows"] == len(candidates)
    assert result["gold_cases"] == len(examples)


def test_ingest_export_defaults_to_the_medliner_workdir(tmp_path, monkeypatch):
    """Without an explicit workdir the stage resolves $MEDLINER_WORKDIR like every other stage."""
    monkeypatch.setenv("MEDLINER_WORKDIR", str(tmp_path / "work"))
    result = ingest_export(FIXTURE)
    assert result["candidates_path"] == tmp_path / "work" / "ingested" / "candidates.jsonl"


def test_ingest_export_is_idempotent(tmp_path):
    """Re-running ingest overwrites the materialized files cleanly."""
    first = ingest_export(FIXTURE, workdir=tmp_path)
    second = ingest_export(FIXTURE, workdir=tmp_path)
    assert first["candidates_path"].read_bytes() == second["candidates_path"].read_bytes()
    assert first["manifest_path"].read_bytes() != b""  # manifest rewritten, not appended


def test_verify_bundle_rejects_an_unknown_schema_version(tmp_path):
    """A schema bump must be a loud error, never a best-effort ingest."""
    bundle = _copy_bundle(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    manifest["schema_version"] = "dakp.medliner.export.v2"
    _write_json(bundle / "manifest.json", manifest)
    with pytest.raises(ExportIngestError, match=r"schema_version"):
        verify_bundle(bundle)


def test_verify_bundle_rejects_a_corrupted_payload(tmp_path):
    """Any byte-level drift in a payload file is caught by the manifest hash check."""
    bundle = _copy_bundle(tmp_path)
    candidates = bundle / "candidates.jsonl"
    candidates.write_text(candidates.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")
    with pytest.raises(ExportIngestError, match=r"candidates\.jsonl: blake3 mismatch"):
        verify_bundle(bundle)


def test_verify_bundle_rejects_a_missing_payload_file(tmp_path):
    bundle = _copy_bundle(tmp_path)
    (bundle / "ner_gold.json").unlink()
    with pytest.raises(ExportIngestError, match="missing payload file ner_gold.json"):
        verify_bundle(bundle)


def test_verify_bundle_rejects_a_bundle_without_a_manifest(tmp_path):
    """An unknown or empty directory fails naming the manifest, not with a stack trace."""
    with pytest.raises(ExportIngestError, match="manifest.json"):
        verify_bundle(tmp_path / "absent")


def test_verify_bundle_rejects_empty_gold_cases(tmp_path):
    """A zero-case benchmark is useless for regression eval, so it is rejected upfront."""
    bundle = _copy_bundle(tmp_path)
    gold = json.loads((bundle / "ner_gold.json").read_text(encoding="utf-8"))
    gold["cases"] = []
    _write_json(bundle / "ner_gold.json", gold)
    _rehash(bundle, "ner_gold.json")  # hashes stay valid so the cases check itself fires
    with pytest.raises(ExportIngestError, match="non-empty"):
        verify_bundle(bundle)


def test_verify_bundle_rejects_a_wrong_gold_schema_version(tmp_path):
    """The gold schema string must equal the benchmark contract, guarding parser assumptions."""
    bundle = _copy_bundle(tmp_path)
    gold = json.loads((bundle / "ner_gold.json").read_text(encoding="utf-8"))
    gold["schema_version"] = "dakp.ner.gold.v0"
    _write_json(bundle / "ner_gold.json", gold)
    _rehash(bundle, "ner_gold.json")
    with pytest.raises(ExportIngestError, match=BENCHMARK_SCHEMA_VERSION.replace(".", r"\.")):
        verify_bundle(bundle)


def test_verify_bundle_accepts_hashes_without_the_b3_prefix(tmp_path):
    """Hash ids are accepted bare so hand-trimmed manifests do not need the prefix."""
    bundle = _copy_bundle(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["files"].values():
        entry["blake3"] = entry["blake3"].removeprefix("b3:")
    _write_json(bundle / "manifest.json", manifest)
    assert verify_bundle(bundle)["schema_version"] == EXPORT_SCHEMA_VERSION


def test_verify_bundle_ignores_extra_files(tmp_path):
    """Forward compatibility: future exporters may add files; ingest must not break."""
    bundle = _copy_bundle(tmp_path)
    (bundle / "future-payload.parquet").write_bytes(b"opaque")
    assert verify_bundle(bundle)["schema_version"] == EXPORT_SCHEMA_VERSION


def test_ingest_export_surfaces_invalid_row_line_numbers(tmp_path):
    """Rows are re-validated by MEDliNER's own rules; the line number must reach the user."""
    bundle = _copy_bundle(tmp_path)
    candidates = bundle / "candidates.jsonl"
    bad_row = json.dumps({"text": "Some observed use.", "task": "diagnosis", "source_family": "faers"})
    candidates.write_text(candidates.read_text(encoding="utf-8") + bad_row + "\n", encoding="utf-8")
    _rehash(bundle, "candidates.jsonl")  # hashes stay valid so row validation itself fires
    with pytest.raises(ExportIngestError, match=r"candidates\.jsonl: invalid candidate at line 9"):
        ingest_export(bundle, workdir=tmp_path / "work")


def test_ingest_export_rejects_a_malformed_gold_case(tmp_path):
    """A gold case that breaks the evaluation parser fails loudly at ingest, not at eval time."""
    bundle = _copy_bundle(tmp_path)
    gold = json.loads((bundle / "ner_gold.json").read_text(encoding="utf-8"))
    gold["cases"][0]["mentions"][0]["surface"] = "not actually present in the text"
    _write_json(bundle / "ner_gold.json", gold)
    _rehash(bundle, "ner_gold.json")
    with pytest.raises(ExportIngestError, match="malformed gold case"):
        ingest_export(bundle, workdir=tmp_path / "work")
