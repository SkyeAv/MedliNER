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


def _rewrite_candidate_line(bundle: Path, index: int, row: dict) -> None:
    """Replace one candidate row (0-based) and re-hash so row validation itself fires."""
    candidates = bundle / "candidates.ndjson"
    lines = candidates.read_text(encoding="utf-8").splitlines()
    lines[index] = json.dumps(row, ensure_ascii=False)
    candidates.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _rehash(bundle, "candidates.ndjson")


def _faers_row() -> dict:
    return json.loads((FIXTURE / "candidates.ndjson").read_text(encoding="utf-8").splitlines()[2])


def _dailymed_row() -> dict:
    return json.loads((FIXTURE / "candidates.ndjson").read_text(encoding="utf-8").splitlines()[0])


def test_end_to_end_ingested_bundle_becomes_label_studio_import(tmp_path):
    """Prove the cross-repo export→ingest contract entirely offline with the committed fixture."""
    result = ingest_export(FIXTURE, workdir=tmp_path)
    bundle_manifest = json.loads((FIXTURE / "manifest.json").read_text(encoding="utf-8"))

    ingested = tmp_path / "ingested"
    # Payload lands as byte copies; the ingest manifest is the only new file.
    assert (ingested / "candidates.ndjson").read_bytes() == (FIXTURE / "candidates.ndjson").read_bytes()
    assert (ingested / "ner_gold.json").read_bytes() == (FIXTURE / "ner_gold.json").read_bytes()

    # Rows validate through MedliNER's own CandidateText rules and feed Label Studio directly.
    candidates = read_candidates(ingested / "candidates.ndjson")
    assert len(candidates) == bundle_manifest["files"]["candidates.ndjson"]["rows"]
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
    assert ingest_manifest["candidates_blake3"] == hash_candidates_file(ingested / "candidates.ndjson")
    assert ingest_manifest["gold_blake3"] == hash_candidates_file(ingested / "ner_gold.json")
    assert result["candidate_rows"] == len(candidates)
    assert result["gold_cases"] == len(examples)


def test_ingest_export_defaults_to_the_medliner_workdir(tmp_path, monkeypatch):
    """Without an explicit workdir the stage resolves $MEDLINER_WORKDIR like every other stage."""
    monkeypatch.setenv("MEDLINER_WORKDIR", str(tmp_path / "work"))
    result = ingest_export(FIXTURE)
    assert result["candidates_path"] == tmp_path / "work" / "ingested" / "candidates.ndjson"


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


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("candidate_schema", "medliner.candidates.v0", "candidate_schema"),
        ("benchmark_schema", "dakp.ner.gold.v0", "benchmark_schema"),
    ],
)
def test_verify_bundle_rejects_manifest_schema_mismatches(tmp_path, field, value, message):
    """WHY: both producer schema declarations must match the payload contracts."""
    bundle = _copy_bundle(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    manifest[field] = value
    _write_json(bundle / "manifest.json", manifest)
    with pytest.raises(ExportIngestError, match=message):
        verify_bundle(bundle)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda manifest: manifest["files"]["candidates.ndjson"].update(rows=7), "row count mismatch"),
        (lambda manifest: manifest["task_counts"].update(indication=5), "task_counts mismatch"),
        (lambda manifest: manifest["family_counts"].update(faers=2), "family_counts mismatch"),
        (lambda manifest: manifest["files"]["ner_gold.json"].update(cases=33), "case count mismatch"),
    ],
)
def test_verify_bundle_rejects_manifest_count_mismatches(tmp_path, mutation, message):
    """WHY: manifest counts must describe payload contents rather than trusted claims."""
    bundle = _copy_bundle(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    mutation(manifest)
    _write_json(bundle / "manifest.json", manifest)
    with pytest.raises(ExportIngestError, match=message):
        verify_bundle(bundle)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        # WHY: generated_at is required metadata; a bundle without it is not self-describing.
        (None, "generated_at"),  # manifest.get() yields None when the key is absent
        # WHY: non-strings and unparseable strings are not ISO-8601 timestamps.
        (20260821, "generated_at"),
        ("not-a-timestamp", "not a valid ISO-8601"),
        # WHY: a naive stamp is ambiguous; the exporter always emits timezone-aware UTC.
        ("2026-08-21T20:28:47", "timezone-aware"),
        # WHY: a non-zero offset is not UTC, which the exporter never emits.
        ("2026-08-21T20:28:47+01:00", "must be UTC"),
    ],
)
def test_verify_bundle_rejects_malformed_generated_at(tmp_path, value, message):
    """WHY: generated_at must be a parseable timezone-aware UTC ISO-8601 string."""
    bundle = _copy_bundle(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if value is None:
        del manifest["generated_at"]
    else:
        manifest["generated_at"] = value
    _write_json(bundle / "manifest.json", manifest)
    with pytest.raises(ExportIngestError, match=message):
        verify_bundle(bundle)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        # WHY: inputs records the consumed interim tables; a bundle without them hides
        # its provenance, and the exporter errors loudly rather than emitting none.
        (lambda manifest: manifest.pop("inputs"), "inputs"),
        (lambda manifest: manifest.update(inputs=[]), "non-empty"),
        (lambda manifest: manifest.update(inputs="b3:only-one"), "non-empty list"),
        # WHY: blank or non-string entries cannot identify any consumed table.
        (lambda manifest: manifest.update(inputs=["", "b3:second"]), "non-blank strings"),
        (lambda manifest: manifest.update(inputs=[123, "b3:second"]), "non-blank strings"),
        # WHY: order must be deterministic so identical inputs produce identical manifests.
        (lambda manifest: manifest.update(inputs=list(reversed(manifest["inputs"]))), "sorted"),
    ],
)
def test_verify_bundle_rejects_malformed_inputs(tmp_path, mutation, message):
    """WHY: inputs must be a non-empty, sorted list of non-blank artifact ids."""
    bundle = _copy_bundle(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    mutation(manifest)
    _write_json(bundle / "manifest.json", manifest)
    with pytest.raises(ExportIngestError, match=message):
        verify_bundle(bundle)


def test_verify_bundle_accepts_a_zulu_generated_at(tmp_path):
    """WHY: ``Z`` is the canonical ISO-8601 UTC suffix and must be accepted like ``+00:00``."""
    bundle = _copy_bundle(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    manifest["generated_at"] = manifest["generated_at"].replace("+00:00", "Z")
    _write_json(bundle / "manifest.json", manifest)
    assert verify_bundle(bundle)["schema_version"] == EXPORT_SCHEMA_VERSION


def test_verify_bundle_rejects_non_exact_file_entries(tmp_path):
    """WHY: exact file count fields prevent omitted or unverified payload metadata."""
    bundle = _copy_bundle(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    manifest["files"]["candidates.ndjson"]["unexpected"] = 1
    _write_json(bundle / "manifest.json", manifest)
    with pytest.raises(ExportIngestError, match=r"files\.candidates\.ndjson"):
        verify_bundle(bundle)


def test_verify_bundle_rejects_a_corrupted_payload(tmp_path):
    """Any byte-level drift in a payload file is caught by the manifest hash check."""
    bundle = _copy_bundle(tmp_path)
    candidates = bundle / "candidates.ndjson"
    candidates.write_text(candidates.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")
    with pytest.raises(ExportIngestError, match=r"candidates\.ndjson: blake3 mismatch"):
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
    """Rows are re-validated by MedliNER's own rules; the line number must reach the user."""
    bundle = _copy_bundle(tmp_path)
    candidates = bundle / "candidates.ndjson"
    bad_row = json.dumps(
        {
            "text": "Some observed use.",
            "task": "diagnosis",
            "source_family": "faers",
            "source_record_id": "9999",
            "source_uri": "https://fis.fda.gov/content/Exports/faers_ascii_2024q3.zip",
        }
    )
    candidates.write_text(candidates.read_text(encoding="utf-8") + bad_row + "\n", encoding="utf-8")
    _rehash(bundle, "candidates.ndjson")  # hashes stay valid so row validation itself fires
    with pytest.raises(ExportIngestError, match=r"candidates\.ndjson: invalid candidate at line 9"):
        ingest_export(bundle, workdir=tmp_path / "work")


@pytest.mark.parametrize(
    "mutation",
    [
        # WHY: Pydantic silently ignores extra fields, so an unknown key must be rejected
        # here — the exporter promises no fields MedliNER ignores.
        lambda row: row | {"source_hash": "b3:deadbeef"},
        # WHY: DailyMed provenance is required, not optional — CandidateText's defaults
        # apply to manual candidates only, never to the export bundle.
        lambda row: {key: value for key, value in row.items() if key != "section"},
        lambda row: {key: value for key, value in row.items() if key != "source_document_id"},
        # WHY: wrong family-specific field — a DailyMed row cannot carry the FAERS record id.
        lambda row: {**{k: v for k, v in row.items() if k != "source_document_id"}, "source_record_id": "9999"},
    ],
)
def test_verify_bundle_rejects_dailymed_rows_violating_the_wire_contract(tmp_path, mutation):
    """WHY: the export contract forbids extras and requires DailyMed provenance fields."""
    bundle = _copy_bundle(tmp_path)
    _rewrite_candidate_line(bundle, 0, mutation(_dailymed_row()))
    with pytest.raises(ExportIngestError, match=r"candidates\.ndjson: line 1"):
        verify_bundle(bundle)


@pytest.mark.parametrize(
    "mutation",
    [
        # WHY: unknown/extra fields are forbidden even when otherwise well-formed.
        lambda row: row | {"source_hash": "b3:deadbeef"},
        # WHY: FAERS provenance (source_record_id / source_uri) is required, not optional.
        lambda row: {key: value for key, value in row.items() if key != "source_record_id"},
        lambda row: {key: value for key, value in row.items() if key != "source_uri"},
        # WHY: wrong family-specific field — a FAERS row cannot carry a DailyMed section.
        lambda row: {**{k: v for k, v in row.items() if k != "source_record_id"}, "section": "34067-9"},
    ],
)
def test_verify_bundle_rejects_faers_rows_violating_the_wire_contract(tmp_path, mutation):
    """WHY: FAERS rows must carry exactly their five contract fields."""
    bundle = _copy_bundle(tmp_path)
    _rewrite_candidate_line(bundle, 2, mutation(_faers_row()))
    with pytest.raises(ExportIngestError, match=r"candidates\.ndjson: line 3"):
        verify_bundle(bundle)


@pytest.mark.parametrize(
    ("line", "message"),
    [
        # WHY: an unknown family has no defined wire shape and must never be ingested.
        (json.dumps({"text": "x", "task": "indication", "source_family": "unknown"}), r"source_family 'unknown'"),
        # WHY: a non-object NDJSON line is not a candidate row at all.
        ('["text", "task"]', "must be a JSON object"),
    ],
)
def test_verify_bundle_rejects_unknown_families_and_non_object_lines(tmp_path, line, message):
    """WHY: malformed rows must name the file and line, like read_candidates does."""
    bundle = _copy_bundle(tmp_path)
    candidates = bundle / "candidates.ndjson"
    candidates.write_text(candidates.read_text(encoding="utf-8") + line + "\n", encoding="utf-8")
    _rehash(bundle, "candidates.ndjson")
    with pytest.raises(ExportIngestError, match=message):
        verify_bundle(bundle)


def test_ingest_export_rejects_a_malformed_gold_case(tmp_path):
    """A gold case that breaks the evaluation parser fails loudly at ingest, not at eval time."""
    bundle = _copy_bundle(tmp_path)
    gold = json.loads((bundle / "ner_gold.json").read_text(encoding="utf-8"))
    gold["cases"][0]["mentions"][0]["surface"] = "not actually present in the text"
    _write_json(bundle / "ner_gold.json", gold)
    _rehash(bundle, "ner_gold.json")
    with pytest.raises(ExportIngestError, match="malformed gold case"):
        ingest_export(bundle, workdir=tmp_path / "work")


def test_ingest_export_rejects_an_overflowing_gold_offset(tmp_path):
    """WHY: an overflowing numeric offset (e.g. 1e999 → inf) escapes the ValueError family
    as OverflowError; it must still surface as ExportIngestError naming ner_gold.json."""
    bundle = _copy_bundle(tmp_path)
    gold = json.loads((bundle / "ner_gold.json").read_text(encoding="utf-8"))
    gold["cases"][0]["mentions"][0]["start"] = float("1e999")  # inf; int(inf) overflows
    # json.dumps emits the ``Infinity`` token, which json.loads round-trips back to inf.
    _write_json(bundle / "ner_gold.json", gold)
    _rehash(bundle, "ner_gold.json")
    with pytest.raises(ExportIngestError, match=r"ner_gold\.json: malformed gold case"):
        ingest_export(bundle, workdir=tmp_path / "work")


def test_verify_bundle_normalizes_directory_payload_errors(tmp_path):
    """WHY: directory payloads must not leak IsADirectoryError through the public ingest API."""
    bundle = _copy_bundle(tmp_path)
    candidates = bundle / "candidates.ndjson"
    candidates.unlink()
    candidates.mkdir()
    with pytest.raises(ExportIngestError, match=r"candidates\.ndjson"):
        verify_bundle(bundle)


def test_verify_bundle_normalizes_invalid_utf8_errors(tmp_path):
    """WHY: invalid candidate bytes must remain an ExportIngestError with the payload path."""
    bundle = _copy_bundle(tmp_path)
    candidates = bundle / "candidates.ndjson"
    candidates.write_bytes(b"\xff\n")
    _rehash(bundle, "candidates.ndjson")
    with pytest.raises(ExportIngestError, match=r"candidates\.ndjson"):
        verify_bundle(bundle)


def test_verify_bundle_normalizes_invalid_gold_utf8_errors(tmp_path):
    """WHY: invalid gold bytes must remain an ExportIngestError with the payload path."""
    bundle = _copy_bundle(tmp_path)
    gold = bundle / "ner_gold.json"
    gold.write_bytes(b"\xff\n")
    _rehash(bundle, "ner_gold.json")
    with pytest.raises(ExportIngestError, match=r"ner_gold\.json"):
        verify_bundle(bundle)


def test_ingest_export_normalizes_output_copy_errors(tmp_path):
    """WHY: output filesystem failures must identify the affected materialized path."""
    bundle = _copy_bundle(tmp_path)
    output_dir = tmp_path / "work" / "ingested"
    output_dir.mkdir(parents=True)
    (output_dir / "candidates.ndjson").mkdir()
    with pytest.raises(ExportIngestError, match=r"candidates\.ndjson"):
        ingest_export(bundle, workdir=tmp_path / "work")
