from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from medliner.candidates import (
    GENERATOR_VERSION,
    CandidateInputError,
    CandidateText,
    build_import_tasks,
    build_warmup_tasks,
    hash_candidates_file,
    import_manifest,
    read_candidates,
    write_import_file,
)


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def _row(text: str = "Indicated for asthma.", task: str = "indication", **extra) -> dict:
    return {"text": text, "task": task, **extra}


def test_read_candidates_parses_jsonl_and_skips_blank_lines(tmp_path):
    path = _write(tmp_path / "candidates.jsonl", [_row(), _row("Contraindicated in asthma.", "contraindication")])
    path.write_text(path.read_text(encoding="utf-8") + "\n\n", encoding="utf-8")
    candidates = read_candidates(path)
    assert len(candidates) == 2
    assert candidates[1].task == "contraindication"
    assert candidates[0].source_family == "unknown"


def test_read_candidates_reports_malformed_jsonl_line(tmp_path):
    path = tmp_path / "candidates.jsonl"
    path.write_text('{"text": "ok", "task": "indication"}\nnot-json\n', encoding="utf-8")
    with pytest.raises(CandidateInputError, match="line 2"):
        read_candidates(path)


def test_candidate_text_rejects_blank_text_and_unknown_task():
    with pytest.raises(Exception, match="non-empty"):
        CandidateText(text="   ", task="indication")
    with pytest.raises(Exception, match="unsupported task"):
        CandidateText(text="Indicated for asthma.", task="diagnosis")


def test_read_candidates_reports_validation_line_number(tmp_path):
    path = _write(tmp_path / "candidates.jsonl", [_row(), _row(task="diagnosis")])
    with pytest.raises(CandidateInputError, match="line 2"):
        read_candidates(path)


def test_import_tasks_are_deterministic_and_match_the_label_studio_contract():
    candidates = [
        CandidateText(
            text="Contraindicated in patients with pulmonary hypertension.",
            task="contraindication",
            source_family="dailymed",
            source_document_id="spl-document-001",
        )
    ]
    stamp = datetime(2026, 1, 1, tzinfo=UTC)
    first = build_import_tasks(candidates, generated_at=stamp)
    second = build_import_tasks(candidates, generated_at=stamp)
    assert first == second
    (task,) = first
    assert task["id"].startswith("medliner-")
    assert task["data"]["text"] == candidates[0].text
    assert task["data"]["task"] == "contraindication"
    assert task["data"]["source_family"] == "dailymed"
    assert task["data"]["source_document_id"] == "spl-document-001"
    assert task["data"]["generator_version"] == GENERATOR_VERSION
    assert "predictions" not in task


def test_import_tasks_dedupe_normalized_text_and_count_duplicates():
    candidates = [
        CandidateText(text="Indicated for asthma.", task="indication", source_document_id="doc-a"),
        CandidateText(text="  indicated  for asthma. ", task="indication", source_document_id="doc-b"),
        CandidateText(text="Indicated for asthma.", task="contraindication", source_document_id="doc-c"),
    ]
    tasks = build_import_tasks(candidates)
    assert len(tasks) == 2  # task kind participates in the dedupe key
    merged = next(task for task in tasks if task["data"]["task"] == "indication")
    assert merged["data"]["duplicate_count"] == 2
    assert merged["data"]["source_document_id"] == "doc-a"


def test_import_manifest_counts_and_hashes_input(tmp_path):
    path = _write(
        tmp_path / "candidates.jsonl",
        [
            _row("Indicated for asthma.", source_family="dailymed"),
            _row("Indicated for hypertension.", source_family="faers"),
            _row("Contraindicated in asthma.", "contraindication", source_family="dailymed"),
        ],
    )
    tasks = build_import_tasks(read_candidates(path))
    manifest = import_manifest(tasks, input_path=path)
    assert manifest["input_hash"] == hash_candidates_file(path)
    assert manifest["task_count"] == 3
    assert manifest["task_counts"] == {"contraindication": 1, "indication": 2}
    assert manifest["family_counts"] == {"dailymed": 2, "faers": 1}
    assert manifest["duplicates_merged"] == 0


def test_write_import_file_round_trips_through_the_label_studio_reader(tmp_path):
    from medliner.label_studio import read_tasks

    tasks = build_import_tasks([CandidateText(text="Indicated for asthma.", task="indication")])
    path = tmp_path / "import.json"
    write_import_file(tasks, path)
    assert read_tasks(path) == tasks


def test_empty_candidates_produce_an_empty_import(tmp_path):
    path = tmp_path / "candidates.jsonl"
    path.write_text("", encoding="utf-8")
    assert read_candidates(path) == []
    assert build_import_tasks([]) == []


def _gold(cases: list[dict]) -> Path:
    path = Path(tempfile.mkdtemp()) / "ner_gold.json"
    payload = {"schema_version": "dakp.ner.gold.v1", "cases": cases}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_build_warmup_tasks_maps_gold_cases_to_demo_tasks():
    gold = _gold(
        [
            {
                "id": "dailymed-ibuprofen",
                "source": "dailymed",
                "text": "Contraindicated in patients with asthma.",
                "mentions": [{"surface": "asthma", "type": "disease"}],
            },
            {
                "id": "faers-case-1",
                "source": "faers",
                "text": "Used for migraine prophylaxis.",
                "mentions": [{"surface": "migraine", "type": "disease", "start": 9}],
            },
        ]
    )
    tasks = build_warmup_tasks(gold)
    assert [task["data"]["task"] for task in tasks] == ["contraindication", "indication"]
    assert all(task["data"]["source_family"] == "gold-warmup" for task in tasks)
    assert all(task["data"]["warmup"] is True for task in tasks)
    ibuprofen, faers = tasks
    assert ibuprofen["id"].startswith("warmup-")
    assert ibuprofen["data"]["gold_mentions"] == [{"start": 33, "end": 39, "label": "disease", "text": "asthma"}]
    assert faers["data"]["gold_mentions"] == [{"start": 9, "end": 17, "label": "disease", "text": "migraine"}]
    # Ids are deterministic (case-id keyed), so re-runs reproduce the same warm-up queue.
    assert [task["id"] for task in build_warmup_tasks(gold)] == [task["id"] for task in tasks]


def test_build_warmup_tasks_honors_the_limit():
    cases = [{"id": f"c{i}", "source": "faers", "text": f"Used for condition {i}.", "mentions": []} for i in range(5)]
    assert len(build_warmup_tasks(_gold(cases), limit=3)) == 3


def test_build_warmup_tasks_rejects_malformed_benchmarks():
    empty = Path(tempfile.mkdtemp()) / "empty.json"
    empty.write_text(json.dumps({"schema_version": "dakp.ner.gold.v1", "cases": []}), encoding="utf-8")
    with pytest.raises(CandidateInputError, match="non-empty"):
        build_warmup_tasks(empty)
    bad_offset = _gold(
        [
            {
                "id": "x",
                "source": "faers",
                "text": "short",
                "mentions": [{"surface": "absent", "type": "disease"}],
            }
        ]
    )
    with pytest.raises(CandidateInputError, match="is not present in its text"):
        build_warmup_tasks(bad_offset)
    with pytest.raises(ValueError, match="at least 1"):
        build_warmup_tasks(_gold([{"id": "a", "source": "faers", "text": "t", "mentions": []}]), limit=0)
