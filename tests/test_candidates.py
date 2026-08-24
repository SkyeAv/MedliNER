from __future__ import annotations

import json
import tempfile
from collections import Counter
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
    import_file_name,
    import_manifest,
    read_candidates,
    sample_tasks,
    stagger_tasks,
    write_import_file,
)


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def _row(text: str = "Indicated for asthma.", task: str = "indication", **extra) -> dict:
    return {"text": text, "task": task, **extra}


def test_read_candidates_parses_jsonl_and_skips_blank_lines(tmp_path):
    path = _write(tmp_path / "candidates.ndjson", [_row(), _row("Contraindicated in asthma.", "contraindication")])
    path.write_text(path.read_text(encoding="utf-8") + "\n\n", encoding="utf-8")
    candidates = read_candidates(path)
    assert len(candidates) == 2
    assert candidates[1].task == "contraindication"
    assert candidates[0].source_family == "unknown"


def test_read_candidates_reports_malformed_jsonl_line(tmp_path):
    path = tmp_path / "candidates.ndjson"
    path.write_text('{"text": "ok", "task": "indication"}\nnot-json\n', encoding="utf-8")
    with pytest.raises(CandidateInputError, match="line 2"):
        read_candidates(path)


def test_candidate_text_rejects_blank_text_and_unknown_task():
    with pytest.raises(Exception, match="non-empty"):
        CandidateText(text="   ", task="indication")
    with pytest.raises(Exception, match="unsupported task"):
        CandidateText(text="Indicated for asthma.", task="diagnosis")


def test_read_candidates_reports_validation_line_number(tmp_path):
    path = _write(tmp_path / "candidates.ndjson", [_row(), _row(task="diagnosis")])
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
    # Pre-annotation is a separate opt-in stage (medliner.prelabel), so this one stays
    # deterministic and free of ML dependencies.
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
        tmp_path / "candidates.ndjson",
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
    assert "sampling" not in manifest


def test_import_manifest_records_the_sampling_block(tmp_path):
    path = _write(tmp_path / "candidates.ndjson", [_row(), _row("Contraindicated in asthma.", "contraindication")])
    tasks = build_import_tasks(read_candidates(path))
    sampling = {
        "targets": {"indication": 1},
        "seed": 7,
        "max_words": 300,
        "max_run": 2,
        "pool_task_counts": {"indication": 1},
    }
    manifest = import_manifest(tasks, input_path=path, sampling=sampling)
    assert manifest["sampling"] == sampling


def _pool(counts: dict[tuple[str, str], int]) -> list[dict]:
    """Build deduplicated import tasks from ``(task, family) -> n`` counts."""
    rows = []
    for (task, family), number in counts.items():
        for index in range(number):
            rows.append(_row(f"{task} {family} number {index}.", task, source_family=family))
    return build_import_tasks([CandidateText.model_validate(row) for row in rows])


def test_sample_tasks_stratifies_families_and_reproduces_deterministically():
    tasks = _pool({("indication", "dailymed"): 6, ("indication", "faers"): 4, ("contraindication", "dailymed"): 5})
    sampled = sample_tasks(tasks, {"indication": 5, "contraindication": 3}, seed=2026)
    assert len(sampled) == 8
    families = Counter((task["data"]["task"], task["data"]["source_family"]) for task in sampled)
    # Largest-remainder split of the indication target across a 6/4 family pool.
    assert families == Counter(
        {("indication", "dailymed"): 3, ("indication", "faers"): 2, ("contraindication", "dailymed"): 3}
    )
    # Same input and configuration always reproduce the same subset.
    assert sample_tasks(tasks, {"indication": 5, "contraindication": 3}, seed=2026) == sampled
    assert sample_tasks(tasks, {"indication": 5, "contraindication": 3}, seed=7) != sampled


def test_sample_tasks_drop_long_texts_unlisted_tasks_and_honors_zero_targets():
    tasks = _pool({("indication", "dailymed"): 2, ("contraindication", "dailymed"): 2})
    tasks.append(build_import_tasks([CandidateText(text=" ".join(["word"] * 301), task="indication")])[0])
    capped = sample_tasks(tasks, {"indication": 10, "contraindication": 10}, max_words=300)
    assert len(capped) == 4  # the 301-word text is filtered out
    assert all(len(task["data"]["text"].split()) <= 300 for task in capped)
    whitelist = sample_tasks(tasks, {"indication": 10})
    assert {task["data"]["task"] for task in whitelist} == {"indication"}
    zeroed = sample_tasks(tasks, {"indication": 10, "contraindication": 0})
    assert {task["data"]["task"] for task in zeroed} == {"indication"}
    assert sample_tasks(tasks, {}) == tasks  # empty targets disable sampling


def test_sample_tasks_rejects_unknown_or_negative_targets():
    tasks = _pool({("indication", "dailymed"): 1})
    with pytest.raises(ValueError, match="unknown sampling task"):
        sample_tasks(tasks, {"indications": 5})
    with pytest.raises(ValueError, match="non-negative"):
        sample_tasks(tasks, {"indication": -1})


def _max_task_run(kinds: list[str]) -> int:
    longest = current = 1
    for previous, item in zip(kinds, kinds[1:], strict=False):  # deliberately off-by-one pairing
        current = current + 1 if item == previous else 1
        longest = max(longest, current)
    return longest


def test_stagger_tasks_bounds_task_runs_and_preserves_membership():
    tasks = _pool({("indication", "dailymed"): 10, ("indication", "faers"): 5, ("contraindication", "dailymed"): 5})
    staggered = stagger_tasks(tasks, max_run=3)
    kinds = [task["data"]["task"] for task in staggered]
    assert len(staggered) == len(tasks)
    assert {task["id"] for task in staggered} == {task["id"] for task in tasks}
    # 15 indications around 5 contraindications needs runs of 3 (ceil(15/6)); the cap holds
    # across the whole sequence at max_run=3, including the tail.
    assert _max_task_run(kinds) <= 3
    # Early positions mix families as well as task types.
    families = [task["data"]["source_family"] for task in staggered[:6]]
    assert len(set(families)) >= 2
    assert stagger_tasks(tasks, max_run=3) == staggered  # deterministic


def test_stagger_task_runs_only_bound_while_multiple_task_types_remain():
    tasks = _pool({("indication", "dailymed"): 8, ("contraindication", "dailymed"): 2})
    staggered = stagger_tasks(tasks, max_run=2)
    kinds = [task["data"]["task"] for task in staggered]
    last_minority = max(index for index, kind in enumerate(kinds) if kind == "contraindication")
    assert _max_task_run(kinds[: last_minority + 1]) <= 2
    assert kinds.count("indication") == 8 and kinds.count("contraindication") == 2


def test_stagger_tasks_rejects_invalid_max_run():
    tasks = _pool({("indication", "dailymed"): 2})
    with pytest.raises(ValueError, match="at least 1"):
        stagger_tasks(tasks, max_run=0)
    assert stagger_tasks([]) == []
    assert stagger_tasks(tasks[:1], max_run=1) == tasks[:1]


def test_import_file_name_legacy_and_sampling_aware():
    digest = "a" * 64
    assert import_file_name(input_hash=digest) == f"import-{digest[:16]}.json"
    sampled = import_file_name(input_hash=digest, sampling="tasks=indication:6000;seed=2026;max_words=300;max_run=3")
    assert sampled.startswith("import-") and sampled != import_file_name(input_hash=digest)
    other = import_file_name(input_hash=digest, sampling="tasks=indication:5000;seed=2026;max_words=300;max_run=3")
    assert other != sampled  # the configuration is part of the cache key
    assert (
        import_file_name(input_hash=digest, sampling="tasks=indication:6000;seed=2026;max_words=300;max_run=3")
        == sampled
    )


def test_write_import_file_round_trips_through_the_label_studio_reader(tmp_path):
    from medliner.label_studio import read_tasks

    tasks = build_import_tasks([CandidateText(text="Indicated for asthma.", task="indication")])
    path = tmp_path / "import.json"
    write_import_file(tasks, path)
    assert read_tasks(path) == tasks


def test_empty_candidates_produce_an_empty_import(tmp_path):
    path = tmp_path / "candidates.ndjson"
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
