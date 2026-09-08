from __future__ import annotations

import json
import tempfile
from collections import Counter
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

import pytest

from medliner.candidates import (
    GENERATOR_VERSION,
    CandidateInputError,
    CandidateText,
    PinnedSpl,
    apply_pins,
    build_import_tasks,
    build_warmup_tasks,
    hash_candidates_file,
    import_file_name,
    import_manifest,
    read_candidates,
    read_pins,
    sample_tasks,
    source_reference,
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


def test_difficulty_score_ranks_annotation_traps_above_plain_text():
    from medliner.candidates import difficulty_score

    plain = difficulty_score("Indicated for the treatment of asthma.")
    traps = difficulty_score(
        "Contraindicated in patients with suspected active peptic ulcer disease and recent gastrointestinal "
        "bleeding; caution is advised in women of childbearing potential. Not recommended with aspirin 81 mg, "
        "warfarin, or other NSAIDs (see WARNINGS, CABG)."
    )
    assert traps > plain
    assert difficulty_score("same text") == difficulty_score("same text")  # deterministic


def test_sample_tasks_prefers_edge_cases_and_keeps_a_control_slice():
    hard_text = (
        "Contraindicated in patients with suspected severe hepatic impairment; caution in women of "
        "childbearing potential. Do not coadminister with probenecid, aspirin 81 mg, or NSAIDs."
    )
    rows = [_row(f"Case {i}. {hard_text}", source_family="dailymed", source_document_id=f"hard-{i}") for i in range(5)]
    rows += [
        _row(f"Indicated for condition number {i}.", source_family="dailymed", source_document_id=f"easy-{i}")
        for i in range(5)
    ]
    tasks = build_import_tasks([CandidateText.model_validate(row) for row in rows])
    sampled = sample_tasks(tasks, {"indication": 5}, seed=2026, edge_fraction=0.8)
    assert len(sampled) == 5
    hard_selected = [task for task in sampled if "Contraindicated" in task["data"]["text"]]
    assert len(hard_selected) == 4  # 80% of the allocation goes to the hardest texts
    # Deterministic, and the remaining slot comes from the hash-random control slice.
    assert sample_tasks(tasks, {"indication": 5}, seed=2026, edge_fraction=0.8) == sampled
    # edge_fraction=0 restores the plain hash-random selection.
    assert sample_tasks(tasks, {"indication": 5}, seed=2026, edge_fraction=0.0) == sample_tasks(
        tasks, {"indication": 5}, seed=2026
    )


def test_sample_tasks_rejects_out_of_range_edge_fraction():
    tasks = _pool({("indication", "dailymed"): 1})
    with pytest.raises(ValueError, match="edge_fraction"):
        sample_tasks(tasks, {"indication": 1}, edge_fraction=1.5)


def _max_task_run(kinds: list[str]) -> int:
    longest = current = 1
    for previous, item in pairwise(kinds):  # successive pairs
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
    assert kinds.count("indication") == 8
    assert kinds.count("contraindication") == 2


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
    assert sampled.startswith("import-")
    assert sampled != import_file_name(input_hash=digest)
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
                "mentions": [{"surface": "asthma", "type": "DiseaseOrPhenotypicFeature"}],
            },
            {
                "id": "faers-case-1",
                "source": "faers",
                "text": "Used for migraine prophylaxis.",
                "mentions": [{"surface": "migraine", "type": "DiseaseOrPhenotypicFeature", "start": 9}],
            },
        ]
    )
    tasks = build_warmup_tasks(gold)
    assert [task["data"]["task"] for task in tasks] == ["contraindication", "indication"]
    assert all(task["data"]["source_family"] == "gold-warmup" for task in tasks)
    assert all(task["data"]["warmup"] is True for task in tasks)
    ibuprofen, faers = tasks
    assert ibuprofen["id"].startswith("warmup-")
    assert ibuprofen["data"]["gold_mentions"] == [
        {"start": 33, "end": 39, "label": "DiseaseOrPhenotypicFeature", "text": "asthma"}
    ]
    assert faers["data"]["gold_mentions"] == [
        {"start": 9, "end": 17, "label": "DiseaseOrPhenotypicFeature", "text": "migraine"}
    ]
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
                "mentions": [{"surface": "absent", "type": "DiseaseOrPhenotypicFeature"}],
            }
        ]
    )
    with pytest.raises(CandidateInputError, match="is not present in its text"):
        build_warmup_tasks(bad_offset)
    with pytest.raises(ValueError, match="at least 1"):
        build_warmup_tasks(_gold([{"id": "a", "source": "faers", "text": "t", "mentions": []}]), limit=0)


# --- annotation-screen display fields -----------------------------------------------------------


def test_import_tasks_always_carry_the_screen_display_fields():
    """Label Studio renders a missing "$var" literally, so both keys exist on every task."""
    tasks = build_import_tasks(
        [
            CandidateText(text="Contraindicated in asthma.", task="contraindication"),
            CandidateText(
                text="Indicated for asthma.", task="indication", source_family="dailymed", source_document_id="spl-1"
            ),
        ]
    )
    assert all("shortened_note" in task["data"] and "source_ref" in task["data"] for task in tasks)
    assert all(task["data"]["shortened_note"] == "" for task in tasks)  # nothing shortened yet


def test_warmup_tasks_carry_the_screen_display_fields():
    payload = {
        "cases": [
            {
                "id": "c1",
                "source": "dailymed",
                "text": "Contraindicated in asthma.",
                "mentions": [{"surface": "asthma", "type": "DiseaseOrPhenotypicFeature"}],
            }
        ]
    }
    with tempfile.TemporaryDirectory() as tmp:
        gold = Path(tmp) / "ner_gold.json"
        gold.write_text(json.dumps(payload), encoding="utf-8")
        (task,) = build_warmup_tasks(gold)
    assert task["data"]["shortened_note"] == ""
    assert "source_ref" in task["data"]


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        # An explicit URI is the most trustworthy link and wins over any derived one.
        (
            {"family": "dailymed", "document_id": "spl-1", "source_uri": "https://example.test/spl-1"},
            '<a href="https://example.test/spl-1" target="_blank" rel="noopener">DailyMed SPL spl-1</a>',
        ),
        # A real setid is linkable; DailyMed resolves it directly.
        (
            {"family": "dailymed", "document_id": "3e2f1a0b-4c5d-6e7f-8a9b-0c1d2e3f4a5b"},
            '<a href="https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=3e2f1a0b-4c5d-6e7f-8a9b-0c1d2e3f4a5b"'
            ' target="_blank" rel="noopener">DailyMed SPL 3e2f1a0b-4c5d-6e7f-8a9b-0c1d2e3f4a5b</a>',
        ),
        # A placeholder id stays plain text: a dead link is worse than no link.
        ({"family": "dailymed", "document_id": "spl-document-001"}, "DailyMed SPL spl-document-001"),
        ({"family": "faers", "record_id": "case-12345"}, "FAERS case case-12345"),
        ({"family": "unknown", "document_id": "doc-9"}, "unknown doc-9"),
        ({"family": "dailymed"}, ""),  # nothing to point at
    ],
)
def test_source_reference_links_only_what_it_can_resolve(kwargs, expected):
    assert source_reference(**kwargs) == expected


def test_source_reference_escapes_its_inputs():
    """The value is rendered as HTML by <HyperText>, so it may never carry raw markup."""
    rendered = source_reference(family="dailymed", document_id='<script>"x"')
    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered


# --- pinned SPLs ----------------------------------------------------------------------------------

PIN_FIXTURE = Path(__file__).parent / "fixtures" / "pinned_spls.json"
PIN_SETID_1 = "11111111-2222-3333-4444-555555555555"
PIN_SETID_2 = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


def test_read_pins_parses_the_fixture():
    pins = read_pins(PIN_FIXTURE)
    assert [pin.setid for pin in pins.pins] == [PIN_SETID_1, PIN_SETID_2]
    assert pins.pins[0].notes == ["supports asthma"]
    assert pins.pins[1].notes == ["supports migraine", "supports tension headache"]
    assert pins.unattributed_notes[0].note == "unplaced review comment"


def test_read_pins_reports_entry_numbers_and_validates_setids(tmp_path):
    path = tmp_path / "pins.json"
    path.write_text(json.dumps({"version": 1, "pins": [{"setid": "not-a-setid"}]}), encoding="utf-8")
    with pytest.raises(CandidateInputError, match="invalid pin at line 1"):
        read_pins(path)
    path.write_text(json.dumps({"version": 1, "pins": [{"setid": PIN_SETID_1}, {"setid": "bad"}]}), encoding="utf-8")
    with pytest.raises(CandidateInputError, match="invalid pin at line 2"):
        read_pins(path)
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(CandidateInputError, match="cannot read pin file"):
        read_pins(path)
    path.write_text(json.dumps({"version": 2, "pins": []}), encoding="utf-8")
    with pytest.raises(CandidateInputError, match="unsupported pin file version"):
        read_pins(path)
    path.write_text(json.dumps({"version": 1, "unattributed_notes": [{"note": "  "}]}), encoding="utf-8")
    with pytest.raises(CandidateInputError, match="invalid unattributed note at line 1"):
        read_pins(path)


def test_apply_pins_matches_loinc_suffixed_and_bare_setids():
    pin = PinnedSpl(setid=PIN_SETID_1, notes=["supports asthma"])
    tasks = build_import_tasks(
        [
            CandidateText(
                text="Indicated for asthma.",
                task="indication",
                source_family="dailymed",
                source_document_id=f"{PIN_SETID_1}#34067-9",
            ),
            CandidateText(
                text="Contraindicated in asthma.",
                task="contraindication",
                source_family="dailymed",
                source_document_id=PIN_SETID_1,  # bare setid, no #<LOINC> suffix
            ),
            CandidateText(
                text="Indicated for migraine.",
                task="indication",
                source_family="dailymed",
                source_document_id="doc-other",
            ),
        ]
    )
    stats = apply_pins(tasks, [pin])
    assert stats == {"matched": {PIN_SETID_1: 2}, "unmatched": []}
    flags = {task["data"]["text"]: task["data"]["pinned"] for task in tasks}
    assert flags == {
        "Indicated for asthma.": True,
        "Contraindicated in asthma.": True,
        "Indicated for migraine.": False,
    }


def test_apply_pins_reports_unmatched_setids():
    tasks = build_import_tasks([CandidateText(text="Indicated for asthma.", task="indication")])
    stats = apply_pins(tasks, read_pins(PIN_FIXTURE).pins)
    assert stats["matched"] == {}
    assert stats["unmatched"] == [PIN_SETID_1, PIN_SETID_2]


def test_every_task_carries_the_pinned_default():
    tasks = build_import_tasks([CandidateText(text="Indicated for asthma.", task="indication")])
    assert all(task["data"]["pinned"] is False for task in tasks)
    (warmup,) = build_warmup_tasks(
        _gold(
            [
                {
                    "id": "c1",
                    "source": "dailymed",
                    "text": "Contraindicated in asthma.",
                    "mentions": [{"surface": "asthma", "type": "DiseaseOrPhenotypicFeature"}],
                }
            ]
        )
    )
    assert warmup["data"]["pinned"] is False


def test_pinned_tasks_bypass_sampling_caps_and_lead_the_order():
    """Mirrors run_candidates: pins leave the pool before sample_tasks and are prepended after."""
    long_text = " ".join(["word"] * 301)
    rows = [
        CandidateText(
            text=long_text, task="indication", source_family="dailymed", source_document_id=f"{PIN_SETID_1}#34067-9"
        )
    ]
    rows += [
        CandidateText(
            text=f"Indicated for condition number {index}.",
            task="indication",
            source_family="dailymed",
            source_document_id=f"doc-{index}",
        )
        for index in range(4)
    ]
    tasks = build_import_tasks(rows)
    apply_pins(tasks, [PinnedSpl(setid=PIN_SETID_1)])
    pinned = [task for task in tasks if task["data"]["pinned"]]
    rest = [task for task in tasks if not task["data"]["pinned"]]
    final = pinned + sample_tasks(rest, {"indication": 1}, max_words=300)
    assert final[0]["data"]["pinned"] is True
    # The 301-word pinned text survives both the max_words cap and the per-task target of 1.
    assert len(final[0]["data"]["text"].split()) == 301
    assert len(final) == 2


def test_note_text_never_enters_task_data():
    pins = read_pins(PIN_FIXTURE)
    tasks = build_import_tasks(
        [
            CandidateText(
                text="Indicated for asthma.",
                task="indication",
                source_family="dailymed",
                source_document_id=f"{PIN_SETID_1}#34067-9",
            )
        ]
    )
    apply_pins(tasks, pins.pins)
    blob = json.dumps(tasks)
    for note in ("supports asthma", "supports migraine", "supports tension headache", "unplaced review comment"):
        assert note not in blob
