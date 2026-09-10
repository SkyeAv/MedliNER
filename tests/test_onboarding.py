from __future__ import annotations

import json
from pathlib import Path

import pytest

from medliner.onboarding import (
    OnboardingConfig,
    OnboardingError,
    UnknownAnnotatorError,
    build_onboarding_tasks,
    build_test_bank,
    evaluate_attempt,
    filter_production_export,
    promote,
    promoted_users,
    read_reports,
    start_attempt,
    versioned_bank_path,
    write_report,
    write_test_bank,
)


def _benchmark(path: Path, *, changed: bool = False) -> Path:
    cases = []
    for index in range(10):
        if index == 0:
            text = "No condition is present."
            mentions: list[dict] = []
        else:
            surface = "asthma" if not changed or index != 1 else "migraine"
            text = f"Used for {surface}."
            mentions = [{"surface": surface, "type": "disease", "start": 9}]
        cases.append({"id": f"case-{index}", "source": "faers", "text": text, "mentions": mentions})
    path.write_text(json.dumps({"schema_version": "dakp.ner.gold.v1", "cases": cases}), encoding="utf-8")
    return path


def _context(tmp_path: Path, *, changed: bool = False):
    gold = _benchmark(tmp_path / ("gold-changed.json" if changed else "gold.json"), changed=changed)
    config = OnboardingConfig(case_ids=[f"case-{index}" for index in range(10)])
    manifest = build_test_bank(gold, config, generated_at="2026-01-01T00:00:00+00:00")
    write_test_bank(manifest, versioned_bank_path(tmp_path, manifest))
    attempt = start_attempt(tmp_path, manifest, config, "alice")
    return gold, config, manifest, attempt


def _export_for_attempt(
    tmp_path: Path, manifest, attempt, *, wrong: set[str] | None = None, user: str = "alice"
) -> Path:
    wrong = wrong or set()
    by_id = {case.task_id: case for case in manifest.cases}
    tasks = []
    for task_id in attempt.selected_task_ids:
        case = by_id[task_id]
        results = []
        for index, span in enumerate(case.gold):
            start = span.start + (1 if task_id in wrong else 0)
            results.append(
                {
                    "id": f"result-{task_id}-{index}",
                    "type": "labels",
                    "value": {
                        "start": start,
                        "end": span.end,
                        "text": case.text[start : span.end],
                        "labels": [span.label],
                    },
                }
            )
        tasks.append(
            {
                "id": task_id,
                "data": {"text": case.text, "task": case.task, "source_family": "onboarding"},
                "annotations": [{"id": f"annotation-{task_id}", "created_username": user, "result": results}],
            }
        )
    output = tmp_path / "export.json"
    output.write_text(json.dumps(tasks), encoding="utf-8")
    return output


def test_bank_and_tasks_are_versioned_and_answer_free(tmp_path):
    _gold, _config, manifest, _attempt = _context(tmp_path)
    tasks = build_onboarding_tasks(manifest)
    assert len(tasks) == 10
    assert all("gold" not in task["data"] and "gold_mentions" not in task["data"] for task in tasks)
    assert versioned_bank_path(tmp_path, manifest).name.endswith(f"{manifest.test_bank_hash}.json")


def test_selection_is_deterministic_and_retry_changes_selection(tmp_path):
    _gold, config, manifest, first = _context(tmp_path)
    second = start_attempt(tmp_path, manifest, config, "alice")
    assert len(first.selected_task_ids) == len(second.selected_task_ids) == 4
    assert set(first.selected_task_ids) != set(second.selected_task_ids)
    other = start_attempt(tmp_path, manifest, config, "bob")
    assert other.selected_task_ids != first.selected_task_ids


def test_empty_gold_and_empty_submission_is_correct(tmp_path):
    _gold, config, manifest, attempt = _context(tmp_path)
    export = _export_for_attempt(tmp_path, manifest, attempt)
    # Force one selected case to be the empty-gold case and submit an empty result.
    empty = next(case for case in manifest.cases if not case.gold)
    attempt.selected_task_ids[-1] = empty.task_id
    payload = json.loads(export.read_text(encoding="utf-8"))
    payload[-1] = {
        "id": empty.task_id,
        "data": {"text": empty.text, "task": empty.task, "source_family": "onboarding"},
        "annotations": [{"id": "empty", "created_username": "alice", "result": []}],
    }
    export.write_text(json.dumps(payload), encoding="utf-8")
    report = evaluate_attempt(export, tmp_path, manifest, config, attempt)
    assert report.status == "passed"
    assert report.correct_tasks == 4


def test_three_of_four_passes_and_wrong_boundary_fails(tmp_path):
    _gold, config, manifest, attempt = _context(tmp_path)
    wrong = {attempt.selected_task_ids[0]}
    export = _export_for_attempt(tmp_path, manifest, attempt, wrong=wrong)
    report = evaluate_attempt(export, tmp_path, manifest, config, attempt)
    assert report.status == "passed"
    assert report.correct_tasks == 3
    assert report.score == pytest.approx(0.75)

    nonempty = {case.task_id for case in manifest.cases if case.gold}
    wrong_two = [task_id for task_id in attempt.selected_task_ids if task_id in nonempty][:2]
    export = _export_for_attempt(tmp_path, manifest, attempt, wrong=set(wrong_two))
    report = evaluate_attempt(export, tmp_path, manifest, config, attempt)
    assert report.status == "failed"
    assert report.correct_tasks == 2


def test_wrong_label_fails_strict_task_correctness(tmp_path):
    _gold, config, manifest, attempt = _context(tmp_path)
    export = _export_for_attempt(tmp_path, manifest, attempt)
    payload = json.loads(export.read_text(encoding="utf-8"))
    index = next(index for index, task in enumerate(payload) if task["annotations"][0]["result"])
    labels = payload[index]["annotations"][0]["result"][0]["value"]["labels"]
    labels[0] = "phenotype" if labels[0] == "disease" else "disease"
    export.write_text(json.dumps(payload), encoding="utf-8")
    report = evaluate_attempt(export, tmp_path, manifest, config, attempt)
    assert report.correct_tasks == 3
    assert report.status == "passed"


def test_incomplete_attempt_is_not_a_pass(tmp_path):
    _gold, config, manifest, attempt = _context(tmp_path)
    export = _export_for_attempt(tmp_path, manifest, attempt)
    payload = json.loads(export.read_text(encoding="utf-8"))
    payload.pop()
    export.write_text(json.dumps(payload), encoding="utf-8")
    report = evaluate_attempt(export, tmp_path, manifest, config, attempt)
    assert report.status == "incomplete"
    assert report.score is None


def test_unknown_user_is_rejected(tmp_path):
    _gold, config, manifest, attempt = _context(tmp_path)
    export = _export_for_attempt(tmp_path, manifest, attempt, user="mallory")
    with pytest.raises(UnknownAnnotatorError):
        evaluate_attempt(export, tmp_path, manifest, config, attempt)


def test_duplicate_submissions_use_latest_annotation(tmp_path):
    _gold, config, manifest, attempt = _context(tmp_path)
    export = _export_for_attempt(tmp_path, manifest, attempt)
    payload = json.loads(export.read_text(encoding="utf-8"))
    nonempty_index = next(
        index
        for index, task in enumerate(payload)
        if task["id"] in attempt.selected_task_ids and task["annotations"][0]["result"]
    )
    payload[nonempty_index]["annotations"].append(
        {"id": "latest", "created_username": "alice", "updated_at": "9999", "result": []}
    )
    export.write_text(json.dumps(payload), encoding="utf-8")
    report = evaluate_attempt(export, tmp_path, manifest, config, attempt)
    assert report.status == "passed"
    assert report.correct_tasks == 3


def test_report_and_promotion_are_idempotent(tmp_path):
    _gold, config, manifest, attempt = _context(tmp_path)
    export = _export_for_attempt(tmp_path, manifest, attempt)
    report = evaluate_attempt(export, tmp_path, manifest, config, attempt)
    first_path = write_report(report, tmp_path)
    first_contents = first_path.read_text(encoding="utf-8")
    second_path = write_report(evaluate_attempt(export, tmp_path, manifest, config, attempt), tmp_path)
    assert first_path == second_path
    assert second_path.read_text(encoding="utf-8") == first_contents
    record = promote(tmp_path, report, manifest)
    assert promoted_users(tmp_path, manifest) == {"alice"}
    assert promote(tmp_path, report, manifest).promoted_at == record.promoted_at
    assert len(read_reports(tmp_path)) == 1


def test_changed_test_bank_version_cannot_score_old_attempt(tmp_path):
    _gold, config, manifest, attempt = _context(tmp_path)
    changed_gold = _benchmark(tmp_path / "changed.json", changed=True)
    changed = build_test_bank(changed_gold, config, generated_at="2026-01-02T00:00:00+00:00")
    export = _export_for_attempt(tmp_path, manifest, attempt)
    with pytest.raises(OnboardingError, match="different onboarding"):
        evaluate_attempt(export, tmp_path, changed, config, attempt)


def test_production_filter_preserves_excluded_audit(tmp_path):
    _gold, config, manifest, attempt = _context(tmp_path)
    report = evaluate_attempt(_export_for_attempt(tmp_path, manifest, attempt), tmp_path, manifest, config, attempt)
    promote(tmp_path, report, manifest)
    tasks = [
        {
            "id": "production-1",
            "data": {"text": "Used for asthma.", "task": "indication"},
            "annotations": [{"id": "x", "created_username": "mallory", "result": []}],
        },
        {
            "id": "production-2",
            "data": {"text": "Used for asthma.", "task": "indication"},
            "annotations": [
                {"id": "alice", "created_username": "alice", "result": []},
                {"id": "mallory", "created_username": "mallory", "result": []},
            ],
        },
        {"id": "production-3", "data": {"text": "Used for asthma.", "task": "indication"}, "annotations": []},
    ]
    source = tmp_path / "production.json"
    source.write_text(json.dumps(tasks), encoding="utf-8")
    eligible, audit, count = filter_production_export(source, tmp_path, manifest, {"alice"})
    assert count == 2
    eligible_tasks = json.loads(eligible.read_text(encoding="utf-8"))
    assert [item["id"] for item in eligible_tasks] == ["production-2"]
    assert [item["id"] for item in eligible_tasks[0]["annotations"]] == ["alice"]
    audit_payload = json.loads(audit.read_text(encoding="utf-8"))
    assert audit_payload["excluded_count"] == 2
    assert {item["reason"] for item in audit_payload["excluded"]} == {"annotator_not_promoted", "not_annotated"}
