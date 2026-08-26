from __future__ import annotations

import json
from pathlib import Path

from medliner import cli

FIXTURE = Path(__file__).parent / "fixtures" / "dakp_export" / "ner_gold.json"

ACCOUNTS = ["alice", "bob", "medliner@localhost"]


def _fake_provision(monkeypatch, calls):
    def fake_provision(**kwargs):
        calls.append(kwargs)
        return {
            "url": "http://127.0.0.1:9030",
            "tasks_in_project": 10,
            "existing_tasks": 0,
            "usernames": list(ACCOUNTS),
        }

    monkeypatch.setattr(cli, "provision", fake_provision)


def _run_onboarding(tmp_path, monkeypatch) -> list[dict]:
    monkeypatch.setenv("MEDLINER_WORKDIR", str(tmp_path / "work"))
    monkeypatch.setenv("MEDLINER_BENCHMARK", str(FIXTURE))
    calls: list[dict] = []
    _fake_provision(monkeypatch, calls)
    assert cli.main(["onboarding"]) == 0
    return calls


def _correct_submissions(manifest, attempts: dict) -> list[dict]:
    """Build an Onboarding export with fully correct annotations per user."""
    from medliner.onboarding import OnboardingAttempt

    by_task_id = {case.task_id: case for case in manifest.cases}
    tasks_by_id: dict[str, dict] = {}
    for user, attempt in attempts.items():
        assert isinstance(attempt, OnboardingAttempt)
        for task_id in attempt.selected_task_ids:
            case = by_task_id[task_id]
            results = [
                {
                    "id": f"result-{user}-{task_id}-{index}",
                    "type": "labels",
                    "value": {"start": span.start, "end": span.end, "text": span.text, "labels": [span.label]},
                }
                for index, span in enumerate(case.gold)
            ]
            entry = tasks_by_id.setdefault(
                task_id,
                {
                    "id": task_id,
                    "data": {"text": case.text, "task": case.task, "source_family": "onboarding"},
                    "annotations": [],
                },
            )
            entry["annotations"].append(
                {"id": f"annotation-{user}-{task_id}", "created_username": user, "result": results}
            )
    return list(tasks_by_id.values())


def test_onboarding_cli_provisions_answer_free_project_and_assigns_everyone(tmp_path, monkeypatch, capsys):
    calls = _run_onboarding(tmp_path, monkeypatch)
    assert calls[0]["project_title"] == "Onboarding"
    tasks = json.loads(Path(calls[0]["import_file"]).read_text(encoding="utf-8"))
    assert len(tasks) == 10
    assert all("gold" not in task["data"] and "gold_mentions" not in task["data"] for task in tasks)
    output = capsys.readouterr().out
    assert "answer-free" in output
    # Every non-admin account got a quiz without anyone being named on the command line.
    attempts = {(item.username, item.attempt_number) for item in cli.read_attempts(tmp_path / "work")}
    assert ("alice", 1) in attempts
    assert ("bob", 1) in attempts
    assert not any(username == "medliner@localhost" for username, _ in attempts)


def test_onboarding_promote_scores_and_promotes_everyone_at_once(tmp_path, monkeypatch, capsys):
    workdir = tmp_path / "work"
    monkeypatch.setenv("MEDLINER_WORKDIR", str(workdir))
    _run_onboarding(tmp_path, monkeypatch)
    _config, manifest, _bank_path = cli._onboarding_context()
    attempts = {item.username: item for item in cli.read_attempts(workdir)}
    payload = _correct_submissions(manifest, attempts)
    export_path = workdir / "onboarding" / "export.json"
    export_path.parent.mkdir(parents=True, exist_ok=True)

    def fake_export_project(**kwargs):
        Path(kwargs["output_path"]).write_text(json.dumps(payload), encoding="utf-8")
        return {"tasks_annotated": len(payload), "tasks_exported": len(payload), "output": kwargs["output_path"]}

    monkeypatch.setattr(cli, "export_project", fake_export_project)
    assert cli.main(["onboarding-promote"]) == 0
    output = capsys.readouterr().out
    assert "promoted alice" in output
    assert "promoted bob" in output


def test_prepare_runs_candidates_and_prelabel(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDLINER_WORKDIR", str(tmp_path / "work"))
    import_file = tmp_path / "import.json"
    prelabeled = tmp_path / "prelabeled.json"
    seen: list[str] = []

    monkeypatch.setattr(cli, "raw_candidates_path", lambda value=None: tmp_path / "raw.ndjson")
    monkeypatch.setattr(cli, "run_candidates", lambda input_path: seen.append("candidates") or import_file)
    monkeypatch.setattr(cli, "run_prelabel", lambda *args, **kwargs: seen.append("prelabel") or prelabeled)
    assert cli.main(["prepare"]) == 0
    assert seen == ["candidates", "prelabel"]
