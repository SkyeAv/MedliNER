from __future__ import annotations

import json
from pathlib import Path

from medliner import cli

FIXTURE = Path(__file__).parent / "fixtures" / "dakp_export" / "ner_gold.json"


def test_onboarding_cli_provisions_answer_free_project(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MEDLINER_WORKDIR", str(tmp_path / "work"))
    monkeypatch.setenv("MEDLINER_BENCHMARK", str(FIXTURE))
    calls: list[dict] = []

    def fake_provision(**kwargs):
        calls.append(kwargs)
        return {"url": "http://127.0.0.1:9030", "tasks_in_project": 10}

    monkeypatch.setattr(cli, "provision", fake_provision)
    assert cli.main(["onboarding", "--annotator", "alice:pw"]) == 0
    assert calls[0]["project_title"] == "Onboarding"
    tasks = json.loads(Path(calls[0]["import_file"]).read_text(encoding="utf-8"))
    assert len(tasks) == 10
    assert all("gold" not in task["data"] and "gold_mentions" not in task["data"] for task in tasks)
    assert "answer-free" in capsys.readouterr().out


def test_onboarding_start_cli_records_four_tasks(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MEDLINER_WORKDIR", str(tmp_path / "work"))
    monkeypatch.setenv("MEDLINER_BENCHMARK", str(FIXTURE))
    assert cli.main(["onboarding-start", "--user", "alice"]) == 0
    output = capsys.readouterr().out
    assert "attempt 1" in output
    assert output.count("onboarding-") >= 4


def test_dataset_gate_requires_a_promotion(tmp_path, monkeypatch, capsys):
    export = tmp_path / "production.json"
    export.write_text("[]", encoding="utf-8")
    monkeypatch.setenv("MEDLINER_WORKDIR", str(tmp_path / "work"))
    monkeypatch.setenv("MEDLINER_BENCHMARK", str(FIXTURE))
    monkeypatch.setenv("MEDLINER_ONBOARDING_REQUIRED", "1")
    monkeypatch.setenv("MEDLINER_LABEL_STUDIO_EXPORT", str(export))
    assert cli.main(["dataset"]) == 1
    assert "no annotator has passed onboarding" in capsys.readouterr().err
