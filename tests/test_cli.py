from __future__ import annotations

import json

import pytest

from medliner.cli import main
from medliner.dataset import read_examples
from medliner.schema import SplitManifest

TEXT = "Contraindicated in patients with asthma."


def _task(task_id: str, document: str) -> dict:
    start = TEXT.index("asthma")
    return {
        "id": task_id,
        "data": {
            "text": TEXT,
            "task": "contraindication",
            "source_family": "dailymed",
            "source_document_id": document,
        },
        "annotations": [
            {
                "id": 1,
                "created_username": "annotator",
                "result": [
                    {
                        "id": f"{task_id}-r1",
                        "type": "labels",
                        "from_name": "label",
                        "to_name": "text",
                        "value": {"start": start, "end": start + 6, "text": "asthma", "labels": ["disease"]},
                    }
                ],
            }
        ],
    }


def test_normalize_then_split_round_trip(tmp_path, capsys):
    export = tmp_path / "export.json"
    export.write_text(json.dumps([_task(f"t{index}", f"doc-{index}") for index in range(5)]), encoding="utf-8")
    dataset = tmp_path / "normalized" / "examples.jsonl"

    assert main(["normalize", str(export), str(dataset)]) == 0
    assert len(read_examples(dataset)) == 5
    assert json.loads((dataset.with_name("manifest.json")).read_text(encoding="utf-8"))["example_count"] == 5

    splits = tmp_path / "splits"
    assert main(["split", str(dataset), str(splits), "--seed", "7"]) == 0
    manifest = SplitManifest.model_validate_json((splits / "manifest.json").read_text(encoding="utf-8"))
    assert manifest.seed == 7
    assert sum(len(ids) for ids in manifest.example_ids.values()) == 5
    assert all((splits / f"{name}.jsonl").exists() for name in ("train", "validation", "test"))
    assert "split_hash" in capsys.readouterr().out


def test_split_withholds_regression_ids(tmp_path):
    export = tmp_path / "export.json"
    export.write_text(json.dumps([_task(f"t{index}", f"doc-{index}") for index in range(5)]), encoding="utf-8")
    dataset = tmp_path / "examples.jsonl"
    main(["normalize", str(export), str(dataset)])

    regression = tmp_path / "regression.json"
    regression.write_text(json.dumps(["t0"]), encoding="utf-8")
    splits = tmp_path / "splits"
    assert main(["split", str(dataset), str(splits), "--regression-ids", str(regression)]) == 0

    manifest = SplitManifest.model_validate_json((splits / "manifest.json").read_text(encoding="utf-8"))
    assert manifest.held_out_ids == ["t0"]
    assert "t0" not in {item for ids in manifest.example_ids.values() for item in ids}


def test_unreviewed_tasks_are_blocked_by_default(tmp_path):
    task = _task("t0", "doc-0")
    task["data"]["annotation_status"] = "draft"
    export = tmp_path / "export.json"
    export.write_text(json.dumps([task]), encoding="utf-8")
    with pytest.raises(Exception, match="unreviewed tasks cannot enter training"):
        main(["normalize", str(export), str(tmp_path / "examples.jsonl")])


def test_unknown_command_exits_non_zero():
    with pytest.raises(SystemExit):
        main(["nope"])


def test_evaluate_and_bundle_subcommands_wire_through(tmp_path, monkeypatch, capsys):
    from medliner import cli

    recorded = {}

    def fake_evaluate(checkpoint, split_dir, output, *, include_baselines, threshold):
        recorded.update(
            checkpoint=checkpoint, split_dir=split_dir, output=output, baselines=include_baselines, threshold=threshold
        )
        return {"evaluated_split": "test", "tuned": {"overall": {"strict": {"f1": 0.75}}}}

    monkeypatch.setattr(cli, "evaluate_checkpoint", fake_evaluate)
    assert (
        cli.main(
            [
                "evaluate",
                "ckpt",
                "splits",
                str(tmp_path / "report.json"),
                "--threshold",
                "0.5",
                "--no-baselines",
            ]
        )
        == 0
    )
    assert recorded["threshold"] == 0.5
    assert recorded["baselines"] is False
    assert "0.7500" in capsys.readouterr().out

    monkeypatch.setattr(cli, "build_export_bundle", lambda **kwargs: tmp_path / "bundle")
    assert cli.main(["bundle", "ckpt", "metrics", "dataset", "splits", str(tmp_path / "bundle")]) == 0


def test_train_subcommand_passes_smoke_and_resume(tmp_path, monkeypatch):
    from medliner import cli

    recorded = {}

    def fake_train(split_dir, output_dir, *, config_path, resume_from_checkpoint, smoke_test):
        recorded.update(config=config_path, resume=resume_from_checkpoint, smoke=smoke_test)
        return tmp_path / "final"

    monkeypatch.setattr(cli, "train_from_split_directory", fake_train)
    assert (
        cli.main(
            ["train", "splits", "out", "--smoke", "--config", "cfg.yaml", "--resume-from-checkpoint", "checkpoint-3"]
        )
        == 0
    )
    assert recorded == {"config": "cfg.yaml", "resume": "checkpoint-3", "smoke": True}


def test_allow_unreviewed_imports_draft_tasks(tmp_path):
    task = _task("t0", "doc-0")
    task["data"]["annotation_status"] = "draft"
    export = tmp_path / "export.json"
    export.write_text(json.dumps([task]), encoding="utf-8")
    dataset = tmp_path / "examples.jsonl"
    assert main(["normalize", str(export), str(dataset), "--allow-unreviewed"]) == 0
    assert read_examples(dataset)[0].annotation_status.value == "draft"
