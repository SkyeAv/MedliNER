from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("dagster")

from medliner.dagster_defs import definitions, frozen_splits_are_disjoint, normalized_dataset_is_nonempty  # noqa: E402
from medliner.dataset import read_examples, write_examples  # noqa: E402
from medliner.schema import Annotation, Example  # noqa: E402


def test_definitions_expose_minimal_asset_graph():
    defs = definitions()
    keys = {key.to_user_string() for key in defs.resolve_all_asset_keys()}
    assert {
        "raw_candidate_texts",
        "candidate_tasks",
        "label_studio_server",
        "label_studio_export",
        "normalized_dataset",
        "frozen_splits",
        "training_run",
        "evaluation_report",
        "export_bundle",
    } <= keys


def test_graph_has_no_schedules_or_sensors():
    defs = definitions()
    assert not (getattr(defs, "schedules", None) or [])
    assert not (getattr(defs, "sensors", None) or [])


def _example(identifier: str, document: str) -> Example:
    return Example(
        id=identifier,
        text="asthma",
        task="indication",
        source={"family": "dailymed", "document_id": document},
        annotations=[Annotation(start=0, end=6, label="disease", text="asthma")],
    )


def test_nonempty_check_fails_on_an_empty_dataset(tmp_path):
    path = tmp_path / "examples.jsonl"
    write_examples([], path)
    assert normalized_dataset_is_nonempty(str(path)).passed is False
    write_examples([_example("a", "doc-a")], path)
    assert normalized_dataset_is_nonempty(str(path)).passed is True


def test_disjointness_check_detects_a_leaked_source_group(tmp_path):
    for name in ("train", "validation", "test"):
        write_examples([_example(f"{name}-1", "doc-shared")], tmp_path / f"{name}.jsonl")
    result = frozen_splits_are_disjoint(str(tmp_path))
    assert result.passed is False
    assert "doc-shared" in result.metadata["error"].value


def _write_reviewed_export(path: Path, count: int) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "id": f"t{index}",
                    "data": {
                        "text": "Contraindicated in patients with asthma.",
                        "task": "contraindication",
                        "source_family": "dailymed",
                        "source_document_id": f"doc-{index}",
                    },
                    "annotations": [
                        {
                            "id": index,
                            "created_username": "annotator",
                            "result": [
                                {
                                    "id": f"r{index}",
                                    "type": "labels",
                                    "from_name": "label",
                                    "to_name": "text",
                                    "value": {"start": 33, "end": 39, "text": "asthma", "labels": ["disease"]},
                                }
                            ],
                        }
                    ],
                }
                for index in range(count)
            ]
        ),
        encoding="utf-8",
    )


def test_assets_materialize_a_normalized_dataset_and_frozen_splits(tmp_path, monkeypatch):
    from dagster import build_asset_context

    from medliner.dagster_defs import frozen_splits, label_studio_export, normalized_dataset

    export = tmp_path / "export.json"
    _write_reviewed_export(export, 6)
    monkeypatch.setenv("MEDLINER_LABEL_STUDIO_EXPORT", str(export))
    monkeypatch.setenv("MEDLINER_WORKDIR", str(tmp_path / "work"))
    monkeypatch.delenv("MEDLINER_REGRESSION_IDS", raising=False)

    assert label_studio_export() == str(export)

    dataset_path = normalized_dataset(build_asset_context(), str(export))
    assert len(read_examples(dataset_path)) == 6

    split_dir = frozen_splits(build_asset_context(), dataset_path)
    assert frozen_splits_are_disjoint(split_dir).passed is True
    assert normalized_dataset_is_nonempty(dataset_path).passed is True
    manifest = json.loads((Path(split_dir) / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["example_count"] == 6


def test_frozen_splits_honor_regression_ids_from_environment(tmp_path, monkeypatch):
    from dagster import build_asset_context

    from medliner.dagster_defs import frozen_splits, normalized_dataset

    export = tmp_path / "export.json"
    _write_reviewed_export(export, 6)
    regression = tmp_path / "regression.json"
    regression.write_text(json.dumps(["t0"]), encoding="utf-8")
    monkeypatch.setenv("MEDLINER_LABEL_STUDIO_EXPORT", str(export))
    monkeypatch.setenv("MEDLINER_WORKDIR", str(tmp_path / "work"))
    monkeypatch.setenv("MEDLINER_REGRESSION_IDS", str(regression))

    dataset_path = normalized_dataset(build_asset_context(), str(export))
    split_dir = frozen_splits(build_asset_context(), dataset_path)
    manifest = json.loads((Path(split_dir) / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["held_out_ids"] == ["t0"]
    assert "t0" not in {item for ids in manifest["example_ids"].values() for item in ids}


def test_training_run_exposes_a_smoke_run_config():
    from medliner.dagster_defs import TrainingRunConfig, training_run

    assert TrainingRunConfig().smoke is False
    assert "smoke" in training_run.op.config_schema.config_type.fields


def test_candidate_assets_materialize_an_import_file(tmp_path, monkeypatch):
    from dagster import build_asset_context

    from medliner.dagster_defs import candidate_tasks, candidate_tasks_are_nonempty, raw_candidate_texts

    raw = tmp_path / "candidates.jsonl"
    raw.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"text": "Indicated for asthma.", "task": "indication", "source_family": "dailymed"},
                {"text": "Indicated for asthma.", "task": "indication", "source_family": "dailymed"},
                {"text": "Contraindicated in asthma.", "task": "contraindication", "source_family": "faers"},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MEDLINER_RAW_CANDIDATES", str(raw))
    monkeypatch.setenv("MEDLINER_WORKDIR", str(tmp_path / "work"))

    assert raw_candidate_texts() == str(raw)
    import_path = candidate_tasks(build_asset_context(), str(raw))
    tasks = json.loads(Path(import_path).read_text(encoding="utf-8"))
    assert len(tasks) == 2  # the duplicate indication text is merged
    assert candidate_tasks_are_nonempty(import_path).passed is True
    manifest = json.loads((Path(import_path).with_suffix(".manifest.json")).read_text(encoding="utf-8"))
    assert manifest["task_counts"] == {"contraindication": 1, "indication": 1}
    assert manifest["duplicates_merged"] == 1


def test_raw_candidates_missing_file_is_an_explicit_error(monkeypatch, tmp_path):
    from medliner.dagster_defs import raw_candidate_texts

    monkeypatch.setenv("MEDLINER_RAW_CANDIDATES", str(tmp_path / "absent.jsonl"))
    with pytest.raises(FileNotFoundError, match="MEDLINER_RAW_CANDIDATES"):
        raw_candidate_texts()


def test_label_studio_server_exposes_a_reimport_run_config():
    from medliner.dagster_defs import LabelStudioServerConfig, label_studio_server

    assert LabelStudioServerConfig().reimport is False
    assert "reimport" in label_studio_server.op.config_schema.config_type.fields


def test_missing_export_environment_is_an_explicit_error(monkeypatch):
    from medliner.dagster_defs import label_studio_export

    monkeypatch.delenv("MEDLINER_LABEL_STUDIO_EXPORT", raising=False)
    with pytest.raises(RuntimeError, match="MEDLINER_LABEL_STUDIO_EXPORT"):
        label_studio_export()


def test_export_path_must_exist(monkeypatch, tmp_path):
    from medliner.dagster_defs import label_studio_export

    monkeypatch.setenv("MEDLINER_LABEL_STUDIO_EXPORT", str(tmp_path / "absent.json"))
    with pytest.raises(FileNotFoundError):
        label_studio_export()
