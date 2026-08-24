from __future__ import annotations

import json
from pathlib import Path

from medliner import cli
from medliner.dataset import read_examples, write_examples
from medliner.schema import Annotation, Example

DAKP_EXPORT_FIXTURE = Path(__file__).parent / "fixtures" / "dakp_export"


def _example(identifier: str, document: str) -> Example:
    return Example(
        id=identifier,
        text="asthma",
        task="indication",
        source={"family": "dailymed", "document_id": document},
        annotations=[Annotation(start=0, end=6, label="disease", text="asthma")],
    )


def _write_raw_candidates(path: Path) -> None:
    path.write_text(
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


def test_ingest_materializes_the_dakp_bundle(tmp_path, monkeypatch, capsys):
    """The CLI stage turns a verified DAKP bundle into the ingested working set."""
    monkeypatch.setenv("MEDLINER_WORKDIR", str(tmp_path / "work"))
    assert cli.main(["ingest", "--bundle", str(DAKP_EXPORT_FIXTURE)]) == 0
    ingested = tmp_path / "work" / "ingested"
    assert (ingested / "candidates.ndjson").exists()
    assert (ingested / "ner_gold.json").exists()
    assert (ingested / "ingest-manifest.json").exists()
    out = capsys.readouterr().out
    assert "8 candidates" in out  # the committed fixture manifest records 8 rows
    assert str(ingested) in out
    # The bundle directory also resolves from the environment when the flag is absent.
    monkeypatch.setenv("MEDLINER_EXPORT_BUNDLE", str(DAKP_EXPORT_FIXTURE))
    assert cli.main(["ingest"]) == 0


def test_ingest_missing_bundle_is_an_explicit_error(monkeypatch, capsys):
    """Neither flag nor env var set must name both in the error, like the other stages."""
    monkeypatch.delenv("MEDLINER_EXPORT_BUNDLE", raising=False)
    assert cli.main(["ingest"]) == 1
    error = capsys.readouterr().err
    assert "--bundle" in error
    assert "MEDLINER_EXPORT_BUNDLE" in error


def test_candidates_builds_an_import_file(tmp_path, monkeypatch, capsys):
    raw = tmp_path / "candidates.ndjson"
    _write_raw_candidates(raw)
    monkeypatch.setenv("MEDLINER_RAW_CANDIDATES", str(raw))
    monkeypatch.setenv("MEDLINER_WORKDIR", str(tmp_path / "work"))

    assert cli.main(["candidates"]) == 0
    import_path = Path(capsys.readouterr().out.split("->")[-1].strip())
    tasks = json.loads(import_path.read_text(encoding="utf-8"))
    assert len(tasks) == 2  # the duplicate indication text is merged
    manifest = json.loads(import_path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert manifest["task_counts"] == {"contraindication": 1, "indication": 1}
    assert manifest["duplicates_merged"] == 1


def test_candidates_missing_input_is_an_explicit_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MEDLINER_RAW_CANDIDATES", str(tmp_path / "absent.jsonl"))
    assert cli.main(["candidates"]) == 1
    assert "MEDLINER_RAW_CANDIDATES" in capsys.readouterr().err


def test_candidates_empty_input_fails(tmp_path, monkeypatch, capsys):
    raw = tmp_path / "candidates.ndjson"
    raw.write_text("", encoding="utf-8")
    monkeypatch.setenv("MEDLINER_RAW_CANDIDATES", str(raw))
    monkeypatch.setenv("MEDLINER_WORKDIR", str(tmp_path / "work"))
    assert cli.main(["candidates"]) == 1
    assert "no import tasks" in capsys.readouterr().err


def test_label_studio_provisions_with_the_import_file(tmp_path, monkeypatch, capsys):
    raw = tmp_path / "candidates.ndjson"
    _write_raw_candidates(raw)
    monkeypatch.setenv("MEDLINER_RAW_CANDIDATES", str(raw))
    monkeypatch.setenv("MEDLINER_WORKDIR", str(tmp_path / "work"))
    calls = {}

    def fake_provision(**kwargs):
        calls.update(kwargs)
        return {"url": "http://127.0.0.1:9030", "container": "medliner-label-studio", "tasks_in_project": 2}

    monkeypatch.setattr(cli, "provision", fake_provision)
    assert cli.main(["label-studio", "--input", str(raw), "--reimport"]) == 0
    assert Path(calls["import_file"]).name.startswith("import-")
    assert calls["reimport"] is True
    assert "http://127.0.0.1:9030" in capsys.readouterr().out


def test_label_studio_annotator_flag_parses_pairs(tmp_path, monkeypatch, capsys):
    raw = tmp_path / "candidates.ndjson"
    _write_raw_candidates(raw)
    monkeypatch.setenv("MEDLINER_RAW_CANDIDATES", str(raw))
    monkeypatch.setenv("MEDLINER_WORKDIR", str(tmp_path / "work"))
    calls = {}

    def fake_provision(**kwargs):
        calls.update(kwargs)
        return {"url": "http://127.0.0.1:9030", "container": "c", "tasks_in_project": 2, "annotators_created": 1}

    monkeypatch.setattr(cli, "provision", fake_provision)
    assert cli.main(["label-studio", "--input", str(raw), "--annotator", "alice:pw-a"]) == 0
    assert calls["annotators"] == [("alice", "pw-a")]
    out = capsys.readouterr().out
    assert "1 annotator account(s)" in out


def test_label_studio_annotator_env_and_validation(tmp_path, monkeypatch, capsys):
    raw = tmp_path / "candidates.ndjson"
    _write_raw_candidates(raw)
    monkeypatch.setenv("MEDLINER_RAW_CANDIDATES", str(raw))
    monkeypatch.setenv("MEDLINER_WORKDIR", str(tmp_path / "work"))
    monkeypatch.setenv("MEDLINER_LABEL_STUDIO_ANNOTATORS", "alice:pw-a,bob:pw-b")
    calls = {}

    def fake_provision(**kwargs):
        calls.update(kwargs)
        return {"url": "http://127.0.0.1:9030", "container": "c", "tasks_in_project": 2, "annotators_created": 0}

    monkeypatch.setattr(cli, "provision", fake_provision)
    assert cli.main(["label-studio", "--input", str(raw)]) == 0
    assert calls["annotators"] == [("alice", "pw-a"), ("bob", "pw-b")]

    # A pair without a separator is rejected loudly before anything is provisioned.
    monkeypatch.setenv("MEDLINER_LABEL_STUDIO_ANNOTATORS", "alicepw")
    assert cli.main(["label-studio", "--input", str(raw)]) == 1
    assert "username:password" in capsys.readouterr().err


def test_label_studio_warmup_provisions_the_separate_project(tmp_path, monkeypatch, capsys):
    raw = tmp_path / "candidates.ndjson"
    _write_raw_candidates(raw)
    monkeypatch.setenv("MEDLINER_RAW_CANDIDATES", str(raw))
    monkeypatch.setenv("MEDLINER_WORKDIR", str(tmp_path / "work"))
    monkeypatch.setenv("MEDLINER_BENCHMARK", str(DAKP_EXPORT_FIXTURE / "ner_gold.json"))
    calls = []

    def fake_provision(**kwargs):
        calls.append(kwargs)
        return {"url": "http://127.0.0.1:9030", "container": "c", "tasks_in_project": 2, "annotators_created": 0}

    monkeypatch.setattr(cli, "provision", fake_provision)
    assert cli.main(["label-studio", "--input", str(raw), "--warmup", "--warmup-limit", "3"]) == 0
    assert len(calls) == 2
    assert calls[0]["project_title"] == "MedliNER medical NER"
    assert calls[1]["project_title"] == "MedliNER medical NER — Warm-up"
    warmup_tasks = json.loads(Path(calls[1]["import_file"]).read_text(encoding="utf-8"))
    assert len(warmup_tasks) == 3
    assert all(task["data"]["source_family"] == "gold-warmup" for task in warmup_tasks)
    assert all(task["data"]["gold_mentions"] is not None for task in warmup_tasks)
    assert "warm-up tasks" in capsys.readouterr().out


def test_label_studio_warmup_requires_the_gold_benchmark(tmp_path, monkeypatch, capsys):
    raw = tmp_path / "candidates.ndjson"
    _write_raw_candidates(raw)
    monkeypatch.setenv("MEDLINER_RAW_CANDIDATES", str(raw))
    monkeypatch.setenv("MEDLINER_WORKDIR", str(tmp_path / "work"))
    monkeypatch.setenv("MEDLINER_BENCHMARK", str(tmp_path / "absent-gold.json"))
    monkeypatch.setattr(
        cli, "provision", lambda **k: {"url": "u", "container": "c", "tasks_in_project": 0, "annotators_created": 0}
    )
    assert cli.main(["label-studio", "--input", str(raw), "--warmup"]) == 1
    assert "MEDLINER_BENCHMARK" in capsys.readouterr().err


def test_label_studio_export_downloads_via_the_api(tmp_path, monkeypatch, capsys):
    output = tmp_path / "reviewed.json"
    monkeypatch.setenv("MEDLINER_LABEL_STUDIO_EXPORT", str(output))
    calls = {}

    def fake_export(**kwargs):
        calls.update(kwargs)
        return {
            "url": "http://127.0.0.1:9030",
            "project_id": 1,
            "tasks_exported": 4,
            "tasks_annotated": 3,
            "output": output,
        }

    monkeypatch.setattr(cli, "export_project", fake_export)
    assert cli.main(["label-studio-export"]) == 0
    assert Path(calls["output_path"]) == output
    out = capsys.readouterr().out
    assert "3/4 annotated" in out

    # An explicit --output wins over the environment.
    override = tmp_path / "elsewhere.json"
    assert cli.main(["label-studio-export", "--output", str(override)]) == 0
    assert Path(calls["output_path"]) == override


def test_label_studio_export_requires_a_destination(monkeypatch, capsys):
    monkeypatch.delenv("MEDLINER_LABEL_STUDIO_EXPORT", raising=False)
    assert cli.main(["label-studio-export"]) == 1
    error = capsys.readouterr().err
    assert "--output" in error and "MEDLINER_LABEL_STUDIO_EXPORT" in error


def test_label_studio_stop_reports_container_state(monkeypatch, capsys):
    monkeypatch.setattr(cli, "stop_container", lambda: True)
    assert cli.main(["label-studio-stop"]) == 0
    assert "removed" in capsys.readouterr().out


def test_dataset_and_splits_materialize(tmp_path, monkeypatch):
    export = tmp_path / "export.json"
    _write_reviewed_export(export, 6)
    monkeypatch.setenv("MEDLINER_LABEL_STUDIO_EXPORT", str(export))
    monkeypatch.setenv("MEDLINER_WORKDIR", str(tmp_path / "work"))
    monkeypatch.delenv("MEDLINER_REGRESSION_IDS", raising=False)

    assert cli.main(["dataset"]) == 0
    dataset_path = tmp_path / "work" / "normalized" / "examples.jsonl"
    assert len(read_examples(dataset_path)) == 6

    assert cli.main(["splits"]) == 0
    manifest = json.loads((tmp_path / "work" / "splits" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["example_count"] == 6


def test_splits_honor_regression_ids_from_environment(tmp_path, monkeypatch):
    export = tmp_path / "export.json"
    _write_reviewed_export(export, 6)
    regression = tmp_path / "regression.json"
    regression.write_text(json.dumps(["t0"]), encoding="utf-8")
    monkeypatch.setenv("MEDLINER_LABEL_STUDIO_EXPORT", str(export))
    monkeypatch.setenv("MEDLINER_WORKDIR", str(tmp_path / "work"))
    monkeypatch.setenv("MEDLINER_REGRESSION_IDS", str(regression))

    assert cli.main(["dataset"]) == 0
    assert cli.main(["splits"]) == 0
    manifest = json.loads((tmp_path / "work" / "splits" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["held_out_ids"] == ["t0"]
    assert "t0" not in {item for ids in manifest["example_ids"].values() for item in ids}


def test_splits_refuse_a_leaked_partition(tmp_path, monkeypatch, capsys):
    dataset_path = tmp_path / "examples.jsonl"
    write_examples([_example("a", "doc-a")], dataset_path)
    monkeypatch.setenv("MEDLINER_WORKDIR", str(tmp_path / "work"))
    leaked = {"train": [_example("t", "doc-shared")], "validation": [_example("v", "doc-shared")], "test": []}
    monkeypatch.setattr(cli, "split_examples", lambda *a, **k: (leaked, None))
    assert cli.main(["splits", "--dataset", str(dataset_path)]) == 1
    assert "doc-shared" in capsys.readouterr().err


def test_missing_export_environment_is_an_explicit_error(monkeypatch, capsys):
    monkeypatch.delenv("MEDLINER_LABEL_STUDIO_EXPORT", raising=False)
    assert cli.main(["dataset"]) == 1
    assert "MEDLINER_LABEL_STUDIO_EXPORT" in capsys.readouterr().err


def test_export_path_must_exist(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MEDLINER_LABEL_STUDIO_EXPORT", str(tmp_path / "absent.json"))
    assert cli.main(["dataset"]) == 1


def test_empty_dataset_fails(tmp_path, monkeypatch, capsys):
    export = tmp_path / "export.json"
    export.write_text("[]", encoding="utf-8")
    monkeypatch.setenv("MEDLINER_LABEL_STUDIO_EXPORT", str(export))
    monkeypatch.setenv("MEDLINER_WORKDIR", str(tmp_path / "work"))
    assert cli.main(["dataset"]) == 1
    assert "empty" in capsys.readouterr().err


def test_train_smoke_flag_reaches_training(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDLINER_WORKDIR", str(tmp_path / "work"))
    calls = {}

    def fake_train(split_dir, output_dir, *, config_path, smoke_test):
        calls["smoke_test"] = smoke_test
        return Path(output_dir) / "final"

    monkeypatch.setattr("medliner.training.train_from_split_directory", fake_train)
    assert cli.main(["train", "--smoke"]) == 0
    assert calls["smoke_test"] is True
    assert cli.main(["train"]) == 0
    assert calls["smoke_test"] is False


def _fake_gliner(monkeypatch, entities_for):
    """Replace model loading with a callable, so CLI tests never touch torch or a GPU."""
    from medliner import prelabel

    loaded: list[str] = []

    def fake_load_model(model_id, *, device=None):
        loaded.append(model_id)
        return object(), device or "cpu"

    def fake_batch_predictor(model, *, threshold, labels=prelabel.PRELABEL_LABELS, batch_size=8):
        return lambda texts: [entities_for(text) for text in texts]

    monkeypatch.setattr(prelabel, "load_model", fake_load_model)
    monkeypatch.setattr(prelabel, "batch_predictor", fake_batch_predictor)
    return loaded


def _asthma_entities(text):
    if "asthma" not in text:
        return []
    start = text.index("asthma")
    return [{"start": start, "end": start + 6, "label": "disease", "score": 0.91}]


def test_prelabel_writes_predictions_and_a_manifest(tmp_path, monkeypatch, capsys):
    raw = tmp_path / "candidates.ndjson"
    _write_raw_candidates(raw)
    monkeypatch.setenv("MEDLINER_RAW_CANDIDATES", str(raw))
    monkeypatch.setenv("MEDLINER_WORKDIR", str(tmp_path / "work"))
    _fake_gliner(monkeypatch, _asthma_entities)

    assert cli.main(["prelabel", "--input", str(raw)]) == 0

    (output,) = sorted((tmp_path / "work" / "label-studio").glob("import-*.prelabeled.json"))
    tasks = json.loads(output.read_text(encoding="utf-8"))
    assert tasks
    for task in tasks:
        (prediction,) = task["predictions"]
        assert prediction["model_version"].endswith("@0.35")
        (region,) = prediction["result"]
        assert region["value"]["labels"] == ["disease"]
        assert region["value"]["text"] == "asthma"
    manifest = json.loads(output.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "medliner.prelabel.manifest.v1"
    assert manifest["label_counts"] == {"disease": 2}
    assert manifest["labels"] == ["disease", "phenotype"]
    assert "2 suggestions" in capsys.readouterr().out


def test_prelabel_reruns_from_cache_without_loading_the_model(tmp_path, monkeypatch, capsys):
    raw = tmp_path / "candidates.ndjson"
    _write_raw_candidates(raw)
    monkeypatch.setenv("MEDLINER_RAW_CANDIDATES", str(raw))
    monkeypatch.setenv("MEDLINER_WORKDIR", str(tmp_path / "work"))
    loaded = _fake_gliner(monkeypatch, _asthma_entities)

    assert cli.main(["prelabel", "--input", str(raw)]) == 0
    assert loaded == ["gliner-community/gliner_large-v2.5"]
    capsys.readouterr()

    assert cli.main(["prelabel", "--input", str(raw)]) == 0
    assert loaded == ["gliner-community/gliner_large-v2.5"]  # second run never loaded it again
    assert "2 cached, 0 predicted" in capsys.readouterr().out


def test_prelabel_force_ignores_the_cache(tmp_path, monkeypatch, capsys):
    raw = tmp_path / "candidates.ndjson"
    _write_raw_candidates(raw)
    monkeypatch.setenv("MEDLINER_RAW_CANDIDATES", str(raw))
    monkeypatch.setenv("MEDLINER_WORKDIR", str(tmp_path / "work"))
    loaded = _fake_gliner(monkeypatch, _asthma_entities)

    assert cli.main(["prelabel", "--input", str(raw)]) == 0
    assert cli.main(["prelabel", "--input", str(raw), "--force"]) == 0
    assert len(loaded) == 2
    assert "0 cached, 2 predicted" in capsys.readouterr().out


def test_label_studio_prelabel_imports_the_prelabeled_file_and_turns_on_prefill(tmp_path, monkeypatch, capsys):
    raw = tmp_path / "candidates.ndjson"
    _write_raw_candidates(raw)
    monkeypatch.setenv("MEDLINER_RAW_CANDIDATES", str(raw))
    monkeypatch.setenv("MEDLINER_WORKDIR", str(tmp_path / "work"))
    _fake_gliner(monkeypatch, _asthma_entities)
    calls = {}

    def fake_provision(**kwargs):
        calls.update(kwargs)
        return {"url": "http://127.0.0.1:9030", "container": "c", "tasks_in_project": 2, "prelabeled": True}

    monkeypatch.setattr(cli, "provision", fake_provision)
    assert cli.main(["label-studio", "--input", str(raw), "--prelabel"]) == 0
    assert Path(calls["import_file"]).name.endswith(".prelabeled.json")
    assert calls["prelabel_model_version"] == "gliner_large-v2.5@0.35"
    assert "accept, correct, or delete every span" in capsys.readouterr().out


def test_label_studio_without_prelabel_imports_plain_tasks(tmp_path, monkeypatch, capsys):
    raw = tmp_path / "candidates.ndjson"
    _write_raw_candidates(raw)
    monkeypatch.setenv("MEDLINER_RAW_CANDIDATES", str(raw))
    monkeypatch.setenv("MEDLINER_WORKDIR", str(tmp_path / "work"))
    calls = {}

    def fake_provision(**kwargs):
        calls.update(kwargs)
        return {"url": "http://127.0.0.1:9030", "container": "c", "tasks_in_project": 2}

    monkeypatch.setattr(cli, "provision", fake_provision)
    assert cli.main(["label-studio", "--input", str(raw)]) == 0
    assert not Path(calls["import_file"]).name.endswith(".prelabeled.json")
    assert calls["prelabel_model_version"] is None
    capsys.readouterr()
