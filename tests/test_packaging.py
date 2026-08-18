from __future__ import annotations

import json

import pytest

from medliner.packaging import BUNDLE_MARKER, build_export_bundle


def _checkpoint(tmp_path):
    checkpoint = tmp_path / "final"
    checkpoint.mkdir()
    (checkpoint / "pytorch_model.bin").write_bytes(b"weights")
    (checkpoint / "medliner-training.json").write_text(json.dumps({"model_id": "m"}), encoding="utf-8")
    return checkpoint


def _inputs(tmp_path):
    dataset = tmp_path / "examples.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    metrics = tmp_path / "report.json"
    metrics.write_text("{}\n", encoding="utf-8")
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    (split_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
    return dataset, metrics, split_dir


def test_bundle_contains_the_uploadable_artifacts(tmp_path):
    dataset, metrics, split_dir = _inputs(tmp_path)
    output = build_export_bundle(
        checkpoint_dir=_checkpoint(tmp_path),
        evaluation_path=metrics,
        dataset_path=dataset,
        split_dir=split_dir,
        output_dir=tmp_path / "bundle",
    )
    names = {item.name for item in output.iterdir()}
    assert {"checkpoint", "labels.json", "metrics.json", "dataset.jsonl", "provenance.json"} <= names
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["checkpoint_tree_sha256"]
    assert provenance["dataset_sha256"]
    assert json.loads((output / "labels.json").read_text(encoding="utf-8"))["labels"] == [
        "disease",
        "phenotype",
        "drug",
    ]


def test_rebuilding_over_a_previous_bundle_is_allowed(tmp_path):
    dataset, metrics, split_dir = _inputs(tmp_path)
    kwargs = {
        "checkpoint_dir": _checkpoint(tmp_path),
        "evaluation_path": metrics,
        "dataset_path": dataset,
        "split_dir": split_dir,
        "output_dir": tmp_path / "bundle",
    }
    first = build_export_bundle(**kwargs)
    assert (first / BUNDLE_MARKER).exists()
    assert build_export_bundle(**kwargs).exists()


def test_a_populated_non_bundle_directory_is_never_deleted(tmp_path):
    dataset, metrics, split_dir = _inputs(tmp_path)
    output = tmp_path / "not-a-bundle"
    output.mkdir()
    (output / "important.txt").write_text("keep me", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to delete"):
        build_export_bundle(
            checkpoint_dir=_checkpoint(tmp_path),
            evaluation_path=metrics,
            dataset_path=dataset,
            split_dir=split_dir,
            output_dir=output,
        )
    assert (output / "important.txt").read_text(encoding="utf-8") == "keep me"


def test_missing_checkpoint_is_reported(tmp_path):
    dataset, metrics, split_dir = _inputs(tmp_path)
    with pytest.raises(FileNotFoundError):
        build_export_bundle(
            checkpoint_dir=tmp_path / "absent",
            evaluation_path=metrics,
            dataset_path=dataset,
            split_dir=split_dir,
            output_dir=tmp_path / "bundle",
        )


def test_bundle_records_the_configuration_the_run_actually_used(tmp_path):
    import yaml

    dataset, metrics, split_dir = _inputs(tmp_path)
    checkpoint = tmp_path / "final"
    checkpoint.mkdir()
    (checkpoint / "pytorch_model.bin").write_bytes(b"weights")
    (checkpoint / "medliner-training.json").write_text(
        json.dumps({"model_id": "m", "config": {"num_train_epochs": 3, "model_id": "m"}}), encoding="utf-8"
    )
    repository_default = tmp_path / "train-small.yaml"
    repository_default.write_text("num_train_epochs: 5\n", encoding="utf-8")

    output = build_export_bundle(
        checkpoint_dir=checkpoint,
        evaluation_path=metrics,
        dataset_path=dataset,
        split_dir=split_dir,
        output_dir=tmp_path / "bundle",
        training_config_path=repository_default,
    )
    recorded = yaml.safe_load((output / "training_config.yaml").read_text(encoding="utf-8"))
    assert recorded["num_train_epochs"] == 3


def test_bundle_falls_back_to_the_config_file_without_run_metadata(tmp_path):
    dataset, metrics, split_dir = _inputs(tmp_path)
    checkpoint = tmp_path / "final"
    checkpoint.mkdir()
    (checkpoint / "pytorch_model.bin").write_bytes(b"weights")
    repository_default = tmp_path / "train-small.yaml"
    repository_default.write_text("num_train_epochs: 5\n", encoding="utf-8")

    output = build_export_bundle(
        checkpoint_dir=checkpoint,
        evaluation_path=metrics,
        dataset_path=dataset,
        split_dir=split_dir,
        output_dir=tmp_path / "bundle",
        training_config_path=repository_default,
    )
    assert (output / "training_config.yaml").read_text(encoding="utf-8") == "num_train_epochs: 5\n"
