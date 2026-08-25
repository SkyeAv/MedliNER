from __future__ import annotations

import json
from pathlib import Path

import pytest

from medliner.dataset import hash_file
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
        "DiseaseOrPhenotypicFeature",
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


def _bundle_kwargs(tmp_path, checkpoint, **overrides):
    dataset, metrics, split_dir = _inputs(tmp_path)
    return {
        "checkpoint_dir": checkpoint,
        "evaluation_path": metrics,
        "dataset_path": dataset,
        "split_dir": split_dir,
        "output_dir": tmp_path / "bundle",
        **overrides,
    }


def _run_checkpoint(tmp_path, metadata: dict) -> Path:
    """Checkpoint carrying the given `medliner-training.json` run metadata."""
    checkpoint = tmp_path / "final"
    checkpoint.mkdir()
    (checkpoint / "pytorch_model.bin").write_bytes(b"weights")
    (checkpoint / "medliner-training.json").write_text(json.dumps(metadata), encoding="utf-8")
    return checkpoint


def _pool(tmp_path) -> Path:
    """Minimal synthetic pool: accepted examples plus the synthesis manifest."""
    pool = tmp_path / "synthetic"
    pool.mkdir()
    (pool / "examples.jsonl").write_text('{"id": "gold-a-synth-paraphrase"}\n', encoding="utf-8")
    (pool / "manifest.json").write_text('{"schema_version": "medliner.synthesis.manifest.v1"}\n', encoding="utf-8")
    return pool


def test_bundle_includes_the_synthetic_pool_the_run_used(tmp_path):
    # A semi-supervised run must ship its evidence: the synthetic examples and the manifest that
    # gated them travel with the bundle, and provenance records weight/count/manifest hash so
    # the mix is auditable without the workdir.
    pool = _pool(tmp_path)
    checkpoint = _run_checkpoint(
        tmp_path,
        {
            "model_id": "m",
            "synthetic_examples": 1,
            "synthetic_weight": 0.1,
            "synthetic_dataset_hash": hash_file(pool / "examples.jsonl"),
        },
    )

    output = build_export_bundle(**_bundle_kwargs(tmp_path, checkpoint, synthetic_dir=pool))

    assert (output / "synthetic_examples.jsonl").read_text(encoding="utf-8") == '{"id": "gold-a-synth-paraphrase"}\n'
    assert (output / "synthetic_manifest.json").exists()
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["synthetic_weight"] == 0.1
    assert provenance["synthetic_count"] == 1
    assert provenance["synthetic_manifest_sha256"] == hash_file(pool / "manifest.json")
    assert "inherit its source licensing" in provenance["license_notes"]


def test_a_synthetic_run_without_a_pool_directory_is_reported(tmp_path):
    # Run metadata claiming synthetic examples with no pool given cannot produce a provenance-
    # backed bundle; fail loudly instead of recording a count the bundle cannot evidence.
    checkpoint = _run_checkpoint(tmp_path, {"model_id": "m", "synthetic_examples": 2, "synthetic_weight": 0.1})
    with pytest.raises(FileNotFoundError, match="no synthetic pool"):
        build_export_bundle(**_bundle_kwargs(tmp_path, checkpoint))


def test_a_synthetic_run_with_missing_pool_artifacts_is_reported(tmp_path):
    # The manifest is the audit trail of the divergence gates; a pool directory without it (or
    # without the examples) names exactly what is missing instead of bundling half the evidence.
    pool = _pool(tmp_path)
    (pool / "manifest.json").unlink()  # the gate manifest is the missing artifact
    checkpoint = _run_checkpoint(tmp_path, {"model_id": "m", "synthetic_examples": 2, "synthetic_weight": 0.1})
    with pytest.raises(FileNotFoundError, match="manifest.json"):
        build_export_bundle(**_bundle_kwargs(tmp_path, checkpoint, synthetic_dir=pool))


def test_a_regenerated_pool_cannot_impersonate_the_trained_one(tmp_path):
    # The trainer records the pool hash it learned from; a pool regenerated afterwards (for
    # example with a different gate setting) must not ship as the pool the checkpoint used.
    pool = _pool(tmp_path)
    checkpoint = _run_checkpoint(
        tmp_path,
        {"model_id": "m", "synthetic_examples": 1, "synthetic_weight": 0.1, "synthetic_dataset_hash": "deadbeef"},
    )
    with pytest.raises(ValueError, match="changed since training"):
        build_export_bundle(**_bundle_kwargs(tmp_path, checkpoint, synthetic_dir=pool))


def test_a_gold_only_run_bundles_no_synthetic_artifacts_even_with_a_stale_pool(tmp_path):
    # The bundle documents what the run used, not what sits in the workdir: a gold-only run
    # (no synthetic fields, or a zero count after --no-synthetic) excludes a stale pool and
    # records nulls instead of claiming a mix that never happened.
    pool = _pool(tmp_path)
    output = build_export_bundle(**_bundle_kwargs(tmp_path, _checkpoint(tmp_path), synthetic_dir=pool))
    assert not (output / "synthetic_examples.jsonl").exists()
    assert not (output / "synthetic_manifest.json").exists()
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["synthetic_weight"] is None
    assert provenance["synthetic_count"] is None
    assert provenance["synthetic_manifest_sha256"] is None
    assert "inherit its source licensing" not in provenance["license_notes"]


def test_a_no_synthetic_run_records_zero_not_a_stale_pool(tmp_path):
    # --no-synthetic runs keep the configured weight in metadata with a zero count; provenance
    # mirrors that honestly (weight set, nothing used) and still bundles no pool artifacts.
    pool = _pool(tmp_path)
    checkpoint = _run_checkpoint(tmp_path, {"model_id": "m", "synthetic_examples": 0, "synthetic_weight": 0.1})
    output = build_export_bundle(**_bundle_kwargs(tmp_path, checkpoint, synthetic_dir=pool))
    assert not (output / "synthetic_examples.jsonl").exists()
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert (provenance["synthetic_weight"], provenance["synthetic_count"]) == (0.1, 0)
    assert provenance["synthetic_manifest_sha256"] is None


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
