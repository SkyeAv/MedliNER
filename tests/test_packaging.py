from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from medliner.dataset import hash_file
from medliner.packaging import BUNDLE_MARKER, _tree_hash, build_export_bundle


def _checkpoint(tmp_path):
    checkpoint = tmp_path / "final"
    checkpoint.mkdir()
    # Mirrors a real `final/`: GLiNER.from_pretrained needs both the weights and gliner_config.json.
    (checkpoint / "pytorch_model.bin").write_bytes(b"weights")
    (checkpoint / "gliner_config.json").write_text(json.dumps({"max_len": 384}), encoding="utf-8")
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
    (checkpoint / "gliner_config.json").write_text(json.dumps({"max_len": 384}), encoding="utf-8")
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
    (checkpoint / "gliner_config.json").write_text(json.dumps({"max_len": 384}), encoding="utf-8")
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
    (checkpoint / "gliner_config.json").write_text(json.dumps({"max_len": 384}), encoding="utf-8")
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


def test_a_failed_build_leaves_the_previous_bundle_intact(tmp_path):
    """A mid-build failure must not destroy the last good bundle nor wedge the next retry."""
    dataset, metrics, split_dir = _inputs(tmp_path)
    output_dir = tmp_path / "bundle"
    checkpoint = _checkpoint(tmp_path)
    good = build_export_bundle(
        checkpoint_dir=checkpoint,
        evaluation_path=metrics,
        dataset_path=dataset,
        split_dir=split_dir,
        output_dir=output_dir,
    )
    before = sorted(item.name for item in good.iterdir())
    provenance_before = (good / "provenance.json").read_text(encoding="utf-8")

    # A run claiming synthetic examples with no pool directory fails partway through the build.
    broken_root = tmp_path / "broken"
    broken_root.mkdir()
    broken = _run_checkpoint(broken_root, {"model_id": "m", "synthetic_examples": 2})
    with pytest.raises(FileNotFoundError):
        build_export_bundle(
            checkpoint_dir=broken,
            evaluation_path=metrics,
            dataset_path=dataset,
            split_dir=split_dir,
            output_dir=output_dir,
            synthetic_dir=tmp_path / "missing-pool",
        )

    assert sorted(item.name for item in output_dir.iterdir()) == before
    assert (output_dir / "provenance.json").read_text(encoding="utf-8") == provenance_before
    # ...and the corrected rebuild still succeeds rather than hitting a "not empty" refusal.
    rebuilt = build_export_bundle(
        checkpoint_dir=checkpoint,
        evaluation_path=metrics,
        dataset_path=dataset,
        split_dir=split_dir,
        output_dir=output_dir,
    )
    assert (rebuilt / BUNDLE_MARKER).exists()


def test_a_checkpoint_inside_the_output_directory_is_never_deleted(tmp_path):
    """Bundling in place must fail loudly instead of rmtree-ing the trained weights."""
    dataset, metrics, split_dir = _inputs(tmp_path)
    output_dir = tmp_path / "bundle"
    output_dir.mkdir()
    checkpoint = _checkpoint(output_dir)

    with pytest.raises(ValueError, match="inside output_dir"):
        build_export_bundle(
            checkpoint_dir=checkpoint,
            evaluation_path=metrics,
            dataset_path=dataset,
            split_dir=split_dir,
            output_dir=output_dir,
        )
    assert (checkpoint / "pytorch_model.bin").read_bytes() == b"weights"


def test_a_weightless_checkpoint_is_refused(tmp_path):
    """An empty checkpoint directory would bundle successfully but load nowhere."""
    dataset, metrics, split_dir = _inputs(tmp_path)
    empty = tmp_path / "empty-final"
    empty.mkdir()

    with pytest.raises(FileNotFoundError, match="no model weights"):
        build_export_bundle(
            checkpoint_dir=empty,
            evaluation_path=metrics,
            dataset_path=dataset,
            split_dir=split_dir,
            output_dir=tmp_path / "bundle",
        )


def test_missing_evidence_files_are_refused_not_silently_skipped(tmp_path):
    """Metrics/dataset/split-manifest are the checkpoint's evidence; a bundle without them lies."""
    dataset, metrics, split_dir = _inputs(tmp_path)
    metrics.unlink()

    with pytest.raises(FileNotFoundError, match="evaluation report"):
        build_export_bundle(
            checkpoint_dir=_checkpoint(tmp_path),
            evaluation_path=metrics,
            dataset_path=dataset,
            split_dir=split_dir,
            output_dir=tmp_path / "bundle",
        )


def test_tree_hash_distinguishes_trees_that_concatenate_identically(tmp_path):
    """`{a: b"bc"}` and `{ab: b"c"}` share a byte stream; their digests must still differ."""
    first = tmp_path / "one"
    (first / "sub").mkdir(parents=True)
    (first / "a").write_bytes(b"bc")
    second = tmp_path / "two"
    second.mkdir()
    (second / "ab").write_bytes(b"c")

    assert _tree_hash(first) != _tree_hash(second)
    # An empty tree must not hash to the digest of the empty string.
    empty = tmp_path / "empty"
    empty.mkdir()
    assert _tree_hash(empty) != hashlib.sha256(b"").hexdigest()


def test_a_refused_rebuild_leaves_no_staging_tree_behind(tmp_path):
    """Refusing to overwrite a non-bundle directory must not leak a staged checkpoint copy."""
    dataset, metrics, split_dir = _inputs(tmp_path)
    output_dir = tmp_path / "bundle"
    output_dir.mkdir()
    (output_dir / "unrelated.txt").write_text("not a bundle", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to delete"):
        build_export_bundle(
            checkpoint_dir=_checkpoint(tmp_path),
            evaluation_path=metrics,
            dataset_path=dataset,
            split_dir=split_dir,
            output_dir=output_dir,
        )
    assert (output_dir / "unrelated.txt").exists()
    assert not list(tmp_path.glob(".bundle.staging-*"))
