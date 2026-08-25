"""Standalone artifact bundle creation."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import yaml

from .dataset import hash_file
from .schema import ALLOWED_LABELS

BUNDLE_MARKER = "provenance.json"


def _tree_hash(path: Path) -> str:
    """Stream the checkpoint files; a GLiNER checkpoint is too large to slurp into memory."""
    digest = hashlib.sha256()
    for item in sorted(path.rglob("*")):
        if not item.is_file():
            continue
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _prepare_output_dir(output_dir: Path) -> None:
    """Replace a previous bundle, but never delete a directory this build did not create."""
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        return
    if not output_dir.is_dir():
        raise NotADirectoryError(output_dir)
    contents = list(output_dir.iterdir())
    if contents and not (output_dir / BUNDLE_MARKER).exists():
        raise FileExistsError(
            f"{output_dir} is not empty and has no {BUNDLE_MARKER}; refusing to delete it. "
            "Point --output-dir at a new or previously built bundle directory."
        )
    shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)


def _run_metadata(checkpoint_dir: Path) -> dict[str, Any]:
    path = checkpoint_dir / "medliner-training.json"
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _copy_synthetic_artifacts(
    run_metadata: dict[str, Any], synthetic_dir: Path | None, output_dir: Path
) -> dict[str, Any]:
    """Bundle the synthetic pool the run actually used and prove it is that exact pool.

    The checkpoint's run metadata is the source of truth: a count of zero (or an older gold-only
    run without the field) bundles nothing even when a stale pool still sits in the workdir,
    while a positive count without its artifacts fails loudly rather than shipping a provenance
    claim the bundle cannot back. The examples are re-hashed against the hash the trainer
    recorded, so a pool regenerated after training cannot pass itself off as the data the
    checkpoint learned from.
    """
    count = run_metadata.get("synthetic_examples")
    weight = run_metadata.get("synthetic_weight")
    if not count:
        return {"synthetic_weight": weight, "synthetic_count": count, "synthetic_manifest_sha256": None}
    if synthetic_dir is None:
        raise FileNotFoundError(
            f"run metadata records {count} synthetic training examples but no synthetic pool "
            "directory was given (default: $MEDLINER_WORKDIR/synthetic)"
        )
    examples = synthetic_dir / "examples.jsonl"
    manifest = synthetic_dir / "manifest.json"
    missing = [str(path) for path in (examples, manifest) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"run metadata records {count} synthetic training examples but the pool artifacts are missing: {missing}"
        )
    actual_hash = hash_file(examples)
    recorded_hash = run_metadata.get("synthetic_dataset_hash")
    if recorded_hash is not None and actual_hash != recorded_hash:
        raise ValueError(
            f"synthetic pool {examples} changed since training (sha256 {actual_hash} != recorded "
            f"{recorded_hash}); a bundle can only prove provenance for the exact pool the "
            "checkpoint learned from"
        )
    shutil.copy2(examples, output_dir / "synthetic_examples.jsonl")
    shutil.copy2(manifest, output_dir / "synthetic_manifest.json")
    return {
        "synthetic_weight": weight,
        "synthetic_count": count,
        "synthetic_manifest_sha256": hash_file(manifest),
    }


def _write_training_config(checkpoint_dir: Path, fallback: Path, destination: Path) -> None:
    """Prefer the config the run actually used over whatever is in `configs/` today.

    `MEDLINER_TRAIN_CONFIG` accepts any path, so copying the repository default would record a
    configuration the checkpoint may never have seen. The effective config is captured verbatim
    in the run metadata; fall back to the file only when that metadata is absent.
    """
    config = _run_metadata(checkpoint_dir).get("config")
    if isinstance(config, dict):
        destination.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
        return
    if fallback.exists():
        shutil.copy2(fallback, destination)


def build_export_bundle(
    *,
    checkpoint_dir: str | Path,
    evaluation_path: str | Path,
    dataset_path: str | Path,
    split_dir: str | Path,
    output_dir: str | Path,
    annotation_policy_path: str | Path = "docs/ANNOTATION_GUIDE.md",
    training_config_path: str | Path = "configs/train-small.yaml",
    synthetic_dir: str | Path | None = None,
) -> Path:
    """Copy immutable model/data metadata into a later-uploadable directory.

    A run that mixed in the synthetic pool also ships its evidence — ``synthetic_examples.jsonl``
    plus the synthesis manifest — and the provenance records the synthetic weight, count, and
    manifest hash. ``synthetic_dir`` defaults to ``$MEDLINER_WORKDIR/synthetic`` in the CLI; a
    run whose metadata records no synthetic examples bundles no synthetic artifacts.
    """
    checkpoint_dir = Path(checkpoint_dir)
    evaluation_path = Path(evaluation_path)
    dataset_path = Path(dataset_path)
    split_dir = Path(split_dir)
    output_dir = Path(output_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(checkpoint_dir)
    _prepare_output_dir(output_dir)
    shutil.copytree(checkpoint_dir, output_dir / "checkpoint")
    for source, destination in (
        (evaluation_path, output_dir / "metrics.json"),
        (dataset_path, output_dir / "dataset.jsonl"),
        (split_dir / "manifest.json", output_dir / "split_manifest.json"),
    ):
        if source.exists():
            shutil.copy2(source, destination)
    if Path(annotation_policy_path).exists():
        shutil.copy2(annotation_policy_path, output_dir / "annotation_policy.md")
    _write_training_config(checkpoint_dir, Path(training_config_path), output_dir / "training_config.yaml")
    (output_dir / "labels.json").write_text(
        json.dumps({"labels": list(ALLOWED_LABELS)}, indent=2) + "\n", encoding="utf-8"
    )
    run_metadata = _run_metadata(checkpoint_dir)
    synthetic_provenance = _copy_synthetic_artifacts(
        run_metadata, None if synthetic_dir is None else Path(synthetic_dir), output_dir
    )
    split_manifest_path = split_dir / "manifest.json"
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8")) if split_manifest_path.exists() else {}
    provenance: dict[str, Any] = {
        "artifact_schema": "medliner.bundle.v1",
        "labels": list(ALLOWED_LABELS),
        "base_model_id": run_metadata.get("model_id"),
        "selected_checkpoint": run_metadata.get("selected_checkpoint"),
        "best_validation_strict_f1": run_metadata.get("best_validation_strict_f1"),
        "checkpoint_tree_sha256": _tree_hash(output_dir / "checkpoint"),
        "dataset_sha256": hash_file(dataset_path) if dataset_path.exists() else None,
        "metrics_sha256": hash_file(evaluation_path) if evaluation_path.exists() else None,
        "split_hash": split_manifest.get("split_hash"),
        "held_out_example_ids": split_manifest.get("held_out_ids", []),
        **synthetic_provenance,
        "license_notes": (
            "Review source-data and base-checkpoint licenses before public upload."
            + (
                " Synthetic examples are paraphrases of the reviewed dataset and inherit its source licensing."
                if synthetic_provenance["synthetic_count"]
                else ""
            )
        ),
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "MODEL_CARD_INPUTS.md").write_text(
        "# MedliNER model-card inputs\n\n"
        f"- Labels: {', '.join(ALLOWED_LABELS)}\n"
        "- Tasks: indication and contraindication context NER\n"
        "- Base checkpoint and training parameters: see `checkpoint/medliner-training.json` and `training_config.yaml`.\n"
        "- Evaluation: see `metrics.json`.\n"
        "- Data policy and provenance: see `annotation_policy.md`, `dataset.jsonl`, and `provenance.json`.\n",
        encoding="utf-8",
    )
    return output_dir


__all__ = ["BUNDLE_MARKER", "build_export_bundle"]
