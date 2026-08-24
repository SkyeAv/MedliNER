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
) -> Path:
    """Copy immutable model/data metadata into a later-uploadable directory."""
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
        "license_notes": "Review source-data and base-checkpoint licenses before public upload.",
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "MODEL_CARD_INPUTS.md").write_text(
        "# MedliNER model-card inputs\n\n"
        "- Labels: disease, phenotype\n"
        "- Tasks: indication and contraindication context NER\n"
        "- Base checkpoint and training parameters: see `checkpoint/medliner-training.json` and `training_config.yaml`.\n"
        "- Evaluation: see `metrics.json`.\n"
        "- Data policy and provenance: see `annotation_policy.md`, `dataset.jsonl`, and `provenance.json`.\n",
        encoding="utf-8",
    )
    return output_dir


__all__ = ["BUNDLE_MARKER", "build_export_bundle"]
