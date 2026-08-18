"""Standalone artifact bundle creation."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from .dataset import hash_file
from .schema import ALLOWED_LABELS


def _tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(path.rglob("*")):
        if item.is_file():
            digest.update(item.relative_to(path).as_posix().encode("utf-8"))
            digest.update(item.read_bytes())
    return digest.hexdigest()


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
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
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
    if Path(training_config_path).exists():
        shutil.copy2(training_config_path, output_dir / "training_config.yaml")
    (output_dir / "labels.json").write_text(
        json.dumps({"labels": list(ALLOWED_LABELS)}, indent=2) + "\n", encoding="utf-8"
    )
    provenance: dict[str, Any] = {
        "artifact_schema": "medliner.bundle.v1",
        "checkpoint_tree_sha256": _tree_hash(output_dir / "checkpoint"),
        "dataset_sha256": hash_file(dataset_path) if dataset_path.exists() else None,
        "metrics_sha256": hash_file(evaluation_path) if evaluation_path.exists() else None,
        "license_notes": "Review source-data and base-checkpoint licenses before public upload.",
        "dakp_runtime_integration": "deferred",
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "MODEL_CARD_INPUTS.md").write_text(
        "# MEDliNER model-card inputs\n\n"
        "- Labels: disease, phenotype, drug\n"
        "- Tasks: indication and contraindication context NER\n"
        "- Base checkpoint and training parameters: see `checkpoint/medliner-training.json` and `training_config.yaml`.\n"
        "- Evaluation: see `metrics.json`.\n"
        "- Data policy and provenance: see `annotation_policy.md`, `dataset.jsonl`, and `provenance.json`.\n",
        encoding="utf-8",
    )
    return output_dir


__all__ = ["build_export_bundle"]
