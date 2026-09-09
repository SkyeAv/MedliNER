"""Standalone artifact bundle creation."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import yaml

from .dataset import hash_file
from .schema import ALLOWED_LABELS

BUNDLE_MARKER = "provenance.json"


# A GLiNER checkpoint is unloadable without weights and its config; `from_pretrained` needs both.
# Bundling a directory that merely *exists* ships an artifact nothing downstream can open.
_CHECKPOINT_WEIGHT_FILES = ("pytorch_model.bin", "model.safetensors")
_CHECKPOINT_CONFIG_FILE = "gliner_config.json"


def _tree_hash(path: Path) -> str:
    """Stream the checkpoint files; a GLiNER checkpoint is too large to slurp into memory.

    Each entry is framed with its path and content lengths. Concatenating the raw bytes would be
    ambiguous -- ``{a: b"bc"}`` and ``{ab: b"c"}`` produce an identical byte stream and therefore
    an identical digest, so two distinct checkpoints could share one provenance hash.
    """
    digest = hashlib.sha256()
    files = [item for item in sorted(path.rglob("*")) if item.is_file()]
    digest.update(f"files:{len(files)}\n".encode())
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(f"path:{len(relative)}:".encode())
        digest.update(relative)
        digest.update(f"size:{item.stat().st_size}:".encode())
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _assert_loadable_checkpoint(checkpoint_dir: Path) -> None:
    """Refuse to bundle a checkpoint directory that GLiNER could not load."""
    if not any((checkpoint_dir / name).is_file() for name in _CHECKPOINT_WEIGHT_FILES):
        raise FileNotFoundError(
            f"{checkpoint_dir} has no model weights (expected one of {', '.join(_CHECKPOINT_WEIGHT_FILES)}); "
            "it would produce a bundle no consumer can load"
        )
    if not (checkpoint_dir / _CHECKPOINT_CONFIG_FILE).is_file():
        raise FileNotFoundError(
            f"{checkpoint_dir} has no {_CHECKPOINT_CONFIG_FILE}; GLiNER.from_pretrained would fail on this bundle"
        )


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
    checkpoint_dir = Path(checkpoint_dir).resolve()
    evaluation_path = Path(evaluation_path)
    dataset_path = Path(dataset_path)
    split_dir = Path(split_dir)
    output_dir = Path(output_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(checkpoint_dir)
    _assert_loadable_checkpoint(checkpoint_dir)
    # Rebuilding deletes `output_dir`. An input living inside it would be destroyed before it is
    # read -- with `checkpoint_dir` nested, that is irrecoverable loss of the trained weights.
    resolved_output = output_dir.resolve()
    for name, candidate in (
        ("checkpoint_dir", checkpoint_dir),
        ("evaluation_path", evaluation_path.resolve()),
        ("dataset_path", dataset_path.resolve()),
        ("split_dir", split_dir.resolve()),
    ):
        if candidate == resolved_output or resolved_output in candidate.parents:
            raise ValueError(
                f"{name} ({candidate}) is inside output_dir ({resolved_output}); building the bundle "
                "would delete it. Point the output at a directory outside the inputs."
            )
    if checkpoint_dir == resolved_output or checkpoint_dir in resolved_output.parents:
        # `staging` is created inside `checkpoint_dir` here, so `copytree` would recurse into it.
        raise ValueError(
            f"output_dir ({resolved_output}) is inside checkpoint_dir ({checkpoint_dir}); the bundle "
            "cannot be built inside its own checkpoint. Point the output elsewhere."
        )
    # Replaceability is checked before the expensive build: a refusal (a populated directory this
    # build did not create) must not leave a staged copy of the checkpoint behind.
    _assert_output_dir_replaceable(output_dir)
    # Stage beside the destination and swap in only once complete. A failure during the build
    # leaves the previous good bundle intact; only in the tiny window between deleting the old
    # bundle and renaming the staged one could a crash lose it (the staged copy still survives
    # for manual recovery).
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{resolved_output.name}.staging-", dir=resolved_output.parent))
    try:
        _build_bundle_contents(
            checkpoint_dir=checkpoint_dir,
            evaluation_path=evaluation_path,
            dataset_path=dataset_path,
            split_dir=split_dir,
            staging=staging,
            annotation_policy_path=Path(annotation_policy_path),
            training_config_path=Path(training_config_path),
            synthetic_dir=None if synthetic_dir is None else Path(synthetic_dir),
        )
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    # `_assert_output_dir_replaceable` already vetted the destination; `_prepare_output_dir`
    # clears it and recreates it empty so the swap below has a removable directory to replace.
    _prepare_output_dir(output_dir)
    output_dir.rmdir()
    staging.replace(output_dir)
    return output_dir


def _assert_output_dir_replaceable(output_dir: Path) -> None:
    """Refuse, before any build work, a destination the final swap would be unable to delete."""
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(output_dir)
    if output_dir.is_dir() and list(output_dir.iterdir()) and not (output_dir / BUNDLE_MARKER).exists():
        raise FileExistsError(
            f"{output_dir} is not empty and has no {BUNDLE_MARKER}; refusing to delete it. "
            "Point the output at a new or previously built bundle directory."
        )


def _build_bundle_contents(
    *,
    checkpoint_dir: Path,
    evaluation_path: Path,
    dataset_path: Path,
    split_dir: Path,
    staging: Path,
    annotation_policy_path: Path,
    training_config_path: Path,
    synthetic_dir: Path | None,
) -> None:
    """Materialize the whole bundle into `staging`; raises before the destination is touched."""
    output_dir = staging
    shutil.copytree(checkpoint_dir, output_dir / "checkpoint")
    for name, source, destination in (
        ("evaluation report", evaluation_path, output_dir / "metrics.json"),
        ("dataset", dataset_path, output_dir / "dataset.jsonl"),
        ("split manifest", split_dir / "manifest.json", output_dir / "split_manifest.json"),
    ):
        # A bundle without its metrics, dataset, or split manifest carries no evidence for the
        # checkpoint it ships; skipping them silently produced an unfalsifiable artifact.
        if not source.exists():
            raise FileNotFoundError(f"{name} not found at {source}; the bundle would ship without it")
        shutil.copy2(source, destination)
    if annotation_policy_path.exists():
        shutil.copy2(annotation_policy_path, output_dir / "annotation_policy.md")
    _write_training_config(checkpoint_dir, training_config_path, output_dir / "training_config.yaml")
    (output_dir / "labels.json").write_text(
        json.dumps({"labels": list(ALLOWED_LABELS)}, indent=2) + "\n", encoding="utf-8"
    )
    run_metadata = _run_metadata(checkpoint_dir)
    synthetic_provenance = _copy_synthetic_artifacts(run_metadata, synthetic_dir, output_dir)
    split_manifest_path = split_dir / "manifest.json"
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    provenance: dict[str, Any] = {
        "artifact_schema": "medliner.bundle.v1",
        "labels": list(ALLOWED_LABELS),
        "base_model_id": run_metadata.get("model_id"),
        "selected_checkpoint": run_metadata.get("selected_checkpoint"),
        "best_validation_strict_f1": run_metadata.get("best_validation_strict_f1"),
        "checkpoint_tree_sha256": _tree_hash(output_dir / "checkpoint"),
        "dataset_sha256": hash_file(dataset_path),
        "metrics_sha256": hash_file(evaluation_path),
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
        "- Labels: disease, phenotype\n"
        "- Tasks: indication and contraindication context NER\n"
        "- Base checkpoint and training parameters: see `checkpoint/medliner-training.json` and `training_config.yaml`.\n"
        "- Evaluation: see `metrics.json`.\n"
        "- Data policy and provenance: see `annotation_policy.md`, `dataset.jsonl`, and `provenance.json`.\n",
        encoding="utf-8",
    )


__all__ = ["BUNDLE_MARKER", "build_export_bundle"]
