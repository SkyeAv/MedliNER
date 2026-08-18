"""Minimal local Dagster asset graph for MEDliNER."""

from __future__ import annotations

import json
import os
from pathlib import Path

from dagster import AssetCheckResult, Definitions, asset, asset_check

from .dataset import hash_file, manifest_for, read_examples, write_examples, write_manifest
from .evaluation import evaluate_checkpoint
from .label_studio import normalize_export
from .packaging import build_export_bundle
from .splits import assert_no_group_leakage, split_examples
from .training import train_from_split_directory


def workdir() -> Path:
    return Path(os.environ.get("MEDLINER_WORKDIR", "data/materialized"))


@asset(description="Path to a reviewed Label Studio JSON/JSONL export supplied outside MEDliNER.")
def label_studio_export() -> str:
    value = os.environ.get("MEDLINER_LABEL_STUDIO_EXPORT")
    if not value:
        raise RuntimeError("set MEDLINER_LABEL_STUDIO_EXPORT to a reviewed Label Studio export")
    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(path)
    return str(path)


@asset(description="Validated canonical reviewed examples and dataset manifest.")
def normalized_dataset(context, label_studio_export: str) -> str:
    output = workdir() / "normalized" / "examples.jsonl"
    manifest_path = output.with_name("manifest.json")
    examples = normalize_export(label_studio_export, require_reviewed=True)
    write_examples(examples, output)
    manifest = manifest_for(examples, input_export_hash=hash_file(label_studio_export), dataset_id=hash_file(output))
    write_manifest(manifest, manifest_path)
    context.add_output_metadata({"path": str(output), "examples": len(examples), "dataset_hash": manifest.dataset_id})
    return str(output)


@asset(description="Frozen grouped train/validation/test JSONL files and split manifest.")
def frozen_splits(context, normalized_dataset: str) -> str:
    output_dir = workdir() / "splits"
    examples = read_examples(normalized_dataset)
    regression_path = os.environ.get("MEDLINER_REGRESSION_IDS")
    regression_ids = set(json.loads(Path(regression_path).read_text(encoding="utf-8"))) if regression_path else set()
    splits, manifest = split_examples(examples, regression_ids=regression_ids)
    assert_no_group_leakage(splits)
    if len({item.source.grouping_key for item in examples}) >= 3 and (not splits["validation"] or not splits["test"]):
        raise RuntimeError("at least three source groups require non-empty validation and test splits")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, members in splits.items():
        write_examples(members, output_dir / f"{name}.jsonl")
    (output_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    context.add_output_metadata(
        {"path": str(output_dir), "split_hash": manifest.split_hash, "examples": manifest.example_count}
    )
    return str(output_dir)


@asset(description="Small-GLiNER fine-tuning output/checkpoint directory.")
def training_run(context, frozen_splits: str) -> str:
    output = workdir() / "training"
    config_path = os.environ.get("MEDLINER_TRAIN_CONFIG", "configs/train-small.yaml")
    result = train_from_split_directory(frozen_splits, output, config_path=config_path)
    context.add_output_metadata({"path": str(result), "config": config_path})
    return str(result)


@asset(description="Strict/lenient evaluation report for the tuned checkpoint.")
def evaluation_report(context, training_run: str, frozen_splits: str) -> str:
    output = workdir() / "evaluation" / "report.json"
    result = evaluate_checkpoint(training_run, frozen_splits, output)
    context.add_output_metadata({"path": str(output), "strict_f1": result["tuned"]["overall"]["strict"]["f1"]})
    return str(output)


@asset(description="Self-contained standalone model artifact for later upload.")
def export_bundle(
    context, training_run: str, evaluation_report: str, normalized_dataset: str, frozen_splits: str
) -> str:
    output = workdir() / "bundle"
    result = build_export_bundle(
        checkpoint_dir=training_run,
        evaluation_path=evaluation_report,
        dataset_path=normalized_dataset,
        split_dir=frozen_splits,
        output_dir=output,
    )
    context.add_output_metadata({"path": str(result)})
    return str(result)


@asset_check(asset=normalized_dataset)
def normalized_dataset_is_nonempty(normalized_dataset: str) -> AssetCheckResult:
    examples = read_examples(normalized_dataset)
    return AssetCheckResult(passed=bool(examples), metadata={"examples": len(examples)})


@asset_check(asset=frozen_splits)
def frozen_splits_are_disjoint(frozen_splits: str) -> AssetCheckResult:
    directory = Path(frozen_splits)
    splits = {name: read_examples(directory / f"{name}.jsonl") for name in ("train", "validation", "test")}
    try:
        assert_no_group_leakage(splits)
    except AssertionError as exc:
        return AssetCheckResult(passed=False, metadata={"error": str(exc)})
    return AssetCheckResult(passed=True, metadata={"splits": {name: len(items) for name, items in splits.items()}})


def definitions() -> Definitions:
    return Definitions(
        assets=[label_studio_export, normalized_dataset, frozen_splits, training_run, evaluation_report, export_bundle],
        asset_checks=[normalized_dataset_is_nonempty, frozen_splits_are_disjoint],
    )


_defs = definitions()
