"""Minimal local Dagster asset graph for MEDliNER."""

# No `from __future__ import annotations` here: this Dagster version cannot resolve
# stringified annotations on pythonic Config classes (TrainingRunConfig) at decoration time.

import json
import os
from pathlib import Path

from dagster import AssetCheckResult, Config, Definitions, asset, asset_check

from .candidates import build_import_tasks, import_manifest, read_candidates, write_import_file
from .dataset import hash_file, manifest_for, read_examples, write_examples, write_manifest
from .evaluation import evaluate_checkpoint
from .label_studio import normalize_export
from .label_studio_server import DEFAULT_IMAGE, DEFAULT_PORT, provision
from .packaging import build_export_bundle
from .splits import assert_no_group_leakage, group_key, split_examples
from .training import train_from_split_directory


def workdir() -> Path:
    return Path(os.environ.get("MEDLINER_WORKDIR", "data/materialized"))


def repo_root() -> Path:
    """Dagster's working directory is not the repository, so anchor committed configs."""
    return Path(__file__).resolve().parents[2]


@asset(
    description="Path to the user-authored raw candidates JSONL (texts derived from "
    "intermediate DAKP inputs; see docs/CANDIDATE_TASKS.md)."
)
def raw_candidate_texts() -> str:
    value = os.environ.get("MEDLINER_RAW_CANDIDATES", "data/label-studio/candidates.jsonl")
    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(f"raw candidates file not found: {path} (MEDLINER_RAW_CANDIDATES)")
    return str(path)


@asset(description="Validated, deduplicated plain-text Label Studio import file and manifest.")
def candidate_tasks(context, raw_candidate_texts: str) -> str:
    tasks = build_import_tasks(read_candidates(raw_candidate_texts))
    manifest = import_manifest(tasks, input_path=raw_candidate_texts)
    output = workdir() / "label-studio" / f"import-{manifest['input_hash'][:16]}.json"
    write_import_file(tasks, output)
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    context.add_output_metadata(
        {
            "path": str(output),
            "tasks": manifest["task_count"],
            "duplicates_merged": manifest["duplicates_merged"],
            "task_counts": manifest["task_counts"],
            "family_counts": manifest["family_counts"],
        }
    )
    return str(output)


class LabelStudioServerConfig(Config):
    """Materialize label_studio_server with {'reimport': true} to replace existing project tasks."""

    reimport: bool = False
    port: int = DEFAULT_PORT
    image: str = DEFAULT_IMAGE


@asset(
    description="Podman-hosted Label Studio server with the candidate tasks imported. "
    "Annotate in the browser, export JSON manually, then materialize label_studio_export."
)
def label_studio_server(context, config: LabelStudioServerConfig, candidate_tasks: str) -> str:
    result = provision(
        import_file=candidate_tasks,
        label_config_path=repo_root() / "configs" / "label_studio_ner.xml",
        port=config.port,
        image=config.image,
        data_dir=workdir() / "label-studio" / "server-data",
        username=os.environ.get("MEDLINER_LABEL_STUDIO_USERNAME", "medliner@localhost"),
        password=os.environ.get("MEDLINER_LABEL_STUDIO_PASSWORD", "medliner-local"),
        token=os.environ.get("MEDLINER_LABEL_STUDIO_TOKEN") or None,
        reimport=config.reimport,
    )
    context.add_output_metadata(result)
    return str(result["url"])


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
    splits, manifest = split_examples(
        examples, seed=int(os.environ.get("MEDLINER_SPLIT_SEED", "2026")), regression_ids=regression_ids
    )
    assert_no_group_leakage(splits)
    # Must be the splitter's own grouping, or this guard measures a different partition.
    if len({group_key(item) for item in examples}) >= 3 and (not splits["validation"] or not splits["test"]):
        raise RuntimeError("at least three source groups require non-empty validation and test splits")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, members in splits.items():
        write_examples(members, output_dir / f"{name}.jsonl")
    (output_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    context.add_output_metadata(
        {
            "path": str(output_dir),
            "split_hash": manifest.split_hash,
            "examples": manifest.example_count,
            "held_out": len(manifest.held_out_ids),
        }
    )
    return str(output_dir)


class TrainingRunConfig(Config):
    """Set smoke=true in the launchpad for the required first one-step GPU check."""

    smoke: bool = False


@asset(
    description="Small-GLiNER fine-tuning output/checkpoint directory. "
    "Materialize with {'smoke': true} run config for the one-step GPU sanity check."
)
def training_run(context, config: TrainingRunConfig, frozen_splits: str) -> str:
    output = workdir() / "training"
    config_path = os.environ.get("MEDLINER_TRAIN_CONFIG", "configs/train-small.yaml")
    result = train_from_split_directory(frozen_splits, output, config_path=config_path, smoke_test=config.smoke)
    context.add_output_metadata({"path": str(result), "config": config_path, "smoke": config.smoke})
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


@asset_check(asset=candidate_tasks)
def candidate_tasks_are_nonempty(candidate_tasks: str) -> AssetCheckResult:
    tasks = json.loads(Path(candidate_tasks).read_text(encoding="utf-8"))
    return AssetCheckResult(passed=bool(tasks), metadata={"tasks": len(tasks)})


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
        assets=[
            raw_candidate_texts,
            candidate_tasks,
            label_studio_server,
            label_studio_export,
            normalized_dataset,
            frozen_splits,
            training_run,
            evaluation_report,
            export_bundle,
        ],
        asset_checks=[candidate_tasks_are_nonempty, normalized_dataset_is_nonempty, frozen_splits_are_disjoint],
    )


_defs = definitions()
