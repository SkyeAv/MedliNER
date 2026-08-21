"""Command-line interface for the MEDliNER pipeline stages.

The pipeline runs in three phases, each a subcommand (see the Makefile wrappers):

- before Label Studio: ``candidates`` builds the validated import file;
- Label Studio: ``label-studio``/``label-studio-stop`` manage the podman annotation server,
  annotation and export itself remain manual browser steps;
- after Label Studio: ``dataset`` → ``splits`` → ``train`` → ``evaluate`` → ``bundle``
  (or all five via ``pipeline``).

Configuration comes from the ``MEDLINER_*`` environment variables, with flags overriding
where offered. Heavy ML imports (training, evaluation) are deferred to their subcommands so
the data-stage commands stay stdlib-light.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .candidates import build_import_tasks, hash_candidates_file, import_manifest, read_candidates, write_import_file
from .dataset import hash_file, manifest_for, read_examples, write_examples, write_manifest
from .label_studio import normalize_export
from .label_studio_server import DEFAULT_IMAGE, DEFAULT_PORT, provision, stop_container
from .packaging import build_export_bundle
from .splits import assert_no_group_leakage, group_key, split_examples


def workdir() -> Path:
    return Path(os.environ.get("MEDLINER_WORKDIR", "data/materialized"))


def repo_root() -> Path:
    """Anchor committed configs regardless of the caller's working directory."""
    return Path(__file__).resolve().parents[2]


def raw_candidates_path(value: str | None = None) -> Path:
    raw = value or os.environ.get("MEDLINER_RAW_CANDIDATES", "data/label-studio/candidates.jsonl")
    path = Path(raw)
    if not path.exists():
        raise FileNotFoundError(f"raw candidates file not found: {path} (MEDLINER_RAW_CANDIDATES)")
    return path


def export_path(value: str | None = None) -> Path:
    raw = value or os.environ.get("MEDLINER_LABEL_STUDIO_EXPORT")
    if not raw:
        raise RuntimeError("set MEDLINER_LABEL_STUDIO_EXPORT to a reviewed Label Studio export")
    path = Path(raw)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def run_candidates(input_path: Path) -> Path:
    """Validate/dedupe raw candidates into the Label Studio import file; returns its path."""
    tasks = build_import_tasks(read_candidates(input_path))
    if not tasks:
        raise ValueError(f"no import tasks produced from {input_path}")
    manifest = import_manifest(tasks, input_path=input_path)
    output = workdir() / "label-studio" / f"import-{manifest['input_hash'][:16]}.json"
    write_import_file(tasks, output)
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"candidates: {manifest['task_count']} tasks ({manifest['duplicates_merged']} duplicates merged) -> {output}")
    return output


def ensure_import_file(input_path: Path) -> Path:
    """Return the import file for the current input hash, building it when absent."""
    expected = workdir() / "label-studio" / f"import-{hash_candidates_file(input_path)[:16]}.json"
    return expected if expected.exists() else run_candidates(input_path)


def run_dataset(path: Path) -> Path:
    """Validate the reviewed export into the normalized dataset; returns the JSONL path."""
    examples = normalize_export(path, require_reviewed=True)
    if not examples:
        raise ValueError(f"normalized dataset from {path} is empty")
    output = workdir() / "normalized" / "examples.jsonl"
    write_examples(examples, output)
    manifest = manifest_for(examples, input_export_hash=hash_file(path), dataset_id=hash_file(output))
    write_manifest(manifest, output.with_name("manifest.json"))
    print(f"dataset: {len(examples)} examples -> {output}")
    return output


def run_splits(dataset_path: Path) -> Path:
    """Freeze grouped train/validation/test splits; returns the split directory."""
    examples = read_examples(dataset_path)
    regression_path = os.environ.get("MEDLINER_REGRESSION_IDS")
    regression_ids = set(json.loads(Path(regression_path).read_text(encoding="utf-8"))) if regression_path else set()
    splits, manifest = split_examples(
        examples, seed=int(os.environ.get("MEDLINER_SPLIT_SEED", "2026")), regression_ids=regression_ids
    )
    try:
        assert_no_group_leakage(splits)
    except AssertionError as exc:
        raise RuntimeError(str(exc)) from exc
    # Must be the splitter's own grouping, or this guard measures a different partition.
    if len({group_key(item) for item in examples}) >= 3 and (not splits["validation"] or not splits["test"]):
        raise RuntimeError("at least three source groups require non-empty validation and test splits")
    output_dir = workdir() / "splits"
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, members in splits.items():
        write_examples(members, output_dir / f"{name}.jsonl")
    (output_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(f"splits: {manifest.example_count} examples ({len(manifest.held_out_ids)} held out) -> {output_dir}")
    return output_dir


def run_training(smoke: bool) -> Path:
    """Fine-tune the small GLiNER checkpoint (or the one-step smoke check); returns the checkpoint dir."""
    from .training import train_from_split_directory

    output = workdir() / "training"
    config_path = os.environ.get("MEDLINER_TRAIN_CONFIG", "configs/train-small.yaml")
    result = train_from_split_directory(workdir() / "splits", output, config_path=config_path, smoke_test=smoke)
    print(f"training ({'smoke' if smoke else 'full'}): checkpoint -> {result}")
    return result


def run_evaluation(checkpoint: Path, split_dir: Path) -> Path:
    """Score the tuned checkpoint against baselines; returns the report path."""
    from .evaluation import evaluate_checkpoint

    output = workdir() / "evaluation" / "report.json"
    result = evaluate_checkpoint(checkpoint, split_dir, output)
    print(f"evaluation: strict F1 {result['tuned']['overall']['strict']['f1']:.3f} -> {output}")
    return output


def run_bundle(checkpoint: Path, report: Path, dataset_path: Path, split_dir: Path) -> Path:
    """Assemble the standalone export bundle; returns its directory."""
    result = build_export_bundle(
        checkpoint_dir=checkpoint,
        evaluation_path=report,
        dataset_path=dataset_path,
        split_dir=split_dir,
        output_dir=workdir() / "bundle",
        annotation_policy_path=repo_root() / "docs" / "ANNOTATION_GUIDE.md",
        training_config_path=os.environ.get("MEDLINER_TRAIN_CONFIG", "configs/train-small.yaml"),
    )
    print(f"bundle: {result}")
    return result


def cmd_candidates(args: argparse.Namespace) -> None:
    run_candidates(raw_candidates_path(args.input))


def cmd_label_studio(args: argparse.Namespace) -> None:
    import_file = ensure_import_file(raw_candidates_path(args.input))
    port = args.port or int(os.environ.get("MEDLINER_LABEL_STUDIO_PORT", str(DEFAULT_PORT)))
    image = args.image or os.environ.get("MEDLINER_LABEL_STUDIO_IMAGE", DEFAULT_IMAGE)
    result = provision(
        import_file=import_file,
        label_config_path=repo_root() / "configs" / "label_studio_ner.xml",
        port=port,
        image=image,
        data_dir=workdir() / "label-studio" / "server-data",
        username=os.environ.get("MEDLINER_LABEL_STUDIO_USERNAME", "medliner@localhost"),
        password=os.environ.get("MEDLINER_LABEL_STUDIO_PASSWORD", "medliner-local"),
        token=os.environ.get("MEDLINER_LABEL_STUDIO_TOKEN") or None,
        reimport=args.reimport,
    )
    print(f"label-studio: {result['tasks_in_project']} tasks at {result['url']} (container {result['container']})")


def cmd_label_studio_stop(_args: argparse.Namespace) -> None:
    removed = stop_container()
    print("label-studio: container removed" if removed else "label-studio: no container to remove")


def cmd_dataset(args: argparse.Namespace) -> None:
    run_dataset(export_path(args.export))


def cmd_splits(args: argparse.Namespace) -> None:
    run_splits(Path(args.dataset) if args.dataset else workdir() / "normalized" / "examples.jsonl")


def cmd_train(args: argparse.Namespace) -> None:
    run_training(smoke=args.smoke)


def cmd_evaluate(args: argparse.Namespace) -> None:
    checkpoint = Path(args.checkpoint) if args.checkpoint else workdir() / "training" / "final"
    split_dir = Path(args.splits) if args.splits else workdir() / "splits"
    run_evaluation(checkpoint, split_dir)


def cmd_bundle(args: argparse.Namespace) -> None:
    materialized = workdir()
    run_bundle(
        checkpoint=Path(args.checkpoint) if args.checkpoint else materialized / "training" / "final",
        report=materialized / "evaluation" / "report.json",
        dataset_path=materialized / "normalized" / "examples.jsonl",
        split_dir=materialized / "splits",
    )


def cmd_pipeline(args: argparse.Namespace) -> None:
    """The full post-annotation chain: dataset → splits → train → evaluate → bundle."""
    dataset_path = run_dataset(export_path(args.export))
    split_dir = run_splits(dataset_path)
    checkpoint = run_training(smoke=args.smoke)
    report = run_evaluation(checkpoint, split_dir)
    run_bundle(checkpoint, report, dataset_path, split_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="medliner", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    candidates = sub.add_parser("candidates", help="build the Label Studio import file from raw candidates")
    candidates.add_argument("--input", help="raw candidates JSONL (default: $MEDLINER_RAW_CANDIDATES)")
    candidates.set_defaults(func=cmd_candidates)

    server = sub.add_parser("label-studio", help="start the podman Label Studio server with tasks imported")
    server.add_argument("--input", help="raw candidates JSONL (default: $MEDLINER_RAW_CANDIDATES)")
    server.add_argument("--reimport", action="store_true", help="replace existing project tasks")
    server.add_argument("--port", type=int, help="host port (default: $MEDLINER_LABEL_STUDIO_PORT)")
    server.add_argument("--image", help="container image (default: $MEDLINER_LABEL_STUDIO_IMAGE)")
    server.set_defaults(func=cmd_label_studio)

    stop = sub.add_parser("label-studio-stop", help="remove the Label Studio container (annotations survive)")
    stop.set_defaults(func=cmd_label_studio_stop)

    dataset = sub.add_parser("dataset", help="validate the reviewed export into the normalized dataset")
    dataset.add_argument("--export", help="reviewed export (default: $MEDLINER_LABEL_STUDIO_EXPORT)")
    dataset.set_defaults(func=cmd_dataset)

    splits = sub.add_parser("splits", help="freeze grouped train/validation/test splits")
    splits.add_argument("--dataset", help="normalized JSONL (default: $MEDLINER_WORKDIR/normalized/examples.jsonl)")
    splits.set_defaults(func=cmd_splits)

    train = sub.add_parser("train", help="fine-tune the small GLiNER checkpoint")
    train.add_argument("--smoke", action="store_true", help="one-step GPU sanity check (run this first)")
    train.set_defaults(func=cmd_train)

    evaluate = sub.add_parser("evaluate", help="strict/lenient evaluation report for the tuned checkpoint")
    evaluate.add_argument("--checkpoint", help="checkpoint dir (default: $MEDLINER_WORKDIR/training/final)")
    evaluate.add_argument("--splits", help="split dir (default: $MEDLINER_WORKDIR/splits)")
    evaluate.set_defaults(func=cmd_evaluate)

    bundle = sub.add_parser("bundle", help="assemble the standalone export bundle")
    bundle.add_argument("--checkpoint", help="checkpoint dir (default: $MEDLINER_WORKDIR/training/final)")
    bundle.set_defaults(func=cmd_bundle)

    pipeline = sub.add_parser("pipeline", help="dataset → splits → train → evaluate → bundle")
    pipeline.add_argument("--export", help="reviewed export (default: $MEDLINER_LABEL_STUDIO_EXPORT)")
    pipeline.add_argument("--smoke", action="store_true", help="run the one-step smoke training check")
    pipeline.set_defaults(func=cmd_pipeline)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"medliner: error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
