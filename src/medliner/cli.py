"""Small command-line entry point for local MEDliNER runs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from . import __version__
from .dataset import hash_file, manifest_for, read_examples, write_examples, write_manifest
from .evaluation import evaluate_checkpoint
from .label_studio import normalize_export
from .packaging import build_export_bundle
from .splits import assert_no_group_leakage, split_examples
from .training import train_from_split_directory

DEFAULT_TRAIN_CONFIG = "configs/train-small.yaml"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="medliner")
    result.add_argument("--version", action="version", version=f"medliner {__version__}")
    sub = result.add_subparsers(dest="command", required=True)

    normalize = sub.add_parser("normalize", help="validate a reviewed Label Studio export into canonical JSONL")
    normalize.add_argument("export")
    normalize.add_argument("output")
    normalize.add_argument(
        "--allow-unreviewed",
        action="store_true",
        help="import draft/rejected tasks too (they are never training candidates)",
    )

    split = sub.add_parser("split", help="freeze deterministic, leakage-resistant grouped splits")
    split.add_argument("dataset")
    split.add_argument("output_dir")
    split.add_argument("--seed", type=int, default=int(os.environ.get("MEDLINER_SPLIT_SEED", "2026")))
    split.add_argument("--train-ratio", type=float, default=0.8)
    split.add_argument("--validation-ratio", type=float, default=0.1)
    split.add_argument("--test-ratio", type=float, default=0.1)
    split.add_argument(
        "--regression-ids",
        default=os.environ.get("MEDLINER_REGRESSION_IDS"),
        help="JSON file of example ids to withhold from every split",
    )

    train = sub.add_parser("train", help="fine-tune the configured small GLiNER checkpoint")
    train.add_argument("split_dir")
    train.add_argument("output_dir")
    train.add_argument("--config", default=os.environ.get("MEDLINER_TRAIN_CONFIG", DEFAULT_TRAIN_CONFIG))
    train.add_argument("--smoke", action="store_true", help="run a single step through the full training code path")
    train.add_argument("--resume-from-checkpoint", default=None)

    evaluate = sub.add_parser("evaluate", help="score a checkpoint against reviewed data and baselines")
    evaluate.add_argument("checkpoint")
    evaluate.add_argument("split_dir")
    evaluate.add_argument("output")
    evaluate.add_argument("--threshold", type=float, default=0.3)
    evaluate.add_argument("--no-baselines", action="store_true")

    bundle = sub.add_parser("bundle", help="assemble the standalone uploadable artifact directory")
    bundle.add_argument("checkpoint")
    bundle.add_argument("metrics")
    bundle.add_argument("dataset")
    bundle.add_argument("split_dir")
    bundle.add_argument("output_dir")
    return result


def _normalize(args: argparse.Namespace) -> int:
    examples = normalize_export(args.export, require_reviewed=not args.allow_unreviewed)
    output = Path(args.output)
    dataset_hash = write_examples(examples, output)
    manifest = manifest_for(examples, input_export_hash=hash_file(args.export), dataset_id=dataset_hash)
    write_manifest(manifest, output.with_name("manifest.json"))
    print(f"{len(examples)} examples -> {output} (sha256 {dataset_hash[:12]})")
    return 0


def _split(args: argparse.Namespace) -> int:
    examples = read_examples(args.dataset)
    regression_ids = (
        set(json.loads(Path(args.regression_ids).read_text(encoding="utf-8"))) if args.regression_ids else set()
    )
    splits, manifest = split_examples(
        examples,
        seed=args.seed,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        regression_ids=regression_ids,
    )
    assert_no_group_leakage(splits)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, members in splits.items():
        write_examples(members, output / f"{name}.jsonl")
    (output / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    sizes = ", ".join(f"{name}={len(members)}" for name, members in sorted(splits.items()))
    print(f"{sizes}; held_out={len(manifest.held_out_ids)}; split_hash={manifest.split_hash[:12]}")
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    report = evaluate_checkpoint(
        args.checkpoint,
        args.split_dir,
        args.output,
        include_baselines=not args.no_baselines,
        threshold=args.threshold,
    )
    print(f"strict f1 on {report['evaluated_split']}: {report['tuned']['overall']['strict']['f1']:.4f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "normalize":
        return _normalize(args)
    if args.command == "split":
        return _split(args)
    if args.command == "train":
        result = train_from_split_directory(
            args.split_dir,
            args.output_dir,
            config_path=args.config,
            resume_from_checkpoint=args.resume_from_checkpoint,
            smoke_test=args.smoke,
        )
        print(f"checkpoint -> {result}")
        return 0
    if args.command == "evaluate":
        return _evaluate(args)
    if args.command == "bundle":
        result = build_export_bundle(
            checkpoint_dir=args.checkpoint,
            evaluation_path=args.metrics,
            dataset_path=args.dataset,
            split_dir=args.split_dir,
            output_dir=args.output_dir,
        )
        print(f"bundle -> {result}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
