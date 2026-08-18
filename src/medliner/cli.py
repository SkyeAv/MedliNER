"""Small command-line entry point for local MEDliNER runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from .dataset import read_examples, write_examples
from .evaluation import evaluate_checkpoint
from .label_studio import normalize_export
from .packaging import build_export_bundle
from .splits import assert_no_group_leakage, split_examples
from .training import train_from_split_directory


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="medliner")
    sub = result.add_subparsers(dest="command", required=True)
    normalize = sub.add_parser("normalize")
    normalize.add_argument("export")
    normalize.add_argument("output")
    split = sub.add_parser("split")
    split.add_argument("dataset")
    split.add_argument("output_dir")
    train = sub.add_parser("train")
    train.add_argument("split_dir")
    train.add_argument("output_dir")
    train.add_argument("--config", default="configs/train-small.yaml")
    train.add_argument("--smoke", action="store_true")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("checkpoint")
    evaluate.add_argument("split_dir")
    evaluate.add_argument("output")
    bundle = sub.add_parser("bundle")
    bundle.add_argument("checkpoint")
    bundle.add_argument("metrics")
    bundle.add_argument("dataset")
    bundle.add_argument("split_dir")
    bundle.add_argument("output_dir")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "normalize":
        examples = normalize_export(args.export)
        write_examples(examples, args.output)
        return 0
    if args.command == "split":
        examples = read_examples(args.dataset)
        splits, manifest = split_examples(examples)
        assert_no_group_leakage(splits)
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        for name, members in splits.items():
            write_examples(members, output / f"{name}.jsonl")
        (output / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return 0
    if args.command == "train":
        train_from_split_directory(args.split_dir, args.output_dir, config_path=args.config, smoke_test=args.smoke)
        return 0
    if args.command == "evaluate":
        evaluate_checkpoint(args.checkpoint, args.split_dir, args.output)
        return 0
    if args.command == "bundle":
        build_export_bundle(
            checkpoint_dir=args.checkpoint,
            evaluation_path=args.metrics,
            dataset_path=args.dataset,
            split_dir=args.split_dir,
            output_dir=args.output_dir,
        )
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
