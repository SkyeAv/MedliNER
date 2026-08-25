"""Canonical JSONL dataset I/O and manifests."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from .schema import DatasetManifest, Example


def read_examples(path: str | Path) -> list[Example]:
    examples: list[Example] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            examples.append(Example.model_validate_json(line))
        except Exception as exc:  # Pydantic supplies the detailed field path.
            raise ValueError(f"invalid canonical example at line {line_number}: {exc}") from exc
    return examples


def write_examples(examples: Iterable[Example], path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [example.model_dump_json() for example in examples]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return hash_file(path)


def hash_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_for(examples: Iterable[Example], *, input_export_hash: str, dataset_id: str) -> DatasetManifest:
    values = list(examples)
    return DatasetManifest(
        dataset_id=dataset_id,
        input_export_hash=input_export_hash,
        example_count=len(values),
        label_counts=dict(
            sorted(Counter(annotation.label for item in values for annotation in item.annotations).items())
        ),
        task_counts=dict(sorted(Counter(item.task for item in values).items())),
        origin_counts=dict(
            sorted(
                Counter(annotation.origin or "unrecorded" for item in values for annotation in item.annotations).items()
            )
        ),
        provenance_counts=dict(
            sorted(Counter(annotation.provenance for item in values for annotation in item.annotations).items())
        ),
    )


def write_manifest(manifest: DatasetManifest, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")


__all__ = ["hash_file", "manifest_for", "read_examples", "write_examples", "write_manifest"]
