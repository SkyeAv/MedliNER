"""Deterministic leakage-resistant grouped splits."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable

from .schema import Example, SplitManifest


def group_key(example: Example) -> str:
    """Keep source-document duplicates together; fall back to normalized content identity."""
    source = example.source
    if source.document_id:
        return f"document:{source.family}:{source.document_id}"
    if source.record_id:
        return f"record:{source.family}:{source.record_id}"
    normalized = " ".join(example.text.lower().split())
    return f"text:{source.family}:{example.task}:{hashlib.sha256(normalized.encode('utf-8')).hexdigest()}"


def _ordered_groups(examples: list[Example], seed: int) -> list[tuple[str, list[Example]]]:
    groups: dict[str, list[Example]] = defaultdict(list)
    for example in examples:
        groups[group_key(example)].append(example)
    return sorted(
        groups.items(),
        key=lambda item: hashlib.sha256(f"{seed}:{item[0]}".encode()).hexdigest(),
    )


def split_examples(
    examples: Iterable[Example],
    *,
    seed: int = 2026,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    regression_ids: set[str] | None = None,
) -> tuple[dict[str, list[Example]], SplitManifest]:
    """Split by source group, never by an individual duplicated sentence."""
    values = list(examples)
    ratios = {"train": train_ratio, "validation": validation_ratio, "test": test_ratio}
    if any(value <= 0 for value in ratios.values()) or abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError("split ratios must be positive and sum to 1")
    regression_ids = regression_ids or set()
    kept = [example for example in values if example.id not in regression_ids]
    held_out = sorted(example.id for example in values if example.id in regression_ids)
    groups = _ordered_groups(kept, seed)
    targets = {name: len(kept) * ratio for name, ratio in ratios.items()}
    counts = {name: 0 for name in ratios}
    output: dict[str, list[Example]] = {name: [] for name in ratios}
    pending = list(groups)
    # Preserve a usable validation and test set whenever there are at least three source groups.
    # Without this reservation, a few small groups can all be greedily assigned to train.
    if len(pending) >= 3:
        for reserved_split in ("validation", "test"):
            _key, members = pending.pop(0)
            output[reserved_split].extend(sorted(members, key=lambda item: item.id))
            counts[reserved_split] += len(members)
    for _key, members in pending:
        # Put the next source group into the split with the largest remaining target deficit.
        chosen = max(ratios, key=lambda name: (targets[name] - counts[name], -list(ratios).index(name)))
        output[chosen].extend(sorted(members, key=lambda item: item.id))
        counts[chosen] += len(members)
    for members in output.values():
        members.sort(key=lambda item: item.id)
    ids = {name: [item.id for item in members] for name, members in output.items()}
    digest_input = "\n".join(f"{name}:{item_id}" for name in ids for item_id in ids[name])
    split_hash = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()
    manifest = SplitManifest(
        seed=seed,
        ratios=ratios,
        group_count=len(groups),
        example_count=len(kept),
        example_ids=ids,
        held_out_ids=held_out,
        split_hash=split_hash,
    )
    return output, manifest


def assert_no_group_leakage(splits: dict[str, list[Example]]) -> None:
    seen: dict[str, str] = {}
    for split, examples in splits.items():
        for example in examples:
            key = group_key(example)
            prior = seen.setdefault(key, split)
            if prior != split:
                raise AssertionError(f"source group {key!r} appears in {prior} and {split}")


__all__ = ["assert_no_group_leakage", "group_key", "split_examples"]
