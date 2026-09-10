"""Canonical schema contracts and deterministic split behaviour."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from medliner.schema import Annotation, Example
from medliner.splits import assert_no_group_leakage, split_examples


def _example(identifier: str, document: str) -> Example:
    text = "asthma"
    return Example(
        id=identifier,
        text=text,
        task="indication",
        source={"family": "dailymed", "document_id": document},
        annotations=[Annotation(start=0, end=6, label="disease", text=text)],
    )


def test_grouped_splits_are_deterministic_and_leakage_safe():
    examples = [
        _example("a", "doc-a"),
        _example("b", "doc-a"),
        _example("c", "doc-c"),
        _example("d", "doc-d"),
        _example("e", "doc-e"),
    ]
    first, manifest1 = split_examples(examples, seed=7)
    second, manifest2 = split_examples(examples, seed=7)
    assert manifest1.split_hash == manifest2.split_hash
    assert {key: [item.id for item in value] for key, value in first.items()} == {
        key: [item.id for item in value] for key, value in second.items()
    }
    assert_no_group_leakage(first)
    assert first["validation"] and first["test"]
    assert {item.id for item in first["train"] + first["validation"] + first["test"]} == {"a", "b", "c", "d", "e"}


def test_split_ratios_must_be_positive_and_sum_to_one():
    with pytest.raises(ValueError, match="must be positive and sum to 1"):
        split_examples([_example("a", "doc-a")], train_ratio=0.9, validation_ratio=0.1, test_ratio=0.1)


def test_regression_ids_are_withheld_and_recorded():
    examples = [_example(name, f"doc-{name}") for name in "abcde"]
    splits, manifest = split_examples(examples, seed=7, regression_ids={"a"})
    assert manifest.held_out_ids == ["a"]
    assert manifest.example_count == 4
    assert "a" not in {item.id for members in splits.values() for item in members}


def test_seed_changes_the_assignment_but_not_the_membership():
    examples = [_example(name, f"doc-{name}") for name in "abcdefgh"]
    first, manifest_a = split_examples(examples, seed=1)
    second, manifest_b = split_examples(examples, seed=2)
    assert manifest_a.split_hash != manifest_b.split_hash
    assert {item.id for members in first.values() for item in members} == {
        item.id for members in second.values() for item in members
    }


def test_examples_from_one_document_never_straddle_splits():
    examples = [_example(f"{document}-{index}", document) for document in "abcd" for index in range(3)]
    splits, _manifest = split_examples(examples, seed=3)
    assert_no_group_leakage(splits)
    placement = {item.id.split("-")[0]: name for name, members in splits.items() for item in members}
    for name, members in splits.items():
        for item in members:
            assert placement[item.id.split("-")[0]] == name


def test_overlapping_annotations_are_rejected_by_the_contract():
    with pytest.raises(ValidationError, match="overlapping annotations"):
        Example(
            id="x",
            text="pulmonary hypertension",
            task="indication",
            annotations=[
                Annotation(start=0, end=22, label="disease", text="pulmonary hypertension"),
                Annotation(start=10, end=22, label="disease", text="hypertension"),
            ],
        )


def test_annotation_text_must_match_the_source_slice():
    with pytest.raises(ValidationError, match="annotation text mismatch"):
        Example(
            id="x",
            text="asthma",
            task="indication",
            annotations=[Annotation(start=0, end=6, label="disease", text="ashtma")],
        )


def test_reviewed_examples_cannot_carry_model_suggestions():
    with pytest.raises(ValidationError, match="model suggestions"):
        Example(
            id="x",
            text="asthma",
            task="indication",
            annotations=[
                Annotation(
                    start=0, end=6, label="disease", text="asthma", status="draft", provenance="model_suggestion"
                )
            ],
        )


def test_unsupported_label_and_task_values_are_rejected():
    with pytest.raises(ValidationError, match="unsupported label"):
        Annotation(start=0, end=6, label="gene", text="asthma")
    with pytest.raises(ValidationError, match="unsupported task"):
        Example(id="x", text="asthma", task="warning")


def test_content_hash_is_stable_across_equal_examples():
    first = _example("a", "doc-a")
    second = _example("a", "doc-a")
    assert first.content_hash() == second.content_hash()
    assert first.content_hash() != _example("b", "doc-a").content_hash()
