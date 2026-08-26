"""Canonical schema contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from medliner.schema import Annotation, Example


def _example(identifier: str, document: str) -> Example:
    text = "asthma"
    return Example(
        id=identifier,
        text=text,
        task="indication",
        source={"family": "dailymed", "document_id": document},
        annotations=[Annotation(start=0, end=6, label="DiseaseOrPhenotypicFeature", text=text)],
    )


def test_overlapping_annotations_are_rejected_by_the_contract():
    with pytest.raises(ValidationError, match="overlapping annotations"):
        Example(
            id="x",
            text="pulmonary hypertension",
            task="indication",
            annotations=[
                Annotation(start=0, end=22, label="DiseaseOrPhenotypicFeature", text="pulmonary hypertension"),
                Annotation(start=10, end=22, label="DiseaseOrPhenotypicFeature", text="hypertension"),
            ],
        )


def test_annotation_text_must_match_the_source_slice():
    with pytest.raises(ValidationError, match="annotation text mismatch"):
        Example(
            id="x",
            text="asthma",
            task="indication",
            annotations=[Annotation(start=0, end=6, label="DiseaseOrPhenotypicFeature", text="ashtma")],
        )


def test_reviewed_examples_cannot_carry_model_suggestions():
    with pytest.raises(ValidationError, match="model suggestions"):
        Example(
            id="x",
            text="asthma",
            task="indication",
            annotations=[
                Annotation(
                    start=0,
                    end=6,
                    label="DiseaseOrPhenotypicFeature",
                    text="asthma",
                    status="draft",
                    provenance="model_suggestion",
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
