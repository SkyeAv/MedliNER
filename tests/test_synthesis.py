"""Contract tests for synthetic provenance (the schema/dataset story of the synthesis backlog).

The synthesis engine does not exist yet. These tests pin only the contract it must satisfy:
synthetic data validates as synthetic, and it can never masquerade as human work.
"""

from __future__ import annotations

from typing import get_args

import pytest
from pydantic import ValidationError

from medliner.schema import PROVENANCE_VALUES, Annotation, Example, Provenance


def _synthetic_example(provenance: str) -> Example:
    return Example(
        id="synth-1",
        text="asthma",
        task="indication",
        source={"family": "synthetic"},
        annotations=[Annotation(start=0, end=6, label="disease", text="asthma", provenance=provenance)],
    )


def test_synthetic_provenance_is_accepted():
    """A machine-synthesized example must validate end-to-end carrying provenance='synthetic'.

    The synthesis pipeline cannot emit data the canonical contract rejects, so acceptance is
    pinned before any generator exists.
    """
    example = _synthetic_example("synthetic")
    assert example.annotations[0].provenance == "synthetic"
    # PROVENANCE_VALUES must stay derived from the literal, never a hand-maintained tuple.
    assert "synthetic" in PROVENANCE_VALUES
    assert PROVENANCE_VALUES == get_args(Provenance)


@pytest.mark.parametrize("claimed", ["human", "adjudicated"])
def test_synthetic_annotation_cannot_claim_human_provenance(claimed):
    """A synthetic example claiming human (or adjudicated) provenance must be rejected.

    Adjudicated is a human claim too — an adjudicator resolved the span. Letting synthetic data
    claim either value would silently contaminate every audit that trusts provenance: review
    effort accounting, dataset manifests, and downstream trust policies.
    """
    with pytest.raises(ValidationError, match="claims human provenance"):
        _synthetic_example(claimed)
