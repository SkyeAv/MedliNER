from __future__ import annotations

import json

import pytest

from medliner.benchmark import load_gold_benchmark, score_examples
from medliner.schema import Annotation, Example


def test_strict_boundary_and_no_entity_metrics():
    examples = [
        Example(
            id="positive",
            text="asthma and nausea",
            task="indication",
            source={"family": "faers"},
            annotations=[
                Annotation(start=0, end=6, label="DiseaseOrPhenotypicFeature", text="asthma"),
                Annotation(start=11, end=17, label="DiseaseOrPhenotypicFeature", text="nausea"),
            ],
        ),
        Example(id="negative", text="patients only", task="contraindication", source={"family": "dailymed"}),
    ]

    def predictor(text):
        if text.startswith("asthma"):
            return [
                {"start": 0, "end": 6, "label": "DiseaseOrPhenotypicFeature"},
                # Over-extended right boundary: a strict miss that stays a miss under
                # boundary-only scoring too, since span boundaries must match exactly.
                {"start": 11, "end": 18, "label": "DiseaseOrPhenotypicFeature"},
            ]
        return [{"start": 0, "end": 8, "label": "DiseaseOrPhenotypicFeature"}]

    report = score_examples(predictor, examples)
    assert report["overall"]["strict"]["tp"] == 1
    assert report["overall"]["strict"]["fp"] == 2
    assert report["overall"]["strict"]["fn"] == 1
    assert report["overall"]["boundary_only"]["tp"] == 1
    assert report["no_entity"]["false_positive_rate"] == 1.0
    assert report["by_task"]["indication"]["strict"]["f1"] < 1.0


def test_each_example_is_predicted_exactly_once_per_report():
    # Per-task and per-source slices reuse the same counts; re-running a GLiNER forward pass
    # for every slice would multiply scoring cost.
    calls: list[str] = []
    examples = [
        Example(
            id="a",
            text="asthma",
            task="indication",
            source={"family": "faers"},
            annotations=[Annotation(start=0, end=6, label="DiseaseOrPhenotypicFeature", text="asthma")],
        ),
        Example(id="b", text="patients only", task="contraindication", source={"family": "dailymed"}),
    ]

    def predictor(text):
        calls.append(text)
        return []

    score_examples(predictor, examples)
    assert calls == ["asthma", "patients only"]


def test_degenerate_predictions_are_ignored():
    examples = [
        Example(
            id="a",
            text="asthma",
            task="indication",
            source={"family": "faers"},
            annotations=[Annotation(start=0, end=6, label="DiseaseOrPhenotypicFeature", text="asthma")],
        )
    ]
    report = score_examples(lambda _text: [{"start": 3, "end": 3, "label": "DiseaseOrPhenotypicFeature"}], examples)
    assert report["overall"]["strict"]["fp"] == 0
    assert report["overall"]["strict"]["fn"] == 1


def test_type_key_is_accepted_as_a_label_alias():
    examples = [
        Example(
            id="a",
            text="asthma",
            task="indication",
            source={"family": "faers"},
            annotations=[Annotation(start=0, end=6, label="DiseaseOrPhenotypicFeature", text="asthma")],
        )
    ]
    # `type` stays accepted as GLiNER's alias for `label`, and the label is matched
    # case-insensitively onto the canonical form.
    report = score_examples(lambda _text: [{"start": 0, "end": 6, "type": "diseaseorphenotypicfeature"}], examples)
    assert report["overall"]["strict"]["tp"] == 1


def test_gold_benchmark_rejects_an_ambiguous_surface(tmp_path):
    path = tmp_path / "ner_gold.json"
    path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "id": "c1",
                        "source": "dailymed",
                        "text": "asthma and asthma",
                        "mentions": [{"surface": "asthma", "type": "DiseaseOrPhenotypicFeature"}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="add an explicit 'start' offset"):
        load_gold_benchmark(path)


def test_gold_benchmark_uses_an_explicit_offset_when_supplied(tmp_path):
    path = tmp_path / "ner_gold.json"
    path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "id": "c1",
                        "source": "faers",
                        "text": "asthma and asthma",
                        "mentions": [{"surface": "asthma", "type": "DiseaseOrPhenotypicFeature", "start": 11}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    example = load_gold_benchmark(path)[0]
    assert (example.annotations[0].start, example.annotations[0].end) == (11, 17)
    assert example.task == "indication"


def test_gold_benchmark_rejects_an_offset_that_does_not_match_the_surface(tmp_path):
    path = tmp_path / "ner_gold.json"
    path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "id": "c1",
                        "source": "faers",
                        "text": "Used for asthma.",
                        "mentions": [{"surface": "asthma", "type": "DiseaseOrPhenotypicFeature", "start": 0}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not contain"):
        load_gold_benchmark(path)
