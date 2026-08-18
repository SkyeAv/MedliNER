from __future__ import annotations

from medliner.evaluation import score_examples
from medliner.schema import Annotation, Example


def test_strict_boundary_and_no_entity_metrics():
    examples = [
        Example(
            id="positive",
            text="asthma and ibuprofen",
            task="indication",
            source={"family": "faers"},
            annotations=[
                Annotation(start=0, end=6, label="disease", text="asthma"),
                Annotation(start=11, end=20, label="drug", text="ibuprofen"),
            ],
        ),
        Example(id="negative", text="patients only", task="contraindication", source={"family": "dailymed"}),
    ]

    def predictor(text):
        if text.startswith("asthma"):
            return [
                {"start": 0, "end": 6, "label": "disease"},
                {"start": 11, "end": 20, "label": "phenotype"},
            ]
        return [{"start": 0, "end": 8, "label": "disease"}]

    report = score_examples(predictor, examples)
    assert report["overall"]["strict"]["tp"] == 1
    assert report["overall"]["strict"]["fp"] == 2
    assert report["overall"]["strict"]["fn"] == 1
    assert report["overall"]["boundary_only"]["tp"] == 2
    assert report["no_entity"]["false_positive_rate"] == 1.0
    assert report["by_task"]["indication"]["strict"]["f1"] < 1.0
