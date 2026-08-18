from __future__ import annotations

import json

import pytest

from medliner.gliner_data import char_span_to_token_span, split_words, to_gliner_record
from medliner.label_studio import LabelStudioExportError, normalize_export
from medliner.schema import Annotation, Example
from medliner.splits import assert_no_group_leakage, split_examples


def _export(tmp_path):
    text = "Contraindicated in patients with pulmonary hypertension and ibuprofen 400 mg orally."
    start_condition = text.index("pulmonary hypertension")
    start_drug = text.index("ibuprofen")
    payload = [
        {
            "id": "task-1",
            "data": {
                "text": text,
                "task": "contraindication",
                "source_family": "dailymed",
                "source_document_id": "doc-1",
            },
            "annotations": [
                {
                    "id": 11,
                    "created_username": "annotator",
                    "result": [
                        {
                            "id": "r1",
                            "type": "labels",
                            "from_name": "label",
                            "to_name": "text",
                            "value": {
                                "start": start_condition,
                                "end": start_condition + len("pulmonary hypertension"),
                                "text": "pulmonary hypertension",
                                "labels": ["disease"],
                            },
                        },
                        {
                            "id": "r2",
                            "type": "labels",
                            "from_name": "label",
                            "to_name": "text",
                            "value": {
                                "start": start_drug,
                                "end": start_drug + len("ibuprofen"),
                                "text": "ibuprofen",
                                "labels": ["drug"],
                            },
                        },
                    ],
                }
            ],
        },
        {
            "id": "task-empty",
            "data": {
                "text": "Contraindicated in women of childbearing potential.",
                "task": "contraindication",
                "source_family": "dailymed",
            },
            "annotations": [{"id": 12, "created_username": "annotator", "result": []}],
        },
    ]
    path = tmp_path / "export.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_label_studio_normalization_has_char_offsets_and_empty_examples(tmp_path):
    examples = normalize_export(_export(tmp_path))
    assert [example.id for example in examples] == ["task-1", "task-empty"]
    assert [(item.text, item.label) for item in examples[0].annotations] == [
        ("pulmonary hypertension", "disease"),
        ("ibuprofen", "drug"),
    ]
    assert examples[1].annotations == []


def test_label_studio_rejects_bad_export(tmp_path):
    path = _export(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[0]["annotations"][0]["result"][0]["value"]["text"] = "hypertension"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(LabelStudioExportError, match="text mismatch"):
        normalize_export(path)


def test_token_conversion_uses_inclusive_end():
    text = "severe pulmonary hypertension"
    tokens = split_words(text)
    assert char_span_to_token_span(text, 7, len(text), tokens) == (1, 2)
    example = Example(
        id="x",
        text=text,
        task="contraindication",
        annotations=[Annotation(start=7, end=len(text), label="disease", text="pulmonary hypertension")],
    )
    assert to_gliner_record(example)["ner"] == [(1, 2, "disease")]


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
