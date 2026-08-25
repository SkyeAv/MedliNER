from __future__ import annotations

import pytest

from medliner.dataset import hash_file, manifest_for, read_examples, write_examples, write_manifest
from medliner.schema import Annotation, DatasetManifest, Example
from medliner.splits import group_key


def _example(identifier: str, **source) -> Example:
    return Example(
        id=identifier,
        text="asthma",
        task="indication",
        source=source or {"family": "dailymed", "document_id": "doc-a"},
        annotations=[Annotation(start=0, end=6, label="disease", text="asthma")],
    )


def test_jsonl_round_trip_and_hash(tmp_path):
    path = tmp_path / "examples.jsonl"
    digest = write_examples([_example("a"), _example("b")], path)
    assert digest == hash_file(path)
    assert [item.id for item in read_examples(path)] == ["a", "b"]


def test_blank_lines_are_ignored(tmp_path):
    path = tmp_path / "examples.jsonl"
    write_examples([_example("a")], path)
    path.write_text(path.read_text(encoding="utf-8") + "\n\n", encoding="utf-8")
    assert len(read_examples(path)) == 1


def test_a_corrupt_line_is_reported_with_its_line_number(tmp_path):
    path = tmp_path / "examples.jsonl"
    write_examples([_example("a")], path)
    path.write_text(path.read_text(encoding="utf-8") + "{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid canonical example at line 2"):
        read_examples(path)


def test_empty_dataset_writes_an_empty_file(tmp_path):
    path = tmp_path / "examples.jsonl"
    write_examples([], path)
    assert path.read_text(encoding="utf-8") == ""
    assert read_examples(path) == []


def test_manifest_counts_labels_and_tasks(tmp_path):
    manifest = manifest_for([_example("a"), _example("b")], input_export_hash="abc", dataset_id="def")
    assert manifest.label_counts == {"disease": 2}
    assert manifest.task_counts == {"indication": 2}
    path = tmp_path / "manifest.json"
    write_manifest(manifest, path)
    assert '"example_count": 2' in path.read_text(encoding="utf-8")


def test_group_key_prefers_document_then_record_then_text():
    assert group_key(_example("a", family="dailymed", document_id="doc-1")) == "document:dailymed:doc-1"
    assert group_key(_example("a", family="faers", record_id="rec-1")) == "record:faers:rec-1"
    text_key = group_key(_example("a", family="faers"))
    assert text_key.startswith("text:faers:indication:")
    # Whitespace and case differences must not split one repeated sentence across groups.
    other = Example(id="b", text="  ASTHMA  ", task="indication", source={"family": "faers"})
    assert group_key(other) == text_key


def test_manifest_counts_how_much_of_the_dataset_is_an_untouched_model_span():
    # The central risk of pre-labeling is a reviewer rubber-stamping suggestions. That is only
    # auditable if the dataset manifest reports it.
    examples = [
        Example(
            id="a",
            text="asthma and nausea",
            task="indication",
            source={"family": "faers"},
            annotations=[
                Annotation(start=0, end=6, label="disease", text="asthma", origin="prediction"),
                Annotation(start=11, end=17, label="phenotype", text="nausea", origin="prediction-changed"),
            ],
        ),
        Example(
            id="b",
            text="asthma",
            task="indication",
            source={"family": "faers"},
            annotations=[Annotation(start=0, end=6, label="disease", text="asthma", origin="manual")],
        ),
    ]
    manifest = manifest_for(examples, input_export_hash="h", dataset_id="d")
    assert manifest.origin_counts == {"manual": 1, "prediction": 1, "prediction-changed": 1}


def test_spans_from_an_export_without_origins_are_counted_as_unrecorded():
    manifest = manifest_for([_example("a")], input_export_hash="h", dataset_id="d")
    assert manifest.origin_counts == {"unrecorded": 1}


def test_manifest_counts_annotation_provenance():
    """The human/synthetic mix must be visible in the manifest artifact itself.

    Downstream consumers (trust policies, training-time weighting) decide based on this count;
    without it the synthetic share of a dataset is invisible until someone re-derives it.
    """
    mixed = Example(
        id="a",
        text="asthma and nausea",
        task="indication",
        source={"family": "dailymed"},
        annotations=[
            Annotation(start=0, end=6, label="disease", text="asthma", provenance="human"),
            Annotation(start=11, end=17, label="phenotype", text="nausea", provenance="human"),
        ],
    )
    also_synthetic = Example(
        id="b",
        text="asthma and nausea",
        task="indication",
        source={"family": "synthetic"},
        annotations=[
            Annotation(start=0, end=6, label="disease", text="asthma", provenance="synthetic"),
            Annotation(start=11, end=17, label="phenotype", text="nausea", provenance="synthetic"),
        ],
    )
    manifest = manifest_for([mixed, also_synthetic], input_export_hash="h", dataset_id="d")
    assert manifest.provenance_counts == {"human": 2, "synthetic": 2}


def test_manifests_written_before_provenance_counts_still_validate():
    """Additive compatibility: a manifest from an older MedliNER version has no provenance_counts.

    The field is optional with a default so historical artifacts keep validating instead of
    forcing a rewrite of every stored manifest the moment the schema learns about synthetic data.
    """
    legacy_json = (
        '{"schema_version": "medliner.dataset.v1", "dataset_id": "d", '
        '"input_export_hash": "h", "example_count": 1, '
        '"label_counts": {"disease": 1}, "task_counts": {"indication": 1}, '
        '"origin_counts": {"unrecorded": 1}}'
    )
    manifest = DatasetManifest.model_validate_json(legacy_json)
    assert manifest.provenance_counts == {}
    assert manifest.example_count == 1
