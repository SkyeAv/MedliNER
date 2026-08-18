from __future__ import annotations

import pytest

from medliner.dataset import hash_file, manifest_for, read_examples, write_examples, write_manifest
from medliner.schema import Annotation, Example
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
