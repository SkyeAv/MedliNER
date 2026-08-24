from __future__ import annotations

import json

import pytest

from medliner.label_studio import LabelStudioExportError, normalize_export, normalize_task

TEXT = "Contraindicated in patients with pulmonary hypertension and ibuprofen 400 mg orally."


def _span(start: int, end: int, label: str, *, result_id: str = "r1") -> dict:
    return {
        "id": result_id,
        "type": "labels",
        "from_name": "label",
        "to_name": "text",
        "value": {"start": start, "end": end, "text": TEXT[start:end], "labels": [label]},
    }


def _task(results: list[dict], **annotation_overrides) -> dict:
    annotation = {"id": 11, "created_username": "annotator", "result": results}
    annotation.update(annotation_overrides)
    return {
        "id": "task-1",
        "data": {"text": TEXT, "task": "contraindication", "source_family": "dailymed"},
        "annotations": [annotation],
    }


def test_drag_selected_trailing_whitespace_is_trimmed():
    start = TEXT.index("pulmonary hypertension")
    end = start + len("pulmonary hypertension ")  # the drag picked up the following space
    example = normalize_task(_task([_span(start, end, "disease")]))
    annotation = example.annotations[0]
    assert annotation.text == "pulmonary hypertension"
    assert (annotation.start, annotation.end) == (start, end - 1)


def test_whitespace_only_span_is_rejected():
    start = TEXT.index(" and")
    with pytest.raises(LabelStudioExportError, match="whitespace-only span"):
        normalize_task(_task([_span(start, start + 1, "disease")]))


def test_cancelled_annotations_are_not_silently_negative_examples():
    task = _task([], was_cancelled=True)
    with pytest.raises(LabelStudioExportError, match="cancelled/skipped"):
        normalize_task(task)


def test_deliberately_empty_annotation_stays_a_negative_example():
    example = normalize_task(_task([]))
    assert example.annotations == []
    assert example.metadata["annotation_count"] == 0


def test_overlapping_spans_raise_the_adapter_error(tmp_path):
    start = TEXT.index("pulmonary hypertension")
    task = _task(
        [
            _span(start, start + len("pulmonary hypertension"), "disease", result_id="r1"),
            _span(start, start + len("pulmonary"), "phenotype", result_id="r2"),
        ]
    )
    with pytest.raises(LabelStudioExportError, match="not a valid MedliNER example"):
        normalize_task(task)


def test_conflicting_duplicate_span_labels_raise():
    start = TEXT.index("ibuprofen")
    end = start + len("ibuprofen")
    task = _task([_span(start, end, "drug", result_id="r1"), _span(start, end, "disease", result_id="r2")])
    with pytest.raises(LabelStudioExportError, match="conflicting duplicate span"):
        normalize_task(task)


def test_identical_duplicate_spans_are_collapsed():
    start = TEXT.index("ibuprofen")
    end = start + len("ibuprofen")
    task = _task([_span(start, end, "drug", result_id="r1"), _span(start, end, "drug", result_id="r2")])
    assert len(normalize_task(task).annotations) == 1


def test_duplicate_task_ids_are_rejected(tmp_path):
    path = tmp_path / "export.json"
    path.write_text(json.dumps([_task([]), _task([])]), encoding="utf-8")
    with pytest.raises(LabelStudioExportError, match="duplicate task ids"):
        normalize_export(path)


def test_unreviewed_tasks_are_blocked_by_default(tmp_path):
    task = _task([])
    task["data"]["annotation_status"] = "draft"
    path = tmp_path / "export.json"
    path.write_text(json.dumps([task]), encoding="utf-8")
    with pytest.raises(LabelStudioExportError, match="unreviewed tasks cannot enter training"):
        normalize_export(path)


def test_jsonl_export_is_accepted(tmp_path):
    path = tmp_path / "export.jsonl"
    second = _task([])
    second["id"] = "task-2"
    path.write_text("\n".join(json.dumps(item) for item in (_task([]), second)), encoding="utf-8")
    assert [item.id for item in normalize_export(path)] == ["task-1", "task-2"]


def test_unresolved_multiple_annotation_sets_raise():
    task = _task([])
    task["annotations"].append({"id": 12, "created_username": "other", "result": []})
    with pytest.raises(LabelStudioExportError, match="annotation sets"):
        normalize_task(task)


def test_adjudicated_set_is_selected_and_marked():
    task = _task([])
    task["data"]["annotation_status"] = "adjudicated"
    task["annotations"].append({"id": 12, "created_username": "adjudicator", "result": [], "ground_truth": True})
    example = normalize_task(task)
    assert example.annotation_status.value == "adjudicated"
    assert example.annotation_set_id == "12"


def test_empty_export_yields_no_examples(tmp_path):
    path = tmp_path / "export.json"
    path.write_text("   ", encoding="utf-8")
    assert normalize_export(path) == []


def test_malformed_jsonl_line_is_reported_with_its_number(tmp_path):
    path = tmp_path / "export.jsonl"
    path.write_text('{"id": 1}\nnot json\n', encoding="utf-8")
    with pytest.raises(LabelStudioExportError, match="invalid JSONL at line 2"):
        normalize_export(path)


def test_scalar_export_is_rejected(tmp_path):
    path = tmp_path / "export.json"
    path.write_text("42", encoding="utf-8")
    with pytest.raises(LabelStudioExportError, match="must be a JSON object, array, or JSONL"):
        normalize_export(path)


def test_single_object_export_is_accepted(tmp_path):
    path = tmp_path / "export.json"
    path.write_text(json.dumps(_task([])), encoding="utf-8")
    assert [item.id for item in normalize_export(path)] == ["task-1"]


def test_task_without_an_id_is_rejected():
    task = _task([])
    del task["id"]
    with pytest.raises(LabelStudioExportError, match="task has no id"):
        normalize_task(task)


def test_missing_or_invalid_task_metadata_is_rejected():
    task = _task([])
    task["data"]["task"] = "warning"
    with pytest.raises(LabelStudioExportError, match="invalid task metadata"):
        normalize_task(task)


def test_missing_text_is_rejected():
    task = _task([])
    del task["data"]["text"]
    with pytest.raises(LabelStudioExportError, match="has no string data.text"):
        normalize_task(task)


def test_raw_text_key_is_accepted_as_an_alias():
    task = _task([])
    task["data"]["raw_text"] = task["data"].pop("text")
    assert normalize_task(task).text == TEXT


def test_unsupported_label_is_rejected():
    start = TEXT.index("asthma") if "asthma" in TEXT else TEXT.index("ibuprofen")
    with pytest.raises(LabelStudioExportError, match="unsupported Label Studio label"):
        normalize_task(_task([_span(start, start + 4, "gene")]))


def test_span_outside_the_text_is_rejected():
    with pytest.raises(LabelStudioExportError, match="invalid character span"):
        normalize_task(
            _task([{**_span(0, 5, "disease"), "value": {"start": 0, "end": 9999, "text": TEXT, "labels": ["disease"]}}])
        )


def test_span_text_disagreeing_with_the_source_is_rejected():
    span = _span(0, 5, "disease")
    span["value"]["text"] = "wrong"
    with pytest.raises(LabelStudioExportError, match="span text mismatch"):
        normalize_task(_task([span]))


def test_span_must_carry_exactly_one_label():
    span = _span(TEXT.index("ibuprofen"), TEXT.index("ibuprofen") + 9, "drug")
    span["value"]["labels"] = ["drug", "disease"]
    with pytest.raises(LabelStudioExportError, match="exactly one string label"):
        normalize_task(_task([span]))


def test_non_label_results_are_ignored():
    task = _task([{"id": "rel", "type": "relation", "value": {}}])
    assert normalize_task(task).annotations == []


def test_malformed_annotations_container_is_rejected():
    task = _task([])
    task["annotations"] = "not a list"
    with pytest.raises(LabelStudioExportError, match="malformed annotations"):
        normalize_task(task)


def test_invalid_annotation_status_is_rejected():
    task = _task([])
    task["data"]["annotation_status"] = "maybe"
    with pytest.raises(LabelStudioExportError, match="invalid annotation_status"):
        normalize_task(task)


def test_source_metadata_survives_normalization():
    task = _task([])
    task["data"]["source"] = {
        "family": "faers",
        "document_id": "doc-9",
        "record_id": "rec-9",
        "section": "34070-3",
        "source_uri": "https://example.invalid/spl",
        "source_hash": "abc123",
    }
    source = normalize_task(task, export_id="export-1").source
    assert (source.family, source.document_id, source.record_id) == ("faers", "doc-9", "rec-9")
    assert (source.section, source.source_uri, source.source_hash) == (
        "34070-3",
        "https://example.invalid/spl",
        "abc123",
    )


def test_final_annotation_id_resolves_competing_sets():
    task = _task([])
    task["annotations"].append({"id": 12, "created_username": "other", "result": []})
    task["data"]["final_annotation_id"] = 12
    assert normalize_task(task).annotation_set_id == "12"
