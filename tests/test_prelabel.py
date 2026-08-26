"""Pre-labeling is a suggestion pipeline, so every test here is about what a human sees.

None of these tests load a model: the seam is a plain callable, matching how
``tests/test_evaluation.py`` exercises the evaluation predictor.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import pytest

from medliner.prelabel import (
    DEFAULT_MODEL_ID,
    POPULATION_PHRASES,
    PrelabelCache,
    PrelabelError,
    Suggestion,
    attach_predictions,
    build_prediction,
    cache_key,
    check_model_budgets,
    model_max_width,
    model_version,
    normalize_surface,
    prelabel_texts,
    select_spans,
    suggest,
    suggestions_from_windows,
    token_budget,
    trim_hedges,
    windows,
)
from medliner.schema import ALLOWED_LABELS

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "label_studio_ner.xml"


def _entity(text: str, surface: str, label: str, score: float = 0.9) -> dict[str, object]:
    start = text.index(surface)
    return {"start": start, "end": start + len(surface), "label": label, "score": score}


def _single_window(text: str, entities: list[dict[str, object]]) -> list[Suggestion]:
    """Post-process one already-windowed prediction, which is the common case."""
    return suggestions_from_windows(text, [(0, text)], [entities])


# --- windowing ------------------------------------------------------------------------------


def test_windows_tile_the_text_exactly_so_offsets_remap_by_addition():
    text = "Contraindicated in active liver disease. Indicated for asthma. Also for nausea."
    for start, window in windows(text, budget=4):
        assert text[start : start + len(window)] == window


def test_windows_never_exceed_the_token_budget():
    # GLiNER truncates past config.max_len with only a UserWarning, which silently costs recall.
    text = " ".join(f"token{index}" for index in range(50)) + "."
    assert all(len(window.split()) <= 6 for _, window in windows(text, budget=6))


def test_a_single_over_budget_sentence_is_hard_split_rather_than_truncated():
    text = " ".join(f"word{index}" for index in range(30))  # one piece, no terminal punctuation
    parts = windows(text, budget=7)
    assert len(parts) > 1
    assert "".join(window for _, window in parts).replace(" ", "") == text.replace(" ", "")


def test_blank_text_yields_no_windows():
    assert windows("   \n ", budget=10) == []


def test_token_budget_prefers_the_model_then_the_default():
    model = type("Model", (), {"config": type("Config", (), {"max_len": 768})()})()
    assert token_budget(model) == 768
    assert token_budget(object()) == 384
    assert token_budget(model, 0) == 1  # never below 1


# --- guide rule 2: hedges -------------------------------------------------------------------


def test_leading_hedges_are_trimmed_off_a_span():
    text = "Contraindicated in patients with recent myocardial infarction."
    start = text.index("recent myocardial infarction")
    end = start + len("recent myocardial infarction")
    assert trim_hedges(text, start, end) == (text.index("myocardial infarction"), end)


def test_a_span_of_nothing_but_hedges_is_dropped():
    text = "in patients with asthma"
    assert trim_hedges(text, 0, text.index("asthma")) is None


def test_clinical_qualifiers_survive_the_trim():
    # The hedge list is a closed class; anything unlisted counts as a real qualifier.
    for phrase in ("severe heart failure", "active liver disease", "pulmonary hypertension"):
        text = f"Contraindicated in {phrase}."
        start = text.index(phrase)
        assert trim_hedges(text, start, start + len(phrase)) == (start, start + len(phrase))


def test_hedge_trim_is_applied_to_model_output():
    text = "Contraindicated in patients with recent myocardial infarction."
    drops: Counter[str] = Counter()
    spans = suggestions_from_windows(
        text, [(0, text)], [[_entity(text, "recent myocardial infarction", "DiseaseOrPhenotypicFeature")]], drops=drops
    )
    assert [span.text for span in spans] == ["myocardial infarction"]
    assert drops["hedge"] == 0  # trimmed, not dropped


# --- guide rule 3: population descriptors ----------------------------------------------------


def test_population_descriptors_are_never_suggested():
    text = "Contraindicated in women of childbearing potential."
    drops: Counter[str] = Counter()
    spans = suggestions_from_windows(
        text,
        [(0, text)],
        [[_entity(text, "women of childbearing potential", "DiseaseOrPhenotypicFeature")]],
        drops=drops,
    )
    assert spans == []
    assert drops["population"] == 1


def test_a_population_descriptor_hiding_behind_a_hedge_is_still_dropped():
    # "in pregnant women" survives the population check verbatim, then trims to "pregnant women".
    text = "Contraindicated in pregnant women."
    drops: Counter[str] = Counter()
    assert (
        suggestions_from_windows(
            text, [(0, text)], [[_entity(text, "in pregnant women", "DiseaseOrPhenotypicFeature")]], drops=drops
        )
        == []
    )
    assert drops["population"] == 1


def test_every_population_phrase_normalizes_to_itself():
    assert all(normalize_surface(phrase) == phrase for phrase in POPULATION_PHRASES)


# --- guide rule 7: no overlap ----------------------------------------------------------------


def test_overlapping_suggestions_collapse_to_the_longest():
    text = "Contraindicated in pulmonary hypertension."
    spans = _single_window(
        text,
        [
            _entity(text, "hypertension", "DiseaseOrPhenotypicFeature", score=0.99),
            _entity(text, "pulmonary hypertension", "DiseaseOrPhenotypicFeature", score=0.5),
        ],
    )
    assert [span.text for span in spans] == ["pulmonary hypertension"]


def test_de_overlap_is_deterministic_regardless_of_model_order():
    spans = [
        Suggestion(start=0, end=6, label="DiseaseOrPhenotypicFeature", text="asthma", score=0.5),
        Suggestion(start=0, end=6, label="DiseaseOrPhenotypicFeature", text="asthma", score=0.5),
    ]
    assert select_spans(spans) == select_spans(list(reversed(spans)))


def test_disjoint_suggestions_are_all_kept_and_sorted_by_position():
    text = "Indicated for asthma and nausea."
    spans = _single_window(
        text,
        [_entity(text, "nausea", "DiseaseOrPhenotypicFeature"), _entity(text, "asthma", "DiseaseOrPhenotypicFeature")],
    )
    assert [(span.start, span.text) for span in spans] == [(14, "asthma"), (25, "nausea")]


# --- budgets and label filtering --------------------------------------------------------------


def test_a_span_wider_than_max_width_is_dropped_before_a_human_can_accept_it():
    # gliner_data refuses to convert a gold span wider than max_width, so accepting one would
    # break the dataset build long after the annotator moved on.
    surface = " ".join(f"word{index}" for index in range(15))
    text = f"Contraindicated in {surface} disease."
    drops: Counter[str] = Counter()
    assert (
        suggestions_from_windows(
            text, [(0, text)], [[_entity(text, surface, "DiseaseOrPhenotypicFeature")]], max_width=12, drops=drops
        )
        == []
    )
    assert drops["width"] == 1


def test_labels_outside_the_schema_are_dropped():
    text = "Indicated for asthma."
    drops: Counter[str] = Counter()
    assert suggestions_from_windows(text, [(0, text)], [[_entity(text, "asthma", "drug")]], drops=drops) == []
    assert drops["label"] == 1


def test_suggestion_text_is_always_an_exact_slice_of_the_source():
    text = "Contraindicated in patients with severe heart failure and asthma."
    spans = suggest(
        lambda window: [
            _entity(window, "severe heart failure", "DiseaseOrPhenotypicFeature"),
            _entity(window, "asthma", "DiseaseOrPhenotypicFeature"),
        ],
        text,
    )
    assert spans
    assert all(text[span.start : span.end] == span.text for span in spans)


def test_a_mention_cut_by_a_hard_window_split_is_rejoined():
    # Sentence-piece windows tile exactly, but a budget hard split drops the whitespace between
    # windows and can cut a multiword mention in two. Left unmerged the annotator sees
    # "myasthenia" and "gravis" as separate spans.
    text = " ".join(["filler"] * 6) + " myasthenia gravis " + " ".join(["filler"] * 6)
    parts = windows(text, budget=7)
    assert [window for _, window in parts] == [
        "filler filler filler filler filler filler myasthenia",
        "gravis filler filler filler filler filler filler",
    ]
    raw = [
        [
            {
                "start": parts[0][1].index("myasthenia"),
                "end": len(parts[0][1]),
                "label": "DiseaseOrPhenotypicFeature",
                "score": 0.8,
            }
        ],
        [{"start": 0, "end": len("gravis"), "label": "DiseaseOrPhenotypicFeature", "score": 0.6}],
    ]
    (span,) = suggestions_from_windows(text, parts, raw)
    assert span.text == "myasthenia gravis"
    assert text[span.start : span.end] == span.text
    assert span.score == 0.8  # the higher-scoring side supplies label and score


# --- Label Studio wire shape --------------------------------------------------------------------


def test_prediction_control_names_match_the_labeling_config():
    # Label Studio silently ignores a prediction whose from_name/to_name it cannot resolve.
    root = ET.parse(CONFIG).getroot()
    labels_node = root.find("Labels")
    text_node = root.find("Text")
    assert labels_node is not None
    assert text_node is not None
    (region,) = build_prediction(
        "medliner-abc", [Suggestion(0, 6, "DiseaseOrPhenotypicFeature", "asthma", 0.9)], version="v"
    )["result"]
    assert region["from_name"] == labels_node.get("name")
    assert region["to_name"] == text_node.get("name")
    assert region["type"] == "labels"
    assert region["value"] == {"start": 0, "end": 6, "text": "asthma", "labels": ["DiseaseOrPhenotypicFeature"]}


def test_prediction_labels_are_all_offered_by_the_labeling_config():
    offered = {node.get("value") for node in ET.parse(CONFIG).iter("Label")}
    assert offered == set(ALLOWED_LABELS)


def test_predictions_are_byte_identical_across_runs():
    spans = [
        Suggestion(0, 6, "DiseaseOrPhenotypicFeature", "asthma", 0.9),
        Suggestion(11, 17, "DiseaseOrPhenotypicFeature", "nausea", 0.7),
    ]
    first = build_prediction("medliner-abc", spans, version="m@0.35")
    assert json.dumps(first, sort_keys=True) == json.dumps(
        build_prediction("medliner-abc", spans, version="m@0.35"), sort_keys=True
    )


def test_region_ids_differ_per_task_so_two_tasks_never_share_one():
    span = Suggestion(0, 6, "DiseaseOrPhenotypicFeature", "asthma", 0.9)
    left = build_prediction("medliner-aaa", [span], version="v")["result"][0]["id"]
    right = build_prediction("medliner-bbb", [span], version="v")["result"][0]["id"]
    assert left != right


def test_a_task_with_no_suggestions_still_carries_an_empty_prediction():
    # An empty prediction says "the model looked and found nothing"; a missing one says nothing
    # at all, and the two are very different for a reviewer.
    (task,) = attach_predictions([{"id": "medliner-abc", "data": {"text": "patients only"}}], {}, version="v")
    assert task["predictions"] == [{"model_version": "v", "score": 0.0, "result": []}]


def test_attach_predictions_does_not_mutate_the_input_tasks():
    tasks = [{"id": "medliner-abc", "data": {"text": "asthma"}}]
    attach_predictions(
        tasks, {"medliner-abc": [Suggestion(0, 6, "DiseaseOrPhenotypicFeature", "asthma", 0.9)]}, version="v"
    )
    assert "predictions" not in tasks[0]


def test_model_version_names_the_checkpoint_and_threshold():
    assert model_version(DEFAULT_MODEL_ID, 0.35) == "gliner_large-v2.5@0.35"


# --- batching and caching ------------------------------------------------------------------------


def test_all_windows_are_predicted_in_one_batched_call():
    calls: list[int] = []

    def predict(texts):
        calls.append(len(texts))
        return [[] for _ in texts]

    prelabel_texts(predict, {f"t{index}": f"Indicated for asthma {index}." for index in range(5)})
    assert calls == [5]


def test_batches_are_ordered_longest_first_to_limit_padding_waste():
    seen: list[str] = []

    def predict(texts):
        seen.extend(texts)
        return [[] for _ in texts]

    prelabel_texts(predict, {"short": "asthma", "long": "Contraindicated in active liver disease."})
    assert seen == sorted(seen, key=len, reverse=True)


def test_a_cache_hit_skips_the_model_entirely(tmp_path):
    text = "Indicated for asthma."
    key = cache_key(text, model_id=DEFAULT_MODEL_ID, threshold=0.35, labels=ALLOWED_LABELS, budget=384)
    cache = PrelabelCache(tmp_path / "cache.json")
    cache.put(key, [Suggestion(14, 20, "DiseaseOrPhenotypicFeature", "asthma", 0.9)])
    cache.save()

    reloaded = PrelabelCache(tmp_path / "cache.json").load()

    def predict(texts):
        raise AssertionError(f"model was called for {texts!r}")

    result = prelabel_texts(predict, {"t1": text}, cache=reloaded)
    assert result == {"t1": [Suggestion(14, 20, "DiseaseOrPhenotypicFeature", "asthma", 0.9)]}
    assert reloaded.hits == 1


def test_changing_the_threshold_invalidates_the_cache_key():
    text = "Indicated for asthma."
    common = {"model_id": DEFAULT_MODEL_ID, "labels": ALLOWED_LABELS, "budget": 384}
    assert cache_key(text, threshold=0.35, **common) != cache_key(text, threshold=0.5, **common)


def test_a_corrupt_cache_degrades_to_a_miss_rather_than_blocking_annotation(tmp_path):
    path = tmp_path / "cache.json"
    path.write_text("{ this is not json", encoding="utf-8")
    assert PrelabelCache(path).load().entries == {}


def test_a_predictor_returning_the_wrong_number_of_results_is_a_loud_error():
    with pytest.raises(PrelabelError, match="returned 0 results"):
        prelabel_texts(lambda texts: [], {"t1": "Indicated for asthma."})


def _model(max_len: int = 768, max_width: int = 12):
    return type("Model", (), {"config": type("Config", (), {"max_len": max_len, "max_width": max_width})()})()


def test_a_window_budget_the_checkpoint_cannot_honour_is_refused():
    # GLiNER truncates past max_len with only a UserWarning; silently losing recall is the exact
    # failure windowing exists to prevent.
    with pytest.raises(PrelabelError, match="exceeds the checkpoint's max_len"):
        check_model_budgets(_model(max_len=384), budget=768, max_width=12)


def test_a_span_width_the_checkpoint_cannot_enumerate_is_refused():
    with pytest.raises(PrelabelError, match="exceeds the checkpoint's max_width"):
        check_model_budgets(_model(), budget=384, max_width=20)


def test_the_shipped_defaults_fit_the_pre_labeling_checkpoint():
    # gliner_large-v2.5 ships max_len 768 / max_width 12.
    assert check_model_budgets(_model(), budget=384, max_width=12) is None


def test_max_width_falls_back_when_the_checkpoint_declares_none():
    assert model_max_width(object()) == 12
    assert model_max_width(_model(), 3) == 3


def test_text_the_sentence_splitter_cannot_tile_becomes_one_window():
    # Leading terminal punctuation defeats the piece regex; coverage must stay contiguous or
    # offsets would silently drift.
    text = ". asthma"
    assert windows(text, budget=50) == [(0, text)]


def test_a_model_span_with_impossible_offsets_is_dropped():
    text = "Indicated for asthma."
    drops: Counter[str] = Counter()
    assert (
        suggestions_from_windows(
            text,
            [(0, text)],
            [[{"start": 0, "end": len(text) + 10, "label": "DiseaseOrPhenotypicFeature", "score": 0.9}]],
            drops=drops,
        )
        == []
    )
    assert drops["label"] == 1


def test_an_all_hedge_span_is_counted_as_a_hedge_drop():
    text = "Contraindicated in patients with asthma."
    drops: Counter[str] = Counter()
    start = text.index("in patients with")
    assert (
        suggestions_from_windows(
            text,
            [(0, text)],
            [
                [
                    {
                        "start": start,
                        "end": start + len("in patients with"),
                        "label": "DiseaseOrPhenotypicFeature",
                        "score": 0.9,
                    }
                ]
            ],
            drops=drops,
        )
        == []
    )
    assert drops["hedge"] == 1
