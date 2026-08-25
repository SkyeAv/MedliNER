from __future__ import annotations

import pytest

from medliner.gliner_data import ModelLimits, char_span_to_token_span, split_words, to_gliner_dataset, to_gliner_record
from medliner.schema import Annotation, Example


def _example(text: str, spans: list[tuple[int, int, str]]) -> Example:
    return Example(
        id="x",
        text=text,
        task="contraindication",
        annotations=[Annotation(start=s, end=e, label=label, text=text[s:e]) for s, e, label in spans],
    )


def test_token_conversion_round_trips_to_the_original_character_span():
    text = "severe pulmonary hypertension"
    tokens = split_words(text)
    first, last = char_span_to_token_span(text, 7, len(text), tokens)
    assert (first, last) == (1, 2)
    assert text[tokens[first].start : tokens[last].end] == "pulmonary hypertension"


def test_unaligned_span_names_the_nearest_whole_token_span():
    text = "severe pulmonary hypertension"
    tokens = split_words(text)
    with pytest.raises(ValueError, match="nearest whole-token span"):
        char_span_to_token_span(text, 8, len(text), tokens)


def test_span_wider_than_max_width_is_refused_rather_than_silently_dropped():
    words = " ".join(f"w{index}" for index in range(20))
    example = _example(words, [(0, len(words), "DiseaseOrPhenotypicFeature")])
    with pytest.raises(ValueError, match="exceeding GLiNER max_width"):
        to_gliner_record(example, limits=ModelLimits(max_len=384, max_width=12))


def test_span_within_max_width_is_kept():
    words = " ".join(f"w{index}" for index in range(20))
    end = words.index("w5") + len("w5")
    record = to_gliner_record(
        _example(words, [(0, end, "DiseaseOrPhenotypicFeature")]), limits=ModelLimits(max_len=384, max_width=12)
    )
    assert record["ner"] == [(0, 5, "DiseaseOrPhenotypicFeature")]


def test_text_longer_than_max_len_is_refused_rather_than_truncated():
    words = " ".join(f"w{index}" for index in range(20))
    with pytest.raises(ValueError, match="exceeding GLiNER max_len"):
        to_gliner_record(_example(words, []), limits=ModelLimits(max_len=10, max_width=12))


def test_record_preserves_character_annotations_for_audit():
    text = "asthma"
    record = to_gliner_record(_example(text, [(0, 6, "DiseaseOrPhenotypicFeature")]))
    assert record["char_annotations"][0]["start"] == 0
    assert record["char_annotations"][0]["end"] == 6
    assert record["tokenized_text"] == ["asthma"]


def test_the_model_word_splitter_is_preferred_over_the_regex_fallback():
    class Model:
        data_processor = type(
            "Processor", (), {"words_splitter": staticmethod(lambda text: [("whole", 0, len(text))])}
        )()

    tokens = split_words("pulmonary hypertension", model=Model())
    assert [token.text for token in tokens] == ["whole"]


def test_the_regex_fallback_matches_gliners_whitespace_splitter():
    # gliner.data_processing.tokenizer.WhitespaceTokenSplitter uses this exact pattern.
    assert [token.text for token in split_words("co-administration of drug-X, 400 mg.")] == [
        "co-administration",
        "of",
        "drug-X",
        ",",
        "400",
        "mg",
        ".",
    ]


@pytest.mark.parametrize("start,end", [(-1, 5), (5, 5), (0, 999)])
def test_out_of_range_character_spans_are_rejected(start, end):
    text = "asthma"
    with pytest.raises(ValueError, match="invalid character span"):
        char_span_to_token_span(text, start, end, split_words(text))


def test_a_span_covering_no_whole_token_is_rejected():
    text = "asthma"
    with pytest.raises(ValueError, match="no complete model tokens"):
        char_span_to_token_span(text, 1, 3, split_words(text))


def test_limits_are_read_from_the_model_config():
    from medliner.gliner_data import model_limits

    class Model:
        config = type("Config", (), {"max_len": 384, "max_width": 12})()

    assert model_limits(Model()) == ModelLimits(max_len=384, max_width=12)
    assert model_limits(None) == ModelLimits(max_len=None, max_width=None)


def test_record_weight_is_carried_only_when_not_default():
    """Default-weight records keep the exact pre-weight key set.

    The record shape is a consumed contract (training, packaging, audits); adding a key by
    default would change every artifact hash for no information gain. A non-default weight is
    carried so semi-supervised mixes can down-weight synthetic records.
    """
    default = to_gliner_record(_example("asthma", [(0, 6, "DiseaseOrPhenotypicFeature")]))
    assert "weight" not in default
    explicit_default = to_gliner_record(_example("asthma", [(0, 6, "DiseaseOrPhenotypicFeature")]), weight=1.0)
    assert "weight" not in explicit_default
    weighted = to_gliner_record(_example("asthma", [(0, 6, "DiseaseOrPhenotypicFeature")]), weight=0.25)
    assert weighted["weight"] == 0.25


def test_dataset_weight_applies_to_every_record():
    records = to_gliner_dataset(
        [_example("asthma", [(0, 6, "DiseaseOrPhenotypicFeature")]), _example("nausea", [])], weight=0.5
    )
    assert [record["weight"] for record in records] == [0.5, 0.5]


@pytest.mark.parametrize("bad_weight", [0, -1, float("inf"), float("nan")])
def test_non_positive_or_non_finite_weights_are_rejected(bad_weight):
    """A zero, negative, or NaN weight would silently corrupt the loss instead of failing loudly."""
    with pytest.raises(ValueError, match="weight must be a positive finite number"):
        to_gliner_record(_example("asthma", []), weight=bad_weight)
