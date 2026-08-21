from __future__ import annotations

import json

import pytest

from medliner.evaluation import load_gold_benchmark, score_examples
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


def test_each_example_is_predicted_exactly_once_per_report():
    # Per-task and per-source slices reuse the same counts; re-running a GLiNER forward pass
    # for every slice would multiply evaluation cost and the training-time validation callback.
    calls: list[str] = []
    examples = [
        Example(
            id="a",
            text="asthma",
            task="indication",
            source={"family": "faers"},
            annotations=[Annotation(start=0, end=6, label="disease", text="asthma")],
        ),
        Example(id="b", text="patients only", task="contraindication", source={"family": "dailymed"}),
    ]

    def predictor(text):
        calls.append(text)
        return []

    score_examples(predictor, examples)
    assert calls == ["asthma", "patients only"]


def test_truncation_report_flags_texts_over_the_model_word_budget():
    long_text = " ".join(f"w{index}" for index in range(30))
    examples = [Example(id="long", text=long_text, task="indication", source={"family": "faers"})]
    report = score_examples(lambda _text: [], examples, max_words=10)
    assert report["truncation"]["over_budget_examples"] == 1
    assert report["truncation"]["example_ids"] == ["long"]


def test_truncation_is_not_claimed_when_the_budget_is_unknown():
    examples = [Example(id="a", text="asthma", task="indication", source={"family": "faers"})]
    assert score_examples(lambda _text: [], examples)["truncation"] == {"max_words": None, "checked": False}


def test_degenerate_predictions_are_ignored():
    examples = [
        Example(
            id="a",
            text="asthma",
            task="indication",
            source={"family": "faers"},
            annotations=[Annotation(start=0, end=6, label="disease", text="asthma")],
        )
    ]
    report = score_examples(lambda _text: [{"start": 3, "end": 3, "label": "disease"}], examples)
    assert report["overall"]["strict"]["fp"] == 0
    assert report["overall"]["strict"]["fn"] == 1


def test_type_key_is_accepted_as_a_label_alias():
    examples = [
        Example(
            id="a",
            text="asthma",
            task="indication",
            source={"family": "faers"},
            annotations=[Annotation(start=0, end=6, label="disease", text="asthma")],
        )
    ]
    report = score_examples(lambda _text: [{"start": 0, "end": 6, "type": "Disease"}], examples)
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
                        "mentions": [{"surface": "asthma", "type": "disease"}],
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
                        "mentions": [{"surface": "asthma", "type": "disease", "start": 11}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    example = load_gold_benchmark(path)[0]
    assert (example.annotations[0].start, example.annotations[0].end) == (11, 17)
    assert example.task == "indication"


def _dataset(tmp_path, examples):
    from medliner.dataset import write_examples

    split_dir = tmp_path / "splits"
    write_examples(examples, split_dir / "test.jsonl")
    write_examples(examples, split_dir / "validation.jsonl")
    return split_dir


def _gold(tmp_path):
    example = Example(
        id="a",
        text="asthma",
        task="indication",
        source={"family": "faers"},
        annotations=[Annotation(start=0, end=6, label="disease", text="asthma")],
    )
    return _dataset(tmp_path, [example])


def _write_gold(tmp_path):
    """One gold case matching the `_gold` split example, so a perfect predictor scores f1=1.0."""
    path = tmp_path / "ner_gold.json"
    path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "id": "g1",
                        "source": "faers",
                        "text": "asthma",
                        "mentions": [{"surface": "asthma", "type": "disease"}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


def test_evaluate_checkpoint_reports_tuned_and_baseline_systems(tmp_path, monkeypatch):
    from medliner import evaluation

    split_dir = _gold(tmp_path)
    checkpoint = tmp_path / "final"
    checkpoint.mkdir()
    (checkpoint / "medliner-training.json").write_text(json.dumps({"model_id": "base"}), encoding="utf-8")

    def perfect(_checkpoint, *, threshold=0.3):
        return lambda _text: [{"start": 0, "end": 6, "label": "disease"}]

    monkeypatch.setattr(evaluation, "make_gliner_predictor", perfect)
    monkeypatch.setenv("MEDLINER_BENCHMARK", str(_write_gold(tmp_path)))

    report = evaluation.evaluate_checkpoint(checkpoint, split_dir, tmp_path / "report.json")
    assert report["evaluated_split"] == "test"
    assert report["tuned"]["overall"]["strict"]["f1"] == 1.0
    assert report["tuned_gold_regression"]["overall"]["strict"]["f1"] == 1.0
    assert report["baselines"]["untuned_gliner"]["overall"]["strict"]["f1"] == 1.0
    assert report["baselines"]["untuned_gliner_gold_regression"]["overall"]["strict"]["f1"] == 1.0
    assert json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))["checkpoint"] == str(checkpoint)


def test_a_missing_baseline_does_not_void_the_tuned_report(tmp_path, monkeypatch):
    from medliner import evaluation

    split_dir = _gold(tmp_path)
    checkpoint = tmp_path / "final"
    checkpoint.mkdir()
    (checkpoint / "medliner-training.json").write_text(json.dumps({"model_id": "base"}), encoding="utf-8")

    def only_tuned_is_available(checkpoint_arg, *, threshold=0.3):
        if str(checkpoint_arg) == "base":
            raise ModuleNotFoundError("base checkpoint unavailable")
        return lambda _text: [{"start": 0, "end": 6, "label": "disease"}]

    monkeypatch.setattr(evaluation, "make_gliner_predictor", only_tuned_is_available)
    monkeypatch.setenv("MEDLINER_BENCHMARK", str(_write_gold(tmp_path)))

    report = evaluation.evaluate_checkpoint(checkpoint, split_dir, tmp_path / "report.json")
    assert report["tuned"]["overall"]["strict"]["f1"] == 1.0
    assert "ModuleNotFoundError" in report["baselines"]["untuned_gliner_error"]


def test_validation_split_is_used_when_no_test_split_exists(tmp_path, monkeypatch):
    from medliner import evaluation
    from medliner.dataset import write_examples

    split_dir = tmp_path / "splits"
    write_examples(
        [Example(id="a", text="asthma", task="indication", source={"family": "faers"})],
        split_dir / "validation.jsonl",
    )
    monkeypatch.setattr(evaluation, "make_gliner_predictor", lambda *a, **k: lambda _text: [])
    monkeypatch.setenv("MEDLINER_BENCHMARK", str(_write_gold(tmp_path)))

    report = evaluation.evaluate_checkpoint(tmp_path, split_dir, tmp_path / "report.json", include_baselines=False)
    assert report["evaluated_split"] == "validation"
    assert "baselines" not in report


def test_a_missing_benchmark_fails_loudly_with_an_ingest_hint(tmp_path, monkeypatch):
    from medliner import evaluation

    split_dir = _gold(tmp_path)
    monkeypatch.setattr(
        evaluation, "make_gliner_predictor", lambda *a, **k: lambda _text: [{"start": 0, "end": 6, "label": "disease"}]
    )
    monkeypatch.setenv("MEDLINER_BENCHMARK", str(tmp_path / "absent" / "ner_gold.json"))

    with pytest.raises(RuntimeError, match="medliner ingest"):
        evaluation.evaluate_checkpoint(tmp_path, split_dir, tmp_path / "report.json")


def test_gliner_predictor_wrapper_normalizes_model_output():
    from medliner.evaluation import GLiNERPredictor

    class Model:
        config = type("Config", (), {"max_len": 384})()

        def predict_entities(self, text, labels, threshold):
            assert labels == ["disease", "phenotype", "drug"]
            return [{"start": 0, "end": 6, "label": "Disease", "score": 0.9}]

    predictor = GLiNERPredictor(Model(), threshold=0.3)
    assert predictor.max_words == 384
    assert predictor("asthma") == [{"start": 0, "end": 6, "label": "disease", "text": "asthma", "score": 0.9}]
