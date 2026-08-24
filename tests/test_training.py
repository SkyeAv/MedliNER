from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from medliner.schema import Annotation, Example
from medliner.training import (
    FixedLabelCollator,
    ValidationF1Callback,
    _device,
    _enable_memory_saving,
    _eval_strategy_field,
    _export_best_checkpoint,
    _latest_checkpoint,
    _precision,
    _seed_everything,
    _training_arguments,
    load_config,
)


def test_latest_checkpoint_ignores_non_numeric_siblings(tmp_path):
    for name in ("checkpoint-2", "checkpoint-10", "final", "checkpoint-final"):
        (tmp_path / name).mkdir()
    assert _latest_checkpoint(tmp_path) == str(tmp_path / "checkpoint-10")


def test_latest_checkpoint_is_none_when_absent(tmp_path):
    assert _latest_checkpoint(tmp_path) is None


def test_fixed_label_collator_supplies_per_example_entity_types():
    seen = {}

    def collator(features, **kwargs):
        seen.update(kwargs)
        return {"features": features}

    labels = ["disease", "phenotype"]
    FixedLabelCollator(collator, labels)([{"ner": []}, {"ner": []}])
    # A flat list leaves GLiNER's classes_to_id as one dict, which create_labels then indexes
    # positionally; the batch must carry one label list per example.
    assert seen["entity_types"] == [labels, labels]


def test_precision_rejects_both_half_precision_modes():
    with pytest.raises(ValueError, match="both fp16 and bf16"):
        _precision({"fp16": True, "bf16": True}, "cuda")


def test_precision_is_disabled_on_cpu():
    assert _precision({"bf16": True}, "cpu") == (False, False)


def _checkpoint(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "pytorch_model.bin").write_bytes(b"weights")
    (path / "gliner_config.json").write_text("{}", encoding="utf-8")
    (path / "optimizer.pt").write_bytes(b"optimizer")
    (path / "trainer_state.json").write_text("{}", encoding="utf-8")
    return path


def test_best_checkpoint_is_exported_without_trainer_state(tmp_path):
    best = _checkpoint(tmp_path / "checkpoint-5")
    trainer = SimpleNamespace(state=SimpleNamespace(best_model_checkpoint=str(best)))
    final = tmp_path / "final"

    assert _export_best_checkpoint(trainer, final) == str(best)
    assert sorted(item.name for item in final.iterdir()) == ["gliner_config.json", "pytorch_model.bin"]


def test_export_falls_back_to_the_last_step_when_no_best_checkpoint(tmp_path):
    saved = {}
    trainer = SimpleNamespace(
        state=SimpleNamespace(best_model_checkpoint=None),
        save_model=lambda path: saved.setdefault("path", path),
    )
    final = tmp_path / "final"

    assert _export_best_checkpoint(trainer, final) is None
    assert saved["path"] == str(final)


def test_save_steps_must_land_on_an_evaluation(tmp_path):
    # Best-checkpoint selection can only point at a step that was both evaluated and saved.
    with pytest.raises(ValueError, match="must be a multiple of eval_steps"):
        _training_arguments(object(), {"eval_steps": 10, "save_steps": 25}, tmp_path, "cpu")


def test_load_config_rejects_a_non_mapping(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("- not-a-mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a mapping"):
        load_config(path)


def test_load_config_reads_the_committed_laptop_config():
    config = load_config(Path("configs/train-small.yaml"))
    assert config["model_id"] == "urchade/gliner_small-v2.1"
    assert config["per_device_train_batch_size"] == 1
    assert config["save_steps"] % config["eval_steps"] == 0


def test_device_falls_back_to_cpu_without_matching_kernels(monkeypatch):
    fake = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_capability=lambda: (12, 0),
            get_arch_list=lambda: ["sm_90"],
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake)
    assert _device() == "cpu"

    fake.cuda.get_arch_list = lambda: ["sm_90", "sm_120"]
    assert _device() == "cuda"

    fake.cuda.is_available = lambda: False
    assert _device() == "cpu"


def test_seeding_touches_every_available_rng(monkeypatch):
    seen = {}
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            manual_seed=lambda value: seen.setdefault("torch", value),
            cuda=SimpleNamespace(is_available=lambda: False),
        ),
    )
    _seed_everything(7)
    assert seen["torch"] == 7


def test_eval_strategy_field_matches_the_installed_transformers():
    import dataclasses

    from gliner.training import TrainingArguments

    names = {field.name for field in dataclasses.fields(TrainingArguments)}
    assert _eval_strategy_field() in names


def test_gradient_checkpointing_is_enabled_on_the_encoder_when_exposed():
    calls = []
    encoder_model = SimpleNamespace(gradient_checkpointing_enable=lambda: calls.append("on"))
    model = SimpleNamespace(
        model=SimpleNamespace(token_rep_layer=SimpleNamespace(bert_layer=SimpleNamespace(model=encoder_model)))
    )
    _enable_memory_saving(model, {"gradient_checkpointing": True})
    assert calls == ["on"]

    calls.clear()
    _enable_memory_saving(model, {"gradient_checkpointing": False})
    assert calls == []


def test_freezing_the_text_encoder_is_delegated_to_gliner():
    frozen = []
    model = SimpleNamespace(freeze_component=frozen.append)
    _enable_memory_saving(model, {"freeze_text_encoder": True, "gradient_checkpointing": False})
    assert frozen == ["text_encoder"]


def test_validation_callback_publishes_strict_f1_and_stops_on_patience():
    example = Example(
        id="a",
        text="asthma",
        task="indication",
        source={"family": "faers"},
        annotations=[Annotation(start=0, end=6, label="disease", text="asthma")],
    )
    callback = ValidationF1Callback([example], labels=["disease"], patience=2)

    class Model:
        training = True

        def eval(self):
            self.training = False

        def train(self):
            self.training = True

        def predict_entities(self, text, labels, threshold):
            return [{"start": 0, "end": 6, "label": "disease"}]

    model = Model()
    control = SimpleNamespace(should_training_stop=False)
    metrics: dict[str, float] = {}

    callback.on_evaluate(None, None, control, model=model, metrics=metrics)
    assert metrics["eval_strict_f1"] == 1.0
    assert callback.best_f1 == 1.0
    assert control.should_training_stop is False
    # The model must be handed back to the trainer in training mode.
    assert model.training is True

    # Two consecutive non-improving evaluations exhaust the patience budget.
    callback.on_evaluate(None, None, control, model=model, metrics=metrics)
    assert control.should_training_stop is False
    callback.on_evaluate(None, None, control, model=model, metrics=metrics)
    assert control.should_training_stop is True


def test_validation_callback_is_inert_without_a_model_or_metrics():
    callback = ValidationF1Callback([], labels=["disease"])
    control = SimpleNamespace(should_training_stop=False)
    assert callback.on_evaluate(None, None, control, model=None, metrics={}) is control
    assert callback.best_f1 == -1.0


def test_training_arguments_carry_the_laptop_safe_settings(tmp_path):
    captured = {}

    class Model:
        @staticmethod
        def create_training_args(**kwargs):
            captured.update(kwargs)
            return kwargs

    _training_arguments(
        Model(),
        {"eval_steps": 5, "save_steps": 10, "bf16": True, "gradient_accumulation_steps": 8, "seed": 7},
        tmp_path,
        "cuda",
    )
    assert captured["per_device_train_batch_size"] == 1
    assert captured["gradient_accumulation_steps"] == 8
    assert captured["use_cpu"] is False
    assert captured["seed"] == 7
    assert captured["dataloader_num_workers"] == 0
    assert captured["report_to"] == []
    # Selection is done by `_export_best_checkpoint`; the trainer's own reload cannot read
    # GLiNER's checkpoint key namespace.
    assert captured["load_best_model_at_end"] is False
    assert captured["metric_for_best_model"] == "eval_strict_f1"
    assert captured[_eval_strategy_field()] == "steps"
    assert captured["save_strategy"] == "steps"


def test_training_arguments_force_cpu_when_no_usable_gpu(tmp_path):
    captured = {}

    class Model:
        @staticmethod
        def create_training_args(**kwargs):
            captured.update(kwargs)
            return kwargs

    _training_arguments(Model(), {"bf16": True}, tmp_path, "cpu")
    assert captured["use_cpu"] is True
    assert captured["bf16"] is False


def test_list_dataset_is_indexable():
    from medliner.training import ListDataset

    dataset = ListDataset([{"a": 1}, {"a": 2}])
    assert len(dataset) == 2
    assert dataset[1] == {"a": 2}


def test_smoke_runs_do_not_auto_resume(tmp_path, monkeypatch):
    from medliner import training

    captured = {}

    def fake_train(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop before touching a model")

    monkeypatch.setattr(training, "load_model", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("halt")))

    config_path = tmp_path / "config.yaml"
    config_path.write_text("model_id: m\nresume: true\neval_steps: 25\nsave_steps: 25\n", encoding="utf-8")
    (tmp_path / "checkpoint-1").mkdir()

    # `load_model` halts the run right after the smoke overrides are applied, which is where the
    # resume decision is made; the surviving evidence is that the config was rewritten.
    seen = {}
    original = training.load_config

    def spy(path):
        seen.update(original(path))
        return seen

    monkeypatch.setattr(training, "load_config", spy)
    with pytest.raises(RuntimeError, match="halt"):
        training.train_from_split_directory(tmp_path, tmp_path / "out", config_path=config_path, smoke_test=True)
    assert seen["resume"] is False
    assert seen["max_steps"] == 1
