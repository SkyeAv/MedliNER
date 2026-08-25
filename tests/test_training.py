from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from gliner.training import Trainer as GLiNERTrainer

from medliner.dataset import hash_file, write_examples
from medliner.schema import Annotation, Example
from medliner.training import (
    FixedLabelCollator,
    ValidationF1Callback,
    WeightedCollator,
    WeightedTrainer,
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
    # The committed mix down-weights synthetic examples 10x relative to gold (1.0).
    assert config["synthetic_weight"] == 0.1


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


def _gliner_trainer_args() -> SimpleNamespace:
    """Only the focal/loss knobs GLiNER's compute_loss reads; no Trainer __init__ machinery."""
    return SimpleNamespace(
        focal_loss_alpha=-1,
        focal_loss_gamma=0,
        rel_focal_loss_alpha=None,
        rel_focal_loss_gamma=None,
        focal_loss_prob_margin=0,
        label_smoothing=0,
        loss_reduction="sum",
        negatives=1.0,
        masking="global",
    )


def _bare_trainer(cls=WeightedTrainer):
    """Bypass transformers' __init__ (model, accelerator, ...): compute_loss needs only `args`."""
    trainer = object.__new__(cls)
    trainer.args = _gliner_trainer_args()
    return trainer


class _LossModel:
    """Records every keyword it receives and returns a fixed scalar loss, like GLiNER's wrapper."""

    def __init__(self, loss: float) -> None:
        self.loss = torch.tensor(loss)
        self.seen: dict | None = None

    def __call__(self, **kwargs):
        self.seen = dict(kwargs)
        return SimpleNamespace(loss=self.loss)


def test_compute_loss_pops_and_scales_by_sample_weight():
    # The weighting contract: the tensor leaves the inputs here and only here, and the batch
    # loss is multiplied by it — a 0.1-weighted synthetic step must contribute a tenth.
    model = _LossModel(3.0)
    trainer = _bare_trainer()
    inputs = {"input_ids": torch.tensor([1]), "sample_weight": torch.tensor([0.1])}

    result = trainer.compute_loss(model, inputs)

    assert torch.isclose(result, torch.tensor(0.3))
    assert "sample_weight" not in inputs
    assert model.seen is not None and "sample_weight" not in model.seen


def test_compute_loss_forwards_no_sample_weight_to_model():
    # Without a weight the override must be a transparent pass-through of the GLiNER trainer:
    # identical inputs reach the model and the loss comes back untouched.
    model = _LossModel(2.0)
    inputs = {"input_ids": torch.tensor([1]), "labels": torch.tensor([[0]])}

    result = _bare_trainer().compute_loss(model, inputs)

    assert torch.equal(result, torch.tensor(2.0))
    # GLiNER's compute_loss adds its focal-loss kwargs to the model call; the pass-through must
    # still forward every input unchanged and never leak the weighting side channel.
    assert set(inputs) <= set(model.seen)
    assert all(model.seen[key] is value for key, value in inputs.items())
    assert "sample_weight" not in model.seen


def test_weight_one_is_equivalent_to_unweighted_loss():
    # Weight 1.0 everywhere must reproduce the historical gold-only loss bit-for-bit, or a
    # configured-but-inert synthetic_weight would still perturb training numerics.
    inputs = {"input_ids": torch.tensor([1])}

    unweighted = _bare_trainer(GLiNERTrainer).compute_loss(_LossModel(1.25), dict(inputs))
    weighted = _bare_trainer(WeightedTrainer).compute_loss(
        _LossModel(1.25), {**inputs, "sample_weight": torch.tensor([1.0])}
    )

    assert torch.equal(unweighted, weighted)
    assert torch.equal(weighted, torch.tensor(1.25))


def test_mixed_weights_reject_batch_size_above_one():
    # GLiNER reduces a batch to one loss scalar, so gold and synthetic gradients cannot be
    # scaled apart after collation; a mixed batch (only possible with batch size > 1) must
    # fail loudly naming both weights instead of silently averaging the populations.
    def collator(features, **kwargs):
        return {"num_tokens": torch.tensor([len(f["tokenized_text"]) for f in features])}

    with pytest.raises(ValueError, match=r"sample weights 0\.1 and 1\.0"):
        WeightedCollator(collator, ["disease"])(
            [
                {"tokenized_text": ["a"], "ner": [], "weight": 1.0},
                {"tokenized_text": ["b"], "ner": [], "weight": 0.1},
            ]
        )


def test_weighted_collator_injects_defaulted_sample_weight_tensor():
    # Records keep no `weight` key at the 1.0 default, so the collator must default missing
    # keys to gold weight 1.0 and still attach the tensor for uniform batches.
    def collator(features, **kwargs):
        return {"num_tokens": torch.tensor([len(f["tokenized_text"]) for f in features])}

    batch = WeightedCollator(collator, ["disease"])([{"tokenized_text": ["a"], "ner": []}])
    assert torch.equal(batch["sample_weight"], torch.tensor([1.0]))

    batch = WeightedCollator(collator, ["disease"])([{"tokenized_text": ["b"], "ner": [], "weight": 0.1}])
    assert torch.equal(batch["sample_weight"], torch.tensor([0.1]))


def test_compute_loss_rejects_mixed_sample_weight_tensor():
    # Defense in depth: a hand-built batch bypassing WeightedCollator cannot be silently
    # averaged either; the trainer names both weights and fails before the forward pass.
    model = _LossModel(1.0)
    inputs = {"input_ids": torch.tensor([1, 1]), "sample_weight": torch.tensor([1.0, 0.1])}
    with pytest.raises(ValueError, match=r"uniform within a batch, got 0\.1 and 1\.0"):
        _bare_trainer().compute_loss(model, inputs)
    assert model.seen is None


@pytest.mark.parametrize("bad", [0, -0.5, 1.5, float("inf"), float("nan"), "heavy", None])
def test_load_config_rejects_synthetic_weight_outside_unit_range(tmp_path, bad):
    # synthetic_weight > 1 would up-weight the noisier population and <= 0 would silently
    # drop it from the loss; both must be refused at config load, before any model is touched.
    path = tmp_path / "config.yaml"
    path.write_text(f"synthetic_weight: {bad!r}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="synthetic_weight must be a number in \\(0, 1\\]"):
        load_config(path)


@pytest.mark.parametrize("good", [0.1, 1.0])
def test_load_config_accepts_synthetic_weight_in_unit_range(tmp_path, good):
    path = tmp_path / "config.yaml"
    path.write_text(f"synthetic_weight: {good}\n", encoding="utf-8")
    assert load_config(path)["synthetic_weight"] == good


def _gold_example(example_id: str, text: str = "asthma", span: tuple[int, int] = (0, 6)) -> Example:
    return Example(
        id=example_id,
        text=text,
        task="indication",
        source={"family": "faers"},
        annotations=[Annotation(start=span[0], end=span[1], label="disease", text=text[span[0] : span[1]])],
    )


def _stub_model() -> SimpleNamespace:
    # `_training_arguments` only calls `create_training_args`; conversion falls back to the
    # regex splitter because the stub has no data_processor.
    return SimpleNamespace(create_training_args=lambda **kwargs: kwargs)


def _stub_trainer() -> SimpleNamespace:
    return SimpleNamespace(
        state=SimpleNamespace(best_model_checkpoint=None),
        callback_handler=SimpleNamespace(callbacks=[]),
        train=lambda **kwargs: None,
        save_model=lambda path: Path(path).mkdir(parents=True, exist_ok=True),
    )


def _write_splits(split_dir: Path) -> None:
    split_dir.mkdir(parents=True)
    write_examples([_gold_example("gold-1"), _gold_example("gold-2")], split_dir / "train.jsonl")
    write_examples([_gold_example("val-1", "nausea", (0, 6))], split_dir / "validation.jsonl")
    write_examples([_gold_example("test-1", "asthma", (0, 6))], split_dir / "test.jsonl")


def _patch_training_internals(monkeypatch, captured: dict) -> None:
    monkeypatch.setattr("medliner.training.load_model", lambda *a, **k: _stub_model())

    def fake_make_trainer(model, train_records, eval_records, eval_examples, args, config, *, weighted=False):
        captured["train_records"] = train_records
        captured["weighted"] = weighted
        return _stub_trainer()

    monkeypatch.setattr("medliner.training._make_trainer", fake_make_trainer)


def test_train_mixes_synthetic_examples_at_configured_weight(tmp_path, monkeypatch):
    # End-to-end wiring: the pool is read from $MEDLINER_WORKDIR, synthetic records carry the
    # configured weight while gold stays at the implicit 1.0, and the run metadata records the
    # mix counts, the weight, and the pool's content hash for provenance.
    workdir = tmp_path / "work"
    monkeypatch.setenv("MEDLINER_WORKDIR", str(workdir))
    split_dir = tmp_path / "splits"
    _write_splits(split_dir)
    pool = [_gold_example("gold-1-synth-patient", "asthma flares", (0, 6))]
    pool_path = workdir / "synthetic" / "examples.jsonl"
    write_examples(pool, pool_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "model_id: stub\neval_steps: 25\nsave_steps: 25\nresume: false\nsynthetic_weight: 0.1\n", encoding="utf-8"
    )
    captured: dict = {}
    _patch_training_internals(monkeypatch, captured)
    from medliner.training import train_from_split_directory

    final = train_from_split_directory(split_dir, tmp_path / "out", config_path=config_path)

    records = captured["train_records"]
    assert [record["id"] for record in records] == ["gold-1", "gold-2", "gold-1-synth-patient"]
    assert all("weight" not in record for record in records[:2])
    assert records[2]["weight"] == 0.1
    assert captured["weighted"] is True
    metadata = json.loads((final / "medliner-training.json").read_text(encoding="utf-8"))
    assert metadata["gold_train_examples"] == 2
    assert metadata["synthetic_examples"] == 1
    assert metadata["train_examples"] == 3
    assert metadata["synthetic_weight"] == 0.1
    assert metadata["synthetic_dataset_hash"] == hash_file(pool_path)


def test_missing_synthetic_pool_with_configured_weight_is_a_hard_error(tmp_path, monkeypatch):
    # A configured synthetic_weight with no pool must not degrade into a silent gold-only run:
    # the operator asked for a semi-supervised mix. --no-synthetic is the explicit opt-out.
    workdir = tmp_path / "work"
    monkeypatch.setenv("MEDLINER_WORKDIR", str(workdir))
    split_dir = tmp_path / "splits"
    _write_splits(split_dir)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "model_id: stub\neval_steps: 25\nsave_steps: 25\nresume: false\nsynthetic_weight: 0.1\n", encoding="utf-8"
    )
    captured: dict = {}
    _patch_training_internals(monkeypatch, captured)
    from medliner.training import train_from_split_directory

    with pytest.raises(ValueError, match=r"synthetic pool.*--no-synthetic"):
        train_from_split_directory(split_dir, tmp_path / "out", config_path=config_path)

    final = train_from_split_directory(split_dir, tmp_path / "out", config_path=config_path, no_synthetic=True)
    assert [record["id"] for record in captured["train_records"]] == ["gold-1", "gold-2"]
    assert captured["weighted"] is False
    metadata = json.loads((final / "medliner-training.json").read_text(encoding="utf-8"))
    assert metadata["synthetic_examples"] == 0
    assert metadata["synthetic_dataset_hash"] is None


def test_synthetic_pool_without_configured_weight_is_a_hard_error(tmp_path, monkeypatch):
    # A present pool must not silently fall back to weight 1.0: machine-paraphrased examples
    # would then train at full gold strength and steer the model without any operator intent.
    # --no-synthetic stays the explicit gold-only opt-out even with a pool sitting in the workdir.
    workdir = tmp_path / "work"
    monkeypatch.setenv("MEDLINER_WORKDIR", str(workdir))
    split_dir = tmp_path / "splits"
    _write_splits(split_dir)
    write_examples(
        [_gold_example("gold-1-synth-paraphrase", "asthma flares", (0, 6))],
        workdir / "synthetic" / "examples.jsonl",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model_id: stub\neval_steps: 25\nsave_steps: 25\nresume: false\n", encoding="utf-8")
    captured: dict = {}
    _patch_training_internals(monkeypatch, captured)
    from medliner.training import train_from_split_directory

    with pytest.raises(ValueError, match=r"sets no synthetic_weight.*--no-synthetic"):
        train_from_split_directory(split_dir, tmp_path / "out", config_path=config_path)

    train_from_split_directory(split_dir, tmp_path / "out", config_path=config_path, no_synthetic=True)
    assert [record["id"] for record in captured["train_records"]] == ["gold-1", "gold-2"]
    assert captured["weighted"] is False


def test_train_without_synthetic_pool_or_synthetic_weight_is_gold_only(tmp_path, monkeypatch):
    # No pool and no configured weight is the historical default: the plain
    # FixedLabelCollator + GLiNER Trainer path, bit-for-bit unchanged.
    monkeypatch.setenv("MEDLINER_WORKDIR", str(tmp_path / "work"))
    split_dir = tmp_path / "splits"
    _write_splits(split_dir)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model_id: stub\neval_steps: 25\nsave_steps: 25\nresume: false\n", encoding="utf-8")
    captured: dict = {}
    _patch_training_internals(monkeypatch, captured)
    from medliner.training import train_from_split_directory

    train_from_split_directory(split_dir, tmp_path / "out", config_path=config_path)

    assert captured["weighted"] is False
    assert all("weight" not in record for record in captured["train_records"])


def test_synthetic_ids_in_held_out_splits_are_rejected(tmp_path, monkeypatch):
    # Held-out splits measure gold performance; a synthetic id appearing in validation or test
    # would inflate the selection metric with data the model was trained on.
    workdir = tmp_path / "work"
    monkeypatch.setenv("MEDLINER_WORKDIR", str(workdir))
    split_dir = tmp_path / "splits"
    _write_splits(split_dir)
    write_examples([_gold_example("val-1", "nausea", (0, 6))], workdir / "synthetic" / "examples.jsonl")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "model_id: stub\neval_steps: 25\nsave_steps: 25\nresume: false\nsynthetic_weight: 0.1\n", encoding="utf-8"
    )
    captured: dict = {}
    _patch_training_internals(monkeypatch, captured)
    from medliner.training import train_from_split_directory

    with pytest.raises(AssertionError, match="leaked into validation/test.*val-1"):
        train_from_split_directory(split_dir, tmp_path / "out", config_path=config_path)
