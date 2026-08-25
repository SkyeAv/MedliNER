# Laptop-safe training

The first supported target is `urchade/gliner_small-v2.1`. This is deliberate; a successful small-model run is more useful than an OOM-prone large run.

## Required sequence

1. Materialize a reviewed dataset and frozen splits.
2. Materialize `training_run` with run config `{"smoke": true}` for one batch/one step.
3. Confirm the checkpoint and loss are written.
4. Materialize `training_run` without the smoke flag for the bounded training configuration.
5. Resume with the same output directory if interrupted; the trainer discovers the latest numbered `checkpoint-*` directory. Smoke runs never auto-resume, so re-running one repeats the same one-step check instead of appending a step.
6. Evaluate the final/best checkpoint on reviewed validation/test data.

The default configuration uses sequence length 384, micro-batch size 1, gradient accumulation 8, mixed precision on CUDA, gradient checkpointing where the encoder exposes it, five maximum epochs, and two retained checkpoints. Adjust sequence length downward before increasing batch size if VRAM is tight.

`bf16` is enabled by default for the RTX 5070 Ti, which supports it in the validated Torch environment. Use `fp16: true` instead if a different CUDA/Torch combination lacks BF16 support. CPU execution is supported for validation but is not a practical full training target.

## Checkpoint semantics

The output directory contains Hugging Face-compatible checkpoints and `final/medliner-training.json`. Resume records the prior checkpoint path and the exact configuration. Never overwrite a dataset or split manifest while resuming a run.

Training uses the GLiNER 0.2.28 bundled trainer API with canonical records converted to `tokenized_text` and inclusive token spans. A conversion test checks that every token span maps back to the original Label Studio character span.

## Large checkpoint

`gliner_large-v2.5` is optional. Do not make it a gate for the project. If evaluated later, begin with a one-step smoke test and consider freezing the text encoder; full Adam fine-tuning can exceed 12 GB VRAM even with mixed precision.

## Fixed label vocabulary

GLiNER normally derives each batch's entity vocabulary from the gold labels that happen to be
present, plus sampled negatives. That interacts badly with two MedliNER decisions: reviewed
no-entity examples are first-class training signal, and the micro-batch size is 1. A batch whose
only example has no annotations then has *zero* entity types, and the loss fails on
`scores.view(BS, -1, CL)` with `CL == 0`.

`FixedLabelCollator` pins the vocabulary to `disease` and `phenotype` for every batch. This
also matches inference, where all three labels are always queried, and makes the loss independent
of which labels a given batch happened to contain. Override with `labels:` in the training config
if the schema ever changes.

## Per-sample weighting of synthetic examples

GLiNER 0.2.28 has no native per-sample weighting: `Trainer.compute_loss` reduces the whole
batch to a single loss scalar, and neither the GLiNER trainer API nor the underlying training
arguments carry per-example weights. MedliNER implements the semi-supervised mix itself, with
tests pinning the numerics:

- `WeightedCollator` attaches a per-batch `sample_weight` tensor — gold 1.0, synthetic
  `synthetic_weight` (default 0.1, i.e. ten times less) — and rejects any batch that mixes the
  two populations, because the batch's one loss scalar could not scale their gradients apart
  afterwards. With `per_device_train_batch_size: 1` a mixed batch is impossible anyway; the
  check keeps larger batch sizes honest.
- `WeightedTrainer.compute_loss` — a tested override, the only method overridden — pops the
tensor and multiplies the batch loss by it before gradient accumulation. Weight 1.0 everywhere
reproduces the unweighted loss exactly, so the gold-only path (`--no-synthetic` or no pool) is
numerically untouched.

The pool comes from `medliner synthesize` (`make llm` → `make synthesize` → `make train`; ratio,
gates, and workers are the `MEDLINER_SYNTH_*` environment variables documented in the
README's semi-supervised section). A configured `synthetic_weight` with no pool present is a
hard error, not a silent gold-only run: generate the pool or opt out explicitly with
`medliner train --no-synthetic`.

## Conversion budgets

GLiNER discards supervision silently in two places, so MedliNER refuses the record instead:

- text longer than `config.max_len` (384 word tokens) is truncated with only a `UserWarning`;
- a gold span wider than `config.max_width` (12 word tokens) is never enumerated as a span
  candidate, so it receives no label at all.

Both raise during `to_gliner_dataset`, naming the example and the offending span. Shorten the
candidate text upstream, or raise `max_length` if VRAM allows.

## Best-checkpoint selection

`load_best_model_at_end` is deliberately off. GLiNER saves the *inner* module's state dict
(`token_rep_layer.*`) while transformers reloads it into the GLiNER wrapper
(`model.token_rep_layer.*`); every key mismatches, so the reload is a silent no-op that leaves
the last step's weights in place — the opposite of selecting on validation F1.

Instead, `metric_for_best_model="eval_strict_f1"` records `best_model_checkpoint`, and MedliNER
copies that checkpoint into `final/` after training, dropping optimizer/scheduler/RNG state. The
selected path is recorded as `selected_checkpoint` in `final/medliner-training.json` alongside
`best_validation_strict_f1`.
