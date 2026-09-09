# Laptop-safe training

The first supported target is `urchade/gliner_small-v2.1`. This is deliberate; a successful small-model run is more useful than an OOM-prone large run.

## Required sequence

1. Materialize a reviewed dataset and frozen splits.
2. Run `medliner train --smoke` (one batch / one step).
3. Confirm the checkpoint and loss are written.
4. Materialize `training_run` without the smoke flag for the bounded training configuration.
5. Resume with the same output directory if interrupted; the trainer discovers the latest numbered `checkpoint-*` directory. Smoke runs never auto-resume, so re-running one repeats the same one-step check instead of appending a step.
6. Evaluate the final/best checkpoint on reviewed validation/test data.

The default configuration uses sequence length 384, micro-batch size 1, gradient accumulation 8, mixed precision on CUDA, gradient checkpointing where the encoder exposes it, five maximum epochs, and two retained checkpoints. Adjust sequence length downward before increasing batch size if VRAM is tight.

GLiNER trains the pretrained encoder (`token_rep_layer.*`) and the span/prompt stack at separate rates. The default config keeps that split: `learning_rate` (the encoder) is 1e-5 and `others_lr` (the head) is 5e-5, tracking the released v2.1 checkpoint's own recipe. Omitting `others_lr` would collapse both to a single rate and train the 141M-parameter encoder at the head's rate -- on a small dataset that is a catastrophic-forgetting risk. Set `freeze_text_encoder: true` to freeze the encoder outright for the smallest runs; the head (`others_lr`) then carries all adaptation.

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
also matches inference, where both labels are always queried, and makes the loss independent
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

## Bundling

`medliner bundle` hard-requires the evaluation report, the normalized dataset, and the split
manifest: a bundle without them ships a checkpoint with no evidence. Run `medliner evaluate`
(automatically included in `medliner pipeline`) before bundling; a standalone bundle built without
it now fails loudly instead of silently omitting the files.

## Selection and stopping semantics

- `eval_strict_f1` is computed by `ValidationF1Callback` over the reviewed validation split only.
  A validation split with no annotated examples is rejected at trainer construction: strict F1 is
  structurally `0.0` there, so it could neither rank checkpoints nor make early stopping meaningful.
- Over-budget validation text is truncated by GLiNER; the callback now prints a warning naming how
  many examples exceed the model's `max_len` word budget, because truncation silently depresses the
  metric used for selection.
- The `strict` metrics dicts carry `measurable: false` when a slice has no gold and no predictions
  (counts `0/0/0`). F1 there is reported as `0.0` but is not evidence about the model; consumers
  that rank on a slice should check `measurable` first.

## Fine-tuning practices (evidence-based)

Sourced from the official GLiNER repo/docs and the author's issue-track guidance; treat the numbered
items as the defaults to start from on this dataset's scale.

- **Separate encoder/head learning rates** (see above). The single most-gotten-wrong knob.
  [GLiNER `examples/finetune.ipynb`; `gliner/training/trainer.py#create_optimizer`]
- **Select on F1, never on `eval_loss`.** With `loss_reduction="sum"` the loss scales with
  sequence length x span count and is not comparable across checkpoints; the author recommends the
  F1 score instead. [urchade/GLiNER issue #163]
- **Catastrophic forgetting is the main risk on small supervised sets.** The author recommends
  mixing in Pile-NER replay data at roughly 2x the task data (a 200-example user needed ~5x),
  resampled each epoch. [urchade/GLiNER issue #163] MedliNER's down-weighted synthetic pool
  (`synthetic_weight`) plays the analogous regularizer role; `--no-synthetic` removes it.
- **Freeze the encoder on the smallest runs** (`freeze_text_encoder: true`): the official docs
  recommend it for small datasets; adaptation then happens entirely in the span/prompt stack at
  `others_lr`. [urchade.github.io/GLiNER/training.html]
- **Prefer a short, step-bounded schedule and early stopping.** The official fine-tuning notebook
  uses `max_steps=500` at batch 8, and a community reproduction of the large model found quality
  peaked well before the full schedule. [finetune.ipynb; urchade/GLiNER issue #209]
- **Audit gold spans wider than `max_width` (12 tokens).** GLiNER never enumerates such spans as
  candidates, so they silently contribute no supervision; MedliNER refuses them at conversion time
  (see Conversion budgets). [gliner/data_processing/utils.py#prepare_span_idx]
- **Pin `gliner` and treat `transformers` as bound by it.** `gliner 0.2.28` is exactly pinned; the
  `eval_strategy`/`evaluation_strategy` rename is handled at runtime (`_eval_strategy_field`), but a
  future `transformers` major is gated by gliner's own requirement, not by MedliNER's.
