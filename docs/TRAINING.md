# Laptop-safe training

The first supported target is `urchade/gliner_small-v2.1`. This is deliberate; a successful small-model run is more useful than an OOM-prone large run.

## Required sequence

1. Materialize a reviewed dataset and frozen splits.
2. Run `uv run medliner train ... --smoke` for one batch/one step.
3. Confirm the checkpoint and loss are written.
4. Run the bounded training configuration.
5. Resume with the same output directory if interrupted; the trainer discovers the latest `checkpoint-*` directory.
6. Evaluate the final/best checkpoint on reviewed validation/test data.

The default configuration uses sequence length 384, micro-batch size 1, gradient accumulation 8, mixed precision on CUDA, gradient checkpointing where the encoder exposes it, five maximum epochs, and two retained checkpoints. Adjust sequence length downward before increasing batch size if VRAM is tight.

`bf16` is enabled by default for the RTX 5070 Ti, which supports it in the validated Torch environment. Use `fp16: true` instead if a different CUDA/Torch combination lacks BF16 support. CPU execution is supported for validation but is not a practical full training target.

## Checkpoint semantics

The output directory contains Hugging Face-compatible checkpoints and `final/medliner-training.json`. Resume records the prior checkpoint path and the exact configuration. Never overwrite a dataset or split manifest while resuming a run.

Training uses the GLiNER 0.2.28 bundled trainer API with canonical records converted to `tokenized_text` and inclusive token spans. A conversion test checks that every token span maps back to the original Label Studio character span.

## Large checkpoint

`gliner_large-v2.5` is optional. Do not make it a gate for the project. If evaluated later, begin with a one-step smoke test and consider freezing the text encoder; full Adam fine-tuning can exceed 12 GB VRAM even with mixed precision.
