# MEDliNER

MEDliNER is a standalone, local pipeline for producing a reviewed medical NER dataset and fine-tuning a small GLiNER checkpoint. Its first use cases are extracting `disease`, `phenotype`, and `drug` mentions from `indication` and `contraindication` text.

DAKP runtime integration is deliberately deferred. The only training input is a reviewed Label Studio export.

## Labels and tasks

Entity labels:

- `disease`
- `phenotype`
- `drug`

Every example also has task metadata:

- `indication`
- `contraindication`

The task is context metadata, not an entity label. GLiNER can be queried with all three labels, while evaluation reports indication and contraindication separately.

Read [`docs/ANNOTATION_GUIDE.md`](docs/ANNOTATION_GUIDE.md) before annotation.

## Annotation

Label Studio Community Edition is free to self-host locally and provides a browser UI. Install it separately using pip or Docker; see [`docs/LABEL_STUDIO.md`](docs/LABEL_STUDIO.md).

Annotators do not count offsets:

> open task → click-drag/highlight condition or drug phrase → choose label → submit

Label Studio records character offsets automatically. MEDliNER validates those offsets and converts them to GLiNER token spans.

## Install and run MEDliNER

Review the safe example paths in `.envrc`, then enable direnv:

```bash
direnv allow
make sync
```

The checked-in `.envrc` exports `MEDLINER_LABEL_STUDIO_EXPORT`, `MEDLINER_WORKDIR`, `MEDLINER_TRAIN_CONFIG`, and `DAGSTER_HOME`. For private local overrides, create the ignored `.envrc.local`; do not put secrets or machine-specific paths into the committed `.envrc`.

Label Studio is intentionally not a MEDliNER dependency. Start the local Dagster deployment/UI with the phony `UP` target:

```bash
make UP
# `make up` is a lowercase convenience alias.
```

Materialize assets from the UI, or use the Makefile stages:

```bash
make normalize
make split
make smoke     # required first GPU check
make train
make evaluate
make bundle
```

The smoke run performs one training step using the same code path as the full run. Override any environment path without editing files, for example:

```bash
MEDLINER_LABEL_STUDIO_EXPORT=$PWD/data/label-studio/reviewed.json make UP
```

## Dagster graph

The initial graph is intentionally small:

```text
Label Studio export
        ↓
validated/normalized dataset
        ↓
frozen grouped splits
        ↓
small-GLiNER training run
        ↓
evaluation report
        ↓
standalone export bundle
```

There are no schedules, sensors, deployment services, or remote executors. Dagster records materializations and metadata under the local run storage. The source export, normalized JSONL, split manifest, training configuration, checkpoint, and evaluation report are all explicit artifacts.

## Training target

The default base model is `urchade/gliner_small-v2.1`. A smaller checkpoint is intentional: full training of a large GLiNER checkpoint is not required for this project and may exceed 12 GB VRAM. The configuration uses batch size 1, gradient accumulation, bounded sequence length, mixed precision when supported, checkpointing, and resume behavior. Large-checkpoint training is an optional later comparison. See [`docs/HARDWARE.md`](docs/HARDWARE.md): the RTX 5070 Ti requires a Torch wheel containing `sm_120` kernels.

GLiNER 0.2.28's training records use model tokens and inclusive token end indexes. MEDliNER preserves the original character-level annotations so this conversion is auditable and tested.

## Evaluation gates

Reports include:

- strict exact `(start, end, label)` precision/recall/F1;
- lenient boundary-only F1;
- indication versus contraindication metrics;
- source-family metrics;
- no-entity false-positive rate;
- DAKP regression benchmark results when `../DAKP/tests/eval/ner_gold.json` is available.

The tuned model must be compared with the untuned small checkpoint and the DAKP gazetteer baseline before it is selected for packaging. The committed DAKP benchmark remains held out from training.

## Artifact bundle

`medliner bundle` creates a standalone directory containing:

- `checkpoint/`;
- `labels.json`;
- `annotation_policy.md`;
- `dataset.jsonl` and split manifest;
- training configuration and metadata;
- `metrics.json`;
- provenance/license notes;
- model-card inputs.

Review licensing for source text and the base checkpoint before uploading anything publicly.
