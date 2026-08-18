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

Label Studio Community Edition is free to self-host locally and provides a browser UI. The
pipeline runs it in a podman container via the `label_studio_server` asset — no separate
install needed; see [`docs/LABEL_STUDIO.md`](docs/LABEL_STUDIO.md).

Annotators do not count offsets:

> open task → click-drag/highlight condition or drug phrase → choose label → submit

Label Studio records character offsets automatically. MEDliNER validates those offsets and converts them to GLiNER token spans.

## Install and run MEDliNER

Review the safe example paths in `.envrc`, then enable direnv:

```bash
direnv allow
make sync
```

The checked-in `.envrc` exports:

| Variable | Purpose |
| --- | --- |
| `MEDLINER_RAW_CANDIDATES` | raw candidates JSONL you author from DAKP intermediates (see [`docs/CANDIDATE_TASKS.md`](docs/CANDIDATE_TASKS.md)) |
| `MEDLINER_LABEL_STUDIO_EXPORT` | reviewed export that feeds the pipeline |
| `MEDLINER_WORKDIR` | root for normalized data, splits, checkpoints, and reports |
| `MEDLINER_TRAIN_CONFIG` | training configuration YAML |
| `MEDLINER_DAKP_ROOT` | sibling DAKP checkout used for baselines and the regression fixture |
| `MEDLINER_LABEL_STUDIO_PORT` / `_IMAGE` | podman Label Studio container port and image |
| `MEDLINER_LABEL_STUDIO_USERNAME` / `_PASSWORD` / `_TOKEN` | Label Studio login created on first container boot, or an explicit API token |
| `MEDLINER_SPLIT_SEED` / `MEDLINER_REGRESSION_IDS` | split seed and IDs withheld from every split |
| `DAGSTER_HOME` | local Dagster run storage |
| `TRITON_LIBCUDA_PATH` | set automatically when the system has no `/sbin/ldconfig` (see [`docs/HARDWARE.md`](docs/HARDWARE.md)) |

Print the resolved values with `make env`. For private local overrides, create the ignored `.envrc.local`; do not put secrets or machine-specific paths into the committed `.envrc`.

Label Studio runs in a podman container started by the pipeline; it is intentionally not a
MEDliNER Python dependency. The Dagster UI is the only pipeline entry point. Start the local
deployment with the phony `UP` target:

```bash
make UP
# `make up` is a lowercase convenience alias.
```

Then open <http://localhost:3000>. The full flow is:

1. Author `data/label-studio/candidates.jsonl` from intermediate DAKP inputs
   ([`docs/CANDIDATE_TASKS.md`](docs/CANDIDATE_TASKS.md)).
2. Materialize `label_studio_server` — this builds the Label Studio import file
   (`candidate_tasks`) and starts the annotation server with those tasks imported.
3. Annotate in the browser at <http://localhost:9030>, export the reviewed JSON manually,
   and point `MEDLINER_LABEL_STUDIO_EXPORT` at it (default
   `data/label-studio/reviewed.json`).
4. Materialize `export_bundle`; the remaining assets (`label_studio_export` →
   `normalized_dataset` → `frozen_splits` → `training_run` → `evaluation_report`)
   materialize automatically in order.

Stop the annotation server with `make label-studio-stop`; annotations survive in the
container's data volume directory under `$MEDLINER_WORKDIR/label-studio/server-data`.

The required first GPU check is a one-step smoke run of `training_run` using the same code
path as full training: materialize that asset with run config `{"smoke": true}` from the
launchpad (Shift-click "Materialize" to open it) before launching a full run.

`make check` runs the tests, lint, format checks, and `dagster definitions validate`;
`make coverage` adds a coverage report. `make UP` and `make validate` seed `$DAGSTER_HOME` from
the committed `configs/dagster.yaml`, since the instance directory itself is gitignored.

Override any environment path without editing files, for example:

```bash
MEDLINER_LABEL_STUDIO_EXPORT=$PWD/data/label-studio/reviewed.json make UP
```

## Dagster graph

The graph covers the whole workflow except the human annotation step itself:

```text
raw candidates JSONL (authored from DAKP intermediates)
        ↓
Label Studio import tasks (validated, deduplicated)
        ↓
Label Studio server (podman container + project + import)
        ↓
[human annotation in the browser → manual JSON export]
        ↓
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

Two GLiNER behaviours would otherwise discard supervision without saying so, and are turned into
errors at conversion time: text beyond `max_len` (384 word tokens) is truncated with only a
warning, and a gold span wider than `max_width` (12 word tokens) is never enumerated as a span
candidate. The batch entity vocabulary is also pinned to the three labels, because a batch of
purely no-entity examples otherwise has zero entity types and the loss fails outright. See
[`docs/TRAINING.md`](docs/TRAINING.md).

## Evaluation gates

Reports include:

- strict exact `(start, end, label)` precision/recall/F1;
- lenient boundary-only F1;
- indication versus contraindication metrics;
- source-family metrics;
- no-entity false-positive rate;
- DAKP regression benchmark results when `$MEDLINER_DAKP_ROOT/tests/eval/ner_gold.json` is available;
- a truncation block naming any example over the model's word budget.

The tuned model must be compared with the untuned small checkpoint and the DAKP gazetteer baseline before it is selected for packaging. The committed DAKP benchmark remains held out from training.

## Artifact bundle

The `export_bundle` asset creates a standalone directory containing:

- `checkpoint/`;
- `labels.json`;
- `annotation_policy.md`;
- `dataset.jsonl` and split manifest;
- `training_config.yaml` — the configuration the run actually used, taken from the checkpoint's
  own metadata rather than the current contents of `configs/`;
- `metrics.json`;
- `provenance.json` — base model, selected checkpoint, best validation F1, dataset/metrics/tree
  hashes, split hash, and held-out example IDs;
- model-card inputs.

Rebuilding over a previous bundle is allowed; the build refuses to delete a non-empty directory
that is not a bundle.

Review licensing for source text and the base checkpoint before uploading anything publicly.
