# Implementation checklist and verification

## Files

- `src/medliner/schema.py`: canonical contracts and policy enums.
- `src/medliner/candidates.py`: raw candidate NDJSON validation, BLAKE3 task IDs, dedupe, and Label Studio import generation.
- `src/medliner/label_studio_server.py`: podman container lifecycle and stdlib Label Studio REST client (project + task import).
- `src/medliner/label_studio.py`: reviewed Label Studio JSON/JSONL adapter.
- `src/medliner/gliner_data.py`: character-to-token span conversion.
- `src/medliner/dataset.py`, `splits.py`: JSONL, manifests, grouped splits, hashes.
- `src/medliner/training.py`: small GLiNER training and resume.
- `src/medliner/evaluation.py`: strict/lenient metrics and baselines.
- `src/medliner/packaging.py`: standalone bundle.
- `src/medliner/cli.py`: argparse CLI for the pipeline stages (`uv run medliner <cmd>`, wrapped by the Makefile).
- `configs/label_studio_ner.xml`, `configs/train-small.yaml`: external annotation and training configuration.
- `tests/test_contracts.py`: canonical schema contracts and deterministic split behaviour.
- `tests/test_candidates.py`: candidate validation, deterministic IDs, dedupe, and import shape.
- `tests/test_label_studio_server.py`: mocked podman/API lifecycle, auth, project, and import behavior.
- `tests/test_label_studio.py`: export adapter, offsets, whitespace, overlap, and review status.
- `tests/test_gliner_data.py`: token conversion and the `max_len`/`max_width` budgets.
- `tests/test_training.py`: training arguments, precision, collator, and checkpoint selection.
- `tests/test_evaluation.py`: metrics, truncation reporting, and baseline fallbacks.
- `tests/test_packaging.py`, `tests/test_dataset.py`, `tests/test_cli.py`.

## Commands

```bash
direnv allow
make setup
make check       # tests, lint, format
make env         # resolved pipeline environment

# Verify the target CUDA wheel before training
uv run python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda)
print(torch.cuda.get_device_capability(), torch.cuda.get_arch_list())
assert "sm_120" in torch.cuda.get_arch_list()
PY

# Pipeline stages run through the Makefile/CLI
make data
make annotate
# One-step smoke run of the whole post-annotation chain before full training
SMOKE=1 make train
```

## End-to-end gate

0. Run `make data` to build the sampled import batch from the export
   (`$MEDLINER_RAW_CANDIDATES`, default `data/label-studio/candidates.ndjson`; set it in the ignored `.envrc.local` for a ready export)
   and run `make annotate`; confirm http://localhost:9030 shows the
   project with the imported tasks.
1. Annotate in the browser and export the reviewed JSON manually; confirm annotators can
   highlight text without entering offsets.
2. Run `make train` and inspect rejected spans/errors from the dataset stage.
4. Verify the split manifest hash under `$MEDLINER_WORKDIR/splits/`.
5. Run the one-step smoke run (`SMOKE=1 make train`) and confirm `training/final/` is loadable with `GLiNER.from_pretrained` and records `selected_checkpoint`.
6. Run full bounded training with resume enabled (`make train`).
7. Inspect the tuned, untuned, and gold-regression metrics in the evaluation report.
8. Inspect the standalone bundle's provenance/model-card files.

The ingested DAKP gold benchmark is a regression test only and must not appear in train, validation, or test training files.
