# Implementation checklist and verification

## Files

- `src/medliner/schema.py`: canonical contracts and policy enums.
- `src/medliner/candidates.py`: raw candidate JSONL validation, BLAKE3 task IDs, dedupe, and Label Studio import generation.
- `src/medliner/label_studio_server.py`: podman container lifecycle and stdlib Label Studio REST client (project + task import).
- `src/medliner/label_studio.py`: reviewed Label Studio JSON/JSONL adapter.
- `src/medliner/gliner_data.py`: character-to-token span conversion.
- `src/medliner/dataset.py`, `splits.py`: JSONL, manifests, grouped splits, hashes.
- `src/medliner/training.py`: small GLiNER training and resume.
- `src/medliner/evaluation.py`: strict/lenient metrics and baselines.
- `src/medliner/packaging.py`: standalone bundle.
- `src/medliner/dagster_defs.py`: local asset graph.
- `configs/label_studio_ner.xml`, `configs/train-small.yaml`: external annotation and training configuration.
- `configs/dagster.yaml`: local Dagster instance config, copied into the gitignored `$DAGSTER_HOME` by `make UP`/`make validate`.
- `tests/test_contracts.py`: canonical schema contracts and deterministic split behaviour.
- `tests/test_candidates.py`: candidate validation, deterministic IDs, dedupe, and import shape.
- `tests/test_label_studio_server.py`: mocked podman/API lifecycle, auth, project, and import behavior.
- `tests/test_label_studio.py`: export adapter, offsets, whitespace, overlap, and review status.
- `tests/test_gliner_data.py`: token conversion and the `max_len`/`max_width` budgets.
- `tests/test_training.py`: training arguments, precision, collator, and checkpoint selection.
- `tests/test_evaluation.py`: metrics, truncation reporting, and baseline fallbacks.
- `tests/test_packaging.py`, `tests/test_dataset.py`, `tests/test_dagster.py`.

## Commands

```bash
direnv allow
make sync
make check       # tests, lint, format, Dagster definitions
make validate    # Dagster definitions only, without starting a server
make coverage    # tests with a coverage report
make env         # resolved pipeline environment

# Verify the target CUDA wheel before training
uv run python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda)
print(torch.cuda.get_device_capability(), torch.cuda.get_arch_list())
assert "sm_120" in torch.cuda.get_arch_list()
PY

# Local Dagster deployment/UI — all pipeline stages run here
make UP
# One-step smoke test: materialize `training_run` with run config {"smoke": true}
```

## End-to-end gate

0. Author `data/label-studio/candidates.jsonl` from intermediate DAKP inputs and materialize
   `label_studio_server`; confirm http://localhost:9030 shows the project with the imported tasks.
1. Annotate in the browser and export the reviewed JSON manually; confirm annotators can
   highlight text without entering offsets.
2. Materialize `label_studio_export` → `normalized_dataset` and inspect rejected spans/errors.
4. Materialize frozen splits and verify the split manifest hash.
5. Run the one-step smoke test and confirm `training/final/` is loadable with `GLiNER.from_pretrained` and records `selected_checkpoint`.
6. Run bounded training with resume enabled.
7. Evaluate tuned, untuned, gazetteer, and DAKP regression metrics.
8. Build the standalone bundle and inspect its provenance/model-card files.

The committed DAKP benchmark is a regression test only and must not appear in train, validation, or test training files.
