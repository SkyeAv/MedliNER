# Implementation checklist and verification

## Files

- `src/medliner/schema.py`: canonical contracts and policy enums.
- `src/medliner/label_studio.py`: reviewed Label Studio JSON/JSONL adapter.
- `src/medliner/gliner_data.py`: character-to-token span conversion.
- `src/medliner/dataset.py`, `splits.py`: JSONL, manifests, grouped splits, hashes.
- `src/medliner/training.py`: small GLiNER training and resume.
- `src/medliner/evaluation.py`: strict/lenient metrics and baselines.
- `src/medliner/packaging.py`: standalone bundle.
- `src/medliner/dagster_defs.py`: local asset graph.
- `configs/label_studio_ner.xml`, `configs/train-small.yaml`: external annotation and training configuration.
- `tests/`: contract, adapter, split, evaluation, and Dagster graph tests.

## Commands

```bash
direnv allow
make sync
make check

# Verify the target CUDA wheel before training
uv run python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda)
print(torch.cuda.get_device_capability(), torch.cuda.get_arch_list())
assert "sm_120" in torch.cuda.get_arch_list()
PY

# One-step smoke test
uv run medliner train data/splits data/smoke-training --config configs/train-small.yaml --smoke

# Local Dagster deployment/UI
make UP
```

## End-to-end gate

1. Import reviewed Label Studio export into the UI.
2. Confirm annotators can highlight text without entering offsets.
3. Run normalization and inspect rejected spans/errors.
4. Materialize frozen splits and verify the split manifest hash.
5. Run the one-step smoke test and confirm a checkpoint is loadable.
6. Run bounded training with resume enabled.
7. Evaluate tuned, untuned, gazetteer, and DAKP regression metrics.
8. Build the standalone bundle and inspect its provenance/model-card files.

The committed DAKP benchmark is a regression test only and must not appear in train, validation, or test training files.
