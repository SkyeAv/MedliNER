# RTX 5070 Ti environment

The target laptop GPU reports compute capability `sm_120`. The Torch wheel must include `sm_120` kernels. The sibling DAKP environment currently uses Torch `2.8.0+cu126`, whose architecture list stops at `sm_90`; attempting to move a GLiNER model to CUDA there fails with `CUDA error: no kernel image is available for execution on the device`.

MedliNER pins the Torch dependency range and selects the PyTorch cu130 index in `pyproject.toml` for the target laptop. Recreate/sync the MedliNER environment rather than reusing the DAKP virtualenv:

```bash
uv sync
uv run python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda)
print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
print(torch.cuda.get_arch_list())
assert "sm_120" in torch.cuda.get_arch_list()
PY
```

If the assertion fails, do not start training. Install a Torch CUDA wheel with `sm_120` support and verify the NVIDIA driver first. MedliNER detects this mismatch and falls back to CPU instead of triggering the kernel-image crash, but CPU is only suitable for data/evaluation checks, not the intended training run.

## Triton and the libcuda lookup

Torch's backward pass dispatches into Triton kernels, and Triton locates `libcuda.so.1` by
shelling out to `/sbin/ldconfig`. On systems that ship no loader cache (this laptop included)
that call fails with `FileNotFoundError: '/sbin/ldconfig'` — not at import time, but partway
through the first `loss.backward()`, which makes it look like a training bug.

Point Triton at the driver directory instead:

```bash
export TRITON_LIBCUDA_PATH=/run/opengl-driver/lib   # directory containing libcuda.so.1
```

`.envrc` and the `Makefile` both set this automatically when `/sbin/ldconfig` is absent and a
`libcuda.so.1` is found; an empty value restores Triton's default lookup, so the setting is
inert on systems with a normal loader cache. Check the resolved value with `make env`.
