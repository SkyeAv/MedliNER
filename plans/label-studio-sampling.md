# Label Studio task sampling, staggering, and rename

## Context

The Desktop DAKP export (`$MEDLINER_EXPORT_BUNDLE`) holds 93,328 candidates: 75,542 indication (57,968 dailymed + 17,574 faers) and 17,786 contraindication (all dailymed) — an 81/19 skew. `medliner candidates` currently imports **all** of them, which is far too many for the annotators and imbalanced for fine-tuning.

User requirements:

1. **~1K tasks** in Label Studio, **more indications than contraindications** but more even than the raw 81/19 → default **600 indication / 400 contraindication** (60/40), tunable via env.
2. **Staggered task order** so labelers working top-to-bottom don't hit long runs of one task type.
3. Project title **`MedliNER`** (drop " medical NER").
4. Empty/nothing-to-label question answered from existing policy (no code change; see "Empty tasks" below).

Two supporting facts from the code that shape the design:

- **Length cap is required, not cosmetic:** `configs/train-small.yaml` pins `max_length: 384` and `to_gliner_record` (`src/medliner/gliner_data.py`) *raises* on examples exceeding it — a labeled 640–3,704-word task (p99–max of this bundle) would hard-fail `train`. Filtering to ≤300 words keeps 93% of indications / 97% of contraindications.
- **Sampling must happen after dedupe:** `build_import_tasks` dedupes on normalized text+task; sampling deduped tasks prevents duplicate FAERS strings from burning multiple slots.

## Approach

Add pure, deterministic sampling/staggering functions to `candidates.py`, wire them into `run_candidates` behind env vars, and rename the project title everywhere.

### Sampling (`sample_tasks`)

1. Start from the deduped output of `build_import_tasks` (existing behavior preserved when sampling is disabled).
2. Drop tasks whose whitespace-word count exceeds `MEDLINER_SAMPLE_MAX_WORDS` (default **300**).
3. Within each `task`, stratify by `source_family` proportionally (indication 600 → ~460 dailymed / ~140 faers at current ratios) so FAERS short strings stay represented.
4. Deterministic selection: rank candidates by `blake3(f"{seed}:{task_id}")`, take the lowest ranks. No RNG state → same input + same config always yields the same import file (matching the repo's deterministic-by-hash style, e.g. `_task_id`, `split_examples`).

Config via env, parsed in `cli.py`:

- `MEDLINER_SAMPLE_TASKS` — `indication:600,contraindication:400` (default). Empty string / `all` disables sampling (current full-import behavior).
- `MEDLINER_SAMPLE_SEED` — default `2026` (matches split seed convention).
- `MEDLINER_SAMPLE_MAX_WORDS` — default `300`; `0` disables the cap.

### Staggering (`stagger_tasks`)

After sampling, order tasks so no more than `MEDLINER_SAMPLE_MAX_RUN` (default **3**) consecutive tasks share a `task` value:

1. Deterministic hash order within each `(task, source_family)` stratum.
2. Weighted round-robin interleave of strata (indications-faers, indications-dailymed, contraindications) so families mix too, not just tasks.
3. Greedy spread pass that swaps an offending task with the next eligible later task when a run exceeds the cap; pure index swaps keep it deterministic.

### Import-file naming + manifest

The import filename is currently `import-{input_hash[:16]}.json`; content now depends on sampling config, so fold it in: `import-{blake3(input_hash + sampling_config)[:16]}.json` via one shared helper used by both `run_candidates` and `ensure_import_file` (a changed env config must not silently reuse a stale 93K import). `import_manifest` gains a `sampling` block (per-task targets, seed, max_words, max_run, pre/post counts per task and family) so the annotation batch is auditable.

### Project title

- `DEFAULT_PROJECT_TITLE = "MedliNER"` and `WARMUP_PROJECT_TITLE = "MedliNER — Warm-up"` in `label_studio_server.py`.
- `configs/label_studio_ner.xml`: `<Header value="MedliNER" />`.
- Note for the user: `ensure_project` finds projects by exact title, so the first `make label-studio` after this change creates a fresh project; the old "MedliNER medical NER" project stays in the server data dir, unused.

### Empty tasks (answer, no code change)

Already solved by the repo's annotation policy:

- **Text genuinely has no allowed entity** → annotator **submits an empty annotation**. That is *positive, deliberate* negative signal ("nothing to label here") and the pipeline fully supports it — `FixedLabelCollator` (`training.py`) exists precisely so empty examples train correctly instead of crashing the loss. See `docs/ANNOTATION_GUIDE.md` §6.
- **Task is unusable/garbage** → annotator **skips** it. Skipped tasks are *rejected at import* (`_annotation_sets` in `label_studio.py` raises on all-cancelled annotations) so they never become silent fake negatives. See guide §10.

## Files to modify

- `src/medliner/candidates.py` — add `sample_tasks`, `stagger_tasks`, `import_file_name` helper; extend `import_manifest`.
- `src/medliner/cli.py` — parse `MEDLINER_SAMPLE_*` env vars; apply in `run_candidates`/`ensure_import_file`; print sampled counts per task/family.
- `src/medliner/label_studio_server.py` — the two title constants.
- `configs/label_studio_ner.xml` — header text.
- `tests/test_candidates.py` — sampling determinism, stratification, word cap, stagger max-run bound, disabled-mode passthrough, manifest block.
- `tests/test_cli.py` — env parsing, filename sensitivity to sampling config, updated title assertions (currently expect `"MedliNER medical NER"`).
- `tests/test_label_studio_server.py` — no functional change expected (uses the constant) but verify.
- `docs/LABEL_STUDIO.md`, `docs/CANDIDATE_TASKS.md` — new titles, sampling env vars, 1K default.

## Reuse

- `build_import_tasks` / `_normalized` / `_task_id` (`src/medliner/candidates.py`) — dedupe precedes sampling.
- `blake3` hashing pattern (`hash_candidates_file`, `_task_id`) — deterministic ranking seed.
- `import_manifest` counters (`Counter(task)`, `Counter(source_family)`) — extended, not replaced.
- `ensure_import_file` hash-name convention (`src/medliner/cli.py`) — sampling-aware filename slots straight in.
- `FixedLabelCollator` (`src/medliner/training.py`) — already makes the empty-annotation workflow safe; nothing new needed.

## Steps

- [ ] Add `sample_tasks` + `stagger_tasks` + `import_file_name` to `candidates.py` with deterministic hash ordering.
- [ ] Extend `import_manifest` with the `sampling` block.
- [ ] Wire env parsing + application into `cli.py` (`run_candidates`, `ensure_import_file`), update the printed summary.
- [ ] Rename project titles in `label_studio_server.py` and the label config header.
- [ ] Update/add tests (`test_candidates.py`, `test_cli.py`; verify `test_label_studio_server.py`).
- [ ] Update `docs/LABEL_STUDIO.md` and `docs/CANDIDATE_TASKS.md`.
- [ ] Run `uv run pytest` and `make check`.

## Verification

1. `uv run pytest` — new and updated tests green.
2. `uv run medliner candidates` against the Desktop bundle → manifest shows 1,000 tasks (≈600/400), sampling block correct; output filename differs from the unsampled name.
3. Spot-check staggering: `jq -r '.[].data.task' <import>.json | awk` max consecutive same-task run ≤ 3, and dailymed/faers indications interleave.
4. Word cap: `jq` max whitespace-word count over the import file ≤ 300.
5. `MEDLINER_SAMPLE_TASKS= uv run medliner candidates` still produces the full 93,328-task import (backward compatible).
6. Optional manual: `make label-studio` creates the `MedliNER` project with 1,000 tasks (fresh title, so a new project appears).
