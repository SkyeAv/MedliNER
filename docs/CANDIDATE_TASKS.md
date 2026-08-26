# Candidate task creation

The recommended source is a DAKP/MedliNER export directory configured in the ignored
`.envrc.local`: its `candidates.ndjson` already matches the raw candidates contract below. The
older DAKP bundle layout is still supported via
`uv run medliner ingest --bundle <dir>`, which materializes raw candidates at
`$MEDLINER_WORKDIR/ingested/candidates.ndjson`. You can also
author the small NDJSON file yourself, typically derived from intermediate DAKP inputs
(DailyMed SPL section text, FAERS indication strings). Running
`make prepare` validates that file, deduplicates it,
and converts it into the Label Studio import shape documented in `docs/LABEL_STUDIO.md`.
Reviewed exports remain the only training input.

## Raw candidates contract

One JSON object per line, at `$MEDLINER_RAW_CANDIDATES` (default
`data/label-studio/candidates.ndjson`):

```json
{"text": "Contraindicated in patients with pulmonary hypertension.", "task": "contraindication", "source_family": "dailymed", "source_document_id": "spl-document-001", "section": "34070-3"}
{"text": "Indicated for the treatment of partial-onset seizures.", "task": "indication", "source_family": "dailymed", "source_document_id": "spl-document-002", "section": "34067-9"}
{"text": "CHRONIC KIDNEY DISEASE", "task": "indication", "source_family": "faers", "source_record_id": "case-12345"}
```

Fields:

- `text` (required, non-empty): the exact text the annotator will highlight.
- `task` (required): `indication` or `contraindication`.
- `source_family` (optional, default `unknown`): e.g. `dailymed`, `faers`.
- `source_document_id` / `source_record_id` (optional): stable upstream IDs; these drive
  leakage-safe grouped splitting later, so fill them in whenever the source has a document
  or case identity.
- `section`, `source_uri`, `source_hash` (optional): extra provenance, preserved into the
  imported task and the normalized dataset.

Deriving rows from DAKP intermediates: pull section text from the DailyMed SPL inputs
(contraindication sections `LOINC 34070-3`, indications-and-usage `LOINC 34067-9`) and
indication strings from the FAERS case table, and write one row per text. No DAKP runtime
is required — the file is plain NDJSON and can come from a notebook, a SQL export, or a
script against DAKP's intermediate artifacts.

## Generation rules applied by `make prepare`

- Task IDs are deterministic: `medliner-<blake3(task + normalized text)[:16]>`, so re-running
  over the same input reproduces the same import file and Label Studio re-imports are
  recognizable.
- Rows are deduplicated on normalized text + task; the first occurrence wins and the merged
  task records a `duplicate_count` in its metadata.
- Tasks are plain text only. The candidates stage never runs a model; pre-annotations are a
  separate step (`make prepare` also runs `medliner prelabel`; see `docs/LABEL_STUDIO.md`) so
  candidate selection stays deterministic, offline, and free of ML dependencies.
- Each task carries `generator_version` and `generated_at` in its `data` payload.
- A `*.manifest.json` next to the import file records the BLAKE3 input hash and per-task /
  per-family counts.

## Sampling and task staggering (import build)
Full candidate pools are usually far larger than the annotation budget (the first DAKP export
held 93,328 rows), so `make prepare` samples a bounded, balanced subset and interleaves it
before import. Configuration is environment-only:

| Variable | Default | Meaning |
|---|---|---|
| `MEDLINER_SAMPLE_TASKS` | `indication:600,contraindication:400` | `task:count` pairs. Unlisted tasks are dropped; empty or `all` disables sampling entirely. |
| `MEDLINER_SAMPLE_SEED` | `2026` | Seed folded into the per-task `blake3` rank; changing it picks a different subset. |
| `MEDLINER_SAMPLE_MAX_WORDS` | `300` | Drops texts longer than this many whitespace words. `0` disables the cap. |
| `MEDLINER_SAMPLE_MAX_RUN` | `3` | Cap on consecutive tasks sharing one task value in the import order. |
| `MEDLINER_SAMPLE_EDGE_FRACTION` | `0.8` | Share of each stratum filled with the highest-difficulty texts; the rest stays hash-random as a control slice. `0` restores pure random selection. |

Behavior:

- The 600/400 default yields ~1K tasks with more indications than contraindications —
  sized for a limited SME annotation session, closer to the real-world mix than the raw
  81/19 pool while still boosting the minority task for fine-tuning. Adjust the pair to
  rebalance.
- **Edge cases first.** With `edge_fraction` above 0, most of each stratum is filled by the
  highest-`difficulty_score` texts rather than random ones. The score counts the patterns
  the annotation guide calls out as traps — hedges to exclude, population descriptors that
  are not entities, maximal-span modifiers, coordination lists, dosages, abbreviations,
  negation — plus a length component. The remaining `1 - edge_fraction` share stays
  hash-random so the batch keeps distribution coverage and an honest control slice.
- Sampling happens **after** deduplication, so repeated FAERS strings cannot burn multiple slots.
- Within a task, selection is stratified across `source_family` proportionally (indications
  split dailymed/faers at the pool's ratio), so both families stay represented.
- Texts over `max_words` are dropped before selection: GLiNER truncates texts beyond its
  `max_len` word budget with only a warning, so annotating them is wasted effort. To
  recover those rows instead, see "LLM shortening" below.
- Selection and ordering are deterministic (`blake3` ranks) — same input + same env always
  reproduce the same import file, and the import filename is keyed on the input hash **and**
  the sampling config, so changing the config cannot silently reuse a stale import.
- Ordering interleaves task values *and* families so labelers working top-to-bottom see a mix;
  when one task type outnumbers the others by more than `max_run`, an unbounded tail of the
  majority type is unavoidable after the others are exhausted.
- The manifest records a `sampling` block (targets, seed, caps, `edge_fraction`, pool counts,
  and pool-vs-selected mean difficulty) for auditability.

## LLM shortening (sampled batch)

`make prepare` shortens automatically as part of building the import file: after sampling, every sampled text over
`MEDLINER_SHORTEN_MAX_WORDS` words (default 48, ≈ 3-4 short sentences) is rewritten by the local LLM (Ornith-1.0-9B,
served from the directory configured in `MODELS_DIR` with `make medliner`) into a shorter text that keeps every
condition mention verbatim. Only the sampled ~1k batch is sent to the model — never the whole candidate pool — and if
the LLM is not running, prepare skips shortening with a notice and long texts stay as-is:

```bash
make llm            # optional; without it prepare just skips the shortening step
make prepare        # sample → shorten (LLM) → attach GLiNER suggestions
make llm-stop
```

Every rewrite is validated (non-empty, actually shorter) and failures keep the original text, all counted in the import
manifest's `sampling.llm_shorten` block. Texts are sent up to `MEDLINER_SHORTEN_WORKERS` at a time (default 4, matching
the server's four slots). Successful replies are cached in `$MEDLINER_SHORTEN_CACHE` (sqlite), so re-running prepare or
overlapping texts never hit the model twice. Rows the model judges to contain no condition mention are recorded as
`empty_hints` — a review signal only, never a drop decision.

The standalone `medliner shorten --input <file>` remains available for rewriting an arbitrary candidates file in place
(with manifest-based resume via `--limit`; `--force` re-runs everything); `make shorten LIMIT=8 MAX_WORDS=48` wraps it.

## Raw-pool authoring guidance

- Keep `task=indication` and `task=contraindication` balanced enough for review and evaluation.
- Include positive examples and deliberately empty/no-entity examples.
- Include short and long text, multiword qualified conditions, and conjunctions. Medication
  names and dosage/route phrases are useful precisely because they are distractors: they get
  no span (`docs/ANNOTATION_GUIDE.md` rule 4), so the model has to learn to leave them alone.
- Keep repeated sentences from one source document together for later leakage-safe splitting.

## Next step

`make prepare` also attaches GLiNER suggestions (`medliner prelabel`), then `make annotate`
serves these tasks in a browser (the pre-labeled file is picked up automatically); see
`docs/LABEL_STUDIO.md`. Model suggestions are suggestions only and never training gold.
