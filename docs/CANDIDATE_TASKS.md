# Candidate task creation

The recommended source is the DAKP training-data export bundle: run `make ingest` and it
materializes raw candidates at `$MEDLINER_WORKDIR/ingested/candidates.ndjson`. You can also
author the small NDJSON file yourself, typically derived from intermediate DAKP inputs
(DailyMed SPL section text, FAERS indication strings). Running
`make candidates` validates that file, deduplicates it,
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

## Generation rules applied by `make candidates`

- Task IDs are deterministic: `medliner-<blake3(task + normalized text)[:16]>`, so re-running
  over the same input reproduces the same import file and Label Studio re-imports are
  recognizable.
- Rows are deduplicated on normalized text + task; the first occurrence wins and the merged
  task records a `duplicate_count` in its metadata.
- Tasks are plain text only. `make candidates` never runs a model; pre-annotations are a
  separate opt-in stage (`make prelabel`, see `docs/LABEL_STUDIO.md`) so this stage stays
  deterministic, offline, and free of ML dependencies.
- Each task carries `generator_version` and `generated_at` in its `data` payload.
- A `*.manifest.json` next to the import file records the BLAKE3 input hash and per-task /
  per-family counts.

## Sampling and task staggering (import build)
Full candidate pools are usually far larger than the annotation budget (the first DAKP export
held 93,328 rows), so `make candidates` samples a bounded, balanced subset and interleaves it
before import. Configuration is environment-only:

| Variable | Default | Meaning |
|---|---|---|
| `MEDLINER_SAMPLE_TASKS` | `indication:6000,contraindication:4000` | `task:count` pairs. Unlisted tasks are dropped; empty or `all` disables sampling entirely. |
| `MEDLINER_SAMPLE_SEED` | `2026` | Seed folded into the per-task `blake3` rank; changing it picks a different subset. |
| `MEDLINER_SAMPLE_MAX_WORDS` | `300` | Drops texts longer than this many whitespace words. `0` disables the cap. |
| `MEDLINER_SAMPLE_MAX_RUN` | `3` | Cap on consecutive tasks sharing one task value in the import order. |

Behavior:

- The 6,000/4,000 default yields ~10K tasks with more indications than contraindications —
  closer to the real-world mix than the raw 81/19 pool while still boosting the minority task
  for fine-tuning. Adjust the pair to rebalance.
- Sampling happens **after** deduplication, so repeated FAERS strings cannot burn multiple slots.
- Within a task, selection is stratified across `source_family` proportionally (indications
  split dailymed/faers at the pool's ratio), so both families stay represented.
- Texts over `max_words` are dropped before selection: GLiNER conversion refuses examples
  beyond `max_length` (`configs/train-small.yaml`), so annotating them is wasted effort.
- Selection and ordering are deterministic (`blake3` ranks) — same input + same env always
  reproduce the same import file, and the import filename is keyed on the input hash **and**
  the sampling config, so changing the config cannot silently reuse a stale import.
- Ordering interleaves task values *and* families so labelers working top-to-bottom see a mix;
  when one task type outnumbers the others by more than `max_run`, an unbounded tail of the
  majority type is unavoidable after the others are exhausted.
- The manifest records a `sampling` block (targets, seed, caps, and the pre-sampling pool
  counts) for auditability.

## Raw-pool authoring guidance

- Keep `task=indication` and `task=contraindication` balanced enough for review and evaluation.
- Include positive examples and deliberately empty/no-entity examples.
- Include short and long text, multiword qualified conditions, and conjunctions. Medication
  names and dosage/route phrases are useful precisely because they are distractors: they get
  no span (`docs/ANNOTATION_GUIDE.md` rule 4), so the model has to learn to leave them alone.
- Keep repeated sentences from one source document together for later leakage-safe splitting.

## Next step

Optionally run `make prelabel` to attach GLiNER suggestions, then `make label-studio` (add
`PRELABEL=1` to import the pre-labeled file) to serve these tasks in a browser; see
`docs/LABEL_STUDIO.md`. Model suggestions are suggestions only and never training gold.
