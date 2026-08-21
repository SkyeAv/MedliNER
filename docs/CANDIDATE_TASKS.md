# Candidate task creation

The recommended source is the DAKP training-data export bundle: run `make ingest` and it
materializes raw candidates at `$MEDLINER_WORKDIR/ingested/candidates.jsonl`. You can also
author the small JSONL file yourself, typically derived from intermediate DAKP inputs
(DailyMed SPL section text, FAERS indication strings). Running
`make candidates` validates that file, deduplicates it,
and converts it into the Label Studio import shape documented in `docs/LABEL_STUDIO.md`.
Reviewed exports remain the only training input.

## Raw candidates contract

One JSON object per line, at `$MEDLINER_RAW_CANDIDATES` (default
`data/label-studio/candidates.jsonl`):

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
is required — the file is plain JSONL and can come from a notebook, a SQL export, or a
script against DAKP's intermediate artifacts.

## Generation rules applied by `make candidates`

- Task IDs are deterministic: `medliner-<blake3(task + normalized text)[:16]>`, so re-running
  over the same input reproduces the same import file and Label Studio re-imports are
  recognizable.
- Rows are deduplicated on normalized text + task; the first occurrence wins and the merged
  task records a `duplicate_count` in its metadata.
- Tasks are plain text only — no pre-annotations or `predictions` are generated.
- Each task carries `generator_version` and `generated_at` in its `data` payload.
- A `*.manifest.json` next to the import file records the BLAKE3 input hash and per-task /
  per-family counts.

## Sampling guidance

- Keep `task=indication` and `task=contraindication` balanced enough for review and evaluation.
- Include positive examples and deliberately empty/no-entity examples.
- Include short and long text, multiword qualified conditions, conjunctions, medication
  mentions, and dosage/route distractors.
- Keep repeated sentences from one source document together for later leakage-safe splitting.

## Next step

Run `make label-studio` to serve these tasks in a browser; see
`docs/LABEL_STUDIO.md`. Model suggestions, if ever added later, are suggestions only and
never training gold.
