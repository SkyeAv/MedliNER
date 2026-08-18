# Candidate task creation contract

Candidate text creation happens outside MEDliNER. The repository consumes only the reviewed Label Studio export.

## Recommended upstream sources

Create short, self-contained tasks from:

- DailyMed contraindication sections (`LOINC 34070-3`) and filtered contraindication sentences from indication/warning sections.
- DailyMed indications-and-usage sections (`LOINC 34067-9`).
- FAERS `indication` strings used for approved-treatment and observed-use contexts.

A candidate generator may be implemented in DAKP or a separate notebook/script. It should emit the Label Studio import shape documented in `docs/LABEL_STUDIO.md`, including `task`, source family, document/record IDs, section identifiers, and source hashes where available.

## Sampling

- Keep `task=indication` and `task=contraindication` balanced enough for review and evaluation.
- Deduplicate normalized FAERS indication strings, while retaining a count and representative source IDs in metadata.
- Include positive examples and deliberately empty/no-entity examples.
- Include short and long text, multiword qualified conditions, conjunctions, medication mentions, and dosage/route distractors.
- Keep repeated sentences from one source document together for later leakage-safe splitting.

## Optional pre-annotations

Existing DAKP gazetteer or GLiNER output may be included as Label Studio `predictions`. Predictions are useful for active-learning candidate selection and annotator speed, but they are never training gold by default. Human reviewers must accept, edit, delete, or add every final span. The reviewed `annotations` array—not `predictions`—is the only training input.

## Provenance minimum

Every candidate should carry:

- stable task ID;
- task kind (`indication` or `contraindication`);
- source family and source document/record ID;
- section or field name;
- source URI/hash when available;
- generator version and candidate-generation timestamp;
- optional prediction model ID/revision.

No raw DAKP runtime or database is required by MEDliNER. Once the export is reviewed, copy the export into the local ignored data directory and materialize it through Dagster.
