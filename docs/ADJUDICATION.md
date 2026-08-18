# Review, adjudication, and dataset versioning

## Statuses

- `draft`: candidate or incomplete annotation; never eligible for training.
- `reviewed`: one annotator completed the task and the export contains the final human spans, including an intentional empty set.
- `adjudicated`: multiple annotations were resolved by an adjudicator; this is preferred for validation/test examples.
- `rejected`: unusable or withdrawn; exclude from all splits.

Label Studio predictions have provenance `model_suggestion` and are never silently promoted to gold. The adapter accepts only the completed `annotations` array, not `predictions`, and rejects unresolved multiple annotation sets unless the task identifies a final annotation or adjudication record.

## Adjudication workflow

1. Annotator A labels the task in the browser.
2. Annotator B independently labels a sampled or high-risk subset.
3. Disagreements are reviewed by an adjudicator, who chooses the final maximal span and label.
4. Preserve the raw export and annotation-set IDs; store the adjudicated result as the selected final annotation set.
5. Materialize a new normalized dataset version. Never mutate an old normalized export in place.

## Versioning

A dataset version is identified by:

- raw Label Studio export content hash;
- annotation policy version;
- normalized JSONL content hash;
- split manifest hash;
- generator/importer version.

Changing a span, label policy, task metadata, or adjudication creates a new version. The old artifact remains available for comparison. Training runs record the exact dataset and split hashes.

## Duplicate and overlap policy

- Exact duplicate spans with the same label inside one selected annotation set are collapsed by the importer.
- Exact duplicate spans with conflicting labels are an error.
- Partial, nested, or crossing overlaps are errors; the annotator must select one maximal span.
- Repeated mentions at different offsets are valid and retained.
- Empty examples are valid reviewed examples and are required for measuring false positives.
- Duplicate task IDs inside one export are an error; the ID is the example identity used by splits and manifests.

## Skipped and cancelled tasks

Label Studio marks a skipped annotation `was_cancelled`. The importer drops cancelled sets, but a
task left with *only* cancelled annotations is an error rather than an empty example — silently
importing it would teach the model that the text contains no entities, which is precisely the
judgement the annotator declined to make. Resolve or remove those tasks before export.

## Whitespace

A click-drag selection routinely captures the space after the final word. The importer verifies
the exported offsets against the source text, then trims leading and trailing whitespace and
re-derives the surface from the tightened offsets. A span containing only whitespace is an error.
This happens at import, where the source text is available, rather than surfacing much later as a
token-alignment failure during GLiNER conversion.

## What the bundle records

The `export_bundle` asset writes `provenance.json` with the raw export hash lineage: base model ID,
selected checkpoint, best validation strict F1, the normalized dataset SHA-256, the metrics
SHA-256, the checkpoint tree hash, the split hash, and the held-out example IDs. `training_config.yaml`
is reconstructed from the checkpoint's own run metadata rather than copied from `configs/`, so it
always reflects the configuration that produced the weights.
