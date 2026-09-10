# Evaluation gates

The primary gate is strict typed span matching: a prediction is correct only when `(start, end, label)` exactly matches the reviewed annotation.

Also report:

- **Boundary-only F1:** exact `(start, end)` while ignoring label;
- **per-task metrics:** `indication` and `contraindication`;
- **per-source metrics:** DailyMed, FAERS, and any later source family;
- **no-entity false-positive rate:** fraction of reviewed empty examples receiving at least one prediction;
- **baseline comparison:** tuned small GLiNER versus untuned small GLiNER;
- **gold-benchmark regression:** the DAKP NER gold benchmark ingested by `medliner ingest` remains held out from training.

Do not select a checkpoint using training loss alone. The reviewed validation split is the selection set; the test split and gold benchmark are reported after selection. For very small datasets, report counts alongside scores and avoid interpreting a single percentage as stable evidence.

## Truncation

Every report includes a `truncation` block. GLiNER truncates inputs beyond `config.max_len`
word tokens with only a warning, which shows up as unexplained missing recall rather than an
error, so the report states the word budget and lists any examples that exceed it. A non-zero
`over_budget_examples` invalidates recall for those examples — shorten the text or window it
before comparing systems.

## Cost

Each example is predicted exactly once per report. The per-task and per-source breakdowns reuse
those counts rather than re-running inference, which matters because the same scorer runs inside
the training-time validation callback on every evaluation step.

## Locating the gold benchmark

Evaluation scores the regression set from the gold benchmark at `$MEDLINER_BENCHMARK`
(default `data/materialized/ingested/ner_gold.json`; set it in the ignored `.envrc.local` for a
ready DAKP export. The older bundle flow materializes it under `$MEDLINER_WORKDIR/ingested/`
via `medliner ingest`). When the file is missing,
evaluation fails with an error; the regression set is never silently skipped.
