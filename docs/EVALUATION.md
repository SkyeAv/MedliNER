# Evaluation gates

The primary gate is strict typed span matching: a prediction is correct only when `(start, end, label)` exactly matches the reviewed annotation.

Also report:

- **Boundary-only F1:** exact `(start, end)` while ignoring label;
- **per-task metrics:** `indication` and `contraindication`;
- **per-source metrics:** DailyMed, FAERS, and any later source family;
- **no-entity false-positive rate:** fraction of reviewed empty examples receiving at least one prediction;
- **baseline comparison:** tuned small GLiNER, untuned small GLiNER, and DAKP's deterministic gazetteer when the sibling checkout is available;
- **DAKP regression:** the committed `../DAKP/tests/eval/ner_gold.json` remains held out from training.

Do not select a checkpoint using training loss alone. The reviewed validation split is the selection set; the test split and DAKP regression fixture are reported after selection. For very small datasets, report counts alongside scores and avoid interpreting a single percentage as stable evidence.
