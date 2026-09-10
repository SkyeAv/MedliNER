# MedliNER annotation guide

## Goal

Annotate condition mentions in indication and contraindication text. The task context is metadata; it is not an entity label. Annotators only highlight text and choose a label in Label Studio. Do not count or type character offsets.

Allowed labels:

- `disease`: named diseases, disorders, syndromes, and disease/organ impairment states.
- `phenotype`: symptoms, findings, clinical condition-states, and borderline states such as hypersensitivity, pregnancy, bleeding, seizures, pain, nausea, vomiting, fatigue, or laboratory elevations.

## Span policy

1. **Use maximal concept spans.** Include words that change the clinical concept: anatomical or etiological qualifiers (`pulmonary hypertension`), severity (`severe heart failure`), and activity/course (`active liver disease`).
2. **Exclude evidential and temporal hedges.** Do not include `recent`, `known`, `suspected`, `history of`, or similar patient-record qualifiers. Annotate `myocardial infarction`, not `recent myocardial infarction`.
3. **Exclude population descriptors.** `women of childbearing potential`, `patients`, `children`, and similar subject groups are not disease or phenotype entities.
4. **Medications get no span.** MedliNER labels conditions only. Do not annotate `ibuprofen`, `Advil`, active ingredients, brand names, therapeutic classes, or any dosage/route/frequency attribute, even when the sentence is mostly about the drug.
5. **Do not annotate relations.** Two conditions in the same sentence receive separate spans; the drug they relate to is not annotated at all. The `indication`/`contraindication` task value records the reviewed context.
6. **No entity is valid.** Submit an empty annotation for text that contains no allowed entity, including population-only and dosage-only examples.
7. **No nested or overlapping spans.** Choose one maximal span. If a generic head and a qualified term both appear, annotate only the maximal term supported by the text.
8. **Punctuation and whitespace are excluded.** Label Studio's word granularity helps avoid accidental partial-word spans, and the importer trims any leading/trailing whitespace a drag selection picked up.
9. **Keep spans at most 12 words.** GLiNER only enumerates span candidates up to its `max_width`; a longer gold span is silently unlearnable, so MedliNER rejects it at conversion time. If a concept genuinely needs more than 12 words, record it for adjudication instead of annotating it.
10. **Skip means skip.** If a task cannot be annotated, skip it in the UI rather than submitting an empty annotation. An empty submission is a positive claim that the text contains no entity; a skipped task is rejected at import so it never becomes silent negative training signal.

## Review rules

- Model pre-annotations are suggestions only. Accept, correct, delete, or add spans; never treat an untouched prediction as gold without human review.
- A reviewed task must have `reviewed` or `adjudicated` status in the downstream manifest.
- If annotators disagree, an adjudicator resolves the final span and label. Preserve the original annotations and annotator IDs in the export/provenance record.
- When uncertain between `disease` and `phenotype`, use the definitions above and record the case for adjudication rather than inventing a new label.
