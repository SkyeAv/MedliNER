# MEDliNER annotation guide

## Goal

Annotate entity mentions in indication and contraindication text. The task context is metadata; it is not an entity label. Annotators only highlight text and choose a label in Label Studio. Do not count or type character offsets.

Allowed labels:

- `disease`: named diseases, disorders, syndromes, and disease/organ impairment states.
- `phenotype`: symptoms, findings, clinical condition-states, and borderline states such as hypersensitivity, pregnancy, bleeding, seizures, pain, nausea, vomiting, fatigue, or laboratory elevations.
- `drug`: a medication, active ingredient, generic name, brand name, or explicitly named drug combination.

## Span policy

1. **Use maximal concept spans.** Include words that change the clinical concept: anatomical or etiological qualifiers (`pulmonary hypertension`), severity (`severe heart failure`), activity/course (`active liver disease`), and medication salt/form when it is part of the name.
2. **Exclude evidential and temporal hedges.** Do not include `recent`, `known`, `suspected`, `history of`, or similar patient-record qualifiers. Annotate `myocardial infarction`, not `recent myocardial infarction`.
3. **Exclude population descriptors.** `women of childbearing potential`, `patients`, `children`, and similar subject groups are not disease, phenotype, or drug entities.
4. **Drug names only.** Annotate `ibuprofen` or `Advil` in `ibuprofen 400 mg orally twice daily`; do not include `400 mg`, `orally`, `twice daily`, tablet counts, or other dosage/route/frequency attributes. Do not annotate a bare therapeutic class unless it is being used as a medication name in context.
5. **Do not annotate relations.** A drug and condition in the same sentence receive separate spans. The `indication`/`contraindication` task value records the reviewed context.
6. **No entity is valid.** Submit an empty annotation for text that contains no allowed entity, including population-only and dosage-only examples.
7. **No nested or overlapping spans.** Choose one maximal span. If a generic head and a qualified term both appear, annotate only the maximal term supported by the text.
8. **Punctuation and whitespace are excluded** unless they are genuinely part of a medication name. Label Studio's word granularity helps avoid accidental partial-word spans.

## Review rules

- Model pre-annotations are suggestions only. Accept, correct, delete, or add spans; never treat an untouched prediction as gold without human review.
- A reviewed task must have `reviewed` or `adjudicated` status in the downstream manifest.
- If annotators disagree, an adjudicator resolves the final span and label. Preserve the original annotations and annotator IDs in the export/provenance record.
- When uncertain between `disease` and `phenotype`, use the definitions above and record the case for adjudication rather than inventing a new label.
