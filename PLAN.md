# GLiNER Medical NER Fine-tuning + Dagster Data Platform

## Context

MedliNER is currently empty, while the neighboring `../DAKP` repository already contains a production GLiNER-backed `DiseaseNER` implementation and a hand-labeled benchmark for DailyMed contraindications and FAERS indications. The goal is to design a reproducible path to fine-tune GLiNER for DAKP-specific medical NER tasks, use Dagster to generate/track training data and experiments, and produce a checkpoint/dataset that can later be uploaded to Hugging Face.

Initial findings and decisions:

- DAKP's current backend is a gazetteer-first composite around `gliner-community/gliner_large-v2.5`; it emits text spans plus types, not ontology CURIEs.
- DAKP has a small strict benchmark (`../DAKP/tests/eval/ner_gold.json`, 34 cases / 42 spans) and documented span/label policies.
- DAKP already handles long GLiNER inputs via sentence-aware windows and has model caching, but no training pipeline or experiment/data lineage layer.
- The first corpus will cover DailyMed contraindication sections, FAERS indication strings, and DAKP approved-treatment/observed-use text. Other DAKP sources are out of scope for the first pass.
- Human annotators are available. Weak/pseudo-labels may bootstrap candidate selection, but reviewed gold data must drive validation and final model decisions.
- The annotation source will be Label Studio exports. MedliNER will not invoke DAKP extraction or depend on DAKP runtime artifacts in the first version; it will ingest reviewed indication/contraindication tasks exported from Label Studio.
- Label Studio Community Edition is a practical local browser-based choice: it is self-hostable, free for this use, supports text span NER through its `Text` + `Labels` configuration, and exports JSON annotations. Annotators do not count indexes: they open a task, click-drag across the answer text in the browser, and choose `disease` or `phenotype`; Label Studio records the character start/end offsets automatically. The UI can also display whether the task is an indication or contraindication. It is an annotation UI, not the canonical training schema, so MedliNER must normalize and validate its export rather than train directly on Label Studio JSON.
- DAKP integration is explicitly deferred. This repository's initial deliverable is a standalone, reproducible training/evaluation project and uploadable model artifact.
- The target training hardware is one mobile RTX 5070 Ti with 12 GB VRAM. The plan must use conservative sequence lengths, micro-batches, gradient accumulation, mixed precision, and checkpoint/resume behavior.
- Dagster should start as a local development orchestrator. It is appropriate if kept to a small asset graph with a local run/data directory and Dagster UI; distributed deployment is not part of the initial design.
- Fine-tuning is technically supported by the installed GLiNER 0.2.28 package: `GLiNER.train_model(train_dataset, eval_dataset, ...)` uses its bundled Hugging Face-style trainer and accepts span records as tokenized text plus `(start_token, end_token, label)` tuples. The installed span processor uses token-index spans with an inclusive end, so Label Studio's half-open character offsets must be converted carefully and covered by round-trip tests. This is not a blocked project.
- The initial checkpoint will be the smaller GLiNER model rather than treating `gliner_large-v2.5` as a requirement. DAKP documents `urchade/gliner_small-v2.1` as a known small checkpoint; the exact base/revision will be pinned after a compatibility smoke test. The large checkpoint can remain an optional later comparison, but a successful small-model fine-tune is the primary deliverable for the 12 GB laptop.

### Indications and contraindications are first-class tasks

The first milestone should extract condition mentions from both indication and contraindication contexts. Keep the NER labels as `disease` and `phenotype`; add a separate immutable `task` field with values `indication` or `contraindication` for sampling, splits, review, and metrics. The task field is not an entity label.

This distinction matters: GLiNER can identify the condition span, but plain NER does not by itself assert whether that condition is an indication or contraindication. For DAKP's structured sections, the source section and existing contraindication sentence filters provide that context. If later users need arbitrary mixed prose classified as indication versus contraindication, add a separate assertion-role classifier or a four-label formulation; do not silently overload the medical entity labels in the first model.

### Do GLiNER examples need entity types?

Yes, even though GLiNER inference is query-driven. At inference time, labels are the natural-language queries supplied to `predict_entities`, and returned entities carry the matching label. During supervised fine-tuning, each gold span still needs a target label so the model learns which query should retrieve it; the training record is effectively text plus span boundaries plus entity label(s). The labels do not need to be ontology CURIEs, and they can be intentionally coarse.

For the initial MedliNER corpus, use two entity labels: `disease` and `phenotype`. These are exactly DAKP's `CONTRAINDICATION_DISEASE_TYPES`, so the annotation policy, the gold benchmark, and the GLiNER pre-labeling prompts stay compatible with DAKP without translation. A third `drug` label was specified in the original version of this plan and has since been dropped from the schema; medications get no span (`docs/ANNOTATION_GUIDE.md` rule 4). We can separately evaluate a type-agnostic boundary metric later.

## Approach

Build a small, local Dagster-orchestrated data and training system in MedliNER:

1. Import reviewed Label Studio JSON/JSONL exports containing indication and contraindication text tasks; retain imported task/source metadata and annotation provenance.
2. Annotators work entirely by highlighting answer text in the browser and selecting a label; they never enter indexes. Normalize the exported result into a versioned internal schema, validating the automatically recorded Label Studio character offsets, label names, duplicate/overlapping spans, empty examples, and task values.
3. Convert the validated records into GLiNER's training representation: model-tokenized text plus `(start_token, end_token_inclusive, label)` tuples for `disease` and `phenotype`; keep original Label Studio character offsets for audit/evaluation and test the conversion against GLiNER's actual `words_splitter`.
4. Freeze a leakage-resistant split, keeping the existing DAKP gold fixture as a regression set rather than silently absorbing it into training.
5. Run a GLiNER training smoke test, then fine-tune the selected small checkpoint on the laptop. Record parameters and artifacts in Dagster metadata and compare against the untuned small checkpoint, the existing large-model baseline where available, and the gazetteer baseline.
6. Export the best standalone checkpoint with a model card, label schema, task definition, dataset manifest, evaluation report, and reproducible configuration suitable for later Hugging Face upload.

Medical Hugging Face datasets and DAKP integration remain explicit follow-up phases, not dependencies of the first implementation.

## Files to modify

- `pyproject.toml` — MedliNER runtime/dev dependencies, pinned GLiNER training compatibility, Dagster entry point, and GPU-friendly environment notes. Label Studio remains an external local service rather than a required training dependency.
- `README.md` — setup, Label Studio workflow, export contract, Dagster commands, training/evaluation flow, and Hugging Face artifact layout.
- `src/medliner/schema.py` — canonical examples, annotations, provenance, task metadata, and serialized dataset contracts.
- `src/medliner/label_studio.py` — Label Studio JSON/JSONL importer, validation, normalization, and character-offset handling.
- `src/medliner/gliner_data.py` — conversion from canonical character spans to the installed GLiNER token-span representation.
- `src/medliner/splits.py` — deterministic, provenance-aware train/validation/test splitting and split manifests.
- `src/medliner/evaluation.py` — strict typed-span, boundary-only, per-task/source, and no-entity metrics; DAKP benchmark adapter.
- `src/medliner/training.py` — small-checkpoint loading, laptop-safe training arguments, checkpoint/resume, and artifact metadata.
- `src/medliner/dagster_defs.py` — minimal local Dagster asset graph and asset checks; no schedules/sensors initially.
- `configs/label_studio_ner.xml` — Label Studio browser configuration for `disease`/`phenotype` spans and indication/contraindication context display.
- `configs/train-small.yaml` — pinned base checkpoint, sequence/window limits, batch/accumulation, precision, seed, and output settings.
- `tests/` — importer/offset tests, GLiNER conversion tests, split determinism/leakage tests, metric tests, Dagster asset tests, and a mocked one-step training smoke test.
- `data/`/`artifacts/` conventions documented but gitignored — Label Studio exports, normalized datasets, split manifests, checkpoints, and reports remain local or are explicitly packaged for upload.

No changes to `../DAKP` are required in this phase.

## Reuse

- `../DAKP/src/dakp_pipeline/ner/ner.py` — `DiseaseNER`, model configuration, span offsets, thresholds, and composite merge behavior.
- `../DAKP/src/dakp_pipeline/ner/README.md` and `BENCHMARK.md` — existing label policy, known failure modes, and evaluation conventions.
- `../DAKP/tests/eval/benchmark_ner.py` and `ner_gold.json` — starting regression benchmark and gold format.
- `../DAKP/src/dakp_pipeline/io/schemas.py` — source/table contracts, especially DailyMed SPL document fields and FAERS case fields.
- `../DAKP/src/dakp_pipeline/assertions/contraindications.py` — how DailyMed contraindication text is selected, filtered, and paired with evidence.
- `../DAKP/src/dakp_pipeline/assertions/approved_treats.py` — how DailyMed indication sections and FAERS candidates are corroborated.
- `../DAKP/src/dakp_pipeline/assertions/observed_uses.py` — FAERS indication normalization, stop-list behavior, and unique indication sampling.
- `../DAKP/src/dakp_pipeline/io/artifact_store.py` — content-addressed artifact and manifest pattern that can inspire MedliNER's local dataset artifacts.
- `../DAKP/src/dakp_pipeline/ner/model_cache.py` — model acquisition/caching pattern to assess for reuse.

## Steps

- [x] Finalize the two-label schema (`disease`, `phenotype`) and annotation guide, including maximal-span, qualifier, hedge, population, medication-exclusion, and no-entity policies.
- [x] Define the Label Studio Community Edition setup: local self-hosted install (pip or Docker), NER labeling configuration with `disease` and `phenotype`, visible indication/contraindication task metadata, and JSON/JSONL export procedure. Document the annotator workflow as: open task → click-drag/highlight the condition phrase → choose its label → submit; no manual offset counting. Keep Label Studio outside the core training environment if desired.
- [x] Define the Label Studio import/export contract and MedliNER adapter: character-offset validation, the three allowed labels, task/source metadata, annotation provenance, duplicate/overlap policy, empty examples, and conversion to GLiNER token-span records.
- [x] ~~Define how candidate tasks are created outside this repository~~ Candidate generation moved in-repo: the user authors raw candidates NDJSON from intermediate DAKP inputs, and the `raw_candidate_texts` → `candidate_tasks` assets validate/dedupe it into a Label Studio import file; the `label_studio_server` asset runs Label Studio in a podman container and imports the tasks. Reviewed exports (manual browser export) are still the only training input.
- [x] Define adjudication, annotation status, and versioning. Model suggestions are suggestions, never silently treated as gold.
- [x] Design normalized GLiNER dataset contracts, deterministic splits, and leakage controls. Split by source document/label family where possible, not by randomly duplicated sentence; keep the committed DAKP benchmark held out as a regression test.
- [x] Design a minimal local Dagster asset graph: raw candidates → Label Studio import tasks → Label Studio server (podman) → [manual annotation/export] → Label Studio export → validated/normalized dataset → frozen splits → training run → evaluation report → export bundle. Assets, materialization, metadata, and asset checks only; no schedules/sensors or deployment infrastructure.
- [x] Design laptop-safe small-GLiNER fine-tuning, checkpoint/resume, and hyperparameter configs for 12 GB VRAM: first run a one-batch/one-step smoke test, then use bounded windows, fp16 or bf16 as supported, micro-batch plus gradient accumulation, conservative max epochs, and early stopping on reviewed validation F1. Keep large-checkpoint training optional rather than a gate.
- [x] Define evaluation gates: strict `(start, end, type)` precision/recall/F1, lenient boundary-only F1, per-source/task metrics for indication and contraindication, no-entity false-positive rate, and comparison with untuned GLiNER plus the gazetteer baseline.
- [x] Define standalone artifact packaging for later Hugging Face upload: checkpoint directory, label list, annotation policy, dataset manifest, split hashes, training config, metrics, provenance/license notes, and model card inputs. Defer DAKP runtime integration.
- [x] Specify implementation files, tests, commands, and end-to-end verification.

## Verification

The first implementation should verify: a local annotator can open Label Studio in a browser, import tasks, click-drag/highlight condition phrases, choose `disease`/`phenotype`, and export JSON without manually entering offsets; MedliNER deterministically validates and converts that export; source/task metadata and annotation provenance survive conversion; deterministic re-runs produce the same normalized dataset and split hashes; no source/document leakage occurs across splits; training completes or resumes on the 12 GB GPU; the tuned checkpoint is compared with the untuned GLiNER and gazetteer baselines using strict and type-agnostic span metrics; and the exported artifact contains the model, label schema, dataset/config manifests, benchmark report, and model-card inputs.

Verification completed: `uv sync`, six unit/contract tests, Ruff checks, Dagster definition validation, a local Dagster materialization of normalized data and frozen splits, and an actual one-step BF16 smoke fine-tune on the RTX 5070 Ti succeeded after using a cu130 Torch wheel with `sm_120` support. The sibling DAKP cu126 Torch environment was confirmed incompatible with this GPU and is documented as a non-training environment.
