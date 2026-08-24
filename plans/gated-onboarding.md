# Label Studio gated onboarding

## Context

The current checkout (including the working-tree changes layered on `main`) runs Label Studio Community Edition in a Podman container and provisions one owner account plus optional annotator accounts. Its current workflow is `make setup` → `make data` → `make annotate` → `make export` → `make train`, with optional local-LLM `make shorten` preprocessing. It has a manual warm-up project, but no onboarding attempt state, per-annotator scoring, retry flow, or production promotion gate.

The requested feature is rubric-gated onboarding: each annotator completes a short gold-backed quiz, receives an objective score, and is admitted to production annotation only after passing. The selected design uses two projects on the existing local Label Studio deployment: **`Onboarding`** for training/quiz work and **`MedliNER`** for production work.

Confirmed requirements:

- 4 questions per attempt from a 10-question test bank.
- Passing requires at least 75% exact task correctness, i.e. at least 3 of 4 tasks.
- Unlimited retries; every retry receives a new 4-question selection.
- Keep the workflow local and Podman-managed, with no external SaaS dependency.
- Retain attempt history and promote an annotator when any attempt passes.

## Approach

- Add an `Onboarding` project alongside the existing `MedliNER` project on the same local Label Studio server.
- Build a deterministic 10-case test bank from `$MEDLINER_BENCHMARK` (repo-relative by default; set it in the ignored `.envrc.local` for a ready DAKP export; the legacy ingest path remains supported), and generate a per-annotator/per-attempt selection of 4 task IDs. Public task JSON contains only quiz text and metadata; gold answers remain in a local sidecar manifest under the workdir.
- Keep onboarding independent of the current `make data` sampling/prelabel pipeline and optional `make shorten` step: production candidate preparation remains unchanged, while onboarding uses adjudicated benchmark cases directly.
- Add onboarding state and scoring code that records username, attempt number, selected task IDs, submission/export identity, per-task exact correctness, score, and pass/fail. A task is correct only when the submitted set of `(start, end, label)` spans exactly equals gold, including valid empty annotations.
- Add CLI/Make targets to start the onboarding project, begin an attempt, evaluate the submitted quiz, show status, and promote a passing annotator for production. Keep retries append-only and choose a fresh deterministic 4-task selection from the 10-case bank.
- Change production provisioning/import guidance so only promoted annotators are treated as production annotators and production annotations from non-passed users are not accepted into the reviewed dataset.
- Keep the CE limitation explicit: Community Edition does not provide per-user project visibility/assignment, so two projects provide a robust operational gate and separate queues, not an adversarial security boundary. A user who can already access the shared CE instance may technically open the production project; strict UI/API blocking would require a later proxy, custom frontend, or separate production instance.

## Files to modify

- `src/medliner/onboarding.py` — test-bank selection, hidden gold sidecar, attempt state, export grouping, exact scoring, and promotion records.
- `src/medliner/candidates.py` — replace or supersede the current presenter-only warm-up task builder with answer-free onboarding task generation while retaining deterministic IDs and validation.
- `src/medliner/label_studio_server.py` — support the `Onboarding` project, project-specific exports, and any API helpers needed to identify completed annotator submissions.
- `src/medliner/cli.py` — add onboarding start/evaluate/status/promote commands and wire project provisioning to the existing credentials/container; preserve current `annotate`/`export` behavior and optionally retain `--warmup` as a compatibility alias.
- `Makefile` — add concise `onboarding`/`onboarding-start`/`onboarding-evaluate`/`onboarding-status` wrappers alongside the current `data`, `annotate`, `export`, and `train` targets; do not fold onboarding into `make data` or the optional `make shorten` path.
- `configs/onboarding.json` — declare the 10-case bank size, 4-question attempt size, 0.75 threshold, selection seed, and project title without storing answer spans.
- `docs/LABEL_STUDIO.md` and `README.md` — document the annotator workflow, operator commands, promotion semantics, and CE limitation.
- `tests/test_onboarding.py`, `tests/test_candidates.py`, `tests/test_label_studio_server.py`, and `tests/test_cli.py` — cover deterministic selection, hidden answers, exact scoring, retries, persistence, idempotence, and CLI wiring.

## Reuse

- `build_warmup_tasks()` in `src/medliner/candidates.py` already validates benchmark cases, maps source/task metadata, and creates deterministic task IDs; onboarding should make its gold metadata private rather than sending `gold_mentions` to Label Studio.
- `load_gold_benchmark()` in `src/medliner/evaluation.py` already converts the benchmark into canonical `Example` objects with validated character offsets.
- `score_example()` in `src/medliner/evaluation.py` already defines strict span/type comparison; onboarding can use its exact sets for per-task correctness while avoiding model-oriented aggregate recall semantics.
- `LabelStudioClient.export_annotations()` and `export_project()` already retrieve Label Studio JSON and preserve per-annotation authorship (`created_username` / `completed_by`).
- `LabelStudioClient.ensure_project()`, `list_users()`, `create_user()`, and `provision()` already provide idempotent local project/account setup.
- Existing workdir conventions, JSON manifests, `--annotator` parsing, and focused fake-API tests provide persistence and test seams.
- The current `make data` target remains responsible for the sampled 5K edge-case production import and GLiNER prelabels; onboarding should not alter its sampling or local-LLM behavior.

## Decisions

- Use two projects on the same local Label Studio instance: `Onboarding` and `MedliNER`.
- Keep the test bank at 10 cases and assign 4 per attempt, with deterministic selection from a stable seed plus annotator/attempt identity.
- Score only once all 4 assigned tasks have a completed annotation; exact task correctness is all-or-nothing on the complete span/type set.
- Retain every attempt and its raw export linkage; a single passing attempt marks the annotator passed permanently for the current test-bank/config version.
- Do not expose gold answers in task data. Operator-only sidecar/state files may contain answers and scores.
- Treat project separation and production-data filtering as the CE-compatible operational gate. Document that it is not protection against a user who can deliberately bypass the shared CE UI/API.

## Steps

- [x] Confirm the meaning: short gold-labeled quizzes plus an evaluation threshold before production annotation.
- [x] Confirm quiz size: 4 questions from a 10-question bank; pass at 75% exact task correctness (at least 3 of 4).
- [x] Confirm unlimited retries with a new 4-question selection per attempt.
- [x] Select two local Label Studio projects: `Onboarding` and `MedliNER` production.
- [x] Define the onboarding JSON schemas and versioned state paths, including test-bank hash and attempt manifest.
- [x] Implement deterministic 4-of-10 selection and answer-free import task generation.
- [x] Implement export parsing by annotator, strict per-task correctness, 3-of-4 thresholding, and append-only attempt reports.
- [x] Add project provisioning commands for onboarding and production, plus status/promotion commands, using the current `make annotate`/`make export` naming and `MEDLINER_BENCHMARK` path.
- [x] Gate downstream production acceptance on the promotion manifest and preserve non-passed annotations as excluded audit data without changing the existing `make data` → `make annotate` → `make export` preparation flow.
- [x] Add tests for empty gold cases, wrong labels/boundaries, incomplete attempts, retries, duplicate submissions, unknown users, changed test-bank versions, and idempotent reruns.
- [x] Update operator/annotator documentation with the exact command sequence and limitation.

## Verification

- Build an `Onboarding` project with 10 answer-free tasks and verify the private sidecar contains the matching gold spans while the Label Studio import does not.
- Start two attempts for the same annotator and verify each has 4 distinct deterministic assignments, a new selection on retry, and retained history.
- Export a fully annotated attempt and verify exact task scoring: 3/4 and 4/4 pass, 2/4 fails, wrong label/boundary fails, and empty-gold/empty-annotation passes.
- Verify incomplete attempts are not scored or promoted, duplicate/replayed evaluation is idempotent, and a passing attempt produces a durable promotion record.
- Verify production setup reports only promoted annotators as eligible and downstream dataset ingestion rejects or excludes non-promoted production annotations with an actionable message.
- Verify both projects, users, annotations, and onboarding state survive container restart.
- Run the focused onboarding tests, the existing Label Studio fake-API tests, and the current repository commands (`make check`; no LLM server is needed for onboarding tests). This verification passes: the full pytest suite, Ruff lint, and Ruff format checks are green.
