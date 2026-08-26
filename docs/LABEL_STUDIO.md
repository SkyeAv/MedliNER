# Label Studio Community Edition workflow

Label Studio Community Edition is a free, locally self-hosted browser UI for this project. It
runs in a podman container managed by the MedliNER CLI; it is not a MedliNER Python
dependency.

## Managed server (default)

`make annotate` starts the stock `heartexlabs/label-studio` image with podman,
waits for health, creates the `MedliNER` project from
`configs/label_studio_ner.xml`, and imports the tasks built by `make prepare` (run
automatically when the import file for the current input hash and sampling config is
missing; see the sampling table in `docs/CANDIDATE_TASKS.md` for the `MEDLINER_SAMPLE_*`
variables that bound the import to ~1K mostly-edge-case, balanced, staggered tasks):

```bash
make annotate
```

Open <http://localhost:9030> and log in with `$MEDLINER_LABEL_STUDIO_USERNAME` /
`$MEDLINER_LABEL_STUDIO_PASSWORD` (local defaults are in `.envrc`; the container image creates
that account on first boot). Server state — accounts, projects, annotations — lives in
`$MEDLINER_WORKDIR/label-studio/server-data` and survives container restarts.

Behavior notes:

- The project title is `MedliNER` (previously `MedliNER medical NER`). Projects are matched by
  exact title, so the first run after the rename creates a fresh project; any old-titled
  project remains in the server data directory, unused.
- Re-running `make annotate` reuses a running container and an existing project, and skips
  the import when the project already has tasks. To replace project tasks, run
  `uv run medliner label-studio --reimport`.
- If the default-account login ever fails (e.g. image behavior changes), create an account in
  the browser, copy an access token from Account & Settings, and set
  `MEDLINER_LABEL_STUDIO_TOKEN` in `.envrc.local`; the client then sends it as a Bearer
  credential and skips the login form. (Legacy `Token` API auth is disabled by default in
  Label Studio ≥ 1.23, which is why the default path uses session login.)
- Stop the server with `make stop` (removes the container, keeps the data dir).

## Optional annotator onboarding (presentation mode)

The repository ships a separate `Onboarding` project on the same local server for qualifying
annotators during a live session — for example, onboarding everyone at once during a
presentation. It is fully optional; production training runs fine without it.

The project contains ten answer-free benchmark tasks; the gold spans are kept in a versioned
sidecar under `$MEDLINER_WORKDIR/onboarding/`. Each annotator account gets a deterministic
four-task attempt. At least three of four tasks must be exactly correct (character boundaries and
label included) before promotion.

Two commands cover the whole flow — nobody is ever named on the command line:

```bash
make setup
export MEDLINER_LABEL_STUDIO_ANNOTATORS="alice:pw-a,bob:pw-b"
make onboarding            # provisions the project and assigns a quiz to EVERY account at once
# everyone annotates their four assigned tasks in the Onboarding project
make onboarding-promote    # exports, scores every attempt, promotes everyone passing (≥3/4)
```

Rerun `make onboarding` for another round; each round selects a new four-task subset per user from
the ten-case bank. After promotion, run the unchanged production flow (`make annotate`,
`make export`).

The test-bank and attempt files include benchmark/config hashes, so changing the benchmark starts
a new onboarding version and old passes do not unlock it. Reports are append-only and stay on disk
as the operational record of who passed.

**Community Edition limitation:** CE does not provide per-user project visibility or task
assignment. A user who already has access to the shared CE instance may technically open the
production project. Onboarding promotion is an operational record, not a hard UI/API access
barrier; a hard barrier would require a custom proxy/frontend or a separate production instance.

## Group annotation sessions (e.g. a presentation)

Label Studio Community Edition has **no limits on users, annotators, or tasks**, but it has
no task-assignment or role layer (those are Enterprise features): everyone with an account
on the shared instance sees every project. The managed flow supports a group session:

1. **Expose the server on the network** with `MEDLINER_LABEL_STUDIO_HOST=0.0.0.0`
   (default `127.0.0.1` keeps it private). Attendees then reach
   `http://<this-machine>:$MEDLINER_LABEL_STUDIO_PORT` from the same LAN; the CLI prints a
   reminder when the bind address is public.
2. **Pre-create annotator accounts** so nobody registers during the session:
   `MEDLINER_LABEL_STUDIO_ANNOTATORS="alice:pw-a,bob:pw-b" make annotate`. Accounts are created
   via `POST /api/users` and existing usernames are skipped, so the command stays idempotent.
3. **Avoid collisions**: CE lets two annotators open the same task, and tasks are served in
   queue order. For a short session either assign each person a slice of the task list, or
   rely on the natural staggering of the sequential queue — both work with this pipeline
   because exports keep per-annotation authorship.
4. **Use onboarding** with `make onboarding` when annotator qualification matters: it assigns
   everyone their quiz at once and `make onboarding-promote` records each annotator's score and
   promotes the passing ones. `uv run medliner label-studio --warmup` remains available as an
   informal demo; its gold spans are intentionally visible to the presenter and it is not a
   qualification gate.
5. **Speed up labeling** with the hotkey baked into `configs/label_studio_ner.xml`:
   `1` = DiseaseOrPhenotypicFeature after selecting a span.

## Export

Download the reviewed annotations over the API instead of the browser:

```bash
make export            # writes $MEDLINER_LABEL_STUDIO_EXPORT
MEDLINER_LABEL_STUDIO_EXPORT=/tmp/reviewed.json make export
```

The command finds the `MedliNER` project by title, downloads the JSON export,
and prints the annotated/total task counts. The manual browser route still works: export as
JSON from the project UI and save it anywhere, then override `MEDLINER_LABEL_STUDIO_EXPORT`
in the ignored `.envrc.local` if the path differs from the example in `.envrc`.

```bash
cat > .envrc.local <<'EOF'
export MEDLINER_LABEL_STUDIO_EXPORT="$PWD/data/label-studio/indications-2026-01.json"
EOF
direnv allow
make export
```

## Import task JSON

Candidate tasks come from `make prepare` (see `docs/CANDIDATE_TASKS.md`). Each
task exposes at least:

```json
{
  "id": "dailymed-001",
  "data": {
    "text": "Contraindicated in patients with pulmonary hypertension.",
    "task": "contraindication",
    "source_family": "dailymed",
    "source_document_id": "spl-document-001"
  }
}
```

The task and source fields are displayed for context and are preserved when exported. They are not labels to be highlighted.

## Alternative: run Label Studio yourself

If you prefer not to use the managed container, install Label Studio separately:

```bash
# Dedicated environment, separate from MedliNER
python -m venv .label-studio-venv
source .label-studio-venv/bin/activate
pip install label-studio
label-studio start --port 9030
```

or:

```bash
podman run --rm -it -p 9030:8080 \
  -v "$PWD/.label-studio-data:/label-studio/data:Z" \
  docker.io/heartexlabs/label-studio:latest
```

Then create a local account/project, paste the contents of `configs/label_studio_ner.xml`
into the project's labeling configuration, and import the JSON file written by
`make prepare` (its path is printed to stdout).

## Annotator workflow

1. Open a task.
2. Read the visible indication or contraindication context.
3. Click-drag/highlight the complete condition phrase.
4. Choose `DiseaseOrPhenotypicFeature`.
5. Correct/delete/add spans as needed.
6. Submit the task.

There is no manual offset counting. Label Studio records the highlighted phrase and its half-open character `start`/`end` offsets automatically. Empty tasks should be submitted with no spans.

## Pre-annotations

Annotators are far faster correcting a span than drawing one, so `make prepare` also runs
`medliner prelabel` to attach model suggestions to the import file before the session starts:

```bash
make prepare                          # candidates + prelabel: import-<hash>.prelabeled.json + manifests
make annotate                         # serves the pre-labeled file with prediction pre-fill on
```

The suggestions come from `gliner-community/gliner_large-v2.5` prompted with the single merged
condition label `DiseaseOrPhenotypicFeature` at threshold `0.35` — the same checkpoint and
threshold the sibling DAKP pipeline mines contraindications with; DAKP prompts disease/phenotype
types separately but merges them downstream, so MedliNER prompts the merged label directly
(`MEDLINER_PRELABEL_MODEL` / `MEDLINER_PRELABEL_THRESHOLD`
override them). Raw model output does not obey the annotation guide, so the same cleanup DAKP
applies is applied here: leading hedges are trimmed (`recent myocardial infarction` →
`myocardial infarction`, guide rule 2), population descriptors are dropped (`patients`, `women of
childbearing potential`, rule 3), overlapping spans collapse to the longest (rule 7), and spans
wider than the model's `max_width` are dropped because GLiNER never enumerates a span candidate
that wide.

`PRELABEL=1` also turns on the project's `show_collab_predictions`, which is what puts the spans
in front of the annotator; without it Label Studio stores the predictions and never shows them.
Opening a pre-labeled task pre-fills the draft annotation with the model's spans.

**They are suggestions.** Accept, correct, or delete each one, and add what the model missed —
an untouched prediction is not gold. MedliNER's adapter reads only the completed `annotations`
array and never `predictions`, so a task nobody submitted contributes nothing. Each submitted
span carries an `origin` (`prediction`, `prediction-changed`, or `manual`) in the export, so an
untouched model span stays distinguishable from one a human drew or corrected.

Re-running `uv run medliner prelabel` is cheap: suggestions are cached per text under
`$MEDLINER_WORKDIR/label-studio/prelabel-cache.json`, keyed by model, threshold, labels, window
budget, and normalized text, so adding candidates only runs the model over the new ones.
`FORCE=1` (or `--force`) ignores the cache.

Before trusting suggestions in front of a room, score them against the gold benchmark:

```bash
uv run medliner prelabel --score-gold
```

That reports strict and boundary-only F1 over the ingested `ner_gold.json` cases. Pre-labels
materially worse than the annotators' own first guess cost time rather than saving it.

## Export details worth knowing

The raw export is retained as provenance. MedliNER converts it to its canonical schema and validates offsets, labels, overlap, task metadata, review status, and text slices. JSONL is also accepted by MedliNER when one task object is stored per line.

Two export details are worth knowing before the first review round:

- **Whitespace.** Dragging across a phrase usually captures the following space. The importer
  checks the exported offsets against the source text, then trims the edges and re-derives the
  surface, so annotators do not have to be precise about it.
- **Skipped tasks.** A task whose annotations are all `was_cancelled` is rejected rather than
  imported as an empty example, because an empty example is a positive claim that the text has no
  entity. Resolve or remove skipped tasks before exporting.
