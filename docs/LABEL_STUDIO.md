# Label Studio Community Edition workflow

Label Studio Community Edition is a free, locally self-hosted browser UI for this project. It
runs in a podman container managed by the Dagster pipeline; it is not a MEDliNER Python
dependency.

## Dagster-managed server (default)

The `label_studio_server` asset starts the stock `heartexlabs/label-studio` image with podman,
waits for health, creates the `MEDliNER medical NER` project from
`configs/label_studio_ner.xml`, and imports the tasks built by the upstream `candidate_tasks`
asset:

```bash
make UP
# In the Dagster UI, materialize `label_studio_server`.
# Its upstream assets `raw_candidate_texts` and `candidate_tasks` materialize first.
```

Open <http://localhost:9030> and log in with `$MEDLINER_LABEL_STUDIO_USERNAME` /
`$MEDLINER_LABEL_STUDIO_PASSWORD` (local defaults are in `.envrc`; the container image creates
that account on first boot). Server state — accounts, projects, annotations — lives in
`$MEDLINER_WORKDIR/label-studio/server-data` and survives container restarts.

Behavior notes:

- Re-materializing the asset reuses a running container and an existing project, and skips
  the import when the project already has tasks. To replace project tasks, materialize with
  run config `{"reimport": true}` (Shift-click "Materialize" to open the launchpad).
- If the default-account login ever fails (e.g. image behavior changes), create an account in
  the browser, copy an access token from Account & Settings, and set
  `MEDLINER_LABEL_STUDIO_TOKEN` in `.envrc.local`; the asset then sends it as a Bearer
  credential and skips the login form. (Legacy `Token` API auth is disabled by default in
  Label Studio ≥ 1.23, which is why the default path uses session login.)
- Stop the server with `make label-studio-stop` (removes the container, keeps the data dir).

## Import task JSON

Candidate tasks come from the `candidate_tasks` asset (see `docs/CANDIDATE_TASKS.md`). Each
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

If you prefer not to use the Dagster-managed container, install Label Studio separately:

```bash
# Dedicated environment, separate from MEDliNER
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
into the project's labeling configuration, and import the JSON file written by the
`candidate_tasks` asset (its path is on the asset's metadata in the Dagster UI).

## Annotator workflow

1. Open a task.
2. Read the visible indication or contraindication context.
3. Click-drag/highlight the complete condition or drug phrase.
4. Choose `disease`, `phenotype`, or `drug`.
5. Correct/delete/add spans as needed.
6. Submit the task.

There is no manual offset counting. Label Studio records the highlighted phrase and its half-open character `start`/`end` offsets automatically. Empty tasks should be submitted with no spans.

## Pre-annotations

Optional model suggestions may be imported using Label Studio's prediction format. They must be visibly treated as predictions and reviewed by a human. The MEDliNER adapter rejects a model-only prediction set as training gold unless the completed annotation has human provenance/status.

## Export

From the project, export the completed annotations as JSON. JSONL is also accepted by MEDliNER when one task object is stored per line. Save the export under a local ignored directory such as `data/label-studio/indications-2026-01.json`, then override `MEDLINER_LABEL_STUDIO_EXPORT` in the ignored `.envrc.local` if the path differs from the example in `.envrc`.

```bash
cat > .envrc.local <<'EOF'
export MEDLINER_LABEL_STUDIO_EXPORT="$PWD/data/label-studio/indications-2026-01.json"
EOF
direnv allow
make UP
```

The raw export is retained as provenance. MEDliNER converts it to its canonical schema and validates offsets, labels, overlap, task metadata, review status, and text slices before training.

Two export details are worth knowing before the first review round:

- **Whitespace.** Dragging across a phrase usually captures the following space. The importer
  checks the exported offsets against the source text, then trims the edges and re-derives the
  surface, so annotators do not have to be precise about it.
- **Skipped tasks.** A task whose annotations are all `was_cancelled` is rejected rather than
  imported as an empty example, because an empty example is a positive claim that the text has no
  entity. Resolve or remove skipped tasks before exporting.
