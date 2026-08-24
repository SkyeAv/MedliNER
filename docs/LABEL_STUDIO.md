# Label Studio Community Edition workflow

Label Studio Community Edition is a free, locally self-hosted browser UI for this project. It
runs in a podman container managed by the MEDliNER CLI; it is not a MEDliNER Python
dependency.

## Managed server (default)

`make label-studio` starts the stock `heartexlabs/label-studio` image with podman,
waits for health, creates the `MEDliNER medical NER` project from
`configs/label_studio_ner.xml`, and imports the tasks built by `make candidates` (run
automatically when the import file for the current input hash is missing):

```bash
make label-studio
```

Open <http://localhost:9030> and log in with `$MEDLINER_LABEL_STUDIO_USERNAME` /
`$MEDLINER_LABEL_STUDIO_PASSWORD` (local defaults are in `.envrc`; the container image creates
that account on first boot). Server state — accounts, projects, annotations — lives in
`$MEDLINER_WORKDIR/label-studio/server-data` and survives container restarts.

Behavior notes:

- Re-running `make label-studio` reuses a running container and an existing project, and skips
  the import when the project already has tasks. To replace project tasks, run
  `REIMPORT=1 make label-studio`.
- If the default-account login ever fails (e.g. image behavior changes), create an account in
  the browser, copy an access token from Account & Settings, and set
  `MEDLINER_LABEL_STUDIO_TOKEN` in `.envrc.local`; the client then sends it as a Bearer
  credential and skips the login form. (Legacy `Token` API auth is disabled by default in
  Label Studio ≥ 1.23, which is why the default path uses session login.)
- Stop the server with `make label-studio-stop` (removes the container, keeps the data dir).

## Group annotation sessions (e.g. a presentation)

Label Studio Community Edition has **no limits on users, annotators, or tasks**, but it has
no task-assignment or role layer (those are Enterprise features): everyone with an account
on the shared instance sees every project. The managed flow supports a group session:

1. **Expose the server on the network** with `MEDLINER_LABEL_STUDIO_HOST=0.0.0.0`
   (default `127.0.0.1` keeps it private). Attendees then reach
   `http://<this-machine>:$MEDLINER_LABEL_STUDIO_PORT` from the same LAN; the CLI prints a
   reminder when the bind address is public.
2. **Pre-create annotator accounts** so nobody registers during the session:
   `ANNOTATORS="alice:pw-a,bob:pw-b" make label-studio` (repeatable `--annotator user:pass`
   flags, or `MEDLINER_LABEL_STUDIO_ANNOTATORS` comma-separated). Accounts are created via
   `POST /api/users` and existing usernames are skipped, so the command stays idempotent.
3. **Avoid collisions**: CE lets two annotators open the same task, and tasks are served in
   queue order. For a short session either assign each person a slice of the task list, or
   rely on the natural staggering of the sequential queue — both work with this pipeline
   because exports keep per-annotation authorship.
4. **Seed a warm-up round** with `WARMUP=1 make label-studio`: gold-benchmark cases are
   imported into a *separate* project (`MEDliNER medical NER — Warm-up`, `--warmup-limit`
   tasks, default 10) so annotators can practice and compare against known answers without
   gold leaking into the main queue. The gold spans travel with each task in its
   `gold_mentions` data field (visible in the Data Manager). The warm-up project needs the
   ingested benchmark (`make ingest`), and is never consumed by training.
5. **Speed up labeling** with the hotkeys baked into `configs/label_studio_ner.xml`:
   `1` = disease, `2` = phenotype, `3` = drug after selecting a span.

## Export

Download the reviewed annotations over the API instead of the browser:

```bash
make label-studio-export            # writes $MEDLINER_LABEL_STUDIO_EXPORT
OUTPUT=/tmp/reviewed.json make label-studio-export
```

The command finds the `MEDliNER medical NER` project by title, downloads the JSON export,
and prints the annotated/total task counts. The manual browser route still works: export as
JSON from the project UI and save it anywhere, then override `MEDLINER_LABEL_STUDIO_EXPORT`
in the ignored `.envrc.local` if the path differs from the example in `.envrc`.

```bash
cat > .envrc.local <<'EOF'
export MEDLINER_LABEL_STUDIO_EXPORT="$PWD/data/label-studio/indications-2026-01.json"
EOF
direnv allow
make pipeline
```

## Import task JSON

Candidate tasks come from `make candidates` (see `docs/CANDIDATE_TASKS.md`). Each
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
into the project's labeling configuration, and import the JSON file written by
`make candidates` (its path is printed to stdout).

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

## Export details worth knowing

The raw export is retained as provenance. MEDliNER converts it to its canonical schema and validates offsets, labels, overlap, task metadata, review status, and text slices before training. JSONL is also accepted by MEDliNER when one task object is stored per line.

Two export details are worth knowing before the first review round:

- **Whitespace.** Dragging across a phrase usually captures the following space. The importer
  checks the exported offsets against the source text, then trims the edges and re-derives the
  surface, so annotators do not have to be precise about it.
- **Skipped tasks.** A task whose annotations are all `was_cancelled` is rejected rather than
  imported as an empty example, because an empty example is a positive claim that the text has no
  entity. Resolve or remove skipped tasks before exporting.
