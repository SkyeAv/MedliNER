# Label Studio Community Edition workflow

Label Studio Community Edition is a free, locally self-hosted browser UI for this project. Keep it outside the MEDliNER Python environment: it is only used to create reviewed exports.

## Install locally

Choose one method:

```bash
# Dedicated environment, separate from MEDliNER
python -m venv .label-studio-venv
source .label-studio-venv/bin/activate
pip install label-studio
label-studio start --port 8080
```

or:

```bash
docker run --rm -it -p 8080:8080 \
  -v "$PWD/.label-studio-data:/label-studio/data" \
  heartexlabs/label-studio:latest
```

Open <http://localhost:8080>, create a local account/project, and paste the contents of
`configs/label_studio_ner.xml` into the project's labeling configuration.

## Import task JSON

Candidate tasks are JSON/JSONL files produced outside MEDliNER. Each task must expose at least:

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
