"""Command-line interface for the MedliNER pipeline stages.

The pipeline runs in three phases, each a subcommand (see the Makefile wrappers):

- before Label Studio: ``prepare`` builds the sampled import file and attaches GLiNER
  suggestions in one go (``ingest``, ``candidates``, ``prelabel``, and the opt-in
  ``shorten`` remain available as individual stages);
- Label Studio: ``label-studio``/``label-studio-stop`` manage the podman annotation server;
  the optional Onboarding project is provisioned by ``onboarding``, which assigns quiz
  attempts to every annotator account at once (presentation mode), and
  ``onboarding-promote`` exports, scores, and promotes every passing annotator in one go;
- after Label Studio: ``pipeline`` runs dataset → splits → train → evaluate → bundle.

Configuration comes from the ``MEDLINER_*`` environment variables, with flags overriding
where offered. Heavy ML imports (training, evaluation) are deferred to their subcommands so
the data-stage commands stay stdlib-light.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .candidates import (
    build_import_tasks,
    build_warmup_tasks,
    hash_candidates_file,
    import_file_name,
    import_manifest,
    read_candidates,
    sample_tasks,
    stagger_tasks,
    write_import_file,
)
from .dataset import hash_file, manifest_for, read_examples, write_examples, write_manifest
from .export_ingest import ingest_export
from .label_studio import normalize_export
from .label_studio_server import (
    DEFAULT_IMAGE,
    DEFAULT_PORT,
    DEFAULT_PROJECT_TITLE,
    ONBOARDING_PROJECT_TITLE,
    WARMUP_PROJECT_TITLE,
    export_project,
    provision,
    stop_container,
)
from .onboarding import (
    DEFAULT_CONFIG_PATH as ONBOARDING_CONFIG_PATH,
)
from .onboarding import (
    OnboardingError,
    build_onboarding_tasks,
    build_test_bank,
    evaluate_attempt,
    filter_production_export,
    promoted_users,
    read_attempts,
    start_attempt,
    versioned_bank_path,
    write_current_bank_pointer,
    write_report,
    write_test_bank,
)
from .onboarding import (
    load_config as load_onboarding_config,
)
from .onboarding import (
    promote as promote_onboarding_user,
)
from .packaging import build_export_bundle
from .splits import assert_no_group_leakage, group_key, split_examples


def workdir() -> Path:
    return Path(os.environ.get("MEDLINER_WORKDIR", "data/materialized"))


def repo_root() -> Path:
    """Anchor committed configs regardless of the caller's working directory."""
    return Path(__file__).resolve().parents[2]


DEFAULT_SAMPLE_TARGETS = "indication:3000,contraindication:2000"
DEFAULT_SAMPLE_SEED = 2026
DEFAULT_SAMPLE_MAX_WORDS = 300
#: Shorten stage threshold: texts over this many words (≈ 3-4 short sentences) are rewritten.
DEFAULT_SHORTEN_MAX_WORDS = 48
#: Parallel chat requests; default matches the server's four slots (-np 4) with continuous
#: batching — extra requests would just queue client-side for no gain.
DEFAULT_SHORTEN_WORKERS = 4
DEFAULT_SAMPLE_MAX_RUN = 3
#: Share of each sampling stratum filled with the hardest texts; the rest stays hash-random.
DEFAULT_SAMPLE_EDGE_FRACTION = 0.8


@dataclass(frozen=True)
class SamplingSettings:
    """Resolved ``MEDLINER_SAMPLE_*`` configuration for the Label Studio import."""

    targets: dict[str, int]
    seed: int
    max_words: int
    max_run: int
    edge_fraction: float

    @property
    def config(self) -> str | None:
        """Canonical configuration string; ``None`` keeps the legacy unsampled import-file name."""
        if not self.targets:
            return None
        spec = ",".join(f"{name}:{self.targets[name]}" for name in sorted(self.targets))
        return (
            f"tasks={spec};seed={self.seed};max_words={self.max_words};"
            f"max_run={self.max_run};edge_fraction={self.edge_fraction}"
        )


def sampling_settings() -> SamplingSettings:
    """Parse the ``MEDLINER_SAMPLE_*`` environment; an empty/all task list disables sampling."""
    raw = os.environ.get("MEDLINER_SAMPLE_TASKS", DEFAULT_SAMPLE_TARGETS).strip()
    if raw.lower() in ("", "all", "none"):
        return SamplingSettings(
            {}, DEFAULT_SAMPLE_SEED, DEFAULT_SAMPLE_MAX_WORDS, DEFAULT_SAMPLE_MAX_RUN, DEFAULT_SAMPLE_EDGE_FRACTION
        )
    targets: dict[str, int] = {}
    for part in raw.split(","):
        name, separator, value = part.strip().partition(":")
        if not separator or not name:
            raise ValueError(f"MEDLINER_SAMPLE_TASKS expects task:count pairs, got {part!r}")
        if not value.strip().isdigit():
            raise ValueError(f"MEDLINER_SAMPLE_TASKS count for {name!r} must be a non-negative integer, got {value!r}")
        targets[name.strip().lower()] = int(value)
    settings = SamplingSettings(
        targets=targets,
        seed=int(os.environ.get("MEDLINER_SAMPLE_SEED", str(DEFAULT_SAMPLE_SEED))),
        max_words=int(os.environ.get("MEDLINER_SAMPLE_MAX_WORDS", str(DEFAULT_SAMPLE_MAX_WORDS))),
        max_run=int(os.environ.get("MEDLINER_SAMPLE_MAX_RUN", str(DEFAULT_SAMPLE_MAX_RUN))),
        edge_fraction=float(os.environ.get("MEDLINER_SAMPLE_EDGE_FRACTION", str(DEFAULT_SAMPLE_EDGE_FRACTION))),
    )
    if settings.seed < 0:
        raise ValueError("MEDLINER_SAMPLE_SEED must be non-negative")
    if settings.max_words < 0:
        raise ValueError("MEDLINER_SAMPLE_MAX_WORDS must be non-negative (0 disables the cap)")
    if settings.max_run < 1:
        raise ValueError("MEDLINER_SAMPLE_MAX_RUN must be at least 1")
    if not 0.0 <= settings.edge_fraction <= 1.0:
        raise ValueError("MEDLINER_SAMPLE_EDGE_FRACTION must be between 0 and 1")
    return settings


def raw_candidates_path(value: str | None = None) -> Path:
    raw = value or os.environ.get("MEDLINER_RAW_CANDIDATES", "data/label-studio/candidates.ndjson")
    path = Path(raw)
    if not path.exists():
        raise FileNotFoundError(f"raw candidates file not found: {path} (MEDLINER_RAW_CANDIDATES)")
    return path


def export_path(value: str | None = None) -> Path:
    raw = value or os.environ.get("MEDLINER_LABEL_STUDIO_EXPORT")
    if not raw:
        raise RuntimeError("set MEDLINER_LABEL_STUDIO_EXPORT to a reviewed Label Studio export")
    path = Path(raw)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def bundle_path(value: str | None = None) -> Path:
    raw = value or os.environ.get("MEDLINER_EXPORT_BUNDLE")
    if not raw:
        raise RuntimeError("set --bundle or MEDLINER_EXPORT_BUNDLE to a DAKP export bundle directory")
    path = Path(raw)
    if not path.is_dir():
        raise FileNotFoundError(f"export bundle directory not found: {path} (--bundle / MEDLINER_EXPORT_BUNDLE)")
    return path


def run_candidates(input_path: Path) -> Path:
    """Validate/dedupe/sample raw candidates into the Label Studio import file; returns its path."""
    from .candidates import difficulty_score

    settings = sampling_settings()
    tasks = build_import_tasks(read_candidates(input_path))
    if not tasks:
        raise ValueError(f"no import tasks produced from {input_path}")
    sampling_manifest: dict[str, Any] | None = None
    if settings.targets:
        pool_difficulty = [difficulty_score(task["data"]["text"]) for task in tasks]
        sampling_manifest = {
            "targets": settings.targets,
            "seed": settings.seed,
            "max_words": settings.max_words,
            "max_run": settings.max_run,
            "edge_fraction": settings.edge_fraction,
            "pool_task_counts": dict(sorted(Counter(task["data"]["task"] for task in tasks).items())),
            "pool_family_counts": dict(sorted(Counter(task["data"]["source_family"] for task in tasks).items())),
            "pool_difficulty_mean": round(sum(pool_difficulty) / len(pool_difficulty), 3),
        }
        tasks = sample_tasks(
            tasks,
            settings.targets,
            seed=settings.seed,
            max_words=settings.max_words or None,
            edge_fraction=settings.edge_fraction,
        )
        tasks = stagger_tasks(tasks, max_run=settings.max_run, seed=settings.seed)
        if not tasks:
            raise ValueError(
                f"sampling produced no tasks from {input_path} "
                f"(targets {settings.targets}, max_words {settings.max_words})"
            )
        # Shorten only what was sampled: the batch that annotators will actually see. When the
        # LLM is down this is skipped and long texts simply stay as-is (the sampling cap above
        # still bounds them), so prepare remains usable offline.
        from . import llm

        if llm.health(url=None):
            shorten_stats = shorten_task_texts(tasks, max_words=shorten_max_words(), url=None)
            sampling_manifest["llm_shorten"] = shorten_stats
            print(
                f"candidates: shortened {shorten_stats['shortened']}/"
                f"{shorten_stats['over_threshold']} over-{shorten_stats['max_words']}-word texts via LLM"
            )
        else:
            print(
                f"candidates: LLM not healthy at {llm.llm_url()}; skipping text shortening (start 'make llm' to enable)"
            )
        selected_difficulty = [difficulty_score(task["data"]["text"]) for task in tasks]
        sampling_manifest["selected_difficulty_mean"] = round(sum(selected_difficulty) / len(selected_difficulty), 3)
    manifest = import_manifest(tasks, input_path=input_path, sampling=sampling_manifest)
    output = workdir() / "label-studio" / import_file_name(input_hash=manifest["input_hash"], sampling=settings.config)
    write_import_file(tasks, output)
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    composition = ", ".join(f"{count} {name}" for name, count in manifest["task_counts"].items())
    if sampling_manifest:
        print(
            f"candidates: sampled {manifest['task_count']} tasks ({composition}; "
            f"{manifest['duplicates_merged']} duplicates merged; runs capped at {settings.max_run}) -> {output}"
        )
    else:
        print(
            f"candidates: {manifest['task_count']} tasks ({manifest['duplicates_merged']} duplicates merged) -> {output}"
        )
    return output


def ensure_import_file(input_path: Path) -> Path:
    """Return the import file for the current input hash and sampling config, building when absent."""
    expected = (
        workdir()
        / "label-studio"
        / import_file_name(input_hash=hash_candidates_file(input_path), sampling=sampling_settings().config)
    )
    return expected if expected.exists() else run_candidates(input_path)


def run_prelabel(
    input_path: Path,
    *,
    model_id: str,
    threshold: float,
    device: str | None,
    batch_size: int,
    word_budget: int,
    max_width: int,
    force: bool,
) -> Path:
    """Attach GLiNER suggestions to the import file; returns the pre-labeled file's path.

    The model is loaded lazily: a run whose texts are all in the prelabel cache never imports
    torch at all, which is what makes re-running this after adding a few candidates cheap.
    """
    from . import prelabel

    import_file = ensure_import_file(input_path)
    tasks = json.loads(import_file.read_text(encoding="utf-8"))
    texts = {str(task["id"]): str(task["data"]["text"]) for task in tasks}
    cache_path = workdir() / "label-studio" / "prelabel-cache.json"
    cache = prelabel.PrelabelCache(cache_path)
    if not force:
        cache.load()

    state: dict[str, Any] = {"device": device or "not loaded", "predict": None}

    def predict(batch: Sequence[str]) -> list[list[dict[str, Any]]]:
        if state["predict"] is None:
            model, resolved = prelabel.load_model(model_id, device=device)
            prelabel.check_model_budgets(model, budget=word_budget, max_width=max_width)
            state["device"] = resolved
            state["predict"] = prelabel.batch_predictor(
                model, threshold=threshold, labels=prelabel.PRELABEL_LABELS, batch_size=batch_size
            )
            print(f"prelabel: loaded {model_id} on {resolved}")
        return state["predict"](batch)

    drops: Counter[str] = Counter()
    started = time.monotonic()
    suggestions = prelabel.prelabel_texts(
        predict,
        texts,
        budget=word_budget,
        max_width=max_width,
        cache=cache,
        model_id=model_id,
        threshold=threshold,
        drops=drops,
    )
    elapsed = time.monotonic() - started
    cache.save()

    version = prelabel.model_version(model_id, threshold)
    output = import_file.with_name(f"{import_file.stem}.prelabeled.json")
    write_import_file(prelabel.attach_predictions(tasks, suggestions, version=version), output)
    manifest = prelabel.prelabel_manifest(
        tasks,
        suggestions,
        model_id=model_id,
        threshold=threshold,
        labels=prelabel.PRELABEL_LABELS,
        budget=word_budget,
        max_width=max_width,
        device=str(state["device"]),
        version=version,
        drops=drops,
        elapsed_seconds=elapsed,
        cache_hits=cache.hits,
        cache_misses=cache.misses,
    )
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"prelabel: {manifest['suggestion_count']} suggestions {manifest['label_counts']} "
        f"across {manifest['tasks_with_suggestions']}/{manifest['task_count']} tasks "
        f"({cache.hits} cached, {cache.misses} predicted) -> {output}"
    )
    print(f"prelabel: dropped {manifest['dropped']} (suggestions only; a human accepts or replaces every span)")
    return output


def score_prelabeler(*, model_id: str, threshold: float, device: str | None, word_budget: int, max_width: int) -> None:
    """Score the pre-labeler against the ingested gold benchmark.

    Pre-labels that are worse than the annotators' own first guess cost time instead of saving
    it, so this is the gate to run before showing suggestions to a room full of people.
    """
    from . import prelabel
    from .evaluation import benchmark_path, load_gold_benchmark, score_examples

    gold = benchmark_path()
    if not gold.exists():
        raise FileNotFoundError(f"gold benchmark not found: {gold} (check MEDLINER_BENCHMARK)")
    model, resolved = prelabel.load_model(model_id, device=device)
    prelabel.check_model_budgets(model, budget=word_budget, max_width=max_width)
    batch = prelabel.batch_predictor(model, threshold=threshold, labels=prelabel.PRELABEL_LABELS)

    def predictor(text: str) -> list[dict[str, Any]]:
        return [
            span.as_dict()
            for span in prelabel.suggest(
                lambda window: batch([window])[0], text, budget=word_budget, max_width=max_width
            )
        ]

    report = score_examples(predictor, load_gold_benchmark(gold))
    strict = report["overall"]["strict"]
    boundary = report["overall"]["boundary_only"]
    print(
        f"prelabel score ({resolved}): strict P {strict['precision']:.3f} R {strict['recall']:.3f} F1 {strict['f1']:.3f}"
    )
    print(f"prelabel score: boundary-only F1 {boundary['f1']:.3f} over {report['examples']} gold cases")
    print(f"prelabel score: no-entity false-positive rate {report['no_entity']['false_positive_rate']:.3f}")


def run_dataset(path: Path) -> Path:
    """Validate the reviewed export into the normalized dataset; returns the JSONL path."""
    source_path = path
    excluded = 0
    if os.environ.get("MEDLINER_ONBOARDING_REQUIRED", "0").lower() in {"1", "true", "yes", "on"}:
        _config, bank, _bank_path = _onboarding_context()
        allowed = promoted_users(workdir(), bank)
        if not allowed:
            raise RuntimeError("no annotator has passed onboarding for the current test-bank; run onboarding-promote")
        source_path, audit_path, excluded = filter_production_export(path, workdir(), bank, allowed)
        if not source_path.exists():
            raise RuntimeError(f"onboarding produced no eligible production export; audit: {audit_path}")
        print(f"dataset: onboarding gate allowed {len(allowed)} users; excluded {excluded} task(s) -> {audit_path}")
    examples = normalize_export(source_path, require_reviewed=True)
    if not examples:
        raise ValueError(f"normalized dataset from {source_path} is empty")
    output = workdir() / "normalized" / "examples.jsonl"
    write_examples(examples, output)
    manifest = manifest_for(examples, input_export_hash=hash_file(source_path), dataset_id=hash_file(output))
    write_manifest(manifest, output.with_name("manifest.json"))
    print(f"dataset: {len(examples)} examples -> {output}")
    return output


def run_splits(dataset_path: Path) -> Path:
    """Freeze grouped train/validation/test splits; returns the split directory."""
    examples = read_examples(dataset_path)
    regression_path = os.environ.get("MEDLINER_REGRESSION_IDS")
    regression_ids = set(json.loads(Path(regression_path).read_text(encoding="utf-8"))) if regression_path else set()
    splits, manifest = split_examples(
        examples, seed=int(os.environ.get("MEDLINER_SPLIT_SEED", "2026")), regression_ids=regression_ids
    )
    try:
        assert_no_group_leakage(splits)
    except AssertionError as exc:
        raise RuntimeError(str(exc)) from exc
    # Must be the splitter's own grouping, or this guard measures a different partition.
    if len({group_key(item) for item in examples}) >= 3 and (not splits["validation"] or not splits["test"]):
        raise RuntimeError("at least three source groups require non-empty validation and test splits")
    output_dir = workdir() / "splits"
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, members in splits.items():
        write_examples(members, output_dir / f"{name}.jsonl")
    (output_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(f"splits: {manifest.example_count} examples ({len(manifest.held_out_ids)} held out) -> {output_dir}")
    return output_dir


def run_training(smoke: bool) -> Path:
    """Fine-tune the small GLiNER checkpoint (or the one-step smoke check); returns the checkpoint dir."""
    from .training import train_from_split_directory

    output = workdir() / "training"
    config_path = os.environ.get("MEDLINER_TRAIN_CONFIG", "configs/train-small.yaml")
    result = train_from_split_directory(workdir() / "splits", output, config_path=config_path, smoke_test=smoke)
    print(f"training ({'smoke' if smoke else 'full'}): checkpoint -> {result}")
    return result


def run_evaluation(checkpoint: Path, split_dir: Path) -> Path:
    """Score the tuned checkpoint against baselines; returns the report path."""
    from .evaluation import evaluate_checkpoint

    output = workdir() / "evaluation" / "report.json"
    result = evaluate_checkpoint(checkpoint, split_dir, output)
    print(f"evaluation: strict F1 {result['tuned']['overall']['strict']['f1']:.3f} -> {output}")
    return output


def run_bundle(checkpoint: Path, report: Path, dataset_path: Path, split_dir: Path) -> Path:
    """Assemble the standalone export bundle; returns its directory."""
    result = build_export_bundle(
        checkpoint_dir=checkpoint,
        evaluation_path=report,
        dataset_path=dataset_path,
        split_dir=split_dir,
        output_dir=workdir() / "bundle",
        annotation_policy_path=repo_root() / "docs" / "ANNOTATION_GUIDE.md",
        training_config_path=os.environ.get("MEDLINER_TRAIN_CONFIG", "configs/train-small.yaml"),
    )
    print(f"bundle: {result}")
    return result


def cmd_ingest(args: argparse.Namespace) -> None:
    result = ingest_export(bundle_path(args.bundle))
    print(
        f"ingest: {result['candidate_rows']} candidates {result['task_counts']}, "
        f"{result['gold_cases']} gold cases {result['family_counts']} -> {result['candidates_path'].parent}"
    )
    print(f"next: medliner candidates --input {result['candidates_path']}")


def cmd_candidates(args: argparse.Namespace) -> None:
    run_candidates(raw_candidates_path(args.input))


def shorten_max_words() -> int:
    """Row length threshold for the shorten stage ($MEDLINER_SHORTEN_MAX_WORDS, default 48)."""
    return int(os.environ.get("MEDLINER_SHORTEN_MAX_WORDS", str(DEFAULT_SHORTEN_MAX_WORDS)))


def rewrite_texts(texts: list[str], *, max_words: int, url: str | None) -> list[tuple[str, bool, bool]]:
    """Rewrite each text through the local LLM (validated); returns per input
    ``(shortened_text, empty_hint, cached_hit)``. Failures keep the original text."""
    from concurrent.futures import ThreadPoolExecutor

    from . import llm

    workers = max(1, int(os.environ.get("MEDLINER_SHORTEN_WORKERS", str(DEFAULT_SHORTEN_WORKERS))))
    cache_path = llm.default_cache_path()

    def rewrite_one(text: str) -> tuple[str, bool, bool]:
        cached = llm.cache_lookup(cache_path, text, max_words=max_words) is not None
        shortened, empty_hint = llm.shorten_text(text, max_words=max_words, url=url, cache=cache_path)
        return shortened, empty_hint, cached

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(rewrite_one, texts))


def shorten_task_texts(tasks: list[dict[str, Any]], *, max_words: int, url: str | None) -> dict[str, Any]:
    """Rewrite sampled import-task texts over ``max_words`` words through the LLM, in place.

    Only the already-sampled batch is sent to the model — never the whole candidate pool.
    Every rewrite is validated by :func:`medliner.llm.shorten_text`; failures and empty
    hints keep the original text. Returns manifest statistics.
    """
    from .candidates import _word_count

    long_indices = [index for index, task in enumerate(tasks) if _word_count(task["data"]["text"]) > max_words]
    stats: dict[str, Any] = {
        "max_words": max_words,
        "over_threshold": len(long_indices),
        "shortened": 0,
        "empty_hints": 0,
        "cached_replies": 0,
    }
    results = rewrite_texts([tasks[index]["data"]["text"] for index in long_indices], max_words=max_words, url=url)
    for index, (shortened, empty_hint, cached_hit) in zip(long_indices, results, strict=True):
        original = tasks[index]["data"]["text"]
        if shortened != original:
            tasks[index]["data"]["text"] = shortened
            stats["shortened"] += 1
        stats["empty_hints"] += empty_hint
        stats["cached_replies"] += cached_hit
    return stats


def run_shorten(input_path: Path, *, limit: int | None, max_words: int, url: str | None, force: bool = False) -> Path:
    """Rewrite candidate texts over ``max_words`` words through the LLM; returns the output path.

    Opt-in stage, never part of the default flow. Every rewrite is validated by
    :func:`medliner.llm.shorten_text`; failures keep the original text and are counted in
    the manifest rather than silently corrupting the pool.

    Resumable: when the previous manifest matches this input (same file hash and
    threshold), rows already processed last run keep their rewritten text and only the
    remainder is sent to the model — an interrupted long run loses no work. Pass
    ``force=True`` (CLI ``--force``) to ignore prior progress and re-run everything.

    Successful replies are also persisted in a sqlite rewrite cache
    ($MEDLINER_SHORTEN_CACHE), so overlapping texts across candidate files are never sent
    to the model twice.
    """
    from . import llm
    from .candidates import _word_count

    if not llm.health(url):
        raise RuntimeError(f"LLM server not healthy at {llm.llm_url(url)} (start it with 'make llm')")
    candidates = read_candidates(input_path)
    rows = [candidate.model_dump(exclude_none=True) for candidate in candidates]
    original_texts = [row["text"] for row in rows]
    input_hash = hash_candidates_file(input_path)
    over_long = [index for index, row in enumerate(rows) if _word_count(row["text"]) > max_words]

    output = input_path.with_name(f"{input_path.stem}.shortened{input_path.suffix}")
    manifest_path = output.with_suffix(".manifest.json")
    # index -> final text from the previous run; plus indices the model flagged as entity-free.
    done_texts: dict[int, str] = {}
    done_hints: set[int] = set()
    if not force and manifest_path.exists():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        except ValueError:
            previous = {}
        resumable = (
            previous.get("schema_version") == "medliner.shorten.manifest.v1"
            and previous.get("input_hash") == input_hash
            and previous.get("max_words") == max_words
            and isinstance(previous.get("processed_indices"), list)
            and output.exists()
        )
        if resumable:
            previous_rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            if len(previous_rows) == len(rows):
                for index in previous["processed_indices"]:
                    if isinstance(index, int) and 0 <= index < len(previous_rows):
                        done_texts[index] = previous_rows[index]["text"]
                done_hints = {i for i in previous.get("empty_hint_indices", []) if i in done_texts}

    pending = [index for index in over_long if index not in done_texts]
    if limit is not None:
        pending = pending[:limit]

    cache_path_str = str(llm.default_cache_path())
    resumed_count = len(done_texts)
    results = rewrite_texts([rows[index]["text"] for index in pending], max_words=max_words, url=url)
    cached_count = 0
    for index, (shortened, empty_hint, cached_hit) in zip(pending, results, strict=True):
        rows[index]["text"] = shortened
        done_texts[index] = shortened
        cached_count += cached_hit
        if empty_hint:
            done_hints.add(index)
    for index, text in done_texts.items():
        rows[index]["text"] = text

    # Only rows actually sent to the model count as processed; --limit rows stay pending so a
    # later resume picks them up.
    attempted = sorted(done_texts)
    shortened_count = sum(rows[index]["text"] != original_texts[index] for index in attempted)
    manifest = {
        "schema_version": "medliner.shorten.manifest.v1",
        "input_path": str(input_path),
        "input_hash": input_hash,
        "llm_url": llm.llm_url(url),
        "max_words": max_words,
        "rows": len(rows),
        "over_long": len(over_long),
        "shortened": shortened_count,
        "unchanged_over_long": len(attempted) - shortened_count,
        "empty_hints": len(done_hints),
        "resumed_from_previous": resumed_count,
        "cached_replies": cached_count,
        "cache": cache_path_str,
        "processed_indices": attempted,
        "empty_hint_indices": sorted(done_hints),
        "workers": max(1, int(os.environ.get("MEDLINER_SHORTEN_WORKERS", str(DEFAULT_SHORTEN_WORKERS)))),
    }
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"shorten: {shortened_count}/{len(attempted)} over-{max_words}-word texts shortened "
        f"({len(done_hints)} empty-entity hints; unchanged rows kept verbatim; "
        f"{resumed_count} resumed; {cached_count} served from cache) -> {output}"
    )
    return output


def cmd_shorten(args: argparse.Namespace) -> None:
    run_shorten(
        raw_candidates_path(args.input),
        limit=args.limit,
        max_words=args.max_words or shorten_max_words(),
        url=args.url,
        force=args.force,
    )
    print("next: MEDLINER_RAW_CANDIDATES=<shortened file> medliner candidates")


def _prelabel_options(args: argparse.Namespace) -> dict[str, Any]:
    from . import prelabel

    return {
        "model_id": args.model or os.environ.get("MEDLINER_PRELABEL_MODEL", prelabel.DEFAULT_MODEL_ID),
        "threshold": args.threshold
        if args.threshold is not None
        else float(os.environ.get("MEDLINER_PRELABEL_THRESHOLD", str(prelabel.DEFAULT_THRESHOLD))),
        "device": args.device or os.environ.get("MEDLINER_PRELABEL_DEVICE") or None,
        "word_budget": args.word_budget or prelabel.DEFAULT_WORD_BUDGET,
        "max_width": args.max_width or prelabel.DEFAULT_MAX_WIDTH,
    }


def cmd_prelabel(args: argparse.Namespace) -> None:
    options = _prelabel_options(args)
    if args.score_gold:
        score_prelabeler(**options)
        return
    run_prelabel(raw_candidates_path(args.input), batch_size=args.batch_size, force=args.force, **options)
    print("next: medliner label-studio --prelabel")


def cmd_prepare(_args: argparse.Namespace) -> None:
    """Build the sampled import file and attach GLiNER suggestions in one go."""
    import_file = run_candidates(raw_candidates_path())
    options = _prelabel_options(argparse.Namespace(model=None, threshold=None, device=None, word_budget=0, max_width=0))
    output = run_prelabel(import_file, batch_size=8, force=False, **options)
    print(f"prepare: import file with suggestions -> {output}")


def _onboarding_context() -> tuple[Any, Any, Path]:
    """Load the current versioned onboarding bank, creating its private sidecar if needed."""
    from .evaluation import benchmark_path

    config_path = Path(os.environ.get("MEDLINER_ONBOARDING_CONFIG", str(repo_root() / ONBOARDING_CONFIG_PATH)))
    config = load_onboarding_config(config_path)
    gold = benchmark_path()
    if not gold.exists():
        raise FileNotFoundError(f"gold benchmark not found: {gold} (check MEDLINER_BENCHMARK; run ingest if needed)")
    manifest = build_test_bank(gold, config)
    bank_path = versioned_bank_path(workdir(), manifest)
    if not bank_path.exists():
        write_test_bank(manifest, bank_path)
    write_current_bank_pointer(workdir(), manifest)
    return config, manifest, bank_path


def _label_studio_credentials() -> dict[str, Any]:
    return {
        "username": os.environ.get("MEDLINER_LABEL_STUDIO_USERNAME", "medliner@localhost"),
        "password": os.environ.get("MEDLINER_LABEL_STUDIO_PASSWORD", "medliner-local"),
        "token": os.environ.get("MEDLINER_LABEL_STUDIO_TOKEN") or None,
    }


def _onboarding_import_path(manifest: Any) -> Path:
    return workdir() / "onboarding" / f"import-{manifest.test_bank_hash}.json"


def cmd_onboarding(args: argparse.Namespace) -> None:
    """Provision the Onboarding project and assign quiz attempts to every account at once."""
    config, manifest, _bank_path = _onboarding_context()
    import_path = _onboarding_import_path(manifest)
    import_was_missing = not import_path.exists()
    if import_was_missing or args.reimport:
        write_import_file(build_onboarding_tasks(manifest), import_path)
    raw = os.environ.get("MEDLINER_LABEL_STUDIO_ANNOTATORS")
    annotator_values = [item.strip() for item in raw.split(",") if item.strip()] if raw else None
    result = provision(
        import_file=import_path,
        label_config_path=repo_root() / "configs" / "label_studio_ner.xml",
        port=args.port or int(os.environ.get("MEDLINER_LABEL_STUDIO_PORT", str(DEFAULT_PORT))),
        image=args.image or os.environ.get("MEDLINER_LABEL_STUDIO_IMAGE", DEFAULT_IMAGE),
        data_dir=workdir() / "label-studio" / "server-data",
        project_title=config.project_title or ONBOARDING_PROJECT_TITLE,
        publish_host=args.host or os.environ.get("MEDLINER_LABEL_STUDIO_HOST", "127.0.0.1"),
        annotators=_annotator_pairs(annotator_values),
        reimport=args.reimport,
        **_label_studio_credentials(),
    )
    if import_was_missing and result.get("existing_tasks", 0) and not args.reimport:
        raise OnboardingError(
            "a new onboarding bank was prepared but the project already has tasks; "
            "rerun with --reimport to replace the old bank"
        )
    admin = os.environ.get("MEDLINER_LABEL_STUDIO_USERNAME", "medliner@localhost")
    usernames = [name for name in result["usernames"] if name != admin]
    for username in usernames:
        attempt = start_attempt(workdir(), manifest, config, username)
        print(f"onboarding: {username} -> tasks {', '.join(attempt.selected_task_ids)}")
    print(
        f"onboarding: {result['tasks_in_project']} answer-free tasks at {result['url']} "
        f"for {len(usernames)} annotator(s) (project {config.project_title}; bank {manifest.test_bank_hash})"
    )
    print(f"onboarding: private bank -> {_bank_path}")


def cmd_onboarding_promote(_args: argparse.Namespace) -> None:
    """Export the Onboarding project, score every attempt, and promote everyone passing."""
    config, manifest, _bank_path = _onboarding_context()
    export_path = Path(os.environ.get("MEDLINER_ONBOARDING_EXPORT") or str(workdir() / "onboarding" / "export.json"))
    result = export_project(
        output_path=export_path,
        port=int(os.environ.get("MEDLINER_LABEL_STUDIO_PORT", str(DEFAULT_PORT))),
        project_title=config.project_title or ONBOARDING_PROJECT_TITLE,
        **_label_studio_credentials(),
    )
    print(f"onboarding-export: {result['tasks_annotated']}/{result['tasks_exported']} annotated tasks -> {export_path}")
    attempts = [
        item
        for item in read_attempts(workdir())
        if item.config_hash == manifest.config_hash and item.test_bank_hash == manifest.test_bank_hash
    ]
    if not attempts:
        raise OnboardingError("no onboarding attempts recorded; run 'make onboarding' first")
    passed: list[Any] = []
    for attempt in attempts:
        report = evaluate_attempt(export_path, workdir(), manifest, config, attempt)
        report_path = write_report(report, workdir())
        score = "incomplete" if report.score is None else f"{report.correct_tasks}/{report.total_tasks}"
        print(f"onboarding: {report.username}: {report.status} {score} -> {report_path}")
        if report.status == "passed":
            passed.append(report)
    if not passed:
        raise OnboardingError("no annotator has a passing attempt yet; nothing promoted")
    for report in sorted(passed, key=lambda item: item.username):
        record = promote_onboarding_user(workdir(), report, manifest)
        print(f"onboarding: promoted {record.username} for production (attempt {record.attempt_id})")


def cmd_label_studio(args: argparse.Namespace) -> None:
    input_path = raw_candidates_path(args.input)
    prelabel_version: str | None = None
    if args.prelabel:
        from . import prelabel as prelabel_module

        options = _prelabel_options(args)
        import_file = run_prelabel(input_path, batch_size=args.batch_size, force=args.force, **options)
        prelabel_version = prelabel_module.model_version(options["model_id"], options["threshold"])
    else:
        import_file = ensure_import_file(input_path)
        # A prepared session imports the suggestions too: pick up the pre-labeled file written
        # by 'medliner prepare'/'prelabel' when it exists for this exact import file.
        prelabeled = import_file.with_name(f"{import_file.stem}.prelabeled.json")
        manifest_path = prelabeled.with_suffix(".manifest.json")
        if prelabeled.exists():
            import_file = prelabeled
            if manifest_path.exists():
                prelabel_version = str(json.loads(manifest_path.read_text(encoding="utf-8"))["model_version"])
            print(f"label-studio: serving pre-labeled import file {prelabeled}")
    port = args.port or int(os.environ.get("MEDLINER_LABEL_STUDIO_PORT", str(DEFAULT_PORT)))
    image = args.image or os.environ.get("MEDLINER_LABEL_STUDIO_IMAGE", DEFAULT_IMAGE)
    host = args.host or os.environ.get("MEDLINER_LABEL_STUDIO_HOST", "127.0.0.1")
    annotator_values = args.annotator
    if not annotator_values:
        env_annotators = os.environ.get("MEDLINER_LABEL_STUDIO_ANNOTATORS")
        annotator_values = (
            [item.strip() for item in env_annotators.split(",") if item.strip()] if env_annotators else None
        )
    annotators = _annotator_pairs(annotator_values)
    credentials = {
        "username": os.environ.get("MEDLINER_LABEL_STUDIO_USERNAME", "medliner@localhost"),
        "password": os.environ.get("MEDLINER_LABEL_STUDIO_PASSWORD", "medliner-local"),
        "token": os.environ.get("MEDLINER_LABEL_STUDIO_TOKEN") or None,
    }
    result = provision(
        import_file=import_file,
        label_config_path=repo_root() / "configs" / "label_studio_ner.xml",
        port=port,
        image=image,
        data_dir=workdir() / "label-studio" / "server-data",
        project_title=DEFAULT_PROJECT_TITLE,
        publish_host=host,
        annotators=annotators,
        reimport=args.reimport,
        prelabel_model_version=prelabel_version,
        **credentials,
    )
    print(f"label-studio: {result['tasks_in_project']} tasks at {result['url']} (container {result['container']})")
    if result.get("prelabeled"):
        print(
            f"label-studio: model suggestions pre-filled from {prelabel_version}; "
            "annotators must accept, correct, or delete every span"
        )
    if result.get("annotators_created"):
        print(f"label-studio: created {result['annotators_created']} annotator account(s)")
    if host not in ("127.0.0.1", "localhost"):
        print(f"label-studio: reachable on the network via {host}:{port} (share http://<this-host>:{port})")
    if args.warmup:
        from .evaluation import benchmark_path

        gold = benchmark_path()
        if not gold.exists():
            raise FileNotFoundError(f"gold benchmark not found: {gold} (check MEDLINER_BENCHMARK)")
        warmup_file = workdir() / "label-studio" / "warmup.json"
        write_import_file(build_warmup_tasks(gold, limit=args.warmup_limit), warmup_file)
        warmup = provision(
            import_file=warmup_file,
            label_config_path=repo_root() / "configs" / "label_studio_ner.xml",
            port=port,
            image=image,
            data_dir=workdir() / "label-studio" / "server-data",
            project_title=WARMUP_PROJECT_TITLE,
            reimport=args.reimport,
            **credentials,
        )
        print(
            f"label-studio: {warmup['tasks_in_project']} warm-up tasks at {result['url']} "
            f"(project {WARMUP_PROJECT_TITLE}; gold answers travel with each task)"
        )


def _annotator_pairs(values: list[str] | None) -> list[tuple[str, str]]:
    """Parse ``username:password`` pairs; empty when none were requested."""
    pairs: list[tuple[str, str]] = []
    for value in values or []:
        username, separator, password = value.partition(":")
        if not separator or not username.strip() or not password:
            raise ValueError(f"--annotator expects username:password, got {value!r}")
        pairs.append((username.strip(), password))
    return pairs


def cmd_label_studio_export(args: argparse.Namespace) -> None:
    output = args.output or os.environ.get("MEDLINER_LABEL_STUDIO_EXPORT")
    if not output:
        raise RuntimeError("pass --output or set MEDLINER_LABEL_STUDIO_EXPORT for the export destination")
    result = export_project(
        output_path=output,
        port=int(os.environ.get("MEDLINER_LABEL_STUDIO_PORT", str(DEFAULT_PORT))),
        username=os.environ.get("MEDLINER_LABEL_STUDIO_USERNAME", "medliner@localhost"),
        password=os.environ.get("MEDLINER_LABEL_STUDIO_PASSWORD", "medliner-local"),
        token=os.environ.get("MEDLINER_LABEL_STUDIO_TOKEN") or None,
    )
    print(
        f"label-studio-export: {result['tasks_annotated']}/{result['tasks_exported']} annotated tasks "
        f"-> {result['output']}"
    )
    print("next: medliner dataset --export <file> (or make train with MEDLINER_LABEL_STUDIO_EXPORT set)")


def cmd_label_studio_stop(_args: argparse.Namespace) -> None:
    removed = stop_container()
    print("label-studio: container removed" if removed else "label-studio: no container to remove")


def cmd_dataset(args: argparse.Namespace) -> None:
    run_dataset(export_path(args.export))


def cmd_splits(args: argparse.Namespace) -> None:
    run_splits(Path(args.dataset) if args.dataset else workdir() / "normalized" / "examples.jsonl")


def cmd_train(args: argparse.Namespace) -> None:
    run_training(smoke=args.smoke)


def cmd_evaluate(args: argparse.Namespace) -> None:
    checkpoint = Path(args.checkpoint) if args.checkpoint else workdir() / "training" / "final"
    split_dir = Path(args.splits) if args.splits else workdir() / "splits"
    run_evaluation(checkpoint, split_dir)


def cmd_bundle(args: argparse.Namespace) -> None:
    materialized = workdir()
    run_bundle(
        checkpoint=Path(args.checkpoint) if args.checkpoint else materialized / "training" / "final",
        report=materialized / "evaluation" / "report.json",
        dataset_path=materialized / "normalized" / "examples.jsonl",
        split_dir=materialized / "splits",
    )


def cmd_pipeline(args: argparse.Namespace) -> None:
    """The full post-annotation chain: dataset → splits → train → evaluate → bundle."""
    dataset_path = run_dataset(export_path(args.export))
    split_dir = run_splits(dataset_path)
    checkpoint = run_training(smoke=args.smoke)
    report = run_evaluation(checkpoint, split_dir)
    run_bundle(checkpoint, report, dataset_path, split_dir)


def _add_prelabel_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags shared by ``prelabel`` and ``label-studio --prelabel``."""
    parser.add_argument("--model", help="GLiNER checkpoint (default: $MEDLINER_PRELABEL_MODEL)")
    parser.add_argument("--threshold", type=float, help="score floor (default: $MEDLINER_PRELABEL_THRESHOLD, 0.35)")
    parser.add_argument("--device", help="cuda/cpu override (default: $MEDLINER_PRELABEL_DEVICE, autodetected)")
    parser.add_argument("--batch-size", type=int, default=8, help="windows per forward pass (default: 8)")
    parser.add_argument("--word-budget", type=int, help="window size in GLiNER word tokens (default: 384)")
    parser.add_argument("--max-width", type=int, help="widest suggested span in word tokens (default: 12)")
    parser.add_argument("--force", action="store_true", help="ignore the prelabel cache and re-run the model")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="medliner", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="verify + materialize a DAKP export bundle")
    ingest.add_argument("--bundle", help="DAKP export bundle directory (default: $MEDLINER_EXPORT_BUNDLE)")
    ingest.set_defaults(func=cmd_ingest)

    candidates = sub.add_parser("candidates", help="build the Label Studio import file from raw candidates")
    candidates.add_argument("--input", help="raw candidates NDJSON (default: $MEDLINER_RAW_CANDIDATES)")
    candidates.set_defaults(func=cmd_candidates)

    shorten = sub.add_parser(
        "shorten",
        help="rewrite over-long candidate texts through the local LLM (opt-in; start it with 'make llm')",
    )
    shorten.add_argument("--input", help="raw candidates NDJSON (default: $MEDLINER_RAW_CANDIDATES)")
    shorten.add_argument("--limit", type=int, help="process at most this many not-yet-processed rows (for trial runs)")
    shorten.add_argument(
        "--max-words",
        type=int,
        help=f"row length threshold in words, ≈3-4 short sentences "
        f"(default: $MEDLINER_SHORTEN_MAX_WORDS, {DEFAULT_SHORTEN_MAX_WORDS})",
    )
    shorten.add_argument("--url", help="LLM server base URL (default: $MEDLINER_LLM_URL, http://127.0.0.1:8080)")
    shorten.add_argument("--force", action="store_true", help="ignore previous progress and re-shorten everything")
    shorten.set_defaults(func=cmd_shorten)

    prepare = sub.add_parser("prepare", help="build the sampled import file and attach GLiNER suggestions in one go")
    prepare.set_defaults(func=cmd_prepare)

    prelabel = sub.add_parser("prelabel", help="attach GLiNER model suggestions to the Label Studio import file")
    prelabel.add_argument("--input", help="raw candidates NDJSON (default: $MEDLINER_RAW_CANDIDATES)")
    prelabel.add_argument(
        "--score-gold",
        action="store_true",
        help="score the pre-labeler against the ingested gold benchmark instead of writing an import file",
    )
    _add_prelabel_arguments(prelabel)
    prelabel.set_defaults(func=cmd_prelabel)

    server = sub.add_parser("label-studio", help="start the podman Label Studio server with tasks imported")
    server.add_argument("--input", help="raw candidates NDJSON (default: $MEDLINER_RAW_CANDIDATES)")
    server.add_argument("--reimport", action="store_true", help="replace existing project tasks")
    server.add_argument("--port", type=int, help="host port (default: $MEDLINER_LABEL_STUDIO_PORT)")
    server.add_argument("--image", help="container image (default: $MEDLINER_LABEL_STUDIO_IMAGE)")
    server.add_argument("--host", help="port-publish bind address (default: $MEDLINER_LABEL_STUDIO_HOST, 127.0.0.1)")
    server.add_argument(
        "--annotator",
        action="append",
        metavar="USERNAME:PASSWORD",
        help="ensure an extra annotator account (repeatable; also MEDLINER_LABEL_STUDIO_ANNOTATORS, comma-separated)",
    )
    server.add_argument(
        "--warmup",
        action="store_true",
        help="also import gold-benchmark warm-up tasks into a separate project (needs the ingested benchmark)",
    )
    server.add_argument("--warmup-limit", type=int, default=10, help="maximum warm-up tasks to import (default: 10)")
    server.add_argument(
        "--prelabel",
        action="store_true",
        help="attach GLiNER suggestions first and turn on Label Studio's prediction pre-fill",
    )
    _add_prelabel_arguments(server)
    server.set_defaults(func=cmd_label_studio)

    export = sub.add_parser("label-studio-export", help="download the reviewed annotations from the running server")
    export.add_argument("--output", help="export destination (default: $MEDLINER_LABEL_STUDIO_EXPORT)")
    export.set_defaults(func=cmd_label_studio_export)

    onboarding = sub.add_parser(
        "onboarding",
        help="provision the Onboarding project and assign quiz attempts to every annotator account",
    )
    onboarding.add_argument("--reimport", action="store_true", help="replace the Onboarding project tasks")
    onboarding.add_argument("--port", type=int, help="host port (default: $MEDLINER_LABEL_STUDIO_PORT)")
    onboarding.add_argument("--image", help="container image (default: $MEDLINER_LABEL_STUDIO_IMAGE)")
    onboarding.add_argument("--host", help="port-publish bind address (default: $MEDLINER_LABEL_STUDIO_HOST)")
    onboarding.set_defaults(func=cmd_onboarding)

    onboarding_promote = sub.add_parser(
        "onboarding-promote",
        help="export the Onboarding project, score every attempt, promote everyone passing",
    )
    onboarding_promote.set_defaults(func=cmd_onboarding_promote)

    stop = sub.add_parser("label-studio-stop", help="remove the Label Studio container (annotations survive)")
    stop.set_defaults(func=cmd_label_studio_stop)

    dataset = sub.add_parser("dataset", help="validate the reviewed export into the normalized dataset")
    dataset.add_argument("--export", help="reviewed export (default: $MEDLINER_LABEL_STUDIO_EXPORT)")
    dataset.set_defaults(func=cmd_dataset)

    splits = sub.add_parser("splits", help="freeze grouped train/validation/test splits")
    splits.add_argument("--dataset", help="normalized JSONL (default: $MEDLINER_WORKDIR/normalized/examples.jsonl)")
    splits.set_defaults(func=cmd_splits)

    train = sub.add_parser("train", help="fine-tune the small GLiNER checkpoint")
    train.add_argument("--smoke", action="store_true", help="one-step GPU sanity check (run this first)")
    train.set_defaults(func=cmd_train)

    evaluate = sub.add_parser("evaluate", help="strict/lenient evaluation report for the tuned checkpoint")
    evaluate.add_argument("--checkpoint", help="checkpoint dir (default: $MEDLINER_WORKDIR/training/final)")
    evaluate.add_argument("--splits", help="split dir (default: $MEDLINER_WORKDIR/splits)")
    evaluate.set_defaults(func=cmd_evaluate)

    bundle = sub.add_parser("bundle", help="assemble the standalone export bundle")
    bundle.add_argument("--checkpoint", help="checkpoint dir (default: $MEDLINER_WORKDIR/training/final)")
    bundle.set_defaults(func=cmd_bundle)

    pipeline = sub.add_parser("pipeline", help="dataset → splits → train → evaluate → bundle")
    pipeline.add_argument("--export", help="reviewed export (default: $MEDLINER_LABEL_STUDIO_EXPORT)")
    pipeline.add_argument("--smoke", action="store_true", help="run the one-step smoke training check")
    pipeline.set_defaults(func=cmd_pipeline)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"medliner: error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
