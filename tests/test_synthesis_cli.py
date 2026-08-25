"""Tests for the ``medliner synthesize`` CLI stage.

The synthesis engine's gates are pinned in ``test_synthesis.py``; these tests pin the stage
contract around them: the 10x target pool with a full manifest, the loud acceptance floor, the
health gate before generation, resume via manifest slots, ``--force`` cache bypass, ``--limit``
trials that still enforce the floor, ``MEDLINER_SYNTH_*`` env parsing, and byte-identical
determinism on a warm-cache rerun.
"""

from __future__ import annotations

import json
import shutil
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from medliner import cli
from medliner.dataset import read_examples, write_examples
from medliner.schema import Annotation, Example, SourceMetadata

REPLY_A = "Indicated for asthma and chronic eczema in adults."
REPLY_A_ALT = "Approved for use in asthma and chronic eczema therapy."
REPLY_B = "Avoid use in patients with severe migraine present."
REPLY_B_ALT = "Avoid when severe migraine is active."
#: Passes no gate that depends on mentions, so every slot rejects with 'missing_mention'.
UNRELATED = "completely unrelated filler text"
SYNTH_ENV_VARS = (
    "MEDLINER_SYNTH_RATIO",
    "MEDLINER_SYNTH_MIN_RATIO",
    "MEDLINER_SYNTH_MAX_ATTEMPTS",
    "MEDLINER_SYNTH_MAX_WORDS",
    "MEDLINER_SYNTH_MIN_SIMILARITY",
    "MEDLINER_SYNTH_WORKERS",
    "MEDLINER_SYNTH_CACHE",
)


def _annotated(text: str, mention_labels: list[tuple[str, str]], **kwargs) -> Example:
    """Valid reviewed Example with spans located sequentially (never hand-counted offsets)."""
    kwargs.setdefault("task", "indication")
    annotations: list[Annotation] = []
    cursor = 0
    for mention, label in mention_labels:
        start = text.index(mention, cursor)
        end = start + len(mention)
        annotations.append(Annotation(start=start, end=end, label=label, text=mention))
        cursor = end
    return Example(
        text=text,
        source=SourceMetadata(family="dailymed", document_id=f"doc-{kwargs['id']}"),
        annotations=annotations,
        **kwargs,
    )


def _gold() -> list[Example]:
    """Two reviewed gold examples covering both tasks and both entity labels."""
    return [
        _annotated(
            "Approved for asthma and chronic eczema.",
            [("asthma", "disease"), ("chronic eczema", "phenotype")],
            id="gold-a",
            task="indication",
        ),
        _annotated(
            "Avoid in patients with severe migraine.",
            [("severe migraine", "disease")],
            id="gold-b",
            task="contraindication",
        ),
    ]


class StubSynthHandler(BaseHTTPRequestHandler):
    """/health + /v1/chat/completions stub that answers from the prompt's own text.

    Replies are selected by marker substrings of the source text inside the prompt, so the stub
    stays correct under parallel workers and any slot ordering. An optional ``script`` list pins
    replies by request order (single-worker tests only) and takes precedence.
    """

    marker_replies: dict[str, str] = {}
    default_reply: str = UNRELATED
    script: list[str] = []
    request_count: int = 0

    def log_message(self, *_args):  # keep the test output quiet
        pass

    def do_GET(self):
        if self.path == "/health":
            self._respond({"status": "ok"})
        else:
            self._respond({"error": "not found"}, status=404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length).decode())
        type(self).request_count += 1
        prompt = str(body["messages"][-1]["content"])
        if type(self).script:
            reply = type(self).script.pop(0)
        else:
            reply = next((text for marker, text in type(self).marker_replies.items() if marker in prompt), None)
            if reply is None:
                reply = type(self).default_reply
        self._respond({"choices": [{"message": {"role": "assistant", "content": reply}}]})

    def _respond(self, payload: dict, *, status: int = 200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def server():
    StubSynthHandler.marker_replies = {"eczema": REPLY_A, "migraine": REPLY_B}
    StubSynthHandler.default_reply = UNRELATED
    StubSynthHandler.script = []
    StubSynthHandler.request_count = 0
    httpd = HTTPServer(("127.0.0.1", 0), StubSynthHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()
    thread.join()


def _stage(monkeypatch, server_url: str, workdir: Path, **synth_env: str) -> Path:
    """Point the stage at the stub server and a scratch workdir; SYNTH_* starts clean."""
    for name in SYNTH_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MEDLINER_LLM_URL", server_url)
    monkeypatch.setenv("MEDLINER_WORKDIR", str(workdir))
    for key, value in synth_env.items():
        monkeypatch.setenv(f"MEDLINER_SYNTH_{key}", str(value))
    return workdir


def _write_train(workdir: Path, examples: list[Example]) -> Path:
    splits = workdir / "splits"
    splits.mkdir(parents=True, exist_ok=True)
    write_examples(examples, splits / "train.jsonl")
    return splits


def _synthetic_dir(workdir: Path) -> Path:
    return workdir / "synthetic"


# -----------------------------------------------------------------------------------------------
# Stage contract
# -----------------------------------------------------------------------------------------------


def test_synthesize_help_documents_the_stage(capsys):
    """--help must keep documenting the knobs operators actually tune (target, floor, limits).

    The stage is operated by humans mid-annotation-campaign; a flag that silently disappears
    from --help (or loses its 'make llm' hint) becomes undiscoverable exactly when needed.
    """
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["synthesize", "--help"])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    for flag in (
        "--ratio",
        "--min-ratio",
        "--min-similarity",
        "--max-attempts",
        "--limit",
        "--force",
        "--url",
        "--workers",
    ):
        assert flag in out
    assert "make llm" in out


def test_synthesize_generates_the_target_pool_with_a_full_manifest(server, tmp_path, monkeypatch, capsys):
    """A healthy run fills ratio variants per gold example and writes an auditable manifest.

    Ten distinct prompt styles per source give every slot a unique id and cache key; the
    manifest must carry everything an operator needs to trust the pool: counters that partition
    exactly, achieved ratio against target and floor, similarity stats, label/task mixes, and
    the LLM URL the pool actually came from.
    """
    workdir = _stage(monkeypatch, server, tmp_path / "work", WORKERS=4)  # exercises the bounded pool
    _write_train(workdir, _gold())

    assert cli.main(["synthesize"]) == 0
    examples = read_examples(_synthetic_dir(workdir) / "examples.jsonl")
    assert len(examples) == 20
    assert [item.id for item in examples] == sorted(item.id for item in examples)  # deterministic ordering
    assert all(item.source.family == "synthetic" for item in examples)
    assert all(annotation.provenance == "synthetic" for item in examples for annotation in item.annotations)

    manifest = json.loads((_synthetic_dir(workdir) / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "medliner.synthesis.manifest.v1"
    assert manifest["ratio"] == 10 and manifest["min_ratio"] == 5.0
    assert manifest["gold_count"] == 2 and manifest["target_count"] == 20 and manifest["floor_count"] == 10.0
    assert manifest["accepted"] == 20 and manifest["achieved_ratio"] == 10.0
    assert manifest["llm_url"] == server
    counters = manifest["counters"]
    assert counters["attempts"] == 20 and counters["accepted_this_run"] == 20
    assert counters["attempts"] == counters["accepted_this_run"] + sum(counters["rejections"].values())
    assert manifest["gate"] == {"passed": True, "enforced": True, "required": 10.0}
    assert manifest["trial"] is False and manifest["resumed"] == 0
    assert manifest["similarity"]["min"] is not None and manifest["similarity"]["min"] >= 0.3
    assert manifest["similarity"]["min"] <= manifest["similarity"]["mean"] <= manifest["similarity"]["max"] <= 1.0
    assert manifest["label_counts"] == {"disease": 20, "phenotype": 10}
    assert manifest["task_counts"] == {"contraindication": 10, "indication": 10}
    slots = manifest["slots"]
    assert len(slots) == 20
    assert slots == sorted(slots, key=lambda record: (record["source_id"], record["slot"]))  # variants by slot
    per_gold_a = [record["variant"] for record in slots if record["source_id"] == "gold-a"]
    assert per_gold_a == list(cli.SYNTH_VARIANT_STYLES)  # slot k gets style k; every slot distinct
    assert all(record["attempts"] == 1 for record in slots)
    assert "20/20 variants accepted" in capsys.readouterr().out


def test_synthesize_fails_below_min_ratio(server, tmp_path, monkeypatch, capsys):
    """Below the floor the run exits 1 with the shortfall spelled out — after writing outputs.

    The manifest and (here empty) examples file must exist even on failure: operators need the
    rejection accounting to decide whether to fix the prompt, the floor, or the server, and a
    missing artifact would hide exactly the evidence that matters.
    """
    workdir = _stage(monkeypatch, server, tmp_path / "work", MAX_ATTEMPTS=1, WORKERS=1)
    StubSynthHandler.marker_replies = {}  # every reply is the unrelated filler → missing_mention
    _write_train(workdir, _gold())

    assert cli.main(["synthesize"]) == 1
    error = capsys.readouterr().err
    assert "floor requires" in error
    assert "MEDLINER_SYNTH_MIN_RATIO" in error
    assert "manifest" in error
    manifest = json.loads((_synthetic_dir(workdir) / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["accepted"] == 0 and manifest["gate"]["passed"] is False
    assert manifest["counters"]["rejections"]["missing_mention"] == 20
    assert (_synthetic_dir(workdir) / "examples.jsonl").exists()


def test_synthesize_refuses_when_llm_down(tmp_path, monkeypatch, capsys):
    """An unhealthy server is refused up front with the fix in the message, before any output.

    Generating against a dead server would burn the whole run into llm_error rejections; the
    stage must fail fast, point at 'make llm', and leave no partial artifacts behind.
    """
    workdir = _stage(monkeypatch, "http://127.0.0.1:1", tmp_path / "work")
    _write_train(workdir, _gold())
    assert cli.main(["synthesize"]) == 1
    error = capsys.readouterr().err
    assert "not healthy" in error
    assert "make llm" in error
    assert not _synthetic_dir(workdir).exists()


def test_synthesize_requires_an_existing_non_empty_train_split(server, tmp_path, monkeypatch, capsys):
    """Missing and empty train splits are loud input errors, not zero-gold runs.

    With zero gold examples every floor passes vacuously (0 accepted >= 0 required), so an
    empty split would silently write an empty pool that looks like a successful synthesis.
    """
    workdir = _stage(monkeypatch, server, tmp_path / "work")
    assert cli.main(["synthesize"]) == 1  # no splits directory at all
    assert "train.jsonl" in capsys.readouterr().err

    splits = workdir / "splits"
    splits.mkdir(parents=True)
    (splits / "train.jsonl").write_text("", encoding="utf-8")
    assert cli.main(["synthesize"]) == 1
    assert "empty" in capsys.readouterr().err


def test_synthesize_refuses_unannotated_gold_examples(server, tmp_path, monkeypatch, capsys):
    """Gold rows without annotations are refused before any request: nothing can be preserved.

    The engine raises for unannotated sources; without an up-front check that would surface as
    a mid-run crash after burning server time on the other slots.
    """
    workdir = _stage(monkeypatch, server, tmp_path / "work")
    _write_train(workdir, [Example(id="bare", text="no entities here", task="indication")])
    assert cli.main(["synthesize"]) == 1
    assert "annotations" in capsys.readouterr().err
    assert StubSynthHandler.request_count == 0


def test_synthesize_resumes_an_interrupted_run(server, tmp_path, monkeypatch, capsys):
    """An interrupted run loses nothing: completed slots resume from the manifest.

    The limited trial run ends below the floor and so exits non-zero, but it still writes its
    outputs — that manifest is exactly what the resume consumes. The resume must neither
    re-request accepted variants (the server counter proves it) nor duplicate them, and the
    final manifest accounts for both the resumed examples and this run's fresh attempts, so the
    floor gate judges the finished pool, not the last session.
    """
    workdir = _stage(monkeypatch, server, tmp_path / "work", WORKERS=1)
    _write_train(workdir, _gold())

    assert cli.main(["synthesize", "--limit", "4"]) == 1  # 4 accepted < floor of 10: loud trial
    assert "floor requires" in capsys.readouterr().err
    first = read_examples(_synthetic_dir(workdir) / "examples.jsonl")
    assert len(first) == 4
    assert StubSynthHandler.request_count == 4

    assert cli.main(["synthesize"]) == 0
    assert StubSynthHandler.request_count == 20  # only the 16 pending slots hit the server
    resumed = read_examples(_synthetic_dir(workdir) / "examples.jsonl")
    assert len(resumed) == 20
    assert {item.id for item in first} <= {item.id for item in resumed}
    manifest = json.loads((_synthetic_dir(workdir) / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["resumed"] == 4
    assert manifest["counters"]["attempts"] == 16
    assert manifest["counters"]["accepted_this_run"] == 16
    assert manifest["accepted"] == 20 and manifest["achieved_ratio"] == 10.0
    assert len(manifest["slots"]) == 20
    assert manifest["gate"]["passed"] is True


def test_synthesize_force_regenerates_everything_without_the_cache(server, tmp_path, monkeypatch):
    """--force ignores previous progress and the reply cache; every slot goes back to the server.

    Without the cache bypass a --force run would silently replay the old replies and change
    nothing — the one operator outcome --force exists to guarantee.
    """
    workdir = _stage(monkeypatch, server, tmp_path / "work", WORKERS=1)
    _write_train(workdir, _gold())
    assert cli.main(["synthesize"]) == 0
    assert StubSynthHandler.request_count == 20
    first_texts = {item.text for item in read_examples(_synthetic_dir(workdir) / "examples.jsonl")}

    StubSynthHandler.marker_replies = {"eczema": REPLY_A_ALT, "migraine": REPLY_B_ALT}
    assert cli.main(["synthesize", "--force"]) == 0
    assert StubSynthHandler.request_count == 40  # no reply served from the cache
    manifest = json.loads((_synthetic_dir(workdir) / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["resumed"] == 0 and manifest["counters"]["attempts"] == 20
    second_texts = {item.text for item in read_examples(_synthetic_dir(workdir) / "examples.jsonl")}
    assert second_texts and not (first_texts & second_texts)  # genuinely regenerated


def test_synthesize_warm_cache_rerun_is_byte_identical(server, tmp_path, monkeypatch):
    """Deleting the outputs but keeping the cache reproduces both files byte for byte.

    Raw replies are cached and the gates are pure, so a rerun must be a replay, not a fresh
    roll of the sampler; any byte difference would make the synthetic pool unreproducible.
    """
    workdir = _stage(monkeypatch, server, tmp_path / "work", WORKERS=1)
    _write_train(workdir, _gold())
    assert cli.main(["synthesize"]) == 0
    examples_path = _synthetic_dir(workdir) / "examples.jsonl"
    manifest_path = _synthetic_dir(workdir) / "manifest.json"
    first = (examples_path.read_bytes(), manifest_path.read_bytes())
    assert StubSynthHandler.request_count == 20

    shutil.rmtree(_synthetic_dir(workdir))  # keep the cache, lose the outputs
    assert cli.main(["synthesize"]) == 0
    assert StubSynthHandler.request_count == 20  # entirely served from the warm cache
    assert (examples_path.read_bytes(), manifest_path.read_bytes()) == first


def test_synthesize_limit_trial_still_enforces_the_floor_gate(server, tmp_path, monkeypatch, capsys):
    """--limit caps attempted slots but never silently disables the floor gate.

    A trial below the floor exits non-zero after writing the manifest, exactly like a full run:
    otherwise a limited run could masquerade as success with a tiny pool. The manifest records
    ``trial``/``limit`` so tooling can still tell trials from full runs.
    """
    workdir = _stage(monkeypatch, server, tmp_path / "work", MAX_ATTEMPTS=1, WORKERS=1)
    StubSynthHandler.marker_replies = {}
    _write_train(workdir, _gold())
    assert cli.main(["synthesize", "--limit", "3"]) == 1
    error = capsys.readouterr().err
    assert "floor requires" in error and "MEDLINER_SYNTH_MIN_RATIO" in error
    manifest = json.loads((_synthetic_dir(workdir) / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["trial"] is True and manifest["limit"] == 3
    assert manifest["gate"] == {"passed": False, "enforced": True, "required": 10.0}
    assert manifest["counters"]["attempts"] == 3
    assert (_synthetic_dir(workdir) / "examples.jsonl").exists()


def test_synthesize_rejects_non_positive_limits(server, tmp_path, monkeypatch, capsys):
    """--limit 0/negative is a usage error named before any server contact."""
    workdir = _stage(monkeypatch, server, tmp_path / "work")
    _write_train(workdir, _gold())
    for raw in ("0", "-3"):
        assert cli.main(["synthesize", "--limit", raw]) == 1, raw
        assert "--limit must be a positive" in capsys.readouterr().err
        assert StubSynthHandler.request_count == 0


def test_synthesize_env_vars_parse_with_clear_errors(tmp_path, monkeypatch, capsys):
    """Every MEDLINER_SYNTH_* misconfiguration names its variable — before any server contact.

    A generic 'invalid literal' would send operators debugging the LLM instead of their env;
    parse errors must also precede the health check so the true cause is the only message.
    """
    workdir = _stage(monkeypatch, "http://127.0.0.1:1", tmp_path / "work")  # dead on purpose
    _write_train(workdir, _gold())
    cases = [
        ("MEDLINER_SYNTH_RATIO", "many", "MEDLINER_SYNTH_RATIO must be an integer"),
        ("MEDLINER_SYNTH_RATIO", "0", "at least 1"),
        ("MEDLINER_SYNTH_MIN_RATIO", "eight", "must be a number"),
        ("MEDLINER_SYNTH_MAX_ATTEMPTS", "0", "at least 1"),
        ("MEDLINER_SYNTH_MAX_WORDS", "0", "at least 1"),
        ("MEDLINER_SYNTH_MIN_SIMILARITY", "1.5", "within [0.0, 1.0]"),
        ("MEDLINER_SYNTH_WORKERS", "lots", "must be an integer"),
    ]
    for name, raw, expected in cases:
        monkeypatch.setenv(name, raw)
        assert cli.main(["synthesize"]) == 1, (name, raw)
        error = capsys.readouterr().err
        assert expected in error, (name, raw, error)
        assert "make llm" not in error  # the configuration error, not the dead-server error
        monkeypatch.delenv(name)
    # The floor may not exceed the target: that configuration could never pass.
    monkeypatch.setenv("MEDLINER_SYNTH_MIN_RATIO", "11")
    assert cli.main(["synthesize"]) == 1
    assert "must not exceed" in capsys.readouterr().err


def test_synthesize_flags_override_the_environment(server, tmp_path, monkeypatch):
    """Flags beat MEDLINER_SYNTH_* so one-off runs need no env surgery."""
    workdir = _stage(monkeypatch, server, tmp_path / "work", RATIO=10, WORKERS=1)
    _write_train(workdir, _gold())
    assert cli.main(["synthesize", "--ratio", "1", "--min-ratio", "1", "--min-similarity", "0.2"]) == 0
    manifest = json.loads((_synthetic_dir(workdir) / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["ratio"] == 1 and manifest["min_ratio"] == 1.0
    assert manifest["similarity_floor"] == 0.2  # canonical setting, legacy manifest field name
    assert len(manifest["slots"]) == 2


def test_synthesize_similarity_floor_alias_matches_min_similarity(server, tmp_path, monkeypatch):
    """--similarity-floor stays accepted as a backwards-compatible alias of --min-similarity."""
    workdir = _stage(monkeypatch, server, tmp_path / "work", RATIO=1, MIN_RATIO=1, WORKERS=1)
    _write_train(workdir, _gold())
    assert cli.main(["synthesize", "--similarity-floor", "0.1"]) == 0
    manifest = json.loads((_synthetic_dir(workdir) / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["similarity_floor"] == 0.1


def test_synthesize_retries_rejected_slots_up_to_max_attempts(server, tmp_path, monkeypatch):
    """A rejected slot is retried under a fresh variant name, and the retry is auditable.

    Retries must change the cache key (the variant name) or a warm cache would replay the
    rejected reply forever; the slot record and example id therefore carry the -rN suffix.
    """
    workdir = _stage(monkeypatch, server, tmp_path / "work", RATIO=1, MIN_RATIO=1, MAX_ATTEMPTS=2, WORKERS=1)
    _write_train(workdir, _gold())
    StubSynthHandler.script = [UNRELATED, REPLY_A, UNRELATED, REPLY_B]
    assert cli.main(["synthesize"]) == 0
    manifest = json.loads((_synthetic_dir(workdir) / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["counters"]["attempts"] == 4
    assert manifest["counters"]["accepted_this_run"] == 2
    assert manifest["counters"]["rejections"]["missing_mention"] == 2
    assert [record["variant"] for record in manifest["slots"]] == ["paraphrase-r2", "paraphrase-r2"]
    assert all(record["attempts"] == 2 for record in manifest["slots"])
    examples = read_examples(_synthetic_dir(workdir) / "examples.jsonl")
    assert [item.id for item in examples] == ["gold-a-synth-paraphrase-r2", "gold-b-synth-paraphrase-r2"]


def test_synthesize_max_attempts_exhaustion_falls_through_to_the_loud_floor(server, tmp_path, monkeypatch, capsys):
    """When retries are exhausted the shortfall must reach the floor gate, not exit quietly.

    Capping attempts without the floor check would let a pathological server produce a
    silently tiny pool with exit code 0 — the exact failure mode the loud floor exists for.
    """
    workdir = _stage(monkeypatch, server, tmp_path / "work", RATIO=1, MIN_RATIO=1, MAX_ATTEMPTS=1, WORKERS=1)
    StubSynthHandler.marker_replies = {}
    _write_train(workdir, _gold())
    assert cli.main(["synthesize"]) == 1
    assert "floor requires" in capsys.readouterr().err
    manifest = json.loads((_synthetic_dir(workdir) / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["counters"]["attempts"] == 2 and manifest["accepted"] == 0
