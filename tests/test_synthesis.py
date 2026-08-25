"""Tests for the synthesis engine, plus its schema contracts.

The first two tests pin the provenance contract the engine must satisfy (synthetic data
validates as synthetic and can never masquerade as human work); the rest exercise the engine
itself: span mapping, the ordered divergence gates, the sqlite reply cache, and the prompt.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import get_args

import pytest
from blake3 import blake3
from pydantic import ValidationError

from medliner import synthesis
from medliner.gliner_data import ModelLimits
from medliner.schema import PROVENANCE_VALUES, Annotation, Example, Provenance, SourceMetadata


def _synthetic_example(provenance: str) -> Example:
    return Example(
        id="synth-1",
        text="asthma",
        task="indication",
        source={"family": "synthetic"},
        annotations=[Annotation(start=0, end=6, label="disease", text="asthma", provenance=provenance)],
    )


def test_synthetic_provenance_is_accepted():
    """A machine-synthesized example must validate end-to-end carrying provenance='synthetic'.

    The synthesis pipeline cannot emit data the canonical contract rejects, so acceptance is
    pinned before any generator exists.
    """
    example = _synthetic_example("synthetic")
    assert example.annotations[0].provenance == "synthetic"
    # PROVENANCE_VALUES must stay derived from the literal, never a hand-maintained tuple.
    assert "synthetic" in PROVENANCE_VALUES
    assert PROVENANCE_VALUES == get_args(Provenance)


@pytest.mark.parametrize("claimed", ["human", "adjudicated"])
def test_synthetic_annotation_cannot_claim_human_provenance(claimed):
    """A synthetic example claiming human (or adjudicated) provenance must be rejected.

    Adjudicated is a human claim too — an adjudicator resolved the span. Letting synthetic data
    claim either value would silently contaminate every audit that trusts provenance: review
    effort accounting, dataset manifests, and downstream trust policies.
    """
    with pytest.raises(ValidationError, match="claims human provenance"):
        _synthetic_example(claimed)


# -----------------------------------------------------------------------------------------------
# Engine tests
# -----------------------------------------------------------------------------------------------


def _annotated(text: str, mention_labels: list[tuple[str, str]], **kwargs) -> Example:
    """Build a valid reviewed Example by locating mentions sequentially (no hand-counted spans)."""
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
        source=SourceMetadata(family="dailymed", document_id="doc-1"),
        annotations=annotations,
        **kwargs,
    )


def _reviewed() -> Example:
    return _annotated(
        "Approved for asthma and chronic eczema.",
        [("asthma", "disease"), ("chronic eczema", "phenotype")],
        id="src-1",
        task="indication",
    )


_ACCEPTED_REPLY = "Indicated for asthma and chronic eczema in adults."


class StubSynthHandler(BaseHTTPRequestHandler):
    """Minimal /v1/chat/completions stub: returns a canned reply and records each request."""

    reply_content = _ACCEPTED_REPLY
    last_body: dict = {}
    request_count = 0

    def log_message(self, *_args):  # keep the test output quiet
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        type(self).last_body = json.loads(self.rfile.read(length).decode())
        type(self).request_count += 1
        message = {"role": "assistant", "content": self.reply_content}
        self._respond({"choices": [{"message": message}]})

    def _respond(self, payload: dict, *, status: int = 200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def server():
    StubSynthHandler.reply_content = _ACCEPTED_REPLY
    StubSynthHandler.last_body = {}
    StubSynthHandler.request_count = 0
    httpd = HTTPServer(("127.0.0.1", 0), StubSynthHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()
    thread.join()


# --- span mapping -------------------------------------------------------------------------------


def test_map_mentions_returns_half_open_spans_in_document_order():
    """Sequential scan maps repeated mentions to distinct occurrences, spans slice to the text.

    Half-open [start, end) spans that satisfy text[start:end] == mention are exactly what the
    canonical Annotation contract stores, so mapping must produce nothing else.
    """
    text = "asthma therapy for asthma patients."
    spans = synthesis.map_mentions(text, ["asthma", "asthma"])
    assert spans == [(0, 6), (19, 25)]
    assert all(text[start:end] == "asthma" for start, end in spans)
    assert spans == sorted(spans) and spans[0][1] <= spans[1][0]  # ordered, non-overlapping


def test_map_mentions_rejects_a_missing_mention_with_none():
    """A mention absent from the rewrite maps to None — the engine never guesses a span.

    Guessing (fuzzy match, nearest word, learned offsets) would fabricate supervision: the
    synthetic annotation would point at text nobody verified. None rejects the whole rewrite.
    """
    assert synthesis.map_mentions("treats asthma only.", ["asthma", "eczema"]) is None
    assert synthesis.map_mentions("any text", [""]) is None  # an empty mention can never match


def test_map_mentions_rejects_mentions_that_only_appear_out_of_order():
    """A rewrite that swaps mention order fails the sequential scan and returns None.

    Each search starts after the previous match, so out-of-order (or nested/overlapping)
    occurrences cannot be mapped; this keeps annotation order stable relative to the source.
    """
    assert synthesis.map_mentions("eczema then asthma", ["asthma", "eczema"]) is None


# --- gate evaluation ----------------------------------------------------------------------------


def test_evaluate_reply_accepts_a_clean_paraphrase():
    """A passing reply becomes a synthetic Example with correct spans and full provenance stamps.

    Pins the acceptance shape end to end: mentions map to their true positions in the rewrite,
    the entity multiset is exactly the source's, and family/provenance mark it machine-made.
    """
    result = synthesis.evaluate_reply(_reviewed(), _ACCEPTED_REPLY, variant="paraphrase")
    assert result.accepted and result.reason is None
    example = result.example
    assert example is not None
    assert example.text == _ACCEPTED_REPLY
    assert example.id == "src-1-synth-paraphrase"
    assert example.task == "indication"
    assert example.source.family == "synthetic"
    assert example.source.document_id == "doc-1"  # original traceability survives the copy
    assert example.source.synth_source_id == "src-1" and example.source.synth_variant == "paraphrase"
    asthma, eczema = example.annotations
    assert (asthma.start, asthma.end, asthma.text, asthma.label) == (14, 20, "asthma", "disease")
    assert (eczema.start, eczema.end, eczema.text, eczema.label) == (25, 39, "chronic eczema", "phenotype")
    assert all(a.provenance == "synthetic" for a in example.annotations)
    assert Counter((a.text, a.label) for a in example.annotations) == Counter(
        (a.text, a.label) for a in _reviewed().annotations
    )


def test_duplicate_gate_rejects_a_token_identical_rewrite():
    """An unchanged rewrite is rejected as a duplicate even though similarity/ratio would pass.

    The rewrite passes every later gate (Jaccard 1.0, ratio 1.0), so reason 'duplicate' proves
    gate 1 fires first — and that a no-op paraphrase can never enter the synthetic pool.
    Whitespace/case-only changes count as duplicates too.
    """
    source = _reviewed()
    result = synthesis.evaluate_reply(source, "Approved   for ASTHMA and chronic eczema .")
    assert not result.accepted and result.reason == "duplicate"


def test_missing_mention_gate_rejects_rewrites_that_drop_a_mention():
    """Dropping a mention rejects with 'missing_mention' and names the missing mention.

    A rewrite that loses an entity silently loses supervision for it; the detail must say
    which mention so operators can judge whether the prompt or the gate needs attention.
    """
    result = synthesis.evaluate_reply(_reviewed(), "Indicated for asthma.")
    assert not result.accepted and result.reason == "missing_mention"
    assert "chronic eczema" in (result.detail or "")


def test_missing_mention_gate_rejects_rewrites_that_replace_a_mention():
    """Replacing a mention with a synonym rejects with 'missing_mention', never a guessed span.

    'reactive airway disease' means the same as 'asthma' but is not the verbatim mention; the
    engine must not map it, or the synthetic span would carry text the source never asserted.
    (A superstring like 'asthmatics' does contain the mention and falls through to the budget
    gate instead — pinned by its own test below.)
    """
    result = synthesis.evaluate_reply(_reviewed(), "Indicated for reactive airway disease and chronic eczema.")
    assert not result.accepted and result.reason == "missing_mention"


def test_gate_order_missing_mention_before_similarity():
    """A reply with neither mentions nor shared content words reports 'missing_mention'.

    The reply fails gates 2 and 3 simultaneously; only gate order decides the reason, so this
    pins that span mapping is checked before similarity.
    """
    result = synthesis.evaluate_reply(_reviewed(), "zebra quill walnut mango")
    assert not result.accepted and result.reason == "missing_mention"


def test_similarity_gate_rejects_low_content_word_overlap():
    """Mentions preserved but vocabulary mostly new rejects with 'low_similarity'.

    The rewrite still contains both mentions and stays inside the length bounds, so the only
    failing gate is the Jaccard floor: the rewrite drifted too far to be a faithful variant.
    """
    reply = "asthma and chronic eczema zebra quill walnut mango peach olive basil"
    result = synthesis.evaluate_reply(_reviewed(), reply)
    assert not result.accepted and result.reason == "low_similarity"
    assert "Jaccard" in (result.detail or "")


def test_gate_order_similarity_before_length():
    """A reply failing both similarity and length reports 'low_similarity'.

    Pins gate 3 before gate 4 so rejection reasons stay predictable for operators reading logs.
    """
    reply = "asthma and chronic eczema zebra quill walnut mango peach olive basil cedar spruce pine birch"
    result = synthesis.evaluate_reply(_reviewed(), reply)
    assert not result.accepted and result.reason == "low_similarity"


def test_length_ratio_gate_rejects_rewrites_outside_bounds():
    """Both directions of unbounded drift — far too long and far too short — reject.

    Each reply keeps similarity above the floor, so the ratio bound is the only failing gate;
    without it a rewrite could balloon or collapse while still sharing its content words.
    """
    source = _annotated("asthma eczema flare therapy", [("asthma", "disease"), ("eczema", "phenotype")], id="src-2")
    too_long = synthesis.evaluate_reply(source, "asthma eczema flare therapy zebra quill walnut mango peach olive")
    assert not too_long.accepted and too_long.reason == "length_ratio"
    source = _annotated(
        "asthma eczema flare therapy tonight", [("asthma", "disease"), ("eczema", "phenotype")], id="src-3"
    )
    too_short = synthesis.evaluate_reply(source, "asthma eczema")
    assert not too_short.accepted and too_short.reason == "length_ratio"


def test_gate_order_length_before_budget():
    """A reply failing both the ratio and the token budget reports 'length_ratio'.

    Pins gate 4 before gate 5 so the cheaper textual check wins and logs stay predictable.
    """
    source = _annotated("asthma eczema", [("asthma", "disease"), ("eczema", "phenotype")], id="src-4")
    result = synthesis.evaluate_reply(
        source, "asthma eczema zebra quill walnut mango", limits=ModelLimits(max_len=5, max_width=12)
    )
    assert not result.accepted and result.reason == "length_ratio"


def test_budget_gate_rejects_rewrites_over_the_word_token_limit():
    """A rewrite longer than max_len rejects with 'budget_exceeded' instead of training truncated.

    GLiNER truncates over-long token sequences with only a UserWarning; refusing the rewrite is
    the only way supervision cannot be silently amputated. Both the configured limit and the
    real 384-token default are pinned.
    """
    source = _annotated("asthma eczema flare therapy", [("asthma", "disease"), ("eczema", "phenotype")], id="src-5")
    tight = synthesis.evaluate_reply(
        source, "asthma eczema flare therapy zebra", limits=ModelLimits(max_len=4, max_width=12)
    )
    assert not tight.accepted and tight.reason == "budget_exceeded"
    assert "max_len" in (tight.detail or "")

    text = "asthma " + " ".join(f"w{i}" for i in range(200))
    long_source = _annotated(text, [("asthma", "disease")], id="src-6")
    long_reply = text + " " + " ".join(f"junk{i}" for i in range(184))  # 385 word tokens
    default_limit = synthesis.evaluate_reply(long_source, long_reply)  # default limits: 384/12
    assert not default_limit.accepted and default_limit.reason == "budget_exceeded"


def test_budget_gate_rejects_spans_wider_than_max_width():
    """A verbatim mention wider than max_width rejects with 'budget_exceeded'.

    GLiNER never enumerates spans wider than max_width as candidates, so the mention's gold
    label would be unreachable — silently dropped supervision. The gate must refuse it, and
    this also protects against source examples whose own spans already violate the budget.
    """
    source = _annotated("chronic severe asthma care continues", [("chronic severe asthma", "disease")], id="src-7")
    result = synthesis.evaluate_reply(
        source, "chronic severe asthma improved", limits=ModelLimits(max_len=384, max_width=2)
    )
    assert not result.accepted and result.reason == "budget_exceeded"
    assert "max_width" in (result.detail or "")


def test_budget_gate_rejects_mentions_embedded_inside_longer_words():
    """A mention surviving only as characters inside another word rejects at the budget gate.

    'asthmatics' contains 'asthma', so the exact scan finds a span — but it does not align to
    whole word tokens and GLiNER could never supervise it. The rewrite must be rejected, never
    accepted on the strength of a substring.
    """
    source = _annotated("asthma therapy helps patients", [("asthma", "disease")], id="src-8")
    result = synthesis.evaluate_reply(source, "asthmatics therapy helps folks")
    assert not result.accepted and result.reason == "budget_exceeded"


def test_schema_gate_maps_construction_failures_to_schema_violation(monkeypatch):
    """Example-construction failures surface as 'schema_violation', not as exceptions.

    The mapping makes real schema failures unreachable (spans are exact, ordered, and
    non-overlapping by construction), so this pins the defensive wiring: if the contract ever
    refuses a synthetic twin, the engine reports it as data, not as a crash mid-batch.
    """

    def _explode(*_args, **_kwargs):
        raise ValueError("synthetic construction exploded")

    monkeypatch.setattr(synthesis, "_build_synthetic_example", _explode)
    result = synthesis.evaluate_reply(_reviewed(), _ACCEPTED_REPLY)
    assert not result.accepted and result.reason == "schema_violation"
    assert "exploded" in (result.detail or "")


def test_evaluate_reply_rejects_unannotated_sources_loudly():
    """A source with no annotations is an input-contract violation, not a gate rejection.

    Synthesis exists to preserve mentions; with none there is nothing to preserve and the
    result would be an unsupervised paraphrase masquerading as reviewed data. Failing loudly
    beats silently emitting an unannotated synthetic row.
    """
    bare = Example(id="bare", text="nothing to preserve", task="indication")
    with pytest.raises(ValueError, match="no annotations"):
        synthesis.evaluate_reply(bare, "anything")


def test_configuration_is_validated_up_front(server):
    """Nonsensical floors and word budgets raise immediately, before any server call.

    A misconfigured run must fail loudly at the call site rather than burn model requests or
    produce rejections that look like model failures.
    """
    source = _reviewed()
    with pytest.raises(ValueError, match="similarity_floor"):
        synthesis.synthesize_variant(source, similarity_floor=1.5, url=server)
    with pytest.raises(ValueError, match="max_words"):
        synthesis.synthesize_variant(source, max_words=0, url=server)
    assert StubSynthHandler.request_count == 0


# --- full path through the LLM client ------------------------------------------------------------


def test_synthesize_variant_sends_a_verbatim_preserving_prompt(server):
    """The engine reuses the llama-server client and its prompt demands verbatim mentions.

    The whole data-integrity story rests on the model keeping mentions character-identical
    and returning only the rewrite; the prompt is the only place that contract is stated, so
    it is pinned here. Reasoning stays disabled (the reasoner server's requirement).
    """
    result = synthesis.synthesize_variant(_reviewed(), url=server, max_words=48)
    assert result.accepted and result.example is not None
    assert result.example.text == _ACCEPTED_REPLY
    assert StubSynthHandler.request_count == 1
    prompt = StubSynthHandler.last_body["messages"][0]["content"]
    assert "VERBATIM" in prompt
    assert "Return only the rewritten text" in prompt
    assert "at most 48 words" in prompt
    assert "style: paraphrase" in prompt
    assert "- asthma\n- chronic eczema" in prompt
    assert "Approved for asthma and chronic eczema." in prompt
    assert StubSynthHandler.last_body["chat_template_kwargs"] == {"enable_thinking": False}


def test_synthesize_variant_maps_server_failures_to_llm_error():
    """An unreachable server is reported as 'llm_error', never raised and never a fake accept.

    Transport failures have no reply to gate; they must come back as data so batches can count
    them and callers can retry or surface the outage.
    """
    result = synthesis.synthesize_variant(_reviewed(), url="http://127.0.0.1:1")
    assert not result.accepted and result.reason == "llm_error"
    assert result.example is None and result.detail


# --- sqlite reply cache --------------------------------------------------------------------------


def test_cache_key_pins_namespace_source_variant_and_budget():
    """The cache key is the hash of an exact preimage containing every documented field.

    Pinning the preimage (namespace tag, source id, variant, max_words, text) guarantees a
    prompt or keying change cannot silently reuse stale replies across versions.
    """
    text = "Approved for asthma and chronic eczema."
    preimage = "medliner-synth-v1\nsrc-1\nparaphrase\n48\n" + text
    assert (
        synthesis.cache_key(text, source_id="src-1", variant="paraphrase", max_words=48)
        == blake3(preimage.encode()).hexdigest()
    )
    # Every key field changes the key: no accidental cross-variant or cross-budget reuse.
    base = synthesis.cache_key(text, source_id="src-1", variant="paraphrase", max_words=48)
    assert synthesis.cache_key(text, source_id="src-2", variant="paraphrase", max_words=48) != base
    assert synthesis.cache_key(text, source_id="src-1", variant="formal", max_words=48) != base
    assert synthesis.cache_key(text, source_id="src-1", variant="paraphrase", max_words=64) != base
    assert synthesis.cache_key(text + "!", source_id="src-1", variant="paraphrase", max_words=48) != base


def test_synthesis_cache_prevents_a_second_server_hit(server, tmp_path):
    """A second identical synthesis is served from sqlite; the server is never re-consulted.

    Re-runs and retries are the norm in this pipeline; the cache keeps them from burning model
    time, and — because raw replies are cached and re-gated — the second result is identical.
    A different variant or word budget is a different key and goes back to the server.
    """
    cache = tmp_path / "cache" / "synth.sqlite3"
    source = _reviewed()
    first = synthesis.synthesize_variant(source, url=server, cache=cache)
    assert first.accepted and StubSynthHandler.request_count == 1

    # Poison the server: a second hit would produce a different (rejected) reply.
    StubSynthHandler.reply_content = "Completely different server answer."
    second = synthesis.synthesize_variant(source, url=server, cache=cache)
    assert second.accepted and second.example is not None
    assert second.example.text == _ACCEPTED_REPLY  # served from the cache, not the server
    assert StubSynthHandler.request_count == 1

    with sqlite3.connect(cache) as connection:
        rows = connection.execute("SELECT key, reply FROM synthetic_replies").fetchall()
    assert len(rows) == 1
    assert rows[0][1] == _ACCEPTED_REPLY
    assert rows[0][0] == synthesis.cache_key(source.text, source_id="src-1", variant="paraphrase", max_words=48)

    # Variant and budget are key fields: each forces a fresh server round-trip.
    synthesis.synthesize_variant(source, url=server, cache=cache, variant="formal")
    assert StubSynthHandler.request_count == 2
    synthesis.synthesize_variant(source, url=server, cache=cache, max_words=64)
    assert StubSynthHandler.request_count == 3
    with sqlite3.connect(cache) as connection:
        assert connection.execute("SELECT COUNT(*) FROM synthetic_replies").fetchone()[0] == 3


def test_synthesis_tolerates_a_broken_cache(server, tmp_path):
    """An unopenable cache degrades to a miss: synthesis still works, just uncached.

    A corrupted cache file must never fail a data-generation run; the cost is a redundant
    server hit, which the request counter proves happens (store failed, so nothing is served).
    """
    broken = tmp_path / "broken"  # a directory: sqlite cannot open it as a database file
    broken.mkdir()
    source = _reviewed()
    first = synthesis.synthesize_variant(source, url=server, cache=broken)
    assert first.accepted and StubSynthHandler.request_count == 1
    second = synthesis.synthesize_variant(source, url=server, cache=broken)
    assert second.accepted and StubSynthHandler.request_count == 2  # nothing could be stored


# --- counters -----------------------------------------------------------------------------------


def test_counters_partition_attempts_exactly(monkeypatch):
    """Every attempt lands in exactly one bucket: attempts == accepted + all rejections.

    Each rejection reason is produced once, so the partition is checked against real hits, not
    zeros; an unknown reason is refused loudly so counters can never hide a mis-wired gate.
    """
    reviewed = _reviewed()
    four_words = _annotated("asthma eczema flare therapy", [("asthma", "disease"), ("eczema", "phenotype")], id="src-2")

    def _explode(*_args, **_kwargs):
        raise ValueError("synthetic construction exploded")

    results = [
        synthesis.evaluate_reply(reviewed, _ACCEPTED_REPLY),  # accepted
        synthesis.evaluate_reply(reviewed, reviewed.text),  # duplicate
        synthesis.evaluate_reply(reviewed, "Indicated for asthma."),  # missing_mention
        synthesis.evaluate_reply(
            reviewed, "asthma and chronic eczema zebra quill walnut mango peach olive basil"
        ),  # low_similarity
        synthesis.evaluate_reply(
            four_words, "asthma eczema flare therapy zebra quill walnut mango peach olive"
        ),  # length_ratio
        synthesis.evaluate_reply(
            four_words, "asthma eczema flare therapy zebra", limits=ModelLimits(max_len=4, max_width=12)
        ),  # budget_exceeded
    ]
    # The schema gate is defensive (mapping makes failures unreachable), so it is exercised by
    # patching the builder for exactly this one call.
    monkeypatch.setattr(synthesis, "_build_synthetic_example", _explode)
    results.append(synthesis.evaluate_reply(reviewed, _ACCEPTED_REPLY))  # schema_violation
    monkeypatch.undo()
    results.append(synthesis.synthesize_variant(four_words, url="http://127.0.0.1:1"))  # llm_error
    assert [r.reason for r in results[1:]] == [
        "duplicate",
        "missing_mention",
        "low_similarity",
        "length_ratio",
        "budget_exceeded",
        "schema_violation",
        "llm_error",
    ]

    counters = synthesis.SynthesisCounters()
    for result in results:
        counters.record(result)
    assert counters.attempts == 8
    assert counters.accepted == 1
    assert counters.total_rejections() == 7
    assert counters.partition_exactly()
    assert set(counters.rejections) == set(synthesis.REJECTION_REASONS)
    assert all(count == 1 for count in counters.rejections.values())

    with pytest.raises(ValueError, match="unknown rejection reason"):
        synthesis.SynthesisResult.reject("nope")


def test_synthesize_variants_counts_and_collects_accepted_examples(server):
    """The batch helper returns only accepted synthetic twins plus exact accounting.

    One source's mentions survive the canned reply, the other's do not: the batch must keep
    the first, count the second as 'missing_mention', and partition both attempts exactly.
    """
    good = _reviewed()
    bad = _annotated("treats migraine", [("migraine", "disease")], id="src-9")
    accepted, counters = synthesis.synthesize_variants([good, bad], url=server)
    assert [example.id for example in accepted] == ["src-1-synth-paraphrase"]
    assert accepted[0].source.family == "synthetic"
    assert counters.attempts == 2
    assert counters.accepted == 1
    assert counters.rejections["missing_mention"] == 1
    assert counters.partition_exactly()
