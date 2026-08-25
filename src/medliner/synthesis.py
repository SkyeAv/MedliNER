"""Semi-supervised synthesis core: paraphrase a reviewed example into a synthetic twin.

The engine asks the local llama-server (through :mod:`medliner.llm` — no new client) to
rewrite an example while keeping every entity mention verbatim, then rebuilds annotations by
mapping each mention onto the rewrite with a sequential exact-match scan, and finally runs
divergence gates that decide whether the rewrite may become training data. Nothing is ever
guessed: a mention that cannot be located exactly rejects the whole rewrite.

Every accepted rewrite becomes a canonical :class:`~medliner.schema.Example` stamped
``source.family='synthetic'`` and ``provenance='synthetic'``; the schema itself refuses
synthetic data claiming human provenance, so machine-made rows stay auditable forever.

Gate order (first failure wins; the reason is stable and machine-readable):

1. ``duplicate``       — the rewrite is token-identical to the source text.
2. ``missing_mention`` — some mention is absent, altered, or unmappable in order.
3. ``low_similarity``  — content-word Jaccard similarity below the configured floor.
4. ``length_ratio``    — rewrite/source word-token ratio outside ``[0.5, 2.0]``.
5. ``schema_violation`` / ``budget_exceeded`` — the synthetic example fails schema
   validation, or cannot become a GLiNER record within the model budgets
   (default 384 word tokens, 12-token spans).

``llm_error`` covers transport failures (unreachable server, empty completion); it is
reported instead of a gate reason because there is no reply to evaluate. Counters partition
attempts exactly: ``attempts == accepted + sum(rejections)``.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from . import llm
from .gliner_data import GLINER_TOKEN, ModelLimits, to_gliner_record
from .schema import SYNTHETIC_SOURCE_FAMILY, Annotation, Example

#: Version tag for the sqlite reply cache; bump when the prompt or key fields change.
CACHE_NAMESPACE = "medliner-synth-v1"
DEFAULT_VARIANT = "paraphrase"
DEFAULT_MAX_WORDS = 48
DEFAULT_SIMILARITY_FLOOR = 0.3
LENGTH_RATIO_MIN = 0.5
LENGTH_RATIO_MAX = 2.0
#: The GLiNER budgets an accepted rewrite must fit; callers with a real model pass
#: ``model_limits(model)`` instead so the engine uses the model's own configuration.
DEFAULT_LIMITS = ModelLimits(max_len=384, max_width=12)

#: Stable, machine-readable rejection reasons. ``llm_error`` is transport, not a gate.
REJECTION_REASONS = (
    "duplicate",
    "missing_mention",
    "low_similarity",
    "length_ratio",
    "schema_violation",
    "budget_exceeded",
    "llm_error",
)

SYNTH_PROMPT = """\
Paraphrase the following medical {task} text (style: {variant}). Keep every entity mention \
listed below VERBATIM: exactly the same characters, unedited, unsplit, untranslated, and in \
the same order. Keep the meaning, change the surrounding wording, and use at most {max_words} \
words. Return only the rewritten text, with no commentary, quotes, or explanation.

Entity mentions that must appear unchanged:
{mentions}

Text:
{text}"""

#: English function words excluded from the similarity vocabulary so the Jaccard gate
#: measures overlap of meaningful (content) words, not shared grammar.
STOPWORDS = frozenset(
    """
    a an the and or but if then than so because while until of at by for with about against
    between into through during before after above below to from up down in out on off over
    under again further once here there when where why how all any both each few more most
    other some such no nor not only own same too very just also can will should would could
    may might must shall is are was were be been being am do does did doing have has had
    having i you he she it we they me him her us them my your his its our their who whom
    this that these those what which
    """.split()
)


@dataclass(frozen=True)
class SynthesisResult:
    """Outcome of one synthesis attempt; ``reason`` is None iff accepted."""

    accepted: bool
    example: Example | None = None
    reason: str | None = None
    detail: str | None = None

    @classmethod
    def accept(cls, example: Example) -> SynthesisResult:
        return cls(accepted=True, example=example)

    @classmethod
    def reject(cls, reason: str, *, detail: str | None = None) -> SynthesisResult:
        if reason not in REJECTION_REASONS:
            raise ValueError(f"unknown rejection reason {reason!r}; expected one of {REJECTION_REASONS}")
        return cls(accepted=False, reason=reason, detail=detail)


@dataclass
class SynthesisCounters:
    """Attempt accounting that must partition exactly: attempts = accepted + rejections."""

    attempts: int = 0
    accepted: int = 0
    rejections: dict[str, int] = field(default_factory=lambda: dict.fromkeys(REJECTION_REASONS, 0))

    def record(self, result: SynthesisResult) -> None:
        self.attempts += 1
        if result.accepted:
            self.accepted += 1
            return
        if result.reason not in self.rejections:
            raise ValueError(f"unknown rejection reason {result.reason!r}; expected one of {REJECTION_REASONS}")
        self.rejections[result.reason] += 1

    def total_rejections(self) -> int:
        return sum(self.rejections.values())

    def partition_exactly(self) -> bool:
        return self.attempts == self.accepted + self.total_rejections()


def default_cache_path() -> Path:
    """Cache location ($MEDLINER_SYNTH_CACHE, default ``<workdir>/synth-cache.sqlite3``)."""
    from .cli import workdir  # local import: cli pulls heavy deps

    return Path(os.environ.get("MEDLINER_SYNTH_CACHE", str(workdir() / "synth-cache.sqlite3")))


def cache_key(text: str, *, source_id: str, variant: str, max_words: int) -> str:
    """Content key for a synthetic rewrite; version-tagged so prompt changes invalidate entries."""
    from blake3 import blake3

    return blake3(f"{CACHE_NAMESPACE}\n{source_id}\n{variant}\n{max_words}\n{text}".encode()).hexdigest()


def cache_lookup(cache: str | Path, text: str, *, source_id: str, variant: str, max_words: int) -> str | None:
    """Cached raw model reply, or None on a miss (or any cache problem).

    A broken/unreadable cache degrades to a cache miss, never to a failed run.
    """
    try:
        with sqlite3.connect(str(cache)) as connection:
            row = connection.execute(
                "SELECT reply FROM synthetic_replies WHERE key = ?",
                (cache_key(text, source_id=source_id, variant=variant, max_words=max_words),),
            ).fetchone()
    except (sqlite3.Error, OSError):
        return None
    return str(row[0]) if row else None


def cache_store(cache: str | Path, text: str, *, source_id: str, variant: str, max_words: int, reply: str) -> None:
    """Persist a raw reply (it is re-gated deterministically on every read); failures are swallowed."""
    try:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(cache)) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS synthetic_replies ("
                "key TEXT PRIMARY KEY, reply TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT (datetime('now'))"
                ")"
            )
            connection.execute(
                "INSERT OR REPLACE INTO synthetic_replies (key, reply) VALUES (?, ?)",
                (cache_key(text, source_id=source_id, variant=variant, max_words=max_words), reply),
            )
    except (sqlite3.Error, OSError):
        pass


def map_mentions(text: str, mentions: Sequence[str]) -> list[tuple[int, int]] | None:
    """Map each mention onto ``text`` with a sequential exact-match scan.

    Mentions are located left to right: each search starts where the previous match ended, so
    the resulting half-open spans follow document order and cannot overlap. Any mention that
    cannot be found after the previous one — missing, replaced, reordered, or only occurring
    inside an already-consumed span — returns None. Spans are never guessed or fuzzy-matched;
    by construction ``text[start:end] == mention`` for every returned span.
    """
    spans: list[tuple[int, int]] = []
    cursor = 0
    for mention in mentions:
        if not mention:
            return None
        start = text.find(mention, cursor)
        if start < 0:
            return None
        end = start + len(mention)
        spans.append((start, end))
        cursor = end
    return spans


def content_words(text: str) -> set[str]:
    """Lowercased word tokens minus function words — the Jaccard gate's vocabulary."""
    return {token.lower() for token in GLINER_TOKEN.findall(text) if token.lower() not in STOPWORDS}


def jaccard_similarity(left: set[str], right: set[str]) -> float:
    """|A ∩ B| / |A ∪ B|; two empty vocabularies are defined as identical (1.0)."""
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def word_count(text: str) -> int:
    """Number of GLiNER word tokens — the length the ratio gate and max_len budget share."""
    return len(GLINER_TOKEN.findall(text))


def _validate_floor(similarity_floor: float) -> None:
    if not 0.0 <= similarity_floor <= 1.0:
        raise ValueError(f"similarity_floor must be within [0.0, 1.0], got {similarity_floor!r}")


def _token_key(text: str) -> str:
    """Whitespace/case-normalized token sequence, so purely cosmetic rewrites count as duplicates."""
    return " ".join(token.lower() for token in GLINER_TOKEN.findall(text))


def _build_prompt(source: Example, *, variant: str, max_words: int) -> str:
    ordered = sorted(source.annotations, key=lambda annotation: (annotation.start, annotation.end, annotation.label))
    mentions = "\n".join(f"- {annotation.text}" for annotation in ordered)
    return SYNTH_PROMPT.format(
        task=source.task, variant=variant, max_words=max_words, mentions=mentions, text=source.text
    )


def _build_synthetic_example(
    source: Example, *, text: str, spans: Sequence[tuple[int, int]], ordered: Sequence[Annotation], variant: str
) -> Example:
    annotations = [
        Annotation(start=start, end=end, label=annotation.label, text=annotation.text, provenance="synthetic")
        for annotation, (start, end) in zip(ordered, spans, strict=True)
    ]
    synth_source = source.source.model_copy(
        update={"family": SYNTHETIC_SOURCE_FAMILY, "synth_source_id": source.id, "synth_variant": variant}
    )
    return Example(
        id=f"{source.id}-synth-{variant}",
        text=text,
        task=source.task,
        source=synth_source,
        annotations=annotations,
        metadata={**source.metadata, "synthetic_variant": variant},
    )


def evaluate_reply(
    source: Example,
    reply: str,
    *,
    variant: str = DEFAULT_VARIANT,
    similarity_floor: float = DEFAULT_SIMILARITY_FLOOR,
    limits: ModelLimits | None = None,
) -> SynthesisResult:
    """Run the ordered divergence gates on one model reply; never raises on gate failures.

    Pure and deterministic: given the same source, reply, and configuration the result is
    always the same, so a cached raw reply re-evaluates identically on every run.
    """
    _validate_floor(similarity_floor)
    if not source.annotations:
        raise ValueError(
            f"example {source.id!r} has no annotations; synthesis preserves mentions, so there is nothing to preserve"
        )
    text = reply.strip()
    ordered = sorted(source.annotations, key=lambda annotation: (annotation.start, annotation.end, annotation.label))

    # Gate 1: a token-identical rewrite adds no divergence and would just duplicate the source.
    if _token_key(text) == _token_key(source.text):
        return SynthesisResult.reject("duplicate", detail="rewrite is token-identical to the source text")

    # Gate 2: every mention must survive verbatim, in order, at a locatable position.
    spans = map_mentions(text, [annotation.text for annotation in ordered])
    if spans is None:
        absent = [annotation.text for annotation in ordered if annotation.text not in text]
        detail = f"mentions not preserved verbatim/in order: {absent or [a.text for a in ordered]}"
        return SynthesisResult.reject("missing_mention", detail=detail)

    # Gate 3: enough shared content words that the rewrite is still about the same thing.
    similarity = jaccard_similarity(content_words(source.text), content_words(text))
    if similarity < similarity_floor:
        return SynthesisResult.reject(
            "low_similarity", detail=f"content-word Jaccard {similarity:.3f} < floor {similarity_floor:.3f}"
        )

    # Gate 4: bounded divergence in length.
    source_tokens, reply_tokens = word_count(source.text), word_count(text)
    ratio = reply_tokens / source_tokens if source_tokens else float("inf")
    if not LENGTH_RATIO_MIN <= ratio <= LENGTH_RATIO_MAX:
        return SynthesisResult.reject(
            "length_ratio",
            detail=f"{reply_tokens}/{source_tokens} word tokens = ratio {ratio:.2f} "
            f"outside [{LENGTH_RATIO_MIN}, {LENGTH_RATIO_MAX}]",
        )

    # Gate 5a: the synthetic twin must satisfy the canonical contract itself.
    try:
        synthetic = _build_synthetic_example(source, text=text, spans=spans, ordered=ordered, variant=variant)
    except (ValidationError, ValueError) as exc:  # ValidationError subclasses ValueError
        return SynthesisResult.reject("schema_violation", detail=f"synthetic example failed schema validation: {exc}")

    # Gate 5b: it must fit the GLiNER budgets — buildable as a training record, not silently
    # truncated (max_len) or dropped (max_width), and mentions must sit on whole word tokens.
    try:
        to_gliner_record(synthetic, limits=limits if limits is not None else DEFAULT_LIMITS)
    except ValueError as exc:
        return SynthesisResult.reject("budget_exceeded", detail=str(exc))

    return SynthesisResult.accept(synthetic)


def synthesize_variant(
    source: Example,
    *,
    variant: str = DEFAULT_VARIANT,
    max_words: int = DEFAULT_MAX_WORDS,
    url: str | None = None,
    cache: str | Path | None = None,
    similarity_floor: float = DEFAULT_SIMILARITY_FLOOR,
    limits: ModelLimits | None = None,
    max_tokens: int = 2048,
    timeout: float = 120.0,
) -> SynthesisResult:
    """Paraphrase ``source`` into one synthetic variant through the local llama-server.

    The reply is fetched through :func:`medliner.llm.chat` (or the sqlite cache keyed by
    namespace, source id, variant, and word budget) and then gated by :func:`evaluate_reply`.
    Gate rejections and transport failures come back as results, never exceptions.
    """
    _validate_floor(similarity_floor)
    if max_words < 1:
        raise ValueError(f"max_words must be at least 1, got {max_words!r}")
    reply: str | None = (
        cache_lookup(cache, source.text, source_id=source.id, variant=variant, max_words=max_words) if cache else None
    )
    if reply is None:
        prompt = _build_prompt(source, variant=variant, max_words=max_words)
        try:
            reply = llm.chat([{"role": "user", "content": prompt}], url=url, max_tokens=max_tokens, timeout=timeout)
        except llm.LLMError as exc:
            return SynthesisResult.reject("llm_error", detail=str(exc))
        if cache:
            cache_store(cache, source.text, source_id=source.id, variant=variant, max_words=max_words, reply=reply)
    return evaluate_reply(source, reply, variant=variant, similarity_floor=similarity_floor, limits=limits)


def synthesize_variants(
    sources: Iterable[Example],
    *,
    variant: str = DEFAULT_VARIANT,
    max_words: int = DEFAULT_MAX_WORDS,
    url: str | None = None,
    cache: str | Path | None = None,
    similarity_floor: float = DEFAULT_SIMILARITY_FLOOR,
    limits: ModelLimits | None = None,
    max_tokens: int = 2048,
    timeout: float = 120.0,
) -> tuple[list[Example], SynthesisCounters]:
    """Synthesize every source; returns ``(accepted_examples, counters)``.

    The counters partition attempts exactly, so a run's report can account for every input:
    ``counters.attempts == counters.accepted + counters.total_rejections()``.
    """
    counters = SynthesisCounters()
    accepted: list[Example] = []
    for source in sources:
        result = synthesize_variant(
            source,
            variant=variant,
            max_words=max_words,
            url=url,
            cache=cache,
            similarity_floor=similarity_floor,
            limits=limits,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        counters.record(result)
        if result.accepted and result.example is not None:
            accepted.append(result.example)
    return accepted, counters


__all__ = [
    "CACHE_NAMESPACE",
    "DEFAULT_LIMITS",
    "DEFAULT_MAX_WORDS",
    "DEFAULT_SIMILARITY_FLOOR",
    "DEFAULT_VARIANT",
    "LENGTH_RATIO_MAX",
    "LENGTH_RATIO_MIN",
    "REJECTION_REASONS",
    "STOPWORDS",
    "SYNTH_PROMPT",
    "SynthesisCounters",
    "SynthesisResult",
    "cache_key",
    "cache_lookup",
    "cache_store",
    "content_words",
    "default_cache_path",
    "evaluate_reply",
    "jaccard_similarity",
    "map_mentions",
    "synthesize_variant",
    "synthesize_variants",
    "word_count",
]
