"""GLiNER pre-annotation of Label Studio import tasks.

Annotators are much faster correcting a span than drawing one, so candidate tasks can carry
model suggestions. The suggestions are Label Studio *predictions*, never annotations: a human
accepts, corrects, or deletes every one, and :mod:`medliner.label_studio` still reads only the
completed ``annotations`` array when the export comes back.

The inference contract is ported from the sibling DAKP pipeline
(``dakp_pipeline/ner/ner.py``) so the two projects extract with the same model, the same label
prompts, and the same thresholds. DAKP's span cleanup is ported with it, because that cleanup
is what makes the output obey ``docs/ANNOTATION_GUIDE.md`` — raw GLiNER routinely returns
``recent myocardial infarction`` (guide rule 2), ``patients`` (rule 3), and overlapping spans
(rule 7). DAKP's gazetteer contest is deliberately *not* ported: it needs DAKP's curated
dictionary, which MedliNER does not carry.

Everything above :class:`GLiNERPrelabeler` is pure and importable without torch.
"""

from __future__ import annotations

import itertools
import json
import re
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from blake3 import blake3

from .gliner_data import GLINER_TOKEN
from .schema import ALLOWED_LABELS

#: HuggingFace id of the pre-labeling checkpoint (DAKP ``ner.py`` ``DEFAULT_MODEL``).
DEFAULT_MODEL_ID = "gliner-community/gliner_large-v2.5"
#: Generation threshold passed to GLiNER (DAKP ``ner.py`` ``DEFAULT_THRESHOLD``).
DEFAULT_THRESHOLD = 0.35
#: Window budget in GLiNER word tokens when the model exposes no ``config.max_len``
#: (DAKP ``ner.py`` ``_DEFAULT_WORD_BUDGET``). The shipped large-v2.5 sets ``max_len: 768``.
DEFAULT_WORD_BUDGET = 384
#: GLiNER never enumerates a span candidate wider than ``config.max_width`` (12 for large-v2.5),
#: and :mod:`medliner.gliner_data` refuses to convert a gold span wider than that, so a wider
#: suggestion would break the dataset build the moment a human accepted it.
DEFAULT_MAX_WIDTH = 12
DEFAULT_BATCH_SIZE = 8

#: The label prompts sent to GLiNER. Identical to DAKP's ``CONTRAINDICATION_DISEASE_TYPES``.
PRELABEL_LABELS: tuple[str, ...] = ALLOWED_LABELS

#: ``from_name``/``to_name`` are fixed by ``configs/label_studio_ner.xml``; a prediction whose
#: control names do not match the labeling config is silently ignored by Label Studio.
FROM_NAME = "label"
TO_NAME = "text"

MANIFEST_SCHEMA_VERSION = "medliner.prelabel.manifest.v1"
CACHE_SCHEMA_VERSION = "medliner.prelabel.cache.v1"

# Sentence-ish piece for window packing: a run of non-terminal characters, trailing terminal
# punctuation, trailing whitespace (DAKP ``ner.py`` ``_SENTENCE_PIECE``).
SENTENCE_PIECE = re.compile(r"[^.!?;]+[.!?;]*\s*")
_HTML_TAG = re.compile(r"<[^>]+>")

# Population/demographic descriptors GLiNER likes to tag as phenotypes in contraindication text.
# They are subject populations, not condition mentions — guide rule 3. Verbatim from DAKP
# ``ner.py`` ``_POPULATION_PHRASES``. Normalized exact match only.
POPULATION_PHRASES: frozenset[str] = frozenset(
    (
        "women",
        "men",
        "children",
        "patients",
        "individuals",
        "subjects",
        "women of childbearing potential",
        "childbearing potential",
        "women of childbearing age",
        "pregnant women",
    )
)

# Non-clinical tokens trimmed off the LEFT edge of a model span — guide rule 2. Verbatim from
# DAKP ``ner.py`` ``_HEDGE_TOKENS``. This is a CLOSED class and the polarity matters: anything
# NOT listed counts as a clinical qualifier and is kept, so `severe` / `active` / `pulmonary`
# survive. Left edge only, because trimming the right edge would turn the population descriptor
# "pregnant women" into the emittable "pregnant" and defeat POPULATION_PHRASES.
HEDGE_TOKENS: frozenset[str] = frozenset(
    (
        # determiners / quantifiers
        "a",
        "an",
        "the",
        "any",
        "all",
        "some",
        "other",
        "certain",
        "this",
        "these",
        "those",
        # prepositions / conjunctions
        "of",
        "with",
        "in",
        "to",
        "and",
        "or",
        "for",
        "on",
        "at",
        # temporal / evidential hedges
        "recent",
        "recently",
        "prior",
        "previous",
        "previously",
        "history",
        "known",
        "suspected",
        "possible",
        "potential",
        "current",
        "currently",
        "ongoing",
        "documented",
        "existing",
        "preexisting",
        "underlying",
        # population heads (the subject, not the condition)
        "patient",
        "patients",
        "women",
        "men",
        "children",
        "individuals",
        "subjects",
    )
)

#: Why a raw model span did not become a suggestion. Reported in the prelabel manifest so a
#: surprising suggestion count is diagnosable without re-running the model.
DROP_REASONS = ("label", "population", "hedge", "width", "overlap")

#: One text in, its entities out — the shape of ``GLiNER.predict_entities``.
Predictor = Callable[[str], list[dict[str, Any]]]
#: Many texts in, entities per text out — the shape of ``GLiNER.inference``.
BatchPredictor = Callable[[Sequence[str]], list[list[dict[str, Any]]]]


class PrelabelError(RuntimeError):
    """Raised when pre-labeling cannot produce a usable import file."""


@dataclass(frozen=True)
class Suggestion:
    """One model span in full-text coordinates. ``text == source[start:end]`` always holds."""

    start: int
    end: int
    label: str
    text: str
    score: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "label": self.label,
            "text": self.text,
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Suggestion:
        return cls(
            start=int(value["start"]),
            end=int(value["end"]),
            label=str(value["label"]),
            text=str(value["text"]),
            score=float(value["score"]),
        )


def normalize_surface(text: str) -> str:
    """Canonical comparison form for the blocklists (DAKP ``dictionary.normalize_text``).

    Lowercase, strip HTML tags, drop possessive ``'s``, collapse every non-alphanumeric ASCII
    run to a single space, trim.
    """
    lowered = _HTML_TAG.sub(" ", text.lower()).replace("'s", " ")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", lowered).split())


# --- windowing ----------------------------------------------------------------------------


def sentence_piece_spans(text: str) -> list[tuple[int, int]]:
    """Char spans of sentence-ish pieces tiling ``text`` exactly (gap-free, in order).

    Falls back to a single whole-text piece when the piece regex cannot tile the text (leading
    punctuation, dangling terminals), so callers always get contiguous coverage.
    """
    spans = [(match.start(), match.end()) for match in SENTENCE_PIECE.finditer(text)]
    if (
        not spans
        or spans[0][0] != 0
        or spans[-1][1] != len(text)
        or any(prev_end != next_start for (_, prev_end), (next_start, _) in itertools.pairwise(spans))
    ):
        return [(0, len(text))]
    return spans


def hard_split_spans(text: str, start: int, end: int, budget: int) -> list[tuple[int, int, int]]:
    """Split one over-budget ``text[start:end]`` slice into budget-sized ``(start, end, tokens)``
    windows at GLiNER token boundaries (each window stays an exact substring)."""
    matches = list(GLINER_TOKEN.finditer(text, start, end))
    return [
        (window[0].start(), window[-1].end(), len(window))
        for window in (matches[index : index + budget] for index in range(0, len(matches), budget))
    ]


def windows(text: str, budget: int) -> list[tuple[int, str]]:
    """Slice ``text`` into exact-substring windows of at most ``budget`` GLiNER word tokens.

    GLiNER truncates anything past ``config.max_len`` with only a ``UserWarning``, which silently
    costs recall on long DailyMed sections. Returns ``(start, window)`` pairs with
    ``window == text[start:start + len(window)]``, so an entity predicted on a window remaps to
    full-text coordinates by adding ``start``. Blank text yields no windows.
    """
    if not text.strip():
        return []
    budget = max(1, budget)
    pieces: list[tuple[int, int, int]] = []
    for piece_start, piece_end in sentence_piece_spans(text):
        tokens = len(GLINER_TOKEN.findall(text[piece_start:piece_end]))
        if tokens > budget:
            pieces.extend(hard_split_spans(text, piece_start, piece_end, budget))
        else:
            pieces.append((piece_start, piece_end, tokens))
    packed: list[tuple[int, str]] = []
    window_start = window_end = token_total = 0
    open_window = False
    for piece_start, piece_end, tokens in pieces:
        if open_window and token_total + tokens > budget:
            packed.append((window_start, text[window_start:window_end]))
            open_window = False
            token_total = 0
        if not open_window:
            window_start = piece_start
            open_window = True
        window_end = piece_end
        token_total += tokens
    if open_window:
        packed.append((window_start, text[window_start:window_end]))
    return packed


def token_budget(model: Any, override: int | None = None) -> int:
    """Window budget in word tokens: explicit ``override``, else ``config.max_len``, else the
    default — never below 1."""
    if override is not None:
        return max(1, int(override))
    max_len = getattr(getattr(model, "config", None), "max_len", None)
    if isinstance(max_len, int) and max_len >= 1:
        return max_len
    return DEFAULT_WORD_BUDGET


def model_max_width(model: Any, override: int | None = None) -> int:
    """Widest span the model can enumerate, in word tokens."""
    if override is not None:
        return max(1, int(override))
    max_width = getattr(getattr(model, "config", None), "max_width", None)
    if isinstance(max_width, int) and max_width >= 1:
        return max_width
    return DEFAULT_MAX_WIDTH


# --- span cleanup -------------------------------------------------------------------------


def trim_hedges(text: str, start: int, end: int) -> tuple[int, int] | None:
    """Trim leading :data:`HEDGE_TOKENS` off a span; ``None`` when nothing survives.

    Offsets move only to GLiNER token boundaries, so ``text[start:end]`` stays an exact slice.
    A span with no leading hedge is returned untouched.
    """
    tokens = list(GLINER_TOKEN.finditer(text, start, end))
    if not tokens:
        return None
    for position, token in enumerate(tokens):
        if normalize_surface(token.group()) not in HEDGE_TOKENS:
            return (token.start(), end) if position else (start, end)
    return None  # every token was a hedge ("patients with") — not a mention


def merge_straddling(window_spans: list[tuple[int, str]], spans_by_window: list[list[Suggestion]]) -> None:
    """Rejoin, in place, spans a hard window split cut across a phrase boundary.

    Sentence-piece windows tile the text exactly, so a mention cannot straddle those boundaries;
    only budget hard splits drop the whitespace between one window's last token and the next
    window's first, which can cut a multiword mention in two (``myasthenia | gravis``). When one
    window's last span ends flush at its end and the next window's first span begins flush at its
    start, the pair is re-unified; label and score come from the higher-scoring side, ties left.
    Merged spans re-enter the next pair, so a mention cut across two boundaries chains too.
    """
    for window_pair, span_pair in zip(
        itertools.pairwise(window_spans), itertools.pairwise(spans_by_window), strict=True
    ):
        (prev_start, prev_window), (curr_start, _curr_window) = window_pair
        prev_spans, curr_spans = span_pair
        prev_end = prev_start + len(prev_window)
        if curr_start == prev_end:
            continue  # contiguous (sentence-piece) boundary: mentions cannot span punctuation
        left = next((span for span in reversed(prev_spans) if span.end == prev_end), None)
        right = next((span for span in curr_spans if span.start == curr_start), None)
        if left is None or right is None:
            continue
        anchor = left if left.score >= right.score else right
        prev_spans.remove(left)
        curr_spans.remove(right)
        curr_spans.append(
            Suggestion(
                start=left.start,
                end=right.end,
                label=anchor.label,
                text="",  # re-derived from the source text by the caller
                score=max(left.score, right.score),
            )
        )


def select_spans(spans: Iterable[Suggestion]) -> list[Suggestion]:
    """De-overlap suggestions, most specific first: longest wins (guide rule 7).

    Ties break on score then ``(start, end, label)`` so the result does not depend on the order
    GLiNER returned its entities in. The canonical :class:`~medliner.schema.Example` refuses
    overlapping annotations outright, so an overlapping suggestion is not merely untidy — it is
    a span a human could accept and then fail to export.
    """
    ordered = sorted(spans, key=lambda item: (-(item.end - item.start), -item.score, item.start, item.end, item.label))
    kept: list[Suggestion] = []
    for span in ordered:
        if not any(span.start < other.end and other.start < span.end for other in kept):
            kept.append(span)
    return kept


def suggestions_from_windows(
    text: str,
    window_spans: list[tuple[int, str]],
    raw_by_window: Sequence[Sequence[dict[str, Any]]],
    *,
    max_width: int = DEFAULT_MAX_WIDTH,
    labels: Sequence[str] = PRELABEL_LABELS,
    drops: Counter[str] | None = None,
) -> list[Suggestion]:
    """Turn raw per-window GLiNER output into guide-conforming, full-text suggestions.

    The whole cleanup chain, in DAKP's order: label filter, offset remap, population blocklist,
    cross-window rejoin, left-edge hedge trim, population re-check on the trimmed surface, width
    cap, de-overlap. ``drops`` accumulates why spans were discarded.
    """
    drops = Counter() if drops is None else drops
    allowed = {label.strip().lower() for label in labels}
    spans_by_window: list[list[Suggestion]] = []
    for (window_start, _window), raw in zip(window_spans, raw_by_window, strict=True):
        kept: list[Suggestion] = []
        for entity in raw:
            label = str(entity.get("label", entity.get("type", ""))).strip().lower()
            if label not in allowed:
                drops["label"] += 1
                continue
            start = window_start + int(entity["start"])
            end = window_start + int(entity["end"])
            if start < 0 or end <= start or end > len(text):
                drops["label"] += 1
                continue
            if normalize_surface(text[start:end]) in POPULATION_PHRASES:
                drops["population"] += 1
                continue
            kept.append(
                Suggestion(
                    start=start, end=end, label=label, text=text[start:end], score=float(entity.get("score", 0.0))
                )
            )
        spans_by_window.append(kept)
    merge_straddling(window_spans, spans_by_window)

    trimmed: list[Suggestion] = []
    for span in itertools.chain.from_iterable(spans_by_window):
        bounds = trim_hedges(text, span.start, span.end)
        if bounds is None:
            drops["hedge"] += 1
            continue
        start, end = bounds
        surface = text[start:end]
        if normalize_surface(surface) in POPULATION_PHRASES:
            drops["population"] += 1
            continue
        if len(GLINER_TOKEN.findall(surface)) > max_width:
            drops["width"] += 1
            continue
        trimmed.append(Suggestion(start=start, end=end, label=span.label, text=surface, score=span.score))

    selected = select_spans(trimmed)
    drops["overlap"] += len(trimmed) - len(selected)
    return sorted(selected, key=lambda item: (item.start, item.end, item.label, item.text))


def suggest(
    predictor: Predictor,
    text: str,
    *,
    budget: int = DEFAULT_WORD_BUDGET,
    max_width: int = DEFAULT_MAX_WIDTH,
    labels: Sequence[str] = PRELABEL_LABELS,
    drops: Counter[str] | None = None,
) -> list[Suggestion]:
    """Pre-label one text with a single-text predictor. Convenience wrapper for tests and scoring."""
    window_spans = windows(text, budget)
    raw = [predictor(window) for _, window in window_spans]
    return suggestions_from_windows(text, window_spans, raw, max_width=max_width, labels=labels, drops=drops)


# --- Label Studio prediction shape ---------------------------------------------------------


def model_version(model_id: str = DEFAULT_MODEL_ID, threshold: float = DEFAULT_THRESHOLD) -> str:
    """Stable identifier for one (model, threshold) pair, shown in the Label Studio UI."""
    return f"{model_id.rsplit('/', 1)[-1]}@{threshold:g}"


def _region_id(task_id: str, span: Suggestion) -> str:
    """Deterministic region id so re-running the pre-labeler reproduces the import file byte for
    byte, matching the guarantee ``candidates.build_import_tasks`` already makes for task ids."""
    digest = blake3(f"{task_id}\n{span.start}\n{span.end}\n{span.label}".encode()).hexdigest()
    return f"pl-{digest[:12]}"


def build_prediction(task_id: str, spans: Sequence[Suggestion], *, version: str) -> dict[str, Any]:
    """One Label Studio prediction object for a task's suggestions.

    ``from_name``/``to_name`` must match ``configs/label_studio_ner.xml``; Label Studio drops a
    prediction whose control names it cannot resolve, without reporting an error.
    """
    result = [
        {
            "id": _region_id(task_id, span),
            "from_name": FROM_NAME,
            "to_name": TO_NAME,
            "type": "labels",
            "score": round(span.score, 6),
            "value": {"start": span.start, "end": span.end, "text": span.text, "labels": [span.label]},
        }
        for span in spans
    ]
    score = round(sum(span.score for span in spans) / len(spans), 6) if spans else 0.0
    return {"model_version": version, "score": score, "result": result}


def attach_predictions(
    tasks: Sequence[dict[str, Any]],
    suggestions: dict[str, list[Suggestion]],
    *,
    version: str,
) -> list[dict[str, Any]]:
    """Copy ``tasks`` with a ``predictions`` array attached to each.

    A task with no suggestions still gets an empty prediction: that is what tells Label Studio
    the model has seen the task and found nothing, rather than that it was never pre-labeled.
    """
    attached: list[dict[str, Any]] = []
    for task in tasks:
        task_id = str(task["id"])
        copied = dict(task)
        copied["predictions"] = [build_prediction(task_id, suggestions.get(task_id, []), version=version)]
        attached.append(copied)
    return attached


# --- per-text cache -------------------------------------------------------------------------


def cache_key(text: str, *, model_id: str, threshold: float, labels: Sequence[str], budget: int) -> str:
    """Key a text's suggestions by everything that can change them.

    Mirrors DAKP's mention-cache key material (model, configuration fingerprint, normalized
    text), so adding candidates re-runs the model only over the genuinely new texts.
    """
    material = "|".join([model_id, f"{threshold:g}", ",".join(labels), str(budget), " ".join(text.split()).lower()])
    return blake3(material.encode()).hexdigest()


class PrelabelCache:
    """A plain JSON map of :func:`cache_key` to suggestions.

    Every failure degrades to a miss: a cache is a build-time convenience, and a corrupt one must
    never be able to block annotation.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.entries: dict[str, list[Suggestion]] = {}
        self.hits = 0
        self.misses = 0

    def load(self) -> PrelabelCache:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return self
        if not isinstance(payload, dict) or payload.get("schema_version") != CACHE_SCHEMA_VERSION:
            return self
        entries = payload.get("entries")
        if not isinstance(entries, dict):
            return self
        for key, spans in entries.items():
            try:
                self.entries[str(key)] = [Suggestion.from_dict(span) for span in spans]
            except (AttributeError, KeyError, TypeError, ValueError):
                continue
        return self

    def get(self, key: str) -> list[Suggestion] | None:
        found = self.entries.get(key)
        if found is None:
            self.misses += 1
            return None
        self.hits += 1
        return found

    def put(self, key: str, spans: list[Suggestion]) -> None:
        self.entries[key] = spans

    def save(self) -> None:
        payload = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "entries": {key: [span.as_dict() for span in spans] for key, spans in sorted(self.entries.items())},
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        except OSError:
            return


# --- batched model runner ---------------------------------------------------------------------


def _cuda_device() -> str:
    """``cuda`` only when torch actually ships kernels for this GPU's architecture.

    Same guard as ``evaluation._cuda_available``: an RTX 5070 Ti is ``sm_120`` and a wheel built
    without those kernels reports ``is_available()`` true and then fails at the first kernel.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return "cpu"
        major, minor = torch.cuda.get_device_capability()
        return "cuda" if f"sm_{major}{minor}" in set(torch.cuda.get_arch_list()) else "cpu"
    except (ImportError, RuntimeError):
        return "cpu"


def check_model_budgets(model: Any, *, budget: int, max_width: int) -> None:
    """Refuse to run with budgets the loaded checkpoint cannot honour.

    GLiNER truncates input past ``config.max_len`` with only a ``UserWarning`` and never
    enumerates a span wider than ``config.max_width``. Both failures are silent losses of recall,
    which is exactly the thing windowing exists to prevent, so they are errors here.
    """
    model_budget = token_budget(model)
    if budget > model_budget:
        raise PrelabelError(
            f"word budget {budget} exceeds the checkpoint's max_len {model_budget}; "
            "GLiNER would truncate each window with only a warning"
        )
    model_width = model_max_width(model)
    if max_width > model_width:
        raise PrelabelError(
            f"max span width {max_width} exceeds the checkpoint's max_width {model_width}; "
            "GLiNER never enumerates a wider span, so the extra width is unreachable"
        )


def load_model(model_id: str = DEFAULT_MODEL_ID, *, device: str | None = None) -> tuple[Any, str]:
    """Load the pre-labeling checkpoint; returns ``(model, device)``."""
    try:
        from gliner import GLiNER
    except ImportError as exc:  # pragma: no cover - gliner is a hard dependency
        raise PrelabelError("pre-labeling requires the 'gliner' package; run 'make sync'") from exc
    resolved = device or _cuda_device()
    model = GLiNER.from_pretrained(model_id, map_location=resolved)
    model.eval()
    return model, resolved


def batch_predictor(
    model: Any,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    labels: Sequence[str] = PRELABEL_LABELS,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> BatchPredictor:
    """A :data:`BatchPredictor` over ``GLiNER.inference``.

    ``inference`` is 0.2.28's batched entry point (``batch_predict_entities`` forwards to it and
    warns). ``flat_ner``/``multi_label`` are GLiNER's defaults but are passed explicitly because
    the no-overlap span policy depends on them.
    """

    def predict(texts: Sequence[str]) -> list[list[dict[str, Any]]]:
        if not texts:
            return []
        return model.inference(
            list(texts),
            list(labels),
            flat_ner=True,
            threshold=threshold,
            multi_label=False,
            batch_size=batch_size,
        )

    return predict


def prelabel_texts(
    predict: BatchPredictor,
    texts_by_id: dict[str, str],
    *,
    budget: int = DEFAULT_WORD_BUDGET,
    max_width: int = DEFAULT_MAX_WIDTH,
    labels: Sequence[str] = PRELABEL_LABELS,
    cache: PrelabelCache | None = None,
    model_id: str = DEFAULT_MODEL_ID,
    threshold: float = DEFAULT_THRESHOLD,
    drops: Counter[str] | None = None,
) -> dict[str, list[Suggestion]]:
    """Pre-label many texts in as few forward passes as possible.

    Every window of every uncached text goes into one flat list, sorted longest-first so a batch
    is filled with similarly-sized inputs and padding waste stays low, and is handed to ``predict``
    in a single call. DAKP calls the model once per window sequentially; batching is the main
    reason this is fast enough to run over the whole candidate set before a session.
    """
    drops = Counter() if drops is None else drops
    results: dict[str, list[Suggestion]] = {}
    keys: dict[str, str] = {}
    pending: dict[str, list[tuple[int, str]]] = {}

    for task_id, text in texts_by_id.items():
        key = cache_key(text, model_id=model_id, threshold=threshold, labels=labels, budget=budget)
        keys[task_id] = key
        cached = cache.get(key) if cache is not None else None
        if cached is not None:
            results[task_id] = cached
            continue
        pending[task_id] = windows(text, budget)

    flat: list[tuple[str, int, str]] = [
        (task_id, index, window)
        for task_id, window_spans in pending.items()
        for index, (_, window) in enumerate(window_spans)
    ]
    # Longest first: GLiNER pads each batch to its longest member, so mixing a 400-token window
    # with 3-token FAERS strings would pad the short ones ~130x.
    flat.sort(key=lambda item: (-len(item[2]), item[0], item[1]))
    predicted = predict([window for _, _, window in flat]) if flat else []
    if len(predicted) != len(flat):
        raise PrelabelError(f"predictor returned {len(predicted)} results for {len(flat)} windows")

    raw_by_id: dict[str, list[list[dict[str, Any]]]] = {
        task_id: [[] for _ in window_spans] for task_id, window_spans in pending.items()
    }
    for (task_id, index, _), entities in zip(flat, predicted, strict=True):
        raw_by_id[task_id][index] = list(entities)

    for task_id, window_spans in pending.items():
        spans = suggestions_from_windows(
            texts_by_id[task_id],
            window_spans,
            raw_by_id[task_id],
            max_width=max_width,
            labels=labels,
            drops=drops,
        )
        results[task_id] = spans
        if cache is not None:
            cache.put(keys[task_id], spans)
    return results


def prelabel_manifest(
    tasks: Sequence[dict[str, Any]],
    suggestions: dict[str, list[Suggestion]],
    *,
    model_id: str,
    threshold: float,
    labels: Sequence[str],
    budget: int,
    max_width: int,
    device: str,
    version: str,
    drops: Counter[str],
    elapsed_seconds: float,
    cache_hits: int = 0,
    cache_misses: int = 0,
) -> dict[str, Any]:
    """Everything needed to reproduce or explain a pre-labeling run."""
    spans = [span for task in tasks for span in suggestions.get(str(task["id"]), [])]
    suggested_tasks = sum(1 for task in tasks if suggestions.get(str(task["id"])))
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "model_id": model_id,
        "model_version": version,
        "threshold": threshold,
        "labels": list(labels),
        "word_budget": budget,
        "max_width": max_width,
        "device": device,
        "task_count": len(tasks),
        "tasks_with_suggestions": suggested_tasks,
        "suggestion_count": len(spans),
        "label_counts": dict(sorted(Counter(span.label for span in spans).items())),
        "dropped": {reason: drops.get(reason, 0) for reason in DROP_REASONS},
        "cache": {"hits": cache_hits, "misses": cache_misses},
        "elapsed_seconds": round(elapsed_seconds, 3),
    }


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_MAX_WIDTH",
    "DEFAULT_MODEL_ID",
    "DEFAULT_THRESHOLD",
    "DEFAULT_WORD_BUDGET",
    "DROP_REASONS",
    "FROM_NAME",
    "HEDGE_TOKENS",
    "MANIFEST_SCHEMA_VERSION",
    "POPULATION_PHRASES",
    "PRELABEL_LABELS",
    "TO_NAME",
    "BatchPredictor",
    "PrelabelCache",
    "PrelabelError",
    "Predictor",
    "Suggestion",
    "attach_predictions",
    "batch_predictor",
    "build_prediction",
    "cache_key",
    "check_model_budgets",
    "load_model",
    "merge_straddling",
    "model_max_width",
    "model_version",
    "normalize_surface",
    "prelabel_manifest",
    "prelabel_texts",
    "select_spans",
    "suggest",
    "suggestions_from_windows",
    "token_budget",
    "trim_hedges",
    "windows",
]
