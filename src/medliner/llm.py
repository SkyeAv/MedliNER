"""Client for the local llama.cpp chat server (the ``medliner`` target in ``$MODELS_DIR``).

The server (Ornith-1.0-9B, ``llama-server -np 2 -cb --kv-unified``) speaks the
OpenAI-compatible chat-completions API. Two details of this deployment shape the client:

- the model is a reasoner: unless ``enable_thinking`` is turned off it spends the whole
  token budget on ``reasoning_content`` and returns an empty ``content``;
- it serves two parallel slots with continuous batching, so callers may run two requests
  concurrently without queuing.

Nothing here is required by the deterministic pipeline: the LLM only rewrites over-long
candidate texts on explicit request (``medliner shorten``), and every rewrite is validated
before it replaces the original.
"""

from __future__ import annotations

import json
import os
import sqlite3
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_LLM_URL = "http://127.0.0.1:8080"
#: Marker the model is told to return when a text has no condition mention to preserve.
NO_ENTITY_MARKER = "NONE"

SHORTEN_PROMPT = """\
Shorten the following medical text to at most {max_words} words while preserving every \
mention of a disease, condition, or phenotype verbatim (same wording as the original). \
Remove boilerplate, cross-references, and dosage/administration detail first. \
Return only the shortened text, with no commentary. \
If the text contains no disease, condition, or phenotype mention at all, reply with exactly {marker}.

Text:
{text}"""


def default_cache_path() -> Path:
    """Cache location ($MEDLINER_SHORTEN_CACHE, default ``<workdir>/shorten-cache.sqlite3``)."""
    from .cli import workdir  # local import: cli pulls heavy deps

    return Path(os.environ.get("MEDLINER_SHORTEN_CACHE", str(workdir() / "shorten-cache.sqlite3")))


def _cache_key(text: str, max_words: int) -> str:
    """Content key for a rewrite; version-tagged so prompt changes invalidate old entries."""
    from blake3 import blake3

    return blake3(f"medliner-shorten-v1\n{max_words}\n{text}".encode()).hexdigest()


def cache_lookup(cache: str | Path, text: str, *, max_words: int) -> str | None:
    """Cached raw model reply for ``text``, or None on a miss (or any cache problem).

    A broken/unreadable cache degrades to a cache miss, never to a failed run.
    """
    try:
        with sqlite3.connect(str(cache)) as connection:
            row = connection.execute(
                "SELECT reply FROM rewrites WHERE key = ?", (_cache_key(text, max_words),)
            ).fetchone()
    except (sqlite3.Error, OSError):
        return None
    return str(row[0]) if row else None


def cache_store(cache: str | Path, text: str, *, max_words: int, reply: str) -> None:
    """Persist a successful raw reply; failures are swallowed (the run must not die on I/O)."""
    try:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(cache)) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS rewrites ("
                "key TEXT PRIMARY KEY, reply TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT (datetime('now'))"
                ")"
            )
            connection.execute(
                "INSERT OR REPLACE INTO rewrites (key, reply) VALUES (?, ?)",
                (_cache_key(text, max_words), reply),
            )
    except (sqlite3.Error, OSError):
        pass


class LLMError(RuntimeError):
    """Raised when the local LLM server is unreachable or returns an unusable reply."""


def llm_url(value: str | None = None) -> str:
    return (value or os.environ.get("MEDLINER_LLM_URL", DEFAULT_LLM_URL)).rstrip("/")


def health(url: str | None = None, *, timeout: float = 2.0) -> bool:
    """True when the server answers ``/health`` with an ok status."""
    try:
        with urllib.request.urlopen(f"{llm_url(url)}/health", timeout=timeout) as response:
            payload = json.loads(response.read().decode())
    except (OSError, ValueError, urllib.error.URLError):
        return False
    return isinstance(payload, dict) and payload.get("status") == "ok"


def chat(
    messages: list[dict[str, str]],
    *,
    url: str | None = None,
    max_tokens: int = 512,
    timeout: float = 120.0,
) -> str:
    """One chat completion with reasoning disabled; falls back to ``reasoning_content``.

    Raises :class:`LLMError` when both channels come back empty (usually a sign that the
    token budget was consumed by truncated reasoning).
    """
    request = urllib.request.Request(
        f"{llm_url(url)}/v1/chat/completions",
        data=json.dumps(
            {
                "messages": messages,
                "max_tokens": max_tokens,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload: dict[str, Any] = json.loads(response.read().decode())
    except (OSError, ValueError) as exc:
        raise LLMError(f"LLM request to {llm_url(url)} failed: {exc}") from exc
    message = (payload.get("choices") or [{}])[0].get("message") or {}
    content = str(message.get("content") or "").strip()
    if not content:
        content = str(message.get("reasoning_content") or "").strip()
    if not content:
        raise LLMError("LLM returned an empty completion")
    return content


def shorten_text(
    text: str,
    *,
    max_words: int,
    url: str | None = None,
    max_tokens: int = 2048,
    cache: str | Path | None = None,
) -> tuple[str, bool]:
    """Shorten ``text`` to at most ``max_words`` words, preserving entity mentions.

    Returns ``(text, empty_hint)``. On any failure — unreachable server, empty reply, or a
    reply that is not actually shorter — the original text is returned unchanged, so a
    failed run never corrupts the candidate pool. ``empty_hint`` is True when the model
    reported no condition mention; it is a review signal only, never a drop decision.

    Successful replies are cached in the sqlite database at ``cache`` (if given), keyed by
    content + threshold + prompt version, so re-runs and overlapping inputs skip the model.
    """
    reply: str | None = cache_lookup(cache, text, max_words=max_words) if cache else None
    if reply is None:
        prompt = SHORTEN_PROMPT.format(max_words=max_words, marker=NO_ENTITY_MARKER, text=text)
        try:
            reply = chat([{"role": "user", "content": prompt}], url=url, max_tokens=max_tokens)
        except LLMError:
            return text, False
        if cache:
            cache_store(cache, text, max_words=max_words, reply=reply)
    if reply.strip().upper() == NO_ENTITY_MARKER:
        return text, True
    if len(reply.split()) >= len(text.split()):
        return text, False
    return reply, False
