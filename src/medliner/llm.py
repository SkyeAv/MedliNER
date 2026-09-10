"""Client for OpenAI-compatible chat servers: the local llama.cpp server and 9router.

The primary server is llama.cpp (Ornith-1.0-9B, ``llama-server -np 4 -cb --kv-unified``),
started by ``make llm``. Two details of that deployment shape the client:

- the model is a reasoner: unless ``enable_thinking`` is turned off it spends the whole
  token budget on ``reasoning_content`` and returns an empty ``content``;
- it serves four parallel slots with continuous batching, so callers may run several requests
  concurrently without queuing.

A second endpoint — the 9router combo proxy — can be configured through
``MEDLINER_9ROUTER_URL`` / ``MEDLINER_9ROUTER_API_KEY`` / ``MEDLINER_9ROUTER_MODEL``
(:func:`router_endpoint`). It speaks the same chat-completions API but needs a Bearer token
and a ``model`` field, and must NOT receive llama.cpp's ``chat_template_kwargs`` (upstream
providers may reject unknown fields). 9router also has no ``/health`` (it is a Next.js app),
so health checks fall back to an authenticated ``GET /v1/models``.

Nothing here is required by the deterministic pipeline: the LLM only rewrites over-long
candidate texts on explicit request (``medliner shorten``) and paraphrases gold examples
(``medliner synthesize``), and every reply is validated before it is used.
"""

from __future__ import annotations

import json
import os
import sqlite3
import urllib.error
import urllib.request
from dataclasses import dataclass
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
    """Raised when the chat server is unreachable or returns an unusable reply."""


@dataclass(frozen=True)
class Endpoint:
    """One OpenAI-compatible chat endpoint (llama.cpp locally, or the 9router proxy).

    ``send_thinking_kwargs`` selects llama.cpp's ``chat_template_kwargs`` reasoning switch;
    proxies forward unknown fields to upstream providers that may reject them, so only the
    local server gets it. ``name`` is stable and machine-readable: it lands in cache keys,
    manifests, and the synthetic examples' ``source.generator`` audit stamp.
    """

    name: str
    url: str
    api_key: str | None = None
    model: str | None = None
    send_thinking_kwargs: bool = False


def default_endpoint(url: str | None = None) -> Endpoint:
    """The local llama.cpp endpoint; ``MEDLINER_LLM_API_KEY``/``MEDLINER_LLM_MODEL`` augment it."""
    return Endpoint(
        name="llama.cpp",
        url=llm_url(url),
        api_key=os.environ.get("MEDLINER_LLM_API_KEY") or None,
        model=os.environ.get("MEDLINER_LLM_MODEL") or None,
        send_thinking_kwargs=True,
    )


def router_endpoint() -> Endpoint | None:
    """The 9router combo endpoint, or None when ``MEDLINER_9ROUTER_URL`` is not configured.

    The API key is mandatory once the URL is set: without it every request would 401, which
    would surface as synthesis ``llm_error`` rejections — failing here makes the
    misconfiguration obvious instead.
    """
    url = os.environ.get("MEDLINER_9ROUTER_URL")
    if not url:
        return None
    api_key = os.environ.get("MEDLINER_9ROUTER_API_KEY")
    if not api_key:
        raise LLMError("MEDLINER_9ROUTER_URL is set but MEDLINER_9ROUTER_API_KEY is not")
    return Endpoint(
        name="9router",
        url=url.rstrip("/"),
        api_key=api_key,
        model=os.environ.get("MEDLINER_9ROUTER_MODEL", "9router"),
        send_thinking_kwargs=False,
    )


def llm_url(value: str | None = None) -> str:
    return (value or os.environ.get("MEDLINER_LLM_URL", DEFAULT_LLM_URL)).rstrip("/")


def _get_json(endpoint: Endpoint, path: str, *, timeout: float) -> Any:
    headers = {"Accept": "application/json"}
    if endpoint.api_key:
        headers["Authorization"] = f"Bearer {endpoint.api_key}"
    request = urllib.request.Request(f"{endpoint.url}{path}", headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def health(url: str | None = None, *, endpoint: Endpoint | None = None, timeout: float = 2.0) -> bool:
    """True when the endpoint answers ``/health`` (llama.cpp) or ``/v1/models`` (OpenAI-style).

    9router has no ``/health`` route — it is a Next.js app that 404s — so an authenticated
    model listing doubles as its liveness probe.
    """
    endpoint = endpoint or default_endpoint(url)
    try:
        payload = _get_json(endpoint, "/health", timeout=timeout)
        if isinstance(payload, dict) and payload.get("status") == "ok":
            return True
    except (OSError, ValueError):
        pass
    try:
        payload = _get_json(endpoint, "/v1/models", timeout=timeout)
    except (OSError, ValueError):
        return False
    return isinstance(payload, dict) and isinstance(payload.get("data"), list)


def chat(
    messages: list[dict[str, str]],
    *,
    url: str | None = None,
    endpoint: Endpoint | None = None,
    max_tokens: int = 512,
    timeout: float = 120.0,
) -> str:
    """One chat completion; falls back to ``reasoning_content`` when ``content`` is empty.

    Raises :class:`LLMError` when both channels come back empty (usually a sign that the
    token budget was consumed by truncated reasoning).
    """
    endpoint = endpoint or default_endpoint(url)
    payload: dict[str, Any] = {"messages": messages, "max_tokens": max_tokens}
    if endpoint.model:
        payload["model"] = endpoint.model
    if endpoint.send_thinking_kwargs:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    headers = {"Content-Type": "application/json"}
    if endpoint.api_key:
        headers["Authorization"] = f"Bearer {endpoint.api_key}"
    request = urllib.request.Request(
        f"{endpoint.url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers=headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            reply: dict[str, Any] = json.loads(response.read().decode())
    except (OSError, ValueError) as exc:
        raise LLMError(f"LLM request to {endpoint.name} ({endpoint.url}) failed: {exc}") from exc
    message = (reply.get("choices") or [{}])[0].get("message") or {}
    content = str(message.get("content") or "").strip()
    if not content:
        content = str(message.get("reasoning_content") or "").strip()
    if not content:
        raise LLMError(f"{endpoint.name} returned an empty completion")
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
