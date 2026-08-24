"""Tests for the local LLM client against a stubbed llama.cpp server."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from medliner import llm


class StubHandler(BaseHTTPRequestHandler):
    """Minimal /health + /v1/chat/completions stub; records the last request body."""

    reply_content = "shortened text"
    reply_reasoning: str | None = None
    last_body: dict = {}
    request_count = 0

    def log_message(self, *_args):  # keep the test output quiet
        pass

    def do_GET(self):
        if self.path == "/health":
            self._respond({"status": "ok"})
        else:
            self._respond({"error": "not found"}, status=404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        type(self).last_body = json.loads(self.rfile.read(length).decode())
        type(self).request_count += 1
        message = {"role": "assistant", "content": self.reply_content}
        if self.reply_reasoning is not None:
            message["reasoning_content"] = self.reply_reasoning
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
    StubHandler.reply_content = "shortened text"
    StubHandler.reply_reasoning = None
    StubHandler.last_body = {}
    StubHandler.request_count = 0
    httpd = HTTPServer(("127.0.0.1", 0), StubHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()
    thread.join()


def test_health_reports_ok(server):
    assert llm.health(server)
    assert not llm.health("http://127.0.0.1:1")  # connection refused, not an exception


def test_chat_disables_reasoning_in_the_request(server):
    assert llm.chat([{"role": "user", "content": "hi"}], url=server) == "shortened text"
    assert StubHandler.last_body["chat_template_kwargs"] == {"enable_thinking": False}


def test_chat_falls_back_to_reasoning_content(server):
    StubHandler.reply_content = ""
    StubHandler.reply_reasoning = "reasoned answer"
    assert llm.chat([{"role": "user", "content": "hi"}], url=server) == "reasoned answer"


def test_chat_raises_when_both_channels_are_empty(server):
    StubHandler.reply_content = ""
    StubHandler.reply_reasoning = ""
    with pytest.raises(llm.LLMError, match="empty completion"):
        llm.chat([{"role": "user", "content": "hi"}], url=server)


def test_shorten_text_returns_validated_rewrite(server):
    StubHandler.reply_content = "contraindicated in active peptic ulcer disease"
    original = " ".join(["word"] * 400)
    shortened, empty_hint = llm.shorten_text(original, max_words=300, url=server)
    assert shortened == "contraindicated in active peptic ulcer disease"
    assert not empty_hint
    assert "preserving" in StubHandler.last_body["messages"][0]["content"]


def test_shorten_text_keeps_original_when_reply_is_not_shorter(server):
    original = " ".join(["word"] * 10)
    StubHandler.reply_content = " ".join(["word"] * 50)
    shortened, empty_hint = llm.shorten_text(original, max_words=5, url=server)
    assert shortened == original
    assert not empty_hint


def test_shorten_text_flags_the_no_entity_marker(server):
    StubHandler.reply_content = llm.NO_ENTITY_MARKER
    shortened, empty_hint = llm.shorten_text(" ".join(["word"] * 400), max_words=300, url=server)
    assert shortened != llm.NO_ENTITY_MARKER  # original text is kept
    assert empty_hint


def test_shorten_text_keeps_original_when_the_server_is_down():
    original = " ".join(["word"] * 400)
    shortened, empty_hint = llm.shorten_text(original, max_words=300, url="http://127.0.0.1:1")
    assert shortened == original
    assert not empty_hint


def test_shorten_text_caches_replies_in_sqlite(server, tmp_path):
    cache = tmp_path / "cache" / "shorten.sqlite3"
    original = " ".join(["word"] * 400)
    StubHandler.reply_content = "a much shorter text"
    assert llm.shorten_text(original, max_words=48, url=server, cache=cache) == ("a much shorter text", False)
    assert StubHandler.request_count == 1
    # Same content + threshold: served from sqlite, the server is never hit again.
    StubHandler.reply_content = "different reply that would win if the server were asked"
    assert llm.shorten_text(original, max_words=48, url=server, cache=cache) == ("a much shorter text", False)
    assert StubHandler.request_count == 1
    # A different threshold is a different key: goes back to the server.
    llm.shorten_text(original, max_words=300, url=server, cache=cache)
    assert StubHandler.request_count == 2
    with __import__("sqlite3").connect(cache) as connection:
        rows = connection.execute("SELECT key, reply FROM rewrites").fetchall()
    assert len(rows) == 2
    assert all(row[1] for row in rows)


def test_shorten_text_tolerates_a_broken_cache(server, tmp_path):
    broken = tmp_path / "dir"  # a directory: sqlite cannot open it as a database file
    broken.mkdir()
    original = " ".join(["word"] * 400)
    StubHandler.reply_content = "a shorter text"
    shortened, empty_hint = llm.shorten_text(original, max_words=48, url=server, cache=broken)
    assert (shortened, empty_hint) == ("a shorter text", False)  # cache failure degrades to a miss


def test_run_shorten_rewrites_over_long_rows_via_the_cli(tmp_path, monkeypatch, capsys, server):
    import json

    from medliner import cli

    long_text = " ".join(["filler"] * 60)  # over the default 48-word threshold, under the old 300
    medium_text = " ".join(["detail"] * 55)
    raw = tmp_path / "candidates.ndjson"
    rows = [
        {"text": "Indicated for asthma.", "task": "indication", "source_family": "dailymed"},
        {"text": long_text, "task": "contraindication", "source_family": "dailymed"},
        {"text": medium_text, "task": "indication", "source_family": "dailymed"},
    ]
    raw.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    monkeypatch.setenv("MEDLINER_RAW_CANDIDATES", str(raw))
    monkeypatch.setenv("MEDLINER_WORKDIR", str(tmp_path / "work"))
    monkeypatch.setenv("MEDLINER_LLM_URL", server)
    StubHandler.reply_content = "Contraindicated in active peptic ulcer disease."

    assert cli.main(["shorten"]) == 0
    assert "2/2 over-48-word texts shortened" in capsys.readouterr().out
    output = tmp_path / "candidates.shortened.ndjson"
    rewritten = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rewritten[0]["text"] == "Indicated for asthma."  # under the cap: never sent to the LLM
    assert rewritten[1]["text"] == "Contraindicated in active peptic ulcer disease."
    assert rewritten[2]["text"] == "Contraindicated in active peptic ulcer disease."
    manifest = json.loads(output.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert manifest["rows"] == 3
    assert manifest["over_long"] == 2
    assert manifest["shortened"] == 2
    assert manifest["processed_indices"] == [1, 2]
    assert manifest["llm_url"] == server


def test_run_shorten_resumes_an_interrupted_run(tmp_path, monkeypatch, capsys, server):
    import json

    from medliner import cli

    raw = tmp_path / "candidates.ndjson"
    rows = [
        {"text": " ".join(["filler"] * 60), "task": "indication", "source_family": "dailymed"},
        {"text": " ".join(["filler"] * 70), "task": "indication", "source_family": "dailymed"},
        {"text": " ".join(["filler"] * 80), "task": "indication", "source_family": "dailymed"},
    ]
    raw.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    monkeypatch.setenv("MEDLINER_RAW_CANDIDATES", str(raw))
    monkeypatch.setenv("MEDLINER_WORKDIR", str(tmp_path / "work"))
    monkeypatch.setenv("MEDLINER_LLM_URL", server)
    StubHandler.reply_content = "short version"

    # First run processes one row then "crashes" (simulated by limit); second run finishes the rest.
    assert cli.main(["shorten", "--limit", "1"]) == 0
    first = json.loads((tmp_path / "candidates.shortened.manifest.json").read_text(encoding="utf-8"))
    assert first["processed_indices"] == [0]
    assert StubHandler.request_count == 1

    assert cli.main(["shorten"]) == 0
    out = capsys.readouterr().out
    assert "1 resumed" in out
    assert StubHandler.request_count == 3  # only the two remaining rows hit the server
    manifest = json.loads((tmp_path / "candidates.shortened.manifest.json").read_text(encoding="utf-8"))
    assert manifest["processed_indices"] == [0, 1, 2]
    assert manifest["resumed_from_previous"] == 1
    final = [
        json.loads(line) for line in (tmp_path / "candidates.shortened.ndjson").read_text(encoding="utf-8").splitlines()
    ]
    assert all(row["text"] == "short version" for row in final)

    # --force re-runs everything despite complete progress.
    assert cli.main(["shorten", "--force"]) == 0
    assert StubHandler.request_count == 6
