SHELL := /usr/bin/env bash
.DEFAULT_GOAL := help

# Repo-relative defaults keep a fresh checkout portable; `.envrc.local` can point at a ready DAKP export.
export MEDLINER_RAW_CANDIDATES ?= $(CURDIR)/data/label-studio/candidates.ndjson
export MEDLINER_BENCHMARK ?= $(CURDIR)/data/materialized/ingested/ner_gold.json
export MEDLINER_EXPORT_BUNDLE ?= $(CURDIR)/data/dakp-export
export MEDLINER_LABEL_STUDIO_EXPORT ?= $(CURDIR)/data/label-studio/reviewed.json
export MEDLINER_WORKDIR ?= $(CURDIR)/data/materialized
# Pre-labeling checkpoint and score floor. Same values the sibling DAKP pipeline mines with.
export MEDLINER_PRELABEL_MODEL ?= gliner-community/gliner_large-v2.5
export MEDLINER_PRELABEL_THRESHOLD ?= 0.35
export MEDLINER_LABEL_STUDIO_PORT ?= 9030
export MEDLINER_LABEL_STUDIO_IMAGE ?= docker.io/heartexlabs/label-studio:latest
export MEDLINER_LABEL_STUDIO_HOST ?= 127.0.0.1
# Comma-separated user:password accounts ensured at start.
export MEDLINER_LABEL_STUDIO_ANNOTATORS ?=
export MEDLINER_LLM_URL ?= http://127.0.0.1:8080
# Model checkout with the `medliner` llama-server target; falls back to ~/Desktop/MODELS.
MODELS_DIR ?= $(if $(wildcard $(CURDIR)/models/Makefile),$(CURDIR)/models,$(HOME)/Desktop/MODELS)
LLM_TMUX_SESSION ?= medliner-llm
# Quick (account-less, non-scoped) Cloudflare tunnel: `make tunnel` publishes the Label Studio
# port on a random https://<slug>.trycloudflare.com URL so annotators off the LAN can reach it.
# cloudflared is the single command of a detached tmux pane, so the pane PID is the cloudflared
# PID; log, URL, and PID live under data/tunnel/ (gitignored). The `--logfile` path in the process
# argv is the identity marker that separates this repo's tunnel from any other cloudflared here.
TUNNEL_TMUX_SESSION ?= medliner-tunnel
TUNNEL_DIR ?= $(CURDIR)/data/tunnel
TUNNEL_LOG = $(TUNNEL_DIR)/cloudflared.log
TUNNEL_URL_FILE = $(TUNNEL_DIR)/url.txt
TUNNEL_PID_FILE = $(TUNNEL_DIR)/cloudflared.pid
# A wildcard bind address is a listen address, not a dial address: aim the tunnel at loopback.
# An explicitly published host (a LAN address) still has to be dialed as it is.
TUNNEL_BIND_HOST = $(strip $(MEDLINER_LABEL_STUDIO_HOST))
TUNNEL_ORIGIN_HOST = $(if $(TUNNEL_BIND_HOST),$(if $(filter 0.0.0.0 :: ::0 0 [::],$(TUNNEL_BIND_HOST)),127.0.0.1,$(TUNNEL_BIND_HOST)),127.0.0.1)
TUNNEL_ORIGIN = http://$(TUNNEL_ORIGIN_HOST):$(strip $(MEDLINER_LABEL_STUDIO_PORT))
TUNNEL_URL_PATTERN = https://[a-z0-9-]+\.trycloudflare\.com
# Shorten stage: word threshold (≈3-4 short sentences), parallel requests, reply cache.
export MEDLINER_SHORTEN_MAX_WORDS ?= 48
export MEDLINER_SHORTEN_WORKERS ?= 4
export MEDLINER_SHORTEN_CACHE ?= $(CURDIR)/data/materialized/shorten-cache.sqlite3

# Triton locates libcuda through /sbin/ldconfig; without a loader cache that call fails inside
# the backward pass. An empty value is ignored by Triton, so this is safe on normal systems.
export TRITON_LIBCUDA_PATH ?= $(shell test -x /sbin/ldconfig || for d in /run/opengl-driver/lib /usr/lib64 /usr/lib/x86_64-linux-gnu /usr/lib; do test -e $$d/libcuda.so.1 && echo $$d && break; done)

.PHONY: help setup llm llm-stop shorten prepare annotate stop export tunnel tunnel-stop lint fmt fmt-check test check clean

help:
	@printf '%s\n' \
		'MedliNER pipeline:' \
		'  make setup              Install or update the uv environment' \
		'' \
		'Data (everything before Label Studio, one command):' \
		'  make prepare            Sample, shorten long texts via the LLM (if healthy), attach GLiNER suggestions' \
		'' \
		'Label Studio:' \
		'  make annotate           Start the production Label Studio server with tasks imported' \
		'  make stop               Remove the Label Studio container (annotations survive in its data volume)' \
		'  make export             Download the reviewed production annotations from the running server' \
		'  make tunnel             Publish the server on a public trycloudflare.com URL (detached tmux)' \
		'  make tunnel-stop        End the public tunnel' \
		'' \
		'Local LLM (used by make prepare and make shorten):' \
		'  make llm                Start the LLM used by make prepare / make shorten (detached tmux)' \
		'  make llm-stop           Kill the LLM tmux session' \
		'  make shorten            Rewrite texts over MAX_WORDS words via the LLM; resumes interrupted runs' \
		'' \
		'Development:' \
		'  make check              Run lint, formatting checks, and tests' \
		'  make clean              Remove caches and local build output'

setup:
	uv sync

llm:
	@if curl -sf -m 2 $(MEDLINER_LLM_URL)/health >/dev/null 2>&1; then \
		echo "llm: already healthy at $(MEDLINER_LLM_URL)"; \
	else \
		tmux new-session -d -s $(LLM_TMUX_SESSION) 'cd $(MODELS_DIR) && make medliner' && \
		echo "llm: started detached tmux session $(LLM_TMUX_SESSION); waiting for $(MEDLINER_LLM_URL)"; \
		for i in $$(seq 1 90); do \
			if curl -sf -m 2 $(MEDLINER_LLM_URL)/health >/dev/null 2>&1; then \
				echo "llm: healthy at $(MEDLINER_LLM_URL)"; exit 0; \
			fi; \
			sleep 2; \
		done; \
		echo "llm: server did not become healthy; check 'tmux attach -t $(LLM_TMUX_SESSION)'" >&2; exit 1; \
	fi

llm-stop:
	@tmux kill-session -t $(LLM_TMUX_SESSION) 2>/dev/null \
		&& echo "llm: tmux session $(LLM_TMUX_SESSION) stopped" \
		|| echo "llm: no tmux session named $(LLM_TMUX_SESSION)"

shorten:
	uv run medliner shorten $(if $(LIMIT),--limit $(LIMIT),) $(if $(MAX_WORDS),--max-words $(MAX_WORDS),)

prepare:
	uv run medliner prepare

annotate:
	uv run medliner label-studio

stop:
	uv run medliner label-studio-stop

export:
	uv run medliner label-studio-export

# Publish Label Studio on a random public URL without a Cloudflare account, DNS record, or token.
# One shell block (like `llm`) so the already-running guard exits the whole recipe, not one line.
tunnel:
	@command -v cloudflared >/dev/null 2>&1 || { \
		echo "tunnel: cloudflared is not on PATH (install it first, e.g. 'nix profile install nixpkgs#cloudflared')" >&2; \
		exit 1; }
	@command -v tmux >/dev/null 2>&1 || { echo "tunnel: tmux is not on PATH" >&2; exit 1; }
	@set -e; \
	ours() { ps -o args= -p "$$1" 2>/dev/null | grep -qF -- "--logfile $(TUNNEL_LOG)"; }; \
	alive() { case "$$(ps -o state= -p "$$1" 2>/dev/null | tr -d ' ')" in ''|Z*) return 1;; *) return 0;; esac; }; \
	mkdir -p "$(TUNNEL_DIR)"; \
	pane=$$(tmux list-panes -t "$(TUNNEL_TMUX_SESSION)" -F '#{pane_pid}' 2>/dev/null | head -1); \
	if [ -n "$$pane" ]; then \
		if ! ours "$$pane"; then \
			echo "tunnel: tmux session $(TUNNEL_TMUX_SESSION) runs something else; stop it or set TUNNEL_TMUX_SESSION" >&2; exit 1; \
		fi; \
		if ! ps -o args= -p "$$pane" | grep -qF -- "--url $(TUNNEL_ORIGIN)"; then \
			echo "tunnel: the live tunnel serves a different origin than $(TUNNEL_ORIGIN); run 'make tunnel-stop' first" >&2; exit 1; \
		fi; \
		url=$$(grep -Eo '$(TUNNEL_URL_PATTERN)' "$(TUNNEL_LOG)" 2>/dev/null | tail -1); \
		echo "tunnel: already running in tmux session $(TUNNEL_TMUX_SESSION)"; \
		if [ -n "$$url" ]; then printf '%s\n' "$$url" > "$(TUNNEL_URL_FILE)"; echo "tunnel: public URL $$url"; \
		else echo "tunnel: no public URL in $(TUNNEL_LOG) yet"; fi; \
		exit 0; \
	fi; \
	rm -f "$(TUNNEL_LOG)" "$(TUNNEL_URL_FILE)" "$(TUNNEL_PID_FILE)"; \
	curl -sf -m 5 "$(TUNNEL_ORIGIN)/health" >/dev/null 2>&1 || \
		echo "tunnel: warning: Label Studio is not answering at $(TUNNEL_ORIGIN) yet; the public URL stays a 502 until 'make annotate' finishes" >&2; \
	tmux new-session -d -s "$(TUNNEL_TMUX_SESSION)" \
		'cloudflared tunnel --no-autoupdate --url "$(TUNNEL_ORIGIN)" --logfile "$(TUNNEL_LOG)"' || { \
		echo "tunnel: could not start tmux session $(TUNNEL_TMUX_SESSION) (a concurrent 'make tunnel' may have won; rerun to read its URL)" >&2; exit 1; }; \
	pid=$$(tmux list-panes -t "$(TUNNEL_TMUX_SESSION)" -F '#{pane_pid}' | head -1); \
	printf '%s\n' "$$pid" > "$(TUNNEL_PID_FILE)"; \
	echo "tunnel: started a quick tunnel to $(TUNNEL_ORIGIN) in tmux session $(TUNNEL_TMUX_SESSION) (cloudflared PID $$pid); waiting for the public URL"; \
	for i in $$(seq 1 30); do \
		url=$$(grep -Eo '$(TUNNEL_URL_PATTERN)' "$(TUNNEL_LOG)" 2>/dev/null | tail -1); \
		if [ -n "$$url" ]; then \
			printf '%s\n' "$$url" > "$(TUNNEL_URL_FILE)"; \
			echo "tunnel: public URL $$url"; \
			echo "tunnel: it can take a few seconds to become reachable, and it stays public on the internet until 'make tunnel-stop'; log $(TUNNEL_LOG)"; \
			exit 0; \
		fi; \
		alive "$$pid" || { \
			echo "tunnel: cloudflared exited before publishing a URL; see $(TUNNEL_LOG)" >&2; exit 1; }; \
		sleep 1; \
	done; \
	echo "tunnel: no public URL after 30s; cloudflared ($$pid) is still up, so re-run 'make tunnel' to re-read it, or check $(TUNNEL_LOG)" >&2; \
	exit 1

# `tmux kill-session` alone is not enough: cloudflared can survive the pane teardown, so signal the
# PID (TERM, then KILL) and verify it is gone before claiming the URL is dead. Only a process whose
# argv carries this repo's `--logfile` path is ever signalled, so a stale PID file or another
# project's tunnel cannot be killed by mistake.
tunnel-stop:
	@set -e; \
	ours() { ps -o args= -p "$$1" 2>/dev/null | grep -qF -- "--logfile $(TUNNEL_LOG)"; }; \
	alive() { case "$$(ps -o state= -p "$$1" 2>/dev/null | tr -d ' ')" in ''|Z*) return 1;; *) return 0;; esac; }; \
	url=$$(cat "$(TUNNEL_URL_FILE)" 2>/dev/null || true); \
	pane=$$(tmux list-panes -t "$(TUNNEL_TMUX_SESSION)" -F '#{pane_pid}' 2>/dev/null | head -1); \
	recorded=$$(cat "$(TUNNEL_PID_FILE)" 2>/dev/null || true); \
	pid=""; \
	if [ -n "$$pane" ] && ours "$$pane"; then pid=$$pane; \
	elif [ -n "$$recorded" ] && ours "$$recorded"; then pid=$$recorded; \
	elif [ -n "$$pane" ] && alive "$$pane"; then \
		echo "tunnel: tmux session $(TUNNEL_TMUX_SESSION) is not running this repo's tunnel; leaving it alone" >&2; exit 1; \
	fi; \
	if [ -z "$$pid" ] || ! alive "$$pid"; then \
		echo "tunnel: no tunnel running for $(TUNNEL_ORIGIN)"; \
		rm -f "$(TUNNEL_URL_FILE)" "$(TUNNEL_PID_FILE)"; exit 0; \
	fi; \
	kill "$$pid" 2>/dev/null || true; \
	for i in $$(seq 1 10); do alive "$$pid" || break; sleep 0.5; done; \
	if alive "$$pid"; then kill -9 "$$pid" 2>/dev/null || true; sleep 0.5; fi; \
	if [ -n "$$pane" ] && ours "$$pane"; then tmux kill-session -t "$(TUNNEL_TMUX_SESSION)" 2>/dev/null || true; fi; \
	rm -f "$(TUNNEL_URL_FILE)" "$(TUNNEL_PID_FILE)"; \
	if alive "$$pid"; then \
		echo "tunnel: cloudflared ($$pid) is STILL RUNNING, so the public URL$${url:+ $$url} is still live; kill it manually" >&2; exit 1; \
	fi; \
	if [ -n "$$url" ]; then echo "tunnel: stopped; the public URL $$url is dead"; \
	else echo "tunnel: stopped (no public URL had been recorded)"; fi

# Run Python lint checks.
lint:
	uv run ruff check .

# Format Python code.
fmt:
	uv run ruff format .

# Check Python formatting without changing files.
fmt-check:
	uv run ruff format --check .

test:
	uv run pytest

check: lint fmt-check test

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov coverage.xml
	find src tests -name __pycache__ -type d -prune -exec rm -rf {} +
