SHELL := /usr/bin/env bash
.DEFAULT_GOAL := help

# Repo-relative defaults keep a fresh checkout portable; `.envrc.local` can point at a ready DAKP export.
export MEDLINER_RAW_CANDIDATES ?= $(CURDIR)/data/label-studio/candidates.ndjson
export MEDLINER_BENCHMARK ?= $(CURDIR)/data/materialized/ingested/ner_gold.json
export MEDLINER_EXPORT_BUNDLE ?= $(CURDIR)/data/dakp-export
export MEDLINER_LABEL_STUDIO_EXPORT ?= $(CURDIR)/data/label-studio/reviewed.json
export MEDLINER_WORKDIR ?= $(CURDIR)/data/materialized
export MEDLINER_TRAIN_CONFIG ?= $(CURDIR)/configs/train-small.yaml
# Pre-labeling checkpoint and score floor. Same values the sibling DAKP pipeline mines with.
export MEDLINER_PRELABEL_MODEL ?= gliner-community/gliner_large-v2.5
export MEDLINER_PRELABEL_THRESHOLD ?= 0.35
export MEDLINER_LABEL_STUDIO_PORT ?= 9030
export MEDLINER_LABEL_STUDIO_IMAGE ?= docker.io/heartexlabs/label-studio:latest
export MEDLINER_LABEL_STUDIO_HOST ?= 127.0.0.1
# Comma-separated user:password accounts ensured at start; onboarding assigns everyone a quiz.
export MEDLINER_LABEL_STUDIO_ANNOTATORS ?=
export MEDLINER_ONBOARDING_CONFIG ?= $(CURDIR)/configs/onboarding.json
export MEDLINER_ONBOARDING_EXPORT ?= $(CURDIR)/data/materialized/onboarding/export.json
export MEDLINER_LLM_URL ?= http://127.0.0.1:8080
# Model checkout with the `medliner` llama-server target; falls back to ~/Desktop/MODELS.
MODELS_DIR ?= $(if $(wildcard $(CURDIR)/models/Makefile),$(CURDIR)/models,$(HOME)/Desktop/MODELS)
LLM_TMUX_SESSION ?= medliner-llm
# Shorten stage: word threshold (≈3-4 short sentences), parallel requests, reply cache.
export MEDLINER_SHORTEN_MAX_WORDS ?= 48
export MEDLINER_SHORTEN_WORKERS ?= 4
export MEDLINER_SHORTEN_CACHE ?= $(CURDIR)/data/materialized/shorten-cache.sqlite3

# Triton locates libcuda through /sbin/ldconfig; without a loader cache that call fails inside
# the backward pass. An empty value is ignored by Triton, so this is safe on normal systems.
export TRITON_LIBCUDA_PATH ?= $(shell test -x /sbin/ldconfig || for d in /run/opengl-driver/lib /usr/lib64 /usr/lib/x86_64-linux-gnu /usr/lib; do test -e $$d/libcuda.so.1 && echo $$d && break; done)

.PHONY: help setup llm llm-stop shorten prepare onboarding onboarding-promote annotate stop export train check clean

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
		'' \
		'Optional Onboarding (presentation mode; quizzes assigned to every account at once):' \
		'  make onboarding         Create the Onboarding project and assign a quiz to every annotator' \
		'  make onboarding-promote Export the quiz, score every attempt, promote everyone passing' \
		'' \
		'Training (everything after Label Studio, one command):' \
		'  make train              dataset → splits → train → evaluate → bundle' \
		'' \
		'Local LLM (used only by make shorten):' \
		'  make llm                Start the LLM used by make prepare / make shorten (detached tmux)' \
		'  make llm-stop           Kill the LLM tmux session' \
		'  make shorten            Rewrite texts over MAX_WORDS words via the LLM; resumes interrupted runs' \
		'' \
		'Development:' \
		'  make check              Run tests, lint, and formatting checks' \
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

onboarding:
	uv run medliner onboarding

onboarding-promote:
	uv run medliner onboarding-promote

annotate:
	uv run medliner label-studio

stop:
	uv run medliner label-studio-stop

export:
	uv run medliner label-studio-export

train:
	uv run medliner pipeline

check:
	uv run pytest -q && uv run ruff check src tests && uv run ruff format --check src tests

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov coverage.xml
	find src tests -name __pycache__ -type d -prune -exec rm -rf {} +
