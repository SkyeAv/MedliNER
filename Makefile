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
export MEDLINER_LABEL_STUDIO_USERNAME ?= medliner@localhost
export MEDLINER_LABEL_STUDIO_PASSWORD ?= medliner-local
export MEDLINER_LABEL_STUDIO_HOST ?= 127.0.0.1
export MEDLINER_ONBOARDING_CONFIG ?= $(CURDIR)/configs/onboarding.json
export MEDLINER_ONBOARDING_EXPORT ?= $(CURDIR)/data/materialized/onboarding/export.json
# Production dataset acceptance requires a passing onboarding record. Set to 0 only for legacy runs.
MEDLINER_ONBOARDING_REQUIRED ?= 1
export MEDLINER_LLM_URL ?= http://127.0.0.1:8080
MODELS_DIR ?= $(CURDIR)/models
LLM_TMUX_SESSION ?= medliner-llm

# Triton locates libcuda through /sbin/ldconfig; without a loader cache that call fails inside
# the backward pass. An empty value is ignored by Triton, so this is safe on normal systems.
export TRITON_LIBCUDA_PATH ?= $(shell test -x /sbin/ldconfig || for d in /run/opengl-driver/lib /usr/lib64 /usr/lib/x86_64-linux-gnu /usr/lib; do test -e $$d/libcuda.so.1 && echo $$d && break; done)

.PHONY: help setup llm llm-stop data shorten onboarding onboarding-start onboarding-export onboarding-evaluate onboarding-status onboarding-promote annotate annotate-stop export train check clean env

help:
	@printf '%s\n' \
		'MedliNER pipeline (run in order):' \
		'  make setup              Install or update the uv environment' \
		'  make data               Build the sampled 5K edge-case import file and attach GLiNER suggestions' \
		'  make onboarding        Create the answer-free Onboarding project and private 10-case test bank' \
		'  make onboarding-start  Assign 4 deterministic quiz tasks (USER=<username>)' \
		'  make onboarding-export Download the Onboarding project export' \
		'  make onboarding-evaluate Score a quiz attempt (USER=<username>)' \
		'  make onboarding-status Show attempts and passing users' \
		'  make onboarding-promote Promote a passing annotator (USER=<username>)' \
		'  make annotate           Start the production Label Studio server with tasks imported (REIMPORT=1 replaces tasks,' \
		'                          PRELABEL=1 pre-fills model suggestions)' \
		'  [complete Onboarding before annotating production tasks]' \
		'  make export             Download the reviewed production annotations from the running server' \
		'  make train              dataset → splits → train → evaluate → bundle in one go (SMOKE=1 for the one-step GPU check)' \
		'' \
		'Local LLM (Ornith-1.0-9B via llama-server, used only by make shorten):' \
		'  make llm                Start the LLM in a detached tmux session and wait for health' \
		'  make llm-stop           Kill the LLM tmux session' \
		'  make shorten            Rewrite over-long candidate texts through the LLM (LIMIT=8 for a trial run)' \
		'' \
		'Other targets:' \
		'  make annotate-stop      Remove the Label Studio container (annotations survive in its data volume)' \
		'  make check              Run tests, lint, and formatting checks' \
		'  make clean              Remove caches and local build output' \
		'  make env                Print the resolved pipeline environment'

setup:
	uv sync

# --- Local LLM (detached tmux; the server defines -np 2 -cb --kv-unified) ---

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

# --- Before Label Studio ---

data:
	uv run medliner candidates
	uv run medliner prelabel

# Opt-in LLM rewrite of over-long texts; never run implicitly. Point MEDLINER_RAW_CANDIDATES
# at the shortened file afterwards if the manifest looks right.
shorten:
	uv run medliner shorten $(if $(LIMIT),--limit $(LIMIT),)

# --- Label Studio ---

onboarding:
	uv run medliner onboarding $(if $(REIMPORT),--reimport,) $(foreach A,$(ANNOTATORS),--annotator $(A))

onboarding-start:
	@test -n "$(ANNOTATOR)" || test "$(origin USER)" = "command line" || (echo 'set ANNOTATOR=<Label Studio username> (or pass USER= explicitly)' >&2; exit 1)
	uv run medliner onboarding-start --user $(if $(ANNOTATOR),$(ANNOTATOR),$(USER))

onboarding-export:
	uv run medliner onboarding-export $(if $(OUTPUT),--output $(OUTPUT),)

onboarding-evaluate:
	@test -n "$(ANNOTATOR)" || test "$(origin USER)" = "command line" || (echo 'set ANNOTATOR=<Label Studio username> (or pass USER= explicitly)' >&2; exit 1)
	uv run medliner onboarding-evaluate --user $(if $(ANNOTATOR),$(ANNOTATOR),$(USER)) $(if $(ATTEMPT),--attempt $(ATTEMPT),) $(if $(INPUT),--export $(INPUT),)

onboarding-status:
	uv run medliner onboarding-status $(if $(ANNOTATOR),--user $(ANNOTATOR),$(if $(filter command line,$(origin USER)),--user $(USER),))

onboarding-promote:
	@test -n "$(ANNOTATOR)" || test "$(origin USER)" = "command line" || (echo 'set ANNOTATOR=<Label Studio username> (or pass USER= explicitly)' >&2; exit 1)
	uv run medliner onboarding-promote --user $(if $(ANNOTATOR),$(ANNOTATOR),$(USER)) $(if $(ATTEMPT),--attempt $(ATTEMPT),)

annotate:
	uv run medliner label-studio $(if $(INPUT),--input $(INPUT),) $(if $(REIMPORT),--reimport,) \
	$(if $(WARMUP),--warmup,) $(if $(PRELABEL),--prelabel,) $(foreach A,$(ANNOTATORS),--annotator $(A))

# The Label Studio database lives in $MEDLINER_WORKDIR/label-studio/server-data, so
# annotations survive this.
annotate-stop:
	uv run medliner label-studio-stop

export:
	uv run medliner label-studio-export $(if $(OUTPUT),--output $(OUTPUT),)

# --- After Label Studio ---

train:
	MEDLINER_ONBOARDING_REQUIRED=$(MEDLINER_ONBOARDING_REQUIRED) uv run medliner pipeline $(if $(SMOKE),--smoke,)

# --- Development ---

check:
	uv run pytest -q
	uv run ruff check src tests
	uv run ruff format --check src tests

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov coverage.xml
	find src tests -name __pycache__ -type d -prune -exec rm -rf {} +

# Print the resolved environment the pipeline stages will run under.
env:
	@printf '%s\n' \
		'MEDLINER_RAW_CANDIDATES=$(MEDLINER_RAW_CANDIDATES)' \
		'MEDLINER_BENCHMARK=$(MEDLINER_BENCHMARK)' \
		'MEDLINER_EXPORT_BUNDLE=$(MEDLINER_EXPORT_BUNDLE)' \
		'MEDLINER_LABEL_STUDIO_EXPORT=$(MEDLINER_LABEL_STUDIO_EXPORT)' \
		'MEDLINER_WORKDIR=$(MEDLINER_WORKDIR)' \
		'MEDLINER_TRAIN_CONFIG=$(MEDLINER_TRAIN_CONFIG)' \
		'MEDLINER_PRELABEL_MODEL=$(MEDLINER_PRELABEL_MODEL)' \
		'MEDLINER_PRELABEL_THRESHOLD=$(MEDLINER_PRELABEL_THRESHOLD)' \
		'MEDLINER_LABEL_STUDIO_PORT=$(MEDLINER_LABEL_STUDIO_PORT)' \
		'MEDLINER_LABEL_STUDIO_IMAGE=$(MEDLINER_LABEL_STUDIO_IMAGE)' \
		'MEDLINER_LABEL_STUDIO_HOST=$(MEDLINER_LABEL_STUDIO_HOST)' \
		'MEDLINER_ONBOARDING_CONFIG=$(MEDLINER_ONBOARDING_CONFIG)' \
		'MEDLINER_ONBOARDING_EXPORT=$(MEDLINER_ONBOARDING_EXPORT)' \
		'MEDLINER_ONBOARDING_REQUIRED=$(MEDLINER_ONBOARDING_REQUIRED)' \
		'MEDLINER_LABEL_STUDIO_USERNAME=$(MEDLINER_LABEL_STUDIO_USERNAME)' \
		'MEDLINER_LLM_URL=$(MEDLINER_LLM_URL)' \
		'MODELS_DIR=$(MODELS_DIR)' \
		'TRITON_LIBCUDA_PATH=$(TRITON_LIBCUDA_PATH)'
