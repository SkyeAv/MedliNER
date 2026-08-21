SHELL := /usr/bin/env bash
.DEFAULT_GOAL := help

export MEDLINER_EXPORT_BUNDLE ?= $(CURDIR)/data/dakp-export
export MEDLINER_BENCHMARK ?= $(CURDIR)/data/materialized/ingested/ner_gold.json
export MEDLINER_RAW_CANDIDATES ?= $(CURDIR)/data/label-studio/candidates.jsonl
export MEDLINER_LABEL_STUDIO_EXPORT ?= $(CURDIR)/data/label-studio/reviewed.json
export MEDLINER_WORKDIR ?= $(CURDIR)/data/materialized
export MEDLINER_TRAIN_CONFIG ?= $(CURDIR)/configs/train-small.yaml
export MEDLINER_LABEL_STUDIO_PORT ?= 9030
export MEDLINER_LABEL_STUDIO_IMAGE ?= docker.io/heartexlabs/label-studio:latest
export MEDLINER_LABEL_STUDIO_USERNAME ?= medliner@localhost
export MEDLINER_LABEL_STUDIO_PASSWORD ?= medliner-local

# Triton locates libcuda through /sbin/ldconfig; without a loader cache that call fails inside
# the backward pass. An empty value is ignored by Triton, so this is safe on normal systems.
export TRITON_LIBCUDA_PATH ?= $(shell test -x /sbin/ldconfig || for d in /run/opengl-driver/lib /usr/lib64 /usr/lib/x86_64-linux-gnu /usr/lib; do test -e $$d/libcuda.so.1 && echo $$d && break; done)

.PHONY: help ingest candidates label-studio label-studio-stop pipeline sync test coverage lint format check clean env

help:
	@printf '%s\n' \
		'MEDliNER pipeline (run in order):' \
		'  make ingest             Verify + materialize the DAKP export bundle (BUNDLE=<dir> overrides)' \
		'  make candidates         Build the Label Studio import file from raw candidates' \
		'  make label-studio       Start the podman Label Studio server with tasks imported (REIMPORT=1 replaces tasks)' \
		'  [annotate in the browser, export the reviewed JSON manually]' \
		'  make pipeline           dataset → splits → train → evaluate → bundle in one go (SMOKE=1 for the one-step GPU check)' \
		'' \
		'Other targets:' \
		'  make label-studio-stop  Remove the Label Studio container (annotations survive in its data volume)' \
		'  make sync               Install or update the uv environment' \
		'  make check              Run tests, lint, and formatting checks' \
		'  make coverage           Run the test suite with a coverage report' \
		'  make clean              Remove caches and local build output' \
		'  make env                Print the resolved pipeline environment'

# --- Before Label Studio ---

ingest:
	uv run medliner ingest $(if $(BUNDLE),--bundle $(BUNDLE),)

candidates:
	uv run medliner candidates

# --- Label Studio ---

label-studio:
	uv run medliner label-studio $(if $(INPUT),--input $(INPUT),) $(if $(REIMPORT),--reimport,)

# The Label Studio database lives in $MEDLINER_WORKDIR/label-studio/server-data, so
# annotations survive this.
label-studio-stop:
	uv run medliner label-studio-stop

# --- After Label Studio ---

pipeline:
	uv run medliner pipeline $(if $(SMOKE),--smoke,)

# --- Development ---

sync:
	uv sync

test:
	uv run pytest -q

coverage:
	uv run pytest -q --cov=medliner --cov-report=term-missing

lint:
	uv run ruff check src tests

format:
	uv run ruff format --check src tests

check: test lint format

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov coverage.xml
	find src tests -name __pycache__ -type d -prune -exec rm -rf {} +

# Print the resolved environment the pipeline stages will run under.
env:
	@printf '%s\n' \
		'MEDLINER_EXPORT_BUNDLE=$(MEDLINER_EXPORT_BUNDLE)' \
		'MEDLINER_BENCHMARK=$(MEDLINER_BENCHMARK)' \
		'MEDLINER_RAW_CANDIDATES=$(MEDLINER_RAW_CANDIDATES)' \
		'MEDLINER_LABEL_STUDIO_EXPORT=$(MEDLINER_LABEL_STUDIO_EXPORT)' \
		'MEDLINER_WORKDIR=$(MEDLINER_WORKDIR)' \
		'MEDLINER_TRAIN_CONFIG=$(MEDLINER_TRAIN_CONFIG)' \
		'MEDLINER_LABEL_STUDIO_PORT=$(MEDLINER_LABEL_STUDIO_PORT)' \
		'MEDLINER_LABEL_STUDIO_IMAGE=$(MEDLINER_LABEL_STUDIO_IMAGE)' \
		'MEDLINER_LABEL_STUDIO_USERNAME=$(MEDLINER_LABEL_STUDIO_USERNAME)' \
		'TRITON_LIBCUDA_PATH=$(TRITON_LIBCUDA_PATH)'
