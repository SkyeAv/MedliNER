SHELL := /usr/bin/env bash
.DEFAULT_GOAL := help

export MEDLINER_LABEL_STUDIO_EXPORT ?= $(CURDIR)/data/label-studio/reviewed.json
export MEDLINER_WORKDIR ?= $(CURDIR)/data/materialized
export MEDLINER_TRAIN_CONFIG ?= $(CURDIR)/configs/train-small.yaml
export MEDLINER_DAKP_ROOT ?= $(CURDIR)/../DAKP
export DAGSTER_HOME ?= $(CURDIR)/.dagster

# Triton locates libcuda through /sbin/ldconfig; without a loader cache that call fails inside
# the backward pass. An empty value is ignored by Triton, so this is safe on normal systems.
export TRITON_LIBCUDA_PATH ?= $(shell test -x /sbin/ldconfig || for d in /run/opengl-driver/lib /usr/lib64 /usr/lib/x86_64-linux-gnu /usr/lib; do test -e $$d/libcuda.so.1 && echo $$d && break; done)

.PHONY: help UP up sync test coverage lint format validate check clean env dagster-home normalize split smoke train evaluate bundle

help:
	@printf '%s\n' \
		'MEDliNER targets:' \
		'  make UP       Start the local Dagster deployment/UI' \
		'  make sync     Install or update the uv environment' \
		'  make check    Run tests, lint, formatting, and Dagster definition checks' \
		'  make validate Validate the Dagster definitions without starting a server' \
		'  make coverage Run the test suite with a coverage report' \
		'  make clean    Remove caches and local build output' \
		'  make env      Print the resolved pipeline environment' \
		'  make normalize Normalize the reviewed Label Studio export' \
		'  make split    Create deterministic train/validation/test splits' \
		'  make smoke    Run a one-step GPU training smoke test' \
		'  make train    Run configured small-GLiNER training' \
		'  make evaluate Evaluate the final checkpoint' \
		'  make bundle   Build the uploadable artifact bundle'

# Seed $DAGSTER_HOME from the committed instance config. Without a dagster.yaml there, every
# Dagster command warns and falls back to undeclared defaults.
dagster-home:
	@mkdir -p "$(DAGSTER_HOME)"
	@test -f "$(DAGSTER_HOME)/dagster.yaml" || cp "$(CURDIR)/configs/dagster.yaml" "$(DAGSTER_HOME)/dagster.yaml"

# Uppercase is the deployment target requested for this repository; lowercase is a convenience alias.
UP: dagster-home
	@echo "Starting Dagster at http://localhost:3000"
	@echo "Label Studio export: $(MEDLINER_LABEL_STUDIO_EXPORT)"
	@echo "Dagster home: $(DAGSTER_HOME)"
	uv run dagster dev -m medliner.dagster_defs

up: UP

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

# Loads the asset graph in-process; catches a broken definition without starting a server.
validate: dagster-home
	uv run dagster definitions validate -m medliner.dagster_defs

check: test lint format validate

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov coverage.xml
	find src tests -name __pycache__ -type d -prune -exec rm -rf {} +

normalize:
	uv run medliner normalize "$(MEDLINER_LABEL_STUDIO_EXPORT)" "$(MEDLINER_WORKDIR)/normalized/examples.jsonl"

split: normalize
	uv run medliner split "$(MEDLINER_WORKDIR)/normalized/examples.jsonl" "$(MEDLINER_WORKDIR)/splits"

smoke: split
	uv run medliner train "$(MEDLINER_WORKDIR)/splits" "$(MEDLINER_WORKDIR)/smoke-training" --config "$(MEDLINER_TRAIN_CONFIG)" --smoke

train: split
	uv run medliner train "$(MEDLINER_WORKDIR)/splits" "$(MEDLINER_WORKDIR)/training" --config "$(MEDLINER_TRAIN_CONFIG)"

evaluate: train
	uv run medliner evaluate "$(MEDLINER_WORKDIR)/training/final" "$(MEDLINER_WORKDIR)/splits" "$(MEDLINER_WORKDIR)/evaluation/report.json"

bundle: evaluate
	uv run medliner bundle "$(MEDLINER_WORKDIR)/training/final" "$(MEDLINER_WORKDIR)/evaluation/report.json" "$(MEDLINER_WORKDIR)/normalized/examples.jsonl" "$(MEDLINER_WORKDIR)/splits" "$(MEDLINER_WORKDIR)/bundle"

# Print the resolved environment the pipeline stages will run under.
env:
	@printf '%s\n' \
		'MEDLINER_LABEL_STUDIO_EXPORT=$(MEDLINER_LABEL_STUDIO_EXPORT)' \
		'MEDLINER_WORKDIR=$(MEDLINER_WORKDIR)' \
		'MEDLINER_TRAIN_CONFIG=$(MEDLINER_TRAIN_CONFIG)' \
		'MEDLINER_DAKP_ROOT=$(MEDLINER_DAKP_ROOT)' \
		'DAGSTER_HOME=$(DAGSTER_HOME)' \
		'TRITON_LIBCUDA_PATH=$(TRITON_LIBCUDA_PATH)'
