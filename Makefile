SHELL := /usr/bin/env bash
.DEFAULT_GOAL := help

export MEDLINER_RAW_CANDIDATES ?= $(CURDIR)/data/label-studio/candidates.jsonl
export MEDLINER_LABEL_STUDIO_EXPORT ?= $(CURDIR)/data/label-studio/reviewed.json
export MEDLINER_WORKDIR ?= $(CURDIR)/data/materialized
export MEDLINER_TRAIN_CONFIG ?= $(CURDIR)/configs/train-small.yaml
export MEDLINER_DAKP_ROOT ?= $(CURDIR)/../DAKP
export MEDLINER_LABEL_STUDIO_PORT ?= 9030
export MEDLINER_LABEL_STUDIO_IMAGE ?= docker.io/heartexlabs/label-studio:latest
export MEDLINER_LABEL_STUDIO_USERNAME ?= medliner@localhost
export MEDLINER_LABEL_STUDIO_PASSWORD ?= medliner-local
export DAGSTER_HOME ?= $(CURDIR)/.dagster

# Triton locates libcuda through /sbin/ldconfig; without a loader cache that call fails inside
# the backward pass. An empty value is ignored by Triton, so this is safe on normal systems.
export TRITON_LIBCUDA_PATH ?= $(shell test -x /sbin/ldconfig || for d in /run/opengl-driver/lib /usr/lib64 /usr/lib/x86_64-linux-gnu /usr/lib; do test -e $$d/libcuda.so.1 && echo $$d && break; done)

.PHONY: help UP up sync test coverage lint format validate check clean env dagster-home label-studio-stop

help:
	@printf '%s\n' \
		'MEDliNER targets:' \
		'  make UP                 Start the local Dagster deployment/UI (all pipeline stages run here)' \
		'  make label-studio-stop  Remove the podman Label Studio container started by the DAG' \
		'  make sync               Install or update the uv environment' \
		'  make check              Run tests, lint, formatting, and Dagster definition checks' \
		'  make validate           Validate the Dagster definitions without starting a server' \
		'  make coverage           Run the test suite with a coverage report' \
		'  make clean              Remove caches and local build output' \
		'  make env                Print the resolved pipeline environment'

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

# Stops the annotation server started by the label_studio_server asset. The Label Studio
# database lives in $MEDLINER_WORKDIR/label-studio/server-data, so annotations survive this.
label-studio-stop:
	podman rm -f medliner-label-studio

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

# Print the resolved environment the pipeline stages will run under.
env:
	@printf '%s\n' \
		'MEDLINER_RAW_CANDIDATES=$(MEDLINER_RAW_CANDIDATES)' \
		'MEDLINER_LABEL_STUDIO_EXPORT=$(MEDLINER_LABEL_STUDIO_EXPORT)' \
		'MEDLINER_WORKDIR=$(MEDLINER_WORKDIR)' \
		'MEDLINER_TRAIN_CONFIG=$(MEDLINER_TRAIN_CONFIG)' \
		'MEDLINER_DAKP_ROOT=$(MEDLINER_DAKP_ROOT)' \
		'MEDLINER_LABEL_STUDIO_PORT=$(MEDLINER_LABEL_STUDIO_PORT)' \
		'MEDLINER_LABEL_STUDIO_IMAGE=$(MEDLINER_LABEL_STUDIO_IMAGE)' \
		'MEDLINER_LABEL_STUDIO_USERNAME=$(MEDLINER_LABEL_STUDIO_USERNAME)' \
		'DAGSTER_HOME=$(DAGSTER_HOME)' \
		'TRITON_LIBCUDA_PATH=$(TRITON_LIBCUDA_PATH)'
