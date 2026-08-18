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

.PHONY: help UP up sync test coverage lint format validate check clean env dagster-home

help:
	@printf '%s\n' \
		'MEDliNER targets:' \
		'  make UP       Start the local Dagster deployment/UI (all pipeline stages run here)' \
		'  make sync     Install or update the uv environment' \
		'  make check    Run tests, lint, formatting, and Dagster definition checks' \
		'  make validate Validate the Dagster definitions without starting a server' \
		'  make coverage Run the test suite with a coverage report' \
		'  make clean    Remove caches and local build output' \
		'  make env      Print the resolved pipeline environment'

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

# Print the resolved environment the pipeline stages will run under.
env:
	@printf '%s\n' \
		'MEDLINER_LABEL_STUDIO_EXPORT=$(MEDLINER_LABEL_STUDIO_EXPORT)' \
		'MEDLINER_WORKDIR=$(MEDLINER_WORKDIR)' \
		'MEDLINER_TRAIN_CONFIG=$(MEDLINER_TRAIN_CONFIG)' \
		'MEDLINER_DAKP_ROOT=$(MEDLINER_DAKP_ROOT)' \
		'DAGSTER_HOME=$(DAGSTER_HOME)' \
		'TRITON_LIBCUDA_PATH=$(TRITON_LIBCUDA_PATH)'
