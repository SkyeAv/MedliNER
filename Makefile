SHELL := /usr/bin/env bash
.DEFAULT_GOAL := help

export MEDLINER_LABEL_STUDIO_EXPORT ?= $(CURDIR)/data/label-studio/reviewed.json
export MEDLINER_WORKDIR ?= $(CURDIR)/data/materialized
export MEDLINER_TRAIN_CONFIG ?= $(CURDIR)/configs/train-small.yaml
export DAGSTER_HOME ?= $(CURDIR)/.dagster

.PHONY: help UP up sync test lint format check normalize split smoke train evaluate bundle

help:
	@printf '%s\n' \
		'MEDliNER targets:' \
		'  make UP       Start the local Dagster deployment/UI' \
		'  make sync     Install or update the uv environment' \
		'  make check    Run tests, lint, and formatting checks' \
		'  make normalize Normalize the reviewed Label Studio export' \
		'  make split    Create deterministic train/validation/test splits' \
		'  make smoke    Run a one-step GPU training smoke test' \
		'  make train    Run configured small-GLiNER training' \
		'  make evaluate Evaluate the final checkpoint' \
		'  make bundle   Build the uploadable artifact bundle'

# Uppercase is the deployment target requested for this repository; lowercase is a convenience alias.
UP:
	@mkdir -p "$(DAGSTER_HOME)"
	@echo "Starting Dagster at http://localhost:3000"
	@echo "Label Studio export: $(MEDLINER_LABEL_STUDIO_EXPORT)"
	@echo "Dagster home: $(DAGSTER_HOME)"
	uv run dagster dev -m medliner.dagster_defs

up: UP

sync:
	uv sync

test:
	uv run pytest -q

lint:
	uv run ruff check src tests

format:
	uv run ruff format --check src tests

check: test lint format

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
