from __future__ import annotations

import pytest


def test_definitions_expose_minimal_asset_graph():
    pytest.importorskip("dagster")
    from medliner.dagster_defs import definitions

    defs = definitions()
    keys = {key.to_user_string() for key in defs.resolve_all_asset_keys()}
    assert {
        "label_studio_export",
        "normalized_dataset",
        "frozen_splits",
        "training_run",
        "evaluation_report",
        "export_bundle",
    } <= keys
    assert not any("schedule" in repr(item).lower() for item in (getattr(defs, "schedules", None) or []))
