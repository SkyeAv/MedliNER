"""Shared test setup: every test runs with a hermetic ``MEDLINER_*`` environment.

``make test`` exports the repo's real pipeline settings into pytest (``MEDLINER_WORKDIR``,
``MEDLINER_SHORTEN_CACHE``, ``MEDLINER_LABEL_STUDIO_EXPORT``, ...), and a leaked value silently
changes what a test exercises. Two concrete leaks this prevents: the exported shorten cache made
``medliner shorten`` answer from the developer's real sqlite file, so the stub LLM server was never
called and it also stored the stub replies there; the exported workdir pointed outside the test's
temporary directory, so a test could read or overwrite real pipeline artifacts.

Tests that need a setting still set it themselves with ``monkeypatch.setenv``, which runs after
this fixture, so nothing has to change in the individual tests.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def hermetic_medliner_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Delete every ambient ``MEDLINER_*`` variable for the duration of one test."""
    for name in tuple(os.environ):
        if name.startswith("MEDLINER_"):
            monkeypatch.delenv(name, raising=False)
