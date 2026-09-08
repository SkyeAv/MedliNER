from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _isolate_medliner_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run every test against a clean MEDLINER_* environment.

    The Makefile ``export``s the whole pipeline configuration, so ``make check`` runs pytest
    with ``MEDLINER_WORKDIR``, ``MEDLINER_SHORTEN_CACHE``, ``MEDLINER_ONBOARDING_EXPORT`` and
    friends already pointing at the repo's real ``data/`` directory. Tests that set only the
    variables they care about then silently inherit the rest, so the suite passed from a bare
    shell and failed under ``make`` — and could read or write real pipeline artifacts. Tests
    opt in to the values they need via ``monkeypatch.setenv``.
    """
    for name in [key for key in os.environ if key.startswith("MEDLINER_")]:
        monkeypatch.delenv(name, raising=False)
