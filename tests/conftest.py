"""Pytest bootstrap — fail loudly if the test run is using the wrong interpreter.

Most of the suite imports cleanly under almost any Python that has the heavy deps
(langchain, pydantic, …), so a run launched with a *system* interpreter that is
missing a declared dependency can pass the majority of tests while silently being
in the wrong environment. That once masked a real problem (a run under the system
`python3`, which lacks ``textual``, "passed" 329/330 and made a single failure look
pre-existing).

This guard converts that silent, misleading state into an immediate, explanatory
abort: if a representative import-time dependency is not importable, the whole
session stops with a message pointing at the project virtualenv. It is keyed on
*dependency presence* (not a hardcoded path), so it stays correct for any
contributor's venv.
"""

from __future__ import annotations

import importlib.util
import sys

import pytest

# Representative import-time dependencies that a fully-provisioned project env has
# but a bare/system interpreter typically does not. ``textual`` is the canonical
# example: it is a declared core dependency, imported at module load by
# ``bamboo.scripts.review_drafts`` (so the CLI import path needs it), yet absent on
# a stock framework/system Python — exactly the gap that caused the silent failure.
_REQUIRED_IMPORTS = ("textual",)


@pytest.fixture(autouse=True)
def _no_panda_in_field_tests(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """The field tests never reach PanDA: a part their fakes do not hand in fails here, not on the network.

    ``fetch.Parts`` defaults to panda_client's calls, so a route a test's fake does not cover (a default way of
    reading that changed under it) would call the production servers (2026-11-19: making the endpoints the default
    sent four tests' dataset reads to aipanda09x).  A test that stands in for panda_client's transport patches it
    itself, after this.
    """
    if not request.module.__name__.rpartition(".")[2].startswith("test_fieldtest"):
        return
    from bamboo.utils import panda_client  # noqa: PLC0415

    def refuse(method: str, endpoint: str, *args: object, **kwargs: object) -> None:
        raise AssertionError(f"a field test reached PanDA: {method} {endpoint}")

    monkeypatch.setattr(panda_client, "_call_api_full", refuse)
    monkeypatch.setattr(panda_client, "_call_api", refuse)


def pytest_configure(config: pytest.Config) -> None:
    missing = [m for m in _REQUIRED_IMPORTS if importlib.util.find_spec(m) is None]
    if missing:
        raise pytest.UsageError(
            "Wrong Python interpreter for this project: missing declared "
            f"dependenc{'y' if len(missing) == 1 else 'ies'} {', '.join(missing)} "
            f"under {sys.executable!r}.\n"
            "You are almost certainly not in the project virtualenv. Run the tests "
            "through the project venv's interpreter (e.g. activate it, or use its "
            "`bin/python -m pytest`) — do not fall back to the system `python3`. "
            "See CLAUDE.md / CLAUDE.local.md."
        )
