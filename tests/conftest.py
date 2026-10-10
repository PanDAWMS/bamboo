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
import os
import sys

import pytest

# Representative import-time dependencies that a fully-provisioned project env has
# but a bare/system interpreter typically does not. ``textual`` is the canonical
# example: it is a declared core dependency, imported at module load by
# ``bamboo.scripts.review_drafts`` (so the CLI import path needs it), yet absent on
# a stock framework/system Python — exactly the gap that caused the silent failure.
_REQUIRED_IMPORTS = ("textual",)


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


#: Why a test marked ``reassembly(kind)`` is skipped when ``BAMBOO_SQL_EVAL=1``
#: reads statements with the evaluator (``codemap.panda.sqleval``) in place of
#: sql.py's reassembly.  Each kind has tests of what the evaluator reads
#: instead, in ``test_codemap_sqleval.py``; the marked tests go with the
#: reassembly.
_REASSEMBLY_KINDS = {
    "representation": (
        "the evaluator reads a statement an `if` adds a fragment to as two, with "
        "the fragment and without; the reassembly as one holding it as optional"
    ),
    "python": "the evaluator reads what the Python builds where the reassembly leaves a hole or misreads it",
    "union": (
        "the evaluator reads the columns a loop over a dict's keys writes as their "
        "union where it cannot read a skip; the reassembly names none"
    ),
    "inline": "the evaluator reads a statement written into the call; the reassembly only one held by a name",
    "cover": (
        "known cost: the evaluator's statements hold every fragment, not every "
        "combination of fragments as a statement of its own"
    ),
    "one_pass": "known cost: the evaluator runs a loop over an input of unknown length at least once",
    "per_call": (
        "the evaluator gives a call the spans of what it runs; the reassembly "
        "those of every statement the name held"
    ),
}


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    evaluated = os.environ.get("BAMBOO_SQL_EVAL") == "1"
    for item in items:
        marker = item.get_closest_marker("reassembly")
        if marker is not None:
            (kind,) = marker.args
            if kind not in _REASSEMBLY_KINDS:
                raise pytest.UsageError(f"{item.nodeid}: no reassembly kind {kind!r}")
            if evaluated:
                item.add_marker(
                    pytest.mark.skip(
                        reason=f"pins the reassembly ({kind}): {_REASSEMBLY_KINDS[kind]}"
                    )
                )
        gap = item.get_closest_marker("sqleval_gap")
        if gap is not None and evaluated:
            (why,) = gap.args
            item.add_marker(
                pytest.mark.xfail(
                    strict=True, reason=f"the evaluator does not read this yet: {why}"
                )
            )
