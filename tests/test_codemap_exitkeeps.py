"""What each exit of a function keeps a run from -- ``codemap.panda.exitkeeps`` on ``flow.kept_from``."""

from __future__ import annotations

import ast
from pathlib import Path

from bamboo.codemap import trace
from bamboo.codemap.models import (
    Anchor,
    Branch,
    FunctionPrintsNode,
    JunctionNode,
    MapFragment,
    SourceModule,
)
from bamboo.codemap.panda import exitkeeps

PACKAGE = "pkg"
FILE = f"{PACKAGE}/m"

TEXT = """\
import logging
log = logging.getLogger("x")


class Proxy:
    def finished(self, tasks):
        try:
            if not self.commit():
                raise RuntimeError("commit")  # X1
            for task in tasks:
                if task.locked:
                    log.debug("skip")  # P1
                    continue  # X2
                task.status = "running"  # A1
            log.debug("got")  # P2
            return tasks  # X3
        except Exception:
            log.error("failed")  # P3
            return None  # X4


def plain(job):
    job.status = "done"  # A2


def noisy(c):
    if c:
        return  # X5
    log.debug("after")  # P4
"""

LINES = {held.rsplit("# ", 1)[1]: number for number, held in enumerate(TEXT.splitlines(), 1) if "  # " in held}


def _fragment(tmp_path: Path) -> MapFragment:
    root = tmp_path / PACKAGE
    root.mkdir(parents=True, exist_ok=True)
    (root / "m").write_text(TEXT)
    prints = [
        FunctionPrintsNode(
            name=f"m:{found.owner}",
            map_id="m",
            derived_from="v",
            owner=found.owner,
            cls=found.cls,
            function=found.func.name,
            count=len(found.statements),
            statements=found.statements,
            anchor=Anchor(package=PACKAGE, file=FILE, line_start=found.func.lineno, line_end=found.func.end_lineno),
        )
        for found in trace.function_prints({PACKAGE: root}, [FILE])
    ]
    junctions = [
        JunctionNode(
            name=f"m:{owner}:{marker}",
            map_id="m",
            derived_from="v",
            subject="Task.status",
            owner=f"{FILE}::{owner}",
            anchor=Anchor(package=PACKAGE, file=FILE, line_start=LINES[marker]),
            branches=[Branch(outcome=marker, line=LINES[marker])],
        )
        for owner, marker in (("finished", "A1"), ("plain", "A2"))
    ]
    fragment = MapFragment(map_id="m", derived_from="v", junctions=junctions, function_prints=prints)
    module = SourceModule(package=PACKAGE, rel_path=FILE, tree=ast.parse(TEXT), source=TEXT, blob_sha="0")
    exitkeeps.attach(fragment, [module])
    return fragment


def _said(fragment: MapFragment) -> dict[str, tuple[str, list[str], list[str]]]:
    markers = {number: marker for marker, number in LINES.items()}
    return {
        markers[int(one.exit.rpartition("@")[2])]: (
            one.owner,
            [markers[line] for line in one.arms],
            [markers[line] for line in one.prints],
        )
        for one in fragment.exit_keeps
    }


def test_every_exit_of_a_function_holding_arms_or_prints_says_what_it_keeps_a_run_from(tmp_path: Path) -> None:
    """The raise before the loop keeps the run from the loop's arm and the prints after it, not the handler's print;
    the ``continue`` from the rest of the turn; the returns from nothing, listed with empty lists; a function
    holding only prints is read too, and one with no exit has nothing to list."""
    fragment = _fragment(tmp_path)
    finished, noisy = f"{FILE}::finished", f"{FILE}::noisy"
    assert _said(fragment) == {
        "X1": (finished, ["A1"], ["P1", "P2"]),
        "X2": (finished, ["A1"], []),
        "X3": (finished, [], []),
        "X4": (finished, [], []),
        "X5": (noisy, [], ["P4"]),
    }
    assert [one.exit for one in fragment.exit_keeps] == [
        f"raise@{LINES['X1']}",
        f"continue@{LINES['X2']}",
        f"return@{LINES['X3']}",
        f"return@{LINES['X4']}",
        f"return@{LINES['X5']}",
    ]
