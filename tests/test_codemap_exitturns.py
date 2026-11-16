"""What an exit keeps a run from over one turn of each loop holding it, the calls into arms it keeps a run from, and
the loops holding each print -- ``ExitKeeps.turns``, ``ExitKeeps.calls`` and ``MapFragment.print_loops``, for a
reader whose pass is one turn of the loop its entry row's print is in."""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import pytest

from bamboo.codemap import calls as calls_mod
from bamboo.codemap import reach, trace
from bamboo.codemap.gitsource import blob_sha
from bamboo.codemap.models import (
    Anchor,
    Branch,
    ExitKeeps,
    FunctionPrintsNode,
    JunctionNode,
    MapFragment,
    PrintLoops,
    SourceModule,
)
from bamboo.codemap.panda import exitkeeps, flow, mustprints
from bamboo.codemap.panda.recognizers import sqlwrite

PACKAGE = "pandajedi"
FILE = "pandajedi/jediorder/Worker.py"

TEXT = textwrap.dedent(
    """\
    import logging

    log = logging.getLogger("x")


    def write(item):
        item.status = "stored"  # A4


    def save(item):
        write(item)


    def note(item):
        log.debug("note")


    class Worker:
        def runImpl(self, queue):
            while True:  # L1
                items = queue.get()
                if not items:
                    log.debug("done")  # P5
                    return  # X6
                for item in items:  # L2
                    log.info("start")  # P6
                    for again in range(2):  # L3
                        if item.ready:
                            item.status = "done"  # A3
                            save(item)  # C1
                            break  # X7
                        if again > 0:
                            break  # X8
                        log.info("kill")  # P7
                    note(item)  # C2
                    write(item)  # C3
    """
)

LINES = {held.rsplit("# ", 1)[1]: number for number, held in enumerate(TEXT.splitlines(), 1) if "  # " in held}
MARKERS = {number: marker for marker, number in LINES.items()}


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> MapFragment:
    tmp_path: Path = tmp_path_factory.mktemp("exitturns")
    path = tmp_path / FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(TEXT)
    parsed = calls_mod.parsed(path)
    assert parsed is not None
    modules = [SourceModule(package=PACKAGE, rel_path=FILE, tree=parsed[1], source=TEXT, blob_sha=blob_sha(TEXT))]
    calls = sqlwrite.corpus_calls(modules, {PACKAGE: tmp_path / PACKAGE})
    assert calls is not None
    record = reach.ReachGraph(reach.graph(calls, "m", "v"))
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
        for found in trace.function_prints({PACKAGE: tmp_path / PACKAGE}, [FILE])
    ]
    junctions = [
        JunctionNode(
            name=f"m:{owner}:{marker}",
            map_id="m",
            derived_from="v",
            subject="Item.status",
            owner=f"{FILE}::{owner}",
            anchor=Anchor(package=PACKAGE, file=FILE, line_start=LINES[marker]),
            branches=[Branch(outcome=marker, line=LINES[marker])],
        )
        for owner, marker in (("runImpl", "A3"), ("write", "A4"))
    ]
    fragment = MapFragment(map_id="m", derived_from="v", junctions=junctions, function_prints=prints)
    exitkeeps.attach(fragment, modules, record)
    return fragment


def _exit(fragment: MapFragment, marker: str) -> ExitKeeps:
    (one,) = [one for one in fragment.exit_keeps if one.exit.endswith(f"@{LINES[marker]}")]
    return one


def _named(lines: list[int]) -> list[str]:
    return [MARKERS[line] for line in lines]


def _turns(one: ExitKeeps) -> list[tuple[str, list[str], list[str], list[str]]]:
    return [(MARKERS[turn.loop], _named(turn.arms), _named(turn.prints), _named(turn.calls)) for turn in one.turns]


def test_a_return_outside_a_loop_s_body_has_no_turn_of_it_and_keeps_its_own_loop_s_turn_without_the_round(
    built: MapFragment,
) -> None:
    """The return at X6 sits in the while's body, before the for: a pass that is one turn of the for cannot leave by
    it.  Over the call it keeps the run from P5 too (the while's next turn); over one turn of the while, not."""
    one = _exit(built, "X6")
    assert (_named(one.arms), _named(one.prints), _named(one.calls)) == (["A3"], ["P5", "P6", "P7"], ["C1", "C3"])
    assert _turns(one) == [("L1", ["A3"], ["P6", "P7"], ["C1", "C3"])]


def test_a_break_keeps_a_run_from_what_a_further_turn_of_its_own_loop_reaches_over_the_turn_around_it(
    built: MapFragment,
) -> None:
    """The break at X8 leaves the inner loop: over its own turn it keeps the run from P7 alone; over a turn of the
    for around it, from what another turn of the inner loop reaches -- the arm and the call into one --; over a turn
    of the while, from nothing, since the for's next turn reaches them all again.  The first turn is its own loop's,
    the same as its own lists."""
    one = _exit(built, "X8")
    assert (_named(one.arms), _named(one.prints), _named(one.calls)) == ([], ["P7"], [])
    assert _turns(one) == [
        ("L3", [], ["P7"], []),
        ("L2", ["A3"], ["P7"], ["C1"]),
        ("L1", [], [], []),
    ]


def test_a_call_kept_from_is_one_whose_callee_holds_an_arm_or_reaches_one(built: MapFragment) -> None:
    """C3 calls write, which holds A4; C1 calls save, which reaches write; C2 calls note, which reaches no arm."""
    one = _exit(built, "X6")
    assert "C2" not in _named(one.calls)
    assert all(turn.loop != LINES["L2"] for turn in one.turns)


def test_each_print_in_a_loop_s_body_says_its_loops_innermost_first(built: MapFragment) -> None:
    owner = f"{FILE}::runImpl"
    assert built.print_loops == [
        PrintLoops(owner=owner, line=LINES["P5"], loops=[LINES["L1"]]),
        PrintLoops(owner=owner, line=LINES["P6"], loops=[LINES["L2"], LINES["L1"]]),
        PrintLoops(owner=owner, line=LINES["P7"], loops=[LINES["L3"], LINES["L2"], LINES["L1"]]),
    ]


def test_without_the_reach_record_no_call_is_said(built: MapFragment) -> None:
    """The calls are read off the reach record; a build without one says none, and the rest is the same."""
    module = SourceModule(package=PACKAGE, rel_path=FILE, tree=ast.parse(TEXT), source=TEXT, blob_sha="0")
    fragment = built.model_copy(update={"exit_keeps": [], "print_loops": []})
    exitkeeps.attach(fragment, [module])
    one = _exit(fragment, "X8")
    assert one.calls == [] and all(turn.calls == [] for turn in one.turns)
    assert _turns(one)[1][:3] == ("L2", ["A3"], ["P7"])


def test_a_turn_is_asked_only_of_a_loop_whose_body_holds_the_exit() -> None:
    """The for at L2 does not hold the return at X6: no turn of it is a question about that exit."""
    tree = ast.parse(TEXT)
    mustprints._parented(tree)
    found = {(type(node).__name__, getattr(node, "lineno", 0)): node for node in ast.walk(tree)}
    with pytest.raises(ValueError):
        flow.kept_from(found[("Return", LINES["X6"])], [], over=found[("For", LINES["L2"])])
