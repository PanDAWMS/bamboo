"""Every way through a region of one function -- ``codemap.panda.walker``, the walker the querier's P reads a run's
rows with -- and the hooks a reader adds its own rules by."""

from __future__ import annotations

import ast
from dataclasses import dataclass

import pytest

from bamboo.codemap.panda import evaluate, pathcond, walker

SOURCE = """\
def checked(job, log, flag):
    if flag:  # S1
        job.status = "a"  # A1
        return
    log.debug("not flagged")  # P1
    job.status = "b"  # A2


def called(job, log, proxy):
    if job:  # S2
        job.status = "a"  # A3
        log.info(f"status {job.status}")  # P2
        n = len(job.status)  # A4
        proxy.update(job)  # C1
        log.info("updated")  # P3


def looped(job, items, log):
    isOK = True
    if not isOK:  # S3
        return
    for item in items:  # L1
        job.status = item
        if item:
            break
    log.debug("after")  # P4


def settled(job, toSkip, log):
    if not toSkip:  # S4
        log.debug("kept")  # P5
    if toSkip:  # S5
        job.status = "skipped"  # A5
"""


@pytest.fixture(scope="module")
def module() -> ast.Module:
    tree = ast.parse(SOURCE)
    pathcond.attach_parents(tree)
    return tree


def _func(module: ast.Module, name: str) -> ast.FunctionDef:
    return next(
        n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == name
    )


def _at(module: ast.Module, mark: str) -> ast.stmt:
    line = next(
        i
        for i, text in enumerate(SOURCE.splitlines(), 1)
        if text.rstrip().endswith(f"# {mark}")
    )
    return next(
        n for n in ast.walk(module) if isinstance(n, ast.stmt) and n.lineno == line
    )


def _both(w: walker.Walker, s: ast.If, p: walker.Path) -> list[bool]:
    return [True, False]


def _decide(w: walker.Walker, s: ast.If, p: walker.Path) -> list[bool]:
    v = w.truth(s.test, p, w.env(p))
    return [True, False] if v is evaluate.UNKNOWN else [bool(v)]


def _passes(w: walker.Walker, line: int, p: walker.Path) -> str:
    return ""


def _walker(
    module,
    func,
    arm,
    *,
    target="",
    relevant=(),
    rows=(),
    decide=_both,
    check_row=_passes,
    **kw,
):
    order = {
        _at(module, mark).lineno: (_at(module, mark), rank)
        for rank, mark in enumerate(rows)
    }
    return walker.Walker(
        _func(module, func),
        _at(module, arm),
        target=target,
        relevant=set(relevant),
        order=order,
        decide=decide,
        check_row=check_row,
        **kw,
    )


def test_an_if_goes_both_ways_and_a_jump_leaves(module):
    w = _walker(module, "checked", "A1", target="job.status", require_all=False)
    paths = w.run(walker.region_of(_at(module, "S1")))
    left = sorted(p.left for p in paths)
    assert left == ["", "return at 4"]
    through = next(p for p in paths if not p.left)
    returned = next(p for p in paths if p.left)
    assert (returned.arm, returned.state["job.status"]) == (True, "a")
    assert (through.arm, through.state["job.status"]) == (False, "b")


def test_a_call_forks_a_copy_that_leaves_and_forgets_what_it_is_handed(module):
    w = _walker(
        module, "called", "A3", target="job.status", relevant={"n"}, require_all=False
    )
    paths = w.run(walker.region_of(_at(module, "S2")))
    # the way past the if, and the arm's; a logger's call forks nothing, len() changes nothing but may raise
    assert sorted(p.left for p in paths) == ["", "", "call at 13", "call at 14"]
    on = next(p for p in paths if p.arm and not p.left)
    assert on.target_called
    assert (
        on.state["job.status"] == "a"
    )  # the target's own value stays, read through target_called
    # what a call gives is the reader's to say (the environment's call): the walk's own gives none
    assert on.state["n"] is evaluate.UNKNOWN
    assert "job()" in on.marks


def test_the_rows_are_passed_in_order_and_each_one_asked(module):
    asked: list[int] = []

    def check(w: walker.Walker, line: int, p: walker.Path) -> str:
        asked.append(line)
        return "too early" if line == _at(module, "P3").lineno and not p.arm else ""

    w = _walker(
        module,
        "called",
        "A3",
        target="job.status",
        rows=("P2", "P3"),
        check_row=check,
        track=True,
    )
    paths = w.run(walker.region_of(_at(module, "S2")))
    assert [p.seen for p in paths] == [
        (_at(module, "P2").lineno, _at(module, "P3").lineno)
    ]
    assert paths[0].at_row == (
        (_at(module, "P2").lineno, "a"),
        (_at(module, "P3").lineno, evaluate.UNKNOWN),
    )
    # the way not into the if and the copy left at the call miss the rows
    assert w.dropped["E2 missed"] == 2
    assert asked == [_at(module, "P2").lineno, _at(module, "P3").lineno]


def test_a_row_out_of_its_order_drops_the_path(module):
    w = _walker(module, "called", "A3", target="job.status", rows=("P3", "P2"))
    assert w.run(walker.region_of(_at(module, "S2"))) == []
    assert w.dropped["E2 order"] == 1


def test_a_loop_is_passed_whole(module):
    w = _walker(
        module,
        "looped",
        "S3",
        target="job.status",
        relevant={"isOK"},
        decide=_decide,
        require_all=False,
    )
    paths = w.run(walker.region_of(_at(module, "S3")))
    # isOK holds True at S3 (held_at), so the return is not taken; a loop that may divert leaves a copy, named
    # a call's where no return or raise is in it
    assert sorted(p.left for p in paths) == ["", "call in for at 22"]
    on = next(p for p in paths if not p.left)
    assert on.state["isOK"] is True
    assert on.state["job.status"] is evaluate.UNKNOWN


def test_a_branch_taken_settles_its_atoms(module):
    w = _walker(
        module,
        "settled",
        "A5",
        target="job.status",
        relevant={"toSkip"},
        decide=_decide,
        require_all=False,
    )
    paths = w.run(walker.region_of(_at(module, "S4")))
    # the second if is decided by what taking the first one settled: two ways, not four
    assert len(paths) == 2
    assert sorted((p.facts.get("toSkip"), p.arm) for p in paths) == [
        (False, False),
        (True, True),
    ]


def test_equal_paths_merge_and_many_are_widened(monkeypatch):
    a = walker.Path(state={"x": 1, "y": {"b", "a"}})
    b = walker.Path(state={"x": 1, "y": {"a", "b"}})
    assert walker.merge([a, b]) == [a]
    c = walker.Path(state={"x": 2, "y": {"a", "b"}}, facts={"z": True})
    joined = walker.widen([a, c], "x")
    assert len(joined) == 1
    assert joined[0].state == {"x": evaluate.UNKNOWN, "y": {"a", "b"}}
    assert joined[0].facts == {}
    # paths that differ in what agree() reads are not joined
    assert len(walker.widen([a, walker.Path(state={"x": 1}, arm=True)], "x")) == 2


def test_too_many_paths_stop_the_walk(module, monkeypatch):
    monkeypatch.setattr(walker, "MAX_PATHS", 1)
    w = _walker(module, "checked", "A1", target="job.status", require_all=False)
    with pytest.raises(walker.TooMany):
        w.run(walker.region_of(_at(module, "S1")))


def test_a_reader_adds_its_rules_by_the_hooks(module):
    @dataclass
    class Held(walker.Partial):
        text: str

    @dataclass
    class ReaderPath(walker.Path):
        notes: tuple = ()

        def key_parts(self) -> tuple:
            return super().key_parts() + (self.notes,)

    class Reader(walker.Walker):
        path_type = ReaderPath

        def value_of(self, q, t, vnode):
            value = super().value_of(q, t, vnode)
            return (
                Held(ast.unparse(vnode))
                if value is evaluate.UNKNOWN and vnode is not None
                else value
            )

        def bound(self, q, t, value):
            q.notes += (t,)

        def raises(self, node):
            return False

    w = Reader(
        _func(module, "called"),
        _at(module, "A3"),
        target="job.status",
        relevant={"n"},
        order={},
        decide=_both,
        check_row=_passes,
        require_all=False,
    )
    paths = w.run(walker.region_of(_at(module, "S2")))
    assert all(isinstance(p, ReaderPath) for p in paths)
    assert [p.left for p in paths] == ["", ""]  # no call raises for this reader
    on = next(p for p in paths if p.arm)
    assert on.notes == ("job.status", "n")
    # a Partial is not a value the evaluator reads
    on.state["n"] = Held("x")
    assert w.env(on).stale["n"] == walker.BOUND_ON_PATH


def test_the_static_parts():
    # each chain as long as it goes, the ones inside an item's key and a call's arguments too
    assert walker.chains("a.b[c.d] + f(e)") == frozenset({"a.b[c.d]", "c.d", "f", "e"})
    assert walker.root("self.job.Files[0].status") == "self"
    assert walker.affects("job", {"job.status"}) and walker.affects(
        "job.status", {"job"}
    )
    assert not walker.affects("job2", {"job.status"})
    tree = ast.parse("a, b.c = f()\nx += 1\nfor i, j in y: pass\nwith m() as k: pass\n")
    assert [walker.binds(s) for s in tree.body][0] == [("a", None), ("b.c", None)]
    assert walker.binds(tree.body[1]) == [("x", None)]
    assert walker.binds(tree.body[2]) == [("i", None), ("j", None)]
    assert walker.binds(tree.body[3]) == [("k", None)]
    call = ast.parse("save(job, len(rows))").body[0]
    # a reader builtin is no call of its own, but the call it is in is handed every name in it
    assert walker.mutating_roots(call) == {"save", "job", "len", "rows"}
    assert walker.mutating_roots(ast.parse("len(rows)").body[0]) == set()
    assert walker.below("job.status", None, "job") and not walker.below(
        "job", "a", "job"
    )


def test_a_logger_call_is_told_by_the_map_levels():
    tree = ast.parse('log.debug(f"x {y}")\nlog.publish("x")\nlog.info()\n')
    pathcond.attach_parents(tree)
    calls = [s.value for s in tree.body]
    assert [walker.is_log_call(c) for c in calls] == [True, False, False]
    assert walker.can_raise(tree.body[1]) and not walker.can_raise(tree.body[0])
