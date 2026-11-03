"""Every way through a region of one function -- ``codemap.panda.walker``, the walker the querier's P reads a run's
rows with -- and the hooks a reader adds its own rules by."""

from __future__ import annotations

import ast
from dataclasses import dataclass

import pytest

from bamboo.codemap.calls import Calls
from bamboo.codemap.gitsource import blob_sha
from bamboo.codemap.models import SourceModule
from bamboo.codemap.panda import evaluate, pathcond, walker
from bamboo.codemap.panda.recognizers import sqlwrite

PACKAGE = "pkg"
FILE = f"{PACKAGE}/m.py"

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


class Task:
    splitRule: str

    def useScout(self) -> bool:
        return "SC" in self.splitRule


def read_only(job):
    if job.status:
        pass


def store(job):
    job.status = "b"


def add(job):
    job.files.append(1)


def rule(task):
    task.splitRule = "x"


def fill(items):
    items.append(1)


def put(m):
    m["j"] = 1


def asked(job, task: Task, items, m, flag: bool, held):
    read_only(job)  # C2
    store(job)  # C3
    add(job)  # C4
    rule(task)  # C5
    fill(items)  # C6
    put(m)  # C7
    held.append(job)  # C8
    fill(flag)  # C9
"""


@pytest.fixture(scope="module")
def corpus(tmp_path_factory) -> tuple[ast.Module, Calls]:
    """The module as a corpus of one file, and the map's calls of it: the walker reads a call as the map does."""
    root = tmp_path_factory.mktemp("walker") / PACKAGE
    root.mkdir()
    (root / "__init__.py").write_text("")
    (root / "m.py").write_text(SOURCE)
    tree = ast.parse(SOURCE)
    pathcond.attach_parents(tree)
    source = SourceModule(
        package=PACKAGE,
        rel_path=FILE,
        tree=tree,
        source=SOURCE,
        blob_sha=blob_sha(SOURCE),
    )
    calls = sqlwrite.corpus_calls([source], {PACKAGE: root})
    assert calls is not None
    return tree, calls


@pytest.fixture(scope="module")
def module(corpus: tuple[ast.Module, Calls]) -> ast.Module:
    return corpus[0]


@pytest.fixture(scope="module")
def calls(corpus: tuple[ast.Module, Calls]) -> Calls:
    return corpus[1]


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
    calls,
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
        corpus=calls,
        file=FILE,
        **kw,
    )


def test_an_if_goes_both_ways_and_a_jump_leaves(module, calls):
    w = _walker(module, calls, "checked", "A1", target="job.status", require_all=False)
    paths = w.run(walker.region_of(_at(module, "S1")))
    left = sorted(p.left for p in paths)
    # a store and a logger's call can raise too (reaching.may_raise): the copies leaving at 5 and 6 are alike to
    # the one leaving at 3, and merge into it
    assert left == ["", "call at 3", "return at 4"]
    through = next(p for p in paths if not p.left)
    returned = next(p for p in paths if p.left == "return at 4")
    assert (returned.arm, returned.state["job.status"]) == (True, "a")
    assert (through.arm, through.state["job.status"]) == (False, "b")


def test_a_call_forks_a_copy_that_leaves_and_forgets_what_it_is_handed(module, calls):
    w = _walker(
        module,
        calls,
        "called",
        "A3",
        target="job.status",
        relevant={"n"},
        require_all=False,
    )
    paths = w.run(walker.region_of(_at(module, "S2")))
    # the way past the if, and the arm's; each statement that can raise forks a copy, the one at 13 alike to the
    # one at 12 (len() changes nothing)
    assert sorted(p.left for p in paths) == [
        "",
        "",
        "call at 11",
        "call at 12",
        "call at 14",
        "call at 15",
    ]
    on = next(p for p in paths if p.arm and not p.left)
    assert on.target_called
    assert (
        on.state["job.status"] == "a"
    )  # the target's own value stays, read through target_called
    # what a call gives is the reader's to say (the environment's call): the walk's own gives none
    assert on.state["n"] is evaluate.UNKNOWN
    assert "job()" in on.marks


def test_the_rows_are_passed_in_order_and_each_one_asked(module, calls):
    asked: list[int] = []

    def check(w: walker.Walker, line: int, p: walker.Path) -> str:
        asked.append(line)
        return "too early" if line == _at(module, "P3").lineno and not p.arm else ""

    w = _walker(
        module,
        calls,
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
    # the way not into the if and the copies left at 11, 12, 14 and 15 miss a row
    assert w.dropped["E2 missed"] == 5
    assert asked == [_at(module, "P2").lineno, _at(module, "P3").lineno]


def test_a_row_out_of_its_order_drops_the_path(module, calls):
    w = _walker(module, calls, "called", "A3", target="job.status", rows=("P3", "P2"))
    assert w.run(walker.region_of(_at(module, "S2"))) == []
    assert w.dropped["E2 order"] == 1


def test_a_loop_is_passed_whole(module, calls):
    w = _walker(
        module,
        calls,
        "looped",
        "S3",
        target="job.status",
        relevant={"isOK"},
        decide=_decide,
        require_all=False,
    )
    paths = w.run(walker.region_of(_at(module, "S3")))
    # isOK holds True at S3 (held_at), so the return is not taken; S3's ``not`` can raise (reaching reads every
    # operator so), and the copy the loop leaves, able to divert, is alike to that one; the logger's call at 26
    # forks one
    assert sorted(p.left for p in paths) == ["", "call at 20", "call at 26"]
    on = next(p for p in paths if not p.left)
    assert on.state["isOK"] is True
    assert on.state["job.status"] is evaluate.UNKNOWN


def test_a_branch_taken_settles_its_atoms(module, calls):
    w = _walker(
        module,
        calls,
        "settled",
        "A5",
        target="job.status",
        relevant={"toSkip"},
        decide=_decide,
        require_all=False,
    )
    paths = w.run(walker.region_of(_at(module, "S4")))
    # the second if is decided by what taking the first one settled: two ways, not four (beside the copies left
    # where a statement can raise)
    through = [p for p in paths if not p.left]
    assert len(through) == 2
    assert sorted((p.facts.get("toSkip"), p.arm) for p in through) == [
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


def test_too_many_paths_stop_the_walk(module, calls, monkeypatch):
    monkeypatch.setattr(walker, "MAX_PATHS", 1)
    w = _walker(module, calls, "checked", "A1", target="job.status", require_all=False)
    with pytest.raises(walker.TooMany):
        w.run(walker.region_of(_at(module, "S1")))


def test_a_reader_adds_its_rules_by_the_hooks(module, calls):
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
        corpus=calls,
        file=FILE,
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


def _after(module, calls, mark, facts=(), state=None):
    """The facts and the values a path holding *facts* and *state* keeps across the call at *mark* in ``asked``."""
    w = walker.Walker(
        _func(module, "asked"),
        _at(module, mark),
        target="",
        relevant={"job", "task", "items", "m", "flag", "held"},
        order={},
        decide=_both,
        check_row=_passes,
        corpus=calls,
        file=FILE,
    )
    q = walker.Path(facts=dict.fromkeys(facts, True), state=dict(state or {}))
    w.calls(q, _at(module, mark))
    return set(q.facts), q.state


def test_a_call_drops_what_it_may_change_of_a_chain_it_is_handed_as_the_map_reads_the_call(
    module, calls
):
    """By the map's reading of the call (changes.call_changes), at the chain's first step: a callee that only reads
    keeps all; a store of ``.status`` drops what reads it; an append to ``job.files`` changes what it holds, not
    which list it is; a method reads its receiver's content; a bare name's identity no call changes; an item by its
    key; an object put in a list goes where it is not followed; a ``bool`` cannot change."""
    on_job = ["job.status == 'a'", "job.files", "job.files is not None"]
    assert _after(module, calls, "C2", on_job)[0] == set(on_job)
    assert _after(module, calls, "C3", on_job)[0] == {
        "job.files",
        "job.files is not None",
    }
    assert _after(module, calls, "C4", on_job)[0] == {
        "job.status == 'a'",
        "job.files is not None",
    }
    assert _after(module, calls, "C5", ["task.useScout()"])[0] == set()
    assert _after(module, calls, "C6", ["items", "items is not None"])[0] == {
        "items is not None"
    }
    assert _after(module, calls, "C7", ["m['k'] == 1", "m['j'] == 1"])[0] == {
        "m['k'] == 1"
    }
    assert _after(module, calls, "C8", ["job.status == 'a'"])[0] == set()
    assert _after(module, calls, "C9", ["flag"])[0] == {"flag"}
    values = {"items": [1], "job.status": "a"}
    assert _after(module, calls, "C2", state=values)[1] == values
    assert _after(module, calls, "C3", state=values)[1] == {
        "items": [1],
        "job.status": evaluate.UNKNOWN,
    }
    assert _after(module, calls, "C6", state=values)[1] == {
        "items": evaluate.UNKNOWN,
        "job.status": "a",
    }


def test_a_test_reads_a_chain_by_its_identity_or_by_what_it_holds():
    """``is``, a literal compared, a type and a key read a chain's identity; anything else what it holds; a method
    called reads its receiver's content, and a function called by its name nothing of its own."""
    read = {
        ast.unparse(chain): how
        for chain, how in walker.readings(
            "job.status == 'a' and x is None and isinstance(y, str)"
        )
    }
    assert read == {
        "job.status": walker.IDENTITY,
        "x": walker.IDENTITY,
        "y": walker.IDENTITY,
        "str": walker.IDENTITY,
    }
    read = {
        ast.unparse(chain): how
        for chain, how in walker.readings(
            "items and n in names and m[k] and task.useScout() and f(z)"
        )
    }
    assert read == {
        "items": walker.CONTENT,
        "n": walker.CONTENT,
        "names": walker.CONTENT,
        "m[k]": walker.CONTENT,
        "k": walker.IDENTITY,
        "task": walker.CONTENT,
        "z": walker.CONTENT,
    }
    assert {ast.unparse(c): h for c, h in walker.readings("s in ('a', 'b')")} == {
        "s": walker.IDENTITY
    }


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


def test_a_statement_can_raise_as_the_map_reads_exceptions():
    """reaching.may_raise: any expression but a literal, a name or a display of them -- an attribute, an f-string, a
    logger's call --, an unpacking, an ``assert``; a test or a context expression by any part of it."""
    tree = ast.parse(
        "x = 1\ny = [x, 'a']\nz = a.b\nw = f'{x}'\nlog.debug('x')\na, b = c\nassert x\n"
        "if x:\n    pass\nif x.y:\n    pass\n"
    )
    pathcond.attach_parents(tree)
    assert [walker.can_raise(s) for s in tree.body[:7]] == [
        False,
        False,
        True,
        True,
        True,
        True,
        True,
    ]
    assert not walker.can_raise(tree.body[7].test) and walker.can_raise(
        tree.body[8].test
    )
    loop = ast.parse("for i in xs:\n    y = i\n").body[0]
    pathcond.attach_parents(loop)
    assert not walker.may_divert(loop)
