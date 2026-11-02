"""A test's value from what is known, and what a local holds where it is read -- ``codemap.panda.evaluate`` and
``pathcond.decided`` / ``pathcond.held_at``."""

from __future__ import annotations

import ast

import pytest

from bamboo.codemap.panda import evaluate, pathcond
from bamboo.codemap.panda.evaluate import UNKNOWN, Env


def _env(**values: object) -> Env:
    env = Env()
    for name, value in values.items():
        env.put(name, value, "given")
    return env


def test_a_test_is_decided_with_what_is_known_and_short_circuits_past_what_is_not() -> (
    None
):
    """``and`` / ``or`` stop where Python stops; unknown only where nothing known decides it."""
    test = 'status == "running" or useScout'
    assert evaluate.evaluate(test, _env(status="running")) is True
    assert evaluate.evaluate(test, _env(status="done")) is UNKNOWN
    assert evaluate.evaluate(test, _env(status="done", useScout=False)) is False
    assert evaluate.evaluate("isOK and not toSkip", _env(isOK=False)) is False
    assert evaluate.evaluate("tmpStat == -1", _env(tmpStat=-1)) is True
    assert evaluate.evaluate("one in limit", _env(one="a", limit=3)) is UNKNOWN


def test_a_call_is_looked_up_by_its_text_unless_the_reader_hands_the_function() -> None:
    """The map runs no corpus code: a call is what a printed hole of it says, or what the reader's function
    returns for it, with its arguments known."""
    env = _env(**{"isMergeAtOS(job)": True})
    assert evaluate.evaluate("isMergeAtOS(job)", env) is True
    assert env.used["isMergeAtOS(job)"] == "given"
    run = Env(call={"twice": lambda value: value * 2}.get)
    run.put("n", 3, "printed")
    assert evaluate.evaluate("twice(n) == 6", run) is True
    assert run.used["twice(n)"] == "called twice"
    assert evaluate.evaluate("twice(m) == 6", run) is UNKNOWN


def test_a_test_is_decidable_where_the_evaluator_decides_it_once_the_named_values_come() -> (
    None
):
    """The question a reader asks before a run's values are read, on ``ev``'s own shapes: a printed hole known by
    its text, a value held, a literal, a call the reader runs; not an ordering comparison, nor a ``**`` spread."""

    def node(text: str) -> ast.expr:
        return ast.parse(text, mode="eval").body

    env = Env(call={"twice": lambda value: value * 2}.get)
    env.put("limit", 3, "given")
    both = node('job.status == "running" and limit')
    assert evaluate.decidable(both, {"job.status"}, env)
    env.put("job.status", "done", "printed")
    assert evaluate.evaluate(both, env) is False
    assert not evaluate.decidable(node("job.status == other"), set(), env)
    assert not evaluate.decidable(node("nJobs < limit"), {"nJobs"}, env)
    assert evaluate.decidable(node("twice(n) in [2, 4]"), {"n"}, env)
    assert not evaluate.decidable(node("twice(**opts)"), {"opts"}, env)
    env.put("opts", {"value": 1}, "printed")
    assert evaluate.evaluate("twice(**opts)", env) is UNKNOWN
    assert not evaluate.decidable(node("other(n)"), {"n"}, env)
    env.stale["limit"] = "bound again before the test"
    assert not evaluate.decidable(node("limit"), set(), env)


def test_a_printed_value_is_read_as_python_reads_its_repr_and_a_stale_one_is_unknown() -> (
    None
):
    env = Env()
    env.put_printed("str(job.altStgOutFileList())", "[]", "line 12")
    assert evaluate.evaluate("job.altStgOutFileList() == []", env) is True
    env.stale["job.altStgOutFileList()"] = "written after the decision"
    assert evaluate.evaluate("job.altStgOutFileList() == []", env) is UNKNOWN
    assert env.used["job.altStgOutFileList()"] == "unknown: written after the decision"


def test_a_known_outcome_settles_the_atoms_it_leaves_no_choice_for() -> None:
    """``a and b`` taken settles both; a settled equality hands the known side's value to the other."""
    solved = evaluate.solve("isOK and not toSkip", True, Env())
    assert solved.settled == {"isOK": True, "toSkip": False}
    assert solved.consistent
    solved = evaluate.solve("taskSpec.status == status", True, _env(status="running"))
    assert solved.derived["taskSpec.status"][0] == "running"
    assert evaluate.solve("isOK or toSkip", False, Env()).settled == {
        "isOK": False,
        "toSkip": False,
    }
    assert evaluate.solve("isOK or toSkip", True, Env()).open == ["isOK", "toSkip"]


def test_a_path_condition_s_annotation_is_dropped_before_it_is_read() -> None:
    assert evaluate.strip_annotation("isOK  [isOK := True]") == "isOK"
    assert evaluate.read_elsewhere("job.status == 'x'  [at 120]")
    assert evaluate.read_elsewhere("flag  [in check()]")
    assert not evaluate.read_elsewhere("flag")


def test_a_flag_parameter_s_test_is_decided_by_the_constant_a_caller_hands() -> None:
    """``pathcond.decided`` reads the test with the one evaluator: the shapes it read before, a literal it folds
    (``-1``), and a name on the right (``killJobs``'s ``"keepUnmerged" in killOptions``, every caller but one
    handing the default ``[]``)."""

    def test(text: str) -> ast.expr:
        return ast.parse(text, mode="eval").body

    assert pathcond.decided(test("not forAnal"), {"forAnal": True}) is False
    assert pathcond.decided(test('mode in ("a", "b")'), {"mode": "b"}) is True
    assert (
        pathcond.decided(test("forAnal or other"), {"forAnal": False})
        is pathcond.UNDECIDED
    )
    assert pathcond.decided(test("forAnal or other"), {"forAnal": True}) is True
    assert pathcond.decided(test("nRow == -1"), {"nRow": -1}) is True
    assert (
        pathcond.decided(test('"keepUnmerged" in killOptions'), {"killOptions": []})
        is False
    )
    flag = test("forAnal and True")
    assert pathcond.decided(flag, {"other": True}) is pathcond.UNDECIDED
    assert pathcond.decided(flag, {"forAnal": True}) is True
    assert pathcond.decided(test("not 0"), {"forAnal": True}) is True


HELD = """\
def flags(res, items):
    isOK = True
    if isOK:  # H1
        pass
    frozen = None
    if res is not None:
        status, frozen = res
    if frozen is None:  # H2
        pass
    for one in items:
        skip = False
        if one:
            skip = True
            continue
        if not skip:  # H3
            pass
    count = 0
    for one in items:
        if count:  # H4
            pass
        count += 1
    limit = 3
    for one in items:
        if limit:  # H5
            pass
    got = None
    try:
        check(res)
        got = True
    finally:
        if got:  # H6
            pass
    if res:  # H7
        pass
    if never:  # H8
        pass
"""


def _held(marker: str, name: str) -> pathcond.Held:
    tree = ast.parse(HELD)
    pathcond.attach_parents(tree)
    func = tree.body[0]
    line = next(
        n for n, text in enumerate(HELD.splitlines(), 1) if text.endswith(f"# {marker}")
    )
    (at,) = [
        node
        for node in ast.walk(func)
        if isinstance(node, ast.If) and node.lineno == line
    ]
    return pathcond.held_at(func, name, at)  # type: ignore[arg-type]


def test_a_local_holds_the_one_literal_every_way_to_its_read_binds_last() -> None:
    """One binding of a literal reaching the read; one outside a loop nothing in it binds again; and the
    ``False`` a ``continue`` keeps the ``True`` from reaching past."""
    assert _held("H1", "isOK")[:2] == (pathcond.HELD_CONSTANT, True)
    assert _held("H3", "skip")[:2] == (pathcond.HELD_CONSTANT, False)
    assert _held("H5", "limit")[:2] == (pathcond.HELD_CONSTANT, 3)


def test_a_local_two_bindings_can_reach_holds_no_one_value() -> None:
    """A tuple's target binds as an ``=`` does, a ``+=`` of a pass before reaches the next, and a ``try`` body's
    binding reaches its ``finally`` beside the one before the ``try`` where a call before it may raise."""
    for marker, name in (("H2", "frozen"), ("H4", "count"), ("H6", "got")):
        held = _held(marker, name)
        assert (held.kind, held.value, len(held.sites)) == (
            pathcond.HELD_SEVERAL,
            None,
            2,
        ), marker


def test_a_name_only_the_function_s_entry_reaches_is_its_parameter_or_unbound() -> None:
    assert _held("H7", "res").kind == pathcond.HELD_PARAMETER
    assert _held("H8", "never").kind == pathcond.HELD_UNBOUND


def test_a_tree_without_parent_links_is_refused_rather_than_read_as_reached_from_the_entry() -> (
    None
):
    func = ast.parse(HELD).body[0]
    at = next(node for node in ast.walk(func) if isinstance(node, ast.If))
    with pytest.raises(ValueError, match="parent links"):
        pathcond.held_at(func, "isOK", at)  # type: ignore[arg-type]
