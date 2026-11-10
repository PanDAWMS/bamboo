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


def test_an_ordering_and_a_chain_are_decided_as_python_makes_them() -> None:
    """``nRow > 0`` with the count known; a chain is false where any pair is, whatever the others are; a comparison
    Python cannot make is unknown."""
    assert evaluate.evaluate("nRow > 0", _env(nRow=1)) is True
    assert evaluate.evaluate("parse_result < 2", _env(parse_result=2)) is False
    assert evaluate.evaluate("0 < nRow <= limit", _env(nRow=3, limit=3)) is True
    assert evaluate.evaluate("0 < nRow <= limit", _env(nRow=4, limit=3)) is False
    assert evaluate.evaluate("0 < nRow <= limit", _env(limit=3)) is UNKNOWN
    assert evaluate.evaluate("other < nRow <= limit", _env(nRow=4, limit=3)) is False
    assert evaluate.evaluate("nRow > 0", _env(nRow=None)) is UNKNOWN


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


def test_a_call_whose_whole_text_was_printed_takes_the_printed_value_before_it_is_run() -> (
    None
):
    """``isJumboJob(job)`` printed as a hole is what the run saw: run on what is known of ``job`` it could say
    otherwise, as the reader knows the job's row now; a print known stale is run as before."""
    env = Env(call={"isJumboJob": lambda job: False}.get)
    env.put("job", {"eventService": 4}, "rows")
    env.put("isJumboJob(job)", True, "printed")
    assert evaluate.evaluate("isJumboJob(job)", env) is True
    assert env.used["isJumboJob(job)"] == "printed"
    env.stale["isJumboJob(job)"] = "printed before the job changed"
    assert evaluate.evaluate("isJumboJob(job)", env) is False
    assert env.used["isJumboJob(job)"] == "called isJumboJob"


def test_a_substitution_whose_pattern_does_not_match_the_string_is_the_string() -> None:
    """insertNewJob's rewrite of a jobName: a name without the token is the name, whatever ``originPandaID`` is;
    a match is unknown, even with a known replacement; so is an unknown or non-string string, a keyword, a pattern
    that does not compile.  A printed hole comes first, and the call is not decidable before the values."""
    rewrite = r"re.sub('\\$ORIGINPANDAID', str(originPandaID), PRIOR)"
    env = _env(PRIOR="mc23_13p6TeV.801234.build")
    assert evaluate.evaluate(rewrite, env) == "mc23_13p6TeV.801234.build"
    assert env.used[rewrite] == "re.sub: the pattern does not match"
    assert (
        evaluate.evaluate(f"{rewrite} == taskName", _env(PRIOR="a.b", taskName="a.b"))
        is True
    )
    assert evaluate.evaluate(rewrite, _env(PRIOR="a.b.$ORIGINPANDAID")) is UNKNOWN
    assert evaluate.evaluate("re.sub('\\r', '', PRIOR)", _env(PRIOR="ab")) == "ab"
    assert evaluate.evaluate("re.sub('\\r', '', PRIOR)", _env(PRIOR="a\rb")) is UNKNOWN
    assert evaluate.evaluate(rewrite, _env()) is UNKNOWN
    assert evaluate.evaluate(rewrite, _env(PRIOR=None)) is UNKNOWN
    assert evaluate.evaluate(
        "re.sub('x', '', PRIOR, flags=re.I)", _env(PRIOR="ab")
    ) is (UNKNOWN)
    assert evaluate.evaluate("re.sub('(', '', PRIOR)", _env(PRIOR="ab")) is UNKNOWN
    printed = _env(PRIOR="a.b")
    printed.put(rewrite, "a.b.7", "printed")
    assert evaluate.evaluate(rewrite, printed) == "a.b.7"
    node = ast.parse(rewrite, mode="eval").body
    assert not evaluate.decidable(node, {"PRIOR"}, Env())


def test_a_standard_function_is_run_on_a_known_value_and_says_what_it_raises() -> None:
    """Whether ``json.loads('')`` or ``int(None)`` raises is Python's to say; a function not run, or a value not
    known, is unknown."""
    assert evaluate.run("json.loads", "") == evaluate.Raised("JSONDecodeError")
    assert evaluate.run("json.loads", None) == evaluate.Raised("TypeError")
    assert evaluate.run("json.loads", '{"a": 1}') == {"a": 1}
    assert evaluate.run("int", None) == evaluate.Raised("TypeError")
    assert evaluate.run("int", "7") == 7
    assert evaluate.run("math.ceil", None) == evaluate.Raised("TypeError")
    assert evaluate.run("str", None) is UNKNOWN
    assert evaluate.run("int", UNKNOWN) is UNKNOWN


def test_a_test_is_decidable_where_the_evaluator_decides_it_once_the_named_values_come() -> (
    None
):
    """The question a reader asks before a run's values are read, on ``ev``'s own shapes: a printed hole known by
    its text, a value held, a literal, an ordering, a call the reader runs; not a ``**`` spread."""

    def node(text: str) -> ast.expr:
        return ast.parse(text, mode="eval").body

    env = Env(call={"twice": lambda value: value * 2}.get)
    env.put("limit", 3, "given")
    both = node('job.status == "running" and limit')
    assert evaluate.decidable(both, {"job.status"}, env)
    env.put("job.status", "done", "printed")
    assert evaluate.evaluate(both, env) is False
    assert not evaluate.decidable(node("job.status == other"), set(), env)
    assert evaluate.decidable(node("nJobs < limit"), {"nJobs"}, env)
    assert evaluate.decidable(node("0 < nJobs <= limit"), {"nJobs"}, env)
    assert not evaluate.decidable(node("nJobs < other"), {"nJobs"}, env)
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


SINCE = """\
def holds(n):
    log.debug(f"n={n}")  # P
    x = 1
    if n == 2:  # S
        pass


def rebound(n, flag):
    log.debug(f"n={n}")  # P
    if flag:
        n = 3
    if n == 2:  # S
        pass


def elsewhere(n, flag):
    log.debug(f"n={n}")  # P
    if flag:
        n = 3
        return
    if n == 2:  # S
        pass


def looped(n, items):
    log.debug(f"n={n}")  # P
    for one in items:
        if n == 2:  # S
            pass
        n = one


def once(n):
    tries = 1
    log.debug(f"n={n}")  # P
    for attempt in range(tries):
        if n == 2:  # S
            pass
        n = 3


def deleted(n, flag):
    log.debug(f"n={n}")  # P
    if flag:
        del n
    if n == 2:  # S
        pass


def maybe(n, flag):
    if flag:
        log.debug(f"n={n}")  # P
    if n == 2:  # S
        pass


def walrus(n, items):
    log.debug(f"n={n}")  # P
    if (n := len(items)) > 0:
        pass
    if n == 2:  # S
        pass


def nested(n):
    log.debug(f"n={n}")  # P

    def inner():
        n = 5
        return n

    if n == 2:  # S
        pass


def converted(code):
    log.debug(f"code={code}")  # P
    if isinstance(code, int):
        code = str(code)
    if code in ["2", "4"]:  # S
        pass


def flagged(code, flag):
    log.debug(f"code={code}")  # P
    if flag:
        code = str(code)
    if code == "2":  # S
        pass


def second(code, flag):
    log.debug(f"code={code}")  # P
    if flag:
        flag = False
        code = str(code)
    if code == "2":  # S
        pass


def otherwise(code, flag):
    log.debug(f"code={code}")  # P
    if flag:
        pass
    else:
        code = str(code)
    if code == "2":  # S
        pass


def always(code):
    log.debug(f"code={code}")  # P
    code = str(code)
    if code == "2":  # S
        pass


def prefixed(code, prefix):
    log.debug(f"code={code}")  # P
    code = prefix + code
    if code == "x2":  # S
        pass


def twice(code):
    log.debug(f"code={code}")  # P
    code = str(code)
    code = code.strip()
    if code == "2":  # S
        pass
"""


def _since(function: str, name: str) -> pathcond.HeldSince:
    tree = ast.parse(SINCE)
    pathcond.attach_parents(tree)
    (func,) = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function
    ]
    lines = SINCE.splitlines()

    def statement(marker: str) -> ast.stmt:
        (line,) = [
            n
            for n in range(func.lineno, func.end_lineno + 1)  # type: ignore[operator]
            if lines[n - 1].endswith(f"# {marker}")
        ]
        (found,) = [
            node
            for node in ast.walk(func)
            if isinstance(node, ast.stmt) and node.lineno == line
        ]
        return found

    return pathcond.held_since(func, name, statement("S"), statement("P"))


def _lines(sites: tuple[ast.AST, ...]) -> list[object]:
    return [
        "entry" if isinstance(site, ast.arguments) else site.lineno  # type: ignore[attr-defined]
        for site in sites
    ]


def test_a_shown_value_holds_where_no_other_write_reaches_the_read_past_the_print() -> (
    None
):
    """A binding on a way that leaves before the read, a nested scope's own name, and a loop's later binding on a
    ``range`` of a name bound once to 1 (one pass, as :func:`reaching` reads it) reach nothing past the print."""
    for function in ("holds", "elsewhere", "once", "nested"):
        assert _since(function, "n") == pathcond.HeldSince((), True, ()), function


def test_a_write_reaching_the_read_past_the_print_is_another_value() -> None:
    """A binding on a way, one a loop brings back round, a ``del``, a walrus, and the entry where the print is not
    on every way."""
    assert _lines(_since("rebound", "n").others) == [11]
    assert _lines(_since("looped", "n").others) == [30]
    assert _lines(_since("deleted", "n").others) == [45]
    assert _lines(_since("walrus", "n").others) == [59]
    held = _since("maybe", "n")
    assert (_lines(held.others), held.raw) == (["entry"], True)


def test_a_rebinding_turning_the_shown_value_is_a_conversion_with_the_tests_it_runs_under() -> (
    None
):
    """killJob's ``if isinstance(code, int): code = str(code)``: the int way is text, any other as it was."""
    held = _since("converted", "code")
    assert (held.others, held.raw) == ((), True)
    ((conversion),) = held.conversions
    assert ast.unparse(conversion.value) == "str(code)"
    assert [ast.unparse(test) for test in conversion.tests] == ["isinstance(code, int)"]
    assert conversion.decided is True
    flagged = _since("flagged", "code").conversions[0]
    assert ([ast.unparse(test) for test in flagged.tests], flagged.decided) == (
        ["flag"],
        True,
    )
    always = _since("always", "code")
    assert (always.raw, always.conversions[0].tests, always.conversions[0].decided) == (
        False,
        (),
        True,
    )


def test_a_conversion_not_opening_its_if_or_in_an_else_is_not_decided() -> None:
    assert _since("second", "code").conversions[0].decided is False
    assert _since("otherwise", "code").conversions[0].decided is False


def test_a_rebinding_reading_another_name_or_another_conversion_is_another_value() -> (
    None
):
    prefixed = _since("prefixed", "code")
    assert (_lines(prefixed.others), prefixed.raw, prefixed.conversions) == (
        [120],
        False,
        (),
    )
    twice = _since("twice", "code")
    assert (_lines(twice.others), twice.raw, twice.conversions) == ([128], False, ())


def test_held_since_refuses_a_tree_without_parent_links() -> None:
    func = ast.parse(SINCE).body[0]
    statements = [node for node in ast.walk(func) if isinstance(node, ast.stmt)]
    with pytest.raises(ValueError, match="parent links"):
        pathcond.held_since(func, "n", statements[-1], statements[1])  # type: ignore[arg-type]


GUARDS = """
def run(job, items, flag):
    if job is None:  # E1
        return
    for item in items:  # L1
        if not item:  # E2
            continue
        if flag:  # G1
            if item.ok:  # G2
                pass
            else:
                item.mark()  # A1
"""


def _guards_at() -> tuple[ast.AST, dict[str, ast.stmt]]:
    tree = ast.parse(GUARDS)
    pathcond.attach_parents(tree)
    func = tree.body[0]
    marked = {
        text.rpartition("# ")[2]: n
        for n, text in enumerate(GUARDS.splitlines(), 1)
        if "# " in text
    }
    at = {
        marker: next(
            node
            for node in ast.walk(func)
            if isinstance(node, ast.stmt) and node.lineno == line
        )
        for marker, line in marked.items()
    }
    return func, at


def test_the_guards_on_a_node_are_its_ifs_and_the_early_exits_above_it_innermost_first() -> (
    None
):
    """Each ``if`` deciding a node with the value it runs on, its ``if`` and its kind: the ``else`` it is in, the
    ``if`` around that, the ``continue`` above in the loop, the ``return`` above the loop."""
    func, at = _guards_at()
    found = pathcond.guards(at["A1"], func)
    assert [(ast.unparse(g.test), g.runs_on, g.at, g.exit) for g in found] == [
        ("item.ok", False, at["G2"], False),
        ("flag", True, at["G1"], False),
        ("not item", False, at["E2"], True),
        ("job is None", False, at["E1"], True),
    ]
    assert pathcond._tests_on(at["A1"], func) == [(g.test, g.runs_on) for g in found]
    assert pathcond.loops_holding(at["A1"]) == [at["L1"]]


def test_a_head_keeps_the_guards_at_it_or_inside_it_and_must_hold_the_node() -> None:
    """From an ``if`` the run reached, its own test and the ones below; from the loop, the exit inside it too, not
    the one above it.  A head that is the node leaves none, and one beside it is refused."""
    func, at = _guards_at()

    def tests(head: ast.AST) -> list[str]:
        return [ast.unparse(g.test) for g in pathcond.guards(at["A1"], func, head)]

    assert tests(at["G1"]) == ["item.ok", "flag"]
    assert tests(at["L1"]) == ["item.ok", "flag", "not item"]
    assert tests(at["A1"]) == []
    with pytest.raises(ValueError):
        pathcond.guards(at["A1"], func, at["E1"])
