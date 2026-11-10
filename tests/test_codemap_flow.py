"""Which statements one call can run after which -- ``codemap.panda.flow`` on reaching's flow -- and the premise it
rests on, no ``with`` going on past a raise in its body."""

from __future__ import annotations

import ast

import pytest

from bamboo.codemap.gates import with_exits_are_read
from bamboo.codemap.models import MapFragment
from bamboo.codemap.panda import flow, pathcond, reaching

FLOW = """\
def handled(proxy, log):
    log.debug("start")  # F1
    try:
        proxy.connect()  # F2
    except Exception:
        log.error("failed")  # F3
    log.debug("done")  # F4


def looped(items, log):
    log.debug("begin")  # L1
    for item in items:  # L2
        if item:
            log.debug("one")  # L3
            continue
        use(item)  # L4
    log.debug("end")  # L5


def forever(log, module):
    log.info("start looping")  # W1
    while True:
        ret = module.main()  # W2
        if not ret:
            break
    log.info("finish looping")  # W3


def leaves(log, flag):
    if flag:
        log.debug("left")  # R1
        return
    log.debug("stayed")  # R2
    try:
        work()  # R3
        return
    finally:
        log.debug("finally")  # R4


def retried(proxy, log):
    for attempt in range(3):  # O1
        if proxy.connect():
            log.debug("connected")  # O2
        else:
            log.debug("retry")  # O3


def noisy(proxy, log):
    log.debug("a")  # N1
    try:
        log.debug(f"b {proxy.name}")  # N2
        proxy.run()  # N3
    except Exception:
        pass
    log.debug("c")  # N4


def frozen(ddm, log, spec):
    try:
        log.info(f"freezing {spec.name}")  # G1
        ddm.freeze(spec.name)  # G2
    except Exception:
        log.warning("failed")  # G3


def setup(log, items, flag):
    log.debug("running")  # P1
    if flag:
        prepare(1)  # P2
    else:
        prepare(2)  # P3
    log.debug("prepared")  # P4
    for item in items:
        log.debug("item")  # P5
        prepare(item)  # P6
    try:
        log.debug("trying")  # P7
        prepare(0)  # P8
    except Exception:
        log.error("failed")  # P9
    log.debug(f"done {prepare(3)}")  # P10


def guarded(log):
    log.debug("start")  # B1
    try:
        x = 1  # B2
        prepare(1)  # B3
        prepare(2)  # B4
    except Exception:
        log.error("failed")  # B5
    log.debug("end")  # B6


def nested(log):
    try:
        x = 1  # C0
        try:
            x = 2  # C1
            x = 3  # C2
        except Exception:
            pass  # C3
    except Exception:
        log.error(f"outer {x}")  # C4


def checked(self):
    if self.job_status in ["failed", "cancelled", "closed"]:
        return  # E1
    for file in self.job.Files:
        if file.lfn not in self.extra_info["nevents"]:
            self.logger.error(f"nEvents is missing for {file.lfn}")  # E2
            self.job.jobStatus = "failed"  # E3
            return  # E4


def post_processed(self, task_list):
    for task_spec in task_list:
        tmp_log = MsgWrapper(self.logger, f"<jediTaskID={task_spec.jediTaskID}>")
        tmp_log.info("start")  # Q1
        impl = self.implFactory.instantiateImpl(task_spec.vo)
        if impl is None:
            tmp_log.error("post-processor is undefined")  # Q2
            tmp_stat = Interaction.SC_FATAL
        else:
            tmp_log.info(f"post-process with {impl.__class__.__name__}")  # Q3
            try:
                tmp_stat = impl.doPostProcess(task_spec, tmp_log)  # Q4
            except Exception:
                tmp_log.error("post-process failed")  # Q5
                tmp_stat = Interaction.SC_FATAL
        if tmp_stat == Interaction.SC_FAILED:
            tmp_log.info("post-processing temporarily failed")  # Q6
            continue  # Q7
        tmp_log.info("done")  # Q8


def fetched(proxy):
    try:
        return proxy.get()  # Z1
    except Exception:
        return None  # Z2


def polled(log, c):
    while True:
        poll()  # V1
        log.info("start")  # V2
        if c:
            return  # V3


def scanned(log, c, items):
    for item in items:
        poll()  # U1
        log.info("start")  # U2
        if c:
            return  # U3
"""


def _statements() -> dict[str, ast.stmt]:
    tree = ast.parse(FLOW)
    pathcond.attach_parents(tree)
    lines = {
        text.rsplit("# ", 1)[1]: number
        for number, text in enumerate(FLOW.splitlines(), 1)
        if "  # " in text
    }
    by_line = {
        node.lineno: node
        for node in ast.walk(tree)
        if isinstance(node, ast.stmt) and not isinstance(node, ast.FunctionDef)
    }
    return {marker: by_line[line] for marker, line in lines.items()}


S = _statements()


def _prints(*markers: str) -> list[ast.stmt]:
    return [S[marker] for marker in markers]


def test_a_statement_that_raised_ran_and_its_raise_leads_on_to_the_handler() -> None:
    """``proxy.connect()`` raising is a pass by it: the handler's print can follow it with nothing between."""
    prints = _prints("F1", "F3", "F4")
    assert flow.runs_between(S["F1"], S["F3"], S["F2"], prints)
    assert flow.passes_through(S["F1"], S["F4"], S["F2"], prints)
    assert not flow.runs_between(S["F3"], S["F4"], S["F2"], prints)


def test_a_print_that_raised_printed_nothing_so_a_way_through_its_raise_passes_no_print() -> (
    None
):
    """A stop that raised is not passed; a print read as printed did not raise: ``freezing ...`` printed, so
    ``ddm.freeze`` ran before the handler's print."""
    prints = _prints("N1", "N2", "N4")
    assert flow.runs_after(S["N1"], S["N4"], prints, printed=True)
    assert not flow.runs_between(S["N1"], S["N4"], S["N3"], prints)
    assert flow.passes_through(S["G1"], S["G3"], S["G2"], _prints("G1", "G3"))


def test_a_loop_is_passed_once_its_iterable_is_read_and_a_while_true_leaves_by_its_break() -> (
    None
):
    """``for`` entered with no pass is still passed, and its iterable is read once, not again after a pass;
    ``use(item)`` is not on every way, and runs after a pass that printed and went round."""
    prints = _prints("L1", "L3", "L5")
    assert flow.passes_through(S["L1"], S["L5"], S["L2"], prints)
    assert not flow.runs_between(S["L3"], S["L5"], S["L2"], prints)
    assert not flow.passes_through(S["L1"], S["L5"], S["L4"], prints)
    assert flow.runs_between(S["L3"], S["L5"], S["L4"], prints)
    assert flow.passes_through(S["W1"], S["W3"], S["W2"], _prints("W1", "W3"))


def test_one_call_runs_both_only_on_a_way_that_holds_both() -> None:
    """A ``return`` ends the call; a ``finally`` runs on the way out of it."""
    assert not flow.together(S["R1"], S["R2"])
    assert not flow.together(S["R1"], S["R4"])
    assert flow.together(S["R4"], S["R2"])
    assert flow.together(S["R4"], S["R3"])


def test_a_print_every_way_to_which_passes_a_call_says_the_call_ran() -> None:
    """``prepared`` follows either arm's call: handed both, every way passes one, and one alone is not enough.
    A loop's call is passed by no print before it in the pass, nor by one after a loop that may run none; a call
    the print holds is passed before it prints."""
    assert flow.passes_before(S["P4"], [S["P2"], S["P3"]])
    assert not flow.passes_before(S["P4"], [S["P2"]])
    assert not flow.passes_before(
        S["P1"], [S["P2"], S["P3"], S["P6"], S["P8"], S["P10"]]
    )
    assert not flow.passes_before(S["P5"], [S["P6"]])
    assert not flow.passes_before(S["P7"], [S["P6"]])
    assert flow.passes_before(S["P10"], [S["P10"]])


def test_a_handler_print_says_a_call_ran_only_if_nothing_before_it_can_raise() -> None:
    """``log.debug("trying")`` can raise before ``prepare(0)`` is entered, so the handler's ``failed`` does not
    say the call ran; the logger call itself, entered before it can raise, is passed."""
    assert not flow.passes_before(S["P9"], [S["P8"]])
    assert flow.passes_before(S["P9"], [S["P7"]])


def _handler(marker: str) -> ast.ExceptHandler:
    handler = S[marker].parent  # type: ignore[attr-defined]
    assert isinstance(handler, ast.ExceptHandler)
    return handler


def test_a_run_with_no_mark_ran_every_try_to_its_end() -> None:
    """Handed no handler, no statement raises: the handler is not entered, and the try's last call ran before
    ``end`` -- where the map's own reading has ``prepare(1)`` raise past it."""
    assert not flow.runs_after(S["B1"], S["B5"], raised=[])
    assert flow.passes_before(S["B6"], [S["B4"]], raised=[])
    assert not flow.passes_before(S["B6"], [S["B4"]])


def test_a_handler_whose_mark_the_run_shows_is_entered_from_any_statement_of_its_try() -> (
    None
):
    """Which statement raised is not said: the try may have been left before ``x = 1`` as well, a literal the
    map's own reading never has raise."""
    raised = [_handler("B5")]
    assert flow.runs_after(S["B1"], S["B5"], raised=raised)
    assert not flow.passes_before(S["B5"], [S["B3"]], raised=raised)
    assert flow.passes_before(S["B5"], [S["B3"]])
    assert flow.runs_after(S["B1"], S["B6"], raised=raised)


def test_a_catch_all_the_run_did_not_enter_stops_a_raise_inside_it() -> None:
    """The silent inner handler would have caught a raise of ``x = 2`` or ``x = 3``: only the outer try's own
    statements raise into the outer handler, which then reads ``x = 1`` alone."""
    outer = _handler("C4")
    read = next(
        node
        for node in ast.walk(S["C4"])
        if isinstance(node, ast.Name) and node.id == "x"
    )
    writes = [S["C0"], S["C1"], S["C2"]]
    assert reaching.reaching(read, writes, raised=[outer]) == [S["C0"]]
    assert reaching.reaching(read, writes) == []
    assert not flow.runs_after(S["C0"], S["C3"], raised=[outer])
    assert reaching.catching_try(S["C1"], [outer]) is None
    assert reaching.catching_try(S["C0"], [outer]) is outer.parent  # type: ignore[attr-defined]
    assert reaching.catching_try(S["B6"], [_handler("B5")]) is None


def test_a_catching_try_reads_the_same_where_ast_has_no_try_star(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Python 3.10, which the project still runs on, has no ``ast.TryStar``."""
    outer = _handler("C4")
    monkeypatch.delattr(ast, "TryStar", raising=False)
    assert reaching.catching_try(S["C0"], [outer]) is outer.parent  # type: ignore[attr-defined]
    assert reaching.catching_try(S["C1"], [outer]) is None


def test_a_loop_the_run_went_round_once_has_no_way_back() -> None:
    loop = S["O1"]
    assert flow.together(S["O2"], S["O3"])
    assert not flow.together(S["O2"], S["O3"], once=[loop])
    assert not flow.runs_after(S["O2"], S["O3"], once=[loop])


def test_the_flow_is_asked_of_one_function_on_a_tree_with_parent_links() -> None:
    with pytest.raises(ValueError, match="one function"):
        flow.runs_after(S["F1"], S["L1"])
    bare = ast.parse(FLOW).body[0].body[0]  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="parent links"):
        flow.runs_after(bare, bare)


def test_a_print_from_another_parse_is_refused_not_answered() -> None:
    """The same print read off a tree of its own is another function's: without the refusal no way from that
    function's entry would meet the calls, and every print would say they ran."""
    other = ast.parse(FLOW)
    pathcond.attach_parents(other)
    again = next(
        node
        for node in ast.walk(other)
        if isinstance(node, ast.stmt) and node.lineno == S["P1"].lineno
    )
    with pytest.raises(ValueError, match="one function"):
        flow.passes_before(again, [S["P2"], S["P3"]])


def _args(marker: str) -> ast.arguments:
    node: ast.AST = S[marker]
    while not isinstance(node, ast.FunctionDef):
        node = node.parent  # type: ignore[attr-defined]
    return node.args


def _rows_of_the_end(
    first: ast.stmt | ast.arguments, prints: list[ast.stmt], passed: list[ast.stmt]
) -> list[ast.stmt]:
    return [p for p in prints if not flow.ends_after(first, [p], passed=passed)]


def test_a_pass_ends_past_every_candidate_where_the_function_runs_off_its_end() -> None:
    """``check_output_file_metadata``: a pass that takes no ``return`` and no arm ends after its loop over the
    files, and no print is on every such way, so nothing a run printed rules that end out."""
    passed = [S["E1"], S["E3"], S["E4"]]
    assert flow.ends_after(_args("E1"), passed=passed)
    assert _rows_of_the_end(_args("E1"), [S["E2"]], passed) == []


def test_a_pass_from_its_first_print_ends_at_that_print_again_or_off_the_end() -> None:
    """``post_process_tasks``: from ``start`` past the call and the ``continue``, a pass goes round to the next
    task's ``start`` or off the end once the loop is done, printing ``undefined`` and ``done`` on every way.  The
    flow reads the ``if`` on ``tmp_stat`` both ways, as it reads every branch."""
    passed = [S["Q4"], S["Q7"]]
    prints = _prints("Q2", "Q3", "Q5", "Q6", "Q8")
    assert flow.ends_after(S["Q1"], passed=passed)
    assert _rows_of_the_end(S["Q1"], prints, passed) == [S["Q2"], S["Q8"]]


def test_a_return_or_a_raise_out_of_the_function_is_not_its_end() -> None:
    """A function every way of which ends in a ``return`` or a raise -- one inside a ``try`` with a ``finally``
    too -- has no end past its candidates, whichever are handed; a ``break`` out of a ``while True`` leads on to
    it."""
    assert not flow.ends_after(_args("Z1"))
    assert not flow.ends_after(_args("Z1"), passed=[S["Z1"], S["Z2"]])
    assert not flow.ends_after(_args("R1"))
    assert flow.ends_after(S["W1"])
    assert not flow.ends_after(S["W1"], [S["W3"]])


def test_a_pass_runs_to_the_next_first_print_not_to_the_loop_head() -> None:
    """What runs between the loop's head and ``start`` is the previous pass's: past ``poll()`` and the
    ``return`` no ``start`` comes again, and a ``while True`` has no end; a ``for`` ends when its items do."""
    assert not flow.ends_after(S["V2"], passed=[S["V1"], S["V3"]])
    assert flow.ends_after(S["V2"], passed=[S["V3"]])
    assert flow.ends_after(S["U2"], passed=[S["U1"], S["U3"]])


def test_the_end_is_asked_of_one_function_s_pass() -> None:
    with pytest.raises(ValueError, match="one function"):
        flow.ends_after(S["F1"], [S["L1"]])
    module = ast.parse("x = 1\n")
    pathcond.attach_parents(module)
    with pytest.raises(ValueError, match="a function's pass"):
        flow.ends_after(module.body[0])


MANAGERS = """\
from contextlib import contextmanager, suppress


@contextmanager
def pooled(pool):
    proxy = pool.get()
    try:
        yield proxy
    finally:
        pool.put(proxy)


@contextmanager
def logged(log):
    try:
        yield
    except Exception:
        log.error("failed")


@contextmanager
def again(log):
    try:
        yield
    except Exception:
        log.error("failed")
        raise


class Quiet:
    def __exit__(self, kind, value, trace):
        return True


class Loud:
    def __exit__(self, kind, value, trace):
        return False


def careless():
    with suppress(KeyError):
        pass
"""


def test_a_context_manager_that_may_swallow_its_with_body_s_raise_is_listed_and_fails_the_gate() -> (
    None
):
    """A handler around the ``yield`` that does not raise again, an ``__exit__`` returning true,
    ``suppress``; a ``finally`` alone, a handler raising again, an ``__exit__`` returning False do not."""
    read, found = flow.swallowing_managers([("managers.py", ast.parse(MANAGERS))])
    assert read == 6
    assert sorted((one.name, one.kind) for one in found) == [
        ("Quiet.__exit__", "exit"),
        ("logged", "contextmanager"),
        ("suppress(KeyError)", "suppress"),
    ]
    fragment = MapFragment(
        map_id="panda",
        derived_from="panda-server-source 1.0.2",
        swallowing_managers=found,
        managers_read=read,
    )
    result = with_exits_are_read(fragment)
    assert not result.passed and len(result.failures) == 3
    assert with_exits_are_read(
        fragment.model_copy(update={"swallowing_managers": []})
    ).passed
