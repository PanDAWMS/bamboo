"""Which statements one call can run after which -- ``codemap.panda.flow`` on reaching's flow -- and the premise it
rests on, no ``with`` going on past a raise in its body."""

from __future__ import annotations

import ast

import pytest

from bamboo.codemap import pathcond, reaching
from bamboo.codemap.gates import with_exits_are_read
from bamboo.codemap.models import MapFragment
from bamboo.codemap.panda import flow

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


def finished(proxy, log, tasks):
    try:
        if not proxy.commit():
            raise RuntimeError("commit")  # K1
        for task in tasks:
            try:
                proxy.lock(task)
            except Exception:
                log.debug("skip")  # K2
                if not proxy.commit():
                    raise RuntimeError("commit")  # K3
                continue  # K4
            if task.scouted:
                task.status = "running"  # K5
            else:
                task.locked = True  # K6
            for chunk in task.chunks:
                if chunk:
                    break  # K7
                chunk.read = True  # K8
            else:
                task.read = True  # K9
        log.debug("got")  # K10
        return tasks  # K11
    except Exception:
        log.error("failed")  # K12
        return None  # K13


def tolerant(log, task):
    try:
        if task.bad:
            raise ValueError("bad")  # T1
        task.checked = True  # T2
    except ValueError:
        log.debug("bad")  # T3
    task.done = True  # T4


def tolerant_each(log, tasks):
    for task in tasks:
        try:
            if task.bad:
                raise ValueError("bad")  # T5
            task.checked = True  # T6
        except ValueError:
            log.debug("bad")  # T7
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


def test_a_raise_or_return_keeps_a_run_from_what_its_pass_would_have_reached_and_no_handler_reaches() -> None:
    """A ``raise`` before the loop that the outer ``except`` catches keeps the run from the loop's writes and the
    print after it, not from the handler's print; the ``raise`` in the loop's handler from the rest of the loop; the
    last ``return`` from nothing."""
    asked = _prints("K5", "K6", "K8", "K9", "K10", "K12")
    assert flow.kept_from(S["K1"], asked) == _prints("K5", "K6", "K8", "K9", "K10")
    assert flow.kept_from(S["K3"], asked) == _prints("K5", "K6", "K8", "K9", "K10")
    assert flow.kept_from(S["K11"], asked) == []
    assert flow.kept_from(S["K13"], asked) == []


def test_a_continue_or_break_keeps_a_run_from_the_rest_of_its_loop_s_turn_only() -> None:
    """``continue`` keeps the run from the turn's writes after it, not from the print after the loop nor the
    handler's print before it; ``break`` from the rest of its loop's turn and the loop's ``else``."""
    assert flow.kept_from(S["K4"], _prints("K2", "K5", "K6", "K8", "K9", "K10")) == _prints("K5", "K6", "K8", "K9")
    assert flow.kept_from(S["K7"], _prints("K5", "K8", "K9", "K10")) == _prints("K8", "K9")


def test_a_raise_a_handler_catches_keeps_the_run_only_from_what_the_handler_skips() -> None:
    """``raise ValueError`` caught in the function skips ``task.checked``; the handler's print and the write after
    the ``try`` are reached either way.  Caught inside a loop, the run goes on to the next turn, which reaches
    ``task.checked`` again: a raise is read over the call's pass, not the turn."""
    assert flow.kept_from(S["T1"], _prints("T2", "T3", "T4")) == _prints("T2")
    assert flow.kept_from(S["T5"], _prints("T6", "T7")) == []


def test_an_exit_read_as_a_pass_goes_on_and_its_flow_is_kept_apart_from_the_exit_s() -> None:
    """reaching's *passing*: the ``raise`` read as a ``pass`` reaches ``task.checked``; asked again without it, the
    flow kept for the first question is not the answer."""
    assert reaching.reaching(S["T2"], [S["T1"]], entered=[S["T1"]]) == []
    assert reaching.reaching(S["T2"], [S["T1"]], entered=[S["T1"]], passing=[S["T1"]]) == [S["T1"]]
    assert reaching.reaching(S["T2"], [S["T1"]], entered=[S["T1"]]) == []


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


ARMS = """\
def reactivate(rows, cur, log, limit):
    for row, frozen in rows:
        flag = False
        if frozen < limit:
            flag = True
            sql = "TO"  # A1
        else:
            sql = "TU"  # A2
        cur.execute(sql)
        if cur.rowcount > 0:
            if flag:
                log.info("timeout")  # A3
            else:
                log.info("reactivate")  # A4


def rebound(rows, cur, log):
    for row in rows:
        flag = False
        sql = "TU"  # B1
        flag = cur.check(row)
        if flag:
            log.info("checked")  # B2


def either(rows, cur, log):
    flag = False
    for row in rows:
        sql = "TU"  # C1
        if flag:
            log.info("again")  # C2
        flag = True


def finish(tasks, spec, log, update):
    for task in tasks:
        if update:
            spec.status = "done"  # D1
        spec.save()
        msg = f"set task_status={spec.status}"
        log.info(msg)  # D2
        if task.bad:
            raise ValueError(task)
        log.info("checked")  # D3
        if task.skip:
            continue
        log.info("kept")  # D4
    log.info("all")  # D5


def after_loop(rows, log):
    for row in rows:
        flag = False
        if row:
            flag = True
        else:
            sql = "TU"  # E1
    if flag:
        log.info("last flagged")  # E2
"""


def _marked(source: str) -> dict[str, ast.stmt]:
    tree = ast.parse(source)
    pathcond.attach_parents(tree)
    lines = {text.rsplit("# ", 1)[1]: number for number, text in enumerate(source.splitlines(), 1) if "  # " in text}
    by_line = {
        node.lineno: node for node in ast.walk(tree) if isinstance(node, ast.stmt) and not isinstance(node, ast.FunctionDef)
    }
    return {marker: by_line[line] for marker, line in lines.items()}


def test_a_literal_an_arm_s_way_binds_keeps_it_from_a_print_the_literal_decides_against() -> None:
    """The ``sql = "TO"`` arm runs beside ``flag = True``, the other with the turn's ``flag = False``: each is
    followed only by the print ``if flag:`` lets through for it."""
    arms = _marked(ARMS)
    assert flow.ruled_out_after(arms["A1"], arms["A4"])
    assert not flow.ruled_out_after(arms["A1"], arms["A3"])
    assert flow.ruled_out_after(arms["A2"], arms["A3"])
    assert not flow.ruled_out_after(arms["A2"], arms["A4"])


def test_a_name_bound_again_after_the_arm_or_held_two_ways_rules_on_nothing() -> None:
    """``flag = cur.check(row)`` between the arm and the test, and a ``flag`` the arm is reached with as ``False``
    from before the loop or ``True`` from the last turn: neither test is decided."""
    arms = _marked(ARMS)
    assert not flow.ruled_out_after(arms["B1"], arms["B2"])
    assert not flow.ruled_out_after(arms["C1"], arms["C2"])


def test_a_print_past_the_loop_reads_the_last_turn_s_binding_not_the_arm_s() -> None:
    """``if flag:`` past the loop reads what the last turn bound: the arm's turn binds ``flag = False``, a later
    turn may bind ``True``, so the arm does not rule the print out."""
    arms = _marked(ARMS)
    assert flow.turn_loops(arms["E1"], arms["E2"]) == []
    assert not flow.ruled_out_after(arms["E1"], arms["E2"])


def test_a_print_every_way_after_a_statement_passes_in_its_turn_follows_it() -> None:
    """``set task_status=...`` follows the arm on every way of a run that entered no handler -- ``spec.save()``
    raises only into a handler the run shows; a ``raise``, a ``continue`` and the end of the turn leave before the
    others, and a print past the loop is in no turn of it."""
    arms = _marked(ARMS)
    assert flow.passes_after(arms["D1"], arms["D2"])
    assert not flow.passes_after(arms["D1"], arms["D3"])
    assert not flow.passes_after(arms["D3"], arms["D4"])
    assert not flow.passes_after(arms["D1"], arms["D5"])
    assert not flow.passes_after(arms["A1"], arms["A3"])


LOGGERS = """\
def pre_staging(self):
    try:
        tmpLog = MsgWrapper(logger, "label")  # G1
        tmpLog.debug("start")  # G2
        got = self.lock()
        if not got:
            tmpLog.debug("skip")  # G3
            return
        tmpLog.debug("done")  # G4
    except Exception:
        tmpLog.error("failed")  # G5


def bound_late(self):
    try:
        self.prepare()
        tmpLog = make_logger()  # L1
    except Exception:
        tmpLog.error("failed")  # L3
    tmpLog.debug("start")  # L2


def attribute_late(self):
    try:
        self.prepare()
        self.ready = True  # A1
    except Exception:
        self.logger.error("failed")  # A3
    self.logger.debug("start")  # A2


def two_loggers(self):
    tmpLog = MsgWrapper(logger, "first")  # K1
    tmpLog.debug("start")  # K2
    tmpLog = MsgWrapper(logger, "second")  # K3
    tmpLog.debug("begin")  # K4
    tmpLog.debug("end")  # K5
"""


def test_the_entry_print_is_read_for_a_run_by_the_handlers_its_marks_show() -> None:
    """``doForPreStaging``'s shape: ``start`` is before every other print of a run that shows no handler's mark; where
    the run shows the handler, or every statement may raise, ``start`` may itself raise into it, so it is no entry
    print -- and it still comes before ``skip``, which only a way past it reaches."""
    marked = _marked(LOGGERS)
    prints = [marked[one] for one in ("G2", "G3", "G4", "G5")]
    handler = marked["G5"].parent
    assert isinstance(handler, ast.ExceptHandler)
    assert flow.entry_print(prints, raised=[]) is marked["G2"]
    assert flow.entry_print(prints) is None
    assert flow.entry_print(prints, raised=[handler]) is None
    assert flow.printed_before(marked["G2"], marked["G3"], raised=[handler])
    assert not flow.printed_before(marked["G2"], marked["G5"], raised=[handler])


def test_a_print_through_a_logger_the_function_binds_is_counted_only_on_ways_through_a_binding() -> None:
    """The handler's print reads ``tmpLog`` before the ``try`` ever bound it on every way into the handler: it prints
    nothing there, so ``start`` after the ``try`` is before it on every way that prints it.  The same shape through
    ``self.logger``, bound from the entry, is no such print: ``self.prepare()`` raising reaches it."""
    marked = _marked(LOGGERS)
    assert flow.printed_before(marked["L2"], marked["L3"])
    assert not flow.printed_before(marked["A2"], marked["A3"])


def test_prints_through_a_name_bound_again_for_another_mark_are_another_logger_s() -> None:
    """``tmpLog`` bound twice in one call: the prints after the second binding go through a logger of their own,
    whose entry print is ``begin``; spelt alike, ``start`` would read as the entry print of all three."""
    marked = _marked(LOGGERS)
    prints = [marked[one] for one in ("K2", "K4", "K5")]
    groups = flow.logger_groups(prints)
    assert [[one.lineno for one in group] for group in groups] == [
        [marked["K2"].lineno],
        [marked["K4"].lineno, marked["K5"].lineno],
    ]
    assert flow.entry_print(groups[1]) is marked["K4"]


def test_a_logger_name_bound_again_to_the_same_making_is_one_logger() -> None:
    """``master.main`` binds ``main_log = get_logger()`` at its start and again before its last line: one logger."""
    source = """\
def main():
    main_log = get_logger()
    main_log.info("main start")  # M1
    run()
    main_log = get_logger()
    main_log.info("main end")  # M2
"""
    marked = _marked(source)
    group = [marked["M1"], marked["M2"]]
    assert flow.logger_groups(group) == [group]
    assert flow.entry_print(group, raised=[]) is marked["M1"]
