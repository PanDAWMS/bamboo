"""Which statements one call can run after which -- ``codemap.panda.flow`` on reaching's flow -- and the premise it
rests on, no ``with`` going on past a raise in its body."""

from __future__ import annotations

import ast

import pytest

from bamboo.codemap.gates import with_exits_are_read
from bamboo.codemap.models import MapFragment
from bamboo.codemap.panda import flow, pathcond

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
