"""Walking the source of one arm to say why it ran with the value it did.

Against snippets written to a temporary tree rather than against the corpus:
the walk is about Python, and a fixture that spells out the shape under test
says what is being claimed in a way a real 600-line function does not.  The
corpus checks that follow are in ``test_codemap_strategy``.

Nothing here reads the installed distribution and nothing here touches
production.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from bamboo.codemap import models
from bamboo.codemap import trace as trace_mod
from bamboo.codemap.models import (
    STOP_CONFIG,
    STOP_EXTERNAL,
    STOP_PARAMETER,
    STOP_UPSTREAM,
    TRACE_BINDING,
    TRACE_HANDOVER,
    TRACE_LOOP,
    TRACE_UNBOUND,
    TRACE_WRITE,
    Handover,
)
from bamboo.codemap.panda import pathcond, provenance

PACKAGE = "pandajedi"


def _tree(tmp_path: Path, **files: str) -> dict[str, Path]:
    root = tmp_path / PACKAGE
    root.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (root / name).write_text(text)
    return {PACKAGE: root}


def _walked(roots, *, file, owner, lines, **kwargs):
    return trace_mod.walk(
        roots,
        file=f"{PACKAGE}/{file}",
        owner=f"{PACKAGE}/{file}::{owner}",
        lines=lines,
        classify=provenance.classify,
        **kwargs,
    )


def _walk(roots, *, file, owner, lines, **kwargs):
    """The two answers most of these tests are about: the steps and the note."""
    walked = _walked(roots, file=file, owner=owner, lines=lines, **kwargs)
    return walked.steps, walked.note


KNIGHT = '''\
class Feeder:
    def run(self):
        for spec in self.specs:
            broken = False
            try:
                meta = self.ddmIF.getDatasetMetaData(spec.name)
            except Exception:
                if errtype == Fatal:
                    broken = True
            if spec.empty():
                broken = True
            if broken:
                spec.status = "tobroken"
'''
BROKEN_LINE = 13


def test_the_arm_is_the_first_step_and_carries_its_own_guard(tmp_path):
    roots = _tree(tmp_path, feeder=KNIGHT)

    steps, note = _walk(roots, file="feeder", owner="run", lines=[BROKEN_LINE])

    assert note == ""
    first = steps[0]
    assert (first.kind, first.line, first.value) == (TRACE_WRITE, BROKEN_LINE, "'tobroken'")
    assert first.guards == ["broken"]
    # The loop is not an ``if``, so the path condition cannot mention it.
    assert any(u.startswith(pathcond.UNSEEN_LOOP) for u in first.unseen)


def test_a_flag_set_in_several_places_comes_back_as_several_steps(tmp_path):
    # The eager version had to fold this into one condition string per branch,
    # and that is where its loop rule and its length cap came from.  A
    # use-time answer may be a set, so three sites are three steps.
    roots = _tree(tmp_path, feeder=KNIGHT)

    steps, _ = _walk(roots, file="feeder", owner="run", lines=[BROKEN_LINE])

    bindings = [s for s in steps if s.kind == TRACE_BINDING and s.name == "broken"]
    assert sorted(s.value for s in bindings) == ["False", "True", "True"]


def test_a_handler_gets_the_terminal_of_what_the_try_body_called(tmp_path):
    # The handler ran because something above it raised, and naming what that
    # was is the only way the walk reaches "DDM said so" from an arm whose own
    # right-hand side is the constant True.
    roots = _tree(tmp_path, feeder=KNIGHT)

    steps, _ = _walk(roots, file="feeder", owner="run", lines=[BROKEN_LINE])

    raised = [s for s in steps if s.terminal == STOP_EXTERNAL]
    assert [s.detail for s in raised] == ["DDM / Rucio"]
    assert any(u.startswith(pathcond.UNSEEN_EXCEPT) for u in raised[0].unseen)


def test_a_guard_is_walked_as_well_as_the_value(tmp_path):
    roots = _tree(
        tmp_path,
        feeder='''\
class Feeder:
    def run(self):
        threshold = getConfigValue("jedi", "SCOUT_LIMIT")
        if self.count > threshold:
            spec.status = "exhausted"
''',
    )

    steps, _ = _walk(roots, file="feeder", owner="run", lines=[5])

    # The arm's own value is a literal, so everything below comes from the
    # guard: the threshold it compares against, and the attribute it reads.
    threshold = [s for s in steps if s.name == "threshold"]
    assert [(s.terminal, s.detail) for s in threshold] == [
        (STOP_CONFIG, "jedi.SCOUT_LIMIT")
    ]


def test_a_loop_target_is_followed_to_what_is_iterated(tmp_path):
    roots = _tree(
        tmp_path,
        commando='''\
class Worker:
    def runImpl(self):
        tasks = self.taskList.get(10)
        for task_id, command_map in tasks:
            command = command_map["command"]
            spec.status = command_map["done"]
''',
    )

    steps, _ = _walk(
        roots,
        file="commando",
        owner="runImpl",
        lines=[6],
        handovers=[
            Handover(
                entry=f"{PACKAGE}/commando",
                via="start",
                reached_by="dispatch",
                fields={"taskList": "taskList"},
            )
        ],
    )

    loop = [s for s in steps if s.kind == TRACE_LOOP]
    assert [(s.name, s.value) for s in loop] == [("command_map", "an element of tasks")]


def test_a_worker_field_crosses_to_the_function_that_built_it(tmp_path):
    # The edge no call graph has: the knight hands its rows over through a
    # constructor and a thread, so the arm's own module cannot reach them.
    roots = _tree(
        tmp_path,
        commando='''\
class Commando:
    def start(self):
        rows = self.taskBufferIF.getTasksToExecCommand_JEDI(vo, label)
        taskList = ListWithLock(rows)
        thr = CommandoThread(taskList)


class CommandoThread:
    def runImpl(self):
        tasks = self.taskList.get(10)
        spec.status = tasks[0]
''',
    )

    steps, _ = _walk(
        roots,
        file="commando",
        owner="runImpl",
        lines=[11],
        handovers=[
            Handover(
                entry=f"{PACKAGE}/commando",
                via="start",
                reached_by="dispatch",
                fields={"taskList": "taskList"},
            )
        ],
    )

    crossed = [s for s in steps if s.kind == TRACE_HANDOVER]
    assert [(s.name, s.value, s.owner.endswith("::start")) for s in crossed] == [
        ("self.taskList", "taskList", True)
    ]
    assert [(s.terminal, s.detail) for s in steps if s.terminal] == [
        (STOP_UPSTREAM, "getTasksToExecCommand_JEDI")
    ]


def test_a_field_no_handover_explains_stops_instead_of_guessing(tmp_path):
    roots = _tree(
        tmp_path,
        commando='''\
class Worker:
    def runImpl(self):
        spec.status = self.mystery
''',
    )

    steps, _ = _walk(roots, file="commando", owner="runImpl", lines=[3])

    stopped = [s for s in steps if s.name == "self.mystery"]
    assert [(s.kind, s.terminal) for s in stopped] == [(TRACE_UNBOUND, STOP_PARAMETER)]


def test_a_parameter_stops_at_the_call_site(tmp_path):
    roots = _tree(
        tmp_path,
        refiner='''\
class Refiner:
    def apply(self, new_status):
        spec.status = new_status
''',
    )

    steps, _ = _walk(roots, file="refiner", owner="apply", lines=[3])

    assert [(s.name, s.terminal) for s in steps if s.terminal] == [
        ("new_status", STOP_PARAMETER)
    ]


def test_a_name_bound_from_itself_does_not_loop(tmp_path):
    roots = _tree(
        tmp_path,
        refiner='''\
class Refiner:
    def apply(self):
        count = 0
        count = count + 1
        spec.status = count
''',
    )

    steps, note = _walk(roots, file="refiner", owner="apply", lines=[5])

    assert note == ""
    assert sum(1 for s in steps if s.name == "count" and s.kind == TRACE_BINDING) == 2


def test_the_walk_refuses_a_tree_the_map_was_not_built_from(tmp_path):
    # A map built from the installed 1.0.2 read against a checkout printed a
    # plausible line of the wrong version once already.  A trace is shaped
    # like an answer, so it is refused rather than warned about.
    roots = _tree(tmp_path, feeder=KNIGHT)

    steps, note = _walk(
        roots,
        file="feeder",
        owner="run",
        lines=[BROKEN_LINE],
        expected_sha="0" * 40,
    )

    assert steps == []
    assert "not the one that was mapped" in note


def test_a_budget_that_runs_out_is_recorded_rather_than_hidden(tmp_path):
    roots = _tree(tmp_path, feeder=KNIGHT)

    steps, note = _walk(
        roots,
        file="feeder",
        owner="run",
        lines=[BROKEN_LINE],
        budget=trace_mod.Budget(files=1, functions=1, steps=2),
    )

    assert len(steps) == 2
    assert "budget" in note


def test_an_imported_name_is_not_reported_as_unexplained(tmp_path):
    # ``JediTaskSpec`` and ``Interaction`` are named by half the conditions in
    # this corpus and the walk has nothing to say about either, so keeping
    # them costs one unexplained step per mention and buries the rest.
    roots = _tree(
        tmp_path,
        commando='''\
from pandajedi.jedicore import Interaction


class Worker:
    def runImpl(self):
        if tmpStat == Interaction.SC_SUCCEEDED:
            spec.status = "passed"
''',
    )

    steps, _ = _walk(roots, file="commando", owner="runImpl", lines=[7])

    assert "Interaction" not in {s.name for s in steps}
    assert "tmpStat" in {s.name for s in steps}


def test_a_call_handover_carries_a_parameter_back_to_the_caller(tmp_path):
    # The two kinds of handover are keyed differently -- an attribute of the
    # worker for a dispatch, a parameter for a call -- and both are the same
    # crossing as far as the walk is concerned.
    roots = _tree(
        tmp_path,
        refiner='''\
class Refiner:
    def apply(self, new_status):
        spec.status = new_status
''',
    )

    steps, _ = _walk(
        roots,
        file="refiner",
        owner="apply",
        lines=[3],
        handovers=[
            Handover(
                entry=f"{PACKAGE}/refiner",
                via="apply",
                reached_by="call",
                fields={"new_status": "taskSpec.oldStatus"},
            )
        ],
    )

    crossed = [s for s in steps if s.kind == TRACE_HANDOVER]
    assert [(s.name, s.value) for s in crossed] == [("new_status", "taskSpec.oldStatus")]


def test_a_row_read_through_the_cursor_comes_from_the_database(tmp_path):
    # The commonest ending in the corpus, 120 steps of 4852.  Without this it
    # reads as "chosen by whoever called or built this", which is true of the
    # cursor object and false of the row.
    roots = _tree(
        tmp_path,
        proxy='''\
class DBProxy:
    def getTask(self):
        self.cur.execute(sql, varMap)
        res = self.cur.fetchone()
        spec.status = res[0]
''',
    )

    steps, _ = _walk(roots, file="proxy", owner="getTask", lines=[5])

    assert [(s.name, s.terminal) for s in steps if s.terminal] == [
        ("res", STOP_UPSTREAM)
    ]


def test_calling_a_method_on_self_is_not_reading_a_field(tmp_path):
    # The value is what the call returns; the method's body is a hop
    # downwards this walk does not take.  Counted as a field it produced
    # thirty steps saying a caller chose ``self.getClobObj``.
    roots = _tree(
        tmp_path,
        proxy='''\
class DBProxy:
    def getTask(self):
        spec.status = self.decode(blob)
''',
    )

    steps, _ = _walk(roots, file="proxy", owner="getTask", lines=[3])

    assert "self.decode" not in {s.name for s in steps}
    assert "blob" in {s.name for s in steps}


def test_a_caught_exception_names_what_the_body_could_raise(tmp_path):
    roots = _tree(
        tmp_path,
        feeder='''\
class Feeder:
    def run(self):
        try:
            self.ddmIF.getDatasetMetaData(name)
        except Exception as exc:
            spec.status = str(exc)
''',
    )

    steps, _ = _walk(roots, file="feeder", owner="run", lines=[6])

    caught = [s for s in steps if s.name == "exc"]
    assert [(s.terminal, s.detail) for s in caught] == [(STOP_EXTERNAL, "DDM / Rucio")]


def test_a_field_assigned_in_the_same_function_is_read_there(tmp_path):
    # The one the `holding` walkthrough turned up: the arm reads
    # ``self.job_status`` and the assignment is eleven lines above it, and the
    # walk handed the question to the constructor without ever looking.
    roots = _tree(
        tmp_path,
        adder='''\
class Adder:
    def process(self):
        if self.job.isCancelled():
            self.job_status = "failed"
        spec.status = self.job_status
''',
    )

    steps, _ = _walk(roots, file="adder", owner="process", lines=[5])

    assert [(s.kind, s.line, s.value) for s in steps if s.name == "self.job_status"] == [
        (TRACE_BINDING, 4, "'failed'")
    ]


def test_a_field_assigned_in_a_sibling_method_is_read_in_that_method(tmp_path):
    # Two claims in one: the sibling is found at all, and its guard is the
    # one that dominates *it*.  Reporting ``reset``'s ``if`` as though it had
    # dominated ``process``'s arm would be a lie with the shape of an answer.
    roots = _tree(
        tmp_path,
        adder='''\
class Adder:
    def reset(self, why):
        if why is None:
            self.job_status = "failed"

    def process(self):
        spec.status = self.job_status
''',
    )

    steps, _ = _walk(roots, file="adder", owner="process", lines=[7])

    bound = [s for s in steps if s.name == "self.job_status"]
    assert [(s.kind, s.line, s.value) for s in bound] == [(TRACE_BINDING, 4, "'failed'")]
    assert bound[0].owner.endswith("::reset")
    assert bound[0].guards == ["why is None"]


def test_a_field_written_in_several_methods_comes_back_as_several_steps(tmp_path):
    # ``self.jobs`` is written in three methods of ``setupper_atlas_plugin``
    # and which of them ran last is a fact about the run.  Folding them would
    # answer a question the text cannot settle.
    roots = _tree(
        tmp_path,
        setupper='''\
class Setupper:
    def run(self):
        self.jobs = fetched

    def correct(self):
        self.jobs = corrected

    def emit(self):
        spec.status = self.jobs
''',
    )

    steps, _ = _walk(roots, file="setupper", owner="emit", lines=[9])

    assert sorted(s.value for s in steps if s.name == "self.jobs") == [
        "corrected",
        "fetched",
    ]


def test_a_constructor_assignment_carries_the_question_to_the_call_site(tmp_path):
    # Stage three's reason for existing: ``self.job_status = job_status`` is
    # what turns a worker's attribute into a question about whoever built it.
    roots = _tree(
        tmp_path,
        adder='''\
class Adder:
    def __init__(self, job_status):
        self.job_status = job_status

    def process(self):
        spec.status = self.job_status
''',
    )

    steps, _ = _walk(roots, file="adder", owner="process", lines=[6])

    assert ("self.job_status", "job_status") in [(s.name, s.value) for s in steps]
    assert ("job_status", STOP_PARAMETER) in [(s.name, s.terminal) for s in steps]


def test_a_named_interface_is_still_the_interface_and_not_its_constructor(tmp_path):
    # Stage one stays first, and this is the shape that makes it load-bearing.
    # A field a *value* reads is settled by ``classify`` before the walk ever
    # asks for it -- ``self.cur.fetchall()`` terminates where it stands.  It
    # is a field a *guard* reads that arrives here with nothing decided, and
    # answering that one from its ``__init__`` would replace "this came from
    # the database" with "this came from an argument": true of the handle and
    # false of the value.  Measured over the corpus, this reaches the field
    # reading 13 times for ``self.cur``, twice for ``self.ddmIF`` and once for
    # ``self.taskBufferIF``.
    roots = _tree(
        tmp_path,
        proxy='''\
class DBProxy:
    def __init__(self, conn):
        self.cur = conn.cursor()

    def getTask(self):
        if self.cur.rowcount > 0:
            spec.status = "done"
''',
    )

    steps, _ = _walk(roots, file="proxy", owner="getTask", lines=[7])

    assert [(s.name, s.terminal) for s in steps if s.name == "self.cur"] == [
        ("self.cur", STOP_UPSTREAM)
    ]
    assert not [s for s in steps if s.value == "conn.cursor()"]


def test_a_field_nothing_assigns_still_stops_rather_than_guessing(tmp_path):
    # The fallback has to survive the three stages above it, or "the map names
    # no handover for this" becomes unreachable and a genuinely unexplained
    # field reads as an empty answer.
    roots = _tree(
        tmp_path,
        commando='''\
class Worker:
    def runImpl(self):
        spec.status = self.mystery
''',
    )

    steps, _ = _walk(roots, file="commando", owner="runImpl", lines=[3])

    assert [(s.kind, s.terminal) for s in steps if s.name == "self.mystery"] == [
        (TRACE_UNBOUND, STOP_PARAMETER)
    ]


def test_a_field_a_closure_assigns_is_not_read_as_the_enclosing_guard(tmp_path):
    # A nested function runs when it is called, not where it is written, so
    # its guards never dominated the arm.  Left out rather than reported with
    # the wrong condition attached.
    roots = _tree(
        tmp_path,
        worker='''\
class Worker:
    def run(self):
        def later():
            self.flag = "set-later"
        self.flag = "set-here"
        spec.status = self.flag
''',
    )

    steps, _ = _walk(roots, file="worker", owner="run", lines=[6])

    assert [s.value for s in steps if s.name == "self.flag"] == ["'set-here'"]


def test_a_field_a_base_class_in_the_same_module_assigns_is_found(tmp_path):
    roots = _tree(
        tmp_path,
        plugin='''\
class Base:
    def prepare(self):
        self.mode = "scouting"

class Impl(Base):
    def run(self):
        spec.status = self.mode
''',
    )

    steps, _ = _walk(roots, file="plugin", owner="run", lines=[7])

    assert [s.value for s in steps if s.name == "self.mode"] == ["'scouting'"]


def test_a_name_bound_by_a_with_statement_is_not_called_unbound(tmp_path):
    # ``with self.proxyPool.get() as proxy:`` is how every TaskBuffer method
    # reaches the database, and reaching definitions do not cover the form, so
    # the walk used to say the function does not bind a name it plainly binds.
    roots = _tree(
        tmp_path,
        buffer='''\
class Buffer:
    def store(self):
        with self.pool.get() as proxy:
            spec.status = proxy
''',
    )

    steps, _ = _walk(roots, file="buffer", owner="store", lines=[4])

    held = [s for s in steps if s.name == "proxy"]
    assert [s.kind for s in held] == [TRACE_BINDING]
    assert held[0].value == "self.pool.get()"


def test_a_name_bound_by_a_walrus_is_not_called_unbound(tmp_path):
    roots = _tree(
        tmp_path,
        feeder='''\
class Feeder:
    def run(self):
        if (staging := self.count_staging()) > 0:
            spec.status = staging
''',
    )

    steps, _ = _walk(roots, file="feeder", owner="run", lines=[4])

    held = [s for s in steps if s.name == "staging"]
    assert [s.kind for s in held] == [TRACE_BINDING]
    assert held[0].value == "self.count_staging()"


def test_a_module_level_constant_is_named_rather_than_stopped_at(tmp_path):
    # A fact reachable forward from the anchor belongs to the trace, and a
    # constant in the same file is as reachable as one a line above the arm.
    roots = _tree(
        tmp_path,
        broker='''\
SKIP_TYPES = ["prod_test"]

class Broker:
    def schedule(self):
        spec.status = SKIP_TYPES
''',
    )

    steps, _ = _walk(roots, file="broker", owner="schedule", lines=[5])

    held = [s for s in steps if s.name == "SKIP_TYPES"]
    assert [s.kind for s in held] == [TRACE_BINDING]
    assert held[0].value == "['prod_test']"


def test_a_name_nothing_in_the_module_binds_still_stops(tmp_path):
    roots = _tree(
        tmp_path,
        broker='''\
class Broker:
    def schedule(self):
        spec.status = mystery
''',
    )

    steps, _ = _walk(roots, file="broker", owner="schedule", lines=[3])

    assert [s.kind for s in steps if s.name == "mystery"] == [TRACE_UNBOUND]


def test_a_binding_from_a_nested_function_says_it_may_not_run_here(tmp_path):
    # ``assigned_expressions`` is shared with the build and walks into nested
    # defs, so the row is kept -- for a daemon whose work lives in inner
    # functions it is the only answer -- but it is not this function's own.
    roots = _tree(
        tmp_path,
        daemon='''\
class Daemon:
    def main(self):
        def later():
            end_status = "deleted"
            spec.status = end_status
''',
    )

    steps, _ = _walk(roots, file="daemon", owner="main", lines=[5])

    held = [s for s in steps if s.name == "end_status"]
    assert [s.value for s in held] == ["'deleted'"]
    assert any("nested-def" in entry for entry in held[0].unseen)


def test_a_binding_from_the_functions_own_body_is_not_marked(tmp_path):
    roots = _tree(
        tmp_path,
        daemon='''\
class Daemon:
    def main(self):
        end_status = "deleted"
        spec.status = end_status
''',
    )

    steps, _ = _walk(roots, file="daemon", owner="main", lines=[4])

    held = [s for s in steps if s.name == "end_status"]
    assert [s.value for s in held] == ["'deleted'"]
    assert not any("nested-def" in entry for entry in held[0].unseen)


# ---------------------------------------------------------------------------
# The line production prints for this arm
# ---------------------------------------------------------------------------


def _skeleton(roots, *, file, owner, lines, **kwargs):
    return _walked(roots, file=file, owner=owner, lines=lines, **kwargs).skeleton


def _printed(roots, *, file, owner, lines, **kwargs):
    return [
        row
        for row in _skeleton(roots, file=file, owner=owner, lines=lines, **kwargs)
        if row.kind == models.SKELETON_PRINT
    ]


def test_the_line_is_anchored_either_side_of_the_hole_the_value_fills(tmp_path):
    """What the map's shared sentence cannot do.

    The pooled question this replaced anchored on the literal before the
    *first* hole, whichever hole that is.  Here the value goes in the second,
    and for 80 of the 182 subjects that got a question at all the first was
    the wrong one.  ``_value_hole`` is where that behaviour lives now.
    """
    roots = _tree(
        tmp_path,
        refiner='''\
class Refiner:
    def apply(self, taskSpec):
        taskSpec.status = "exhausted"
        logger.debug(f"task {taskSpec.taskID} set to {taskSpec.status} by the goal check")
''',
    )

    (row,) = _printed(
        roots, file="refiner", owner="apply", lines=[3], observed="exhausted"
    )

    assert row.line == 4
    assert row.arms == [3]
    assert row.hole == "taskSpec.status"
    assert row.because == "the arm writes it"
    assert row.value == r"task\ [^\ ]*\ set\ to\ exhausted\ by\ the\ goal\ check"
    assert row.pattern == r"task\ [^\ ]*\ set\ to\ [^\ ]*\ by\ the\ goal\ check"


def test_a_hole_is_closed_against_the_literal_that_follows_it(tmp_path):
    """The defect this round opens with.

    ``set\\ .*=None`` matched ``set task_status=pending oldTask=False ...``:
    the run crossed the field name, its value and two more fields to reach an
    ``=None`` belonging to nothing in the statement.  The literal after the
    hole says where the hole has to stop.
    """
    roots = _tree(
        tmp_path,
        proxy='''\
class Proxy:
    def reassign(self, spec):
        spec.cloud = None
        logger.debug(f"reassigning to set {target}={value} for this task now")
''',
    )

    (row,) = _printed(roots, file="proxy", owner="reassign", lines=[3], observed="None")

    assert row.pattern == r"reassigning\ to\ set\ [^=]*=[^\ ]*\ for\ this\ task\ now"
    assert ".*" not in row.pattern
    assert not re.search(row.pattern, "set task_status=pending oldTask=False with (True, 1)")


def test_a_value_with_no_literal_after_it_gets_a_right_hand_anchor(tmp_path):
    """``newPrio=100`` matched ``newPrio=1000``.

    With nothing after the hole there is no literal to say the value ended,
    and a question that cannot miss is read by the eliminator as one that was
    answered.
    """
    roots = _tree(
        tmp_path,
        proxy='''\
class Proxy:
    def bump(self, spec):
        spec.currentPriority = newPrio
        logger.debug(f"raising the task priority to newPrio={newPrio}")
''',
    )

    (row,) = _printed(roots, file="proxy", owner="bump", lines=[3], observed="100")

    assert row.value.endswith(r"newPrio=100(?![\w.-])")
    assert re.search(row.value, "raising the task priority to newPrio=100")
    assert not re.search(row.value, "raising the task priority to newPrio=1000")


def test_a_message_with_too_little_literal_is_rendered_and_marked(tmp_path):
    """Marked rather than refused: a question only with the entity's key.

    ``to`` is two characters of literal and matches half a log, and among one
    job's lines it is exactly the question -- ``Try:`` is what tells a
    temporary failure from a fatal one in ``register_files``.  The reader
    always narrows by the key, so the row says that and gives the pattern.
    """
    roots = _tree(
        tmp_path,
        proxy='''\
class Proxy:
    def run(self, spec):
        spec.status = "deleting"
        logger.debug(f"to {spec.status}")
''',
    )

    (row,) = _printed(roots, file="proxy", owner="run", lines=[3], observed="deleting")

    assert row.pattern == r"to\ "
    assert row.keyed is True
    assert row.refused == ""
    assert re.search(row.value, "to deleting")


def test_a_message_with_no_literal_at_all_is_still_refused(tmp_path):
    """Nothing to ask with, and said rather than left blank: a row that
    silently lost its pattern reads as *production does not print here*."""
    roots = _tree(
        tmp_path,
        proxy='''\
class Proxy:
    def run(self, spec):
        spec.status = "deleting"
        logger.debug(f"{spec.status}")
''',
    )

    (row,) = _printed(roots, file="proxy", owner="run", lines=[3], observed="deleting")

    assert row.pattern == ""
    assert row.refused == "no literal text to ask with"


def test_a_hole_printing_a_dict_may_hold_spaces(tmp_path):
    """``registerFilesInDatasets {str(dest_id_map)} zip=`` prints a dict, and a
    hole closed at the first space never reached ``zip=``: none of the job's
    lines matched."""
    roots = _tree(
        tmp_path,
        adder='''\
class Adder:
    def register(self, dest_id_map, cont_zip_map):
        self.result = 1
        self.logger.debug(f"registerFilesInDatasets {str(dest_id_map)} zip={str(cont_zip_map)}")
''',
    )

    (row,) = _printed(roots, file="adder", owner="register", lines=[3])

    line = "7338795608 registerFilesInDatasets {'UNI': {'ds_sub1': [{'guid': 'F3'}]}} zip={}"
    assert re.search(row.pattern, line)
    assert not row.keyed


def test_a_number_and_a_fragment_the_function_wrote_out_are_matched(tmp_path):
    """``reg_msg_str`` is one of two f-strings and the line prints either.
    Written out, each is a variant; the ``:03d`` hole is a number."""
    roots = _tree(
        tmp_path,
        adder='''\
class Adder:
    def register(self, reg_num_files, reg_time):
        if self.add_to_top_only:
            reg_msg_str = f"File registration for {reg_num_files} files "
        else:
            reg_msg_str = f"File registration with rucio for {reg_num_files} files "
        self.result = 1
        self.logger.debug(f"{reg_msg_str} took {reg_time.seconds}.{reg_time.microseconds // 1000:03d} sec")
''',
    )

    rows = _printed(roots, file="adder", owner="register", lines=[7])

    line = "7338795608 File registration with rucio for 2 files  took 0.047 sec"
    assert len(rows) == 2
    assert [bool(re.search(row.pattern, line)) for row in rows] == [False, True]
    assert not any(row.keyed for row in rows)


def test_the_line_the_arm_writes_the_value_of_is_matched_too(tmp_path):
    """An assignment is one fact said from two ends, and production reports
    either end about as often."""
    roots = _tree(
        tmp_path,
        refiner='''\
class Refiner:
    def apply(self, taskSpec):
        newStatus = "exhausted"
        taskSpec.status = newStatus
        logger.debug(f"moving the task over to {newStatus} right now")
''',
    )

    (row,) = _printed(
        roots, file="refiner", owner="apply", lines=[4], observed="exhausted"
    )

    assert row.hole == "newStatus"
    assert row.value == r"moving\ the\ task\ over\ to\ exhausted\ right\ now"


def test_a_line_from_a_branch_the_arm_excludes_is_not_reachable_from_it(tmp_path):
    """The ``else`` of the arm's own ``if`` cannot have printed for this arm.

    Both lines are in the skeleton, because both are lines this function
    prints and a reader laying a region against it needs to see both.  What
    the arm's exclusion settles is which of them can have been printed
    *alongside the arm*, and that is what ``arms`` carries.
    """
    roots = _tree(
        tmp_path,
        refiner='''\
class Refiner:
    def apply(self, taskSpec):
        if taskSpec.useJumbo:
            taskSpec.status = "exhausted"
            logger.debug(f"jumbo task has been set to {taskSpec.status} here")
        else:
            taskSpec.status = "exhausted"
            logger.debug(f"plain task has been set to {taskSpec.status} here")
''',
    )

    rows = _printed(roots, file="refiner", owner="apply", lines=[4], observed="exhausted")

    jumbo = next(row for row in rows if "jumbo" in row.text)
    plain = next(row for row in rows if "plain" in row.text)
    assert jumbo.arms == [4]
    assert plain.arms == []


def test_a_line_before_a_continue_the_arm_is_past_is_not_reachable_from_it(tmp_path):
    """The exclusion a path condition alone cannot see.

    ``if X: ...; continue`` puts ``not (X)`` on everything after it, and only
    :func:`pathcond.enclosing_guards` knows that -- without it the skipped
    block's line reads as compatible with the write below, and the arm gets
    handed a sentence printed on the path it did not take.  Found in the
    corpus, in ``setTobeDeletedToDis``.
    """
    roots = _tree(
        tmp_path,
        proxy='''\
class Proxy:
    def setTobeDeleted(self, dsList):
        for tmpDS in dsList:
            if tmpDS.status == 'deleting':
                logger.debug(f"skipping {tmpDS.name} since its status={tmpDS.status}")
                continue
            tmpDS.status = 'deleting'
            logger.debug(f"now setting the status to {tmpDS.status} for {tmpDS.name}")
''',
    )

    rows = _printed(
        roots, file="proxy", owner="setTobeDeleted", lines=[7], observed="deleting"
    )

    skipped = next(row for row in rows if "skipping" in row.text)
    written = next(row for row in rows if "now setting" in row.text)
    assert skipped.arms == []
    assert written.arms == [7]


def test_a_hole_with_no_literal_beside_it_is_not_an_anchor(tmp_path):
    """``.*deleting.*`` matches every line in the file that says the word, and
    a question that cannot miss is read by the eliminator as one that was
    answered."""
    roots = _tree(
        tmp_path,
        proxy='''\
class Proxy:
    def run(self, spec):
        spec.status = "deleting"
        logger.debug(f"the dataset in question here{spec.name}{spec.status}")
''',
    )

    (row,) = _printed(roots, file="proxy", owner="run", lines=[3], observed="deleting")

    assert row.value == ""
    assert row.pattern == r"the\ dataset\ in\ question\ here"


def test_a_hole_reaching_the_value_is_used_only_where_the_arm_names_none(tmp_path):
    """Second best, and said so: a hole the arm names is a fact about this
    write, where one merely holding the value is a fact about the function."""
    roots = _tree(
        tmp_path,
        proxy='''\
class Proxy:
    def run(self, spec):
        newStatus = "deleting"
        spec.status = self.decide()
        logger.debug(f"we are about to move it over to {newStatus} now")
''',
    )

    (row,) = _printed(roots, file="proxy", owner="run", lines=[4], observed="deleting")

    assert row.hole == "newStatus"
    assert row.because == "newStatus is set to this value above"


def test_the_arms_own_hole_wins_over_one_that_merely_holds_the_value(tmp_path):
    roots = _tree(
        tmp_path,
        proxy='''\
class Proxy:
    def run(self, spec):
        newStatus = "deleting"
        spec.status = newStatus
        logger.debug(f"we are planning to use {newStatus} very shortly")
        logger.debug(f"we have now moved it to {spec.status} at long last")
''',
    )

    rows = _printed(roots, file="proxy", owner="run", lines=[4], observed="deleting")

    assert [row.because for row in rows] == ["the arm writes it", "the arm writes it"]
    assert sorted(row.hole for row in rows) == ["newStatus", "spec.status"]


def test_no_observed_value_means_no_value_in_any_hole(tmp_path):
    """Half the question is the value, and inventing one would put a pattern in
    front of a reader that nothing in the record supports.

    The skeleton still comes back.  A line that reports nothing about the
    value still says the code around it ran, which is the whole reason this
    round stopped requiring every row to name one.
    """
    roots = _tree(
        tmp_path,
        refiner='''\
class Refiner:
    def apply(self, taskSpec):
        taskSpec.status = "exhausted"
        logger.debug(f"the task has been set to {taskSpec.status} by the goal check")
''',
    )

    (row,) = _printed(roots, file="refiner", owner="apply", lines=[3])

    assert row.value == ""
    assert row.pattern == r"the\ task\ has\ been\ set\ to\ [^\ ]*\ by\ the\ goal\ check"


def test_a_tree_that_is_not_the_maps_has_no_skeleton(tmp_path):
    """The same refusal the steps get, for the same reason: a plausible line
    computed from the wrong release reads exactly like an answer."""
    roots = _tree(
        tmp_path,
        refiner='''\
class Refiner:
    def apply(self, taskSpec):
        taskSpec.status = "exhausted"
        logger.debug(f"the task has been set to {taskSpec.status} by the goal check")
''',
    )

    walked = _walked(
        roots,
        file="refiner",
        owner="apply",
        lines=[3],
        observed="exhausted",
        expected_sha="0" * 40,
    )

    assert walked.skeleton == []
    assert walked.note == "the walk was not run: this tree is not the one that was mapped"


def test_a_region_cuts_the_skeleton_to_one_statements_stretch(tmp_path):
    """Three queries in one function are three answers, not one.

    ``copyArchive.py::main`` is 1353 lines and runs the ``holding`` query
    three times with three different time limits; handed the whole function
    all three come back with the same rows, and the reader cannot tell which
    limit rejected theirs.  The region is what the read side records to say
    which stretch is whose.
    """
    roots = _tree(
        tmp_path,
        sweeper='''\
class Sweeper:
    def run(self):
        limit = self.first_limit()
        rows = self.query(limit)
        logger.debug("first sweep found %s", rows)
        limit = self.second_limit()
        rows = self.query(limit)
        logger.debug("second sweep found %s", rows)
''',
    )

    whole = _walked(roots, file="sweeper", owner="run", lines=[3])
    first = _walked(roots, file="sweeper", owner="run", lines=[3], region=(3, 5))
    second = _walked(roots, file="sweeper", owner="run", lines=[6], region=(6, 8))

    printed = [row.line for row in whole.skeleton if row.kind == models.SKELETON_PRINT]
    assert printed == [5, 8]
    assert [row.line for row in first.skeleton if row.kind == models.SKELETON_PRINT] == [5]
    assert [row.line for row in second.skeleton if row.kind == models.SKELETON_PRINT] == [8]


def test_an_exits_guard_is_the_block_it_is_taken_in(tmp_path):
    """What the code says when a row stops here, without the branch it did not take.

    Cutting the statement's region at the exit line instead reads the other
    branch back to the reader: ``copyArchive.py``'s ``holding`` query logs one
    line when nothing came back and another when something did, and a reader
    told to grep for both has been told something false about one of them.
    """
    roots = _tree(
        tmp_path,
        sweeper="""\
class Sweeper:
    def run(self):
        rows = self.query()
        if rows is None:
            logger.debug("nothing came back")
            return None
        logger.debug("found %s", rows)
        for row in rows:
            if row in self.skipped:
                logger.debug("already seen -> skip %s", row)
                continue
            self.hand_on(row)
""",
    )
    seen = {"file": f"{PACKAGE}/sweeper", "owner": f"{PACKAGE}/sweeper::run"}

    early = trace_mod.exit_guard(roots, line=6, **seen)
    skipped = trace_mod.exit_guard(roots, line=11, **seen)

    # Decided at the ``if``, and the block is the two lines it guards.
    assert early == (4, 5, 6)
    assert skipped == (9, 10, 11)


def test_an_exits_guard_carries_only_what_that_block_prints(tmp_path):
    """The lines a reader can actually grep for, and silence when there are none."""
    roots = _tree(
        tmp_path,
        sweeper="""\
class Sweeper:
    def run(self):
        rows = self.query()
        if rows is None:
            return None
        logger.debug("found %s", rows)
        for row in rows:
            if row in self.skipped:
                logger.debug("already seen -> skip %s", row)
                continue
""",
    )

    quiet = _walked(roots, file="sweeper", owner="run", lines=[4], region=(4, 5))
    noisy = _walked(roots, file="sweeper", owner="run", lines=[8], region=(9, 10))

    # The early return prints nothing at all -- the finding, not a gap: a grep
    # of the log cannot tell this exit from the row never arriving.
    assert [row for row in quiet.skeleton if row.kind == models.SKELETON_PRINT] == []
    assert [
        row.line for row in noisy.skeleton if row.kind == models.SKELETON_PRINT
    ] == [9]


def test_an_except_handler_is_addressed_by_its_own_line(tmp_path):
    """``ExceptHandler`` is not an ``ast.stmt``, so it is looked for by hand.

    Reached through the statement walk it is invisible, which over the corpus
    read as zero ``except`` exits where there are eight.
    """
    roots = _tree(
        tmp_path,
        sweeper="""\
class Sweeper:
    def run(self):
        rows = self.query()
        for row in rows:
            try:
                self.hand_on(row)
            except Exception:
                logger.error("handing on failed")
                continue
""",
    )

    guard = trace_mod.exit_guard(
        roots, file=f"{PACKAGE}/sweeper", owner=f"{PACKAGE}/sweeper::run", line=7
    )

    assert guard == (7, 8, 9)


def test_a_name_bound_in_the_stretch_is_explained_from_there_only(tmp_path):
    """One function, three queries, one name -- and the walk offered all three.

    ``copyArchive.py::main`` assigns ``res = taskBuffer.querySQLS(...)`` nine
    times in 1353 lines, so the exit taken on line 390 came back explained by
    the assignments on lines 263 and 284: other queries, whose rows the row
    that stopped here never was.
    """
    roots = _tree(
        tmp_path,
        sweeper="""\
class Sweeper:
    def run(self):
        limit = self.first_limit()
        self.handle(limit)
        limit = self.second_limit()
        for row in self.query():
            row.status = limit
""",
    )

    whole, _note = _walk(roots, file="sweeper", owner="run", lines=[7])
    ours, _note = _walk(
        roots, file="sweeper", owner="run", lines=[7], bindings_within=(5, 7)
    )

    assert [step.line for step in whole if step.kind == models.TRACE_BINDING] == [3, 5]
    assert [step.line for step in ours if step.kind == models.TRACE_BINDING] == [5]


def test_a_name_bound_nowhere_in_the_stretch_still_gets_the_function(tmp_path):
    """Inside *or* nothing, not inside and then the rest.

    A name the statement did not fill is one the whole function is the right
    place to look for: above the flagship exit, ``xmlIDs`` is built far from
    the query and still explains it.
    """
    roots = _tree(
        tmp_path,
        sweeper="""\
class Sweeper:
    def run(self):
        skipped = self.load_skips()
        for row in self.query():
            row.status = skipped
""",
    )

    ours, _note = _walk(
        roots, file="sweeper", owner="run", lines=[5], bindings_within=(4, 5)
    )

    assert [step.line for step in ours if step.kind == models.TRACE_BINDING] == [3]


def test_a_header_above_a_shown_line_survives_the_region(tmp_path):
    """The nesting is what a reader is here for, so it is not cut by its line.

    A header outside the region stays when it encloses something inside it,
    and goes when it does not -- which is the same rule the skeleton already
    uses to keep itself from being the source again.
    """
    roots = _tree(
        tmp_path,
        guarded='''\
class Guarded:
    def run(self, spec):
        if spec.enabled:
            limit = self.limit()
            logger.debug("inside found %s", limit)
        logger.debug("outside")
''',
    )

    walked = _walked(roots, file="guarded", owner="run", lines=[4], region=(4, 5))

    kinds = [(row.kind, row.line) for row in walked.skeleton]
    assert (models.SKELETON_PRINT, 5) in kinds
    assert (models.SKELETON_PRINT, 6) not in kinds
    # The ``if`` above line 5 is outside the region and still shown.
    assert any(line == 3 for _kind, line in kinds)


def test_a_subtree_with_no_line_and_no_arm_is_left_out(tmp_path):
    """A skeleton is not the source again.

    A 540-line function whose every branch is reproduced buries the handful of
    rows a reader came for, which is the failure this round is elsewhere
    fixing in the report.
    """
    roots = _tree(
        tmp_path,
        proxy='''\
class Proxy:
    def run(self, spec):
        if spec.quiet:
            spec.counter += 1
            spec.other = self.decide()
        if spec.loud:
            spec.status = "deleting"
            logger.debug(f"we have now set the status to {spec.status} here")
''',
    )

    rows = _skeleton(roots, file="proxy", owner="run", lines=[7], observed="deleting")

    assert [row.text for row in rows if row.kind == models.SKELETON_BRANCH] == [
        "if spec.loud:"
    ]
    assert [row.line for row in rows if row.kind == models.SKELETON_ARM] == [7]


def test_the_nesting_says_which_rows_could_have_been_printed_together(tmp_path):
    """The part a flat list of lines cannot carry.

    Two rows under one ``if`` were printed together or not at all; two rows
    either side of an ``else`` cannot both have been.  A reader with a grepped
    region reads that off the indentation, so the headers are the artefact and
    not decoration.
    """
    roots = _tree(
        tmp_path,
        proxy='''\
class Proxy:
    def run(self, spec):
        try:
            if spec.loud:
                logger.debug("about to start working on this dataset")
                spec.status = "deleting"
        except Exception:
            logger.debug("failed while working on this dataset here")
''',
    )

    rows = _skeleton(roots, file="proxy", owner="run", lines=[6], observed="deleting")

    shape = [
        (row.kind, row.text if row.kind == models.SKELETON_BRANCH else row.line, row.depth)
        for row in rows
    ]
    assert shape == [
        (models.SKELETON_BRANCH, "try:", 0),
        (models.SKELETON_BRANCH, "if spec.loud:", 1),
        (models.SKELETON_PRINT, 5, 2),
        (models.SKELETON_ARM, 6, 2),
        (models.SKELETON_BRANCH, "except Exception:", 0),
        (models.SKELETON_PRINT, 8, 1),
    ]


def test_an_assignment_the_call_cannot_reach_is_not_one_of_its_messages(tmp_path):
    """``_logged_arguments`` hands back every assignment to the name.

    ``log.debug(msg_str)`` resolved one hop through a local gets all five
    assignments to ``msg_str`` in the function, and four of them are in
    branches the call cannot be reached from.  The build cannot narrow it --
    a stored answer has to hold for every caller -- but at use time it can:
    an assignment below the call cannot have run before it, and one the
    call's own path condition contradicts cannot have run on the way to it.
    """
    roots = _tree(
        tmp_path,
        proxy='''\
class Proxy:
    def run(self, spec):
        if spec.loud:
            msg_str = "the first branch decided to stop here"
        else:
            msg_str = "the second branch decided to stop here"
        logger.debug(msg_str)
        spec.status = "deleting"
        msg_str = "this one is assigned below the call above"
''',
    )

    rows = _printed(roots, file="proxy", owner="run", lines=[8], observed="deleting")

    assert [row.pattern for row in rows] == [
        r"the\ first\ branch\ decided\ to\ stop\ here",
        r"the\ second\ branch\ decided\ to\ stop\ here",
    ]



def test_a_shown_block_keeps_the_header_that_explains_it(tmp_path):
    """An ``except`` with no ``try`` over it, and an ``else`` with no condition.

    Dropping the headers of blocks that hold nothing produced both, and the
    nesting is the thing a reader is here for -- a header that explains a
    shown block is not decoration.  The rule that a statement holding neither
    a line nor an arm is left out entirely still applies; this is about the
    blocks *inside* one that is shown.
    """
    roots = _tree(
        tmp_path,
        proxy='''\
class Proxy:
    def run(self, spec):
        try:
            spec.counter += 1
        except Exception:
            logger.debug("failed while working on this dataset here")
        if spec.quiet:
            spec.other = self.decide()
        else:
            spec.status = "deleting"
''',
    )

    rows = _skeleton(roots, file="proxy", owner="run", lines=[10], observed="deleting")

    assert [row.text for row in rows if row.kind == models.SKELETON_BRANCH] == [
        "try:",
        "except Exception:",
        "if spec.quiet:",
        "else:",
    ]


# ---------------------------------------------------------------------------
# What a site has to be able to reach
# ---------------------------------------------------------------------------


def test_an_arm_that_is_a_test_follows_the_names_it_tests(tmp_path):
    """An exit is armed at the ``if`` that decides it, and that ``if`` has no value.

    ``_value_of`` answers for assignments; ``_guard_reads`` answers for the
    tests *above* a statement and correctly leaves the statement's own out.
    Between them the one name that decides an exit went unfollowed --
    ``copyArchive.py``'s ``if int(id) in xmlIDs: continue`` was explained by
    the rows it was iterating and never by ``xmlIDs``.
    """
    roots = _tree(
        tmp_path,
        sweeper="""\
class Sweeper:
    def run(self):
        skipped = self.already_done()
        for row in self.query():
            if row in skipped:
                continue
""",
    )

    steps, _note = _walk(roots, file="sweeper", owner="run", lines=[5])

    assert "skipped" in steps[0].reads
    assert ("skipped", 3) in [(step.name, step.line) for step in steps]


def test_a_container_filled_by_add_answers_with_what_went_into_it(tmp_path):
    """``xmlIDs = set()`` is honest and useless; the ``.add`` is the answer.

    A name reached only this way is assigned exactly once, to an empty
    container, so stopping at the assignment tells a reader nothing about why
    their row was skipped.  Twelve names in the corpus are reached this way.
    """
    roots = _tree(
        tmp_path,
        sweeper="""\
class Sweeper:
    def run(self):
        skipped = set()
        for report in self.reports():
            skipped.add(int(report.id))
        for row in self.query():
            if row in skipped:
                continue
""",
    )

    steps, _note = _walk(roots, file="sweeper", owner="run", lines=[7])

    bound = [(step.name, step.line) for step in steps if step.kind == models.TRACE_BINDING]
    assert ("skipped", 5) in bound
    # And the walk goes on from it: what was added is the question's next hop.
    assert any(step.name == "report" for step in steps)


def test_a_container_the_code_also_assigns_keeps_the_assignment(tmp_path):
    """The members are offered only where the assignments leave the reader nowhere.

    A name given a value has that value to answer with, and listing every
    ``append`` beside it would bury it.
    """
    roots = _tree(
        tmp_path,
        sweeper="""\
class Sweeper:
    def run(self):
        skipped = self.already_done()
        skipped.add(9)
        for row in self.query():
            if row in skipped:
                continue
""",
    )

    steps, _note = _walk(roots, file="sweeper", owner="run", lines=[6])

    bound = [(step.name, step.line) for step in steps if step.kind == models.TRACE_BINDING]
    assert ("skipped", 3) in bound
    assert ("skipped", 4) not in bound


def test_a_key_named_with_no_default_is_not_a_value_the_container_holds(tmp_path):
    """``d.setdefault(key)``'s last argument is the key, not what is in ``d``."""
    roots = _tree(
        tmp_path,
        sweeper="""\
class Sweeper:
    def run(self):
        seen = {}
        for row in self.query():
            seen.setdefault(row.site)
        if seen:
            return None
""",
    )

    steps, _note = _walk(roots, file="sweeper", owner="run", lines=[6])

    assert [(step.name, step.line) for step in steps if step.kind == models.TRACE_BINDING] == [
        ("seen", 3)
    ]


def test_a_binding_below_the_arm_is_not_offered_as_an_explanation(tmp_path):
    """Pruning is by elimination, so a site that cannot have run on the way
    does not merely add noise -- it dilutes "one of these two" into "one of
    these three" and sends a reader down a path the code did not take."""
    roots = _tree(
        tmp_path,
        m="""
def f(spec):
    status = "new"
    if spec.kind == "merge":
        status = "merging"
    spec.status = status
    status = "done"
    return status
""",
    )

    steps, _note = _walk(roots, file="m", owner="f", lines=[6])

    listed = {(step.line, step.value) for step in steps if step.name == "status"}
    assert listed == {(3, "'new'"), (5, "'merging'")}


def test_a_binding_below_the_arm_inside_a_loop_that_holds_it_stays(tmp_path):
    """The next iteration reaches it.  Without this half the position rule is
    not a refinement but an error -- and it is the half ``reachable_calls`` has
    been missing since it was written."""
    roots = _tree(
        tmp_path,
        m="""
def f(items):
    token = "first"
    for item in items:
        spec.status = token
        token = item.next_token
""",
    )

    steps, _note = _walk(roots, file="m", owner="f", lines=[5])

    listed = {step.line for step in steps if step.name == "token"}
    assert listed == {3, 6}


def test_a_name_bound_only_below_the_arm_says_which(tmp_path):
    """"Not bound in this function" would be false and silence would leave the
    guard unexplained, so the step names the third possibility."""
    roots = _tree(
        tmp_path,
        m="""
def f(spec):
    if later == "x":
        spec.status = "done"
    later = "x"
""",
    )

    steps, _note = _walk(roots, file="m", owner="f", lines=[4])

    (step,) = [s for s in steps if s.name == "later"]
    assert step.kind == TRACE_UNBOUND
    assert "only below the arms" in step.detail
    assert step.terminal != STOP_PARAMETER


def test_a_message_assigned_at_the_foot_of_a_loop_still_reaches_its_call(tmp_path):
    """``reachable_calls`` dropped it on the plain position test, although
    every iteration but the first reaches it -- three rows came back over the
    41-case sample when the loop half was added."""
    roots = _tree(
        tmp_path,
        m="""
def f(items, log):
    message = "starting the first pass over the list"
    for item in items:
        log.debug(message)
        spec.status = "done"
        message = f"finished the pass over {item} in the list"
""",
    )

    walked = _walked(roots, file="m", owner="f", lines=[6])

    printed = [
        row.text
        for row in walked.skeleton
        if row.kind == models.SKELETON_PRINT and row.pattern
    ]
    assert any("finished the pass over" in one for one in printed), printed
    assert any("starting the first pass" in one for one in printed), printed


def test_a_message_the_walk_cannot_resolve_is_refused_rather_than_dropped(tmp_path):
    """A bare name bound only by an unpacking resolves to nothing, and the call
    used to leave no row at all -- where the same call holding an expression
    leaves a refusal.  A reader lays the skeleton over a grep of the log
    region, so a line there and not here reads as certainty."""
    roots = _tree(
        tmp_path,
        m="""
def f(api, log):
    out, err = api.list_datasets()
    if out is None:
        log.error(f"failed to list the datasets with {err}")
    else:
        log.debug(out)
        spec.status = "done"
""",
    )

    walked = _walked(roots, file="m", owner="f", lines=[8])

    refused = {
        row.line: row.refused
        for row in walked.skeleton
        if row.kind == models.SKELETON_PRINT and row.refused
    }
    assert refused == {7: trace_mod.REFUSED_RETURNED}


def test_an_unpacked_binding_says_which_slot_it_took(tmp_path):
    """``tmpStat, taskSpec = get(...)`` rendered as ``taskSpec = get(...)`` says
    the call returns the spec, when it returns a pair whose first element is
    the status deciding whether the second means anything."""
    roots = _tree(
        tmp_path,
        m="""
def f(db):
    status, spec, extra = db.get_task()
    job.state = spec
""",
    )

    steps, _note = _walk(roots, file="m", owner="f", lines=[4])

    (step,) = [s for s in steps if s.name == "spec"]
    assert step.value == "db.get_task()[1]"


def test_a_starred_target_leaves_the_slot_unsaid(tmp_path):
    """Where the position of everything after the star depends on how long the
    value is, no slot is better than a guessed one."""
    roots = _tree(
        tmp_path,
        m="""
def f(db):
    first, *rest, spec = db.get_task()
    job.state = spec
""",
    )

    steps, _note = _walk(roots, file="m", owner="f", lines=[4])

    (step,) = [s for s in steps if s.name == "spec"]
    assert step.value == "db.get_task()"


def test_a_later_unconditional_binding_cancels_the_one_before_it(tmp_path):
    """Three time limits, one query -- and the walk offered all three.

    ``copyArchive.py::main`` rebuilds ``timeLimit`` before each of its
    twenty-odd queries, so the query on line 381 came back explained by the
    assignments on 298 and 321 as well as its own on 360.  Only 360 can be
    what it read: the two above it were overwritten on the way.
    """
    roots = _tree(
        tmp_path,
        sweeper="""\
class Sweeper:
    def run(self):
        limit = self.first_limit()
        limit = self.second_limit()
        limit = self.third_limit()
        for row in self.query():
            row.status = limit
""",
    )

    whole, _note = _walk(roots, file="sweeper", owner="run", lines=[7])
    ours, _note = _walk(roots, file="sweeper", owner="run", lines=[7], reading_at=6)

    assert [step.line for step in whole if step.kind == models.TRACE_BINDING] == [3, 4, 5]
    assert [step.line for step in ours if step.kind == models.TRACE_BINDING] == [5]


def test_a_binding_under_a_condition_cancels_nothing(tmp_path):
    """Only a write that ran on *every* path to the read can overwrite one.

    An assignment in an ``if`` may not have happened, so the one above it is
    still an answer.  The rule is conservative in the direction that leaves
    the reader more to look at rather than less.
    """
    roots = _tree(
        tmp_path,
        sweeper="""\
class Sweeper:
    def run(self):
        limit = self.first_limit()
        if self.urgent():
            limit = self.second_limit()
        for row in self.query():
            row.status = limit
""",
    )

    ours, _note = _walk(roots, file="sweeper", owner="run", lines=[7], reading_at=6)

    assert [step.line for step in ours if step.kind == models.TRACE_BINDING] == [3, 5]


def test_a_write_in_an_except_body_cancels_nothing_outside_it(tmp_path):
    """A handler runs only when something raised, and the path condition cannot see it.

    ``reachable_when`` renders ``ast.If`` tests and nothing else, so an
    assignment in an ``except`` read as one that always ran::

        try:    nRanges = int(nRanges)
        except: nRanges = 8

    and ``getEventRanges`` was answered with "the request always asks for
    eight".  Uncounted conditions make the rendered set smaller and the
    subset test easier to pass, so this rule errs towards cancelling -- the
    opposite direction from the rendering mismatch above it.
    """
    roots = _tree(
        tmp_path,
        sweeper="""\
class Sweeper:
    def run(self):
        limit = self.first_limit()
        try:
            limit = self.parse(limit)
        except Exception:
            limit = 8
        for row in self.query():
            row.status = limit
""",
    )

    ours, _note = _walk(roots, file="sweeper", owner="run", lines=[9], reading_at=8)

    # 5 is in the ``try`` body, which runs unless something raised first, so
    # it does cancel 3 as an answer to line 8.  7 is the handler and cancels
    # neither.  3 is still walked -- one hop further, as what line 5 parses.
    answers = [s.line for s in ours if s.kind == models.TRACE_BINDING and s.depth == 1]
    assert answers == [5, 7]
    assert [(s.line, s.depth) for s in ours if s.line == 3] == [(3, 2)]


def test_a_write_in_the_same_except_body_still_cancels_the_one_above_it(tmp_path):
    """Both are live only where the handler ran, and there the later one won.

    Subtracting the *killed* write's handlers as well as the read's is what
    keeps this: ``getTasksToRefine_JEDI`` rebuilds ``varMap`` three times
    inside one ``except``, and a rule that looked only at the read would hand
    a reader all three.
    """
    roots = _tree(
        tmp_path,
        sweeper="""\
class Sweeper:
    def run(self):
        try:
            limit = self.parse()
        except Exception:
            limit = self.fallback()
            limit = self.floor(limit)
        for row in self.query():
            row.status = limit
""",
    )

    ours, _note = _walk(roots, file="sweeper", owner="run", lines=[9], reading_at=8)

    # 6 is not an answer to line 8 -- 7 replaced it -- and is walked one hop
    # further, as the ``limit`` that ``self.floor`` was given.
    answers = [s.line for s in ours if s.kind == models.TRACE_BINDING and s.depth == 1]
    assert answers == [4, 7]
    assert [(s.line, s.depth) for s in ours if s.line == 6] == [(6, 2)]


def test_cancelling_never_leaves_a_name_with_no_binding_at_all(tmp_path):
    """An empty answer would be the rule cancelling the write that ran.

    The two arms of an ``if`` cannot both run, so the only assignment to
    ``limit`` is on a path this read is not on -- ``pathcond.can_reach`` says
    no to all of them.  "Bound nowhere" would be false where the code plainly
    binds it, so the unfiltered set comes back instead.
    """
    roots = _tree(
        tmp_path,
        sweeper="""\
class Sweeper:
    def run(self, row):
        if self.urgent():
            limit = self.fast_limit()
        else:
            row.status = limit
""",
    )

    ours, _note = _walk(roots, file="sweeper", owner="run", lines=[6], reading_at=6)

    assert [step.line for step in ours if step.kind == models.TRACE_BINDING] == [4]


def test_a_compound_assignment_cancels_nothing_because_it_reads_first(tmp_path):
    """``timeOutVal *= 60`` needs whatever was there, so both writes stand.

    ``copyArchive.py`` reads a configured holding limit, falls back to
    forty-eight, and converts to minutes.  Taking the conversion for an
    overwrite left the map saying the limit was sixty minutes -- not a short
    answer but a wrong one, and the same shape dropped the ``SELECT`` from two
    SQL strings assembled with ``+=``.
    """
    roots = _tree(
        tmp_path,
        sweeper="""\
class Sweeper:
    def run(self):
        limit = self.configured_limit()
        if not limit:
            limit = 48
        limit *= 60
        for row in self.query():
            row.status = limit
""",
    )

    ours, _note = _walk(roots, file="sweeper", owner="run", lines=[8], reading_at=7)

    assert [step.line for step in ours if step.kind == models.TRACE_BINDING] == [3, 5, 6]


def test_a_plain_assignment_still_cancels_a_compound_one(tmp_path):
    """The exemption is one-way: what runs after needs nothing of what was there."""
    roots = _tree(
        tmp_path,
        sweeper="""\
class Sweeper:
    def run(self):
        limit = self.configured_limit()
        limit *= 60
        limit = self.override()
        for row in self.query():
            row.status = limit
""",
    )

    ours, _note = _walk(roots, file="sweeper", owner="run", lines=[7], reading_at=6)

    assert [step.line for step in ours if step.kind == models.TRACE_BINDING] == [5]


def test_a_name_that_reads_itself_is_followed_back_past_its_own_line(tmp_path):
    """``getScoutJobData_JEDI`` builds the scout's ``ramCount`` in five steps on
    one name -- a percentile, a margin, a unit, a cast, a floor -- and returns
    the last.

    Narrowed to the return, the walk kept the cast and the floor and cut
    the rest: ``mem = int(mem)`` reads the ``mem`` above it, and that read was
    measured against the return rather than against the cast, where the only
    write reaching is the one on the line before.  The percentile and both
    scout settings behind it went with it.
    """
    roots = _tree(
        tmp_path,
        scout="""\
class Scout:
    def run(self, sizes, margin, out):
        mem = self.percentile(sizes)
        mem = mem * (100 + margin) // 100
        mem /= 1024
        mem = int(mem)
        if mem < 0:
            mem = 1
        out["ram"] = mem
""",
    )

    whole, _note = _walk(roots, file="scout", owner="run", lines=[9])
    ours, _note = _walk(roots, file="scout", owner="run", lines=[9], reading_at=9)

    bound = [s.line for s in ours if s.kind == models.TRACE_BINDING and s.name == "mem"]
    assert sorted(bound) == [3, 4, 5, 6, 8]
    assert sorted(bound) == sorted(
        s.line for s in whole if s.kind == models.TRACE_BINDING and s.name == "mem"
    )
    assert "margin" in {s.name for s in ours if s.kind == models.TRACE_UNBOUND}
    # Asked four times, walked once each.
    assert len(bound) == len(set(bound))


def test_a_name_read_on_the_way_is_answered_where_it_was_read(tmp_path):
    """``y = t + 1`` reads ``t`` on line 4, and line 4 is where its writes are
    counted from -- not the line the walk started at.

    Counted from the start, the write on line 5 cancelled the one on line 3
    and was offered as the ``t`` that ``y`` was built from: a confident answer
    from a line that ran after the read.
    """
    roots = _tree(
        tmp_path,
        sweeper="""\
class Sweeper:
    def run(self, row):
        t = self.first()
        y = t + 1
        t = self.second()
        row.status = y
""",
    )

    ours, _note = _walk(roots, file="sweeper", owner="run", lines=[6], reading_at=6)

    assert [s.line for s in ours if s.kind == models.TRACE_BINDING and s.name == "t"] == [3]


def test_a_name_carried_round_a_loop_keeps_its_first_value_and_its_own(tmp_path):
    """``total = total + r`` in a loop reads the value before the loop on the
    first pass and its own on every other, and cancels neither."""
    roots = _tree(
        tmp_path,
        sweeper="""\
class Sweeper:
    def run(self, rows, row):
        total = 0
        for r in rows:
            total = total + r
        row.status = total
""",
    )

    ours, _note = _walk(roots, file="sweeper", owner="run", lines=[6], reading_at=6)

    bound = [s.line for s in ours if s.kind == models.TRACE_BINDING and s.name == "total"]
    assert sorted(bound) == [3, 5]


def test_an_add_keeps_what_the_container_held(tmp_path):
    """``seen.add(b)`` leaves ``a`` in the set, so it cancels nothing.

    The rule already said this of ``timeOutVal *= 60`` and not of a method
    that fills a container, and the members a narrowed walk answered with
    were the last one added.
    """
    roots = _tree(
        tmp_path,
        sweeper="""\
class Sweeper:
    def run(self, a, b, row):
        seen = set()
        seen.add(a)
        seen.add(b)
        row.status = seen
""",
    )

    ours, _note = _walk(roots, file="sweeper", owner="run", lines=[6], reading_at=6)

    bound = sorted(s.line for s in ours if s.kind == models.TRACE_BINDING and s.name == "seen")
    assert bound == [3, 4, 5]


def test_a_guards_names_are_read_where_the_if_reads_them(tmp_path):
    """``if x > 0`` read the ``x`` from line 3; the ``x = 0`` under it ran after.

    Counted from the line the guard dominates, the test was explained by the
    write inside its own body -- ``x = 0`` offered as what ``x > 0`` saw, from
    a line that had not run when the test did.  Nothing else here reads that
    ``x``, so it is no answer at all.
    """
    roots = _tree(
        tmp_path,
        sweeper="""\
class Sweeper:
    def run(self, row):
        x = self.first()
        if x > 0:
            x = 0
            y = 1
            row.status = y
""",
    )

    ours, _note = _walk(roots, file="sweeper", owner="run", lines=[7], reading_at=7)

    assert [s.line for s in ours if s.kind == models.TRACE_BINDING and s.name == "x"] == [3]


def test_a_test_on_a_name_rebound_since_does_not_keep_two_writes_apart(tmp_path):
    """``AtlasProdWatchDog.doActionForReassign`` skips a task on ``if tmpStat
    is not True: continue``, then sets ``tmpStat`` afresh per dataset and marks
    the task failed under ``if tmpStat is not True``.

    Read as text, everything after the skip is "not (tmpStat is not True)" and
    the failure "tmpStat is not True", so the two could not meet, and a walk
    narrowed to what reaches ``if isOK`` lost the ``isOK = False`` that says
    why the task was not reassigned.  The skip's test is about the first
    ``tmpStat``; by the failure it is another.
    """
    roots = _tree(
        tmp_path,
        dog="""\
class Dog:
    def run(self, tasks, sets):
        for task in tasks:
            stat = self.check(task)
            if stat is not True:
                continue
            ok = True
            for one in sets:
                stat = self.move(one)
                if stat is not True:
                    ok = False
            if ok:
                task.status = "done"
""",
    )

    ours, _note = _walk(roots, file="dog", owner="run", lines=[13], reading_at=13)

    assert [s.line for s in ours if s.kind == models.TRACE_BINDING and s.name == "ok"] == [7, 11]


def test_a_write_in_a_try_whose_handler_passes_leaves_the_one_above_standing(tmp_path):
    """``TaskRefiner.runImpl``: ``dataset_did = None``, then ``try: dataset_did
    = rucioAPI.get_did_str(...)`` with ``except Exception: pass``, then ``if
    dataset_did is not None``.

    Whenever Rucio raised, the test sees the ``None``.  Read as "the ``try``
    body runs unless something raised", the call's write replaced it, and the
    one path the test exists for was gone from the answer.
    """
    roots = _tree(
        tmp_path,
        refiner="""\
class Refiner:
    def run(self, name, out):
        did = None
        try:
            did = self.rucio.get_did_str(name)
        except Exception:
            pass
        if did is not None:
            out.append(did)
""",
    )

    ours, _note = _walk(roots, file="refiner", owner="run", lines=[9], reading_at=9)

    assert [s.line for s in ours if s.kind == models.TRACE_BINDING and s.name == "did"] == [3, 5]


def test_a_binding_says_how_its_value_reaches_the_name(tmp_path):
    """The step carries the right-hand side alone, so it has to carry the rest.

    Printed as ``limit = 60`` a reader takes sixty for the value, which is the
    one number the line does not produce.
    """
    roots = _tree(
        tmp_path,
        sweeper="""\
class Sweeper:
    def run(self):
        limit = self.configured_limit()
        limit *= 60
        for row in self.query():
            row.status = limit
""",
    )

    ours, _note = _walk(roots, file="sweeper", owner="run", lines=[6], reading_at=5)

    bound = {step.line: step for step in ours if step.kind == models.TRACE_BINDING}
    assert (bound[3].operator, bound[3].value) == ("=", "self.configured_limit()")
    assert (bound[4].operator, bound[4].value) == ("*=", "60")


def test_bindings_are_explained_in_the_order_the_file_states_them(tmp_path):
    """``ast.walk`` is breadth-first, and reading them in that order misleads.

    The fallback is one level in, so unsorted it comes back *after* the
    conversion that follows it and reads as if forty-eight were applied last.
    """
    roots = _tree(
        tmp_path,
        sweeper="""\
class Sweeper:
    def run(self):
        limit = self.configured_limit()
        if not limit:
            limit = 48
        limit *= 60
        for row in self.query():
            row.status = limit
""",
    )

    ours, _note = _walk(roots, file="sweeper", owner="run", lines=[8], reading_at=7)

    shown = [step.line for step in ours if step.kind == models.TRACE_BINDING]
    assert shown == sorted(shown)


ADDER = '''\
class Adder:
    def execute(self):
        src_site_spec = self.siteMapper.getSite(self.job.computingSite)
        scope = select_scope(src_site_spec, self.job.prodSourceLabel)
        tmp_src_ddm = src_site_spec.ddm_output[scope]
        task = self.taskBufferIF.getTaskWithID_JEDI(task_id)
        if tmp_src_ddm == task.ddm:
            pass
'''


def test_a_value_another_system_supplies_still_says_what_it_was_asked_for(tmp_path):
    """CRIC answered for ``job.computingSite``, and stopping at "CRIC" lost
    which site was asked -- the one difference between two functions deciding
    the same ``src == dst`` two ways.  The step keeps its terminal; the key is
    walked like any other name."""
    roots = _tree(tmp_path, adder=ADDER)

    steps, _note = _walk(roots, file="adder", owner="execute", lines=[7], reaching=True)

    by = {(s.name, s.line): s for s in steps}
    assert by[("tmp_src_ddm", 5)].detail == "CRIC site declaration"
    asked = by[("src_site_spec", 3)]
    assert asked.detail == "CRIC site declaration"
    assert asked.value == "self.siteMapper.getSite(self.job.computingSite)"
    assert ("scope", 4) in by
    assert any(s.name == "self.job" for s in steps)


def test_a_value_the_database_supplies_is_still_where_the_walk_stops(tmp_path):
    """The database's arguments flow into statements this map reads -- a call
    and its arguments, which is another change -- so ``task_id`` is not
    walked from here."""
    roots = _tree(tmp_path, adder=ADDER)

    steps, _note = _walk(roots, file="adder", owner="execute", lines=[7], reaching=True)

    assert any(s.name == "task" and s.detail == "getTaskWithID_JEDI" for s in steps)
    assert not any(s.name == "task_id" for s in steps)


_SITE_MAPPER = '''\
class SiteMapper:
    def getSite(self, name) -> "SiteSpec":
        return self.sites[name]
'''

_TASK_BUFFER = '''\
from pandaserver.brokerage.SiteMapper import SiteMapper


class TaskBuffer:
    def get_site_mapper(self) -> SiteMapper:
        return SiteMapper(self)
'''

_PLUGIN_BASE = '''\
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pandaserver.brokerage.SiteMapper import SiteMapper


class PluginBase:
    site_mapper: "SiteMapper"
'''

_PLUGIN = '''\
from pandaserver.brokerage.SiteMapper import SiteMapper
from pandaserver.dataservice.plugin_base import PluginBase
from pandaserver.taskbuffer.TaskBuffer import TaskBuffer


class Plugin(PluginBase):
    def by_field(self):
        spec = self.site_mapper.getSite(self.job.computingSite)
        return spec.ddm_output

    def by_constructor(self, tb):
        mapper = SiteMapper(tb)
        spec = mapper.getSite(self.job.computingSite)
        return spec.ddm_output

    def by_parameter(self, mapper: SiteMapper | None):
        spec = mapper.getSite(self.job.computingSite)
        return spec.ddm_output

    def by_return(self, tb: TaskBuffer):
        mapper = tb.get_site_mapper()
        spec = mapper.getSite(self.job.computingSite)
        return spec.ddm_output
'''

_LOOKALIKE = '''\
class SiteMapper:
    def getSite(self, name):
        return name


class Other:
    def run(self):
        mapper = SiteMapper()
        spec = mapper.getSite(self.name)
        return spec.ddm_output
'''


def _server(tmp_path: Path, **files: str) -> dict[str, Path]:
    root = tmp_path / "pandaserver"
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return {"pandaserver": root}


@pytest.mark.parametrize("method, line", [
    ("by_field", 9), ("by_constructor", 14), ("by_parameter", 18), ("by_return", 23),
])
def test_a_site_mapper_is_known_by_what_it_is_declared_to_be(tmp_path, method, line):
    """``self.site_mapper`` is the ``SiteMapper`` ``self.siteMapper`` is, and
    matching the spelling classified one and walked through the other.  The
    class is read the way a type checker reads it -- a base's annotation in
    another file, a constructor, a parameter's annotation, a method's return
    annotation -- and none of these spells ``siteMapper``."""
    roots = _server(
        tmp_path,
        **{
            "brokerage/SiteMapper.py": _SITE_MAPPER,
            "taskbuffer/TaskBuffer.py": _TASK_BUFFER,
            "dataservice/plugin_base.py": _PLUGIN_BASE,
            "dataservice/plugin.py": _PLUGIN,
        },
    )

    walked = trace_mod.walk(
        roots,
        file="pandaserver/dataservice/plugin.py",
        owner=f"pandaserver/dataservice/plugin.py::Plugin.{method}",
        lines=[line],
        classify=provenance.classify,
    )

    (spec,) = [s for s in walked.steps if s.name == "spec"]
    assert spec.detail == "CRIC site declaration", method
    assert any(s.name == "self.job" for s in walked.steps), method


def test_a_class_that_is_only_named_like_the_interface_is_not_it(tmp_path):
    """The key is the file that declares the class: a ``SiteMapper`` of the
    same name elsewhere is not the one CRIC is behind."""
    roots = _server(tmp_path, **{"daemons/other.py": _LOOKALIKE})

    walked = trace_mod.walk(
        roots,
        file="pandaserver/daemons/other.py",
        owner="pandaserver/daemons/other.py::Other.run",
        lines=[10],
        classify=provenance.classify,
    )

    (spec,) = [s for s in walked.steps if s.name == "spec"]
    assert spec.terminal == ""
    # Not an interface, so a function of the corpus: the walk goes into it.
    assert spec.detail == "returns from pandaserver/daemons/other.py::getSite"


# ---------------------------------------------------------------------------
# A message the call does not write out: the syntax it is built in
# ---------------------------------------------------------------------------


def _rows_at(roots, line, **kwargs):
    rows = _printed(roots, file="m", owner="f", lines=[kwargs.pop("arm", 2)], **kwargs)
    return [row for row in rows if row.line == line]


def test_a_message_built_with_plus_is_written_out(tmp_path):
    """154 lines of the corpus print ``a + b``, and every one was refused."""
    roots = _tree(
        tmp_path,
        m='''\
def f(job, log):
    job.status = "failed"
    log.error("failed to get " + job.name + " from " + str(job.site) + traceback.format_exc())
''',
    )

    (row,) = _rows_at(roots, 3)

    assert row.pattern == r"failed\ to\ get\ [^\ ]*\ from\ "
    assert [hole.expression for hole in row.holes] == [
        "job.name",
        "str(job.site)",
        "traceback.format_exc()",
    ]
    assert [hole.prints for hole in row.holes] == ["token", "text", "text"]


def test_a_percent_template_is_written_out_with_its_numbers(tmp_path):
    roots = _tree(
        tmp_path,
        m='''\
def f(job, log, took):
    job.status = "done"
    log.debug("end: took %s.%03d sec in total" % (took.seconds, took.microseconds / 1000))
''',
    )

    (row,) = _rows_at(roots, 3)

    assert row.pattern.startswith(r"end:\ took\ [^\.]*\.[-+]?[\d.]+")
    assert row.pattern.endswith(r"\ sec\ in\ total")
    assert [hole.prints for hole in row.holes] == ["token", "number"]


def test_a_format_template_is_written_out_with_its_specs(tmp_path):
    roots = _tree(
        tmp_path,
        m='''\
def f(site, log, n, rate):
    site.status = "throttled"
    log.info("throttle {0} since nQ({1}) < {2:.3f}".format(site.name, n, rate))
''',
    )

    (row,) = _rows_at(roots, 3)

    assert row.pattern == r"throttle\ [^\ ]*\ since\ nQ\([^\)]*\)\ <\ "
    assert [(hole.expression, hole.prints) for hole in row.holes] == [
        ("site.name", "token"),
        ("n", "token"),
        ("rate", "number"),
    ]


def test_a_name_added_to_with_plus_equals_prints_one_row_not_two(tmp_path):
    """``err_str += traceback.format_exc()`` made a second row of its own, which
    could never be rendered -- and a refused row beside a shaped one tells a
    reader the line's absence proves nothing.  The line is the one the first
    binding wrote, with the traceback after it."""
    roots = _tree(
        tmp_path,
        m='''\
def f(job, log):
    job.status = "failed"
    try:
        job.run()
    except Exception as e:
        err_str = f"execute() : {type(e)} {e}"
        err_str += traceback.format_exc()
        log.debug(err_str)
''',
    )

    (row,) = _rows_at(roots, 8)

    assert row.pattern == r"execute\(\)\ :\ [^\ ]*\ "
    assert row.holes[-1].appended is True


def test_what_plus_equals_adds_with_text_of_its_own_is_a_row_too(tmp_path):
    """``for merge`` and ``for retry`` are what say which branch ran."""
    roots = _tree(
        tmp_path,
        m='''\
def f(job, log, merging):
    job.status = "sent"
    msg = f"sent {job.id} "
    if merging:
        msg += "for merge"
    else:
        msg += "for retry"
    log.debug(msg)
''',
    )

    patterns = {row.pattern for row in _rows_at(roots, 8)}

    assert patterns == {r"sent\ [^\ ]*\ ", r"for\ merge", r"for\ retry"}


def test_an_f_string_of_a_local_prints_the_rows_the_local_does(tmp_path):
    """``f"{error_message}"`` and ``error_message`` are one message.  The first
    was read by a narrower rule, which gave up on a local that ``+=`` also
    writes and left the line with no text at all."""
    source = '''\
def f(job, log):
    job.status = "failed"
    error_message = ""
    try:
        job.register()
    except RegistrationError as e:
        error_message = f"{str(e)} : {job.label} in {job.status}"
        error_message += "\\n" + traceback.format_exc()
    except Exception as e:
        error_message = f"{type(e)} : {e}"
    log.error(%s)
'''
    bare = _tree(tmp_path / "bare", m=source % "error_message")
    fstring = _tree(tmp_path / "fstring", m=source % 'f"{error_message}"')

    said = [{(row.pattern, row.refused) for row in _rows_at(roots, 11)} for roots in (bare, fstring)]

    assert said[0] == said[1]
    assert (r"\ :\ [^\ ]*\ in\ ", "") in said[0]
    assert ("", "no literal text to ask with") in said[0]


def test_a_field_assigned_just_above_the_call_is_written_out(tmp_path):
    """``self.logger.error(self.job.ddmErrorDiag)`` on the line after the
    assignment prints the assignment, and nothing from outside can reach it."""
    roots = _tree(
        tmp_path,
        m='''\
def f(self, file):
    self.job.jobStatus = "failed"
    self.job.ddmErrorDiag = f"expected output {file.lfn} is missing in pilot JSON"
    self.logger.error(self.job.ddmErrorDiag)
''',
    )

    (row,) = _rows_at(roots, 4)

    assert row.pattern == r"expected\ output\ [^\ ]*\ is\ missing\ in\ pilot\ JSON"


def test_a_field_assigned_only_on_one_path_may_hold_what_came_in(tmp_path):
    """Where no binding here runs on every path, the field may still hold what
    it held on the way in, and that way is refused rather than dropped."""
    roots = _tree(
        tmp_path,
        m='''\
def f(self, bad):
    self.job.jobStatus = "failed"
    if bad:
        self.job.ddmErrorDiag = "the file was bad on arrival"
    self.logger.error(self.job.ddmErrorDiag)
''',
    )

    said = {(row.pattern, row.refused) for row in _rows_at(roots, 5)}

    assert said == {
        (r"the\ file\ was\ bad\ on\ arrival", ""),
        ("", trace_mod.REFUSED_SET_ELSEWHERE),
    }


def test_a_parameter_passed_straight_to_the_log_is_the_callers(tmp_path):
    roots = _tree(
        tmp_path,
        m='''\
def f(job, log, msg):
    job.status = "failed"
    log.error(msg)
''',
    )

    (row,) = _rows_at(roots, 3)

    assert row.refused == trace_mod.REFUSED_HANDED_IN


def test_a_local_bound_to_a_bare_word_stays_a_hole(tmp_path):
    """``newStatus = "exhausted"`` is the value the row's ``value`` pattern
    names; writing it out would erase the hole the observed value fills."""
    roots = _tree(
        tmp_path,
        m='''\
def f(task, log):
    newStatus = "exhausted"
    task.status = newStatus
    log.info(f"set status={newStatus} for task")
''',
    )

    (row,) = _printed(roots, file="m", owner="f", lines=[3])

    assert row.pattern == r"set\ status=[^\ ]*\ for\ task"


# ---------------------------------------------------------------------------
# Starting from a line production printed
# ---------------------------------------------------------------------------


def test_a_print_line_is_walked_from_the_names_its_message_reads(tmp_path):
    """The reader holds ``waiting 26 jobs for soft finish``, not a guard.  The
    logger it went through says nothing about the 26 and is not walked."""
    roots = _tree(
        tmp_path,
        m='''\
def f(self, task_id, log):
    pandaIDs = self.tb.getPandaIDsWithTask_JEDI(task_id, True)
    tmpMsg = f"waiting {len(pandaIDs)} jobs for soft finish"
    log.info(tmpMsg)
''',
    )

    for line in (3, 4):
        walked = trace_mod.walk_from_print(
            roots, file=f"{PACKAGE}/m", line=line, classify=provenance.classify
        )
        kinds = [(step.kind, step.name or step.value) for step in walked.steps]
        assert kinds[0][0] == models.TRACE_PRINT
        assert (models.TRACE_BINDING, "pandaIDs") in kinds
        assert not any(name == "log" for _kind, name in kinds)


def test_a_line_that_neither_prints_nor_writes_a_message_says_so(tmp_path):
    roots = _tree(
        tmp_path,
        m='''\
def f(job, log):
    job.status = "done"
    log.info("done")
''',
    )

    walked = trace_mod.walk_from_print(roots, file=f"{PACKAGE}/m", line=2)

    assert walked.steps == []
    assert walked.note == "no logging call prints what is on this line"


def test_a_print_line_from_another_tree_is_refused(tmp_path):
    roots = _tree(tmp_path, m='def f(log):\n    log.info("done here")\n')

    walked = trace_mod.walk_from_print(
        roots, file=f"{PACKAGE}/m", line=2, expected_sha="0" * 40
    )

    assert walked.steps == []
    assert "not the one that was mapped" in walked.note


# ---------------------------------------------------------------------------
# Across a function's edge: calls, parameters, methods that change an object
# ---------------------------------------------------------------------------


def _walked_in(roots, file, owner, lines, **kwargs):
    return trace_mod.walk(
        roots,
        file=f"{PACKAGE}/{file}",
        owner=f"{PACKAGE}/{file}::{owner}",
        lines=lines,
        classify=provenance.classify,
        reaching=True,
        **kwargs,
    )


def _said(walked):
    return [(s.kind, s.name, s.owner.rsplit("::", 1)[-1], s.line, s.value) for s in walked.steps]


def test_a_value_bound_from_a_call_is_what_the_call_returns(tmp_path):
    """``where, binds = self.construct(site)`` is answered inside ``construct``."""
    roots = _tree(
        tmp_path,
        m='''\
class Proxy:
    def construct(self, site, mem):
        where = "WHERE site=:site "
        if mem:
            where += "AND ram<=:mem "
        return where, {":site": site}

    def get_jobs(self, site, mem):
        where, binds = self.construct(site, mem)
        sql = "SELECT id FROM jobs " + where
        spec.status = sql
''',
    )

    walked = _walked_in(roots, "m", "get_jobs", [11])
    said = _said(walked)

    assert ("return", "construct()", "construct", 6, "where") in said
    assert ("binding", "where", "construct", 5, "'AND ram<=:mem '") in said
    # The parameter the fragment depends on, handed in by the call.
    assert ("handover", "mem", "get_jobs", 9, "mem") in said


def test_the_answer_about_the_function_itself_does_not_change_when_the_walk_goes_further(tmp_path):
    """Going into a call is spent from an allowance of its own, afterwards."""
    roots = _tree(
        tmp_path,
        m='''\
def helper(x):
    return x + 1

def f(a):
    b = helper(a)
    spec.status = b
''',
    )

    walked = _walked_in(roots, "m", "f", [6])
    own = [step for step in walked.steps if step.owner.endswith("::f")]

    assert [(s.kind, s.name, s.line) for s in own][:2] == [("write", "", 6), ("binding", "b", 5)]
    assert walked.steps[1].detail == f"returns from {PACKAGE}/m::helper"


def test_a_guard_that_asks_a_method_names_the_calls_that_changed_its_answer(tmp_path):
    """``not self.result.is_succeeded()`` is decided by which ``set_*()`` ran --
    listed at the calls that made them, with what printed beside each."""
    roots = _tree(
        tmp_path,
        **{"result.py": '''\
class Result:
    rc_ok = 0
    rc_fatal = 2

    def __init__(self):
        self.status_code = None

    def set_fatal(self):
        self.status_code = self.rc_fatal

    def is_succeeded(self):
        return self.status_code == self.rc_ok
''',
        "plugin.py": '''\
from pandajedi.result import Result

class Plugin:
    def __init__(self):
        self.result = Result()

    def register(self, log, ok):
        if not ok:
            log.error("registration failed for good")
            self.result.set_fatal()
''',
        "gen.py": '''\
from pandajedi.plugin import Plugin

class Gen:
    def run(self, job):
        self.plugin = Plugin()
        self.add_result = self.plugin.result
        if not self.add_result.is_succeeded():
            job.status = "failed"
'''},
    )

    walked = _walked_in(roots, "gen.py", "run", [8])
    said = _said(walked)

    assert ("return", "is_succeeded()", "is_succeeded", 12, "self.status_code == self.rc_ok") in said
    (called,) = [s for s in walked.steps if s.value == "self.result.set_fatal()"]
    assert (called.owner.rsplit("::", 1)[-1], called.line) == ("register", 10)
    assert called.printed == [r"registration\ failed\ for\ good"]
    assert ("binding", "self.rc_fatal", "Result", 3, "2") in said


def test_an_entry_read_by_its_key_is_what_the_callee_put_under_it(tmp_path):
    """``ok, data = get(...)`` then ``data["ram"]``: the callee's ``returnMap["ram"] = ...``."""
    roots = _tree(
        tmp_path,
        m='''\
def get(task):
    returnMap = {}
    returnMap["ram"] = task.maxPSS * 2
    returnMap["wall"] = task.wall
    return True, returnMap

def f(task):
    ok, data = get(task)
    spec.ramCount = data["ram"]
''',
    )

    walked = _walked_in(roots, "m", "f", [9])
    said = _said(walked)

    assert ("binding", "returnMap['ram']", "get", 3, "task.maxPSS * 2") in said


def test_a_container_filled_by_item_is_what_went_into_it(tmp_path):
    roots = _tree(
        tmp_path,
        m='''\
def f(task):
    returnMap = {}
    returnMap["ram"] = task.maxPSS
    spec.values = returnMap
''',
    )

    said = _said(_walked_in(roots, "m", "f", [4]))

    assert ("binding", "returnMap", "f", 3, "task.maxPSS") in said


# ---------------------------------------------------------------------------
# Where a value stops: what it was asked with, and what could have raised
# ---------------------------------------------------------------------------


def test_a_plugin_built_with_the_task_buffer_is_not_the_databases_value():
    """An argument that is the task buffer does not make the call's value a row."""
    expression = ast.parse(
        "self.adder_plugin_class(self.job, taskBuffer=self.taskBuffer)", mode="eval"
    ).body

    assert provenance.classify(expression) == ("", "")


def test_a_builtin_still_hands_on_where_its_argument_came_from():
    expression = ast.parse(
        'int(self.getConfigValue("dbproxy", "SCOUT_RAMCOUNT_RANK", "jedi"))', mode="eval"
    ).body

    assert provenance.classify(expression) == (
        models.STOP_CONFIG,
        "dbproxy.SCOUT_RAMCOUNT_RANK.jedi",
    )


def test_a_configuration_key_is_written_as_the_code_spells_every_part_of_it():
    """``watchdog.jedi`` was a key that is not in the table."""
    expression = ast.parse(
        'self.taskBufferIF.getConfigValue("watchdog", f"PENDING_TIMEOUT_{label}", "jedi", vo)',
        mode="eval",
    ).body

    assert provenance.classify(expression)[1] == "watchdog.PENDING_TIMEOUT_{label}.jedi.{vo}"


def test_a_handler_names_every_call_in_its_body_that_could_have_raised(tmp_path):
    roots = _server(
        tmp_path,
        **{
            "dataservice/ddm.py": "class RucioAPI:\n    def register(self, x):\n        return x\n\nrucioAPI = RucioAPI()\n",
            "dataservice/plugin.py": '''\
from pandaserver.dataservice.ddm import rucioAPI


def register(self, files):
    is_fatal = False
    try:
        rucioAPI.register(files)
        rucioAPI.register(files[0])
        self.siteMapper.getSite("x")
    except Exception as e:
        is_fatal = True
    return is_fatal
''',
        },
    )

    walked = trace_mod.walk(
        roots,
        file="pandaserver/dataservice/plugin.py",
        owner="pandaserver/dataservice/plugin.py::register",
        lines=[12],
        classify=provenance.classify,
    )

    (step,) = [s for s in walked.steps if s.line == 11 and s.name == "is_fatal"]
    assert step.terminal == models.STOP_EXTERNAL
    assert step.detail == (
        "raised by one of: rucioAPI.register (DDM / Rucio); "
        "self.siteMapper.getSite (CRIC site declaration)"
    )


def test_a_value_the_database_supplies_says_which_table_and_key_it_was_asked_with(tmp_path):
    roots = _server(
        tmp_path,
        **{
            "taskbuffer/OraDBProxy.py": '''\
class DBProxy:
    def getPandaIDsWithTask_JEDI(self, jediTaskID, onlyActive):
        sql = "SELECT PandaID FROM ATLAS_PANDA.jobsActive4 WHERE jediTaskID=:jediTaskID "
        varMap = {}
        varMap[":jediTaskID"] = jediTaskID
        self.cur.execute(sql, varMap)
        return self.cur.fetchall()
''',
            "daemons/commando.py": '''\
def run(self, jediTaskID, log):
    pandaIDs = self.taskBufferIF.getPandaIDsWithTask_JEDI(jediTaskID, True)
    log.info(f"waiting {len(pandaIDs)} jobs for soft finish")
''',
        },
    )

    walked = trace_mod.walk_from_print(
        roots, file="pandaserver/daemons/commando.py", line=3, classify=provenance.classify
    )

    (step,) = [s for s in walked.steps if s.name == "pandaIDs"]
    assert step.terminal == models.STOP_UPSTREAM
    assert step.detail == "getPandaIDsWithTask_JEDI -- reads jobsActive4 by :jediTaskID"


def test_a_logger_built_by_a_helper_says_its_prefix_through_the_helper(tmp_path):
    """``start`` is printed by hundreds of proxy methods; the prefix
    ``create_tagged_logger`` builds from ``comment`` is what names this one."""
    roots = _tree(
        tmp_path,
        **{
            "base.py": '''\
from pandacommon.pandalogger.LogWrapper import LogWrapper


class Base:
    def create_tagged_logger(self, comment, tag=None):
        """Create logger from function comment and tag."""
        method_name = comment.split(" ")[-2].split(".")[-1]
        if tag is None:
            tag = now()
        method_name += f" < {tag} >"
        tmp_log = LogWrapper(self._log_stream, method_name)
        return tmp_log
''',
            "proxy.py": '''\
from pandajedi.base import Base


class Proxy(Base):
    def getPandaIDs(self, jediTaskID):
        comment = " /* JediDBProxy.getPandaIDs */"
        tmpLog = self.create_tagged_logger(comment, f"jediTaskID={jediTaskID}")
        tmpLog.debug("start")
        spec.status = "done"
''',
        },
    )

    walked = _walked_in(roots, "proxy.py", "getPandaIDs", [9])
    (row,) = [r for r in walked.skeleton if r.kind == models.SKELETON_PRINT]

    assert row.pattern == "start"
    assert row.prefix == r"getPandaIDs\ <\ .*?\ >\ "


def test_a_logger_built_here_says_its_prefix(tmp_path):
    roots = _tree(
        tmp_path,
        m='''\
def f(self, job):
    tmp_log = LogWrapper(self.logger, "<setup_destination>")
    tmp_log.debug(f"made sub:{job.name} for nFiles=3")
    job.status = "done"
''',
    )

    (row,) = _printed(roots, file="m", owner="f", lines=[4])

    assert row.prefix == r"<setup_destination>\ "


def test_the_rows_a_loop_is_on_are_told_apart_from_how_a_value_was_worked_out(tmp_path):
    roots = _tree(
        tmp_path,
        m='''\
def f(self, limit):
    for job in self.jobs:
        size = job.size * 2
        if size > limit:
            job.status = "failed"
''',
    )

    walked = _walked_in(roots, "m", "f", [5])
    roles = {(s.name, s.role) for s in walked.steps if s.owner.endswith("::f")}

    assert ("job", "iteration") in roles
    assert ("size", "") in roles


def test_a_line_that_prints_nothing_is_walked_from_what_it_computes(tmp_path):
    roots = _tree(
        tmp_path,
        m='''\
def ram(task):
    return max(task.ramCount, 2000)

def f(task):
    minRam = ram(task)
    return minRam
''',
    )

    walked = trace_mod.walk_from_line(roots, file=f"{PACKAGE}/m", line=5, classify=provenance.classify)
    said = _said(walked)

    assert ("return", "ram()", "ram", 2, "max(task.ramCount, 2000)") in said


def test_a_decision_whose_silent_branch_reads_configuration_is_listed(tmp_path):
    from bamboo.scripts.panda import silent_exits

    root = tmp_path / "pandaserver"
    (root / "daemons").mkdir(parents=True)
    (root / "daemons" / "m.py").write_text(
        """\
def f(self, jobs, log):
    for job in jobs:
        limit = self.taskBuffer.getConfigValue("dog", "LIMIT", "jedi")
        if job.n > limit:
            continue
        log.info("kept a job")
"""
    )

    listed = silent_exits.scan(tmp_path, {"config"})

    (entry,) = listed
    assert (entry["at"], entry["branch"], entry["leaves"]) == ("pandaserver/daemons/m.py:4", "body", True)
    assert entry["reads"] == ["a configuration value, set outside the code -- dog.LIMIT.jedi"]


def test_a_parameter_placed_at_many_calls_is_said_and_not_walked(tmp_path):
    """The first eight of twelve callers are as arbitrary as any eight."""
    callers = "\n".join(f"def c{i}():\n    return g({i})\n" for i in range(12))
    roots = _tree(tmp_path, **{"m.py": f"def g(x):\n    spec.status = x\n\n{callers}"})

    walked = _walked_in(roots, "m.py", "g", [2])

    (stopped,) = [s for s in walked.steps if s.name == "x" and s.kind == "unbound"]
    assert stopped.detail == "a parameter; 12 calls are placed at g, too many to follow"
    assert not [s for s in walked.steps if s.kind == "handover"]


def test_a_call_to_another_systems_interface_is_gone_into_on_this_side(tmp_path):
    """``SiteMapper.getSite`` is CRIC's answer, and it is also PanDA code that
    resolves ``nucleus:X`` to one of its sites first."""
    roots = _server(
        tmp_path,
        **{
            "brokerage/SiteMapper.py": '''\
class SiteMapper:
    def getSite(self, site_name):
        site_name = self.clean(site_name)
        return self.siteSpecList[site_name]

    def clean(self, site_name):
        return site_name.split(":")[-1]
''',
            "dataservice/plugin.py": '''\
from pandaserver.brokerage.SiteMapper import SiteMapper


class Plugin:
    def __init__(self, siteMapper: SiteMapper):
        self.siteMapper = siteMapper

    def run(self, job):
        dst = self.siteMapper.getSite(job.destinationSE)
        job.dst = dst
''',
        },
    )

    walked = trace_mod.walk(
        roots,
        file="pandaserver/dataservice/plugin.py",
        owner="pandaserver/dataservice/plugin.py::run",
        lines=[10],
        classify=provenance.classify,
        reaching=True,
    )
    said = _said(walked)

    (bound,) = [s for s in walked.steps if s.name == "dst"]
    assert bound.terminal == models.STOP_EXTERNAL
    assert ("return", "getSite()", "getSite", 4, "self.siteSpecList[site_name]") in said
