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

from bamboo.codemap import calls as calls_mod
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
    # ``walk`` takes no default for ``reaching``; these tests were written
    # against the one it had, and say so here once.
    kwargs.setdefault("reaching", False)
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


def test_the_note_names_every_budget_that_ran_out_on_each_side(tmp_path):
    """``setCloudToTasks_JEDI``'s walk ran out of depth beyond the function
    and then of steps; a note naming the first hid the steps it left out."""
    roots = _tree(tmp_path, feeder=KNIGHT)
    walk = trace_mod._Walk(roots, provenance.classify, trace_mod.Budget())
    walk._exhausted("depth")
    assert walk.note == "the walk's depth budget ran out; what follows is part of the answer"
    walk.crossing = True
    walk._exhausted("steps")
    walk._exhausted("depth")
    walk._exhausted("steps")

    assert walk.crossing_note == (
        "the walk's steps and depth budgets beyond this function ran out; "
        "what the calls hand back is part of the answer"
    )
    assert walk.note == "the walk's depth budget ran out; what follows is part of the answer"
    # Budget's order, whatever order they ran out in.
    walk._exhausted("files")
    walk._exhausted("functions")
    assert walk.crossing_note.startswith("the walk's files, functions, steps and depth budgets beyond")


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


def test_another_classs_method_of_the_same_name_is_not_the_siblings(tmp_path):
    # ``ThreadUtils.py`` and ``JobGenerator.py`` define ``__init__`` in two
    # classes.  The frame for one was keyed by the file and the name alone,
    # so the second class's walk read the first class's ``__init__``.  Each
    # step says which class's method it is.
    roots = _tree(
        tmp_path,
        threads='''\
class Lock:
    def __init__(self):
        self.mode = "shared"

    def take(self):
        spec.mode = self.mode


class Worker:
    def __init__(self):
        self.mode = "exclusive"

    def run(self):
        spec.mode = self.mode
''',
    )

    first, _ = _walk(roots, file="threads", owner="take", lines=[6])
    steps, _ = _walk(roots, file="threads", owner="run", lines=[14])

    assert [(s.line, s.value, s.cls) for s in first if s.name == "self.mode"] == [
        (3, "'shared'", f"{PACKAGE}/threads::Lock")
    ]
    assert [(s.line, s.value, s.cls) for s in steps if s.name == "self.mode"] == [
        (11, "'exclusive'", f"{PACKAGE}/threads::Worker")
    ]
    # One walk entering both: the second ``__init__`` asked for, by line and
    # as a sibling, is not handed the first one's frame.
    walk = trace_mod._Walk(roots, provenance.classify, trace_mod.Budget())
    file = f"{PACKAGE}/threads"
    locked = walk.frame(file, f"{file}::__init__", line=3)
    working = walk.frame(file, f"{file}::__init__", line=11)
    assert (locked.func.lineno, working.func.lineno) == (2, 10)
    assert working.owner == f"{file}::Worker.__init__"
    run = walk.frame(file, f"{file}::run", line=14)
    worker_init = working.func
    walk.frames.pop((file, working.owner))
    assert walk._sibling(run, worker_init).func is worker_init


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


def test_a_write_in_a_nested_class_does_not_replace_the_functions_own(tmp_path):
    """``add_main.main`` sets ``taskBuffer = tbuf`` and a thread class inside it
    sets ``taskBuffer = self.taskBuffer`` in its own ``run()``.

    That is the nested function's variable.  Taken for an overwrite, it left
    ``main``'s read answered by the thread's line and nothing else.
    """
    roots = _tree(
        tmp_path,
        adder='''\
class Adder:
    def main(self, tbuf, row):
        taskBuffer = tbuf
        class Thread:
            def run(self):
                taskBuffer = self.taskBuffer
                taskBuffer.go()
        row.status = taskBuffer
''',
    )

    ours, _note = _walk(roots, file="adder", owner="main", lines=[8], reading_at=8)

    assert [s.line for s in ours if s.kind == models.TRACE_BINDING and s.name == "taskBuffer"] == [3]


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


def test_a_fill_in_the_stretch_keeps_the_container_built_outside_it(tmp_path):
    """A fill adds to what the name holds; it does not bind the name anew.

    ``archiveJob`` builds ``toBeClosedSubList = {}`` above its loop and fills
    it by item inside the statement's stretch.  Counting the fill as the
    stretch's own binding dropped the ``{}`` the test on the way reads first.
    """
    roots = _tree(
        tmp_path,
        sweeper="""\
class Sweeper:
    def run(self):
        seen = {}
        for row in self.query():
            if row.key not in seen:
                seen[row.key] = []
                row.status = 1
""",
    )

    ours, _note = _walk(
        roots, file="sweeper", owner="run", lines=[7], bindings_within=(4, 7)
    )

    assert [
        step.line
        for step in ours
        if step.kind == models.TRACE_BINDING and step.name == "seen"
    ] == [3, 6]


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


def test_a_line_printed_in_a_case_is_in_the_skeleton(tmp_path):
    """``workflow_core.py`` prints fifteen lines inside ``case`` blocks, and
    none of them was in any skeleton -- a reader laying one over a log reads
    such a line as one the function cannot print."""
    roots = _tree(
        tmp_path,
        flow='''\
class Flow:
    def check(self, spec, result):
        match result.status:
            case Status.done:
                spec.status = "checked"
            case Status.partial if spec.loud:
                logger.info(f"partial for {spec.name}")
            case _:
                logger.warning("unexpected status; skipped")
                return
''',
    )

    rows = _skeleton(roots, file="flow", owner="check", lines=[5])

    shape = [
        (row.kind, row.text if row.kind == models.SKELETON_BRANCH else row.line, row.depth)
        for row in rows
    ]
    assert shape == [
        (models.SKELETON_BRANCH, "match result.status:", 0),
        (models.SKELETON_BRANCH, "case Status.done:", 0),
        (models.SKELETON_ARM, 5, 1),
        (models.SKELETON_BRANCH, "case Status.partial if spec.loud:", 0),
        (models.SKELETON_PRINT, 7, 1),
        (models.SKELETON_BRANCH, "case _:", 0),
        (models.SKELETON_PRINT, 9, 1),
    ]


def test_every_function_that_prints_has_its_rows_with_nothing_asked(tmp_path):
    """What the build stores: the skeleton's print rows, field for field the
    walk's, less what an asking adds -- for every function that prints, each
    under the name that tells it from another of its name."""
    roots = _tree(
        tmp_path,
        daemon='''\
class Closer:
    def run(self, spec):
        logger.info(f"closing {spec.name} now")
        if spec.done:
            msg = "the dataset is already closed"
        else:
            msg = "the dataset is closing now"
        logger.debug(msg)
        spec.status = "closed"


class Freezer:
    def run(self, spec):
        def warn(text):
            logger.warning(f"freezer says {text}")
        warn("cold")
        logger.error("the dataset is frozen")
''',
    )

    found = {one.owner: one for one in trace_mod.function_prints(roots, [f"{PACKAGE}/daemon"])}

    closer = f"{PACKAGE}/daemon::Closer.run"
    assert sorted(found) == [closer, f"{PACKAGE}/daemon::Freezer.run", f"{PACKAGE}/daemon::Freezer.run.warn"]
    assert found[closer].cls == f"{PACKAGE}/daemon::Closer"
    assert found[f"{PACKAGE}/daemon::Freezer.run.warn"].cls == ""
    assert [(s.line, s.level, len(s.rows)) for s in found[closer].statements] == [
        (3, "info", 1),
        (8, "debug", 2),
    ]
    assert [s.line for s in found[f"{PACKAGE}/daemon::Freezer.run"].statements] == [17]
    walked = _printed(roots, file="daemon", owner="run", lines=[9], observed="closed")
    assert [
        (row.line, row.text, row.pattern, row.refused, row.keyed, row.holes, row.prefix)
        for row in walked
    ] == [
        (s.line, row.text, row.pattern, row.refused, row.keyed, row.holes, row.prefix)
        for s in found[closer].statements
        for row in s.rows
    ]


def test_a_call_whose_value_is_used_is_not_a_print(tmp_path):
    """``raise socket.error("empty packet")`` is a method named ``error``, and
    the level is read off the name: it put two rows in a skeleton that no log
    can hold."""
    roots = _tree(
        tmp_path,
        bridge='''\
class Bridge:
    def recv(self, spec):
        if spec.master:
            raise socket.error("empty packet")
        logger.error("the child closed the socket on us")
        spec.status = "closed"
''',
    )

    rows = _printed(roots, file="bridge", owner="recv", lines=[6])

    assert [row.line for row in rows] == [5]


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


def test_a_message_written_further_down_the_loop_is_not_what_the_top_prints(tmp_path):
    """``TaskCommando.runImpl``: ``tmpMsg = f"executing ..."`` is printed on the next line.

    The ``set task_status=`` written below it comes round the loop, but the
    binding at the top replaces it on every pass before the call.  Read as a
    way of the top line, it had four lines printing the status where one does.
    """
    roots = _tree(
        tmp_path,
        m="""
def f(items, log):
    for item in items:
        message = f"executing {item}"
        log.info(message)
        spec.status = "ready"
        message = f"set task_status={spec.status}"
        log.info(message)
""",
    )

    top = _rows_at(roots, 5, arm=6)
    bottom = _rows_at(roots, 8, arm=6)

    assert [row.pattern for row in top] == [r"executing\ "]
    assert [row.pattern for row in bottom] == [r"set\ task_status="]


def test_a_message_overwritten_in_the_handler_is_not_what_the_handler_prints(tmp_path):
    """``AtlasDDMClient.getAvailableFiles``: the ``try`` sets the message and returns.

    The call that logs it can raise into the handler, so the ``try``'s binding
    does reach the handler -- and the handler writes its own before printing.
    """
    roots = _tree(
        tmp_path,
        m="""
def f(spec, log):
    spec.status = "ready"
    try:
        error_message = "dataset spec carries no dataset name"
        log.error(error_message)
        return 1
    except Exception as e:
        error_message = f"failed with {e}"
        log.error(error_message)
""",
    )

    (row,) = _rows_at(roots, 10)

    assert row.pattern == r"failed\ with\ "


def test_what_plus_equals_adds_is_printed_until_the_message_is_replaced(tmp_path):
    roots = _tree(
        tmp_path,
        m="""
def f(spec, log):
    spec.status = "ready"
    message = "head"
    message += " for the tail"
    log.info(message)
    message = "replaced outright"
    log.info(message)
""",
    )

    kept = [row.pattern for row in _rows_at(roots, 6)]
    replaced = [row.pattern for row in _rows_at(roots, 8)]

    assert r"head" in kept and any("for\\ the\\ tail" in one for one in kept), kept
    assert replaced == [r"replaced\ outright"]


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

_PROXY_POOL = '''\
from collections.abc import Iterator
from contextlib import contextmanager

from pandaserver.taskbuffer.OraDBProxy import DBProxy


class DBProxyPool:
    @contextmanager
    def get(self) -> Iterator[DBProxy]:
        yield DBProxy()
'''


def _forwarding_task_buffer(*methods: str) -> str:
    """``TaskBuffer``, the declared facade, whose methods borrow a proxy and call its method of the same name.

    Each of *methods* is ``name``, then its parameters as one string.
    """
    pairs = list(zip(methods[::2], methods[1::2], strict=True))
    body = "".join(
        f"""
    def {name}(self, {parameters}):
        with self.proxyPool.get() as proxy:
            return proxy.{name}({parameters})
"""
        for name, parameters in pairs
    )
    return f"""from pandaserver.taskbuffer.DBProxyPool import DBProxyPool


class TaskBuffer:
    proxyPool: DBProxyPool
{body}"""


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
        reaching=False,
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
        reaching=False,
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


def test_a_caller_is_placed_whichever_reading_parsed_its_file_first(tmp_path):
    """One tree per file.  The caller index is built the first time a walk
    asks for callers, from whatever parse of each file exists then; the walk
    parsed ``front.py`` again for itself on the way, and ``self.front``'s
    class was then looked up by the identity of ``run`` in the walk's tree,
    where that ``run`` is not.  ``self.front.put(7)`` was placed nowhere, and
    the corpus's DBProxy methods lost their ``JediTaskBuffer`` callers by
    which walks had run before them in the process."""
    roots = _tree(
        tmp_path,
        **{
            "store.py": '''\
class Store:
    def save(self, value):
        log.info(f"saved {value}")
''',
            "front.py": '''\
from pandajedi.store import Store


class Front:
    store: Store

    def put(self, v):
        self.store.save(v)


class Outer:
    front: Front

    def run(self):
        self.front.put(7)
''',
        },
    )

    walked = trace_mod.walk_from_print(roots, file=f"{PACKAGE}/store.py", line=3, classify=provenance.classify)

    said = [(step.kind, step.name, step.value, step.detail) for step in walked.steps]
    assert (TRACE_HANDOVER, "value", "v", "passed to save here") in said
    assert (TRACE_HANDOVER, "v", "7", "passed to put here") in said


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


_OPEN_ENTRY = '''\
from typing import Any


class Interface:
    def __getattr__(self, name: str) -> Any:
        return getattr(self.remote, name)


class Buffer:
    def getTask(self, x):
        return x + 1

    def plain(self, x):
        return x - 1


class Feeder:
    def __init__(self, source: "Interface | Buffer", buffer: Buffer):
        self.source = source
        self.buffer = buffer

    def run(self, a):
        v = {first}
        w = {second}
        u = self.buffer.plain(a)
        spec.status = v + w + u
'''


@pytest.mark.parametrize(
    "first, second",
    [
        ("self.source.getTask(a)", "self.buffer.getTask(a)"),
        ("self.buffer.getTask(a)", "self.source.getTask(a)"),
    ],
)
def test_a_function_some_call_entered_open_says_so_on_every_return_whichever_call_came_first(tmp_path, first, second):
    """``self.source.getTask`` lands at ``Buffer.getTask`` or wherever the interface forwards it.

    Its returns are recorded once, by whichever call enters first -- here a
    call placed closed as often as not -- and they say the value may come
    from elsewhere either way.
    """
    roots = _tree(tmp_path, m=_OPEN_ENTRY.format(first=first, second=second))

    walked = _walked_in(roots, "m", "run", [26])
    returned = {(s.name, s.line, s.open) for s in walked.steps if s.kind == models.TRACE_RETURN}
    entering = next(s for s in walked.steps if s.kind == TRACE_BINDING and s.value == "self.source.getTask(a)")

    assert returned == {("getTask()", 11, calls_mod.UNPLACED_OPEN), ("plain()", 14, "")}
    assert entering.detail == f"returns from {PACKAGE}/m::getTask; {calls_mod.UNPLACED_OPEN}"


_SET_BY_A_SETTER = {
    "result.py": '''\
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
''',
}


def test_a_guard_that_asks_a_method_names_the_calls_that_changed_its_answer(tmp_path):
    """``not self.result.is_succeeded()`` is decided by which ``set_*()`` ran --
    listed at the calls that made them, with what printed beside each."""
    roots = _tree(tmp_path, **_SET_BY_A_SETTER)

    walked = _walked_in(roots, "gen.py", "run", [8])
    said = _said(walked)

    assert ("return", "is_succeeded()", "is_succeeded", 12, "self.status_code == self.rc_ok") in said
    (called,) = [s for s in walked.steps if s.value == "self.result.set_fatal()"]
    assert (called.owner.rsplit("::", 1)[-1], called.line) == ("register", 10)
    assert called.printed == [r"registration\ failed\ for\ good"]
    assert ("binding", "self.rc_fatal", "Result", 3, "2") in said


def test_a_guard_enters_the_calls_its_test_makes_and_not_the_calls_under_it(tmp_path):
    """The ``if`` decides by its test; a call in its body is another statement."""
    roots = _tree(
        tmp_path,
        m='''\
def decides(x):
    return x > 1

def unrelated(y):
    return y * 2

def f(a, b):
    if decides(a):
        c = unrelated(b)
        spec.status = "done"
''',
    )

    walked = _walked_in(roots, "m", "f", [10])
    entered = {s.name for s in walked.steps if s.kind == "return"}

    assert "decides()" in entered
    assert "unrelated()" not in entered


def test_a_call_the_walk_arrives_at_twice_hands_its_arguments_over_once(tmp_path):
    """Two statements under one ``if`` each enter its test's call; ``x`` was passed once."""
    roots = _tree(
        tmp_path,
        m='''\
def allowed(x):
    return x in ("a", "b")

def f(s, p, q):
    if allowed(s):
        p = 1
        q = 2
    spec.status = p + q
''',
    )

    walked = _walked_in(roots, "m", "f", [8])
    handed = [(s.line, s.value) for s in walked.steps if s.kind == "handover" and s.name == "x"]

    assert handed == [(5, "s")]


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
        reaching=False,
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
            "taskbuffer/DBProxyPool.py": _PROXY_POOL,
            "taskbuffer/TaskBuffer.py": _forwarding_task_buffer("getPandaIDsWithTask_JEDI", "jediTaskID, onlyActive"),
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


def test_an_empty_prefix_says_why_where_the_logger_may_write_one(tmp_path):
    """``self.logger = LogWrapper(_logger, str(pandaID))`` in ``__init__``
    writes the PandaID before every message ``run`` prints; a row saying no
    prefix there read as a run with no mark (Watcher.run, the adder).  The
    prefix itself is read no further: only why it is not."""
    roots = _tree(
        tmp_path,
        m='''\
_logger = PandaLogger().getLogger("m")


class Watcher:
    def __init__(self, pandaID):
        self.logger = LogWrapper(_logger, str(pandaID))

    def run(self, job, handed):
        self.logger.debug("start")
        handed.debug("handed in")
        _logger.debug("the module's")
        stamped = LogWrapper(_logger, None)
        stamped.debug("stamped")
        plain = LogWrapper(_logger)
        plain.debug("plain")
        wrapped = MsgWrapper(_logger, "token")
        wrapped.debug("wrapped")
        twice = LogWrapper(_logger, "<a>")
        if job:
            twice = LogWrapper(_logger, "<b>")
        twice.debug("twice")
        keyed = LogWrapper(_logger, prefix="<k>")
        keyed.debug("keyed")
        said = LogWrapper(_logger, f"<run {job.PandaID}>")
        said.debug("said")
        said.debug(f"{job.name}")
''',
    )

    (found,) = [
        one for one in trace_mod.function_prints(roots, [f"{PACKAGE}/m"]) if one.owner.endswith("Watcher.run")
    ]
    rows = {row.text: (row.prefix, row.prefix_hole) for statement in found.statements for row in statement.rows}

    assert rows == {
        "'start'": ("", trace_mod.PREFIX_NOT_A_NAME.format(logger="self.logger")),
        "'handed in'": ("", trace_mod.PREFIX_PARAMETER.format(logger="handed")),
        '"the module\'s"': ("", trace_mod.PREFIX_NOT_BOUND.format(logger="_logger")),
        "'stamped'": ("", trace_mod.PREFIX_STAMP.format(logger="stamped")),
        "'plain'": ("", ""),
        "'wrapped'": ("", trace_mod.PREFIX_NOT_FOLLOWED.format(logger="wrapped", call="MsgWrapper")),
        "'twice'": ("", trace_mod.PREFIX_REBOUND.format(logger="twice")),
        "'keyed'": ("", trace_mod.PREFIX_KEYWORD.format(logger="keyed")),
        "'said'": (r"<run\ .*?>\ ", ""),
        "f'{job.name}'": ("", trace_mod.PREFIX_REFUSED),
    }


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
    assert entry["cut"] is False


def test_a_silent_decision_is_listed_by_its_own_test_and_not_the_ones_above(tmp_path):
    """``if minRamCount > site_maxmemory`` was listed as reading the I/O cutoff
    its function tests at the top: the guards above the ``if`` were walked as
    though the test read them."""
    from bamboo.scripts.panda import silent_exits

    root = tmp_path / "pandaserver"
    (root / "daemons").mkdir(parents=True)
    (root / "daemons" / "m.py").write_text(
        """\
def f(self, jobs, log):
    cutoff = self.taskBuffer.getConfigValue("dog", "CUTOFF", "jedi")
    if cutoff:
        for job in jobs:
            if job.n > 5:
                continue
            log.info("kept a job")
"""
    )

    assert silent_exits.scan(tmp_path, {"config"}) == []


def test_a_parameter_placed_at_many_calls_is_said_and_not_walked(tmp_path):
    """The first eight of twelve callers are as arbitrary as any eight."""
    callers = "\n".join(f"def c{i}():\n    return g({i})\n" for i in range(12))
    roots = _tree(tmp_path, **{"m.py": f"def g(x):\n    spec.status = x\n\n{callers}"})

    walked = _walked_in(roots, "m.py", "g", [2])

    (stopped,) = [s for s in walked.steps if s.name == "x" and s.kind == "unbound"]
    assert stopped.detail == "a parameter; 12 calls are placed at g, too many to follow"
    assert not [s for s in walked.steps if s.kind == "handover"]


@pytest.mark.parametrize(
    ("budget", "said", "handed"),
    [
        (
            trace_mod.Budget(functions=1, crossed_functions=1),
            "2 of them not walked, the walk's functions budget ran out: pandajedi/m.py:8, pandajedi/m.py:11",
            ["0"],
        ),
        (
            trace_mod.Budget(steps=2, crossed_steps=2),
            "1 of them not walked, the walk's steps budget ran out: pandajedi/m.py:11",
            ["0", "1"],
        ),
    ],
)
def test_a_caller_the_walks_budget_leaves_unwalked_is_said_on_the_stop_that_counted_it(tmp_path, budget, said, handed):
    """"Followed to the 3 call(s)" over one handover step said three were walked."""
    callers = "\n".join(f"def c{i}():\n    return g({i})\n" for i in range(3))
    roots = _tree(tmp_path, **{"m.py": f"def g(x):\n    spec.status = x\n\n{callers}"})

    walked = _walked_in(roots, "m.py", "g", [2], budget=budget)

    (stopped,) = [s for s in walked.steps if s.name == "x" and s.kind == "unbound"]
    assert stopped.detail == f"a parameter; followed to the 3 call(s) placed at g; {said}"
    assert [s.value for s in walked.steps if s.kind == "handover"] == handed


_RUN = '''\
    def run(self, log):
        got = self.box.take()
        log.info(f"took {got}")
'''


def _boxes(knight: str, worker: str = _RUN, others: int = 9) -> str:
    """A ``Box`` built by a knight for its worker, and by *others* functions for nobody here."""
    built = "\n".join(f"def other{i}():\n    return Box([{i}])\n" for i in range(others))
    return f"""\
class Box:
    def __init__(self, items):
        self.items = items

    def take(self):
        return self.items


class Worker:
    box: Box
    spare: Box

    def __init__(self, box, spare=None):
        self.box = box
        self.spare = spare

{worker}

class Knight:
{knight}


{built}"""


def _built_handovers(walked):
    return sorted(
        (s.owner.rsplit("::", 1)[-1], s.value)
        for s in walked.steps
        if s.kind == TRACE_HANDOVER and s.name == "items"
    )


def test_a_constructor_parameter_is_answered_by_what_built_the_object_the_walk_came_in_on(tmp_path):
    """``ListWithLock`` is built by twelve knights, and the list a worker reads
    through ``self.taskList.get(n)`` is the one its own knight built: the other
    eleven are other workers' lists, and twelve was too many to follow at all."""
    knight = "    def start(self):\n        fetched = fetch()\n        box = Box(fetched)\n        Worker(box)\n"
    roots = _tree(tmp_path, **{"m.py": _boxes(knight)})

    walked = trace_mod.walk_from_print(roots, file=f"{PACKAGE}/m.py", line=19, classify=provenance.classify)

    (stopped,) = [s for s in walked.steps if s.name == "items" and s.kind == TRACE_UNBOUND]
    assert stopped.detail == (
        "a parameter; followed to the 1 of 10 call(s) placed at __init__: "
        "the ones that built the object the walk came in on"
    )
    assert _built_handovers(walked) == [("start", "fetched")]


def test_a_setter_a_base_in_another_file_declares_is_answered_by_the_calls_on_the_object(tmp_path):
    """``self.items`` assigned by a base's setter one file over is asked about the object the walk came in on,
    as a setter in the class's own file is: ``spare.put(...)`` is another box's."""
    roots = _tree(
        tmp_path,
        **{
            "base.py": "class Holder:\n    def put(self, items):\n        self.items = items\n",
            "m.py": '''\
from pandajedi.base import Holder


class Box(Holder):
    def take(self):
        return self.items


class Worker:
    def __init__(self):
        self.box = Box()

    def run(self, log):
        got = self.box.take()
        log.info(f"took {got}")

    def fill(self):
        self.box.put(fetch())


def other(i):
    spare = Box()
    spare.put([i])
''',
        },
    )

    walked = trace_mod.walk_from_print(roots, file=f"{PACKAGE}/m.py", line=15, classify=provenance.classify)

    (stopped,) = [s for s in walked.steps if s.name == "items" and s.kind == TRACE_UNBOUND]
    assert stopped.owner == f"{PACKAGE}/base.py::put"
    assert stopped.detail == (
        "a parameter; followed to the 1 of 2 call(s) placed at put: "
        "the ones made on the object the walk came in on"
    )
    handed = [(s.owner.rsplit("::", 1)[-1], s.value) for s in walked.steps if s.kind == TRACE_HANDOVER and s.name == "items"]
    assert handed == [("fill", "fetch()")]


def _plugins(deeper: bool = False) -> dict[str, str]:
    """``AdderPluginBase`` and its plugins: a base one file over, three siblings chaining to its ``__init__``."""
    dummy_reads = (
        "    def execute(self):\n        got = self.fetch()\n        return got\n\n"
        "    def fetch(self):\n        found = self.lookup()\n        return found\n\n"
        "    def lookup(self):\n        return self.job\n"
        if deeper
        else "    def execute(self):\n        return self.job\n"
    )
    return {
        "base.py": "class PluginBase:\n    def __init__(self, job, params):\n        self.job = job\n",
        "atlas.py": (
            "from pandajedi.base import PluginBase\n\n\nclass AtlasPlugin(PluginBase):\n"
            "    def __init__(self, job, **params):\n        PluginBase.__init__(self, job, params)\n\n"
            "    def execute(self):\n        return self.job\n"
        ),
        "dummy.py": (
            "from pandajedi.base import PluginBase\n\n\nclass DummyPlugin(PluginBase):\n"
            "    def __init__(self, job, **params):\n        super().__init__(job, params)\n\n" + dummy_reads
        ),
        "simple.py": (
            "from pandajedi.base import PluginBase\n\n\nclass SimplePlugin(PluginBase):\n"
            "    def __init__(self, job, **params):\n        super().__init__(job, params)\n"
        ),
        "runner.py": '''\
from pandajedi.atlas import AtlasPlugin
from pandajedi.dummy import DummyPlugin


def one(log, plugin: AtlasPlugin):
    x = plugin.execute()
    log.info(f"{x}")


def both(log, atlas: AtlasPlugin, dummy: DummyPlugin):
    x = atlas.execute()
    y = dummy.execute()
    log.info(f"{x} {y}")
''',
    }


def _job_steps(walked):
    return [
        (s.kind, s.owner.split("/", 1)[1], s.value, s.detail)
        for s in walked.steps
        if s.name == "job" and s.kind in (TRACE_HANDOVER, TRACE_UNBOUND)
    ]


def test_a_base_constructor_is_answered_by_the_chain_from_the_class_that_read_the_field(tmp_path):
    """``AdderAtlasPlugin`` reads ``self.job``; ``AdderDummyPlugin``'s chain to the base builds another object."""
    roots = _tree(tmp_path, **_plugins())

    walked = trace_mod.walk_from_print(roots, file=f"{PACKAGE}/runner.py", line=7, classify=provenance.classify)

    assert _job_steps(walked) == [
        (
            TRACE_UNBOUND, "base.py::__init__", "job",
            "a parameter; followed to the 1 of 3 call(s) placed at __init__: the ones that can build "
            "an object of AtlasPlugin, the class the walk read the field in",
        ),
        (TRACE_HANDOVER, "atlas.py::__init__", "job", "passed to __init__ here"),
        (
            TRACE_UNBOUND, "atlas.py::__init__", "job",
            "a parameter, so its value is chosen at the call site; AtlasPlugin: no call in the corpus names it "
            "-- reached from outside",
        ),
    ]


def test_a_base_constructor_asked_from_two_classes_follows_each_ones_chain(tmp_path):
    """Asked from both at once, the stop names both; asked from the second later, its chain is followed then."""
    together = trace_mod.walk_from_print(
        _tree(tmp_path / "together", **_plugins()), file=f"{PACKAGE}/runner.py", line=13, classify=provenance.classify
    )
    later = trace_mod.walk_from_print(
        _tree(tmp_path / "later", **_plugins(deeper=True)), file=f"{PACKAGE}/runner.py", line=13,
        classify=provenance.classify,
    )

    (stopped,) = [step for step in _job_steps(together) if step[1] == "base.py::__init__"]
    assert stopped[3] == (
        "a parameter; followed to the 2 of 3 call(s) placed at __init__: the ones that can build "
        "an object of AtlasPlugin or DummyPlugin, the classes the walk read the field in"
    )
    (stopped,) = [step for step in _job_steps(later) if step[1] == "base.py::__init__"]
    assert stopped[3] == (
        "a parameter; followed to the 1 of 3 call(s) placed at __init__: the ones that can build "
        "an object of AtlasPlugin, the class the walk read the field in; "
        "asked again from DummyPlugin, and its 1 call(s) are followed too"
    )
    for walked in (together, later):
        handed = {owner for kind, owner, _value, _detail in _job_steps(walked) if kind == TRACE_HANDOVER}
        assert handed == {"atlas.py::__init__", "dummy.py::__init__"}


def test_a_construction_handed_a_constant_builds_nothing_the_walk_came_in_on(tmp_path):
    """``JobGeneratorThread(None, ...)`` in a test script hands its worker no
    list, so it is not a way the knight's list could have been built."""
    knight = (
        "    def start(self):\n        box = Box(fetch())\n        Worker(box)\n\n"
        "    def check(self):\n        Worker(None)\n"
    )
    roots = _tree(tmp_path, **{"m.py": _boxes(knight)})

    walked = trace_mod.walk_from_print(roots, file=f"{PACKAGE}/m.py", line=19, classify=provenance.classify)

    assert _built_handovers(walked) == [("start", "fetch()")]


def test_an_object_whose_construction_is_not_found_keeps_every_call(tmp_path):
    """A box from ``make_box()`` could be any of them: nothing is narrowed."""
    knight = "    def start(self):\n        box = make_box()\n        Worker(box)\n"
    roots = _tree(tmp_path, **{"m.py": _boxes(knight)})

    walked = trace_mod.walk_from_print(roots, file=f"{PACKAGE}/m.py", line=19, classify=provenance.classify)

    (stopped,) = [s for s in walked.steps if s.name == "items" and s.kind == TRACE_UNBOUND]
    assert stopped.detail == "a parameter; 9 calls are placed at __init__, too many to follow"
    assert _built_handovers(walked) == []


def test_every_object_the_walk_came_in_on_has_its_construction_followed(tmp_path):
    """One frame per function: ``Box.take`` entered on ``self.box`` and, later in
    the walk, from ``backup`` on ``self.spare`` asks for ``self.items`` once --
    settled on the first box before the second came in -- and both boxes'
    builders answer it."""
    knight = (
        "    def start(self):\n        box = Box(fetch())\n        spare = Box(extra())\n"
        "        Worker(box, spare)\n"
    )
    worker = (
        "    def run(self, log):\n"
        "        got = self.box.take()\n"
        "        more = self.backup()\n"
        '        log.info(f"took {got} {more}")\n'
        "\n"
        "    def backup(self):\n"
        "        spare = self.spare.take()\n"
        "        return spare\n"
    )
    roots = _tree(tmp_path, **{"m.py": _boxes(knight, worker)})

    walked = trace_mod.walk_from_print(roots, file=f"{PACKAGE}/m.py", line=20, classify=provenance.classify)

    assert _built_handovers(walked) == [("start", "extra()"), ("start", "fetch()")]


def test_another_object_whose_construction_is_not_found_is_said_and_not_followed(tmp_path):
    """The spare box comes from ``make_box()``: any of the other nine could be
    it, so none is followed for it, and the step says why."""
    knight = "    def start(self):\n        box = Box(fetch())\n        Worker(box, make_box())\n"
    worker = (
        "    def run(self, log):\n"
        "        got = self.box.take()\n"
        "        more = self.backup()\n"
        '        log.info(f"took {got} {more}")\n'
        "\n"
        "    def backup(self):\n"
        "        spare = self.spare.take()\n"
        "        return spare\n"
    )
    roots = _tree(tmp_path, **{"m.py": _boxes(knight, worker)})

    walked = trace_mod.walk_from_print(roots, file=f"{PACKAGE}/m.py", line=20, classify=provenance.classify)

    (stopped,) = [s for s in walked.steps if s.name == "items" and s.kind == TRACE_UNBOUND]
    assert stopped.detail == (
        "a parameter; followed to the 1 of 10 call(s) placed at __init__: "
        "the ones that built the object the walk came in on; "
        f"another object came in later on self.spare at {PACKAGE}/m.py:23, "
        "whose construction is not found, so 9 more call(s) are not followed"
    )
    assert _built_handovers(walked) == [("start", "fetch()")]


def _filled(knight: str, elsewhere: str, box: str = "") -> str:
    """A ``Box`` its knight fills for a worker through ``add``; *elsewhere* fills other boxes."""
    return f"""\
class Box:
    def __init__(self, items):
        self.items = items

    def add(self, item):
        self.items.append(item)

    def take(self):
        return self.items
{box}

class Worker:
    box: Box

    def __init__(self, box):
        self.box = box

    def run(self, log):
        got = self.box.take()
        log.info(f"took {{got}}")


class Knight:
{knight}

{elsewhere}"""


_JUNK = "def other():\n    junk = Box([])\n    junk.add(spill())\n"


def _added(walked):
    return sorted(
        (s.owner.rsplit("::", 1)[-1], s.value)
        for s in walked.steps
        if s.kind == TRACE_HANDOVER and s.name in ("self.items", "item")
    )


def test_a_setter_called_on_another_object_is_not_walked(tmp_path):
    """``ListWithLock.append`` is called on a broker's ``inputListWorld`` and on
    ``JobGeneratorThread``'s ``brokerageLockIDs``; neither is the list a
    ``TaskRefinerThread`` reads, and following them took the walk into
    ``JobGeneratorThread.runImpl``."""
    knight = "    def start(self):\n        box = Box([])\n        box.add(fetch())\n        Worker(box)\n"
    roots = _tree(tmp_path, **{"m.py": _filled(knight, _JUNK)})

    walked = trace_mod.walk_from_print(roots, file=f"{PACKAGE}/m.py", line=20, classify=provenance.classify)

    assert _added(walked) == [("start", "box.add(fetch())"), ("start", "fetch()")]
    (handed,) = [s for s in walked.steps if s.kind == TRACE_HANDOVER and s.name == "self.items"]
    assert handed.detail == "calls add here, which sets items; 1 call(s) of add on other objects not walked"
    (stopped,) = [s for s in walked.steps if s.name == "item" and s.kind == TRACE_UNBOUND]
    assert stopped.detail == (
        "a parameter; followed to the 1 of 2 call(s) placed at add: "
        "the ones made on the object the walk came in on"
    )


def test_a_setter_called_on_self_is_kept(tmp_path):
    """``InputChunk.__init__`` calls ``self.addMasterDS(...)``: the object is
    the one being built, whichever that is."""
    box = "\n    def refill(self):\n        self.add(more())\n"
    knight = "    def start(self):\n        box = Box([])\n        Worker(box)\n"
    roots = _tree(tmp_path, **{"m.py": _filled(knight, _JUNK, box)})

    walked = trace_mod.walk_from_print(roots, file=f"{PACKAGE}/m.py", line=23, classify=provenance.classify)

    assert _added(walked) == [("refill", "more()"), ("refill", "self.add(more())")]


def test_a_setter_never_called_on_the_object_says_so(tmp_path):
    """Every ``add`` is on another box: nothing is walked, and both the
    parameter and what ``add`` assigns say why."""
    knight = "    def start(self):\n        box = Box([])\n        Worker(box)\n"
    roots = _tree(tmp_path, **{"m.py": _filled(knight, _JUNK)})

    walked = trace_mod.walk_from_print(roots, file=f"{PACKAGE}/m.py", line=20, classify=provenance.classify)

    assert _added(walked) == []
    (stopped,) = [s for s in walked.steps if s.name == "item" and s.kind == TRACE_UNBOUND]
    assert stopped.detail == "a parameter; none of the 1 call(s) placed at add is made on the object the walk came in on"
    (filled,) = [s for s in walked.steps if s.kind == TRACE_BINDING and s.owner.endswith("::add")]
    assert filled.detail.endswith("; none of the 1 call(s) of add is made on the object the walk came in on")


def test_where_an_object_was_built_does_not_depend_on_the_walks_budget(tmp_path):
    """A silent-scan walk out of files had made ``JobGeneratorThread``'s
    ``brokerageLockIDs`` an object of unknown construction; a print walk
    found ``JobGenerator.py:410``."""
    knight = "    def start(self):\n        box = Box(fetch())\n        Worker(box)\n"
    roots = _tree(tmp_path, **{"m.py": _boxes(knight)})
    file = f"{PACKAGE}/m.py"
    walk = trace_mod._Walk(roots, provenance.classify, trace_mod.Budget(files=0))
    tree = walk.types.tree(file)
    run = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "run")

    made = walk._made_at(file, run, ast.parse("self.box", mode="eval").body)

    assert made == {(file, 24, 14)}
    assert walk.note == ""


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


def test_a_default_and_the_called_object_are_not_said_to_be_passed(tmp_path):
    """A parameter the call leaves alone holds the signature's default; ``cls`` holds the class called on."""
    roots = _tree(
        tmp_path,
        **{"m.py": '''\
class Spec:
    @classmethod
    def columns(cls, prefix=None):
        if prefix:
            names = prefix
        else:
            names = cls.attributes
        spec.status = names

def build():
    Spec.columns()
'''},
    )

    walked = _walked_in(roots, "m.py", "columns", [8])
    handed = {s.name: s for s in walked.steps if s.kind == "handover"}

    assert handed["prefix"].detail == "left to the default columns's signature gives it"
    assert (handed["prefix"].value, handed["prefix"].reads) == ("None", [])
    assert handed["cls"].detail == "the object columns is called on"
    assert handed["cls"].value == "Spec"
    assert all("passed through * or **" not in s.detail for s in walked.steps)


def test_a_method_read_off_its_class_and_one_called_through_super_hand_the_callers_own_object(tmp_path):
    """``Base.extract(self, 42)`` passes 42 to ``task_id``, not ``self``; ``super().columns()``'s object is ``cls``."""
    roots = _tree(
        tmp_path,
        **{"m.py": '''\
class Base:
    def extract(self, task_id):
        spec.status = task_id

    @classmethod
    def columns(cls):
        spec.names = cls.attributes

class Refiner(Base):
    def run(self):
        Base.extract(self, 42)

    @classmethod
    def columns(cls):
        return super().columns()
'''},
    )

    extract = _walked_in(roots, "m.py", "Base.extract", [3])
    columns = _walked_in(roots, "m.py", "Base.columns", [7])
    handed = [(s.name, s.line, s.value, s.detail) for w in (extract, columns) for s in w.steps if s.kind == "handover"]

    assert ("task_id", 11, "42", "passed to extract here") in handed
    assert ("cls", 15, "cls", "the object columns is called on") in handed
    assert all(value not in ("self", "super()") for _name, _line, value, _detail in handed)


def test_every_walk_in_bamboo_says_whether_bindings_must_reach():
    """``reaching`` is a choice each caller makes; a call that leaves it out made it by default."""
    import ast as ast_mod

    root = Path(trace_mod.__file__).resolve().parents[1]
    silent = []
    for path in sorted(root.rglob("*.py")):
        tree = ast_mod.parse(path.read_text())
        for node in ast_mod.walk(tree):
            if not (isinstance(node, ast_mod.Call) and isinstance(node.func, (ast_mod.Attribute, ast_mod.Name))):
                continue
            called = node.func.attr if isinstance(node.func, ast_mod.Attribute) else node.func.id
            receiver = ast_mod.unparse(node.func.value) if isinstance(node.func, ast_mod.Attribute) else ""
            is_walk = (called == "walk" and receiver in ("trace", "trace_mod")) or (
                path.name == "trace.py" and isinstance(node.func, ast_mod.Name) and called == "walk"
            )
            if is_walk and not any(keyword.arg in ("reaching", None) for keyword in node.keywords):
                silent.append(f"{path.relative_to(root)}:{node.lineno}")

    assert silent == []


_RETRY_PROXY = '''\
def retryJob(jobID, recoverableEsMerge=False):
    spec.status = recoverableEsMerge
'''

_RETRY_FACADE = '''\
from pandajedi import proxy


def retryJob(jobID, recoverableEsMerge=False):
    return proxy.retryJob(jobID, recoverableEsMerge)
'''

_COPY_ARCHIVE = Handover(
    entry=f"{PACKAGE}/copyArchive.py",
    via="retryJob",
    reached_by="call",
    fields={"recoverableEsMerge": "True"},
)


def test_a_call_handover_does_not_hide_the_calls_it_does_not_cross(tmp_path):
    """``retryJob``'s ``recoverableEsMerge``: the handover from ``copyArchive`` crosses the facade's ``retryJob``.

    ``killJobs`` calls the same method itself, and the walk answered from the
    handover alone.  The facade method the handover goes through is that
    handover's own call and is not asked again.
    """
    killer = _RETRY_FACADE + "\n\ndef killJobs(ids):\n    return proxy.retryJob(ids[0], recoverableEsMerge=True)\n"
    roots = _tree(tmp_path, **{"proxy.py": _RETRY_PROXY, "facade.py": killer})

    walked = _walked_in(roots, "proxy.py", "retryJob", [2], handovers=[_COPY_ARCHIVE])
    handed = [(s.owner.rsplit("::", 1)[-1], s.value) for s in walked.steps if s.kind == "handover"]

    assert ("retryJob", "True") in handed
    assert ("killJobs", "True") in handed
    assert not [s for s in walked.steps if s.kind == "handover" and s.file.endswith("facade.py") and s.owner.endswith("::retryJob")]


def test_a_call_handover_is_the_whole_answer_where_it_crosses_every_call(tmp_path):
    roots = _tree(tmp_path, **{"proxy.py": _RETRY_PROXY, "facade.py": _RETRY_FACADE})

    walked = _walked_in(roots, "proxy.py", "retryJob", [2], handovers=[_COPY_ARCHIVE])

    assert [s.kind for s in walked.steps if s.name == "recoverableEsMerge"] == ["handover"]


# -- X5: a parameter's stop says why the calls placed may not be all --------------


def _stop_on(walked, name):
    (stopped,) = [s for s in walked.steps if s.name == name and s.kind == "unbound"]
    return stopped


def test_a_parameter_with_no_caller_says_whether_a_call_by_its_name_was_not_placed(tmp_path):
    """"Chosen at the call site" alone hid that ``obj.g(1)`` may be the call; the stop names it."""
    roots = _tree(tmp_path, **{"m.py": "def g(x):\n    spec.status = x\n\ndef c(obj):\n    obj.g(1)\n"})

    stopped = _stop_on(_walked_in(roots, "m.py", "g", [2]), "x")

    assert stopped.detail == (
        "a parameter, so its value is chosen at the call site; "
        "g: 1 call(s) by its name not placed -- the receiver's class is not declared (obj 1)"
    )


def test_a_parameter_of_a_function_nothing_calls_says_it_is_reached_from_outside(tmp_path):
    roots = _tree(tmp_path, **{"m.py": "def g(x):\n    spec.status = x\n"})

    stopped = _stop_on(_walked_in(roots, "m.py", "g", [2]), "x")

    assert stopped.detail == (
        "a parameter, so its value is chosen at the call site; g: no call in the corpus names it -- reached from outside"
    )


def test_a_parameter_followed_to_its_callers_says_when_more_calls_by_its_name_were_not_placed(tmp_path):
    roots = _tree(tmp_path, **{"m.py": "def g(x):\n    spec.status = x\n\ndef c():\n    g(1)\n\ndef d(obj):\n    obj.g(2)\n"})

    stopped = _stop_on(_walked_in(roots, "m.py", "g", [2]), "x")

    assert stopped.detail == (
        "a parameter; followed to the 1 call(s) placed at g; "
        "g: 1 call(s) by its name not placed -- the receiver's class is not declared (obj 1)"
    )


def test_a_parameter_handed_in_a_modules_own_body_is_written_there_and_not_walked_on(tmp_path):
    """``g(LIMIT)`` at import and ``g(5)`` under ``__main__`` are g's callers; the walk stops at what each hands."""
    roots = _tree(
        tmp_path,
        **{"m.py": 'def g(x):\n    spec.status = x\n\nLIMIT = 3\ng(LIMIT)\n\nif __name__ == "__main__":\n    g(5)\n'},
    )

    walked = _walked_in(roots, "m.py", "g", [2])

    assert _stop_on(walked, "x").detail == "a parameter; followed to the 2 call(s) placed at g"
    handed = [s for s in walked.steps if s.kind == "handover" and s.name == "x"]
    assert [(s.owner, s.line, s.value, s.unseen) for s in handed] == [
        (f"{PACKAGE}/m.py::module-scope", 5, "LIMIT", ["module-scope at 5: run when the file is imported, not on the way here"]),
        (f"{PACKAGE}/m.py::module-scope", 8, "5", ["module-scope at 8: run when the file is run as a script, not on the way here"]),
    ]
    assert not [s for s in walked.steps if s.name == "LIMIT"]


def test_a_modules_own_body_has_no_parameters_for_a_reader_handed_a_call_made_there():
    """A call in a module's body is a caller (:class:`calls.CallSite`), and the readers that ask its function's
    parameters are handed the module: ``JobGeneratorThread(..., brokerageLockIDs, ...)`` in a JEDI test script
    stopped the F case's walk where ``_constructions`` asked the module for its parameters."""
    from bamboo.codemap.panda.recognizers import sqlwrite

    module = ast.parse("brokerageLockIDs = []\nrun(brokerageLockIDs)\n")

    assert trace_mod._parameters(module) == set()
    assert sqlwrite._parameters(module) == []


_OPEN_CALLER = '''\
from typing import Any


class Interface:
    def __getattr__(self, name: str) -> Any:
        return getattr(self.remote, name)


class Buffer:
    def put(self, x):
        spec.status = x


class Feeder:
    def __init__(self, tb: "Interface | Buffer"):
        self.tb = tb

    def run(self):
        self.tb.put(7)
'''


def test_a_handover_from_a_call_placed_open_says_the_call_may_land_elsewhere(tmp_path):
    """``self.tb.put(7)`` lands at ``Buffer.put`` or wherever the interface forwards it: 7 may not be what ``x`` held."""
    roots = _tree(tmp_path, **{"m.py": _OPEN_CALLER})

    walked = _walked_in(roots, "m.py", "Buffer.put", [11])

    (handed,) = [s for s in walked.steps if s.kind == "handover" and s.name == "x"]
    assert (handed.value, handed.line, handed.open) == ("7", 19, calls_mod.UNPLACED_OPEN)


def test_a_value_the_database_supplies_is_read_at_the_call_that_asked_for_it(tmp_path):
    """``getPandaIDsWithTask_JEDI(jediTaskID, True)`` reads the two tables ``onlyActive`` leaves, through the buffer's forwarder.

    A call the map places has routes into the proxy method, read with what
    each hands; a call it cannot place has none, and reads as every run.
    """
    roots = _server(
        tmp_path,
        **{
            "taskbuffer/OraDBProxy.py": '''\
from pandaserver.config import panda_config


class DBProxy:
    def getPandaIDsWithTask_JEDI(self, jediTaskID, onlyActive):
        tables = [f"{panda_config.schemaPANDA}.jobsDefined4", f"{panda_config.schemaPANDA}.jobsActive4"]
        if not onlyActive:
            tables += [f"{panda_config.schemaPANDA}.jobsArchived4"]
        sqlP = ""
        for tableName in tables:
            if sqlP != "":
                sqlP += "UNION ALL "
            sqlP += f"SELECT PandaID FROM {tableName} WHERE jediTaskID=:jediTaskID "
        varMap = {}
        varMap[":jediTaskID"] = jediTaskID
        self.cur.execute(sqlP, varMap)
        return self.cur.fetchall()
''',
            "taskbuffer/DBProxyPool.py": _PROXY_POOL,
            "taskbuffer/TaskBuffer.py": _forwarding_task_buffer("getPandaIDsWithTask_JEDI", "jediTaskID, onlyActive"),
            "daemons/commando.py": '''\
from pandaserver.taskbuffer.TaskBuffer import TaskBuffer


class Commando:
    def __init__(self, taskBufferIF: TaskBuffer):
        self.taskBufferIF = taskBufferIF

    def run(self, jediTaskID, log):
        active = self.taskBufferIF.getPandaIDsWithTask_JEDI(jediTaskID, True)
        every = self.taskBufferIF.getPandaIDsWithTask_JEDI(jediTaskID, False)
        unknown = self.taskBuffer.getPandaIDsWithTask_JEDI(jediTaskID, True)
        log.info(f"{active} {every} {unknown}")
''',
        },
    )

    walked = trace_mod.walk_from_print(
        roots, file="pandaserver/daemons/commando.py", line=12, classify=provenance.classify
    )

    said = {s.name: s.detail for s in walked.steps if s.terminal == models.STOP_UPSTREAM}
    assert said == {
        "active": "getPandaIDsWithTask_JEDI -- reads jobsDefined4, jobsActive4 by :jediTaskID",
        "every": "getPandaIDsWithTask_JEDI -- reads jobsDefined4, jobsActive4, jobsArchived4 by :jediTaskID",
        "unknown": "getPandaIDsWithTask_JEDI -- reads jobsDefined4, jobsActive4, jobsArchived4 by :jediTaskID",
    }


def test_a_value_the_database_supplies_is_read_from_what_the_facade_calls(tmp_path):
    """The facade's method says which proxy methods run, not its name.

    ``querySQL`` calls ``querySQLS``, and the proxy's own ``querySQL`` --
    another method, reading another table -- was what the same name read.
    ``peekJobs`` calls ``peekJob`` once per job and the proxy has no
    ``peekJobs`` at all, so the same name read nothing.  A call the facade
    makes under an ``if`` is read where the caller's constants let it run --
    ``getFiles`` defaults to ``False``.  A name no facade has says nothing,
    though the proxy has a method of it.  A proxy method whose statements are
    handed to it is named beside the tables the others read, not dropped from
    them; one whose statements the call's constants rule out is not, and
    neither is one whose statement names no table.
    """
    roots = _server(
        tmp_path,
        **{
            "taskbuffer/OraDBProxy.py": '''\
class DBProxy:
    def checkBan(self, user):
        sql = "SELECT status FROM ATLAS_PANDA.users WHERE name=:name "
        varMap = {}
        varMap[":name"] = user
        self.cur.execute(sql, varMap)
        return self.cur.fetchall()

    def fetchIDs(self, n):
        sql = "SELECT ATLAS_PANDA.JOBS_SEQ.nextval FROM dual "
        varMap = {}
        varMap[":n"] = n
        self.cur.execute(sql, varMap)
        return self.cur.fetchall()

    def bulkInsert(self, sql, varMaps):
        self.cur.executemany(sql, varMaps)

    def markDone(self, taskID, force):
        if force:
            sql = "UPDATE ATLAS_PANDA.tasksTable SET status=:status WHERE taskID=:taskID "
            varMap = {}
            varMap[":taskID"] = taskID
            varMap[":status"] = "done"
            self.cur.execute(sql, varMap)

    def querySQL(self, sql):
        sql = "SELECT name FROM ATLAS_PANDA.wrongTable WHERE id=:id "
        varMap = {}
        varMap[":id"] = 1
        self.cur.execute(sql, varMap)
        return self.cur.fetchall()

    def querySQLS(self, sql, varMap):
        sql = "SELECT name FROM ATLAS_PANDA.rightTable WHERE name=:name "
        varMap = {}
        varMap[":name"] = "x"
        self.cur.execute(sql, varMap)
        return self.cur.fetchall()

    def peekJob(self, pandaID):
        sql = "SELECT PandaID FROM ATLAS_PANDA.jobsDefined4 WHERE PandaID=:PandaID "
        varMap = {}
        varMap[":PandaID"] = pandaID
        self.cur.execute(sql, varMap)
        return self.cur.fetchall()

    def getDatasets(self, taskID):
        sql = "SELECT datasetID FROM ATLAS_PANDA.datasetsTable WHERE taskID=:taskID "
        varMap = {}
        varMap[":taskID"] = taskID
        self.cur.execute(sql, varMap)
        return self.cur.fetchall()

    def getFiles(self, taskID):
        sql = "SELECT fileID FROM ATLAS_PANDA.filesTable WHERE taskID=:taskID "
        varMap = {}
        varMap[":taskID"] = taskID
        self.cur.execute(sql, varMap)
        return self.cur.fetchall()

    def lonely(self, pandaID):
        sql = "SELECT PandaID FROM ATLAS_PANDA.jobsActive4 WHERE PandaID=:PandaID "
        varMap = {}
        varMap[":PandaID"] = pandaID
        self.cur.execute(sql, varMap)
        return self.cur.fetchall()
''',
            "taskbuffer/DBProxyPool.py": _PROXY_POOL,
            "taskbuffer/TaskBuffer.py": '''\
from pandaserver.taskbuffer.DBProxyPool import DBProxyPool


class TaskBuffer:
    proxyPool: DBProxyPool

    def querySQL(self, sql, varMap):
        with self.proxyPool.get() as proxy:
            ret = proxy.querySQLS(sql, varMap)[1]
        return ret

    def peekJobs(self, pandaIDs):
        found = []
        with self.proxyPool.get() as proxy:
            for pandaID in pandaIDs:
                found.append(proxy.peekJob(pandaID))
        return found

    def getDatasets(self, taskID, getFiles=False):
        with self.proxyPool.get() as proxy:
            ret = proxy.getDatasets(taskID)
            if getFiles:
                proxy.getFiles(taskID)
        return ret

    def storeJobs(self, user, sql, varMaps):
        with self.proxyPool.get() as proxy:
            ret = proxy.checkBan(user)
            proxy.fetchIDs(len(varMaps))
            proxy.bulkInsert(sql, varMaps)
        return ret

    def insertOnly(self, sql, varMaps):
        with self.proxyPool.get() as proxy:
            ret = proxy.bulkInsert(sql, varMaps)
        return ret

    def finish(self, taskID):
        with self.proxyPool.get() as proxy:
            ret = proxy.getDatasets(taskID)
            proxy.markDone(taskID, False)
        return ret
''',
            "daemons/commando.py": '''\
from pandaserver.taskbuffer.TaskBuffer import TaskBuffer


class Commando:
    def __init__(self, taskBufferIF: TaskBuffer):
        self.taskBufferIF = taskBufferIF

    def run(self, pandaIDs, log):
        names = self.taskBufferIF.querySQL("SELECT", {})
        jobs = self.taskBufferIF.peekJobs(pandaIDs)
        alone = self.taskBufferIF.lonely(pandaIDs)
        plain = self.taskBufferIF.getDatasets(1)
        full = self.taskBufferIF.getDatasets(1, getFiles=True)
        stored = self.taskBufferIF.storeJobs("me", "INSERT", [])
        inserted = self.taskBufferIF.insertOnly("INSERT", [])
        finished = self.taskBufferIF.finish(1)
        log.info(f"{names} {jobs} {alone} {plain} {full} {stored} {inserted} {finished}")
''',
        },
    )

    walked = trace_mod.walk_from_print(
        roots, file="pandaserver/daemons/commando.py", line=17, classify=provenance.classify
    )

    said = {s.name: s.detail for s in walked.steps if s.terminal == models.STOP_UPSTREAM}
    assert said == {
        "plain": "getDatasets -- reads datasetsTable by :taskID",
        "full": "getDatasets -- reads datasetsTable, filesTable by :taskID",
        "names": "querySQL -- reads rightTable by :name",
        "jobs": "peekJobs -- reads jobsDefined4 by :PandaID",
        "alone": "lonely",
        "stored": "storeJobs -- reads users by :name; what bulkInsert runs is not read",
        "inserted": "insertOnly",
        "finished": "finish -- reads datasetsTable by :taskID",
    }


def test_what_start_returns_is_not_what_run_returns(tmp_path):
    """``thr.start()`` hands ``run`` to another thread and returns ``None``; the walk does not answer it from ``run``."""
    roots = _tree(
        tmp_path,
        m='''\
import threading

class Worker(threading.Thread):
    def run(self):
        return "from run"

def go(spec):
    thr = Worker()
    started = thr.start()
    spec.status = started
''',
    )

    said = _said(_walked_in(roots, "m", "go", [10]))

    assert not [one for one in said if one[2] == "run"]


def test_a_function_a_wrapper_hands_its_arguments_to_runs_on_the_object_it_was_read_off(tmp_path):
    """``timed.run()`` calls ``Spec.columns()`` (``calls.Placed.forwarded``): ``cls`` holds ``Spec``, the object the
    function was read off, not the ``Timed`` the call is made on."""
    roots = _tree(
        tmp_path,
        **{"m.py": '''\
class Timed:
    def __init__(self, method, timeout):
        self.method = method

    def run(self, *var, **kwargs):
        self.method(*var, **kwargs)


class Spec:
    @classmethod
    def columns(cls, prefix=None):
        if prefix:
            names = prefix
        else:
            names = cls.attributes
        spec.status = names


def build():
    timed = Timed(Spec.columns, 10)
    timed.run()
'''},
    )

    walked = _walked_in(roots, "m.py", "columns", [16])
    handed = {s.name: s for s in walked.steps if s.kind == "handover"}

    assert handed["cls"].detail == "the object columns is called on"
    assert handed["cls"].value == "Spec"


def _asker_in(roots, file, owner, lines, **kwargs):
    return trace_mod.Asker(
        roots,
        file=f"{PACKAGE}/{file}",
        owner=f"{PACKAGE}/{file}::{owner}",
        lines=lines,
        classify=provenance.classify,
        reaching=True,
        **kwargs,
    )


def _asked_through(asker):
    """``{question: answer}``: the start's under ``None``, then each frontier question's once, breadth first."""
    answers = {None: asker.start()}
    queue = list(answers[None].frontier)
    while queue:
        question = queue.pop(0)
        if question not in answers:
            answers[question] = asker.ask(question)
            queue.extend(answers[question].frontier)
    return answers


def _step_keys(steps):
    return {(s.kind, s.file, s.owner, s.line, s.name, s.value) for s in steps}


def test_a_question_is_answered_one_hop_and_every_question_asked_once_is_the_walk(tmp_path):
    """The querier's trace questions.  The start is the arm alone, what it reads and calls its frontier; each
    answer is one depth and hands back what it did not follow; the calls of the setter of a field are a question
    of their own.  Every question asked once, the answers hold the walk's steps."""
    roots = _tree(tmp_path, **_SET_BY_A_SETTER)

    answers = _asked_through(_asker_in(roots, "gen.py", "run", [8]))

    start = answers[None]
    assert [(s.kind, s.line) for s in start.steps] == [(TRACE_WRITE, 8)]
    assert {(q.kind, q.name, q.line) for q in start.frontier} == {
        (trace_mod.ASK_BINDINGS, "self.add_result", 7),
        (trace_mod.ASK_RETURNS, "self.add_result.is_succeeded()", 7),
    }
    assert all(len({s.depth for s in answer.steps}) <= 1 for answer in answers.values())
    (setter,) = [q for q in answers if q is not None and q.kind == trace_mod.ASK_SETTER_CALLS]
    assert (setter.name, setter.owner.rsplit("::", 1)[-1], [m.kind for m in setter.chain]) == (
        "self.status_code",
        "set_fatal",
        [trace_mod.MOVE_DOWN, trace_mod.MOVE_INTO],
    )
    assert [(s.owner.rsplit("::", 1)[-1], s.line, s.value) for s in answers[setter].steps] == [
        ("register", 10, "self.result.set_fatal()")
    ]
    walked = _walked_in(roots, "gen.py", "run", [8])
    assert _step_keys(s for answer in answers.values() for s in answer.steps) == _step_keys(walked.steps)


def test_a_question_s_answer_is_the_same_whatever_was_asked_before(tmp_path):
    """The querier keeps an answer per question and asks in its own order: each is a walk of its own."""
    roots = _tree(tmp_path, **_SET_BY_A_SETTER)
    answers = _asked_through(_asker_in(roots, "gen.py", "run", [8]))
    other = _asker_in(roots, "gen.py", "run", [8])

    for question in reversed([q for q in answers if q is not None]):
        again = other.ask(question)
        assert (again.steps, again.frontier) == (answers[question].steps, answers[question].frontier)


def test_a_question_in_a_function_the_walk_went_into_carries_the_call_it_came_by(tmp_path):
    """``helper``'s ``x`` is what ``f``'s call hands it, as the walk reads it there, and not ``g``'s ``7``: the
    question's chain names the call."""
    roots = _tree(
        tmp_path,
        m='''\
def helper(x):
    return x + 1

def f(a):
    b = helper(a)
    spec.status = b

def g():
    return helper(7)
''',
    )

    answers = _asked_through(_asker_in(roots, "m", "f", [6]))

    (asked,) = [q for q in answers if q is not None and q.name == "x"]
    assert [(m.kind, m.owner.rsplit("::", 1)[-1]) for m in asked.chain] == [(trace_mod.MOVE_DOWN, "helper")]
    assert [(s.kind, s.owner.rsplit("::", 1)[-1], s.value) for s in answers[asked].steps] == [(TRACE_HANDOVER, "f", "a")]


def test_a_name_the_arms_read_is_their_question_however_the_walk_comes_back_to_it(tmp_path):
    """``f``'s ``a`` is what the arm reads, and what ``f`` hands ``count``'s ``n``.  The walk answers it once, as
    the arm's read; asked again from ``n``, it is the arms' question still, not a second one."""
    roots = _tree(tmp_path, m="def count(n):\n    return n + 1\n\ndef f(a):\n    spec.status = count(a)\n")

    answers = _asked_through(_asker_in(roots, "m", "f", [5]))

    assert [q.armed for q in answers if q is not None and q.name == "a"] == [True]
    assert {(q.name, q.armed) for q in answers if q is not None and q.name == "n"} == {("n", False)}


def test_a_question_s_chain_stops_at_the_depth_budget_where_a_recursion_grows_it(tmp_path):
    """``count`` calls itself, and its ``n`` is what each call hands it: going up through that call comes back to
    ``count`` one move longer each time.  A question has no depth; the depth budget caps its chain's moves, and
    asked through, the questions end.  The start keeps the empty chain where a move comes back to it, so ``f``,
    one call up from the start, is one move away."""
    roots = _tree(
        tmp_path,
        **{
            "m.py": '''\
def count(n, spec):
    if n > 0:
        count(n - 1, spec)
    spec.status = n

def f(spec, k):
    count(k, spec)
'''
        },
    )
    budget = trace_mod.Budget(depth=3, crossed_depth=0)

    answers = _asked_through(_asker_in(roots, "m.py", "count", [4], budget=budget))

    assert max(len(q.chain) for q in answers if q is not None) == 2
    assert any("depth budget" in answer.note for answer in answers.values())
    assert {tuple(one.owner.rsplit("::", 1)[-1] for one in q.chain) for q in answers if q is not None and q.name == "k"} == {
        ("f",),
        ("count", "f"),
    }
    walked = _walked_in(roots, "m.py", "count", [4], budget=budget)
    assert _step_keys(s for answer in answers.values() for s in answer.steps) == _step_keys(walked.steps)


def test_a_question_about_what_a_call_returns_answers_for_that_call(tmp_path):
    """``getTask`` is entered by a call placed open and by one placed closed.  The walk marks its return open
    once any call placed open entered it; the question about the closed call answers for that call alone."""
    roots = _tree(
        tmp_path, m=_OPEN_ENTRY.format(first="self.source.getTask(a)", second="self.buffer.getTask(a)")
    )

    answers = _asked_through(_asker_in(roots, "m", "run", [26]))

    marks = {
        q.name: {s.open for s in answer.steps}
        for q, answer in answers.items()
        if q is not None and q.kind == trace_mod.ASK_RETURNS and q.name.endswith("getTask(a)")
    }
    assert marks == {"self.source.getTask(a)": {calls_mod.UNPLACED_OPEN}, "self.buffer.getTask(a)": {""}}


def test_a_question_says_on_the_field_s_step_that_no_setter_call_is_on_the_object(tmp_path):
    """Every ``add`` is on another box: the walk says so on the step of what ``add`` assigns, and the question
    about ``self.items`` does too -- there is no call of the setter to ask about."""
    knight = "    def start(self):\n        box = Box([])\n        Worker(box)\n"
    roots = _tree(tmp_path, **{"m.py": _filled(knight, _JUNK)})

    answers = _asked_through(_asker_in(roots, "m.py", "Worker.run", [19]))

    (filled,) = [
        s for answer in answers.values() for s in answer.steps if s.kind == TRACE_BINDING and s.owner.endswith("::add")
    ]
    assert filled.detail.endswith("; none of the 1 call(s) of add is made on the object the walk came in on")
    assert not [q for q in answers if q is not None and q.kind == trace_mod.ASK_SETTER_CALLS]


def test_a_question_spells_a_function_by_its_file_whichever_frame_came_first(tmp_path):
    """``Box`` and ``Worker`` each have an ``__init__``.  The walk spells the first frame it makes plainly and the
    second with its class, so a question that makes only one would say ``__init__`` where another answer says
    ``Box.__init__`` -- and the querier joins answers by the owner.  A question spells both with the class, and
    a function whose name is the file's alone plainly."""
    knight = "    def start(self):\n        box = Box([])\n        Worker(box)\n"
    roots = _tree(tmp_path, **{"m.py": _filled(knight, _JUNK)})

    answers = _asked_through(_asker_in(roots, "m.py", "Worker.run", [19]))

    owners = {s.owner for answer in answers.values() for s in answer.steps}
    owners |= {one.owner for q in answers if q is not None for one in (q, *q.chain)}
    assert {owner.rsplit("::", 1)[-1] for owner in owners if owner.endswith("__init__")} == {
        "Box.__init__",
        "Worker.__init__",
    }
    assert f"{PACKAGE}/m.py::start" in owners


def test_a_slot_of_a_split_prints_as_a_token_not_as_the_list(tmp_path):
    """``source, destination = src_dst.split(":")`` (``NetworkConfigurator.process_nws_dump``): a split's value prints
    as a list, and each slot is one element of it -- a token, read by its own name.  ``f"{parts}"`` of the whole list
    prints as a text."""
    roots = _tree(
        tmp_path,
        m='''\
def f(job, log, src_dst):
    job.status = "failed"
    source, destination = src_dst.split(":")
    log.error(f"entry ({source}->{destination}) failed")
    parts = src_dst.split(":")
    log.error(f"entry {parts} failed")
''',
    )

    (row,) = _rows_at(roots, 4)
    (whole,) = _rows_at(roots, 6)

    assert [(hole.expression, hole.prints) for hole in row.holes] == [("source", "token"), ("destination", "token")]
    assert [hole.prints for hole in whole.holes] == ["text"]
