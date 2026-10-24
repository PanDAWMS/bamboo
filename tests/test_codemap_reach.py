"""Whether a function reaches a write, recorded per function and answered from the record -- ``codemap.reach``."""

from __future__ import annotations

import ast
from pathlib import Path

from bamboo.codemap import calls as calls_mod
from bamboo.codemap import reach
from bamboo.codemap.models import Anchor, Branch, JunctionNode

PACKAGE = "pkg"

_WORKERS = {
    "threads": (
        "import threading\n\n"
        "class WorkerThread(threading.Thread):\n"
        "    def runImpl(self):\n"
        "        raise NotImplementedError\n\n"
        "    def run(self):\n"
        "        self.runImpl()\n"
    ),
    "gen": (
        "from pkg.threads import WorkerThread\n\n"
        "class JobGeneratorThread(WorkerThread):\n"
        "    def runImpl(self):\n"
        "        return 'gen'\n\n"
        "def generate():\n"
        "    thr = JobGeneratorThread()\n"
        "    thr.start()\n"
    ),
    "cmd": (
        "from pkg.threads import WorkerThread\n\n"
        "class TaskCommandoThread(WorkerThread):\n"
        "    def runImpl(self):\n"
        "        write_status()\n\n"
        "def command():\n"
        "    thr = TaskCommandoThread()\n"
        "    thr.start()\n\n"
        "def write_status():\n"
        "    pass\n"
    ),
}


def _tree(tmp_path: Path, **files: str) -> dict[str, Path]:
    root = tmp_path / PACKAGE
    root.mkdir(parents=True, exist_ok=True)
    (root / "__init__.py").write_text("")
    for name, text in files.items():
        (root / f"{name}.py").write_text(text)
    return {PACKAGE: root}


def _record(tmp_path: Path, leaves=None, **files: str) -> reach.ReachGraph:
    calls = calls_mod.Calls(calls_mod.Types(_tree(tmp_path, **files), {}))
    return reach.ReachGraph(reach.graph(calls, "m", "v", leaves))


def _to(record: reach.ReachGraph, owner: str, how: str = "") -> list[str]:
    return sorted(
        call.to for call in record.rows[owner].calls if not how or call.how == how
    )


def _writing(owner: str) -> reach.Destination:
    return reach.Destination({owner: ["done@1"]}, [])


def test_a_thread_started_as_one_class_reaches_that_class_s_override_alone(tmp_path):
    """``WorkerThread.run`` entered as a JobGeneratorThread's goes on into its ``runImpl``, not TaskCommandoThread's."""
    record = _record(tmp_path, **_WORKERS)
    destination = _writing("pkg/cmd.py::write_status")

    generated = record.reach(["pkg/gen.py::generate"], destination)
    commanded = record.reach(["pkg/cmd.py::command"], destination)

    assert generated.writes == []
    assert generated.closure == 3
    (write,) = commanded.writes
    assert write.function == "pkg/cmd.py::write_status"
    assert write.chain == ["pkg/cmd.py:9", "pkg/threads.py:8", "pkg/cmd.py:5"]


def test_a_call_on_self_is_recorded_once_for_each_class_self_can_be(tmp_path):
    """The record keeps which class each landing holds for, so no reader has to resolve inheritance again."""
    record = _record(tmp_path, **_WORKERS)

    run = record.rows["pkg/threads.py::WorkerThread.run"]

    assert sorted((call.for_classes, call.runs_as, call.to) for call in run.calls) == [
        (
            ["pkg/cmd.py::TaskCommandoThread"],
            ["pkg/cmd.py::TaskCommandoThread"],
            "pkg/cmd.py::TaskCommandoThread.runImpl",
        ),
        (
            ["pkg/gen.py::JobGeneratorThread"],
            ["pkg/gen.py::JobGeneratorThread"],
            "pkg/gen.py::JobGeneratorThread.runImpl",
        ),
        (
            ["pkg/threads.py::WorkerThread"],
            ["pkg/threads.py::WorkerThread"],
            "pkg/threads.py::WorkerThread.runImpl",
        ),
    ]
    (start,) = record.rows["pkg/gen.py::generate"].calls
    assert start.thread and start.runs_as == ["pkg/gen.py::JobGeneratorThread"]


def test_calls_going_one_way_are_one_row_holding_every_line(tmp_path):
    """``insertNewJob`` calls ``recordStatusChange`` in a ``try`` and after it: one row, both lines, as each class's."""
    record = _record(
        tmp_path,
        jobs=(
            "def record(job):\n"
            "    pass\n"
            "def insert(job):\n"
            "    try:\n"
            "        record(job)\n"
            "    except Exception:\n"
            "        pass\n"
            "    record(job)\n"
        ),
        steps=(
            "class Base:\n"
            "    def step(self):\n"
            "        pass\n"
            "    def run(self, again):\n"
            "        if again:\n"
            "            self.step()\n"
            "        self.step()\n"
            "class Child(Base):\n"
            "    def step(self):\n"
            "        pass\n"
        ),
    )

    (call,) = record.rows["pkg/jobs.py::insert"].calls
    assert (call.to, call.lines, call.line) == ("pkg/jobs.py::record", [5, 8], 8)
    (write,) = record.reach(
        ["pkg/jobs.py::insert"], _writing("pkg/jobs.py::record")
    ).writes
    assert write.chain == ["pkg/jobs.py:8"]
    assert sorted(
        (call.to, call.lines) for call in record.rows["pkg/steps.py::Base.run"].calls
    ) == [("pkg/steps.py::Base.step", [6, 7]), ("pkg/steps.py::Child.step", [6, 7])]


def test_a_method_entered_with_no_class_said_runs_as_every_class_self_can_be(tmp_path):
    """Asked from ``WorkerThread.run`` itself, the object may be any worker: every override is in the closure."""
    record = _record(tmp_path, **_WORKERS)

    answer = record.reach(
        ["pkg/threads.py::WorkerThread.run"], _writing("pkg/cmd.py::write_status")
    )

    assert [write.function for write in answer.writes] == ["pkg/cmd.py::write_status"]


def test_no_write_and_a_hole_is_no_answer_and_no_write_and_no_hole_is_none_reached(
    tmp_path,
):
    """``None`` and ``[]`` are two answers: only the second says the start cannot reach the write."""
    record = _record(
        tmp_path,
        mod=(
            "def closed():\n"
            "    len([])\n\n"
            "def opened(fn):\n"
            "    fn()\n\n"
            "def write():\n"
            "    pass\n"
        ),
    )
    destination = _writing("pkg/mod.py::write")

    assert record.reach(["pkg/mod.py::closed"], destination).writes == []
    answer = record.reach(["pkg/mod.py::opened"], destination)
    assert answer.writes is None
    assert answer.holes == [f"pkg/mod.py::opened:5 fn -- {reach.HOLE_LOCAL_CALLED}"]


def test_a_function_handed_as_an_argument_is_taken_as_called_and_a_class_handed_to_isinstance_is_not(
    tmp_path,
):
    """``Thread(target=self.work)`` goes on into ``work``; ``isinstance(x, Spec)`` builds no ``Spec``."""
    record = _record(
        tmp_path,
        mod=(
            "import threading\n\n"
            "class Spec:\n"
            "    def __init__(self):\n"
            "        pass\n\n"
            "class Agent:\n"
            "    def work(self):\n"
            "        pass\n\n"
            "    def go(self, x):\n"
            "        threading.Thread(target=self.work).start()\n"
            "        isinstance(x, Spec)\n"
        ),
    )

    assert _to(record, "pkg/mod.py::Agent.go", "handed") == ["pkg/mod.py::Agent.work"]
    assert _to(record, "pkg/mod.py::Agent.go") == ["pkg/mod.py::Agent.work"]


def test_a_call_in_a_lambda_s_body_is_taken_as_made_by_the_function_holding_it(
    tmp_path,
):
    record = _record(
        tmp_path,
        mod=(
            "def write():\n    pass\n\ndef later(rows):\n    rows.sort(key=lambda row: write())\n"
        ),
    )

    assert _to(record, "pkg/mod.py::later", "lambda") == ["pkg/mod.py::write"]


def test_putting_a_value_on_a_queue_is_a_hole(tmp_path):
    """Whoever ``get()``s it runs elsewhere, and what it does with the value is in no closure of this one."""
    record = _record(
        tmp_path,
        mod=(
            "import multiprocessing\n\n"
            "class Master:\n"
            "    def __init__(self):\n"
            "        self.msg_queue: multiprocessing.Queue[str] = multiprocessing.Queue()\n\n"
            "    def hand(self, name: str):\n"
            "        self.msg_queue.put(name)\n"
            "        self.msg_queue.qsize()\n"
        ),
    )

    holes = record.rows["pkg/mod.py::Master.hand"].holes

    assert [(hole.line, hole.why) for hole in holes] == [(8, reach.HOLE_QUEUE)]


def test_a_call_on_a_builtin_value_is_closed_by_a_premise_and_one_on_an_outside_value_by_an_assumption(
    tmp_path,
):
    """``rows.append`` on a list is a builtin's method; ``resp.json()`` on what ``requests`` returns rests on it returning no corpus object."""
    record = _record(
        tmp_path,
        mod=(
            "import requests\n\n"
            "class Holder:\n"
            "    def json(self):\n"
            "        pass\n\n"
            "def fetch(rows: list):\n"
            "    rows.append(1)\n"
            "    resp = requests.get('https://example.invalid')\n"
            "    resp.json()\n"
        ),
    )

    row = record.rows["pkg/mod.py::fetch"]

    assert row.holes == []
    assert row.settled == {"outside_returns_outside": 1}


def test_what_outside_code_returns_from_a_corpus_object_it_was_handed_is_not_taken_as_outside(
    tmp_path,
):
    """``random.choice(job)`` may return the corpus object it is handed: a call on it is a hole, not an outside call.

    ``copy.copy`` and ``copy.deepcopy`` were the example once; they are read
    as the class they copy now (``calls._COPIES``), and a call on a copy is
    placed.
    """
    record = _record(
        tmp_path,
        mod=(
            "import random\n\n"
            "class Job:\n"
            "    def flush(self):\n"
            "        pass\n\n"
            "def clone(job: Job):\n"
            "    random.choice(job).flush()\n"
        ),
    )

    (hole,) = record.rows["pkg/mod.py::clone"].holes

    assert hole.callee == "random.choice(job).flush"
    assert "handed job" in hole.why


def test_a_name_rebound_from_itself_is_what_its_other_bindings_are(tmp_path):
    """``diag = re.sub(..., diag)`` beside ``diag: str`` adds nothing: ``diag.encode()`` is a str's, not a hole.

    The rebinding is read inside its own reading; read as unknown, it made
    every caller of ``JediTaskSpec.setErrDiag`` a closure with a hole.
    """
    record = _record(
        tmp_path,
        mod=(
            "import re\n\n"
            "class Job:\n"
            "    def encode(self):\n"
            "        pass\n\n"
            "def set_diag(diag: str | None):\n"
            "    if diag:\n"
            "        diag.encode()\n"
            "        diag = re.sub('x', 'y', diag)\n\n"
            "def mixed(diag: str, job: Job):\n"
            "    diag = re.sub('x', 'y', diag)\n"
            "    diag = job\n"
            "    diag.encode()\n"
        ),
    )

    assert record.rows["pkg/mod.py::set_diag"].holes == []
    # The rebinding adds nothing, and the other bindings still say what they say.
    assert [hole.callee for hole in record.rows["pkg/mod.py::mixed"].holes] == [
        "diag.encode"
    ]


def test_a_method_no_class_of_the_corpus_defines_is_closed_by_assuming_none_is_added_at_run_time(
    tmp_path,
):
    record = _record(
        tmp_path,
        mod=(
            "class Job:\n    def flush(self):\n        pass\n\ndef tidy(x):\n    x.strip()\n    x.flush()\n"
        ),
    )

    row = record.rows["pkg/mod.py::tidy"]

    assert row.settled == {"no_runtime_methods": 1}
    assert [hole.callee for hole in row.holes] == ["x.flush"]


def test_a_method_a_class_lacks_is_closed_by_a_premise_where_it_has_an_outside_base(
    tmp_path,
):
    """``self.getName()`` on a ``threading.Thread`` subclass is Thread's own; on a class with no outside base it is not there."""
    record = _record(
        tmp_path,
        mod=(
            "import threading\n\n"
            "class Worker(threading.Thread):\n"
            "    def tell(self):\n"
            "        self.getName()\n\n"
            "class Plain:\n"
            "    def tell(self):\n"
            "        self.getName()\n"
        ),
    )

    assert record.rows["pkg/mod.py::Worker.tell"].settled == {}
    assert record.rows["pkg/mod.py::Plain.tell"].settled == {"no_runtime_methods": 1}


def test_a_method_one_declared_class_lacks_and_has_no_outside_base_for_rests_on_an_assumption(
    tmp_path,
):
    """``w: Worker | Plain``: ``getName`` is Thread's on a Worker, and on a Plain only if something adds it at run time."""
    record = _record(
        tmp_path,
        mod=(
            "import threading\n\n"
            "class Worker(threading.Thread):\n"
            "    pass\n\n"
            "class Plain:\n"
            "    pass\n\n"
            "def tell(w: Worker | Plain):\n"
            "    w.getName()\n"
        ),
    )

    assert record.rows["pkg/mod.py::tell"].settled == {"no_runtime_methods": 1}


def test_a_base_named_through_a_subscript_is_the_corpus_class_and_generic_supplies_no_method(
    tmp_path,
):
    """``class JobBroker(FactoryBase[JobBrokerBase])``: FactoryBase is no outside base, and ``Generic[T]`` supplies nothing.

    A name FactoryBase's order lacks rests on the assumption, as on a class
    with no base -- not on an outside base that might supply it.
    """
    record = _record(
        tmp_path,
        mod=(
            "from typing import Generic, TypeVar\n\n"
            "T = TypeVar('T')\n\n"
            "class FactoryBase(Generic[T]):\n"
            "    def requireImpl(self):\n"
            "        self.lost()\n\n"
            "class JobBroker(FactoryBase['JobBroker']):\n"
            "    def doBrokerage(self):\n"
            "        self.requireImpl()\n"
            "        self.lost()\n"
        ),
    )

    broker = record.rows["pkg/mod.py::JobBroker.doBrokerage"]
    assert _to(record, "pkg/mod.py::JobBroker.doBrokerage") == [
        "pkg/mod.py::FactoryBase.requireImpl"
    ]
    assert broker.settled == {"no_runtime_methods": 1}
    assert record.rows["pkg/mod.py::FactoryBase.requireImpl"].settled == {
        "no_runtime_methods": 1
    }


def test_start_on_a_class_built_on_an_outside_class_that_has_a_run_is_a_hole(tmp_path):
    """``GenericThread`` is a thread nothing in the corpus says is one: its ``start()`` may run ``AdderThread.run``."""
    record = _record(
        tmp_path,
        mod=(
            "from pandacommon.pandautils.thread_utils import GenericThread\n\n"
            "class AdderThread(GenericThread):\n"
            "    def run(self):\n"
            "        pass\n\n"
            "def main():\n"
            "    thr = AdderThread()\n"
            "    thr.start()\n"
        ),
    )

    assert [hole.why for hole in record.rows["pkg/mod.py::main"].holes] == [
        reach.HOLE_OUTSIDE_START
    ]


def test_a_class_handed_to_a_builtin_that_never_calls_it_builds_nothing(tmp_path):
    """``super(Base, self)`` and ``getattr(Base, name)`` read the class; neither runs its ``__init__``."""
    record = _record(
        tmp_path,
        mod=(
            "class Base:\n"
            "    def __init__(self):\n"
            "        pass\n\n"
            "class Child(Base):\n"
            "    def __init__(self):\n"
            "        super(Base, self).__init__()\n"
            "        getattr(Base, 'x', None)\n"
        ),
    )

    assert _to(record, "pkg/mod.py::Child.__init__", "handed") == []


def test_a_function_set_on_a_class_is_handed_and_taken_as_called(tmp_path):
    """``setattr(Workflow, "scatter", scatter)`` installs it as a method something calls later: an edge, not a builtin's read."""
    record = _record(
        tmp_path,
        mod=(
            "from snakemake.workflow import Workflow\n\n"
            "def scatter(self):\n"
            "    pass\n\n"
            "def inject():\n"
            "    setattr(Workflow, 'scatter', scatter)\n"
        ),
    )

    assert _to(record, "pkg/mod.py::inject", "handed") == ["pkg/mod.py::scatter"]


def test_an_operation_is_read_from_its_operands_and_from_what_the_corpus_s_dunders_return(
    tmp_path,
):
    """``a * b`` may run ``Share.__imul__``, which returns a ``Share``; ``a + b`` runs no corpus code, nor ``n * 2`` on ints."""
    record = _record(
        tmp_path,
        mod=(
            "class Share:\n"
            "    def __imul__(self, other: float) -> 'Share':\n"
            "        return self\n\n"
            "    def flush(self):\n"
            "        pass\n\n"
            "def scale(x, n: int):\n"
            "    (x * 2).flush()\n"
            "    (x + 2).flush()\n"
            "    (n * 2).flush()\n"
        ),
    )

    row = record.rows["pkg/mod.py::scale"]

    assert [hole.line for hole in row.holes] == [9]
    assert row.settled == {"outside_returns_outside": 1}


def test_a_repeated_name_in_a_file_is_told_apart(tmp_path):
    record = _record(
        tmp_path,
        mod=(
            "class Spec:\n"
            "    @property\n"
            "    def status(self):\n"
            "        return 1\n\n"
            "    @status.setter\n"
            "    def status(self, value):\n"
            "        pass\n"
        ),
    )

    assert sorted(record.rows) == [
        "pkg/mod.py::Spec.status",
        "pkg/mod.py::Spec.status#2",
    ]


def _junction(owner: str, line: int, arms: list[tuple[str, list[str]]]) -> JunctionNode:
    return JunctionNode(
        name=f"m:{owner}",
        map_id="m",
        derived_from="v",
        subject="JobSpec.jobStatus",
        owner=owner,
        anchor=Anchor(package=PACKAGE, file=owner.partition("::")[0], line_start=line),
        branches=[
            Branch(outcome=outcome, line=line, row_precondition=guard)
            for outcome, guard in arms
        ],
    )


def test_the_arms_that_move_a_subject_off_a_value_leave_out_the_ones_writing_it_and_the_ones_that_need_another(
    tmp_path,
):
    """From ``running``: not an arm writing ``running``, nor one whose row must hold ``activated``; one that may hold either is kept."""
    record = _record(tmp_path, mod=("def update(job):\n    pass\n"))
    junctions = [
        _junction(
            "pkg/mod.py::update",
            2,
            [
                ("running", []),
                ("failed", ["jobStatus=activated"]),
                ("finished", ["jobStatus IN (activated,running)"]),
                ("cancelled", []),
            ],
        ),
        _junction("pkg/gone.py::vanished", 3, [("failed", [])]),
    ]

    destination = reach.moving_off(junctions, record, "JobSpec.jobStatus", "running")

    assert destination.arms == {"pkg/mod.py::update": ["finished@2", "cancelled@2"]}
    assert destination.unmatched == ["pkg/gone.py::vanished"]
    answer = record.reach(["pkg/mod.py::update"], destination)
    assert [write.function for write in answer.writes] == ["pkg/mod.py::update"]
    assert answer.holes == [
        "pkg/gone.py::vanished -- a write the record has no function for"
    ]


def test_a_write_the_record_has_no_function_for_leaves_no_answer(tmp_path):
    """A junction no row was found for may be anywhere: "reaches none" would be a guess."""
    record = _record(tmp_path, mod=("def quiet():\n    pass\n"))

    answer = record.reach(
        ["pkg/mod.py::quiet"], reach.Destination({}, ["pkg/gone.py::vanished"])
    )

    assert answer.writes is None


def test_a_build_s_record_is_the_same_twice(tmp_path):
    roots = _tree(tmp_path, **_WORKERS)
    first = reach.graph(calls_mod.Calls(calls_mod.Types(roots, {})), "m", "v")
    second = reach.graph(calls_mod.Calls(calls_mod.Types(roots, {})), "m", "v")

    assert [row.model_dump() for row in first] == [row.model_dump() for row in second]


def _database(tmp_path, text: str) -> reach.ReachGraph:
    """The record of ``pkg/db.py`` as a build makes it: the plugin's database leaves, statements read through the module's calls."""
    from bamboo.codemap.gitsource import blob_sha
    from bamboo.codemap.models import SourceModule
    from bamboo.codemap.panda import pathcond, sql
    from bamboo.codemap.panda.plugin import _DatabaseLeaves
    from bamboo.codemap.panda.recognizers import sqlwrite

    roots = _tree(tmp_path, db=text)
    tree = ast.parse(text)
    pathcond.attach_parents(tree)
    module = SourceModule(
        package=PACKAGE,
        rel_path="pkg/db.py",
        tree=tree,
        source=text,
        blob_sha=blob_sha(text),
    )
    calls = sqlwrite.corpus_calls([module], roots)
    with sql.across_calls(calls):
        return reach.ReachGraph(reach.graph(calls, "m", "v", _DatabaseLeaves()))


def test_a_statement_the_map_reads_is_the_database_and_one_it_cannot_read_is_a_hole(
    tmp_path,
):
    """The read statement and the driver's ``commit`` are leaves, and so is a statement handed in by the caller,
    read at the callers; one whose text is nowhere here is a hole."""
    record = _database(
        tmp_path,
        (
            "class Proxy:\n"
            "    def update(self, status):\n"
            "        varMap = {':status': status}\n"
            "        self.cur.execute('UPDATE ATLAS_PANDA.jobsActive4 SET jobStatus=:status', varMap)\n"
            "        self.conn.commit()\n\n"
            "    def run_any(self, sql, varMap):\n"
            "        self.cur.execute(sql, varMap)\n\n"
            "    def run_elsewhere(self, varMap):\n"
            "        self.cur.execute(SQL_FROM_ELSEWHERE, varMap)\n"
        ),
    )

    update = record.rows["pkg/db.py::Proxy.update"]
    run_any = record.rows["pkg/db.py::Proxy.run_any"]
    run_elsewhere = record.rows["pkg/db.py::Proxy.run_elsewhere"]

    assert update.database == [4, 5]
    assert update.settled == {"driver_talks_only_to_db": 1, "runner_runs_only_sql": 1}
    assert update.holes == []
    assert (run_any.database, run_any.settled, run_any.holes) == (
        [8],
        {"runner_runs_callers_sql": 1},
        [],
    )
    # The driver call is no leaf here: what it runs is read nowhere.
    assert run_elsewhere.database == []
    assert [hole.line for hole in run_elsewhere.holes] == [11]
    assert run_elsewhere.holes[0].why.startswith("the statement is not read")


def test_a_call_handing_a_runner_its_statement_is_the_database_unless_the_runner_runs_its_own(
    tmp_path,
):
    """``getSiteInfo`` hands ``getClobObj`` its ``SELECT``: the statement is read at the call, and the call is the
    database.  ``queryTasksToBePending_JEDI`` runs the ``SELECT`` it is handed and updates the tasks it picks: the
    call is followed, and the update stays in the closure."""
    record = _database(
        tmp_path,
        (
            "class Proxy:\n"
            "    def getClobObj(self, sql, varMap):\n"
            "        self.cur.execute(sql, varMap)\n\n"
            "    def queryTasksToBePending_JEDI(self, sql_query, params_map):\n"
            "        self.cur.execute(sql_query, params_map)\n"
            "        self.cur.execute('UPDATE ATLAS_PANDA.JEDI_Tasks SET status=:s', {':s': 'pending'})\n\n"
            "    def getSiteInfo(self):\n"
            "        self.getClobObj('SELECT siteid FROM ATLAS_PANDA.schedconfig', {})\n\n"
            "    def pend(self):\n"
            "        self.queryTasksToBePending_JEDI('SELECT jediTaskID FROM ATLAS_PANDA.JEDI_Tasks', {})\n\n\n"
            "class Buffer:\n"
            "    def __init__(self):\n"
            "        self.proxy = Proxy()\n\n"
            "    def queryTasksToBePending_JEDI(self, sql_query, params_map):\n"
            "        return self.proxy.queryTasksToBePending_JEDI(sql_query, params_map)\n"
        ),
    )

    site_info = record.rows["pkg/db.py::Proxy.getSiteInfo"]
    pend = record.rows["pkg/db.py::Proxy.pend"]
    handing = record.rows["pkg/db.py::Buffer.queryTasksToBePending_JEDI"]

    assert (
        site_info.database,
        site_info.settled,
        _to(record, "pkg/db.py::Proxy.getSiteInfo"),
    ) == (
        [10],
        {"runner_runs_only_sql": 1},
        [],
    )
    assert (pend.database, _to(record, "pkg/db.py::Proxy.pend")) == (
        [],
        ["pkg/db.py::Proxy.queryTasksToBePending_JEDI"],
    )
    assert record.rows["pkg/db.py::Proxy.getClobObj"].settled == {
        "runner_runs_callers_sql": 1
    }
    # Handing its caller's statement on to a runner that updates the tasks it picks is followed too.
    assert (
        handing.database,
        _to(record, "pkg/db.py::Buffer.queryTasksToBePending_JEDI"),
    ) == (
        [],
        ["pkg/db.py::Proxy.queryTasksToBePending_JEDI"],
    )


def _fragment(record: reach.ReachGraph, junctions=()):
    from bamboo.codemap.models import MapFragment

    return MapFragment(
        map_id="m",
        derived_from="v",
        reach_graph=list(record.rows.values()),
        junctions=list(junctions),
    )


def test_the_record_s_edges_and_writes_resolve_and_a_dropped_one_fails_the_gate(
    tmp_path,
):
    from bamboo.codemap import gates
    from bamboo.codemap.models import ReachCall

    record = _record(tmp_path, **_WORKERS)
    junction = _junction("pkg/cmd.py::write_status", 12, [("done", [])])
    assert gates.reach_records_resolve(_fragment(record, [junction])).passed

    record.rows["pkg/cmd.py::command"].calls.append(
        ReachCall(line=9, callee="gone", to="pkg/cmd.py::gone")
    )
    # A row whose line is not among its lines hides where the call is.
    record.rows["pkg/cmd.py::command"].calls.append(
        ReachCall(line=7, lines=[5], callee="w", to="pkg/cmd.py::write_status")
    )
    stray = _junction("pkg/gone.py::vanished", 3, [("failed", [])])
    at_import = _junction("pkg/cmd.py::<module>", 1, [("default", [])])
    result = gates.reach_records_resolve(
        _fragment(record, [junction, stray, at_import])
    )

    assert not result.passed
    assert result.failures == [
        "pkg/cmd.py::command:9 goes into pkg/cmd.py::gone, no row of the record",
        "pkg/cmd.py::command:7 goes into pkg/cmd.py::write_status from lines [5],"
        " not holding it in order",
        "pkg/gone.py::vanished: a junction no function of the record holds",
    ]
    # A write in a module's own body has no row, and is said, not failed.
    assert "['pkg/cmd.py::<module>']" in result.note


def test_a_dunder_python_runs_unasked_that_reaches_a_write_fails_the_gate(tmp_path):
    """``with Lock():`` runs ``__enter__`` with no call naming it: a write behind it is one no closure takes."""
    from bamboo.codemap import gates

    record = _record(
        tmp_path,
        mod=(
            "def write():\n"
            "    pass\n\n"
            "class Lock:\n"
            "    def __enter__(self):\n"
            "        write()\n\n"
            "    def __exit__(self, *args):\n"
            "        len([])\n"
        ),
    )
    result = gates.implicit_dunders_reach_no_write(
        _fragment(record, [_junction("pkg/mod.py::write", 2, [("done", [])])])
    )

    assert result.checked == 2
    assert result.failures == [
        "pkg/mod.py::Lock.__enter__ reaches ['pkg/mod.py::write']"
    ]


#: A class defining the methods the tests below call, so that a call on an unknown receiver is a hole, not settled as
#: a method no class of the corpus defines.
_TAKER = (
    "class Taker:\n"
    "    def update(self, x):\n"
    "        pass\n\n"
    "    def append(self, x):\n"
    "        pass\n\n"
    "    def move(self):\n"
    "        pass\n\n"
)


def _callees(record: reach.ReachGraph, owner: str) -> list[str]:
    return [hole.callee for hole in record.rows[owner].holes]


def test_a_container_s_copy_is_the_container_and_an_augmented_assignment_leaves_a_name_what_it_was(
    tmp_path,
):
    """``msg_dict = update_msg_dict.copy()`` is a dict; ``found = []`` ... ``found += other()`` is a list.

    Read before as one of what the dict holds, and as what ``other()`` is: push_job_status_message's ``.update`` and
    retryTask_JEDI's ``retried_tasks.append`` were holes.  A value the dict holds is still one of what it holds.
    """
    record = _record(
        tmp_path,
        mod=(
            _TAKER + "def other():\n"
            "    return 1\n\n"
            "def push(orig: dict):\n"
            "    msg = orig.copy()\n"
            "    msg.update({'a': 1})\n"
            "    found = []\n"
            "    found += other()\n"
            "    found.append(1)\n"
            "    held = orig.get('k')\n"
            "    held.update({})\n"
        ),
    )

    assert _callees(record, "pkg/mod.py::push") == ["held.update"]


def test_a_name_unpacked_from_a_call_is_the_slot_its_tuple_is_declared_with(tmp_path):
    """``ok, data = pair()`` on ``-> tuple[bool, dict]``: ``data`` is a dict, where the record found no binding of it.

    A slot declared ``Any`` says nothing; a row the database hands back is what code outside the corpus made.
    """
    record = _record(
        tmp_path,
        mod=(
            _TAKER + "from typing import Any\n\n"
            "def pair() -> tuple[bool, dict]:\n"
            "    return True, {}\n\n"
            "def anything() -> tuple[bool, Any]:\n"
            "    return True, None\n\n"
            "def use():\n"
            "    ok, data = pair()\n"
            "    data.update({})\n"
            "    ok, held = anything()\n"
            "    held.update({})\n"
            "    first, second = (1, {})\n"
            "    second.update({})\n"
        ),
    )

    assert _callees(record, "pkg/mod.py::use") == ["held.update"]


def test_a_local_container_s_values_are_what_the_function_puts_into_it(tmp_path):
    """``files[name] = {'lfns': []}`` then ``files[name]['lfns'].append(name)``: a list, put there by the function.

    Not where the container goes where other code can fill it -- handed to a call outside the corpus --, nor where
    a put is not read (``update`` with a mapping handed in from nowhere the corpus calls), nor where nothing the
    function puts is at that place.
    """
    record = _record(
        tmp_path,
        mod=(
            "import json\n\n" + _TAKER + "def fill(names: list, more: dict):\n"
            "    files = {}\n"
            "    for name in names:\n"
            "        files[name] = {'lfns': [], 'job': None}\n"
            "        files[name]['lfns'].append(name)\n"
            "    by_weight = {}\n"
            "    by_weight[1] = []\n"
            "    by_weight[1].append(2)\n"
            "    shared = {}\n"
            "    shared['x'] = []\n"
            "    json.dump(shared, None)\n"
            "    shared['x'].append(1)\n"
            "    merged = {}\n"
            "    merged['x'] = []\n"
            "    merged.update(more)\n"
            "    merged['x'].append(1)\n"
            "    files[name]['job'].move()\n"
        ),
    )

    assert sorted(_callees(record, "pkg/mod.py::fill")) == [
        "merged['x'].append",
        "shared['x'].append",
    ]


def test_a_container_s_values_are_read_from_every_name_holding_it(tmp_path):
    """What ``d[k]...`` is, read from every name holding d's container: across calls, through parts, aliases, loops.

    ``record`` is handed ``extracted`` by ``insert``; ``insert`` by ``bulk``'s ``insert(*args, **kwargs)`` after
    ``kwargs.update(new_kwargs)`` -- the ``**`` holds the key on every way, so the ``*`` does not fill it -- and by
    ``store`` with the default: ``extracted['state']['vars']`` is the list ``setdefault`` puts there.  Where the
    update is on some ways only (``bulk2``), the ``*`` may fill it and the call stays a hole.  A value put through a
    part (``setdefault(...).append``, ``get(...).append``), an alias (``e = d or more``), a loop over the values or
    items or a builtin's copy of them, or by a callee handed the container, is one of what it holds there: the
    undeclared one leaves ``d['a'][-1].move`` a hole.  A callee that only reads it leaves it closed.
    """
    shapes = {
        "chain_setdefault": "    d.setdefault('a', []).append(undeclared())\n",
        "chain_get": "    d.get('a').append(undeclared())\n",
        "loop_values": "    for v in d.values():\n        v.append(undeclared())\n",
        "loop_items": "    for k, v in d.items():\n        v.append(undeclared())\n",
        "alias": "    e = d or {}\n    e['a'].append(undeclared())\n",
        "copy_values": "    for v in list(d.values()):\n        v.append(undeclared())\n",
        "callee_puts": "    puts(d)\n",
        "callee_reads": "    looks(d)\n",
    }
    record = _record(
        tmp_path,
        mod=(
            _TAKER + "def record(job, extracted: dict | None = None):\n"
            "    if extracted is not None:\n"
            "        extracted.setdefault('state', {'sql': 's', 'vars': []})\n"
            "        extracted['state']['vars'].append(job)\n\n"
            "def insert(job, user, flag=True, extracted: dict | None = None):\n"
            "    record(job, extracted=extracted)\n\n"
            "def bulk(arg_list: list):\n"
            "    for args, kwargs in arg_list:\n"
            "        tmp = {}\n"
            "        new_kwargs = {'extracted': tmp}\n"
            "        kwargs.update(new_kwargs)\n"
            "        insert(*args, **kwargs)\n\n"
            "def store(jobs: list):\n"
            "    params = []\n"
            "    for job in jobs:\n"
            "        params.append([[job, 'u'], {'flag': False}])\n"
            "    bulk(params)\n"
            "    insert(jobs[0], 'u')\n\n"
            "def record2(job, extracted: dict | None = None):\n"
            "    extracted.setdefault('state', {'vars': []})\n"
            "    extracted['state']['vars'].append(job)\n\n"
            "def insert2(job, extracted: dict | None = None):\n"
            "    record2(job, extracted=extracted)\n\n"
            "def bulk2(arg_list: list, c):\n"
            "    for args, kwargs in arg_list:\n"
            "        if c:\n"
            "            kwargs.update({'extracted': {}})\n"
            "        insert2(*args, **kwargs)\n\n"
            "def store2(jobs: list):\n"
            "    bulk2([[[job], {}] for job in jobs], True)\n\n"
            "def puts(d):\n"
            "    d['a'].append(undeclared())\n\n"
            "def looks(d):\n"
            "    print(len(d))\n\n"
            + "".join(
                f"def {name}():\n    d = {{'a': [1]}}\n{body}    d['a'][-1].move()\n\n"
                for name, body in shapes.items()
            )
        ),
    )

    def holes(owner: str) -> list[str]:
        return sorted(_callees(record, f"pkg/mod.py::{owner}"))

    assert "extracted['state']['vars'].append" not in holes("record")
    assert "extracted['state']['vars'].append" in holes("record2")
    for name in shapes:
        assert ("d['a'][-1].move" in holes(name)) is (name != "callee_reads"), name


def test_a_value_unpacked_into_a_container_s_place_is_the_slot_the_call_is_declared_to_return(
    tmp_path,
):
    """``(fetched[site], held[site]) = buffer.outputs()``, declared ``tuple[list, Any]``: ``fetched[site]`` is a list
    (doGenerate's ``fetched_out_sub_chunks[site_name]``, put by getOutputFiles_JEDI's first slot), ``held[site]``
    what the second slot is declared -- not one of what the tuple holds."""
    record = _record(
        tmp_path,
        mod=(
            "from typing import Any\n\n" + _TAKER + "class Buffer:\n"
            "    def outputs(self) -> tuple[list, Any]:\n"
            "        return [], None\n\n"
            "def generate(buffer: Buffer, site):\n"
            "    fetched = {}\n"
            "    held = {}\n"
            "    (fetched[site], held[site]) = buffer.outputs()\n"
            "    fetched[site].append(1)\n"
            "    held[site].move()\n"
        ),
    )
    holes = record.rows["pkg/mod.py::generate"].holes
    assert [hole.callee for hole in holes] == ["held[site].move"]
    assert holes[0].why.endswith("declared Any")


def test_a_loop_over_a_name_bound_to_a_call_is_over_what_the_call_is_declared_to_return(
    tmp_path,
):
    """``taskList = buffer.tasks()`` then ``for task in taskList``: each a ``Task`` (doActionForReassign's)."""
    record = _record(
        tmp_path,
        mod=(
            _TAKER + "class Task:\n"
            "    def move(self):\n"
            "        pass\n\n"
            "class Buffer:\n"
            "    def tasks(self) -> list[Task]:\n"
            "        return []\n\n"
            "def reassign(buffer: Buffer):\n"
            "    found = buffer.tasks()\n"
            "    for task in found:\n"
            "        task.move()\n"
        ),
    )

    assert record.rows["pkg/mod.py::reassign"].holes == []
    assert "pkg/mod.py::Task.move" in _to(record, "pkg/mod.py::reassign")


def test_a_hole_names_what_its_receiver_is_one_of_with_its_article(tmp_path):
    """``for idx, item in enumerate(items)``: one of what an iterator holds -- "an iterator", not "a iterator"."""
    record = _record(
        tmp_path,
        mod=(
            _TAKER + "def run(items):\n"
            "    for idx, item in enumerate(items):\n"
            "        item.move()\n"
        ),
    )

    assert [hole.why for hole in record.rows["pkg/mod.py::run"].holes] == [
        "the receiver's class is not declared: an element of an iterator"
    ]


def test_the_keys_a_mapping_may_hold_are_read_from_every_put_into_every_name_holding_it(
    tmp_path,
):
    """``update``'s ``param`` holds the keys every caller puts into what it hands: ``first``, through a wrapper whose
    ``run`` hands the function it was built with its own arguments (the read of ``update`` that built it is no read
    as a value), puts at a literal list's first slots, at a literal loop's, at a literal; ``second`` hands a dict
    literal.  ``update``'s own put at a key taken from ``param`` adds none.  Both ways, as every holder is read:
    ``first``'s mapping is ``update``'s ``param``, which ``second`` fills too -- more than ``first`` puts, never
    less.  Not read: a key no literal is (an f-string, a parameter), a mapping handed outside the corpus, one a
    call returns."""
    timed = (
        "class Timed:\n"
        "    def __init__(self, method, timeout):\n"
        "        self.method = method\n\n"
        "    def run(self, *var, **kwargs):\n"
        "        self.method(*var, **kwargs)\n"
    )
    mod = (
        "import json\n"
        "from pkg.timed import Timed\n\n"
        "def update(job_id, param):\n"
        "    for key in list(param):\n"
        "        param[key] = str(param[key])\n\n"
        "def first(job_id, metrics):\n"
        "    param = {}\n"
        "    fields = [('jobMetrics', metrics, int), ('nEvents', 1, int)]\n"
        "    for key, value, cast in fields:\n"
        "        param[key] = cast(value)\n"
        "    for key, value in [('startTime', 1), ('endTime', 2)]:\n"
        "        param[key] = value\n"
        "    param['diag'] = None\n"
        "    timed = Timed(update, 10)\n"
        "    timed.run(job_id, param)\n\n"
        "def second(job_id):\n"
        "    update(job_id, {'cloud': 'x'})\n\n"
        "def by_fstring(param, n):\n"
        "    param[f'x{n}'] = 1\n\n"
        "def by_parameter(param, name):\n"
        "    param[name] = 1\n\n"
        "def outside(param):\n"
        "    json.dumps(param)\n\n"
        "def made():\n"
        "    return json.loads('{}')\n\n"
        "def returned():\n"
        "    param = made()\n"
        "    param['a'] = 1\n"
    )
    calls = calls_mod.Calls(calls_mod.Types(_tree(tmp_path, timed=timed, mod=mod), {}))
    receivers = reach._Receivers(calls, None)
    tree = calls.types.tree("pkg/mod.py")

    def keys(owner: str):
        func = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == owner
        )
        return receivers.keys("pkg/mod.py", func, "param")

    assert keys("update") == {
        "jobMetrics",
        "nEvents",
        "startTime",
        "endTime",
        "diag",
        "cloud",
    }
    assert keys("first") == keys("update")
    for owner in ("by_fstring", "by_parameter", "outside", "returned"):
        assert keys(owner) is None, owner
