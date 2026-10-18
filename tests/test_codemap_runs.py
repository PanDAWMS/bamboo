"""Where runs start, and the history rows written in the same pass as an arm -- ``codemap.panda.runs``."""

from __future__ import annotations

import ast
from pathlib import Path

from bamboo.codemap import reach, trace
from bamboo.codemap.gates import history_pairs_resolve
from bamboo.codemap.gitsource import blob_sha
from bamboo.codemap.models import (
    HISTORY_AFTER,
    HISTORY_BEFORE,
    HISTORY_ROW_SAME,
    RUN_KEY_TIME,
    Anchor,
    Branch,
    FunctionPrintsNode,
    JunctionNode,
    MapFragment,
    RunKey,
    SourceModule,
)
from bamboo.codemap.panda import pathcond, runs, sql
from bamboo.codemap.panda.plugin import _DatabaseLeaves
from bamboo.codemap.panda.recognizers import sqlwrite

PACKAGE = "pkg"

SPEC = """\
class JobSpec:
    pass


class JediDatasetSpec:
    pass
"""

PROXY = """\
from pandacommon.pandalogger.LogWrapper import LogWrapper

from pkg.spec import JobSpec

_logger = None


class DBProxy:
    def note_change(self, panda_id, status):
        sql = "INSERT INTO ATLAS_PANDA.jobs_StatusLog (PandaID,jobStatus) VALUES (:PandaID,:jobStatus)"
        self.cur.execute(sql, {":PandaID": panda_id, ":jobStatus": status})

    def archiveJob(self, job: JobSpec, other: JobSpec, c):
        tmp_log = LogWrapper(_logger, f"PandaID={job.PandaID}")  # R0
        tmp_log.debug("start")  # P0
        if c:
            job.jobStatus = "failed"  # A1
        else:
            job.jobStatus = "finished"  # A2
        self.note_change(job.PandaID, job.jobStatus)  # H1
        self.note_change(other.PandaID, other.jobStatus)  # H2

    def update(self, job: JobSpec, c):
        if c:
            job.jobStatus = "running"  # A3
        self.note_change(job.PandaID, job.jobStatus)  # H3

    def loop(self, jobs):
        for job in jobs:
            self.note_change(job.PandaID, "sent")  # H4
            job.jobStatus = "sent"  # A4

    def set_status(self, job):
        job.jobStatus = "holding"  # A5

    def run(self, job: JobSpec):
        tmp_log = LogWrapper(_logger, "<PandaID={0}>".format(job.PandaID))  # R1
        tmp_log.debug("run")
        self.set_status(job)  # C1
        self.archive_history(job)  # C2

    def archive_history(self, job):
        self.note_change(job.PandaID, job.jobStatus)  # H5

    def handed(self, job, pool):
        job.jobStatus = "closed"  # A6
        pool.submit(self.archive_history, job)  # C3

    def dataset(self, ds):
        ds.status = "done"  # D1
        self.note_change(1, "x")
"""

WATCHER = """\
from pandacommon.pandalogger.LogWrapper import LogWrapper

from pkg.proxy import DBProxy
from pkg.spec import JobSpec

_logger = None


class Watcher:
    def __init__(self, pandaID):
        self.logger = LogWrapper(_logger, str(pandaID))  # W0

    def run(self, job: JobSpec, proxy: DBProxy):
        self.logger.debug("watching")  # W1
        job.jobStatus = "failed"  # A7
        proxy.note_change(job.PandaID, job.jobStatus)  # W2


class Rebinder:
    def __init__(self):
        self.logger = LogWrapper(_logger, "fixed")

    def reset(self):
        self.logger = LogWrapper(_logger, None)

    def run(self, job: JobSpec, proxy: DBProxy):
        self.logger.debug("rebinding")
        job.jobStatus = "failed"  # A8
        proxy.note_change(job.PandaID, job.jobStatus)
"""

MARKS = """\
from pandacommon.pandalogger.LogWrapper import LogWrapper
from pandacommon.pandalogger.logger_utils import make_logger

from pandajedi.jedicore.MsgWrapper import MsgWrapper

_logger = None


def marks(task, n):
    a = MsgWrapper(_logger)
    b = make_logger(_logger)
    c = LogWrapper(_logger, None)
    d = LogWrapper(_logger)
    e = MsgWrapper(_logger, token=f"<jediTaskID={task.jediTaskID} n={n}>")
    a.debug("1")
    b.debug("2")
    c.debug("3")
    d.debug("4")
    e.debug("5")
"""


def _line(text: str, marker: str) -> int:
    (found,) = [
        number
        for number, held in enumerate(text.splitlines(), 1)
        if held.rstrip().endswith(f"# {marker}")
    ]
    return found


def _build(
    tmp_path: Path, arms: list[tuple[str, str, str, str]], **files: str
) -> MapFragment:
    """Read *files* as the build does, with one junction per (file, owner, marker, subject) of *arms*."""
    root = tmp_path / PACKAGE
    root.mkdir(parents=True, exist_ok=True)
    (root / "__init__.py").write_text("")
    modules = []
    for name, text in files.items():
        (root / f"{name}.py").write_text(text)
        tree = ast.parse(text)
        pathcond.attach_parents(tree)
        modules.append(
            SourceModule(
                package=PACKAGE,
                rel_path=f"{PACKAGE}/{name}.py",
                tree=tree,
                source=text,
                blob_sha=blob_sha(text),
            )
        )
    roots = {PACKAGE: root}
    calls = sqlwrite.corpus_calls(modules, roots)
    with sql.across_calls(calls):
        nodes = [
            FunctionPrintsNode(
                name=f"m:{found.owner}",
                map_id="m",
                derived_from="v",
                owner=found.owner,
                cls=found.cls,
                function=found.func.name,
                count=len(found.statements),
                statements=found.statements,
                anchor=Anchor(
                    package=PACKAGE,
                    file=found.file,
                    line_start=found.func.lineno,
                    line_end=found.func.end_lineno,
                ),
            )
            for found in trace.function_prints(
                roots, [module.rel_path for module in modules]
            )
        ]
        junctions = []
        for name, owner, marker, subject in arms:
            line = _line(files[name], marker)
            junctions.append(
                JunctionNode(
                    name=f"m:{owner}:{marker}",
                    map_id="m",
                    derived_from="v",
                    subject=subject,
                    owner=f"{PACKAGE}/{name}.py::{owner}",
                    anchor=Anchor(
                        package=PACKAGE, file=f"{PACKAGE}/{name}.py", line_start=line
                    ),
                    branches=[Branch(outcome=marker, line=line)],
                )
            )
        fragment = MapFragment(
            map_id="m",
            derived_from="v",
            junctions=junctions,
            function_prints=nodes,
            reach_graph=reach.graph(calls, "m", "v", _DatabaseLeaves()),
        )
        runs.attach(fragment, modules, calls)
    return fragment


def _branch(fragment: MapFragment, marker: str) -> Branch:
    (branch,) = [
        b for j in fragment.junctions for b in j.branches if b.outcome == marker
    ]
    return branch


def _proxy(tmp_path: Path, *markers: tuple[str, str]) -> MapFragment:
    arms = [("proxy", owner, marker, "JobSpec.jobStatus") for owner, marker in markers]
    return _build(tmp_path, arms, spec=SPEC, proxy=PROXY)


def test_a_writer_is_found_by_what_its_sql_inserts_into_not_by_its_name(
    tmp_path: Path,
) -> None:
    fragment = _proxy(tmp_path)
    rows = {row.owner: row for row in fragment.reach_graph}
    archive = rows["pkg/proxy.py::DBProxy.archiveJob"]
    assert [(held.line, held.subject) for held in archive.history] == [
        (_line(PROXY, "H1"), "JobSpec.jobStatus"),
        (_line(PROXY, "H2"), "JobSpec.jobStatus"),
    ]
    # The writer itself and a function calling nothing that writes have none.
    assert rows["pkg/proxy.py::DBProxy.note_change"].history == []
    assert rows["pkg/proxy.py::DBProxy.set_status"].history == []


def test_a_history_call_records_the_arms_that_reach_it_and_a_way_past_them(
    tmp_path: Path,
) -> None:
    fragment = _proxy(
        tmp_path,
        ("DBProxy.archiveJob", "A1"),
        ("DBProxy.archiveJob", "A2"),
        ("DBProxy.update", "A3"),
    )
    rows = {row.owner: row for row in fragment.reach_graph}
    (first, _second) = rows["pkg/proxy.py::DBProxy.archiveJob"].history
    assert (first.values, first.passed) == (["A1", "A2"], False)
    (only,) = rows["pkg/proxy.py::DBProxy.update"].history
    assert (only.values, only.passed) == (["A3"], True)
    # update prints nothing, so its runs have no node to be on.
    (pair,) = _branch(fragment, "A3").history or []
    assert (pair.runs, pair.run_hole) == ([], runs.RUN_NO_PRINTS)


def test_an_arm_pairs_with_the_writer_called_after_it_on_the_same_row(
    tmp_path: Path,
) -> None:
    fragment = _proxy(tmp_path, ("DBProxy.archiveJob", "A1"))
    branch = _branch(fragment, "A1")
    assert branch.history is not None, branch.history_hole
    same, another = branch.history
    assert (same.level, same.point, same.call, same.order) == (
        0,
        _line(PROXY, "A1"),
        _line(PROXY, "H1"),
        HISTORY_AFTER,
    )
    assert same.writes == [f"pkg/proxy.py::DBProxy.archiveJob:{_line(PROXY, 'H1')}"]
    assert same.history == ["JobSpec.jobStatus"]
    assert (same.row, same.row_why) == (HISTORY_ROW_SAME, "")
    assert same.runs == [_line(PROXY, "R0")]
    assert (another.call, another.row) == (
        _line(PROXY, "H2"),
        "another: other.PandaID",
    )


def test_the_next_turn_of_a_loop_is_another_pass(tmp_path: Path) -> None:
    """The writer before the arm in the loop is before it, never after: the next turn is another job's."""
    fragment = _proxy(tmp_path, ("DBProxy.loop", "A4"))
    (pair,) = _branch(fragment, "A4").history or []
    assert (pair.call, pair.order) == (_line(PROXY, "H4"), HISTORY_BEFORE)


def test_an_arm_in_a_callee_pairs_at_the_caller_and_its_row_is_not_read(
    tmp_path: Path,
) -> None:
    fragment = _proxy(tmp_path, ("DBProxy.set_status", "A5"))
    (pair,) = _branch(fragment, "A5").history or []
    assert (pair.level, pair.caller, pair.point, pair.call) == (
        1,
        "pkg/proxy.py::DBProxy.run",
        _line(PROXY, "C1"),
        _line(PROXY, "C2"),
    )
    assert pair.writes == [
        f"pkg/proxy.py::DBProxy.archive_history:{_line(PROXY, 'H5')}"
    ]
    assert (pair.row, pair.row_why) == ("", runs.ROW_ARM_BELOW)
    assert pair.runs == [_line(PROXY, "R1")]


def test_a_function_handed_on_is_not_the_same_pass(tmp_path: Path) -> None:
    fragment = _proxy(tmp_path, ("DBProxy.handed", "A6"))
    branch = _branch(fragment, "A6")
    assert (branch.history, branch.history_hole) == ([], "")


def test_a_subject_no_history_table_records_is_not_read(tmp_path: Path) -> None:
    fragment = _build(
        tmp_path,
        [("proxy", "DBProxy.dataset", "D1", "JediDatasetSpec.status")],
        spec=SPEC,
        proxy=PROXY,
    )
    branch = _branch(fragment, "D1")
    assert branch.history is None
    assert branch.history_hole == runs.HISTORY_NOT_RECORDED.format(
        cls="JediDatasetSpec"
    )


def test_a_run_says_its_mark_and_the_column_a_declared_hole_holds(
    tmp_path: Path,
) -> None:
    fragment = _proxy(tmp_path)
    prints = {node.owner: node for node in fragment.function_prints}
    (archive,) = prints["pkg/proxy.py::DBProxy.archiveJob"].runs
    assert (archive.logger, archive.builder, archive.mark, archive.object) == (
        "tmp_log",
        "LogWrapper",
        "PandaID={}",
        False,
    )
    assert archive.key == [RunKey(expression="job.PandaID", column="JobSpec.PandaID")]
    (run,) = prints["pkg/proxy.py::DBProxy.run"].runs
    assert run.mark == "<PandaID={}>"
    assert [key.column for key in run.key] == ["JobSpec.PandaID"]
    (statement,) = prints["pkg/proxy.py::DBProxy.archiveJob"].statements
    assert statement.runs == [_line(PROXY, "R0")]


def test_a_mark_given_none_is_a_time_stamp_or_nothing_as_its_factory_writes(
    tmp_path: Path,
) -> None:
    fragment = _build(tmp_path, [], marks=MARKS)
    (node,) = fragment.function_prints
    by_logger = {run.logger: run for run in node.runs}
    time = [RunKey(expression="", column=RUN_KEY_TIME)]
    assert by_logger["a"].key == time
    assert by_logger["b"].key == []
    assert by_logger["c"].key == time
    assert by_logger["d"].key == []
    assert by_logger["e"].mark == "<jediTaskID={} n={}>"
    assert [(key.column, key.why) for key in by_logger["e"].key] == [
        ("", runs.KEY_UNDECLARED.format(name="task")),
        ("", runs.KEY_PARAMETER),
    ]
    assert [statement.runs for statement in node.statements] == [
        [by_logger[name].line] for name in "abcde"
    ]


def test_an_objects_run_is_its_init_logger_unless_bound_again(tmp_path: Path) -> None:
    fragment = _build(
        tmp_path,
        [
            ("watcher", "Watcher.run", "A7", "JobSpec.jobStatus"),
            ("watcher", "Rebinder.run", "A8", "JobSpec.jobStatus"),
        ],
        spec=SPEC,
        proxy=PROXY,
        watcher=WATCHER,
    )
    prints = {node.owner: node for node in fragment.function_prints}
    (run,) = prints["pkg/watcher.py::Watcher.run"].runs
    assert (run.line, run.at, run.logger, run.object) == (
        _line(WATCHER, "W0"),
        "pkg/watcher.py::Watcher.__init__",
        "self.logger",
        True,
    )
    assert run.key == [RunKey(expression="str(pandaID)", why=runs.KEY_PARAMETER)]
    (statement,) = prints["pkg/watcher.py::Watcher.run"].statements
    assert statement.runs == [_line(WATCHER, "W0")]
    (pair,) = _branch(fragment, "A7").history or []
    assert (pair.call, pair.runs, pair.row) == (
        _line(WATCHER, "W2"),
        [_line(WATCHER, "W0")],
        HISTORY_ROW_SAME,
    )
    assert prints["pkg/watcher.py::Rebinder.run"].runs == []
    (rebound,) = _branch(fragment, "A8").history or []
    assert (rebound.runs, rebound.run_hole) == (
        [],
        runs.RUN_OBJECT_REBOUND.format(
            logger="self.logger", where="pkg/watcher.py::Rebinder.reset"
        ),
    )


def test_the_gate_passes_on_the_build_and_fails_on_a_write_no_reader_finds(
    tmp_path: Path,
) -> None:
    fragment = _proxy(
        tmp_path, ("DBProxy.archiveJob", "A1"), ("DBProxy.set_status", "A5")
    )
    assert history_pairs_resolve(fragment).passed
    for row in fragment.reach_graph:
        row.history = []
    result = history_pairs_resolve(fragment)
    assert not result.passed
    assert any("is no history call" in failure for failure in result.failures)
