"""Where runs start, and the history rows written in the same pass as an arm -- ``codemap.panda.runs``."""

from __future__ import annotations

import ast
from pathlib import Path

from bamboo.codemap import calls as calls_mod
from bamboo.codemap import reach, trace
from bamboo.codemap.gates import history_pairs_resolve
from bamboo.codemap.gitsource import blob_sha
from bamboo.codemap.models import (
    HISTORY_AFTER,
    HISTORY_BEFORE,
    HISTORY_ROW_CAN,
    HISTORY_ROW_SAME,
    RUN_KEY_TIME,
    Anchor,
    Branch,
    EntityNode,
    FunctionPrintsNode,
    HistoryPair,
    HistoryRekey,
    JunctionNode,
    KeyStop,
    MapFragment,
    RunKey,
    RunRef,
    SourceModule,
)
from bamboo.codemap.panda import objects as panda_objects
from bamboo.codemap.panda import pathcond, runs, sql
from bamboo.codemap.panda.plugin import _DatabaseLeaves
from bamboo.codemap.panda.recognizers import sqlwrite

PACKAGE = "pkg"

SPEC = """\
class JobSpec:
    _attributes = ("PandaID", "jobStatus")

    def set_failed(self):
        self.jobStatus = "failed"


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

    def archive_all(self, jobs):
        done = []
        for job in jobs:  # B1
            done.append(job)  # B2
        for one in done:  # B3
            self.note_change(one.PandaID, one.jobStatus)  # H6

    def finish(self, job: JobSpec):
        job.jobStatus = "finished"  # A9
        self.archive_all([job])  # C4

    def kill(self, panda_id):
        var_map = {}
        var_map[":PandaID"] = panda_id
        var_map[":jobStatus"] = "cancelled"  # A10
        self.cur.execute("UPDATE ATLAS_PANDA.jobsActive4 SET jobStatus=:jobStatus WHERE PandaID=:PandaID", var_map)
        self.note_change(panda_id, "cancelled")  # H7

    def peek(self, panda_id):
        job = JobSpec()
        job.jobStatus = "closed"  # A11
        return 1

    def close(self, job: JobSpec):
        self.peek(job.PandaID)  # C5
        self.note_change(job.PandaID, "closed")  # H8

    def hold(self, job: JobSpec):
        job.set_failed()  # A12
        self.note_change(job.PandaID, "failed")  # H9

    def lock(self, panda_id):
        var_map = {}
        var_map[":PandaID"] = panda_id
        var_map[":jobStatus"] = "locked"  # A13
        self.cur.execute("UPDATE ATLAS_PANDA.jobsActive4 SET jobStatus=:jobStatus WHERE PandaID=:PandaID", var_map)

    def refine(self, panda_id):
        self.lock(panda_id)  # C6
        holder = Holder()
        holder.extract(panda_id)  # X1
        self.note_change(holder.job.PandaID, "locked")  # H10


class Holder:
    def extract(self, panda_id):
        job = JobSpec()
        job.PandaID = panda_id
        self.job = job
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

FLAGS = """\
from pkg.proxy import DBProxy
from pkg.spec import JobSpec


class Flags(DBProxy):
    def set_diag(self, job: JobSpec, append=False):
        if append:
            job.jobStatus = "appended"  # F1
        else:
            job.jobStatus = "set"  # F2

    def plain(self, job: JobSpec):
        self.set_diag(job)
        self.note_change(job.PandaID, job.jobStatus)  # G1

    def appending(self, job: JobSpec):
        self.set_diag(job, True)
        self.note_change(job.PandaID, job.jobStatus)  # G2

    def forward(self, job: JobSpec, flag):
        self.set_diag(job, append=flag)

    def forwarding(self, job: JobSpec):
        self.forward(job, False)
        self.note_change(job.PandaID, job.jobStatus)  # G3

    def via_true(self, job: JobSpec):
        self.set_diag(job, True)

    def via_plain(self, job: JobSpec):
        self.set_diag(job)

    def top(self, job: JobSpec):
        self.via_true(job)
        self.via_plain(job)
        self.note_change(job.PandaID, job.jobStatus)  # G4

    def peek_one(self, job: JobSpec, for_anal=False):
        if for_anal:
            return None
        job.jobStatus = "unknown"  # F3
        return job

    def analysed(self, job: JobSpec):
        self.peek_one(job, for_anal=True)
        self.note_change(job.PandaID, job.jobStatus)  # G5
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
    e = MsgWrapper(_logger, token=f"<jediTaskID={task.jediTaskID} n={n}>")  # M0
    a.debug("1")
    b.debug("2")
    c.debug("3")
    d.debug("4")
    e.debug("5")
"""


#: The tables the spec classes hold, as the build learns them (EntityNode.spec_class).
ENTITIES = [
    EntityNode(
        name="jobsactive4",
        map_id="m",
        derived_from="v",
        spec_class="JobSpec",
        tables=["jobsActive4"],
    ),
    EntityNode(
        name="jedi_tasks",
        map_id="m",
        derived_from="v",
        spec_class="JediTaskSpec",
        tables=["JEDI_Tasks"],
    ),
]


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
        # The process's one tree for the file, as the build reads it (``calls.parsed``).
        _source, tree = calls_mod.parsed(root / f"{name}.py")
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
                roots, [module.rel_path for module in modules], calls
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
            entities=[one.model_copy() for one in ENTITIES],
            junctions=junctions,
            function_prints=nodes,
            reach_graph=reach.graph(calls, "m", "v", _DatabaseLeaves()),
        )
        panda_objects.attach(fragment, modules, calls)
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


KEYED = """\
from pkg.spec import JobSpec


class DBProxy:
    def note_change(self, panda_id, status):
        sql = "INSERT INTO ATLAS_PANDA.jobs_StatusLog (PandaID,jobStatus) VALUES (:PandaID,:jobStatus)"
        self.cur.execute(sql, {":PandaID": panda_id, ":jobStatus": status})

    def update_status(self, panda_id, status, param):
        sql = "UPDATE ATLAS_PANDA.jobsActive4 SET jobStatus=:jobStatus"
        varMap = {}
        varMap[":jobStatus"] = status
        for key in list(param):
            sql += f",{key}=:{key}"
            varMap[f":{key}"] = param[key]  # K1
        sql += " WHERE PandaID=:PandaID"
        varMap[":PandaID"] = panda_id
        self.cur.execute(sql, varMap)
        self.note_change(panda_id, status)  # K2


def update_job(proxy: DBProxy, panda_id, code):
    param = {}
    param["transExitCode"] = code
    proxy.update_status(panda_id, "running", param)
"""


def test_a_column_named_by_a_mapping_s_key_is_on_the_row_its_statement_keys(
    tmp_path: Path,
) -> None:
    """``updateJobStatus``' ``,{key}=:{key}`` writes ``transExitCode`` from the put at K1 (sql.with_keyed_columns):
    the statement's row is the one ``WHERE PandaID=:PandaID`` names, and the history call after it, on
    ``panda_id``, can be that row's -- as for a column the statement spells."""
    spec = SPEC.replace(
        '("PandaID", "jobStatus")', '("PandaID", "jobStatus", "transExitCode")'
    )
    arms = [("keyed", "DBProxy.update_status", "K1", "JobSpec.transExitCode")]
    fragment = _build(tmp_path, arms, spec=spec, keyed=KEYED)
    branch = _branch(fragment, "K1")
    assert branch.history is not None, branch.history_hole
    (pair,) = branch.history
    assert (pair.call, pair.order) == (_line(KEYED, "K2"), HISTORY_AFTER)
    assert (pair.row, pair.row_why, pair.object) == (HISTORY_ROW_CAN, "", "panda_id")


def test_the_next_turn_of_a_loop_is_another_pass(tmp_path: Path) -> None:
    """The writer before the arm in the loop is before it, never after: the next turn is another job's."""
    fragment = _proxy(tmp_path, ("DBProxy.loop", "A4"))
    (pair,) = _branch(fragment, "A4").history or []
    assert (pair.call, pair.order) == (_line(PROXY, "H4"), HISTORY_BEFORE)


def test_an_arm_in_a_callee_pairs_at_the_caller_with_its_object_followed_to_the_writer(
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
    write = f"pkg/proxy.py::DBProxy.archive_history:{_line(PROXY, 'H5')}"
    assert pair.writes == [write]
    assert (pair.row, pair.row_why, pair.object) == (HISTORY_ROW_CAN, "", "job")
    assert pair.row_via == [
        f"pkg/proxy.py::DBProxy.run:{_line(PROXY, 'C2')} -> pkg/proxy.py::DBProxy.archive_history job",
        f"{write} job.PandaID",
    ]
    assert pair.runs == [_line(PROXY, "R1")]


def test_a_row_written_for_each_element_of_a_list_can_be_the_arms(
    tmp_path: Path,
) -> None:
    """``[job]`` handed on, put in a local list and run over: the row is one of the elements', the arm's among them."""
    fragment = _proxy(tmp_path, ("DBProxy.finish", "A9"))
    (pair,) = _branch(fragment, "A9").history or []
    write = f"pkg/proxy.py::DBProxy.archive_all:{_line(PROXY, 'H6')}"
    assert (pair.level, pair.call, pair.row, pair.object) == (
        0,
        _line(PROXY, "C4"),
        HISTORY_ROW_CAN,
        "job",
    )
    archive_all = "pkg/proxy.py::DBProxy.archive_all"
    assert pair.row_via == [
        f"pkg/proxy.py::DBProxy.finish:{_line(PROXY, 'C4')} -> {archive_all} jobs[]",
        f"{archive_all}:{_line(PROXY, 'B1')} element job",
        f"{archive_all}:{_line(PROXY, 'B2')} fill done[]",
        f"{archive_all}:{_line(PROXY, 'B3')} element one",
        f"{write} one.PandaID",
    ]


def test_an_sql_arm_is_the_row_its_statement_keys(tmp_path: Path) -> None:
    fragment = _proxy(tmp_path, ("DBProxy.kill", "A10"))
    (pair,) = _branch(fragment, "A10").history or []
    assert (pair.call, pair.row, pair.object) == (
        _line(PROXY, "H7"),
        HISTORY_ROW_CAN,
        "panda_id",
    )


def test_a_method_of_the_class_run_on_the_object_is_an_arm_on_it(
    tmp_path: Path,
) -> None:
    fragment = _proxy(tmp_path, ("DBProxy.hold", "A12"))
    (pair,) = _branch(fragment, "A12").history or []
    assert (pair.call, pair.row, pair.object) == (
        _line(PROXY, "H9"),
        HISTORY_ROW_CAN,
        "job",
    )


def test_a_key_a_callee_keeps_on_its_self_comes_back_to_the_caller(
    tmp_path: Path,
) -> None:
    """TaskRefiner: extractCommon keeps jediTaskID on self.taskSpec, and the caller's history call names impl.taskSpec.jediTaskID."""
    fragment = _proxy(tmp_path, ("DBProxy.lock", "A13"))
    (pair,) = _branch(fragment, "A13").history or []
    refine = "pkg/proxy.py::DBProxy.refine"
    assert (pair.level, pair.caller, pair.point, pair.call) == (
        1,
        refine,
        _line(PROXY, "C6"),
        _line(PROXY, "H10"),
    )
    assert (pair.row, pair.object) == (HISTORY_ROW_CAN, "panda_id")
    assert pair.row_via == [
        f"{refine}:{_line(PROXY, 'X1')} <- pkg/proxy.py::Holder.extract holder.job.PandaID",
        f"{refine}:{_line(PROXY, 'H10')} holder.job.PandaID",
    ]
    assert history_pairs_resolve(fragment).passed


def test_an_object_the_callee_builds_is_not_the_callers(tmp_path: Path) -> None:
    fragment = _proxy(tmp_path, ("DBProxy.peek", "A11"))
    (pair,) = _branch(fragment, "A11").history or []
    assert (pair.level, pair.caller, pair.call) == (
        1,
        "pkg/proxy.py::DBProxy.close",
        _line(PROXY, "H8"),
    )
    assert (pair.row, pair.row_why, pair.object) == ("", runs.ROW_NOT_HANDED, "")


MADE = """\
from pkg.proxy import DBProxy
from pkg.spec import JobSpec


class Made(DBProxy):
    def generate(self):
        made = []
        job = JobSpec()
        job.jobStatus = "generated"  # G1
        made.append(job)
        return True, made

    def submit(self):
        ok, jobs = self.generate()  # G2
        self.archive_all(jobs[0:10])  # G3
"""


def test_an_object_the_callee_builds_and_returns_is_followed_from_the_caller(
    tmp_path: Path,
) -> None:
    """runImpl: doGenerate's jobs come back in a list in a tuple, unpacked, and a slice of the list is stored."""
    fragment = _build(
        tmp_path,
        [("made", "Made.generate", "G1", "JobSpec.jobStatus")],
        spec=SPEC,
        proxy=PROXY,
        made=MADE,
    )
    (pair,) = _branch(fragment, "G1").history or []
    submit, archive_all = (
        "pkg/made.py::Made.submit",
        "pkg/proxy.py::DBProxy.archive_all",
    )
    assert (pair.level, pair.caller, pair.point, pair.call) == (
        1,
        submit,
        _line(MADE, "G2"),
        _line(MADE, "G3"),
    )
    assert (pair.row, pair.row_why, pair.object) == (HISTORY_ROW_CAN, "", "jobs[]")
    assert pair.row_via == [
        f"{submit}:{_line(MADE, 'G3')} -> {archive_all} jobs[]",
        f"{archive_all}:{_line(PROXY, 'B1')} element job",
        f"{archive_all}:{_line(PROXY, 'B2')} fill done[]",
        f"{archive_all}:{_line(PROXY, 'B3')} element one",
        f"{archive_all}:{_line(PROXY, 'H6')} one.PandaID",
    ]
    assert history_pairs_resolve(fragment).passed


def test_a_function_handed_on_is_not_the_same_pass(tmp_path: Path) -> None:
    fragment = _proxy(tmp_path, ("DBProxy.handed", "A6"))
    branch = _branch(fragment, "A6")
    assert (branch.history, branch.history_hole) == ([], "")


WAYS = """\
from pkg.proxy import DBProxy
from pkg.spec import JobSpec


class Ways(DBProxy):
    def kill_both(self, job: JobSpec):
        var_map = {}
        var_map[":PandaID"] = job.PandaID
        var_map[":jobStatus"] = "cancelled"  # V1
        self.cur.execute("UPDATE ATLAS_PANDA.jobsActive4 SET jobStatus=:jobStatus WHERE PandaID=:PandaID", var_map)
        self.update_both([job])  # V2

    def update_both(self, jobs):
        for job in jobs:  # V11
            params = {"PandaID": job.PandaID}  # V12
            self.archive_both(params, job)  # V13

    def archive_both(self, async_params, job):
        done = [job]  # V3
        for one in done:  # V4
            self.note_change(one.PandaID, one.jobStatus)  # V5

    def peek_job(self, panda_id):
        job = JobSpec()
        job.PandaID = panda_id
        return job

    def kill_ids(self, ids):
        for one_id in ids:  # V6
            found = self.peek_job(one_id)  # V7
            self.note_change(found.PandaID, "killed")  # V8

    def kill_one(self, panda_id):
        var_map = {}
        var_map[":PandaID"] = panda_id
        var_map[":jobStatus"] = "cancelled"  # V9
        self.cur.execute("UPDATE ATLAS_PANDA.jobsActive4 SET jobStatus=:jobStatus WHERE PandaID=:PandaID", var_map)
        self.kill_ids([panda_id])  # V10
"""


def _ways(tmp_path: Path, marker: str) -> MapFragment:
    owner = {"V1": "Ways.kill_both", "V9": "Ways.kill_one"}[marker]
    return _build(
        tmp_path,
        [("ways", owner, marker, "JobSpec.jobStatus")],
        spec=SPEC,
        proxy=PROXY,
        ways=WAYS,
    )


def test_a_way_goes_by_the_path_that_reaches_the_write_not_the_first_one_in(
    tmp_path: Path,
) -> None:
    """updateJobs: one call hands archiveJob the job and a dict of its PandaID; the dict comes in first, the job's row
    is written from the job."""
    fragment = _ways(tmp_path, "V1")
    (pair,) = _branch(fragment, "V1").history or []
    update, both = "pkg/ways.py::Ways.update_both", "pkg/ways.py::Ways.archive_both"
    assert (pair.row, pair.object) == (HISTORY_ROW_CAN, "job.PandaID")
    assert pair.row_via == [
        f"pkg/ways.py::Ways.kill_both:{_line(WAYS, 'V2')} -> {update} jobs[].PandaID",
        f"{update}:{_line(WAYS, 'V11')} element job.PandaID",
        f"{update}:{_line(WAYS, 'V13')} -> {both} job.PandaID",
        f"{both}:{_line(WAYS, 'V3')} fill done[].PandaID",
        f"{both}:{_line(WAYS, 'V4')} element one.PandaID",
        f"{both}:{_line(WAYS, 'V5')} one.PandaID",
    ]
    assert history_pairs_resolve(fragment).passed
    pair.row_via = [
        *pair.row_via[:2],
        f"{update}:{_line(WAYS, 'V12')} fill params['PandaID']",
        f"{update}:{_line(WAYS, 'V13')} -> {both} async_params['PandaID']",
        pair.row_via[-1],
    ]
    assert any(
        "below no path it holds" in failure
        for failure in history_pairs_resolve(fragment).failures
    )


def test_a_way_through_a_call_that_hands_the_object_back_says_how_it_went_in(
    tmp_path: Path,
) -> None:
    """killJobs: each id of the list goes into peekJob, whose job comes back with it as its PandaID."""
    fragment = _ways(tmp_path, "V9")
    (pair,) = _branch(fragment, "V9").history or []
    ways, kill_ids = "pkg/ways.py::Ways", "pkg/ways.py::Ways.kill_ids"
    assert (pair.row, pair.object) == (HISTORY_ROW_CAN, "panda_id")
    assert pair.row_via == [
        f"{ways}.kill_one:{_line(WAYS, 'V10')} -> {kill_ids} ids[]",
        f"{kill_ids}:{_line(WAYS, 'V6')} element one_id",
        f"{kill_ids}:{_line(WAYS, 'V7')} <- {ways}.peek_job found.PandaID",
        f"{kill_ids}:{_line(WAYS, 'V8')} found.PandaID",
    ]
    assert history_pairs_resolve(fragment).passed
    pair.row_via = [pair.row_via[0], *pair.row_via[2:]]
    assert any(
        "the call takes no path held" in failure
        for failure in history_pairs_resolve(fragment).failures
    )


REKEYS = """\
from pkg.proxy import DBProxy
from pkg.spec import JobSpec


class Rekeys(DBProxy):
    def retry(self, job: JobSpec, new_id):
        job.jobStatus = "retried"  # K1
        job.PandaID = int(new_id)  # K2
        self.note_change(job.PandaID, job.jobStatus)  # K3

    def drop(self, job: JobSpec):
        job.jobStatus = "dropped"  # K4
        self.clear_key(job)  # K5
        self.note_change(job.PandaID, job.jobStatus)  # K6

    def clear_key(self, job):
        job.PandaID = None  # K7

    def store(self, jobs, new_ids):
        for i, job in enumerate(jobs):  # K8
            job.PandaID = new_ids[i]  # K9
            job.jobStatus = "defined"  # K10
            self.note_change(job.PandaID, job.jobStatus)  # K11

    def renew(self, job: JobSpec):
        job.jobStatus = "renewed"  # K12
        self.renew_and_note(job)  # K13

    def renew_and_note(self, job):
        job.PandaID = int(self.new_id())  # K14
        self.note_change(job.PandaID, job.jobStatus)  # K15

    def new_id(self):
        return 1

    def check(self):
        self.job.jobStatus = "checked"  # K16
        self.reload_job()  # K17
        self.note_change(self.job.PandaID, self.job.jobStatus)  # K18

    def reload_job(self):
        self.job = self.peek_new(self.job_id)  # K19

    def peek_new(self, panda_id):
        job = JobSpec()
        job.PandaID = panda_id  # K20
        return job
"""


def _rekeys(tmp_path: Path, *arms: tuple[str, str]) -> MapFragment:
    return _build(
        tmp_path,
        [
            ("rekeys", f"Rekeys.{owner}", marker, "JobSpec.jobStatus")
            for owner, marker in arms
        ],
        spec=SPEC,
        proxy=PROXY,
        rekeys=REKEYS,
    )


def test_a_key_given_again_between_the_arm_and_the_history_call_is_said(
    tmp_path: Path,
) -> None:
    """retryJob: the job is given the new PandaID after its arms and before its history row, which is the new key's."""
    fragment = _rekeys(
        tmp_path,
        ("retry", "K1"),
        ("drop", "K4"),
        ("store", "K10"),
        ("renew", "K12"),
        ("check", "K16"),
    )
    rekeys = "pkg/rekeys.py::Rekeys"
    (retried,) = _branch(fragment, "K1").history or []
    assert (retried.row, retried.call) == (HISTORY_ROW_SAME, _line(REKEYS, "K3"))
    assert retried.rekeyed == [
        HistoryRekey(at=f"{rekeys}.retry:{_line(REKEYS, 'K2')}", path="job.PandaID")
    ]
    # A call between them that is handed the job and clears its key; the literal is said.
    (dropped,) = _branch(fragment, "K4").history or []
    assert dropped.rekeyed == [
        HistoryRekey(
            at=f"{rekeys}.clear_key:{_line(REKEYS, 'K7')}",
            path="job.PandaID",
            value="None",
        )
    ]
    # The key given before the arm in the same pass is the pass's own; the next pass binds job again.
    (stored,) = _branch(fragment, "K10").history or []
    assert (stored.row, stored.rekeyed) == (HISTORY_ROW_SAME, [])
    # In the function the way goes into.
    (renewed,) = _branch(fragment, "K12").history or []
    assert renewed.row == HISTORY_ROW_CAN
    assert renewed.rekeyed == [
        HistoryRekey(
            at=f"{rekeys}.renew_and_note:{_line(REKEYS, 'K14')}", path="job.PandaID"
        )
    ]
    # A call between that binds the path again first: another object is loaded, not this one given a key.
    (checked,) = _branch(fragment, "K16").history or []
    assert (checked.row, checked.rekeyed) == (HISTORY_ROW_SAME, [])
    assert history_pairs_resolve(fragment).passed
    retried.rekeyed = [
        HistoryRekey(at=f"{rekeys}.retry:{_line(REKEYS, 'K1')}", path="job.PandaID")
    ]
    assert any(
        "no assignment of it on the line" in failure
        for failure in history_pairs_resolve(fragment).failures
    )


SPEC_PARENT = SPEC.replace(
    '("PandaID", "jobStatus")', '("PandaID", "jobStatus", "parentID")'
)

NEW_KEYS = """\
from pandacommon.pandalogger.LogWrapper import LogWrapper

from pkg.proxy import DBProxy
from pkg.spec import JobSpec

_logger = None


class NewKeys(DBProxy):
    def retry(self, job: JobSpec, new_id):
        tmp_log = LogWrapper(_logger, f"PandaID={job.PandaID}")  # N0
        tmp_log.debug(f"retry {job.PandaID}")  # N1
        job.jobStatus = "retried"  # N2
        job.parentID = job.PandaID  # N3
        job.PandaID = int(new_id)  # N4
        tmp_log.debug(f"new PandaID {job.parentID} -> {job.PandaID}")  # N5
        self.note_change(job.PandaID, job.jobStatus)  # N6

    def clone(self, job: JobSpec, new_id):
        tmp_log = LogWrapper(_logger, f"PandaID={job.PandaID}")  # N7
        job.jobStatus = "cloned"  # N8
        job.PandaID = int(new_id)  # N9
        self.note_change(job.PandaID, job.jobStatus)  # N10
        tmp_log.debug("done")  # N11

    def lose(self, job: JobSpec, other: JobSpec, new_id):
        tmp_log = LogWrapper(_logger, f"PandaID={job.PandaID}")  # N12
        job.jobStatus = "lost"  # N13
        job.parentID = job.PandaID  # N14
        job.parentID = None  # N15
        job.PandaID = int(new_id)  # N16
        tmp_log.debug(f"other {other.PandaID}")  # N17
        self.note_change(job.PandaID, job.jobStatus)  # N18
"""


def test_a_key_given_again_names_where_the_old_key_is_kept_and_where_the_new_one_is_printed(
    tmp_path: Path,
) -> None:
    """retryJob: parentID is given the old PandaID before the new one, and a print after it prints the new one."""
    fragment = _build(
        tmp_path,
        [
            ("new_keys", f"NewKeys.{owner}", marker, "JobSpec.jobStatus")
            for owner, marker in (("retry", "N2"), ("clone", "N8"), ("lose", "N13"))
        ],
        spec=SPEC_PARENT,
        proxy=PROXY,
        new_keys=NEW_KEYS,
    )
    owner = "pkg/new_keys.py::NewKeys"
    (retried,) = _branch(fragment, "N2").history or []
    # The print before the new key is the old key's; the one after prints both.
    assert retried.rekeyed == [
        HistoryRekey(
            at=f"{owner}.retry:{_line(NEW_KEYS, 'N4')}",
            path="job.PandaID",
            kept_in=["parentID"],
            printed_at=[_line(NEW_KEYS, "N5")],
        )
    ]
    # Nothing keeps the old key, and no print after names the new one: said by both lists empty.
    (cloned,) = _branch(fragment, "N8").history or []
    assert cloned.rekeyed == [
        HistoryRekey(at=f"{owner}.clone:{_line(NEW_KEYS, 'N9')}", path="job.PandaID")
    ]
    # The old key kept, then given another value; a print of another object's key.
    (lost,) = _branch(fragment, "N13").history or []
    assert lost.rekeyed == [
        HistoryRekey(at=f"{owner}.lose:{_line(NEW_KEYS, 'N16')}", path="job.PandaID")
    ]
    assert history_pairs_resolve(fragment).passed
    retried.rekeyed[0].kept_in = ["jobStatus"]
    retried.rekeyed[0].printed_at = [_line(NEW_KEYS, "N2")]
    failures = history_pairs_resolve(fragment).failures
    assert any(
        "jobStatus is bound to the key by no alias of the row" in one
        for one in failures
    )
    assert any(
        "is no print of the function spelling the key" in one for one in failures
    )


TASKS = """\
from pkg.proxy import DBProxy
from pkg.spec import JobSpec


class JediTaskSpec:
    _attributes = ("jediTaskID", "status")


class Tasks(DBProxy):
    def note_task(self, task_id, status):
        sql = "INSERT INTO ATLAS_PANDA.TASKS_STATUSLOG (jediTaskID,status) VALUES (:jediTaskID,:status)"
        self.cur.execute(sql, {":jediTaskID": task_id, ":status": status})

    def clear(self, spec):
        spec.jediTaskID = None  # T1

    def job_side(self, job: JobSpec):
        job.jobStatus = "x"  # T2
        self.clear(job)  # T3
        self.note_change(job.PandaID, job.jobStatus)  # T4

    def task_side(self, task: JediTaskSpec):
        task.status = "y"  # T5
        self.clear(task)  # T6
        self.note_task(task.jediTaskID, task.status)  # T7
"""


def test_a_call_between_is_read_for_the_key_of_the_pairs_class(tmp_path: Path) -> None:
    """One callee, one path, asked for a job's PandaID and then for a task's jediTaskID: each its own answer."""
    fragment = _build(
        tmp_path,
        [
            ("tasks", "Tasks.job_side", "T2", "JobSpec.jobStatus"),
            ("tasks", "Tasks.task_side", "T5", "JediTaskSpec.status"),
        ],
        spec=SPEC,
        proxy=PROXY,
        tasks=TASKS,
    )
    (job,) = _branch(fragment, "T2").history or []
    (task,) = _branch(fragment, "T5").history or []
    assert (job.row, job.rekeyed) == (HISTORY_ROW_SAME, [])
    assert task.row == HISTORY_ROW_SAME
    assert task.rekeyed == [
        HistoryRekey(
            at=f"pkg/tasks.py::Tasks.clear:{_line(TASKS, 'T1')}",
            path="spec.jediTaskID",
            value="None",
        )
    ]


LOOPS = """\
from pkg.proxy import DBProxy


class JediTaskSpec:
    _attributes = ("jediTaskID", "status")


class Loops(DBProxy):
    def note_task(self, task_id, status):
        sql = "INSERT INTO ATLAS_PANDA.TASKS_STATUSLOG (jediTaskID,status) VALUES (:jediTaskID,:status)"
        self.cur.execute(sql, {":jediTaskID": task_id, ":status": status})

    def restart(self, first, second):
        sql = "UPDATE ATLAS_PANDA.JEDI_Tasks SET status=:status WHERE jediTaskID=:jediTaskID"
        for task_id in first:  # L1F
            var_map = {}
            var_map[":jediTaskID"] = task_id
            var_map[":status"] = "defined"  # L1
            self.cur.execute(sql, var_map)  # L1X
        for task_id in second:  # L2F
            var_map = {}
            var_map[":jediTaskID"] = task_id
            var_map[":status"] = "ready"  # L2
            self.cur.execute(sql, var_map)  # L2X
            self.note_task(task_id, "ready")  # L3

    def retry(self, task_id, done):
        new_status = None  # L4
        if done:
            new_status = "finished"  # L5
        else:
            new_status = "ready"
        var_map = {":jediTaskID": task_id, ":status": new_status}
        self.cur.execute("UPDATE ATLAS_PANDA.JEDI_Tasks SET status=:status WHERE jediTaskID=:jediTaskID", var_map)
        self.note_task(task_id, new_status)  # L6

    def broken(self, task_id):
        sql = "UPDATE ATLAS_PANDA.JEDI_Tasks SET status='broken' "  # L7
        sql += "WHERE jediTaskID=:jediTaskID"
        var_map = {":jediTaskID": task_id}
        self.cur.execute(sql, var_map)  # L7X
        self.cur.execute(sql, var_map)  # L7Y
        self.note_task(task_id, "broken")  # L8

    def retry_text(self, task_id, ok):
        sql_new ="UPDATE ATLAS_PANDA.JEDI_Tasks SET status='ready' WHERE jediTaskID=:jediTaskID"  # R1
        sql_back = "UPDATE ATLAS_PANDA.JEDI_Tasks SET modificationTime=CURRENT_DATE WHERE jediTaskID=:jediTaskID"
        var_map = {":jediTaskID": task_id}
        if ok:
            self.cur.execute(sql_new, var_map)
            self.note_task(task_id, "ready")  # R2
        else:
            self.cur.execute(sql_back, var_map)
            self.note_task(task_id, "kept")  # R3
"""


def test_an_sql_arm_is_the_row_of_the_calls_it_gives_its_value_to(
    tmp_path: Path,
) -> None:
    """A loop's map bound again is that loop's own; a value through a name, or in the text, is the arm's."""
    fragment = _build(
        tmp_path,
        [
            ("loops", f"Loops.{owner}", marker, "JediTaskSpec.status")
            for owner, marker in (
                ("restart", "L1"),
                ("restart", "L2"),
                ("retry", "L4"),
                ("retry", "L5"),
                ("broken", "L7"),
            )
        ],
        spec=SPEC,
        proxy=PROXY,
        loops=LOOPS,
    )
    rows = {row.owner: row for row in fragment.reach_graph}
    # One statement run in two loops, each with its own map, is two writes; run twice with one map, one.
    first, second = (f"task_id@{_line(LOOPS, marker)}" for marker in ("L1F", "L2F"))
    assert [
        (write.lines, [key.path for key in write.keys])
        for write in rows["pkg/loops.py::Loops.restart"].writes
    ] == [([_line(LOOPS, "L1X")], [first]), ([_line(LOOPS, "L2X")], [second])]
    (twice,) = rows["pkg/loops.py::Loops.broken"].writes
    assert twice.lines == [_line(LOOPS, "L7X"), _line(LOOPS, "L7Y")]

    def row(marker: str) -> tuple[str, str]:
        (pair,) = _branch(fragment, marker).history or []
        return pair.row, pair.row_why

    assert row("L1") == (f"{runs.ROW_ANOTHER}{second}", "")
    assert row("L2") == (HISTORY_ROW_CAN, "")
    # None is assigned again on every way to the bind: no call writes it.
    assert row("L4") == ("", runs.ROW_NO_START)
    assert row("L5") == (HISTORY_ROW_CAN, "")
    assert row("L7") == (HISTORY_ROW_CAN, "")


ONEPASS = """\
from pkg.proxy import DBProxy


class OnePass(DBProxy):
    def sort_jobs(self, jobs):
        activate = []
        update = []
        for job in jobs:
            if job.jobStatus == "failed":
                continue
            elif job.dispatchDBlock == "NULL":
                activate.append(job)  # O1
            else:
                update.append(job)  # O2
        for job in activate:
            job.jobStatus = "finished"  # O3
        for other in update:
            self.record(other)  # O4

    def record(self, job):
        self.note_change(job.PandaID, job.jobStatus)

    def jumbo(self, jobs):
        failed = []
        for job in jobs:
            if job.jobStatus == "failed":
                failed.append(job)  # J0
                continue
            try:
                self.register(job)
            except Exception:
                job.jobStatus = "failed"  # J1
                failed.append(job)  # J2
                continue
        for one in failed:
            self.note_change(one.PandaID, one.jobStatus)  # J3

    def check(self, jobs, candidates):
        for job in jobs:
            if not candidates:
                continue
            job.jobStatus = "checked"  # K1
            taken = job  # K2
            candidates = [one for one in candidates if one != job.computingSite]
            if not candidates:
                self.note_change(taken.PandaID, taken.jobStatus)  # K3
"""


def test_a_way_is_one_that_runs_in_one_pass(tmp_path: Path) -> None:
    """The record reads a container's elements without order or branches; a way it gives is checked before ``can``.

    ``Setupper.update_jobs``: a job goes into ``activate`` in the ``elif`` or
    into ``update`` in the ``else``, so ``activate``'s jobs are not
    ``update``'s.  ``setup_jumbo_jobs``: a job failed in the ``except`` goes
    into ``failed`` at J2, not at J0 above it, which a pass that reaches J1 has
    left by its ``continue``.  ``AtlasProdTaskBroker``: ``if not candidates``
    tested after ``candidates`` is given a new value does not contradict the
    early exit above.  An SQL arm in an ``if`` is not the row the ``else``
    records.
    """
    fragment = _build(
        tmp_path,
        [
            ("onepass", "OnePass.sort_jobs", "O3", "JobSpec.jobStatus"),
            ("onepass", "OnePass.jumbo", "J1", "JobSpec.jobStatus"),
            ("onepass", "OnePass.check", "K1", "JobSpec.jobStatus"),
            ("loops", "Loops.retry_text", "R1", "JediTaskSpec.status"),
        ],
        spec=SPEC,
        proxy=PROXY,
        onepass=ONEPASS,
        loops=LOOPS,
    )

    def pairs(marker: str) -> list[HistoryPair]:
        return _branch(fragment, marker).history or []

    (sorted_out,) = pairs("O3")
    assert (sorted_out.row, sorted_out.row_why.split(":")[0]) == (
        "",
        runs.ROW_NO_WAY.split(":")[0],
    )
    assert "OnePass.sort_jobs:" in sorted_out.row_why

    (failed,) = pairs("J1")
    assert failed.row == HISTORY_ROW_CAN
    assert any(f":{_line(ONEPASS, 'J2')} " in step for step in failed.row_via)
    assert not any(f":{_line(ONEPASS, 'J0')} " in step for step in failed.row_via)

    (checked,) = pairs("K1")
    assert checked.row == HISTORY_ROW_CAN

    rows = {pair.call: pair.row for pair in pairs("R1")}
    assert rows == {_line(LOOPS, "R2"): HISTORY_ROW_CAN, _line(LOOPS, "R3"): ""}


def _flags(tmp_path: Path, *markers: str) -> MapFragment:
    arms = [
        ("flags", f"Flags.{owner}", marker, "JobSpec.jobStatus")
        for owner, marker in (
            ("set_diag", "F1"),
            ("set_diag", "F2"),
            ("peek_one", "F3"),
        )
        if marker in markers
    ]
    return _build(tmp_path, arms, spec=SPEC, proxy=PROXY, flags=FLAGS)


def _callers(branch: Branch) -> set[str]:
    return {pair.caller.rpartition(".")[2] for pair in branch.history or []}


def test_a_flag_a_call_hands_rules_out_the_arm_it_tests(tmp_path: Path) -> None:
    """``setErrDiag(diag)`` leaves ``append`` False: its append arm is not on that caller's pass, and is on the one handing True."""
    fragment = _flags(tmp_path, "F1", "F2")
    assert _callers(_branch(fragment, "F1")) == {"appending", "top"}
    assert _callers(_branch(fragment, "F2")) == {"plain", "forwarding", "top"}


def test_a_flag_handed_on_is_read_where_it_is_given(tmp_path: Path) -> None:
    """``forward(job, flag)`` hands ``flag`` on as ``append``; ``forwarding`` gives it False."""
    fragment = _flags(tmp_path, "F1")
    assert "forwarding" not in _callers(_branch(fragment, "F1"))


def test_a_way_ruled_out_does_not_stop_another_way_to_the_same_caller(
    tmp_path: Path,
) -> None:
    """``top`` calls ``via_true`` and ``via_plain``: the arm each rules out is still ``top``'s through the other."""
    fragment = _flags(tmp_path, "F1", "F2")
    for marker, through in (("F1", "via_true"), ("F2", "via_plain")):
        (pair,) = [
            one
            for one in _branch(fragment, marker).history or []
            if one.caller.endswith("top")
        ]
        assert pair.level == 2
        assert f"Flags.{through}" not in pair.caller


def test_an_early_exit_a_call_decides_rules_out_what_follows_it(
    tmp_path: Path,
) -> None:
    fragment = _flags(tmp_path, "F3")
    branch = _branch(fragment, "F3")
    assert branch.history == []
    assert runs.HISTORY_RULED_OUT in branch.history_hole


def test_ruled_out_by_reads_an_early_exit_and_leaves_an_open_one_open() -> None:
    tree = ast.parse(
        "def f(a, b, c):\n"
        "    if a:\n"
        "        return\n"
        "    for x in c:\n"
        "        if b is None:\n"
        "            continue\n"
        "        y = 1\n"
        "    if not a and c:\n"
        "        raise ValueError\n"
        "    z = 2\n"
    )
    pathcond.attach_parents(tree)
    func = tree.body[0]
    y = func.body[1].body[1]
    z = func.body[3]
    assert pathcond.ruled_out_by(y, func, {"a": True})
    assert pathcond.ruled_out_by(y, func, {"a": False, "b": None})
    assert not pathcond.ruled_out_by(y, func, {"a": False, "b": 1})
    assert not pathcond.ruled_out_by(z, func, {"a": False})  # c is open
    assert pathcond.ruled_out_by(z, func, {"a": True})
    assert [ast.unparse(one) for one in pathcond.tests_on(z, func)] == [
        "a",
        "not a and c",
    ]


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
    assert archive.key == [
        RunKey(expression="job.PandaID", columns=["JobSpec.PandaID"])
    ]
    (run,) = prints["pkg/proxy.py::DBProxy.run"].runs
    assert run.mark == "<PandaID={}>"
    assert [key.columns for key in run.key] == [["JobSpec.PandaID"]]
    (statement,) = prints["pkg/proxy.py::DBProxy.archiveJob"].statements
    assert statement.runs == [_line(PROXY, "R0")]


def test_a_mark_given_none_is_a_time_stamp_or_nothing_as_its_factory_writes(
    tmp_path: Path,
) -> None:
    fragment = _build(tmp_path, [], marks=MARKS)
    (node,) = fragment.function_prints
    by_logger = {run.logger: run for run in node.runs}
    time = [RunKey(expression="", columns=[RUN_KEY_TIME])]
    assert by_logger["a"].key == time
    assert by_logger["b"].key == []
    assert by_logger["c"].key == time
    assert by_logger["d"].key == []
    assert by_logger["e"].mark == "<jediTaskID={} n={}>"
    line = _line(MARKS, "M0")
    no_use = KeyStop(kind="no_use", at=["marks"])
    assert [(key.columns, key.stops) for key in by_logger["e"].key] == [
        ([], [KeyStop(kind="undeclared", name="task", at=[f"marks:{line}"]), no_use]),
        ([], [KeyStop(kind="parameter", name="n", at=["marks"]), no_use]),
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
    assert run.key == [
        RunKey(
            expression="str(pandaID)",
            stops=[
                KeyStop(kind="parameter", name="pandaID", at=["Watcher.__init__"]),
                KeyStop(kind="no_use", at=["Watcher.__init__"]),
            ],
        )
    ]
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
    (followed,) = _branch(fragment, "A5").history or []
    followed.row_via = [
        "pkg/proxy.py::DBProxy.run:1 -> pkg/proxy.py::DBProxy.archive_history job",
        *followed.row_via[1:],
    ]
    result = history_pairs_resolve(fragment)
    assert not result.passed
    assert any(
        "no call binding of the line into the callee" in failure
        for failure in result.failures
    )
    followed.row_via = [
        f"pkg/proxy.py::DBProxy.run:{_line(PROXY, 'C2')} -> pkg/proxy.py::DBProxy.archive_history other",
        *followed.row_via[1:],
    ]
    result = history_pairs_resolve(fragment)
    assert any(
        "the call hands no path held as the one the callee takes" in failure
        for failure in result.failures
    )
    for row in fragment.reach_graph:
        row.history = []
    result = history_pairs_resolve(fragment)
    assert not result.passed
    assert any("is no history call" in failure for failure in result.failures)


SETUP = """\
from pkg.proxy import DBProxy
from pkg.spec import JobSpec


class Setup:
    def __init__(self, jobs: list[JobSpec]):
        self.jobs = jobs
        self.proxy: DBProxy = DBProxy()

    def run(self):
        tag = self.jobs[0]
        for one in self.jobs:
            pass
        self.assign()  # S1
        self.archive()  # S2

    def assign(self):
        for job in self.jobs + [None]:
            job.jobStatus = "assigned"  # A20

    def archive(self):
        for job in self.jobs:
            self.proxy.note_change(job.PandaID, job.jobStatus)  # H20
"""


def test_a_loop_over_a_sum_runs_over_its_terms_and_the_object_is_any_element(
    tmp_path: Path,
) -> None:
    """setupper: ``for job in self.jobs + [None]``; the caller reads ``self.jobs[0]`` too, but the arm's is any job."""
    fragment = _build(
        tmp_path,
        [("setup", "Setup.assign", "A20", "JobSpec.jobStatus")],
        spec=SPEC,
        proxy=PROXY,
        setup=SETUP,
    )
    (pair,) = [
        one
        for one in _branch(fragment, "A20").history or []
        if one.call == _line(SETUP, "S2")
    ]
    assert (pair.level, pair.caller, pair.point, pair.row, pair.object) == (
        1,
        "pkg/setup.py::Setup.run",
        _line(SETUP, "S1"),
        HISTORY_ROW_CAN,
        "self.jobs[]",
    )


KEYS = """\
from pandacommon.pandalogger.LogWrapper import LogWrapper

from pkg.proxy import DBProxy
from pkg.spec import JobSpec

_logger = None


def sender_of():
    return None


class Keys(DBProxy):
    def kill(self, panda_id):
        tmp_log = LogWrapper(_logger, f"PandaID={panda_id}")  # K1
        tmp_log.debug("kill")
        self.cur.execute("UPDATE ATLAS_PANDA.jobsActive4 SET jobStatus=:s WHERE PandaID=:PandaID", {":PandaID": panda_id, ":s": "x"})

    def peek(self, pid):
        tmp_log = LogWrapper(_logger, f"<pid={pid}>")  # K2
        tmp_log.debug("peek")
        self.cur.execute("SELECT jobStatus FROM ATLAS_PANDA.jobsActive4 WHERE pandaid=:id", {":id": pid})

    def refine(self, job_id):
        tmp_log = LogWrapper(_logger, f"<job={job_id}>")  # K3
        tmp_log.debug("refine")
        self.read_params(job_id)

    def read_params(self, panda_id):
        self.select_params(panda_id)

    def select_params(self, one):
        self.cur.execute("SELECT jobStatus FROM ATLAS_PANDA.jobsActive4 WHERE PandaID=:p", {":p": one})

    def other_table(self, task_id):
        tmp_log = LogWrapper(_logger, f"<task={task_id}>")  # K4
        tmp_log.debug("other")
        self.cur.execute("SELECT taskName FROM ATLAS_PANDA.T_TASK WHERE taskID=:t", {":t": task_id})  # K5
        self.cur.execute("SELECT status FROM ATLAS_PANDA.T_TASK WHERE taskID=:t", {":t": task_id})  # K5b

    def listed(self, pid):
        tmp_log = LogWrapper(_logger, f"<pid={pid}>")  # K6
        tmp_log.debug("listed")
        self.peek_jobs([pid])  # K7

    def peek_jobs(self, ids):
        for one in ids:
            self.cur.execute("SELECT jobStatus FROM ATLAS_PANDA.jobsActive4 WHERE PandaID=:p", {":p": one})

    def aliased(self, panda_id):
        tmp_log = LogWrapper(_logger, f"<id={panda_id}>")  # K11
        tmp_log.debug("aliased")
        tmp_id = panda_id
        self.cur.execute("SELECT jobStatus FROM ATLAS_PANDA.jobsActive4 WHERE PandaID=:p", {":p": tmp_id})

    def part(self, task):
        tmp_log = LogWrapper(_logger, f"<id={task.panda_id}>")  # K12
        tmp_log.debug("part")
        self.select_part(task)

    def select_part(self, one):
        self.cur.execute("SELECT jobStatus FROM ATLAS_PANDA.jobsActive4 WHERE PandaID=:p", {":p": one.panda_id})

    def made(self, pid):
        held = self.make(pid)
        tmp_log = LogWrapper(_logger, f"<id={held.panda_id}>")  # K13
        tmp_log.debug("made")

    def make(self, one):
        held = Box()
        held.panda_id = one
        self.cur.execute("SELECT jobStatus FROM ATLAS_PANDA.jobsActive4 WHERE PandaID=:p", {":p": held.panda_id})
        return held

    def handed_on(self, pid):
        tmp_log = LogWrapper(_logger, f"<pid={pid}>")  # K14
        tmp_log.debug("handed on")
        self.many(pid)  # K15
        senders = {"x": sender_of()}
        senders["x"].send(pid)  # K16

    def many(self, *ids):
        pass

    def deep(self, pid):
        tmp_log = LogWrapper(_logger, f"<pid={pid}>")  # K17
        tmp_log.debug("deep")
        box = Box()
        box.a.b.c.d.e.f.g = pid  # K18

    def bound(self, job: JobSpec):
        tmp_id = job.PandaID
        tmp_log = LogWrapper(_logger, f"<PandaID={tmp_id}>")  # K8
        tmp_log.debug("bound")


class Closer:
    def __init__(self, job: JobSpec):
        self.panda_id = job.PandaID
        self.logger = LogWrapper(_logger, f"<PandaID={self.panda_id}>")  # K9

    def run(self):
        self.logger.debug("close")


class Adder:
    def __init__(self, job_id):
        self.job_id = job_id
        self.logger = LogWrapper(_logger, str(self.job_id))  # K10

    def run(self):
        self.logger.debug("add")


class Box:
    def send(self, one):
        pass


class Sender:
    def __init__(self, pid):
        self.pid = pid
        self.logger = LogWrapper(_logger, str(self.pid))  # K19
        senders = {"x": sender_of()}
        senders["x"].send(self.pid)  # K20

    def run(self):
        self.logger.debug("send")
"""


def _keys(tmp_path: Path) -> dict[int, RunKey]:
    """The one key of each run of KEYS, by the line the run starts on."""
    fragment = _build(tmp_path, [], spec=SPEC, proxy=PROXY, keys=KEYS)
    found: dict[int, RunKey] = {}
    for node in fragment.function_prints:
        for run in node.runs:
            (found[run.line],) = run.key
    return found


def test_a_key_no_declaration_names_is_the_column_its_function_uses_it_as(
    tmp_path: Path,
) -> None:
    """A key of a row the function writes, a term it reads rows by (spelled as the class declares it), two calls down."""
    keys = _keys(tmp_path)
    assert (keys[_line(KEYS, "K1")].columns, keys[_line(KEYS, "K1")].stops) == (
        ["JobSpec.PandaID"],
        [],
    )
    assert (keys[_line(KEYS, "K2")].columns, keys[_line(KEYS, "K2")].stops) == (
        ["JobSpec.PandaID"],
        [],
    )
    assert (keys[_line(KEYS, "K3")].columns, keys[_line(KEYS, "K3")].stops) == (
        ["JobSpec.PandaID"],
        [],
    )
    assert (keys[_line(KEYS, "K8")].columns, keys[_line(KEYS, "K8")].stops) == (
        ["JobSpec.PandaID"],
        [],
    )


def test_a_key_is_the_column_it_is_used_as_wherever_it_is_followed(
    tmp_path: Path,
) -> None:
    """Followed as an object is: inside a list to a callee looping over it, through a name the function gives it, as a part of an object handed whole, and into the call returning the object it is read from."""
    keys = _keys(tmp_path)
    for marker in ("K6", "K11", "K12", "K13"):
        assert (keys[_line(KEYS, marker)].columns, keys[_line(KEYS, marker)].stops) == (
            ["JobSpec.PandaID"],
            [],
        ), marker


def test_a_key_its_uses_name_no_rows_column_says_where_the_reading_stopped(
    tmp_path: Path,
) -> None:
    """A table no class holds names no column, its sites said once; where following it stops, the stop says where."""
    keys = _keys(tmp_path)
    sites = [
        f"Keys.other_table:{_line(KEYS, 'K5')}",
        f"Keys.other_table:{_line(KEYS, 'K5b')}",
    ]
    assert (keys[_line(KEYS, "K4")].columns, keys[_line(KEYS, "K4")].stops) == (
        [],
        [
            KeyStop(kind="parameter", name="task_id", at=["Keys.other_table"]),
            KeyStop(kind="no_class", name="T_TASK.taskID", at=sites),
        ],
    )
    assert (keys[_line(KEYS, "K14")].columns, keys[_line(KEYS, "K14")].stops) == (
        [],
        [
            KeyStop(kind="parameter", name="pid", at=["Keys.handed_on"]),
            KeyStop(kind="spread", at=[f"Keys.handed_on:{_line(KEYS, 'K15')}"]),
            KeyStop(kind="hole", at=[f"Keys.handed_on:{_line(KEYS, 'K16')}"]),
        ],
    )
    assert (keys[_line(KEYS, "K17")].columns, keys[_line(KEYS, "K17")].stops) == (
        [],
        [
            KeyStop(kind="parameter", name="pid", at=["Keys.deep"]),
            KeyStop(
                kind="capped",
                name="box.a.b.c.d.e.f.g",
                at=[f"Keys.deep:{_line(KEYS, 'K18')}"],
            ),
        ],
    )


def test_a_field_of_a_class_holding_no_row_is_read_in_its_class(tmp_path: Path) -> None:
    """``Closer.panda_id`` is assigned ``job.PandaID``; ``Adder.job_id`` a parameter, its callers not read."""
    keys = _keys(tmp_path)
    assert (keys[_line(KEYS, "K9")].columns, keys[_line(KEYS, "K9")].stops) == (
        ["Closer.panda_id", "JobSpec.PandaID"],
        [],
    )
    adder = keys[_line(KEYS, "K10")]
    assert adder.columns == ["Adder.job_id"]
    assert adder.stops == [
        KeyStop(kind="no_row", name="Adder"),
        KeyStop(kind="parameter", name="job_id", at=["Adder.__init__"]),
        KeyStop(kind="no_use", at=["Adder.__init__"]),
    ]
    sender = keys[_line(KEYS, "K19")]
    assert sender.stops == [
        KeyStop(kind="no_row", name="Sender"),
        KeyStop(kind="parameter", name="pid", at=["Sender.__init__"]),
        KeyStop(kind="hole", at=[f"Sender.__init__:{_line(KEYS, 'K20')}"]),
    ]


def test_a_keys_why_is_its_stops_said_and_is_not_kept_in_the_dump() -> None:
    """The wording is the reader's: rewording a stop moves no map."""
    key = RunKey(
        expression="task_id",
        stops=[
            KeyStop(kind="parameter", name="task_id", at=["Keys.other_table"]),
            KeyStop(
                kind="no_class",
                name="T_TASK.taskID",
                at=["Keys.other_table:5", "Keys.other_table:6"],
            ),
        ],
    )
    assert key.why == (
        "task_id is a parameter of Keys.other_table: what the callers hand is not read; "
        "keys T_TASK.taskID, a table no class holds, at Keys.other_table:5, Keys.other_table:6"
    )
    assert "why" not in key.model_dump()
    cut = RunKey(
        expression="pid",
        stops=[KeyStop(kind="cut", what="states", at=["Keys.deep"])],
    )
    assert cut.why == (
        "following it from Keys.deep stopped at its cap on states, so it can be used further than read"
    )
    assert RunKey(expression="", columns=[RUN_KEY_TIME]).why == ""


HANDED = """\
from pandacommon.pandalogger.LogWrapper import LogWrapper

from pkg.proxy import DBProxy
from pkg.spec import JobSpec

_logger = None


class Post(DBProxy):
    def process(self, job: JobSpec):
        tmp_log = LogWrapper(_logger, f"<PandaID={job.PandaID}>")  # L0
        tmp_log.debug("start")
        self.post(job, tmp_log)

    def post(self, job: JobSpec, log):
        log.debug("post")
        job.jobStatus = "done"  # L1
        self.note_change(job.PandaID, job.jobStatus)

    def quiet(self, job: JobSpec):
        _logger.debug("quiet")
        job.jobStatus = "quiet"  # L2
        self.note_change(job.PandaID, job.jobStatus)
"""


def test_a_pair_printing_through_a_handed_logger_is_in_the_run_of_who_builds_it(
    tmp_path: Path,
) -> None:
    """``doPostProcess(task_spec, tmp_log)``: the runs are the caller's that builds the logger; the module's marks none."""
    fragment = _build(
        tmp_path,
        [
            ("handed", "Post.post", "L1", "JobSpec.jobStatus"),
            ("handed", "Post.quiet", "L2", "JobSpec.jobStatus"),
        ],
        spec=SPEC,
        proxy=PROXY,
        handed=HANDED,
    )
    (handed,) = _branch(fragment, "L1").history or []
    assert (handed.caller, handed.runs, handed.runs_in, handed.run_hole) == (
        "pkg/handed.py::Post.post",
        [_line(HANDED, "L0")],
        "pkg/handed.py::Post.process",
        "",
    )
    (quiet,) = _branch(fragment, "L2").history or []
    assert (quiet.runs, quiet.runs_in, quiet.run_hole) == (
        [],
        "",
        runs.RUN_MODULE.format(logger="_logger"),
    )
    assert history_pairs_resolve(fragment).passed
    handed.runs_in = ""
    assert any(
        "is not one of the caller's runs" in one
        for one in history_pairs_resolve(fragment).failures
    )


def test_a_statement_printing_through_a_handed_logger_names_the_run_that_builds_it(
    tmp_path: Path,
) -> None:
    """``doGenerate`` prints through the ``tmpLog`` its caller binds: the statement names that run where a reader
    finds it, the caller's node and line.  A statement on the owner's own logger, or the module's, names none."""
    fragment = _build(tmp_path, [], spec=SPEC, proxy=PROXY, handed=HANDED)
    prints = {node.owner: node for node in fragment.function_prints}
    built = RunRef(owner="pkg/handed.py::Post.process", line=_line(HANDED, "L0"))
    (post,) = prints["pkg/handed.py::Post.post"].statements
    assert (post.runs, post.handed_runs) == ([], [built])
    (start,) = prints["pkg/handed.py::Post.process"].statements
    assert (start.runs, start.handed_runs) == ([_line(HANDED, "L0")], [])
    (quiet,) = prints["pkg/handed.py::Post.quiet"].statements
    assert (quiet.runs, quiet.handed_runs) == ([], [])
    assert history_pairs_resolve(fragment).passed
    post.handed_runs = [RunRef(owner="pkg/handed.py::Post.quiet", line=built.line)]
    assert any(
        "is in no node" in one for one in history_pairs_resolve(fragment).failures
    )


HANDED_MANY = """\
from pandacommon.pandalogger.LogWrapper import LogWrapper

from pkg.spec import JobSpec

_logger = None


class Gen:
    def __init__(self, job: JobSpec):
        self.logger = LogWrapper(_logger, f"<PandaID={job.PandaID}>")  # G0

    def run(self):
        self.logger.debug("run")
        Plugin(self.logger).execute()


class Plugin:
    def __init__(self, log):
        self.log = log

    def execute(self):
        self.log.debug("execute")


def first(job: JobSpec):
    tmp_log = LogWrapper(_logger, f"<PandaID={job.PandaID}>")  # F0
    tmp_log.debug("first")
    helper(tmp_log)


def second(job: JobSpec):
    tmp_log = LogWrapper(_logger, f"<jediTaskID={job.jediTaskID}>")  # S0
    tmp_log.debug("second")
    helper(tmp_log)


class Proxy:
    def create_tagged_logger(self, comment, tag=None):
        return LogWrapper(_logger, f"{comment} < {tag} >")

    def archive(self, job: JobSpec):
        tmp_log = self.create_tagged_logger("archive", f"PandaID={job.PandaID}")  # C0
        tmp_log.debug("archive")
        helper(tmp_log)


def helper(log):
    log.debug("help")
"""


def test_a_handed_logger_names_every_run_it_can_be_and_an_objects_run_by_a_node_holding_it(
    tmp_path: Path,
) -> None:
    """``dump_error_message`` is handed every DB proxy method's logger, and names a run for each -- one made by
    ``create_tagged_logger`` where the caller binds it.  An object's run whose ``__init__`` prints nothing is named
    by the class's method that prints and so holds it."""
    fragment = _build(tmp_path, [], spec=SPEC, many=HANDED_MANY)
    prints = {node.owner: node for node in fragment.function_prints}
    (helped,) = prints["pkg/many.py::helper"].statements
    assert helped.handed_runs == [
        RunRef(owner="pkg/many.py::Proxy.archive", line=_line(HANDED_MANY, "C0")),
        RunRef(owner="pkg/many.py::first", line=_line(HANDED_MANY, "F0")),
        RunRef(owner="pkg/many.py::second", line=_line(HANDED_MANY, "S0")),
    ]
    assert "pkg/many.py::Gen.__init__" not in prints
    (executed,) = prints["pkg/many.py::Plugin.execute"].statements
    assert executed.handed_runs == [
        RunRef(owner="pkg/many.py::Gen.run", line=_line(HANDED_MANY, "G0"))
    ]
    assert history_pairs_resolve(fragment).passed
