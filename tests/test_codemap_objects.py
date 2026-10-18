"""Where an object goes and which of its columns are assigned -- ``codemap.panda.objects`` and ``codemap.objects``."""

from __future__ import annotations

import ast
from pathlib import Path

from bamboo.codemap import objects, reach
from bamboo.codemap.gates import object_records_resolve
from bamboo.codemap.gitsource import blob_sha
from bamboo.codemap.models import CallBinding, MapFragment, SourceModule
from bamboo.codemap.panda import objects as panda_objects
from bamboo.codemap.panda import pathcond, sql
from bamboo.codemap.panda.plugin import _DatabaseLeaves
from bamboo.codemap.panda.recognizers import sqlwrite

PACKAGE = "pkg"

SPEC = """\
class JobSpec:
    _attributes = ("PandaID", "jobStatus", "ddmErrorCode", "jobMetrics", "modificationTime")

    def __init__(self):
        for attr in self._attributes:
            object.__setattr__(self, attr, None)

    def pack(self, values):
        for attr, value in zip(self._attributes, values):
            object.__setattr__(self, attr, value)

    def setErrDiag(self, diag):
        self.ddmErrorCode = 1
"""

PROXY = """\
from pkg.spec import JobSpec


class DBProxy:
    def archiveJob(self, job: JobSpec):
        job.modificationTime = 1
        updated = [job]
        other = JobSpec()
        updated.append(other)
        other.jobMetrics = "another job's"
        varMap = {}
        varMap[":modificationTime"] = job.modificationTime
        varMap[":PandaID"] = job.PandaID
        sql = "UPDATE ATLAS_PANDA.jobsActive4 SET modificationTime=:modificationTime WHERE PandaID=:PandaID"
        self.cur.execute(sql, varMap)
        sql = "INSERT INTO ATLAS_PANDA.jobsArchived4 (PandaID,jobMetrics) VALUES (:PandaID,:jobMetrics)"
        self.cur.execute(sql, job.valuesMap())

    def killJob(self, panda_id):
        sql = "UPDATE ATLAS_PANDA.jobsActive4 SET jobMetrics=:jobMetrics WHERE PandaID=:PandaID"
        self.cur.execute(sql, {":jobMetrics": "killed", ":PandaID": panda_id})

    def insertNewJob(self, job: JobSpec):
        sql = "INSERT INTO ATLAS_PANDA.jobsDefined4 (PandaID,jobMetrics) VALUES (:PandaID,:jobMetrics)"
        self.cur.execute(sql, job.valuesMap(useSeq=True))

    def peekJob(self, panda_id) -> JobSpec:
        job = JobSpec()
        job.pack([panda_id])
        job.jobMetrics = "read"
        return job


class TaskBuffer:
    def __init__(self):
        self.proxy = DBProxy()

    def updateJobs(self, jobs: list[JobSpec]):
        for idx, job in enumerate(jobs):
            self.proxy.archiveJob(job)

    def killJobs(self, ids):
        for one in ids:
            self.proxy.killJob(one)

    def storeJobs(self, jobs):
        for job in jobs:
            self.proxy.insertNewJob(job)

    def peekJobs(self, ids) -> list[JobSpec]:
        found = []
        for one in ids:
            found.append(self.proxy.peekJob(one))
        return found
"""

PLUGIN = """\
class Plugin:
    def __init__(self, job):
        self.job = job

    def execute(self):
        self.job.ddmErrorCode = 2
"""

ADDER = """\
import threading

from pkg.plugin import Plugin
from pkg.proxy import TaskBuffer
from pkg.spec import JobSpec


class Adder:
    def __init__(self, tb: TaskBuffer):
        self.tb = tb
        self.job_id = 1
        self.job: JobSpec = JobSpec()

    def process(self, other):
        self.job.jobStatus = "failed"
        self.finish()
        plugin = Plugin(self.job)
        plugin.execute()
        self.tb.updateJobs([self.job])
        other.execute(self.job)
        other.execute(5)
        len(self.job.jobStatus)
        threading.Thread(target=self.worker, args=(self.job,)).start()

    def finish(self):
        self.job.jobMetrics = "m"

    def worker(self, job):
        pass

    def run(self):
        self.job = self.tb.peekJobs([1])[0]
        self.process(None)

    def close(self):
        self.job.modificationTime = 2
        self.tb.updateJobs([self.job])
        self.tb.killJobs([self.job_id])
        self.tb.storeJobs([JobSpec()])
"""


def _build(tmp_path: Path, **files: str) -> MapFragment:
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
    calls = sqlwrite.corpus_calls(modules, {PACKAGE: root})
    with sql.across_calls(calls):
        fragment = MapFragment(
            map_id="m",
            derived_from="v",
            reach_graph=reach.graph(calls, "m", "v", _DatabaseLeaves()),
        )
        panda_objects.attach(fragment, modules, calls)
    return fragment


def _adder(tmp_path: Path) -> dict:
    fragment = _build(tmp_path, spec=SPEC, proxy=PROXY, plugin=PLUGIN, adder=ADDER)
    return {row.owner: row for row in fragment.reach_graph}


def _line(text: str, marker: str) -> int:
    (line,) = [n for n, one in enumerate(text.splitlines(), 1) if marker in one]
    return line


def test_an_object_is_followed_into_self_calls_constructors_and_list_literals(tmp_path):
    """The adder's job: assigned in place, through ``self.finish()``, by the plugin it builds, and by archiveJob."""
    rows = _adder(tmp_path)
    found = objects.follow(rows, "pkg/adder.py::Adder.process", "self.job")

    where = {
        attr: sorted((site.owner.split("::")[1], site.line) for site in sites)
        for attr, sites in found.assigned.items()
    }
    assert where["jobStatus"] == [
        ("Adder.process", _line(ADDER, 'jobStatus = "failed"'))
    ]
    # archiveJob's other job shares a list with it, and is not it.
    assert where["jobMetrics"] == [("Adder.finish", _line(ADDER, 'jobMetrics = "m"'))]
    # Plugin(self.job) keeps it as self.job, and plugin.execute() runs on that.
    assert where["ddmErrorCode"] == [
        ("Plugin.execute", _line(PLUGIN, "ddmErrorCode = 2"))
    ]
    # [self.job] -> jobs -> enumerate(jobs) -> archiveJob(job)
    assert where["modificationTime"] == [
        ("DBProxy.archiveJob", _line(PROXY, "modificationTime = 1"))
    ]
    assert "PandaID" not in where


def test_only_the_holes_the_object_reaches_count(tmp_path):
    """``other.execute(self.job)`` is the object's hole; ``other.execute(5)`` is not, and ``len`` is outside."""
    rows = _adder(tmp_path)
    found = objects.follow(rows, "pkg/adder.py::Adder.process", "self.job")

    lines = sorted({site.line for site in found.holes})
    assert lines == [
        _line(ADDER, "other.execute(self.job)"),
        _line(ADDER, "threading.Thread("),
    ]
    (thread,) = [
        site for site in found.holes if site.line == _line(ADDER, "threading.Thread(")
    ]
    assert thread.how == panda_objects.HANDED_WITH_A_CALLABLE
    # The closure's own holes include the unrelated call; the object's do not.
    assert _line(ADDER, "other.execute(5)") in {
        hole.line for hole in rows["pkg/adder.py::Adder.process"].holes
    }


def test_an_object_returned_by_a_call_is_followed_into_where_it_was_made(tmp_path):
    """``self.job = self.tb.peekJobs([1])[0]``: peekJob made it, packed it from the row and set jobMetrics."""
    rows = _adder(tmp_path)
    found = objects.follow(rows, "pkg/adder.py::Adder.run", "self.job")

    where = {
        attr: sorted({(site.owner.split("::")[1], site.how) for site in sites})
        for attr, sites in found.assigned.items()
    }
    assert ("DBProxy.peekJob", "=") in where["jobMetrics"]
    # pack sets every column from the row: a write of the row's own values, said as such.
    assert ("JobSpec.pack", "row_read") in where["PandaID"]
    assert ("JobSpec.__init__", "construct") in where["PandaID"]
    # and it goes on into process as before
    assert ("Plugin.execute", "=") in where["ddmErrorCode"]
    came = {(site.owner.split("::")[1], site.how) for site in found.came_from}
    assert {
        ("Adder.run", "self.job"),
        ("TaskBuffer.peekJobs", "found[]"),
        ("DBProxy.peekJob", "job"),
    } <= came


def test_the_records_are_paths_of_the_function(tmp_path):
    """The receiver of a placed call is bound as ``self``; a module's names are not paths."""
    rows = _adder(tmp_path)
    process = rows["pkg/adder.py::Adder.process"]
    finish = [c for c in process.call_bindings if c.to == "pkg/adder.py::Adder.finish"]
    assert [(c.parameter, c.path) for c in finish] == [("self", "self")]
    updated = [
        c
        for c in process.call_bindings
        if c.to == "pkg/proxy.py::TaskBuffer.updateJobs"
    ]
    assert {(c.parameter, c.path, c.element) for c in updated} == {
        ("jobs", "self.job", True),
        ("self", "self.tb", False),
    }
    assert not any(c.path.startswith("threading") for c in process.call_bindings)


def test_the_gate_holds_every_record_to_its_function_and_its_rows(tmp_path):
    """object-records-resolve passes on what the build writes and fails on a call binding into nothing."""
    fragment = _build(tmp_path, spec=SPEC, proxy=PROXY, plugin=PLUGIN, adder=ADDER)
    result = object_records_resolve(fragment)
    assert result.passed, result.failures
    assert result.checked > 0

    process = next(
        row
        for row in fragment.reach_graph
        if row.owner == "pkg/adder.py::Adder.process"
    )
    process.call_bindings.append(
        CallBinding(
            line=process.anchor.line_start,
            to="pkg/gone.py::gone",
            parameter="x",
            path="self.job",
        )
    )
    process.call_bindings.append(
        CallBinding(
            line=process.anchor.line_start, to="", parameter="", path="self.job"
        )
    )
    failures = object_records_resolve(fragment).failures
    assert any("pkg/gone.py::gone, which has no reach row" in one for one in failures)
    assert any("a hole with no reason" in one for one in failures)


def test_a_class_held_in_a_field_builds_an_object_it_has_no_receiver(tmp_path):
    """``self.plugin_class(self.job)`` builds the plugin: its ``self`` is the result, not the caller's ``self``."""
    adder = ADDER.replace(
        "        plugin = Plugin(self.job)\n",
        "        self.plugin_class = Plugin\n        plugin = self.plugin_class(self.job)\n",
    )
    fragment = _build(tmp_path, spec=SPEC, proxy=PROXY, plugin=PLUGIN, adder=adder)
    rows = {row.owner: row for row in fragment.reach_graph}
    line = _line(adder, "plugin = self.plugin_class(self.job)")
    built = [
        c for c in rows["pkg/adder.py::Adder.process"].call_bindings if c.line == line
    ]
    assert {(c.to, c.parameter, c.path) for c in built} == {
        ("pkg/plugin.py::Plugin.__init__", "job", "self.job")
    }
    found = objects.follow(rows, "pkg/adder.py::Adder.process", "self.job")
    assert "pkg/plugin.py::Plugin.execute" in found.functions


def test_a_column_changes_by_the_object_written_back_or_by_a_key_of_its_row(tmp_path):
    """close(): modificationTime assigned and written by archiveJob; jobMetrics written by killJob on self.job_id's row."""
    rows = _adder(tmp_path)
    close = "pkg/adder.py::Adder.close"

    written = objects.may_change(rows, close, "self.job", "modificationTime")
    assert written.verdict is True
    assert {site.owner.split("::")[1] for site in written.because} == {
        "Adder.close",
        "DBProxy.archiveJob",
    }
    assert "written_back_as_read" in written.premises

    by_key = objects.may_change(
        rows, close, "self.job", "jobMetrics", keys=("self.job_id",)
    )
    assert by_key.verdict is True
    assert [site.owner.split("::")[1] for site in by_key.because] == ["DBProxy.killJob"]

    # The key not given: killJob's row is named by panda_id, not followed to the job's.
    unkeyed = objects.may_change(rows, close, "self.job", "jobMetrics")
    assert unkeyed.verdict is None
    assert [site.how for site in unkeyed.open] == [
        objects.NOT_FOLLOWED_KEY.format(column="jobMetrics", key="panda_id")
    ]
    # insertNewJob's row is new: the sequence names it.
    assert not any(
        "insertNewJob" in site.owner for site in [*unkeyed.open, *unkeyed.because]
    )


def test_a_column_assigned_and_never_written_does_not_change_the_row(tmp_path):
    """No statement of close()'s closure writes PandaID on the job's row: killJob names it in WHERE, insertNewJob's row is new."""
    rows = _adder(tmp_path)
    answer = objects.may_change(
        rows, "pkg/adder.py::Adder.close", "self.job", "PandaID"
    )
    assert answer.verdict is False
    assert answer.open == [] and answer.because == []


def test_the_objects_own_key_is_followed_as_a_value(tmp_path):
    """``self.tb.killJobs([self.job.PandaID])``: the job's key, handed as a value, names killJob's row."""
    adder = ADDER.replace(
        "        self.tb.killJobs([self.job_id])\n",
        "        self.tb.killJobs([self.job.PandaID])\n",
    )
    fragment = _build(tmp_path, spec=SPEC, proxy=PROXY, plugin=PLUGIN, adder=adder)
    rows = {row.owner: row for row in fragment.reach_graph}
    answer = objects.may_change(
        rows, "pkg/adder.py::Adder.close", "self.job", "jobMetrics"
    )
    assert answer.verdict is True
    assert [site.owner.split("::")[1] for site in answer.because] == ["DBProxy.killJob"]


def test_a_whole_row_write_of_the_object_takes_what_was_assigned_on_it(tmp_path):
    """process(): finish() assigns jobMetrics, and archiveJob writes the whole job (``job.valuesMap()``)."""
    rows = _adder(tmp_path)
    archive = rows["pkg/proxy.py::DBProxy.archiveJob"]
    (whole,) = [w for w in archive.writes if w.table == "jobsArchived4"]
    assert (whole.whole, whole.new_row) == ("job", False)
    (new,) = rows["pkg/proxy.py::DBProxy.insertNewJob"].writes
    assert (new.whole, new.new_row) == ("job", True)

    answer = objects.may_change(
        rows, "pkg/adder.py::Adder.process", "self.job", "jobMetrics"
    )
    assert answer.verdict is True
    owners = {site.owner.split("::")[1] for site in answer.because}
    assert owners == {"Adder.finish", "DBProxy.archiveJob"}


def test_a_map_bound_again_for_each_statement_is_the_one_that_reaches_the_call(
    tmp_path,
):
    """``varMap = file.valuesMap(onlyChanged=True)`` is one statement's map, not every statement's that uses the name."""
    proxy = (
        PROXY
        + """

class Files:
    def update(self, job: JobSpec, file):
        varMap = file.valuesMap(onlyChanged=True)
        self.cur.execute("UPDATE ATLAS_PANDA.filesTable4 SET status=:status WHERE row_ID=:row_ID", varMap)
        varMap = {}
        varMap[":PandaID"] = job.PandaID
        varMap[":modificationTime"] = job.modificationTime
        self.cur.execute("UPDATE ATLAS_PANDA.metaTable SET modificationTime=:modificationTime WHERE PandaID=:PandaID", varMap)
"""
    )
    fragment = _build(tmp_path, spec=SPEC, proxy=proxy, plugin=PLUGIN, adder=ADDER)
    rows = {row.owner: row for row in fragment.reach_graph}
    writes = {w.table: w for w in rows["pkg/proxy.py::Files.update"].writes}
    assert (writes["filesTable4"].whole, writes["filesTable4"].changed) == (
        "file",
        True,
    )
    assert writes["metaTable"].whole == ""
    assert [(v.column, v.path) for v in writes["metaTable"].columns] == [
        ("modificationTime", "job.modificationTime")
    ]


def test_a_write_of_a_table_of_another_class_is_not_the_objects_row(tmp_path):
    """A whole row of a file on filesTable4 writes every FileSpec column, and none of the job's."""
    proxy = PROXY.replace(
        """    def killJob(self, panda_id):""",
        """    def touchFiles(self, job: JobSpec, file):
        self.cur.execute("INSERT INTO ATLAS_PANDA.filesTable4 (PandaID) VALUES (:PandaID)", file.valuesMap())

    def killJob(self, panda_id):""",
    )
    adder = ADDER.replace(
        "        self.tb.storeJobs([JobSpec()])\n",
        "        self.tb.storeJobs([JobSpec()])\n        self.tb.proxy.touchFiles(self.job, None)\n",
    )
    fragment = _build(tmp_path, spec=SPEC, proxy=proxy, plugin=PLUGIN, adder=adder)
    rows = {row.owner: row for row in fragment.reach_graph}
    close = "pkg/adder.py::Adder.close"
    loose = objects.may_change(rows, close, "self.job", "PandaID")
    assert any("touchFiles" in site.owner for site in loose.open)
    kept = objects.may_change(
        rows,
        close,
        "self.job",
        "PandaID",
        spec_class="JobSpec",
        table_classes={"filestable4": "FileSpec"},
    )
    assert not any("touchFiles" in site.owner for site in [*kept.open, *kept.because])
