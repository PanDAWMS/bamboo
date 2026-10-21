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
    assert {(c.parameter, c.path, c.at) for c in updated} == {
        ("jobs", "self.job", "[]"),
        ("self", "self.tb", ""),
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


def test_a_map_bound_again_with_an_annotation_is_bound_again(tmp_path):
    """``varMap: dict = {}`` binds the map again as ``varMap = {}`` does (retryJob): round the loop the file's map does not reach."""
    proxy = (
        PROXY
        + """

class Retry:
    def retry(self, job: JobSpec, file):
        for _ in range(3):
            varMap: dict = {}
            varMap[":PandaID"] = job.PandaID
            varMap[":modificationTime"] = job.modificationTime
            self.cur.execute("UPDATE ATLAS_PANDA.metaTable SET modificationTime=:modificationTime WHERE PandaID=:PandaID", varMap)
            varMap = file.valuesMap(onlyChanged=True)
            self.cur.execute("UPDATE ATLAS_PANDA.filesTable4 SET status=:status WHERE row_ID=:row_ID", varMap)
"""
    )
    fragment = _build(tmp_path, spec=SPEC, proxy=proxy, plugin=PLUGIN, adder=ADDER)
    rows = {row.owner: row for row in fragment.reach_graph}
    writes = {w.table: w for w in rows["pkg/proxy.py::Retry.retry"].writes}
    assert writes["filesTable4"].whole == "file"
    assert writes["metaTable"].whole == ""
    assert [(v.column, v.path) for v in writes["metaTable"].keys] == [
        ("PandaID", "job.PandaID")
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


def test_a_slice_holds_the_containers_elements_not_the_container(tmp_path):
    """setupper's ``jobs_list = self.jobs`` or ``self.jobs[a:b]``: jobs_list's elements are self.jobs's, never a job.

    A slice is a new list of the container's elements, at positions not
    known -- ``self.jobs[1:][0]`` is not ``self.jobs[0]``.
    """
    batch = """\
from pkg.spec import JobSpec


class Batch:
    def __init__(self, jobs: list[JobSpec]):
        self.jobs = jobs

    def run(self, start, n):
        if n is None:
            jobs_list = self.jobs
        else:
            jobs_list = self.jobs[start : start + n]
        for job in jobs_list:
            job.jobStatus = "failed"
        rest = self.split()[1:]
        return rest

    def split(self):
        return [self.jobs]
"""
    fragment = _build(tmp_path, spec=SPEC, batch=batch)
    rows = {row.owner: row for row in fragment.reach_graph}
    run = rows["pkg/batch.py::Batch.run"]
    # Bound at two statements, the name is two: jobs_list@10 and jobs_list@12.
    assert {
        (b.name, b.path) for b in run.bindings if b.name.startswith("jobs_list")
    } == {("jobs_list@10", "self.jobs"), ("jobs_list@12[]", "self.jobs[]")}
    assert [(b.name, b.path) for b in run.bindings if b.name.startswith("rest")] == [
        ("rest[]", "[]")
    ]
    container = objects.follow(rows, "pkg/batch.py::Batch.run", "self.jobs")
    assert "job" not in container.held["pkg/batch.py::Batch.run"]
    assert "jobStatus" not in container.assigned
    element = objects.follow(rows, "pkg/batch.py::Batch.run", "self.jobs[]")
    assert "job" in element.held["pkg/batch.py::Batch.run"]
    assert "jobStatus" in element.assigned


def test_a_name_bound_in_two_places_is_two_names_and_a_field_is_read_back_only_by_the_binding_the_object_came_by(
    tmp_path,
):
    """setupper's ``tag_job = self.jobs[0]`` / ``tag_job = self.jumbo_jobs[0]``: a job of one list is not one of the other.

    A local name bound at two statements is a name for each, and a use is
    the ones that reach it; a field is one name however often it is bound,
    and is read back only by the binding the object came in by.
    """
    batch = """\
from pkg.spec import JobSpec


class Batch:
    def __init__(self, jobs: list[JobSpec], jumbo: list[JobSpec]):
        self.jobs = jobs
        self.jumbo_jobs = jumbo

    def run(self):
        tag_job = None
        if self.jobs:
            tag_job = self.jobs[0]
        else:
            tag_job = self.jumbo_jobs[0]
        tag_job.jobStatus = "failed"

    def mark(self):
        if self.jobs:
            self.tag = self.jobs[0]
        else:
            self.tag = self.jumbo_jobs[0]
        self.tag.jobStatus = "failed"
"""
    fragment = _build(tmp_path, spec=SPEC, batch=batch)
    rows = {row.owner: row for row in fragment.reach_graph}
    run = "pkg/batch.py::Batch.run"
    jobs = objects.follow(rows, run, "self.jobs[]")
    assert set(jobs.held[run]) == {"self.jobs[]", "tag_job@12"}
    assert "jobStatus" in jobs.assigned
    jumbo = objects.follow(rows, run, "self.jumbo_jobs[]")
    assert set(jumbo.held[run]) == {"self.jumbo_jobs[]", "tag_job@14"}
    # The use at 15 is reached by both, not by tag_job = None (no branch keeps it).
    assert {(a.path, a.line) for a in rows[run].assigns} == {
        ("tag_job@12", 15),
        ("tag_job@14", 15),
    }
    assert objects.steps_to(rows[run], "self.jobs[]", "tag_job@12") == [
        (12, "alias", "tag_job@12")
    ]
    mark = "pkg/batch.py::Batch.mark"
    jobs = objects.follow(rows, mark, "self.jobs[]")
    assert "self.tag" in jobs.held[mark]
    assert "self.jumbo_jobs[0]" not in jobs.held[mark]
    either = objects.follow(rows, mark, "self.tag")
    assert {"self.jobs[0]", "self.jumbo_jobs[0]"} <= set(either.held[mark])


def test_a_name_filled_and_bound_again_to_a_calls_result_is_not_followed_into_the_call(
    tmp_path,
):
    """makeFakeCoJumbo: ``varMap`` filled with the old job's PandaID, then bound to the new job's ``valuesMap()``."""
    spec = (
        SPEC
        + """
    def valuesMap(self):
        return [self.PandaID]
"""
    )
    maker = """\
from pkg.spec import JobSpec


class Maker:
    def make(self, old: JobSpec):
        var_map = {}
        var_map[":PandaID"] = old.PandaID
        new = JobSpec()
        var_map = new.valuesMap()
        return new
"""
    fragment = _build(tmp_path, spec=spec, maker=maker)
    rows = {row.owner: row for row in fragment.reach_graph}
    make = "pkg/maker.py::Maker.make"
    old = objects.follow(rows, make, "old.PandaID")
    assert "var_map@6[':PandaID']" in old.held[make]
    assert not any(one.startswith("var_map@9") for one in old.held[make])
    assert "new.PandaID" not in old.held[make]
    # The new job's own key still goes into its valuesMap and back.
    new = objects.follow(rows, make, "var_map@9[]")
    assert "new.PandaID" in new.held[make]


def test_a_path_handed_to_a_method_and_back_keeps_how_it_was_reached(tmp_path):
    """TaskRefiner: ``tmpTaskSpec = impl.taskSpec`` or a new spec given ``jediTaskID``, then ``tmpTaskSpec.setErrDiag()``."""
    refiner = """\
from pkg.spec import JobSpec


class Refiner:
    def run(self, impl, task_id):
        if impl is None:
            spec = JobSpec()
            spec.PandaID = task_id
        else:
            spec = impl.job
        spec.setErrDiag("x")
"""
    fragment = _build(tmp_path, spec=SPEC, refiner=refiner)
    rows = {row.owner: row for row in fragment.reach_graph}
    run = "pkg/refiner.py::Refiner.run"
    key = objects.follow(rows, run, "impl.job.PandaID")
    assert "spec@10.PandaID" in key.held[run]
    assert "spec@7.PandaID" not in key.held[run]
    assert "task_id" not in key.held[run]
    assert "spec@7.PandaID" in objects.follow(rows, run, "task_id").held[run]


def test_a_name_bound_again_is_split_unless_every_binding_spells_a_str_or_a_number(
    tmp_path,
):
    """``found += more`` keeps what found held and adds more's elements; ``sql``, only ever a str, is not split.

    A use after ``found += more`` is reached by both bindings (``+=`` kills
    nothing), so the new one holds only what it adds.
    """
    batch = """\
from pkg.spec import JobSpec


class Batch:
    def run(self, jobs: list[JobSpec], more: list[JobSpec]):
        found = jobs
        found += more
        sql = "SELECT PandaID FROM jobs"
        sql += " WHERE jobStatus=:status"
        for job in found:
            job.jobStatus = "failed"
        return sql
"""
    fragment = _build(tmp_path, spec=SPEC, batch=batch)
    rows = {row.owner: row for row in fragment.reach_graph}
    run = "pkg/batch.py::Batch.run"
    assert sorted((b.name, b.path, b.how) for b in rows[run].bindings) == [
        ("<return>", "sql", "return"),
        ("found@6", "jobs", "alias"),
        ("found@7[]", "more[]", "fill"),
        ("job", "found@6[]", "element"),
        ("job", "found@7[]", "element"),
    ]
    assert "jobStatus" in objects.follow(rows, run, "jobs[]").assigned
    assert "jobStatus" in objects.follow(rows, run, "more[]").assigned


def test_a_parameter_bound_again_keeps_its_name_only_where_its_entry_reaches(tmp_path):
    """``if x: job = other`` then ``job = other``: what a caller hands as ``job`` reaches the first use, not the second."""
    fixer = """\
from pkg.spec import JobSpec


class Fixer:
    def fix(self, job: JobSpec, other: JobSpec, x):
        if x:
            job = other
        job.jobStatus = "failed"
        job = other
        job.ddmErrorCode = 1

    def run(self, mine: JobSpec, theirs: JobSpec):
        self.fix(mine, theirs, True)
"""
    fragment = _build(tmp_path, spec=SPEC, fixer=fixer)
    rows = {row.owner: row for row in fragment.reach_graph}
    fix = "pkg/fixer.py::Fixer.fix"
    assert {(a.path, a.line) for a in rows[fix].assigns} == {
        ("job", 8),
        ("job@7", 8),
        ("job@9", 10),
    }
    handed = objects.follow(rows, "pkg/fixer.py::Fixer.run", "mine")
    assert "jobStatus" in handed.assigned
    assert "ddmErrorCode" not in handed.assigned
    other = objects.follow(rows, "pkg/fixer.py::Fixer.run", "theirs")
    assert {"jobStatus", "ddmErrorCode"} <= set(other.assigned)


def test_the_record_keeps_every_path_and_follow_says_where_its_cap_stops_it(
    tmp_path,
):
    """A file's owner is the job (JobSpec.addFile), so ``job.Files[]._owner`` names the job again, and again: the
    parts cap is what stops the cycle, and follow says where it did; may_change names ``follow_capped``.  The
    record keeps a path however long -- how far one is read is follow's."""
    owned = """\
class FileSpec:
    def setOwner(self, owner):
        self._owner = owner


class Job:
    def __init__(self):
        self.Files = []

    def addFile(self, file: FileSpec):
        file.setOwner(self)
        self.Files.append(file)


class Runner:
    def run(self, job: Job, f: FileSpec):
        job.addFile(f)
        deep = {"a": {"b": {"c": {"d": {"e": {"f": {"g": job}}}}}}}
        return deep

    def other(self, job: Job):
        job.jobStatus = "failed"
"""
    fragment = _build(tmp_path, spec=SPEC, owned=owned)
    rows = {row.owner: row for row in fragment.reach_graph}
    run = "pkg/owned.py::Runner.run"
    deep = "deep['a']['b']['c']['d']['e']['f']['g']"
    assert (deep, "job") in {(b.name, b.path) for b in rows[run].bindings}

    found = objects.follow(rows, run, "job")
    assert found.held[run] == [
        "job",
        "job.Files[]._owner",
        "job.Files[]._owner.Files[]._owner",
    ]
    spent = objects.PARTS_SPENT.format(path="")
    assert {
        (site.owner, site.line, site.how[len(spent) :]) for site in found.capped
    } == {
        (run, _line(owned, "deep = "), deep),
        (run, _line(owned, "deep = "), deep + ".Files[]._owner"),
        (run, _line(owned, "deep = "), deep + ".Files[]._owner.Files[]._owner"),
        (
            "pkg/owned.py::FileSpec.setOwner",
            _line(owned, "self._owner = owner"),
            "self._owner.Files[]._owner.Files[]._owner",
        ),
    }
    assert all(
        objects._parts(path) <= objects.MAX_PARTS
        for paths in found.held.values()
        for path in paths
    )
    assert "follow_capped" in objects.may_change(rows, run, "job", "jobStatus").premises
    plain = objects.may_change(rows, "pkg/owned.py::Runner.other", "job", "jobStatus")
    assert "follow_capped" not in plain.premises
    # Parts are counted as _under reads them: a literal key is one, whatever it holds.
    assert objects._parts("varMap[':a.b']") == 1


FORMS = """\
from typing import Iterator

from pkg.spec import JobSpec


class Store:
    def get(self, key):
        return key


class Mapper:
    def pick(self, job: JobSpec) -> JobSpec:
        return job


class Forms:
    def __init__(self):
        self.store: Store = Store()

    def make(self, spec: JobSpec) -> tuple[bool, JobSpec]:
        return True, spec

    def wrap(self, job: JobSpec) -> JobSpec:
        return job

    def handle(self, job: JobSpec):
        job.jobStatus = "handled"

    def mapper(self) -> Mapper:
        return Mapper()

    def each(self, jobs: list[JobSpec]) -> Iterator[JobSpec]:
        for one in jobs:
            yield one

    def keys(self, job: JobSpec, other, k):
        var_map = {":PandaID": job.PandaID, ":status": other}
        status = var_map[":status"]
        any_one = var_map[k]
        return status, any_one

    def unpack(self, spec: JobSpec):
        status, job = self.make(spec)
        job.jobMetrics = "made"
        return status

    def pairs(self, job: JobSpec, n):
        pairs = []
        pairs.append((job, n))
        for first, second in pairs:
            first.ddmErrorCode = 1
            second.modificationTime = 2

    def forms(self, job: JobSpec, other: JobSpec, jobs: list[JobSpec], c):
        either = job or other
        either.ddmErrorCode = 3
        maybe = job if c else None
        found = [one for one in jobs]
        for one in found:
            one.jobStatus = "found"
        last = jobs.pop()
        stored = self.store.get(1)
        return maybe, last, stored

    def plus(self, job: JobSpec, extra):
        for each in job.Files + extra:
            each.jobStatus = "file"

    def generated(self, jobs: list[JobSpec]):
        for one in self.each(jobs):
            one.jobMetrics = "each"

    def nested(self, job: JobSpec):
        self.handle(self.wrap(job))
        picked = self.mapper().pick(job)
        return picked

    def reordered(self, job: JobSpec, other, z):
        t = (job, other)
        t = list(t)
        t.insert(0, z)
        second = t[1]
        return second

    def columns(self, job: JobSpec):
        job.jobStatus, job.PandaID = "x", 1
"""

_FORMS = "pkg/forms.py::Forms"


def _forms(tmp_path: Path) -> tuple[MapFragment, dict]:
    fragment = _build(tmp_path, spec=SPEC, forms=FORMS)
    return fragment, {row.owner: row for row in fragment.reach_graph}


def _assigned(found: objects.ObjectReach) -> dict[str, set[str]]:
    return {
        attr: {site.owner.split("::")[1] for site in sites}
        for attr, sites in found.assigned.items()
    }


def test_a_dict_literal_keeps_each_value_under_its_key(tmp_path):
    """``{":PandaID": job.PandaID, ":status": other}``: the key is read at its own key, and at one computed."""
    _fragment, rows = _forms(tmp_path)
    owner = f"{_FORMS}.keys"
    key = objects.follow(rows, owner, "job.PandaID").held[owner]
    assert "var_map[':PandaID']" in key
    assert "any_one" in key
    assert "status" not in key
    other = objects.follow(rows, owner, "other").held[owner]
    assert {"status", "any_one"} <= set(other)


def test_a_tuple_target_takes_each_position_of_what_a_call_returns(tmp_path):
    """``status, job = self.make(spec)`` with ``return True, spec``: job is the spec, status is not."""
    _fragment, rows = _forms(tmp_path)
    owner = f"{_FORMS}.unpack"
    results = {
        (b.name, b.path, tuple(b.to)) for b in rows[owner].bindings if b.how == "result"
    }
    assert results == {
        ("status", "[0]", (f"{_FORMS}.make",)),
        ("job", "[1]", (f"{_FORMS}.make",)),
    }
    found = objects.follow(rows, owner, "spec")
    assert "job" in found.held[owner]
    assert "status" not in found.held[owner]
    assert _assigned(found)["jobMetrics"] == {"Forms.unpack"}


def test_a_tuple_put_in_a_list_keeps_its_positions(tmp_path):
    """``pairs.append((job, n))`` then ``for first, second in pairs``: first is the job, second is not."""
    _fragment, rows = _forms(tmp_path)
    found = _assigned(objects.follow(rows, f"{_FORMS}.pairs", "job"))
    assert "ddmErrorCode" in found
    assert "modificationTime" not in found


def test_or_if_else_a_comprehension_pop_and_plus_carry_what_they_hold(tmp_path):
    _fragment, rows = _forms(tmp_path)
    owner = f"{_FORMS}.forms"
    job = objects.follow(rows, owner, "job")
    assert "ddmErrorCode" in _assigned(job)  # either = job or other
    assert "maybe" in job.held[owner]  # job if c else None
    element = objects.follow(rows, owner, "jobs[]")
    assert "jobStatus" in _assigned(element)  # [one for one in jobs], then a loop
    assert "last" in element.held[owner]  # jobs.pop()
    files = objects.follow(rows, f"{_FORMS}.plus", "job.Files[]")
    assert "jobStatus" in _assigned(files)  # for each in job.Files + extra


def test_a_corpus_method_named_as_a_containers_is_read_as_its_call(tmp_path):
    """``self.store.get(1)`` runs ``Store.get``: its result, not a value of ``self.store`` under a key."""
    _fragment, rows = _forms(tmp_path)
    stored = [b for b in rows[f"{_FORMS}.forms"].bindings if b.name == "stored"]
    assert [(b.how, b.path, b.to) for b in stored] == [
        ("result", "", ["pkg/forms.py::Store.get"])
    ]


def test_what_a_generator_yields_are_the_elements_of_its_result(tmp_path):
    _fragment, rows = _forms(tmp_path)
    found = objects.follow(rows, f"{_FORMS}.generated", "jobs[]")
    assert _assigned(found)["jobMetrics"] == {"Forms.generated"}


def test_a_result_is_its_own_calls_and_one_no_name_takes_is_followed(tmp_path):
    """``self.handle(self.wrap(job))``: wrap's result goes into handle.  ``self.mapper().pick(job)``: picked is
    pick's result, not mapper's."""
    _fragment, rows = _forms(tmp_path)
    owner = f"{_FORMS}.nested"
    found = objects.follow(rows, owner, "job")
    assert _assigned(found)["jobStatus"] == {"Forms.handle"}
    (picked,) = [b for b in rows[owner].bindings if b.name == "picked"]
    assert picked.to == ["pkg/forms.py::Mapper.pick"]
    assert f"{_FORMS}.mapper" not in objects.follow(rows, owner, "picked").functions


def test_a_list_made_of_a_tuple_is_read_at_any_position(tmp_path):
    """``t = list(t)`` then ``t.insert(0, z)``: the job is no longer at [0], so ``t[1]`` can be it."""
    _fragment, rows = _forms(tmp_path)
    owner = f"{_FORMS}.reordered"
    assert "second" in objects.follow(rows, owner, "job").held[owner]


def test_a_tuple_targets_columns_are_assigned(tmp_path):
    _fragment, rows = _forms(tmp_path)
    assigns = {(a.path, *a.attrs) for a in rows[f"{_FORMS}.columns"].assigns}
    assert assigns == {("job", "jobStatus"), ("job", "PandaID")}


def test_the_gate_reads_keys_positions_and_a_calls_own_result(tmp_path):
    fragment, _rows = _forms(tmp_path)
    result = object_records_resolve(fragment)
    assert result.passed, result.failures
