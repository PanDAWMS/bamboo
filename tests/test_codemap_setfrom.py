"""What each write gives a column a guard's way ends at -- ``SetFrom``, made by ``sqlwrite.extract`` and read by
``heldfrom.guard_froms`` -- on a proxy shaped as PanDA's: a task's parameters read and written back, a task's
counters, a job's status flushed from its spec."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from bamboo.codemap import calls as calls_mod
from bamboo.codemap import reach
from bamboo.codemap.gitsource import blob_sha
from bamboo.codemap.models import (
    Anchor,
    Branch,
    JunctionNode,
    MapFragment,
    SetFrom,
    SourceModule,
)
from bamboo.codemap.panda import attribution, heldfrom, sql
from bamboo.codemap.panda.plugin import _DatabaseLeaves
from bamboo.codemap.panda.recognizers import progress, sqlwrite

SPEC = "pandaserver/taskbuffer/JobSpec.py"
PROXY = "pandaserver/taskbuffer/db_proxy_mods/task_module.py"

SOURCES = {
    SPEC: """\
        class JobSpec:
            _attributes = ("PandaID", "jobStatus", "ddmErrorCode", "modificationTime")

            def __init__(self):
                for attr in self._attributes:
                    object.__setattr__(self, attr, None)

            def valuesMap(self, onlyChanged=False):
                ret = {}
                for attr in self._attributes:
                    ret[f":{attr}"] = getattr(self, attr)
                return ret

            def bindUpdateChangesExpression(self):
                ret = ""
                for attr in self._attributes:
                    ret += f"{attr}=:{attr},"
                return ret[:-1] + " "
        """,
    PROXY: """\
        import json

        from pandaserver.taskbuffer.JobSpec import JobSpec


        class TaskModule:
            def refineTask(self, jediTaskID):
                sql = "SELECT taskParams FROM ATLAS_PANDA.JEDI_TaskParams WHERE jediTaskID=:jediTaskID"
                self.cur.execute(sql, {":jediTaskID": jediTaskID})  # R1
                (text,) = self.cur.fetchone()
                taskParams = json.loads(text)
                self.checkParams(taskParams)
                taskParams["checked"] = True
                varMap = {}
                varMap[":jediTaskID"] = jediTaskID
                varMap[":taskParams"] = json.dumps(taskParams)
                sql = "UPDATE ATLAS_PANDA.JEDI_TaskParams SET taskParams=:taskParams WHERE jediTaskID=:jediTaskID"
                self.cur.execute(sql, varMap)  # W1
                taskParams["late"] = True

            def checkParams(self, taskParams):
                if "reqID" in taskParams:
                    self.a = 1  # G1
                taskParams["reqID"] = 0

            def retryTask(self, jediTaskID, increase):
                sql = "SELECT attemptNr,status,oldStatus FROM ATLAS_PANDA.JEDI_Tasks WHERE jediTaskID=:jediTaskID"
                self.cur.execute(sql, {":jediTaskID": jediTaskID})
                attemptNr, status, oldStatus = self.cur.fetchone()
                if attemptNr > 3 and status != "done" and oldStatus is None:
                    self.b = 1  # G2
                varMap = {}
                varMap[":jediTaskID"] = jediTaskID
                varMap[":increase"] = increase
                sql = (
                    "UPDATE ATLAS_PANDA.JEDI_Tasks SET attemptNr=attemptNr+:increase,status='ready',oldStatus=status,"
                    "modificationTime=CURRENT_DATE WHERE jediTaskID=:jediTaskID"
                )
                self.cur.execute(sql, varMap)  # W2

            def keepJob(self, pandaID):
                sql = "SELECT PandaID,jobStatus FROM ATLAS_PANDA.jobsActive4 WHERE PandaID=:PandaID"
                self.cur.execute(sql, {":PandaID": pandaID})  # R3
                res = self.cur.fetchone()
                jobStatus = res[1]
                if jobStatus == "running":
                    self.c = 1  # G3
                job = JobSpec()
                job.PandaID = pandaID
                job.jobStatus = jobStatus
                sql = f"UPDATE ATLAS_PANDA.jobsActive4 SET {job.bindUpdateChangesExpression()} WHERE PandaID=:PandaID"
                varMap = job.valuesMap(onlyChanged=True)
                self.cur.execute(sql, varMap)  # W3

            def touchTask(self, jediTaskID):
                varMap = {}
                varMap[":jediTaskID"] = jediTaskID
                sql = "UPDATE ATLAS_PANDA.JEDI_Tasks SET status=:status WHERE jediTaskID=:jediTaskID"
                self.cur.execute(sql, varMap)  # W4

            def getParams(self, jediTaskID):
                sql = "SELECT taskParams FROM ATLAS_PANDA.JEDI_TaskParams WHERE jediTaskID=:jediTaskID"
                self.cur.execute(sql, {":jediTaskID": jediTaskID})
                (text,) = self.cur.fetchone()
                return json.loads(text)

            def guardParams(self, jediTaskID):
                params = self.getParams(jediTaskID)
                if "prodSourceLabel" in params:
                    self.d = 1  # G5

            def saveParams(self, jediTaskID):
                params = self.getParams(jediTaskID)
                varMap = {":jediTaskID": jediTaskID, ":taskParams": json.dumps(params)}
                sql = "UPDATE ATLAS_PANDA.JEDI_TaskParams SET taskParams=:taskParams WHERE jediTaskID=:jediTaskID"
                self.cur.execute(sql, varMap)  # W5

            def loopParams(self, ids):
                sql = "SELECT taskParams FROM ATLAS_PANDA.JEDI_TaskParams WHERE jediTaskID=:jediTaskID"
                write = "UPDATE ATLAS_PANDA.JEDI_TaskParams SET taskParams=:taskParams WHERE jediTaskID=:jediTaskID"
                for jediTaskID in ids:
                    self.cur.execute(sql, {":jediTaskID": jediTaskID})
                    (text,) = self.cur.fetchone()
                    taskParams = json.loads(text)
                    if "reqID" in taskParams:
                        self.e = 1  # G6
                    taskParams["seen"] = True
                    varMap = {":jediTaskID": jediTaskID, ":taskParams": taskParams}
                    self.cur.execute(write, varMap)  # W6
                    taskParams["late"] = True

            def batchParams(self, jediTaskID, ids):
                sql = "SELECT taskParams FROM ATLAS_PANDA.JEDI_TaskParams WHERE jediTaskID=:jediTaskID"
                self.cur.execute(sql, {":jediTaskID": jediTaskID})
                (text,) = self.cur.fetchone()
                taskParams = json.loads(text)
                if "reqID" in taskParams:
                    self.f = 1  # G7
                write = "UPDATE ATLAS_PANDA.JEDI_TaskParams SET taskParams=:taskParams WHERE jediTaskID=:jediTaskID"
                for one in ids:
                    taskParams["n"] = one
                    varMap = {":jediTaskID": one, ":taskParams": taskParams}
                    self.cur.execute(write, varMap)  # W7

            def refineSpec(self, jediTaskID):
                sql = "SELECT taskParams FROM ATLAS_PANDA.JEDI_TaskParams WHERE jediTaskID=:jediTaskID"
                self.cur.execute(sql, {":jediTaskID": jediTaskID})
                (text,) = self.cur.fetchone()
                taskParams = json.loads(text)
                self.checkSpec(taskParams)
                varMap = {}
                varMap[":jediTaskID"] = jediTaskID
                varMap[":taskParams"] = json.dumps(taskParams)
                sql = "UPDATE ATLAS_PANDA.JEDI_TaskParams SET taskParams=:taskParams WHERE jediTaskID=:jediTaskID"
                self.cur.execute(sql, varMap)  # W8

            def checkSpec(self, taskParams):
                spec = {"status": taskParams["status"]}
                if spec["status"] == "running":
                    self.g = 1  # G8
                spec["status"] = "done"
                taskParams["specChecked"] = True

            def statusJob(self, pandaID):
                sql = "SELECT jobStatus FROM ATLAS_PANDA.jobsActive4 WHERE PandaID=:PandaID"
                self.cur.execute(sql, {":PandaID": pandaID})
                (jobStatus,) = self.cur.fetchone()
                self.checkStatus(pandaID, jobStatus)

            def checkStatus(self, pandaID, jobStatus):
                if jobStatus == "running":
                    self.h = 1  # G9
                varMap = {":PandaID": pandaID, ":jobStatus": jobStatus}
                sql = "UPDATE ATLAS_PANDA.jobsActive4 SET jobStatus=:jobStatus WHERE PandaID=:PandaID"
                self.cur.execute(sql, varMap)  # W9

            def refinePart(self, jediTaskID):
                sql = "SELECT taskParams FROM ATLAS_PANDA.JEDI_TaskParams WHERE jediTaskID=:jediTaskID"
                self.cur.execute(sql, {":jediTaskID": jediTaskID})
                (text,) = self.cur.fetchone()
                taskParams = json.loads(text)
                self.checkPart(taskParams)
                varMap = {}
                varMap[":jediTaskID"] = jediTaskID
                varMap[":taskParams"] = json.dumps(taskParams)
                sql = "UPDATE ATLAS_PANDA.JEDI_TaskParams SET taskParams=:taskParams WHERE jediTaskID=:jediTaskID"
                self.cur.execute(sql, varMap)  # W10

            def checkPart(self, taskParams):
                job = taskParams["job"]
                if job["type"] == "template":
                    self.k = 1  # G10
                job["type"] = "done"
                taskParams["partChecked"] = True
        """,
}

ARMS = (
    ("checkParams", "G1"),
    ("retryTask", "G2"),
    ("keepJob", "G3"),
    ("guardParams", "G5"),
    ("loopParams", "G6"),
    ("batchParams", "G7"),
    ("checkSpec", "G8"),
    ("checkStatus", "G9"),
    ("checkPart", "G10"),
)


def _line(marker: str) -> int:
    (found,) = [
        number
        for number, held in enumerate(textwrap.dedent(SOURCES[PROXY]).splitlines(), 1)
        if held.rstrip().endswith(f"# {marker}")
    ]
    return found


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> MapFragment:
    """The proxy read as the build does: the writes' records made in the SQL slice, read with the guards'."""
    tmp_path: Path = tmp_path_factory.mktemp("setfrom")
    modules = []
    for rel, text in SOURCES.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text))
        parsed = calls_mod.parsed(path)
        assert parsed is not None
        modules.append(
            SourceModule(
                package="pandaserver",
                rel_path=rel,
                tree=parsed[1],
                source=textwrap.dedent(text),
                blob_sha=blob_sha(text),
            )
        )
    calls = sqlwrite.corpus_calls(modules, {"pandaserver": tmp_path / "pandaserver"})
    assert calls is not None
    junctions = [
        JunctionNode(
            name=f"m:{owner}:{marker}",
            map_id="m",
            derived_from="v",
            subject="JobSpec.jobStatus",
            owner=f"{PROXY}::{owner}",
            anchor=Anchor(package="pandaserver", file=PROXY, line_start=_line(marker)),
            branches=[Branch(outcome=marker, line=_line(marker))],
        )
        for owner, marker in ARMS
    ]
    writes: list[SetFrom] = []
    with sql.across_calls(calls):
        attributor = attribution.SpecAttributor(
            progress.spec_attributes(modules), attribution.class_bases(modules)
        )
        attributor.learn_table_classes(modules)
        sqlwrite.extract(modules, "m", "v", attributor, key_froms=[], set_froms=writes)
        sites = sqlwrite.read_site_nodes(
            sqlwrite.read_side(modules, attributor, calls).statements, "m", "v"
        )
        record = reach.ReachGraph(reach.graph(calls, "m", "v", _DatabaseLeaves()))
        guards, parameters, sets, _written = heldfrom.guard_froms(
            junctions,
            sites,
            {module.rel_path: module.tree for module in modules},
            calls,
            record,
            writes,
        )
    return MapFragment(
        map_id="m",
        derived_from="v",
        junctions=junctions,
        guard_froms=guards,
        parameter_froms=parameters,
        set_froms=sets,
    )


def _sets(fragment: MapFragment, owner: str) -> dict[str, SetFrom]:
    found = [one for one in fragment.set_froms if one.owner == f"{PROXY}::{owner}"]
    assert len({one.column for one in found}) == len(found)
    return {one.column: one for one in found}


def _ends(sources) -> set[tuple[str, int, str]]:
    return {
        (one.site_owner, one.site_line, one.column)
        for one in sources
        if one.kind == "rows"
    }


def test_a_column_written_back_from_what_was_read_has_its_way_end_at_the_read(
    built: MapFragment,
) -> None:
    """refineTask reads the task's parameters, hands them to checkParams' guard and writes them back: the write's
    way ends at the statement and column the guard's way ends at, through the parameter."""
    (written,) = _sets(built, "refineTask").values()
    assert (written.column, written.value, written.text, written.new_row) == (
        "jedi_taskparams.taskparams",
        "bind",
        ":taskParams",
        False,
    )
    read = (f"{PROXY}::refineTask", _line("R1"), "jedi_taskparams.taskparams")
    assert _ends(written.sources) == {read}
    (guard,) = [one for one in built.guard_froms if one.owner.endswith("::checkParams")]
    (handed,) = guard.sources
    assert handed.kind == "parameter"
    (parameter,) = [
        one for one in built.parameter_froms if one.owner.endswith("::checkParams")
    ]
    assert _ends(parameter.sources) == {read}


def test_each_value_is_said_as_written_and_only_binds_are_read(
    built: MapFragment,
) -> None:
    """A counter of a bind, a literal and a column, each a column the guard reads; modificationTime no guard reads
    has no record."""
    found = _sets(built, "retryTask")
    assert {column: (one.value, one.text) for column, one in found.items()} == {
        "jedi_tasks.attemptnr": ("expression", "attemptNr+:increase"),
        "jedi_tasks.status": ("literal", "ready"),
        "jedi_tasks.oldstatus": ("column", "status"),
    }
    assert [(one.kind, one.name) for one in found["jedi_tasks.attemptnr"].sources] == [
        ("parameter", "increase")
    ]
    assert (
        not found["jedi_tasks.status"].sources and not found["jedi_tasks.status"].stops
    )
    assert (
        not found["jedi_tasks.oldstatus"].sources
        and not found["jedi_tasks.oldstatus"].stops
    )


def test_an_object_write_gives_the_field_as_the_function_assigns_it(
    built: MapFragment,
) -> None:
    """keepJob flushes the job it filled from the row it read: jobStatus is the guard's column, its way the
    assignment's; PandaID no guard reads has no record."""
    found = _sets(built, "keepJob")
    assert set(found) == {"jobsactive4.jobstatus"}
    written = found["jobsactive4.jobstatus"]
    assert (written.value, written.text, written.new_row) == (
        "object",
        "job.jobStatus",
        False,
    )
    assert _ends(written.sources) == {
        (f"{PROXY}::keepJob", _line("R3"), "jobsactive4.jobstatus")
    }


def test_a_bind_no_assignment_fills_is_a_stop(built: MapFragment) -> None:
    (written,) = _sets(built, "touchTask").values()
    assert (written.column, written.value) == ("jedi_tasks.status", "bind")
    assert not written.sources
    assert [(one.kind, one.name) for one in written.stops] == [("no_bind", ":status")]


def test_the_changes_after_the_read_are_the_ones_before_the_write(
    built: MapFragment,
) -> None:
    """checkParams stores reqID after its guard reads the parameters; refineTask marks them checked before it writes
    them back, and late after: the first two, each before the write, in the order they run."""
    (guard,) = [one for one in built.guard_froms if one.owner.endswith("::checkParams")]
    (way,) = guard.sources
    ref = f"{PROXY}::refineTask:{_line('W1')}:jedi_taskparams.taskparams"
    assert [
        (one.kind, one.key, one.name, one.before, one.either) for one in way.after
    ] == [
        ("store", "'reqID'", "taskParams", [ref], []),
        ("store", "'checked'", "taskParams", [ref], []),
    ]
    assert not way.after_stops


def test_the_changes_after_the_read_are_of_the_object_the_write_gives_back(
    built: MapFragment,
) -> None:
    """checkPart's guard reads an item of the task's parameters: its store on the item and its store on the
    parameters, each with the name it is read on, the item's key the guard's value's.  checkSpec's guard reads a map
    built of a part of the parameters: its store on that map is not one the write gives back, its store on the
    parameters is.  checkStatus's and keepJob's guards read a job's status no way decodes: a value not changed in
    place, nothing after it -- not the map it is put in."""
    (guard,) = [one for one in built.guard_froms if one.owner.endswith("::checkPart")]
    (way,) = guard.sources
    assert (way.kind, way.name, way.steps, way.paths) == (
        "parameter",
        "taskParams",
        ["['job']"],
        [],
    )
    ref = f"{PROXY}::refinePart:{_line('W10')}:jedi_taskparams.taskparams"
    assert [(one.kind, one.key, one.name, one.before) for one in way.after] == [
        ("store", "'type'", "job", [ref]),
        ("store", "'partChecked'", "taskParams", [ref]),
    ]
    (guard,) = [one for one in built.guard_froms if one.owner.endswith("::checkSpec")]
    (way,) = guard.sources
    assert (way.kind, way.name, way.paths) == (
        "parameter",
        "taskParams",
        ["['status']"],
    )
    ref = f"{PROXY}::refineSpec:{_line('W8')}:jedi_taskparams.taskparams"
    assert [
        (one.kind, one.key, one.name, one.before, one.either) for one in way.after
    ] == [
        ("store", "'specChecked'", "taskParams", [ref], []),
    ]
    assert not way.after_stops
    for owner in ("checkStatus", "keepJob"):
        (guard,) = [
            one for one in built.guard_froms if one.owner.endswith(f"::{owner}")
        ]
        (way,) = guard.sources
        assert not way.after and not way.after_stops


def test_a_write_in_another_call_of_the_reader_meets_the_guard_nowhere(
    built: MapFragment,
) -> None:
    """guardParams and saveParams each call getParams: no one call reads for the guard and writes back."""
    (guard,) = [one for one in built.guard_froms if one.owner.endswith("::guardParams")]
    (way,) = guard.sources
    assert not way.after
    assert [(one.kind, one.name) for one in way.after_stops] == [
        ("no_meeting", f"{PROXY}::saveParams:{_line('W5')}:jedi_taskparams.taskparams")
    ]


def test_a_change_runs_after_the_write_too_only_where_the_loop_keeps_the_object(
    built: MapFragment,
) -> None:
    """loopParams reads each task's parameters anew on each pass: seen, and the map the write is handed (``*``, put
    in ``varMap``), run before its write only, late after it only.  batchParams reads them once and writes them on
    each pass, setting n and the map before: both also run after a write."""
    (guard,) = [one for one in built.guard_froms if one.owner.endswith("::loopParams")]
    (way,) = guard.sources
    ref = f"{PROXY}::loopParams:{_line('W6')}:jedi_taskparams.taskparams"
    assert [(one.kind, one.key, one.before, one.either) for one in way.after] == [
        ("store", "'seen'", [ref], []),
        ("escaped", "*", [ref], []),
    ]
    (guard,) = [one for one in built.guard_froms if one.owner.endswith("::batchParams")]
    (way,) = guard.sources
    ref = f"{PROXY}::batchParams:{_line('W7')}:jedi_taskparams.taskparams"
    assert [(one.kind, one.key, one.before, one.either) for one in way.after] == [
        ("store", "'n'", [ref], [ref]),
        ("escaped", "*", [ref], [ref]),
    ]
