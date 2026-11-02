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
        from pandaserver.taskbuffer.JobSpec import JobSpec


        class TaskModule:
            def refineTask(self, jediTaskID):
                sql = "SELECT taskParams FROM ATLAS_PANDA.JEDI_TaskParams WHERE jediTaskID=:jediTaskID"
                self.cur.execute(sql, {":jediTaskID": jediTaskID})  # R1
                (taskParams,) = self.cur.fetchone()
                self.checkParams(taskParams)
                varMap = {}
                varMap[":jediTaskID"] = jediTaskID
                varMap[":taskParams"] = taskParams
                sql = "UPDATE ATLAS_PANDA.JEDI_TaskParams SET taskParams=:taskParams WHERE jediTaskID=:jediTaskID"
                self.cur.execute(sql, varMap)  # W1

            def checkParams(self, taskParams):
                if "reqID" in taskParams:
                    self.a = 1  # G1

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
        """,
}

ARMS = (("checkParams", "G1"), ("retryTask", "G2"), ("keepJob", "G3"))


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
        guards, parameters, sets = heldfrom.guard_froms(
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
