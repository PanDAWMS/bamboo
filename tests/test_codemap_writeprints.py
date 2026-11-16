"""Where a write statement's values print: ``WritePrint``, made by ``sqlwrite.extract`` for each write the function
prints the text and the bind map of, its arms from the statement's ``KeyFrom.writes`` (``sqlwrite.attach_write_arms``),
and the gate that holds it to the print's rows."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from bamboo.codemap import calls as calls_mod
from bamboo.codemap import gates, reach, trace
from bamboo.codemap.gitsource import blob_sha
from bamboo.codemap.models import (
    Anchor,
    Branch,
    EntityNode,
    FunctionPrintsNode,
    JunctionNode,
    KeyFrom,
    MapFragment,
    PrintedBy,
    SourceModule,
    WriteBind,
    WritePrint,
)
from bamboo.codemap.objects import across_records
from bamboo.codemap.panda import attribution, sql
from bamboo.codemap.panda import objects as panda_objects
from bamboo.codemap.panda.plugin import _DatabaseLeaves
from bamboo.codemap.panda.recognizers import progress, sqlwrite

PACKAGE = "pandaserver"
SPEC = "pandaserver/taskbuffer/JobSpec.py"
ODD = "pandaserver/taskbuffer/OddSpec.py"
PROXY = "pandaserver/taskbuffer/db_proxy_mods/job_module.py"

SOURCES = {
    SPEC: """\
        class JobSpec:
            _attributes = ("PandaID", "jobStatus", "modificationTime")

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
    ODD: """\
        class OddSpec:
            _attributes = ("oddID", "state")

            def __init__(self):
                for attr in self._attributes:
                    object.__setattr__(self, attr, None)

            def valuesMap(self, onlyChanged=False):
                ret = {}
                for attr in self._attributes:
                    ret[attr.upper()] = getattr(self, attr)
                return ret

            def bindUpdateChangesExpression(self):
                ret = ""
                for attr in self._attributes:
                    ret += f"{attr}=:{attr.upper()},"
                return ret[:-1] + " "
        """,
    PROXY: """\
        import datetime
        import logging

        from pandaserver.taskbuffer.JobSpec import JobSpec
        from pandaserver.taskbuffer.OddSpec import OddSpec

        _logger = logging.getLogger("proxy")


        class JobModule:
            def updateJob(self, pandaID):
                comment = " /* DBProxy.updateJob */"
                job = JobSpec()
                job.PandaID = pandaID
                job.modificationTime = datetime.datetime.now()  # A1
                sql = f"UPDATE ATLAS_PANDA.jobsActive4 SET {job.bindUpdateChangesExpression()} WHERE PandaID=:PandaID"
                varMap = job.valuesMap(onlyChanged=True)
                varMap[":PandaID"] = job.PandaID
                _logger.debug(sql + comment + str(varMap))  # P1
                self.cur.execute(sql + comment, varMap)  # W1

            def updateStatus(self, jediTaskID, newStatus):
                varMap = {}
                varMap[":jediTaskID"] = jediTaskID
                varMap[":status"] = newStatus  # A2
                varMap[":updateTime"] = datetime.datetime.now()  # A3
                if newStatus == "ready":
                    varMap[":updateTime"] = datetime.datetime.now() - datetime.timedelta(hours=6)  # A4
                varMap[":inc"] = 1  # A5
                sql = (
                    "UPDATE ATLAS_PANDA.JEDI_Tasks SET status=:status,modificationTime=:updateTime,"
                    "attemptNr=attemptNr+:inc WHERE jediTaskID=:jediTaskID"
                )
                _logger.debug("running sql: " + sql + " " + str(varMap))  # P2
                self.cur.execute(sql, varMap)  # W2

            def notPrinted(self, jediTaskID, newStatus):
                varMap = {":jediTaskID": jediTaskID, ":status": newStatus}  # A6
                sql = "UPDATE ATLAS_PANDA.JEDI_Tasks SET status=:status WHERE jediTaskID=:jediTaskID"
                _logger.debug(f"status {newStatus}")
                self.cur.execute(sql, varMap)  # W3

            def otherPrinted(self, jediTaskID, newStatus):
                other = {":jediTaskID": jediTaskID}
                varMap = {":jediTaskID": jediTaskID, ":status": newStatus}  # A7
                sql = "UPDATE ATLAS_PANDA.JEDI_Tasks SET status=:status WHERE jediTaskID=:jediTaskID"
                _logger.debug(sql + str(other))
                self.cur.execute(sql, varMap)  # W4

            def changedAfterPrint(self, jediTaskID, newStatus):
                varMap = {":jediTaskID": jediTaskID}
                varMap[":status"] = newStatus  # A8
                sql = "UPDATE ATLAS_PANDA.JEDI_Tasks SET status=:status WHERE jediTaskID=:jediTaskID"
                _logger.debug(sql + str(varMap))
                varMap[":status"] = "done"  # A9
                self.cur.execute(sql, varMap)  # W5

            def updateOdd(self, oddID):
                odd = OddSpec()
                odd.oddID = oddID
                odd.state = "done"  # A10
                sql = f"UPDATE ATLAS_PANDA.odd_table SET {odd.bindUpdateChangesExpression()} WHERE oddID=:ODDID"
                varMap = odd.valuesMap(onlyChanged=True)
                _logger.debug(sql + str(varMap))  # P6
                self.cur.execute(sql, varMap)  # W6

            def updateKeys(self, jediTaskID, lfn):
                varMap = {":jediTaskID": jediTaskID, ":status": "ready"}  # A11
                sql = "UPDATE ATLAS_PANDA.JEDI_Tasks SET status=:status"
                for tmpKey in ["lfn"]:
                    tmpMapKey = f":{tmpKey}"
                    sql += f",{tmpKey}={tmpMapKey}"
                    varMap[tmpMapKey] = lfn
                sql += " WHERE jediTaskID=:jediTaskID"
                _logger.debug(sql + str(varMap))  # P7
                self.cur.execute(sql, varMap)  # W7
        """,
}


def _line(marker: str) -> int:
    (found,) = [
        number
        for number, held in enumerate(textwrap.dedent(SOURCES[PROXY]).splitlines(), 1)
        if held.rstrip().endswith(f"# {marker}")
    ]
    return found


def _modules(tmp_path: Path) -> list[SourceModule]:
    modules = []
    for rel, text in SOURCES.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text))
        parsed = calls_mod.parsed(path)
        assert parsed is not None
        modules.append(
            SourceModule(
                package=PACKAGE, rel_path=rel, tree=parsed[1], source=textwrap.dedent(text), blob_sha=blob_sha(text)
            )
        )
    return modules


#: The attribute slice's arms on the objects the writes persist: a KeyFrom of an object's write takes them
#: (``sqlwrite.attach_key_sources``).
OBJECT_ARMS = (("updateJob", "JobSpec.modificationTime", "A1"), ("updateOdd", "OddSpec.state", "A10"))


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> MapFragment:
    """The proxy read as the build does: the writes' records made in the SQL slice, the objects' arms joined to their
    KeyFroms after, the arms attached last."""
    tmp_path: Path = tmp_path_factory.mktemp("writeprints")
    modules = _modules(tmp_path)
    calls = sqlwrite.corpus_calls(modules, {PACKAGE: tmp_path / PACKAGE})
    assert calls is not None
    key_froms: list[KeyFrom] = []
    write_prints: list[WritePrint] = []
    record = reach.ReachGraph(reach.graph(calls, "m", "v", _DatabaseLeaves()))
    # The objects' records the build holds from before its SQL slice: what a write's valuesMap() takes.
    entities = [
        EntityNode(name=table.lower(), map_id="m", derived_from="v", spec_class=spec, tables=[table])
        for spec, table in (("JobSpec", "jobsActive4"), ("OddSpec", "odd_table"))
    ]
    panda_objects.attach(
        MapFragment(map_id="m", derived_from="v", entities=entities, reach_graph=list(record.rows.values())),
        modules,
        calls,
    )
    with sql.across_calls(calls), across_records(record.rows, reach._Names(calls.types).of):
        attributor = attribution.SpecAttributor(progress.spec_attributes(modules), attribution.class_bases(modules))
        attributor.learn_table_classes(modules)
        _subjects, junctions, *_rest = sqlwrite.extract(
            modules, "m", "v", attributor, key_froms=key_froms, set_froms=[], write_prints=write_prints
        )
        junctions += [
            JunctionNode(
                name=f"m:{subject}:{function}",
                map_id="m",
                derived_from="v",
                subject=subject,
                owner=f"{PROXY}::{function}",
                anchor=Anchor(package=PACKAGE, file=PROXY, line_start=_line(marker)),
                branches=[Branch(outcome=marker, line=_line(marker))],
            )
            for function, subject, marker in OBJECT_ARMS
        ]
        sqlwrite.attach_key_sources([], junctions, calls, record, key_froms=key_froms)
    sqlwrite.attach_write_arms(write_prints, key_froms)
    prints = [
        FunctionPrintsNode(
            name=f"m:{found.owner}",
            map_id="m",
            derived_from="v",
            owner=found.owner,
            cls=found.cls,
            function=found.func.name,
            count=len(found.statements),
            statements=found.statements,
            anchor=Anchor(package=PACKAGE, file=PROXY, line_start=found.func.lineno, line_end=found.func.end_lineno),
        )
        for found in trace.function_prints({PACKAGE: tmp_path / PACKAGE}, [PROXY])
    ]
    return MapFragment(
        map_id="m",
        derived_from="v",
        junctions=junctions,
        key_froms=key_froms,
        write_prints=write_prints,
        function_prints=prints,
    )


def _of(built: MapFragment, function: str) -> list[WritePrint]:
    return [one for one in built.write_prints if one.owner == f"{PROXY}::{function}"]


def _binds(one: WritePrint) -> list[tuple[str, str]]:
    return [(bind.column.rsplit(".", 1)[1].lower(), bind.bind) for bind in one.binds]


def test_an_object_s_map_printed_before_its_write_says_each_attribute_s_bind_and_that_only_changes_are_in_it(
    built: MapFragment,
) -> None:
    """updateJob prints its UPDATE's text and the job's changed fields before running it: the record names the print,
    the job as the map's object, each of JobSpec's attributes bound ``:`` and its name -- the key form its valuesMap
    stores --, and the arm that set modificationTime."""
    (one,) = _of(built, "updateJob")
    assert (one.statement, one.table) == (_line("W1"), "jobsActive4")
    assert one.printed_by == [PrintedBy(line=_line("P1"), where="before", text="whole")]
    assert (one.object_map, one.only_changed) == ("job", True)
    assert _binds(one) == [("pandaid", ":PandaID"), ("jobstatus", ":jobStatus"), ("modificationtime", ":modificationTime")]
    assert one.arms["JobSpec.modificationTime"] == [_line("A1")]


def test_a_spelled_set_names_each_column_by_the_bind_that_writes_it_and_leaves_an_expression_out(
    built: MapFragment,
) -> None:
    """updateStatus writes modificationTime from ``:updateTime`` -- set at A3, or at A4 on the other branch -- and
    attemptNr from an expression of ``:inc``: the printed ``:updateTime`` is the value written, the printed ``:inc``
    is not attemptNr's."""
    (one,) = _of(built, "updateStatus")
    assert one.printed_by == [PrintedBy(line=_line("P2"), where="before", text="whole", lead="running sql: ", gap=" ")]
    assert (one.object_map, one.only_changed) == ("", False)
    assert sorted(_binds(one)) == [("modificationtime", ":updateTime"), ("status", ":status")]
    modified = [lines for subject, lines in one.arms.items() if subject.lower().endswith(".modificationtime")]
    assert modified == [[_line("A3"), _line("A4")]]


def test_no_record_where_no_print_holds_the_text_and_the_map_the_write_runs_with(built: MapFragment) -> None:
    """A write printing something else, another map, or its map before an entry is filled again says nothing of its
    values: those lines are not the statement that ran."""
    for function in ("notPrinted", "otherPrinted", "changedAfterPrint"):
        assert _of(built, function) == []


def test_an_object_map_whose_keys_are_not_read_says_where_it_prints_and_no_value(built: MapFragment) -> None:
    """OddSpec's valuesMap keys its entries by ``attr.upper()``, not a fixed prefix and the name: the print is said,
    the binds are not, so no value is read off it."""
    (one,) = _of(built, "updateOdd")
    assert one.printed_by == [PrintedBy(line=_line("P6"), where="before", text="whole")]
    assert (one.object_map, one.binds) == ("odd", [])


def test_a_spec_s_map_key_form_is_read_from_its_valuesmap(tmp_path: Path) -> None:
    """``ret[f":{attr}"]`` over the class's attributes is the prefix ``:``; a key built any other way is no form."""
    assert progress.spec_map_keys(_modules(tmp_path)) == {"JobSpec": ":"}


def test_a_write_s_print_is_one_whose_rows_hold_its_map_and_its_binds(built: MapFragment) -> None:
    """The gate: every print a record names is one of the function's prints whose rows print a map whole, every bind
    a key of that map or of the object's, and the statement a KeyFrom's.  updateKeys makes ``:lfn`` in a loop, a key
    the print's map holds as computed: such a map may hold any bind, so ``:lfn`` is inconclusive, not a failure."""
    (keys,) = _of(built, "updateKeys")
    assert sorted(bind.bind for bind in keys.binds) == [":lfn", ":status"]
    result = gates.write_prints_are_rows(built)
    assert result.passed, result.failures
    assert result.checked == 4
    assert result.inconclusive == [f"{PROXY}::updateKeys:{_line('W7')}: bind :lfn may be a computed key"]
    (one,) = _of(built, "updateStatus")
    wrong = one.model_copy(update={"binds": [*one.binds, WriteBind(column="JEDI_Tasks.nothing", bind=":nothing")]})
    others = [other for other in built.write_prints if other is not one]
    failed = gates.write_prints_are_rows(built.model_copy(update={"write_prints": [*others, wrong]}))
    assert [failure.split(": ", 1)[1] for failure in failed.failures] == ["bind :nothing is no key of the printed map"]
