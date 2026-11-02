"""What a name a guard reads holds, way by way across calls -- ``codemap.panda.heldfrom`` -- on the shape of JEDI's
task refiner: the task's parameters fetched by the proxy, returned through the task buffer, decoded, handed to the
refiner's guards."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from bamboo.codemap import calls as calls_mod
from bamboo.codemap import reach
from bamboo.codemap.gates import guard_parameters_are_read, guard_rows_are_read_sites
from bamboo.codemap.gitsource import blob_sha
from bamboo.codemap.models import (
    Anchor,
    Branch,
    GuardFrom,
    HeldSource,
    JunctionNode,
    MapFragment,
    ParameterFrom,
    SourceModule,
)
from bamboo.codemap.panda import attribution, heldfrom, sql
from bamboo.codemap.panda.plugin import _DatabaseLeaves
from bamboo.codemap.panda.recognizers import progress, sqlwrite

PROXY = "pandaserver/taskbuffer/db_proxy_mods/task_module.py"
DBPROXY = "pandaserver/taskbuffer/OraDBProxy.py"
BUFFER = "pandajedi/jedicore/JediTaskBuffer.py"
UTILS = "pandajedi/jedirefine/RefinerUtils.py"
BASE = "pandajedi/jedirefine/TaskRefinerBase.py"
REFINER = "pandajedi/jediorder/TaskRefiner.py"

SOURCES = {
    PROXY: """\
        class TaskModule:
            def getTaskParamsWithID_JEDI(self, jediTaskID):
                comment = " /* JediDBProxy.getTaskParamsWithID_JEDI */"
                try:
                    sql = f"SELECT taskParams FROM {panda_config.schemaJEDI}.JEDI_TaskParams WHERE jediTaskID=:jediTaskID "
                    varMap = {}
                    varMap[":jediTaskID"] = jediTaskID
                    self.cur.execute(sql + comment, varMap)
                    retStr = ""
                    for (tmpItem,) in self.cur:
                        retStr = tmpItem
                        break
                    return retStr
                except Exception:
                    return None
        """,
    DBPROXY: """\
        from pandaserver.taskbuffer.db_proxy_mods.task_module import TaskModule

        class DBProxy(TaskModule):
            pass
        """,
    BUFFER: """\
        from pandaserver.taskbuffer.OraDBProxy import DBProxy

        class JediTaskBuffer:
            def __init__(self):
                self.proxy = DBProxy()

            def getTaskParamsWithID_JEDI(self, jediTaskID):
                return self.proxy.getTaskParamsWithID_JEDI(jediTaskID)
        """,
    UTILS: """\
        import json

        def unicodeConvert(input):
            return input

        def decodeJSON(inputStr):
            return json.loads(inputStr, object_hook=unicodeConvert)
        """,
    BASE: """\
        class TaskRefinerBase:
            def extractCommon(self, jediTaskID, taskParamMap):
                if "reqID" in taskParamMap:  # T1
                    self.reqID = taskParamMap["reqID"]  # R1
                else:
                    self.reqID = jediTaskID  # R2
                self.setSplitRule(taskParamMap, "nFilesPerJob", "NF")
                self.setSplitRule(None, taskParamMap["tgtNumEventsPerJob"], "NE")
                status = "fresh"
                if self.flag:
                    status = self.oldStatus
                if status == "fresh":  # T2
                    self.reset = True  # R3

            def setSplitRule(self, taskParamMap, key_or_value, rule_token):
                if taskParamMap is not None:  # T3
                    if key_or_value not in taskParamMap:  # T4
                        self.splitRule = None  # S1
        """,
    REFINER: """\
        from pandajedi.jedicore.JediTaskBuffer import JediTaskBuffer
        from pandajedi.jedirefine import RefinerUtils
        from pandajedi.jedirefine.TaskRefinerBase import TaskRefinerBase

        class TaskRefiner:
            def __init__(self, taskBufferIF: JediTaskBuffer):
                self.taskBufferIF = taskBufferIF

            def runImpl(self, jediTaskID, impl: TaskRefinerBase):
                taskParam = self.taskBufferIF.getTaskParamsWithID_JEDI(jediTaskID)  # C1
                taskParamMap = RefinerUtils.decodeJSON(taskParam)  # C2
                impl.extractCommon(jediTaskID, taskParamMap)  # C3
        """,
}


def _line(rel: str, marker: str) -> int:
    (found,) = [
        number
        for number, held in enumerate(textwrap.dedent(SOURCES[rel]).splitlines(), 1)
        if held.rstrip().endswith(f"# {marker}")
    ]
    return found


def _at(rel: str, marker: str) -> str:
    return f"{rel}:{_line(rel, marker)}"


def _build(tmp_path: Path) -> MapFragment:
    """Read the sources as the build does, with a junction for each arm marked in the refiner's base."""
    modules = []
    for rel, text in SOURCES.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text))
        parsed = calls_mod.parsed(path)
        assert parsed is not None
        _source, tree = parsed
        modules.append(
            SourceModule(
                package=rel.split("/", 1)[0],
                rel_path=rel,
                tree=tree,
                source=textwrap.dedent(text),
                blob_sha=blob_sha(text),
            )
        )
    roots = {
        "pandaserver": tmp_path / "pandaserver",
        "pandajedi": tmp_path / "pandajedi",
    }
    calls = sqlwrite.corpus_calls(modules, roots)
    assert calls is not None
    junctions = [
        JunctionNode(
            name=f"m:{owner}:{marker}",
            map_id="m",
            derived_from="v",
            subject="JediTaskSpec.splitRule",
            owner=f"{BASE}::{owner}",
            anchor=Anchor(
                package="pandajedi", file=BASE, line_start=_line(BASE, marker)
            ),
            branches=[Branch(outcome=marker, line=_line(BASE, marker))],
        )
        for owner, marker in (
            ("extractCommon", "R1"),
            ("extractCommon", "R2"),
            ("extractCommon", "R3"),
            ("setSplitRule", "S1"),
        )
    ]
    with sql.across_calls(calls):
        attributor = attribution.SpecAttributor(
            progress.spec_attributes(modules), attribution.class_bases(modules)
        )
        sites = sqlwrite.read_site_nodes(
            sqlwrite.read_side(modules, attributor, calls).statements, "m", "v"
        )
        record = reach.ReachGraph(reach.graph(calls, "m", "v", _DatabaseLeaves()))
        guards, parameters = heldfrom.guard_froms(
            junctions,
            sites,
            {module.rel_path: module.tree for module in modules},
            calls,
            record,
        )
    return MapFragment(
        map_id="m",
        derived_from="v",
        junctions=junctions,
        read_sites=sites,
        guard_froms=guards,
        parameter_froms=parameters,
    )


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> MapFragment:
    return _build(tmp_path_factory.mktemp("heldfrom"))


def _guard(fragment: MapFragment, owner: str, marker: str, name: str) -> GuardFrom:
    (found,) = [
        one
        for one in fragment.guard_froms
        if one.owner == f"{BASE}::{owner}"
        and one.test_line == _line(BASE, marker)
        and one.name == name
    ]
    return found


def _parameter(fragment: MapFragment, owner: str, name: str) -> ParameterFrom:
    (found,) = [
        one
        for one in fragment.parameter_froms
        if one.owner == f"{BASE}::{owner}" and one.parameter == name
    ]
    return found


def _ways(sources: list[HeldSource]) -> set[tuple]:
    return {
        (
            one.kind,
            one.name,
            one.column,
            tuple(one.steps),
            tuple(one.through),
            tuple(one.route),
        )
        for one in sources
    }


def test_a_guard_s_parameter_is_read_at_its_callers_back_to_the_rows_a_fetch_returns(
    built: MapFragment,
) -> None:
    """``'reqID' in taskParamMap``: the parameter, and what ``runImpl`` hands it -- the proxy's column where a row
    came, ``""`` where none did, ``None`` where it raised -- each returned through the task buffer and
    ``runImpl``'s two calls, gone through ``json.loads``, handed in at ``extractCommon``'s call."""
    guard = _guard(built, "extractCommon", "T1", "taskParamMap")
    assert guard.arms == [_line(BASE, "R1"), _line(BASE, "R2")]
    ((source,),) = [guard.sources]
    assert (source.kind, source.name, source.site_owner) == (
        "parameter",
        "taskParamMap",
        f"{BASE}::extractCommon",
    )
    route = (
        f"{BUFFER}:{_line_of(BUFFER, 'return self.proxy')}",
        _at(REFINER, "C1"),
        _at(REFINER, "C2"),
        _at(REFINER, "C3"),
    )
    through = (f"json.loads {UTILS}:{_line_of(UTILS, 'return json.loads')}",)
    site = (
        f"panda:{PROXY}::getTaskParamsWithID_JEDI@{_line_of(PROXY, 'self.cur.execute')}"
    )
    assert _ways(_parameter(built, "extractCommon", "taskParamMap").sources) == {
        (
            "rows",
            site.replace("panda:", "m:"),
            "jedi_taskparams.taskparams",
            (),
            through,
            route,
        ),
        ("constant", "''", "", (), through, route),
        ("constant", "None", "", (), through, route),
    }


def _line_of(rel: str, text: str) -> int:
    (found,) = [
        number
        for number, held in enumerate(textwrap.dedent(SOURCES[rel]).splitlines(), 1)
        if text in held
    ]
    return found


def test_an_item_is_its_root_s_ways_with_the_step_and_each_caller_is_a_way(
    built: MapFragment,
) -> None:
    """``setSplitRule`` is handed the task's parameters and ``None``; its ``key_or_value`` a literal and an item of
    the parameters, ``['tgtNumEventsPerJob']``, not the parameters themselves."""
    first = f"{BASE}:{_line_of(BASE, 'self.setSplitRule(taskParamMap')}"
    second = f"{BASE}:{_line_of(BASE, 'self.setSplitRule(None')}"
    assert _ways(_parameter(built, "setSplitRule", "taskParamMap").sources) == {
        ("parameter", "taskParamMap", "", (), (), (first,)),
        ("constant", "None", "", (), (), (second,)),
    }
    assert _ways(_parameter(built, "setSplitRule", "key_or_value").sources) == {
        ("constant", "'nFilesPerJob'", "", (), (), (first,)),
        ("parameter", "taskParamMap", "", ("['tgtNumEventsPerJob']",), (), (second,)),
    }
    assert {
        one.name for one in built.guard_froms if one.owner.endswith("::setSplitRule")
    } == {
        "taskParamMap",
        "key_or_value",
    }


def test_a_local_is_one_way_per_binding_that_reaches_the_read(
    built: MapFragment,
) -> None:
    """``status`` is ``"fresh"`` or, where the flag held, the object's ``oldStatus``."""
    guard = _guard(built, "extractCommon", "T2", "status")
    assert _ways(guard.sources) == {
        ("constant", "'fresh'", "", (), (), ()),
        ("self_field", "self.oldStatus", "", (), (), ()),
    }


def test_the_cursor_a_statement_runs_on_is_no_source_of_what_a_loop_over_it_gives(
    built: MapFragment,
) -> None:
    sources = [one for guard in built.guard_froms for one in guard.sources]
    sources += [one for parameter in built.parameter_froms for one in parameter.sources]
    assert not [
        one for one in sources if one.kind == "self_field" and one.name == "self.cur"
    ]


def test_the_ways_name_read_sites_and_recorded_parameters(built: MapFragment) -> None:
    assert guard_rows_are_read_sites(built).passed
    assert guard_parameters_are_read(built).passed
    assert not guard_rows_are_read_sites(
        built.model_copy(update={"read_sites": []})
    ).passed
    kept = [
        one for one in built.parameter_froms if one.owner != f"{BASE}::extractCommon"
    ]
    assert not guard_parameters_are_read(
        built.model_copy(update={"parameter_froms": kept})
    ).passed
