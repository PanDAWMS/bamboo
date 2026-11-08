"""What a name a guard reads holds, way by way across calls -- ``codemap.panda.heldfrom`` -- on the shape of JEDI's
task refiner: the task's parameters fetched by the proxy, returned through the task buffer, decoded, handed to the
refiner's guards."""

from __future__ import annotations

import ast
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
    KeyStop,
    MapFragment,
    ParameterFrom,
    SourceModule,
)
from bamboo.codemap.panda import attribution, heldfrom, sql, sqltext
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


def _line(rel: str, marker: str, sources: dict[str, str] = SOURCES) -> int:
    (found,) = [
        number
        for number, held in enumerate(textwrap.dedent(sources[rel]).splitlines(), 1)
        if held.rstrip().endswith(f"# {marker}")
    ]
    return found


def _at(rel: str, marker: str) -> str:
    return f"{rel}:{_line(rel, marker)}"


def _build(
    tmp_path: Path,
    sources: dict[str, str] = SOURCES,
    arms: tuple[tuple[str, str], ...] = (
        ("extractCommon", "R1"),
        ("extractCommon", "R2"),
        ("extractCommon", "R3"),
        ("setSplitRule", "S1"),
    ),
    base: str = BASE,
) -> MapFragment:
    """Read the sources as the build does, with a junction for each arm marked in the refiner's base."""
    modules = []
    for rel, text in sources.items():
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
            owner=f"{base}::{owner}",
            anchor=Anchor(
                package="pandajedi",
                file=base,
                line_start=_line(base, marker, sources),
            ),
            branches=[Branch(outcome=marker, line=_line(base, marker, sources))],
        )
        for owner, marker in arms
    ]
    with sql.across_calls(calls):
        attributor = attribution.SpecAttributor(
            progress.spec_attributes(modules), attribution.class_bases(modules)
        )
        sites = sqlwrite.read_site_nodes(
            sqlwrite.read_side(modules, attributor, calls).statements, "m", "v"
        )
        record = reach.ReachGraph(reach.graph(calls, "m", "v", _DatabaseLeaves()))
        guards, parameters, _writes, _written = heldfrom.guard_froms(
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
            tuple(
                (call.call, call.at, call.gives, tuple(call.hooks))
                for call in one.through
            ),
            tuple(one.route),
        )
        for one in sources
    }


def test_a_guard_s_parameter_is_read_at_its_callers_back_to_the_rows_a_fetch_returns(
    built: MapFragment,
) -> None:
    """``'reqID' in taskParamMap``: the parameter, and what ``runImpl`` hands it -- the proxy's column where a row
    came, returned through the task buffer and ``runImpl``'s two calls, read again by ``json.loads`` with the
    corpus's ``unicodeConvert`` as its hook, handed in at ``extractCommon``'s call.  ``""`` where no row came and
    ``None`` where the read raised go no further than ``json.loads``, which raises on both."""
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
    loads = f"{UTILS}:{_line_of(UTILS, 'return json.loads')}"
    through = (("json.loads", loads, "decoded", (f"{UTILS}::unicodeConvert",)),)
    site = (
        f"panda:{PROXY}::getTaskParamsWithID_JEDI@{_line_of(PROXY, 'self.cur.execute')}"
    )
    parameter = _parameter(built, "extractCommon", "taskParamMap")
    assert _ways(parameter.sources) == {
        (
            "rows",
            site.replace("panda:", "m:"),
            "jedi_taskparams.taskparams",
            (),
            through,
            route,
        ),
    }
    assert sorted(parameter.stops, key=lambda one: one.name) == [
        KeyStop(
            kind="raises",
            name=name,
            what=f"json.loads raises {error}",
            at=[*route, loads],
        )
        for name, error in (("''", "JSONDecodeError"), ("None", "TypeError"))
    ]


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


CHANGING = "pandajedi/jedirefine/BaseRefiner.py"
SUBCLASS = "pandajedi/jedirefine/AnalRefiner.py"
DRIVER = "pandajedi/jediorder/Refiner.py"

CHANGED = {
    CHANGING: """\
        class BaseRefiner:
            def extractCommon(self, jediTaskID, taskParamMap):
                self.fill(taskParamMap)  # P1
                if "transPath" in taskParamMap:  # T1
                    self.transPath = taskParamMap["transPath"]  # R1
                    taskParamMap["late"] = 2  # L1
                self.queue.append(taskParamMap)  # Q1
                if "filled" in taskParamMap:  # T2
                    self.filled = True  # R2

            def fill(self, params):
                params["filled"] = True  # F1

            def touch(self):
                self.saved["touched"] = 3  # H1

            def count(self, jediTaskID: int, varMap):
                varMap[":id"] = jediTaskID
                varMap[":map"] = varMap.copy()
                if jediTaskID > 0 and ":map" in varMap:  # T3
                    self.counted = True  # R3
        """,
    SUBCLASS: """\
        from pandajedi.jedirefine.BaseRefiner import BaseRefiner

        class AnalRefiner(BaseRefiner):
            def extractCommon(self, jediTaskID, taskParamMap):
                if "transPath" not in taskParamMap:
                    taskParamMap["transPath"] = "runGen"  # A1
                self.saved = taskParamMap  # A2
                BaseRefiner.extractCommon(self, jediTaskID, taskParamMap)  # A3
        """,
    DRIVER: """\
        import json

        from pandajedi.jedirefine.AnalRefiner import AnalRefiner

        class Refiner:
            def runImpl(self, impl: AnalRefiner, text):
                taskParamMap = json.loads(text)  # C1
                impl.extractCommon(1, taskParamMap)  # C2
        """,
}


@pytest.fixture(scope="module")
def changed(tmp_path_factory: pytest.TempPathFactory) -> MapFragment:
    return _build(
        tmp_path_factory.mktemp("heldchanges"),
        CHANGED,
        (("extractCommon", "R1"), ("extractCommon", "R2"), ("count", "R3")),
        CHANGING,
    )


def _changes(source: HeldSource) -> list[tuple]:
    return [
        (
            one.kind,
            one.key,
            one.at,
            one.via,
            [(way.kind, way.name) for way in one.value],
        )
        for one in source.changes
    ]


def test_a_way_names_each_change_of_the_object_in_the_functions_it_passes(
    changed: MapFragment,
) -> None:
    """``'transPath' in taskParamMap`` in the base reads the map ``AnalRefiner.extractCommon`` hands it, after storing
    ``transPath`` there -- and after putting it in ``self.saved``, through which ``touch`` stores ``touched`` -- and
    what ``fill`` stored in the base before the test.  The base's own store under the test, and what follows, come
    after it; before the second test, they are on the way: ``self.queue.append`` keeps the map where it is not
    followed."""

    def at(rel: str, marker: str) -> str:
        return f"{rel}:{_line(rel, marker, CHANGED)}"

    (guard,) = [
        one
        for one in changed.guard_froms
        if one.test_line == _line(CHANGING, "T1", CHANGED)
    ]
    ((source,),) = [guard.sources]
    filled = (
        "store",
        "'filled'",
        at(CHANGING, "F1"),
        [at(CHANGING, "P1")],
        [("constant", "True")],
    )
    assert _changes(source) == [filled]
    parameter = next(
        one
        for one in changed.parameter_froms
        if one.owner == f"{CHANGING}::extractCommon"
    )
    ((handed,),) = [parameter.sources]
    assert (handed.kind, handed.name, handed.route) == (
        "parameter",
        "taskParamMap",
        [at(SUBCLASS, "A3")],
    )
    assert _changes(handed) == [
        ("store", "'transPath'", at(SUBCLASS, "A1"), [], [("constant", "'runGen'")]),
        (
            "store",
            "'touched'",
            at(CHANGING, "H1"),
            [at(SUBCLASS, "A2")],
            [("constant", "3")],
        ),
    ]
    (second,) = [
        one
        for one in changed.guard_froms
        if one.test_line == _line(CHANGING, "T2", CHANGED)
    ]
    assert [(kind, key) for kind, key, *_rest in _changes(second.sources[0])] == [
        ("store", "'filled'"),
        ("store", "'late'"),
        ("escaped", "*"),
    ]
    driven = next(
        one
        for one in changed.parameter_froms
        if one.owner == f"{SUBCLASS}::extractCommon"
    )
    assert [_changes(one) for one in driven.sources] == [[]]


def test_a_value_no_change_in_place_can_touch_is_asked_of_none(
    changed: MapFragment,
) -> None:
    """``jediTaskID: int`` put in ``varMap`` is not changed through it; ``varMap`` itself is, before the test."""
    by_name = {
        one.name: one
        for one in changed.guard_froms
        if one.test_line == _line(CHANGING, "T3", CHANGED)
    }
    assert [source.changes for source in by_name["jediTaskID"].sources] == [[]]
    ((source,),) = [by_name["varMap"].sources]
    assert [(one.kind, one.key) for one in source.changes] == [
        ("store", "':id'"),
        ("store", "':map'"),
    ]


LITERALS = "pandajedi/jedirefine/LiteralRefiner.py"

THROUGH = {
    LITERALS: """\
        import copy
        import json
        import math as m

        def hook(pairs):
            return pairs

        class LiteralRefiner:
            def refine(self, flag):
                text = ""
                count = None
                if flag:
                    text = '{"a": 1}'
                    count = "3"
                params = json.loads(text, object_hook=hook)  # L1
                limit = int(count)  # I1
                nested = int(json.loads("null"))  # N1
                kept = copy.copy(params)  # K1
                cores = m.ceil(None)  # M1
                size = len(params)  # S1
                custom = json.loads("", cls=hook)  # J1
                item = json.loads(text)["a"]  # A1
                pair = None
                first = int(pair[0])  # P1
                if limit and "a" in kept and nested and cores and size and custom and item and first:  # T1
                    self.ok = True  # R1
        """,
}


@pytest.fixture(scope="module")
def through(tmp_path_factory: pytest.TempPathFactory) -> MapFragment:
    return _build(
        tmp_path_factory.mktemp("heldthrough"), THROUGH, (("refine", "R1"),), LITERALS
    )


def test_a_call_gone_through_says_what_its_value_is_and_a_literal_it_raises_on_stops(
    through: MapFragment,
) -> None:
    """Each call outside the corpus a way goes through says what its value is of what it is handed, as callfacts
    says it, ``math.ceil`` by its module's name under ``import math as m``.  A literal goes no further than the call
    that raises on it, run in turn on what the one before made -- ``int(json.loads("null"))`` raises at ``int`` --
    while each is one evaluate runs and is handed the literal itself: ``json.loads`` handed a hook makes what the
    hook makes, one handed ``cls`` parses its own way, and ``int(pair[0])`` is handed an item, not the literal; an
    item taken after the call (``json.loads(text)["a"]``) does not save a literal the call raises on."""

    def at(marker: str) -> str:
        return f"{LITERALS}:{_line(LITERALS, marker, THROUGH)}"

    guards = {
        one.name: one
        for one in through.guard_froms
        if one.test_line == _line(LITERALS, "T1", THROUGH)
    }
    loads = ("json.loads", at("L1"), "decoded", (f"{LITERALS}::hook",))

    def said(name: str) -> tuple[set[tuple], list[tuple]]:
        guard = guards[name]
        return _ways(guard.sources), [
            (one.kind, one.name, one.what, one.at) for one in guard.stops
        ]

    assert said("limit") == (
        {("constant", "'3'", "", (), (("int", at("I1"), "converted", ()),), ())},
        [("raises", "None", "int raises TypeError", [at("I1")])],
    )
    assert said("nested") == (
        set(),
        [("raises", "'null'", "int raises TypeError", [at("N1")])],
    )
    assert said("kept") == (
        {
            (
                "constant",
                """'{"a": 1}'""",
                "",
                (),
                (loads, ("copy.copy", at("K1"), "copy", ())),
                (),
            )
        },
        [("raises", "''", "json.loads raises JSONDecodeError", [at("L1")])],
    )
    assert said("cores") == (
        set(),
        [("raises", "None", "math.ceil raises TypeError", [at("M1")])],
    )
    assert said("size")[0] == {
        (
            "constant",
            """'{"a": 1}'""",
            "",
            (),
            (loads, ("len", at("S1"), "computed", ())),
            (),
        )
    }
    assert said("custom") == (
        {
            (
                "constant",
                "''",
                "",
                (),
                (("json.loads", at("J1"), "decoded", (f"{LITERALS}::hook",)),),
                (),
            )
        },
        [],
    )
    assert said("item") == (
        {
            (
                "constant",
                """'{"a": 1}'""",
                "",
                ("['a']",),
                (("json.loads", at("A1"), "decoded", ()),),
                (),
            )
        },
        [("raises", "''", "json.loads raises JSONDecodeError", [at("A1")])],
    )
    assert said("first") == (
        {("constant", "None", "", ("[0]",), (("int", at("P1"), "converted", ()),), ())},
        [],
    )
    # Where each call comes among the steps: ``int`` is handed ``pair[0]``, ``json.loads`` the text itself.
    assert [
        [call.after_steps for call in source.through]
        for name in ("first", "item")
        for source in guards[name].sources
    ] == [[1], [0]]


RETURNS_SPEC = "pandaserver/taskbuffer/JediTaskSpec.py"
RETURNS_PROXY = "pandaserver/taskbuffer/db_proxy_mods/task_utils_module.py"

RETURNS = {
    RETURNS_SPEC: """\
        class JediTaskSpec(object):
            _attributes = ("jediTaskID", "status", "splitRule")

            def __init__(self):
                for attr in self._attributes:
                    object.__setattr__(self, attr, None)

            def pack(self, values):
                for i in range(len(self._attributes)):
                    attr = self._attributes[i]
                    val = values[i]
                    object.__setattr__(self, attr, val)
        """,
    RETURNS_PROXY: """\
        from pandaserver.taskbuffer.JediTaskSpec import JediTaskSpec

        comment = " /* x */"


        class TaskUtilsModule:
            def getTaskWithID_JEDI(self, jediTaskID):
                sql = "SELECT jediTaskID,status,splitRule FROM ATLAS_PANDA.JEDI_Tasks WHERE jediTaskID=:jediTaskID "
                varMap = {}
                varMap[":jediTaskID"] = jediTaskID
                self.cur.execute(sql + comment, varMap)  # SELECT
                res = self.cur.fetchone()
                if res is None:
                    return False, None
                taskSpec = JediTaskSpec()
                taskSpec.pack(res)
                return True, taskSpec

            def fresh(self, jediTaskID):
                taskSpec = JediTaskSpec()
                taskSpec.jediTaskID = jediTaskID
                return True, taskSpec

            def facade(self, jediTaskID):
                return self.getTaskWithID_JEDI(jediTaskID)  # FACADE

            def caller(self, jediTaskID):
                tmpStat, taskSpec = self.getTaskWithID_JEDI(jediTaskID)
                unpacked = taskSpec.splitRule  # UNPACKED
                tmpStat, other = self.fresh(jediTaskID)
                built = other.splitRule  # BUILT
                indexed = self.getTaskWithID_JEDI(jediTaskID)[1].splitRule  # INDEXED
                tmpStat, viaFacade = self.facade(jediTaskID)
                faced = viaFacade.splitRule  # FACED
                status = taskSpec.status  # STATUS
        """,
}


@pytest.fixture(scope="module")
def returned(tmp_path_factory: pytest.TempPathFactory) -> dict[str, set[tuple]]:
    """What each local of ``caller`` holds, read as the build reads it: by the reader over the read sites, the corpus's
    calls and the reach record, under the schema."""
    tmp_path: Path = tmp_path_factory.mktemp("returned")
    modules = []
    for rel, text in RETURNS.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text))
        parsed = calls_mod.parsed(path)
        assert parsed is not None
        _source, tree = parsed
        modules.append(
            SourceModule(
                package="pandaserver",
                rel_path=rel,
                tree=tree,
                source=textwrap.dedent(text),
                blob_sha=blob_sha(text),
            )
        )
    calls = sqlwrite.corpus_calls(modules, {"pandaserver": tmp_path / "pandaserver"})
    assert calls is not None
    trees = {module.rel_path: module.tree for module in modules}
    caller = next(
        node
        for node in ast.walk(trees[RETURNS_PROXY])
        if isinstance(node, ast.FunctionDef) and node.name == "caller"
    )
    schema = {"jedi_tasks": frozenset({"jeditaskid", "status", "splitrule"})}
    with (
        sql.across_calls(calls),
        sqltext.with_schema(schema, {"jedi_tasks": frozenset({"jeditaskid"})}, {}),
    ):
        attributor = attribution.SpecAttributor(
            progress.spec_attributes(modules), attribution.class_bases(modules)
        )
        attributor.learn_table_classes(modules)
        sites = sqlwrite.read_site_nodes(
            sqlwrite.read_side(modules, attributor, calls).statements, "m", "v"
        )
        record = reach.ReachGraph(reach.graph(calls, "m", "v", _DatabaseLeaves()))
        reader = heldfrom._Reader(calls, record, sites, set(), trees)
        found: dict[str, set[tuple]] = {}
        for stmt in caller.body:
            if not isinstance(stmt, ast.Assign):
                continue
            target = stmt.targets[0]
            if isinstance(target, ast.Name) and target.id != "tmpStat":
                found[target.id] = {
                    (
                        one.kind,
                        one.name.rpartition("::")[2],
                        one.steps,
                        one.column,
                        tuple(int(where.rpartition(":")[2]) for where in one.route),
                    )
                    for one in reader.read(stmt.value, stmt, RETURNS_PROXY, caller)
                }
    return found


def test_a_field_of_what_a_call_returns_is_read_in_the_callee_at_its_returns(
    returned: dict[str, set[tuple]],
) -> None:
    """``taskSpec.splitRule`` of ``tmpStat, taskSpec = self.getTaskWithID_JEDI(jediTaskID)``: the callee packs the
    row it selects into the object it returns, so the field is the row's column -- read there, at the return, not
    as the object the caller would see built (``JediTaskSpec()``).  So too a call's element read directly and through
    a facade; the failed ``return False, None`` gives a field of ``None``.  An object the callee builds itself stays
    an object with the field's step."""

    def at(marker: str) -> int:
        return _line(RETURNS_PROXY, marker, RETURNS)

    select = f"getTaskWithID_JEDI@{at('SELECT')}"

    def row(column: str, *route: int) -> set[tuple]:
        return {
            ("rows", select, (), f"jedi_tasks.{column}", route),
            (
                "constant",
                "None",
                (f".{'splitRule' if column == 'splitrule' else column}",),
                "",
                route,
            ),
        }

    assert returned["unpacked"] == row("splitrule", at("UNPACKED") - 1)
    assert returned["indexed"] == row("splitrule", at("INDEXED"))
    assert returned["faced"] == row("splitrule", at("FACED") - 1, at("FACADE"))
    assert returned["status"] == row("status", at("UNPACKED") - 1)
    assert returned["built"] == {
        ("object", "JediTaskSpec", (".splitRule",), "", (at("BUILT") - 1,))
    }
