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
        guards, parameters, _writes, _written, fields = heldfrom.guard_froms(
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
        field_froms=fields,
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


def test_a_step_cuts_a_way_s_paths_and_a_part_no_way_reaches_is_read_as_unknown() -> (
    None
):
    """adder_gen 265's shape, ``self.adder_plugin.result``: a way that is the receiver takes the step; a way at parts
    of the receiver keeps the parts under the step, cut by it; one with no part under it is not in the value."""
    plugin = heldfrom._Way("self_field", "self.adder_plugin")
    job = heldfrom._Way("self_field", "self.job", paths=(".job", ".result.files[*]"))
    assert heldfrom._taken([plugin, job], ".result") == [
        plugin._replace(steps=(".result",)),
        job._replace(paths=(".files[*]",)),
    ]
    # A part taken whole is the source itself.
    assert heldfrom._taken([job._replace(paths=(".result",))], ".result") == [
        job._replace(paths=())
    ]
    # No way reaches the part: an attribute's value, not a way of a source that is elsewhere.
    alone = job._replace(paths=(".job",))
    assert heldfrom._taken([alone], ".result") == [
        heldfrom._Way("attribute", "self.job.result")
    ]
    # A with's __enter__() mostly gives the context itself: its paths stay.
    assert heldfrom._taken([alone], heldfrom._ENTER) == [
        alone._replace(steps=(heldfrom._ENTER,))
    ]
    # A loop over a dict takes its keys; an index does not.
    keyed = job._replace(paths=("[key]",))
    assert heldfrom._taken([keyed], "[*]", element=True) == [keyed._replace(paths=())]
    assert heldfrom._taken([keyed], "[*]") == [heldfrom._Way("unknown", "self.job[*]")]


def test_a_part_the_carry_places_a_source_at_is_not_the_source_but_a_part_the_reader_places_it_at_is() -> (
    None
):
    """``jobSpec.Files + pseudoFiles``: the carry says the sum holds ``jobSpec`` or a part of it at ``[*]``; an
    element is no job.  ``[taskSpec.site]`` holds ``taskSpec.site`` at ``[0]``, which the reader placed."""
    carried = heldfrom._Way("parameter", "jobSpec", paths=("[*]",), loose=True)
    assert heldfrom._taken([carried], "[*]", element=True) == [
        heldfrom._Way("unknown", "jobSpec[*]")
    ]
    unread = heldfrom._Way("unknown", "fileSpec")
    assert heldfrom._taken(
        [carried, carried._replace(name="other")], "[*]", unread=unread
    ) == [unread]
    site = heldfrom._Way("parameter", "taskSpec", steps=(".site",))
    placed = heldfrom._at("[0]", [site])
    assert placed == [site._replace(paths=("[0]",))]
    assert heldfrom._taken(placed, "[*]", element=True) == [site]
    # A comprehension at a place is taken by the place's step, not its element's.
    made = heldfrom._Through("<listcomp>", "x.py:1", "computed")
    listed = heldfrom._at("[0]", [site._replace(through=(made._replace(steps=1),))])
    assert heldfrom._taken(listed, "[0]") == [
        site._replace(through=(made._replace(steps=1),))
    ]
    # What a callee holds at a place stays there at the caller: ``return (self.SC_FAILED, inputChunk)``.
    chunk = heldfrom._Way("parameter", "chunk")
    returned = heldfrom._Way("parameter", "inputChunk", paths=("[1]",))
    assert heldfrom._onto(chunk, returned, (), ("x.py:2",)) == [
        chunk._replace(paths=("[1]",), route=("x.py:2",))
    ]
    # An argument at parts of what is handed is cut by the callee's steps.
    job = heldfrom._Way("parameter", "job", paths=(".Files[*]", ".extra"))
    files = heldfrom._Way("parameter", "j", steps=(".Files",))
    assert heldfrom._onto(job, files, (), ()) == [job._replace(paths=("[*]",))]
    # A container made empty and filled: what was put, cut by the steps after the element's.
    put = heldfrom._Change(
        "x.py:3",
        heldfrom.CHANGE_METHOD,
        heldfrom.changes_mod.EVERY,
        "append",
        (),
        tuple(placed),
    )
    rows = heldfrom._Way(
        "unknown", "[]", made_empty="collection", steps=("[0]", "[0]"), changes=(put,)
    )
    assert heldfrom._joined(rows) == [site]


PLACED = {
    BASE: """\
        class TaskRefinerBase:
            def __init__(self):
                self.items = []
                self.specs = {}

            def extractCommon(self, jediTaskID, taskSpec, jobSpec):
                sites = [taskSpec.site]
                for site in sites:
                    if site == "a":  # T1
                        self.a = 1  # R1
                pair = (taskSpec.status, jobSpec)
                if pair[0] == "b":  # T2
                    self.b = 1  # R2
                names = []
                names += [taskSpec.name]
                for n in names:
                    if n == "d":  # T4
                        self.d = 1  # R4
                labels = {"k": taskSpec.label}
                if labels["k"] == "e":  # T5
                    self.e = 1  # R5
                chosen = taskSpec.first if jediTaskID else taskSpec.second
                if chosen == "f":  # T6
                    self.f = 1  # R6
                self.items.append(taskSpec.item)
                for item in self.items:
                    if item == "g":  # T7
                        self.g = 1  # R7
                for g in jobSpec.Files + self.pseudo(jobSpec):
                    if g.lfn == "h":  # T8
                        self.h = 1  # R8
                for key in list(self.specs):
                    self.specs[key + "x"] = self.specs[key]
                spec = self.specs["a"]
                if spec == "i":  # T9
                    self.i = 1  # R9

            def pseudo(self, jobSpec):
                made = []
                made.append(jobSpec.PandaID)
                return made
        """,
}


def test_a_literal_a_conditional_and_a_sequence_joined_are_read_where_they_hold_each_value(
    tmp_path: Path,
) -> None:
    """The shapes the carry answered with a source or a part of it: a list, a tuple and a dict literal, an ``if``
    expression, ``+=`` and ``+`` of sequences.  The reader reads them, each value where it sits; what the carry
    alone says sits at a field's element (T7) is a value not read, not the task.  A field read as itself holds its
    own parts at its parts (T9, ``SiteMapper.siteSpecList``'s shape): its steps take them."""
    arms = tuple(("extractCommon", f"R{n}") for n in (1, 2, 4, 5, 6, 7, 8, 9))
    fragment = _build(tmp_path, PLACED, arms)

    def held(marker: str, name: str) -> set[tuple]:
        (guard,) = [
            one
            for one in fragment.guard_froms
            if one.test_line == _line(BASE, marker, PLACED) and one.name == name
        ]
        return {
            (one.kind, one.name, tuple(one.steps), tuple(one.paths))
            for one in guard.sources
        }

    assert held("T1", "site") == {("parameter", "taskSpec", (".site",), ())}
    assert held("T2", "pair") == {
        ("parameter", "taskSpec", (".status",), ("[0]",)),
        ("parameter", "jobSpec", (), ("[1]",)),
    }
    # The empty list's element is a value nothing put there names, as before.
    assert held("T4", "n") == {
        ("unknown", "[]", ("[*]",), ()),
        ("parameter", "taskSpec", (".name",), ()),
    }
    assert held("T5", "labels") == {
        ("parameter", "taskSpec", (".label",), ("['k']",)),
        ("constant", "'k'", (), ("[key]",)),
    }
    assert held("T6", "chosen") == {
        ("parameter", "taskSpec", (".first",), ()),
        ("parameter", "taskSpec", (".second",), ()),
    }
    assert held("T7", "item") == {
        ("unknown", "taskSpec[*]", (), ()),
        ("self_field", "self.items", ("[*]",), ()),
    }
    assert held("T8", "g") == {
        ("parameter", "jobSpec", (".Files", "[*]"), ()),
        ("parameter", "jobSpec", (".PandaID",), ()),
    }
    assert held("T9", "spec") == {("self_field", "self.specs", ("['a']",), ())}


DICT_KEYS = {
    BASE: """\
        class TaskRefinerBase:
            def __init__(self):
                self.a = None

            def extractCommon(self, ddm_if, dataset_name):
                replica_map = {dataset_name: ddm_if.listDatasetReplicas(dataset_name)}
                info = {}
                for rse, data_list in replica_map[dataset_name].items():
                    flag = False
                    for data in data_list:
                        if data["found"]:
                            flag = True
                    info[rse] = {"flag": flag}
                for rse in info:
                    if info[rse]["flag"]:  # T1
                        self.a = 1  # R1
                for name in {"x": 1, "y": 2}:
                    if name:  # T2
                        self.a = 2  # R2
                for value in [3, 4]:
                    if value:  # T3
                        self.a = 3  # R3
        """,
}


def test_a_loop_over_a_value_known_as_a_dict_takes_its_keys_not_what_its_values_hold(
    tmp_path: Path,
) -> None:
    """``get_sites_with_data``'s shape: ``replica_availability_info[tmp_rse] = {...}`` puts a dict under a key that is
    not a constant, so its value sits at ``[*]``, where a list's element sits too; ``for tmp_rse in
    replica_availability_info`` read the dict put as ``tmp_rse`` -- its ``False`` / ``True`` and its key ``'flag'``.
    A loop over a value known as a dict (the path carry's ``_is_dict``) takes its keys alone: a put's key is kept as
    its spelling only, so ``rse`` is the empty dict's element, a value no way reaches; a dict literal's keys are
    ``'x'`` and ``'y'``, not its values too; a list's elements are as before."""
    arms = tuple(("extractCommon", f"R{n}") for n in (1, 2, 3))
    fragment = _build(tmp_path, DICT_KEYS, arms)

    def held(marker: str, name: str) -> set[tuple]:
        (guard,) = [
            one
            for one in fragment.guard_froms
            if one.test_line == _line(BASE, marker, DICT_KEYS) and one.name == name
        ]
        return {
            (one.kind, one.name, tuple(one.steps), tuple(one.paths))
            for one in guard.sources
        }

    assert held("T1", "rse") == {("unknown", "{}", ("[*]",), ())}
    assert held("T2", "name") == {
        ("constant", "'x'", (), ()),
        ("constant", "'y'", (), ()),
    }
    assert held("T3", "value") == {
        ("constant", "3", (), ()),
        ("constant", "4", (), ()),
    }


SITE_DATA = "pandaserver/taskbuffer/db_proxy_mods/task_module.py"
KEYED_PUTS = {
    BASE: """\
        from pandaserver.taskbuffer.db_proxy_mods.task_module import Checker


        class TaskRefinerBase:
            def __init__(self):
                self.a = None

            def extractCommon(self, scanSiteList):
                nWNmap = Checker().getCurrentSiteData()
                for tmpSiteName in scanSiteList:
                    one = nWNmap[tmpSiteName]["getJob"]
                    if one == 0:  # T1
                        self.a = 1  # R1
                    other = nWNmap[tmpSiteName]["flag"]
                    if other:  # T2
                        self.a = 2  # R2
        """,
    SITE_DATA: """\
        class Checker:
            def getCurrentSiteData(self):
                sql = "SELECT siteid,getJob,updateJob FROM ATLAS_PANDAMETA.SiteData WHERE HOURS=3 "
                self.cur.execute(sql, {})
                res = self.cur.fetchall()
                ret = {}
                for site, getJob, updateJob in res:
                    ret[site] = {"getJob": getJob, "updateJob": updateJob}
                return ret
        """,
}


def test_a_step_cutting_away_some_of_a_value_s_ways_leaves_them_out_and_falls_back_once_where_it_cuts_away_all(
    tmp_path: Path,
) -> None:
    """``doBrokerage``'s ``nPilot``: the callee puts ``{"getJob": getJob, "updateJob": updateJob}`` into ``ret``
    under each site, and the caller reads ``nWNmap[tmpSiteName]["getJob"]``.  The step ``['getJob']`` takes the
    local ``getJob`` (the rows' second column) and cuts away the literal's keys and its other value.  Each way cut
    away was read as a value no way reaches -- ``'getJob'['getJob']``, ``'updateJob'['getJob']``,
    ``self.cur.fetchall[*][2]['getJob']`` --; now they are left out, and only where the step leaves none of the
    value's ways (``["flag"]``, which no put holds) is the part a value no way reaches, once: here the dict's own
    element, which a put under a key not read may hold."""
    arms = (("extractCommon", "R1"), ("extractCommon", "R2"))
    fragment = _build(tmp_path, KEYED_PUTS, arms)

    def held(marker: str, name: str) -> set[tuple]:
        (guard,) = [
            one
            for one in fragment.guard_froms
            if one.test_line == _line(BASE, marker, KEYED_PUTS) and one.name == name
        ]
        return {
            (one.kind, one.name, tuple(one.steps), tuple(one.paths))
            for one in guard.sources
        }

    assert held("T1", "one") == {
        ("database", "self.cur.fetchall", ("[*]", "[1]"), ()),
        ("unknown", "{}", ("[*]", "['getJob']"), ()),
    }
    assert held("T2", "other") == {("unknown", "{}", ("[*]", "['flag']"), ())}


VIEWS = {
    BASE: """\
        class TaskRefinerBase:
            def __init__(self):
                self.specs = {}

            def extractCommon(self, jediTaskID, taskSpec, jobSpec):
                for spec in self.specs.values():
                    if spec.site == "a":  # T1
                        self.a = 1  # R1
                chosen = list(self.specs.values())
                if chosen[0] == "b":  # T2
                    self.b = 1  # R2
                for key, value in taskSpec.params.items():
                    if key == "c" and value == "d":  # T3
                        self.c = 1  # R3
                kept = jobSpec.Files.copy()
                if kept[0].lfn == "e":  # T4
                    self.e = 1  # R4
                pair = taskSpec.params.popitem()
                if pair[1] == "f":  # T5
                    self.f = 1  # R5
                names = list(self.specs.keys())
                if names[0] == "g":  # T6
                    self.g = 1  # R6
                local = {"k": jobSpec}
                for one in local.values():
                    if one.PandaID == 1:  # T7
                        self.h = 1  # R7
        """,
}


def test_a_view_or_a_copy_of_a_container_holds_its_values_where_callfacts_says_and_none_of_its_keys(
    tmp_path: Path,
) -> None:
    """``d.values()`` and ``xs.copy()`` hold the container's values at ``[*]``, ``d.items()`` at each pair's
    second, ``d.popitem()`` at its second (callfacts.Give, as the path carry reads them): a loop over a view takes
    the container's value.  A key is not read -- what is put records its spelling, not its ways --: a pair's first is
    a value not read, a view of the keys a call not read, and a dict literal's key is not one of its values (T7)."""
    arms = tuple(("extractCommon", f"R{n}") for n in range(1, 8))
    fragment = _build(tmp_path, VIEWS, arms)

    def held(marker: str, name: str) -> set[tuple]:
        (guard,) = [
            one
            for one in fragment.guard_froms
            if one.test_line == _line(BASE, marker, VIEWS) and one.name == name
        ]
        return {
            (one.kind, one.name, tuple(one.steps), tuple(one.paths))
            for one in guard.sources
        }

    assert held("T1", "spec") == {("self_field", "self.specs", ("[*]",), ())}
    assert held("T2", "chosen") == {("self_field", "self.specs", ("[*]",), ("[*]",))}
    assert held("T3", "key") == {("unknown", "taskSpec.params[*][0]", (), ())}
    assert held("T3", "value") == {("parameter", "taskSpec", (".params", "[*]"), ())}
    assert held("T4", "kept") == {("parameter", "jobSpec", (".Files", "[*]"), ("[*]",))}
    assert held("T5", "pair") == {
        ("parameter", "taskSpec", (".params", "[*]"), ("[1]",))
    }
    assert {kind for kind, *_rest in held("T6", "names")} == {"call"}
    assert held("T7", "one") == {("parameter", "jobSpec", (), ())}


JOB_MODULE = "pandaserver/taskbuffer/db_proxy_mods/job_module.py"
JOINED = {
    JOB_MODULE: """\
        comment = " /* x */"


        class JobModule:
            def listReports(self, limit):
                sql = "SELECT PandaID,jobStatus,attemptNr FROM ATLAS_PANDA.Job_Output_Report WHERE rownum<=:limit "
                self.cur.execute(sql + comment, {":limit": limit})
                retVal = self.cur.fetchall()
                head = retVal[: limit // 10]
                if head == []:  # T1
                    self.a = 1  # R1
                retVal = head + retVal[limit // 10 :]
                retVal = retVal[:limit]
                for one in retVal:
                    if one[2] == 3:  # T2
                        self.b = 1  # R2
                    tries = one[2] + 1
                    if tries == 4:  # T3
                        self.c = 1  # R3
                total = 0
                for one in retVal:
                    total += one[2]
                if total == 5:  # T4
                    self.d = 1  # R4
        """,
}


def test_a_slice_is_the_value_itself_and_a_plus_holds_its_sides_whose_element_is_a_side_s(
    tmp_path: Path,
) -> None:
    """``listJobOutputReport``'s shape: a slice of the rows is the rows (T1), not a row; the rows joined of two
    slices and sliced again are the rows, a row of them a row (T2).  A ``+`` holds each side's ways through the
    ``+`` -- ``one[2] + 1`` computed from the row's ``[2]`` and ``1`` (T3) --; an element of its value is a side's,
    the ``+`` left.  A ``+=`` is a ``+`` of what the name held and its value, each ``+=`` once in a loop (T4)."""
    arms = tuple(("listReports", f"R{n}") for n in range(1, 5))
    fragment = _build(tmp_path, JOINED, arms, base=JOB_MODULE)

    def held(marker: str, name: str) -> set[tuple]:
        (guard,) = [
            one
            for one in fragment.guard_froms
            if one.test_line == _line(JOB_MODULE, marker, JOINED) and one.name == name
        ]
        return {
            (
                one.kind,
                one.name,
                tuple(one.steps),
                tuple(one.paths),
                tuple((t.call, t.gives, t.after_steps) for t in one.through),
            )
            for one in guard.sources
        }

    added = ("+", "added")
    assert held("T1", "head") == {("database", "self.cur.fetchall", (), (), ())}
    assert held("T2", "one") == {("database", "self.cur.fetchall", ("[*]",), (), ())}
    assert held("T3", "tries") == {
        ("database", "self.cur.fetchall", ("[*]", "[2]"), (), ((*added, 2),)),
        ("constant", "1", (), (), ((*added, 0),)),
    }
    assert held("T4", "total") == {
        ("constant", "0", (), (), ()),
        ("constant", "0", (), (), ((*added, 0),)),
        ("database", "self.cur.fetchall", ("[*]", "[2]"), (), ((*added, 2),)),
    }


BINDINGS = {
    BASE: """\
        class TaskRefinerBase:
            def __init__(self):
                self.flag = None

            def extractCommon(self, jediTaskID, taskSpec, jobSpec):
                if jobSpec:  # T2
                    modTime = taskSpec.modificationTime
                else:
                    modTime = 0  # B3
                if (jediTaskID - modTime) > 24:  # T0
                    ignore = False  # B1
                else:
                    ignore = True  # B2
                self.check(ignore)
                self.check(self.chosen(jediTaskID))

            def chosen(self, value):
                if value:  # T4
                    return "a"  # B4
                else:
                    return "b"  # B5

            def check(self, ignore_error, strict=False):  # B6
                if ignore_error or strict:  # T1
                    self.a = 1  # R1
        """,
}


def test_the_tests_on_the_way_to_a_source_s_own_statement_have_guards_on_through_their_own_sources(
    tmp_path: Path,
) -> None:
    """Which of the constants one call hands in a run made is told by the test on the way to its binding
    (``add_main``'s ``if (timeNow - modTime) > ...: ignoreTmpError = False``): each source says where it ends
    (:attr:`HeldSource.at`) and the tests on the way there have GuardFroms, its line among their *calls* -- in the
    callee for a value a return hands back (T4), and on through those guards' own sources (T2 guards ``modTime``'s
    ``0``).  A parameter's default the call leaves it to is where it is written, in the callee."""
    fragment = _build(tmp_path, BINDINGS, (("check", "R1"),))

    def at(marker: str) -> str:
        return f"{BASE}:{_line(BASE, marker, BINDINGS)}"

    def ends(func: str, parameter: str) -> set[tuple]:
        (record,) = [
            one
            for one in fragment.parameter_froms
            if one.owner.endswith(f"::{func}") and one.parameter == parameter
        ]
        return {(one.kind, one.name, tuple(one.at)) for one in record.sources}

    assert ends("check", "ignore_error") == {
        ("constant", "False", (at("B1"),)),
        ("constant", "True", (at("B2"),)),
        ("constant", "'a'", (at("B4"),)),
        ("constant", "'b'", (at("B5"),)),
    }
    assert ends("check", "strict") == {("constant", "False", (at("B6"),))}
    # A parameter is bound where the function is entered, by no statement.
    assert ends("chosen", "value") == {("parameter", "jediTaskID", ())}

    def calls(func: str, marker: str) -> set[int]:
        guards = [
            one
            for one in fragment.guard_froms
            if one.owner.endswith(f"::{func}")
            and one.test_line == _line(BASE, marker, BINDINGS)
        ]
        assert guards, (func, marker)
        return {line for one in guards for line in one.calls}

    def line(marker: str) -> int:
        return _line(BASE, marker, BINDINGS)

    assert calls("extractCommon", "T0") == {line("B1"), line("B2")}
    assert calls("chosen", "T4") == {line("B4"), line("B5")}
    assert calls("extractCommon", "T2") == {line("B3")}


CHECKER = "pandajedi/jedicore/Checker.py"
DEFAULTED = {
    BASE: """\
        from pandajedi.jedicore.Checker import Checker


        class TaskRefinerBase:
            def extractCommon(self, jediTaskID):
                checker = Checker()
                checker.look(jediTaskID)
                checker.look(jediTaskID, use_json=True)  # C1
        """,
    CHECKER: """\
        class Checker:
            def __init__(self):
                self.a = None

            def look(self, jobs, use_json=False):  # D1
                if use_json:  # T1
                    self.a = 1  # R1
        """,
}


def test_a_parameter_s_default_is_placed_where_it_is_written_not_in_the_caller_s_file(
    tmp_path: Path,
) -> None:
    """``peekJobs``' ``use_json: bool = False``, which a call in ``copyArchive.main`` leaves to its default: the
    default ends at its line in the callee's file, though the caller's function spans the same line number in its
    own file."""
    fragment = _build(tmp_path, DEFAULTED, (("look", "R1"),), base=CHECKER)
    (record,) = [
        one
        for one in fragment.parameter_froms
        if one.owner.endswith("::look") and one.parameter == "use_json"
    ]
    assert _line(CHECKER, "D1", DEFAULTED) == 5
    assert 5 in range(5, _line(BASE, "C1", DEFAULTED) + 1)
    assert {(one.kind, one.name, tuple(one.at)) for one in record.sources} == {
        ("constant", "False", (f"{CHECKER}:5",)),
        ("constant", "True", (f"{BASE}:{_line(BASE, 'C1', DEFAULTED)}",)),
    }


EXCEPTIONS = "pandaserver/srvcore/exceptions.py"
DDM = "pandaserver/dataservice/ddm.py"
PLUGIN = "pandaserver/dataservice/adder_atlas_plugin.py"
CAUGHT = {
    EXCEPTIONS: """\
        class FileRegistrationError(Exception):
            def __init__(self, message: str = "", fatal: bool = False):  # E1
                super().__init__(message)
                self.fatal = fatal
        """,
    DDM: """\
        from pandaserver.srvcore.exceptions import FileRegistrationError


        class RucioAPI:
            def register(self, files):
                if not files:
                    raise FileRegistrationError("no files", fatal=True)  # F1
                raise FileRegistrationError("failed")  # F2
        """,
    PLUGIN: """\
        from pandaserver.dataservice.ddm import RucioAPI
        from pandaserver.srvcore.exceptions import FileRegistrationError


        class AdderAtlasPlugin:
            def __init__(self):
                self.result = None

            def register_files(self, files):
                try:
                    RucioAPI().register(files)
                except FileRegistrationError as e:  # H1
                    is_fatal = e.fatal
                    if is_fatal:  # T1
                        self.result = 1  # R1
                except Exception as e:
                    if e.args:  # T2
                        self.result = 2  # R2
        """,
}


def test_a_name_a_handler_binds_is_an_object_of_the_class_caught_whose_field_is_read_from_the_raises(
    tmp_path: Path,
) -> None:
    """``AdderAtlasPlugin.register_files``' ``except FileRegistrationError as e: is_fatal = e.fatal``: ``e`` is an
    object of the class, placed at the handler, so ``fatal`` is read where the class sets it -- the ``__init__``'s
    parameter, handed ``True`` by one raise and left to its default by the other.  ``except Exception as e`` names
    no class of the corpus: no object."""
    fragment = _build(
        tmp_path,
        CAUGHT,
        (("register_files", "R1"), ("register_files", "R2")),
        base=PLUGIN,
    )
    (fatal,) = [
        one
        for one in fragment.guard_froms
        if one.test_line == _line(PLUGIN, "T1", CAUGHT)
    ]
    assert fatal.name == "is_fatal"
    assert {
        (one.kind, one.name, tuple(one.steps), one.site_owner, tuple(one.at))
        for one in fatal.sources
    } == {
        (
            "object",
            "FileRegistrationError",
            (".fatal",),
            f"{EXCEPTIONS}::FileRegistrationError",
            (f"{PLUGIN}:{_line(PLUGIN, 'H1', CAUGHT)}",),
        )
    }
    (field,) = [
        one
        for one in fragment.field_froms
        if one.owner == f"{EXCEPTIONS}::FileRegistrationError" and one.field == "fatal"
    ]
    assert {
        (source.kind, source.name)
        for binding in field.bindings
        for source in binding.sources
    } == {("parameter", "fatal")}
    (handed,) = [
        one
        for one in fragment.parameter_froms
        if one.owner == f"{EXCEPTIONS}::__init__" and one.parameter == "fatal"
    ]
    assert {(one.kind, one.name, tuple(one.at)) for one in handed.sources} == {
        ("constant", "True", (f"{DDM}:{_line(DDM, 'F1', CAUGHT)}",)),
        ("constant", "False", (f"{EXCEPTIONS}:{_line(EXCEPTIONS, 'E1', CAUGHT)}",)),
    }
    (other,) = [
        one
        for one in fragment.guard_froms
        if one.test_line == _line(PLUGIN, "T2", CAUGHT)
    ]
    assert "object" not in {one.kind for one in other.sources}


WALRUS = {
    BASE: """\
        class TaskRefinerBase:
            def __init__(self):
                self.a = None

            def extractCommon(self, ds_list_dict, f):
                if pseudo := ds_list_dict["pseudo"]:  # T1
                    self.a = 1  # R1
                if (n := f.count()) is not None and n == 0:  # T2
                    self.a = 2  # R2
                if m := f.other():  # T3
                    if m > 3:  # T4
                        self.a = 3  # R3
        """,
}


def test_a_walrus_is_its_value_where_its_own_test_reads_it_and_after(
    tmp_path: Path,
) -> None:
    """``if pseudo_coll_list := ds_list_dict["pseudo_coll_list"]:`` in ``TaskRefiner.runImpl``: the guard reads the
    name the test binds -- at the walrus itself, or after it in the same test (``nStaging == 0``) -- as its value;
    bindings are read by statements, which once left it unbound there.  A later statement reads it the same."""
    fragment = _build(
        tmp_path,
        WALRUS,
        (("extractCommon", "R1"), ("extractCommon", "R2"), ("extractCommon", "R3")),
    )

    def ways(marker: str, name: str) -> set[tuple]:
        (guard,) = [
            one
            for one in fragment.guard_froms
            if one.test_line == _line(BASE, marker, WALRUS) and one.name == name
        ]
        assert guard.stops == []
        return {
            (one.kind, one.name, tuple(one.steps), tuple(one.at))
            for one in guard.sources
        }

    def at(marker: str) -> tuple[str]:
        return (f"{BASE}:{_line(BASE, marker, WALRUS)}",)

    assert ways("T1", "pseudo") == {("parameter", "ds_list_dict", ("['pseudo']",), ())}
    assert ways("T2", "n") == {("call", "f.count", (), at("T2"))}
    assert ways("T3", "m") == {("call", "f.other", (), at("T3"))}
    assert ways("T4", "m") == {("call", "f.other", (), at("T3"))}


MERGE = "pandaserver/taskbuffer/db_proxy_mods/job_module.py"
RETURNED = {
    BASE: """\
        from pandaserver.taskbuffer.db_proxy_mods.job_module import JobModule


        class TaskRefinerBase:
            def __init__(self):
                self.a = None

            def extractCommon(self, pandaID):
                isValid, tmpMsg = JobModule().viaName(pandaID)
                if isValid is False:  # T1
                    self.a = 1  # R1
                pair = JobModule().isValidMergeJob(pandaID)
                if pair:  # T2
                    self.a = 2  # R2
        """,
    MERGE: """\
        class JobModule:
            def isValidMergeJob(self, pandaIDs):
                try:
                    retVal = True  # V1
                    retMsg = ""  # M1
                    sqlC = "SELECT jobStatus FROM ATLAS_PANDA.jobsActive4 WHERE PandaID=:PandaID "
                    for tmpPandaID in pandaIDs:
                        varMap = {}
                        varMap[":PandaID"] = tmpPandaID
                        self.cur.execute(sqlC, varMap)
                        resC = self.cur.fetchone()
                        if resC[0] != "merging":
                            retVal = False  # V2
                            retMsg = resC[0]  # M2
                            break
                    return retVal, retMsg
                except Exception:
                    return None, ""  # E1

            def viaName(self, pandaIDs):
                ret = self.isValidMergeJob(pandaIDs)
                return ret
        """,
}


def test_a_returned_tuple_holding_a_row_s_value_holds_its_other_elements_too(
    tmp_path: Path,
) -> None:
    """``isValid, tmpMsg = taskBuffer.isValidMergeJob(pandaID, jediTaskID)`` in ``copyArchive.main``: the proxy's
    ``return retVal, retMsg``, handed back through ``TaskBuffer``'s ``ret``, holds ``retVal``'s ``True`` and
    ``False`` at ``[0]`` though ``retMsg`` holds a row's value -- a literal is read by its elements before the carry
    is asked whether it is rows, which once took the tuple whole for the row and left only the handler's ``None``."""
    fragment = _build(
        tmp_path, RETURNED, (("extractCommon", "R1"), ("extractCommon", "R2"))
    )

    def ways(marker: str) -> set[tuple]:
        (guard,) = [
            one
            for one in fragment.guard_froms
            if one.owner.startswith(f"{BASE}::")
            and one.test_line == _line(BASE, marker, RETURNED)
        ]
        return {(one.kind, one.name, tuple(one.at)) for one in guard.sources}

    def at(*markers: str) -> tuple[str, ...]:
        return tuple(f"{MERGE}:{_line(MERGE, marker, RETURNED)}" for marker in markers)

    assert ways("T1") == {
        ("constant", "True", at("V1")),
        ("constant", "False", at("V2")),
        ("constant", "None", at("E1")),
    }
    assert {(kind, name) for kind, name, _at in ways("T2")} >= {
        ("constant", "True"),
        ("constant", "False"),
        ("constant", "''"),
        ("constant", "None"),
    }
    assert "rows" in {kind for kind, _name, _at in ways("T2")}


UNBOUND = {
    BASE: """\
        SETTING = None


        class TaskRefinerBase:
            def __init__(self):
                self.a = None

            def extractCommon(self, rows, flag):
                global SETTING
                try:
                    retVal: list | None = None  # B1
                    retVal = rows.fetch()
                except Exception:
                    if retVal:  # T1
                        self.a = 1  # R1
                if flag:
                    mode = "x"  # B3
                if mode == "x":  # T2
                    self.a = 2  # R2
                if SETTING:  # T3
                    self.a = 3  # R3
                SETTING = 1  # B5
                if SETTING:  # T5
                    self.a = 5  # R5
                if flag:
                    import json as codec
                else:
                    codec = None  # B4
                if codec:  # T4
                    self.a = 4  # R4
                from os import sep as separator
                if separator:  # T6
                    self.a = 6  # R6
        """,
}


def test_the_entry_reaching_a_local_gives_no_way_but_a_binding_not_read_is_a_stop(
    tmp_path: Path,
) -> None:
    """A local read where no binding of it may have run -- ``return retVal`` in the handler of a ``try`` whose first
    statement binds it (``listJobOutputReport``), ``mode`` bound under one test and read under another -- gets no
    value by that way: the read raises, or the way is not taken.  A name declared ``global`` is the module's by
    that way, and its binding's where the binding reaches the read.  A name an import may bind is the function's
    own, an import its only binding too; it is read as far as the bindings read, and a stop says the import."""
    fragment = _build(
        tmp_path, UNBOUND, tuple(("extractCommon", f"R{n}") for n in range(1, 7))
    )

    def guard(marker: str) -> GuardFrom:
        (found,) = [
            one
            for one in fragment.guard_froms
            if one.test_line == _line(BASE, marker, UNBOUND) and one.name != "flag"
        ]
        return found

    def at(marker: str) -> tuple[str]:
        return (f"{BASE}:{_line(BASE, marker, UNBOUND)}",)

    def ways(marker: str) -> set[tuple]:
        return {(one.kind, one.name, tuple(one.at)) for one in guard(marker).sources}

    assert ways("T1") == {("constant", "None", at("B1"))}
    assert ways("T2") == {("constant", "'x'", at("B3"))}
    assert ways("T3") == {("global", "SETTING", at("T3"))}
    assert ways("T4") == {("constant", "None", at("B4"))}
    assert ways("T5") == {("constant", "1", at("B5"))}
    assert ways("T6") == set()
    for marker in ("T1", "T2", "T3", "T5"):
        assert guard(marker).stops == []
    assert [(one.kind, one.name, one.what, one.at) for one in guard("T4").stops] == [
        ("unread_binding", "codec", "an import", list(at("T4")))
    ]
    assert [(one.kind, one.name, one.what, one.at) for one in guard("T6").stops] == [
        ("unread_binding", "separator", "an import", list(at("T6")))
    ]


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


_FACTORY = """\
    from typing import Generic, TypeVar

    T = TypeVar("T")


    class FactoryBase(Generic[T]):
        def __init__(self) -> None:
            self.implMap: dict[str, T] = {}

        def getImpl(self, vo: str) -> T:
            tmpImpl = self.implMap[vo]
            tmpImpl.refresh()
            return tmpImpl

        def requireImpl(self, vo: str) -> T:
            impl = self.getImpl(vo)
            return impl
"""

_PLUGINS = """\
    class BrokerBase:
        def refresh(self) -> None:
            self.siteMapper = 1  # B


    class DogBase:
        def refresh(self) -> None:
            self.lastRun = 2  # D
"""

_KNIGHTS = """\
    from pkg.factory import FactoryBase
    from pkg.plugins import BrokerBase, DogBase


    class Broker(FactoryBase[BrokerBase]):
        pass


    class WatchDog(FactoryBase[DogBase]):
        pass


    def run(broker: Broker, dog: WatchDog) -> None:
        brokered = broker.requireImpl("atlas")
        watched = dog.requireImpl("atlas")
        if brokered and watched:
            pass
"""


def test_a_way_through_a_factory_reads_the_changes_of_the_family_the_factory_hands(
    tmp_path: Path,
) -> None:
    """``jobBroker.requireImpl(...)`` on a ``JobBroker(FactoryBase[JobBrokerBase])``: ``FactoryBase.getImpl`` runs
    ``tmpImpl.refresh()`` before it returns the plugin, and that is ``JobBrokerBase.refresh`` -- read with ``self`` a
    ``JobBroker`` there, the way's changes are that family's stores, not every family's (FGT)."""
    root = tmp_path / "pkg"
    root.mkdir()
    (root / "__init__.py").write_text("")
    for name, text in (("factory", _FACTORY), ("plugins", _PLUGINS), ("m", _KNIGHTS)):
        (root / f"{name}.py").write_text(textwrap.dedent(text))
    calls = calls_mod.Calls(calls_mod.Types({"pkg": root}, {}))
    trees = {
        f"pkg/{name}.py": calls.types.tree(f"pkg/{name}.py")
        for name in ("factory", "plugins", "m")
    }
    run = next(
        n
        for n in ast.walk(trees["pkg/m.py"])
        if isinstance(n, ast.FunctionDef) and n.name == "run"
    )
    test = next(n for n in ast.walk(run) if isinstance(n, ast.If))

    class _Record:
        def is_database(self, file: str, node: ast.AST) -> bool:
            return False

    def changed(name: str) -> set[str]:
        node = next(
            n for n in ast.walk(test.test) if isinstance(n, ast.Name) and n.id == name
        )
        with sql.across_calls(calls):
            reader = heldfrom._Reader(calls, _Record(), [], set(), trees)
            ways = reader.read(node, test, "pkg/m.py", run)
        return {change.at for way in ways for change in way.changes}

    plugins = textwrap.dedent(_PLUGINS).splitlines()
    store = {
        mark: f"pkg/plugins.py:{plugins.index(next(t for t in plugins if t.endswith(mark))) + 1}"
        for mark in ("# B", "# D")
    }
    assert changed("brokered") == {store["# B"]}
    assert changed("watched") == {store["# D"]}
