"""Where each function holding an arm is entered from -- ``codemap.panda.entered`` -- on the shape of JEDI's job
generator: ``doGenerate`` calls ``doGeneratePrePro`` under ``taskSpec.usePrePro()`` and ``doGenerateBuild`` under
``elif taskSpec.useBuild()``, each from that one place."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from bamboo.codemap import callfacts, reach
from bamboo.codemap import calls as calls_mod
from bamboo.codemap.gitsource import blob_sha
from bamboo.codemap.models import (
    ENTERED_HOLE,
    ENTERED_NAMED_AT_RUN_TIME,
    ENTERED_TAKEN,
    Anchor,
    Branch,
    EnteredFrom,
    GuardFrom,
    JunctionNode,
    MapFragment,
    SourceModule,
)
from bamboo.codemap.panda import entered, heldfrom, runnable, sql
from bamboo.codemap.panda.plugin import _DatabaseLeaves
from bamboo.codemap.panda.recognizers import sqlwrite

SPEC = "pandaserver/taskbuffer/JediTaskSpec.py"
GEN = "pandajedi/jediorder/JobGenerator.py"
PROXY = "pandaserver/taskbuffer/OraDBProxy.py"
BRIDGE = "pandaserver/taskbuffer/ConBridge.py"
METRIC = "pandaserver/daemons/scripts/metric_collector.py"
UTILS = "pandaserver/taskbuffer/EventServiceUtils.py"
LOADER = "pandaserver/userinterface/Loader.py"
FACTORY = "pandajedi/jedicore/FactoryBase.py"
KNIGHT = "pandajedi/jediorder/TaskRefiner.py"
SERVER = "pandaserver/server/panda.py"

SOURCES = {
    SPEC: """\
        class JediTaskSpec:
            def __init__(self) -> None:
                self.splitRule = ""
                self.attemptNr = 0

            def usePrePro(self) -> bool:
                return "PP" in self.splitRule

            def useBuild(self) -> bool:
                return "BU" in self.splitRule

            def setOnHold(self) -> None:
                self.attemptNr = 1  # H1
        """,
    GEN: """\
        from pandaserver.taskbuffer.JediTaskSpec import JediTaskSpec


        class JobGeneratorThread:
            def doGenerate(self, taskSpec: JediTaskSpec, readStat: int) -> None:
                if not readStat:
                    return None
                if taskSpec.usePrePro():
                    self.doGeneratePrePro(taskSpec)  # C-PP
                elif taskSpec.useBuild():  # T-BU
                    self.doGenerateBuild(taskSpec)  # C-BU
                if readStat == 2:
                    taskSpec.setOnHold()
                return None

            def doGeneratePrePro(self, taskSpec: JediTaskSpec) -> None:
                taskSpec.attemptNr = 2  # P1

            def doGenerateBuild(self, taskSpec: JediTaskSpec) -> None:
                taskSpec.attemptNr = 3  # B1


        def hold(obj, flag: bool) -> None:
            if flag:
                obj.setOnHold()  # HOLE
        """,
    PROXY: """\
        from pandaserver.taskbuffer.JediTaskSpec import JediTaskSpec


        class DBProxy:
            def retryTask(self, taskSpec: JediTaskSpec, again: bool) -> None:
                if again:
                    self._reset(taskSpec)

            def _reset(self, taskSpec: JediTaskSpec) -> None:
                taskSpec.attemptNr = 4  # R1
        """,
    BRIDGE: """\
        from pandaserver.taskbuffer.OraDBProxy import DBProxy


        class ConBridge:
            def __init__(self) -> None:
                self.proxy = DBProxy()

            def bridge_run(self, comStr: str) -> None:
                method = getattr(self.proxy, comStr)  # FAR
                method()
        """,
    METRIC: """\
        from pandaserver.taskbuffer.JediTaskSpec import JediTaskSpec


        class FetchData:
            def __init__(self, taskSpec: JediTaskSpec) -> None:
                self.taskSpec = taskSpec

            def wait_time(self) -> None:
                self.taskSpec.attemptNr = 5  # F1


        def main(taskSpec: JediTaskSpec, metric_name: str, dry: bool) -> None:
            fetcher = FetchData(taskSpec)
            if dry:
                fetcher.wait_time()
            the_method = getattr(fetcher, metric_name)  # GETATTR
            the_method()


        def on_done(taskSpec: JediTaskSpec) -> None:
            taskSpec.attemptNr = 6  # D1


        def register(taskSpec: JediTaskSpec, done: bool) -> dict:
            if done:
                on_done(taskSpec)
            return {"done": on_done}  # TAKEN
        """,
}

#: A plugin loader: a ``getattr`` off a module may take any of a module's names.
MODULE_SOURCES = {
    UTILS: """\
        from pandaserver.taskbuffer.JediTaskSpec import JediTaskSpec


        def setMerge(taskSpec: JediTaskSpec) -> None:
            taskSpec.attemptNr = 7  # M1


        def merge(taskSpec: JediTaskSpec, go: bool) -> None:
            if go:
                setMerge(taskSpec)


        class Merger:
            def apply(self, taskSpec: JediTaskSpec) -> None:
                taskSpec.attemptNr = 8  # A1

            def run(self, taskSpec: JediTaskSpec, go: bool) -> None:
                if go:
                    self.apply(taskSpec)
        """,
    LOADER: """\
        import importlib

        api = importlib.import_module("pandaserver.api")
        exposed = getattr(api, "pandaserver".upper())  # BODY

        def instantiate(moduleName: str, className: str) -> None:
            mod = __import__(moduleName)
            for subModuleName in moduleName.split(".")[1:]:
                mod = getattr(mod, subModuleName)
            cls = getattr(mod, className)  # PLUGIN
            cls()
        """,
}

ARMS = (
    (GEN, "doGeneratePrePro", "P1"),
    (GEN, "doGenerateBuild", "B1"),
    (SPEC, "setOnHold", "H1"),
    (PROXY, "_reset", "R1"),
    (METRIC, "wait_time", "F1"),
    (METRIC, "on_done", "D1"),
)


def _line(sources: dict[str, str], rel: str, marker: str) -> int:
    (found,) = [
        number
        for number, text in enumerate(textwrap.dedent(sources[rel]).splitlines(), 1)
        if text.rstrip().endswith(f"# {marker}")
    ]
    return found


def _build(tmp_path: Path, sources: dict[str, str], arms: tuple[tuple[str, str, str], ...]) -> MapFragment:
    """Read the sources as the build does, with a junction for each arm marked, then the ways in and their guards."""
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
    roots = {"pandaserver": tmp_path / "pandaserver", "pandajedi": tmp_path / "pandajedi"}
    calls = sqlwrite.corpus_calls(modules, roots)
    assert calls is not None
    junctions = [
        JunctionNode(
            name=f"m:{rel}:{owner}:{marker}",
            map_id="m",
            derived_from="v",
            subject="JediTaskSpec.attemptNr",
            owner=f"{rel}::{owner}",
            anchor=Anchor(package=rel.split("/", 1)[0], file=rel, line_start=_line(sources, rel, marker)),
            branches=[Branch(outcome=marker, line=_line(sources, rel, marker))],
        )
        for rel, owner, marker in arms
    ]
    trees = {module.rel_path: module.tree for module in modules}
    with sql.across_calls(calls):
        record = reach.ReachGraph(reach.graph(calls, "m", "v", _DatabaseLeaves()))
        ways = entered.entered_froms(junctions, trees, calls, record)
        guards, parameters, _writes, _written, _fields = heldfrom.guard_froms(junctions, [], trees, calls, record, entered=ways)
        runs = runnable.runnables(junctions, trees, calls, record, guards, parameters, entered=ways)
    return MapFragment(
        map_id="m",
        derived_from="v",
        junctions=junctions,
        guard_froms=guards,
        parameter_froms=parameters,
        runnables=runs,
        entered_froms=ways,
    )


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> MapFragment:
    return _build(tmp_path_factory.mktemp("entered"), SOURCES, ARMS)


def _way(fragment: MapFragment, rel: str, name: str) -> EnteredFrom:
    (found,) = [one for one in fragment.entered_froms if one.owner == f"{rel}::{name}"]
    return found


def test_an_arm_s_function_called_from_one_place_says_the_tests_the_call_sits_under(built: MapFragment) -> None:
    way = _way(built, GEN, "doGeneratePrePro")
    assert way.arms == [_line(SOURCES, GEN, "P1")]
    assert way.open == []
    ((site),) = way.sites
    assert (site.owner, site.line, site.how) == (f"{GEN}::doGenerate", _line(SOURCES, GEN, "C-PP"), "call")
    assert site.unread == ""
    assert site.conditions == ["taskSpec.usePrePro()", "not (not readStat)"]


def test_an_elif_s_call_sits_under_its_test_and_not_the_tests_before_it(built: MapFragment) -> None:
    ((site),) = _way(built, GEN, "doGenerateBuild").sites
    assert site.line == _line(SOURCES, GEN, "C-BU")
    assert site.conditions == ["not (taskSpec.usePrePro())", "taskSpec.useBuild()", "not (not readStat)"]


def test_a_caller_s_test_has_a_guard_record_naming_the_calls_it_is_on_the_way_to(built: MapFragment) -> None:
    def guard(line: int) -> GuardFrom:
        (found,) = [
            one
            for one in built.guard_froms
            if one.owner == f"{GEN}::doGenerate" and one.test_line == line and one.name == "taskSpec"
        ]
        return found

    assert guard(_line(SOURCES, GEN, "C-PP") - 1).calls == [_line(SOURCES, GEN, "C-PP"), _line(SOURCES, GEN, "C-BU")]
    assert guard(_line(SOURCES, GEN, "T-BU")).calls == [_line(SOURCES, GEN, "C-BU")]
    assert all(one.arms == [] for one in built.guard_froms if one.owner == f"{GEN}::doGenerate")


def test_a_method_a_way_in_s_test_calls_is_asked_whether_it_is_runnable(built: MapFragment) -> None:
    (use_build,) = [one for one in built.runnables if one.owner == f"{SPEC}::useBuild"]
    assert use_build.asked_at == [f"{GEN}:{_line(SOURCES, GEN, 'T-BU')}"]
    assert use_build.runnable
    assert use_build.reads == ["self.splitRule"]
    (use_pre_pro,) = [one for one in built.runnables if one.owner == f"{SPEC}::usePrePro"]
    assert use_pre_pro.asked_at == [f"{GEN}:{_line(SOURCES, GEN, 'C-PP') - 1}"]


#: one caller calling the arm's function at two lines, under different tests: the reach record holds the two calls
#: as one row (same callee, way and classes) with both lines.
TWICE = {
    SPEC: SOURCES[SPEC],
    GEN: """\
        from pandaserver.taskbuffer.JediTaskSpec import JediTaskSpec


        class JobGeneratorThread:
            def doGenerate(self, taskSpec: JediTaskSpec) -> None:
                if taskSpec.usePrePro():  # T-1
                    self.doGenerateBuild(taskSpec)  # C-1
                if taskSpec.useBuild():  # T-2
                    self.doGenerateBuild(taskSpec)  # C-2
                return None

            def doGenerateBuild(self, taskSpec: JediTaskSpec) -> None:
                taskSpec.attemptNr = 3  # B1
        """,
}


def test_a_function_called_at_two_lines_of_one_caller_has_a_site_at_each_under_its_own_tests(tmp_path: Path) -> None:
    fragment = _build(tmp_path, TWICE, ((GEN, "doGenerateBuild", "B1"),))
    way = _way(fragment, GEN, "doGenerateBuild")
    assert way.open == []
    assert [(site.line, site.conditions) for site in way.sites] == [
        (_line(TWICE, GEN, "C-1"), ["taskSpec.usePrePro()"]),
        (_line(TWICE, GEN, "C-2"), ["taskSpec.useBuild()"]),
    ]
    (second,) = [
        one
        for one in fragment.guard_froms
        if one.owner == f"{GEN}::doGenerate" and one.test_line == _line(TWICE, GEN, "T-2") and one.name == "taskSpec"
    ]
    assert second.calls == [_line(TWICE, GEN, "C-2")]
    (use_build,) = [one for one in fragment.runnables if one.owner == f"{SPEC}::useBuild"]
    assert use_build.asked_at == [f"{GEN}:{_line(TWICE, GEN, 'T-2')}"]


def test_a_hole_made_with_the_function_s_name_may_enter_it(built: MapFragment) -> None:
    way = _way(built, SPEC, "setOnHold")
    assert [(one.kind, one.at, one.text) for one in way.open] == [
        (ENTERED_HOLE, f"{GEN}:{_line(SOURCES, GEN, 'HOLE')}", "obj.setOnHold")
    ]


def test_a_forwarder_s_far_end_is_no_way_in(
    built: MapFragment, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    way = _way(built, PROXY, "_reset")
    assert way.open == []
    assert [site.owner for site in way.sites] == [f"{PROXY}::retryTask"]
    monkeypatch.setattr(callfacts, "FORWARDERS", {})
    unlisted = _way(_build(tmp_path, SOURCES, ARMS), PROXY, "_reset")
    assert [(one.kind, one.at, one.text) for one in unlisted.open] == [
        (ENTERED_NAMED_AT_RUN_TIME, f"{BRIDGE}:{_line(SOURCES, BRIDGE, 'FAR')}", "getattr(self.proxy, comStr)")
    ]


def test_a_method_a_getattr_named_at_run_time_may_take_off_its_class_may_be_entered_there(built: MapFragment) -> None:
    way = _way(built, METRIC, "wait_time")
    assert [(one.kind, one.at) for one in way.open] == [
        (ENTERED_NAMED_AT_RUN_TIME, f"{METRIC}:{_line(SOURCES, METRIC, 'GETATTR')}")
    ]


def test_a_function_whose_value_is_read_uncalled_may_be_entered_by_whoever_holds_it(built: MapFragment) -> None:
    way = _way(built, METRIC, "on_done")
    assert [(one.kind, one.at, one.text) for one in way.open] == [
        (ENTERED_TAKEN, f"{METRIC}:{_line(SOURCES, METRIC, 'TAKEN')}", "on_done")
    ]


def test_a_getattr_off_a_module_may_take_any_function_a_module_holds_and_no_method(tmp_path: Path) -> None:
    """The plugin loader walks ``__import__``'s dotted parts with ``getattr`` before it takes the class: a module."""
    arms = ((UTILS, "setMerge", "M1"), (UTILS, "apply", "A1"))
    fragment = _build(tmp_path, {SPEC: SOURCES[SPEC], **MODULE_SOURCES}, arms)
    way = _way(fragment, UTILS, "setMerge")
    assert [(one.kind, one.at) for one in way.open] == [
        (ENTERED_NAMED_AT_RUN_TIME, f"{LOADER}:{_line(MODULE_SOURCES, LOADER, 'BODY')}"),
        (ENTERED_NAMED_AT_RUN_TIME, f"{LOADER}:{_line(MODULE_SOURCES, LOADER, 'PLUGIN')}"),
    ]
    assert way.sites[0].conditions == ["go"]
    method = _way(fragment, UTILS, "apply")
    assert method.open == []
    assert method.sites[0].conditions == ["go"]


#: JEDI's plugin factory: ``FactoryBase(Generic[T])`` loads a class the configuration names, and each factory says T.
FACTORY_SOURCES = {
    FACTORY: """\
        from typing import Generic, TypeVar

        T = TypeVar("T")


        class FactoryBase(Generic[T]):
            def initializeMods(self, moduleName: str, className: str) -> None:
                mod = __import__(moduleName)
                for subModuleName in moduleName.split(".")[1:]:
                    mod = getattr(mod, subModuleName)
                cls = getattr(mod, className)  # LOAD
                cls()
        """,
    KNIGHT: """\
        from pandajedi.jedicore.FactoryBase import FactoryBase
        from pandaserver.taskbuffer.JediTaskSpec import JediTaskSpec


        class TaskRefinerBase:
            def __init__(self) -> None:
                self.taskSpec = JediTaskSpec()
                self.taskSpec.attemptNr = 9  # I1


        class GenTaskRefiner(TaskRefinerBase):
            pass


        class Other:
            def __init__(self) -> None:
                self.taskSpec = JediTaskSpec()
                self.taskSpec.attemptNr = 10  # O1


        class TaskRefiner(FactoryBase[TaskRefinerBase]):
            def make(self, go: bool) -> None:
                if go:
                    GenTaskRefiner()
                    Other()
        """,
    SERVER: """\
        def module_mapping(name: str) -> dict:
            return {"module": __import__(name)}


        def application(name: str, method_name: str) -> None:
            mapping = module_mapping(name)
            module = mapping["module"]
            tmp_method = getattr(module, method_name)  # DISPATCH
            tmp_method()
        """,
}


def test_a_plugin_factory_builds_only_a_subclass_of_what_it_says_its_type_parameter_is(tmp_path: Path) -> None:
    arms = ((UTILS, "setMerge", "M1"), (KNIGHT, "__init__", "I1"), (KNIGHT, "__init__", "O1"))
    sources = {SPEC: SOURCES[SPEC], UTILS: MODULE_SOURCES[UTILS], **FACTORY_SOURCES}
    fragment = _build(tmp_path, sources, arms)
    # Nor does the server's dispatch, which takes an endpoint's function: setMerge is no endpoint.
    assert _way(fragment, UTILS, "setMerge").open == []
    (base, other) = sorted(
        (one for one in fragment.entered_froms if one.owner == f"{KNIGHT}::__init__"), key=lambda one: one.line
    )
    assert [(one.kind, one.at) for one in base.open] == [
        (ENTERED_NAMED_AT_RUN_TIME, f"{FACTORY}:{_line(FACTORY_SOURCES, FACTORY, 'LOAD')}")
    ]
    assert other.open == []
    assert [site.conditions for site in other.sites] == [["go"]]
