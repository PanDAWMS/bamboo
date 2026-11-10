"""FieldFrom: a way ending at a field of ``self`` or of an object a class builds reads on in the class; and a callee's
``self`` is read back at the call's receiver (:mod:`bamboo.codemap.panda.heldfrom`)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from bamboo.codemap import calls as calls_mod
from bamboo.codemap import reach
from bamboo.codemap.gates import guard_fields_are_read, guard_parameters_are_read
from bamboo.codemap.gitsource import blob_sha
from bamboo.codemap.models import (
    Anchor,
    Branch,
    FieldFrom,
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

RESULT = "pandaserver/dataservice/adder_result.py"
BASE = "pandaserver/dataservice/adder_plugin_base.py"
PLUGIN = "pandaserver/dataservice/adder_atlas_plugin.py"
GEN = "pandaserver/dataservice/adder_gen.py"
HOLDER = "pandaserver/dataservice/holder.py"
SETUP_BASE = "pandaserver/dataservice/setupper_plugin_base.py"
SETUP = "pandaserver/dataservice/setupper_atlas_plugin.py"
OTHER = "pandaserver/dataservice/other.py"
BUFFER = "pandaserver/taskbuffer/task_buffer.py"
QUEUE = "pandaserver/dataservice/adder_queue.py"

SOURCES = {
    RESULT: """\
        class AdderResult:
            rc_succeeded = 0
            rc_temporary = 1
            rc_fatal = 2  # RC

            def __init__(self):
                self.status_code = None

            def set_temporary(self):
                self.status_code = self.rc_temporary  # ST

            def set_fatal(self):
                self.status_code = self.rc_fatal  # SF

            def is_succeeded(self):
                return self.status_code == self.rc_succeeded
        """,
    BASE: """\
        from pandaserver.dataservice.adder_result import AdderResult

        class AdderPluginBase:
            def __init__(self, job):
                self.job = job
                self.result = AdderResult()  # B1
        """,
    PLUGIN: """\
        from pandaserver.dataservice.adder_plugin_base import AdderPluginBase

        class AdderAtlasPlugin(AdderPluginBase):
            def execute(self):
                self.register_files()

            def register_files(self):
                try:
                    out = self.job.register()
                except Exception:
                    self.result.set_fatal()  # P1
                    return
                if out is None:  # PG
                    self.result.set_temporary()  # P2
        """,
    HOLDER: """\
        class Holder:
            def __init__(self, items):
                self.items = items  # H0

            def get(self):
                return self.items  # H1

        class SubHolder(Holder):
            def get(self):
                return super().get()  # H2
        """,
    GEN: """\
        from pandaserver.dataservice.adder_atlas_plugin import AdderAtlasPlugin
        from pandaserver.dataservice.holder import Holder

        class AdderGen:
            def __init__(self, job, attempt_nr=None):
                self.job = job
                self.attempt_nr = attempt_nr  # G0
                self.add_result = None  # G1
                self.holder = Holder([])  # G4

            def execute_plugin(self):
                self.adder_plugin = AdderAtlasPlugin(self.job)  # G2
                self.adder_plugin.execute()
                self.add_result = self.adder_plugin.result  # G3

            def own(self):
                return self.attempt_nr

            def process_job_report(self, holder: Holder):
                self.execute_plugin()
                if self.attempt_nr is not None:  # A0
                    if self.add_result is None or not self.add_result.is_succeeded():  # A1
                        self.job.jobStatus = "failed"  # ARM
                got = holder.get()  # R0
                mine = self.own()  # R1
                held = self.holder.get()  # R2
                if got and mine and held:  # R3
                    self.job.jobStatus = "closed"  # ARM2
        """,
    SETUP_BASE: """\
        class SetupperPluginBase:
            def __init__(self, jobs):
                self.jobs = []  # S0
                for job in jobs:
                    self.jobs.append(job)

            def load(self, values):
                self.__dict__.update(values)  # D1
        """,
    SETUP: """\
        from pandaserver.dataservice.setupper_plugin_base import SetupperPluginBase

        class SetupperAtlasPlugin(SetupperPluginBase):
            def run(self):
                tmp_job_list = []
                for job in self.jobs:
                    tmp_job_list.append(job)
                self.jobs = tmp_job_list  # S1
                self.subscribe()

            def _commit(self):
                return True

            def tune(self, name, value):
                setattr(self, name, value)  # S2

            def __str__(self):
                return ", ".join(f"{key}={self.__dict__[key]}" for key in self.__dict__)  # D0

            def subscribe(self):
                for job in self.jobs:  # S3
                    if job.jobStatus != "failed" and self._commit():  # S4
                        job.dispatchDBlock = "x"  # ARM3
        """,
    OTHER: """\
        def elsewhere(plugin, jobs):
            plugin.jobs = jobs  # O1
        """,
    # TaskBuffer.peekJobs' and add_main's shapes: a list filled by append, a WeightedLists filled by add and taken
    # from by pop.
    BUFFER: """\
        class JobSpec:
            def __init__(self, panda_id):
                self.PandaID = panda_id

        class TaskBuffer:
            def peek_jobs(self, ids):
                ret_jobs = []  # T0
                for one in ids:
                    res = self.peek(one)  # T5
                    if res is None:
                        ret_jobs.append(None)  # T1
                    else:
                        ret_jobs.append(res)  # T2
                return ret_jobs

            def peek(self, one):
                return JobSpec(one)

            def list_reports(self):
                return [(1, "finished")]  # T4

            def collect(self, more):
                found = []  # T6
                found.append(2)
                found.extend(*more)
                return found[0]
        """,
    QUEUE: """\
        import multiprocessing

        from pandacommon.pandautils.thread_utils import WeightedLists
        from pandaserver.taskbuffer.task_buffer import TaskBuffer

        class Reader:
            def __init__(self, task_buffer: TaskBuffer, job_status):  # QI
                self.job = task_buffer.peek_jobs([1])[0]  # Q0
                self.job_status = job_status  # Q1
                self.extra = task_buffer.collect([])  # Q3

            def check(self):
                if self.job is None or self.job_status == "failed" or self.extra:  # Q2
                    self.job.jobStatus = "closed"  # ARM4

        class Worker:
            def __init__(self, task_buffer: TaskBuffer, reports: WeightedLists):  # WI
                self.task_buffer = task_buffer
                self.reports = reports  # W0

            def run(self):
                while True:
                    one = self.reports.pop()  # W1
                    if not one:
                        break
                    panda_id, job_status = one  # W2
                    Reader(self.task_buffer, job_status).check()  # W3

        def main(task_buffer: TaskBuffer):
            lists = WeightedLists(multiprocessing.Lock())  # M0
            others = task_buffer.list_reports()  # M1
            lists.add(3, others)  # M2
            Worker(task_buffer, lists).run()  # M3
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


ARMS = (
    (GEN, "process_job_report", "ARM", "JobSpec.jobStatus"),
    (GEN, "process_job_report", "ARM2", "JobSpec.jobStatus"),
    (SETUP, "subscribe", "ARM3", "FileSpec.dispatchDBlock"),
    (QUEUE, "check", "ARM4", "JobSpec.jobStatus"),
)


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> MapFragment:
    tmp_path: Path = tmp_path_factory.mktemp("fieldfrom")
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
    calls = sqlwrite.corpus_calls(modules, {"pandaserver": tmp_path / "pandaserver"})
    assert calls is not None
    junctions = [
        JunctionNode(
            name=f"m:{owner}:{marker}",
            map_id="m",
            derived_from="v",
            subject=subject,
            owner=f"{rel}::{owner}",
            anchor=Anchor(
                package="pandaserver", file=rel, line_start=_line(rel, marker)
            ),
            branches=[Branch(outcome=marker, line=_line(rel, marker))],
        )
        for rel, owner, marker, subject in ARMS
    ]
    with sql.across_calls(calls):
        attributor = attribution.SpecAttributor(
            progress.spec_attributes(modules), attribution.class_bases(modules)
        )
        sites = sqlwrite.read_site_nodes(
            sqlwrite.read_side(modules, attributor, calls).statements, "m", "v"
        )
        record = reach.ReachGraph(reach.graph(calls, "m", "v", _DatabaseLeaves()))
        guards, parameters, _sets, written, fields = heldfrom.guard_froms(
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
        write_froms=written,
        field_froms=fields,
    )


def _guard(fragment: MapFragment, rel: str, marker: str, name: str) -> GuardFrom:
    (found,) = [
        one
        for one in fragment.guard_froms
        if one.owner.startswith(f"{rel}::")
        and one.test_line == _line(rel, marker)
        and one.name == name
    ]
    return found


def _field(fragment: MapFragment, owner: str, field: str) -> FieldFrom:
    (found,) = [
        one for one in fragment.field_froms if one.owner == owner and one.field == field
    ]
    return found


def _ends(sources: list[HeldSource]) -> set[tuple]:
    return {
        (one.kind, one.name, one.site_owner, tuple(one.steps), tuple(one.route))
        for one in sources
    }


GEN_CLASS = f"{GEN}::AdderGen"
PLUGIN_CLASS = f"{PLUGIN}::AdderAtlasPlugin"
RESULT_CLASS = f"{RESULT}::AdderResult"
SETUP_CLASS = f"{SETUP}::SetupperAtlasPlugin"
READER_CLASS = f"{QUEUE}::Reader"
WORKER_CLASS = f"{QUEUE}::Worker"


def _parameter(
    fragment: MapFragment, rel: str, line_marker: str, name: str
) -> ParameterFrom:
    (found,) = [
        one
        for one in fragment.parameter_froms
        if one.owner.startswith(f"{rel}::")
        and one.line == _line(rel, line_marker)
        and one.parameter == name
    ]
    return found


def test_case_a_reads_the_field_on_through_the_class_to_the_call_that_changes_it(
    built: MapFragment,
) -> None:
    # 440's shape: the guard's way ends at ``self.add_result``, and names its class.
    guard = _guard(built, GEN, "A1", "self.add_result")
    assert ("self_field", "self.add_result", GEN_CLASS, (), ()) in _ends(guard.sources)
    # The class binds it to None and, in execute_plugin, to the plugin's result.
    add_result = _field(built, GEN_CLASS, "add_result")
    assert {(one.at, one.how) for one in add_result.bindings} == {
        (_at(GEN, "G1"), "assign"),
        (_at(GEN, "G3"), "assign"),
    }
    (alias,) = [one for one in add_result.bindings if one.at == _at(GEN, "G3")]
    # (A part of the plugin holds self.job, handed to it: a way with paths, read as before.)
    assert ("self_field", "self.adder_plugin", GEN_CLASS, (".result",), ()) in _ends(
        alias.sources
    )
    # The plugin is an object AdderAtlasPlugin builds ...
    plugin = _field(built, GEN_CLASS, "adder_plugin")
    (made,) = plugin.bindings
    assert _ends(made.sources) == {("object", "AdderAtlasPlugin", PLUGIN_CLASS, (), ())}
    # ... whose result its base's __init__ binds, and whose register_files changes it through set_fatal and
    # set_temporary: the store, the call on the way and the value stored.
    result = _field(built, PLUGIN_CLASS, "result")
    (built_result,) = result.bindings
    assert built_result.at == _at(BASE, "B1")
    assert built_result.owner == f"{BASE}::__init__"
    assert _ends(built_result.sources) == {
        ("object", "AdderResult", RESULT_CLASS, (), ())
    }
    stored = {
        (
            one.at,
            one.kind,
            one.key,
            tuple(one.via),
            tuple(source.name for source in one.value),
        )
        for one in result.changes
    }
    assert (
        _at(RESULT, "SF"),
        "store",
        ".status_code",
        (_at(PLUGIN, "P1"),),
        ("self.rc_fatal",),
    ) in stored
    assert (
        _at(RESULT, "ST"),
        "store",
        ".status_code",
        (_at(PLUGIN, "P2"),),
        ("self.rc_temporary",),
    ) in stored
    # The value stored reads on in its class: a constant of the class's body.
    fatal = _field(built, RESULT_CLASS, "rc_fatal")
    (constant,) = fatal.bindings
    assert (constant.at, constant.how, constant.owner) == (
        _at(RESULT, "RC"),
        "class",
        RESULT_CLASS,
    )
    assert _ends(constant.sources) == {("constant", "2", "", (), ())}


def test_a_test_on_the_way_to_a_change_of_a_way_s_object_guards_that_change(
    built: MapFragment,
) -> None:
    """register_files 788's shape: ``if out is None: self.result.set_temporary()`` is on no arm's way, but on the
    way to a change the plugin's result FieldFrom records (P2); its GuardFrom holds that line among its calls.  P1,
    in an except handler, has no test on its way and no guard."""
    guard = _guard(built, PLUGIN, "PG", "out")
    assert guard.calls == [_line(PLUGIN, "P2")]
    assert guard.arms == []
    assert guard.owner.endswith("register_files")
    assert {
        (one.name, one.test_line)
        for one in built.guard_froms
        if one.owner.startswith(f"{PLUGIN}::")
    } == {("out", _line(PLUGIN, "PG"))}


def test_a_guard_reading_self_itself_asks_each_field_it_reads_and_not_a_method_it_calls(
    built: MapFragment,
) -> None:
    guard = _guard(built, GEN, "A0", "self")
    assert _ends(guard.sources) == {("self_field", "self", GEN_CLASS, (), ())}
    attempt = _field(built, GEN_CLASS, "attempt_nr")
    assert [(one.at, one.how) for one in attempt.bindings] == [
        (_at(GEN, "G0"), "assign")
    ]
    assert [source.kind for source in attempt.bindings[0].sources] == ["parameter"]
    # S4 calls self._commit(): no FieldFrom is asked for it.
    assert not [one for one in built.field_froms if one.field == "_commit"]


def test_a_callee_s_self_is_read_back_at_the_receiver(built: MapFragment) -> None:
    got = _guard(built, GEN, "R3", "got")
    # holder.get() returns Holder's self.items: the parameter holder of the caller, with the field as a step --
    # and through SubHolder.get's super().get(), SubHolder's self, read back the same way.
    caller = f"{GEN}::process_job_report"
    assert _ends(got.sources) == {
        ("parameter", "holder", caller, (".items",), (_at(GEN, "R0"),)),
        (
            "parameter",
            "holder",
            caller,
            (".items",),
            (_at(HOLDER, "H2"), _at(GEN, "R0")),
        ),
    }
    # self.own() returns AdderGen's self.attempt_nr: the caller's own self, with the field.
    mine = _guard(built, GEN, "R3", "mine")
    assert _ends(mine.sources) == {
        ("self_field", "self", GEN_CLASS, (".attempt_nr",), (_at(GEN, "R1"),))
    }
    # self.holder.get() returns the held Holder's self.items: the caller's field, then the step.
    held = _guard(built, GEN, "R3", "held")
    assert _ends(held.sources) == {
        ("self_field", "self.holder", GEN_CLASS, (".items",), (_at(GEN, "R2"),)),
        (
            "self_field",
            "self.holder",
            GEN_CLASS,
            (".items",),
            (_at(HOLDER, "H2"), _at(GEN, "R2")),
        ),
    }
    # And the step past the field is read on: the Holder AdderGen builds binds items to its parameter.
    holder = _field(built, GEN_CLASS, "holder")
    assert {source.name for one in holder.bindings for source in one.sources} == {
        "Holder"
    }
    items = _field(built, f"{HOLDER}::Holder", "items")
    assert [
        (one.at, [source.kind for source in one.sources]) for one in items.bindings
    ] == [(_at(HOLDER, "H0"), ["parameter"])]


def test_a_field_reads_every_binding_of_the_class_and_its_bases_and_says_what_it_does_not_read(
    built: MapFragment,
) -> None:
    guard = _guard(built, SETUP, "S4", "job")
    assert ("self_field", "self.jobs", SETUP_CLASS, ("[*]",), ()) in _ends(
        guard.sources
    )
    jobs = _field(built, SETUP_CLASS, "jobs")
    assert {(one.at, one.owner) for one in jobs.bindings} == {
        (_at(SETUP_BASE, "S0"), f"{SETUP_BASE}::__init__"),
        (_at(SETUP, "S1"), f"{SETUP}::run"),
    }
    # The base's append is a change of the list the field holds; the value it puts in is not read.
    assert ("method", "*", "append") in {
        (one.kind, one.key, one.what) for one in jobs.changes
    }
    stops = {(one.kind, tuple(one.at)) for one in jobs.stops}
    # A name computed in setattr may be the field; elsewhere stores the field on another object.
    assert ("computed", (_at(SETUP, "S2"),)) in stops
    assert ("outside", (_at(OTHER, "O1"),)) in stops


def test_a_read_of_self_dict_sets_no_field(built: MapFragment) -> None:
    # __str__ only reads self.__dict__ (JediDatasetSpec's, InputChunk's): no field is set there.
    jobs = _field(built, SETUP_CLASS, "jobs")
    assert not [
        one
        for one in jobs.stops
        if one.kind == "computed" and _at(SETUP, "D0") in one.at
    ]


def test_self_dict_changed_in_place_may_set_any_field(built: MapFragment) -> None:
    # The base's load() updates self.__dict__: the field read, like any, may be set there.
    jobs = _field(built, SETUP_CLASS, "jobs")
    assert [
        one.at
        for one in jobs.stops
        if one.kind == "computed" and _at(SETUP_BASE, "D1") in one.at
    ] == [[_at(SETUP_BASE, "D1")]]


def test_the_gates_find_a_record_for_each_field_and_parameter_a_way_ends_at(
    built: MapFragment,
) -> None:
    fields = guard_fields_are_read(built)
    assert fields.passed, fields.failures
    assert fields.checked > 0
    parameters = guard_parameters_are_read(built)
    assert parameters.passed, parameters.failures


def test_an_element_of_a_list_filled_by_append_is_what_was_appended(
    built: MapFragment,
) -> None:
    # TaskBuffer.peekJobs' shape: ``retJobs = []`` filled by ``append``, read at ``[0]`` by the caller.
    job = _field(built, READER_CLASS, "job")
    (bound,) = job.bindings
    assert _ends(bound.sources) == {
        ("constant", "None", "", (), (_at(QUEUE, "Q0"),)),
        (
            "object",
            "JobSpec",
            f"{BUFFER}::JobSpec",
            (),
            (_at(BUFFER, "T5"), _at(QUEUE, "Q0")),
        ),
    }
    assert not any(one.made_empty for one in bound.sources)
    # A put whose value is not read (``extend(*more)``) leaves the list's own end too.
    (extra,) = _field(built, READER_CLASS, "extra").bindings
    assert _ends(extra.sources) == {
        ("constant", "2", "", (), (_at(QUEUE, "Q3"),)),
        ("unknown", "[]", "", ("[0]",), (_at(QUEUE, "Q3"),)),
    }
    (kept,) = [one for one in extra.sources if one.kind == "unknown"]
    assert kept.made_empty == "collection"
    assert [(one.what, len(one.value)) for one in kept.changes] == [
        ("append", 1),
        ("extend", 0),
    ]


def test_a_take_from_a_container_reads_its_receiver_s_element(
    built: MapFragment,
) -> None:
    # add_main's shape: ``one_jor = self.job_output_reports.pop()`` unpacked, its second the job's status.
    status = _parameter(built, QUEUE, "QI", "job_status")
    assert _ends(status.sources) == {
        (
            "self_field",
            "self.reports",
            WORKER_CLASS,
            ("[*]", "[1]"),
            (_at(QUEUE, "W3"),),
        ),
    }
    # The field read on with the steps past it: its binding's parameter, its caller's WeightedLists.
    (bound,) = _field(built, WORKER_CLASS, "reports").bindings
    assert _ends(bound.sources) == {
        ("parameter", "reports", f"{QUEUE}::__init__", (), ())
    }


def test_a_container_outside_the_corpus_holds_what_its_method_puts(
    built: MapFragment,
) -> None:
    # ``WeightedLists(lock)`` made empty, then ``add(3, others)`` puts others' elements.
    reports = _parameter(built, QUEUE, "WI", "reports")
    (made,) = reports.sources
    assert (made.kind, made.name, made.made_empty, made.route) == (
        "call",
        "WeightedLists",
        "collection",
        [_at(QUEUE, "M3")],
    )
    (add,) = made.changes
    assert (add.at, add.kind, add.what) == (_at(QUEUE, "M2"), "method", "add")
    assert _ends(add.value) == {
        ("unknown", "[(1, 'finished')]", "", ("[*]",), (_at(QUEUE, "M1"),))
    }
    # A step taken past the record's end reads on into what was put, through the route to the container.
    for step in ("[*]", "[0]"):
        values, stays = heldfrom.held_elements(made, step)
        assert _ends(values) == {
            (
                "unknown",
                "[(1, 'finished')]",
                "",
                ("[*]",),
                (_at(QUEUE, "M1"), _at(QUEUE, "M3")),
            ),
        }
        assert not stays
    assert heldfrom.held_elements(add.value[0], "[0]") == ([], True)
