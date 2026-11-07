"""What the object an arm writes, and the value a passthrough arm hands on, hold -- :class:`WriteFrom` from
``codemap.panda.heldfrom`` -- and the GuardFroms of objects on ``self``, on the shapes of JEDI's job generator
(``jobSpec = JobSpec()`` then ``jobSpec.jediTaskID = taskSpec.jediTaskID`` under ``taskSpec.usePrePro()``), the task's
``setPrePro`` and the refiner's ``setSplitRule``: a reader binds a guard's name to the arm's row where the two ways end
alike."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from bamboo.codemap import calls as calls_mod
from bamboo.codemap import gates, reach
from bamboo.codemap.gitsource import blob_sha
from bamboo.codemap.models import (
    KEY_OBJECT,
    KEY_PARAMETER,
    KEY_SELF_FIELD,
    WRITE_OBJECT,
    WRITE_VALUE,
    Anchor,
    Branch,
    GuardFrom,
    HeldSource,
    JunctionNode,
    MapFragment,
    SourceModule,
    WriteFrom,
)
from bamboo.codemap.panda import heldfrom, sql
from bamboo.codemap.panda.plugin import _DatabaseLeaves
from bamboo.codemap.panda.recognizers import sqlwrite

SPEC = "pandaserver/taskbuffer/JediTaskSpec.py"
JOB = "pandaserver/taskbuffer/JobSpec.py"
GEN = "pandajedi/jediorder/JobGenerator.py"
REFINER = "pandajedi/jedirefine/TaskRefinerBase.py"

SOURCES = {
    SPEC: """\
        class JediTaskSpec:
            def __init__(self) -> None:
                self.splitRule = None
                self.jediTaskID = 0
                self.status = ""

            def usePrePro(self) -> bool:
                return self.splitRule is not None and "PP=1" in self.splitRule

            def setPrePro(self) -> None:
                if self.splitRule is None:
                    # new
                    self.splitRule = "PP" + "=" + "1"  # S1
                else:
                    # append
                    self.splitRule += "," + "PP" + "=" + "1"  # S2

            def setOnHold(self) -> None:
                self.status = "pending"  # H1
        """,
    JOB: """\
        class JobSpec:
            def __init__(self) -> None:
                self.jediTaskID = None
                self.attemptNr = 0
        """,
    GEN: """\
        from pandaserver.taskbuffer.JediTaskSpec import JediTaskSpec
        from pandaserver.taskbuffer.JobSpec import JobSpec


        class Recorder:
            def note(self, taskSpec: JediTaskSpec) -> None:
                taskSpec.status = "noted"


        class JobGeneratorThread:
            def __init__(self) -> None:
                self.recorder = Recorder()

            def doGenerate(self, taskSpec: JediTaskSpec, inSubChunks: list) -> list:
                jobs = []
                for inSubChunk in inSubChunks:
                    jobSpec = JobSpec()  # MADE
                    jobSpec.attemptNr = 0
                    if taskSpec.usePrePro():  # T-PP
                        jobSpec.jediTaskID = taskSpec.jediTaskID  # J1
                    else:
                        setattr(jobSpec, "jediTaskID", 0)  # J2
                    jobSpec.jediTaskID, jobSpec.attemptNr = taskSpec.jediTaskID, 1  # J3
                    jobs.append(jobSpec)
                return jobs

            def runImpl(self, taskSpec: JediTaskSpec, goForward: bool) -> None:
                varMap = {}
                if not goForward:
                    taskSpec.setOnHold()  # C1
                    self.recorder.note(taskSpec)  # C2
                    varMap[":status"] = "pending"  # Q1

            def generate(self, inputList: list) -> None:
                for tmpInputItem in inputList:
                    taskSpec, cloudName, inputChunk = tmpInputItem
                    if not cloudName:
                        taskSpec.setOnHold()  # C3
                        inputChunk.close()  # C4
        """,
    REFINER: """\
        from pandaserver.taskbuffer.JediTaskSpec import JediTaskSpec


        class TaskRefinerBase:
            def __init__(self, taskSpec: JediTaskSpec) -> None:
                self.taskSpec = taskSpec

            def setSplitRule(self, key_or_value: str, rule_token: str) -> None:
                tmpStr = f"{rule_token}={key_or_value}"
                if not self.taskSpec.splitRule:  # T-SR
                    self.taskSpec.splitRule = tmpStr  # R1
                else:
                    self.taskSpec.splitRule = self.taskSpec.splitRule.replace(rule_token, tmpStr)  # R2
        """,
}

#: (file, function, marker, subject, outcome) of each arm.
ARMS = (
    (GEN, "doGenerate", "J1", "JobSpec.jediTaskID", "passthrough(JediTaskSpec.jediTaskID)"),
    (GEN, "doGenerate", "J2", "JobSpec.jediTaskID", "0"),
    (GEN, "doGenerate", "J3", "JobSpec.jediTaskID", "passthrough(JediTaskSpec.jediTaskID)"),
    (GEN, "runImpl", "C1", "JediTaskSpec.status", "pending"),
    (GEN, "runImpl", "C2", "JediTaskSpec.status", "noted"),
    (GEN, "runImpl", "Q1", "JEDI_Tasks.status", "pending"),
    (GEN, "generate", "C3", "JediTaskSpec.status", "pending"),
    (GEN, "generate", "C4", "JediTaskSpec.status", "pending"),
    (SPEC, "setPrePro", "S1", "JediTaskSpec.splitRule", "runtime(PP=1)"),
    (SPEC, "setPrePro", "S2", "JediTaskSpec.splitRule", "runtime(PP=1)"),
    (SPEC, "setOnHold", "H1", "JediTaskSpec.status", "pending"),
    (REFINER, "setSplitRule", "R1", "JediTaskSpec.splitRule", "runtime(tmpStr)"),
    (REFINER, "setSplitRule", "R2", "JediTaskSpec.splitRule", "runtime(replace())"),
)


def _line(rel: str, marker: str) -> int:
    (found,) = [
        number
        for number, text in enumerate(textwrap.dedent(SOURCES[rel]).splitlines(), 1)
        if text.rstrip().endswith(f"# {marker}")
    ]
    return found


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> MapFragment:
    """Read the sources as the build does, with a junction for each arm marked, then the guards and the writes."""
    tmp_path: Path = tmp_path_factory.mktemp("writefrom")
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
    roots = {"pandaserver": tmp_path / "pandaserver", "pandajedi": tmp_path / "pandajedi"}
    calls = sqlwrite.corpus_calls(modules, roots)
    assert calls is not None
    junctions = [
        JunctionNode(
            name=f"m:{rel}:{owner}:{marker}",
            map_id="m",
            derived_from="v",
            subject=subject,
            owner=f"{rel}::{owner}",
            anchor=Anchor(package=rel.split("/", 1)[0], file=rel, line_start=_line(rel, marker)),
            branches=[Branch(outcome=outcome, line=_line(rel, marker))],
        )
        for rel, owner, marker, subject, outcome in ARMS
    ]
    trees = {module.rel_path: module.tree for module in modules}
    with sql.across_calls(calls):
        record = reach.ReachGraph(reach.graph(calls, "m", "v", _DatabaseLeaves()))
        guards, parameters, _sets, written = heldfrom.guard_froms(junctions, [], trees, calls, record)
    return MapFragment(
        map_id="m",
        derived_from="v",
        junctions=junctions,
        guard_froms=guards,
        parameter_froms=parameters,
        write_froms=written,
    )


def _writes(fragment: MapFragment, rel: str, marker: str) -> list[WriteFrom]:
    return [one for one in fragment.write_froms if one.owner.startswith(f"{rel}::") and one.line == _line(rel, marker)]


def _guard(fragment: MapFragment, rel: str, marker: str, name: str) -> GuardFrom:
    (found,) = [
        one
        for one in fragment.guard_froms
        if one.owner.startswith(f"{rel}::") and one.test_line == _line(rel, marker) and one.name == name
    ]
    return found


def _end(source: HeldSource) -> tuple:
    """What a reader binding two ways compares: where each ends."""
    return (source.kind, source.name, source.site_owner, source.site_line)


def test_the_job_an_arm_writes_and_the_task_it_hands_the_id_of_are_read_as_a_guard_s_name_is(
    built: MapFragment,
) -> None:
    """``jobSpec.jediTaskID = taskSpec.jediTaskID``: the object is the job built at MADE, the value the task
    doGenerate is handed, with its step -- the parameter the guard ``taskSpec.usePrePro()`` ends at, so the guard
    reads the task whose ID the job is given."""
    (job, value) = sorted(_writes(built, GEN, "J1"), key=lambda one: one.role)
    assert (job.role, job.expression, job.subject) == (WRITE_OBJECT, "jobSpec", "JobSpec.jediTaskID")
    assert [(one.kind, one.name, one.line) for one in job.sources] == [(KEY_OBJECT, "JobSpec", _line(GEN, "MADE"))]
    # Which object, not what it holds: ``jobSpec.attemptNr = 0`` between the binding and the write is not said.
    assert [one.changes for one in job.sources] == [[]]
    assert (value.role, value.expression) == (WRITE_VALUE, "taskSpec.jediTaskID")
    (source,) = value.sources
    assert (source.kind, source.name, source.site_owner, source.steps) == (
        KEY_PARAMETER,
        "taskSpec",
        f"{GEN}::doGenerate",
        [".jediTaskID"],
    )
    guard = _guard(built, GEN, "T-PP", "taskSpec")
    assert [_end(one) for one in guard.sources] == [_end(source)]
    # The parameter the value ends at is read on, as a guard's is.
    assert (f"{GEN}::doGenerate", "taskSpec") in {(one.owner, one.parameter) for one in built.parameter_froms}


def test_a_tuple_target_and_setattr_write_their_object_and_only_a_passthrough_hands_a_value(built: MapFragment) -> None:
    """J3 writes two fields of ``jobSpec`` in one statement and hands the task's ID to one of them; ``setattr`` at J2
    writes ``jobSpec`` a literal, which is no passthrough."""
    assert sorted((one.role, one.expression) for one in _writes(built, GEN, "J3")) == [
        (WRITE_OBJECT, "jobSpec"),
        (WRITE_VALUE, "taskSpec.jediTaskID"),
    ]
    assert [(one.role, one.expression) for one in _writes(built, GEN, "J2")] == [(WRITE_OBJECT, "jobSpec")]


def test_a_setter_called_at_the_arm_writes_its_receiver_and_another_class_s_method_or_a_bind_writes_none(
    built: MapFragment,
) -> None:
    """``taskSpec.setOnHold()`` is a method of the subject's class: its receiver, runImpl's parameter, is the task
    written.  ``self.recorder.note(taskSpec)`` is a method of another class, its receiver not the task; and an SQL
    arm fills a bind map, writing its statement's rows."""
    (held,) = _writes(built, GEN, "C1")
    assert (held.role, held.expression) == (WRITE_OBJECT, "taskSpec")
    assert [(one.kind, one.name, one.site_owner) for one in held.sources] == [
        (KEY_PARAMETER, "taskSpec", f"{GEN}::runImpl")
    ]
    assert _writes(built, GEN, "C2") == []
    assert _writes(built, GEN, "Q1") == []


def test_a_setter_on_a_receiver_of_no_class_the_build_tells_writes_its_receiver_by_the_method_s_name(
    built: MapFragment,
) -> None:
    """JobGenerator's runImpl takes ``taskSpec`` out of an input item, ``taskSpec, cloudName, inputChunk =
    tmpInputItem``: no call of it is placed, and the arm at ``taskSpec.setOnHold()`` still writes ``taskSpec`` --
    setOnHold is a method of the subject's class.  ``inputChunk.close()`` names none of JediTaskSpec's."""
    assert [(one.role, one.expression) for one in _writes(built, GEN, "C3")] == [(WRITE_OBJECT, "taskSpec")]
    assert _writes(built, GEN, "C4") == []


def test_a_test_reading_a_field_of_self_has_a_guard_on_self_ending_where_the_arm_s_object_does(
    built: MapFragment,
) -> None:
    """``self.splitRule is None`` in setPrePro: a GuardFrom named ``self``, its way ending where the object of
    ``self.splitRule = ...`` does -- the task the setter is called on, for both arms (S2 is ``+=``)."""
    (guard,) = [one for one in built.guard_froms if one.owner == f"{SPEC}::setPrePro"]
    assert (guard.name, guard.test) == ("self", "self.splitRule is None")
    assert guard.arms == [_line(SPEC, "S1"), _line(SPEC, "S2")]
    for marker in ("S1", "S2"):
        (held,) = _writes(built, SPEC, marker)
        assert (held.role, held.expression) == (WRITE_OBJECT, "self")
        assert [_end(one) for one in held.sources] == [_end(one) for one in guard.sources]
    assert [one.kind for one in guard.sources] == [KEY_SELF_FIELD]


def test_a_field_of_self_s_object_is_guarded_by_its_path_and_not_again_by_self(built: MapFragment) -> None:
    """``not self.taskSpec.splitRule`` reads the refiner's task: one GuardFrom, ``self.taskSpec``, ending where the
    object of ``self.taskSpec.splitRule = tmpStr`` does; not one for ``self`` as well."""
    guards = [one for one in built.guard_froms if one.owner == f"{REFINER}::setSplitRule"]
    assert [(one.name, one.test_line) for one in guards] == [("self.taskSpec", _line(REFINER, "T-SR"))]
    for marker in ("R1", "R2"):
        objects = [one for one in _writes(built, REFINER, marker) if one.role == WRITE_OBJECT]
        assert [(one.expression, [_end(way) for way in one.sources]) for one in objects] == [
            ("self.taskSpec", [_end(way) for way in guards[0].sources])
        ]


def test_every_way_of_a_write_ending_at_a_parameter_has_that_parameter_s_record(built: MapFragment) -> None:
    """The gate the guards' ways answer, over the writes' too."""
    result = gates.guard_parameters_are_read(built)
    assert result.passed, result.failures
    assert result.checked >= 4
