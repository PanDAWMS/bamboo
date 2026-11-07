"""The ways of a name a loop, a with or an unpacking binds -- ``codemap.panda.heldfrom`` -- read as an
assignment's value is, on the shapes the querier met: ``update_jobs``' ``for job in job_list``, and ``archiveJob``'s
``retEvS, retNewPandaID = self.ppEventServiceJob(job, ...)`` whose returned tuple holds a copy's ``PandaID``.  Each
way keeps the element or slot step, the attribute steps after it, a copy's ``through`` and the route through a return,
as an assignment's does.  A generator's call is the generator: a loop over it takes, and a ``with`` over it enters
(``DataCarousel``'s ``with self.request_lock(...) as locked_spec``), what it yields; read otherwise, it is a stop."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from bamboo.codemap import calls as calls_mod
from bamboo.codemap import reach
from bamboo.codemap.gitsource import blob_sha
from bamboo.codemap.models import (
    KEY_CONSTANT,
    KEY_PARAMETER,
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

JOB = "pandaserver/taskbuffer/JobSpec.py"
PROXY = "pandaserver/taskbuffer/db_proxy_mods/job_complex_module.py"

SOURCES = {
    JOB: """\
        class JobSpec:
            def __init__(self) -> None:
                self.PandaID = 0
                self.jobStatus = ""
                self.attemptNr = 0
        """,
    PROXY: """\
        import contextlib
        import copy

        from pandaserver.taskbuffer.JobSpec import JobSpec


        class Holder:
            def __init__(self, job: JobSpec) -> None:
                self.job = job

            def __enter__(self) -> JobSpec:
                return self.job

            def __exit__(self, *args) -> None:
                return None


        class JobComplexModule:
            def update_jobs(self, job_list: list[JobSpec]) -> None:
                for job in job_list:
                    job.jobStatus = "assigned"  # U1

            def update_pairs(self, job_pairs: list[tuple[int, JobSpec]]) -> None:
                for idx, job in job_pairs:
                    job.jobStatus = "paired"  # P1

            def archiveJob(self, job: JobSpec) -> None:
                retEvS, retNewPandaID = self.ppEventServiceJob(job)
                if retNewPandaID is None:  # T-RET
                    job.jobStatus = "failed"  # A1

            def ppEventServiceJob(self, job: JobSpec) -> tuple:
                retValue = 1, None
                jobSpec = copy.copy(job)
                if job.attemptNr > 1:
                    retValue = 0, jobSpec.PandaID
                return retValue

            def hold(self, job: JobSpec) -> None:
                with Holder(job) as held:
                    held.jobStatus = "held"  # W1

            @contextlib.contextmanager
            def job_lock(self, job: JobSpec):
                if job.attemptNr > 0:
                    locked = copy.copy(job)
                    yield locked
                else:
                    yield None

            def lock_job(self, job: JobSpec) -> None:
                with self.job_lock(job) as locked:
                    if locked is None:  # T-LOCK
                        job.jobStatus = "unlocked"  # L1

            def pending(self, jobs: list[JobSpec]):
                for one in jobs:
                    yield one

            def update_pending(self, jobs: list[JobSpec]) -> None:
                for job in self.pending(jobs):
                    job.jobStatus = "pending"  # G1

            def count_pending(self, jobs: list[JobSpec]) -> None:
                waiting = self.pending(jobs)
                if waiting:  # T-GEN
                    jobs[0].jobStatus = "counted"  # C1
        """,
}

#: (file, function, arm marker, subject, outcome)
ARMS = (
    (PROXY, "update_jobs", "U1", "JobSpec.jobStatus", "'assigned'"),
    (PROXY, "update_pairs", "P1", "JobSpec.jobStatus", "'paired'"),
    (PROXY, "archiveJob", "A1", "JobSpec.jobStatus", "'failed'"),
    (PROXY, "hold", "W1", "JobSpec.jobStatus", "'held'"),
    (PROXY, "lock_job", "L1", "JobSpec.jobStatus", "'unlocked'"),
    (PROXY, "update_pending", "G1", "JobSpec.jobStatus", "'pending'"),
    (PROXY, "count_pending", "C1", "JobSpec.jobStatus", "'counted'"),
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
    tmp_path: Path = tmp_path_factory.mktemp("els")
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


def _object(fragment: MapFragment, marker: str) -> WriteFrom:
    (found,) = [
        one
        for one in fragment.write_froms
        if one.owner == f"{PROXY}::{one.owner.rpartition('::')[2]}"
        and one.line == _line(PROXY, marker)
        and one.role == "object"
    ]
    return found


def _guard(fragment: MapFragment, marker: str, name: str) -> GuardFrom:
    (found,) = [
        one
        for one in fragment.guard_froms
        if one.owner.startswith(f"{PROXY}::") and one.test_line == _line(PROXY, marker) and one.name == name
    ]
    return found


def _ends(sources: list[HeldSource]) -> list[tuple]:
    """Each way's end and what it went through, as a reader binding two ways compares them."""
    return sorted(
        (one.kind, one.name, one.site_owner.rpartition("::")[2], tuple(one.steps), tuple(t.call for t in one.through))
        for one in sources
    )


def test_a_loop_s_element_is_its_iterable_s_with_an_element_step(built: MapFragment) -> None:
    assert _ends(_object(built, "U1").sources) == [(KEY_PARAMETER, "job_list", "update_jobs", ("[*]",), ())]


def test_an_unpacked_loop_target_is_its_slot_of_the_element(built: MapFragment) -> None:
    assert _ends(_object(built, "P1").sources) == [(KEY_PARAMETER, "job_pairs", "update_pairs", ("[*]", "[1]"), ())]


def test_an_unpacked_call_s_slot_is_the_element_its_returned_tuples_hold_there(built: MapFragment) -> None:
    guard = _guard(built, "T-RET", "retNewPandaID")
    assert {one.kind for one in guard.sources} == {KEY_CONSTANT, KEY_PARAMETER}
    assert {one.name for one in guard.sources if one.kind == KEY_CONSTANT} == {"None"}
    (copied,) = [one for one in guard.sources if one.kind == KEY_PARAMETER]
    assert (copied.name, copied.site_owner) == ("job", f"{PROXY}::archiveJob")
    assert copied.steps == [".PandaID"]
    assert [t.call for t in copied.through] == ["copy.copy"]
    assert len(copied.route) == 1 and copied.route[0].endswith(f":{_line(PROXY, 'T-RET') - 1}")


def test_a_with_target_is_its_context_s_enter(built: MapFragment) -> None:
    (way,) = _object(built, "W1").sources
    assert way.steps[-1] == ".__enter__()"


def test_a_with_over_a_generator_holds_what_it_yields(built: MapFragment) -> None:
    guard = _guard(built, "T-LOCK", "locked")
    assert _ends(guard.sources) == [
        (KEY_CONSTANT, "None", "", (), ()),
        (KEY_PARAMETER, "job", "lock_job", (), ("copy.copy",)),
    ]
    assert not guard.stops


def test_a_loop_over_a_generator_takes_what_it_yields(built: MapFragment) -> None:
    (way,) = _object(built, "G1").sources
    assert (way.kind, way.name, way.site_owner, way.steps) == (KEY_PARAMETER, "jobs", f"{PROXY}::update_pending", ["[*]"])
    assert way.route and way.route[0].endswith(f":{_line(PROXY, 'G1') - 1}")


def test_a_generator_s_call_read_as_a_value_is_a_stop(built: MapFragment) -> None:
    guard = _guard(built, "T-GEN", "waiting")
    assert not guard.sources
    assert [(one.kind, one.name) for one in guard.stops] == [("generator", f"{PROXY}::pending")]
