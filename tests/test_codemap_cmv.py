"""A comprehension's value -- ``codemap.panda.heldfrom`` -- read as the collection of its element: the element's ways,
its ``for`` clauses' targets read as a loop's are (``pathcond.comprehension_clause``), through the comprehension.
``adder_gen.parse_job_output_report``'s ``log_out = [f for f in self.job.Files if ...]`` holds files of the job, not
the job; an element taken of such a list is the element again, not an element of it."""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import pytest

from bamboo.codemap import calls as calls_mod
from bamboo.codemap import pathcond, reach
from bamboo.codemap.gitsource import blob_sha
from bamboo.codemap.models import (
    KEY_CONSTANT,
    KEY_PARAMETER,
    Anchor,
    Branch,
    GuardFrom,
    HeldSource,
    JunctionNode,
    SourceModule,
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
                self.Files = []
        """,
    PROXY: """\
        from pandaserver.taskbuffer.JobSpec import JobSpec


        class JobComplexModule:
            def logs(self, job: JobSpec) -> None:
                log_out = [f for f in job.Files if f.type in ("log", "output")]
                if not log_out:  # T-LOGS
                    job.jobStatus = "a"  # A-LOGS

            def ids(self, jobs: list[JobSpec]) -> None:
                ids = [one.PandaID for one in jobs]
                if ids:  # T-IDS
                    jobs[0].jobStatus = "b"  # A-IDS

            def firsts(self, pairs: list[tuple[int, JobSpec]], job: JobSpec) -> None:
                firsts = {a for a, b in pairs}
                if firsts:  # T-FIRSTS
                    job.jobStatus = "c"  # A-FIRSTS

            def files(self, jobs: list[JobSpec]) -> None:
                files = [y for x in jobs for y in x.Files]
                if files:  # T-FILES
                    jobs[0].jobStatus = "d"  # A-FILES

            def outer(self, job: JobSpec) -> None:
                f = job
                names = [f for f in job.Files]
                if f.PandaID and names:  # T-OUTER
                    job.jobStatus = "e"  # A-OUTER

            def first(self, job: JobSpec) -> None:
                first = [f for f in job.Files if f.type == "log"][0]
                if first:  # T-FIRST
                    job.jobStatus = "f"  # A-FIRST

            def each(self, job: JobSpec) -> None:
                logs = [f for f in job.Files if f.type == "log"]
                for x in logs:
                    if x.PandaID:  # T-EACH
                        job.jobStatus = "g"  # A-EACH

            def by_name(self, job: JobSpec) -> None:
                by_name = {f.lfn: f for f in job.Files}
                for key in by_name:
                    if key:  # T-KEYS
                        job.jobStatus = "h"  # A-KEYS

            def count(self, job: JobSpec) -> None:
                n = sum(1 for f in job.Files)
                if n:  # T-COUNT
                    job.jobStatus = "i"  # A-COUNT
        """,
}

#: (function, arm marker)
ARMS = (
    ("logs", "A-LOGS"),
    ("ids", "A-IDS"),
    ("firsts", "A-FIRSTS"),
    ("files", "A-FILES"),
    ("outer", "A-OUTER"),
    ("first", "A-FIRST"),
    ("each", "A-EACH"),
    ("by_name", "A-KEYS"),
    ("count", "A-COUNT"),
)


def _line(rel: str, marker: str) -> int:
    (found,) = [
        number
        for number, text in enumerate(textwrap.dedent(SOURCES[rel]).splitlines(), 1)
        if text.rstrip().endswith(f"# {marker}")
    ]
    return found


@pytest.fixture(scope="module")
def guards(tmp_path_factory: pytest.TempPathFactory) -> list[GuardFrom]:
    """Read the sources as the build does, with a junction for each arm marked, then the guards."""
    tmp_path: Path = tmp_path_factory.mktemp("cmv")
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
            name=f"m:{PROXY}:{owner}:{marker}",
            map_id="m",
            derived_from="v",
            subject="JobSpec.jobStatus",
            owner=f"{PROXY}::{owner}",
            anchor=Anchor(package="pandaserver", file=PROXY, line_start=_line(PROXY, marker)),
            branches=[Branch(outcome="'x'", line=_line(PROXY, marker))],
        )
        for owner, marker in ARMS
    ]
    trees = {module.rel_path: module.tree for module in modules}
    with sql.across_calls(calls):
        record = reach.ReachGraph(reach.graph(calls, "m", "v", _DatabaseLeaves()))
        found, _parameters, _sets, _written, _fields = heldfrom.guard_froms(junctions, [], trees, calls, record)
    return found


def _ways(guards: list[GuardFrom], marker: str, name: str) -> list[tuple]:
    (guard,) = [one for one in guards if one.test_line == _line(PROXY, marker) and one.name == name]
    return [_way(one) for one in guard.sources]


def _way(one: HeldSource) -> tuple:
    return (one.kind, one.name, tuple(one.steps), tuple(t.call for t in one.through))


def test_a_comprehension_s_value_is_its_element_s_ways_through_it(guards: list[GuardFrom]) -> None:
    assert _ways(guards, "T-LOGS", "log_out") == [(KEY_PARAMETER, "job", (".Files", "[*]"), ("<listcomp>",))]


def test_a_mapped_element_keeps_its_attribute_step(guards: list[GuardFrom]) -> None:
    assert _ways(guards, "T-IDS", "ids") == [(KEY_PARAMETER, "jobs", ("[*]", ".PandaID"), ("<listcomp>",))]


def test_an_unpacking_clause_reads_its_target_s_slot(guards: list[GuardFrom]) -> None:
    assert _ways(guards, "T-FIRSTS", "firsts") == [(KEY_PARAMETER, "pairs", ("[*]", "[0]"), ("<setcomp>",))]


def test_a_later_clause_reads_an_earlier_clause_s_target(guards: list[GuardFrom]) -> None:
    assert _ways(guards, "T-FILES", "files") == [(KEY_PARAMETER, "jobs", ("[*]", ".Files", "[*]"), ("<listcomp>",))]


def test_a_clause_s_target_is_not_the_function_s_name_of_the_same_spelling(guards: list[GuardFrom]) -> None:
    assert _ways(guards, "T-OUTER", "f") == [(KEY_PARAMETER, "job", (), ())]


def test_an_element_of_a_comprehension_s_list_is_its_element(guards: list[GuardFrom]) -> None:
    assert _ways(guards, "T-FIRST", "first") == [(KEY_PARAMETER, "job", (".Files", "[*]"), ())]
    assert _ways(guards, "T-EACH", "x") == [(KEY_PARAMETER, "job", (".Files", "[*]"), ())]


def test_a_dict_s_keys_are_not_the_values_it_is_read_as(guards: list[GuardFrom]) -> None:
    assert _ways(guards, "T-KEYS", "key") == [(KEY_PARAMETER, "job", (".Files", "[*]", "[*]"), ("<dictcomp>",))]


def test_a_generator_handed_to_a_call_goes_through_both(guards: list[GuardFrom]) -> None:
    assert _ways(guards, "T-COUNT", "n") == [(KEY_CONSTANT, "1", (), ("<genexpr>", "sum"))]


def _function(text: str) -> ast.FunctionDef:
    tree = ast.parse(textwrap.dedent(text))
    pathcond.attach_parents(tree)
    return tree.body[0]  # type: ignore[return-value]


def test_the_first_iterable_is_read_outside_the_comprehension() -> None:
    func = _function(
        """\
        def f(x):
            return [x for x in x.Files if x]
        """
    )
    (comp,) = [node for node in ast.walk(func) if isinstance(node, ast.ListComp)]
    (clause,) = comp.generators
    iterable = clause.iter.value  # type: ignore[attr-defined]
    assert pathcond.comprehension_clause(func, "x", iterable) is None
    assert pathcond.held_at(func, "x", iterable).kind == pathcond.HELD_PARAMETER
    assert pathcond.comprehension_clause(func, "x", comp.elt) is clause
    assert pathcond.comprehension_clause(func, "x", clause.ifs[0]) is clause
    assert pathcond.held_at(func, "x", comp.elt).sites == (clause,)
    binding = pathcond.clause_binding(clause, "x")
    assert binding is not None and (binding.kind, binding.nested) == (pathcond.BIND_FOR, True)
    assert not [one for one in pathcond.bindings(func, "x") if one.site is clause]
