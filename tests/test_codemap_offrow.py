"""Arms whose object is written to no row -- ``codemap.panda.offrow``, and the readers that skip them."""

from __future__ import annotations

import ast
from pathlib import Path

from bamboo.codemap import reach
from bamboo.codemap.gitsource import blob_sha
from bamboo.codemap.models import (
    OFF_ROW_NO_WRITE,
    OFF_ROW_NOT_TAKEN,
    Anchor,
    Branch,
    EntityNode,
    JunctionNode,
    MapFragment,
    OffRow,
    SourceModule,
)
from bamboo.codemap.panda import objects as panda_objects
from bamboo.codemap.panda import offrow, pathcond, sql
from bamboo.codemap.panda.plugin import _DatabaseLeaves
from bamboo.codemap.panda.recognizers import sqlwrite
from bamboo.codemap.strategy import _candidate, _reaching

PACKAGE = "pkg"

SPEC = """\
class JobSpec:
    _attributes = ("PandaID", "jobStatus", "specialHandling")

    def __init__(self):
        for attr in self._attributes:
            object.__setattr__(self, attr, None)

    def is_debug(self):
        return "debug" in (self.specialHandling or "")
"""

PROXY = """\
from pkg.spec import JobSpec


class DBProxy:
    def ask(self, handling):
        job = JobSpec()
        job.specialHandling = handling  # O1
        return job.is_debug()

    def carry(self, panda_id):
        job = JobSpec()
        job.PandaID = panda_id  # O2
        self.cur.execute("SELECT jobStatus FROM ATLAS_PANDA.jobsActive4 WHERE PandaID=:PandaID", {":PandaID": job.PandaID})
        return self.cur.fetchall()

    def lock(self, panda_id, handling):
        job = JobSpec()
        job.specialHandling = handling  # O3
        if job.is_debug():
            var_map = {":sh": "locked", ":PandaID": panda_id}
            self.cur.execute("UPDATE ATLAS_PANDA.jobsActive4 SET specialHandling=:sh WHERE PandaID=:PandaID", var_map)

    def set_column(self, panda_id):
        job = JobSpec()
        job.specialHandling = "set"  # W1
        var_map = {":sh": job.specialHandling, ":PandaID": panda_id}
        self.cur.execute("UPDATE ATLAS_PANDA.jobsActive4 SET specialHandling=:sh WHERE PandaID=:PandaID", var_map)

    def via(self):
        job = JobSpec()
        job.specialHandling = "via"  # W2
        self.archive(job)

    def archive(self, job):
        var_map = {":sh": job.specialHandling, ":PandaID": job.PandaID}
        self.cur.execute("UPDATE ATLAS_PANDA.jobsActive4 SET specialHandling=:sh WHERE PandaID=:PandaID", var_map)

    def make(self):
        job = JobSpec()
        job.specialHandling = "made"  # W3
        return job

    def handed(self, job: JobSpec):
        job.specialHandling = "handed"  # W4
        return job.is_debug()

    def queued(self, pool):
        job = JobSpec()
        job.specialHandling = "queued"  # W5
        pool.put(job)

    def kept(self, jobs):
        job = JobSpec()
        job.specialHandling = "kept"  # W6
        jobs.append(job)

    def stashed(self, holder):
        job = JobSpec()
        job.specialHandling = "stashed"  # W7
        setattr(holder, "job", job)
"""

ENTITIES = [
    EntityNode(
        name="jobsactive4",
        map_id="m",
        derived_from="v",
        spec_class="JobSpec",
        tables=["jobsActive4"],
    ),
]


def _line(text: str, marker: str) -> int:
    (found,) = [
        number
        for number, held in enumerate(text.splitlines(), 1)
        if held.rstrip().endswith(f"# {marker}")
    ]
    return found


def _build(tmp_path: Path, arms: list[tuple[str, str, str]]) -> MapFragment:
    """Read SPEC and PROXY as the build does, with one junction per (owner, marker, subject) of *arms*."""
    root = tmp_path / PACKAGE
    root.mkdir(parents=True, exist_ok=True)
    (root / "__init__.py").write_text("")
    modules = []
    for name, text in {"spec": SPEC, "proxy": PROXY}.items():
        (root / f"{name}.py").write_text(text)
        tree = ast.parse(text)
        pathcond.attach_parents(tree)
        modules.append(
            SourceModule(
                package=PACKAGE,
                rel_path=f"{PACKAGE}/{name}.py",
                tree=tree,
                source=text,
                blob_sha=blob_sha(text),
            )
        )
    calls = sqlwrite.corpus_calls(modules, {PACKAGE: root})
    with sql.across_calls(calls):
        junctions = [
            JunctionNode(
                name=f"m:{owner}:{marker}",
                map_id="m",
                derived_from="v",
                subject=subject,
                owner=f"{PACKAGE}/proxy.py::{owner}",
                anchor=Anchor(
                    package=PACKAGE,
                    file=f"{PACKAGE}/proxy.py",
                    line_start=_line(PROXY, marker),
                ),
                branches=[Branch(outcome=marker, line=_line(PROXY, marker))],
            )
            for owner, marker, subject in arms
        ]
        fragment = MapFragment(
            map_id="m",
            derived_from="v",
            entities=[one.model_copy() for one in ENTITIES],
            junctions=junctions,
            reach_graph=reach.graph(calls, "m", "v", _DatabaseLeaves()),
        )
        panda_objects.attach(fragment, modules, calls)
        offrow.attach(fragment, modules, calls)
    return fragment


def _off(fragment: MapFragment, marker: str) -> OffRow | None:
    (branch,) = [
        b for j in fragment.junctions for b in j.branches if b.outcome == marker
    ]
    return branch.off_row


def test_a_spec_made_to_ask_or_to_carry_keys_is_written_to_no_row(
    tmp_path: Path,
) -> None:
    """getScoutJobData_JEDI's taskSpec, made to run getOutDiskSize(); checkAllEventsDone's fileSpec, a SELECT's keys."""
    fragment = _build(
        tmp_path,
        [
            ("DBProxy.ask", "O1", "JobSpec.specialHandling"),
            ("DBProxy.carry", "O2", "JobSpec.PandaID"),
        ],
    )
    asked = _off(fragment, "O1")
    assert asked is not None and asked.why == OFF_ROW_NO_WRITE and asked.writes == []
    assert asked.functions == [
        "pkg/proxy.py::DBProxy.ask",
        "pkg/spec.py::JobSpec.__init__",
        "pkg/spec.py::JobSpec.is_debug",
    ]
    carried = _off(fragment, "O2")
    assert carried is not None and carried.why == OFF_ROW_NO_WRITE


def test_a_write_of_its_table_that_takes_other_values_leaves_it_off_its_row(
    tmp_path: Path,
) -> None:
    """prepareTasksToBeFinished_JEDI: the lock is an UPDATE of the row, lockedBy bound from another name."""
    fragment = _build(tmp_path, [("DBProxy.lock", "O3", "JobSpec.specialHandling")])
    locked = _off(fragment, "O3")
    assert locked is not None and locked.why == OFF_ROW_NOT_TAKEN
    assert locked.writes == [
        f"pkg/proxy.py::DBProxy.lock:{_line(PROXY, 'O3') + 3} update jobsActive4"
    ]


def test_an_object_written_returned_handed_in_or_lost_in_a_hole_may_be_on_its_row(
    tmp_path: Path,
) -> None:
    fragment = _build(
        tmp_path,
        [
            ("DBProxy.set_column", "W1", "JobSpec.specialHandling"),
            ("DBProxy.via", "W2", "JobSpec.specialHandling"),
            ("DBProxy.make", "W3", "JobSpec.specialHandling"),
            ("DBProxy.handed", "W4", "JobSpec.specialHandling"),
            ("DBProxy.queued", "W5", "JobSpec.specialHandling"),
            ("DBProxy.kept", "W6", "JobSpec.specialHandling"),
            ("DBProxy.stashed", "W7", "JobSpec.specialHandling"),
        ],
    )
    # Its attribute a column of a write here, or of one in the function it is handed to; returned; a parameter;
    # handed to a call of no known class, which the records do not follow; put into what the caller handed; set
    # on another object by a computed name.
    assert [
        _off(fragment, marker) for marker in ("W1", "W2", "W3", "W4", "W5", "W6", "W7")
    ] == [None] * 7


def test_a_reader_of_the_rows_value_skips_an_arm_off_its_row() -> None:
    off = OffRow(why=OFF_ROW_NO_WRITE, functions=["f"])
    junction = JunctionNode(
        name="m:j",
        map_id="m",
        derived_from="v",
        subject="JobSpec.specialHandling",
        owner="pkg/proxy.py::DBProxy.ask",
        branches=[
            Branch(outcome="x", line=1, off_row=off),
            Branch(outcome="runtime(v)", line=2, tier=2),
        ],
    )
    # The arm stating the value writes no row: the run-time one is what can have, as when none states it.
    assert [b.line for b in _reaching(junction, "x")] == [2]
    assert _candidate(junction, "x").tier == 2
    junction.branches[1].off_row = off
    assert _reaching(junction, "x") == []
    destination = reach.moving_off(
        [junction], reach.ReachGraph([]), junction.subject, "y"
    )
    assert destination.arms == {}
