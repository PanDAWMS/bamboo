"""Arms that cannot leave a value other than the row's on a row made before them -- ``codemap.panda.aftermade``."""

from __future__ import annotations

import ast
from pathlib import Path

from bamboo.codemap import pathcond, reach
from bamboo.codemap.gitsource import blob_sha
from bamboo.codemap.models import (
    AFTER_MADE_KEY_ELSEWHERE,
    AFTER_MADE_KEY_SET,
    AFTER_MADE_MADE_BEFORE,
    AFTER_MADE_NEW_ROW,
    AFTER_MADE_NO_ROW_YET,
    AFTER_MADE_ROW_OBJECT,
    AFTER_MADE_SAME_VALUE_WHERE,
    AfterMade,
    Anchor,
    Branch,
    EntityNode,
    JunctionNode,
    KeyFrom,
    MapFragment,
    SourceModule,
    TableRecord,
)
from bamboo.codemap.panda import aftermade, offrow, sql
from bamboo.codemap.panda import objects as panda_objects
from bamboo.codemap.panda.plugin import _DatabaseLeaves
from bamboo.codemap.panda.recognizers import sqlwrite

PACKAGE = "pkg"

SPEC = """\
class JobSpec:
    _attributes = ("PandaID", "jobName", "jobStatus")

    def __init__(self):
        for attr in self._attributes:
            object.__setattr__(self, attr, None)

    def pack(self, values):
        for i in range(len(self._attributes)):
            attr = self._attributes[i]
            val = values[i]
            object.__setattr__(self, attr, val)

    def bindUpdateChangesExpression(self):
        ret = ""
        for attr in self._attributes:
            ret += f"{attr}=:{attr},"
        return ret[:-1] + " "

    def valuesMap(self, onlyChanged=False):
        ret = {}
        for attr in self._attributes:
            ret[f":{attr}"] = getattr(self, attr)
        return ret
"""

PROXY = """\
from pkg.spec import JobSpec


class Keeper:
    def keep(self, job):
        self.kept = job


class DBProxy:
    def generate(self, name):
        job = JobSpec()
        job.jobName = name  # A_NEW
        return [job]

    def store(self, jobs):
        ids = self.fetch_ids(len(jobs))
        for i, job in enumerate(jobs):
            job.PandaID = ids[i]  # K_STORE
            self.insert(job)

    def fetch_ids(self, n):
        self.cur.execute("SELECT ATLAS_PANDA.JOBSDEFINED4_PANDAID_SEQ.nextval FROM dual", {":n": n})
        res = self.cur.fetchall()
        ids = []
        for (one,) in res:
            ids.append(one)
        return sorted(ids)

    def insert(self, job):
        self.cur.execute("INSERT INTO ATLAS_PANDA.jobsDefined4 (PandaID,jobName,jobStatus) VALUES (:PandaID,:jobName,:jobStatus)", job.valuesMap())

    def peek(self, panda_id):
        job = JobSpec()
        self.cur.execute("SELECT PandaID,jobName,jobStatus FROM ATLAS_PANDA.jobsDefined4 WHERE PandaID=:PandaID", {":PandaID": panda_id})
        res = self.cur.fetchone()
        job.pack(res)
        job.jobStatus = "peeked"  # A_PACKED
        return job

    def update_status(self, job):
        job.jobStatus = "running"  # A_ROW
        self.cur.execute("UPDATE ATLAS_PANDA.jobsDefined4 SET " + job.bindUpdateChangesExpression() + "WHERE PandaID=:PandaID", job.valuesMap())

    def made_then_changed(self, name):
        job = JobSpec()
        job.jobName = name
        self.insert(job)
        job.jobStatus = "late"  # A_LATE

    def set_status(self, panda_id):
        var_map = {":PandaID": panda_id}
        var_map[":jobStatus"] = "held"  # S_UPDATE
        self.cur.execute("UPDATE ATLAS_PANDA.jobsDefined4 SET jobStatus=:jobStatus WHERE PandaID=:PandaID AND jobStatus=:old", var_map)

    def insert_new(self, name):
        var_map = {":jobName": name}
        var_map[":jobStatus"] = "defined"  # S_INSERT
        self.cur.execute("INSERT INTO ATLAS_PANDA.jobsDefined4 (PandaID,jobName,jobStatus) VALUES (ATLAS_PANDA.JOBSDEFINED4_PANDAID_SEQ.nextval,:jobName,:jobStatus)", var_map)

    def rekey(self, job, panda_id):
        job.PandaID = panda_id  # K_OLD

    def handed_then_set(self, name, keeper):
        job = JobSpec()
        job.jobName = name
        keeper.keep(job)  # H_KEEP
        job.jobStatus = "first"
        job.jobStatus = self.next_status()  # A_AGAIN

    def next_status(self):
        return "again"
"""

ENTITIES = [
    EntityNode(
        name="jobsdefined4",
        map_id="m",
        derived_from="v",
        spec_class="JobSpec",
        tables=["jobsDefined4"],
    )
]
ARMS = [
    ("DBProxy.generate", "A_NEW", "JobSpec.jobName"),
    ("DBProxy.store", "K_STORE", "JobSpec.PandaID"),
    ("DBProxy.peek", "A_PACKED", "JobSpec.jobStatus"),
    ("DBProxy.update_status", "A_ROW", "JobSpec.jobStatus"),
    ("DBProxy.made_then_changed", "A_LATE", "JobSpec.jobStatus"),
    ("DBProxy.set_status", "S_UPDATE", "JobSpec.jobStatus"),
]


def _line(marker: str) -> int:
    (found,) = [
        number
        for number, held in enumerate(PROXY.splitlines(), 1)
        if held.rstrip().endswith(f"# {marker}")
    ]
    return found


def _build(
    tmp_path: Path, arms: list[tuple[str, str, str]], key_froms: list[KeyFrom]
) -> MapFragment:
    """Read SPEC and PROXY as the build does, one junction per (owner, marker, subject) of *arms*, with the table's key
    and *key_froms* as the build learns them before this step."""
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
                    line_start=_line(marker),
                ),
                branches=[
                    Branch(
                        outcome=marker,
                        line=_line(marker),
                        row_precondition=["jobStatus=S_UPDATE"]
                        if marker == "S_UPDATE"
                        else [],
                    )
                ],
            )
            for owner, marker, subject in arms
        ]
        fragment = MapFragment(
            map_id="m",
            derived_from="v",
            entities=[one.model_copy() for one in ENTITIES],
            junctions=junctions,
            reach_graph=reach.graph(calls, "m", "v", _DatabaseLeaves()),
            tables=[
                TableRecord(
                    name="jobsdefined4",
                    identity_attrs=["JobSpec.PandaID"],
                    spec_class="JobSpec",
                )
            ],
            key_froms=key_froms,
        )
        panda_objects.attach(fragment, modules, calls)
        offrow.attach(fragment, modules, calls)
        aftermade.attach(
            fragment, modules, calls, reach.ReachGraph(fragment.reach_graph)
        )
    return fragment


def _made(fragment: MapFragment, marker: str) -> AfterMade | None:
    (branch,) = [
        b for j in fragment.junctions for b in j.branches if b.outcome == marker
    ]
    return branch.after_made


def _update_key(owner: str, marker: str) -> KeyFrom:
    """The KeyFrom the build reads for the UPDATE after *marker*: its row named by the object's key."""
    return KeyFrom(
        owner=f"{PACKAGE}/proxy.py::{owner}",
        file=f"{PACKAGE}/proxy.py",
        statement=_line(marker) + 1,
        table="jobsDefined4",
        term="PandaID=:PandaID",
        object="job",
        attr="JobSpec.PandaID",
    )


def test_a_new_object_given_no_key_has_no_row_yet_where_every_key_set_elsewhere_is_new(
    tmp_path: Path,
) -> None:
    fragment = _build(tmp_path, ARMS, [_update_key("update_status", "A_ROW")])

    assert _made(fragment, "A_NEW") == AfterMade(
        counted=False, by=AFTER_MADE_NO_ROW_YET
    )


def test_a_row_object_a_packed_one_and_one_made_before_are_counted(
    tmp_path: Path,
) -> None:
    fragment = _build(tmp_path, ARMS, [_update_key("update_status", "A_ROW")])

    row = _made(fragment, "A_ROW")
    assert row is not None and row.counted and AFTER_MADE_ROW_OBJECT in row.why
    packed = _made(fragment, "A_PACKED")
    assert packed is not None and packed.counted and AFTER_MADE_KEY_SET in packed.why
    late = _made(fragment, "A_LATE")
    assert late is not None and late.counted and late.why == [AFTER_MADE_MADE_BEFORE]
    assert late.lines == [_line("A_LATE") - 1]


def test_a_place_that_may_make_the_row_comes_before_an_arm_whatever_is_assigned_between(
    tmp_path: Path,
) -> None:
    # The hole may write the job's row; that the status is set again between does not unmake it.  The arm's line is a
    # call's too (clonePandaJob's FileSpec defines __setattr__), so the record says the status is overwritten between.
    arms = [*ARMS, ("DBProxy.handed_then_set", "A_AGAIN", "JobSpec.jobStatus")]
    fragment = _build(tmp_path, arms, [_update_key("update_status", "A_ROW")])

    again = _made(fragment, "A_AGAIN")
    assert again is not None and again.counted and AFTER_MADE_MADE_BEFORE in again.why
    assert again.lines == [_line("H_KEEP")]


def test_an_sql_arm_is_counted_by_its_update_unless_its_where_asks_the_same_value(
    tmp_path: Path,
) -> None:
    fragment = _build(tmp_path, ARMS, [_update_key("update_status", "A_ROW")])

    same = _made(fragment, "S_UPDATE")
    assert same == AfterMade(counted=False, by=AFTER_MADE_SAME_VALUE_WHERE)
    other = _build(
        tmp_path / "b", [("DBProxy.set_status", "S_UPDATE", "JobSpec.jobName")], []
    )
    counted = _made(other, "S_UPDATE")
    assert counted is not None and counted.counted


def test_an_old_key_set_on_a_handed_object_breaks_no_row_yet_for_the_class(
    tmp_path: Path,
) -> None:
    fragment = _build(
        tmp_path,
        [*ARMS, ("DBProxy.rekey", "K_OLD", "JobSpec.PandaID")],
        [_update_key("update_status", "A_ROW")],
    )

    broken = _made(fragment, "A_NEW")
    assert (
        broken is not None
        and broken.counted
        and broken.why == [AFTER_MADE_KEY_ELSEWHERE]
    )
    assert broken.elsewhere == [f"{PACKAGE}/proxy.py::DBProxy.rekey:{_line('K_OLD')}"]


def test_a_whole_update_naming_its_row_by_another_key_breaks_no_row_yet(
    tmp_path: Path,
) -> None:
    fragment = _build(tmp_path, ARMS, [])

    broken = _made(fragment, "A_NEW")
    assert broken is not None and broken.why == [AFTER_MADE_KEY_ELSEWHERE]
    assert broken.elsewhere == [
        f"{PACKAGE}/proxy.py::DBProxy.update_status:{_line('A_ROW') + 1}"
    ]


def test_an_insert_making_its_row_takes_an_sql_arm_out(tmp_path: Path) -> None:
    fragment = _build(
        tmp_path, [("DBProxy.insert_new", "S_INSERT", "JobSpec.jobStatus")], []
    )

    assert _made(fragment, "S_INSERT") == AfterMade(
        counted=False, by=AFTER_MADE_NEW_ROW
    )
