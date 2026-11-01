"""The prints every path to an arm passes, and the holes that name the written row -- ``codemap.panda.mustprints``."""

from __future__ import annotations

import ast
from pathlib import Path

from bamboo.codemap import trace
from bamboo.codemap.gates import must_prints_resolve
from bamboo.codemap.models import (
    LOG_FILES_FOLLOWED,
    LOG_FILES_OF_MODULE,
    MUST_PART_MARK,
    MUST_PART_MESSAGE,
    MUST_ROW_WRITTEN,
    Anchor,
    Branch,
    FunctionPrintsNode,
    JunctionNode,
    KeyFrom,
    KeySource,
    MapFragment,
    MustName,
    SourceModule,
    TableRecord,
)
from bamboo.codemap.panda import mustprints
from bamboo.codemap.panda.plugin import _merge_junctions

PACKAGE = "pkg"
FILE = f"{PACKAGE}/m"
DECLARED = {"JobSpec": {"PandaID", "jobStatus", "computingSite"}}
TABLES = (
    TableRecord(
        name="jobsactive4", rows_of="JobSpec", identity_attrs=["JobSpec.PandaID"]
    ),
    TableRecord(
        name="jedi_tasks",
        rows_of="JediTaskSpec",
        identity_attrs=["JediTaskSpec.jediTaskID"],
    ),
)


def _line(text: str, marker: str) -> int:
    (found,) = [
        number
        for number, held in enumerate(text.splitlines(), 1)
        if held.rstrip().endswith(f"# {marker}")
    ]
    return found


def _build(
    tmp_path: Path,
    text: str,
    *arms: tuple[str, str],
    subject: str = "JobSpec.jobStatus",
    others: tuple[str, ...] = (),
    key_froms: tuple[KeyFrom, ...] = (),
    declared: dict[str, set[str]] = DECLARED,
) -> MapFragment:
    """Read *text* as the build does and attach must prints to one junction per (owner, marker) of *arms*.

    *others* are the markers of the writes the first arm's branch also stands for.
    """
    root = tmp_path / PACKAGE
    root.mkdir(parents=True, exist_ok=True)
    (root / "m").write_text(text)
    nodes = [
        FunctionPrintsNode(
            name=f"m:{found.owner}",
            map_id="m",
            derived_from="v",
            owner=found.owner,
            cls=found.cls,
            function=found.func.name,
            count=len(found.statements),
            statements=[
                statement.model_copy(
                    update={
                        "log_files": ["a.log"],
                        "log_files_basis": LOG_FILES_FOLLOWED,
                    }
                )
                for statement in found.statements
            ],
            anchor=Anchor(
                package=PACKAGE,
                file=FILE,
                line_start=found.func.lineno,
                line_end=found.func.end_lineno,
            ),
        )
        for found in trace.function_prints({PACKAGE: root}, [FILE])
    ]
    junctions = [
        JunctionNode(
            name=f"m:{owner}:{marker}",
            map_id="m",
            derived_from="v",
            subject=subject,
            owner=f"{FILE}::{owner}",
            anchor=Anchor(package=PACKAGE, file=FILE, line_start=_line(text, marker)),
            branches=[
                Branch(
                    outcome=marker,
                    line=_line(text, marker),
                    other_lines=[_line(text, other) for other in others]
                    if index == 0
                    else [],
                )
            ],
        )
        for index, (owner, marker) in enumerate(arms)
    ]
    module = SourceModule(
        package=PACKAGE, rel_path=FILE, tree=ast.parse(text), source=text, blob_sha="0"
    )
    mustprints.attach(junctions, nodes, [module], declared, key_froms, TABLES)
    return MapFragment(
        map_id="m", derived_from="v", junctions=junctions, function_prints=nodes
    )


def _must(fragment: MapFragment, text: str, marker: str) -> list[str]:
    """The markers of the must prints of the arm at *marker*."""
    (branch,) = [
        b for j in fragment.junctions for b in j.branches if b.outcome == marker
    ]
    assert branch.must_prints is not None, branch.must_hole
    lines = {
        number: held.rsplit("# ", 1)[-1]
        for number, held in enumerate(text.splitlines(), 1)
        if "# " in held
    }
    return [lines[line] for line in branch.must_prints]


def _branch(fragment: MapFragment, marker: str) -> Branch:
    (branch,) = [
        b for j in fragment.junctions for b in j.branches if b.outcome == marker
    ]
    return branch


FLOW = """\
import logging
log = logging.getLogger("x")

def f(job, c, items):
    log.debug("start")  # P0
    if c:
        log.debug("one side")  # P1
    else:
        log.info("other side")  # P2
    job.jobStatus = "a"  # A1
    if c:
        log.debug("returning")  # P3
        return
    for item in items:
        log.debug(f"item {item}")  # P4
        job.jobStatus = "b"  # A2
    job.jobStatus = "c"  # A3
    try:
        log.debug("trying")  # P5
        risky()
    except Exception:
        log.error("failed")  # P6
    finally:
        log.debug("finally")  # P7
    job.jobStatus = "d"  # A4
"""


def test_one_side_of_an_if_is_not_a_must_print(tmp_path: Path) -> None:
    fragment = _build(tmp_path, FLOW, ("f", "A1"))
    assert _must(fragment, FLOW, "A1") == ["P0"]


def test_a_print_on_a_path_that_returns_is_not_a_must_print(tmp_path: Path) -> None:
    fragment = _build(tmp_path, FLOW, ("f", "A3"))
    assert "P3" not in _must(fragment, FLOW, "A3")


def test_a_loop_may_run_no_times(tmp_path: Path) -> None:
    fragment = _build(tmp_path, FLOW, ("f", "A2"), ("f", "A3"))
    assert _must(fragment, FLOW, "A2") == ["P0", "P4"]
    assert _must(fragment, FLOW, "A3") == ["P0"]


def test_a_try_whose_handler_goes_on_is_not_a_must_print_and_finally_is(
    tmp_path: Path,
) -> None:
    fragment = _build(tmp_path, FLOW, ("f", "A4"))
    # P5 can raise before it has printed, and the handler goes on to the arm.
    assert _must(fragment, FLOW, "A4") == ["P0", "P7"]


NAMES = """\
import logging
log = logging.getLogger("x")

def plain(job):
    log.debug(f"PandaID={job.PandaID} site={job.computingSite} n={job.nEvents}")  # P0
    job.jobStatus = "a"  # A1

def rebound(job, other):
    log.debug(f"PandaID={job.PandaID}")  # P0
    job = other
    job.jobStatus = "a"  # A2

def stored(job):
    log.debug(f"PandaID={job.PandaID}")  # P0
    job.PandaID = 7
    job.jobStatus = "a"  # A3

def each(jobs):
    for job in jobs:
        log.debug(f"PandaID={job.PandaID}")  # P0
        job.jobStatus = "a"  # A4

def later(job, nexts):
    while nexts:
        log.debug(f"PandaID={job.PandaID}")  # P0
        if nexts.pop():
            job = nexts.pop()
        job.jobStatus = "a"  # A5

def handed(job, other):
    log.debug(f"PandaID={job.PandaID}")  # P0
    other.retry(job)
    job.jobStatus = "a"  # A8

def called(job, other):
    log.debug(f"PandaID={job.PandaID}")  # P0
    job.reset()
    job.jobStatus = "a"  # A9

def apart(job, other):
    log.debug(f"PandaID={job.PandaID}")  # P0
    other.retry(other.PandaID)
    job.jobStatus = "a"  # A10

class Holder:
    def held(self, other):
        log.debug(f"PandaID={self.job.PandaID}")  # P0
        self.job.jobStatus = "a"  # A6

    def replaced(self, other):
        log.debug(f"PandaID={self.job.PandaID}")  # P0
        self.job = other
        self.job.jobStatus = "a"  # A7
"""


def _names(fragment: MapFragment, marker: str) -> list[tuple[str, str, str]]:
    return [
        (name.expression, name.column, name.row)
        for name in _branch(fragment, marker).must_names or []
    ]


def test_a_hole_on_the_written_object_names_the_row(tmp_path: Path) -> None:
    fragment = _build(tmp_path, NAMES, ("plain", "A1"))
    # nEvents is no column JobSpec declares here.
    assert _names(fragment, "A1") == [
        ("job.PandaID", "JobSpec.PandaID", MUST_ROW_WRITTEN),
        ("job.computingSite", "JobSpec.computingSite", MUST_ROW_WRITTEN),
    ]


def test_a_rebinding_between_print_and_arm_says_no_name(tmp_path: Path) -> None:
    fragment = _build(
        tmp_path,
        NAMES,
        ("rebound", "A2"),
        ("stored", "A3"),
        ("Holder.replaced", "A7"),
        ("handed", "A8"),
        ("called", "A9"),
    )
    for marker in ("A2", "A3", "A7", "A8", "A9"):
        (name,) = _branch(fragment, marker).must_names or []
        assert (name.row, name.why) == ("", mustprints.WHY_REBOUND), marker


def test_a_call_on_another_object_does_not_unsay_a_name(tmp_path: Path) -> None:
    fragment = _build(tmp_path, NAMES, ("apart", "A10"))
    assert _names(fragment, "A10") == [
        ("job.PandaID", "JobSpec.PandaID", MUST_ROW_WRITTEN)
    ]


def test_a_loop_item_is_named_within_its_pass(tmp_path: Path) -> None:
    fragment = _build(
        tmp_path, NAMES, ("each", "A4"), ("later", "A5"), ("Holder.held", "A6")
    )
    assert _names(fragment, "A4") == [
        ("job.PandaID", "JobSpec.PandaID", MUST_ROW_WRITTEN)
    ]
    # The print of one pass, then ``job`` rebound, then the arm: the same
    # bindings reach both, and the line names the job before.
    assert _names(fragment, "A5") == [("job.PandaID", "JobSpec.PandaID", "")]
    assert _names(fragment, "A6") == [
        ("self.job.PandaID", "JobSpec.PandaID", MUST_ROW_WRITTEN)
    ]


HOLES = """\
import logging
log = logging.getLogger("x")

def sql(cur, varMap):
    log.debug("start")  # P0
    varMap[":jobStatus"] = "a"  # A1
    cur.execute("UPDATE jobs SET jobStatus=:jobStatus", varMap)

def dead(job):
    log.debug("start")  # P1
    while True:
        pass
    job.jobStatus = "a"  # A2

def main():
    class Thr:
        def run(self, job):
            log.debug(f"PandaID={job.PandaID}")  # P2
            job.jobStatus = "a"  # A3
    log.debug("main")  # P3
"""


def test_unread_names_and_arms_say_why(tmp_path: Path) -> None:
    fragment = _build(tmp_path, HOLES, ("sql", "A1"), ("dead", "A2"), ("run", "A3"))
    sql = _branch(fragment, "A1")
    assert sql.must_prints is not None and sql.must_names is None
    assert sql.must_hole == mustprints.NAMES_NOT_ATTRIBUTE
    dead = _branch(fragment, "A2")
    assert (dead.must_prints, dead.must_hole) == (None, mustprints.HOLE_NOT_REACHED)
    # The junction says ``run``; the prints' owner spells ``main.Thr.run``.
    assert _must(fragment, HOLES, "A3") == ["P2"]
    assert _names(fragment, "A3") == [
        ("job.PandaID", "JobSpec.PandaID", MUST_ROW_WRITTEN)
    ]


def test_a_reader_joins_the_lines_from_the_dump(tmp_path: Path) -> None:
    fragment = _build(tmp_path, HOLES, ("run", "A3"), ("sql", "A1"))
    junction = fragment.junctions[0]
    answer = mustprints.must_prints(
        fragment.function_prints, junction, junction.branches[0]
    )
    assert answer.prints is not None
    (printed,) = answer.prints
    # ``log`` is the module's ``logging.getLogger``: its lines carry no mark, so none is unread.
    assert (printed.line, printed.level, printed.log_files, printed.holes) == (
        _line(HOLES, "P2"),
        "debug",
        ["a.log"],
        [],
    )
    assert [name.expression for name in printed.names] == ["job.PandaID"]
    # Files read off the module are a hole of the line, and so are names not read.
    for node in fragment.function_prints:
        for statement in node.statements:
            statement.log_files_basis = LOG_FILES_OF_MODULE
    other = fragment.junctions[1]
    (held,) = (
        mustprints.must_prints(
            fragment.function_prints, other, other.branches[0]
        ).prints
        or []
    )
    assert held.holes[1:] == [mustprints.NAMES_NOT_ATTRIBUTE]
    assert held.holes[0].startswith("files read off the module")
    before = Branch(outcome="x", line=1)
    assert mustprints.must_prints(
        fragment.function_prints, junction, before
    ).why.startswith("not read")


def test_the_gate_passes_on_the_build_and_fails_on_a_line_no_reader_finds(
    tmp_path: Path,
) -> None:
    fragment = _build(tmp_path, FLOW, ("f", "A1"), ("f", "A4"))
    assert must_prints_resolve(fragment).passed
    _branch(fragment, "A1").must_prints = [_line(FLOW, "A4")]
    _branch(fragment, "A4").must_hole = ""
    _branch(fragment, "A4").must_prints = None
    result = must_prints_resolve(fragment)
    assert not result.passed and len(result.failures) == 2


SITES = """\
import logging
log = logging.getLogger("x")

def two(job, c):
    log.debug(f"start {job.PandaID}")  # P0
    if c:
        log.debug("c")  # P1
        job.jobStatus = "x"  # A1
    else:
        job = c
        job.jobStatus = "x"  # A2
"""


def test_a_branch_standing_for_two_writes_has_what_both_have(tmp_path: Path) -> None:
    alone = _build(tmp_path, SITES, ("two", "A1"))
    assert _must(alone, SITES, "A1") == ["P0", "P1"]
    assert _names(alone, "A1") == [("job.PandaID", "JobSpec.PandaID", MUST_ROW_WRITTEN)]
    both = _build(tmp_path, SITES, ("two", "A1"), others=("A2",))
    assert _must(both, SITES, "A1") == ["P0"]
    # At A2 ``job`` is another object than the one P0 printed.
    assert _names(both, "A1") == [("job.PandaID", "JobSpec.PandaID", "")]


def test_merged_junctions_keep_the_line_of_the_write_they_drop() -> None:
    def junction(line: int, others: list[int]) -> JunctionNode:
        return JunctionNode(
            name="m:s",
            map_id="m",
            derived_from="v",
            subject="JobSpec.jobStatus",
            owner=f"{FILE}::f",
            branches=[Branch(outcome="failed", line=line, other_lines=others)],
        )

    (merged,) = _merge_junctions([junction(5, []), junction(9, [12]), junction(5, [])])
    assert [(b.line, b.other_lines) for b in merged.branches] == [(5, [9, 12])]


MARKED = """\
import logging

_logger = logging.getLogger("m")


def f(job):
    if job is None:
        tmp_id = None
    else:
        tmp_id = job.PandaID
    tmp_log = LogWrapper(_logger, f"PandaID={tmp_id}")
    if job is None:
        return
    tmp_log.debug("start")  # P
    job.jobStatus = "activated"  # A


def g(job, other):
    tmp_id = job.PandaID
    tmp_log = LogWrapper(_logger, f"PandaID={tmp_id}")
    tmp_log.debug("start")  # Q
    job = other
    job.jobStatus = "activated"  # B
"""


def test_a_mark_names_the_written_row_where_the_logger_is_made_after_its_key(
    tmp_path: Path,
) -> None:
    """``activateJob`` makes its logger with ``f"PandaID={tmp_id}"`` after ``tmp_id = job.PandaID``, and every
    line it prints carries that mark: the mark names the job the arm writes.  ``tmp_id = None`` is on no run
    that reaches the arm, which the same ``if job is None`` leaves.  Where ``job`` is bound again before the arm,
    the mark names another job."""
    fragment = _build(tmp_path, MARKED, ("f", "A"), ("g", "B"))

    assert _branch(fragment, "A").must_names == [
        MustName(
            line=_line(MARKED, "P"),
            expression="tmp_id",
            column="JobSpec.PandaID",
            row=MUST_ROW_WRITTEN,
            part=MUST_PART_MARK,
        )
    ]
    assert _branch(fragment, "B").must_names == []


SQL_ARM = """\
import logging

_logger = logging.getLogger("m")


def h(self, jediTaskID, status):
    tmp_log = LogWrapper(_logger, f"<jediTaskID={jediTaskID}>")
    tmp_log.debug(f"set {status} for {jediTaskID}")  # S
    varMap = {}
    varMap[":jediTaskID"] = jediTaskID  # K
    varMap[":status"] = status  # W
    self.cur.execute(SQL, varMap)  # X


def k(self, jediTaskID, status):
    tmp_log = LogWrapper(_logger, f"<jediTaskID={jediTaskID}>")
    tmp_log.debug(f"set {status} for {jediTaskID}")  # T
    jediTaskID = int(jediTaskID)
    varMap = {":jediTaskID": jediTaskID}  # L
    varMap[":status"] = status  # V
    self.cur.execute(SQL, varMap)  # Y
"""


def _term(owner: str, ran: str, writes: str, filled: str) -> KeyFrom:
    """``jediTaskID=:jediTaskID`` of the UPDATE *owner* runs at *ran*, writing the status at *writes*."""
    return KeyFrom(
        owner=f"{FILE}::{owner}",
        file=FILE,
        statement=_line(SQL_ARM, ran),
        table="JEDI_Tasks",
        term="jediTaskID=:jediTaskID",
        writes={"JediTaskSpec.status": [_line(SQL_ARM, writes)]},
        sources=[
            KeySource(
                kind="parameter",
                line=_line(SQL_ARM, filled),
                name="jediTaskID",
                paths=["()"],
            )
        ],
    )


def test_an_sql_arm_is_named_by_what_its_statement_s_key_bind_is_filled_with(
    tmp_path: Path,
) -> None:
    """An SQL arm fills a bind of a statement whose ``WHERE jediTaskID=:jediTaskID`` names the rows it writes: a
    must print holding what that bind is filled with -- the message's hole, the mark's -- names them, where the
    print and the fill hold one value.  Bound again between the print and the fill, it names another task."""
    fragment = _build(
        tmp_path,
        SQL_ARM,
        ("h", "W"),
        ("k", "V"),
        subject="JediTaskSpec.status",
        key_froms=(_term("h", "X", "W", "K"), _term("k", "Y", "V", "L")),
        declared={**DECLARED, "JediTaskSpec": {"jediTaskID", "status"}},
    )

    named = _branch(fragment, "W").must_names
    assert named is not None
    assert sorted((one.expression, one.column, one.row, one.part) for one in named) == [
        ("jediTaskID", "JediTaskSpec.jediTaskID", MUST_ROW_WRITTEN, MUST_PART_MARK),
        ("jediTaskID", "JediTaskSpec.jediTaskID", MUST_ROW_WRITTEN, MUST_PART_MESSAGE),
    ]
    assert _branch(fragment, "V").must_names == []
