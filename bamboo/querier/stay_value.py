"""What a time column of the row held while its status held the observed value (M3(a)-1: analyzer handover
out/gm/qk-design.md "M3(a)-1 v2", tmaeno's OK 2026-11-14), where the rows file holds only the value after the move
(late's ``after``) and a mover's selecting statement compares the column -- copyArchive's
``modificationTime < now - 30 min`` on a job at ``sent``.  A time column is one an arm writes the time it runs
(``attr_time.NOW``); any other is not read here.

* The value as the stay began: every arm that can write the status's value has, in its pass, an arm of the column
  that writes the time it runs, tied to the same history call (the junction branches' ``HistoryPair``).  The column
  then holds the time of the history row that began the stay (part H: a history row's time is when its value was
  written).  getJobs' 2455 ``sent`` and 2452 ``CURRENT_DATE`` are both tied to its 2910 recordStatusChange.  An arm
  writing the status from a value known only at run time is read as not writing the value: a premise, said.
  Where the map ties an arm of the column to the status's arm itself (:func:`tied`: ``same_pass``, or the one arm of
  the column at its statement) that arm is its pass's, before any of the history call (bundle A, A-2); one writing
  the clock with an offset leaves the value to the timeline, the offset kept (``Before.offsets``).
* The column's arms that may write it while the value holds, the value's own list (tmaeno, 11-14: lockJobsForFinisher
  writes it only at transferring).  The map leaves an arm out where:
  - it is a birth, written as the row is made;
  - every statement the map has writing it there runs from no placed caller (``never_runs``);
  - a statement writing it there writes the status too: it writes only as the status moves, and the history keeps
    every move;
  - its row precondition asks the status another value;
  - every statement writing it there has a row-naming term that takes no row at the value (``_term_off``): a term on
    the status filled only by other literals, or one every source of which is a selecting statement's rows restricting
    the status to other values -- lockJobsForFinisher's 906 by the PandaIDs its 900 selects at transferring;
  - every statement writing it there writes a table that holds no row of the class (the map's ``TableRecord.rows_of``)
    -- archiveJob's job.modificationTime copied to metaTable and jobParamsTable.  A statement whose table the map does
    not read (``''``: updateTask_JEDI's UPDATE built from ``bindUpdateChangesExpression``) is not known to write
    another table, and leaves the arm.
  The others may write the column in the stay: the value is then not known, and they are said.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from bamboo.codemap.models import KEY_CONSTANT, KEY_ROWS
from bamboo.querier.attr_time import NOW

#: an arm's outcome that is a value known only at run time, not a literal
RUN_TIME = ("runtime(", "passthrough(")


@dataclass(frozen=True)
class Before:
    """What the map says of a time column's value while the status held the value."""

    #: the column held the time of the history row that began the stay, all the stay
    known: bool
    #: each arm writing the value, with the column's arm of its pass: "getJobs:2455 with 2452"
    starts: tuple[str, ...] = ()
    #: arms writing the value with no such arm: "func:line"
    lacking: tuple[str, ...] = ()
    #: arms writing the status from a value known at run time, read as not writing the value: "func:line"
    runtime: tuple[str, ...] = ()
    #: the column's arms that may write it while the value holds: "func:line"
    left: tuple[str, ...] = ()
    #: the column's arms left out, each with why: ("func:line", why)
    out: tuple[tuple[str, str], ...] = ()
    #: each of *starts* whose column arm writes the clock with an offset, and the offset in seconds (bundle A, A-2:
    #: updateTaskStatusByContFeeder_JEDI's 1370 'assigning' with 1373 ``naive_utcnow() - timedelta(hours=6)``, -21600)
    offsets: tuple[tuple[str, float], ...] = ()
    #: arms writing the value that make the row, left out where the row was there before the stay (A-3): "func:line"
    births: tuple[str, ...] = ()


def _name(owner: str, line) -> str:
    return f"{owner.rpartition('::')[2]}:{line}"


def _pairs(branch) -> set[tuple[str, int]]:
    """The history calls a branch's pass makes: ``(caller, call line)`` of each of its ``HistoryPair``."""
    return {(p.caller, p.call) for p in getattr(branch, "history", None) or []}


def _lines(branch) -> set[int]:
    return {branch.line, *(getattr(branch, "other_lines", None) or [])} - {None}


def _now_only(branch) -> float | None:
    """The seconds past the clock an arm writes, read only for the clock itself (:data:`attr_time.NOW`): 0; None for
    any other value.  ``before``'s callers hand a reader of the map's ``amount`` (rerun's ``_written_offset``)."""
    return 0.0 if branch.outcome in NOW else None


def tied(b, owner: str, subject: str, column: str, arms: list, key_froms: list) -> list:
    """The column's *arms* (of *owner*) the map ties to the status arm *b* (bundle A, A-2, out/gm/qk-design.md,
    tmaeno's OK 2026-11-16): those whose ``same_pass`` names *b*'s line for the status, or that *b*'s ``same_pass``
    names -- updateTaskStatusByContFeeder_JEDI's 1373 names 1370 'assigning', where its 1352 (``timeNow``) is on
    another path; else the one arm of the column at every statement *b* writes at (``KeyFrom.writes``):
    prepareTasksToBeFinished_JEDI's 4075 writes 3971 'tobroken' and lockedBy only by 3839 (NULL).  Empty where the map
    ties none, or a statement of *b*'s holds more than one arm of the column and no ``same_pass`` picks one."""
    attr, col = subject.partition(".")[2], column.partition(".")[2]
    by_pass = [c for c in arms
               if any(s.column == attr and s.line in _lines(b) for s in getattr(c, "same_pass", None) or [])
               or any(s.column == col and s.line in _lines(c) for s in getattr(b, "same_pass", None) or [])]
    if by_pass:
        return by_pass
    statements = [kf for kf in key_froms if kf.owner == owner and set(kf.writes.get(subject, [])) & _lines(b)]
    lines = {line for kf in statements for line in kf.writes.get(column, [])}
    if len(lines) != 1 or not all(kf.writes.get(column) for kf in statements):
        return []
    return [c for c in arms if _lines(c) & lines]


def time_column(column_junctions: list) -> bool:
    """Whether an arm of the column writes the time it runs."""
    return any(b.outcome in NOW for j in column_junctions for b in j.branches)


def asked_values(texts: list[str], attr: str) -> set[str] | None:
    """The values ``attr = v`` or ``attr IN (a, b)`` in *texts* ask (a branch's ``row_precondition``); None where
    none of them tests *attr*."""
    found: set[str] = set()
    tested = False
    for text in texts or []:
        m = re.fullmatch(rf"\s*{re.escape(attr)}\s*(?:=\s*(.+?)|\s+IN\s*\((.*)\))\s*", text, flags=re.I)
        if not m:
            continue
        tested = True
        found |= {v.strip().strip("'\"") for v in (m[1] or m[2] or "").split(",") if v.strip()}
    return found if tested else None


def restricted_values(statement, subject: str) -> set[str] | None:
    """The values a selecting statement restricts *subject* to (its ``restricts`` pairs); None where it does not."""
    values = {p.partition("=")[2] for p in getattr(statement, "restricts", None) or [] if p.partition("=")[0] == subject}
    return values or None


def before(status_junctions: list, column_junctions: list, subject: str, value: str, column: str, key_froms: list,
           running: Callable[[object], bool], read_at: Callable[[str, int], object | None], row_tables: set[str],
           born: Callable[[str, object], bool], nulls: bool = False, existed: bool = False,
           status_born: Callable[[str, object], bool] | None = None,
           offset_of: Callable[[object], float | None] = _now_only) -> Before | None:
    """*column* (``Class.attr``) of a row while *subject* held *value*: see the module; None for a column no arm
    writes the time it runs.  *status_junctions* and *column_junctions* are the map's writers of the two; *key_froms*
    the dump's KeyFroms (each statement's row-naming terms, the subjects it writes by line, its table, its sources),
    *running* whether a placed caller may run one; *read_at* (owner, line) the map's selecting statement there;
    *row_tables* the tables (lower case) that hold rows of the column's class; *born* (owner, branch) whether the
    branch makes the row.

    *nulls*: the column is one a selecting statement tests ``IS NULL`` (Q1 batch 2 part 1), its states read on a
    timeline: an arm of the value's pass may be tied with any arm of it, whatever that writes, each such arm kept
    (``starts``: "func:line with l1,l2"), and the value is never ``known`` here.

    *existed* (bundle A, A-3): the row's status history holds a state before the stay's, so an arm writing the value
    that makes the row (*status_born*, owner and branch) did not begin it: left out (``births``) --
    getTasksToRefine_JEDI's 1190 INSERT of a task at 'registered' for NT-registered-late, whose stay
    reactivatePendingTasks_JEDI's ``SET status=oldStatus`` began.  *offset_of*: the seconds past the clock an arm writes
    (rerun.s ``_written_offset``), None for no clock; the default reads the clock itself only."""
    if not nulls and not time_column(column_junctions):
        return None
    attr = subject.partition(".")[2]
    now_arms: dict[str, list] = {}
    column_arms: dict[str, list] = {}
    for j in column_junctions:
        for b in j.branches:
            column_arms.setdefault(j.owner, []).append(b)
            if nulls or b.outcome in NOW:
                now_arms.setdefault(j.owner, []).append(b)
    starts, lacking, runtime, births = [], [], [], []
    offsets: list[tuple[str, float]] = []
    for j in status_junctions:
        if j.subject != subject:
            continue
        for b in j.branches:
            outcome = b.outcome or ""
            if outcome.startswith(RUN_TIME):
                runtime.append(_name(j.owner, b.line))
                continue
            if outcome.strip("'\"") != value:
                continue
            if existed and status_born is not None and status_born(j.owner, b):
                births.append(_name(j.owner, b.line))
                continue
            # A-2: the column's arms the map ties to this one (same_pass, or its statement's one arm of the column) --
            # a clock with an offset among them -- before those of its history call
            ties = [c for c in tied(b, j.owner, subject, column, column_arms.get(j.owner, []), key_froms)
                    if nulls or offset_of(c) is not None]
            if ties:
                mates = sorted({c.line for c in ties})
            else:
                pairs = _pairs(b)
                mates = sorted(c.line for c in now_arms.get(j.owner, []) if pairs & _pairs(c))
            if mates:
                said = f"{_name(j.owner, b.line)} with {','.join(map(str, mates)) if nulls else mates[0]}"
                starts.append(said)
                shift = {offset_of(c) for c in ties if c.line == mates[0]} if ties and not nulls else {0.0}
                if shift - {0.0, None}:
                    offsets.append((said, min(s for s in shift if s is not None)))
            else:
                lacking.append(_name(j.owner, b.line))
    left: list[str] = []
    out: list[tuple[str, str]] = []
    for j in column_junctions:
        for b in j.branches:
            name, lines = _name(j.owner, b.line), _lines(b)
            every = [kf for kf in key_froms if kf.owner == j.owner and set(kf.writes.get(column, [])) & lines]
            statements = [kf for kf in every if running(kf)]
            asked = asked_values(getattr(b, "row_precondition", None) or [], attr)
            if born(j.owner, b):
                out.append((name, "a birth: written as the row is made"))
            elif every and not statements:
                out.append((name, f"its statement(s) {sorted({kf.statement for kf in every})} run from no placed caller"))
            elif with_status := next((kf for kf in statements if subject in kf.writes), None):
                out.append((name, f"its statement at {with_status.statement} writes {attr} too"))
            elif asked is not None and value not in asked:
                out.append((name, f"its row precondition asks {attr} in {sorted(asked)}"))
            elif statements and all(offs := [_statement_off(terms, subject, value, read_at)
                                             for terms in _by_statement(statements).values()]):
                out.append((name, "; ".join(sorted(set(offs)))))
            elif statements and all(kf.table for kf in statements) and not {kf.table.lower() for kf in statements} & row_tables:
                out.append((name, f"it writes {sorted({kf.table for kf in statements if kf.table})}, which hold no row of "
                                  f"{column.partition('.')[0]}"))
            else:
                left.append(name)
    # a start whose column arm writes the clock with an offset leaves the value to the timeline (``offsets``)
    known = bool(starts) and not lacking and not left and not nulls and not offsets
    return Before(known, tuple(starts), tuple(lacking), tuple(runtime), tuple(left), tuple(out), tuple(offsets),
                  tuple(births))


def _by_statement(key_froms: list) -> dict[tuple[str, int], list]:
    """KeyFroms by their statement: the map keeps one for each row-naming term of a statement."""
    out: dict[tuple[str, int], list] = {}
    for kf in key_froms:
        out.setdefault((kf.owner, kf.statement), []).append(kf)
    return out


def _statement_off(terms: list, subject: str, value: str, read_at: Callable[[str, int], object | None]) -> str:
    """Why a statement, by its terms' KeyFroms, takes no row holding *value*: one of its terms takes none
    (``_term_off``); "" where none of them says so."""
    return next((said for kf in terms if (said := _term_off(kf, subject, value, read_at))), "")


def _term_off(kf, subject: str, value: str, read_at: Callable[[str, int], object | None]) -> str:
    """Why a statement's row-naming term (its KeyFrom) takes no row holding *value*: a term on *subject*'s column
    filled only by literals of other values (updateJobStatus' 297 asks jobStatus 'waiting'), or a term every source of
    which is a selecting statement's rows restricting *subject* to other values (lockJobsForFinisher's 906 by the
    PandaIDs its 900 selects at transferring) -- rows of another function's statement only where the map places every
    route into the statement (``routes_open`` false).  A source of any other kind (a parameter an API fills: killJob's
    pandaID) leaves the term open: ""."""
    attr = subject.partition(".")[2]
    sources = getattr(kf, "sources", None) or []
    if not sources:
        return ""
    column = re.split(r"\s*(?:!=|<>|>=|<=|=|<|>|\bIN\b)", kf.term or "", maxsplit=1, flags=re.I)[0].strip()
    if column.lower() == attr.lower() and all(getattr(s, "kind", "") == KEY_CONSTANT for s in sources):
        constants = {s.name.strip("'\"") for s in sources}
        return f"its statement at {kf.statement} asks {attr} in {sorted(constants)}" if value not in constants else ""
    said = []
    for s in sources:
        owner = getattr(s, "site_owner", "") or kf.owner
        if getattr(s, "kind", "") != KEY_ROWS or not getattr(s, "site_line", None):
            return ""
        if owner != kf.owner and getattr(kf, "routes_open", True) is not False:
            return ""
        rs = read_at(owner, s.site_line)
        values = restricted_values(rs, subject) if rs is not None else None
        if values is None or value in values:
            return ""
        said.append(f"{owner.rpartition('::')[2]}@{s.site_line} ({attr} in {sorted(values)})")
    return f"its statement at {kf.statement} takes the rows of {', '.join(said)}"
