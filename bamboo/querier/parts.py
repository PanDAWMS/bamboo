"""A stand-in for the PanDA API parts the querier asks rows of.

The status log and the like come through APIs PanDA
provides, and a job/task status history API is a shared part many cases use.
The querier combines parts and never names a table.  Until PanDA answers,
this module draws each question as SQL, and every table name lives here:
the querier hands a part a question as a structure -- the class, the key or
the transition, the window, the terms -- and prints what comes back.  When
the API exists, the same structures become its calls and this module goes.

Each history states its own shape: the subject it records and the columns
of the row it holds besides the key, the value and the time.  The querier
matches a call's WHERE against that shape; a querier that needed a branch
per class would have turned the glue between parts into a per-case table.

* H, status history: by key (the rows file's ``statuslog`` for now) and by
  transition and window (:class:`MovedOn`).
* A1, a row by key wherever it lives now (live, archived, the archive
  schema), for the set of keys inside :class:`MovedOn`.
* A2, a statement's rows counted now (:class:`GroupsNow`).  The API takes the
  map's statement and the call's binds; the stand-in takes the table and the
  WHERE the call printed.
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass, field

ORACLE_TIME = "'YYYY-MM-DD HH24:MI:SS'"


@dataclass(frozen=True)
class History:
    """What part H states about the history of one subject."""

    subject: str  # ``Class.attr`` whose moves it records
    held: frozenset[str]  # the other columns of the row it holds, lower case: what a question may filter on
    created: str = ""  # the class's attribute set once, as its row is made: the earliest the first row's values came


#: the stand-in's knowledge of where each history is kept (base_module.py recordStatusChange and
#: record_task_status_change) and what orders its rows: the job's modificationTime is CURRENT_DATE, to the second,
#: so modiftime_extended (CURRENT_TIMESTAMP) breaks ties; the value column is spelled as the subject's attribute.
#: A job's creationTime is set only just before its INSERT (job_complex_module.py:3034, 4172, 5490,
#: task_event_module.py:4493) and no statement updates it; the task's is not checked
_HISTORIES = {
    "JobSpec.jobStatus": (
        History("JobSpec.jobStatus", frozenset({"computingsite", "prodsourcelabel", "cloud"}), "creationTime"),
        "ATLAS_PANDA.jobs_StatusLog",
        "modificationTime, modiftime_extended",
    ),
    "JediTaskSpec.status": (
        History("JediTaskSpec.status", frozenset({"attemptnr", "reason"})),
        "ATLAS_PANDA.TASKS_STATUSLOG",
        "modificationTime",
    ),
}
#: where a row of the class lives now, and a term bounding a table to the window's start ("{start}");
#: the archive schema is partitioned by modificationTime
_ROW_TABLES = {
    "JobSpec": [
        ("ATLAS_PANDA.jobsDefined4", ""),
        ("ATLAS_PANDA.jobsActive4", ""),
        ("ATLAS_PANDA.jobsArchived4", ""),
        ("ATLAS_PANDAARCH.jobsArchived", "modificationTime >= {start}"),
    ],
    "JediTaskSpec": [("ATLAS_PANDA.JEDI_Tasks", "")],
}


def history(subject: str) -> History | None:
    """Part H's shape for ``Class.attr``; None when no history records the subject."""
    found = _HISTORIES.get(subject)
    return found[0] if found else None


def history_of_class(cls: str) -> History | None:
    """The history of *cls*'s status: what a rows file's ``statuslog`` holds for a row of the class; None for a class
    no history records."""
    return next((found for found, _, _ in _HISTORIES.values() if found.subject.partition(".")[0] == cls), None)


def history_holding(subject: str) -> History | None:
    """The history whose rows hold ``Class.attr`` beside the status they record (``JobSpec.computingSite`` in the
    job's); None for a status itself and for a subject no history holds."""
    cls, _, attr = subject.partition(".")
    for found, _, _ in _HISTORIES.values():
        if found.subject.partition(".")[0] == cls and attr.lower() in found.held:
            return found
    return None


def history_by_table(name: str) -> History | None:
    """The history kept in the table a map names (``jobs_statuslog``, ``jobs_StatusLog``, with or without the schema):
    how a statement joining the log is read against a row's history.  The map spells a class-less table as each
    statement's SQL does, so the name is matched in any case."""
    for found, log, _ in _HISTORIES.values():
        if log.split(".")[-1].lower() == name.split(".")[-1].lower():
            return found
    return None


def history_column(found: History, attr: str) -> str | None:
    """Which field of the status history by key (the rows file's ``statuslog`` states, part H's stand-in) holds the
    log's column *attr*: the value (``status``) or the time (``time``; ``time_extended`` breaks ties).  None for a
    column the log has and the states do not carry (``computingSite``)."""
    attr = attr.lower()
    if attr == found.subject.partition(".")[2].lower():
        return "status"
    return {"modificationtime": "time", "modiftime_extended": "time_extended"}.get(attr)


@dataclass
class GroupsNow:
    """A2: the rows that pass a call's WHERE now, counted by columns, with the least key of each group."""

    statement: str  # ``owner@line``, what the API takes
    table: str  # the stand-in's: the table the call printed
    where: str  # the call's WHERE, its binds put in
    cols: list[tuple[str, str]]  # (column, asc|desc|ranked)
    key: str
    blank: list[str]  # order terms the map leaves blank, said and left out
    #: a blank order term the map says ranks by a column (its ``order_holes``): (term, column, where the ranking is
    #: computed at run time) -- grouped by, its ranking not known here
    ranked: list[tuple[str, str, str]] = field(default_factory=list)


@dataclass
class MovedOn:
    """H and A1: the rows whose subject left the observed value in a window, under terms the history holds, by
    the value each went to, with how long each had held it, counted by columns of the row as it is now.

    Every next value, not only the one this row went to: a group starved of one way out leaves by another (a
    7-day kill), and only the two side by side say so.  Counts are moves, not keys."""

    history: History
    observed: str
    since: datetime.datetime
    moved: datetime.datetime
    terms: list[str]  # the call's WHERE terms on columns the history holds
    cols: list[tuple[str, str]]  # (column, asc|desc|ranked)
    key: str


def sql_literal(value) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


def _oracle_time(t: datetime.datetime) -> str:
    return f"TO_DATE('{t:%Y-%m-%d %H:%M:%S}', {ORACLE_TIME})"


def draw(q: GroupsNow | MovedOn) -> str:
    """The question as the SQL that answers it."""
    return _groups_now_sql(q) if isinstance(q, GroupsNow) else _moved_on_sql(q)


def db_read(q: GroupsNow | MovedOn) -> dict | None:
    """The question as a reader of one table by a WHERE, counting by columns, takes it: a GroupsNow's table, WHERE,
    the columns it groups by and the key it takes the least of.  None for a MovedOn, a window over the history (LAG),
    which only its SQL says."""
    if not isinstance(q, GroupsNow):
        return None
    return {"table": q.table, "where": q.where, "group_by": [c for c, _ in q.cols], "least": q.key}


def _sorted(c: str, d: str, alias: str = "") -> str:
    """An ORDER BY item: a ranked column (ranked at run time, not here) by itself."""
    return f"{alias}{c}" if d == "ranked" else f"{alias}{c} {d}"


def _groups_now_sql(q: GroupsNow) -> str:
    names = ", ".join(c for c, _ in q.cols)
    return "\n".join([
        "-- the rows that pass the call's WHERE now, by the order's columns"
        + "".join(f"; {c} for {t!r}, ranked at run time ({at})" for t, c, at in q.ranked)
        + (f"; left out, blank in the map: {q.blank}" if q.blank else ""),
        f"SELECT {names}, COUNT(*) AS n, MIN({q.key}) AS min_{q.key}",
        f"FROM {q.table}",
        f"WHERE {q.where}",
        f"GROUP BY {names}",
        f"ORDER BY {', '.join(_sorted(c, d) for c, d in q.cols)};",
    ])


def _moved_on_sql(q: MovedOn) -> str:
    _, log, ordered_by = _HISTORIES[q.history.subject]
    cls, _, value = q.history.subject.partition(".")
    key = q.key
    names = ", ".join(c for c, _ in q.cols)
    start, end = _oracle_time(q.since), _oracle_time(q.moved)
    tables = [
        f"    SELECT {key}, {names} FROM {t} WHERE {key} IN (SELECT {key} FROM s)"
        + (f" AND {bound.format(start=start)}" if bound else "")
        for t, bound in _ROW_TABLES[cls]
    ]
    took = "s.moved_at - s.since_at"
    order = "ORDER BY " + ", ".join(f"l.{c.strip()}" for c in ordered_by.split(","))
    return "\n".join([
        f"-- the rows that left {q.observed!r} in the row's window under the call's terms the status log holds, by the"
        f" value each went to; hours each had held it, by the order's columns as they are now (moves, not keys)",
        "WITH n AS (",
        f"  SELECT DISTINCT {key} FROM {log}",
        f"  WHERE modificationTime >= {start} AND modificationTime <= {end}" + "".join(f" AND {t}" for t in q.terms) + "),",
        "p AS (",
        f"  SELECT l.{key}, l.{value} AS next_value, l.modificationTime AS moved_at,",
        f"         LAG(l.{value}) OVER (PARTITION BY l.{key} {order}) AS before_value,",
        f"         LAG(l.modificationTime) OVER (PARTITION BY l.{key} {order}) AS since_at",
        # LAG looks back only: rows after the window are not needed; rows before it are (a wait can start long before)
        f"  FROM {log} l WHERE l.{key} IN (SELECT {key} FROM n) AND l.modificationTime <= {end}),",
        "s AS (",
        f"  SELECT {key}, next_value, moved_at, since_at FROM p",
        f"  WHERE before_value = {sql_literal(q.observed)} AND moved_at >= {start} AND moved_at <= {end}),",
        "j AS (",
        f"  SELECT {key}, " + ", ".join(f"MAX({c}) AS {c}" for c, _ in q.cols) + " FROM (",
        "\n    UNION ALL\n".join(tables),
        f"  ) GROUP BY {key})",
        f"SELECT s.next_value, {', '.join(f'j.{c}' for c, _ in q.cols)}, COUNT(*) AS n,"
        f" ROUND(MIN({took}) * 24, 1) AS min_h,",
        f"       ROUND(MEDIAN({took}) * 24, 1) AS median_h, ROUND(MAX({took}) * 24, 1) AS max_h",
        f"FROM s JOIN j ON j.{key} = s.{key}",
        f"GROUP BY s.next_value, {', '.join(f'j.{c}' for c, _ in q.cols)}",
        f"ORDER BY s.next_value, {', '.join(_sorted(c, d, 'j.') for c, d in q.cols)};",
    ])
