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

* H, status history: by key (the rows file's ``statuslog`` for now), by
  transition and window (:class:`MovedOn`), and held at a moment
  (:class:`HeldAt`).
* A1, a row by key wherever it lives now (live, archived, the archive
  schema), for the set of keys inside :class:`MovedOn` and :class:`HeldAt`.

Each question is asked with its structure beside its SQL (:func:`asked`), so a
fetch can answer it by these parts rather than by the SQL.
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass

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
class HeldAt:
    """H and A1: the rows whose subject held a value at a moment -- their last history row before it is the value --
    under a call's WHERE, counted by an order's items before the key, with how many in each group rank ahead of the
    row by the key, the row itself, and the earliest entry counted.  The WHERE's terms on columns the history holds
    are read on that last row; the others on the row as it is now, wherever it lives (A1).  An entry before
    *lookback* is not asked: a row that held the value from earlier is not counted."""

    history: History
    observed: str
    at: datetime.datetime  # the moment: the rows that held the value just before it
    lookback: datetime.datetime
    held: list[str]  # the call's WHERE terms on columns the history holds, read at the moment
    now: list[str]  # its other terms, read on the row as it is now
    cols: list[tuple[str, str]]  # (column, asc|desc|ranked): the order's items before the key
    key: str
    mine: object  # the row's key
    #: how the key ranks within a group (asc|desc); None for an order without the key, whose ties the database orders
    key_order: str | None


@dataclass
class MovedOn:
    """H and A1: the rows whose subject left the observed value in a window, under a call's WHERE, by the value
    each went to, with how long each had held it, counted by columns of the row as it is now, and how many of them had
    a key ranking ahead of the row's.  As :class:`HeldAt` reads them: the terms on columns the history holds on the
    history row each left, the others on the row as it is now (A1).  The keys asked are those with a history row in
    the window under the held terms: one whose rows there all differ from the row it left (a site changed at the move)
    is not counted.

    Every next value, not only the one this row went to: a group starved of one way out leaves by another (a
    7-day kill), and only the two side by side say so.  Counts are moves, not keys."""

    history: History
    observed: str
    since: datetime.datetime
    moved: datetime.datetime
    held: list[str]  # the call's WHERE terms on columns the history holds, read on the row each left
    now: list[str]  # its other terms, read on the row as it is now
    cols: list[tuple[str, str]]  # (column, asc|desc|ranked)
    key: str
    mine: object = None
    key_order: str | None = None


def sql_literal(value) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


def _oracle_time(t: datetime.datetime) -> str:
    return f"TO_DATE('{t:%Y-%m-%d %H:%M:%S}', {ORACLE_TIME})"


def draw(q: HeldAt | MovedOn) -> str:
    """The question as the SQL that answers it."""
    return _held_at_sql(q) if isinstance(q, HeldAt) else _moved_on_sql(q)


def columns(q: HeldAt | MovedOn) -> list[str]:
    """The columns an answer to *q* holds, as its SQL names them: an answer does not carry the SQL it answers."""
    ahead = ["n_key_ahead"] if q.key_order else []
    if isinstance(q, HeldAt):
        return [c for c, _ in q.cols] + ["n", *ahead, "n_self", "first_entered"]
    return ["next_value", *(c for c, _ in q.cols), "n", *ahead, "min_h", "median_h", "max_h"]


def asked(q: HeldAt | MovedOn) -> dict:
    """*q* as a structure, beside its SQL: what parts H and A1 take, for a fetch that answers it by them (the history
    by key, the rows by key) rather than by the SQL."""
    out = {"subject": q.history.subject, "observed": q.observed, "group_by": [[c, d] for c, d in q.cols], "key": q.key,
           "key_value": q.mine, "key_order": q.key_order, "columns": columns(q)}
    if isinstance(q, HeldAt):
        return {"part": "H held at", "at": f"{q.at:%Y-%m-%d %H:%M:%S}", "lookback": f"{q.lookback:%Y-%m-%d %H:%M:%S}",
                "terms_at": q.held, "terms_now": q.now} | out
    return {"part": "H moved on", "since": f"{q.since:%Y-%m-%d %H:%M:%S}", "moved": f"{q.moved:%Y-%m-%d %H:%M:%S}",
            "terms_at": q.held, "terms_now": q.now} | out


def _sorted(c: str, d: str, alias: str = "") -> str:
    """An ORDER BY item: a ranked column (ranked at run time, not here) by itself."""
    return f"{alias}{c}" if d == "ranked" else f"{alias}{c} {d}"


def _key_counts(q: HeldAt | MovedOn, alias: str) -> list[str]:
    """How many of a group rank ahead of the row by the key, and, for :class:`HeldAt`, the row itself."""
    out = []
    if q.key_order:
        op = "<" if q.key_order == "asc" else ">"
        out.append(f"COUNT(CASE WHEN {alias}{q.key} {op} {sql_literal(q.mine)} THEN 1 END) AS n_key_ahead")
    if isinstance(q, HeldAt):
        out.append(f"COUNT(CASE WHEN {alias}{q.key} = {sql_literal(q.mine)} THEN 1 END) AS n_self")
    return out


def _rows_now(q: HeldAt | MovedOn, keys: str, start: str, terms: list[str]) -> list[str]:
    """A1: the keys of *keys* as their rows are now, in whichever table holds them, with the order's columns, under
    *terms* read on that row."""
    cls = q.history.subject.partition(".")[0]
    names = "".join(f", {c}" for c, _ in q.cols)
    where = "".join(f" AND {t}" for t in terms)
    tables = [
        f"    SELECT {q.key}{names} FROM {t} WHERE {q.key} IN (SELECT {q.key} FROM {keys}){where}"
        + (f" AND {bound.format(start=start)}" if bound else "")
        for t, bound in _ROW_TABLES[cls]
    ]
    return [
        "j AS (",
        f"  SELECT {q.key}" + "".join(f", MAX({c}) AS {c}" for c, _ in q.cols) + " FROM (",
        "\n    UNION ALL\n".join(tables),
        f"  ) GROUP BY {q.key})",
    ]


def _held_at_sql(q: HeldAt) -> str:
    _, log, ordered_by = _HISTORIES[q.history.subject]
    _, _, value = q.history.subject.partition(".")
    key = q.key
    at, start = _oracle_time(q.at), _oracle_time(q.lookback)
    held = "".join(f" AND {t}" for t in q.held)
    latest = ", ".join(f"l.{c.strip()} DESC" for c in ordered_by.split(","))
    ranked = "".join(f"; {c} ranked at run time" for c, d in q.cols if d == "ranked")
    lines = [
        f"-- the rows that held {q.observed!r} just before {q.at:%Y-%m-%d %H:%M:%S} (their last status-log row before it),"
        f" under the call's terms the status log holds read there and the others on the row as it is now; by the order's"
        f" items before the key{ranked}, with how many of each group rank ahead of the row by the key"
        + (f" ({key} {q.key_order})" if q.key_order else " (not in the order: not counted)")
        + f", the row itself, and the earliest entry counted (entries from {q.lookback:%Y-%m-%d %H:%M:%S} on)",
        "WITH e AS (",
        f"  SELECT DISTINCT {key} FROM {log}",
        f"  WHERE {value} = {sql_literal(q.observed)} AND modificationTime >= {start} AND modificationTime < {at}{held}),",
        "r AS (",
        f"  SELECT l.*, ROW_NUMBER() OVER (PARTITION BY l.{key} ORDER BY {latest}) AS rn",
        f"  FROM {log} l WHERE l.{key} IN (SELECT {key} FROM e) AND l.modificationTime < {at}),",
        "h AS (",
        f"  SELECT {key}, modificationTime AS entered FROM r",
        f"  WHERE rn = 1 AND {value} = {sql_literal(q.observed)}{held}),",
        *_rows_now(q, "h", start, q.now),
        f"SELECT {''.join(f'j.{c}, ' for c, _ in q.cols)}COUNT(*) AS n, "
        + "".join(f"{c}, " for c in _key_counts(q, "h."))
        + "MIN(h.entered) AS first_entered",
        f"FROM h JOIN j ON j.{key} = h.{key}",
    ]
    if q.cols:
        lines += [f"GROUP BY {', '.join(f'j.{c}' for c, _ in q.cols)}", f"ORDER BY {', '.join(_sorted(c, d, 'j.') for c, d in q.cols)};"]
    else:
        lines[-1] += ";"
    return "\n".join(lines)


def _moved_on_sql(q: MovedOn) -> str:
    _, log, ordered_by = _HISTORIES[q.history.subject]
    _, _, value = q.history.subject.partition(".")
    key = q.key
    start, end = _oracle_time(q.since), _oracle_time(q.moved)
    took = "s.moved_at - s.since_at"
    order = "ORDER BY " + ", ".join(f"l.{c.strip()}" for c in ordered_by.split(","))
    held = "".join(f" AND {t}" for t in q.held)
    joined = bool(q.cols or q.now)
    lines = [
        f"-- the rows that left {q.observed!r} in the row's window under the call's terms the status log holds, read on the"
        f" row each left, and the others on the row as it is now; by the value each went to; hours each had held it"
        + (", by the order's columns as they are now" if q.cols else "") + " (moves, not keys)"
        + (f", and how many had a key ranking ahead of the row's ({key} {q.key_order})" if q.key_order else ""),
        "WITH n AS (",
        f"  SELECT DISTINCT {key} FROM {log}",
        f"  WHERE modificationTime >= {start} AND modificationTime <= {end}{held}),",
        "p AS (",
        f"  SELECT l.{key}, l.{value} AS next_value, l.modificationTime AS moved_at,",
        f"         LAG(l.{value}) OVER (PARTITION BY l.{key} {order}) AS before_value,",
        # the row it left holds the columns the held terms read
        *(f"         LAG(l.{c}) OVER (PARTITION BY l.{key} {order}) AS {c}," for c in sorted(q.history.held)),
        f"         LAG(l.modificationTime) OVER (PARTITION BY l.{key} {order}) AS since_at",
        # LAG looks back only: rows after the window are not needed; rows before it are (a wait can start long before)
        f"  FROM {log} l WHERE l.{key} IN (SELECT {key} FROM n) AND l.modificationTime <= {end}),",
        "s AS (",
        f"  SELECT {key}, next_value, moved_at, since_at FROM p",
        f"  WHERE before_value = {sql_literal(q.observed)} AND moved_at >= {start} AND moved_at <= {end}{held})"
        + ("," if joined else ""),
        *(_rows_now(q, "s", start, q.now) if joined else []),
        f"SELECT s.next_value, {''.join(f'j.{c}, ' for c, _ in q.cols)}COUNT(*) AS n, "
        + "".join(f"{c}, " for c in _key_counts(q, "s."))
        + f"ROUND(MIN({took}) * 24, 1) AS min_h,",
        f"       ROUND(MEDIAN({took}) * 24, 1) AS median_h, ROUND(MAX({took}) * 24, 1) AS max_h",
        f"FROM s JOIN j ON j.{key} = s.{key}" if joined else "FROM s",
        f"GROUP BY s.next_value{''.join(f', j.{c}' for c, _ in q.cols)}",
        f"ORDER BY s.next_value{''.join(', ' + _sorted(c, d, 'j.') for c, d in q.cols)};",
    ]
    return "\n".join(lines)
