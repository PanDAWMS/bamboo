"""Which of the rows file's values a step may read as the row's at the time of its run (KNOWN_ISSUES.md K22).

A step reads the row as it was when its run read it -- setupper's guard at the job's activation (H7), the adder's
inputs as it ran (R2 to R5) -- and the rows file holds the row as it was when it was read, ``as_of``, often days
later.  A value read then is the run's only where no write between the two can have changed the column: which
writes those are (the run each statuslog row was written by, and the writes that leave no row), and which columns
each can change (the attributes a run can set on the object, the SET columns a statement fills with another value
than the one at hand), are the map's to say, and it says neither yet.  So such a value is not the run's, and the
step is told why: when it was read and how many statuslog rows lie between -- rows, not writes: a write that moves
no status leaves none (updateJobMetrics_JEDI's jobMetrics).

* A value with no ``as_of`` is the hand's, read for the case (tmaeno's SQL), and is read as before.
* The value the row was asked by (``--key``) names the row and is read whatever the time.
* Every other value the builder read is the row as it was at ``as_of``, as the columns of the rows file's
  ``after`` are (the builder's split puts there what may have been written at the decision or later): ``split``
  reads the two alike, so the steps that read ``after`` at the window's end (stuck, K17) read these there too,
  and a rows file with every column in ``rows`` reads as one with them in ``after`` (rows-at-time-design.md §3).
* A file's rows (``files``) are read as before: their writes are the job's runs too, and wait for the same facts
  (design D1, tmaeno 2026-10-18).
"""
from __future__ import annotations

from bamboo.querier import windows as W

#: the fields of a rows file's section that are not columns
SECTION = ("source", "as_of", "clock")


def split(data: dict, subject: str | None, key: str | None) -> tuple[dict[str, dict], dict]:
    """The rows file's ``rows`` and ``after`` as the steps read them.

    A value of ``rows`` the builder read (it has an ``as_of``), other than *key*, is left out of ``rows``; one of
    the row of *subject*'s class goes to ``after``, as its value, with the section's ``source``, ``as_of`` and
    ``clock`` taken from it where ``after`` names none.
    """
    entity = next((e for e, c in data.get("classes", {}).items() if c == subject), None)
    rows: dict[str, dict] = {}
    after = dict(data.get("after") or {})
    for name, row in data.get("rows", {}).items():
        rows[name] = {}
        for column, cell in row.items():
            if "as_of" not in cell or (key is not None and str(cell["value"]) == key):
                rows[name][column] = cell
            elif name == entity:
                after.setdefault(column, cell["value"])
                for field in SECTION:
                    if field in cell:
                        after.setdefault(field, cell[field])
    return rows, after


def at_run(data: dict, subject: str | None, key: str | None, at: str | None) -> tuple[dict[str, dict], dict[str, str]]:
    """The rows file's entity rows as a step whose run began *at* may read them: ``(rows, unread)``.

    *rows* is ``split``'s; *unread* is ``{"<entity>.column": why}`` for every value it left out of them and every
    column of ``after``.
    """
    states = (data.get("statuslog") or {}).get("states") or []
    rows, after = split(data, subject, key)
    unread: dict[str, str] = {}
    for name, row in data.get("rows", {}).items():
        for column, cell in row.items():
            if column not in rows[name]:
                unread[f"<{name}>.{column}"] = _why(cell["as_of"], cell.get("clock"), states, at)
    entity = next((e for e, c in data.get("classes", {}).items() if c == subject), None)
    if entity is not None:
        why = _why(after.get("as_of"), after.get("clock"), states, at)
        for column in after:
            if column not in SECTION and column not in rows.get(entity, {}):
                unread.setdefault(f"<{entity}>.{column}", why)
    return rows, unread


def rows_after(states: list[dict], at: str, as_of: str | None) -> list[dict]:
    """The statuslog rows written after a run that began *at*, up to *as_of* where it is given: the rows whose writers
    F1 asks the logs for, and the count a value read at *as_of* is said to lie behind."""
    since = W.when(at)
    return [s for s in states if W.when(s.get("time_extended") or s["time"]) > since
            and (as_of is None or W.when(s["time"]) <= W.when(as_of))]


def _why(as_of: str | None, clock: str | None, states: list[dict], at: str | None) -> str:
    read = f"read at {as_of} ({clock} clock)" if as_of else "in the rows file's after, written at the decision or later"
    if at is None:
        return f"{read}, not at the step's run; which writes between can change it is not known yet (KNOWN_ISSUES.md K22)"
    n = len(rows_after(states, at, as_of))
    return (f"{read}; {n} statuslog row(s) after the run's {at}, and writes that leave no such row may lie between "
            f"too; which writes between can change it is not known yet (KNOWN_ISSUES.md K22)")
