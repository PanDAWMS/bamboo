"""Which columns of a row may hold a value written at or after the decision.

The rows the API returns are the rows as they are now.  The querier reads a
row's values as the ones its code saw when it decided, so a column written at
the decision or later must not be handed to it as such: it goes under the rows
file's ``after`` instead (the querier reads it there only to say why it cannot
answer).

Three facts decide a column:

* when the decision was, t_d -- from the status history and the kind of
  question (:func:`decision_time`);
* which writes reached the row at or after t_d -- the history's rows from t_d
  on, and one more no history recorded when the row's ``modificationTime`` is
  later than its last history row (:func:`writes_from`);
* which columns each of those writes set.  Nothing states this yet: the map
  reads a flush's SET but does not export it, and a SET built at run time
  (``updateJobStatus`` from the pilot's ``param``) is known only from the log
  of that run.  Until then every write may have set every column
  (:data:`ANY_COLUMN`), which keeps only the key.

This module takes and returns values only, so that it can move to the querier,
which knows the decision better than a time taken from the history.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Callable

#: a write whose columns no fact states
ANY_COLUMN = None


@dataclass(frozen=True)
class Write:
    """A write that reached the row: when, and the value of the history it wrote (None when unrecorded)."""

    at: datetime.datetime
    status: str | None


def decision_time(
    kind: str, observed: str, states: list[dict]
) -> tuple[datetime.datetime | None, str]:
    """t_d, and how it was taken.  *states* are the history's rows, oldest first, with ``status`` and ``time``.

    * default -- the value became *observed*: the write of it decided.
    * late -- *observed* was moved on late: the write that moved it on.
    * stuck -- nothing moved *observed* on: every write from its arrival on may be after a decision.

    The first row with *observed* is taken, as the querier's L1 does.
    """
    at = next((n for n, s in enumerate(states) if s["status"] == observed), None)
    if at is None:
        return None, f"no history row with {observed!r}"
    if kind == "late":
        if at + 1 >= len(states):
            return None, f"no history row after {observed!r}"
        return _time(states[at + 1]), f"the row after {observed!r} (late)"
    return _time(states[at]), f"the first row with {observed!r} ({kind})"


def writes_from(
    t_d: datetime.datetime, states: list[dict], row_modified: datetime.datetime | None
) -> list[Write]:
    """The writes at or after *t_d*: the history's, and one unrecorded write when the row changed after its last row."""
    out = [Write(_time(s), s["status"]) for s in states if _time(s) >= t_d]
    last = max((_time(s) for s in states), default=None)
    if row_modified is not None and (last is None or row_modified > last):
        out.append(Write(row_modified, None))
    return out


def split(
    row: dict,
    key: str,
    kind: str,
    observed: str,
    states: list[dict],
    row_modified: datetime.datetime | None,
    columns_of: Callable[[Write], set[str] | None] = lambda write: ANY_COLUMN,
) -> tuple[dict, dict, str]:
    """Split *row* into the columns kept for the decision and those under ``after``, and say why.

    *columns_of* gives the columns a write set, or :data:`ANY_COLUMN` when no
    fact states them.  The key column is never after: it is what the row was
    asked by.  When t_d cannot be taken, every column but the key is after.
    """
    t_d, how = decision_time(kind, observed, states)
    if t_d is None:
        written = set(row) - {key}
        why = f"no decision time ({how}): every column but {key} is after"
    else:
        writes = writes_from(t_d, states, row_modified)
        written: set[str] = set()
        for w in writes:
            cols = columns_of(w)
            written |= (
                (set(row) - {key})
                if cols is ANY_COLUMN
                else (set(cols) & set(row)) - {key}
            )
        unknown = sum(1 for w in writes if columns_of(w) is ANY_COLUMN)
        why = (
            f"decision at {t_d:%Y-%m-%d %H:%M:%S.%f}, {how}; {len(writes)} write(s) from then on"
            + (f", {unknown} of them with no stated columns" if unknown else "")
        )
    kept = {c: v for c, v in row.items() if c not in written}
    after = {c: v for c, v in row.items() if c in written}
    return kept, after, why


def _time(state: dict) -> datetime.datetime:
    return datetime.datetime.fromisoformat(state.get("time_extended") or state["time"])
