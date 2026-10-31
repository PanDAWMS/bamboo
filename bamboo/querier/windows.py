"""When a step needs the key's runs, and on which machine: the window it asks the logs over.

A door that searches the servers' logs (``recorded.RecordedLogs``) reads only the files a window reaches, so every
ask there has one; the door over files handed in reads each file whole and needs none.  Each builder returns an
:class:`Asking`: the window and machine, or None and why there is none -- a step stops on that only where its door
needs a window (``Logs.needs_window``), so the cases run on files go on as before.

* H2 (``at_transition``): the point the statuslog row was written, on the machine it names.  The row's
  ``modificationHost`` is the machine whose process wrote the row (``socket.getfqdn()``; the only INSERTs into the
  status histories are base_module.py:604 and :849), so the run that wrote it is on that machine and no other.
* H2 (``value_held``), for a subject the history's rows hold beside the status (``parts.history_holding``): from the
  row before the last stretch of rows holding the value (or the row's creation) to the stretch's first row, on every
  machine.  A row holding the value says it was there by then, not which run wrote it: the history row is written by
  whichever run moves the status, and the run that wrote the value need not be that one (h2-value-time-design.md).
* H2 (``lifetime``, F3), for an attribute's writer on no history row of the object that marks its runs by the
  object's key: from the row's creation to when the value was read, on every machine (KNOWN_ISSUES.md K37).
* H6 (``rows_until``): the job's own statuslog rows up to the run the object was missing in, each at its own time on
  the machine that wrote it, as H2's.  A row is the job's, so a run holding its second on its host can be one that
  handled the job; H6 takes a maker only from a run that also names the job.  Not the Datasets row: Datasets holds
  one row per vuid (md5 of ``scope:name``), inserted by the first run to use it (insertDataset), so its
  ``creationdate`` is that run's, and a name many runs share was not made by this job's (panda-server,
  live-harness/notes/from-panda-server-13.md 1-2).
* S1, S2 (``stuck``): from the observed value's statuslog row less SLACK, but no earlier than ``as_of`` less
  :data:`STUCK_SPAN`, to ``as_of``, when the statuslog was read.  A mover retries every few hours (tmaeno,
  2026-10-13), so the last six hours are asked first; but no pass in six hours is not no pass in the window, so S1
  asks the rest of it (``rest``) for a mover with no run of its own there (build order 2a; tmaeno, 2026-10-16).

SLACK is the margin the server's statuslog rows were measured to need against their runs' log times (a second;
e1, to-querier-3), widened; JEDI's clocks were measured on five machines of nine (to-querier-13).
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass

from bamboo.querier.records import Window

SLACK = datetime.timedelta(minutes=10)
#: the part of a stuck window asked first: an account of how often a mover retries, not a measurement (KNOWN_ISSUES.md K5)
STUCK_SPAN = datetime.timedelta(hours=6)


@dataclass
class Asking:
    """A window and machine to ask the logs over, and how they were made; or no window, and why."""

    window: Window | None
    machine: str | None
    why: str
    #: the strings to ask for, where not the key alone: F3's writers' marks holding the key (``rerun.key_marks``)
    strings: tuple[str, ...] = ()


def when(text: str) -> datetime.datetime:
    """``2026-01-01 00:00:00``, with ``.ffffff`` or a record's ``,fff`` when it has them."""
    return datetime.datetime.fromisoformat(text.replace(",", "."))


def _down(t: datetime.datetime) -> str:
    return f"{t:%Y-%m-%d %H:%M:%S}"


def at_transition(state: dict) -> Asking:
    """H2: the statuslog row's own time, on the machine that wrote it (none named: every machine)."""
    t = state["time"]
    host = state.get("host") or None
    where = f"on {host}, the row's host" if host else "on every machine (the row names no host)"
    return Asking((t, t), host, f"the statuslog's {state['status']!r} at {t}, {where}")


def value_held(states: list[dict], column: str, value: str, created: str | None) -> Asking:
    """H2 for a value the statuslog's rows hold in *column* beside the status: the write is after the last row holding
    another (a second after it: the row's time is the second it was written in) and by the first row of the last
    stretch holding *value*; a stretch from the first row, from the row's creation *created* (None where not known).
    No machine.  No window where no row carries the column, where the last row holds another value (written after it:
    no row closes the window), or where the stretch starts at the first row and *created* is not known."""
    if not any(column in s for s in states):
        return Asking(None, None, f"the statuslog rows carry no {column}")
    if states[-1].get(column) != value:
        last = states[-1]
        return Asking(None, None, f"the last statuslog row ({last['status']!r} at {last['time']}) holds {column}="
                                  f"{last.get(column)!r}: written after it, and no row closes the window")
    k = len(states) - 1
    while k > 0 and states[k - 1].get(column) == value:
        k -= 1
    first = states[k]
    if k:
        before = states[k - 1]
        lo = _down(when(before["time"]) + datetime.timedelta(seconds=1))
        since = f"after the statuslog's {before['status']!r} at {before['time']} ({column}={before.get(column)!r})"
    elif created:
        lo = min(_down(when(created)), first["time"])
        since = f"from the row's creation at {created}"
    else:
        return Asking(None, None, f"the first statuslog row ({first['status']!r} at {first['time']}) holds {column}="
                                  f"{value!r} and the row's creation time is not known")
    return Asking((lo, first["time"]), None, f"{column}={value!r} {since} to the statuslog's {first['status']!r} at "
                                             f"{first['time']}, on every machine (no row says which run wrote it)")


def value_now(column: str, value: str) -> Asking:
    """H2 for a value every writer of *column* sets to the time it runs (``attr_time.NOW``): the value is the write's
    time, on the database's clock or the writer's, so the run is asked at it on every machine.  No window where the
    value is not a time."""
    try:
        t = _down(when(value))
    except ValueError:
        return Asking(None, None, f"{column}={value!r}, which its writers set to the time they run, is not a time")
    return Asking((t, t), None, f"{column}={value!r}, the time its writer ran: every writer of it writes the time it "
                                f"runs (the last write), on every machine")


def lifetime(made: str | None, read: str | None, strings: tuple[str, ...]) -> Asking:
    """H2's F3, for an attribute's writer on no history row of the object that marks its runs by the object's key
    (``attr_time``, KNOWN_ISSUES.md K37): from the row's creation *made* to when the value was read *read*, on every
    machine -- the write that left the value is in it, whenever it ran -- for *strings*, the writers' marks holding the
    key, not the key alone: a task's ID is on every line of the calls listing it.  No window where either end is not
    known."""
    if not made:
        return Asking(None, None, "the row's creation time is not known")
    if not read:
        return Asking(None, None, "when the value was read is not known")
    lo, hi = _down(when(made)), _down(when(read))
    return Asking((lo, hi), None, f"the object's life, from the row's creation at {lo} to when the value was read at {hi}, "
                                  f"on every machine, for {', '.join(repr(s) for s in strings)}", strings)


def rows_until(states: list[dict], until: str) -> list[tuple[dict, Asking]]:
    """H6: the statuslog rows written by *until* (the last record of the run the object was missing in), in order, each
    with its own time on the machine that wrote it (:func:`at_transition`)."""
    return [(s, at_transition(s)) for s in states if s["time"][:19] <= until[:19]]


def stuck(statuslog: dict, observed: str) -> Asking:
    """S1, S2: from the observed value's row less SLACK, no earlier than as_of less STUCK_SPAN, to as_of."""
    as_of = statuslog.get("as_of")
    if not as_of:
        return Asking(None, None, "the statuslog says no as_of (when it was read)")
    rows = [s for s in statuslog.get("states", []) if s["status"] == observed]
    if not rows:
        return Asking(None, None, f"the statuslog has no row for {observed!r}")
    since, upto = when(rows[-1]["time"]), when(as_of)
    start = max(since - SLACK, upto - STUCK_SPAN)
    bound = "its row less" if start == since - SLACK else "as_of less"
    return Asking(
        (_down(start), _down(upto)),
        None,
        f"{observed!r} since {rows[-1]['time']}; from {bound} {_minutes(SLACK if bound == 'its row less' else STUCK_SPAN)} "
        f"to the statuslog's as_of {as_of}",
    )


def rest(statuslog: dict, observed: str) -> Asking | None:
    """S1: the stuck window before :func:`stuck`'s, from the observed value's row less SLACK to ``as_of`` less
    STUCK_SPAN; None where :func:`stuck`'s window already starts at the row, or there is none."""
    first = stuck(statuslog, observed)
    if first.window is None:
        return None
    since = [s for s in statuslog["states"] if s["status"] == observed][-1]["time"]
    start = when(since) - SLACK
    if start >= when(statuslog["as_of"]) - STUCK_SPAN:
        return None
    return Asking(
        (_down(start), first.window[0]),
        None,
        f"{observed!r} since {since}; the rest of the window, from its row less {_minutes(SLACK)} to the last "
        f"{_minutes(STUCK_SPAN)}, for a mover with no run of its own in those",
    )


def _minutes(span: datetime.timedelta) -> str:
    m = int(span.total_seconds() // 60)
    return f"{m // 60} h" if m % 60 == 0 else f"{m} min"
