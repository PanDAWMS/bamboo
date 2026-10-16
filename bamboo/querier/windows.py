"""When a step needs the key's runs, and on which machine: the window it asks the logs over.

A door that searches the servers' logs (``recorded.RecordedLogs``) reads only the files a window reaches, so every
ask there has one; the door over files handed in reads each file whole and needs none.  Each builder returns an
:class:`Asking`: the window and machine, or None and why there is none -- a step stops on that only where its door
needs a window (``Logs.needs_window``), so the cases run on files go on as before.

* H2 (``at_transition``): the point the statuslog row was written, on the machine it names.  The row's
  ``modificationHost`` is the machine whose process wrote the row (``socket.getfqdn()``; the only INSERTs into the
  status histories are base_module.py:604 and :849), so the run that wrote it is on that machine and no other.
* H6 (``around_creation``): the Datasets row's ``creationdate`` +- :data:`SLACK`.  The row is inserted by the run
  that registers the dataset (setupper_atlas_plugin's insertDatasets calls, in the function of the make call), on
  that machine's clock.  A creationdate before the job was created (its first statuslog row) less SLACK is not this
  job's dataset's: no window.
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


def when(text: str) -> datetime.datetime:
    """``2026-01-01 00:00:00``, with ``.ffffff`` or a record's ``,fff`` when it has them."""
    return datetime.datetime.fromisoformat(text.replace(",", "."))


def _down(t: datetime.datetime) -> str:
    return f"{t:%Y-%m-%d %H:%M:%S}"


def _up(t: datetime.datetime) -> str:
    return _down(t.replace(microsecond=0) + datetime.timedelta(seconds=1) if t.microsecond else t)


def at_transition(state: dict) -> Asking:
    """H2: the statuslog row's own time, on the machine that wrote it (none named: every machine)."""
    t = state["time"]
    host = state.get("host") or None
    where = f"on {host}, the row's host" if host else "on every machine (the row names no host)"
    return Asking((t, t), host, f"the statuslog's {state['status']!r} at {t}, {where}")


def around_creation(datasets: dict | None, name: str, created: str) -> Asking:
    """H6: the Datasets row of *name*: its creationdate +- SLACK; or why there is no window.

    *created* is when the job was created (its first statuslog row, "" when there is none).  The section lists every name asked
    (``asked``): a name not asked is the builder's gap, a name asked with no row is not in the table now.
    """
    if datasets is None:
        return Asking(None, None, "the rows file has no datasets section")
    if name not in datasets.get("asked", []):
        return Asking(None, None, f"the datasets section did not ask for {name}")
    rows = [r for r in datasets.get("rows", []) if r.get("name") == name]
    if not rows:
        return Asking(None, None, f"no {datasets.get('table', 'Datasets')} row for {name} (asked; it may have been purged)")
    stamps = sorted({r.get("creationdate") for r in rows}, key=str)
    if len(stamps) > 1 or not stamps[0]:
        return Asking(None, None, f"{len(rows)} {datasets.get('table', 'Datasets')} rows for {name}, creationdate {stamps}")
    if not created:
        return Asking(None, None, "the statuslog has no row to say when the job was created")
    made = when(stamps[0])
    anchor = f"{datasets.get('class', '?')} {datasets.get('table', '?')} creationdate {stamps[0]} (the inserting machine's clock)"
    if made < when(created) - SLACK:
        return Asking(None, None, f"{anchor} is before the job was created ({created}) less {_minutes(SLACK)}")
    return Asking((_down(made - SLACK), _up(made + SLACK)), None, f"{anchor} +- {_minutes(SLACK)}")


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
