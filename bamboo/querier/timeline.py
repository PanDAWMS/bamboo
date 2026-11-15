"""What a time column of the row held over a stay, read from the records (Q1 part 1: analyzer handover
out/gm/qk-design.md, the Q1 design summary, tmaeno's OK 2026-11-15 23:06), where M3(a)-1 (``stay_value``) leaves it open:
an arm that may write it in the stay (its ``left``), or one that may begin the stay without writing it (``lacking``).

A write is read where it shows, not where a run of the writer's own mark does: the map's must prints of the arm
(``Branch.must_prints``) whose hole it names as the row's key (``Branch.must_names``), found among the key's records
under any mark -- ``_read_task_with_unprocessed_inputs``' lock UPDATE prints its binds under its caller's
``getTasksToBeProcessed_JEDI < jediTasks=[...] >``.  A print of the statement's own bind map (part ``bind``) says the
statement ran there (``ran``); one of the message or the logger's mark says only that the pass reached the print, and
the arm may have run (``reached``).

The column's value is kept as bounds, not one value: the least and the most it may hold.  A write the arm makes with the
clock (``attr_time.NOW``) holds the print's time; any other value is not read here (the map gives no base and offset
for a write's outcome) and leaves the bounds open.  A ``reached`` write widens the bounds by its value; a ``ran`` one
sets them.  An arm neither read nor shown not to have written (judgement 3's premises) may write at any moment of the
stay: its writes widen the bounds from then on, unless the reader names it as having not (``assume``).

A term ``col < clock + offset`` is then surely true for the row at a moment where the most the column may hold is
below it, and surely false where the least is not.  NT-tobroken: modificationTime held at most 20:29:16.42 (the stay's
start) until the mover's own UPDATE at 20:34:44.34, so ``modificationtime < now - 5 min`` surely held from 20:34:16.42.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass

#: the bounds of a value nothing bounds
LOW = datetime.datetime.min
HIGH = datetime.datetime.max

#: how a print shows a write: the statement ran with the printed binds, or the pass reached the print
RAN = "ran"
REACHED = "reached"


@dataclass(frozen=True)
class Write:
    """A write of the column the records show, at the print's time."""

    at: datetime.datetime
    #: the least and the most the column holds after it (``LOW``/``HIGH`` where its value is not read)
    lo: datetime.datetime
    hi: datetime.datetime
    #: ``RAN`` or ``REACHED``
    how: str
    #: the arms the print reads for, "func:line" each
    arms: tuple[str, ...]
    #: the print, "func:line"
    where: str
    machine: str


@dataclass(frozen=True)
class Opening:
    """An arm that may write the column at any moment of the stay: neither read nor shown not to have written."""

    arm: str
    #: the least and the most it writes, as offsets from the moment it runs (``None``: not read, any value)
    lo: datetime.timedelta | None
    hi: datetime.timedelta | None


@dataclass(frozen=True)
class Timeline:
    """The column over [start, end]: its bounds as the stay began, the writes read in the stay, the arms open."""

    start: datetime.datetime
    end: datetime.datetime
    #: the bounds as the stay began; and the same where the open arms are named as having not written it
    first: tuple[datetime.datetime, datetime.datetime]
    first_assumed: tuple[datetime.datetime, datetime.datetime]
    writes: tuple[Write, ...] = ()
    opens: tuple[Opening, ...] = ()
    #: the arms shown not to have written it (judgement 3's premises held), "func:line" each
    absent: tuple[str, ...] = ()

    def pieces(self, assume: bool = False) -> list[tuple[datetime.datetime, datetime.datetime, datetime.datetime,
                                                          datetime.datetime, tuple[Opening, ...]]]:
        """``(from, to, lo, hi, opens)`` over [start, end], cut at each write read: the bounds the writes read leave,
        and the arms that may write it in the piece (none where *assume* names them as having not)."""
        out = []
        lo, hi = self.first_assumed if assume else self.first
        at = self.start
        opens = () if assume else self.opens
        for w in sorted(self.writes, key=lambda w: w.at):
            if not self.start < w.at <= self.end:
                continue
            out.append((at, w.at, lo, hi, opens))
            lo, hi = (w.lo, w.hi) if w.how == RAN else (min(lo, w.lo), max(hi, w.hi))
            at = w.at
        out.append((at, self.end, lo, hi, opens))
        return out


def fold(writes, lo: datetime.datetime, hi: datetime.datetime) -> tuple[datetime.datetime, datetime.datetime]:
    """The bounds *writes* leave, in time order, from *lo* and *hi*: a ``RAN`` write sets them, a ``REACHED`` one widens
    them by its value."""
    for w in sorted(writes, key=lambda w: w.at):
        lo, hi = (w.lo, w.hi) if w.how == RAN else (min(lo, w.lo), max(hi, w.hi))
    return lo, hi


def first_bounds(begun: datetime.datetime, since: datetime.datetime, starts: bool, lacking: bool, writes,
                 opens) -> tuple[datetime.datetime, datetime.datetime]:
    """The column's bounds as the stay began at *begun*: *begun* itself where an arm writing the value and the column
    may have begun it (*starts*); where one writing the value alone may have (*lacking*), or none writing both is
    read, what the writes read from *since* to *begun* leave, from no bound, widened by *opens* -- the last write
    before the stay may be older than *since*, and is then not read."""
    ways = []
    if starts:
        ways.append((begun, begun))
    if lacking or not starts:
        lo, hi = fold([w for w in writes if since <= w.at <= begun], LOW, HIGH)
        for o in opens:
            lo = LOW if o.lo is None else min(lo, _plus(since, o.lo))
            hi = HIGH if o.hi is None else max(hi, _plus(begun, o.hi))
        ways.append((lo, hi))
    return min(w[0] for w in ways), max(w[1] for w in ways)


def _plus(t: datetime.datetime, d: datetime.timedelta) -> datetime.datetime:
    try:
        return t + d
    except OverflowError:
        return HIGH if d > datetime.timedelta(0) else LOW


def bounds_at(piece, t: datetime.datetime) -> tuple[datetime.datetime, datetime.datetime]:
    """The column's bounds at *t* inside *piece*: the piece's, widened by what an open arm may have written from the
    piece's start to *t* -- its value at a moment *u* lies in [u + lo, u + hi]."""
    a, _, lo, hi, opens = piece
    for o in opens:
        lo = LOW if o.lo is None else min(lo, _plus(a, o.lo))
        hi = HIGH if o.hi is None else max(hi, _plus(t, o.hi))
    return lo, hi


def surely_true(op: str, offset: datetime.timedelta, pieces) -> list[tuple[datetime.datetime, datetime.datetime]]:
    """Where ``col <op> clock + offset`` surely holds: below it, the most the column may hold; above it, the least.
    An open arm whose most is the moment it runs (the clock) never lets ``<`` with a negative offset surely hold."""
    out = []
    for p in pieces:
        a, b = p[0], p[1]
        if op in ("<", "<="):
            # col < t + offset for every value the column may hold at t: an open arm's write at u <= t holds at most
            # u + o.hi <= t + o.hi, below t + offset only where o.hi < offset; the writes read hold at most hi
            if any(o.hi is None or o.hi >= offset for o in p[4]):
                continue
            turn = _plus(p[3], -offset)
            if turn < b:
                out.append((max(a, turn), b))
        elif op in (">", ">="):
            lo, _ = bounds_at(p, a)
            turn = _plus(lo, -offset)
            if turn > a:
                out.append((a, min(b, turn)))
    # the pieces' spans joined where they meet
    joined: list[tuple[datetime.datetime, datetime.datetime]] = []
    for a, b in out:
        if joined and a <= joined[-1][1]:
            joined[-1] = (joined[-1][0], max(joined[-1][1], b))
        else:
            joined.append((a, b))
    return joined


def surely_false_all(op: str, offset: datetime.timedelta, pieces) -> bool:
    """Whether ``col <op> clock + offset`` surely fails at every moment of the pieces."""
    return bool(pieces) and surely_false_between(op, offset, pieces, pieces[0][0], pieces[-1][1])


def surely_false_between(op: str, offset: datetime.timedelta, pieces, a: datetime.datetime, b: datetime.datetime) -> bool:
    """Whether ``col <op> clock + offset`` surely fails at every moment of [a, b]: a selection that ran then did not
    return the row on it (P5).  An open arm writes from its piece's start, not from *a*."""
    met = False
    for p in pieces:
        lo_t, hi_t = max(p[0], a), min(p[1], b)
        if lo_t > hi_t:
            continue
        met = True
        if op in ("<", "<="):
            # col >= t + offset for every value at t: lo(t) >= t + offset, worst at the latest t
            lo, _ = bounds_at(p, hi_t)
            if lo == LOW or _plus(lo, -offset) < hi_t:
                return False
        elif op in (">", ">="):
            # col <= t + offset for every value at t: an open arm's at most t + o.hi, the writes read's at most hi
            if any(o.hi is None or o.hi > offset for o in p[4]) or p[3] == HIGH or _plus(p[3], -offset) > lo_t:
                return False
        else:
            return False
    return met
