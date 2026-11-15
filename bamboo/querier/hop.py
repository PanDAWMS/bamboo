"""H: a selection that waits on another object's rows (out/gm/qk-design.md "H", tmaeno's OK 2026-11-15 13:27:03).

A mover's selecting statement can hold a term on other rows than the selected one: prepareTasksToBeFinished_JEDI takes
a task only where ``NOT EXISTS`` a dataset of it with files left to process.  Such a term holds once those rows have
moved on, and what moves them on is another object's transitions: a job's end writes its input files' counts
(propagateResultToJEDI, called by archiveJob as it records the job's status).  So the time the term began to hold is the
time the last of the other objects reached such a status, read from their histories; a stay that began earlier waited
on them, and the question goes on to that object (KNOWN_ISSUES.md K77).

The parts read the map only: the term as the map gives it (H-a, a condition node's ``rows``: its op, inner tables,
``correlated`` pairs, inner tree), the writers of its inner columns (junctions), and the functions nearest them in the
reach graph that record the other class's status history with the values they record (the carriers, H-b).  The rows
and their histories are the rows file's ``linked`` and ``statuslogs`` (H-c).  Nothing here reads the statement's text.
"""

from __future__ import annotations

import datetime
from collections.abc import Iterable
from dataclasses import dataclass

from bamboo.querier import value_reach as VR

#: how many calls above a writer of the term's columns a carrier is looked for (the nearest level that has one wins)
DEPTH = VR.DEPTH


@dataclass(frozen=True)
class Term:
    """A term of a statement selecting the asked value that is a subquery the map structures (H-a)."""

    where: str  # ``function@line`` of the read site
    owner: str  # the read site's owner, ``file::function``
    polarity: str  # the map's ``rows.op``: ``EXISTS``, ``NOT EXISTS``, ``IN``, ``NOT IN``
    inner: frozenset[str]  # the inner tree's and the correlation's inner subjects, ``Class.attr``
    correlation: tuple[tuple[str, str], ...]  # (inner, outer) pairs, ``Class.attr`` each
    text: str
    tested: frozenset[str] = frozenset()  # the inner tree's subjects alone: what moving rows on changes


@dataclass(frozen=True)
class Carriers:
    """Who moves the term's rows: the writers of its inner columns and the functions nearest them that record *cls*'s
    status history, each with the values it records."""

    cls: str
    status: str  # the history's subject, ``JobSpec.jobStatus``
    writers: tuple[str, ...]
    carriers: tuple[tuple[str, frozenset[str]], ...]

    @property
    def values(self) -> frozenset[str]:
        return frozenset(v for _, vs in self.carriers for v in vs)


@dataclass(frozen=True)
class Reading:
    """The other rows over the window: when each last reached a carrier value up to the window's end (*reached*), and
    those not in one there (*open*: key, its value then or None for a row with no history row by then)."""

    reached: tuple[tuple[datetime.datetime, str, str, str], ...]  # (at, key, value, host), in time
    open: tuple[tuple[str, str | None], ...]
    made_later: int  # rows made after the window's end, not read

    @property
    def turn(self) -> tuple[datetime.datetime, str, str, str] | None:
        return self.reached[-1] if self.reached and not self.open else None


def _walk(node) -> Iterable:
    if node is None:
        return
    yield node
    for item in getattr(node, "items", None) or []:
        yield from _walk(item)


def _subjects(node) -> set[str]:
    """The subjects a tree tests: each term's column, and the columns of a value made of the row's columns
    (``nFilesFinished + nFilesFailed``, a value's ``columns``)."""
    out = set()
    for n in _walk(node):
        if getattr(n, "subject", ""):
            out.add(n.subject)
        for v in getattr(n, "values", None) or []:
            out.update(getattr(v, "columns", None) or [])
    return out


def _pairs(correlated: Iterable[str]) -> tuple[tuple[str, str], ...]:
    """The map's ``correlated`` (``inner=outer``, ``JediDatasetSpec.jediTaskID=JediTaskSpec.jediTaskID``) as pairs."""
    out = []
    for c in correlated:
        inner, _, outer = str(c).partition("=")
        if inner and outer:
            out.append((inner.strip(), outer.strip()))
    return tuple(out)


def subquery_terms(read_sites: Iterable, owner: str, subject: str, observed: str) -> list[Term]:
    """The subquery terms the map structures (a node's ``rows``) in *owner*'s statements selecting *subject* at
    *observed*, one per term text and site.  None where the map gives the term as text only (``other`` with no
    ``rows``)."""
    out: dict[tuple[str, str], Term] = {}
    selecting = f"{subject}={observed}"
    for site in read_sites:
        if VR.bare(getattr(site, "owner", "")) != VR.bare(owner) or selecting not in (getattr(site, "selects", None) or []):
            continue
        where = f"{site.name.rpartition('::')[2]}"
        for reading in getattr(site, "condition_trees", None) or []:
            for n in _walk(reading.tree):
                rows = getattr(n, "rows", None)
                if rows is None:
                    continue
                pairs = _pairs(getattr(rows, "correlated", None) or [])
                tested = frozenset(_subjects(getattr(rows, "tree", None)))
                inner = tested | {i for i, _ in pairs}
                out.setdefault((where, n.text), Term(where, VR.bare(site.owner), str(rows.op).upper(), inner, pairs, n.text, tested))
    return list(out.values())


def carriers(junctions: Iterable, reach_graph: Iterable, inner: Iterable[str], status: str, depth: int = DEPTH) -> Carriers | None:
    """H-b: the writers of the *inner* columns (the map's junctions) and the functions nearest them, up the reach graph
    to *depth* calls, that record the history of *status* (a ``history`` call of theirs), with the values each records.
    None where no writer is found, or no such function within *depth*."""
    inner = set(inner)
    writers = sorted({VR.bare(j.owner) for j in junctions if getattr(j, "subject", "") in inner})
    if not writers:
        return None
    nodes = list(reach_graph)
    calls = VR.calls_of(nodes)
    callers: dict[str, set[str]] = {}
    for a, bs in calls.items():
        for b in bs:
            callers.setdefault(b, set()).add(a)
    records: dict[str, frozenset[str]] = {}
    for node in nodes:
        vals = {v for h in getattr(node, "history", None) or [] if h.subject == status for v in h.values}
        if vals:
            records[VR.bare(node.owner)] = records.get(VR.bare(node.owner), frozenset()) | frozenset(vals)
    seen = set(writers)
    level = set(writers)
    for n in range(1, depth + 1):
        level = {c for f in level for c in callers.get(f, ())} - seen
        seen |= level
        found = sorted((f, records[f]) for f in level if f in records)
        if found:
            # the writers the carriers reach in as many calls: the ones whose writes they carry
            carried = VR.closure([f for f, _ in found], calls, n) & set(writers)
            return Carriers(status.partition(".")[0], status, tuple(sorted(carried)), tuple(found))
    return None


def states_of(entry) -> list[dict]:
    """A ``statuslogs`` id's states: the fetch's list, or a rows answer's ``{"source", "retained_from", "as_of",
    "states"}`` (each id with the bounds of the call that read it)."""
    if isinstance(entry, dict):
        return list(entry.get("states") or [])
    return list(entry or [])


def _when(row: dict) -> datetime.datetime:
    text = str(row.get("time_extended") or row["time"]).replace("T", " ")
    return datetime.datetime.fromisoformat(text[:26])


def read_rows(history: dict[str, list[dict]], made: dict[str, datetime.datetime | None], values: frozenset[str],
              until: datetime.datetime) -> Reading:
    """H-c: over the other rows (*history* by key; *made*, each row's making where the rows file holds it), the last
    time each entered one of *values* up to *until*, and those not in one there.  A row made after *until* is not
    read; a row with no history row by then is open (it had not moved on)."""
    reached = []
    open_ = []
    later = 0
    for key in sorted(set(history) | set(made)):
        born = made.get(key)
        if born is not None and born > until:
            later += 1
            continue
        rows = sorted((r for r in history.get(key, []) if _when(r) <= until), key=_when)
        if not rows:
            open_.append((key, None))
            continue
        last = rows[-1]
        if last.get("status") not in values:
            open_.append((key, last.get("status")))
            continue
        # when it entered the value it holds at the end -- the first of the rows holding it last -- not an earlier carrier
        # value: a scout job merging, then finished when its merge job ended, entered its last value as finished
        k = len(rows) - 1
        while k > 0 and rows[k - 1].get("status") == last.get("status"):
            k -= 1
        reached.append((_when(rows[k]), key, str(rows[k].get("status")), str(rows[k].get("host") or "")))
    reached.sort()
    return Reading(tuple(reached), tuple(open_), later)


def current_stay(history: list[dict], until: datetime.datetime) -> tuple[str, str] | None:
    """The value a row held at *until* and since when (the first of the rows holding it last): a stuck question's
    next one (H-d); None for a row with no history row by then."""
    rows = sorted((r for r in history if _when(r) <= until), key=_when)
    if not rows:
        return None
    value = rows[-1].get("status")
    k = len(rows) - 1
    while k > 0 and rows[k - 1].get("status") == value:
        k -= 1
    return str(value), f"{_when(rows[k]):%Y-%m-%d %H:%M:%S.%f}"[:23]


def longest_stay(history: list[dict], start: datetime.datetime, at: datetime.datetime) -> tuple[str, str, str] | None:
    """The next question's (H-d): of the values a row held from *start* (or its first row, if later) to *at*, the one
    it held longest without a break, ``(value, since, until)``; None where it held none in that span.  The row's delay
    in reaching what it reached at *at* is the sum of its stays, and the longest is where the question goes on."""
    rows = sorted(history, key=_when)
    stays = []
    for r, nxt in zip(rows, [*rows[1:], None], strict=True):
        a, b = _when(r), (_when(nxt) if nxt is not None else at)
        a, b = max(a, start), min(b, at)
        if b > a:
            stays.append((b - a, str(r.get("status")), a, b))
    # a value held across several rows is one stay where they follow one another
    merged: list[list] = []
    for d, v, a, b in stays:
        if merged and merged[-1][1] == v and merged[-1][3] == a:
            merged[-1][0] += d
            merged[-1][3] = b
        else:
            merged.append([d, v, a, b])
    if not merged:
        return None
    _, value, a, b = max(merged, key=lambda m: m[0])
    return value, f"{a:%Y-%m-%d %H:%M:%S.%f}"[:23], f"{b:%Y-%m-%d %H:%M:%S.%f}"[:23]
