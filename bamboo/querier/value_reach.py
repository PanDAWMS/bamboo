"""M2: the movers relevant to a stay at a value are the writers the rows of a read selecting it reach.

A mover is a read, a modify and a write in a chain: to move a row on from a value it reads the row first.  So the
writers that can move a row on from value V are the ones the rows of a statement selecting V reach -- the statement's
own function where the map places the rows' arms (``rerun._reached``), what it hands them to (``crosses_to``), and what
those call, up to :data:`DEPTH` calls in the map's reach graph (out/gm/qk-design.md, the M2 design, tmaeno's OK
2026-11-14 23:02:48).

A writer those rows do not reach is left out only where the object's rows come to it from statements selecting the
subject at other values alone (:func:`other_values`): doActionForReassign gets tasks at ``reassigning`` and TaskRefiner
at ``registered``, ``rerefine`` and ``staged``, so neither moves a task on from ``aborting``.  One also handed the
object's rows by a statement taking it by its key -- TaskCommando, whose tasks come with a command, so it moves
``exhausted`` on by a retry -- or none at all by a statement the map places (getTasksToExecCommand_JEDI's command path,
which moves ``paused`` on by a resume command) takes its row by another key and stays: what drives it is a hop the
reader names (M3(c)), not a writer to drop.  So does one an API endpoint reaches (:func:`entered`), and every writer
where the reach of V is not closed: a statement selecting V with its callers not complete, or a crossing the map could
not place (``crosses_open``).

Spellings: the map's read sites, their crossings and the junctions name a function ``file::function``; its reach graph
names it ``file::Class.function``.  Both are read as ``file::function`` (:func:`bare`): two classes of one file with a
method of one name are one function here, as they are in the crossings the map writes.  The walk itself is the map's
closure over its reach graph (:func:`closure`), each call followed for the classes it holds for: a ``file::function``
start starts as each definition of that name, and what it reaches is read as ``file::function`` again.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from bamboo.querier import map_reach as MR

#: how many calls past a function the rows are handed to are followed in the map's reach graph (the M2 measure's,
#: ``m2_down3``: 26 of the 29 late questions named their movers at 3)
DEPTH = 3


def bare(owner: str) -> str:
    """*owner* as ``file::function``, the class dropped."""
    rel, _, qual = owner.partition("::")
    return f"{rel}::{qual.rpartition('.')[2]}"


@dataclass(frozen=True)
class Reach:
    """Where the rows of the statements selecting a value go.

    ``reads`` names each statement (``function@line``), ``reached`` the functions its rows reach, ``other`` those that
    get the object's rows only from statements selecting the subject at other values, ``entered`` the functions an API endpoint
    reaches, and ``open`` why the reach is not closed (a statement with callers not complete, or a crossing the map
    could not place): there nothing is left out."""

    reads: tuple[str, ...]
    reached: frozenset[str]
    entered: frozenset[str]
    open: tuple[str, ...]
    other: frozenset[str] = frozenset()

    @property
    def closed(self) -> bool:
        """Whether a writer it does not reach can be left out: some statement selects the value, and every one of them
        has its callers complete and every crossing placed."""
        return bool(self.reads) and not self.open

    def leaves_out(self, owner: str) -> bool:
        """Whether *owner*, a writer, is no mover of the stay: the reach is closed, its rows do not reach it, it gets
        the object's rows only from statements selecting other values, and no API endpoint reaches it."""
        b = bare(owner)
        return self.closed and b not in self.reached and b in self.other and b not in self.entered


def closure(starts: Iterable[str], depth: int | None = DEPTH, stop: frozenset[str] = frozenset()) -> frozenset[str]:
    """*starts* and what they call, *depth* calls deep (None: all of it), not going into a function in *stop*: the
    map's closure (:func:`map_reach.reached`), each call followed for the classes it holds for and into those it runs
    as.  A start the reach graph names (``file::Class.function``) starts as that one, a ``file::function`` one as each
    definition of that name."""
    starts = list(starts)
    cut = (lambda node, call: bare(call.to) in stop) if stop else None
    return frozenset({bare(s) for s in starts} | {bare(o) for o in MR.reached(starts, depth, cut)})


def entered(endpoints: Iterable) -> frozenset[str]:
    """Every function an API endpoint of the map reaches, its own and all it calls."""
    return closure((e.function for e in endpoints if getattr(e, "function", None)), None)


def other_values(read_sites: Iterable, subject: str, observed: str) -> frozenset[str]:
    """The functions that get rows of *subject*'s object only from the map's statements selecting it at values other
    than *observed*, and what those call up to :data:`DEPTH` calls (not through such a statement's own function).

    A statement reads the object where its ``selects`` or ``keys`` name an attribute of the subject's class.  Each
    function a statement hands its rows to (``crosses_to``) is told by what the statements handing it the object's
    rows select: *subject* at other values only, the rows holding such a value where they reach it
    (``crossing_selects``, else the statement's ``selects``), puts it here.  One also handed rows by a statement that
    takes the object by its key (TaskCommando's runImpl, the task by ``getTaskWithID_JEDI``'s jediTaskID after a
    command), or by one selecting *observed*, is not here, nor is anything it calls: its row may be any value's.  An
    aggregate hands on no row."""
    cls = f"{subject.partition('.')[0]}."
    pre = f"{subject}="
    kinds: dict[str, set[str]] = {}
    own: set[str] = set()
    for rs in read_sites:
        selects = list(getattr(rs, "selects", None) or [])
        if getattr(rs, "result", "") == "aggregate" or not any(
                n.startswith(cls) for n in [*selects, *(getattr(rs, "keys", None) or [])]):
            continue
        values = {s for s in selects if s.startswith(pre)}
        crossing = getattr(rs, "crossing_selects", None) or {}
        if values and f"{pre}{observed}" not in values:
            own.add(bare(rs.owner))
        for c in getattr(rs, "crosses_to", None) or []:
            held = {s for s in crossing.get(c, selects) if s.startswith(pre)}
            kind = "value" if f"{pre}{observed}" in held else "other" if held else "key"
            kinds.setdefault(bare(c), set()).add(kind)
    only = frozenset(t for t, k in kinds.items() if k == {"other"})
    rest = frozenset(t for t, k in kinds.items() if k != {"other"})
    return (closure(only, stop=frozenset(own - only)) | only) - closure(rest)


def reach(statements: list, crossed: Iterable[str], movers: Iterable[str], entered_by: frozenset[str],
          other: frozenset[str] = frozenset()) -> Reach:
    """The reach of *statements*, those selecting the value that hand on rows.

    *movers* are the writers ``rerun._reached`` gives the rows to, *crossed* the functions the statements (or one
    reading their rows again) hand them to.  The functions they call are reached up to :data:`DEPTH` calls; a
    selecting statement's own function is not reached again through a call -- a mover calling it reads anew, and its
    other statements are reached only as ``_reached`` places them.  *other* is :func:`other_values`'."""
    reads = tuple(f"{rs.owner.split('::')[-1]}@{rs.line}" for rs in statements)
    own = frozenset(bare(rs.owner) for rs in statements)
    starts = [*crossed, *movers]
    reached = closure(starts, stop=own - {bare(o) for o in starts})
    open_ = tuple(
        f"{rs.owner.split('::')[-1]}@{rs.line}: "
        + ("callers not complete" if not getattr(rs, "callers_complete", False) else "a crossing not placed")
        for rs in statements if not getattr(rs, "callers_complete", False) or getattr(rs, "crosses_open", None))
    return Reach(reads, reached, entered_by, open_, other)
