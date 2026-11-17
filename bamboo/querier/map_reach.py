"""The map's reach record walked by the map's own rule.

Every walk of the querier's over the map's calls (``reach_graph``) further than one call goes through here, and here
only the map's :meth:`ReachGraph.closure <bamboo.codemap.reach.ReachGraph.closure>` walks: each function as each class
``self`` can be there (``ReachNode.classes``), a call followed only for the classes it holds for
(``ReachCall.for_classes``), into the classes it runs as (``ReachCall.runs_as``, else each the callee can be).  The
querier keeps no rule of its own for which calls go on.

A walk of the calls alone, whatever ``self`` is, goes from PostProcessor.start's ``thr.start()`` through
WorkerThread.run's ``self.runImpl()`` into every agent's runImpl: 1042 functions within six calls, where the map's rule
reaches 192, PostProcessorThread's runImpl the only one (batch 2 part 3, out/gm/qk-design.md)."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from bamboo.codemap import reach as RE
from bamboo.querier import mapdump

Cut = Callable[[tuple[str, str], Any], bool]


def no_threads(node: tuple[str, str], call: Any) -> bool:
    """A cut: a thread started is not followed (``ReachCall.thread``)."""
    return bool(call.thread)


class _Held:
    """The dump the graph was read from, the graph, and the walks asked of it."""

    def __init__(self, fragment: Any = None, record: RE.ReachGraph | None = None):
        self.fragment = fragment
        self.record = record
        self.by_call: dict[tuple[str, int | None, Cut | None], list[tuple[Any, dict[str, int]]]] = {}
        self.callers: dict[str, set[str]] | None = None


_HELD = _Held()


def graph() -> RE.ReachGraph | None:
    """The dump's ``reach_graph`` as the map reads it, read once per dump; None where the dump has none."""
    global _HELD
    fragment = mapdump.fragment()
    if _HELD.fragment is not fragment:
        rows = getattr(fragment, "reach_graph", None) if fragment is not None else None
        _HELD = _Held(fragment, RE.ReachGraph(list(rows)) if rows else None)
    return _HELD.record


def owners(name: str, record: RE.ReachGraph) -> list[str]:
    """The record's functions *name* means: itself where it is one (``file::Class.method``), else each definition its
    ``file::name`` can mean (:meth:`ReachGraph.named`, a junction's owner)."""
    return [name] if name in record.rows else record.named(name)


def _fewest(came: RE.Came) -> dict[str, int]:
    """Each function of a closure with the fewest calls it took (the closure is breadth first; a start 0), the least
    over the classes it was reached as."""
    out: dict[str, int] = {}
    for node in came:
        n = len(RE.way(came, node))
        out[node[0]] = min(out.get(node[0], n), n)
    return out


def reached(starts: Iterable[str], depth: int | None = None, cut: Cut | None = None) -> dict[str, int]:
    """Each function *starts* reach by the map's closure within *depth* calls (None: any number), with the fewest calls
    it takes, a start 0.  A start not in the record means each definition of its name (:func:`owners`); a call *cut*
    says of is not followed.  Empty where the dump has no reach record."""
    record = graph()
    if record is None:
        return {}
    begin = list(dict.fromkeys(o for s in starts for o in owners(s, record)))
    if not begin:
        return {}
    return {o: n for o, n in _fewest(record.closure(begin, cut)).items() if depth is None or n <= depth}


def by_call(owner: str, depth: int | None = None, cut: Cut | None = None) -> list[tuple[Any, dict[str, int]]]:
    """Each call of *owner*'s (``ReachNode.calls``, in order) with what it reaches by the map's closure within *depth*
    calls, its callee at 1.  Each call is asked on its own, from a copy of *owner*'s row holding that call alone, its
    classes kept: the call is followed for the classes it holds for and enters the classes it runs as.  A way that comes
    back into *owner* goes on through its other calls, as it runs there, and counts as inside the call; *owner* itself
    is not among what a call reaches.  A call *cut* says of is not followed, the first one too.  Empty where *owner* is
    not in the record."""
    record = graph()
    if record is None or (row := record.rows.get(owner)) is None:
        return []
    asked = _HELD.by_call
    key = (owner, depth, cut)
    if key not in asked:
        copies = [row.model_copy(update={"owner": f"{owner}#by-call-{n}", "calls": [c]}) for n, c in enumerate(row.calls)]
        with_copies = RE.ReachGraph([*record.rows.values(), *copies])
        out = []
        for call, one in zip(row.calls, copies, strict=True):
            fewest = _fewest(with_copies.closure([one.owner], cut))
            out.append((call, {o: n for o, n in fewest.items()
                               if o not in (owner, one.owner) and (depth is None or n <= depth)}))
        asked[key] = out
    return asked[key]


def callers(owner: str) -> set[str]:
    """The functions with a call into *owner*, every call whatever its classes: whom to ask, not an answer."""
    record = graph()
    if record is None:
        return set()
    if _HELD.callers is None:
        index: dict[str, set[str]] = {}
        for row in record.rows.values():
            for call in row.calls:
                index.setdefault(call.to, set()).add(row.owner)
        _HELD.callers = index
    return _HELD.callers.get(owner, set())


def reaching(target: str, depth: int) -> set[str]:
    """The functions whose calls reach *target* within *depth* calls by the map's closure, *target* aside.  Whom to ask
    is found by reading the calls backwards (:func:`callers`, each call an edge whatever its classes: a function the
    closure reaches *target* from is among them), and each is asked forward (:func:`reached`)."""
    seen: set[str] = set()
    frontier = {target}
    for _ in range(depth):
        frontier = {c for f in frontier for c in callers(f)} - seen
        seen |= frontier
    return {c for c in seen - {target} if target in reached([c], depth)}
