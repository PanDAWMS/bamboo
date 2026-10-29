"""When an attribute no history records took the value it holds: H2 for such a subject (the harness's ask b, 10-29).

A job's and a task's history keep their status (and a few columns beside it, ``windows.value_held``); an attribute
such as ``JobSpec.nucleus`` or ``JediTaskSpec.rescueTime`` has no row of its own to say when it was written.  The map
says two things that give the time of the write that left the value:

* the value is the time: every arm that can write it writes the time it runs -- ``runtime(CURRENT_DATE)`` on the
  database's clock, ``runtime(naive_utcnow())`` on the writer's (``NOW``).  A name bound to the time earlier in the
  pass (``timeNow``) or the time less a span (``CURRENT_DATE-1``) is not read so.
* the arm's pass writes a history row of the object: the junction branch's :class:`HistoryPair` with ``row`` same or
  can (``bamboo.codemap.panda.runs``) -- at the arm's function or a caller on the way up.  The rows that pass can
  write are the status values its history call records, where they are literals, or any row.

An arm on no row of the object -- its pairs are another object's (``another: ...``), or the record does not follow
the object to the call (``row`` empty), or it has none -- leaves the time not known: its run may have written the
value after any row (KNOWN_ISSUES.md K37).  Only the last write is asked: a write of the same value before it is not.
"""
from __future__ import annotations

from dataclasses import dataclass

#: an arm's outcome that is the time the write runs
NOW = frozenset({"runtime(CURRENT_DATE)", "runtime(naive_utcnow())"})
#: :attr:`HistoryPair.row` values that put the history row on the arm's object
ON_THE_ROW = frozenset({"same", "can"})


@dataclass(frozen=True)
class Arm:
    """One writer line of the subject, as the map's junction branch has it."""

    owner: str  # the reading's ``file::function``
    line: int
    outcome: str | None
    #: the pairs whose history row is the arm's object's
    placed: tuple
    #: why it is on no row of the object; "" when placed
    unplaced: str


def arms(fragment, subject: str, readings: list) -> list[Arm]:
    """Each line of *readings* (H1's writers of *subject*) with the junction branch the map has for it.  A reading's
    owner is ``file::function``, a junction's ``file::Class.function``: they are matched by the file, the function's
    name and the line."""
    branches: dict[tuple[str, str, int], object] = {}
    for j in (fragment.junctions if fragment else []):
        if j.subject != subject:
            continue
        file, _, qual = j.owner.partition("::")
        for b in j.branches:
            branches.setdefault((file, qual.rpartition(".")[2], b.line), b)
    out = []
    for r in readings:
        file, _, func = r.owner.partition("::")
        for line in r.lines:
            b = branches.get((file, func.rpartition(".")[2], line))
            if b is None:
                out.append(Arm(r.owner, line, None, (), "the map has no branch of it"))
                continue
            pairs = b.history or []
            placed = tuple(p for p in pairs if p.row in ON_THE_ROW)
            why = "" if placed else _unplaced(pairs, b.history_hole)
            out.append(Arm(r.owner, line, b.outcome, placed, why))
    return out


def _unplaced(pairs: list, hole: str) -> str:
    if not pairs:
        return hole or "no call in its pass writes a history row"
    another = sorted({p.row.removeprefix("another: ") for p in pairs if p.row.startswith("another: ")})
    if another:
        return f"its passes' history rows are another object's ({', '.join(another)})"
    return f"the object is not followed to its passes' history call ({pairs[0].row_why})"


def now_valued(found: list[Arm]) -> bool:
    """Every arm writes the time it runs."""
    return bool(found) and all(a.outcome in NOW for a in found)


def recorded_values(fragment) -> dict[str, list[str]]:
    """``owner:line`` of each history write -> the values its row can record (:class:`HistoryCall`)."""
    return {f"{n.owner}:{h.line}": list(h.values) for n in (fragment.reach_graph if fragment else []) for h in (n.history or [])}


def candidate_rows(found: list[Arm], states: list[dict], values: dict[str, list[str]]) -> list[dict]:
    """The statuslog rows the placed arms' passes can write, newest first: a pair whose history call records literal
    statuses can write the rows holding one of them; any other, any row."""
    keep: set[int] = set()
    for a in found:
        for p in a.placed:
            recorded = [v for w in p.writes for v in values.get(w, ["runtime(?)"])]
            literal = bool(recorded) and not any(v.startswith("runtime(") for v in recorded)
            keep |= {i for i, s in enumerate(states) if not literal or s["status"] in recorded}
    return [states[i] for i in sorted(keep, reverse=True)]
