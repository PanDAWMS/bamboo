"""The writes that can make a row a statement selects pass the statement's terms on another column (6 b).

ContentsFeeder takes a task at ``defined`` only when one of its datasets is ``defined`` or ``toupdate``:
getDatasetsToFeedContents_JEDI's 130 reads ``(tabT.status=:ts_defined AND tabD.status IN (:dsStatus___in)) OR
(...)``.  A task left at ``defined`` may never have been taken because nothing made its datasets so, and
restartTasksForContentsUpdate_JEDI is what does, at 1844, for the tasks its 1830 selects at ``defined``.  Each of those
facts was in the map -- the term in :attr:`~bamboo.codemap.models.ReadSiteNode.condition_trees`, the write in a
junction's branch, where the write's key comes from in a :class:`~bamboo.codemap.models.KeyFrom` -- but nothing joined
them, so a reader asking why a row was not moved could name the mover and not what wakes it (the querier's ask, MV
item 6 b of 10-25; tmaeno 「buildにいれる」 of 10-30).

For each pair ``S=V`` the statement selects, a term of a conjunction admitting ``V`` -- one every term on ``S`` of
which can hold ``V``, the condition read as an ``any`` of ``all``s -- that tests another subject ``T`` for equality
with settled values is asked of a row at ``V``.  Each branch that can write one of those values to ``T`` is a
:class:`~bamboo.codemap.models.Waker`: one whose outcome states it, or a run-time one whose own frames do not rule it
out (:func:`~bamboo.codemap.models.branch_excludes`, the rule ``strategy._reaching`` reads an observed value by), and
none whose object no row is written from (:attr:`~bamboo.codemap.models.Branch.off_row`).  Its ``keyed_by`` are the
statements whose rows fill a key of the write holding the branch -- the KeyFroms with a ``rows`` source, and the
branches at their ``writes`` lines, as the KeyFrom's docstring reads them -- and ``keyed_at`` the pairs one of those
other than the statement itself selects as well: the write is handed rows at the value the reader asks about.  The
statement's own rows are left out there: a write keyed by them is its mover's, on rows it took, and wakes none it
refused (1871's rows key _check_task_with_unprocessed_inputs' task writes).

Composed from the fragment alone, once the junctions, the read sites and the key sources are made and the arms off
their row are marked: no record is read here, and every fact a waker joins is in the dump beside it.  Not read: a term
whose value nothing settles; ``<>`` / ``NOT IN`` (any other value would do) and ``IS NULL`` (a write of no value);
a column compared with a column or a time; a term on ``S`` itself under another alias, which the admission reads as
``S``.  Nor a value no branch holds -- one an ``INSERT`` gives a row of a table no spec holds (PRODSYS_COMM's
commands) -- or a test in Python on a value a statement read, before the call that leads to the mover
(AtlasAnalWatchDog's lib file): those are map_todo's, after ix.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterator, Optional

from bamboo.codemap.models import (
    KEY_ROWS,
    Branch,
    ConditionNode,
    MapFragment,
    ReadSiteNode,
    Waker,
    branch_excludes,
)

# A condition whose ``any`` of ``all``s grows past this many conjunctions is read whole: each of its terms is taken as
# asked of every pair.  None does on pin m.
CONJUNCTIONS_CAP = 256


def attach(fragment: MapFragment) -> int:
    """Set every read site's ``wakers``; return how many read sites have one."""
    cuts = {subject.name: subject.truncated_to for subject in fragment.subjects}
    writes: dict[str, list[tuple[str, Branch]]] = defaultdict(list)
    for junction in fragment.junctions:
        for branch in junction.branches:
            if branch.off_row is None:
                writes[junction.subject].append((junction.owner, branch))
    keyed: dict[tuple[str, int], set[str]] = defaultdict(set)
    for key_from in fragment.key_froms:
        rows = {source.name for source in key_from.sources if source.kind == KEY_ROWS}
        for lines in key_from.writes.values():
            for line in lines:
                keyed[(key_from.owner, line)] |= rows
    selects = {site.name: set(site.selects) for site in fragment.read_sites}
    count = 0
    for site in fragment.read_sites:
        site.wakers = _wakers(site, writes, cuts, keyed, selects)
        count += bool(site.wakers)
    return count


def _wakers(
    site: ReadSiteNode,
    writes: dict[str, list[tuple[str, Branch]]],
    cuts: dict[str, Optional[int]],
    keyed: dict[tuple[str, int], set[str]],
    selects: dict[str, set[str]],
) -> list[Waker]:
    asked: dict[tuple[str, str, str], set[str]] = defaultdict(
        set
    )  # (term, subject, value) -> pairs
    trees = [reading.tree for reading in site.condition_trees]
    trees += [
        reading.tree
        for route in site.call_readings
        for reading in route.condition_trees
    ]
    for tree in trees:
        conjunctions = _expand(tree)
        for pair in site.selects:
            subject, _, value = pair.partition("=")
            for conjunction in (
                conjunctions if conjunctions is not None else [list(_terms(tree))]
            ):
                if conjunctions is not None and not all(
                    _admits(term, value)
                    for term in conjunction
                    if term.subject == subject
                ):
                    continue
                for term in conjunction:
                    if (
                        term.subject
                        and term.subject != subject
                        and term.node == "equal"
                        and term.settled
                    ):
                        for one in {v.constant for v in term.values if v.constant}:
                            asked[(term.text, term.subject, one)].add(pair)
    found: dict[
        tuple[str, str, str, str, int], bool
    ] = {}  # one per line: a line's branches are one write
    for text, subject, value in asked:
        for owner, branch in writes.get(subject, []):
            stated = branch.outcome == value
            if stated or (
                branch.tier == 2
                and not branch_excludes(branch, value, cuts.get(subject))
            ):
                at = (text, subject, value, owner, branch.line)
                found[at] = found.get(at, False) or stated
    out = []
    for (text, subject, value, owner, line), stated in found.items():
        pairs = asked[(text, subject, value)]
        by = sorted(keyed.get((owner, line), set()))
        out.append(
            Waker(
                pairs=sorted(pairs),
                term=text,
                subject=subject,
                value=value,
                owner=owner,
                line=line,
                stated=stated,
                keyed_by=by,
                keyed_at=sorted(
                    pair
                    for pair in pairs
                    if any(
                        pair in selects.get(name, ())
                        for name in by
                        if name != site.name
                    )
                ),
            )
        )
    return sorted(out, key=lambda w: (w.subject, w.value, w.term, w.owner, w.line))


def _expand(tree: ConditionNode) -> Optional[list[list[ConditionNode]]]:
    """The tree as an ``any`` of ``all``s of its terms; None past :data:`CONJUNCTIONS_CAP`."""
    if tree.node == "any":
        out: list[list[ConditionNode]] = []
        for item in tree.items:
            found = _expand(item)
            if found is None:
                return None
            out += found
        return out if len(out) <= CONJUNCTIONS_CAP else None
    if tree.node == "all":
        out = [[]]
        for item in tree.items:
            found = _expand(item)
            if found is None:
                return None
            out = [left + right for left in out for right in found]
            if len(out) > CONJUNCTIONS_CAP:
                return None
        return out
    return [[tree]]


def _terms(tree: ConditionNode) -> Iterator[ConditionNode]:
    if tree.node in ("any", "all"):
        for item in tree.items:
            yield from _terms(item)
    else:
        yield tree


def _admits(term: ConditionNode, value: str) -> bool:
    """Whether a row holding *value* in the term's subject can meet it: only a settled (un)equality can say no."""
    held = {v.constant for v in term.values}
    if term.node == "equal" and term.settled:
        return value in held
    if term.node == "unequal" and term.settled:
        return value not in held
    return True
