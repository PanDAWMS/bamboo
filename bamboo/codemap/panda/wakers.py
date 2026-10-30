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
other than the statement itself selects as well, read through its ``crossing_selects`` for the write's function: the
write is handed rows at the value the reader asks about.  The statement's own rows are left out there: a write keyed
by them is its mover's, on rows it took, and wakes none it refused (1871's rows key _check_task_with_unprocessed_inputs'
task writes).

**A row born with the value wakes it too.**  TaskCommando moves a task by the rows getTasksToExecCommand_JEDI's 4339
selects from PRODSYS_COMM, ``comm_cmd IN (commandStatusMap().keys())``, and those rows are made by
sendCommandTaskPanda's ``INSERT`` (4038), no branch: ``COMM_CMD`` takes its parameter ``comStr``, which
TypicalWatchDogBase hands ``"finish"`` for the tasks getAchievedTasks_JEDI's 3297 selects.  So, given the corpus's
calls, an ``INSERT`` of a reach record (:attr:`~bamboo.codemap.models.ReachNode.writes`) whose column is the term's is a
waker of kind ``insert`` where it writes the value as a literal, or takes a parameter the function does not bind
again and some route into it hands the value (:meth:`~bamboo.codemap.calls.Calls.route_constant`, as
:attr:`~bamboo.codemap.models.KeyFrom.out_on` reads routes): ``routes`` are those, and a waker stated ``False`` holds
the routes whose constant is not known.  Its ``keyed_by`` are the statements whose rows the function's other
parameter columns come from on those routes: the ``rows`` sources of the function's KeyFroms that read such a
parameter, each whose ``via`` ends with a route's calls, outward-in -- 3297 for ``COMM_TASK``'s ``jediTaskID`` on the
route through TypicalWatchDogBase 103.

Composed once the junctions, the read sites, the key sources and the reach records are made and the arms off their
row are marked; without the corpus's calls, from the fragment alone and branches only.  Not read: a term whose value
nothing settles; ``<>`` / ``NOT IN`` (any other value would do) and ``IS NULL`` (a write of no value); a column
compared with a column or a time; a term on ``S`` itself under another alias, which the admission reads as ``S``; an
``INSERT``'s column from an object's attribute (the branches on the object say it), a local or an expression; and a
test in Python on a value a statement read, before the call that leads to the mover (AtlasAnalWatchDog's lib file),
which is map_todo's.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from typing import TYPE_CHECKING, Iterator, Optional

from bamboo.codemap.models import (
    KEY_PARAMETER,
    KEY_ROWS,
    Branch,
    ConditionNode,
    MapFragment,
    ReadSiteNode,
    Waker,
    branch_excludes,
)
from bamboo.codemap.panda import pathcond

if TYPE_CHECKING:
    from bamboo.codemap.calls import Calls

# A condition whose ``any`` of ``all``s grows past this many conjunctions is read whole: each of its terms is taken as
# asked of every pair.  None does on pin m.
CONJUNCTIONS_CAP = 256

WAKER_BRANCH = "branch"
WAKER_INSERT = "insert"


def attach(fragment: MapFragment, calls: Optional["Calls"] = None) -> int:
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
    sites = {site.name: site for site in fragment.read_sites}
    inserts = _Inserts(fragment, calls) if calls is not None else None
    count = 0
    for site in fragment.read_sites:
        site.wakers = _wakers(site, writes, cuts, keyed, sites, inserts)
        count += bool(site.wakers)
    return count


def _wakers(
    site: ReadSiteNode,
    writes: dict[str, list[tuple[str, Branch]]],
    cuts: dict[str, Optional[int]],
    keyed: dict[tuple[str, int], set[str]],
    sites: dict[str, ReadSiteNode],
    inserts: Optional["_Inserts"],
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

    def at(pairs: set[str], by: list[str], owner: str) -> list[str]:
        """The pairs a statement other than this one keying the write selects, on the way its rows go to *owner*."""
        return sorted(
            pair
            for pair in pairs
            if any(
                pair in sites[name].crossing_selects.get(owner, sites[name].selects)
                for name in by
                if name != site.name and name in sites
            )
        )

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
                key = (text, subject, value, owner, branch.line)
                found[key] = found.get(key, False) or stated
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
                keyed_at=at(pairs, by, owner),
            )
        )
    if inserts is not None:
        # One per write and line, however many reach records hold the write: their routes and statements together.
        held: dict[
            tuple[str, str, str, str, int, bool], tuple[set[tuple[str, ...]], set[str]]
        ] = {}
        for text, subject, value in asked:
            for owner, line, stated, routes, by in inserts.of(subject, value):
                ways, names = held.setdefault(
                    (text, subject, value, owner, line, stated), (set(), set())
                )
                ways.update(tuple(route) for route in routes)
                names.update(by)
        for (text, subject, value, owner, line, stated), (ways, names) in held.items():
            pairs = asked[(text, subject, value)]
            by = sorted(names)
            out.append(
                Waker(
                    pairs=sorted(pairs),
                    term=text,
                    subject=subject,
                    value=value,
                    owner=owner,
                    line=line,
                    stated=stated,
                    kind=WAKER_INSERT,
                    routes=[list(route) for route in sorted(ways)],
                    keyed_by=by,
                    keyed_at=at(pairs, by, owner),
                )
            )
    return sorted(
        out,
        key=lambda w: (
            w.subject,
            w.value,
            w.term,
            w.kind,
            w.owner,
            w.line,
            not w.stated,
        ),
    )


class _Inserts:
    """The ``INSERT``s of the reach records, by the column each writes: who writes a value there, on which routes."""

    def __init__(self, fragment: MapFragment, calls: "Calls"):
        self.calls = calls
        self.tables: dict[str, set[str]] = defaultdict(set)
        for entity in fragment.entities:
            if entity.spec_class:
                self.tables[entity.spec_class] |= {
                    table.lower() for table in entity.tables
                }
        # (table, column) -> [(reach node, the write's lines, the column's value or path, the write's columns)]
        self.columns: dict[tuple[str, str], list] = defaultdict(list)
        for node in fragment.reach_graph:
            for write in node.writes:
                if write.kind != "insert":
                    continue
                for column in write.columns:
                    self.columns[(write.table.lower(), column.column.lower())].append(
                        (node, write, column)
                    )
        # (owner as the KeyFroms spell it, parameter) -> [(rows source name, via)]
        self.carried: dict[tuple[str, str], list[tuple[str, tuple[str, ...]]]] = (
            defaultdict(list)
        )
        for key_from in fragment.key_froms:
            names = {
                source.name
                for source in key_from.sources
                if source.kind == KEY_PARAMETER
            }
            rows = [
                (source.name, tuple(source.via))
                for source in key_from.sources
                if source.kind == KEY_ROWS
            ]
            for name in names:
                self.carried[(key_from.owner, name)].extend(rows)
        self.held: dict[str, Optional[tuple]] = {}

    def of(
        self, subject: str, value: str
    ) -> list[tuple[str, int, bool, list[list[str]], list[str]]]:
        """``(owner, line, stated, routes, keyed_by)`` for each ``INSERT`` that can give *subject* the *value*."""
        cls, _, attr = subject.partition(".")
        out = []
        for table in self.tables.get(cls) or {cls.lower()}:
            for node, write, column in self.columns.get((table, attr.lower()), []):
                owner = f"{node.owner.split('::')[0]}::{node.function}"
                if column.path:
                    found = self._handing(node, column.path, value)
                    if found is None:
                        continue
                    handing, unknown = found
                    ways = [(True, handing), (False, unknown)]
                elif _literal(column.value) == value:
                    ways = [(True, [])]
                else:
                    continue
                others = {
                    one.path for one in write.columns if one.path and one is not column
                }
                for stated, routes in ways:
                    if column.path and not routes:
                        continue
                    by = self._keyed_by(owner, others, routes)
                    for line in write.lines:
                        out.append(
                            (owner, line, stated, [list(route) for route in routes], by)
                        )
        return out

    def _handing(
        self, node, path: str, value: str
    ) -> Optional[tuple[list[tuple[str, ...]], list[tuple[str, ...]]]]:
        """The routes handing parameter *path* the *value*, and those whose constant is not known; None if *path* is
        not a parameter the function leaves as handed."""
        held = self._routes(node)
        if held is None or path not in held[1]:
            return None
        routes, _parameters = held
        handing, unknown = [], []
        for chain, constants in routes:
            got = constants.get(path)
            if got is None:
                unknown.append(chain)
            elif str(got) == value:
                handing.append(chain)
        return handing, unknown

    def _routes(self, node) -> Optional[tuple]:
        """``([(chain, {parameter: constant})], parameters)`` for the reach node's function, read once."""
        if node.owner in self.held:
            return self.held[node.owner]
        from bamboo.codemap.calls import UNSET, Target

        self.held[node.owner] = None
        file = node.owner.split("::")[0]
        tree = self.calls.types.tree(file)
        line = node.anchor.line_start if node.anchor is not None else None
        func = (
            next(
                (
                    one
                    for one, _cls in pathcond.functions_with_owner(tree)
                    if one.lineno == line
                ),
                None,
            )
            if tree is not None and line is not None
            else None
        )
        if func is None:
            return None
        args = func.args
        named = {
            one.arg for one in (*args.posonlyargs, *args.args, *args.kwonlyargs)
        } - {"self", "cls"}
        rebound = {
            one.id
            for one in ast.walk(func)
            if isinstance(one, ast.Name) and isinstance(one.ctx, (ast.Store, ast.Del))
        }
        parameters = named - rebound
        owned = self.calls.types.method_class(file, func)
        target = Target(file, func, owned[1] if owned else None)
        routes, _stops = self.calls.routes(target)
        read = []
        for route in routes:
            constants = {}
            for name in parameters:
                got = self.calls.route_constant(name, target, route.sites)
                if got is not UNSET:
                    constants[name] = got
            read.append((route.chain, constants))
        self.held[node.owner] = (read, parameters)
        return self.held[node.owner]

    def _keyed_by(
        self, owner: str, others: set[str], routes: list[tuple[str, ...]]
    ) -> list[str]:
        """The statements whose rows the write's other parameter columns are carried from, on *routes* (any route
        where none is named)."""
        out = set()
        for name in others:
            for source, via in self.carried.get((owner, name), []):
                if not routes or any(
                    _ends_with(via, tuple(reversed(route))) for route in routes
                ):
                    out.add(source)
        return sorted(out)


def _ends_with(via: tuple[str, ...], tail: tuple[str, ...]) -> bool:
    return len(via) >= len(tail) and via[len(via) - len(tail) :] == tail


def _literal(text: str) -> Optional[str]:
    """A column's value as the reach record spells a literal -- ``'DEFT'`` -- or None."""
    try:
        held = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None
    return held if isinstance(held, str) else None


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
