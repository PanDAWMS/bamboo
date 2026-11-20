"""The writes that can make a row a statement selects pass the statement's terms on another column (6 b).

ContentsFeeder takes a task at ``defined`` only when one of its datasets is ``defined`` or ``toupdate``:
getDatasetsToFeedContents_JEDI's 130 reads ``(tabT.status=:ts_defined AND tabD.status IN (:dsStatus___in)) OR
(...)``.  A task left at ``defined`` may never have been taken because nothing made its datasets so, and
restartTasksForContentsUpdate_JEDI is what does, at 1844, for the tasks its 1830 selects at ``defined``.  Each of those
facts was in the map -- the term in :attr:`~bamboo.codemap.models.ReadSiteNode.condition_trees`, the write in a
junction's branch, where the write's key comes from in a :class:`~bamboo.codemap.models.KeyFrom` -- but nothing joined
them, so a reader asking why a row was not moved could name the mover and not what wakes it (the querier's ask, MV
item 6 b of 10-25; tmaeno: "put it in the build", 10-30).

For each pair ``S=V`` the statement selects, a term of a conjunction admitting ``V`` -- one every term on ``S`` of
which can hold ``V``, the condition read as an ``any`` of ``all``s -- that tests another subject ``T`` for equality
with settled values is asked of a row at ``V``; so is a plain conjunct on ``T``, which no tree holds --
:attr:`~bamboo.codemap.models.ReadSiteNode.conditions` are the terms ``selects`` does not say -- but
:attr:`~bamboo.codemap.models.ReadSiteNode.restricts` does: restartTasksForContentsUpdate_JEDI's 1830 asks
``tabD.state=:dsState`` of every row, spelled as its pair ``JediDatasetSpec.state=mutable`` (the first build read
the trees alone, and missed 88 such terms on 56 statements).  Each branch that can write one of those values to
``T`` is a :class:`~bamboo.codemap.models.Waker`: one whose outcome states it, or a run-time one whose own frames
do not rule it out (:func:`~bamboo.codemap.models.branch_excludes`, the rule ``strategy._reaching`` reads an
observed value by), and
none whose object no row is written from (:attr:`~bamboo.codemap.models.Branch.off_row`) or whose writes leave its
column as it was (:attr:`~bamboo.codemap.models.Branch.unchanged`).  Its ``keyed_by`` are the
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
from typing import TYPE_CHECKING, Any, Callable, Iterator, NamedTuple, Optional, cast

from bamboo.codemap import pathcond
from bamboo.codemap.models import (
    BORN_AFTER,
    BORN_DEFAULT,
    BORN_NOT,
    BORN_OPEN,
    BORN_UNDECIDED,
    BORN_UNTIL,
    BORN_WRITTEN,
    KEY_PARAMETER,
    KEY_ROWS,
    BornTerm,
    Branch,
    ConditionNode,
    ConditionValue,
    MapFragment,
    ReadSiteNode,
    Waker,
    branch_excludes,
    writes_its_value,
)

if TYPE_CHECKING:
    from bamboo.codemap.calls import Calls

# A condition whose ``any`` of ``all``s grows past this many conjunctions is read whole: each of its terms is taken as
# asked of every pair.  None does on pin m.
CONJUNCTIONS_CAP = 256

WAKER_BRANCH = "branch"
WAKER_INSERT = "insert"
WAKER_BORN = "born"


def attach(
    fragment: MapFragment,
    calls: Optional["Calls"] = None,
    defaults: Optional[dict[str, dict[str, Optional[str]]]] = None,
) -> int:
    """Set every read site's ``wakers``; return how many read sites have one.  *defaults* -- the DDL's
    (:func:`sqltext.read_defaults`) -- with *calls*, read the ``born`` ones."""
    cuts = {subject.name: subject.truncated_to for subject in fragment.subjects}
    writes: dict[str, list[tuple[str, Branch]]] = defaultdict(list)
    for junction in fragment.junctions:
        for branch in junction.branches:
            if writes_its_value(branch):
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
    born = _Born(inserts, defaults) if inserts is not None and defaults is not None else None
    for site in fragment.read_sites:
        site.wakers = _wakers(site, writes, cuts, keyed, sites, inserts)
        if born is not None:
            site.wakers += born.of(site)
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
    # A plain conjunct is in no tree -- conditions are what selects does not say -- but restricts holds it: every row
    # the statement returns holds one of its values, whatever pair it is asked about.
    tested = {term.subject for tree in trees for term in _terms(tree)}
    restricted: dict[str, set[str]] = defaultdict(set)
    for pair in site.restricts:
        subject, _, value = pair.partition("=")
        restricted[subject].add(value)
    for other, held in restricted.items():
        if other in tested:
            continue
        for pair in site.selects:
            if pair.partition("=")[0] != other:
                for one in held:
                    asked[(f"{other}={one}", other, one)].add(pair)

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
                key = (text, subject, value, owner, cast(int, branch.line))  # a write's branch has its line
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
        merged: dict[
            tuple[str, str, str, str, int, bool], tuple[set[tuple[str, ...]], set[str]]
        ] = {}
        for text, subject, value in asked:
            for owner, line, stated, routes, by in inserts.of(subject, value):
                ways, names = merged.setdefault(
                    (text, subject, value, owner, line, stated), (set(), set())
                )
                ways.update(tuple(route) for route in routes)
                names.update(by)
        for (text, subject, value, owner, line, stated), (ways, names) in merged.items():
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
                owner = node.named_as
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


# -- a row born taken ----------------------------------------------------------- #

#: The database's clocks, as a :class:`~bamboo.codemap.models.ConditionValue`'s ``base`` names them.
_DATABASE_CLOCKS = ("CURRENT_DATE", "SYSDATE", "CURRENT_TIMESTAMP", "SYSTIMESTAMP")
#: Python's, by what the call resolves to, as the base names them.
_PYTHON_CLOCKS = ("naive_utcnow", "datetime.utcnow", "datetime.now")
#: SQL's ``NULL`` in the new row: a column the ``INSERT`` leaves to a DDL that gives no default, or writes ``None``.
_NULL = object()


class _OnRow(NamedTuple):
    """What a column holds in the row an ``INSERT`` makes."""

    how: str  # BORN_WRITTEN, BORN_DEFAULT, BORN_OPEN or BORN_UNDECIDED
    value: Any  # a literal, _NULL, or None where open or undecided
    clock: str  # the clock the column is stamped with, or ""
    spelled: str


def _clock(text: str) -> str:
    """The clock *text* reads, named as a base names it -- ``naive_utcnow()`` is ``naive_utcnow``, a DDL's
    ``sysdate`` ``SYSDATE`` --, or ``""``.  Clocks are named, not equated."""
    spelled = text.strip()
    if spelled.upper() in _DATABASE_CLOCKS:
        return spelled.upper()
    called = spelled[:-2].replace("datetime.datetime.", "datetime.") if spelled.endswith("()") else ""
    return called if called in _PYTHON_CLOCKS else ""


def _literal_value(text: str) -> Any:
    """A literal as the reach record or the DDL spells it -- ``'failed'``, ``0``, ``None``, ``NULL`` -- or ``...``."""
    if text.strip().upper() == "NULL":
        return _NULL
    try:
        held = ast.literal_eval(text.strip())
    except (ValueError, SyntaxError):
        return ...
    if held is None:
        return _NULL
    return held if isinstance(held, (str, int, float)) and not isinstance(held, bool) else ...


def _spelled(value: Any) -> str:
    """A literal as ``selects`` spells values: ``user``, ``800``."""
    return value if isinstance(value, str) else str(value)


class _Born:
    """The ``INSERT``s whose new rows a poll's terms take (:data:`WAKER_BORN`)."""

    def __init__(self, inserts: _Inserts, defaults: dict[str, dict[str, Optional[str]]]):
        self.inserts = inserts
        self.defaults = defaults
        self.by_table: dict[str, list] = defaultdict(list)
        for (table, _column), held in inserts.columns.items():
            for node, write, _one in held:
                if all(write is not seen for _node, seen in self.by_table[table]):
                    self.by_table[table].append((node, write))

    def of(self, site: ReadSiteNode) -> list[Waker]:
        """The ``born`` wakers of *site*: a poll of one table, no key bound, a term every row it takes holds."""
        if len(site.tables) != 1 or site.keys:
            return []
        unconditional = set(site.conditions)
        trees = [reading.tree for reading in site.condition_trees if reading.entry in unconditional]
        added = [reading.tree for reading in site.condition_trees if reading.entry not in unconditional]
        if not trees and not site.restricts:
            return []
        conjunctions: list[list[ConditionNode]] = [[]]
        for tree in trees:
            found = _expand(tree)
            if found is None:
                return []
            conjunctions = [left + right for left in conjunctions for right in found]
            if len(conjunctions) > CONJUNCTIONS_CAP:
                return []
        restricted: dict[str, set[str]] = defaultdict(set)
        for pair in site.restricts:
            subject, _, value = pair.partition("=")
            restricted[subject].add(value)
        table = site.tables[0].lower()
        held: dict[tuple[str, int], tuple[list[BornTerm], set[tuple[str, ...]], set[str], list[str]]] = {}
        for node, write in self.by_table.get(table, []):
            columns = {one.column.lower(): one for one in write.columns}
            best: Optional[tuple[tuple[int, int, int], list[BornTerm]]] = None
            for conjunction in conjunctions:
                terms = [self._term(term, columns, table) for term in conjunction]
                terms += [self._restricted(subject, values, columns, table) for subject, values in restricted.items()]
                if any(verdict in (BORN_NOT, BORN_UNDECIDED) for verdict, _term in terms):
                    continue
                said = [one for _verdict, one in terms]
                rank = (
                    sum(one.holds == BORN_OPEN for one in said),
                    sum(one.holds in (BORN_AFTER, BORN_UNTIL) for one in said),
                    len(said),
                )
                if best is None or rank < best[0]:
                    best = (rank, said)
            if best is None:
                continue
            said = best[1] + [
                one.model_copy(update={"added": True})
                for tree in added
                for _verdict, one in (self._term(term, columns, table) for term in _terms(tree))
            ]
            read = self.inserts._routes(node)
            routes = {tuple(chain) for chain, _constants in read[0]} if read is not None else set()
            others = {one.path for one in write.columns if one.path}
            by = self.inserts._keyed_by(node.named_as, others, sorted(routes))
            pairs = [pair for pair in site.selects if self._admits(pair, columns, table)]
            for line in write.lines:
                terms_held, ways, names, kept = held.setdefault((node.named_as, line), (said, set(), set(), pairs))
                ways.update(routes)
                names.update(by)
        out = []
        for (owner, line), (said, ways, names, pairs) in held.items():
            met = [one.term for one in said if not one.added]
            out.append(
                Waker(
                    pairs=sorted(pairs),
                    term=" AND ".join(met),
                    subject=site.tables[0],
                    value="",
                    owner=owner,
                    line=line,
                    stated=not any(one.holds == BORN_OPEN for one in said if not one.added),
                    kind=WAKER_BORN,
                    routes=[list(route) for route in sorted(ways)],
                    keyed_by=sorted(names),
                    born=said,
                )
            )
        return sorted(out, key=lambda w: (w.owner, w.line))

    def _on_row(self, subject: str, columns: dict, table: str) -> _OnRow:
        """What *subject*'s column holds in the new row: the ``INSERT``'s value, else the DDL's default."""
        column = subject.rpartition(".")[2].lower()
        written = columns.get(column)
        if written is not None:
            if written.path:
                return _OnRow(BORN_OPEN, None, "", written.path)
            spelled = written.value or ""
            clock = _clock(spelled)
            if clock:
                return _OnRow(BORN_WRITTEN, None, clock, clock)
            value = _literal_value(spelled)
            if value is ...:
                return _OnRow(BORN_OPEN, None, "", spelled)
            return _OnRow(BORN_WRITTEN, value, "", spelled)
        default = self.defaults.get(table, {}).get(column, "NULL")
        if default is None:
            return _OnRow(BORN_UNDECIDED, None, "", "a BEFORE INSERT trigger's")
        text = default
        clock = _clock(text)
        if clock:
            return _OnRow(BORN_DEFAULT, None, clock, clock)
        value = _literal_value(text)
        if value is ...:
            return _OnRow(BORN_UNDECIDED, None, "", text)
        return _OnRow(BORN_DEFAULT, value, "", "NULL" if value is _NULL else text)

    def _term(self, term: ConditionNode, columns: dict, table: str) -> tuple[str, BornTerm]:
        """Whether *term* holds on the new row -- the verdict ``holds``, :data:`BORN_NOT`, :data:`BORN_OPEN` or
        :data:`BORN_UNDECIDED` -- and how, said."""
        row = self._on_row(term.subject, columns, table) if term.subject else _OnRow(BORN_UNDECIDED, None, "", "")

        def said(verdict: str, holds: str, value=None) -> tuple[str, BornTerm]:
            return verdict, BornTerm(
                term=term.text,
                subject=term.subject,
                holds=holds,
                value=row.spelled,
                offset_seconds=value.offset_seconds if value is not None else None,
                scale_seconds=value.scale_seconds if value is not None else None,
                needs=list(value.needs) if value is not None else [],
            )

        if row.how == BORN_UNDECIDED:
            return said(BORN_UNDECIDED, BORN_UNDECIDED)
        if term.node in ("null", "not_null"):
            if row.how == BORN_OPEN:
                return said(BORN_OPEN, BORN_OPEN)
            empty = row.value is _NULL and not row.clock
            return said("holds", row.how) if empty == (term.node == "null") else said(BORN_NOT, BORN_NOT)
        if term.node in ("equal", "unequal"):
            constants = {one.constant for one in term.values}
            if not term.settled or not term.values or "" in constants:
                return said(BORN_UNDECIDED, BORN_UNDECIDED)
            if row.how == BORN_OPEN:
                return said(BORN_OPEN, BORN_OPEN)
            if row.clock:
                return said(BORN_UNDECIDED, BORN_UNDECIDED)
            if row.value is _NULL:
                return said(BORN_NOT, BORN_NOT)  # NULL is neither in a list nor out of it
            inside = _spelled(row.value) in constants
            return said("holds", row.how) if inside == (term.node == "equal") else said(BORN_NOT, BORN_NOT)
        if term.node == "compare" and len(term.values) == 1:
            value = term.values[0]
            if value.base in _DATABASE_CLOCKS or value.base in _PYTHON_CLOCKS:
                if value.offset_seconds is None and value.scale_seconds is None:
                    return said(BORN_UNDECIDED, BORN_UNDECIDED)
                if row.how == BORN_OPEN:
                    return said(BORN_OPEN, BORN_OPEN, value)
                if row.value is _NULL and not row.clock:
                    return said(BORN_NOT, BORN_NOT)
                if row.clock and row.clock == value.base:
                    return said("holds", BORN_AFTER if term.op in ("<", "<=") else BORN_UNTIL, value)
                return said(BORN_UNDECIDED, BORN_UNDECIDED)
            number = _literal_value(value.constant) if value.constant and not value.base else ...
            if isinstance(number, (int, float)):
                if row.how == BORN_OPEN:
                    return said(BORN_OPEN, BORN_OPEN)
                if row.value is _NULL and not row.clock:
                    return said(BORN_NOT, BORN_NOT)
                if isinstance(row.value, (int, float)) and not row.clock:
                    holds = _COMPARE_OPS[term.op](row.value, number)
                    return said("holds", row.how) if holds else said(BORN_NOT, BORN_NOT)
        return said(BORN_UNDECIDED, BORN_UNDECIDED)

    def _restricted(self, subject: str, values: set[str], columns: dict, table: str) -> tuple[str, BornTerm]:
        """A plain conjunct :attr:`ReadSiteNode.restricts` holds, read on the new row as an ``IN`` of its values."""
        term = ConditionNode(
            node="equal",
            text=", ".join(f"{subject}={one}" for one in sorted(values)),
            subject=subject,
            op="IN",
            values=[ConditionValue(kind="constant", text=one, constant=one) for one in sorted(values)],
            settled=True,
        )
        return self._term(term, columns, table)

    def _admits(self, pair: str, columns: dict, table: str) -> bool:
        """Whether the new row can be at the ``selects`` *pair*: not where it holds another literal there."""
        subject, _, value = pair.partition("=")
        row = self._on_row(subject, columns, table)
        if row.how in (BORN_OPEN, BORN_UNDECIDED) or row.clock:
            return True
        return row.value is not _NULL and _spelled(row.value) == value


_COMPARE_OPS: dict[str, Callable[[Any, Any], bool]] = {
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
}
