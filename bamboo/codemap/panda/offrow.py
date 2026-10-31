"""Arms whose object is written to no row: what a reader of a row's value skips.

The querier's ask OS of 10-29 17:00, built on the map's side (tmaeno, 10-29
23:20), in the form its reader asked for: a record on the branch.  An arm is
an assignment of its subject's attribute, and a reader of a value a row holds
-- :meth:`~bamboo.codemap.lookup.CodeMap.producers_of`,
:func:`~bamboo.codemap.strategy._reaching`,
:func:`~bamboo.codemap.reach.moving_off` -- takes every arm as one that can
have written it.  Some assign on a spec their function makes only to ask it
something or to carry values: ``getScoutJobData_JEDI``'s ``taskSpec`` is given
``outDiskCount`` to run ``getOutDiskSize()``, ``checkAllEventsDone``'s
``fileSpec`` the keys of a ``SELECT``.  No row is written from that object, and
:attr:`~bamboo.codemap.models.Branch.off_row` says so where the records
(:mod:`bamboo.codemap.panda.objects`) prove it, at every write of the branch:

* the object is a local the arm's function binds to a new object of the
  subject's class -- what a call of the class's ``__init__`` returns, at every
  binding of the name the arm assigns on;
* followed from there (:func:`bamboo.codemap.objects.follow`), it is not kept
  on ``self``, and meets no hole, no cap and no cut;
* returned, it is followed on into every call of its function the records hold
  (:meth:`_Reader.calls_of`), from the name the call binds its result to, and
  all of this holds there too (:meth:`_Reader.returned`):
  ``get_data_carousel_queued_requests_JEDI``'s ``task_spec`` goes in a list to
  the task buffer's wrapper, which returns it, and on to
  ``DataCarouselInterface.get_requests_to_stage``, whose ``self.taskBufferIF``
  is declared ``Any`` -- the call a hole, of that name --, and is read there
  into a dataframe.  Not where a call binds the result to no name, where the
  function's name is met in the corpus other than as the name a call is made
  by -- ``TimedMethod(global_task_buffer.getJobs, timeout)`` hands the method
  on as a value, and a string may be a ``getattr``'s --, nor more than
  :data:`OUT_DEPTH` calls up;
* in no function it is held in is it put under a parameter other than
  ``self`` and the one it came in by -- ``jobs.append(job)`` puts it into what
  the caller handed, and follow hands nothing back through a parameter --, nor
  is a name holding it handed to a call the records do not follow and the map
  does not say keeps nothing (:meth:`_Reader.unread`): ``pool.put(job)`` on a
  ``pool`` of no known class, ``copy.copy(job)``.  So what is done with it is
  done in its function and in the calls it is handed to, each read;
* no function it is held in writes a table of the class
  (:data:`OFF_ROW_NO_WRITE`), or each write of one there names the columns it
  writes and takes neither the object whole nor the attribute of it
  (:data:`OFF_ROW_NOT_TAKEN`): ``prepareTasksToBeFinished_JEDI`` locks the
  task by an ``UPDATE`` of its row, ``lockedBy`` bound from ``pid``, not from
  the ``taskSpec`` it gives ``lockedBy = None``.

A class no table is learned for is not read: no row, so not this question.

**What it rests on.**  The records, not the branch's own text as
:func:`~bamboo.codemap.models.branch_excludes` reads it: the premises of
:data:`bamboo.codemap.objects.PREMISES`, and what the records do not read
that the checks above do not see -- a lambda's or a nested function's use of
a name, a ``with`` target, a container a module or a class names, which no
path is rooted at (:mod:`bamboo.codemap.panda.objects` lists the rest).  The
object is followed, not its values: a value read off it into another
statement's columns -- ``x = spec.splitRule``, then ``:splitRule`` bound to
``x`` -- is not read here.  For an object returned, that a function of the
corpus whose name is met only as the name calls are made by is called at the
calls the records hold and nowhere else: placed at it, or left open, as
``closed_by_name`` reads an open call (:data:`bamboo.codemap.reach.PREMISES`).
"""

from __future__ import annotations

import ast
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, NamedTuple, Optional

from bamboo.codemap import callfacts, objects, reach
from bamboo.codemap import calls as C
from bamboo.codemap.models import (
    BIND_RESULT,
    OFF_ROW_NO_WRITE,
    OFF_ROW_NOT_TAKEN,
    Branch,
    JunctionNode,
    MapFragment,
    OffRow,
    PathBinding,
    ReachNode,
    SourceModule,
)
from bamboo.codemap.panda import mustprints, specwrites
from bamboo.codemap.panda.objects import is_parameter

#: How :func:`callfacts.meaning` reads a call the records follow, or one that keeps nothing it is handed.
_FOLLOWED = (callfacts.DEFINITION, callfacts.LOGGER, callfacts.CLIENT)
#: ... and a method whose puts the records read (:mod:`bamboo.codemap.panda.objects`' fills).
_FILLING = (callfacts.LIBRARY, callfacts.BUILTIN, callfacts.ASSUMED)
#: How many calls up a returned object is followed: the proxy's method, the task buffer's wrapper and its caller
#: are two.  Further, it is not read.
OUT_DEPTH = 4


def attach(fragment: MapFragment, modules: list[SourceModule], calls: C.Calls) -> int:
    """Set :attr:`Branch.off_row` on every branch of *fragment* the records prove it for; returns how many."""
    reader = _Reader(fragment, modules, calls)
    found = 0
    for junction in fragment.junctions:
        for branch in junction.branches:
            branch.off_row = reader.off_row(junction, branch)
            found += branch.off_row is not None
    return found


class _Reader:
    def __init__(
        self, fragment: MapFragment, modules: list[SourceModule], calls: C.Calls
    ):
        self.calls = calls
        self.rows = {row.owner: row for row in fragment.reach_graph}
        self.names = reach._Names(calls.types)
        self.funcs = {owner: (file, func) for file, func, owner in self.names.functions}
        self.trees = {module.rel_path: module.tree for module in modules}
        self.classes = objects.table_classes(fragment.entities)
        self._calls: Optional[_Calls] = None

    def off_row(self, junction: JunctionNode, branch: Branch) -> Optional[OffRow]:
        """Why no row is written from the object *branch* assigns on, at each of its writes; ``None`` where one may be."""
        cls, _sep, attr = junction.subject.rpartition(".")
        tables = {table for table, of in self.classes.items() if of == cls}
        sites = mustprints._sites(junction, branch, self.trees) if tables else []
        if isinstance(sites, str) or not sites:
            return None
        found = _Found(attr, tables)
        for site in sites:
            owner = self.names.of(site.func)
            row = self.rows.get(owner) if owner is not None else None
            if owner is None or row is None:
                return None
            lines = range(
                site.statement.lineno,
                (site.statement.end_lineno or site.statement.lineno) + 1,
            )
            paths = {
                one.path
                for one in row.assigns
                if one.line in lines and attr in one.attrs
            }
            if not paths:
                return None
            for path in paths:
                if not _made_here(row, path, cls) or not self.held_off(
                    owner, path, found, 0
                ):
                    return None
        return OffRow(
            why=OFF_ROW_NOT_TAKEN if found.writes else OFF_ROW_NO_WRITE,
            functions=sorted(found.functions),
            writes=sorted(found.writes),
        )

    def held_off(self, owner: str, path: str, found: "_Found", depth: int) -> bool:
        """Whether no row of *found*'s tables is written from what *owner* holds by *path*: the object followed
        (:func:`~bamboo.codemap.objects.follow`), and where it is returned, on into the calls of *owner*
        (:meth:`returned`), *depth* calls up from the arm's function.  What it is held in, and the writes that do not
        take it, go into *found*."""
        if (owner, path) in found.seen:
            return True
        found.seen.add((owner, path))
        followed = objects.follow(self.rows, owner, path)
        if followed.holes or followed.cut or followed.capped:
            return False
        came: dict[str, set[str]] = {}
        for at, path_in in [(owner, path), *followed.entered]:
            came.setdefault(at, set()).add(objects._root(path_in))
        for at, held in followed.held.items():
            if self.left_by_a_parameter(at, held, came.get(at, set())) or self.unread(
                at, held
            ):
                return False
            for write in self.rows[at].writes:
                if write.table.rpartition(".")[2].lower() not in found.tables:
                    continue
                if write.unread or not (write.whole or write.columns):
                    return False
                if write.whole and objects._holds(set(held), write.whole):
                    return False
                if any(
                    value.path and objects._under(one, value.path) == f".{found.attr}"
                    for value in write.columns
                    for one in held
                ):
                    return False
                found.writes |= {
                    f"{at}:{line} {write.kind} {write.table}" for line in write.lines
                }
        found.functions |= set(followed.held)
        for escape in followed.escapes:
            ends = [
                one
                for one in followed.held.get(owner, [])
                if one.startswith(objects.RETURN)
            ]
            if (
                escape.how != objects.RETURNED
                or not ends
                or not self.returned(owner, ends, found, depth)
            ):
                return False
        return True

    def returned(
        self, function: str, ends: list[str], found: "_Found", depth: int
    ) -> bool:
        """Whether no row is written from what *function* returns as *ends*, in any call of it (:meth:`calls_of`):
        the name a call binds its result to holds it, as :func:`~bamboo.codemap.objects.follow` hands a callee's
        return back to a call it is followed in by -- ``x = f()[0]`` holds what is at ``[0]`` --, and
        :meth:`held_off` reads it there, or this again, where the caller returns the call's result in turn.  Not
        more than :data:`OUT_DEPTH` calls up."""
        if depth >= OUT_DEPTH:
            return False
        calls = self.calls_of(function)
        if calls is None:
            return False
        for caller, result in calls:
            for end in ends:
                below = objects._under(objects.RETURN + result.path, end)
                if below is None:
                    continue
                name = result.name + below
                if name.startswith(objects.RETURN):
                    found.functions.add(caller)
                    held = self.returned(caller, [name], found, depth + 1)
                else:
                    held = self.held_off(caller, name, found, depth + 1)
                if not held:
                    return False
        return True

    def calls_of(self, function: str) -> Optional[list[tuple[str, PathBinding]]]:
        """Each call of *function* the records hold, as the caller and a binding of the call's result: a call placed
        at it, or a hole whose line makes a call of its name -- read from the source, as a hole's callee is cut at
        80 characters; a hole calling no name calls what a name holds, which the name met as a value says.  None
        where they cannot be said: none is held; its name is met in the corpus other than as the name a call is
        made by (:func:`_met_otherwise`); a hole's call is of a name not read; a call binds its result to no name
        -- used as it is, or dropped."""
        calls = self._index()
        name = function.rpartition("::")[2].rpartition(".")[2]
        sites = calls.placed.get(function, []) + calls.opened.get(name, [])
        if calls.met[name] or not sites or calls.opened.get(""):
            return None
        found = []
        for caller, line, placed in sites:
            results = [
                binding
                for binding in self.rows[caller].bindings
                if binding.how == BIND_RESULT
                and binding.line == line
                and (function in binding.to if placed else not binding.to)
            ]
            if not results:
                return None
            found += [(caller, result) for result in results]
        return found

    def _index(self) -> "_Calls":
        """The calls the records hold, read once: :class:`_Calls`."""
        if self._calls is None:
            placed: dict[str, list[tuple[str, int, bool]]] = defaultdict(list)
            opened: dict[str, list[tuple[str, int, bool]]] = defaultdict(list)
            for owner, row in self.rows.items():
                for call in row.calls:
                    placed[call.to].append((owner, call.line, True))
                func = self.funcs[owner][1] if owner in self.funcs else None
                for hole in row.holes:
                    for name in _names_called(func, hole.line, hole.callee):
                        opened[name].append((owner, hole.line, False))
            self._calls = _Calls(placed, opened, _met_otherwise(self.trees.values()))
        return self._calls

    def left_by_a_parameter(self, owner: str, held: list[str], came: set[str]) -> bool:
        """Whether *owner* holds the object under a parameter other than ``self`` it did not come in by: it is then in
        what the caller handed, which :func:`~bamboo.codemap.objects.follow` does not hand back."""
        if owner not in self.funcs:
            return True
        func = self.funcs[owner][1]
        return any(
            root not in came and root != "self" and is_parameter(func, root)
            for root in {objects._root(one) for one in held}
        )

    def unread(self, owner: str, held: list[str]) -> list[str]:
        """The calls of *owner* handed a name holding the object -- an argument, or a method's receiver -- that the
        records do not follow and the map does not say keep nothing: ``call@line``, empty where there is none.

        Followed: a definition of the corpus; a builtin's, a library's or an assumed method, whose puts the records
        read; a builtin function :data:`callfacts.BUILTIN_FUNCTIONS` says what it gives.  Keeping nothing: a logger,
        another system (:data:`callfacts.ASSUMPTIONS`).  Not a hand-off: an attribute set on the object by a computed
        name (:mod:`bamboo.codemap.panda.specwrites` -- a spec's ``__init__`` and ``pack`` run
        ``object.__setattr__(self, attr, val)``), the object its target only.
        """
        if owner not in self.funcs:
            return [f"{owner}: no function"]
        file, func = self.funcs[owner]
        roots = {objects._root(one).partition("@")[0] for one in held}
        found = []
        for node in ast.walk(func):
            if not isinstance(node, ast.Call) or not _handed(node, roots):
                continue
            written = specwrites._write_of(node)
            if (
                written is not None
                and _named(written.target, roots)
                and not any(
                    _named(one, roots)
                    for one in (written.name, written.value)
                    if one is not None
                )
            ):
                continue  # an attribute set on it by a computed name, as ``x.attr = v`` sets one: nothing handed
            builtin = callfacts.builtin_function(self.calls.types, file, func, node)
            if builtin is not None:
                if builtin.gives is None:
                    found.append(f"{ast.unparse(node.func)}@{node.lineno}")
                continue
            meant = callfacts.meaning(self.calls, file, func, node)
            if meant.kind in _FOLLOWED or (
                meant.kind in _FILLING and meant.method is not None
            ):
                continue
            found.append(f"{ast.unparse(node.func)}@{node.lineno}")
        return found


def _handed(call: ast.Call, roots: set[str]) -> bool:
    """Whether *call* is handed one of *roots* by its name: an argument, a keyword's value or a method's receiver."""
    values = [*call.args, *(keyword.value for keyword in call.keywords)]
    if isinstance(call.func, ast.Attribute):
        values.append(call.func.value)
    return any(_named(one, roots) for one in values)


def _named(expression: ast.expr, roots: set[str]) -> bool:
    """Whether *expression* is one of *roots* by its name, starred or not."""
    if isinstance(expression, ast.Starred):
        expression = expression.value
    return isinstance(expression, ast.Name) and expression.id in roots


def _made_here(row: ReachNode, path: str, cls: str) -> bool:
    """Whether *path* is a local of *row*'s function that every binding of it gives a new object of *cls*."""
    if objects._root(path) != path:
        return False
    bound = [binding for binding in row.bindings if binding.name == path]
    return bool(bound) and all(
        binding.how == BIND_RESULT
        and not binding.path
        and any(to.rpartition("::")[2] == f"{cls}.__init__" for to in binding.to)
        for binding in bound
    )


@dataclass
class _Found:
    """What one arm's object is read to: its subject's attribute and the tables of its class; the functions it is
    held in and the writes of those tables that do not take it, so far; the (function, path) states read."""

    attr: str
    tables: set[str]
    functions: set[str] = field(default_factory=set)
    writes: set[str] = field(default_factory=set)
    seen: set[tuple[str, str]] = field(default_factory=set)


class _Calls(NamedTuple):
    """The calls the records hold, as ``(the caller, the line, placed)``: by the function each is placed at, and by
    each name a call on a hole's line is made by (:func:`_names_called`); and how often each name is met other than
    as the name a call is made by (:func:`_met_otherwise`)."""

    placed: dict[str, list[tuple[str, int, bool]]]
    opened: dict[str, list[tuple[str, int, bool]]]
    met: Counter[str]


def _names_called(func: Optional[ast.AST], line: int, callee: str) -> list[str]:
    """The names the calls of *func* on *line* are made by, a hole's call among them.  A call of no name --
    ``srcImplMap[subType](...)`` -- calls what a name holds, which a function comes to only where its name is met
    as a value (:func:`_met_otherwise`), so it adds none.  Where the source holds no call there, the last name of
    *callee*, the hole's text; a cut text names none, and ``""`` stands for it: a call of any name
    (:meth:`_Reader.calls_of` then says no function's calls)."""
    calls = (
        [
            node
            for node in ast.walk(func)
            if isinstance(node, ast.Call)
            and node.lineno <= line <= (node.end_lineno or node.lineno)
        ]
        if func is not None
        else []
    )
    if calls:
        return [
            node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
            for node in calls
            if isinstance(node.func, (ast.Attribute, ast.Name))
        ]
    # reach cuts a callee's text at 80 characters, the last one an ellipsis (reach._named_call reads it so too)
    if callee.endswith("\N{HORIZONTAL ELLIPSIS}"):
        return [""]
    _head, _dot, last = callee.rpartition(".")
    return [last] if last.isidentifier() else []


def _met_otherwise(trees: Iterable[ast.AST]) -> Counter[str]:
    """How often each name is met in *trees* other than as the name a call is made by: an attribute or a name read as
    a value -- a bound method handed on, to be called where no call of its name is -- or a string, which a
    ``getattr`` may take."""
    trees = list(trees)
    called = {
        id(node.func)
        for tree in trees
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    met: Counter[str] = Counter()
    for tree in trees:
        for node in ast.walk(tree):
            if id(node) in called:
                continue
            if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                met[node.attr] += 1
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                met[node.id] += 1
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                met[node.value] += 1
    return met
