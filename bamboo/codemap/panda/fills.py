"""Which writes of a value a function reads filled it from a row a function fetched: one question on its flow,
read on across calls.

A reader of a run asks it of a statement and a value the statement reads -- a local, ``oldJobStatus``, or an
attribute chain on a name, ``job.jobStatus`` --: which writes of the value reach the read, and of them which filled
it from a row of a statement a function ran, with the statement and what its rows were asked by.  ``killJob``
reads ``job.jobStatus`` at 1017 after ``job.pack(res[0])`` at 998, ``res`` the rows ``fetchall`` gives of the
``SELECT`` run at 992 ``WHERE PandaID=:PandaID``, the bind filled with the parameter ``pandaID``: a reader holding
the run's key can put the row's state there.  At 1071 the arms storing ``job.jobStatus = "closed"`` reach as well,
and are said apart.

* **The writes.**  A local's bindings and ``del`` s.  A chain's: its name's bindings, a store or ``del`` of the chain
  or of a part of it before the column (``self.job = ...``), and a call reading a row into the object holding the
  column (:data:`callfacts.ROW_READ`, ``job.pack(row)``), which fills every column.  The ones that can reach the read
  are :func:`reaching`'s, the function's entry among them.
* **A fill** is such a call, or a binding of a local, whose row is a cursor's fetch (:data:`callfacts.FETCHES`):
  ``fetchone`` 's row, an element at a constant index of the rows ``fetchall`` gives, or a ``for`` target over
  them -- through the names holding them, every binding of each that can reach a fetch's.  A local is filled with
  a column, at its place in the row, or with the row whole.  The statement is the run whose region holds the fetch
  (:func:`sql.regions`), once for each text of it (``% table`` over two tables), with its row tests
  (:func:`sqltext.row_tests`) and the values filling their binds (:func:`sql.bound_values`).
* **Across calls** the reading goes on where a write is not one of the function's own rows:

  - from the **entry**, where the name is a parameter or ``self``, to each call of the function
    (:meth:`Calls.callers`): the same question there of what the call hands it -- the argument, or the object the
    method is called on.  ``AdderGen.process_job_report`` reads ``self.job.jobStatus`` at 415, and ``run`` calls it
    at 114 on ``self``, after storing ``self.job`` at 111;
  - from a **call's value** -- a store of a part of the chain, a binding, a value handed in -- to each return of the
    function it lands in (:meth:`Calls.callees`), the rest of the chain asked of what is returned: of a name a list
    is built in, each element it is given (``append``, ``insert``, a list display), the ones the call's constants
    rule out (:func:`objects.route_constants`, :func:`pathcond.ruled_out_by`) left out.  ``peekJobs(...)[0]`` is an
    element of the list ``peekJobs`` appends each ``proxy.peekJob(...)`` to, and ``peekJob`` returns the job it packed
    from its ``SELECT``;
  - a **constant None** given that way does not reach where a test on the read's way does not run on ``None`` for
    the name or chain holding it, past the write (:func:`pathcond.guards`): ``peekJobs`` appends ``res`` only
    ``if res:``, and ``process_job_report`` returns first ``if self.job is None``.

  Each fill and other says the calls crossed to it (:class:`Hop`, *route*), in the order the reading went, and the
  file it is in.
* **The others**: the entry with no call of the function, a binding not from a row, a store, a ``del``, a
  row-reading call whose row is not followed to a fetch, a call the map does not place or reads no return of, a
  ``None`` no test rules out, and what is past the most calls the reading crosses (``not followed``).  A reader
  holding the rows of the fills alone reads the value; one with any other does not.
* **Between a fill and the read** a call may change the object in place: each change of the chain's first
  attribute or of every attribute that :func:`changes.object_changes` names is kept on the fill, in each function
  the reading passed, from where the value came in to where it went on.  The function's own stores are among the
  others already.

Not read: rows a runner other than the cursor hands back (``querySQLS``), a row through a name one of whose reaching
bindings is not a fetch's, a list grown by ``extend`` or ``+=``, an element of a list a loop takes other than of a
call's value or a name a call's value is bound to, a generator's yields, and a value a thread or a forwarding object
hands on.  The tree needs parent links (:func:`pathcond.attach_parents`).
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING, Iterator, NamedTuple, Optional, Sequence

from bamboo.codemap import callfacts
from bamboo.codemap.calls import CallSite, Target, bind
from bamboo.codemap.models import CHANGE_STORE
from bamboo.codemap.panda import changes, objects, pathcond, sql, sqltext
from bamboo.codemap.panda.reaching import reaching

if TYPE_CHECKING:
    from bamboo.codemap.calls import Calls

#: How a fill takes its row of what the fetch gives (:attr:`Fill.pick`): ``fetchone`` 's row; an element of the
#: rows at a constant index; a ``for`` target over them, one row a pass.
PICK_ONE, PICK_AT, PICK_EACH = "one", "at", "each"

#: What another write reaching the read is (:attr:`Other.kind`).
OTHER_ENTRY = "entry"
OTHER_BINDING = "binding"
OTHER_STORE = "store"
OTHER_DELETE = "delete"
OTHER_ROW_UNREAD = "row unread"
#: Past the most calls a reading crosses, or back in a reading it is in.
OTHER_UNFOLLOWED = "not followed"

#: Which way a reading crossed a call (:attr:`Hop.way`): up, from a function's entry to a call of it; down, from a
#: call's value into the function it lands in.
HOP_UP, HOP_DOWN = "up", "down"

_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)
_NESTED = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
#: The most names a row is followed through: ``rows = res`` and a ``for`` over it is two.
_HOPS = 4
#: The most calls a reading crosses, up and down: ``AdderGen``'s ``self.job`` is three -- ``run`` 's call of
#: ``process_job_report``, its ``peekJobs``, and that one's ``peekJob``.
_CROSSINGS = 4
#: The most readings open at once, one in another: each name a chain is followed through, in this function or
#: across a call -- a loop binding a name from an attribute of itself (``node = node.parent``) adds one each
#: time round, which no reading repeats.
_DEPTH = 12
#: The element a reading asks of a list where any of them may be the one (a loop, an index into a list built).
_EACH = -1


class RowTest(NamedTuple):
    """One test every row of a run holds, and for each bind it reads the values filling it for that run."""

    test: sqltext.Test
    binds: tuple[tuple[str, tuple[sql.Bind, ...]], ...]


class RowRead(NamedTuple):
    """One text of the statement whose row a fill takes: the run, its rows' tests, and the value's column in its
    rows -- for a chain, the attribute the row-reading call fills by name; for a local, the select item at the
    value's place as :func:`sqltext.select_items` names it; ``""`` where it is not told (a hole, an expression,
    the row whole)."""

    run: sql.Execution
    tests: tuple[RowTest, ...]
    column: str


class Hop(NamedTuple):
    """A call a reading crossed: *way* :data:`HOP_UP`, from a function's entry to *site*, a call of it, where the
    value is handed in; or :data:`HOP_DOWN`, from *site*'s value into the function it lands in, where it is
    returned."""

    way: str
    site: CallSite


class Fill(NamedTuple):
    """A write filling the value from a row: the statement, the row it takes, the fetch the row comes from and how
    it is picked of it, each text of the statement run, and the in-place changes between it and the read; the file
    of the function it is in, and the calls crossed from the read to it."""

    site: ast.stmt
    row: ast.expr
    fetch: ast.Call
    pick: str
    reads: tuple[RowRead, ...]
    changes: tuple[changes.ObjectChange, ...] = ()
    file: str = ""
    route: tuple[Hop, ...] = ()


class Other(NamedTuple):
    """A write reaching the read that is no fill: the statement, or the function's ``args`` for its entry; the file
    it is in, and the calls crossed from the read to it."""

    site: ast.AST
    kind: str
    file: str = ""
    route: tuple[Hop, ...] = ()


class RowHeld(NamedTuple):
    """The writes reaching a read (:func:`row_fills`), in the order they are met: the fills and the others."""

    fills: tuple[Fill, ...]
    others: tuple[Other, ...]


class _Origin(NamedTuple):
    """Where a value comes from of a fetch: *shape* :data:`callfacts.ROWS` or :data:`callfacts.ROW`, or ``column``
    at *place*; *pick* how its row is taken."""

    fetch: ast.Call
    shape: str
    pick: str
    place: Optional[int] = None


class _None(NamedTuple):
    """A constant ``None`` a write reaching the read gives: its statement, file and route, and how many of the
    chain's attributes come after the part it is the value of (``self.job = ...[0]`` read as
    ``self.job.jobStatus``: one)."""

    site: ast.AST
    file: str
    route: tuple[Hop, ...]
    rest: int


class _Found:
    """What a reading found: fills, others, and the ``None`` s no test has ruled out yet."""

    def __init__(self) -> None:
        self.fills: list[Fill] = []
        self.others: list[Other] = []
        self.nones: list[_None] = []

    def add(self, other: "_Found") -> None:
        self.fills += other.fills
        self.others += other.others
        self.nones += other.nones


_COLUMN = "column"


def row_fills(
    calls: "Calls",
    file: str,
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    read: ast.expr,
    bound: tuple = (),
) -> RowHeld:
    """The writes of what *read* -- a name, or an attribute chain on one, in *func* of *file* -- holds that can reach
    it, each a :class:`Fill` from a row a function fetched or an :class:`Other`, read on across calls.  *bound*: the
    classes ``self`` can be in *func* (:attr:`Target.bound`), for the in-place changes."""
    if getattr(read, "parent", None) is None:
        raise ValueError(
            f"row_fills needs parent links on the tree holding line {getattr(read, 'lineno', '?')}"
        )
    chain = _chain(read)
    if chain is None:
        raise ValueError(
            f"row_fills reads a name or an attribute chain on one, not {ast.unparse(read)}"
        )
    found = _Walk(calls).level(file, func, chain[0], chain[1], read, bound, (), 0)
    nones = [Other(one.site, OTHER_BINDING, one.file, one.route) for one in found.nones]
    return RowHeld(tuple(found.fills), tuple(found.others + nones))


class _Walk:
    """One question's reading: the functions it is in and the calls it crosses."""

    def __init__(self, calls: "Calls"):
        self.calls = calls
        self._active: set[tuple[int, str, tuple[str, ...], int]] = set()

    def level(
        self,
        file: str,
        func: ast.AST,
        name: str,
        attributes: tuple[str, ...],
        at: ast.AST,
        bound: tuple,
        route: tuple[Hop, ...],
        crossed: int,
    ) -> _Found:
        """The writes of *name* followed by *attributes* reaching *at* in *func* of *file*."""
        key = (id(func), name, attributes, id(at))
        found = _Found()
        if key in self._active or len(self._active) >= _DEPTH:
            found.others.append(Other(_statement(at), OTHER_UNFOLLOWED, file, route))
            return found
        self._active.add(key)
        try:
            self._level(found, file, func, name, attributes, at, bound, route, crossed)
        finally:
            self._active.discard(key)
        return found

    def _level(
        self,
        found: _Found,
        file: str,
        func: ast.AST,
        name: str,
        attributes: tuple[str, ...],
        at: ast.AST,
        bound: tuple,
        route: tuple[Hop, ...],
        crossed: int,
    ) -> None:
        arguments = func.args  # type: ignore[attr-defined]
        binds = _unique(
            [one.site for one in pathcond.bindings(func, name) if not one.nested]
        )
        deletes = _deletes(func, name, attributes)
        stores = _stores(func, name, attributes)
        readers = _row_readers(func, name, attributes[:-1]) if attributes else []
        reader = _Reader(func)  # type: ignore[arg-type]
        statement = _statement(at)
        own = {(file, one.lineno) for one in [*stores, *deletes]}

        def moved(first: ast.AST) -> tuple[changes.ObjectChange, ...]:
            if not attributes:
                return ()
            return tuple(
                change
                for change in changes.object_changes(
                    self.calls, file, func, name, first, statement, bound
                )
                if change.key in (f".{attributes[0]}", changes.EVERY)
                and not (
                    change.kind == CHANGE_STORE and (change.file, change.line) in own
                )
            )

        for site in reaching(at, [arguments, *binds, *stores, *deletes, *readers]):
            if any(site is one for one in deletes):
                found.others.append(Other(site, OTHER_DELETE, file, route))
                continue
            if any(site is one for one in readers):
                assert isinstance(site, ast.Expr) and isinstance(site.value, ast.Call)
                call = site.value
                origins = reader.origins(call.args[0]) if call.args else None
                if not origins or any(one.shape != callfacts.ROW for one in origins):
                    found.others.append(Other(site, OTHER_ROW_UNREAD, file, route))
                    continue
                kept = moved(site)
                found.fills += [
                    Fill(
                        site,
                        call.args[0],
                        one.fetch,
                        one.pick,
                        reader.reads(one, attributes[-1]),
                        kept,
                        file,
                        route,
                    )
                    for one in origins
                ]
                continue
            if site is arguments:
                got = self._entry(file, func, name, attributes, bound, route, crossed)
            elif any(site is one for one in stores):
                got = self._stored(
                    file, func, name, attributes, site, bound, route, crossed
                )
            else:
                got = self._bound(
                    file, func, name, attributes, site, reader, bound, route, crossed
                )
            if got.fills:
                kept = moved(site)
                found.fills += [
                    one._replace(changes=one.changes + kept) for one in got.fills
                ]
            found.others += got.others
            found.nones += [
                one
                for one in got.nones
                if not _none_ruled_out(func, at, name, attributes, one.rest, site)
            ]

    def _entry(
        self,
        file: str,
        func: ast.AST,
        name: str,
        attributes: tuple[str, ...],
        bound: tuple,
        route: tuple[Hop, ...],
        crossed: int,
    ) -> _Found:
        """The value *name* holds where *func* is entered: what each call of it hands it there."""
        found = _Found()
        arguments = func.args  # type: ignore[attr-defined]
        tree = self.calls.types.tree(file)
        cls = pathcond.class_of(tree, func) if tree is not None else None
        handed_self = name in ("self", "cls") and cls is not None
        if not handed_self and not objects.is_parameter(func, name):
            found.others.append(Other(arguments, OTHER_ENTRY, file, route))
            return found
        if crossed >= _CROSSINGS:
            found.others.append(Other(arguments, OTHER_UNFOLLOWED, file, route))
            return found
        target = Target(file, func, cls, bound)
        sites = self.calls.callers(target)
        if not sites:
            found.others.append(Other(arguments, OTHER_ENTRY, file, route))
            return found
        for site in sites:
            hop = route + (Hop(HOP_UP, site),)
            handed = (
                None
                if site.in_module
                else self._handed(site, target, name, handed_self)
            )
            if handed is None:
                found.others.append(
                    Other(_statement(site.call), OTHER_ENTRY, site.file, hop)
                )
                continue
            found.add(
                self._value(
                    site.file,
                    site.func,
                    handed,
                    attributes,
                    OTHER_ENTRY,
                    (),
                    hop,
                    crossed + 1,
                )
            )
        return found

    def _handed(
        self, site: CallSite, target: Target, name: str, handed_self: bool
    ) -> Optional[ast.expr]:
        """What the call at *site* hands *target*'s *name*: the object a method is called on for ``self``, else the
        argument, or the default the definition states where that is a constant; ``None`` where it is not said, or
        the default is not a constant."""
        placed = self.calls.callees(site.file, site.func, site.call)
        if handed_self and not placed.on_class:
            called = site.call.func
            return called.value if isinstance(called, ast.Attribute) else None
        handed = bind(site.call, target, placed.on_class).get(name)
        if handed is None or isinstance(handed, ast.Constant):
            return handed
        return handed if pathcond.enclosing_function(handed) is site.func else None

    def _stored(
        self,
        file: str,
        func: ast.AST,
        name: str,
        attributes: tuple[str, ...],
        site: ast.AST,
        bound: tuple,
        route: tuple[Hop, ...],
        crossed: int,
    ) -> _Found:
        """A store of the chain or of a part of it: a part's value read on for the rest of the chain; the column's own
        store, a ``+=`` and a value not told, an other."""
        if not isinstance(site, ast.AugAssign):
            for target, value in pathcond.assigned_targets(site):
                for one in _written(target):
                    chain = _chain(one)
                    if (
                        chain is None
                        or chain[0] != name
                        or attributes[: len(chain[1])] != chain[1]
                        or len(chain[1]) >= len(attributes)
                    ):
                        continue
                    if value is None:
                        break
                    return self._value(
                        file,
                        func,
                        value,
                        attributes[len(chain[1]) :],
                        OTHER_STORE,
                        bound,
                        route,
                        crossed,
                    )
        found = _Found()
        found.others.append(Other(site, OTHER_STORE, file, route))
        return found

    def _bound(
        self,
        file: str,
        func: ast.AST,
        name: str,
        attributes: tuple[str, ...],
        site: ast.AST,
        reader: "_Reader",
        bound: tuple,
        route: tuple[Hop, ...],
        crossed: int,
    ) -> _Found:
        """A binding of the name: a local filled from a fetched row, a value read on, a loop's element of a call's
        value; else an other."""
        found = _Found()
        if not attributes:
            origins = reader.bound(name, site)
            if origins:
                if any(one.shape == callfacts.ROWS for one in origins):
                    found.others.append(Other(site, OTHER_BINDING, file, route))
                    return found
                assert isinstance(site, ast.stmt)
                found.fills += [
                    Fill(
                        site,
                        _row_of(site),
                        one.fetch,
                        one.pick,
                        reader.reads(one, None),
                        (),
                        file,
                        route,
                    )
                    for one in origins
                ]
                return found
        held = [
            one
            for one in pathcond.bindings(func, name)
            if one.site is site and not one.nested
        ]
        binding = held[0] if len(held) == 1 else None
        if (
            binding is not None
            and isinstance(site, (ast.For, ast.AsyncFor))
            and not binding.unpacked
        ):
            return self._looped(file, func, site, attributes, bound, route, crossed)
        if isinstance(site, (ast.Assign, ast.AnnAssign)):
            value = next(
                (
                    given
                    for target, given in pathcond.assigned_targets(site)
                    if isinstance(target, ast.Name) and target.id == name
                ),
                None,
            )
            if value is not None:
                return self._value(
                    file, func, value, attributes, OTHER_BINDING, bound, route, crossed
                )
        found.others.append(Other(site, OTHER_BINDING, file, route))
        return found

    def _looped(
        self,
        file: str,
        func: ast.AST,
        site: ast.For | ast.AsyncFor,
        attributes: tuple[str, ...],
        bound: tuple,
        route: tuple[Hop, ...],
        crossed: int,
    ) -> _Found:
        """A loop's element: of a call's value, or of a name every binding of which reaching the loop is a call's
        value; else an other."""
        iterable = site.iter
        if isinstance(iterable, ast.Call):
            return self._returned(
                file,
                func,
                iterable,
                _EACH,
                attributes,
                OTHER_BINDING,
                bound,
                route,
                crossed,
            )
        found = _Found()
        if isinstance(iterable, ast.Name):
            sites = [
                one.site
                for one in pathcond.bindings(func, iterable.id)
                if not one.nested
            ]
            given: list[ast.Call] = []
            for one in reaching(iterable, [func.args, *_unique(sites)]):  # type: ignore[attr-defined]
                value = (
                    pathcond.written_value(one) if isinstance(one, ast.Assign) else None
                )
                if not isinstance(value, ast.Call):
                    given = []
                    break
                given.append(value)
            for value in given:
                found.add(
                    self._returned(
                        file,
                        func,
                        value,
                        _EACH,
                        attributes,
                        OTHER_BINDING,
                        bound,
                        route,
                        crossed,
                    )
                )
            if given:
                return found
        found.others.append(Other(site, OTHER_BINDING, file, route))
        return found

    def _value(
        self,
        file: str,
        func: ast.AST,
        value: ast.expr,
        attributes: tuple[str, ...],
        kind: str,
        bound: tuple,
        route: tuple[Hop, ...],
        crossed: int,
    ) -> _Found:
        """What *value*, written or handed in *func*, holds of *attributes*: a chain's writes there, a call's
        returns, a constant ``None``; else an other of *kind*."""
        chain = _chain(value)
        if chain is not None and getattr(value, "parent", None) is not None:
            return self.level(
                file,
                func,
                chain[0],
                chain[1] + attributes,
                value,
                bound,
                route,
                crossed,
            )
        found = _Found()
        if isinstance(value, ast.Constant) and value.value is None:
            found.nones.append(_None(_statement(value), file, route, len(attributes)))
            return found
        call, index = _call_at(value)
        if call is not None:
            return self._returned(
                file, func, call, index, attributes, kind, bound, route, crossed
            )
        found.others.append(Other(_statement(value), kind, file, route))
        return found

    def _returned(
        self,
        file: str,
        func: ast.AST,
        call: ast.Call,
        index: Optional[int],
        attributes: tuple[str, ...],
        kind: str,
        bound: tuple,
        route: tuple[Hop, ...],
        crossed: int,
    ) -> _Found:
        """What *call* returns -- whole, its element at a constant *index*, or any element (:data:`_EACH`) -- read
        for *attributes* in each function it lands in; a call that may land beyond them (:meth:`Calls.opened`) is
        also an other."""
        found = _Found()
        statement = _statement(call)
        if crossed >= _CROSSINGS:
            found.others.append(Other(statement, OTHER_UNFOLLOWED, file, route))
            return found
        placed = self.calls.callees(file, func, call, bound)
        if (
            not placed.targets
            or placed.thread
            or placed.forwarded
            or self.calls.types.class_named(file, call.func) is not None
        ):
            found.others.append(Other(statement, kind, file, route))
            return found
        if placed.unplaced:
            found.others.append(Other(statement, kind, file, route))
        site = CallSite(file, func, call)
        down = route + (Hop(HOP_DOWN, site),)
        into = (site,) + tuple(one.site for one in reversed(_last_downs(route)))
        for target in placed.targets:
            if _yields(target.func):
                found.others.append(Other(statement, kind, file, route))
                continue
            values = objects.route_constants(self.calls, target, into)
            for value, at in _returns(target.func, index, values):
                if value is None:
                    found.others.append(Other(at, OTHER_BINDING, target.file, down))
                    continue
                found.add(
                    self._value(
                        target.file,
                        target.func,
                        value,
                        attributes,
                        OTHER_BINDING,
                        target.bound,
                        down,
                        crossed + 1,
                    )
                )
        return found


class _Reader:
    """One function's runs and the rows its names hold, read once for a question."""

    def __init__(self, func: ast.FunctionDef | ast.AsyncFunctionDef):
        self.func = func
        self._runs: Optional[list[sql.Execution]] = None
        self._regions: dict[int, tuple[int, int]] = {}

    def origins(self, value: ast.expr, hops: int = _HOPS) -> Optional[list[_Origin]]:
        """Where *value* comes from of a fetch, each way; ``None`` where a way is not a fetch's."""
        if isinstance(value, ast.Call):
            called = value.func
            if (
                isinstance(called, ast.Attribute)
                and called.attr in callfacts.FETCHES
                and not value.args
            ):
                gives = callfacts.FETCHES[called.attr].gives
                return [
                    _Origin(value, gives, PICK_ONE if gives == callfacts.ROW else "")
                ]
            return None
        if isinstance(value, ast.Subscript):
            index = value.slice
            if not (isinstance(index, ast.Constant) and type(index.value) is int):
                return None
            inner = self.origins(value.value, hops)
            return None if inner is None else _at(inner, index.value)
        if isinstance(value, ast.Name) and hops > 0:
            sites = _unique(
                [
                    one.site
                    for one in pathcond.bindings(self.func, value.id)
                    if not one.nested
                ]
            )
            found: list[_Origin] = []
            for site in reaching(value, [self.func.args, *sites]):
                if site is self.func.args:
                    return None
                held = self.bound(value.id, site, hops - 1)
                if held is None:
                    return None
                found += held
            return found or None
        return None

    def bound(
        self, name: str, site: ast.AST, hops: int = _HOPS
    ) -> Optional[list[_Origin]]:
        """What the binding *site* gives *name* of a fetch: a ``for`` target, one row of the rows each pass, an
        assignment its value -- at the name's place where it unpacks."""
        held = [
            one
            for one in pathcond.bindings(self.func, name)
            if one.site is site and not one.nested
        ]
        if len(held) != 1:
            return None
        binding = held[0]
        if isinstance(site, (ast.For, ast.AsyncFor)):
            rows = self.origins(site.iter, hops)
            if rows is None or any(one.shape != callfacts.ROWS for one in rows):
                return None
            each = [one._replace(shape=callfacts.ROW, pick=PICK_EACH) for one in rows]
        elif (
            isinstance(site, (ast.Assign, ast.AnnAssign)) and binding.value is not None
        ):
            found = self.origins(binding.value, hops)
            if found is None:
                return None
            each = found
        else:
            return None
        if binding.unpacked:
            return None if binding.slot is None else _at(each, binding.slot)
        return each

    def runs(self) -> tuple[list[sql.Execution], dict[int, tuple[int, int]]]:
        if self._runs is None:
            self._runs = sql.executions(self.func)
            self._regions = sql.regions(self.func, self._runs)
        return self._runs, self._regions

    def reads(self, origin: _Origin, attribute: Optional[str]) -> tuple[RowRead, ...]:
        """Each text of the run whose region holds *origin*'s fetch, with its row tests and the value's column."""
        runs, regions = self.runs()
        line = origin.fetch.lineno
        found = []
        for run in runs:
            region = regions.get(run.call.lineno)
            if region is None or not region[0] <= line <= region[1]:
                continue
            if attribute is not None:
                column = attribute
            else:
                items = sqltext.select_items(run.reading) or ()
                at = origin.place
                column = (
                    items[at].column
                    if origin.shape == _COLUMN
                    and at is not None
                    and at < len(items)
                    and items[at].kind == sqltext.ITEM_COLUMN
                    else ""
                )
            found.append(RowRead(run, self._tests(run), column))
        return tuple(found)

    def _tests(self, run: sql.Execution) -> tuple[RowTest, ...]:
        found = []
        for test in sqltext.row_tests(run.reading):
            binds = []
            for side in test.sides:
                for key in side.binds:
                    values = sql.bound_values(
                        self.func,
                        run.varmap or "",
                        key,
                        run.window,
                        across=True,
                        at=run,
                    )
                    if not run.varmap:
                        values = values + sql.inline_binds(run.call, key)
                    kept = tuple(
                        one
                        for one in values
                        if pathcond.enclosing_function(one.value) is self.func
                    )
                    binds.append((key, kept))
            found.append(RowTest(test, tuple(binds)))
        return tuple(found)


def _at(origins: list[_Origin], index: int) -> Optional[list[_Origin]]:
    """*origins* at a constant *index*: an element of the rows is a row, of a row a column."""
    found = []
    for one in origins:
        if one.shape == callfacts.ROWS:
            found.append(one._replace(shape=callfacts.ROW, pick=PICK_AT))
        elif one.shape == callfacts.ROW:
            found.append(one._replace(shape=_COLUMN, place=index))
        else:
            return None
    return found


def _row_of(site: ast.AST) -> ast.expr:
    """What a local's binding takes its row from: a ``for`` 's iterable, an assignment's value."""
    if isinstance(site, (ast.For, ast.AsyncFor)):
        return site.iter
    value = pathcond.written_value(site)
    assert value is not None
    return value


def _call_at(value: ast.expr) -> tuple[Optional[ast.Call], Optional[int]]:
    """*value* as a call, or its element at a constant index: the call and the index; ``(None, None)`` else."""
    if isinstance(value, ast.Call):
        return value, None
    if (
        isinstance(value, ast.Subscript)
        and isinstance(value.value, ast.Call)
        and isinstance(value.slice, ast.Constant)
        and type(value.slice.value) is int
    ):
        return value.value, value.slice.value
    return None, None


def _last_downs(route: tuple[Hop, ...]) -> tuple[Hop, ...]:
    """The calls down the route has crossed since it last went up: the ones a constant is handed down through."""
    found: list[Hop] = []
    for hop in reversed(route):
        if hop.way != HOP_DOWN:
            break
        found.append(hop)
    return tuple(reversed(found))


def _yields(func: ast.AST) -> bool:
    """Whether *func*'s own body yields: a call to it makes a generator."""
    return any(isinstance(node, (ast.Yield, ast.YieldFrom)) for node in _own(func))


def _returns(
    func: ast.AST, index: Optional[int], values: dict[str, object]
) -> list[tuple[Optional[ast.expr], ast.AST]]:
    """What *func* returns where *values* do not rule the ``return`` out -- whole (*index* ``None``), its element at
    a constant *index*, or any element (:data:`_EACH`) -- each with its statement; ``None`` for a value not told."""
    found: list[tuple[Optional[ast.expr], ast.AST]] = []
    for node in _own(func):
        if not isinstance(node, ast.Return):
            continue
        if values and pathcond.ruled_out_by(node, func, values):
            continue
        value = node.value
        if value is None:
            nothing = ast.Constant(value=None)
            ast.copy_location(nothing, node)
            nothing.parent = node  # type: ignore[attr-defined]
            found.append((nothing, node))
        elif index is None:
            found.append((value, node))
        elif isinstance(value, ast.Tuple) and index != _EACH:
            elements = value.elts
            told = index < len(elements) and not any(
                isinstance(one, ast.Starred) for one in elements
            )
            found.append((elements[index] if told else None, node))
        elif isinstance(value, (ast.Tuple, ast.List)):
            if any(isinstance(one, ast.Starred) for one in value.elts):
                found.append((None, node))
            else:
                found += [(one, node) for one in value.elts]
        elif isinstance(value, ast.Name):
            found += _elements(func, value, values)
        else:
            found.append((None, node))
    return found


def _elements(
    func: ast.AST, name: ast.Name, values: dict[str, object]
) -> list[tuple[Optional[ast.expr], ast.AST]]:
    """Each element of the list *name* holds where it is returned: the displays binding it and what is appended or
    inserted into it, the lines *values* rule out left out; ``None`` for a list not built that way."""
    statement = _statement(name)
    sites = _unique(
        [one.site for one in pathcond.bindings(func, name.id) if not one.nested]
    )
    found: list[tuple[Optional[ast.expr], ast.AST]] = []
    for site in reaching(name, [func.args, *sites]):  # type: ignore[attr-defined]
        value = (
            pathcond.written_value(site)
            if isinstance(site, (ast.Assign, ast.AnnAssign))
            else None
        )
        if not isinstance(value, (ast.List, ast.Tuple)) or any(
            isinstance(one, ast.Starred) for one in value.elts
        ):
            return [(None, statement)]
        found += [(one, site) for one in value.elts]
    for node in _own(func):
        if isinstance(node, ast.AugAssign) and _on(node.target, name.id, ()):
            return [(None, node)]
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and _on(node.func.value, name.id, ())
        ):
            continue
        at = _statement(node)
        if node.func.attr in ("append", "insert") and node.args:
            if not (values and pathcond.ruled_out_by(at, func, values)):
                found.append((node.args[-1], at))
        elif node.func.attr in ("extend", "__setitem__", "__iadd__"):
            return [(None, at)]
    return found


def _none_ruled_out(
    func: ast.AST,
    at: ast.AST,
    name: str,
    attributes: tuple[str, ...],
    rest: int,
    write: ast.AST,
) -> bool:
    """Whether a test on *at* 's way past *write* does not run where the part of the chain a ``None`` is the value
    of -- *name* and *attributes* but the last *rest* -- is ``None``."""
    if rest > len(attributes):
        return False
    holder = (name, attributes[: len(attributes) - rest])
    entry = write is getattr(func, "args", None)
    for guard in pathcond.guards(at, func):
        if not entry and guard.at.lineno <= getattr(write, "lineno", 0):
            continue
        if _not_none(guard.test, guard.runs_on, holder):
            return True
    return False


def _not_none(
    test: ast.expr, runs_on: bool, holder: tuple[str, tuple[str, ...]]
) -> bool:
    """Whether *test* coming out *runs_on* says *holder* is not ``None``: ``X``, ``X is not None``, ``X != None``
    and ``isinstance(X, ...)`` true, ``X is None`` and ``X == None`` false, through ``not``, an ``and`` true and an
    ``or`` false."""
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        return _not_none(test.operand, not runs_on, holder)
    if isinstance(test, ast.BoolOp):
        if isinstance(test.op, ast.And) and runs_on:
            return any(_not_none(one, True, holder) for one in test.values)
        if isinstance(test.op, ast.Or) and not runs_on:
            return any(_not_none(one, False, holder) for one in test.values)
        return False
    if _chain(test) == holder:
        return runs_on
    if (
        isinstance(test, ast.Call)
        and isinstance(test.func, ast.Name)
        and test.func.id == "isinstance"
        and test.args
        and _chain(test.args[0]) == holder
    ):
        return runs_on
    if (
        isinstance(test, ast.Compare)
        and len(test.ops) == 1
        and _chain(test.left) == holder
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value is None
    ):
        if isinstance(test.ops[0], (ast.IsNot, ast.NotEq)):
            return runs_on
        if isinstance(test.ops[0], (ast.Is, ast.Eq)):
            return not runs_on
    return False


def _own(func: ast.AST) -> Iterator[ast.AST]:
    """``ast.walk`` below *func* that does not go into a nested function, class or lambda."""
    stack = list(ast.iter_child_nodes(func))
    while stack:
        one = stack.pop()
        yield one
        if not isinstance(one, _NESTED):
            stack.extend(ast.iter_child_nodes(one))


def _chain(node: ast.AST) -> Optional[tuple[str, tuple[str, ...]]]:
    """``("job", ("jobStatus",))`` for ``job.jobStatus``, ``("job", ())`` for ``job``; ``None`` for anything else."""
    attributes: list[str] = []
    while isinstance(node, ast.Attribute):
        attributes.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    return node.id, tuple(reversed(attributes))


def _on(node: ast.AST, name: str, attributes: Sequence[str]) -> bool:
    """Whether *node* is the chain *name* followed by *attributes*."""
    return _chain(node) == (name, tuple(attributes))


def _written(target: ast.expr) -> Iterator[ast.expr]:
    """What a target writes: itself, or each element of a tuple or list, ``*`` taken off."""
    if isinstance(target, (ast.Tuple, ast.List)):
        for one in target.elts:
            yield from _written(one)
    elif isinstance(target, ast.Starred):
        yield from _written(target.value)
    else:
        yield target


def _stores(func: ast.AST, name: str, attributes: tuple[str, ...]) -> list[ast.stmt]:
    """The statements storing the chain or a part of it before the column, ``+=`` among them."""
    parts = [attributes[:end] for end in range(1, len(attributes) + 1)]
    found = []
    for node in _own(func):
        targets = (
            [node.target]
            if isinstance(node, ast.AugAssign)
            else pathcond.targets_of(node)
        )
        if any(
            _on(one, name, part)
            for target in targets
            for one in _written(target)
            for part in parts
        ):
            found.append(node)
    return _sorted(found)


def _deletes(func: ast.AST, name: str, attributes: tuple[str, ...]) -> list[ast.stmt]:
    """The ``del`` s of the name, or of the chain or a part of it."""
    parts = [attributes[:end] for end in range(len(attributes) + 1)]
    found = [
        node
        for node in _own(func)
        if isinstance(node, ast.Delete)
        and any(_on(one, name, part) for one in node.targets for part in parts)
    ]
    return _sorted(found)


def _row_readers(func: ast.AST, name: str, holder: tuple[str, ...]) -> list[ast.stmt]:
    """The statements calling a row-reading method of the object *name* followed by *holder* holds."""
    found = [
        node
        for node in _own(func)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr in callfacts.ROW_READ
        and _on(node.value.func.value, name, holder)
    ]
    return _sorted(found)


def _statement(node: ast.AST) -> ast.stmt:
    current: Optional[ast.AST] = node
    while current is not None and not isinstance(current, ast.stmt):
        current = getattr(current, "parent", None)
    assert isinstance(current, ast.stmt)
    return current


def _sorted(nodes: list) -> list:
    return sorted(nodes, key=lambda node: (node.lineno, node.col_offset))


def _unique(nodes: list) -> list:
    seen: set[int] = set()
    found = []
    for node in nodes:
        if id(node) not in seen:
            seen.add(id(node))
            found.append(node)
    return found
