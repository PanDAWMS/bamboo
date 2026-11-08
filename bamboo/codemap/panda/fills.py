"""Which writes of a value a function reads filled it from a row the function fetched: one question on its flow.

A reader of a run asks it of a statement and a value the statement reads -- a local, ``oldJobStatus``, or an
attribute chain on a name, ``job.jobStatus`` --: which writes of the value reach the read, and of them which filled
it from a row of a statement the function ran, with the statement and what its rows were asked by.  ``killJob``
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
* **The others**: the entry, a binding not from a row, a store, a ``del``, a row-reading call whose row is not
  followed to a fetch.  A reader holding the rows of the fills alone reads the value; one with any other does not.
* **Between a fill and the read** a call may change the object in place: each change of the chain's first
  attribute or of every attribute that :func:`changes.object_changes` names is kept on the fill.  The function's own
  stores are among the others already.

Not read: a value filled in another function (a parameter, a field of ``self`` a caller filled, an object a call
hands back), rows a runner other than the cursor hands back (``querySQLS``), and a row through a name one of whose
reaching bindings is not a fetch's.  The tree needs parent links (:func:`pathcond.attach_parents`).
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING, Iterator, NamedTuple, Optional, Sequence

from bamboo.codemap import callfacts
from bamboo.codemap.models import CHANGE_STORE
from bamboo.codemap.panda import changes, pathcond, sql, sqltext
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

_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)
_NESTED = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
#: The most names a row is followed through: ``rows = res`` and a ``for`` over it is two.
_HOPS = 4


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


class Fill(NamedTuple):
    """A write filling the value from a row: the statement, the row it takes, the fetch the row comes from and how
    it is picked of it, each text of the statement run, and the in-place changes between it and the read."""

    site: ast.stmt
    row: ast.expr
    fetch: ast.Call
    pick: str
    reads: tuple[RowRead, ...]
    changes: tuple[changes.ObjectChange, ...] = ()


class Other(NamedTuple):
    """A write reaching the read that is no fill: the statement, or the function's ``args`` for its entry."""

    site: ast.AST
    kind: str


class RowHeld(NamedTuple):
    """The writes reaching a read (:func:`row_fills`), in source order: the fills and the others."""

    fills: tuple[Fill, ...]
    others: tuple[Other, ...]


class _Origin(NamedTuple):
    """Where a value comes from of a fetch: *shape* :data:`callfacts.ROWS` or :data:`callfacts.ROW`, or ``column``
    at *place*; *pick* how its row is taken."""

    fetch: ast.Call
    shape: str
    pick: str
    place: Optional[int] = None


_COLUMN = "column"


def row_fills(
    calls: "Calls",
    file: str,
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    read: ast.expr,
    bound: tuple = (),
) -> RowHeld:
    """The writes of what *read* -- a name, or an attribute chain on one, in *func* of *file* -- holds that can reach
    it, each a :class:`Fill` from a row the function fetched or an :class:`Other`.  *bound*: the classes ``self``
    can be in *func* (:attr:`Target.bound`), for the in-place changes."""
    if getattr(read, "parent", None) is None:
        raise ValueError(
            f"row_fills needs parent links on the tree holding line {getattr(read, 'lineno', '?')}"
        )
    chain = _chain(read)
    if chain is None:
        raise ValueError(
            f"row_fills reads a name or an attribute chain on one, not {ast.unparse(read)}"
        )
    name, attributes = chain
    binds = _unique(
        [one.site for one in pathcond.bindings(func, name) if not one.nested]
    )
    deletes = _deletes(func, name, attributes)
    stores = _stores(func, name, attributes)
    readers = _row_readers(func, name, attributes[:-1]) if attributes else []
    reader = _Reader(func)
    statement = _statement(read)
    own = {(file, one.lineno) for one in [*stores, *deletes]}
    fills: list[Fill] = []
    others: list[Other] = []
    for site in reaching(read, [func.args, *binds, *stores, *deletes, *readers]):
        if site is func.args:
            others.append(Other(site, OTHER_ENTRY))
        elif any(site is one for one in stores):
            others.append(Other(site, OTHER_STORE))
        elif any(site is one for one in deletes):
            others.append(Other(site, OTHER_DELETE))
        elif any(site is one for one in readers):
            assert isinstance(site, ast.Expr) and isinstance(site.value, ast.Call)
            call = site.value
            origins = reader.origins(call.args[0]) if call.args else None
            if not origins or any(one.shape != callfacts.ROW for one in origins):
                others.append(Other(site, OTHER_ROW_UNREAD))
                continue
            moved = tuple(
                change
                for change in changes.object_changes(
                    calls, file, func, name, site, statement, bound
                )
                if change.key in (f".{attributes[0]}", changes.EVERY)
                and not (
                    change.kind == CHANGE_STORE and (change.file, change.line) in own
                )
            )
            fills += [
                Fill(
                    site,
                    call.args[0],
                    one.fetch,
                    one.pick,
                    reader.reads(one, attributes[-1]),
                    moved,
                )
                for one in origins
            ]
        elif attributes:
            others.append(Other(site, OTHER_BINDING))
        else:
            origins = reader.bound(name, site)
            if not origins or any(one.shape == callfacts.ROWS for one in origins):
                others.append(Other(site, OTHER_BINDING))
                continue
            assert isinstance(site, ast.stmt)
            fills += [
                Fill(site, _row_of(site), one.fetch, one.pick, reader.reads(one, None))
                for one in origins
            ]
    return RowHeld(tuple(fills), tuple(others))


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
