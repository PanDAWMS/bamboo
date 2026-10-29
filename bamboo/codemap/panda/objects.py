"""Where a function puts the objects it names, and which of their columns it assigns.

bamboo-c1's rows-at-time question (to-map-8, ask 1) is whether a run can
change a column of the row it holds.  Asked of a closure the answer is no
answer: the adder's ``process_job_report`` reaches 426 functions with 239
holes, and a hole may be where the column is set.  Asked of the object the
run holds, only the holes that object reaches count -- so the question is
followed with the object, through the names that hold it, the calls it is
handed to and the containers it is put in (:func:`bamboo.codemap.objects.follow`).

This module records, once per function, what that following reads:

* :class:`~bamboo.codemap.models.PathAssign` -- a column of a spec assigned
  on a path (``self.job.jobStatus = "failed"``); a write whose name is
  computed, with the role :mod:`specwrites` gives it;
* :class:`~bamboo.codemap.models.PathBinding` -- a name holding what a path
  holds: ``job = self.job``, ``for job in jobs``, ``jobs.append(job)``,
  ``self.job = job``, ``x = f(...)``, ``return job``;
* :class:`~bamboo.codemap.models.CallBinding` -- a path a call hands on and
  the parameter that takes it, the receiver as ``self``; a hole's own.

A path is rooted at a name of the function -- a parameter, ``self``, a local
-- so a module's logger or a class read for its methods is not one --, and
is kept however long it is: a statement's paths are as long as its source,
and how far an object is followed is follow's to say
(:data:`bamboo.codemap.objects.MAX_PARTS`).  A local name bound at two or
more statements is a name for each, ``res@96`` and ``res@472`` -- a
parameter's entry is one, under its bare name --, and a use is the ones
that reach it (:class:`_Versions`); a field and what a
container holds are read without the order of the statements -- bound to a
path anywhere, held everywhere in the function --, which can only say an
object reaches more than it does.

**One rule** (:class:`_Values`): a value carries what its parts carry, each
at a position inside it, and a target takes, at each position, what the
value has there.  A position is ``[]`` -- an element, where it is not known
--, ``[i]`` -- a tuple's i-th --, or ``['k']`` -- the value under a literal
key.  So ``status, job = self.make(spec)`` binds ``job`` to what is at
``[1]`` of what ``make`` returns, ``{":PandaID": job.PandaID}`` puts the key
at ``[':PandaID']``, ``pairs.append((job, n))`` then ``for first, second in
pairs`` makes ``first`` the job and not ``second``.  The parts read: a
tuple, list, set or dict literal, a starred element, an if-else, ``or`` /
``and``, ``+`` (the terms' elements), a comprehension, a slice (the
container's elements: a slice moves positions), a subscript or an attribute
of any of them, an await, a walrus, a call's result; the targets of an
assignment (a tuple target at any depth, ``*rest``), an augmented
assignment, a loop, a return and a yield, a call's arguments and receiver.
Only a tuple makes ``[i]`` -- it cannot be reordered, and whatever else puts
into a container puts at ``[]``, which every position matches -- and a
literal int written to (``x[0] = v``) is ``[]``.

A call that lands in a function of the corpus is that call: its result is
bound with the functions it lands in (``PathBinding.to``), not with the
line, and a result no name takes -- ``self.handle(self.wrap(job))``,
``self.mapper().pick(job)`` -- is held by a name of the call's own,
``<result@line:col>``.  A builtin (``list``, ``sorted``, ``enumerate``,
``zip``, ``dict`` ...) or a builtin container's method (``append``,
``extend``, ``update``, ``setdefault``, ``pop``, ``get``, ``values``,
``items`` ...) is read for what it puts in or hands back only where the
call lands in no function of the corpus: ``self.proxyPool.get()`` runs
``DBProxyPool.get``.

Not read, and so an object can reach more than this says: a ``with``'s
target (no item of pin k's is a path), a call inside a lambda's body, a
nested function's use of the names of the one around it,
``setattr(job, "jobStatus", v)`` with the name spelled, and the values of
an outside call (``copy.copy(job)`` is a new object).  Iterating a dict
gives its keys, but ``for k in d`` is read as d's values (more, not less).

:meth:`_Reader.writes` adds the statements that write rows
(:class:`~bamboo.codemap.models.DbWrite`): each written column's value as a
path or another value, the keys of the row, and whether the bind map is the
whole of one object (``job.valuesMap()``) -- read at each call that runs the
statement, the calls with the same values and keys one write.  A spec's
reset of its changed attributes (``resetChangedAttr``, ``resetChangedList``)
is read with them, not as an assignment, and :func:`order.attach` reads the
order of each function's lines and what its resets clear.
"""

from __future__ import annotations

import ast
import re
from typing import NamedTuple, Optional

from bamboo.codemap import callfacts, reach
from bamboo.codemap import calls as C
from bamboo.codemap.models import (
    BIND_ALIAS,
    BIND_ELEMENT,
    BIND_FILL,
    BIND_RESULT,
    BIND_RETURN,
    MAP_KEY,
    CallBinding,
    DbWrite,
    MapFragment,
    PathAssign,
    PathBinding,
    SourceModule,
    WrittenValue,
)
from bamboo.codemap.objects import RETURN, table_classes
from bamboo.codemap.panda import order, pathcond, specwrites, sql, sqltext
from bamboo.codemap.panda.recognizers import progress

#: The classes whose columns an assignment is kept for (tmaeno, 10-18: D1).
COLUMN_CLASSES = ("JobSpec", "JediTaskSpec", "FileSpec")

#: How :func:`callfacts.meaning` reads a call that puts into its object (:meth:`_Values._fill`).
_FILLING = (callfacts.LIBRARY, callfacts.BUILTIN, callfacts.ASSUMED)
#: A literal key a path keeps (8,937 of pin k's 8,950); another reads as ``[]``.
_KEY = re.compile(r"[A-Za-z0-9_:\-]*")


def is_parameter(func: ast.AST, name: str) -> bool:
    """Whether *name* is one of *func*'s parameters."""
    arguments = getattr(func, "args", None)
    if arguments is None:
        return False
    held = [
        *arguments.posonlyargs,
        *arguments.args,
        *arguments.kwonlyargs,
        arguments.vararg,
        arguments.kwarg,
    ]
    return any(one is not None and one.arg == name for one in held)


#: Why a hole the reach record does not hold is one.
HANDED_WITH_A_CALLABLE = "handed to code outside the corpus together with a function of the corpus, which it may call with it"


def attach(
    fragment: MapFragment, modules: list[SourceModule], calls: C.Calls
) -> tuple[int, int, int, int]:
    """Read every function's object records into its :class:`ReachNode`; how many assigns, bindings, call bindings and writes.

    Inside the build's reading context (:func:`sql.across_calls`), so the
    statements are the ones every slice read.  Then the order of their lines
    and the resets between them (:func:`order.attach`).
    """
    reader = _Reader(modules, calls, table_classes(fragment.entities))
    rows = {row.owner: row for row in fragment.reach_graph}
    counts = [0, 0, 0, 0]
    functions: dict[str, ast.AST] = {}
    resets: dict[str, list[order.Reset]] = {}
    for file, func, owner in reader.names.functions:
        row = rows.get(owner)
        if row is None:
            continue
        functions[owner] = func
        row.assigns, row.bindings, row.call_bindings, resets[owner] = reader.read(
            file, func, row
        )
        row.writes = reader.writes(func)
        counts[0] += len(row.assigns)
        counts[1] += len(row.bindings)
        counts[2] += len(row.call_bindings)
        counts[3] += len(row.writes)
    order.attach(functions, rows, resets)
    return counts[0], counts[1], counts[2], counts[3]


class _Reader:
    def __init__(
        self,
        modules: list[SourceModule],
        calls: C.Calls,
        tables: Optional[dict[str, Optional[str]]] = None,
    ):
        self.calls = calls
        self.types = calls.types
        self.names = reach._Names(calls.types)
        #: What a mapping holds, read from every name holding it (:meth:`reach._Receivers.keys`).
        self.receivers = reach._Receivers(calls, None)
        declared = progress.spec_attributes(modules)
        self.columns: frozenset[str] = frozenset().union(
            *(declared.get(name, set()) for name in COLUMN_CLASSES)
        )
        #: Each table, folded to lower case, and the spec class it holds (:func:`table_classes`).
        self.tables = tables or {}
        #: Each spec class's attributes by their lower-case spelling: a statement's column, case aside.
        self.attributes = {
            cls: {name.lower(): name for name in names}
            for cls, names in declared.items()
        }

    def read(
        self, file: str, func: ast.AST, row
    ) -> tuple[
        list[PathAssign], list[PathBinding], list[CallBinding], list[order.Reset]
    ]:
        values = _Values(self, file, func, row)
        for node in C.own_nodes(func):
            values.site(node)
        values.row_keys()
        for write in specwrites.writes_in(self.types, file, func):
            target = _path(_parsed(write.target), values.names)
            if target is None:
                continue
            attrs = (
                sorted(write.names & self.columns) if write.names is not None else ["*"]
            )
            if not attrs:
                continue
            values.assigns.setdefault(
                (write.line, target, write.role, tuple(attrs)),
                PathAssign(
                    line=write.line,
                    path=target,
                    attrs=attrs,
                    how=write.role or "no_role",
                ),
            )
        key = lambda one: (one.line, *(str(v) for v in one.model_dump().values()))  # noqa: E731
        return (
            sorted(values.assigns.values(), key=key),
            sorted(values.bindings.values(), key=key),
            sorted(values.handed.values(), key=key),
            list(values.resets.values()),
        )

    def writes(self, func: ast.AST) -> list[DbWrite]:
        """Each statement *func* runs that writes rows, with where its values and keys come from.

        A statement run from two or more places (:attr:`sql.Execution.calls`)
        is read at each: the calls whose values and keys are the same are one
        write, with their lines, and the others a write each.  Read at all of
        them at once, one call's keys stood beside another's values --
        ``restartTasksForContentsUpdate_JEDI`` runs one ``UPDATE`` in a loop
        over one query's tasks and again in a loop over another's, and the
        write named both loops' tasks as the row of each.
        """
        names = _local_names(func)
        versions = _Versions(func, names)
        found: list[DbWrite] = []
        try:
            runs = sql.executions(func)  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001 -- the reach row has the hole
            return found
        for run in runs:
            alike: dict[str, DbWrite] = {}
            for call in run.run_at:
                for one in self._run_writes(
                    func, run._replace(calls=(call,)), versions
                ):
                    same = one.model_dump_json(exclude={"lines"})
                    if same in alike:
                        alike[same].lines = sorted({*alike[same].lines, *one.lines})
                    else:
                        alike[same] = one
            found.extend(alike.values())
        # One statement read in several variants of its text says the same write more than once.
        unique: dict[str, DbWrite] = {}
        for one in found:
            unique.setdefault(one.model_dump_json(), one)
        return list(unique.values())

    def _run_writes(
        self, func: ast.AST, run: sql.Execution, versions: "_Versions"
    ) -> list[DbWrite]:
        """The writes of *run*'s statement, read at its calls."""
        found: list[DbWrite] = []
        text = run.reading
        try:
            read = sqltext.writes(text)
            tables = [(table, "insert") for table in sqltext.creates(text)] + [
                (table, "update") for table in sqltext.updates(text)
            ]
        except Exception:  # noqa: BLE001
            return found
        if not read and not tables:
            return found
        whole, changed, sequenced = _whole_row(func, run, versions)
        keyed = [
            self._keyed_columns(func, run, one.column_holes, versions) for one in read
        ]
        spelt = {item for _values, items in keyed for item in items}
        # A SET item ``{}=:{}`` read as a column for each key of the mapping it walks is not a column left unread.
        unread = any(
            stop.reason != sqltext.SET_COLUMN_IS_A_HOLE or stop.source not in spelt
            for stop in run.unread_columns
        )
        # A row is new only where an INSERT makes it; an UPDATE whose bind map a sequence filled on some way
        # (updateJobStatus 242 after the corrupted files' valuesMap(useSeq=True)) still writes a row that was there.
        new_row = sequenced or any(
            "useSeq=True" in (hole.source or "").replace(" ", "")
            for _offset, hole in run.holes
        )
        lines = sorted({call.lineno for call in run.run_at})
        seen: set[str] = set()
        for one, (by_keys, _items) in zip(read, keyed, strict=True):
            seen.add(one.table.lower())
            found.append(
                DbWrite(
                    lines=lines,
                    table=one.table,
                    kind=one.kind,
                    columns=[
                        value
                        for column, given in sorted(one.columns.items())
                        for value in self._values(
                            func, run, column, given, whole, versions
                        )
                    ]
                    + by_keys,
                    whole=whole,
                    changed=changed,
                    unread=unread,
                    keys=[
                        value
                        for column, bind in sorted(one.keys.items())
                        for value in self._values(
                            func, run, column, sqltext.classify(bind), "", versions
                        )
                    ]
                    + _map_keys(func, run, one.key_holes, versions),
                    new_row=one.kind == "insert"
                    and (
                        new_row
                        or any(
                            "nextval" in given.text.lower()
                            for given in one.columns.values()
                        )
                    ),
                )
            )
        for table, kind in tables:
            bare = table.rpartition(".")[2]
            # Nothing named, but rows written: a whole row of {}, or a text no tree reads -- each said by a stop on
            # the run.  A build left no stop writes nothing the call's other builds do not (sql._executions_read): a
            # SET {} in which no attribute changed.  A SET only of names made at run time is read above, with its
            # WHERE's keys (sqltext.writes).
            if bare.lower() not in seen and unread:
                seen.add(bare.lower())
                found.append(
                    DbWrite(
                        lines=lines,
                        table=bare,
                        kind=kind,
                        whole=whole,
                        changed=changed,
                        unread=True,
                        new_row=kind == "insert" and new_row,
                    )
                )
        return found

    def _keyed_columns(
        self, func: ast.AST, run: sql.Execution, column_holes, versions: "_Versions"
    ) -> tuple[list[WrittenValue], list[str]]:
        """The columns *run*'s ``SET`` items ``{}=:{}`` write, where a loop over a mapping's keys fills the holes with
        its key (:func:`sql.map_key`): one for each key the mapping can hold (:meth:`reach._Receivers.keys`) that
        the loop's ``continue`` tests let through (:func:`sql.kept_map_keys`), valued by the mapping at it -- and the
        items so read, as the statement spells them (``SET {}=:{}``).  ``updateJobStatus``' ``for key in
        list(param): ... sql1 += f",{key}=:{key}"; varMap[f":{key}"] = param[key]``, ``param`` filled by
        ``pilot_api.update_job``."""
        file = self.types.file_of(func)
        holes = run.reading_holes
        found: dict[str, WrittenValue] = {}
        items: list[str] = []
        for column, bind, numbers in column_holes:
            if file is None or not all(
                0 < number <= len(holes) for number in (column, *numbers)
            ):
                continue
            hole = holes[column - 1]
            try:
                mapping = sql.map_key(
                    func,  # type: ignore[arg-type]
                    run,
                    hole,
                    bind,
                    tuple(holes[number - 1] for number in numbers),
                )
                keys = (
                    self.receivers.keys(file, func, mapping.id)
                    if isinstance(mapping, ast.Name)
                    else None
                )
                kept = (
                    sql.kept_map_keys(hole, sorted(keys)) if keys is not None else None
                )
            except Exception:  # noqa: BLE001 -- not read: the item stays a hole
                kept = None
            if kept is None:
                continue
            items.append(f"SET {{}}={bind}")
            for key in kept:
                value = ast.Subscript(
                    value=mapping, slice=ast.Constant(value=key), ctx=ast.Load()
                )
                held = versions.paths(value)
                for one in [WrittenValue(column=key, path=path) for path in held] or [
                    WrittenValue(column=key, value=_cut(value))
                ]:
                    found.setdefault(one.model_dump_json(), one)
        return list(found.values()), items

    def _values(
        self, func, run, column: str, given, whole: str, versions: "_Versions"
    ) -> list[WrittenValue]:
        """Where *column*'s value comes from in *run*: a path, another value, or why neither is read."""
        if given.kind != "bind":
            return [
                WrittenValue(column=column, value=f"{given.kind}:{given.text}"[:80])
            ]
        if whole:
            return [
                WrittenValue(column=column, path=f"{whole}.{given.text.lstrip(':')}")
            ]
        if not run.varmap:
            # ``execute(sql, {":PandaID": panda_id})``: the map is written into the call.
            binds = [
                bind
                for call in run.run_at
                for bind in sql.inline_binds(call, given.text)
            ]
            if not binds:
                return [WrittenValue(column=column, why=_NO_MAP)]
        else:
            try:
                binds = sql.bound_values(func, run.varmap, given.text, at=run)  # type: ignore[arg-type]
            except Exception:  # noqa: BLE001
                binds = []
        if not binds:
            return [WrittenValue(column=column, why=_NOT_BOUND_HERE)]
        found = []
        for bind in binds:
            held = versions.paths(bind.value)
            if held:
                found += [WrittenValue(column=column, path=one) for one in held]
            else:
                found.append(WrittenValue(column=column, value=_cut(bind.value)))
        # One value read twice -- the same bind for two calls of the run -- is one value.
        values: dict[str, WrittenValue] = {}
        for one in found:
            values.setdefault(one.model_dump_json(), one)
        return list(values.values())

    def _hands_a_callable(self, file: str, func: ast.AST, call: ast.Call) -> bool:
        for value in reach._handed_values(call):
            try:
                if self.calls.named_callables(file, func, value, ()):
                    return True
            except Exception:  # noqa: BLE001
                return True
        return False


class _Carry(NamedTuple):
    """One thing a value holds: at ``at`` inside the value, what ``path`` holds -- or, for a call's result, what is
    at ``below`` in what ``call`` returns."""

    at: str
    path: str = ""
    call: Optional[ast.Call] = None
    below: str = ""


class Paths:
    """The paths the record names an expression by, in its function: what a reader matches against the record's."""

    def __init__(self) -> None:
        self._versions: dict[int, _Versions] = {}

    def of(self, func: ast.AST, expr: Optional[ast.expr]) -> frozenset[str]:
        if id(func) not in self._versions:
            self._versions[id(func)] = _Versions(func, _local_names(func))
        return frozenset(self._versions[id(func)].paths(expr))


class _Versions:
    """A local name bound at two or more statements, split by the binding that reaches each use.

    ``status, res = taskBuffer.querySQLS(...)`` at line 96 and again at 472
    are two lists, so the record names them ``res@96`` and ``res@472``: a
    binding is its statement's, and a use is each one that reaches it
    (:func:`pathcond.surviving_writes`, as the walk reads ``timeLimit``).  A
    parameter is bound on entry, by the call, and that binding keeps the bare
    name -- the one a caller hands the object in by --, so ``if days is None:
    days = 3`` leaves ``days`` and ``days@N`` after it.  ``self`` and a name
    bound once keep their name, and so does a name every binding of which is
    a str or a number by its spelling (:func:`_scalar`: ``sql = "..."``,
    ``sql += "..."``): it holds no path, and its versions would change no
    answer.
    """

    def __init__(self, func: ast.AST, names: frozenset[str]):
        self.names = names
        args = func.args  # type: ignore[attr-defined]
        given = {
            one.arg
            for one in [
                *args.posonlyargs,
                *args.args,
                *args.kwonlyargs,
                args.vararg,
                args.kwarg,
            ]
            if one is not None
        }
        self.sites: dict[str, list[ast.AST]] = {}
        for name in sorted(names - {"self", "cls"}):
            bound = [one for one in pathcond.bindings(func, name) if not one.nested]
            sites = {id(one.site): one.site for one in bound}
            if name in given:
                # Bound again: the parameter list is its binding on entry (reaching reads it so).
                if sites:
                    self.sites[name] = [args, *sites.values()]
                continue
            if all(
                one.kind in _SPELLED and not one.unpacked and _scalar(one.value)
                for one in bound
            ):
                continue
            if len({getattr(site, "lineno", 0) for site in sites.values()}) >= 2:
                self.sites[name] = list(sites.values())
        self._reaching: dict[tuple[str, int], list[str]] = {}

    def paths(self, expr: Optional[ast.expr], store: bool = False) -> list[str]:
        """*expr* as the paths it is (:func:`_path`), its root split by the bindings that reach it."""
        held = _path(expr, self.names, store)
        if held is None:
            return []
        base = expr
        while isinstance(base, (ast.Attribute, ast.Subscript)):
            base = base.value
        name = base.id  # type: ignore[union-attr]
        sites = self.sites.get(name)
        if not sites:
            return [held]
        rest = held[len(name) :]
        if base is expr and isinstance(base.ctx, ast.Store):  # type: ignore[union-attr]
            return [f"{name}@{self._site(sites, base)}{rest}"]
        return [one + rest for one in self.reaching(name, base)]

    def reaching(self, name: str, node: ast.AST) -> list[str]:
        key = (name, id(node))
        found = self._reaching.get(key)
        if found is None:
            try:
                kept = pathcond.surviving_writes(self.sites[name], node, as_run=True)
            except Exception:  # noqa: BLE001 -- not read: any of them
                kept = self.sites[name]
            found = sorted(
                {
                    name if isinstance(site, ast.arguments) else f"{name}@{site.lineno}"  # type: ignore[attr-defined]
                    for site in kept
                }
            ) or [name]
            self._reaching[key] = found
        return found

    @staticmethod
    def _site(sites: list[ast.AST], node: ast.AST) -> int:
        """The line of the binding statement *node*, a target, is in: the innermost holding it."""
        line = getattr(node, "lineno", 0)
        holding = [
            site.lineno  # type: ignore[attr-defined]
            for site in sites
            if not isinstance(site, ast.arguments)
            and site.lineno
            <= line
            <= (getattr(site, "end_lineno", None) or site.lineno)  # type: ignore[attr-defined]
        ]
        return max(holding, default=line)


class _Values:
    """One function's object records, read by one rule: a value carries what its parts carry, each at a position
    inside it, and a target takes, at each position, what the value has there (:meth:`carried`, :meth:`take`)."""

    def __init__(self, reader: "_Reader", file: str, func: ast.AST, row):
        self.reader = reader
        self.file = file
        self.func = func
        self.names = _local_names(func)
        self.versions = _Versions(func, self.names)
        self.assigns: dict[tuple, PathAssign] = {}
        self.bindings: dict[tuple, PathBinding] = {}
        self.handed: dict[tuple, CallBinding] = {}
        #: Its resets of a spec's changed attributes (:func:`order.reset_columns`): not assignments, so not in assigns.
        self.resets: dict[tuple, order.Reset] = {}
        self.hole_lines = {(hole.line, hole.callee) for hole in row.holes}
        self.database = set(row.database)
        self._placed: dict[int, C.Placed] = {}

    # -- what a value carries ------------------------------------------------------------------------------------

    def carried(self, expr: Optional[ast.expr]) -> list[_Carry]:
        """What *expr*'s value holds, each with where in the value."""
        if expr is None:
            return []
        held = self.versions.paths(expr)
        if held:
            return [_Carry("", one) for one in held]
        if isinstance(expr, ast.Tuple):
            found: list[_Carry] = []
            known = True  # a position after a starred element is not known
            for index, one in enumerate(expr.elts):
                if isinstance(one, ast.Starred):
                    known = False
                    found += _shift(_index(self.carried(one.value), "[]"), "[]")
                else:
                    found += _shift(self.carried(one), f"[{index}]" if known else "[]")
            return found
        if isinstance(expr, (ast.List, ast.Set)):
            return [carry for one in expr.elts for carry in self._element(one)]
        if isinstance(expr, ast.Dict):
            found = []
            for key, value in zip(expr.keys, expr.values, strict=True):
                if key is None:  # {**d}
                    found += _shift(_index(self.carried(value), "[]"), "[]")
                else:
                    found += _shift(self.carried(value), _key(key))
            return found
        if isinstance(expr, ast.IfExp):
            return self.carried(expr.body) + self.carried(expr.orelse)
        if isinstance(expr, ast.BoolOp):
            return [carry for one in expr.values for carry in self.carried(one)]
        if isinstance(expr, (ast.Await, ast.NamedExpr)):
            return self.carried(expr.value)
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
            # a + b holds the elements of a and of b; of a string or a number nothing is followed.
            return [
                carry
                for one in (expr.left, expr.right)
                for carry in _shift(_index(self.carried(one), "[]"), "[]")
            ]
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Mult):
            return [
                carry
                for one in (expr.left, expr.right)
                if isinstance(one, (ast.List, ast.Tuple))
                for carry in _shift(_index(self.carried(one), "[]"), "[]")
            ]
        if isinstance(expr, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            return _shift(self.carried(expr.elt), "[]")
        if isinstance(expr, ast.DictComp):
            return _shift(self.carried(expr.value), "[]")
        if isinstance(expr, ast.Subscript):
            if isinstance(expr.slice, ast.Slice):
                # a slice moves positions: the container's elements, at positions not known
                return _shift(_index(self.carried(expr.value), "[]"), "[]")
            return _index(self.carried(expr.value), _subscript(expr.slice))
        if isinstance(expr, ast.Attribute):
            return _attribute(self.carried(expr.value), expr.attr)
        if isinstance(expr, ast.Call):
            return self._result(expr)
        return []

    def _element(self, one: ast.expr) -> list[_Carry]:
        """What an element of a list or set literal puts at a position not known: ``[a, *b]``."""
        if isinstance(one, ast.Starred):
            return _shift(_index(self.carried(one.value), "[]"), "[]")
        return _shift(self.carried(one), "[]")

    def _result(self, call: ast.Call) -> list[_Carry]:
        """What a call's value holds: its own result, or for a builtin or a container's method that runs no
        function of the corpus, what it hands back of what it is given."""
        if self.placed(call).targets:
            return [_Carry("", call=call)]
        called = call.func
        args = call.args
        function = callfacts.builtin_function(
            self.reader.types, self.file, self.func, call
        )
        if function is not None:
            # A builtin function, by what its value holds of what it is handed (callfacts.Function).
            if function.gives == callfacts.COPY and args:
                return _shift(_index(self.carried(args[0]), "[]"), "[]")
            if function.gives == callfacts.ONE and args:
                elements, whole = callfacts.picked(function, call)
                return [
                    carry
                    for one in elements
                    for carry in _index(self.carried(one), "[]")
                ] + [carry for one in whole for carry in self.carried(one)]
            if function.gives == callfacts.NUMBERED and args:
                return _shift(_index(self.carried(args[0]), "[]"), "[][1]")
            if function.gives == callfacts.ZIPPED:
                return [
                    carry
                    for index, one in enumerate(args)
                    for carry in _shift(_index(self.carried(one), "[]"), f"[][{index}]")
                ]
            if function.gives == callfacts.DICT:
                # a mapping's values, or the second of each pair; and the keywords under their names
                found = [
                    carry
                    for one in args
                    for elements in (_index(self.carried(one), "[]"),)
                    for carry in _shift(elements + _index(elements, "[1]"), "[]")
                ]
                return found + [
                    carry
                    for keyword in call.keywords
                    if keyword.arg is not None
                    for carry in _shift(
                        self.carried(keyword.value), _named(keyword.arg)
                    )
                ]
        elif (
            isinstance(called, ast.Attribute)
            and (gives := callfacts.taken(called.attr)) is not None
        ):
            # A builtin container's method, by what it hands back of what is in it (callfacts.Give).
            receiver = self.carried(called.value)
            if (
                gives.part == callfacts.AT
                and gives.key is not None
                and len(args) > gives.key
            ):
                found = _index(receiver, _subscript(args[gives.key]))
                beside = (
                    [args[gives.default]]
                    if gives.default is not None and len(args) > gives.default
                    else []
                )
                return found + [carry for one in beside for carry in self.carried(one)]
            if gives.part == callfacts.AT and not args:
                return _index(receiver, "[]")
            if gives.part in (callfacts.VALUES, callfacts.SELF) and not args:
                return _shift(_index(receiver, "[]"), "[]")
            if gives.part == callfacts.ITEMS and not args:
                return _shift(_index(receiver, "[]"), "[][1]")
            if gives.part == callfacts.ITEM and not args:
                return _shift(_index(receiver, "[]"), "[1]")
        return [_Carry("", call=call)]

    # -- a row's key ---------------------------------------------------------------------------------------------

    def row_keys(self) -> None:
        """Bind each attribute a spec reads from a row to what the bind the row was asked by holds.

        A row a query returns holds, in each column its own ``WHERE`` sets
        equal to one bind (:func:`sqltext.row_tests`), what the bind was
        filled with.  So where ``O.pack(r)`` (:data:`specwrites.ROW_READ`)
        takes such a row, ``O.PandaID`` holds what ``varMap[":PandaID"]``
        does: ``peekJob``'s ``job.pack(res[0])`` after ``SELECT ... WHERE
        PandaID=:PandaID`` is the ``job.PandaID = pandaID`` its not-found
        branch writes by hand, and is bound as that is.  Without it the key a
        run is asked by reaches the object read by it only where such a line
        happens to be written, and a write naming the row by ``job.PandaID``
        reads as another row's.

        The row is a statement's where the values come, through the bindings,
        from a ``fetch*`` in the stretch of a run (:func:`sql.regions`) or from
        a run's own call.  Each variant of the statement that returns rows
        holds the test, the table aside -- ``peekJob`` runs one statement on
        ``jobsDefined4``, ``jobsActive4`` and ``jobsArchived4`` in turn --, and
        one that does not parse ties nothing.  A version of the values that
        comes from anywhere else (a parameter, a field, another call, a list
        a branch fills with objects) ties nothing; one only ever given a
        constant (``res = None`` before the ``try``) holds no row.  A
        ``fetch*`` on a cursor the corpus does not declare is bound to nothing
        in the record, and is read off the code.
        """
        packs: dict[tuple[int, str], dict[str, list[str]]] = {}
        for bound in self.handed.values():
            if (
                bound.to.rpartition(".")[2] not in specwrites.ROW_READ
                or not bound.parameter
                or bound.at
                or not bound.path
            ):
                continue
            side = "self" if bound.parameter == "self" else "row"
            packs.setdefault((bound.line, bound.to), {}).setdefault(side, []).append(
                bound.path
            )
        for (line, to), handed in sorted(packs.items()):
            cls = to.rpartition("::")[2].rpartition(".")[0]
            attributes = self.reader.attributes.get(cls)
            if not attributes or "self" not in handed or "row" not in handed:
                continue
            ties = self._row_ties(cls, attributes, handed["row"])
            for obj in handed["self"]:
                for attribute, path in sorted(ties):
                    self._bind(line, f"{obj}.{attribute}", path, BIND_ALIAS)

    def _row_ties(
        self, cls: str, attributes: dict[str, str], rows: list[str]
    ) -> set[tuple[str, str]]:
        """``(attribute, path)`` for each column of *cls* every row *rows* can hold is asked by, ``=`` one bind."""
        executions = sql.executions(self.func)  # type: ignore[arg-type]
        regions = sql.regions(self.func, executions)  # type: ignore[arg-type]
        fetched = {
            node.lineno
            for node in C.own_nodes(self.func)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in sql.FETCHES
        }
        ties: Optional[set[tuple[str, str]]] = None
        for path in rows:
            runs = self._runs_holding(path, regions, fetched)
            if runs is None:
                return set()
            for run in executions:
                if run.call.lineno not in runs:
                    continue
                returns = sqltext.returns_rows(run.reading)
                if returns is None:
                    return set()
                if returns:
                    found = self._tied(cls, attributes, run)
                    ties = found if ties is None else ties & found
        return ties or set()

    def _runs_holding(
        self, path: str, regions: dict[int, tuple[int, int]], fetched: set[int]
    ) -> Optional[set[int]]:
        """The call lines of the runs whose rows *path* holds, through the bindings; ``None`` where a way comes
        from anywhere else."""
        runs: set[int] = set()
        work, seen = [path], set()
        while work:
            root = _head(work.pop())
            if root in seen:
                continue
            seen.add(root)
            bound = [one for one in self.bindings.values() if _head(one.name) == root]
            if not bound:
                given = self._given_rows(root, regions, fetched)
                if given is None:
                    return None
                runs |= given
                continue
            for one in bound:
                if one.how == BIND_RESULT:
                    run = _run_at(one.line, regions, fetched)
                    if run is None:
                        return None
                    runs.add(run)
                elif one.how in (BIND_ALIAS, BIND_ELEMENT, BIND_FILL) and one.path:
                    work.append(one.path)
                else:
                    return None
        return runs

    def _given_rows(
        self, root: str, regions: dict[int, tuple[int, int]], fetched: set[int]
    ) -> Optional[set[int]]:
        """The runs whose rows the local *root* -- a name, or a name's version ``res@1826`` -- is given where the
        record binds it to nothing: a ``fetch*`` on a cursor outside the corpus, or a constant, which holds no row;
        ``None`` for a parameter, a field or anything else."""
        name, _, at = root.partition("@")
        if name in ("self", "cls") or (not at and is_parameter(self.func, name)):
            return None
        given = [
            node
            for node in C.own_nodes(self.func)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            and (not at or node.lineno == int(at))
            and any(
                isinstance(target, ast.Name) and target.id == name
                for target in pathcond.targets_of(node)
            )
        ]
        if not given:
            return None
        runs: set[int] = set()
        for node in given:
            if isinstance(node.value, ast.Constant):
                continue
            run = (
                _run_at(node.value.lineno, regions, fetched)
                if isinstance(node.value, ast.Call)
                else None
            )
            if run is None:
                return None
            runs.add(run)
        return runs

    def _tied(
        self, cls: str, attributes: dict[str, str], run: sql.Execution
    ) -> set[tuple[str, str]]:
        """``(attribute, path)`` for each test *run*'s rows hold that sets a column of *cls* equal to one bind."""
        found: set[tuple[str, str]] = set()
        for test in sqltext.row_tests(run.reading):
            if test.op != "=" or len(test.sides) != 1:
                continue
            side = test.sides[0]
            if side.kind != "bind" or len(side.binds) != 1:
                continue
            if self.reader.tables.get(test.table.rpartition(".")[2].lower()) != cls:
                continue
            attribute = attributes.get(test.column.lower())
            if attribute is None:
                continue
            key = side.binds[0]
            binds = sql.bound_values(
                self.func,
                run.varmap or "",
                key,
                run.window,
                across=True,
                at=run,  # type: ignore[arg-type]
            )
            if not run.varmap:
                binds = binds + sql.inline_binds(run.call, key)
            for bind in binds:
                if pathcond.enclosing_function(bind.value) is not self.func:
                    continue
                for path in self.versions.paths(
                    callfacts.unconverted(
                        self.reader.types, self.file, self.func, bind.value
                    )
                ):
                    found.add((attribute, path))
        return found

    # -- where it goes ---------------------------------------------------------------------------------------------

    def site(self, node: ast.AST) -> None:
        """Read the bindings and assignments one statement or expression of the function makes."""
        line = getattr(node, "lineno", 0)
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                self.assign(target, node.value, line)
        elif isinstance(node, ast.AugAssign):
            self._column(node.target, line, "+=")
            if isinstance(node.op, ast.Add):
                # t += v is t = t + v, t's elements and v's: v's are put in here, and t's come by the
                # binding before it, which reaches every use this one does (x += y kills nothing:
                # pathcond.surviving_writes).
                elements = _shift(_index(self.carried(node.value), "[]"), "[]")
                for base in self.target_paths(node.target):
                    self.put(line, base, elements, BIND_FILL)
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
            self.take(
                node.target,
                _index(self.carried(node.iter), "[]"),
                getattr(node, "lineno", getattr(node.iter, "lineno", 0)),
                BIND_ELEMENT,
            )
        elif isinstance(node, ast.NamedExpr):
            self.take(node.target, self.carried(node.value), line, BIND_ALIAS)
        elif isinstance(node, ast.Return) and node.value is not None:
            self.put(line, RETURN, self.carried(node.value), BIND_RETURN)
        elif isinstance(node, ast.Yield) and node.value is not None:
            self.put(line, RETURN + "[]", self.carried(node.value), BIND_RETURN)
        elif isinstance(node, ast.YieldFrom):
            self.put(
                line, RETURN + "[]", _index(self.carried(node.value), "[]"), BIND_RETURN
            )
        elif isinstance(node, ast.Call):
            self._call(node)

    def assign(self, target: ast.expr, value: ast.expr, line: int) -> None:
        """``target = value``: a tuple literal into a tuple target of its length element by element, else
        :meth:`take`."""
        if (
            isinstance(target, (ast.Tuple, ast.List))
            and isinstance(value, (ast.Tuple, ast.List))
            and len(target.elts) == len(value.elts)
            and not any(
                isinstance(one, ast.Starred) for one in [*target.elts, *value.elts]
            )
        ):
            for one, given in zip(target.elts, value.elts, strict=True):
                self.assign(one, given, line)
            return
        self.take(target, self.carried(value), line, BIND_ALIAS)

    def take(self, target: ast.expr, items: list[_Carry], line: int, how: str) -> None:
        """*target* takes, at each position, what *items* have there: ``a, b = f()`` takes [0] and [1]."""
        if isinstance(target, (ast.Tuple, ast.List)):
            starred = next(
                (
                    n
                    for n, one in enumerate(target.elts)
                    if isinstance(one, ast.Starred)
                ),
                None,
            )
            for index, one in enumerate(target.elts):
                if isinstance(one, ast.Starred):
                    self.take(one.value, _shift(_index(items, "[]"), "[]"), line, how)
                else:
                    known = starred is None or index < starred
                    self.take(
                        one, _index(items, f"[{index}]" if known else "[]"), line, how
                    )
            return
        if how == BIND_ALIAS:
            self._column(target, line, "=")
        for base in self.target_paths(target):
            self.put(line, base, items, how)

    def put(self, line: int, base: str, items: list[_Carry], how: str) -> None:
        """*base*, at each item's position, holds what the item says; an assignment that puts at a position fills."""
        for item in items:
            name = base + item.at
            label = BIND_FILL if how == BIND_ALIAS and item.at else how
            if item.call is not None:
                self._bind_result(name, item.call, item.below)
            else:
                self._bind(line, name, item.path, label)

    def _bind(self, line: int, name: str, held: str, how: str) -> None:
        if name == held and "[" not in name:
            return  # x = x; but x = list(x) says x's elements are at any position
        self.bindings.setdefault(
            (line, name, held, how),
            PathBinding(line=line, name=name, path=held, how=how),
        )

    def _bind_result(self, name: str, call: ast.Call, below: str) -> None:
        to = self.to_of(call)
        self.bindings.setdefault(
            (call.lineno, name, below, BIND_RESULT, tuple(to)),
            PathBinding(
                line=call.lineno, name=name, path=below, how=BIND_RESULT, to=to
            ),
        )

    def held_by(self, item: _Carry) -> Optional[str]:
        """The path that holds *item* in the function: its own, or the name of the call whose result it is in --
        where that call lands in the corpus."""
        if item.call is None:
            return item.path
        if not self.to_of(item.call):
            return None
        name = f"<result@{item.call.lineno}:{item.call.col_offset}>"
        self._bind_result(name, item.call, "")
        return name + item.below

    def target_paths(self, target: ast.expr) -> list[str]:
        """The paths an assignment to *target* writes: its own, or below what the value it is on holds."""
        held = self.versions.paths(target, store=True)
        if held:
            return held
        if isinstance(target, ast.Attribute):
            part = "." + target.attr
        elif isinstance(target, ast.Subscript) and not isinstance(
            target.slice, ast.Slice
        ):
            part = _subscript(target.slice, store=True)
        else:
            return []
        found = []
        for item in self.carried(target.value):
            base = self.held_by(item) if not item.at else None
            if base is not None:
                found.append(base + part)
        return found

    def _column(self, target: ast.expr, line: int, how: str) -> None:
        if not (
            isinstance(target, ast.Attribute) and target.attr in self.reader.columns
        ):
            return
        for on in self.target_paths(target.value):
            self.assigns.setdefault(
                (line, on, how, target.attr),
                PathAssign(line=line, path=on, attrs=[target.attr], how=how),
            )

    # -- calls -----------------------------------------------------------------------------------------------------

    def placed(self, call: ast.Call) -> C.Placed:
        found = self._placed.get(id(call))
        if found is None:
            try:
                found = self.reader.calls.callees(self.file, self.func, call)
            except Exception:  # noqa: BLE001 -- reach has the hole
                found = C.Placed()
            self._placed[id(call)] = found
        return found

    def to_of(self, call: ast.Call) -> list[str]:
        return sorted(
            {
                to
                for target in self.placed(call).targets
                if (to := self.reader.names.of(target.func)) is not None
            }
        )

    def _fill(self, call: ast.Call, line: int) -> None:
        """What a method no definition of the corpus is placed at puts into the object it is made on, as
        :func:`callfacts.meaning` reads it: a builtin container's, a library object's (kept whole)."""
        called = call.func
        meant = callfacts.meaning(
            self.reader.calls, self.file, self.func, call, self.placed(call)
        )
        if meant.method is None or meant.kind not in _FILLING:
            return
        items = []
        for value, put, key, keyword in callfacts.put_arguments(meant.method, call):
            if put.into == callfacts.ELEMENT:
                items += _shift(self.carried(value), "[]")
            elif put.into == callfacts.MERGE and isinstance(value, ast.Dict):
                items += self.carried(value)  # a dict literal's values under their keys
            elif put.into in (callfacts.ELEMENTS, callfacts.MERGE):
                items += _shift(_index(self.carried(value), "[]"), "[]")
            elif put.into == callfacts.VALUE and keyword:
                items += _shift(self.carried(value), _named(keyword))
            elif put.into == callfacts.VALUE and key is not None:
                items += _shift(self.carried(value), _subscript(key, store=True))
            elif put.into == callfacts.WHOLE:
                items += self.carried(value)
        if not items:
            return
        for base in self.target_paths(called.value):  # type: ignore[attr-defined]
            self.put(line, base, items, BIND_FILL)

    def _call(self, call: ast.Call) -> None:
        line = call.lineno
        called = call.func
        reset = order.reset_columns(call)
        if reset is not None and isinstance(called, ast.Attribute):
            columns = tuple(
                one
                for one in reset
                if one == order.ALL_COLUMNS or one in self.reader.columns
            )
            for item in self.carried(called.value) if columns else []:
                held = self.held_by(item) if not item.at else None
                if held is not None:
                    statement = pathcond.site_of(call)
                    self.resets.setdefault(
                        (id(statement), held, columns),
                        order.Reset(statement, held, columns),
                    )
        placed = self.placed(call)
        if isinstance(called, ast.Attribute) and not placed.targets:
            self._fill(call, line)
        passed: list[tuple[ast.expr, str, str]] = []
        for argument in [*call.args, *(keyword.value for keyword in call.keywords)]:
            value = argument.value if isinstance(argument, ast.Starred) else argument
            for item in self.carried(value):
                held = self.held_by(item)
                if held is not None:
                    passed.append((argument, held, item.at))
        receivers: list[str] = []
        if isinstance(called, ast.Attribute):
            if _is_super(called.value):
                receivers = ["self"]
            else:
                receivers = sorted(
                    {
                        held
                        for item in self.carried(called.value)
                        if not item.at and (held := self.held_by(item)) is not None
                    }
                )
        reader = self.reader
        for value in reach._handed_values(call):
            # ``Thread(target=self.run)``: the function handed runs with its receiver.
            for bound in (
                self.versions.paths(value.value)
                if isinstance(value, ast.Attribute)
                else []
            ):
                try:
                    landed = reader.calls.named_callables(
                        self.file, self.func, value, ()
                    )
                except Exception:  # noqa: BLE001 -- reach has the hole
                    landed = []
                for target in landed:
                    if (to := reader.names.of(target.func)) is not None:
                        _add(
                            self.handed,
                            CallBinding(line=line, to=to, parameter="self", path=bound),
                        )
        if not passed and not receivers:
            return
        if line in self.database:
            return  # a statement run: what it writes is read where it runs
        if not placed.targets:
            if (line, reach._cut(called)) in self.hole_lines:
                for _argument, held, at in passed:
                    _add(
                        self.handed,
                        CallBinding(line=line, to="", parameter="", path=held, at=at),
                    )
                for receiver in receivers:
                    _add(
                        self.handed,
                        CallBinding(line=line, to="", parameter="self", path=receiver),
                    )
            elif passed and reader._hands_a_callable(self.file, self.func, call):
                for _argument, held, at in passed:
                    _add(
                        self.handed,
                        CallBinding(
                            line=line,
                            to="",
                            parameter="",
                            path=held,
                            at=at,
                            why=HANDED_WITH_A_CALLABLE,
                        ),
                    )
            return
        for target in placed.targets:
            to = reader.names.of(target.func)
            if to is None:
                continue
            given = C.bind(call, target, placed.on_class)
            for argument, held, at in passed:
                taken = [name for name, value in given.items() if value is argument]
                if not taken and isinstance(argument, ast.Starred):
                    taken = [""]
                if not taken:
                    taken = [_rest_parameter(target.func, argument in call.args)]
                for name in taken:
                    _add(
                        self.handed,
                        CallBinding(line=line, to=to, parameter=name, path=held, at=at),
                    )
            # The receiver is ``self`` only where the method called is the one named: ``self.plugin_class(job)``
            # builds a new object, which is the call's result, not ``self``.
            if (
                not placed.on_class
                and isinstance(called, ast.Attribute)
                and called.attr == getattr(target.func, "name", None)
            ):
                for receiver in receivers:
                    _add(
                        self.handed,
                        CallBinding(line=line, to=to, parameter="self", path=receiver),
                    )


def _shift(items: list[_Carry], part: str) -> list[_Carry]:
    """*items* put at *part* inside a new value: ``[a]`` holds a at ``[]``."""
    return [one._replace(at=part + one.at) for one in items]


def _index(items: list[_Carry], part: str) -> list[_Carry]:
    """What is at *part* of a value holding *items*: ``[]`` matches every position, a position or key itself."""
    found = []
    for one in items:
        if not one.at:
            found.append(
                one._replace(below=one.below + part)
                if one.call is not None
                else one._replace(path=one.path + part)
            )
            continue
        first = one.at[: one.at.index("]") + 1]
        if first == part or first == "[]" or part == "[]":
            found.append(one._replace(at=one.at[len(first) :]))
    return found


def _attribute(items: list[_Carry], name: str) -> list[_Carry]:
    """An attribute of a value holding *items*: of what it is, not of what it holds."""
    return [
        one._replace(below=one.below + "." + name)
        if one.call is not None
        else one._replace(path=one.path + "." + name)
        for one in items
        if not one.at
    ]


def _subscript(key: ast.expr, store: bool = False) -> str:
    """The part a subscript reads: ``['k']`` for a literal str key, ``[i]`` for a literal int read, else ``[]``.

    An int written to (``x[0] = v``) is ``[]``: a list it is written in can
    be reordered later, and only a tuple, which cannot, keeps its positions.
    """
    if isinstance(key, ast.Constant):
        if isinstance(key.value, str):
            return _named(key.value)
        if type(key.value) is int and key.value >= 0 and not store:
            return f"[{key.value}]"
    return "[]"


def _key(key: ast.expr) -> str:
    """The part a dict literal puts a value at: its literal str key, else any."""
    if isinstance(key, ast.Constant) and isinstance(key.value, str):
        return _named(key.value)
    return "[]"


def _named(key: str) -> str:
    return f"['{key}']" if _KEY.fullmatch(key) else "[]"


def _add(handed: dict, binding: CallBinding) -> None:
    handed.setdefault(tuple(binding.model_dump().values()), binding)


#: Why a written column's value is not read.
_NO_MAP = "the statement runs with no bind map"
_NOT_BOUND_HERE = "the bind is not filled in this function (the map is handed in, or filled by a computed key)"


def _map_keys(
    func: ast.AST, run, key_holes, versions: "_Versions"
) -> list[WrittenValue]:
    """The keys a loop over a mapping names *run*'s row by (:data:`MAP_KEY`, :func:`sql.map_key`), one per path the
    mapping is; a term ``{}=:cr_{}`` whose holes are no such loop's key names nothing."""
    if not key_holes:
        return []
    holes = run.reading_holes
    found: dict[str, WrittenValue] = {}
    for column, bind, numbers in key_holes:
        if not all(0 < number <= len(holes) for number in (column, *numbers)):
            continue
        mapping = sql.map_key(
            func,  # type: ignore[arg-type]
            run,
            holes[column - 1],
            bind,
            tuple(holes[number - 1] for number in numbers),
        )
        for held in versions.paths(mapping) if mapping is not None else []:
            found.setdefault(held, WrittenValue(column=MAP_KEY, path=f"{held}[]"))
    return list(found.values())


def _whole_row(func: ast.AST, run, versions: "_Versions") -> tuple[str, bool, bool]:
    """The path whose ``valuesMap()`` the run's bind map is; whether only its changed attributes are in it; whether its key is new.

    ``execute(sql, job.valuesMap())`` hands the map in the call, and
    ``varMap = job.valuesMap(useSeq=True)`` binds it to a name first: the
    run says the name only (:attr:`sql.Execution.varmap`), so the call's
    argument is read here, and a name's every value.  A map filled entry by
    entry from one -- ``for k, v in task.valuesMap(onlyChanged=True).items():
    varMap[k] = v`` -- holds what it holds, beside what else it is given.
    """
    given = [(call, call.args[1]) for call in run.run_at if len(call.args) >= 2]
    values: list[ast.expr] = []
    for call, one in given:
        if isinstance(one, ast.Name):
            # ``varMap`` is bound again for each statement: only the binding that reaches this call is its map.
            # ``varMap: dict[str, Any] = {}`` binds it again as ``varMap = {}`` does (pathcond.targets_of).
            sites = [
                node
                for node in C.own_nodes(func)
                if len(targets := pathcond.targets_of(node)) == 1
                and isinstance(targets[0], ast.Name)
                and targets[0].id == one.id
            ]
            # A copy adds to the map and takes nothing from it: read as a reread, which overwrites nothing.
            copies = _row_copies(func, one.id)
            try:
                live = (
                    pathcond.surviving_writes(
                        sites + list(copies.values()),
                        call,
                        rereads=list(copies.values()),
                        as_run=True,
                    )
                    if sites or copies
                    else []
                )
            except Exception:  # noqa: BLE001 -- not read, so no whole row is said
                live = []
            values.extend(
                copies_from
                if (copies_from := _copied(copies, site)) is not None
                else site.value  # type: ignore[attr-defined]
                for site in live
            )
        else:
            values.append(one)
    for value in values:
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == "valuesMap"
        ):
            held = next(iter(versions.paths(value.func.value)), None)
            if held is None:
                continue
            flags = {
                keyword.arg
                for keyword in value.keywords
                if isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
            }
            return held, "onlyChanged" in flags, "useSeq" in flags
    return "", False, False


def _row_copies(func: ast.AST, name: str) -> dict[ast.expr, ast.AST]:
    """Each ``name[k] = v`` in ``for k, v in X.valuesMap(...).items()``, under the ``X.valuesMap(...)`` it copies."""
    found: dict[ast.expr, ast.AST] = {}
    for loop in C.own_nodes(func):
        if not (
            isinstance(loop, ast.For)
            and isinstance(loop.iter, ast.Call)
            and isinstance(loop.iter.func, ast.Attribute)
            and loop.iter.func.attr == "items"
            and not loop.iter.args
            and isinstance(copied := loop.iter.func.value, ast.Call)
            and isinstance(copied.func, ast.Attribute)
            and copied.func.attr == "valuesMap"
            and isinstance(loop.target, ast.Tuple)
            and len(loop.target.elts) == 2
            and all(isinstance(part, ast.Name) for part in loop.target.elts)
        ):
            continue
        key, value = (part.id for part in loop.target.elts)  # type: ignore[attr-defined]
        for node in loop.body:
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(target := node.targets[0], ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == name
                and isinstance(target.slice, ast.Name)
                and target.slice.id == key
                and isinstance(node.value, ast.Name)
                and node.value.id == value
            ):
                found[copied] = node
    return found


def _copied(copies: dict[ast.expr, ast.AST], site: ast.AST) -> Optional[ast.expr]:
    """The ``X.valuesMap(...)`` *site* copies entry by entry, if it is one of *copies*."""
    return next((copied for copied, node in copies.items() if node is site), None)


#: How a binding gives its name a value it spells: not a loop's, a ``with``'s or a caught exception's.
_SPELLED = (
    pathcond.BIND_ASSIGN,
    pathcond.BIND_ANNOTATED,
    pathcond.BIND_AUGMENTED,
    pathcond.BIND_WALRUS,
)


def _scalar(expr: Optional[ast.expr]) -> bool:
    """Whether *expr* is a str or a number by its own spelling, which holds no path: a literal (``None`` too), an
    f-string, a sign or ``not`` on one, a ``+`` with one as a term, ``"..." % x``."""
    if isinstance(expr, (ast.Constant, ast.JoinedStr)):
        return True
    if isinstance(expr, ast.UnaryOp):
        return isinstance(expr.op, ast.Not) or _scalar(expr.operand)
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
        return _scalar(expr.left) or _scalar(expr.right)
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Mod):
        return isinstance(expr.left, ast.JoinedStr) or (
            isinstance(expr.left, ast.Constant) and isinstance(expr.left.value, str)
        )
    return False


def _cut(node: ast.AST) -> str:
    text = ast.unparse(node)
    return text if len(text) <= 80 else text[:77] + "..."


def _rest_parameter(func: ast.AST, positional: bool) -> str:
    """The ``*args`` or ``**kwargs`` an argument no named parameter takes falls into; empty where there is none."""
    args = func.args  # type: ignore[attr-defined]
    rest = args.vararg if positional else args.kwarg
    return f"*{rest.arg}" if rest is not None else ""


def _local_names(func: ast.AST) -> frozenset[str]:
    args = func.args  # type: ignore[attr-defined]
    found = {one.arg for one in [*args.posonlyargs, *args.args, *args.kwonlyargs]}
    found |= {one.arg for one in (args.vararg, args.kwarg) if one is not None}
    for node in C.own_nodes(func):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            found.add(node.id)
    return frozenset(found | {"self", "cls"})


def _path(
    expr: Optional[ast.expr], names: frozenset[str], store: bool = False
) -> Optional[str]:
    """*expr* as a path rooted at one of *names*; ``None`` where it is not one.

    A subscript is the part :func:`_subscript` reads -- *store* where *expr*
    is written to.  A slice is not a path: it holds the container's elements,
    at positions not known (:meth:`_Values.carried`).
    """
    parts: list[str] = []
    outer = True
    while isinstance(expr, (ast.Attribute, ast.Subscript)):
        if isinstance(expr, ast.Attribute):
            parts.append("." + expr.attr)
        elif isinstance(expr.slice, ast.Slice):
            return None
        else:
            parts.append(_subscript(expr.slice, store=store and outer))
        outer = False
        expr = expr.value
    if not isinstance(expr, ast.Name) or expr.id not in names:
        return None
    return expr.id + "".join(reversed(parts))


def _is_super(expr: ast.expr) -> bool:
    return (
        isinstance(expr, ast.Call)
        and isinstance(expr.func, ast.Name)
        and expr.func.id == "super"
    )


def _parsed(text: str) -> Optional[ast.expr]:
    try:
        return ast.parse(text, mode="eval").body
    except SyntaxError:
        return None


def _run_at(
    line: int, regions: dict[int, tuple[int, int]], fetched: set[int]
) -> Optional[int]:
    """The run whose rows a call's result at *line* is: the run's own call, or a ``fetch*`` in its region."""
    if line in regions:
        return line
    held = [at for at, (_first, last) in regions.items() if at <= line <= last]
    return max(held) if line in fetched and held else None


def _head(path: str) -> str:
    """*path*'s first part, the name it is below: ``res@992`` of ``res@992[0]``, ``self`` of ``self.job``."""
    return re.split(r"[.\[]", path, maxsplit=1)[0]
