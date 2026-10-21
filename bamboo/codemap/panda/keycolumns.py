"""What columns a value is, read where its function uses it: the hole of a run's mark (:class:`RunKey`).

A run's mark holds a value -- ``<jediTaskID={jediTaskID}>`` -- and a reader
matching a history row to the run (the querier's F1) asks which column of
the row's class the value is.  A declaration says it where the hole is an
attribute of a name declared as one class; this reads the rest, in the
function, from the records the build already keeps.

**Forward, where the function uses it** -- each use a column of the rows a
statement writes or reads, so a column is what the value is there:

- a key of a row a statement of the function writes, or a column of a row
  it inserts (:attr:`ReachNode.writes`): ``killJob``'s ``pandaID`` keys the
  job rows it updates;
- a term a statement it runs picks its rows by, filled from the value -- a
  row's own column compared with ``=`` or ``IN``, not excluding, deciding the
  rows (:func:`sqltext.terms`, :func:`sql.bound_values`, as the read side
  reads a bind);
- a parameter of a call it is handed whole to (:attr:`ReachNode.call_bindings`),
  read there the same way: ``TaskRefinerThread.runImpl``'s ``jediTaskID`` is
  the key of the rows ``getTaskParamsWithID_JEDI`` reads, two calls down.

A value handed inside a container -- ``peekJobs([self.job_id])``, a dict's
value -- or to ``*args`` is not followed, and the why says where.

**Backward, in the function.**  A binding reaching the hole whose value is
an attribute of a name declared as one class gives that attribute
(``tmp_id = job.PandaID``); a name bound to a name is read again, as it
runs (:func:`pathcond.surviving_writes`).  What the callers hand a
parameter is not read: the callers are many and the value's own uses say
more (on pin k, the walk up named 3 pair holes the uses did not).

**A field of a class holding no row.**  ``AdderGen.job_id``,
``Closer.panda_id``: the attribute of a class no table is held by is no
row's column, so the field is read in its class -- each assignment of it,
read backward there, and each method reading it, read forward there:
``Closer.__init__``'s ``self.panda_id = job.PandaID``.

**Spelled as the class declares it.**  A table's column is the attribute of
the class the table holds (:func:`bamboo.codemap.objects.table_classes`)
declared by that name, case aside (``jeditaskid`` is
``JediTaskSpec.jediTaskID``).  A table no class holds, or a column its class
does not declare, gives no column: no history row has it, and the why names
it where nothing else is named.
"""

from __future__ import annotations

import ast
from typing import Optional

from bamboo.codemap import calls as C
from bamboo.codemap import objects, reach
from bamboo.codemap.models import KeyStop, MapFragment, SourceModule
from bamboo.codemap.panda import objects as panda_objects
from bamboo.codemap.panda import pathcond, sql, sqltext
from bamboo.codemap.panda.objects import is_parameter, unconverted
from bamboo.codemap.panda.recognizers import progress


class Reader:
    """The columns of values, read in the functions of *fragment*'s reach record."""

    def __init__(
        self,
        fragment: MapFragment,
        modules: list[SourceModule],
        calls: C.Calls,
        names: reach._Names,
        funcs: dict[str, tuple[str, ast.AST]],
    ):
        self.types = calls.types
        self.names = names
        self.funcs = funcs
        self.rows = {row.owner: row for row in fragment.reach_graph}
        self.tables = objects.table_classes(fragment.entities)
        #: The classes a table holds: a column of one is a row's.
        self.holding = frozenset(cls for cls in self.tables.values() if cls)
        self.declared = {
            cls: {name.lower(): name for name in attributes}
            for cls, attributes in progress.spec_attributes(modules).items()
        }
        self.paths = panda_objects.Paths()
        self._executions: dict[int, list[sql.Execution]] = {}

    def holds_rows(self, column: str) -> bool:
        """Whether ``Class.attribute`` *column* is a column of a class a table holds."""
        return column.partition(".")[0] in self.holding

    def read(
        self, file: str, func: ast.AST, node: ast.expr
    ) -> tuple[list[str], list[KeyStop]]:
        """``(columns, stops)`` of the value *node* holds in *func*: the columns sorted; the stops where each reading
        ended without a row's column, what the value is bound to first, then its uses.

        An attribute of a name declared as a class holding rows is that column
        and nothing else is read: the declaration says it.
        """
        found: set[str] = set()
        bound: list[KeyStop] = []
        uses: list[KeyStop] = []
        seen: set[tuple[str, str]] = set()
        owner = self.names.of(func) or ""
        node = unconverted(node)
        try:
            self._backward(file, func, owner, node, found, bound, seen)
            if isinstance(node, ast.Attribute) and any(
                self.holds_rows(one) for one in found
            ):
                return sorted(found), []
            before = {one for one in found if self.holds_rows(one)}
            self._forward(owner, func, self.paths.of(func, node), found, uses, seen)
            if {one for one in found if self.holds_rows(one)} == before and not uses:
                uses.append(_stop("no_use", at=_short(owner)))
        except Exception as error:  # noqa: BLE001 -- a key unread, not the build's end
            uses.append(_stop("failed", what=type(error).__name__))
        return sorted(found), _merged([*bound, *uses])

    # -- the readings --------------------------------------------------------- #

    def _forward(
        self,
        owner: str,
        func: ast.AST,
        held: frozenset[str],
        found: set[str],
        stops: list[KeyStop],
        seen: set[tuple[str, str]],
    ) -> None:
        """The columns *func* uses the value held by the paths *held* as: its writes' keys, its statements' terms, the calls it hands it to."""
        row = self.rows.get(owner)
        if not held or row is None:
            return
        here = _short(owner)
        for write in row.writes:
            named = [*write.keys, *(write.columns if write.kind == "insert" else [])]
            for value in named:
                if value.path and value.path in held:
                    self._column(
                        write.table,
                        value.column,
                        f"{here}:{write.lines[0]}",
                        found,
                        stops,
                    )
        for run in self._executions_of(func):
            try:
                reading = sqltext.terms(run.reading)
            except Exception:  # noqa: BLE001 -- the statement's terms are not read
                continue
            for term in reading.terms:
                if term.scope != sqltext.ROW or term.excluding or not term.decided:
                    continue
                for key in term.binds:
                    if self._filled(func, run, key, held):
                        self._column(
                            term.table,
                            term.column,
                            f"{here}:{run.call.lineno}",
                            found,
                            stops,
                        )
        for bound in row.call_bindings:
            if bound.path not in held or bound.parameter == "self":
                continue
            at = f"{here}:{bound.line}"
            if not bound.to:
                stops.append(_stop("hole", at=at))
            elif bound.at:
                stops.append(_stop("container", at=at))
            elif not bound.parameter:
                stops.append(_stop("spread", at=at))
            elif (bound.to, bound.parameter) not in seen and bound.to in self.funcs:
                seen.add((bound.to, bound.parameter))
                callee = self.funcs[bound.to][1]
                self._forward(
                    bound.to, callee, frozenset({bound.parameter}), found, stops, seen
                )

    def _filled(
        self, func: ast.AST, run: sql.Execution, key: str, held: frozenset[str]
    ) -> bool:
        """Whether *run* fills its bind *key* with the value *held* names in *func*."""
        binds = sql.bound_values(
            func, run.varmap or "", key, run.window, across=True, at=run.run_at
        )  # type: ignore[arg-type]
        if not run.varmap:
            binds = binds + sql.inline_binds(run.call, key)
        return any(
            pathcond.enclosing_function(bind.value) is func
            and self.paths.of(func, unconverted(bind.value)) & held
            for bind in binds
        )

    def _column(
        self, table: str, column: str, at: str, found: set[str], stops: list[KeyStop]
    ) -> None:
        """*table*'s *column* as the attribute its class declares, or the stop saying why it is none."""
        cls = self.tables.get(table.rpartition(".")[2].lower())
        spelled = f"{table}.{column}"
        if cls is None:
            stops.append(_stop("no_class", name=spelled, at=at))
            return
        attribute = self.declared.get(cls, {}).get(column.lower())
        if attribute is None:
            stops.append(_stop("not_declared", name=spelled, what=cls, at=at))
            return
        found.add(f"{cls}.{attribute}")

    def _backward(
        self,
        file: str,
        func: ast.AST,
        owner: str,
        node: ast.expr,
        found: set[str],
        stops: list[KeyStop],
        seen: set[tuple[str, str]],
    ) -> None:
        """What the bindings of *func* reaching *node* give it: an attribute of a declared name, read again through names."""
        at = f"{_short(owner)}:{getattr(node, 'lineno', '?')}"
        if isinstance(node, ast.Attribute):
            root = node.value
            try:
                if isinstance(root, ast.Name) and root.id == "self":
                    held = self.types.method_class(file, func)
                    declared = frozenset({held}) if held is not None else frozenset()
                else:
                    declared = self.types.declared(file, func, root) or frozenset()
            except Exception as error:  # noqa: BLE001 -- a declaration unread, not the build's end
                stops.append(
                    _stop(
                        "declaration_failed", name=_cut(root), what=type(error).__name__
                    )
                )
                return
            if not declared:
                stops.append(_stop("undeclared", name=_cut(root), at=at))
            elif len({cls.name for _where, cls in declared}) > 1:
                names = ", ".join(sorted({cls.name for _where, cls in declared}))
                stops.append(_stop("several", name=_cut(root), what=names, at=at))
            else:
                for where, cls in sorted(declared, key=lambda one: one[0]):
                    column = f"{cls.name}.{node.attr}"
                    found.add(column)
                    if not self.holds_rows(column):
                        self._field(where, cls, node.attr, found, stops, seen)
            return
        if not isinstance(node, ast.Name):
            stops.append(_stop("value", what=_kind(node)))
            return
        if (owner, f"{node.id}@{id(node)}") in seen:
            return
        seen.add((owner, f"{node.id}@{id(node)}"))
        entry, bindings = self._reaching(func, node.id, node)
        if entry:
            stops.append(_stop("parameter", name=node.id, at=_short(owner)))
        elif not bindings:
            stops.append(_stop("outside", name=node.id, at=_short(owner)))
        for binding in bindings:
            where = f"{_short(owner)}:{getattr(binding.site, 'lineno', '?')}"
            value = binding.value
            if binding.kind == pathcond.BIND_FOR:
                iterable = getattr(binding.site, "iter", None)
                what = (
                    _kind(unconverted(iterable)) if iterable is not None else "a loop"
                )
                stops.append(_stop("element", name=node.id, what=what, at=where))
            elif value is None:
                stops.append(
                    _stop(
                        "bound", name=node.id, what=f"a {binding.kind} target", at=where
                    )
                )
            elif binding.unpacked:
                if (
                    binding.slot is not None
                    and isinstance(value, (ast.Tuple, ast.List))
                    and len(value.elts) > binding.slot
                ):
                    self._bound(
                        file,
                        func,
                        owner,
                        node.id,
                        value.elts[binding.slot],
                        found,
                        stops,
                        seen,
                    )
                else:
                    stops.append(
                        _stop(
                            "bound",
                            name=node.id,
                            what=f"a position of {_kind(unconverted(value))}",
                            at=where,
                        )
                    )
            else:
                self._bound(file, func, owner, node.id, value, found, stops, seen)

    def _bound(
        self,
        file: str,
        func: ast.AST,
        owner: str,
        name: str,
        value: ast.expr,
        found: set[str],
        stops: list[KeyStop],
        seen: set[tuple[str, str]],
    ) -> None:
        """What *name* is given by *value*, bound to it in *func*: a name or an attribute read back, else the stop saying what it is."""
        value = unconverted(value)
        if isinstance(value, (ast.Name, ast.Attribute)):
            self._backward(file, func, owner, value, found, stops, seen)
            return
        at = f"{_short(owner)}:{getattr(value, 'lineno', '?')}"
        stops.append(_stop("bound", name=name, what=_kind(value), at=at))

    def _reaching(
        self, func: ast.AST, name: str, node: ast.expr
    ) -> tuple[bool, list[pathcond.Binding]]:
        """Whether *func*'s entry binding of the parameter *name* reaches *node*, and the bindings in it that do."""
        bound = [one for one in pathcond.bindings(func, name) if not one.nested]
        sites = list({id(one.site): one.site for one in bound}.values())
        entry = is_parameter(func, name)
        if entry:
            sites = [func.args, *sites]  # type: ignore[attr-defined]
        if not sites:
            return False, []
        try:
            kept = pathcond.surviving_writes(sites, node, as_run=True)
        except Exception:  # noqa: BLE001 -- read as if every binding reached it
            kept = sites
        alive = {id(one) for one in kept}
        return (
            entry and id(func.args) in alive,  # type: ignore[attr-defined]
            [one for one in bound if id(one.site) in alive],
        )

    # -- a field -------------------------------------------------------------- #

    def _field(
        self,
        file: str,
        cls: ast.ClassDef,
        attribute: str,
        found: set[str],
        stops: list[KeyStop],
        seen: set[tuple[str, str]],
    ) -> None:
        """*cls*'s *attribute*, *cls* holding no row, read in its lineage: each assignment of it, and each method reading it."""
        found.add(f"{cls.name}.{attribute}")
        if (f"{file}::{cls.name}", attribute) in seen:
            return
        seen.add((f"{file}::{cls.name}", attribute))
        stops.append(_stop("no_row", name=cls.name))
        for holder_file, holder in self.types.lineage(file, cls):
            for method in pathcond.methods_of(holder):
                owner = self.names.of(method)
                if owner is None:
                    continue
                for value in pathcond.attribute_expressions(method, attribute):
                    self._bound(
                        holder_file,
                        method,
                        owner,
                        f"self.{attribute}",
                        value,
                        found,
                        stops,
                        seen,
                    )
                self._forward(
                    owner, method, frozenset({f"self.{attribute}"}), found, stops, seen
                )

    def _executions_of(self, func: ast.AST) -> list[sql.Execution]:
        if id(func) not in self._executions:
            try:
                self._executions[id(func)] = sql.executions(func)  # type: ignore[arg-type]
            except Exception:  # noqa: BLE001 -- the reach row has the hole
                self._executions[id(func)] = []
        return self._executions[id(func)]


def _stop(kind: str, name: str = "", what: str = "", at: str = "") -> KeyStop:
    return KeyStop(kind=kind, name=name, what=what, at=[at] if at else [])


def _merged(stops: list[KeyStop]) -> list[KeyStop]:
    """*stops* once each, those saying one thing at several sites as one, with each site."""
    kept: dict[tuple[str, str, str], KeyStop] = {}
    for stop in stops:
        held = kept.setdefault(
            (stop.kind, stop.name, stop.what),
            KeyStop(kind=stop.kind, name=stop.name, what=stop.what),
        )
        held.at.extend(site for site in stop.at if site not in held.at)
    return list(kept.values())


def _short(owner: str) -> str:
    return owner.rpartition("::")[2]


def _cut(node: ast.AST) -> str:
    text = ast.unparse(node)
    return text if len(text) <= 60 else text[:57] + "..."


def _kind(node: Optional[ast.AST]) -> str:
    """What *node* is, as a stop says it."""
    if isinstance(node, ast.Constant):
        return "a literal"
    if isinstance(node, ast.Call):
        return f"what {_cut(node.func)}() returns"
    if isinstance(node, ast.Subscript):
        return f"an item of {_cut(node.value)}"
    if isinstance(node, (ast.Name, ast.Attribute)):
        return _cut(node)
    return "a value made from parts"
