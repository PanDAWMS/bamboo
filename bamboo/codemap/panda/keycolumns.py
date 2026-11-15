"""What columns a value is, read where its function uses it: the hole of a run's mark (:class:`RunKey`).

A run's mark holds a value -- ``<jediTaskID={jediTaskID}>`` -- and a reader
matching a history row to the run (the querier's F1) asks which column of
the row's class the value is.  A declaration says it where the hole is an
attribute of a name declared as one class; this reads the rest from the
records the build already keeps.

**Forward, where it is used** -- each use a column of the rows a statement
writes or reads, so a column is what the value is there.  The value is
followed as an object is (:func:`bamboo.codemap.objects.follow`): through
the names the function gives it, the calls it is handed to -- whole, inside
a list or a dict, or as a part of an object -- and what a call hands back.
In each function it is held in, by any path:

- a key of a row a statement of the function writes, or a column of a row
  it inserts (:attr:`ReachNode.writes`): ``killJob``'s ``pandaID`` keys the
  job rows it updates;
- a term a statement it runs picks its rows by, filled from the value -- a
  row's own column compared with ``=`` or ``IN``, not excluding, deciding the
  rows (:func:`sqltext.terms`, :func:`sql.bound_values`, as the read side
  reads a bind).

``TaskRefinerThread.runImpl``'s ``jediTaskID`` is the key of the rows
``getTaskParamsWithID_JEDI`` reads, two calls down; ``AdderGen``'s
``peekJobs([self.job_id])`` hands it to ``peekJob``, which selects the job
by ``PandaID``.  Where the walk stops -- a call not placed, ``*args``, its
caps -- the stop says where.

**Backward, in the function.**  A binding reaching the hole whose value is
an attribute of a name declared as one class gives that attribute
(``tmp_id = job.PandaID``); a name bound to a name is read again, as it
runs (:func:`pathcond.surviving_writes`).

**A statement's rows.**  A binding read no further -- a loop's target, a
call's value, an item or a position of one -- is read again as the reader
of the guards' ways reads it (:meth:`heldfrom._Reader.binding_rows`): a
value that is a select item of a statement's rows is that item's column.
``copyArchive``'s ``for (id,) in res`` over ``SELECT PandaID FROM
jobsActive4`` prints ``id``, which is ``JobSpec.PandaID``, as the guard
``int(id) in xmlIDs`` beside it reads it.  A part of the rows that is no one
item -- a row whole, a part of an item -- says so (``row_item``); a binding
some of whose ways are not rows keeps its stop as well.

**Backward, up the callers.**  A parameter is what each call placed at the
function hands it (:meth:`Calls.callers`, :meth:`Calls.handed`), read
backward there in the same way, up to :data:`CALLERS_UP` calls; the columns
are every caller's.  ``recordRetryHistoryJEDI``'s ``newPandaID`` is
``JobSpec.PandaID`` and ``JobSpec.jobsetID``: ``insertNewJob`` hands it
``job.PandaID`` at one call and ``job.jobsetID`` at another.  The uses are
not followed at a caller: what the value is used as is read where the key
is.  A parameter a call leaves to its default holds the default, read
where the function's signature writes it.  Where no call is placed, a
call's ``*args`` or ``**kwargs`` may fill it, or the callers go further up,
the stop says so.  On pin k the uses alone had named
all but 3 pair holes; on pin m 66 of the 382 run keys stopping at a
parameter are named this way.

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
from typing import TYPE_CHECKING, Callable, Optional

from bamboo.codemap import callfacts, objects, pathcond, reach
from bamboo.codemap import calls as C
from bamboo.codemap.models import MAP_KEY, KeyStop, MapFragment, SourceModule
from bamboo.codemap.panda import objects as panda_objects
from bamboo.codemap.panda import sql, sqltext
from bamboo.codemap.panda.objects import is_parameter
from bamboo.codemap.panda.recognizers import progress
from bamboo.codemap.reaching import reaching

if TYPE_CHECKING:
    from bamboo.codemap.panda import heldfrom

#: How many calls up a parameter is read at its callers.
CALLERS_UP = 6

#: ``(name, binding, file, function) -> (the ways ending at a statement's rows, whether every way does)``:
#: :meth:`heldfrom._Reader.binding_rows`.
RowItems = Callable[
    [ast.Name, ast.AST, str, ast.AST], tuple[list["heldfrom.RowItem"], bool]
]


class Reader:
    """The columns of values, read in the functions of *fragment*'s reach record."""

    def __init__(
        self,
        fragment: MapFragment,
        modules: list[SourceModule],
        calls: C.Calls,
        names: reach._Names,
        funcs: dict[str, tuple[str, ast.AST]],
        follow: Callable[[str, str, int], objects.ObjectReach],
        row_items: Optional[RowItems] = None,
    ):
        self.calls = calls
        #: :meth:`heldfrom._Reader.binding_rows`, where the build reads the guards' ways.
        self.row_items = row_items
        self.types = calls.types
        self.trees = {module.rel_path: module.tree for module in modules}
        self.names = names
        self.funcs = funcs
        #: :func:`objects.follow` on the reach record, read once per state by its caller's cache.
        self.follow = follow
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
        #: (statement, bind) -> the paths of its function the bind is filled with (:meth:`_filled`).
        self._binds: dict[tuple[int, str], frozenset[str]] = {}
        #: The states each follow of the reading in progress reads (:meth:`read`'s *states*).
        self._states = objects.MAX_STATES

    def holds_rows(self, column: str) -> bool:
        """Whether ``Class.attribute`` *column* is a column of a class a table holds."""
        return column.partition(".")[0] in self.holding

    def read(
        self,
        file: str,
        func: ast.AST,
        node: ast.expr,
        states: int = objects.MAX_STATES,
    ) -> tuple[list[str], list[KeyStop]]:
        """``(columns, stops)`` of the value *node* holds in *func*: the columns sorted; the stops where each reading
        ended without a row's column, what the value is bound to first, then its uses, each follow of them reading
        at most *states* states.

        An attribute of a name declared as a class holding rows is that column
        and nothing else is read: the declaration says it.
        """
        found: set[str] = set()
        bound: list[KeyStop] = []
        uses: list[KeyStop] = []
        seen: set[tuple[str, str]] = set()
        owner = self.names.of(func) or ""
        node = callfacts.unconverted(self.types, file, func, node)
        self._states = states
        try:
            self._backward(file, func, owner, node, found, bound, seen)
            if isinstance(node, ast.Attribute) and any(
                self.holds_rows(one) for one in found
            ):
                return sorted(found), []
            used: set[str] = set()
            self._forward(owner, self.paths.of(func, node), used, uses)
            if not any(self.holds_rows(one) for one in used) and not uses:
                uses.append(_stop("no_use", at=_short(owner)))
            found |= used
        except Exception as error:  # noqa: BLE001 -- a key unread, not the build's end
            uses.append(_stop("failed", what=type(error).__name__))
        return sorted(found), _merged([*bound, *uses])

    # -- the readings --------------------------------------------------------- #

    def _forward(
        self,
        owner: str,
        held: frozenset[str],
        found: set[str],
        stops: list[KeyStop],
    ) -> None:
        """The columns the value *owner* holds by the paths *held* is used as, wherever it is followed from there; where the walk stops short, the stop.

        Read again where a field's reading has read it (:meth:`_field`): the stops go to the list asked for.
        """
        for path in sorted(held):
            reach = self.follow(owner, path, self._states)
            for at, paths in reach.held.items():
                if at in self.funcs:
                    self._used(at, *self.funcs[at], set(paths), found, stops)
            for site in reach.holes:
                kind = "spread" if site.how == objects.NO_PARAMETER else "hole"
                stops.append(_stop(kind, at=f"{_short(site.owner)}:{site.line}"))
            if reach.cut:
                what = "paths" if reach.cut == objects.HELD_SPENT else "states"
                stops.append(_stop("cut", what=what, at=_short(owner)))
            for site in reach.capped:
                stops.append(
                    _stop(
                        "capped",
                        name=site.how[len(_CAPPED) :],
                        at=f"{_short(site.owner)}:{site.line}",
                    )
                )

    def _used(
        self,
        owner: str,
        file: str,
        func: ast.AST,
        held: set[str],
        found: set[str],
        stops: list[KeyStop],
    ) -> None:
        """The columns *func* uses the value held by the paths *held* as -- its writes' keys, its statements' terms -- a path matched part by part (:func:`objects._holds`)."""
        row = self.rows.get(owner)
        if row is None:
            return
        here = _short(owner)
        for write in row.writes:
            named = [*write.keys, *(write.columns if write.kind == "insert" else [])]
            for value in named:
                if value.column == MAP_KEY:
                    columns = objects.map_key_columns(value.path, held)
                elif value.path and objects._holds(held, value.path):
                    columns = {value.column}
                else:
                    continue
                for column in sorted(columns):
                    self._column(
                        write.table,
                        column,
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
                    if self._filled(file, func, run, key, held):
                        self._column(
                            term.table,
                            term.column,
                            f"{here}:{run.call.lineno}",
                            found,
                            stops,
                        )

    def _filled(
        self, file: str, func: ast.AST, run: sql.Execution, key: str, held: set[str]
    ) -> bool:
        """Whether *run* fills its bind *key* with the value *held* names in *func*: the binds read once per
        statement and bind, as every value read meets the same statements."""
        if (id(run), key) not in self._binds:
            binds = sql.bound_values(
                func, run.varmap or "", key, run.window, across=True, at=run
            )  # type: ignore[arg-type]
            if not run.varmap:
                binds = binds + sql.inline_binds(run.call, key)
            self._binds[(id(run), key)] = frozenset(
                path
                for bind in binds
                if pathcond.enclosing_function(bind.value) is func
                for path in self.paths.of(
                    func, callfacts.unconverted(self.types, file, func, bind.value)
                )
            )
        return any(objects._holds(held, path) for path in self._binds[(id(run), key)])

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
        up: int = 0,
    ) -> None:
        """What the bindings of *func* reaching *node* give it: an attribute of a declared name, read again through
        names, and a parameter what its callers hand (:meth:`_handed`), *up* calls up already."""
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
        # An item or a slice of a list the function makes and only appends to holds what it appended:
        # copyArchive's ``str(jobs[iJob:iJob + nJob])``, ``jobs.append(id)`` for each row's PandaID.
        listed = node.value if isinstance(node, ast.Subscript) else node
        items = appended(func, listed) if isinstance(listed, ast.Name) else None
        if items is not None:
            for item in items:
                self._backward(file, func, owner, item, found, stops, seen, up)
            return
        if not isinstance(node, ast.Name):
            stops.append(_stop("value", what=_kind(node), at=at))
            return
        if (owner, f"{node.id}@{id(node)}") in seen:
            return
        seen.add((owner, f"{node.id}@{id(node)}"))
        entry, bindings = self._reaching(func, node.id, node)
        if entry:
            self._handed(file, func, owner, node.id, found, stops, seen, up)
        elif not bindings:
            stops.append(_stop("outside", name=node.id, at=_short(owner)))
        for binding in bindings:
            where = f"{_short(owner)}:{getattr(binding.site, 'lineno', '?')}"
            value = binding.value
            if binding.kind == pathcond.BIND_FOR:
                iterable = getattr(binding.site, "iter", None)
                what = (
                    _kind(callfacts.unconverted(self.types, file, func, iterable))
                    if iterable is not None
                    else "a loop"
                )
                stop = _stop("element", name=node.id, what=what, at=where)
                self._at_binding(file, func, node, binding.site, stop, found, stops)
            elif value is None:
                stop = _stop(
                    "bound", name=node.id, what=f"a {binding.kind} target", at=where
                )
                self._at_binding(file, func, node, binding.site, stop, found, stops)
            elif binding.unpacked:
                if (
                    binding.slot is not None
                    and isinstance(value, (ast.Tuple, ast.List))
                    and len(value.elts) > binding.slot
                ):
                    self._given(
                        file,
                        func,
                        owner,
                        node,
                        binding.site,
                        value.elts[binding.slot],
                        found,
                        stops,
                        seen,
                        up,
                    )
                else:
                    stop = _stop(
                        "bound",
                        name=node.id,
                        what=f"a position of {_kind(callfacts.unconverted(self.types, file, func, value))}",
                        at=where,
                    )
                    self._at_binding(file, func, node, binding.site, stop, found, stops)
            else:
                self._given(
                    file, func, owner, node, binding.site, value, found, stops, seen, up
                )

    def _given(
        self,
        file: str,
        func: ast.AST,
        owner: str,
        node: ast.Name,
        site: ast.AST,
        value: ast.expr,
        found: set[str],
        stops: list[KeyStop],
        seen: set[tuple[str, str]],
        up: int,
    ) -> None:
        """What *value*, bound to *node*'s name at *site* of *func*, gives it: a name or an attribute read back
        (:meth:`_backward`), else the binding read as a statement's rows (:meth:`_at_binding`)."""
        value = callfacts.unconverted(self.types, file, func, value)
        if isinstance(value, (ast.Name, ast.Attribute)):
            self._backward(file, func, owner, value, found, stops, seen, up)
            return
        at = f"{_short(owner)}:{getattr(value, 'lineno', '?')}"
        stop = _stop("bound", name=node.id, what=_kind(value), at=at)
        self._at_binding(file, func, node, site, stop, found, stops)

    def _at_binding(
        self,
        file: str,
        func: ast.AST,
        node: ast.Name,
        site: ast.AST,
        stop: KeyStop,
        found: set[str],
        stops: list[KeyStop],
    ) -> None:
        """The reading of *node* stopped at its binding *site* (*stop*): the binding read again as a statement's rows
        (:attr:`row_items`).  A way that is a select item whole is its column, spelled as the table's class declares
        it (:meth:`_column`); one that is the rows but no one item whole says so (``row_item``); *stop* stands unless
        every way is the rows."""
        items: list = []
        every = False
        if self.row_items is not None:
            try:
                items, every = self.row_items(node, site, file, func)
            except Exception:  # noqa: BLE001 -- the binding unread as rows: its stop stands
                items, every = [], False
        if not every:
            stops.append(stop)
        where = stop.at[0] if stop.at else ""
        for item in items:
            if item.whole and item.column:
                table, _dot, column = item.column.rpartition(".")
                self._column(table, column, where, found, stops)
            elif item.whole and item.attr and self.holds_rows(item.attr):
                found.add(item.attr)
            else:
                what = item.why or (
                    f"a value made from {item.column} ({item.made})"
                    if item.made and item.column
                    else f"a part of {item.column}"
                    if item.column
                    else ""
                )
                stops.append(_stop("row_item", name=item.site, what=what, at=where))

    def _handed(
        self,
        file: str,
        func: ast.AST,
        owner: str,
        name: str,
        found: set[str],
        stops: list[KeyStop],
        seen: set[tuple[str, str]],
        up: int,
    ) -> None:
        """What each call placed at *func* hands its parameter *name*, read backward at the caller (:meth:`_backward`),
        up to :data:`CALLERS_UP` calls; where a call says nothing of it, the stop."""
        if (owner, f"{name}@parameter") in seen:
            return
        seen.add((owner, f"{name}@parameter"))
        if up >= CALLERS_UP:
            stops.append(
                _stop("callers_up", name=name, what=str(CALLERS_UP), at=_short(owner))
            )
            return
        tree = self.trees.get(file)
        target = C.Target(
            file, func, pathcond.class_of(tree, func) if tree is not None else None
        )
        try:
            sites = self.calls.callers(target)
        except Exception as error:  # noqa: BLE001 -- the callers unread, not the build's end
            stops.append(
                _stop("callers_failed", what=type(error).__name__, at=_short(owner))
            )
            return
        if not sites:
            stops.append(_stop("no_caller", name=name, at=_short(owner)))
        for site in sites:
            handed = self.calls.handed(site, target).get(name)
            caller = self.names.of(site.func) or ""
            if handed is None:
                at = f"{_short(caller) or site.file}:{site.call.lineno}"
                stops.append(_stop("handed_spread", name=name, at=at))
                continue
            on_class = self.calls.callees(site.file, site.func, site.call).on_class
            if name not in C.said(site.call, target, on_class):
                # Left to its default: written in the function's own signature, not at the call.
                at = f"{_short(owner)}:{getattr(handed, 'lineno', '?')}"
                stops.append(_stop("value", what=_kind(handed), at=at))
                continue
            value = callfacts.unconverted(self.types, site.file, site.func, handed)
            self._backward(
                site.file, site.func, caller, value, found, stops, seen, up + 1
            )

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
        up: int = 0,
    ) -> None:
        """What *name* is given by *value*, bound to it in *func*: a name or an attribute read back, else the stop saying what it is."""
        value = callfacts.unconverted(self.types, file, func, value)
        if isinstance(value, (ast.Name, ast.Attribute)):
            self._backward(file, func, owner, value, found, stops, seen, up)
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
                self._forward(owner, frozenset({f"self.{attribute}"}), found, stops)

    def _executions_of(self, func: ast.AST) -> list[sql.Execution]:
        if id(func) not in self._executions:
            try:
                self._executions[id(func)] = sql.executions(func)  # type: ignore[arg-type]
            except Exception:  # noqa: BLE001 -- the reach row has the hole
                self._executions[id(func)] = []
        return self._executions[id(func)]


#: What :attr:`objects.ObjectReach.capped` says before the path not followed.
_CAPPED = objects.PARTS_SPENT.format(path="")


#: The calls a list is read by and not changed: its length, a printed or sorted copy of it.
_READS_A_LIST = frozenset({"len", "str", "repr", "sorted", "list", "tuple", "set"})


def appended(func: ast.AST, read: ast.Name) -> Optional[list[ast.expr]]:
    """What the local list *read* reads in *func* holds, where every binding of the name reaching *read* binds it to
    an empty list and the uses of the name those bindings reach only append to it: each value appended --
    copyArchive's ``jobs = []`` at 875 and ``jobs.append(id)`` at 878, read at 885.  The name's other bindings are
    other lists and are not asked: the same ``jobs`` made again for other queries at 754, 857 and 924, handed whole to
    ``Client.kill_jobs`` at 760, or bound to ``jobsMap[site]`` at 907.  Read where it is sliced, indexed, iterated,
    tested, measured or copied; ``None`` where anything else is done with it -- handed to a call, extended, put
    into, returned, bound to another name -- or nothing is appended."""
    own = [one for one in pathcond.bindings(func, read.id) if not one.nested]
    sites = [one.site for one in own]
    made = (
        {id(site) for site in reaching(pathcond.site_of(read), sites)}
        if sites
        else set()
    )
    if not made or not all(_empty_list(one) for one in own if id(one.site) in made):
        return None
    values: list[ast.expr] = []
    for node in C.own_nodes(func):
        if not (
            isinstance(node, ast.Name)
            and node.id == read.id
            and isinstance(node.ctx, ast.Load)
        ):
            continue
        if not any(
            id(site) in made for site in reaching(pathcond.site_of(node), sites)
        ):
            continue
        parent = getattr(node, "parent", None)
        if isinstance(parent, ast.Attribute):
            call = getattr(parent, "parent", None)
            if not (
                parent.attr == "append"
                and isinstance(call, ast.Call)
                and call.func is parent
                and len(call.args) == 1
            ):
                return None
            values.append(call.args[0])
        elif (
            isinstance(parent, ast.Subscript)
            and parent.value is node
            and isinstance(parent.ctx, ast.Load)
        ):
            continue
        elif isinstance(parent, ast.Call) and node in parent.args:
            if not (
                isinstance(parent.func, ast.Name) and parent.func.id in _READS_A_LIST
            ):
                return None
        elif not isinstance(
            parent,
            (
                ast.If,
                ast.While,
                ast.IfExp,
                ast.UnaryOp,
                ast.BoolOp,
                ast.Compare,
                ast.For,
            ),
        ):
            return None
        elif isinstance(parent, ast.For) and parent.iter is not node:
            return None
    return values or None


def _empty_list(binding: pathcond.Binding) -> bool:
    """Whether *binding* binds its name, whole, to ``[]``."""
    return (
        binding.kind in (pathcond.BIND_ASSIGN, pathcond.BIND_ANNOTATED)
        and not binding.unpacked
        and isinstance(binding.value, ast.List)
        and not binding.value.elts
    )


def listed(types, file: str, func: ast.AST, node: ast.expr) -> bool:
    """Whether *node* prints several values of a list :func:`appended` reads -- the list, or a slice of it."""
    node = callfacts.unconverted(types, file, func, node)
    if isinstance(node, ast.Subscript):
        if not isinstance(node.slice, ast.Slice):
            return False
        node = node.value
    return isinstance(node, ast.Name) and appended(func, node) is not None


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
