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
-- so a module's logger or a class read for its methods is not one.  The
bindings are read without the order of the statements: a name bound to a
path anywhere holds it everywhere in the function, which can only say an
object reaches more than it does.

Not read, and so an object can reach more than this says: a call inside a
lambda's body, a nested function's use of the names of the one around it,
and ``setattr(job, "jobStatus", v)`` with the name spelled (an assignment by
another spelling).  On pin k a search of the source finds none of the three
handing or setting a JobSpec, JediTaskSpec or FileSpec -- a search, not a
reading of the map.

:meth:`_Reader.writes` adds the statements that write rows
(:class:`~bamboo.codemap.models.DbWrite`): each written column's value as a
path or another value, the keys of the row, and whether the bind map is the
whole of one object (``job.valuesMap()``).
"""

from __future__ import annotations

import ast
from typing import Iterator, Optional

from bamboo.codemap import calls as C
from bamboo.codemap import reach
from bamboo.codemap.models import (
    BIND_ALIAS,
    BIND_ELEMENT,
    BIND_FILL,
    BIND_RESULT,
    BIND_RETURN,
    CallBinding,
    DbWrite,
    MapFragment,
    PathAssign,
    PathBinding,
    SourceModule,
    WrittenValue,
)
from bamboo.codemap.panda import pathcond, specwrites, sql, sqltext
from bamboo.codemap.panda.recognizers import progress

#: The classes whose columns an assignment is kept for (tmaeno, 10-18: D1).
COLUMN_CLASSES = ("JobSpec", "JediTaskSpec", "FileSpec")

#: The longest path kept, in parts after the root: deeper reads are of objects
#: an object holds, not of the object.
MAX_PARTS = 5

#: Builtins that hand back the elements of what they are given, in some order.
_ELEMENTS_OF = {"enumerate", "sorted", "reversed", "list", "tuple", "set"}
#: A container's own methods that put their argument into it.
_FILLERS = {"append", "insert", "add"}
_EXTENDERS = {"extend", "update"}

#: Why a hole the reach record does not hold is one.
HANDED_WITH_A_CALLABLE = "handed to code outside the corpus together with a function of the corpus, which it may call with it"


def attach(
    fragment: MapFragment, modules: list[SourceModule], calls: C.Calls
) -> tuple[int, int, int, int]:
    """Read every function's object records into its :class:`ReachNode`; how many assigns, bindings, call bindings and writes.

    Inside the build's reading context (:func:`sql.across_calls`), so the
    statements are the ones every slice read.
    """
    reader = _Reader(modules, calls)
    rows = {row.owner: row for row in fragment.reach_graph}
    counts = [0, 0, 0, 0]
    for file, func, owner in reader.names.functions:
        row = rows.get(owner)
        if row is None:
            continue
        row.assigns, row.bindings, row.call_bindings = reader.read(file, func, row)
        row.writes = reader.writes(func)
        counts[0] += len(row.assigns)
        counts[1] += len(row.bindings)
        counts[2] += len(row.call_bindings)
        counts[3] += len(row.writes)
    return counts[0], counts[1], counts[2], counts[3]


class _Reader:
    def __init__(self, modules: list[SourceModule], calls: C.Calls):
        self.calls = calls
        self.types = calls.types
        self.names = reach._Names(calls.types)
        declared = progress.spec_attributes(modules)
        self.columns: frozenset[str] = frozenset().union(
            *(declared.get(name, set()) for name in COLUMN_CLASSES)
        )

    def read(
        self, file: str, func: ast.AST, row
    ) -> tuple[list[PathAssign], list[PathBinding], list[CallBinding]]:
        names = _local_names(func)
        assigns: dict[tuple, PathAssign] = {}
        bindings: dict[tuple, PathBinding] = {}
        handed: dict[tuple, CallBinding] = {}

        def path(expr: Optional[ast.expr]) -> Optional[str]:
            return _path(expr, names)

        def bind(line: int, name: Optional[str], held: Optional[str], how: str) -> None:
            if name is not None and held is not None and name != held:
                bindings.setdefault(
                    (line, name, held, how),
                    PathBinding(line=line, name=name, path=held, how=how),
                )

        def result(name: Optional[str], value: ast.expr) -> None:
            """*name* holds what the call *value* returns, or an element of it (``f()[0]``)."""
            call = value.value if isinstance(value, ast.Subscript) else value
            if name is not None and isinstance(call, ast.Call):
                element = "[]" if call is not value else ""
                bindings.setdefault(
                    (call.lineno, name, element, BIND_RESULT),
                    PathBinding(
                        line=call.lineno, name=name, path=element, how=BIND_RESULT
                    ),
                )

        hole_lines = {(hole.line, hole.callee) for hole in row.holes}
        database = set(row.database)
        for node in C.own_nodes(func):
            line = getattr(node, "lineno", 0)
            if (
                isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign))
                and node.value is not None
            ):
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                for target in targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and target.attr in self.columns
                    ):
                        on = path(target.value)
                        if on is not None:
                            how = "+=" if isinstance(node, ast.AugAssign) else "="
                            assigns.setdefault(
                                (line, on, how, target.attr),
                                PathAssign(
                                    line=line, path=on, attrs=[target.attr], how=how
                                ),
                            )
                    if isinstance(node, ast.AugAssign):
                        if isinstance(node.value, (ast.List, ast.Tuple)):
                            for element in node.value.elts:
                                bind(line, path(target), path(element), BIND_FILL)
                        continue
                    named = (
                        path(target)
                        if isinstance(target, (ast.Name, ast.Attribute, ast.Subscript))
                        else None
                    )
                    value = node.value
                    if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
                        for element in value.elts:
                            bind(line, named, path(element), BIND_FILL)
                            result(named + "[]" if named is not None else None, element)
                    elif _is_result(value):
                        result(named, value)
                    else:
                        bind(line, named, path(value), BIND_ALIAS)
            elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
                for name, held in _iterated(node.target, node.iter, names):
                    bind(
                        getattr(node, "lineno", getattr(node.iter, "lineno", 0)),
                        name,
                        held,
                        BIND_ELEMENT,
                    )
            elif isinstance(node, ast.Return) and node.value is not None:
                if isinstance(node.value, (ast.List, ast.Tuple)):
                    for element in node.value.elts:
                        bind(line, "<return>[]", path(element), BIND_RETURN)
                        result("<return>[]", element)
                elif _is_result(node.value):
                    result("<return>", node.value)
                else:
                    bind(line, "<return>", path(node.value), BIND_RETURN)
            elif isinstance(node, ast.Call):
                self._call(
                    file, func, node, path, bind, result, handed, hole_lines, database
                )
        for write in specwrites.writes_in(self.types, file, func):
            target = _path(_parsed(write.target), names)
            if target is None:
                continue
            attrs = (
                sorted(write.names & self.columns) if write.names is not None else ["*"]
            )
            if not attrs:
                continue
            assigns.setdefault(
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
            sorted(assigns.values(), key=key),
            sorted(bindings.values(), key=key),
            sorted(handed.values(), key=key),
        )

    def _call(
        self,
        file,
        func,
        call: ast.Call,
        path,
        bind,
        result,
        handed,
        hole_lines,
        database,
    ) -> None:
        line = call.lineno
        called = call.func
        if isinstance(called, ast.Attribute):
            container = path(called.value)
            if called.attr in _FILLERS and call.args:
                bind(line, container, path(call.args[-1]), BIND_FILL)
                # ``found.append(self.proxy.peekJob(one))``: an element is what the call returns.
                result(
                    container + "[]" if container is not None else None, call.args[-1]
                )
            elif called.attr in _EXTENDERS and call.args:
                argument = call.args[0]
                if isinstance(argument, (ast.List, ast.Tuple, ast.Set)):
                    for element in argument.elts:
                        bind(line, container, path(element), BIND_FILL)
                else:
                    held = path(argument)
                    bind(
                        line,
                        container,
                        held + "[]" if held is not None else None,
                        BIND_FILL,
                    )
        passed: list[tuple[ast.expr, str, bool]] = []
        for argument in [*call.args, *(keyword.value for keyword in call.keywords)]:
            value = argument.value if isinstance(argument, ast.Starred) else argument
            held = path(value)
            if held is not None:
                passed.append((argument, held, False))
            elif isinstance(value, (ast.List, ast.Tuple, ast.Set)):
                passed.extend(
                    (argument, one, True)
                    for element in value.elts
                    if (one := path(element)) is not None
                )
        receiver = None
        if isinstance(called, ast.Attribute):
            receiver = "self" if _is_super(called.value) else path(called.value)
        for value in reach._handed_values(call):
            # ``Thread(target=self.run)``: the function handed runs with its receiver.
            if (
                isinstance(value, ast.Attribute)
                and (bound := path(value.value)) is not None
            ):
                try:
                    landed = self.calls.named_callables(file, func, value, ())
                except Exception:  # noqa: BLE001 -- reach has the hole
                    landed = []
                for target in landed:
                    if (to := self.names.of(target.func)) is not None:
                        _add(
                            handed,
                            CallBinding(line=line, to=to, parameter="self", path=bound),
                        )
        if not passed and receiver is None:
            return
        if line in database:
            return  # a statement run: what it writes is read where it runs
        try:
            placed = self.calls.callees(file, func, call)
        except Exception:  # noqa: BLE001 -- reach has the hole
            placed = C.Placed()
        if not placed.targets:
            if (line, reach._cut(called)) in hole_lines:
                for _argument, held, element in passed:
                    _add(
                        handed,
                        CallBinding(
                            line=line, to="", parameter="", path=held, element=element
                        ),
                    )
                if receiver is not None:
                    _add(
                        handed,
                        CallBinding(line=line, to="", parameter="self", path=receiver),
                    )
            elif passed and self._hands_a_callable(file, func, call):
                for _argument, held, element in passed:
                    _add(
                        handed,
                        CallBinding(
                            line=line,
                            to="",
                            parameter="",
                            path=held,
                            element=element,
                            why=HANDED_WITH_A_CALLABLE,
                        ),
                    )
            return
        for target in placed.targets:
            to = self.names.of(target.func)
            if to is None:
                continue
            given = C.bind(call, target, placed.on_class)
            for argument, held, element in passed:
                taken = [name for name, value in given.items() if value is argument]
                if not taken and isinstance(argument, ast.Starred):
                    taken = [""]
                if not taken:
                    taken = [_rest_parameter(target.func, argument in call.args)]
                for name in taken:
                    _add(
                        handed,
                        CallBinding(
                            line=line, to=to, parameter=name, path=held, element=element
                        ),
                    )
            # The receiver is ``self`` only where the method called is the one named: ``self.plugin_class(job)``
            # builds a new object, which is the call's result, not ``self``.
            if (
                receiver is not None
                and not placed.on_class
                and isinstance(called, ast.Attribute)
                and called.attr == getattr(target.func, "name", None)
            ):
                _add(
                    handed,
                    CallBinding(line=line, to=to, parameter="self", path=receiver),
                )

    def writes(self, func: ast.AST) -> list[DbWrite]:
        """Each statement *func* runs that writes rows, with where its values and keys come from."""
        names = _local_names(func)
        found: list[DbWrite] = []
        try:
            runs = sql.executions(func)  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001 -- the reach row has the hole
            return found
        for run in runs:
            text = run.reading
            try:
                read = sqltext.writes(text)
                tables = [(table, "insert") for table in sqltext.creates(text)] + [
                    (table, "update") for table in sqltext.updates(text)
                ]
            except Exception:  # noqa: BLE001
                continue
            if not read and not tables:
                continue
            whole, changed, sequenced = _whole_row(func, run, names)
            unread = bool(run.unread_columns)
            new_row = sequenced or any(
                "useSeq=True" in (hole.source or "").replace(" ", "")
                for _offset, hole in run.holes
            )
            lines = sorted({call.lineno for call in run.run_at})
            seen: set[str] = set()
            for one in read:
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
                                func, run, column, given, whole, names
                            )
                        ],
                        whole=whole,
                        changed=changed,
                        unread=unread,
                        keys=[
                            value
                            for column, bind in sorted(one.keys.items())
                            for value in self._values(
                                func, run, column, sqltext.classify(bind), "", names
                            )
                        ],
                        new_row=new_row
                        or any(
                            "nextval" in given.text.lower()
                            for given in one.columns.values()
                        ),
                    )
                )
            for table, kind in tables:
                bare = table.rpartition(".")[2]
                if bare.lower() not in seen:
                    # Nothing named, but rows written: a whole row of {}, or a SET only of names made at run time.
                    seen.add(bare.lower())
                    found.append(
                        DbWrite(
                            lines=lines,
                            table=bare,
                            kind=kind,
                            whole=whole,
                            changed=changed,
                            unread=True,
                            new_row=new_row,
                        )
                    )
        # One statement read in several variants of its text says the same write more than once.
        unique: dict[str, DbWrite] = {}
        for one in found:
            unique.setdefault(one.model_dump_json(), one)
        return list(unique.values())

    def _values(
        self, func, run, column: str, given, whole: str, names
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
                binds = sql.bound_values(func, run.varmap, given.text, at=run.run_at)  # type: ignore[arg-type]
            except Exception:  # noqa: BLE001
                binds = []
        if not binds:
            return [WrittenValue(column=column, why=_NOT_BOUND_HERE)]
        found = []
        for bind in binds:
            held = _path(bind.value, names)
            if held is not None:
                found.append(WrittenValue(column=column, path=held))
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


def _add(handed: dict, binding: CallBinding) -> None:
    handed.setdefault(tuple(binding.model_dump().values()), binding)


#: Why a written column's value is not read.
_NO_MAP = "the statement runs with no bind map"
_NOT_BOUND_HERE = "the bind is not filled in this function (the map is handed in, or filled by a computed key)"


def _whole_row(func: ast.AST, run, names: frozenset[str]) -> tuple[str, bool, bool]:
    """The path whose ``valuesMap()`` the run's bind map is; whether only its changed attributes are in it; whether its key is new.

    ``execute(sql, job.valuesMap())`` hands the map in the call, and
    ``varMap = job.valuesMap(useSeq=True)`` binds it to a name first: the
    run says the name only (:attr:`sql.Execution.varmap`), so the call's
    argument is read here, and a name's every value.
    """
    given = [(call, call.args[1]) for call in run.run_at if len(call.args) >= 2]
    values: list[ast.expr] = []
    for call, one in given:
        if isinstance(one, ast.Name):
            # ``varMap`` is bound again for each statement: only the binding that reaches this call is its map.
            sites = [
                node
                for node in C.own_nodes(func)
                if isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == one.id
            ]
            try:
                live = (
                    pathcond.surviving_writes(sites, call, as_run=True) if sites else []
                )
            except Exception:  # noqa: BLE001 -- not read, so no whole row is said
                live = []
            values.extend(site.value for site in live)  # type: ignore[attr-defined]
        else:
            values.append(one)
    for value in values:
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == "valuesMap"
        ):
            held = _path(value.func.value, names)
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


def _path(expr: Optional[ast.expr], names: frozenset[str]) -> Optional[str]:
    """*expr* as a path rooted at one of *names*; ``None`` where it is not one."""
    parts: list[str] = []
    while isinstance(expr, (ast.Attribute, ast.Subscript)):
        parts.append("." + expr.attr if isinstance(expr, ast.Attribute) else "[]")
        expr = expr.value
    if not isinstance(expr, ast.Name) or expr.id not in names or len(parts) > MAX_PARTS:
        return None
    return expr.id + "".join(reversed(parts))


def _is_result(value: ast.expr) -> bool:
    """A call, or an element of what one returns: ``f(...)``, ``f(...)[0]``."""
    return isinstance(value, ast.Call) or (
        isinstance(value, ast.Subscript) and isinstance(value.value, ast.Call)
    )


def _is_super(expr: ast.expr) -> bool:
    return (
        isinstance(expr, ast.Call)
        and isinstance(expr.func, ast.Name)
        and expr.func.id == "super"
    )


def _iterated(
    target: ast.expr, iterated: ast.expr, names: frozenset[str]
) -> Iterator[tuple[str, str]]:
    """``(name, path[])`` for each name a loop binds to an element of a path."""
    pairs = False
    if (
        isinstance(iterated, ast.Call)
        and isinstance(iterated.func, ast.Name)
        and iterated.args
    ):
        if iterated.func.id == "enumerate":
            pairs, iterated = True, iterated.args[0]
        elif iterated.func.id in _ELEMENTS_OF:
            iterated = iterated.args[0]
    elif (
        isinstance(iterated, ast.Call)
        and isinstance(iterated.func, ast.Attribute)
        and not iterated.args
    ):
        if iterated.func.attr == "values":
            iterated = iterated.func.value
        elif iterated.func.attr == "items":
            pairs, iterated = True, iterated.func.value
    held = _path(iterated, names)
    if held is None:
        return
    if pairs:
        if (
            isinstance(target, ast.Tuple)
            and len(target.elts) == 2
            and isinstance(target.elts[1], ast.Name)
        ):
            yield target.elts[1].id, held + "[]"
    elif isinstance(target, ast.Name):
        yield target.id, held + "[]"


def _parsed(text: str) -> Optional[ast.expr]:
    try:
        return ast.parse(text, mode="eval").body
    except SyntaxError:
        return None
