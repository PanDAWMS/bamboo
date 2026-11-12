"""Where each function holding a junction's arm is entered from (:class:`~bamboo.codemap.models.EnteredFrom`).

A reader ruling an arm out by the tests its function was called under -- the querier's R3, which needs
``doGeneratePrePro``'s arms to say they run only under ``doGenerate`` 1150's ``taskSpec.usePrePro()`` -- asks two
things of the map: the tests each call into the function sits under, and whether those calls are every way in.  The
first is the reach record's edges into the function (:class:`~bamboo.codemap.models.ReachCall`, every ``how``), each
read in its caller by :func:`~bamboo.codemap.pathcond.arm_condition`.  The second is every way in the record does
not hold, each said as a record (:class:`~bamboo.codemap.models.EnteredOpen`):

- a hole, or a call placed elsewhere and left open, made with the function's name: it may land there;
- the function's value read where nothing calls it, outside a hand-off the record holds: whatever is handed it may
  call it -- a method's as an attribute, a function's as a name or an attribute, a class's for its ``__init__``, and
  ``getattr`` with the name written;
- a ``getattr`` whose name is read at run time, off a module (every function a module holds, and every ``__init__``),
  off an object of the function's class where what it takes is called, or off an object no reading places where what
  it takes is called (every function); off an object outside the corpus, none;
- an endpoint or a handler: the server runs it for a request.

A local or a parameter called runs a value handed to it or taken somewhere, both of which are said above, so it is no
way in of its own.  Nor is the far end of a forwarder (:data:`~bamboo.codemap.callfacts.FORWARDERS`), which runs a call
its near end made.
"""

from __future__ import annotations

import ast
import logging
from typing import Iterable, Optional

from bamboo.codemap import callfacts, pathcond, reach
from bamboo.codemap.calls import UNPLACED_DYNAMIC, UNPLACED_OUTSIDE, Calls
from bamboo.codemap.models import (
    ENTERED_ENDPOINT,
    ENTERED_HANDLER,
    ENTERED_HOLE,
    ENTERED_NAMED_AT_RUN_TIME,
    ENTERED_OPEN_CALL,
    ENTERED_TAKEN,
    CallStop,
    EnteredFrom,
    EnteredOpen,
    EnteredSite,
    JunctionNode,
    ReachCall,
    ReachNode,
)

logger = logging.getLogger(__name__)

_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)
_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
#: The holes whose callee is a value, not a name: what it runs was handed or taken, or a ``getattr`` took it.
_VALUE_CALLED = (reach.HOLE_LOCAL_CALLED, UNPLACED_DYNAMIC)
#: The calls that hand back a module.
_MODULE_MAKERS = frozenset({"__import__", "import_module", "importlib.import_module"})
#: What a ``getattr`` with a name read at run time may take: a module's names, an object's of these classes, any.
_MODULE, _ALL = "module", "all"
#: What a plugin loader may build, read off :data:`~bamboo.codemap.callfacts.MODULE_TAKES`: any class, or these.
_ANY_CLASS = "any class"
_BUILDS = "builds"
#: Why a site's conditions were not read.
UNREAD_NO_CALLER = "the caller's definition is not in the trees"
UNREAD_NO_STATEMENT = "no node at the edge's line matches it"


def _file(owner: str) -> str:
    return owner.partition("::")[0]


def _cut(node: ast.AST) -> str:
    return reach._cut(node)


def _definition(trees: dict[str, ast.AST], row: ReachNode) -> Optional[ast.FunctionDef | ast.AsyncFunctionDef]:
    """The ``def`` a reach row is, by its anchor: the innermost function of its name holding the anchor's line."""
    anchor = row.anchor
    tree = trees.get(anchor.file) if anchor is not None else None
    if tree is None or anchor is None:
        return None
    best = None
    for node in ast.walk(tree):
        if (
            isinstance(node, _FUNCTIONS)
            and node.name == row.function
            and min([node.lineno, *(one.lineno for one in node.decorator_list)])
            <= anchor.line_start
            <= (node.end_lineno or node.lineno)
        ):
            if best is None or node.lineno > best.lineno:
                best = node
    return best


def _own_nodes(func: ast.AST) -> Iterable[ast.AST]:
    """The nodes of *func* outside the functions and classes nested in it (a lambda's are its own)."""
    todo = list(ast.iter_child_nodes(func))
    while todo:
        node = todo.pop()
        yield node
        if not isinstance(node, _SCOPES):
            todo.extend(ast.iter_child_nodes(node))


def _site_node(func: ast.AST, call: ReachCall, line: int) -> Optional[ast.AST]:
    """The node an edge was read at on *line*, one of its lines: a call of its callee, or the expression it names."""
    found: list[ast.expr] = []
    for node in _own_nodes(func):
        if getattr(node, "lineno", None) != line:
            continue
        if isinstance(node, ast.Call) and _cut(node.func) == call.callee:
            return node
        if isinstance(node, ast.expr) and _cut(node) == call.callee:
            found.append(node)
    return min(found, key=lambda one: (one.lineno, one.col_offset)) if found else None


def _typed(tree: ast.AST) -> set[int]:
    """The nodes of *tree* that name a type rather than hand a value on: a class's bases, an annotation, what a
    subscript is subscripted with (``FactoryBase[TaskRefinerBase]``), ``isinstance`` / ``issubclass``'s classes and an
    ``except``'s."""
    roots: list[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            roots += node.bases + [one.value for one in node.keywords]
        elif isinstance(node, _FUNCTIONS):
            arguments = node.args
            every = arguments.posonlyargs + arguments.args + arguments.kwonlyargs
            every += [one for one in (arguments.vararg, arguments.kwarg) if one is not None]
            roots += [one.annotation for one in every if one.annotation is not None]
            roots += [node.returns] if node.returns is not None else []
        elif isinstance(node, ast.AnnAssign):
            roots.append(node.annotation)
        elif isinstance(node, ast.Subscript):
            roots.append(node.slice)
        elif isinstance(node, ast.Call) and _cut(node.func) in ("isinstance", "issubclass") and len(node.args) == 2:
            roots.append(node.args[1])
        elif isinstance(node, ast.ExceptHandler) and node.type is not None:
            roots.append(node.type)
    return {id(one) for root in roots for one in ast.walk(root)}


class _Taken:
    """Where each name's value is read without a call, over the corpus: ``name -> [(file, line, form, text)]``; a
    form ``type`` where the read names a type (:func:`_typed`), which hands no value on."""

    def __init__(self, trees: dict[str, ast.AST]):
        self.places: dict[str, list[tuple[str, int, str, str]]] = {}
        for file, tree in trees.items():
            called = {id(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
            typed = _typed(tree)
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load) and id(node) not in called:
                    self._add(node.attr, file, node, "type" if id(node) in typed else "attribute")
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and id(node) not in called:
                    self._add(node.id, file, node, "type" if id(node) in typed else "name")
                elif (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "getattr"
                    and len(node.args) >= 2
                    and isinstance(node.args[1], ast.Constant)
                    and isinstance(node.args[1].value, str)
                ):
                    self._add(node.args[1].value, file, node, "getattr")

    def _add(self, name: str, file: str, node: ast.AST, form: str) -> None:
        self.places.setdefault(name, []).append((file, node.lineno, form, _cut(node)))  # type: ignore[attr-defined]

    def of(self, name: str, forms: frozenset[str]) -> list[tuple[str, int, str, str]]:
        return [one for one in self.places.get(name, []) if one[2] in forms]


def _bindings(func: ast.AST, name: str) -> list[ast.expr]:
    """The values *func* binds *name* to with ``=`` (its own statements)."""
    found: list[ast.expr] = []
    for node in _own_nodes(func):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            found.append(node.value)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
            and node.value
        ):
            found.append(node.value)
    return found


def _rebinds(value: ast.AST, receiver: ast.expr) -> bool:
    """Whether *value* is bound by ``=`` to the very name *receiver* is: ``mod = getattr(mod, part)``."""
    parent = getattr(value, "parent", None)
    return (
        isinstance(parent, ast.Assign)
        and isinstance(receiver, ast.Name)
        and any(isinstance(target, ast.Name) and target.id == receiver.id for target in parent.targets)
    )


def _is_getattr_by_value(node: ast.AST) -> bool:
    """``getattr(o, name)`` whose name is not written."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) >= 2
        and not (isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str))
    )


class _RunTimeNames:
    """Each ``getattr`` whose name is read at run time, and what it may take: ``[(at, text, scope)]``, a scope being
    :data:`_MODULE` (a module's function or class: no reading says which), :data:`_ALL`, :data:`_ANY_CLASS`,
    ``(_BUILDS, classes)`` (one of these classes, built) or a frozenset of ``file::Class`` (a method of an object of
    one of them).  Off a module, what the code says it takes (:data:`~bamboo.codemap.callfacts.MODULE_TAKES`);
    :attr:`unlisted` holds the places off a module it does not say of."""

    def __init__(self, trees: dict[str, ast.AST], record: reach.ReachGraph, calls: Calls):
        self.trees, self.record, self.calls = trees, record, calls
        self.takes: list[tuple[str, str, object]] = []
        self.unlisted: list[str] = []
        seen: set[tuple[str, int, int]] = set()
        # A module's own body (run at import, no row of the record): ``panda_config`` loads its plugins' classes there.
        for file, tree in trees.items():
            for node in _own_nodes(tree):
                if (
                    _is_getattr_by_value(node)
                    and self._module(f"{file}::<module>", tree, node.args[0])  # type: ignore[attr-defined]
                    and not _rebinds(node, node.args[0])  # type: ignore[attr-defined]
                ):
                    self._off_module(seen, f"{file}::<module>", file, node)
        for row in record.rows.values():
            if row.owner in callfacts.FORWARDERS:
                continue
            func = _definition(trees, row)
            if func is None:
                continue
            file = row.anchor.file  # type: ignore[union-attr]
            # A module's names are taken whatever is done with them: a plugin's class is kept and built elsewhere.
            # But for a step down to a submodule, bound back to the name it was taken off (:meth:`_module`).
            for node in _own_nodes(func):
                if (
                    _is_getattr_by_value(node)
                    and self._module(row.owner, func, node.args[0])  # type: ignore[attr-defined]
                    and not _rebinds(node, node.args[0])  # type: ignore[attr-defined]
                ):
                    self._off_module(seen, row.owner, file, node)
            # Off anything else, what is taken counts where the function calls it.
            for hole in row.holes:
                if not hole.why.startswith(_VALUE_CALLED):
                    continue
                for node in _own_nodes(func):
                    if not (isinstance(node, ast.Call) and node.lineno == hole.line and _cut(node.func) == hole.callee):
                        continue
                    callee = node.func
                    taken = [callee] if _is_getattr_by_value(callee) else []
                    if isinstance(callee, ast.Name):
                        taken = [one for one in _bindings(func, callee.id) if _is_getattr_by_value(one)]
                    for one in taken:
                        scope = self._scope(row, func, one.args[0])  # type: ignore[attr-defined]
                        if scope is not None:
                            self._take(seen, file, one, scope)

    def _off_module(self, seen: set, owner: str, file: str, node: ast.AST) -> None:
        receiver = node.args[0]  # type: ignore[attr-defined]
        reading = callfacts.MODULE_TAKES.get((owner, receiver.id)) if isinstance(receiver, ast.Name) else None
        if reading is None:
            self.unlisted.append(f"{file}:{node.lineno}")  # type: ignore[attr-defined]
            self._take(seen, file, node, _MODULE)
            return
        kind, what = reading
        if kind == callfacts.TAKES_ENDPOINTS:
            return  # an endpoint's function, open as an endpoint already
        if kind == callfacts.TAKES_CLASSES:
            self._take(seen, file, node, _ANY_CLASS)
            return
        self._take(seen, file, node, (_BUILDS, self._classes(kind, what)))

    def _classes(self, kind: str, what: str) -> frozenset[str]:
        """``file::Class`` of each class a loader the code says *kind* of may build."""
        roots: list[tuple[str, ast.ClassDef]] = []
        if kind == callfacts.TAKES_TYPE_PARAMETER:
            for file, tree in self.trees.items():
                for node in ast.walk(tree):
                    if isinstance(node, ast.Subscript) and _cut(node.value).rpartition(".")[2] == what:
                        found = self.calls.types.class_named(file, node.slice)
                        if found is not None:
                            roots.append(found)
        elif kind == callfacts.TAKES_SUBCLASSES:
            file, _, name = what.partition("::")
            cls = self.calls.types.class_def(file, name)
            if cls is not None:
                roots.append((file, cls))
        elif kind == callfacts.TAKES_NAMED_IN:
            file, _, name = what.partition("::")
            holder = self.trees.get(file)
            table = next(
                (
                    one.value
                    for one in (holder.body if isinstance(holder, ast.Module) else ())
                    if isinstance(one, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == name for target in one.targets)
                ),
                None,
            )
            named = {
                one.value
                for one in (ast.walk(table) if table is not None else ())
                if isinstance(one, ast.Constant) and isinstance(one.value, str)
            }
            return frozenset(
                f"{where}::{node.name}"
                for where, tree in self.trees.items()
                for node in ast.walk(tree)
                if isinstance(node, ast.ClassDef) and node.name in named
            )
        found_keys = set()
        for file, cls in roots:
            found_keys.add(f"{file}::{cls.name}")
            found_keys |= {f"{where}::{one.name}" for where, one in self.calls.subclasses(file, cls)}
        return frozenset(found_keys)

    def _take(self, seen: set, file: str, node: ast.AST, scope: object) -> None:
        key = (file, node.lineno, node.col_offset)  # type: ignore[attr-defined]
        if key not in seen:
            seen.add(key)
            self.takes.append((f"{file}:{node.lineno}", _cut(node), scope))  # type: ignore[attr-defined]

    def _module(self, owner: str, func: ast.AST, receiver: ast.expr, deciding: frozenset[str] = frozenset()) -> bool:
        """Whether *receiver* holds a module: every binding of it in *func* makes one, or the facts say it does.

        A name bound again to a ``getattr`` off itself is decided by its other bindings: ``mod = __import__(name)``
        and then ``mod = getattr(mod, part)`` for each dotted part of *name*, which ``__import__`` bound as an
        attribute of its parent -- each step a submodule."""
        if not isinstance(receiver, ast.Name):
            return False
        if receiver.id in deciding or (owner, receiver.id) in callfacts.MODULE_TAKES:
            return True
        values = _bindings(func, receiver.id)
        inner = deciding | {receiver.id}
        return bool(values) and all(
            isinstance(value, ast.Call)
            and (
                _cut(value.func) in _MODULE_MAKERS
                or (_is_getattr_by_value(value) and self._module(owner, func, value.args[0], inner))
            )
            for value in values
        )

    def _outside(self, row: ReachNode, func: ast.AST, receiver: ast.expr) -> bool:
        """Whether every binding of *receiver* in *func* is what a call outside the corpus made."""
        if not isinstance(receiver, ast.Name):
            return False
        values = _bindings(func, receiver.id)
        file = row.anchor.file  # type: ignore[union-attr]
        for value in values:
            if not isinstance(value, ast.Call):
                return False
            placed = self.calls.callees(file, func, value)
            if placed.targets or placed.unplaced != UNPLACED_OUTSIDE:
                return False
        return bool(values)

    def _scope(self, row: ReachNode, func: ast.AST, receiver: ast.expr) -> Optional[object]:
        if self._module(row.owner, func, receiver):
            return None  # taken off a module: read where it is taken, whatever is done with it (:meth:`_off_module`)
        if self._outside(row, func, receiver):
            return None
        found, opened = self.calls.types.candidates(row.anchor.file, func, receiver)  # type: ignore[union-attr]
        classes = frozenset(f"{where}::{name}" for where, name in found if where != "<builtins>")
        if opened or not found:
            return _ALL
        return classes or None


def entered_froms(
    junctions: Iterable[JunctionNode],
    trees: dict[str, ast.AST],
    calls: Optional[Calls],
    record: Optional[reach.ReachGraph],
    open_calls: Iterable[CallStop] = (),
    routes: Iterable[tuple[str, str]] = (),
    handlers: Iterable[tuple[str, str]] = (),
) -> list[EnteredFrom]:
    """An :class:`EnteredFrom` for each function holding a junction's arm that the reach record has an edge into.
    *routes* and *handlers* are ``(route, owner)`` of what the server runs for a request.  Nothing without *calls* or
    *record*."""
    if calls is None or record is None:
        return []
    arms: dict[str, set[int]] = {}
    for junction in junctions:
        file, _, name = junction.owner.partition("::")
        if not name or name.startswith("<"):
            continue
        for branch in junction.branches:
            for line in [branch.line, *branch.other_lines]:
                owner = record.function_at(file, line) if line is not None else None
                if owner is not None:
                    arms.setdefault(owner, set()).add(line)  # type: ignore[arg-type]
    into: dict[str, list[tuple[ReachNode, ReachCall]]] = {}
    for row in record.rows.values():
        for call in row.calls:
            into.setdefault(call.to, []).append((row, call))
    taken = _Taken(trees)
    run_time = _RunTimeNames(trees, record, calls)
    if run_time.unlisted:
        logger.info(
            "ways in: a getattr off a module no MODULE_TAKES entry reads, taken as any of a module's: %s",
            run_time.unlisted,
        )
    served = [(route, owner, ENTERED_ENDPOINT) for route, owner in routes] + [
        (route, owner, ENTERED_HANDLER) for route, owner in handlers
    ]
    stops = list(open_calls)
    found: list[EnteredFrom] = []
    for owner, lines in arms.items():
        edges = into.get(owner)
        row = record.rows[owner]
        func = _definition(trees, row)
        if not edges or func is None:
            continue
        file = _file(owner)
        sites: list[EnteredSite] = []
        seen: set[tuple[str, int, str]] = set()
        handed: set[tuple[str, int]] = set()
        for caller, call in edges:
            if call.how in ("handed", "lambda"):
                handed |= {(_file(caller.owner), one) for one in call.lines or [call.line]}
            # A row stands for every call going one way, with each one's line: each is a site, under its own tests.
            for line in call.lines or [call.line]:
                key = (caller.owner, line, call.how)
                if key in seen:
                    continue
                seen.add(key)
                sites.append(_site(trees, caller, call, line))
        opened = _open(row, func, file, record, taken, run_time, stops, served, handed, edges)
        found.append(
            EnteredFrom(
                owner=f"{file}::{row.function}",
                line=func.lineno,
                arms=sorted(lines),
                sites=sorted(sites, key=lambda one: (one.owner, one.line, one.how)),
                open=opened,
            )
        )
    return sorted(found, key=lambda one: (one.owner, one.line))


def _site(trees: dict[str, ast.AST], caller: ReachNode, call: ReachCall, line: int) -> EnteredSite:
    func = _definition(trees, caller)
    owner = f"{_file(caller.owner)}::{caller.function}"
    if func is None:
        return EnteredSite(owner=owner, owner_line=0, line=line, how=call.how, unread=UNREAD_NO_CALLER)
    node = _site_node(func, call, line)
    if node is None:
        return EnteredSite(owner=owner, owner_line=func.lineno, line=line, how=call.how, unread=UNREAD_NO_STATEMENT)
    return EnteredSite(
        owner=owner, owner_line=func.lineno, line=line, how=call.how, conditions=pathcond.arm_condition(node)
    )


def _open(
    row: ReachNode,
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    file: str,
    record: reach.ReachGraph,
    taken: _Taken,
    run_time: _RunTimeNames,
    stops: list[CallStop],
    served: list[tuple[str, str, str]],
    handed: set[tuple[str, int]],
    edges: list[tuple[ReachNode, ReachCall]],
) -> list[EnteredOpen]:
    found: list[EnteredOpen] = []
    name = row.function
    for owner, line, callee in record.holes_spelled(name):
        found.append(EnteredOpen(kind=ENTERED_HOLE, at=f"{_file(owner)}:{line}", text=callee))
    entering = {(caller.owner, call.line) for caller, call in edges}
    for stop in stops:
        if stop.callee.rpartition(".")[2] == name and (stop.owner, stop.line) not in entering:
            found.append(EnteredOpen(kind=ENTERED_OPEN_CALL, at=f"{stop.file}:{stop.line}", text=stop.callee))
    builds = name in ("__init__", "__new__")
    if builds:
        places = taken.of(row.cls.rpartition("::")[2].rpartition(".")[2], frozenset({"attribute", "name", "getattr"}))
    elif row.cls:
        places = taken.of(name, frozenset({"attribute", "getattr"}))
    else:
        places = taken.of(name, frozenset({"attribute", "name", "getattr"}))
    for where, line, _form, text in places:
        if (where, line) not in handed:
            found.append(EnteredOpen(kind=ENTERED_TAKEN, at=f"{where}:{line}", text=text))
    classes = set(row.classes) | ({row.cls} if row.cls else set())
    for at, text, scope in run_time.takes:
        if (
            scope == _ALL
            or (scope == _MODULE and (builds or not row.cls))
            or (scope == _ANY_CLASS and builds)
            or (isinstance(scope, tuple) and builds and scope[1] & classes)
            or (isinstance(scope, frozenset) and scope & classes)
        ):
            found.append(EnteredOpen(kind=ENTERED_NAMED_AT_RUN_TIME, at=at, text=text))
    bare = f"{file}::{name}"
    for route, owner, kind in served:
        if owner in (bare, row.owner):
            found.append(EnteredOpen(kind=kind, at=f"{file}:{func.lineno}", text=route))
    return found
