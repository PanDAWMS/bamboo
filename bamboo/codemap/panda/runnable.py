"""Which definitions of the corpus a reader may run with a run's values and get what the run got: :class:`Runnable`.

A reader deciding an arm's test with a run's values meets calls in it: ``DataServiceUtils.getDestinationSE(token)``,
``taskSpec.useScout()``, ``EventServiceUtils.isEventServiceJob(job)``.  It can run such a definition where running it
changes nothing and gives what it gave in the run -- its value is what it is handed alone.  Whether that holds is a
reading of the code, so the map says it, one record per definition a call in an arm's path condition can land at
-- read in the function the condition names, the owner's or the one an ``[in f()]`` entry is tested in --, a JSON
maker on a way is handed as its hook (``RefinerUtils.unicodeConvert``), or a way's decoder returns
(``decodeJSON``).  An annotation is not read: it is not run when the definition is.

A definition is runnable where nothing it runs, itself or a definition it calls, to the end:

- **changes** what it is handed, ``self``, a module's or a class's names: an item or attribute stored or deleted on
  one, a builtin container's changing method on one (:func:`changes.object_changes`' reading, as
  :meth:`changes._Reading.handed_changes` follows a name into what it runs), a ``global`` name;
- **reaches the database** (the reach record's :attr:`ReachNode.database`), or **outside** -- a file, a process,
  another system (:func:`callfacts.change`), ``open`` and ``input``;
- reads **the clock**, a draw or the machine (:data:`callfacts.TIME`, ``DRAW``, ``MACHINE``), **the configuration**
  (:data:`callfacts.CONFIGURATION`), or **a module's or a class's name not a constant**: a constant is bound once,
  where the module or the class is -- its last binding there before it is read -- to a value made of literals and
  constants -- ``singleToken = JobSpec._tagForSH["jobCloning"]``, a class's dict literal, a comprehension over
  constants, a builtin's or a :data:`callfacts.RUNS_ALIKE` call's value of constants, an object a class of the corpus
  builds of constants with a runnable ``__init__`` (``SC_FAILED = StatusCode(1)``), a logger -- and is changed
  nowhere in its module or by its class's methods;
- calls **a library** function or class :data:`callfacts.RUNS_ALIKE` does not hold, a builtin
  :data:`callfacts.BUILTIN_FUNCTIONS` does not say (a builtin exception class built is none), or a method **not
  placed** -- of a receiver of no known class, every method of the corpus of its name is read -- that no table says
  of: a str's
  (:data:`callfacts.TEXT_METHODS`), a builtin container's (:data:`callfacts.BUILTIN_METHODS`), a value's of a
  library's (:data:`callfacts.VALUE_METHODS`), a logger's (:data:`callfacts.LOG_METHODS`: it writes a line, ASSUMPTIONS
  ``logger_keeps_nothing``, and so is a logger made);
- calls a definition that is not runnable (**callee**).

What one call does is :func:`read_call`'s reading, which the trace's steps share (:attr:`TraceStep.calls`).

An ``__init__``'s stores on ``self`` make the object, they change nothing handed.  What it reads of what it is
handed is said (*reads*): ``job.specialHandling``, ``self.splitRule`` -- a field of an
object a reader builds from the run's row; a class's constant (``self.splitRuleToken``) is not.  A definition reached
again on its own way (``unicodeConvert`` calls itself) adds nothing new.
"""

from __future__ import annotations

import ast
import builtins
import re
from typing import Iterable, NamedTuple, Optional

from bamboo.codemap import callfacts
from bamboo.codemap.calls import Calls, Target, own_nodes
from bamboo.codemap.models import (
    CALL_CHANGES,
    CALL_CORPUS,
    CALL_DRAW,
    CALL_HANDED,
    CALL_LIBRARY,
    CALL_LOGGER,
    CALL_MACHINE,
    CALL_OUTSIDE,
    CALL_TIME,
    CALL_UNPLACED,
    CHANGE_DELETE,
    CHANGE_INNER,
    CHANGE_METHOD,
    CHANGE_STORE,
    CHANGE_UNREAD,
    RUN_CALLEE,
    RUN_CHANGES,
    RUN_CLOCK,
    RUN_CONDITION,
    RUN_CONFIG,
    RUN_DATABASE,
    RUN_DECODER,
    RUN_GLOBAL,
    RUN_HOOK,
    RUN_LIBRARY,
    RUN_OUTSIDE,
    RUN_UNPLACED,
    GuardFrom,
    HeldSource,
    JunctionNode,
    ParameterFrom,
    Runnable,
    RunStop,
)
from bamboo.codemap.panda import changes as changes_mod

_CHANGING = (CHANGE_STORE, CHANGE_DELETE, CHANGE_INNER, CHANGE_METHOD)
_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)
#: Builtins that change what they are handed, reach outside, or run text as code.
_BUILTIN_CHANGES = frozenset({"setattr", "delattr"})
_BUILTIN_OUTSIDE = frozenset({"open", "input"})
_CONTAINER_METHODS = callfacts.PUTTING | callfacts.CHANGING | callfacts.TAKING
#: A scope binding nothing: where a builtin's name is read in a body that is not a function's.
_NOWHERE = ast.Module(body=[], type_ignores=[])


class Stop(NamedTuple):
    """Why a definition is not runnable: a ``RUN_*`` kind, ``file:line``, and what."""

    kind: str
    at: str
    what: str = ""


class Verdict(NamedTuple):
    owner: str
    line: int
    stops: tuple[Stop, ...]
    reads: tuple[str, ...]

    @property
    def runnable(self) -> bool:
        return not self.stops


class _Own(NamedTuple):
    """What one definition's own statements say: its stops but the callees', its reads, and the definitions it calls,
    each with the call's place."""

    stops: tuple[Stop, ...]
    reads: tuple[str, ...]
    callees: tuple[tuple[str, Target], ...]


def _key(target: Target) -> tuple:
    return (id(target.func), tuple(id(one[1]) for one in target.bound))


def _owner(target: Target) -> str:
    return f"{target.file}::{getattr(target.func, 'name', '<lambda>')}"


class CallRead(NamedTuple):
    """How one call makes its value (:func:`read_call`): *made* a ``CALL_*`` kind; *what* the builtin's name, the
    library function's spelling, the method's name, or why the call is not placed; *targets* the definitions of the
    corpus it lands at; *thread* whether it starts a thread."""

    made: str
    what: str = ""
    targets: tuple[Target, ...] = ()
    thread: bool = False


_RUN_DEPENDENT = {"time": CALL_TIME, "draw": CALL_DRAW, "machine": CALL_MACHINE}
#: The :class:`Runnable` stop each ``CALL_*`` kind is; a kind not here stops nothing.
_STOP_OF = {
    CALL_TIME: RUN_CLOCK,
    CALL_DRAW: RUN_CLOCK,
    CALL_MACHINE: RUN_CLOCK,
    CALL_CHANGES: RUN_CHANGES,
    CALL_OUTSIDE: RUN_OUTSIDE,
    CALL_LIBRARY: RUN_LIBRARY,
    CALL_UNPLACED: RUN_UNPLACED,
}


def read_call(calls: Calls, file: str, func: ast.AST, call: ast.Call, bound: tuple = ()) -> CallRead:
    """How *call*, in *func* of *file*, makes its value: the one reading a :class:`Runnable`'s stops and a trace
    step's calls (:attr:`TraceStep.calls`) share, so that the two cannot say different things of one call.

    A builtin by its name; a call placed only at definitions of the corpus is theirs; a name outside the corpus by
    the tables of :mod:`callfacts` -- a logger made, :data:`callfacts.TIME` (unless handed its time), ``DRAW`` and
    ``MACHINE``, a change outside, :data:`callfacts.RUNS_ALIKE`; a method no definition is placed for by the tables of
    a str's, a builtin container's, a library value's and a logger's methods, then every method of the corpus of its
    name (a receiver of no known class, as the changes read it); else not placed."""
    types = calls.types
    called = call.func
    if isinstance(called, ast.Name) and types.is_builtin(file, func, called.id):
        if called.id in _BUILTIN_CHANGES:
            return CallRead(CALL_CHANGES, called.id)
        if called.id in _BUILTIN_OUTSIDE:
            return CallRead(CALL_OUTSIDE, called.id)
        if called.id not in callfacts.BUILTIN_FUNCTIONS and not _exception(called.id):
            return CallRead(CALL_LIBRARY, called.id)
        return CallRead(CALL_HANDED, called.id)
    placed = calls.callees(file, func, call, bound)
    targets, thread = tuple(placed.targets), placed.thread
    if placed.targets and not placed.unplaced:
        return CallRead(CALL_CORPUS, "", targets, thread)
    name = callfacts.spelled(types, file, called)
    if name is not None and name.split(".")[0] not in types.roots:
        if name.rsplit(".", 1)[-1] in callfacts.LOGGER_MAKERS:
            # A logger made: it writes lines and keeps nothing (logger_keeps_nothing).
            return CallRead(CALL_LOGGER, name, targets, thread)
        run = callfacts.run_dependent(name, call)
        if run:
            return CallRead(_RUN_DEPENDENT[run], name, targets, thread)
        if callfacts.change(types, file, func, call) is not None:
            return CallRead(CALL_OUTSIDE, name, targets, thread)
        if name in callfacts.TIME_UNLESS_HANDED or callfacts.listed(name, callfacts.RUNS_ALIKE):
            return CallRead(CALL_HANDED, name, targets, thread)
        return CallRead(CALL_LIBRARY, name, targets, thread)
    if isinstance(called, ast.Attribute) and not placed.targets:
        method = called.attr
        if method in callfacts.TEXT_METHODS or method in _CONTAINER_METHODS or method in callfacts.VALUE_METHODS:
            return CallRead(CALL_HANDED, method, (), thread)
        if method in callfacts.LOG_METHODS:
            return CallRead(CALL_LOGGER, method, (), thread)
        named = changes_mod._reading(calls).methods().get(method, [])
        if named:
            return CallRead(CALL_CORPUS, method, tuple(named), thread)
    return CallRead(CALL_UNPLACED, f"{ast.unparse(called)}: {placed.unplaced or 'not placed'}", targets, thread)


class _Judge:
    """The reading, over one build's calls and reach record; each definition's own statements read once.  A
    definition is runnable where no definition it reaches through its calls, itself included, has a stop of its own:
    what definitions calling each other do is what any of them does."""

    def __init__(self, calls: Calls, record, trees: dict[str, ast.AST]):
        self.calls, self.record, self.trees = calls, record, trees
        self.types = calls.types
        self.reading = changes_mod._reading(calls)
        self._own: dict[tuple, _Own] = {}
        self._clean: dict[tuple, bool] = {}
        self._constants: dict[tuple, bool] = {}

    def of(self, target: Target) -> Verdict:
        own = self.own(target)
        stops = list(own.stops)
        for at, callee in own.callees:
            if not self.clean(callee):
                stops.append(Stop(RUN_CALLEE, at, _owner(callee)))
        return Verdict(_owner(target), getattr(target.func, "lineno", 0), tuple(dict.fromkeys(stops)), own.reads)

    def clean(self, target: Target) -> bool:
        """Whether no definition *target* reaches through its calls, itself included, has a stop of its own."""
        key = _key(target)
        held = self._clean.get(key)
        if held is not None:
            return held
        seen = {key: target}
        pending = [target]
        while pending:
            one = pending.pop()
            own = self.own(one)
            if own.stops or self._clean.get(_key(one)) is False:
                self._clean[key] = False
                return False
            for _at, callee in own.callees:
                if _key(callee) not in seen and self._clean.get(_key(callee)) is not True:
                    seen[_key(callee)] = callee
                    pending.append(callee)
        # All it reaches is clean, and so is all each of them reaches.
        for reached in seen:
            self._clean[reached] = True
        return True

    def own(self, target: Target) -> _Own:
        key = _key(target)
        held = self._own.get(key)
        if held is None:
            stops: list[Stop] = []
            reads: list[str] = []
            callees: list[tuple[str, Target]] = []
            try:
                self._read(target, stops, reads, callees)
            except Exception as error:  # noqa: BLE001 -- a reading that fails is a stop, not a failed build
                at = f"{target.file}:{getattr(target.func, 'lineno', 0)}"
                stops.append(Stop(RUN_UNPLACED, at, f"{type(error).__name__}: {error}"))
            held = self._own[key] = _Own(
                tuple(dict.fromkeys(stops)), tuple(dict.fromkeys(reads)), tuple(dict.fromkeys(callees))
            )
        return held

    # -- one definition ------------------------------------------------------

    def _read(
        self, target: Target, stops: list[Stop], reads: list[str], callees: list[tuple[str, Target]]
    ) -> None:
        file, func = target.file, target.func
        here = lambda node: f"{file}:{getattr(node, 'lineno', 0)}"  # noqa: E731
        parameters = _parameters(func)
        local = _bound(func) | parameters
        made = getattr(func, "name", "") == "__init__"
        for name in sorted(parameters):
            if made and name == "self":
                continue  # the object being made, not one handed
            for change in self.reading.handed_changes(target, frozenset({name})):
                if change.kind in _CHANGING:
                    stops.append(Stop(RUN_CHANGES, f"{change.file}:{change.line}", f"{name} {change.kind} {change.key}"))
                elif change.kind == CHANGE_UNREAD:
                    stops.append(Stop(RUN_UNPLACED, f"{change.file}:{change.line}", change.what))
        database = self.record.rows.get(_record_owner(self.record, file, func)) if self.record is not None else None
        for line in database.database if database is not None else ():
            stops.append(Stop(RUN_DATABASE, f"{file}:{line}"))
        skipped = _annotations(func)
        for node in own_nodes(func):
            if id(node) in skipped:
                continue
            if isinstance(node, (ast.Global, ast.Nonlocal)):
                stops.append(Stop(RUN_CHANGES, here(node), f"global {', '.join(node.names)}"))
            elif isinstance(node, (ast.Attribute, ast.Subscript)) and isinstance(node.ctx, (ast.Store, ast.Del)):
                root = _root(node)
                if root is None or root not in local:
                    stops.append(Stop(RUN_CHANGES, here(node), ast.unparse(node)))
            elif isinstance(node, ast.Call):
                self._call(target, node, stops, callees)
            elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                self._attribute(target, node, parameters, stops, reads)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id not in local:
                self._global(file, func, node, stops)

    def _call(
        self, target: Target, call: ast.Call, stops: list[Stop], callees: list[tuple[str, Target]]
    ) -> None:
        at = f"{target.file}:{call.lineno}"
        read = read_call(self.calls, target.file, target.func, call, target.bound)
        if read.thread:
            stops.append(Stop(RUN_OUTSIDE, at, "a thread"))
        callees.extend((at, one) for one in read.targets)
        stop = _STOP_OF.get(read.made)
        if stop is not None:
            stops.append(Stop(stop, at, read.what))

    def _attribute(
        self, target: Target, node: ast.Attribute, parameters: set[str], stops: list[Stop], reads: list[str]
    ) -> None:
        value = node.value
        if isinstance(getattr(node, "parent", None), ast.Call) and node.parent.func is node:  # type: ignore[attr-defined]
            return  # a method called: read as the call
        at = f"{target.file}:{node.lineno}"
        if isinstance(value, ast.Name) and value.id in ("self", "cls"):
            classes = list(target.bound) or (
                [(target.file, target.cls)] if target.cls is not None else []
            )
            if not classes or not all(self._class_attribute(file, cls, node.attr, ()) for file, cls in classes):
                reads.append(f"{value.id}.{node.attr}")
            return
        if isinstance(value, ast.Name) and value.id in parameters:
            reads.append(f"{value.id}.{node.attr}")
            return
        name = callfacts.spelled(self.types, target.file, node)
        if name is None:
            return
        if any(name == one or name.startswith(f"{one}.") for one in callfacts.CONFIGURATION):
            stops.append(Stop(RUN_CONFIG, at, name))
        elif callfacts.listed(name, callfacts.CLOCK):
            stops.append(Stop(RUN_CLOCK, at, name))
        elif name.split(".")[0] in self.types.roots and not self._expression(target.file, None, node, (), frozenset()):
            stops.append(Stop(RUN_GLOBAL, at, ast.unparse(node)))

    def _global(self, file: str, func: ast.AST, node: ast.Name, stops: list[Stop]) -> None:
        if self.types.is_builtin(file, func, node.id):
            return
        name = callfacts.spelled(self.types, file, node)
        if name is not None and name in callfacts.CONFIGURATION:
            stops.append(Stop(RUN_CONFIG, f"{file}:{node.lineno}", name))
        elif not self._module_name(file, node.id, ()):
            stops.append(Stop(RUN_GLOBAL, f"{file}:{node.lineno}", node.id))

    # -- constants -------------------------------------------------------------

    def _module_name(self, file: str, name: str, seen: tuple, before: Optional[int] = None) -> bool:
        """Whether *name* of *file*'s module is a definition, a class, a module, or a constant there -- as the
        module's last binding of it before its statement *before* binds it, its last where *before* is ``None``."""
        key = ("module", file, name, before)
        if key in seen:
            return False
        held = self._constants.get(key)
        if held is None:
            held = self._constants[key] = self._read_module_name(file, name, (*seen, key), before)
        return held

    def _read_module_name(self, file: str, name: str, seen: tuple, before: Optional[int]) -> bool:
        imported = self.types.imports(file).get(name)
        if imported is not None:
            module, attribute = imported
            source = self.types.module_file(module) if attribute else None
            if source is None or self.types.module_file(f"{module}.{attribute}") is not None:
                return True  # a module, or a name of the library's
            return self._module_name(source, attribute, seen)
        tree = self.trees.get(file) or self.types.tree(file)
        if tree is None:
            return False
        last = _last_binding(getattr(tree, "body", []), name, before)
        if last is None:
            return False
        index, value = last
        if value is None:
            return True  # a definition or a class
        return self._expression(file, None, value, seen, frozenset(), index) and not _changed_in(tree, name)

    def _class_attribute(
        self, file: str, cls: ast.ClassDef, attr: str, seen: tuple, before: Optional[int] = None
    ) -> bool:
        """Whether *attr* of *cls* is a method or a constant of the class or a base, no method of *cls* binding it on
        the object -- as the class body's last binding of it before its statement *before*."""
        key = ("class", id(cls), attr, before)
        if key in seen:
            return False
        held = self._constants.get(key)
        if held is None:
            held = self._constants[key] = self._read_class_attribute(file, cls, attr, (*seen, key), before)
        return held

    def _read_class_attribute(
        self, file: str, cls: ast.ClassDef, attr: str, seen: tuple, before: Optional[int]
    ) -> bool:
        for node in ast.walk(cls):
            if isinstance(node, (ast.Attribute, ast.Subscript)) and isinstance(node.ctx, (ast.Store, ast.Del)):
                inner: ast.AST = node
                while isinstance(inner, ast.Subscript):
                    inner = inner.value
                if (
                    isinstance(inner, ast.Attribute)
                    and inner.attr == attr
                    and isinstance(inner.value, ast.Name)
                    and inner.value.id in ("self", "cls", cls.name)
                ):
                    return False
        last = _last_binding(cls.body, attr, before)
        if last is not None:
            index, value = last
            return value is None or self._expression(file, cls, value, seen, frozenset(), index)
        if before is not None:
            return False
        for base in cls.bases:
            named = self.types.class_named(file, base)
            if named is not None and self._class_attribute(named[0], named[1], attr, seen):
                return True
        return False

    def _expression(
        self,
        file: str,
        cls: Optional[ast.ClassDef],
        value: Optional[ast.AST],
        seen: tuple,
        names: frozenset[str],
        at: Optional[int] = None,
    ) -> bool:
        """Whether *value*, bound where *file*'s module or *cls*'s body is (by its statement *at*), is made of
        constants: literals, the names of constants and of *names* (a comprehension's own), a builtin's or
        :data:`callfacts.RUNS_ALIKE`'s value of them, a str's or a builtin container's method that takes a part of
        one."""

        def made(one: Optional[ast.AST], more: frozenset[str] = names) -> bool:
            return self._expression(file, cls, one, seen, more, at)

        if value is None:
            return False
        if isinstance(value, ast.Constant):
            return True
        if isinstance(value, (ast.Tuple, ast.List, ast.Set)):
            return all(made(one) for one in value.elts)
        if isinstance(value, ast.Dict):
            return all(one is not None and made(one) for one in value.keys) and all(made(one) for one in value.values)
        if isinstance(value, ast.BinOp):
            return made(value.left) and made(value.right)
        if isinstance(value, ast.UnaryOp):
            return made(value.operand)
        if isinstance(value, ast.BoolOp):
            return all(made(one) for one in value.values)
        if isinstance(value, ast.Compare):
            return made(value.left) and all(made(one) for one in value.comparators)
        if isinstance(value, ast.IfExp):
            return made(value.test) and made(value.body) and made(value.orelse)
        if isinstance(value, ast.JoinedStr):
            return all(made(one) for one in value.values)
        if isinstance(value, ast.FormattedValue):
            return made(value.value)
        if isinstance(value, ast.Subscript):
            return made(value.value) and made(value.slice)
        if isinstance(value, ast.Slice):
            return all(one is None or made(one) for one in (value.lower, value.upper, value.step))
        if isinstance(value, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            inner = set(names)
            for generator in value.generators:
                if not made(generator.iter, frozenset(inner)):
                    return False
                inner |= {one.id for one in ast.walk(generator.target) if isinstance(one, ast.Name)}
                if not all(made(one, frozenset(inner)) for one in generator.ifs):
                    return False
            parts = [value.key, value.value] if isinstance(value, ast.DictComp) else [value.elt]
            return all(made(one, frozenset(inner)) for one in parts)
        if isinstance(value, ast.Name):
            if value.id in names:
                return True
            if cls is not None and _last_binding(cls.body, value.id, at) is not None:
                return self._class_attribute(file, cls, value.id, seen, at)
            if self.types.is_builtin(file, cls or self.trees.get(file) or _NOWHERE, value.id):
                return True
            return self._module_name(file, value.id, seen, at if cls is None else None)
        if isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name):
            module = self.types.module_named(file, value.value)
            if module is not None:
                return self._module_name(module, value.attr, seen)
            named = self.types.class_named(file, value.value)
            if named is not None:
                return self._class_attribute(named[0], named[1], value.attr, seen)
            return False
        if isinstance(value, ast.Call) and not value.keywords or isinstance(value, ast.Call) and all(
            one.arg is not None for one in value.keywords
        ):
            arguments = [*value.args, *(one.value for one in value.keywords)]  # type: ignore[union-attr]
            if not all(made(one) for one in arguments):
                return False
            called = value.func  # type: ignore[union-attr]
            maker = called.attr if isinstance(called, ast.Attribute) else getattr(called, "id", "")
            if maker in callfacts.LOGGER_MAKERS:
                return True  # a logger: it writes lines and keeps nothing (logger_keeps_nothing)
            built = self.types.class_named(file, called) if isinstance(called, (ast.Name, ast.Attribute)) else None
            if built is not None:
                init = next(
                    (one for one in built[1].body if isinstance(one, _FUNCTIONS) and one.name == "__init__"), None
                )
                return init is None or self.clean(Target(built[0], init, built[1]))
            if isinstance(called, ast.Name) and called.id in callfacts.BUILTIN_FUNCTIONS:
                return called.id not in _BUILTIN_OUTSIDE and self.types.is_builtin(
                    file, cls or self.trees.get(file) or _NOWHERE, called.id
                )
            if isinstance(called, ast.Attribute) and (
                called.attr in callfacts.TEXT_METHODS or called.attr in callfacts.TAKING
            ):
                return made(called.value)
            spelled = callfacts.spelled(self.types, file, called)
            return spelled is not None and callfacts.listed(spelled, callfacts.RUNS_ALIKE)
        return False


def _exception(name: str) -> bool:
    """Whether *name* is a builtin exception class: built to be raised, it changes nothing."""
    held = getattr(builtins, name, None)
    return isinstance(held, type) and issubclass(held, BaseException)


def _last_binding(body: list[ast.stmt], name: str, before: Optional[int]) -> Optional[tuple[int, Optional[ast.expr]]]:
    """The last statement of *body* before *before* binding *name*: its index, and the value an assignment binds --
    ``None`` for a definition, a class, an import."""
    found: Optional[tuple[int, Optional[ast.expr]]] = None
    for index, node in enumerate(body):
        if before is not None and index >= before:
            break
        if isinstance(node, (*_FUNCTIONS, ast.ClassDef)) and node.name == name:
            found = (index, None)
        elif isinstance(node, (ast.Import, ast.ImportFrom)) and any(
            (alias.asname or alias.name.split(".")[0]) == name for alias in node.names
        ):
            found = (index, None)
        elif isinstance(node, ast.Assign) and any(
            isinstance(one, ast.Name) and one.id == name for target in node.targets for one in ast.walk(target)
        ):
            found = (index, node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            found = (index, node.value) if node.value is not None else found
    return found


def _bound_by(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Assign):
        return {one.id for target in node.targets for one in ast.walk(target) if isinstance(one, ast.Name)}
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return {node.target.id}
    if isinstance(node, _FUNCTIONS):
        return {node.name}
    return set()


def _changed_in(tree: ast.AST, name: str) -> bool:
    """Whether *tree* binds *name* again with ``global``, or stores, deletes or changes an item or attribute on it."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Global) and name in node.names:
            return True
        if isinstance(node, (ast.Attribute, ast.Subscript)) and isinstance(node.ctx, (ast.Store, ast.Del)):
            if _root(node) == name:
                return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and _root(node.func.value) == name:
            if node.func.attr in callfacts.CHANGING:
                return True
    return False


def _annotations(func: ast.AST) -> set[int]:
    """The ids of the nodes of *func*'s annotations: its parameters', its return's, its locals'."""
    found: list[ast.AST] = []
    arguments = getattr(func, "args", None)
    if arguments is not None:
        for one in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs, arguments.vararg, arguments.kwarg):
            if one is not None and one.annotation is not None:
                found.append(one.annotation)
    if getattr(func, "returns", None) is not None:
        found.append(func.returns)  # type: ignore[attr-defined]
    for node in own_nodes(func):
        if isinstance(node, ast.AnnAssign):
            found.append(node.annotation)
    return {id(node) for one in found for node in ast.walk(one)}


def _root(node: ast.AST) -> Optional[str]:
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _parameters(func: ast.AST) -> set[str]:
    arguments = getattr(func, "args", None)
    if arguments is None:
        return set()
    names = {one.arg for one in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs)}
    return names | {one.arg for one in (arguments.vararg, arguments.kwarg) if one is not None}


def _bound(func: ast.AST) -> set[str]:
    found: set[str] = set()
    for node in own_nodes(func):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            found.add(node.id)
        elif isinstance(node, (*_FUNCTIONS, ast.ClassDef)) and node is not func:
            found.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            found.update((alias.asname or alias.name.split(".")[0]) for alias in node.names)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            found.add(node.name)
    return found


def _record_owner(record, file: str, func: ast.AST) -> Optional[str]:
    owner: Optional[str] = record.function_at(file, getattr(func, "lineno", 0))
    return owner


_MARK = re.compile(r"\s+\[(in (?P<callee>\w+)\(\)|at \d+)\]$")


def asked(
    junctions: Iterable[JunctionNode], trees: dict[str, ast.AST], calls: Calls
) -> tuple[dict[tuple, tuple[Target, list[str]]], int]:
    """Each definition a call in an arm's path condition can land at, with each such call's place; and how many
    calls of the conditions were found in no function they are read in.  A condition is read in the arm's function,
    one marked ``[in f()]`` in each function of that name the corpus has."""
    from bamboo.codemap.panda.heldfrom import _function_at

    by_name: dict[str, list[tuple[str, ast.AST]]] = {}
    for file, tree in trees.items():
        for node in ast.walk(tree):
            if isinstance(node, _FUNCTIONS):
                by_name.setdefault(node.name, []).append((file, node))
    spelled: dict[int, dict[str, list[ast.Call]]] = {}

    def calls_in(func: ast.AST) -> dict[str, list[ast.Call]]:
        held = spelled.get(id(func))
        if held is None:
            held = spelled[id(func)] = {}
            for node in own_nodes(func):
                if isinstance(node, ast.Call):
                    held.setdefault(ast.unparse(node), []).append(node)
        return held

    found: dict[tuple, tuple[Target, list[str]]] = {}
    missed = 0
    for junction in junctions:
        file, _, name = junction.owner.partition("::")
        owner_tree = trees.get(file)
        for branch in junction.branches:
            held_in: Optional[ast.AST] = (
                _function_at(owner_tree, name, branch.line)
                if owner_tree is not None and branch.line is not None
                else None
            )
            if held_in is None:
                continue
            found_in: ast.AST = held_in
            for condition in branch.path_condition:
                mark = _MARK.search(condition)
                text = condition[: mark.start()] if mark is not None else condition
                scopes = by_name.get(mark.group("callee"), []) if mark is not None and mark.group("callee") else [
                    (file, found_in)
                ]
                try:
                    parsed = ast.parse(text, mode="eval")
                except SyntaxError:
                    continue
                for node in ast.walk(parsed):
                    if not isinstance(node, ast.Call):
                        continue
                    spelling = ast.unparse(node)
                    places = [(where, scope, one) for where, scope in scopes for one in calls_in(scope).get(spelling, [])]
                    if not places and not (isinstance(node.func, ast.Name) and node.func.id in callfacts.BUILTIN_FUNCTIONS):
                        missed += 1
                    for where, scope, one in places:
                        for target in calls.callees(where, scope, one).targets:
                            key = (id(target.func), tuple(id(bound[1]) for bound in target.bound))
                            held = found.setdefault(key, (target, []))
                            place = f"{where}:{one.lineno}"
                            if place not in held[1]:
                                held[1].append(place)
    return found, missed


def runnables(
    junctions: Iterable[JunctionNode],
    trees: dict[str, ast.AST],
    calls: Optional[Calls],
    record,
    guard_froms: Iterable[GuardFrom] = (),
    parameter_froms: Iterable[ParameterFrom] = (),
) -> list[Runnable]:
    """A :class:`Runnable` for each definition a call in an arm's path condition can land at, each a JSON maker on a
    guard's way is handed as a hook, and each returning a JSON maker's value on a way.  Nothing without *calls* or
    *record*: the reading places calls, and the record says where the database is reached."""
    if calls is None or record is None:
        return []
    wanted: dict[tuple, tuple[Target, list[str], list[str]]] = {}

    def add(target: Target, why: str, place: str) -> None:
        key = (id(target.func), tuple(id(one[1]) for one in target.bound))
        held = wanted.setdefault(key, (target, [], []))
        if why not in held[1]:
            held[1].append(why)
        if place and place not in held[2]:
            held[2].append(place)

    conditions, _missed = asked(junctions, trees, calls)
    for target, places in conditions.values():
        for place in places:
            add(target, RUN_CONDITION, place)
    for source in _ways([*guard_froms, *parameter_froms]):
        for call in source.through:
            for hook in call.hooks:
                for target in _defined(trees, hook):
                    add(target, RUN_HOOK, call.at)
            file, _, line = call.at.rpartition(":")
            tree = trees.get(file)
            if call.gives == "decoded" and tree is not None and any(
                isinstance(node, ast.Return) and node.lineno == int(line) for node in ast.walk(tree)
            ):
                owner = record.function_at(file, int(line))
                for target in _defined(trees, owner) if owner is not None else ():
                    add(target, RUN_DECODER, call.at)
    judge = _Judge(calls, record, trees)
    found: list[Runnable] = []
    for target, why, places in wanted.values():
        verdict = judge.of(target)
        found.append(
            Runnable(
                owner=verdict.owner,
                line=verdict.line,
                bound=[f"{file}::{cls.name}" for file, cls in target.bound],
                asked=why,
                asked_at=places,
                runnable=verdict.runnable,
                reads=list(verdict.reads),
                stops=[RunStop(kind=one.kind, at=one.at, what=one.what) for one in verdict.stops],
            )
        )
    return sorted(found, key=lambda one: (one.owner, one.line, one.bound))


def _ways(records: Iterable) -> Iterable[HeldSource]:
    """Every way of *records*, the ways of their changes' values too."""
    pending = [source for one in records for source in one.sources]
    while pending:
        source = pending.pop()
        yield source
        for change in source.changes:
            pending.extend(change.value)


def _defined(trees: dict[str, ast.AST], owner: str) -> list[Target]:
    """The definitions *owner* (``file::name``, ``file::Class.name``, ``#n`` after a repeated one) names: each
    function of that name in the file, in that class where it names one."""
    file, _, name = owner.partition("::")
    tree = trees.get(file)
    if tree is None:
        return []
    qualified = name.split("#")[0].split(".")
    bare, holder = qualified[-1], (qualified[-2] if len(qualified) > 1 else None)
    found: list[Target] = []
    for cls in [tree, *(one for one in ast.walk(tree) if isinstance(one, ast.ClassDef))]:
        if holder is not None and getattr(cls, "name", None) != holder:
            continue
        for node in getattr(cls, "body", ()):
            if isinstance(node, _FUNCTIONS) and node.name == bare:
                found.append(Target(file, node, cls if isinstance(cls, ast.ClassDef) else None))
    return found
