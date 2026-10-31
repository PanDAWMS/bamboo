"""Where a call lands, what it is handed, and what comes back.

The walk crossed a function boundary in two ways only: a handover the map had
recorded, and a ``self.<field>`` answered from the class's other methods.  It
never followed a value into the function that returned it or out to the call
that passed it in, so every cross-function shape the corpus has was either a
rule of its own -- a literal returned one hop up, a mapping read out of a
zero-argument method, ``.add()`` on a bare name -- or a stop.  Python has no
end of those shapes, and each case found the next one.

This module is the one place a call is read, three ways:

* :meth:`Calls.callees` -- which function definitions a call can land in,
  read the way Python finds them: ``self.m`` and ``obj.m`` through the method
  resolution order of each class the object can be -- the method's class or
  what ``obj`` is declared to be, and their subclasses -- a bare name through
  the enclosing function, the module and its imports, ``Class(...)`` as its
  ``__init__``, ``thread.start()`` as its ``run``.  Never by the method's name alone -- a crossing that matched
  callees by bare name gave 222 read sites the same two false leads -- so a
  call it cannot place comes back with the reason, not a guess.
* :meth:`Calls.bind` -- which expression at the call each parameter holds.
* :meth:`Calls.returns` -- what the function hands back, slot by slot.

What a method changes on its object is read by :meth:`Calls.writes`, so that
``self.result.set_fatal()`` is a write of ``status_code`` like an assignment.

It spends none of the walk's budget, like the declared-type reading it is
built on, and reads the same trees -- :func:`parsed`.
"""

from __future__ import annotations

import ast
import builtins
import hashlib
import json
import re
import weakref
from pathlib import Path
from typing import Callable, Iterable, Iterator, NamedTuple, Optional

from bamboo.codemap import callfacts
from bamboo.codemap.astwalk import walk
from bamboo.codemap.panda import pathcond

#: Why a call could not be placed.  Each is a class of the census, so the
#: count of calls the walk could not enter says what would enter them.
UNPLACED_OUTSIDE = "outside the corpus"
UNPLACED_RECEIVER = "the receiver's class is not declared"
UNPLACED_FORWARDED = "the receiver forwards every call through __getattr__"
UNPLACED_NO_METHOD = "the declared class has no such method"
#: ``self.conn.close()`` on ``WrappedOracleConn``: the class defines no
#: ``close``, and its ``__getattribute__`` hands the name to the connection it
#: wraps -- as a type checker reads it, the name may be supplied there.  A
#: method the class does define is still placed: ``JobSpec``'s
#: ``__getattribute__`` filters values and supplies no name of its own.
UNPLACED_GETATTRIBUTE = "the declared class has no such method, and __getattribute__ may supply one"
UNPLACED_DYNAMIC = "the callee is chosen at run time"
UNPLACED_NAME = "the name is bound to something that is not a function here"
#: ``Interaction.JEDITemporaryError(...)``, ``DataCarouselRequestSpec()``: the
#: class and its bases in the corpus define no ``__init__``, so what building
#: one runs is ``object``'s, ``Exception``'s or a base's outside the corpus.
UNPLACED_NO_INIT = "the class it builds has no __init__ in the corpus"
#: Beside targets, not instead of them: the receiver is one of these classes
#: or another one some way of building it does not show -- or it is one of
#: them that supplies the method only by forwarding the call elsewhere.  Said
#: on its own like the others -- :attr:`TraceStep.open` holds nothing else --
#: and not as a clause that reads only after the targets are named.
UNPLACED_OPEN = "the call may also land somewhere this reading cannot name"
#: What a class says of a name it supplies only by forwarding, the reason
#: that says more first.
_FORWARDING = (UNPLACED_FORWARDED, UNPLACED_GETATTRIBUTE)

#: Where :meth:`Types.candidates` says a value is an instance of a builtin
#: type -- ``tmp_list = []``, ``seen = set()`` -- as ``(BUILTIN, "list")``.
#: Not a file: no class of the corpus is defined there, so no method of one is
#: a call's target.
BUILTIN = "<builtins>"
#: The names Python's builtins hold: a call of one no scope binds is the builtin's (:meth:`Types.is_builtin`).
_BUILTIN_NAMES = frozenset(dir(builtins))
#: The standard library's container classes, by the module they are imported from (:data:`callfacts.CONTAINERS`).
_LIBRARY_CONTAINERS = {(one.module, name) for name, one in callfacts.CONTAINERS.items() if one.module == "collections"}
#: The library classes whose ``start()`` runs the object's ``run`` on another
#: thread or process (:data:`callfacts.STARTERS`).  A class built on one that
#: defines ``run`` in the corpus has its ``start()`` placed there (:attr:`Placed.thread`).
_STARTERS = callfacts.STARTERS
#: The standard library's copies, and the method a class defines to copy
#: itself some other way.  ``copy.copy(x)`` is an object of ``x``'s class --
#: typeshed states ``def copy(x: _T) -> _T`` -- unless that class defines the
#: hook, and then it is what the hook returns.  The map reads the corpus's
#: annotations and not typeshed's, so it is said here.
_COPIES = callfacts.COPIES


class RecordedClass(NamedTuple):
    """A class in an instance's record, by the dotted path Python prints it with."""

    dotted: str


#: An instance's recorded run-time values of module names, ``{(file, name): value}`` -- what a configuration made of
#: them, which the code alone cannot say: ``panda_config.g_pluginMap`` is filled by importing the classes the
#: configuration names, and a name that does not import is passed over (:func:`read_recorded`).
Recorded = dict[tuple[str, str], object]


def _recorded_value(value: object) -> object:
    if isinstance(value, dict):
        if set(value) == {"class"}:
            return RecordedClass(str(value["class"]))
        return {key: _recorded_value(one) for key, one in value.items()}
    if isinstance(value, list):
        return [_recorded_value(one) for one in value]
    return value


def read_recorded(path: Path) -> tuple[Recorded, str]:
    """The record of an instance at *path*, and its stamp ``<path>: <instance>, <n> value(s), <digest>``.

    A JSON object: ``instance`` (which deployment), ``taken`` (by whom, when, by what command) and ``values``, a list
    of ``{"file", "name", "value"}`` -- the name's value in that module once the code's own setup has run, a class
    written ``{"class": "<dotted path>"}``.  Another instance, another experiment's included, may configure other
    classes: a build without a record reads the names as the code alone says them.  A name no module-level
    assignment of the corpus binds stops :class:`Types`: a record of another tree.
    """
    raw = path.read_bytes()
    held = json.loads(raw)
    recorded: Recorded = {
        (str(one["file"]), str(one["name"])): _recorded_value(one["value"]) for one in held.get("values", [])
    }
    stamp = f"{path}: {held.get('instance', 'an instance not named')}, {len(recorded)} value(s), "
    return recorded, stamp + hashlib.sha256(raw).hexdigest()[:16]


#: Parsed modules shared by every reading of the same files.  A walk is made
#: once per arm, per statement and per printed line, and the corpus index
#: behind :meth:`Calls.callers` would otherwise be rebuilt for each.  Keyed by
#: the file's stat, so a tree edited under a reading is read afresh.
_PARSED: dict[tuple[str, int, int], Optional[tuple[str, ast.Module]]] = {}
#: ``callers`` index and answers, per set of roots.
_INDEXES: dict[tuple, dict[str, list["CallSite"]]] = {}
_CALLERS: dict[tuple, list["CallSite"]] = {}
#: The other names each name is imported or bound as, the locals and fields
#: each is bound to, and each class's subclasses, per set of roots --
#: :meth:`Calls.spellings`.
_ALIASES: dict[tuple, dict[str, set[str]]] = {}
_HOLDERS: dict[tuple, dict[str, set[str]]] = {}
_SUBCLASSES: dict[tuple, dict[tuple[str, int], list[tuple[str, ast.ClassDef]]]] = {}
#: Where each class is read as a value, per set of roots -- :meth:`Calls.read_as_value`.
_VALUES: dict[tuple, dict[tuple[str, int], list[str]]] = {}


def parsed(path: Path) -> Optional[tuple[str, ast.Module]]:
    """``(source, tree)`` for *path*, parents attached; ``None`` where it will not read or parse.

    **One tree per file.**  The index behind :meth:`Calls.callers` holds the
    nodes of whatever parse of a file existed when it was built, and a method
    is found by the identity of its node; the walk used to parse each file
    again for itself.  An index built before the walk reached
    ``JediTaskBuffer.py`` held the other parse's methods, ``self.proxyPool``
    could not be typed in them, and the DBProxy methods lost their callers --
    by which walks had run before in the process: 9 of 144 sampled print
    walks.  So every reading takes its tree from here, and the walk's
    re-parsing of each file went with it.
    """
    try:
        stat = path.stat()
    except OSError:
        return None
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    if key not in _PARSED:
        try:
            source = path.read_text(errors="replace")
            tree = ast.parse(source, filename=str(path))
        except (OSError, SyntaxError, ValueError):
            _PARSED[key] = None
        else:
            pathcond.attach_parents(tree)
            _PARSED[key] = (source, tree)
    return _PARSED[key]


# Test code is not the system's behaviour.  It writes the same attributes with
# the same literals, so leaving it in files fixtures as junctions and doubles
# the ambiguous attribution cases -- 88 of 376 modules, and 47% of the writes
# whose spec class could not be settled came from them.  A junction pointing at
# a test would send an investigation to code that never runs in production;
# so would a caller: a JEDI test script's own body builds ``JobGeneratorThread``
# and calls ``doGenerate``, and listed among a function's callers it pushed the
# daemons' calls past the count a list names.
_TEST_PATH = re.compile(r"(^|/)(tests?|jeditest)(/|$)", re.IGNORECASE)
_TEST_FILE = re.compile(r"(^test_|_tests?$)", re.IGNORECASE)


def is_test_module(rel_path: str) -> bool:
    """True when *rel_path* is test code rather than system behaviour."""
    return bool(_TEST_PATH.search(rel_path) or _TEST_FILE.search(Path(rel_path).stem))


def _roots_key(roots: dict[str, Path]) -> tuple:
    return tuple(sorted((package, str(root)) for package, root in roots.items()))


#: What :meth:`Calls.route_constant` answers where the route fixes no constant.
UNSET = object()


class Displayed(list):
    """What :meth:`Calls.route_constant` answers for a list, tuple or set display a route hands: as many items as it
    spells, none of them read.

    Not ``None``, and true exactly when it spells an item -- what a test on a flag parameter asks of
    ``simTasks=[tmpTaskItem]`` (TaskBroker 259) under ``if simTasks is not None``.  An item read as a value is
    unknown (:func:`~bamboo.codemap.panda.sqleval.from_py` reads one so).  Equal to no string, number or ``None``;
    against a container its items would decide, so it raises the ``TypeError``
    :func:`~bamboo.codemap.panda.pathcond.decided` reads as undecided.  Spelled as a list of as many ``...``: a
    route's constants are kept as their reprs, a record, not words.
    """

    def __init__(self, items: int) -> None:
        super().__init__(object() for _ in range(items))

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if isinstance(other, (list, tuple, set, frozenset, dict)):
            raise TypeError("a display's items are not read")
        return False

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return "[" + ", ".join(["..."] * len(self)) + "]"


class Target(NamedTuple):
    """One function definition a call can land in, and the classes its object can be when it runs there.

    *cls* is the class that defines the method.  *bound* is each class the
    object can be when the call runs this definition -- the class the method
    was looked up for, not the one it was found in.  ``thr.start()`` on a
    ``JobGeneratorThread`` runs ``WorkerThread.run`` as JobGeneratorThread's,
    so ``self.runImpl()`` in it is JobGeneratorThread's: inheritance is read
    to find the definition, and the object stays what it was.  Empty where the
    call does not say -- a function, a method read off its class on an object
    other than ``self`` -- and then a call on ``self`` in it is placed for
    every class ``self`` can be there (:meth:`Calls.objects_of`).
    """

    file: str
    func: ast.AST
    cls: Optional[ast.ClassDef]
    bound: tuple[tuple[str, ast.ClassDef], ...] = ()

    @property
    def owner(self) -> str:
        return f"{self.file}::{self.func.name}"  # type: ignore[attr-defined]


class Placed(NamedTuple):
    """Where a call lands: the definitions, or why none is named.

    *on_class*: the call reads its method off a class --
    ``JobBrokerBase.__init__(self, ...)`` -- so a method that takes the
    object is handed it as the call's first argument (:func:`bind`).

    *thread*: the call starts the definitions on another thread or process
    -- ``thr.start()`` on a ``threading.Thread`` runs the object's ``run``
    there (:data:`_STARTERS`).  It returns at once, and nothing a definition
    returns comes back through it: a reader of what a call returns takes
    nothing from these.

    *forwarded*: the call is made on an object built with a function, and the
    method called hands that function its own arguments
    (:meth:`Calls.forwarding`) -- ``timed_method.run(job_id, ...)`` after
    ``timed_method = TimedMethod(global_task_buffer.updateJobStatus, t)``.
    The targets are the function's, handed the call's arguments as a call of
    it would (:func:`bind`); these are the expressions the function was
    handed as, whose receiver is the object it runs on (:meth:`Calls.made_on`).
    The method returns nothing, and nothing the function returns comes back
    through the call.

    *kept_at*: for a call *forwarded*, the field of the object the method
    keeps what the function returns in -- ``TimedMethod.__call__``'s
    ``self.result = self.method(*var, **kwargs)``, so ``timed_method.result``
    holds what ``updateJobStatus`` returned once ``timed_method.run(...)``
    has; empty where it keeps nothing, or the classes the object can be keep
    it in different fields.
    """

    targets: tuple[Target, ...] = ()
    unplaced: str = ""
    on_class: bool = False
    thread: bool = False
    forwarded: tuple[ast.expr, ...] = ()
    kept_at: str = ""

    @property
    def returns_none(self) -> bool:
        """Whether the call's value is ``None`` whatever the targets return: *thread* or *forwarded*."""
        return self.thread or bool(self.forwarded)


#: The owner of a call in a module's own body -- the walk's spelling of the
#: scope a module binds its names in (``trace.MODULE_SCOPE``).
MODULE_SCOPE = "module-scope"

#: How :meth:`Calls.missed` says a function is called from nowhere in the corpus.
_OUTSIDE = "no call in the corpus names it -- reached from outside"
#: ... and that the calls naming it all land outside the corpus.
_LANDS_OUTSIDE = "no call this reading places lands at it"
#: ... and that its class is held as a value something may build from -- :meth:`Calls.read_as_value`.
_READ_AS_VALUE = "no call in the corpus names it; the class is read as a value at {}, and a call of that value may build one"
#: How many of those reads it names before it counts the rest.
_READS = 5


class Route(NamedTuple):
    """One way calls reach a function: the call into it first, out to where the values are written -- :meth:`Calls.routes`."""

    sites: tuple["CallSite", ...]

    @property
    def chain(self) -> tuple[str, ...]:
        """``file:line`` of each call, the call into the function first."""
        return tuple(f"{site.file}:{site.call.lineno}" for site in self.sites)


class CallSite(NamedTuple):
    """One call, in the function that makes it -- or in the module's own body, *func* the module.

    A call in the module's body runs when the file is imported, or under
    ``if __name__ == "__main__":`` when it is run as a script: ``taskBuffer =
    TaskBuffer()`` at the foot of ``TaskBuffer.py``.  It has no caller of its
    own and no parameters, so what it hands a function is written there.
    """

    file: str
    func: ast.AST
    call: ast.Call

    @property
    def in_module(self) -> bool:
        return isinstance(self.func, ast.Module)

    @property
    def owner(self) -> str:
        return f"{self.file}::{MODULE_SCOPE if self.in_module else self.func.name}"  # type: ignore[attr-defined]


def own_nodes(scope: ast.AST):
    """Every node of *scope*'s own body; a nested function or class runs elsewhere."""
    for child in ast.iter_child_nodes(scope):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        yield child
        yield from own_nodes(child)


def bound_defs(scope: ast.AST) -> Iterator[ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef]:
    """The ``def`` and ``class`` statements *scope*'s own body runs -- under an ``if`` or a ``try`` too -- each binding
    its name there; one nested in another function or class binds it in that one."""
    for child in ast.iter_child_nodes(scope):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield child
        elif not isinstance(child, ast.Lambda):
            yield from bound_defs(child)


#: The names each scope binds (:func:`_scope_names`), kept while its tree is.
_SCOPES: "weakref.WeakKeyDictionary[ast.AST, frozenset[str]]" = weakref.WeakKeyDictionary()


def _scope_names(scope: ast.AST) -> frozenset[str]:
    """The names *scope* binds.  A function: its parameters, the names its own body stores or imports, the ``def``
    and ``class`` statements it runs.  A module: an assignment at its level, any import in it (the names
    :meth:`Types.imports` reads), a ``def`` or ``class`` it runs."""
    if scope in _SCOPES:
        return _SCOPES[scope]
    found = {one.name for one in bound_defs(scope)}
    if isinstance(scope, ast.Module):
        for node in _module_statements(scope.body):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                found |= {target.id for target in targets if isinstance(target, ast.Name)}
        for node in walk(scope):
            if isinstance(node, ast.Import):
                found |= {alias.asname or alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                found |= {alias.asname or alias.name for alias in node.names}
    else:
        arguments = getattr(scope, "args", None)
        every = (
            [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs, arguments.vararg, arguments.kwarg]
            if arguments is not None
            else []
        )
        found |= {one.arg for one in every if one is not None}
        for node in own_nodes(scope):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                found.add(node.id)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                found |= {(alias.asname or alias.name).split(".")[0] for alias in node.names}
    _SCOPES[scope] = frozenset(found)
    return _SCOPES[scope]


def is_builtin_in(func: ast.AST, module: Optional[ast.Module], name: str) -> bool:
    """Whether *name*, read in *func* of *module*, is Python's builtin of the name: one ``builtins`` holds that
    neither the function -- a parameter, a name its own body binds or imports, a ``def`` or ``class`` it runs -- nor
    the module -- a ``def``, a ``class``, an import, an assignment at its level -- binds."""
    if name not in _BUILTIN_NAMES:
        return False
    if not isinstance(func, ast.Module) and name in _scope_names(func):
        return False
    return module is None or name not in _scope_names(module)


def is_builtin_at(node: ast.AST, name: str) -> bool:
    """:func:`is_builtin_in` for *node* where no :class:`Types` is at hand: the function and the module that hold it,
    read off its parents (:func:`pathcond.attach_parents`)."""
    func: Optional[ast.AST] = None
    here: Optional[ast.AST] = node
    while here is not None and not isinstance(here, ast.Module):
        if func is None and isinstance(here, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func = here
        here = getattr(here, "parent", None)
    if here is None:
        raise ValueError(f"no module holds {ast.dump(node)[:80]}: its tree has no parents")
    return is_builtin_in(func if func is not None else here, here, name)


def binding(
    defs: Iterable[ast.stmt], name: str
) -> Optional[ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef]:
    """The ``def`` or ``class`` among *defs* -- one scope's statements, in source order -- that *name* is bound to
    once the scope has run: the last that defines it, as Python binds it.

    ``@overload`` stubs come before the definition that replaces them (``CoreUtils.clean_user_id``'s three,
    ``_make_dicts_tasks_datasets_with_unprocessed_inputs``' two).  A property's ``@<name>.setter`` makes a property
    from the one before, so it stands for that one, the getter.  Defined in both arms of an ``if`` or a ``try``,
    the name is the one that ran; the last in source order is taken.
    """
    found = None
    for node in defs:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) or node.name != name:
            continue
        if found is None or not _extends_property(node):
            found = node
    return found


def _extends_property(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> bool:
    return any(
        isinstance(decorator, ast.Attribute)
        and decorator.attr in ("setter", "getter", "deleter")
        and isinstance(decorator.value, ast.Name)
        and decorator.value.id == node.name
        for decorator in node.decorator_list
    )


def _function_binding(defs: Iterable[ast.stmt], name: str) -> Optional[ast.FunctionDef | ast.AsyncFunctionDef]:
    """:func:`binding` among the function definitions of *defs*."""
    found = binding((node for node in defs if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))), name)
    return found if isinstance(found, (ast.FunctionDef, ast.AsyncFunctionDef)) else None


def returns(func: ast.AST) -> list[ast.expr]:
    """The values *func* hands back, in source order; a bare ``return`` is ``None``."""
    return [
        node.value if node.value is not None else ast.Constant(value=None)
        for node in own_nodes(func)
        if isinstance(node, ast.Return)
    ]


def _is_static(func: ast.AST) -> bool:
    return any(
        isinstance(decorator, ast.Name) and decorator.id == "staticmethod"
        for decorator in getattr(func, "decorator_list", [])
    )


def _is_classmethod(func: ast.AST) -> bool:
    return any(
        isinstance(decorator, ast.Name) and decorator.id == "classmethod"
        for decorator in getattr(func, "decorator_list", [])
    )


def _positional(target: Target, on_class: bool) -> list[ast.arg]:
    """The parameters a call to *target* fills by position.

    Past ``self`` or ``cls`` where the call names the object it is made on.
    A method read off its class (*on_class*, :class:`Placed`) is handed its
    object by position like any other argument --
    ``TaskRefinerBase.extractCommon(self, jediTaskID, ...)`` hands ``self``
    to ``self``, not to ``jediTaskID`` -- while a classmethod is handed its
    class either way.
    """
    args = target.func.args  # type: ignore[attr-defined]
    positional = [*args.posonlyargs, *args.args]
    if (
        target.cls is not None
        and not _is_static(target.func)
        and positional
        and not (on_class and not _is_classmethod(target.func))
    ):
        positional = positional[1:]
    return positional


def bind(call: ast.Call, target: Target, on_class: bool = False) -> dict[str, Optional[ast.expr]]:
    """``{parameter: the expression the call hands it}``; ``None`` where it is not said.

    Positional, keyword and default, in Python's order.  A ``*args`` or
    ``**kwargs`` at the call makes every parameter it could fill unknown
    rather than guessed; a parameter the call leaves to its default holds the
    default.  ``self`` is left out of a method reached through an instance,
    and bound to the first argument of one read off its class (*on_class*,
    :func:`_positional`).
    """
    args = target.func.args  # type: ignore[attr-defined]
    positional = _positional(target, on_class)
    defaults = dict(
        zip([p.arg for p in positional][len(positional) - len(args.defaults):], args.defaults, strict=True)
    )
    passed = list(call.args)
    spread = any(isinstance(one, ast.Starred) for one in passed)
    named = {keyword.arg: keyword.value for keyword in call.keywords if keyword.arg}
    spread_named = any(keyword.arg is None for keyword in call.keywords)
    bound: dict[str, Optional[ast.expr]] = {}
    for index, parameter in enumerate(positional):
        if not spread and index < len(passed) and not isinstance(passed[index], ast.Starred):
            bound[parameter.arg] = passed[index]
        elif parameter.arg in named:
            bound[parameter.arg] = named[parameter.arg]
        elif spread or spread_named:
            bound[parameter.arg] = None
        else:
            bound[parameter.arg] = defaults.get(parameter.arg)
    for parameter, default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
        if parameter.arg in named:
            bound[parameter.arg] = named[parameter.arg]
        else:
            bound[parameter.arg] = None if spread_named else default
    return bound


def said(call: ast.Call, target: Target, on_class: bool = False) -> set[str]:
    """The parameters *call* fills itself, by position or by keyword -- not by default.

    ``self``/``cls`` is not among them: the call names the object it is made
    on, which is a different thing from passing it -- unless the method is
    read off its class (*on_class*), where the object is passed like the
    rest.
    """
    positional = _positional(target, on_class)
    filled = {
        parameter.arg
        for index, parameter in enumerate(positional)
        if index < len(call.args) and not any(isinstance(one, ast.Starred) for one in call.args[: index + 1])
    }
    filled |= {keyword.arg for keyword in call.keywords if keyword.arg}
    return filled


def receiver_parameter(target: Target) -> Optional[str]:
    """``self`` or ``cls`` -- the parameter a method call fills with the object it is made on."""
    args = target.func.args  # type: ignore[attr-defined]
    positional = [*args.posonlyargs, *args.args]
    if target.cls is None or _is_static(target.func) or not positional:
        return None
    return positional[0].arg


class Calls:
    """Calls read against one tree, with a corpus index built the first time it is asked."""

    def __init__(self, types: Types) -> None:
        self.types = types
        #: By the call, and by the classes ``self`` is bound to where the call reads it.
        self._placed: dict[object, Placed] = {}
        self._by_name: Optional[dict[str, list[CallSite]]] = None
        self._callers: dict[tuple[str, int], list[CallSite]] = {}
        self._missed: dict[tuple[str, int], list[str]] = {}
        #: The parameters :meth:`handed_classes` is reading now: one asked
        #: again on the way is a cycle, and says nothing.
        self._handing: set[tuple[str, int, str]] = set()
        #: :meth:`forwarding` by class, the locals :meth:`_wrapped` reads, and
        #: what :meth:`_wrappings` finds across the corpus.
        self._forwarders: dict[tuple[str, int], dict[str, tuple[str, str]]] = {}
        self._wraps: dict[tuple[int, str], Optional[tuple[list[tuple[tuple[str, ast.ClassDef], ast.Call]], list[ast.Call]]]] = {}
        self._wrapping_names: Optional[set[str]] = None
        self._wrapping_classes: Optional[set[str]] = None
        self._wrapped_by: Optional[list[tuple[str, ast.AST, ast.Call, ast.expr, list[ast.Call]]]] = None
        self._wrapped_spellings: Optional[dict[str, set[str]]] = None
        self._forwarded_reads: Optional[set[int]] = None
        #: :meth:`stored` by the store, each call kept for as long as this reading is.
        self._stores: dict[int, ast.Call] = {}
        types.handed_to = self.handed_classes

    # -- where a call lands -------------------------------------------------

    def callees(
        self, file: str, func: ast.AST, call: ast.Call, bound: tuple[tuple[str, ast.ClassDef], ...] = ()
    ) -> Placed:
        """Where *call*, made in *func* of *file*, lands.

        *bound*: the classes ``self`` can be in *func* -- the :attr:`Target.bound`
        a reader entered *func* with.  Without it, every class ``self`` can be
        there (:meth:`objects_of`).  Only a call that reads ``self`` -- on
        ``self`` or ``cls``, through ``super()``, or handing ``self`` to a
        method read off its class -- is placed differently for it.
        """
        key: object = (id(call), _bound_key(bound)) if bound and _reads_self(call) else id(call)
        if key not in self._placed:
            self._placed[key] = self._place(file, func, call, bound if bound and _reads_self(call) else ())
        return self._placed[key]

    def stored(self, store: ast.Attribute | ast.Call, name: ast.expr, value: Optional[ast.expr]) -> ast.Call:
        """The call Python makes for an attribute store -- ``x.a = value``, or ``setattr(x, name, value)``, the call
        *store* -- ``type(x).__setattr__(x, name, value)``.

        Spelled ``x.__setattr__(name, value)``, its receiver the store's, so
        :meth:`callees` places it as it places any method call on ``x`` -- at a
        corpus class's ``__setattr__`` where ``x`` can be one that has it, at
        nothing where its classes take ``object``'s.  No call of the corpus is
        it, and none is: :meth:`callers` does not find it.  One call per store,
        kept here, because a placing is kept by the call's id.
        """
        held = self._stores.get(id(store))
        if held is None:
            receiver = store.args[0] if isinstance(store, ast.Call) else store.value
            held = ast.Call(
                func=ast.Attribute(value=receiver, attr="__setattr__", ctx=ast.Load()),
                args=[name, value if value is not None else ast.Constant(value=None)],
                keywords=[],
            )
            ast.copy_location(held, store)
            ast.copy_location(held.func, store)
            held.parent = getattr(store, "parent", None)  # type: ignore[attr-defined]
            self._stores[id(store)] = held
        return held

    def named_callables(
        self, file: str, func: ast.AST, value: ast.expr, bound: tuple[tuple[str, ast.ClassDef], ...] = ()
    ) -> tuple[Target, ...]:
        """The definitions *value* names, handed to a call rather than called: ``Thread(target=self.run)``, ``submit(fn)``.

        What a call of *value* would land at, where *value* names a function,
        a method or a class -- read as :meth:`callees` reads the callee, for
        the classes ``self`` can be (*bound*).  Most arguments name data, and
        nothing is called where one is handed, so nothing is read further for
        them: not what a local or a field holds (:meth:`_constructed`), and
        not what an undeclared object can be (:meth:`Types.candidates`).
        Empty where *value* names no definition this reads.
        """
        if isinstance(value, ast.Name):
            return self._named(file, func, value.id).targets
        if not isinstance(value, ast.Attribute):
            return ()
        receiver = value.value
        if isinstance(receiver, ast.Name) and receiver.id in ("self", "cls"):
            cls = self.types.method_class(file, func)
            if cls is None:
                return ()
            return self._method_of_any(list(bound) or self.objects_of(cls), value.attr).targets
        if isinstance(receiver, ast.Call) and isinstance(receiver.func, ast.Name) and receiver.func.id == "super":
            cls = self.types.method_class(file, func)
            return self._after(cls, value.attr, bound).targets if cls is not None else ()
        module = self.types.module_named(file, receiver)
        if module is not None:
            return self._module_member(module, value.attr).targets
        named = self.types.class_named(file, receiver)
        if named is not None:
            return tuple(one._replace(bound=()) for one in self._method_of(named, value.attr).targets)
        declared = self.types.declared(file, func, receiver)
        if declared is not None:
            classes = list(dict.fromkeys(one for each in sorted(declared, key=_class_key) for one in self.objects_of(each)))
            return self._method_of_any(classes, value.attr).targets
        return ()

    def objects_of(self, declared: tuple[str, ast.ClassDef]) -> list[tuple[str, ast.ClassDef]]:
        """The classes an object declared as *declared* can be: it and each of its subclasses in the corpus.

        ``self`` in a method of a class, and a value declared as one.  A
        class outside the corpus built on it is not among them.
        """
        return [declared, *self.subclasses(*declared)]

    def _place(
        self, file: str, func: ast.AST, call: ast.Call, bound: tuple[tuple[str, ast.ClassDef], ...] = ()
    ) -> Placed:
        called = call.func
        if isinstance(called, ast.Name):
            named = self._named(file, func, called.id)
            if named.unplaced == UNPLACED_NAME and func is not None:
                return self._constructed(file, func, called) or named
            return named
        if not isinstance(called, ast.Attribute):
            return Placed((), UNPLACED_DYNAMIC)
        receiver = called.value
        if isinstance(receiver, (ast.Constant, ast.JoinedStr, ast.List, ast.Dict, ast.Set, ast.Tuple)):
            # ``",".join(...)``, ``f"...".format``: a builtin type's method.
            return Placed((), UNPLACED_OUTSIDE)
        if isinstance(receiver, ast.Name) and receiver.id in ("self", "cls"):
            cls = self.types.method_class(file, func)
            if cls is None:
                return Placed((), UNPLACED_RECEIVER)
            classes = list(bound) or self.objects_of(cls)
            method = self._method_of_any(classes, called.attr)
            if method.unplaced == UNPLACED_NO_METHOD:
                started = self._started(classes, called.attr)
                if started is not None:
                    return started
                if receiver.id == "self":
                    return self._constructed(file, func, called) or method
            return method
        if (
            isinstance(receiver, ast.Call)
            and isinstance(receiver.func, ast.Name)
            and receiver.func.id == "super"
        ):
            cls = self.types.method_class(file, func)
            if cls is None:
                return Placed((), UNPLACED_RECEIVER)
            return self._after(cls, called.attr, bound)
        module = self.types.module_named(file, receiver)
        if module is not None:
            return self._module_member(module, called.attr)
        # ``JobBrokerBase.__init__(self, ...)``: a method read off its class.
        named = self.types.class_named(file, receiver)
        if named is not None:
            placed = self._method_of(named, called.attr)._replace(on_class=True)
            return placed._replace(targets=tuple(one._replace(bound=self._handed_self(file, func, call, bound)) for one in placed.targets))
        # ``timed_method.run(job_id, ...)``: the function it was built with, called with them.
        forwarded = self._handed_on(file, func, call)
        if forwarded is not None:
            return forwarded
        declared = self.types.declared(file, func, receiver)
        if declared is not None:
            # ``self.conn: WrappedOracleConn | WrappedPostgresConn`` -- each
            # class it is declared as, as for the values below, and each
            # class an object declared so can be.
            classes = list(dict.fromkeys(one for each in sorted(declared, key=_class_key) for one in self.objects_of(each)))
            placed = self._method_of_any(classes, called.attr)
            if placed.unplaced == UNPLACED_NO_METHOD:
                return self._started(classes, called.attr) or placed
            return placed
        # Nothing declares it -- or it is declared ``Any`` -- so read what it
        # was built from.  Every class it can be is a target; where some way
        # of building it cannot be read, that is said beside them.
        classes, opened = self.types.candidates(file, func, receiver)
        builtin = {name for where, name in classes if where == BUILTIN}
        if builtin and len(builtin) == len(classes) and not opened:
            # ``tmp_list = []`` ... ``tmp_list.append(x)``: a builtin type's
            # method, as on a literal above -- and no call to a corpus
            # function of that name (:meth:`Calls.missed`).
            return Placed((), UNPLACED_OUTSIDE)
        # Beside a class of the corpus, a builtin is as open as it was before
        # it was told apart: the call may land on either.
        opened = opened or bool(builtin)
        built = [
            (where, cls)
            for where, name in sorted(classes)
            if where != BUILTIN and (cls := self.types.class_def(where, name)) is not None
        ]
        placed = self._method_of_any(built, called.attr, opened)
        if placed.targets or placed.unplaced in _FORWARDING:
            return placed
        started = self._started(built, called.attr)
        if started is not None:
            return started._replace(unplaced=UNPLACED_OPEN) if opened else started
        root = receiver
        while isinstance(root, ast.Attribute):
            root = root.value
        if isinstance(root, ast.Name) and self.types.imported_from_outside(file, root.id):
            return Placed((), UNPLACED_OUTSIDE)
        return Placed((), UNPLACED_RECEIVER)

    def _handed_self(
        self, file: str, func: ast.AST, call: ast.Call, bound: tuple[tuple[str, ast.ClassDef], ...]
    ) -> tuple[tuple[str, ast.ClassDef], ...]:
        """The classes of the object ``Base.m(self, ...)`` hands its method: ``self``'s, where the call hands it ``self``."""
        first = call.args[0] if call.args else None
        if not (isinstance(first, ast.Name) and first.id == "self"):
            return ()
        owned = self.types.method_class(file, func)
        if owned is None:
            return ()
        return tuple(sorted(bound or self.objects_of(owned), key=_class_key))

    def _declared_in(self, file: str, cls: ast.ClassDef, name: str) -> Optional[Target]:
        node = _function_binding(cls.body, name)
        return Target(file, node, cls) if node is not None else None

    def _method_of(self, declared: tuple[str, ast.ClassDef], name: str) -> Placed:
        """Where *name* lands on an object whose class is *declared* itself: the first definition in its method resolution order."""
        file, cls = declared
        order = self.types.mro(file, cls)
        for where, holder in order:
            found = self._declared_in(where, holder, name)
            if found is not None:
                return Placed((found._replace(bound=(declared,)),))
        for hook, reason in (("__getattr__", UNPLACED_FORWARDED), ("__getattribute__", UNPLACED_GETATTRIBUTE)):
            if any(self._declared_in(where, holder, hook) is not None for where, holder in order):
                return Placed((), reason)
        return Placed((), UNPLACED_NO_METHOD)

    def _after(
        self, owned: tuple[str, ast.ClassDef], name: str, bound: tuple[tuple[str, ast.ClassDef], ...] = ()
    ) -> Placed:
        """Where ``super().<name>(...)`` in a method of *owned* lands.

        At the next class after *owned* in the method resolution order of
        the object's class (:meth:`Types.mro`) that defines *name* -- for an
        object of each class it can be: *bound*, or else *owned* itself and
        each of its subclasses in the corpus, so a mixin's
        ``super().__init__(log_stream)`` lands at the base it names and, on
        an ``OraDBProxy.DBProxy``, at the next mixin, whose constructor no
        other call reaches.  *owned*'s own order first.
        """
        file, cls = owned
        found_for: list[tuple[Target, tuple[str, ast.ClassDef]]] = []
        for where, holder in list(bound) or self.objects_of(owned):
            order = self.types.mro(where, holder)
            at = next((index for index, (_file, one) in enumerate(order) if one is cls), None)
            for place, base in order[at + 1:] if at is not None else ():
                found = self._declared_in(place, base, name)
                if found is not None:
                    found_for.append((found, (where, holder)))
                    break
        targets = _grouped(found_for)
        return Placed(targets) if targets else Placed((), UNPLACED_NO_METHOD)

    def _method_of_any(
        self, classes: list[tuple[str, ast.ClassDef]], name: str, opened: bool = False
    ) -> Placed:
        """Where a call of *name* lands on an object that is one of *classes*.

        At the method each class finds first in its method resolution order,
        once per definition, with the classes that find it as its
        :attr:`Target.bound`.  A class that supplies the
        name only by forwarding it -- ``JediTaskBufferInterface``'s
        ``__getattr__``, ``WrappedOracleConn``'s ``__getattribute__`` -- adds
        no definition: what runs is somewhere this reading does not follow, so
        the placement is open beside the definitions found.  Where no class
        has the method, the reason is the forwarding one if any class
        forwards, else that none has it.  *opened* is the caller's own reason
        to say so: some way the object is built cannot be read.
        """
        found_for: list[tuple[Target, tuple[str, ast.ClassDef]]] = []
        forwarded: set[str] = set()
        for one in classes:
            placed = self._method_of(one, name)
            found_for.extend((target, one) for target in placed.targets)
            if not placed.targets and placed.unplaced in _FORWARDING:
                forwarded.add(placed.unplaced)
        targets = _grouped(found_for)
        if targets:
            return Placed(targets, UNPLACED_OPEN if opened or forwarded else "")
        for reason in _FORWARDING:
            if reason in forwarded:
                return Placed((), reason)
        return Placed((), UNPLACED_NO_METHOD)

    def _started(self, classes: list[tuple[str, ast.ClassDef]], name: str) -> Optional[Placed]:
        """``thr.start()`` on an object built on ``threading.Thread`` or ``multiprocessing.Process``: its ``run``, elsewhere.

        Python's rule (:data:`_STARTERS`): ``start()`` has the object's
        ``run`` called on a new thread, or in a new process, and returns.  A
        class that defines ``start`` itself is placed there before this is
        asked.  ``None`` where no class is built on one, or none has a ``run``
        in the corpus -- ``Thread(target=f)`` runs the ``f`` it was handed,
        which is not a call this reads.
        """
        if name != callfacts.START:
            return None
        starting = [one for one in classes if self.starts_run(one)]
        placed = self._method_of_any(starting, callfacts.RUN) if starting else None
        if placed is None or not placed.targets:
            return None
        return placed._replace(thread=True)

    def starts_run(self, declared: tuple[str, ast.ClassDef]) -> bool:
        """Whether *declared* is built on a library class whose ``start()`` runs ``run`` (:data:`_STARTERS`)."""
        for where, holder in self.types.mro(*declared):
            for base in holder.bases:
                if self.types.library_class(where, base) in _STARTERS:
                    return True
        return False

    # -- a call a wrapper hands on -------------------------------------------

    def forwarding(self, declared: tuple[str, ast.ClassDef], name: str) -> Optional[str]:
        """The parameter of *declared*'s ``__init__`` holding the function its method *name* calls with its own arguments.

        ``TimedMethod(method, timeout)`` keeps ``self.method = method``; its
        ``run(self, *var, **kwargs)`` starts ``threading.Thread(target=self,
        args=var, kwargs=kwargs)`` and joins it, and the thread calls
        ``__call__(self, *var, **kwargs)``, which sets ``self.result =
        self.method(*var, **kwargs)``: ``run(a, b)`` calls ``method(a, b)``
        and returns, with nothing, once it has returned.  Python's rules, read
        where every step is said: the method takes its object, ``*var`` and
        maybe ``**kw`` and nothing else, returns no value, and one statement
        of its own body hands them on, read nowhere else -- a call
        ``self.<field>(*var, **kw)``, a call ``self(*var, **kw)``, or a
        ``Thread`` or ``Process`` (:data:`_STARTERS`) built with
        ``target=self, args=var, kwargs=kw``, started and joined by the next
        two statements, the last two read through ``__call__`` the same way
        -- the call's value dropped, or bound to a field of the object
        (:attr:`Placed.kept_at`);
        ``__init__`` binds the field to the parameter and reads the parameter
        nowhere else; and no method of the class or its bases binds the field
        again or reads it but to make that call.  ``None`` otherwise.
        """
        found = self._forwarding_methods(declared).get(name)
        return found[1] if found is not None else None

    def _forwarding_methods(self, declared: tuple[str, ast.ClassDef]) -> dict[str, tuple[str, str, str]]:
        """``{method: (field, parameter, the field it keeps the value in)}`` for each method of *declared*
        :meth:`forwarding` reads."""
        key = (declared[0], declared[1].lineno)
        if key not in self._forwarders:
            names = {
                node.name
                for _where, holder in self.types.mro(*declared)
                for node in holder.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.args.vararg is not None
            }
            self._forwarders[key] = {
                name: found for name in sorted(names) if (found := self._forwarder(declared, name)) is not None
            }
        return self._forwarders[key]

    def _forwarder(self, declared: tuple[str, ast.ClassDef], name: str) -> Optional[tuple[str, str, str]]:
        """``(field, parameter, the field it keeps the value in)`` where :meth:`forwarding` reads *declared*'s method
        *name*."""
        found = self._handing_on(declared, name, 0)
        init = self._method_of(declared, "__init__").targets
        if found is None or len(init) != 1:
            return None
        field, made, kept = found
        stores = [
            statement
            for statement in init[0].func.body  # type: ignore[attr-defined]
            if isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Attribute)
            and _self_field(statement.targets[0]) == field
            and isinstance(statement.value, ast.Name)
        ]
        if len(stores) != 1:
            return None
        parameter = stores[0].value.id  # type: ignore[attr-defined]
        if parameter not in {one.arg for one in _positional(init[0], False)} | {
            one.arg for one in init[0].func.args.kwonlyargs  # type: ignore[attr-defined]
        }:
            return None
        if sum(isinstance(node, ast.Name) and node.id == parameter for node in walk(init[0].func)) != 1:
            return None
        for _where, holder in self.types.mro(*declared):
            for node in walk(holder):
                if not (isinstance(node, ast.Attribute) and node.attr == field and _self_field(node) == field):
                    continue
                if node is stores[0].targets[0]:
                    continue
                if not (isinstance(node.ctx, ast.Load) and any(node is call.func for call in made)):
                    return None
        return field, parameter, kept

    def _handing_on(
        self, declared: tuple[str, ast.ClassDef], name: str, depth: int
    ) -> Optional[tuple[str, list[ast.Call], str]]:
        """``(field, the calls of it, the field the value is kept in)`` where *declared*'s method *name* hands its own
        arguments to ``self.<field>``: ``self.result = self.method(*var, **kwargs)`` keeps it in ``result``."""
        targets = self._method_of(declared, name).targets
        if len(targets) != 1 or depth > 1:
            return None
        func = targets[0].func
        args = func.args  # type: ignore[attr-defined]
        if (
            args.posonlyargs
            or len(args.args) != 1
            or args.kwonlyargs
            or args.vararg is None
            or _is_static(func)
            or _is_classmethod(func)
            or any(isinstance(node, ast.Return) and node.value is not None for node in own_nodes(func))
        ):
            return None
        own, var = args.args[0].arg, args.vararg.arg
        kw = args.kwarg.arg if args.kwarg is not None else None
        body = func.body  # type: ignore[attr-defined]
        for index, statement in enumerate(body):
            value = statement.value if isinstance(statement, (ast.Expr, ast.Assign)) else None
            if not isinstance(value, ast.Call):
                continue
            called = value.func
            found: Optional[tuple[str, list[ast.Call], str]]
            if (
                _hands_on(value, var, kw)
                and isinstance(called, ast.Attribute)
                and isinstance(called.value, ast.Name)
                and called.value.id == own
            ):
                # ``self.result = self.method(*var, **kwargs)``: the value kept in a field of the object.
                stored = statement.targets if isinstance(statement, ast.Assign) else []
                kept = (
                    stored[0].attr
                    if len(stored) == 1
                    and isinstance(stored[0], ast.Attribute)
                    and isinstance(stored[0].value, ast.Name)
                    and stored[0].value.id == own
                    else ""
                )
                found = (called.attr, [value], kept)
            elif (_hands_on(value, var, kw) and isinstance(called, ast.Name) and called.id == own) or (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and self.types.library_class(targets[0].file, called) in _STARTERS
                and not value.args
                and {one.arg: getattr(one.value, "id", None) for one in value.keywords}
                == {"target": own, "args": var, **({"kwargs": kw} if kw else {})}
                and [_called_on(one) for one in body[index + 1 : index + 3]]
                == [(statement.targets[0].id, callfacts.START), (statement.targets[0].id, callfacts.JOIN)]
            ):
                found = self._handing_on(declared, "__call__", depth + 1)
            else:
                continue
            inside = {id(node) for node in walk(statement)}
            if found is None or any(
                isinstance(node, ast.Name) and node.id in (var, kw) and id(node) not in inside
                for node in own_nodes(func)
            ):
                return None
            return found
        return None

    def _wrapped(
        self, file: str, func: ast.AST, name: str
    ) -> Optional[tuple[list[tuple[tuple[str, ast.ClassDef], ast.Call]], list[ast.Call]]]:
        """``(each (class, construction) the local *name* is bound to, the calls made on it)`` where every use is read.

        Every binding of *name* in *func* is ``name = C(...)``, ``C`` a class
        of the corpus with a method :meth:`forwarding` reads; every other use
        is an attribute read off it -- a call of such a method, or a field that
        does not hold the function; it is not a parameter, not declared
        ``global`` or ``nonlocal``, and no nested scope uses it.  So the
        function the object was built with is called by those calls alone.
        ``None`` otherwise.
        """
        key = (id(func), name)
        if key in self._wraps:
            return self._wraps[key]
        self._wraps[key] = None
        args = getattr(func, "args", None)
        if args is None or name in {
            one.arg for one in (*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg) if one is not None
        }:
            return None
        own = {id(node) for node in own_nodes(func)}
        built: list[tuple[tuple[str, ast.ClassDef], ast.Call]] = []
        read: list[tuple[str, Optional[ast.Call]]] = []
        for node in walk(func):
            if isinstance(node, (ast.Global, ast.Nonlocal)) and name in node.names:
                return None
            if not (isinstance(node, ast.Name) and node.id == name):
                continue
            parent = getattr(node, "parent", None)
            if id(node) not in own:
                return None
            if isinstance(node.ctx, ast.Store):
                if not (
                    isinstance(parent, ast.Assign) and parent.targets == [node] and isinstance(parent.value, ast.Call)
                ):
                    return None
                cls = self.types.class_named(file, parent.value.func)
                if cls is None or not self._forwarding_methods(cls):
                    return None
                built.append((cls, parent.value))
            elif isinstance(node.ctx, ast.Load) and isinstance(parent, ast.Attribute) and isinstance(parent.ctx, ast.Load):
                above = getattr(parent, "parent", None)
                read.append((parent.attr, above if isinstance(above, ast.Call) and above.func is parent else None))
            else:
                return None
        if not built:
            return None
        for cls, _construction in built:
            methods = self._forwarding_methods(cls)
            fields = {field for field, _parameter, _kept in methods.values()}
            if any((attr not in methods) if call is not None else (attr in fields) for attr, call in read):
                return None
        found = (built, [call for _attr, call in read if call is not None])
        self._wraps[key] = found
        return found

    def _handed_on(self, file: str, func: ast.AST, call: ast.Call) -> Optional[Placed]:
        """Where *call*, made on a local :meth:`_wrapped` reads, lands: the functions it was built with (:attr:`Placed.forwarded`)."""
        called = call.func
        if not (
            isinstance(called, ast.Attribute)
            and isinstance(called.value, ast.Name)
            and called.attr in self._wrapping_methods()
        ):
            return None
        wrapped = self._wrapped(file, func, called.value.id)
        if wrapped is None:
            return None
        handed: list[ast.expr] = []
        kept: set[str] = set()
        for cls, construction in wrapped[0]:
            value = self._built_with(cls, construction, called.attr)
            if value is None:
                return None
            handed.append(value)
            kept.add(self._forwarding_methods(cls)[called.attr][2])
        targets: list[Target] = []
        on_class: set[bool] = set()
        for value in handed:
            landed = self.named_callables(file, func, value)
            if not landed:
                return None
            targets += [one for one in landed if one not in targets]
            on_class.add(isinstance(value, ast.Attribute) and self.types.class_named(file, value.value) is not None)
        if len(on_class) != 1:
            return None
        return Placed(
            tuple(targets),
            on_class=on_class.pop(),
            forwarded=tuple(handed),
            kept_at=kept.pop() if len(kept) == 1 else "",
        )

    def _built_with(self, cls: tuple[str, ast.ClassDef], construction: ast.Call, method: str) -> Optional[ast.expr]:
        """What *construction* hands the parameter whose function *cls*'s *method* calls; ``None`` where it does not say."""
        found = self._forwarding_methods(cls).get(method)
        init = self._method_of(cls, "__init__").targets
        if found is None or len(init) != 1 or found[1] not in said(construction, init[0]):
            return None
        return bind(construction, init[0]).get(found[1])

    def _wrapping_methods(self) -> set[str]:
        """The names of the methods :meth:`forwarding` reads, on any class of the corpus."""
        return self._wrappers()[0]

    def _wrappers(self) -> tuple[set[str], set[str]]:
        """``(the methods' names, the classes' names)`` of every class of the corpus with a method :meth:`forwarding` reads."""
        if self._wrapping_names is None or self._wrapping_classes is None:
            methods: set[str] = set()
            classes: set[str] = set()
            for file in self.types.corpus_files():
                tree = self.types.tree(file)
                for node in walk(tree) if tree is not None else ():
                    if isinstance(node, ast.ClassDef) and (found := self._forwarding_methods((file, node))):
                        methods |= set(found)
                        classes.add(node.name)
            self._wrapping_names, self._wrapping_classes = methods, classes
        return self._wrapping_names, self._wrapping_classes

    def _wrappings(self) -> list[tuple[str, ast.AST, ast.Call, ast.expr, list[ast.Call]]]:
        """``(file, function, construction, the function it is built with, the calls made on it)`` across the corpus.

        Each ``name = C(...)`` in a function, where :meth:`_wrapped` reads
        *name*, for each method called on it -- a candidate list: what a call
        lands at is :meth:`callees`'.
        """
        if self._wrapped_by is not None:
            return self._wrapped_by
        classes = self._wrappers()[1]
        aliases = self._aliases()
        names = classes | {other for one in classes for other in aliases.get(one, ())}
        found: list[tuple[str, ast.AST, ast.Call, ast.expr, list[ast.Call]]] = []
        for file in self.types.corpus_files() if names else []:
            tree = self.types.tree(file)
            for func in walk(tree) if tree is not None else ():
                if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                for node in own_nodes(func):
                    if not (
                        isinstance(node, ast.Assign)
                        and len(node.targets) == 1
                        and isinstance(node.targets[0], ast.Name)
                        and isinstance(node.value, ast.Call)
                        and _head_name(node.value.func) in names
                    ):
                        continue
                    wrapped = self._wrapped(file, func, node.targets[0].id)
                    if wrapped is None:
                        continue
                    for cls, construction in wrapped[0]:
                        if construction is not node.value:
                            continue
                        for method in dict.fromkeys(one.func.attr for one in wrapped[1]):  # type: ignore[attr-defined]
                            value = self._built_with(cls, construction, method)
                            if value is not None:
                                found.append((file, func, construction, value, wrapped[1]))
        self._wrapped_by = found
        return found

    def _forwarding_spellings(self, target: Target) -> list[str]:
        """The names of the methods called on an object built with a function spelt as *target* (:meth:`_wrappings`)."""
        if self._wrapped_spellings is None:
            spelt: dict[str, set[str]] = {}
            for _file, _func, _construction, value, made in self._wrappings():
                held = value.attr if isinstance(value, ast.Attribute) else value.id if isinstance(value, ast.Name) else ""
                if held:
                    spelt.setdefault(held, set()).update(one.func.attr for one in made)  # type: ignore[attr-defined]
            self._wrapped_spellings = spelt
        return sorted({one for name in self.spellings(target) for one in self._wrapped_spellings.get(name, ())})

    def forwarded_reads(self) -> set[int]:
        """``id`` of each function read as a value only to build an object whose every call hands it on (:attr:`Placed.forwarded`).

        ``global_task_buffer.updateJobStatus`` in ``TimedMethod(global_task_buffer.updateJobStatus, t)``:
        nothing calls it but ``timed_method.run(...)``, placed there with its arguments.
        """
        if self._forwarded_reads is None:
            self._forwarded_reads = {
                id(value)
                for file, func, _construction, value, made in self._wrappings()
                if all(value in self.callees(file, func, one).forwarded for one in made)
            }
        return self._forwarded_reads

    def made_on(self, file: str, func: ast.AST, call: ast.Call) -> Optional[ast.expr]:
        """The expression whose object the method *call*, made in *func*, runs on; ``None`` where it says none.

        The receiver it is called on -- or, for a call :attr:`Placed.forwarded`
        a function read off an object, that object: ``timed_method.run(...)``
        runs ``updateJobStatus`` on ``global_task_buffer``.
        """
        forwarded = self.callees(file, func, call).forwarded
        if forwarded:
            (handed, *more) = forwarded
            return handed.value if not more and isinstance(handed, ast.Attribute) else None
        return call.func.value if isinstance(call.func, ast.Attribute) else None

    def _constructed(self, file: str, func: ast.AST, holder: ast.expr) -> Optional[Placed]:
        """Where a call on a local or a ``self.<field>`` holding a class lands: that class's ``__init__``.

        ``setupper_plugin_class(self.task_buffer, self.jobs, tmp_log, ...)``
        builds whatever ``panda_config.getPlugin`` returned, or
        ``SetupperAtlasPlugin`` where it returned nothing; ``AdderGen`` keeps
        its class in ``self.adder_plugin_class``.  Neither is a function or a
        method, so the call was placed nowhere, and the plugins' constructors
        had no caller: a logger, a job or a task buffer handed to them could
        not be followed past them.  The classes are what the holder is bound
        to (:meth:`Types._values`).

        A binding this cannot read -- ``getPlugin`` returns ``type[Any]`` --
        leaves the call open beside the classes found, and the corpus's other
        subclasses of their base are taken as well: a configuration names a
        plugin of the same family, and those are the ones the corpus has
        (``trigger.dispatch_fanouts`` reads the same set at the same three
        sites).  A parameter holding the class is open the same way, assigned
        or not: ``make(job, cls=None)`` builds whatever class a caller hands
        it.  ``None`` where the holder names no class with a constructor in
        the corpus.
        """
        classes, opened = self.types._values(file, func, holder, 0, instances=False, seen=frozenset())
        arguments = getattr(func, "args", None)
        if isinstance(holder, ast.Name) and arguments is not None:
            every = [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]
            opened = opened or any(one.arg == holder.id for one in every)
        built = [
            (where, cls)
            for where, name in sorted(classes)
            if where != BUILTIN and (cls := self.types.class_def(where, name)) is not None
        ]
        if not built:
            return None
        if opened:
            for where, cls in list(built):
                lineage = self.types.lineage(where, cls)
                if len(lineage) > 1:
                    built += [one for one in self.subclasses(*lineage[1]) if one not in built]
        placed = self._method_of_any(built, "__init__", opened)
        return placed if placed.targets else None

    def _named(self, file: str, func: ast.AST, name: str) -> Placed:
        # A function or a class defined inside this one, or inside one that
        # holds it: ``datasetManager.main`` defines ``setTobeDeletedToDis`` and
        # ``FinisherThr``, and builds one with ``FinisherThr(...)``.
        scope: Optional[ast.AST] = func
        while scope is not None:
            node = binding(bound_defs(scope), name)
            if isinstance(node, ast.ClassDef):
                init = self._method_of((file, node), "__init__")
                return init if init.targets else Placed((), UNPLACED_NO_INIT)
            if node is not None:
                return Placed((Target(file, node, None),))
            scope = self.types.enclosing_function(file, scope)
        tree = self.types.tree(file)
        if tree is None:
            return Placed((), UNPLACED_OUTSIDE)
        defined = _function_binding(tree.body, name)
        if defined is not None:
            return Placed((Target(file, defined, None),))
        constructed = self.types.class_named(file, ast.Name(id=name, ctx=ast.Load()))
        if constructed is not None:
            init = self._method_of(constructed, "__init__")
            return init if init.targets else Placed((), UNPLACED_NO_INIT)
        held = self.types.imports(file).get(name)
        if held is not None:
            module, attribute = held
            target = self.types.module_file(module)
            if target is None:
                return Placed((), UNPLACED_OUTSIDE)
            return self._module_member(target, attribute or name)
        return Placed((), UNPLACED_NAME)

    def _module_member(self, file: str, name: str) -> Placed:
        tree = self.types.tree(file)
        if tree is None:
            return Placed((), UNPLACED_OUTSIDE)
        node = binding(tree.body, name)
        if isinstance(node, ast.ClassDef):
            init = self._method_of((file, node), "__init__")
            return init if init.targets else Placed((), UNPLACED_NO_INIT)
        if node is not None:
            return Placed((Target(file, node, None),))
        held = self.types.imports(file).get(name)
        if held is not None and held[0]:
            target = self.types.module_file(held[0])
            if target is not None and target != file:
                return self._module_member(target, held[1] or name)
        return Placed((), UNPLACED_NAME)

    # -- who calls a function ----------------------------------------------

    def every_call(self) -> dict[str, list[CallSite]]:
        """Every call in the corpus, by the name it calls -- see :meth:`_index`."""
        return self._index()

    def _index(self) -> dict[str, list[CallSite]]:
        """Every call in the corpus, by the name it calls -- a candidate list only.

        A name is how to *find* the calls worth placing, never how to place
        one: each is resolved by :meth:`callees` before it is kept.

        The calls in each module's own body come after every function's
        (:class:`CallSite`): ``extract_allowed_methods(...)`` at import,
        ``main()`` under ``if __name__ == "__main__":``.  A class's body,
        whose calls the corpus places at none of its functions, and a
        lambda's, which runs when it is called, are not read.
        """
        if self._by_name is not None:
            return self._by_name
        shared = _INDEXES.get(_roots_key(self.types.roots))
        if shared is not None:
            self._by_name = shared
            return shared
        found: dict[str, list[CallSite]] = {}

        def add(file: str, scope: ast.AST) -> None:
            for node in own_nodes(scope):
                if not isinstance(node, ast.Call):
                    continue
                called = node.func
                name = (
                    called.attr if isinstance(called, ast.Attribute)
                    else called.id if isinstance(called, ast.Name) else ""
                )
                if name:
                    found.setdefault(name, []).append(CallSite(file, scope, node))

        trees = [(file, tree) for file in self.types.corpus_files() if (tree := self.types.tree(file)) is not None]
        for file, tree in trees:
            for func in walk(tree):
                if isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    add(file, func)
        for file, tree in trees:
            add(file, tree)
        self._by_name = found
        _INDEXES[_roots_key(self.types.roots)] = found
        return found

    def callers(self, target: Target) -> list[CallSite]:
        """The calls in the corpus that :meth:`callees` places at *target*.

        Every call :meth:`callees` places at *target* is here -- the
        consistency the tests and the census hold it to.  The candidates are
        the calls spelled with any of :meth:`spellings`, then with a method
        called on an object built with a function spelt so
        (:meth:`_forwarding_spellings`); the target's own name comes first,
        so a list that was whole under that name alone keeps its order.

        A *target* bound to classes (:attr:`Target.bound`) is called by the
        calls that run it on an object of one of them: ``WorkerThread.run``
        bound to ``JobGeneratorThread`` by the ``start()`` of a
        JobGeneratorThread, not of a TaskCommandoThread.  A call placed with
        nothing said of its object may run it on any.
        """
        every = self._callers_of(target)
        if not target.bound:
            return every
        wanted = set(_bound_key(target.bound))
        return [
            site
            for site in every
            if any(
                one.func is target.func and (not one.bound or wanted & set(_bound_key(one.bound)))
                for one in self.callees(site.file, site.func, site.call).targets
            )
        ]

    def _callers_of(self, target: Target) -> list[CallSite]:
        """:meth:`callers` of *target*'s definition, whatever its object."""
        key = (target.file, target.func.lineno)  # type: ignore[attr-defined]
        if key in self._callers:
            return self._callers[key]
        shared = (_roots_key(self.types.roots), *key)
        if shared in _CALLERS:
            self._callers[key] = _CALLERS[shared]
            return _CALLERS[shared]
        kept = [
            site
            for name in dict.fromkeys([*self.spellings(target), *self._forwarding_spellings(target)])
            for site in self._index().get(name, [])
            if any(
                (placed.file, placed.func.lineno) == key  # type: ignore[attr-defined]
                for placed in self.callees(site.file, site.func, site.call).targets
            )
        ]
        self._callers[key] = kept
        _CALLERS[shared] = kept
        return kept

    def handed(self, site: CallSite, target: Target) -> dict[str, Optional[ast.expr]]:
        """:func:`bind` for *site*, its method read off a class where :meth:`callees` placed it so."""
        return bind(site.call, target, self.callees(site.file, site.func, site.call).on_class)

    def handed_classes(self, file: str, func: ast.AST, name: str, depth: int) -> Optional[Classes]:
        """The classes every call of *func* hands its parameter *name*, declared ``Any``; ``None`` where one is not declared.

        PanDA writes ``Any`` where no one class it can name covers what
        arrives: ``WorkflowInterface(task_buffer: Any)`` is built with
        JEDI's ``JediTaskBufferInterface`` by its watchdog and with the
        server's ``TaskBuffer`` by ``workflow_api`` -- and naming the JEDI
        class would make ``pandaserver`` import ``pandajedi``
        (``pandaserver.asyncprocess.TaskBufferLike`` says so).  Each call
        says which one it hands, so the parameter holds what the calls hand
        it, read by their declarations: every class any of them hands, as an
        annotation naming them all would.  A ``None`` handed adds nothing,
        as an assignment of ``None`` does not (:meth:`Types.declared`).

        Nothing where a call hands what is not declared, or what a ``*args``
        hides, or where no call is placed: the daemons' ``main(tbuf: Any)``
        is called by the name a configuration gives ``importlib``.
        """
        key = (file, getattr(func, "lineno", 0), name)
        if depth > self.types.DEPTH or key in self._handing:
            return None
        self._handing.add(key)
        try:
            owned = self.types.method_class(file, func)
            target = Target(file, func, owned[1] if owned else None)
            sites = self.callers(target)
            found: set[tuple[str, ast.ClassDef]] = set()
            for site in sites:
                value = self.handed(site, target).get(name)
                if value is None:
                    return None
                if isinstance(value, ast.Constant) and value.value is None:
                    continue
                classes = self.types.declared(site.file, site.func, value, depth + 1)
                if classes is None:
                    return None
                found |= classes
            return frozenset(found) or None
        finally:
            self._handing.discard(key)

    def filled(self, site: CallSite, target: Target) -> set[str]:
        """:func:`said` for *site*, as :meth:`handed`."""
        return said(site.call, target, self.callees(site.file, site.func, site.call).on_class)

    def spellings(self, target: Target) -> list[str]:
        """Every name a call that lands at *target* can be spelled with, its own name first.

        A function's own name.  An ``__init__``'s class name; ``__init__``
        itself, which is how ``super().__init__(...)`` and
        ``Base.__init__(self, ...)`` spell the chain from a subclass's
        constructor; and the name of each subclass that defines no
        ``__init__`` of its own, whose construction runs this one.  Then every
        other name any of those is imported or bound as in the corpus --
        ``from pandaserver.api.v1.http_client import HttpClient as
        HttpClientV1`` built twenty clients no list of ``HttpClient``'s callers
        held.  Last, for an ``__init__``, each local or field a statement binds
        a class of its family to: ``setupper_plugin_class(...)`` builds
        ``SetupperAtlasPlugin`` (:meth:`_constructed`).  A name matching more
        than the calls landing here costs only their placing: :meth:`callees`
        keeps the ones that land.
        """
        name = target.func.name  # type: ignore[attr-defined]
        if name == "__init__" and target.cls is not None:
            seeds = [target.cls.name, *self._built_through(target)]
        elif name == callfacts.RUN and target.cls is not None and any(
            self.starts_run(one) for one in self.objects_of((target.file, target.cls))
        ):
            # ``thr.start()`` runs it (:meth:`_started`).
            seeds = [name, callfacts.START]
        else:
            seeds = [name]
        aliases = self._aliases()
        found = list(dict.fromkeys(seeds))
        queue = list(found)
        while queue:
            for other in sorted(aliases.get(queue.pop(0), ())):
                if other not in found:
                    found.append(other)
                    queue.append(other)
        if name == "__init__" and target.cls is not None:
            found.insert(1, "__init__")
            # A local or a field holding the class, or another of its family,
            # is called to build it (:meth:`_constructed`).
            holders = self._holders()
            for one in self._families(target):
                found += [other for other in sorted(holders.get(one, ())) if other not in found]
        return found

    def _families(self, target: Target) -> list[str]:
        """Names of the classes whose construction :meth:`_constructed` may place at *target*, an ``__init__``.

        Its class and the subclasses built through it, and for each the
        other subclasses of its base -- the plugins one configuration may
        name in place of another.
        """
        assert target.cls is not None
        through = set(self._built_through(target))
        classes = [(target.file, target.cls)] + [
            one for one in self._subclasses().get((target.file, target.cls.lineno), []) if one[1].name in through
        ]
        names: list[str] = []
        for where, cls in classes:
            lineage = self.types.lineage(where, cls)
            for _file, one in [(where, cls), *(self.subclasses(*lineage[1]) if len(lineage) > 1 else [])]:
                if one.name not in names:
                    names.append(one.name)
        return names

    def _holders(self) -> dict[str, set[str]]:
        """``{name: each local or self.<field> a statement binds it to}``, across the corpus -- a candidate list only."""
        shared = _roots_key(self.types.roots)
        if shared not in _HOLDERS:
            found: dict[str, set[str]] = {}
            for file in self.types.corpus_files():
                tree = self.types.tree(file)
                for node in walk(tree) if tree is not None else ():
                    if not isinstance(node, (ast.Assign, ast.AnnAssign)) or not isinstance(node.value, (ast.Name, ast.Attribute)):
                        continue
                    held = node.value.id if isinstance(node.value, ast.Name) else node.value.attr
                    for one in node.targets if isinstance(node, ast.Assign) else [node.target]:
                        local = (
                            one.id if isinstance(one, ast.Name)
                            else one.attr if isinstance(one, ast.Attribute) and getattr(one.value, "id", "") == "self"
                            else ""
                        )
                        if local and local != held:
                            found.setdefault(held, set()).add(local)
            _HOLDERS[shared] = found
        return _HOLDERS[shared]

    def _built_through(self, target: Target) -> list[str]:
        """Names of *target*'s class's subclasses whose construction lands at *target*, an ``__init__``."""
        assert target.cls is not None
        held = self._subclasses().get((target.file, target.cls.lineno), [])
        return sorted(
            {
                cls.name
                for file, cls in held
                if any(one.func is target.func for one in self._method_of((file, cls), "__init__").targets)
            }
        )

    def subclasses(self, file: str, cls: ast.ClassDef) -> list[tuple[str, ast.ClassDef]]:
        """Every corpus class with *cls* among its bases."""
        return list(self._subclasses().get((file, cls.lineno), []))

    def _subclasses(self) -> dict[tuple[str, int], list[tuple[str, ast.ClassDef]]]:
        """``{(file, line of a class): every corpus class with it among its bases}``."""
        shared = _roots_key(self.types.roots)
        if shared not in _SUBCLASSES:
            found: dict[tuple[str, int], list[tuple[str, ast.ClassDef]]] = {}
            for file in self.types.corpus_files():
                tree = self.types.tree(file)
                for node in walk(tree) if tree is not None else ():
                    if isinstance(node, ast.ClassDef):
                        for where, base in self.types.lineage(file, node)[1:]:
                            found.setdefault((where, base.lineno), []).append((file, node))
            _SUBCLASSES[shared] = found
        return _SUBCLASSES[shared]

    def _aliases(self) -> dict[str, set[str]]:
        """``{name: each other name it is imported as or bound to at module level}``, across the corpus."""
        shared = _roots_key(self.types.roots)
        if shared not in _ALIASES:
            found: dict[str, set[str]] = {}
            for file in self.types.corpus_files():
                for local, (_module, attribute) in self.types.imports(file).items():
                    if attribute and local != attribute:
                        found.setdefault(attribute, set()).add(local)
                for local, values in self.types._module_bound(file).items():
                    for value in values:
                        held = value.id if isinstance(value, ast.Name) else value.attr  # type: ignore[attr-defined]
                        if held != local:
                            found.setdefault(held, set()).add(local)
            _ALIASES[shared] = found
        return _ALIASES[shared]

    def opened(self, site: CallSite) -> str:
        """Why *site*'s call may land beyond the definitions it is placed at; ``""`` where it is placed closed.

        :meth:`callers` lists a call placed open beside the ones placed
        closed -- ``self.taskBufferIF.getTaskWithID_JEDI(...)`` lands at
        ``JediTaskBuffer``'s, or wherever the interface forwards it.  A reader
        listing callers says so beside it; a reader taking one definition as
        the call's does not take it.
        """
        placed = self.callees(site.file, site.func, site.call)
        return placed.unplaced if placed.targets else ""

    def hop(self, site: CallSite) -> str:
        """``file:line`` of *site*, and why it may land elsewhere where it is placed open."""
        why = self.opened(site)
        return f"{site.file}:{site.call.lineno}" + (f" (placed open -- {why})" if why else "")

    def called_from(self, target: Target) -> list[str]:
        """``file:line`` of each call placed at *target*, and one hop out where its caller is called.

        Sorted, and every chain listed: a reader asking whether a function
        calls *target* within these two hops takes one it does not find as
        no, which a list cut short cannot say.
        A hop placed open says so after its ``file:line`` (:meth:`opened`).
        A call in a module's body is the end of its chain.  The hop out is
        taken on the objects the first one lands at *target* for
        (:meth:`landing`): ``self.runImpl()`` in ``WorkerThread.run`` lands at
        ``JobGeneratorThread.runImpl`` only on a JobGeneratorThread, so the
        hop out is that class's ``start()``, not every worker's.
        """
        chains: list[str] = []
        for site in self.callers(target):
            at = self.hop(site)
            if site.in_module:
                chains.append(at)
                continue
            owned = self.types.method_class(site.file, site.func)
            outer = self.callers(Target(site.file, site.func, owned[1] if owned else None, self.landing(site, target)))
            if outer:
                chains.extend(f"{self.hop(one)} -> {at}" for one in outer)
            else:
                chains.append(at)
        return sorted(set(chains))

    def landing(self, site: CallSite, target: Target) -> tuple[tuple[str, ast.ClassDef], ...]:
        """The classes ``self`` can be in *site*'s function for its call to land at *target*; ``()`` where any of them.

        Only a call reading ``self`` (:func:`_reads_self`) depends on it.
        """
        owned = self.types.method_class(site.file, site.func) if not site.in_module else None
        if owned is None or not _reads_self(site.call):
            return ()
        every = self.objects_of(owned)
        wanted = set(_bound_key(target.bound))
        found = [
            one
            for one in every
            if any(
                placed.func is target.func and (not wanted or not placed.bound or wanted & set(_bound_key(placed.bound)))
                for placed in self.callees(site.file, site.func, site.call, (one,)).targets
            )
        ]
        return tuple(found) if len(found) < len(every) else ()

    #: How many calls out a constant is looked for.
    CONSTANT_HOPS = 3

    def handed_constants(
        self, target: Target, names: set[str], depth: int = 0
    ) -> list[tuple[tuple[str, ...], dict[str, object]]]:
        """Per call placed at *target*: where it came from, and the constants it hands *names*.

        ``(("pkg/tb.py:551", "pkg/cmd.py:143"), {"onlyActive": True})`` -- the
        call into *target* first.  A name the caller hands on from a parameter
        of its own that it never rebinds is looked for at that caller's
        callers, so ``JediTaskBuffer``'s forwarding method is read through to
        ``TaskCommando``.  A call handing none of *names* a constant is left
        out: it runs the function as the general reading already reads it.
        Every call placed at *target* is read, however many there are.
        """
        sites = self.callers(target)
        if not sites:
            return []
        found: list[tuple[tuple[str, ...], dict[str, object]]] = []
        for site in sites:
            bound = self.handed(site, target)
            at = f"{site.file}:{site.call.lineno}"
            values: dict[str, object] = {}
            forwarded: dict[str, str] = {}
            for name in names:
                handed = bound.get(name)
                if isinstance(handed, ast.Constant):
                    values[name] = handed.value
                elif (
                    isinstance(handed, ast.Name)
                    and handed.id in _parameter_names(site.func)
                    and not _rebinds(site.func, handed.id)
                ):
                    forwarded[name] = handed.id
            outer: list[tuple[tuple[str, ...], dict[str, object]]] = []
            if forwarded and depth + 1 < self.CONSTANT_HOPS:
                owned = self.types.method_class(site.file, site.func)
                caller = Target(site.file, site.func, owned[1] if owned else None)
                outer = self.handed_constants(caller, set(forwarded.values()), depth + 1)
            for via, more in outer:
                merged = dict(values)
                merged.update({name: more[held] for name, held in forwarded.items() if held in more})
                found.append(((at, *via), merged))
            if not outer and values:
                found.append(((at,), values))
        return found

    def route_constant(self, name: str, target: Target, sites: tuple) -> object:
        """The constant the route *sites* hands *target*'s parameter *name*, or :data:`UNSET`.

        *sites* is the call into *target* first, then outward.  Followed out
        through each forwarder that hands the parameter on unchanged; a call
        that does not pass it hands the default it states.  A value the
        route's outermost caller takes as its own parameter is not fixed:
        that caller's callers are not asked.  Nor is a forwarder's caller
        asked for a parameter the forwarder may bind again on the route
        before handing it on (:meth:`bound_again`): ``getTasksToBeProcessed_JEDI``
        sets ``is_dry_run = True`` whenever ``simTasks`` is given, whatever its
        caller passed -- on a route that gives it, not on one that leaves it
        ``None``.  Where the route's constants say which binding runs, the
        parameter holds what that one binds (:meth:`bound_on_route`).  A list,
        tuple or set display is :class:`Displayed`.
        """
        site = sites[0]
        handed = self.handed(site, target).get(name)
        if isinstance(handed, ast.Constant):
            return handed.value
        if (displayed := _displayed(handed)) is not None:
            return displayed
        if (
            isinstance(handed, ast.Name)
            and name in self.filled(site, target)
            and len(sites) > 1
            and handed.id in _parameter_names(site.func)
        ):
            owned = self.types.method_class(site.file, site.func)
            outer = Target(site.file, site.func, owned[1] if owned else None)
            if self.bound_again(handed.id, outer, sites[1:]):
                return self.bound_on_route(handed.id, outer, sites[1:], site.call)
            return self.route_constant(handed.id, outer, sites[1:])
        return UNSET

    def bound_on_route(self, name: str, target: Target, sites: tuple, call: ast.AST) -> object:
        """What *target*'s parameter *name* holds at *call* when the route *sites* calls it and binds it again on the
        way: what the one binding the route's constants say runs before *call* binds (:meth:`route_constant` of a
        parameter it binds from), or :data:`UNSET`.

        ``getTasksToBeProcessed_JEDI`` binds ``target_tasks = simTasks`` under ``if simTasks is not None`` (3245)
        before handing ``target_tasks`` on (3287): on TaskBroker 259's route, which hands ``simTasks=[tmpTaskItem]``,
        that binding runs and ``target_tasks`` holds the display.  Only a plain ``name = value`` held by nothing but
        ``if``s of *target*'s own, every test on its way decided for it (:func:`pathcond.runs_by`), the only binding
        of the name not ruled out, and above *call*: a loop's binding, one the constants leave open, or two that
        may run say nothing.
        """
        func = target.func
        again = [binding.site for binding in pathcond.bindings(func, name) if not binding.nested]
        if not again or any(
            isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Del) for node in own_nodes(func)
        ):
            return UNSET
        tested = {
            one.id
            for site in again
            for test in pathcond.tests_on(site, func)
            for one in walk(test)
            if isinstance(one, ast.Name) and one.id in _parameter_names(func)
        }
        constants = {
            one: value for one in sorted(tested) if (value := self.route_constant(one, target, sites)) is not UNSET
        }
        running = [site for site in again if not pathcond.ruled_out_by(site, func, constants)]
        if len(running) != 1:
            return UNSET
        site = running[0]
        if not (
            isinstance(site, ast.Assign)
            and len(site.targets) == 1
            and isinstance(site.targets[0], ast.Name)
            and site.lineno < getattr(call, "lineno", 0)
            and _held_by_ifs(site, func)
            and pathcond.runs_by(site, func, constants)
        ):
            return UNSET
        value = site.value
        if isinstance(value, ast.Constant):
            return value.value
        if (displayed := _displayed(value)) is not None:
            return displayed
        if isinstance(value, ast.Name) and value.id in _parameter_names(func) and not _rebinds(func, value.id):
            return self.route_constant(value.id, target, sites)
        return UNSET

    def bound_again(self, name: str, target: Target, sites: tuple) -> bool:
        """Whether *target*'s function may bind its parameter *name* again when the route *sites* calls it.

        A binding the function's own tests rule out, with the constants the
        route hands the parameters they test, is not on the route
        (:func:`pathcond.ruled_out_by`) -- the parameter itself among them:
        ``maxNumJobs -= ...`` under ``if maxNumJobs is not None`` never runs
        on a route handing ``None``, and where no binding runs the name holds
        what it was handed.  A ``del`` always may.
        """
        func = target.func
        if not _rebinds(func, name):
            return False
        if any(isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Del) for node in walk(func)):
            return True
        again = [binding.site for binding in pathcond.bindings(func, name) if not binding.nested]
        tested = {
            one.id
            for site in again
            for test in pathcond.tests_on(site, func)
            for one in walk(test)
            if isinstance(one, ast.Name) and one.id in _parameter_names(func)
        }
        constants = {
            one: value for one in sorted(tested) if (value := self.route_constant(one, target, sites)) is not UNSET
        }
        return any(not pathcond.ruled_out_by(site, func, constants) for site in again)

    #: How many calls out a route goes, the call into the function counted.
    ROUTE_HOPS = 4

    def routes(self, target: Target, depth: int = 0) -> tuple[list["Route"], list[str]]:
        """Every route of calls into *target*, and why the list may miss some (:meth:`missed`).

        A route starts at a call placed at *target* and goes on outward while
        the caller hands *target* one of its own parameters unchanged -- a
        forwarder, ``TaskBuffer.lockJobsForReassign`` handing on what
        ``copyArchive`` gave it -- so that the outermost call of a route is
        where the values are written down, or where the caller's own callers
        are unknown.  Up to :data:`ROUTE_HOPS` calls out, from every call
        placed at *target*, however many there are.
        """
        sites = self.callers(target)
        stops = list(self.missed(target))
        found: list[Route] = []
        for site in sites:
            handed = self.handed(site, target)
            filled = self.filled(site, target)
            forwards = any(
                isinstance(value, ast.Name)
                and name in filled
                and value.id in _parameter_names(site.func)
                and not _rebinds(site.func, value.id)
                for name, value in handed.items()
            )
            if forwards and depth + 1 < self.ROUTE_HOPS:
                owned = self.types.method_class(site.file, site.func)
                outer, more = self.routes(Target(site.file, site.func, owned[1] if owned else None), depth + 1)
                if not outer:
                    # The forwarder is the route's outermost function: what
                    # it hands on is its parameters' values, unfixed, and no
                    # caller of it is missing from a list it has none in.
                    more = [one for one in more if not one.endswith(_OUTSIDE)]
                stops.extend(one for one in more if one not in stops)
                if outer:
                    found.extend(Route((site, *route.sites)) for route in outer)
                    continue
            found.append(Route((site,)))
        return found, stops

    def missed(self, target: Target) -> list[str]:
        """Why the calls placed at *target* may not be every call reaching it; empty where they are.

        A call spelled with *target*'s name that is placed nowhere may be a
        call to it, so the list is said to stop there -- by reason, and by the
        receiver the call is made on -- though the call may be to another
        function of that name: this can call a complete list incomplete, never
        the other way round.  So is a call placed at other definitions and
        open beside them: ``self.taskBufferIF.getTaskWithID_JEDI(...)`` lands
        at ``JediTaskBuffer``'s, or wherever the interface forwards it.  A
        function no call in the corpus names is reached from outside it: an
        endpoint, a thread's ``run``.

        A call placed outside the corpus is not a call to *target* and is not
        listed: ``tmp_list.append(x)`` on a list, ``os.kill(...)``,
        ``json.dump(...)``.  Where every call by the name lands there, that is
        what is said -- not that nothing names it, and not that it is reached
        from outside: ``ListWithLock.__contains__`` is asked by ``in``, which
        this reading does not read as a call.

        The calls looked at are those spelled with any of :meth:`spellings`.
        One spelled ``__init__`` is a call to a constructor only in a subclass
        of *target*'s class -- where the chain from a subclass runs --
        and ``threading.Thread.__init__(self)`` in any other class is not.
        """
        key = (target.file, target.func.lineno)  # type: ignore[attr-defined]
        if key in self._missed:
            return self._missed[key]
        name = target.func.name  # type: ignore[attr-defined]
        if name == "__init__" and target.cls is not None:
            name = target.cls.name
        placed = self.callers(target)
        why: list[str] = []
        unplaced: dict[str, dict[str, int]] = {}
        elsewhere: dict[str, dict[str, int]] = {}
        outside: dict[str, int] = {}
        for site in (site for spelled in self.spellings(target) for site in self._index().get(spelled, [])):
            if site.func is target.func:
                continue
            landed = self.callees(site.file, site.func, site.call)
            if landed.targets and (not landed.unplaced or any(one.func is target.func for one in landed.targets)):
                continue
            called = site.call.func
            if isinstance(called, ast.Attribute) and called.attr == "__init__" and not self._in_subclass(site, target):
                continue
            if landed.targets:
                held = elsewhere.setdefault(landed.unplaced, {})
            elif landed.unplaced == UNPLACED_OUTSIDE:
                held = outside
            else:
                held = unplaced.setdefault(landed.unplaced or "not placed", {})
            receiver = ast.unparse(site.call.func.value) if isinstance(site.call.func, ast.Attribute) else "(a bare name)"
            held[receiver] = held.get(receiver, 0) + 1
        for reason, receivers in sorted(unplaced.items()):
            why.append(f"{name}: {sum(receivers.values())} call(s) by its name not placed -- {reason} ({_on(receivers)})")
        for reason, receivers in sorted(elsewhere.items()):
            why.append(f"{name}: {sum(receivers.values())} call(s) by its name placed elsewhere -- {reason} ({_on(receivers)})")
        if not placed and not unplaced and not elsewhere:
            if outside:
                why.append(
                    f"{name}: {_LANDS_OUTSIDE} -- {sum(outside.values())} call(s) by its name land "
                    f"{UNPLACED_OUTSIDE} ({_on(outside)})"
                )
            elif read := self.read_as_value(target):
                shown = ", ".join(read[:_READS]) + (f", … {len(read) - _READS} more" if len(read) > _READS else "")
                why.append(f"{name}: {_READ_AS_VALUE.format(shown)}")
            else:
                why.append(f"{name}: {_OUTSIDE}")
        self._missed[key] = why
        return why

    def read_as_value(self, target: Target) -> list[str]:
        """``file:line`` of each place a class whose construction runs *target*, an ``__init__``, is read as a value.

        ``self.adder_plugin_class = AdderAtlasPlugin`` holds the class, and
        ``self.adder_plugin_class(job, ...)`` builds one: no call names
        ``AdderAtlasPlugin``, and it is not reached from outside.  A read a
        call of the value may build from -- bound, passed, returned, held in
        a container -- and not one that cannot: a base list, ``isinstance``,
        ``super()``'s argument, an ``except`` clause, a comparison, an
        annotation.  Empty for any other function.
        """
        if target.cls is None or getattr(target.func, "name", "") != "__init__":
            return []
        held = self._subclasses().get((target.file, target.cls.lineno), [])
        built = [(target.file, target.cls)] + [
            (file, cls)
            for file, cls in held
            if any(one.func is target.func for one in self._method_of((file, cls), "__init__").targets)
        ]
        read = self._values_read()
        return sorted({where for file, cls in built for where in read.get((file, cls.lineno), [])})

    def _values_read(self) -> dict[tuple[str, int], list[str]]:
        """``{(file, line of a class): file:line of each read of it as a value a call may build from}``."""
        shared = _roots_key(self.types.roots)
        if shared not in _VALUES:
            found: dict[tuple[str, int], list[str]] = {}
            for file in self.types.corpus_files():
                tree = self.types.tree(file)
                for node in walk(tree) if tree is not None else ():
                    if (
                        isinstance(node, (ast.Name, ast.Attribute))
                        and isinstance(node.ctx, ast.Load)
                        and _may_build_from(node)
                        and (named := self.types.class_named(file, node)) is not None
                    ):
                        found.setdefault((named[0], named[1].lineno), []).append(f"{file}:{node.lineno}")
            _VALUES[shared] = found
        return _VALUES[shared]

    def _in_subclass(self, site: CallSite, target: Target) -> bool:
        """Whether *site* is made in a method of a strict subclass of *target*'s class."""
        held = self.types.method_class(site.file, site.func)
        return held is not None and any(cls is target.cls for _where, cls in self.types.lineage(*held)[1:])

    # -- what a method changes on its object --------------------------------

    def writes(self, target: Target, depth: int = 0) -> list[tuple[str, ast.expr, ast.AST, Target]]:
        """``(field, value, site, where)`` for each write *target* makes to its own object.

        An assignment to ``self.<field>`` (``+=`` included), a container on it
        filled -- ``self.<field>.append(x)``, ``self.<field>[k] = x`` -- and,
        through a call on ``self``, what the called method writes.  So
        ``set_fatal()`` is ``status_code = self.rc_fatal`` at the call that
        makes it, which is what a reader of ``status_code`` needs.
        """
        found: list[tuple[str, ast.expr, ast.AST, Target]] = []
        for node in own_nodes(target.func):
            if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                value = node.value
                if value is None:
                    continue
                for held in targets:
                    field = _self_field(held)
                    if field:
                        found.append((field, value, node, target))
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                field = _self_field(node.func.value)
                filled = callfacts.by_name(node.func.attr) if field else None
                if filled is not None and filled.puts:
                    # Read by the name: the field's class is what is being read.
                    found.extend(
                        (field, value, node, target)
                        for value, put, _key, _keyword in callfacts.put_arguments(filled, node)
                        if put.into != callfacts.KEY
                    )
                elif (
                    isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "self"
                    and depth < 3
                    and target.cls is not None
                ):
                    classes = list(target.bound) or self.objects_of((target.file, target.cls))
                    for inner in self._method_of_any(classes, node.func.attr).targets:
                        found.extend(self.writes(inner, depth + 1))
        return found


#: How many receivers :func:`_on` names before it counts the rest.
_RECEIVERS = 5


def _on(receivers: dict[str, int]) -> str:
    """``self.taskBufferIF 19, tbIF 2`` -- the receivers calls were made on, most first, the first five.

    A method named ``append`` or ``update`` shares its name with every
    container's: the calls on ``varMap`` and ``tmp_list`` that may be calls to
    it run to 168 receivers and eleven thousand characters, so past
    :data:`_RECEIVERS` the rest are counted.
    """
    ranked = sorted(receivers.items(), key=lambda one: (-one[1], one[0]))
    said = ", ".join(f"{receiver} {count}" for receiver, count in ranked[:_RECEIVERS])
    return said + (f", … {len(ranked) - _RECEIVERS} more" if len(ranked) > _RECEIVERS else "")


def _parameter_names(func: ast.AST) -> set[str]:
    args = getattr(func, "args", None)
    if args is None:
        return set()
    return {a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)}


def _displayed(node: Optional[ast.AST]) -> Optional[Displayed]:
    """*node* as :class:`Displayed` where it is a list, tuple or set display that spells each item; else ``None``."""
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)) and not any(isinstance(one, ast.Starred) for one in node.elts):
        return Displayed(len(node.elts))
    return None


def _held_by_ifs(node: ast.AST, func: ast.AST) -> bool:
    """Whether every statement holding *node* up to *func* is an ``if`` (or *func* itself): no loop, ``try`` or
    ``with`` decides whether, or how often, it runs."""
    parent = getattr(node, "parent", None)
    while parent is not None and parent is not func:
        if not isinstance(parent, ast.If):
            return False
        parent = getattr(parent, "parent", None)
    return parent is func


def _rebinds(func: ast.AST, name: str) -> bool:
    """Whether *func*'s own body binds *name* anywhere -- so a parameter's value may not be the one handed in.

    A :class:`pathcond.Binding` outside a nested scope, or a ``del``.  Read
    off the names' store context before, which counted a comprehension's own
    variable and a bare annotation and missed ``except ... as``.
    """
    if any(not binding.nested for binding in pathcond.bindings(func, name)):
        return True
    return any(
        isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Del)
        for node in own_nodes(func)
    )


def _self_field(node: ast.AST) -> str:
    """``status_code`` for ``self.status_code``, ``x`` for ``self.x[k]``; else ``""``."""
    if isinstance(node, ast.Subscript):
        node = node.value
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
        return node.attr
    return ""


def _may_build_from(node: ast.expr) -> bool:
    """Whether *node*, a name or attribute read, holds a value a later call may build from -- :meth:`Calls.read_as_value`.

    Not a call's callee or an attribute's object -- those use it here -- and
    not a place that only tests or declares a class: a base list, a class
    statement's keywords, an argument of a builtin that neither calls nor
    keeps it (:func:`callfacts.builds_nothing`: ``isinstance``, ``hasattr``,
    ``super``'s first), an ``except`` clause, a comparison, an annotation.
    """
    parent = getattr(node, "parent", None)
    if isinstance(parent, ast.Call) and parent.func is node:
        return False
    if isinstance(parent, ast.Attribute) and parent.value is node:
        return False
    if isinstance(parent, ast.Compare):
        return False
    child: ast.AST = node
    while parent is not None and not isinstance(parent, ast.stmt):
        if (
            isinstance(parent, ast.Call)
            and child is not parent.func
            and (function := callfacts.builtin_function_at(parent)) is not None
            and callfacts.builds_nothing(function, parent, child.value if isinstance(child, ast.keyword) else child)
        ):
            return False
        if isinstance(parent, ast.ExceptHandler):
            return False
        if isinstance(parent, ast.arg) and child is parent.annotation:
            return False
        child, parent = parent, getattr(parent, "parent", None)
    if isinstance(parent, ast.ClassDef) and child in (*parent.bases, *parent.keywords):
        return False
    if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)) and child is not None and child is parent.returns:
        return False
    if isinstance(parent, ast.AnnAssign) and child is parent.annotation:
        return False
    if isinstance(parent, ast.ExceptHandler):
        return False
    return True


def _unquoted(annotation: ast.expr) -> ast.expr:
    """"SiteMapper" as the expression it quotes; anything else as it is."""
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        try:
            return ast.parse(annotation.value.strip(), mode="eval").body
        except SyntaxError:
            return annotation
    return annotation

def _element_annotation(annotation: ast.expr) -> Optional[ast.expr]:
    """The element type a container annotation states: ``dict[K, V]`` -> ``V``, ``list[T]`` -> ``T``."""
    annotation = _unquoted(annotation)
    if not isinstance(annotation, ast.Subscript):
        return None
    wrapper = annotation.value
    name = wrapper.attr if isinstance(wrapper, ast.Attribute) else getattr(wrapper, "id", "")
    inner = annotation.slice
    if name in _MAPPINGS:
        return inner.elts[1] if isinstance(inner, ast.Tuple) and len(inner.elts) == 2 else None
    if name in _COLLECTIONS:
        return _first_argument(inner)
    return None


def _first_argument(inner: ast.expr) -> ast.expr:
    """The first argument of a subscript whose arguments are *inner*: ``Generator[Y, S, R]``'s ``Y``."""
    return inner.elts[0] if isinstance(inner, ast.Tuple) and inner.elts else inner


#: Containers a loop takes its target from, by the annotation's name -- :meth:`Types._iterated` -- and a subscript
#: its value (:data:`callfacts.CONTAINERS`): a mapping its keys, the value under one; a collection its elements.
_MAPPINGS = callfacts.containers(callfacts.MAPPING)
_COLLECTIONS = callfacts.containers(callfacts.COLLECTION)
#: The containers holding each argument at its own position: ``tuple[A, B]``.
_RECORDS = callfacts.containers(callfacts.RECORD)


def _slot_of(annotation: ast.expr, slot: Optional[int]) -> Optional[ast.expr]:
    """Slot *slot* of a ``tuple[A, B]`` annotation; ``None`` for any other, or ``tuple[A, ...]``."""
    annotation = _unquoted(annotation)
    if (
        slot is None
        or not isinstance(annotation, ast.Subscript)
        or getattr(annotation.value, "id", getattr(annotation.value, "attr", "")) not in _RECORDS
        or not isinstance(annotation.slice, ast.Tuple)
        or slot >= len(annotation.slice.elts)
        or any(isinstance(one, ast.Constant) and one.value is Ellipsis for one in annotation.slice.elts)
    ):
        return None
    return annotation.slice.elts[slot]


#: What a context manager is declared to yield its element through: a context manager, or what a generator function
#: (a ``contextmanager``) is declared to return.
_YIELDING = callfacts.containers(callfacts.CONTEXT) | {
    name for name, one in callfacts.CONTAINERS.items() if one.generator
}

#: The classes a declaration states -- one, or each member of a union.
Classes = frozenset[tuple[str, ast.ClassDef]]

#: What a declaration binds type parameters to: (``id`` of the class taking it, its name) -> (file, the argument).
Bound = dict[tuple[int, str], tuple[str, ast.expr]]


def _head_name(expression: ast.expr) -> str:
    """The name a ``Name`` or the last part of an ``Attribute`` spells: ``Generic`` for ``typing.Generic``."""
    return expression.attr if isinstance(expression, ast.Attribute) else getattr(expression, "id", "")


def _subscript_args(annotation: ast.Subscript) -> list[ast.expr]:
    """The arguments of ``X[A]`` or ``X[A, B]``."""
    inner = annotation.slice
    return list(inner.elts) if isinstance(inner, ast.Tuple) else [inner]


def _one(classes: Optional[Classes]) -> Optional[tuple[str, ast.ClassDef]]:
    """The class of a declaration that states exactly one."""
    if classes is None or len(classes) != 1:
        return None
    return next(iter(classes))


def _class_key(one: tuple[str, ast.ClassDef]) -> tuple[str, str, int]:
    return one[0], one[1].name, one[1].lineno


def _bound_key(bound: tuple[tuple[str, ast.ClassDef], ...]) -> tuple[tuple[str, int], ...]:
    return tuple((file, cls.lineno) for file, cls in bound)


def _hands_on(call: ast.Call, var: str, kw: Optional[str]) -> bool:
    """Whether *call* hands ``*var`` and ``**kw`` and nothing else -- ``*var`` alone where *kw* is ``None``."""
    star = call.args[0] if len(call.args) == 1 else None
    return (
        isinstance(star, ast.Starred)
        and isinstance(star.value, ast.Name)
        and star.value.id == var
        and [(one.arg, getattr(one.value, "id", None)) for one in call.keywords] == ([(None, kw)] if kw else [])
    )


def _called_on(statement: ast.stmt) -> Optional[tuple[str, str]]:
    """``("thr", "start")`` for the statement ``thr.start()``; ``None`` for any other."""
    call = statement.value if isinstance(statement, ast.Expr) else None
    if (
        isinstance(call, ast.Call)
        and not call.args
        and not call.keywords
        and isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
    ):
        return call.func.value.id, call.func.attr
    return None


def _reads_self(call: ast.Call) -> bool:
    """Whether where *call* lands can depend on the class of ``self``.

    A call on ``self`` or ``cls``, through ``super()``, or handing ``self``
    to a method read off its class (``Base.m(self, ...)``).
    """
    called = call.func
    if not isinstance(called, ast.Attribute):
        return False
    receiver = called.value
    if isinstance(receiver, ast.Name) and receiver.id in ("self", "cls"):
        return True
    if isinstance(receiver, ast.Call) and isinstance(receiver.func, ast.Name) and receiver.func.id == "super":
        return True
    return bool(call.args) and isinstance(call.args[0], ast.Name) and call.args[0].id == "self"


def _grouped(found_for: list[tuple[Target, tuple[str, ast.ClassDef]]]) -> tuple[Target, ...]:
    """One target per definition, in the order first found, bound to every class found to run it."""
    order: list[tuple[str, int]] = []
    held: dict[tuple[str, int], tuple[Target, list[tuple[str, ast.ClassDef]]]] = {}
    for target, one in found_for:
        key = (target.file, id(target.func))
        if key not in held:
            held[key] = (target, [])
            order.append(key)
        classes = held[key][1]
        for each in target.bound or (one,):
            if each not in classes:
                classes.append(each)
    return tuple(held[key][0]._replace(bound=tuple(sorted(held[key][1], key=_class_key))) for key in order)


def _members(annotation: ast.expr) -> list[ast.expr]:
    """Every member of a union annotation, through quotes and nested unions, ``None`` left out."""
    annotation = _unquoted(annotation)
    members = _union_members(annotation)
    if members is None:
        return [] if isinstance(annotation, ast.Constant) and annotation.value is None else [annotation]
    return [one for member in members for one in _members(member)]


def _annotation_names(annotation: ast.expr) -> list[ast.expr]:
    """Every name and dotted name an annotation spells, through quotes: ``"dict[str, PassiveProxies] | None"`` has
    ``dict``, ``str`` and ``PassiveProxies``."""
    found: list[ast.expr] = []
    for node in walk(_unquoted(annotation)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            unquoted = _unquoted(node)
            if unquoted is not node:
                found.extend(_annotation_names(unquoted))
        elif isinstance(node, (ast.Name, ast.Attribute)):
            found.append(node)
    return found


def _union_members(annotation: ast.expr) -> Optional[list[ast.expr]]:
    """The members of ``A | B``, ``Union[A, B]`` or ``Optional[A]``; ``None`` for anything else."""
    if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
        return [annotation.left, annotation.right]
    if isinstance(annotation, ast.Subscript):
        wrapper = annotation.value
        name = wrapper.attr if isinstance(wrapper, ast.Attribute) else getattr(wrapper, "id", "")
        if name == "Optional":
            return [annotation.slice]
        if name == "Union":
            inner = annotation.slice
            return list(inner.elts) if isinstance(inner, ast.Tuple) else [inner]
    return None


def _module_statements(body: list[ast.stmt]):
    """Every statement run at module level: the module's own, and those under its ``if``, ``try`` and ``with``."""
    for node in body:
        yield node
        if isinstance(node, ast.If):
            yield from _module_statements(node.body)
            yield from _module_statements(node.orelse)
        elif isinstance(node, ast.Try):
            yield from _module_statements(node.body)
            for handler in node.handlers:
                yield from _module_statements(handler.body)
            yield from _module_statements(node.orelse)
            yield from _module_statements(node.finalbody)
        elif isinstance(node, ast.With):
            yield from _module_statements(node.body)


class Types:
    """What class the code *declares* a receiver to be.

    Read the way a type checker reads it -- an annotation, a constructor, the
    return annotation of the method called -- and nothing else.  The walk
    needs it for one question, which system is on the other side of a call:
    ``self.site_mapper.getSite(x)`` and ``self.siteMapper.getSite(x)`` are one
    interface spelled two ways, and matching the spelling classified one and
    walked straight through the other.  What the object *is* does not depend
    on what someone called it.

    Where the code declares ``Any``, a type checker learns nothing more; this
    asks the calls what they hand (:meth:`Calls.handed_classes`), since PanDA
    writes ``Any`` where the classes that arrive differ by caller.

    Its own record of what it read, apart from the walk's: reading a base
    class or the interface's own module to learn a type is not a step on the
    path, so it does not spend the walk's file budget.  The trees are the
    same ones -- :func:`parsed` -- because ``pathcond.class_of`` finds a
    method by identity.
    """

    #: How deep a declaration is followed through fields, locals and calls.
    #: JEDI's ``self.siteMapper`` is four links from what declares it -- the
    #: field's assignment, the call, ``self.taskBufferIF``'s assignment, the
    #: ``__init__`` parameter's annotation -- and each of those is two levels
    #: here, so four stopped one short of the answer.
    DEPTH = 8

    #: How many classes a lineage holds before it stops.  ``DBProxy`` is
    #: thirteen mixins and itself; at twelve the last two were cut, and 38
    #: ``proxy.<method>()`` calls read as "the declared class has no such
    #: method".
    LINEAGE = 40

    def __init__(
        self,
        roots: dict[str, Path],
        walked: dict[str, tuple[str, ast.Module]],
        recorded: Optional[Recorded] = None,
    ):
        self.roots = roots
        self.walked = walked
        #: The instance's recorded values of module names (:func:`read_recorded`); empty reads the code alone.
        self.recorded: Recorded = recorded or {}
        self.parsed: dict[str, Optional[ast.Module]] = {}
        self._imports: dict[str, dict[str, tuple[str, str]]] = {}
        #: :meth:`_module_bound` per file.
        self._module_bindings: dict[str, dict[str, list[ast.expr]]] = {}
        self._assigned: dict[int, dict[str, list[ast.expr]]] = {}
        #: :meth:`bound` per function.
        self._bound: dict[int, dict[str, list[ast.expr]]] = {}
        # (file, id(func), name, depth) -> (func, classes): see ``declared``.
        self._names: dict[tuple, tuple[ast.AST, Optional[Classes]]] = {}
        self._files: dict[int, str] = {}
        self._mro: dict[int, tuple[ast.ClassDef, list[tuple[str, ast.ClassDef]]]] = {}
        #: :meth:`annotated_libraries`, once a tree.
        self._annotated: Optional[frozenset[tuple[str, str]]] = None
        #: The classes the calls of a function hand its parameter declared
        #: ``Any`` -- :meth:`Calls.handed_classes`, set by the :class:`Calls`
        #: that reads this tree; ``None`` reads the declaration alone.
        self.handed_to: Optional[Callable[[str, ast.AST, str, int], Optional[Classes]]] = None
        unbound = [f"{file} {name}" for file, name in sorted(self.recorded) if name not in self._module_names(file)]
        if unbound:
            # A record whose name the corpus does not bind would be read nowhere, and the build would read the
            # code alone under the record's stamp: a module moved or a name renamed since the record was taken.
            raise ValueError(f"the instance record names what no module-level assignment binds: {', '.join(unbound)}")

    def is_builtin(self, file: str, func: ast.AST, name: str) -> bool:
        """Whether *name*, read in *func* of *file*, is Python's builtin of the name (:func:`is_builtin_in`)."""
        return is_builtin_in(func, self.tree(file), name)

    def _module_names(self, file: str) -> set[str]:
        """The names an assignment at *file*'s module level binds, whatever the value."""
        tree = self.tree(file)
        names: set[str] = set()
        for node in _module_statements(tree.body) if tree is not None else ():
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            else:
                continue
            names.update(target.id for target in targets if isinstance(target, ast.Name))
        return names

    def assigned(self, func: ast.AST) -> dict[str, list[ast.expr]]:
        """``pathcond.assigned_expressions``, once per function."""
        if id(func) not in self._assigned:
            self._assigned[id(func)] = pathcond.assigned_expressions(func)
        return self._assigned[id(func)]

    def bound(self, func: ast.AST) -> dict[str, list[ast.expr]]:
        """The values a name is bound to, read for what it is: :meth:`assigned` less the right side of ``x op= y``.

        An augmented assignment leaves a name what it was: a builtin's in-place
        operator returns the object itself (a list's ``+=``) or a value of its
        type, and no class of the corpus defines one; its right side is a piece
        the text readers want (``errMsg += ...``), not a value the name takes.
        ``candidateSpecList = []`` ... ``candidateSpecList += sitesWithWeight``
        is a list.  A class of the corpus defining the binary operator would be
        read past -- ``x *= y`` on a ``GlobalShares`` is a float, its
        ``__mul__``'s -- as leaving ``x`` what it was.
        """
        if id(func) not in self._bound:
            augmented = {id(node.value) for node in walk(func) if isinstance(node, ast.AugAssign)}
            self._bound[id(func)] = {
                name: kept
                for name, values in self.assigned(func).items()
                if (kept := [value for value in values if id(value) not in augmented])
            }
        return self._bound[id(func)]

    # -- modules -----------------------------------------------------------

    def tree(self, file: str) -> Optional[ast.Module]:
        if file in self.walked:
            return self.walked[file][1]
        if file not in self.parsed:
            package, _, relative = file.partition("/")
            root = self.roots.get(package)
            held = parsed(root / relative) if root is not None and relative else None
            self.parsed[file] = held[1] if held is not None else None
        return self.parsed[file]

    def file_of(self, node: ast.AST) -> Optional[str]:
        """The corpus file *node* belongs to, if it is in a tree this reader holds."""
        module = node
        while getattr(module, "parent", None) is not None:
            module = module.parent  # type: ignore[attr-defined]
        if not isinstance(module, ast.Module):
            return None
        if id(module) not in self._files:
            self._files.update({id(tree): file for file, (_sha, tree) in self.walked.items()})
            self._files.update({id(tree): file for file, tree in self.parsed.items() if tree is not None})
        return self._files.get(id(module))

    def corpus_files(self) -> list[str]:
        """Every module of the corpus, as the map spells a file; tests left out, by the build's rule (:func:`is_test_module`)."""
        found: list[str] = []
        for package, root in sorted(self.roots.items()):
            for path in sorted(root.rglob("*.py")):
                file = f"{package}/{path.relative_to(root)}"
                if not is_test_module(file):
                    found.append(file)
        return found

    def enclosing_function(self, file: str, node: ast.AST) -> Optional[ast.AST]:
        """The function a name *node* does not bind is read from next, or ``None`` at module scope.

        Python's scoping: a class body is no scope for the functions in it, so
        a method of a class a function defines reads that function's names --
        ``datasetManager.main`` defines ``setTobeDeletedToDis`` and the class
        ``CloserThr``, whose ``run`` calls it.
        """
        parent = getattr(node, "parent", None)
        while parent is not None:
            if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return parent
            if isinstance(parent, ast.Module):
                return None
            parent = getattr(parent, "parent", None)
        return None

    def method_class(self, file: str, func: ast.AST) -> Optional[tuple[str, ast.ClassDef]]:
        """The class whose ``self`` *func* reads: its own, or its enclosing method's.

        ``getJobs`` builds its ``WHERE`` in a closure, ``build_where_clause``,
        which calls ``self.construct_where_clause`` -- the ``self`` of the
        method it is nested in.
        """
        node: Optional[ast.AST] = func
        while node is not None:
            parent = getattr(node, "parent", None)
            if isinstance(parent, ast.ClassDef) and node in parent.body:
                return file, parent
            if isinstance(parent, ast.Module) or parent is None:
                break
            node = parent
        tree = self.tree(file)
        if tree is not None:
            cls = pathcond.class_of(tree, func)
            if cls is not None:
                return file, cls
        return None

    def imported_from_outside(self, file: str, name: str) -> bool:
        """Whether *name* in *file* is imported from a module this corpus does not hold."""
        held = self.imports(file).get(name)
        if held is None:
            return False
        module = held[0]
        return module.split(".")[0] not in self.roots

    def module_named(self, file: str, expression: ast.expr) -> Optional[str]:
        """The file of the module *expression* names in *file*, when it names one."""
        if not isinstance(expression, ast.Name):
            return None
        held = self.imports(file).get(expression.id)
        if held is None:
            return None
        module, attribute = held
        if not attribute:
            return self.module_file(module)
        return self.module_file(f"{module}.{attribute}")

    def module_file(self, dotted: str) -> Optional[str]:
        """``pandaserver.brokerage.SiteMapper`` -> its file, when the tree holds it."""
        parts = dotted.split(".")
        if not parts or parts[0] not in self.roots:
            return None
        for candidate in ("/".join(parts) + ".py", "/".join(parts) + "/__init__.py"):
            if self.tree(candidate) is not None:
                return candidate
        return None

    def imports(self, file: str) -> dict[str, tuple[str, str]]:
        """``{local name: (module, attribute)}`` for every import in *file*.

        Every import, wherever it sits: ``if TYPE_CHECKING:`` is where PanDA
        puts the ones only an annotation needs, and those are the ones this
        reads.  A relative import is resolved against the file's own package.
        """
        if file in self._imports:
            return self._imports[file]
        found: dict[str, tuple[str, str]] = {}
        tree = self.tree(file)
        package = file[: -len(".py")].split("/")[:-1] if file.endswith(".py") else []
        for node in walk(tree) if tree is not None else ():
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname:
                        found[alias.asname] = (alias.name, "")
                    else:
                        head = alias.name.split(".")[0]
                        found.setdefault(head, (head, ""))
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = package[: len(package) - (node.level - 1)] if node.level > 1 else package
                    module = ".".join([*base, *([node.module] if node.module else [])])
                else:
                    module = node.module or ""
                for alias in node.names:
                    found[alias.asname or alias.name] = (module, alias.name)
        self._imports[file] = found
        return found

    def library_class(self, file: str, expression: ast.expr) -> Optional[tuple[str, str]]:
        """``(module, name)`` of the imported class *expression* names in *file*, or ``None``.

        ``threading.Thread`` under ``import threading``, ``Thread`` under
        ``from threading import Thread``, ``msg_processor.MsgProcAgentBase``
        under ``from pandacommon.pandamsgbkr import msg_processor`` (the
        module ``pandacommon.pandamsgbkr.msg_processor``'s, as under ``import
        ... as msg_processor``; a class nested in an imported class reads the
        same): read through the imports alone, so a name bound any other way
        is ``None``.
        """
        if isinstance(expression, ast.Name):
            held = self.imports(file).get(expression.id)
            return held if held is not None and held[1] else None
        if isinstance(expression, ast.Attribute) and isinstance(expression.value, ast.Name):
            held = self.imports(file).get(expression.value.id)
            if held is not None:
                return (f"{held[0]}.{held[1]}" if held[1] else held[0]), expression.attr
        return None

    # -- classes -----------------------------------------------------------

    def class_named(self, file: str, expression: ast.expr, depth: int = 0) -> Optional[tuple[str, ast.ClassDef]]:
        """The class *expression* names in *file*: declared there, imported, or bound to a name there.

        A name the module neither declares nor imports is read from what the
        module binds it to -- ``_Proxied: Any = JediTaskBuffer``, so that
        ``class JediTaskBufferInterface(_Proxied)`` has its base.  The value
        is what the name is, whatever the annotation says: ``Any`` says
        nothing.  A binding to no class of the corpus -- ``_Proxied =
        object`` in the other branch -- is passed over; two bindings to two
        corpus classes name neither.
        """
        if depth > self.DEPTH:
            return None
        tree = self.tree(file)
        if tree is None:
            return None
        if isinstance(expression, ast.Name):
            # A class a function defines is a name of that function's, read
            # from where the name is written: ``finThr = FinisherThr(...)`` in
            # ``datasetManager.main``.  A name made here for the lookup has no
            # place, and is the module's.
            scope = self.enclosing_function(file, expression)
            while scope is not None:
                node = binding(bound_defs(scope), expression.id)
                if node is not None:
                    return (file, node) if isinstance(node, ast.ClassDef) else None
                scope = self.enclosing_function(file, scope)
            node = binding((one for one in tree.body if isinstance(one, ast.ClassDef)), expression.id)
            if isinstance(node, ast.ClassDef):
                return file, node
            held = self.imports(file).get(expression.id)
            if held is None:
                return self._aliased(file, expression.id, depth)
            module, attribute = held
            if not attribute:
                return None
            # ``from pkg.mod import Cls`` -- or ``from pkg import mod``, a module.
            target = self.module_file(module)
            if target is not None:
                return self.class_named(target, ast.Name(id=attribute, ctx=ast.Load()), depth + 1)
            return None
        if isinstance(expression, ast.Attribute) and isinstance(expression.value, ast.Name):
            # ``SiteMapper.SiteMapper`` -- a module imported, then its class.
            held = self.imports(file).get(expression.value.id)
            if held is None:
                return None
            module, attribute = held
            dotted = f"{module}.{attribute}" if attribute else module
            target = self.module_file(dotted)
            if target is None:
                return None
            return self.class_named(target, ast.Name(id=expression.attr, ctx=ast.Load()), depth + 1)
        return None

    def _aliased(self, file: str, name: str, depth: int) -> Optional[tuple[str, ast.ClassDef]]:
        """The corpus class every module-level binding of *name* in *file* that names one names."""
        found = {
            named
            for value in self._module_bound(file).get(name, [])
            if (named := self.class_named(file, value, depth + 1)) is not None
        }
        return found.pop() if len(found) == 1 else None

    def _module_bound(self, file: str) -> dict[str, list[ast.expr]]:
        """``{name: [value]}`` for each ``name = value`` / ``name: T = value`` run at module level, the value a name or ``a.b``."""
        if file not in self._module_bindings:
            found: dict[str, list[ast.expr]] = {}
            tree = self.tree(file)
            for node in _module_statements(tree.body) if tree is not None else ():
                if isinstance(node, ast.Assign):
                    names = [target.id for target in node.targets if isinstance(target, ast.Name)]
                elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
                    names = [node.target.id]
                else:
                    continue
                if isinstance(node.value, (ast.Name, ast.Attribute)):
                    for one in names:
                        found.setdefault(one, []).append(node.value)
            self._module_bindings[file] = found
        return self._module_bindings[file]

    def annotated(self, file: str, annotation: Optional[ast.expr]) -> Optional[tuple[str, ast.ClassDef]]:
        """The class an annotation states, where it states exactly one -- :meth:`annotated_all`."""
        return _one(self.annotated_all(file, annotation))

    def annotated_all(self, file: str, annotation: Optional[ast.expr]) -> Optional[Classes]:
        """Every corpus class an annotation states, through quotes, ``Optional`` and unions.

        ``WrappedOracleConn | WrappedPostgresConn`` is both; ``Foo | None`` and
        ``Foo | str`` are ``Foo`` -- a member naming no class of the corpus
        says nothing this reading can follow.  ``None`` where it states none.
        """
        if annotation is None:
            return None
        annotation = _unquoted(annotation)
        members = _union_members(annotation)
        if members is not None:
            found: set[tuple[str, ast.ClassDef]] = set()
            for member in members:
                if isinstance(member, ast.Constant) and member.value is None:
                    continue
                found |= self.annotated_all(file, member) or frozenset()
            return frozenset(found) or None
        if isinstance(annotation, ast.Subscript):
            # ``FactoryBase[PostProcessorBase]``: a class of the corpus taking
            # type parameters, given them, is that class (what it binds them
            # to, :meth:`_binds`).  ``list[X]`` states no class: a container.
            head = annotation.value
            named = self.class_named(file, head) if isinstance(head, (ast.Name, ast.Attribute)) else None
            return frozenset({named}) if named is not None and self.type_parameters(*named) else None
        named = self.class_named(file, annotation)
        return frozenset({named}) if named is not None else None

    def type_parameters(self, file: str, cls: ast.ClassDef) -> tuple[str, ...]:
        """The type parameters *cls* takes, as its ``Generic[...]`` base names them: ``("T",)`` for ``FactoryBase``.

        Read off the class itself: a class handing its base a parameter of
        its own (``class Sub(Base[T])``) takes none here, and the corpus has
        none -- its two generic classes, ``FactoryBase`` and ``SpecColumn``,
        are subclassed with a class given (``FactoryBase[WatchDogBase]``).
        """
        for base in cls.bases:
            if isinstance(base, ast.Subscript) and _head_name(base.value) == "Generic":
                return tuple(one.id for one in _subscript_args(base) if isinstance(one, ast.Name))
        return ()

    def _binds(self, file: str, annotation: Optional[ast.expr]) -> Bound:
        """What *annotation* binds type parameters to: ``FactoryBase[PostProcessorBase]`` binds ``FactoryBase``'s ``T``."""
        if annotation is None:
            return {}
        annotation = _unquoted(annotation)
        members = _union_members(annotation)
        if members is not None:
            bound: Bound = {}
            for member in members:
                bound.update(self._binds(file, member))
            return bound
        if not isinstance(annotation, ast.Subscript) or not isinstance(annotation.value, (ast.Name, ast.Attribute)):
            return {}
        named = self.class_named(file, annotation.value)
        if named is None:
            return {}
        parameters = self.type_parameters(*named)
        arguments = _subscript_args(annotation)
        if not parameters or len(parameters) != len(arguments):
            return {}
        return {(id(named[1]), name): (file, argument) for name, argument in zip(parameters, arguments, strict=True)}

    def _binds_of_class(self, file: str, cls: ast.ClassDef) -> Bound:
        """What *cls*'s bases bind type parameters to, the nearest first: ``class WatchDog(JediKnight, FactoryBase[WatchDogBase])``."""
        bound: Bound = {}
        for where, holder in self.lineage(file, cls):
            for base in holder.bases:
                for key, value in self._binds(where, base).items():
                    bound.setdefault(key, value)
        return bound

    def _annotated_bound(
        self, file: str, annotation: Optional[ast.expr], holder: ast.ClassDef, bound: Bound
    ) -> Optional[Classes]:
        """:meth:`annotated_all` of an annotation written in *holder*, each of its type parameters read as *bound* binds it.

        ``instantiateImpl(...) -> T | None`` in ``FactoryBase``, on a
        ``FactoryBase[PostProcessorBase]``, is a ``PostProcessorBase``.  A
        parameter nothing binds names no class, as before.
        """
        if annotation is None:
            return None
        annotation = _unquoted(annotation)
        members = _union_members(annotation)
        if members is not None:
            found: set[tuple[str, ast.ClassDef]] = set()
            for member in members:
                if isinstance(member, ast.Constant) and member.value is None:
                    continue
                found |= self._annotated_bound(file, member, holder, bound) or frozenset()
            return frozenset(found) or None
        if isinstance(annotation, ast.Name) and (id(holder), annotation.id) in bound:
            return self.annotated_all(*bound[(id(holder), annotation.id)])
        return self.annotated_all(file, annotation)

    def _receiver_binds(
        self, file: str, func: ast.AST, receiver: ast.expr, owner: tuple[str, ast.ClassDef]
    ) -> Bound:
        """What a call's receiver binds type parameters to: its class's bases, then its own declaration's arguments.

        ``jobBroker.requireImpl(...)`` on a ``JobBroker(FactoryBase[JobBrokerBase])``;
        ``self.implFactory.instantiateImpl(...)`` where ``implFactory: FactoryBase[PostProcessorBase]``.
        """
        bound = self._binds_of_class(*owner)
        if not (isinstance(receiver, ast.Name) and receiver.id == "self"):
            held = self._declaring_annotation(file, func, receiver)
            if held is not None:
                bound = {**bound, **self._binds(*held)}
        return bound

    def _declaring_annotation(self, file: str, func: ast.AST, expression: ast.expr) -> Optional[tuple[str, ast.expr]]:
        """:meth:`_annotation_of`, a field read through what every assignment of it was declared as.

        ``self.implFactory = implFactory`` in ``__init__``, the parameter
        declared ``FactoryBase[PostProcessorBase]``: the field carries that
        declaration, as an alias does in :meth:`_annotation_of`.  Asked only
        for what a receiver binds type parameters to.
        """
        held = self._annotation_of(file, func, expression)
        if held is not None or not (
            isinstance(expression, ast.Attribute)
            and isinstance(expression.value, ast.Name)
            and expression.value.id == "self"
        ):
            return held
        owner = self.method_class(file, func)
        if owner is None:
            return None
        found: list[Optional[tuple[str, ast.expr]]] = []
        for where, cls in self.lineage(*owner):
            for method in pathcond.methods_of(cls):
                for value in pathcond.attribute_expressions(method, expression.attr):
                    if isinstance(value, ast.Constant) and value.value is None:
                        continue
                    found.append(self._annotation_of(where, method, value))
        texts = {ast.unparse(one[1]) for one in found if one is not None}
        if found and len(texts) == 1 and all(one is not None for one in found):
            return found[0]
        return None

    def _field_read(self, file: str, annotation: ast.expr) -> Optional[Classes]:
        """The classes an instance's field declared *annotation* reads as: a descriptor's (one defining ``__get__``) by what ``__get__`` returns.

        ``PandaID: SpecColumn[int]`` reads as ``_T | Null`` with ``_T`` the
        ``int`` (``spec_column.py``): no class of the corpus, as it read
        before a subscripted annotation stated a class.
        """
        declared = self.annotated_all(file, annotation)
        if declared is None:
            return None
        found: set[tuple[str, ast.ClassDef]] = set()
        for where, cls in declared:
            getter = self._method_in(where, cls, "__get__")
            if getter is None:
                found.add((where, cls))
                continue
            got = self._annotated_bound(getter[0], getter[2].returns, getter[1], self._binds(file, annotation))
            found |= got or frozenset()
        return frozenset(found) or None

    def states_any(self, file: str, annotation: Optional[ast.expr], depth: int = 0) -> bool:
        """Whether *annotation* declares ``Any`` and nothing else, ``| None`` aside -- spelled so, or by an alias of it.

        ``TaskBufferLike: TypeAlias = Any`` in ``pandaserver.asyncprocess``
        is one: the alias says why the parameter is left open.
        """
        if annotation is None or depth > 2:
            return False
        annotation = _unquoted(annotation)
        members = _union_members(annotation)
        if members is not None:
            rest = [one for one in members if not (isinstance(one, ast.Constant) and one.value is None)]
            return bool(rest) and all(self.states_any(file, one, depth) for one in rest)
        if isinstance(annotation, ast.Attribute):
            return annotation.attr == "Any"
        if not isinstance(annotation, ast.Name):
            return False
        held = self.imports(file).get(annotation.id)
        if annotation.id == "Any":
            return held is None or held[0] == "typing"
        where, alias = file, annotation.id
        if held is not None:
            where, alias = self.module_file(held[0]) or "", held[1] or annotation.id
        tree = self.tree(where) if where else None
        for node in tree.body if tree is not None else ():
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == alias:
                return node.value is not None and self.states_any(where, node.value, depth + 1)
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == alias for t in node.targets):
                return self.states_any(where, node.value, depth + 1)
        return False

    def lineage(self, file: str, cls: ast.ClassDef) -> list[tuple[str, ast.ClassDef]]:
        """*cls* and its bases, nearest first, followed across files through imports."""
        order: list[tuple[str, ast.ClassDef]] = [(file, cls)]
        queue = [(file, base) for base in cls.bases]
        seen = {(file, cls.name)}
        while queue and len(order) < self.LINEAGE:
            where, base = queue.pop(0)
            found = self.class_named(where, pathcond.unsubscripted(base))
            if found is None or (found[0], found[1].name) in seen:
                continue
            seen.add((found[0], found[1].name))
            order.append(found)
            queue.extend((found[0], one) for one in found[1].bases)
        return order

    def built_outside(self, file: str, cls: ast.ClassDef) -> bool:
        """Whether a class in *cls*'s order has a base outside the corpus that may supply methods.

        A base is read as :meth:`lineage` reads it, a subscript taken off --
        ``FactoryBase[JobBrokerBase]`` is FactoryBase.  ``object`` and the
        typing bases a class is declared generic or a protocol with supply no
        method.
        """
        for where, holder in self.mro(file, cls):
            for base in holder.bases:
                head = pathcond.unsubscripted(base)
                named = head.attr if isinstance(head, ast.Attribute) else getattr(head, "id", "")
                if named in ("object", "Generic", "Protocol") or self.class_named(where, head) is not None:
                    continue
                return True
        return False

    def outside_bases(self, file: str, cls: ast.ClassDef) -> Optional[frozenset[tuple[str, str]]]:
        """``(module, name)`` of each base outside the corpus in *cls*'s order (:meth:`built_outside`'s), read through
        the imports (:meth:`library_class`); ``None`` where one cannot be read so -- ``class MySMTP(smtplib.SMTP)``
        is ``("smtplib", "SMTP")``."""
        found: set[tuple[str, str]] = set()
        for where, holder in self.mro(file, cls):
            for base in holder.bases:
                head = pathcond.unsubscripted(base)
                named = head.attr if isinstance(head, ast.Attribute) else getattr(head, "id", "")
                if named in ("object", "Generic", "Protocol") or self.class_named(where, head) is not None:
                    continue
                library = self.library_class(where, head) if isinstance(head, (ast.Name, ast.Attribute)) else None
                if library is None:
                    return None
                found.add(library)
        return frozenset(found) or None

    def library_classes(self, file: str, func: ast.AST, expression: ast.expr) -> Optional[frozenset[tuple[str, str]]]:
        """``(module, name)`` of each class outside the corpus *expression*'s object can be, where every one it can be
        is such a class; ``None`` otherwise.

        Declared so -- a field's, a parameter's or a local's annotation, each member of a union a class outside the
        corpus, none typing's or collections' (their values are the builtins') -- or a local bound only to calls of
        a name outside the corpus that the corpus uses as an annotation somewhere.  A call alone cannot tell a class
        from a function there: ``WeightedLists(lock)`` and ``json.loads(text)`` read the same, and the corpus does
        not hold their modules.  An annotation can: AdderThread declares ``job_output_reports: WeightedLists``.
        """
        declared = self._annotation_of(file, func, expression)
        if declared is not None:
            where, annotation = declared
            found: set[tuple[str, str]] = set()
            for member in _members(annotation):
                head = member.value if isinstance(member, ast.Subscript) else member
                library = self.library_class(where, head) if isinstance(head, (ast.Name, ast.Attribute)) else None
                if library is None or not self._library(library):
                    return None
                found.add(library)
            return frozenset(found) or None
        if not isinstance(expression, ast.Name):
            return None
        values = self.bound(func).get(expression.id)
        if not values:
            return None
        found = set()
        for value in values:
            if not isinstance(value, ast.Call) or not isinstance(value.func, (ast.Name, ast.Attribute)):
                return None
            library = self.library_class(file, value.func)
            if library is None or not self._library(library) or library not in self.annotated_libraries():
                return None
            found.add(library)
        return frozenset(found)

    def _library(self, library: tuple[str, str]) -> bool:
        top = library[0].split(".")[0]
        return top not in self.roots and top not in ("typing", "collections", "builtins")

    def annotated_libraries(self) -> frozenset[tuple[str, str]]:
        """``(module, name)`` of each name outside the corpus the corpus uses in an annotation: a parameter's, a
        return's, a local's or a field's, quoted or not."""
        if self._annotated is None:
            found: set[tuple[str, str]] = set()
            for file in self.corpus_files():
                tree = self.tree(file)
                for node in walk(tree) if tree is not None else ():
                    annotations: list[ast.expr] = []
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        arguments = node.args
                        every = [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]
                        every += [one for one in (arguments.vararg, arguments.kwarg) if one is not None]
                        annotations += [one.annotation for one in every if one.annotation is not None]
                        if node.returns is not None:
                            annotations.append(node.returns)
                    elif isinstance(node, ast.AnnAssign):
                        annotations.append(node.annotation)
                    for annotation in annotations:
                        for named in _annotation_names(annotation):
                            library = self.library_class(file, named)
                            if library is not None and self._library(library):
                                found.add(library)
            self._annotated = frozenset(found)
        return self._annotated

    def mro(self, file: str, cls: ast.ClassDef) -> list[tuple[str, ast.ClassDef]]:
        """*cls* and its corpus bases in the order Python resolves a method on an object of *cls*.

        :meth:`lineage` reads the bases breadth first, and in the corpus that
        finds the same definition of every name a lookup asks.  ``super()``
        needs the order itself: in a class with several bases it runs the
        class after the caller's in the order of the object's class, so
        ``OraDBProxy.DBProxy``'s thirteen mixins each hand the call on to the
        next.  C3, as Python builds it; a base outside the corpus stands for
        itself, its own bases unknown.  Where the bases admit no such order,
        :meth:`lineage`.
        """
        held = self._mro.get(id(cls))
        if held is not None and held[0] is cls:
            return held[1]
        # A class that reaches itself through its bases reads its lineage.
        self._mro[id(cls)] = (cls, self.lineage(file, cls))
        found = self._linearized(file, cls)
        order = found if found is not None else self.lineage(file, cls)
        self._mro[id(cls)] = (cls, order)
        return order

    def _linearized(self, file: str, cls: ast.ClassDef) -> Optional[list[tuple[str, ast.ClassDef]]]:
        """C3 over *cls*'s bases, ``None`` where they admit no order."""
        held: dict[object, tuple[str, ast.ClassDef]] = {}

        def token(one: tuple[str, ast.ClassDef]) -> object:
            held[id(one[1])] = one
            return id(one[1])

        direct: list[object] = []
        sequences: list[list[object]] = []
        for base in cls.bases:
            found = self.class_named(file, pathcond.unsubscripted(base))
            if found is None:
                outside = f"<outside>{ast.unparse(base)}"
                direct.append(outside)
                sequences.append([outside])
                continue
            direct.append(token(found))
            sequences.append([token(one) for one in self.mro(*found)])
        sequences.append(list(direct))
        merged: list[object] = [token((file, cls))]
        while True:
            sequences = [one for one in sequences if one]
            if not sequences:
                break
            for sequence in sequences:
                head = sequence[0]
                if not any(head in other[1:] for other in sequences):
                    break
            else:
                return None
            merged.append(head)
            for sequence in sequences:
                if sequence[0] == head:
                    del sequence[0]
        return [held[one] for one in merged if one in held]

    def method(self, file: str, cls: ast.ClassDef, name: str):
        found = self._method_in(file, cls, name)
        return (found[0], found[2]) if found is not None else None

    def _method_in(
        self, file: str, cls: ast.ClassDef, name: str
    ) -> Optional[tuple[str, ast.ClassDef, ast.FunctionDef | ast.AsyncFunctionDef]]:
        """:meth:`method`, with the class of the lineage that defines it."""
        for where, holder in self.lineage(file, cls):
            node = _function_binding(holder.body, name)
            if node is not None:
                return where, holder, node
        return None

    # -- declarations ------------------------------------------------------

    def of_field(self, file: str, cls: ast.ClassDef, field: str, depth: int) -> Optional[Classes]:
        """``self.<field>``'s classes: its annotation in the class or a base, else what every assignment says."""
        lineage = self.lineage(file, cls)
        for where, holder in lineage:
            for node in holder.body:
                if (
                    isinstance(node, ast.AnnAssign)
                    and isinstance(node.target, ast.Name)
                    and node.target.id == field
                ):
                    return self._field_read(where, node.annotation)
            for method in pathcond.methods_of(holder):
                for node in walk(method):
                    if (
                        isinstance(node, ast.AnnAssign)
                        and isinstance(node.target, ast.Attribute)
                        and isinstance(node.target.value, ast.Name)
                        and node.target.value.id == "self"
                        and node.target.attr == field
                    ):
                        return self._field_read(where, node.annotation)
        if depth >= self.DEPTH:
            return None
        found: set[Classes] = set()
        for where, holder in lineage:
            for method in pathcond.methods_of(holder):
                for expression in pathcond.attribute_expressions(method, field):
                    if isinstance(expression, ast.Constant) and expression.value is None:
                        continue
                    one = self.declared(where, method, expression, depth + 1)
                    if one is None:
                        return None
                    found.add(one)
        return found.pop() if len(found) == 1 else None

    def _with_held(self, file: str, func: ast.AST, name: str, depth: int) -> Optional[Classes]:
        """The classes ``with <call> as <name>`` binds, from what the call is declared to yield.

        Every ``TaskBuffer`` method reaches the database through ``with
        self.proxyPool.get() as proxy``, and ``get`` is declared ``->
        Iterator["DBProxy.DBProxy"]``: the object is the element, not the
        iterator.
        """
        for node in walk(func):
            if not isinstance(node, (ast.With, ast.AsyncWith)):
                continue
            for item in node.items:
                held = item.optional_vars
                if not (isinstance(held, ast.Name) and held.id == name):
                    continue
                context = item.context_expr
                if not (isinstance(context, ast.Call) and isinstance(context.func, ast.Attribute)):
                    return self.declared(file, func, context, depth + 1)
                receiver = context.func.value
                if isinstance(receiver, ast.Name) and receiver.id == "self":
                    owner = self.method_class(file, func)
                else:
                    owner = self.of_expression(file, func, receiver, depth + 1)
                if owner is None:
                    return None
                found = self.method(owner[0], owner[1], context.func.attr)
                if found is None:
                    return None
                where, method = found
                returned = _unquoted(method.returns) if method.returns is not None else None
                if isinstance(returned, ast.Subscript):
                    wrapper = returned.value
                    wrapped = wrapper.attr if isinstance(wrapper, ast.Attribute) else getattr(wrapper, "id", "")
                    if wrapped in _YIELDING:
                        inner = returned.slice
                        if isinstance(inner, ast.Tuple) and inner.elts:
                            inner = inner.elts[0]
                        return self.annotated_all(where, inner)
                return self.annotated_all(where, returned)
        return None

    def _annotation_of(
        self, file: str, func: ast.AST, expression: ast.expr, seen: frozenset = frozenset()
    ) -> Optional[tuple[str, ast.expr]]:
        """The annotation *expression* was declared with: a field's, a parameter's or a local's."""
        if (
            isinstance(expression, ast.Attribute)
            and isinstance(expression.value, ast.Name)
            and expression.value.id == "self"
        ):
            holder = self.method_class(file, func)
            return self._field_annotation(holder, expression.attr) if holder is not None else None
        if isinstance(expression, ast.Name):
            args = getattr(func, "args", None)
            for argument in (*args.posonlyargs, *args.args, *args.kwonlyargs) if args else ():
                if argument.arg == expression.id and argument.annotation is not None:
                    return file, argument.annotation
            for node in walk(func):
                if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == expression.id:
                    return file, node.annotation
            # ``target = self.nuclei`` or ``target = self.satellites``: an
            # alias carries what every value it is bound to was declared as.
            bound = self.bound(func).get(expression.id, [])
            if bound and expression.id not in seen:
                found = [
                    self._annotation_of(file, func, value, seen | {expression.id}) for value in bound
                ]
                texts = {ast.unparse(one[1]) for one in found if one is not None}
                if len(texts) == 1 and all(one is not None for one in found):
                    return found[0]
        return None

    def _field_annotation(self, holder: tuple[str, ast.ClassDef], field: str) -> Optional[tuple[str, ast.expr]]:
        """The annotation *holder*'s class or a base declares *field* with: in the class body or as ``self.<field>: T``."""
        for where, cls in self.lineage(*holder):
            for node in cls.body:
                if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == field:
                    return where, node.annotation
            for method in pathcond.methods_of(cls):
                for node in walk(method):
                    if (
                        isinstance(node, ast.AnnAssign)
                        and isinstance(node.target, ast.Attribute)
                        and isinstance(node.target.value, ast.Name)
                        and node.target.value.id == "self"
                        and node.target.attr == field
                    ):
                        return where, node.annotation
        return None

    def _function_named(self, file: str, name: str) -> Optional[tuple[str, ast.AST]]:
        """The module-level function *name* is in *file*: defined there or imported."""
        tree = self.tree(file)
        if tree is None:
            return None
        node = _function_binding(tree.body, name)
        if node is not None:
            return file, node
        held = self.imports(file).get(name)
        if held is None or not held[1]:
            return None
        target = self.module_file(held[0])
        if target is None or target == file:
            return None
        other = self.tree(target)
        node = _function_binding(other.body, held[1]) if other is not None else None
        return (target, node) if node is not None else None

    def _outside_instance(self, file: str, name: str, depth: int) -> Optional[Classes]:
        """The classes of a name this function does not bind: a module's own instance.

        ``taskBuffer = TaskBuffer()`` at the foot of ``TaskBuffer.py`` and
        ``rucioAPI = RucioAPI()`` at the foot of ``ddm.py`` are what every
        daemon imports and calls; the module's own assignment says what it is.
        """
        if depth > self.DEPTH:
            return None
        tree = self.tree(file)
        if tree is None:
            return None
        for node in tree.body:
            # ``global_task_buffer: TaskBuffer = None``: the annotation is the
            # declaration, whatever the placeholder value is.
            if (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == name
            ):
                declared = self.annotated_all(file, node.annotation)
                if declared is not None:
                    return declared
        # ``x: Any = Foo()``: an annotation that declares nothing leaves the value to read, as ``x = Foo()``.
        values = [
            node.value
            for node in tree.body
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            and node.value is not None
            and any(isinstance(target, ast.Name) and target.id == name for target in pathcond.targets_of(node))
        ]
        if values:
            found = {self.declared(file, tree, value, depth + 1) for value in values}
            return found.pop() if len(found) == 1 else None
        return self._imported_instance(file, name, depth)

    def _imported_instance(self, file: str, name: str, depth: int) -> Optional[Classes]:
        """:meth:`_outside_instance` of the module *file* imports *name* from."""
        held = self.imports(file).get(name)
        if held is None or not held[1]:
            return None
        target = self.module_file(held[0])
        if target is None or target == file:
            return None
        return self._outside_instance(target, held[1], depth + 1)

    def _of_name(self, file: str, func: ast.AST, name: str, depth: int) -> Optional[Classes]:
        """:meth:`declared` for a bare name, read in *func*."""
        args = getattr(func, "args", None)
        for argument in (*args.posonlyargs, *args.args, *args.kwonlyargs) if args else ():
            if argument.arg == name:
                if self.handed_to is not None and self.states_any(file, argument.annotation):
                    return self.handed_to(file, func, name, depth)
                return self.annotated_all(file, argument.annotation)
        for node in walk(func):
            if (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == name
            ):
                return self.annotated_all(file, node.annotation)
        # Every assignment, less the ``None`` a name is cleared with:
        # ``siteMapper = SiteMapper(taskBuffer)`` and, at the end of the
        # daemon, ``siteMapper = None``.  The evidence for a local is all in
        # this function, so it is read rather than asked for; the answer
        # is classes only where every assignment says the same ones.  So
        # are the bindings an assignment is not (:meth:`_other_bindings`).
        bound = self._assigned_here(func, name)
        others = self._other_bindings(file, func, name, depth)
        held = self._with_held(file, func, name, depth)
        if not bound and held is None and not others:
            # A name a module's own body reads and does not bind is one it
            # imports: ``initializer.init()`` and ``taskBuffer.init(...)`` at
            # the top of ``panda.py``, a module's caller (:class:`CallSite`).
            if isinstance(func, ast.Module):
                return self._imported_instance(file, name, depth)
            return self._outside_instance(file, name, depth)
        found = {self.declared(file, func, value, depth + 1) for value in bound}
        found.update(one for one in others if one is not None)
        if held is not None:
            found.add(held)
        return found.pop() if len(found) == 1 else None

    def _assigned_here(self, func: ast.AST, name: str) -> list[ast.expr]:
        """What *func*'s own assignments bind *name* to (:meth:`bound`: an augmented one adds nothing), less the ``None`` it is cleared with.

        Not a function's, lambda's or class's nested in it: ``taskBuffer =
        ...`` inside ``add_main.main``'s inner function binds that function's
        own name.
        """
        nested = {id(one.value) for one in pathcond.bindings(func, name) if one.nested and one.value is not None}
        return [
            value
            for value in self.bound(func).get(name, [])
            if not (isinstance(value, ast.Constant) and value.value is None) and id(value) not in nested
        ]

    def _other_bindings(self, file: str, func: ast.AST, name: str, depth: int) -> list[Optional[Classes]]:
        """What each binding of *name* in *func* that is not an assignment makes it, by its declarations; ``None`` for one they do not say.

        A loop's target is an element of what it iterates
        (:meth:`_iterated`): ``for file_spec in job.Files``, ``Files:
        list[FileSpec]``.  A name unpacked from a call is its slot of the
        ``tuple[...]`` the callee returns; ``:=`` is an assignment;
        ``except X as e`` is an ``X``.  Read by the declarations, as the
        rest of this class reads -- not by which binding reaches the call.
        A binding they do not type adds nothing and takes nothing away, as
        before these were read: ``for tmpNucleusSpec in nucleusList.items()``
        on an undeclared ``nucleusList`` beside a loop over ``dict[str,
        NucleusSpec]``.  Saying "open" for it lost 65 placements whose
        readings were right (X8: 33 closed, 32 builtin), where the 154 the
        rule left closed held no wrong one (X7).  A ``with`` is
        :meth:`_with_held`'s; a nested scope's binding is its own.

        A parameter is not one of these, nor an import inside the function:
        where the function also assigns the name, the assignments answer
        alone.  ``add_main.main``'s ``taskBuffer = tbuf`` is a ``TaskBuffer``
        by ``tbuf = TaskBuffer()`` further down, while the daemon may pass the
        interface proxy (``daemons/utils.py``) -- which hands the call by its
        name to a ``TaskBuffer`` in another process, so the method, the
        arguments and the constants are the ones placed here and the process
        is not.  18 placements rest on such a name (X8), 6 of them behind
        that proxy.
        """
        found: list[Optional[Classes]] = []
        for one in pathcond.bindings(func, name):
            if one.nested:
                continue
            if one.kind in (pathcond.BIND_ASSIGN, pathcond.BIND_ANNOTATED, pathcond.BIND_AUGMENTED, pathcond.BIND_WITH):
                if not one.unpacked or one.kind == pathcond.BIND_WITH:
                    continue
                found.append(self._unpacked(file, func, one.value, one.slot, depth))
            elif one.kind == pathcond.BIND_WALRUS:
                found.append(self.declared(file, func, one.value, depth + 1) if one.value is not None else None)
            elif one.kind == pathcond.BIND_EXCEPT:
                found.append(self._caught(file, one.site))
            elif one.kind == pathcond.BIND_FOR:
                loop = one.site
                found.append(
                    self._iterated(file, func, loop.iter, one.slot if one.unpacked else None, one.unpacked, depth)  # type: ignore[attr-defined]
                )
            else:
                found.append(None)
        return found

    def _unpacked(
        self, file: str, func: ast.AST, value: Optional[ast.expr], slot: Optional[int], depth: int
    ) -> Optional[Classes]:
        """The classes slot *slot* of *value* is declared as: a tuple literal's element, or a call's ``tuple[...]``."""
        if value is None or slot is None:
            return None
        if isinstance(value, (ast.Tuple, ast.List)):
            return self.declared(file, func, value.elts[slot], depth + 1) if slot < len(value.elts) else None
        returned = self._returned(file, func, value, depth)
        if returned is None:
            return None
        where, annotation = returned
        element = _slot_of(annotation, slot)
        return self.annotated_all(where, element) if element is not None else None

    def _returned(self, file: str, func: ast.AST, value: ast.expr, depth: int) -> Optional[tuple[str, ast.expr]]:
        """``(file, return annotation)`` of the function *value*, a call, lands in, read as :meth:`declared` reads it."""
        if not isinstance(value, ast.Call):
            return None
        called = value.func
        if isinstance(called, ast.Name):
            function = self._function_named(file, called.id)
            definition = function[1] if function is not None else None
            where = function[0] if function is not None else file
        elif isinstance(called, ast.Attribute):
            receiver = called.value
            if isinstance(receiver, ast.Name) and receiver.id == "self":
                owner = self.method_class(file, func)
            else:
                owner = self.of_expression(file, func, receiver, depth + 1)
            found = self.method(owner[0], owner[1], called.attr) if owner is not None else None
            where, definition = found if found is not None else (file, None)
        else:
            return None
        returns = getattr(definition, "returns", None)
        return (where, returns) if returns is not None else None

    def _caught(self, file: str, handler: ast.AST) -> Optional[Classes]:
        """The corpus classes ``except X as e`` binds ``e`` to; ``None`` where one is not the corpus's."""
        caught = getattr(handler, "type", None)
        members = caught.elts if isinstance(caught, ast.Tuple) else [caught] if caught is not None else []
        found = [self.class_named(file, one) for one in members]
        if not found or any(one is None for one in found):
            return None
        return frozenset(found)  # type: ignore[arg-type]

    def _iterated(
        self, file: str, func: ast.AST, iterable: ast.expr, slot: Optional[int], unpacked: bool, depth: int
    ) -> Optional[Classes]:
        """The classes a loop's target is, from what it iterates; slot *slot* of each element where it is unpacked.

        ``list(x)``, ``sorted(x)``, ``reversed(x)`` iterate what ``x`` does;
        ``enumerate(x)``'s second slot is ``x``'s element.  ``d.items()``,
        ``.values()`` and ``.keys()`` on a ``dict[K, V]`` are ``(K, V)``,
        ``V`` and ``K``; a container is its element, a mapping its key.
        """
        if isinstance(iterable, ast.Call) and isinstance(iterable.func, ast.Name) and len(iterable.args) == 1:
            if iterable.func.id in ("list", "sorted", "reversed", "tuple", "set", "frozenset") and not unpacked:
                return self._iterated(file, func, iterable.args[0], None, False, depth)
            if iterable.func.id == "enumerate" and unpacked and slot == 1:
                return self._iterated(file, func, iterable.args[0], None, False, depth)
            return None
        view = ""
        source: ast.expr = iterable
        if (
            isinstance(iterable, ast.Call)
            and isinstance(iterable.func, ast.Attribute)
            and iterable.func.attr in ("items", "values", "keys")
            and not iterable.args
        ):
            view, source = iterable.func.attr, iterable.func.value
        declared = self._annotation_of(file, func, source)
        if declared is None and isinstance(source, ast.Attribute):
            # ``job.Files``, on a ``job`` declared a ``JobSpec``.
            owner = self.of_expression(file, func, source.value, depth + 1)
            declared = self._field_annotation(owner, source.attr) if owner is not None else None
        if declared is None:
            declared = self._returned(file, func, source, depth)
        if declared is None and isinstance(source, ast.Name):
            declared = self._bound_to_calls(file, func, source.id, depth)
        if declared is None:
            return None
        where, annotation = declared
        annotation = _unquoted(annotation)
        if not isinstance(annotation, ast.Subscript):
            return None
        wrapper = annotation.value
        kind = wrapper.attr if isinstance(wrapper, ast.Attribute) else getattr(wrapper, "id", "")
        inner = annotation.slice
        if kind in _MAPPINGS:
            if not (isinstance(inner, ast.Tuple) and len(inner.elts) == 2):
                return None
            key, value = inner.elts
            if view == "items":
                element = (key, value)[slot] if unpacked and slot in (0, 1) else None
            else:
                element = value if view == "values" else key
                element = None if unpacked else element
        elif kind in _COLLECTIONS and not view:
            first = _first_argument(inner)
            element = first if not unpacked else _slot_of(first, slot)
        else:
            element = None
        return self.annotated_all(where, element) if element is not None else None

    def _bound_to_calls(self, file: str, func: ast.AST, name: str, depth: int) -> Optional[tuple[str, ast.expr]]:
        """What the calls a local *name* is bound to are declared to return, where every binding of it is one and they
        say one thing: ``taskList = self.taskBufferIF.getTasksToReassign_JEDI(...)`` then ``for taskSpec in taskList``.

        For a loop over the name (:meth:`_iterated`); a parameter, or a name bound any other way -- unpacked, by a
        loop or a ``with`` -- says nothing here.
        """
        if name in _parameter_names(func):
            return None
        sites = [one for one in pathcond.bindings(func, name) if not one.nested and one.kind != pathcond.BIND_AUGMENTED]
        if not sites or any(
            one.kind != pathcond.BIND_ASSIGN or one.unpacked or not isinstance(one.value, ast.Call) for one in sites
        ):
            return None
        found = [self._returned(file, func, one.value, depth) for one in sites]
        held = [one for one in found if one is not None]
        if len(held) != len(found) or len({ast.unparse(annotation) for _where, annotation in held}) != 1:
            return None
        return held[0]

    def of_expression(self, file: str, func: ast.AST, expression: ast.expr, depth: int = 0) -> Optional[tuple[str, ast.ClassDef]]:
        """The class *expression*, read in *func* of *file*, is declared to be, where it is declared one.

        The walk's typer asks this, and so does the owner of a method whose
        return annotation is read: a call on an object declared as two
        classes returns what either method does, and that is not read.
        """
        return _one(self.declared(file, func, expression, depth))

    def declared(self, file: str, func: ast.AST, expression: ast.expr, depth: int = 0) -> Optional[Classes]:
        """The classes *expression*, read in *func* of *file*, is declared to be.

        One, or each member of a union.  A union is read wherever a single
        class is -- a field's, a parameter's or a local's annotation, a
        module's instance, what a ``with`` yields, a container's element, a
        return annotation -- and where every assignment of a name or field
        must say the same, it must say the same classes.
        """
        if depth > self.DEPTH:
            return None
        tree = self.tree(file)
        if tree is None:
            return None
        if isinstance(expression, ast.Name):
            # What a name is depends on the function, the name and how deep
            # the question already is -- not on which ``Name`` asked.
            # ``JobGenerator`` rebinds locals from calls on locals that are
            # rebound too, and reading each afresh cost the product of their
            # assignment counts: a census stopped in that one file for hours.
            # The function is kept with the answer, so its id is not reused.
            key = (file, id(func), expression.id, depth)
            hit = self._names.get(key)
            if hit is not None and hit[0] is func:
                return hit[1]
            answer = self._of_name(file, func, expression.id, depth)
            self._names[key] = (func, answer)
            return answer
        if (
            isinstance(expression, ast.Attribute)
            and isinstance(expression.value, ast.Name)
            and expression.value.id == "self"
        ):
            cls = pathcond.class_of(tree, func)
            return self.of_field(file, cls, expression.attr, depth + 1) if cls is not None else None
        if isinstance(expression, ast.Subscript):
            # ``self.nuclei[name]``, where ``self.nuclei: dict[str, NucleusSpec]``.
            container = self._annotation_of(file, func, expression.value)
            if container is not None:
                where, annotation = container
                element = _element_annotation(annotation)
                if element is not None:
                    return self.annotated_all(where, element)
            return None
        if isinstance(expression, ast.Call):
            called = expression.func
            constructed = self.class_named(file, called) if isinstance(called, (ast.Name, ast.Attribute)) else None
            if constructed is not None:
                return frozenset({constructed})
            hook = self._copy_hook(file, called)
            if hook is not None:
                # ``jobSpec = copy.copy(job)``: what ``job`` is declared to be.
                copied = self.declared(file, func, expression.args[0], depth + 1) if expression.args else None
                return self._copied(copied, hook)
            if isinstance(called, ast.Name):
                # ``get_task_utils_module(self)``, declared ``-> TaskUtilsModule``.
                function = self._function_named(file, called.id)
                if function is not None:
                    where, definition = function
                    return self.annotated_all(where, definition.returns)
                return None
            if isinstance(called, ast.Attribute):
                receiver = called.value
                if isinstance(receiver, ast.Name) and receiver.id == "self":
                    owner = self.method_class(file, func)
                else:
                    owner = self.of_expression(file, func, receiver, depth + 1)
                if owner is None:
                    return None
                held = self._method_in(owner[0], owner[1], called.attr)
                if held is None:
                    return None
                where, holder, method = held
                if not self.type_parameters(where, holder):
                    return self.annotated_all(where, method.returns)
                bound = self._receiver_binds(file, func, receiver, owner)
                return self._annotated_bound(where, method.returns, holder, bound)
        return None

    # -- what a value can be, where nothing declares it --------------------

    def candidates(
        self, file: str, func: ast.AST, expression: ast.expr, depth: int = 0
    ) -> tuple[frozenset[tuple[str, str]], bool]:
        """``({(file, class name)}, open)``: the classes *expression* can be an instance of.

        Asked where the declaration says nothing -- no annotation, or ``Any``.
        ``AdderGen`` declares ``self.adder_plugin: Any`` and builds it as
        ``self.adder_plugin_class(...)``, where ``adder_plugin_class`` is what
        ``get_plugin_class`` returns: ``AdderAtlasPlugin``, or a class the
        configuration names.  Read from the values rather than the
        declaration, that is ``{AdderAtlasPlugin}`` and *open* -- there is
        another way, and this reading cannot say which class it is.
        """
        declared = self.declared(file, func, expression, 0)
        if declared is not None:
            return frozenset((where, cls.name) for where, cls in declared), False
        return self._values(file, func, expression, depth, instances=True, seen=frozenset())

    def class_def(self, file: str, name: str) -> Optional[ast.ClassDef]:
        tree = self.tree(file)
        for node in walk(tree) if tree is not None else ():
            if isinstance(node, ast.ClassDef) and node.name == name:
                return node
        return None

    def _values(
        self,
        file: str,
        func: ast.AST,
        expression: ast.expr,
        depth: int,
        *,
        instances: bool,
        seen: frozenset,
    ) -> tuple[frozenset[tuple[str, str]], bool]:
        """Classes an expression holds -- instances of, or the class objects themselves."""
        if depth > self.DEPTH:
            return frozenset(), True
        key = (file, id(func), ast.unparse(expression), instances)
        if key in seen:
            return frozenset(), False
        seen = seen | {key}
        found: set[tuple[str, str]] = set()
        opened = False

        def add(result: tuple[frozenset, bool]) -> None:
            nonlocal opened
            found.update(result[0])
            opened = opened or result[1]

        if isinstance(expression, ast.Constant) and expression.value is None:
            return frozenset(), False
        if not instances and isinstance(expression, ast.Subscript):
            recorded = self._recorded_at(file, func, expression)
            if recorded is not None:
                return recorded
        if instances and (builtin := self._builtin_instance(file, func, expression)):
            return frozenset({(BUILTIN, builtin)}), False
        if isinstance(expression, ast.Call):
            hook = self._copy_hook(file, expression.func) if instances else None
            if hook is not None:
                # A copy is an instance of what it copies; a class that copies
                # itself with the hook is what the hook returns, unread here.
                if not expression.args:
                    return frozenset(), True
                copied, opened = self._values(file, func, expression.args[0], depth + 1, instances=True, seen=seen)
                for where, name in copied:
                    cls = self.class_def(where, name) if where != BUILTIN else None
                    if cls is not None and self.method(where, cls, hook) is not None:
                        opened = True
                    else:
                        found.add((where, name))
                return frozenset(found), opened
            if instances:
                # An instance of whatever class the callee is: ``Cls(...)``,
                # or ``self.plugin_class(...)`` holding a class.
                add(self._values(file, func, expression.func, depth + 1, instances=False, seen=seen))
                return frozenset(found), opened
            # A call that returns a class: what its body returns.
            owner = self._owner_of(file, func, expression.func)
            if owner is None:
                return frozenset(), True
            where, method = owner
            for node in walk(method):
                if isinstance(node, ast.Return) and node.value is not None:
                    add(self._values(where, method, node.value, depth + 1, instances=False, seen=seen))
            return frozenset(found), opened
        if isinstance(expression, ast.Name):
            if not instances:
                named = self.class_named(file, expression)
                if named is not None:
                    return frozenset({(named[0], named[1].name)}), False
            bound = self._assigned_here(func, expression.id)
            # A loop's target, a name unpacked or caught: the classes their
            # declarations say, as :meth:`declared` reads them; one they do
            # not type adds nothing (:meth:`_other_bindings`).
            others = self._other_bindings(file, func, expression.id, depth)
            if not bound and not others:
                return frozenset(), True
            for value in bound:
                # What a binding is declared as, where it is: ``site_spec_tmp
                # = self.siteSpecList[name]`` is a ``SiteSpec`` beside a loop
                # over a ``dict[Any, Any]`` -- read as :meth:`_of_name` reads
                # each binding, and from what it is built where nothing
                # declares it.
                typed = self.declared(file, func, value, depth + 1) if instances else None
                if typed is not None:
                    found.update((where, cls.name) for where, cls in typed)
                    continue
                add(self._values(file, func, value, depth + 1, instances=instances, seen=seen))
            readable = [one for one in others if one is not None and instances]
            if not bound and not readable:
                return frozenset(), True
            for one in readable:
                found.update((where, cls.name) for where, cls in one)
            return frozenset(found), opened
        if (
            isinstance(expression, ast.Attribute)
            and isinstance(expression.value, ast.Name)
            and expression.value.id == "self"
        ):
            holder = self.method_class(file, func)
            if holder is None:
                return frozenset(), True
            any_value = False
            for where, cls in self.lineage(*holder):
                for method in pathcond.methods_of(cls):
                    for value in pathcond.attribute_expressions(method, expression.attr):
                        if isinstance(value, ast.Constant) and value.value is None:
                            continue
                        any_value = True
                        add(self._values(where, method, value, depth + 1, instances=instances, seen=seen))
            return frozenset(found), opened or not any_value
        if isinstance(expression, ast.Attribute):
            # ``obj.field`` where ``obj`` is one of some classes: what each assigns it.
            owners, owner_open = self.candidates(file, func, expression.value, depth + 1)
            opened = owner_open
            for where, name in owners:
                if where == BUILTIN:
                    # ``self.task_setupper_map = {}`` ... ``.get(key)``: what
                    # a builtin's method returns, or its field holds, is not
                    # read here.
                    opened = True
                    continue
                cls = self.class_def(where, name)
                if cls is None:
                    continue
                typed = self.of_field(where, cls, expression.attr, depth + 1)
                if typed is not None and instances:
                    found.update((one, held.name) for one, held in typed)
                    continue
                for place, holder in self.lineage(where, cls):
                    for method in pathcond.methods_of(holder):
                        for value in pathcond.attribute_expressions(method, expression.attr):
                            if isinstance(value, ast.Constant) and value.value is None:
                                continue
                            add(self._values(place, method, value, depth + 1, instances=instances, seen=seen))
            return frozenset(found), opened
        return frozenset(), True

    def _builtin_instance(self, file: str, func: ast.AST, expression: ast.expr) -> str:
        """The builtin type *expression* builds an instance of -- ``"list"`` for ``[]`` or ``sorted(x)`` -- or ``""``.

        A literal, a comprehension, a string constant; a call of a builtin
        type's name the module and the function leave unbound; a container
        class of the standard library, imported.  ``tmp_list.append(x)`` on
        such a value is a builtin's method, not a call to a corpus function
        named ``append`` -- ``ListWithLock.append`` had a thousand calls by
        its name that could not be told from it.
        """
        literal = (
            ((ast.List, ast.ListComp), "list"),
            ((ast.Dict, ast.DictComp), "dict"),
            ((ast.Set, ast.SetComp), "set"),
            ((ast.Tuple,), "tuple"),
            ((ast.JoinedStr,), "str"),
        )
        for shapes, kind in literal:
            if isinstance(expression, shapes):
                return kind
        if isinstance(expression, ast.Constant):
            return type(expression.value).__name__
        if not isinstance(expression, ast.Call):
            return ""
        called = expression.func
        if isinstance(called, ast.Name):
            held = self.imports(file).get(called.id)
            if held is not None:
                return held[1] if held in _LIBRARY_CONTAINERS else ""
            function = callfacts.builtin_function(self, file, func, expression)
            return (function.result or "") if function is not None else ""
        if isinstance(called, ast.Attribute) and isinstance(called.value, ast.Name):
            held = self.imports(file).get(called.value.id)
            if held is not None and not held[1] and (held[0], called.attr) in _LIBRARY_CONTAINERS:
                return called.attr
        return ""

    def _copy_hook(self, file: str, called: ast.expr) -> Optional[str]:
        """The hook a class would copy itself with, where *called* is one of :data:`_COPIES`."""
        imported = self.imports(file)
        if isinstance(called, ast.Name):
            held = imported.get(called.id)
            return _COPIES.get(held) if held is not None else None
        if isinstance(called, ast.Attribute) and isinstance(called.value, ast.Name):
            held = imported.get(called.value.id)
            if held is not None and not held[1]:
                return _COPIES.get((held[0], called.attr))
        return None

    def _copied(self, copied: Optional[Classes], hook: str) -> Optional[Classes]:
        """What a copy of an object of the classes *copied* is declared to be.

        The same classes; where one defines *hook*, what the hook is declared
        to return, and nothing where it declares nothing.
        """
        if copied is None:
            return None
        found: set[tuple[str, ast.ClassDef]] = set()
        for where, cls in copied:
            hooked = self.method(where, cls, hook)
            if hooked is None:
                found.add((where, cls))
                continue
            returned = self.annotated_all(hooked[0], hooked[1].returns)
            if returned is None:
                return None
            found |= returned
        return frozenset(found)

    def _owner_of(self, file: str, func: ast.AST, called: ast.expr) -> Optional[tuple[str, ast.AST]]:
        """The function *called* names: a method on ``self``, a module's function by a plain name, or ``module.f``.

        ``panda_config.getPlugin(...)`` returns the class ``setupper_plugin_class`` builds; read as nothing, the call
        left the holder open before getPlugin's body was read.
        """
        if (
            isinstance(called, ast.Attribute)
            and isinstance(called.value, ast.Name)
            and called.value.id == "self"
        ):
            holder = self.method_class(file, func)
            if holder is None:
                return None
            return self.method(holder[0], holder[1], called.attr)
        if isinstance(called, ast.Name):
            return None if self._local(func, called.id) else self._function_named(file, called.id)
        if isinstance(called, ast.Attribute) and isinstance(called.value, ast.Name):
            module = None if self._local(func, called.value.id) else self.module_named(file, called.value)
            return self._function_named(module, called.attr) if module is not None else None
        return None

    def _recorded_at(
        self, file: str, func: ast.AST, expression: ast.Subscript
    ) -> Optional[tuple[frozenset[tuple[str, str]], bool]]:
        """The classes at ``m[k1]...[kn]``, where *m* is a name of *file*'s module the instance's record gives.

        Every value the record holds that many keys down, at any key: the keys are read at run time.  A class of the
        corpus is a class found; one outside it, a value not a class, or a key into a value not a mapping leaves the
        reading open.  ``None`` where the record says nothing of the name, or *func* binds it itself -- read as the
        code says it.  ``getPlugin``'s ``g_pluginMap[modConfigName][vo]`` holds nothing on ATLAS's production
        instance, so the call returns ``None`` alone.
        """
        root: ast.expr = expression
        depth = 0
        while isinstance(root, ast.Subscript):
            root, depth = root.value, depth + 1
        if not isinstance(root, ast.Name) or (file, root.id) not in self.recorded or self._local(func, root.id):
            return None
        values = [self.recorded[(file, root.id)]]
        for _ in range(depth):
            if not all(isinstance(one, dict) for one in values):
                return frozenset(), True
            values = [one for held in values for one in held.values()]  # type: ignore[attr-defined]
        found: set[tuple[str, str]] = set()
        opened = False
        for one in values:
            placed = self._recorded_class(one) if isinstance(one, RecordedClass) else None
            if placed is None:
                opened = opened or one is not None
            else:
                found.add(placed)
        return frozenset(found), opened

    def _recorded_class(self, one: RecordedClass) -> Optional[tuple[str, str]]:
        """``(file, name)`` of the corpus class a record names by its dotted path, or ``None``."""
        module, _, name = one.dotted.rpartition(".")
        where = self.module_file(module) if module else None
        return (where, name) if where is not None and self.class_def(where, name) is not None else None

    def _local(self, func: ast.AST, name: str) -> bool:
        """Whether *func* binds *name* itself -- assigned there or a parameter --, so it names no module's."""
        arguments = getattr(func, "args", None)
        every = (
            [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs, arguments.vararg, arguments.kwarg]
            if arguments is not None
            else []
        )
        return name in self.assigned(func) or any(one is not None and one.arg == name for one in every)

