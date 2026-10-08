"""Which log file a module's diagnostics land in.

The map's purpose is to say which component's log to read and what to look
for, so the file is part of the answer rather than a detail of fetching it.
It is also what makes the production check practical: PanDA writes one file
per logger, not one per service, so a question about brokerage rejections
goes to ``panda-AtlasProdJobBroker.log`` and comes back narrow instead of
arriving as a truncated slice of everything the service logged.

``PandaLogger.getLogger(name)`` opens ``<logdir>/panda-<name>.log``, and the
corpus names loggers two ways::

    logger = PandaLogger().getLogger(__name__.split(".")[-1])   # the module
    _logger = PandaLogger().getLogger("api_async_process")      # a literal

The split runs along package lines -- JEDI takes the module's own name, the
server mostly spells one out -- but both forms appear in both, so the shape is
what is read, not the package.  pandacommon's ``logger_utils.setup_logger(name)``
is the same call under another name -- it returns
``PandaLogger().getLogger(name)`` -- and the message processors and the daemons
open their loggers with it.  ``logging.getLogger`` is not read.  It is the
standard library's, and its lines go wherever its handlers say: the daemon
master's go to stdout, and ``panda-PanDA-Daemon-Master.log``, which this used
to name for it, is never written.

**A module can have more than one answer, and that is the interesting case.**
The ``db_proxy_mods`` modules -- 175 of the map's junctions, by far the
largest group -- declare no logger.  They are mixins: ``OraDBProxy.DBProxy``
inherits all thirteen of them and ``JediDBProxy.DBProxy`` inherits that, and
each of those two *does* declare one.  So the same junction writes to
``panda-DBProxy.log`` when the server calls it and ``panda-JediDBProxy.log``
when a knight does.

That is not an ambiguity to be resolved.  It is the JEDI/server crossing
stated exactly: the file depends on which process ran the code, which is a
runtime fact, and naming both candidates is the true answer.  Hence a list
rather than a single name.

**A module with neither is left empty.**  ``JobBrokerBase`` logs through a
``MsgWrapper`` built by whoever instantiated it and is mixed into nothing, so
there is no evidence to read.  A wrong filename would be worse than none: the
query comes back empty, and empty reads as "production never emitted this".

**A call is read by its own logger.**  The module's file answers where a
module's code lives.  Where one logging call's lines land is the file of the
logger that call is made on, and :class:`LoggerFiles` follows that logger to
where it was opened.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from typing import NamedTuple, Optional

from bamboo.codemap.calls import Calls, CallSite, Target, bind
from bamboo.codemap.models import (
    LOG_FILES_FOLLOWED,
    LOG_FILES_OF_MODULE,
    LOG_FILES_PARTLY,
    LogSiteNode,
    SourceModule,
)
from bamboo.codemap.panda import pathcond
from bamboo.codemap.panda.recognizers import trigger

# PandaLogger.getLogger(name) opens "<logdir>/panda-<name>.log".
_FILENAME = "panda-{}.log"

#: The calls that open a ``PandaLogger`` file by name.
_OPENERS = frozenset({"getLogger", "setup_logger"})
#: The calls that wrap a logger and write through it, the logger first:
#: ``LogWrapper(logger, prefix)``, JEDI's ``MsgWrapper(logger, ...)`` and
#: pandacommon's ``make_logger(logger, ...)``, which builds a ``LogWrapper``.
_WRAPPERS = frozenset({"LogWrapper", "MsgWrapper", "make_logger"})
#: How many callers deep a logger handed in as a parameter is followed.  The
#: proxy mixins take theirs two constructors up: ``BaseModule.__init__`` from
#: each mixin's ``__init__``, and that from ``OraDBProxy.DBProxy``'s.
_HOPS = 3


def _module_basename(rel_path: str) -> str:
    """``pandajedi/jediorder/JobGenerator.py`` -> ``JobGenerator``."""
    return rel_path.rsplit("/", 1)[-1][: -len(".py")]


def _names_itself(node: ast.expr) -> bool:
    """Whether the argument is the ``__name__.split(".")[-1]`` idiom.

    Matched structurally rather than by unparsing, so that spacing and the
    quote style cannot change the answer.
    """
    if not isinstance(node, ast.Subscript):
        return False
    call = node.value
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
        return False
    if call.func.attr != "split":
        return False
    receiver = call.func.value
    return isinstance(receiver, ast.Name) and receiver.id == "__name__"


def _opened_name(call: ast.Call, rel_path: str) -> Optional[str]:
    """The ``PandaLogger`` name *call* opens in module *rel_path*, or None.

    None for a call that opens none, for the standard library's
    ``logging.getLogger``, and for a name spelled any way but the two above.
    ``setup_logger()`` with no name takes its caller's module's, as
    pandacommon reads it off the stack.
    """
    called = call.func
    name = called.attr if isinstance(called, ast.Attribute) else getattr(called, "id", "")
    if name not in _OPENERS:
        return None
    if isinstance(called, ast.Attribute) and isinstance(called.value, ast.Name) and called.value.id == "logging":
        return None
    if not call.args:
        return _module_basename(rel_path) if name == "setup_logger" else None
    argument = call.args[0]
    if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
        return argument.value
    if _names_itself(argument):
        return _module_basename(rel_path)
    return None


def logger_name(module: SourceModule) -> Optional[str]:
    """The logger this module declares, or None when it declares none.

    The first declaration wins, and it is the module's file.  Three modules
    declare two: ``OraDBProxy`` and ``JediDBProxy`` each a ``...Filtered``
    beside their own, and ``api/v1/pilot_api`` ``PilotRequests`` beside
    ``api_pilot``.  The first is the one the module's own calls use; a call
    made on the other is read where it is made, by :class:`LoggerFiles`.
    """
    for node in ast.walk(module.tree):
        if isinstance(node, ast.Call):
            name = _opened_name(node, module.rel_path)
            if name:
                return name
    return None


def declared_files(modules: list[SourceModule]) -> dict[str, str]:
    """Return ``{rel_path: log filename}`` for every module that declares one."""
    resolved: dict[str, str] = {}
    for module in modules:
        name = logger_name(module)
        if name:
            resolved[module.rel_path] = _FILENAME.format(name)
    return resolved


def _subclass_edges(modules: list[SourceModule]) -> dict[str, list[tuple[str, str]]]:
    """Return ``{base class name: [(subclass, its module), ...]}``.

    The names of :func:`trigger.subclass_edges`, which does the walking.  Two
    slices read inheritance -- this one to follow a mixin's output to the log
    of whatever inherits it, the trigger slice to find the workers a knight
    dispatches -- and the tree is walked once so the two cannot drift.
    """
    return {
        base: [(node.name, where) for node, where in places]
        for base, places in trigger.subclass_edges(modules).items()
    }


def _classes_in(module: SourceModule) -> list[str]:
    return [n.name for n in ast.walk(module.tree) if isinstance(n, ast.ClassDef)]


def inherited_files(
    modules: list[SourceModule], declared: dict[str, str]
) -> dict[str, list[str]]:
    """Files a mixin's output reaches, through the classes that inherit it.

    Walked transitively, because the chain has two links: the proxy modules
    are mixed into ``OraDBProxy.DBProxy``, which is itself subclassed by
    ``JediDBProxy.DBProxy``.  Stopping at one would name the server's log and
    silently omit JEDI's -- the half most junctions actually run under.
    """
    edges = _subclass_edges(modules)
    reached: dict[str, list[str]] = {}
    for module in modules:
        if module.rel_path in declared:
            continue
        files: set[str] = set()
        seen: set[tuple[str, str]] = set()
        frontier = [(name, module.rel_path) for name in _classes_in(module)]
        while frontier:
            class_name, where = frontier.pop()
            if (class_name, where) in seen:
                continue
            seen.add((class_name, where))
            if where != module.rel_path and where in declared:
                files.add(declared[where])
            frontier.extend(edges.get(class_name, ()))
        if files:
            reached[module.rel_path] = sorted(files)
    return reached


def files_of(
    rel_path: str, declared: dict[str, str], inherited: dict[str, list[str]]
) -> list[str]:
    """The files one module's output can reach, declaration first."""
    if rel_path in declared:
        return [declared[rel_path]]
    return inherited.get(rel_path, [])


class Landing(NamedTuple):
    """Where one logging call's lines land, and how that was found (``LOG_FILES_*``)."""

    files: list[str]
    basis: str


class _Return(NamedTuple):
    """A call being followed into the function it lands in, for that function's parameters."""

    func: ast.AST
    bound: dict[str, Optional[ast.expr]]
    file: str
    caller: ast.AST
    at: ast.AST
    way: "_Way"


class _Way(NamedTuple):
    """What one path of the following carries.

    *instance* is the class of the object whose ``self`` is being read, so
    that a constructor chain is followed through that object's own classes
    only.  *returning* is set while a call's returned value is followed, so
    that the callee's parameters are the call's arguments rather than every
    caller's.
    """

    seen: frozenset
    hops: int = 0
    instance: Optional[tuple[str, ast.ClassDef]] = None
    returning: Optional[_Return] = None


class _Trail:
    """The files the paths reached, whether any path reached a logger, and whether any stopped short of one.

    *standard* is a path that reached the standard library's logger, which
    prints wherever its handlers say and into none of PandaLogger's files.
    """

    def __init__(self) -> None:
        self.files: set[str] = set()
        self.reached = False
        self.standard = False
        self.open = False


class LoggerFiles:
    """Where each logging call's lines land: the files of the logger it is made on.

    The module's own file is where most calls land, and not where all of them
    do.  ``WatchDog`` opens ``tmpLog`` on its own logger and calls
    ``impl.pre_action(tmpLog, ...)``: every line ``pre_action`` prints is in
    ``panda-WatchDog.log``, and none in the files its module's subclasses
    declare.  So the logger is followed from the call back to where it was
    opened:

    * ``getLogger(name)`` and ``setup_logger(name)`` open the file the name
      gives (:func:`_opened_name`);
    * ``LogWrapper(logger, ...)`` and the other wrappers write through their
      first argument;
    * a call the call graph places at one function returns what that
      function returns, its parameters bound to the call's arguments --
      ``self.create_tagged_logger(comment)`` is how the proxy mixins get theirs;
    * a local is every binding of it that can reach the call;
    * a parameter no binding overwrites first is what each caller passes;
    * ``self.<field>`` is every value the object's classes assign to it; a
      constructor's parameter there is what the object's own constructors
      and ``super().__init__`` calls pass;
    * a module's name is its binding there, and the binding another module
      makes: ``JediDBProxy`` runs ``OraDBProxy._logger = logger``, so the
      proxy mixins print into ``panda-JediDBProxy.log`` in a JEDI process;
    * an imported name, or ``module.name``, is that module's binding.

    A path that ends anywhere else -- a closure's variable, a name opened by a
    spelling :func:`_opened_name` does not read, a parameter with no caller
    -- leaves the trail open.

    A caller is a call the call graph places at the function.  Where it
    places none, every call by the function's name that it places nowhere is
    taken instead: ``impl.pre_action`` is called on a plugin chosen at run
    time, and is placed nowhere.  Taking every such call can name a caller
    that never reaches this function, so the files can be too many.  They
    cannot be too few, though, so a line absent from all of them was still
    not printed.

    The object matters for a constructor.  The JEDI knights' threads take
    ``self.logger`` from a base class's ``__init__``, and every knight calls
    that ``__init__`` through ``super()``.  Taking all of those calls gave
    each knight's ``runImpl`` all eight knights' files.  Only the calls made
    by the reading class's own lineage, and by its subclasses, are taken.
    """

    def __init__(self, calls: Calls, declared: dict[str, str], inherited: dict[str, list[str]]) -> None:
        self.calls = calls
        self.types = calls.types
        self.declared = declared
        self.inherited = inherited
        self._rebound = self._rebindings()
        self._callers: dict[tuple, list[tuple[CallSite, bool]]] = {}

    def of_call(self, file: str, func: ast.AST, call: ast.Call) -> Landing:
        """Where the lines of logging *call*, made in *func* of *file*, land."""
        trail = _Trail()
        receiver = call.func.value if isinstance(call.func, ast.Attribute) else call.func
        self._value(trail, file, func, receiver, pathcond.site_of(call), _Way(frozenset()))
        own = files_of(file, self.declared, self.inherited)
        if trail.standard and trail.files:
            # Some paths print into a file and some do not: the snakemake
            # parser uses the logger it is handed, and its own standard
            # library one when it is handed none.  A line absent from the
            # file may have gone the other way, so absence proves nothing.
            return Landing(sorted(trail.files | set(own)), LOG_FILES_PARTLY)
        if trail.reached and not trail.open:
            # Possibly no file at all: the daemon master's logger is the
            # standard library's, writing to stdout, and the module's file
            # standing in for it named one these lines are never in.
            return Landing(sorted(trail.files), LOG_FILES_FOLLOWED)
        if trail.files:
            return Landing(sorted(trail.files | set(own)), LOG_FILES_PARTLY)
        return Landing(list(own), LOG_FILES_OF_MODULE)

    # -- following ----------------------------------------------------------

    def _value(self, trail: _Trail, file: str, func: Optional[ast.AST], expr: ast.expr, at: Optional[ast.AST], way: _Way) -> None:
        if isinstance(expr, ast.Constant) and expr.value is None:
            # No logger on this path, so nothing is printed on it.
            return
        if isinstance(expr, ast.Call):
            self._call(trail, file, func, expr, at, way)
        elif isinstance(expr, ast.Name):
            self._name(trail, file, func, expr.id, at, way)
        elif isinstance(expr, ast.Attribute) and isinstance(expr.value, ast.Name):
            if expr.value.id in ("self", "cls") and func is not None:
                self._field(trail, file, func, expr.attr, way)
                return
            module = self.types.module_named(file, expr.value)
            if module is None:
                trail.open = True
                return
            self._module_name(trail, module, expr.attr, way)
        elif isinstance(expr, ast.IfExp):
            self._value(trail, file, func, expr.body, at, way)
            self._value(trail, file, func, expr.orelse, at, way)
        elif isinstance(expr, ast.BoolOp):
            for one in expr.values:
                self._value(trail, file, func, one, at, way)
        else:
            trail.open = True

    def _call(self, trail: _Trail, file: str, func: Optional[ast.AST], call: ast.Call, at: Optional[ast.AST], way: _Way) -> None:
        opened = _opened_name(call, file)
        if opened:
            trail.files.add(_FILENAME.format(opened))
            trail.reached = True
            return
        called = call.func
        name = called.attr if isinstance(called, ast.Attribute) else getattr(called, "id", "")
        if name in _OPENERS and isinstance(called, ast.Attribute) and getattr(called.value, "id", "") == "logging":
            # A logger, and none of PandaLogger's files.
            trail.reached = True
            trail.standard = True
            return
        if name in _WRAPPERS and call.args:
            self._value(trail, file, func, call.args[0], at, way)
            return
        placed = self.calls.callees(file, func, call) if func is not None and name not in _OPENERS else None
        # One definition, placed closed: a call placed open may return
        # another function's file.
        if placed is None or len(placed.targets) != 1 or placed.unplaced:
            trail.open = True
            return
        (target,) = placed.targets
        key = ("returns", target.file, target.func.lineno)  # type: ignore[attr-defined]
        if key in way.seen:
            return
        returned = [
            node.value
            for node in ast.walk(target.func)
            if isinstance(node, ast.Return)
            and node.value is not None
            and pathcond.enclosing_function(node) is target.func
        ]
        if not returned:
            trail.open = True
            return
        on_self = isinstance(called, ast.Attribute) and isinstance(called.value, ast.Name) and called.value.id in ("self", "cls")
        inner = _Way(
            way.seen | {key},
            way.hops,
            (way.instance or self.types.method_class(file, func)) if on_self else None,
            _Return(target.func, bind(call, target, placed.on_class), file, func, at, way),
        )
        for value in returned:
            self._value(trail, target.file, target.func, value, value, inner)

    def _name(self, trail: _Trail, file: str, func: Optional[ast.AST], name: str, at: Optional[ast.AST], way: _Way) -> None:
        if func is None:
            self._module_name(trail, file, name, way)
            return
        key = ("name", file, id(func), name)
        if key in way.seen:
            return
        way = way._replace(seen=way.seen | {key})
        values = [
            value
            for value in pathcond.assigned_expressions(func).get(name, [])  # type: ignore[arg-type]
            if at is None or pathcond.can_reach(value, at, current=True)
        ]
        for value in values:
            self._value(trail, file, func, value, value, way)
        if _is_parameter(func, name):
            if at is None or not any(_runs_first(pathcond.site_of(value), at) for value in values):
                self._parameter(trail, file, func, name, way)
            return
        if not values:
            self._module_name(trail, file, name, way)

    def _module_name(self, trail: _Trail, file: str, name: str, way: _Way) -> None:
        key = ("module", file, name)
        if key in way.seen:
            return
        way = _Way(way.seen | {key}, way.hops)
        tree = self.types.tree(file)
        values = _module_values(tree, name) if tree is not None else []
        rebound = self._rebound.get((file, name), [])
        for value in values:
            self._value(trail, file, None, value, None, way)
        for where, value in rebound:
            self._value(trail, where, None, value, None, way)
        if values or rebound:
            return
        held = self.types.imports(file).get(name)
        other = self.types.module_file(held[0]) if held and held[1] else None
        if other is None:
            trail.open = True
            return
        self._module_name(trail, other, held[1], way)  # type: ignore[index]

    def _parameter(self, trail: _Trail, file: str, func: ast.AST, name: str, way: _Way) -> None:
        returning = way.returning
        if returning is not None and returning.func is func:
            argument = returning.bound.get(name)
            if argument is None:
                trail.open = True
                return
            self._value(
                trail, returning.file, returning.caller, argument, returning.at,
                returning.way._replace(seen=way.seen),
            )
            return
        if way.hops >= _HOPS:
            trail.open = True
            return
        held = self.types.method_class(file, func)
        target = Target(file, func, held[1] if held else None)
        sites = self._callers_of(target, way.instance)
        if not sites:
            trail.open = True
            return
        for site, same_object in sites:
            argument = self.calls.handed(site, target).get(name)
            if argument is None:
                trail.open = True
                continue
            self._value(
                trail, site.file, site.func, argument, site.call,
                _Way(way.seen, way.hops + 1, way.instance if same_object else None),
            )

    def _field(self, trail: _Trail, file: str, func: ast.AST, field: str, way: _Way) -> None:
        held = self.types.method_class(file, func)
        if held is None:
            trail.open = True
            return
        current = way.instance
        if current is not None and self._related(held, current) and any(
            cls is held[1] for _where, cls in self.types.lineage(*current)
        ):
            instance = current
        else:
            instance = held
        key = ("field", instance[0], instance[1].name, field)
        if key in way.seen:
            return
        inner = _Way(way.seen | {key}, way.hops, instance)
        found = False
        for where, holder in self.types.lineage(*instance):
            for method in pathcond.methods_of(holder):
                for value in pathcond.attribute_expressions(method, field):
                    found = True
                    self._value(trail, where, method, value, value, inner)
        if not found:
            trail.open = True

    # -- callers ------------------------------------------------------------

    def _callers_of(
        self, target: Target, instance: Optional[tuple[str, ast.ClassDef]]
    ) -> list[tuple[CallSite, bool]]:
        """``(site, same object)`` for each call that hands *target* its arguments.

        A constructor's include the chain from a subclass's --
        ``super().__init__(...)``, ``Base.__init__(self, ...)`` -- made on
        the object being built, which is how the proxy mixins' logger
        arrives, and the links of that chain the call graph does not place
        (:meth:`_along_the_mro`).  With *instance* known, only the calls on a
        related class.
        """
        key = (target.file, target.func.lineno, id(instance[1]) if instance else None)  # type: ignore[attr-defined]
        if key in self._callers:
            return self._callers[key]
        constructor = getattr(target.func, "name", "") == "__init__" and target.cls is not None
        placed = self.calls.callers(target)
        found: list[tuple[CallSite, bool]] = []
        for site in placed:
            chained = constructor and isinstance(site.call.func, ast.Attribute) and site.call.func.attr == "__init__"
            if constructor and instance is not None:
                held = self.types.method_class(site.file, site.func) if chained else None
                if chained and (held is None or not self._related(held, instance)):
                    continue
                if not chained and not self._builds_related(site, instance):
                    continue
            found.append((site, chained))
        if constructor:
            found += self._along_the_mro(target, instance, {id(site.call) for site in placed})
        elif not found:
            found = self._unplaced(target)
        self._callers[key] = found
        return found

    def _along_the_mro(
        self, target: Target, instance: Optional[tuple[str, ast.ClassDef]], placed: set[int]
    ) -> list[tuple[CallSite, bool]]:
        """``super().__init__(...)`` in a subclass of *target*'s class that the call graph places at another base.

        A class with several bases runs each base's constructor in turn, along
        its MRO: ``OraDBProxy.DBProxy``'s ``super().__init__(_logger)``
        reaches all thirteen mixins, each handing the logger on.  The call
        graph places ``super()`` at the first base that has the method and no
        further, so the other mixins' constructors have no caller there.  The
        chain is read as the MRO runs it, the subclass's call standing for
        each link.  ``Base.__init__(self, ...)`` names its base, and the call
        graph places it there.
        """
        found: list[tuple[CallSite, bool]] = []
        for site in self.calls.every_call().get("__init__", []):
            call = site.call
            if id(call) in placed or not isinstance(call.func, ast.Attribute):
                continue
            receiver = call.func.value
            if not (isinstance(receiver, ast.Call) and getattr(receiver.func, "id", "") == "super"):
                continue
            held = self.types.method_class(site.file, site.func)
            if held is None or not any(cls is target.cls for _where, cls in self.types.lineage(*held)[1:]):
                continue
            if instance is not None and not self._related(held, instance):
                continue
            found.append((site, True))
        return found

    def _unplaced(self, target: Target) -> list[tuple[CallSite, bool]]:
        """Calls by *target*'s name, on another object, that the call graph places nowhere, or elsewhere and open.

        As :meth:`Calls.missed` counts them: a call placed at other definitions
        and open beside them may land here too.
        """
        found: list[tuple[CallSite, bool]] = []
        for site in self.calls.every_call().get(getattr(target.func, "name", ""), []):
            called = site.call.func
            if not isinstance(called, ast.Attribute):
                continue
            if isinstance(called.value, ast.Name) and called.value.id in ("self", "cls"):
                continue
            landed = self.calls.callees(site.file, site.func, site.call)
            if landed.targets and (not landed.unplaced or any(one.func is target.func for one in landed.targets)):
                continue
            found.append((site, False))
        return found

    def _builds_related(self, site: CallSite, instance: tuple[str, ast.ClassDef]) -> bool:
        built = self.types.class_named(site.file, site.call.func)
        return built is None or self._related(built, instance)

    def _related(self, one: tuple[str, ast.ClassDef], other: tuple[str, ast.ClassDef]) -> bool:
        """Whether one class is the other, a base of it, or a subclass of it."""
        return any(cls is other[1] for _where, cls in self.types.lineage(*one)) or any(
            cls is one[1] for _where, cls in self.types.lineage(*other)
        )

    def _rebindings(self) -> dict[tuple[str, str], list[tuple[str, ast.expr]]]:
        """``{(module, name): [(module that rebinds it, value)]}`` for ``module.name = value`` at module scope."""
        found: dict[tuple[str, str], list[tuple[str, ast.expr]]] = defaultdict(list)
        for file in self.types.corpus_files():
            tree = self.types.tree(file)
            if tree is None:
                continue
            for statement in _module_statements(tree):
                if not isinstance(statement, ast.Assign):
                    continue
                for held in statement.targets:
                    if not (isinstance(held, ast.Attribute) and isinstance(held.value, ast.Name)):
                        continue
                    module = self.types.module_named(file, held.value)
                    if module is not None:
                        found[(module, held.attr)].append((file, statement.value))
        return dict(found)


def _module_statements(tree: ast.Module):
    """The module's own statements, into ``if`` and ``try`` blocks but not into a ``def`` or ``class``."""
    pending = list(tree.body)
    while pending:
        statement = pending.pop(0)
        yield statement
        if isinstance(statement, (ast.If, ast.Try, ast.With)):
            for field in ("body", "orelse", "finalbody"):
                pending.extend(getattr(statement, field, []))
            for handler in getattr(statement, "handlers", []):
                pending.extend(handler.body)


def _module_values(tree: ast.Module, name: str) -> list[ast.expr]:
    """What the module binds *name* to at its own scope."""
    found: list[ast.expr] = []
    for statement in _module_statements(tree):
        if isinstance(statement, ast.Assign):
            if any(isinstance(held, ast.Name) and held.id == name for held in statement.targets):
                found.append(statement.value)
        elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
            if isinstance(statement.target, ast.Name) and statement.target.id == name:
                found.append(statement.value)
    return found


def _is_parameter(func: ast.AST, name: str) -> bool:
    args = getattr(func, "args", None)
    if args is None:
        return False
    every = [*args.posonlyargs, *args.args, *args.kwonlyargs]
    return any(argument.arg == name for argument in every)


def _runs_first(site: ast.AST, target: ast.AST) -> bool:
    """Whether statement *site* runs on every path to *target*: above it in a block that holds it."""
    below = target
    for ancestor in pathcond.ancestors(target):
        for field in ("body", "orelse", "finalbody"):
            block = getattr(ancestor, field, None)
            if isinstance(block, list) and any(one is below for one in block):
                if any(one is site for one in block):
                    return block.index(site) < block.index(below)
                break
        if isinstance(ancestor, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            return False
        below = ancestor
    return False


def _caller_files(
    owner: str,
    mine: list[str],
    declared: dict[str, str],
    inherited: dict[str, list[str]],
    inward,
) -> list[str]:
    """Files belonging to the modules that reach *owner*, minus its own.

    One reading for the two nodes that carry it.  The enclosing class is part
    of ``owner`` but not of the call graph's keys, which are bare method names
    -- the same last-segment rule the attribution slice uses to name a
    function.
    """
    where, _, method = owner.partition("::")
    reached = inward.get(where, {}).get(method.split(".")[-1], ())
    return sorted(
        {file for entry, _door, _call in reached for file in files_of(entry, declared, inherited)}
        - set(mine)
    )


def attach(
    fragment,
    modules: list[SourceModule],
    cadence: Optional[dict[str, set[str]]] = None,
) -> tuple[int, int]:
    """Record each node's candidate log files.  Returns ``(resolved, total)``.

    Junctions, filter stages and loop cuts all carry it because each is a thing
    an investigation is pointed at: a stage or a cut says why candidates were
    dropped, a junction says why a value was settled, and none of them can be
    checked without knowing which file to read.

    Junctions additionally get ``caller_log_files``, because for the largest
    group in the map the module that holds the code is not the one that logs
    about it.  A ``db_proxy_mods`` method inherits ``panda-DBProxy.log`` and
    ``panda-JediDBProxy.log`` from the proxy classes that mix it in, and those
    are true -- the SQL comment trace lands there -- but the line an
    investigation greps for, ``set task_status=``, is written by the knight
    that made the call.  Production settles it: of 33 files asked, that line is
    in exactly five -- ContentsFeeder, JobGenerator, PostProcessor,
    TaskCommando and TaskRefiner -- and in neither proxy file.  Each of the
    five reaches a proxy junction as a caller.

    Kept out of ``log_files`` on purpose.  One list would answer "where does
    this code live" and "whose log mentions it" with the same value, and
    conflating those is what made eleven candidates for a pending task look
    indistinguishable when nine of them are separable.

    ``owns_logger`` records which of the two ``log_files`` is.  A reader that
    cannot tell an inherited file from a declared one has no way to know that
    asking the proxy files for a caller's line always returns nothing, and would
    read that nothing as "the junction did not fire" -- for every proxy
    candidate at once.

    The same three facts are recorded for the owners the map names as reading
    or writing a value and that settle nothing themselves -- see
    :func:`_log_sites`.  Done here rather than in a pass of its own because the
    declaration, the inheritance chain and the callers are all already in hand;
    a second pass would be the same three readings again.
    """
    declared = declared_files(modules)
    inherited = inherited_files(modules, declared)
    inward = trigger.reaching_modules(modules)
    # A reader's own cadence, handed in rather than recomputed: the trigger
    # slice has already classified every module and a second call would be the
    # same reading twice, free to drift.  ``_follow_up`` asks whether anything
    # will re-evaluate the row and, with nothing on the log site, answers from
    # the *writers* -- so a value whose only self-repairing reader is a daemon
    # script that settles nothing reads as "only command, request".
    resolved = total = 0
    for node in (
        list(fragment.filter_stages) + list(fragment.loop_cuts) + list(fragment.junctions)
    ):
        total += 1
        node.log_files = files_of(node.owner.split("::")[0], declared, inherited)
        if node.log_files:
            resolved += 1

    for junction in fragment.junctions:
        junction.owns_logger = junction.owner.split("::")[0] in declared
        junction.caller_log_files = _caller_files(
            junction.owner, junction.log_files, declared, inherited, inward
        )
    fragment.log_sites.extend(
        _log_sites(fragment, declared, inherited, inward, cadence)
    )
    return resolved, total


def _actor_owners(fragment) -> set[str]:
    """Every ``module::function`` the map names as reading or writing a value.

    Both verbs and both models.  A descent follows ``read_by``, but a question
    about a value that only an ``UPDATE ... WHERE`` acts on still has to say
    which log will show that update running, and the answer is found the same
    way.
    """
    owners: set[str] = set()
    for subject in fragment.subjects:
        for names in list(subject.selected_by.values()) + list(subject.updated_by.values()):
            owners.update(names)
    for entity in fragment.entities:
        owners.update(entity.read_by)
        # All three writing verbs.  Which log will show the statement running
        # is the same question whether the row was made, moved or removed.
        owners.update(entity.created_by)
        owners.update(entity.updated_by)
        owners.update(entity.deleted_by)
    return owners


def _log_sites(
    fragment,
    declared: dict[str, str],
    inherited: dict[str, list[str]],
    inward,
    triggers: Optional[dict[str, set[str]]] = None,
) -> list[LogSiteNode]:
    """Where the readers and writers that own no junction write their diagnostics.

    ``_follow_up`` looks a reader up among the *writers* of the subject, so it
    only ever finds one that happens to settle a value too.  Measured on the
    installed corpus: 279 such owners, 94 of them junctions, and every one of
    the remaining 185 resolves to a file -- 53 to its own and 132 to a
    caller's as well.  The question "which log will say the query ran" was
    therefore unanswerable for two thirds of the readers the map names, not
    because the fact is missing but because nowhere held it.

    Owners that already own a junction are skipped.  Storing the same three
    fields twice is how one of them comes to disagree with the other.
    """
    known = {junction.owner for junction in fragment.junctions}
    sites: list[LogSiteNode] = []
    for owner in sorted(_actor_owners(fragment) - known):
        where = owner.split("::")[0]
        mine = files_of(where, declared, inherited)
        sites.append(
            LogSiteNode(
                map_id=fragment.map_id,
                derived_from=fragment.derived_from,
                name=owner,
                owner=owner,
                log_files=mine,
                caller_log_files=_caller_files(owner, mine, declared, inherited, inward),
                owns_logger=where in declared,
                triggers=sorted((triggers or {}).get(where, ())),
            )
        )
    return sites
