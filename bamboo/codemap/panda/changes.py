"""Whether a call may change a value an attribute chain holds: ``job.PandaID`` across ``job.valuesMap()``.

A reader that saw a value at one statement asks whether it still holds at another, and a call between the two is
the question: :mod:`mustprints` asks it of a print and the arm it names, the querier of a printed value and a later
statement.  :func:`may_change` answers it for one call, with why (:class:`Change`), or ``None`` where the call
cannot change it.

**The object, followed through the code.**  A call handed the object -- ``job`` for ``job.PandaID``, or ``self``
for ``self.job.PandaID`` -- runs with it under the names it is handed to: the definition's parameter, ``self`` for a
method called on it (:func:`calls.bind`), and the names bound to it there (``j = job``, ``spec = self.job``).  The
call changes the value where something it runs stores the chain's next attribute on one of those names
(``job.PandaID = 3``, ``self.job = other``), writes a name it computes on one (``setattr(job, name, v)``,
``job.__dict__.update(d)``, ``vars(job)``), or hands it on to a call that does.  A store on another object does not
count: ``JobSpec()`` builds a new one, and ``file.status = ...`` is not ``taskSpec.status``.  As :mod:`mustprints`
reads a call it is not handed, an object is taken to change only through the names it is handed under.

**Where it cannot be followed.**  A definition that puts the object where the reading loses it -- into another
object's attribute or a container, a ``return``, a closure, a value that may be the object (``a or b``, a
builtin's ``max(a, b)``, a library's value), a call placed nowhere or one that keeps what it is handed -- may let
code reach it under any name.  The call is then read by the attribute's name instead: it may change the value where
something it runs stores an attribute named one of the chain's, on any object, or writes a name it computes.

**What a call runs.**  The definitions :meth:`Calls.callees` places it at, and what those call, to the end.  A
call no definition is placed for is read so:

- a method on a receiver whose class is not known: every method of the corpus of its name, as
  :meth:`Calls.missed` reads such a call -- save ``execute`` and ``executemany``, which land in the database's
  driver (:data:`sql.CURSOR_RUNNERS`), and a name no method of the corpus has, which is a library's;
- a library's function or method, a class built on one, a builtin class built (``RuntimeError(...)``): stores no
  attribute of the corpus's objects (:data:`callfacts.ASSUMPTIONS` ``library_sets_no_corpus_attribute``), though it
  may keep what it is handed; a function of the corpus it is handed -- ``executor.submit(self.process, job)`` -- is
  read as called;
- a builtin function: as :data:`callfacts.BUILTIN_FUNCTIONS` says it -- one calling a function it is handed
  (``sorted(xs, key=f)``) calls it -- and ``setattr`` and ``delattr`` by the name they are handed; any other builtin
  may write every name;
- a callee chosen at run time, or a name bound to something that is not a function: may write every name.

Read by the name, a store to an attribute a property of the corpus sets runs its setter, and a store to an item runs
the corpus's ``__setitem__`` or ``__delitem__``; a spec's ``__setattr__`` writes the name its caller assigned
(:mod:`specwrites`' ``mechanism``).  An operator on an object of the corpus (``a + b`` running ``__add__``) is not
read.

**Cycles.**  A definition reached again adds nothing new: what a set of definitions that call each other does is
what any of them does, so the answer is the union over all a call reaches.
"""

from __future__ import annotations

import ast
import builtins
import weakref
from collections import deque
from typing import TYPE_CHECKING, NamedTuple, Optional

from bamboo.codemap import callfacts
from bamboo.codemap.calls import BUILTIN as BUILTIN_CLASS
from bamboo.codemap.calls import (
    UNPLACED_FORWARDED,
    UNPLACED_GETATTRIBUTE,
    UNPLACED_NO_INIT,
    UNPLACED_NO_METHOD,
    UNPLACED_OPEN,
    UNPLACED_OUTSIDE,
    UNPLACED_RECEIVER,
    Target,
    bind,
    own_nodes,
)

if TYPE_CHECKING:
    from bamboo.codemap.calls import Calls

#: :attr:`Change.name` where every name may be written.
EVERY = "*"
#: :attr:`Change.why`: an attribute of the name stored; a name computed at run time; a call this reading cannot follow.
STORED, COMPUTED, UNREAD = "stored", "computed", "unread"

#: The calls running a statement on the driver's cursor (:data:`sql.CURSOR_RUNNERS`): they land in the driver, never
#: in a function of the corpus of that name.
_DRIVER = frozenset(callfacts.CURSOR_RUNS)
#: Placed nowhere because the receiver's class cannot be said: read by the method's name.
_BY_NAME = frozenset(
    {UNPLACED_RECEIVER, UNPLACED_NO_METHOD, UNPLACED_GETATTRIBUTE, UNPLACED_FORWARDED}
)
#: Placed outside the corpus: a library's.
_OUTSIDE = frozenset({UNPLACED_OUTSIDE, UNPLACED_NO_INIT})
#: Builtins writing an attribute by the name they are handed second.
_NAMED_WRITES = frozenset({"setattr", "delattr"})
_SETATTR_METHODS = frozenset({"__setattr__", "__delattr__"})
#: The attribute whose store gives an object another class's attributes, and the one holding them all.
_CLASS, _DICT = "__class__", "__dict__"
#: A builtin handing back what holds every attribute of what it is handed.
_VARS = "vars"
#: What a builtin function's value holds of what it is handed where it holds none of it (:class:`callfacts.Function`).
_KEEPS_NONE = frozenset(
    {
        callfacts.TEXT,
        callfacts.NUMBER,
        callfacts.COMPUTED,
        callfacts.TELLS,
        callfacts.NOTHING,
    }
)
#: Where the value of a test is only tested: what reads ``a or b`` there takes no object from it.
_TESTS = (ast.If, ast.While, ast.Assert, ast.UnaryOp, ast.Compare)
#: Where an object read hands it nowhere: a comparison, a text it is formatted into, a statement of its own.
_READ_ONLY = (
    ast.Compare,
    ast.FormattedValue,
    ast.Expr,
    ast.If,
    ast.While,
    ast.Assert,
    ast.UnaryOp,
)


class Change(NamedTuple):
    """Why a call may change a value: at *file* and *line*, *name* stored (:data:`EVERY` for any) -- *why* is
    :data:`STORED`, :data:`COMPUTED` or :data:`UNREAD`, *detail* the call or builtin not followed."""

    file: str
    line: int
    name: str
    why: str
    detail: str = ""


class _Follow(NamedTuple):
    """A definition run with names holding the object asked of: each name, and the attributes from it to the
    value."""

    target: Target
    held: tuple[tuple[str, tuple[str, ...]], ...]


#: Where the object goes where it cannot be followed: the call is read by the attribute's name.
_ESCAPED = Change("", 0, EVERY, UNREAD, "the object goes where it is not followed")


class _Effects(NamedTuple):
    """What one definition's own body does: the attribute names it stores (the first place each), the first place
    it may write every name, and the definitions it runs."""

    stores: dict[str, Change]
    every: Optional[Change]
    runs: list[Target]


def may_change(
    calls: "Calls",
    file: str,
    func: ast.AST,
    call: ast.Call,
    value: ast.expr,
    bound: tuple = (),
) -> Optional[Change]:
    """Why *call*, made in *func* of *file*, may change what *value* holds there -- an attribute chain on a name,
    ``job.PandaID``, ``self.job.PandaID`` -- or ``None`` where it cannot.  *bound*: the classes ``self`` can be in
    *func* (:attr:`Target.bound`)."""
    chain = _chain(value)
    if chain is None:
        return Change(
            file, call.lineno, EVERY, UNREAD, "not an attribute chain on a name"
        )
    root, attributes = chain
    reading = _reading(calls)
    # The name holding the object, with the attributes from it to the value: ``job`` holds it through ``PandaID``.
    found = reading.followed(file, func, call, {root: attributes}, bound)
    if found is not _ESCAPED:
        return found
    asked = frozenset(attributes)
    stores, every, runs = reading.call(file, func, call, bound)
    return _first(stores, every, asked) or reading.reached(runs, asked)


def _chain(value: ast.expr) -> Optional[tuple[str, tuple[str, ...]]]:
    """``("job", ("PandaID",))`` for ``job.PandaID``."""
    attributes: list[str] = []
    while isinstance(value, ast.Attribute):
        attributes.append(value.attr)
        value = value.value
    if not isinstance(value, ast.Name) or not attributes:
        return None
    return value.id, tuple(reversed(attributes))


def _held(
    expression: ast.expr, held: dict[str, tuple[str, ...]]
) -> Optional[tuple[str, ...]]:
    """The attributes from *expression* to the value, where it is a name of *held* or an attribute on the way from
    one -- ``self.job`` for ``self`` holding ``job.PandaID`` -- and not the value itself."""
    if isinstance(expression, ast.Name):
        return held.get(expression.id)
    if isinstance(expression, ast.Attribute):
        rest = _held(expression.value, held)
        if rest is not None and len(rest) > 1 and expression.attr == rest[0]:
            return rest[1:]
    return None


def _first(
    stores: dict[str, Change], every: Optional[Change], asked: frozenset[str]
) -> Optional[Change]:
    if every is not None:
        return every
    return next((stores[name] for name in sorted(asked) if name in stores), None)


#: One reading per corpus's calls, while they live.
_READINGS: "weakref.WeakKeyDictionary[Calls, _Reading]" = weakref.WeakKeyDictionary()


def _reading(calls: "Calls") -> "_Reading":
    if calls not in _READINGS:
        _READINGS[calls] = _Reading(calls)
    return _READINGS[calls]


def _key(target: Target) -> tuple:
    return (
        target.file,
        id(target.func),
        tuple((file, id(cls)) for file, cls in target.bound),
    )


class _Reading:
    """The effects of the corpus's definitions, read once each, and which reached sets were found storing none of
    the names asked."""

    def __init__(self, calls: "Calls") -> None:
        self.calls = calls
        self.types = calls.types
        # Keyed by id; each value holds its definition, so the id is not reused while the entry lives.
        self.effects: dict[tuple, tuple[ast.AST, _Effects]] = {}
        self.clean: dict[frozenset[str], set[tuple]] = {}
        self.follows: dict[
            tuple, tuple[ast.AST, tuple[Optional[Change], list[_Follow]]]
        ] = {}
        self.followed_clean: set[tuple] = set()
        self._methods: Optional[dict[str, list[Target]]] = None
        self._functions: Optional[dict[str, list[Target]]] = None
        self._setters: Optional[dict[str, list[Target]]] = None

    def followed(
        self,
        file: str,
        func: ast.AST,
        call: ast.Call,
        held: dict[str, tuple[str, ...]],
        bound: tuple,
    ) -> Optional[Change]:
        """The first change *call* makes to the value, following the object the names *held* hold (each with the
        attributes from it to the value) into what it runs; :data:`_ESCAPED` where it goes where it is not
        followed."""
        first = self.handed_objects(file, func, call, held, bound)
        if isinstance(first, Change):
            return first
        seen: set[tuple] = set()
        queue = deque(first)
        while queue:
            node = queue.popleft()
            key = (_key(node.target), node.held)
            if key in seen or key in self.followed_clean:
                continue
            seen.add(key)
            if key not in self.follows:
                self.follows[key] = (node.target.func, self._follow(node))
            found, runs = self.follows[key][1]
            if found is not None:
                return found
            queue.extend(runs)
        # Every definition reached was read with the object and none changes it or loses it.
        self.followed_clean |= seen
        return None

    def handed_objects(
        self,
        file: str,
        func: ast.AST,
        call: ast.Call,
        held: dict[str, tuple[str, ...]],
        bound: tuple,
    ) -> "Change | list[_Follow]":
        """What *call* does with the object the names *held* hold (each with the attributes from it to the value):
        a change it makes itself, :data:`_ESCAPED`, or the definitions it runs with it."""

        def held_by(expression: ast.expr) -> Optional[tuple[str, ...]]:
            return _held(expression, held)

        called = call.func
        receiver = held_by(called.value) if isinstance(called, ast.Attribute) else None
        values = [*call.args, *(keyword.value for keyword in call.keywords)]
        # Handed inside something else -- ``self.sort([job])`` -- the object is lost.
        if any(
            isinstance(one, ast.Name) and one.id in held and _loses(one, held)
            for value in values
            for one in ast.walk(value)
        ):
            return _ESCAPED
        handed = [
            value
            for value in values
            if held_by(value.value if isinstance(value, ast.Starred) else value)
        ]
        if receiver is None and not handed:
            return []
        if any(isinstance(value, ast.Starred) for value in handed) or any(
            keyword.arg is None and held_by(keyword.value) for keyword in call.keywords
        ):
            return _ESCAPED
        if isinstance(called, ast.Name) and self.types.is_builtin(
            file, func, called.id
        ):
            return self._builtin_handed(file, call, held_by)
        if isinstance(called, ast.Attribute) and called.attr in _SETATTR_METHODS:
            off_class = isinstance(called.value, ast.Name) and (
                called.value.id == "object"
                or self.types.class_named(file, called.value) is not None
            )
            obj = call.args[0] if off_class and call.args else called.value
            return self._set_by_name(file, call, held_by, obj, 1 if off_class else 0)
        try:
            placed = self.calls.callees(file, func, call, bound)
        except Exception:  # noqa: BLE001 -- a placing that fails says nothing
            return _ESCAPED
        if placed.thread or placed.forwarded:
            return _ESCAPED
        targets = list(placed.targets)
        if not targets or placed.unplaced == UNPLACED_OPEN:
            if isinstance(called, ast.Attribute) and (
                placed.unplaced in _BY_NAME or placed.unplaced == UNPLACED_OPEN
            ):
                if called.attr in _DRIVER and not targets:
                    return []
                named = (
                    []
                    if self._outside_receiver(file, func, called.value)
                    else self.methods().get(called.attr, [])
                )
                if not named:
                    # Where else it lands is a library's or a builtin's method: it sets no attribute of the
                    # corpus's, and may keep one it is handed.
                    if handed:
                        return _ESCAPED
                    if not targets:
                        return []
                targets += named
            elif placed.unplaced in _OUTSIDE:
                # A library's function: it sets no attribute of the corpus's, but its value may be the object, and a
                # function of the corpus it is handed may be handed the object.
                kept = not isinstance(
                    getattr(call, "parent", None), ast.Expr
                ) or self._handed(file, call)
                return _ESCAPED if kept else []
            else:
                return _ESCAPED
        follows: list[_Follow] = []
        for target in targets:
            try:
                parameters = bind(call, target, placed.on_class)
            except Exception:  # noqa: BLE001
                return _ESCAPED
            given = [
                (name, rest)
                for name, value in parameters.items()
                if value is not None and (rest := held_by(value))
            ]
            if sum(
                1
                for value in parameters.values()
                if value is not None and held_by(value)
            ) < len(handed):
                # An object handed to no parameter this reading names: a ``*args``.
                return _ESCAPED
            if receiver is not None:
                own = _self_parameter(target, placed.on_class)
                if own is None:
                    return _ESCAPED
                given.append((own, receiver))
            follows.append(_Follow(target, tuple(sorted(set(given)))))
        return follows

    def _builtin_handed(
        self, file: str, call: ast.Call, held_by
    ) -> "Change | list[_Follow]":
        name = call.func.id  # type: ignore[attr-defined]
        if name in _NAMED_WRITES:
            return self._set_by_name(
                file, call, held_by, call.args[0] if call.args else None, 1
            )
        if name == _VARS:
            return Change(file, call.lineno, EVERY, COMPUTED, name)
        said = callfacts.BUILTIN_FUNCTIONS.get(name)
        if said is not None and said.gives in _KEEPS_NONE and not said.calls:
            return []
        return _ESCAPED

    def _set_by_name(
        self, file: str, call: ast.Call, held_by, obj: Optional[ast.expr], position: int
    ) -> "Change | list[_Follow]":
        """``setattr(obj, name, v)``, ``obj.__setattr__(name, v)``: a change where *obj* holds the value and the
        name is the next attribute or is computed; the object lost where it is the value set."""
        rest = held_by(obj) if obj is not None else None
        others = [one for one in call.args if one is not obj and held_by(one)]
        if others or rest is None:
            return _ESCAPED
        named = call.args[position] if len(call.args) > position else None
        if isinstance(named, ast.Constant) and isinstance(named.value, str):
            return (
                Change(file, call.lineno, named.value, STORED)
                if named.value == rest[0]
                else []
            )
        return Change(file, call.lineno, EVERY, COMPUTED, ast.unparse(call.func))

    def _follow(self, node: _Follow) -> tuple[Optional[Change], list[_Follow]]:
        """What a definition run with the object does to it: the first change, :data:`_ESCAPED`, or the definitions
        it hands the object to."""
        target = node.target
        func = target.func
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return _ESCAPED, []
        held = dict(node.held)
        nodes = list(own_nodes(func))
        # The names bound to an object held: ``j = job``, ``spec = self.job``.
        grown = True
        while grown:
            grown = False
            for one in nodes:
                if (
                    not isinstance(one, (ast.Assign, ast.AnnAssign))
                    or one.value is None
                ):
                    continue
                rest = _held(one.value, held)
                if rest is None:
                    continue
                for bound_to in (
                    one.targets if isinstance(one, ast.Assign) else [one.target]
                ):
                    if not isinstance(bound_to, ast.Name):
                        return _ESCAPED, []
                    if held.get(bound_to.id, rest) != rest:
                        return _ESCAPED, []
                    if bound_to.id not in held:
                        held[bound_to.id] = rest
                        grown = True
        for inner in ast.walk(func):
            if inner is not func and isinstance(
                inner, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
            ):
                if any(
                    isinstance(one, ast.Name) and one.id in held
                    for one in ast.walk(inner)
                ):
                    return _ESCAPED, []

        def held_by(expression: ast.expr) -> Optional[tuple[str, ...]]:
            return _held(expression, held)

        runs: list[_Follow] = []
        for one in nodes:
            line = getattr(one, "lineno", 0)
            if isinstance(one, ast.Attribute) and isinstance(
                one.ctx, (ast.Store, ast.Del)
            ):
                rest = held_by(one.value)
                if rest is not None and one.attr == rest[0]:
                    return Change(target.file, line, one.attr, STORED), []
                if rest is not None and one.attr in (_CLASS, _DICT):
                    return Change(target.file, line, EVERY, COMPUTED, one.attr), []
            elif (
                isinstance(one, ast.Attribute)
                and one.attr == _DICT
                and held_by(one.value)
                and _changed_through(one)
            ):
                return Change(target.file, line, EVERY, COMPUTED, _DICT), []
            elif (
                isinstance(one, ast.Subscript)
                and isinstance(one.ctx, (ast.Store, ast.Del))
                and held_by(one.value)
            ):
                return _ESCAPED, []
            elif (
                isinstance(one, ast.Name)
                and isinstance(one.ctx, ast.Load)
                and one.id in held
            ):
                if _loses(one, held):
                    return _ESCAPED, []
            elif isinstance(one, ast.Call):
                found = self.handed_objects(target.file, func, one, held, target.bound)
                if isinstance(found, Change):
                    return found, []
                runs.extend(found)
        return None, runs

    def reached(self, starts: list[Target], asked: frozenset[str]) -> Optional[Change]:
        """The first change of *asked* anything *starts* runs, to the end, may make."""
        clean = self.clean.setdefault(asked, set())
        seen: set[tuple] = set()
        queue = deque(start for start in starts if _key(start) not in clean)
        while queue:
            target = queue.popleft()
            key = _key(target)
            if key in seen or key in clean:
                continue
            seen.add(key)
            effects = self.of(target)
            found = _first(effects.stores, effects.every, asked)
            if found is not None:
                return found
            queue.extend(effects.runs)
        # Every definition reached was read and none writes the names: none reaches one that does.
        clean |= seen
        return None

    def of(self, target: Target) -> _Effects:
        key = _key(target)
        if key not in self.effects:
            self.effects[key] = (target.func, self._read(target))
        return self.effects[key][1]

    def _read(self, target: Target) -> _Effects:
        stores: dict[str, Change] = {}
        every: Optional[Change] = None
        runs: list[Target] = []
        # A lambda's calls are placed in the definition holding it, whose names it reads.
        lambda_ = isinstance(target.func, ast.Lambda)
        nodes = ast.walk(target.func.body) if lambda_ else own_nodes(target.func)  # type: ignore[attr-defined]
        placing = _enclosing_definition(target.func) if lambda_ else target.func
        for node in nodes:
            line = getattr(node, "lineno", 0)
            if isinstance(node, ast.Attribute) and isinstance(
                node.ctx, (ast.Store, ast.Del)
            ):
                if node.attr in (_CLASS, _DICT):
                    every = every or Change(
                        target.file, line, EVERY, COMPUTED, node.attr
                    )
                stores.setdefault(
                    node.attr, Change(target.file, line, node.attr, STORED)
                )
                runs.extend(self.setters().get(node.attr, ()))
            elif (
                isinstance(node, ast.Attribute)
                and node.attr == _DICT
                and _changed_through(node)
            ):
                every = every or Change(target.file, line, EVERY, COMPUTED, _DICT)
            elif isinstance(node, ast.Subscript) and isinstance(
                node.ctx, (ast.Store, ast.Del)
            ):
                runs.extend(
                    self.methods().get(
                        "__setitem__"
                        if isinstance(node.ctx, ast.Store)
                        else "__delitem__",
                        (),
                    )
                )
            elif isinstance(node, ast.Call) and placing is not None:
                more, wide, landed = self.call(
                    target.file, placing, node, () if lambda_ else target.bound
                )
                for name, change in more.items():
                    stores.setdefault(name, change)
                every = every or wide
                runs.extend(landed)
        return _Effects(stores, every, runs)

    def call(
        self, file: str, func: ast.AST, call: ast.Call, bound: tuple
    ) -> tuple[dict[str, Change], Optional[Change], list[Target]]:
        """What *call* does itself -- a builtin writing a name -- and the definitions it runs."""
        line = call.lineno
        called = call.func
        if isinstance(called, ast.Name) and self.types.is_builtin(
            file, func, called.id
        ):
            return self._builtin(file, func, call)
        if isinstance(called, ast.Attribute) and called.attr in _SETATTR_METHODS:
            # Read off a class -- ``object.__setattr__(self, name, value)`` -- the object comes first.
            off_class = isinstance(called.value, ast.Name) and (
                called.value.id == "object"
                or self.types.class_named(file, called.value) is not None
            )
            return self._named_write(file, func, call, 1 if off_class else 0)
        try:
            placed = self.calls.callees(file, func, call, bound)
        except Exception:  # noqa: BLE001 -- a placing that fails says nothing
            return {}, Change(file, line, EVERY, UNREAD, "not placed"), []
        runs = list(placed.targets)
        if placed.targets and placed.unplaced != UNPLACED_OPEN:
            return {}, None, runs
        if isinstance(called, ast.Attribute) and (
            placed.unplaced in _BY_NAME or placed.unplaced == UNPLACED_OPEN
        ):
            if called.attr in _DRIVER:
                return {}, None, runs
            if self._outside_receiver(file, func, called.value):
                return {}, None, runs + self._handed(file, call)
            named = self.methods().get(called.attr, [])
            return {}, None, runs + (named or self._handed(file, call))
        if placed.unplaced in _OUTSIDE:
            return {}, None, runs + self._handed(file, call)
        return (
            {},
            Change(file, line, EVERY, UNREAD, placed.unplaced or "not placed"),
            runs,
        )

    def _builtin(
        self, file: str, func: ast.AST, call: ast.Call
    ) -> tuple[dict[str, Change], Optional[Change], list[Target]]:
        name = call.func.id  # type: ignore[attr-defined]
        if name in _NAMED_WRITES:
            return self._named_write(file, func, call, 1)
        if name == _VARS:
            return {}, Change(file, call.lineno, EVERY, COMPUTED, name), []
        said = callfacts.BUILTIN_FUNCTIONS.get(name)
        if said is not None:
            return {}, None, self._handed(file, call) if said.calls else []
        if isinstance(getattr(builtins, name, None), type):
            return {}, None, []
        return {}, Change(file, call.lineno, EVERY, UNREAD, name), []

    def _named_write(
        self, file: str, func: ast.AST, call: ast.Call, position: int
    ) -> tuple[dict[str, Change], Optional[Change], list[Target]]:
        """``setattr(x, "a", v)``, ``x.__setattr__("a", v)``: the name at *position*, or every name where it is
        computed -- save in a ``__setattr__`` itself, which writes the name its caller assigned."""
        handed = call.args[position] if len(call.args) > position else None
        if isinstance(handed, ast.Constant) and isinstance(handed.value, str):
            return (
                {handed.value: Change(file, call.lineno, handed.value, STORED)},
                None,
                [],
            )
        if getattr(func, "name", "") in _SETATTR_METHODS:
            return {}, None, []
        return (
            {},
            Change(file, call.lineno, EVERY, COMPUTED, ast.unparse(call.func)),
            [],
        )

    def _outside_receiver(self, file: str, func: ast.AST, receiver: ast.expr) -> bool:
        """Whether *receiver* is known to be an object of a library's class or of a builtin one -- a literal, or a
        name every value of which is one (``update_msg_dict = {...}``), as :func:`callfacts.meaning` reads it."""
        if isinstance(
            receiver,
            (
                ast.List,
                ast.ListComp,
                ast.Dict,
                ast.DictComp,
                ast.Set,
                ast.SetComp,
                ast.Constant,
            ),
        ):
            return True
        try:
            if self.types.library_classes(file, func, receiver):
                return True
            classes, opened = self.types.candidates(file, func, receiver)
        except Exception:  # noqa: BLE001 -- a reading that fails says nothing
            return False
        return (
            bool(classes)
            and not opened
            and all(where == BUILTIN_CLASS for where, _name in classes)
        )

    def _handed(self, file: str, call: ast.Call) -> list[Target]:
        """The functions of the corpus *call* is handed, read as called: a lambda's body, and every definition of
        the name handed (``self.process`` -- every method ``process``; ``process`` -- every function or method)."""
        found: list[Target] = []
        for value in [*call.args, *(keyword.value for keyword in call.keywords)]:
            if isinstance(value, ast.Starred):
                value = value.value
            if isinstance(value, ast.Lambda):
                found.append(Target(file, value, None))
            elif isinstance(value, ast.Attribute):
                found.extend(self.methods().get(value.attr, ()))
            elif isinstance(value, ast.Name):
                found.extend(self.functions().get(value.id, ()))
                found.extend(self.methods().get(value.id, ()))
        return found

    def methods(self) -> dict[str, list[Target]]:
        """Every method of the corpus by its name."""
        if self._methods is None:
            self._index()
        return self._methods  # type: ignore[return-value]

    def functions(self) -> dict[str, list[Target]]:
        """Every function of the corpus not in a class, by its name."""
        if self._functions is None:
            self._index()
        return self._functions  # type: ignore[return-value]

    def setters(self) -> dict[str, list[Target]]:
        """Every property setter of the corpus, by the property's name."""
        if self._setters is None:
            self._index()
        return self._setters  # type: ignore[return-value]

    def _index(self) -> None:
        self._methods, self._functions, self._setters = {}, {}, {}
        for file, (_sha, tree) in sorted(self.types.walked.items()):
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    for member in node.body:
                        if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            target = Target(file, member, node)
                            self._methods.setdefault(member.name, []).append(target)
                            if any(_is_setter(one) for one in member.decorator_list):
                                self._setters.setdefault(member.name, []).append(target)
                elif isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef)
                ) and not isinstance(getattr(node, "parent", None), ast.ClassDef):
                    self._functions.setdefault(node.name, []).append(
                        Target(file, node, None)
                    )


def _is_setter(decorator: ast.expr) -> bool:
    return isinstance(decorator, ast.Attribute) and decorator.attr in (
        "setter",
        "deleter",
    )


def _enclosing_definition(node: ast.AST) -> Optional[ast.AST]:
    here = getattr(node, "parent", None)
    while here is not None and not isinstance(
        here, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)
    ):
        here = getattr(here, "parent", None)
    return here


def _changed_through(node: ast.Attribute) -> bool:
    """Whether ``x.__dict__`` is changed where it is read: an item stored in it, or a method changing a dict called
    on it (:data:`callfacts.BUILTIN_METHODS`)."""
    parent = getattr(node, "parent", None)
    if isinstance(parent, ast.Subscript) and parent.value is node:
        return isinstance(parent.ctx, (ast.Store, ast.Del))
    if isinstance(parent, ast.Attribute) and parent.value is node:
        method = callfacts.BUILTIN_METHODS.get(("dict", parent.attr))
        return method is None or method.changes
    return False


def _self_parameter(target: Target, on_class: bool) -> Optional[str]:
    """The parameter a method called on an object is handed it as: its first, unless it is a static or a class
    method, or read off the class (then the object is an argument, which :func:`calls.bind` binds)."""
    func = target.func
    if (
        target.cls is None
        or on_class
        or not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef))
    ):
        return None
    decorators = {
        getattr(one, "id", getattr(one, "attr", "")) for one in func.decorator_list
    }
    if decorators & {"staticmethod", "classmethod"}:
        return None
    positional = [*func.args.posonlyargs, *func.args.args]
    return positional[0].arg if positional else None


def _loses(name: ast.Name, held: dict[str, tuple[str, ...]]) -> bool:
    """Whether this read of a name holding the object -- or of an attribute on the way from it to the value -- puts
    the object where the reading loses it: anywhere but an attribute read off it, a call it is handed to or made on,
    a name bound to it, a test or a text."""
    here: ast.expr = name
    rest = held[name.id]
    parent = getattr(here, "parent", None)
    while (
        isinstance(parent, ast.Attribute)
        and parent.value is here
        and len(rest) > 1
        and parent.attr == rest[0]
    ):
        here, rest = parent, rest[1:]
        parent = getattr(here, "parent", None)
    if isinstance(parent, ast.Attribute) and parent.value is here:
        return False
    if isinstance(parent, ast.Call) and any(one is here for one in parent.args):
        return False
    if isinstance(parent, ast.keyword) and parent.arg is not None:
        return False
    if isinstance(parent, (ast.Assign, ast.AnnAssign)) and parent.value is here:
        return False
    if (
        isinstance(parent, ast.Subscript)
        and parent.value is here
        and isinstance(parent.ctx, ast.Load)
    ):
        return False
    if isinstance(parent, (ast.BoolOp, ast.IfExp)):
        return not _only_tested(parent)
    return not isinstance(parent, _READ_ONLY)


def _only_tested(node: ast.expr) -> bool:
    """Whether ``a or b`` here is only tested, its value taken by nothing."""
    parent = getattr(node, "parent", None)
    if isinstance(parent, (ast.BoolOp, ast.IfExp)):
        return _only_tested(parent)
    return isinstance(parent, _TESTS)
