"""What a call does where no body of the corpus says it: one table of facts, and one order to read a call by.

The readers of the map ask different questions of a call -- where a statement's rows go (the crossing walk), whether
a value is read (its twin), where an object of the corpus is put (the reach record), where a key column goes (the
objects reading), what a name holds (the trace) -- and before this each kept its own list of the method names a
container is filled with, with its own contents and its own order: one asked the placing first, one the name alone.
Here the facts are said once, as what the callee does (:data:`BUILTIN_METHODS`), and :func:`meaning` reads a call in
one order:

1. a protocol answering ahead of a body -- ``x.__setstate__(y.__getstate__())`` is pickle's round trip, a copy of
   ``y`` into ``x``, whatever the two bodies are read as (:data:`PICKLE_ROUND_TRIP`);
2. a definition of the corpus the call lands at: its body says what it does (:data:`DEFINITION`);
3. a logger's method, by its name (:data:`LOG_METHODS`), where no definition answers: it writes a line and keeps
   nothing it is handed (:data:`LOGGER`, the assumption ``logger_keeps_nothing``) -- ``LogWrapper`` is a library
   class, and step 5 would read ``tmp_log.debug(msg)`` as keeping the message;
4. an object of another system's client -- its class, or the base outside the corpus a class of the corpus is built
   on, defined in a package :data:`OTHER_SYSTEMS` names: it hands what it is given to that system, another run's
   input, and keeps nothing of it (:data:`CLIENT`, the assumption ``other_systems_keep_nothing``) -- Rucio's
   ``Client``, ``smtplib.SMTP``, the message broker's agent;
5. an object of any other class outside the corpus (:meth:`calls.Types.library_classes`), or of a class of the
   corpus built on a base outside it that supplies the method (:meth:`calls.Types.built_outside`): nothing here says
   what the method does with what it is handed, so it keeps all of it (:data:`LIBRARY`);
6. a builtin container: its type's method (:data:`BUILTIN`);
7. an object whose class cannot be said, or a value a call outside the corpus made -- ``json.loads``', a row the
   database returned: the builtin's method of the name (:data:`ASSUMED`, the assumption ``builtin_by_name``);
8. anything else: nothing is said (:data:`UNKNOWN`) -- a module's function, a method read off a class, a class of
   the corpus that has no such method.

Each reader keeps its question and answers it from the :class:`Method` it is given.  A reader that reads text alone,
with no placing -- the trace, what may change a name between two lines -- reads the facts by the name
(:func:`by_name`), which is step 7's assumption for every call.

The facts here are the putting ones, which ``append``, ``add``, ``update`` and the like are, the taking ones --
what ``get``, ``pop``, ``items``, ``copy`` and the like hand back of the receiver (:class:`Give`) --, whether a
method changes its receiver, a logger's methods, and what a builtin function's value holds of what it is handed --
``list(xs)``, ``max(a, b)``, ``enumerate(xs)``, ``getattr(x, name)`` (:class:`Function`), read where the name is
the builtin's (:func:`builtin_function`); what else a builtin function does still sits with its readers.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING, NamedTuple, Optional

if TYPE_CHECKING:
    from bamboo.codemap.calls import Calls, Placed, Types

#: Where a method puts one of what it is handed: as one element, its elements as elements, as a key, as the value
#: under a key it is also handed, a mapping's keys and values merged, or the whole of it.
ELEMENT, ELEMENTS, KEY, VALUE, MERGE, WHOLE = (
    "element",
    "elements",
    "key",
    "value",
    "merge",
    "whole",
)
#: Every positional argument, every keyword argument.
EVERY_POSITIONAL, EVERY_KEYWORD = "*", "**"


class Put(NamedTuple):
    """A method puts one of what it is handed into its receiver.

    *argument* is a position, a keyword's name, :data:`EVERY_POSITIONAL` or :data:`EVERY_KEYWORD`; *into* is
    :data:`ELEMENT`, :data:`ELEMENTS`, :data:`KEY`, :data:`VALUE`, :data:`MERGE` or :data:`WHOLE`; *at*, for a
    :data:`VALUE` under a key the call is handed, that key's position.  A keyword's value is a :data:`VALUE` under
    the keyword's name (``d.update(k=v)``)."""

    argument: int | str
    into: str
    at: Optional[int] = None


#: What a method hands back of its receiver: another of the same make-up, one of what it holds, a view of its keys,
#: values or (key, value) pairs, or one (key, value) pair.
SELF, AT, KEYS, VALUES, ITEMS, ITEM = ("self", "at", "keys", "values", "items", "item")


class Give(NamedTuple):
    """What a method hands back of its receiver.

    *part* is :data:`SELF` (a copy), :data:`AT` (one of what it holds: the one the argument at position *key* names,
    any one where the call hands none -- ``d.get(k)``, ``xs.pop()``), :data:`KEYS` / :data:`VALUES` / :data:`ITEMS`
    (a view whose elements are its keys, its values, its (key, value) pairs) or :data:`ITEM` (one such pair);
    *default* the position of an argument the value may be instead (``d.get(k, default)``)."""

    part: str
    key: Optional[int] = None
    default: Optional[int] = None


class Method(NamedTuple):
    """What a method of a class the corpus does not hold does: what it puts into its receiver, whether it changes
    the receiver at all, and what it hands back of it."""

    puts: tuple[Put, ...] = ()
    changes: bool = False
    gives: Optional[Give] = None


def _changing(gives: Optional[Give] = None) -> Method:
    return Method((), True, gives)


def _giving(gives: Give) -> Method:
    return Method((), False, gives)


#: The builtin containers' methods that put into them, change them or hand back a part of them, by ``(type, name)``.
#: ``collections.deque``'s are here too: :meth:`calls.Types.library_classes` reads ``collections`` as the builtins'
#: (its values are).
BUILTIN_METHODS: dict[tuple[str, str], Method] = {
    ("list", "append"): Method((Put(0, ELEMENT),), True),
    ("list", "insert"): Method((Put(1, ELEMENT),), True),
    ("list", "extend"): Method((Put(0, ELEMENTS),), True),
    ("list", "pop"): _changing(Give(AT, 0)),
    ("list", "remove"): _changing(),
    ("list", "clear"): _changing(),
    ("list", "sort"): _changing(),
    ("list", "reverse"): _changing(),
    ("list", "copy"): _giving(Give(SELF)),
    ("list", "__getitem__"): _giving(Give(AT, 0)),
    ("set", "add"): Method((Put(0, ELEMENT),), True),
    ("set", "update"): Method((Put(EVERY_POSITIONAL, ELEMENTS),), True),
    ("set", "symmetric_difference_update"): Method((Put(0, ELEMENTS),), True),
    ("set", "difference_update"): _changing(),
    ("set", "intersection_update"): _changing(),
    ("set", "discard"): _changing(),
    ("set", "remove"): _changing(),
    ("set", "pop"): _changing(Give(AT)),
    ("set", "clear"): _changing(),
    ("set", "copy"): _giving(Give(SELF)),
    ("dict", "update"): Method((Put(0, MERGE), Put(EVERY_KEYWORD, VALUE)), True),
    ("dict", "setdefault"): Method(
        (Put(0, KEY), Put(1, VALUE, at=0)), True, Give(AT, 0, 1)
    ),
    ("dict", "pop"): _changing(Give(AT, 0, 1)),
    ("dict", "popitem"): _changing(Give(ITEM)),
    ("dict", "clear"): _changing(),
    ("dict", "get"): _giving(Give(AT, 0, 1)),
    ("dict", "__getitem__"): _giving(Give(AT, 0)),
    ("dict", "keys"): _giving(Give(KEYS)),
    ("dict", "values"): _giving(Give(VALUES)),
    ("dict", "items"): _giving(Give(ITEMS)),
    ("dict", "copy"): _giving(Give(SELF)),
    ("deque", "append"): Method((Put(0, ELEMENT),), True),
    ("deque", "appendleft"): Method((Put(0, ELEMENT),), True),
    ("deque", "insert"): Method((Put(1, ELEMENT),), True),
    ("deque", "extend"): Method((Put(0, ELEMENTS),), True),
    ("deque", "extendleft"): Method((Put(0, ELEMENTS),), True),
    ("deque", "pop"): _changing(Give(AT)),
    ("deque", "popleft"): _changing(Give(AT)),
    ("deque", "remove"): _changing(),
    ("deque", "clear"): _changing(),
    ("deque", "rotate"): _changing(),
    ("deque", "reverse"): _changing(),
    ("deque", "copy"): _giving(Give(SELF)),
}

#: What a method of a class outside the corpus does where this table says nothing of it: it keeps every argument it is
#: handed in its object, whole (:data:`ASSUMPTIONS` ``library_keeps_what_it_is_handed``).
KEEPS_ALL = Method((Put(EVERY_POSITIONAL, WHOLE), Put(EVERY_KEYWORD, WHOLE)), True)

#: A logger's methods, by their names: a call of one writes a line and keeps nothing it is handed (:data:`ASSUMPTIONS`
#: ``logger_keeps_nothing``).  ``sendMsg`` / ``keepMsg`` are JEDI's ``MsgWrapper``'s, ``log`` ``logging.Logger``'s.
LOG_METHODS = frozenset(
    {
        "debug",
        "info",
        "warning",
        "warn",
        "error",
        "exception",
        "critical",
        "sendMsg",
        "keepMsg",
        "log",
    }
)

#: The other systems PanDA is a client of -> the packages it reaches each through.  What a client's method is handed
#: goes to that system: another run's input, not this run's (reach's premise ``cross_run_not_reach``).  Not the
#: database's drivers (the driver rule says where those go), not a library that works on what it is handed here
#: (``numpy``, ``yaml``, ``snakemake``), not a server the corpus runs (``fastmcp``, ``uvicorn``).  A package is a
#: module's name or the start of one: ``pandacommon.pandamsgbkr`` is the broker's, ``pandacommon.pandalogger`` not.
#: :data:`bamboo.codemap.panda.provenance.OUTSIDE_PACKAGES` reads this table.
OTHER_SYSTEMS: dict[str, tuple[str, ...]] = {
    "DDM / Rucio": ("rucio",),
    "iDDS": ("idds",),
    "message broker": ("pandacommon.pandamsgbkr",),
    "Kafka": ("pandacommon.kafkapublisher",),
    "PanDA's API": ("pandaclient",),
    "HTTP": ("requests", "urllib3", "httpx"),
    "mail": ("smtplib",),
}


def other_system(module: str) -> Optional[str]:
    """The system of :data:`OTHER_SYSTEMS` whose package *module* is in, or ``None``."""
    for system, packages in OTHER_SYSTEMS.items():
        if any(
            module == package or module.startswith(f"{package}.")
            for package in packages
        ):
            return system
    return None


#: The assumptions a reading by this module rests on, as reach's are said (:data:`bamboo.codemap.reach.ASSUMPTIONS`).
ASSUMPTIONS: dict[str, str] = {
    "builtin_by_name": (
        "A method whose receiver's class cannot be said -- or a value a call outside the corpus made -- is the "
        "builtin container's of its name."
    ),
    "other_systems_keep_nothing": (
        "A method of another system's client (OTHER_SYSTEMS: its class, or the base outside the corpus a class of the "
        "corpus is built on, is in one of its packages) hands what it is given to that system and keeps nothing of it."
    ),
    "library_keeps_what_it_is_handed": (
        "A method of an object whose class is outside the corpus, or is built on a base outside it that supplies the "
        "method, keeps all it is handed in that object."
    ),
    "logger_keeps_nothing": (
        "A method named as a logger's (debug, info, warning, error, ...) that no definition of the corpus answers for "
        "writes a line and keeps nothing it is handed."
    ),
    "pickle_round_trip": (
        "x.__setstate__(y.__getstate__()) makes x hold what y holds, attribute for attribute: __setstate__ restores "
        "what __getstate__ saved."
    ),
}

#: How :func:`meaning` reads a call (the module's order).
PROTOCOL, DEFINITION, LOGGER, CLIENT, LIBRARY, BUILTIN, ASSUMED, UNKNOWN = (
    "protocol",
    "definition",
    "logger",
    "client",
    "library",
    "builtin",
    "assumed",
    "unknown",
)
#: The protocol :func:`meaning` reads ahead of a body: its ASSUMPTIONS key.
PICKLE_ROUND_TRIP = "pickle_round_trip"


class Meaning(NamedTuple):
    """What a call does, as :func:`meaning` reads it: *kind* the step that answered, *method* the fact for
    :data:`LOGGER`, :data:`CLIENT`, :data:`LIBRARY`, :data:`BUILTIN` and :data:`ASSUMED`, *assumption* the
    :data:`ASSUMPTIONS` key the answer rests on, *copied_from* the ``y`` a :data:`PROTOCOL` copies into the call's
    receiver."""

    kind: str
    method: Optional[Method] = None
    assumption: Optional[str] = None
    copied_from: Optional[ast.expr] = None


_UNKNOWN = Meaning(UNKNOWN)


def by_name(name: str) -> Optional[Method]:
    """The facts of every builtin container's method called *name*, together: ``update`` is ``dict.update``'s and
    ``set.update``'s.  ``None`` where no builtin has one."""
    found = [
        method for (_type, called), method in BUILTIN_METHODS.items() if called == name
    ]
    return _together(found)


def _of_types(names: frozenset[str], name: str) -> Optional[Method]:
    found = [
        BUILTIN_METHODS[(one, name)]
        for one in sorted(names)
        if (one, name) in BUILTIN_METHODS
    ]
    return _together(found)


def _together(found: list[Method]) -> Optional[Method]:
    """Several types' facts of one name as one: every put, a change where one changes, and what they hand back where
    every one hands back the same part -- ``pop`` is one of what it holds for each, at the key a dict's is handed."""
    if not found:
        return None
    parts = {
        method.gives.part if method.gives is not None else None for method in found
    }
    gives = None
    if len(parts) == 1 and None not in parts:
        gives = Give(
            parts.pop(),
            next(
                (m.gives.key for m in found if m.gives and m.gives.key is not None),
                None,
            ),
            next(
                (
                    m.gives.default
                    for m in found
                    if m.gives and m.gives.default is not None
                ),
                None,
            ),
        )
    return Method(
        tuple(dict.fromkeys(put for method in found for put in method.puts)),
        any(m.changes for m in found),
        gives,
    )


#: The builtin container types the table says the methods of.
_TYPES = frozenset(one for one, _name in BUILTIN_METHODS)


def builtin_method(type_name: str, name: str) -> Optional[Method]:
    """*type_name*'s method *name*, where a reader knows the builtin type of its receiver; by the name alone
    (:func:`by_name`) where the type is none the table holds."""
    if type_name in _TYPES:
        return _of_types(frozenset({type_name}), name)
    return by_name(name)


def taken(name: str, type_name: Optional[str] = None) -> Optional[Give]:
    """What a builtin container's method *name* hands back of its receiver: *type_name*'s where a reader knows the
    builtin type and the type has the method, every type's of the name together otherwise -- a receiver read as a
    list and handed ``.get`` is not read right, and the name says more than the type there.  ``None`` where no
    builtin's method of the name hands back any."""
    if type_name in _TYPES and (type_name, name) in BUILTIN_METHODS:
        return BUILTIN_METHODS[(type_name, name)].gives
    method = by_name(name)
    return method.gives if method is not None else None


#: The names some builtin container's method puts with, the names some changes its receiver with, and the names
#: some hands back a part of its receiver with.
PUTTING = frozenset(
    called for (_type, called), method in BUILTIN_METHODS.items() if method.puts
)
CHANGING = frozenset(
    called for (_type, called), method in BUILTIN_METHODS.items() if method.changes
)
TAKING = frozenset(
    called
    for (_type, called), method in BUILTIN_METHODS.items()
    if method.gives is not None
)


#: What a builtin function's value holds of what it is handed (:class:`Function`): a new container or iterator of
#: the first argument's elements, one of them, (position, element) pairs of them, tuples of each argument's elements
#: in step, a dict of the first argument's pairs and of the keywords, the second argument's elements the first keeps,
#: what the first makes of each of the second's, the first's attribute the second names.
COPY, ONE, NUMBERED, ZIPPED, DICT, KEPT, MADE, ATTRIBUTE = (
    "copy",
    "one",
    "numbered",
    "zipped",
    "dict",
    "kept",
    "made",
    "attribute",
)


class Function(NamedTuple):
    """What a builtin function's value holds of what it is handed.

    *gives* is :data:`COPY` (``list(xs)``, ``sorted(xs)``, ``iter(xs)``), :data:`ONE` (one of the first argument's
    elements; one of the arguments where it is handed several and *default* is none -- ``max(a, b)``),
    :data:`NUMBERED` (``enumerate(xs)``), :data:`ZIPPED` (``zip(xs, ys)``), :data:`DICT` (``dict(pairs, k=v)``: a
    mapping's or the (key, value) pairs' and the keywords'), :data:`KEPT` (``filter(f, xs)``), :data:`MADE`
    (``map(f, xs)``) or :data:`ATTRIBUTE` (``getattr(x, name)``); *default* the position of an argument the value may
    be instead (``next(it, default)``, ``getattr(x, name, default)``)."""

    gives: str
    default: Optional[int] = None


#: The builtin functions whose value holds a part of what they are handed, by name.
BUILTIN_FUNCTIONS: dict[str, Function] = {
    "list": Function(COPY),
    "tuple": Function(COPY),
    "set": Function(COPY),
    "frozenset": Function(COPY),
    "sorted": Function(COPY),
    "reversed": Function(COPY),
    "iter": Function(COPY),
    "max": Function(ONE),
    "min": Function(ONE),
    "next": Function(ONE, 1),
    "enumerate": Function(NUMBERED),
    "zip": Function(ZIPPED),
    "dict": Function(DICT),
    "filter": Function(KEPT),
    "map": Function(MADE),
    "getattr": Function(ATTRIBUTE, 2),
}


def builtin_function(
    types: "Types", file: str, func: ast.AST, call: ast.Call
) -> Optional[Function]:
    """What *call* does where it calls a builtin function by its name -- one no scope binds there
    (:meth:`calls.Types.is_builtin`); ``None`` for any other call, and for a builtin :data:`BUILTIN_FUNCTIONS` does
    not say."""
    called = call.func
    if not isinstance(called, ast.Name) or called.id not in BUILTIN_FUNCTIONS:
        return None
    return (
        BUILTIN_FUNCTIONS[called.id]
        if types.is_builtin(file, func, called.id)
        else None
    )


def picked(function: Function, call: ast.Call) -> tuple[list[ast.expr], list[ast.expr]]:
    """For a :data:`ONE` call: the arguments whose element the value may be, and the arguments it may be itself --
    ``([it], [default])`` for ``next(it, default)``, ``([xs], [])`` for ``max(xs)``, ``([], [a, b])`` for ``max(a,
    b)``, ``([xs], [a])`` for ``max(*xs, a)``."""
    args = [one for one in call.args if not isinstance(one, ast.Starred)]
    starred = [one.value for one in call.args if isinstance(one, ast.Starred)]
    if starred:
        return starred, args
    if function.default is None and len(args) > 1:
        return [], args
    beside = (
        args[function.default : function.default + 1]
        if function.default is not None
        else []
    )
    return args[:1], beside


def round_trip(call: ast.Call) -> Optional[ast.expr]:
    """The ``y`` of ``x.__setstate__(y.__getstate__())``; ``None`` for any other call."""
    func = call.func
    if (
        not isinstance(func, ast.Attribute)
        or func.attr != "__setstate__"
        or len(call.args) != 1
        or call.keywords
    ):
        return None
    state = call.args[0]
    if (
        isinstance(state, ast.Call)
        and isinstance(state.func, ast.Attribute)
        and state.func.attr == "__getstate__"
        and not state.args
        and not state.keywords
    ):
        return state.func.value
    return None


def put_arguments(
    method: Method, call: ast.Call
) -> list[tuple[ast.expr, Put, Optional[ast.expr], Optional[str]]]:
    """``(value, put, key, keyword)`` for each expression *call* hands that *method* puts: *key* the expression a
    :data:`VALUE` is put under, *keyword* the name a keyword's value is put under.

    An expression put whole or merged is not put as its elements too: what holds it whole holds what each of its
    elements holds -- ``update`` by its name is ``dict.update``'s merge and ``set.update``'s elements, and the merge
    says both."""
    positional = [
        one.value if isinstance(one, ast.Starred) else one for one in call.args
    ]
    found: list[tuple[ast.expr, Put, Optional[ast.expr], Optional[str]]] = []
    for put in method.puts:
        if put.argument == EVERY_POSITIONAL:
            found.extend((one, put, None, None) for one in positional)
        elif put.argument == EVERY_KEYWORD:
            found.extend(
                (keyword.value, put, None, keyword.arg)
                for keyword in call.keywords
                if keyword.arg
            )
        elif isinstance(put.argument, int):
            if put.argument < len(positional):
                key = (
                    positional[put.at]
                    if put.at is not None and put.at < len(positional)
                    else None
                )
                found.append((positional[put.argument], put, key, None))
        else:
            found.extend(
                (keyword.value, put, None, keyword.arg)
                for keyword in call.keywords
                if keyword.arg == put.argument
            )
    whole = {
        id(one) for one, put, _key, _keyword in found if put.into in (MERGE, WHOLE)
    }
    return [
        one for one in found if not (one[1].into == ELEMENTS and id(one[0]) in whole)
    ]


def meaning(
    calls: "Calls",
    file: str,
    func: ast.AST,
    call: ast.Call,
    placed: Optional["Placed"] = None,
    bound: tuple = (),
) -> Meaning:
    """What *call*, made in *func* of *file*, does: read in the module's order.  *placed* is where
    :meth:`calls.Calls.callees` places it, where the caller has it already."""
    copied = round_trip(call)
    if copied is not None:
        return Meaning(PROTOCOL, assumption=PICKLE_ROUND_TRIP, copied_from=copied)
    from bamboo.codemap.calls import BUILTIN as BUILTIN_CLASS
    from bamboo.codemap.calls import UNPLACED_NO_METHOD

    if placed is None:
        try:
            placed = calls.callees(file, func, call, bound)
        except Exception:  # noqa: BLE001 -- a reading that fails says nothing
            return _UNKNOWN
    if placed.targets:
        return Meaning(DEFINITION)  # a method or a function of the corpus alike
    if not isinstance(call.func, ast.Attribute):
        return _UNKNOWN
    name, receiver, types = call.func.attr, call.func.value, calls.types
    if (
        types.module_named(file, receiver) is not None
        or types.class_named(file, receiver) is not None
        or (isinstance(receiver, ast.Name) and receiver.id in types.imports(file))
    ):
        return _UNKNOWN
    if name in LOG_METHODS:
        return Meaning(LOGGER, Method(), "logger_keeps_nothing")
    try:
        library = types.library_classes(file, func, receiver)
    except Exception:  # noqa: BLE001
        library = None
    if library:
        if all(other_system(module) for module, _name in library):
            return Meaning(CLIENT, Method(), "other_systems_keep_nothing")
        return Meaning(LIBRARY, KEEPS_ALL, "library_keeps_what_it_is_handed")
    if placed.unplaced == UNPLACED_NO_METHOD:
        try:
            declared = types.declared(file, func, receiver)
        except Exception:  # noqa: BLE001
            declared = None
        classes = [one for each in declared or () for one in calls.objects_of(each)]
        if classes and all(types.built_outside(*one) for one in classes):
            bases = [types.outside_bases(*one) for one in classes]
            if all(
                based and all(other_system(module) for module, _name in based)
                for based in bases
            ):
                return Meaning(CLIENT, Method(), "other_systems_keep_nothing")
            return Meaning(LIBRARY, KEEPS_ALL, "library_keeps_what_it_is_handed")
        return _UNKNOWN
    literal = _literal_type(receiver)
    if literal is not None:
        method = _of_types(frozenset({literal}), name)
        return Meaning(BUILTIN, method) if method is not None else _UNKNOWN
    try:
        classes, opened = types.candidates(file, func, receiver)
    except Exception:  # noqa: BLE001
        classes, opened = frozenset(), True
    if (
        classes
        and not opened
        and all(where == BUILTIN_CLASS for where, _name in classes)
    ):
        method = _of_types(frozenset(one for _where, one in classes), name)
        return Meaning(BUILTIN, method) if method is not None else _UNKNOWN
    method = by_name(name)
    return (
        Meaning(ASSUMED, method, "builtin_by_name") if method is not None else _UNKNOWN
    )


def _literal_type(expression: ast.expr) -> Optional[str]:
    if isinstance(expression, (ast.List, ast.ListComp)):
        return "list"
    if isinstance(expression, (ast.Set, ast.SetComp)):
        return "set"
    if isinstance(expression, (ast.Dict, ast.DictComp)):
        return "dict"
    return None
