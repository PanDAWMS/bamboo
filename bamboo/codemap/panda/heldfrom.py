"""What a name a guard reads holds, way by way, read back across calls: :class:`GuardFrom` and :class:`ParameterFrom`;
what each write gives a column a guard's way ends at, read the same way: :class:`SetFrom`; and what the object an
arm writes, and the value a passthrough arm hands on, hold: :class:`WriteFrom`.

A reader deciding a junction's arm with a run's values -- the querier's H3, reading ``'reqID' in taskParamMap``
in the rows it has -- needs to know what the name *is*: here the task's ``JEDI_TaskParams.taskParams``, decoded.
:func:`pathcond.held_at` answers within one function; this reads on, one way per binding that can reach the read,
the parts the map already has put together:

- **a local**: each binding :func:`pathcond.held_at` says reaches the read is a way, its value read the same way;
- **a parameter**: the function's own ends the way (kind ``parameter``) and is read once, in a
  :class:`ParameterFrom`: each call :meth:`Calls.callers` places at the function, its argument read in the caller;
- **a call's value**: each return of each function :meth:`Calls.callees` places the call at, the callee's
  parameters read back at the call -- not at every caller of the callee -- and the call on the way (*route*);
- **a call outside the corpus** handed one positional argument (``json.loads(inputStr, object_hook=...)``) is gone
  through and named (*through*), with what its value is of the argument as :mod:`callfacts` says it (decoded, a
  copy, the same value as another type ...) and the corpus's functions a JSON maker is handed to make it with; a
  literal one of the standard functions :func:`evaluate.run` runs raises on there (``json.loads('')``,
  ``int(None)``) goes no further: a stop (``raises``), not a way;
- **an item or attribute**: its root's ways, with the step (*steps*): ``taskParamMap["tgtNumEventsPerJob"]`` is an
  item of the decoded task parameters, not the parameters;
- the ends: the function's own statement's rows (:class:`sqlwrite._Filled`, with the column), asked first of any
  value that is not a name, and of a name a fetch binds where the name is read (the rows are seeded after the
  statement); a literal; a field of ``self`` -- not one the function calls the database on: a loop over
  ``self.cur`` gives the statement's rows; an object a corpus class builds; a module's name; a call not read; a
  value not followed.  Where a way cannot be read on, a stop (:class:`KeyStop`).

A way's end says what the value was, not what was done to it on the way: ``AtlasAnalTaskRefiner.extractCommon``
stores ``taskParamMap["transPath"]`` before it hands the map to the base's guard ``'transPath' in taskParamMap``.
So each function a way passes is asked what may change the object in place there, after the name is bound and
before it is read or handed on (:func:`changes.object_changes`), and the way names each such place (*changes*),
what a store stores read as a way is.

A way is read in the order it is met going back from the read; *route*, *through* and *changes* are kept in the
order the value went, as :attr:`KeySource.via` is.

What a guard read may be written back: a :class:`SetFrom` is read the same way, for each write of a column a
guard's way ends at.  Where the two ways end at one statement's rows and column, the guard's way says what may change
the object after the read and before the write (:attr:`HeldSource.after`): each way's calls down from the function
both are on, the deepest -- in the guard's function from the read and in each caller between from its call, to their
ends; in that function between its two calls -- each change with the writes it can run before.  The object is the
way's: in the guard's function the parameter the way ends at, and the guard's name where its value is the object or
a part of it reached by items and attributes, not one built of its parts; a column's value no way decodes is not
changed in place, and its changes are not read.
"""

from __future__ import annotations

import ast
import re
from typing import Iterable, NamedTuple, Optional

from bamboo.codemap import callfacts, evaluate, pathcond
from bamboo.codemap.astwalk import walk
from bamboo.codemap.calls import (
    Calls,
    CallSite,
    Target,
    _rebinds,
    bind,
    keyed_setattr,
    own_nodes,
    returns,
)
from bamboo.codemap.lowlink import Lowlink
from bamboo.codemap.models import (
    CHANGE_ESCAPED,
    CHANGE_INNER,
    CHANGE_METHOD,
    CHANGE_STORE,
    CHANGE_UNREAD,
    FIELD_ANNOTATED,
    FIELD_ASSIGN,
    FIELD_AUGMENTED,
    FIELD_CLASS,
    FIELD_SETATTR,
    KEY_ATTRIBUTE,
    KEY_CALL,
    KEY_CONSTANT,
    KEY_DATABASE,
    KEY_GLOBAL,
    KEY_OBJECT,
    KEY_PARAMETER,
    KEY_ROWS,
    KEY_SELF_FIELD,
    KEY_SEQUENCE,
    KEY_UNKNOWN,
    THROUGH_ADDED,
    THROUGH_COMPUTED,
    THROUGH_CONVERTED,
    THROUGH_COPY,
    THROUGH_DECODED,
    THROUGH_TEXT,
    THROUGH_UNKNOWN,
    WRITE_OBJECT,
    WRITE_VALUE,
    AfterChange,
    EnteredFrom,
    FieldBinding,
    FieldFrom,
    GuardFrom,
    HeldChange,
    HeldSource,
    HeldThrough,
    JunctionNode,
    KeyStop,
    ParameterFrom,
    ReadSiteNode,
    SetFrom,
    WriteFrom,
)
from bamboo.codemap.panda import changes as changes_mod
from bamboo.codemap.panda import flow, sqltext

_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)
#: The classes ``self`` can be in the function a reading is in, where the call entering it says
#: (:attr:`Target.bound`): ``FactoryBase.getImpl`` entered through ``jobBroker.requireImpl(...)`` is a
#: ``JobBroker``'s, and its ``tmpImpl.refresh()`` runs ``JobBrokerBase.refresh``.  Empty: every class it can be.
_Selves = tuple[tuple[str, ast.ClassDef], ...]


def _selves_key(selves: _Selves) -> tuple[tuple[str, int], ...]:
    """*selves* as a key of what is read once for them."""
    return tuple((file, cls.lineno) for file, cls in selves)


_STOP = "stop"
#: The hooks a JSON maker hands what it parsed: a text the parse cannot read raises with them as without them --
#: ``cls`` parses itself.
_PARSE_HOOKS = callfacts.JSON_HOOKS - {"cls"}


class _Way(NamedTuple):
    """One way's end while it is read: a :data:`HELD_KINDS` kind, or :data:`_STOP` with the KeyStop's kind."""

    kind: str
    name: str = ""
    line: Optional[int] = None
    site_owner: str = ""
    site_line: Optional[int] = None
    paths: tuple[str, ...] = ()
    column: str = ""
    attr: str = ""
    column_unread: str = ""
    stop: str = ""
    what: str = ""
    #: A stop's ``file:line``.
    at: str = ""
    #: As applied to the end: the value is the end's ``steps[0]``, then ``steps[1]``.
    steps: tuple[str, ...] = ()
    #: In the order met going back from the read.
    through: tuple["_Through", ...] = ()
    route: tuple[str, ...] = ()
    #: In the order run.
    changes: tuple["_Change", ...] = ()
    #: Where the end is a container made empty: what it holds (:data:`callfacts.COLLECTION`, :data:`callfacts.MAPPING`).
    made_empty: str = ""
    #: The *paths* are the path carry's (:attr:`KeySource.paths`): the source or a part of it sits there, which one
    #: not known -- not where the reader placed the source itself (:func:`_at`).
    loose: bool = False
    #: The file of the function the end was read in, its *line* among the function's (:func:`_placed_in`): the end
    #: is at ``end_file:line`` (:attr:`HeldSource.at`).  Empty for a parameter, a stop, an end read elsewhere.
    end_file: str = ""


def _placed_in(ways: list[_Way], file: str, func: ast.AST) -> list[_Way]:
    """*ways*, each end not placed yet placed in *file* where its line is among *func*'s: the innermost read that
    made the end places it.  Not a parameter, which no statement binds, nor a stop, nor an end whose line is outside
    *func*.  An end read in another file is placed by the read there: a parameter's default is read in its own
    function (:func:`_where_read`)."""
    first, last = getattr(func, "lineno", None), getattr(func, "end_lineno", None)
    if first is None or last is None:
        return ways
    return [
        one._replace(end_file=file)
        if not one.end_file
        and one.kind not in (_STOP, KEY_PARAMETER)
        and one.line is not None
        and first <= one.line <= last
        else one
        for one in ways
    ]


class _Through(NamedTuple):
    """One :class:`HeldThrough` while it is read, and whether :func:`evaluate.run` runs it as it is called."""

    call: str
    at: str
    gives: str
    hooks: tuple[str, ...] = ()
    runs: bool = False
    #: How many of the way's steps were applied before the call: it is handed the end with ``steps[:steps]``.
    steps: int = 0


class _Change(NamedTuple):
    """One :class:`HeldChange` while it is read: what a store stores, as ways."""

    at: str
    kind: str
    key: str
    what: str
    via: tuple[str, ...]
    value: tuple[_Way, ...]


def _stop(stop: str, name: str = "", what: str = "", at: str = "") -> _Way:
    return _Way(_STOP, name=name, stop=stop, what=what, at=at)


#: A comprehension and the name its value's ``through`` is called by, as Python names its code.
_COMPREHENSIONS = {
    ast.ListComp: "<listcomp>",
    ast.SetComp: "<setcomp>",
    ast.DictComp: "<dictcomp>",
    ast.GeneratorExp: "<genexpr>",
}
_ELEMENT_COMPREHENSIONS = frozenset({"<listcomp>", "<setcomp>", "<genexpr>"})
#: What a ``+``'s value's ``through`` is called by (:data:`THROUGH_ADDED`).
_ADD = "+"


#: A ``with``'s step: its ``__enter__`` mostly gives the context itself, so a way's paths stay as they are.
_ENTER = ".__enter__()"


def _unread_part(way: _Way, step: str) -> _Way:
    """The part *step* of *way*'s value as a value no way reaches: an attribute's (a ``.field`` step) or an unknown
    value, spelled as the way's end with its steps."""
    kind = KEY_ATTRIBUTE if step.startswith(".") else KEY_UNKNOWN
    return _Way(kind, way.name + "".join(way.steps) + step, way.line)


def _step(way: _Way, step: str, element: bool = False, unread: Optional[_Way] = None) -> list[_Way]:
    """*way* with *step* taken: an element of a list, set or generator a comprehension made last is its element
    (``[f for f in job.Files][0]`` is ``job.Files``'s ``[*]``, not ``[*][0]``); a dict's elements are its keys, not
    the values it is read as.  An element of a ``+``'s value is an element of one of its sides, the ``+`` left
    (:func:`_through_added`): ``(ret_head + ret_tail)[0]`` is ``ret_head``'s or ``ret_tail``'s ``[0]``.

    A way whose source sits at parts of the value (*paths*) is cut by the step, as the path carry takes a part: the
    parts under it, with the rest of their steps; none where no part is under it, the source not in the part
    (``self.adder_plugin.result`` holds none of what the plugin holds at ``.job``).  A part the step takes whole is
    the source itself, a way of its own with no paths, where the reader placed it (``[taskSpec.site][0]`` is
    ``taskSpec.site``); where the paths are the carry's (*loose*) it is the source or a part of it, not known which:
    *unread*, else :func:`_unread_part` (``jobSpec.Files + pseudoFiles``, which the carry says holds ``jobSpec`` at
    ``[*]``, has files, not jobs).  *element*: the step is the element a loop takes, a dict's key too."""
    while (
        step.startswith("[")
        and way.through
        and way.through[0].call == _ADD
        and way.through[0].steps == len(way.steps)
    ):
        way = way._replace(through=way.through[1:])
    if (
        step.startswith("[")
        and not way.paths
        and way.through
        and way.through[0].call in _ELEMENT_COMPREHENSIONS
        and way.through[0].steps == len(way.steps)
    ):
        return [way._replace(through=way.through[1:])]
    if not way.paths or step == _ENTER:
        return [way._replace(steps=way.steps + (step,))]
    from bamboo.codemap.panda.recognizers.sqlwrite import paths_under

    under = paths_under(step, way.paths, element)
    parts = tuple(path for path in under if path != "()")
    found = [way._replace(paths=parts)] if parts else []
    if "()" in under:
        found.insert(0, unread or _unread_part(way, step) if way.loose else way._replace(paths=()))
    return found


def _then(way: _Way, steps: tuple[str, ...]) -> list[_Way]:
    """*way* with *steps* taken one after another as :func:`_step` takes each, a stop as it is; where none is left,
    the part no way reaches (:func:`_unread_part`)."""
    ways = [way]
    for step in steps:
        ways = [taken for one in ways for taken in ([one] if one.kind == _STOP else _step(one, step))]
    return ways or [_unread_part(way, "".join(steps))]


def _taken(ways: list[_Way], step: str, element: bool = False, unread: Optional[_Way] = None) -> list[_Way]:
    """*ways* with *step* taken (:func:`_step`) and joined (:func:`_joined`), a stop as it is.  Where the step leaves
    none of them -- every source sits only at other parts -- the part is a value no way reaches: *unread*, else
    :func:`_unread_part` of the first way.  *unread* is given once where several ways land on it."""
    found: list[_Way] = []
    for one in ways:
        for stepped in [one] if one.kind == _STOP else _step(one, step, element, unread):
            if stepped is unread and unread in found:
                continue
            found += [stepped] if stepped.kind == _STOP else _joined(stepped)
    if ways and not found:
        return [unread or _unread_part(ways[0], step)]
    return found


def _onto(outer: _Way, inner: _Way, fields: tuple[str, ...], route: tuple[str, ...]) -> list[_Way]:
    """What a call gives where its callee's way *inner* ends at what the call hands the callee -- *outer*, an
    argument, or the receiver with the *fields* of ``self`` *inner* names: *outer* with *fields* and *inner*'s steps
    taken (:func:`_then`, which cuts a way at parts of the argument), held where *inner* says the callee holds its
    end (*inner*'s paths, before the argument's own: ``return (self.SC_FAILED, inputChunk)`` holds the argument at
    ``[1]``), *inner*'s calls after the steps they come after, *route* its route."""
    steps = fields + inner.steps
    if not outer.paths or outer.kind == _STOP:
        ways = [outer._replace(steps=outer.steps + steps)]
    else:
        ways = _then(outer, steps)
    found = []
    for one in ways:
        if inner.paths and one.kind != _STOP:
            held = tuple(spot + path for spot in inner.paths for path in one.paths) or inner.paths
            one = one._replace(paths=held, loose=inner.loose or one.loose)
        through = tuple(
            call._replace(steps=min(call.steps + len(fields) + len(outer.steps), len(one.steps)))
            for call in inner.through
        )
        found.append(one._replace(through=through + outer.through, route=route, changes=one.changes + inner.changes))
    return found


def _at(spot: str, ways: list[_Way]) -> list[_Way]:
    """*ways*, of a value a container holds at *spot* (``[0]``, ``['k']``, ``[*]``), as the container's: each source
    at *spot*, below it where it sits at parts of the value -- placed by the reader, so a step taking *spot* takes
    the source itself (:func:`_step`); a stop as it is."""
    return [
        one if one.kind == _STOP else one._replace(paths=tuple(spot + path for path in one.paths) or (spot,))
        for one in ways
    ]


#: The changes of a container that may change an element it holds: carried onto the element's ways.
_ELEMENT_CHANGES = frozenset({CHANGE_INNER, CHANGE_ESCAPED, CHANGE_UNREAD})
#: What :func:`callfacts.meaning` reads a call as where a container's method takes out of it.
_TAKING = frozenset({callfacts.LIBRARY, callfacts.BUILTIN, callfacts.ASSUMED})
#: Where what a container's method hands back holds what the container holds (:class:`callfacts.Give`): a view of
#: its values and a copy at each of theirs, a view of its (key, value) pairs at each pair's second, one such pair at
#: its second -- each a value the container holds, none its key.  A view of the keys is not here: what is put into
#: a container is read with its key's spelling (:attr:`HeldChange.key`), not the key's ways.
_HOLDS_ELEMENTS_AT = {
    callfacts.VALUES: "[*]",
    callfacts.SELF: "[*]",
    callfacts.ITEMS: "[*][1]",
    callfacts.ITEM: "[1]",
}


def _puts_at(change, step: str) -> bool:
    """Whether *change* (a :class:`_Change` or a :class:`HeldChange`), of a container, puts a value it reads that
    the element step *step* may take: a method's put or an item's store, under any key or under the step's."""
    if not change.value or change.kind not in (CHANGE_METHOD, CHANGE_STORE) or change.key.startswith("."):
        return False
    return step == "[*]" or change.key == changes_mod.EVERY or step == f"[{change.key}]"


def _puts_unread(change) -> bool:
    """Whether *change* may put into a container what it does not read: the container gone where it is not followed,
    a call not read, a put or an item's store with no value (``xs.append(*ys)``, ``d[k] += 1``)."""
    if change.kind in (CHANGE_ESCAPED, CHANGE_UNREAD):
        return True
    if change.value:
        return False
    return (change.kind == CHANGE_METHOD and change.what in callfacts.PUTTING) or (
        change.kind == CHANGE_STORE and not change.key.startswith(".")
    )


def _joined(way: _Way) -> list[_Way]:
    """*way*, where its end is a container made empty and its first step takes an element: the ways of the values
    its changes put that the step may take, with the rest of the steps -- each through the calls to its put
    (*via*), carrying what may change an element in the container --; *way* too where a put is not read or the step
    may take a mapping's key (``[*]``).  ``TaskBuffer.peekJobs``' ``retJobs = []`` filled by ``retJobs.append(res)``:
    ``retJobs[0]`` is ``res``.  Any other way, itself."""
    if way.kind == _STOP or not way.made_empty or way.through or not way.steps or not way.steps[0].startswith("["):
        return [way]
    step, rest = way.steps[0], way.steps[1:]
    puts = [one for one in way.changes if _puts_at(one, step)]
    if not puts:
        return [way]
    kept = tuple(one for one in way.changes if one.kind in _ELEMENT_CHANGES)
    found = [
        stepped
        for one in puts
        for value in one.value
        for stepped in (
            [value]
            if value.kind == _STOP
            else _then(value._replace(route=way.route + one.via + value.route, changes=value.changes + kept), rest)
        )
    ]
    if any(_puts_unread(one) for one in way.changes) or (way.made_empty == callfacts.MAPPING and step == "[*]"):
        found.append(way)
    return found


def held_elements(source: HeldSource, step: str) -> tuple[list[HeldSource], bool]:
    """:func:`_joined` on a record's way, for a step taken past its end (a FieldFrom's or ParameterFrom's way, read on
    with the steps still to take): where *source* ends at a container made empty and is the container whole, the
    ways of the values its changes put that the element step *step* may take -- each with the calls to its put and
    *source*'s route after its own, carrying what may change an element in the container -- and whether *source*
    stays a way too (a put not read, a mapping's ``[*]``).  ``([], True)`` for any other source."""
    if not source.made_empty or source.steps or source.through or not step.startswith("["):
        return [], True
    puts = [one for one in source.changes if _puts_at(one, step)]
    kept = [one for one in source.changes if one.kind in _ELEMENT_CHANGES]
    values = [
        value.model_copy(
            update={"route": [*value.route, *reversed(one.via), *source.route], "changes": [*value.changes, *kept]}
        )
        for one in puts
        for value in one.value
    ]
    stays = (
        not puts
        or any(_puts_unread(one) for one in source.changes)
        or any(one.value_stops for one in puts)
        or (source.made_empty == callfacts.MAPPING and step == "[*]")
    )
    return values, stays


def _empty_literal(expr: ast.expr) -> str:
    """What a literal making an empty container holds: ``[]`` a collection, ``{}`` a mapping; ``""`` for another."""
    if isinstance(expr, (ast.List, ast.Set)) and not expr.elts:
        return callfacts.COLLECTION
    if isinstance(expr, ast.Dict) and not expr.keys:
        return callfacts.MAPPING
    return ""


def _where_read(
    given: ast.expr, call: ast.AST, file: str, func: ast.AST, target_file: str, target: ast.AST
) -> tuple[ast.AST, str, ast.AST]:
    """Where to read *given*, what *call* in *func* of *file* binds a parameter of *target* (of *target_file*) to
    (:func:`calls.bind`), as (at, file, function): at the call, else -- one of *target*'s defaults, the call handing
    the parameter nothing -- where it is written, in *target*, its line there (``use_json: bool = False`` at
    ``TaskBuffer.py:1015``, not a line of the caller's file)."""
    args = getattr(target, "args", None)
    if args is not None and any(given is one for one in [*args.defaults, *args.kw_defaults]):
        return given, target_file, target
    return call, file, func


def _added_sides(expr: ast.expr) -> list[ast.expr]:
    """The operands of *expr*'s ``+`` -- ``a + b + c``'s three --, *expr* itself where it is no ``+``."""
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
        return _added_sides(expr.left) + _added_sides(expr.right)
    return [expr]


def _through_added(ways: list[_Way], place: str) -> list[_Way]:
    """*ways*, of a side of the ``+`` at *place*, as the ``+``'s value's: each through the ``+``
    (:data:`THROUGH_ADDED`) after its own steps, a stop as it is."""
    made = _Through(_ADD, place, THROUGH_ADDED)
    return [
        one if one.kind == _STOP else one._replace(through=(made._replace(steps=len(one.steps)),) + one.through)
        for one in ways
    ]


def _key_spot(key: ast.expr) -> str:
    """Where a dict literal puts the value under *key*, as a subscript reads it: ``['k']`` or ``[0]`` under a text
    or an integer literal, else ``[*]``."""
    if isinstance(key, ast.Constant) and isinstance(key.value, (str, int)) and not isinstance(key.value, bool):
        return f"[{key.value!r}]"
    return "[*]"


def owner_of(file: str, func: ast.AST) -> str:
    return f"{file}::{getattr(func, 'name', '<module>')}"


def _class_key(file: str, cls: ast.ClassDef) -> str:
    """``file::Class`` -- ``Outer.Inner`` for a nested class, where the tree has parent links."""
    try:
        name = pathcond.qualified_name(cls)
    except Exception:  # noqa: BLE001 -- a tree without parent links names the class alone
        name = cls.name
    return f"{file}::{name or cls.name}"


def _method_class(func: ast.AST) -> Optional[ast.ClassDef]:
    """The class whose method *func* is or is nested in -- the class its ``self`` is of -- by parent links."""
    method = func
    here = getattr(func, "parent", None)
    while here is not None:
        if isinstance(here, ast.ClassDef):
            return here if any(one is method for one in here.body) else None
        if isinstance(here, _FUNCTIONS):
            method = here
        here = getattr(here, "parent", None)
    return None


#: A step that is a field: ``.status_code``, not ``[0]`` nor ``.__enter__()``.
FIELD_STEP = re.compile(r"^\.([A-Za-z_]\w*)$")
#: How many steps past a field a way's end is read on for the FieldFroms further down (``self.a.b.c``).  A record is
#: made once a key; the steps only choose the key asked next.  The asks still need a cap: some grow with no period
#: (``storeJobs``' ``jobs`` takes ``[0][0].PandaID``, ``[0][0].jobsetID``, ... one more each round).  Raised to 24,
#: with a repeated round stopped, the corpus's records came out the same as at 6.
_FIELD_REST = 6


def end_path(source: HeldSource) -> Optional[tuple[str, ...]]:
    """The steps from where *source* ends to the value, from its ``self`` or object: ``(".result", ".x")`` for
    ``self.result`` with the step ``.x``; ``None`` for another end.  An end that may be only a part of the value
    (*paths*: ``self.job`` at ``.job`` of the plugin ``AdderAtlasPlugin(self.job)`` builds) is read as the whole --
    the record keeps no ``()`` beside a part's path to tell them apart --: a FieldFrom too many, never one short."""
    if source.kind == KEY_SELF_FIELD:
        parts = source.name.split(".")
        if parts[0] != "self":
            return None
        return tuple(f".{part}" for part in parts[1:]) + tuple(source.steps)
    if source.kind == KEY_OBJECT:
        return tuple(source.steps)
    return None


def _first_positional(func: ast.AST) -> Optional[str]:
    arguments = getattr(func, "args", None)
    if arguments is None:
        return None
    first = (arguments.posonlyargs + arguments.args)[:1]
    return first[0].arg if first else None


def _is_field(node: ast.AST, me: str, field: str) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == field
        and isinstance(node.value, ast.Name)
        and node.value.id == me
    )


def _with_functions(func: ast.AST) -> Iterable[tuple[ast.AST, ast.AST]]:
    """Each node of *func* with the innermost definition it is in -- not into a nested class, whose methods have
    their own ``self``."""
    todo: list[tuple[ast.AST, ast.AST]] = [(child, func) for child in ast.iter_child_nodes(func)]
    while todo:
        node, inner = todo.pop()
        yield node, inner
        if isinstance(node, ast.ClassDef):
            continue
        onward = node if isinstance(node, _FUNCTIONS) else inner
        todo.extend((child, onward) for child in ast.iter_child_nodes(node))


def _parameters(func: ast.AST) -> set[str]:
    arguments = getattr(func, "args", None)
    if arguments is None:
        return set()
    names = {
        one.arg
        for one in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs)
    }
    names |= {one.arg for one in (arguments.vararg, arguments.kwarg) if one is not None}
    return names - {"self", "cls"}


def _bound_here(func: ast.AST, name: str) -> bool:
    return any(not binding.nested for binding in pathcond.bindings(func, name))


#: A binding :func:`pathcond.bindings` does not list, by what makes it.
_UNREAD_FORMS = {ast.alias: "an import", ast.FunctionDef: "a def", ast.AsyncFunctionDef: "a def", ast.ClassDef: "a class"}


def _declared_or_unread(func: ast.AST) -> tuple[frozenset[str], dict[str, str]]:
    """The names *func* declares ``global`` or ``nonlocal``, and those it binds by a form
    :func:`pathcond.bindings` does not list -- an import, a nested ``def`` or ``class``, a ``match`` capture --, each
    with its form; *func*'s own statements only, not a nested scope's.  Read once per function and kept on it."""
    kept: Optional[tuple[frozenset[str], dict[str, str]]] = getattr(func, "_heldfrom_declared_or_unread", None)
    if kept is not None:
        return kept
    declared: set[str] = set()
    unread: dict[str, str] = {}
    stack = list(ast.iter_child_nodes(func))
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            declared.update(node.names)
        elif isinstance(node, ast.alias):
            unread.setdefault(node.asname or node.name.split(".")[0], _UNREAD_FORMS[ast.alias])
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            unread.setdefault(node.name, "a match capture")
        elif isinstance(node, ast.MatchMapping) and node.rest:
            unread.setdefault(node.rest, "a match capture")
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            unread.setdefault(node.name, _UNREAD_FORMS[type(node)])
            stack.extend([*node.decorator_list, *getattr(node, "bases", ()), *getattr(node, "keywords", ())])
            continue
        if isinstance(node, ast.Lambda):
            continue
        stack.extend(ast.iter_child_nodes(node))
    held = (frozenset(declared), unread)
    func._heldfrom_declared_or_unread = held  # type: ignore[attr-defined]
    return held


def _binds(func: ast.AST, name: str) -> bool:
    """Whether *func*'s own statements bind *name* by any form -- an import, a ``def``, a ``match`` capture too --,
    which makes it *func*'s own by Python's rule, or, declared ``global`` or ``nonlocal``, read by those bindings
    (:meth:`_Reader._bound`).  A name it binds by none is a module's."""
    return _bound_here(func, name) or name in _declared_or_unread(func)[1]


def _index(expr: ast.expr) -> Optional[int]:
    """A subscript's constant integer index, else ``None``."""
    if isinstance(expr, ast.Subscript) and isinstance(expr.slice, ast.Constant):
        value = expr.slice.value
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _field_of(value: ast.expr, field: Optional[str]) -> ast.expr:
    """*value*'s *field* (``value.field``), a node standing where *value* does; *value* itself with no *field*."""
    if not field:
        return value
    node = ast.Attribute(value=value, attr=field, ctx=ast.Load())
    ast.copy_location(node, value)
    node.parent = getattr(value, "parent", None)  # type: ignore[attr-defined]
    return node


def _literal_of(site: ast.AST, name: str) -> Optional[ast.expr]:
    """The tuple or list literal *site*, an ``=`` to *name* alone, gives it; else ``None``."""
    targets = pathcond.targets_of(site)
    value = getattr(site, "value", None)
    if len(targets) == 1 and isinstance(targets[0], ast.Name) and targets[0].id == name:
        return value if isinstance(value, (ast.Tuple, ast.List)) else None
    return None


def _literal_element(literal: Optional[ast.expr], index: int) -> Optional[ast.expr]:
    """*literal*'s element at *index*, where no starred element puts it out of place; else ``None``."""
    if not isinstance(literal, (ast.Tuple, ast.List)) or any(isinstance(one, ast.Starred) for one in literal.elts):
        return None
    if -len(literal.elts) <= index < len(literal.elts):
        return literal.elts[index]
    return None


class _Reader:
    """The reading, over one build's calls, reach record and read sites; each function's returns and each
    parameter read once."""

    def __init__(
        self,
        calls: Calls,
        record,
        sites: Iterable[ReadSiteNode],
        owners: set[str],
        trees: dict[str, ast.AST],
    ):
        from bamboo.codemap.panda.recognizers import sqlwrite

        self.calls, self.record, self.trees, self._sqlwrite = (
            calls,
            record,
            trees,
            sqlwrite,
        )
        self.reading = sqlwrite._Reading(calls, owners, record)
        self.sites_of: dict[str, list[ReadSiteNode]] = {}
        for site in sites:
            self.sites_of.setdefault(site.owner, []).append(site)
        self._filled: dict[int, object] = {}
        self._cursor_fields: dict[int, set[str]] = {}
        #: What a callee returns or yields, by the function and the classes ``self`` is there (:data:`_Selves`).
        self._returns: dict[tuple[int, tuple, Optional[str]], list[_Way]] = {}
        self._returns_at: dict[tuple[int, tuple, int, Optional[str]], list[_Way]] = {}
        self._yields: dict[tuple[int, tuple], list[_Way]] = {}
        self._generators: dict[int, bool] = {}
        self._active: set[tuple[int, int]] = set()
        #: ``(owner, def line, parameter)`` -> its ParameterFrom; the ones a record's way ends at, not yet read;
        #: and the function of each a way has ended at -- a callee's, filled at its call, ends in no record.
        self.parameters: dict[tuple[str, int, str], ParameterFrom] = {}
        self._pending: dict[tuple[str, int, str], tuple[str, ast.AST]] = {}
        self._functions: dict[tuple[str, int, str], tuple[str, ast.AST]] = {}
        #: ``(function, name, binding, statement read at)`` -> the changes between, by ids of the build's trees.
        self._hops: dict[tuple, tuple[_Change, ...]] = {}
        #: The keys of *_hops* still being read: what is read under one is kept once it is done.
        self._lowlink = Lowlink()
        #: ``(parameter, file:line)`` -> the call there handing it, read once (:meth:`hop`).
        self._hop_at: dict[tuple, Optional[_Hop]] = {}
        #: ``(class, field)`` -> its FieldFrom; what is asked of the FieldFroms and ParameterFroms with the steps past
        #: their ends still to be read on (:meth:`_ask`), and what was asked.
        self.fields: dict[tuple[str, str], FieldFrom] = {}
        self._asked: dict[tuple[str, tuple, tuple[str, ...]], None] = {}
        self._asks: list[tuple[str, tuple, tuple[str, ...]]] = []
        self._owners: dict[int, str] = {}
        self._classes: Optional[dict[str, tuple[str, ast.ClassDef]]] = None
        self._stored: Optional[dict[str, list[tuple[str, str]]]] = None

    def filled(self, file: str, func: ast.AST):
        held = self._filled.get(id(func))
        if held is None:
            first, last = func.lineno, func.end_lineno or func.lineno  # type: ignore[attr-defined]
            here = [
                site
                for site in self.sites_of.get(owner_of(file, func), ())
                if first <= site.anchor.line_start <= last
            ]
            assert isinstance(func, _FUNCTIONS)
            held = self._filled[id(func)] = self._sqlwrite._Filled(
                self.reading, file, func, here
            )
        return held

    def class_owner(self, file: str, func: ast.AST) -> str:
        """The class *func*'s ``self`` is of, ``file::Class``; empty where *func* is no method's."""
        held = self._owners.get(id(func))
        if held is None:
            cls = _method_class(func)
            if cls is None and getattr(func, "parent", None) is None and file in self.trees:
                cls = pathcond.class_of(self.trees[file], func)
            held = self._owners[id(func)] = _class_key(file, cls) if cls is not None else ""
        return held

    # -- the ends ----------------------------------------------------------

    def _parameter(
        self, file: str, func: ast.AST, name: str, paths: tuple[str, ...] = ()
    ) -> _Way:
        key = (owner_of(file, func), func.lineno, name)  # type: ignore[attr-defined]
        self._functions.setdefault(key, (file, func))
        return _Way(
            KEY_PARAMETER,
            name=name,
            line=func.lineno,  # type: ignore[attr-defined]
            site_owner=key[0],
            site_line=key[1],
            paths=paths,
        )

    def _sources(
        self, expr: ast.expr, at: ast.AST, file: str, func: ast.AST
    ) -> list[_Way]:
        """What :class:`sqlwrite._Filled` says *expr* holds where *at* runs -- but a field the function calls the
        database on: what a loop over ``self.cur`` gives is the statement's rows, and its ``rowcount`` the
        statement's count, not a field a run's object holds (key sources read them as the field).

        The carry's paths say where the source or a part of it sits (*loose*), but a statement's rows' and a
        sequence's, a part of which is named by its *column*.  The field *expr* reads is the field itself: what the
        carry says sits at its parts is parts of it, which its steps take (``self.siteSpecList`` read as itself at
        ``[*]`` is ``self.siteSpecList[*]``)."""
        found = []
        cursors = self._cursors(file, func)
        read = ast.unparse(expr) if isinstance(expr, ast.Attribute) else ""
        for source in self.filled(file, func).sources(expr, at):
            if source.kind == KEY_SELF_FIELD and source.name in cursors:
                continue
            parts = tuple(path for path in source.paths if path != "()")
            if source.kind == KEY_SELF_FIELD and source.name == read:
                parts = ()
            loose = source.kind not in (KEY_ROWS, KEY_SEQUENCE)
            # The value itself and parts of it too: two ways, the whole (no paths) and the parts -- a way with paths
            # sits at those parts only.
            for paths in ([()] if "()" in source.paths or not parts else []) + ([parts] if parts else []):
                if source.kind == KEY_PARAMETER:
                    found.append(self._parameter(file, func, source.name, paths)._replace(loose=bool(paths) and loose))
                    continue
                found.append(
                    _Way(
                        source.kind,
                        name=source.name,
                        line=source.line,
                        site_owner=(
                            self.class_owner(file, func) if source.kind == KEY_SELF_FIELD else source.site_owner
                        ),
                        site_line=source.site_line,
                        paths=paths,
                        column=source.column,
                        attr=source.attr,
                        column_unread=source.column_unread,
                        loose=bool(paths) and loose,
                    )
                )
        return found

    def _cursors(self, file: str, func: ast.AST) -> set[str]:
        held = self._cursor_fields.get(id(func))
        if held is None:
            held = self._cursor_fields[id(func)] = {
                f"self.{node.func.value.attr}"
                for node in walk(func)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Attribute)
                and isinstance(node.func.value.value, ast.Name)
                and node.func.value.value.id == "self"
                and self.record.is_database(file, node)
            }
        return held

    def _rows(
        self, expr: ast.expr, at: ast.AST, file: str, func: ast.AST
    ) -> list[_Way]:
        return [
            one
            for one in self._sources(expr, at, file, func)
            if one.kind in (KEY_ROWS, KEY_SEQUENCE)
        ]

    # -- the reading ---------------------------------------------------------

    def read(self, expr: ast.expr, at: ast.AST, file: str, func: ast.AST, selves: _Selves = ()) -> list[_Way]:
        """The ways of *expr*, read where *at* runs in *func*; one ending at *func*'s own parameter is kind
        ``parameter``.  Each end made here is placed in *file* (:func:`_placed_in`).  *selves*: the classes ``self``
        can be in *func*, as the call entering it placed it (:data:`_Selves`); a call there is placed, and what may
        change a name's object in place is read, for them."""
        return _placed_in(self._ways_of(expr, at, file, func, selves=selves), file, func)

    def _ways_of(self, expr: ast.expr, at: ast.AST, file: str, func: ast.AST, selves: _Selves = ()) -> list[_Way]:
        if isinstance(expr, ast.Constant):
            return [
                _Way(
                    KEY_CONSTANT,
                    name=repr(expr.value),
                    line=getattr(expr, "lineno", None),
                )
            ]
        if not isinstance(func, _FUNCTIONS):
            return [
                _Way(
                    KEY_UNKNOWN,
                    name=ast.unparse(expr),
                    line=getattr(expr, "lineno", None),
                )
            ]
        mark = (id(func), id(expr))
        if mark in self._active:
            return [
                _stop(
                    "cycle",
                    ast.unparse(expr),
                    at=f"{file}:{getattr(expr, 'lineno', '')}",
                )
            ]
        self._active.add(mark)
        try:
            return self._read(expr, at, file, func, selves=selves)
        finally:
            self._active.discard(mark)

    def _read(
        self, expr: ast.expr, at: ast.AST, file: str, func: ast.AST, selves: _Selves = ()
    ) -> list[_Way]:
        line = getattr(expr, "lineno", None)
        if isinstance(expr, ast.Name):
            return self._name(expr, at, file, func, selves=selves)
        # A literal is read by its elements before the carry is asked whether it is rows: ``return retVal, retMsg``,
        # whose ``retMsg`` holds a row's value, holds ``retVal`` too.
        placed = self._placed(expr, at, file, func, selves=selves)
        if placed is not None:
            return placed
        rows = self._rows(expr, at, file, func)
        if rows:
            return rows
        if isinstance(expr, ast.Call):
            return self._call(expr, file, func, selves=selves)
        if isinstance(expr, ast.Subscript) and isinstance(expr.slice, ast.Slice):
            # A slice of a list or a text is one of its kind holding what it holds, not an element: the value itself.
            return self.read(expr.value, at, file, func, selves=selves)
        if isinstance(expr, (ast.Subscript, ast.Attribute)) and not (
            isinstance(expr.value, ast.Name) and expr.value.id in ("self", "cls")
        ):
            position = _index(expr)
            if position is not None:
                chosen = self._element(expr.value, position, at, file, func, selves=selves)
                if chosen is not None:
                    return chosen
            if isinstance(expr, ast.Attribute):
                returned = self._returned_field(expr, at, file, func, selves=selves)
                if returned is not None:
                    return returned
            if isinstance(expr, ast.Subscript):
                index = expr.slice
                step = (
                    f"[{index.value!r}]" if isinstance(index, ast.Constant) else "[*]"
                )
            else:
                step = f".{expr.attr}"
            unread = _Way(KEY_ATTRIBUTE if isinstance(expr, ast.Attribute) else KEY_UNKNOWN, ast.unparse(expr), line)
            return _taken(self.read(expr.value, at, file, func, selves=selves), step, unread=unread)
        if isinstance(expr, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            # A new collection of the element's values: the element's ways, through the comprehension.
            element = expr.value if isinstance(expr, ast.DictComp) else expr.elt
            made = _Through(_COMPREHENSIONS[type(expr)], f"{file}:{line}", THROUGH_COMPUTED)
            return [
                one
                if one.kind == _STOP
                else one._replace(through=(made._replace(steps=len(one.steps)),) + one.through)
                for one in self.read(element, element, file, func, selves=selves)
            ]
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
            return self._added(_added_sides(expr), at, f"{file}:{line}", file, func, selves=selves)
        found = self._sources(expr, at, file, func)
        empty = _empty_literal(expr)
        if empty:
            found = [one._replace(made_empty=empty) if one.kind == KEY_UNKNOWN else one for one in found]
        return found or [
            _Way(
                KEY_ATTRIBUTE if isinstance(expr, ast.Attribute) else KEY_UNKNOWN,
                ast.unparse(expr),
                line,
                made_empty=empty,
            )
        ]

    def _placed(
        self, expr: ast.expr, at: ast.AST, file: str, func: ast.AST, selves: _Selves = ()
    ) -> Optional[list[_Way]]:
        """The ways of a literal holding values -- a list's, a tuple's or a set's elements, a dict's values and keys,
        no ``*`` or ``**`` among them -- or of an ``if`` expression, read here, not asked of the carry, which says
        only where a source or a part of it sits: each element's ways at its place (:func:`_at`) -- ``[i]`` of a list
        or a tuple, ``[*]`` of a set, a dict's value at ``['k']`` (:func:`_key_spot`) and its key at ``[key]`` --;
        either branch's of an ``if`` expression.  ``[taskSpec.site]`` holds ``taskSpec.site``, not ``taskSpec``.
        ``None`` for another expression, an empty literal too (:func:`_empty_literal`)."""
        if isinstance(expr, ast.IfExp):
            return self.read(expr.body, at, file, func, selves=selves) + self.read(
                expr.orelse, at, file, func, selves=selves
            )
        if isinstance(expr, (ast.List, ast.Tuple, ast.Set)):
            if not expr.elts or any(isinstance(one, ast.Starred) for one in expr.elts):
                return None
            spots = [("[*]" if isinstance(expr, ast.Set) else f"[{index}]", one) for index, one in enumerate(expr.elts)]
        elif isinstance(expr, ast.Dict):
            if not expr.keys or any(key is None for key in expr.keys):
                return None
            from bamboo.codemap.panda.recognizers.sqlwrite import KEY_STEP

            spots = []
            for key, value in zip(expr.keys, expr.values, strict=True):
                assert key is not None
                spots += [(_key_spot(key), value), (KEY_STEP, key)]
        else:
            return None
        return [placed for spot, one in spots for placed in _at(spot, self.read(one, at, file, func, selves=selves))]

    def _added(
        self, sides: Iterable[ast.expr], at: ast.AST, place: str, file: str, func: ast.AST, selves: _Selves = ()
    ) -> list[_Way]:
        """The ways of the ``+`` at *place* of *sides* (``a + b``; the ``b`` of an ``x += b``), read here, not asked
        of the carry: each side's, through the ``+`` (:func:`_through_added`) -- a number or a text computed from
        them, or a sequence joined of them, whose element is a side's (:func:`_step`).  ``ret_head + ret_tail`` holds
        the rows both hold; ``job.attemptNr + 1`` is computed from ``job.attemptNr`` and ``1``."""
        return _through_added([one for side in sides for one in self.read(side, at, file, func, selves=selves)], place)

    def _accumulated(
        self, expr: ast.Name, site: ast.AugAssign, file: str, func: ast.AST, seen: set[int], selves: _Selves = ()
    ) -> list[_Way]:
        """What ``x += b`` (*site*) leaves *expr*: a ``+`` (:meth:`_added`) of what ``x`` held before and ``b`` -- a
        ``+=`` reaching it adds its own ``b``, each once (``scanSiteList += ...`` in a loop)."""
        seen.add(id(site))
        place = f"{file}:{site.lineno}"
        found = self._added((site.value,), site, place, file, func, selves=selves)
        for before in pathcond.held_at(func, expr.id, site).sites:  # type: ignore[arg-type]
            if id(before) in seen:
                continue
            if (
                isinstance(before, ast.AugAssign)
                and isinstance(before.op, ast.Add)
                and isinstance(before.target, ast.Name)
                and before.target.id == expr.id
            ):
                found += self._accumulated(expr, before, file, func, seen, selves=selves)
                continue
            found += _through_added(self._site_ways(expr, site, before, file, func, site, selves=selves), place)
        return found

    def _returned_field(
        self, expr: ast.Attribute, at: ast.AST, file: str, func: ast.AST, selves: _Selves = ()
    ) -> Optional[list[_Way]]:
        """The ways of a field of what a call of the corpus returns, read in the callee at its returns as that field
        -- a field the callee filled from a row (``taskSpec.pack(res)``) is the row's column --: of the call, of its
        element at a constant index, or of a name every reaching binding of which gives it one of these
        (``tmpStat, taskSpec = self.taskBufferIF.getTaskWithID_JEDI(jediTaskID)``); ``None`` else, for the field's
        step to be taken on the value's ways."""
        value = expr.value
        called = self._called_at(value, file, func, selves=selves)
        if called is not None:
            return self._call(called[0], file, func, called[1], field=expr.attr, selves=selves)
        if not isinstance(value, ast.Name) or value.id in ("self", "cls") or not _bound_here(func, value.id):
            return None
        point = value if getattr(value, "parent", None) is not None else at
        bound: list[tuple[ast.AST, tuple[ast.Call, Optional[int]]]] = []
        for site in pathcond.held_at(func, value.id, point).sites:  # type: ignore[arg-type]
            if not isinstance(site, (ast.Assign, ast.AnnAssign)):
                return None
            given = next(
                (
                    held
                    for target, held in pathcond.assigned_targets(site)
                    if isinstance(target, ast.Name) and target.id == value.id
                ),
                None,
            )
            called = self._called_at(given, file, func, selves=selves) if given is not None else None
            if called is None:
                return None
            bound.append((site, called))
        if not bound:
            return None
        last = _statement_of(point)
        found: list[_Way] = []
        for site, (call, index) in bound:
            ways = self._call(call, file, func, index, field=expr.attr, selves=selves)
            found += self._with_changes(value, site, ways, file, func, last, selves=selves)
        return found

    def _called_at(
        self, expr: ast.expr, file: str, func: ast.AST, selves: _Selves = ()
    ) -> Optional[tuple[ast.Call, Optional[int]]]:
        """*expr* as a call of the corpus that returns (not the database's, not a generator's), with the constant
        index of its element *expr* takes, if any; else ``None``."""
        index = _index(expr)
        call = expr.value if index is not None else expr  # type: ignore[attr-defined]
        if not isinstance(call, ast.Call):
            return None
        if self.record is not None and self.record.is_database(file, call):
            return None
        placed = self.calls.callees(file, func, call, bound=selves)
        if not placed.targets or placed.thread or any(self._is_generator(one) for one in placed.targets):
            return None
        return call, index

    def _element(
        self,
        value: ast.expr,
        index: int,
        at: ast.AST,
        file: str,
        func: ast.AST,
        field: Optional[str] = None,
        selves: _Selves = (),
    ) -> Optional[list[_Way]]:
        """The ways of *value*'s element at a constant *index* where a tuple or list literal gives it: the literal's
        own, a name's binding to one (each binding: the others' ways with the step ``[index]``), a call's returns
        (each return: the same); ``None`` where nothing does, for the caller to add the step.  ``retEvS, retNewPandaID
        = self.ppEventServiceJob(...)`` reads the call's returned ``(9, jobSpec.PandaID)``'s ``jobSpec.PandaID``, not
        the tuple.  With *field*, the ways of that field of the element (:meth:`_returned_field`)."""
        if isinstance(value, (ast.Tuple, ast.List)):
            element = _literal_element(value, index)
            return self.read(_field_of(element, field), at, file, func, selves=selves) if element is not None else None
        if isinstance(value, ast.Call):
            placed = self.calls.callees(file, func, value, bound=selves)
            if not placed.targets or placed.thread:
                return None
            return self._call(value, file, func, index, field=field, selves=selves)
        if not isinstance(value, ast.Name) or value.id in ("self", "cls") or not _bound_here(func, value.id):
            return None
        point = value if getattr(value, "parent", None) is not None else at
        sites = pathcond.held_at(func, value.id, point).sites  # type: ignore[arg-type]
        if not any(_literal_element(_literal_of(site, value.id), index) is not None for site in sites):
            return None
        found: list[_Way] = []
        last = _statement_of(point)
        for site in sites:
            element = _literal_element(_literal_of(site, value.id), index)
            if element is not None:
                found += self.read(_field_of(element, field), site, file, func, selves=selves)
            else:
                ways = self._stepped(
                    self._site_ways(value, point, site, file, func, last, selves=selves), f"[{index!r}]"
                )
                found += self._of_field(ways, field)
        return found

    def _name(
        self, expr: ast.Name, at: ast.AST, file: str, func: ast.AST, selves: _Selves = ()
    ) -> list[_Way]:
        if expr.id in ("self", "cls"):
            return [_Way(KEY_SELF_FIELD, name=expr.id, line=expr.lineno, site_owner=self.class_owner(file, func))]
        point = expr if getattr(expr, "parent", None) is not None else at
        if (
            expr.id not in _parameters(func)
            and not _binds(func, expr.id)
            and pathcond.comprehension_clause(func, expr.id, point) is None
        ):
            return [_Way(KEY_GLOBAL, name=expr.id, line=expr.lineno)]
        last = _statement_of(point)
        if any(one.kind == pathcond.BIND_WALRUS and not one.nested for one in pathcond.bindings(func, expr.id)):
            walrus = _walrus_before(expr.id, point, last)
            if walrus is not None:
                return self._site_ways(expr, point, walrus, file, func, last, selves=selves)
        found: list[_Way] = []
        for site in pathcond.held_at(func, expr.id, point).sites:  # type: ignore[arg-type]
            found += self._site_ways(expr, point, site, file, func, last, selves=selves)
        return found

    def _site_ways(
        self, expr: ast.Name, point: ast.AST, site: ast.AST, file: str, func: ast.AST, last: Optional[ast.stmt],
        selves: _Selves = (),
    ) -> list[_Way]:
        """The ways *site*, a binding reaching the read at *point*, gives *expr*, with what may change the object
        in place between."""
        return self._with_changes(
            expr, site, self._bound(expr, point, site, file, func, selves=selves), file, func, last, selves=selves
        )

    def _with_changes(
        self, expr: ast.Name, site: ast.AST, ways: list[_Way], file: str, func: ast.AST, last: Optional[ast.stmt],
        selves: _Selves = (),
    ) -> list[_Way]:
        """*ways*, what *site* gives *expr*, with what may change the object in place between it and the read."""
        changed: tuple[_Change, ...] = ()
        if not _unchanging(func, expr.id, site, ways):
            changed = self._changed(file, func, expr.id, site, last, selves=selves)
        return [one._replace(changes=one.changes + changed) if changed and one.kind != _STOP else one for one in ways]

    def _bound(
        self, expr: ast.Name, point: ast.AST, site: ast.AST, file: str, func: ast.AST, selves: _Selves = ()
    ) -> list[_Way]:
        """The ways of what *site*, a binding reaching the read, gives *expr*.  The function's entry, reaching the
        read of a local not a parameter, gives none: that way raises ``UnboundLocalError`` or is never taken (a
        test the binding's own test rules out, a statement before it that cannot raise) -- no value comes by it
        (``return retVal`` in the handler of a ``try`` whose first statement binds ``retVal``).  A name the
        function declares ``global`` or ``nonlocal`` is the module's by that way, and a binding of the function that
        reaches the read gives its own value, no other writer being read between (``global_task_buffer =
        task_buffer`` read on the next statement).  Where the function binds the name by a form not read
        (:func:`_declared_or_unread`), a stop ``unread_binding``."""
        if site is func.args:  # type: ignore[attr-defined]
            if expr.id in _parameters(func):
                return [self._parameter(file, func, expr.id)]
            declared, unread = _declared_or_unread(func)
            if expr.id in declared:
                return [_Way(KEY_GLOBAL, name=expr.id, line=expr.lineno)]
            form = unread.get(expr.id)
            if form is not None:
                return [_stop("unread_binding", expr.id, what=form, at=f"{file}:{expr.lineno}")]
            return []
        targets = pathcond.targets_of(site)
        value = getattr(site, "value", None)
        if (
            isinstance(site, (ast.Assign, ast.AnnAssign))
            and len(targets) == 1
            and isinstance(targets[0], ast.Name)
            and targets[0].id == expr.id
            and value is not None
        ):
            if (
                isinstance(value, ast.Call)
                and self.record is not None
                and self.record.is_database(file, value)
            ):
                # A fetch: its rows are seeded after the statement, so they are asked where the name is read.
                return self._rows(expr, point, file, func) or [
                    _Way(KEY_DATABASE, ast.unparse(value.func), value.lineno)
                ]
            return self.read(value, site, file, func, selves=selves)
        return self._carried(expr, point, site, file, func, selves=selves)

    def _caught(self, handler: ast.ExceptHandler, file: str) -> list[_Way]:
        """The ways of the name an ``except X as e`` binds: an object of X, as a call building X is
        (:meth:`_call`) -- of each class a tuple names --, so a field of it is read where the class sets it
        (``e.fatal`` from FileRegistrationError's ``self.fatal = fatal``, from each raise's ``fatal=``).  Placed at
        the handler: the raise is not the handler's to name.  Empty where a class caught is not the corpus's
        (``except Exception as e``): what the carry says then.  A subclass's object caught is X's too: X's
        :class:`~bamboo.codemap.models.FieldFrom` reads its corpus subclasses' bindings."""
        caught = handler.type
        named = caught.elts if isinstance(caught, ast.Tuple) else [caught] if caught is not None else []
        found: list[_Way] = []
        for one in named:
            cls = self.calls.types.class_named(file, one)
            if cls is None:
                return []
            found.append(_Way(KEY_OBJECT, cls[1].name, handler.lineno, site_owner=_class_key(*cls)))
        return found

    def _carried(
        self, expr: ast.Name, point: ast.AST, site: ast.AST, file: str, func: ast.AST, selves: _Selves = ()
    ) -> list[_Way]:
        """The ways of what a binding other than an ``=`` to the name alone gives *expr*, read as an assignment's
        value is, so they keep the steps, ``through`` and route an assignment's way has: an unpacking ``=``'s element
        at the name's place (:func:`pathcond.assigned_targets`), a loop's element of its iterable (``[*]``, then the
        name's slot where the target unpacks), a ``with``'s ``__enter__()`` of its context -- a generator's yields
        where the iterable or the context is a call to one (:meth:`_entered`).  A comprehension's ``for`` clause is a
        loop (:func:`pathcond.comprehension_clause`): ``f`` in ``[f for f in self.job.Files]`` is a file there too.

        ``update_jobs``' ``for job in job_list`` read ``job`` as the list whole; ``retEvS, retNewPandaID =
        self.ppEventServiceJob(job, ...)`` read the new job's ``PandaID`` as ``job``.  Where the carry says the value is
        a statement's rows -- a fetch unpacked, a loop over fetched rows -- its ways are kept, a loop's slot given its
        select item's column (``for jediTaskID, commandStr, ... in resList``).  A ``+=`` is a ``+`` of what the name
        held and its value (:meth:`_accumulated`).  A ``:=`` is its value, read as an ``=``'s is -- also where the
        read is in its own statement (:func:`_walrus_before`).  An ``except X as e`` is an object of X where the
        corpus has the class (:meth:`_caught`); another augmented assignment and another ``except ... as`` keep what
        the carry says."""
        binding = (
            pathcond.clause_binding(site, expr.id)
            if isinstance(site, ast.comprehension)
            else next(
                (one for one in pathcond.bindings(func, expr.id) if one.site is site and not one.nested),
                None,
            )
        )
        if binding is not None and binding.kind == pathcond.BIND_EXCEPT:
            caught = self._caught(site, file)  # type: ignore[arg-type]
            if caught:
                return caught
        if binding is not None and binding.kind == pathcond.BIND_WALRUS and binding.value is not None:
            return self.read(binding.value, site, file, func, selves=selves)
        carried = self._sources(expr, point, file, func) or [
            _Way(KEY_UNKNOWN, expr.id, getattr(site, "lineno", None))
        ]
        if binding is None or binding.kind == pathcond.BIND_EXCEPT:
            return carried
        if binding.kind == pathcond.BIND_AUGMENTED:
            if isinstance(site, ast.AugAssign) and isinstance(site.op, ast.Add):
                return self._accumulated(expr, site, file, func, set(), selves=selves)
            return carried
        if any(one.kind in (KEY_ROWS, KEY_SEQUENCE) for one in carried):
            if binding.kind == pathcond.BIND_FOR and binding.unpacked and binding.slot is not None:
                return [self._slot_column(one, binding.slot, expr, file, func) for one in carried]
            return carried
        if binding.kind in (pathcond.BIND_ASSIGN, pathcond.BIND_ANNOTATED):
            value = next(
                (
                    held
                    for target, held in pathcond.assigned_targets(site)
                    if isinstance(target, ast.Name) and target.id == expr.id
                ),
                None,
            )
            return self.read(value, site, file, func, selves=selves) if value is not None else carried
        if binding.kind == pathcond.BIND_FOR:
            # A comprehension's generator has no line: its iterable is read where its statement runs.
            here = site if isinstance(site, ast.stmt) else (_statement_of(site) or point)
            ways = self._entered(site.iter, here, "[*]", file, func, selves=selves)  # type: ignore[attr-defined]
        elif binding.kind == pathcond.BIND_WITH and binding.value is not None:
            ways = self._entered(binding.value, site, _ENTER, file, func, selves=selves)
        else:
            return carried
        if binding.unpacked:
            ways = self._stepped(ways, f"[{binding.slot!r}]" if binding.slot is not None else "[*]")
        return ways

    def _entered(
        self, value: ast.expr, at: ast.AST, step: str, file: str, func: ast.AST, selves: _Selves = ()
    ) -> list[_Way]:
        """The ways of the element a loop over *value* takes (*step* ``[*]``) or of the value a ``with`` over it
        enters (``.__enter__()``): *value*'s with *step*; where *value* calls generators only, what they yield
        (:meth:`_call`) -- ``with self.request_lock(request_id) as locked_spec`` holds the request
        ``request_lock`` yields or its ``None``."""
        if isinstance(value, ast.Call):
            placed = self.calls.callees(file, func, value, bound=selves)
            if placed.targets and not placed.thread and all(self._is_generator(one) for one in placed.targets):
                return self._call(value, file, func, yielded=True, selves=selves)
        return self._stepped(self.read(value, at, file, func, selves=selves), step, element=step == "[*]")

    @staticmethod
    def _stepped(ways: list[_Way], step: str, element: bool = False) -> list[_Way]:
        return _taken(ways, step, element)

    @classmethod
    def _of_field(cls, ways: list[_Way], field: Optional[str]) -> list[_Way]:
        """*ways* with the step ``.field`` where a *field* is asked."""
        return cls._stepped(ways, f".{field}") if field else ways

    def _slot_column(self, way: _Way, slot: int, expr: ast.Name, file: str, func: ast.AST) -> _Way:
        """*way*, a statement's rows a loop unpacks, with the column of the select item at *slot* where it has none."""
        if way.kind != KEY_ROWS or way.column:
            return way
        filled = self.filled(file, func)
        site = next((one for one in filled.sites if one.name == way.name), None)
        if site is None:
            return way
        column, attr, unread = self._sqlwrite._column_choice(filled.items(site), [slot], expr, site, filled.attr)
        return way._replace(column=column, attr=attr or way.attr, column_unread=unread)

    def _changed(
        self,
        file: str,
        func: ast.AST,
        name: str,
        site: ast.AST,
        last: Optional[ast.stmt],
        selves: _Selves = (),
    ) -> tuple[_Change, ...]:
        """What may change in place the object *site* binds *name* to, before *last* reads it or hands it on --
        ``self`` one of *selves* there: ``tmpImpl.refresh()`` in ``FactoryBase.getImpl``, entered as a ``JobBroker``'s,
        runs ``JobBrokerBase.refresh`` alone."""
        if last is None:
            return ()
        key = (id(func), name, id(site), id(last), _selves_key(selves))
        at = last.lineno

        def read() -> tuple[_Change, ...]:
            try:
                raw = changes_mod.object_changes(self.calls, file, func, name, site, last, bound=selves)
            except Exception as error:  # noqa: BLE001 -- as _safely
                raw = [
                    changes_mod.ObjectChange(
                        file,
                        at,
                        CHANGE_UNREAD,
                        changes_mod.EVERY,
                        f"{type(error).__name__}: {error}",
                    )
                ]
            return self._made(raw)

        # Reached again while its values are read, it adds nothing new (:class:`lowlink.Lowlink`).
        return self._lowlink.read(self._hops, key, read, again=())

    def _made(self, raw: Iterable[changes_mod.ObjectChange]) -> tuple[_Change, ...]:
        """*raw*, each once, a store's value read as a way."""
        seen: set[tuple] = set()
        made: list[_Change] = []
        for one in raw:
            mark = (one.file, one.line, one.kind, one.key, one.what, one.via)
            if mark in seen:
                continue
            seen.add(mark)
            value: tuple[_Way, ...] = ()
            if one.value is not None and isinstance(one.func, _FUNCTIONS):
                read = self._safely(one.value, one.value, one.file, one.func)
                if one.element:
                    read = self._stepped(read, one.element)
                # Its own way's end, not what was done to the stored value before: a record of a record.
                value = tuple(way._replace(changes=()) for way in read)
            made.append(
                _Change(
                    f"{one.file}:{one.line}",
                    one.kind,
                    one.key,
                    one.what,
                    one.via,
                    value,
                )
            )
        return tuple(made)

    def after(
        self,
        file: str,
        func: ast.AST,
        name: str,
        first: ast.stmt,
        last: Optional[ast.stmt],
    ) -> tuple[_Change, ...]:
        """What may change in place the object *name* holds after the statement *first* -- and before *last*, or to
        the function's end -- but what the two do themselves: the call handing the object down to the read, and the
        write's, whose changes the guard's and the write's own ways say."""
        key = ("after", id(func), name, id(first), id(last))
        own = {f"{file}:{first.lineno}"}
        if last is not None:
            own.add(f"{file}:{last.lineno}")

        def read() -> tuple[_Change, ...]:
            try:
                raw = changes_mod.object_changes(
                    self.calls, file, func, name, first, last
                )
            except Exception as error:  # noqa: BLE001 -- as _safely
                raw = [
                    changes_mod.ObjectChange(
                        file,
                        first.lineno,
                        CHANGE_UNREAD,
                        changes_mod.EVERY,
                        f"{type(error).__name__}: {error}",
                    )
                ]
            return self._made(
                one
                for one in raw
                if (one.via[0] if one.via else f"{one.file}:{one.line}") not in own
            )

        return self._lowlink.read(self._hops, key, read, again=())

    def hop(self, key: tuple[str, int, str], at: str) -> Optional["_Hop"]:
        """The call at *at* (``file:line``) that hands the parameter *key* names, as :meth:`_parameter_from` read it."""
        if (key, at) not in self._hop_at:
            found = None
            placed = self._functions.get(key)
            if placed is not None:
                file, func = placed
                tree = self.trees.get(file)
                target = Target(
                    file,
                    func,
                    pathcond.class_of(tree, func) if tree is not None else None,
                )
                for site in self.calls.callers(target):
                    if f"{site.file}:{site.call.lineno}" != at:
                        continue
                    try:
                        handed = self.calls.handed(site, target).get(key[2])
                    except Exception:  # noqa: BLE001 -- as _parameter_from
                        handed = None
                    found = _Hop(
                        site.file,
                        site.func,
                        owner_of(site.file, site.func),
                        site.call,
                        handed,
                    )
                    break
            self._hop_at[(key, at)] = found
        return self._hop_at[(key, at)]

    def _call(
        self,
        call: ast.Call,
        file: str,
        func: ast.AST,
        index: Optional[int] = None,
        yielded: bool = False,
        field: Optional[str] = None, selves: _Selves = (),
    ) -> list[_Way]:
        """The ways of what *call* returns -- of its element at a constant *index*, where one is read
        (:meth:`_element`).  A generator's call is the generator, not what the generator returns: with *yielded*,
        what it yields (:meth:`_yielded`) -- the element a loop over it takes, the value a ``with`` over it enters
        (:meth:`_entered`) --, else a stop.  With *field*, the ways of that field of it, read in the callee at its
        returns (:meth:`_returned_field`)."""
        placed = self.calls.callees(file, func, call, bound=selves)
        here = f"{file}:{call.lineno}"
        if placed.targets:
            found: list[_Way] = []
            for target in placed.targets:
                if placed.thread or getattr(target.func, "name", "") == "__init__":
                    # The class built: the one the definition was looked up for (``AdderAtlasPlugin(job)`` runs
                    # its base's ``__init__``), else the one defining it.
                    steps = ((f"[{index!r}]",) if index is not None else ()) + ((f".{field}",) if field else ())
                    if target.bound:
                        builds = [(name.name, _class_key(where, name)) for where, name in target.bound]
                    elif target.cls is not None:
                        builds = [(target.cls.name, _class_key(target.file, target.cls))]
                    else:
                        builds = [(owner_of(target.file, target.func), "")]
                    for built, key in builds:
                        found.append(_Way(KEY_OBJECT, built, call.lineno, site_owner=key, steps=steps))
                    continue
                generator = self._is_generator(target)
                if generator and not yielded:
                    found.append(_stop("generator", owner_of(target.file, target.func), at=here))
                    continue
                handed = bind(call, target, placed.on_class)
                callee = (owner_of(target.file, target.func), target.func.lineno)  # type: ignore[attr-defined]
                receiver: Optional[list[_Way]] = None
                if generator:
                    returned = self._of_field(self._yielded(target), field)
                elif index is not None:
                    returned = self._returned_at(target, index, field)
                else:
                    returned = self._returned(target, field)
                for inner in returned:
                    if (
                        inner.kind == KEY_PARAMETER
                        and (inner.site_owner, inner.site_line) == callee
                    ):
                        given = handed.get(inner.name)
                        if given is None:
                            found.append(_stop("handed_spread", inner.name, at=here))
                            continue
                        read_at, read_file, read_func = _where_read(given, call, file, func, target.file, target.func)
                        # A default is read in the callee, ``self`` its classes there; a value handed, in the caller.
                        read_selves = target.bound if read_func is target.func else selves
                        for outer in self.read(given, read_at, read_file, read_func, selves=read_selves):
                            found += _onto(outer, inner, (), (here,) + inner.route + outer.route)
                    elif inner.kind == KEY_SELF_FIELD and inner.name.split(".")[0] == "self":
                        # The callee's ``self`` is what the call hands it, as a parameter is: its receiver.
                        if receiver is None:
                            receiver = self._receiver(call, target, placed.on_class, handed, file, func, selves=selves)
                        if not receiver:
                            found.append(_stop("receiver", owner_of(target.file, target.func), at=here))
                            continue
                        fields = tuple(f".{part}" for part in inner.name.split(".")[1:])
                        for outer in receiver:
                            found += _onto(outer, inner, fields, (here,) + inner.route + outer.route)
                    else:
                        found.append(inner._replace(route=(here,) + inner.route))
            if placed.unplaced:
                found.append(
                    _stop("open", ast.unparse(call.func), what=placed.unplaced, at=here)
                )
            return found
        taken = self._taken(call, placed, file, func, selves=selves)
        if taken is not None:
            return self._of_field(taken, field)
        # A container made, not a value made of what it is handed: ``WeightedLists(lock)``.
        made = self._made_empty(call, file, func)
        named = None if made else self._through(call, file, func)
        if named is not None:
            ways = [
                one._replace(
                    through=(named._replace(steps=len(one.steps)),) + one.through
                )
                for one in self.read(call.args[0], call, file, func, selves=selves)
            ]
        else:
            ways = [
                _Way(
                    KEY_CALL,
                    ast.unparse(call.func),
                    call.lineno,
                    what=placed.unplaced,
                    made_empty=made,
                )
            ]
        return self._of_field(ways, field)

    def _taken(self, call: ast.Call, placed, file: str, func: ast.AST, selves: _Selves = ()) -> Optional[list[_Way]]:
        """The ways of what a container's method outside the corpus hands back of its receiver, as callfacts says
        (:class:`callfacts.Give`).  One of what it holds (:data:`callfacts.AT`): the receiver's, with the step of the
        constant key the call names, ``[*]`` where it names none or computes it; and the default's, or ``None``,
        where the value may be either instead (``d.get(k, default)``, ``d.get(k)``, a ``WeightedLists``' ``pop()``).
        A view or a copy holding what the receiver holds (:data:`_HOLDS_ELEMENTS_AT`): the receiver's ``[*]``, placed
        where the value holds it -- ``d.values()``' ``[*]``, ``d.items()``' ``[*][1]``.  ``None`` for another call,
        a view of the keys among them."""
        called = call.func
        if not isinstance(called, ast.Attribute) or any(isinstance(one, ast.Starred) for one in call.args):
            return None
        meant = callfacts.meaning(self.calls, file, func, call, placed)
        gives = meant.method.gives if meant.kind in _TAKING and meant.method is not None else None
        if gives is None:
            return None
        spot = _HOLDS_ELEMENTS_AT.get(gives.part)
        if spot is not None:
            if call.args or call.keywords:
                return None
            return _at(spot, self._stepped(self.read(called.value, call, file, func, selves=selves), "[*]"))
        if gives.part != callfacts.AT:
            return None
        key = call.args[gives.key] if gives.key is not None and gives.key < len(call.args) else None
        step = f"[{key.value!r}]" if isinstance(key, ast.Constant) else "[*]"
        found = self._stepped(self.read(called.value, call, file, func, selves=selves), step)
        if gives.default is not None and gives.default < len(call.args):
            found += self.read(call.args[gives.default], call, file, func, selves=selves)
        elif gives.none:
            found.append(_Way(KEY_CONSTANT, name="None", line=call.lineno))
        return found

    def _made_empty(self, call: ast.Call, file: str, func: ast.AST) -> str:
        """What *call* makes empty holds: a builtin container's of no argument (``dict()``), a class outside the
        corpus callfacts says a container (:data:`callfacts.LIBRARY_CONTAINERS`: ``WeightedLists(lock)``); ``""``
        for another call."""
        called = call.func
        if isinstance(called, ast.Name) and not call.args and not call.keywords:
            made = callfacts.CONTAINERS.get(called.id)
            if (
                made is not None
                and made.builtin == called.id
                and made.holds in (callfacts.COLLECTION, callfacts.MAPPING)
                and self.calls.types.is_builtin(file, func, called.id)
            ):
                return made.holds
        if not isinstance(called, (ast.Name, ast.Attribute)):
            return ""
        library = self.calls.types.library_class(file, called)
        return callfacts.LIBRARY_CONTAINERS.get(library, "") if library is not None else ""

    def _receiver(
        self, call: ast.Call, target: Target, on_class: bool, handed: dict, file: str, func: ast.AST,
        selves: _Selves = (),
    ) -> list[_Way]:
        """The ways of the object *call* runs *target*, a method, on: its receiver (``job`` in ``job.isCancelled()``),
        the caller's ``self`` for ``super().m()``, the first argument of a method read off its class; none for a
        static or class method or a call naming no object (``m = job.isCancelled; m()``)."""
        if target.cls is None or any(
            isinstance(one, ast.Name) and one.id in ("staticmethod", "classmethod")
            for one in getattr(target.func, "decorator_list", ())
        ):
            return []
        if on_class:
            arguments = target.func.args  # type: ignore[attr-defined]
            first = (arguments.posonlyargs + arguments.args)[:1]
            given = handed.get(first[0].arg) if first else None
            return self.read(given, call, file, func, selves=selves) if given is not None else []
        if not isinstance(call.func, ast.Attribute):
            return []
        value = call.func.value
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "super":
            return [_Way(KEY_SELF_FIELD, name="self", line=call.lineno, site_owner=self.class_owner(file, func))]
        return self.read(value, call, file, func, selves=selves)

    def _through(self, call: ast.Call, file: str, func: ast.AST) -> Optional[_Through]:
        """A call outside the corpus that is handed one positional argument -- a function or a module's
        (``json.loads``), not a method of a value --, with what its value is of the argument; else ``None``."""
        if (
            len(call.args) != 1
            or isinstance(call.args[0], ast.Starred)
            or any(one.arg is None for one in call.keywords)
        ):
            return None
        types = self.calls.types
        called = call.func
        if isinstance(called, ast.Name):
            if called.id in _parameters(func) or _bound_here(func, called.id):
                return None
        elif not (
            isinstance(called, ast.Attribute)
            and isinstance(called.value, ast.Name)
            and called.value.id in types.imports(file)
            and not _bound_here(func, called.value.id)
        ):
            return None
        library = types.library_class(file, called)
        named = ".".join(library) if library is not None else ast.unparse(called)
        hooks: list[str] = []
        if library in callfacts.JSON_MAKERS:
            for one in call.keywords:
                if one.arg in callfacts.JSON_HOOKS:
                    hooks += [
                        owner_of(target.file, target.func)
                        for target in self.calls.named_callables(file, func, one.value)
                    ] or [ast.unparse(one.value)]
        handed = {one.arg for one in call.keywords}
        runs = (
            named in evaluate.RUNS
            and (library is not None or types.is_builtin(file, func, named))
            and handed <= (_PARSE_HOOKS if library in callfacts.JSON_MAKERS else set())
        )
        return _Through(
            named,
            f"{file}:{call.lineno}",
            _gives(types, file, func, call, library),
            tuple(hooks),
            runs,
        )

    def _returned(self, target: Target, field: Optional[str] = None) -> list[_Way]:
        """The ways of what *target* returns -- of its *field*, read at each return, where one is asked --, ``self``
        there each class the call placed it for (:attr:`Target.bound`), read once for each."""
        key = (id(target.func), _selves_key(target.bound), field)
        held = self._returns.get(key)
        if held is not None:
            return held
        self._returns[key] = [
            _stop(
                "cycle",
                owner_of(target.file, target.func),
                at=f"{target.file}:{getattr(target.func, 'lineno', '')}",
            )
        ]
        found: list[_Way] = []
        for value in returns(target.func):
            found += self._safely(_field_of(value, field), value, target.file, target.func, selves=target.bound)
        self._returns[key] = found
        return found

    def _returned_at(self, target: Target, index: int, field: Optional[str] = None) -> list[_Way]:
        """:meth:`_returned`'s, of each return's element at *index*: the element where a tuple or list literal gives
        it (:meth:`_element`), else the return's ways with the step ``[index]`` -- and its *field*, where one is
        asked."""
        key = (id(target.func), _selves_key(target.bound), index, field)
        held = self._returns_at.get(key)
        if held is not None:
            return held
        self._returns_at[key] = [
            _stop(
                "cycle",
                owner_of(target.file, target.func),
                at=f"{target.file}:{getattr(target.func, 'lineno', '')}",
            )
        ]
        found: list[_Way] = []
        for value in returns(target.func):
            try:
                chosen = self._element(value, index, value, target.file, target.func, field, selves=target.bound)
            except Exception as error:  # noqa: BLE001 -- as _safely
                chosen = [
                    _stop(
                        "failed",
                        ast.unparse(value),
                        what=f"{type(error).__name__}: {error}",
                        at=f"{target.file}:{getattr(value, 'lineno', '')}",
                    )
                ]
            if chosen is None:
                whole = self._safely(value, value, target.file, target.func, selves=target.bound)
                chosen = self._of_field(self._stepped(whole, f"[{index!r}]"), field)
            found += chosen
        self._returns_at[key] = found
        return found

    def _is_generator(self, target: Target) -> bool:
        """Whether *target*'s own body yields: a call to it makes a generator."""
        held = self._generators.get(id(target.func))
        if held is None:
            held = any(isinstance(node, (ast.Yield, ast.YieldFrom)) for node in own_nodes(target.func))
            self._generators[id(target.func)] = held
        return held

    def _yielded(self, target: Target) -> list[_Way]:
        """:meth:`_returned`'s, of a generator: the ways of each value its own body yields -- a bare ``yield``'s
        ``None``, a ``yield from``'s iterable's element (``[*]``)."""
        key = (id(target.func), _selves_key(target.bound))
        held = self._yields.get(key)
        if held is not None:
            return held
        self._yields[key] = [
            _stop(
                "cycle",
                owner_of(target.file, target.func),
                at=f"{target.file}:{getattr(target.func, 'lineno', '')}",
            )
        ]
        found: list[_Way] = []
        for node in own_nodes(target.func):
            if isinstance(node, ast.Yield):
                value = node.value if node.value is not None else ast.Constant(value=None)
                found += self._safely(value, node, target.file, target.func, selves=target.bound)
            elif isinstance(node, ast.YieldFrom):
                stepped = self._safely(node.value, node, target.file, target.func, selves=target.bound)
                found += self._stepped(stepped, "[*]")
        self._yields[key] = found
        return found

    def _safely(
        self, expr: ast.expr, at: ast.AST, file: str, func: ast.AST, selves: _Selves = ()
    ) -> list[_Way]:
        try:
            return self.read(expr, at, file, func, selves=selves)
        except Exception as error:  # noqa: BLE001 -- a reading that fails is a stop, not a failed build
            what = f"{type(error).__name__}: {error}"
            return [
                _stop(
                    "failed",
                    ast.unparse(expr),
                    what=what,
                    at=f"{file}:{getattr(at, 'lineno', '')}",
                )
            ]

    # -- the parameters --------------------------------------------------------

    def queue(self, sources: Iterable[HeldSource], rest: tuple[str, ...] = ()) -> None:
        """The parameters *sources*, a record's ways, end at: to be read; and the field of each ``self`` or object
        one ends at (:class:`FieldFrom`) -- with *rest*, the steps still to be taken past the ends, read on through
        the ParameterFroms and FieldFroms they lead to."""
        for source in sources:
            key = (source.site_owner, source.site_line or 0, source.name)
            if source.kind == KEY_PARAMETER:
                if key not in self.parameters:
                    self._pending.setdefault(key, self._functions[key])
                onward = tuple(source.steps) + rest
                if onward:
                    self._ask("parameter", key, onward)
            elif source.site_owner:
                path = end_path(source)
                if path is not None and path + rest:
                    path += rest
                    field = FIELD_STEP.match(path[0])
                    if field is not None:
                        self._ask("field", (source.site_owner, field.group(1)), path[1:])
            for change in source.changes:
                self.queue(change.value)
            if rest:
                # A container made empty, read on at an element: what its changes put.
                values, _stays = held_elements(source, rest[0])
                self.queue(values, rest[1:])

    def ask_field(self, owner: str, field: str) -> None:
        """Read the FieldFrom of *owner*'s *field*: one a guard reading ``self.<field>`` off ``self`` names."""
        if owner:
            self._ask("field", (owner, field), ())

    def _ask(self, kind: str, key: tuple, rest: tuple[str, ...]) -> None:
        if len(rest) <= _FIELD_REST and (kind, key, rest) not in self._asked:
            self._asked[(kind, key, rest)] = None
            self._asks.append((kind, key, rest))

    def settle(self) -> None:
        """Read every parameter a record's way ends at, and the ones those reach; and every field asked."""
        while self._pending or self._asks:
            while self._pending:
                key, (file, func) = self._pending.popitem()
                if key in self.parameters:
                    continue
                made = self.parameters[key] = self._parameter_from(key, file, func)
                self.queue(made.sources)
            if not self._asks:
                continue
            kind, asked, rest = self._asks.pop(0)
            if kind == "parameter":
                parameter = self.parameters.get(asked)
                if parameter is not None:
                    self.queue(parameter.sources, rest)
                continue
            owner, name = asked
            field = self.fields.get((owner, name))
            if field is None:
                field = self.fields[(owner, name)] = self._field_from(owner, name)
                for binding in field.bindings:
                    self.queue(binding.sources)
                for change in field.changes:
                    self.queue(change.value)
            if rest:
                for binding in field.bindings:
                    self.queue(binding.sources, rest)

    # -- the fields ------------------------------------------------------------

    def _class_at(self, owner: str) -> Optional[tuple[str, ast.ClassDef]]:
        if self._classes is None:
            self._classes = {}
            for file, tree in self.trees.items():
                for node in walk(tree):
                    if isinstance(node, ast.ClassDef):
                        self._classes.setdefault(_class_key(file, node), (file, node))
        return self._classes.get(owner)

    def _family(self, file: str, cls: ast.ClassDef) -> list[tuple[str, ast.ClassDef]]:
        """*cls*'s corpus bases in its order, and its corpus subclasses: the classes ``self`` may be of."""
        found: dict[tuple[str, int], tuple[str, ast.ClassDef]] = {}
        for one_file, one in self.calls.types.mro(file, cls):
            found.setdefault((one_file, one.lineno), (one_file, one))
        todo = [(file, cls)]
        while todo:
            here_file, here = todo.pop()
            for one_file, one in self.calls.subclasses(here_file, here):
                if (one_file, one.lineno) not in found:
                    found[(one_file, one.lineno)] = (one_file, one)
                    todo.append((one_file, one))
        return list(found.values())

    def _stores(self) -> dict[str, list[tuple[str, str]]]:
        """``{field: [(file:line, class)]}``: each store of a field of the corpus -- ``class`` the method's whose
        ``self`` it is stored on, empty for another object (``x.f = v``, ``setattr(x, "f", v)``)."""
        if self._stored is None:
            self._stored = {}
            for file, tree in self.trees.items():
                for node, inner in _with_functions(tree):
                    me = _first_positional(inner) if _method_class(inner) is not None else None
                    stored: list[tuple[ast.expr, str]] = []
                    if isinstance(node, (ast.Assign, ast.AnnAssign)):
                        stored = [(target, "") for target, _value in pathcond.assigned_targets(node)]
                    elif isinstance(node, ast.AugAssign):
                        stored = [(node.target, "")]
                    elif (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == "setattr"
                        and len(node.args) == 3
                        and isinstance(node.args[1], ast.Constant)
                        and isinstance(node.args[1].value, str)
                    ):
                        stored = [(node.args[0], node.args[1].value)]
                    for target, named in stored:
                        if named:
                            holder, field = target, named
                        elif isinstance(target, ast.Attribute):
                            holder, field = target.value, target.attr
                        else:
                            continue
                        own = isinstance(holder, ast.Name) and me is not None and holder.id == me
                        self._stored.setdefault(field, []).append(
                            (f"{file}:{getattr(node, 'lineno', 0)}", self.class_owner(file, inner) if own else "")
                        )
        return self._stored

    def _field_from(self, owner: str, field: str) -> FieldFrom:
        """What *field* of an object of *owner* may hold: each binding of it in the class's methods and body and in
        its bases' and subclasses', read as a guard's name is; what may change the object it holds there; and why it
        may hold more."""
        place = self._class_at(owner)
        if place is None:
            return FieldFrom(
                owner=owner, field=field, stops=[KeyStop(kind="failed", name=field, what="no class of the corpus", at=[owner])]
            )
        try:
            family = self._family(*place)
        except Exception as error:  # noqa: BLE001 -- as _safely
            family = [place]
            failed = [KeyStop(kind="failed", name=field, what=f"{type(error).__name__}: {error}", at=[owner])]
        else:
            failed = []
        bindings: list[FieldBinding] = []
        immutable: list[bool] = []
        changes: list[_Change] = []
        stops: list[KeyStop] = list(failed)
        for file, cls in family:
            for statement in cls.body:
                if isinstance(statement, _FUNCTIONS) and statement.name == field:
                    stops.append(KeyStop(kind="method", name=field, at=[f"{file}:{statement.lineno}"]))
                elif isinstance(statement, (ast.Assign, ast.AnnAssign)) and getattr(statement, "value", None) is not None:
                    for target, value in pathcond.assigned_targets(statement):
                        if isinstance(target, ast.Name) and target.id == field and value is not None:
                            bindings.append(
                                self._binding(file, statement, _class_key(file, cls), FIELD_CLASS, value, cls, immutable)
                            )
            for method in pathcond.methods_of(cls):
                me = _first_positional(method)
                if me is None or any(
                    isinstance(one, ast.Name) and one.id == "staticmethod" for one in method.decorator_list
                ):
                    continue
                touched = through_dict = False
                for node, inner in _with_functions(method):
                    if _is_field(node, me, field):
                        touched = True
                    if isinstance(node, (ast.Assign, ast.AnnAssign)):
                        how = FIELD_ANNOTATED if isinstance(node, ast.AnnAssign) else FIELD_ASSIGN
                        for target, value in pathcond.assigned_targets(node):
                            if not _is_field(target, me, field):
                                continue
                            if value is None:
                                stops.append(
                                    KeyStop(kind="element", name=field, what="a starred unpacking", at=[f"{file}:{node.lineno}"])
                                )
                                continue
                            bindings.append(self._binding(file, node, owner_of(file, method), how, value, inner, immutable))
                    elif isinstance(node, ast.AugAssign) and _is_field(node.target, me, field):
                        bindings.append(
                            self._binding(file, node, owner_of(file, method), FIELD_AUGMENTED, node.value, inner, immutable)
                        )
                    elif isinstance(node, (ast.For, ast.AsyncFor, ast.With, ast.AsyncWith)):
                        bound = (
                            [node.target]
                            if isinstance(node, (ast.For, ast.AsyncFor))
                            else [item.optional_vars for item in node.items if item.optional_vars is not None]
                        )
                        if any(_is_field(one, me, field) for target in bound for one in walk(target)):
                            stops.append(
                                KeyStop(kind="bound", name=field, what="a loop's or a with's target", at=[f"{file}:{node.lineno}"])
                            )
                    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "setattr":
                        if len(node.args) == 3 and isinstance(node.args[0], ast.Name) and node.args[0].id == me:
                            name = node.args[1]
                            if isinstance(name, ast.Constant):
                                if name.value == field:
                                    bindings.append(
                                        self._binding(
                                            file, node, owner_of(file, method), FIELD_SETATTR, node.args[2], inner, immutable
                                        )
                                    )
                            else:
                                keyed = self._set_by_keyword(file, method, inner, node, field, family)
                                if keyed is None:
                                    stops.append(
                                        KeyStop(kind="computed", name=field, what=ast.unparse(node), at=[f"{file}:{node.lineno}"])
                                    )
                                elif keyed:
                                    bindings.append(
                                        _field_binding(file, node, owner_of(file, method), FIELD_SETATTR, keyed, inner, immutable)
                                    )
                    elif isinstance(node, ast.Attribute) and node.attr == "__dict__" and isinstance(node.value, ast.Name) and node.value.id == me:
                        through_dict = True
                if through_dict:
                    # Read -- ``__str__``'s ``for key in self.__dict__`` -- it sets nothing; changed in place, as
                    # the changes reading reads a field's object, it may set any field.
                    try:
                        raw = changes_mod.field_changes(self.calls, file, method, f"{me}.__dict__")
                    except Exception as error:  # noqa: BLE001 -- as _changed
                        raw = [
                            changes_mod.ObjectChange(
                                file, method.lineno, CHANGE_UNREAD, changes_mod.EVERY, f"{type(error).__name__}: {error}"
                            )
                        ]
                    sites = sorted({f"{one.file}:{one.line}" for one in raw})
                    if sites:
                        stops.append(
                            KeyStop(kind="computed", name=field, what=f"{me}.__dict__ changed in place", at=sites)
                        )
                if touched:
                    try:
                        raw = changes_mod.field_changes(self.calls, file, method, f"{me}.{field}")
                    except Exception as error:  # noqa: BLE001 -- as _changed
                        raw = [
                            changes_mod.ObjectChange(
                                file, method.lineno, CHANGE_UNREAD, changes_mod.EVERY, f"{type(error).__name__}: {error}"
                            )
                        ]
                    changes += self._made(raw)
        outside = sorted(at for at, stored_on in self._stores().get(field, ()) if not stored_on)
        if outside:
            stops.append(KeyStop(kind="outside", name=field, what="stored on another object, by the name", at=outside))
        if bindings and all(immutable) and not stops:
            # Each binding holds a value nothing changes in place -- a text, a number, a column's value, a parameter
            # annotated so: the field is only bound again, as its bindings say, and what it is handed to keeps a
            # value no later step can change.
            changes = []
        seen: set[tuple] = set()
        said: list[HeldChange] = []
        for change in changes:
            mark = (change.at, change.kind, change.key, change.what, change.via)
            if mark not in seen:
                seen.add(mark)
                said.append(_change_said(change))
        unique_stops: dict[tuple, KeyStop] = {}
        for stop in stops:
            unique_stops.setdefault((stop.kind, stop.name, stop.what, tuple(stop.at)), stop)
        return FieldFrom(
            owner=owner,
            field=field,
            bindings=sorted(bindings, key=lambda one: (one.at.rsplit(":", 1)[0], int(one.at.rsplit(":", 1)[1]))),
            changes=said,
            stops=list(unique_stops.values()),
        )

    def _binding(
        self,
        file: str,
        statement: ast.AST,
        owner: str,
        how: str,
        value: ast.expr,
        func: ast.AST,
        immutable: list[bool],
    ) -> FieldBinding:
        """The binding of *value* at *statement*; whether it holds a value nothing changes in place goes on
        *immutable* (:func:`_immutable`)."""
        return _field_binding(file, statement, owner, how, self._safely(value, statement, file, func), func, immutable)

    def _set_by_keyword(
        self,
        file: str,
        method: ast.AST,
        inner: ast.AST,
        call: ast.Call,
        field: str,
        family: list[tuple[str, ast.ClassDef]],
    ) -> Optional[list[_Way]]:
        """The ways of what ``setattr(self, key, value)`` in *method* sets *field* to, where the loop over
        ``P.items()`` around it names each key and value and P is *method*'s parameter: at each caller, the value of the
        keyword *field* that P's dict holds (:meth:`_keywords_handed`).  Empty where no caller hands the name; ``None``
        for another setattr, or where a caller cannot be read -- the name may be any."""
        if inner is not method:
            return None
        handed = _items_of_parameter(method, call)
        if handed is None:
            return None
        return self._keywords_handed(file, method, handed, field, family, 1)

    def _keywords_handed(
        self,
        file: str,
        func: ast.AST,
        name: str,
        field: str,
        family: list[tuple[str, ast.ClassDef]],
        level: int,
    ) -> Optional[list[_Way]]:
        """The ways of the value each call of *func* hands under the key *field* in its parameter *name*'s dict -- a
        dict of constant keys (a literal, or the keywords :func:`bind` leaves to ``**kwargs``), or the caller's own
        parameter, read on at its callers up to :data:`_KEYWORD_LEVELS` calls -- each with the calls it came through
        on its route.  ``None`` where any caller cannot be read, as a ParameterFrom stops: no caller, a call Calls
        misses, a ``**`` at the call, another value, a level too deep.  A method read off its class that hands its own
        ``self`` from a class outside *family* is skipped: that call sets the field of an object of another class."""
        tree = self.trees.get(file)
        target = Target(file, func, pathcond.class_of(tree, func) if tree is not None else None)
        sites = self.calls.callers(target)
        if not sites or self.calls.missed(target):
            return None
        ways: list[_Way] = []
        for site in sites:
            here = f"{site.file}:{site.call.lineno}"
            try:
                placed = self.calls.callees(site.file, site.func, site.call)
                if placed.on_class and self._hands_other_self(site, family):
                    continue
                given = bind(site.call, target, placed.on_class).get(name)
            except Exception:  # noqa: BLE001 -- as _safely
                return None
            if given is None:
                return None
            at, read_file, read_func = _where_read(given, site.call, site.file, site.func, file, func)
            if isinstance(given, ast.Dict) and all(
                isinstance(key, ast.Constant) and isinstance(key.value, str) for key in given.keys
            ):
                for key, value in zip(given.keys, given.values, strict=True):
                    if isinstance(key, ast.Constant) and key.value == field:
                        ways += [
                            one._replace(route=(here,) + one.route)
                            for one in self._safely(value, at, read_file, read_func)
                        ]
            elif (
                read_func is site.func
                and isinstance(given, ast.Name)
                and given.id in _parameters(site.func)
                and not _rebinds(site.func, given.id)
                and level < _KEYWORD_LEVELS
            ):
                outer = self._keywords_handed(site.file, site.func, given.id, field, family, level + 1)
                if outer is None:
                    return None
                ways += [one._replace(route=(here,) + one.route) for one in outer]
            else:
                return None
        return ways

    def _hands_other_self(self, site: CallSite, family: list[tuple[str, ast.ClassDef]]) -> bool:
        """Whether *site*, a method read off its class, hands its own ``self`` and is a method of a class outside
        *family*."""
        me = _first_positional(site.func)
        if me is None or not site.call.args or not (
            isinstance(site.call.args[0], ast.Name) and site.call.args[0].id == me
        ):
            return False
        tree = self.trees.get(site.file)
        cls = pathcond.class_of(tree, site.func) if tree is not None else None
        return cls is not None and (site.file, cls.lineno) not in {(one, held.lineno) for one, held in family}

    def _parameter_from(
        self, key: tuple[str, int, str], file: str, func: ast.AST
    ) -> ParameterFrom:
        owner, line, name = key
        tree = self.trees.get(file)
        target = Target(
            file, func, pathcond.class_of(tree, func) if tree is not None else None
        )
        ways: list[_Way] = []
        sites: list[CallSite] = self.calls.callers(target)
        for site in sites:
            here = f"{site.file}:{site.call.lineno}"
            try:
                handed = self.calls.handed(site, target)
            except Exception as error:  # noqa: BLE001 -- as _safely
                ways.append(
                    _stop(
                        "failed", name, what=f"{type(error).__name__}: {error}", at=here
                    )
                )
                continue
            given = handed.get(name)
            if given is None:
                ways.append(_stop("handed_spread", name, at=here))
                continue
            ways += [
                one._replace(route=(here,) + one.route)
                for one in self._safely(given, *_where_read(given, site.call, site.file, site.func, file, func))
            ]
        if not sites:
            ways.append(_stop("no_caller", name, at=f"{owner}:{line}"))
        for why in self.calls.missed(target):
            ways.append(_stop("missed", owner, what=why, at=f"{owner}:{line}"))
        sources, stops = _said(ways)
        return ParameterFrom(
            owner=owner, line=line, parameter=name, sources=sources, stops=stops
        )


def _gives(
    types, file: str, func: ast.AST, call: ast.Call, library: Optional[tuple[str, str]]
) -> str:
    """What the value of *call*, outside the corpus, is of its one argument, as :mod:`callfacts` says: a function
    of an imported module's (*library*), else a builtin's."""
    if library is None:
        function = callfacts.builtin_function(types, file, func, call)
        if function is None or function.gives is None:
            return THROUGH_UNKNOWN
        if function.gives in callfacts.CONVERTING:
            return THROUGH_CONVERTED
        return function.gives
    module, name = library
    if library in callfacts.JSON_MAKERS or name in callfacts.READ_AGAIN:
        return THROUGH_DECODED
    if library in callfacts.COPIES:
        return THROUGH_COPY
    if name == callfacts.SERIALIZING:
        return THROUGH_TEXT
    if module.split(".")[0] in callfacts.COMPUTING_MODULES:
        return THROUGH_COMPUTED
    return THROUGH_UNKNOWN


def _raised(literal: str, through: list[_Through]) -> Optional[tuple[_Through, str]]:
    """The call of *through*, in the order run, that raises on the literal *literal* (its repr) and the exception's
    name: each run on what the one before made while :func:`evaluate.run` runs it and no step comes between, a JSON
    maker handed hooks the last (they make its value); ``None`` where none is known to raise."""
    try:
        value = ast.literal_eval(literal)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return None
    for call in through:
        if not call.runs or call.steps:
            return None
        made = evaluate.run(call.call, value)
        if isinstance(made, evaluate.Raised):
            return call, made.error
        if made is evaluate.UNKNOWN or call.hooks:
            return None
        value = made
    return None


def _said(ways: list[_Way]) -> tuple[list[HeldSource], list[KeyStop]]:
    """*ways* as the record keeps them, each once; *route* and *through* in the order the value went.  A literal a
    call on the way raises on is a stop there."""
    sources: dict[tuple, HeldSource] = {}
    stops: dict[tuple, KeyStop] = {}
    for one in ways:
        route, through = list(reversed(one.route)), list(reversed(one.through))
        raised = (
            _raised(one.name, through) if one.kind == KEY_CONSTANT and through else None
        )
        if raised is not None:
            call, error = raised
            one = _stop("raises", one.name, f"{call.call} raises {error}", call.at)
        if one.kind == _STOP:
            at = [*route, one.at] if one.at else route
            stops.setdefault(
                (one.stop, one.name, one.what, tuple(at)),
                KeyStop(kind=one.stop, name=one.name, what=one.what, at=at),
            )
            continue
        key = (one._replace(line=None, end_file=""), tuple(route))
        held = sources.setdefault(
            key,
            HeldSource(
                kind=one.kind,
                line=one.line,
                name=one.name,
                site_owner=one.site_owner,
                site_line=one.site_line,
                paths=list(one.paths),
                column=one.column,
                attr=one.attr,
                column_unread=one.column_unread,
                steps=list(one.steps),
                through=[
                    HeldThrough(
                        call=call.call,
                        at=call.at,
                        gives=call.gives,
                        hooks=list(call.hooks),
                        after_steps=call.steps,
                    )
                    for call in through
                ],
                route=route,
                changes=[_change_said(change) for change in one.changes],
                made_empty=one.made_empty,
            ),
        )
        if one.end_file and one.line is not None and f"{one.end_file}:{one.line}" not in held.at:
            held.at.append(f"{one.end_file}:{one.line}")
    for held in sources.values():
        held.at.sort(key=lambda place: (place.rpartition(":")[0], int(place.rpartition(":")[2])))
    return list(sources.values()), list(stops.values())


def _end_sites(guards: Iterable[GuardFrom], reader: _Reader) -> set[str]:
    """``file:line`` of each place a way ends at (:attr:`HeldSource.at`): the source's own statement, on a
    GuardFrom's, a ParameterFrom's or a FieldFrom binding's way, as :func:`_change_sites` reads their changes."""
    found: set[str] = set()
    for guard in guards:
        for source in guard.sources:
            found.update(source.at)
    for parameter in reader.parameters.values():
        for source in parameter.sources:
            found.update(source.at)
    for field in reader.fields.values():
        for binding in field.bindings:
            for source in binding.sources:
                found.update(source.at)
    return found


def _unchanging(func: ast.AST, name: str, site: ast.AST, ways: list[_Way]) -> bool:
    """Whether nothing can change in place what *site* gives *name*: a parameter annotated with such types only, or
    a value every way of which ends at a literal or a column's value, through no call."""
    if site is getattr(func, "args", None):
        return callfacts.immutable_annotation(_annotation(func, name))
    ends = [one for one in ways if one.kind != _STOP]
    return bool(ends) and all(
        one.kind in (KEY_CONSTANT, KEY_ROWS, KEY_SEQUENCE)
        and not one.through
        and not one.steps
        for one in ends
    )


def _field_binding(
    file: str, statement: ast.AST, owner: str, how: str, ways: list[_Way], func: ast.AST, immutable: list[bool]
) -> FieldBinding:
    """The binding at *statement* whose value's ways are *ways*; whether it holds a value nothing changes in place
    goes on *immutable* (:func:`_immutable`)."""
    immutable.append(_immutable(func, ways))
    sources, stops = _said(ways)
    return FieldBinding(
        at=f"{file}:{getattr(statement, 'lineno', 0)}", owner=owner, how=how, sources=sources, stops=stops
    )


#: How many calls back a parameter's dict of keywords is read for a setattr's name: the plugin's ``__init__(self,
#: job, **params)`` hands its ``params`` on to the base's, and the call that builds the plugin names the keywords.
_KEYWORD_LEVELS = 3


def _items_of_parameter(method: ast.AST, call: ast.Call) -> Optional[str]:
    """P where *call* is ``setattr(self, key, value)`` in a ``for key, value in P.items()`` of *method*'s own body
    (:func:`calls.keyed_setattr`), P a parameter *method* does not bind again (``AdderPluginBase.__init__``'s
    ``params``); else ``None``."""
    held = keyed_setattr(method, call)
    if held is None or held.id not in _parameters(method) or _rebinds(method, held.id):
        return None
    return held.id


def _immutable(func: ast.AST, ways: list[_Way]) -> bool:
    """Whether every way of a field's binding ends at a value nothing can change in place: a literal or a column's
    value, through no call and stepped into nothing; or a parameter of *func* itself annotated with such types only
    (``job_status: str``).  A way that stops says nothing, so it is not."""
    return bool(ways) and all(
        not one.steps
        and not one.through
        and (
            one.kind in (KEY_CONSTANT, KEY_ROWS, KEY_SEQUENCE)
            or (
                one.kind == KEY_PARAMETER
                and not one.route
                and callfacts.immutable_annotation(_annotation(func, one.name))
            )
        )
        for one in ways
    )


def _annotation(func: ast.AST, name: str) -> Optional[ast.expr]:
    arguments: Optional[ast.arguments] = getattr(func, "args", None)
    if arguments is None:
        return None
    for one in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs):
        if one.arg == name:
            return one.annotation
    return None


def _change_said(change: _Change) -> HeldChange:
    value, stops = _said(list(change.value))
    return HeldChange(
        at=change.at,
        kind=change.kind,
        key=change.key,
        what=change.what,
        via=list(change.via),
        value=value,
        value_stops=stops,
    )


def _statement_of(node: ast.AST) -> Optional[ast.stmt]:
    here: Optional[ast.AST] = node
    while here is not None and not isinstance(here, ast.stmt):
        here = getattr(here, "parent", None)
    return here


def _walrus_before(name: str, point: ast.AST, statement: Optional[ast.stmt]) -> Optional[ast.NamedExpr]:
    """The ``:=`` binding *name* in *statement*'s own expressions -- not its body's statements, not a lambda's --
    whose target *point* is, else the last one *point* comes after: the binding of a read in the statement that
    binds it.  :func:`pathcond.held_at` reads bindings by statements, so a ``:=`` reaches no read in its own
    statement: ``if pseudo_coll_list := ds_list_dict["pseudo_coll_list"]:`` and ``nStaging == 0`` in ``if (nStaging
    := ...) is not None and nStaging == 0:`` read it.  After is by place in the text, the order Python runs an
    expression in."""
    if statement is None:
        return None
    here = (getattr(point, "lineno", 0), getattr(point, "col_offset", 0))
    found: Optional[ast.NamedExpr] = None
    stack = [
        child
        for child in ast.iter_child_nodes(statement)
        if not isinstance(child, (ast.stmt, ast.ExceptHandler, ast.match_case))
    ]
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Lambda):
            continue
        if isinstance(node, ast.NamedExpr) and isinstance(node.target, ast.Name) and node.target.id == name:
            if node.target is point:
                return node
            end = (node.end_lineno or 0, node.end_col_offset or 0)
            if end <= here and (found is None or end > (found.end_lineno or 0, found.end_col_offset or 0)):
                found = node
        stack.extend(ast.iter_child_nodes(node))
    return found


def _function_at(tree: ast.AST, name: str, line: int) -> Optional[ast.AST]:
    for node in walk(tree):
        if (
            isinstance(node, _FUNCTIONS)
            and node.name == name
            and node.lineno <= line <= (node.end_lineno or node.lineno)
        ):
            return node
    return None


def _statement_at(func: ast.AST, line: int) -> Optional[ast.stmt]:
    for node in walk(func):
        if isinstance(node, ast.stmt) and node is not func and node.lineno == line:
            return node
    return None


def _names(test: ast.expr, func: ast.AST) -> dict[str, ast.expr]:
    """The names *test* reads that are *func*'s parameters or locals -- not a module's name or a comprehension's
    own -- each at its first read; and each object rooted at ``self`` or ``cls`` that a field or method is read off,
    by its path: ``self`` for ``self.splitRule is None``, ``self.taskSpec`` for ``self.taskSpec.usePrePro()`` -- the
    object a :class:`WriteFrom` of an arm writing ``self.splitRule`` or ``self.taskSpec.splitRule`` writes."""
    inner = {
        target.id
        for node in walk(test)
        if isinstance(node, ast.comprehension)
        for target in walk(node.target)
        if isinstance(target, ast.Name)
    }
    own = _parameters(func)
    found: dict[str, ast.expr] = {}
    parts = {id(node.value) for node in walk(test) if isinstance(node, ast.Attribute)}
    for node in walk(test):
        if isinstance(node, ast.Attribute) and id(node) not in parts:
            path = _self_path(node.value)
            if path is not None:
                found.setdefault(path, node.value)
        elif (
            isinstance(node, ast.Name)
            and node.id not in ("self", "cls")
            and node.id not in inner
        ):
            if node.id in own or _binds(func, node.id):
                found.setdefault(node.id, node)
    return found


def _self_values(test: ast.expr) -> list[str]:
    """The fields *test* reads as values off ``self`` itself -- ``attempt_nr`` in ``self.attempt_nr is not None`` --,
    not a method it calls (``self.isCancelled()``)."""
    called = {id(node.func) for node in walk(test) if isinstance(node, ast.Call)}
    return sorted(
        {
            node.attr
            for node in walk(test)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and id(node) not in called
        }
    )


def _self_path(node: ast.expr) -> Optional[str]:
    """``self.taskSpec`` for that expression: attributes on ``self`` or ``cls``, or ``None``."""
    here = node
    while isinstance(here, ast.Attribute):
        here = here.value
    if isinstance(here, ast.Name) and here.id in ("self", "cls"):
        return ast.unparse(node)
    return None


def _written(
    statement: ast.stmt,
    subject: str,
    calls: Calls,
    file: str,
    func: ast.AST,
    methods: dict[str, set[str]],
) -> list[tuple[ast.expr, Optional[ast.expr]]]:
    """The objects *statement*, an arm writing *subject*, writes the field of, each with the value it gives there:
    ``(jobSpec, taskSpec.jediTaskID)`` for ``jobSpec.jediTaskID = taskSpec.jediTaskID``, at any depth of a tuple
    target (:func:`pathcond.assigned_targets`); ``(o, v)`` for ``setattr(o, "f", v)``; the receiver and no value
    for ``o.f += v`` and for a call at the line of a method of the subject's class (``taskSpec.setOnHold()``) --
    placed there, or, a receiver the build tells no class of (``taskSpec, cloudName, inputChunk = tmpInputItem``),
    named as a method the class defines (*methods*, by class name).  An SQL arm's statement fills a bind or the
    statement's text (``varMap[":status"] = ...``), or hands the text on (``taskBuffer.querySQLS(...)``): none."""
    owner, _, field = subject.rpartition(".")
    found: list[tuple[ast.expr, Optional[ast.expr]]] = []
    if isinstance(statement, ast.AugAssign):
        reread = statement.target
        if isinstance(reread, ast.Attribute) and reread.attr == field:
            found.append((reread.value, None))
        return found
    for target, value in pathcond.assigned_targets(statement):
        if isinstance(target, ast.Attribute) and target.attr == field:
            found.append((target.value, value))
    if found or not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        return found
    call = statement.value
    if isinstance(call.func, ast.Name) and call.func.id == "setattr":
        if len(call.args) == 3 and isinstance(call.args[1], ast.Constant) and call.args[1].value == field:
            found.append((call.args[0], call.args[2]))
        return found
    if isinstance(call.func, ast.Attribute):
        placed = calls.callees(file, func, call)
        if placed.targets:
            if any(target.cls is not None and target.cls.name == owner for target in placed.targets):
                found.append((call.func.value, None))
        elif call.func.attr in methods.get(owner, ()):
            found.append((call.func.value, None))
    return found


def _object_ways(
    said: tuple[list[HeldSource], list[KeyStop]],
) -> tuple[list[HeldSource], list[KeyStop]]:
    """A write's ways without what may change the object on them (*changes*): which object a name holds, not what it
    holds -- a reader binding it to a guard's compares where the ways end; ways the same once bare are one."""
    sources: list[HeldSource] = []
    for one in said[0]:
        bare = one.model_copy(update={"changes": []})
        if bare not in sources:
            sources.append(bare)
    return sources, said[1]


def _methods(trees: dict[str, ast.AST]) -> dict[str, set[str]]:
    """The names of the methods each class of the corpus defines, by the class's name."""
    found: dict[str, set[str]] = {}
    for tree in trees.values():
        for node in walk(tree):
            if isinstance(node, ast.ClassDef):
                found.setdefault(node.name, set()).update(
                    one.name for one in node.body if isinstance(one, _FUNCTIONS)
                )
    return found


def guard_froms(
    junctions: Iterable[JunctionNode],
    read_sites: Iterable[ReadSiteNode],
    trees: dict[str, ast.AST],
    calls: Optional[Calls],
    record,
    writes: Iterable[SetFrom] = (),
    entered: Iterable[EnteredFrom] = (),
    fields: Iterable[tuple[str, str]] = (),
) -> tuple[list[GuardFrom], list[ParameterFrom], list[SetFrom], list[WriteFrom], list[FieldFrom]]:
    """A :class:`GuardFrom` for each test on the way to each junction's arm, and to each site of *entered* (its
    :attr:`GuardFrom.calls`), and each name it reads (:func:`_names`); each of *writes*
    (:func:`sqlwrite.extract`'s) whose column a guard's way ends at -- through the ParameterFroms -- its ways read as
    the guards', and on a guard's way ending where one of its column's does, what may change the object between
    (:func:`_after_reads`); a :class:`WriteFrom` for the object each arm writes and the value each passthrough arm
    hands on (:func:`_written`), read as the guards' names are; a :class:`ParameterFrom` for each parameter a
    way of any ends at, and those reach; and a :class:`FieldFrom` for each field of a ``self`` or an object a way of
    any ends at -- its first step past the end, then the next ones past the FieldFroms' and ParameterFroms' ends --
    each field a test reads off ``self`` itself, and each of *fields*, ``(file::Class, field)``: the fields of
    ``self`` the definitions the arms' conditions call read (:func:`runnable.condition_fields`).  Nothing without
    *calls* or *record*: the reading places calls and reads fetches by them."""
    if calls is None or record is None:
        return [], [], [], [], []
    junctions = list(junctions)
    reader = _Reader(
        calls, record, read_sites, {junction.owner for junction in junctions}, trees
    )
    found: dict[tuple[str, int, int, str], GuardFrom] = {}
    written: dict[tuple[str, int, str, str, str], WriteFrom] = {}
    read_once: dict[int, tuple[list[HeldSource], list[KeyStop]]] = {}
    methods = _methods(trees)

    def guards_on(owner: str, file: str, func: ast.AST, stmt: ast.stmt) -> list[GuardFrom]:
        """The GuardFrom of each test on the way to *stmt* in *func* and name it reads, each read once."""
        on: list[GuardFrom] = []
        for test in pathcond.tests_on(stmt, func):
            for read, node in _names(test, func).items():
                key = (owner, test.lineno, test.col_offset, read)
                guard = found.get(key)
                if guard is None:
                    sources, stops = _said(reader._safely(node, node, file, func))
                    reader.queue(sources)
                    if read == "self":
                        for field in _self_values(test):
                            reader.ask_field(reader.class_owner(file, func), field)
                    guard = found[key] = GuardFrom(
                        owner=owner,
                        test_line=test.lineno,
                        test=ast.unparse(test),
                        name=read,
                        sources=sources,
                        stops=stops,
                    )
                on.append(guard)
        return on

    for junction in junctions:
        file, _, name = junction.owner.partition("::")
        tree = trees.get(file)
        for branch in junction.branches:
            line = branch.line
            func = (
                _function_at(tree, name, line)
                if tree is not None and line is not None
                else None
            )
            stmt = (
                _statement_at(func, line)
                if func is not None and line is not None
                else None
            )
            if func is None or stmt is None or line is None:
                continue
            passes = branch.outcome.startswith("passthrough(")
            for target_object, target_value in _written(stmt, junction.subject, calls, file, func, methods):
                for role, part in ((WRITE_OBJECT, target_object), (WRITE_VALUE, target_value if passes else None)):
                    if part is None:
                        continue
                    expression = ast.unparse(part)
                    write_key = (junction.owner, line, junction.subject, role, expression)
                    if write_key in written:
                        continue
                    part_ways = read_once.get(id(part))
                    if part_ways is None:
                        part_ways = read_once[id(part)] = _object_ways(
                            _said(reader._safely(part, part, file, func))
                        )
                        reader.queue(part_ways[0])
                    written[write_key] = WriteFrom(
                        owner=junction.owner,
                        line=line,
                        subject=junction.subject,
                        role=role,
                        expression=expression,
                        sources=part_ways[0],
                        stops=part_ways[1],
                    )
            for guard in guards_on(junction.owner, file, func, stmt):
                if line not in guard.arms:
                    guard.arms.append(line)
    for way in entered:
        for site in way.sites:
            if site.unread:
                continue
            file, _, name = site.owner.partition("::")
            tree = trees.get(file)
            func = next(
                (
                    node
                    for node in (walk(tree) if tree is not None else ())
                    if isinstance(node, _FUNCTIONS) and node.name == name and node.lineno == site.owner_line
                ),
                None,
            )
            stmt = _statement_holding(func, site.line) if func is not None else None
            if func is None or stmt is None:
                continue
            for guard in guards_on(site.owner, file, func, stmt):
                if site.line not in guard.calls:
                    guard.calls.append(site.line)
    for owner, field in fields:
        reader.ask_field(owner, field)
    # The tests on the way to each place a way's object may be changed, as to an entered site's call: which change
    # a run made is read off them (``if is_fatal: self.result.set_fatal()``); and to each place a way ends at, the
    # source's own statement: which binding a run made is (``if (timeNow - modTime) > ...: ignoreTmpError = False``).
    # Their ways' too, until no place is new.
    guarded: set[str] = set()
    held: dict[str, list[ast.AST]] = {}
    while True:
        reader.settle()
        fresh = sorted((_change_sites(found.values(), reader) | _end_sites(found.values(), reader)) - guarded)
        if not fresh:
            break
        guarded.update(fresh)
        for at in fresh:
            file, _, number = at.rpartition(":")
            tree = trees.get(file)
            if tree is None or not number.isdigit():
                continue
            if file not in held:
                held[file] = [node for node in walk(tree) if isinstance(node, _FUNCTIONS)]
            line = int(number)
            func = _function_holding(held[file], line)
            stmt = _statement_holding(func, line) if func is not None else None
            if func is None or stmt is None:
                continue
            for guard in guards_on(owner_of(file, func), file, func, stmt):
                if line not in guard.calls:
                    guard.calls.append(line)
    ended = _ended_columns(found.values(), reader.parameters)
    sets = [one for one in writes if one.column in ended]
    for one in sets:
        ways: list[_Way] = []
        for file, func, read in one._reads:
            here = f"{file}:{one.statement}"
            try:
                binds = read()
            except Exception as error:  # noqa: BLE001 -- as _safely
                ways.append(
                    _stop(
                        "failed",
                        one.text,
                        what=f"{type(error).__name__}: {error}",
                        at=here,
                    )
                )
                continue
            if not binds and sqltext.bind_keys(one.text):
                ways.append(_stop("no_bind", one.text, at=here))
            for filled in binds:
                ways += reader._safely(filled.value, filled.site, file, func)
        one.sources, one.stops = _said(ways)
        reader.queue(one.sources)
    reader.settle()
    _after_reads(reader, found.values(), sets)
    reader.settle()
    for guard in found.values():
        guard.arms.sort()
        guard.calls.sort()
    return (
        sorted(found.values(), key=lambda one: (one.owner, one.test_line, one.name)),
        sorted(
            reader.parameters.values(),
            key=lambda one: (one.owner, one.line, one.parameter),
        ),
        sorted(sets, key=lambda one: (one.owner, one.statement, one.column, one.text)),
        sorted(written.values(), key=lambda one: (one.owner, one.line, one.subject, one.role, one.expression)),
        sorted(reader.fields.values(), key=lambda one: (one.owner, one.field)),
    )


# -- after the read (WB-2) ---------------------------------------------------------------------------------------

#: The ways read down to one end, each side of a guard and a write; past it, a stop (``cut``).
_CHAINS = 16


class _Hop(NamedTuple):
    """A call a way goes down through to the next function: the caller's file, function and owner, the call, and
    what it hands the parameter."""

    file: str
    func: ast.AST
    owner: str
    call: ast.Call
    handed: Optional[ast.expr]


def _rows_end(source: HeldSource) -> Optional[tuple[str, int, str]]:
    """A ``rows`` end's statement and column."""
    if source.kind == KEY_ROWS and source.column:
        return (source.site_owner, source.site_line or 0, source.column)
    return None


def _parameter_ends(
    parameters: dict[tuple[str, int, str], ParameterFrom], values: bool
) -> dict[tuple[str, int, str], frozenset[tuple[str, int, str]]]:
    """The rows ends each ParameterFrom's ways reach, on through the ParameterFroms they end at -- and, *values*,
    through their changes' stored values: one pass over each, then the links to a fixpoint."""
    direct: dict[tuple[str, int, str], set] = {}
    links: dict[tuple[str, int, str], set] = {}

    def scan(sources: Iterable[HeldSource], key: tuple[str, int, str]) -> None:
        for source in sources:
            end = _rows_end(source)
            if end is not None:
                direct[key].add(end)
            elif source.kind == KEY_PARAMETER:
                links[key].add((source.site_owner, source.site_line or 0, source.name))
            if values:
                for change in source.changes:
                    scan(change.value, key)

    for key, made in parameters.items():
        direct[key], links[key] = set(), set()
        scan(made.sources, key)
    changed = True
    while changed:
        changed = False
        for key, ends in direct.items():
            for other in links[key]:
                more = direct.get(other, set()) - ends
                if more:
                    ends |= more
                    changed = True
    return {key: frozenset(ends) for key, ends in direct.items()}


def _source_ends(
    sources: Iterable[HeldSource],
    ends: dict[tuple[str, int, str], frozenset],
    values: bool,
) -> set[tuple[str, int, str]]:
    found: set[tuple[str, int, str]] = set()
    for source in sources:
        end = _rows_end(source)
        if end is not None:
            found.add(end)
        elif source.kind == KEY_PARAMETER:
            found |= ends.get(
                (source.site_owner, source.site_line or 0, source.name), frozenset()
            )
        if values:
            for change in source.changes:
                found |= _source_ends(change.value, ends, values)
    return found


def _chains(
    reader: _Reader,
    sources: Iterable[HeldSource],
    end: tuple[str, int, str],
    ends: dict[tuple[str, int, str], frozenset],
    values: bool,
) -> tuple[list[tuple[_Hop, ...]], bool, bool]:
    """The calls each way of *sources* ending at *end* goes down through to the record's function, outermost first,
    each once; a change's stored value's way (*values*) is the value's and adds none, so a store whose value reaches
    *end* ends one there.  At most :data:`_CHAINS`; whether more were left; whether a way, any, went through a
    decoder (the column's text made a container)."""
    found: list[tuple[_Hop, ...]] = []
    more = False
    decoded = False

    def ends_here(hops: tuple[_Hop, ...], decodes: bool) -> None:
        nonlocal more, decoded
        decoded = decoded or decodes
        if hops in found:
            return
        if len(found) >= _CHAINS:
            more = True
            return
        found.append(hops)

    def down(
        sources: Iterable[HeldSource],
        hops: tuple[_Hop, ...],
        active: frozenset,
        decodes: bool,
    ) -> None:
        for source in sources:
            decodes_here = decodes or any(
                one.gives == THROUGH_DECODED for one in source.through
            )
            if _rows_end(source) == end:
                ends_here(hops, decodes_here)
                continue
            if source.kind == KEY_PARAMETER:
                key = (source.site_owner, source.site_line or 0, source.name)
                made = reader.parameters.get(key)
                if made is not None and key not in active and end in ends.get(key, ()):
                    for one in made.sources:
                        hop = reader.hop(key, one.route[-1]) if one.route else None
                        if hop is not None:
                            down([one], (hop, *hops), active | {key}, decodes_here)
            if values and any(
                end in _source_ends(change.value, ends, values)
                for change in source.changes
            ):
                ends_here(hops, decodes_here)

    down(sources, (), frozenset(), False)
    return found, more, decoded


def _change_sites(guards: Iterable[GuardFrom], reader: _Reader) -> set[str]:
    """``file:line`` of each place a way's object may be changed: each change on a GuardFrom's, a ParameterFrom's or
    a FieldFrom binding's way, and each of a FieldFrom's -- the change's first call or statement in the way's function
    (*via*), else the change itself."""
    found: set[str] = set()

    def add(changes: Iterable[HeldChange]) -> None:
        for one in changes:
            found.add(one.via[0] if one.via else one.at)

    for guard in guards:
        for source in guard.sources:
            add(source.changes)
    for parameter in reader.parameters.values():
        for source in parameter.sources:
            add(source.changes)
    for field in reader.fields.values():
        add(field.changes)
        for binding in field.bindings:
            for source in binding.sources:
                add(source.changes)
    return found


def _function_holding(functions: list[ast.AST], line: int) -> Optional[ast.AST]:
    """The innermost of *functions* whose lines hold *line*."""
    found: Optional[ast.AST] = None
    for node in functions:
        if node.lineno <= line <= (node.end_lineno or node.lineno) and (  # type: ignore[attr-defined]
            found is None or node.lineno >= found.lineno  # type: ignore[attr-defined]
        ):
            found = node
    return found


def _statement_holding(func: ast.AST, line: int) -> Optional[ast.stmt]:
    """The innermost statement of *func* whose lines hold *line*."""
    found: Optional[ast.stmt] = None
    for node in walk(func):
        if (
            isinstance(node, ast.stmt)
            and node is not func
            and node.lineno <= line <= (node.end_lineno or node.lineno)
            and (found is None or node.lineno >= found.lineno)
        ):
            found = node
    return found


def _held_name(hop: _Hop) -> Optional[str]:
    return hop.handed.id if isinstance(hop.handed, ast.Name) else None


def _own_names(guard: GuardFrom, source: HeldSource) -> list[str]:
    """The names *guard*'s function holds the object of *source*'s way by: the guard's own where its value is the
    object or a part of it reached by items and attributes -- not an object built of its parts (*paths*), as
    ``AtlasAnalTaskRefiner.doRefine``'s ``datasetSpec`` of ``taskParamMap``, nor a path on ``self`` --, and the
    parameter the way ends at."""
    names = [] if source.paths or not guard.name.isidentifier() or guard.name in ("self", "cls") else [guard.name]
    if (
        source.kind == KEY_PARAMETER
        and source.site_owner == guard.owner
        and source.name
        and source.name not in names
    ):
        names.append(source.name)
    return names


def _after_reads(
    reader: _Reader, guards: Iterable[GuardFrom], sets: list[SetFrom]
) -> None:
    """:attr:`HeldSource.after` and *after_stops* of each guard's way ending at the rows a SetFrom of the same
    column's way ends at: what may change the object between the guard's read and the write."""
    if not sets:
        return
    plain = _parameter_ends(reader.parameters, False)
    stored = _parameter_ends(reader.parameters, True)
    by_end: dict[
        tuple[str, int, str], list[tuple[SetFrom, list[tuple[_Hop, ...]], bool]]
    ] = {}
    for write in sets:
        for end in _source_ends(write.sources, stored, True):
            if end[2] == write.column:
                chains, more, _ = _chains(reader, write.sources, end, stored, True)
                by_end.setdefault(end, []).append((write, chains, more))
    for guard in guards:
        file, _, name = guard.owner.partition("::")
        tree = reader.trees.get(file)
        func = _function_at(tree, name, guard.test_line) if tree is not None else None
        test = _statement_at(func, guard.test_line) if func is not None else None
        if func is None or test is None:
            continue
        for source in guard.sources:
            for end in sorted(_source_ends([source], plain, False)):
                if end not in by_end:
                    continue
                chains, more, decoded = _chains(reader, [source], end, plain, False)
                if not decoded:
                    # A column's value no way decodes is not changed in place.
                    continue
                for write, written, also in by_end[end]:
                    _after_pair(
                        reader,
                        guard,
                        (file, func, test),
                        source,
                        write,
                        chains,
                        written,
                        more or also,
                    )


def _after_pair(
    reader: _Reader,
    guard: GuardFrom,
    read: tuple[str, ast.AST, ast.stmt],
    source: HeldSource,
    write: SetFrom,
    chains: list[tuple[_Hop, ...]],
    written: list[tuple[_Hop, ...]],
    more: bool,
) -> None:
    """The changes after *guard*'s read on *source* and before *write*, for each way down to each, put on *source*."""
    file, func, test = read
    ref = f"{write.owner}:{write.statement}:{write.column}"
    here = f"{file}:{guard.test_line}"
    stops: dict[tuple[str, str], KeyStop] = {}

    def stop(kind: str, at: str, what: str = "") -> None:
        stops.setdefault((kind, at), KeyStop(kind=kind, name=ref, what=what, at=[at]))

    if more:
        stop("cut", here, f"{_CHAINS} ways a side")
    own = _own_names(guard, source)
    for down in chains:
        lstack: list[tuple[str, Optional[_Hop]]] = [
            *((hop.owner, hop) for hop in down),
            (guard.owner, None),
        ]
        for across in written:
            wstack: list[tuple[str, Optional[_Hop]]] = [
                *((hop.owner, hop) for hop in across),
                (write.owner, None),
            ]
            owners = [owner for owner, _ in wstack]
            meet = next(
                (
                    (i, owners.index(owner))
                    for i, (owner, _) in reversed(list(enumerate(lstack)))
                    if owner in owners
                ),
                None,
            )
            if meet is None:
                stop("no_meeting", here)
                continue
            i, j = meet
            lhop, whop = lstack[i][1], wstack[j][1]
            first: Optional[ast.stmt]
            held: list[str]
            if lhop is None:
                m_file, m_func, first, held = file, func, test, own
            else:
                handed = _held_name(lhop)
                m_file, m_func, first, held = (
                    lhop.file,
                    lhop.func,
                    _statement_of(lhop.call),
                    [handed] if handed is not None else [],
                )
            last = (
                _statement_of(whop.call)
                if whop is not None
                else _statement_holding(m_func, write.statement)
            )
            at = f"{m_file}:{getattr(first, 'lineno', '')}"
            if first is None or last is None:
                stop("unnamed", at, "no statement")
                continue
            try:
                forward = flow.runs_after(first, last)
            except Exception:  # noqa: BLE001 -- a flow that cannot say keeps the write
                forward = True
            if not forward:
                stop("write_first", at)
                continue
            # Down from the meeting function: the guard's own from its read, each caller between from its call, in
            # the order they return -- to their ends, before the meeting function runs on to the write.
            levels: list[tuple[str, ast.AST, str, ast.stmt]] = []
            if lhop is not None:
                levels += [(file, func, name, test) for name in own]
                if not own:
                    stop("unnamed", here, guard.name)
                for _, hop in reversed(lstack[i + 1 : -1]):
                    called = _statement_of(hop.call) if hop is not None else None
                    name = _held_name(hop) if hop is not None else None
                    if hop is not None and called is not None and name is not None:
                        levels.append((hop.file, hop.func, name, called))
                    elif hop is not None and called is not None:
                        stop(
                            "unnamed",
                            f"{hop.file}:{called.lineno}",
                            ast.unparse(hop.handed) if hop.handed is not None else "",
                        )
            for level_file, level_func, name, level_first in levels:
                for change in reader.after(
                    level_file, level_func, name, level_first, None
                ):
                    _put(reader, source, change, name, ref, False)
            if not held:
                stop(
                    "unnamed",
                    at,
                    ast.unparse(lhop.handed)
                    if lhop is not None and lhop.handed is not None
                    else guard.name,
                )
                continue
            # Run again after the write on a later pass, the name given no new value on the way round: a statement
            # binding it (a parameter's is the entry, no stop on the way round).
            for name in held:
                killers = [
                    one.site
                    for one in pathcond.bindings(m_func, name)
                    if not one.nested and isinstance(one.site, ast.stmt)
                ]
                for change in reader.after(m_file, m_func, name, first, last):
                    placed = change.via[0] if change.via else change.at
                    statement = _statement_holding(
                        m_func, int(placed.rpartition(":")[2] or 0)
                    )
                    try:
                        either = statement is not None and flow.runs_after(
                            last, statement, killers
                        )
                    except Exception:  # noqa: BLE001 -- as above
                        either = True
                    _put(reader, source, change, name, ref, either)
    source.after_stops += [
        one for one in stops.values() if one not in source.after_stops
    ]


def _put(
    reader: _Reader,
    source: HeldSource,
    change: _Change,
    name: str,
    ref: str,
    either: bool,
) -> None:
    """*change*, read on *name*, on *source*'s :attr:`HeldSource.after`, once, with the write *ref* it can run
    before."""
    for one in source.after:
        if (one.at, one.kind, one.key, one.what, tuple(one.via), one.name) == (
            change.at,
            change.kind,
            change.key,
            change.what,
            change.via,
            name,
        ):
            break
    else:
        said = _change_said(change)
        one = AfterChange(
            name=name,
            **{field: getattr(said, field) for field in HeldChange.model_fields},
        )
        source.after.append(one)
        reader.queue(one.value)
    if ref not in one.before:
        one.before.append(ref)
    if either and ref not in one.either:
        one.either.append(ref)


def _ended_columns(
    guards: Iterable[GuardFrom], parameters: dict[tuple[str, int, str], ParameterFrom]
) -> set[str]:
    """The columns the guards' ways end at (``rows``), on through the ParameterFroms of the parameters they end at."""
    found: set[str] = set()
    seen: set[tuple[str, int, str]] = set()
    todo = [source for guard in guards for source in guard.sources]
    while todo:
        source = todo.pop()
        if source.kind == KEY_ROWS and source.column:
            found.add(source.column)
        elif source.kind == KEY_PARAMETER:
            key = (source.site_owner, source.site_line or 0, source.name)
            if key in parameters and key not in seen:
                seen.add(key)
                todo += parameters[key].sources
    return found
