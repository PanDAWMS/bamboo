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
from typing import Iterable, NamedTuple, Optional

from bamboo.codemap import callfacts
from bamboo.codemap.astwalk import walk
from bamboo.codemap.calls import Calls, CallSite, Target, bind, own_nodes, returns
from bamboo.codemap.models import (
    CHANGE_UNREAD,
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
from bamboo.codemap.panda import evaluate, flow, pathcond, sqltext

_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)
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


def _step(way: _Way, step: str) -> _Way:
    """*way* with *step* taken: an element of a list, set or generator a comprehension made last is its element
    (``[f for f in job.Files][0]`` is ``job.Files``'s ``[*]``, not ``[*][0]``), as a tuple literal's is
    (:meth:`_Reader._element`); a dict's elements are its keys, not the values it is read as."""
    if (
        step.startswith("[")
        and way.through
        and way.through[0].call in _ELEMENT_COMPREHENSIONS
        and way.through[0].steps == len(way.steps)
    ):
        return way._replace(through=way.through[1:])
    return way._replace(steps=way.steps + (step,))


def owner_of(file: str, func: ast.AST) -> str:
    return f"{file}::{getattr(func, 'name', '<module>')}"


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
        self._returns: dict[tuple[int, Optional[str]], list[_Way]] = {}
        self._returns_at: dict[tuple[int, int, Optional[str]], list[_Way]] = {}
        self._yields: dict[int, list[_Way]] = {}
        self._generators: dict[int, bool] = {}
        self._active: set[tuple[int, int]] = set()
        #: ``(owner, def line, parameter)`` -> its ParameterFrom; the ones a record's way ends at, not yet read;
        #: and the function of each a way has ended at -- a callee's, filled at its call, ends in no record.
        self.parameters: dict[tuple[str, int, str], ParameterFrom] = {}
        self._pending: dict[tuple[str, int, str], tuple[str, ast.AST]] = {}
        self._functions: dict[tuple[str, int, str], tuple[str, ast.AST]] = {}
        #: ``(function, name, binding, statement read at)`` -> the changes between, by ids of the build's trees.
        self._hops: dict[tuple, tuple[_Change, ...]] = {}
        #: ``(parameter, file:line)`` -> the call there handing it, read once (:meth:`hop`).
        self._hop_at: dict[tuple, Optional[_Hop]] = {}

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
        statement's count, not a field a run's object holds (key sources read them as the field)."""
        found = []
        cursors = self._cursors(file, func)
        for source in self.filled(file, func).sources(expr, at):
            if source.kind == KEY_SELF_FIELD and source.name in cursors:
                continue
            paths = tuple(path for path in source.paths if path != "()")
            if source.kind == KEY_PARAMETER:
                found.append(self._parameter(file, func, source.name, paths))
                continue
            found.append(
                _Way(
                    source.kind,
                    name=source.name,
                    line=source.line,
                    site_owner=source.site_owner,
                    site_line=source.site_line,
                    paths=paths,
                    column=source.column,
                    attr=source.attr,
                    column_unread=source.column_unread,
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

    def read(self, expr: ast.expr, at: ast.AST, file: str, func: ast.AST) -> list[_Way]:
        """The ways of *expr*, read where *at* runs in *func*; one ending at *func*'s own parameter is kind
        ``parameter``."""
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
            return self._read(expr, at, file, func)
        finally:
            self._active.discard(mark)

    def _read(
        self, expr: ast.expr, at: ast.AST, file: str, func: ast.AST
    ) -> list[_Way]:
        line = getattr(expr, "lineno", None)
        if isinstance(expr, ast.Name):
            return self._name(expr, at, file, func)
        rows = self._rows(expr, at, file, func)
        if rows:
            return rows
        if isinstance(expr, ast.Call):
            return self._call(expr, file, func)
        if isinstance(expr, (ast.Subscript, ast.Attribute)) and not (
            isinstance(expr.value, ast.Name) and expr.value.id in ("self", "cls")
        ):
            position = _index(expr)
            if position is not None:
                chosen = self._element(expr.value, position, at, file, func)
                if chosen is not None:
                    return chosen
            if isinstance(expr, ast.Attribute):
                returned = self._returned_field(expr, at, file, func)
                if returned is not None:
                    return returned
            if isinstance(expr, ast.Subscript):
                index = expr.slice
                step = (
                    f"[{index.value!r}]" if isinstance(index, ast.Constant) else "[*]"
                )
            else:
                step = f".{expr.attr}"
            return [_step(one, step) for one in self.read(expr.value, at, file, func)]
        if isinstance(expr, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            # A new collection of the element's values: the element's ways, through the comprehension.
            element = expr.value if isinstance(expr, ast.DictComp) else expr.elt
            made = _Through(_COMPREHENSIONS[type(expr)], f"{file}:{line}", THROUGH_COMPUTED)
            return [
                one
                if one.kind == _STOP
                else one._replace(through=(made._replace(steps=len(one.steps)),) + one.through)
                for one in self.read(element, element, file, func)
            ]
        found = self._sources(expr, at, file, func)
        return found or [
            _Way(
                KEY_ATTRIBUTE if isinstance(expr, ast.Attribute) else KEY_UNKNOWN,
                ast.unparse(expr),
                line,
            )
        ]

    def _returned_field(self, expr: ast.Attribute, at: ast.AST, file: str, func: ast.AST) -> Optional[list[_Way]]:
        """The ways of a field of what a call of the corpus returns, read in the callee at its returns as that field
        -- a field the callee filled from a row (``taskSpec.pack(res)``) is the row's column --: of the call, of its
        element at a constant index, or of a name every reaching binding of which gives it one of these
        (``tmpStat, taskSpec = self.taskBufferIF.getTaskWithID_JEDI(jediTaskID)``); ``None`` else, for the field's
        step to be taken on the value's ways."""
        value = expr.value
        called = self._called_at(value, file, func)
        if called is not None:
            return self._call(called[0], file, func, called[1], field=expr.attr)
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
            called = self._called_at(given, file, func) if given is not None else None
            if called is None:
                return None
            bound.append((site, called))
        if not bound:
            return None
        last = _statement_of(point)
        found: list[_Way] = []
        for site, (call, index) in bound:
            ways = self._call(call, file, func, index, field=expr.attr)
            found += self._with_changes(value, site, ways, file, func, last)
        return found

    def _called_at(self, expr: ast.expr, file: str, func: ast.AST) -> Optional[tuple[ast.Call, Optional[int]]]:
        """*expr* as a call of the corpus that returns (not the database's, not a generator's), with the constant
        index of its element *expr* takes, if any; else ``None``."""
        index = _index(expr)
        call = expr.value if index is not None else expr  # type: ignore[attr-defined]
        if not isinstance(call, ast.Call):
            return None
        if self.record is not None and self.record.is_database(file, call):
            return None
        placed = self.calls.callees(file, func, call)
        if not placed.targets or placed.thread or any(self._is_generator(one) for one in placed.targets):
            return None
        return call, index

    def _element(
        self, value: ast.expr, index: int, at: ast.AST, file: str, func: ast.AST, field: Optional[str] = None
    ) -> Optional[list[_Way]]:
        """The ways of *value*'s element at a constant *index* where a tuple or list literal gives it: the literal's
        own, a name's binding to one (each binding: the others' ways with the step ``[index]``), a call's returns
        (each return: the same); ``None`` where nothing does, for the caller to add the step.  ``retEvS, retNewPandaID
        = self.ppEventServiceJob(...)`` reads the call's returned ``(9, jobSpec.PandaID)``'s ``jobSpec.PandaID``, not
        the tuple.  With *field*, the ways of that field of the element (:meth:`_returned_field`)."""
        if isinstance(value, (ast.Tuple, ast.List)):
            element = _literal_element(value, index)
            return self.read(_field_of(element, field), at, file, func) if element is not None else None
        if isinstance(value, ast.Call):
            placed = self.calls.callees(file, func, value)
            if not placed.targets or placed.thread:
                return None
            return self._call(value, file, func, index, field=field)
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
                found += self.read(_field_of(element, field), site, file, func)
            else:
                ways = self._stepped(self._site_ways(value, point, site, file, func, last), f"[{index!r}]")
                found += self._of_field(ways, field)
        return found

    def _name(
        self, expr: ast.Name, at: ast.AST, file: str, func: ast.AST
    ) -> list[_Way]:
        if expr.id in ("self", "cls"):
            return [_Way(KEY_SELF_FIELD, name=expr.id, line=expr.lineno)]
        point = expr if getattr(expr, "parent", None) is not None else at
        if (
            expr.id not in _parameters(func)
            and not _bound_here(func, expr.id)
            and pathcond.comprehension_clause(func, expr.id, point) is None
        ):
            return [_Way(KEY_GLOBAL, name=expr.id, line=expr.lineno)]
        last = _statement_of(point)
        found: list[_Way] = []
        for site in pathcond.held_at(func, expr.id, point).sites:  # type: ignore[arg-type]
            found += self._site_ways(expr, point, site, file, func, last)
        return found

    def _site_ways(
        self, expr: ast.Name, point: ast.AST, site: ast.AST, file: str, func: ast.AST, last: Optional[ast.stmt]
    ) -> list[_Way]:
        """The ways *site*, a binding reaching the read at *point*, gives *expr*, with what may change the object
        in place between."""
        return self._with_changes(expr, site, self._bound(expr, point, site, file, func), file, func, last)

    def _with_changes(
        self, expr: ast.Name, site: ast.AST, ways: list[_Way], file: str, func: ast.AST, last: Optional[ast.stmt]
    ) -> list[_Way]:
        """*ways*, what *site* gives *expr*, with what may change the object in place between it and the read."""
        changed = () if _unchanging(func, expr.id, site, ways) else self._changed(file, func, expr.id, site, last)
        return [one._replace(changes=one.changes + changed) if changed and one.kind != _STOP else one for one in ways]

    def _bound(
        self, expr: ast.Name, point: ast.AST, site: ast.AST, file: str, func: ast.AST
    ) -> list[_Way]:
        """The ways of what *site*, a binding reaching the read, gives *expr*."""
        if site is func.args:  # type: ignore[attr-defined]
            if expr.id in _parameters(func):
                return [self._parameter(file, func, expr.id)]
            return [_stop("unbound", expr.id, at=f"{file}:{expr.lineno}")]
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
            return self.read(value, site, file, func)
        return self._carried(expr, point, site, file, func)

    def _carried(
        self, expr: ast.Name, point: ast.AST, site: ast.AST, file: str, func: ast.AST
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
        select item's column (``for jediTaskID, commandStr, ... in resList``).  ``+=``, ``except ... as`` and ``:=``
        keep what the carry says (:func:`pathcond.held_at` gives a ``:=`` no reaching binding)."""
        carried = self._sources(expr, point, file, func) or [
            _Way(KEY_UNKNOWN, expr.id, getattr(site, "lineno", None))
        ]
        binding = (
            pathcond.clause_binding(site, expr.id)
            if isinstance(site, ast.comprehension)
            else next(
                (one for one in pathcond.bindings(func, expr.id) if one.site is site and not one.nested),
                None,
            )
        )
        if binding is None or binding.kind in (pathcond.BIND_AUGMENTED, pathcond.BIND_EXCEPT):
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
            return self.read(value, site, file, func) if value is not None else carried
        if binding.kind == pathcond.BIND_FOR:
            # A comprehension's generator has no line: its iterable is read where its statement runs.
            here = site if isinstance(site, ast.stmt) else (_statement_of(site) or point)
            ways = self._entered(site.iter, here, "[*]", file, func)  # type: ignore[attr-defined]
        elif binding.kind == pathcond.BIND_WITH and binding.value is not None:
            ways = self._entered(binding.value, site, ".__enter__()", file, func)
        else:
            return carried
        if binding.unpacked:
            ways = self._stepped(ways, f"[{binding.slot!r}]" if binding.slot is not None else "[*]")
        return ways

    def _entered(self, value: ast.expr, at: ast.AST, step: str, file: str, func: ast.AST) -> list[_Way]:
        """The ways of the element a loop over *value* takes (*step* ``[*]``) or of the value a ``with`` over it
        enters (``.__enter__()``): *value*'s with *step*; where *value* calls generators only, what they yield
        (:meth:`_call`) -- ``with self.request_lock(request_id) as locked_spec`` holds the request
        ``request_lock`` yields or its ``None``."""
        if isinstance(value, ast.Call):
            placed = self.calls.callees(file, func, value)
            if placed.targets and not placed.thread and all(self._is_generator(one) for one in placed.targets):
                return self._call(value, file, func, yielded=True)
        return self._stepped(self.read(value, at, file, func), step)

    @staticmethod
    def _stepped(ways: list[_Way], step: str) -> list[_Way]:
        return [one if one.kind == _STOP else _step(one, step) for one in ways]

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
    ) -> tuple[_Change, ...]:
        """What may change in place the object *site* binds *name* to, before *last* reads it or hands it on."""
        if last is None:
            return ()
        key = (id(func), name, id(site), id(last))
        held = self._hops.get(key)
        if held is None:
            # Reached again while its values are read, it adds nothing new.
            self._hops[key] = ()
            try:
                raw = changes_mod.object_changes(
                    self.calls, file, func, name, site, last
                )
            except Exception as error:  # noqa: BLE001 -- as _safely
                raw = [
                    changes_mod.ObjectChange(
                        file,
                        last.lineno,
                        CHANGE_UNREAD,
                        changes_mod.EVERY,
                        f"{type(error).__name__}: {error}",
                    )
                ]
            held = self._hops[key] = self._made(raw)
        return held

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
                # Its own way's end, not what was done to the stored value before: a record of a record.
                value = tuple(
                    way._replace(changes=())
                    for way in self._safely(one.value, one.value, one.file, one.func)
                )
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
        held = self._hops.get(key)
        if held is None:
            self._hops[key] = ()
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
            own = {f"{file}:{first.lineno}"}
            if last is not None:
                own.add(f"{file}:{last.lineno}")
            held = self._hops[key] = self._made(
                one
                for one in raw
                if (one.via[0] if one.via else f"{one.file}:{one.line}") not in own
            )
        return held

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
        field: Optional[str] = None,
    ) -> list[_Way]:
        """The ways of what *call* returns -- of its element at a constant *index*, where one is read
        (:meth:`_element`).  A generator's call is the generator, not what the generator returns: with *yielded*,
        what it yields (:meth:`_yielded`) -- the element a loop over it takes, the value a ``with`` over it enters
        (:meth:`_entered`) --, else a stop.  With *field*, the ways of that field of it, read in the callee at its
        returns (:meth:`_returned_field`)."""
        placed = self.calls.callees(file, func, call)
        here = f"{file}:{call.lineno}"
        if placed.targets:
            found: list[_Way] = []
            for target in placed.targets:
                if placed.thread or getattr(target.func, "name", "") == "__init__":
                    built = (
                        target.cls.name
                        if target.cls is not None
                        else owner_of(target.file, target.func)
                    )
                    made = _Way(KEY_OBJECT, built, call.lineno)
                    steps = ((f"[{index!r}]",) if index is not None else ()) + ((f".{field}",) if field else ())
                    found.append(made._replace(steps=steps))
                    continue
                generator = self._is_generator(target)
                if generator and not yielded:
                    found.append(_stop("generator", owner_of(target.file, target.func), at=here))
                    continue
                handed = bind(call, target, placed.on_class)
                callee = (owner_of(target.file, target.func), target.func.lineno)  # type: ignore[attr-defined]
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
                        for outer in self.read(given, call, file, func):
                            found.append(
                                outer._replace(
                                    steps=outer.steps + inner.steps,
                                    through=tuple(
                                        call._replace(
                                            steps=call.steps + len(outer.steps)
                                        )
                                        for call in inner.through
                                    )
                                    + outer.through,
                                    route=(here,) + inner.route + outer.route,
                                    changes=outer.changes + inner.changes,
                                )
                            )
                    else:
                        found.append(inner._replace(route=(here,) + inner.route))
            if placed.unplaced:
                found.append(
                    _stop("open", ast.unparse(call.func), what=placed.unplaced, at=here)
                )
            return found
        named = self._through(call, file, func)
        if named is not None:
            ways = [
                one._replace(
                    through=(named._replace(steps=len(one.steps)),) + one.through
                )
                for one in self.read(call.args[0], call, file, func)
            ]
        else:
            ways = [_Way(KEY_CALL, ast.unparse(call.func), call.lineno, what=placed.unplaced)]
        return self._of_field(ways, field)

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
        """The ways of what *target* returns -- of its *field*, read at each return, where one is asked."""
        key = (id(target.func), field)
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
            found += self._safely(_field_of(value, field), value, target.file, target.func)
        self._returns[key] = found
        return found

    def _returned_at(self, target: Target, index: int, field: Optional[str] = None) -> list[_Way]:
        """:meth:`_returned`'s, of each return's element at *index*: the element where a tuple or list literal gives
        it (:meth:`_element`), else the return's ways with the step ``[index]`` -- and its *field*, where one is
        asked."""
        key = (id(target.func), index, field)
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
                chosen = self._element(value, index, value, target.file, target.func, field)
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
                whole = self._safely(value, value, target.file, target.func)
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
        held = self._yields.get(id(target.func))
        if held is not None:
            return held
        self._yields[id(target.func)] = [
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
                found += self._safely(value, node, target.file, target.func)
            elif isinstance(node, ast.YieldFrom):
                found += self._stepped(self._safely(node.value, node, target.file, target.func), "[*]")
        self._yields[id(target.func)] = found
        return found

    def _safely(
        self, expr: ast.expr, at: ast.AST, file: str, func: ast.AST
    ) -> list[_Way]:
        try:
            return self.read(expr, at, file, func)
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

    def queue(self, sources: Iterable[HeldSource]) -> None:
        """The parameters *sources*, a record's ways, end at: to be read."""
        for source in sources:
            key = (source.site_owner, source.site_line or 0, source.name)
            if source.kind == KEY_PARAMETER and key not in self.parameters:
                self._pending.setdefault(key, self._functions[key])
            for change in source.changes:
                self.queue(change.value)

    def settle(self) -> None:
        """Read every parameter a record's way ends at, and the ones those reach."""
        while self._pending:
            key, (file, func) = self._pending.popitem()
            if key in self.parameters:
                continue
            made = self.parameters[key] = self._parameter_from(key, file, func)
            self.queue(made.sources)

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
                for one in self._safely(given, site.call, site.file, site.func)
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
        key = (one._replace(line=None), tuple(route))
        sources.setdefault(
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
            ),
        )
    return list(sources.values()), list(stops.values())


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
            if node.id in own or _bound_here(func, node.id):
                found.setdefault(node.id, node)
    return found


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
) -> tuple[list[GuardFrom], list[ParameterFrom], list[SetFrom], list[WriteFrom]]:
    """A :class:`GuardFrom` for each test on the way to each junction's arm, and to each site of *entered* (its
    :attr:`GuardFrom.calls`), and each name it reads (:func:`_names`); each of *writes*
    (:func:`sqlwrite.extract`'s) whose column a guard's way ends at -- through the ParameterFroms -- its ways read as
    the guards', and on a guard's way ending where one of its column's does, what may change the object between
    (:func:`_after_reads`); a :class:`WriteFrom` for the object each arm writes and the value each passthrough arm
    hands on (:func:`_written`), read as the guards' names are; and a :class:`ParameterFrom` for each parameter a
    way of any ends at, and those reach.  Nothing without *calls* or *record*: the reading places calls and reads
    fetches by them."""
    if calls is None or record is None:
        return [], [], [], []
    junctions = list(junctions)
    reader = _Reader(
        calls, record, read_sites, {junction.owner for junction in junctions}, trees
    )
    found: dict[tuple[str, int, int, str], GuardFrom] = {}
    written: dict[tuple[str, int, str, str, str], WriteFrom] = {}
    read_once: dict[int, tuple[list[HeldSource], list[KeyStop]]] = {}
    methods = _methods(trees)
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
            for test in pathcond.tests_on(stmt, func):
                for read, node in _names(test, func).items():
                    key = (junction.owner, test.lineno, test.col_offset, read)
                    guard = found.get(key)
                    if guard is None:
                        sources, stops = _said(reader._safely(node, node, file, func))
                        reader.queue(sources)
                        guard = found[key] = GuardFrom(
                            owner=junction.owner,
                            test_line=test.lineno,
                            test=ast.unparse(test),
                            name=read,
                            sources=sources,
                            stops=stops,
                        )
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
            for test in pathcond.tests_on(stmt, func):
                for read, node in _names(test, func).items():
                    key = (site.owner, test.lineno, test.col_offset, read)
                    guard = found.get(key)
                    if guard is None:
                        sources, stops = _said(reader._safely(node, node, file, func))
                        reader.queue(sources)
                        guard = found[key] = GuardFrom(
                            owner=site.owner,
                            test_line=test.lineno,
                            test=ast.unparse(test),
                            name=read,
                            sources=sources,
                            stops=stops,
                        )
                    if site.line not in guard.calls:
                        guard.calls.append(site.line)
    reader.settle()
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
