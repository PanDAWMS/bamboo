"""What a name a guard reads holds, way by way, read back across calls: :class:`GuardFrom` and :class:`ParameterFrom`.

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
  through and named (*through*): whether it keeps the value is not read here;
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
"""

from __future__ import annotations

import ast
from typing import Iterable, NamedTuple, Optional

from bamboo.codemap.astwalk import walk
from bamboo.codemap.calls import Calls, CallSite, Target, bind, returns
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
    GuardFrom,
    HeldChange,
    HeldSource,
    JunctionNode,
    KeyStop,
    ParameterFrom,
    ReadSiteNode,
)
from bamboo.codemap.panda import changes as changes_mod
from bamboo.codemap.panda import pathcond

_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)
_STOP = "stop"


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
    through: tuple[str, ...] = ()
    route: tuple[str, ...] = ()
    #: In the order run.
    changes: tuple["_Change", ...] = ()


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
        self._returns: dict[int, list[_Way]] = {}
        self._active: set[tuple[int, int]] = set()
        #: ``(owner, def line, parameter)`` -> its ParameterFrom; the ones a record's way ends at, not yet read;
        #: and the function of each a way has ended at -- a callee's, filled at its call, ends in no record.
        self.parameters: dict[tuple[str, int, str], ParameterFrom] = {}
        self._pending: dict[tuple[str, int, str], tuple[str, ast.AST]] = {}
        self._functions: dict[tuple[str, int, str], tuple[str, ast.AST]] = {}
        #: ``(function, name, binding, statement read at)`` -> the changes between, by ids of the build's trees.
        self._hops: dict[tuple, tuple[_Change, ...]] = {}

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
            if isinstance(expr, ast.Subscript):
                index = expr.slice
                step = (
                    f"[{index.value!r}]" if isinstance(index, ast.Constant) else "[*]"
                )
            else:
                step = f".{expr.attr}"
            return [
                one._replace(steps=one.steps + (step,))
                for one in self.read(expr.value, at, file, func)
            ]
        found = self._sources(expr, at, file, func)
        return found or [
            _Way(
                KEY_ATTRIBUTE if isinstance(expr, ast.Attribute) else KEY_UNKNOWN,
                ast.unparse(expr),
                line,
            )
        ]

    def _name(
        self, expr: ast.Name, at: ast.AST, file: str, func: ast.AST
    ) -> list[_Way]:
        if expr.id in ("self", "cls"):
            return [_Way(KEY_SELF_FIELD, name=expr.id, line=expr.lineno)]
        if expr.id not in _parameters(func) and not _bound_here(func, expr.id):
            return [_Way(KEY_GLOBAL, name=expr.id, line=expr.lineno)]
        point = expr if getattr(expr, "parent", None) is not None else at
        last = _statement_of(point)
        found: list[_Way] = []
        for site in pathcond.held_at(func, expr.id, point).sites:  # type: ignore[arg-type]
            ways = self._bound(expr, point, site, file, func)
            changed = (
                ()
                if _unchanging(func, expr.id, site, ways)
                else self._changed(file, func, expr.id, site, last)
            )
            found += [
                one._replace(changes=one.changes + changed)
                if changed and one.kind != _STOP
                else one
                for one in ways
            ]
        return found

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
        # A loop's, a with's, a tuple's binding: what the carry says the name holds where it is read.
        return self._sources(expr, point, file, func) or [
            _Way(KEY_UNKNOWN, expr.id, getattr(site, "lineno", None))
        ]

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
                        for way in self._safely(
                            one.value, one.value, one.file, one.func
                        )
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
            held = self._hops[key] = tuple(made)
        return held

    def _call(self, call: ast.Call, file: str, func: ast.AST) -> list[_Way]:
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
                    found.append(_Way(KEY_OBJECT, built, call.lineno))
                    continue
                handed = bind(call, target, placed.on_class)
                callee = (owner_of(target.file, target.func), target.func.lineno)  # type: ignore[attr-defined]
                for inner in self._returned(target):
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
                                    through=inner.through + outer.through,
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
        if named:
            return [
                one._replace(through=(f"{named} {here}",) + one.through)
                for one in self.read(call.args[0], call, file, func)
            ]
        return [
            _Way(KEY_CALL, ast.unparse(call.func), call.lineno, what=placed.unplaced)
        ]

    def _through(self, call: ast.Call, file: str, func: ast.AST) -> str:
        """The name of a call outside the corpus that is handed one positional argument -- a function or a module's
        (``json.loads``), not a method of a value -- else ``""``."""
        if (
            len(call.args) != 1
            or isinstance(call.args[0], ast.Starred)
            or any(one.arg is None for one in call.keywords)
        ):
            return ""
        if isinstance(call.func, ast.Name):
            local = call.func.id in _parameters(func) or _bound_here(func, call.func.id)
            return "" if local else call.func.id
        if isinstance(call.func, ast.Attribute) and isinstance(
            call.func.value, ast.Name
        ):
            module = call.func.value.id
            if module in self.calls.types.imports(file) and not _bound_here(
                func, module
            ):
                return f"{module}.{call.func.attr}"
        return ""

    def _returned(self, target: Target) -> list[_Way]:
        held = self._returns.get(id(target.func))
        if held is not None:
            return held
        self._returns[id(target.func)] = [
            _stop(
                "cycle",
                owner_of(target.file, target.func),
                at=f"{target.file}:{getattr(target.func, 'lineno', '')}",
            )
        ]
        found: list[_Way] = []
        for value in returns(target.func):
            found += self._safely(value, value, target.file, target.func)
        self._returns[id(target.func)] = found
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


def _said(ways: list[_Way]) -> tuple[list[HeldSource], list[KeyStop]]:
    """*ways* as the record keeps them, each once; *route* and *through* in the order the value went."""
    sources: dict[tuple, HeldSource] = {}
    stops: dict[tuple, KeyStop] = {}
    for one in ways:
        route, through = list(reversed(one.route)), list(reversed(one.through))
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
                through=through,
                route=route,
                changes=[_change_said(change) for change in one.changes],
            ),
        )
    return list(sources.values()), list(stops.values())


#: The names an annotation spells a value no change in place can touch with: ``int``, ``str | None``,
#: ``Optional[datetime.datetime]``.
_IMMUTABLE = frozenset(
    {
        "int",
        "str",
        "float",
        "bool",
        "bytes",
        "complex",
        "None",
        "Optional",
        "Union",
        "Literal",
        "typing",
        "datetime",
        "date",
        "time",
        "timedelta",
        "Decimal",
        "decimal",
    }
)


def _unchanging(func: ast.AST, name: str, site: ast.AST, ways: list[_Way]) -> bool:
    """Whether nothing can change in place what *site* gives *name*: a parameter annotated with such types only, or
    a value every way of which ends at a literal or a column's value, through no call."""
    if site is getattr(func, "args", None):
        return _immutable(_annotation(func, name))
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


def _immutable(annotation: Optional[ast.expr]) -> bool:
    if annotation is None:
        return False
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        try:
            annotation = ast.parse(annotation.value, mode="eval").body
        except SyntaxError:
            return False
    names: list[str] = []
    for node in ast.walk(annotation):
        if isinstance(node, ast.Name):
            names.append(node.id)
        elif isinstance(node, ast.Attribute):
            names.append(node.attr)
        elif isinstance(node, ast.Constant) and node.value is None:
            names.append("None")
        elif isinstance(node, ast.Constant) and not isinstance(
            node.value, (str, int, float, bool)
        ):
            return False
    return bool(names) and all(one in _IMMUTABLE for one in names)


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


def _names(test: ast.expr, func: ast.AST) -> dict[str, ast.Name]:
    """The names *test* reads that are *func*'s parameters or locals -- not ``self``, a module's name or a
    comprehension's own -- each at its first read."""
    inner = {
        target.id
        for node in walk(test)
        if isinstance(node, ast.comprehension)
        for target in walk(node.target)
        if isinstance(target, ast.Name)
    }
    own = _parameters(func)
    found: dict[str, ast.Name] = {}
    for node in walk(test):
        if (
            isinstance(node, ast.Name)
            and node.id not in ("self", "cls")
            and node.id not in inner
        ):
            if node.id in own or _bound_here(func, node.id):
                found.setdefault(node.id, node)
    return found


def guard_froms(
    junctions: Iterable[JunctionNode],
    read_sites: Iterable[ReadSiteNode],
    trees: dict[str, ast.AST],
    calls: Optional[Calls],
    record,
) -> tuple[list[GuardFrom], list[ParameterFrom]]:
    """A :class:`GuardFrom` for each test on the way to each junction's arm and each name it reads, and a
    :class:`ParameterFrom` for each parameter a way ends at, and the ones those reach.  Nothing without *calls* or
    *record*: the reading places calls and reads fetches by them."""
    if calls is None or record is None:
        return [], []
    junctions = list(junctions)
    reader = _Reader(
        calls, record, read_sites, {junction.owner for junction in junctions}, trees
    )
    found: dict[tuple[str, int, int, str], GuardFrom] = {}
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
    reader.settle()
    for guard in found.values():
        guard.arms.sort()
    return (
        sorted(found.values(), key=lambda one: (one.owner, one.test_line, one.name)),
        sorted(
            reader.parameters.values(),
            key=lambda one: (one.owner, one.line, one.parameter),
        ),
    )
