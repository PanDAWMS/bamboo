"""What a printed map may hold where it is printed: each entry and what fills it (:class:`MapEntry`).

``tmpLog.debug(sqlU + sql + comment + str(varMap))`` prints the statement and the map it runs with, and a reader
holding the record reads each value back against what fills its entry -- ``':status': 'running'`` is
``taskSpec.status``.  :func:`entries` gives the entries of a map at a statement, and the lines where the map may get
entries the source does not spell; :attr:`SkeletonHole.entries` and :attr:`SkeletonHole.entries_open` hold them for
a hole printing a map whole.

**The binding the print sees.**  The bindings of the name that reach the statement (:func:`reaching`) are the maps
it may print.  An entry counts where it is filled on a path to the statement with no binding of the name between,
and is not filled again before it: ``varMap[":status"] = ...``, or the entry of a literal bound (``varMap =
{":status": ...}``).  A key the source computes (``varMap[crKey] = v``) is kept with its spelling, and a copy of a
spec's ``valuesMap()`` entry by entry (:func:`row_copies`), or the map bound to one, says the spec whose row it holds.

**Where it opens.**  Where something on a path to the statement may put an entry the source does not spell, its line
is said: a method putting into a dict (``update``, ``setdefault``), ``varMap |= other``, ``varMap[k] += v``, the map
handed to a call or put where another name reaches it (``maps.append(varMap)``, ``other = varMap``), and a binding to
anything but a map built here -- a call's value, a copy, a loop's target, a parameter (0).  Read, not put into:
indexing, a dict's other methods, a builtin function's argument, a logger's, the cursor's ``execute``
(:data:`sql.CURSOR_RUNNERS`), which hands the map to the driver.

A name is read as a map where a binding reaching the statement builds one (a literal, ``dict()``, a comprehension, a
``valuesMap()``) or an entry filled reaches it; elsewhere :func:`entries` says ``None``.
"""

from __future__ import annotations

import ast
from typing import Optional

from bamboo.codemap import callfacts
from bamboo.codemap.calls import own_nodes
from bamboo.codemap.models import MapEntry
from bamboo.codemap.panda import pathcond
from bamboo.codemap.panda.reaching import reaching

#: A dict's methods that put entries into it.
_PUTTING = frozenset(
    called
    for (kind, called), method in callfacts.BUILTIN_METHODS.items()
    if kind == "dict" and method.puts
)

#: The cursor's methods running a statement with the map they are handed (:data:`sql.CURSOR_RUNNERS`).
_RUNNERS = frozenset({"execute", "executemany"})

#: What a hole's expression is read through to the map it prints whole.
_WHOLE = frozenset({"str", "repr"})


def printed_map(expression: ast.expr) -> str:
    """The name a hole prints whole: ``varMap`` for ``str(varMap)``, ``repr(varMap)`` or ``varMap``."""
    if (
        isinstance(expression, ast.Call)
        and isinstance(expression.func, ast.Name)
        and expression.func.id in _WHOLE
        and len(expression.args) == 1
        and not expression.keywords
    ):
        expression = expression.args[0]
    return expression.id if isinstance(expression, ast.Name) else ""


def entries(
    func: ast.AST, name: str, at: ast.stmt
) -> Optional[tuple[list[MapEntry], list[int]]]:
    """The entries map *name* may hold at statement *at* of *func*, and the lines where it may get entries the
    source does not spell; ``None`` where *name* is not read as a map there."""
    fills = _fills(func).get(name, [])
    bound = [
        binding
        for binding in pathcond.bindings(func, name)
        if not binding.nested and binding.kind != pathcond.BIND_AUGMENTED
    ]
    if not fills and not any(_builds(binding.value) for binding in bound):
        return None
    rebinds: list[ast.AST] = [binding.site for binding in bound]
    arguments = getattr(func, "args", None)
    if arguments is not None and any(
        isinstance(arg, ast.arg) and arg.arg == name for arg in ast.walk(arguments)
    ):
        rebinds.append(arguments)
    live = reaching(at, rebinds)
    values = {id(binding.site): binding.value for binding in bound}
    found: list[MapEntry] = []
    opened: set[int] = set()
    constant: dict[str, list[ast.AST]] = {}
    computed: list[ast.AST] = []
    for site in live:
        value = values.get(id(site))
        if site is arguments:
            opened.add(0)
        elif isinstance(value, ast.Dict):
            for spelled, held in zip(value.keys, value.values, strict=True):
                if spelled is None:
                    opened.add(getattr(site, "lineno", 0))
                elif isinstance(spelled, ast.Constant) and isinstance(
                    spelled.value, str
                ):
                    constant.setdefault(spelled.value, [])
                else:
                    found.append(
                        _entry(ast.unparse(spelled), held, site, computed=True)
                    )
        elif isinstance(value, ast.Call) and _spec_of(value):
            found.append(_entry("", value, site, computed=True, row=_spec_of(value)))
        else:
            if isinstance(value, ast.Call):
                for keyword in value.keywords:
                    if keyword.arg is not None:
                        constant.setdefault(keyword.arg, [])
            if not _builds(value) or _given(value):
                opened.add(getattr(site, "lineno", 0))
    for fill in fills:
        slot = _filled(fill, name)
        if isinstance(slot, ast.Constant) and isinstance(slot.value, str):
            constant.setdefault(slot.value, []).append(fill)
        else:
            computed.append(fill)
    for key, stores in constant.items():
        written = stores + [
            site for site in rebinds if _holds(values.get(id(site)), key)
        ]
        for site in reaching(
            at,
            written,
            [site for site in rebinds if all(site is not one for one in written)],
        ):
            filled_with = _value(site, key)
            if filled_with is not None:
                found.append(_entry(key, filled_with, site))
    copies = {id(store): spec for spec, store in row_copies(func, name).items()}
    adding = computed + _opening(func, name)
    for site in reaching(at, adding, rebinds, rereads=adding):
        slot = _filled(site, name)
        if (
            slot is not None
            and isinstance(site, (ast.Assign, ast.AnnAssign))
            and site.value is not None
        ):
            spec = copies.get(id(site))
            row = _spec_of(spec) if spec is not None else ""
            found.append(
                _entry(ast.unparse(slot), site.value, site, computed=True, row=row)
            )
        else:
            opened.add(getattr(site, "lineno", 0))
    found.sort(key=lambda entry: (entry.line, entry.key))
    return found, sorted(opened)


def _fills(func: ast.AST) -> dict[str, list[ast.AST]]:
    """Each map name of *func* and the statements filling an entry of it (:func:`_filled`), read once and kept on
    *func*: a function asks it of every hole printing a name."""
    held: Optional[dict[str, list[ast.AST]]] = getattr(func, "_mapentries_fills", None)
    if held is None:
        held = {}
        for node in own_nodes(func):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                target = (
                    node.targets[0] if isinstance(node, ast.Assign) else node.target
                )
                if isinstance(target, ast.Subscript) and isinstance(
                    target.value, ast.Name
                ):
                    if _filled(node, target.value.id) is not None:
                        held.setdefault(target.value.id, []).append(node)
        func._mapentries_fills = held  # type: ignore[attr-defined]
    return held


def row_copies(func: ast.AST, name: str) -> dict[ast.expr, ast.AST]:
    """Each ``name[k] = v`` in ``for k, v in X.valuesMap(...).items()``, under the ``X.valuesMap(...)`` it copies."""
    found: dict[ast.expr, ast.AST] = {}
    for loop in own_nodes(func):
        if not (
            isinstance(loop, ast.For)
            and isinstance(loop.iter, ast.Call)
            and isinstance(loop.iter.func, ast.Attribute)
            and loop.iter.func.attr == "items"
            and not loop.iter.args
            and isinstance(copied := loop.iter.func.value, ast.Call)
            and isinstance(copied.func, ast.Attribute)
            and copied.func.attr == "valuesMap"
            and isinstance(loop.target, ast.Tuple)
            and len(loop.target.elts) == 2
            and all(isinstance(part, ast.Name) for part in loop.target.elts)
        ):
            continue
        key, value = (part.id for part in loop.target.elts)  # type: ignore[attr-defined]
        for node in loop.body:
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(target := node.targets[0], ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == name
                and isinstance(target.slice, ast.Name)
                and target.slice.id == key
                and isinstance(node.value, ast.Name)
                and node.value.id == value
            ):
                found[copied] = node
    return found


def _entry(
    key: str, value: ast.expr, site: ast.AST, computed: bool = False, row: str = ""
) -> MapEntry:
    return MapEntry(
        key=key,
        expression=ast.unparse(value),
        line=getattr(site, "lineno", 0),
        computed=computed,
        row=row,
    )


def _filled(node: ast.AST, name: str) -> Optional[ast.expr]:
    """The key *node* fills an entry of map *name* at: ``varMap[":status"] = v``, annotated or not."""
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
    elif isinstance(node, ast.AnnAssign) and node.value is not None:
        target = node.target
    else:
        return None
    if (
        isinstance(target, ast.Subscript)
        and isinstance(target.value, ast.Name)
        and target.value.id == name
    ):
        return target.slice
    return None


def _builds(value: Optional[ast.expr]) -> bool:
    """Whether *value* builds a map here: a literal, a comprehension, ``dict(...)``, a spec's ``valuesMap()``."""
    if isinstance(value, (ast.Dict, ast.DictComp)):
        return True
    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == "dict"
    ):
        return True
    return bool(_spec_of(value))


def _given(value: Optional[ast.expr]) -> bool:
    """Whether a map *value* builds holds entries no key here spells: a comprehension's, ``dict(other)``'s; a
    ``valuesMap()`` is a spec's row."""
    if isinstance(value, ast.DictComp):
        return True
    return (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and bool(value.args)
    )


def _spec_of(value: Optional[ast.expr]) -> str:
    """The spec whose ``valuesMap()`` *value* is: ``taskSpec`` for ``taskSpec.valuesMap(useSeq=True)``."""
    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr == "valuesMap"
    ):
        return ast.unparse(value.func.value)
    return ""


def _holds(value: Optional[ast.expr], key: str) -> bool:
    """Whether a bound literal *value* holds an entry *key*: ``{":status": ...}`` or ``dict(status=...)``."""
    if isinstance(value, ast.Dict):
        return any(
            isinstance(one, ast.Constant) and one.value == key for one in value.keys
        )
    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == "dict"
    ):
        return any(keyword.arg == key for keyword in value.keywords)
    return False


def _value(site: ast.AST, key: str) -> Optional[ast.expr]:
    """What *site* fills entry *key* with: its store's value, or the entry of the literal it binds."""
    if not isinstance(site, (ast.Assign, ast.AnnAssign)) or site.value is None:
        return None
    target = site.targets[0] if isinstance(site, ast.Assign) else site.target
    value = site.value
    if isinstance(target, ast.Subscript):
        return value
    if isinstance(value, ast.Dict):
        held = zip(value.keys, value.values, strict=True)
        return next(
            (
                one
                for spelled, one in held
                if isinstance(spelled, ast.Constant) and spelled.value == key
            ),
            None,
        )
    if isinstance(value, ast.Call):
        return next(
            (keyword.value for keyword in value.keywords if keyword.arg == key), None
        )
    return None


def _opening(func: ast.AST, name: str) -> list[ast.AST]:
    """The statements of *func* that may put into map *name* an entry the source does not spell."""
    found: list[ast.AST] = []
    for node in own_nodes(func):
        if isinstance(node, ast.AugAssign) and (
            (isinstance(node.target, ast.Name) and node.target.id == name)
            or (
                isinstance(node.target, ast.Subscript)
                and isinstance(node.target.value, ast.Name)
                and node.target.value.id == name
            )
        ):
            found.append(node)
        elif (
            isinstance(node, ast.Name)
            and node.id == name
            and isinstance(node.ctx, ast.Load)
            and _lets_out(node)
        ):
            statement = pathcond.site_of(node)
            if isinstance(statement, ast.stmt) and all(
                statement is not one for one in found
            ):
                found.append(statement)
    return found


def _lets_out(node: ast.Name) -> bool:
    """Whether the map *node* reads may get entries where it is used: put into, handed on, or bound elsewhere."""
    parent = getattr(node, "parent", None)
    if isinstance(parent, ast.Subscript) and parent.value is node:
        return False
    if isinstance(parent, ast.Attribute) and parent.value is node:
        called = getattr(parent, "parent", None)
        return (
            isinstance(called, ast.Call)
            and called.func is parent
            and parent.attr in _PUTTING
        )
    if isinstance(parent, ast.keyword):
        if parent.arg is None:
            return False
        parent = getattr(parent, "parent", None)
    if isinstance(parent, ast.Call) and parent.func is not node:
        called = parent.func
        if isinstance(called, ast.Attribute) and (
            called.attr in callfacts.LOG_METHODS or called.attr in _RUNNERS
        ):
            return False
        return callfacts.builtin_function_at(parent) is None
    return not isinstance(
        parent,
        (
            ast.FormattedValue,
            ast.JoinedStr,
            ast.BinOp,
            ast.Compare,
            ast.BoolOp,
            ast.UnaryOp,
            ast.If,
            ast.While,
            ast.IfExp,
            ast.Return,
            ast.Assert,
            ast.Expr,
            ast.comprehension,
            ast.For,
            ast.AsyncFor,
        ),
    )
