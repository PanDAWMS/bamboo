"""ORM write forms in one function, each attributed to the model it writes.

Recognised:

* ``obj.<field> = value`` (saved later by ``obj.save()``).  The model is
  ``certain`` where the function binds ``obj`` from ``Model.objects...get()`` /
  ``create()`` / ``first()``, ``Model(...)``, ``get_object_or_404(Model, ..)``,
  a loop over a ``Model.objects`` queryset, an annotation, or ``self`` inside
  the model; ``structural`` where exactly one model declares every field the
  function touches on ``obj``; ``heuristic`` where the variable's name ends a
  model's name; else ``unresolved``.
* ``Model.objects....update(f=v)``, ``.create(f=v)``, ``update_or_create`` and
  ``get_or_create`` (keywords and ``defaults=``), also through a queryset
  variable -- ``certain``.
* ``Model.objects.bulk_update(objs, ['f', ..])`` -- ``certain``, the value is
  whatever the objects hold (no expression).
* A create that leaves a field with a literal ``default=`` out --
  ``.objects.create`` / ``get_or_create`` / ``update_or_create`` or
  ``Model(...)`` -- writes the default (form ``orm.<method>(default)``).
* ``F('other')`` as a value is kept as the expression; :mod:`.resolve` reads it
  as a copy of another column.

The scan only finds and attributes; the value and the conditions a write holds
under are :mod:`.resolve`'s, read from the statement with the shared
``pathcond`` machinery, so every write carries the same kind of condition the
PanDA map's do.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Iterator, Optional

from bamboo.codemap.orm.schema import DjangoModel, choice_values, dotted, is_state_field

SINGLE_OBJECT = frozenset(
    {"get", "create", "first", "last", "latest", "earliest", "get_object_or_404"}
)
ORM_WRITE = frozenset(
    {"update", "create", "update_or_create", "get_or_create", "bulk_update"}
)
ORM_SELECT = frozenset({"filter", "exclude", "get"})
#: Write forms that make a row rather than change one.
CREATING = frozenset(
    {
        "orm.create",
        "orm.get_or_create",
        "orm.create(default)",
        "orm.get_or_create(default)",
        "orm.update_or_create(default)",
        "orm.<init>(default)",
    }
)
#: Calls that make a row and so write each default they are not given.
_DEFAULTED = frozenset({"create", "get_or_create", "update_or_create", "<init>"})


@dataclass
class OrmWrite:
    """One write of a field, before its value is resolved."""

    model: Optional[str]
    field: str
    #: the written expression; None where the form gives none (``bulk_update``)
    value: Optional[ast.expr]
    #: the node the write happens at: the assignment, or the ORM call
    at: ast.AST
    form: str
    attribution: str
    structural: Optional[str] = None

    @property
    def line(self) -> int:
        return getattr(self.at, "lineno", 0)


@dataclass
class FunctionScan:
    writes: list[OrmWrite] = field(default_factory=list)
    #: (model, field) -> the values filters select rows on
    selected: dict[tuple[str, str], set[str]] = field(default_factory=dict)


def orm_chain(node: ast.AST) -> tuple[Optional[str], list[str]]:
    """``Model.objects.filter(..).update(..)`` -> (``Model``, [filter, update]); a chain on a plain
    name -> (None, [..., ``<name>``])."""
    methods: list[str] = []
    while True:
        if isinstance(node, ast.Call):
            node = node.func
            continue
        if isinstance(node, ast.Attribute):
            if node.attr == "objects" and isinstance(node.value, ast.Name):
                return node.value.id, list(reversed(methods))
            methods.append(node.attr)
            node = node.value
            continue
        if isinstance(node, ast.Name):
            return None, list(reversed(methods)) + [f"<{node.id}>"]
        return None, []


def _calls_in_chain(node: ast.AST) -> Iterator[ast.Call]:
    while True:
        if isinstance(node, ast.Call):
            yield node
            node = node.func
        elif isinstance(node, ast.Attribute):
            node = node.value
        else:
            return


def _own_expressions(stmt: ast.stmt) -> Iterator[ast.AST]:
    """The statement's own expressions, not those of nested blocks."""
    for name, value in ast.iter_fields(stmt):
        if name in ("body", "orelse", "finalbody", "handlers", "cases"):
            continue
        if isinstance(value, ast.AST):
            yield value
        elif isinstance(value, list):
            yield from (v for v in value if isinstance(v, ast.AST))


def _nested_blocks(stmt: ast.stmt) -> Iterator[list[ast.stmt]]:
    for name in ("body", "orelse", "finalbody"):
        block = getattr(stmt, name, None)
        if isinstance(block, list):
            yield block
    for handler in getattr(stmt, "handlers", []) or []:
        yield handler.body
    for case in getattr(stmt, "cases", []) or []:
        yield case.body


def _heuristic(models: dict[str, DjangoModel], var: str, attr: str) -> Optional[str]:
    stem = var.lower().lstrip("_").rstrip("s")
    if len(stem) < 3:
        return None
    hits = [
        m.name
        for m in models.values()
        if not m.abstract and attr in m.fields and m.name.lower().endswith(stem)
    ]
    return hits[0] if len(hits) == 1 else None


def scan_function(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    cls: Optional[str],
    models: dict[str, DjangoModel],
    constants: dict[str, list[str]],
    state_only: bool = True,
) -> FunctionScan:
    """The ORM writes and filter reads of *func*'s own body (nested functions are their own).

    *cls* is the enclosing class's name, so ``self`` inside a model is that model.  With *state_only* only
    state-like fields (:func:`~bamboo.codemap.orm.schema.is_state_field`) are recorded.
    """
    scan = FunctionScan()
    bound: dict[str, str] = {}  # var -> model (one object)
    querysets: dict[str, str] = {}  # var -> model (a queryset)
    touched: dict[str, set[str]] = {}  # var -> attributes touched
    pending: list[tuple[str, str, ast.expr, ast.stmt, Optional[str]]] = []

    def wanted(model: DjangoModel, name: str) -> bool:
        return name in model.fields and (not state_only or name in model.state_fields())

    if cls in models:
        for a in func.args.args[:1]:
            bound[a.arg] = cls  # type: ignore[assignment]
    for a in func.args.args + func.args.kwonlyargs:
        if a.annotation is not None:
            ann = (dotted(a.annotation) or "").split(".")[-1]
            if ann in models:
                bound[a.arg] = ann

    def bind(target: ast.AST, value: ast.AST) -> None:
        if isinstance(value, ast.Call):
            model, methods = orm_chain(value)
            fname = (dotted(value.func) or "").split(".")[-1]
            if model is None and fname in models:
                model, methods = fname, ["<init>"]
            if model is None and fname == "get_object_or_404" and value.args:
                model = (dotted(value.args[0]) or "").split(".")[-1] or None
                methods = ["get"]
            if model in models:
                single = (
                    not methods
                    or methods[-1] in SINGLE_OBJECT
                    or methods[-1] == "<init>"
                )
                if (
                    isinstance(target, ast.Tuple)
                    and methods
                    and methods[-1] in ("get_or_create", "update_or_create")
                ):
                    if target.elts and isinstance(target.elts[0], ast.Name):
                        bound[target.elts[0].id] = model  # type: ignore[assignment]
                elif isinstance(target, ast.Name):
                    (bound if single else querysets)[target.id] = model  # type: ignore[assignment]
                    (querysets if single else bound).pop(target.id, None)
                return
        if isinstance(target, ast.Name):
            bound.pop(target.id, None)
            querysets.pop(target.id, None)

    def defaults(model: str, given: set[str], call: ast.Call, method: str) -> None:
        info = models[model]
        if info.abstract:
            return
        for fld, default in sorted(info.defaults.items()):
            if fld not in given and wanted(info, fld):
                scan.writes.append(
                    OrmWrite(
                        model,
                        fld,
                        ast.Constant(default),
                        call,
                        f"orm.{method}(default)",
                        "certain",
                    )
                )

    def visit_call(call: ast.Call) -> None:
        if isinstance(call.func, ast.Name) and call.func.id in models:
            # Model(...): a row, saved later, holding each default it is not given
            defaults(
                call.func.id, {kw.arg for kw in call.keywords if kw.arg}, call, "<init>"
            )
            return
        fname = call.func.attr if isinstance(call.func, ast.Attribute) else None
        if fname is None:
            return
        model, methods = orm_chain(call)
        if model is None and methods and methods[-1].startswith("<"):
            model = querysets.get(methods[-1][1:-1])
        if model not in models:
            return
        info = models[model]  # type: ignore[index]
        for c in _calls_in_chain(call):
            if isinstance(c.func, ast.Attribute) and c.func.attr in ORM_SELECT:
                for kw in c.keywords:
                    if not kw.arg:
                        continue
                    fld, _, lookup = kw.arg.partition("__")
                    if fld not in info.state_fields():
                        continue
                    vals: list[str] = []
                    if lookup == "in":
                        vals = choice_values(kw.value, constants) or []
                    elif not lookup:
                        if isinstance(kw.value, ast.Constant) and isinstance(
                            kw.value.value, str
                        ):
                            vals = [kw.value.value]
                        elif (
                            isinstance(kw.value, ast.Name)
                            and len(constants.get(kw.value.id, [])) == 1
                        ):
                            vals = constants[kw.value.id]
                    scan.selected.setdefault((model, fld), set()).update(vals)  # type: ignore[arg-type]
        if fname not in ORM_WRITE:
            return
        if fname == "bulk_update":
            names = (
                call.args[1]
                if len(call.args) > 1
                else next((k.value for k in call.keywords if k.arg == "fields"), None)
            )
            for fld in (
                (choice_values(names, constants) or []) if names is not None else []
            ):
                if wanted(info, fld):
                    scan.writes.append(
                        OrmWrite(model, fld, None, call, "orm.bulk_update", "certain")
                    )
            return
        kwargs: list[tuple[str, ast.expr]] = []
        for kw in call.keywords:
            if kw.arg == "defaults" and isinstance(kw.value, ast.Dict):
                for k, v in zip(kw.value.keys, kw.value.values, strict=True):
                    if isinstance(k, ast.Constant) and isinstance(k.value, str):
                        kwargs.append((k.value, v))
            elif kw.arg and "__" not in kw.arg:
                kwargs.append((kw.arg, kw.value))
        for fld, value in kwargs:
            if wanted(info, fld):
                scan.writes.append(
                    OrmWrite(model, fld, value, call, f"orm.{fname}", "certain")
                )
        if fname in _DEFAULTED and not any(
            kw.arg is None for kw in call.keywords
        ):  # no **kwargs
            defaults(model, {f for f, _v in kwargs} | {kw.arg.partition("__")[0] for kw in call.keywords if kw.arg}, call, fname)  # type: ignore[arg-type]

    def visit(stmts: list[ast.stmt]) -> None:
        for stmt in stmts:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            for expr in _own_expressions(stmt):
                for node in ast.walk(expr):
                    if isinstance(node, ast.Attribute) and isinstance(
                        node.value, ast.Name
                    ):
                        touched.setdefault(node.value.id, set()).add(node.attr)
                    if isinstance(node, ast.Call):
                        visit_call(node)
            if isinstance(stmt, (ast.Assign, ast.AnnAssign)) and stmt.value is not None:
                targets = (
                    stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
                )
                for target in targets:
                    if isinstance(target, ast.Attribute) and isinstance(
                        target.value, ast.Name
                    ):
                        # Attribution is settled once every touched attribute is known, but the binding is
                        # captured now: a later rebinding must not re-attribute it.
                        var = target.value.id
                        pending.append(
                            (var, target.attr, stmt.value, stmt, bound.get(var))
                        )
                    else:
                        bind(target, stmt.value)
            if isinstance(stmt, (ast.For, ast.AsyncFor)) and isinstance(
                stmt.target, ast.Name
            ):
                model, _ = orm_chain(stmt.iter)
                if model in models:
                    bound[stmt.target.id] = model  # type: ignore[assignment]
                elif isinstance(stmt.iter, ast.Name) and stmt.iter.id in querysets:
                    bound[stmt.target.id] = querysets[stmt.iter.id]
            for block in _nested_blocks(stmt):
                visit(block)

    visit(func.body)

    all_fields = {f for m in models.values() for f in m.fields}
    for var, attr, value, stmt, at_bind in pending:
        if var == "self" and at_bind is None:
            continue  # an attribute of a non-model class: not a row write
        attrs = (touched.get(var, set()) & all_fields) | {attr}
        classes = sorted(
            m.name for m in models.values() if not m.abstract and attrs <= set(m.fields)
        )
        structural = classes[0] if len(classes) == 1 else None
        if at_bind is not None:
            model, how = at_bind, "certain"
        elif structural is not None:
            model, how = structural, "structural"
        else:
            model = _heuristic(models, var, attr)
            how = "heuristic" if model else "unresolved"
        if model is not None and not wanted(models[model], attr):
            continue
        if model is None and not is_state_field(attr, False):
            continue
        scan.writes.append(
            OrmWrite(model, attr, value, stmt, "attribute", how, structural)
        )
    scan.writes.sort(key=lambda w: w.line)
    return scan
