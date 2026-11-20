"""Django models as the code declares them: classes, tables, fields and allowed values."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Iterable, Optional

#: Base classes that make a class a Django model.
MODEL_BASES = frozenset({"models.Model", "Model", "django.db.models.Model"})
#: Base classes of a Django enumeration (``class Status(models.TextChoices)``).
CHOICES_BASES = frozenset(
    {
        "models.TextChoices",
        "TextChoices",
        "models.IntegerChoices",
        "IntegerChoices",
        "models.Choices",
        "Choices",
    }
)


def is_state_field(name: str, declares_choices: bool) -> bool:
    """A field that holds a state: one with ``choices=``, or named ``status``, ``state`` or ``*_status``."""
    return declares_choices or name in ("status", "state") or name.endswith("_status")


@dataclass
class DjangoModel:
    """One model class, its own and inherited fields."""

    name: str
    module: str
    rel_path: str
    bases: list[str]
    #: field -> its declared allowed values, ``[]`` where it declares none
    fields: dict[str, list[str]] = field(default_factory=dict)
    #: fields that declare ``choices=`` (even where the values could not be read)
    has_choices: set[str] = field(default_factory=set)
    #: class-level choice lists and enumerations, ``STATUS_CHOICES`` / ``Status`` -> values
    choice_sets: dict[str, list[str]] = field(default_factory=dict)
    #: field -> its literal ``default=``, which a create that leaves the field out writes
    defaults: dict[str, str] = field(default_factory=dict)
    db_table: Optional[str] = None
    abstract: bool = False

    def state_fields(self) -> set[str]:
        return {f for f in self.fields if is_state_field(f, f in self.has_choices)}


def dotted(node: ast.AST) -> Optional[str]:
    """``a.b.c`` for an attribute chain on a name, else None."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def choice_values(
    node: ast.AST, constants: dict[str, list[str]]
) -> Optional[list[str]]:
    """The stored values of a choices list, an ``__in`` list or a constant naming one; None where the code
    does not state them.  A ``(value, label)`` pair contributes its value."""
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    if isinstance(node, ast.Attribute):
        # Status.choices, Model.STATUS_CHOICES, Model.Status.choices
        head = dotted(node)
        if head is not None:
            parts = head.split(".")
            for i in range(len(parts), 0, -1):
                key = ".".join(parts[:i])
                if key in constants and (
                    i == len(parts)
                    or parts[i:] == ["choices"]
                    or parts[i:] == ["values"]
                ):
                    return constants[key]
        return None
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        out: list[str] = []
        for elt in node.elts:
            if isinstance(elt, (ast.Tuple, ast.List)) and elt.elts:
                elt = elt.elts[0]
            if isinstance(elt, ast.Constant) and isinstance(elt.value, (str, int)):
                out.append(str(elt.value))
            elif isinstance(elt, ast.Name) and len(constants.get(elt.id, [])) == 1:
                out.append(constants[elt.id][0])
            else:
                return None
        return out
    return None


def _enum_members(cls: ast.ClassDef) -> list[tuple[str, str]]:
    """``(member, value)`` of ``DONE = "done", "Done"`` / ``DONE = "done"`` in a TextChoices/IntegerChoices class."""
    out: list[tuple[str, str]] = []
    for stmt in cls.body:
        if not (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
        ):
            continue
        value = stmt.value
        if isinstance(value, ast.Tuple) and value.elts:
            value = value.elts[0]
        if isinstance(value, ast.Constant) and isinstance(value.value, (str, int)):
            out.append((stmt.targets[0].id, str(value.value)))
    return out


def module_constants(tree: ast.Module) -> dict[str, list[str]]:
    """Module- and class-level ``NAME = 'x'``, ``NAME = [('x', ..), ..]`` and Django enumerations.

    Class-level names are recorded both bare and as ``Class.NAME``, enumerations as ``Class`` and
    ``Outer.Class``, so ``choices=STATUS_CHOICES``, ``choices=Task.Status.choices`` and
    ``Task.STATUS_CHOICES`` all read the same values.
    """
    constants: dict[str, list[str]] = {}

    def visit(body: list[ast.stmt], prefix: str) -> None:
        for stmt in body:
            if (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
            ):
                name, value = stmt.targets[0].id, stmt.value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    found: Optional[list[str]] = [value.value]
                else:
                    found = choice_values(value, constants)
                if found is not None:
                    constants[name] = found
                    if prefix:
                        constants[f"{prefix}{name}"] = found
            elif isinstance(stmt, ast.ClassDef):
                bases = {dotted(b) or "" for b in stmt.bases}
                if bases & CHOICES_BASES:
                    members = _enum_members(stmt)
                    for qual in {stmt.name, f"{prefix}{stmt.name}"}:
                        constants[qual] = [v for _m, v in members]
                        for member, v in members:
                            constants[f"{qual}.{member}"] = [v]
                    continue  # its members are (value, label) pairs, not lists of values
                visit(stmt.body, f"{prefix}{stmt.name}.")

    visit(tree.body, "")
    return constants


def _is_field_call(func: str) -> bool:
    tail = func.split(".")[-1]
    return func.startswith("models.") and (
        tail.endswith("Field") or tail.endswith("Key")
    )


def read_models(
    modules: Iterable[tuple[str, str, ast.Module]],
) -> dict[str, DjangoModel]:
    """Every concrete-or-abstract Django model the modules declare, by class name.

    *modules* is ``(rel_path, dotted_module, tree)``.  Fields of abstract bases are inherited; a model without
    ``Meta.db_table`` gets Django's default ``<app>_<model>``.
    """
    raw: dict[str, DjangoModel] = {}
    for rel, module, tree in modules:
        constants = module_constants(tree)
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            info = DjangoModel(
                node.name, module, rel, [dotted(b) or "" for b in node.bases]
            )
            for key, values in constants.items():
                if key.startswith(f"{node.name}."):
                    info.choice_sets[key[len(node.name) + 1 :]] = values
            for stmt in node.body:
                if isinstance(stmt, ast.ClassDef) and stmt.name == "Meta":
                    for m in stmt.body:
                        if isinstance(m, ast.Assign) and isinstance(
                            m.targets[0], ast.Name
                        ):
                            if m.targets[0].id == "db_table" and isinstance(
                                m.value, ast.Constant
                            ):
                                info.db_table = m.value.value
                            if m.targets[0].id == "abstract" and isinstance(
                                m.value, ast.Constant
                            ):
                                info.abstract = bool(m.value.value)
                    continue
                target = value = None
                if (
                    isinstance(stmt, ast.Assign)
                    and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name)
                ):
                    target, value = stmt.targets[0].id, stmt.value
                elif (
                    isinstance(stmt, ast.AnnAssign)
                    and isinstance(stmt.target, ast.Name)
                    and stmt.value
                ):
                    target, value = stmt.target.id, stmt.value
                if (
                    target is None
                    or not isinstance(value, ast.Call)
                    or not _is_field_call(dotted(value.func) or "")
                ):
                    continue
                choices: list[str] = []
                for kw in value.keywords:
                    if kw.arg == "choices":
                        info.has_choices.add(target)
                        local = {
                            **constants,
                            **{
                                f"{node.name}.{k}": v
                                for k, v in info.choice_sets.items()
                            },
                        }
                        choices = choice_values(kw.value, local) or []
                    elif (
                        kw.arg == "default"
                        and isinstance(kw.value, ast.Constant)
                        and isinstance(kw.value.value, str)
                    ):
                        info.defaults[target] = kw.value.value
                info.fields[target] = choices
            raw[node.name] = info

    def is_model(name: str, seen: tuple[str, ...] = ()) -> bool:
        info = raw.get(name)
        if info is None or name in seen:
            return False
        return any(
            b in MODEL_BASES or is_model(b.split(".")[-1], seen + (name,))
            for b in info.bases
        )

    models: dict[str, DjangoModel] = {}
    for name, info in raw.items():
        if not is_model(name):
            continue
        stack = [b.split(".")[-1] for b in info.bases]
        seen: set[str] = set()
        while stack:
            parent_name = stack.pop()
            parent = raw.get(parent_name)
            if parent is None or parent_name in seen:
                continue
            seen.add(parent_name)
            for f, ch in parent.fields.items():
                info.fields.setdefault(f, ch)
            info.has_choices |= parent.has_choices
            for f, d in parent.defaults.items():
                info.defaults.setdefault(f, d)
            for k, v in parent.choice_sets.items():
                info.choice_sets.setdefault(k, v)
            stack.extend(b.split(".")[-1] for b in parent.bases)
        if info.db_table is None:
            info.db_table = f"{info.module.split('.')[0]}_{name.lower()}"
        models[name] = info
    return models
