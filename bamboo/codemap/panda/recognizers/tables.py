"""Each table the corpus names, with what the schema and the corpus say of it (:class:`TableRecord`).

A reader asking rows of a kind -- the live-case harness's fetch -- needs four facts it kept by hand: how a row is
named (the table's key, as the class spells it), which tables hold one class's rows and are read together (a
``UNION``'s arms), how a table is partitioned (an archive asked by ``modificationTime``), and when a row was made
(the class's creation column).  The schema says the first and the third, the corpus the second and the fourth.
"""

from __future__ import annotations

import ast
from typing import Callable, Iterable, Optional

from bamboo.codemap.models import (
    EntityNode,
    KeyFrom,
    ReadSiteNode,
    SourceModule,
    TableRecord,
)
from bamboo.codemap.panda import sql, sqltext, values
from bamboo.codemap.panda.attribution import SpecAttributor
from bamboo.codemap.panda.pathcond import functions_with_owner
from bamboo.codemap.panda.runs import HISTORY_TABLES

#: The database's clocks a statement can write.
_DATABASE_CLOCKS = sqltext.DATABASE_CLOCKS


def records(
    modules: Iterable[SourceModule],
    entities: Iterable[EntityNode],
    read_sites: Iterable[ReadSiteNode],
    key_froms: Iterable[KeyFrom],
    attributor: SpecAttributor,
    table_class: Callable[[str], Optional[str]],
    columns: dict[str, frozenset[str]],
    keys: dict[str, frozenset[str]],
    partitions: dict[str, tuple[str, tuple[str, ...]]],
    declaring: dict[str, tuple[str, ...]],
) -> list[TableRecord]:
    """One :class:`TableRecord` per table an entity, a read site, a write record or a family names, by its name
    lowered.  Read inside :func:`sql.across_calls` and :func:`sqltext.with_schema`, as the read sites were: a
    family is read off the texts each read site's run has."""
    modules, read_sites, key_froms = list(modules), list(read_sites), list(key_froms)
    spellings: dict[str, set[str]] = {}

    def name(spelled: str) -> None:
        if (
            spelled
            and "{" not in spelled
            and "$" not in spelled
            and not sqltext.is_hole(spelled)
        ):
            spellings.setdefault(spelled.lower(), set()).add(spelled)

    by_entity = {entity.name: entity for entity in entities}
    for entity in by_entity.values():
        for spelled in entity.tables:
            name(spelled)
    for site in read_sites:
        for spelled in site.tables:
            name(spelled)
    for key_from in key_froms:
        name(key_from.table)
    families = _families(modules, read_sites)
    for member in families:
        spellings.setdefault(member, set())
    classes = {
        cls: attributor.declared_attributes(cls)
        for cls in sorted(attributor.declared_classes())
    }
    created = _creation(key_froms, classes)
    found = []
    for table in sorted(spellings):
        held = columns.get(table, frozenset())
        rows_of = [
            cls
            for cls, names in classes.items()
            if names and {n.lower() for n in names} <= held
        ]
        cls = rows_of[0] if len(rows_of) == 1 else ""
        spelled = {n.lower(): n for n in classes.get(cls, ())}
        key = sorted(keys.get(table, ()))
        identity = sorted(sqltext.table_identity(table))
        kind, by = partitions.get(table, ("", ()))
        entity = by_entity.get(table)
        found.append(
            TableRecord(
                name=table,
                spellings=sorted(spellings[table]),
                schema_names=list(declaring.get(table, ())),
                key=key,
                identity=identity,
                identity_attrs=[
                    f"{cls}.{spelled[c]}" for c in identity if c in spelled
                ],
                partition_kind=kind,
                partitioned_by=list(by),
                partitioned_by_attrs=[
                    f"{cls}.{spelled[c]}" for c in by if c in spelled
                ],
                rows_of=cls,
                spec_class=(entity.spec_class if entity is not None else None)
                or table_class(table)
                or "",
                inserted=bool(entity is not None and entity.created_by),
                family=sorted(families.get(table, set()) - {table}),
                creation=created.get(cls, []) if cls else [],
                history_of=HISTORY_TABLES.get(table, ""),
            )
        )
    return found


def _families(
    modules: list[SourceModule], read_sites: list[ReadSiteNode]
) -> dict[str, set[str]]:
    """``{table: its family}``: the tables a read site's texts' ``UNION`` arms read in one place
    (:func:`sqltext.union_places`), joined across statements."""
    functions: dict[tuple[str, str], list[ast.AST]] = {}
    for module in modules:
        for func, _owner in functions_with_owner(module.tree):
            functions.setdefault((module.rel_path, func.name), []).append(func)
    parent: dict[str, str] = {}

    def root(table: str) -> str:
        parent.setdefault(table, table)
        while parent[table] != table:
            parent[table] = parent[parent[table]]
            table = parent[table]
        return table

    for site in read_sites:
        line = site.anchor.line_start
        owned = [
            func
            for func in functions.get(
                (site.anchor.file, site.owner.rsplit("::", 1)[-1]), ()
            )
            if func.lineno <= line <= (func.end_lineno or func.lineno)
        ]
        if not owned:
            continue
        texts = {
            run.reading for run in sql.executions(owned[-1]) if run.call.lineno == line
        }  # type: ignore[arg-type]
        for text in texts:
            for a, b in sqltext.union_places(text):
                parent[root(a)] = root(b)
    joined: dict[str, set[str]] = {}
    for table in list(parent):
        joined.setdefault(root(table), set()).add(table)
    return {table: members for members in joined.values() for table in members}


def _creation(
    key_froms: list[KeyFrom], classes: dict[str, set[str]]
) -> dict[str, list[str]]:
    """``{class: its attributes set when a row is made and never after}``: every write record writing the subject
    is an ``INSERT``'s (``new_row`` said, or why not), and every write puts the time now -- a clock, no offset,
    in Python (:func:`values.stated`) or in the statement (``CURRENT_DATE``)."""
    writes: dict[str, list[tuple[KeyFrom, int]]] = {}
    for key_from in key_froms:
        for subject, lines in key_from.writes.items():
            writes.setdefault(subject, []).extend((key_from, line) for line in lines)
    found: dict[str, list[str]] = {}
    for cls, names in classes.items():
        for attribute in sorted(names):
            listed = writes.get(f"{cls}.{attribute}", [])
            if listed and all(
                _is_insert(record) and _now(record, attribute, line)
                for record, line in listed
            ):
                found.setdefault(cls, []).append(f"{cls}.{attribute}")
    return found


def _is_insert(record: KeyFrom) -> bool:
    return record.new_row is not None or bool(record.new_row_by)


def _now(record: KeyFrom, attribute: str, line: int) -> bool:
    """Whether *record*'s write of *attribute* at *line* puts the time now: an assignment to the field there, or the
    value the statement's ``INSERT`` gives the column."""
    func = (record._runs or record._binds or (None,))[0]
    if func is None:
        return False
    for node in ast.walk(func):
        if getattr(node, "lineno", None) != line or not isinstance(
            node, (ast.Assign, ast.AnnAssign)
        ):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(
            isinstance(t, ast.Attribute) and t.attr.lower() == attribute.lower()
            for t in targets
        ):
            return node.value is not None and _clock(node.value, func)
    told = False
    for run in record._executions:
        for written in sqltext.writes(run.reading):
            value = next(
                (
                    v
                    for c, v in written.columns.items()
                    if c.lower() == attribute.lower()
                ),
                None,
            )
            if written.kind != "insert" or value is None:
                continue
            if (
                value.kind == "expression"
                and value.text.strip().upper() in _DATABASE_CLOCKS
            ):
                told = True
            elif value.kind == "bind" and run.varmap is not None:
                bound = sql.bound_values(func, run.varmap, value.text, at=run)
                if not bound or not all(_clock(bind.value, func) for bind in bound):
                    return False
                told = True
            else:
                return False
    return told


def _clock(expression: ast.expr, func: ast.AST) -> bool:
    stated = values.stated(expression, func)
    return values.is_clock(stated.base) and stated.offset == 0
