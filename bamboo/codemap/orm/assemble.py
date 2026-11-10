"""Subjects, entities and junctions from resolved field writes, named as the PanDA map names them."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Optional

from bamboo.codemap.models import Anchor, Branch, EntityNode, JunctionNode, SubjectNode
from bamboo.codemap.orm.resolve import Resolved
from bamboo.codemap.orm.schema import DjangoModel
from bamboo.codemap.orm.writes import CREATING
from bamboo.codemap.panda import pathcond

#: Weakest first is last: a junction is as attributed as its weakest arm.
ATTRIBUTION_RANK = ("certain", "structural", "heuristic", "unresolved")


@dataclass
class FieldWrite:
    """One write of a field with its values resolved -- from the ORM scan or a plugin's own form (raw SQL)."""

    model: Optional[str]
    field: str
    #: ``<rel_path>::<qualified function>``, the PanDA map's owner form
    owner: str
    rel_path: str
    line: int
    form: str
    attribution: str
    arms: list[Resolved]
    #: the write's own path condition: enclosing tests and the early exits above it
    path_condition: list[str] = field(default_factory=list)
    structural: Optional[str] = None


def write_condition(at: ast.AST) -> list[str]:
    """The conditions on reaching *at*: its arm's tests, then the early exits above it, by line."""
    seen: list[str] = []
    for text in pathcond.path_condition(at) + pathcond.exits_above(at):
        if text not in seen:
            seen.append(text)
    return seen


def assemble(
    writes: list[FieldWrite],
    models: dict[str, DjangoModel],
    map_id: str,
    derived_from: str,
    shas: dict[str, str],
    selected: Optional[dict[tuple[str, str], set[str]]] = None,
) -> tuple[list[SubjectNode], list[EntityNode], list[JunctionNode], int]:
    """``(subjects, entities, junctions, unresolved_writes)``; one junction per (owner, subject)."""
    selected = selected or {}
    subjects: dict[str, SubjectNode] = {}
    entities: dict[str, EntityNode] = {}
    groups: dict[tuple[str, str], list[FieldWrite]] = {}
    unresolved = 0
    for w in writes:
        subject = SubjectNode.make_name(w.model or "?", w.field)
        groups.setdefault((w.owner, subject), []).append(w)
        if w.model is None:
            unresolved += 1
            continue
        info = models[w.model]
        if subject not in subjects:
            subjects[subject] = SubjectNode(
                name=subject,
                map_id=map_id,
                derived_from=derived_from,
                spec_class=w.model,
                attribute=w.field,
                vocabulary=list(info.fields.get(w.field, [])),
                selected_values=sorted(selected.get((w.model, w.field), set())),
                metadata={"table": info.db_table, "module": info.module},
            )
        entity = entities.setdefault(
            w.model,
            EntityNode(
                name=f"{map_id}::{w.model}",
                map_id=map_id,
                derived_from=derived_from,
                spec_class=w.model,
                tables=[info.db_table] if info.db_table else [],
            ),
        )
        bucket = entity.created_by if w.form in CREATING else entity.updated_by
        if w.owner not in bucket:
            bucket.append(w.owner)

    junctions: list[JunctionNode] = []
    for (owner, subject), group in sorted(groups.items()):
        group.sort(key=lambda w: w.line)
        first = group[0]
        attribution = max((w.attribution for w in group), key=ATTRIBUTION_RANK.index)
        structural = {w.structural for w in group if w.structural}
        branches: list[Branch] = []
        for w in group:
            for arm in w.arms:
                branches.append(
                    Branch(
                        outcome=arm.outcome,
                        path_condition=[
                            *w.path_condition,
                            *(c for c in arm.conditions if c not in w.path_condition),
                        ],
                        order=len(branches),
                        tier=arm.tier,
                        line=arm.line or w.line,
                        tags=[w.form],
                    )
                )
        junctions.append(
            JunctionNode(
                name=JunctionNode.make_name(map_id, subject, owner),
                map_id=map_id,
                derived_from=derived_from,
                subject=subject,
                owner=owner,
                attribution=attribution,
                structural_subject=(
                    f"{structural.pop()}.{first.field}"
                    if len(structural) == 1
                    else None
                ),
                anchor=Anchor(
                    package=first.rel_path.split("/")[0],
                    file=first.rel_path,
                    line_start=first.line,
                    line_end=group[-1].line,
                    blob_sha=shas.get(first.rel_path),
                ),
                branches=branches,
            )
        )
    return (
        sorted(subjects.values(), key=lambda s: s.name),
        sorted(entities.values(), key=lambda e: e.name),
        junctions,
        unresolved,
    )
