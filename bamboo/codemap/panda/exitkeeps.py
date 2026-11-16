"""What each exit of a function keeps a run from: the junction arms and the prints a run leaving there can no longer
reach, and could have reached had it gone on past it (:class:`~bamboo.codemap.models.ExitKeeps`).

A reader of a pass that ran none of a junction's arms has two reasons to tell apart: the pass left by an exit before
the arms, or it went past every exit and a test on the way sent it another way.  The exits it may have left by are
the function's own ``return``, ``raise``, ``continue`` and ``break`` statements; what each keeps a run from is a
question on the flow the map reads values by (:func:`bamboo.codemap.panda.flow.kept_from`) -- a ``return`` or a
raise over the call's pass, a ``continue`` or a ``break`` over one turn of its loop.

The functions read are those holding an arm (located as :mod:`mustprints` locates a branch's statements) or a print
(:class:`~bamboo.codemap.models.FunctionPrintsNode`); every exit of each is listed, keeping a run from nothing or not.
"""

from __future__ import annotations

import ast
from typing import Optional, Sequence

from bamboo.codemap.calls import own_nodes
from bamboo.codemap.models import (
    ExitKeeps,
    FunctionPrintsNode,
    JunctionNode,
    MapFragment,
    SourceModule,
)
from bamboo.codemap.panda import flow, mustprints

#: The statements leaving a pass or a turn.
EXITS = (ast.Return, ast.Raise, ast.Continue, ast.Break)


class _Function:
    """One function read: its arms' and prints' statements, each with the lines it stands for."""

    __slots__ = ("func", "file", "arms", "prints")

    def __init__(self, func: ast.AST, file: str) -> None:
        self.func = func
        self.file = file
        #: id(statement) -> (statement, the branch lines it stands for)
        self.arms: dict[int, tuple[ast.stmt, set[int]]] = {}
        self.prints: dict[int, tuple[ast.stmt, set[int]]] = {}


def attach(fragment: MapFragment, modules: Sequence[SourceModule]) -> int:
    """Fill *fragment*'s :attr:`~bamboo.codemap.models.MapFragment.exit_keeps`.  Returns how many exits keep a run
    from an arm."""
    trees = {module.rel_path: module.tree for module in modules}
    functions = _functions(fragment.junctions, fragment.function_prints, trees)
    found: list[ExitKeeps] = []
    for held in sorted(functions.values(), key=lambda one: (one.file, getattr(one.func, "lineno", 0))):
        found += _keeps(held)
    fragment.exit_keeps = found
    return sum(1 for one in found if one.arms)


def _functions(
    junctions: Sequence[JunctionNode], function_prints: Sequence[FunctionPrintsNode], trees: dict[str, ast.AST]
) -> dict[int, _Function]:
    functions: dict[int, _Function] = {}
    for junction in junctions:
        for branch in junction.branches:
            sites = mustprints._sites(junction, branch, trees)
            if isinstance(sites, str):
                continue
            for site in sites:
                held = functions.setdefault(id(site.func), _Function(site.func, site.owner.partition("::")[0]))
                held.arms.setdefault(id(site.statement), (site.statement, set()))[1].add(site.line)
    for node in function_prints:
        file = node.owner.partition("::")[0]
        tree = trees.get(file)
        if tree is None or node.anchor is None:
            continue
        mustprints._parented(tree)
        func = _defined_at(tree, node.anchor.line_start)
        if func is None:
            continue
        held = functions.setdefault(id(func), _Function(func, file))
        for statement in node.statements:
            found = mustprints._statement_at(func, statement.line)
            if found is not None:
                held.prints.setdefault(id(found), (found, set()))[1].add(statement.line)
    return functions


def _defined_at(tree: ast.AST, line: int) -> Optional[ast.AST]:
    """The ``def`` starting at *line*."""
    func = mustprints._function_at(tree, line)
    return func if func is not None and func.lineno == line else None  # type: ignore[attr-defined]


def _keeps(held: _Function) -> list[ExitKeeps]:
    owner = f"{held.file}::{getattr(held.func, 'name', '')}"
    asked = [statement for statement, _lines in (*held.arms.values(), *held.prints.values())]
    found = []
    for node in own_nodes(held.func):
        if not isinstance(node, EXITS):
            continue
        kept = {id(statement) for statement in flow.kept_from(node, asked)} if asked else set()
        found.append(
            ExitKeeps(
                owner=owner,
                exit=f"{type(node).__name__.lower()}@{node.lineno}",
                arms=sorted({line for key, (_s, lines) in held.arms.items() if key in kept for line in lines}),
                prints=sorted({line for key, (_s, lines) in held.prints.items() if key in kept for line in lines}),
            )
        )
    return sorted(found, key=lambda one: int(one.exit.rpartition("@")[2]))
