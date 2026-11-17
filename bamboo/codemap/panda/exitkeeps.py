"""What each exit of a function keeps a run from: the junction arms, the prints and the calls into arms a run leaving
there can no longer reach, and could have reached had it gone on past it (:class:`~bamboo.codemap.models.ExitKeeps`).

A reader of a pass that ran none of a junction's arms has two reasons to tell apart: the pass left by an exit before
the arms, or it went past every exit and a test on the way sent it another way.  The exits it may have left by are
the function's own ``return``, ``raise``, ``continue`` and ``break`` statements; what each keeps a run from is a
question on the flow the map reads values by (:func:`bamboo.codemap.panda.flow.kept_from`) -- a ``return`` or a
raise over the call's pass, a ``continue`` or a ``break`` over one turn of its loop -- and over one turn of each loop
holding it, for a reader whose pass is one turn of the loop its entry row's print is in (each print's loops are said
too, :class:`~bamboo.codemap.models.PrintLoops`).

The functions read are those holding an arm (located as :mod:`mustprints` locates a branch's statements) or a print
(:class:`~bamboo.codemap.models.FunctionPrintsNode`); every exit of each is listed, keeping a run from nothing or not.
A call into arms is an edge of the reach record whose callee holds an arm -- the function the record places an arm's
line in -- or reaches one through the record's edges.
"""

from __future__ import annotations

import ast
from typing import Optional, Sequence

from bamboo.codemap import reach
from bamboo.codemap.calls import def_name, own_nodes
from bamboo.codemap.models import (
    ExitKeeps,
    FunctionPrintsNode,
    JunctionNode,
    MapFragment,
    PrintLoops,
    SourceModule,
    TurnKeeps,
)
from bamboo.codemap.panda import flow, mustprints

#: The statements leaving a pass or a turn.
EXITS = (ast.Return, ast.Raise, ast.Continue, ast.Break)


class _Function:
    """One function read: its arms' and prints' statements, each with the lines it stands for."""

    __slots__ = ("func", "file", "arms", "prints", "calls")

    def __init__(self, func: ast.AST, file: str) -> None:
        self.func = func
        self.file = file
        #: id(statement) -> (statement, the branch lines it stands for)
        self.arms: dict[int, tuple[ast.stmt, set[int]]] = {}
        self.prints: dict[int, tuple[ast.stmt, set[int]]] = {}
        self.calls: dict[int, tuple[ast.stmt, set[int]]] = {}


def attach(fragment: MapFragment, modules: Sequence[SourceModule], record: Optional[reach.ReachGraph] = None) -> int:
    """Fill *fragment*'s :attr:`~bamboo.codemap.models.MapFragment.exit_keeps` and
    :attr:`~bamboo.codemap.models.MapFragment.print_loops`; the calls into arms from *record*, none without it.
    Returns how many exits keep a run from an arm."""
    trees = {module.rel_path: module.tree for module in modules}
    functions = _functions(fragment.junctions, fragment.function_prints, trees)
    if record is not None:
        _calls(functions, fragment.junctions, record)
    found: list[ExitKeeps] = []
    loops: list[PrintLoops] = []
    for held in sorted(functions.values(), key=lambda one: (one.file, getattr(one.func, "lineno", 0))):
        found += _keeps(held)
        loops += _print_loops(held)
    fragment.exit_keeps = found
    fragment.print_loops = loops
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


def _calls(functions: dict[int, _Function], junctions: Sequence[JunctionNode], record: reach.ReachGraph) -> None:
    """Each function's calls into arms: the lines of its row's edges whose callee holds an arm or reaches one."""
    holding = set()
    for junction in junctions:
        file = junction.owner.partition("::")[0]
        for branch in junction.branches:
            for line in [branch.line, *branch.other_lines]:
                owner = record.function_at(file, line) if line is not None else None
                if owner is not None:
                    holding.add(owner)
    callers: dict[str, set[str]] = {}
    for caller_row in record.rows.values():
        for call in caller_row.calls:
            callers.setdefault(call.to, set()).add(caller_row.owner)
    reaching = set(holding)
    stack = list(holding)
    while stack:
        for caller in callers.get(stack.pop(), ()):
            if caller not in reaching:
                reaching.add(caller)
                stack.append(caller)
    for held in functions.values():
        owner = record.function_at(held.file, getattr(held.func, "lineno", 0))
        row = record.rows.get(owner) if owner is not None else None
        if row is None:
            continue
        for call in row.calls:
            if call.to not in reaching:
                continue
            for line in call.lines or [call.line]:
                statement = mustprints._statement_at(held.func, line)
                if statement is not None:
                    held.calls.setdefault(id(statement), (statement, set()))[1].add(line)


def _defined_at(tree: ast.AST, line: int) -> Optional[ast.AST]:
    """The ``def`` starting at *line*."""
    func = mustprints._function_at(tree, line)
    return func if func is not None and func.lineno == line else None  # type: ignore[attr-defined]


def _keeps(held: _Function) -> list[ExitKeeps]:
    owner = f"{held.file}::{def_name(held.func)}"
    asked = [statement for statement, _lines in (*held.arms.values(), *held.prints.values(), *held.calls.values())]

    def lines(kept: set[int], of: dict[int, tuple[ast.stmt, set[int]]]) -> list[int]:
        return sorted({line for key, (_s, held_lines) in of.items() if key in kept for line in held_lines})

    found = []
    for node in own_nodes(held.func):
        if not isinstance(node, EXITS):
            continue
        kept = {id(statement) for statement in flow.kept_from(node, asked)} if asked else set()
        turns = []
        for loop in flow.loops_around(node):
            over = {id(statement) for statement in flow.kept_from(node, asked, over=loop)} if asked else set()
            turns.append(
                TurnKeeps(
                    loop=getattr(loop, "lineno", 0),
                    arms=lines(over, held.arms),
                    prints=lines(over, held.prints),
                    calls=lines(over, held.calls),
                )
            )
        found.append(
            ExitKeeps(
                owner=owner,
                exit=f"{type(node).__name__.lower()}@{node.lineno}",
                arms=lines(kept, held.arms),
                prints=lines(kept, held.prints),
                calls=lines(kept, held.calls),
                turns=turns,
            )
        )
    return sorted(found, key=lambda one: int(one.exit.rpartition("@")[2]))


def _print_loops(held: _Function) -> list[PrintLoops]:
    owner = f"{held.file}::{def_name(held.func)}"
    found = []
    for statement, lines in held.prints.values():
        loops = [getattr(loop, "lineno", 0) for loop in flow.loops_around(statement)]
        if loops:
            found += [PrintLoops(owner=owner, line=line, loops=loops) for line in sorted(lines)]
    return sorted(found, key=lambda one: one.line)
