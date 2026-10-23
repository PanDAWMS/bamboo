"""Which lines of a function can run before which, and between which a reset takes a change back.

:func:`bamboo.codemap.objects.may_change` says an assignment of a column on
an object changes the row a write of the object lands on.  It changes it
only where it can run before that write on one run, on the same object,
and -- for a write of the object's changed attributes
(``bindUpdateChangesExpression``, ``valuesMap(onlyChanged=True)``) -- where
no reset of the change (``resetChangedAttr``, ``resetChangedList``) stands
on every way between.  ``retryJob`` writes the job back at 4146 under ``if
not getNewPandaID`` and sets its ``parentID`` in the ``else`` at 4171;
``registerTaskInOneShot_JEDI`` resets ``taskSpec``'s ``jediTaskID`` at 5800
and writes its changed attributes at 5832.  Answered from the dump, the
order is read here, once per function, into its
:class:`~bamboo.codemap.models.ReachNode`:

* :attr:`~bamboo.codemap.models.ReachNode.order` -- of the lines of its
  assignments, writes and calls, which can run before which: on one pass, by
  a way with no jump away (:func:`reaching.reaching`, the innermost loop
  holding both as the end of a pass) and no test parting them
  (:func:`pathcond.parted`); or on a later pass of a loop holding both, by a
  way round it whose passes no test parts (``parted`` *around* the loop).
  The lines' order says most of it (:func:`bamboo.codemap.objects.by_line_order`),
  and the record keeps the loops, the lines whose statement starts
  elsewhere and the pairs where the order is wrong; and, for a pair meeting
  only on a later pass, the names a binding renews between (:func:`_renewed`):
  an object a later pass holds by one of them is another one;
* :attr:`~bamboo.codemap.models.ReachNode.cleared` -- for each object and
  the columns a set of resets takes back, the lines every way between which
  passes one (:func:`reaching.reaching`, the resets as killers), the entry
  and the normal exit among them.  A call of a function that resets what it
  is handed on every way from its entry to its normal exit is a reset where
  it is called -- ``resetRefinedAttrs`` resets ``splitRule`` and
  ``eventService``, and ``reserve_old_attributes``' names, by its call of it
  --, found to a fixed point (:func:`attach`).

A line is the statement starting there, the innermost, or else the statement
holding an expression starting there; lines in one statement are not
ordered.  A line in a nested function or a lambda is not read: its order is
its own function's.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from typing import NamedTuple, Optional

from bamboo.codemap import calls as C
from bamboo.codemap.models import (
    ENTRY_LINE,
    EXIT_LINE,
    ClearedLines,
    LineOrder,
    LinesApart,
    LineStart,
    LoopLines,
    ReachNode,
    Renewed,
)
from bamboo.codemap.objects import (
    APART,
    ONE_PASS,
    ROUND,
    by_line_order,
    line_paths,
    root_name,
)
from bamboo.codemap.panda import pathcond
from bamboo.codemap.panda.reaching import reaching_each, reaching_exit

#: The spec methods that take a changed attribute back out (one, by name) or all of them.
RESET_ONE = "resetChangedAttr"
RESET_ALL = "resetChangedList"
#: The column :class:`ClearedLines` says all of them by.
ALL_COLUMNS = "*"


class Reset(NamedTuple):
    """A statement taking back, on the object at *path*, the change of *columns* (``*`` for all)."""

    statement: ast.AST
    path: str
    columns: tuple[str, ...]


def reset_columns(call: ast.Call) -> Optional[tuple[str, ...]]:
    """The columns *call* resets, where it is a reset: ``("*",)`` for ``resetChangedList()``; for
    ``resetChangedAttr(name)``, the literal, or each literal str of the list a ``for`` binding the name runs over
    (``reserve_old_attributes``).  ``None`` where it is not a reset or its name is not read."""
    called = call.func
    if not isinstance(called, ast.Attribute):
        return None
    if called.attr == RESET_ALL and not call.args:
        return (ALL_COLUMNS,)
    if called.attr != RESET_ONE or len(call.args) != 1:
        return None
    argument = call.args[0]
    if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
        return (argument.value,)
    if not isinstance(argument, ast.Name):
        return None
    for ancestor in pathcond.ancestors(call):
        if isinstance(ancestor, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            return None
        if (
            isinstance(ancestor, (ast.For, ast.AsyncFor))
            and isinstance(ancestor.target, ast.Name)
            and ancestor.target.id == argument.id
        ):
            items = ancestor.iter
            if isinstance(items, (ast.List, ast.Tuple)) and all(
                isinstance(one, ast.Constant) and isinstance(one.value, str)
                for one in items.elts
            ):
                return tuple(one.value for one in items.elts)  # type: ignore[attr-defined]
            return None
    return None


def statements(func: ast.AST) -> dict[int, ast.AST]:
    """Each line of *func*'s own body and the statement it is read as (see the module docstring)."""
    found: dict[int, ast.AST] = {}
    for node in C.own_nodes(func):
        if not isinstance(node, ast.stmt):
            continue
        held = found.get(node.lineno)
        if held is None or (hasattr(held, "body") and not hasattr(node, "body")):
            found[node.lineno] = node
    for node in C.own_nodes(func):
        line = getattr(node, "lineno", None)
        if line is None or line in found or isinstance(node, ast.stmt):
            continue
        found[line] = pathcond.site_of(node)
    return found


def changes_and_writes(row: ReachNode) -> tuple[set[int], set[int]]:
    """The lines a change can be made at -- an assignment, a call -- and those a write can be made at -- a write,
    a call."""
    called = {line for call in row.calls for line in (call.lines or [call.line])}
    return {one.line for one in row.assigns} | called, {
        line for write in row.writes for line in write.lines
    } | called


def line_order(
    func: ast.AST, row: ReachNode, at: dict[int, ast.AST]
) -> Optional[LineOrder]:
    """:attr:`ReachNode.order` of *row*, its function *func*; None where it has fewer than two lines or nothing to say."""
    changes, writes = changes_and_writes(row)
    lines = sorted(line for line in changes | writes if line in at)
    if len(lines) < 2:
        return None
    looped = [
        node
        for node in C.own_nodes(func)
        if isinstance(node, (ast.For, ast.AsyncFor, ast.While)) and node.body
    ]
    held = [loop for loop in looped if any(_in_body(at[line], loop) for line in lines)]
    loops = [
        LoopLines(
            line=loop.lineno,
            start=loop.body[0].lineno,
            end=getattr(loop.body[-1], "end_lineno", None) or loop.body[-1].lineno,
            again=isinstance(loop, ast.While),
        )
        for loop in held
    ]
    starts = {line: at[line].lineno for line in lines}  # type: ignore[attr-defined]
    flows = _Flows(at, lines)
    apart: dict[int, list[int]] = defaultdict(list)
    rounds: dict[int, list[int]] = defaultdict(list)
    before: dict[int, list[int]] = defaultdict(list)
    met_round: list[tuple[int, int]] = []
    for one in lines:
        for other in lines:
            if one == other or at[one] is at[other]:
                continue
            said = by_line_order(starts[one], starts[other], loops)
            kind = _kind(at[one], at[other], flows)
            if kind == ROUND:
                met_round.append((one, other))
            if kind == said:
                continue
            if kind == APART:
                apart[one].append(other)
            elif kind == ROUND and said == ONE_PASS:
                rounds[one].append(other)
            elif kind == ONE_PASS:
                before[one].append(other)
            else:
                # A round where the order says apart: the record's loops hold the two lines (by_line_order), so
                # only a loop the record does not keep could say it; read as one pass, the safe side.
                before[one].append(other)
    found = LineOrder(
        loops=loops,
        starts=[
            LineStart(line=line, start=start)
            for line, start in starts.items()
            if start != line
        ],
        apart=[
            LinesApart(line=one, lines=others) for one, others in sorted(apart.items())
        ],
        rounds=[
            LinesApart(line=one, lines=others) for one, others in sorted(rounds.items())
        ],
        before=[
            LinesApart(line=one, lines=others) for one, others in sorted(before.items())
        ],
        renewed=_renewed(func, row, at, held, met_round),
    )
    if not (found.loops or found.starts or found.apart or found.before):
        return None
    return found


def _in_body(node: ast.AST, loop: ast.AST) -> bool:
    """Whether *loop*'s body holds *node*, by the lines the record keeps (:class:`LoopLines`)."""
    body = loop.body  # type: ignore[attr-defined]
    end = getattr(body[-1], "end_lineno", None) or body[-1].lineno
    return body[0].lineno <= getattr(node, "lineno", 0) <= end


class _Flows:
    """One function's ways between its record's lines, each target read once: every way, and the ways within
    one pass of a loop (its top a killer)."""

    def __init__(self, at: dict[int, ast.AST], lines: list[int]):
        seen: dict[int, ast.AST] = {}
        for line in lines:
            seen.setdefault(id(at[line]), at[line])
        self.events = list(seen.values())
        self._alive: dict[tuple[int, int], set[int]] = {}

    def reaches(
        self, one: ast.AST, other: ast.AST, loop: Optional[ast.AST] = None
    ) -> bool:
        within = id(loop) if loop is not None else 0
        alive = self._alive.get((within, id(other)))
        if alive is None:
            if loop is None:
                sites, killers = self.events, []
            else:
                sites = [one for one in self.events if _in_body(one, loop)]
                killers = [loop]
            for target, live in zip(
                sites, reaching_each(sites, sites, killers, sites), strict=True
            ):
                self._alive[(within, id(target))] = {id(site) for site in live}
            alive = self._alive.setdefault((within, id(other)), set())
        return id(one) in alive


def _kind(one: ast.AST, other: ast.AST, flows: _Flows) -> str:
    """Whether *one* can run before *other* on one pass, only on a later pass of a loop holding both, or not."""
    if isinstance(other, ast.While) and _in_body(one, other):
        # A while tests again after each pass of its body.
        if flows.reaches(one, other) and not pathcond.parted(one, other):
            return ONE_PASS
        return APART
    theirs = {id(loop) for loop in pathcond._loops_holding(other)}
    common = [loop for loop in pathcond._loops_holding(one) if id(loop) in theirs]
    within = common[0] if common else None
    if flows.reaches(one, other, within) and not pathcond.parted(one, other):
        return ONE_PASS
    if (
        common
        and flows.reaches(one, other)
        and any(not pathcond.parted(one, other, around=loop) for loop in common)
    ):
        return ROUND
    return APART


def _renewed(
    func: ast.AST,
    row: ReachNode,
    at: dict[int, ast.AST],
    loops: list[ast.AST],
    met_round: list[tuple[int, int]],
) -> list[Renewed]:
    """For each loop holding a pair that meets only on a later pass, the names it binds to another object between.

    The names are the roots of the paths the pairs' second lines take an
    object by (:func:`bamboo.codemap.objects.line_paths`): what a later pass
    writes, it takes by them.  Each is read with its bindings inside the loop
    that give it another object (:class:`_Afresh`) as killers, the first lines
    and the loop's top as sites that kill nothing: after a first line where it
    does not reach the top, before a second line the top does not reach.  A
    loop binding the name itself (its target) renews it at every top.  A name
    the first line binds is not read as renewed after it (what it hands a
    call is the value before).
    """
    if not met_round:
        return []
    paths = line_paths(row)
    found: list[Renewed] = []
    for loop in loops:
        pairs = [
            (one, other)
            for one, other in met_round
            if _in_body(at[one], loop) and _in_body(at[other], loop)
        ]
        if not pairs:
            continue
        firsts = sorted({one for one, _other in pairs})
        seconds = sorted({other for _one, other in pairs})
        names = {
            root_name(path) for line in seconds for path in paths.get(line, ())
        } - {"self", "cls"}
        afresh = _Afresh(func, loop, at)
        for name in sorted(names):
            if name.startswith("<return>"):
                continue
            killers = afresh.killers(name)
            if not killers:
                continue
            if any(killer is loop for killer in killers):
                found.append(
                    Renewed(loop=loop.lineno, name=name, before=seconds)  # type: ignore[attr-defined]
                )
                continue
            bound = {id(killer) for killer in killers}
            sites = [at[one] for one in firsts if id(at[one]) not in bound]
            sites.append(loop)
            top, *live = reaching_each(
                [loop.body[0], *(at[other] for other in seconds)],  # type: ignore[attr-defined]
                sites,
                killers,
                sites,
            )
            carried = {id(site) for site in top}
            after = [
                one
                for one in firsts
                if id(at[one]) not in bound and id(at[one]) not in carried
            ]
            before = [
                other
                for other, held in zip(seconds, live, strict=True)
                if not any(site is loop for site in held)
            ]
            if after or before:
                found.append(
                    Renewed(loop=loop.lineno, name=name, after=after, before=before)  # type: ignore[attr-defined]
                )
    return found


#: How a binding gives its name a value no expression spells: a loop's element, a ``with``'s, a caught exception.
_ELEMENT_KINDS = (pathcond.BIND_FOR, pathcond.BIND_WITH, pathcond.BIND_EXCEPT)
#: Values that are another object, or none: a literal, a display, a comprehension.
_LITERALS = (
    ast.Constant,
    ast.JoinedStr,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.Set,
    ast.ListComp,
    ast.DictComp,
    ast.SetComp,
    ast.GeneratorExp,
)


class _Afresh:
    """A loop's bindings that give a name another object on a pass: an element of what the loop or a ``with``
    holds, a call's result, a literal -- or a path rooted at a name the loop binds so (``tmpTaskSpec =
    impl.taskSpec``, ``impl`` a call's result).  ``job = self.job`` gives ``job`` the object every pass holds,
    and is not one.  A call's result, ``<result@line:col>``, is bound where its call runs."""

    def __init__(self, func: ast.AST, loop: ast.AST, at: dict[int, ast.AST]):
        self.func = func
        self.loop = loop
        self.at = at
        self._found: dict[str, list[ast.AST]] = {}
        self._reading: set[str] = set()

    def killers(self, name: str) -> list[ast.AST]:
        found = self._found.get(name)
        if found is not None:
            return found
        if name in self._reading:
            return []
        self._reading.add(name)
        if name.startswith("<result@"):
            line = int(name[len("<result@") :].split(":", 1)[0])
            held = self.at.get(line)
            found = [held] if held is not None and self._inside(held) else []
        else:
            found = [
                one.site
                for one in pathcond.bindings(self.func, name)
                if not one.nested
                and self._inside(one.site)
                and self._another(name, one)
            ]
        self._reading.discard(name)
        self._found[name] = found
        return found

    def _inside(self, site: ast.AST) -> bool:
        return site is self.loop or _in_body(site, self.loop)

    def _another(self, name: str, binding: pathcond.Binding) -> bool:
        if binding.kind in _ELEMENT_KINDS:
            return True
        if binding.kind == pathcond.BIND_AUGMENTED:
            return False
        value = binding.value
        if (
            binding.slot is not None
            and isinstance(value, (ast.Tuple, ast.List))
            and binding.slot < len(value.elts)
        ):
            value = value.elts[binding.slot]
        if isinstance(value, ast.Await):
            value = value.value
        if isinstance(value, (ast.Call, *_LITERALS)):
            return True
        while isinstance(value, (ast.Attribute, ast.Subscript)):
            value = value.value
        if isinstance(value, ast.Name) and value.id != name:
            return bool(self.killers(value.id))
        return False


def cleared(
    func: ast.AST, row: ReachNode, at: dict[int, ast.AST], resets: list[Reset]
) -> list[ClearedLines]:
    """:attr:`ReachNode.cleared` of *row*: for each object and the columns one set of *resets* takes back, the lines
    every way between which passes one -- from each change line or the entry, to each write or call line or the
    normal exit."""
    if not resets:
        return []
    changes, writes = changes_and_writes(row)
    by_path: dict[str, list[Reset]] = defaultdict(list)
    for one in resets:
        by_path[one.path].append(one)
    found: list[ClearedLines] = []
    for path, held in sorted(by_path.items()):
        groups: dict[tuple[int, ...], tuple[list[str], list[ast.AST]]] = {}
        for column in sorted({column for one in held for column in one.columns}):
            killers = [
                one.statement
                for one in held
                if column in one.columns or ALL_COLUMNS in one.columns
            ]
            key = tuple(sorted({id(one) for one in killers}))
            groups.setdefault(key, ([], killers))[0].append(column)
        for columns, killers in groups.values():
            found.extend(_cleared_by(func, path, columns, killers, changes, writes, at))
    return found


def _cleared_by(
    func: ast.AST,
    path: str,
    columns: list[str],
    killers: list[ast.AST],
    changes: set[int],
    writes: set[int],
    at: dict[int, ast.AST],
) -> list[ClearedLines]:
    entry = func.args  # type: ignore[attr-defined]
    killed = {id(one) for one in killers}
    sources = {
        line: at[line]
        for line in sorted(changes)
        if line in at and id(at[line]) not in killed
    }
    sites = [entry, *sources.values()]
    lines = {id(entry): ENTRY_LINE}
    for line, statement in sources.items():
        lines.setdefault(id(statement), line)
    gone: dict[int, list[int]] = defaultdict(list)
    targets = [(line, at[line]) for line in sorted(writes) if line in at]
    statements = [statement for _line, statement in targets]
    everywhere = reaching_each(statements, sites, (), sites)
    passing = reaching_each(statements, sites, killers, sites)
    for (line, _statement), free, kept in zip(
        targets, everywhere, passing, strict=True
    ):
        for key in {id(one) for one in free} - {id(one) for one in kept}:
            if lines[key] != line:
                gone[lines[key]].append(line)
    free = {id(one) for one in reaching_exit(func, sites, (), sites)}
    kept = {id(one) for one in reaching_exit(func, sites, killers, sites)}
    for key in free - kept:
        gone[lines[key]].append(EXIT_LINE)
    return [
        ClearedLines(path=path, columns=columns, line=line, lines=sorted(set(others)))
        for line, others in sorted(gone.items())
    ]


def clears(func: ast.AST, found: list[ClearedLines]) -> dict[str, set[str]]:
    """What a function resets on every way from its entry to its normal exit: each parameter (``self`` too), as it
    was handed, with the columns."""
    arguments = func.args  # type: ignore[attr-defined]
    given = {
        one.arg
        for one in [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]
    }
    taken: dict[str, set[str]] = defaultdict(set)
    for one in found:
        if one.line == ENTRY_LINE and EXIT_LINE in one.lines and one.path in given:
            taken[one.path].update(one.columns)
    return dict(taken)


def attach(
    functions: dict[str, ast.AST],
    rows: dict[str, ReachNode],
    own: dict[str, list[Reset]],
) -> None:
    """Fill every row's :attr:`~ReachNode.order` and :attr:`~ReachNode.cleared`.

    *functions* is each row's function, *own* the resets its own statements
    make.  A call handing a path to a function that resets it on every way
    (:func:`clears`) is a reset of that path where the call is, and such
    calls are found until none is new.
    """
    places = {owner: statements(func) for owner, func in functions.items()}
    summaries: dict[str, dict[str, set[str]]] = {}
    found: dict[str, list[ClearedLines]] = {}
    work = [owner for owner in sorted(own) if own[owner]]
    while work:
        changed: set[str] = set()
        for owner in work:
            resets = _resets(owner, rows[owner], places[owner], own, summaries)
            found[owner] = cleared(functions[owner], rows[owner], places[owner], resets)
            summary = clears(functions[owner], found[owner])
            if summary != summaries.get(owner, {}):
                summaries[owner] = summary
                changed.add(owner)
        work = sorted(
            owner
            for owner, row in rows.items()
            if owner in functions
            and any(call.to in changed for call in row.call_bindings)
        )
    for owner, func in functions.items():
        row = rows[owner]
        row.order = line_order(func, row, places[owner])
        row.cleared = found.get(owner, [])


def _resets(
    owner: str,
    row: ReachNode,
    at: dict[int, ast.AST],
    own: dict[str, list[Reset]],
    summaries: dict[str, dict[str, set[str]]],
) -> list[Reset]:
    """*owner*'s own resets, and its calls handing a path to a function that resets it on every way."""
    found = list(own.get(owner, []))
    for call in row.call_bindings:
        columns = summaries.get(call.to, {}).get(call.parameter)
        if columns and not call.at and call.line in at:
            found.append(Reset(at[call.line], call.path, tuple(sorted(columns))))
    return found
