"""What a function prints, and which of its lines a run through a given site must or cannot show.

Three readings of one function, all from the tree:

* **the row of a print statement** -- how to recognise it in a record.  The
  map's skeleton pattern comes first.  Where the map refused one, and only
  there, a pattern is built here (``outside:3-4``, the querier's own stand-in
  for a pattern the map does not give): literals escaped, holes left open, and a name or
  attribute the function binds to a literal before the print expanded into
  it.  Short literals are not refused, because every record read here was
  already found by the entity key.  ``capture`` is the separate reading of an
  f-string's holes, one group per hole, for values (outside part 2).
* **must rows** of a site -- the print statements that run on every path
  through it in one call: the ones before it in each block that encloses it,
  and the ones after it until something could jump or raise.  A handler
  implies nothing about the ``try`` body it caught, unless a traceback frame
  says which line raised.
* **cannot rows** -- when the path from a site ends in ``return``/``raise``,
  the prints further down the function, which that call can no longer reach.

The map's ``arms`` are compatibility of path conditions and say nothing about
an early ``return`` nested inside a branch, so they are only used in the
direction that is sound: a printed row whose ``arms`` leave a site out.
Anything two sites share a loop over is left alone -- another iteration can
print what this one did not.
"""
from __future__ import annotations

import ast
import functools
import hashlib
import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable

from bamboo.codemap import pathcond, trace
from bamboo.codemap.models import PrintRow
from bamboo.codemap.panda.plugin import PandaCodeMapPlugin
from bamboo.codemap.panda.provenance import classify
from bamboo.codemap.panda.recognizers.selection import log_level

from . import TREE, mapdump
from .records import CORPUS, LEVEL_OF_METHOD, LEVEL_RANK, Record, Run, head_word

#: the bamboo tree the map's code is imported from (the package's own; see :mod:`bamboo.querier`)
BAMBOO_TREE = str(TREE)

ROOTS = PandaCodeMapPlugin._resolve_roots(CORPUS)
LOOPS = (ast.For, ast.AsyncFor, ast.While)
EXITS = (ast.Return, ast.Raise)
JUMPS = (ast.Return, ast.Raise, ast.Continue, ast.Break)
DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
#: a row proves a path only when it says something -- a pattern of fewer
#: literal characters than this can be present by accident, and proves only
#: as a whole message under a mark one function writes (``Seen.proves``)
MIN_PROOF = 8
#: where a function's print rows come from: ``walk`` (the skeleton of the map's walk, ``trace.Asker.start``, per
#: asking) or ``graph`` (the map's ``FunctionPrintsNode``), whose
#: rows carry no arms -- those are computed here as the walk computes them.  ``graph``
#: stops when the map read another
#: version of a file than the corpus holds (``anchor.blob_sha``)
ROWS_FROM = os.environ.get("ROWS_FROM", "graph")
PATTERN_MODES = ("map", "outside")


#: id of every module ``tree`` parsed -> its corpus path, to tell a corpus function from one built from a string
TREE_REL: dict[int, str] = {}


@functools.cache
def tree(rel: str) -> ast.Module:
    from bamboo.codemap.calls import parsed as map_parsed

    package, _, relative = rel.partition("/")
    held = map_parsed(ROOTS[package] / relative) if package in ROOTS and relative else None
    if held is None:
        parsed = ast.parse((CORPUS / rel).read_text())
        pathcond.attach_parents(parsed)
    else:
        parsed = held[1]
    TREE_REL[id(parsed)] = rel
    return parsed


def functions(rel: str, name: str) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    return [f for f, _ in pathcond.functions_with_owner(tree(rel)) if f.name == name]


def own_nodes(root: ast.AST) -> Iterable[ast.AST]:
    """``ast.walk`` that does not go into nested functions, classes or lambdas."""
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, DEFS):
                stack.append(child)


def statement_at(func: ast.AST, line: int) -> ast.stmt | None:
    """The innermost statement of *func* that starts on *line* (the first in ``own_nodes``' order of equal depth).

    The function's statements are indexed by line once and the index kept on its node, as ``pathcond`` keeps each
    node's parent: ``which_ran.rows_around`` asks for a pair of lines at a time, and walked the whole function anew
    for each (superStatus 53354352: 64% of its time)."""
    index = getattr(func, "statements_at", None)
    if index is None:
        index = {}
        for node in own_nodes(func):
            if isinstance(node, ast.stmt) and node is not func:
                found = index.get(node.lineno)
                if found is None or _depth(node) > _depth(found):
                    index[node.lineno] = node
        func.statements_at = index  # type: ignore[attr-defined]
    return index.get(line)


def _depth(node: ast.AST) -> int:
    return sum(1 for _ in pathcond.ancestors(node))


def block_of(parent: ast.AST, child: ast.AST) -> tuple[str, list] | None:
    for name in ("body", "orelse", "finalbody", "handlers"):
        block = getattr(parent, name, None)
        if isinstance(block, list) and child in block:
            return name, block
    return None


def contains(stmt: ast.AST, kinds: tuple) -> bool:
    return any(isinstance(n, kinds) for n in own_nodes(stmt))


def is_log_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in LEVEL_OF_METHOD
        and bool(node.args)
        and log_level(node.args[0]) is not None
    )


def is_print(stmt: ast.AST) -> bool:
    return isinstance(stmt, ast.Expr) and is_log_call(stmt.value)


def may_divert(stmt: ast.stmt) -> bool:
    """Whether control can leave *stmt* other than by falling through it."""
    if isinstance(stmt, JUMPS) or contains(stmt, JUMPS):
        return True
    return any(isinstance(n, ast.Call) and not is_log_call(n) for n in own_nodes(stmt))


def loops_over(node: ast.AST, func: ast.AST) -> set[int]:
    found = set()
    previous = node
    for ancestor in pathcond.ancestors(node):
        if ancestor is func:
            break
        if isinstance(ancestor, LOOPS) and previous in ancestor.body:
            found.add(id(ancestor))
        previous = ancestor
    return found


def share_loop(a: ast.AST, b: ast.AST, func: ast.AST, once: Iterable[int] = ()) -> bool:
    """Whether *a* and *b* sit in one loop of *func*, which a pass may go round between them; *once* (ids, as
    :func:`once_loops` gives them): loops the pass went round once, which do not count."""
    return bool((loops_over(a, func) & loops_over(b, func)) - set(once))


def completes(block: list) -> bool:
    """Whether a run of *block* may reach its end, rather than leave by a return, raise, break or continue on every
    way through it; True where unsure (a ``with`` may swallow what its body raised, a loop may end)."""
    if not block:
        return True
    last = block[-1]
    if isinstance(last, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
        return False
    if isinstance(last, ast.If):
        return completes(last.body) or completes(last.orelse)
    if isinstance(last, (ast.Try, getattr(ast, "TryStar", ast.Try))):
        normal = completes(last.body) and completes(last.orelse)
        return (normal or any(completes(h.body) for h in last.handlers)) and completes(last.finalbody)
    return True


def once_loops(func: ast.AST, absent) -> dict[int, str]:
    """``{id(loop): why}``: the loops of *func* one pass went round once.  No way through the body reaches its end
    (:func:`completes`), and before each ``continue`` of the loop a row inside it must print (:func:`must_rows`) that
    *absent* (a print statement -> bool) says the pass did not print -- the DB proxy's retry, ``for iTry in
    range(nTry): try: ... return ... except: ... tmp_log.debug(f"retry: {iTry}"); continue`` (activateJob), with no
    "retry" in the run.  A ``continue`` with no such row, or a body that may run to its end, leaves the loop out:
    the pass may have gone round again."""
    out = {}
    for loop in own_nodes(func):
        if not isinstance(loop, LOOPS) or completes(loop.body):
            continue
        rows = []
        for c in own_nodes(loop):
            if isinstance(c, ast.Continue) and _loop_of(c, func) is loop:
                gone = [s.lineno for s in must_rows(func, c)[0] if pathcond._within(s, loop) and absent(s)]
                if not gone:
                    break
                rows.append(gone[0])
        else:
            out[id(loop)] = (f"the loop at {loop.lineno} went round once: its way back prints row(s) {sorted(rows)}, "
                             "not in the run" if rows else f"the loop at {loop.lineno} cannot go round: no way back")
    return out


def free_jumps(func: ast.AST, lines) -> set[int]:
    """The lines of *func*'s ``continue`` and ``break`` statements whose loop holds none of *lines* (the arms a pass is
    judged on): taken, such a jump keeps a pass from no arm -- it goes round that loop again, or on past it.
    datasetManager's FinisherThr.run 688 and 690, in the loop over a job's files, the Kill arm 735 outside it."""
    holding = [n for n in own_nodes(func)
               if isinstance(n, LOOPS) and any(n.lineno <= line <= (n.end_lineno or n.lineno) for line in lines)]
    return {n.lineno for n in own_nodes(func)
            if isinstance(n, (ast.Continue, ast.Break)) and not any(_loop_of(n, func) is loop for loop in holding)}


def _direct_prints(block: list, stop: ast.AST | None = None) -> list[ast.stmt]:
    out = []
    for s in block:
        if s is stop:
            break
        if is_print(s):
            out.append(s)
    return out


def _holds(stmt: ast.stmt, lines: set[int]) -> bool:
    return any(stmt.lineno <= line <= (stmt.end_lineno or stmt.lineno) for line in lines)


def _raised_prefix(stmt: ast.stmt, raised_at: set[int]) -> list[ast.stmt]:
    """Prints a ``try`` body ran before the line a traceback frame says raised."""
    if not isinstance(stmt, ast.Try):
        return []
    for s in stmt.body:
        if _holds(s, raised_at):
            return _direct_prints(stmt.body, stop=s)
    return []


def must_rows(func: ast.AST, site: ast.stmt, raised_at: set[int] | None = frozenset()) -> tuple[list, list]:
    """``(before, after)``: the prints every path through *site* runs in one call.

    *raised_at* is where a traceback of the run shows the function left by an exception (:func:`raised_in`): a
    ``try`` body before *site* printed up to that line, and no print from the statement holding it on is a must
    row -- the call may have left there, *site* itself included.  None when a frame of the function cannot be
    placed: where it left is not known, and no print after *site* is a must row.

    A *site* that leaves (``return``, ``raise``) has no print after it: setup_jumbo_jobs's ``if
    len(self.jumbo_jobs) == 0: return`` at 1967 does not print "start" at 1968, which only the other way prints.
    """
    before: list[ast.stmt] = []
    after: list[ast.stmt] = []
    open_after = raised_at is not None and not _holds(site, raised_at) and not isinstance(site, EXITS)
    left_at = raised_at or set()
    node: ast.AST = site
    while node is not func:
        parent = node.parent  # type: ignore[attr-defined]
        placed = block_of(parent, node)
        if placed is None:
            node = parent
            continue
        name, block = placed
        index = block.index(node)
        for s in block[:index]:
            if is_print(s):
                before.append(s)
            elif left_at:
                before.extend(_raised_prefix(s, left_at))
        if open_after:
            for s in block[index + 1 :]:
                if _holds(s, left_at):
                    open_after = False
                    break
                if is_print(s):
                    after.append(s)
                elif may_divert(s):
                    open_after = False
                    break
        if parent is func:
            break
        if isinstance(parent, LOOPS) and name == "body" and contains(parent, EXITS):
            open_after = False
        if isinstance(parent, ast.Try) and name == "body" and open_after:
            for s in parent.orelse:
                if _holds(s, left_at):
                    open_after = False
                    break
                if is_print(s):
                    after.append(s)
                elif may_divert(s):
                    open_after = False
                    break
        if isinstance(parent, ast.ExceptHandler):
            # entered from anywhere in the try body: nothing there is implied,
            # so the next block read is the one the try itself sits in
            node = parent.parent
            continue
        node = parent
    return before, after


TRYS = (ast.Try, *((ast.TryStar,) if hasattr(ast, "TryStar") else ()))


def _loop_of(stmt: ast.AST, func: ast.AST) -> ast.AST | None:
    """The loop a ``break`` or ``continue`` leaves or goes back to: the nearest one whose body holds it."""
    node = stmt
    while node is not func:
        parent = node.parent  # type: ignore[attr-defined]
        if isinstance(parent, LOOPS) and node in parent.body:
            return parent
        node = parent
    return None


def implying_rows(func: ast.AST, site: ast.stmt) -> list[ast.stmt]:
    """Prints in the site's own block that no path prints without passing the site.

    Any print in a later statement of its block, nested or not: nothing enters
    a block below its top, so a printed row there proves the site ran (a call
    or a jump in between only means the row may be missing when it did).  Or a
    row before it with nothing on its way to the site that could leave: at the
    block's top, or nested in an earlier statement it falls out of
    (:func:`_falls_out`).  ``handle_failed_job`` prints 370 ("... is back")
    last in a ``try`` of an ``if``, and 374 follows that ``if``; 363, before
    the call 370 waits on, proves nothing: the call may leave.  The other must
    rows prove only that the function got that far: ``871`` prints on every
    run of ``parse_job_output_report`` whichever arm follows it.
    """
    placed = block_of(site.parent, site)  # type: ignore[attr-defined]
    if placed is None:
        return []
    _, block = placed
    index = block.index(site)
    found = []
    for s in block[index + 1 :]:
        found += sorted((n for n in own_nodes(s) if isinstance(n, ast.stmt) and is_print(n)), key=lambda n: n.lineno)
    for s in reversed(block[:index]):
        if is_print(s):
            found.append(s)
            continue
        found += sorted((n for n in own_nodes(s) if isinstance(n, ast.stmt) and is_print(n) and _falls_out(n, s)),
                        key=lambda n: n.lineno, reverse=True)
        if may_divert(s):
            break
    return found


def _falls_out(stmt: ast.stmt, outer: ast.stmt) -> bool:
    """Whether every way from *stmt*'s end leaves *outer*, which holds it, by falling through it, nothing on the way
    able to leave (:func:`may_divert`): the rest of each block it sits in, a ``try``'s ``else`` and ``finally`` after
    its body, the ``finally`` after a handler.  A handler is not entered: nothing after *stmt* in the body can raise
    then.  None falls out of a loop's body, which goes back to the loop; of a ``finally``, which a raise may be
    passing through; of a ``with``, whose exit is a call; or of a ``match``, not read (KNOWN_ISSUES.md K50)."""
    node = stmt
    while node is not outer:
        parent = node.parent  # type: ignore[attr-defined]
        if isinstance(parent, TRYS) and node in parent.handlers:
            name, later = "handlers", []
        elif (placed := block_of(parent, node)) is not None:
            name, block = placed
            later = block[block.index(node) + 1 :]
        else:
            return False
        if any(may_divert(s) for s in later):
            return False
        if isinstance(parent, TRYS):
            if name == "finalbody":
                return False
            if any(may_divert(s) for s in (parent.orelse if name == "body" else []) + parent.finalbody):
                return False
        elif isinstance(parent, LOOPS):
            if name == "body":
                return False
        elif not isinstance(parent, (ast.If, ast.ExceptHandler)):
            return False
        node = parent
    return True


def iterables_over(func: ast.AST, site: ast.stmt) -> list[ast.expr]:
    """What each ``for`` around the site iterates, innermost first."""
    found = []
    previous: ast.AST = site
    for ancestor in pathcond.ancestors(site):
        if ancestor is func:
            break
        if isinstance(ancestor, (ast.For, ast.AsyncFor)) and previous in ancestor.body:
            found.append(ancestor.iter)
        previous = ancestor
    return found


# ---------------------------------------------------------------- patterns

Part = tuple[str, str]  # ("lit", text) | ("hole", expression)


def _binding_sites(func: ast.AST, text: str, before: int) -> list[ast.stmt]:
    found = []
    for node in own_nodes(func):
        if node.lineno < before if isinstance(node, ast.stmt) else False:
            if _binds(node, text):
                found.append(node)
    return sorted(found, key=lambda n: n.lineno)


def _binds(node: ast.AST, text: str) -> bool:
    if isinstance(node, ast.AnnAssign) and node.value is None:
        return False  # a bare annotation binds nothing
    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        return any(ast.unparse(t) == text for t in targets)
    return False


def _dominating_binding(func: ast.AST, stmt: ast.stmt, text: str) -> ast.stmt | None:
    """The nearest binding of *text* before *stmt* in a block that encloses it.

    A compound statement in between that only appends to it (``if
    back_to_old_status: tmpMsg += ...``) is passed over: the value is the
    binding before it, with or without the tail (:func:`_maybe_appended`).
    None as soon as a compound statement in between binds it otherwise: the
    value is then whichever of them ran, which the union of sites answers.
    """
    node: ast.AST = stmt
    while node is not func:
        parent = node.parent  # type: ignore[attr-defined]
        placed = block_of(parent, node)
        if placed is not None:
            _, block = placed
            for s in reversed(block[: block.index(node)]):
                if _binds(s, text):
                    return s
                inner = [n for n in own_nodes(s) if _binds(n, text)]
                if inner and not all(isinstance(n, ast.AugAssign) for n in inner):
                    return None
        node = parent
    return None


def _maybe_appended(func: ast.AST, base: ast.stmt, stmt: ast.stmt, text: str, tail: list) -> bool:
    """Whether an append to *text* inside a compound statement lies between *base* and *stmt*."""
    return any(
        isinstance(n, ast.AugAssign) and _binds(n, text) and n not in tail and base.lineno < n.lineno < stmt.lineno
        for n in own_nodes(func)
    )


def _alternatives(
    expr: ast.expr, func: ast.AST, stmt: ast.stmt, depth: int = 0, expand: bool = True
) -> list[list[Part]]:
    """Every shape the message can take, as literal and hole parts.

    *expand* False keeps an f-string's holes as holes, which is what reading
    a value out of a line needs: expanding ``{self.job.jobStatus}`` into the
    ``"failed"`` the function assigns above it leaves nothing to read.
    """
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return [[("lit", expr.value)]]
    if isinstance(expr, ast.JoinedStr):
        shapes: list[list[Part]] = [[]]
        for part in expr.values:
            if isinstance(part, ast.Constant):
                shapes = [s + [("lit", str(part.value))] for s in shapes]
            else:
                inner = (
                    _alternatives(part.value, func, stmt, depth + 1)
                    if expand and part.format_spec is None
                    else None
                )
                if not inner or len(shapes) * len(inner) > 16:
                    inner = [[("hole", ast.unparse(part.value))]]
                shapes = [s + i for s in shapes for i in inner]
        return shapes
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
        left = _alternatives(expr.left, func, stmt, depth + 1, expand)
        right = _alternatives(expr.right, func, stmt, depth + 1, expand)
        return [a + b for a in left for b in right][:16]
    if isinstance(expr, (ast.Name, ast.Attribute)) and depth < 3:
        text = ast.unparse(expr)
        near = _dominating_binding(func, stmt, text)
        chosen: list[ast.stmt]
        if near is not None:
            # an augmented binding appends to whatever came before it
            chosen, tail = [near], []
            while isinstance(chosen[0], ast.AugAssign):
                base = _dominating_binding(func, chosen[0], text)
                tail.append(chosen[0])
                if base is None:
                    return [[("hole", text)]]
                chosen = [base]
            base_shapes = _alternatives(chosen[0].value, func, chosen[0], depth + 1, expand)
            appended = [s + [("hole", f"{text} (appended)")] for s in base_shapes]
            if tail:
                return appended
            return base_shapes + appended if _maybe_appended(func, chosen[0], stmt, text, tail) else base_shapes
        sites = _binding_sites(func, text, stmt.lineno)
        plain = [s for s in sites if not isinstance(s, ast.AugAssign)]
        if not plain:
            return [[("hole", text)]]
        shapes = []
        appended = any(isinstance(s, ast.AugAssign) for s in sites)
        for s in plain:
            for shape in _alternatives(s.value, func, s, depth + 1, expand):
                shapes.append(shape + ([("hole", f"{text} (appended)")] if appended else []))
        # a local every path binds is covered by its sites; anything else can
        # also hold what the function was handed
        if not isinstance(expr, ast.Name):
            shapes.append([("hole", text)])
        return shapes[:16]
    return [[("hole", ast.unparse(expr))]]


def _capture_shapes(message: ast.expr, func: ast.AST, stmt: ast.stmt) -> list[list[Part]]:
    """The shapes to read hole values with: the f-string as written, unless it is all hole."""
    direct = _alternatives(message, func, stmt, expand=False)
    if any(literal_of(s) for s in direct):
        return direct
    target = message
    if isinstance(message, ast.JoinedStr) and len(message.values) == 1:
        target = message.values[0].value  # type: ignore[attr-defined]
    return _alternatives(target, func, stmt, expand=False)


def literal_of(shape: list[Part]) -> int:
    return sum(len(t) for k, t in shape if k == "lit")


#: a shape whose literal has fewer non-blank characters than this matches any line
MIN_SHAPE = 3


def usable(shape: list[Part]) -> bool:
    return sum(len(t.replace(" ", "")) for k, t in shape if k == "lit") >= MIN_SHAPE


def _regex(shape: list[Part], capture: bool) -> tuple[str, dict[str, str]]:
    out, groups = [], {}
    holes = [i for i, (k, _) in enumerate(shape) if k == "hole"]
    for i, (kind, text) in enumerate(shape):
        if kind == "lit":
            out.append(re.escape(text))
        elif capture:
            name = f"g{i}"
            groups[name] = text
            out.append(f"(?P<{name}>[^\\n]*)" if i == holes[-1] and i == len(shape) - 1 else f"(?P<{name}>.*?)")
        else:
            out.append(".*?")
    if not capture:
        # every reader of a non-capturing pattern only asks whether re.search finds it, and a leading .*? can
        # match nothing, so it finds the same; kept, it retries from every position of a long record (adder_gen
        # 957's f"{lfn} is not found in filesTable": 25 of one case's 53 s over an adder log)
        while out and out[0] == ".*?":
            out.pop(0)
    return "".join(out), groups


#: every hole a skeleton row's pattern can hold (``bamboo/codemap/trace.py::_bound``):
#: a token closed against the next character, a number, text up to the next
#: literal, text to the end.  re.S throughout: ``re.escape`` writes a newline as backslash-newline, an escaped
#: character like any other
_MAP_HOLE = re.compile(
    r"\[\^(?:\\.|[^\]\\])*\]\*"
    + "|" + re.escape(r"[-+]?[\d.]+(?:[eE][-+]?\d+)?%?")
    + r"|\.\*\?|\.\*",
    re.S,
)
_UNESCAPED_META = re.compile(r"(?<!\\)[][(){}?*+|^$.]")
#: a print whose text begins with a hole and a space (``f"{job.PandaID} Finisher : Kill"``): ``Row.texts``
LEADING_HOLE = re.compile(r"f?[\"']\{[^}]+\}[ ]")


def _map_literal(pattern: str) -> int:
    """Characters of literal text in a skeleton pattern: what is left once its holes are taken out.

    Raises on anything left that is neither a hole nor an escaped literal --
    a hole shape the map started rendering that this reading does not know,
    and would otherwise count as text (``a0e2ac1`` added three).
    """
    return len(_map_text(pattern))


def _map_text(pattern: str) -> str:
    """The literal text of a skeleton pattern, holes taken out and escapes undone (raises as ``_map_literal``)."""
    rest = _MAP_HOLE.sub("", pattern)
    stray = _UNESCAPED_META.search(re.sub(r"\\.", "", rest, flags=re.S))
    if stray:
        raise ValueError(f"unknown construct {stray.group()!r} in map pattern {pattern!r}")
    return re.sub(r"\\(.)", r"\1", rest, flags=re.S)


def _map_usable(pattern: str) -> bool:
    r"""A map pattern held to what ``usable`` asks of a shape built here: ``MIN_SHAPE`` non-blank literal characters.

    A shorter one is refused: ``to\ `` (ppEventServiceJob 5667, an append's text drawn alone) is in any line
    holding "to ", and most readers of a row's hits ask only whether there is one."""
    return len(_map_text(pattern).replace(" ", "")) >= MIN_SHAPE


@dataclass
class Row:
    rel: str
    func: str
    stmt: ast.stmt
    level: str
    text: str
    map_patterns: list[str] = field(default_factory=list)
    map_refused: list[str] = field(default_factory=list)
    #: the map's word that its pattern is a question only together with the entity's
    #: key; every record read here was found by the key, so it is carried, not acted on
    map_keyed: bool = False
    shapes: list[list[Part]] = field(default_factory=list)
    capture_shapes: list[list[Part]] = field(default_factory=list)
    #: the map's prefix patterns of this statement's rows, when every row has one: what its logger writes before the
    #: message (``archiveJob\ <\ .*?\ >\ `` for a proxy method's create_tagged_logger), as the message alone
    #: (``start``, ``not found``) is shared by many writers; it narrows which records count and adds no literal
    map_prefixes: list[str] = field(default_factory=list)
    #: the files the map places this statement's lines in (``PrintStatement.log_files``, its logger followed); None
    #: where it does not place them (another ``log_files_basis``, no statement, ``ROWS_FROM=walk``)
    log_files: list[str] | None = None
    #: the map's rows of this statement (``PrintRow``, or the walk's ``SkeletonLine``), for what only the map reads:
    #: a record read whole (:meth:`reads_whole`)
    map_rows: list = field(default_factory=list, repr=False, compare=False)
    #: :meth:`patterns` per mode, made on the first ask: a row is built once per function (``_function``) and its
    #: patterns and shapes are not changed after, while ``matches`` asks for them once per record
    compiled: dict[str, list[tuple[re.Pattern[str], int, str]]] = field(default_factory=dict, repr=False, compare=False)
    #: the first words of the map's patterns (``records.head_word``), made on the first ask (:meth:`texts`)
    heads: frozenset[str] | None = field(default=None, repr=False, compare=False)

    def lands_in(self, name: str) -> bool:
        """Whether a line of this row can be in the log file *name*: yes where the map does not place it."""
        return self.log_files is None or name in self.log_files

    def written_by(self, record: Record) -> bool:
        """Whether *record* can be this row's line by its prefix: always when the map gives
        none; else the prefix stands where the line starts (``archiveJob < ... >`` is the token and the head of
        the message) or right after the token (``LogWrapper(self.logger, "<run>")``: the outer wrapper's mark
        comes first).  Checked apart from the patterns: a map pattern need not start where the message does
        (``sqlDelE + comment`` is drawn from `` /* DBProxy...``).  A prefix's leading blanks are not asked for: the
        record's cut (``records.PREFIX``) reads the level with the blanks after it, a mark's own among them
        (``MsgWrapper(self.logger, f" < jediTaskID={jediTaskID} >")``), while the map's prefix is what the logger
        writes (tmaeno, 10-30 15:4x)."""
        if not self.map_prefixes:
            return True
        regex = _prefix_regex(tuple(self.map_prefixes))
        line = f"{record.token} {record.text}"
        return bool(regex.match(line) or regex.match(line, len(record.token) + 1))

    @property
    def line(self) -> int:
        return self.stmt.lineno

    def map_covers(self) -> bool:
        """Whether the map's patterns cover every way this row can print."""
        return bool(self.map_patterns) and not self.map_refused

    def _outside_shapes(self) -> list[list[Part]]:
        return [s for s in self.shapes if usable(s)]

    def patterns(self, mode: str) -> list[tuple[re.Pattern[str], int, str]]:
        """``(regex, literal characters, source)`` the row can be recognised by.

        ``outside`` adds the shapes built here to the map's patterns rather
        than replacing them: a map pattern can be present and still too
        narrow (a ``str(dict)`` hole read as having no
        blank), and a narrow pattern turns a line that printed into one that
        did not.  The union can only find more.

        Rule 22 once kept only the shapes built here for a name one
        binding decides, because the map drew every binding of the name in the
        function; the map now draws the ones that can still reach the print
        (of 358 such rows, none draws one that cannot).
        """
        if mode not in self.compiled:
            found =[(re.compile(p, re.S), _map_literal(p), "map") for p in self.map_patterns]
            if mode == "outside":
                found += [(re.compile(_regex(s, False)[0], re.S), literal_of(s), "outside:3-4") for s in self._outside_shapes()]
            self.compiled[mode] = found
        return self.compiled[mode]

    def absence_counts(self, mode: str) -> bool:
        """Whether this row being missing from a run says it did not print."""
        if self.map_covers():
            return True
        return mode == "outside" and bool(self.shapes) and all(usable(s) for s in self.shapes)

    def texts(self, record: Record) -> list[str]:
        """What this row's patterns are read on in *record*: its message, and for a print that begins with a hole the
        line as the logger wrote it.  A logger with no mark of its own writes that hole's value first, and the log's
        parser takes it for the token (``records.PREFIX``): ``f"{job.PandaID} Finisher : Kill"`` is the token
        ``7392812614`` and the message ``Finisher : Kill``, while the map's pattern begins at the space after the hole
        (KNOWN_ISSUES.md K53).  So for a print with no prefix that begins with a word: the parser takes the word,
        ``failed`` of ``self.logger.debug(f"failed to get taskSpec for jediTaskID={jediTaskID}")``
        (ContentsFeeder.py:157), for the token, and the message alone holds no pattern of the row (K70)."""
        if self.heads is None:
            self.heads = frozenset() if self.map_prefixes else frozenset(w for p in self.map_patterns if (w := head_word(p)))
        if record.token and (LEADING_HOLE.match(self.text) or record.token in self.heads):
            return [record.text, f"{record.token} {record.text}"]
        return [record.text]

    def matches(self, record: Record, mode: str) -> tuple[int, str]:
        """``(literal, source)`` of the best pattern that finds the record (:meth:`texts`); ``(0, "")`` when none
        does."""
        best = (0, "")
        if not self.written_by(record):
            return best
        texts = self.texts(record)
        for regex, literal, source in self.patterns(mode):
            if literal > best[0] and any(regex.search(t) for t in texts):
                best = (literal, source)
        return best

    def reads_whole(self, record: Record) -> bool | None:
        """Whether the map reads *record* as this row's line whole (``trace.reads_whole``: its mark, then its
        message, anchored at both ends), not only a line holding its text: ``start`` finds ``start 3/10
        jediTaskID=...`` too, which is another row's line.  None where none of the map's rows of it can say: no
        capture, or a mark its logger may write that the map did not read."""
        said = {trace.reads_whole(m, record.token, record.text) for m in self.map_rows}
        return True if True in said else False if False in said else None

    def reads_literally(self, record: Record) -> bool | None:
        """Whether the map reads *record* as this row's line whole with every hole taking any characters
        (``trace.reads_literally``): what :meth:`reads_whole` leaves None where a hole's guessed class does not take
        the text -- killJob's ``user={user} wg=`` closes ``user`` at the first blank.  None where none of the map's
        rows of it can say, as :meth:`reads_whole`."""
        said = {trace.reads_literally(m, record.token, record.text) for m in self.map_rows}
        return True if True in said else False if False in said else None

    def capture(self, record: Record) -> dict[str, str]:
        """Hole values in *record*, keyed by the expression each hole prints; none from a record the search cut."""
        if record.cut:
            return {}
        for shape in sorted(self.capture_shapes, key=literal_of, reverse=True):
            if not usable(shape):
                continue
            pattern, groups = _regex(shape, True)
            for text in self.texts(record):
                if m := re.search(pattern, text, re.S):
                    return {groups[g]: v for g, v in m.groupdict().items() if not groups[g].endswith("(appended)")}
        return {}


@dataclass
class Function:
    rel: str
    name: str
    node: ast.FunctionDef | ast.AsyncFunctionDef
    rows: list[Row]
    arms: list[int]

    @property
    def owner(self) -> str:
        return f"{self.rel}::{self.name}"

    def row_at(self, stmt: ast.stmt) -> Row | None:
        return next((r for r in self.rows if r.stmt is stmt), None)


_GRAPH: dict = {}


def _function_prints(owner: str):
    """The map's ``FunctionPrintsNode`` for *owner*, or None: one reader of the installed dump per process
    (:func:`bamboo.querier.mapdump.code_map`); with no dump installed it stops."""
    if owner in _GRAPH:
        return _GRAPH[owner]
    import asyncio

    if "loop" not in _GRAPH:
        _GRAPH["loop"], _GRAPH["map"] = asyncio.new_event_loop(), mapdump.code_map(f"the prints of {owner}")
    found = _GRAPH["loop"].run_until_complete(_GRAPH["map"].function_prints([owner]))
    _GRAPH[owner] = found.get(owner)
    return _GRAPH[owner]


def print_lines(owner: str) -> list[int]:
    """The lines of the print statements the map's ``FunctionPrintsNode`` of *owner* (``rel::Class.method``) holds,
    in order: a line inside the function, for a reader holding only its owner (``line_writers``' answer)."""
    found = _function_prints(owner)
    return sorted(s.line for s in found.statements) if found is not None else []


def _prints_of(owner: str, lines: Iterable[int] = ()):
    """The map's prints for the def *owner* names (``rel::name``, the name with or without its class), or None.

    A reading's owner carries the bare name (``adder_gen.py::process_job_report``)
    and the map's the qualified one (``AdderGen.process_job_report``): the def is
    found as ``function`` finds it, by name and the lines asked, and named as
    ``_graph_rows`` names it.
    """
    rel, name = owner.split("::", 1)
    found = functions(rel, name.split(".")[-1])
    if not found:
        return None
    return _function_prints(f"{rel}::{pathcond.qualified_name(_holding(found, tuple(lines)))}")


def own_log_files(owner: str, today: list[str], lines: Iterable[int] = ()) -> list[str]:
    """The files *owner*'s own printed lines land in, by the map; *today* where the map cannot place them.

    Every statement of the function must have basis ``logger`` (the logger it is
    made on, followed to where it was opened, through a logger handed in).
    ``module`` and ``partly`` say absence there proves nothing -- the adder and
    setupper plugins, whose class ``panda_config.getPlugin`` chooses, are
    ``module`` -- and then the reading's files stay.  A function is one or the
    other, never both.  A
    function the map has no prints for keeps *today* too: it has no rows to read.
    An empty answer is one: a standard-library logger lands in no file.
    """
    if ROWS_FROM != "graph":
        return today
    found = _prints_of(owner, lines)
    if found is None or not found.statements or any(s.log_files_basis != "logger" for s in found.statements):
        return today
    return sorted({f for s in found.statements for f in s.log_files})


def split_files(owner: str, lines: Iterable[int] = ()) -> list[list[str]]:
    """The distinct file sets *owner*'s placed statements land in, when there is more than one.

    Such a function is read in one run of one file, where a row the map places
    only in another file is neither shown nor missing (:meth:`Row.lands_in`):
    its lines are not read (KNOWN_ISSUES.md K23).  2 functions of 1093
    (pilot_api's ``acquire_jobs``, ``update_job``), in no baseline.
    """
    found = _prints_of(owner, lines) if ROWS_FROM == "graph" else None
    if found is None:
        return []
    sets = sorted({tuple(sorted(s.log_files)) for s in found.statements if s.log_files_basis == "logger"})
    return [list(x) for x in sets] if len(sets) > 1 else []


@functools.cache
def _blob_sha(rel: str) -> str:
    data = (CORPUS / rel).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def check_blobs(answer: object, what: str) -> None:
    """Stop where a map answer was read from another version of a file than the corpus holds.

    Every ``(file, blob_sha)`` pair at any depth of *answer* -- a reading's, a statement's, a junction's ``anchor``
    -- against the corpus file's git blob.  FunctionPrints is checked where it is read (:func:`_graph_rows`); this is
    for what ``derive_strategy._investigate`` and ``CodeMap.writers_of`` hand the querier, whose line numbers were
    otherwise applied to the corpus as they came: a map built from another commit than ``PANDA_PIN`` read the wrong
    statements and said nothing.  An empty ``blob_sha`` is not checked.
    """
    bad: dict[str, tuple[str, str]] = {}

    def visit(x: object) -> None:
        if hasattr(x, "model_dump"):
            x = x.model_dump()  # type: ignore[union-attr]
        if isinstance(x, dict):
            rel, blob = x.get("file"), x.get("blob_sha")
            if isinstance(rel, str) and rel and isinstance(blob, str) and blob:
                here = _blob_sha(rel) if (CORPUS / rel).is_file() else "absent"
                if here != blob:
                    bad.setdefault(rel, (blob, here))
            for v in x.values():
                visit(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                visit(v)

    visit(answer)
    if bad:
        shown = "; ".join(f"{rel} map {b[:12]} corpus {h[:12]}" for rel, (b, h) in sorted(bad.items())[:5])
        raise RuntimeError(
            f"{what}: the map ({mapdump.described()}) read {len(bad)} file(s) at another version than the corpus "
            f"({CORPUS}): {shown} -- run with the map built from this corpus (MAP_DUMP)"
        )


def _graph_rows(rel: str, node: ast.AST) -> tuple[dict[int, list], dict[int, list[str]]]:
    """``(line -> the node's rows, line -> files)``: the files where the map places a statement's lines by its logger
    (``log_files_basis`` ``logger``)."""
    found = _function_prints(f"{rel}::{pathcond.qualified_name(node)}")
    if found is not None and found.anchor.blob_sha != _blob_sha(rel):
        # the map read another version of the file: its rows sit at other lines
        raise RuntimeError(
            f"the map's prints for {rel} are of blob {found.anchor.blob_sha[:12]}, the corpus file is "
            f"{_blob_sha(rel)[:12]} ({CORPUS}), the map is {mapdump.described()}: run with ROWS_FROM=walk or the map's corpus"
        )
    by_line = {s.line: list(s.rows) for s in found.statements} if found is not None else {}
    files_of = {s.line: list(s.log_files) for s in found.statements if s.log_files_basis == "logger"} if found is not None else {}
    return by_line, files_of


@functools.cache
def _prefix_regex(prefixes: tuple[str, ...]) -> re.Pattern[str]:
    # the leading blanks optional, not dropped: right after an outer mark's token a mark's own blank stays in the text
    bare = [re.sub(r"^(?:\\?\s)+", "", p) for p in prefixes]
    return re.compile(r"\s*" + (bare[0] if len(bare) == 1 else "(?:" + "|".join(bare) + ")"), re.S)


#: ``[fragment, [(pattern, owner, prefix regex or None, logs or None, map row)], {text: [(owner, prefix, logs, map
#: row)]}, {(log, token, text): writers}, {(log, token, text): whole writers}, {first word: rows with no prefix},
#: {(log, token, text): literal writers}]`` for the installed dump (:func:`line_writers`, :func:`whole_writers`,
#: :func:`literal_writers`); the fragment is held, so a dump installed after is told from it by identity
_WRITERS: list = []


def _writers() -> list | None:
    """:data:`_WRITERS` for the installed dump, built on its first use; None without a dump."""
    fragment = mapdump.fragment()
    if fragment is None:
        return None
    if not _WRITERS or _WRITERS[0] is not fragment:
        rows = []
        # the rows with no prefix by their pattern's first word: a line its logger writes with no mark is cut there (K70)
        heads: dict[str, list] = {}
        for f in fragment.function_prints:
            for st in f.statements:
                logs = frozenset(st.log_files) if st.log_files_basis == "logger" else None
                for r in st.rows:
                    if r.pattern and _map_usable(r.pattern):
                        row = (re.compile(r.pattern, re.S), f.owner, _prefix_regex((r.prefix,)) if r.prefix else None, logs, r)
                        rows.append(row)
                        if not r.prefix and (word := head_word(r.pattern)):
                            heads.setdefault(word, []).append(row)
        _WRITERS[:] = [fragment, rows, {}, {}, {}, heads, {}]
    return _WRITERS


def _found(writers: list, record: Record, log: str) -> list[tuple[str, PrintRow]]:
    """``(owner, map row)`` of the rows :func:`line_writers` reads *record* by; a row with no prefix whose pattern's
    first word is the record's token reads the line as written, that word on (K70)."""
    by_text = writers[2]
    if record.text not in by_text:
        by_text[record.text] = [(o, p, logs, r) for regex, o, p, logs, r in writers[1] if regex.search(record.text)]
    line = f"{record.token} {record.text}"
    found = [(o, r) for o, p, logs, r in by_text[record.text] if (logs is None or log in logs) and (p is None or p.match(line))]
    found += [(o, r) for regex, o, _, logs, r in writers[5].get(record.token, ())
              if (logs is None or log in logs) and regex.match(line) and (o, r) not in found]
    return found


def line_writers(record: Record, log: str) -> frozenset[str]:
    """The functions (``rel::Class.method``) of the installed dump that can have printed *record*, a line of the log
    *log*: those with a row whose map pattern is found in the record's text, whose prefix stands where the record's
    line starts or which has none (the map gives none where the logger is made with different marks on its paths:
    calculateRWwithPrio_JEDI's two tags -- such a row holds any mark), and whose statement's lines can land in *log*
    (``PrintStatement.log_files``; anywhere where the map did not follow the logger to them, ``log_files_basis``
    other than ``logger``).  A comment copied from another method gives two methods one mark
    (configurator_read_panda_sites's ``configurator_read_sites < ... >``).  A row with no pattern the map can ask with
    (all holes) is not counted: it can print any text at any length, which MIN_PROOF does not read either.  A mark
    inside an outer one's text (:meth:`Row.written_by`) is not read: the message is then more than the row's text,
    which proves nothing below MIN_PROOF.  None without a dump."""
    if (writers := _writers()) is None:
        return frozenset()
    known: dict[tuple[str, str, str], frozenset[str]] = writers[3]
    key = (log, record.token, record.text)
    if key not in known:
        known[key] = frozenset(o for o, _ in _found(writers, record, log))
    return known[key]


def whole_writers(record: Record, log: str) -> frozenset[str]:
    """The functions of :func:`line_writers` with a row the map reads *record* as whole (``trace.reads_whole``: its
    mark, then its message, anchored at both ends), not only a line holding its text: AtlasProdTaskRefiner.doRefine's
    "start taskType=..." and "done" (107, 174) are AtlasAnalTaskRefiner.doRefine's (205, 280) too, while
    PostProcessor's "done" rows are not "doBasicPostProcess done with taskStatus=...".  A row the map cannot read
    whole (None: no capture, or a mark its logger may write that the map did not read) is not counted: the rows with
    no prefix, PostProcessor.start's "done" and TaskRefiner.start's "start" among them, would take every such text of
    their log (KNOWN_ISSUES.md K42).  None without a dump."""
    if (writers := _writers()) is None:
        return frozenset()
    known: dict[tuple[str, str, str], frozenset[str]] = writers[4]
    key = (log, record.token, record.text)
    if key not in known:
        known[key] = frozenset(
            o for o, row in _found(writers, record, log) if trace.reads_whole(row, record.token, record.text) is True)
    return known[key]


def literal_writers(record: Record, log: str) -> frozenset[str]:
    """The functions of the installed dump with a row that reads *record* as its line whole with every hole taking any
    characters (``trace.reads_literally``), whose prefix stands where the record's line starts or which has none, and
    whose statement's lines can land in *log* -- :func:`line_writers` with the holes loosened.  The map's own pattern
    keeps a hole's guessed class: killJob 840's ``user=[^\\ ]*\\ wg=`` does not find ``user=finish by ... wg=``, so
    line_writers holds no function for that record, killJob's own row neither; this reading holds killJob alone.  A row
    with no pattern the map can ask with (all holes) is not counted, as in line_writers.  None without a dump."""
    if (writers := _writers()) is None:
        return frozenset()
    known: dict[tuple[str, str, str], frozenset[str]] = writers[6]
    key = (log, record.token, record.text)
    if key not in known:
        line = f"{record.token} {record.text}"
        known[key] = frozenset(
            o for _, o, p, logs, row in writers[1]
            if (logs is None or log in logs) and (p is None or p.match(line))
            and trace.reads_literally(row, record.token, record.text) is True)
    return known[key]


def _prefixes(mapped: list) -> list[str]:
    """The distinct prefixes of a statement's map rows, when every one has one (59 of 5732 statements have
    several: a logger bound two ways); none when any row lacks one, which reads every record as before."""
    found = [getattr(m, "prefix", "") or "" for m in mapped]
    return sorted(set(found)) if found and all(found) else []


def _holding(found: list, arms: Iterable[int]) -> ast.FunctionDef | ast.AsyncFunctionDef:
    # one name can be several defs (datasetManager.py has seven ``run``; 80 names of the corpus repeat in their
    # file): the one holding the asked lines, as the walk picks its frame by line.
    # A repeated name the lines pick no def for stops -- taking the first read another def's rows without a word
    holding = [f for f in found if arms and all(f.lineno <= a <= (f.end_lineno or f.lineno) for a in arms)]
    if holding:
        return min(holding, key=lambda f: (f.end_lineno or f.lineno) - f.lineno)
    if len(found) > 1:
        raise LookupError(
            f"{found[0].name} is defined {len(found)} times (lines {[f.lineno for f in found]}) and lines "
            f"{sorted(arms) or 'none'} pick none of them"
        )
    return found[0]


def def_at(rel: str, name: str, lines: Iterable[int]) -> ast.FunctionDef | ast.AsyncFunctionDef:
    """The def of *name* in *rel* holding *lines* -- the one ``function(rel, name, within=lines)`` reads."""
    found = functions(rel, name)
    if not found:
        raise LookupError(f"no function {name} in {rel}")
    return _holding(found, tuple(lines))


def def_holding(rel: str, line: int) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """The innermost def in *rel* holding *line*, or None: the def a ``called_from`` site calls from."""
    holding = [f for f, _ in pathcond.functions_with_owner(tree(rel)) if f.lineno <= line <= (f.end_lineno or f.lineno)]
    return min(holding, key=lambda f: (f.end_lineno or f.lineno) - f.lineno) if holding else None


@functools.cache
def _function(rel: str, name: str, arms: tuple[int, ...], within: tuple[int, ...] = ()) -> Function:
    found = functions(rel, name)
    if not found:
        raise LookupError(f"no function {name} in {rel}")
    node = _holding(found, arms or within)
    by_line: dict[int, list] = {}
    files_of: dict[int, list[str]] = {}
    if ROWS_FROM == "graph":
        by_line, files_of = _graph_rows(rel, node)
    else:
        from .asked import _types  # asked imports this module

        walked = trace.Asker(ROOTS, file=rel, owner=f"{rel}::{name}", lines=list(arms) or [node.lineno], classify=classify,
                             reaching=False, types=_types()).start()
        for sk in walked.skeleton:
            if sk.kind == "print":
                by_line.setdefault(sk.line, []).append(sk)
    rows = []
    for stmt in own_nodes(node):
        if not is_print(stmt):
            continue
        call = stmt.value
        mapped = by_line.get(stmt.lineno, [])
        short = [f"map pattern too short: {m.pattern}" for m in mapped if m.pattern and not _map_usable(m.pattern)]
        rows.append(
            Row(
                rel,
                name,
                stmt,
                LEVEL_OF_METHOD[call.func.attr],
                ast.unparse(call.args[0]),
                [m.pattern for m in mapped if m.pattern and _map_usable(m.pattern)],
                [m.refused for m in mapped if not m.pattern] + short or ([] if mapped else ["the map has no row here"]),
                any(m.keyed for m in mapped if m.pattern),
                _alternatives(call.args[0], node, stmt),
                _capture_shapes(call.args[0], node, stmt),
                _prefixes(mapped),
                files_of.get(stmt.lineno),
                mapped,
            )
        )
    rows.sort(key=lambda r: r.line)
    return Function(rel, name, node, rows, list(arms))


def function(rel: str, name: str, arms: Iterable[int] = (), within: Iterable[int] = ()) -> Function:
    """*name*'s rows in *rel*, armed for *arms*.  *within* (lines the def holds) only picks which def, when the
    name repeats and no arms are asked -- arms change what the rows say, so a caller that knows a line but asks
    no arm passes it here."""
    # within is dropped when it cannot matter, so the one Function object stays the one every caller holds
    picks = not arms and len(functions(rel, name)) > 1
    return _function(rel, name, tuple(sorted(set(arms))), tuple(sorted(set(within))) if picks else ())


def mark_heads(rel: str, name: str) -> tuple[str, ...]:
    """The fixed text each of *name*'s own rows below WARNING starts with, up to its first line's end: the door asks
    the method's mark for these rows (``recorded._narrowed``).  Empty where one may start otherwise -- a shape whose
    first part is a hole (an f-string's leading expression, a message handed in), a row with no shape -- or where
    *rel* defines *name* more than once: the mark is asked whole."""
    if len(functions(rel, name)) != 1:
        return ()
    heads = set()
    for row in function(rel, name).rows:
        if LEVEL_RANK[row.level] >= LEVEL_RANK["WARNING"]:
            continue
        if not row.shapes or any(not shape or shape[0][0] != "lit" for shape in row.shapes):
            return ()
        heads |= {shape[0][1].split("\n", 1)[0] for shape in row.shapes}
    return tuple(sorted(heads))


# ---------------------------------------------------------------- whether the map's pattern can be trusted to be absent

_LOG_COUNTS: dict[tuple[int, str, int], tuple[int, int]] = {}


def pattern_finds(row: Row, log) -> tuple[int, int]:
    """``(records the map's patterns find, records the shapes built here find)`` in the whole file.

    A map pattern that finds nothing in a file where the row's own shape
    finds lines is a pattern that is wrong, not a row that never printed --
    and read as absence it excludes the site that ran (a pattern
    ``[^\\ ]*`` for a ``str(dict)`` hole).  Counted over the file rather than
    the run, which is rule 8 in its strong form: the real run could read
    ``861`` as absent because the same row printed nineteen times elsewhere.
    """
    key = (id(row.stmt), row.rel, id(log))
    if key not in _LOG_COUNTS:
        maps = [re.compile(p, re.S) for p in row.map_patterns]
        shapes = [re.compile(_regex(s, False)[0], re.S) for s in row._outside_shapes()]
        mine = [r for r in log.records if row.written_by(r)]
        by_map = sum(1 for r in mine if any(m.search(r.text) for m in maps)) if maps else 0
        by_shape = sum(1 for r in mine if any(m.search(r.text) for m in shapes)) if shapes else 0
        _LOG_COUNTS[key] = (by_map, by_shape)
    return _LOG_COUNTS[key]


#: :func:`absence_trusted`'s reason for a row the map places in other files than the run's
ANOTHER_FILE = "another file"


def absence_trusted(row: Row, log, mode: str) -> tuple[bool, str]:
    """Whether the row missing from a run of *log* is evidence; the reason when not."""
    if not row.lands_in(log.name):
        return False, ANOTHER_FILE
    if not row.absence_counts(mode):
        return False, "unreadable"
    if row.map_covers():
        by_map, by_shape = pattern_finds(row, log)
        if by_map == 0 and by_shape > 0:
            return False, (
                f"map pattern for {row.line} finds no line of {Path(log.path).name}, "
                f"where the row's shape finds {by_shape}"
            )
    return True, ""


# ---------------------------------------------------------------- a function against a run


@dataclass
class Seen:
    """How the rows of one function show up in one run."""

    fn: Function
    run: Run
    mode: str
    hits: dict[int, list[tuple[int, int, str]]]  # row line -> [(record index, literal, source)]
    unambiguous: dict[int, list[int]]  # row line -> record indexes only this row matches, no other function's whole

    def present(self, row: Row) -> bool:
        return bool(self.hits.get(row.line))

    def would_show(self, row: Row) -> bool:
        """Whether the run would show *row* had it printed: rule 4's test of a row missing from a run
        (:func:`absence_trusted`, ``Run.would_print``).  A print it would not show may have printed unseen."""
        return absence_trusted(row, self.run.log, self.mode)[0] and self.run.would_print(row.level, row.func)

    def proven(self, row: Row) -> bool:
        """Whether a record of the run is this row's, read by no other row of the function (a text two rows print
        alike is either one's: activateJob's filesTable4 UPDATE, 92 in the arm that activates and 124 in the one that
        does not) and proving it (:meth:`proves`)."""
        mine = set(self.unambiguous.get(row.line, ()))
        return any(i in mine and self.proves(row, i, lit) for i, lit, _ in self.hits.get(row.line, ()))

    def proves(self, row: Row, i: int, literal: int) -> bool:
        """Whether record *i*, which *row* finds at *literal* characters, proves the row: at MIN_PROOF; below it where
        the record cannot be another line -- its message is the row's text whole, and no other function can have
        printed it there (:func:`line_writers`): activateJob's "start" and "done" (5 and 4 characters) under
        ``activateJob < PandaID=... >``, since the map gives a mark to every logger that prints "start" in its logs.  A
        message that only holds the text (``restart``; a traceback another function prints with the caller's logger)
        proves nothing below MIN_PROOF, nor does a record the search cut (KNOWN_ISSUES.md K44)."""
        if literal >= MIN_PROOF:
            return True
        record = self.run.records[i]
        if not row.map_prefixes or record.cut:
            return False
        if not any(regex.fullmatch(record.text.strip()) for regex, _, _ in row.patterns(self.mode)):
            return False
        return line_writers(record, self.run.log.name) == {f"{self.fn.rel}::{pathcond.qualified_name(self.fn.node)}"}

    def single_call(self) -> bool:
        return all(len(v) <= 1 for v in self.unambiguous.values())

    def without(self, lines: Iterable[int]) -> Seen:
        """This run less the records of the rows at *lines*: the run an arm's proofs read where those rows printed
        for other objects than the one the arm writes (``which_ran.other_objects``)."""
        drop = set(lines)
        if not drop:
            return self
        return replace(self, hits={k: v for k, v in self.hits.items() if k not in drop},
                       unambiguous={k: v for k, v in self.unambiguous.items() if k not in drop})


def seen(fn: Function, run: Run, mode: str) -> Seen:
    hits: dict[int, list[tuple[int, int, str]]] = {}
    for i, record in enumerate(run.records):
        for row in fn.rows:
            literal, source = row.matches(record, mode)
            if literal:
                hits.setdefault(row.line, []).append((i, literal, source))
    return _seen(fn, run, mode, hits)


def _seen(fn: Function, run: Run, mode: str, hits: dict[int, list[tuple[int, int, str]]]) -> Seen:
    """A record is placed (``Seen.unambiguous``) where one row of *fn* reads it and no other function of the dump
    prints it whole (:func:`whole_writers`): a text printed alike proves neither, of two rows or of two functions
    (KNOWN_ISSUES.md K42)."""
    per_record: dict[int, set[int]] = {}
    for line, hs in hits.items():
        for i, _, _ in hs:
            per_record.setdefault(i, set()).add(line)
    unambiguous: dict[int, list[int]] = {}
    owner = ""
    for i, lines in sorted(per_record.items()):
        if len(lines) != 1:
            continue
        if others := whole_writers(run.records[i], run.log.name):
            owner = owner or f"{fn.rel}::{pathcond.qualified_name(fn.node)}"
            if others - {owner}:
                continue
        unambiguous.setdefault(next(iter(lines)), []).append(i)
    return Seen(fn, run, mode, hits, unambiguous)


def raised_in(fn: Function, run: Run) -> tuple[bool, set[int] | None]:
    """``(a traceback in *run* names fn, the lines its frames of fn show)``, at the pin's lines (a frame of another
    version of the code is read at the one line of fn holding its source, ``records._place``); lines None when a
    frame of fn cannot be placed so, or is outside the corpus or of another file of the same name."""
    named, lines = False, set()
    base = fn.rel.rsplit("/", 1)[-1]
    for record in run.records:
        for ch in record.chains():
            for f in ch.frames:
                if f.func != fn.name or f.path.rsplit("/", 1)[-1] != base:
                    continue
                named = True
                if not (f.in_corpus and f.same_source and f.rel == fn.rel):
                    return True, None
                lines.add(f.line)
    return named, lines


def claim(seens: list[Seen]) -> list[Seen]:
    """Give each record only to the rows that read it with the most literal among *seens*.

    A thin shape of one function (``{str(e)} : {…} in {…}``) also matches a
    line another function printed, and would count as printed there and read
    a value out of text it never wrote (register_files:778
    matched execute:168's traceback and read ``prodSourceLabel`` out of it).
    Only the functions given compete: a writer outside them does not take its
    record back from a thin shape inside them (one that prints it whole leaves
    it placed by none, :func:`_seen`).
    """
    strongest: dict[tuple[int, int], int] = {}
    for sn in seens:
        for hs in sn.hits.values():
            for i, literal, _ in hs:
                key = (id(sn.run), i)
                strongest[key] = max(strongest.get(key, 0), literal)
    out = []
    for sn in seens:
        hits = {}
        for line, hs in sn.hits.items():
            kept = [h for h in hs if h[1] >= strongest[(id(sn.run), h[0])]]
            if kept:
                hits[line] = kept
        out.append(_seen(sn.fn, sn.run, sn.mode, hits))
    return out


# ---------------------------------------------------------------- one mark, several passes


def _logger_of(row: Row) -> str:
    """The logger a row prints through, as spelled (``tmpLog``, ``self.logger``)."""
    return ast.unparse(row.stmt.value.func.value)  # type: ignore[attr-defined]


def entry_row(fn: Function) -> Row | None:
    """The row every other row printed through the same logger comes after, in one pass.

    A mark that is the entity's key (``< jediTaskID=12345 >``, the adder's
    PandaID) is written by every pass over that entity, so one mark can hold
    several passes: TaskCommando's ``runImpl`` picked one task up four
    times in 26 minutes.  Each pass prints this row once, before the others
    (rule 20).  None when no row comes before all the others -- the function
    is then not split, and the caller says so.
    """
    found = []
    for r in fn.rows:
        others = [s for s in fn.rows if s is not r and _logger_of(s) == _logger_of(r)]
        if others and all(r.stmt in must_rows(fn.node, s.stmt)[0] for s in others):
            found.append(r)
    return min(found, key=lambda r: r.line) if found else None


def entries(fn: Function, run: Run, mode: str) -> list[int]:
    """Indexes of the records in *run* that the map reads as *fn*'s entry row whole (:meth:`Row.reads_whole`): one
    per pass of *fn*.  The row's pattern finds more: the ``start`` of prepareTasksToBeFinished_JEDI finds its loop's
    ``start 3/10 jediTaskID=...``, and TaskRefiner's runImpl's finds doRefine's ``start taskType=anal`` through the
    same logger -- not a pass begun.  A record the map cannot read whole does not cut either: the passes it may begin
    stay one (KNOWN_ISSUES.md K47) -- unless it is the entry row's line with its holes taking any characters and
    nothing else can have printed it (:func:`_entry_reads`)."""
    return sorted(i for i, whole, _ in _entry_reads(fn, run, mode) if whole)


def _entry_reads(fn: Function, run: Run, mode: str) -> list[tuple[int, bool | None, bool]]:
    """Each record of *run* *fn*'s entry row finds: ``(index, whether it is the row's line, read so only literally)``.

    It is the row's line where the map reads it so whole (:meth:`Row.reads_whole`).  Where the map cannot say (None),
    it is the row's line too where the map reads it so with every hole taking any characters
    (:meth:`Row.reads_literally`) and nothing else can have printed it: no other row of *fn* finds it, and no other
    function of the dump reads it so (:func:`literal_writers`; :func:`line_writers` keeps the guessed classes, and
    holds no function at all for such a record).
    killJob's 840 ``code={code} role={prodManager} user={user} wg=...`` closes ``user`` at its first blank, so a user
    holding blanks (``finish by ...``) is read whole nowhere: MJE-7396802500's two calls under one mark (05:27:04,
    skipped with NOWAIT, and 05:40:46) stayed one pass, which J does not read (out/gm/qk-design.md, SC)."""
    entry = entry_row(fn)
    if entry is None:
        return []
    hits = seen(fn, run, mode).hits
    finders: dict[int, set[int]] = {}
    for line, found in hits.items():
        for i, _, _ in found:
            finders.setdefault(i, set()).add(line)
    owner = f"{fn.rel}::{pathcond.qualified_name(fn.node)}"
    out: list[tuple[int, bool | None, bool]] = []
    for i, _, _ in hits.get(entry.line, ()):
        record = run.records[i]
        whole = entry.reads_whole(record)
        literal = (whole is None and finders[i] == {entry.line} and entry.reads_literally(record) is True
                   and literal_writers(record, run.log.name) == {owner})
        out.append((i, True if literal else whole, literal))
    return out


def passes(fn: Function, run: Run, mode: str) -> tuple[list[Run], str]:
    """*run* cut at each record of *fn*'s entry row, with a word on how it was cut.

    Records before the first entry are not a pass of *fn*: another function
    that shares the mark printed them, or the log starts inside a pass.  They
    are counted in the word, not returned.  Whether the map can read the entry
    row whole is the row's, not a record's (:meth:`Row.reads_whole` is None only
    where none of its map rows has a capture and a mark read), so a run with a
    record the map cannot read whole has no entry read whole: it is not split,
    and the word counts those records.
    """
    entry = entry_row(fn)
    if entry is None:
        return [run], f"{fn.name} has no entry row: the run is not split"
    read = _entry_reads(fn, run, mode)
    starts = sorted(i for i, whole, _ in read if whole)
    if not starts:
        unread = sum(1 for _, whole, _ in read if whole is None)
        found = (f"is read whole at none of its {len(read)} record(s) in the run"
                 + (f" ({unread} the map cannot read whole)" if unread else "")) if read else "is not in the run"
        return [run], f"entry row {entry.line} of {fn.name} {found}: the run is not split"
    cuts = starts + [len(run.records)]
    parts = [run.part(a, b) for a, b in zip(cuts, cuts[1:], strict=False)]
    before = f"; {starts[0]} record(s) before the first are not a pass of it" if starts[0] else ""
    literal = sum(1 for _, _, lit in read if lit)
    by = (f" ({literal} read as its line only with its holes taking any characters, nothing else printing it)"
          if literal else "")
    return parts, f"cut at entry row {entry.line} of {fn.name}{by}: {len(parts)} pass(es){before}"


def prefix_holes(fn: Function, row: Row, record: Record) -> dict[str, str]:
    """Hole values of the mark *record* carries, read against the logger *row* prints through.

    The function builds its logger with the mark in it --
    ``tmpLog = MsgWrapper(self.logger, f" < jediTaskID={jediTaskID} >")`` -- and
    the mark on the record is that f-string filled in.  Only a logger bound in
    the function itself is read; one handed in says nothing here.
    """
    logger = _logger_of(row)
    for node in own_nodes(fn.node):
        if not (isinstance(node, ast.Assign) and any(ast.unparse(t) == logger for t in node.targets)):
            continue
        if not isinstance(node.value, ast.Call):
            continue
        for arg in node.value.args:
            if not isinstance(arg, ast.JoinedStr):
                continue
            for shape in _alternatives(arg, fn.node, node, expand=False):
                stripped = [(k, t.strip() if k == "lit" else t) for k, t in shape]
                pattern, groups = _regex(stripped, True)
                m = re.fullmatch(pattern, record.token.strip(), re.S)
                if m:
                    return {groups[g]: v.strip() for g, v in m.groupdict().items()}
    return {}


def binding_count(func: ast.AST, name: str) -> int:
    """How many places in *func* (not in nested defs) bind the bare *name*."""
    count = 0
    args = func.args if isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)) else None
    if args is not None:
        count += sum(1 for a in [*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg] if a is not None and a.arg == name)
    for node in own_nodes(func):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id == name:
            count += 1
        elif isinstance(node, ast.ExceptHandler) and node.name == name:
            count += 1
    return count


def reads_bound_once(func: ast.AST, expr: str) -> bool:
    """Whether every bare name *expr* reads (other than a called one) is bound once in *func*.

    A printed local means the same thing wherever the function reads it only
    then: ``commandStr`` bound on 126 is the pass's; ``tmpRet`` bound on eight
    lines is whichever came last (rule 17).
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return False
    called = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and id(n) not in called}
    return bool(names) and all(binding_count(func, n) == 1 for n in names)
