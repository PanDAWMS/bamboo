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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from bamboo.codemap import trace
from bamboo.codemap.panda import pathcond
from bamboo.codemap.panda.plugin import PandaCodeMapPlugin
from bamboo.codemap.panda.provenance import classify
from bamboo.codemap.panda.recognizers.selection import log_level

from . import TREE
from .records import CORPUS, LEVEL_OF_METHOD, Record, Run

#: the bamboo tree the map's code is imported from (the package's own; see :mod:`bamboo.querier`)
BAMBOO_TREE = str(TREE)

ROOTS = PandaCodeMapPlugin._resolve_roots(CORPUS)
LOOPS = (ast.For, ast.AsyncFor, ast.While)
EXITS = (ast.Return, ast.Raise)
JUMPS = (ast.Return, ast.Raise, ast.Continue, ast.Break)
DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
#: a row proves a path only when it says something -- a pattern of fewer
#: literal characters than this can be present by accident
MIN_PROOF = 8
#: where a function's print rows come from: ``walk`` (``trace.walk``'s skeleton, per
#: asking) or ``graph`` (the map's ``FunctionPrintsNode``), whose
#: rows carry no arms -- those are computed here as the walk computes them.  ``graph``
#: stops when the map read another
#: version of a file than the corpus holds (``anchor.blob_sha``)
ROWS_FROM = os.environ.get("ROWS_FROM", "graph")
#: where a function's own rows are looked for: ``ROWS_OWN_FILES=1`` (the default) the files
#: the map places its printed statements in (``PrintStatement.log_files``); ``0`` the files the map's
#: reading names, a junction's ``observable_log_files`` -- the caller's, which is where a line saying the junction
#: settled a value lands, not where the function's own lines do
OWN_FILES = os.environ.get("ROWS_OWN_FILES", "1") != "0"
#: ``ROWS_PREFIX=1``: a row whose map rows all carry ``PrintRow.prefix`` (what
#: its logger writes before the message: ``archiveJob\ <\ .*?\ >\ `` for a proxy method's create_tagged_logger) finds
#: only a record where that prefix stands right before the message -- the message alone (``start``, ``start with
#: ...``, ``not found``) is shared by many writers.  The prefix narrows which records count; it adds no literal.
PREFIX = os.environ.get("ROWS_PREFIX", "1") != "0"
PATTERN_MODES = ("map", "outside")
#: ``ROWS_OLD_EXITS=1`` reads the call's exit after a statement the earlier way:
#: a ``raise`` some ``try`` of the function catches taken as the exit, and a call on the way that a ``try`` catches
#: passed over -- its handler runs, and what lies below it may too
OLD_EXITS = os.environ.get("ROWS_OLD_EXITS", "0") == "1"
#: ``ROWS_KEEP_SHORT_MAP=1`` finds lines with a map pattern of fewer than ``MIN_SHAPE`` non-blank literal characters,
#: the earlier behaviour.  By default such a pattern is refused, as a shape built
#: here that is not ``usable`` is: ``to\ `` (ppEventServiceJob 5667, an append's text drawn alone) is in any line
#: holding "to ", and most readers of a row's hits ask only whether there is one
KEEP_SHORT_MAP = os.environ.get("ROWS_KEEP_SHORT_MAP", "0") == "1"


#: id of every module ``tree`` parsed -> its corpus path, to tell a corpus function from one built from a string
TREE_REL: dict[int, str] = {}


@functools.cache
def tree(rel: str) -> ast.Module:
    parsed = ast.parse((CORPUS / rel).read_text())
    pathcond.attach_parents(parsed)
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
    """The innermost statement of *func* that starts on *line*."""
    found = None
    for node in own_nodes(func):
        if isinstance(node, ast.stmt) and node.lineno == line and node is not func:
            if found is None or _depth(node) > _depth(found):
                found = node
    return found


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


def share_loop(a: ast.AST, b: ast.AST, func: ast.AST) -> bool:
    return bool(loops_over(a, func) & loops_over(b, func))


def _direct_prints(block: list, stop: ast.AST | None = None) -> list[ast.stmt]:
    out = []
    for s in block:
        if s is stop:
            break
        if is_print(s):
            out.append(s)
    return out


def _raised_prefix(stmt: ast.stmt, raised_at: set[int]) -> list[ast.stmt]:
    """Prints a ``try`` body ran before the line a traceback frame says raised."""
    if not isinstance(stmt, ast.Try):
        return []
    for s in stmt.body:
        if any(s.lineno <= line <= (s.end_lineno or s.lineno) for line in raised_at):
            return _direct_prints(stmt.body, stop=s)
    return []


def must_rows(func: ast.AST, site: ast.stmt, raised_at: set[int] = frozenset()) -> tuple[list, list]:
    """``(before, after)``: the prints every path through *site* runs in one call."""
    before: list[ast.stmt] = []
    after: list[ast.stmt] = []
    open_after = True
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
            elif raised_at:
                before.extend(_raised_prefix(s, raised_at))
        if open_after:
            for s in block[index + 1 :]:
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


def implying_rows(func: ast.AST, site: ast.stmt) -> list[ast.stmt]:
    """Prints in the site's own block that no path prints without passing the site.

    Any print in a later statement of its block, nested or not: nothing enters
    a block below its top, so a printed row there proves the site ran (a call
    or a jump in between only means the row may be missing when it did).  Or a
    row before it with nothing in between that could leave the block.  The
    other must rows prove only that the function got that far: ``871`` prints
    on every run of ``parse_job_output_report`` whichever arm follows it.
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
        elif may_divert(s):
            break
    return found


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


def exit_after(func: ast.AST, site: ast.stmt) -> ast.stmt | None:
    """The ``return``/``raise`` every path from *site* ends at, if there is one in reach.

    A ``raise`` some ``try`` around it in the function catches is none: the
    handler runs, and may go on below (8 prints of the corpus end at one).
    """
    end, _ = _exit_path(func, site)
    if isinstance(end, ast.Raise) and not OLD_EXITS and _catcher(func, end) is not None:
        return None
    return end


def _catcher(func: ast.AST, node: ast.AST, within: ast.AST | None = None) -> ast.Try | None:
    """The innermost ``try`` with handlers whose body holds *node*, inside *func* -- and outside *within*, when
    given: a ``try`` inside that statement catches what it raises before it is done."""
    previous: ast.AST = node
    for ancestor in pathcond.ancestors(node):
        if ancestor is func:
            return None
        if isinstance(ancestor, ast.Try) and ancestor.handlers and previous in ancestor.body:
            if within is not None and (ancestor is within or pathcond._within(ancestor, within)):
                return None
            return ancestor
        previous = ancestor
    return None


def _exit_path(func: ast.AST, site: ast.stmt) -> tuple[ast.stmt | None, list[ast.stmt]]:
    """``(the exit, the statements run on the way to it)`` -- a ``try``'s ``else`` among them."""
    node: ast.AST = site
    between: list[ast.stmt] = []
    while node is not func:
        parent = node.parent  # type: ignore[attr-defined]
        if isinstance(parent, ast.ExceptHandler):
            placed = ("body", parent.body)
        else:
            placed = block_of(parent, node)
        if placed is None:
            node = parent
            continue
        name, block = placed
        for s in block[block.index(node) + 1 :]:
            if isinstance(s, EXITS):
                return s, between
            if isinstance(s, (ast.Continue, ast.Break)) or contains(s, (ast.Continue, ast.Break)):
                return None, between
            between.append(s)
        if isinstance(parent, LOOPS):
            return None, between
        if isinstance(parent, ast.Try) and name == "body":
            between += parent.orelse
        if isinstance(parent, ast.ExceptHandler):
            node = parent.parent
            continue
        node = parent
    return None, between


def _caught_on_the_way(func: ast.AST, site: ast.stmt, end: ast.stmt) -> list[ast.Try]:
    """The ``try``s of the function that catch a call made after *site* on the way to *end*, *end*'s own included."""
    _, between = _exit_path(func, site)
    found: list[ast.Try] = []
    for s in [*between, end]:
        for n in own_nodes(s):
            if isinstance(n, ast.Call) and not is_log_call(n):
                t = _catcher(func, n, within=s if s is not end else None)
                if t is not None and t not in found:
                    found.append(t)
    return found


def beyond_exit(func: ast.AST, first: ast.stmt, other: ast.AST) -> ast.stmt | None:
    """The exit every path from *first* ends at, when *other* lies below it -- so one call that ran *first* did not
    run *other* afterwards, nor before (no loop holds both).

    A ``finally`` around *first* still runs.  So does the handler of a ``try``
    that catches a call on the way, and when a handler does not end the call
    (a ``return``, or a ``raise`` nothing in the function catches), what lies
    below it too -- then there is no exit to speak of.
    """
    end = exit_after(func, first)
    if end is None or other.lineno <= (end.end_lineno or end.lineno):  # type: ignore[attr-defined]
        return None
    for a in pathcond.ancestors(first):
        if a is func:
            break
        if isinstance(a, ast.Try) and any(other is s or pathcond._within(other, s) for s in a.finalbody):
            return None
    if share_loop(other, first, func):
        return None
    if not OLD_EXITS:
        for t in _caught_on_the_way(func, first, end):
            for h in t.handlers:
                if pathcond._within(other, h):
                    return None
                last = h.body[-1]
                if not (isinstance(last, ast.Return) or (isinstance(last, ast.Raise) and _catcher(func, t) is None)):
                    return None
    return end


def cannot_rows(func: ast.AST, site: ast.stmt) -> list[ast.stmt]:
    """The prints a call that ran *site* cannot show: below the exit the path from *site* ends at."""
    return [s for s in own_nodes(func) if is_print(s) and beyond_exit(func, site, s) is not None]


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


#: ``ROWS_KEEP_LEADING_HOLE=1`` keeps a non-capturing pattern's leading ``.*?``
KEEP_LEADING_HOLE = os.environ.get("ROWS_KEEP_LEADING_HOLE", "0") == "1"


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
    if not capture and not KEEP_LEADING_HOLE:
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
    """A map pattern held to what ``usable`` asks of a shape built here: ``MIN_SHAPE`` non-blank literal characters."""
    return KEEP_SHORT_MAP or len(_map_text(pattern).replace(" ", "")) >= MIN_SHAPE


@dataclass
class Row:
    rel: str
    func: str
    stmt: ast.stmt
    level: str
    text: str
    map_patterns: list[str] = field(default_factory=list)
    map_refused: list[str] = field(default_factory=list)
    map_arms: set[int] | None = None
    #: the map's word that its pattern is a question only together with the entity's
    #: key; every record read here was found by the key, so it is carried, not acted on
    map_keyed: bool = False
    shapes: list[list[Part]] = field(default_factory=list)
    capture_shapes: list[list[Part]] = field(default_factory=list)
    #: the map's prefix patterns of this statement's rows, when every row has one (``PREFIX``)
    map_prefixes: list[str] = field(default_factory=list)

    def written_by(self, record: Record) -> bool:
        """Whether *record* can be this row's line by its prefix: always when ``PREFIX`` is off or the map gives
        none; else the prefix stands where the line starts (``archiveJob < ... >`` is the token and the head of
        the message) or right after the token (``LogWrapper(self.logger, "<run>")``: the outer wrapper's mark
        comes first).  Checked apart from the patterns: a map pattern need not start where the message does
        (``sqlDelE + comment`` is drawn from `` /* DBProxy...``)."""
        if not (PREFIX and self.map_prefixes):
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
        found =[(re.compile(p, re.S), _map_literal(p), "map") for p in self.map_patterns]
        if mode == "outside":
            found += [(re.compile(_regex(s, False)[0], re.S), literal_of(s), "outside:3-4") for s in self._outside_shapes()]
        return found

    def absence_counts(self, mode: str) -> bool:
        """Whether this row being missing from a run says it did not print."""
        if self.map_covers():
            return True
        return mode == "outside" and bool(self.shapes) and all(usable(s) for s in self.shapes)

    def matches(self, record: Record, mode: str) -> tuple[int, str]:
        """``(literal, source)`` of the best pattern that finds the record; ``(0, "")`` when none does."""
        best = (0, "")
        if not self.written_by(record):
            return best
        for regex, literal, source in self.patterns(mode):
            if literal > best[0] and regex.search(record.text):
                best = (literal, source)
        return best

    def capture(self, record: Record) -> dict[str, str]:
        """Hole values in *record*, keyed by the expression each hole prints."""
        for shape in sorted(self.capture_shapes, key=literal_of, reverse=True):
            if not usable(shape):
                continue
            pattern, groups = _regex(shape, True)
            m = re.search(pattern, record.text, re.S)
            if m:
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
    """The map's ``FunctionPrintsNode`` for *owner*, or None: one connection per process.

    The client class is looked up at call time, so the
    in-memory map of :func:`bamboo.querier.mapdump.install` stands in for Neo4j the same way it does for the rest.
    """
    if owner in _GRAPH:
        return _GRAPH[owner]
    import asyncio
    import atexit

    from bamboo.codemap.lookup import CodeMap
    from bamboo.database import graph_database_client

    if "loop" not in _GRAPH:
        loop = asyncio.new_event_loop()
        client = graph_database_client.GraphDatabaseClient()
        loop.run_until_complete(client.connect())
        _GRAPH["loop"], _GRAPH["map"] = loop, CodeMap(client, map_id="panda", version=None)
        atexit.register(lambda: loop.run_until_complete(client.close()))
    found = _GRAPH["loop"].run_until_complete(_GRAPH["map"].function_prints([owner]))
    _GRAPH[owner] = found.get(owner)
    return _GRAPH[owner]


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
    if not OWN_FILES or ROWS_FROM != "graph":
        return today
    found = _prints_of(owner, lines)
    if found is None or not found.statements or any(s.log_files_basis != "logger" for s in found.statements):
        return today
    return sorted({f for s in found.statements for f in s.log_files})


def split_files(owner: str, lines: Iterable[int] = ()) -> list[list[str]]:
    """The distinct file sets *owner*'s placed statements land in, when there is more than one.

    Such a function's rows are all read in one run of one file, so a row that
    lands only in another file reads as missing there.  Not handled: 2 functions
    of 1093 (pilot_api's ``acquire_jobs``, ``update_job``), in no baseline.
    """
    found = _prints_of(owner, lines) if OWN_FILES and ROWS_FROM == "graph" else None
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
            f"{what}: the map read {len(bad)} file(s) at another version than the corpus ({CORPUS}): {shown}"
            " -- run with the map built from this corpus (MAP_DUMP)"
        )


def _graph_rows(rel: str, node: ast.AST, arms: tuple[int, ...]) -> tuple[dict[int, list], dict[int, list[int]]]:
    """``(line -> the node's rows, line -> arms)``: the arms as ``trace._Walk._rows_for`` gives them.

    An asked line is armed with the statement ``trace._statement_at`` finds for
    it, and is an arm of a print when ``pathcond.compatible(that statement, the
    call's message, current=True)``.
    """
    found = _function_prints(f"{rel}::{pathcond.qualified_name(node)}")
    if found is not None and found.anchor.blob_sha != _blob_sha(rel):
        # the map read another version of the file: its rows sit at other lines
        raise RuntimeError(
            f"the map's prints for {rel} are of blob {found.anchor.blob_sha[:12]}, the corpus file is "
            f"{_blob_sha(rel)[:12]} ({CORPUS}): run with ROWS_FROM=walk or the map's corpus"
        )
    by_line = {s.line: list(s.rows) for s in found.statements} if found is not None else {}
    armed = {}
    for line in list(arms) or [node.lineno]:
        held = trace._statement_at(node, line)
        if held is not None:
            armed[line] = held
    arms_of = {}
    for stmt in own_nodes(node):
        if is_print(stmt):
            message = stmt.value.args[0]
            arms_of[stmt.lineno] = sorted(a for a, held in armed.items() if pathcond.compatible(held, message, current=True))
    return by_line, arms_of


@functools.cache
def _prefix_regex(prefixes: tuple[str, ...]) -> re.Pattern[str]:
    return re.compile(prefixes[0] if len(prefixes) == 1 else "(?:" + "|".join(prefixes) + ")", re.S)


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


@functools.cache
def _function(rel: str, name: str, arms: tuple[int, ...], within: tuple[int, ...] = ()) -> Function:
    found = functions(rel, name)
    if not found:
        raise LookupError(f"no function {name} in {rel}")
    node = _holding(found, arms or within)
    by_line: dict[int, list] = {}
    arms_of: dict[int, list[int]] = {}
    if ROWS_FROM == "graph":
        by_line, arms_of = _graph_rows(rel, node, arms)
    else:
        walked = trace.walk(ROOTS, file=rel, owner=f"{rel}::{name}", lines=list(arms) or [node.lineno], classify=classify, reaching=False)
        for sk in walked.skeleton:
            if sk.kind == "print":
                by_line.setdefault(sk.line, []).append(sk)
    rows = []
    for stmt in own_nodes(node):
        if not is_print(stmt):
            continue
        call = stmt.value
        mapped = by_line.get(stmt.lineno, [])
        if ROWS_FROM == "graph":
            map_arms = set(arms_of.get(stmt.lineno, ())) if mapped else None
        else:
            map_arms = set().union(*[set(m.arms) for m in mapped]) if mapped else None
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
                map_arms,
                any(m.keyed for m in mapped if m.pattern),
                _alternatives(call.args[0], node, stmt),
                _capture_shapes(call.args[0], node, stmt),
                _prefixes(mapped),
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


def absence_trusted(row: Row, log, mode: str) -> tuple[bool, str]:
    """Whether the row missing from a run of *log* is evidence; the reason when not."""
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
    unambiguous: dict[int, list[int]]  # row line -> record indexes only this row matches

    def present(self, row: Row) -> bool:
        return bool(self.hits.get(row.line))

    def proven(self, row: Row) -> bool:
        return any(lit >= MIN_PROOF for _, lit, _ in self.hits.get(row.line, ()))

    def single_call(self) -> bool:
        return all(len(v) <= 1 for v in self.unambiguous.values())


def seen(fn: Function, run: Run, mode: str) -> Seen:
    hits: dict[int, list[tuple[int, int, str]]] = {}
    for i, record in enumerate(run.records):
        for row in fn.rows:
            literal, source = row.matches(record, mode)
            if literal:
                hits.setdefault(row.line, []).append((i, literal, source))
    return _seen(fn, run, mode, hits)


def _seen(fn: Function, run: Run, mode: str, hits: dict[int, list[tuple[int, int, str]]]) -> Seen:
    per_record: dict[int, set[int]] = {}
    for line, hs in hits.items():
        for i, _, _ in hs:
            per_record.setdefault(i, set()).add(line)
    unambiguous: dict[int, list[int]] = {}
    for i, lines in sorted(per_record.items()):
        if len(lines) == 1:
            unambiguous.setdefault(next(iter(lines)), []).append(i)
    return Seen(fn, run, mode, hits, unambiguous)


def claim(seens: list[Seen]) -> list[Seen]:
    """Give each record only to the rows that read it with the most literal among *seens*.

    A thin shape of one function (``{str(e)} : {…} in {…}``) also matches a
    line another function printed, and would count as printed there and read
    a value out of text it never wrote (register_files:778
    matched execute:168's traceback and read ``prodSourceLabel`` out of it).
    Only the functions given compete: a writer outside them does not take its
    record back from a thin shape inside them.
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
    """Indexes of the records in *run* that *fn*'s entry row reads: one per pass of *fn*."""
    entry = entry_row(fn)
    if entry is None:
        return []
    return sorted(i for i, _, _ in seen(fn, run, mode).hits.get(entry.line, ()))


def passes(fn: Function, run: Run, mode: str) -> tuple[list[Run], str]:
    """*run* cut at each record of *fn*'s entry row, with a word on how it was cut.

    Records before the first entry are not a pass of *fn*: another function
    that shares the mark printed them, or the log starts inside a pass.  They
    are counted in the word, not returned.
    """
    entry = entry_row(fn)
    if entry is None:
        return [run], f"{fn.name} has no entry row: the run is not split"
    starts = entries(fn, run, mode)
    if not starts:
        return [run], f"entry row {entry.line} of {fn.name} is not in the run: the run is not split"
    cuts = starts + [len(run.records)]
    parts = [run.part(a, b) for a, b in zip(cuts, cuts[1:], strict=False)]
    before = f"; {starts[0]} record(s) before the first are not a pass of it" if starts[0] else ""
    return parts, f"cut at entry row {entry.line} of {fn.name}: {len(parts)} pass(es){before}"


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
