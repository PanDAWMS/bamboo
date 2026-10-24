"""P: rows printed outside an arm's block that prove the arm ran.

``rows.implying_rows`` reads only the arm's own block.  A row elsewhere proves
an arm too when what it shows leaves no other way through:
``ppE -> {retEvS}`` printed just before archiveJob's ``if retEvS ... elif``
chain, or ``done new status={newTaskStatus}`` printed after the six arms of
prepareTasksToBeFinished_JEDI that assign it.

S* is an ``if`` on the arm's chain -- the arm reached from it through ``if``
bodies and ``else``s (and ``with`` bodies) only -- and the region is S* with
the statements after it in its block.  Every path through the region is
walked: an ``if`` both ways, a jump ends the path, a statement that calls
anything but a logger forks a copy that leaves there (the call raised), and a
loop or ``try`` is passed whole (what it binds becomes unknown).  A path is
dropped when

* **E1** -- a test on it comes out the other way from printed holes still
  valid there: nothing on the path, or between the row and S*, binds what
  they read (an attribute or a container: nor calls anything naming it);
* **E2** -- it misses a row the run printed in the region, or passes them in
  another order than their records;
* **E3** -- a test on it comes out the other way from a constant the path
  bound itself.  This is how a row after the arms rules out the arms that set
  the flag it sits under (``isOK = False``, ``timeoutFlag = True``);
* **E4** -- a printed row holds the arm's target, and the path's value there
  is another one; or, the path not binding it, a test the path passed says
  it cannot be the printed one (``if taskSpec.status not in ["topreprocess"]:
  taskSpec.status = "defined"``); or it is a local nothing binds before S*;
* **E5** (with :data:`ROW_RENDER`) -- the row as the path would print it,
  its message rendered with the bare names the path bound itself (a value,
  or the literal parts of the text it built, the rest left open), does not
  match the record: ``tmpLog.info(tmpErrStr)`` printed ``pending ...`` rules
  out the branch that bound ``tmpErrStr = f"finishing due to ..."``.

The arm is **positive** when some path is left and every one left passes it;
**soft** when the only ones left that miss it are copies that left at a call
-- proven if no call raised, and not counted.  **inconsistent**: no path fits
the printed rows, which says a row was read from another pass or another line.

Only rows with one record in the run, read by that row alone at
``rows.MIN_PROOF``, count; a row outside every loop printed twice means two
calls, a row of the loop around S* printed twice means two iterations, and
either stops the reading.  What this cannot see: two iterations that each
printed different rows once read as one pass (E2 then usually finds no path,
"inconsistent", but not always).  Values are printed holes only -- never the
question's DB rows, which hold what came after.

Beyond ``WIDEN_AT`` paths, paths that agree on the arm, the rows passed and
the target are joined into one that knows only what all of them knew
(:func:`widen`): that can turn a positive into nothing, never the reverse.

:func:`census` asks the same of the code alone: if the arm ran, which rows
would print, and would they rule out everything else.
"""
from __future__ import annotations

import ast
import functools
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

from bamboo.codemap.panda import pathcond  # noqa: E402

from . import evaluate as E
from . import handlers as H
from . import rows as R

MAX_PATHS = 3000
#: beyond this many paths, paths that agree on what the verdict reads are joined (:func:`widen`)
WIDEN_AT = 400
#: ``P_WIDEN_BY_CALLED=1`` joins only paths that agree on whether the target was handed to a call since it was bound,
#: the earlier behaviour.  By default that is joined as "any of them was": with P_WRITTEN_BACK an execute that writes
#: nothing back no longer hands the bind map, and updateJobStatus 241's paths that did and did not pass the
#: corrupted-zip INSERT stopped joining -- 138 -> 250 arm paths, 86 -> 146 s
WIDEN_BY_CALLED = os.environ.get("P_WIDEN_BY_CALLED", "0") == "1"
#: census walks every arm path a second time (with reuse it costs little).  P_MAX_ARM_PATHS=48
#: brings back the stride sample of that many, under which one path in or out moved verdicts and minutes
MAX_ARM_PATHS = int(os.environ["P_MAX_ARM_PATHS"]) if os.environ.get("P_MAX_ARM_PATHS") else None
#: E5: check a printed row against the path's own rendering of it (on by default; P_ROW_RENDER=0 turns it off)
ROW_RENDER = os.environ.get("P_ROW_RENDER", "1") != "0"
#: ``P_CALLS_RAISE=0`` (``census_p.py --calls-never-raise``): no call leaves by an exception -- the upper bound of
#: knowing which calls cannot raise
CALLS_RAISE = os.environ.get("P_CALLS_RAISE", "1") != "0"
#: E6: a path that leaves by an exception is decided by the rows of the handler that catches it (:mod:`bamboo.querier.handlers`):
#: against a run, no such row drops it; in census, a traceback at a line after the arm proves an arm path, and
#: a printing handler settles a path for the log to decide (on by default; P_HANDLER_RULE=0 turns it off)
HANDLER_RULE = os.environ.get("P_HANDLER_RULE", "1") != "0"
#: census: an arm path whose second walk would read what an earlier one read takes that walk's outcome
#: (:func:`_census_at`; on by default, P_REUSE_WALKS=0 turns it off -- the output must not move)
REUSE_WALKS = os.environ.get("P_REUSE_WALKS", "1") != "0"
#: census: among S* of one kind (positive, by handler), the one where it holds on the larger share of arm paths
#: (:func:`_share`, the measure ``every`` is 1 of).  P_LEVEL_BY=count brings back the larger
#: number of proven paths, which favours an outer S* for having more paths
LEVEL_BY = os.environ.get("P_LEVEL_BY", "share")
#: a cursor's ``execute(sql, V)`` / ``executemany(sql, V)`` whose statement the map read and found no
#: ``RETURNING ... INTO`` in (``sql.written_back_at`` answers ``()``) is not taken to change V: PanDA's
#: WrappedCursor touches no key but those binds.
#: P_WRITTEN_BACK=0 takes every such call to rewrite V whole, as before
WRITTEN_BACK = os.environ.get("P_WRITTEN_BACK", "1") != "0"
OPAQUE = (ast.For, ast.AsyncFor, ast.While, ast.Try, ast.Match) + ((ast.TryStar,) if hasattr(ast, "TryStar") else ())
NAMED_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
#: builtins that read their arguments and change nothing (a call to one still may raise)
READERS = {
    "str", "len", "int", "float", "bool", "repr", "sorted", "list", "dict", "set", "tuple", "min", "max", "sum",
    "any", "all", "isinstance", "type", "format", "abs", "round", "enumerate", "zip", "range", "hasattr", "frozenset",
}


class _NotBound:
    def __repr__(self) -> str:
        return "NOT_BOUND"


NOT_BOUND: Any = _NotBound()


class TooMany(Exception):
    pass


# ---------------------------------------------------------------- reading expressions


@functools.lru_cache(maxsize=None)
def chains(text: str) -> frozenset[str]:
    """The name / attribute / subscript chains *text* reads, each as long as it goes."""
    try:
        node = ast.parse(text, mode="eval").body
    except SyntaxError:
        return frozenset()
    return frozenset(_chains(node))


def _chains(node: ast.AST) -> set[str]:
    out: set[str] = set()
    if isinstance(node, (ast.Name, ast.Attribute, ast.Subscript)):
        out.add(ast.unparse(node))
        cur: ast.AST = node
        while True:
            if isinstance(cur, ast.Attribute):
                cur = cur.value
            elif isinstance(cur, ast.Subscript):
                out |= _chains(cur.slice)
                cur = cur.value
            elif isinstance(cur, ast.Call):
                for a in [*cur.args, *(k.value for k in cur.keywords)]:
                    out |= _chains(a)
                cur = cur.func
            else:
                break
        if not isinstance(cur, ast.Name):
            out |= _chains(cur)
        return out
    for child in ast.iter_child_nodes(node):
        out |= _chains(child)
    return out


@functools.lru_cache(maxsize=None)
def root(chain: str) -> str:
    try:
        node = ast.parse(chain, mode="eval").body
    except SyntaxError:
        return ""
    while isinstance(node, (ast.Attribute, ast.Subscript, ast.Call)):
        node = node.func if isinstance(node, ast.Call) else node.value
    return node.id if isinstance(node, ast.Name) else ""


def affects(bound: str, read: frozenset[str] | set[str]) -> bool:
    """Whether binding *bound* changes what any of the chains *read* holds."""
    return any(
        c == bound or c.startswith((bound + ".", bound + "[")) or bound.startswith((c + ".", c + "["))
        for c in read
    )


def _flat(target: ast.expr) -> list[ast.expr]:
    if isinstance(target, (ast.Tuple, ast.List)):
        return [t for e in target.elts for t in _flat(e)]
    if isinstance(target, ast.Starred):
        return _flat(target.value)
    return [target]


def binds(stmt: ast.AST) -> list[tuple[str, ast.expr | None]]:
    """``(text bound, value node or None)`` for what this one node binds, not what is nested in it."""
    out: list[tuple[str, ast.expr | None]] = []
    if isinstance(stmt, ast.Assign):
        for t in stmt.targets:
            single = not isinstance(t, (ast.Tuple, ast.List))
            out += [(ast.unparse(x), stmt.value if single else None) for x in _flat(t)]
    elif isinstance(stmt, ast.AugAssign):
        out.append((ast.unparse(stmt.target), None))
    elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
        out.append((ast.unparse(stmt.target), stmt.value))
    elif isinstance(stmt, ast.Delete):
        out += [(ast.unparse(x), None) for t in stmt.targets for x in _flat(t)]
    elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
        out += [((a.asname or a.name).split(".")[0], None) for a in stmt.names]
    elif isinstance(stmt, NAMED_DEFS):
        out.append((stmt.name, None))
    elif isinstance(stmt, (ast.For, ast.AsyncFor)):
        out += [(ast.unparse(x), None) for x in _flat(stmt.target)]
    elif isinstance(stmt, (ast.With, ast.AsyncWith)):
        out += [(ast.unparse(x), None) for i in stmt.items if i.optional_vars is not None for x in _flat(i.optional_vars)]
    elif isinstance(stmt, ast.ExceptHandler) and stmt.name:
        out.append((stmt.name, None))
    elif isinstance(stmt, ast.NamedExpr):
        out.append((ast.unparse(stmt.target), None))
    return out


def _own_binds(stmt: ast.AST) -> list[tuple[str, ast.expr | None]]:
    """What a simple statement binds, walrus included."""
    out = binds(stmt)
    out += [(ast.unparse(n.target), None) for n in R.own_nodes(stmt) if isinstance(n, ast.NamedExpr)]
    return out


def deep_binds(stmt: ast.AST) -> set[str]:
    return {t for n in R.own_nodes(stmt) for t, _ in binds(n)}


def can_raise(node: ast.AST) -> bool:
    return CALLS_RAISE and any(isinstance(n, ast.Call) and not R.is_log_call(n) for n in R.own_nodes(node))


def mutating_roots(node: ast.AST) -> set[str]:
    """Names a call in *node* is handed or called on -- what it could change."""
    out: set[str] = set()
    for n in R.own_nodes(node):
        if isinstance(n, ast.Call) and not R.is_log_call(n):
            if isinstance(n.func, ast.Name) and n.func.id in READERS:
                continue
            kept = _read_only_map(n)
            out |= {m.id for m in ast.walk(n) if isinstance(m, ast.Name) and id(m) not in kept}
    return out


_WRITTEN_BACK: dict[tuple[str, int], tuple[str, ...] | None] = {}
_written_back_at: Callable[..., Any] | None = None


def written_back(call: ast.Call) -> tuple[str, ...] | None:
    """The map's ``sql.written_back_at`` for *call*: the keys its statement writes back, () for none, None unknown."""
    global _written_back_at
    module = next((a for a in pathcond.ancestors(call) if isinstance(a, ast.Module)), None)
    rel = R.TREE_REL.get(id(module)) if module is not None else None
    if rel is None:
        return None
    key = (rel, call.lineno)
    if key not in _WRITTEN_BACK:
        if _written_back_at is None:
            try:
                from bamboo.codemap.panda.sql import written_back_at as found
            except ImportError:
                import sys

                print(f"printed: {R.BAMBOO_TREE} has no sql.written_back_at; every execute rewrites its map", file=sys.stderr)
                found = lambda path, line: None  # noqa: E731
            _written_back_at = found
        _WRITTEN_BACK[key] = _written_back_at(R.CORPUS / rel, call.lineno)
    return _WRITTEN_BACK[key]


def _read_only_map(call: ast.Call) -> set[int]:
    """ids of the Name nodes in a cursor execute's map argument, when the map says its statement writes no key back."""
    if not (
        WRITTEN_BACK and isinstance(call.func, ast.Attribute) and call.func.attr in ("execute", "executemany")
        and len(call.args) >= 2
    ):
        return set()
    if written_back(call) != ():
        return set()
    inside = [m for m in ast.walk(call.args[1]) if isinstance(m, ast.Name)]
    # a name the call also reads elsewhere (the statement, the receiver) is still handed there
    elsewhere = {m.id for m in ast.walk(call) if isinstance(m, ast.Name) and all(m is not i for i in inside)}
    return {id(m) for m in inside if m.id not in elsewhere}


def chain_of_arm(func: ast.AST, stmt: ast.stmt) -> list[ast.If]:
    """The ``if``s the arm sits under through bodies and ``else``s only, innermost first."""
    found = []
    node: ast.AST = stmt
    while node is not func:
        parent = node.parent  # type: ignore[attr-defined]
        if isinstance(parent, ast.If) and (node in parent.body or node in parent.orelse):
            found.append(parent)
        elif not (isinstance(parent, (ast.With, ast.AsyncWith)) and node in parent.body):
            break
        node = parent
    return found


def region_of(s_star: ast.stmt) -> list[ast.stmt]:
    placed = R.block_of(s_star.parent, s_star)  # type: ignore[attr-defined]
    if placed is None:
        return []
    _, block = placed
    return block[block.index(s_star) :]


def in_region(node: ast.AST, region: list[ast.stmt]) -> ast.stmt | None:
    """The region statement *node* is in, or None."""
    ids = {id(s): s for s in region}
    for a in [node, *pathcond.ancestors(node)]:
        if id(a) in ids:
            return ids[id(a)]
    return None


def innermost_loop(node: ast.AST, func: ast.AST) -> ast.AST | None:
    previous = node
    for a in pathcond.ancestors(node):
        if a is func:
            return None
        if isinstance(a, R.LOOPS) and previous in a.body:
            return a
        previous = a
    return None


def loops_around(node: ast.AST, func: ast.AST) -> list[ast.AST]:
    out = []
    previous = node
    for a in pathcond.ancestors(node):
        if a is func:
            break
        if isinstance(a, R.LOOPS) and previous in a.body:
            out.append(a)
        previous = a
    return out


def target_of(stmt: ast.stmt) -> str:
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
        t = stmt.targets[0]
    elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
        t = stmt.target
    else:
        return ""
    return ast.unparse(t) if isinstance(t, (ast.Name, ast.Attribute, ast.Subscript)) else ""


def _params(func: ast.AST) -> set[str]:
    a = func.args  # type: ignore[attr-defined]
    return {x.arg for x in [*a.posonlyargs, *a.args, *a.kwonlyargs, a.vararg, a.kwarg] if x is not None}


def unbound_before(func: ast.AST, s_star: ast.stmt, name: str) -> bool:
    """Whether a bare local can hold nothing at S*: no binding before it, none in a loop around it."""
    if not name.isidentifier() or name in _params(func):
        return False
    loops = loops_around(s_star, func)
    for n in R.own_nodes(func):
        if not any(t == name for t, _ in binds(n)):
            continue
        if getattr(n, "lineno", 10**9) < s_star.lineno:
            return False
        if any(lp in pathcond.ancestors(n) for lp in loops):
            return False
    return True


def entry_state(func: ast.AST, s_star: ast.stmt, names: set[str]) -> dict[str, Any]:
    """Constants bare locals hold at S*: the binding every path to it passes last, in the same iteration.

    ``isOK = True`` four lines above ``if not toSkip: if isOK:`` holds at the
    inner ``if`` whatever ran before.  A binding before a loop around S* does
    not: another iteration may have changed it.
    """
    out: dict[str, Any] = {}
    loops = loops_around(s_star, func)
    for name in names:
        if not name.isidentifier():
            continue
        b = R._dominating_binding(func, s_star, name)
        if not (
            isinstance(b, ast.Assign) and len(b.targets) == 1 and isinstance(b.targets[0], ast.Name)
            and isinstance(b.value, ast.Constant)
        ):
            continue
        around_b = loops_around(b, func)
        if any(lp not in around_b for lp in loops) or R._maybe_appended(func, b, s_star, name, []):
            continue
        out[name] = b.value.value
    return out


# ---------------------------------------------------------------- rendering a row (E5)

Part = tuple[str, str]  # ("lit", text) or ("any", "")
ANY: Part = ("any", "")


@dataclass(frozen=True)
class Shape:
    """A text known only in part: literal parts in order, the rest open."""

    parts: tuple[Part, ...]

    def __repr__(self) -> str:
        return "Shape(" + "".join(t if k == "lit" else "*" for k, t in self.parts) + ")"


def _norm(parts: list[Part]) -> list[Part]:
    out: list[Part] = []
    for k, t in parts:
        if k == "lit" and not t:
            continue
        if out and k == "lit" and out[-1][0] == "lit":
            out[-1] = ("lit", out[-1][1] + t)
        elif out and k == "any" and out[-1][0] == "any":
            continue
        else:
            out.append((k, t))
    return out


def _as_text(v: Any) -> list[Part] | None:
    """A path value as text parts, when it is one an f-string prints as ``str(v)``."""
    if isinstance(v, Shape):
        return list(v.parts)
    if v is None or type(v) in (str, int, bool):
        return [("lit", str(v))]
    return None


def _is_text(node: ast.AST, state: dict[str, Any]) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, str)
    if isinstance(node, ast.JoinedStr):
        return True
    if isinstance(node, ast.Name):
        v = state.get(node.id, E.UNKNOWN)
        return isinstance(v, (str, Shape))
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _is_text(node.left, state) or _is_text(node.right, state)
    return False


def render(node: ast.AST, state: dict[str, Any]) -> tuple[list[Part], bool]:
    """*node*'s text with the bare names *state* holds, and whether any was used."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [("lit", node.value)], False
    if isinstance(node, ast.JoinedStr):
        out: list[Part] = []
        used = False
        for part in node.values:
            if isinstance(part, ast.Constant):
                out.append(("lit", str(part.value)))
            elif (
                isinstance(part, ast.FormattedValue) and part.conversion == -1 and part.format_spec is None
                and isinstance(part.value, ast.Name) and part.value.id in state
                and _as_text(state[part.value.id]) is not None
            ):
                out += _as_text(state[part.value.id]) or []
                used = True
            else:
                out.append(ANY)
        return _norm(out), used
    if isinstance(node, ast.Name) and node.id in state and isinstance(state[node.id], (str, Shape)):
        return _norm(_as_text(state[node.id]) or []), True
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add) and _is_text(node, state):
        left, lu = render(node.left, state)
        right, ru = render(node.right, state)
        return _norm(left + right), lu or ru
    return [ANY], False


def shape_value(node: ast.AST, state: dict[str, Any], before: Any = None) -> Any:
    """The text shape *node* builds (appended to *before* for ``+=``), or E.UNKNOWN."""
    if not _is_text(node, state) and before is None:
        return E.UNKNOWN
    parts, _ = render(node, state)
    if before is not None:
        head = _as_text(before) if isinstance(before, (str, Shape)) else None
        if head is None:
            return E.UNKNOWN
        parts = _norm(head + parts)
    if not any(k == "lit" for k, _ in parts):
        return E.UNKNOWN
    if all(k == "lit" for k, _ in parts):
        return "".join(t for _, t in parts)
    return Shape(tuple(parts))


def _regex(parts: list[Part], whole: bool) -> str:
    out = []
    for i, (k, t) in enumerate(parts):
        if k == "lit":
            if not whole:
                if "\n" in t:  # a record may hold only the first line of it
                    out.append(re.escape(t.split("\n", 1)[0]))
                    break
                if i == len(parts) - 1:
                    t = t.rstrip()
            out.append(re.escape(t))
        else:
            out.append(".*?" if not whole else ".*")
    return "".join(out)


def renders_in(parts: list[Part], text: str) -> bool:
    """Whether a record's text can hold the message rendered as *parts* (searched, as rows captures)."""
    return re.search(_regex(parts, False), text, re.S) is not None


def disjoint(a: list[Part], b: list[Part]) -> bool:
    """Whether no one text can be both *a* and *b* (each the whole message)."""
    a, b = _norm(list(a)), _norm(list(b))
    a_open, b_open = any(k == "any" for k, _ in a), any(k == "any" for k, _ in b)
    if not a_open and not b_open:
        return "".join(t for _, t in a) != "".join(t for _, t in b)
    if not a_open or not b_open:
        known, pattern = (a, b) if not a_open else (b, a)
        return re.fullmatch(_regex(pattern, True), "".join(t for _, t in known), re.S) is None
    head_a = a[0][1] if a[0][0] == "lit" else ""
    head_b = b[0][1] if b[0][0] == "lit" else ""
    if not (head_a.startswith(head_b) or head_b.startswith(head_a)):
        return True
    tail_a = a[-1][1] if a[-1][0] == "lit" else ""
    tail_b = b[-1][1] if b[-1][0] == "lit" else ""
    return not (tail_a.endswith(tail_b) or tail_b.endswith(tail_a))


def _rendered_names(node: ast.AST | None) -> set[str]:
    """The bare names :func:`render` can put into *node*'s text."""
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, ast.JoinedStr):
        return {v.value.id for v in node.values if isinstance(v, ast.FormattedValue) and isinstance(v.value, ast.Name)}
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _rendered_names(node.left) | _rendered_names(node.right)
    return set()


def message_of(stmt: ast.stmt) -> ast.expr | None:
    call = getattr(stmt, "value", None)
    if isinstance(call, ast.Call) and call.args:
        return call.args[0]
    return None


# ---------------------------------------------------------------- holes


@dataclass
class Hole:
    expr: str
    value: Any  # the printed text, or None when only the shape is known (census)
    source: str
    chains: frozenset[str]
    roots: frozenset[str]
    #: an attribute, an item, or a container: a call naming it can change it
    sensitive: bool

    @classmethod
    def make(cls, expr: str, value: str | None, source: str) -> "Hole":
        cs = chains(expr)
        sensitive = not expr.isidentifier()
        if value is not None and not sensitive:
            try:
                sensitive = isinstance(ast.literal_eval(value), (list, dict, set, tuple))
            except (ValueError, SyntaxError, MemoryError, RecursionError):
                pass
        return cls(expr, value, source, cs, frozenset(root(c) for c in cs) - {""}, sensitive)

    def valid_after(self, marks: frozenset[str] | set[str]) -> bool:
        for m in marks:
            if m.endswith("()"):
                if self.sensitive and m[:-2] in self.roots:
                    return False
            elif affects(m, self.chains):
                return False
        return True


@functools.lru_cache(maxsize=None)
def _statements(func: ast.AST) -> list[tuple[int, ast.AST, frozenset[str], frozenset[str]]]:
    """``(line, node, texts bound, names a call is handed)`` for every node of *func* that binds or calls."""
    out = []
    for n in R.own_nodes(func):
        if not isinstance(n, (ast.stmt, ast.ExceptHandler)) or n is func:
            continue
        texts = frozenset(t for t, _ in binds(n))
        if isinstance(n, (ast.If, ast.While)):
            calls = frozenset(mutating_roots(n.test))
        elif isinstance(n, (ast.For, ast.AsyncFor)):
            calls = frozenset(mutating_roots(n.iter))
        elif isinstance(n, (ast.With, ast.AsyncWith)):
            calls = frozenset(set().union(*[mutating_roots(i.context_expr) for i in n.items]))
        elif isinstance(n, (ast.Try, ast.ExceptHandler, ast.Match) + NAMED_DEFS):
            calls = frozenset()
        else:
            calls = frozenset(mutating_roots(n))
        if texts or calls:
            out.append((n.lineno, n, texts, calls))
    return out


def valid_before(func: ast.AST, row_stmt: ast.stmt, s_star: ast.stmt, hole: Hole) -> bool:
    """Whether a hole printed at *row_stmt* still holds at S*: nothing in between binds it (or, sensitive, is handed it)."""
    loops = [lp for lp in loops_around(s_star, func) if lp not in list(pathcond.ancestors(row_stmt))]

    def breaks(texts: frozenset[str], calls: frozenset[str]) -> bool:
        if any(affects(t, hole.chains) for t in texts):
            return True
        return hole.sensitive and bool(calls & hole.roots)

    for line, node, texts, calls in _statements(func):
        between = row_stmt.lineno < line < s_star.lineno
        looped = any(lp is node or lp in pathcond.ancestors(node) for lp in loops)
        if (between or looped) and breaks(texts, calls):
            return False
    return True


# ---------------------------------------------------------------- paths through a region


@dataclass
class Path:
    #: texts bound on the path -> value (E.UNKNOWN when not known)
    state: dict[str, Any] = field(default_factory=dict)
    arm: bool = False
    left: str = ""
    seen: tuple[int, ...] = ()
    #: printed row line -> what was bound / handed to a call after it
    since: dict[int, frozenset[str]] = field(default_factory=dict)
    marks: frozenset[str] = frozenset()
    #: (test, outcome, state then) for tests that read the target while the path has not bound it
    prior: tuple = ()
    target_called: bool = False
    tests: tuple = ()
    at_row: tuple = ()
    #: atom -> truth a branch taken earlier on the path settled (``not toSkip`` false: ``toSkip`` true)
    facts: dict[str, bool] = field(default_factory=dict)
    #: (row line, its message as this path renders it) -- E5, census only
    rendered: tuple = ()

    def copy(self) -> "Path":
        return Path(
            dict(self.state), self.arm, self.left, self.seen, dict(self.since), self.marks, self.prior,
            self.target_called, self.tests, self.at_row, dict(self.facts), self.rendered,
        )

    def key(self) -> str:
        # since's sets are sorted: a set prints in an order that depends on the hash seed and on how it was
        # built, and two equal paths whose sets printed differently did not merge (PYTHONHASHSEED moved
        # census_p's path counts).  Values go through _value_key; the rest is repr, which is fast.
        return repr((
            self.arm, bool(self.left), self.left.split(" ")[0] if self.left else "", self.seen,
            sorted((k, _value_key(v)) for k, v in self.state.items()), sorted((line, sorted(m)) for line, m in self.since.items()),
            sorted(self.marks), self.prior, self.target_called, self.at_row, sorted(self.facts.items()), self.rendered,
        ))


def _canon(v: Any) -> str:
    """``repr`` with sets and dicts in sorted order: equal values give one text.

    A set iterates in an order that depends on the hash seed and on how it was
    built, so ``repr`` of two equal sets can differ -- and two equal paths
    would not merge (PYTHONHASHSEED moved census_p's path counts).
    """
    if isinstance(v, (set, frozenset)):
        return f"{type(v).__name__}({sorted(_canon(x) for x in v)})"
    if isinstance(v, dict):
        return "{" + ", ".join(sorted(f"{_canon(k)}: {_canon(x)}" for k, x in v.items())) + "}"
    if isinstance(v, (list, tuple)):
        inner = ", ".join(_canon(x) for x in v)
        return f"[{inner}]" if isinstance(v, list) else f"({inner}{',' if len(v) == 1 else ''})"
    return repr(v)


def _value_key(v: Any) -> str:
    """``repr`` of a value, through :func:`_canon` only when it is a container (one may hold a set)."""
    return _canon(v) if isinstance(v, (set, frozenset, dict, list, tuple)) else repr(v)


def merge(paths: list[Path]) -> list[Path]:
    kept: dict[str, Path] = {}
    for p in paths:
        kept.setdefault(p.key(), p)
    return list(kept.values())


def widen(paths: list[Path], target: str) -> list[Path]:
    """Join paths that agree on the arm, the rows passed, how they left and the target, keeping what all of them know.

    A joined path knows less -- a value that differs becomes unknown, a fact or
    a constraint only one of them had is dropped, a binding any of them made
    counts -- so it can only be dropped by less: joining can lose a positive,
    never make one.  A call any of them handed the target to counts too
    (``target_called``: the value at a row is then unknown and E4 says
    nothing), unless ``WIDEN_BY_CALLED``.
    """
    groups: dict[str, list[Path]] = {}
    for p in paths:
        agree = (p.arm, p.left.split(" ")[0] if p.left else "", p.seen, p.at_row, target in p.state)
        groups.setdefault(_canon(agree + ((p.target_called,) if WIDEN_BY_CALLED else ())), []).append(p)
    out = []
    for group in groups.values():
        base = group[0].copy()
        for q in group[1:]:
            keys = set(base.state) | set(q.state)
            base.state = {
                k: base.state[k] if k in base.state and k in q.state and _value_key(base.state[k]) == _value_key(q.state[k]) else E.UNKNOWN
                for k in keys
            }
            base.since = {line: base.since.get(line, frozenset()) | q.since.get(line, frozenset()) for line in set(base.since) | set(q.since)}
            base.marks |= q.marks
            base.target_called = base.target_called or q.target_called
            base.facts = {k: v for k, v in base.facts.items() if q.facts.get(k) == v}
            base.prior = tuple(x for x in base.prior if x in q.prior)
            base.rendered = tuple(x for x in base.rendered if x in q.rendered)
        out.append(base)
    return out


class Walker:
    """Every path through a region, with the printed rows and their holes as the evidence."""

    def __init__(
        self,
        func: ast.AST,
        arm: ast.stmt,
        *,
        target: str,
        relevant: set[str],
        order: dict[int, tuple[ast.stmt, int]],
        before: list[Hole],
        holes_of: dict[int, list[Hole]],
        decide: Callable[["Walker", ast.If, Path], list[bool]],
        check_row: Callable[["Walker", int, Path], str],
        track: bool = False,
        require_all: bool = True,
    ) -> None:
        self.func = func
        self.arm = arm
        self.target = target
        self.target_attr = bool(target) and not target.isidentifier()
        self.target_root = root(target) if target else ""
        self.relevant = set(relevant) | ({target} if target else set())
        self.rel_roots = {root(c) for c in self.relevant} - {""}
        self.order = order
        self.before = before
        self.holes_of = holes_of
        self.decide = decide
        self.check_row = check_row
        self.track = track
        self.require_all = require_all
        self.dropped: Counter[str] = Counter()
        self.cuts: Counter[str] = Counter()
        self.widened = 0
        #: the most paths any statement left (census_p --record)
        self.most = 0
        self.facts: dict[str, str] = {}
        #: E5: row line -> the record's text (against a run) / the arm path's rendering (census)
        self.row_texts: dict[int, str] = {}
        self.row_renders: dict[int, tuple] = {}
        self.region: list[ast.stmt] = []

    # -- environment

    def env(self, p: Path, with_holes: bool = True) -> E.Env:
        env = E.Env()
        if with_holes:
            for h in self.before:
                if h.value is not None and h.valid_after(p.marks):
                    env.put_printed(h.expr, h.value, h.source)
            for line in p.seen:
                for h in self.holes_of.get(line, ()):
                    if h.value is not None and h.valid_after(p.since[line]):
                        env.put_printed(h.expr, h.value, h.source)
        for t, v in p.state.items():
            if v is E.UNKNOWN or isinstance(v, Shape):
                env.stale[t] = "bound on the path"
            else:
                env.put(t, v, "bound on the path")
        return env

    def available(self, p: Path) -> set[str]:
        out = {h.expr for h in self.before if h.valid_after(p.marks)}
        for line in p.seen:
            out |= {h.expr for h in self.holes_of.get(line, ()) if h.valid_after(p.since[line])}
        return out

    # -- walking

    def run(self, region: list[ast.stmt]) -> list[Path]:
        if self.require_all and self.order:
            # nothing after the last printed row can drop a path or change whether it passed the arm
            last = max(region.index(in_region(stmt, region)) for stmt, _ in self.order.values())
            region = region[: last + 1]
        self.region = region
        start = Path(state=entry_state(self.func, region[0], self.relevant))
        paths = self.block(region, [start])
        if not self.require_all:
            return paths
        out = []
        for p in paths:
            if set(self.order) - set(p.seen):
                self.dropped["E2 missed"] += 1
            else:
                out.append(p)
        return out

    def block(self, stmts: list[ast.stmt], paths: list[Path]) -> list[Path]:
        for s in stmts:
            nxt: list[Path] = []
            for p in paths:
                nxt.extend([p] if p.left else self.stmt(s, p))
            paths = merge(nxt)
            if len(paths) > WIDEN_AT:
                paths = widen(paths, self.target)
                self.widened += 1
            if len(paths) > self.most:
                self.most = len(paths)
            if len(paths) > MAX_PATHS:
                raise TooMany(len(paths))
        return paths

    def leave(self, p: Path, why: str) -> Path:
        q = p.copy()
        q.left = why
        return q

    def stmt(self, s: ast.stmt, p: Path) -> list[Path]:
        if isinstance(s, ast.If):
            return self.branch(s, p)
        if isinstance(s, (ast.With, ast.AsyncWith)):
            out = []
            if any(can_raise(i.context_expr) for i in s.items):
                out.append(self.leave(p, f"call at {s.lineno}"))
            q = p.copy()
            self.call(q, set().union(*[mutating_roots(i.context_expr) for i in s.items]))
            self.bind(q, binds(s))
            return out + self.block(s.body, [q])
        if isinstance(s, R.JUMPS):
            return [self.leave(p, f"{type(s).__name__.lower()} at {s.lineno}")]
        if isinstance(s, OPAQUE):
            return self.opaque(s, p)
        return self.simple(s, p)

    def branch(self, s: ast.If, p: Path) -> list[Path]:
        out = []
        if can_raise(s.test):
            out.append(self.leave(p, f"call at {s.lineno}"))
        q0 = p.copy()
        self.call(q0, mutating_roots(s.test))
        for outcome in self.decide(self, s, q0):
            q = q0.copy()
            if self.track:
                q.tests += ((s.lineno, outcome),)
            if self.target and self.target not in q.state and affects(self.target, chains(ast.unparse(s.test))):
                q.prior += ((ast.unparse(s.test), outcome, tuple(sorted(q.state.items(), key=lambda kv: kv[0]))),)
            self.settle(q, s.test, outcome)
            out.extend(self.block(s.body if outcome else s.orelse, [q]))
        return out

    def settle(self, q: Path, test: ast.expr, outcome: bool) -> None:
        """Keep what taking this branch says about the test's atoms the path did not know."""
        found = E.atoms(test)
        if len(found) > 10:
            return
        try:
            solved = E.solve(ast.unparse(test), outcome, self.env(q, with_holes=False))
        except Exception:
            return
        for atom, truth in solved.settled.items():
            if chains(atom) and all(c in self.relevant or any(affects(c, {r}) for r in self.relevant) for c in chains(atom)):
                q.facts[atom] = truth

    def truth(self, test: ast.expr, p: Path, env: E.Env) -> Any:
        """The test's value from the path's facts and *env*; E.UNKNOWN when they do not decide it."""
        try:
            return E._truth(test, p.facts, env)
        except Exception:
            return E.UNKNOWN

    def simple(self, s: ast.stmt, p: Path) -> list[Path]:
        out = []
        if can_raise(s):
            out.append(self.leave(p, f"call at {s.lineno}"))
        q = p.copy()
        if s is self.arm:
            q.arm = True
        appended = None
        if (
            ROW_RENDER and isinstance(s, ast.AugAssign) and isinstance(s.op, ast.Add) and isinstance(s.target, ast.Name)
            and isinstance(q.state.get(s.target.id), (str, Shape))
        ):
            appended = shape_value(s.value, q.state, before=q.state[s.target.id])
        self.call(q, mutating_roots(s))
        self.bind(q, _own_binds(s))
        if appended is not None and s.target.id in q.state:  # type: ignore[union-attr]
            q.state[s.target.id] = appended  # type: ignore[union-attr]
        if s.lineno in self.order and self.order[s.lineno][0] is s:
            if not self.pass_row(q, s.lineno):
                return out
        out.append(q)
        return out

    def opaque(self, s: ast.stmt, p: Path) -> list[Path]:
        out = []
        if R.may_divert(s):
            jumps = R.contains(s, (ast.Return, ast.Raise))
            out.append(self.leave(p, f"{'jump' if jumps else 'call'} in {type(s).__name__.lower()} at {s.lineno}"))
        q = p.copy()
        self.call(q, mutating_roots(s))
        self.bind(q, [(t, None) for t in deep_binds(s)])
        inside = sorted(
            (line for line, (stmt, _) in self.order.items() if s in pathcond.ancestors(stmt)),
            key=lambda line: self.order[line][1],
        )
        for line in inside:
            if not self.pass_row(q, line):
                return out
        out.append(q)
        return out

    def call(self, q: Path, roots: set[str]) -> None:
        roots = roots & self.rel_roots
        if not roots:
            return
        marks = frozenset(f"{r}()" for r in roots)
        q.marks |= marks
        q.since = {line: m | marks for line, m in q.since.items()}
        for atom in list(q.facts):
            if any(root(c) in roots and not c.isidentifier() for c in chains(atom)):
                del q.facts[atom]
        if self.target_attr and self.target_root in roots:
            q.target_called = True

    def bind(self, q: Path, pairs: list[tuple[str, ast.expr | None]]) -> None:
        for t, vnode in pairs:
            hit = [c for c in self.relevant if affects(t, {c})]
            if not hit:
                continue
            value: Any = E.UNKNOWN
            if isinstance(vnode, ast.Constant):
                value = vnode.value
            elif vnode is not None:
                try:
                    value = E.ev(vnode, self.env(q))
                except Exception:
                    value = E.UNKNOWN
                if ROW_RENDER and value is E.UNKNOWN and t.isidentifier():
                    value = shape_value(vnode, q.state)
            for c in hit:
                if c != t:
                    q.state[c] = E.UNKNOWN
            q.state[t] = value
            for atom in list(q.facts):
                if affects(t, chains(atom)):
                    del q.facts[atom]
            q.marks |= {t}
            q.since = {line: m | {t} for line, m in q.since.items()}
            if t == self.target:
                q.target_called = False

    def pass_row(self, q: Path, line: int) -> bool:
        rank = self.order[line][1]
        if q.seen and self.order[q.seen[-1]][1] > rank:
            self.dropped["E2 order"] += 1
            return False
        q.seen += (line,)
        q.since = {**q.since, line: frozenset()}
        if self.track and self.target:
            value = q.state.get(self.target, NOT_BOUND)
            if self.target_attr and q.target_called:
                value = E.UNKNOWN
            q.at_row += ((line, value),)
        why = self.check_row(self, line, q)
        if not why and ROW_RENDER:
            why = self.render_check(q, line)
        if why:
            self.dropped[why] += 1
            return False
        return True

    def render_check(self, q: Path, line: int) -> str:
        """E5: the row as this path renders it against the record (a run) or the arm path's rendering (census)."""
        stmt = self.order[line][0]
        message = message_of(stmt)
        if message is None or not self.region or any(isinstance(a, OPAQUE) for a in _between(stmt, self.region)):
            return ""
        parts, used = render(message, q.state)
        if self.track:
            q.rendered += ((line, tuple(parts)),)
            return ""
        if line in self.row_texts:
            return "" if not used or renders_in(parts, self.row_texts[line]) else "E5 row"
        if line in self.row_renders:
            return "E5 row" if disjoint(parts, list(self.row_renders[line])) else ""
        return ""

    # -- E4

    def value_check(self, q: Path, same: Callable[[Any], bool], put: Callable[[E.Env], None]) -> str:
        """Why the path cannot have the printed target value at this row, or ''."""
        if self.target_attr and q.target_called:
            return ""
        if self.target in q.state:
            v = q.state[self.target]
            if v is E.UNKNOWN or isinstance(v, Shape):
                return ""
            return "" if same(v) else "E4 value"
        if getattr(self, "target_unbound", False):
            return "E4 unbound"
        for test, outcome, items in q.prior:
            env = self.env(Path(state=dict(items)), with_holes=False)
            for h in self.before:
                if h.value is not None and h.valid_after(q.marks):
                    env.put_printed(h.expr, h.value, h.source)
            put(env)
            try:
                v = E.evaluate(test, env)
            except Exception:
                continue
            if v is not E.UNKNOWN:
                try:
                    if bool(v) != outcome:
                        return "E4 prior"
                except TypeError:
                    continue
        return ""


def _note_cut(w: Walker, s: ast.If, env: E.Env, outcome: bool) -> None:
    from_path = all(src == "bound on the path" for src in env.used.values()) if env.used else True
    w.cuts[f"{'E3' if from_path else 'E1'} {s.lineno}"] += 1
    for k, src in env.used.items():
        if src.startswith("unknown"):
            continue
        key = env.canonical(k)
        if key in env.values:
            w.facts[k] = f"{env.values[key].value!r} ({src})"


def decide_run(w: Walker, s: ast.If, p: Path) -> list[bool]:
    env = w.env(p)
    v = w.truth(s.test, p, env)
    if v is E.UNKNOWN:
        return [True, False]
    try:
        outcome = bool(v)
    except Exception:
        return [True, False]
    _note_cut(w, s, env, outcome)
    return [outcome]


def covered(node: ast.AST, avail: set[str], state: dict[str, Any]) -> bool:
    """Whether printed holes (by name only) and the path's own constants decide *node*."""
    text = ast.unparse(node)
    if text in avail or (text in state and state[text] is not E.UNKNOWN and not isinstance(state[text], Shape)):
        return True
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return all(covered(e, avail, state) for e in node.elts)
    if isinstance(node, ast.BoolOp):
        return all(covered(v, avail, state) for v in node.values)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return covered(node.operand, avail, state)
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        return covered(node.left, avail, state) and covered(node.comparators[0], avail, state)
    if isinstance(node, ast.Call):
        if not E.PURE:
            E.PURE.update(E._pure())
        return ast.unparse(node.func) in E.PURE and all(
            covered(a, avail, state) for a in [*node.args, *(k.value for k in node.keywords)]
        )
    return False


# ---------------------------------------------------------------- against a run


@dataclass
class Proof:
    kind: str  # positive | soft | none | inconsistent | skipped
    s_star: int = 0
    reason: str = ""
    dropped: dict[str, int] = field(default_factory=dict)


RANK = {"positive": 4, "inconsistent": 3, "soft": 2, "none": 1, "skipped": 0}


def _relevant(region: list[ast.stmt], holes: list[Hole]) -> set[str]:
    out: set[str] = set()
    for s in region:
        for n in [s, *R.own_nodes(s)]:
            if isinstance(n, (ast.If, ast.While)):
                out |= chains(ast.unparse(n.test))
            elif ROW_RENDER and isinstance(n, ast.stmt) and R.is_print(n):
                out |= _rendered_names(message_of(n))
    for h in holes:
        out |= h.chains
    return out


def _straight_before(s_star: ast.stmt) -> list[ast.stmt]:
    """Prints before S* in its block with nothing that can divert in between, nearest first."""
    placed = R.block_of(s_star.parent, s_star)  # type: ignore[attr-defined]
    if placed is None:
        return []
    _, block = placed
    out = []
    for s in reversed(block[: block.index(s_star)]):
        if R.is_print(s):
            out.append(s)
        elif R.may_divert(s):
            break
    return out


def _proven_hits(sn: R.Seen, row: R.Row) -> int:
    return sum(1 for _, lit, _ in sn.hits.get(row.line, ()) if lit >= R.MIN_PROOF)


def prove(fn: R.Function, sn: R.Seen, stmt: ast.stmt) -> Proof:
    """P for the arm *stmt* of *fn* against one run (the records *sn* read)."""
    func = fn.node
    levels = chain_of_arm(func, stmt)
    if not levels:
        return Proof("none", reason="P: the arm is not under an if")
    for row in fn.rows:
        if innermost_loop(row.stmt, func) is None and _proven_hits(sn, row) > 1:
            return Proof("skipped", reason=f"P: row {row.line} printed {_proven_hits(sn, row)} times: more than one call")
    printed: dict[int, tuple[R.Row, int]] = {}
    for row in fn.rows:
        hs = sn.hits.get(row.line, [])
        if len(hs) == 1 and hs[0][0] in sn.unambiguous.get(row.line, ()) and sn.proven(row):
            printed[row.line] = (row, hs[0][0])
    best: Proof | None = None
    for s_star in levels:
        found = _prove_at(fn, sn, stmt, s_star, printed)
        if found.kind == "positive":
            return found
        # among equals, the level a printed row reached says more than one nothing reached
        if best is None or RANK[found.kind] > RANK[best.kind] or (
            RANK[found.kind] == RANK[best.kind] and best.reason.startswith("P: no printed row") and not found.reason.startswith("P: no printed row")
        ):
            best = found
    return best or Proof("none", reason="P: nothing printed reaches it")


def _prove_at(fn: R.Function, sn: R.Seen, stmt: ast.stmt, s_star: ast.If, printed: dict[int, tuple[R.Row, int]]) -> Proof:
    func = fn.node
    region = region_of(s_star)
    loops = loops_around(s_star, func)
    for row in fn.rows:
        if innermost_loop(row.stmt, func) in loops and loops and _proven_hits(sn, row) > 1:
            return Proof("skipped", s_star.lineno, f"P: row {row.line} printed {_proven_hits(sn, row)} times in the loop around {s_star.lineno}")
    order = {line: (row.stmt, idx) for line, (row, idx) in printed.items() if in_region(row.stmt, region) is not None}
    anchor_before = next((s.lineno for s in _straight_before(s_star) if s.lineno in printed), None)
    if not order and anchor_before is None:
        return Proof("none", s_star.lineno, f"P: no printed row shows {s_star.lineno} was reached")
    first = min((idx for _, idx in order.values()), default=None)
    must_before = {id(s) for s in R.must_rows(func, s_star)[0]}
    before: list[Hole] = []
    for line, (row, idx) in printed.items():
        if line in order or line >= s_star.lineno or (first is not None and idx > first):
            continue
        if R.share_loop(row.stmt, s_star, func) and id(row.stmt) not in must_before:
            continue
        for expr, value in row.capture(sn.run.records[idx]).items():
            h = Hole.make(expr, value, f"row {line}")
            if valid_before(func, row.stmt, s_star, h):
                before.append(h)
    holes_of: dict[int, list[Hole]] = {}
    for line, (rstmt, idx) in order.items():
        if any(isinstance(a, OPAQUE) for a in _between(rstmt, region)):
            continue  # a value from inside a loop or try says nothing about one pass through it
        row = printed[line][0]
        holes_of[line] = [Hole.make(e, v, f"row {line}") for e, v in row.capture(sn.run.records[idx]).items()]
    target = target_of(stmt)
    printed_target = {line: h.value for line, hs in holes_of.items() for h in hs if h.expr == target and target}

    def check_row(w: Walker, line: int, q: Path) -> str:
        if line not in printed_target:
            return ""
        text = printed_target[line].strip()
        return w.value_check(
            q,
            lambda v: str(v) == text,
            lambda env: env.put_printed(target, text, f"row {line}"),
        )

    w = Walker(
        func, stmt, target=target,
        relevant=_relevant(region, before + [h for hs in holes_of.values() for h in hs]),
        order=order, before=before, holes_of=holes_of, decide=decide_run, check_row=check_row,
    )
    w.target_unbound = bool(target) and unbound_before(func, s_star, target)  # type: ignore[attr-defined]
    # with the token: a log with no instance mark reads the message's first word as one
    w.row_texts = {
        line: f"{sn.run.records[idx].token} {sn.run.records[idx].text}" for line, (_, idx) in order.items()
    }
    try:
        paths = w.run(region)
    except TooMany as e:
        return Proof("none", s_star.lineno, f"P: more than {MAX_PATHS} paths through {s_star.lineno} ({e})")
    if HANDLER_RULE:
        paths = _handler_rule(w, fn, sn, paths)
    return _verdict(w, paths, s_star, sorted(order), anchor_before)


def _callers_in_run(fn: R.Function, sn: R.Seen) -> bool:
    """Whether every caller tied to *fn*'s run only by ``same?`` (:mod:`bamboo.querier.callers`) prints a row in *sn*'s run."""
    from . import callers

    for rel, caller, call_line in callers.unconfirmed(fn.node):
        theirs = R.seen(R.function(rel, caller.name, (call_line,)), sn.run, sn.mode)
        if not any(theirs.hits.values()):
            return False
    return True


def _handler_rule(w: Walker, fn: R.Function, sn: R.Seen, paths: list[Path]) -> list[Path]:
    """E6 against a run: drop a path that left by an exception whose handler would have printed, and did not.

    ``TB``: the handler prints a traceback through the run's logger; no frame of
    the function in the run, or frames only at other statements, and the path's
    exception did not happen.  A TB with no handler is the callers' (lib/callers,
    ``P_FOLLOW_OUT``): their traceback holds this function's frames too, and a
    caller tied only by ``same?`` must print in the run first.  ``LOG``: a row at the top of the handler whose
    absence counts (``Row.absence_counts``) is missing from the run, a run of a file it lands in (``Row.lands_in``).
    """
    raised: tuple[bool, set[int] | None] | None = None
    out = []
    for p in paths:
        cls, line, handler, raising = H.leave(fn.node, p.left) if p.left else ("NOTEXC", None, None, None)
        if cls == "TB" and raising is not None and handler is None and not _callers_in_run(fn, sn):
            out.append(p)  # a caller's traceback, but the run does not show that caller prints into it
            continue
        if cls == "TB" and raising is not None:
            if raised is None:
                raised = R.raised_in(fn, sn.run)
            named, lines = raised
            span = range(raising.lineno, (raising.end_lineno or raising.lineno) + 1)
            if not named or (lines is not None and not any(n in span for n in lines)):
                w.dropped["E6 handler"] += 1
                continue
        elif cls == "LOG" and handler is not None:
            top = [fn.row_at(st) for st in handler.body]
            if any(r is not None and r.absence_counts(sn.mode) and r.lands_in(sn.run.log.name) and not sn.present(r) for r in top):
                w.dropped["E6 handler"] += 1
                continue
        out.append(p)
    return out


def _between(node: ast.AST, region: list[ast.stmt]) -> list[ast.AST]:
    """Ancestors of *node* up to and including the region statement it is in."""
    out = []
    top = in_region(node, region)
    for a in pathcond.ancestors(node):
        out.append(a)
        if a is top:
            break
    if node is top:
        return [node]
    return out


def _verdict(w: Walker, paths: list[Path], s_star: ast.If, rows: list[int], anchor: int | None) -> Proof:
    dropped = dict(w.dropped)
    shown = f"rows {rows}" if rows else f"row {anchor} just before it"
    if not paths:
        return Proof("inconsistent", s_star.lineno, f"P: no path through {s_star.lineno} fits {shown} (dropped {dropped})", dropped)
    missing = [p for p in paths if not p.arm]
    cuts = ", ".join(sorted(w.cuts))
    facts = "; ".join(f"{k}={v}" for k, v in sorted(w.facts.items()))
    if not missing:
        how = ", ".join(f"{k} {v}" for k, v in sorted(dropped.items()))
        return Proof(
            "positive", s_star.lineno,
            f"P: every path through {s_star.lineno} that fits {shown} passes it ({how}{'; cut at ' + cuts if cuts else ''}"
            f"{'; ' + facts if facts else ''})",
            dropped,
        )
    if all(p.left.startswith("call") for p in missing):
        return Proof("soft", s_star.lineno, f"P soft: through {s_star.lineno} only a call raising misses it ({sorted({p.left for p in missing})})", dropped)
    other = sorted({p.left or "falls through" for p in missing})
    return Proof("none", s_star.lineno, f"P: {len(missing)} path(s) through {s_star.lineno} miss it ({other[:4]})", dropped)


# ---------------------------------------------------------------- the code alone


def static_holes(func: ast.AST, stmt: ast.stmt) -> list[str] | None:
    """Holes every readable shape of a print has; None when no shape can recognise its line."""
    msg = stmt.value.args[0]  # type: ignore[attr-defined]
    if not any(R.usable(s) for s in R._alternatives(msg, func, stmt)):
        return None
    shapes = [s for s in R._capture_shapes(msg, func, stmt) if R.usable(s)]
    if not shapes:
        return []
    sets = [{t for k, t in s if k == "hole" and not t.endswith("(appended)")} for s in shapes]
    return sorted(set.intersection(*sets))


def _prints_in(region: list[ast.stmt]) -> list[ast.stmt]:
    return sorted(
        (n for s in region for n in [s, *R.own_nodes(s)] if isinstance(n, ast.stmt) and R.is_print(n)),
        key=lambda n: n.lineno,
    )


@dataclass
class Census:
    kind: str  # positive | by handler | soft | none | inconsistent | no anchor | too many | not under an if
    s_star: int = 0
    arm_paths: int = 0
    proven_paths: int = 0
    #: E6: arm paths a printing handler leaves for the log to decide (they do not keep census looking outwards)
    settled_paths: int = 0
    #: the kind holds on every arm path at this S*, not only on the best one (positive and by handler)
    every: bool = False
    note: str = ""
    dropped: Counter = field(default_factory=Counter)
    cuts: Counter = field(default_factory=Counter)


CRANK = {"positive": 6, "by handler": 5.5, "soft": 5, "inconsistent": 4, "none": 3, "too many": 2.5, "no anchor": 2, "not under an if": 0}


def census(func: ast.AST, stmt: ast.stmt, record: list | None = None) -> Census:
    """If the arm ran, would the rows it prints rule out every other way through some S*?

    *record*, when given, gets one dict per S* tried (:func:`_census_at`): the
    verdict of each arm path (``reused`` when it took an earlier walk's), how it
    left, and the walks' seconds.
    """
    levels = chain_of_arm(func, stmt)
    if not levels:
        return Census("not under an if")
    best = Census("no anchor")
    for s_star in levels:
        found = _census_at(func, stmt, s_star, record)
        if CRANK[found.kind] > CRANK[best.kind] or (found.kind == best.kind and _beats(found, best)):
            best = found
        if found.kind == "positive" and found.every:
            break
    return best


def _share(c: Census) -> float:
    """The share of *c*'s arm paths its kind holds on: 1 exactly when ``every``."""
    return (c.proven_paths + c.settled_paths) / c.arm_paths if c.arm_paths else 0.0


def _beats(found: Census, best: Census) -> bool:
    """Whether *found*, an outer S* of *best*'s kind, says more of the arm than *best* (the inner one stays on a tie)."""
    if LEVEL_BY == "count":
        return found.kind == "positive" and found.proven_paths > best.proven_paths
    return found.kind in ("positive", "by handler") and _share(found) > _share(best)


def _census_at(func: ast.AST, stmt: ast.stmt, s_star: ast.If, record: list | None = None) -> Census:
    region = region_of(s_star)
    target = target_of(stmt)
    readable = {}
    for p in _prints_in(region):
        hs = static_holes(func, p)
        if hs is not None:
            readable[p.lineno] = (p, hs)
    holes_of = {
        line: [Hole.make(e, None, f"row {line}") for e in hs]
        for line, (p, hs) in readable.items()
        if not any(isinstance(a, OPAQUE) for a in _between(p, region))
    }
    before: list[Hole] = []
    for p in R.must_rows(func, s_star)[0]:
        hs = static_holes(func, p)
        for e in hs or []:
            h = Hole.make(e, None, f"row {p.lineno}")
            if valid_before(func, p, s_star, h):
                before.append(h)
    anchor = any(static_holes(func, p) is not None for p in _straight_before(s_star))
    relevant = _relevant(region, before + [h for hs in holes_of.values() for h in hs])
    # a row inside a loop or try may not print when the arm runs: only the ones a pass walks through count
    all_order = {line: (p, line) for line, (p, _) in readable.items() if line in holes_of}

    def flags_only(w: Walker, s: ast.If, p: Path) -> list[bool]:
        v = w.truth(s.test, p, w.env(p, with_holes=False))
        try:
            return [True, False] if v is E.UNKNOWN else [bool(v)]
        except Exception:
            return [True, False]

    level: dict | None = None
    if record is not None:
        level = {"s_star": s_star.lineno, "paths": []}
        record.append(level)
    first = Walker(
        func, stmt, target=target, relevant=relevant, order=all_order, before=[], holes_of={},
        decide=flags_only, check_row=lambda w, line, q: "", track=True, require_all=False,
    )
    started = time.time()
    try:
        paths = first.run(region)
    except TooMany:
        if level is not None:
            level.update(kind="too many", first_s=round(time.time() - started, 2), first_most=first.most)
        return Census("too many", s_star.lineno)
    arms = [p for p in paths if p.arm]
    if level is not None:
        level.update(first_s=round(time.time() - started, 2), first_most=first.most, arm_paths_found=len(arms))
    if not arms:
        if level is not None:
            level["kind"] = "none"
        return Census("none", s_star.lineno, note="no path through S* reaches the arm")
    if MAX_ARM_PATHS and len(arms) > MAX_ARM_PATHS:
        step = len(arms) / MAX_ARM_PATHS
        arms = [arms[int(i * step)] for i in range(MAX_ARM_PATHS)]
    result = Census("none", s_star.lineno, arm_paths=len(arms))
    kinds: Counter[str] = Counter()
    # A second walk reads the arm path's seen, at_row and rendered, and its tests only at an if whose truth it
    # cannot settle.  Another arm path with the same three whose tests agree (presence too) on every such if
    # walks the same way up to the first decide that reads them differently, and there is none.
    done: dict[tuple, list[dict]] = {}
    missing = object()
    for a in arms:
        if not a.seen and not anchor:
            kind = "no anchor"
            if HANDLER_RULE and a.left:
                cls, line = H.leave_class(func, a.left)
                if cls == "TB" and line and H.after_arm(func, stmt, line):
                    kind = "positive"  # the traceback's frame is the row it prints
                elif cls in ("TB", "LOG"):
                    kind = "by handler"
            kinds[kind] += 1
            if level is not None:
                level["paths"].append({"kind": kind, "left": a.left or "falls through"})
            continue
        tests = dict(a.tests)
        values = dict(a.at_row)
        order = {line: (all_order[line][0], i) for i, line in enumerate(a.seen)}
        same = (a.seen, repr(a.at_row), repr(a.rendered))
        earlier = next(
            (d for d in done.get(same, ()) if all(tests.get(line, missing) == d["tests"].get(line, missing) for line in d["consulted"])),
            None,
        ) if REUSE_WALKS else None
        if earlier is not None:
            kinds[earlier["kind"]] += 1
            if level is not None:
                entry = {"kind": earlier["kind"], "s": 0.0, "most": earlier["most"]}
                if earlier["missing"] is not None:
                    entry["missing"] = list(earlier["missing"])
                level["paths"].append({**entry, "reused": True})
            result.dropped.update(earlier["dropped"])
            result.cuts.update(earlier["cuts"])
            continue
        consulted: set[int] = set()

        def decide(w: Walker, s: ast.If, p: Path, tests=tests, consulted=consulted) -> list[bool]:
            v = w.truth(s.test, p, w.env(p, with_holes=False))
            try:
                if v is not E.UNKNOWN:
                    w.cuts[f"E3 {s.lineno}"] += 1
                    return [bool(v)]
            except Exception:
                pass
            consulted.add(s.lineno)
            if s.lineno in tests and covered(s.test, w.available(p), p.state):
                w.cuts[f"E1 {s.lineno}"] += 1
                return [tests[s.lineno]]
            return [True, False]

        def check_row(w: Walker, line: int, q: Path, values=values) -> str:
            if not target or line not in values or not any(h.expr == target for h in w.holes_of.get(line, ())):
                return ""
            wanted = values[line]
            if wanted is NOT_BOUND or wanted is E.UNKNOWN or isinstance(wanted, Shape):
                return ""
            return w.value_check(q, lambda got: str(got) == str(wanted), lambda env: env.put(target, wanted, f"row {line}"))

        w = Walker(
            func, stmt, target=target, relevant=relevant, order=order, before=before,
            holes_of={line: hs for line, hs in holes_of.items() if line in order}, decide=decide, check_row=check_row,
        )
        w.target_unbound = bool(target) and unbound_before(func, s_star, target)  # type: ignore[attr-defined]
        w.row_renders = dict(a.rendered)
        started = time.time()
        try:
            survivors = w.run(region)
        except TooMany:
            kinds["too many"] += 1
            if level is not None:
                level["paths"].append({"kind": "too many", "s": round(time.time() - started, 2), "most": w.most})
            done.setdefault(same, []).append(
                {"consulted": consulted, "tests": tests, "kind": "too many", "most": w.most, "missing": None, "dropped": Counter(), "cuts": []}
            )
            continue
        v = _verdict(w, survivors, s_star, list(a.seen), None)
        kind = v.kind
        if HANDLER_RULE and kind == "soft" and all(H.leave_class(func, p.left)[0] in ("TB", "LOG") for p in survivors if not p.arm):
            kind = "by handler"  # only a call raising misses the arm, and the log shows whether one did
        kinds[kind] += 1
        left_by = sorted({p.left or "falls through" for p in survivors if not p.arm})
        if level is not None:
            level["paths"].append({"kind": kind, "s": round(time.time() - started, 2), "most": w.most, "missing": list(left_by)})
        cut_kinds = [k.split(" ")[0] for k in w.cuts]
        result.dropped.update(w.dropped)
        result.cuts.update(cut_kinds)
        done.setdefault(same, []).append(
            {"consulted": consulted, "tests": tests, "kind": kind, "most": w.most, "missing": left_by, "dropped": Counter(w.dropped), "cuts": cut_kinds}
        )
    result.proven_paths = kinds["positive"]
    result.settled_paths = kinds["by handler"]
    # the kind is the best arm path's; whether the run could have taken another way through the arm, which the
    # log would not show, is every
    for kind in ("positive", "by handler", "soft", "inconsistent", "none", "no anchor", "too many"):
        if kinds[kind]:
            result.kind = kind
            break
    result.every = result.kind in ("positive", "by handler") and result.proven_paths + result.settled_paths == result.arm_paths
    result.note = ", ".join(f"{k} {n}" for k, n in kinds.most_common())
    if level is not None:
        level.update(kind=result.kind, proven=result.proven_paths, arm_paths=result.arm_paths, every=result.every)
    return result
