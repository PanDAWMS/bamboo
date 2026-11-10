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
* **E5** -- the row as the path would print it,
  its message rendered with the bare names the path bound itself (a value,
  or the literal parts of the text it built, the rest left open), does not
  match the record: ``tmpLog.info(tmpErrStr)`` printed ``pending ...`` rules
  out the branch that bound ``tmpErrStr = f"finishing due to ..."``.

The arm is **positive** when some path is left and every one left passes it;
**soft** when the only ones left that miss it are copies that left at a call
-- proven if no call raised, and not counted.  **inconsistent**: no path fits
the printed rows, which says a row was read from another pass or another line.

Only rows with one record in the run, read by that row alone and proving it
(``rows.Seen.proves``: at ``rows.MIN_PROOF``, or a short row's whole message
under a mark only its function writes), count; a row outside every loop printed twice means two
calls, a row of the loop around S* printed twice means two iterations, and
either stops the reading.  What this cannot see: two iterations that each
printed different rows once read as one pass (E2 then usually finds no path,
"inconsistent", but not always).  Values are printed holes only -- never the
question's DB rows, which hold what came after.

Beyond ``WIDEN_AT`` paths, paths that agree on the arm, the rows passed and
the target are joined into one that knows only what all of them knew
(:func:`widen`): that can turn a positive into nothing, never the reverse.

A call leaves the path knowing nothing of what it can change: an attribute
or an item below what it is handed, or a container it is handed (the value
the path bound there is dropped, not only the holes printed before).

Against a run (:func:`prove`) four readings go further (attr-time-shape.md
7.3):

* **(a)** an ``if`` before the arm's chain, in the same block, that binds the
  target is an S* too: the adder's ``if self.job.jobStatus not in
  ["transferring"]: self.job.jobStatus = self.job_status`` above ``if ... :
  self.job.jobStatus = "failed"``;
* **(b)** a target bound to an expression the path cannot read holds that
  expression (:class:`Deferred`): at a later row printing both, the row's
  value of it is the target's, where nothing on the path bound it or was
  handed it since (**E4 deferred**);
* **(c)** a call of the function's own class, ``self.m()``, changes only what
  m binds of ``self`` and what the calls in it are handed of it -- m as every
  class ``self`` can be resolves it (the map's ``calls.Types``: the class and
  its corpus subclasses, each along its order); any other call, only what it
  is handed below ``self`` (``self.taskBuffer.updateJobs([self.job])``:
  ``self.taskBuffer`` and ``self.job``, not ``self.job_status``).  ``self``
  itself handed on, a name resolved to more than one method or none, or an
  object of ``self`` bound to another name in m, and the call changes all of
  ``self``, as before;
* **(d)** with the judge's other verdicts (:class:`Reaching`), at a row that
  prints the observed value, a call changes the target only where the map
  reaches, from it, a writer of that value the judge has not excluded: the
  writers of the observed value are the candidates, so the value cannot
  have come from elsewhere.  At a row printing another value any call it is
  handed to may have written it, as before.

What (c) does not see: a base outside the corpus listed before the class in
a subclass's bases that defines the same name (the map's order stands it for
itself), and an object of ``self`` handed out earlier and changed through
that alias later.

:func:`census` asks the same of the code alone: if the arm ran, which rows
would print, and would they rule out everything else.
"""
from __future__ import annotations

import ast
import functools
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, cast

from bamboo.codemap.panda import pathcond  # noqa: E402
from bamboo.codemap.panda import walker as W
from bamboo.codemap.panda.walker import (  # noqa: F401 -- read as printed.<name> by the querier's modules
    MAX_PATHS,
    NAMED_DEFS,
    NOT_BOUND,
    OPAQUE,
    READERS,
    WIDEN_AT,
    Partial,
    TooMany,
    affects,
    binds,
    chain_of_arm,
    chains,
    deep_binds,
    entry_state,
    in_region,
    merge,
    region_of,
    root,
    target_of,
    unbound_before,
    widen,
)
from bamboo.codemap.panda.walker import below as _below  # noqa: F401
from bamboo.codemap.panda.walker import canon as _canon  # noqa: F401
from bamboo.codemap.panda.walker import node_chains as _chains  # noqa: F401
from bamboo.codemap.panda.walker import own_binds as _own_binds  # noqa: F401
from bamboo.codemap.panda.walker import value_key as _value_key  # noqa: F401

from . import evaluate as E
from . import handlers as H
from . import rows as R

# ---------------------------------------------------------------- reading expressions


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
    """ids of the Name nodes in a cursor execute's map argument, when the map says its statement writes no key back:
    a cursor's ``execute(sql, V)`` / ``executemany(sql, V)`` whose statement the map read and found no ``RETURNING ...
    INTO`` in (``sql.written_back_at`` answers ``()``) is not taken to change V -- PanDA's WrappedCursor touches no
    key but those binds."""
    if not (
        isinstance(call.func, ast.Attribute) and call.func.attr in ("execute", "executemany")
        and len(call.args) >= 2
    ):
        return set()
    if written_back(call) != ():
        return set()
    inside = [m for m in ast.walk(call.args[1]) if isinstance(m, ast.Name)]
    # a name the call also reads elsewhere (the statement, the receiver) is still handed there
    elsewhere = {m.id for m in ast.walk(call) if isinstance(m, ast.Name) and all(m is not i for i in inside)}
    return {id(m) for m in inside if m.id not in elsewhere}


# ---------------------------------------------------------------- (c) calls in a method, of self

FUNCS = (ast.FunctionDef, ast.AsyncFunctionDef)
#: a dunder of self (``self.__dict__.update(...)``) changes any attribute
_DUNDER = re.compile(r"self\.__\w+__")
_RUNTIME: dict[tuple[str, int], list[list[tuple[str, ast.ClassDef]]] | None] = {}


@functools.cache
def _calls_index(types: Any) -> Any:
    from bamboo.codemap.calls import Calls

    return Calls(types)


def _runtime(where: tuple[str, int]) -> list[list[tuple[str, ast.ClassDef]]] | None:
    """The orders (the map's ``calls.Types.mro``) of the classes ``self`` can be in a method of the class at *where*
    (its file and line): the class and its corpus subclasses.  None where the class is not found."""
    if where not in _RUNTIME:
        from . import asked

        types = asked._types()
        file, line = where
        tree = types.tree(file)
        cls = next((n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.lineno == line), None) if tree is not None else None
        _RUNTIME[where] = (
            None if cls is None else [types.mro(f, k) for f, k in [(file, cls), *_calls_index(types).subclasses(file, cls)]]
        )
    return _RUNTIME[where]


@functools.cache
def _methods(where: tuple[str, int]) -> frozenset[str]:
    return frozenset(m.name for order in _runtime(where) or () for _, k in order for m in k.body if isinstance(m, FUNCS))


@functools.cache
def _attributes(where: tuple[str, int]) -> frozenset[str]:
    """The names a method of a class ``self`` can be binds as ``self.<name>``."""
    return frozenset(
        t.split(".")[1]
        for order in _runtime(where) or () for _, k in order for m in k.body if isinstance(m, FUNCS)
        for st in ast.walk(m) for t, _ in binds(st) if root(t) == "self" and re.fullmatch(r"self\.\w+", t)
    )


@functools.cache
def _self_method(where: tuple[str, int], name: str) -> tuple[str, ast.AST] | None:
    """The one method ``self.<name>`` is on every class ``self`` can be; None where they differ or one has none."""
    found: dict[tuple[str, int], tuple[str, ast.AST]] = {}
    for order in _runtime(where) or ():
        hit = next(((f, m) for f, k in order for m in k.body if isinstance(m, FUNCS) and m.name == name), None)
        if hit is None:
            return None
        found[(hit[0], hit[1].lineno)] = hit
    return next(iter(found.values())) if len(found) == 1 else None


def _as_value(e: ast.AST, s: str, methods: frozenset[str]) -> bool:
    """Whether *e* holds *s* (self) itself, or a method bound to it, as a value -- a call's own arguments are read
    where the call is."""
    if isinstance(e, ast.Name):
        return e.id == s
    if isinstance(e, ast.Attribute) and isinstance(e.value, ast.Name) and e.value.id == s:
        return e.attr in methods
    if isinstance(e, ast.Call):
        return False
    return any(_as_value(c, s, methods) for c in ast.iter_child_nodes(e))


def _lets_self_out(m: ast.AST, methods: frozenset[str]) -> bool:
    """Whether method *m* hands ``self`` (or a method bound to it) to a call, binds or returns it, or closes over it."""
    if not m.args.args or any(isinstance(d, ast.Name) and d.id in ("staticmethod", "classmethod") for d in m.decorator_list):  # type: ignore[attr-defined]
        return False
    s = m.args.args[0].arg  # type: ignore[attr-defined]
    for n in ast.walk(m):
        if isinstance(n, (ast.Lambda, *FUNCS)) and n is not m:
            if any(isinstance(x, ast.Name) and x.id == s for x in ast.walk(n)):
                return True
        elif isinstance(n, ast.Call):
            if R.is_log_call(n) or (isinstance(n.func, ast.Name) and (n.func.id in READERS or n.func.id == "super")):
                continue
            if any(_as_value(a, s, methods) for a in [*n.args, *(k.value for k in n.keywords)]):
                return True
        elif isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.NamedExpr, ast.Return, ast.Yield, ast.YieldFrom)):
            if n.value is not None and _as_value(n.value, s, methods):
                return True
    return False


@functools.cache
def _self_stays(where: tuple[str, int]) -> bool:
    """Whether no method of a class ``self`` can be lets it out: then only a method of self rebinds an attribute of
    it, and another call changes only the objects below self it is handed."""
    orders = _runtime(where)
    if orders is None:
        return False
    methods = _methods(where)
    held = {(f, k.lineno): k for order in orders for f, k in order}
    return not any(_lets_self_out(m, methods) for k in held.values() for m in k.body if isinstance(m, FUNCS))


def _self_call(x: ast.AST, where: tuple[str, int]) -> bool:
    return (
        isinstance(x, ast.Call) and isinstance(x.func, ast.Attribute) and isinstance(x.func.value, ast.Name)
        and x.func.value.id == "self" and x.func.attr in _methods(where)
    )


def _effects(where: tuple[str, int], node: ast.AST, seen: frozenset) -> tuple[frozenset[str], frozenset[str]] | None:
    """``(chains of self bound, chains below self handed)`` by the calls in *node*; None where they can change any of
    self."""
    bound: set[str] = set()
    handed: set[str] = set()
    for n in R.own_nodes(node):
        if not isinstance(n, ast.Call) or R.is_log_call(n) or (isinstance(n.func, ast.Name) and n.func.id in READERS):
            continue
        f = n.func
        if any(isinstance(x, ast.Name) and x.id == "super" for x in ast.walk(f)):
            return None  # a base's method, run on self
        given: list[ast.AST] = [*n.args, *(k.value for k in n.keywords)]
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "self":
            if f.attr in _methods(where):
                method = _self_method(where, f.attr)
                if method is None:
                    return None
                if (key := (method[0], method[1].lineno)) not in seen:
                    inner = _method_effects(where, method[1], seen | {key})
                    if inner is None:
                        return None
                    bound |= inner[0]
                    handed |= inner[1]
            elif f.attr in _attributes(where):
                given.append(f)  # an attribute of self holding a callable: the object called
            else:
                return None  # neither: a method of a base outside the corpus, say
        else:
            given.append(f.value if isinstance(f, ast.Attribute) else f)
        # what a method of self returns may be any object of self, handed on here (another call's result is reached
        # from its receiver or arguments, handed already)
        if any(_self_call(x, where) for a in given for x in ast.walk(a)):
            return None
        handed |= {c for a in given for c in _chains(a) if root(c) == "self"}
    if any(_DUNDER.match(c) for c in bound | handed):
        return None
    return frozenset(bound), frozenset(handed)


def _method_effects(where: tuple[str, int], m: ast.AST, seen: frozenset) -> tuple[frozenset[str], frozenset[str]] | None:
    """What calling method *m* of self does to self: what it binds of self, the objects below self it binds another
    name to (an alias it may change them through, counted as handed), and its own calls' effects."""
    bound: set[str] = set()
    handed: set[str] = set()
    for st in ast.walk(m):
        for t, _ in binds(st):
            if root(t) == "self":
                bound.add(t)
        value = st.iter if isinstance(st, (ast.For, ast.AsyncFor)) else getattr(st, "value", None)
        if isinstance(st, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.NamedExpr, ast.For, ast.AsyncFor)) and value is not None:
            handed |= {c for c in _chains(value) if root(c) == "self"}
        if isinstance(st, (ast.With, ast.AsyncWith)):
            handed |= {c for i in st.items if i.optional_vars is not None for c in _chains(i.context_expr) if root(c) == "self"}
    inner = _effects(where, m, seen)
    if inner is None:
        return None
    return frozenset(bound | inner[0]), frozenset(handed | inner[1])


@functools.lru_cache(maxsize=None)
def self_effects(func: ast.AST, node: ast.AST) -> tuple[frozenset[str], frozenset[str]] | None:
    """(c): what the calls in *node*, in method *func*, can change of ``self`` -- ``(chains of self bound, chains below
    self handed)`` -- or None where they can change any of it (``self`` let out by a method of a class it can be, a
    name resolved to more than one method or none)."""
    cls = next((a for a in pathcond.ancestors(func) if isinstance(a, ast.ClassDef)), None)
    module = next((a for a in pathcond.ancestors(func) if isinstance(a, ast.Module)), None)
    rel = R.TREE_REL.get(id(module)) if module is not None else None
    args = getattr(func, "args", None)
    if cls is None or rel is None or args is None or not args.args or args.args[0].arg != "self":
        return None
    where = (rel, cls.lineno)
    if not _self_stays(where):
        return None
    return _effects(where, node, frozenset())


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


# ---------------------------------------------------------------- rendering a row (E5)

Part = tuple[str, str]  # ("lit", text) or ("any", "")
ANY: Part = ("any", "")


@dataclass(frozen=True)
class Shape(Partial):
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
                handed = m[:-2]
                # a name handed to a call; or (c) a chain below self, whose object the call can change
                if handed.isidentifier():
                    if self.sensitive and handed in self.roots:
                        return False
                elif affects(handed, self.chains):
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


@dataclass(frozen=True)
class Deferred(Partial):
    """(b): the target bound to *expr*, which the path cannot read, at line ``-key`` -- ``key`` is the binding's
    entry in :attr:`Path.since`, beside the rows' lines."""

    expr: str
    key: int


@dataclass(frozen=True)
class Reaching:
    """(d): the judge's other verdicts -- the lines of the function's calls from which the map reaches a writer of
    *value* (the observed one) the judge has not excluded."""

    value: str
    lines: frozenset[int]


@functools.cache
def _map_calls() -> Any:
    from bamboo.codemap.calls import Calls, Types

    return Calls(Types(R.ROOTS, {}))


def _file_of(func: ast.AST) -> str:
    module = next((a for a in pathcond.ancestors(func) if isinstance(a, ast.Module)), None)
    return R.TREE_REL.get(id(module), "") if module is not None else ""


@dataclass
class Path(W.Path):
    #: (row line, its message as this path renders it) -- E5, census only
    rendered: tuple = ()
    #: (d) a call the target was handed to since it was bound reaches a writer of the observed value still standing
    target_reached: bool = False

    def key_parts(self) -> tuple:
        return super().key_parts() + (self.rendered, self.target_reached)

    def join(self, q: "Path") -> None:  # type: ignore[override]
        super().join(q)
        self.target_reached = self.target_reached or q.target_reached
        self.rendered = tuple(x for x in self.rendered if x in q.rendered)


class Walker(W.Walker):
    """Every path through a region, with the printed rows and their holes as the evidence."""

    path_type = Path

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
        against_run: bool = False,
        reaching: Reaching | None = None,
    ) -> None:
        super().__init__(
            func, arm, target=target, relevant=relevant, order=order, decide=decide, check_row=check_row,
            corpus=_map_calls(), file=_file_of(func), track=track, require_all=require_all,
        )
        #: (b) and (c): read against a run only; the census reads the code as before
        self.against_run = against_run
        self.reaching = reaching
        self.before = before
        self.holes_of = holes_of
        #: E5: row line -> the record's text (against a run) / the arm path's rendering (census)
        self.row_texts: dict[int, str] = {}
        self.row_renders: dict[int, tuple] = {}

    def run(self, region: list[ast.stmt]) -> list[Path]:  # type: ignore[override]
        return cast("list[Path]", super().run(region))  # each path is a path_type

    def env(self, p: W.Path, with_holes: bool = True) -> E.Env:
        return cast(E.Env, super().env(p, with_holes))  # built by make_env

    # -- hooks

    def make_env(self) -> E.Env:
        return E.Env()

    def put_holes(self, env: E.Env, p: Path) -> None:  # type: ignore[override]
        for h in self.before:
            if h.value is not None and h.valid_after(p.marks):
                env.put_printed(h.expr, h.value, h.source)
        for line in p.seen:
            for h in self.holes_of.get(line, ()):
                if h.value is not None and h.valid_after(p.since[line]):
                    env.put_printed(h.expr, h.value, h.source)

    def reaches(self, node: ast.AST) -> bool:
        return self.reaching is None or any(
            n.lineno in self.reaching.lines for n in R.own_nodes(node) if isinstance(n, ast.Call)
        )

    def handed_target(self, q: Path, node: ast.AST) -> None:  # type: ignore[override]
        q.target_reached = q.target_reached or self.reaches(node)

    def value_of(self, q: Path, t: str, vnode: ast.expr | None) -> Any:  # type: ignore[override]
        value = super().value_of(q, t, vnode)
        if value is E.UNKNOWN and vnode is not None and t.isidentifier():
            value = shape_value(vnode, q.state)
        if (
            self.against_run and value is E.UNKNOWN and t == self.target and isinstance(vnode, (ast.Name, ast.Attribute))
            and ast.unparse(vnode) in self.relevant
        ):
            # (b): a later row may print what it holds
            value = Deferred(ast.unparse(vnode), -vnode.lineno)
        return value

    def bound(self, q: Path, t: str, value: Any) -> None:  # type: ignore[override]
        if isinstance(value, Deferred):
            q.since[value.key] = frozenset()
        if t == self.target:
            q.target_reached = False

    def appended(self, s: ast.stmt, q: Path) -> Any:  # type: ignore[override]
        if (
            isinstance(s, ast.AugAssign) and isinstance(s.op, ast.Add) and isinstance(s.target, ast.Name)
            and isinstance(q.state.get(s.target.id), (str, Shape))
        ):
            return shape_value(s.value, q.state, before=q.state[s.target.id])
        return None

    def row_rule(self, q: Path, line: int) -> str:  # type: ignore[override]
        return self.render_check(q, line)

    # -- the querier's

    def available(self, p: Path) -> set[str]:
        out = {h.expr for h in self.before if h.valid_after(p.marks)}
        for line in p.seen:
            out |= {h.expr for h in self.holes_of.get(line, ()) if h.valid_after(p.since[line])}
        return out

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

    def value_check(
        self, q: Path, same: Callable[[Any], bool], put: Callable[[E.Env], None], called: bool | None = None
    ) -> str:
        """Why the path cannot have the printed target value at this row, or ''.  *called*: whether a call since the
        target was bound may have changed it (``target_called`` unless the caller says)."""
        if self.target_attr and (q.target_called if called is None else called):
            return ""
        if self.target in q.state:
            v = q.state[self.target]
            if v is E.UNKNOWN or isinstance(v, (Shape, Deferred)):
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
    """Whether printed holes (by name only) and the path's own constants decide *node*: the map's
    ``evaluate.decidable``, so a test said covered is one the evaluator decides once the holes' values come."""
    env = E.Env()
    for text, value in state.items():
        if value is not E.UNKNOWN and not isinstance(value, Shape):
            env.put(text, value, "bound on the path")
    return E.decidable(node, avail, env)


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
            elif isinstance(n, ast.stmt) and R.is_print(n):
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
    return sum(1 for i, lit, _ in sn.hits.get(row.line, ()) if sn.proves(row, i, lit))


def binding_ifs(top: ast.If, target: str) -> list[ast.If]:
    """(a): the ``if``s before *top* in its block that bind *target*, nearest first."""
    placed = R.block_of(top.parent, top)  # type: ignore[attr-defined]
    if placed is None or not target:
        return []
    _, block = placed
    return [s for s in reversed(block[: block.index(top)]) if isinstance(s, ast.If) and any(affects(t, {target}) for t in deep_binds(s))]


def prove(fn: R.Function, sn: R.Seen, stmt: ast.stmt, reaching: Reaching | None = None) -> Proof:
    """P for the arm *stmt* of *fn* against one run (the records *sn* read); *reaching*, the judge's other verdicts
    (d), where it has them."""
    func = fn.node
    levels = chain_of_arm(func, stmt)
    if not levels:
        return Proof("none", reason="P: the arm is not under an if")
    above = binding_ifs(levels[-1], target_of(stmt))
    levels += above
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
        found = _prove_at(fn, sn, stmt, s_star, printed, reaching, binding=s_star in above)
        if found.kind == "positive":
            return found
        # among equals, the level a printed row reached says more than one nothing reached
        if best is None or RANK[found.kind] > RANK[best.kind] or (
            RANK[found.kind] == RANK[best.kind] and best.reason.startswith("P: no printed row") and not found.reason.startswith("P: no printed row")
        ):
            best = found
    return best or Proof("none", reason="P: nothing printed reaches it")


def _prove_at(
    fn: R.Function, sn: R.Seen, stmt: ast.stmt, s_star: ast.If, printed: dict[int, tuple[R.Row, int]], reaching: Reaching | None,
    binding: bool = False,
) -> Proof:
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
    if binding and not printed_target:
        # (a): the level is there for E4 to read the target at a printed row; with no such row it is not walked (a long
        # function's region costs minutes: archiveJob's 1301 for its 1813)
        return Proof("none", s_star.lineno, f"P: no printed row in {s_star.lineno}'s region shows {target}")

    def check_row(w: Walker, line: int, q: Path) -> str:
        if line not in printed_target:
            return ""
        text = printed_target[line].strip()
        # (d): only a writer of the observed value can have made the target print it
        called = q.target_reached if reaching is not None and text == reaching.value.strip() else q.target_called
        held = q.state.get(target)
        if isinstance(held, Deferred) and not (w.target_attr and called):
            got = next(
                (h.value.strip() for h in holes_of.get(line, ())
                 if h.expr == held.expr and h.value is not None and h.valid_after(q.since.get(held.key, frozenset()))),
                None,
            )
            if got is not None:
                return "" if got == text else "E4 deferred"
        return w.value_check(
            q,
            lambda v: str(v) == text,
            lambda env: env.put_printed(target, text, f"row {line}"),
            called,
        )

    w = Walker(
        func, stmt, target=target,
        relevant=_relevant(region, before + [h for hs in holes_of.values() for h in hs]),
        order=order, before=before, holes_of=holes_of, decide=decide_run, check_row=check_row,
        against_run=True, reaching=reaching,
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
    exception did not happen.  A TB with no handler is the callers' (lib/callers):
    their traceback holds this function's frames too, and a
    caller tied only by ``same?`` must print in the run first.  ``LOG``: a row at the top of the handler whose
    absence counts (``Row.absence_counts``) is missing from the run, a run of a file it lands in (``Row.lands_in``)
    asked for its lines (``Run.holds``).
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
            if any(r is not None and r.absence_counts(sn.mode) and r.lands_in(sn.run.log.name) and sn.run.holds(r.func, r.level)
                   and not sn.present(r) for r in top):
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
            if a.left:
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
        )
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
        if kind == "soft" and all(H.leave_class(func, p.left)[0] in ("TB", "LOG") for p in survivors if not p.arm):
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
