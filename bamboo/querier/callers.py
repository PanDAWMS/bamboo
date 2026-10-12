"""Where an exception that leaves a function (``OUT`` in :mod:`bamboo.querier.handlers`) shows: its callers' handlers.

Every call of the function's name in the corpus (tests and examples left out), the
innermost ``try`` around the call that catches everything, classed as
:mod:`bamboo.querier.handlers` classes a function's own handler, and whether that handler's
row is tied to the function's own run:

  same     -- the caller's handler prints through the function's run: an instance attribute logger of the
              same class bound with a mark (AdderGen's ``self.logger``), or a run logger the caller passes
              into the call;
  same?    -- both wrap the same ``self.X`` (``LogWrapper(self.logger, "<run>")``): the mark is self.X's when
              self.X is a wrapper itself (LogWrapper puts the inner prefix first), which the class does not say;
  token K  -- the two loggers' bindings both name the entity key K= (jediTaskID, PandaID; case ignored);
  msg K    -- the caller's run prints K= before the call, so the run is found, though not as the function's;
  none     -- neither.  "no own run" when the function prints through no run logger.

A caller whose call leaves it too (OUT) is followed to its own callers, up to
:data:`DEPTH` hops.  Calls through ``getattr`` or a plugin are not found.

:func:`out_class` is what :func:`bamboo.querier.handlers.leave` asks: ``TB`` when every
call site ends, followed up through OUT, at a handler that prints a traceback, with no conditional handler on
the way and a tie of one run (``same`` or ``same?``) -- the caller's traceback holds the function's frames and
lands in the function's run, so the run can say whether the path's exception happened, as it does for a
function's own TB handler.  Otherwise ``OUT``.  A ``same?`` tie is checked against the run before a path is
dropped (:func:`unconfirmed`).  Ties across runs (token, msg) are not followed: the calls they would settle
are ones that do not raise in practice (``setErrDiag``), and they need rules to find the other run.
Moved here from ``out_callers.py``, which uses it.
"""
from __future__ import annotations

import ast
import collections
import functools
import re

from bamboo.codemap.panda import pathcond

from . import handlers as H
from . import rows as R
from .records import CORPUS

KEY = re.compile(r"(\w+)=\{")
ENTITY = {"jeditaskid", "pandaid"}
WRAPPERS = {"LogWrapper", "MsgWrapper"}
SKIP_PARTS = {"test", "tests", "examples"}
DEPTH = 3
ONE_RUN = ("same", "same?")


@functools.cache
def corpus_files() -> tuple[str, ...]:
    out = []
    for top in ("pandaserver", "pandajedi"):
        for p in sorted((CORPUS / top).rglob("*.py")):
            rel = p.relative_to(CORPUS)
            if SKIP_PARTS & set(rel.parts) or rel.name.startswith("test_") or "Test" in rel.name:
                continue
            out.append(str(rel))
    return tuple(out)


@functools.cache
def _text(rel: str) -> str:
    return (CORPUS / rel).read_text()


@functools.cache
def calls_of(name: str) -> tuple[tuple[str, ast.Call], ...]:
    """Every call of *name* (``name(...)`` or ``x.name(...)``) in the corpus, in file order."""
    said = re.compile(rf"\b{re.escape(name)}\s*\(")
    out = []
    for rel in corpus_files():
        if not said.search(_text(rel)):
            continue
        try:
            tree = R.tree(rel)
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.Call):
                f = n.func
                if (f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)) == name:
                    out.append((rel, n))
    return tuple(out)


@functools.cache
def defs_named(name: str) -> int:
    said = re.compile(rf"\bdef\s+{re.escape(name)}\s*\(")
    n = 0
    for rel in corpus_files():
        if said.search(_text(rel)):
            try:
                n += sum(1 for d in ast.walk(R.tree(rel)) if isinstance(d, (ast.FunctionDef, ast.AsyncFunctionDef)) and d.name == name)
            except SyntaxError:
                pass
    return n


def in_corpus(func: ast.AST) -> bool:
    """Whether *func* sits in a corpus file (``rows.tree`` parsed it), not in a tree built from a string."""
    root = func
    for root in pathcond.ancestors(func):  # noqa: B007 -- the last ancestor is read below
        pass
    return id(root) in R.TREE_REL


def owner_class(func: ast.AST) -> ast.ClassDef | None:
    return next((a for a in pathcond.ancestors(func) if isinstance(a, ast.ClassDef)), None)


def wrapper(v: ast.expr) -> ast.Call | None:
    if isinstance(v, ast.Call):
        f = v.func
        name = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else "")
        if name in WRAPPERS:
            return v
    return None


def self_attr(e: ast.expr) -> str:
    return ast.unparse(e) if isinstance(e, ast.Attribute) and isinstance(e.value, ast.Name) and e.value.id == "self" else ""


@functools.cache
def class_bindings(cls: ast.ClassDef) -> dict[str, list[ast.expr]]:
    out: dict[str, list[ast.expr]] = collections.defaultdict(list)
    for n in ast.walk(cls):
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if self_attr(t):
                    out[ast.unparse(t)].append(n.value)
    return out


def attr_loggers(func: ast.AST) -> dict[str, str]:
    """:func:`bamboo.querier.handlers.attribute_loggers`, with each one's binding."""
    cls = owner_class(func)
    return {name: ast.unparse(class_bindings(cls)[name][0]) for name in H.attribute_loggers(func)}


@functools.cache
def loggers_of(func: ast.AST) -> dict[str, ast.expr | str]:
    """The run's loggers of *func*: name -> its binding (``"param"`` for a parameter)."""
    out: dict[str, ast.expr | str] = {}
    args = func.args
    params = {a.arg for a in [*args.posonlyargs, *args.args, *args.kwonlyargs]}
    for name in sorted(H.run_loggers(func)):
        if name in params:
            out[name] = "param"
            continue
        vals = [
            n.value
            for n in R.own_nodes(func)
            if isinstance(n, (ast.Assign, ast.AnnAssign)) and n.value is not None
            and any(isinstance(t, ast.Name) and t.id == name for t in H._targets(n))
        ]
        out[name] = vals[-1] if vals else ""
    cls = owner_class(func)
    for name in attr_loggers(func):
        out[name] = class_bindings(cls)[name][0]
    return out


def keys(b: ast.expr | str) -> set[str]:
    text = b if isinstance(b, str) else ast.unparse(b)
    return {k.lower() for k in KEY.findall(text)} & ENTITY


def show_binding(b: ast.expr | str) -> str:
    return b if isinstance(b, str) else ast.unparse(b)


def catching(func: ast.AST, node: ast.AST, loggers: frozenset[str]) -> list[tuple[str, ast.ExceptHandler | None]]:
    """The catching handlers from *node* outwards in *func*: ``cond`` ones, then the one that decides."""
    out = []
    chain = [node, *pathcond.ancestors(node)]
    for child, parent in zip(chain, chain[1:], strict=False):
        if parent is func:
            break
        if isinstance(parent, ast.Try) and any(child is b for b in parent.body):
            for h in parent.handlers:
                if H.catches_all(h):
                    c = H.handler_class(h, loggers)
                    if c == "RERAISE":
                        break
                    if c in ("TB", "LOG") and H.conditional(h, loggers):
                        out.append(("cond " + c, h))
                        break
                    out.append((c, h))
                    return out
    out.append(("OUT", None))
    return out


def prints_traceback(h: ast.ExceptHandler) -> bool:
    calls = [n for s in h.body for n in R.own_nodes(s) if isinstance(n, ast.Call)]
    names = {(c.func.attr if isinstance(c.func, ast.Attribute) else getattr(c.func, "id", "")) for c in calls}
    return bool(names & H.TRACEBACK_CALLS) or any(k.arg == "exc_info" for c in calls for k in c.keywords)


def handler_logger(h: ast.ExceptHandler, loggers) -> str:
    for s in h.body:
        for n in R.own_nodes(s):
            if isinstance(n, ast.Call) and H._logger_of(n) in loggers:
                return H._logger_of(n)
    return ""


def wrapped_attr(b: ast.expr | str) -> str:
    w = wrapper(b) if not isinstance(b, str) else None
    return self_attr(w.args[0]) if w is not None and w.args else ""


def tie(origin: ast.AST, call: ast.Call | None, caller: ast.AST, lg: str) -> str:
    mine, theirs = loggers_of(origin), loggers_of(caller)
    same_class = owner_class(origin) is not None and owner_class(origin) is owner_class(caller)
    b = theirs.get(lg, "")
    if same_class and lg.startswith("self.") and lg in attr_loggers(caller):
        return "same"
    if call is not None and "param" in mine.values():
        passed = {ast.unparse(a) for a in [*call.args, *(k.value for k in call.keywords)]}
        if lg in passed:
            return "same"
    if same_class and wrapped_attr(b) and wrapped_attr(b) in {wrapped_attr(x) for x in mine.values()}:
        return "same?"
    prefix = "" if mine else "no own run; "
    shared = set().union(*(keys(x) for x in mine.values())) & keys(b) if mine else set()
    if shared:
        return prefix + "token " + ",".join(sorted(shared))
    wanted = set().union(*(keys(x) for x in mine.values())) if mine else ENTITY
    line = call.lineno if call is not None else 10**9
    said = set()
    for n in R.own_nodes(caller):
        if R.is_print(n) and H._logger_of(n.value) == lg and n.lineno < line:
            said |= {k.lower() for k in re.findall(r"(\w+)=", ast.unparse(n.value))} & wanted
    if said:
        return prefix + "msg " + ",".join(sorted(said))
    return prefix + "none" + (f" (caller token {sorted(keys(b)) or show_binding(b)[:50]})" if not mine else "")


def follow(func: ast.AST, depth: int, origin: ast.AST, seen: frozenset = frozenset()) -> list[dict]:
    """Every call site of *func*: ``where``, ``recv``, the handler ``chain`` (class, prints a traceback, tie),
    the caller (``rel``, ``caller``), and ``up``, the caller's own sites when the call leaves it too."""
    out = []
    for crel, call in calls_of(func.name):
        g = pathcond.enclosing_function(call)
        where = f"{crel.rsplit('/', 1)[-1]}:{call.lineno} {g.name if g else '<module>'}"
        recv = ast.unparse(call.func.value) if isinstance(call.func, ast.Attribute) else ""
        if g is None:
            out.append({"where": where, "recv": recv, "chain": [("TOP", False, "-")], "up": [], "rel": crel, "caller": None, "call": call})
            continue
        lgs = loggers_of(g)
        chain = []
        for cls, h in catching(g, call, frozenset(lgs)):
            tb = h is not None and prints_traceback(h)
            lg = handler_logger(h, lgs) if h is not None else ""
            t = tie(origin, call if origin is func else None, g, lg) if cls.split()[-1] in ("TB", "LOG") else "-"
            chain.append((cls, tb, t))
        up = []
        if chain[-1][0] == "OUT" and depth > 1 and (crel, g.lineno) not in seen:
            up = follow(g, depth - 1, origin, seen | {(crel, g.lineno)})
        out.append({"where": where, "recv": recv, "chain": chain, "up": up, "rel": crel, "caller": g, "call": call})
    return out


def _ends(sites: list[dict]) -> list[dict] | None:
    """The deciding sites, followed up through OUT; None when one of them is not a TB tied to one run."""
    if not sites:
        return None
    out = []
    for s in sites:
        if len(s["chain"]) > 1:
            return None  # a conditional handler on the way: one of its ways prints no frame
        c, _, t = s["chain"][-1]
        if c == "OUT":
            up = _ends(s["up"])
            if up is None:
                return None
            out += up
        elif c == "TB" and t in ONE_RUN:
            out.append(s)
        else:
            return None
    return out


@functools.cache
def _decided(func: ast.AST) -> tuple[dict, ...] | None:
    if not in_corpus(func):
        return None
    ends = _ends(follow(func, DEPTH, func))
    return tuple(ends) if ends is not None else None


def out_class(func: ast.AST) -> str:
    """``TB`` when every caller prints the exception's traceback into the function's run, else ``OUT``."""
    return "TB" if _decided(func) is not None else "OUT"


def unconfirmed(func: ast.AST) -> list[tuple[str, ast.AST, int]]:
    """``(rel, caller, call line)`` of the deciding sites whose tie is ``same?``: the run must hold the caller."""
    return [(s["rel"], s["caller"], s["call"].lineno) for s in _decided(func) or () if s["chain"][-1][2] == "same?"]
