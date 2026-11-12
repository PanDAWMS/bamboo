"""Which of a field's bindings a read took, and which of its changes ran between, on the run's path (S4b).

A field of ``self``, or of an object a corpus class builds, is bound in its classes' methods and changed in place in
them, and the map's ``FieldFrom`` says them in no order.  A read of the field took the last binding before it on the
path its run went, with the changes that ran between that binding and the read (out/gm/s4-design.md rule 4, with
out/gm/s4b-order.md's (a)-(d)):

* **(a) The callers' chain.**  The read's function is entered from the one call its ``EnteredFrom`` places, where no
  way in is open; a function the map makes no ``EnteredFrom`` for (it holds no arm), from the one edge of the reach
  record into it.  That caller is entered from its own in the same way, and so on.  In each, what ran before the call
  that went on is before the read.  A function entered from several calls stops the chain: a binding of one of
  those callers may be the last, so nothing is off the path and nothing is chosen past it.
* **(b) Callees at their call.**  A binding or change in a function a call goes on into (the map's ``ReachNode.calls``,
  a thread's start aside) is placed at that call.
* **(c) ``__init__`` first.**  Where the chain stops before the call that built the object, the field as
  ``__init__`` left it is what the first function of the chain was entered with.
* **(d) Off the run's path.**  A binding or change none of these places is not taken, and is named.  The premise is
  that only the run's path changes the object read.  Where a call on the path may go anywhere (``ReachNode.holes``),
  nothing is off the path.
* **(e) A binding's parameter by the call the path entered its function by** (out/gm/s4b-site.md).  Where (a) or (b)
  places a binding in a function the run's path entered by one call, or by each of a few, the value the binding took
  of a parameter came in by those calls: the parameter is read up only the ways the map's ``ParameterFrom`` says come
  in by them (:attr:`Choice.entered`).  Where (c) places it, or the path is open, every caller's way is read.
  **(e2)** An ``__init__``'s binding of a field of ``self`` came in by the call that built the object read, open path or
  not: up the chain's calls, the one handing the callee's ``self`` a local of the caller (the map's
  ``ReachNode.call_bindings``), where every binding of that local reaching the call is what the class's ``__init__``
  call returns (``ReachNode.bindings``, how ``result``).  A call cannot bind its caller's local again.  Where the
  ``self`` comes from anything else, (e) is read.

In one function, which placed binding reaches the read is the map's flow's (``reaching``).  A binding runs over the
ones before it.  A call whose callee binds on some ways only, or only changes the value, runs over none (a *reread*).
What the run printed narrows it, as H2 reads a call: a statement no call can run with a printed line did not run
(``flow.together``), and a handler whose own prints the run would show and does not was not entered (J's
``_not_entered``).  A run of two calls of the function is not narrowed.  A change placed at the first call of its
*via* is read further in the same way: each later call of its *via*, and its own statement, by the printing of the
function holding it; where one is shown not run, the change did not run (out/gm/gm5-design.md G1).

A change counts only where the read reads what it changes: an attribute the read takes off the field, or the whole
value (a method called on it, the value handed on).  A change by a key it does not say (``*``) always counts.
"""

from __future__ import annotations

import ast
import functools
from dataclasses import dataclass
from typing import Any

from bamboo.codemap import models as M
from bamboo.codemap.panda import flow, pathcond
from bamboo.codemap.panda import reaching as RE

from . import mapdump
from . import rows as R
from . import sources as SR
from . import which_ran as WR

#: the value the field held as a function was entered (an index into the items, beside theirs)
ENTRY = -1
#: how many callers and callees deep the path is read
DEPTH = 8
#: why a binding or change is not taken
REBOUND = "rebound-before-the-read"
#: a change placed on the path, not between the binding taken and the read
NOT_BETWEEN = "not-between"
OFF_PATH = "off-the-run-s-path"
NOT_RUN = "not-run"
#: a change of what the read does not take off the value (S4b's rule 4; a method the test calls, by what the map says
#: it reads, S4c's scope 2)
NOT_READ = "not-read"


@dataclass(frozen=True)
class Choice:
    """The bindings a read of a field may have taken, the changes that may have run between them and the read, and
    the others with why they are not (``file:line``, why).  *entry* says where the value as a function was entered is
    left, read no further: a function whose callers the map does not narrow to one, with no ``__init__`` binding."""

    bindings: tuple[Any, ...]
    changes: tuple[Any, ...]
    dropped: tuple[tuple[str, str], ...]
    entry: str = ""
    #: (e): a binding taken (its ``file:line``) with the calls the run's path entered its function by (``file:line``)
    entered: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def entered_by(self, at: str) -> tuple[str, ...]:
        """The calls the run's path entered the function of the binding at *at* by; empty where any of its callers."""
        return next((sites for b, sites in self.entered if b == at), ())


@functools.cache
def _reach_index(fragment_id: int) -> dict[str, Any]:
    fragment = mapdump.fragment()
    return {n.owner: n for n in getattr(fragment, "reach_graph", None) or []}


def _reach() -> dict[str, Any]:
    fragment = mapdump.fragment()
    return _reach_index(id(fragment)) if fragment is not None else {}


@functools.cache
def _closure_in(fragment_id: int, owner: str) -> frozenset[str]:
    reach = _reach_index(fragment_id)
    seen: set[str] = set()
    todo = [owner]
    while todo:
        for c in getattr(reach.get(todo.pop()), "calls", None) or []:
            if not c.thread and c.to not in seen:
                seen.add(c.to)
                todo.append(c.to)
    return frozenset(seen)


def _closure(owner: str) -> frozenset[str]:
    """The functions a call of *owner* goes on into, any number of calls deep (threads aside)."""
    fragment = mapdump.fragment()
    return _closure_in(id(fragment), owner) if fragment is not None else frozenset()


@functools.cache
def _callers_in(fragment_id: int) -> dict[str, list[tuple[str, int]]]:
    out: dict[str, list[tuple[str, int]]] = {}
    for n in _reach_index(fragment_id).values():
        for c in getattr(n, "calls", None) or []:
            out.setdefault(c.to, []).extend((n.owner, ln) for ln in c.lines or [c.line])
    return out


def _caller(node: ast.AST, owner: str) -> tuple[str, int] | None | str:
    """The one call *node* (*owner*, ``file::Class.method``) is entered from, as ``(rel, line)``: its ``EnteredFrom``'s
    one site with no way in open, or, with no such record, the one edge of the map's reach record into it (a thread's
    start included: what ran before the start ran before the thread).  None where nothing in the corpus enters it (a
    daemon's or an API's entry); a str saying why where it is entered from more than one, or a way in is open."""
    rel = owner.partition("::")[0]
    rec = SR.entered(f"{rel}::{getattr(node, 'name', '')}", getattr(node, "lineno", 0))
    if rec is not None:
        sites = [(s.owner.partition("::")[0], s.line) for s in getattr(rec, "sites", None) or []]
        if getattr(rec, "open", None):
            return "a way in is open"
    else:
        fragment = mapdump.fragment()
        sites = [(o.partition("::")[0], ln) for o, ln in (_callers_in(id(fragment)).get(owner, []) if fragment else [])]
    sites = sorted(set(sites))
    if not sites:
        return None
    return sites[0] if len(sites) == 1 else f"{len(sites)} calls enter it"


def _def(rel: str, line: int) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef, str] | None:
    """The innermost def of *rel* holding *line*, with its owner as ``ReachNode`` spells it (``file::Class.method``)."""
    try:
        node = R.def_holding(rel, line)
    except (OSError, SyntaxError, KeyError):
        return None
    return (node, f"{rel}::{pathcond.qualified_name(node)}") if node is not None else None


def _node(owner: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """The def a ``ReachNode`` owner names (``file::Class.method``; a ``#n`` repeat is not told apart)."""
    rel, _, name = owner.partition("::")
    name = name.split("#")[0]
    try:
        found = [f for f in R.functions(rel, name.rpartition(".")[2]) if pathcond.qualified_name(f) == name]
    except (OSError, SyntaxError, KeyError):
        return None
    return found[0] if len(found) == 1 else None


def _handed_self(caller: str, line: int, callee: str) -> str | None:
    """The path the call at *line* of *caller* hands *callee* as ``self`` (the map's ``call_bindings``); None where it
    says none, or several."""
    paths = {b.path for b in getattr(_reach().get(caller), "call_bindings", None) or []
             if b.line == line and b.to == callee and b.parameter == "self"}
    return paths.pop() if len(paths) == 1 else None


def _built_at(node: ast.FunctionDef | ast.AsyncFunctionDef, caller: str, line: int, local: str) -> dict[str, set[str]]:
    """(e2): the calls that built the object *caller*'s *local* holds at its call at *line*, by the ``__init__`` each
    went into (``file:line``): every binding of *local* reaching the call (the map's ``ReachNode.bindings``, the flow's
    ``reaching``) is all of what a call into an ``__init__`` returns.  Empty where one is anything else, or the
    function's entry (a parameter) reaches it."""
    reach = _reach().get(caller)
    target = R.statement_at(node, line)
    if reach is None or target is None:
        return {}
    binds = [(b, st) for b in getattr(reach, "bindings", None) or []
             if b.name == local and (st := R.statement_at(node, b.line)) is not None and st is not target]
    entry = local in {a.arg for a in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]}
    sites: list[ast.AST] = ([node.args] if entry else []) + [st for _, st in binds]
    try:
        got = RE.reaching(target, sites)
    except (ValueError, LookupError, RecursionError):
        return {}
    if not got or any(g is node.args for g in got):
        return {}
    out: dict[str, set[str]] = {}
    rel = caller.partition("::")[0]
    for b, st in binds:
        if not any(g is st for g in got):
            continue
        if b.how != M.BIND_RESULT or b.path or not b.to or any(t.rpartition(".")[2] != "__init__" for t in b.to):
            return {}
        for t in b.to:
            lines = {ln for c in getattr(reach, "calls", None) or [] if c.to == t for ln in c.lines or [c.line]
                     if R.statement_at(node, ln) is st}
            if not lines:
                return {}
            out.setdefault(t, set()).update(f"{rel}:{ln}" for ln in lines)
    return out


def _place(item: Any) -> tuple[str, int]:
    """Where *item* runs: a binding's statement; a change's first call from the method of the classes it is made in
    (*via*), else its own statement."""
    at = item.via[0] if getattr(item, "via", None) else item.at
    rel, _, line = at.rpartition(":")
    return rel, int(line)


def _deeper(item: Any) -> list[tuple[str, int]]:
    """Where a change placed at its first call runs further in: each later call of its *via*, then its own statement
    (out/gm/gm5-design.md G1).  Empty where it is placed at its own statement."""
    via = list(getattr(item, "via", None) or [])
    out = []
    for at in [*via[1:], item.at] if via else []:
        rel, _, line = at.rpartition(":")
        out.append((rel, int(line)))
    return out


class _Path:
    """The items (a FieldFrom's bindings and the changes that count) placed on a run's path, function by function."""

    def __init__(self, items: list[tuple[Any, str]], run: Any, mode: str) -> None:
        self.items = items
        self.run, self.mode = run, mode
        self.where = [_place(it) for it, _ in items]
        self.deeper = [_deeper(it) if kind == "change" else [] for it, kind in items]
        self.owners = [d[1] if (d := _def(rel, line)) is not None else "" for rel, line in self.where]
        #: each function's printed statements and the handlers it may have entered, read once
        self.narrowed: dict[str, tuple[list[ast.stmt], list[ast.ExceptHandler] | None]] = {}
        self.placed: set[int] = set()
        #: the items whose statement the run shows did not run
        self.not_run: set[int] = set()
        #: a call on the read path may go anywhere, or the callers' chain stopped at several: nothing is off the path
        self.open = False
        self.memo: dict[tuple[str, int | None], frozenset[int]] = {}
        #: the calls the path entered each function by (``file:line``); None where it is not told which (e)
        self.entered: dict[str, set[str] | None] = {}
        #: the calls that built the object read, by the ``__init__`` each went into (e2)
        self.built: dict[str, set[str]] = {}

    def enter(self, owner: str, sites: set[str] | None) -> None:
        """The path entered *owner* by *sites* (None: by a call it does not tell)."""
        if sites is None or self.entered.get(owner, set()) is None:
            self.entered[owner] = None
        else:
            self.entered.setdefault(owner, set()).update(sites)  # type: ignore[union-attr]

    def _narrowing(self, node: ast.FunctionDef | ast.AsyncFunctionDef, owner: str
                   ) -> tuple[list[ast.stmt], list[ast.ExceptHandler] | None]:
        """The statements *node*'s rows printed in the run, and the handlers it may have entered (None: any)."""
        if self.run is None:
            return [], None
        rel = owner.partition("::")[0]
        try:
            fn = R.function(rel, node.name, within=[node.lineno])
            sn = R.seen(fn, self.run, self.mode)
        except (OSError, SyntaxError, LookupError, ValueError):
            return [], None
        if not sn.single_call():
            return [], None
        printed = [s for ln, hits in sn.hits.items() if hits and (s := R.statement_at(node, ln)) is not None]
        marks = WR._handler_marks(owner)
        handlers = [h for t in R.own_nodes(node) if isinstance(t, ast.Try) for h in t.handlers]
        return printed, [h for h in handlers if not WR._not_entered(sn, marks.get(h.lineno))]

    def _narrowed(self, node: ast.FunctionDef | ast.AsyncFunctionDef, owner: str
                  ) -> tuple[list[ast.stmt], list[ast.ExceptHandler] | None]:
        """:meth:`_narrowing`, read once a function."""
        if owner not in self.narrowed:
            self.narrowed[owner] = self._narrowing(node, owner)
        return self.narrowed[owner]

    @staticmethod
    def _not_run(stmt: ast.stmt, printed: list[ast.stmt], raised: list[ast.ExceptHandler] | None) -> bool:
        try:
            return any(p is not stmt and not flow.together(p, stmt, raised=raised) for p in printed)
        except (ValueError, LookupError):
            return False

    def _not_run_deeper(self, i: int) -> bool:
        """Whether the run shows that a call further in on item *i*'s way, or its own statement, did not run (G1):
        each read by its function's printing, as a function the path goes on into is."""
        for rel, line in self.deeper[i]:
            d = _def(rel, line)
            stmt = R.statement_at(d[0], line) if d is not None else None
            if d is not None and stmt is not None and self._not_run(stmt, *self._narrowed(*d)):
                return True
        return False

    def values(self, node: ast.FunctionDef | ast.AsyncFunctionDef, owner: str, target: ast.stmt | None,
               depth: int = 0) -> frozenset[int]:
        """What the field may hold at *target* of *node* (None: as the function returns): the items whose effect is
        live there, and :data:`ENTRY` where the value it was entered with is."""
        key = (owner, target.lineno if target is not None else None)
        if key in self.memo:
            return self.memo[key]
        self.memo[key] = frozenset({ENTRY})
        rel = owner.partition("::")[0]
        reach = _reach().get(owner)
        if reach is None or getattr(reach, "holes", None):
            self.open = True
        own: dict[int, tuple[ast.stmt, list[int]]] = {}
        for i, (where, item_owner) in enumerate(zip(self.where, self.owners, strict=True)):
            if where[0] == rel and item_owner == owner and (stmt := R.statement_at(node, where[1])) is not None:
                own.setdefault(id(stmt), (stmt, []))[1].append(i)
                self.placed.add(i)
        held = {o for o in self.owners if o}
        # each statement's calls on into a function holding an item, with the lines it calls each at
        calls: dict[int, tuple[ast.stmt, dict[str, set[int]]]] = {}
        for c in getattr(reach, "calls", None) or []:
            if c.thread or not ({c.to} | _closure(c.to)) & held:
                continue
            for ln in c.lines or [c.line]:
                if (stmt := R.statement_at(node, ln)) is not None:
                    calls.setdefault(id(stmt), (stmt, {}))[1].setdefault(c.to, set()).add(ln)
        printed, raised = self._narrowed(node, owner)
        # sites bind on every way through them; rereads may change or bind, and run over nothing
        sites: list[ast.AST] = []
        rereads: list[ast.AST] = []
        effect: dict[int, frozenset[int]] = {}
        for sid in sorted({*own, *calls}, key=lambda s: (own.get(s) or calls[s])[0].lineno):
            stmt = own[sid][0] if sid in own else calls[sid][0]
            mine = own[sid][1] if sid in own else []
            if self._not_run(stmt, printed, raised):
                self.not_run.update(mine)
                continue
            deep = {i for i in mine if self._not_run_deeper(i)}
            self.not_run |= deep
            mine = [i for i in mine if i not in deep]
            # what the statement's calls leave: ENTRY where a way through them binds nothing
            inner: set[int] = {ENTRY}
            if sid in calls and depth < DEPTH:
                inner = set()
                for to, lines in calls[sid][1].items():
                    self.enter(to, {f"{rel}:{ln}" for ln in lines})
                    callee = _node(to)
                    if callee is None:
                        self.open = True
                    inner |= self.values(callee, to, None, depth + 1) if callee is not None else {ENTRY}
            got = frozenset(mine) | (inner - {ENTRY})
            if not got:
                continue
            effect[sid] = got
            binds = any(self.items[i][1] == "binding" for i in mine) or (
                ENTRY not in inner and any(self.items[i][1] == "binding" for i in inner))
            (sites if binds else rereads).append(stmt)
        returns = [n for n in R.own_nodes(node) if isinstance(n, ast.Return)] if target is None else []

        def live(x: ast.AST) -> bool:
            """Whether *x*'s effect can be what *target* (or the return) reads: a way from it with no site after."""
            killers = [s for s in sites if s is not x]
            try:
                if target is not None:
                    return bool(RE.reaching(target, [x], killers, raised=raised))
                return bool(RE.reaching_end(node, [x], killers, raised=raised)) or any(
                    RE.reaching(r, [x], killers, raised=raised) for r in returns)
            except (ValueError, LookupError, RecursionError):
                return True

        reaching = [x for x in [node.args, *sites, *rereads] if live(x)]
        out: set[int] = set()
        for g in reaching:
            out |= {ENTRY} if g is node.args else effect.get(id(g), frozenset())
        self.memo[key] = frozenset(out)
        return self.memo[key]


def choose(ff: Any, rel: str, line: int, run: Any, mode: str, keys: frozenset[str] | None, of_self: bool = False) -> Choice:
    """Of *ff*'s bindings and the changes that count for a read taking *keys*, those the read at *line* of *rel* may
    have taken on *run*'s path (module docstring).  *of_self*: the field read is one of ``self`` of the read's function
    (e2 follows that object up the chain)."""
    items = [(b, "binding") for b in ff.bindings] + [(c, "change") for c in ff.changes if SR.change_counts(c, keys)]
    unread = tuple((c.at, NOT_READ) for c in ff.changes if not SR.change_counts(c, keys))
    path = _Path(items, run, mode)
    d = _def(rel, line)
    target = R.statement_at(d[0], line) if d is not None else None
    if d is None or target is None:
        return Choice(tuple(ff.bindings), tuple(it for it, k in items if k == "change"), unread, entry="no-def")
    node, owner = d
    held = path.values(node, owner, target)
    entry = ""
    seen = {owner}
    # (e2) the chain's function whose self is the object read
    same = of_self
    for _ in range(DEPTH):
        site = _caller(node, owner)
        path.enter(owner, {f"{site[0]}:{site[1]}"} if isinstance(site, tuple) else None)
        if ENTRY not in held:
            break
        if isinstance(site, str):
            # a caller of several may have bound it before its call: not off the path, not chosen
            entry, path.open = f"{node.name}: {site}", True
            break
        up = _def(*site) if site is not None else None
        at = R.statement_at(up[0], site[1]) if up is not None and site is not None else None
        if up is None or at is None or up[1] in seen:
            break
        if same and site is not None:
            handed = _handed_self(up[1], site[1], owner)
            same = handed == "self"
            if handed is not None and handed.isidentifier() and handed != "self":
                path.built = _built_at(up[0], up[1], site[1], handed)
        node, owner = up
        seen.add(owner)
        held = (held - {ENTRY}) | path.values(node, owner, at)
    if ENTRY in held:
        inits = {path.owners[i] for i, (_, k) in enumerate(items)
                 if k == "binding" and path.owners[i].rpartition(".")[2] == "__init__"}
        if inits:
            # (c): the value the chain's first function was entered with, as __init__ left it
            held -= {ENTRY}
            for o in sorted(inits):
                # by whichever call built the object: not told (e)
                path.enter(o, None)
                n = _node(o)
                held |= (path.values(n, o, None) - {ENTRY}) if n is not None else frozenset(
                    i for i, ow in enumerate(path.owners) if ow == o)
        elif not entry:
            entry = f"{node.name}: entered with what no binding on the path gave"
    taken = {i for i in held if i != ENTRY}
    init = {i for i, o in enumerate(path.owners) if o.rpartition(".")[2] == "__init__"}
    if path.open:
        # a call on the path may go anywhere, or a caller of several may have run first: what is not placed may run
        # (``__init__`` ran before all of it, and is read where the value as entered is left)
        taken |= {i for i in range(len(items)) if i not in path.placed and i not in init}
    dropped = tuple((it.at, NOT_RUN if i in path.not_run else OFF_PATH if i not in path.placed and i not in init
                     else NOT_BETWEEN if kind == "change" else REBOUND)
                    for i, (it, kind) in enumerate(items) if i not in taken) + unread
    def by(i: int) -> set[str] | None:
        """(e2) the calls that built the object read, for its __init__'s binding; else (e) the calls the path entered
        the binding's function by, where no call on the path may go anywhere."""
        if path.owners[i] in path.built:
            return path.built[path.owners[i]]
        return None if path.open or i not in path.placed else path.entered.get(path.owners[i])

    entered = tuple((items[i][0].at, tuple(sorted(sites))) for i in sorted(taken)
                    if items[i][1] == "binding" and (sites := by(i)))
    return Choice(tuple(items[i][0] for i in sorted(taken) if items[i][1] == "binding"),
                  tuple(items[i][0] for i in sorted(taken) if items[i][1] == "change"), dropped, entry, entered)
