"""Rule R: of one pass's writes of the observed value, the last that changed it (KNOWN_ISSUES.md K51).

H2 judges which candidates ran.  Where one pass ran several, the row holds the value the pass assigned last, and the
write that put it there is the last one that changed it: the adder's ``process_job_report`` sets
``self.job.jobStatus`` from the pilot's report at 416, ``parse_job_output_report`` sets it "failed" at 919, it sets it
"failed" again at 441, and ``handle_failed_job`` at 374 through ``finalize_job_status``, all before updateJobs writes
the row at 472.  A write that assigns the value already there is a *stay*, as the census reads a row written again with
its own value.

* **The pass.**  The candidates H2 left (positive and kept) of the question's run and of the runs inside a call it
  made (``which_ran.held_in_calls``).
* **One field of one object.**  One function of the pass, the *anchor*, its rows read as one call
  (``Seen.single_call``), assigns the subject's chain itself, or methods of its class assign the same chain of
  ``self`` -- the one object a call on ``self`` hands on.  With them go the candidates of the runs inside its calls:
  the pass's write, which takes the chain as it is at the call.  An assignment in another class, of an object handed
  on, is not read as the same object and is left as H2 left it.
* **The order** (i).  Each of them is placed at a statement of the anchor -- its own, the anchor's calls on into its
  function (the map's reach, *called_at*), or the call a run inside ran in -- and two are in order where one call can
  run the second after the first and not the first after the second (``flow.runs_after``).  With any two out of
  order, R says nothing.
* **The value before a write** (ii).  A printed hole of the chain at a row before it, or the constant an earlier
  assignment that ran left there, where nothing on the way binds the chain or hands what holds it to a call that can
  change it (P's reading of a call, ``printed.self_effects``).  The way is the rest of the source's function, the
  anchor's statements between the two places, and the write's own function before it.  The frames between the anchor
  and a function it reaches through others are not read.  Before the pass's write, the value at the call it ran in.
* **The value an assignment assigns.**  Its constant, or a printed hole of the expression it assigns at a row on
  either side of it, nothing between binding that expression.
* **The answer** (iv).  A write with the observed value before it is a stay, and an assignment of another value is
  no write of it: both are excluded.  The rest, read from the last back, may each be the last change, down to one
  that ran with another value before it: that one changed it, and the ones before it did not change it last.  With
  one left, it is chosen if it ran; with more, they are named and none is chosen.

What it does not read: the status at the pass's start (the statuslog's row before, design (iii) of pv-design.md), and a
call that keeps the observed value because every write it reaches assigns that value (the map's reach leaves the
adder's calls at 421 and 425 open).
"""
from __future__ import annotations

import ast
import functools
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bamboo.codemap.panda import flow, pathcond
from bamboo.querier import printed as PR
from bamboo.querier import rows as R
from bamboo.querier.records import Run


@dataclass
class Point:
    """A place of the pass: *stmt* of *fn*, and where it sits in the anchor (*at*: the statement itself, or the
    anchor's calls on into *fn*).  A row (*printed*) has run once it printed."""

    fn: R.Function
    stmt: ast.stmt
    at: list[ast.stmt]
    printed: bool = False


@dataclass
class Change:
    #: the pass's candidates in the order they run
    order: list[Any]
    #: the stays and the writes of another value, each with why
    dropped: list[tuple[Any, str]] = field(default_factory=list)
    #: the candidates that may have changed it last, the last first
    left: list[Any] = field(default_factory=list)
    chosen: Any = None
    how: str = ""


def _name(v: Any) -> str:
    return f"{Path(v.candidate.rel).name}::{v.candidate.func}:{v.line}"


def _sites(site: str) -> list[tuple[str, int]]:
    """``held_in_calls``' ``"a.py:9 or a.py:11"`` as ``[(file, line)]``."""
    return [(f, int(n)) for one in site.split(" or ") for f, _, n in [one.rpartition(":")]]


def _same_frame(a: Point, b: Point, anchor: R.Function) -> bool:
    return a.fn.owner == b.fn.owner and (a.fn.owner == anchor.owner or (len(a.at) == 1 and a.at == b.at))


def _flow_order(first: ast.stmt, then: ast.stmt, first_printed: bool, then_printed: bool) -> bool | None:
    """True where one call runs *then* after *first* and never the other way, False for the reverse, else None."""
    if first is then:
        return None
    ab = flow.runs_after(first, then, printed=first_printed)
    ba = flow.runs_after(then, first, printed=then_printed)
    return True if ab and not ba else False if ba and not ab else None


def precedes(a: Point, b: Point, anchor: R.Function) -> bool | None:
    """Whether the pass runs *a* before *b* (None: not read)."""
    if _same_frame(a, b, anchor):
        return _flow_order(a.stmt, b.stmt, a.printed, b.printed)
    if not a.at or not b.at:
        return None
    order = {_flow_order(x, y, a.printed and a.fn.owner == anchor.owner, b.printed and b.fn.owner == anchor.owner)
             for x in a.at for y in b.at}
    return next(iter(order)) if len(order) == 1 else None


def _parts(s: ast.stmt) -> list[ast.AST]:
    """What a statement runs itself: a compound one's header, its body's statements read on their own."""
    if isinstance(s, (ast.If, ast.While)):
        return [s.test]
    if isinstance(s, (ast.For, ast.AsyncFor)):
        return [s.iter]
    if isinstance(s, (ast.With, ast.AsyncWith)):
        return [i.context_expr for i in s.items]
    if isinstance(s, ast.Match):
        return [s.subject]
    if isinstance(s, (ast.Try, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return []
    return [s]


def breaks(fn: R.Function, s: ast.stmt, chain: str) -> bool:
    """Whether statement *s* of *fn* can change what *chain* holds: it binds the chain, or a call in it can (P's reading
    of a call: what a method of self binds of it and the objects below it the calls are handed, else every call handed
    the chain's root)."""
    nodes = [n for part in _parts(s) for n in R.own_nodes(part)]
    if any(PR.affects(t, {chain}) for n in [s, *nodes] for t, _ in PR.binds(n)):
        return True
    for part in _parts(s):
        if not any(isinstance(n, ast.Call) and not R.is_log_call(n) for n in R.own_nodes(part)):
            continue
        if PR.root(chain) == "self" and (effects := PR.self_effects(fn.node, part)) is not None:
            bound, handed = effects
            if any(PR.affects(b, {chain}) for b in bound) or any(chain.startswith((h + ".", h + "[")) for h in handed):
                return True
        elif PR.root(chain) in PR.mutating_roots(part):
            return True
    return False


@functools.lru_cache(maxsize=None)
def _between(func: ast.AST, first: ast.stmt | None, then: ast.stmt | None, printed: bool) -> tuple[ast.stmt, ...]:
    """The statements of *func* one call can run after *first* and before *then* (None: the start, the end)."""
    out = []
    for s in R.own_nodes(func):
        if not isinstance(s, ast.stmt) or s is func or s is first or s is then:
            continue
        if first is not None and not flow.runs_after(first, s, printed=printed):
            continue
        if then is not None and not flow.runs_after(s, then):
            continue
        out.append(s)
    return tuple(out)


def _class(fn: R.Function) -> ast.ClassDef | None:
    return next((a for a in pathcond.ancestors(fn.node) if isinstance(a, ast.ClassDef)), None)


def holds(q: Point, p: Point, chain: str, anchor: R.Function) -> bool:
    """Whether what *chain* held at *q* still holds at *p*, a later place of the pass: nothing on the way can change it.
    Across functions only a chain of ``self`` in methods of the anchor's class is one object in both, and the way is
    the rest of *q*'s function, the anchor's statements between the two, and *p*'s before it."""
    if _same_frame(q, p, anchor):
        return not any(breaks(q.fn, s, chain) for s in _between(q.fn.node, q.stmt, p.stmt, q.printed))
    cls = _class(anchor)
    if PR.root(chain) != "self" or cls is None or _class(q.fn) is not cls or _class(p.fn) is not cls:
        return False
    if len(q.at) != 1 or len(p.at) != 1 or q.at[0] is p.at[0]:
        return False
    ways: list[tuple[R.Function, tuple[ast.stmt, ...]]] = []
    if q.fn.owner != anchor.owner:
        ways.append((q.fn, _between(q.fn.node, q.stmt, None, q.printed)))
    ways.append((anchor, _between(anchor.node, q.at[0], p.at[0], q.printed and q.fn.owner == anchor.owner)))
    if p.fn.owner != anchor.owner:
        ways.append((p.fn, _between(p.fn.node, None, p.stmt, False)))
    return not any(breaks(fn, s, chain) for fn, way in ways for s in way)


def assignment(stmt: ast.stmt | None, subject: str) -> tuple[str, ast.expr] | None:
    """``(chain, value)`` where *stmt* assigns one attribute chain ending in *subject*."""
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
        target, value = stmt.targets[0], stmt.value
    elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
        target, value = stmt.target, stmt.value
    else:
        return None
    if isinstance(target, ast.Attribute) and target.attr == subject:
        return ast.unparse(target), value
    return None


def last_change(
    verdicts: list,
    seens: dict[str, R.Seen],
    runs_of: dict[str, Run | None],
    question: Run,
    joined: dict[int, str],
    subject: str,
    observed: str,
    called_at: Callable[[R.Function, Any], list[str]] | None,
) -> Change | None:
    """Rule R on a judgement's verdicts: None where it does not apply -- no anchor whose pass holds two writes of one
    chain, or two of them out of order.  *joined*: ``id(run)`` -> the calls a run inside the question's ran in."""
    in_pass = [
        v for v in verdicts
        if v.kind in ("positive", "kept") and (r := runs_of.get(v.candidate.owner)) is not None
        and (r is question or id(r) in joined) and v.candidate.owner in seens
    ]
    if len(in_pass) < 2:
        return None
    stmts: dict[int, ast.stmt] = {}
    for v in in_pass:
        if (stmt := R.statement_at(seens[v.candidate.owner].fn.node, v.line)) is None:
            return None
        stmts[id(v)] = stmt
    owners = list(dict.fromkeys(v.candidate.owner for v in in_pass))
    of: dict[str, Any] = {v.candidate.owner: v.candidate for v in in_pass}

    def inside(v: Any) -> bool:
        return id(seens[v.candidate.owner].run) in joined

    def placed(anchor: R.Function, owner: str, stmt: ast.stmt) -> list[ast.stmt]:
        sn = seens[owner]
        if id(sn.run) in joined:
            found = [statement_at for file, line in _sites(joined[id(sn.run)]) if file == anchor.rel
                     and (statement_at := R.statement_at(anchor.node, line)) is not None]
            return found if len(found) == len(_sites(joined[id(sn.run)])) else []
        if owner == anchor.owner:
            return [stmt]
        if called_at is None:
            return []
        lines = sorted({int(n) for site in called_at(anchor, of[owner]) for f, _, n in [site.rpartition(":")] if f == anchor.rel})
        return [s for n in lines if (s := R.statement_at(anchor.node, n)) is not None]

    def scope(anchor: R.Function) -> tuple[str, list] | None:
        """The chain the pass assigns and its writes of it: the assignments of one chain in the anchor, or of one chain
        of self in the methods of its class, and the candidates of the runs inside its calls (the write)."""
        cls = _class(anchor)
        chains: dict[int, str] = {}
        for v in in_pass:
            if inside(v) or (bound := assignment(stmts[id(v)], subject)) is None:
                continue
            fn = seens[v.candidate.owner].fn
            if fn.owner == anchor.owner or (PR.root(bound[0]) == "self" and cls is not None and _class(fn) is cls):
                chains[id(v)] = bound[0]
        if len(set(chains.values())) != 1:
            return None
        return next(iter(chains.values())), [v for v in in_pass if id(v) in chains or inside(v)]

    anchor, chain = None, ""
    members: list[Any] = []
    for owner in owners:
        fn = seens[owner].fn
        if id(seens[owner].run) in joined or not seens[owner].single_call() or (found := scope(fn)) is None:
            continue
        if len(found[1]) >= 2 and all(placed(fn, v.candidate.owner, stmts[id(v)]) for v in found[1]):
            anchor, (chain, members) = fn, found
            break
    if anchor is None:
        return None
    points = {id(v): Point(seens[v.candidate.owner].fn, stmts[id(v)], placed(anchor, v.candidate.owner, stmts[id(v)]))
              for v in members}
    # the pass's order: every pair read, or nothing
    for i, a in enumerate(members):
        for b in members[i + 1:]:
            if precedes(points[id(a)], points[id(b)], anchor) is None:
                return None
    order = sorted(members, key=lambda v: sum(bool(precedes(points[id(w)], points[id(v)], anchor)) for w in members if w is not v))

    # what the pass's rows printed, each row printed once: (place, hole, value, where)
    printed: list[tuple[Point, str, str, str]] = []
    for owner in owners:
        sn = seens[owner]
        for row in sn.fn.rows:
            if len(at := sn.unambiguous.get(row.line, [])) != 1 or not (place := placed(anchor, owner, row.stmt)):
                continue
            record = sn.run.records[at[0]]
            for expr, value in row.capture(record).items():
                printed.append((Point(sn.fn, row.stmt, place, True), expr, value.strip(), f"row {sn.fn.name}:{row.line} at {record.span}"))

    def read(p: Point, wanted: str, later: bool) -> tuple[str, str] | None:
        """*wanted*'s value at *p* from the rows before it (or, *later*, after it too), every one that holds agreeing;
        said by the first of them in the pass's rows."""
        found: list[tuple[str, str]] = []
        for q, expr, value, where in printed:
            if expr != wanted:
                continue
            if precedes(q, p, anchor) and holds(q, p, wanted, anchor):
                found.append((value, where))
            elif later and precedes(p, q, anchor) and holds(p, q, wanted, anchor):
                found.append((value, where))
        return found[0] if len({value for value, _ in found}) == 1 else None

    assigns: dict[int, tuple[str, str] | None] = {}
    for v in order:
        stmt = stmts[id(v)]
        rhs = stmt.value if isinstance(stmt, (ast.Assign, ast.AnnAssign)) else None
        if isinstance(rhs, ast.Constant) and isinstance(rhs.value, str):
            assigns[id(v)] = (rhs.value, "its constant")
        elif isinstance(rhs, (ast.Name, ast.Attribute)) and not inside(v):
            got = read(points[id(v)], ast.unparse(rhs), True)
            assigns[id(v)] = (got[0], f"{ast.unparse(rhs)}, printed at {got[1]}") if got else None
        else:
            assigns[id(v)] = None

    change = Change(order)
    before: dict[int, tuple[str, str] | None] = {}
    for n, v in enumerate(order):
        p = points[id(v)]
        if inside(v):
            # the write: the row takes the chain as it is at the call, and its own target is not bound before it
            own = assignment(stmts[id(v)], subject)
            if len(p.at) != 1 or (own is not None and any(breaks(p.fn, s, own[0]) for s in _between(p.fn.node, None, p.stmt, False))):
                before[id(v)] = None
                continue
            p = Point(anchor, p.at[0], p.at)
        # said by a printed row where one holds, else by the first assignment
        held: list[tuple[str, str]] = []
        if (shown := read(p, chain, False)) is not None:
            held.append(shown)
        for e in order[:n]:
            if e.kind == "positive" and not inside(e) and (given := assigns[id(e)]) is not None \
                    and holds(points[id(e)], p, chain, anchor):
                held.append((given[0], f"assigned by {_name(e)}"))
        before[id(v)] = held[0] if len({value for value, _ in held}) == 1 else None
    for v in order:
        if (b := before[id(v)]) is not None and b[0] == observed:
            frames = "" if v.candidate.owner == anchor.owner else f"; the frames between {anchor.name} and {v.candidate.func} not read"
            if inside(v):
                calls = " or ".join(f"{Path(f).name}:{line}" for f, line in _sites(joined[id(seens[v.candidate.owner].run)]))
                why = f"the pass writes {chain} as it is at the call at {calls}, {observed!r} there"
            else:
                why = f"{chain} already held {observed!r} before it"
            change.dropped.append((v, f"rule R: a stay, {why} ({b[1]}, nothing on the way binding it{frames})"))
        elif not inside(v) and (a := assigns[id(v)]) is not None and a[0] != observed:
            change.dropped.append((v, f"rule R: it assigns {a[0]!r} ({a[1]}), no write of {observed!r}"))
    dropped = {id(v) for v, _ in change.dropped}
    for v in reversed(order):
        if id(v) in dropped:
            continue
        change.left.append(v)
        if v.kind == "positive" and before[id(v)] is not None:
            break
    if not change.left:
        return None
    if len(change.left) == 1 and change.left[0].kind == "positive":
        change.chosen, change.how = change.left[0], "the last change in the pass"
    else:
        change.how = f"{len(change.left)} may have changed it last in the pass: {', '.join(_name(v) for v in change.left)}"
    return change
