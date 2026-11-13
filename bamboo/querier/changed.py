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
  ``self`` -- the one object a call on ``self`` hands on.  With them go the candidates there at a statement that runs a
  call (R1: ``self.finalize_job_status()``, an arm by the map's reading; what the call writes is not read, only the
  value before it, so it is never a write of another value, and what it runs before its write is not read either; a
  ``raise`` writes through a handler, and is none), and the candidates of the runs inside its calls: the
  pass's write, which takes the chain as it is at the call.  An assignment in another class, of an object handed on,
  is not read as the same object and is left as H2 left it.
* **The order** (i).  Each of them is placed at a statement of the anchor -- its own, the anchor's calls on into its
  function (the map's reach, *called_at*), or the call a run inside ran in -- and two are in order where one call can
  run the second after the first and not the first after the second (``flow.runs_after``).  With any two out of
  order, R says nothing.  A stay before a call or a write inside one, read from the rows alone, is out before the
  order is read (R3): the order is read between the rest.
* **The value before a write** (ii).  A printed hole of the chain at a row before it, or the constant an earlier
  assignment that ran left there, where nothing on the way binds the chain or hands what holds it to a call that can
  change it (P's reading of a call, ``printed.self_effects``).  The way is the rest of the source's function, the
  anchor's statements between the two places, and the write's own function before it.  The frames between the anchor
  and a function it reaches through others are not read.  Before the pass's write, its own frame first (R2): a hole of
  the chain it assigns at a row every way to it prints first, printed once in its run, so by the call that ran the
  write, carried to it past the statements the run shows did not run -- a must row of theirs missing, its absence
  trusted, as rule 4 reads it (archiveJob's 1268 ``start status=finished`` past 1495 ``job.jobStatus = "failed"``,
  whose 1494 the run lacks).  Else the value at the call it ran in.
* **What keeps the value on the way.**  An assignment of the value's own constant to the chain, and a call P reads as
  able to change it that cannot bind what holds it anew -- for a chain of ``self``, ``self_effects`` binds none of
  its prefixes; else the chain is the field of the name the call is handed -- where the map's reach from that call
  alone goes on to no arm assigning the field another value in memory, with no hole left (*keeps*, rerun's).  It
  rests on the reach's premises and on the junctions as every assignment of the field by its name: a write of an
  attribute by a name made at run time (``JobSpec.pack``) is none.
* **The value an assignment assigns.**  Its constant, or a printed hole of the expression it assigns at a row on
  either side of it, nothing between binding that expression.
* **The answer** (iv).  A write with the observed value before it is a stay, and an assignment of another value is
  no write of it: both are excluded.  The rest, read from the last back, may each be the last change, down to one
  that ran with another value before it: that one changed it, and the ones before it did not change it last.  With
  one left, it is chosen if it ran; with more, they are named and none is chosen.

What it does not read: the status at the pass's start (the statuslog's row before, design (iii) of pv-design.md).
"""
from __future__ import annotations

import ast
import functools
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bamboo.codemap import pathcond
from bamboo.codemap.panda import flow
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
    #: the pass's candidates in the order they run, less the stays read without it
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


def _flow_order(first: ast.stmt, then: ast.stmt, first_printed: bool, then_printed: bool,
                once: tuple[ast.AST, ...] = ()) -> bool | None:
    """True where one call runs *then* after *first* and never the other way, False for the reverse, else None.
    *once*: loops the pass went round once."""
    if first is then:
        return None
    ab = flow.runs_after(first, then, printed=first_printed, once=once)
    ba = flow.runs_after(then, first, printed=then_printed, once=once)
    return True if ab and not ba else False if ba and not ab else None


def precedes(a: Point, b: Point, anchor: R.Function, once: tuple[ast.AST, ...] = ()) -> bool | None:
    """Whether the pass runs *a* before *b* (None: not read)."""
    if _same_frame(a, b, anchor):
        return _flow_order(a.stmt, b.stmt, a.printed, b.printed, once)
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


def _calls(s: ast.stmt) -> bool:
    """Whether *s* runs a call itself, a log call aside."""
    return any(isinstance(n, ast.Call) and not R.is_log_call(n) for part in _parts(s) for n in R.own_nodes(part))


#: rerun's reading of a statement's calls by the map's reach: ``(fn, node, value)`` -> how many calls it closed by their
#: name, where the calls in *node* assign the subject's field no other value than *value*, else None
Keeps = Callable[[R.Function, ast.AST, str], "int | None"]


@dataclass
class Way:
    """What kept a value on its way: assignments of its own constant, calls the map's reach reads as keeping it, and
    statements that could change it the run shows did not run."""

    binds: list[str] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)
    closed: int = 0
    unrun: list[str] = field(default_factory=list)

    def said(self, attr: str, value: str) -> str:
        out = "nothing on the way binding it"
        if self.binds:
            out += f" but to {value!r} again at {', '.join(dict.fromkeys(self.binds))}"
        if self.calls:
            closed = f", {self.closed} call(s) closed by their name" if self.closed else ""
            out += (f"; the calls at {', '.join(dict.fromkeys(self.calls))} reach no assignment of {attr} but "
                    f"{value!r} (the map's reach{closed})")
        if self.unrun:
            out += f"; not run: {', '.join(dict.fromkeys(self.unrun))}"
        return out

    def add(self, other: Way) -> None:
        self.binds += other.binds
        self.calls += other.calls
        self.closed += other.closed
        self.unrun += other.unrun


def _place(fn: R.Function, node: ast.AST) -> str:
    return f"{Path(fn.rel).name}:{getattr(node, 'lineno', '?')}"


def _kept(fn: R.Function, part: ast.AST, value: str | None, keeps: Keeps | None, way: Way | None) -> bool:
    if value is None or keeps is None or (closed := keeps(fn, part, value)) is None:
        return False
    if way is not None:
        way.calls.append(_place(fn, part))
        way.closed += closed
    return True


def breaks(fn: R.Function, s: ast.stmt, chain: str, value: str | None = None, keeps: Keeps | None = None,
           way: Way | None = None) -> bool:
    """Whether statement *s* of *fn* can change what *chain* holds: it binds the chain, or a call in it can (P's reading
    of a call: what a method of self binds of it and the objects below it the calls are handed, else every call handed
    the chain's root).  Given the *value* it holds, an assignment of that constant keeps it, and so does a call that
    cannot bind what holds the chain anew where *keeps* says the map's reach from it assigns the field no other value;
    each is noted in *way*."""
    nodes = [n for part in _parts(s) for n in R.own_nodes(part)]
    for n in [s, *nodes]:
        for target, given in PR.binds(n):
            if not PR.affects(target, {chain}):
                continue
            if value is not None and target == chain and isinstance(given, ast.Constant) and given.value == value:
                if way is not None:
                    way.binds.append(_place(fn, n))
                continue
            return True
    for part in _parts(s):
        if not any(isinstance(n, ast.Call) and not R.is_log_call(n) for n in R.own_nodes(part)):
            continue
        if PR.root(chain) == "self" and (effects := PR.self_effects(fn.node, part)) is not None:
            bound, handed = effects
            if any(chain.startswith((b + ".", b + "[")) for b in bound):
                # what holds the chain bound anew: the reach of a field does not follow it
                return True
            if (any(PR.affects(b, {chain}) for b in bound) or any(chain.startswith((h + ".", h + "[")) for h in handed)) \
                    and not _kept(fn, part, value, keeps, way):
                return True
        elif PR.root(chain) in PR.mutating_roots(part):
            # a call cannot bind a name it is handed anew, only change its object: the field of that name alone
            if PR.root(chain) == "self" or chain.count(".") != 1 or not _kept(fn, part, value, keeps, way):
                return True
    return False


@functools.lru_cache(maxsize=None)
def _between(func: ast.AST, first: ast.stmt | None, then: ast.stmt | None, printed: bool,
             once: tuple[ast.AST, ...] = ()) -> tuple[ast.stmt, ...]:
    """The statements of *func* one call can run after *first* and before *then* (None: the start, the end)."""
    out = []
    for s in R.own_nodes(func):
        if not isinstance(s, ast.stmt) or s is func or s is first or s is then:
            continue
        if first is not None and not flow.runs_after(first, s, printed=printed, once=once):
            continue
        if then is not None and not flow.runs_after(s, then, once=once):
            continue
        out.append(s)
    return tuple(out)


def _class(fn: R.Function) -> ast.ClassDef | None:
    return next((a for a in pathcond.ancestors(fn.node) if isinstance(a, ast.ClassDef)), None)


#: rule 4's reading of a statement in one run: the lines of its must rows the run lacks where that is evidence it did
#: not run (:func:`unrun`), none where it may have run
Unrun = Callable[[ast.stmt], list[int]]


def unrun(sn: R.Seen) -> Unrun:
    """Rule 4's reading of a statement of *sn*'s function in its run: a must row of it (``rows.must_rows``, as a
    traceback of the run places it) missing from the run, its absence trusted at a level the run prints
    (``Seen.would_show``).  Every run of the function is read: a row no call printed shows that no call ran it."""
    named, raised = R.raised_in(sn.fn, sn.run)

    def lacking(s: ast.stmt) -> list[int]:
        before, after = R.must_rows(sn.fn.node, s, raised if named else set())
        return [row.line for m in before + after
                if (row := sn.fn.row_at(m)) is not None and not sn.present(row) and sn.would_show(row)]
    return lacking


def holds(q: Point, p: Point, chain: str, anchor: R.Function, value: str | None = None,
          keeps: Keeps | None = None, once: tuple[ast.AST, ...] = (), skip: Unrun | None = None) -> Way | None:
    """Whether what *chain* held at *q*, *value* where it is known, still holds at *p*, a later place of the pass:
    nothing on the way can change it (:func:`breaks`), and what kept it.  Across functions only a chain of ``self`` in
    methods of the anchor's class is one object in both, and the way is the rest of *q*'s function, the anchor's
    statements between the two, and *p*'s before it.  *once*: loops the pass went round once, read in one frame;
    *skip*: in one frame, a statement that could change it is passed where the run shows it did not run (:func:`unrun`)."""
    way = Way()
    if _same_frame(q, p, anchor):
        for s in _between(q.fn.node, q.stmt, p.stmt, q.printed, once):
            on = Way()
            if not breaks(q.fn, s, chain, value, keeps, on):
                way.add(on)
            elif skip is not None and (lacking := skip(s)):
                way.unrun.append(f"{_place(q.fn, s)} (must row(s) {lacking} not in the run)")
            else:
                return None
        return way
    cls = _class(anchor)
    if PR.root(chain) != "self" or cls is None or _class(q.fn) is not cls or _class(p.fn) is not cls:
        return None
    if len(q.at) != 1 or len(p.at) != 1 or q.at[0] is p.at[0]:
        return None
    ways: list[tuple[R.Function, tuple[ast.stmt, ...]]] = []
    if q.fn.owner != anchor.owner:
        ways.append((q.fn, _between(q.fn.node, q.stmt, None, q.printed)))
    ways.append((anchor, _between(anchor.node, q.at[0], p.at[0], q.printed and q.fn.owner == anchor.owner)))
    if p.fn.owner != anchor.owner:
        ways.append((p.fn, _between(p.fn.node, None, p.stmt, False)))
    return None if any(breaks(fn, s, chain, value, keeps, way) for fn, stmts in ways for s in stmts) else way


def printed_at(fn: R.Function, sn: R.Seen, at: ast.stmt, wanted: str, read: Callable[[R.Row], list[int]],
               once: tuple[ast.AST, ...] = ()) -> tuple[str, str] | None:
    """*wanted*'s value at *at*, a statement of *fn*, from what *fn*'s run *sn* printed, with where: a hole of *wanted*
    at *at* itself, or at a row before or after it that holds there (:func:`holds`); or a hole of a chain that one
    assignment after *at* binds to *wanted*, *wanted* holding up to it and the chain from it to the row (1107
    ``job.ddmErrorDiag = disp_error[disp]``, printed at 1108).  Every one agreeing, said by the first.  *read*: the
    record indexes a row is read at, one or none; *once*: loops the pass went round once."""
    p = Point(fn, at, [at])
    found: list[tuple[str, str]] = []
    for row in fn.rows:
        if len(at_ := read(row)) != 1:
            continue
        record = sn.run.records[at_[0]]
        q = Point(fn, row.stmt, [row.stmt], True)
        where = f"row {fn.name}:{row.line} at {record.span}"
        after = row.stmt is not at and precedes(p, q, fn, once)
        for expr, value in row.capture(record).items():
            value = value.strip()
            # a hole str(x) prints what {x} prints
            if expr in (wanted, f"str({wanted})"):
                if row.stmt is at or (precedes(q, p, fn, once) and holds(q, p, wanted, fn, value, once=once) is not None) \
                        or (after and holds(p, q, wanted, fn, value, once=once) is not None):
                    found.append((value, where))
                continue
            if not after:
                continue
            for a in _between(fn.node, at, row.stmt, False, once):
                if isinstance(a, ast.Assign) and len(a.targets) == 1 and ast.unparse(a.targets[0]) == expr \
                        and ast.unparse(a.value) == wanted:
                    bound = Point(fn, a, [a])
                    if holds(p, bound, wanted, fn, once=once) is not None and holds(bound, q, expr, fn, value, once=once) is not None:
                        found.append((value, f"{where}, as {expr} bound at {Path(fn.rel).name}:{a.lineno}"))
    return found[0] if len({value for value, _ in found}) == 1 else None


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
    keeps: Keeps | None = None,
) -> Change | None:
    """Rule R on a judgement's verdicts: None where it does not apply -- no anchor whose pass holds two writes of one
    chain, or two of them out of order.  *joined*: ``id(run)`` -> the calls a run inside the question's ran in.
    *keeps* reads a call on a way of the subject's field by the map's reach (:func:`breaks`)."""
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
        of self in the methods of its class; the candidates there at a statement that runs a call (R1: what the call
        writes is not read, only the value before it); and the candidates of the runs inside its calls (the write)."""
        cls = _class(anchor)

        def ours(fn: R.Function, c: str) -> bool:
            return fn.owner == anchor.owner or (PR.root(c) == "self" and cls is not None and _class(fn) is cls)

        chains: dict[int, str] = {}
        for v in in_pass:
            if inside(v) or (bound := assignment(stmts[id(v)], subject)) is None:
                continue
            if ours(seens[v.candidate.owner].fn, bound[0]):
                chains[id(v)] = bound[0]
        if len(set(chains.values())) != 1:
            return None
        chain = next(iter(chains.values()))
        for v in in_pass:
            # a raise writes through the handler it leaves for, not through the call it builds its exception with
            if not inside(v) and id(v) not in chains and assignment(stmts[id(v)], subject) is None \
                    and not isinstance(stmts[id(v)], ast.Raise) and _calls(stmts[id(v)]) \
                    and ours(seens[v.candidate.owner].fn, chain):
                calls.add(id(v))
        return chain, [v for v in in_pass if id(v) in chains or id(v) in calls or inside(v)]

    anchor, chain = None, ""
    members: list[Any] = []
    #: the members at a statement that runs a call (R1), of the anchor found
    calls: set[int] = set()
    for owner in owners:
        fn = seens[owner].fn
        calls.clear()
        if id(seens[owner].run) in joined or not seens[owner].single_call() or (found := scope(fn)) is None:
            continue
        if len(found[1]) >= 2 and all(placed(fn, v.candidate.owner, stmts[id(v)]) for v in found[1]):
            anchor, (chain, members) = fn, found
            break
    if anchor is None:
        return None
    points = {id(v): Point(seens[v.candidate.owner].fn, stmts[id(v)], placed(anchor, v.candidate.owner, stmts[id(v)]))
              for v in members}

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

    def keeping(c: str) -> Keeps | None:
        """*keeps* for a chain of the subject's field: the reach answers for that field alone."""
        return keeps if c.rsplit(".", 1)[-1] == subject else None

    def read(p: Point, wanted: str, later: bool) -> tuple[str, str, Way] | None:
        """*wanted*'s value at *p* from the rows before it (or, *later*, after it too), every one that holds agreeing;
        said by the first of them in the pass's rows."""
        found: list[tuple[str, str, Way]] = []
        for q, expr, value, where in printed:
            if expr != wanted:
                continue
            if precedes(q, p, anchor) and (way := holds(q, p, wanted, anchor, value, keeping(wanted))) is not None:
                found.append((value, where, way))
            elif later and precedes(p, q, anchor) and (way := holds(p, q, wanted, anchor, value, keeping(wanted))) is not None:
                found.append((value, where, way))
        return found[0] if len({value for value, _, _ in found}) == 1 else None

    def own_frame(v: Any) -> tuple[str, str, Way] | None:
        """R2: the value before a write inside a call, read in its own frame -- a hole of the chain it assigns at a row
        every way to it prints first, printed once in its run (so by the call that ran the write), carried to it
        (:func:`holds`) past the statements the run shows did not run (:func:`unrun`); every one agreeing."""
        sn = seens[v.candidate.owner]
        fn, stmt = sn.fn, stmts[id(v)]
        if (own := assignment(stmt, subject)) is None:
            return None
        named, raised = R.raised_in(fn, sn.run)
        skip = unrun(sn)
        found: list[tuple[str, str, Way]] = []
        for m in R.must_rows(fn.node, stmt, raised if named else set())[0]:
            row = fn.row_at(m)
            if row is None or len(sn.hits.get(row.line, [])) != 1 or len(at := sn.unambiguous.get(row.line, [])) != 1:
                continue
            record = sn.run.records[at[0]]
            for expr, value in row.capture(record).items():
                if expr == own[0] and (way := holds(Point(fn, row.stmt, [row.stmt], True), Point(fn, stmt, [stmt]), own[0], fn,
                                                    value.strip(), keeping(own[0]), skip=skip)) is not None:
                    found.append((value.strip(), f"row {fn.name}:{row.line} at {record.span}", way))
        return found[0] if len({value for value, _, _ in found}) == 1 else None

    def at_call(v: Any) -> Point | None:
        """Where the pass's write inside a call takes the chain: the call, its own target not bound before it."""
        p = points[id(v)]
        own = assignment(stmts[id(v)], subject)
        if len(p.at) != 1 or (own is not None and any(breaks(p.fn, s, own[0]) for s in _between(p.fn.node, None, p.stmt, False))):
            return None
        return Point(anchor, p.at[0], p.at)

    def stay(v: Any, b: tuple[str, str, Way], own: bool) -> str:
        if own and (target := assignment(stmts[id(v)], subject)) is not None:
            return f"rule R: a stay, {target[0]} already held {observed!r} before it in its own frame ({b[1]}, {b[2].said(subject, observed)})"
        frames = "" if v.candidate.owner == anchor.owner else f"; the frames between {anchor.name} and {v.candidate.func} not read"
        if id(v) in calls:
            # R1: the write the call stands for is in what it runs, and the way there is not read
            frames += "; what the call runs before its write not read"
        if inside(v):
            sites = " or ".join(f"{Path(f).name}:{line}" for f, line in _sites(joined[id(seens[v.candidate.owner].run)]))
            why = f"the pass writes {chain} as it is at the call at {sites}, {observed!r} there"
        else:
            why = f"{chain} already held {observed!r} before it"
        return f"rule R: a stay, {why} ({b[1]}, {b[2].said(subject, observed)}{frames})"

    change = Change([])
    # R3: the stays read without the pass's order -- before a call (R1), before a write inside one (R2) -- out first;
    # the order is read between the rest
    owns: dict[int, tuple[str, str, Way] | None] = {}
    for v in members:
        if inside(v):
            owns[id(v)] = own_frame(v)
            if owns[id(v)] is not None:
                b, own = owns[id(v)], True
            else:
                b, own = (read(p, chain, False) if (p := at_call(v)) is not None else None), False
        elif id(v) in calls:
            b, own = read(points[id(v)], chain, False), False
        else:
            continue
        if b is not None and b[0] == observed:
            change.dropped.append((v, stay(v, b, own)))
    rest = [v for v in members if all(v is not d for d, _ in change.dropped)]
    # the pass's order: every pair read, or nothing
    for i, a in enumerate(rest):
        for b_ in rest[i + 1:]:
            if precedes(points[id(a)], points[id(b_)], anchor) is None:
                return None
    order = sorted(rest, key=lambda v: sum(bool(precedes(points[id(w)], points[id(v)], anchor)) for w in rest if w is not v))
    change.order = order

    assigns: dict[int, tuple[str, str] | None] = {}
    for v in order:
        stmt = stmts[id(v)]
        rhs = stmt.value if isinstance(stmt, (ast.Assign, ast.AnnAssign)) and id(v) not in calls else None
        if isinstance(rhs, ast.Constant) and isinstance(rhs.value, str):
            assigns[id(v)] = (rhs.value, "its constant")
        elif isinstance(rhs, (ast.Name, ast.Attribute)) and not inside(v):
            got = read(points[id(v)], ast.unparse(rhs), True)
            assigns[id(v)] = (got[0], f"{ast.unparse(rhs)}, printed at {got[1]}") if got else None
        else:
            assigns[id(v)] = None

    before: dict[int, tuple[str, str, Way] | None] = {}
    for n, v in enumerate(order):
        p = points[id(v)]
        if inside(v):
            # the write: read in its own frame (R2), else the row takes the chain as it is at the call, its own target
            # not bound before it
            before[id(v)] = owns[id(v)]
            if before[id(v)] is not None or (call := at_call(v)) is None:
                continue
            p = call
        # said by a printed row where one holds, else by the first assignment
        held: list[tuple[str, str, Way]] = []
        if (shown := read(p, chain, False)) is not None:
            held.append(shown)
        for e in order[:n]:
            if e.kind == "positive" and not inside(e) and (given := assigns[id(e)]) is not None \
                    and (way := holds(points[id(e)], p, chain, anchor, given[0], keeping(chain))) is not None:
                held.append((given[0], f"assigned by {_name(e)}", way))
        before[id(v)] = held[0] if len({value for value, _, _ in held}) == 1 else None
    for v in order:
        if (b := before[id(v)]) is not None and b[0] == observed:
            change.dropped.append((v, stay(v, b, False)))
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
