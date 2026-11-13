"""(C) (out/gm/qk-design.md): why a caller ran on the row when it did, read from its agent's cycles and their selection.

A JEDI agent's main loop calls a selecting function every cycle and hands the list it returns to worker threads, which
take parts of it (``self.taskList.get(10)``) and call the mover's caller on them: PostProcessor.start calls
getTasksToBeFinished_JEDI (80), wraps the tasks in a ListWithLock (88) and builds PostProcessorThreads on it (91),
whose runImpl hands each part to post_process_tasks (201).  When no pass of the caller took the row in the stay (U2,
KNOWN_ISSUES.md K55), what decides is which cycle's selection took the row, and when that batch's worker reached it.

- C1 :func:`selection_of`: from the caller to the selecting call, by the map's records: the entry the caller is
  called at (``EnteredFrom``), the worker's field it takes its part from (``FieldFrom``), the constructor's parameter
  that field is given (``ParameterFrom``, an object made at the list's line), and in the agent's loop the binding the
  list is made of -- the selecting call.
- C2 :func:`cycle_rows` and :func:`ask_strings`: the agent's rows around that call in its innermost loop -- the ones
  before it (a label's, a "getting" line's) and after it (the count) -- and the one literal they all hold, to ask the
  agent's logs by on every machine (no key: they print none).
- C3 :func:`selections` and :func:`batch_of`: the cycles those lines make on each machine (one mark a cycle, from
  ``MsgWrapper``'s time token), each selection's time and count; the pass of the caller on machine M at T belongs to
  M's last selection before T of the row's label, where those selections come one after another (one process).
- C4 :func:`lock_of`: why the other machines' selections did not take the row once one had: the selecting function's
  lock term (``lockedTime<:timeLimit``) and the time its limit is set back by.
"""

from __future__ import annotations

import ast
import datetime
import re
from dataclasses import dataclass
from typing import Any

from bamboo.codemap.panda import pathcond
from bamboo.querier import rows as R
from bamboo.querier import sources as SR
from bamboo.querier import windows as W
from bamboo.querier.records import unmarked

#: a literal the agent's lines are asked by is at least this long (a shorter one, "got ", is every other row's)
MIN_LITERAL = 5


def literals(pattern: str) -> list[str]:
    """The literal runs of a map row's *pattern* outside groups and classes: ``got\\ [-+]?[\\d.]+(?:...)?%?\\ tasks``
    gives ``["got ", " tasks"]``.  A character a quantifier follows is not certain, so it ends the run before it."""
    out: list[str] = []
    run: list[str] = []
    i, depth = 0, 0

    def close() -> None:
        if run:
            out.append("".join(run))
            run.clear()

    while i < len(pattern):
        ch = pattern[i]
        if ch == "\\" and i + 1 < len(pattern):
            nxt = pattern[i + 1]
            if depth == 0:
                if nxt.isalnum():
                    close()
                else:
                    run.append(nxt)
            i += 2
            continue
        if ch == "[":
            close()
            j = i + 1
            while j < len(pattern) and pattern[j] != "]":
                j += 2 if pattern[j] == "\\" else 1
            i = j + 1
            continue
        if ch == "(":
            close()
            depth += 1
        elif ch == ")":
            depth = max(depth - 1, 0)
        elif ch in "?*+{":
            if run:
                run.pop()
            close()
            if ch == "{":
                i = pattern.find("}", i) + 1 or len(pattern)
                continue
        elif ch in "|^$.":
            close()
        elif depth == 0:
            run.append(ch)
        i += 1
    close()
    return [s for s in out if s]


def ask_strings(rows: list[R.Row]) -> tuple[str, ...]:
    """What *rows* are asked by: the longest literal (:data:`MIN_LITERAL` or longer) every one of them holds, else each
    row's own longest literal.  PostProcessor's "preparing tasks to be finished for vo=.. label=..", "getting tasks to be
    finished" and "got N tasks" all hold " tasks"."""
    pieces = [[lit for p in row.map_patterns for lit in literals(p)] for row in rows]
    if not pieces or not all(pieces):
        return ()
    first = sorted({lit[i:j] for lit in pieces[0] for i in range(len(lit)) for j in range(i + MIN_LITERAL, len(lit) + 1)},
                   key=lambda s: (-len(s), s))
    common = next((s for s in first if all(any(s in lit for lit in ps) for ps in pieces[1:])), None)
    if common is not None:
        return (common,)
    return tuple(dict.fromkeys(max(ps, key=len) for ps in pieces if max(ps, key=len)))


@dataclass(frozen=True)
class Selection:
    """Where the list a worker takes its parts from is made (C1).  *worker* is the worker's class (``rel::Class``) and
    *field* the field its part comes from; the list is made at *rel*:*made* in the agent's loop *agent*
    (``rel::function``) of the value the call at *call* to *callee* returned, and handed to the worker at *handed*.
    *others*: the other ways the worker's list is given (``file:line``), a message processor's ``None``, whose
    passes this does not read."""

    worker: str
    field: str
    agent: str
    rel: str
    made: int
    handed: int
    call: int
    callee: str
    others: tuple[str, ...] = ()

    @property
    def agent_name(self) -> str:
        return self.agent.split("::")[-1]


def _takes(node: ast.AST) -> str | None:
    """The field *node* takes a part of a list from: ``x = self.F.get(...)``."""
    for n in ast.walk(node):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "get"
                and isinstance(n.func.value, ast.Attribute) and isinstance(n.func.value.value, ast.Name)
                and n.func.value.value.id == "self"):
            return n.func.value.attr
    return None


def _handed(site_fn: ast.AST, line: int, func: str) -> str | None:
    """The field the call to *func* at *line* of *site_fn* is handed a part of: its argument is a local bound from
    ``self.F.get(...)``."""
    stmt = R.statement_at(site_fn, line)
    if stmt is None:
        return None
    for call in (n for n in ast.walk(stmt) if isinstance(n, ast.Call)):
        name = call.func.attr if isinstance(call.func, ast.Attribute) else getattr(call.func, "id", "")
        if name != func:
            continue
        for arg in call.args:
            if not isinstance(arg, ast.Name):
                continue
            for b in pathcond.bindings(site_fn, arg.id):
                if not b.nested and b.value is not None and (got := _takes(b.value)) is not None:
                    return got
    return None


def selection_of(owner: str, line: int) -> tuple[Selection | None, str]:
    """C1 for the caller *owner* whose ``def`` is at *line*: the selecting call its worker's list comes from, or None
    with why not."""
    rel, func = owner.split("::", 1)
    holder = R.def_holding(rel, line)
    if holder is None:
        return None, f"{func} is not read"
    took = _takes(holder)
    if took is None:
        rec = SR.entered(owner, line)
        for site in getattr(rec, "sites", None) or []:
            srel = site.owner.split("::")[0]
            sfn = R.def_holding(srel, site.line)
            if sfn is not None and (took := _handed(sfn, site.line, func)) is not None:
                holder, rel = sfn, srel
                break
    if took is None:
        return None, (f"{func} takes no part of a worker's list (self.<field>.get()), nor is handed one by the map's "
                      "calls into it")
    qual = pathcond.qualified_name(holder)
    cls = f"{rel}::{qual.rpartition('.')[0]}" if "." in qual else ""
    ff = SR._field_froms().get((cls, took))
    params = [s for b in getattr(ff, "bindings", None) or [] for s in getattr(b, "sources", None) or []
              if getattr(s, "kind", "") == "parameter"]
    if len(params) != 1:
        return None, f"{cls.split('::')[-1]}.{took} is not given by one parameter in the map ({len(params)})"
    p = params[0]
    pf = SR._callers()[0].get((p.site_owner, p.name))
    sources = getattr(pf, "sources", None) or []
    made = [s for s in sources if getattr(s, "kind", "") == "object" and getattr(s, "at", None)]
    others = tuple(sorted({a for s in sources if s not in made for a in (getattr(s, "at", None) or [])}))
    if len(made) != 1:
        return None, f"{p.site_owner.split('::')[-1]}'s {p.name} is made at {len(made)} place(s) in the map, not one"
    mrel, _, mline = made[0].at[0].rpartition(":")
    handed = made[0].route[-1].rpartition(":")[2] if made[0].route else mline
    agent = R.def_holding(mrel, int(mline))
    stmt = R.statement_at(agent, int(mline)) if agent is not None else None
    call = next((n for n in ast.walk(stmt) if isinstance(n, ast.Call) and n.args), None) if stmt is not None else None
    arg = call.args[0] if call is not None else None
    if agent is None or not isinstance(arg, ast.Name):
        return None, f"the list made at {made[0].at[0]} is not made of a local"
    line_of = lambda b: getattr(b.site, "lineno", 0)  # noqa: E731
    before = [b for b in pathcond.bindings(agent, arg.id) if not b.nested and line_of(b) < int(mline)]
    if not before:
        return None, f"{arg.id} at {made[0].at[0]} has no binding before it"
    bound = max(before, key=line_of)
    value = bound.value
    if not isinstance(value, ast.Call):
        return None, f"{arg.id} is bound at {mrel}:{line_of(bound)} to no call"
    callee = value.func.attr if isinstance(value.func, ast.Attribute) else getattr(value.func, "id", "")
    return Selection(cls, took, f"{mrel}::{pathcond.qualified_name(agent)}", mrel, int(mline), int(handed),
                     line_of(bound), callee, others), ""


@dataclass(frozen=True)
class CycleRows:
    """C2: the agent's rows in the innermost loop around the selecting call: *before* it (in order) and *after* it;
    *label*, the hole a row before it prints the loop's element by (``prod_source_label``), or ""."""

    before: tuple[R.Row, ...]
    after: tuple[R.Row, ...]
    label: str = ""

    @property
    def rows(self) -> tuple[R.Row, ...]:
        return (*self.before, *self.after)


def cycle_rows(sel: Selection) -> CycleRows | None:
    """The agent's rows around the selecting call (C2), or None where the call is in no loop of it."""
    fn = R.function(sel.rel, sel.agent_name.split(".")[-1], within=[sel.call])
    stmt = R.statement_at(fn.node, sel.call)
    loop = next((a for a in pathcond.ancestors(stmt) if isinstance(a, (ast.For, ast.AsyncFor, ast.While))), None) if stmt else None
    if loop is None:
        return None
    inside = {id(n) for b in loop.body for n in ast.walk(b)}
    rows = sorted((r for r in fn.rows if id(r.stmt) in inside), key=lambda r: r.line)
    before = tuple(r for r in rows if r.line < sel.call)
    after = tuple(r for r in rows if r.line > sel.call)
    targets = {n.id for n in ast.walk(loop.target) if isinstance(n, ast.Name)} if isinstance(loop, (ast.For, ast.AsyncFor)) else set()
    label = next((t for r in before for t in targets if t in _holes(r)), "")
    return CycleRows(before, after, label)


def _holes(row: R.Row) -> set[str]:
    try:
        node = ast.parse(row.text, mode="eval").body
    except SyntaxError:
        return set()
    return {ast.unparse(v.value) for v in ast.walk(node) if isinstance(v, ast.FormattedValue)}


@dataclass(frozen=True)
class Pick:
    """One selection of a cycle (C3): on *machine*, under the cycle's *mark*, for *label* (or ""), the SELECT ran after
    *after* (the line before it in the cycle, or "") and returned *count* rows by *at* (the count's line)."""

    machine: str
    mark: str
    label: str
    after: str
    at: str
    count: str


def selections(records: list, crows: CycleRows, mode: str, log: str = "") -> tuple[dict[str, list[Pick]], dict[str, dict[str, tuple[str, str]]]]:
    """C3: each machine's selections in *records* (the agent's lines), in time order, and each machine's cycles as
    ``{mark: (first, last)}``.  A line is a row's where the row's pattern finds it; a count is a selection's end.  *log*:
    the log the records are of, whose lines with no mark (:func:`records.unmarked`) are no cycle's."""
    by_mark: dict[tuple[str, str], list] = {}
    for r in sorted(records, key=lambda r: r.stamp):
        # a line its logger writes with no mark (ContentsFeeder's "got N datasets") is no cycle's (K70)
        if not r.token or (log and unmarked(log, r.token, r.text)):
            continue
        by_mark.setdefault((r.machine, r.token), []).append(r)
    picks: dict[str, list[Pick]] = {}
    spans: dict[str, dict[str, tuple[str, str]]] = {}
    for (machine, mark), recs in by_mark.items():
        label, prev = "", ""
        found = False
        for r in recs:
            if any(row.matches(r, mode)[0] for row in crows.before):
                found = True
                for row in crows.before:
                    if crows.label and row.matches(r, mode)[0] and crows.label in (got := row.capture(r)):
                        label = got[crows.label].strip()
                prev = r.stamp
                continue
            count = next((row.capture(r) for row in crows.after if row.matches(r, mode)[0]), None)
            if count is not None:
                found = True
                n = next(iter(count.values()), "").strip() if count else ""
                picks.setdefault(machine, []).append(Pick(machine, mark, label, prev, r.stamp, n))
                prev = r.stamp
        if found:
            spans.setdefault(machine, {})[mark] = (recs[0].stamp, recs[-1].stamp)
    for m in picks:
        picks[m].sort(key=lambda p: p.at)
    return picks, spans


def one_process(picks: list[Pick], spans: dict[str, tuple[str, str]]) -> bool:
    """Whether the cycles *picks* come from follow one another: no two of their marks' spans overlap.  What P1 can be
    checked by, not P1: two processes whose short cycles take turns overlap nowhere, so P1 stays a premise."""
    marks = sorted({p.mark for p in picks}, key=lambda m: spans[m][0])
    return all(spans[a][1] <= spans[b][0] for a, b in zip(marks, marks[1:], strict=False))


def batch_of(picks: list[Pick], at: str, label: str, since: str) -> tuple[Pick | None, str]:
    """C3: the selection the pass beginning *at* belongs to: the last of *picks* (one machine's, in order) for *label*
    ending before *at*, which must end after the row became selectable (*since*)."""
    mine = [p for p in picks if not label or p.label == label]
    before = [p for p in mine if p.at[:23] <= at[:23]]
    if not before:
        return None, f"no selection before {at[:19]}"
    pick = before[-1]
    if pick.at[:23] < since[:23]:
        return None, f"the last selection before {at[:19]} ended at {pick.at[11:23]}, before the row could be selected ({since[:19]})"
    return pick, ""


@dataclass(frozen=True)
class Lock:
    """C4: the selecting function skips a row locked less than *span* ago: its *term* compares the row's lock time with
    *limit*, bound *bound* (``now - timedelta(...)``), and a take sets the lock time (*sets*)."""

    span: datetime.timedelta
    term: str
    bound: str
    sets: str


def lock_of(sel: Selection) -> Lock | None:
    """C4: the lock term of the selecting function *sel.callee*, read where the corpus defines it once in a
    database module, or None."""
    owners = [o for o in _defs_named(sel.callee) if "/db_proxy_mods/" in o or "DBProxy" in o]
    if len(owners) != 1:
        return None
    rel, func = owners[0].split("::", 1)
    try:
        fn = R.function(rel, func.split(".")[-1])
    except (LookupError, ValueError, SyntaxError):
        return None
    texts = [n.value for n in ast.walk(fn.node) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    term = next((m for t in texts for m in re.findall(r"lockedTime\s*<\s*:(\w+)", t)), None)
    sets = next((m.group(0) for t in texts if (m := re.search(r"lockedTime\s*=\s*CURRENT_DATE", t))), "")
    if term is None or not sets:
        return None
    for b in (b for name in _names_given(fn.node, f":{term}") for b in pathcond.bindings(fn.node, name)):
        span = _timedelta_back(b.value)
        if span is not None:
            return Lock(span, f"lockedTime<:{term}", ast.unparse(b.site).strip(), sets)
    return None


def passed_over(picks: dict[str, list[Pick]], first: Pick, label: str, since: str) -> list[Pick]:
    """The selections for *label* that began after the row became selectable (*since*) and ended before the first take
    *first* began, on any machine: each could have taken the row and did not, and nothing here reads why (a lock
    another agent holds, the count's cap, a bound on the key)."""
    lo = max(when(first.after or first.at), when(since))
    return [q for qs in picks.values() for q in qs if q is not first and (not label or q.label == label)
            and when(q.after or q.at) >= when(since) and when(q.at) < lo]


def kept_off(take: Pick, nxt: Pick, between: list[Pick], lock: Lock, since: str) -> tuple[bool, str]:
    """C4: whether *lock* says why the selections *between* the take *take* and the next one, *nxt*, did not take the
    row: the take locked it at some time from its SELECT's start (or the row's becoming selectable, *since*, if later)
    to its count's line, so the lock lasted to that time plus *lock.span*; each selection between began before the
    latest end, and the next take ended after the earliest.  ``(True, "")``, or False with why not."""
    lo = max(when(take.after or take.at), when(since))
    hi = when(take.at)
    if when(nxt.at) < lo + lock.span:
        return False, f"the next take ended at {nxt.at[11:23]}, before the lock from {take.at[11:23]} could end"
    late = [p for p in between if when(p.after or p.at) > hi + lock.span]
    if late:
        return False, f"{len(late)} selection(s) began after the lock had ended ({(hi + lock.span):%H:%M:%S}) and did not take it"
    return True, ""


def _names_given(node: ast.AST, key: str) -> list[str]:
    """The locals ``varMap[key] = name`` gives *key*."""
    out = []
    for n in ast.walk(node):
        if (isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Subscript)
                and isinstance(n.targets[0].slice, ast.Constant) and n.targets[0].slice.value == key
                and isinstance(n.value, ast.Name)):
            out.append(n.value.id)
    return list(dict.fromkeys(out))


def _timedelta_back(value: Any) -> datetime.timedelta | None:
    """``now() - datetime.timedelta(minutes=10)``'s timedelta, of constant keywords only."""
    if not (isinstance(value, ast.BinOp) and isinstance(value.op, ast.Sub) and isinstance(value.right, ast.Call)):
        return None
    call = value.right
    name = call.func.attr if isinstance(call.func, ast.Attribute) else getattr(call.func, "id", "")
    if name != "timedelta" or call.args:
        return None
    kw = {k.arg: float(k.value.value) for k in call.keywords
          if k.arg and isinstance(k.value, ast.Constant) and isinstance(k.value.value, (int, float))}
    if len(kw) != len(call.keywords):
        return None
    try:
        return datetime.timedelta(**kw)
    except TypeError:
        return None


def _defs_named(name: str) -> list[str]:
    """Every corpus def named *name* the map holds an entry of (``rel::qualname``)."""
    _params, entries = SR._callers()
    return sorted({o for o, _line in entries if o.split("::")[-1].split(".")[-1] == name})


def when(stamp: str) -> datetime.datetime:
    return W.when(stamp)
