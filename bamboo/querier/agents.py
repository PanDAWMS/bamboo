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
- M3(a)-cap :func:`cap_take` (tmaeno's OK 2026-11-15 01:16:35, its build 01:46:09): where the selecting statement
  returns at most a cap of rows (the map's ``limits``, ``rownum<1000``), a selection that returned fewer took every row
  selectable then; from the time the statement began to return the row (the turn) the first such selection is the
  one that took it, and those before it returned the cap (others first) or none (a selection beside them took the same
  rows).
"""

from __future__ import annotations

import ast
import datetime
import re
from dataclasses import dataclass
from typing import Any

from bamboo.codemap import pathcond
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
    *label*, the hole a row before it prints the loop's element by (``prod_source_label``), or "" -- a row every pass
    to the call prints (:func:`_straight`), not one a branch or a handler prints on the way that skips the call
    (TaskCommando's "failed to get lock" line; P7, out/gm/qk-design.md, tmaeno's OK 2026-11-15 23:06)."""

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
    label = next((t for r in before if _straight(r.stmt, loop) for t in targets if t in _holes(r)), "")
    return CycleRows(before, after, label)


def _straight(stmt: ast.AST, loop: ast.For | ast.AsyncFor | ast.While) -> bool:
    """Whether *loop*'s body reaches *stmt* straight: through the bodies of ``with`` and ``try`` only, not an ``if``'s
    branch, a handler, an ``else`` or a nested loop -- so every pass of the body that reaches a later statement of it
    printed *stmt* first."""
    child = stmt
    for parent in pathcond.ancestors(stmt):
        if parent is loop:
            return any(c is child for c in loop.body)
        if isinstance(parent, (ast.With, ast.AsyncWith, ast.Try)) and any(c is child for c in parent.body):
            child = parent
            continue
        return False
    return False


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


@dataclass(frozen=True)
class Idle:
    """P8: a cycle that printed nothing while other machines' cycles selected: its line before (row *line* at *start*),
    its next line after (row *next_line* at *end*), and the agent's calls in its loop between the two rows (*calls*)."""

    machine: str
    mark: str
    start: str
    line: int
    end: str
    next_line: int
    calls: tuple[str, ...]


def idle_cycles(sel: Selection, crows: CycleRows, records: dict[str, list], mode: str, machines: set[str], lo: str,
                hi: str, label: str = "") -> list[Idle]:
    """P8 (the investigator's u2 NT-breaking-late step 5): on each of *machines*, which selected nothing from *lo* to
    *hi*, the agent's cycles (one mark each) whose lines straddle that span -- a line at or before *lo*, the next at or
    after *hi*, none between -- each with the agent's calls in its loop between the two lines' rows, where it was
    meanwhile.  PostProcessor's cycle on aipanda081 printed "preparing tasks to be finished" at 04:02:41 and its next
    line, "getting tasks to be finished", at 04:12:44: it was in prepareTasksToBeFinished_JEDI while six other machines
    selected.  *records*: the agent's lines by log, as :func:`selections` reads them; *label*: the row's, where the
    cycle's lines print one -- a cycle for another label is not the one that would have selected the row."""
    fn = R.function(sel.rel, sel.agent_name.split(".")[-1], within=[sel.call])
    stmt = R.statement_at(fn.node, sel.call)
    loop = next((a for a in pathcond.ancestors(stmt) if isinstance(a, (ast.For, ast.AsyncFor, ast.While))), None) if stmt else None
    if loop is None:
        return []
    rows = [*crows.before, *crows.after]
    printing = {id(n) for r in fn.rows for n in ast.walk(r.stmt)}
    by_mark: dict[tuple[str, str], list] = {}
    for log, recs in records.items():
        for r in recs:
            if r.machine in machines and r.token and not unmarked(log, r.token, r.text):
                by_mark.setdefault((r.machine, r.token), []).append(r)
    out = []
    for (machine, mark), recs in sorted(by_mark.items()):
        recs.sort(key=lambda r: r.stamp)
        before = [r for r in recs if r.stamp[:23] <= lo[:23]]
        after = [r for r in recs if r.stamp[:23] >= hi[:23]]
        if not before or not after or any(lo[:23] < r.stamp[:23] < hi[:23] for r in recs):
            continue
        row1 = next((row for row in rows if row.matches(before[-1], mode)[0]), None)
        row2 = next((row for row in rows if row.matches(after[0], mode)[0]), None)
        if row1 is None or row2 is None or row1.line >= row2.line:
            continue
        held = ""
        for r in before:
            for row in crows.before:
                if crows.label and row.matches(r, mode)[0] and crows.label in (got := row.capture(r)):
                    held = got[crows.label].strip()
        if label and held and held != label:
            continue
        calls = tuple(dict.fromkeys(
            name for n in ast.walk(loop) if isinstance(n, ast.Call) and row1.line < n.lineno < row2.line
            and id(n) not in printing and (name := _called(n)) and name not in _BUILTINS))
        if calls:
            out.append(Idle(machine, mark, before[-1].stamp, row1.line, after[0].stamp, row2.line, calls))
    return out


_BUILTINS = frozenset(dir(__import__("builtins")))


def _called(call: ast.Call) -> str:
    """The name a call calls: an attribute's, or a bare name's."""
    f = call.func
    return f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else ""


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


def lock_writes(func: ast.AST) -> list[ast.stmt]:
    """P55 (out/gm/qk-design.md, the lock timeline's draft): the statements of *func* that run a SQL text setting the
    lock column (``UPDATE ... SET lockedBy=``), read off the string bound to the name the execute's first argument
    starts with -- lockJobOutputReport's ``self.cur.execute(sqlUL + comment, varMap)``."""
    texts: dict[str, str] = {}
    for n in ast.walk(func):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            parts = [c.value for c in ast.walk(n.value) if isinstance(c, ast.Constant) and isinstance(c.value, str)]
            if parts:
                texts[n.targets[0].id] = " ".join(parts)
    out: list[ast.stmt] = []
    for n in ast.walk(func):
        if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and getattr(n.value.func, "attr", "") == "execute"
                and n.value.args):
            continue
        first = n.value.args[0]
        name = first.left if isinstance(first, ast.BinOp) else first
        text = texts.get(name.id, "") if isinstance(name, ast.Name) else ""
        if re.search(r"\bUPDATE\b", text, re.I) and re.search(r"\bSET\b[^;]*\blockedBy\s*=", text, re.I):
            out.append(n)
    return out


def lock_before(mover: str, hops: int = 3) -> tuple[str, str, int] | None:
    """P55 (out/gm/qk-design.md, the lock timeline's draft): the lock function whose result guards a call on the way
    into *mover* (``file::qualname``): up the map's calls into it (:func:`sources.calls_into`), at most *hops*, a caller
    whose call into the chain stands under a guard (``pathcond.arm_condition``, early exits too) naming a local bound to
    a call of a database module's function with a lock write (:func:`lock_writes`).  add_main's AdderThread.run calls
    AdderGen.run at 150 only past ``if not got_lock: continue``, ``got_lock = taskBuffer.lockJobOutputReport(...)``.
    ``(the lock function, the caller, the call line)``, or None."""
    frontier, seen = [mover], {mover}
    for _ in range(hops):
        found: list[str] = []
        for owner in frontier:
            rel, _, qual = owner.partition("::")
            for caller, lines in SR.calls_into(rel, qual):
                if caller in seen:
                    continue
                seen.add(caller)
                found.append(caller)
                crel, _, cqual = caller.partition("::")
                try:
                    fn = R.function(crel, cqual.split(".")[-1], list(lines))
                except (LookupError, ValueError, SyntaxError):
                    continue
                for line in lines:
                    stmt = R.statement_at(fn.node, line)
                    if stmt is None:
                        continue
                    for guard in pathcond.arm_condition(stmt):
                        try:
                            tree = ast.parse(guard, mode="eval").body
                        except SyntaxError:
                            continue
                        for name in sorted({x.id for x in ast.walk(tree) if isinstance(x, ast.Name)}):
                            for b in pathcond.bindings(fn.node, name):
                                value = getattr(b, "value", None)
                                if not isinstance(value, ast.Call):
                                    continue
                                callee = str(getattr(value.func, "attr", None) or getattr(value.func, "id", "") or "")
                                for o in _defs_named(callee):
                                    if "/db_proxy_mods/" not in o:
                                        continue
                                    lrel, _, lqual = o.partition("::")
                                    try:
                                        lf = R.function(lrel, lqual.split(".")[-1])
                                    except (LookupError, ValueError, SyntaxError):
                                        continue
                                    if lock_writes(lf.node):
                                        return o, caller, line
        frontier = found
    return None


def take_rows(func: ast.AST, rows: list[int]) -> set[int]:
    """P55: of the print lines *rows* of *func*, those every way to which runs a lock write (:func:`lock_writes`) first:
    a statement holding the row, or holding a statement that does, sits in the same block as a lock write and after it.
    A call printing one took the lock; one printing none did not -- lockJobOutputReport's 'successfully locked record
    by ...' after the UPDATE in its loop, not its 'done' after the loop."""
    writes = lock_writes(func)
    out: set[int] = set()
    for n in ast.walk(func):
        for name in ("body", "orelse", "finalbody"):
            block = getattr(n, name, None)
            if not isinstance(block, list):
                continue
            for i, s in enumerate(block):
                if not any(w is s for w in writes):
                    continue
                for later in block[i + 1:]:
                    out.update(line for line in rows if any(getattr(x, "lineno", None) == line and isinstance(x, ast.stmt)
                                                            for x in ast.walk(later)))
    return out


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


def waker_readers(key_froms: list, read_sites: list, row_tables: set[str], owner: str, subject: str,
                   lines: set[int], rows_kind: str = "rows") -> tuple[list[tuple[Any, str]], list] | None:
    """A2: the selections whose rows alone give *owner*'s writes of *subject* at *lines* the key they take the row by,
    each with the column the key is read off (``(read site, column)``), and the inserts the map's ``wakers`` of them
    name -- getTasksToExecCommand_JEDI's 4483 by 4339's ``prodsys_comm.comm_task``, sendCommandTaskPanda's 4038.  None
    where a write takes its key from anything else, a selection reads a table holding rows of the subject's class
    (*row_tables*, lower case), or no waker inserts.  *owner* and the KeyFroms' owners compare as ``file::function``."""

    def bare(o: str) -> str:
        return R.named(o)

    sites = {(r.owner, r.anchor.line_start): r for r in read_sites}
    readers: list[tuple[Any, str]] = []
    for kf in key_froms:
        if bare(kf.owner) != bare(owner) or not set(kf.writes.get(subject, [])) & lines:
            continue
        for s in kf.sources:
            site = sites.get((s.site_owner, s.site_line)) if s.kind == rows_kind else None
            if site is None:
                return None
            readers.append((site, s.column))
    if not readers or any({t.lower() for t in r.tables} & row_tables for r, _ in readers):
        return None
    wakers = _inserting(readers)
    return (readers, wakers) if wakers else None


#: the wakers that make a row: an insert of a value the selection asks for, or a new row its terms take (the map's
#: ``born``, codemap-xliv's P45 part: Job_Output_Report's INSERTs for listJobOutputReport@2401)
INSERTING = ("insert", "born")


def _inserting(readers: list[tuple[Any, str]]) -> list:
    """The inserting wakers of *readers*' selections, one per writer's line."""
    out: dict[tuple, Any] = {}
    for r, _ in readers:
        for w in getattr(r, "wakers", None) or []:
            if w.kind in INSERTING:
                out.setdefault((w.owner, w.line, w.value), w)
    return list(out.values())


def crossed_readers(key_froms: list, read_sites: list, reach_graph: list, row_tables: set[str], owner: str, subject: str,
                    lines: set[int], kept_kind: str, ident: set[str], depth: int = 2) -> tuple[list[tuple[Any, str]], list, list[str]] | None:
    """P45 (the investigator's u10; A2 widened): where *owner*'s writes of *subject* at *lines* carry no key of their own
    -- no KeyFrom of them has a source, and one keeps the object it was handed (*kept_kind*; another may hand it to a
    helper, as P10 reads doGenerate's 1277) -- the key came with the object from its callers: the selections on another table (none of *row_tables*) whose rows the map's
    ``crosses_to`` hands to *owner* or a caller of it within *depth* calls (the reach graph), with the inserts their
    ``wakers`` name, and the callers reached.  A selection keyed by the object's identity (a key whose column is
    one of *ident*, lower case: peekJob's files by ``FileSpec.PandaID``) reads by the key, so gives none.  The Adder's parse_job_output_report keeps the job its AdderGen was handed;
    listJobOutputReport@2401's rows cross to process_job_report.  None where any write has a key of its own, no such
    selection is found, or none has an inserting waker."""

    def bare(o: str) -> str:
        return R.named(o)

    mine = [kf for kf in key_froms if bare(kf.owner) == bare(owner) and set(kf.writes.get(subject, [])) & lines]
    if (not mine or any(getattr(kf, "sources", None) for kf in mine)
            or not any(getattr(kf, "unread", "") == kept_kind for kf in mine)):
        return None
    callers: dict[str, set[str]] = {}
    for node in reach_graph:
        for c in getattr(node, "calls", None) or []:
            callers.setdefault(bare(c.to), set()).add(bare(node.owner))
    reach = level = {bare(owner)}
    for _ in range(depth):
        level = {c for f in level for c in callers.get(f, ())} - reach
        reach = reach | level
    readers: list[tuple[Any, str]] = []
    via: set[str] = set()
    for r in read_sites:
        hit = {bare(c) for c in getattr(r, "crosses_to", None) or []} & reach
        if not hit or {t.lower() for t in getattr(r, "tables", None) or []} & row_tables:
            continue
        if {str(k).rpartition(".")[2].lower() for k in getattr(r, "keys", None) or []} & ident:
            continue
        readers.append((r, ""))
        via |= hit
    wakers = _inserting(readers)
    return (readers, wakers, sorted(via)) if readers and wakers else None


def cap_rows(limits: list[str]) -> int | None:
    """The most rows a statement returns by the map's ``limits`` (``< 1000``: 999, ``<= 50``: 50), the least where it
    has several; None where none is a number (``< nTasks``)."""
    caps = []
    for limit in limits or []:
        m = re.fullmatch(r"\s*(<=?)\s*(\d+)\s*", limit)
        if m:
            caps.append(int(m.group(2)) - (m.group(1) == "<"))
    return min(caps) if caps else None


@dataclass(frozen=True)
class CapTake:
    """M3(a)-cap: the selection *take* that took the row, the first after the turn to return fewer rows than *cap*, and
    those of the row's label from the turn to it: *full* returned the cap, *empty* none.  Where the lines name no label,
    the row's is the *place*-th (from 0) of each cycle's *width* selections -- the only place *take*'s cycle returned
    rows at -- and *others* are the selections of cycles of another width in that time (another process's, another
    label's); *place* is -1 where the lines name the label."""

    take: Pick
    cap: int
    full: tuple[Pick, ...]
    empty: tuple[Pick, ...]
    place: int = -1
    width: int = 0
    others: tuple[Pick, ...] = ()


def _count(p: Pick) -> int | None:
    return int(p.count) if p.count.isdigit() else None


def cap_take(picks: dict[str, list[Pick]], machine: str, first: str, turn: datetime.datetime, cap: int,
             label: str) -> tuple[CapTake | None, str]:
    """M3(a)-cap: which selection took the row the pass on *machine* beginning at *first* moved on, where the selecting
    statement returns at most *cap* rows and began to return the row at *turn*: the machine's last selection of the
    row's label before *first* that returned some rows, fewer than *cap*; every one of the label's from *turn* to it, on
    any machine, returned *cap* rows or none.  With *label* the selections are the label's; without, the row's place in
    a cycle is the one place the take's cycle returned rows at.  ``(CapTake, "")``, or None with why not.

    Without *label*, where the machine's selections that returned rows come from cycles of more than one width (more
    than one process selects there), the take is not the last of them all (bundle C, C-3 (b); out/gm/qk-design.md,
    tmaeno's choice 2026-11-17): each width is read in turn as the row's, the take being the last of that width's, and
    a width with a selection of the row's place below the cap that did not take the row is not the row's.  Where one
    width is left it is read; where none or more than one, the take is not told.  NT-registered-late's aipanda081: the
    1-label process's cycles are not the row's (aipanda086's one-row selection did not take it), and the 4-label
    process's last selection before the pass returned the cap, 999."""
    cycles: dict[tuple[str, str], list[Pick]] = {}
    for qs in picks.values():
        for q in qs:
            cycles.setdefault((q.machine, q.mark), []).append(q)
    for qs in cycles.values():
        qs.sort(key=lambda q: q.at)

    def began(q: Pick) -> datetime.datetime:
        return when(q.after or q.at)

    def width_of(q: Pick) -> int:
        return len(cycles[(q.machine, q.mark)])

    hi = when(first)
    took = [q for q in picks.get(machine, []) if (not label or q.label == label) and began(q) >= turn
            and when(q.at) <= hi and (_count(q) or 0) > 0]
    if not took:
        return None, f"no selection on {machine.split('.')[0]} from {turn:%H:%M:%S} to {first[11:19]} returned rows"
    widths = [] if label else sorted({width_of(q) for q in took}, reverse=True)
    if len(widths) <= 1:
        found, why, _ = _cap_read(took[-1], picks, cycles, machine, turn, cap, label, began, "")
        return found, why
    reads = {w: _cap_read([q for q in took if width_of(q) == w][-1], picks, cycles, machine, turn, cap, label, began,
                          " of them") for w in widths}
    left = [w for w in widths if not reads[w][2]]
    ruled = "; ".join(f"cycles of {w} selection(s) are not: {reads[w][1]}" for w in widths if w not in left)
    if len(left) == 1:
        w = left[0]
        found, why, _ = reads[w]
        return found, ("" if found is not None else f"cycles of {w} selection(s) are the row's ({ruled}): {why}")
    if not left:
        return None, f"no width of the cycles on {machine.split('.')[0]} that returned rows is the row's ({ruled})"
    return None, (f"cycles of {' and '.join(str(w) for w in left)} selection(s) on {machine.split('.')[0]} returned rows "
                  "before the pass, and none of them is shown not to be the row's: which one's selection took it is "
                  "not told" + (f" ({ruled})" if ruled else ""))


def _cap_read(take: Pick, picks: dict[str, list[Pick]], cycles: dict[tuple[str, str], list[Pick]], machine: str,
              turn: datetime.datetime, cap: int, label: str, began, of: str) -> tuple[CapTake | None, str, bool]:
    """:func:`cap_take` read on *take* as the last selection of the row's before the pass: ``(CapTake or None, why
    not, whether a selection of the row's place below the cap did not take it)`` -- the last only rules the place out
    as the row's.  *of*: " of them" where *take* is read as one width's among others."""
    n = _count(take)
    if n is None or n >= cap:
        return None, (f"the last selection{of} on {machine.split('.')[0]} before the pass returned {take.count} row(s), "
                      f"not fewer than {cap}"), False
    place, width = -1, 0
    others: tuple[Pick, ...] = ()
    if label:
        mine = [q for qs in picks.values() for q in qs if q.label == label]
    else:
        cycle = cycles[(take.machine, take.mark)]
        rows_at = [i for i, q in enumerate(cycle) if (_count(q) or 0) > 0]
        if len(rows_at) != 1:
            return None, f"the take's cycle {take.mark} returned rows at {len(rows_at)} of its {len(cycle)} selections", False
        place, width = rows_at[0], len(cycle)
        mine = [qs[place] for qs in cycles.values() if len(qs) == width]
        others = tuple(q for qs in cycles.values() if len(qs) != width for q in qs
                       if began(q) >= turn and when(q.at) < began(take))
    before = sorted((q for q in mine if began(q) >= turn and when(q.at) < began(take)), key=lambda q: q.at)
    odd = [q for q in before if _count(q) not in (0, cap)]
    if odd:
        q = odd[0]
        return None, (f"{q.machine.split('.')[0]}'s selection at {q.at[11:23]} returned {q.count} row(s), fewer than "
                      f"{cap}, and did not take it"), True
    return CapTake(take, cap, tuple(q for q in before if _count(q) == cap), tuple(q for q in before if _count(q) == 0),
                   place, width, others), "", False


def _defs_named(name: str) -> list[str]:
    """Every corpus def named *name* the map holds an entry of (``rel::qualname``)."""
    _params, entries = SR._callers()
    return sorted({o for o, _line in entries if o.split("::")[-1].split(".")[-1] == name})


def when(stamp: str) -> datetime.datetime:
    return W.when(stamp)
