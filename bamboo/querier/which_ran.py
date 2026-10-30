"""Step 2 of the loop: which of the candidate sites ran, from one run's records.

Every candidate comes out as one of four, with the rule that put it there:

* **excluded** --
  a row every path through the site prints is missing from the run (rule 4;
  only when the run printed something at that level, rule 8);
  a row that path can no longer reach after its ``return`` is in the run;
  a printed row's ``arms`` leave the site out (outside a shared loop);
  the site sits after the record that already showed the observed value, or
  before the one that still showed the old value (rule 3) -- for a run of
  another log that ran inside a call the question's run made, where that
  call sits (:func:`inside_call`);
  the site is reached only when the value is already the observed one
  (rule 6, solved from the guards on the way to it, the caller's included);
* **positive** -- every row that must print does, one of them says enough to
  prove it, and a hole of the subject in them holds the observed value; or,
  that failing, the rows printed around the ``if`` chain it sits under leave no
  other way through (P, :mod:`bamboo.querier.printed`) -- read once more after
  every other verdict, where a call changes the subject to the observed value
  only through a writer of it not excluded (``printed.Reaching``);
* **unexamined** -- no run to read it against: no log to hand, or the log does
  not cover the time the question is about;
* **kept** -- read, and nothing above decided it.

The choice: one positive candidate; otherwise, when the list is complete and
every other one is excluded, the one left (elimination).  ``--decide strict``
takes neither while anything is unexamined.

P is read for every candidate, whatever else decided it: a candidate another
rule excluded that P proves is printed as a ``P twin`` note -- one of the two
rules is wrong there -- and one no path fits as ``P inconsistent``.
"""
from __future__ import annotations

import ast
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from bamboo.codemap.panda import pathcond  # noqa: E402
from bamboo.querier import evaluate as E  # noqa: E402
from bamboo.querier import printed as PR  # noqa: E402
from bamboo.querier import rows as R  # noqa: E402
from bamboo.querier.records import Run  # noqa: E402

#: ``WHICH_RUN_BY_ROWS=0``: a candidate with no run of its own takes the key's first run in the window, whatever
#: function it is.  With several runs there -- a DB proxy log, whose token is the method --
#: that is another function's run, and its rows read as missing
RUN_BY_ROWS = os.environ.get("WHICH_RUN_BY_ROWS", "1") != "0"
#: ``WHICH_RUN_ROW_EXIT=0``: no rule from a printed row the call leaves right after.
#: On, a row printed once whose path ends at a return/raise rules out
#: every site below that exit -- the arms (path conditions) do not read the jump
ROW_EXIT = os.environ.get("WHICH_RUN_ROW_EXIT", "1") != "0"
#: ``WHICH_RUN_JOIN_CALLS=0``: runs of the key in several logs stay several runs, and none is the question's
#: (2026-10-14), even where one ran inside a call the other made (the adder's updateJobs and DBProxy's archiveJob)
JOIN_CALLS = os.environ.get("WHICH_RUN_JOIN_CALLS", "1") != "0"
#: ``WHICH_RUN_ROWS_AROUND=0``: a line is proven only by a row on a path through it (implying rows, P), not by two
#: rows printed in a row that only a path through it prints (:func:`rows_around`)
ROWS_AROUND = os.environ.get("WHICH_RUN_ROWS_AROUND", "1") != "0"


def run_holding(c: "Candidate", found: list[Run], mode: str) -> Run | None:
    """Of the key's runs in the window, the one where most of the candidate's rows are proven; None when there
    are several and none holds one.  A single run is taken as it is, as before."""
    if len(found) == 1 or not RUN_BY_ROWS:
        return found[0]
    fn = R.function(c.rel, c.func, c.lines)
    counts = [sum(1 for row in fn.rows if R.seen(fn, r, mode).proven(row)) for r in found]
    best = max(counts)
    return found[counts.index(best)] if best else None


@dataclass
class Candidate:
    rel: str
    func: str
    lines: list[int]
    log_files: list[str] = field(default_factory=list)
    run: Run | None = None  # set when the caller already knows which run it belongs to
    note: str = ""

    @property
    def owner(self) -> str:
        return f"{self.rel}::{self.func}"


@dataclass
class Verdict:
    candidate: Candidate
    line: int
    kind: str  # excluded | positive | unexamined | kept
    reasons: list[str] = field(default_factory=list)

    def describe(self) -> str:
        return f"{self.kind:10s} {Path(self.candidate.rel).name}::{self.candidate.func}:{self.line}  " + "; ".join(self.reasons)


@dataclass
class Judgement:
    verdicts: list[Verdict]
    chosen: Verdict | None
    how: str
    bracket: tuple[int, int] | None
    notes: list[str] = field(default_factory=list)
    #: each candidate's run, by owner, as the verdicts read it (None where it had none)
    runs: dict[str, Run | None] = field(default_factory=dict)
    #: the run the value changed in where the judgement found it (:func:`inside_call`), not the caller
    question: Run | None = None

    def count(self, kind: str) -> int:
        return sum(1 for v in self.verdicts if v.kind == kind)

    def report(self, run: Run | None = None) -> str:
        out = []
        run = run if run is not None else self.question
        if self.bracket and run is not None:
            a, b = self.bracket
            out.append(f"bracket: {run.records[a].span} -> {run.records[b].span}")
        out.extend(self.notes)
        for kind in ("positive", "kept", "excluded", "unexamined"):
            for v in self.verdicts:
                if v.kind == kind:
                    out.append("  " + v.describe())
        tally = ", ".join(f"{k} {self.count(k)}" for k in ("positive", "kept", "excluded", "unexamined"))
        best: dict[str, str] = {}
        order = ["positive", "kept", "unexamined", "excluded"]
        for v in self.verdicts:
            k = v.candidate.owner
            best[k] = min(best.get(k, "excluded"), v.kind, key=order.index)
        tally += "  | functions: " + ", ".join(f"{k} {list(best.values()).count(k)}" for k in order)
        pick = f"{Path(self.chosen.candidate.rel).name}::{self.chosen.candidate.func}:{self.chosen.line}" if self.chosen else "none"
        out.append(f"=> chosen: {pick} ({self.how})   [{tally}]")
        return "\n".join(out)


def subject_values(seens: list[R.Seen], attr: str) -> list[tuple[int, str, str]]:
    """``(record index, value, row)`` for every printed hole that spells ``<x>.<attr>``."""
    found = []
    for sn in seens:
        for row in sn.fn.rows:
            for i, _, _ in sn.hits.get(row.line, ()):
                for expr, value in row.capture(sn.run.records[i]).items():
                    if re.fullmatch(rf"[\w.]+\.{re.escape(attr)}", expr):
                        found.append((i, value.strip(), f"{sn.fn.name}:{row.line}"))
    return sorted(set(found))


def class_of(fn: R.Function) -> str:
    return next((a.name for a in pathcond.ancestors(fn.node) if isinstance(a, ast.ClassDef)), "")


def stable_holes(seens: list[R.Seen], env: E.Env, cls: str | None = None, local_to: str = "") -> dict[str, str]:
    """Put in *env* every printed hole that shows one value throughout its run.

    Only what means the same thing wherever it is read: an expression about
    a row (``self.job.prodSourceLabel``, ``isJumboJob(self.job)``), and --
    given *cls* -- an attribute of ``self`` printed by a method of that class.
    A bare local is the function's own: ``tmp_src_ddm`` printed by
    ``execute`` says nothing about ``update_outputs``'s -- so it is put only
    for *local_to* (``module::function``), by that function's own rows, for a
    question about that function (rule 17).
    """
    seen_values: dict[str, set[str]] = {}
    where: dict[str, str] = {}
    for sn in R.claim(seens):
        own = class_of(sn.fn)
        for row in sn.fn.rows:
            for i, _, _ in sn.hits.get(row.line, ()):
                for expr, value in row.capture(sn.run.records[i]).items():
                    about_row = "<" in env.canonical(expr)
                    of_class = cls is not None and own == cls and expr.startswith("self.")
                    own_local = (
                        bool(local_to) and sn.fn.owner == local_to and not expr.startswith("self.")
                        and R.reads_bound_once(sn.fn.node, expr)
                    )
                    if not (about_row or of_class or own_local):
                        continue
                    seen_values.setdefault(expr, set()).add(value.strip())
                    where.setdefault(expr, f"hole of {sn.fn.name}:{row.line} at {sn.run.records[i].span}")
    put = {}
    for expr, values in seen_values.items():
        if len(values) == 1:
            try:
                ast.parse(expr, mode="eval")
            except SyntaxError:
                continue
            env.put_printed(expr, next(iter(values)), where[expr])
            put[expr] = next(iter(values))
    return put


def bracket_of(values: list[tuple[int, str, str]], observed: str) -> tuple[int, int] | None:
    """The last record still showing another value, and the first after it showing *observed*."""
    for (i, v, _), (j, w, _) in zip(values, values[1:], strict=False):
        if v != observed and w == observed:
            return i, j
    return None


def inside_call(caller: R.Seen, inner: Run, site: str) -> tuple[int, int] | None:
    """``(i, j)``: the records of *caller*'s run printed just before and just after the call at *site* (``file:line``)
    whose times hold every record of *inner*, or None.

    The records are the call's ``must_rows`` in the caller, so each pair is one pass through the call: the last
    record before it, and the first after.  A call with no print every path runs on one side -- the next statement
    calls too, or the prints after it are in arms -- is placed by the run instead (:func:`_between_records`).  A log
    line names no process, so this says only that *inner* ran while the caller was inside that call, on the machine
    both name -- the caller of :func:`judge` checks that the call can reach *inner*'s function.
    """
    rel, _, line = site.rpartition(":")
    if rel != caller.fn.rel or not line.isdigit():
        return None
    stmt = R.statement_at(caller.fn.node, int(line))
    if stmt is None:
        return None
    before, after = R.must_rows(caller.fn.node, stmt)
    if not any(caller.fn.row_at(s) for s in before) or not any(caller.fn.row_at(s) for s in after):
        return _between_records(caller, inner, stmt)

    def printed(stmts: list[ast.stmt]) -> list[int]:
        return sorted(i for s in stmts if (row := caller.fn.row_at(s)) is not None for i in caller.unambiguous.get(row.line, ()))

    starts, ends = printed(before), printed(after)
    records = caller.run.records
    for j in ends:
        i = max((x for x in starts if x < j), default=None)
        if i is not None and records[i].stamp <= inner.first and inner.last <= records[j].stamp:
            return i, j
    return None


def _between_records(caller: R.Seen, inner: Run, stmt: ast.stmt) -> tuple[int, int] | None:
    """``(i, j)`` for a call with no print every path runs on one side of it: the caller's records just before and
    just after *inner*, two in a row of the function's (none of its rows read between them), where the call's
    statement is on a path from the one's print to the other's that prints nothing else (``rows.runs_between``), or
    None.

    ``updateTaskStatusByContFeeder_JEDI`` ends an arm of feed_contents_to_tasks and the next print is in another arm,
    so no print every path runs follows it; the run's "set task_status=..." before it and its next record after hold
    the JediDBProxy run between them."""
    node, records = caller.fn.node, caller.run.records
    placed = sorted((x, line) for line, xs in caller.unambiguous.items() for x in xs)
    before = [p for p in placed if records[p[0]].stamp <= inner.first]
    after = [p for p in placed if inner.last <= records[p[0]].stamp]
    if not before or not after or before[-1][0] >= after[0][0]:
        return None
    (i, first_line), (j, last_line) = before[-1], after[0]
    # a row of the function read between the two: it printed there, so the two are not a pass's
    if any(i < x < j for found in getattr(caller, "hits", {}).values() for x, *_ in found):
        return None
    prints = {id(s) for s in R.own_nodes(node) if isinstance(s, ast.stmt) and R.is_print(s) and caller.fn.row_at(s) is not None}
    first, last = R.statement_at(node, first_line), R.statement_at(node, last_line)
    if first is None or last is None or id(first) not in prints or id(last) not in prints:
        return None
    return (i, j) if R.runs_between(node, first, last, stmt, prints) else None


def rows_around(fn: R.Function, sn: R.Seen, stmt: ast.stmt) -> tuple[int, int] | None:
    """``(first, last)``: two of *fn*'s rows the run printed in a row -- no row of it read between them -- that only a
    path through *stmt* prints so (``rows.passes_through``), or None.  A line no row is on every path through (a call
    of a function that prints nothing) is shown run by the rows around it: ``TaskRefinerThread.runImpl``'s 205 and 563
    around 211, ``extractCommon``.  A row read but not placed (``Seen.hits`` beyond ``unambiguous``) between the two
    breaks the pair, as in :func:`_between_records`."""
    node = fn.node
    placed = sorted((x, line) for line, xs in sn.unambiguous.items() for x in xs)
    read = sorted(x for found in getattr(sn, "hits", {}).values() for x, *_ in found)
    prints = {id(s) for s in R.own_nodes(node) if isinstance(s, ast.stmt) and R.is_print(s) and fn.row_at(s) is not None}
    for (i, a), (j, b) in zip(placed, placed[1:], strict=False):
        if any(i < x < j for x in read):
            continue
        first, last = R.statement_at(node, a), R.statement_at(node, b)
        if first is None or last is None or id(first) not in prints or id(last) not in prints:
            continue
        if R.passes_through(node, first, last, stmt, prints):
            return a, b
    return None


def held_in_calls(
    inner: Run, pairs: list[tuple[Candidate, R.Seen]], called_at: Callable[[R.Function, Candidate], list[str]]
) -> tuple[int, int, str] | None:
    """``(i, j, calls)``: where *inner* ran inside the calls of an outer run's functions, for each (candidate of
    *inner*, function seen in the outer run) in *pairs* -- the calls by *called_at*, each placed by :func:`inside_call`
    -- or None.

    The records cannot tell apart calls between the same two prints (one in each arm of an ``if``), so every call
    whose pass holds *inner* is named (``"a.py:9 or a.py:11"``), and (i, j) is the span of their passes: *inner* sits
    somewhere in it."""
    held = [(site, at) for c, sn in pairs for site in called_at(sn.fn, c) if (at := inside_call(sn, inner, site))]
    if not held:
        return None
    return min(at[0] for _, at in held), max(at[1] for _, at in held), " or ".join(dict.fromkeys(site for site, _ in held))


def callers_guards(fn: R.Function) -> list[tuple[str, str]]:
    """``(guard, where)`` on the way to every ``self.<fn>()`` call in the same class."""
    tree = R.tree(fn.rel)
    cls = next((a for a in pathcond.ancestors(fn.node) if isinstance(a, ast.ClassDef)), None)
    if cls is None:
        return []
    out = []
    for node in ast.walk(cls):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == fn.name
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
        ):
            caller = pathcond.enclosing_function(node)
            for test in pathcond.path_condition(node):
                out.append((test, f"{caller.name if caller else '?'}:{node.lineno}"))
    del tree
    return out


EMPTY = {"{}", "[]", "()", "set()", ""}


def callee_values(fn: R.Function, run: Run, mode: str) -> dict[str, tuple[str, str]]:
    """``param -> (printed value, where)`` from the one ``self.<fn>(...)`` call in its class.

    A parameter takes the value of the argument in its position.  The value
    is the caller's own printed hole of that argument, when the caller does
    not bind it again between the print and the call (outside part 3).
    """
    cls = next((a for a in pathcond.ancestors(fn.node) if isinstance(a, ast.ClassDef)), None)
    if cls is None:
        return {}
    calls = [
        n
        for n in ast.walk(cls)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == fn.name
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "self"
    ]
    if len(calls) != 1:
        return {}
    call = calls[0]
    caller = pathcond.enclosing_function(call)
    if caller is None:
        return {}
    params = [a.arg for a in fn.node.args.args][1:]
    g = R.function(fn.rel, caller.name, within=[caller.lineno])
    sg = R.seen(g, run, mode)
    out = {}
    for position, arg in enumerate(call.args):
        if position >= len(params) or not isinstance(arg, ast.Name):
            continue
        for row in g.rows:
            if row.line >= call.lineno or R._binding_sites(caller, arg.id, call.lineno) and any(
                b.lineno > row.line for b in R._binding_sites(caller, arg.id, call.lineno)
            ):
                continue
            for i, _, _ in sg.hits.get(row.line, ()):
                value = row.capture(run.records[i]).get(arg.id)
                if value is not None:
                    out[params[position]] = (value.strip(), f"{caller.name}:{row.line} at {run.records[i].span}, passed at {call.lineno}")
    return out


def empty_loops(fn: R.Function, stmt: ast.stmt, callee: dict[str, tuple[str, str]]) -> str:
    for it in R.iterables_over(fn.node, stmt):
        base = it
        while isinstance(base, (ast.Call, ast.Attribute)):
            base = base.func if isinstance(base, ast.Call) else base.value
        if isinstance(base, ast.Name) and base.id in callee and callee[base.id][0] in EMPTY:
            value, where = callee[base.id]
            return f"inside 'for ... in {ast.unparse(it)}', and {base.id} printed {value!r} ({where})"
    return ""


#: a loop's rows read per element; P_LOOP_ELEMENT=0 reads them per function
LOOP_ELEMENT = os.environ.get("P_LOOP_ELEMENT", "1") != "0"


def other_elements(fn: R.Function, stmt: ast.stmt, row: R.Row, sn: R.Seen, env: E.Env | None) -> str:
    """Why *row*, which the run printed, did not print for the loop element *stmt* is asked about; '' if it may have.

    A row inside the same ``for`` as *stmt* prints once per element.  When *env* binds the loop's variable (H7 binds
    the make call's name to the object the outside says is missing) and every record of the row holds another value
    of it, the row printed for the other elements only (``register_dataset {name}`` at 861 printed for
    the two top-level names and not for the ``_sub``, so the make call under it did not run for the ``_sub``).  Not
    when the body rebinds the variable, nor when a record's value cannot be read.
    """
    if env is None:
        return ""
    for loop in pathcond.ancestors(stmt):
        if loop is fn.node:
            break
        if not isinstance(loop, (ast.For, ast.AsyncFor)) or not isinstance(loop.target, ast.Name):
            continue
        var = loop.target.id
        if env.canonical(var) not in env.values or not any(n is row.stmt for b in loop.body for n in ast.walk(b)):
            continue
        if any(isinstance(n, ast.Name) and n.id == var and isinstance(n.ctx, ast.Store) for b in loop.body for n in ast.walk(b)):
            continue
        values = []
        for i, _, _ in sn.hits.get(row.line, ()):
            value = row.capture(sn.run.records[i]).get(var)
            if value is None:
                return ""
            values.append(value.strip())
        wanted = str(env.lookup(var))
        if values and wanted not in values:
            return f"row {row.line} printed only for other values of {var} ({len(values)} time(s)), not {wanted!r}"
    return ""


#: why rule 6 solves a guard with ``<x>.<attr>`` unknown
ASKED = "the value the question is about: whether the guard needs it is asked, not read"


def rewriter(fn: R.Function, stmt: ast.stmt, attr: str, observed: str, env: E.Env) -> str:
    """Why *stmt* can only be reached once ``<x>.<attr>`` already is *observed*, or ''.

    Each guard is solved with ``<x>.<attr>`` unknown: whether the guard needs the value is the question, and a value
    *env* holds for it -- a hole another run printed after the move (DBProxy's archiveJob ``start status=failed``
    inside the adder's call) -- would decide the term before the guard could, so it would never be settled.
    """
    guards = [(g, f"{fn.name}:{stmt.lineno}") for g in pathcond.path_condition(stmt)] + callers_guards(fn)
    for guard, where in guards:
        asked = set(re.findall(rf"(?<![\w.])[\w.]+\.{re.escape(attr)}(?!\w)", guard))
        hidden = E.Env(dict(env.values), dict(env.rows), env.table, {**env.stale, **{s: ASKED for x in asked for s in (x, env.canonical(x))}},
                       unread=env.unread, at=env.at)
        try:
            solved = E.solve(guard, True, hidden)
        except SyntaxError:
            continue
        for atom, truth in solved.settled.items():
            m = re.fullmatch(rf"([\w.]+\.{re.escape(attr)}) (==|!=) ['\"]{re.escape(observed)}['\"]", atom)
            if m and ((m[2] == "==" and truth) or (m[2] == "!=" and not truth)):
                used = {k: v for k, v in {**env.used, **hidden.used}.items() if k in guard and k not in asked}
                return f"rule 6: reached only when {m[1]} is already {observed!r} (guard {guard!r} at {where}; {used})"
    return ""


def judge(
    candidates: list[Candidate],
    logs: dict[str, list[Run]],
    *,
    mode: str = "outside",
    decide: str = "positive",
    window: tuple[str, str] | None = None,
    subject: str = "",
    observed: str = "",
    question_run: Run | None = None,
    env: E.Env | None = None,
    complete: bool = True,
    called_at: Callable[[R.Function, Candidate], list[str]] | None = None,
) -> Judgement:
    """*called_at* says where a function calls on, by the map, into a candidate's function: the ``file:line`` of every
    call from which it goes on there.  Given it, and no *question_run*, runs of the key in several logs are read as
    one where every run but one ran inside a call that one made (:func:`inside_call`, on the machine both name), and
    that one is the question's run.

    A candidate's must rows are read with where a traceback of its run shows its function left by an exception
    (:func:`rows.raised_in`)."""
    verdicts: list[Verdict] = []
    map_findings: set[str] = set()
    p_notes: list[str] = []
    seens: dict[str, R.Seen] = {}
    runs_of: dict[str, Run | None] = {}
    why_not: dict[str, str] = {}
    for c in candidates:
        run = c.run
        if run is None:
            to_hand = [f for f in c.log_files if f in logs]
            if not to_hand:
                why_not[c.owner] = f"no log to hand ({', '.join(c.log_files) or 'none recorded'})"
            # every log handed says why it holds no run of the candidate, in the candidate's order of its logs
            reasons: list[str] = []
            for name in to_hand:
                found = [r for r in logs[name] if window is None or (r.first[:19] <= window[1] and r.last[:19] >= window[0])]
                if found:
                    run = run_holding(c, found, mode)
                    if run is not None:
                        break
                    reasons.append(f"none of the key's {len(found)} runs in {name} at {window[0] if window else 'any time'} holds a row of it")
                    continue
                covering = logs[name][0].log if logs[name] else None
                if covering is not None and covering.records[0].machine:
                    # a recorded search says what it searched by machine (Log.holes), not by its first and last record
                    reasons.append(f"{name}: no run of the key at {window}" + (f"; not searched: {'; '.join(covering.holes)}" if covering.holes else ""))
                elif covering is not None and window is not None and not (
                    covering.covers[0][:19] <= window[1] and covering.covers[1][:19] >= window[0]
                ):
                    reasons.append(f"{name} covers {covering.covers[0]}..{covering.covers[1]}, not {window[0]}..{window[1]}")
                else:
                    reasons.append(f"{name}: no run of the key at {window}")
            if run is None and reasons:
                why_not[c.owner] = "; ".join(reasons)
        runs_of[c.owner] = run
        if run is not None:
            seens[c.owner] = R.seen(R.function(c.rel, c.func, c.lines), run, mode)
    # a record counts as printed only by the rows that read it best (rows.claim)
    seens = dict(zip(seens, R.claim(list(seens.values())), strict=False))

    # a run inside a call another run made is a part of that run: (outer run, record before, record after, calls)
    joined: dict[int, tuple[Run, int, int, str]] = {}
    join_notes: list[str] = []
    if question_run is None and called_at is not None and JOIN_CALLS and subject and observed:
        runs = list({id(r): r for r in runs_of.values() if r is not None}.values())
        for inner in runs:
            for outer in runs:
                if outer is inner or outer.log.name == inner.log.name or not outer.machine or outer.machine != inner.machine:
                    continue
                pairs = [(c, sn) for c in candidates if runs_of.get(c.owner) is inner for sn in seens.values() if sn.run is outer]
                if (found := held_in_calls(inner, pairs, called_at)) is not None:
                    joined[id(inner)] = (outer, *found)
                    break
        roots = [r for r in runs if id(r) not in joined]
        if joined and len(roots) == 1 and all(outer is roots[0] for outer, *_ in joined.values()):
            question_run = roots[0]
            for inner in runs:
                if id(inner) in joined:
                    outer, i, j, site = joined[id(inner)]
                    join_notes.append(
                        f"one run: {inner.log.name} {inner.first}..{inner.last} ran inside the call at {site} of the run in "
                        f"{outer.log.name} (between {outer.records[i].span} and {outer.records[j].span}), on {inner.machine}"
                    )
        else:
            joined = {}

    bracket = None
    if subject and observed and question_run is not None:
        in_run = [sn for sn in seens.values() if sn.run is question_run]
        bracket = bracket_of(subject_values(in_run, subject), observed)

    for c in candidates:
        run = runs_of[c.owner]
        if run is None:
            for line in c.lines:
                verdicts.append(Verdict(c, line, "unexamined", [why_not.get(c.owner, "no run")]))
            continue
        sn = seens[c.owner]
        fn = sn.fn
        present_once = sn.single_call()
        callee = callee_values(fn, run, mode)
        named, raised = R.raised_in(fn, run)
        for line in c.lines:
            stmt = R.statement_at(fn.node, line)
            v = Verdict(c, line, "kept")
            if stmt is None:
                v.reasons.append("no statement on this line")
                verdicts.append(v)
                continue
            before, after = R.must_rows(fn.node, stmt, raised if named else set())
            must = before + after
            absent, neutral, shown, proof, wrong_value, per_element = [], [], [], False, [], []
            for s in must:
                row = fn.row_at(s)
                if row is None:
                    continue
                if sn.present(row):
                    elsewhere = other_elements(fn, stmt, row, sn, env) if LOOP_ELEMENT else ""
                    if elsewhere and R.absence_trusted(row, run.log, mode)[0] and run.would_print(row.level):
                        absent.append(row.line)
                        per_element.append(elsewhere)
                        continue
                    shown.append(row.line)
                    proof = proof or sn.proven(row)
                    # a row after the write shows what it wrote; one before shows what it replaced
                    if subject and observed and s in after:
                        for i, _, _ in sn.hits[row.line]:
                            for expr, val in row.capture(run.records[i]).items():
                                if re.fullmatch(rf"[\w.]+\.{re.escape(subject)}", expr) and val.strip() != observed:
                                    wrong_value.append(f"{row.line} shows {expr}={val.strip()}")
                elif (trust := R.absence_trusted(row, run.log, mode))[0] and run.would_print(row.level):
                    absent.append(row.line)
                else:
                    neutral.append(row.line)
                    if trust[1] not in ("", "unreadable", R.ANOTHER_FILE):
                        map_findings.add(trust[1])
            if absent:
                v.kind = "excluded"
                v.reasons.append(f"rule 4: must row(s) {absent} not in the run")
                v.reasons.extend(per_element)
            if present_once:
                for s in R.cannot_rows(fn.node, stmt):
                    if sn.unambiguous.get(s.lineno):
                        v.kind = "excluded"
                        v.reasons.append(f"row {s.lineno} printed after the return this path takes")
                        break
                for row in fn.rows if ROW_EXIT else ():
                    if sn.unambiguous.get(row.line) and (end := R.beyond_exit(fn.node, row.stmt, stmt)) is not None:
                        v.kind = "excluded"
                        v.reasons.append(f"row {row.line} printed and the call leaves at {end.lineno} before {line}")
                        break
                for row in fn.rows:
                    if (
                        row.map_arms is not None
                        and line not in row.map_arms
                        and sn.unambiguous.get(row.line)
                        and not R.share_loop(row.stmt, stmt, fn.node)
                    ):
                        v.kind = "excluded"
                        v.reasons.append(f"row {row.line} printed and its arms {sorted(row.map_arms)} leave {line} out")
                        break
                if bracket is not None and run is question_run:
                    lo = max(
                        (i for r in fn.rows if r.line < line and not R.share_loop(r.stmt, stmt, fn.node) for i in sn.unambiguous.get(r.line, ())),
                        default=None,
                    )
                    hi = min(
                        (i for r in fn.rows if r.line > line and not R.share_loop(r.stmt, stmt, fn.node) for i in sn.unambiguous.get(r.line, ())),
                        default=None,
                    )
                    if lo is not None and lo >= bracket[1]:
                        v.kind = "excluded"
                        v.reasons.append(f"rule 3: runs after {run.records[lo].span}, when {subject} already showed {observed!r}")
                    if hi is not None and hi <= bracket[0]:
                        v.kind = "excluded"
                        v.reasons.append(f"rule 3: runs before {run.records[hi].span}, when {subject} still showed another value")
            # a run inside a call sits where that one pass through the call does, however often it printed a row
            if bracket is not None and (part := joined.get(id(run))) is not None:
                outer, lo, hi, site = part
                if lo >= bracket[1]:
                    v.kind = "excluded"
                    v.reasons.append(f"rule 3: runs inside the call at {site}, after {outer.records[lo].span}, when {subject} already showed {observed!r}")
                if hi <= bracket[0]:
                    v.kind = "excluded"
                    v.reasons.append(f"rule 3: runs inside the call at {site}, before {outer.records[hi].span}, when {subject} still showed another value")
            if subject and observed and env is not None:
                own = E.Env(dict(env.values), dict(env.rows), env.table, dict(env.stale), unread=env.unread, at=env.at)
                stable_holes([x for x in seens.values() if x.run is run], own, class_of(fn))
                why = rewriter(fn, stmt, subject, observed, own)
                if why:
                    v.kind = "excluded"
                    v.reasons.append(why)
            if wrong_value:
                v.kind = "excluded"
                v.reasons.append(f"value: {wrong_value}")
            if env is not None:
                for guard in pathcond.path_condition(stmt):
                    try:
                        if E.evaluate(guard, env) is False:
                            v.kind = "excluded"
                            used = {k: val for k, val in env.used.items() if k in guard}
                            v.reasons.append(f"step 5: guard {guard!r} is false ({used})")
                            break
                    except SyntaxError:
                        continue
            if loops_empty := empty_loops(fn, stmt, callee):
                v.kind = "excluded"
                v.reasons.append(loops_empty)
            proof = PR.prove(fn, sn, stmt)
            if v.kind == "kept":
                implying = [fn.row_at(s) for s in R.implying_rows(fn.node, stmt)]
                proven = [r.line for r in implying if r is not None and sn.proven(r)]
                if proven:
                    v.kind = "positive"
                    v.reasons.append(f"row(s) {proven} print only on a path through it")
                elif proof.kind == "positive":
                    v.kind = "positive"
                    v.reasons.append(proof.reason)
                elif ROWS_AROUND and (around := rows_around(fn, sn, stmt)) is not None:
                    v.kind = "positive"
                    v.reasons.append(f"rows {around[0]} and {around[1]} printed in a row, which only a path through it prints")
                else:
                    v.reasons.append(
                        f"must rows: printed {shown}, unreadable {neutral}; none prints only through it"
                        if must
                        else "no row every path through it prints"
                    )
            if named:
                v.reasons.append(
                    f"a traceback frame raised at {sorted(raised)}" if raised is not None
                    else f"a traceback frame of {fn.name} cannot be placed in the corpus: no row after {line} is required"
                )
            where = f"{Path(c.rel).name}::{c.func}:{line}"
            if proof.kind == "positive" and v.kind == "excluded":
                p_notes.append(f"P twin: {where} excluded ({'; '.join(v.reasons)}), but {proof.reason}")
            elif proof.kind == "inconsistent":
                p_notes.append(f"P inconsistent: {where}: {proof.reason}")
            verdicts.append(v)

    # P again on each kept candidate, with the other verdicts: a call changes the subject to the observed value only
    # where the map reaches, from it, a writer of that value not excluded (printed.Reaching)
    if subject and observed and called_at is not None:
        standing = [c for c in candidates if any(v.candidate is c and v.kind != "excluded" for v in verdicts)]
        for v in verdicts:
            c = v.candidate
            if v.kind != "kept" or c.owner not in seens or (stmt := R.statement_at(seens[c.owner].fn.node, v.line)) is None:
                continue
            sn = seens[c.owner]
            lines = {
                int(line) for other in standing if other.owner != c.owner
                for site in called_at(sn.fn, other) for file, _, line in [site.rpartition(":")] if file == c.rel
            }
            proof = PR.prove(sn.fn, sn, stmt, PR.Reaching(observed, frozenset(lines)))
            if proof.kind == "positive":
                v.kind = "positive"
                v.reasons.append(f"{proof.reason}; read with the other verdicts: a call reaching no writer of {observed!r} left standing does not write it")

    positives = [v for v in verdicts if v.kind == "positive"]
    left = [v for v in verdicts if v.kind in ("positive", "kept", "unexamined")]
    unexamined = [v for v in verdicts if v.kind == "unexamined"]
    in_run = [
        v
        for v in verdicts
        if v.kind in ("positive", "kept")
        and ((r := runs_of[v.candidate.owner]) is question_run or (r is not None and id(r) in joined))
    ]
    chosen, how = None, "undecided"
    if len(positives) == 1 and (decide != "strict" or not unexamined):
        chosen, how = positives[0], "positive evidence"
    elif len(positives) > 1:
        how = f"{len(positives)} positive"
    elif complete and len(left) == 1 and not unexamined:
        chosen, how = left[0], "elimination over a complete list"
    elif decide == "strict" and unexamined:
        how = f"strict: {len(unexamined)} unexamined"
    elif question_run is not None and bracket is not None and len(in_run) == 1:
        others = len({v.candidate.owner for v in unexamined})
        chosen = in_run[0]
        how = f"elimination within the run the value changed in; {others} function(s) never placed in it are not excluded"
    elif left:
        how = f"{len(left)} left: " + ", ".join(sorted({f"{v.candidate.func}:{v.line}" for v in left if v.kind != 'unexamined'}))
    notes = join_notes + [f"MAP: {m}" for m in sorted(map_findings)] + p_notes
    return Judgement(verdicts, chosen, how, bracket, notes, runs_of, question_run if joined else None)

