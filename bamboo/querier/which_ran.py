"""Step 2 of the loop: which of the candidate sites ran, from one run's records.

Every candidate comes out as one of four, with the rule that put it there:

* **excluded** --
  a row every path through the site prints is missing from the run (rule 4;
  only when the run printed something at that level, rule 8);
  a row is in the run that one call cannot print and run the site with (the
  map's ``flow.together``: below the return every way from the print ends at,
  in another arm no loop holds with it, in another arm of a pass of a loop the
  run went round once, ``rows.once_loops``);
  the site sits after the record that already showed the observed value, or
  before the one that still showed the old value, where the flow lets the
  two go one way only (rule 3, ``flow.runs_after``) -- for a run of
  another log that ran inside a call the question's run made, where that
  call sits (:func:`inside_call`);
  the site is reached only when the value is already the observed one
  (rule 6, solved from the guards on the way to it, the caller's included);
  of one pass's candidates left, one with the observed value already there
  before it (a stay) or one assigning another value (rule R,
  :mod:`bamboo.querier.changed`);
* **positive** -- every row that must print does, one of them says enough to
  prove it, and a hole of the subject in them holds the observed value; or,
  that failing, the rows printed around the ``if`` chain it sits under leave no
  other way through (P, :mod:`bamboo.querier.printed`) -- read once more after
  every other verdict, where a call changes the subject to the observed value
  only through a writer of it not excluded (``printed.Reaching``);
* **unexamined** -- no run to read it against: no log to hand, the log does
  not cover the time the question is about, or no pass of the candidate's
  function at the window holds the key (:func:`passes_at`);
* **kept** -- read, and nothing above decided it.

The choice: where rule R orders one pass's candidates, the last that changed the
value, if it ran, and none where more than one may have; otherwise one positive
candidate; otherwise, when the list is complete and
every other one is excluded, the one left (elimination) -- unexamined too, in
a window that holds the write itself (*holds_write*).  ``--decide strict``
takes neither while anything is unexamined.

P is read for every candidate, whatever else decided it: a candidate another
rule excluded that P proves is printed as a ``P twin`` note -- one of the two
rules is wrong there -- and one no path fits as ``P inconsistent``.
"""
from __future__ import annotations

import ast
import datetime
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bamboo.codemap import trace  # noqa: E402
from bamboo.codemap.calls import Target  # noqa: E402
from bamboo.codemap.panda import changes as MC  # noqa: E402
from bamboo.codemap.panda import evaluate as SE  # noqa: E402
from bamboo.codemap.panda import fills, flow, pathcond  # noqa: E402
from bamboo.querier import asked, row_names  # noqa: E402
from bamboo.querier import changed as CH  # noqa: E402
from bamboo.querier import evaluate as E  # noqa: E402
from bamboo.querier import printed as PR  # noqa: E402
from bamboo.querier import rows as R  # noqa: E402
from bamboo.querier import windows as W  # noqa: E402
from bamboo.querier.recorded import EDGE  # noqa: E402
from bamboo.querier.records import Run  # noqa: E402

#: ``WHICH_RUN_BY_ROWS=0``: a candidate with no run of its own takes the key's first run in the window, whatever
#: function it is.  With several runs there -- a DB proxy log, whose token is the method --
#: that is another function's run, and its rows read as missing
RUN_BY_ROWS = os.environ.get("WHICH_RUN_BY_ROWS", "1") != "0"
#: ``WHICH_RUN_JOIN_CALLS=0``: runs of the key in several logs stay several runs, and none is the question's
#: (2026-10-14), even where one ran inside a call the other made (the adder's updateJobs and DBProxy's archiveJob)
JOIN_CALLS = os.environ.get("WHICH_RUN_JOIN_CALLS", "1") != "0"
#: ``WHICH_RUN_ROWS_AROUND=0``: a line is proven only by a row on a path through it (implying rows, P), not by two
#: rows printed in a row that only a path through it prints (:func:`rows_around`)
ROWS_AROUND = os.environ.get("WHICH_RUN_ROWS_AROUND", "1") != "0"
#: ``WHICH_RUN_ONCE_LOOPS=0``: every loop a row and a site share may have gone round between them, the DB proxy's
#: retry too (``rows.once_loops``): activateJob's else row 111 then leaves arm 71 standing
ONCE_LOOPS = os.environ.get("WHICH_RUN_ONCE_LOOPS", "1") != "0"


def once_loops_of(fn, sn, run: Run, mode: str) -> dict[int, str]:
    """``rows.once_loops`` of *fn* in one call's run *run*: a row is not printed where rule 4 would read it as
    missing -- at a level the run prints, its absence trusted."""
    def absent(stmt: ast.stmt) -> bool:
        row = fn.row_at(stmt)
        return (row is not None and not sn.present(row) and R.absence_trusted(row, run.log, mode)[0]
                and run.would_print(row.level, row.func))
    return R.once_loops(fn.node, absent) if ONCE_LOOPS and sn.single_call() else {}


def run_holding(c: "Candidate", found: list[Run], mode: str) -> Run | None:
    """Of the key's runs in the window, the one where most of the candidate's rows are proven; None when there
    are several and none holds one.  A single run is taken as it is, as before."""
    if len(found) == 1 or not RUN_BY_ROWS:
        return found[0]
    fn = R.function(c.rel, c.func, c.lines)
    counts = [sum(1 for row in fn.rows if sn.proven(row)) for sn in (R.seen(fn, r, mode) for r in found)]
    best = max(counts)
    return found[counts.index(best)] if best else None


def of_mover(run: Run, owner: str) -> bool:
    """Whether *run* can be a run of the mover *owner*: a mark that names a method (a DB proxy's tagged logger,
    ``getJobs < SITE 2026-01-01T00:00:00 >``) is one call of that method, so it is the mover's only when the method is
    the mover's function -- a DB proxy's log holds every method's calls of the key (S1's rule, tmaeno 2026-10-13).  A
    mark that names none (``< jediTaskID=1 >``, the adder's PandaID) may be any function's of the worker.  The name is
    the tag's, which the method writes in its own comment: a comment copied from another method hides its runs."""
    m = re.match(r"([A-Za-z_]\w*) <", run.token)
    return m is None or m[1] == owner.split("::")[-1]


def passes_of(fn: R.Function, run: Run, mode: str) -> tuple[list[Run], str]:
    """*run* cut into *fn*'s passes (``rows.passes``), with a word on how.  A run kept whole that holds no record only
    *fn*'s rows can have printed (:func:`_rows_shown`), under a mark that does not name *fn*, is *fn*'s pass only where
    it shows *fn* was called (:func:`called_in`); otherwise it is not shown to be one, and is set aside.  Another
    function shares the mark -- TaskBrokerThread's runImpl prints under the same time token as TaskCheckerThread's --
    or the function's caller ran without calling it: 085's post-processing pass stopped after 104 and never reached
    doBasicPostProcess.  Kept whole, judge read the absent rows as the way the pass went and chose an exit by
    elimination (KNOWN_ISSUES.md K69).  Set aside is not shown not to be: a caller's row after the call that another
    class's method prints alike does not show it (AtlasAnal.doRefine's "done", AtlasProd.doRefine's too)."""
    parts, how = R.passes(fn, run, mode)
    if parts != [run]:
        return parts, how
    m = re.match(r"([A-Za-z_]\w*) <", run.token)
    if m is not None and m[1] == fn.name:
        return parts, how
    if _rows_shown(fn, run, mode):
        return parts, how
    shown = called_in(fn, run, mode)
    if shown:
        return parts, f"{how}; no record only {fn.name} prints, its call shown by {shown}"
    return [], f"{how}; no record only {fn.name} prints, nor a caller's row after its call: not shown to be a pass of it"


def called_in(fn: R.Function, run: Run, mode: str) -> str:
    """A record in *run* that says *fn* was called, as ``caller:line after its call(s) at [lines]``; '' for none: a
    row of a caller of *fn* (the map's ``Calls.callers``) the run printed, which every way from the caller's entry
    reaches only through one of its calls of *fn* (``flow.passes_before``, every call of *fn* in that caller handed at
    once).  setupper's run prints ``end`` at 206 only after calling setup_jumbo_jobs at 202.  The caller's rows and
    statements are read on the tree the map's calls index walked, as the flow asks."""
    try:
        types = asked._types()
        tree = types.tree(fn.rel)
        if tree is None:
            return ""
        func = next(f for f, _ in pathcond.functions_with_owner(tree) if f.lineno == fn.node.lineno and f.name == fn.name)
        owned = types.method_class(fn.rel, func)
        sites = PR._calls_index(types).callers(Target(fn.rel, func, owned[1] if owned else None))
    except (StopIteration, LookupError, ValueError):
        return ""
    callers: dict[tuple[str, int], tuple[str, Any, list[ast.stmt]]] = {}
    for site in sites:
        if site.in_module:
            continue
        stmt = pathcond.site_of(site.call)
        if isinstance(stmt, ast.stmt):
            callers.setdefault((site.file, id(site.func)), (site.file, site.func, []))[2].append(stmt)
    for rel, g, calls in callers.values():
        try:
            gfn = R.function(rel, g.name, within=[g.lineno])
        except LookupError:
            continue
        for row in _rows_shown(gfn, run, mode):
            printed = R.statement_at(g, row.line)
            if printed is None:
                continue
            try:
                if flow.passes_before(printed, calls):
                    return f"{g.name}:{row.line} after its call(s) at {sorted({c.lineno for c in calls})}"
            except ValueError:
                continue
    return ""


def _rows_shown(fn: R.Function, run: Run, mode: str) -> list[R.Row]:
    """*fn*'s rows with a record in *run* that no other function's print can have written (``rows.line_writers``;
    one the map names no writer of counts too).  TaskCheckerThread.runImpl and TaskBrokerThread.runImpl both print
    "getting Impl" through loggers with the same time-token mark: such a record does not say which ran."""
    owner = f"{fn.rel}::{pathcond.qualified_name(fn.node)}"
    sn = R.seen(fn, run, mode)
    shown = []
    for row in fn.rows:
        for i, _, _ in sn.hits.get(row.line, ()):
            if R.line_writers(run.records[i], run.log.name) <= {owner}:
                shown.append(row)
                break
    return shown


def spans(run: Run, window: tuple[str, str]) -> bool:
    """Whether *run*'s span, its first record to its last, meets *window*: how the runs at a window are gathered (H2,
    F1, H7) and how judgement 3 reads a mover as run.  A pass is judged at the window by :func:`at_window`."""
    return run.first[:19] <= window[1] and run.last[:19] >= window[0]


def at_window(run: Run, window: tuple[str, str]) -> bool:
    """Whether *run*, or a pass of one, is at *window*: its span holds the window and one of its records lies within
    EDGE of it -- the door's part near the window (``recorded._runs``).  A run :func:`passes_at` cannot cut, with no
    entry row among the records at hand, spans its separate passes, and by its span alone is at every time between
    them: a child task's two "pending until parent task 53199080 is done" at 15:53 and 22:35, under its own mark, were
    read for the parent's write at 18:49:05 (MT-scouted-late, KNOWN_ISSUES.md K57).  A pass silent for longer than EDGE
    around the window is not at it either, and its candidate is left unexamined there."""
    if not spans(run, window):
        return False
    lo, hi = (datetime.datetime.fromisoformat(window[0][:19]) - EDGE).isoformat(" "), (
        datetime.datetime.fromisoformat(window[1][:19]) + EDGE).isoformat(" ")
    return any(lo <= r.stamp[:19] <= hi for r in run.records)


def _key_through_call(fn: R.Function, part: Run, logs: dict[str, list[Run]], mode: str,
                      called_at: Callable[[R.Function, Candidate], list[str]] | None) -> str:
    """A run of the key that ran inside one of *part*'s calls, said as where it is; '' for none.  *part* is a pass of
    *fn* whose mark and records hold no key (``reactivatePendingTasks_JEDI < vo=atlas label=user ... >``), and the
    inner run's mark does (``release_task_on_hold < jediTaskID=... >``): its function is the one the map's rows say
    printed its key records (``rows.line_writers``), and *fn* must call on into it at a call the pass was inside
    (:func:`inside_call`, on the machine both name: a log line names no process, so a concurrent thread's run in the
    same seconds reads the same)."""
    if called_at is None:
        return ""
    caller: R.Seen | None = None
    for inner in (r for runs in logs.values() for r in runs):
        if (inner.token == part.token or inner.machine != part.machine or not inner.found_by
                or not (part.first <= inner.first and inner.last <= part.last)):
            continue
        owners = set.intersection(*(set(R.line_writers(r, inner.log.name)) for r in inner.found_by))
        lines = R.print_lines(next(iter(owners))) if len(owners) == 1 else []
        if not lines:
            continue
        owner = next(iter(owners))
        callee = Candidate(owner.partition("::")[0], owner.rpartition("::")[2].rpartition(".")[2], lines[:1])
        caller = caller or R.seen(fn, part, mode)
        for site in called_at(fn, callee):
            if inside_call(caller, inner, site) is not None:
                return f"{inner.describe()}, inside the call at {site}"
    return ""


def holds_key(fn: R.Function, part: Run, logs: dict[str, list[Run]], mode: str,
              called_at: Callable[[R.Function, Candidate], list[str]] | None = None) -> bool:
    """Whether *part*, a pass of *fn*, holds the key: in one of its records, or in a run of the key inside one of its
    calls (:func:`_key_through_call`).  H2 judges a candidate (:func:`passes_at`) and S2 a mover's way
    (``Rerun.passes``) only on such a pass: a poller's pass that never touched the row went its way for other rows
    (KNOWN_ISSUES.md K47)."""
    return bool(part.found_by) or bool(_key_through_call(fn, part, logs, mode, called_at))


def recorded_in_pass(fn: R.Function, found: list[Run], window: tuple[str, str], inner: list[Run], sites: list[str],
                     mode: str) -> tuple[Run, str] | None:
    """``(the pass, the call)``: a pass of *fn* at *window* inside whose call at one of *sites* (``file:line``) every run
    of *inner* ran, on the machine both name; None where none is (KNOWN_ISSUES.md K67).  *found* are runs of *fn*'s
    log under its own marks, *inner* the runs of the method that records the row, of the key near it.  A run is cut into
    passes and a pass is at the window as :func:`passes_at` reads them; a run inside a call as :func:`inside_call` reads
    it -- which says only that the inner run ran while the pass was inside that call."""
    if not inner:
        return None
    for run in found:
        if not of_mover(run, fn.owner):
            continue
        cuts = sorted({0, *R.entries(fn, run, mode), len(run.records)})
        parts = [run] if len(cuts) == 2 else [run.part(a, b) for a, b in zip(cuts, cuts[1:], strict=False)]
        for part in parts:
            if not at_window(part, window):
                continue
            seen = R.seen(fn, part, mode)
            for site in sites:
                if all(m.machine == part.machine and inside_call(seen, m, site) is not None for m in inner):
                    return part, site
    return None


def passes_at(c: Candidate, found: list[Run], window: tuple[str, str], logs: dict[str, list[Run]], mode: str,
              called_at: Callable[[R.Function, Candidate], list[str]] | None = None) -> tuple[list[Run], list[str]]:
    """``(passes, why the rest went)``: of the key's runs whose span holds *window*, the passes of *c*'s function the
    window is in and that hold the key -- what *c* is judged on (KNOWN_ISSUES.md K47).

    A run whose mark names another method is not *c*'s (:func:`of_mover`).  A run is cut into passes at the
    function's entry row (``rows.entries``), the records before the first entry a pass of their own -- the log can
    start inside one; a function with no entry row, or one pass, keeps the run whole.  A pass is at the window where
    its span holds it and a record of it lies near (:func:`at_window`), and it holds the key where one of its records
    does, or where a run of the key ran inside one of its calls (:func:`_key_through_call`).  A mark that names no key
    spans every pass of a worker:
    rescueUnLockedTasksWithPicked_JEDI's ``< vo=atlas label=user >`` on aipanda081 held 155 passes, the key in one at
    19:11, and was read for a write at 13:09:14 that aipanda082's pass made."""
    fn = R.function(c.rel, c.func, c.lines)
    kept: list[Run] = []
    why: list[str] = []
    others: dict[str, int] = {}
    for run in found:
        where = f"{run.machine.split('.')[0] or run.log.name} {run.token!r}"
        if not of_mover(run, c.owner):
            method = run.token.partition(" <")[0]
            others[method] = others.get(method, 0) + 1
            continue
        cuts = sorted({0, *R.entries(fn, run, mode), len(run.records)})
        parts = [run] if len(cuts) == 2 else [run.part(a, b) for a, b in zip(cuts, cuts[1:], strict=False)]
        at = [p for p in parts if at_window(p, window)]
        if not at:
            far = sum(1 for p in parts if spans(p, window))
            why.append(f"{where}: none of its {len(parts)} pass(es) of {fn.name} is at the window"
                       + (f" ({far} span(s) it with no record within {EDGE.seconds // 60} min of it)" if far else ""))
            continue
        keyed = [p for p in at if holds_key(fn, p, logs, mode, called_at)]
        if not keyed:
            why.append(f"{where}: its pass at the window holds no record of the key, nor a run of it inside a call")
            continue
        kept += keyed
    if others:
        why.append(f"{sum(others.values())} run(s) under another method's mark ({', '.join(sorted(others))})")
    return kept, why


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
    printers: dict[str, set[str]] = {}
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
                    printers.setdefault(expr, set()).add(sn.fn.owner)
    put = {}
    for expr, values in seen_values.items():
        if len(values) == 1:
            try:
                ast.parse(expr, mode="eval")
            except SyntaxError:
                continue
            env.put_hole(expr, next(iter(values)), where[expr], printers[expr])
            put[expr] = next(iter(values))
    return put


def printed_before(fn: R.Function, sn: R.Seen, stmt: ast.stmt, before: list[ast.stmt], env: E.Env) -> dict[str, str]:
    """Put in *env* each bare local (``{N}`` or ``{str(N)}``) that a row of *fn* printed before *stmt* on every path to
    it (*before*, :func:`rows.must_rows`) -- a record of the row placed in the run (``Seen.unambiguous``) -- with one
    value in every record of the run the row reads, where the value shown is the one *stmt* reads: the value the
    pass's guards on the way to *stmt* read.  Not a name *env* already knows, nor a row's spelling.

    What reaches *stmt* from the print is the map's reading (``pathcond.held_since``): no other write of N may reach
    it, and a value that is not an immutable scalar must not be changed in place on the way
    (``changes.object_changes``).  Where only the shown value reaches, it is put as ``put_printed`` reads every hole.
    Where a rebinding turns it on its way (a conversion), the type decides which way it took, so the text is read as
    each value it can be -- its literal and itself, each printing as the text -- and put only when every way gives one
    value (:func:`_through_conversions`).

    Rule 17 (:func:`stable_holes`) puts a local bound once in the whole function, wherever it is read; this reads one
    print and one statement.  killJob's ``code`` is a parameter and is bound again at 853 (``code = str(code)`` under
    ``if isinstance(code, int):``), so rule 17 never puts it.  Printed at 840 as ``code=2``, an int 2 takes 853's way
    and the text "2" the other, both ``"2"``: ``code in ['2', '4']`` holds at 1028 and ``code == '51'`` is false at
    1047.  KNOWN_ISSUES.md K68."""
    values: dict[str, set[str]] = {}
    last: dict[str, tuple[ast.stmt, str, str]] = {}
    for s in before:
        row = fn.row_at(s)
        # a row is read only where the run places a record of it: a text two rows (or two functions) print alike is
        # either one's -- TaskCommando's runImpl prints ``executing {commandStr}`` at 133 under ``commandStr in ['kill',
        # 'finish', 'reassign']`` and at 235 in the other arm, and 235's "executing incexec" is no record of 133
        if row is None or not (placed := sn.unambiguous.get(row.line)):
            continue
        for i, _, _ in sn.hits.get(row.line, ()):
            for expr, value in row.capture(sn.run.records[i]).items():
                inner = re.fullmatch(r"str\((\w+)\)", expr)
                name = inner[1] if inner else expr
                if not name.isidentifier() or env.canonical(name) != name or env.knows(name) or env.knows(expr):
                    continue
                # every record the row reads counts against one value, the placed ones name the source
                values.setdefault(name, set()).add(value.strip())
                if i in placed and (name not in last or last[name][0].lineno < s.lineno):
                    last[name] = (s, expr, f"hole of {fn.name}:{row.line} at {sn.run.records[i].span}")
    put: dict[str, str] = {}
    for name, seen_values in values.items():
        if len(seen_values) != 1 or name not in last:
            continue
        text = next(iter(seen_values))
        s, expr, source = last[name]
        held = pathcond.held_since(fn.node, name, stmt, s)
        if held.others or not (held.raw or held.conversions):
            continue
        if not _immutable_scalar(text) and _changed_in_place(fn, name, s, stmt):
            continue
        if not held.conversions:
            env.put_printed(expr, text, source)
            put[expr] = text
            continue
        # the conversion's way is str() of the shown value: only where the print wrote it so
        value = _through_conversions(held, name, text) if trace.renders_as_str(s, expr) else E.UNKNOWN
        if value is not E.UNKNOWN:
            turned = "; ".join(f"{ast.unparse(c.site)} at {c.site.lineno}" for c in held.conversions)
            env.put(name, value, f"{source}, through {turned}")
            put[name] = text
    return put


#: how far a log's stamps are from the database's clock on the server's machines: their statuslog rows fell inside
#: their runs' log times to the second (``recorded``, bamboo-e1); JEDI's are read within the windows' slack
CLOCK = datetime.timedelta(seconds=1)


def state_before(run: Run, states: list[dict]) -> tuple[dict | None, str]:
    """``(row, "")``: the statuslog row in force when *run* began, the last one before its first record; or ``(None,
    why)``: none is before it, or a row of another machine falls within its span -- a write the pass's read may have
    come before or after.  A row within the span written by the run's own machine is taken for the pass's own write,
    after its read: A-rec's updateJobStatus pass (aipanda092, 10:14:50,068 .. ,077) prints "old=starting new=running"
    and the statuslog holds running at 10:14:50.111 from aipanda092, starting at 10:14:39.53 before it.  A run's stamps
    are the database's clock within :data:`CLOCK` on the server's machines and within ``windows.SLACK`` on JEDI's
    (``Log.jedi``)."""
    margin = W.SLACK if run.machine in run.log.jedi else CLOCK
    first, last = W.when(run.first) - margin, W.when(run.last) + margin

    def at(s: dict) -> datetime.datetime:
        return W.when(s.get("time_extended") or s["time"])

    within = [s for s in states if first <= at(s) <= last and (s.get("host") or "") != run.machine]
    if within:
        return None, f"statuslog row(s) {sorted({s['status'] for s in within})} of another machine within the pass"
    before = [s for s in states if at(s) < first]
    if not before:
        return None, "no statuslog row before the pass"
    return max(before, key=at), ""


def row_state(fn: R.Function, stmt: ast.stmt, run: Run, env: E.Env) -> dict[str, str]:
    """Put in *env* the question's column where a test reaching *stmt* reads it from the row the function fetched in
    this pass: an attribute chain spelling the question's row and column (``job.jobStatus``, ``env.subject``) whose
    every write reaching the read is a fill from a fetched row (the map's ``fills.row_fills``) -- one row
    (``fetchone``'s, or the rows' element at a constant index), of a statement testing a key column equal to a bind,
    the column the subject's, no call changing it in place on the way.  The value is the statuslog's state when the pass
    began (:func:`state_before`): the row's value as the pass read it.  killJob reads ``job.jobStatus`` at 1017 after
    ``job.pack(res[0])`` at 998, ``res`` 992's rows ``WHERE PandaID=:PandaID``; at 1071 the stores 1028 .. 1063 reach
    too, so it is not put there.  Not a name *env* knows (a print showed it).  A local filled from a select item
    (updateJobStatus's ``oldJobStatus``) is not read: the map's parts do not say which table the item is of.

    Premises: every change of the column leaves a statuslog row (the current PanDA version); a key column tested equal
    reads the question's row.  KNOWN_ISSUES.md K71."""
    if not env.subject or not env.statuslog:
        return {}
    column = env.subject.rpartition(".")[2].lower()
    keys = {k.lower() for k in env.keys}
    reads: dict[str, list[ast.expr]] = {}
    try:
        types = asked._types()
        tree = types.tree(fn.rel)
        if tree is None:
            return {}
        func = next(f for f, _ in pathcond.functions_with_owner(tree) if f.lineno == fn.node.lineno and f.name == fn.name)
        node = R.statement_at(func, stmt.lineno)
        if node is None:
            return {}
        for test in pathcond.tests_on(node, func):
            for read in _chains(test):
                text = ast.unparse(read)
                if env.canonical(text) == env.subject and not env.knows(text):
                    reads.setdefault(text, []).append(read)
        if not reads:
            return {}
        calls = PR._calls_index(types)
        owned = types.method_class(fn.rel, func)
    except (StopIteration, LookupError, ValueError):
        return {}
    bound = tuple(calls.objects_of(owned)) if owned else ()
    put: dict[str, str] = {}
    state: dict | None = None
    for text, nodes in reads.items():
        if not all(_row_filled(calls, fn.rel, func, read, bound, column, keys) for read in nodes):
            continue
        if state is None:
            state, _why = state_before(run, env.statuslog)
            if state is None:
                return put
        when = state.get("time_extended") or state["time"]
        env.put(text, state["status"], f"the statuslog row before the pass, {state['status']!r} at {when}")
        put[text] = state["status"]
    return put


def _chains(test: ast.expr) -> list[ast.Attribute]:
    """The attribute chains on a name *test* reads, each whole (``job.jobStatus``, not ``job``), a method's receiver
    among them (``job.jobStatus.lower()`` reads ``job.jobStatus``)."""
    out: list[ast.Attribute] = []

    def visit(n: ast.AST) -> None:
        if isinstance(n, ast.Attribute):
            base = n.value
            while isinstance(base, ast.Attribute):
                base = base.value
            if isinstance(base, ast.Name):
                out.append(n)
                return
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            visit(n.func.value)
            for a in [*n.args, *(k.value for k in n.keywords)]:
                visit(a)
            return
        for child in ast.iter_child_nodes(n):
            visit(child)

    visit(test)
    return out


#: :func:`_row_filled`'s answers, by the read's place and what it is asked of
_FILLED: dict[tuple[str, int, int, str, frozenset[str]], bool] = {}


def _row_filled(calls: Any, rel: str, func: ast.AST, read: ast.expr, bound: tuple, column: str, keys: set[str]) -> bool:
    """Whether every write of *read* reaching it fills it from one row of a statement testing a key column equal to a
    bind, as *column*, with no in-place change on the way (``fills.row_fills``)."""
    key = (rel, read.lineno, read.col_offset, column, frozenset(keys))
    if key in _FILLED:
        return _FILLED[key]
    try:
        held = fills.row_fills(calls, rel, func, read, bound)  # type: ignore[arg-type]
    except ValueError:
        held = None
    ok = held is not None and bool(held.fills) and not held.others and all(
        fill.pick in (fills.PICK_ONE, fills.PICK_AT) and not fill.changes and fill.reads and all(
            one.column.lower() == column
            and any(t.test.kind == "equal" and t.test.column.lower() in keys and t.binds for t in one.tests)
            for one in fill.reads)
        for fill in held.fills)
    _FILLED[key] = ok
    return ok


#: what a conversion's tests and value may call: builtins with no effect, and the builtin types a test names
_PURE_BUILTINS: dict[str, Callable[..., Any]] = {f.__name__: f for f in (isinstance, str, int, float, bool, len, repr, abs, type)}
_BUILTIN_TYPES = {t.__name__: t for t in (int, str, float, bool, bytes, list, dict, tuple, set, frozenset)}
#: a value no call can change in place (agreed with the map for PVH): anything else is read with the calls on the way
_SCALARS = (str, int, float, bool, bytes, type(None))


def _immutable_scalar(text: str) -> bool:
    """Whether the printed *text* reads back as an immutable scalar; a text it cannot read (an object's str) is not."""
    try:
        return isinstance(ast.literal_eval(text), _SCALARS)
    except (ValueError, SyntaxError, MemoryError, RecursionError, TypeError):
        return False


def _changed_in_place(fn: R.Function, name: str, since: ast.stmt, at: ast.stmt) -> bool:
    """Whether the object *name* holds may be changed in place between the print *since* and *at*
    (``changes.object_changes``, on the corpus reading the map's calls index walked; True where it cannot be asked)."""
    try:
        types = asked._types()
        tree = types.tree(fn.rel)
        if tree is None:
            return True
        func = next(f for f, _ in pathcond.functions_with_owner(tree) if f.lineno == fn.node.lineno and f.name == fn.name)
        first, last = R.statement_at(func, since.lineno), R.statement_at(func, at.lineno)
        if first is None or last is None:
            return True
        return bool(MC.object_changes(PR._calls_index(types), fn.rel, func, name, first, last))
    except (StopIteration, LookupError, ValueError):
        return True


def _through_conversions(held: pathcond.HeldSince, name: str, text: str) -> Any:
    """The value a conversion-turned local holds where it is read, or ``UNKNOWN``: each value the printed *text* can be
    (itself, and its literal where that prints as the text) takes a way -- a decided conversion's where its tests hold,
    the shown value's where none does, both where that is not known -- and only one value on every way is put.
    killJob's ``code=2``: an int 2 takes ``code = str(code)`` under ``isinstance(code, int)``, the text "2" the other
    way, and both are ``"2"``."""
    candidates: list[Any] = [text]
    try:
        literal = ast.literal_eval(text)
    except (ValueError, SyntaxError, MemoryError, RecursionError, TypeError):
        pass
    else:
        if str(literal) == text and not isinstance(literal, str):
            candidates.append(literal)
    got: dict[tuple[str, str], Any] = {}
    for candidate in candidates:
        env = SE.Env(call=_PURE_BUILTINS.get)
        for type_name, kind in _BUILTIN_TYPES.items():
            env.put(type_name, kind, "a builtin")
        env.put(name, candidate, "the value shown")
        ways: list[Any] = []
        raw = held.raw
        for conversion in held.conversions:
            results = [SE.evaluate(test, env) for test in conversion.tests]
            if any(r is not SE.UNKNOWN and not r for r in results):
                continue
            ways.append(conversion.value)
            if conversion.decided and all(r is not SE.UNKNOWN for r in results):
                raw = False
        values = [SE.evaluate(way, env) for way in ways] + ([candidate] if raw else [])
        if any(v is SE.UNKNOWN for v in values):
            return E.UNKNOWN
        got.update({(type(v).__name__, repr(v)): v for v in values})
    return next(iter(got.values())) if len(got) == 1 else E.UNKNOWN


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
    statement is on a path from the one's print to the other's that prints nothing else (the map's
    ``flow.runs_between``), or None.

    ``updateTaskStatusByContFeeder_JEDI`` ends an arm of feed_contents_to_tasks and the next print is in another arm,
    so no print every path runs follows it; the run's "set task_status=..." before it and its next record after hold
    the JediDBProxy run between them.  Only a print the run would show closes a path between the two
    (:func:`_shown_by`)."""
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
    prints = _row_prints(caller.fn)
    first, last = R.statement_at(node, first_line), R.statement_at(node, last_line)
    if first is None or last is None or not {id(first), id(last)} <= {id(s) for s in prints}:
        return None
    return (i, j) if flow.runs_between(first, last, stmt, _shown_by(caller.fn, caller.would_show, prints, first, last)) else None


def _row_prints(fn: R.Function) -> list[ast.stmt]:
    """*fn*'s prints that are rows: what a run's records can show."""
    return [s for s in R.own_nodes(fn.node) if isinstance(s, ast.stmt) and R.is_print(s) and fn.row_at(s) is not None]


def _shown_by(fn: R.Function, shows: Callable[[R.Row], bool], prints: list[ast.stmt], *read: ast.stmt) -> list[ast.stmt]:
    """Of *prints*, the two records in a row *read* and the prints the run would show had they printed between them
    (*shows*, ``Seen.would_show``).  A print the run cannot show may have printed between the two unseen, so it closes
    no path between them: runImpl's ``tmpLog.debug(main_stop_watch.get_elapsed_time(...))`` at 838 is on every path
    from its print at 837 to 883, around the storeJobs call at 845, and its record between them is read as no row
    (KNOWN_ISSUES.md K67).  For :func:`_between_records` (a path through the call) this can only add a pass; for
    :func:`rows_around` (every path through the line) only take a proof away."""
    return [s for s in prints if any(s is r for r in read) or ((row := fn.row_at(s)) is not None and shows(row))]


def _order(printed: ast.stmt, site: ast.stmt, once: list[ast.AST]) -> str | None:
    """``"before"`` or ``"after"``: where one call that printed *printed* and ran *site* printed it, when the map's flow
    lets it go one way only; None where it goes both (a loop between them goes round) or neither (one call cannot run
    both: ``flow.together`` excludes the site instead)."""
    site_after = flow.runs_after(printed, site, once=once, printed=True)
    print_after = flow.runs_after(site, printed, once=once)
    if site_after == print_after:
        return None
    return "before" if site_after else "after"


def rows_around(fn: R.Function, sn: R.Seen, stmt: ast.stmt) -> tuple[int, int] | None:
    """``(first, last)``: two of *fn*'s rows the run printed in a row -- no row of it read between them -- that only a
    path through *stmt* prints so (the map's ``flow.passes_through``), or None.  A line no row is on every path through (a call
    of a function that prints nothing) is shown run by the rows around it: ``TaskRefinerThread.runImpl``'s 205 and 563
    around 211, ``extractCommon``.  A row read but not placed (``Seen.hits`` beyond ``unambiguous``) between the two
    breaks the pair, as in :func:`_between_records`, and only a print the run would show closes a path between them
    (:func:`_shown_by`)."""
    node = fn.node
    placed = sorted((x, line) for line, xs in sn.unambiguous.items() for x in xs)
    read = sorted(x for found in getattr(sn, "hits", {}).values() for x, *_ in found)
    prints = _row_prints(fn)
    ids = {id(s) for s in prints}
    for (i, a), (j, b) in zip(placed, placed[1:], strict=False):
        if any(i < x < j for x in read):
            continue
        first, last = R.statement_at(node, a), R.statement_at(node, b)
        if first is None or last is None or id(first) not in ids or id(last) not in ids:
            continue
        if flow.passes_through(first, last, stmt, _shown_by(fn, sn.would_show, prints, first, last)):
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
    """``(guard, where)`` on the way to every ``self.<fn>()`` call in the same class: the call's ``if``s and the early
    exits above it (``pathcond.arm_condition``)."""
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
            for test in pathcond.arm_condition(node):
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


def other_objects(fn: R.Function, stmt: ast.stmt, line: int, row: R.Row, sn: R.Seen, env: E.Env | None) -> str:
    """Why *row*, which the run printed, did not print for the question's row in a pass through the arm at *line*;
    '' if it may have (KNOWN_ISSUES.md K64).

    The names the map's records bind to the object the arm writes hold the row in a pass that wrote it
    (:mod:`row_names`).  A row printing such a name's key (``Env.keys``), under the binding that reaches the arm and
    in the arm's pass (every loop holding the arm holds the row), shows which object the pass wrote.  Where every
    record of it holds another key than the row's, the run printed it for other objects only: updateUnmergedJobs'
    ``update unmerged PandaID={umJob.PandaID}`` at 1252 printed the unmerged jobs' keys under the merge job's mark,
    so 1236, which writes umJob, did not write the merge job.  Only the key, which no pass changes; not where the
    map records nothing the arm writes, nor for a name whose value a record does not show.
    """
    m = re.fullmatch(r"<(\w+)>\.\w+", env.subject) if env is not None else None
    if env is None or m is None or not env.keys or not env.written:
        return ""
    entity = m[1]
    bound = row_names.names(fn.owner, line, env.written, entity, env.elements)
    if not bound or not R.loops_over(stmt, fn.node) <= R.loops_over(row.stmt, fn.node):
        return ""
    captures = [row.capture(sn.run.records[i]) for i, _, _ in sn.hits.get(row.line, ())]
    for expr in sorted({e for c in captures for e in c}):
        name, _, column = expr.rpartition(".")
        held = env.table.get(entity, {}).get(column)
        if column not in env.keys or bound.get(name) != entity or held is None:
            continue
        values = [c.get(expr) for c in captures]
        if any(v is None for v in values) or not row_names.same_binding(fn.node, name, [row.line], line):
            continue
        wanted = str(held.value)
        if wanted not in [str(v).strip() for v in values]:
            return f"row {row.line} printed only for other values of {expr} ({len(values)} time(s)), not {wanted!r}"
    return ""


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
    inside the adder's call) -- would decide the term before the guard could, so it would never be settled.  The
    guards are the statement's and its callers' ``pathcond.arm_condition``; one read at another line
    (``E.read_elsewhere``) is not solved here.
    """
    guards = [(g, f"{fn.name}:{stmt.lineno}") for g in pathcond.arm_condition(stmt)] + callers_guards(fn)
    for guard, where in guards:
        if E.read_elsewhere(guard):
            continue
        asked = set(re.findall(rf"(?<![\w.])[\w.]+\.{re.escape(attr)}(?!\w)", guard))
        # read in the function the guard is tested in: the statement's, or the caller's (``where`` is "name:line")
        name, _, line = where.rpartition(":")
        hidden = env.reading(f"{fn.rel}::{name}", int(line))
        hidden.stale.update({s: ASKED for x in asked for s in (x, env.canonical(x))})
        try:
            solved = E.solve(guard, True, hidden)
        except SyntaxError:
            continue
        for atom, truth in solved.settled.items():
            m = re.fullmatch(rf"([\w.]+\.{re.escape(attr)}) (==|!=) ['\"]{re.escape(observed)}['\"]", atom)
            # only the row's own column: a name the arm's env does not read as the row (a caller's job handing on
            # another object, KNOWN_ISSUES.md K64) is another object's status
            if m and env.subject and hidden.canonical(m[1]) != env.subject:
                continue
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
    holds_write: bool = False,
    keeps: CH.Keeps | None = None,
) -> Judgement:
    """*called_at* says where a function calls on, by the map, into a candidate's function: the ``file:line`` of every
    call from which it goes on there.  Given it, and no *question_run*, runs of the key in several logs are read as
    one where every run but one ran inside a call that one made (:func:`inside_call`, on the machine both name), and
    that one is the question's run.  *keeps* reads, by the map's reach, a call on the way of a value rule R carries
    (``changed.breaks``).

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
                found = [r for r in logs[name] if window is None or spans(r, window)]
                if found and window is not None:
                    found, dropped = passes_at(c, found, window, logs, mode, called_at)
                    if not found:
                        reasons.append(f"{name}: no pass of the key at {window}: {'; '.join(dropped)}")
                        continue
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

    # each arm's run as its proofs read it: less the rows printed for other objects than the one it writes
    arm_seen: dict[tuple[str, int], R.Seen] = {}
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
        once = once_loops_of(fn, sn, run, mode)
        for line in c.lines:
            stmt = R.statement_at(fn.node, line)
            v = Verdict(c, line, "kept")
            if stmt is None:
                v.reasons.append("no statement on this line")
                verdicts.append(v)
                continue
            before, after = R.must_rows(fn.node, stmt, raised if named else set())
            must = before + after
            # rows the run printed for other objects only (other_objects): not in the run for the row, for rule 4 and
            # for every proof of the arm
            for_others = {r.line: why for r in fn.rows if sn.present(r) and (why := other_objects(fn, stmt, line, r, sn, env))}
            arm_seen[(c.owner, line)] = psn = sn.without(for_others)
            absent, neutral, shown, proof, wrong_value, per_element = [], [], [], False, [], []
            for s in must:
                row = fn.row_at(s)
                if row is None:
                    continue
                if sn.present(row):
                    elsewhere = (other_elements(fn, stmt, row, sn, env) if LOOP_ELEMENT else "") or for_others.get(row.line, "")
                    if elsewhere and R.absence_trusted(row, run.log, mode)[0] and run.would_print(row.level, row.func):
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
                elif (trust := R.absence_trusted(row, run.log, mode))[0] and run.would_print(row.level, row.func):
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
                # the map's flow: a printed row and the site that one call cannot both run -- below the return every
                # way from the print ends at, in another arm no loop holds with it, in another arm of a pass of a loop
                # the run went round once
                loops = [n for n in ast.walk(fn.node) if id(n) in once]
                shown_rows = [row for row in fn.rows if sn.unambiguous.get(row.line)]
                for row in shown_rows:
                    if not flow.together(row.stmt, stmt, once=loops):
                        v.kind = "excluded"
                        went = R.loops_over(row.stmt, fn.node) & R.loops_over(stmt, fn.node)
                        v.reasons.append(f"row {row.line} printed, and one call cannot print it and run {line}"
                                         + "".join(f" ({once[i]})" for i in went if i in once))
                        break
                if bracket is not None and run is question_run:
                    order = {r.line: _order(r.stmt, stmt, loops) for r in shown_rows}
                    lo = max((i for r in shown_rows if order[r.line] == "before" for i in sn.unambiguous[r.line]), default=None)
                    hi = min((i for r in shown_rows if order[r.line] == "after" for i in sn.unambiguous[r.line]), default=None)
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
                own = env.reading(fn.owner, stmt.lineno, arm=(fn.owner, line))
                stable_holes([x for x in seens.values() if x.run is run], own, class_of(fn))
                why = rewriter(fn, stmt, subject, observed, own)
                if why:
                    v.kind = "excluded"
                    v.reasons.append(why)
            if wrong_value:
                v.kind = "excluded"
                v.reasons.append(f"value: {wrong_value}")
            if env is not None:
                # the tests reaching it runs under, the early exits above it too; one whose names are as they are at
                # another line (``[at N]``) is not read in the pass's values (KNOWN_ISSUES.md K61); a hole another
                # function printed is read only where the map hands its object here (KNOWN_ISSUES.md K63)
                genv = env.reading(fn.owner, stmt.lineno, arm=(fn.owner, line))
                # the bare locals a print before it on every path showed, not bound again on the way (KNOWN_ISSUES.md K68)
                printed_before(fn, psn, stmt, before, genv)
                # the question's column as the pass read it from the row: the statuslog's state then (K71)
                row_state(fn, stmt, psn.run, genv)
                for guard in pathcond.arm_condition(stmt):
                    if E.read_elsewhere(guard):
                        continue
                    try:
                        if E.evaluate(guard, genv) is False:
                            v.kind = "excluded"
                            used = {k: val for k, val in {**env.used, **genv.used}.items() if k in guard}
                            v.reasons.append(f"step 5: guard {guard!r} is false ({used})")
                            break
                    except SyntaxError:
                        continue
                env.used.update(genv.shared())
            if loops_empty := empty_loops(fn, stmt, callee):
                v.kind = "excluded"
                v.reasons.append(loops_empty)
            proof = PR.prove(fn, psn, stmt)
            if v.kind == "kept":
                implying = [fn.row_at(s) for s in R.implying_rows(fn.node, stmt)]
                proven = [r.line for r in implying if r is not None and psn.proven(r)]
                if proven:
                    v.kind = "positive"
                    v.reasons.append(f"row(s) {proven} print only on a path through it")
                elif proof.kind == "positive":
                    v.kind = "positive"
                    v.reasons.append(proof.reason)
                elif ROWS_AROUND and (around := rows_around(fn, psn, stmt)) is not None:
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
            sn = arm_seen.get((c.owner, v.line), seens[c.owner])
            lines = {
                int(line) for other in standing if other.owner != c.owner
                for site in called_at(sn.fn, other) for file, _, line in [site.rpartition(":")] if file == c.rel
            }
            proof = PR.prove(sn.fn, sn, stmt, PR.Reaching(observed, frozenset(lines)))
            if proof.kind == "positive":
                v.kind = "positive"
                v.reasons.append(f"{proof.reason}; read with the other verdicts: a call reaching no writer of {observed!r} left standing does not write it")

    # rule R: of one pass's writes, the last that changed the value -- the stays and the writes of another value out
    change = None
    if subject and observed and question_run is not None:
        change = CH.last_change(verdicts, seens, runs_of, question_run, {k: site for k, (_, _, _, site) in joined.items()},
                                subject, observed, called_at, keeps)
        for v, why in change.dropped if change is not None else []:
            v.kind = "excluded"
            v.reasons.append(why)

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
    if change is not None and change.chosen is not None and (decide != "strict" or not unexamined):
        chosen, how = change.chosen, change.how
    elif change is not None and len(change.left) > 1:
        how = change.how
    elif len(positives) == 1 and (decide != "strict" or not unexamined):
        chosen, how = positives[0], "positive evidence"
    elif len(positives) > 1:
        how = f"{len(positives)} positive"
    elif complete and len(left) == 1 and (not unexamined or (holds_write and decide != "strict")):
        # the one left wrote it: where it is unexamined, only in a window that holds the write itself (K47) -- a
        # statuslog row asked is a guess at when it was written, and its next row is asked instead
        chosen, how = left[0], "elimination over a complete list"
    elif decide == "strict" and unexamined:
        how = f"strict: {len(unexamined)} unexamined"
    elif question_run is not None and bracket is not None and len(in_run) == 1:
        others = len({v.candidate.owner for v in unexamined})
        chosen = in_run[0]
        how = f"elimination within the run the value changed in; {others} function(s) never placed in it are not excluded"
    elif left:
        read = sorted({f"{v.candidate.func}:{v.line}" for v in left if v.kind != "unexamined"})
        how = f"{len(left)} left: " + ", ".join(read) if read else f"{len(left)} left, all unexamined"
    notes = join_notes + [f"MAP: {m}" for m in sorted(map_findings)] + p_notes
    return Judgement(verdicts, chosen, how, bracket, notes, runs_of, question_run if joined else None)

