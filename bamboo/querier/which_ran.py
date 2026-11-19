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
  only through a writer of it not excluded (``printed.Reaching``); or, one
  binding a local to the observed value, a row of its pass printed the local
  with it, and of the bindings it can have read only this one gives it (P1,
  :func:`printed_local`);
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
import functools
import itertools
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, NamedTuple

from bamboo.codemap import evaluate as SE  # noqa: E402
from bamboo.codemap import (  # noqa: E402
    pathcond,
    reaching,
    trace,  # noqa: E402
)
from bamboo.codemap.calls import Target  # noqa: E402
from bamboo.codemap.models import (  # noqa: E402
    MUST_PART_BIND,
    MUST_PART_MARK,
    MUST_ROW_WRITTEN,
)
from bamboo.codemap.panda import changes as MC  # noqa: E402
from bamboo.codemap.panda import fills, flow, sqltext  # noqa: E402
from bamboo.querier import asked, mapdump, row_names  # noqa: E402
from bamboo.querier import changed as CH  # noqa: E402
from bamboo.querier import evaluate as E  # noqa: E402
from bamboo.querier import map_reach as MR  # noqa: E402
from bamboo.querier import printed as PR  # noqa: E402
from bamboo.querier import rows as R  # noqa: E402
from bamboo.querier import windows as W  # noqa: E402
from bamboo.querier.recorded import EDGE  # noqa: E402
from bamboo.querier.records import LEADING_HOLE, Run  # noqa: E402


def once_loops_of(fn, sn, run: Run, mode: str) -> dict[int, str]:
    """``rows.once_loops`` of *fn* in one call's run *run*: a row is not printed where rule 4 would read it as
    missing -- at a level the run prints, its absence trusted.  And the loops over objects one of whose rows in the
    run printed the run's mark as the element's (:func:`keyed_loops`)."""
    def absent(stmt: ast.stmt) -> bool:
        row = fn.row_at(stmt)
        return (row is not None and not sn.present(row) and R.absence_trusted(row, run.log, mode)[0]
                and run.would_print(row.level, row.func, row))
    if not sn.single_call():
        return {}
    return R.once_loops(fn.node, absent) | keyed_loops(fn, sn, run)


def _names_key(token: str, value: str) -> bool:
    """Whether the mark *token* is the value *value*: the whole mark (the adder's and datasetManager's PandaID), or one
    of its ``name=value`` fields (``< jediTaskID=7 >``) -- not a digit of a time token."""
    value = value.strip()
    return bool(value) and (token.strip().strip("<>").strip() == value
                            or re.search(rf"\b\w+={re.escape(value)}(?![\w.])", token) is not None)


def keyed_loops(fn, sn, run: Run) -> dict[int, str]:
    """``{id(loop): why}``: the loops over objects (``for job in jobs``) that went round once for the run's object.  In
    one call's run (every row printed once) of the records of a key, a row inside the loop printed an attribute of the
    loop's element whose value is the run's mark: those records are of the round on that object.  It rests on the
    object being once in what the loop goes over.  datasetManager's FinisherThr.run 662, whose rows print
    ``job.PandaID``: the pass printed 696 and later rows, so it did not take 664's ``continue`` on that job."""
    out = {}
    for loop in R.own_nodes(fn.node):
        if not isinstance(loop, (ast.For, ast.AsyncFor)) or not isinstance(loop.target, ast.Name):
            continue
        element = loop.target.id + "."
        for row in fn.rows:
            if not loop.lineno <= row.line <= (loop.end_lineno or loop.lineno) or not sn.present(row):
                continue
            named = next((expr for i, _, _ in sn.hits.get(row.line, []) for expr, value in row.capture(run.records[i]).items()
                          if expr.startswith(element) and _names_key(run.token, value)), None)
            if named is not None:
                out[id(loop)] = (f"the loop at {loop.lineno} went round once for the run's object: row {row.line} "
                                 f"printed {named} as the run's mark (the object once in what it goes over)")
                break
    return out


def run_holding(c: "Candidate", found: list[Run], mode: str) -> Run | None:
    """Of the key's runs in the window, the one where most of the candidate's rows are proven; None when there
    are several and none holds one.  A single run is taken as it is, as before."""
    if len(found) == 1:
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


def turn_parts(fn: R.Function, run: Run, mode: str) -> tuple[list[Run], str]:
    """*run* cut at the turns of *fn*'s loop over objects whose rows print the run's mark as the element's
    (:func:`keyed_loops`), with a word on how; ``([run], '')`` where no one such loop is, or the run holds one turn.

    A record starts a turn where its row cannot run after the row of the record before it in one turn of that loop
    (the map's ``flow.runs_after``, the loop gone round once).  A record of no row of *fn*, or of several, starts none,
    and the row before it stays the one read.  A run of a key's records holds one turn per call of a worker that takes
    the object once in what it goes over: datasetManager's FinisherThr.run prints ``{job.PandaID} Cloud:`` at 696 and
    four rows after it in each turn of 662, and NJ-transferring-stuck's run of the job on one machine holds three rounds
    of the Finisher in 15 records.  A turn that leaves before 696 (675's error, then ``continue``) is a turn too: 675
    cannot run after 734 in one turn.  Bundle D, D-2 (P30; out/gm/qk-design.md, tmaeno's OK 2026-11-17)."""
    sn = R.seen(fn, run, mode)
    keyed = keyed_loops(fn, sn, run)
    loops = [node for node in R.own_nodes(fn.node) if isinstance(node, (ast.For, ast.AsyncFor)) and id(node) in keyed]
    if len(loops) != 1:
        return [run], ""
    loop = loops[0]
    stmt_of = {row.line: row.stmt for row in fn.rows}
    lines_of: dict[int, set[int]] = {}
    for line, found in sn.hits.items():
        for i, _, _ in found:
            lines_of.setdefault(i, set()).add(line)
    starts = [0]
    before: ast.stmt | None = None
    for i in range(len(run.records)):
        lines = lines_of.get(i, set())
        if len(lines) != 1:
            continue
        stmt = stmt_of[next(iter(lines))]
        if before is not None:
            try:
                follows = flow.runs_after(before, stmt, once=[loop], printed=True)
            except ValueError:
                follows = True
            if not follows:
                starts.append(i)
        before = stmt
    if len(starts) < 2:
        return [run], ""
    cuts = starts + [len(run.records)]
    parts = [run.part(a, b) for a, b in zip(cuts, cuts[1:], strict=False)]
    return parts, (f"cut at the turns of the loop at {loop.lineno}, whose rows print the run's mark as its element's: "
                   f"{len(parts)} pass(es), one starting where a record's row cannot run after the one before it in a turn")


def cut_passes(fn: R.Function, run: Run, mode: str) -> tuple[list[Run], str]:
    """*run* cut into *fn*'s passes: at its entry row (``rows.passes``), else at the turns of its loop over the run's
    object (:func:`turn_parts`); with a word on how."""
    parts, how = R.passes(fn, run, mode)
    if parts == [run]:
        turns, said = turn_parts(fn, run, mode)
        if len(turns) > 1:
            return turns, f"{how}; {said}"
    return parts, how


def minute_parts(run: Run, how: str, gap: datetime.timedelta = datetime.timedelta(minutes=1)) -> tuple[list[Run], str]:
    """P69 (the investigator's u13): a run no entry row cut, cut where its records stop for *gap* or more, each part a
    pass -- a premise, said: no run of the function pauses that long, and no two of its runs fall within it.
    getTasksToExecCommand_JEDI's runs under the mark its calls share hold hours of calls (aipanda088's 14:12 .. 21:00);
    the Adder's run on a job, one burst, stays whole.  ``([run], how)`` where there is no such gap."""
    try:
        stamps = [r.stamp for r in run.records]
        cuts = [i for i in range(1, len(stamps)) if W.when(stamps[i]) - W.when(stamps[i - 1]) >= gap]
    except (TypeError, AttributeError, ValueError):
        return [run], how
    if not cuts:
        return [run], how
    bounds = [0, *cuts, len(stamps)]
    parts = []
    for lo, hi in zip(bounds, bounds[1:], strict=False):
        records = run.records[lo:hi]
        held = {id(r) for r in records}
        parts.append(replace(run, records=records, found_by=[r for r in run.found_by if id(r) in held]))
    return parts, (f"{how}; cut where its records stop for a minute into {len(parts)} part(s) (P69, a premise: no run "
                   "of it pauses that long, no two fall within it)")


def shared_mark_parts(parts: list[Run], run: Run, how: str) -> tuple[list[Run], str]:
    """P69: :func:`minute_parts` for a run no entry row cut (*parts* is ``[run]``, the run itself: a cut into one pass
    is a copy, and stays as it is) under a mark its method's calls share (the door's ``Log.partial``: the key's records
    under it come EDGE or more apart, so its run is every call in the window) -- getTasksToExecCommand_JEDI's
    ``< vo=None label=None >``.  Any other run stays whole: one execution of a function with no entry row may pause a
    minute (ContentsFeeder's feed_contents_to_tasks waits on its file listing: 2127 records in 14 minutes, a pause of
    60 s between "get files" and "got ... files"), and where it holds an edge of the value's time with no such pause it
    is read there as one execution (``rerun._stay_edges``)."""
    if len(parts) != 1 or parts[0] is not run:
        return parts, how
    partial = getattr(getattr(run, "log", None), "partial", None) or {}
    if partial.get((getattr(run, "machine", None), getattr(run, "token", None))) != "a mark the method's calls share":
        return [run], how
    return minute_parts(run, how)


def passes_of(fn: R.Function, run: Run, mode: str) -> tuple[list[Run], str]:
    """*run* cut into *fn*'s passes (:func:`cut_passes`), with a word on how.  A run kept whole that holds no record only
    *fn*'s rows can have printed (:func:`_rows_shown`), under a mark that does not name *fn*, is *fn*'s pass only where
    it shows *fn* was called (:func:`called_in`); otherwise it is not shown to be one, and is set aside.  Another
    function shares the mark -- TaskBrokerThread's runImpl prints under the same time token as TaskCheckerThread's --
    or the function's caller ran without calling it: 085's post-processing pass stopped after 104 and never reached
    doBasicPostProcess.  Kept whole, judge read the absent rows as the way the pass went and chose an exit by
    elimination (KNOWN_ISSUES.md K69).  Set aside is not shown not to be: a caller's row after the call that another
    class's method prints alike does not show it (AtlasAnal.doRefine's "done", AtlasProd.doRefine's too)."""
    parts, how = cut_passes(fn, run, mode)
    if parts != [run]:
        return parts, how
    m = re.match(r"([A-Za-z_]\w*) <", run.token)
    if m is not None and m[1] == fn.name:
        return shared_mark_parts(parts, run, how)
    if _rows_shown(fn, run, mode):
        return shared_mark_parts(parts, run, how)
    shown = called_in(fn, run, mode)
    if shown:
        parts, how = shared_mark_parts(parts, run, how)
        return parts, f"{how}; no record only {fn.shown} prints, its call shown by {shown}"
    return [], f"{how}; no record only {fn.shown} prints, nor a caller's row after its call: not shown to be a pass of it"


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


def ends_before(run: Run, window: tuple[str, str], since: str) -> bool:
    """Whether *run*, or a pass of one, can have written a statuslog row at *window* after its last record: it began
    after *since* (the time of the row before; "" where none) and by the window's end, and its last record lies before
    the window, within EDGE of it.  Read for a writer whose pass writes the row by a call after its write
    (:attr:`Candidate.row_after`), and only where no pass of it is at the window (:func:`at_window`): its last print
    can come before the row -- insertNewJob in a bulk insert queues its job's statuslog row for bulk_insert_new_jobs'
    executemany, which wrote 'pending' at 21:02:38.14, 2.5 s after the pass's last record at 21:02:35.684
    (NR-J-7424719827-prodUserName, P37).  A pass that began by the row before can have written that row instead."""
    lo, _ = before_stretch(window, since)
    return since[:19] < run.first[:19] <= window[1] and lo <= run.last[:19] < window[0]


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


#: how many calls up :func:`cutter` walks from a candidate's function to a caller printing in the same run, and how many
#: functions it asks there at most
CALLER_DEPTH = 4
CALLER_MAX = 40


def _function_of(owner: str) -> R.Function | None:
    """The querier's function for the map's *owner* (``rel::Class.method``), picked by the lines it prints; None where
    the file holds no def of that name."""
    rel, name = owner.partition("::")[0], owner.rpartition("::")[2].rpartition(".")[2]
    try:
        return R.function(rel, name, within=R.print_lines(owner))
    except (LookupError, OSError, SyntaxError, ValueError):
        return None


def cutter(fn: R.Function, run: Run, mode: str,
           called_at: Callable[[R.Function, Candidate], list[str]] | None) -> R.Function:
    """The function whose passes *run* is cut into for a candidate of *fn* (bundle C, C-1, the step after measure bE):
    *fn*, or the outermost of the functions that call on into it by the map (*called_at*) and print in *run* too.  A
    callee printing through its caller's logger runs inside the caller's pass, and cut at its own entry row it would
    leave the caller's rows before the call in a pass apart: the adder's handle_failed_job, whose ``AdderGen.run will
    call job_failure_postprocessing`` at 363 is its entry row, prints under the PandaID its callers process_job_report
    and AdderGen.run print under too; cut there, the judge read its candidate 374 on one run and the callers' on
    another, joined no run of the key's (one root is needed), and rule R no longer read process_job_report's 444
    before 374 (NJ-starting-late).  The callers are read backwards up the map's calls (``map_reach.callers``) within
    CALLER_DEPTH calls, through those that print nothing in *run* (the adder's finalize_job_status, between
    process_job_report and handle_failed_job), at most CALLER_MAX functions; of them, those that print a record of *run*
    (:func:`_prints_in`) and from which the map's reach goes on into *fn* (*called_at*) print there, and the one of
    them no other reaches is the outermost.  None of them, *fn*; two outermost, which pass holds the call is not told:
    *fn* too."""
    record = MR.graph()
    if called_at is None or record is None or not fn.node.body:
        return fn
    held = _CUTTERS.get((id(run), id(fn.node)))
    if held is not None and held[0] is run:
        return held[1]

    def as_callee(f: R.Function) -> Candidate:
        return Candidate(f.rel, f.node.name, [f.node.body[0].lineno])

    start = record.function_at(fn.rel, fn.node.body[0].lineno)
    asked: set[str] = {start} if start else set()
    frontier = set(asked)
    for _ in range(CALLER_DEPTH):
        frontier = {c for f in frontier for c in MR.callers(f)} - asked
        asked |= frontier
        if not frontier or len(asked) > CALLER_MAX:
            break
    printing = [g for owner in sorted(asked - {start}) if (g := _function_of(owner)) is not None and g.node is not fn.node
                and g.node.body and _prints_in(owner, g, run, mode) and called_at(g, as_callee(fn))]
    outer = [g for g in printing if not any(h is not g and called_at(h, as_callee(g)) for h in printing)]
    found = outer[0] if len(outer) == 1 and len(asked) <= CALLER_MAX else fn
    _CUTTERS[(id(run), id(fn.node))] = (run, found)
    return found


def _prints_in(owner: str, g: R.Function, run: Run, mode: str) -> bool:
    """Whether the map's function *owner* (*g*) printed a record of *run*: a row of *g* reads it, and the map reads
    the record as that row's whole in the run's log (``rows.whole_writers``: its mark, then its message, anchored at
    both ends, through the log files its logger reaches).  A row's text alone proves nothing (KNOWN_ISSUES.md K42), and
    measure bE2 read it so twice: TaskCommando.start's ``start``, printed to panda-TaskCommando.log, read
    getTasksToExecCommand_JEDI's ``start`` in panda-JediDBProxy.log (MT-finishing-late, NT-tobroken-late), and
    TaskRefiner.start's rows, through a logger with no mark, read records under a task's mark in its own log
    (NT-registered-late); cut by them (no entry row there) the run was read whole, its pass at every window."""
    return any(owner in R.whole_writers(run.records[i], run.log.name)
               for at in R.seen(g, run, mode).hits.values() for i, _, _ in at)


#: cutter's answers by the run and the candidate's function, kept with the run (an id is reused only once it is gone)
_CUTTERS: dict[tuple[int, int], tuple[Run, R.Function]] = {}
#: the passes a run is cut into, by the run and its cuts: every candidate cut alike holds the same pass objects, so the
#: judge reads them as one run (its join of runs, rule R's pass)
_PARTS: dict[tuple[int, tuple[int, ...]], tuple[Run, list[Run]]] = {}


def _cut(run: Run, cuts: list[int]) -> list[Run]:
    """*run*'s passes at *cuts* (indexes, its ends among them), the same objects for the same run and cuts."""
    key = (id(run), tuple(cuts))
    held = _PARTS.get(key)
    if held is None or held[0] is not run:
        held = (run, [run] if len(cuts) == 2 else [run.part(a, b) for a, b in zip(cuts, cuts[1:], strict=False)])
        _PARTS[key] = held
    return held[1]


def passes_at(c: Candidate, found: list[Run], window: tuple[str, str], logs: dict[str, list[Run]], mode: str,
              called_at: Callable[[R.Function, Candidate], list[str]] | None = None) -> tuple[list[Run], list[str]]:
    """``(passes, why the rest went)``: of the key's runs whose span holds *window*, the passes of *c*'s function the
    window is in and that hold the key -- what *c* is judged on (KNOWN_ISSUES.md K47).

    A run whose mark names another method is not *c*'s (:func:`of_mover`).  A run is cut into passes at the
    function's entry row (``rows.entries``), the records before the first entry a pass of their own -- the log can
    start inside one; a function with no entry row, or one pass, keeps the run whole.  Where a caller of the function
    prints in the run too, the run is cut at the outermost such caller's entry row instead (:func:`cutter`): the
    function's call is a part of that caller's pass, and every candidate of the run is cut alike, into the same pass
    objects (:func:`_cut`).  A pass is at the window where
    its span holds it and a record of it lies near (:func:`at_window`), and it holds the key where one of its records
    does, or where a run of the key ran inside one of its calls (:func:`_key_through_call`).  A mark that names no key
    spans every pass of a worker:
    rescueUnLockedTasksWithPicked_JEDI's ``< vo=atlas label=user >`` on aipanda081 held 155 passes, the key in one at
    19:11, and was read for a write at 13:09:14 that aipanda082's pass made.

    Where no pass is at the window and *c* writes its row by a call after its write (:attr:`Candidate.row_after`), the
    passes that ended before the row, from the row before on, are read instead (:func:`ends_before`, P37): *found*
    holds the runs meeting that stretch too (:func:`before_stretch`)."""
    kept, why = _passes(c, [r for r in found if spans(r, window)], window, logs, mode, called_at,
                        lambda p: at_window(p, window))
    if kept or c.row_after is None:
        return kept, why
    since = c.row_after
    before, _ = _passes(c, found, window, logs, mode, called_at, lambda p: ends_before(p, window, since))
    return (before, why) if before else (kept, why)


def before_stretch(window: tuple[str, str], since: str) -> tuple[str, str]:
    """The stretch a pass :func:`ends_before` *window* lies in: from the row before (*since*), or from EDGE before the
    window where that is later, to the window's end."""
    lo = (datetime.datetime.fromisoformat(window[0][:19]) - EDGE).isoformat(" ")
    return max(lo, since[:19]), window[1]


def _passes(c: Candidate, found: list[Run], window: tuple[str, str], logs: dict[str, list[Run]], mode: str,
            called_at: Callable[[R.Function, Candidate], list[str]] | None, near: Callable[[Run], bool]) -> tuple[list[Run], list[str]]:
    """:func:`passes_at`'s reading, a pass taken where *near* holds of it."""
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
        by = cutter(fn, run, mode, called_at)
        parts = _cut(run, sorted({0, *R.entries(by, run, mode), len(run.records)}))
        at = [p for p in parts if near(p)]
        if not at:
            far = sum(1 for p in parts if spans(p, window))
            of = fn.shown if by is fn else f"{by.shown} (calling {fn.shown})"
            why.append(f"{where}: none of its {len(parts)} pass(es) of {of} is at the window"
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
    #: where the window is a statuslog row every way of the function writes by a call after its write (the map's
    #: history pairs, order ``after``): the time of the row before ("" where none), from which a pass that ended
    #: before the row is read where none is at it (:func:`ends_before`); None for any other candidate
    row_after: str | None = None

    @property
    def owner(self) -> str:
        return f"{self.rel}::{self.func}"


@dataclass
class Verdict:
    candidate: Candidate
    line: int
    kind: str  # excluded | positive | unexamined | kept
    reasons: list[str] = field(default_factory=list)
    #: the must rows a positive verdict was shown by (``rows.implying_rows`` printed), none for another proof: a reader
    #: that reads a caller's rows on the side of a send its code puts them on asks where they lie (HOP (a'))
    proven: list[int] = field(default_factory=list)
    #: D-4: why the call arm hands no element of the key's turn (:func:`turn_left_before_append`), empty otherwise
    unhanded: str = ""

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
    #: where rule R chose the one write shown of several left in the pass (P34): the others, which may have written it
    #: last in that pass (``changed.Change.others``)
    pass_writes: list[str] = field(default_factory=list)

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
                        found.append((i, value.strip(), f"{sn.fn.shown}:{row.line}"))
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
                    where.setdefault(expr, f"hole of {sn.fn.shown}:{row.line} at {sn.run.records[i].span}")
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


def constant_on_way(fn: R.Function, printed: list[int], test_line: int,
                    bindings: dict[int, Any]) -> tuple[Any, int, int] | None:
    """Bundle F, F-2 (out/gm/qk-design.md, tmaeno's OK 2026-11-17 15:2x): what a name the test at *test_line* of *fn*
    reads holds there in one turn of a pass, where the map's GuardFrom reads it back to more than one literal -- its
    *bindings*, by line -- prepareTasksToBeFinished_JEDI's ``toSkip``, False at 3861 and True at 3939, which 4068's
    ``not toSkip`` reads.  A row the pass printed in that turn (*printed*: the lines whose records there hold the key)
    that every way reaches through one binding (``flow.passes_before``) with no other binding between them, and from
    which the test is reached with no other binding between (``flow.runs_between``), says that binding's value is what
    the test read: 3940's ``done set only scout data for jediTaskID=...`` comes only past 3939, and no binding of
    ``toSkip`` runs from it to 4068.  The turn is the innermost loop holding the print and the test, gone round once.
    ``(value, binding line, print line)``; None where no printed row says, or two say different bindings."""
    node = fn.node
    test = R.statement_at(node, test_line)
    sites: dict[int, ast.stmt] = {}
    for line in bindings:
        if (site := R.statement_at(node, line)) is None:
            return None
        sites[line] = site
    if test is None:
        return None
    holding_test = {id(a) for a in pathcond.ancestors(test)}
    by_value: dict[str, list[int]] = {}
    for b, v in bindings.items():
        by_value.setdefault(repr(v), []).append(b)
    found: dict[str, list[tuple[int, int]]] = {}
    for line in sorted(set(printed)):
        p = R.statement_at(node, line)
        if p is None or p is test or any(p is s for s in sites.values()):
            continue
        once = [a for a in pathcond.ancestors(p) if isinstance(a, R.LOOPS) and id(a) in holding_test][:1]
        try:
            if not flow.runs_after(p, test, once=once, printed=True) or not flow.passes_before(p, list(sites.values()), once=once):
                continue
            for value, lines in by_value.items():
                same = [sites[b] for b in lines]
                other = [s for b, s in sites.items() if b not in lines]
                # every way to the print passed a binding of this value last, and none of another runs on to the test
                if (flow.passes_before(p, same, once=once)
                        and not any(flow.runs_after(o, p, same, once=once) for o in other)
                        and not any(flow.runs_between(p, test, o, [], once=once) for o in other)):
                    found.setdefault(value, []).extend((b, line) for b in lines
                                                       if flow.runs_after(sites[b], p, [], once=once))
        except ValueError:
            continue
    if len(found) != 1:
        return None
    pairs = next(iter(found.values()))
    if not pairs:
        return None
    b = min(b for b, _ in pairs)
    return bindings[b], b, min(line for _, line in pairs)


def printed_before(fn: R.Function, sn: R.Seen, stmt: ast.stmt, before: list[ast.stmt], env: E.Env,
                   several: dict[str, list[Any]] | None = None) -> dict[str, str]:
    """Put in *env* each bare local (``{N}`` or ``{str(N)}``) that a row of *fn* printed before *stmt* on every path to
    it (*before*, :func:`rows.must_rows`) -- a record of the row placed in the run (``Seen.unambiguous``) -- with one
    value in every record of the run the row reads, where the value shown is the one *stmt* reads: the value the
    pass's guards on the way to *stmt* read.  Not a name *env* already knows, nor a row's spelling.

    What reaches *stmt* from the print is the map's reading (``pathcond.held_since``): no other write of N may reach
    it, and a value that is not an immutable scalar must not be changed in place on the way
    (``changes.object_changes``).  Where only the shown value reaches, it is put as ``put_printed`` reads every hole.
    Where a rebinding turns it on its way (a conversion), the type decides which way it took, so the text is read as
    each value it can be -- its literal and itself, each printing as the text -- and put only when every way gives one
    value (:func:`_ways_through`).

    Rule 17 (:func:`stable_holes`) puts a local bound once in the whole function, wherever it is read; this reads one
    print and one statement.  killJob's ``code`` is a parameter and is bound again at 853 (``code = str(code)`` under
    ``if isinstance(code, int):``), so rule 17 never puts it.  Printed at 840 as ``code=2``, an int 2 takes 853's way
    and the text "2" the other, both ``"2"``: ``code in ['2', '4']`` holds at 1028 and ``code == '51'`` is false at
    1047.  KNOWN_ISSUES.md K68.

    *several*, where handed: each name whose ways give more than one value, with those values, put nowhere -- a reader
    of one test may take its answer where every value gives the same (J, :func:`_truth`).  killJob's ``code=None``
    is None or the text "None", and ``code in ['2', '4']`` is false under both."""
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
                    last[name] = (s, expr, f"hole of {fn.shown}:{row.line} at {sn.run.records[i].span}")
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
        ways = _ways_through(held, name, text) if trace.renders_as_str(s, expr) else None
        turned = "; ".join(f"{ast.unparse(c.site)} at {c.site.lineno}" for c in held.conversions)
        if ways is not None and len(ways) == 1:
            env.put(name, ways[0], f"{source}, through {turned}")
            put[name] = text
        elif ways and several is not None:
            several[name] = ways
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


#: :func:`_row_held`'s answers, by the read's place, its start and its end: ``job`` and ``job.jobStatus`` start alike
_HELD: dict[tuple[str, int, int, int, int], Any] = {}


def _row_held(calls: Any, rel: str, func: ast.AST, read: ast.expr, bound: tuple) -> Any:
    """The writes of *read* reaching it (``fills.row_fills``), or None where the map cannot read them."""
    key = (rel, read.lineno, read.col_offset, read.end_lineno or 0, read.end_col_offset or 0)
    if key not in _HELD:
        try:
            _HELD[key] = fills.row_fills(calls, rel, func, read, bound)  # type: ignore[arg-type]
        except ValueError:
            _HELD[key] = None
    return _HELD[key]


def _from_the_row(held: Any, column: str, keys: set[str]) -> bool:
    """Whether *held* has fills and each fills the value from one row of a statement testing a key column equal to a
    bind, as *column*, with no in-place change on the way; its others are not read here."""
    return held is not None and bool(held.fills) and all(
        fill.pick in (fills.PICK_ONE, fills.PICK_AT) and not fill.changes and fill.reads and all(
            one.column.lower() == column
            and any(t.test.kind == "equal" and t.test.column.lower() in keys and t.binds for t in one.tests)
            for one in fill.reads)
        for fill in held.fills)


def _row_filled(calls: Any, rel: str, func: ast.AST, read: ast.expr, bound: tuple, column: str, keys: set[str]) -> bool:
    """Whether every write of *read* reaching it fills it from one row of a statement testing a key column equal to a
    bind, as *column*, with no in-place change on the way (``fills.row_fills``)."""
    held = _row_held(calls, rel, func, read, bound)
    return held is not None and not held.others and _from_the_row(held, column, keys)


def printed_row_column(fn: R.Function, line: int, attr: str, keys: set[str]) -> bool:
    """M3(a)-2 step 1: whether every read of ``.attr`` the statement at *line* of *fn* makes -- a print's hole -- is
    filled from one row of a statement testing a key column (*keys*, lower case) equal to a bind, as the column
    *attr* (:func:`_row_filled`, K71's reading): Watcher.run's ``job.modificationTime`` at 78, *job* filled by
    peekJob from the row of ``PandaID=:PandaID``.  False where it makes no such read or the map's types cannot be
    asked."""
    try:
        types = asked._types()
        tree = types.tree(fn.rel)
        if tree is None:
            return False
        func = next(f for f, _ in pathcond.functions_with_owner(tree) if f.lineno == fn.node.lineno and f.name == fn.name)
        calls = PR._calls_index(types)
        owned = types.method_class(fn.rel, func)
    except (StopIteration, LookupError, ValueError):
        return False
    stmt = R.statement_at(func, line)
    if stmt is None:
        return False
    bound = tuple(calls.objects_of(owned)) if owned else ()
    reads = [n for n in ast.walk(stmt) if isinstance(n, ast.Attribute) and n.attr == attr and isinstance(n.ctx, ast.Load)]
    return bool(reads) and all(_row_filled(calls, fn.rel, func, n, bound, attr.lower(), keys) for n in reads)


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


def _ways_through(held: pathcond.HeldSince, name: str, text: str) -> list[Any] | None:
    """The values a conversion-turned local can hold where it is read, each once, or None where a way's value is not
    known: each value the printed *text* can be (itself, and its literal where that prints as the text) takes a way --
    a decided conversion's where its tests hold, the shown value's where none does, both where that is not known.  One
    value is what :func:`printed_before` puts.  killJob's ``code=2``: an int 2 takes ``code = str(code)`` under
    ``isinstance(code, int)``, the text "2" the other way, and both are ``"2"``; ``code=None``: None and "None"."""
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
            return None
        got.update({(type(v).__name__, repr(v)): v for v in values})
    return list(got.values())


def bracket_of(values: list[tuple[int, str, str]], observed: str) -> tuple[int, int] | None:
    """The last record still showing another value, and the first after it showing *observed*."""
    for (i, v, _), (j, w, _) in zip(values, values[1:], strict=False):
        if v != observed and w == observed:
            return i, j
    return None


class Moment(NamedTuple):
    """A moment read as a run's span (``first`` = ``last``, a record's stamp): the row's time, placed in a call of a pass
    that runs no run of its own -- recordStatusChange, which prints nothing, at updateJobStatus's 484 (KNOWN_ISSUES.md
    K76)."""

    first: str
    last: str


def inside_call(caller: R.Seen, inner: Run | Moment, site: str) -> tuple[int, int] | None:
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


def _between_records(caller: R.Seen, inner: Run | Moment, stmt: ast.stmt) -> tuple[int, int] | None:
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


#: what :func:`_constant_of` answers for a binding whose value no constant spells
_NOT_CONSTANT = object()


def _constant_of(site: ast.AST) -> Any:
    """The constant *site* binds its names to (``x = "prepared"``); ``_NOT_CONSTANT`` for any other binding -- a loop's
    target, an unpacking, a value computed."""
    if isinstance(site, (ast.Assign, ast.AnnAssign)) and isinstance(site.value, ast.Constant):
        targets = site.targets if isinstance(site, ast.Assign) else [site.target]
        if all(isinstance(t, ast.Name) for t in targets):
            return site.value.value
    return _NOT_CONSTANT


def _loops_holding(node: ast.AST, func: ast.AST) -> list[ast.stmt]:
    """The loops of *func* whose body holds *node*, innermost first (as ``rows.loops_over`` reads them)."""
    out: list[ast.stmt] = []
    previous = node
    for ancestor in pathcond.ancestors(node):
        if ancestor is func:
            break
        if isinstance(ancestor, R.LOOPS) and any(previous is s for s in ancestor.body):
            out.append(ancestor)
        previous = ancestor
    return out


def turn_start(fn: R.Function, site: ast.stmt, loop: ast.stmt, raised_at: set[int] | None) -> R.Row | None:
    """The row every turn of *loop* that runs *site* prints before it, first in the loop's body: of *site*'s must rows
    before it (``rows.must_rows``), the first whose innermost loop is *loop*; None where none is.  A turn's records run
    from its record to the next one's: a turn that does not print it does not run *site* either."""
    before, _ = R.must_rows(fn.node, site, raised_at)
    own = [s for s in before if fn.row_at(s) is not None and next(iter(_loops_holding(s, fn.node)), None) is loop]
    first = min(own, key=lambda s: (s.lineno, s.col_offset), default=None)
    return fn.row_at(first) if first is not None else None


def _target_names(target: ast.expr) -> list[str]:
    """The names a ``for`` binds each turn: its target's, a tuple's elements' too (``for jediTaskID, commandStr,
    comComment in resList``)."""
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        return [name for element in target.elts for name in _target_names(element)]
    return []


class Turn(NamedTuple):
    """The key's turn of a loop in a run (:func:`key_turn`): records ``lo`` up to ``hi`` of the run."""

    lo: int
    hi: int
    start: R.Row
    loop: ast.stmt
    where: str


def key_turn(fn: R.Function, sn: R.Seen, site: ast.stmt, keys: set[str], raised_at: set[int] | None) -> Turn | None:
    """The key's turn around *site* in the run *sn* reads: of the ``for`` loops holding it, innermost first, the first
    whose :func:`turn_start` prints the question's key (``Env.keys``) in the hole of a name its target binds -- in the
    message, or in the mark of the logger it prints through (``rows.prefix_holes``) -- once in the run; the turn runs
    from that record to the start's next one.  None where no loop's start prints it, or one prints it twice.

    getTasksToExecCommand_JEDI's ``for jediTaskID, commandStr, comComment in resList`` binds the key with two other
    names; TaskCommando's runImpl prints it only in the mark of ``start for {commandStr}``, and its ``for iLoop in
    range(2)`` inside binds no key, so its turn is the outer loop's."""
    if not keys:
        return None
    records = sn.run.records
    for loop in _loops_holding(site, fn.node):
        if not isinstance(loop, (ast.For, ast.AsyncFor)) or not (names := _target_names(loop.target)):
            continue
        start = turn_start(fn, site, loop, raised_at)
        if start is None or start.stmt is site:
            continue
        starts = sorted(i for i, _, _ in sn.hits.get(start.line, ()))
        mine: list[tuple[int, str, str]] = []
        for i in starts:
            holes = {**R.prefix_holes(fn, start, records[i]), **start.capture(records[i])}
            held = next(((n, holes[n].strip()) for n in names if holes.get(n, "").strip() in keys), None)
            if held is not None:
                mine.append((i, *held))
        if not mine:
            continue
        if len(mine) != 1:
            return None
        lo, name, value = mine[0]
        hi = next((i for i in starts if i > lo), len(records))
        return Turn(lo, hi, start, loop, f"in the turn of {name}={value} (row {start.line} at {records[lo].span})")
    return None


def _key_values(env: E.Env | None) -> set[str]:
    """The question's key as the rows file holds it: the values of its key columns (``Env.keys``) in the row."""
    m = re.fullmatch(r"<(\w+)>\.\w+", env.subject) if env is not None else None
    if env is None or m is None:
        return set()
    row = env.table.get(m[1], {})
    return {str(held.value) for column in env.keys if (held := row.get(column)) is not None}


def _constructed_other(site: Any, attr: str, shown: str) -> bool:
    """P35 (c): whether *site* binds a local to a new object of a corpus class (``X = C()``) whose constructor binds
    *attr* only to literals, none of them *shown* -- the map's ``FieldFrom`` of the class's field, its ways whose
    binding lies in the class's ``__init__``.  JediTaskSpec's ``status`` is None or 0 there."""
    if not (isinstance(site, ast.Assign) and isinstance(site.value, ast.Call) and isinstance(site.value.func, ast.Name)):
        return False
    from bamboo.querier import sources as SR  # sources imports this module
    owners = sorted({c for c, f in SR._field_froms() if f == attr and c.endswith(f"::{site.value.func.id}")})
    if len(owners) != 1:
        return False
    values: set[str] = set()
    for w in SR.field_ways(owners[0], attr):
        site = w.route[0] if len(w.route) == 1 else ""
        rel, _, line = site.rpartition(":")
        if not line.isdigit() or "," in rel:
            continue
        try:
            node = R.def_holding(rel, int(line))
        except (OSError, SyntaxError):
            continue
        if node is None or node.name != "__init__":
            continue
        if w.end != SR.CONSTANT or w.steps or w.why in SR.NO_LITERAL or not w.value:
            return False
        try:
            values.add(str(ast.literal_eval(w.value)))
        except (ValueError, SyntaxError):
            return False
    return bool(values) and shown not in values


def _guard_false(fn: R.Function, site: Any, env: E.Env | None, sn: R.Seen) -> bool:
    """P35 (c): whether a guard *site* stands under (``pathcond.arm_condition``) is false with the values the pass
    printed (``stable_holes`` of its run, on a copy of *env* read at the site's line); a guard reading anything not
    printed is not false here.  TaskCommando's 181 under ``not (commandStr != 'reassign')``, commandStr printed
    'finish' at 133."""
    if not isinstance(site, ast.stmt):
        return False
    e = (env if env is not None else E.Env()).reading(fn.owner, site.lineno)
    stable_holes([sn], e, class_of(fn), local_to=fn.owner)
    for guard in pathcond.arm_condition(site):
        try:
            got = E.evaluate(guard, e)
        except (SyntaxError, ValueError, TypeError):
            continue
        if got is not E.UNKNOWN and not got:
            return True
    return False


def _record_of_row(fn: R.Function, sn: R.Seen, row: R.Row, alike: list, r_i: int) -> bool:
    """P35 (c): whether record *r_i*, which the rows *alike* of *fn* all find, is *row*'s: the last record before it only
    one row finds (``Seen.unambiguous``) is a print from which *row* can run with no print between, and none of the
    others can (``flow.runs_after``).  TaskCommando's 200 and 323 both print ``set task_status=...``; after 188's
    'updating datasets to finish', only 200 is next."""
    others = [rw for rw in alike if rw is not row]
    before = [(i, rw) for rw in fn.rows for i in sn.unambiguous.get(rw.line, ()) if i < r_i]
    if not others or not before:
        return False
    _, prev = max(before, key=lambda x: x[0])
    prints = [rw.stmt for rw in fn.rows]
    try:
        return flow.runs_after(prev.stmt, row.stmt, prints, printed=True) and not any(
            flow.runs_after(prev.stmt, rw.stmt, prints, printed=True) for rw in others)
    except (ValueError, AssertionError):
        return False


def printed_local(fn: R.Function, sn: R.Seen, stmt: ast.stmt, observed: str, env: E.Env | None,
                  raised_at: set[int] | None) -> str:
    """Why the arm *stmt*, binding a local to the observed value as a constant (``newTaskStatus = "prepared"``), ran in
    the pass *sn* reads; '' where that is not read (P1, KNOWN_ISSUES.md K72).

    A row *R* of *fn* printed the local with the observed value in the pass: in the key's turn around *R*
    (:func:`key_turn`) -- or, *R* in no loop, in a run that is one call (``Seen.single_call``);
    one record of *R* there, placed.  *Q*, the latest placed record before it in the turn or call from which every way
    to *R* binds the local (the map's ``flow.runs_after``, the local's bindings and the turn start as stops): what *R*
    printed was bound after *Q*, in the same turn.  Without such a record, *R* in no loop, the call's entry, where no
    way from it reaches *R* unbound (``pathcond.held_at``).  Of the bindings that can be what *R* reads
    (``reaching.reaching``), the ones run after *Q* within the turn may have given it; of those, the ones that can give
    the shown value -- a constant equal to it, or a value no constant spells -- are *stmt* alone.

    The map's reaching reads no path's condition, so it is asked from *Q*, a print of the turn, not from the loop's
    head: ``prepareTasksToBeFinished_JEDI``'s loop at 3852 does not bind ``newTaskStatus`` again at its head, and the
    ways skipping 3906's and 4015's bodies, open only when ``toSkip`` is True, reach 4074 unbound."""
    # P35 (c) (out/gm/qk-design.md, P35 (c)'s walk): a value no constant spells is read too -- the row shows what was
    # bound -- and an attribute of a local (``tmpTaskSpec.status = commandStatusMap()[commandStr]["done"]``, then 200's
    # ``set task_status={tmpTaskSpec.status}``) as a local is
    if (value := _constant_of(stmt)) is not _NOT_CONSTANT and str(value) != observed:
        return ""
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
        target = stmt.targets[0]
    elif isinstance(stmt, ast.AnnAssign):
        target = stmt.target
    else:
        return ""
    func = fn.node
    if isinstance(target, ast.Name):
        name = target.id
        if any(isinstance(n, (ast.Global, ast.Nonlocal)) and name in n.names for n in R.own_nodes(func)):
            return ""
        # a binding's site is a statement, a handler or a ``:=`` (``pathcond.Binding``): the flow reads each as
        # reaching's writes do, though it is typed for statements
        binds: list[Any] = list({id(b.site): b.site for b in pathcond.bindings(func, name) if not b.nested}.values())
    elif isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
        name, base = ast.unparse(target), target.value.id
        if any(isinstance(n, (ast.Global, ast.Nonlocal)) and base in n.names for n in R.own_nodes(func)):
            return ""
        # the attribute's own assignments, and every binding of the local it hangs on
        sites = [n for n in R.own_nodes(func) if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign))
                 and any(ast.unparse(t) == name for t in (n.targets if isinstance(n, ast.Assign) else [n.target]))]
        binds = list({id(x): x for x in [*sites, *(b.site for b in pathcond.bindings(func, base) if not b.nested)]}.values())
    else:
        return ""
    records = sn.run.records
    keys = _key_values(env)
    for row in fn.rows:
        hits = sorted(i for i, _, _ in sn.hits.get(row.line, ()))
        if not hits or pathcond.enclosing_function(row.stmt) is not func:
            continue
        loops = _loops_holding(row.stmt, func)
        turn_stops: list[ast.stmt] = []
        if loops:
            if (turn := key_turn(fn, sn, row.stmt, keys, raised_at)) is None:
                continue
            lo, hi, where = turn.lo, turn.hi, turn.where
            turn_stops = [turn.start.stmt]
        elif sn.single_call():
            lo, hi, where = -1, len(records), "in the call"
        else:
            continue
        at = [i for i in hits if lo < i < hi]
        if len(at) != 1:
            continue
        r_i = at[0]
        # P35 (c): a record two rows of the function find alike (TaskCommando's 199 sendMsg and 200 info of one tmpMsg)
        # reads as either, where each of them prints the bound name
        alike = [rw for rw in fn.rows if any(i == r_i for i, _, _ in sn.hits.get(rw.line, ()))]
        if r_i not in sn.unambiguous.get(row.line, ()) and not (
                len(alike) > 1 and all(any(e in (name, f"str({name})") for e in rw.capture(records[r_i])) for rw in alike)
        ) and not _record_of_row(fn, sn, row, alike, r_i):
            continue
        shown = next((v.strip() for e, v in row.capture(records[r_i]).items() if e in (name, f"str({name})")), None)
        if shown != observed:
            continue
        stops = [*binds, *turn_stops]
        placed = sorted(((i, rw) for rw in fn.rows if rw is not row and pathcond.enclosing_function(rw.stmt) is func
                         for i in sn.unambiguous.get(rw.line, ()) if lo <= i < r_i), key=lambda x: -x[0])
        q: tuple[int, R.Row] | None = None
        asked_rows: set[int] = set()
        for i, rw in placed:
            if rw.line in asked_rows:
                continue
            asked_rows.add(rw.line)
            if not flow.runs_after(rw.stmt, row.stmt, stops, printed=True):
                q = (i, rw)
                break
        if q is not None:
            reached = {id(b) for b in reaching.reaching(row.stmt, binds)}
            sources = [b for b in binds if id(b) in reached and flow.runs_after(q[1].stmt, b, turn_stops, printed=True)]
            since = f"row {q[1].line} (at {records[q[0]].span})"
        elif not loops and isinstance(target, ast.Name):
            held = pathcond.held_at(func, name, row.stmt)  # type: ignore[arg-type]
            if any(getattr(s, "lineno", None) is None for s in held.sites):
                continue
            sources, since = list(held.sites), "the call's entry"
        else:
            continue
        givers = [b for b in sources if (c := _constant_of(b)) is _NOT_CONSTANT or str(c) == shown]
        # P35 (c): a giver under a guard the pass's printed values make false did not run (TaskCommando's 181
        # ``tmpTaskSpec.status = oldStatus`` in the reassign branch, commandStr printed 'finish')
        givers = [b for b in givers if b is stmt or not _guard_false(fn, b, env, sn)]
        # P35 (c): a binding of the local an attribute hangs on to a new object of a corpus class gives the attribute
        # what that class's constructor binds it to (TaskCommando's 154 ``tmpTaskSpec = JediTaskSpec()``: None or 0)
        if "." in name:
            attr = name.rpartition(".")[2]
            givers = [b for b in givers if b is stmt or not _constructed_other(b, attr, shown)]
        if len(givers) == 1 and givers[0] is stmt:
            lines = sorted(getattr(b, "lineno", 0) for b in sources)
            return (f"P1: row {row.line} printed {name}={shown!r} {where}, at {records[r_i].span}; every way from {since} to "
                    f"it binds {name}, and of the bindings run since ({lines}) only {stmt.lineno} gives {shown!r}")
    return ""


_WP: list = [None, {}]


def write_prints(owner: str) -> list:
    """The installed dump's ``write_prints`` of *owner* (``file::function``, a method named without its class): the
    map's ``WritePrint``, a write statement whose function prints its text and the bind map it runs with, and the
    column each bind writes.  Empty before an install or from a dump that has none."""
    wps = getattr(mapdump.fragment(), "write_prints", None) or []
    if _WP[0] is not wps:
        index: dict[str, list] = {}
        for wp in wps:
            index.setdefault(_bare_owner(wp.owner), []).append(wp)
        _WP[:] = [wps, index]
    held: dict[str, list] = _WP[1]
    return held.get(_bare_owner(owner), [])


def _bare_owner(owner: str) -> str:
    return R.named(owner)


class PrintedBinds(NamedTuple):
    """A write statement's print read in the key's turn (:func:`printed_binds`)."""

    row: R.Row
    at: int
    #: the bind map it printed, as the map cuts it (``sqltext.cut``)
    binds: dict[Any, Any]
    #: the map's entries of the printed map (``SkeletonHole.entries``), one per key and line
    entries: list[Any]
    #: whether the map read every way the map is filled (no ``entries_open``): a key it lacks was not filled
    whole: bool
    #: the key's turn; None where the run is one call and the print in no loop
    turn: Turn | None
    where: str


def printed_binds(fn: R.Function, sn: R.Seen, wp: Any, keys: set[str], raised_at: set[int] | None) -> PrintedBinds | None:
    """The one record of a print of *wp*'s statement with the map it runs with (``WritePrint.printed_by``) in the key's
    turn (:func:`key_turn`) -- or, the print in no loop, in a run that is one call -- placed there, cut as the map cuts
    it; None where none is."""
    for p in wp.printed_by:
        stmt = R.statement_at(fn.node, p.line)
        row = fn.row_at(stmt) if stmt is not None else None
        if row is None:
            continue
        turn: Turn | None = None
        if _loops_holding(row.stmt, fn.node):
            if (turn := key_turn(fn, sn, row.stmt, keys, raised_at)) is None:
                continue
            lo, hi, where = turn.lo, turn.hi, turn.where
        elif sn.single_call():
            lo, hi, where = -1, len(sn.run.records), "in the call"
        else:
            continue
        at = [i for i, _, _ in sn.hits.get(row.line, ()) if lo < i < hi]
        if len(at) != 1 or at[0] not in sn.unambiguous.get(row.line, ()):
            continue
        got = sqltext.cut(sn.run.records[at[0]].text, p.lead, p.gap)
        if got is None:
            continue
        holes = [h for m in row.map_rows for h in getattr(m, "holes", ()) if getattr(h, "entries", None) is not None]
        entries = list({(e.key, e.line): e for h in holes for e in h.entries}.values())
        whole = bool(holes) and not any(getattr(h, "entries_open", None) for h in holes)
        return PrintedBinds(row, at[0], got[1], entries, whole, turn, f"{where}, at {sn.run.records[at[0]].span}")
    return None


def _value_match(value: Any, text: str) -> bool | None:
    """Whether a value a statement ran with is the row's value *text*: True, False, or None where it may be either.

    P66 (the investigator's u9 W12): a time printed to the microsecond is the row's value at the precision the row
    holds it.  A DATE column keeps its seconds, the fraction cut or rounded (which, is not read here), so a fraction
    below half a second is the same second either way: updateTask_JEDI's 1483 printed ``':modificationTime':
    datetime.datetime(2026, 11, 15, 4, 37, 31, 304478)``, the row's '2026-11-15 04:37:31'.  From half a second on,
    either of the two seconds may be it."""
    if isinstance(value, datetime.datetime):
        try:
            seen = datetime.datetime.fromisoformat(str(text).strip())
        except ValueError:
            return str(value).strip() == str(text).strip()
        if value == seen:
            return True
        if seen.microsecond or not value.microsecond:
            return False
        cut = value.replace(microsecond=0)
        if value.microsecond < 500000:
            return cut == seen
        return None if seen in (cut, cut + datetime.timedelta(seconds=1)) else False
    return str(value).strip() == str(text).strip()


def _same_value(value: Any, text: str) -> bool:
    """:func:`_value_match` where it is sure the value is the row's."""
    return _value_match(value, text) is True


def _names_read(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _entry_names(fn: R.Function, stmt: ast.stmt, shown: PrintedBinds, loops: list[ast.stmt]) -> list[tuple[str, Any, str]]:
    """``(name, value, source)`` for each name an entry of the printed map is filled from alone (``':status'`` <-
    ``newTaskStatus`` at 4460), with the value printed, where *stmt*'s tests read it as the store left it: *stmt* can
    run after the store in the turn, no binding of the name can, and no test around *stmt* reading it is entered
    before the store."""
    out = []
    for e in shown.entries:
        if e.key not in shown.binds or getattr(e, "computed", False) or sum(1 for o in shown.entries if o.key == e.key) != 1:
            continue
        try:
            node = ast.parse(e.expression, mode="eval").body
        except SyntaxError:
            continue
        if not isinstance(node, ast.Name) or (store := R.statement_at(fn.node, e.line)) is None:
            continue
        name = node.id
        sites = [b.site for b in pathcond.bindings(fn.node, name) if not b.nested]
        # a handler's or a ``:=``'s binding is not read here: the name is left out
        if not all(isinstance(s, ast.stmt) for s in sites):
            continue
        if any(flow.runs_after(store, s, once=loops) for s in sites if isinstance(s, ast.stmt)) \
                or not flow.runs_after(store, stmt, once=loops):
            continue
        around_store = {id(a) for a in pathcond.ancestors(store)}
        tests = [a for a in pathcond.ancestors(stmt) if isinstance(a, (ast.If, ast.While)) and id(a) in around_store]
        if any(name in _names_read(a.test) for a in tests):
            continue
        out.append((name, shown.binds[e.key], f"entry {e.key} of row {shown.row.line}'s map, filled at {e.line}"))
    return out


def _bind_value(stmt: ast.stmt, bind: str, shown: PrintedBinds) -> ast.expr | None:
    """What the arm *stmt* gives the bind: the value of its store into the map (``varMap[":status"] =
    taskSpec.status``), or of its binding of the name the bind's entry is filled from (``newTaskStatus =
    taskOldStatus``); None where it gives it neither way."""
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
        target, value = stmt.targets[0], stmt.value
    elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
        target, value = stmt.target, stmt.value
    else:
        return None
    if isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant) and target.slice.value == bind:
        return value
    if isinstance(target, ast.Name) and any(e.key == bind and e.expression == target.id for e in shown.entries):
        return value
    return None


def _not_by(fn: R.Function, sn: R.Seen, stmt: ast.stmt, shown: PrintedBinds, bind: str, observed: str, env: E.Env,
            mode: str, raised_at: set[int] | None) -> str:
    """Why the arm *stmt* did not give the printed bind its value in the turn *shown* read it in; '' where that is not
    read.  An entry the printed map lacks, filled on every way from the arm to the print; the value it gives the bind,
    read in the turn's values; a test it runs under, false in them.  The turn's values: its own locals printed once
    in it (:func:`stable_holes` for the function), the ones printed before the arm on every way (:func:`printed_before`),
    and the names the printed map's entries are filled from (:func:`_entry_names`)."""
    loops = [shown.turn.loop] if shown.turn is not None else []
    row = shown.row
    if shown.whole and flow.runs_after(stmt, row.stmt, once=loops):
        lacking: dict[str, list[int]] = {}
        for e in shown.entries:
            if e.key not in shown.binds and not getattr(e, "computed", False):
                lacking.setdefault(e.key, []).append(e.line)
        for key, lines in sorted(lacking.items()):
            stores = [s for n in sorted(lines) if (s := R.statement_at(fn.node, n)) is not None]
            if len(stores) == len(lines) and not flow.runs_after(stmt, row.stmt, stores, once=loops):
                return (f"P35: row {row.line} printed a map with no {key} {shown.where}, and every way from it to the "
                        f"print fills {key} (at {sorted(lines)})")
    tenv = env.reading(fn.owner, stmt.lineno, arm=(fn.owner, stmt.lineno))
    turn = sn.run if shown.turn is None else sn.run.part(shown.turn.lo, shown.turn.hi)
    tsn = R.seen(fn, turn, mode)
    stable_holes([tsn], tenv, class_of(fn), local_to=fn.owner)
    before, _ = R.must_rows(fn.node, stmt, raised_at)
    printed_before(fn, tsn, stmt, before, tenv)
    for name, value, source in _entry_names(fn, stmt, shown, loops):
        if not tenv.knows(name):
            tenv.put(name, value, source)
    if (value := _bind_value(stmt, bind, shown)) is not None:
        try:
            got = E.evaluate(ast.unparse(value), tenv)
        except SyntaxError:
            got = E.UNKNOWN
        if got is not E.UNKNOWN and _value_match(got, observed) is False:
            return f"P35: it gives {bind} {ast.unparse(value)}, {got!r} {shown.where} ({_used(tenv, ast.unparse(value))})"
    for guard in pathcond.arm_condition(stmt):
        if E.read_elsewhere(guard):
            continue
        try:
            if E.evaluate(guard, tenv) is False:
                return f"P35: its test {guard!r} is false {shown.where} ({_used(tenv, guard)})"
        except SyntaxError:
            continue
    return ""


def _used(env: E.Env, text: str) -> dict[str, Any]:
    return {k: v for k, v in env.used.items() if k in text}


def printed_write_arms(fn: R.Function, sn: R.Seen, verdicts: list[Verdict], subject: str, observed: str, env: E.Env,
                       mode: str, raised_at: set[int] | None) -> None:
    """P35 (out/gm/qk-design.md, bundle B, tmaeno's OK 2026-11-16 21:40) on *verdicts*, the arms of *fn* standing: a
    write statement of *subject* (the map's ``WritePrint``, binding it to one bind) printed the bind map it ran with in
    the key's turn (:func:`printed_binds`), the bind holding the observed value.  An arm of the statement that did not
    give it that value there (:func:`_not_by`) is excluded; where one arm is left, it gave it, and is positive.

    getTasksToExecCommand_JEDI's 4486 printed ``':status': 'running'`` in NT-exhausted-late's task's turn: 4436
    (``newTaskStatus = taskOldStatus``) gives 'scouting' there (4386 printed ``old:scouting``), 4469's ``SET
    status=oldStatus`` runs under ``newTaskStatus == 'dummy'``, false with the ``:status`` entry's
    ``newTaskStatus='running'``; 4434 is left.  updateTaskStatusByContFeeder_JEDI's 1380 printed no ``:oldStatus``,
    which 1357 fills on every way from 1356 to it: 1356 did not run, and 1376 is left."""
    keys = _key_values(env)
    for wp in write_prints(fn.owner):
        arms = set(wp.arms.get(subject) or ())
        mine = [v for v in verdicts if v.line in arms and v.kind in ("kept", "positive")]
        binds = {w.bind for w in wp.binds if w.column == subject}
        if not mine or len(binds) != 1:
            continue
        bind = binds.pop()
        shown = printed_binds(fn, sn, wp, keys, raised_at)
        if shown is None or bind not in shown.binds or not _same_value(shown.binds[bind], observed):
            continue
        for v in mine:
            stmt = R.statement_at(fn.node, v.line)
            if v.kind != "kept" or stmt is None:
                continue
            if shown.turn is not None and not any(lp is shown.turn.loop for lp in _loops_holding(stmt, fn.node)):
                continue
            if why := _not_by(fn, sn, stmt, shown, bind, observed, env, mode, raised_at):
                v.kind = "excluded"
                v.reasons.append(why)
        left = [v for v in mine if v.kind != "excluded"]
        if len(left) == 1 and left[0].kind == "kept":
            out = sorted(v.line for v in mine if v is not left[0])
            left[0].kind = "positive"
            left[0].reasons.append(
                f"P35: row {shown.row.line} printed the map {wp.statement}'s statement ran with {shown.where}, "
                f"{bind}={observed!r}; of its arms of {subject}, the others ({out}) did not give it there")


def branches_at(subject: str, owner: str, line: int, fragment: Any = None) -> list:
    """The map's branches of *subject* at *owner*'s *line*, in *fragment* (the installed dump's where None).  A
    reading's owner is ``file::function``, a junction's ``file::Class.function``: matched by the file, the function's
    name and the line."""
    fragment = mapdump.fragment() if fragment is None else fragment
    file, _, qual = owner.partition("::")
    name = qual.rpartition(".")[2]
    return [b for j in (getattr(fragment, "junctions", None) or []) if j.subject == subject
            and j.owner.partition("::")[0] == file and j.owner.partition("::")[2].rpartition(".")[2] == name
            for b in j.branches if b.line == line]


def hole_values(row: R.Row, record: Any, name: Any) -> list[str]:
    """What *record*, a line of the print *row*, holds in the hole the map's name *name* reads (``MustName``): an
    entry of its printed bind map (``sqltext.cut``: ``str(var_map)`` after the statement), or a hole of its logger's
    mark or of its message as the map's rows read them back (``trace.hole_values``, the record whole first), each
    stripped of its quotes -- ``updateTask_JEDI < jediTaskID=53637838 >``'s ``taskSpec.jediTaskID``."""
    if name.part == MUST_PART_BIND:
        got = sqltext.cut(record.text)
        values = [got[1].get(name.bind)] if got is not None else []
    else:
        values = []
        for m in row.map_rows:
            for whole in (True, False):
                read = trace.hole_values(m, record.text, record.token, whole=whole)
                if read is None:
                    continue
                holes, printed = (m.prefix_holes, read[1]) if name.part == MUST_PART_MARK else (m.holes, read[0])
                values += [v for h, v in zip(holes, printed, strict=False) if h.expression == name.expression]
                break
    return [str(v).strip().strip("'\"") for v in values if v is not None]


def after_key_others(fn: R.Function, sn: R.Seen, verdicts: list[Verdict], subject: str, env: E.Env) -> None:
    """B-5 (P36, out/gm/qk-design.md's bundle B, tmaeno's OK 2026-11-16 21:40) on *verdicts*, kept arms of *fn* writing
    *subject* in the run *sn*: an arm is excluded where a print the map says runs after its write on every way, the
    object not bound again between (``Branch.after_prints``: ``every_way``, not ``rebound``), holds the written row's
    key column (a name with ``row`` written, a column of the question's key), every loop holding the arm holds the print
    too (a print after the loop shows its last object only), and each of its records in the run -- one at least -- holds
    a value of that hole other than the question's key: no turn of the arm wrote the key's row.  Not where a traceback
    in the run names *fn* (a raise may end a way before the print).  updateUnmergedJobs's 1236 ``umJob.jobStatus =
    job.jobStatus`` runs 50 times in NJ-holding-late's run, each turn's 1252 ``update unmerged PandaID={umJob.PandaID}``
    printing another job's ID.  K64 (:func:`other_objects`) reads such a print the same way where it is one of the
    arm's own must rows, for rule 4; 1252 is not one, and the map says it runs after 1236 on every way."""
    keys = _key_values(env)
    if not keys or R.raised_in(fn, sn.run)[0]:
        return
    # the key's columns under the class the map names them by (env.subject holds the row's name, "<job>.jobStatus")
    columns = {f"{subject.partition('.')[0]}.{k}" for k in env.keys}
    for v in verdicts:
        stmt = R.statement_at(fn.node, v.line) if v.kind == "kept" else None
        if stmt is None:
            continue
        for b in branches_at(subject, fn.owner, v.line):
            for p in getattr(b, "after_prints", None) or ():
                names = [n for n in p.names if n.row == MUST_ROW_WRITTEN and n.column in columns]
                row = next((r for r in fn.rows if r.line == p.line), None)
                if (not p.every_way or p.rebound or not names or row is None
                        or not R.loops_over(stmt, fn.node) <= R.loops_over(row.stmt, fn.node)):
                    continue
                at = [i for i, _, _ in sn.hits.get(row.line, ())]
                held = [hole_values(row, sn.run.records[i], n) for i in at for n in names]
                if held and all(vals and not keys & set(vals) for vals in held):
                    v.kind = "excluded"
                    v.reasons.append(f"P36: row {p.line} prints the written row's {names[0].column} after it on every way, "
                                     f"and its {len(at)} record(s) in the run hold another than the key's: it wrote no "
                                     f"row of the key")
                    break
            if v.kind == "excluded":
                break


#: the installed dump's handler records by owner and ``except`` line (:func:`_handler_marks`), with the fragment read
_MARKS: list[tuple[Any, dict[str, dict[int, Any]]]] = []
_TRIES: tuple[type, ...] = (ast.Try, getattr(ast, "TryStar", ast.Try))


def _handler_marks(owner: str) -> dict[int, Any]:
    """The map's records of *owner*'s ``except`` handlers (``MapFragment.handler_marks``), by line; empty with no dump."""
    fragment = mapdump.fragment()
    if fragment is None:
        return {}
    if not _MARKS or _MARKS[0][0] is not fragment:
        by: dict[str, dict[int, Any]] = {}
        for one in fragment.handler_marks:
            by.setdefault(one.owner, {})[one.line] = one
        _MARKS[:] = [(fragment, by)]
    return _MARKS[0][1].get(owner, {})


def _tries(func: ast.AST) -> list[Any]:
    """The ``try`` statements of *func*'s own body, ``try``/``except*`` too where the Python has it."""
    return [t for t in R.own_nodes(func) if isinstance(t, _TRIES)]


def _unshown(sn: R.Seen, lines: list[int]) -> bool:
    """Whether no row of *sn*'s function at *lines* is in its run, each where the run would show it
    (``Seen.would_show``)."""
    for line in lines:
        row = next((r for r in sn.fn.rows if r.line == line), None)
        if row is None or sn.present(row) or not sn.would_show(row):
            return False
    return True


def _not_entered(sn: R.Seen, mark: Any) -> bool:
    """Whether the run *sn* reads shows the call did not enter the handler *mark* records: every way through it passes
    one of its own prints (the map's ``HandlerMark.prints_on_every_way``), and none of them is in the run where the
    run would show it.  A marking call's prints are not read: the key's run need not hold them.  killJob's 1161 calls
    ``_rollback``, which writes to ``self._log_stream`` with no key, and ``dump_error_message``, whose 319 prints a
    traceback the reader cannot read; its own 1170 (``time=...``) is on every way.  As the map reads every way, a raise
    inside the handler that leaves it is no exit: a ``dump_error_message`` that raised would leave 1161 without 1170."""
    return mark is not None and bool(mark.prints_on_every_way) and _unshown(sn, mark.prints)


def _holds(outer: ast.AST, node: ast.AST) -> bool:
    return any(a is outer for a in pathcond.ancestors(node))


def _head_ran(fn: R.Function, sn: R.Seen, head: ast.stmt) -> str:
    """Why the one call *sn* reads ran the statement *head*, or '': a row the call printed is reached from an earlier
    placed record (or the call's entry) only through *head*; or the value a row printed of a local, bound on every way
    from such a record to it, can be given only by bindings reached from there only through *head* (the givers, as
    :func:`printed_local` reads them).  The map's flow from a print (``flow.runs_after``, *head* a stop)."""
    func, records = fn.node, sn.run.records
    placed = sorted(((i, rw) for rw in fn.rows if pathcond.enclosing_function(rw.stmt) is func
                     for i in sn.unambiguous.get(rw.line, ())), key=lambda x: x[0])
    for k, (r_i, row) in enumerate(placed):
        at = f"row {row.line} at {records[r_i].span}"
        if _holds(head, row.stmt):
            return f"{at} printed inside it"
        froms: list[tuple[Any, str]] = [(rw.stmt, f"row {rw.line} (at {records[i].span})") for i, rw in reversed(placed[:k])]
        froms.append((func.args, "the call's entry"))
        since = next((s for first, s in froms if not flow.runs_after(first, row.stmt, [head], printed=True)), None)
        if since is not None:
            return f"{at} printed, and every way from {since} to it passes {head.lineno}"
        for expr, text in row.capture(records[r_i]).items():
            inner = re.fullmatch(r"str\((\w+)\)", expr)
            name = inner[1] if inner else expr
            if not name.isidentifier():
                continue
            binds: list[Any] = list({id(b.site): b.site for b in pathcond.bindings(func, name) if not b.nested}.values())
            # from the latest record (or the entry) every way from which to the row binds the name
            q = next(((first, s) for first, s in froms if not flow.runs_after(first, row.stmt, binds, printed=True)), None)
            if not binds or q is None:
                continue
            shown = text.strip()
            reached = {id(b) for b in reaching.reaching(row.stmt, binds)}
            givers = [b for b in binds if id(b) in reached and flow.runs_after(q[0], b, printed=True)
                      and ((c := _constant_of(b)) is _NOT_CONSTANT or str(c) == shown)]
            if givers and not any(flow.runs_after(q[0], b, [head], printed=True) for b in givers):
                lines = sorted(getattr(b, "lineno", 0) for b in givers)
                return (f"{at} printed {name}={shown!r}, which of the bindings run since {q[1]} only {lines} can give, "
                        f"every way to them passing {head.lineno}")
    return ""


#: how many readings of a test :func:`_truth` reads at most
_READINGS = 16


def _truth(test: ast.expr, env: E.Env, several: dict[str, list[Any]]) -> tuple[Any, str]:
    """``(value, how)``: *test*'s value with *env*, read under every value *several* gives a name it reads
    (:func:`printed_before`) -- the one answer they all give, or ``UNKNOWN``; *how* says which names were so read."""
    text = ast.unparse(test)
    names = sorted(n for n in several if re.search(rf"(?<![\w.]){re.escape(n)}(?!\w)", text))
    readings = list(itertools.product(*(several[n] for n in names)))
    if len(readings) > _READINGS or (names and env.reader is None):
        return E.UNKNOWN, ""
    answers: set[bool] = set()
    for values in readings:
        # each reading on a copy of the env (Env.reading), what else it used kept on the env
        e = env.reading(*env.reader) if names and env.reader is not None else env
        for name, value in zip(names, values, strict=True):
            e.put(name, value, "one of the values its print can be")
        try:
            got = E.evaluate(text, e)
        except SyntaxError:
            return E.UNKNOWN, ""
        if e is not env:
            env.used.update({k: v for k, v in e.used.items() if k not in names})
        if got is E.UNKNOWN:
            return E.UNKNOWN, ""
        answers.add(bool(got))
    if len(answers) != 1:
        return E.UNKNOWN, ""
    how = "; ".join(f"{n} read as each of {several[n]!r}" for n in names)
    return answers.pop(), how


def ran_down_to(fn: R.Function, sn: R.Seen, psn: R.Seen, stmt: ast.stmt, env: E.Env, raised_at: set[int] | None) -> str:
    """Why the kept arm *stmt* ran in the one call *psn* reads (its run less the rows printed for other objects; *sn*
    the run whole), or '' where that is not read (J, KNOWN_ISSUES.md K73).

    - *The head* H: of the ``if`` s holding *stmt*, outermost first, the first the call is shown to have run
      (:func:`_head_ran`), with no loop, handler or ``match`` between it and *stmt*.
    - *No raise between*: a ``try`` whose body holds H has a catch-all handler, and the call entered none of the
      handlers of the ``try`` s holding H or inside it before *stmt* (:func:`_not_entered`, the map's
      ``handler_marks``).
    - *The tests from H down* (``pathcond.guards``, H's own included, an early exit's on the way it stays) hold, each
      read at its own ``if`` with the values printed before it (:func:`printed_before`, a name its print shows as
      several values read under each, :func:`_truth`) and the question's column the pass filled from the row (K71):
      the statuslog's state before the pass, where each other write reaching the read is of this function and stands
      under a test those values make false.

    killJob in MJE-7397556704: the call printed 840 ``code=50``, 942 and 1145 ``kill=True``; of ``flagKilled``'s
    bindings only 1098 gives True, reached from 840 only through 1017.  1161's own prints (1164, 1170) are not in the
    call.  1017's ``job.jobStatus != "failed"`` holds with 'merging', the state before the pass, and so does 1071's
    ``job.jobStatus != "closed"``: 1028 .. 1063 assign 'closed' under ``code`` tests false with "50".  A run of two
    calls (MJE-7396802500: 840 is printed twice) is not read."""
    func = fn.node
    if pathcond.enclosing_function(stmt) is not func or not psn.single_call() or R.raised_in(fn, sn.run)[0]:
        return ""
    chain = [a for a in pathcond.ancestors(stmt) if isinstance(a, ast.If) and _holds(func, a)]
    marks = _handler_marks(f"{fn.rel}::{pathcond.qualified_name(func)}")
    envs: dict[int, tuple[E.Env, dict[str, list[Any]]]] = {}

    def env_at(at: ast.stmt) -> tuple[E.Env, dict[str, list[Any]]]:
        if at.lineno not in envs:
            e = env.reading(fn.owner, at.lineno, arm=(fn.owner, stmt.lineno))
            several: dict[str, list[Any]] = {}
            printed_before(fn, psn, at, R.must_rows(func, at, raised_at)[0], e, several)
            envs[at.lineno] = (e, several)
        return envs[at.lineno]

    for head in reversed(chain):
        between = [a for a in pathcond.ancestors(stmt) if _holds(head, a)]
        if any(isinstance(a, (*R.LOOPS, ast.ExceptHandler, ast.Match)) for a in between):
            continue
        ran = _head_ran(fn, psn, head)
        if not ran:
            continue
        tries = [t for t in _tries(func)
                 if any(_holds(s, head) or s is head for s in t.body) or (_holds(head, t) and t.lineno <= stmt.lineno)]
        if not any(reaching.catches_all(h) for t in tries if any(_holds(s, head) or s is head for s in t.body)
                   for h in t.handlers):
            return ""
        handlers = [h for t in tries for h in t.handlers]
        if not all(_not_entered(sn, marks.get(h.lineno)) for h in handlers):
            return ""
        tests = []
        for guard in pathcond.guards(stmt, func, head=head):
            e, several = env_at(guard.at)
            for text, (value, why) in _filled_at(fn, psn, guard, env_at).items():
                e.put(text, value, why)
            value, how = _truth(guard.test, e, several)
            if value is E.UNKNOWN or value != guard.runs_on:
                return ""
            used = {k: v for k, v in e.used.items() if k in ast.unparse(guard.test)}
            tests.append(f"{ast.unparse(guard.test)!r} {value} ({used}{'; ' + how if how else ''})")
        entered = sorted(h.lineno for h in handlers)
        return (f"J: the call ran {head.lineno} ({ran}); it entered none of the handlers {entered} around "
                f"{head.lineno}..{stmt.lineno} (their own prints not in the run); the tests from {head.lineno} down to "
                f"{stmt.lineno} hold: {'; '.join(tests)}")
    return ""


def _filled_at(fn: R.Function, sn: R.Seen, guard: pathcond.Guard,
               env_at: Callable[[ast.stmt], tuple[E.Env, dict[str, list[Any]]]]) -> dict[str, tuple[str, str]]:
    """The question's column where *guard*'s test reads it from the row the pass filled, as :func:`row_state` puts it,
    by its text: ``(value, source)``.  Each other write reaching the read (``fills.row_fills``' others) must be of this
    function, reached with no call crossed, and stand under a test (``pathcond.guards``) false with the values printed
    before that test (*env_at*, :func:`_truth`)."""
    env = env_at(guard.at)[0]
    if not env.subject or not env.statuslog:
        return {}
    column = env.subject.rpartition(".")[2].lower()
    keys = {k.lower() for k in env.keys}
    reads = {ast.unparse(r): r for r in _chains(guard.test) if env.canonical(ast.unparse(r)) == env.subject}
    if not reads or any(env.knows(text) for text in reads):
        return {}
    try:
        types = asked._types()
        tree = types.tree(fn.rel)
        if tree is None:
            return {}
        func = next(f for f, _ in pathcond.functions_with_owner(tree) if f.lineno == fn.node.lineno and f.name == fn.name)
        node = R.statement_at(func, guard.at.lineno)
        calls = PR._calls_index(types)
        owned = types.method_class(fn.rel, func)
    except (StopIteration, LookupError, ValueError):
        return {}
    if node is None:
        return {}
    bound = tuple(calls.objects_of(owned)) if owned else ()
    own_reads = {ast.unparse(r): r for r in _chains(node.test)} if isinstance(node, ast.If) else {}
    out: dict[str, tuple[str, str]] = {}
    for text in reads:
        read = own_reads.get(text)
        held = _row_held(calls, fn.rel, func, read, bound) if read is not None else None
        if held is None or not _from_the_row(held, column, keys):
            return {}
        off = []
        for other in held.others:
            site = R.statement_at(fn.node, getattr(other.site, "lineno", 0)) if not other.route and other.file in ("", fn.rel) else None
            if site is None or type(site) is not type(other.site):
                return {}
            why = ""
            for g in pathcond.guards(site, fn.node):
                value, how = _truth(g.test, *env_at(g.at))
                if value is not E.UNKNOWN and value != g.runs_on:
                    why = f"{site.lineno} under {ast.unparse(g.test)!r} {value}" + (f" ({how})" if how else "")
                    break
            if not why:
                return {}
            off.append(why)
        state, _ = state_before(sn.run, env.statuslog)
        if state is None:
            return {}
        when = state.get("time_extended") or state["time"]
        out[text] = (state["status"], f"the statuslog row before the pass, {state['status']!r} at {when}"
                     + (f"; the other writes reaching it do not run: {', '.join(off)}" if off else ""))
    return out


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


@functools.cache
def _handed_in(fragment_id: int) -> dict[tuple[str, int], list]:
    """``(file, line) -> [HeldSource]``: the ways the map reads into a callee's parameter (``ParameterFrom.sources``)
    by the call they end at, the last of each ``route`` (*fragment_id* keys the cache to the dump)."""
    out: dict[tuple[str, int], list] = {}
    for p in getattr(mapdump.fragment(), "parameter_froms", None) or []:
        for s in p.sources:
            file, _, at = (s.route[-1] if s.route else "").rpartition(":")
            if at.isdigit():
                out.setdefault((file, int(at)), []).append(s)
    return out


def appended_turns(fn: R.Function, line: int) -> list[tuple[ast.For | ast.AsyncFor, int, str]]:
    """``(loop, append line, element)`` for each list the call at *line* hands its callee that *fn* makes empty and
    fills only by appending the element of one of its loops the call is not in, each turn its own -- the map's
    ``HeldSource`` of the callee's parameter: its ``route`` the call alone, ``made_empty``, each of its ``changes`` a
    ``method`` ``append`` *fn* runs itself, its ``value_text`` the loop's target.  ActivatorWithRuleThr.run makes
    ``actJobs = []`` at 929, appends ``tmpJob`` at 969 in its loop over the jobs at 931, and hands the list to
    ``activateJobs`` at 973 (D-4).  Empty where the map reads no such way."""
    fragment = mapdump.fragment()
    call = R.statement_at(fn.node, line)
    if fragment is None or call is None:
        return []
    lo, hi = fn.node.lineno, getattr(fn.node, "end_lineno", None) or fn.node.lineno
    out: list[tuple[ast.For | ast.AsyncFor, int, str]] = []
    for s in _handed_in(id(fragment)).get((fn.rel, line), []):
        if not getattr(s, "made_empty", "") or len(s.route) != 1 or not lo <= (s.line or 0) <= hi or not s.changes:
            continue
        sites: list[tuple[ast.For | ast.AsyncFor, int, str]] = []
        for ch in s.changes:
            file, _, at = ch.at.rpartition(":")
            stmt = R.statement_at(fn.node, int(at)) if file == fn.rel and at.isdigit() else None
            loop = next((a for a in pathcond.ancestors(stmt) if isinstance(a, (ast.For, ast.AsyncFor))), None) if stmt else None
            if (loop is None or ch.kind != "method" or ch.what != "append" or ch.via or not lo <= loop.lineno <= hi
                    or not isinstance(loop.target, ast.Name) or loop.target.id != ch.value_text
                    or any(a is loop for a in pathcond.ancestors(call))):
                break
            sites.append((loop, int(at), ch.value_text))
        else:
            if len({id(loop) for loop, _, _ in sites}) == 1:
                out.extend(sites)
    return out


def _turn_hole(row: R.Row, element: str) -> str:
    """The hole a print of *row* begins with, where it is an attribute of the loop's *element* (``tmpJob.PandaID``):
    a logger with no mark writes its value first, the line's mark (:data:`records.LEADING_HOLE`); empty otherwise."""
    m = LEADING_HOLE.match(row.text or "")
    hole = m["hole"].strip() if m else ""
    return hole if hole.partition(".")[0] == element and not row.map_prefixes else ""


def turn_left_before_append(sn: R.Seen, line: int, mode: str) -> str:
    """D-4 (I-1, out/gm/qk-design.md's bundle I second half, tmaeno's OK 2026-11-17 20:3x): why the call at *line*
    hands no element of the key's turn, or empty.  Its list is filled only by appending each turn's element
    (:func:`appended_turns`); a row of the loop the pass printed begins with a hole of the element (963's
    ``{tmpJob.PandaID} skip since ...``), so the pass's run, marked by that value, is the key's turn; another row of
    the loop beginning with the same hole that every way to the append prints (968's ``{tmpJob.PandaID} to activate
    with rule``) is missing from the run where its absence counts: the key's turn did not reach the append, and the
    list the call hands holds no element of it.  Not where the turn's rows are read in no such run."""
    fn, run = sn.fn, sn.run
    for loop, at, element, rows in _turn_rows(fn, line):
        inside = {id(n) for n in ast.walk(loop)}
        for row, hole in rows:
            if sn.present(row):
                continue
            shown = [r for r in fn.rows if id(r.stmt) in inside and sn.present(r) and _turn_hole(r, element) == hole]
            if shown and R.absence_trusted(row, run.log, mode)[0] and run.would_print(row.level, row.func, row):
                return (f"the key's turn of the loop at {loop.lineno} printed {shown[0].line} and not {row.line}, which every "
                        f"way to the append at {at} prints: the call at {line} hands no {element} of the key's turn (D-4)")
    return ""


def _turn_rows(fn: R.Function, line: int) -> list[tuple[ast.For | ast.AsyncFor, int, str, list[tuple[R.Row, str]]]]:
    """``(loop, append line, element, [(row, hole)])`` for each append filling the list the call at *line* hands
    (:func:`appended_turns`): the rows of the loop every way to the append prints that begin with a hole of the
    element (968's ``{tmpJob.PandaID} to activate with rule`` before 969)."""
    out = []
    for loop, at, element in appended_turns(fn, line):
        append = R.statement_at(fn.node, at)
        if append is None:
            continue
        inside = {id(n) for n in ast.walk(loop)}
        before, _ = R.must_rows(fn.node, append, set())
        rows = [(row, hole) for s in before if id(s) in inside and (row := fn.row_at(s)) is not None
                and (hole := _turn_hole(row, element))]
        out.append((loop, at, element, rows))
    return out


def handing_rows(fn: R.Function, lines: Iterable[int]) -> list[int]:
    """I-2's premise for a caller's passes read by D-4 (:func:`turn_left_before_append`): the rows that every way to
    an append filling a list one of the calls at *lines* hands prints, each beginning with a hole of the appended
    element -- so a pass that handed an element prints a record of it.  Empty where a call hands a list the map reads
    no such append for, or an append has no such row."""
    found: list[int] = []
    for line in lines:
        turns = _turn_rows(fn, line)
        if not turns or any(not rows for *_, rows in turns):
            return []
        found += [row.line for *_, rows in turns for row, _ in rows]
    return sorted(set(found))


def empty_loops(fn: R.Function, stmt: ast.stmt, callee: dict[str, tuple[str, str]]) -> str:
    for it in R.iterables_over(fn.node, stmt):
        base = it
        while isinstance(base, (ast.Call, ast.Attribute)):
            base = base.func if isinstance(base, ast.Call) else base.value
        if isinstance(base, ast.Name) and base.id in callee and callee[base.id][0] in EMPTY:
            value, where = callee[base.id]
            return f"inside 'for ... in {ast.unparse(it)}', and {base.id} printed {value!r} ({where})"
    return ""


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

    Bundle G, G-1 (P53, out/gm/qk-design.md, tmaeno's OK 2026-11-17 16:0x): a bare name whose hole the map's prints
    name the written class's key column (``function_prints``' hole columns) is read the same way, bound alike at the
    row and at the arm: AtlasAnalWatchDog's 195 ``action=release_... jediTaskID={taskID}`` before 196 printed other
    tasks' IDs only, so no turn of 196 in the pass wrote the key's task.

    A hole that another hole follows with no literal between captures nothing (:func:`rows._regex` takes it lazily):
    an empty value is no key, and the hole is not read.  prepareTasksToBeFinished_JEDI's 4074 ``... for
    jediTaskID={jediTaskID}{f' since {errorDialog}' if errorDialog else ''}`` printed the key's task, read empty.
    """
    m = re.fullmatch(r"<(\w+)>\.\w+", env.subject) if env is not None else None
    if env is None or m is None or not env.keys or not env.written:
        return ""
    entity = m[1]
    if not R.loops_over(stmt, fn.node) <= R.loops_over(row.stmt, fn.node):
        return ""
    bound = row_names.names(fn.owner, line, env.written, entity, env.elements)
    captures = [row.capture(sn.run.records[i]) for i, _, _ in sn.hits.get(row.line, ())]
    keyed = _bare_key_holes(row, env.written.partition(".")[0], env.keys)
    turns = _turn_elements(stmt, fn.node) if keyed else set()
    for expr in sorted({e for c in captures for e in c}):
        name, _, column = expr.rpartition(".")
        if not name and expr in keyed and expr in turns:
            name, column = expr, keyed[expr]
        elif not bound or bound.get(name) != entity:
            continue
        held = env.table.get(entity, {}).get(column)
        if column not in env.keys or held is None:
            continue
        values = [c.get(expr) for c in captures]
        if (any(v is None or not str(v).strip() for v in values)
                or not row_names.same_binding(fn.node, name, [row.line], line)):
            continue
        wanted = str(held.value)
        if wanted not in [str(v).strip() for v in values]:
            return f"row {row.line} printed only for other values of {expr} ({len(values)} time(s)), not {wanted!r}"
    return ""


def _turn_elements(stmt: ast.AST, func: ast.AST) -> set[str]:
    """G-1 narrowed (T 7292cc2's MT-rerefine-late): the names a loop holding *stmt* binds each element to whole
    (:func:`loop_elements`: ``for taskID in taskIDs``), a turn's own object.  A bare key hole is read as the arm's
    object only where its name is one: TaskRefiner.runImpl's ``for jediTaskID, splitRule, taskStatus, parent_tid in
    taskList`` binds parent_tid to a part of the element, and 262's ``pending until parent task {parent_tid} is done``
    names the parent task by its jediTaskID in the turn of this task, not another task's turn."""
    out: set[str] = set()
    previous = stmt
    for ancestor in pathcond.ancestors(stmt):
        if ancestor is func:
            break
        if isinstance(ancestor, R.LOOPS) and previous in getattr(ancestor, "body", []):
            out |= set(loop_elements(ancestor))
        previous = ancestor
    return out


def _bare_key_holes(row: R.Row, cls: str, keys: frozenset[str]) -> dict[str, str]:
    """G-1: the bare names among *row*'s holes the map's prints name a key column of *cls* by (``columns``), each with
    the column: ``taskID`` -> ``jediTaskID``.  A name the map names two columns by is left out."""
    out: dict[str, str] = {}
    for mr in getattr(row, "map_rows", None) or []:
        for h in getattr(mr, "holes", None) or []:
            expr = getattr(h, "expression", "") or ""
            cols = {c.partition(".")[2] for c in getattr(h, "columns", None) or [] if c.partition(".")[0] == cls} & keys
            if expr.isidentifier() and len(cols) == 1:
                out.setdefault(expr, cols.pop())
    return out


#: the calls a ``while`` takes its next element by (Q5): ``self.job_output_reports.pop()``, ``queue.get()``
_TAKES = frozenset({"pop", "popleft", "get", "get_nowait"})


def loop_elements(loop: ast.AST) -> dict[str, set[int]]:
    """The names *loop* binds each element to, with the ids of the statements that bind them: a ``for``'s name target
    (the loop itself); a ``while``'s names its body binds, at its own level, from a call that takes the next element
    and is handed nothing (``pop()``, ``get()``; ``next(it)``) or unpacked from such a name -- add_main 108 and 112,
    ``one_jor = self.job_output_reports.pop()`` and ``panda_id, job_status, attempt_nr, time_stamp = one_jor``
    (out/gm/qk-design.md Q5)."""
    if isinstance(loop, (ast.For, ast.AsyncFor)):
        return {loop.target.id: {id(loop)}} if isinstance(loop.target, ast.Name) else {}
    if not isinstance(loop, ast.While):
        return {}
    got: dict[str, set[int]] = {}
    for s in loop.body:
        if not isinstance(s, ast.Assign):
            continue
        v = s.value
        takes = isinstance(v, ast.Call) and (
            (isinstance(v.func, ast.Attribute) and v.func.attr in _TAKES and not v.args and not v.keywords)
            or (isinstance(v.func, ast.Name) and v.func.id == "next" and len(v.args) == 1 and not v.keywords))
        if takes or (isinstance(v, ast.Name) and v.id in got):
            for t in s.targets:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                        got.setdefault(n.id, set()).add(id(s))
    return got


def _element_value(fn: R.Function, loop: ast.AST, var: str, got: dict[str, str]) -> str | None:
    """What a record printed of *loop*'s element *var*: its own hole in *got* (the row's capture), or, where the row
    prints it through a local the loop's body binds to an f-string holding it -- add_main 113's ``token_str =
    f"pid={uniq_pid} : job={panda_id}.{attempt_nr}"``, printed at 129 as ``{token_str} last add st=...`` -- read out
    of that local's hole by the f-string's shape (Q5); None where neither reads it."""
    if var in got:
        return got[var]
    for s in ast.walk(loop):
        if not (isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], ast.Name)
                and isinstance(s.value, ast.JoinedStr) and s.targets[0].id in got
                and any(isinstance(n, ast.Name) and n.id == var for n in ast.walk(s.value))):
            continue
        for shape in R._alternatives(s.value, fn.node, s, expand=False):
            pattern, groups = R._regex(shape, True)
            if m := re.fullmatch(pattern, got[s.targets[0].id].strip(), re.S):
                read = {groups[g]: v for g, v in m.groupdict().items()}
                if var in read:
                    return read[var]
    return None


def key_env(c: Candidate, logs: dict[str, list[Run]], key: str) -> E.Env | None:
    """Q5: the element of a loop around *c*'s lines that the key's runs print the key as, bound to the key -- the name
    the records say (add_main's ``panda_id``, printed as ``job=7338795608.1`` through ``token_str``), not one a
    spelling picks; None where no record prints it so, or two names do (:func:`other_elements` then reads nothing)."""
    try:
        fn = R.function(c.rel, c.func, c.lines)
    except (LookupError, ValueError, SyntaxError):
        return None
    pattern = re.compile(rf"(?<![0-9A-Za-z_]){re.escape(key)}(?![0-9A-Za-z_])")
    named: set[str] = set()
    for line in c.lines:
        stmt = R.statement_at(fn.node, line)
        if stmt is None:
            continue
        for loop in pathcond.ancestors(stmt):
            if loop is fn.node:
                break
            if not isinstance(loop, (ast.For, ast.AsyncFor, ast.While)):
                continue
            elements = loop_elements(loop)
            inside = [r for r in fn.rows if any(n is r.stmt for b in loop.body for n in ast.walk(b))]
            if not elements or not inside:
                continue
            for f in c.log_files:
                for run in logs.get(f, []):
                    for record in run.records:
                        if key not in record.message:
                            continue
                        for row in inside:
                            got = row.capture(record)
                            for var in elements:
                                value = _element_value(fn, loop, var, got) if got else None
                                if value is not None and pattern.fullmatch(value.strip()):
                                    named.add(var)
    if len(named) != 1:
        return None
    var = named.pop()
    env = E.Env()
    env.put(var, key, f"the key's records print it as {var} ({c.owner})")
    return env


def other_elements(fn: R.Function, stmt: ast.stmt, row: R.Row, sn: R.Seen, env: E.Env | None) -> str:
    """Why *row*, which the run printed, did not print for the loop element *stmt* is asked about; '' if it may have.

    A row inside the same loop as *stmt* prints once per element: a ``for``'s, or a ``while``'s that takes its next
    element in its body (:func:`loop_elements`, Q5).  When *env* binds an element's name (H7 binds the make call's
    name to the object the outside says is missing; Q5 the name the key's records print the key as, :func:`key_env`)
    and every record of the row holds another value of it -- in its own hole, or through a local the body makes of
    it (:func:`_element_value`) -- the row printed for the other elements only (``register_dataset {name}`` at 861
    printed for the two top-level names and not for the ``_sub``, so the make call under it did not run for the
    ``_sub``; add_main's 129 ``last add st=`` printed for another job's turn).  Not when the body binds the name
    again, nor when a record's value cannot be read.
    """
    if env is None:
        return ""
    for loop in pathcond.ancestors(stmt):
        if loop is fn.node:
            break
        if not isinstance(loop, (ast.For, ast.AsyncFor, ast.While)):
            continue
        elements = loop_elements(loop)
        if not elements or not any(n is row.stmt for b in loop.body for n in ast.walk(b)):
            continue
        for var, binders in sorted(elements.items()):
            if env.canonical(var) not in env.values:
                continue
            # the element's own bindings bind it; any other store binds it again
            bound_by = {id(n) for b in loop.body if id(b) in binders and isinstance(b, ast.Assign)
                        for t in b.targets for n in ast.walk(t)}
            if any(isinstance(n, ast.Name) and n.id == var and isinstance(n.ctx, ast.Store) and id(n) not in bound_by
                   for b in loop.body for n in ast.walk(b)):
                continue
            values = []
            for i, _, _ in sn.hits.get(row.line, ()):
                value = _element_value(fn, loop, var, row.capture(sn.run.records[i]))
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
    guards = [(g, f"{fn.shown}:{stmt.lineno}") for g in pathcond.arm_condition(stmt)] + callers_guards(fn)
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
                stretch = before_stretch(window, c.row_after) if window is not None and c.row_after is not None else None
                found = [r for r in logs[name] if window is None or spans(r, window) or (stretch and spans(r, stretch))]
                if found and window is not None:
                    found, dropped = passes_at(c, found, window, logs, mode, called_at)
                    if not found:
                        reasons.append(f"{name}: no pass of the key at {window}: {'; '.join(dropped)}")
                        continue
                if found:
                    run = run_holding(c, found, mode)
                    if run is not None:
                        if window is not None and not spans(run, window):
                            p_notes.append(f"ROW AFTER: {c.func}'s pass {run.first}..{run.last} ended before the row at "
                                           f"{window[0]}: the map has its history call after its write, and no pass of "
                                           f"it is at the row (P37)")
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
    if question_run is None and called_at is not None and subject and observed:
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
                    elsewhere = other_elements(fn, stmt, row, sn, env) or for_others.get(row.line, "")
                    if elsewhere and R.absence_trusted(row, run.log, mode)[0] and run.would_print(row.level, row.func, row):
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
                elif (trust := R.absence_trusted(row, run.log, mode))[0] and run.would_print(row.level, row.func, row):
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
                    v.proven = proven
                elif proof.kind == "positive":
                    v.kind = "positive"
                    v.reasons.append(proof.reason)
                elif (around := rows_around(fn, psn, stmt)) is not None:
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
                    else f"a traceback frame of {fn.shown} cannot be placed in the corpus: no row after {line} is required"
                )
            where = f"{Path(c.rel).name}::{c.func}:{line}"
            if proof.kind == "positive" and v.kind == "excluded":
                p_notes.append(f"P twin: {where} excluded ({'; '.join(v.reasons)}), but {proof.reason}")
            elif proof.kind == "inconsistent":
                p_notes.append(f"P inconsistent: {where}: {proof.reason}")
            verdicts.append(v)

    # D-4 on each kept call arm: the list it hands holds no element of the key's turn where that turn did not reach
    # the append filling it (turn_left_before_append)
    for v in verdicts:
        if v.kind == "kept" and v.candidate.owner in seens:
            if unhanded := turn_left_before_append(arm_seen.get((v.candidate.owner, v.line), seens[v.candidate.owner]),
                                                   v.line, mode):
                v.kind = "excluded"
                v.reasons.append(unhanded)
                v.unhanded = unhanded

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

    # P36 on each kept arm: a print after its write on every way holds the written row's key, and every record of it in
    # the run holds another key (after_key_others)
    if env is not None and env.written:
        for owner, sn in seens.items():
            if arms_kept := [v for v in verdicts if v.candidate.owner == owner and v.kind == "kept"]:
                after_key_others(sn.fn, sn, arms_kept, env.written, env)

    # P35 on the arms of one write statement: its print in the key's turn shows the bind map it ran with, and the arms
    # that did not give the bind its value there are out (printed_write_arms)
    if observed and env is not None and env.written:
        for owner, sn in seens.items():
            arms_standing = [v for v in verdicts if v.candidate.owner == owner and v.kind in ("kept", "positive")]
            if arms_standing:
                named, raised = R.raised_in(sn.fn, sn.run)
                printed_write_arms(sn.fn, sn, arms_standing, env.written, observed, env, mode, raised if named else set())

    # P1 on each kept candidate: a row of its pass printed the local it binds to the observed value, and of the
    # bindings that can be what the row printed, only it gives that value (printed_local)
    if observed:
        for v in verdicts:
            c = v.candidate
            if v.kind != "kept" or c.owner not in seens or (stmt := R.statement_at(seens[c.owner].fn.node, v.line)) is None:
                continue
            sn = arm_seen.get((c.owner, v.line), seens[c.owner])
            named, raised = R.raised_in(sn.fn, sn.run)
            if why := printed_local(sn.fn, sn, stmt, observed, env, raised if named else set()):
                v.kind = "positive"
                v.reasons.append(why)

    # J on each kept candidate: the call ran the outermost if of its chain a print shows it ran, entered no handler
    # around it, and every test from there down to it holds (ran_down_to)
    if env is not None:
        for v in verdicts:
            c = v.candidate
            if v.kind != "kept" or c.owner not in seens or (stmt := R.statement_at(seens[c.owner].fn.node, v.line)) is None:
                continue
            sn = seens[c.owner]
            named, raised = R.raised_in(sn.fn, sn.run)
            if why := ran_down_to(sn.fn, sn, arm_seen.get((c.owner, v.line), sn), stmt, env, raised if named else set()):
                v.kind = "positive"
                v.reasons.append(why)

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
    chosen, how, pass_writes = None, "undecided", []
    if change is not None and change.chosen is not None and (decide != "strict" or not unexamined):
        chosen, how = change.chosen, change.how
        pass_writes = [CH._name(v) for v in change.others]
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
    return Judgement(verdicts, chosen, how, bracket, notes, runs_of, question_run if joined else None, pass_writes)

