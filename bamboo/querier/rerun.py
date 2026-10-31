"""A case run as one loop, with no one choosing between the steps.

    python -m bamboo.querier.rerun --subject JobSpec.jobStatus --observed failed --key <PandaID> \
        (--log panda-adder.log --log panda-setupper.log | --records answers.jsonl [--asks asks.jsonl]) \
        --rows rows.json [--hand hand.json] [--patterns map|outside] [--decide positive|strict]

Every step takes its arguments from the symptom, the logs, the rows, or the
step before it.  No line number, function or file of a case is written here:
where a step cannot find what the next one needs, it looks in ``--hand`` and
prints ``HAND``, and without an entry there it prints ``STOP`` and ends.  Run
with no hand file, the steps before the first ``STOP`` are the ones that ran
with no judgement.

The logs come through one door (``records.Logs``): the files handed in (``--log``), each read whole, or the
recording step's answers (``--records``, :mod:`bamboo.querier.recorded`), asked by window and machine.  Each step that
reads the logs makes its window from the rows (:mod:`bamboo.querier.windows`).  Through the recorded answers, an ask
they do not hold is written to ``--asks`` with every other ask of its step, and the run stops; it is run again once the
recording step has fetched them.

**The loop**.  Each step answers one
question and returns the next; which question comes next is decided by what
the step found, not by the order of a case.  The questions, the step each is
answered by, and the rule it follows:

``Asked``      H1  candidates           the map's readings for the symptom
               H2  which ran            ``which_ran`` over them, in the run the statuslog time falls in
               H3  what its guard reads an atom that is a method call with no arguments -> ``SetBy``
``SetBy``      H4  which call set it    census, dispatch, value match, then ``which_ran``  -> ``Raised``
``Raised``     H5  what it was handed   the run's traceback through the function chosen:
                                        raised outside the corpus -> the frame at the boundary,
                                        the object the exception names -> ``Missing``;
                                        raised inside -> ``FalseAtom`` (an exception as a false guard)
``Missing``    H6  who makes it         an object the outside says is missing -> the calls that make it
                                        kept when their function printed the name in a log to hand
               H7  which way it went    the make call and every branch that goes round it -> ``Branch``
``Branch``     H8  what the guard held  the guard of the branch taken, solved backwards -> ``Settled``
``Settled``    H9  the same test        norm 1: the same atom in a function this investigation has a run for
                                        -> ``Pair``
``Pair``       H10 where they part      the inputs of the atom, by value, then the bindings that took effect
``FalseAtom``  R1  what raised          :mod:`bamboo.querier.raised`: the raising expression and the guard it says was false
               R2  what the operands held  ``false_guard.py``: each operand by value over the map's walk
               R3  where they part      the alternative binding that would have held, and the guard
                                        that chose against it; its inputs less the loop's row -> ``Reproduce``
``Reproduce``  R4  the same again       ``reproduce.py``: the raising function alone, compiled from the corpus,
                                        run with the inputs R2 rebuilt; then with R3's alternative -> ``Field``
``Field``      R5  what the input held  a row field's value for this instance

A value that does not move (``--stuck``):

``Stuck``      S1  what moves it on     the map's ``follow_up``: the functions a reader statement hands
                                        its rows to (or the reader), and those whose write requires
                                        the value; where the map could not read them (``attach_mover_gaps``)
``Passes``     S2  which way each pass went  the mover's runs of the key, cut into passes at its entry
                                        row (rule 20); in each, ``which_ran`` over the arms that write
                                        another value and every exit of the function
``Blocked``    S3  what kept it from moving  the exit's path condition, solved true; per arm, the guards
                                        the pass made false (not one read elsewhere, ``E.read_elsewhere``);
                                        the arm with fewest is the nearest (rule 21); its guard's ``if``
                                        (``guard_if``)
``Input``      S4  where the guard's name was bound  ``asked.walk`` from the guard's ``if``, the binding
                                        the pass's values leave (``E.taken``); a stop -> its question
``Rows``       S5  what the call returned  a call that returns rows by a key: the call, its
                                        arguments, the tables it reads at its constants
                                        (``proxy_reading``), and the counts the passes printed against
                                        the jobs not yet final at each pass (the job histories); the
                                        tables at one moment (the counts) check that assumption
"""
from __future__ import annotations

import argparse
import ast
import asyncio
import copy
import datetime
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path

from bamboo.codemap import strategy as strategy_mod
from bamboo.codemap.calls import Types
from bamboo.codemap.models import (
    KEY_BOUND,
    KEY_HANDED,
    KEY_KEPT,
    KEY_NO_USE,
    KEY_RETURNED,
    KEY_ROWS,
    KEY_UNKNOWN,
    MUST_ROW_WRITTEN,
    RESULT_AGGREGATE,
    Symptom,
    required_values,
)
from bamboo.codemap.panda import pathcond, sqltext
from bamboo.codemap.panda import sql as sql_mod
from bamboo.codemap.panda.provenance import facade_calls
from bamboo.codemap.panda.recognizers.sqlwrite import _flag_parameters
from bamboo.codemap.reading import containing_function
from bamboo.querier import asked as A
from bamboo.querier import attr_time as AT
from bamboo.querier import census as C
from bamboo.querier import evaluate as E
from bamboo.querier import false_guard as G
from bamboo.querier import held as HD
from bamboo.querier import mapdump
from bamboo.querier import outbound as O
from bamboo.querier import parts as PT
from bamboo.querier import raised as X
from bamboo.querier import reproduce as P
from bamboo.querier import rows as R
from bamboo.querier import value_diff as V
from bamboo.querier import windows as W
from bamboo.querier.recorded import RecordedLogs
from bamboo.querier.records import (
    CORPUS,
    FileLogs,
    Logs,
    Owners,
    Unrecorded,
    Window,
    keyed,
    runs_for,
)
from bamboo.querier.which_ran import Candidate, Verdict, class_of, judge, stable_holes
from bamboo.scripts.derive_strategy import _investigate

BAMBOO = Path(R.BAMBOO_TREE)
#: where the assumption of reading 2 in S5 puts a job out of the active tables -- PanDA's,
#: written here because the map does not say which rows a table holds.  It holds for one class of call only
#: (KNOWN_ISSUES.md K1, K2)
FINAL_STATES = ("finished", "failed", "cancelled", "closed")
#: the tables that assumption is about: a job sits in one of them until it reaches FINAL_STATES -- PanDA's too
ACTIVE_TABLES = ("jobsDefined4", "jobsActive4")


def handed_to(func: ast.FunctionDef | ast.AsyncFunctionDef, call: ast.Call) -> dict[str, ast.expr]:
    """What *call* hands each parameter of the method *func*: the argument it writes, or the default *func* states for
    one it leaves out.  A call with ``*`` or ``**`` gets no defaults (what it spreads may fill any of them), and the
    positional arguments from a ``*`` on are not bound."""
    args = func.args
    positional = [a.arg for a in (*args.posonlyargs, *args.args)][1:]
    handed: dict[str, ast.expr] = {}
    for name, arg in zip(positional, call.args, strict=False):
        if isinstance(arg, ast.Starred):
            break
        handed[name] = arg
    handed |= {k.arg: k.value for k in call.keywords if k.arg}
    if any(isinstance(a, ast.Starred) for a in call.args) or any(k.arg is None for k in call.keywords):
        return handed
    stated = dict(zip(positional[len(positional) - len(args.defaults):], args.defaults, strict=False)) if args.defaults else {}
    stated |= {a.arg: d for a, d in zip(args.kwonlyargs, args.kw_defaults, strict=False) if d is not None}
    return stated | handed


def receiver_methods(method: str, call: ast.Call, file: str, line: int) -> tuple[list[str], set[tuple[str, int]]]:
    """``(the classes the receiver of *call* is declared to be, as file::Class; their methods named *method*, as
    (file, line))``, read with the map's Types in the def of *file* holding *line*; both empty where it declares none.

    The walk stops at the database by the receiver's name (``provenance.DATABASE_INTERFACES``), which does not say
    which facade it is; its declaration does.  JEDI's ``self.taskBufferIF`` is a ``JediTaskBufferInterface``, a
    ``JediTaskBuffer`` to a type checker, so its ``updateTask_JEDI`` is ``JediTaskBuffer``'s and its ``peekJobs`` the
    one it inherits from ``TaskBuffer``.
    """
    types = A._types()
    tree = types.tree(file)
    func = containing_function(tree, line, "") if tree is not None else None
    if func is None or not isinstance(call.func, ast.Attribute):
        return [], set()
    classes = types.declared(file, func, call.func.value) or frozenset()
    found = [types.method(where, cls, method) for where, cls in classes]
    return sorted(f"{where}::{cls.name}" for where, cls in classes), {(f[0], f[1].lineno) for f in found if f is not None}


def proxy_reading(
    method: str, call: ast.Call, theirs: set[tuple[str, int]] | None = None
) -> tuple[list[str], dict[str, object], list[str], list[str], list[str], list[str]] | None:
    """``(parameters, constants, the statement's tables, the tables this call reads, the proxy methods it reaches
    whose statements are not read, the proxy's statements this call runs as ``<file>:<line>``)`` for a call the walk stopped at as the database's, or None where no facade has
    *method* -- or none of *theirs*, the methods its receiver is declared to have (:func:`receiver_methods`).

    The method is the one the walk's stop names, and what it asks the database is what the facades' method of the
    name calls on the proxy it borrows (``provenance.facade_calls``, as ``trace._Walk.database_asked`` reads it):
    ``peekJobs`` reads what ``peekJob`` does.  Where the receiver declares its class, the facade is the one it is:
    ``TaskBuffer.updateTask_JEDI`` hands the proxy ``updateDEFT=True`` by default and ``JediTaskBuffer``'s False, so
    the two read together name two tables a JEDI caller's call never reads.  Where it declares none, every facade's
    method of the name, as the stop was found.  The call is bound to the facade's parameters, with the
    defaults the facade states, and each call the facade makes on the proxy hands the proxy's parameters a constant
    it writes, a parameter of its own it never rebinds, or the proxy's default.  Bound to the proxy's parameters
    directly, a signature the two do not share flips a flag: ``updateTask_JEDI``'s ``updateDEFT`` defaults to False in
    the facade and True in the proxy.

    The constants are those handed to parameters a method tests and never rebinds (``sqlwrite._flag_parameters``, as
    the map's route readings take them): a proxy call under an ``if`` of the facade they decide against is not read
    (``pathcond.ruled_out_by``, as ``trace._Walk._ruled_out_at``), and the call's tables are each proxy method read
    under its own (``sql.at_call``).  ``TaskCommando.py:143`` calls ``getPandaIDsWithTask_JEDI(jediTaskID, True)``,
    and the two archive tables are appended only under ``if not onlyActive``: the statement reads four tables, the
    call two.  They are said by the facade's parameter, or as ``<proxy method>'s <parameter>`` for one the facade
    writes or leaves to the proxy's default.

    A proxy method the call reaches in which no statement is read is named, as ``database_asked`` names it:
    ``storeJobs(..., bulk_job_insert=True)`` reaches ``bulk_insert_new_jobs``, which runs what
    ``insertNewJob`` hands back, and the tables of the rest are not all the call reads.  One whose statements the
    constants all rule out runs nothing here and is not named.
    """
    made = facade_calls(Types(R.ROOTS, {}), method)
    if theirs is not None:
        made = [one for one in made if (one.file, one.method.lineno) in theirs]
    if not made:
        return None
    params = [a.arg for a in (*made[0].method.args.posonlyargs, *made[0].method.args.args)][1:]
    said: dict[str, object] = {}
    statement: list[str] = []
    at_call: list[str] = []
    unread: list[str] = []
    sites: list[str] = []

    def add(named: list[str], func: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for run in sql_mod.executions(func):
            named += [t for t in sqltext.tables_named(run.sql) if t not in named]

    for one in made:
        handed = {p: a.value for p, a in handed_to(one.method, call).items() if isinstance(a, ast.Constant)}
        rebound = {n.id for n in ast.walk(one.method) if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del))}
        tested = {p: v for p, v in handed.items() if p in _flag_parameters(one.method)}
        add(statement, one.proxy_method)
        said |= tested
        if pathcond.ruled_out_by(one.call, one.method, tested):
            continue
        flags = _flag_parameters(one.proxy_method)
        constants: dict[str, object] = {}
        for p, a in handed_to(one.proxy_method, one.call).items():
            if p not in flags:
                continue
            if isinstance(a, ast.Constant):
                constants[p] = a.value
                said[f"{one.proxy_method.name}'s {p}"] = a.value
            elif isinstance(a, ast.Name) and a.id in handed and a.id not in rebound:
                constants[p] = handed[a.id]
                said[a.id] = handed[a.id]
        with sql_mod.at_call(one.proxy_method, constants):
            add(at_call, one.proxy_method)
            for run in sql_mod.executions(one.proxy_method):
                if (site := f"{one.proxy_file}:{run.call.lineno}") not in sites:
                    sites.append(site)
        if not sql_mod.executions(one.proxy_method) and one.proxy_method.name not in unread:
            unread.append(one.proxy_method.name)
    return params, said, statement, at_call, unread, sites


def stop_reading(detail: str) -> tuple[list[str], list[str]]:
    """``(tables, proxy methods not read)`` the walk's stop at the database says: ``reads A, B by :k; what m runs is
    not read`` (``trace._Walk.database_asked``).  The tables are cut at the keys or at what is not read."""
    said = re.search(r" -- reads (.+?)(?: by :|; what |$)", detail or "")
    unread = re.search(r"; what (.+?) runs? is not read", detail or "")
    return (said[1].split(", ") if said else []), (unread[1].split(", ") if unread else [])


def at_one_moment(counts: list[dict], tables: list[str]) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    """The rows file's counts (``table``, ``jobStatus``, ``n``, all of one moment) against reading 2's assumption in
    S5: what *tables* hold, the jobs among them in FINAL_STATES, and the jobs not yet final outside them -- each as
    ``table/jobStatus`` -> n.  The assumption holds at that moment when the last two are empty."""
    read = {t.lower() for t in tables}
    held, final_in, open_out = {}, {}, {}
    for r in counts:
        inside, final = r["table"].lower() in read, r["jobStatus"] in FINAL_STATES
        for into, kept in ((held, inside), (final_in, inside and final), (open_out, not inside and not final)):
            if kept:
                into[f"{r['table']}/{r['jobStatus']}"] = into.get(f"{r['table']}/{r['jobStatus']}", 0) + r["n"]
    return held, final_in, open_out


def through_calls(rel: str, spans: list[tuple[int, int]], writers: list, subject: str, observed: str) -> tuple[list[str], list[str]]:
    """``(reached, cut)``: the writers with an arm that can move *subject* on from *observed* (``arms_of``) that the
    function at *rel* (its defs' line *spans*) calls into by the map's ``JunctionNode.called_from`` -- a call placed in
    it, or one hop out of it -- and those whose list the map cut (``… N more``), which it may call into unlisted.  Two
    hops only (KNOWN_ISSUES.md K14)."""
    reached, cut = [], []
    for w in writers:
        if not arms_of(writers, w.owner, subject, observed):
            continue
        name = w.owner.split("::")[-1]
        places = [(f, int(n)) for entry in w.called_from for f, n in re.findall(r"([\w./-]+\.py):(\d+)", entry)]
        if any(f == rel and lo <= n <= hi for f, n in places for lo, hi in spans):
            reached.append(name)
        elif any(entry.startswith("…") for entry in w.called_from):
            cut.append(name)
    return sorted(set(reached)), sorted(set(cut))


def defs_holding(rel: str, func: str, lines: list[int]) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    """The defs named *func* in *rel* that hold one of *lines* -- the mover is the def its statement is in, not every
    def of the name; every def of the name when there are no *lines* (a mover that takes rows another selects)."""
    defs = R.functions(rel, func)
    return [f for f in defs if any(f.lineno <= n <= f.end_lineno for n in lines)] if lines else defs


def map_gaps(follow) -> dict[str, list]:
    """``{owner: [ReaderGap]}``: where the map could not read a mover's statements on these rows -- a selecting
    function's and a guarding one's, one judgement for both (``strategy.attach_mover_gaps``)."""
    return {**follow.reader_gaps, **{g.owner: g.gaps for g in follow.guarded_by if g.gaps}}


def hides_where(gaps: list) -> bool:
    """Whether a mover's gaps can hide where its rows go, and so a write its reach has no junction for: any gap but a
    statement read only in part, which hides which rows it takes (``ReaderGap.hides_where``).  Only such a gap keeps a
    mover with no arm of its own; every gap is said (``gap_said``)."""
    return any(gap.hides_where for gap in gaps)


def gap_said(gaps: list) -> str:
    """A mover's gaps in the map's words (``strategy.gap_texts``): a text per kind, with every line of the kind."""
    return "; ".join(strategy_mod.gap_texts(gaps))


def kept_without_arms(found: bool, reached: list[str], cut: list[str], hands: bool, hidden: bool) -> list[str]:
    """Why S1 keeps a mover with no arm of its own: it moves the value, if at all, through what it calls or the rows it
    hands on.  None is dropped: ``called_from`` holds two hops, so a write further down is not seen, and "not listed"
    is not "does not reach" (bamboo-85; tmaeno, 2026-10-14).  Dropping one is the map's reach's to say, with its holes
    (``reach_said``); these are the reasons of a mover it keeps, or where the dump has no reach (KNOWN_ISSUES.md K14).
    *hidden*: a gap of the map's may hide where its rows go (``hides_where``)."""
    return [why for why, holds in (
        ("its def is not found", not found),
        (f"it calls into {reached}", reached),
        (f"it may call into {cut}, whose callers the map lists only in part", cut),
        ("it hands its rows on", hands),
        ("the map could not read where all its rows go", hidden),
    ) if holds] or ["no call it makes reaches a writer of another value within the two hops the map lists, "
                    "and a write further down is not seen (KNOWN_ISSUES.md K14)"]


def reach_said(reach: dict | None, hidden: bool) -> tuple[bool, str | None]:
    """``(dropped, line)``: whether S1 drops a mover with no arm of its own by what the map's reach says of it
    (``Rerun._reach_of``), and the line saying so.  Dropped only where it reaches no write of another value and leaves
    no call on the way open (the map's ``[]``, never its ``None``), its own defs hold no arm the junctions place there,
    and no gap of the map's may hide where its rows go (*hidden*, ``hides_where``) -- such a gap may hide a write the
    reach has no junction for (build2-design.md 6, R1).  A drop that rests on calls the reach closed by their name
    (``closed_by_name``) names them.  No line where the map was not asked."""
    if reach is None:
        return False, None
    if reach["own_arms"]:
        return False, (f"reach: its def holds an arm moving it off by the map's junctions {reach['own_arms']}, though S1 "
                       "counts none: kept")
    writes = reach["writes"]
    if writes:
        names = sorted({w["function"].split("::")[-1].split(".")[-1] for w in writes})
        return False, f"reach: reaches {len(writes)} writer(s) of another value: {names}"
    if writes is None:
        return False, (f"reach: reaches no write of another value, but {reach['holes']} call(s) on the way are left "
                       f"open, e.g. {reach['first_holes']}")
    if hidden:
        return False, ("reach: reaches no write of another value and leaves nothing open, but the map could not read "
                       "where all its rows go: kept")
    closed = f"; {reach['closed']} call(s) closed by name, e.g. {reach['first_closed']}" if reach.get("closed") else ""
    return True, (f"no arm of its own, and reaches no write of another value with nothing left open (closure "
                  f"{reach['closure']}; assuming {sorted(reach['assumptions'])}{closed}): dropped")


def prints_naming(writers: list, owner: str, lines, columns: set[str]) -> dict:
    """Judgement 3's third premise for *owner*: every run that reaches one of its arms at *lines* prints the key the
    asks search by -- a line the map says runs before the arm on every path (``Branch.must_prints``) with a hole it
    says holds a column among *columns* of the row the arm writes (``Branch.must_names``).  The map pairs neither a
    run's mark nor an SQL statement's key bind with the written row (``mustprints``): an arm with must prints and no
    such name is not known either way; one with none, no run need print anything before it.  ``{"held": True | False |
    None, "why": ...}``, False where any arm says so, None where any is not known, as ``every_log_asked``."""
    if not columns:
        return {"held": None, "why": "the rows file says no column of the row holds the key"}
    if not lines:
        return {"held": None, "why": "no arm of its own: what it calls prints, if anything"}
    unread, bare, unnamed = [], [], []
    for line in sorted(lines):
        for b in (b for j in writers if j.owner == owner for b in j.branches if b.line == line):
            if b.must_prints is None:
                unread.append(f"{line}: {b.must_hole or 'not read'}")
            elif not b.must_prints:
                bare.append(line)
            elif not any(n.row == MUST_ROW_WRITTEN and n.column in columns for n in b.must_names or []):
                unnamed.append(line)
    if bare:
        return {"held": False, "why": f"no line runs before the arm(s) at {sorted(set(bare))} on every path"}
    if unread:
        return {"held": None, "why": f"the map did not read the lines before {unread}"}
    if unnamed:
        return {"held": None, "why": f"no line before the arm(s) at {sorted(set(unnamed))} is known to name the written row's "
                                     f"{sorted(columns)}: the map pairs neither a run's mark nor an SQL statement's key bind "
                                     f"with it (KNOWN_ISSUES.md K19)"}
    return {"held": True, "why": f"a line before every arm at {sorted(lines)} names the written row's {sorted(columns)}"}


def prints_nothing(owner: str, lines) -> bool:
    """Whether the map has no prints of *owner*'s own (``rows._prints_of``, as ``every_log_asked`` reads it).  Such a
    function has no run of its own in any log: a run found where it is looked for is a caller's, or not even that --
    S1 looks for one with no logs of its own in the selecting statements' -- and its lines say nothing of whether it
    was called (JediTaskSpec.setOnHold, read on TaskCommando's run, which never calls it; KNOWN_ISSUES.md K25).  Where
    its write is read, it is as an arm its callers' junctions place at each call of it."""
    found = R._prints_of(owner, lines)
    return found is None or not found.statements


def every_log_asked(owner: str, lines, asked: list[str]) -> dict:
    """Judgement 3's second premise for *owner*: every log a run of it prints to was asked -- each file the map places
    its prints in by the logger they are made on (``PrintStatement.log_files``; a DB proxy method's are the server's and
    JEDI's both, as the process that called it chooses).  ``{"held": True | False | None, "why": ...}``, None where the
    files are the map's reading's, not where it places the prints (``rows.own_log_files``)."""
    if not R.OWN_FILES or R.ROWS_FROM != "graph":
        return {"held": None, "why": "the logs are the map's reading's, not where it places the prints (ROWS_OWN_FILES, ROWS_FROM)"}
    found = R._prints_of(owner, lines)
    if found is None or not found.statements:
        return {"held": False, "why": "the map has no prints of its own for it"}
    if bases := sorted({s.log_files_basis for s in found.statements} - {"logger"}):
        return {"held": False, "why": f"the map places some of its prints by {bases}, not by their logger: where they land is not known whole"}
    files = sorted({f for s in found.statements for f in s.log_files})
    if not files:
        return {"held": False, "why": "the map places its prints in no file"}
    if missing := [f for f in files if f not in asked]:
        return {"held": False, "why": f"not asked: {missing}"}
    return {"held": True, "why": f"every file the map places its prints in was asked: {files}"}


def arms_of(writers: list, owner: str, subject: str, observed: str) -> dict[int, Arm]:
    """*owner*'s arms that can move *subject* on from *observed*, by line: not one that can only write the observed
    value, nor one whose write requires the row to hold other values of the subject's column -- its
    ``row_precondition``, read as the map reads a guard (``required_values``); a term it cannot read keeps the arm --
    nor a birth, a write that makes the row (``births_of``)."""
    column = subject.partition(".")[2].lower()
    marks = birth_marks(owner, subject)
    found: dict[int, Arm] = {}
    for j in writers:
        if j.owner != owner:
            continue
        for b in j.branches:
            if b.line is None or b.outcome == observed or born(b, marks):
                continue
            required = [rv for term in b.row_precondition or [] if (rv := required_values(term)) and rv[0] == column]
            if any(observed not in values for _, values in required):
                continue
            arm = found.setdefault(b.line, Arm(b.line, [], list(b.path_condition)))
            arm.outcomes.append(b.outcome)
    return found


def guard_if(arm_stmt: ast.AST, guard: str) -> ast.If | ast.While | None:
    """The innermost ``if`` (or ``while``) above *arm_stmt* that an arm's *guard* is the test of, as
    ``pathcond.path_condition`` spells it: the test, the arm in its body, or ``not (<the test>)``, the arm in an
    ``if``'s ``else``.  None for a guard no test above the arm says (KNOWN_ISSUES.md K24)."""
    test = E.strip_annotation(guard)
    previous = arm_stmt
    for a in pathcond.ancestors(arm_stmt):
        if isinstance(a, (ast.If, ast.While)):
            said = ast.unparse(a.test)
            if previous in a.body and test == said:
                return a
            if isinstance(a, ast.If) and previous in a.orelse and test == f"not ({said})":
                return a
        previous = a
    return None


def lines_after(func: ast.AST, arm_stmt: ast.stmt, name: str) -> list[tuple[int, int]]:
    """The spans of the statements after *arm_stmt* in its own block, up to the first that binds *name*: a row printed
    there holds *name* as the arm read it (updateJobStatus' ``varMap[":jobStatus"] = jobStatus`` at 241, then ``new=
    {jobStatus}`` at 244).  A row before the arm, or past a binding, may print another value (228's ``jobStatus =
    oldJobStatus`` lies between 51's ``status={jobStatus}`` and the arm)."""
    for node in ast.walk(func):
        for block in (getattr(node, f, None) for f in ("body", "orelse", "finalbody")):
            if isinstance(block, list) and any(s is arm_stmt for s in block):
                spans = []
                for s in block[block.index(arm_stmt) + 1:]:
                    if any(isinstance(n, ast.Name) and n.id == name and not isinstance(n.ctx, ast.Load) for n in ast.walk(s)):
                        break
                    spans.append((s.lineno, s.end_lineno or s.lineno))
                return spans
    return []


def never_runs(kf) -> bool:
    """Whether the map says *kf*'s statement never runs from a placed caller: it read routes into the statement's
    function (``routes``), each rules the statement out by the constants it hands (``out_on``: an if holding it, an exit
    before it), and none is left open (``routes_open``: a call by the function's name placed nowhere, a route cut) --
    ``retryJob``'s in-place UPDATE (4146) under ``if not getNewPandaID:``, where every caller hands True.  A statement
    with no route placed (the map leaves it open: reached from outside, called from outside the corpus, its class read
    as a value), or a dump without the fields, is not known never to run."""
    routes = getattr(kf, "routes", None) or 0
    return routes > 0 and len(getattr(kf, "out_on", None) or []) == routes and getattr(kf, "routes_open", True) is False


#: ``[the dump's key_froms list, those of it a placed caller may run]`` -- the list is held so that its id is never
#: another list's
_RUNS: list = [None, []]


def key_froms() -> list:
    """The installed dump's ``key_froms``: where each row-naming term of a write was filled from (``KeyFrom``); none
    before an install or from a dump that has none.  A statement no placed caller runs persists nothing, so its records
    are not read (``never_runs``; KNOWN_ISSUES.md K30): every reader of S1, H2 and F1 takes them from here."""
    kfs = getattr(mapdump.fragment(), "key_froms", None) or []
    if _RUNS[0] is not kfs:
        _RUNS[:] = [kfs, [kf for kf in kfs if not never_runs(kf)]]
    return _RUNS[1]


def is_birth(kf) -> bool:
    """Whether the map says *kf*'s write makes a new row (its ``new_row``, BI): an INSERT whose key is new -- a
    sequence's, RETURNING's, a value made in the function, or one not carried from a row or object of the table's own
    class -- not the same entity moved within its class (an archive) or rewritten (DELETE and INSERT of the whole
    key).  The one place S1 reads the mark; an INSERT the map did not read, or a call that updates the table too, says
    None, and is no birth."""
    return getattr(kf, "new_row", None) is True


#: ``[the key_froms list indexed, its index]`` -- the list is held so that its id is never another list's
_BIRTHS: list = [None, {}]


def birth_marks(owner: str, subject: str) -> dict[int, list]:
    """``{line: [KeyFrom, ...]}``: for each line *owner* writes *subject* on (a KeyFrom's ``writes``), the KeyFroms
    writing it there.  The map puts its statement's mark on each KeyFrom of an INSERT, one per key column, so a
    KeyFrom of another kind of write at the same statement (an UPDATE's term) says no birth for the line."""
    kfs = key_froms()
    if _BIRTHS[0] is not kfs:
        index: dict[tuple, dict[int, list]] = {}
        for kf in kfs:
            for subj, lines in kf.writes.items():
                for line in lines:
                    index.setdefault((kf.owner, subj), {}).setdefault(line, []).append(kf)
        _BIRTHS[:] = [kfs, index]
    return _BIRTHS[1].get((owner, subject), {})


def born(branch, marks: dict[int, list]) -> bool:
    """Whether *branch* is a birth: each of its lines (``line`` and ``other_lines``) is written only by KeyFroms the
    map marks as making a new row (*marks*, ``birth_marks``).  A line no KeyFrom names is not known to be one, so the
    branch is not."""
    lines = [branch.line, *(getattr(branch, "other_lines", None) or [])]
    return all(marks.get(line) and all(is_birth(kf) for kf in marks[line]) for line in lines)


def birth_why(owner: str, subject: str, lines: list[int]) -> str:
    """Why the map calls *owner*'s writes of *subject* at *lines* births: each statement's ``new_row_by`` and the line
    that decided it (``new_row_line``), e.g. "sequence at 623"."""
    marks = birth_marks(owner, subject)
    said = {(getattr(kf, "new_row_by", "") or "new", getattr(kf, "new_row_line", None))
            for line in lines for kf in marks.get(line, ())}
    return ", ".join(f"{by} at {at}" if at else by for by, at in sorted(said, key=str))


def births_of(writers: list, owner: str, subject: str, observed: str) -> list[int]:
    """*owner*'s branches of another value than *observed* that ``arms_of`` leaves out as births, by line: a write that
    makes the row moves no row off the value (insertFilesForDataset_JEDI's files INSERTed ready, staging or pending;
    KNOWN_ISSUES.md K29)."""
    marks = birth_marks(owner, subject)
    return sorted({b.line for j in writers if j.owner == owner for b in j.branches
                   if b.line is not None and b.outcome != observed and born(b, marks)})


def born_lines(writers: list, owner: str, subject: str) -> set[int]:
    """*owner*'s branch lines of *subject* that are births (``born``), whatever they write: a write that makes the row
    writes no row that was there before (KNOWN_ISSUES.md K29)."""
    marks = birth_marks(owner, subject)
    return {b.line for j in writers if j.owner == owner for b in j.branches if b.line is not None and born(b, marks)}


def unborn_junctions(junctions: list, subject: str) -> list:
    """*junctions* with their branches of *subject* that are births (``born``) left out, for the map's reach
    (``reach.moving_off``), which reads arms off the junctions' branches as ``arms_of`` does but knows no KeyFrom: an arm
    S1 leaves out as a birth is no arm of a mover's own def for K14's keep (``reach_said``), nor of a writer its calls
    reach -- ``updateJobStatus``'s new FileSpec of type zipinput (434) kept it a mover of ``FileSpec.type=input``
    (KNOWN_ISSUES.md K29)."""
    return [
        _with_branches(j, [b for b in j.branches if not born(b, birth_marks(j.owner, subject))] if j.subject == subject else j.branches)
        for j in junctions
    ]


def on_rows(junctions: list) -> list:
    """*junctions* without the branches the map says write no row (``Branch.off_row``, ix's OS): the arm assigns on an
    object the records prove no write of its class's tables holds, or none takes whole or by the attribute -- a spec
    the function makes and drops, as ``getTaskWithJumbo_JEDI``'s useJumbo.  No writer of the row's value, as H1's
    readings and ``reach.moving_off`` already read it; a junction left with no branch goes, its owner no writer.  A
    dump built before OS says none."""
    out = []
    for j in junctions:
        kept = [b for b in j.branches if not getattr(b, "off_row", None)]
        if kept or not j.branches:
            out.append(_with_branches(j, kept))
    return out


def _with_branches(j, kept: list):
    """*j* with *kept* for its branches: itself where none is left out, else a copy (a dump's model or a stand-in)."""
    if len(kept) == len(j.branches):
        return j
    if hasattr(j, "model_copy"):
        return j.model_copy(update={"branches": kept})
    out = copy.copy(j)
    out.branches = kept
    return out


#: what the fields of an object no write in its function persists reach, as the map's KeyFrom of no term says
UNPERSISTED = (KEY_HANDED, KEY_BOUND, KEY_RETURNED, KEY_KEPT, KEY_NO_USE)

#: ``[the key_froms list, {(owner, statement): the read KeyFroms of that statement}]`` -- the list is held so that its id
#: is never another list's
_READS: list = [None, {}]


def key_chain(kf, rs) -> tuple[bool, list[str]]:
    """Whether *kf*'s key comes from *rs*'s rows -- a ``rows`` source at *rs*, or at a statement of *rs*'s owner whose
    rows are taken by a key that comes from them, read through that statement's own KeyFroms (``reads``) --, and the
    terms the map did not read on the way there (an ``unread``, or a source not followed).  resetInputToReGenCoJumbo_JEDI
    sets ready at 5038 the files of 5026's rows; 5026 takes the job's files by the PandaIDs of 5009's rows, and 5009
    takes those by 5001's datasetID and fileID: 5038's key is 5001's rows' (the movers' O6, 2026-10-29).  Whatever
    column fills a key on the way counts (KNOWN_ISSUES.md K28).  Another function's rows are not followed: *rs*'s rows
    are its owner's, and come to no statement before it."""
    kfs = key_froms()
    if _READS[0] is not kfs:
        reads: dict[tuple, list] = {}
        for one in kfs:
            if getattr(one, "reads", False):
                reads.setdefault((one.owner, getattr(one, "statement", None)), []).append(one)
        _READS[:] = [kfs, reads]
    unread, seen, todo = [], set(), [kf]
    while todo:
        one = todo.pop()
        for s in one.sources:
            if s.kind != KEY_ROWS or s.site_owner != rs.owner:
                continue
            if s.site_line == rs.line:
                return True, unread
            if s.site_line in seen:
                continue
            seen.add(s.site_line)
            for back in _READS[1].get((rs.owner, s.site_line), []):
                if back.unread:
                    unread.append(f"{back.term} at {s.site_line} ({back.unread})")
                unread += [f"{back.term} at {s.site_line} (not followed: {x.name})" for x in back.sources if x.kind == KEY_UNKNOWN]
                todo.append(back)
    return False, unread


def elsewhere_arms(writers: list, rs, subject: str, arms) -> tuple[int, ...]:
    """*arms* (the arms of *rs*'s owner, by line), where none of them is *rs*'s: none lies in *rs*'s ``region``, and the
    map names the row of each by a key it read whole that *rs*'s rows do not fill, on the way either (``key_chain``):
    each writes a row named elsewhere -- ``updateJobStatus``'s 241 the pilot's own job by its ``PandaID``, a parameter,
    where 294 selects the co-jumbo jobs and sets their modificationTime alone.  ``()`` where any arm may be *rs*'s:
    in the region, keyed by its rows, with no KeyFrom, or with a term or a term on the way the map did not read."""
    lo, hi = (list(rs.region) or [rs.line, rs.line])[:2]
    if not arms or any(lo <= line <= hi for line in arms):
        return ()
    named: dict[int, list] = {}
    for kf in key_froms():
        if kf.owner != rs.owner or subject not in kf.writes:
            continue
        lines = set(kf.writes[subject])
        for j in writers:
            if j.owner == rs.owner:
                for b in j.branches:
                    if b.line in arms and lines & {b.line, *(getattr(b, "other_lines", None) or [])}:
                        named.setdefault(b.line, []).append(kf)
    for line in arms:
        if not named.get(line):
            return ()
        for kf in named[line]:
            if not kf.term or kf.unread or any(s.kind == KEY_UNKNOWN for s in kf.sources):
                return ()
            mine, unread = key_chain(kf, rs)
            if mine or unread:
                return ()
    return tuple(sorted(arms))


def keyed_arms(writers: list, rs, subject: str, arms) -> tuple[list[int], dict[int, list[str]], dict[int, list[str]]]:
    """``(keyed, unread, unpersisted)`` over *arms*, the arms of *rs*'s owner by line.  *keyed*: those whose write the
    map says names its row by a key filled from *rs*'s rows -- a ``KeyFrom`` of the owner's writing *subject* on a
    line of the arm's branch (its ``line`` or ``other_lines``), with a ``rows`` source of *rs*'s owner and line, or of
    a statement of the owner's that takes its rows by a key they fill (``key_chain``)
    (prepareTasksToBeFinished_JEDI keys the writes of its later loop by the jediTaskID its statements at 3723 and 3768
    select, past their region).  Whatever column of the rows fills the key counts: the map does not say which
    (KNOWN_ISSUES.md K28).  *unread*: the others with a row-naming term the map did not read -- its ``unread``, or a
    value it did not follow -- each with the terms and why: their rows may be *rs*'s too.  *unpersisted*: the others
    set on an object no write in the function persists -- a KeyFrom of no term, its ``unread`` what the object's
    fields reach (``UNPERSISTED``) -- each with the object and where its fields go: there is no key here to read,
    and the write that persists the value, if any, is the caller's or a callee's (datasetManager.run's ``file.status
    = "failed"`` at 713, a file of the job it hands on)."""
    keyed: set[int] = set()
    unread: dict[int, list[str]] = {}
    unpersisted: dict[int, list[str]] = {}
    for kf in key_froms():
        if kf.owner != rs.owner or subject not in kf.writes:
            continue
        lines = set(kf.writes[subject])
        at = {b.line for j in writers if j.owner == rs.owner for b in j.branches
              if b.line in arms and lines & {b.line, *(getattr(b, "other_lines", None) or [])}}
        if key_chain(kf, rs)[0]:
            keyed |= at
            continue
        if not kf.term and kf.unread in UNPERSISTED:
            to = ", ".join(h.split("::")[-1] for h in getattr(kf, "handed_to", None) or [])
            for line in at:
                unpersisted.setdefault(line, []).append(f"{kf.object} ({kf.unread}{': ' + to if to else ''})")
            continue
        why = [kf.unread] if kf.unread else [f"not followed: {s.name}" for s in kf.sources if s.kind == KEY_UNKNOWN]
        for line in at if why else ():
            unread.setdefault(line, []).append(f"{kf.term} ({'; '.join(why)})")
    return (sorted(keyed), {line: said for line, said in sorted(unread.items()) if line not in keyed},
            {line: said for line, said in sorted(unpersisted.items()) if line not in keyed})


def _by_line(said: dict[int, list[str]]) -> str:
    """``keyed_arms``' *unread* or *unpersisted* as a mover's line says it: "4300: PandaID=:PandaID (no bind found)"."""
    return "; ".join(f"{line}: {', '.join(what)}" for line, what in said.items())


@dataclass(frozen=True)
class Way:
    """A way ``_reached`` gives a function rows: the function, why, the statement whose rows they are (None for a writer
    whose write requires the value), and, where the statement is the function's own and its rows reach none of the
    function's arms, those arms (``elsewhere_arms``)."""

    owner: str
    why: str
    statement: object
    elsewhere: tuple = ()


def crossing_values(rs, owner: str) -> list[str]:
    """The values *rs*'s rows hold where they reach *owner*, one of its ``crosses_to``: those the routes back up to it
    select (the map's ``crossing_selects``, XB, 2026-10-29), or the statement's own ``selects`` for an owner the map
    gives none -- one reached from the statement's function, by a way no route matches, or from a statement with no
    route read -- and for every owner of a dump that has no such field."""
    return (getattr(rs, "crossing_selects", None) or {}).get(owner, rs.selects)


def _reached(follow, writers: list, subject: str, observed: str):
    """A ``Way`` for each way the map gives a function rows of *subject* holding *observed*, in the order S1 asks them;
    its statement is None for a writer whose write requires the value.  A statement that selects
    the value -- ``reader_statements`` also holds an UPDATE-only owner's statements of other values, which give none --
    hands its rows to its owner when an arm of the owner's lies in the statement's ``region``, the span the map reads
    where the rows go next, and to the writers among what the owner calls there (``crosses_to``).  Past the region, an
    arm of the owner's whose write names its row by a key the statement's rows fill is the statement's too, and one
    whose key the map did not read, or whose value no write in the function persists, keeps the owner, said
    (``keyed_arms``); either way the statement hands its rows on
    as one whose owner has no arm in the region does: to what it calls there, or it keeps them (a mover with no arm of
    its own: KNOWN_ISSUES.md K14) -- and where the owner has arms, each naming a row the map says the rows do not fill
    the key of, the way says them (``elsewhere``): S1 asks the map's reach of it as of a mover with none.  A crossing
    hands on the value only where the routes back up to it select it (``crossing_values``): 1898's rows reach
    setCloudToTasks_JEDI only back up TaskBroker 259, whose simTasks take the branch that selects no status.  A
    statement that aggregates (the map's ``result``) gives none: what it hands on is a value computed from the rows,
    not a row (getSiteToRunRateStats' count of a site's jobs goes to JobGenerator, which picks none of them up); one
    that varies, or whose result the map did not read, is read as returning rows.
    Then each writer whose write requires the value (``guarded_by``).  Last, a statement that reads a selecting
    statement's rows again hands them on to writers (``read_again``).  A writer that takes a row by its key alone is in
    none (KNOWN_ISSUES.md K16)."""
    asked = f"{subject}={observed}"
    writing = {j.owner for j in writers if arms_of(writers, j.owner, subject, observed)}
    for rs in follow.reader_statements:
        if asked not in rs.selects or rs.result == RESULT_AGGREGATE:
            continue
        reader = f"{rs.owner.split('::')[-1]}@{rs.line}"
        lo, hi = (list(rs.region) or [rs.line, rs.line])[:2]
        arms = arms_of(writers, rs.owner, subject, observed)
        if any(lo <= line <= hi for line in arms):
            yield Way(rs.owner, f"{reader} selects the value", rs)
            for callee in rs.crosses_to:
                if callee in writing and asked in crossing_values(rs, callee):
                    yield Way(callee, f"{reader} hands the rows it selects to it", rs)
            continue
        keyed, unread, unpersisted = keyed_arms(writers, rs, subject, arms)
        if keyed:
            yield Way(rs.owner, f"{reader} selects the value, and its rows fill the key of the write(s) at {keyed}", rs)
        elif unread or unpersisted:
            why = []
            if unread:
                why.append(f"the map could not read what fills the key of the write(s) at {list(unread)} "
                           f"({_by_line(unread)})")
            if unpersisted:
                why.append(f"no write in the function persists the value(s) set at {list(unpersisted)} "
                           f"({_by_line(unpersisted)})")
            yield Way(rs.owner, f"{reader} selects the value, and {', and '.join(why)}: kept", rs)
        if rs.crosses_to:
            for owner in rs.crosses_to:
                if asked in crossing_values(rs, owner):
                    yield Way(owner, f"{reader} hands the rows it selects to it", rs)
        else:
            elsewhere = () if keyed or unread or unpersisted else elsewhere_arms(writers, rs, subject, arms)
            yield Way(rs.owner, f"{reader} selects the value", rs, elsewhere)
    for g in follow.guarded_by:
        yield Way(g.owner, "its write requires the value", None)
    for rs in follow.reader_statements:
        if asked in rs.selects and rs.result != RESULT_AGGREGATE:
            yield from read_again(writers, writing, subject, observed, rs)


@dataclass(frozen=True)
class ReadAgain:
    """A statement that reads another's rows again, in the fields ``_reached`` reads of a selecting one."""

    owner: str
    line: int
    region: list
    crosses_to: list


#: ``[the key_froms list, {(owner, line): [(read KeyFrom, its rows source)]} of the sources reading that statement's
#: rows again, the read sites by (owner, line)]`` -- the list is held so that its id is never another list's
_AGAIN: list = [None, {}, {}]


def read_again(writers: list, writing: set[str], subject: str, observed: str, rs):
    """``_reached``'s ways through the statements that read *rs*'s rows again: a read KeyFrom (``reads``) with a
    ``rows`` source at *rs*'s owner and line that is ``same_rows`` -- its statement names them by every column of the
    table's key, from those columns of the rows -- and whose ``never_at`` does not hold the value: getTasksToBeFinished_JEDI
    reads 1676's tasks again at 1731 by ``jediTaskID`` for each status but ``scouted`` (1731 is in the else of ``if
    taskStatus == "scouted":``), and 1731 hands them to ``post_process_tasks`` (the movers oracle's A11).  Such a
    statement hands the rows on as a selecting one does: to its owner where an arm of the owner's lies in its region
    or its rows fill an arm's key (``keyed_arms``), and to what it calls (``crosses_to``) -- but only to a function with
    an arm off the value of its own (*writing*), as a selecting statement's region does: one with none came in as a
    mover for K14 to drop, and as one that L2 could no longer rule out.  Through calls (the source's ``via``), to the
    writers among the junctions the rows reach (``reaches``).  One hop: a statement reading those rows once more is not
    followed.  A call is taken whatever test it runs under, and a function reached may write other rows than those it
    is handed (KNOWN_ISSUES.md K31).  Yields as ``_reached``, *rs* the statement: it is the one whose rows were taken."""
    kfs = key_froms()
    if _AGAIN[0] is not kfs:
        again: dict[tuple, list] = {}
        for kf in kfs:
            if getattr(kf, "reads", False):
                for s in kf.sources:
                    if s.kind == KEY_ROWS and getattr(s, "same_rows", False):
                        again.setdefault((s.site_owner, s.site_line), []).append((kf, s))
        sites = {(r.owner, r.anchor.line_start): r for r in getattr(mapdump.fragment(), "read_sites", None) or []}
        _AGAIN[:] = [kfs, again, sites]
    asked = f"{subject}={observed}"
    reader = f"{rs.owner.split('::')[-1]}@{rs.line}"
    for kf, s in _AGAIN[1].get((rs.owner, rs.line), []):
        if asked in (getattr(s, "never_at", None) or []):
            continue
        said = f"{reader} selects the value, and {kf.owner.split('::')[-1]}@{kf.statement} reads its rows again"
        if getattr(s, "via", None):
            for owner in getattr(s, "reaches", None) or []:
                if owner in writing:
                    yield Way(owner, f"{said} and hands them to it", rs)
            continue
        site = _AGAIN[2].get((kf.owner, kf.statement))
        if site is None:
            continue
        r = ReadAgain(site.owner, site.anchor.line_start, list(site.region), list(site.crosses_to))
        lo, hi = (r.region or [r.line, r.line])[:2]
        arms = arms_of(writers, r.owner, subject, observed)
        if any(lo <= line <= hi for line in arms):
            yield Way(r.owner, said, rs)
        else:
            keyed, _, _ = keyed_arms(writers, r, subject, arms)
            if keyed:
                yield Way(r.owner, f"{said}, and they fill the key of the write(s) at {keyed}", rs)
        for owner in r.crosses_to:
            if owner in writing:
                yield Way(owner, f"{said} and hands them to it", rs)


def movers_of(follow, writers: list, subject: str, observed: str) -> dict[str, str]:
    """The functions that can move *subject* on from *observed*, each with the first way the map gives it rows
    (``_reached``)."""
    found: dict[str, str] = {}
    for way in _reached(follow, writers, subject, observed):
        found.setdefault(way.owner, way.why)
    return found


def elsewhere_of(follow, writers: list, subject: str, observed: str) -> dict[str, list[int]]:
    """The movers every way of which is a statement of their own whose rows reach none of their arms (``Way.elsewhere``),
    each with those arms: the rows the mover selects stay with it, and its writes name other rows -- updateJobStatus at
    ``waiting``, its 294 selecting co-jumbo jobs to set their modificationTime, its 241 writing the pilot's own job by
    its PandaID.  S1 asks the map's reach of each as of a mover with no arm of its own, those arms left out of where it
    ends (``Rerun._rows_elsewhere``)."""
    ways: dict[str, list[tuple]] = {}
    for way in _reached(follow, writers, subject, observed):
        ways.setdefault(way.owner, []).append(way.elsewhere)
    return {owner: sorted(set().union(*held)) for owner, held in ways.items() if all(held)}


def by_key_of(writers: list, movers, subject: str, observed: str) -> list[str]:
    """The writers with an arm that can move *subject* on from *observed* that no way of ``_reached`` gives rows: each
    takes a row by its key, as far as the map says (the pilot's report into ``updateJobStatus``, a kill command into
    ``killJob``; KNOWN_ISSUES.md K15 / K16), in the writers' order."""
    owners = dict.fromkeys(j.owner for j in writers)
    return [o for o in owners if o not in movers and arms_of(writers, o, subject, observed)]


def paths_of(follow, writers: list, subject: str, observed: str) -> dict[str, list]:
    """Each mover's selecting statements whose rows reach it (``_reached``); none for a writer whose write requires the
    value and no statement gives it rows."""
    found: dict[str, list] = {}
    for way in _reached(follow, writers, subject, observed):
        paths = found.setdefault(way.owner, [])
        if way.statement is not None and way.statement not in paths:
            paths.append(way.statement)
    return found


@dataclass(frozen=True)
class Wake:
    """A write a mover's rows wait on besides the mover (``wakers_of``): the statement giving the mover rows, its term on
    another column, the value written there, the writing branch, and the functions keying that write that move the
    row on from the value themselves (none: not a candidate)."""

    mover: str
    reader: str
    term: str
    writes: str
    owner: str
    line: int
    candidate: tuple = ()


def wakers_of(follow, writers: list, subject: str, observed: str) -> dict[str, list[Wake]]:
    """Each mover's wakers at *observed*: of the map's ``wakers`` of each statement giving it rows (``paths_of``; ix's
    6 b), those the map states (the branch's outcome is the value) and whose write is handed rows selected at the value
    (``keyed_at``) -- restartTasksForContentsUpdate_JEDI's 1844 sets toupdate the datasets of the tasks its 1830 selects
    at defined, which ContentsFeeder's 130 asks of a task at defined.  A write any pass may make, or one keyed by rows
    of other values, is not said.  Where a function whose statement keys the write moves the subject on from the value
    itself (an arm of its own, ``arms_of``), the waker is a candidate: that pass may take the row on before the mover
    sees it (prepareTasksToBeFinished_JEDI's 4001 for JobGenerator's 1871 at task running, keyed by its own 3694).
    None said is not "nothing wakes it": the map reads no command row and no Python test (KNOWN_ISSUES.md K39)."""
    asked = f"{subject}={observed}"
    sites = {(r.owner, r.anchor.line_start): r for r in getattr(mapdump.fragment(), "read_sites", None) or []}
    by_name = {r.name: r for r in sites.values()}
    found: dict[str, list[Wake]] = {}
    for mover, statements in paths_of(follow, writers, subject, observed).items():
        for rs in statements:
            for w in getattr(sites.get((rs.owner, rs.line)), "wakers", None) or []:
                if asked not in w.pairs or not w.stated or asked not in w.keyed_at:
                    continue
                keying = {by_name[n].owner for n in w.keyed_by if n in by_name and asked in by_name[n].selects}
                moving = tuple(sorted(o for o in keying if arms_of(writers, o, subject, observed)))
                wake = Wake(mover, f"{rs.owner.split('::')[-1]}@{rs.line}", w.term, f"{w.subject}={w.value}", w.owner,
                            w.line, moving)
                if wake not in found.setdefault(mover, []):
                    found[mover].append(wake)
    return found


def wake_said(w: Wake) -> str:
    """One waker as S1 says it under its mover."""
    said = (f"its rows wait also on {w.owner.split('::')[-1]}@{w.line} writing {w.writes}, which {w.reader} asks "
            f"({w.term!r}), for rows selected at the value")
    if w.candidate:
        said += (f" -- a candidate: {', '.join(o.split('::')[-1] for o in w.candidate)}, whose rows key it, moves the "
                 "row on itself")
    return said


def woken_of(entries: list[dict], wakes: dict[str, list[Wake]]) -> dict[str, list[Wake]]:
    """The movers S1 drops as reaching no write that the movers it keeps wait on (``wakers_of``), each with those
    wakers: restartTasksForContentsUpdate_JEDI at task defined, for ContentsFeeder's 130.  What a dropped mover's own
    rows wait on is not asked."""
    staying = [e["owner"] for e in entries if e["verdict"] != "reaches-no-write"]
    found: dict[str, list[Wake]] = {}
    for e in entries:
        woken = [w for m in staying for w in wakes.get(m, []) if w.owner == e["owner"]]
        if e["verdict"] == "reaches-no-write" and woken:
            found[e["owner"]] = woken
    return found


def woken_said(woken: list[Wake]) -> str:
    """A mover dropped as one that reaches no write, said as the waker it is of the movers kept."""
    writes = "; ".join(f"{w.writes} at {w.line}" for w in woken)
    movers = sorted({f"{w.mover.split('::')[-1]} ({w.reader})" for w in woken})
    return (f"not a mover, but it writes {writes} for rows selected at the value, which {', '.join(movers)} asks: "
            "its waker (the map's wakers; KNOWN_ISSUES.md K39)")


def refused_arms(writers: list, owner: str, subject: str, observed: str) -> list[str]:
    """*owner*'s writes of another value that ``arms_of`` drops because each requires the row to hold other values of
    the subject's column: ``line writes outcome only on a row holding column in [values]``."""
    column = subject.partition(".")[2].lower()
    marks = birth_marks(owner, subject)
    out = []
    for j in writers:
        if j.owner != owner:
            continue
        for b in j.branches:
            if b.line is None or b.outcome == observed or born(b, marks):
                continue
            for term in b.row_precondition or []:
                rv = required_values(term)
                if rv and rv[0] == column and observed not in rv[1]:
                    out.append(f"{b.line} writes {b.outcome} only on a row holding {column} in {rv[1]}")
                    break
    return out


def argument_said(param: str | None, arg: ast.expr, value: object) -> str:
    """``onlyActive = True`` for a literal, ``jediTaskID = 12345`` where the argument is the parameter's name,
    ``param = <argument> = value`` otherwise."""
    if param is None or (isinstance(arg, ast.Name) and arg.id == param):
        return f"{ast.unparse(arg)} = {value!r}"
    if isinstance(arg, ast.Constant):
        return f"{param} = {value!r}"
    return f"{param} = {ast.unparse(arg)} = {value!r}"


def _said_value(value: object) -> object:
    """*value* as a rows ask carries it: as JSON holds it, its repr otherwise, ``"unknown"`` where the run's is not
    known."""
    if value is E.UNKNOWN:
        return "unknown"
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return repr(value)
    return value


class Stop(Exception):
    """A step that cannot go on.  *code* names its class where a rule decides it (``asks``: asks of the logs the records
    do not answer; ``rows``: rows the rows file does not have, asked (``Rerun.ask_rows``), where no other rule names
    it; ``one-of``: H2 names the arms that wrote last or may have, KNOWN_ISSUES.md K38; a KNOWN_ISSUES.md number), for
    result.json's ``outcome.code``; None where none is named yet."""

    def __init__(self, what: str, code: str | None = None) -> None:
        super().__init__(what)
        self.code = code


def tree_stamp(path: Path) -> str:
    """``git describe --dirty`` of *path*, or what it is when it is not a git tree (a pinned copy)."""
    done = subprocess.run(["git", "-C", str(path), "describe", "--dirty", "--always"], capture_output=True, text=True)
    return done.stdout.strip() if done.returncode == 0 else f"{path} (not a git tree)"


# ---------------------------------------------------------------- the questions


@dataclass
class Asked:
    """Who wrote this value of a row field."""

    subject: str
    observed: str
    key: str


@dataclass
class SetBy:
    """Which call set what a guard reads through a method."""

    method: str
    reading: object
    question: object  # the run
    guard_record: int
    env: E.Env


@dataclass
class Raised:
    """What raised under the site chosen, in its run."""

    fn: R.Function
    question: object


@dataclass
class Missing:
    """Who makes an object the outside says is missing."""

    obj: str
    inner: object  # the corpus frame at the boundary
    question: object  # the run the outside said it was missing in


@dataclass
class Branch:
    """What the guard of the branch taken held."""

    fn: R.Function
    exit: object  # the Verdict chosen
    run: object
    env: E.Env


@dataclass
class Settled:
    """Whether another decision this investigation read computed the same atom the other way."""

    fn: R.Function
    exit: object
    steps: list
    env: E.Env
    settled: dict[str, bool]


@dataclass
class Pair:
    """Where two computations of one atom part."""

    atom: str
    a: V.Side
    b: V.Side


@dataclass
class FalseAtom:
    """A guard inside the corpus that an exception says came out false."""

    frame: object
    chain: object
    question: object


@dataclass
class Reproduce:
    """Whether the rebuilt inputs raise the same exception, and whether the alternative does not."""

    fn: R.Function
    frame: object
    chain: object
    walk: object
    part: object
    then: "Field"


@dataclass
class Stuck:
    """Why nothing moved this value on."""

    subject: str
    observed: str
    key: str


@dataclass
class Arm:
    """A write of the subject that would have moved it: its line, what it writes, what it needs."""

    line: int
    outcomes: list[str]
    guards: list[str]


@dataclass
class Passes:
    """Which way each pass of a mover went: its runs of the key in *logs*, over *asking*'s window, asked with S1's
    *owners* (so the door answers as it answered S1)."""

    owner: str
    logs: list[str]
    arms: list[Arm]
    key: str
    asking: W.Asking
    owners: Owners | None = None


@dataclass
class Fork:
    """Questions that each go on alone, from the same point: every mover that ran in the window (S1)."""

    branches: list
    labels: list[str]


@dataclass
class Blocked:
    """What kept every pass from the arms that move the value."""

    fn: R.Function
    exit_line: int
    passes: list
    envs: list
    arms: list[Arm]


@dataclass
class Input:
    """Where a name a guard reads was bound, in each pass."""

    fn: R.Function
    guard: str
    line: int  # the ``if`` the guard is the test of
    passes: list
    envs: list
    outcomes: list[str]  # what the arm(s) it kept the passes from would have written


@dataclass
class Rows:
    """What a call that returns rows by a key returned, in each pass."""

    fn: R.Function
    name: str
    step: object
    passes: list
    envs: list
    outcomes: list[str]


@dataclass
class Field:
    """A row field's value for this instance."""

    expr: str
    why: str
    env: E.Env
    #: the method call computed from the field, and the ``rel::function`` the walk
    #: says computes it -- recomputed with PanDA's own code once the field is known
    computed: str = ""
    method: str = ""


@dataclass
class Late:
    """Why this value stayed on so long: what moved it on in the end, and when."""

    subject: str
    observed: str
    key: str


@dataclass
class Wrote:
    """Who wrote a value, as H1 and H2 answer it: the map's readings, the run of the key at the time, the pick."""

    readings: list
    question: object  # the run
    cands: list
    env: object
    chosen: object
    #: the caller's verdict where *chosen* is the arm of a writer that prints nothing, read on that caller's run at its
    #: call (:class:`StandIn`): the run's marks and the guard's record are the call's, the guards the arm's
    via: object = None
    #: the statuslog row whose pass wrote it, where H2 asked at a row (None for a value held or a value that is its time)
    row: dict | None = None
    #: the window H2 asked the logs over, ``(lo, hi)``
    window: tuple[str, str] | None = None
    #: what of the window the chosen writer's logs were not searched for, ``{log: Log.holes}``
    unsearched: dict = field(default_factory=dict)

    @property
    def reading(self):
        c = self.chosen.candidate
        return next(r for r in self.readings if r.file == c.rel and r.owner.endswith("::" + c.func))


def _guard_read(text: str, env: E.Env) -> object:
    """A path condition read in *env*: True or False where the values read decide it; anything else is not read."""
    try:
        return E.evaluate(text, env)
    except SyntaxError:
        return E.UNKNOWN


@dataclass(frozen=True)
class StandIn:
    """A writer arm whose function prints nothing (KNOWN_ISSUES.md K25), read on the run of the caller a history pair
    names, at the call on the way to it (``HistoryPair.caller``, ``point``; attr-time-shape.md 5).  That call shown run
    says the writer was called, not that the arm wrote: its own path condition, and above level 1 the functions
    between, are left open unless the values read decide them."""

    arm: str  # the writer's ``file::function:line``
    level: int
    guards: tuple[str, ...]

    def left(self, env: E.Env) -> list[str] | None:
        """What stays open of the arm once its call ran, or None where a guard is false in *env* (it did not write)."""
        out = []
        for g in self.guards:
            got = _guard_read(g, env)
            if got is False:
                return None
            if got is not True:
                out.append(f"its condition {g!r} not read")
        if self.level > 1:
            out.append(f"the {self.level - 1} function(s) between the call and it not read")
        return out


@dataclass
class Possible:
    """A writer arm that wrote, or may have, in the pass of one statuslog row (H2's scan of the rows)."""

    arm: str
    row: dict | None
    wrote: bool
    #: what is open where it may have: a condition not read, a log not searched whole, the run not showing the line
    premises: list[str]
    #: the answer for a writer shown to write, for the steps after H2 -- one that prints nothing on its caller's run
    answer: Wrote | None = None
    #: the call a writer that prints nothing is read at (``file::function:line``), where it stands in for one
    by: str = ""


@dataclass
class Missed:
    """Why each statement asking rows on the value passed this row over until the value moved on."""

    subject: str
    observed: str
    next_value: str  # what the statuslog shows after the observed value
    since: datetime.datetime
    moved: datetime.datetime
    statements: list  # the map's StatementReading of every statement asking rows on subject=observed
    mover: str  # rel::function of the run that moved the value on
    run: object  # that run
    #: the words for the window's end and start: late's "it moved on" and "the window"; stuck reads at one moment
    end: str = "it moved on"
    start: str = "the window"
    #: the rows file's ``after`` holds the values at the moment read (stuck: near the window's end); late's is after it
    at_end: bool = False
    #: M2: the movers (rel::function -> how the map gives them rows first), their selecting statements, the writers
    #: reached by their key only, those whose write refuses the row, and the map's writers of the subject
    movers: dict = field(default_factory=dict)
    paths: dict = field(default_factory=dict)
    by_key: list = field(default_factory=list)
    refusing: list = field(default_factory=list)
    writers: list = field(default_factory=list)
    gaps: dict = field(default_factory=dict)


@dataclass
class Ordered:
    """A statement returning only the first rows of an order, and the call of it that moved the value on."""

    statement: object
    run: object
    q: Missed


#: the clocks the map names that read one UTC clock: PanDA's Oracle session runs in UTC (WrappedCursor.initialize,
#: ``ALTER SESSION SET TIME_ZONE='UTC'``); SYSDATE, SYSTIMESTAMP and datetime.now read a host's zone, unknown here
UTC_CLOCKS = {"naive_utcnow", "datetime.utcnow", "datetime.now(utc)", "CURRENT_DATE", "CURRENT_TIMESTAMP"}
#: why a term on a column the row holds NULL fails (SQL's NULL; ``_leaf``, ``_pair_fails``)
#: movers_of's word for a mover the map's guarded_by gives (a writer whose write requires the value)
GUARDED = "its write requires the value"


def _identity_attrs(cls: str) -> list[str]:
    """The ``Class.attr`` the dump's tables name *cls*'s identity by; none where no table of it names one."""
    fragment = mapdump.fragment()
    return next((t.identity_attrs for t in (fragment.tables if fragment else []) if t.spec_class == cls and t.identity_attrs), [])


def _statement_subjects(st) -> set[str]:
    """Every subject *st*'s pairs and condition trees read."""
    subjects = {p.partition("=")[0] for p in [*st.selects, *st.excludes, *st.refuses, *st.restricts]}
    return subjects | {leaf for r in st.condition_trees for leaf in _subjects_of(r.tree)}


NULL_FAILS = "the row holds NULL, which no comparison but IS NULL takes"
#: L2's verdicts on a statement, strongest first
KINDS = ["refused", "refused on a call", "refused when", "if required", "not settled"]


@dataclass
class Leaf:
    """A term of a condition tree that is false for the row all the window, and why."""

    subject: str
    text: str
    why: str


def _added_under(st) -> dict[tuple[str, str], dict] | None:
    """The map's ``added_under`` of a statement, by (field, term); None when the map did not compute it."""
    entries = getattr(st, "added_under", None)
    if entries is None:
        return None
    entries = [e if isinstance(e, dict) else e.model_dump() for e in entries]
    return {(e["field"], e["term"]): e for e in entries}


def _by_subject(pairs) -> dict[str, set[str]]:
    """``Class.attr=value`` pairs, by subject."""
    out: dict[str, set[str]] = {}
    for pair in pairs:
        subject, _, value = pair.partition("=")
        out.setdefault(subject, set()).add(value)
    return out


def _at(where: str) -> str:
    return where.rsplit("/", 1)[-1]


def _tests(place: dict) -> str:
    return " and ".join(f"{t['test']} ({_at(t['at'])})" for t in place["tests"]) or "no test"


def _builds(under: dict | None, field: str, terms: list[str], took: bool, decided: bool) -> tuple[str, str]:
    """Which builds refuse the row on a term it fails: L2's verdict and the words for it.  *terms* are the field's
    spellings of it -- one condition or exclude, or every pair of a subject.  *decided*: the map says every build
    refuses the row on it -- ``conditions``, ``refuses``, ``restricts``.  Otherwise each build that holds a term
    deciding it refuses the row (a ``conditions`` term fails whole); one only allowing it, an ``OR`` another branch
    of which may hold, refuses nothing; a term the map reads no decision of is unknown.  *took*: the call that moved
    the value on took the row, so no build of it refused the row."""
    if under is None:
        return "if required", ("not in the build of the call that took it" if took else "if the term is in every build")
    if decided:
        if took:
            return "if required", "in every build, says the map -- yet the call took it"
        return "refused", "in every build"
    said: dict[str, str] = {}
    if any((field, t) not in under for t in terms):
        said["in every build, no term the map reads deciding it"] = "if required"
    for e in (under[(field, t)] for t in terms if (field, t) in under):
        if not e["places"]:
            said[f"not placed by the map, {e['term']}: {e['unplaced']}"] = "if required"
        for p in e["places"]:
            decides = True if field == "conditions" else p.get("decides")
            added = f"added under {_tests(p)} at {_at(p['fragment'])}"
            if decides:
                said[added] = "refused when"
            elif decides is None or not p["term"]:
                said[f"{added}, the map not saying whether its term decides it"] = "if required"
            else:
                said[f"allowed under {_tests(p)} at {_at(p['fragment'])}: {p['term']}"] = "not settled"
    words = list(said)
    if took:
        words.append("not in the build of the call that took it")
    return min(said.values(), key=KINDS.index), "; ".join(words)

def _when(text: str) -> datetime.datetime:
    """A record's ``2026-01-01 00:00:00,000`` or a statuslog row's ``2026-01-01 00:00:00.000000``, as a time."""
    return datetime.datetime.fromisoformat(text.replace(",", "."))


def _iso(text: str | None) -> str | None:
    """A time as result.json writes it: ISO 8601 in UTC with ``Z``, to the precision its source gives (a statuslog row's
    microseconds, a rows file's seconds)."""
    if not text:
        return None
    _when(text)  # a time, or ValueError
    return text.replace(",", ".").replace(" ", "T", 1) + "Z"


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _tree_sha256(root: Path) -> str:
    """One hash of the code a run imports from its tree: the sha256 of what ``shasum -a 256`` prints for every ``*.py``
    under *root*/bamboo, by path (``__pycache__`` left out) -- the map's code the querier calls is there, not in
    bamboo/querier/."""
    files = sorted(str(f.relative_to(root)) for f in (root / "bamboo").rglob("*.py") if "__pycache__" not in f.parts)
    return hashlib.sha256("".join(f"{_sha256(root / f)}  {f}\n" for f in files).encode()).hexdigest()


def _write(path: str, text: str) -> None:
    """*text* to *path* through a file renamed over it: a reader finds the whole file or none."""
    tmp = Path(f"{path}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _read_at(data: dict) -> list[dict] | None:
    """When the rows file's values were read, by section: a section's ``as_of`` and ``clock`` (statuslog, after, ...),
    and each distinct one the values of ``rows`` (by entity) and ``files`` carry; None where none says."""
    out = []
    for name, section in data.items():
        if isinstance(section, dict) and "as_of" in section:
            out.append({"section": name, "at": _iso(section["as_of"]), "clock": section.get("clock")})
        elif name in ("rows", "files"):
            groups = section.items() if isinstance(section, dict) else [("", r) for r in section]
            seen: dict[str, set] = {}
            for entity, row in groups:
                seen.setdefault(entity, set()).update(
                    (v["as_of"], v.get("clock")) for v in row.values() if isinstance(v, dict) and v.get("as_of"))
            for entity, times in seen.items():
                out += [{"section": f"{name}.{entity}" if entity else name, "at": _iso(at), "clock": clock} for at, clock in sorted(times)]
    return out or None


def _overlaps(run, window: Window) -> bool:
    return run.first[:19] <= window[1] and run.last[:19] >= window[0]


def _question_run(at: dict[str, list]):
    """The run H2's question is read in, of *at* (log -> the key's runs at the window): the one run of one log, or
    None, and each candidate is judged on its own run.  Runs in more than one log: the machines' clocks are not known
    finer than a second against the database's, so no run is the question's until the judge picks the candidate that
    ran (Fr1, 2026-10-14) -- taking the first would take the asker's order of the logs, not the case's.  Several runs
    of one log: a window can hold several passes of the key, and so can the row's second -- a mark that spans days
    covers it as well as the pass that wrote (the 'done' second of MR-T-53354352: a two-day
    prepareTasksToBeFinished_JEDI run and updateTask_JEDI's own)."""
    if len(at) != 1:
        return None
    (runs,) = at.values()
    return runs[0] if len(runs) == 1 else None


def _of_mover(run, owner: str) -> bool:
    """Whether *run* can be a run of the mover *owner*: a mark that names a method (a DB proxy's tagged logger,
    ``getJobs < SITE 2026-01-01T00:00:00 >``) is one call of that method, so it is the mover's only when the method is
    the mover's function -- a DB proxy's log holds every method's calls of the key (S1's rule, tmaeno 2026-10-13).  A
    mark that names none (``< jediTaskID=1 >``, the adder's PandaID) may be any function's of the worker.  The name is
    the tag's, which the method writes in its own comment: a comment copied from another method hides its runs."""
    m = re.match(r"([A-Za-z_]\w*) <", run.token)
    return m is None or m[1] == owner.split("::")[-1]


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _value_words(v) -> str:
    """A tree value as the map settles it: ``:creationTime = naive_utcnow - 604800 s``."""
    if v.base:
        offset = f" {'-' if v.offset_seconds < 0 else '+'} {abs(v.offset_seconds):g} s" if v.offset_seconds else ""
        return f"{v.text} = {v.base}{offset}"
    return f"{v.text} = {v.constant}" if v.constant != v.text else v.text


def _settled(v) -> bool:
    return not v.needs and not v.said


def _subjects_of(node) -> set[str]:
    """The subjects a condition tree's terms test."""
    if node.node in ("any", "all"):
        return {x for item in node.items for x in _subjects_of(item)}
    return {node.subject} if node.subject else set()


def _same_order(mine: tuple, ran: tuple) -> bool:
    """The map's order and one a run printed, item by item: a blanked item of the map's is whatever the run built there."""
    return len(mine) == len(ran) and all(
        a.blanked or (a.expression, a.descending) == (b.expression, b.descending) for a, b in zip(mine, ran, strict=True)
    )


def _db_read_where(query, binds: dict) -> tuple[list[list], list[tuple[str, str]]]:
    """The WHERE of *query*, a printed query of one table, as db_read's conditions with the run's *binds*
    (:func:`parts.condition`), and each term none says, with why.  Each term is read as the map reads a WHERE
    (``sqltext.row_tests``) and said whole or not at all: an OR is no one test, and dropping a term would count rows
    the call did not pass."""
    said, unsaid = [], []
    for term in query.where:
        tests = sqltext.row_tests(f"SELECT * FROM {query.spelled[0]} WHERE {term.text}")
        got = [PT.condition(test, binds) for test in tests]
        why = next((c for c in got if isinstance(c, str)), "" if tests else "an OR, or no one column's test")
        if why:
            unsaid.append((term.text, why))
        else:
            said += got
    return said, unsaid


def _intersect(a: list, b: list) -> list:
    """The spans (closed, ``(start, end)``) both lists hold."""
    return [(max(x0, y0), min(x1, y1)) for x0, x1 in a for y0, y1 in b if max(x0, y0) <= min(x1, y1)]


def _spans_words(spans: list, whole: tuple) -> str:
    """``all the window``, or each span ``from .. to ..``."""
    return "all the window" if spans == [whole] else ", ".join(f"from {a} to {b}" for a, b in spans)


def _union(spans: list) -> list:
    """The spans merged where they meet."""
    out: list = []
    for start, end in sorted(spans):
        if out and start <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out


class Rerun:
    def __init__(self, args: argparse.Namespace) -> None:
        mapdump.require("rerun")  # _investigate's walks and _writers_of ask mapdump.code_map
        A.made.clear()
        self.args = args
        self.mode, self.decide = args.patterns, args.decide
        self.data = json.loads(Path(args.rows).read_text())
        self.hand = json.loads(Path(args.hand).read_text()) if args.hand else {}
        self.hand_used: list[str] = []
        self.logs: Logs = RecordedLogs(args.records) if args.records else FileLogs(args.log)
        self.lines: list[str] = []
        self.hop = ""
        self.touched: list[tuple[R.Function, object]] = []  # (function, run) this investigation read
        #: what the runs read so far printed about a row, with where and when -- a value
        #: printed at one time and used for a decision at another says so in its source
        self.row_facts: dict[str, tuple[str, str]] = {}
        self.question = None  # the run the symptom's value changed in
        #: (hop, ask, why) of every ask of the logs the door had no answer for; written to --asks at the end
        self.unanswered: list[tuple[str, object, str]] = []
        #: the rows the rows file does not have, said so a fetch can answer them (``ask_rows``), each once by its text;
        #: written to --rows-asks at the end
        self.rows_asks: dict[str, dict] = {}
        self.windows_said: set = set()
        #: what result.json says besides the outcome: the steps opened, the END lines said (hop, line), each branch's
        #: end, the trees read at the start and the end, and what the steps put there (window, moved_by, movers)
        self.opened: list[str] = []
        self.ends: list[tuple[str, str]] = []
        self.branches: list[dict] = []
        self.stamps: dict[str, dict] = {}
        self.result: dict = {}
        #: the map's reach, read from the dump at its first ask (``_reach_of``): None before, False with none in the dump
        self._reach = None
        #: the dump's reach_graph as read (``_graph``): (the reach module, its ReachGraph, its rows); False with none
        self._graph_read = None
        #: ``_called_at``'s answers: (caller, callee) function -> the lines of the calls that go on into it
        self._calls_into: dict[tuple[str, str], list[str]] = {}
        #: H1's readings by (subject, value), read once (``_writers``); the runs F1 read the rows after (``rows_between``)
        self._readings: dict[tuple[str, str], list] = {}
        self._rows_read: set[str] = set()

    # ------------------------------------------------------------ plumbing
    def say(self, text: str = "") -> None:
        self.lines.append(text)
        print(text, flush=True)

    def need(self, what: str, found, code: str | None = None):
        if found:
            return found
        if self.hop in self.hand:
            self.say(f"HAND {self.hop}: {what} <- {self.hand[self.hop]!r}")
            self.hand_used.append(self.hop)
            return self.hand[self.hop]
        raise Stop(f"{self.hop}: {what}", code)

    def pick(self, what: str, judgement):
        """The judgement's choice, or the verdict ``hand.json`` names as ``func:line``."""
        if judgement.chosen is not None:
            return judgement.chosen
        if self.hop in self.hand:
            named = self.hand[self.hop]
            found = next((v for v in judgement.verdicts if f"{v.candidate.func}:{v.line}" == named), None)
            if found is not None:
                self.say(f"HAND {self.hop}: {what} <- {named!r}")
                self.hand_used.append(self.hop)
                return found
        raise Stop(f"{self.hop}: {what}")

    def start(self, hop: str, title: str) -> None:
        self.hop = hop
        self.opened.append(hop)
        self.say()
        self.say(f"== {hop} {title}")

    def end(self, what: str) -> None:
        """An END line: what the run found (result.json's ``outcome.text``)."""
        self.ends.append((self.hop, f"END {what}"))
        self.say(f"END {what}")

    def files_of(self, owner: str, today: list[str], lines) -> list[str]:
        """Where *owner*'s own rows are looked for (``R.own_log_files``); ``FILES_LOG`` gets both answers."""
        own = R.own_log_files(owner, list(today), lines)
        if os.environ.get("FILES_LOG"):
            saved = R.OWN_FILES
            R.OWN_FILES = True
            would = R.own_log_files(owner, list(today), lines)
            R.OWN_FILES = saved
            found = R._prints_of(owner, lines)
            basis = sorted({st.log_files_basis for st in found.statements}) if found is not None else None
            with open(os.environ["FILES_LOG"], "a") as out:
                out.write(json.dumps({"hop": self.hop, "owner": owner, "today": sorted(today), "own": would, "basis": basis}) + "\n")
        return own

    def warn_split(self, owner: str, lines) -> None:
        if split := R.split_files(owner, lines):
            self.say(f"{owner.split('::')[-1]}'s rows land in different files {split}: a row is read only in a run of a file it lands in (KNOWN_ISSUES.md K23)")

    def base_env(self, file_row: dict | None = None, at: str | None = None) -> E.Env:
        """The row values a step reads, for a run that began *at*: a value read at another time than the run's is not
        among them, and ``Env.unread`` says why (``held.at_run``, KNOWN_ISSUES.md K22)."""
        sp = self.data["spellings"]
        env = E.Env(rows={s: entity for entity, spellings in sp.items() if entity != "about" for s in spellings})
        rows, env.unread = HD.at_run(self.data, self.args.subject.partition(".")[0], self.args.key, at)
        env.at = at
        for entity, row in rows.items():
            env.table[entity] = {k: E.Value(v["value"], v["source"]) for k, v in row.items()}
        if file_row is not None:
            env.table["file"] = {k: E.Value(v["value"], v["source"]) for k, v in file_row.items()}
        for expr, (printed, source) in self.row_facts.items():
            env.put_printed(expr, printed, source)
        return env

    def learn(self, pairs: list[tuple[R.Function, object]]) -> None:
        """Keep what these runs printed about a row, for the steps after."""
        env = self.base_env()
        before = set(env.values)
        put = stable_holes([R.seen(f, q, self.mode) for f, q in pairs], env)
        for expr, printed in put.items():
            key = env.canonical(expr)
            if key not in before and "<" in key:
                self.row_facts[expr] = (printed, env.values[key].source)

    def runs(self, key: str, names: list[str], asking: W.Asking, owners: Owners | None = None) -> dict[str, list]:
        """The key's runs in each of the logs *names* the door can be asked for, over *asking*'s window and machine.

        Every log is asked before a stop, so that a door short of answers names all the step's asks at once.
        """
        out, unanswered, _ = self._ask(key, names, asking, owners)
        self._stop_unanswered(unanswered)
        return out

    def _ask(self, key: str, names: list[str], asking: W.Asking, owners: Owners | None = None) -> tuple[dict[str, list], list, dict[str, list[str]]]:
        """``runs()`` without the stop: the runs found, the asks the door had no answer for, and what of the window each
        log was not searched for (``Log.holes``)."""
        window = self.window(asking)
        out, unanswered, holes = {}, [], {}
        for name in self.logs.names(names):
            try:
                log = self.logs.log(name, key, window, machine=asking.machine, owners=owners)
            except Unrecorded as e:
                unanswered += [(ask, why or asking.why) for ask, why in e.asks]
                continue
            for note in log.notes + [f"not searched: {h}" for h in log.holes]:
                self.say(f"  {name}: {note}")
            if log.holes:
                holes[name] = log.holes
            out[name] = runs_for(log, key)
        return out, unanswered, holes

    def _stop_unanswered(self, unanswered: list) -> None:
        if unanswered:
            self.unanswered += [(self.hop, ask, why) for ask, why in unanswered]
            # each ask once, as the asks file holds it: movers that share a log ask it alike
            n = len({ask.key for ask, _ in unanswered})
            raise Stop(f"{self.hop}: {n} ask(s) of the logs not in the records (written to {self.args.asks})", "asks")

    def ask_rows(self, rows: dict, why: str) -> None:
        """Keep *rows*, what the rows file does not have said as a fetch can answer it -- a collection of a row, the
        rows a call returned, or an SQL question -- for --rows-asks (the harness's rows ask, stage 1)."""
        self.rows_asks.setdefault(json.dumps(rows, sort_keys=True, default=str), {"rows": rows, "step": self.hop, "why": why})

    def ask_collection(self, iterable: str, step, env: E.Env) -> None:
        """A row's collection a walk's loop runs over and the rows file does not have (``Walk.elements``): the row by
        its class and its key, the columns the dump's tables name the class's identity, and the attribute.  The walk
        spells a row by its name in the rows file (``<job>.Files``, ``classes`` giving JobSpec) or, for a loop's
        element, by its class; an iterable it does not read as a row's attribute is no row to ask of."""
        m = re.fullmatch(r"<(\w+)>\.(\w+)", iterable)
        if not m:
            return
        name, attr = m[1], m[2]
        cls = self.data.get("classes", {}).get(name, name)
        row = env.table.get(name, {})
        key = {n: row[n].value for n in (x.partition(".")[2] for x in _identity_attrs(cls)) if n in row}
        self.ask_rows({"kind": "collection", "row": {"entity": cls, "key": key}, "attr": attr, "entity": None},
                      f"@{step.line} {step.owner.split('::')[-1]} loops over {iterable}, which the rows file does not have")

    def ask_linked(self, st, subjects: list[str], q: Missed) -> list[str]:
        """The rows of a class *st* compares (*subjects*) and the rows file has no row of (``rows-no-row``): the rows of
        that class the statement joins to the asked row, by the columns its joins (the map's ``ReadSite.joins``) compare
        with the row's -- the task's datasets by jediTaskID.  A collection with no attribute, the class named, ``where``
        its columns with the row's values and ``via`` each pair of columns by name (``from`` the row's class, ``to`` the
        asked class, as the rows file's ``linked`` names them); one per pair of sources the joins meet by (two aliases
        of one table are two keys).  A class the statement joins to the row only through another table, a class-less
        table (its join side names no attribute: ``T_TASK``) or a column the row does not hold asks nothing; nor do rows
        the file's ``linked`` holds by one of those pairs for the row's value, which no step compares yet (K36).  The
        joins are the dump's read site's, by the map's name of it (``ReadSiteNode.make_name``): the follow-up's
        statement does not carry them.  What is not asked for a reason the caller would not see -- no read site, rows
        held -- is said (returned, for the caller's lines)."""
        if not subjects:
            return []
        cls = q.subject.partition(".")[0]
        key = {n: v for n in (x.partition(".")[2] for x in _identity_attrs(cls)) if (v := self._row_value(f"{cls}.{n}", q)) is not None}
        where = f"{st.owner.split('::')[-1]}@{st.line}"
        name = f"panda:{st.owner}@{st.line}"
        fragment = mapdump.fragment()
        site = next((rs for rs in (fragment.read_sites if fragment else []) if rs.name == name), None)
        if site is None:
            return [f"the dump has no read site {name}: its joins are not read, and the rows of {', '.join(subjects)} not asked"]
        notes = []
        for lacked in dict.fromkeys(s.partition(".")[0] for s in subjects):
            on: dict[tuple[str, str], list[tuple[str, str]]] = {}
            for j in site.joins:
                for (mine, src), (theirs, their_src) in (((j.a_attr, j.a_source), (j.b_attr, j.b_source)),
                                                         ((j.b_attr, j.b_source), (j.a_attr, j.a_source))):
                    if mine.partition(".")[0] == lacked and theirs.partition(".")[0] == cls:
                        on.setdefault((src, their_src), []).append((theirs, mine))
            compared = ", ".join(s.partition(".")[2] for s in subjects if s.partition(".")[0] == lacked)
            for pairs in on.values():
                by = {to.partition(".")[2]: self._row_value(frm, q) for frm, to in pairs}
                if None in by.values():
                    continue
                via = [{"from": frm, "to": to} for frm, to in pairs]
                held = next((e["via"] for e in self.data.get("linked") or [] if e.get("class") == lacked and e.get("via") in via
                             and self._row_value(e["via"]["from"], q) in (e.get("asked") or [])), None)
                if held:
                    notes.append(f"the rows file's linked holds {lacked}'s rows by {held['from']} -> {held['to']}, which this step "
                                 "does not compare (KNOWN_ISSUES.md K36)")
                    continue
                self.ask_rows({"kind": "collection", "row": {"entity": cls, "key": key}, "attr": None, "entity": lacked, "where": by,
                               "via": via},
                              f"{where} compares {lacked}'s {compared} with the row's, and neither the rows file's rows nor its "
                              f"linked hold {lacked}'s")
        return list(dict.fromkeys(notes))

    def window(self, asking: W.Asking) -> Window | None:
        """*asking*'s window; where it has none, a stop if the door needs one (the door over files reads none)."""
        if not self.logs.needs_window:
            return asking.window
        window = self.need(f"a window to ask the logs over: {asking.why}", asking.window)
        # once per machine: rows of one second on two hosts are two asks (H6)
        if (self.hop, window, asking.machine) not in self.windows_said:
            self.windows_said.add((self.hop, window, asking.machine))
            self.say(f"window: {window[0]} .. {window[1]}, {asking.why}")
        return window

    def stamp(self, when: str) -> None:
        """Which trees this run read, said at its start and again at its end.

        The map's code is imported from a checkout another session may be editing,
        and a tree that moved between the two lines is a run of neither commit.
        """
        head, corpus = tree_stamp(BAMBOO), tree_stamp(CORPUS)
        self.stamps[when] = {"bamboo": head, "corpus": corpus}
        self.say(f"# {when}: bamboo {head}, panda-server {corpus}")
        self.say(f"# map: {mapdump.described()}")
        if corpus.endswith("-dirty"):
            self.say("# the corpus is dirty: this run is void")
        if head.endswith("-dirty"):
            self.say("# the bamboo tree is dirty (another session may be editing it): this run does not measure a commit")

    def result_of(self, outcome: dict, asks: dict) -> dict:
        """result.json, version 1 (agreed with the live-case harness, 2026-10-15): the outcome, the question, the steps
        opened in order, every input with its sha256, the asks file, and what the steps found -- the window, the arm
        that wrote an attribute's value last (``wrote``), the mover that moved the value (late), every mover with its
        verdict and, in S1, judgement 3's premises (``_premises``, added in build order 2a) -- with times in UTC, ISO 8601
        with ``Z``."""
        a = self.args
        dump = mapdump.installed()
        return {
            "version": 1,
            "outcome": outcome,
            "question": {"mode": "late" if a.late else "stuck" if a.stuck else "asked", "subject": a.subject,
                         "observed": a.observed, "key": a.key},
            "steps": self.opened,
            "inputs": {
                "querier": {"tree": str(BAMBOO), "stamps": self.stamps,
                            "label": (BAMBOO / "QUERIER").read_text().strip() if (BAMBOO / "QUERIER").exists() else None,
                            "bamboo_py_sha256": _tree_sha256(BAMBOO),
                            "files": {f.name: _sha256(f) for f in sorted(Path(__file__).resolve().parent.glob("*.py"))}},
                "map_dump": {"path": dump["path"], "sha256": dump["sha256"]} if dump else None,
                "rows": {"path": a.rows, "sha256": _sha256(a.rows)},
                "records": {"given": a.records, "files": [{"path": str(f), "sha256": _sha256(f)} for f in self.logs.files]}
                if a.records else None,
                "logs": [{"path": f, "sha256": _sha256(f)} for f in a.log] if a.log else None,
                "hand": {"path": a.hand, "sha256": _sha256(a.hand)} if a.hand else None,
                "patterns": a.patterns,
                "decide": a.decide,
                "reach": self.result.get("reach"),
            },
            "asks": {"path": a.asks, "count": len(asks)} if asks else None,
            "rows_asks": {"path": a.rows_asks or None, "count": len(self.rows_asks)} if self.rows_asks else None,
            "window": self.result.get("window"),
            "values_read_at": _read_at(self.data),
            "readings_at": self.result.get("readings_at"),
            "possibilities": self.result.get("possibilities"),
            "wrote": self.result.get("wrote"),
            "moved_by": self.result.get("moved_by"),
            "movers": self.result.get("movers"),
        }

    def looked_up_with(self, side: V.Side, step, depth: int = 2, seen: set[str] | None = None) -> list[str]:
        """What the names a stopped step reads were bound to, down to the row fields they read.

        A step that stops at a value another system supplies still says what
        it asked that system with, when the walk kept the binding.
        """
        seen = set() if seen is None else seen
        out = []
        names = sorted((x for x in ast.walk(ast.parse(step.value, mode="eval")) if isinstance(x, ast.Name) and x.id != "self"), key=lambda x: x.col_offset)
        for n in dict.fromkeys(x.id for x in names):
            if n in seen:
                continue
            seen.add(n)
            t = E.taken(side.steps, n, side.env, owner=side.owner)
            if t.step is None:
                continue
            leaves = V.leaves(t.step, side.env)
            out.append(f"{n} <- @{t.step.line} {t.step.value}" + (f"  [{'; '.join(leaves)}]" if leaves else ""))
            if depth > 1:
                out.extend("  " + x for x in self.looked_up_with(side, t.step, depth - 1, seen))
        return out

    # ------------------------------------------------------------ the loop
    def run(self) -> None:
        a = self.args
        self.say(f"# rerun: patterns={self.mode}, decide={self.decide}, hand={a.hand or 'none'}")
        self.say(f"# {' '.join(sys.argv)}")
        self.stamp("start")
        steps = {
            Asked: self.asked, SetBy: self.set_by, Raised: self.raised, Missing: self.missing,
            Branch: self.branch, Settled: self.settled, Pair: self.pair, FalseAtom: self.false_atom,
            Reproduce: self.reproduce, Field: self.field,
            Stuck: self.stuck, Passes: self.passes, Blocked: self.blocked, Input: self.input, Rows: self.rows,
            Late: self.late, Missed: self.missed, Ordered: self.ordered,
        }
        if a.late:
            q = Late(a.subject, a.observed, a.key)
        else:
            q = Stuck(a.subject, a.observed, a.key) if a.stuck else Asked(a.subject, a.observed, a.key)
        self.steps = steps
        while q is not None:
            q = self.fork(q) if isinstance(q, Fork) else steps[type(q)](q)

    def fork(self, q: Fork):
        """Each branch to its end or its stop, each from the state the fork was reached in; a stop of any is the run's."""
        stops, codes = [], set()
        kept = (list(self.touched), dict(self.row_facts), self.question)
        at = self.hop
        for n, (branch, label) in enumerate(zip(q.branches, q.labels, strict=True), 1):
            self.touched, self.row_facts, self.question = list(kept[0]), dict(kept[1]), kept[2]
            self.say()
            self.say(f"==== {n} of {len(q.branches)}: {label}")
            said = len(self.ends)
            try:
                b = branch
                while b is not None:
                    b = self.fork(b) if isinstance(b, Fork) else self.steps[type(b)](b)
                self.say(f"==== {n}: reached the end")
                self.branches.append({"label": label, "kind": "end", "step": self.hop,
                                      "text": "\n".join(line for _, line in self.ends[said:]) or None})
            except Stop as stop:
                self.say(f"==== {n}: STOP {stop}")
                stops.append(f"{n} ({stop})")
                codes.add(stop.code)
                self.branches.append({"label": label, "kind": "stop", "step": self.hop, "text": f"STOP {stop}", "code": stop.code})
        self.hop = at
        if stops:
            raise Stop(f"{len(stops)} of {len(q.branches)} branch(es) stopped: {'; '.join(stops)}", codes.pop() if len(codes) == 1 else None)

    # ------------------------------------------------------------ Asked: who wrote the value
    def asked(self, q: Asked):
        w = self._which_wrote(q.subject, q.observed, q.key)
        question, cands, env, chosen = w.question, w.cands, w.env, w.chosen
        for c in cands:
            if question.log.name in c.log_files:
                self.touched.append((R.function(c.rel, c.func, c.lines), question))
        site_fn = R.function(chosen.candidate.rel, chosen.candidate.func, chosen.candidate.lines)
        at_fn, guard_record = self._guard_at(w, site_fn)
        if w.via is not None:
            self.say(f"{site_fn.name}:{chosen.line} prints nothing: read on its caller's run, at {at_fn.name}:{w.via.line} (KNOWN_ISSUES.md K25)")
        # the run was chosen by its mark before the function was known; a mark that is
        # the key can hold several passes of it (rule 20), and then this run is not one
        if len(starts := R.entries(at_fn, question, self.mode)) > 1:
            self.say(f"WARNING: the run holds {len(starts)} passes of {at_fn.name} (rule 20); the steps below read them as one")
        site = R.statement_at(site_fn.node, chosen.line)
        reading = w.reading
        # an attribute's question is when it took the value, and H2 answered it: the pass of the arm that wrote it last
        # (KNOWN_ISSUES.md K37).  A status's goes on to why its writer's arm ran
        attribute = PT.history(q.subject) is None
        if attribute:
            self.result["wrote"] = self._wrote_said(w)

        # H3
        self.start("H3", "what its guard reads")
        guards = pathcond.path_condition(site)
        self.say(f"guards of {site_fn.name}:{chosen.line}: {guards}")
        unread = []
        if attribute:
            shown = chosen.kind == "positive"
            if not shown:
                # an arm left by elimination has no run that shows it ran: the run the judge read it on is any of the
                # key's at the window, on any machine (a server's DBProxy log for a JEDI writer).  Its guards are read in
                # the rows' values alone, and no run is named
                env = self.base_env(at=w.row["time"] if w.row else None)
            read = {g: _guard_read(g, env) for g in guards}
            if all(got is True for got in read.values()):
                lo, hi = w.window
                at = (f"in the pass of the statuslog row {w.row['status']!r} at {w.row['time']}" if w.row else
                      f"at {hi}" if lo == hi else f"in {lo} .. {hi}")
                via = f", called at {at_fn.name}:{w.via.line}" if w.via is not None else ""
                how = (("its guards hold in the run's values" if shown else "its guards hold in the rows' values") if guards
                       else "it has no guard")
                said = f"{q.subject}={q.observed!r} written last by {site_fn.name}:{chosen.line}{via}"
                if shown:
                    self.end(f"{said} (shown in its run; {how}), {at}: {question.describe()}")
                    return None
                holes = "; ".join(f"{log} on {', '.join(dict.fromkeys(h.partition(': ')[0] for h in hs))}"
                                  for log, hs in w.unsearched.items())
                self.end(f"{said} (by elimination: no run shows the line ran; {how}), {at}"
                         + (f"; not searched there: {holes}" if holes else ""))
                return None
            unread = [f"{g!r} {'read false' if got is False else 'not read'}" for g, got in read.items() if got is not True]
            # an arm left by elimination is read in no run: H4 would read its guard's method in a run not its own
            self.need(f"a run that shows {site_fn.name}:{chosen.line} ran, to read its guards in (by elimination; its "
                      f"conditions: {', '.join(unread)})", shown)
        methods = []
        for g in guards:
            for atom in E.atoms(ast.parse(E.strip_annotation(g), mode="eval").body):
                for n in ast.walk(atom):
                    if (
                        isinstance(n, ast.Call) and not n.args and not n.keywords
                        and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, (ast.Attribute, ast.Name))
                        and E.ev(n, env) is E.UNKNOWN
                    ):
                        methods.append((n.func.attr, ast.unparse(n)))
        self.say(f"read through a method: {methods}")
        method = self.need("a method the guard reads through" + (f" (its conditions: {', '.join(unread)})" if unread else ""),
                           methods[0][0] if methods else None)
        return SetBy(method, reading, question, guard_record, env)

    def _wrote_said(self, w: Wrote) -> dict:
        """result.json's ``wrote`` for an attribute: H2's answer -- the arm, whether its run shows the line ran (else it was
        left by elimination), the call a writer that prints nothing is read at, the statuslog row whose pass wrote it, the
        window asked and the run (as ``moved_by``'s; null for an arm left by elimination, which no run shows)."""
        run = w.question
        shown = w.chosen.kind == "positive"
        return {
            "owner": w.reading.owner, "line": w.chosen.line, "shown": shown,
            "via": {"owner": w.via.candidate.owner, "line": w.via.line} if w.via is not None else None,
            "row": {"status": w.row["status"], "time": _iso(w.row.get("time_extended") or w.row["time"])} if w.row else None,
            "window": {"start": _iso(w.window[0]), "end": _iso(w.window[1])} if w.window else None,
            "run": {"log": run.log.name, "machine": run.machine or None, "token": run.token, "first": run.first, "last": run.last}
            if shown else None,
        }

    def _guard_at(self, w: Wrote, site_fn) -> tuple:
        """``(function, record)``: whose passes the run holds and the first record past the rows that must print after
        the chosen line -- what the steps after H3 read the run up to.  The chosen arm's function; for a writer that
        prints nothing, read on its caller's run (``Wrote.via``), the caller at its call."""
        at = w.via or w.chosen
        at_fn = site_fn if w.via is None else R.function(at.candidate.rel, at.candidate.func, at.candidate.lines)
        _, after = R.must_rows(at_fn.node, R.statement_at(at_fn.node, at.line))
        sn = R.seen(at_fn, w.question, self.mode)
        return at_fn, min((i for s in after for i, _, _ in sn.hits.get(s.lineno, ())), default=len(w.question.records))

    def _which_wrote(self, subject: str, value: str, key: str, state: dict | None = None) -> Wrote:
        """H1 and H2: the map's writers of ``subject=value``, and which of them the run of the key at the statuslog
        row *state* shows ran.

        *state* is the statuslog row of *value* unless given: ``--late`` asks it at the row after the observed one.
        Every candidate's logs are asked, at the row's time, on the machine that wrote the row.  A *subject* the
        statuslog's rows hold beside the status (``JobSpec.computingSite``) is asked over the window they bound, on
        every machine (``W.value_held``).
        """
        # H1
        self.start("H1", "candidates")
        readings = self._writers(subject, value, key)
        self.say(f"{len(readings)} reading(s) from the map for {subject}={value}")
        attr = subject.split(".")[-1]

        # H2
        self.start("H2", "which ran")
        holding = PT.history_holding(subject) if state is None else None
        if holding is not None:
            # a value the history's rows hold beside the status: they say by when it was there, not which run wrote it
            asking = W.value_held(self.data["statuslog"]["states"], attr, value, self._created(holding))
            found = self.need(f"a statuslog time for {value}: {asking.why}", asking.window)
            if isinstance(found, str):  # a time from --hand
                asking = W.Asking((found, found), None, f"the time handed for {value}, on every machine")
            tries = [(state, asking, None)]
        elif state is None and PT.history(subject) is None:
            # an attribute no history records or holds: its writers' arms say when it took the value (K37)
            tries = self._attribute_tries(subject, attr, value, readings)
        else:
            if state is None:
                states = self.data["statuslog"]["states"]
                state = self.need(f"a statuslog time for {value}", next((s for s in states if s["status"] == value), None))
                if isinstance(state, str):  # a time from --hand
                    state = {"status": value, "time": state}
            tries = [(state, W.at_transition(state), None)]
        self._stand_ins: dict[tuple[str, int], list[StandIn]] = {}
        names = self._writer_logs(subject, readings, self._h2_candidates(subject, readings) if readings else [])
        if len(tries) > 1:
            # every row's asks in one round, for the recording step to answer together, each of the logs of the
            # writers that can write it; the rows are then read newest first, and the first whose pass a writer is
            # shown to write in ends the scan
            self._stop_unanswered([u for _, asking, own in tries for u in self._ask(
                key, names if own is None else self._writer_logs(subject, own, self._h2_candidates(subject, own)), asking)[1]])
            return self._scan(subject, attr, value, key, readings, tries)
        state, asking, own = tries[0]
        return self._stood(subject, value, self._wrote_at(subject, attr, value, key, readings if own is None else own, state, asking, True), state)

    def _stood(self, subject: str, value: str, wrote: Wrote, state: dict | None) -> Wrote:
        """A single window's pick, where it stands in for writers that print nothing (:class:`StandIn`): the arm shown
        to write, read on the pick's run (:meth:`_answer`), or a stop naming the arms left (K38)."""
        if (wrote.chosen.candidate.owner, wrote.chosen.line) not in self._stand_ins:
            return wrote
        left = self._possibles(wrote.chosen, state, wrote.env, wrote)
        if (answer := self._answer(left)) is not None:
            return answer
        self._one_of(subject, value, left, 1)
        raise AssertionError("_one_of stops")

    def _h2_candidates(self, subject: str, readings: list) -> list[Candidate]:
        """H2's candidates, one per function: each reading's, where its runs are; and for a writer that prints nothing,
        whose arm a history pair places (``attr_time.arms``), the pair's caller at the call on the way to it, read on
        the caller's run (:class:`StandIn`, kept in ``_stand_ins`` by the caller's owner and line).  Only where the map
        places the prints (``rows.own_log_files``): else the readings' logs stand, as before."""
        silent = [r for r in readings if prints_nothing(r.owner, r.lines)] if R.OWN_FILES and R.ROWS_FROM == "graph" else []
        arms = {(a.owner, a.line): a for a in AT.arms(mapdump.fragment(), subject, silent)} if silent else {}
        found: dict[str, Candidate] = {}

        def add(rel: str, func: str, line: int, files: list[str]) -> None:
            c = found.setdefault(f"{rel}::{func}", Candidate(rel, func, [], []))
            c.lines += [line] if line not in c.lines else []
            c.log_files += [f for f in files if f not in c.log_files]

        for r in readings:
            func = r.owner.split("::")[-1]
            own = self.files_of(r.owner, r.log_files, r.lines)
            for line in r.lines:
                a = arms.get((r.owner, line))
                if a is None or not a.placed:
                    add(r.file, func, line, own)
                    continue
                node = next((f for f in R.functions(r.file, func) if f.lineno <= line <= f.end_lineno), None)
                stmt = R.statement_at(node, line) if node is not None else None
                guards = tuple(pathcond.path_condition(stmt)) if stmt is not None else ()
                for caller, point, level in sorted({(p.caller, p.point, p.level) for p in a.placed}):
                    rel, _, qual = caller.partition("::")
                    name = qual.rpartition(".")[2]
                    add(rel, name, point, self.files_of(f"{rel}::{name}", list(r.log_files), [point]))
                    stand = self._stand_ins.setdefault((f"{rel}::{name}", point), [])
                    if (s := StandIn(f"{r.owner}:{line}", level, guards)) not in stand:
                        stand.append(s)
        return list(found.values())

    def _possibles(self, v, state: dict | None, env: E.Env, answer: Wrote | None = None) -> list[Possible]:
        """The arms verdict *v* says wrote, or may have, in the pass of the row *state*: its own line, or the arms of the
        writers that print nothing it stands in for, less those a guard read false shows did not.  One shown to write
        carries *answer*; a stand-in's with the writer's arm chosen, read on the caller's run (``Wrote.via``)."""
        kept = [] if v.kind == "positive" else [f"its run does not show the line ran ({v.reasons[-1] if v.reasons else v.kind})"]
        stands = self._stand_ins.get((v.candidate.owner, v.line))
        if not stands:
            return [Possible(f"{Path(v.candidate.rel).name}::{v.candidate.func}:{v.line}", state, not kept, kept, answer if not kept else None)]
        out = []
        for s in stands:
            left = s.left(env)
            if left is None:
                continue
            premises = kept + left
            owner, _, line = s.arm.rpartition(":")
            rel, _, func = owner.partition("::")
            arm = Verdict(Candidate(rel, func, [int(line)]), int(line), v.kind, list(v.reasons))
            stood = replace(answer, chosen=arm, via=v) if answer is not None and not premises else None
            out.append(Possible(f"{Path(rel).name}::{func}:{line}", state, not premises, premises, stood,
                                f"{Path(v.candidate.rel).name}::{v.candidate.func}:{v.line}"))
        return out

    def _answer(self, left: list[Possible]) -> Wrote | None:
        """H2's answer from the arms a scan left: one arm, shown to write at a row no possibility is newer than.  A writer
        that prints nothing can be called twice in one pass, once shown and once not (setSuperStatus_JEDI from
        updateTask_JEDI and from getTasksToBeFinished_JEDI): the arm and the row are one, so the answer stands, and the
        other calls are said -- which of them was last is not read."""
        if len({p.arm for p in left}) != 1:
            return None
        wrote = next((p for p in left if p.wrote and p.answer is not None), None)
        if wrote is None or any(p.row is not wrote.row for p in left):
            return None
        if others := [p.by or p.arm for p in left if p is not wrote]:
            self.say(f"  {wrote.arm} wrote" + (f", read at {wrote.by}" if wrote.by else "") +
                     f"; in the same pass also {', '.join(others)}, the same arm: which call was last is not read")
        return wrote.answer

    def _scan(self, subject: str, attr: str, value: str, key: str, readings: list, tries: list) -> Wrote:
        """H2 over the statuslog rows the writers' passes can write, newest first (attr-time-shape.md 5): a row whose pass
        a writer is shown to write in ends the scan; one shown to have no writer's pass goes; one that may have stays.
        The answer is a writer that prints shown to write, with nothing newer that may have; else the arms left are
        named (:meth:`_one_of`).  A row is read on the writers that can write it, where the try names them."""
        left: list[Possible] = []
        named = self._named(subject, key, readings, tries)
        for n, (state, asking, own) in enumerate(tries, 1):
            self.say(f"-- row {n} of {len(tries)}: {asking.why}")
            found = self._row_at(subject, attr, value, key, readings if own is None else own, state, asking, named)
            left += found
            if any(p.wrote for p in found):
                break
        if (answer := self._answer(left)) is not None:
            self.question = answer.question
            return answer
        self._one_of(subject, value, left, len(tries))
        raise AssertionError("_one_of stops")

    def _named(self, subject: str, key: str, readings: list, tries: list) -> set[str]:
        """The writers' logs some run of *key* is in, in the records of any row of *tries*: a log shown to name its runs
        by the key, where no run at a row is no pass of its writers (:meth:`_row_at`).  JobGenerator's pass marks its
        lines <jediTaskID=... datasetID=...> and prints no PandaID: its log holds no run of a job it made (the harness's
        to-querier-62)."""
        out: set[str] = set()
        for _, asking, own in tries:
            cands = self._h2_candidates(subject, readings if own is None else own)
            for name in self.logs.names(list(dict.fromkeys(f for c in cands for f in c.log_files))):
                try:
                    log = self.logs.log(name, key, self.window(asking), machine=asking.machine)
                except Unrecorded:
                    continue
                if runs_for(log, key):
                    out.add(name)
        return out

    def _unshown(self, cands: list[Candidate], key: str, state: dict, env: E.Env) -> list[Possible]:
        """The arms of *cands*, writers none of whose logs is shown to name its runs by *key* (:meth:`_named`): no run
        of the key at the row is not their absence, so each may have written -- a stand-in's arm less one its condition
        read false rules out."""
        out = []
        for c in cands:
            why = f"no line of {', '.join(c.log_files)} in the records holds {key}: its runs are not shown to name it"
            for line in c.lines:
                stands = self._stand_ins.get((c.owner, line))
                if not stands:
                    out.append(Possible(f"{Path(c.rel).name}::{c.func}:{line}", state, False, [why]))
                for s in stands or []:
                    if (left := s.left(env)) is not None:
                        out.append(Possible(f"{Path(s.arm.split('::')[0]).name}::{s.arm.split('::')[-1]}", state, False, [why, *left]))
        return out

    def _row_at(self, subject: str, attr: str, value: str, key: str, readings: list, state: dict, asking: W.Asking,
                named: set[str]) -> list[Possible]:
        """One row of :meth:`_scan`: the arms whose pass at the row wrote, or may have.  No run of the key there, in logs
        searched whole and shown to name runs by it (*named*), is no writer's pass; a log not searched whole leaves its
        writers open, and so does one no record shows naming the key (:meth:`_unshown`)."""
        readings, said = self._unborn(readings, subject, state)
        if said:
            self.say(said)
        lo, hi = asking.window
        when = hi if lo == hi else f"{lo}..{hi}"
        cands = self._h2_candidates(subject, readings)
        runs, _, holes = self._ask(key, list(dict.fromkeys(f for c in cands for f in c.log_files)), asking)
        at = {name: held for name, rs in runs.items() if (held := [r for r in rs if r.first[:19] <= hi and r.last[:19] >= lo])}
        env = self.base_env(at=state["time"])
        maybe = self._unshown([c for c in cands if not any(f in named or f in holes for f in c.log_files)], key, state, env)
        if not at and not holes:
            self.say(f"  no run of {key} at {when}: no writer's pass" + (" of a log that names it" if maybe else ""))
            for p in maybe:
                self.say(f"  may have written: {p.arm} ({'; '.join(p.premises)})")
            return maybe
        question = _question_run(at)
        if question is not None:
            self.say(f"the run: {question.describe()}")
        at_hand = [question] if question is not None else [r for rs in at.values() for r in rs]
        for c in cands:
            if any(name in c.log_files for name in at):
                self.warn_split(c.owner, c.lines)
        stable_holes([R.seen(R.function(c.rel, c.func, c.lines), run, self.mode)
                      for c in cands for run in at_hand if run.log.name in c.log_files], env)
        j = judge(
            cands, runs, mode=self.mode, decide=self.decide, window=(lo, hi), subject=attr, observed=value,
            question_run=question, env=env, called_at=self._called_at,
        )
        self.say(j.report(question))
        out = []
        for v in j.verdicts:
            if v.kind == "excluded" or (v.kind == "unexamined" and not any(f in holes for f in v.candidate.log_files)):
                continue
            run = question if question is not None and question.log.name in v.candidate.log_files else j.runs.get(v.candidate.owner)
            answer = Wrote(readings, run, cands, env, v, row=state, window=(lo, hi),
                           unsearched={f: holes[f] for f in v.candidate.log_files if f in holes}) if run is not None else None
            if v.kind == "unexamined":
                v.reasons.append(f"not searched whole: {', '.join(f for f in v.candidate.log_files if f in holes)}")
            out += self._possibles(v, state, env, answer)
        out += maybe
        for p in out:
            self.say(f"  {'wrote' if p.wrote else 'may have written'}: {p.arm}" + (f" ({'; '.join(p.premises)})" if p.premises else ""))
        return out

    def _one_of(self, subject: str, value: str, left: list[Possible], rows: int) -> None:
        """Stop H2 naming the arms that wrote *value* last or may have, each with its rows (KNOWN_ISSUES.md K38): the
        steps after H2 read one arm's run, and are not run for each (attr-time-shape.md 5, the second stage)."""
        arms: dict[str, list[Possible]] = {}
        for p in left:
            arms.setdefault(p.arm, []).append(p)
        self.result["possibilities"] = [
            {"arm": arm, "rows": [{"status": p.row["status"] if p.row else None, "time": p.row["time"] if p.row else None,
                                   "wrote": p.wrote, "premises": p.premises} for p in ps]}
            for arm, ps in arms.items()
        ]
        if not arms:
            raise Stop(f"{self.hop}: a pass that wrote {subject}={value!r} at the {rows} statuslog row(s) its writers' passes "
                       f"can write (none shows a writer's pass: KNOWN_ISSUES.md K38)", "one-of")
        said = []
        for arm, ps in arms.items():
            at = ", ".join(f"{'wrote' if p.wrote else 'may have'} at {p.row['status']!r} {p.row['time']}" if p.row else
                           ("wrote" if p.wrote else "may have") for p in ps)
            said.append(f"{arm} ({at})")
        raise Stop(f"{self.hop}: one arm that wrote {subject}={value!r} last, of {len(arms)} left: {'; '.join(said)} "
                   f"(KNOWN_ISSUES.md K38)", "one-of")

    def _writer_logs(self, subject: str, readings: list, cands: list | None = None) -> list[str]:
        """The logs H2 looks for a run of *subject*'s writers in (*cands*, ``_candidates`` of *readings* where not
        given).  A writer that prints nothing of its own, whose reading names no log and whose arm no history pair
        places, has no run a log holds (KNOWN_ISSUES.md K25): where no writer has a log, H2 stops on that -- not on the
        logs holding no run of the key, which none was asked for; and with no writer at all, on that.  A time handed
        for H2 gives no log."""
        if not readings:
            raise Stop(f"{self.hop}: a writer of {subject} in the map")
        cands = self._candidates(readings) if cands is None else cands
        names = list(dict.fromkeys(f for c in cands for f in c.log_files))
        if not names:
            funcs = ", ".join(dict.fromkeys(r.owner.split("::")[-1] for r in readings))
            raise Stop(f"{self.hop}: a log to look for a run of a writer of {subject} in ({funcs}: the map names none; "
                       f"KNOWN_ISSUES.md K25)")
        return names

    def _attribute_tries(self, subject: str, attr: str, value: str, readings: list) -> list[tuple[dict | None, W.Asking, list | None]]:
        """H2's windows for an attribute no history records (``attr_time``, KNOWN_ISSUES.md K37): the value itself where
        every writer writes the time it runs, else the statuslog rows the writers' passes can write, newest first, each
        with the readings of the writers that can write it (``attr_time.row_writers``), whose logs alone it is asked of."""
        fragment = mapdump.fragment()
        found = AT.arms(fragment, subject, readings)
        self.need(f"a writer of {subject} in the map, to say when it took {value!r}", found)
        if AT.now_valued(found):
            asking = W.value_now(attr, value)
            got = self.need(f"a time for {value}: {asking.why}", asking.window)
            if isinstance(got, str):  # a time from --hand
                asking = W.Asking((got, got), None, f"the time handed for {value}, on every machine")
            self.say(f"{len(found)} writer line(s), each writing the time it runs: {asking.why}")
            return [(None, asking, None)]
        for a in found:
            self.say(f"  {a.owner.split('::')[-1]}:{a.line} " + (f"is on no history row of the object: {a.unplaced}" if a.unplaced else
                     "writes in a pass that writes the object's history row (" + ", ".join(sorted({f"{p.caller.split('::')[-1]}:{p.call} "
                     f"{p.callee.split('.')[-1]} {p.order}" for p in a.placed})) + ")"))
        unplaced = [a for a in found if a.unplaced]
        self.need(f"every writer of {subject} on a history row of the object, to say when it took {value!r} "
                  f"({len(unplaced)} of {len(found)} not: KNOWN_ISSUES.md K37)", not unplaced)
        states = self.data["statuslog"]["states"]
        rows = AT.row_writers(found, states, AT.recorded_values(fragment))
        self.say(f"the writers' passes can write {len(rows)} of the {len(states)} statuslog row(s): asked newest first, "
                 "each of the logs of the writers that can write it")
        if len(found) > 1:
            for a in found:
                self.say(f"  {a.owner.split('::')[-1]}:{a.line}: {sum(a in arms for _, arms in rows)} row(s)")
        got = self.need(f"a statuslog row the writers of {subject} can write", [s for s, _ in rows])
        if isinstance(got, str):  # a time from --hand
            return [({"status": value, "time": got}, W.Asking((got, got), None, f"the time handed for {value}, on every machine"), None)]
        return [(s, W.at_transition(s), AT.readings_of(readings, arms)) for s, arms in rows]

    def _wrote_at(self, subject: str, attr: str, value: str, key: str, readings: list, state: dict | None,
                  asking: W.Asking, last: bool) -> Wrote | None:
        """H2 at one window: the run of the key there and which candidate ran.  None, where *last* is not, for a
        window with no run or no candidate that ran: the next row is tried."""
        readings, said = self._unborn(readings, subject, state)
        if said:
            self.say(said)
        lo, hi = asking.window
        when = hi if lo == hi else f"{lo}..{hi}"
        cands = self._h2_candidates(subject, readings)
        runs, unanswered, holes = self._ask(key, list(dict.fromkeys(f for c in cands for f in c.log_files)), asking)
        self._stop_unanswered(unanswered)
        at = {name: held for name, rs in runs.items() if (held := [r for r in rs if r.first[:19] <= hi and r.last[:19] >= lo])}
        if not at and not last:
            self.say(f"  no run of {key} at {when}")
            return None
        self.need(f"a run of {key} at {when}", at)
        question = _question_run(at)
        if question is not None:
            self.say(f"the run: {question.describe()}")
        elif len(at) > 1:
            self.say(f"runs of {key} at {when} in {len(at)} logs ({', '.join(at)}): each candidate is judged on its own run")
        else:
            self.say(f"{len(next(iter(at.values())))} runs of {key} at {when} in {next(iter(at))}: each candidate is judged on its own run")
        at_hand = [question] if question is not None else [r for rs in at.values() for r in rs]
        for c in cands:
            if any(name in c.log_files for name in at):
                self.warn_split(c.owner, c.lines)
        env = self.base_env(at=state["time"] if state else None)
        seens = [R.seen(R.function(c.rel, c.func, c.lines), run, self.mode)
                 for c in cands for run in at_hand if run.log.name in c.log_files]
        stable_holes(seens, env)
        j = judge(
            cands, runs, mode=self.mode, decide=self.decide, window=(lo, hi),
            subject=attr, observed=value, question_run=question, env=env, called_at=self._called_at,
        )
        self.say(j.report(question))
        if j.chosen is None and not last and self.hop not in self.hand:
            self.say(f"  no writer of {subject} ran at {when}")
            return None
        chosen = self.pick("which candidate ran", j)
        if question is None:
            question = self.need(f"the run of {chosen.candidate.func} at {when}", j.runs.get(chosen.candidate.owner))
            # a candidate left by elimination is shown by no run: this is only the run the judge read it on
            self.say(f"the run: {question.describe()}" if chosen.kind == "positive" else
                     f"the run it was read on (no run shows {chosen.candidate.func}:{chosen.line} ran): {question.describe()}")
        self.question = question
        return Wrote(readings, question, cands, env, chosen, row=state, window=(lo, hi),
                     unsearched={f: holes[f] for f in chosen.candidate.log_files if f in holes})

    def _writers(self, subject: str, value: str, key: str) -> list:
        """H1's reading: the map's writers of ``subject=value`` (derive-strategy's readings), read once per value."""
        if (subject, value) not in self._readings:
            symptom = Symptom(subject=subject, observed=value, task_id=key)
            inv, _ = asyncio.run(_investigate(mapdump.code_map("rerun"), symptom, None, key, None, 1, None))
            R.check_blobs(inv, "derive-strategy's readings")
            self._readings[(subject, value)] = inv.hops[0].strategy.readings
        return self._readings[(subject, value)]

    def _unborn(self, readings: list, subject: str, state: dict | None) -> tuple[list, str]:
        """``(readings, said)``: *readings* less their lines that make the row (births, K29) where the statuslog holds
        a row of the key before *state*.  The row was there, and a write that makes a row writes none that was:
        ppEventServiceJob's new jobs, INSERTed activated with a sequence's PandaID, did not write a job's 'activated'
        after its 'defined'.  A reading left with no line goes; *said* names what went.  With no row before *state*
        (the row's first, a tie, no state) every reading stays: the birth may be the row's."""
        states = (self.data.get("statuslog") or {}).get("states") or []
        marked = {id(r) for r in readings if any(m and all(is_birth(kf) for kf in m)
                                                 for m in map(birth_marks(r.owner, subject).get, r.lines))}
        if not marked or state is None or not HD.row_before(states, state):
            return readings, ""
        writers = asyncio.run(self._writers_of(subject))
        kept, left = [], []
        for r in readings:
            made = born_lines(writers, r.owner, subject) & set(r.lines) if id(r) in marked else set()
            if not made:
                kept.append(r)
                continue
            left.append(f"{r.owner.split('::')[-1]} {sorted(made)} (births: {birth_why(r.owner, subject, sorted(made))})")
            if rest := [line for line in r.lines if line not in made]:
                kept.append(r.model_copy(update={"lines": rest}))
        said = (f"the key's row was there before {state['status']!r} at {state['time']}, so its writes that make a row "
                f"are no candidates: {'; '.join(left)} (KNOWN_ISSUES.md K29)") if left else ""
        return kept, said

    def _candidates(self, readings: list) -> list[Candidate]:
        return [Candidate(r.file, r.owner.split("::")[-1], r.lines, self.files_of(r.owner, r.log_files, r.lines)) for r in readings]

    # ------------------------------------------------------------ F1: who wrote the rows between
    def rows_between(self, at: str | None) -> None:
        """F1: which run wrote each statuslog row between a run that began *at* and ``as_of`` -- the first of the facts
        K22 waits for (rows-at-time-querier.md 1.3) -- asked of the logs as H2 asks for its own row.

        For each row: the map's writers of its status (H1), their logs at the row's time on the row's host
        (``W.at_transition``), by the key.  Every row's asks are gathered before one stop, so that the recording step
        fetches them in one round.  With the answers each row's candidates are judged as H2 judges them, at the row's
        time; a row no candidate is chosen for says so, and the next is read -- nothing here stops the step.  The step
        that read the value stops after, as before: which writes between can change it and the writes that leave no
        row are not read yet.  A run that prints the key in none of its records is not found (KNOWN_ISSUES.md K22).
        """
        if at is None or at in self._rows_read:
            return
        self._rows_read.add(at)
        cls = self.args.subject.partition(".")[0]
        history = PT.history_of_class(cls)
        log = self.data.get("statuslog") or {}
        if history is None or not log.get("states"):
            return
        hop = self.hop
        self.start("F1", "who wrote the rows between")
        states = HD.rows_after(log["states"], at, log.get("as_of"))
        self.say(f"{len(states)} statuslog row(s) after the run's {at}" + (f", to as_of {log['as_of']}" if log.get("as_of") else ""))
        key, attr = self.args.key, history.subject.split(".")[-1]
        asked, unanswered = [], []
        for state in states:
            readings, said = self._unborn(self._writers(history.subject, state["status"], key), history.subject, state)
            cands = self._candidates(readings)
            asking = W.at_transition(state)
            names = list(dict.fromkeys(f for c in cands for f in c.log_files))
            runs, missed, holes = self._ask(key, names, asking)
            unanswered += missed
            asked.append((state, readings, cands, asking, names, runs, holes, said))
        self._stop_unanswered(unanswered)
        chosen = 0
        for state, readings, cands, asking, names, runs, holes, said in asked:
            lo, hi = asking.window
            self.say(f"-- {state['status']!r} at {hi}: {len(readings)} reading(s) from the map, logs {names}")
            if said:
                self.say(f"   {said}")
            at_row = {name: held for name, rs in runs.items() if (held := [r for r in rs if r.first[:19] <= hi and r.last[:19] >= lo])}
            if not at_row:
                unsearched = "".join(f"; {name} not searched: {'; '.join(h)}" for name, h in holes.items())
                self.say(f"   no run of {key} at {hi} in {', '.join(runs) or 'no log asked'}{unsearched}: no writer chosen")
                continue
            # as H2: runs in more than one log are each judged on their own; in one log, the first is the row's
            question = None if len(at_row) > 1 else next(iter(at_row.values()))[0]
            at_hand = [question] if question is not None else [r for rs in at_row.values() for r in rs]
            env = self.base_env(at=state["time"])
            stable_holes([R.seen(R.function(c.rel, c.func, c.lines), run, self.mode)
                          for c in cands for run in at_hand if run.log.name in c.log_files], env)
            j = judge(
                cands, runs, mode=self.mode, decide=self.decide, window=(lo, hi),
                subject=attr, observed=state["status"], question_run=question, env=env, called_at=self._called_at,
            )
            self.say(j.report(question))
            chosen += j.chosen is not None
        self.say(f"F1: a writer chosen for {chosen} of {len(states)} row(s); which of them can change the value, and the "
                 f"writes that leave no row, are not read yet (KNOWN_ISSUES.md K22)")
        self.hop = hop

    # ------------------------------------------------------------ SetBy: which call set it
    def set_by(self, q: SetBy):
        self.start("H4", "which call set it")
        question, env = q.question, q.env
        c = C.census(q.method)
        if c.stop:
            self.need(c.stop, None)
        C.settle_dispatch(c, q.reading.dispatch, question)
        C.match_value(c, question, self.mode, q.guard_record)
        self.say(c.report())
        by_fn: dict[tuple[str, str], list[int]] = {}
        for s in c.kept():
            by_fn.setdefault((s.rel, s.func), []).append(s.line)
        sites = [Candidate(rel, func, lines, run=question) for (rel, func), lines in by_fn.items()]
        j = judge(sites, {question.log.name: [question]}, mode=self.mode, decide=self.decide, question_run=question, env=env)
        self.say(j.report(question))
        chosen = self.pick("which call set it", j)
        for s in sites:
            self.touched.append((R.function(s.rel, s.func, s.lines), question))
        self.learn([(f, q2) for f, q2 in self.touched if q2 is question])
        self.say(f"row facts kept for later hops: { {k: v[0] for k, v in self.row_facts.items()} }")
        # the one way on, whether or not the call failed by an exception (KNOWN_ISSUES.md K11)
        return Raised(R.function(chosen.candidate.rel, chosen.candidate.func, within=chosen.candidate.lines), question)

    # ------------------------------------------------------------ Raised: what it was handed
    def raised(self, q: Raised):
        self.start("H5", "what it was handed")
        fn, question = q.fn, q.question
        chains = [
            (r, ch) for r in question.records for ch in r.chains()
            if any(f.rel == fn.rel and f.func == fn.name for f in ch.frames)
        ]
        record = self.need(f"a traceback through {fn.name} in the run", chains[0][0] if chains else None)
        if record.cut:
            self.need(f"the whole traceback of {record.span} (the search cut it after 200 lines)", None)
        # a traceback prints the lines of the code the run ran: a frame read at another line of the pin says so
        for f in dict.fromkeys(f"{f.rel}:{f.moved_from} {f.func} is line {f.line} of the pin by its source text"
                               for ch in record.chains() for f in ch.frames if f.moved_from is not None):
            self.say(f"frame {f} (the run's code was another version; KNOWN_ISSUES.md K33)")
        boundary = None
        for ch in record.chains():
            for inner, outer in zip(ch.frames, ch.frames[1:], strict=False):
                if inner.in_corpus and not outer.in_corpus:
                    boundary = (inner, outer, ch)
        if boundary is None:
            # the innermost frame is the corpus's own: the line raised, and what it
            # needed not to raise is a guard that came out false
            inside = [(ch, ch.frames[-1]) for ch in record.chains() if ch.frames and ch.frames[-1].in_corpus]
            ch, last = self.need("a frame at the corpus boundary, or an innermost frame in the corpus", inside[-1] if inside else None)
            self.say(f"raised inside the corpus: {last.rel}:{last.line} {last.func} ({last.source})")
            self.say(f"exception: {ch.exc_type}: {ch.exc_message.splitlines()[-1] if ch.exc_message else ''}")
            if not last.same_source:
                if last.source:
                    held = f"at lines {', '.join(map(str, last.printed_at))}" if last.printed_at else "at no line"
                    self.say(f"the pin's line {last.line} holds other text; its {last.func} holds the printed source"
                             f" {held} (the run's code was another version; KNOWN_ISSUES.md K33)")
                self.need("the frame's source line matches the corpus", None)
            return FalseAtom(last, ch, question)
        inner, outer, ch = boundary
        self.say(f"boundary: {inner.rel}:{inner.line} {inner.func} -> {outer.func} ({outer.source})")
        self.say(f"exception: {ch.exc_type}: {ch.exc_message.splitlines()[-1]}")
        earlier = "\n".join(r.text for r in question.records[: question.index(record)])
        names = []
        for quoted in re.findall(r"'([^']+)'", ch.exc_message):
            for cand in (quoted, quoted.split(":", 1)[-1]):
                if cand in earlier and cand not in names:
                    names.append(cand)
        obj = self.need("the object the exception names, seen earlier in the run", names[-1] if names else None)
        missing = bool(re.search(r"NotFound|not found|does not exist", ch.exc_type + " " + ch.exc_message))
        self.say(f"object: {obj}  ({'the outside says it is missing' if missing else 'no word of it missing'})")
        self.need("the outside says the object is missing", missing)
        return Missing(obj, inner, question)

    # ------------------------------------------------------------ Missing: who makes it
    def missing(self, q: Missing):
        # H6: timed by the job's own rows, each on its host, in turn (W.rows_until; KNOWN_ISSUES.md K6)
        self.start("H6", "who makes it")
        obj, inner, question = q.obj, q.inner, q.question
        states = self.data["statuslog"]["states"]
        receivers = O.instances_in(inner.rel)
        receiver = self.need(f"the client {inner.rel} builds", receivers[0] if receivers else None)
        calls = [x for x in O.calls_through(receiver) if x.verb == "make"]
        self.say(f"{receiver}: {len(calls)} make call(s)")
        logs = list(dict.fromkeys(f for x in calls for f in self.files_of(f"{x.rel}::{x.func}", [], [x.line])))
        rows = W.rows_until(states, question.last)
        self.say(f"{len(rows)} statuslog row(s) by the run's last record {question.last}: each asked at its second, in "
                 f"turn, until a run holds it")
        job = keyed(self.args.key)
        makers, unanswered = [], []
        for state, asking in rows:
            runs, missed, _ = self._ask(obj, logs, asking)
            unanswered += missed
            if missed:
                continue
            lo, hi = asking.window
            held = [r for rs in runs.values() for r in rs if r.first[:19] <= hi and r.last[:19] >= lo]
            found = []
            for x in calls:
                f2 = R.function(x.rel, x.func, [x.line])
                for r2 in held:
                    sn2 = R.seen(f2, r2, self.mode)
                    hits = [(row.line, r2.records[i].span) for row in f2.rows for i, _, _ in sn2.hits.get(row.line, ()) if obj in r2.records[i].text]
                    if not hits:
                        continue
                    # a name many runs share is made by each: the job's own run is one that names it
                    if not any(job.search(r.message) or (r.token and job.search(r.token)) for r in r2.records):
                        self.say(f"   {Path(x.rel).name}:{x.line} printed the name in a run that does not name {self.args.key}: "
                                 f"another job's, {r2.describe()}")
                        continue
                    found.append((x, f2, r2, hits))
            # the door over files reads the files given whatever machine wrote them: the row's host is not where it looked
            where = f"on {asking.machine or 'every machine'}" if self.logs.needs_window else "in the files given, their machine not known"
            self.say(f"-- {state['status']!r} at {hi} {where}: {len(held)} run(s) of the name hold that second, {len(found)} "
                     f"with a make call's print and the job's key")
            if found:
                makers = found
                break
        self._stop_unanswered(unanswered)
        for x, _f2, r2, hits in makers:
            self.say(f"  {Path(x.rel).name}:{x.line} {x.func} {x.method}({x.first_arg})  printed the name at {hits}  run {r2.describe()}")
        self.need("a make call whose function printed the object's name, in a run holding one of the job's rows and naming the job", makers)
        if len({(x.rel, x.func) for x, *_ in makers}) != 1:
            self.need("one function among the makers", None)
        x, f2, r2, _ = makers[0]
        make_calls = [m for m, *_ in makers if m.rel == x.rel and m.func == x.func]
        self.say(f"make calls in {x.func}: {[(m.line, m.method) for m in make_calls]}")
        timeline = ", ".join(f"{st['status']} {st['time']}" for st in states)
        self.say(f"the run {r2.first}..{r2.last}; statuslog {timeline}")

        # H7
        self.start("H7", "which way it went")
        make = self.need("the make call of the object", next((m for m in make_calls if m.first_arg), None))
        fn7 = R.function(make.rel, make.func, within=[make.line])
        call_stmt = R.statement_at(fn7.node, make.line)
        loop = next(
            (a2 for a2 in pathcond.ancestors(call_stmt) if isinstance(a2, (ast.For, ast.AsyncFor)) and ast.unparse(a2.target) == make.first_arg),
            None,
        )
        loop = self.need(f"the loop that binds {make.first_arg}", loop)
        arms, implicit = [make.line], []
        previous = call_stmt
        for anc in pathcond.ancestors(call_stmt):
            if anc is loop:
                break
            if isinstance(anc, ast.If):
                other = anc.orelse if previous in anc.body else anc.body
                if other:
                    arms.append(other[0].lineno)
                else:
                    implicit.append(anc)
            previous = anc
        file_row = next((f for f in self.data["files"] if f["destinationDBlock"]["value"] == obj), None)
        env7 = self.base_env(file_row, at=r2.first)
        env7.put(make.first_arg, obj, "the object's name, bound by the loop over names")
        seens7 = [R.seen(fn7, r2, self.mode)]
        stable_holes(seens7, env7)
        self.say(f"candidates: the make call {make.line}, and the branches around it {arms[1:]} + implicit else of {[i.lineno for i in implicit]}")
        # an implicit else has no statement to read rows from: its test is evaluated instead
        for i in implicit:
            value = E.evaluate(ast.unparse(i.test), env7)
            used = {k: v for k, v in env7.used.items() if k in ast.unparse(i.test)}
            unread = env7.unread_in(ast.unparse(i.test))
            used |= {k: f"unknown: {why}" for k, why in unread.items()}
            self.say(f"  implicit else of {i.lineno}: test {ast.unparse(i.test)!r} is {value!r} {used}" + (" -> it did not run" if value is True else ""))
            if value is not True:
                if unread:
                    self.rows_between(env7.at)
                if file_row is None:
                    self.say(f"  the rows file has no file row whose destinationDBlock is {obj!r}: its values are not read")
                self.need(f"the implicit else of {i.lineno}", None)
        cand7 = Candidate(make.rel, make.func, sorted(set(arms)), run=r2)
        j = judge([cand7], {r2.log.name: [r2]}, mode=self.mode, decide=self.decide, question_run=r2, env=env7)
        self.say(j.report(r2))
        exit7 = self.pick("which way it went", j)
        self.touched.append((fn7, r2))
        return Branch(fn7, exit7, r2, env7)

    # ------------------------------------------------------------ Branch: what the guard held
    def branch(self, q: Branch):
        self.start("H8", "what the guard held")
        fn7, exit7, env7 = q.fn, q.exit, q.env
        exit_stmt = R.statement_at(fn7.node, exit7.line)
        steps7 = V.walk_steps(fn7, exit7.line)
        E.written_after(fn7.node, exit7.line, env7)
        solved_atoms: dict[str, bool] = {}
        previous = exit_stmt
        tests = []
        for anc in pathcond.ancestors(exit_stmt):
            if anc is fn7.node:
                break
            if isinstance(anc, ast.If) and (previous in anc.body or previous in anc.orelse):
                tests.append((anc, previous in anc.body))
            previous = anc
        for anc, outcome in tests:
            text = ast.unparse(anc.test)
            for atom in E.atoms(anc.test):
                for n in ast.walk(atom):
                    if isinstance(n, ast.Name) and env7.lookup(n.id) is E.UNKNOWN:
                        t = E.taken(steps7, n.id, env7, owner=fn7.owner)
                        if t.step is not None and t.value is not E.UNKNOWN:
                            env7.put(n.id, t.value, f"binding @{t.step.line} {t.step.value}")
                            self.say(f"  {t.describe()}")
            sol = E.solve(text, outcome, env7)
            self.say(f"  {anc.lineno}: {text}  is {outcome}")
            self.say(f"     known {sol.known}")
            self.say(f"     settled {sol.settled}  open {sol.open}")
            for k, (val, why) in sol.derived.items():
                env7.put(k, val, f"solved from {anc.lineno}: {why}")
                self.say(f"     derived {k} = {val!r}  ({why})")
            solved_atoms.update(sol.settled)
        self.say(f"  values used: {env7.used}")

        def two_sided(text: str) -> bool:
            body = ast.parse(text, mode="eval").body
            return isinstance(body, ast.Compare) and not any(isinstance(x, ast.Constant) for x in [body.left, *body.comparators])

        settled = {t: v for t, v in solved_atoms.items() if two_sided(t)}
        self.need("a settled comparison of two values", settled)
        return Settled(fn7, exit7, steps7, env7, settled)

    # ------------------------------------------------------------ Settled: the same test elsewhere
    def settled(self, q: Settled):
        self.start("H9", "the same test elsewhere")
        fn7, exit7 = q.fn, q.exit
        question = self.question
        other_env = self.base_env(at=question.first if question is not None else None)
        stable_holes([R.seen(f, q2, self.mode) for f, q2 in self.touched if q2 is question], other_env)
        pairs = []
        for atom, truth in q.settled.items():
            found = V.occurrences(atom, [(f, q2) for f, q2 in self.touched if f.owner != fn7.owner], self.mode, other_env)
            seen_at = set()
            self.say(f"  {atom!r} ({truth} at {fn7.name}:{exit7.line}): {len(found)} occurrence(s)")
            for o in found:
                if (o.fn.owner, o.line) in seen_at:
                    continue
                seen_at.add((o.fn.owner, o.line))
                self.say(f"    {o.fn.name}:{o.line} -> {o.value!r}  {dict(o.env.used)}")
                if o.value is (not truth):
                    pairs.append((atom, truth, o))
        self.need("exactly one test that came out the other way", len({(a2, o.fn.owner, o.line) for a2, _, o in pairs}) == 1)
        atom, truth, other = pairs[0]
        self.say(f"contradiction: {fn7.name}:{exit7.line} has {atom} {truth}; {other.fn.name}:{other.line} has it {other.value}")
        a_side = V.Side(f"{fn7.name}", q.steps, q.env, fn7.owner)
        b_side = V.Side(f"{other.fn.name}", V.walk_steps(other.fn, other.line), other.env, other.fn.owner)
        return Pair(atom, a_side, b_side)

    # ------------------------------------------------------------ Pair: where they part
    def pair(self, q: Pair):
        self.start("H10", "where they part")
        atom, a_side, b_side = q.atom, q.a, q.b
        for line in V.diff(atom, a_side, b_side):
            self.say(line)
        ends = []
        for name in V.names_of(atom):
            va, vb = V.value_of(name, a_side)[0], V.value_of(name, b_side)[0]
            if va is E.UNKNOWN or vb is E.UNKNOWN or va == vb:
                continue
            for side in (a_side, b_side):
                t = E.taken(side.steps, name, side.env, owner=side.owner)
                if t.step is not None and t.step.terminal:
                    read = [n.id for n in ast.walk(ast.parse(t.step.value, mode="eval")) if isinstance(n, ast.Name) and n.id != "self"]
                    for n in dict.fromkeys(read):
                        A.bindings(side.steps, t.step.owner, n)
                    bound = {s.name for s in side.steps}
                    loose = [n for n in read if n not in bound]
                    end = f"{side.label}:{t.step.line} stops at {t.step.terminal} ({t.step.detail})"
                    if loose:
                        end += f"; the walk does not say what {loose} was looked up with"
                    ends.append((end, self.looked_up_with(side, t.step)))
        for end, how in ends:
            self.end(end)
            if how:
                self.say("    looked up with:")
                for x in how:
                    self.say(f"      {x}")
        # an input that differs by value and ends in no stop has not said where
        # the two sides part; the two sides being unknown is not an answer either
        self.need("an input the two sides part on, ending in a stop", ends)
        return None

    # ------------------------------------------------------------ FalseAtom: an exception as a false guard
    def false_atom(self, q: FalseAtom):
        # R1
        self.start("R1", "what raised")
        frame, ch, question = q.frame, q.chain, q.question
        fn = R.function(frame.rel, frame.func, within=[frame.line])
        stmt = R.statement_at(fn.node, frame.line)
        self.need(f"the statement at {frame.func}:{frame.line}", stmt)
        self.say(f"{Path(frame.rel).name}:{frame.line} {frame.func}: {ast.unparse(stmt).splitlines()[0]}")
        read = X.read(stmt, ch.exc_type, ch.exc_message)
        for g in read:
            self.say(f"  {g.describe()}")
        if len(read) != 1:
            self.need(f"one expression of the line that raises {ch.exc_type}", None)
        fg = read[0]
        if fg.kind != "in":
            self.need(f"a way to compare the operands of {fg.kind!r}", None)

        # R2
        self.start("R2", "what the operands held")
        env = self.base_env(at=question.first if question is not None else None)
        cls = class_of(fn)
        # a plugin's own attributes (``self.add_to_top_only``) printed by its other methods in the run
        stable_holes([R.seen(f, q2, self.mode) for f, q2 in self.touched if q2 is question and class_of(f) == cls], env, cls)
        steps = A.walk(fn, frame.line)
        self.say(f"walk from {fn.name}:{frame.line}: asked by question" + (f"; {steps.note}" if steps.note else ""))
        ran = [(f"{f.rel}::{f.func}", f.line) for f in ch.frames if f.in_corpus]
        walk = G.Walk(steps, env, self._collections(env), self.data, ran, ask=self.ask_collection)
        if walk.held:
            self.say(f"held (the traceback's frames ran past them): {sorted(f'{w}: {t}' for _, w, t in walk.held)}")
        drawn = walk.element_of(fg.operand, fn.owner)
        self.say(f"{fg.operand} = {fg.printed!r} (the message)" + (f", an element of the keys of {drawn[0]}" if drawn else ""))
        key_side = self.need(f"the container {fg.operand} is drawn from", drawn)
        k_keys = walk.keys(*key_side)
        c_keys = walk.keys(fg.other, fn.owner)
        for label, keys in ((f"keys of {key_side[0]}", k_keys), (f"keys of {fg.other}", c_keys)):
            reach = [o.reached for o in keys.origins]
            self.say(f"  {label}: {len(reach)} write(s) for a row: {reach.count(True)} reached, {reach.count(False)} not, "
                     f"{len(reach) - reach.count(True) - reach.count(False)} unknown"
                     + (f"; opaque {keys.opaque}" if keys.opaque else "") + ("" if keys.complete() else "; not complete"))
        lines, parts = G.part(fg.printed, k_keys.origins, c_keys, walk)
        self.say(f"  which write of {key_side[0]} put the message's value in:")
        for line in lines:
            self.say(line)
        self.say(f"  {fg.other}'s keys ({'complete' if c_keys.complete() else 'as far as the walk says'}):")
        for o in c_keys.origins:
            self.say(f"    {o.where()} = {o.value!r}" + ("" if o.reached is True else f"  (not reached: {o.why})" if o.reached is False else f"  ({o.why})"))
        inside = fg.printed in c_keys.possible()
        self.say(f"  the message's value is {'among' if inside else 'not among'} the {len(c_keys.possible())} key(s) that may have gone in"
                 f" ({len(c_keys.certain())} certainly did)" + ("" if inside else ": consistent with the exception"))
        if inside:
            self.need("the key set the walk gives, consistent with the exception", None)
        if not c_keys.complete():
            self.need(f"a complete key set for {fg.other}", None)

        # R3
        self.start("R3", "where they part")
        self.need("one binding of the key, for the same row, whose value is in the key set", parts)
        if len({(p.alternative.step.line, p.decided_by) for p in parts}) != 1:
            self.need("one alternative and one guard against it", None)
        p = parts[0]
        write, took = p.taken
        self.say(f"took effect: {took.where()} = {took.value!r}" + (f"  [guards {took.step.guards}]" if took.step.guards else ""))
        self.say(f"instead of:  {p.alternative.where()} = {p.alternative.value!r}, which is among {fg.other}'s keys")
        used = {k: v for k, v in p.env.used.items() if k in p.decided_by}
        self.say(f"the guard against it: {p.decided_by!r} is False for {p.alternative.element.label if p.alternative.element else 'this call'}  {used}")
        context, rest = G.inputs(p.decided_by, p.alternative.element)
        self.say(f"its inputs: {rest}; the loop's row: {context}")
        self.need("one input of the guard besides the loop's row", len(rest) == 1)
        expr = rest[0]
        value = E.evaluate(expr, p.env)
        self.say(f"  {expr} = {value!r} ({p.env.used.get(expr, 'unknown')})")
        how, leaves = G.derivation(expr, walk)
        for line in how:
            self.say(f"    {line}")
        leaf = self.need(f"what {expr} is computed from, in the walk", leaves)
        self.need(f"one thing {expr} is computed from", len(leaf) == 1)
        method, owners, opened = G.computed_by(expr, walk)
        if owners and not method:
            self.say(f"  not recomputed: {expr} returns from {owners}" + "".join(f"; {o}" for o in opened))
        field_q = Field(leaf[0], f"{expr} is computed from it; that guard chose {took.where()} over {p.alternative.where()}", p.env, expr, method)
        return Reproduce(fn, frame, ch, walk, p, field_q)

    # ------------------------------------------------------------ Reproduce: the same again
    def reproduce(self, q: Reproduce):
        self.start("R4", "the same again")
        fn, frame, ch, walk = q.fn, q.frame, q.chain, q.walk
        params = [a.arg for a in fn.node.args.args]
        alt = q.part.alternative
        runs = {}
        for label, prefer in (("as it ran", {}), (f"with {alt.step.name} = @{alt.step.line} {alt.step.value}", {alt.step.name: alt.step.line})):
            args = {}
            self.say(f"{fn.name}({', '.join(params)}) {label}:")
            for name in params:
                if name == "self" or not (walk.defs(name, fn.owner) or walk.handed(name, fn.owner)):
                    continue
                value, notes = walk.build(name, fn.owner, prefer)
                if value is None:
                    self.say(f"  {name}: not rebuilt ({'; '.join(notes)})")
                    continue
                args[name] = value
                shown = list(value)[:8] if isinstance(value, dict) else value
                self.say(f"  {name}: {len(value) if isinstance(value, dict) else ''} key(s) {shown}")
                for n in notes:
                    self.say(f"    assumed: {n}")
            opaque = [n for n in params if n not in args]
            self.say(f"  opaque: {opaque}")
            out = P.run(str(CORPUS), fn.rel, fn.name, args, fn.node.lineno)
            self.say(f"  -> {out.describe()}")
            for a in dict.fromkeys(out.assumptions):
                self.say(f"     assumed: {a}")
            runs[label] = out
        first, second = runs.values()
        exc = ch.exc_type.rsplit(".", 1)[-1]
        same = first.raised == exc and first.line == frame.line and first.message == ch.exc_message.strip()
        self.say(f"reproduced: {'yes' if same else 'no'} ({exc} at {frame.line} with {ch.exc_message.strip()!r} in the log)")
        if not same and not first.needed:
            self.need("the rebuilt inputs raise what the log shows", None)
        if same:
            moved = not (second.raised == exc and second.line == frame.line)
            self.say(f"with the alternative: {'does not raise at ' + str(frame.line) if moved else 'raises the same'}"
                     + (f" ({second.describe()})" if moved else ""))
        return q.then

    def _collections(self, env: E.Env) -> dict[str, dict]:
        """The rows file's collections (``job.Files`` -> the file rows), keyed by the canonical spelling."""
        out = {}
        for text, spec in self.data.get("collections", {}).items():
            if text != "about":
                out[env.canonical(text)] = spec
        return out

    # ------------------------------------------------------------ Field: a row field's value
    def field(self, q: Field):
        self.start("R5", "what the input held")
        key = q.env.canonical(q.expr)
        m = re.fullmatch(r"<(\w+)>\.(\w+)", key)
        what = f"{m[1]}.{m[2]}" if m else q.expr
        self.say(f"next question: {what} for this instance ({q.why})")
        value = q.env.lookup(q.expr)
        unread = q.env.unread_in(q.expr)
        why = next(iter(unread.values()), "the rows file does not have it")
        if value is E.UNKNOWN and unread:
            self.rows_between(q.env.at)
        found = self.need(f"the value of {what} (a field of the {m[1] if m else '?'} row: {why})",
                          None if value is E.UNKNOWN else repr(value))
        self.end(f"{what} = {found}")
        if q.computed and q.method and m:
            self.say(f"  {q.computed} again, by {q.method.split('::')[-1]} with this value: {self.recompute(q, m[2], value)}")
        return None

    def recompute(self, q: Field, attr: str, value) -> str:
        """Call the method the walk named on its own class, with the one field it reads set to *value*.

        PanDA's code computes it, not a copy of it here.  The answer is
        compared with what the run printed for the call; a difference says the
        row's value is not the one the code saw.
        """
        rel, func = q.method.split("::", 1)
        cls = class_of(R.function(rel, func))
        module = importlib.import_module(rel.removesuffix(".py").replace("/", "."))
        obj = getattr(module, cls)()
        object.__setattr__(obj, attr, value)
        got = getattr(obj, func)()
        printed = q.env.lookup(q.computed)
        same = "the same as" if got == printed else "NOT the same as"
        return f"{got!r} -- {same} the run printed ({q.env.used.get(q.computed, 'unknown')})"


    # ------------------------------------------------------------ Stuck: what moves it on
    def stuck(self, q: Stuck):
        self.start("S1", "what moves it on")
        statuslog = self.data.get("statuslog", {})
        rows = [s for s in statuslog.get("states", []) if s["status"] == q.observed]
        if rows or statuslog.get("as_of"):
            self.result["window"] = {"start": _iso(rows[-1].get("time_extended") or rows[-1]["time"]) if rows else None,
                                     "end": _iso(statuslog.get("as_of")), "end_kind": "as_of", "end_from": "statuslog"}
        symptom = Symptom(subject=q.subject, observed=q.observed, task_id=q.key)
        inv, _ = asyncio.run(_investigate(mapdump.code_map("rerun"), symptom, None, q.key, None, 1, None))
        R.check_blobs(inv, "derive-strategy's readings")
        strategy = inv.hops[0].strategy
        follow = self.need("the map's answer to what moves the value on", strategy.follow_up)
        strategy_mod.attach_mover_gaps(strategy, R.ROOTS)
        writers = asyncio.run(self._writers_of(q.subject))
        movers = movers_of(follow, writers, q.subject, q.observed)
        gaps = map_gaps(follow)
        # a mover whose rows reach none of its arms, and whose reach writes nothing else, takes a row by its key alone
        elsewhere = self._rows_elsewhere(q, follow, writers, movers, gaps)
        for owner, (to_key, _, _) in elsewhere.items():
            if to_key:
                del movers[owner]
        # a writer that takes a row by its key: asked over the window's last part alone (KNOWN_ISSUES.md K16)
        keyed = by_key_of(writers, movers, q.subject, q.observed)
        # the functions whose runs S1 reads, as _of_mover names them: the door asks no other DB proxy method's mark,
        # and theirs for their own rows (a name two of them share: whole)
        ids = list(dict.fromkeys([*movers, *keyed]))
        names = [o.split("::")[-1] for o in ids]
        owners = {n: R.mark_heads(*o.split("::", 1)) if names.count(n) == 1 else () for n, o in zip(names, ids, strict=True)}
        wakes = wakers_of(follow, writers, q.subject, q.observed)
        self.say(f"selected by {follow.selected_by}; guarded by {[g.owner for g in follow.guarded_by]}")
        for owner, why in movers.items():
            self.say(f"  mover {owner}: {why}")
        for owner in keyed:
            self.say(f"  mover {owner}: a writer that takes a row by its key")
        for owner, (_, said, _) in elsewhere.items():
            self.say(f"  {owner.split('::')[-1]}: {said}")
        said_gaps = {owner: strategy_mod.gap_texts(held) for owner, held in gaps.items()}
        self.say(f"where the map could not read them: {said_gaps or 'nowhere'}" + (f" ({follow.gaps_note})" if follow.gaps_note else ""))
        readings = self._read_at_as_of(q, follow, writers, movers)
        asking = W.stuck(self.data.get("statuslog", {}), q.observed)
        asked = f"{q.subject}={q.observed}"
        # the statements that give rows (_reached): an aggregate hands on none
        giving = [rs for rs in follow.reader_statements if asked in rs.selects and rs.result != RESULT_AGGREGATE]
        handers = {rs.owner for rs in giving if rs.crosses_to}
        ran, unanswered, unsearched = [], [], []
        entries = []
        for owner in [*movers, *keyed]:
            arms = arms_of(writers, owner, q.subject, q.observed)
            files = [f for j in writers if j.owner == owner for f in j.log_files]
            mine = [(s, v) for o, s, v in readings if o == owner]
            way = "key" if owner in keyed else "guard" if movers[owner] == GUARDED else "select"
            entry = {"owner": owner, "reached": way,
                     "statements": [s for s, _ in mine if s], "verdict": None, "reasons": [],
                     "readings": [{"statement": s or None, "verdict": v} for s, v in mine], "premises": None}
            entries.append(entry)
            if owner in elsewhere:
                to_key, said, reach = elsewhere[owner]
                entry["reasons"].append({"code": "K16" if to_key else "K14", "text": said})
                if reach is not None:
                    entry["reach"] = reach
            if any(v == "refused" for _, v in mine):
                entry["reasons"].append({"code": "K17", "text": "refused at the moment read, one moment of the window: not dropped"})
            if made := births_of(writers, owner, q.subject, q.observed):
                told = (f"its writes at {made} make the row (births: {birth_why(owner, q.subject, made)}): not arms"
                        " (KNOWN_ISSUES.md K29)")
                self.say(f"  {owner.split('::')[-1]}: {told}")
                entry["births"] = made
                entry["reasons"].append({"code": "K29", "text": told})
            if not arms:
                # with no arm of its own, a mover moves the value, if at all, through what it calls or the rows it hands
                # on: kept, and said why
                rel, func = owner.split("::", 1)
                lines = [rs.line for rs in giving if rs.owner == owner]
                defs = defs_holding(rel, func, lines)
                reached, cut = through_calls(rel, [(f.lineno, f.end_lineno) for f in defs], writers, q.subject, q.observed)
                hidden = hides_where(gaps.get(owner, []))
                kept = kept_without_arms(bool(defs), reached, cut, owner in handers, hidden)
                # the map's reach decides, where it can say for certain that nothing it goes on into moves the value
                reach = self._reach_of(q, owner, lines) if defs else None
                dropped, said = reach_said(reach, hidden)
                if reach is not None:
                    entry["reach"] = reach
                if dropped:
                    self.say(f"  {func}: {said}")
                    entry["verdict"] = "reaches-no-write"
                    entry["reasons"].append({"code": "reach", "text": said})
                    continue
                self.say(f"  {func}: no arm of its own, kept: {'; '.join(kept)}")
                entry["reasons"].append({"code": "K14", "text": f"no arm of its own, kept: {'; '.join(kept)}"})
                if said:
                    self.say(f"  {func}: {said}")
            for w in wakes.get(owner, []):
                self.say(f"  {owner.split('::')[-1]}: {wake_said(w)}")
                entry.setdefault("wakers", []).append({"waker": w.owner, "line": w.line, "writes": w.writes,
                                                       "reader": w.reader, "term": w.term, "candidate": list(w.candidate)})
            if gaps.get(owner):
                entry["reasons"].append({"code": "map-gap", "text": f"the map could not read all of it ({gap_said(gaps[owner])})"})
            if prints_nothing(owner, list(arms)):
                said = (f"{len(arms)} arm(s) that write another value {[(a.line, a.outcomes) for a in arms.values()]}; the map "
                        f"has no prints of its own for it, so no run in a log is its own: not followed (KNOWN_ISSUES.md K25)")
                self.say(f"  {owner.split('::')[-1]}: {said}")
                entry["verdict"] = "prints-nothing"
                entry["reasons"].append({"code": "K25", "text": said})
                continue
            files = self.files_of(owner, list(dict.fromkeys(files or follow.reader_log_files)), list(arms))
            to_hand = self.logs.names(files)
            if to_hand:
                self.warn_split(owner, list(arms))
            self.say(f"  {owner.split('::')[-1]}: {len(arms)} arm(s) that write another value "
                     f"{[(a.line, a.outcomes) for a in arms.values()]}; logs {files} ({'to hand: ' + str(to_hand) if to_hand else 'none to hand'})")
            if not to_hand:
                entry["verdict"] = "no-log"
                entry["premises"] = self._premises(q, owner, list(arms), to_hand, asking, asking, [], {}, writers)
                continue
            found, missed, holes, searched = self._ask_mover(q, owner, to_hand, asking, rest=way != "key", owners=owners)
            unanswered += missed
            logs = self._runs_of(found, owner, searched)
            if not logs:
                entry["premises"] = self._premises(q, owner, list(arms), to_hand, asking, searched, missed, holes, writers)
            if logs:
                ran.append(Passes(owner, logs, list(arms.values()), q.key, searched, owners))
                entry["verdict"] = "ran"
            elif holes:
                # not found is not "did not run" where the search did not reach
                unsearched.append(owner.split("::")[-1])
                self.say(f"  {owner.split('::')[-1]}: no run of its own found, but {', '.join(holes)} were not searched all the window: it may have run there")
                entry["verdict"] = "not-searched"
            elif missed:
                entry["verdict"] = "asks-unanswered"
            else:
                # not followed; ruled out only where judgement 3's premises all hold (build order 2)
                over = " over the whole window" if searched is not asking else ""
                if way == "key" and self.logs.needs_window and W.rest(statuslog, q.observed) is not None:
                    over = (f" from {asking.window[0]}, the window's last part, which alone is asked of a writer that "
                            f"takes a row by its key (KNOWN_ISSUES.md K16)")
                if all(p["held"] for p in entry["premises"].values()):
                    said = f"no run of its own found in {to_hand}{over}: it did not run (judgement 3)"
                    entry["verdict"] = "did-not-run"
                    entry["reasons"].append({"code": "judgement-3", "text": said})
                else:
                    said = f"no run of its own found in {to_hand}{over}: not followed, though not ruled out (KNOWN_ISSUES.md K19)"
                    entry["verdict"] = "no-run-found"
                    entry["reasons"].append({"code": "K19", "text": said})
                self.say(f"  {owner.split('::')[-1]}: {said}")
        # a mover dropped as reaching no write may be what the movers kept wait on (the map's wakers, ix's 6 b)
        woken = woken_of(entries, wakes)
        for entry in entries:
            if entry["owner"] in woken:
                said = woken_said(woken[entry["owner"]])
                self.say(f"  {entry['owner'].split('::')[-1]}: {said}")
                entry["reasons"].append({"code": "K39", "text": said})
        self.result["movers"] = entries
        self._stop_unanswered(unanswered)
        # every mover that ran and left the value is part of the answer: each goes on alone (tmaeno, 2026-10-13)
        self.need(
            "a mover with a run of the key in a log to hand"
            + (f" ({asking.why})" if self.logs.needs_window else "")
            + (f"; not searched all the window for {unsearched}" if unsearched else ""),
            ran,
        )
        if unsearched:
            self.say(f"movers that may have run where the search did not reach: {unsearched}")
        if len(ran) == 1:
            return ran[0]
        return Fork(ran, [f"mover {p.owner}" for p in ran])

    def _graph(self):
        """The dump's ``reach_graph`` read once: ``(the reach module, its ReachGraph, its rows)``, or None where the dump
        has none."""
        if self._graph_read is None:
            rows = getattr(mapdump.fragment(), "reach_graph", None)
            if rows:
                from bamboo.codemap import reach as RE

                self._graph_read = (RE, RE.ReachGraph(rows), rows)
            else:
                self._graph_read = False
        return self._graph_read or None

    def _called_at(self, caller: R.Function, callee: Candidate) -> list[str]:
        """Where *caller* calls on into *callee*'s function, by the map's reach: the ``file:line`` of each call of
        *caller*'s from which the reach goes on into it, every line of the call (``ReachCall.lines``) -- for H2 to read
        a run inside one of them as a part of the caller's (``which_ran.inside_call``).

        A reach from *caller* gives one way in (``ReachWrite.chain``), and a second call into the same function, or a
        call into another that goes on there too (the adder's ``process_job_report`` reaches ``archiveJob`` through
        ``updateJobs`` at 471 and ``check_file_status_in_jedi`` at 400), would be missed.  So each call is asked on its
        own, as the reach from a copy of *caller*'s row holding that call alone, by the map's own rules (the classes a
        call holds for and enters as).  A way that comes back into *caller* and on through its other calls counts as
        inside the call, as it runs there."""
        graph = self._graph()
        if graph is None or not caller.node.body or not callee.lines:
            return []
        RE, record, rows = graph
        start = record.function_at(caller.rel, caller.node.body[0].lineno)
        end = record.function_at(callee.rel, callee.lines[0])
        if start is None or end is None or start == end:
            return []
        if (start, end) not in self._calls_into:
            row, file = record.rows[start], start.partition("::")[0]
            lines: set[int] = set()
            for n, call in enumerate(row.calls):
                one = row.model_copy(update={"owner": f"{start}#{n}", "calls": [call]})
                # the writes reached alone are read: a call closed by its name adds no way (by_name off)
                reached = RE.ReachGraph([*rows, one]).reach([one.owner], RE.Destination({end: ["called"]}, []), by_name=False)
                if any(w.function == end for w in reached.writes or []):
                    lines.update(call.lines or [call.line])
            self._calls_into[(start, end)] = [f"{file}:{line}" for line in sorted(lines)]
        return self._calls_into[(start, end)]

    def _rows_elsewhere(self, q: Stuck, follow, writers: list, movers: dict, gaps: dict) -> dict[str, tuple[bool, str, dict | None]]:
        """For each mover whose statements' rows reach none of its arms (``elsewhere_of``), what the map's reach says of
        it as of a mover with no arm of its own, those arms left out of where it ends (``_reach_of``'s *without*).
        Where it reaches no write of another value with nothing left open, and no gap of the map's may hide where its
        rows go (``reach_said``), the rows it selects stay with it unwritten, and its arms are those of a writer that
        takes a row by its key (KNOWN_ISSUES.md K16): ``(True, the line saying so, the reach)``.  Else it stays a mover,
        its arms as before, said why: ``(False, ...)``."""
        out: dict[str, tuple[bool, str, dict | None]] = {}
        asked = f"{q.subject}={q.observed}"
        for owner, arms in elsewhere_of(follow, writers, q.subject, q.observed).items():
            if owner not in movers:
                continue
            mine = [rs for rs in follow.reader_statements if rs.owner == owner and asked in rs.selects and rs.result != RESULT_AGGREGATE]
            lines = [rs.line for rs in mine]
            rel, func = owner.split("::", 1)
            reach = self._reach_of(q, owner, lines, without=arms) if defs_holding(rel, func, lines) else None
            dropped, said = reach_said(reach, hides_where(gaps.get(owner, [])))
            whose = (f"the rows {', '.join(f'{func}@{n}' for n in lines)} select(s) reach none of its arms {arms}: the map names "
                     f"each one's row by a key they do not fill")
            if dropped:
                out[owner] = (True, f"{whose}, and the reach from it, those arms left out, writes no other value with nothing "
                                    f"left open (closure {reach['closure']}; assuming {sorted(reach['assumptions'])}): a writer "
                                    f"that takes a row by its key (KNOWN_ISSUES.md K16)", reach)
            else:
                out[owner] = (False, f"{whose}; kept: {said or 'the reach was not asked'}", reach)
        return out

    def _reach_of(self, q: Stuck, owner: str, lines: list[int], without: list[int] | tuple = ()) -> dict | None:
        """What the map's reach says of a mover with no arm of its own: from the defs holding its statements at *lines*
        -- every def of its name where it has none, as it takes rows another selects -- does any function it can go on
        into hold an arm moving *q.subject* off *q.observed* (``reach.moving_off``)?  The writes reached (``[]`` for none
        with nothing left open, ``None`` for none with calls left open), the holes, the assumptions the answer relied
        on and the closure's size, the calls it closed by their name (``closed``); ``own_arms`` names a start the
        junctions give an arm S1 does not count.  *without*: the starts' own arms at those lines are left out of where
        the reach ends -- the rows do not reach them (``_rows_elsewhere``).  The reach
        reads the junctions without their births (``unborn_junctions``), as ``arms_of`` does.  ``None`` where the dump
        has no ``reach_graph``: said once, as the first ask is (build2-design.md 6)."""
        if self._reach is None:
            graph = self._graph()
            if graph is not None:
                RE, record, rows = graph
                self._reach = (RE, record, {})
                self.result["reach"] = {"present": True, "functions": len(rows), "premises": sorted(RE.PREMISES)}
                self.say(f"  reach: the dump's reach_graph, {len(rows)} function(s); every answer rests on {sorted(RE.PREMISES)}")
            else:
                self._reach = False
                self.result["reach"] = {"present": False}
                self.say("  reach: the dump has no reach_graph: a mover with no arm of its own is kept (KNOWN_ISSUES.md K14)")
        if not self._reach:
            return None
        RE, graph, ends = self._reach
        if (q.subject, q.observed) not in ends:
            junctions = unborn_junctions(mapdump.fragment().junctions, q.subject)
            ends[q.subject, q.observed] = RE.moving_off(junctions, graph, q.subject, q.observed)
        destination = ends[q.subject, q.observed]
        rel = owner.split("::", 1)[0]
        starts = [graph.function_at(rel, n) for n in lines] if lines else graph.named(owner)
        starts = list(dict.fromkeys(s for s in starts if s))
        if not starts:
            return {"starts": [], "writes": None, "holes": 1, "first_holes": [f"{owner} -- no function of the reach_graph holds it"],
                    "assumptions": {}, "closure": 0, "own_arms": []}
        if without:
            # an arm is "<outcome>@<line>"
            destination = RE.Destination({f: kept for f, held in destination.arms.items()
                                          if (kept := [a for a in held if f not in starts or int(a.rpartition("@")[2]) not in without])},
                                         destination.unmatched)
        answer = graph.reach(starts, destination)
        return {"starts": starts,
                "writes": None if answer.writes is None else [{"function": w.function, "arms": w.arms, "chain": w.chain}
                                                              for w in answer.writes],
                "holes": len(answer.holes), "first_holes": answer.holes[:3], "assumptions": answer.assumptions,
                "closed": len(answer.closed), "first_closed": answer.closed[:3],
                "closure": answer.closure, "own_arms": [s for s in starts if s in destination.arms]}

    def _ask_mover(self, q: Stuck, owner: str, to_hand: list[str], asking: W.Asking, rest: bool = True,
                   owners: Owners | None = None):
        """S1's asks for one mover: the window's last part first (*asking*, ``STUCK_SPAN``), and the rest of the window
        (``W.rest``) only where every ask of the last part was answered whole and held no run of its own -- no pass in
        six hours is not no pass in the window, and a mover that ran and stopped is the one a stuck value most needs
        named (build order 2a; tmaeno, 2026-10-16).  Not *rest* for a writer that takes a row by its key: its logs are
        mostly the DB proxy's, grepped whole over days (KNOWN_ISSUES.md K16; tmaeno, 2026-10-26).  The door over files
        reads each file whole and asks once.

        Returns what ``_ask`` does, and the asking the runs were looked for over."""
        found, missed, holes = self._ask(q.key, to_hand, asking, owners)
        if not rest or missed or holes or not self.logs.needs_window or self._runs_of(found, owner, asking):
            return found, missed, holes, asking
        rest = W.rest(self.data.get("statuslog", {}), q.observed)
        if rest is None:
            return found, missed, holes, asking
        self.say(f"  {owner.split('::')[-1]}: no run of its own from {asking.window[0]}: asking the rest of the window")
        return *self._ask(q.key, to_hand, rest, owners), rest

    def _runs_of(self, found: dict[str, list], owner: str, asking: W.Asking) -> list[str]:
        """The logs among *found* holding a run of *owner* in *asking*'s window (any run, for the door over files)."""
        window = asking.window if self.logs.needs_window else None
        return [name for name, rs in found.items() if any((window is None or _overlaps(r, window)) and _of_mover(r, owner) for r in rs)]

    def _premises(self, q: Stuck, owner: str, lines, to_hand: list[str], asking: W.Asking, searched: W.Asking,
                  missed: list, holes: dict, writers: list = ()) -> dict:
        """Judgement 3's premises for a mover with no run of its own found: absence rules it out only where all three
        hold (build order 2a; KNOWN_ISSUES.md K19).  ``window``: the whole window was searched; ``logs``: every log its
        runs print to was asked (``every_log_asked``); ``prints``: every run that reaches an arm prints the key the asks
        search by (``prints_naming``, over *writers*).  Each is ``{"held": True | False | None, "why": ...}``, None where
        it is not known.

        *asking* is the window's last part (``W.stuck``), *searched* the part ``_ask_mover`` asked last, and *missed*
        and *holes* what that ask returned."""
        return {"window": self._searched_whole(q, to_hand, asking, searched, missed, holes),
                "logs": every_log_asked(owner, lines, to_hand),
                "prints": prints_naming(list(writers), owner, lines, self._key_columns(q.subject, q.key))}

    def _key_columns(self, subject: str, key: str) -> set[str]:
        """The columns of *subject*'s class whose value in the rows file is *key* -- the row's key, as ``held.split``
        reads it -- spelled ``Class.attribute`` as the map's names are."""
        cls = subject.partition(".")[0]
        return {f"{cls}.{c}" for c, cell in self._row_of(cls).items() if isinstance(cell, dict) and str(cell.get("value")) == key}

    def _searched_whole(self, q: Stuck, to_hand: list[str], asking: W.Asking, searched: W.Asking, missed: list,
                        holes: dict) -> dict:
        """Judgement 3's first premise: the logs were searched over the whole stuck window, from the value's row less
        SLACK to ``as_of``, every ask answered and nothing of it left unsearched (``Log.holes``).  Not known through
        the door over files: a file handed in does not say what of the window it holds."""
        if not to_hand:
            return {"held": False, "why": "no log was asked"}
        if not self.logs.needs_window:
            return {"held": None, "why": "the door over files does not say what of the window a file holds"}
        # a hole first: asking again does not search better what a log's answer did not reach
        if holes:
            return {"held": False, "why": f"{', '.join(holes)} not searched all the window"}
        if missed:
            return {"held": None, "why": f"{len(missed)} ask(s) not answered yet"}
        if searched is asking and W.rest(self.data.get("statuslog", {}), q.observed) is not None:
            return {"held": False, "why": f"searched from {asking.window[0]} only"}
        return {"held": True, "why": f"{searched.window[0]} .. {asking.window[1]} searched in {to_hand}, none of it unsearched"}

    async def _writers_of(self, subject: str) -> list:
        writers = await mapdump.code_map("rerun").writers_of(subject)
        R.check_blobs(writers, f"the map's writers of {subject}")
        return on_rows(writers)

    # ------------------------------------------------------------ Passes: which way each pass went
    def passes(self, q: Passes):
        self.start("S2", "which way each pass went")
        rel, func = q.owner.split("::", 1)
        # the arms S2 and S3 judge are the mover's own; a mover that writes through what it calls has none here
        self.need(f"an arm of {func}'s own that writes another value (S2 reads no arm of what it calls: KNOWN_ISSUES.md K14)", q.arms)
        fn = R.function(rel, func, [a.line for a in q.arms])
        runs = self.runs(q.key, q.logs, q.asking, q.owners)
        found = []
        for name in q.logs:
            for run in runs.get(name, []):
                if not _of_mover(run, q.owner):
                    continue
                parts, how = R.passes(fn, run, self.mode)
                self.say(f"{run.describe()}: {how}")
                found += parts
        # the runs come machine by machine; the passes are read, and the first and the last taken, in time
        found.sort(key=lambda p: p.first)
        states = self.data.get("statuslog", {}).get("states", [])
        # the stint the value is stuck in began at its last row
        since = next((s["time"] for s in reversed(states) if s["status"] == self.args.observed), None)
        if since is None:
            # a log's coverage stands in for the window only in a file handed in: a door that searches has no other
            self.need(f"a statuslog row for {self.args.observed!r} to start the window at", not self.logs.needs_window)
            for name in q.logs:
                first, last = self.logs.log(name, q.key).covers
                self.say(f"window: the log's coverage {first} .. {last} (the statuslog has no row for {self.args.observed!r})")
        else:
            if self.logs.needs_window and q.asking.window[0] > since:
                since = q.asking.window[0]
                self.say(f"window: from {since} ({q.asking.why})")
            else:
                self.say(f"window: from {since}, when the statuslog shows {self.args.observed!r}")
            found = [p for p in found if p.last[:19] >= since]
        if self.logs.needs_window:
            # a search returns its files' records whatever the window; the statuslog says nothing after its as_of
            # (the file door keeps them: KNOWN_ISSUES.md K4)
            later = [p for p in found if p.first[:19] > q.asking.window[1]]
            if later:
                found = [p for p in found if p.first[:19] <= q.asking.window[1]]
                self.say(f"{len(later)} pass(es) after {q.asking.window[1]}, the end of the window, not read")
        self.need("a pass of the mover in the window", found)
        exits = sorted({n.lineno for n in R.own_nodes(fn.node) if isinstance(n, R.JUMPS)})
        lines = sorted({a.line for a in q.arms} | set(exits))
        self.say(f"candidates: arms {[a.line for a in q.arms]}, exits {exits}")
        chosen, envs = [], []
        for n, p in enumerate(found, 1):
            env = self.base_env(at=p.first)
            stable_holes([R.seen(fn, p, self.mode)], env, class_of(fn), local_to=fn.owner)
            self.say(f"-- pass {n}: {p.first} .. {p.last}, {len(p.records)} record(s); printed {self._printed(env)}")
            j = judge([Candidate(rel, func, lines, run=p)], {p.log.name: [p]}, mode=self.mode, decide=self.decide, question_run=p, env=env)
            self.say(j.report(p))
            chosen.append(self.pick(f"which way pass {n} went", j))
            envs.append(env)
        taken = {v.line for v in chosen}
        self.need("one exit every pass took", len(taken) == 1)
        exit_line = taken.pop()
        self.say(f"every pass ({len(found)}) left by {func}:{exit_line}")
        return Blocked(fn, exit_line, found, envs, q.arms)

    def _printed_in(self, fn: R.Function, run, expr: str) -> list[str]:
        """``value at record`` for every hole of *expr* the function's rows read in *run*."""
        sn = R.claim([R.seen(fn, run, self.mode)])[0]
        return [
            f"{row.capture(run.records[i])[expr].strip()!r} at {run.records[i].span} (hole of {row.line})"
            for row in fn.rows for i, _, _ in sn.hits.get(row.line, ())
            if expr in row.capture(run.records[i])
        ]

    @staticmethod
    def _printed(env: E.Env) -> dict:
        return {k: v.value for k, v in env.values.items()}

    # ------------------------------------------------------------ Blocked: what kept it from moving
    def blocked(self, q: Blocked):
        self.start("S3", "what kept it from moving")
        fn = q.fn
        arm = next((a for a in q.arms if a.line == q.exit_line), None)
        if arm is not None:
            return self._arm_taken(q, arm)
        exit_stmt = R.statement_at(fn.node, q.exit_line)
        path = pathcond.path_condition(exit_stmt)
        self.say(f"{fn.name}:{q.exit_line} is reached when {path}")
        per_pass = []
        for env in q.envs:
            fixed: dict[str, bool] = {}
            for g in path:
                sol = E.solve(g, True, env)
                fixed.update(sol.known)
                fixed.update(sol.settled)
            blocking = {}
            for arm in q.arms:
                blocking[arm.line] = [
                    g for g in arm.guards
                    if not E.read_elsewhere(g)
                    and E._truth(ast.parse(E.strip_annotation(g), mode="eval").body, fixed, env) is False
                ]
            per_pass.append((fixed, blocking))
        fixed, blocking = per_pass[0]
        self.need("the same guards against the arms in every pass", all(b == blocking for _, b in per_pass))
        self.say(f"settled by the exit taken: {fixed}")
        for arm in q.arms:
            self.say(f"  arm {arm.line} {arm.outcomes}: needs {arm.guards}; false in the passes: {blocking[arm.line]}")
        self.need("every arm has a guard the passes made false", all(blocking[a.line] for a in q.arms))
        fewest = min(len(v) for v in blocking.values())
        nearest = [a for a in q.arms if len(blocking[a.line]) == fewest]
        guards = sorted({g for a in nearest for g in blocking[a.line]})
        self.need("one guard against the nearest arm(s)", len(guards) == 1)
        guard = guards[0]
        for a in q.arms:
            if a not in nearest:
                self.say(f"  not followed: arm {a.line} also needs {[g for g in blocking[a.line] if g != guard]}")
        self.say(f"the nearest arm(s) {[a.line for a in nearest]} were kept off by: {guard!r} (rule 21)")
        arm_stmt = R.statement_at(fn.node, nearest[0].line)
        where = guard_if(arm_stmt, guard)
        where = self.need(f"the if whose test is {guard!r}", where)
        return Input(fn, guard, where.lineno, q.passes, q.envs, sorted({o for a in nearest for o in a.outcomes}))

    def _arm_taken(self, q: Blocked, arm: Arm) -> None:
        """S3 where every pass left by an arm that writes the subject: no guard kept the passes off it, so what kept the
        value is what the arm wrote -- a value known at run time (``runtime(<name>)``), read off a row after the arm in
        its block that prints it, nothing binding the name between (``lines_after``; KNOWN_ISSUES.md K26).  The value
        observed in every pass: the mover ran and wrote it again, an END (Fr2's updateJobStatus: "old=running
        new=running", the pilot's reports).  Anything else stops, saying what is not read."""
        fn = q.fn
        self.say(f"{fn.name}:{arm.line} is an arm that writes {arm.outcomes}: every pass ({len(q.passes)}) went through it")
        names = {o[len("runtime("):-1] for o in arm.outcomes if o.startswith("runtime(") and o.endswith(")")}
        name = self.need(f"what arm {arm.line} wrote, yet the value stayed {self.args.observed!r}: one value known at run time "
                         f"(it writes {arm.outcomes})", names.pop() if len(names) == 1 and len(arm.outcomes) == 1 else None)
        spans = lines_after(fn.node, R.statement_at(fn.node, arm.line), name)
        wrote = []
        for n, p in enumerate(q.passes, 1):
            seen = R.claim([R.seen(fn, p, self.mode)])[0]
            values = sorted({row.capture(p.records[i])[name].strip()
                             for row in fn.rows if any(a <= row.line <= b for a, b in spans)
                             for i, _, _ in seen.hits.get(row.line, ()) if name in row.capture(p.records[i])})
            self.say(f"  pass {n} ({p.first}): {name} printed after it {values or 'nowhere'}")
            wrote.append(values)
        self.need(f"the value of {name} arm {arm.line} wrote in every pass: a row after it in its block that prints it, "
                  f"nothing binding {name} between", all(len(v) == 1 for v in wrote))
        values = sorted({v[0] for v in wrote})
        self.need(f"arm {arm.line} wrote {values} in the passes, not the value observed alone, yet the value stayed "
                  f"{self.args.observed!r}: not read", values == [self.args.observed])
        self.end(f"{fn.name}:{arm.line} ran in every pass ({len(q.passes)}) and wrote {name} = {self.args.observed!r}, the "
                 f"value observed: the mover kept the value")
        return None

    # ------------------------------------------------------------ Input: where the guard's name was bound
    def input(self, q: Input):
        self.start("S4", "where the guard's name was bound")
        fn = q.fn
        test = ast.parse(E.strip_annotation(q.guard), mode="eval").body
        names = [n.id for n in ast.walk(test) if isinstance(n, ast.Name) and n.id != "self"]
        names = list(dict.fromkeys(n for n in names if all(E.ev(ast.Name(n), env) is E.UNKNOWN for env in q.envs)))
        self.say(f"{q.guard!r} at {q.line} reads {names} (unknown from the passes)")
        name = self.need("one name the guard reads", names[0] if len(names) == 1 else None)
        steps = A.walk(fn, q.line)
        self.say(f"walk from {fn.name}:{q.line}: asked by question" + (f"; {steps.note}" if steps.note else ""))
        takens = [E.taken(steps, name, env, owner=fn.owner) for env in q.envs]
        t = takens[0]
        for step, false in t.dropped:
            self.say(f"  {name} @{step.line} = {step.value}: out, {false!r} is false")
        for step in t.kept:
            self.say(f"  {name} @{step.line} = {step.value}: kept  {step.guards}")
        self.need(f"one binding of {name} in every pass", t.step is not None and all(x.step is t.step for x in takens))
        step = t.step
        self.say(f"took effect: {t.describe()}")
        for n, p in enumerate(q.passes, 1):
            printed = self._printed_in(fn, p, f"len({name})")
            self.say(f"  pass {n} ({p.first}) printed len({name}): {printed or 'nothing'}")
        if not step.terminal:
            self.need(f"a way on from {name} = {step.value}", None)
        self.say(f"stops: {step.terminal} -- {step.detail}")
        return Rows(fn, name, step, q.passes, q.envs, q.outcomes)

    # ------------------------------------------------------------ Rows: what the call returned
    def rows(self, q: Rows):
        self.start("S5", "what the call returned")
        fn, step = q.fn, q.step
        call = ast.parse(step.value, mode="eval").body
        self.need(f"a call behind {q.name}", isinstance(call, ast.Call))
        callee = ast.unparse(call.func).split(".")[-1]
        # the facade is the one the receiver is declared to be, where it declares one (the stop found it by its name)
        classes, theirs = receiver_methods(callee, call, step.file or fn.rel, step.line)
        receiver = ast.unparse(call.func.value) if isinstance(call.func, ast.Attribute) else ast.unparse(call.func)
        if classes:
            self.say(f"the receiver {receiver} is declared {', '.join(classes)}: its {callee} at "
                     f"{', '.join(f'{f}:{n}' for f, n in sorted(theirs)) or 'none'}")
        else:
            self.say(f"the receiver {receiver} declares no class: every facade's {callee}")
        params, constants, statement, at_call, unread, sites = self.need(
            f"a facade's {callee}, the method the walk's stop names, and its calls on the proxy",
            proxy_reading(callee, call, theirs if classes else None),
        )
        entry = R.entry_row(fn)
        binds: dict[str, list] = {}  # each argument's value in each pass, for the rows ask
        for n, (p, env) in enumerate(zip(q.passes, q.envs, strict=False), 1):
            marks = R.prefix_holes(fn, entry, p.records[0]) if entry is not None else {}
            for expr, value in marks.items():
                env.put_printed(expr, value, f"the mark of {p.records[0].span}")
            args = [argument_said(params[i] if i < len(params) else None, a, E.ev(a, env)) for i, a in enumerate(call.args)]
            args += [argument_said(k.arg, k.value, E.ev(k.value, env)) for k in call.keywords]
            self.say(f"pass {n} ({p.first}): {ast.unparse(call.func)}({', '.join(args)})")
            named = [(params[i] if i < len(params) else f"#{i}", a) for i, a in enumerate(call.args)] + [(k.arg, k.value) for k in call.keywords]
            for name, a in named:
                binds.setdefault(name, []).append(_said_value(E.ev(a, env)))
        self.say(f"the map: {step.detail}")
        if constants:
            dropped = [t for t in statement if t not in at_call]
            self.say(f"at this call ({', '.join(f'{k}={v!r}' for k, v in constants.items())}) it reads {', '.join(at_call)}"
                     + (f" -- not {', '.join(dropped)}" if dropped else ""))
        # The stop reads the method at the call through the map's routes (trace._Walk.database_asked);
        # proxy_reading reads it from the call's literal arguments and the defaults, through the facade to the proxy,
        # and asks no caller further out.  Readings 1 and 2 are of proxy_reading's tables, so the two must agree (a
        # stop once named the statement's 4 tables and this the call's 2).
        # database_asked names at most 6 tables, so 6 named are checked as a part of the call's.
        # A proxy method whose statements are not read leaves the tables of the rest, which readings 1 and 2 would take
        # for all the call reads; both sides name it.
        named, named_unread = stop_reading(step.detail)
        self.need(
            f"every proxy method {callee} reaches at this call read (not read: {unread}; the stop: {named_unread or 'none'})",
            not unread,
        )
        self.need(
            f"the walk's stop and this call to name the same tables (the stop: {named or 'none'}, the call: {at_call})"
            + (f" and read every proxy method (the stop does not read {named_unread})" if named_unread else ""),
            named and not named_unread and (set(named) <= set(at_call) if len(named) >= 6 else set(named) == set(at_call)),
        )
        jobs = self.data.get("job_rows", {}).get("rows")
        history = self.data.get("job_statuslog", {}).get("by_pandaid")
        lacking = [n for n, v in (("job_rows", jobs), ("job_statuslog", history), ("counts", self.data.get("counts", {}).get("rows"))) if not v]
        if lacking:
            self.ask_rows({"kind": "statement", "method": callee, "site": sites, "tables": at_call, "binds": binds,
                           "at": [p.first for p in q.passes]},
                          f"the rows {callee} returned in each pass, their history and the tables' counts (the rows file lacks "
                          f"{', '.join(lacking)})")
        self.need(f"the rows {callee} returned in each pass (the rows file has none)", jobs and history)
        self.say(f"rows: {self.data['job_rows']['source']}")
        counts = self.need("the tables' rows of the key by table and jobStatus at one moment (the rows file has no counts)",
                           self.data.get("counts", {}).get("rows"))
        moment = self.data["counts"].get("source", "the rows file's counts")
        ids = sorted({str(r["pandaid"]) for r in jobs})
        finals = {pid: min((r["t_final"] for r in jobs if str(r["pandaid"]) == pid and r.get("t_final")), default=None) for pid in ids}
        # reading 1 and 2 are of the tables this call reads (proxy_reading), not of every table the statement can
        read = {t.lower() for t in at_call}
        held, final_in, open_out = at_one_moment(counts, at_call)

        def state_at(pid: str, when: str) -> str | None:
            before = [h for h in history.get(pid, []) if h["time_extended"] < when]
            return before[-1]["status"] if before else None

        # the rows file holds the tables at one moment, not at the passes: reading 1 gives no count to set against a
        # pass (tmaeno, 2026-10-14); reading 2 gives one from the job histories
        self.say(f"reading 1, the map's tables at this call: every job of the key -- held at one moment only "
                 f"({held or 'no row'}), not at the passes: no count to set against them")
        uncovered = [t for t in at_call if t.lower() not in {a.lower() for a in ACTIVE_TABLES}]
        if not uncovered:
            self.say(f"reading 2, the call's tables {at_call} hold the jobs not yet in {list(FINAL_STATES)} -- "
                     f"an assumption about where PanDA keeps a row (not the map's)")
        else:
            self.say(f"reading 2, an assumption about where PanDA keeps a row (not the map's): the jobs not yet in "
                     f"{list(FINAL_STATES)} -- of {list(ACTIVE_TABLES)} only; this call also reads {uncovered}")
        fits = not uncovered
        active_by_pass = []
        for n, p in enumerate(q.passes, 1):
            when = p.first.replace(",", ".") + "000"
            states = {pid: state_at(pid, when) for pid in ids}
            active = [pid for pid in ids if states[pid] is not None and states[pid] not in FINAL_STATES]
            active_by_pass.append((p, when, active, states))
            printed = self._printed_in(fn, p, f"len({q.name})")
            value = int(re.match(r"'(\d+)'", printed[0])[1]) if printed else None
            fits &= value == len(active)
            groups = Counter(states[pid] for pid in active)
            self.say(f"  pass {n} at {when}: printed {value}; reading 2 {len(active)} {dict(groups)}")
        self.say(f"reading 2 gives every count the passes printed: {'yes' if fits else 'no'}")
        self.need("reading 2 to give every count the passes printed", fits)
        # its assumption, checked where both sides are of one moment: the counts, by table and jobStatus
        self.say(f"at one moment ({moment}): {at_call} hold {sum(held.values())} row(s){f' {held}' if held else ''}; "
                 f"final among them: {final_in or 'none'}; not yet final outside them: {open_out or 'none'}")
        self.need(f"reading 2's assumption at that moment: no final job in {at_call}, and none not yet final outside them",
                  not final_in and not open_out)
        last_pass, _, active, states = active_by_pass[-1]
        for status, _members in sorted(Counter(states[pid] for pid in active).items()):
            pids = [pid for pid in active if states[pid] == status]
            self.say(f"  group at the last pass: {status} {len(pids)}: {pids[:6]}{' ...' if len(pids) > 6 else ''}")
        # criterion 4: did the value move later, and had the input emptied by then?
        # the arm writes one of its outcomes per command; a row after the last pass is the move
        after = last_pass.first[:19]
        later = [st for st in self.data.get("statuslog", {}).get("states", []) if st["time"] > after]
        moved = next((st for st in later if st["status"] in q.outcomes), None)
        if moved is None:
            # a row after the pass with another value is a move elsewhere; no row at all, a wait still open
            self.need(f"a statuslog row after {after} with what the kept-off arm writes {q.outcomes}", not later)
            as_of = self.need("the time the statuslog was read (its as_of), to say the value had not moved by then",
                              self.data.get("statuslog", {}).get("as_of"))
            # the record door drops the passes after as_of (S2), the file door does not: a statuslog read before the
            # last pass says nothing of the time after it (e1, 2026-10-14)
            self.need(f"the statuslog read after the last pass ({after}; it was read at {as_of})", as_of > after)
            # the jobs are named from the tables at the counts' moment: the job rows of that moment, after the last pass
            tables_at = self.need("the time the counts were read (their as_of), to name the jobs in the tables then",
                                  self.data["counts"].get("as_of"))
            rows_at = self.data["job_rows"].get("as_of")
            self.need(f"the job rows read at the counts' moment ({tables_at}; they were read at {rows_at})", rows_at == tables_at)
            self.need(f"the tables read after the last pass ({after}; they were read at {tables_at})", tables_at > after)
            n_held = sum(held.values())
            self.need(f"a job in {at_call} at {tables_at}, the value not moved by {as_of} (none: the arm kept off was open)", n_held)
            open_rows = sorted((str(r["pandaid"]), r["jobstatus"]) for r in jobs if (r.get("tbl") or "").lower() in read)
            self.need(f"the job rows to hold the {n_held} row(s) the counts put in {at_call} (they hold {len(open_rows)})",
                      len(open_rows) == n_held)
            named = [f"{pid} {status}" for pid, status in open_rows]
            self.end(f"a wait still open: no statuslog row after {after} by {as_of}, and {q.name} would hold the "
                     f"{len(named)} job(s) in {at_call} at {tables_at}: {named[:6]}{' ...' if len(named) > 6 else ''}; "
                     f"each is a question of its own (not followed)")
            return None
        # every job made before the move and not yet final at the first pass, not only the last pass's:
        # a job made after the passes also holds the list open
        created = {pid: min(r["creationtime"] for r in jobs if str(r["pandaid"]) == pid) for pid in ids}
        first = q.passes[0].first[:19]
        waited = [pid for pid in ids if created[pid] < moved["time"] and (finals[pid] is None or finals[pid] > first)]
        if any(finals[pid] is None for pid in waited):
            self.need(f"a final state for every job in {q.name} before the value moved", None)
        last = max(finals[pid] for pid in waited)
        made_later = [pid for pid in waited if created[pid] > q.passes[-1].first[:19]]
        self.say(f"the arm kept off wrote {moved['status']!r} at {moved['time']}; of the {len(waited)} job(s) made before it and "
                 f"not final at the first pass ({len(made_later)} made after the last pass), the last reached a final state at {last}")
        if last > moved["time"]:
            self.need(f"{q.name} empty before the value moved", None)
        self.end(f"a legitimate wait: {q.name} emptied at {last} and the value moved "
                 f"at {moved['time']}; each group waited on is a question of its own (not followed)")
        return None

    # ------------------------------------------------------------ Late: the value moved on, late
    def late(self, q: Late):
        self.start("L1", "what moved it on, and when")
        states = self.data.get("statuslog", {}).get("states", [])
        at = next((n for n, s in enumerate(states) if s["status"] == q.observed), None)
        self.need(f"a statuslog row for {q.observed!r}", at is not None)
        after = self.need(f"a statuslog row after {q.observed!r}", states[at + 1] if at + 1 < len(states) else None)
        since = _when(states[at].get("time_extended") or states[at]["time"])
        moved = _when(after.get("time_extended") or after["time"])
        self.result["window"] = {"start": _iso(states[at].get("time_extended") or states[at]["time"]),
                                 "end": _iso(after.get("time_extended") or after["time"]), "end_kind": "moved", "end_from": "statuslog"}
        self.say(f"{q.observed!r} from {since}; {after['status']!r} at {moved}, "
                 f"{(moved - since).total_seconds() / 3600:.1f} h later")
        symptom = Symptom(subject=q.subject, observed=q.observed, task_id=q.key)
        inv, _ = asyncio.run(_investigate(mapdump.code_map("rerun"), symptom, None, q.key, None, 1, None))
        R.check_blobs(inv, "derive-strategy's readings")
        follow = self.need("the map's answer to what moves the value on", inv.hops[0].strategy.follow_up)
        asked = f"{q.subject}={q.observed}"
        statements = [rs for rs in follow.reader_statements if asked in rs.selects]
        # what moved it on is who wrote the next value at the row's time: the question type Asked's H1 and H2.  The
        # mover need not ask on the observed value (the adder takes a job by its key from a report), and a run's
        # mark need not name it (the adder's is the PandaID, one run for every function of a pass)
        w = self._which_wrote(q.subject, after["status"], q.key, after)
        owner = w.reading.owner
        self.say(f"moved on by {owner}:{w.chosen.line}, in {w.question.describe()}")
        run = w.question
        self.result["moved_by"] = {"owner": owner, "line": w.chosen.line, "run": {
            "log": run.log.name, "machine": run.machine or None, "token": run.token, "first": run.first, "last": run.last}}
        # M2: every mover in the window -- by the statements that select the value, by the writes that require it, and
        # every other writer of another value, which takes a row by its key
        strategy_mod.attach_mover_gaps(inv.hops[0].strategy, R.ROOTS)
        gaps = map_gaps(follow)
        writers = asyncio.run(self._writers_of(q.subject))
        movers = movers_of(follow, writers, q.subject, q.observed)
        owners = list(dict.fromkeys(j.owner for j in writers))
        by_key = by_key_of(writers, movers, q.subject, q.observed)
        refusing = [o for o in owners if o not in movers and o not in by_key and refused_arms(writers, o, q.subject, q.observed)]
        return Missed(q.subject, q.observed, after["status"], since, moved, statements, owner, w.question,
                      movers=movers, paths=paths_of(follow, writers, q.subject, q.observed), by_key=by_key,
                      refusing=refusing, writers=writers, gaps=gaps)

    def missed(self, q: Missed):
        self.start("L2", "why the others passed it over")
        cls = q.subject.split(".")[0]
        self.need(f"the rows file's row of {cls} (its \"classes\" say which row is which class)", self._row_of(cls))
        self.say(f"{len(q.statements)} statement(s) ask rows on {q.subject}={q.observed}, from {q.since} to {q.moved}")
        ordered = []
        kinds: dict[tuple[str, int], str] = {}
        for st in q.statements:
            # the call that moved the value on returned the row: a term the row fails was not in that build
            took = st.owner == q.mover
            kind, said = self._passed_over(st, q, took)
            if took:
                kind = f"took it at {q.moved}"
            kinds[(st.owner, st.line)] = kind
            goes_to = ", ".join(c.split("::")[-1] for c in st.crosses_to)
            if st.result == RESULT_AGGREGATE:
                # an aggregate takes no row: what goes on is a value computed from them
                goes = f"a value to {goes_to}" if goes_to else "a value, kept"
            else:
                goes = f"rows to {goes_to or 'itself'}"
            self.say(f"  {st.owner.split('::')[-1]}@{st.line}: {kind} ({goes})")
            for line in said:
                self.say(f"      {line}")
            if (kind in ("not settled", "refused on a call", "refused when") or took) and st.order_by and st.limits:
                self.say(f"      returns the first rows ({', '.join(st.limits)}) of {st.order_by}")
                ordered.append(st)
        self._movers_in_window(q, kinds)
        self._why_late(q, ordered)
        mine = [st for st in ordered if st.owner == q.mover]
        self.need("the statement that moved the value on returns the first rows of an order", len(mine) == 1)
        return Ordered(mine[0], q.run, q)

    def _read_at_as_of(self, q: Stuck, follow, writers: list, movers: dict) -> list[tuple[str, str, str]]:
        """M2, stuck: each mover's statements read at one moment, from the row's values then -- ``refused`` and the
        rest as L2 says them, ``takes it`` where the statement surely returns the row then.  The moment is when the rows
        file's values were read (``after`` and the subject's row say it, on the clock they name); a file that does not
        say, or says more than one time, is read at the statuslog's as_of, and says so.  A reading, not a drop: a value
        at one moment is not the value all the window, and a mover that rewrites the column it compares refuses the row
        between its passes (TaskCommando's modificationtime; KNOWN_ISSUES.md K17).  ``(owner, statement, verdict)``
        each."""
        as_of = self.data.get("statuslog", {}).get("as_of")
        if not as_of:
            self.say("the movers' statements are not read at as_of: the statuslog says no as_of")
            return []
        at, words, header = self._reading_moment(q.subject, as_of)
        asked = f"{q.subject}={q.observed}"
        mq = Missed(q.subject, q.observed, "", at, at, [rs for rs in follow.reader_statements if asked in rs.selects], "", None,
                    end=words, start=words, at_end=True)
        paths = paths_of(follow, writers, q.subject, q.observed)
        self.say(header)
        out = []
        for owner, why in movers.items():
            rel, func = owner.split("::", 1)
            name = f"{Path(rel).name}::{func}"
            if not paths.get(owner):
                self.say(f"  {name}: {why}: taken by its key, not read here")
                out.append((owner, "", "by key"))
                continue
            for st in paths[owner]:
                kind, said = self._passed_over(st, mq)
                if kind != "refused" and self._statement_true(st, mq):
                    kind = "takes it"
                self.say(f"  {name} <- {st.owner.split('::')[-1]}@{st.line}: {kind} then")
                for line in said:
                    self.say(f"      {line}")
                out.append((owner, f"{st.owner.split('::')[-1]}@{st.line}", kind))
        return out

    def _reading_moment(self, subject: str, as_of: str) -> tuple[datetime.datetime, str, str]:
        """When S1 reads the movers' statements: the one time ``after`` and the values of *subject*'s row say they were
        read, or else the statuslog's *as_of* -- with the words for it, the header that says it, and result.json's
        ``readings_at``."""
        row = self._row_of(subject.split(".")[0])
        after = self._read()[1]
        times = {(v["as_of"], v.get("clock")) for v in row.values() if isinstance(v, dict) and v.get("as_of")}
        times |= {(after["as_of"], after.get("clock"))} if after.get("as_of") else set()
        if len(times) == 1:
            (read, clock), = times
            later = "; after it, so the value may have moved by then" if _when(read) > _when(as_of) else ""
            self.result["readings_at"] = {"at": _iso(read), "clock": clock, "from": "rows"}
            return _when(read), "the row's values were read", (
                f"the movers' statements at {read} ({clock} clock), when the rows file's values were read (the "
                f"statuslog's as_of is {as_of}{later}; a reading, not a drop: KNOWN_ISSUES.md K17):")
        self.result["readings_at"] = {"at": _iso(as_of), "clock": self.data["statuslog"].get("clock"), "from": "statuslog"}
        says = f"says its values were read at {sorted(times)}, not at one time" if times else "does not say when its values were read"
        return _when(as_of), "the statuslog was read", (
            f"the movers' statements at the statuslog's as_of {as_of}; the rows file {says} (a reading, not a drop: "
            f"KNOWN_ISSUES.md K17):")

    def _movers_in_window(self, q: Missed, kinds: dict) -> list[tuple[str, str]]:
        """M2, late: every mover in the window with what the map and the row's values alone say of it --
        ``moved it``; ``write refuses`` (a writer every write of another value of which requires another value of the
        column); ``refused`` (no arm of its own, and every statement that gives it rows refuses the row all the
        window); ``could have`` (a statement that gives it rows is surely true for the row in a span of the window);
        ``by key`` (a writer, which takes a row by its key: its runs in the window are not read here, KNOWN_ISSUES.md
        K15); ``undecided`` (what is missing is said).  ``(owner, verdict)`` for each, in the order printed."""
        whole = (q.since, q.moved)
        guarded = {o for o, why in q.movers.items() if why == GUARDED}
        listed = list(dict.fromkeys([*q.movers, *q.by_key, *q.refusing] + ([q.mover] if q.mover not in q.movers and q.mover not in q.by_key else [])))
        self.say(f"movers in the window ({q.since} .. {q.moved}): {len(listed)}")
        out = []
        entries = []
        for owner in listed:
            rel, func = owner.split("::", 1)
            name = f"{Path(rel).name}::{func}"
            armed = arms_of(q.writers, owner, q.subject, q.observed)
            paths = q.paths.get(owner, [])
            said = []
            spans = {}
            for st in paths:
                where = f"{st.owner.split('::')[-1]}@{st.line}"
                kind = kinds.get((st.owner, st.line), "not read")
                true = self._statement_true(st, q)
                if true:
                    spans[where] = true
                said.append(f"{where} {kind}" + (f", holds for the row {_spans_words(true, whole)}" if true else ""))
            reasons = []
            if made := births_of(q.writers, owner, q.subject, q.observed):
                said.append(f"its writes at {made} make the row (births: {birth_why(owner, q.subject, made)}): not arms"
                            " (KNOWN_ISSUES.md K29)")
                reasons.append({"code": "K29", "text": said[-1]})
            if owner == q.mover:
                verdict = "moved it"
                said.insert(0, f"at {q.moved} (L1)")
            elif owner in q.refusing:
                verdict = "write refuses"
                said = [f"its writes of another value refuse a row holding {q.observed!r}: "
                        + "; ".join(refused_arms(q.writers, owner, q.subject, q.observed))]
                reasons.append({"code": "write-refuses", "text": said[0]})
            elif not armed and owner not in guarded:
                if paths and all(kinds.get((st.owner, st.line)) == "refused" for st in paths):
                    verdict = "refused"
                    reasons.append({"code": "refused", "text": "; ".join(said)})
                else:
                    verdict = "could have" if spans else "undecided"
                    said.append("no arm of its own: it writes, if at all, through what it calls (KNOWN_ISSUES.md K14)")
                    reasons.append({"code": "K14", "text": said[-1]})
            else:
                verdict = "could have" if spans else "by key"
                said.append(f"a writer {sorted(armed)}, which takes a row by its key too: its runs in the window are not read "
                            f"here (KNOWN_ISSUES.md K15)" if armed else "its write requires the value: taken by its key "
                            "(KNOWN_ISSUES.md K15)")
                reasons.append({"code": "K15", "text": said[-1]})
            gaps = q.gaps.get(owner) or []
            if gaps:
                said.append(f"the map could not read all of it ({gap_said(gaps)})")
                reasons.append({"code": "map-gap", "text": said[-1]})
            if verdict not in ("moved it", "write refuses", "refused"):
                # what keeps each statement from a verdict: a value the rows file does not hold, a run-time key
                for st in paths:
                    where = f"{st.owner.split('::')[-1]}@{st.line}"
                    if st.keys:
                        reasons.append({"code": "K18", "text": f"{where} asks rows by a value known at run time ({', '.join(st.keys)}): "
                                                               "never surely true"})
                    lacked = []
                    for subject in sorted(_statement_subjects(st)):
                        if why := self._unanswered_why(subject, q):
                            reasons.append({"code": why[0], "text": f"{where}: {subject} not compared ({why[1]})"})
                            lacked += [subject] if why[0] == "rows-no-row" else []
                    if kinds.get((st.owner, st.line)) != "refused":
                        said += [f"{where}: {note}" for note in self.ask_linked(st, lacked, q)]
            self.say(f"  {name}: {verdict} -- {'; '.join(said) if said else q.movers.get(owner, 'a writer of another value')}")
            out.append((owner, verdict))
            entries.append({"owner": owner, "reached": "select" if paths else "guard" if owner in guarded else "key",
                            "statements": [f"{st.owner.split('::')[-1]}@{st.line}" for st in paths],
                            "verdict": verdict.replace(" ", "-"), "reasons": reasons})
        self.result["movers"] = entries
        left = [o for o, v in out if v not in ("moved it", "write refuses", "refused")]
        self.say(f"not ruled out: {len(left)} mover(s)")
        return out

    def _why_late(self, q: Missed, ordered: list) -> None:
        """Judgement 1: why the mover that moved the value on did it only then.  One whose own statement returns the
        first rows of an order goes on to L3 (lost on order); one whose statement took the row in no such order, or that
        takes the row by its key, stops here, saying what is not read yet."""
        rel, func = q.mover.split("::", 1)
        name = f"{Path(rel).name}::{func}"
        took = [st for st in q.statements if st.owner == q.mover]
        self.need(f"why {name} moved it only at {q.moved}: it takes the row by its key, and its runs in the window are "
                  f"read pass by pass (S2-S4 over the window; build order 2, KNOWN_ISSUES.md K15)", took, "K15")
        if any(st.owner == q.mover for st in ordered):
            return
        spans = _union([span for st in took for span in self._statement_true(st, q)])
        where = ", ".join(f"{st.owner.split('::')[-1]}@{st.line}" for st in took)
        held = f"held for the row {_spans_words(spans, (q.since, q.moved))}" if spans else "held for the row at times not known"
        self.need(f"why {name} took it only at {q.moved}: {where} {held}, and it returns no first rows of an order "
                  f"(not read)", None)

    def _row_of(self, cls: str) -> dict:
        """The rows file's row of the class: the one its ``classes`` says is of it (A1 answers a row of a class)."""
        entity = next((e for e, c in self.data.get("classes", {}).items() if c == cls), None)
        return self._read()[0].get(entity, {}) if entity else {}

    def _read(self) -> tuple[dict, dict]:
        """The rows file's ``rows`` and ``after`` as the steps read them: a value the builder read at ``as_of``, other
        than the row's key, is the row as it was then, and is read as ``after`` is (``held.split``, KNOWN_ISSUES.md
        K22)."""
        return HD.split(self.data, self.args.subject.partition(".")[0], self.args.key)

    def _created(self, found: PT.History) -> str | None:
        """When the asked row was made (*found*'s ``created``), from the rows file's row or its ``after``: the column is
        set once, as the row is made, so the value read at any time is the one it had -- the one column of ``after``
        read whatever the time (KNOWN_ISSUES.md K22).  None where the history names none or the rows file lacks it."""
        if not found.created:
            return None
        cls = found.subject.partition(".")[0]
        value = self._row_of(cls).get(found.created, {}).get("value")
        if value is None:
            value = self._read()[1].get(found.created)
            value = value.get("value") if isinstance(value, dict) else value
        return str(value) if value else None

    def _row_value(self, subject: str, q: Missed | None = None):
        """The rows file's value of ``Class.attr`` for this instance, or None; under *q* at the window's end, its
        ``after`` too (``_after``)."""
        cls, _, attr = subject.partition(".")
        row = self._row_of(cls)
        if attr in row:
            return row[attr]["value"]
        return self._after(subject, q)[1] if q is not None else None

    def _row_holds(self, pair: str, q: Missed | None = None) -> bool:
        subject, _, value = pair.partition("=")
        held = self._row_value(subject, q)
        return held is not None and str(held) == value

    def _after(self, subject: str, q: Missed) -> tuple[bool, object]:
        """``(known, value)`` of *subject* from the rows file's ``after`` -- the asked row's columns as the rows file
        read them -- where that time is the moment read (``q.at_end``: stuck, read when the rows file was); late's
        ``after`` was read after the window and answers nothing.  ``source``, ``as_of`` and ``clock`` say where and
        when, and are no column."""
        cls, _, attr = subject.partition(".")
        after = self._read()[1]
        if not q.at_end or cls != q.subject.partition(".")[0] or attr in ("source", "as_of", "clock") or attr not in after:
            return False, None
        return True, after[attr]

    def _held(self, subject: str, q: Missed) -> tuple[bool, object]:
        """``(known, value)`` of *subject* for the row over the window: the asked subject holds the observed value;
        another, what the rows file holds (``None`` there is SQL's NULL), its ``after`` only at the window's end
        (``_after``); a column it lacks is not known."""
        if subject == q.subject:
            return True, q.observed
        cls, _, attr = subject.partition(".")
        row = self._row_of(cls)
        return (True, row[attr]["value"]) if attr in row else self._after(subject, q)

    def _history_rows(self, subject: str, q: Missed) -> list[tuple[object, datetime.datetime]] | None:
        """``(value, written at)`` of each row of the row's status history the subject's column reads, written
        before the value moved on (the move's own row is the mover's), when the subject is a column of a log that
        keeps the asked class's history (part H: ``jobs_statuslog`` for ``JobSpec.jobStatus``) and the history by
        key carries it; None otherwise."""
        owner, _, attr = subject.partition(".")
        found = PT.history_by_table(owner)
        if found is None or found.subject.partition(".")[0] != q.subject.partition(".")[0]:
            return None
        column = PT.history_column(found, attr)
        if column is None:
            return None
        out = []
        for state in self.data.get("statuslog", {}).get("states", []):
            at = _when(state.get("time_extended") or state["time"])
            if at < q.moved:
                value = _when(state[column]) if column.startswith("time") and state.get(column) else state.get(column)
                out.append((value, at))
        return out

    def _unanswered(self, subject: str, q: Missed) -> str | None:
        """``_unanswered_why``'s words."""
        why = self._unanswered_why(subject, q)
        return why[1] if why else None

    def _unanswered_why(self, subject: str, q: Missed) -> tuple[str, str] | None:
        """Why the row's values cannot answer *subject*: no row of its owner in the rows file and no history it reads,
        a log column the history by key does not carry, or a column the owner's row lacks -- kept under the rows
        file's ``after`` (the moved row's columns written at the move or later, so not what the statement read before
        it) or not fetched at all; None when they can.  Said, not skipped in silence.  ``(code, words)``, the code
        result.json's class: ``rows-no-row``, ``history-no-column``, ``after-only``, ``rows-no-column``."""
        owner, _, attr = subject.partition(".")
        if subject == q.subject:
            return None
        row = self._row_of(owner)
        if row:
            if attr in row or self._after(subject, q)[0]:
                return None
            if owner == q.subject.partition(".")[0] and attr in self._read()[1]:
                return "after-only", "written at the move or later; its row here holds only the value after"
            return "rows-no-column", f"its row here holds no {attr}"
        found = PT.history_by_table(owner)
        if found is None or found.subject.partition(".")[0] != q.subject.partition(".")[0]:
            return "rows-no-row", f"no row of {owner}"
        return None if PT.history_column(found, attr) else ("history-no-column", f"its status history here holds no {attr}")

    def _history_leaf(self, t, rows: list, q: Missed) -> str | None:
        """A term on a joined log, false all the window: no row of the history satisfies it at any moment it exists in
        the window (the join passes where some row does)."""
        never = f"no row of its status history before {q.end}"
        if t.node == "null":
            return f"{never} is NULL" if all(v is not None for v, _ in rows) else None
        if t.node == "not_null" or not t.values or not all(_settled(v) for v in t.values):
            return None
        if t.node == "equal":
            if any(v.base or v.kind == "column" or v.constant == "" for v in t.values):
                return None
            wanted = {v.constant for v in t.values}
            return f"{never} holds one of {sorted(wanted)}" if not any(str(v) in wanted for v, _ in rows) else None
        if t.node == "unequal":
            values = {v.constant for v in t.values}
            if len(values) == 1 and not any(v.base for v in t.values) and all(str(v) in values for v, _ in rows):
                return f"every row of its status history before {q.end} holds {next(iter(values))!r}"
            return None
        whys = []
        for v in t.values:
            if v.base not in UTC_CLOCKS or v.offset_seconds is None:
                return None
            if not all(isinstance(held, datetime.datetime) for held, _ in rows):
                return None
            # a row: col < now + offset holds from turn = col - offset on, col > now + offset until turn; and the row
            # exists from its own writing on
            for held, written in rows:
                turn = held - datetime.timedelta(seconds=v.offset_seconds)
                if (t.op in ("<", "<=") and turn < q.moved) or (t.op in (">", ">=") and max(q.since, written) < turn):
                    return None
            whys.append(f"with {_value_words(v)}: {never} is on its side before {q.moved}")
        return "; ".join(whys) or None

    def _pair_fails(self, subject: str, values: set[str], q: Missed) -> str | None:
        """Why the row cannot hold one of *values* of *subject* (a selects or restricts pair set): it holds another,
        or no row of its status history holds one; None when it can, or nothing says."""
        rows = self._history_rows(subject, q)
        if rows is not None:
            if any(str(v) in values for v, _ in rows):
                return None
            return f"no row of its status history before {q.end} holds one ({sorted({str(v) for v, _ in rows})})"
        known, held = self._held(subject, q)
        if known and held is None:
            return NULL_FAILS
        return f"the row holds {held!r}" if known and str(held) not in values else None

    def _truth(self, node, q: Missed) -> tuple[bool | None, list[Leaf]]:
        """A condition tree for the row all the window: ``False`` -- false for it at every moment of every run, with
        the terms that make it so; ``None`` -- not known false.  Being true never decides anything here (no node
        negates), so it is not told from unknown."""
        if node.node in ("any", "all"):
            found = [self._truth(item, q) for item in node.items]
            if node.node == "all":
                false = [leaves for value, leaves in found if value is False]
                return (False, false[0]) if false else (None, [])
            if found and all(value is False for value, _ in found):
                return False, [leaf for _, leaves in found for leaf in leaves]
            return None, []
        why = self._leaf(node, q)
        return (False, [Leaf(node.subject, node.text, why)]) if why else (None, [])

    def _leaf(self, t, q: Missed) -> str | None:
        """Why the term is false for the row all the window, or None.  ``=``/``IN``: every value settled and the row
        holds none; ``<>``/``NOT IN``: one settled value, which the row holds; a time compared with a UTC clock plus a
        settled offset: true only outside the window; a number: false for every value.  A column the row holds NULL
        fails each of these whatever the values, settled or not: SQL compares NULL with nothing, and a WHERE keeps only
        what is true; an unread term (``other``: an NVL, a CASE) may take a NULL."""
        if t.node == "other":
            return None
        if t.node in ("equal", "unequal", "compare") and self._history_rows(t.subject, q) is None:
            known, held = self._held(t.subject, q)
            if known and held is None:
                return NULL_FAILS
        if not t.settled:
            return None
        rows = self._history_rows(t.subject, q)
        if rows is not None:
            return self._history_leaf(t, rows, q)
        known, held = self._held(t.subject, q)
        if not known:
            return None
        if t.node in ("null", "not_null"):
            return f"the row holds {held!r}" if (held is None) == (t.node == "not_null") else None
        if held is None or not t.values or not all(_settled(v) for v in t.values):
            return None
        if t.node == "equal":
            if any(v.base or v.kind == "column" or v.constant == "" for v in t.values):
                return None
            return None if str(held) in {v.constant for v in t.values} else f"the row holds {held!r}"
        if t.node == "unequal":
            values = {v.constant for v in t.values}
            return f"the row holds {held!r}" if values == {str(held)} and not any(v.base for v in t.values) else None
        whys = [self._compared(t.op, held, v, q) for v in t.values]
        return "; ".join(whys) if all(whys) else None

    @staticmethod
    def _compared(op: str, held, v, q: Missed) -> str | None:
        """``col <op> value`` false for the row all the window, for one value of it: why, or None."""
        if v.base:
            if v.base not in UTC_CLOCKS or v.offset_seconds is None:
                return None
            # col < now + offset  <=>  now > col - offset
            turn = _when(str(held)) - datetime.timedelta(seconds=v.offset_seconds)
            if op in ("<", "<=") and turn > q.moved:
                return f"with {_value_words(v)}: true for the row from {turn}, after {q.end}"
            if op in (">", ">=") and turn < q.since:
                return f"with {_value_words(v)}: true for the row until {turn}, before {q.start}"
            return None
        a, b = _number(held), _number(v.constant)
        if v.kind == "column" or a is None or b is None:
            return None
        true = {"<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b}.get(op)
        return f"with {_value_words(v)}: the row holds {held!r}" if true is False else None

    def _true_spans(self, node, q: Missed) -> list:
        """Where in [since, moved] a condition tree is surely true for the row (M2's "could have"): ``all`` where every
        item is, ``any`` where some item is.  A term not known true holds nowhere -- unknown is never true -- and a term
        on a status history is not read here."""
        if node.node in ("any", "all"):
            found = [self._true_spans(item, q) for item in node.items]
            if node.node == "any":
                return _union([span for spans in found for span in spans])
            out = [(q.since, q.moved)]
            for spans in found:
                out = _intersect(out, spans)
            return out
        return self._true_leaf(node, q)

    def _true_leaf(self, t, q: Missed) -> list:
        whole = [(q.since, q.moved)]
        if t.node == "other" or not t.settled:
            return []
        if t.subject != q.subject and self._history_rows(t.subject, q) is not None:
            return []
        known, held = self._held(t.subject, q)
        if not known:
            return []
        if t.node in ("null", "not_null"):
            return whole if (held is None) == (t.node == "null") else []
        if held is None or not t.values or not all(_settled(v) for v in t.values):
            return []
        if t.node in ("equal", "unequal"):
            if any(v.base or v.kind == "column" or v.constant == "" for v in t.values):
                return []
            inside = str(held) in {v.constant for v in t.values}
            return whole if inside == (t.node == "equal") else []
        out = whole
        for v in t.values:
            out = _intersect(out, self._compared_spans(t.op, held, v, q))
        return out

    @staticmethod
    def _compared_spans(op: str, held, v, q: Missed) -> list:
        """Where in [since, moved] ``col <op> value`` is true for the row, for one value of it (``_compared``'s turn)."""
        whole = [(q.since, q.moved)]
        if v.base:
            if v.base not in UTC_CLOCKS or v.offset_seconds is None:
                return []
            try:
                turn = _when(str(held)) - datetime.timedelta(seconds=v.offset_seconds)
            except ValueError:
                return []
            if op in ("<", "<="):
                return _intersect([(turn, q.moved)], whole)
            if op in (">", ">="):
                return _intersect([(q.since, turn)], whole)
            return []
        a, b = _number(held), _number(v.constant)
        if v.kind == "column" or a is None or b is None:
            return []
        return whole if {"<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b}.get(op) else []

    def _pairs_true(self, restricts: list, refuses: list, q: Missed) -> list:
        """The whole window where the row holds one of every restricted subject's values and none it refuses; none
        where a value is not known."""
        for subject, values in _by_subject(restricts).items():
            known, held = self._held(subject, q)
            if not known or held is None or str(held) not in values:
                return []
        for pair in refuses:
            subject, _, value = pair.partition("=")
            known, held = self._held(subject, q)
            if not known or value == "{}" or (held is not None and str(held) == value):
                return []
        return [(q.since, q.moved)]

    def _statement_true(self, st, q: Missed) -> list:
        """Where in [since, moved] the statement surely returns the row: it holds a value of each subject the map says
        every build restricts and none it refuses, every condition tree is true -- each build's and those some build
        adds, as which build ran is not known -- and so is one running route of calls into it, by that route's own
        reading (a route read as the statement adds nothing); a statement whose every route is not run returns
        nothing, one with no route read has no route to hold.  A statement that asks rows by a value chosen at run
        time (the map's ``keys``: ``WHERE jediTaskID=:jediTaskID``) returns the row only when called with the row's
        value, which is not known here: never surely (KNOWN_ISSUES.md K18)."""
        if st.keys:
            return []
        out = self._pairs_true(st.restricts, st.refuses, q)
        for reading in st.condition_trees:
            out = _intersect(out, self._true_spans(reading.tree, q))
        running = [r for r in st.call_readings if not r.not_run]
        if st.call_readings and not running:
            return []
        if running:
            some = []
            for route in running:
                if route.as_statement:
                    some.append((q.since, q.moved))
                    continue
                spans = self._pairs_true(route.restricts, route.refuses, q)
                for reading in route.condition_trees:
                    spans = _intersect(spans, self._true_spans(reading.tree, q))
                some += spans
            out = _intersect(out, _union(some))
        return out

    def _trees_false(self, trees, q: Missed, said_on: set[str], only=None) -> list[tuple[str, list[Leaf]]]:
        """The trees false for the row all the window, ``(entry, leaves)``, less those whose every false term tests a
        subject already said (the map spells a selected or restricted pair as a term too: said once)."""
        out = []
        for reading in trees:
            if only is not None and not only(reading.entry):
                continue
            value, leaves = self._truth(reading.tree, q)
            if value is False and not all(leaf.subject in said_on for leaf in leaves):
                out.append((reading.entry, leaves))
        return out

    @staticmethod
    def _tree_words(entry: str, leaves: list[Leaf]) -> str:
        if len(leaves) == 1:
            return f"{entry} {leaves[0].why}"
        return f"{entry}: " + "; ".join(f"{leaf.text} {leaf.why}" for leaf in leaves)

    def _route_refuses(self, route, q: Missed) -> list[str]:
        """What refuses the row on one route of calls, read from its record alone: a refused pair it holds, a
        restricted subject whose values lack the row's (the asked one compared too: a call that never asks for the
        observed value did not pass the row over), a condition false for it all the window."""
        said, said_on = [], set()
        for pair in route.refuses:
            if self._row_holds(pair, q):
                said.append(f"refuses {pair}; the row holds it")
                said_on.add(pair.partition("=")[0])
        for subject, values in sorted(_by_subject(route.restricts).items()):
            why = self._pair_fails(subject, values, q)
            if why:
                said.append(f"restricts {subject} to {sorted(values)}; {why}")
                said_on.add(subject)
        said += [self._tree_words(entry, leaves) for entry, leaves in self._trees_false(route.condition_trees, q, said_on)]
        return said

    def _routes(self, st, q: Missed) -> tuple[str, list[str]]:
        """L2's verdict from the routes of calls into *st* (the map's ``call_readings``): ``refused`` when the map
        places every call (``callers_complete``) and every route that runs the statement refuses the row;
        ``refused on a call`` for the routes that do; else nothing.  A route reading as the statement refuses only
        what the statement does, and that was said before the routes are asked."""
        lines, running, refusing = [], 0, 0
        for route in st.call_readings:
            if route.not_run:
                continue
            running += 1
            why = [] if route.as_statement else self._route_refuses(route, q)
            if not why:
                continue
            refusing += 1
            constants = ", ".join(f"{k}={v}" for k, v in sorted(route.constants.items()))
            via = f" via {', '.join(route.chain[:-1])}" if len(route.chain) > 1 else ""
            lines.append(f"called from {route.chain[-1]}{via} ({constants}): " + "; ".join(why))
        if not refusing:
            return "not settled", []
        if st.callers_complete and refusing == running:
            return "refused", [f"every call refuses it: {running} route(s), and the map places every call", *lines]
        stop = "; ".join(st.callers_stop) or "the map does not say it places every call"
        return "refused on a call", [*lines, f"{refusing} of {running} route(s) refuse it; {stop}"]

    def _passed_over(self, st, q: Missed, took: bool = False) -> tuple[str, list[str]]:
        """What the map lets one say about *st* and this row: ``refused`` -- whatever the build, the row cannot come
        back; ``refused on a call`` -- on the calls named; ``refused when`` -- on the builds the map's tests pick;
        ``if required`` -- a term the row fails, where the map does not say which builds refuse the row on it; else
        ``not settled``.

        ``selects`` and ``excludes`` gather every build; ``conditions`` holds the terms every build requires, and
        ``refuses`` / ``restricts`` the pairs every build decides (the latter, per subject, the values a row must hold).  ``added_under`` (the map's M1) lists the terms some
        build lacks, each place with the tests that add it and whether its term decides the pair; None is a map that
        did not compute it, and reads as before it.  *took*: *st* is the call that moved the value on, so no build of
        it refused the row.

        ``condition_trees`` (the map's S2) hold each conditions entry and added_under term as a tree the row is
        evaluated against; ``call_readings`` (M2) each route of calls into the function, read in full -- asked after
        the statement's own fields, and only when they do not refuse the row already."""
        self.need("the map's condition trees and routes of calls (R2, 2079b2a and later)",
                  hasattr(st, "condition_trees") and hasattr(st, "call_readings"))
        under = _added_under(st)
        for pair in st.refuses:
            if self._row_holds(pair, q):
                return "refused", [f"refuses {pair}; the row holds it"]
        kind, said = "not settled", []
        said_on: set[str] = set()
        restricted = _by_subject(getattr(st, "restricts", None) or [])

        def fails(text: str, field: str, terms: list[str], decided: bool) -> None:
            nonlocal kind
            verdict, why = _builds(under, field, terms, took, decided)
            said.append(f"{text} -- {why}")
            kind = min(kind, verdict, key=KINDS.index)

        # an exclude the row holds that refuses does not hold (every build deciding it): which builds refuse the row
        if under is not None:
            for pair in sorted(set(st.excludes) - set(st.refuses)):
                if self._row_holds(pair, q):
                    fails(f"refuses {pair}; the row holds it", "excludes", [pair], False)
                    said_on.add(pair.partition("=")[0])
        for subject, values in sorted(_by_subject(st.selects).items()):
            why = self._pair_fails(subject, values, q) if subject != q.subject else None
            if why:
                fails(f"asks {subject} in {sorted(values)}; {why}", "selects",
                      [f"{subject}={v}" for v in sorted(values)], subject in restricted)
                said_on.add(subject)
        # every build's entries first, then the terms some build lacks (they have left conditions)
        trees = sorted(st.condition_trees, key=lambda r: (r.entry not in st.conditions,
                                                         st.conditions.index(r.entry) if r.entry in st.conditions else 0, r.entry))
        for entry, leaves in self._trees_false(trees, q, set()):
            verdict, _ = _builds(under, "conditions", [entry], took, entry in st.conditions)
            if all(leaf.subject in said_on for leaf in leaves) and KINDS.index(verdict) >= KINDS.index(kind):
                continue  # said already, as a selected or excluded pair, and no stronger as a condition
            fails(self._tree_words(entry, leaves), "conditions", [entry], entry in st.conditions)
        # what neither the row nor its history answers is said, not skipped; the rows of a class the file has none of
        # are asked where they could change the verdict
        unanswered = {x: why for x in sorted(_statement_subjects(st)) if (why := self._unanswered_why(x, q))}
        if unanswered:
            said.append("not compared: " + "; ".join(f"{x} ({why})" for x, (_, why) in unanswered.items()))
        if kind == "refused":
            return kind, said
        by_call, calls = self._routes(st, q)
        kind = min(kind, by_call, key=KINDS.index)
        if kind != "refused":
            said += self.ask_linked(st, [x for x, (code, _) in unanswered.items() if code == "rows-no-row"], q)
        return kind, calls + said

    def ordered(self, q: Ordered):
        self.start("L3", "what it lost to")
        st = q.statement
        where = f"{st.owner.split('::')[-1]}@{st.line}"
        self.say(f"{where} returns the first rows ({', '.join(st.limits)}) of one of {len(st.order_by)} order(s)")
        self.need("the map's prints of a statement and their cut (printed_by, sqltext.cut and unfilled; R2, 2079b2a and later)",
                  hasattr(sqltext, "cut") and hasattr(sqltext, "unfilled") and hasattr(st, "printed_by"))
        self.need(f"a print of {where}'s text and binds in its function (the map's printed_by)", st.printed_by)
        # the call's own SQL: the records of the print the map names, cut into the text and the binds it ran with,
        # the text read as the map reads statements; the query an order of the statement's sorted
        rel, func = st.owner.split("::", 1)
        fn = R.function(rel, func, within=[st.line])
        used = {}
        for printed in st.printed_by:
            row = self.need(f"{func}'s print at {printed.line} among its rows", next((r for r in fn.rows if r.line == printed.line), None))
            for r in q.run.records:
                if not row.matches(r, self.args.patterns)[0]:
                    continue
                got = sqltext.cut(r.text, printed.lead, printed.gap)
                if got is None:
                    continue
                text, binds = got
                read = sqltext.printed(text)
                for order in st.order_by:
                    for query in read.queries:
                        if query.order and _same_order(sqltext.order_items(order), query.order):
                            used.setdefault(order, (r, read, query, binds))
        for order, (r, *_) in used.items():
            self.say(f"  the call that moved it on ordered by {order!r} ({r.span}, {r.stamp})")
        self.need("one order the call that moved it on used, in its SQL", len(used) == 1)
        (took, (record, read, query, binds)), = used.items()
        others = [order for order in st.order_by if order != took]
        self.need("one other order the statement can be built with", len(others) == 1)
        other = others[0]
        cls = self.args.subject.split(".")[0]
        row = {k.lower(): v["value"] for k, v in self._row_of(cls).items()}
        key_col = next((k for k, v in row.items() if str(v) == self.args.key), None)
        # the table, the WHERE and the binds of the query that sorted
        table = query.spelled[0] if len(query.spelled) == 1 else None
        unfilled = sorted({b for t in query.where for b in sqltext.unfilled(t.text, binds)})
        self.need(f"the run's value of every bind of the call's WHERE (not written in: {unfilled})", not unfilled)
        filled = [sqltext.filled(t.text, binds) for t in query.where]
        cut = " AND ".join(filled) if filled else None
        self.need("the table, WHERE and binds of the call that moved it on"
                  + (f" ({read.failure})" if read.failure else ""), table and cut and key_col)
        self.say(f"  the call's WHERE: {cut}")
        conditions, unsaid = _db_read_where(query, binds)
        # a blanked item the map's order_holes says ranks by a column of the row (getJobs' (CASE ...) by gshare):
        # grouped by, its ranking computed at run time -- a row of another value is neither ahead nor behind here
        holes = {h.item: h for h in getattr(st, "order_holes", None) or [] if h.order == other}
        spelled, blank, ranked = [], [], []
        for i, item in enumerate(sqltext.order_items(other)):
            hole = holes.get(i)
            if not item.blanked and item.columns == (item.expression,):
                spelled.append((item.expression.split(".")[-1].lower(), "desc" if item.descending else "asc"))
            elif hole and hole.subjects and all(x.partition(".")[0] == cls for x in hole.subjects):
                ranked += [(item.expression, x.partition(".")[2].lower(), hole.computed_at) for x in hole.subjects]
            else:
                blank.append(item.expression)
        sorts = [(c, "ranked") for _, c, _ in ranked] + spelled
        cols = [(c, d) for c, d in sorts if c != key_col]
        self.say(f"  under {other!r} the row sorts by "
                 + "; ".join(f"{c} {'(ranked at run time)' if d == 'ranked' else d} -> {row.get(c, '?')}" for c, d in sorts)
                 + (f"; the map leaves {blank} blank" if blank else ""))
        self.need("a column of the other order to group the rows by", cols)
        history = self.need(f"a history of {self.args.subject} to ask the window of", PT.history(self.args.subject))
        # the call's terms the history can filter on, by the columns it says it holds
        terms = [f for t, f in zip(query.where, filled, strict=True)
                 if t.columns and all(c.split(".")[-1].lower() in history.held for c in t.columns)]
        miss = q.q
        questions = {
            f"{where} groups now": PT.GroupsNow(where, table, cut, cols, key_col, blank, ranked, None if unsaid else conditions),
            f"{where} left {miss.observed!r} in the window, by the value it went to": PT.MovedOn(
                history, miss.observed, miss.since, miss.moved, terms, cols, key_col
            ),
        }
        asks = {ask: PT.draw(pq) for ask, pq in questions.items()}
        answers = self.data.get("answers", {})
        for ask, sql in asks.items():
            self.say(f"-- {ask}{' (answered in the rows file)' if ask in answers else ''}")
            for line in sql.splitlines():
                self.say(f"   {line}")
            if ask not in answers:
                form = PT.db_read(questions[ask])
                for text, why in unsaid if form else []:
                    self.say(f"   -- db_read takes no {text!r} ({why}): its form keeps the WHERE as text")
                self.ask_rows({"kind": "sql", "name": ask, "sql": sql, "binds": {}} | ({"db_read": form} if form else {}),
                              f"L3's answer {ask!r}, not in the rows file")
        self.need(f"the answers to {list(asks)} in the rows file", all(ask in answers for ask in asks))
        now_ask, window_ask = asks
        now, window = answers[now_ask], answers[window_ask]

        def holding(ask: str, answer: dict) -> None:
            """An answer does not carry the SQL it answers: it must at least hold every column L3 groups by."""
            held = {k.lower() for r in answer["rows"] for k in r}
            lacking = [c for c, _ in cols if c not in held]
            self.need(f"the answer to {ask!r} holding the column(s) {lacking} (it holds {sorted(held)}; "
                      f"run the SQL above)", not lacking)

        def number(value):
            try:
                return float(value)
            except (TypeError, ValueError):
                return value

        def other_rank(r: dict) -> bool:
            """Another value of a ranked column: placed by the ranking the run computes, not known here."""
            return any(d == "ranked" and str(r.get(c)) != str(row.get(c)) for c, d in cols)

        def ahead(r: dict) -> bool:
            if other_rank(r):
                return False
            for c, d in cols:
                a, b = number(r.get(c)), number(row.get(c))
                if d != "ranked" and a != b:
                    return a > b if d == "desc" else a < b
            return False

        def same(r: dict) -> bool:
            return all(number(r.get(c)) == number(row.get(c)) for c, _ in cols)

        holding(now_ask, now)
        rows_now = [{k.lower(): v for k, v in r.items()} for r in now["rows"]]
        n_ahead = sum(int(r["n"]) for r in rows_now if ahead(r))
        n_same = sum(int(r["n"]) for r in rows_now if same(r))
        n_other = sum(int(r["n"]) for r in rows_now if other_rank(r))
        names = [c for c, d in cols if d == "ranked"]
        mine = f" of its {', '.join(f'{c} {row.get(c)!r}' for c in names)}" if names else ""
        others = (f", {n_other} of another {'/'.join(names)}, placed by the ranking computed at run time"
                  f" ({'; '.join(at for *_, at in ranked)})") if names else ""
        self.say(f"  now: {n_ahead} row(s){mine} sort ahead of it, {n_same} in its own group{others} ({now['source']})")
        holding(window_ask, window)
        rows_win = [{k.lower(): v for k, v in r.items()} for r in window["rows"]]
        self.say(f"  left {miss.observed!r} in the window, by the value it went to ({window['source']}):")

        def place(r: dict) -> list:
            return [str(r.get(c)) if d == "ranked" else number(r.get(c)) for c, d in cols]

        for nxt in sorted({str(r.get("next_value")) for r in rows_win}):
            for r in sorted((r for r in rows_win if str(r.get("next_value")) == nxt), key=place, reverse=True):
                mark = "its group" if same(r) else "other" if other_rank(r) else "ahead" if ahead(r) else "behind"
                self.say(f"    {mark:9} {' '.join(f'{k}={v}' for k, v in r.items())}")
        own = [r for r in rows_win if same(r)]
        self.need(f"its own group among the rows that left {miss.observed!r} in the window", own)
        left = "; ".join(f"{r.get('n')} to {r.get('next_value')!s} after {r.get('min_h')}-{r.get('max_h')} h" for r in own)
        self.end(f"lost on order: {where} returns the first rows of {other!r} or of {took!r}; under the first, "
                 f"{n_ahead} row(s){mine} at the call's WHERE sort ahead of it now; its group left {miss.observed!r}: {left} "
                 f"(this row to {miss.next_value!r}); the call that took it used {took!r}")
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--observed", required=True)
    ap.add_argument("--key", required=True)
    door = ap.add_mutually_exclusive_group(required=True)
    door.add_argument("--log", action="append", help="a log file handed in (the door over files)")
    door.add_argument("--records", action="append", help="a file of the recording step's answers (the door over them)")
    ap.add_argument("--asks", default="", help="where the asks no record answers are written (default: --out + .asks.jsonl)")
    ap.add_argument("--rows-asks", default="", help="where the asks of rows the rows file does not have are written (none unless given)")
    ap.add_argument("--rows", required=True)
    ap.add_argument("--hand", default="")
    ap.add_argument("--patterns", default="outside", choices=R.PATTERN_MODES)
    ap.add_argument("--decide", default="positive", choices=("positive", "strict"))
    ap.add_argument("--out", default="")
    ap.add_argument("--result", default="", help="where the structured result is written (default: --out + .result.json)")
    ap.add_argument("--stuck", action="store_true", help="ask why nothing moved the observed value on")
    ap.add_argument("--late", action="store_true", help="ask why the observed value was moved on so late")
    args = ap.parse_args()
    args.asks = args.asks or (f"{args.out}.asks.jsonl" if args.out else "asks.jsonl")
    args.result = args.result or (f"{args.out}.result.json" if args.out else "")
    # a run that does not finish leaves none of its files: a reader never takes an earlier run's for this one's
    for path in (args.out, args.asks, args.result, args.rows_asks):
        if path:
            Path(path).unlink(missing_ok=True)
    mapdump.install_from_env()
    r = Rerun(args)
    try:
        r.run()
        r.say()
        for line in A.summary():
            r.say(line)
        r.say(f"== reached the end; HAND used {len(r.hand_used)} time(s) {r.hand_used}")
        ends = [line for _, line in r.ends]
        outcome = {"kind": "end", "step": r.ends[-1][0] if r.ends else r.hop, "text": "\n".join(ends) or None, "code": None}
    except Stop as stop:
        r.say()
        for line in A.summary():
            r.say(line)
        r.say(f"STOP {stop}; HAND used {len(r.hand_used)} time(s) {r.hand_used}")
        # rows asked and no other rule naming the stop: answering them may move it -- not "asks", which the harness
        # records the logs for
        outcome = {"kind": "stop", "step": r.hop, "text": r.lines[-1], "code": stop.code or ("rows" if r.rows_asks else None)}
    r.stamp("end")
    if args.out:
        _write(args.out, "\n".join(r.lines) + "\n")
    asks = {}
    if r.unanswered:
        # every ask once, for the recording step to fetch together (analyzer-handover/stage2-record-format.md)
        asks = {ask.key: {"ask": ask.canonical(), "step": hop, "why": why} for hop, ask, why in r.unanswered}
        _write(args.asks, "".join(json.dumps(a) + "\n" for a in asks.values()))
    if r.rows_asks and args.rows_asks:
        _write(args.rows_asks, "".join(json.dumps(a) + "\n" for a in r.rows_asks.values()))
    if args.result:
        outcome |= {"hand_used": r.hand_used, "branches": r.branches or None}
        _write(args.result, json.dumps(r.result_of(outcome, asks), indent=1) + "\n")


if __name__ == "__main__":
    main()
