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
                                        the pass made false; the arm with fewest is the nearest (rule 21)
``Input``      S4  where the guard's name was bound  ``trace.walk`` from the guard's ``if``, the binding
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
import datetime
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from bamboo.codemap import strategy as strategy_mod
from bamboo.codemap import trace
from bamboo.codemap.calls import Types
from bamboo.codemap.lookup import CodeMap
from bamboo.codemap.models import Symptom, required_values
from bamboo.codemap.panda import pathcond, sqltext
from bamboo.codemap.panda import sql as sql_mod
from bamboo.codemap.panda.provenance import classify, facade_calls
from bamboo.codemap.panda.recognizers.sqlwrite import _flag_parameters
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
from bamboo.querier.records import CORPUS, FileLogs, Logs, Unrecorded, Window, runs_for
from bamboo.querier.which_ran import Candidate, class_of, judge, stable_holes
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


def proxy_reading(method: str, call: ast.Call) -> tuple[list[str], dict[str, object], list[str], list[str], list[str]] | None:
    """``(parameters, constants, the statement's tables, the tables this call reads, the proxy methods it reaches
    whose statements are not read)`` for a call the walk stopped at as the database's, or None where no facade has
    *method*.

    The method is the one the walk's stop names, and what it asks the database is what the facades' method of the
    name calls on the proxy it borrows (``provenance.facade_calls``, as ``trace._Walk.database_asked`` reads it):
    ``peekJobs`` reads what ``peekJob`` does.  The call is bound to the facade's parameters, with the
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
    if not made:
        return None
    params = [a.arg for a in (*made[0].method.args.posonlyargs, *made[0].method.args.args)][1:]
    said: dict[str, object] = {}
    statement: list[str] = []
    at_call: list[str] = []
    unread: list[str] = []

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
        if not sql_mod.executions(one.proxy_method) and one.proxy_method.name not in unread:
            unread.append(one.proxy_method.name)
    return params, said, statement, at_call, unread


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


def kept_without_arms(found: bool, reached: list[str], cut: list[str], hands: bool, gapped: bool) -> list[str]:
    """Why S1 keeps a mover with no arm of its own: it moves the value, if at all, through what it calls or the rows it
    hands on.  None is dropped: ``called_from`` holds two hops, so a write further down is not seen, and "not listed"
    is not "does not reach" (bamboo-85; tmaeno, 2026-10-14).  Dropping one is the map's reach's to say, with its holes
    (``reach_said``); these are the reasons of a mover it keeps, or where the dump has no reach (KNOWN_ISSUES.md K14)."""
    return [why for why, holds in (
        ("its def is not found", not found),
        (f"it calls into {reached}", reached),
        (f"it may call into {cut}, whose callers the map lists only in part", cut),
        ("it hands its rows on", hands),
        ("the map could not read all of it", gapped),
    ) if holds] or ["no call it makes reaches a writer of another value within the two hops the map lists, "
                    "and a write further down is not seen (KNOWN_ISSUES.md K14)"]


def reach_said(reach: dict | None, gapped: bool) -> tuple[bool, str | None]:
    """``(dropped, line)``: whether S1 drops a mover with no arm of its own by what the map's reach says of it
    (``Rerun._reach_of``), and the line saying so.  Dropped only where it reaches no write of another value and leaves
    no call on the way open (the map's ``[]``, never its ``None``), its own defs hold no arm the junctions place there,
    and the map read all of it -- a gap may hide a write the reach has no junction for (build2-design.md 6, R1).  No
    line where the map was not asked."""
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
    if gapped:
        return False, "reach: reaches no write of another value and leaves nothing open, but the map could not read all of it: kept"
    return True, (f"no arm of its own, and reaches no write of another value with nothing left open (closure "
                  f"{reach['closure']}; assuming {sorted(reach['assumptions'])}): dropped")


#: judgement 3's third premise, which nothing the querier reads says yet (KNOWN_ISSUES.md K19)
PRINTS_NOT_KNOWN = ("not known yet: the prints every path to an arm passes are the map's to say (asked 2026-10-16), and the "
                    "asks search by the question's key, which a run need not print (KNOWN_ISSUES.md K19)")


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
    ``row_precondition``, read as the map reads a guard (``required_values``); a term it cannot read keeps the arm."""
    column = subject.partition(".")[2].lower()
    found: dict[int, Arm] = {}
    for j in writers:
        if j.owner != owner:
            continue
        for b in j.branches:
            if b.line is None or b.outcome == observed:
                continue
            required = [rv for term in b.row_precondition or [] if (rv := required_values(term)) and rv[0] == column]
            if any(observed not in values for _, values in required):
                continue
            arm = found.setdefault(b.line, Arm(b.line, [], list(b.path_condition)))
            arm.outcomes.append(b.outcome)
    return found


def _reached(follow, writers: list, subject: str, observed: str):
    """``(owner, why, statement)`` for each way the map gives a function rows of *subject* holding *observed*, in the
    order S1 asks them; *statement* is None for a writer whose write requires the value.  A statement that selects
    the value -- ``reader_statements`` also holds an UPDATE-only owner's statements of other values, which give none --
    hands its rows to its owner when an arm of the owner's lies in the statement's ``region``, the span the map reads
    where the rows go next, and to the writers among what the owner calls there (``crosses_to``); a statement whose
    owner has no arm there hands them to what it calls there, or keeps them (a mover with no arm of its own:
    KNOWN_ISSUES.md K14).  Then each writer whose write requires the value (``guarded_by``).  A writer that takes a row
    by its key alone is in neither (KNOWN_ISSUES.md K16)."""
    asked = f"{subject}={observed}"
    writing = {j.owner for j in writers if arms_of(writers, j.owner, subject, observed)}
    for rs in follow.reader_statements:
        if asked not in rs.selects:
            continue
        reader = f"{rs.owner.split('::')[-1]}@{rs.line}"
        lo, hi = (list(rs.region) or [rs.line, rs.line])[:2]
        if any(lo <= line <= hi for line in arms_of(writers, rs.owner, subject, observed)):
            yield rs.owner, f"{reader} selects the value", rs
            for callee in rs.crosses_to:
                if callee in writing:
                    yield callee, f"{reader} hands the rows it selects to it", rs
            continue
        for owner in rs.crosses_to or [rs.owner]:
            yield owner, f"{reader} hands the rows it selects to it" if rs.crosses_to else f"{reader} selects the value", rs
    for g in follow.guarded_by:
        yield g.owner, "its write requires the value", None


def movers_of(follow, writers: list, subject: str, observed: str) -> dict[str, str]:
    """The functions that can move *subject* on from *observed*, each with the first way the map gives it rows
    (``_reached``)."""
    found: dict[str, str] = {}
    for owner, why, _ in _reached(follow, writers, subject, observed):
        found.setdefault(owner, why)
    return found


def paths_of(follow, writers: list, subject: str, observed: str) -> dict[str, list]:
    """Each mover's selecting statements whose rows reach it (``_reached``); none for a writer whose write requires the
    value and no statement gives it rows."""
    found: dict[str, list] = {}
    for owner, _, rs in _reached(follow, writers, subject, observed):
        paths = found.setdefault(owner, [])
        if rs is not None and rs not in paths:
            paths.append(rs)
    return found


def refused_arms(writers: list, owner: str, subject: str, observed: str) -> list[str]:
    """*owner*'s writes of another value that ``arms_of`` drops because each requires the row to hold other values of
    the subject's column: ``line writes outcome only on a row holding column in [values]``."""
    column = subject.partition(".")[2].lower()
    out = []
    for j in writers:
        if j.owner != owner:
            continue
        for b in j.branches:
            if b.line is None or b.outcome == observed:
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


class Stop(Exception):
    """A step that cannot go on.  *code* names its class where a rule decides it (``asks``: asks of the logs the records
    do not answer; a KNOWN_ISSUES.md number), for result.json's ``outcome.code``; None where none is named yet."""

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
    """Which way each pass of a mover went: its runs of the key in *logs*, over *asking*'s window."""

    owner: str
    logs: list[str]
    arms: list[Arm]
    key: str
    asking: W.Asking


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

    @property
    def reading(self):
        c = self.chosen.candidate
        return next(r for r in self.readings if r.file == c.rel and r.owner.endswith("::" + c.func))


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
        mapdump.require("rerun")  # _investigate and _writers_of open their own graph clients
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
            self.say(f"WARNING: {owner.split('::')[-1]}'s rows land in different files {split}; read in one, the others' read as missing")

    def base_env(self, file_row: dict | None = None, at: str | None = None) -> E.Env:
        """The row values a step reads, for a run that began *at*: a value read at another time than the run's is not
        among them, and ``Env.unread`` says why (``held.at_run``, KNOWN_ISSUES.md K22)."""
        sp = self.data["spellings"]
        env = E.Env(rows={s: entity for entity, spellings in sp.items() if entity != "about" for s in spellings})
        rows, env.unread = HD.at_run(self.data, self.args.subject.partition(".")[0], self.args.key, at)
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

    def runs(self, key: str, names: list[str], asking: W.Asking) -> dict[str, list]:
        """The key's runs in each of the logs *names* the door can be asked for, over *asking*'s window and machine.

        Every log is asked before a stop, so that a door short of answers names all the step's asks at once.
        """
        out, unanswered, _ = self._ask(key, names, asking)
        self._stop_unanswered(unanswered)
        return out

    def _ask(self, key: str, names: list[str], asking: W.Asking) -> tuple[dict[str, list], list, dict[str, list[str]]]:
        """``runs()`` without the stop: the runs found, the asks the door had no answer for, and what of the window each
        log was not searched for (``Log.holes``)."""
        window = self.window(asking)
        out, unanswered, holes = {}, [], {}
        for name in self.logs.names(names):
            try:
                log = self.logs.log(name, key, window, machine=asking.machine)
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

    def window(self, asking: W.Asking) -> Window | None:
        """*asking*'s window; where it has none, a stop if the door needs one (the door over files reads none)."""
        if not self.logs.needs_window:
            return asking.window
        window = self.need(f"a window to ask the logs over: {asking.why}", asking.window)
        if (self.hop, window) not in self.windows_said:
            self.windows_said.add((self.hop, window))
            self.say(f"window: {window[0]} .. {window[1]}, {asking.why}")
        return window

    def frames_of(self, run) -> dict[str, set[int]]:
        out: dict[str, set[int]] = {}
        for r in run.records:
            for ch in r.chains():
                for f in ch.frames:
                    if f.in_corpus and f.same_source and R.functions(f.rel, f.func):
                        out.setdefault(f"{f.rel}::{f.func}", set()).add(f.line)
        return out

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
        opened in order, every input with its sha256, the asks file, and what the steps found -- the window, the mover
        that moved the value (late), every mover with its verdict and, in S1, judgement 3's premises (``_premises``, added
        in build order 2a) -- with times in UTC, ISO 8601 with ``Z``."""
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
            "window": self.result.get("window"),
            "values_read_at": _read_at(self.data),
            "readings_at": self.result.get("readings_at"),
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
        # the run was chosen by its mark before the function was known; a mark that is
        # the key can hold several passes of it (rule 20), and then this run is not one
        if len(starts := R.entries(site_fn, question, self.mode)) > 1:
            self.say(f"WARNING: the run holds {len(starts)} passes of {site_fn.name} (rule 20); the steps below read them as one")
        site = R.statement_at(site_fn.node, chosen.line)
        reading = w.reading
        _, after = R.must_rows(site_fn.node, site)
        sn = R.seen(site_fn, question, self.mode)
        guard_record = min((i for s in after for i, _, _ in sn.hits.get(s.lineno, ())), default=len(question.records))

        # H3
        self.start("H3", "what its guard reads")
        guards = pathcond.path_condition(site)
        self.say(f"guards of {site_fn.name}:{chosen.line}: {guards}")
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
        method = self.need("a method the guard reads through", methods[0][0] if methods else None)
        return SetBy(method, reading, question, guard_record, env)

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
        symptom = Symptom(subject=subject, observed=value, task_id=key)
        inv, _ = asyncio.run(_investigate("panda", None, symptom, None, key, None, 1, None))
        R.check_blobs(inv, "derive-strategy's readings")
        readings = inv.hops[0].strategy.readings
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
        else:
            if state is None:
                states = self.data["statuslog"]["states"]
                state = self.need(f"a statuslog time for {value}", next((s for s in states if s["status"] == value), None))
                if isinstance(state, str):  # a time from --hand
                    state = {"status": value, "time": state}
            asking = W.at_transition(state)
        lo, hi = asking.window
        when = hi if lo == hi else f"{lo}..{hi}"
        cands = [Candidate(r.file, r.owner.split("::")[-1], r.lines, self.files_of(r.owner, r.log_files, r.lines)) for r in readings]
        runs = self.runs(key, list(dict.fromkeys(f for c in cands for f in c.log_files)), asking)
        at = {name: held for name, rs in runs.items() if (held := [r for r in rs if r.first[:19] <= hi and r.last[:19] >= lo])}
        self.need(f"a run of {key} at {when}", at)
        if len(at) > 1:
            # the row's second holds runs in more than one log, and the machines' clocks are not known finer than a
            # second against the database's: no run is the question's until the judge picks the candidate that ran
            # (Fr1, 2026-10-14).  Taking the first would take the asker's order of the logs, not the case's
            question = None
            self.say(f"runs of {key} at {when} in {len(at)} logs ({', '.join(at)}): each candidate is judged on its own run")
        elif lo != hi and len(only := next(iter(at.values()))) > 1:
            # a window, not the row's second, can hold several passes of the key in one log: the same holds
            question = None
            self.say(f"{len(only)} runs of {key} at {when} in {next(iter(at))}: each candidate is judged on its own run")
        else:
            (question, *_), = at.values()
            self.say(f"the run: {question.describe()}")
        at_hand = [question] if question is not None else [r for rs in at.values() for r in rs]
        for r, c in zip(readings, cands, strict=False):
            if any(name in c.log_files for name in at):
                self.warn_split(r.owner, r.lines)
        env = self.base_env(at=state["time"] if state else None)
        seens = [R.seen(R.function(c.rel, c.func, c.lines), run, self.mode)
                 for c in cands for run in at_hand if run.log.name in c.log_files]
        stable_holes(seens, env)
        j = judge(
            cands, runs, mode=self.mode, decide=self.decide, window=(lo, hi),
            subject=attr, observed=value, question_run=question, env=env, called_at=self._called_at,
        )
        self.say(j.report(question))
        chosen = self.pick("which candidate ran", j)
        if question is None:
            question = self.need(f"the run of {chosen.candidate.func} at {when}", j.runs.get(chosen.candidate.owner))
            self.say(f"the run: {question.describe()}")
        self.question = question
        return Wrote(readings, question, cands, env, chosen)

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
        frames = self.frames_of(question)
        j = judge(sites, {question.log.name: [question]}, mode=self.mode, decide=self.decide, question_run=question, raised_at=frames, env=env)
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
        return Missing(obj, inner)

    # ------------------------------------------------------------ Missing: who makes it
    def missing(self, q: Missing):
        # H6
        self.start("H6", "who makes it")
        obj, inner = q.obj, q.inner
        states = self.data["statuslog"]["states"]
        receivers = O.instances_in(inner.rel)
        receiver = self.need(f"the client {inner.rel} builds", receivers[0] if receivers else None)
        calls = [x for x in O.calls_through(receiver) if x.verb == "make"]
        self.say(f"{receiver}: {len(calls)} make call(s)")
        # the window is the Datasets row's creationdate, which only the server's setupper inserts: a make call in
        # JEDI's code is not anchored by it (KNOWN_ISSUES.md K6)
        jedi =[f"{Path(x.rel).name}:{x.line}" for x in calls if x.rel.startswith("pandajedi/")]
        asking = (
            W.Asking(None, None, f"the make call(s) {jedi} run in JEDI, which inserts no Datasets row to time them by")
            if jedi
            else W.around_creation(self.data.get("datasets"), obj, states[0]["time"] if states else "")
        )
        logs = list(dict.fromkeys(f for x in calls for f in self.files_of(f"{x.rel}::{x.func}", [], [x.line])))
        obj_runs = self.runs(obj, logs, asking)
        makers = []
        for x in calls:
            f2 = R.function(x.rel, x.func, [x.line])
            for rs in obj_runs.values():
                for r2 in rs:
                    sn2 = R.seen(f2, r2, self.mode)
                    hits = [(row.line, r2.records[i].span) for row in f2.rows for i, _, _ in sn2.hits.get(row.line, ()) if obj in r2.records[i].text]
                    if hits:
                        makers.append((x, f2, r2, hits))
        for x, _f2, r2, hits in makers:
            self.say(f"  {Path(x.rel).name}:{x.line} {x.func} {x.method}({x.first_arg})  printed the name at {hits}  run {r2.describe()}")
        self.need("a make call whose function printed the object's name", makers)
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
            used |= {k: f"unknown: {why}" for k, why in env7.unread_in(ast.unparse(i.test)).items()}
            self.say(f"  implicit else of {i.lineno}: test {ast.unparse(i.test)!r} is {value!r} {used}" + (" -> it did not run" if value is True else ""))
            if value is not True:
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
                    bound = {s.name for s in side.steps}
                    loose = [n.id for n in ast.walk(ast.parse(t.step.value, mode="eval")) if isinstance(n, ast.Name) and n.id not in bound and n.id != "self"]
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
        walked = trace.walk(R.ROOTS, file=fn.rel, owner=fn.owner, lines=[frame.line], classify=classify, reaching=False)
        steps = walked.steps
        self.say(f"walk from {fn.name}:{frame.line}: {len(steps)} step(s)" + (f"; {walked.note}" if walked.note else ""))
        ran = [(f"{f.rel}::{f.func}", f.line) for f in ch.frames if f.in_corpus]
        walk = G.Walk(steps, env, self._collections(env), self.data, ran)
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
        why = next(iter(q.env.unread_in(q.expr).values()), "the rows file does not have it")
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
        inv, _ = asyncio.run(_investigate("panda", None, symptom, None, q.key, None, 1, None))
        R.check_blobs(inv, "derive-strategy's readings")
        strategy = inv.hops[0].strategy
        follow = self.need("the map's answer to what moves the value on", strategy.follow_up)
        strategy_mod.attach_mover_gaps(strategy, R.ROOTS)
        writers = asyncio.run(self._writers_of(q.subject))
        movers = movers_of(follow, writers, q.subject, q.observed)
        self.say(f"selected by {follow.selected_by}; guarded by {[g.owner for g in follow.guarded_by]}")
        for owner, why in movers.items():
            self.say(f"  mover {owner}: {why}")
        gaps = {**follow.reader_gaps, **{g.owner: g.gaps for g in follow.guarded_by if g.gaps}}
        self.say(f"where the map could not read them: {gaps or 'nowhere'}" + (f" ({follow.gaps_note})" if follow.gaps_note else ""))
        readings = self._read_at_as_of(q, follow, writers, movers)
        asking = W.stuck(self.data.get("statuslog", {}), q.observed)
        asked = f"{q.subject}={q.observed}"
        handers = {rs.owner for rs in follow.reader_statements if rs.crosses_to and asked in rs.selects}
        ran, unanswered, unsearched = [], [], []
        entries = []
        for owner in movers:
            arms = arms_of(writers, owner, q.subject, q.observed)
            files = [f for j in writers if j.owner == owner for f in j.log_files]
            mine = [(s, v) for o, s, v in readings if o == owner]
            entry = {"owner": owner, "reached": "guard" if movers[owner] == GUARDED else "select",
                     "statements": [s for s, _ in mine if s], "verdict": None, "reasons": [],
                     "readings": [{"statement": s or None, "verdict": v} for s, v in mine], "premises": None}
            entries.append(entry)
            if any(v == "refused" for _, v in mine):
                entry["reasons"].append({"code": "K17", "text": "refused at the moment read, one moment of the window: not dropped"})
            if not arms:
                # with no arm of its own, a mover moves the value, if at all, through what it calls or the rows it hands
                # on: kept, and said why
                rel, func = owner.split("::", 1)
                lines = [rs.line for rs in follow.reader_statements if rs.owner == owner and asked in rs.selects]
                defs = defs_holding(rel, func, lines)
                reached, cut = through_calls(rel, [(f.lineno, f.end_lineno) for f in defs], writers, q.subject, q.observed)
                kept = kept_without_arms(bool(defs), reached, cut, owner in handers, owner in gaps)
                # the map's reach decides, where it can say for certain that nothing it goes on into moves the value
                reach = self._reach_of(q, owner, lines) if defs else None
                dropped, said = reach_said(reach, owner in gaps)
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
            if gaps.get(owner):
                entry["reasons"].append({"code": "map-gap", "text": f"the map could not read all of it ({'; '.join(gaps[owner])})"})
            files = self.files_of(owner, list(dict.fromkeys(files or follow.reader_log_files)), list(arms))
            to_hand = self.logs.names(files)
            if to_hand:
                self.warn_split(owner, list(arms))
            self.say(f"  {owner.split('::')[-1]}: {len(arms)} arm(s) that write another value "
                     f"{[(a.line, a.outcomes) for a in arms.values()]}; logs {files} ({'to hand: ' + str(to_hand) if to_hand else 'none to hand'})")
            if not to_hand:
                entry["verdict"] = "no-log"
                entry["premises"] = self._premises(q, owner, list(arms), to_hand, asking, asking, [], {})
                continue
            found, missed, holes, searched = self._ask_mover(q, owner, to_hand, asking)
            unanswered += missed
            if logs := self._runs_of(found, owner, searched):
                ran.append(Passes(owner, logs, list(arms.values()), q.key, searched))
                entry["verdict"] = "ran"
            elif holes:
                # not found is not "did not run" where the search did not reach
                unsearched.append(owner.split("::")[-1])
                self.say(f"  {owner.split('::')[-1]}: no run of its own found, but {', '.join(holes)} were not searched all the window: it may have run there")
                entry["verdict"] = "not-searched"
            elif missed:
                entry["verdict"] = "asks-unanswered"
            else:
                # not followed, and not ruled out either: absence is judgement 3's, with its rule (build order 2)
                over = " over the whole window" if searched is not asking else ""
                said = f"no run of its own found in {to_hand}{over}: not followed, though not ruled out (KNOWN_ISSUES.md K19)"
                self.say(f"  {owner.split('::')[-1]}: {said}")
                entry["verdict"] = "no-run-found"
                entry["reasons"].append({"code": "K19", "text": said})
            if entry["verdict"] != "ran":
                entry["premises"] = self._premises(q, owner, list(arms), to_hand, asking, searched, missed, holes)
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

    def _called_at(self, caller: R.Function, callee: Candidate) -> str | None:
        """Where *caller* calls on into *callee*'s function, by the map's reach: the ``file:line`` of the first call on
        the way (``ReachWrite.chain``), or None -- for H2 to read a run inside that call as a part of the caller's
        (``which_ran.inside_call``).  One way is found, not each: a call it misses leaves the runs apart."""
        graph = self._graph()
        if graph is None or not caller.node.body or not callee.lines:
            return None
        RE, record, _ = graph
        start = record.function_at(caller.rel, caller.node.body[0].lineno)
        end = record.function_at(callee.rel, callee.lines[0])
        if start is None or end is None or start == end:
            return None
        reached = record.reach([start], RE.Destination({end: ["called"]}, []))
        write = next((w for w in reached.writes or [] if w.function == end), None)
        return write.chain[0] if write is not None and write.chain else None

    def _reach_of(self, q: Stuck, owner: str, lines: list[int]) -> dict | None:
        """What the map's reach says of a mover with no arm of its own: from the defs holding its statements at *lines*
        -- every def of its name where it has none, as it takes rows another selects -- does any function it can go on
        into hold an arm moving *q.subject* off *q.observed* (``reach.moving_off``)?  The writes reached (``[]`` for none
        with nothing left open, ``None`` for none with calls left open), the holes, the assumptions the answer relied
        on and the closure's size; ``own_arms`` names a start the junctions give an arm S1 does not count.  ``None``
        where the dump has no ``reach_graph``: said once, as the first ask is (build2-design.md 6)."""
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
            ends[q.subject, q.observed] = RE.moving_off(mapdump.fragment().junctions, graph, q.subject, q.observed)
        destination = ends[q.subject, q.observed]
        rel = owner.split("::", 1)[0]
        starts = [graph.function_at(rel, n) for n in lines] if lines else graph.named(owner)
        starts = list(dict.fromkeys(s for s in starts if s))
        if not starts:
            return {"starts": [], "writes": None, "holes": 1, "first_holes": [f"{owner} -- no function of the reach_graph holds it"],
                    "assumptions": {}, "closure": 0, "own_arms": []}
        answer = graph.reach(starts, destination)
        return {"starts": starts,
                "writes": None if answer.writes is None else [{"function": w.function, "arms": w.arms, "chain": w.chain}
                                                              for w in answer.writes],
                "holes": len(answer.holes), "first_holes": answer.holes[:3], "assumptions": answer.assumptions,
                "closure": answer.closure, "own_arms": [s for s in starts if s in destination.arms]}

    def _ask_mover(self, q: Stuck, owner: str, to_hand: list[str], asking: W.Asking):
        """S1's asks for one mover: the window's last part first (*asking*, ``STUCK_SPAN``), and the rest of the window
        (``W.rest``) only where every ask of the last part was answered whole and held no run of its own -- no pass in
        six hours is not no pass in the window, and a mover that ran and stopped is the one a stuck value most needs
        named (build order 2a; tmaeno, 2026-10-16).  The door over files reads each file whole and asks once.

        Returns what ``_ask`` does, and the asking the runs were looked for over."""
        found, missed, holes = self._ask(q.key, to_hand, asking)
        if missed or holes or not self.logs.needs_window or self._runs_of(found, owner, asking):
            return found, missed, holes, asking
        rest = W.rest(self.data.get("statuslog", {}), q.observed)
        if rest is None:
            return found, missed, holes, asking
        self.say(f"  {owner.split('::')[-1]}: no run of its own from {asking.window[0]}: asking the rest of the window")
        return *self._ask(q.key, to_hand, rest), rest

    def _runs_of(self, found: dict[str, list], owner: str, asking: W.Asking) -> list[str]:
        """The logs among *found* holding a run of *owner* in *asking*'s window (any run, for the door over files)."""
        window = asking.window if self.logs.needs_window else None
        return [name for name, rs in found.items() if any((window is None or _overlaps(r, window)) and _of_mover(r, owner) for r in rs)]

    def _premises(self, q: Stuck, owner: str, lines, to_hand: list[str], asking: W.Asking, searched: W.Asking,
                  missed: list, holes: dict) -> dict:
        """Judgement 3's premises for a mover with no run of its own found: absence rules it out only where all three
        hold (build order 2a; KNOWN_ISSUES.md K19).  ``window``: the whole window was searched; ``logs``: every log its
        runs print to was asked (``every_log_asked``); ``prints``: every run that reaches an arm prints a string the asks
        search by.  Each is ``{"held": True | False | None, "why": ...}``, None where it is not known yet.

        *asking* is the window's last part (``W.stuck``), *searched* the part ``_ask_mover`` asked last, and *missed*
        and *holes* what that ask returned."""
        return {"window": self._searched_whole(q, to_hand, asking, searched, missed, holes),
                "logs": every_log_asked(owner, lines, to_hand),
                "prints": {"held": None, "why": PRINTS_NOT_KNOWN}}

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
        from bamboo.database.graph_database_client import GraphDatabaseClient

        graph_db = GraphDatabaseClient()
        await graph_db.connect()
        try:
            writers = await CodeMap(graph_db, map_id="panda", version=None).writers_of(subject)
            R.check_blobs(writers, f"the map's writers of {subject}")
            return writers
        finally:
            await graph_db.close()

    # ------------------------------------------------------------ Passes: which way each pass went
    def passes(self, q: Passes):
        self.start("S2", "which way each pass went")
        rel, func = q.owner.split("::", 1)
        # the arms S2 and S3 judge are the mover's own; a mover that writes through what it calls has none here
        self.need(f"an arm of {func}'s own that writes another value (S2 reads no arm of what it calls: KNOWN_ISSUES.md K14)", q.arms)
        fn = R.function(rel, func, [a.line for a in q.arms])
        runs = self.runs(q.key, q.logs, q.asking)
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
                    if E._truth(ast.parse(E.strip_annotation(g), mode="eval").body, fixed, env) is False
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
        test = E.strip_annotation(guard)
        where = next(
            (a for a in pathcond.ancestors(arm_stmt) if isinstance(a, (ast.If, ast.While)) and ast.unparse(a.test) == test),
            None,
        )
        where = self.need(f"the if whose test is {guard!r}", where)
        return Input(fn, guard, where.lineno, q.passes, q.envs, sorted({o for a in nearest for o in a.outcomes}))

    # ------------------------------------------------------------ Input: where the guard's name was bound
    def input(self, q: Input):
        self.start("S4", "where the guard's name was bound")
        fn = q.fn
        test = ast.parse(E.strip_annotation(q.guard), mode="eval").body
        names = [n.id for n in ast.walk(test) if isinstance(n, ast.Name) and n.id != "self"]
        names = list(dict.fromkeys(n for n in names if all(E.ev(ast.Name(n), env) is E.UNKNOWN for env in q.envs)))
        self.say(f"{q.guard!r} at {q.line} reads {names} (unknown from the passes)")
        name = self.need("one name the guard reads", names[0] if len(names) == 1 else None)
        walked = trace.walk(R.ROOTS, file=fn.rel, owner=fn.owner, lines=[q.line], classify=classify, reaching=False)
        self.say(f"walk from {fn.name}:{q.line}: {len(walked.steps)} step(s)" + (f"; {walked.note}" if walked.note else ""))
        takens = [E.taken(walked.steps, name, env, owner=fn.owner) for env in q.envs]
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
        params, constants, statement, at_call, unread = self.need(
            f"a facade's {callee}, the method the walk's stop names, and its calls on the proxy", proxy_reading(callee, call)
        )
        entry = R.entry_row(fn)
        for n, (p, env) in enumerate(zip(q.passes, q.envs, strict=False), 1):
            marks = R.prefix_holes(fn, entry, p.records[0]) if entry is not None else {}
            for expr, value in marks.items():
                env.put_printed(expr, value, f"the mark of {p.records[0].span}")
            args = [argument_said(params[i] if i < len(params) else None, a, E.ev(a, env)) for i, a in enumerate(call.args)]
            args += [argument_said(k.arg, k.value, E.ev(k.value, env)) for k in call.keywords]
            self.say(f"pass {n} ({p.first}): {ast.unparse(call.func)}({', '.join(args)})")
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
        inv, _ = asyncio.run(_investigate("panda", None, symptom, None, q.key, None, 1, None))
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
        gaps = {**follow.reader_gaps, **{g.owner: g.gaps for g in follow.guarded_by if g.gaps}}
        writers = asyncio.run(self._writers_of(q.subject))
        movers = movers_of(follow, writers, q.subject, q.observed)
        owners = list(dict.fromkeys(j.owner for j in writers))
        by_key = [o for o in owners if o not in movers and arms_of(writers, o, q.subject, q.observed)]
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
            rows_to = ", ".join(c.split("::")[-1] for c in st.crosses_to) or "itself"
            self.say(f"  {st.owner.split('::')[-1]}@{st.line}: {kind} (rows to {rows_to})")
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
                said.append(f"the map could not read all of it ({'; '.join(gaps)})")
                reasons.append({"code": "map-gap", "text": said[-1]})
            if verdict not in ("moved it", "write refuses", "refused"):
                # what keeps each statement from a verdict: a value the rows file does not hold, a run-time key
                for st in paths:
                    where = f"{st.owner.split('::')[-1]}@{st.line}"
                    if st.keys:
                        reasons.append({"code": "K18", "text": f"{where} asks rows by a value known at run time ({', '.join(st.keys)}): "
                                                               "never surely true"})
                    for subject in sorted(_statement_subjects(st)):
                        if why := self._unanswered_why(subject, q):
                            reasons.append({"code": why[0], "text": f"{where}: {subject} not compared ({why[1]})"})
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
        # what neither the row nor its history answers is said, not skipped
        unanswered = {x: why for x in sorted(_statement_subjects(st)) if (why := self._unanswered(x, q))}
        if unanswered:
            said.append("not compared: " + "; ".join(f"{x} ({why})" for x, why in unanswered.items()))
        if kind == "refused":
            return kind, said
        by_call, calls = self._routes(st, q)
        return min(kind, by_call, key=KINDS.index), calls + said

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
            self.say(f"  the call that moved it on ordered by {order!r} ({Path(r.file).name}:{r.lineno}, {r.stamp})")
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
        asks = {
            f"{where} groups now": PT.draw(PT.GroupsNow(where, table, cut, cols, key_col, blank, ranked)),
            f"{where} left {miss.observed!r} in the window, by the value it went to": PT.draw(
                PT.MovedOn(history, miss.observed, miss.since, miss.moved, terms, cols, key_col)
            ),
        }
        answers = self.data.get("answers", {})
        for ask, sql in asks.items():
            self.say(f"-- {ask}{' (answered in the rows file)' if ask in answers else ''}")
            for line in sql.splitlines():
                self.say(f"   {line}")
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
    for path in (args.out, args.asks, args.result):
        if path:
            Path(path).unlink(missing_ok=True)
    mapdump.install_from_env()
    r = Rerun(args)
    try:
        r.run()
        r.say()
        r.say(f"== reached the end; HAND used {len(r.hand_used)} time(s) {r.hand_used}")
        ends = [line for _, line in r.ends]
        outcome = {"kind": "end", "step": r.ends[-1][0] if r.ends else r.hop, "text": "\n".join(ends) or None, "code": None}
    except Stop as stop:
        r.say()
        r.say(f"STOP {stop}; HAND used {len(r.hand_used)} time(s) {r.hand_used}")
        outcome = {"kind": "stop", "step": r.hop, "text": r.lines[-1], "code": stop.code}
    r.stamp("end")
    if args.out:
        _write(args.out, "\n".join(r.lines) + "\n")
    asks = {}
    if r.unanswered:
        # every ask once, for the recording step to fetch together (analyzer-handover/stage2-record-format.md)
        asks = {ask.key: {"ask": ask.canonical(), "step": hop, "why": why} for hop, ask, why in r.unanswered}
        _write(args.asks, "".join(json.dumps(a) + "\n" for a in asks.values()))
    if args.result:
        outcome |= {"hand_used": r.hand_used, "branches": r.branches or None}
        _write(args.result, json.dumps(r.result_of(outcome, asks), indent=1) + "\n")


if __name__ == "__main__":
    main()
