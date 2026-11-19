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
               H2  which ran            ``which_ran`` over them, in the run the statuslog time falls in;
                                        where it decides nothing, the arms it left -> ``ArmsLeft``
               H3  what its guard reads the values its guards held on; a call in a guard is read as any term (K74)
``ArmsLeft``       each arm H2 left     walked on from H3 (``WroteBy``) to its end, its stop or its guard read
                                        false; the arms not excluded are the answer ("one of N", K38)
``Raised``     H5  what it was handed   the run's traceback through the function chosen (no step leads here
                                        since H4 was retired; S's exception end will, KNOWN_ISSUES.md K74):
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
                                        row (rule 20), those that hold the key (K47); in each,
                                        ``which_ran`` over the arms that write another value and every
                                        exit of the function
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
import builtins
import copy
import datetime
import functools
import hashlib
import importlib
import itertools
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NamedTuple, TypedDict, cast

from bamboo.codemap import pathcond, trace
from bamboo.codemap import strategy as strategy_mod
from bamboo.codemap.calls import Types, call_statements
from bamboo.codemap.models import (
    HISTORY_AFTER,
    KEY_BOUND,
    KEY_CONSTANT,
    KEY_HANDED,
    KEY_KEPT,
    KEY_NO_USE,
    KEY_PARAMETER,
    KEY_RETURNED,
    KEY_ROWS,
    KEY_UNKNOWN,
    MUST_PART_BIND,
    MUST_PART_MESSAGE,
    MUST_ROW_WRITTEN,
    RESULT_AGGREGATE,
    REWRITE_NOT_ATTRIBUTE,
    SKELETON_PRINT,
    Symptom,
    required_values,
)
from bamboo.codemap.panda import flow, sqltext
from bamboo.codemap.panda import sql as sql_mod
from bamboo.codemap.panda.provenance import facade_calls
from bamboo.codemap.panda.recognizers.sqlwrite import _flag_parameters
from bamboo.codemap.reading import containing_function
from bamboo.querier import agents as AG
from bamboo.querier import asked as A
from bamboo.querier import attr_time as AT
from bamboo.querier import changed as CH
from bamboo.querier import evaluate as E
from bamboo.querier import false_guard as G
from bamboo.querier import field_order as FO
from bamboo.querier import handlers as H
from bamboo.querier import held as HD
from bamboo.querier import heldcol as HC
from bamboo.querier import hop as HOP
from bamboo.querier import key_holes as KH
from bamboo.querier import map_reach as MR
from bamboo.querier import mapdump
from bamboo.querier import outbound as O
from bamboo.querier import parts as PT
from bamboo.querier import printed as PR
from bamboo.querier import raised as X
from bamboo.querier import render as T
from bamboo.querier import reproduce as P
from bamboo.querier import rows as R
from bamboo.querier import sources as SR
from bamboo.querier import stay_value as SV
from bamboo.querier import timeline as TL
from bamboo.querier import value_diff as V
from bamboo.querier import value_reach as VR
from bamboo.querier import windows as W
from bamboo.querier.recorded import EDGE, RecordedLogs
from bamboo.querier.records import (
    CORPUS,
    FileLogs,
    Log,
    Logs,
    Owners,
    Record,
    Run,
    Unrecorded,
    Window,
    keyed,
    runs_for,
)
from bamboo.querier.which_ran import (
    Candidate,
    Moment,
    Verdict,
    at_window,
    before_stretch,
    branches_at,
    class_of,
    constant_on_way,
    cut_passes,
    handing_rows,
    holds_key,
    hole_values,
    inside_call,
    judge,
    key_env,
    of_mover,
    once_loops_of,
    passes_of,
    printed_before,
    printed_row_column,
    recorded_in_pass,
    spans,
    stable_holes,
    write_prints,
)
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
    mover with no arm of its own; every gap is kept (``_gaps_json``)."""
    return any(gap.hides_where for gap in gaps)


def _gaps_json(gaps: list) -> list[dict]:
    """A mover's gaps as the map keeps them (``ReaderGap``), for result.json's ``gaps`` and the map-gap reason's record,
    which says them in the map's words (render's ``gap_texts``)."""
    return [g.model_dump(mode="json") for g in gaps]


def kept_without_arms(found: bool, reached: list[str], cut: list[str], hands: bool, hidden: bool) -> list[T.Said]:
    """Why S1 keeps a mover with no arm of its own: it moves the value, if at all, through what it calls or the rows it
    hands on.  None is dropped: ``called_from`` holds two hops, so a write further down is not seen, and "not listed"
    is not "does not reach" (bamboo-85; tmaeno, 2026-10-14).  Dropping one is the map's reach's to say, with its holes
    (``reach_said``); these are the reasons of a mover it keeps, or where the dump has no reach (KNOWN_ISSUES.md K14).
    *hidden*: a gap of the map's may hide where its rows go (``hides_where``)."""
    return [why for why, holds in (
        (T.Said("kept-def-not-found"), not found),
        (T.Said("kept-calls-into", {"reached": reached}), reached),
        (T.Said("kept-may-call-into", {"cut": cut}), cut),
        (T.Said("kept-hands-rows-on"), hands),
        (T.Said("kept-rows-hidden"), hidden),
    ) if holds] or [T.Said("kept-nothing-reached")]


def reach_said(reach: dict | None, hidden: bool) -> tuple[bool, T.Said | None]:
    """``(dropped, line)``: whether S1 drops a mover with no arm of its own by what the map's reach says of it
    (``Rerun._reach_of``), and the line saying so.  Dropped only where it reaches no write of another value and leaves
    no call on the way open (the map's ``[]``, never its ``None``), its own defs hold no arm the junctions place there,
    and no gap of the map's may hide where its rows go (*hidden*, ``hides_where``) -- such a gap may hide a write the
    reach has no junction for (build2-design.md 6, R1).  A drop that rests on calls the reach closed by their name
    (``closed_by_name``) names them.  No line where the map was not asked."""
    if reach is None:
        return False, None
    if reach["own_arms"]:
        return False, T.Said("reach-own-arms", {"own_arms": reach["own_arms"]})
    writes = reach["writes"]
    if writes:
        names = sorted({w["function"].split("::")[-1].split(".")[-1] for w in writes})
        return False, T.Said("reach-reaches-writers", {"n": len(writes), "names": names})
    if writes is None:
        return False, T.Said("reach-left-open", {"holes": reach["holes"], "first_holes": reach["first_holes"]})
    if hidden:
        return False, T.Said("reach-rows-hidden")
    closed = T.Said("reach-closed-by-name", {"n": reach["closed"], "first_closed": reach["first_closed"]}) if reach.get("closed") else ""
    return True, T.Said("reach-dropped", {"closure": reach["closure"], "assumptions": sorted(reach["assumptions"]), "closed": closed})


def _hole_holds(row: R.Row, record, name, key: str) -> bool:
    """Whether *record*, a line of the print *row*, holds *key* in the hole the map's must name *name* reads: an entry
    of its printed bind map (``sqltext.cut``: ``str(var_map)`` after the statement), or a hole of its logger's mark or
    of its message as the map's rows read them back (``trace.hole_values``, the record whole first) --
    ``updateTask_JEDI < jediTaskID=53637838 >``'s ``taskSpec.jediTaskID`` (:func:`which_ran.hole_values`)."""
    return key in hole_values(row, record, name)


def _written_offset(b) -> datetime.timedelta | None:
    """What the arm *b* writes, as an offset from the time it ran: the clock itself (``attr_time.NOW``), or a clock
    plus a settled offset (``Branch.amount``, the map's batch 2 W: updateTask_JEDI 1441's ``timeNow - 1 h``,
    ``CURRENT_DATE-1``); None where it is no such value (a parameter, a bind's scale, a value only said).  K-1b: a copy
    of what its own pass wrote, as that write (:func:`_copied_offset`)."""
    own = _own_offset(b)
    return own if own is not None else _copied_offset(b)


def _own_offset(b) -> datetime.timedelta | None:
    """:func:`_written_offset` of the arm's own outcome: the clock, or the clock with a settled offset."""
    if b.outcome in AT.NOW:
        return datetime.timedelta(0)
    a = getattr(b, "amount", None)
    if (a is None or a.base not in UTC_CLOCKS or a.offset_seconds is None or getattr(a, "scale_seconds", None)
            or getattr(a, "needs", None)):
        return None
    return datetime.timedelta(seconds=a.offset_seconds)


def _copied_offset(b) -> datetime.timedelta | None:
    """K-1b (bundle L; the map's (E), ``Branch.copied_from`` and ``copied_open``): a passthrough arm copying an attribute
    the same function's assignments put on the object in its pass, every way to the copy passing one of them
    (``copied_open`` empty), writes what they write -- the clock with their one offset, where every one of them at the
    lines ``copied_from`` names does: updateJob's 509 ``job.stateChangeTime = job.modificationTime`` after 506's
    ``naive_utcnow()``.  None otherwise (a way passing none, a copy of another object's field, an assignment of
    another value)."""
    copied_open = str(getattr(b, "copied_open", None) or "")
    if not str(getattr(b, "outcome", "") or "").startswith("passthrough(") or copied_open \
            or not getattr(b, "copied_from", None):
        return None
    fragment = mapdump.fragment()
    return _copied_offsets(id(fragment)).get(id(b)) if fragment is not None else None


@functools.cache
def _copied_offsets(fragment_id: int) -> dict[int, datetime.timedelta]:
    """:func:`_copied_offset` for every arm of the installed dump it reads, by the arm's identity."""
    fragment = mapdump.fragment()
    junctions = getattr(fragment, "junctions", None) or []
    by_column: dict[tuple[str, str], list] = {}
    for j in junctions:
        by_column.setdefault((j.owner, getattr(j, "subject", "")), []).extend(j.branches)
    out: dict[int, datetime.timedelta] = {}
    for j in junctions:
        for b in j.branches:
            outcome = str(getattr(b, "outcome", "") or "")
            lines = set(getattr(b, "copied_from", None) or [])
            if not outcome.startswith("passthrough(") or str(getattr(b, "copied_open", None) or "") or not lines:
                continue
            arms = [c for c in by_column.get((j.owner, outcome[len("passthrough("):-1]), [])
                    if lines & {c.line, *(getattr(c, "other_lines", None) or [])}]
            covered = {line for c in arms for line in lines & {c.line, *(getattr(c, "other_lines", None) or [])}}
            offsets = {_own_offset(c) for c in arms}
            off = next(iter(offsets)) if len(offsets) == 1 else None
            if covered == lines and off is not None:
                out[id(b)] = off
    return out


def _time_value(value: object) -> datetime.datetime | None:
    """*value* read as a time (``windows.when``), or None where it is none (P62: a row's window at that time holds its
    write)."""
    try:
        return W.when(str(value))
    except (ValueError, TypeError):
        return None


def _null_states(b) -> tuple[datetime.datetime, datetime.datetime]:
    """The states a write of a column tested ``IS NULL`` leaves (:data:`timeline.NULL`, Q1 batch 2 part 1): the literal
    ``None`` NULL; another literal, or a value known at run time (``runtime(pid)``, read as one that is not None: a
    premise where the map does not rule None out, :func:`_none_premised`), a value; any other (a value handed through,
    none read) either."""
    outcome = str(b.outcome or "").strip()
    if outcome == "None":
        return TL.NULL_ONLY
    if outcome.startswith("runtime(") or re.fullmatch(r"'[^']*'|\"[^\"]*\"|-?\d+(\.\d+)?", outcome):
        return TL.SET_ONLY
    return TL.EITHER


def _none_premised(b, owner: str) -> bool:
    """Whether :func:`_null_states` reads the arm *b* of *owner*'s, its value known only at run time, as one that is not
    None on a premise: where the map does not rule None out.  The map says what the signature of a parameter the arm
    writes as handed says of None (``Branch.handed``, its batch 4 N1): getTasksToBeFinished_JEDI's ``pid: str``
    excludes it, so that arm writes a value with no premise; a parameter with no annotation or a value that is no
    parameter (``runtime(commandMap['oldStatus'])``) says nothing -- premises.  getTaskWithID_JEDI's ``pid: str |
    None`` admits it, yet every way it comes in on that may run the arm holds a value (N1c, :func:`_handed_excludes`):
    no premise either.  K-3 (a): nor one writing the clock (:func:`_own_offset`, getTaskWithID_JEDI's 1818
    ``lockedTime=CURRENT_DATE``), never None."""
    if not str(b.outcome or "").strip().startswith("runtime(") or _own_offset(b) is not None:
        return False
    handed = getattr(b, "handed", None)
    if getattr(handed, "none", None) == SR.NONE_EXCLUDES:
        return False
    return handed is None or not _handed_excludes(b, owner)


@functools.cache
def _routes_into(fragment_id: int) -> dict[str, tuple[frozenset[tuple[str, ...]], bool]]:
    """Each function's routes of calls into it as the map's read sites in it hold them (``call_readings``' chains), and
    whether every call is placed (``callers_complete`` of them all)."""
    got: dict[str, tuple[frozenset[tuple[str, ...]], bool]] = {}
    for rs in getattr(mapdump.fragment(), "read_sites", None) or []:
        chains, complete = got.get(rs.owner, (frozenset(), True))
        got[rs.owner] = (chains | {tuple(c.chain) for c in rs.call_readings}, complete and bool(rs.callers_complete))
    return got


def _handed_excludes(b, owner: str) -> bool:
    """N1c (the map's batch 5): whether every way the parameter the arm *b* writes as handed comes in on
    (:func:`sources.none_ways`) that may run the arm holds a value that is not None.  A way does not run it where every
    route of calls into *owner* it is on (the call into *owner* first, the routes the map's read sites in *owner* hold,
    every call placed) is one the arm's constants rule out (``Branch.not_run_on``); a way on no such route may run it.
    getTaskWithID_JEDI 1847, under ``if lockTask:``: fourteen ways hand ``pid`` None and ``lockTask`` False; the one
    left, ContentsFeeder.py:156, hands ``self.pid``, bound from a ``pid: str``."""
    routes, complete = _routes_into(id(mapdump.fragment())).get(owner, (frozenset(), False))
    ruled = {tuple(r) for r in getattr(b, "not_run_on", None) or []}
    left = []
    for chain, said in SR.none_ways(owner, b.handed.name.rpartition(".")[2]):
        on = [r for r in routes if r[:len(chain)] == chain or chain[:len(r)] == r] if chain else []
        if not (complete and on and all(r in ruled for r in on)):
            left.append(said)
    return bool(left) and all(said == SR.NONE_EXCLUDES for said in left)


def _none_premises(used: dict[str, list[str]]) -> list[T.Said]:
    """The premise of each column tested ``IS NULL`` whose reading took an arm's value known only at run time as one
    that is not None (:func:`_none_premised`): its arms named."""
    return [T.Said("a-run-time-value-not-none", {"column": col, "arms": arms}) for col, arms in sorted(used.items())]


def _write_bounds(value, at: datetime.datetime) -> tuple[datetime.datetime, datetime.datetime]:
    """The bounds an arm's write at *at* leaves: *value* an offset from the time it ran (:func:`_written_offset`), the
    time its statement's printed bind map holds (:func:`_printed_write`), the states of a column tested ``IS NULL``
    (:func:`_null_states`), or None, not read."""
    if isinstance(value, datetime.timedelta):
        return at + value, at + value
    if isinstance(value, datetime.datetime):
        return value, value
    return value if value is not None else (TL.LOW, TL.HIGH)


@functools.cache
def _write_runs(fragment_id: int) -> dict[tuple[str, str], list[tuple[frozenset[int], int]]]:
    """``(owner, subject) -> [(the lines it writes the subject's values on, the line the statement runs on)]`` of the
    installed dump's writes (``KeyFrom.writes`` and ``statement``; *fragment_id* keys the cache to the dump): every
    write the map read, a statement no placed caller runs too."""
    out: dict[tuple[str, str], list[tuple[frozenset[int], int]]] = {}
    for kf in getattr(mapdump.fragment(), "key_froms", None) or []:
        if getattr(kf, "reads", False) or not kf.statement:
            continue
        for subject, lines in (kf.writes or {}).items():
            out.setdefault((_bare(kf.owner), subject), []).append((frozenset(lines), kf.statement))
    return out


def _branch_runs(owner: str, subject: str, b) -> set[int]:
    """The lines the statements writing the arm *b* of *owner*'s *subject* run on: each write whose ``KeyFrom.writes``
    holds the subject at the branch's line or one of its ``other_lines`` (prepareTasksToBeFinished_JEDI's lockedBy
    arm at 3839 with 3844: sqlTU's run at 4075, sqlTUU's at 4090)."""
    lines = {b.line, *(getattr(b, "other_lines", None) or [])}
    return {run for at, run in _write_runs(id(mapdump.fragment())).get((_bare(owner), subject), []) if at & lines}


def _landed(fn, sn, i: int, line: int, b, owner: str = "", subject: str = "") -> tuple[bool | None, int]:
    """R (the map's batch 2, ``Branch.count_prints``): what the first count print of the write's statement in the run
    after the write's print (record *i*, of the print at *line*) and before that print's next record says of the rows
    the run changed -- False for none or not one (lockTask_JEDI's ``done with {} locked by another``, the lock's
    ``failed to lock jediTaskID={}``), True for one or some (``done with {}``); None where none printed there; with
    the count print's line (0 with none).  A record another print reads at least as well is not taken.  A write whose
    count says none is read as not written (E-4, as A-1 reads a guard read false: :meth:`Rerun._column_writes`).

    E-4 (out/gm/qk-design.md's bundle E, tmaeno's OK 2026-11-17 13:42; the map's ``Branch.printed_counts``): a print
    holding the row count of the statement's run in a hole is read the same way, where every way from the function's
    entry to it passes one of the arm's lines (``flow.passes_before``): 0 changed no row, 1 or more changed it -- updateJobStatus's
    243 ``attemptNr={} nUp={} old={} new={}`` after its UPDATE at 241, the heartbeat's modificationTime arm at 238.

    Each count is of one statement's run, the line it runs on its ``run`` (the map's M-E4): it is read where the arm's
    write runs at one of the arm's statements (:func:`_branch_runs` of *owner*'s *subject*), not at another.  Where the
    arm's write runs at several, each statement writes the arm's outcome as the map spells it, the same value where
    that is a literal (the map, 2026-11-17 21:5x: a branch is one outcome and path condition, its statements' writes
    merged into it): for such an arm a count of 1 or more at any of its statements' runs says the value was written,
    and 0 says it was not only where every one of them is counted, each 0 -- prepareTasksToBeFinished_JEDI's lockedBy
    arm at 3839 with 3844 (NULL) writes at sqlTU's run at 4075 and sqlTUU's at 4090; 4092 counts sqlTUU's only, so
    its 1 says NULL was written, its 0 nothing.  An arm whose outcome is no literal (a bind, a name: the statements may
    write different values under one spelling) is read at one statement only (tmaeno's "add" 2026-11-17 21:1x,
    out/gm/qk-design.md).  A record made before M-E4 (``run`` 0) does not say which run it counts: no count is read
    for an arm with ``other_lines`` there, as before it."""
    counts = [*(getattr(b, "count_prints", None) or []), *(getattr(b, "printed_counts", None) or [])]
    if not counts:
        return None, 0
    if any(not getattr(c, "run", 0) for c in counts):
        if getattr(b, "other_lines", None):
            return None, 0
        runs = None
    else:
        runs = _branch_runs(owner, subject, b)
        if not runs or (len(runs) > 1 and not _literal_outcome(b)):
            return None, 0
    several = runs is not None and len(runs) > 1
    nxt = min((k for k, _, _ in sn.hits.get(line, ()) if k > i), default=None)
    #: the first count record after the write's print, by the run it counts where the arm writes at several (else 0)
    found: dict[int, tuple[int, bool, int]] = {}

    def note(run: int, k: int, wrote: bool, at: int) -> None:
        at_run = run if several else 0
        if at_run not in found or k < found[at_run][0]:
            found[at_run] = (k, wrote, at)

    def records_after(crow: R.Row) -> Iterable[int]:
        for k, literal, _ in sn.hits.get(crow.line, ()):
            if k <= i or (nxt is not None and k >= nxt) or any(
                    other != crow.line and any(kk == k and lit >= literal for kk, lit, _ in hs)
                    for other, hs in sn.hits.items()):
                continue
            yield k

    for cp in getattr(b, "count_prints", None) or []:
        stmt = R.statement_at(fn.node, cp.line)
        crow = fn.row_at(stmt) if stmt is not None else None
        if crow is None or (runs is not None and cp.run not in runs):
            continue
        for k in records_after(crow):
            note(getattr(cp, "run", 0), k, cp.rows in ("one", "some"), cp.line)
    printed = getattr(b, "printed_counts", None) or []
    # the arm's lines, every one at once: a way to the print passes one of them (``flow.passes_before``)
    arms = [s for at in (b.line, *(getattr(b, "other_lines", None) or [])) if (s := R.statement_at(fn.node, at)) is not None] if printed else []
    for pc in printed:
        stmt = R.statement_at(fn.node, pc.line)
        crow = fn.row_at(stmt) if stmt is not None else None
        if (stmt is None or crow is None or not arms or (runs is not None and pc.run not in runs)
                or not _passes_arm(stmt, arms)):
            continue
        for k in records_after(crow):
            count = _printed_count(crow, sn.run.records[k], pc.holes)
            if count is not None:
                note(getattr(pc, "run", 0), k, count > 0, pc.line)
    if not several:
        return (found[0][1], found[0][2]) if 0 in found else (None, 0)
    wrote = sorted((k, at) for k, w, at in found.values() if w)
    if wrote:
        return True, wrote[0][1]
    if runs is not None and set(found) == runs:
        return False, min(found.values())[2]
    return None, 0


def _literal_outcome(b) -> bool:
    """Whether the arm *b* writes a value its source spells as a literal (``None``, ``'ready'``, ``0``): the one value
    every statement writing the arm writes (:func:`_landed`); not a bind, a name, a call or the clock."""
    try:
        ast.literal_eval(str(getattr(b, "outcome", "") or "").strip())
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return False
    return True


def _passes_arm(printed: ast.stmt, arms: list[ast.stmt]) -> bool:
    """Whether every way from the function's entry to *printed* passes one of *arms* (``flow.passes_before``)."""
    try:
        return flow.passes_before(printed, arms)
    except (ValueError, LookupError):
        return False


def _printed_count(row: R.Row, record: Any, holes: list[int]) -> int | None:
    """The row count *record*, a line of the print *row*, holds in its message's hole *holes* names (counted from 0,
    as the map's ``PrintedCount``): the first map row that reads the record, its holes in their order; None where none
    reads it or the holes hold no one whole number."""
    for m in row.map_rows:
        for whole in (True, False):
            read = trace.hole_values(m, record.text, record.token, whole=whole)
            if read is None:
                continue
            got = {str(read[0][h]).strip() for h in holes if 0 <= h < len(read[0]) and read[0][h] is not None}
            return int(got.pop()) if len(got) == 1 and next(iter(got)).isdigit() else None
    return None


def _counted_names(b, keys: set[str]) -> list:
    """The holes of the count prints after the arm *b*'s write (the map's ``Branch.after_prints`` whose ``rows`` is
    "one" or "some": the line prints only where the run changed rows) that hold the written row's key (``MustName.row``
    ``written``, its column among *keys*).  A record of one holding the key says the statement changed that row:
    reactivatePendingTasks_JEDI's ``jediTaskID={} action=reactivate`` at 4735 after its ``sqlTU`` arm at 4728, under
    ``if tmpRow > 0:``.  Not every way that changes a row need print it, so its absence says nothing.  Empty where the
    dump reads no after prints (a map before them)."""
    return [n for p in getattr(b, "after_prints", None) or [] if getattr(p, "rows", "") in ("one", "some")
            for n in p.names if n.row == MUST_ROW_WRITTEN and n.column in keys]


#: a write whose statement's printed bind map leaves its column out: an object's ``valuesMap(onlyChanged=True)`` holds
#: only the attributes changed, so the statement ran without writing that column (:func:`_printed_write`)
UNWRITTEN = object()


def _printed_write(fn, sn, run, i: int, line: int, owner: str, b, subject: str) -> tuple[datetime.datetime, Any, int] | None:
    """The value the write of *subject* by the arm *b* ran with, read off the print of its statement's text and bind
    map (the map's ``WritePrint``; out/gm/qk-design.md, the bind-print part, tmaeno's OK 2026-11-16 15:1x): the first
    record of a print the WritePrint names after the write's print (record *i* of *run*, of the print at *line*) and
    before that print's next record, cut as the map cuts it (``sqltext.cut``).  ``(at, value, the print's line)``
    where the map printed holds the column's bind: a time the map reads off its ``str()``, None, or the text it keeps
    (updateTask_JEDI's 1483 ``':modificationTime': datetime.datetime(2026, 11, 15, 6, 46, 49, 943316)``, the arm at
    1441 or 1443, whichever ran); *value* :data:`UNWRITTEN` where the map is an object's
    ``valuesMap(onlyChanged=True)`` that leaves it out.  None where the statement's arms do not hold *b*, it binds the
    column to no one key, or no such print is read.  A record another print reads at least as well is not taken, as
    :func:`_landed`."""
    nxt = min((k for k, _, _ in sn.hits.get(line, ()) if k > i), default=None)
    for wp in write_prints(owner):
        if b.line not in (wp.arms.get(subject) or ()):
            continue
        binds = {w.bind for w in wp.binds if w.column == subject}
        if len(binds) != 1:
            continue
        bind = binds.pop()
        for p in wp.printed_by:
            stmt = R.statement_at(fn.node, p.line)
            prow = fn.row_at(stmt) if stmt is not None else None
            if prow is None:
                continue
            for k, literal, _ in sorted(sn.hits.get(prow.line, ())):
                if k <= i or (nxt is not None and k >= nxt) or any(
                        other != prow.line and any(kk == k and lit >= literal for kk, lit, _ in hs)
                        for other, hs in sn.hits.items()):
                    continue
                record = run.records[k]
                got = sqltext.cut(record.text, p.lead, p.gap)
                if got is None:
                    continue
                if bind in got[1]:
                    return W.when(record.stamp), got[1][bind], p.line
                if wp.object_map and wp.only_changed:
                    return W.when(record.stamp), UNWRITTEN, p.line
                break
    return None


def _hashable(value) -> bool:
    try:
        hash(value)
    except TypeError:
        return False
    return True


def _printed_bounds(value, nulls: bool):
    """What a value read off a write's printed bind map (:func:`_printed_write`) says of the column after it: a time,
    itself; None or another value of a column tested ``IS NULL``, its state.  None where it says nothing the timeline
    reads (a text kept, a value not a time)."""
    if nulls:
        return TL.NULL_ONLY if value is None else TL.SET_ONLY
    return value if isinstance(value, datetime.datetime) else None


def _binds_name(record, key: str) -> bool:
    """Whether *record*'s printed bind map (``sqltext.cut``) holds *key* as the value of any bind: the statement named
    the row (an IN list's ``:jediTaskID1``), whichever bind the write then takes it by."""
    got = sqltext.cut(record.text)
    return got is not None and any(v is not None and str(v).strip().strip("'\"") == key for v in got[1].values())


def prints_naming(writers: list, owner: str, lines, columns: set[str]) -> dict:
    """Judgement 3's third premise for *owner*: every run that reaches one of its arms at *lines* prints the key the
    asks search by -- a line the map says runs before the arm on every path (``Branch.must_prints``) with a hole it
    says holds a column among *columns* of the row the arm writes (``Branch.must_names``: in the message, the run's
    mark, or an SQL arm's bind map).  Where the map read the must prints' names whole (``Branch.must_names_whole``),
    none naming the key says no run need print it; where it did not, an arm with must prints and no such name is not
    known either way.  One with no must prints: no run need print anything before it.  ``{"held": True | False |
    None, "why": ...}``, False where any arm says so, None where any is not known, as ``every_log_asked``."""
    if not columns:
        return {"held": None, "why": T.Said("prints-no-key-column")}
    if not lines:
        return {"held": None, "why": T.Said("prints-no-arm-of-its-own")}
    unread, bare, whole, unnamed = [], [], [], []
    for line in sorted(lines):
        for b in (b for j in writers if j.owner == owner for b in j.branches if b.line == line):
            if b.must_prints is None:
                unread.append(T.Said("line-hole", {"line": line, "hole": b.must_hole or "not read"}))
            elif not b.must_prints:
                bare.append(line)
            elif not any(n.row == MUST_ROW_WRITTEN and n.column in columns for n in b.must_names or []):
                if b.must_names_whole:
                    whole.append(line)
                else:
                    unnamed.append(line)
    if bare:
        return {"held": False, "why": T.Said("prints-no-line-before", {"lines": sorted(set(bare))})}
    if whole:
        return {"held": False, "why": T.Said("prints-read-whole-none-names", {"lines": sorted(set(whole)),
                                                                              "columns": sorted(columns)})}
    if unread:
        return {"held": None, "why": T.Said("prints-not-read-before", {"unread": unread})}
    if unnamed:
        return {"held": None, "why": T.Said("prints-not-known-named", {"lines": sorted(set(unnamed)), "columns": sorted(columns)})}
    return {"held": True, "why": T.Said("prints-name-the-key", {"lines": sorted(lines), "columns": sorted(columns)})}


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
    if R.ROWS_FROM != "graph":
        return {"held": None, "why": T.Said("logs-the-maps-reading")}
    found = R._prints_of(owner, lines)
    if found is None or not found.statements:
        return {"held": False, "why": T.Said("logs-no-prints-of-its-own")}
    if bases := sorted({s.log_files_basis for s in found.statements} - {"logger"}):
        return {"held": False, "why": T.Said("logs-placed-by", {"bases": bases})}
    files = sorted({f for s in found.statements for f in s.log_files})
    if not files:
        return {"held": False, "why": T.Said("logs-in-no-file")}
    if missing := [f for f in files if f not in asked]:
        return {"held": False, "why": T.Said("logs-not-asked", {"missing": missing})}
    return {"held": True, "why": T.Said("logs-every-file-asked", {"files": files})}


def owners_of(ids: list[str]) -> dict[str, tuple]:
    """The functions whose runs a step reads, as ``of_mover`` names them, each with its marks' heads: the door asks no
    other DB proxy method's mark, and theirs for their own rows (a name two of them share: whole)."""
    ids = list(dict.fromkeys(ids))
    names = [o.split("::")[-1] for o in ids]
    return {n: R.mark_heads(*o.split("::", 1)) if names.count(n) == 1 else () for n, o in zip(names, ids, strict=True)}


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


def arm_guards(func: ast.AST, arm: Arm) -> tuple[list[str], list]:
    """``(guards, handlers)``: an arm's guards as S3 reads them -- its branches' path condition, the tests of the
    ``if``s around it and the early exits above it (``if t: return`` puts ``not (t)`` on what follows;
    ``pathcond.arm_condition``, as the map writes them) -- and the ``except`` handlers around it, which no guard says
    (an arm in one is reached only where something above raised).  check_output_file_metadata's arms 200 / 209 follow
    the return at 181 the pass took (KNOWN_ISSUES.md K61)."""
    stmt = R.statement_at(func, arm.line)
    unseen = pathcond.enclosing_guards(stmt) if stmt is not None else []
    return list(arm.guards), [u for u in unseen if u.kind == pathcond.UNSEEN_EXCEPT]


def early_exit_if(func: ast.AST, arm_stmt: ast.AST, guard: str) -> ast.If | None:
    """The ``if`` of the early exit above *arm_stmt* that puts *guard* on it (``pathcond.exits_above_at``): where S3
    goes on from such a guard, as ``guard_if`` finds an enclosing one.  Of two exits that say it, the later: the last
    test the passes went by before the arm.  None where no exit says it, or an ``if`` it names is not found."""
    lines = dict(pathcond.exits_above_at(arm_stmt)).get(guard, ())
    found = [n for n in ast.walk(func) if isinstance(n, ast.If) and n.lineno in lines]
    return max(found, key=lambda n: n.lineno) if found and len(found) == len(lines) else None


def raised_nothing(func: ast.AST, handler_line: int, printed: set[int]) -> int | None:
    """A row of *printed* that shows the ``try`` whose ``except`` is at *handler_line* raised nothing: one in its body
    with nothing after it there, up to the body's end, that may leave other than by falling through
    (``rows.may_divert``, as :func:`rows.must_rows` reads what follows a row), and no loop between.  setupper_atlas_plugin's
    run prints "end" at 206, its try's last statement, so the handler at 207 and the arm 212 in it did not run.  None
    where no row shows it, or where a loop of *func* holds the ``try`` (a later round may raise)."""
    tried = next((n for n in ast.walk(func) if isinstance(n, R.TRYS) and any(h.lineno == handler_line for h in n.handlers)), None)
    if tried is None or R.loops_over(tried, func):
        return None
    for line in sorted(printed, reverse=True):
        node = R.statement_at(func, line)
        while node is not None and node is not tried and node is not func:
            parent = getattr(node, "parent", None)
            placed = R.block_of(parent, node) if parent is not None else None
            if (placed is None or isinstance(parent, R.LOOPS) or (parent is tried and placed[0] != "body")
                    or any(R.may_divert(s) for s in placed[1][placed[1].index(node) + 1:])):
                break
            node = parent
        if node is tried:
            return line
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


def batched(rel: str, func: str, lines: list[int], subject: str) -> tuple[int, Any, int] | None:
    """``(line, KeyFrom, the loop's line)`` where *func* writes *subject* in a batch at one of *lines*: the map's
    KeyFrom of it says the object assigned there is handed to a call (``KEY_HANDED``), and a loop of the function holds
    the line and not the call.  A pass assigns each object in its loop and hands them all on after it, so its run of
    the key can end before the row's second (datasetManager's FinisherThr.run: 735 in the loop at 662, updateJobs at
    761; KNOWN_ISSUES.md K52)."""
    for kf in key_froms():
        if kf.owner != f"{rel}::{func}" or kf.unread != KEY_HANDED or not kf.object or kf.table:
            continue
        try:
            node = R.def_at(rel, func, lines)
        except LookupError:
            continue
        if (call := R.statement_at(node, kf.statement)) is None:
            continue
        around_call = {id(a) for a in pathcond.ancestors(call)}
        for line in sorted(set(lines) & set((kf.writes or {}).get(subject, []))):
            if (site := R.statement_at(node, line)) is None or line >= kf.statement:
                continue
            for a in pathcond.ancestors(site):
                if a is node:
                    break
                if isinstance(a, (ast.For, ast.AsyncFor, ast.While)) and id(a) not in around_call:
                    return line, kf, a.lineno
    return None


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


def made_then_born(owner: str, branch, subject: str) -> bool:
    """P10 (the investigator's u9/u10; the map's 3d30ddc, ``KeptHop.way`` returned / handed): whether the arm *branch* of
    *owner* sets *subject* on an object the function hands on or returns (the map's ``KeyFrom.unread``) that a write the
    map marks as making a new row of the class persists (``kept_hops``' ``persisted``, a statement whose KeyFrom
    :func:`is_birth`) on one of its ways: the value is set before the row is born, as a birth writes it.  JobGenerator's
    doGenerate:1227 ``jobSpec.prodSourceLabel = ...`` on the job it returns (1887), which TaskBuffer.storeJobs hands to
    insertNewJob's INSERT at 3200; the job handed to helpers at 1277 reaches no birth, and no write of the row itself.  Writes persisting the object after the birth write the same value where no other arm writes the column
    (:meth:`Rerun._born_only` asks that of every arm)."""
    return bool(_made_births(owner, branch, subject))


_MADE: list = [None, {}]


def _made_births(owner: str, branch, subject: str) -> tuple[tuple[str, int], ...]:
    """:func:`made_then_born`'s births: the INSERTs (``(function, statement)``) persisting the object *owner*'s arm
    *branch* sets *subject* on, where it is made then born; ``()`` where it is not.  Kept per key_froms."""
    kfs = key_froms()
    if _MADE[0] is not kfs:
        _MADE[:] = [kfs, {}]
    k = (_bare(owner), branch.line, subject)
    if k not in _MADE[1]:
        _MADE[1][k] = _made_births_read(owner, branch, subject, kfs)
    got: tuple[tuple[str, int], ...] = _MADE[1][k]
    return got


def _made_births_read(owner: str, branch, subject: str, kfs: list) -> tuple[tuple[str, int], ...]:
    cls = subject.partition(".")[0]
    mine = [kf for kf in kfs if _bare(kf.owner) == _bare(owner) and branch.line in (kf.writes.get(subject) or [])]
    if not mine or any(getattr(kf, "unread", "") not in (KEY_HANDED, KEY_RETURNED, KEY_KEPT) for kf in mine):
        return ()
    by_statement = {(_bare(kf.owner), getattr(kf, "statement", None)): kf for kf in kfs}
    births = {key for key, kf in by_statement.items() if is_birth(kf) and any(c.partition(".")[0] == cls for c in kf.writes)}
    hops = [h for kf in mine for h in getattr(kf, "kept_hops", None) or []]
    persisted = [(h, (_bare(h.owner), w.statement)) for h in hops for w in getattr(h, "persisted", None) or []]
    born_ways = [list(getattr(h, "via", None) or []) for h, key in persisted if key in births]
    # the object is handed to helpers too (doGenerate's 1277, 'handed to a call'): one way reaching the birth is enough
    if not born_ways:
        return ()
    # the map's caution (2026-11-18): a way persisting the object by a write of a row of the class that is no birth may
    # write a row that existed -- unless that way leaves the birth's way, in the function both pass, at a later call
    # line (setupper's activateJobs/keepJobs/updateJobs/archiveJob from storeJobs' 540, after its INSERT at 455/510)
    for h, key in persisted:
        kf = by_statement.get(key)
        if key in births or kf is None or not any(c.partition(".")[0] == cls for c in kf.writes):
            continue
        if not any(_after_on_its_way(list(getattr(h, "via", None) or []), way) for way in born_ways):
            return ()
    return tuple(sorted({key for _, key in persisted if key in births}, key=str))


def birth_arm(owner: str, branch, subject: str) -> bool:
    """P43: whether *owner*'s arm *branch* writes *subject* as the row is made -- a birth (:func:`born`), or a value set
    on an object a birth then persists (:func:`made_then_born`).  One rule wherever H2 and L2 ask for a birth."""
    return born(branch, birth_marks(owner, subject)) or made_then_born(owner, branch, subject)


def birth_statements(owner: str, branch, subject: str) -> tuple[tuple[str, int], ...]:
    """P43: the INSERTs (``(function, statement)``) that make the row *owner*'s birth arm *branch* writes *subject* in:
    its own statement for a birth (:func:`born`), the persisting ones for a value made then born (insertNewJob's 3200
    for JobGenerator's doGenerate:1250); ``()`` for an arm that is neither."""
    marks = birth_marks(owner, subject)
    if born(branch, marks):
        lines = [branch.line, *(getattr(branch, "other_lines", None) or [])]
        return tuple(sorted({(_bare(kf.owner), int(kf.statement)) for line in lines for kf in marks.get(line, ())
                             if getattr(kf, "statement", None) is not None}, key=str))
    return _made_births(owner, branch, subject)


class _ArmLine(NamedTuple):
    """An arm by its line alone, as :func:`born` and :func:`made_then_born` read a branch."""

    line: int
    other_lines: tuple = ()


def _after_on_its_way(via: list[str], born: list[str]) -> bool:
    """P10: whether the way *via* (``file:line`` calls from the arm's function on) leaves the way *born* in the function
    both reach at a later call line than *born*'s: read by the lines' order in that function (a premise: no loop or
    branch runs the later line first)."""
    k = 0
    while k < min(len(via), len(born)) and via[k] == born[k]:
        k += 1
    if k >= len(via) or k >= len(born):
        return False
    vf, _, vl = via[k].rpartition(":")
    bf, _, bl = born[k].rpartition(":")
    return vf == bf and vl.isdigit() and bl.isdigit() and int(bl) < int(vl)


def birth_why(owner: str, subject: str, lines: list[int]) -> T.Said:
    """Why the map calls *owner*'s writes of *subject* at *lines* births: each statement's ``new_row_by`` and the line
    that decided it (``new_row_line``), e.g. "sequence at 623"."""
    marks = birth_marks(owner, subject)
    said = {(getattr(kf, "new_row_by", "") or "new", getattr(kf, "new_row_line", None))
            for line in lines for kf in marks.get(line, ())}
    return T.Said("births", {"births": [T.Said("birth-by-at", {"by": by, "at": at}) if at else T.Said("birth-by", {"by": by})
                                        for by, at in sorted(said, key=str)]})


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


def _reads_of(owner: str, line) -> list:
    """The read KeyFroms (``reads``) of *owner*'s statement at *line*."""
    kfs = key_froms()
    if _READS[0] is not kfs:
        reads: dict[tuple, list] = {}
        for one in kfs:
            if getattr(one, "reads", False):
                reads.setdefault((one.owner, getattr(one, "statement", None)), []).append(one)
        _READS[:] = [kfs, reads]
    return _READS[1].get((owner, line), [])


def same_rows(owner: str, a, b) -> bool:
    """Whether the rows *owner*'s statement at *a* takes can be those its statement at *b* takes, as the map's read
    KeyFroms say: both take one table, by a term on a column both compare, and every such term of the two is filled
    from the same sources, none a value the map did not follow -- insertFilesForDataset_JEDI's 757 takes the dataset's
    files by datasetSpec's jediTaskID and datasetID, and 923 the dataset's ready and pending ones by the same two.  A
    term only one of them compares narrows its rows, not where they are taken from."""
    ra, rb = _reads_of(owner, a), _reads_of(owner, b)
    tables = {getattr(k, "table", "") for k in ra} & {getattr(k, "table", "") for k in rb}

    def terms(reads: list) -> dict[str, set]:
        out: dict[str, set] = {}
        for k in reads:
            if getattr(k, "table", "") in tables:
                column = re.split(r"\s*(?:!=|<>|>=|<=|=|<|>|\bIN\b)", k.term, maxsplit=1, flags=re.I)[0].strip().lower()
                out.setdefault(column, set()).update((s.kind, s.name, getattr(s, "attr", ""), getattr(s, "column", ""),
                                                      s.site_owner, s.site_line, tuple(getattr(s, "paths", None) or ()))
                                                     for s in k.sources)
        return out

    ta, tb = terms(ra), terms(rb)
    shared = set(ta) & set(tb)
    return bool(shared) and all(ta[c] and ta[c] == tb[c] and all(s[0] != KEY_UNKNOWN for s in ta[c]) for c in shared)


def key_chain(kf, rs) -> tuple[bool, list[str], int | None]:
    """Whether *kf*'s key comes from *rs*'s rows -- a ``rows`` source at *rs*, or at a statement of *rs*'s owner whose
    rows are taken by a key that comes from them, read through that statement's own KeyFroms (``reads``), or whose
    rows can be *rs*'s (``same_rows``) --, the terms the map did not read on the way there (an ``unread``, or a source
    not followed), and the statement whose rows can be *rs*'s it came through (None for none: *rs*'s own rows, or no
    way) -- asked only where no way is *rs*'s own.  resetInputToReGenCoJumbo_JEDI sets ready at 5038 the files of
    5026's rows; 5026 takes the job's files by the PandaIDs of 5009's rows, and 5009 takes those by 5001's datasetID
    and fileID: 5038's key is 5001's rows' (the movers' O6, 2026-10-29).  insertFilesForDataset_JEDI sets lost at
    1113 files of 757's rows, the dataset's, and 923 selects the dataset's ready files by the same values: a file 923
    selects can be the one 1113 writes (the map, 2026-10-31).  Whatever column fills a key on the way counts
    (KNOWN_ISSUES.md K28).  Another function's rows are not followed: *rs*'s rows are its owner's, and come to no
    statement before it."""
    unread, seen, todo = [], {}, [kf]
    while todo:
        one = todo.pop()
        for s in one.sources:
            if s.kind != KEY_ROWS or s.site_owner != rs.owner:
                continue
            if s.site_line == rs.line:
                return True, unread, None
            if s.site_line in seen:
                continue
            seen[s.site_line] = None
            for back in _reads_of(rs.owner, s.site_line):
                if back.unread:
                    unread.append(f"{back.term} at {s.site_line} ({back.unread})")
                unread += [f"{back.term} at {s.site_line} (not followed: {x.name})" for x in back.sources if x.kind == KEY_UNKNOWN]
                todo.append(back)
    via = next((line for line in seen if same_rows(rs.owner, line, rs.line)), None)
    return via is not None, unread, via


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
            mine, unread, _ = key_chain(kf, rs)
            if mine or unread:
                return ()
    return tuple(sorted(arms))


def keyed_arms(writers: list, rs, subject: str, arms) -> tuple[list[int], dict[int, list[str]], dict[int, list[str]], dict[int, list[str]]]:
    """``(keyed, unread, unpersisted, through)`` over *arms*, the arms of *rs*'s owner by line.  *keyed*: those whose
    write the map says names its row by a key filled from *rs*'s rows -- a ``KeyFrom`` of the owner's writing *subject*
    on a line of the arm's branch (its ``line`` or ``other_lines``), with a ``rows`` source of *rs*'s owner and line,
    or of a statement of the owner's that takes its rows by a key they fill, or whose rows can be *rs*'s
    (``key_chain``) (prepareTasksToBeFinished_JEDI keys the writes of its later loop by the jediTaskID its statements
    at 3723 and 3768 select, past their region).  Whatever column of the rows fills the key counts: the map does not
    say which (KNOWN_ISSUES.md K28).  *unread*: the others with a row-naming term the map did not read -- its
    ``unread``, or a value it did not follow -- each with the terms and why: their rows may be *rs*'s too.
    *unpersisted*: the others set on an object no write in the function persists -- a KeyFrom of no term, its
    ``unread`` what the object's fields reach (``UNPERSISTED``) -- each with the object and where its fields go: there
    is no key here to read, and the write that persists the value, if any, is the caller's or a callee's
    (datasetManager.run's ``file.status = "failed"`` at 713, a file of the job it hands on).  *through*: those of
    *keyed* keyed only by rows that can be *rs*'s, each with the terms and the statement: "fileID=:fileID by
    insertFilesForDataset_JEDI@757's rows"."""
    keyed: set[int] = set()
    own: set[int] = set()
    through: dict[int, list[str]] = {}
    unread: dict[int, list[str]] = {}
    unpersisted: dict[int, list[str]] = {}
    for kf in key_froms():
        if kf.owner != rs.owner or subject not in kf.writes:
            continue
        lines = set(kf.writes[subject])
        at = {b.line for j in writers if j.owner == rs.owner for b in j.branches
              if b.line in arms and lines & {b.line, *(getattr(b, "other_lines", None) or [])}}
        mine, _, via = key_chain(kf, rs)
        if mine:
            keyed |= at
            if via is None:
                own |= at
            for line in at if via is not None else ():
                through.setdefault(line, []).append(f"{kf.term} by {rs.owner.split('::')[-1]}@{via}'s rows")
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
            {line: said for line, said in sorted(unpersisted.items()) if line not in keyed},
            {line: said for line, said in sorted(through.items()) if line not in own})


def _before_gaps(found: SV.Before) -> list[T.Said]:
    """Why ``stay_value`` does not settle a column all the stay: the arms writing the value without the column's time,
    and those that may write the column while the value holds."""
    out = []
    if not found.starts and not found.lacking:
        out.append(T.Said("before-no-start"))
    if found.lacking:
        out.append(T.Said("before-written-without", {"arms": list(found.lacking)}))
    if found.left:
        out.append(T.Said("before-may-be-written", {"arms": list(found.left)}))
    return out


def _runtime_apart(before: SV.Before) -> list[str]:
    """M3(a)-1's arms writing the status from a value known only at run time (``runtime``) in a function with no arm
    that may begin the stay writing the value itself (``starts``, ``lacking``): a print of the function says no more
    of which of its arms ran (bundle A, A-3; updateTaskStatusByContFeeder_JEDI's 1356 beside its 1370 'assigning')."""
    funcs = {s.partition(" with ")[0].rpartition(":")[0] for s in before.starts} | {s.rpartition(":")[0] for s in before.lacking}
    return [r for r in before.runtime if r.rpartition(":")[0] not in funcs]


def _left_writers(row_value: tuple, before: SV.Before, begun: datetime.datetime) -> tuple:
    """Bundle F, F-3 (out/gm/qk-design.md, bundle F widened, tmaeno's OK 2026-11-17 15:2x): the row value's writers
    (:meth:`Rerun._row_value_write`), where its write fell inside the stay -- past a second after its start -- are those
    M3(a)-1 leaves as able to write the column while the value holds (``left``); each other is put apart with why it
    was left out.  NT-scouting-late's assessmentTime 19:53:41 was written by prepareTasksToBeFinished_JEDI:3744, not
    getAchievedTasks_JEDI:3283, whose statement takes running and pending rows only.  Where none of them is left, or
    the write fell at or before the start, as it was.  ``(written at, value, arms, as_of, (arm, why) put apart)``, and
    the arms K-1 names (:meth:`Rerun._row_value_write`) after them where it does."""
    at, value, arms, as_of = row_value[:4]
    named = tuple(row_value[5:])
    if at <= begun + datetime.timedelta(seconds=1) or not before.left:
        return (at, value, arms, as_of, (), *named)
    kept = tuple(a for a in arms if a in before.left)
    if not kept or kept == tuple(arms):
        return (at, value, arms, as_of, (), *named)
    why = dict(before.out)
    apart = tuple((a, why.get(a, "not among the arms that may write it while the value holds")) for a in arms if a not in kept)
    return (at, value, kept, as_of, apart, *named)


def _timeline_gaps(tl: TL.Timeline) -> list[T.Said]:
    """Why a timeline (:meth:`Rerun._timeline`) does not settle a column all the stay: the arms no record reads all of,
    besides the writes it read and where the row's value (*row_value*) shows none wrote it since (MT-scouting-late's
    assessmentTime, none from 12:47:03 on, the stay begun at 08:33:31)."""
    out = []
    if tl.writes or tl.absent:
        out.append(T.Said("timeline-read-so-far", {**_timeline_counts(tl), "k": len(tl.absent)}))
    if tl.row_value:
        out.append(T.Said("timeline-gap-row-value", {"at": f"{tl.row_value[0]:%Y-%m-%d %H:%M:%S}", "as_of": tl.row_value[3]}))
    out.append(T.Said("before-may-be-written", {"arms": [o.arm for o in tl.opens]}))
    return out


def _timeline_counts(tl: TL.Timeline) -> dict[str, int]:
    """The writes a timeline read in the stay (*n*) and in the ``windows.SLACK`` before it (*m*), which set where the
    stay began (:func:`timeline.first_bounds`), and that margin in minutes (*slack*)."""
    n = sum(w.at >= tl.start for w in tl.writes)
    return {"n": n, "m": len(tl.writes) - n, "slack": int(W.SLACK.total_seconds() // 60)}


def _timeline_said(subject: str, tl: TL.Timeline) -> T.Said:
    """The words for a column read on its timeline: each write read, marked where it fell before the stay, the arms
    shown not to write, those not read."""
    writes = [T.Said("timeline-write" if w.machine else "timeline-birth-write", {
        "at": f"{w.at:%H:%M:%S.%f}"[:12], "how": w.how, "arms": list(w.arms), "where": w.where, "machine": w.machine,
        "before": T.Said("timeline-before-the-stay") if w.at < tl.start else ""})
              for w in tl.writes]
    return T.Said("timeline-read", {"subject": subject, "start": f"{tl.start:%Y-%m-%d %H:%M:%S}",
                                    "end": f"{tl.end:%Y-%m-%d %H:%M:%S}", "slack": _timeline_counts(tl)["slack"],
                                    "writes": T.Said("timeline-writes", {"writes": writes}) if writes else T.Said("timeline-no-write"),
                                    "rowvalue": T.Said("timeline-row-value", {
                                        "value": f"{tl.row_value[1]:%Y-%m-%d %H:%M:%S}", "as_of": tl.row_value[3],
                                        "at": f"{tl.row_value[0]:%Y-%m-%d %H:%M:%S}", "arms": list(tl.row_value[2]),
                                        "how": (T.Said("timeline-row-value-clock") if len(tl.row_value) < 6
                                                else T.Said("timeline-row-value-matched", {"named": list(tl.row_value[5])}) if tl.row_value[5]
                                                else T.Said("timeline-row-value-matched-clock")),
                                        "apart": T.Said("timeline-row-value-apart", {"apart": [
                                            f"{a} ({why})" for a, why in next(iter(tl.row_value[4:]), ())]})
                                        if next(iter(tl.row_value[4:]), ()) else ""}) if tl.row_value else "",
                                    "confirmed": T.Said("timeline-confirmed", {"writes": [
                                        T.Said("timeline-confirmed-write", {"at": f"{at:%H:%M:%S.%f}"[:12], "arm": arm, **by})
                                        for at, arm, by in tl.confirmed]}) if tl.confirmed else "",
                                    "witnessed": T.Said("timeline-witnessed", {"writes": [
                                        T.Said("timeline-witnessed-write", {
                                            "at": f"{since:%H:%M:%S.%f}"[:12], "arms": list(arms), "where": where,
                                            "machine": machine, "value": f"{value:%Y-%m-%d %H:%M:%S}",
                                            "printed": f"{until:%H:%M:%S.%f}"[:12]})
                                        for since, until, value, arms, where, machine, _ in tl.witnessed]}) if tl.witnessed else "",
                                    "tied": T.Said("timeline-tied", {"writes": [
                                        T.Said("timeline-tied-write", {"at": f"{at:%H:%M:%S.%f}"[:12], "arm": arm, **by})
                                        for at, arm, by in tl.tied]}) if tl.tied else "",
                                    "quiet": [T.Said("timeline-quiet", {
                                        "arms": list(arms), "since": f"{since:%H:%M:%S.%f}"[:12],
                                        "until": f"{until:%H:%M:%S.%f}"[:12], "column": column})
                                        for since, until, arms, column in tl.quiet],
                                    "notrun": T.Said("timeline-not-run", {"writes": [
                                        T.Said("timeline-not-run-write", {"at": f"{at:%H:%M:%S.%f}"[:12], "arm": arm,
                                                                          "machine": machine, "guards": guards})
                                        for at, arm, machine, guards in tl.not_run]}) if tl.not_run else "",
                                    "absent": T.Said("timeline-absent", {"arms": list(tl.absent)}) if tl.absent else "",
                                    "opens": T.Said("timeline-opens", {"arms": [o.arm for o in tl.opens]}) if tl.opens else "",
                                    "begun": [*([T.Said("timeline-begun-by", {"arms": list(tl.begun_by)})] if tl.begun_by else []),
                                              *([T.Said("timeline-begun-unread", {"arms": list(tl.begun_assumed)})]
                                                if tl.begun_assumed else []),
                                              *([T.Said("timeline-begun-runtime", {"arms": list(tl.begun_runtime)})]
                                                if tl.begun_runtime else []),
                                              *([T.Said("timeline-begun-runtime-premise", {"n": len(tl.runtime_unread)})]
                                                if tl.runtime_unread else [])]})


def _held_before_said(subject: str, observed: str, found: SV.Before, start: str) -> T.Said:
    """The words for a column ``stay_value`` settles all the stay, with the premises it rests on."""
    premises = []
    if found.runtime:
        premises.append(T.Said("before-premise-runtime", {"arms": list(found.runtime), "observed": observed}))
    return T.Said("held-before", {"subject": subject, "value": start, "observed": observed, "starts": list(found.starts),
                                  "out": len(found.out), "premises": premises})


def _by_line(said: dict[int, list[str]]) -> str:
    """``keyed_arms``' *unread*, *unpersisted* or *through* as a mover's line says it: "4300: PandaID=:PandaID (no bind
    found)"."""
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
        keyed, unread, unpersisted, through = keyed_arms(writers, rs, subject, arms)
        if keyed:
            alike = f" ({_by_line(through)}, which take the same table by the same values)" if through else ""
            yield Way(rs.owner, f"{reader} selects the value, and its rows fill the key of the write(s) at {keyed}{alike}", rs)
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


def _again_index() -> tuple[dict, dict]:
    """The read KeyFroms reading a statement's rows again, by that statement's ``(owner, line)``, each with its
    ``same_rows`` source, and the read sites by ``(owner, line)``: built once per dump's ``key_froms``."""
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
    return _AGAIN[1], _AGAIN[2]


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
    again, sites = _again_index()
    asked = f"{subject}={observed}"
    reader = f"{rs.owner.split('::')[-1]}@{rs.line}"
    for kf, s in again.get((rs.owner, rs.line), []):
        if asked in (getattr(s, "never_at", None) or []):
            continue
        said = f"{reader} selects the value, and {kf.owner.split('::')[-1]}@{kf.statement} reads its rows again"
        if getattr(s, "via", None):
            for owner in getattr(s, "reaches", None) or []:
                if owner in writing:
                    yield Way(owner, f"{said} and hands them to it", rs)
            continue
        site = sites.get((kf.owner, kf.statement))
        if site is None:
            continue
        r = ReadAgain(site.owner, site.anchor.line_start, list(site.region), list(site.crosses_to))
        lo, hi = (r.region or [r.line, r.line])[:2]
        arms = arms_of(writers, r.owner, subject, observed)
        if any(lo <= line <= hi for line in arms):
            yield Way(r.owner, said, rs)
        else:
            keyed = keyed_arms(writers, r, subject, arms)[0]
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


#: ``[the fragment, its reach graph as calls, the functions its API endpoints reach]``, built once per dump
_REACH: list = [None, frozenset()]


def value_reach_of(follow, subject: str, observed: str, movers) -> VR.Reach:
    """M2 (:mod:`value_reach`): where the rows of the statements selecting *subject*=*observed* go -- *movers*
    (``movers_of``), the functions the statements hand the rows to holding the value (``crossing_values``) or a
    statement reading their rows again hands them to (``read_again``'s), and what those call."""
    fragment = mapdump.fragment()
    if _REACH[0] is not fragment:
        _REACH[:] = [fragment, VR.entered(getattr(fragment, "endpoints", None) or [])]
    asked = f"{subject}={observed}"
    statements = [rs for rs in follow.reader_statements if asked in rs.selects and rs.result != RESULT_AGGREGATE]
    crossed = [c for rs in statements for c in rs.crosses_to if asked in crossing_values(rs, c)]
    again, sites = _again_index()
    for rs in statements:
        for kf, s in again.get((rs.owner, rs.line), []):
            if asked in (getattr(s, "never_at", None) or []):
                continue
            # the statement's own function is a mover only where read_again places an arm (in movers); what it hands
            # the rows to is reached
            crossed += list(getattr(s, "reaches", None) or [])
            if (site := sites.get((kf.owner, kf.statement))) is not None:
                crossed += list(site.crosses_to)
    other = VR.other_values(getattr(fragment, "read_sites", None) or [], subject, observed)
    return VR.reach(statements, crossed, movers, _REACH[1], other)


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


def woken_said(woken: list[Wake]) -> T.Said:
    """A mover dropped as one that reaches no write, said as the waker it is of the movers kept."""
    return T.Said("its-waker", {"writes": [T.Said("writes-at", {"writes": w.writes, "line": w.line}) for w in woken],
                                "movers": sorted({f"{w.mover.split('::')[-1]} ({w.reader})" for w in woken})})


def refused_arms(writers: list, owner: str, subject: str, observed: str) -> list[T.Said]:
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
                    out.append(T.Said("writes-only-on-a-row-holding", {"line": b.line, "outcome": b.outcome, "column": column,
                                                                       "values": rv[1]}))
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
    result.json's ``outcome.code``; None where none is named yet.  *said* is the sentence as data (render.Said)."""

    def __init__(self, what: T.Said | str, code: str | None = None) -> None:
        super().__init__(T.text(what))
        self.said = what
        self.code = code


class Excluded(Stop):
    """A branch of :meth:`Rerun.arms_left` whose arm did not write: its guard read false at each row it may have written at."""


#: the kinds of what is open in an arm H2 left (fk1): the logs searched whole hold no line that tells it; the data may
#: hold it and the querier does not read it; a log not searched whole, an ask the records do not answer
NOT_IN_DATA, NOT_READ, NOT_RECORDED = "not in the data", "not read", "not recorded"


class Premise(str):
    """What an arm H2 left rests on (``Possible.premises``): its record (*said*), and its kind (``NOT_READ`` ...),
    given where it is found, not read off the text.  As a string it is the record rendered, as out.txt and which_ran's
    reasons read it; result.json and the records keep the record (:func:`_premise_json`, :func:`_said`)."""

    kind: str
    said: T.Said | str

    def __new__(cls, said: T.Said | str, kind: str) -> Premise:
        premise = super().__new__(cls, T.text(said))
        premise.kind = kind
        premise.said = said
        return premise


def _said(p: object) -> object:
    """*p* as a record's fact: a premise's record, else *p*."""
    return p.said if isinstance(p, Premise) else p


def _premise_json(p: object) -> object:
    """A premise as result.json keeps it: its record's JSON and its kind beside (``premise_kind``)."""
    if isinstance(p, Premise) and isinstance(p.said, T.Said):
        return {**T.to_json(p.said), "premise_kind": p.kind}
    return p


def _condition_read(p: object, read: set[str]) -> bool:
    """Whether *p* is a condition an arm was left resting on (:meth:`StandIn.left`) that a later step read as holding,
    by its record's guard (KNOWN_ISSUES.md K62)."""
    return (isinstance(p, Premise) and isinstance(p.said, T.Said) and p.said.kind == "its-condition-not-read"
            and p.said.facts.get("guard") in read)


def _branches_at(subject: str, owner: str, line: int) -> list:
    """The map's branches of *subject* at *owner*'s *line* (:func:`which_ran.branches_at`), in the map this module
    reads."""
    return branches_at(subject, owner, line, mapdump.fragment())


#: a path condition's term tested in a method the arm's statement calls, in that method's names: ``X  [in f()]``
#: (``pathcond``'s mark, ``E.read_elsewhere``)
_IN_CALLEE = re.compile(r"(?P<term>.*?)\s+\[in (?P<fn>\w+)\(\)\]")


def callee_ways(subject: str, owner: str, line: int, observed: str) -> list[list[str]]:
    """The map's ways to *observed* at *owner*'s *line* -- its branches there of that outcome, each as its path
    condition -- where one holds a term tested in a method the statement calls (``_IN_CALLEE``): the value is that
    method's, and each way one of its returns (KNOWN_ISSUES.md K65).  ``[]`` where none does."""
    ways = [list(b.path_condition or []) for b in _branches_at(subject, owner, line) if b.outcome == observed]
    return ways if any(_IN_CALLEE.fullmatch(t) for way in ways for t in way) else []


def _in_caller(term: str, passed: dict[str, str]) -> str:
    """*term*, a condition in the names of a method the arm's statement calls, in the caller's: a parameter the call
    hands a path (the map's ``call_bindings``) reads that path; any other name -- the method's own local, a parameter
    left to its default, a module it imports -- one no env holds, so a term needing it is not read."""

    class _Rename(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.AST:
            if node.id in passed:
                return ast.parse(passed[node.id], mode="eval").body
            return ast.copy_location(ast.Name(id=f"_callee_{node.id}", ctx=node.ctx), node)

    return ast.unparse(_Rename().visit(ast.parse(term, mode="eval")))


#: why the judge excludes a candidate by its guards: one read false in the pass's values (step 5), or one that needs the
#: value already there (rule 6) -- not by what its run printed or did not
_GUARD_REASONS = ("step 5:", "rule 6:")


def _bare(owner: str) -> str:
    """A ``file::Class.function`` owner as H2's candidates and the map's records name it (:func:`rows.named`, D-1):
    ``file::function``, or with what encloses it where its module repeats the name."""
    return R.named(owner)


def _atom_text(text: str) -> str:
    """An atom's text as the map's evaluator keys it (``ast.unparse``), so the map's ``ConditionAtom.text`` and the
    atoms :func:`evaluate.truth` meets in a path condition's entry match whatever their spacing and quotes."""
    try:
        return ast.unparse(ast.parse(text.strip(), mode="eval").body)
    except SyntaxError:
        return text.strip()


def _plain(owner: str) -> str:
    """:func:`_bare`, less a repository's ``panda:`` before the file: a read site's owner and a KeyFrom's compare."""
    rel, _, qual = owner.partition("::")
    return R.named(f"{rel.rpartition(':')[2]}::{qual}")


def _unprinted_count(tree: ast.AST, bound: dict[str, Any], counted: str | None,
                     own_values: set[str]) -> tuple[bool, dict[str, bool], list[str], int]:
    """P25 (i)'s rest: what a print's condition *tree*, false for the row in a pass that took it, says of the name
    *counted*, its other names *bound* where known.  The atoms naming one of the pass's *own_values* (a rowcount, a
    container, a parameter) and not the count are put the one way the print needs them, where that way is one; the
    other unknown atoms are tried both ways (``evaluate.truth``).  The count was 0 where the condition is false at 0
    and holds at every count from 1 to a bound past each number the count's atoms hold (their product, at most
    10 000, plus 2).  ``(zero, the own values' way, the atoms left open, the bound)``."""
    found: dict[str, ast.AST] = {}
    for atom in E.shared.atoms(tree):
        found.setdefault(ast.unparse(atom), atom)

    def names_in(node: ast.AST) -> set[str]:
        return {x.id for x in ast.walk(node) if isinstance(x, ast.Name)}

    def env_with(n: int | None) -> Any:
        env = E.shared.Env()
        for name, value in bound.items():
            env.put(name, value, "read")
        if counted is not None and n is not None:
            env.put(counted, n, "a count tried")
        return env

    def fits(n: int | None, fixed: dict[str, bool], outcome: bool) -> list[dict[str, bool]]:
        env = env_with(n)
        free = [t for t, a in found.items() if t not in fixed and E.shared.truth(a, {}, env) is E.shared.UNKNOWN]
        out = []
        for values in itertools.product((True, False), repeat=len(free)):
            one = {**fixed, **dict(zip(free, values, strict=True))}
            if E.shared.truth(tree, one, env) is outcome:
                out.append(one)
        return out

    premise = [t for t, a in found.items() if counted not in names_in(a) and names_in(a) & own_values]
    held = fits(None, {}, True)
    way = {t: held[0][t] for t in premise if held and t in held[0] and all(h.get(t) == held[0][t] for h in held)}
    unknown = [t for t, a in found.items() if t not in way and E.shared.truth(a, {}, env_with(None)) is E.shared.UNKNOWN]
    if counted is None or len(way) != len(premise):
        return False, way, unknown, 0
    # a bound past every number the counted name is compared with: the product of the numbers its atoms hold
    numbers = [abs(v) for a in found.values() if counted in names_in(a) for x in ast.walk(a)
               for v in [x.value if isinstance(x, ast.Constant) else bound.get(x.id) if isinstance(x, ast.Name) else None]
               if isinstance(v, int | float) and not isinstance(v, bool) and v]
    top = int(min(10_000, math.prod(max(1.0, v) for v in numbers))) + 2
    zero = bool(fits(0, way, False)) and all(not fits(n, way, False) for n in range(1, top + 1))
    return zero, way, unknown, top


def _entries_possible(entries: list[str], atoms: list, fixed: dict[str, str], row: dict | None = None) -> bool:
    """Whether a path's *entries* can all hold together (P19): the map's condition *atoms* on a subject compared with
    literals take each subject's one value -- *fixed* where given, else each literal its atoms name or one they do not
    (``<other>``) -- so atoms of one column agree (4008's ``fileSpec.status == 'nooutput'`` and an accumulation's
    ``not in ('nooutput')``); atoms on nothing so placed are tried both ways (``evaluate.truth``).  Entries read
    elsewhere ([at N], [in f()]) are left out.  *row*: a row of the class *fixed*'s first subject is of, its columns
    (lower case) fixing atoms on them where not in *fixed* (NULL a value no literal equals).  True where some
    assignment makes every entry true or leaves it open."""
    trees = []
    for e in entries:
        if E.shared.read_elsewhere(e):
            continue
        try:
            trees.append(ast.parse(E.shared.strip_annotation(e), mode="eval").body)
        except SyntaxError:
            continue
    if not trees:
        return True
    placed: dict[str, list[tuple[str, str, set[str]]]] = {}
    for a in atoms:
        subj, op = str(getattr(a, "subject", "") or ""), str(getattr(a, "op", ""))
        if subj and op in ("==", "!=", "in", "not in"):
            placed.setdefault(subj, []).append((_atom_text(str(a.text)), op, {str(x).strip("'\"") for x in getattr(a, "values", None) or []}))
    subjects = sorted(placed)
    row_cls = next(iter(fixed), "").partition(".")[0]

    def own(s: str) -> list[str] | None:
        cls, _, attr = s.partition(".")
        if row is None or cls != row_cls or attr.lower() not in row:
            return None
        value = row[attr.lower()]
        return ["<none>" if value is None else str(value)]

    choices = [[fixed[s]] if s in fixed else own(s) or sorted({v for _, _, vs in placed[s] for v in vs}) + ["<other>"]
               for s in subjects]
    env = E.shared.Env()
    for pick in itertools.product(*choices):
        known: dict[str, bool] = {}
        for s, value in zip(subjects, pick, strict=True):
            for text, op, vs in placed[s]:
                inside = value in vs
                known[text] = inside if op in ("==", "in") else not inside
        if all(E.shared.truth(t, known, env) is not False for t in trees):
            return True
    return False


def _entry_requests(entries: list[str], endpoints: list) -> list[str]:
    """The routes (``v1/harvester/update_workers``) of the map's *endpoints* whose function holds an entry of a
    waker's ways (``file:line``, a route's outer end), one per entry that one holds; an entry no endpoint's function
    holds gives none."""
    by_function = {_plain(str(e.function)): f"{e.version}/{e.route}" for e in endpoints}
    out = []
    for entry in entries:
        rel, _, line = entry.rpartition(":")
        node = R.def_holding(rel, int(line)) if line.isdigit() else None
        if node is not None and (route := by_function.get(_plain(R.owner_of(rel, node)))) is not None:
            out.append(route)
    return out


@dataclass(frozen=True)
class LaterStep:
    """Q1 batch 2 part 2: one way the selecting statement's later steps drop a row its SELECT returned -- an exit of
    the map's (*kind* ``@`` *line*, decided at *decided*), with the lines its way prints a hole naming the row's key
    (*prints*; none: it drops the row in silence)."""

    kind: str
    line: int
    decided: int
    prints: tuple[int, ...]

    @property
    def way(self) -> str:
        return f"{self.kind}@{self.line}"


def exit_print_lines(st) -> list[int]:
    """The print lines of the ways out of the statement *st* (``StatementExit.skeleton``, which
    ``strategy.attach_reader_traces`` fills)."""
    return sorted({s.line for e in getattr(st, "exits", None) or [] for s in e.skeleton or [] if s.kind == SKELETON_PRINT})


def later_steps(st, keyed: Callable[[int], bool]) -> list[LaterStep]:
    """The ways the statement *st*'s later steps drop a row its SELECT returned: the map's ``exits`` but the no-rows
    ones (nothing came back: the terms' half), each with the print lines of its way (:func:`exit_print_lines`) that
    *keyed* says print a hole the map names as the row's key -- the function's prints' holes (``function_prints``):
    the walk's skeleton names no hole's column.  Ways printing the same lines are one (an ``except`` and the
    ``continue`` it ends in)."""
    out: dict[object, LaterStep] = {}
    for e in getattr(st, "exits", None) or []:
        if e.no_rows:
            continue
        prints = tuple(sorted({s.line for s in e.skeleton or [] if s.kind == SKELETON_PRINT and keyed(s.line)}))
        out.setdefault(prints or (e.kind, e.line), LaterStep(e.kind, e.line, e.decided_at, prints))
    return list(out.values())


#: the map's ``KEY_JOINED`` (codemap-xxxix, batch 5's M-b): a term comparing the column with another table's
KEY_JOINED = "joined"


@dataclass(frozen=True)
class KeyTerms:
    """Q1 batch 2 part 2: a selecting statement's terms no condition tree holds, by where the map says each compares
    its column with (``KeyFrom.sources``): another table's column (*joined*, ``(term, source)``), each read with the
    terms a row need meet only one of (*either*: ``some_rows``); a value the agent hands the call (*handed*); and
    what is none of these (*other*)."""

    joined: tuple[tuple[str, str], ...] = ()
    either: tuple[str, ...] = ()
    handed: tuple[str, ...] = ()
    other: tuple[str, ...] = ()


def key_terms(key_froms: list, owner: str, line: int) -> KeyTerms:
    """The read terms of the statement on *line* of *owner* (:class:`KeyTerms`); the terms of constants alone are
    the condition trees' (``status IN (...)``) and left out.  A joined pair the map records from both tables is one."""
    kfs = [kf for kf in key_froms if getattr(kf, "reads", False) and kf.statement == line and _plain(kf.owner) == _plain(owner)]
    joined: list[tuple[str, str]] = []
    pairs: set[frozenset[str]] = set()
    either, handed, other = [], [], []
    for kf in kfs:
        kinds = {s.kind for s in kf.sources}
        if KEY_JOINED in kinds:
            source = next(s.name for s in kf.sources if s.kind == KEY_JOINED)
            pair = frozenset({kf.table.lower(), source.partition(".")[0].lower()})
            if pair not in pairs:
                pairs.add(pair)
                joined.append((kf.term, source))
        elif kinds and kinds <= {KEY_CONSTANT}:
            continue
        elif kf.some_rows:
            either.append(kf.term)
        elif kinds == {KEY_PARAMETER}:
            handed.append(kf.term)
        else:
            other.append(kf.term)
    return KeyTerms(tuple(joined), tuple(either), tuple(handed), tuple(other))


def taken_as_handed(key_froms: list, owner: str, subject: str, lines: set[int]) -> bool:
    """I-2: whether every write of *subject* at *lines* of *owner* takes its row by a key the map draws, in the owner
    itself, from its parameters alone (its ``KeyFrom`` sources with no ``via``, each ``parameter``; one through a call
    route is where a caller got what it hands): the owner runs on a row only as its callers hand it -- activateJob's
    113 by ``job.PandaID`` (and, through TaskBuffer.py:982, peekJob's rows), not FinisherThr.run's writes, by the rows
    its own selection reads."""
    kfs = [kf for kf in key_froms if _bare(kf.owner) == _bare(owner) and set(kf.writes.get(subject) or []) & lines]
    own = [[s for s in kf.sources if not getattr(s, "via", None)] for kf in kfs]
    return bool(kfs) and all(mine and all(s.kind == KEY_PARAMETER for s in mine) for mine in own)


def _tree_leaves(node) -> list:
    """The terms of a condition tree (the map's ``all``/``any`` nodes opened)."""
    if node.node in ("all", "any"):
        return [leaf for item in node.items for leaf in _tree_leaves(item)]
    return [node]


def _premise_kind(p: object) -> str:
    """The kind of a premise's record (``no-line-holds-the-key``), '' for a premise said as text."""
    return p.said.kind if isinstance(p, Premise) and isinstance(p.said, T.Said) else ""


def _arm_name(arm: str) -> str:
    """A ``file::function:line`` arm as :class:`Possible` names it: the file's name, not its path."""
    owner, _, line = arm.rpartition(":")
    rel, _, func = owner.partition("::")
    return f"{Path(rel).name}::{func}:{line}"


def _a_default(value: str) -> bool:
    """Whether *value* is one a row holds with no write of the column: NULL, or 0 (a spec's zero attribute, as
    ``JobSpec._zeroAttrs`` stores an unset one).  Any 0 is taken as one: the map does not say which columns default to it."""
    text = str(value).strip()
    if text in ("", "None", "NULL"):
        return True
    try:
        return float(text) == 0
    except ValueError:
        return False


def _status_before(states: list[dict], state: dict | None) -> dict | None:
    """The statuslog row just before *state*: what the row's status was as the pass that wrote *state* began, where no
    write between left no row (KNOWN_ISSUES.md K65).  Rows are told apart as ``held.row_before`` tells them."""
    if not state:
        return None
    before = [s for s in states if HD.row_before([s], state)]
    return max(before, key=lambda s: W.when(s.get("time_extended") or s["time"]), default=None)


def _store_unread(subject: str, owner: str, line: int, fn: Any, via: Any) -> T.Said | str:
    """What an attribute's END at *owner*:*line* leaves unread of its value reaching the row (KNOWN_ISSUES.md K41), by
    the map's branch of the arm: an attribute arm's row is written by a later statement or call, which is not read; an
    SQL arm (``REWRITE_NOT_ATTRIBUTE``) read on its caller's run (*via*) whose own function *fn* catches its statement's
    exception may have failed with the call returning.  Nothing for an SQL arm on its own run, whose path is read with
    the statement's raise apart (``printed``'s E6), nor where the map has no branch of the arm."""
    branch = next(iter(_branches_at(subject, owner, line)), None)
    if branch is None:
        return ""
    if not any(o.why == REWRITE_NOT_ATTRIBUTE for o in branch.rewrites_open or []):
        return T.Said("row-written-later-not-read")
    if via is None:
        return ""
    handler = H.leave(fn.node, f"call at {line}")[2]
    return T.Said("failure-caught-not-read", {"fn": fn.shown, "line": handler.lineno}) if handler is not None else ""


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
    #: judgement 4 of a late value (``Rerun._late_passes``): the stay's start, and the second it moved on -- the passes
    #: ending before that second alone are read -- and why a pass not found may still have run ("" where none can)
    since: str | None = None
    before: str | None = None
    unsure: list = field(default_factory=list)
    #: the subject and the mover's ``called_from``: where no pass of it took the row, U1 reads its callers' (K55)
    subject: str | None = None
    entries: list[str] = field(default_factory=list)
    #: a late value's stay ``(start, move)``, for a mover L2 follows that did not move it (``Rerun._late_absence``): its
    #: passes in the stay are read, and none there excludes it
    stay: tuple[str, str] | None = None


@dataclass
class Fork:
    """Questions that each go on alone, from the same point: every mover that ran in the window (S1)."""

    branches: list
    labels: list[T.Said | str]


@dataclass
class Blocked:
    """What kept every pass from the arms that move the value.  *exit_line* is None where every pass took none of the
    candidates (:func:`took_none`), or ran none of the arms (:func:`ran_no_arm`) and not all by one exit shown: then
    *left_by* holds the exits such a pass may have left by, and *took*, where any pass was shown to leave by one, each
    pass's (None for one not shown).  *unhanded*, each pass's D-4 reason where every arm of it was a call handing no
    element of the key's turn (``Verdict.unhanded``), empty for one otherwise."""

    fn: R.Function
    exit_line: int | None
    passes: list
    envs: list
    arms: list[Arm]
    left_by: tuple[int, ...] = ()
    took: tuple[int | None, ...] = ()
    unhanded: tuple[str, ...] = ()


@dataclass
class Caller:
    """A def that calls into a mover, as U1 reads it (:func:`callers_of`): the def holding the call (*owner*, its bare
    name), the lines it calls on from, its hop in ``called_from``, how its runs are asked for the row (*how*: ``marks``,
    its marks holding the key; ``key``, the key alone, a row of it printing a key column; ``none``) and the hop-1 sites
    between, a facade's or a plugin's dispatch, whose way no run shows."""

    owner: str
    lines: list[int]
    hop: int
    how: str
    marks: tuple[str, ...] = ()
    between: list[str] = field(default_factory=list)
    #: U2X (A): the lines of *lines* whose call reaches the mover across a send (SND), and the sends (``file:line``)
    across: list[int] = field(default_factory=list)
    sent: list[str] = field(default_factory=list)
    #: H-3 (P15): each route's outermost hop (``file:line``) with the line of *lines* it reaches the mover by
    outer: list[tuple[str, int]] = field(default_factory=list)


@dataclass
class Callers:
    """Why a mover's passes did not take the row in *window*: its callers' passes of the key there (U1; KNOWN_ISSUES.md
    K55).  *kind* is the stop it goes on from: ``before``, a late mover's passes before the move (K45); ``window``, a
    mover's passes in the window; ``statement``, a statement of the mover's that took the row only at the move (L2's
    judgement 1); ``after``, a stuck value's mover, which runs on the row only as its callers hand it, after its last
    pass on the row (I-2).  *entries* are the mover's ``called_from``; *up*, the hop-1 caller U1 went up from (once:
    the map lists two hops)."""

    mover: str
    arms: list[Arm]
    subject: str
    key: str
    window: tuple[str, str]
    kind: str
    entries: list[str]
    before: str | None = None
    up: str | None = None
    #: I-2, kind ``after``: the end of the mover's last pass on the row (to the microsecond; the window opens at its
    #: second), and what S2 read of its passes
    after: str | None = None
    left: T.Said | None = None


@dataclass
class OtherRows:
    """What other rows a selection of the mover's waited on (O1, design H; KNOWN_ISSUES.md K77): its statements'
    ``NOT EXISTS`` terms read against the rows file's linked rows and their histories, from *since* to *before* (a late
    value's move) or *until* (the window's end).  *then* is what is read where H does not hold: the mover's callers
    (U1), or the stop it said."""

    owner: str
    subject: str
    since: str
    before: str | None
    until: str
    then: Callers | T.Said


@dataclass
class Turned:
    """When the mover's statements that took the row turned for it (L4, out/gm/qk-design.md, next 2): *span* is the last
    stretch before the move where a term of each of them was surely false for the row, read with the arms *assumed*
    names as having not written the columns compared; *then* is what is read where it does not answer -- what other rows
    the selection waited on (O1), the callers (U1), or the stop it said."""

    q: Missed
    took: list
    span: tuple[datetime.datetime, datetime.datetime]
    assumed: dict[str, list[str]]
    #: the trees false all of *span*, by their entries
    words: list[str]
    then: OtherRows | Callers | T.Said


@dataclass
class CallerPasses:
    """Which way each pass of one caller of the mover went on the row (U2): through its call into the mover, or not."""

    caller: Caller
    callers: Callers


@dataclass
class ArmCaller:
    """Which call of its function ran an arm its run shows ran, the arm with no guard of its own (U3)."""

    asked: Asked
    owner: str
    line: int
    run: Any


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
    """Why this value stayed on so long: what moved it on in the end, and when.  *since* names the stay asked, by the
    time of the statuslog row that begins it, where the value has more than one."""

    subject: str
    observed: str
    key: str
    since: str | None = None


@dataclass
class Wrote:
    """Who wrote a value, as H1 and H2 answer it: the map's readings, the run of the key at the time, the pick."""

    readings: list
    question: object  # the run
    cands: list
    env: object
    chosen: Verdict
    #: the caller's verdict where *chosen* is the arm of a writer that prints nothing, read on that caller's run at its
    #: call (:class:`StandIn`): the run's marks and the guard's record are the call's, the guards the arm's
    via: Verdict | None = None
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


def _element_rows(data: dict, cls: str) -> dict[str, str]:
    """The rows file's collections of a *cls* row, as attribute -> the name its element rows go by (``Files`` ->
    ``file``): the class a linked row of *cls* is, named by ``classes``, else by the spelling of its class's name
    without ``Spec`` (``FileSpec`` -> ``file``) where the rows file spells one."""
    classes = {c: e for e, c in (data.get("classes") or {}).items() if e != "about"}
    spelled = {e for e in (data.get("spellings") or {}) if e != "about"}
    out = {}
    for text in data.get("collections") or {}:
        if text == "about" or "." not in text:
            continue
        attr = text.rpartition(".")[2]
        for linked in data.get("linked") or []:
            if str((linked.get("via") or {}).get("from", "")).partition(".")[0] != cls:
                continue
            name = classes.get(linked.get("class", ""))
            guess = str(linked.get("class", "")).removesuffix("Spec").lower()
            if name or guess in spelled:
                out[attr] = name or guess
    return out


def _guard_read(text: str, env: E.Env) -> object:
    """A path condition read in *env*: True or False where the values read decide it; anything else is not read, one
    whose names are as they are at another line (``E.read_elsewhere``) among them."""
    if E.read_elsewhere(text):
        return E.UNKNOWN
    try:
        return E.evaluate(text, env)
    except SyntaxError:
        return E.UNKNOWN


def _at_its_line(text: str) -> str:
    """A guard without the ``  [at N]`` that says its names are as they are when line N's test runs: its text as that
    ``if`` reads it."""
    return re.sub(r"\s+\[at \d+\]$", "", text)


def _orders_texts(text: str, env: E.Env) -> bool:
    """Whether condition *text* orders (<, >) a value *env* holds as text: a printed time read back is a string, and
    text order is not the time's order ('...13:40:26.260951' and '...12:45:38')."""
    try:
        tree = ast.parse(E.strip_annotation(text), mode="eval")
    except SyntaxError:
        return False
    return any(isinstance(n, ast.Compare) and any(isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)) for op in n.ops)
               and any(isinstance(env.lookup(ast.unparse(x)), str) for x in [n.left, *n.comparators])
               for n in ast.walk(tree))


@dataclass(frozen=True)
class Recorder:
    """A call of an arm's pass that records the object's history row after the arm (KNOWN_ISSUES.md K67): a history
    pair of the map's on the arm's object (``attr_time.arms``'s ``placed``) whose call runs after the arm -- the
    function the call is in, its line, each method its ``writes`` name with the lines of its history write, and the
    calls of the function on the way to the arm (``HistoryPair.point``: where a stand-in is read, ``_h2_candidates``)."""

    caller: str  # ``file::Class.function``
    call: int
    methods: tuple[tuple[str, tuple[int, ...]], ...]
    points: tuple[int, ...]


@dataclass(frozen=True)
class _ArmAt:
    """An arm as ``attr_time.arms`` takes a reading: its ``file::function`` and its line."""

    owner: str
    lines: tuple[int, ...]


def _during(inner: list[Run], p: Run, at: datetime.datetime) -> bool:
    """Whether the recorder's calls *inner* ran inside the pass *p* on its machine and one of them was running at *at*,
    the row's time: from its first record to the pass's next record after it (the call returned by then) --
    updateJob's call of the adder's at 'transferring', 08:25:56.779 .. its next record 56.863, the row 56.844 between,
    printing nothing after its UPDATE (KNOWN_ISSUES.md K76).  A call of the method at another time inside a long pass is
    not the one that recorded the row (MJ-holding-late's adder pass from 18:08:55.406, the row's recorder the pilot's
    updateJobStatus)."""
    if not inner or any(m.machine != p.machine or W.when(m.first) < W.when(p.first) or W.when(m.last) > W.when(p.last)
                        for m in inner):
        return False

    def returned(m: Run) -> datetime.datetime:
        return next((W.when(r.stamp) for r in p.records if W.when(r.stamp) > W.when(m.last)), W.when(p.last))

    return any(W.when(m.first) <= at <= returned(m) for m in inner)


def _stamp(t: datetime.datetime) -> str:
    """*t* as a record's stamp, ``2026-01-01 00:00:00,123``: a statuslog row's time compared with the records around it."""
    return f"{t:%Y-%m-%d %H:%M:%S},{t.microsecond // 1000:03d}"


def recorders(subject: str, owner: str, line: int) -> list[Recorder]:
    """The calls that record the history row of *owner*'s arm at *line* after it, in its pass (:class:`Recorder`)."""
    found: dict[tuple, list[int]] = {}
    for a in AT.arms(mapdump.fragment(), subject, [_ArmAt(owner, (line,))]):
        for p in a.placed:
            if p.order != HISTORY_AFTER or not p.writes:
                continue
            methods: dict[str, list[int]] = {}
            for w in p.writes:
                method, _, at = w.rpartition(":")
                if at.isdigit():
                    methods.setdefault(method, []).append(int(at))
            if methods:
                points = found.setdefault((p.caller, p.call, tuple((m, tuple(ls)) for m, ls in methods.items())), [])
                points += [p.point] if p.point not in points else []
    return [Recorder(caller, call, methods, tuple(points)) for (caller, call, methods), points in found.items()]


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
        # read at the arm: another function's hole only where the map hands its object here (KNOWN_ISSUES.md K63)
        owner, _, line = self.arm.rpartition(":")
        at = env.reading(owner, int(line), arm=(owner, int(line))) if line.isdigit() else env
        for g in self.guards:
            got = _guard_read(g, at)
            env.used.update(at.shared())
            if got is False:
                return None
            if got is not True:
                out.append(Premise(T.Said("its-condition-not-read", {"guard": g}), NOT_READ))
        if self.level > 1:
            out.append(Premise(T.Said("functions-between-not-read", {"n": self.level - 1}), NOT_READ))
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
    #: the arm as the steps after H2 read it where it is one of several (:class:`WroteBy`): on its run, shown or not, or
    #: on no run (``question`` None)
    branch: Wrote | None = None


@dataclass
class OneOf:
    """What H2 leaves where no arm is shown to write with nothing newer that may have (KNOWN_ISSUES.md K38): the arms
    that wrote or may have, one :class:`Possible` per row, newest first, over *rows* statuslog rows asked."""

    left: list[Possible]
    rows: int


@dataclass
class WroteBy:
    """H3 on, for the arm H2 chose -- or for one it left (``Rerun.arms_left``), with what it rests on (*premises*) and the
    older rows it may have written at (*older*), read where its guard reads false at this one."""

    asked: Asked
    wrote: Wrote
    premises: list[str] | None = None
    older: list[Possible] = field(default_factory=list)
    #: the one writer of an attribute's value H2 left, the others dropped by no guard (``Rerun.arms_left``): its guards
    #: the values leave unread held, the only way to the value (KNOWN_ISSUES.md K66)
    only_way: bool = False


@dataclass
class ArmsLeft:
    """Who wrote the value, where H2 decides nothing (:class:`OneOf`): each arm it left, walked on."""

    asked: Asked
    one_of: OneOf


class TookBy(NamedTuple):
    """Which of a mover's selecting statements took the row in the pass that moved it (:meth:`Rerun._took_by`)."""

    statement: Any
    #: the line of its print that named the row's key in the pass; None where the others were ruled out instead
    line: int | None
    #: the print's text; where *line* is None, why the others handed the row on to no write
    text: str
    #: ``(line, why)`` of each statement ruled out by its hand-ons (:meth:`Rerun._handed_none`)
    ruled_out: tuple[tuple[int, str], ...] = ()
    #: the print's record in the moving pass (P25 (i) reads the counts it printed); None where *line* is None
    record: Any = None
    #: the print's row the record matched (its holes)
    row: Any = None


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
    #: the logs the map's reading names, for a mover whose prints it places nowhere (judgement 3, as S1's)
    reader_log_files: list = field(default_factory=list)
    #: where the rows of the statements selecting the value go (:mod:`value_reach`); None: not read
    reach: VR.Reach | None = None
    #: what the amounts a statement's runs print are asked over (:meth:`Rerun._fill`): stuck's S1 window, read at one
    #: moment; None: the part of [since, moved] judgement 3 asks first (``W.last_part``)
    amounts: W.Asking | None = None
    #: when the stay began, where the window is one moment (stuck's as_of, ``since`` = ``moved``): the observed value's
    #: last statuslog row (:meth:`Rerun._stamps_itself`); None: ``since``
    began: datetime.datetime | None = None


@dataclass
class Ordered:
    """A statement returning only the first rows of an order, and the call of it that moved the value on."""

    statement: object
    run: object
    q: Missed


#: the clocks the map names that read one UTC clock: PanDA's Oracle session runs in UTC (WrappedCursor.initialize,
#: ``ALTER SESSION SET TIME_ZONE='UTC'``); SYSDATE, SYSTIMESTAMP and datetime.now read a host's zone, unknown here
UTC_CLOCKS = {"naive_utcnow", "datetime.utcnow", "datetime.now(utc)", "CURRENT_DATE", "CURRENT_TIMESTAMP"}
#: K-1: how far the records' clock (a host's, printing the write) and the database's (CURRENT_DATE as it wrote) may be
#: apart when a row's value is matched to the write the records show (:meth:`Rerun._row_value_write`)
CLOCKS_APART = datetime.timedelta(seconds=1)


def _clock_write(value: datetime.datetime, writes: Sequence[TL.Write],
                 offsets: dict[str, datetime.timedelta | None]) -> tuple[TL.Write, tuple[str, ...]] | None:
    """K-1 (:meth:`Rerun._row_value_write`) and K-3 (b) (:meth:`Rerun._printed_witnesses`): the last of *writes* by an
    arm writing the clock with an offset (*offsets*, by "func:line") whose clock then is *value* to its second
    (:data:`CLOCKS_APART`), ran or reached, with no write run after it -- and the arms of it that match; None where a
    write run after any match comes first (what it wrote is not the value), or none matches."""
    for w in sorted(writes, key=lambda w: w.at, reverse=True):
        hits = tuple(n for n in w.arms if (off := offsets.get(n)) is not None
                     and value - CLOCKS_APART <= w.at + off < value + datetime.timedelta(seconds=1) + CLOCKS_APART)
        if hits:
            return w, hits
        if w.how == TL.RAN:
            return None
    return None


def _picks_beyond_key(owner: str, line: int, subject: str, keys: set[str]) -> bool:
    """K-3 (e): whether a statement writing *subject* by *owner*'s arm at *line* picks its row by a term on another
    column than the row's key (*keys*; the map's ``KeyFrom`` terms): its run is no write of the row unless the term held,
    which the run alone does not show (lockTask_JEDI's 2766 ``WHERE jediTaskID=:jediTaskID AND lockedBy=:lockedBy``,
    updateDatasetsToFinishTask_JEDI's release ``lockedBy=:lockedBy``)."""
    return any(getattr(kf, "term", "") and getattr(kf, "attr", "") not in keys
               for kf in key_froms() if _bare(kf.owner) == _bare(owner) and line in (kf.writes.get(subject) or []))


@functools.cache
def _print_holes_of(fragment_id: int) -> dict[tuple[str, str, int], list[tuple[str, tuple[str, ...]]]]:
    """K-3 (b): the holes of the installed dump's prints the map names columns for (``SkeletonHole.columns``), one value
    each (not ``listed``), by ``(file, function, the print's line)``: ``(expression, columns)``."""
    out: dict[tuple[str, str, int], list[tuple[str, tuple[str, ...]]]] = {}
    for f in getattr(mapdump.fragment(), "function_prints", None) or []:
        rel, _, func = _bare(f.owner).partition("::")
        for st in f.statements or []:
            for r in st.rows or []:
                for h in [*(r.holes or []), *(getattr(r, "prefix_holes", None) or [])]:
                    if getattr(h, "columns", None) and h.expression and not getattr(h, "listed", False):
                        out.setdefault((rel.rpartition(":")[2], func.rpartition(".")[2], st.line), []).append(
                            (h.expression, tuple(h.columns)))
    return out


@functools.cache
def _key_hole_lines(fragment_id: int) -> dict[tuple[str, str], dict[int, set[str]]]:
    """P9: the installed dump's prints by ``(file, function)``: each print's line with the columns its holes print,
    a list of them (``listed``) too."""
    out: dict[tuple[str, str], dict[int, set[str]]] = {}
    for f in getattr(mapdump.fragment(), "function_prints", None) or []:
        rel, _, func = _bare(f.owner).partition("::")
        lines = out.setdefault((rel.rpartition(":")[2], func.rpartition(".")[2]), {})
        for st in f.statements or []:
            for r in st.rows or []:
                for h in [*(r.holes or []), *(getattr(r, "prefix_holes", None) or [])]:
                    if getattr(h, "columns", None):
                        lines.setdefault(st.line, set()).update(h.columns)
    return out


def _printed_columns(owner: str, line: int, row, record, cls: str, keys: set[str]) -> list[tuple[str, str]]:
    """K-3 (b): ``(column, value)`` for each hole of *owner*'s print at *line* the map names a column of *cls* other
    than *keys* (the row's key), read off *record* -- the row's value of that column where the pass read it
    (unlockTasks_JEDI's 897 prints the lockedBy and lockedTime its select read, ``unlocked jediTaskID=...
    lockedBy=... lockedTime=...``)."""
    fragment = mapdump.fragment()
    if fragment is None:
        return []
    rel, _, func = _bare(owner).partition("::")
    holes = _print_holes_of(id(fragment)).get((rel.rpartition(":")[2], func.rpartition(".")[2], line))
    if not holes:
        return []
    captured = row.capture(record)
    return [(c, str(value).strip().strip("'\"")) for expr, columns in holes if (value := captured.get(expr)) is not None
            for c in columns if c.partition(".")[0] == cls and c not in keys]


#: why a term on a column the row holds NULL fails (SQL's NULL; ``_leaf``, ``_pair_fails``)
#: movers_of's word for a mover the map's guarded_by gives (a writer whose write requires the value)
GUARDED = "its write requires the value"
#: how far back L3 asks a row's entry into the value it held at the move (``parts.HeldAt``): copyArchive.py:509-512
#: kills an activated job created more than 7 days before, and D's group waited up to 169.1 h (its W2 answer).  A row
#: that entered earlier is not counted; an earliest entry within an hour of the bound says the count is "at least"
L3_LOOKBACK = datetime.timedelta(days=8)


def _identity_attrs(cls: str) -> list[str]:
    """The ``Class.attr`` the dump's tables name *cls*'s identity by (``SR.identity_attrs``)."""
    return SR.identity_attrs(cls)


def key_marks(owner: str, line: int, cls: str, key: str) -> tuple[str, ...]:
    """What *owner*'s lines carry where its run's mark holds *cls*'s key *key*: for a run the map records for the
    function (``FunctionPrintsNode``'s ``runs``) whose mark's key holds the class's identity (``_identity_attrs``),
    each printed prefix of its rows that has the mark's holes (``.*?``, in the mark's order) cut to the text around the
    key's hole, the key in it -- "changeTaskSplitRulePanda < jediTaskID=53351604 >", "; jediTaskID=53351078;
    siteID=".  Every line of such a run holds it, and no line of another function's run does, so asked over a span it
    reads the writer's runs of the key alone (F3, KNOWN_ISSUES.md K37).  None for a function marking no run by the
    class's key: doForPriorityMassage's "label=user"; changeTaskPriorityPanda's "<jediTaskID=...>" for a job."""
    ident = set(_identity_attrs(cls))
    return marks_holding(owner, line, lambda k: bool(ident & set(k.columns)), key) if ident else ()


def referencing(cls: str) -> dict[str, list[str]]:
    """E-1 (W8; out/gm/qk-design.md's bundle E, tmaeno's OK 2026-11-17 13:42; the map's ``TableRecord.references``):
    each ``Class.attr`` whose column holds, by the foreign keys the schema declares, a value of *cls*'s identity, with
    the ``table.column -> table.column`` steps it goes through -- ``JediDatasetSpec.jediTaskID`` holds a
    ``JediTaskSpec.jediTaskID`` (jedi_datasets.jeditaskid -> jedi_tasks.jeditaskid); a composite key is followed a
    column at a time.  Empty where the dump declares none."""
    ident = set(_identity_attrs(cls))
    edges: dict[str, list[tuple[str, str]]] = {}
    for t in getattr(mapdump.fragment(), "tables", None) or []:
        for ref in getattr(t, "references", None) or []:
            for i, attr in enumerate(ref.attrs or []):
                if attr and i < len(ref.table_attrs or []) and ref.table_attrs[i]:
                    edges.setdefault(attr, []).append(
                        (ref.table_attrs[i], f"{t.name}.{ref.columns[i]} -> {ref.table}.{ref.table_columns[i]}"))
    out: dict[str, list[str]] = {}
    for start in edges:
        stack: list[tuple[str, list[str]]] = [(start, [])]
        seen = {start}
        while stack:
            attr, way = stack.pop()
            for to, step in edges.get(attr, []):
                if to in ident:
                    out.setdefault(start, [*way, step])
                elif to not in seen:
                    seen.add(to)
                    stack.append((to, [*way, step]))
    return out


def ref_marks(owner: str, line: int, cls: str, key: str) -> tuple[tuple[str, ...], list[str]]:
    """E-1: :func:`key_marks` where the run's mark holds a column that references *cls*'s identity
    (:func:`referencing`) instead of the identity itself -- insertFilesForDataset_JEDI's ``jediTaskID={} datasetID={}``,
    ``datasetSpec.jediTaskID``: the dataset's task.  ``(the marks, [the attr, the foreign keys it goes through])``."""
    refs = referencing(cls)
    if not refs:
        return (), []
    held: list[str] = []

    def holds(k: Any) -> bool:
        hit = sorted(set(refs) & set(k.columns))
        if hit and not held:
            held[:] = [hit[0], *refs[hit[0]]]
        return bool(hit)

    marks = marks_holding(owner, line, holds, key)
    return marks, held if marks else []


def after_marks(arm: Any, cls: str, key: str) -> tuple[str, ...]:
    """Q5 (P40(b)): what a print after *arm*'s write (``attr_time.Arm``) carries where it holds *cls*'s key *key*: for
    each print the map says the function runs after the write on every way (``AfterPrint.every_way``) with a hole in
    its message holding the written row's identity (``MustName.row`` written), the print's literal text around that
    hole, the key in it -- insertTaskParamsPanda's 3853 ``{comm_cmd} jediTaskID={jediTaskID} with {...}`` after its
    ramCount arm at 3840: "jediTaskID=53652263 with".  A pass that wrote prints it, so asked over the object's life it
    reads the writer's passes of the key, as :func:`key_marks` reads a run's mark (F3; out/gm/qk-design.md, the H2-2
    bundle's Q5, tmaeno's OK 2026-11-17 09:4x).  Empty where no such print is read."""
    ident = set(_identity_attrs(cls))
    out: set[str] = set()
    for p in getattr(arm, "after_prints", None) or ():
        exprs = {n.expression for n in p.names
                 if n.row == MUST_ROW_WRITTEN and n.column in ident and getattr(n, "part", "") == "message"}
        if not getattr(p, "every_way", False) or not exprs:
            continue
        found = R._prints_of(arm.owner, [p.line])
        for st in getattr(found, "statements", None) or []:
            for row in st.rows or [] if st.line == p.line else []:
                for h in row.holes or []:
                    if h.expression in exprs and (text := _around_group(row.capture or "", h.group, key)):
                        out.add(text)
    return tuple(sorted(out))


def _around_group(capture: str, group: int, value: str) -> str:
    """*capture*'s literal text either side of its *group*-th group, *value* in the group, stripped; "" where the
    capture has a group of another kind (``(?``) or not that many, or the text is *value* alone."""
    if "(?" in capture:
        return ""
    parts, cur, depth, i = [], "", 0, 0
    while i < len(capture):
        ch = capture[i]
        if ch == "\\":
            cur += capture[i:i + 2] if depth == 0 else ""
            i += 2
            continue
        if ch == "(":
            if depth == 0:
                parts.append(cur)
                cur = ""
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0:
            cur += ch
        i += 1
    parts.append(cur)
    if not 1 <= group < len(parts):
        return ""
    left, right = (re.sub(r"\\(.)", r"\1", s) for s in (parts[group - 1], parts[group]))
    text = (left + value + right).strip()
    return text if text != value else ""


def marks_holding(owner: str, line: int, holds: Callable[[Any], bool], value: str) -> tuple[str, ...]:
    """What *owner*'s lines carry where its run's mark holds *value* in the first hole of the mark's key that *holds*
    picks: each printed prefix of its rows that has the mark's holes (``.*?``, in the mark's order) cut to the text
    around that hole, *value* in it (:func:`key_marks`, the hole holding the class's identity)."""
    found = R._prints_of(owner, [line])
    if found is None:
        return ()
    prefixes = {r.prefix for st in (getattr(found, "statements", None) or []) for r in st.rows or [] if r.prefix}
    out = set()
    for run in getattr(found, "runs", None) or []:
        at = [i for i, k in enumerate(run.key or []) if holds(k)]
        for prefix in prefixes if at else ():
            parts = [re.sub(r"\\(.)", r"\1", p) for p in prefix.split(".*?")]
            if len(parts) != len(run.key) + 1 or any(p not in "".join(parts) for p in run.mark.split("{}")):
                continue
            text = (parts[at[0]] + value + parts[at[0] + 1]).strip()
            if text != value:
                out.add(text)
    return tuple(sorted(out))


def mark_values(token: str, mark: str) -> list[str] | None:
    """The values a run's *token* holds in the holes of the map's *mark* for it, in the mark's order -- "PandaID={}"
    in "updateUnmergedJobs < PandaID=7397569813 >" holds 7397569813 -- or None where the token has no such mark."""
    parts = mark.split("{}")
    m = re.search("(.+?)".join(re.escape(p) for p in parts) + r"\s*>?\s*$", token)
    return [v.strip() for v in m.groups()] if m and len(parts) > 1 else None


def mark_identities(owner: str, line: int, run: Any) -> list[tuple[str, str]]:
    """Which rows *run*'s mark holds, as the map's runs of *owner*'s function read it: each hole whose columns hold a
    class's identity (the dump's tables, ``JobSpec.PandaID``), as the class with the value *run*'s token shows there --
    whatever the hole's expression spells (``job.PandaID``, ``pandaID``).  Empty where no mark of the function's reads
    in the token (a logger the caller hands in marks the run by the caller's: doBasicPostProcess's tmpLog), or none of
    its holes holds an identity."""
    prints = R._prints_of(owner, [line])
    out: list[tuple[str, str]] = []
    for r in getattr(prints, "runs", None) or []:
        vs = mark_values(run.token, r.mark)
        if vs is None or len(vs) != len(r.key or []):
            continue
        for k, v in zip(r.key, vs, strict=True):
            for column in k.columns or []:
                cls = column.partition(".")[0]
                if column in _identity_attrs(cls) and (cls, v) not in out:
                    out.append((cls, v))
    return out


def print_heads(owner: str, lines: list[int]) -> tuple[str, ...]:
    """The fixed heads of the marks *owner*'s function prints under, as the map's prints read them: each printed prefix
    up to its first hole, unescaped (``kickExhaustedTasks_JEDI < vo=``) -- the strings that ask a mark holding no key
    on its own (P29's B)."""
    return tuple(sorted({re.sub(r"\\(.)", r"\1", r.prefix.split(".*?")[0])
                         for st in getattr(R._prints_of(owner, lines), "statements", None) or []
                         for r in st.rows or [] if getattr(r, "prefix", None)}))


def mark_columns(owner: str, lines: list[int], run: Any) -> dict[str, str]:
    """What *run*'s mark holds, by the columns the map's runs of *owner*'s function name each hole's:
    ``insertNewJob < <JediTaskID=53004228 idPool=885 >`` holds 53004228 for ``JobSpec.jediTaskID`` (KNOWN_ISSUES.md
    K67).  Empty where no mark of the function's reads in the token."""
    out: dict[str, str] = {}
    for r in getattr(R._prints_of(owner, lines), "runs", None) or []:
        vs = mark_values(run.token, r.mark)
        if vs is None or len(vs) != len(r.key or []):
            continue
        for k, v in zip(r.key, vs, strict=True):
            for col in k.columns or []:
                out.setdefault(col, v)
    return out


def printed_parameters(owner: str, lines: list[int]) -> set[str]:
    """The parameters of *owner*'s function a hole of its runs' marks holds as the call hands them, by the map
    (``RunKey.parameter``, its batch 5 M-c): ``reactivatePendingTasks_JEDI.timeLimit`` in ``limit={}``.  Empty for a
    map that does not say (before batch 5)."""
    return {p for r in getattr(R._prints_of(owner, lines), "runs", None) or [] for k in r.key or []
            if (p := getattr(k, "parameter", ""))}


def mark_parameters(owner: str, lines: list[int], run: Any) -> dict[str, str]:
    """What *run*'s mark holds, by the parameter :func:`printed_parameters` names each hole's:
    ``reactivatePendingTasks_JEDI < vo=atlas label=... limit=20 min timeout=48hours minPrio=None >`` holds 20 for
    ``reactivatePendingTasks_JEDI.timeLimit``.  Empty where no mark of the function's reads in the token."""
    out: dict[str, str] = {}
    for r in getattr(R._prints_of(owner, lines), "runs", None) or []:
        vs = mark_values(run.token, r.mark)
        if vs is None or len(vs) != len(r.key or []):
            continue
        for k, v in zip(r.key, vs, strict=True):
            if parameter := getattr(k, "parameter", ""):
                out.setdefault(parameter, v)
    return out


def _pass_marks(owner: str, line: int, values: dict[str, str]) -> list[tuple[str, tuple[str, ...]]]:
    """``[(value, marks)]``: what *owner*'s lines carry where its run's mark holds one of *values* (by column) -- a hole
    of the map's mark whose columns name it (``JobSpec.jediTaskID`` in runImpl's ``<jediTaskID={} datasetID={}>``), the
    value in it (:func:`marks_holding`; KNOWN_ISSUES.md K67)."""
    by: dict[str, set[str]] = {}
    for run in getattr(R._prints_of(owner, [line]), "runs", None) or []:
        for k in run.key or []:
            for col in k.columns or []:
                if (value := values.get(col)) is not None:
                    by.setdefault(value, set()).update(marks_holding(owner, line, _names_column(col), value))
    return [(v, tuple(sorted(ms))) for v, ms in sorted(by.items()) if ms]


def prints_key_before(owner: str, lines: list[int], columns: set[str]) -> bool:
    """Whether every run of *owner* that reaches its history writes at *lines* prints the key first (KNOWN_ISSUES.md
    K67): before each, a print every path runs (``rows.must_rows``) with a hole naming one of *columns*, or any such
    print where every run mark of the function holds one -- each line of its run carries the key (updateTask_JEDI's
    ``jediTaskID={}``).  insertNewJob's "inserted {PandaID}" (3199) runs before both of its writes."""
    rel, _, func = owner.partition("::")
    found = R._prints_of(owner, lines)
    if found is None or not lines:
        return False
    stmts = getattr(found, "statements", None) or []
    runs = getattr(found, "runs", None) or []
    marked = bool(runs) and all(any(columns & set(k.columns or []) for k in r.key or []) for r in runs)
    for line in lines:
        fn = R.function(rel, func, [line])
        stmt = R.statement_at(fn.node, line)
        if stmt is None:
            return False
        before = {s.lineno for s in R.must_rows(fn.node, stmt)[0]}
        printed = [st for st in stmts if st.line in before]
        named = any(columns & {c for r in st.rows or [] for h in [*(r.holes or []), *(r.prefix_holes or [])] for c in h.columns or []}
                    for st in printed)
        if not named and not (marked and printed):
            return False
    return True


def _names_column(col: str) -> Callable[[Any], bool]:
    """Whether a mark's hole names the column *col* (:func:`marks_holding`'s *holds*)."""
    return lambda hole: col in (hole.columns or [])


def prints_key(owner: str, lines, columns: set[str]) -> bool:
    """Whether a row of *owner*'s prints a hole the map says holds one of *columns*, the row's key, in its message or
    its mark: a run of it is found there by the key alone, though a line of it without the key is not (as K45's mover
    whose mark holds none).  AtlasAnalWatchDog.doForPreStaging's "action=release_... jediTaskID=..."."""
    found = R._prints_of(owner, lines)
    return found is not None and any(
        columns & set(h.columns or [])
        for st in found.statements for r in st.rows or [] for h in [*(r.holes or []), *(r.prefix_holes or [])])


def _entries_of(writers: list, owner: str) -> list[str]:
    """The ``called_from`` entries of *owner*'s junctions among *writers*, each once."""
    return list(dict.fromkeys(e for w in writers if w.owner == owner for e in getattr(w, "called_from", None) or []))


def _site(hop: str) -> str | None:
    """The ``file:line`` a ``called_from`` hop starts with, without what the map writes after it."""
    m = re.match(r"[\w./-]+\.py:\d+", hop.strip())
    return m[0] if m else None


#: what a run prints of an empty container
_EMPTY_PRINTS = ("{}", "[]", "()", "set()")


def _printed_value(printed: str) -> Any:
    """What a run printed, as Python reads its repr (Q6): a literal it parses, ``set()`` the empty set; ``E.UNKNOWN``
    for any other print."""
    v = printed.strip()
    if v == "set()":
        return set()
    try:
        return ast.literal_eval(v)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return E.UNKNOWN


def _printed_empty(printed: str) -> bool | None:
    """Q6's (b): whether a container's print says it was empty: True for ``{}``, ``[]``, ``()``, ``set()``; False for
    another a container's repr starts as (one that does not parse holds a value no literal is); None for a print that
    is no container's."""
    v = printed.strip()
    if v in _EMPTY_PRINTS:
        return True
    got = _printed_value(v)
    if isinstance(got, (dict, list, tuple, set, frozenset)):
        return not got
    return False if v[:1] in ("{", "[", "(") or v.startswith(("set(", "frozenset(")) else None


def _short_circuit(test: ast.expr, term: str, printed: dict[str, tuple[str, str]]) -> str | None:
    """Q6's (c): where *test* reads *term* only in operands of an ``and`` or ``or`` after one whose value on the run
    decided it -- False for ``and``, True for ``or``; *printed* what the run printed of each term at the test, by term
    -- that operand's text: Python did not read the ones after it.  None where it read *term*, or the values do not
    tell: an operand whose value is not known stops the reading there."""
    env = E.Env()
    for name, (value, where) in printed.items():
        if (got := _printed_value(value)) is not E.UNKNOWN:
            env.put(name, got, where)
    skipped: dict[int, str] = {}
    for n in ast.walk(test):
        if not isinstance(n, ast.BoolOp):
            continue
        for i, v in enumerate(n.values[:-1]):
            try:
                got = E.evaluate(v, env)
            except Exception:  # noqa: BLE001 -- an operand the evaluator cannot read does not tell
                got = E.UNKNOWN
            if got is E.UNKNOWN or not isinstance(got, (bool, int, float, str, dict, list, tuple, set, frozenset, type(None))):
                break
            if bool(got) is isinstance(n.op, ast.Or):
                for later in n.values[i + 1:]:
                    for x in ast.walk(later):
                        skipped.setdefault(id(x), ast.unparse(v))
                break
    names = [x for x in ast.walk(test) if isinstance(x, (ast.Name, ast.Attribute)) and ast.unparse(x) == term]
    return skipped[id(names[0])] if names and all(id(x) in skipped for x in names) else None


def _held_test(guard: str) -> ast.expr | None:
    """Q7's (f): a guard as its run held it -- its text (``not (...)`` for an else), the ``[at N]`` of a test read at
    another line dropped -- parsed; None where it does not parse."""
    try:
        return ast.parse(re.sub(r"\s+\[at \d+\]$", "", guard), mode="eval").body
    except SyntaxError:
        return None


def _held_rules_out(test: ast.expr | None, name: str, values: tuple[str, ...]) -> bool:
    """Q7's (f): whether *name* (a term, or a part of one: ``job.jobStatus``) holding each of *values* (literals as
    written) would turn *test*, which its run held, false -- read by the evaluator, the test's other names unknown;
    False where a value or the test does not tell."""
    if test is None or not values:
        return False
    for v in values:
        try:
            literal = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return False
        env = E.Env()
        env.put(name, literal, "Q7's (f)")
        try:
            got = E.evaluate(test, env)
        except Exception:  # noqa: BLE001 -- a test the evaluator cannot read does not tell
            return False
        if got is E.UNKNOWN or not isinstance(got, (bool, int, float, str, type(None))) or bool(got):
            return False
    return True


def _way_text(e: SR.End) -> str:
    """One way's end for the walk's lines: its end, why, where, the calls a value came back by (out/gm/gm5-design.md
    G2), the calls it came in by, and the changes on it between there and the read, each with the calls it was made
    through (G4)."""
    return (f"{e.end} ({e.why}{' at ' + e.at if e.at else ''})"
            + (f": {e.decided!r} decided the test on its run" if e.decided else "")
            + (f": its {e.part} as written there, of the object made at {e.made_at}" if e.part else "")
            + (f", in the except handler at {e.handler} (S3)" if e.handler else "")
            + (f", read off the exception the except handler at {e.caught} caught (K)" if e.caught else "")
            + (f" returned{f' from callee line {e.returned_line}' if e.returned_line else ''} by {' -> '.join(e.returned)}"
               if e.returned else "")
            + (f" by {' -> '.join(e.route)}" if len(e.route) > 1 else "")
            + (f" via {' -> '.join(e.via)}" if e.via else "")
            + (", changed between it and the read: " + "; ".join(_change_text(c) for c in e.changes) if e.changes else "")
            + (f", the constants {e.split.call} hands told apart by the if at {', '.join(e.split.ifs)} (S2's (a))"
               if e.split is not None else ""))


def _change_text(c: SR.Changed) -> str:
    """A change on a way by its place and the calls it was made through (G4)."""
    return c.at + (f" made by {' -> '.join(c.made_by)}" if c.made_by else "")


def _a_change(c: SR.Changed) -> T.Said:
    return T.Said("a-change", {"at": c.at, "key": c.key, "made": T.Said("made-by", {"via": list(c.made_by)}) if c.made_by else ""})


def _split_asked(asked: str) -> T.Said:
    """What a split says of its caller's run (``SR.Split.asked``): not asked, or what Q5's ask of it answered."""
    return T.Said(f"split-run-{asked or 'not-asked'}", {})


def _way_said(e: SR.End) -> T.Said:
    """One way's end as result.json keeps it: each place a ``file:line`` as it is, its words in the templates (G3);
    each change on it by its place, key and the calls it was made through (G4); the one S4c chose, with its site read
    as an arm; the ``except`` handler the statement it ends at is in (S3's)."""
    returned = T.Said("returned-by", {
        "route": list(e.returned),
        "line": T.Said("returned-from", {"line": e.returned_line}) if e.returned_line else ""}) if e.returned else ""
    changes = [_a_change(c) for c in e.changes]
    m = e.made
    changed: T.Said | str = T.Said("changed-between", {"changes": changes}) if changes else ""
    how = (T.Said("change-shown-by-its-run") if m is not None and m.how == "shown"
           else T.Said("change-left-start-same" if m.printed_start else "change-left-start-other",
                       {"part": m.part, "start": list(m.start)}) if m is not None and m.start
           else T.Said("change-left-start-unread", {
               "part": m.part, "why": T.Said("no-field-from", {"field": m.no_record}) if m.no_record else ""})
           if m is not None and m.part
           else T.Said("change-left-by-elimination"))
    if m is not None and m.entry is not None:
        changed = T.Said("made-by-a-change", {
            "change": _a_change(m.change), "how": how, "site": m.site,
            "terms": [T.Said("term-ends", {"term": u.term, "ends": [_way_said(x) for x in u.ends]}) for u in m.terms] or ["none"],
            "entry": _way_said(m.entry)})
    elif m is not None:
        changed = T.Said("made-by-a-change-unread", {"change": _a_change(m.change), "how": how, "site": m.site,
                                                     "guards": list(m.guards) or ["none"]})
    at = (T.Said("way-end-unread", {"at": e.at, "operand": e.decided}) if e.decided
          else T.Said("way-end-written", {"at": e.at, "part": e.part, "made": e.made_at}) if e.part
          else T.Said("way-end-at-caught", {"at": e.at, "handler": e.caught}) if e.caught
          else T.Said("way-end-at-handler", {"at": e.at, "handler": e.handler}) if e.handler
          else T.Said("way-end-at", {"at": e.at}) if e.at else "")
    s = e.split
    split = T.Said("told-apart-by-an-if", {
        "call": s.call, "ifs": list(s.ifs), "guards": list(s.guards), "asked": _split_asked(s.asked),
        "terms": [T.Said("term-ends", {"term": u.term, "ends": [_way_said(x) for x in u.ends]}) for u in s.terms] or ["none"],
    }) if s is not None else ""
    return T.Said("way-end", {"end": e.end, "why": e.why,
                              "at": at,
                              "returned": returned,
                              "route": T.Said("came-in-by", {"route": list(e.route)}) if len(e.route) > 1 else "",
                              "via": T.Said("bound-via", {"via": list(e.via)}) if e.via else "",
                              "changes": changed,
                              "split": split})


def _handler_line(fn: R.Function, e: SR.End) -> int:
    """The line of the ``except`` handler of *fn* the statement a way that does not end stops at is in (by
    ``pathcond.ancestors``): what it holds is bound where an exception was caught, S3's to read; 0 where none."""
    rel, _, ln = e.at.rpartition(":")
    if e.end in SR.ENDS or rel != fn.rel or not ln.isdigit():
        return 0
    # the handler's own line: the name it binds (the map's object of the class caught, EXF)
    if any(isinstance(h, ast.ExceptHandler) and h.lineno == int(ln) for h in ast.walk(fn.node)):
        return int(ln)
    stmt = R.statement_at(fn.node, int(ln))
    if stmt is None:
        return 0
    for anc in pathcond.ancestors(stmt):
        if anc is fn.node:
            break
        if isinstance(anc, ast.ExceptHandler):
            return anc.lineno
    return 0


def _handler_around(fn: R.Function, line: int) -> tuple[ast.Try, ast.ExceptHandler] | None:
    """K: the ``try`` of *fn* and its ``except`` handler whose lines (its own and its body's) hold *line*, the innermost
    where handlers nest."""
    held = [(tr, h) for tr in ast.walk(fn.node) if isinstance(tr, ast.Try)
            for h in tr.handlers if h.lineno <= line <= (h.end_lineno or h.lineno)]
    return max(held, key=lambda th: th[1].lineno) if held else None


class _Raise(TypedDict, total=False):
    """K's form: the raise an arm in an ``except`` handler stops at, as :class:`SR.Made`'s keywords (empty where its
    run's tracebacks do not show one)."""

    raised_type: str
    raised_at: str
    caught_at: str
    raised_other: str


def _raised_through(fn: R.Function, tr: ast.Try, run: Any) -> tuple[str, str, str] | None:
    """K: ``(type, file:line, other)`` of the exception *run*'s tracebacks show leaving *tr*'s body -- in each record,
    the last chain with a frame of *fn* in the body (in a ``raise ... from`` series, the one caught; the ones before are
    its causes) -- and where it was raised, the chain's last frame; *other* empty where the pin holds that frame's
    source there, else the frame's function and printed source (the run's code was another version, or outside the
    corpus).  None where no record shows one, or two show different ones."""
    lo, hi = tr.body[0].lineno, max(s.end_lineno or s.lineno for s in tr.body)
    got: set[tuple[str, str, str]] = set()
    for record in getattr(run, "records", None) or ():
        for ch in reversed(record.chains()):
            if ch.frames and any(f.in_corpus and f.same_source and f.rel == fn.rel and lo <= f.line <= hi for f in ch.frames):
                last = ch.frames[-1]
                held = last.in_corpus and last.same_source
                got.add((ch.exc_type, f"{last.rel or last.path}:{last.line}",
                         "" if held else f"{last.func}: {last.source.strip()}"))
                break
    return next(iter(got)) if len(got) == 1 else None


def _traceback_frames(run: Any) -> set[str]:
    """K: ``file:line`` of every corpus frame of *run*'s tracebacks, at the pin's lines: a call standing there ran."""
    return {f"{f.rel}:{f.line}" for record in getattr(run, "records", None) or () for ch in record.chains()
            for f in ch.frames if f.in_corpus and f.same_source}


def _printed_type(cls: str) -> str:
    """``file::Class`` as Python prints the type of its exceptions: ``pandaserver.srvcore.exceptions.FileRegistrationError``."""
    rel, _, name = cls.partition("::")
    module = rel.removesuffix(".py").removesuffix("/__init__").replace("/", ".")
    return f"{module}.{name}"


@functools.cache
def _binds_at_module(rel: str, name: str) -> bool:
    """Whether the module *rel* binds *name* itself (an import, def, class or assignment): ``Exception`` then is not
    the builtin."""
    for n in ast.walk(R.tree(rel)):
        if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
            return True
        if isinstance(n, (ast.Import, ast.ImportFrom)) and any((a.asname or a.name.split(".")[0]) == name for a in n.names):
            return True
        if isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Store):
            return True
    return False


def _in_corpus(exc_type: str) -> bool:
    """Whether the module of *exc_type* (as Python prints it) is the corpus's: a class outside it (a builtin, rucio's)
    has no corpus class among its bases."""
    module = exc_type.rpartition(".")[0].replace(".", "/")
    return bool(module) and ((R.CORPUS / f"{module}.py").exists() or (R.CORPUS / module / "__init__.py").exists())


def _catches(fn: R.Function, n: ast.expr, exc_type: str, cls: str) -> bool | None:
    """K: whether an ``except`` naming *n* catches an exception of *exc_type* (as Python prints it): a builtin class
    by ``issubclass`` where the raised one is a builtin too, else ``Exception`` and ``BaseException`` catch it; the
    map's class of the handler (*cls*, ``file::Class``) where it is *exc_type*, not where *exc_type* is outside the
    corpus; None where not told (another corpus class may be its base)."""
    caught = getattr(builtins, n.id, None) if isinstance(n, ast.Name) and not _binds_at_module(fn.rel, n.id) else None
    raised = getattr(builtins, exc_type, None) if "." not in exc_type else None
    raised = raised if isinstance(raised, type) and issubclass(raised, BaseException) else None
    if isinstance(caught, type) and issubclass(caught, BaseException):
        if raised is not None:
            return issubclass(raised, caught)
        return True if caught in (Exception, BaseException) else None
    if raised is not None:
        # a builtin exception's bases are builtins: a class of the corpus or another package is none of them
        return False
    if cls:
        return True if _printed_type(cls) == exc_type else None if _in_corpus(exc_type) else False
    return None


def _entered_handler(fn: R.Function, tr: ast.Try, exc_type: str, known: dict[int, str]) -> int:
    """K: the line of the handler of *tr* an exception of *exc_type* (as Python prints it) enters: the first one a bare
    ``except`` or a class it names catches (:func:`_catches`; *known*: the class a handler catches, ``file::Class``,
    the map's object of it); 0 where one before it may catch it, not told."""
    for h in tr.handlers:
        if h.type is None:
            return h.lineno
        names = h.type.elts if isinstance(h.type, ast.Tuple) else [h.type]
        verdicts = [_catches(fn, n, exc_type, known.get(h.lineno, "") if len(names) == 1 else "") for n in names]
        if True in verdicts:
            return h.lineno
        if None in verdicts:
            return 0
    return 0


def _binds_on_every_path(h: ast.ExceptHandler, name: str) -> bool:
    """K: whether the handler *h*'s body binds *name* at its top level, so on every path through it."""
    return any(isinstance(s, (ast.Assign, ast.AnnAssign)) and any(ast.unparse(t) == name for t in
               (s.targets if isinstance(s, ast.Assign) else [s.target])) for s in h.body)


def _first_stop(t: SR.Term) -> T.Said:
    """Where the first term that does not end stops: at itself, or, where a way of it was made by a change S4c chose
    and read as an arm, at the first term of that arm that does not end, or, where it stops at constants one call hands
    and an ``if`` tells them apart (S2's (a)), at the first term of that if that does not end -- down the arms and ifs
    so read; with the ``except`` handlers its stops are in (S3's)."""
    steps: list[T.Said] = []
    while True:
        m = next((e.made for e in t.ends if e.made is not None), None)
        s = next((e.split for e in t.ends if e.split is not None), None)
        if m is not None and (nxt := next((u for u in m.terms if not u.ended), None)) is not None:
            steps.append(T.Said("the-arm-of-a-change", {"site": m.site}))
        elif m is None and s is not None and (nxt := next((u for u in s.terms if not u.ended), None)) is not None:
            at = f"{s.fn.partition('::')[0]}:{nxt.line}" if nxt.line else s.ifs[0]
            steps.append(T.Said("the-if-telling-constants-apart", {"at": at, "call": s.call}))
        else:
            break
        t = nxt
    # K's form: the arm of the change is in an except handler and its own terms end -- it stops at the raise its run's
    # traceback shows the handler took (not an end: the raise has terms of its own)
    m = next((e.made for e in t.ends if e.made is not None), None)
    if m is not None and m.raised_at and all(u.ended for u in m.terms):
        return T.Said("first-term-stops-at-a-raise", {
            "term": t.term, "guard": t.guard,
            "made": T.Said("reached-down", {"steps": [*steps, T.Said("the-arm-of-a-change", {"site": m.site})]}),
            "type": m.raised_type, "handler": m.caught_at,
            "at": T.Said("raised-at-other", {"at": m.raised_at, "other": m.raised_other}) if m.raised_other
            else m.raised_at})
    handlers = sorted({e.handler for e in t.ends if e.handler})
    if not steps and not handlers:
        return T.Said("first-term-stops", {"term": t.term, "guard": t.guard})
    return T.Said("first-term-stops-further", {
        "term": t.term, "guard": t.guard,
        "made": T.Said("reached-down", {"steps": steps}) if steps else "",
        "s3": T.Said("bound-in-a-handler", {"lines": handlers}) if handlers else ""})


def _guard_ifs(fn: R.Function, site: ast.AST, guards: list[str]) -> list[list[ast.If]] | None:
    """The ``if`` statements each guard of *site* (``pathcond.arm_condition``'s) is read at: an ``if`` around it, or an
    early exit's own ``if`` above it (``pathcond.exits_above_at``); None where a test ``path_condition`` could not
    render, or an exit's ``if`` is not found -- the guards are not placed."""
    around: list[ast.If] = []
    previous: ast.AST = site
    for anc in pathcond.ancestors(site):
        if isinstance(anc, ast.If) and (previous in anc.body or previous in anc.orelse):
            around.append(anc)
        previous = anc
    around.reverse()
    written = pathcond.path_condition(site)
    exits = dict(pathcond.exits_above_at(site))
    ifs_of: list[list[ast.If]] = [[one] for one in around]
    for g in guards[len(written):]:
        lines = exits.get(g, ())
        ifs_of.append([n for n in ast.walk(fn.node) if isinstance(n, ast.If) and n.lineno in lines])
    if len(around) != len(written) or guards[: len(written)] != written or any(
        not at or len(at) != len(exits.get(g, ())) for g, at in zip(guards[len(written):], ifs_of[len(written):], strict=True)
    ):
        return None
    return ifs_of


def _same_function(a: str, b: str) -> bool:
    """Whether two owners name one function of one file, one as the map's reach spells it (``Class.method``), the
    other as a junction does (the bare name)."""
    (ra, _, fa), (rb, _, fb) = a.partition("::"), b.partition("::")
    return ra == rb and fa.split(".")[-1] == fb.split(".")[-1]


def _bursts(stamps: list[str], gap: datetime.timedelta) -> list[tuple[str, str]]:
    """``(first, last)`` of each run of *stamps* (sorted) cut where they stop for *gap* or more."""
    out: list[tuple[str, str]] = []
    for t in stamps:
        if out and W.when(t) - W.when(out[-1][1]) < gap:
            out[-1] = (out[-1][0], t)
        else:
            out.append((t, t))
    return out


def _callees_by_line(owner: str, depth: int = 6) -> dict[str, tuple[set[int], int]]:
    """The functions *owner* (``file::Class.function``) reaches by the map's closure within *depth* calls
    (:func:`map_reach.by_call`: each call followed for the classes it holds for, into those it runs as), each with the
    lines of *owner*'s own calls it is reached through and the fewest calls it takes: AtlasProdTaskBrokerThread.runImpl
    reaches updateTask_JEDI through 236 and 443, and through post_process_for_error's calls at 285, 300, ...; and
    PostProcessor.start's ``thr.start()`` reaches PostProcessorThread's runImpl, no other agent's."""
    out: dict[str, tuple[set[int], int]] = {}
    for call, reached in MR.by_call(owner, depth):
        lines = set(call.lines or [call.line])
        for o, n in reached.items():
            had, fewest = out.get(o, (set(), n))
            out[o] = (had | lines, min(fewest, n))
    return out


def _others_reaching(target: str, mine: set[str], depth: int = 3) -> int:
    """How many functions outside *mine* reach *target* within *depth* calls by the map's closure
    (:func:`map_reach.reaching`)."""
    return len(MR.reaching(target, depth) - mine)


def callers_of(entries: list[str], cls: str, key: str, columns: set[str]) -> list[Caller]:
    """The defs a mover's ``called_from`` *entries* call into it from, each as its runs can be asked for the row
    (KNOWN_ISSUES.md K55).  An entry is one hop (``file:line``) or two (``outer -> inner``, the inner the call into the
    mover).  Of its hops, inner first, the caller is the first whose run marks hold the key (``key_marks``), else the
    first a row of which prints a key column (:func:`prints_key`), else the innermost that prints, which can say only
    that its runs print no key.  A facade or a plugin's dispatch prints nothing or marks no key, and the caller behind
    it is the one a run shows: feed_contents_to_tasks, behind JediTaskBuffer's updateTaskStatusByContFeeder_JEDI.  An
    entry the map cut (``… N more``) names no site.  A caller is one def, not one name: datasetManager.py has seven
    ``run``, two of them (ActivatorThr's, ActivatorWithRuleThr's) calling activateJobs.  U2X (A): an entry whose
    outermost hop is an endpoint a send enters goes on once past the send (:func:`across_sends`)."""
    out: dict[tuple[str, int], Caller] = {}
    sends = across_sends(entries)
    for entry in [*entries, *sends]:
        hops = []
        for rel, line in reversed([(f, int(n)) for f, n in re.findall(r"([\w./-]+\.py):(\d+)", entry)]):
            node = R.def_holding(rel, line)
            if node is None:
                break
            owner = R.owner_of(rel, node)
            silent = prints_nothing(owner, [line])
            marks = () if silent else key_marks(owner, line, cls, key)
            how = "silent" if silent else "marks" if marks else "key" if prints_key(owner, [line], columns) else "none"
            hops.append((owner, rel, line, how, marks, node.lineno))
        pick = next((h for kind in ("marks", "key", "none") for h in hops if h[3] == kind), None)
        if pick is None:
            continue
        n = hops.index(pick)
        owner, rel, line, how, marks, at = pick
        c = out.setdefault((rel, at), Caller(owner, [], n + 1, how, marks))
        c.lines = sorted({*c.lines, line})
        if sites := re.findall(r"([\w./-]+\.py):(\d+)", entry):
            c.outer = sorted({*c.outer, (f"{sites[0][0]}:{sites[0][1]}", line)})
        c.marks = tuple(sorted({*c.marks, *marks}))
        c.between = sorted({*c.between, *(f"{r}:{n_}" for _, r, n_, _, _, _ in hops[:n])})
        if (send := sends.get(entry)) and send in {f"{r}:{n_}" for _, r, n_, _, _, _ in hops[:n]}:
            c.across = sorted({*c.across, line})
            c.sent = sorted({*c.sent, send})
    return list(out.values())


def across_sends(entries: list[str]) -> dict[str, str]:
    """U2X (A) (out/gm/qk-design.md U2X): the ``called_from`` *entries* go on once past a send.  Where an entry's
    outermost hop is in an endpoint function a send enters (a source of its ``ParameterFrom`` the map marks sent, SND),
    each call into the sender -- the def holding the send, the client's call last on that source's route -- is a hop
    further out (the map's ``ReachNode.calls``): ``job_api.py:572 -> TaskBuffer.py:1190`` goes on to
    ``copyArchive.py:904 -> Client.py:273 -> job_api.py:572 -> TaskBuffer.py:1190``.  Once: the sender's callers'
    callers are not read.  Each new entry with its send."""
    out: dict[str, str] = {}
    for entry in entries:
        sites = re.findall(r"([\w./-]+\.py):(\d+)", entry)
        if not sites:
            continue
        node = R.def_holding(sites[0][0], int(sites[0][1]))
        if node is None:
            continue
        for send in SR.sends_into(f"{sites[0][0]}::{node.name}"):
            srel, _, sln = send.rpartition(":")
            sender = R.def_holding(srel, int(sln))
            if sender is None:
                continue
            # the sender as the map names it: its class's name before its own, a module function's alone
            cls = next((o for f, o in pathcond.functions_with_owner(R.tree(srel)) if f is sender), None)
            for caller, lines in SR.calls_into(srel, f"{cls}.{sender.name}" if cls else sender.name):
                crel = caller.partition("::")[0]
                for n in lines:
                    out.setdefault(f"{crel}:{n} -> {send} -> {entry}", send)
    return out


def _f3(state: dict | None) -> str:
    """The kind of an attribute's H2 try that is no statuslog row (``Rerun._attribute_tries``): "span" (the object's
    life, asked of F3's writers, or around the value for one writing the time it runs), "run" (a run found in it),
    "open" (the life, not searched whole), "made" (a birth's row); "" for a row."""
    return (state or {}).get("f3", "")


def _row_said(row: dict) -> T.Said:
    """A try's row as H2's stops say it: a statuslog row by its status and time, else what F3's or a birth's is."""
    kind = _f3(row)
    if not kind:
        return T.Said("row-statuslog", {"status": row["status"], "time": row["time"]})
    return T.Said({"made": "row-made", "run": "row-run", "span": "row-life", "open": "row-life"}[kind], {"time": row["time"]})


def _records_json(outcome: dict) -> dict:
    """*outcome* with its record, and its branches' labels, records, rows and premises, as result.json keeps them
    (render.to_json, :func:`_premise_json`).  Each record reads back to the sentence the run said, or the run fails here:
    what result.json keeps says what the run said."""
    def kept(d: dict) -> dict:
        out = dict(d)
        for k in ("record", "label", "row"):
            if out.get(k) is not None:
                if not T.same_read_back(out[k]):
                    raise AssertionError(f"result.json's {k} does not read back as said: {T.text(out[k])!r}")
                out[k] = T.to_json(out[k])
        if out.get("older"):
            out["older"] = [T.to_json(r) for r in out["older"]]
        if out.get("premises"):
            out["premises"] = [_premise_json(p) for p in out["premises"]]
        return out
    o = kept(outcome)
    if o.get("branches"):
        o["branches"] = [kept(b) for b in o["branches"]]
    return o



def _kept_said(said: T.Said | str, what: str):
    """*said* as result.json keeps it (render.to_json), or the run fails here: it reads back to what the run said."""
    if not T.same_read_back(said):
        raise AssertionError(f"result.json's {what} does not read back as said: {T.text(said)!r}")
    return T.to_json(said)


def _movers_json(movers: list | None) -> list | None:
    """*movers* as result.json keeps them: each premise's why and each reason's record as records (render.to_json),
    each reading back to the sentence the run said (:func:`_kept_said`)."""
    if not movers:
        return movers
    out = []
    for m in movers:
        m = dict(m)
        if m.get("premises"):
            m["premises"] = {k: {**p, "why": _kept_said(p["why"], f"premise {k}'s why")} for k, p in m["premises"].items()}
        if m.get("reasons"):
            m["reasons"] = [{**r, "record": _kept_said(r["record"], f"reason {r['code']}")} if "record" in r else r
                            for r in m["reasons"]]
        out.append(m)
    return out

def _ends(ends: list) -> T.Said | None:
    """A run's or a branch's END lines (``(hop, said)``) as one record, result.json's ``record``; None where it said none."""
    return T.Said("ends", {"ends": [said for _, said in ends]}) if ends else None


def _at_rows(ps: list) -> T.Said:
    """An arm's rows as H2 names them: whether it wrote or may have at each (:class:`Possible`)."""
    return T.Said("at-rows", {"rows": [
        T.Said("wrote-at" if p.wrote else "may-have-at", {"row": _row_said(p.row)}) if p.row else
        T.Said("wrote" if p.wrote else "may-have") for p in ps]})


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


def _stay_of(states: list[dict], observed: str, since: str | None) -> tuple[int | None, T.Said | str]:
    """``(row, requirement)``: the statuslog row beginning the stay of *observed* a -late question asks about, or None
    and the requirement not met.  A value can come back many times (a task between pending and running), and the first
    stay is not the one a case picks (MT-exhausted-late asked the 240 h from 10-19, and means the 1.7 h of 10-29), so
    where there is more than one, *since* names it: the row's time_extended, or its time in a row with none (two rows
    of one second differ only in time_extended: 'toretry' and 'exhausted' at 09:33:01)."""
    stays = [n for n, s in enumerate(states) if s["status"] == observed]
    if not stays:
        return None, T.Said("a-statuslog-row-for", {"observed": observed})
    if since is not None:
        at = next((n for n in stays if states[n].get("time_extended") == since), None)
        if at is None:
            at = next((n for n in stays if not states[n].get("time_extended") and states[n]["time"] == since[:19]), None)
        return at, T.Said("a-stay-beginning-at", {"observed": observed, "since": since})
    if len(stays) > 1:
        return None, T.Said("one-stay-or-since", {"observed": observed, "stays": _stays_said(states, stays)})
    return stays[0], ""


def _stays_said(states: list[dict], stays: list[int]) -> T.Said:
    """*stays* (the statuslog rows that begin a stay of one value) in a line: how many, the first, the last and the
    longest, each by the time that names it to ``--since`` and how long it lasted ("on" for one with no row after)."""
    def one(n: int) -> T.Said:
        begins = states[n].get("time_extended") or states[n]["time"]
        if n + 1 >= len(states):
            return T.Said("stay-on", {"begins": begins})
        ends = states[n + 1].get("time_extended") or states[n + 1]["time"]
        return T.Said("stay-lasted", {"begins": begins, "hours": (_when(ends) - _when(begins)).total_seconds() / 3600})

    def lasted(n: int) -> float:
        if n + 1 >= len(states):
            return -1.0
        return (_when(states[n + 1].get("time_extended") or states[n + 1]["time"]) - _when(states[n].get("time_extended") or states[n]["time"])).total_seconds()

    return T.Said("stays", {"n": len(stays), "first": one(stays[0]), "last": one(stays[-1]), "longest": one(max(stays, key=lasted))})


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


def _the_arm_taken(j, arms: set[int]):
    """S2's way for a pass whose judgement chose nothing: the one positive verdict on an arm, where every other
    positive is an exit.  One call goes through the arm and then leaves by an exit (updateJobStatus 242, then the
    return at 496 its "done" proves), so an arm and an exit are no two ways; an exit before the arm is a ``continue``
    the call came back from.  None where no arm, or more than one, is positive."""
    positive = [v for v in j.verdicts if v.kind == "positive"]
    taken = [v for v in positive if v.line in arms]
    return taken[0] if len(taken) == 1 and len(positive) > 1 else None


def took_none(j) -> bool:
    """S2's way for a pass whose judgement excluded every candidate, the arms that move the value and the exits alike:
    it took none of them -- it ran to its end, or left by a raise in what it calls -- so no arm of the mover's moved the
    value in it, and S3 reads why (setupper_atlas_plugin's run, whose one arm 212 its row 206 excludes).  A candidate
    kept or not examined may have been taken: False."""
    return j.chosen is None and bool(j.verdicts) and all(v.kind == "excluded" for v in j.verdicts)


def ran_no_arm(j, arms: set[int]) -> tuple[int | None, list[int]] | None:
    """S batch S-1 and S-2a (qk-design.md, tmaeno "OK" 2026-11-16 13:0x and 16:1x): where the judgement *j* excluded
    every one of the *arms*, the pass ran none of them, whichever way it went -- ``(the exit, [])`` where the judgement
    chose one on positive evidence (MT-finishing-late's 218, its row 215 printing only on its path), else ``(None, the
    exits it left kept, sorted)``: one it chose by elimination is no way the pass went, which no row shows -- a pass
    may run on past every exit (setScoutJobData_JEDI's six arms, their must rows unprinted, and its exits 1185, 1215,
    1457 and 1465).  None where an arm is kept or not examined.  Whether such an exit keeps a run from the arms is the
    map's (``ExitKeeps``), and S3 says it."""
    if not arms:
        return None
    kinds = {v.line: v.kind for v in j.verdicts}
    if any(kinds.get(line) != "excluded" for line in arms):
        return None
    if j.chosen is not None and j.chosen.kind == "positive":
        return j.chosen.line, []
    return None, sorted(line for line, kind in kinds.items() if line not in arms and kind != "excluded")


_EK: list = [None, {}]


def exit_keeps(owner: str) -> dict[int, Any] | None:
    """The map's ``ExitKeeps`` of *owner* (``file::function``, as :func:`_bare` spells it; the map's batch 6), by each
    exit's line: the junction arms (``Branch.line``, ``Branch.other_lines``) and the calls into arms (``calls``, the
    map's M-h) a run leaving there can no longer reach and could have reached had it gone on -- a ``return`` or a
    raise over the call, a ``continue`` or a ``break`` over one turn of its loop -- and the same over one turn of each
    loop holding it (``turns``, innermost first, the map's M-g).  An exit it does not list is one not read.  None from
    a dump that reads none (one built before the batch)."""
    eks = getattr(mapdump.fragment(), "exit_keeps", None) or []
    if not eks:
        return None
    if _EK[0] is not eks:
        index: dict[str, dict[int, Any]] = {}
        for ek in eks:
            index.setdefault(_bare(ek.owner), {})[int(ek.exit.rpartition("@")[2])] = ek
        _EK[:] = [eks, index]
    held: dict[str, dict[int, Any]] = _EK[1]
    return held.get(_bare(owner), {})


_PL: list = [None, {}]


def print_loops(owner: str, line: int) -> list[int] | None:
    """The loops whose body holds *owner*'s print statement at *line*, innermost first (the map's ``PrintLoops``): a
    pass whose entry row is that print is one turn of the first.  ``[]`` for a print of a function the map read in no
    loop; None from a dump that reads none."""
    pls = getattr(mapdump.fragment(), "print_loops", None)
    if pls is None or not getattr(mapdump.fragment(), "exit_keeps", None):
        return None
    if _PL[0] is not pls:
        _PL[:] = [pls, {(_bare(p.owner), p.line): list(p.loops) for p in pls}]
    held: dict[tuple[str, int], list[int]] = _PL[1]
    return held.get((_bare(owner), line), [])


def _kept_lines(keeps) -> set[int]:
    """What an ``ExitKeeps`` or a ``TurnKeeps`` keeps a run from that an arm of a junction may be: its arms and its
    calls into arms (the map's M-h: an arm that is a call, AtlasProdTaskBroker's 633)."""
    return set(keeps.arms) | set(getattr(keeps, "calls", None) or [])


_GF: list = [None, {}]


def guard_froms(owner: str) -> list:
    """The map's ``GuardFrom`` of *owner* (``file::function``, as :func:`_bare` spells it): what each name a test on
    the way to its arms reads holds, way by way."""
    gfs = getattr(mapdump.fragment(), "guard_froms", None) or []
    if _GF[0] is not gfs:
        index: dict[str, list] = {}
        for g in gfs:
            index.setdefault(_bare(g.owner), []).append(g)
        _GF[:] = [gfs, index]
    held: dict[str, list] = _GF[1]
    return held.get(_bare(owner), [])


def _held_reading(g, rel: str, lo: int, hi: int) -> tuple | None:
    """S-3 (qk-design.md, tmaeno "OK" 2026-11-16 13:0x): what one ``GuardFrom`` says its name holds at its test, where
    the querier reads it -- ``("constant", value, line)`` where every way ends at one literal (setScoutJobData_JEDI's
    ``nNewJobsCutoff``, 20 at 1225); ``("printed", call, print, hole, expression)`` where every way comes back through
    one call whose callee prints, whole, what it gives (``HeldPrinted``: getScoutJobData_JEDI's 1140 ``extra=`` for
    ``extraInfo`` through the call at 1169, its hole 0 for ``scoutSucceeded`` whether it returned True or False).  None
    otherwise, or where a way changes the object in place within the function (*rel*, lines *lo* .. *hi*)."""
    sources = list(getattr(g, "sources", None) or [])
    if not sources:
        return None
    for s in sources:
        for c in getattr(s, "changes", None) or []:
            crel, _, cline = str(c.at).rpartition(":")
            if crel == rel and cline.isdigit() and lo <= int(cline) <= hi:
                return None
    if all(s.kind == "constant" for s in sources) and len({s.name for s in sources}) == 1:
        try:
            value = ast.literal_eval(sources[0].name)
        except (ValueError, SyntaxError, TypeError):
            return None
        return ("constant", value, sources[0].line) if _hashable(value) else None
    prints: set[tuple] = set()
    for s in sources:
        if getattr(s, "steps", None) or len(getattr(s, "printed", None) or []) != 1 or s.printed[0].steps:
            return None
        p = s.printed[0]
        prints.add((p.at, p.statement, p.hole, p.expression))
    return ("printed", *prints.pop()) if len(prints) == 1 else None


def _hole_value(text: str) -> Any:
    """The value a print's hole holds (:func:`_printed_value`); a dict it does not read, as the map reads a printed
    bind map (``sqltext.cut``: a time read as one, a value with no literal spelling kept as its text, so one never voids
    the rest -- getScoutJobData_JEDI's ``extra=`` holds a ``datetime.timedelta``).  ``E.UNKNOWN`` where neither reads
    it."""
    got = _printed_value(text)
    if got is E.UNKNOWN and text.lstrip().startswith("{"):
        cut = sqltext.cut(text.strip(), "", "")
        if cut is not None:
            return cut[1]
    return got


def _items_spelled(guards: list[str], name: str) -> list[tuple[str, Any]]:
    """``(text, key)`` of each item of *name* the *guards* take by a literal key, as they spell it
    (``extraInfo['nNewJobs']``): the evaluator looks a subscript up by its text."""
    out = []
    for g in guards:
        try:
            tree = ast.parse(E.strip_annotation(g), mode="eval")
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (isinstance(node, ast.Subscript) and ast.unparse(node.value) == name
                    and isinstance(node.slice, ast.Constant) and _hashable(node.slice.value)):
                out.append((ast.unparse(node), node.slice.value))
    return sorted(set(out), key=lambda x: x[0])


def false_by_values(test: ast.AST, env: E.Env) -> list[str] | None:
    """S-3b (qk-design.md, tmaeno "OK" 2026-11-16 16:44): the atoms of *test* whose values *env* reads make it false
    whatever its other atoms hold, or None where *env*'s values alone do not make it false -- a path's settled atoms
    are not read here.  Each atom with a value is read as unknown in turn and stays so where *test* is still false;
    the rest are the ones that make it false (an ``and``'s one false part, an ``or``'s every part)."""
    if E.truth(test, {}, env) is not False:
        return None
    atoms: dict[str, ast.AST] = {}
    for atom in E.atoms(test):
        atoms.setdefault(ast.unparse(atom), atom)  # the text E.truth looks an atom up by, as E.solve keys it
    off: dict[str, Any] = {text: E.UNKNOWN for text, atom in atoms.items() if E.truth(atom, {}, env) is E.UNKNOWN}
    out = []
    for text in atoms:
        if text in off:
            continue
        if E.truth(test, {**off, text: E.UNKNOWN}, env) is False:
            off[text] = E.UNKNOWN
        else:
            out.append(text)
    return out


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


def _named_open(subject: str, found: dict[str, list[str]], tl: TL.Timeline | None) -> list[T.Said]:
    """The premises a term's reading with the open arms named rests on: those arms of *subject* (*found*, the reading's
    ``_assuming``) as having not written it, and the arms its timeline names as having not begun the stay."""
    named = [T.Said("none-of-wrote-it", {"arms": found[subject], "column": subject})] if found.get(subject) else []
    if tl is not None and tl.begun_assumed:
        named.append(T.Said("none-of-began-it", {"arms": list(tl.begun_assumed)}))
    return named


def _state_terms(st) -> dict[str, list]:
    """The terms ``IS NULL`` or ``IS NOT NULL`` of *st*'s condition trees, by the column each tests (K-3 (a))."""
    out: dict[str, list] = {}
    todo = [r.tree for r in getattr(st, "condition_trees", None) or []]
    while todo:
        node = todo.pop()
        if node.node in ("any", "all"):
            todo += list(node.items)
        elif node.node in ("null", "not_null") and node.subject:
            out.setdefault(node.subject, []).append(node)
    return out


def _spans_words(spans: list, whole: tuple) -> T.Said:
    """``all the window``, or each span ``from .. to ..``."""
    return T.Said("all-the-window") if spans == [whole] else T.Said("spans-from-to", {"spans": [
        T.Said("from-to", {"a": a, "b": b}) for a, b in spans]})


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
    #: the arms H2 dropped by a guard read false (``_guard_dropped``, a stand-in's in ``_possibles``), grown as H2 reads: an
    #: arm left alone with one of these is not the only way to the value (KNOWN_ISSUES.md K66)
    _dropped_by_guard: frozenset[str] = frozenset()
    #: what the walk took on before its END, which every END line rests on, grown as it walks: the one arm H2 left,
    #: taken by a late question as what moved the value on (``_the_arm_left``)
    rests: tuple[T.Said, ...] = ()
    #: the movers L2 found ran on the key in a late value's stay, other than the one that moved it, each followed on its
    #: own (``_late_absence``, ``missed``)
    _late_ran: tuple[Passes, ...] = ()
    #: the values each history write's row can record (``attr_time.recorded_values``), read once (P37)
    _recorded_values: dict[str, list[str]] | None = None
    #: the time columns a reading names the arms of as having not written them, by column, while one collects them (a
    #: turn, M3(a)-cap; Q1's assumed spans); None when none does
    _assuming: dict[str, list[str]] | None = None
    #: the columns tested ``IS NULL`` whose reading took a value known only at run time as not None, with those arms
    #: (``_none_premised``), while a reading collects them (P5); None when none does
    _premised_used: dict[str, list[str]] | None = None

    def __init__(self, args: argparse.Namespace) -> None:
        mapdump.require("rerun")  # _investigate's walks and _writers_of ask mapdump.code_map
        A.made.clear()
        self.args = args
        self.mode, self.decide = args.patterns, args.decide
        self.data = json.loads(Path(args.rows).read_text())
        self.hand = json.loads(Path(args.hand).read_text()) if args.hand else {}
        self.hand_used: list[str] = []
        self.logs: Logs = RecordedLogs(args.records) if args.records else FileLogs(args.log)
        if isinstance(self.logs, RecordedLogs):
            # K67b B: a record naming the key only as another column makes no mark of the key's run
            cls = args.subject.partition(".")[0]
            self.logs.other = lambda key, log, r: key == args.key and KH.other_column(r, log, key, cls)
        self.lines: list[str] = []
        self.hop = ""
        self.touched: list[tuple[R.Function, object]] = []  # (function, run) this investigation read
        #: what the runs read so far printed about a row, with where and when -- a value
        #: printed at one time and used for a decision at another says so in its source -- and the functions that
        #: printed it (``Env.put_hole``, KNOWN_ISSUES.md K63)
        self.row_facts: dict[str, tuple[str, str, frozenset[str]]] = {}
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
        self.ends: list[tuple[str, T.Said | str]] = []
        self.branches: list[dict] = []
        #: result.json's ``outcome.code`` for a run that ends: ``one-of`` where its answer is arms H2 left (``arms_left``,
        #: or the one a late question took as its mover, ``rests``)
        self.end_code: str | None = None
        #: the arm a branch of ``arms_left`` is walking, at the row it reads (``wrote_by``)
        self._walking: WroteBy | None = None
        self.stamps: dict[str, dict] = {}
        self.result: dict = {}
        #: the map's reach, read from the dump at its first ask (``_reach_of``): None before, False with none in the dump
        self._reach = None
        #: the dump's reach_graph as read (``_graph``): (the reach module, its ReachGraph, its rows); False with none
        self._graph_read = None
        #: ``_called_at``'s answers: (caller, callee) function -> the lines of the calls that go on into it
        self._calls_into: dict[tuple[str, str], list[str]] = {}
        #: ``_keeps``' answers: (function, first line, last line, subject, value) -> calls closed by their name, or None;
        #: and the in-memory destination per (subject, value)
        self._kept: dict[tuple[str, int, int, str, str], int | None] = {}
        self._in_memory: dict[tuple[str, str], Any] = {}
        #: H1's readings by (subject, value), read once (``_writers``); the runs F1 read the rows after (``rows_between``)
        self._readings: dict[tuple[str, str], list] = {}
        self._rows_read: set[str] = set()

    # ------------------------------------------------------------ plumbing
    def say(self, text: str = "") -> None:
        self.lines.append(text)
        print(text, flush=True)

    def need(self, what: T.Said | str, found, code: str | None = None):
        if found:
            return found
        if self.hop in self.hand:
            self.say(f"HAND {self.hop}: {what} <- {self.hand[self.hop]!r}")
            self.hand_used.append(self.hop)
            return self.hand[self.hop]
        raise Stop(T.at(self.hop, what), code)

    def pick(self, what: T.Said | str, judgement):
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
        raise Stop(T.at(self.hop, what))

    def start(self, hop: str, title: str) -> None:
        self.hop = hop
        self.opened.append(hop)
        self.say()
        self.say(f"== {hop} {title}")

    def end(self, what: T.Said | str) -> None:
        """An END line: what the run found (result.json's ``outcome.record``), with what the walk took on before it
        (``rests``): an answer resting on the one arm H2 left is ``one-of``, as ``arms_left``'s is."""
        if self.rests:
            what = T.Said("an-end-resting-on", {"what": what, "rests": T.Said("rests-on", {"premises": list(self.rests)})})
            self.end_code = self.end_code or "one-of"
        self.ends.append((self.hop, what))
        self.say(f"END {what}")

    def _other_of(self, key: str, log: Log) -> Callable[[Record], bool] | None:
        """K67b B (``key_holes``): ``records.runs_for``'s *other* for the question's key in *log* -- a record whose
        message names it only as another column than the question's class's identity makes no run of it; None for any
        other key (a caller's value, a bare mark, a function's name), read as before, and None where no question is
        held (a Rerun made without one)."""
        args = getattr(self, "args", None)
        if args is None or key != args.key:
            return None
        cls = args.subject.partition(".")[0]
        return lambda r: KH.other_column(r, log.name, key, cls)

    def files_of(self, owner: str, today: list[str], lines) -> list[str]:
        """Where *owner*'s own rows are looked for (``R.own_log_files``)."""
        return R.own_log_files(owner, list(today), lines)

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
        # the question's own column, under its row's name: another function's print of it is read only where the map
        # holds one object (``Env._hidden``, KNOWN_ISSUES.md K63)
        cls, _, column = self.args.subject.partition(".")
        entity = next((e for e, c in (self.data.get("classes") or {}).items() if c == cls and e != "about"), None)
        env.subject = f"<{entity}>.{column}" if entity and column else ""
        # an arm's guards read as the row only the names the map's records bind to what the arm writes (K64)
        env.written = self.args.subject if env.subject else ""
        env.elements = _element_rows(self.data, cls)
        env.keys = frozenset(c.partition(".")[2] for c in self._key_columns(self.args.subject, self.args.key))
        # the question's column as a pass read it from the row: the statuslog's state before the pass (H2 step 2 (iii))
        env.statuslog = list((self.data.get("statuslog") or {}).get("states") or [])
        for entity, row in rows.items():
            env.table[entity] = {k: E.Value(v["value"], v["source"]) for k, v in row.items()}
        if file_row is not None:
            env.table["file"] = {k: E.Value(v["value"], v["source"]) for k, v in file_row.items()}
        for expr, (printed, source, printers) in self.row_facts.items():
            env.put_hole(expr, printed, source, printers)
        return env

    def learn(self, pairs: list[tuple[R.Function, object]]) -> None:
        """Keep what these runs printed about a row, for the steps after."""
        env = self.base_env()
        before = set(env.values)
        put = stable_holes([R.seen(f, q, self.mode) for f, q in pairs], env)
        for expr, printed in put.items():
            key = env.canonical(expr)
            if key not in before and "<" in key:
                printers = frozenset(o for o, e, _ in env.printed_in.get(key, ()) if e == expr)
                self.row_facts[expr] = (printed, env.values[key].source, printers)

    def runs(self, key: str, names: list[str], asking: W.Asking, owners: Owners | None = None) -> dict[str, list]:
        """The key's runs in each of the logs *names* the door can be asked for, over *asking*'s window and machine.

        Every log is asked before a stop, so that a door short of answers names all the step's asks at once.
        """
        out, unanswered, _ = self._ask(key, names, asking, owners)
        self._stop_unanswered(unanswered)
        return out

    def _ask(self, key: str, names: list[str], asking: W.Asking, owners: Owners | None = None,
             whole: bool = False, quiet: bool = False) -> tuple[dict[str, list], list, dict[str, list[str]]]:
        """``runs()`` without the stop: the runs found, the asks the door had no answer for, and what of the window each
        log was not searched for (``Log.holes``).  *owners* and *whole* as ``Logs.log`` takes them; *quiet*: the logs'
        notes are not said (the timeline's asks, one for each arm, say the same notes over and over)."""
        window = self.window(asking)
        out, unanswered, holes = {}, [], {}
        for name in self.logs.names(names):
            try:
                log = self.logs.log(name, key, window, machine=asking.machine, owners=owners, strings=asking.strings, whole=whole)
            except Unrecorded as e:
                # the asks file keeps the why as text (its form is O4's)
                unanswered += [(ask, why or T.text(asking.why)) for ask, why in e.asks]
                continue
            for note in [] if quiet else log.notes + [f"not searched: {h}" for h in log.holes]:
                self.say(f"  {name}: {note}")
            if log.holes:
                holes[name] = log.holes
            out[name] = runs_for(log, key, self._other_of(key, log))
        return out, unanswered, holes

    def _keep_unanswered(self, unanswered: list) -> None:
        """Keep the asks no record answered for the asks file, without a stop: L2's movers are a late answer's context,
        and the walk goes on (judgement 3 for late; tmaeno, 2026-11-01).  None inside a reading that asks nothing
        (``_no_asks``: K-3 (a)'s notes on a term ``IS NULL``, read from the records at hand)."""
        if self.__dict__.get("_no_asks"):
            return
        self.unanswered += [(self.hop, ask, why) for ask, why in unanswered]

    def _stop_unanswered(self, unanswered: list) -> None:
        if unanswered:
            self._keep_unanswered(unanswered)
            # each ask once, as the asks file holds it: movers that share a log ask it alike
            n = len({ask.key for ask, _ in unanswered})
            raise Stop(T.at(self.hop, T.Said("ask-s-of-the-logs-not-in", {"n": n, "asks": self.args.asks})), "asks")

    def ask_rows(self, rows: dict, why: str, place: str) -> None:
        """Keep *rows*, what the rows file does not have said as a fetch can answer it -- a collection of a row, the
        rows a call returned, or an SQL question -- for --rows-asks (the harness's rows ask, stage 1), with the *place*
        that asks it (K36's four: ``loop``, ``call``, ``answers``, ``linked``)."""
        self.rows_asks.setdefault(json.dumps(rows, sort_keys=True, default=str),
                                  {"rows": rows, "step": self.hop, "why": why, "place": place})

    def rows_code(self) -> str | None:
        """The code of a stop no other rule names: ``rows`` where rows are asked at a place whose answer a step reads.
        Not for the 4th place's alone (``ask_linked``): its statement stays not settled whatever the rows, so a stop
        beside it is not one for them (KNOWN_ISSUES.md K36)."""
        return "rows" if any(a["place"] != "linked" for a in self.rows_asks.values()) else None

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
                      f"@{step.line} {step.owner.split('::')[-1]} loops over {iterable}, which the rows file does not have", "loop")

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
                              f"linked hold {lacked}'s", "linked")
        return list(dict.fromkeys(notes))

    def window(self, asking: W.Asking) -> Window | None:
        """*asking*'s window; where it has none, a stop if the door needs one (the door over files reads none)."""
        if not self.logs.needs_window:
            return asking.window
        window = self.need(T.Said("a-window-to-ask-the-logs-over", {"asking_why": asking.why}), asking.window)
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
        """result.json, version 4 (version 1 agreed with the live-case harness, 2026-10-15; 2 keeps the outcome's and
        the branches' sentences as records, ``record`` and ``label``, render.py's; 3 keeps what an arm rests on as
        records too, the branches' ``premises``, ``row`` and ``older`` and the possibilities' premises, each premise with
        its ``premise_kind``; 4 keeps the movers' too, each premise's ``why``, each reason's ``record`` (its text
        dropped), each reading's ``verdict`` as a token, and the map's ``gaps`` of every mover that has any): the outcome, the question, the steps
        opened in order, every input with its sha256, the asks file, and what the steps found -- the window, the arm
        that wrote an attribute's value last (``wrote``), the mover that moved the value (late), every mover with its
        verdict and, in S1, judgement 3's premises (``_premises``, added in build order 2a) -- with times in UTC, ISO 8601
        with ``Z``."""
        a = self.args
        dump = mapdump.installed()
        return {
            "version": 4,
            "outcome": _records_json(outcome),
            "question": {"mode": "late" if a.late else "stuck" if a.stuck else "asked", "subject": a.subject,
                         "observed": a.observed, "key": a.key, **({"since": a.since} if a.since else {})},
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
            "movers": _movers_json(self.result.get("movers")),
            "held_before": self.result.get("held_before"),
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
            Asked: self.asked, WroteBy: self.wrote_by, ArmsLeft: self.arms_left, Raised: self.raised, Missing: self.missing,
            Branch: self.branch, Settled: self.settled, Pair: self.pair, FalseAtom: self.false_atom,
            Reproduce: self.reproduce, Field: self.field,
            Stuck: self.stuck, Passes: self.passes, Callers: self.callers, CallerPasses: self.caller, ArmCaller: self.arm_caller,
            OtherRows: self.other_rows, Turned: self.turned,
            Blocked: self.blocked, Input: self.input, Rows: self.rows,
            Late: self.late, Missed: self.missed, Ordered: self.ordered,
        }
        if a.late:
            q = Late(a.subject, a.observed, a.key, a.since)
        else:
            q = Stuck(a.subject, a.observed, a.key) if a.stuck else Asked(a.subject, a.observed, a.key)
        self.steps = steps
        while q is not None:
            q = self.fork(q) if isinstance(q, Fork) else steps[type(q)](q)

    def fork(self, q: Fork):
        """Each branch to its end or its stop, each from the state the fork was reached in; a stop of any is the run's."""
        stops, codes, excluded = [], set(), 0
        kept = (list(self.touched), dict(self.row_facts), self.question)
        at = self.hop
        for n, (branch, label) in enumerate(zip(q.branches, q.labels, strict=True), 1):
            said = len(self.ends)
            stop = self._branch(n, len(q.branches), branch, label, kept)
            if stop is None:
                self.branches.append({"label": label, "kind": "end", "step": self.hop, "record": _ends(self.ends[said:])})
            elif isinstance(stop, Excluded):
                # a branch excluded did not do it (S3: an arm kept off because nothing raised): neither an end nor a stop
                excluded += 1
                self.branches.append({"label": label, "kind": "excluded", "step": self.hop,
                                      "record": T.Said("excluded-line", {"stop": stop.said}), "code": None})
            else:
                stops.append(T.Said("numbered", {"n": n, "said": stop.said}))
                codes.add(stop.code)
                self.branches.append({"label": label, "kind": "stop", "step": self.hop,
                                      "record": T.Said("stop-line", {"stop": stop.said}), "code": stop.code})
        self.hop = at
        if stops:
            raise Stop(T.Said("of-branch-es-stopped", {"n_stops": len(stops), "n_q_branches": len(q.branches), "stops": stops}), codes.pop() if len(codes) == 1 else None)
        self.need(T.Said("a-branch-not-excluded", {"n_branches": len(q.branches)}), excluded < len(q.branches))

    def _branch(self, n: int, of: int, branch, label: T.Said | str, kept: tuple) -> Stop | None:
        """Branch *n* of a fork walked to its end (None) or its stop, from the state *kept* the fork was reached in."""
        self.touched, self.row_facts, self.question = list(kept[0]), dict(kept[1]), kept[2]
        self.say()
        self.say(f"==== {n} of {of}: {label}")
        try:
            b = branch
            while b is not None:
                b = self.fork(b) if isinstance(b, Fork) else self.steps[type(b)](b)
        except Stop as stop:
            self.say(f"==== {n}: {'excluded' if isinstance(stop, Excluded) else 'STOP'} {stop}")
            return stop
        self.say(f"==== {n}: reached the end")
        return None

    def arms_left(self, q: ArmsLeft):
        """Each arm H2 left (:class:`OneOf`) walked on from H3 (:meth:`wrote_by`), from the state H2 left, at the newest row it
        may have written at: it ends, stops (open: the kinds of what it rests on, and of its stop) or is excluded (its
        guard read false at each of its rows).  The arms not excluded are the answer: one that ended is the END, with
        what it rests on; else END "one of N" (result.json's ``outcome.code`` ``one-of``, each arm in ``branches``).
        None left: no arm could have.  An arm stopped on asks or rows the records do not answer yet stops the run on
        them, for the recording step to fetch, every arm walked first (KNOWN_ISSUES.md K38)."""
        subject, value = q.asked.subject, q.asked.observed
        arms = self._possibilities(q.one_of.left)
        if not arms:
            raise self._one_of(subject, value, q.one_of)
        at, ends, result = self.hop, len(self.ends), dict(self.result)
        kept = (list(self.touched), dict(self.row_facts), self.question)
        self.say(f"{len(arms)} arm(s) left: each walks on from H3, at the newest row it may have written at (KNOWN_ISSUES.md K38)")
        # an attribute's one writer left, no other dropped by a guard and the value no default: the only way to the
        # value, so its guards held (KNOWN_ISSUES.md K66)
        only = (len(arms) == 1 and PT.history(subject) is None and not _a_default(value)
                and not self._dropped_by_guard - set(arms))
        if only:
            self.say(f"{next(iter(arms))} is the one writer of {subject}={value!r} left, none dropped by a guard: the only "
                     f"way to the value (KNOWN_ISSUES.md K66)")
        entries = []
        for n, (arm, ps) in enumerate(arms.items(), 1):
            label = T.Said("arm-label", {"arm": arm, "rows": _at_rows(ps)})
            said = len(self.ends)
            stop = self._branch(n, len(arms), WroteBy(q.asked, ps[0].branch, list(ps[0].premises), ps[1:], only), label, kept)
            walked = self._walking
            premises = list(walked.premises) if walked is not None else []
            kinds = {p.kind for p in premises if isinstance(p, Premise)}
            entry: dict = {"label": label, "arm": arm, "row": _row_said(walked.wrote.row) if walked and walked.wrote.row else None,
                     # the older rows it may have written at instead, not walked
                     "older": [_row_said(p.row) for p in walked.older if p.row] if walked is not None else [],
                     "premises": premises, "step": self.hop}
            if stop is None:
                entry |= {"kind": "end", "record": _ends(self.ends[said:]), "code": None}
            elif isinstance(stop, Excluded):
                entry |= {"kind": "excluded", "record": T.Said("excluded-line", {"stop": stop.said}), "code": None}
            else:
                kinds.add(NOT_RECORDED if stop.code in ("asks", "rows") else NOT_READ)
                entry |= {"kind": "stop", "record": T.Said("stop-line", {"stop": stop.said}), "code": stop.code}
            # what a branch put in result.json (H3's ``wrote``) is its own
            entry |= {"kinds": sorted(kinds), "result": {k: v for k, v in self.result.items() if result.get(k) is not v} or None}
            self.result, self._walking = dict(result), None
            entries.append((n, entry))
        self.branches += [e for _, e in entries]
        self.hop = at
        del self.ends[ends:]
        left = [(n, e) for n, e in entries if e["kind"] != "excluded"]
        excluded = [str(n) for n, e in entries if e["kind"] == "excluded"]
        unrecorded = [(n, e) for n, e in left if e["code"] in ("asks", "rows")]
        if unrecorded:
            codes = {e["code"] for _, e in unrecorded}
            stopped = [T.Said("numbered", {"n": n, "said": e["record"]}) for n, e in unrecorded]
            raise Stop(T.at(self.hop, T.Said("of-the-arm-s-left-stopped-on", {"n_unrecorded": len(unrecorded), "n_entries": len(entries), "stopped": stopped})), "asks" if "asks" in codes else "rows")
        if not left:
            raise Stop(T.at(self.hop, T.Said("an-arm-that-could-have-written-each", {"subject": subject, "value": value, "n_entries": len(entries), "items": [e["record"] for _, e in entries]})), "one-of")
        others = T.Said("others-excluded", {"n_excluded": len(excluded), "excluded": excluded}) if excluded else ""
        if len(left) == 1 and left[0][1]["kind"] == "end":
            (n, e), = left
            self.hop = e["step"]
            self.result |= e["result"] or {}
            self.end_code = "one-of" if e["premises"] else None
            self.end(T.Said("one-arm-ended", {
                "ends": e["record"].facts["ends"] if isinstance(e["record"], T.Said) else [],
                "rests": T.Said("rests-on", {"premises": [_said(p) for p in e["premises"]]}) if e["premises"] else "",
                "older": T.Said("or-older", {"older": e["older"]}) if e["older"] else "",
                "others": others,
            }))
            return None
        self.end_code = "one-of"
        who = T.Said("the-one-arm-left") if len(left) == 1 else T.Said("one-of-the-arms-left", {"n_left": len(left)})
        listed = [T.Said("arm-ended", {"n": n, "arm": e["arm"]}) if e["kind"] == "end" else T.Said("arm-open", {"n": n, "arm": e["arm"], "kinds": e["kinds"]})
                  for n, e in left]
        self.end(T.Said("written-last-by-which-the-data-does", {"subject": subject, "value": value, "who": who, "listed": listed, "others": others}))
        return None

    # ------------------------------------------------------------ Asked: who wrote the value
    def asked(self, q: Asked):
        w = self._which_wrote(q.subject, q.observed, q.key)
        # where H2 decides nothing, each arm it left walks on (KNOWN_ISSUES.md K38)
        return ArmsLeft(q, w) if isinstance(w, OneOf) else self.wrote_by(WroteBy(q, w))

    def wrote_by(self, a: WroteBy):
        """H3 on: the arm H2 chose, or one it left (a branch of :meth:`arms_left`, *a.premises* what it rests on), read at
        its newest row; where an attribute's guard reads false there, at its next (*a.older*), and with none, excluded."""
        q, w, branch = a.asked, a.wrote, a.premises is not None
        if branch:
            self.question, self._walking = w.question, a
        question, cands, env, chosen = w.question, w.cands, w.env, w.chosen
        for c in cands:
            if question is not None and question.log.name in c.log_files:
                self.touched.append((R.function(c.rel, c.func, c.lines), question))
        site_fn = R.function(chosen.candidate.rel, chosen.candidate.func, chosen.candidate.lines)
        # an arm left by elimination with no pass of it at the window has no run to read up to (K47)
        at_fn = self._guard_at(w, site_fn) if question is not None else site_fn
        if w.via is not None and question is None:
            # one of the arms left, read at a call of which no run is read (K38)
            self.say(f"{site_fn.shown}:{chosen.line} prints nothing: read at its caller's call, {w.via.candidate.func}:{w.via.line}, "
                     f"of which no run is read (KNOWN_ISSUES.md K25)")
        elif w.via is not None:
            self.say(f"{site_fn.shown}:{chosen.line} prints nothing: read on its caller's run, at {at_fn.shown}:{w.via.line} (KNOWN_ISSUES.md K25)")
        # the run was chosen by its mark before the function was known; a mark that is
        # the key can hold several passes of it (rule 20), and then this run is not one
        if question is not None and len(starts := R.entries(at_fn, question, self.mode)) > 1:
            self.say(f"WARNING: the run holds {len(starts)} passes of {at_fn.shown} (rule 20); the steps below read them as one")
        site = R.statement_at(site_fn.node, chosen.line)
        # an attribute's question is when it took the value, and H2 answered it: the pass of the arm that wrote it last
        # (KNOWN_ISSUES.md K37).  A status's goes on to why its writer's arm ran
        attribute = PT.history(q.subject) is None
        if attribute:
            self.result["wrote"] = self._wrote_said(w)

        # H3
        self.start("H3", "what its guard reads")
        # the tests reaching it ran under: the ifs around it and the early exits above it (KNOWN_ISSUES.md K61)
        guards = pathcond.arm_condition(site)
        self.say(f"guards of {site_fn.shown}:{chosen.line}: {guards}")
        # an arm H2 left is one that may have written, not one left by elimination (K38)
        left = (T.Said("left-by-elimination") if not branch else T.Said("it-may-have", {"premises": [_said(p) for p in a.premises]}) if a.premises
                else T.Said("shown-to-write"))
        if question is None and not attribute:
            # a status goes on to why its writer's arm ran, which is read in its run
            raise Stop(T.at(self.hop, T.Said("a-run-that-shows-ran", {
                "fn_name": site_fn.shown, "chosen_line": chosen.line,
                "why": T.Said("left-by-elimination-no-pass") if not branch else T.Said("no-pass-at-the-window", {"left": left}),
            })))
        # an arm its run shows ran (H2's positive, its own run: a stand-in's is its caller's) ran under its guards
        # (KNOWN_ISSUES.md K54)
        ran = chosen.kind == "positive" and w.via is None and question is not None
        unread = []
        if attribute:
            shown = chosen.kind == "positive"
            if not shown:
                # an arm left by elimination has no run that shows it ran: the run the judge read it on is any of the
                # key's at the window, on any machine (a server's DBProxy log for a JEDI writer).  Its guards are read in
                # the rows' values alone, and no run is named
                env = self.base_env(at=w.row["time"] if w.row else None)
            # read at the arm: another function's hole only where the map hands its object here (KNOWN_ISSUES.md K63)
            at = (env.reading(chosen.candidate.owner, chosen.line, arm=(chosen.candidate.owner, chosen.line))
                  if isinstance(env, E.Env) else env)
            read = {g: _guard_read(g, at) for g in guards}
            if isinstance(env, E.Env) and isinstance(at, E.Env):
                env.used.update(at.shared())
            # a guard over a column the rows file holds, read in it by the map's ways and their changes (K49)
            held = {}
            for g, got in read.items():
                if got is True or got is False:
                    continue
                h = HC.read(chosen.candidate.owner, g, chosen.line, self.data, question if shown else None, self.mode, env)
                if h is None:
                    continue
                held[g] = h
                self.say(f"{g!r} read in the rows file's {h.section}" + (f" as fetched at {h.as_of}" if h.as_of else "")
                         + f": {h.value!r}" + (f", on {h.shown}, the route its run shows" if h.shown else "") + " (KNOWN_ISSUES.md K49)")
                for k in h.keys:
                    self.say(f"  {T.text(k)}")
                if h.value is True or h.value is False:
                    read[g] = h.value
            if branch and (false := [g for g, got in read.items() if got is False]):
                # it did not write at this row: an older one it may have written at, else it is not among the arms
                h = held.get(false[0])
                where = (f"{site_fn.shown}:{chosen.line}'s guard " + T.text(T.Said("held-read-false", {"guard": false[0], "section": h.section, "as_of": h.as_of}))
                         if h is not None else
                         f"{site_fn.shown}:{chosen.line}'s guard {false[0]!r} read false in the {'run' if shown else 'rows'}' values")
                if a.older:
                    nxt = a.older[0]
                    self.say(f"{where}{f' at {_row_said(w.row)}' if w.row else ''}: it did not write there; its next row, {_row_said(nxt.row)}")
                    return WroteBy(q, nxt.branch, list(nxt.premises), a.older[1:], a.only_way)
                raise Excluded(T.at(self.hop, T.Said("it-did-not-write-at-any-row", {"where": where})))
            # a guard the values leave unread held where the run shows the arm ran, one read through a call too; one
            # read false still decides (KNOWN_ISSUES.md K54, K74)
            by_run = [g for g, got in read.items() if got is not True] if ran and not any(got is False for got in read.values()) else []
            # the one writer of the value left, none dropped by a guard: a guard the values leave unread held, through a
            # method too, since the value has no other way (KNOWN_ISSUES.md K66)
            only = [g for g, got in read.items() if got is not True and g not in by_run] if a.only_way else []
            # a condition H2 left the arm resting on that is read here as holding is what this reading says, not a
            # premise left open (KNOWN_ISSUES.md K62); on the only way, nothing it rested on -- its run not shown, a
            # condition not read -- is open of who wrote it
            if branch:
                done = {g for g, got in read.items() if got is True} | set(by_run)
                a.premises = [] if a.only_way else [p for p in a.premises or [] if not _condition_read(p, done)]
            # W10 (the investigator's u8/u9, NR-J-7423807701-cloud): an arm H2 left among others, no run showing it ran
            # and no guard of its own, is shown nothing: its branch ends on it only if it ran, its callers' conditions not
            # read (doGeneratePrePro:2174, for an evgen job)
            unshown = branch and not shown and not guards and not a.only_way
            if all(got is True or g in by_run or g in only for g, got in read.items()):
                # a birth's row asks no window (H2's scan, K37): the row says where
                lo, hi = w.window or ("", "")
                # a try that is no statuslog row (a birth's, F3's) has no status to name (_row_said)
                at = (T.Said("as-the-row-was-made", {"time": w.row["time"]}) if w.row and _f3(w.row) == "made" else
                      T.Said("at-a-try-s-row", {"row": _row_said(w.row)}) if w.row and _f3(w.row) else
                      T.Said("in-the-pass-of-the-row", {"status": w.row["status"], "time": w.row["time"]}) if w.row else
                      T.Said("at-time", {"time": hi}) if lo == hi else T.Said("in-window", {"lo": lo, "hi": hi}))
                via = T.Said("called-at", {"fn": at_fn.shown, "line": w.via.line}) if w.via is not None else ""
                how = (T.Said("its-guards-hold-in-the-run" if shown else "its-guards-hold-in-the-rows") if guards
                       else T.Said("it-has-no-guard-callers-unread") if unshown else T.Said("it-has-no-guard"))
                used = [g for g in guards if g in held and read[g] is True]
                if by_run:
                    how = T.Said("its-guards-held-as-it-ran", {
                        "guards": [T.Said("guard-quoted", {"guard": g}) for g in by_run],
                        "rest": T.Said("the-rest-in-the-run") if any(read[g] is True and g not in used for g in guards) else "",
                    })
                if only:
                    how = T.Said("its-guards-held-the-only-way", {
                        "guards": [T.Said("guard-quoted", {"guard": g}) for g in only],
                        "rest": (T.Said("the-rest-in-the-run" if shown else "the-rest-in-the-rows")
                                 if any(read[g] is True and g not in used for g in guards) else ""),
                    })
                if used:
                    premise = held[used[0]].premise(used)
                    how = (T.Said("its-guards-hold-held", {"held": premise}) if len(used) == len(guards)
                           else T.Said("its-guards-hold-with-held", {"how": how, "held": premise}))
                said = T.Said("written-last-by-if-it-ran" if unshown else "written-last-by", {
                    "subject": q.subject, "observed": q.observed, "fn": site_fn.shown, "line": chosen.line, "via": via})
                # what it leaves unread of the value reaching the row (KNOWN_ISSUES.md K41)
                store = _store_unread(q.subject, chosen.candidate.owner, chosen.line, site_fn, w.via)
                if shown:
                    self.end(T.Said("shown-in-its-run", {"said": said, "how": how, "at": at, "question_describe": question.said(),
                                                         "store": store}))
                    return None
                holes = [T.Said("log-on", {"log": log, "machines": list(dict.fromkeys(h.partition(": ")[0] for h in hs))})
                         for log, hs in w.unsearched.items()]
                self.end(T.Said("no-run-shows-the-line-ran", {
                    "said": said, "how": how, "at": at,
                    "by_elimination": (T.Said("the-only-writer-left") if a.only_way else T.Said("by-elimination") if not branch else ""),
                    "holes": T.Said("not-searched-there", {"holes": holes}) if holes else "", "store": store,
                }))
                return None
            unread = [T.Said("guard-read-false" if got is False else "guard-not-read-held" if g in held else "guard-not-read",
                             {"guard": g} | ({"keys": held[g].keys} if got is not False and g in held else {}))
                      for g, got in read.items() if got is not True]
            # an arm left by elimination is read in no run: its unread guards are not shown to hold (G is a positive
            # arm's, KNOWN_ISSUES.md K54)
            self.need(T.Said("a-run-that-shows-ran-to-read", {"fn_name": site_fn.shown, "chosen_line": chosen.line, "left": left, "unread": list(unread)}), shown)
        # what H3 lacks is then the values the guards read at the run, a call in a guard read as any term (K74)
        if not guards:
            if ran and not attribute and mark_identities(chosen.candidate.owner, chosen.line, question):
                # what made it run is its callers': the call that ran it, read on the run its own run is inside, by
                # what the call handed the run's mark (U3)
                return ArmCaller(q, chosen.candidate.owner, chosen.line, question)
            # its value is what a method its statement calls returns: the map's ways to it through that method
            # (KNOWN_ISSUES.md K65)
            ways = callee_ways(q.subject, chosen.candidate.owner, chosen.line, q.observed)
            if ways:
                self._read_ways(q, w, site_fn, chosen, ways, ran)
            self.need(T.Said("no-guard-to-read", {"fn": site_fn.shown, "line": chosen.line}), None)
        if ran and not attribute and site is not None:
            # a status's arm ran under its guards: where the values they held on came from, said where its run printed
            # them -- unless they read a guard false, which the run says held
            printed: dict[tuple[int, str], tuple[str, str]] = {}
            held_on, read_false = self._held_on(site_fn, site, guards, question, printed)
            ifs = _guard_ifs(site_fn, site, guards) if read_false is None else None
            if read_false is not None:
                what = T.Said("printed-values-read-false", {"guard": read_false[0], "on": read_false[1]})
            elif ifs is None:
                what = T.Said("values-held-on", {"held": held_on})
            else:
                # where each term's ways end on the map's ways, the first that does not end named (S, gm-design.md)
                terms, entry = self._arm_terms(site_fn, chosen.candidate.owner, chosen.line, guards, ifs, printed,
                                               question, w.window)
                first = next((t for t in terms if not t.ended), None)
                # a term read by several guards the same way is said once, where it is first read
                once: list[SR.Term] = []
                for t in terms:
                    if not any((o.term, o.ends) == (t.term, t.ends) for o in once):
                        once.append(t)
                term_ends = [T.Said("term-ends", {"term": t.term, "ends": [_way_said(e) for e in t.ends]}) for t in once]
                if first is None and entry.end in SR.ENDS:
                    # every value its guards held on, and its entry, end (C2): why it ran is answered
                    self.end(T.Said("ran-where-every-value-ends", {"fn": site_fn.shown, "line": chosen.line, "held": held_on,
                                                                   "terms": term_ends, "entry": _way_said(entry)}))
                    return None
                what = T.Said("values-held-on-from", {
                    "held": held_on, "terms": term_ends, "entry": _way_said(entry),
                    "left": _first_stop(first) if first is not None else T.Said("terms-end-entry-stops"),
                })
        else:
            unread = unread or [T.Said("guard-not-read", {"guard": g}) for g in guards]
            what = T.Said("values-the-guard-reads", {"conditions": T.Said("its-conditions", {"unread": unread})})
        self.need(what, None)
        return None

    def _read_ways(self, q: Asked, w: Wrote, fn: R.Function, chosen: Verdict, ways: list[list[str]], ran: bool) -> None:
        """H3 for a status's arm with no guard of its own whose value a method its statement calls returns: the map's
        ways to the observed value through it (``callee_ways``).  Each term is read in the caller's names at the call
        (``_in_caller``), in the run's values and, for the row's own column, the statuslog row before the question's
        (``_status_before``).  A way a term reads false is not the one taken.  Where one is left and the run shows the
        arm ran, its terms the values leave unread held: it is the only way to the value (inf-design.md 8).  It stops as
        K54's status does: with that way's terms, or the ways left and what they leave unread, or none left though the
        run shows the arm ran (KNOWN_ISSUES.md K65)."""
        owner, line = chosen.candidate.owner, chosen.line
        callee = next(m["fn"] for way in ways for t in way if (m := _IN_CALLEE.fullmatch(t)))
        graph = self._graph()
        passed: dict[str, dict[str, str]] = {}
        for node in graph[2] if graph else []:
            if _same_function(node.owner, owner):
                for b in node.call_bindings or []:
                    if b.line == line and b.parameter and b.to:
                        passed.setdefault(b.to.rpartition(".")[2].rpartition("::")[2], {})[b.parameter] = b.path
        env = w.env if isinstance(w.env, E.Env) else self.base_env(at=w.row["time"] if w.row else None)
        at = env.reading(owner, line, arm=(owner, line))
        at = replace(at, table={k: dict(v) for k, v in at.table.items()})
        before = _status_before((self.data.get("statuslog") or {}).get("states") or [], w.row)
        if before is not None and (m := re.fullmatch(r"<(\w+)>\.(\w+)", at.subject)):
            at.table.setdefault(m[1], {})[m[2]] = E.Value(
                before["status"], f"the statuslog row before it, {before['status']!r} at {before.get('time_extended') or before['time']}")

        def read(term: str) -> tuple[object, T.Said]:
            m = _IN_CALLEE.fullmatch(term)
            one = replace(at, used={})
            got = _guard_read(_in_caller(m["term"], passed.get(m["fn"], {})) if m else term, one)
            if got is True:
                on = [f"{k} ({v})" for k, v in one.used.items()]
                return got, T.Said("guard-held-on", {"guard": term, "on": T.Said("held-on-values", {"values": on}) if on else ""})
            return got, T.Said("guard-read-false" if got is False else "guard-not-read", {"guard": term})

        readings = [[read(t) for t in way] for way in ways]
        left = [i for i, way in enumerate(readings) if not any(got is False for got, _ in way)]
        self.say(f"{fn.shown}:{line}'s value {q.observed!r} through {callee}(): {len(ways)} way(s) to it, "
                 f"{len(left)} left by the values read")
        facts = {"fn": fn.shown, "line": line, "value": q.observed, "callee": callee, "n": len(ways)}
        if not left:
            self.need(T.Said("no-way-to-the-value-holds", facts | {
                "ways": [T.Said("one-way", {"terms": [s for _, s in way]}) for way in readings],
                "shown": T.Said("though-its-run-shows-it-ran") if ran else ""}), None)
        if len(left) == 1 and ran:
            terms = [s if got is True else T.Said("guard-held-the-only-way", {"guard": t})
                     for t, (got, s) in zip(ways[left[0]], readings[left[0]], strict=True)]
            self.need(T.Said("the-only-way-to-the-value", facts | {"terms": terms}), None)
        self.need(T.Said("ways-left-to-the-value", facts | {
            "k": len(left), "ways": [T.Said("one-way", {"terms": [s for _, s in readings[i]]}) for i in left]}), None)

    def _held_on(self, fn: R.Function, site: ast.AST, guards: list[str], run: Any,
                 printed: dict[tuple[int, str], tuple[str, str]] | None = None,
                 quiet: bool = False) -> tuple[list[T.Said], tuple[str, T.Said | str] | None]:
        """Each guard of *site* (``pathcond.arm_condition``'s: the ``if``s around it, outermost first, then the early
        exits above it) with the values its terms held at its ``if`` -- an early exit's at the exit's own ``if``
        (``pathcond.exits_above_at``), which is where a guard ``[at N]`` reads its names too -- where *fn*'s run printed
        them (:func:`CH.printed_at`, read as the map's evaluator reads a printed value), and the first guard they read
        false, with its values (KNOWN_ISSUES.md K54, K61).  Texts are not ordered: a guard ordering a value read as text
        is not read.  In a loop that holds the site, a row is read only at a record the key is on, and only where each of
        the loop's rows printed once among those: the key's one pass of the loop.  *printed* takes each value read, by
        its ``if``'s line and the chain (S4c's V).  *quiet*: the values are not said here (a reader that words them
        itself)."""
        ifs_of = _guard_ifs(fn, site, guards)
        if ifs_of is None:
            # a test path_condition could not render, or an exit's if is not found: the guards are not placed, and no
            # value is read
            return [T.Said("guard-held-on", {"guard": g, "on": ""}) for g in guards], None
        own = {id(n) for n in ast.walk(fn.node)}
        sn = R.seen(fn, run, self.mode)
        loops: list[ast.AST] = []
        previous = site
        for anc in pathcond.ancestors(site):
            if anc is fn.node:
                break
            if isinstance(anc, R.LOOPS) and previous in anc.body:
                loops.append(anc)
            previous = anc
        keyed = {id(r) for r in run.found_by}

        def in_loop(stmt: ast.stmt) -> bool:
            return any(a in loops for a in pathcond.ancestors(stmt))

        once = all(sum(id(run.records[i]) in keyed for i, *_ in sn.hits.get(row.line, ())) <= 1
                   for row in fn.rows if in_loop(row.stmt))

        def read(row: R.Row) -> list[int]:
            at = sn.unambiguous.get(row.line, [])
            if loops and in_loop(row.stmt):
                return [i for i in at if id(run.records[i]) in keyed] if once else []
            return at

        held: list[T.Said] = []
        false: tuple[str, T.Said | str] | None = None
        for g, at in zip(guards, ifs_of, strict=True):
            on: list[T.Said] = []
            read_false = False
            text = _at_its_line(g)
            for one in at:
                here, env = [], E.Env()
                # an if outside the function (one the def sits in) is not this frame's
                terms = sorted(PR.chains(text), key=lambda t: (text.find(t), t)) if id(one) in own else []
                for t in terms:
                    if (got := CH.printed_at(fn, sn, one, t, read, tuple(loops) if once else ())) is not None:
                        here.append(T.Said("term-printed", {"term": t, "value": got[0], "where": got[1]}))
                        env.put_printed(t, got[0], got[1])
                        if not quiet:
                            self.say(f"  {t} = {got[0]!r} ({got[1]})")
                        if printed is not None:
                            printed.setdefault((one.lineno, t), got)
                on += here
                read_false = read_false or bool(here and not _orders_texts(text, env) and _guard_read(text, env) is False)
            said: T.Said | str = T.Said("held-on-values", {"values": on}) if on else ""
            held.append(T.Said("guard-held-on", {"guard": g, "on": said}))
            if false is None and read_false:
                false = (g, said)
        return held, false

    def _field_choice(self, run: Any) -> SR.Choose:
        """S4b's choice of a field's bindings and changes for a read on *run*'s path (:mod:`field_order`), each said
        once: what it took, what it did not and why."""
        said: dict[tuple[Any, ...], FO.Choice] = self.__dict__.setdefault("_field_choices", {})

        def choose(ff: Any, rel: str, line: int, keys: frozenset[str] | None, of_self: bool) -> FO.Choice:
            key = (ff.owner, ff.field, rel, line, keys, of_self, id(run))
            if key not in said:
                c = said[key] = FO.choose(ff, rel, line, run, self.mode, keys, of_self)
                bound = {b.at for b in ff.bindings}
                left = [f"{at} ({why})" for at, why in c.dropped if at in bound]
                changes: dict[str, int] = {}
                for at, why in c.dropped:
                    if at not in bound:
                        changes[why] = changes.get(why, 0) + 1
                self.say(f"  {ff.owner.rpartition('::')[2]}.{ff.field} read at {rel}:{line}"
                         + (f" (taking {', '.join(sorted(keys)) or 'which object'})" if keys is not None else "")
                         + f": took {', '.join(b.at for b in c.bindings) or 'no binding'}"
                         + (f"; changes between: {', '.join(x.at for x in c.changes)}" if c.changes else "")
                         + (f"; not taken: {', '.join(left)}" if left else "")
                         + (f"; changes not taken: {', '.join(f'{n} {why}' for why, n in sorted(changes.items()))}" if changes else "")
                         + (f"; {c.entry}" if c.entry else ""))
            return said[key]

        return choose

    def _arm_terms(self, fn: R.Function, owner: str, line: int, guards: list[str], ifs: list[list[ast.If]],
                   printed: dict[tuple[int, str], tuple[str, str]], run: Any, window: tuple[str, str] | None,
                   seen: frozenset[str] = frozenset(), pad: str = "  ") -> tuple[list[SR.Term], SR.End]:
        """Each term of the guards of *fn*'s arm at *line* (*guards*, read at *ifs*) with where its ways end (S,
        gm-design.md): read up the callers where a way ends at a parameter (:meth:`_with_callers`), a stop at a statement
        in an ``except`` handler of *fn* named (S3's to read), the changes on a way read (S4c, :meth:`_s4c`); and the
        arm's entry.  *printed*: what its run printed at its tests (:meth:`_held_on`); *seen*: the arms read on the way
        here, each read once.  With no run or no window (an attribute's arm read on no run), the changes on a way stay
        on it, named: S4c reads the run."""
        base = SR.terms(fn, owner, list(zip(guards, ifs, strict=True)), choose=self._field_choice(run))
        self._say_cuts(pad)
        terms = self._read_by_its_run(fn, [self._with_callers(self._raised(t, fn, run, pad), fn, owner, run, window)
                                           for t in base], printed, pad)
        out: list[SR.Term] = []
        for t in terms:
            t = replace(t, ends=tuple(replace(e, handler=h) if (h := _handler_line(fn, e)) else e for e in t.ends))
            self.say(f"{pad}term {t.term!r} of {t.guard!r}: " + ", ".join(_way_text(e) for e in t.ends))
            out.append(self._held_drops(self._s4c(t, fn, printed, run, window, seen | {f"{fn.rel}:{line}"}, pad), pad)
                       if run is not None and window is not None else t)
        entry = self._entry_end(fn, owner, line, run, window, out)
        self.say(f"{pad}entry of {fn.shown}: {_way_text(entry)}")
        return out, entry

    def _read_by_its_run(self, fn: R.Function, terms: list[SR.Term], printed: dict[tuple[int, str], tuple[str, str]],
                         pad: str) -> list[SR.Term]:
        """Q6 (out/gm/qk-design.md) on the terms of an arm, with what its run printed at their tests (*printed*, V's:
        :meth:`_held_on`): (c) a term its test reads only after an operand of ``and`` or ``or`` that decided the test on
        the run is not read by it -- one end, :data:`SR.UNREAD`, naming the operand; (b) a container made empty its test
        reads for its truth alone (``SR.truth_only``: its ways its empty start, its puts and the container where a
        change may take out of it) is its empty start and the container where the run printed it empty, and its puts
        where not.  A term with no print at its test, as it was."""
        out: list[SR.Term] = []
        for t in terms:
            stmt = R.statement_at(fn.node, t.line) if t.line else None
            test = getattr(stmt, "test", None)
            at = {term: got for (line, term), got in printed.items() if line == t.line}
            if not isinstance(test, ast.expr) or not at:
                out.append(t)
                continue
            if (decided := _short_circuit(test, t.term, at)) is not None:
                self.say(f"{pad}term {t.term!r} of {t.guard!r}: its run's test did not read it: {decided!r} decided the test")
                out.append(replace(t, ends=(SR.End(t.term, SR.UNREAD, "short-circuit", f"{fn.rel}:{t.line}",
                                                   decided=decided),)))
                continue
            got = at.get(t.term)
            empty = _printed_empty(got[0]) if got is not None else None
            truth = all((e.end == SR.CONSTANT and e.why in (SR.MADE_EMPTY, SR.PUT)) or (e.end == SR.STOP and e.why == SR.CONTAINER)
                        for e in t.ends)
            kept = tuple(e for e in t.ends if (e.why != SR.PUT if empty else e.why == SR.PUT))
            if got is None or empty is None or not truth or not kept or len(kept) == len(t.ends) or not SR.truth_only(test, t.term):
                out.append(t)
                continue
            self.say(f"{pad}term {t.term!r} of {t.guard!r}: its run printed it {'empty' if empty else 'not empty'} at its "
                     f"test ({got[1]}): " + ", ".join(_way_text(e) for e in kept))
            out.append(replace(t, ends=kept))
        return out

    def _held_drops(self, t: SR.Term, pad: str) -> SR.Term:
        """Q7's (f) on a term whose ways end at literals of different values (K75's premise 3: a local two arms bind,
        ``parse_result``'s 0 and 2): a literal that, the term holding it, would turn the test its run held false is not
        what the term held there -- that way goes; where every literal would, none does (the test does not tell)."""
        lits = [e for e in t.ends if e.end == SR.CONSTANT and e.why not in SR.NO_LITERAL and not e.part and not e.changes]
        if len({e.why for e in lits}) < 2 or (held := _held_test(t.guard)) is None:
            return t
        out = tuple(e for e in t.ends if e not in lits or not _held_rules_out(held, t.term, (e.why,)))
        if len(out) == len(t.ends) or not any(e in lits for e in out):
            return t
        self.say(f"{pad}term {t.term!r} of {t.guard!r}: its run held the test, so not "
                 + ", ".join(f"{e.why} at {e.at}" for e in t.ends if e not in out) + " (Q7's (f))")
        return replace(t, ends=out)

    def _s4c(self, t: SR.Term, fn: R.Function, printed: dict[tuple[int, str], tuple[str, str]], run: Any,
             window: tuple[str, str], seen: frozenset[str], pad: str) -> SR.Term:
        """*t* with the changes on each of its ways read (S4c, out/gm/gm5-design.md): a change that stores a value other
        than what the run printed of the part it changes is not what the read reads (V); of those left, one the run
        shows did not run is dropped (H2), and the one left, where its run shows it ran, or where its function's rows
        are in the run and it stores what the run printed (by elimination, as H2 leaves an arm), is read as an arm
        (:meth:`_change_arm`).  Where none is chosen, those left stay on the way, named.  Q7 (out/gm/qk-design.md)
        reads what the chosen change made as the end of the part the test takes (:meth:`_made`)."""
        if not any(e.changes for e in t.ends):
            return t
        stmt = R.statement_at(fn.node, t.line) if t.line else None
        test = getattr(stmt, "test", None)
        taken = SR.parts_taken(test, t.term, fn.rel, t.line) if isinstance(test, ast.expr) else None
        return replace(t, ends=tuple(self._made(e, t, taken, fn, printed, run, window, seen, pad) if e.changes else e
                                     for e in t.ends))

    def _made(self, e: SR.End, t: SR.Term, taken: frozenset[str] | None, fn: R.Function,
              printed: dict[tuple[int, str], tuple[str, str]], run: Any, window: tuple[str, str], seen: frozenset[str],
              pad: str) -> SR.End:
        """S4c on the way *e* of *t* (:meth:`_s4c`); *taken* the parts *t*'s test takes off its value.

        Q7 (out/gm/qk-design.md): what the test takes ends at the statement that last wrote it on the run.  A change
        whose value would turn the test its run held the other way is not that one ((f), :func:`_held_rules_out`).  The
        change chosen ends the way at the literal it stores, where it writes the one part the test takes by name and its
        run shows it ran, or, left by elimination, the object's start held another value of it than the run printed
        (:func:`SR.part_start`; K75's premise 2); its site is read as an arm as before (the term ends where the arm does,
        :func:`SR.arm_ends`).  Where every change is ruled out, the start ends it (:meth:`_start_end`)."""

        def part(c: SR.Changed) -> str:
            # the part of the value a change makes: its attribute, or, where it does not say, the one part the test takes
            if c.key.startswith("."):
                return c.key
            return next(iter(taken)) if c.key == "*" and taken is not None and len(taken) == 1 else ""

        shown = self._parts_printed(e, t, {p for c in e.changes if (p := part(c))}, fn, printed, run)
        held = _held_test(t.guard)
        left: list[SR.Changed] = []
        # the changes V read as storing what the run printed of the part they change
        printed_as: set[SR.Changed] = set()
        for c in e.changes:
            got = shown.get(part(c))
            values = SR.stored(c.record)
            if got is not None and values is not None and not any(SR.same_value(v, got[0]) for v in values):
                self.say(f"{pad}  V: {_change_text(c)} stores {' or '.join(values)}, not the {got[0]!r} printed of "
                         f"{t.term}{part(c)} ({got[1]})")
                continue
            if values is not None and part(c) and _held_rules_out(held, t.term + part(c), values):
                self.say(f"{pad}  F: {_change_text(c)} stores {' or '.join(values)}: {t.term}{part(c)} so would turn "
                         f"{t.guard!r}, which its run held, the other way (Q7's (f))")
                continue
            if got is not None and values is not None:
                printed_as.add(c)
            left.append(c)
        ran = {c: self._change_ran(c, run, window, pad) for c in left}
        kept = tuple(c for c in left if ran[c][0] != "not-run")
        self.say(f"{pad}  S4c on {e.end} ({e.why}): {len(e.changes)} change(s), {len(left)} left by V; H2: "
                 + (", ".join(f"{_change_text(c)} {ran[c][0]}{f' ({ran[c][1]})' if ran[c][1] else ''}" for c in left) or "none"))
        if not kept and (at_start := self._start_end(e, taken, pad)) is not None:
            return at_start
        if len(kept) != 1 or not (ran[kept[0]][0] == "shown" or (ran[kept[0]][0] == "kept" and kept[0] in printed_as)):
            return replace(e, changes=kept)
        c = kept[0]
        how = "shown" if ran[c][0] == "shown" else "left"
        made = self._change_arm(c, ran[c][1], how, run, window, seen, pad + "    ")
        p = part(c)
        got = shown.get(p)
        printed_part = got[0] if got is not None else None
        # the start a class's __init__ binds is an object's that a constructor made and nothing filled (a row's object,
        # filled or returned, starts at the row)
        made_only = e.end == SR.STOP and not e.filled and bool(e.made_of)
        start = SR.part_start(e.made_of, p) if how == "left" and p and made_only else None
        same = start is not None and printed_part is not None and any(SR.same_value(v, printed_part) for v, _ in start)
        if how == "left" and p:
            unread = start is None and made_only and SR._field_froms().get((e.made_of, p[1:])) is None
            made = replace(made, part=p, start=tuple(v for v, _ in start or ()), printed_start=same,
                           no_record=f"{e.made_of}{p}" if unread else "")
        values = SR.stored(c.record)
        known = how == "shown" or (start is not None and printed_part is not None and not same)
        if values is None or len(values) != 1 or not p or c.key != p or taken != frozenset({p}) or not known:
            return replace(e, changes=kept, made=made)
        self.say(f"{pad}  Q7: {t.term}{p} last written by {_change_text(c)}: {values[0]}"
                 + ("" if start is None else f" (its start held {' or '.join(v for v, _ in start)}, not the printed {printed_part!r})"))
        return replace(e, end=SR.CONSTANT, why=values[0], at=c.at, changes=(), made=made, part=p, made_at=e.at)

    def _start_end(self, e: SR.End, taken: frozenset[str] | None, pad: str) -> SR.End | None:
        """Q7's (a), every change on the way *e* ruled out: what the test takes is what the object held at its start --
        the row a fill put (*e.filled*), or the one literal its class's ``__init__`` binds the one part the test takes
        to (:func:`SR.part_start`); None where neither is read, or the way ends at no object a constructor made (a row's
        object, filled or returned, ends at the row as it is)."""
        if e.end != SR.STOP:
            return None
        if e.filled:
            self.say(f"{pad}  Q7: every change ruled out: what the test takes is the row {', '.join(e.filled)} put")
            return replace(e, end=SR.ROW, changes=())
        p = next(iter(taken)) if taken is not None and len(taken) == 1 else ""
        start = SR.part_start(e.made_of, p) if p else None
        if start is None or len(start) != 1:
            return None
        value, at = start[0]
        self.say(f"{pad}  Q7: every change ruled out: {p} is {value}, as the object's start at {at} binds it")
        return replace(e, end=SR.CONSTANT, why=value, at=at, changes=(), part=p, made_at=e.at)

    def _parts_printed(self, e: SR.End, t: SR.Term, wanted: set[str], fn: R.Function,
                       printed: dict[tuple[int, str], tuple[str, str]], run: Any) -> dict[str, tuple[str, str]]:
        """What the run printed of each part (*wanted*, one step: ``.status_code``, ``[disp]``) of the value *t* reads
        on its way *e*, with where: at *t*'s test (:meth:`_held_on`'s), or after the way's first binding, in its
        function, where every change on the way is before that binding -- none is of the read's own field record or of
        a name of the arm (266's ``plugin done with 2`` after 265's binding of ``self.add_result``: S4c's V)."""
        out = {p: got for p in wanted if (got := printed.get((t.line, t.term + p))) is not None}
        rest = wanted - set(out)
        read_at = f"{fn.rel}:{t.line}"
        if rest and e.via and all(c.before and c.before != read_at for c in e.changes):
            out.update(self._printed_after(e.via[0], rest, run))
        return out

    def _printed_after(self, at: str, parts: set[str], run: Any) -> dict[str, tuple[str, str]]:
        """What *run* printed, after the binding at *at* (``file:line``) in its function, of each of *parts* of what it
        binds, nothing between that can change it (``changed.printed_at``)."""
        rel, _, ln = at.rpartition(":")
        node = R.def_holding(rel, int(ln)) if ln.isdigit() else None
        if node is None:
            return {}
        bfn = R.function(rel, node.name, [int(ln)])
        stmt = R.statement_at(bfn.node, int(ln))
        targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target] if isinstance(stmt, ast.AnnAssign) else []
        if stmt is None or len(targets) != 1:
            return {}
        bound = CH.Point(bfn, stmt, [stmt])
        sn = R.seen(bfn, run, self.mode)

        def read(row: R.Row) -> list[int]:
            # a row after the binding: before it, what it binds is another value
            return sn.unambiguous.get(row.line, []) if CH.precedes(bound, CH.Point(bfn, row.stmt, [row.stmt], True), bfn) else []

        out: dict[str, tuple[str, str]] = {}
        for p in sorted(parts):
            chain = ast.unparse(targets[0]) + p
            if (got := CH.printed_at(bfn, sn, stmt, chain, read)) is not None:
                out[p] = got
                self.say(f"    {chain} = {got[0]!r} ({got[1]}, after its binding at {at})")
        return out

    def _change_ran(self, c: SR.Changed, run: Any, window: tuple[str, str], pad: str) -> tuple[str, str]:
        """H2 on a change: each call it was made through and its statement judged as a candidate in the arm's run
        (:meth:`_site_judged`): ``not-run`` where one is excluded; else ``shown`` where one is positive, ``kept`` where
        one is kept (its function's rows in the run, none printed only through the line), with the deepest such -- the
        site S4c reads as an arm; else ``undecided``."""
        verdicts = [(at, self._site_judged(at, run, window, pad)) for at in (*c.made_by, c.at)]
        if (no := next((at for at, k in verdicts if k == "excluded"), None)) is not None:
            return "not-run", no
        for kind, said in (("positive", "shown"), ("kept", "kept")):
            if found := [at for at, k in verdicts if k == kind]:
                return said, found[-1]
        return "undecided", ""

    def _site_judged(self, at: str, run: Any, window: tuple[str, str], pad: str) -> str:
        """H2's verdict on the line at *at* (``file:line``) as a candidate in *run* (its must rows, :func:`judge`), once
        per run; ``unexamined`` where its function's rows are not looked for in the run's log (:meth:`files_of`): their
        absence there says nothing."""
        cache: dict[tuple[str, int], str] = self.__dict__.setdefault("_sites_judged", {})
        if (at, id(run)) in cache:
            return cache[(at, id(run))]
        rel, _, ln = at.rpartition(":")
        node = R.def_holding(rel, int(ln)) if ln.isdigit() else None
        kind = "unexamined"
        if node is not None:
            log = getattr(getattr(run, "log", None), "name", "")
            # where its rows are looked for, as :meth:`_call_shown` looks
            owner = R.owner_of(rel, node)
            today = sorted({f for st in getattr(R._prints_of(owner, [int(ln)]), "statements", None) or [] for f in st.log_files})
            files = self.files_of(owner, today, [int(ln)])
            if log and log in files:
                j = judge([Candidate(rel, node.name, [int(ln)], run=run)], {}, mode=self.mode, decide=self.decide, window=window)
                v = j.verdicts[0] if j.verdicts else None
                kind = v.kind if v is not None else kind
                self.say(f"{pad}    {v.describe() if v is not None else at + ': no verdict'}")
            else:
                self.say(f"{pad}    {at} ({node.name}): prints nothing to {log or 'the run'}"
                         + (f" (its log: {', '.join(sorted(files))})" if files else ""))
        cache[(at, id(run))] = kind
        return kind

    def _change_arm(self, c: SR.Changed, site: str, how: str, run: Any, window: tuple[str, str], seen: frozenset[str],
                    pad: str) -> SR.Made:
        """The change *c* S4c chose (*how*: ``shown`` by its run, or ``left`` by elimination), read as an arm at *site*
        (``file:line``, the deepest of the calls it was made through and its statement H2 left): its guards, each term's
        ends, and its entry, as the arm itself is (:meth:`_arm_terms`); its guards alone where that arm was read on the
        way here, or they are not placed."""
        rel, _, ln = site.rpartition(":")
        node = R.def_holding(rel, int(ln))
        if node is None:
            return SR.Made(c, site, rel, how)
        owner = R.owner_of(rel, node)
        fn = R.function(rel, node.name, [int(ln)])
        stmt = R.statement_at(fn.node, int(ln))
        if stmt is None:
            return SR.Made(c, site, owner, how)
        guards = pathcond.arm_condition(stmt)
        self.say(f"{pad}made at {site} ({node.name}), {'its run showing it' if how == 'shown' else 'left by elimination'}: "
                 f"guards {guards}")
        raised = self._raise_into(fn, int(ln), run, pad)
        if site in seen:
            return SR.Made(c, site, owner, how, tuple(guards), **raised)
        printed: dict[tuple[int, str], tuple[str, str]] = {}
        _held, false = self._held_on(fn, stmt, guards, run, printed)
        ifs = _guard_ifs(fn, stmt, guards) if false is None else None
        if ifs is None:
            return SR.Made(c, site, owner, how, tuple(guards), **raised)
        terms, entry = self._arm_terms(fn, owner, int(ln), guards, ifs, printed, run, window, seen, pad)
        return SR.Made(c, site, owner, how, tuple(guards), tuple(terms), entry, **raised)

    def _raise_into(self, fn: R.Function, line: int, run: Any, pad: str) -> _Raise:
        """K's form (out/gm/qk-design.md), an arm at *line* in an ``except`` handler of *fn* (S3): the exception its
        run's tracebacks show leaving the try's body into that handler (:func:`_raised_through`,
        :func:`_entered_handler`), where it was raised and the handler -- where the arm's own terms end, its stop (not
        an end: the raise has terms of its own); none where they do not show it."""
        if run is None or (got := _handler_around(fn, line)) is None:
            return {}
        tr, h = got
        if (shown := _raised_through(fn, tr, run)) is None:
            return {}
        exc_type, raised, other = shown
        where = raised + (f" ({other}: its run's code, not the pin's)" if other else "")
        if _entered_handler(fn, tr, exc_type, SR.handler_classes(fn)) != h.lineno:
            self.say(f"{pad}  K: its run's traceback shows {exc_type} raised at {where} leaving the try at "
                     f"{fn.rel}:{tr.lineno}; the except handler at {fn.rel}:{h.lineno} is not told to have taken it")
            return {}
        self.say(f"{pad}  K: its run's traceback: {exc_type} raised at {where}, caught by the except handler at "
                 f"{fn.rel}:{h.lineno}")
        return {"raised_type": exc_type, "raised_at": raised, "caught_at": f"{fn.rel}:{h.lineno}", "raised_other": other}

    def _with_callers(self, t: SR.Term, fn: R.Function | None, owner: str, run: Any, window: tuple[str, str] | None,
                      ask: bool = True) -> SR.Term:
        """*t* with each way that ends at a parameter read up its callers (:meth:`_param_end`): a parameter of the arm's
        function, or of the method a field's binding is in (*of*, S4a; the run's prints at the arm's entry say nothing of
        it; read up the calls the run's path entered it by, S4b's (e)), with the steps the read still takes off it (Q2's
        (c)), keeping the bindings the way went through and the changes on it after the binding (out/gm/gm5-design.md
        G4).  *ask* False reads the map's records alone, no caller's run asked (a split's terms, Q2's (b))."""
        ends = tuple(x for e in t.ends for x in (
            tuple(replace(y, via=e.via, changes=(*e.changes, *y.changes), caught=e.caught)
                  for y in self._param_end(e.term, e.of or owner, e.param, None if e.of else fn, run, window, e.entered,
                                           e.steps, ask, e.truth))
            if e.why == SR.PARAMETER and e.param else (e,)))
        return replace(t, ends=ends)

    def _raised(self, t: SR.Term, fn: R.Function, run: Any, pad: str) -> SR.Term:
        """K (out/gm/qk-design.md): *t* with what its run's tracebacks say of each ``try`` of *fn* a way of it leaves by
        a handler of -- bound in one, or read off the exception one names (*caught*).  Where they show the exception
        that left the try's body (:func:`_raised_through`) and the handler it entered (:func:`_entered_handler`), a way
        that cannot reach the guard through that handler goes: one bound in another handler of the try, or, the guard
        after the try, one bound before the handler where its body binds the term again on every path; a field of the
        exception it names is read up the raise the traceback ends at (*entered*).  Where they do not tell, *t* as it
        was."""
        if run is None:
            return t

        def line_of(e: SR.End) -> int:
            rel, _, ln = e.at.rpartition(":")
            return e.caught or (int(ln) if rel == fn.rel and ln.isdigit() else 0)

        tries: dict[int, ast.Try] = {}
        for e in t.ends:
            if (line := line_of(e)) and (got := _handler_around(fn, line)) is not None:
                tries.setdefault(id(got[0]), got[0])
        if not tries:
            return t
        known = SR.handler_classes(fn)
        ends = list(t.ends)
        for tr in tries.values():
            if (shown := _raised_through(fn, tr, run)) is None:
                continue
            exc_type, raised, other = shown
            where = raised + (f" ({other}: its run's code, not the pin's)" if other else "")
            if not (entered := _entered_handler(fn, tr, exc_type, known)):
                self.say(f"{pad}  K: term {t.term!r} of {t.guard!r}: its run's traceback shows {exc_type} raised at "
                         f"{where} leaving the try at {fn.rel}:{tr.lineno}; which handler took it, not told")
                continue
            h = next(x for x in tr.handlers if x.lineno == entered)
            rebinds = t.line > (tr.end_lineno or tr.lineno) and _binds_on_every_path(h, t.term)
            kept: list[SR.End] = []
            gone: list[str] = []
            for e in ends:
                line = line_of(e)
                around = _handler_around(fn, line) if line else None
                if around is not None and around[0] is tr and around[1].lineno != entered:
                    gone.append(e.at or f"{fn.rel}:{line}")
                elif e.caught == entered:
                    # the raise read up only where the pin holds its source there
                    kept.append(replace(e, entered=(raised,)) if e.param and not other else e)
                elif rebinds and line and line < entered and (around is None or around[0] is not tr):
                    gone.append(e.at)
                else:
                    kept.append(e)
            self.say(f"{pad}  K: term {t.term!r} of {t.guard!r}: its run's traceback: {exc_type} raised at {where}, "
                     f"caught by the except handler at {fn.rel}:{entered}"
                     + (f"; a way that cannot reach the guard through it goes: {', '.join(gone)}" if gone else ""))
            ends = kept
        return replace(t, ends=tuple(ends))

    def _say_cuts(self, pad: str) -> None:
        """The ways a callee's test ruled out on its call's constants since last said (CRO (ii), ``SR.take_cuts``)."""
        for cut in SR.take_cuts():
            self.say(f"{pad}ruled out by a callee's test on its call's constants (CRO (ii)): {cut}")

    def _param_end(self, term: str, owner: str, param: str, fn: R.Function | None, run: Any, window: tuple[str, str] | None,
                   entered: tuple[str, ...] = (), steps: tuple[str, ...] = (), ask: bool = True,
                   truth: bool = False) -> tuple[SR.End, ...]:
        """Where the value *owner*'s *param* was handed comes from (S, gm-design.md): its ways up the map's callers
        (``SR.param_ways``; with *entered*, only those by the calls the run's path entered *owner* by, S4b's (e); with
        *steps*, the parts the read takes off the value, Q2's (c)); a constant whose ``str`` is not what the run printed
        of the parameter at its entry (*fn*'s; none read where *fn* is None) is not the one; of those left, the call a
        caller's run shows (:meth:`_call_shown`; none asked where *ask* is False) is -- not asked where two or more are
        left and all come by one call, whose run cannot tell them apart: where they are every way left, the ``if`` that
        tells them apart is read (:meth:`_split`, out/gm/s4-design.md rule 6's (a)) and the stop holds it.  With none
        left, only API parameters (any value) or the database end it, or the one way left where it ends; where every way
        ends and none is a literal a run could tell apart (a constant's part, the database), it ends at each
        (out/gm/qk-design.md Q2); else it stops.  *truth*: the read takes the value's truth alone, so a container made
        empty comes in as its empty start and its puts (Q6's (a))."""
        # the calls by the path a frame of the run's traceback stands at decide too (K)
        key = (owner, param, fn is None, entered, steps, ask,
               frozenset(set(entered) & _traceback_frames(run)) if entered else frozenset(), truth)
        cache: dict[tuple, tuple[SR.End, ...]] = self.__dict__.setdefault("_param_ends", {})
        if key in cache:
            return tuple(replace(e, term=term) for e in cache[key])
        busy: set[tuple] = self.__dict__.setdefault("_param_busy", set())
        if key in busy:
            # a split's term reading the parameter whose split it is
            return (SR.End(term, SR.STOP, "cycle", "", param=param),)
        busy.add(key)
        try:
            got = self._param_end_read(term, owner, param, fn, run, window, entered, steps, ask, truth)
        finally:
            busy.discard(key)
        cache[key] = got
        return got

    def _param_end_read(self, term: str, owner: str, param: str, fn: R.Function | None, run: Any,
                        window: tuple[str, str] | None, entered: tuple[str, ...], steps: tuple[str, ...],
                        ask: bool, truth: bool = False) -> tuple[SR.End, ...]:
        """:meth:`_param_end`, not cached."""
        ways = SR.param_ways(owner, param, entered=entered, steps=steps, truth=truth)
        self._say_cuts("  ")
        off = SR.calls_off_the_path(owner, param, entered)
        value = self._entry_value(fn, run, param) if fn is not None else None
        # a value made of a literal (an item of range(3)) is not the literal: it is not matched to what the run printed
        # a container's empty start (Q1) and a put into it (Q6) are no literal a call hands: two of the other ways
        consts = [w for w in ways if w.end == SR.CONSTANT and w.why not in (SR.MADE_EMPTY, SR.PUT)
                  and (value is None or (w.why != SR.THROUGH and SR.same_value(w.value, value)))]
        rest = [w for w in ways if w.end != SR.CONSTANT or w.why in (SR.MADE_EMPTY, SR.PUT)]
        # a caller's run tells constants apart only by the calls they come by
        calls = {w.route[0] for w in consts if w.route}
        one_call = len(consts) > 1 and len(calls) == 1
        # a call the caller's run shows tells a constant apart only where every way it comes by that the print leaves
        # ends there (a call handing a container whose puts are ways hands each of them, adder_atlas_plugin 677's
        # cont_zip_map; a call handing True or False, job_api 556's prodManager, is told apart by the print first)
        possible = [x for x in ways if value is None or x.end != SR.CONSTANT or x.why in SR.NO_LITERAL
                    or SR.same_value(x.value, value)]
        alone = [w for w in consts if w.route and all(
            (x.end, x.value, x.why) == (w.end, w.value, w.why) for x in possible if x.route[:1] == w.route[:1])]
        # K: a call the run's path entered by (*entered*) that a frame of its run's traceback stands at ran -- the raise
        # that made the exception a handler took; no caller's run asked
        traced = [w for w in alone if w.route[0] in entered and w.route[0] in _traceback_frames(run)]
        # a literal a run could tell apart is left (else each way is read where it ends, Q2 below)
        literal = any(w.end == SR.CONSTANT and w.why not in SR.NO_LITERAL for w in ways)
        # Q5: which of them ran is asked of the caller's own run where each is bound in the caller -- where the
        # traceback has not told it (K) and a literal is left
        q5 = (one_call and not rest and ask and len(traced) != 1 and literal
              and self._bindings_in_caller(consts) is not None)
        printed = f"its run printed {value!r} at its entry" if fn is not None else f"{owner} is not the arm's: no print read"
        self.say(f"  {param}: {len(ways)} way(s) in by the map's callers; {printed}; constant(s) left: "
                 + (", ".join(f"{w.value or w.why} at {w.route[0]}" for w in consts if w.route) or "none")
                 + ("; other ways: " + ", ".join(sorted({f"{w.end} ({w.why})" for w in rest})) if rest else "")
                 + (f"; read by the call(s) the run's path entered it by ({', '.join(entered)}), left out: "
                    + ", ".join(f"{n} by {c}" for c, n in sorted(Counter(off).items())) if off else "")
                 + (f"; all come by {next(iter(calls))}: " + ("which binding ran, by its caller's run (Q5)" if q5
                                                              else "its caller's run not asked") if one_call else "")
                 + (f"; the read takes {''.join(steps)} off it" if steps else ""))
        if traced:
            self.say(f"    {param}: the call at {', '.join(w.route[0] for w in traced)} stands in its run's traceback: it ran")
        judged = [] if traced or one_call or not ask else [(w, self._caller_shows(w, window, run)) for w in alone]
        shown = traced or [w for w, s in judged if s]
        # a caller not judged (its send's entry asked, not answered) may be the one that ran: no other is told alone
        unjudged = [w.route[0] for w, s in judged if s is None]
        if len(shown) == 1 and unjudged:
            self.say(f"    {param}: the call at {shown[0].route[0]} shown, and not told alone: the caller(s) at "
                     f"{', '.join(unjudged)} not judged")
        if len(shown) == 1 and not unjudged:
            # a constant-through keeps its why: its value is not the literal's
            return (SR.End(term, SR.CONSTANT, shown[0].value or shown[0].why, shown[0].route[0], route=shown[0].route),)
        if not consts and rest and all(r.end == SR.API for r in rest):
            return (SR.End(term, SR.API, "no_caller", ", ".join(r.route[0] for r in rest if r.route)),)
        if not consts and rest and all(r.end == SR.DATABASE for r in rest):
            return (SR.End(term, SR.DATABASE, SR.PARAMETER, ""),)
        if not consts and len(rest) == 1:
            # the one way left is where it ends
            return (SR.End(term, rest[0].end, rest[0].why, rest[0].route[0] if rest[0].route else "", route=rest[0].route),)
        if ways and not literal:
            # no literal a run could tell apart is left (modTime: a statement's rows, a part of the None 2371 binds,
            # the + at 2406 not read): each way where it ends or stops, the term ending where every one ends
            # (out/gm/qk-design.md Q2); a place reached by several routes is said once, by the first
            each: dict[tuple, SR.End] = {}
            for w in ways:
                at = w.end_at or (w.route[0] if w.route else "")
                each.setdefault((w.end, w.value or w.why, at, w.changes), SR.End(
                    term, w.end, w.value or w.why, at, route=w.route, changes=w.changes, returned=w.returned,
                    returned_line=w.returned_line))
            if len(each) < len(ways):
                self.say(f"    {param}: {len(ways)} way(s) end at {len(each)} place(s), each said by its first route")
            return tuple(each.values())
        # Q5: every way left a constant the one call hands -- which binding ran, by its caller's own run
        chosen, asked = self._binding_shown(consts, window, getattr(run, "machine", None)) if q5 else (None, SR.NOT_ASKED)
        if chosen is not None:
            crel = chosen.route[0].rpartition(":")[0]
            return (SR.End(term, SR.CONSTANT, chosen.value or chosen.why, f"{crel}:{chosen.bound_at}", route=chosen.route),)
        # S2's (a): not told by it -- the if their bindings sit apart in
        split = self._split(consts, asked) if one_call and not rest else None
        return (SR.End(term, SR.STOP, "parameter-unchosen" if consts else SR.PARAMETER, "", param=param, split=split),)

    def _split(self, consts: list[SR.Way], asked: str = SR.NOT_ASKED) -> SR.Split | None:
        """S2's (a) (out/gm/s4-design.md rule 6): the constants *consts*, all handed by one call, told apart by the
        ``if`` statements their bindings sit in different arms of -- each binding's guards (``pathcond.arm_condition``,
        read at their ifs as H3 reads an arm's) less those every binding shares, an ``if`` read once -- and each term of
        those tests with its ends, by the map's records alone (``SR.terms``): the caller's run is not asked for them, so
        no field choice, change reading or entry is read there; a stop in an ``except`` handler is named (S3's).  *asked*:
        what the caller's run said of which binding ran (Q5, ``SR.Split.asked``).  None where a binding is not in the
        caller (bound in a callee the value came back from), its guards' ifs are not found, or no guard tells the
        bindings apart."""
        call = consts[0].route[0]
        rel, _, ln = call.rpartition(":")
        if any(not w.bound_at for w in consts) or not ln.isdigit():
            return None
        node = R.def_holding(rel, int(ln))
        if node is None:
            return None
        fn = R.function(rel, node.name, [int(ln)])
        owner = R.owner_of(rel, node)
        each: list[dict[tuple[str, tuple[int, ...]], list[ast.If]]] = []
        for w in consts:
            stmt = R.statement_at(fn.node, w.bound_at)
            if stmt is None:
                return None
            guards = pathcond.arm_condition(stmt)
            ifs = _guard_ifs(fn, stmt, guards)
            if ifs is None:
                return None
            each.append({(g, tuple(i.lineno for i in f)): f for g, f in zip(guards, ifs, strict=True)})
        shared = set(each[0]).intersection(*each[1:])
        # an if is read once, by the guard of the first binding that sits in it (its test's terms are the same)
        apart: dict[tuple[int, ...], tuple[str, list[ast.If]]] = {}
        for e in each:
            for (g, lines), f in e.items():
                if (g, lines) not in shared:
                    apart.setdefault(lines, (g, f))
        if not apart:
            return None
        pairs = list(apart.values())
        ifs_at = tuple(dict.fromkeys(f"{rel}:{n}" for lines in apart for n in lines))
        self.say(f"    told apart by the if at {', '.join(ifs_at)} ({node.name}; the bindings at "
                 + ", ".join(f"{rel}:{w.bound_at}" for w in consts) + f"), read by the map's records: {T.text(_split_asked(asked))}")
        # a way ending at a parameter is read up its callers by the map's records too, no run asked (Q2's (b))
        base = SR.terms(fn, owner, pairs)
        self._say_cuts("    ")
        terms = [replace(t, ends=tuple(replace(x, handler=h) if (h := _handler_line(fn, x)) else x for x in t.ends))
                 for t in (self._with_callers(u, None, owner, None, None, ask=False) for u in base)]
        for t in terms:
            self.say(f"      term {t.term!r} of {t.guard!r}: " + ", ".join(_way_text(x) for x in t.ends))
        return SR.Split(call, owner, ifs_at, tuple(g for g, _ in pairs), tuple(terms), asked)

    def _entry_value(self, fn: R.Function, run: Any, param: str) -> str | None:
        """What *run* printed of *fn*'s parameter *param* before any statement binds it again (``killJob`` prints
        ``code=60`` at 840, before 853's ``code = str(code)``); None where it printed none, or several."""
        rebound = [n.lineno for n in ast.walk(fn.node) if isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign))
                   and any(isinstance(x, ast.Name) and x.id == param for x in (n.targets if isinstance(n, ast.Assign) else [n.target]))]
        upto = min(rebound, default=None)
        sn = R.seen(fn, run, self.mode)
        values = {row.capture(run.records[i])[param].strip()
                  for row in fn.rows if upto is None or row.line < upto
                  for i, *_ in sn.hits.get(row.line, ()) if param in row.capture(run.records[i])}
        return values.pop() if len(values) == 1 else None

    def _caller_shows(self, w: SR.Way, window: tuple[str, str] | None, run: Any) -> bool | None:
        """Whether the caller's own run shows the call *w* comes by ran (:meth:`_call_shown`).  A way that came across a
        send (the map's ``HeldSource.sent``, SND) is asked as HOP's reader asks it (out/gm/qk-design.md, HOP (2)): the
        caller is a client in another process, so on every machine, over the arm's window opened windows.SLACK earlier
        and closed at the first line of the endpoint's pass the send entered (:meth:`_entry_pass`) -- what the caller
        printed after its send is not of this call -- else at the window's end.  A caller that prints after its call
        prints once the call has returned, after the endpoint's pass: copyArchive 881 ``Client.kill_jobs(jobs, 2)``,
        882 ``killJobs for Anal Active (...)`` at 15:14:55,289, after kill Done at 15:14:55,287 (MJE-7383377087).
        Where the window before the send shows no call, the caller's rows after its call are asked from the pass's
        first line to its last and windows.SLACK more, and count only where every row that shows the call follows it
        (HOP (a'): each row read on the side of the send its code puts it on).  Where the ask of the endpoint's logs
        is not all answered, the callers are not judged (None): a pass not in logs not recorded yet is no pass that
        did not begin, and the window's end would take in what a caller printed after its send (NJE-7425252125, r3x38:
        an END on copyArchive 634 while that ask was open; out/gm/qk-design.md, the r4 fixes (1))."""
        machine = getattr(run, "machine", None)
        if not w.sent or window is None:
            return self._call_shown(w.route[0], window, machine)
        start = f"{W.when(window[0]) - W.SLACK:%Y-%m-%d %H:%M:%S}"
        # one entry's run is asked once per window, and its windows said once, however many calls send into it
        said: dict[tuple[str, str, tuple[str, str]], tuple[str, str, str | None] | None] = self.__dict__.setdefault("_send_windows", {})
        key = (w.sent_into, w.sent, (start, window[1]))
        if key not in said:
            entry, answered = self._entry_pass(w, (start, window[1]))
            if not answered:
                said[key] = None
                self.say(f"    across the send at {w.sent} into {w.sent_into.split('::')[-1]}: its callers not judged: the "
                         f"ask of the entry's logs from {start} to {window[1]} is not answered yet, and the first line of its "
                         "pass that holds the key would close their window")
            else:
                said[key] = (start, entry[0], entry[1]) if entry else (start, window[1], None)
                closed = ("the first line of the entry's pass that holds the key" if entry
                          else "the window's end: no pass of the entry holding the key began in it")
                self.say(f"    across the send at {w.sent} into {w.sent_into.split('::')[-1]}: its callers asked from "
                         f"{start} to {entry[0] if entry else window[1]} ({closed}), on every machine")
                if entry:
                    self.say(f"    and a caller's rows after its call from {entry[0]} to "
                             f"{W.when(entry[1]) + W.SLACK:%Y-%m-%d %H:%M:%S} (the pass's last line, {entry[1]}, and "
                             f"{int(W.SLACK.total_seconds() // 60)} min more), where the window before shows no call")
        got = said[key]
        if got is None:
            return None
        start, first, last = got
        if self._call_shown(w.route[0], (start, first), None, "for-a-call-across-a-send", {"send": w.sent}):
            return True
        if last is None:
            return False
        after = (first, f"{W.when(last) + W.SLACK:%Y-%m-%d %H:%M:%S}")
        return self._call_shown(w.route[0], after, None, "for-the-rows-after-a-call-across-a-send", {"send": w.sent},
                                after_call=True)

    def _entry_pass(self, w: SR.Way, window: tuple[str, str]) -> tuple[tuple[str, str] | None, bool]:
        """The first and last lines of the pass of the endpoint *w*'s send entered (``w.sent_into``) that holds the key
        and began last inside *window* -- job_api's kill ``kill Start ... job_ids: [...]`` at 16:39:21,051, before
        killJobs ran -- or None where its logs are not to hand or hold no such pass; and whether the ask of its logs
        was answered whole (False: not judged, :meth:`_caller_shows`).  A pass begins at a record the map reads as the
        endpoint's entry row whole (``rows.entries``): a mark many calls share makes one run of hours -- job_api's
        ``kill``, a day's 198 lines on one machine -- whose first line, or its first inside the window, is another
        call's (MJE-7395160958's 16:34:40,732, a kill of other jobs).  A run with no record read so begins no pass here:
        kept whole, its first line may be another call's, or one printed after the call began."""
        entry = w.sent_into
        lines = [n] if (n := SR.def_line(entry)) else []
        today = sorted({f for st in getattr(R._prints_of(entry, lines), "statements", None) or [] for f in st.log_files})
        files = self.files_of(entry, today, lines)
        if not files or not self.logs.names(files):
            self.say(f"    {entry.split('::')[-1]}: no log of it to hand ({', '.join(files) or 'none'})")
            return None, True
        asking = W.Asking(window, None, T.Said("for-the-entry-a-send-entered", {"entry": entry.split("::")[-1], "send": w.sent}))
        runs, unanswered, _holes = self._ask(self.args.key, files, asking, owners_of([entry]), whole=True)
        self._keep_unanswered(unanswered)
        if unanswered:
            return None, False
        rel, _, name = entry.partition("::")
        try:
            fn = R.function(rel, name.rpartition(".")[2], within=lines)
        except LookupError:
            self.say(f"    {name}: not found in {rel}, so no pass of it begins")
            return None, True
        passes = []
        for r in (r for rs in runs.values() for r in rs if of_mover(r, entry)):
            starts = R.entries(fn, r, self.mode)
            if not starts:
                continue
            for a, b in zip(starts, [*starts[1:], len(r.records)], strict=True):
                p = r.part(a, b)
                if p.found_by and window[0][:19] <= p.first[:19] <= window[1][:19]:
                    passes.append((p.first, p.last))
        return (max(passes) if passes else None), True

    def _call_shown(self, site: str, window: tuple[str, str] | None, machine: str | None, why: str = "for-a-call-handing-a-value",
                    facts: dict[str, Any] | None = None, after_call: bool = False) -> bool:
        """Whether the caller's own run shows the call at *site* (``file:line``) ran: H2's judgement of that line as a
        candidate, on the key's runs of the caller's logs over *window* on *machine* (the arm's run's: the call ran it in
        the caller's process) -- its must rows printed (positive).  *why* says what the ask is for, *facts* what more
        it names.  *after_call*: *window* lies after the call returned (:meth:`_caller_shows`), so only must rows the
        call's block prints after it show it there; one before it printed then is a later call's."""
        crel, _, ln = site.rpartition(":")
        node = R.def_holding(crel, int(ln))
        if node is None:
            return False
        cowner = f"{crel}::{node.name}"
        today = sorted({f for st in getattr(R._prints_of(cowner, [int(ln)]), "statements", None) or [] for f in st.log_files})
        files = self.files_of(cowner, today, [int(ln)])
        if not files or not self.logs.names(files):
            self.say(f"    {site} ({node.name}): no log of it to hand ({', '.join(files) or 'none'})")
            return False
        asking = W.Asking(window, machine, T.Said(why, {"site": site, **(facts or {})}))
        # asked as H2 asks a candidate's logs: by its function's service, its marks whole
        runs, unanswered, _holes = self._ask(self.args.key, files, asking, owners_of([cowner]), whole=True)
        self._keep_unanswered(unanswered)
        j = judge([Candidate(crel, node.name, [int(ln)], log_files=files)], runs, mode=self.mode, decide=self.decide, window=window)
        v = j.verdicts[0] if j.verdicts else None
        self.say(f"    {site}: {v.describe() if v is not None else 'no verdict'}")
        if not after_call or v is None or v.kind != "positive":
            return v is not None and v.kind == "positive"
        # the rows of the call's own block (rows.implying_rows) lie before the call or after it, as its statements run
        call = R.statement_at(node, int(ln))
        end = getattr(call, "end_lineno", None) or int(ln)
        before = [x for x in v.proven if x <= end]
        if v.proven and not before:
            return True
        self.say(f"    {site}: not read after its call: "
                 + (f"row(s) {before} print before the call, ending at {end}" if before else "shown by no row printed after it"))
        return False

    @staticmethod
    def _bindings_in_caller(consts: list[SR.Way]) -> tuple[str, ast.FunctionDef | ast.AsyncFunctionDef, list[int]] | None:
        """Q5: the caller's file, its function and the lines *consts* -- handed by one call -- are bound at, where each
        is a statement of that function; None where one is bound in a callee the value came back from, or not read."""
        call = consts[0].route[0]
        crel, _, ln = call.rpartition(":")
        lines = sorted({w.bound_at for w in consts if w.bound_at})
        if len(lines) != len(consts) or not ln.isdigit():
            return None
        node = R.def_holding(crel, int(ln))
        if node is None or any(R.statement_at(node, line) is None for line in lines):
            return None
        return crel, node, lines

    def _binding_shown(self, consts: list[SR.Way], window: tuple[str, str] | None,
                       machine: str | None) -> tuple[SR.Way | None, str]:
        """Q5 (out/gm/qk-design.md): which of the constants *consts*, all handed by one call and bound apart in its
        caller (add_main 130's False and 134's True, handed at 136), the caller's own run shows bound: H2's judgement
        of their binding lines as candidates on the key's runs of the caller's logs over *window* on *machine*, its
        marks whole (as :meth:`_call_shown` asks) -- the one positive, or None with the kind of the answer that did not
        tell it (``SR.Q5_ANSWERS``).  A loop around them that takes one element a turn prints every turn under one mark
        (add_main's ``<pid=...>``): the element the key's records print the key as is bound to it
        (:func:`which_ran.key_env`), so a row printed for another element's turn alone shows nothing of the key's
        (:func:`which_ran.other_elements`)."""
        if (at := self._bindings_in_caller(consts)) is None:
            return None, SR.NOT_ASKED
        call = consts[0].route[0]
        crel, node, lines = at
        cowner = f"{crel}::{node.name}"
        today = sorted({f for st in getattr(R._prints_of(cowner, lines), "statements", None) or [] for f in st.log_files})
        files = self.files_of(cowner, today, lines)
        if not files or not self.logs.names(files):
            self.say(f"    Q5: {call} ({node.name}): no log of it to hand ({', '.join(files) or 'none'}): its caller's "
                     "run not asked")
            return None, "no-log"
        asking = W.Asking(window, machine, T.Said("for-a-binding-a-call-hands", {"site": call}))
        runs, unanswered, _holes = self._ask(self.args.key, files, asking, owners_of([cowner]), whole=True)
        self._keep_unanswered(unanswered)
        c = Candidate(crel, node.name, lines, log_files=files)
        env = key_env(c, runs, self.args.key)
        j = judge([c], runs, mode=self.mode, decide=self.decide, window=window, env=env)
        for v in j.verdicts:
            self.say(f"    {call}: {v.describe()}")
        positive = [v.line for v in j.verdicts if v.kind == "positive"]
        if len(positive) > 1:
            return None, "several-shown"
        if not positive:
            return None, "unexamined" if any(v.kind == "unexamined" for v in j.verdicts) else "none-shown"
        chosen = next(w for w in consts if w.bound_at == positive[0])
        self.say(f"    Q5: {call} handed {chosen.value or chosen.why}, bound at {crel}:{chosen.bound_at}: its caller's run "
                 f"shows it" + (f" (the key's turn read by {', '.join(sorted(env.values))})" if env is not None else ""))
        return chosen, SR.NOT_ASKED

    def _entry_end(self, fn: R.Function, owner: str, line: int, run: Any, window: tuple[str, str] | None, terms: list[SR.Term]) -> SR.End:
        """Where the arm's function was entered from, read on every arm (S, gm-design.md): a function no corpus call
        enters starts there; else the call a parameter's chosen way came in by, or that a parameter the run printed at
        its entry is read up to; else the one call the map's ``EnteredFrom`` holds into it where nothing else may enter
        it (OWI, out/gm/qk-design.md), whose run is not asked; else, where every term ends, the call a caller's run
        shows entered it (F1, :meth:`_entered_by`); else, with nothing open, the one call the values its run printed at
        its entry leave (EBV, :meth:`_entered_by_values`); else it stops."""
        rec = SR.entered(owner, fn.node.lineno)
        if rec is None:
            return SR.End("entry", SR.STOP, "no-entered-from", "")
        if not rec.sites and not rec.open:
            return SR.End("entry", SR.ENTRY, "no-corpus-caller", f"{fn.rel}:{fn.node.lineno}")
        into = {f"{s.owner.partition('::')[0]}:{s.line}" for s in rec.sites}
        chosen = [e for t in terms for e in t.ends if e.route and e.route[-1] in into]
        if not chosen:
            params = [a.arg for a in [*fn.node.args.posonlyargs, *fn.node.args.args, *fn.node.args.kwonlyargs] if a.arg != "self"]
            for p in params:
                if self._entry_value(fn, run, p) is None:
                    continue
                got = self._param_end(p, owner, p, fn, run, window)
                # one way chosen, not every way of several
                if len(got) == 1 and ((got[0].route and got[0].route[-1] in into) or got[0].end == SR.API):
                    chosen.append(got[0])
                    break
        if chosen:
            e = chosen[0]
            return SR.End("entry", e.end, e.why, e.at, route=e.route)
        # the map's model: with nothing open the function ran only from its sites, so from the one there is
        if len(rec.sites) == 1 and not rec.open:
            site = next(iter(into))
            self.say(f"  entry: the map holds one call into {fn.shown}, at {site}, and no other way in")
            return SR.End("entry", SR.CALLER, "only-way-in", site)
        # the callers' logs are asked only where the entry is all the answer waits for (gm2-predictions.txt F4)
        if rec.sites and all(t.ended for t in terms) and (by := self._entered_by(owner, line, run, window, rec)) is not None:
            return by
        if rec.sites and not rec.open and (by := self._entered_by_values(fn, owner, run, into)) is not None:
            return by
        return SR.End("entry", SR.STOP, "entry-unchosen", "")

    def _entered_by_values(self, fn: R.Function, owner: str, run: Any, into: set[str]) -> SR.End | None:
        """EBV (out/gm/qk-design.md): the call into *fn* that the values its run printed at its entry leave, *into* being
        every call into it (the map's ``EnteredFrom`` with nothing open).  A call is not the one where every way of a
        parameter the run printed at its entry (:meth:`_entry_value`) that comes by it ends at a literal other than
        the print -- killJob's ``code`` printed '50': ``"51"`` at task_complex_module 5732 and ``"55"`` at
        job_complex_module 799 rule those calls out, its ways by TaskBuffer 1190 do not.  A default the call leaves is
        such a literal; a value a literal is only part of (:data:`SR.THROUGH`), a container's empty start or put, and
        any other end are not; a call no way of the parameter comes by stays.  The one call left is the entry; None
        where nothing was printed, none was ruled out, several are left, or a way of a printed parameter comes by no
        call of *into* (it stops at *fn* itself)."""
        if run is None:
            return None
        params = [a.arg for a in [*fn.node.args.posonlyargs, *fn.node.args.args, *fn.node.args.kwonlyargs] if a.arg != "self"]
        left = set(into)
        out: dict[str, list[str]] = {}
        for p in params:
            value = self._entry_value(fn, run, p)
            if value is None:
                continue
            ways = SR.param_ways(owner, p)
            if any(not w.route or w.route[-1] not in into for w in ways):
                return None
            for call in sorted(into):
                by = [w for w in ways if w.route[-1] == call]
                if by and all(w.end == SR.CONSTANT and w.why not in SR.NO_LITERAL and not SR.same_value(w.value, value)
                              for w in by):
                    left.discard(call)
                    out.setdefault(call, []).append(f"{p} {'/'.join(sorted({w.value for w in by}))}, printed {value!r}")
        if not out or len(left) != 1:
            return None
        call = next(iter(left))
        self.say(f"  entry: its run's prints at its entry rule out the call(s) into {fn.shown} at "
                 + "; ".join(f"{c} ({', '.join(why)})" for c, why in sorted(out.items())) + f": {call} is left")
        return SR.End("entry", SR.CALLER, "by-values", call)

    def _entered_by(self, owner: str, line: int, run: Any, window: tuple[str, str] | None, rec: Any) -> SR.End | None:
        """F1 (out/gm/s3-f1-design.md): the call at one of the map's ``EnteredFrom`` sites its caller's run shows
        entered *owner*'s function, where no printed argument chose it.  A run whose mark holds a row's identity --
        ``killUnusedEventServiceConsumers < PandaID=7398538282 >``, the job archiveJob archives, not the key's -- is
        found in the run of a caller whose mark holds the same row, holding the arm's run whole (U3's parts); where
        that caller's run holds several of the calls, H2 judges each in it.  A mark holding no identity asks each
        caller's logs by the key (:meth:`_call_shown`).  None where not exactly one call is shown."""
        sites = [f"{s.owner.partition('::')[0]}:{s.line}" for s in rec.sites]
        ident = mark_identities(owner, line, run)
        if ident:
            self.say(f"  entry: its run {run.token!r} marks {ident}")
            tied = self._tie_calls(owner, ident, sites)
            to_hand = self.logs.names(self._tied_files(tied)) if tied else []
            if not to_hand:
                return None
            holding, _found, missed = self._callers_holding(tied, run, to_hand)
            self._keep_unanswered(missed)
            if not holding:
                return None
            cowner, _, crun = max(holding, key=lambda h: h[2].first)
            lines = sorted({ln for c, ln, r in holding if c == cowner and r is crun})
            if len(lines) > 1:
                lines = self._calls_judged(cowner, lines, crun, window)
            shown = [f"{cowner.partition('::')[0]}:{ln}" for ln in lines]
        else:
            shown = [s for s in sites if self._call_shown(s, window, getattr(run, "machine", None), "for-a-call-that-may-have-entered-it")]
        if len(shown) != 1:
            self.say(f"  entry: {len(shown)} of the call(s) at {', '.join(sites)} shown by a caller's run")
            return None
        return SR.End("entry", SR.CALLER, "entered-by", shown[0])

    def _calls_judged(self, cowner: str, lines: list[int], crun: Any, window: tuple[str, str] | None) -> list[int]:
        """Of the calls at *lines* of *cowner*, those its run *crun* shows ran: H2's judgement of each as a candidate in
        that run (positive)."""
        crel, cname = cowner.split("::", 1)
        j = judge([Candidate(crel, cname, [ln], run=crun) for ln in lines], {}, mode=self.mode, decide=self.decide, window=window)
        for v in j.verdicts:
            self.say(f"    {v.describe()}")
        return sorted({v.line for v in j.verdicts if v.kind == "positive"})

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

    def _guard_at(self, w: Wrote, site_fn: R.Function) -> R.Function:
        """The function whose passes the run holds: the chosen arm's; for a writer that prints nothing, read on its
        caller's run (``Wrote.via``), the caller at its call."""
        if w.via is None:
            return site_fn
        return R.function(w.via.candidate.rel, w.via.candidate.func, w.via.candidate.lines)

    def _which_wrote(self, subject: str, value: str, key: str, state: dict | None = None, reach: VR.Reach | None = None,
                     before: str = "") -> Wrote | OneOf:
        """H1 and H2: the map's writers of ``subject=value``, and which of them the run of the key at the statuslog
        row *state* shows ran -- or, where that decides nothing, the arms left (:class:`OneOf`).

        *state* is the statuslog row of *value* unless given: ``--late`` asks it at the row after the observed one.
        Every candidate's logs are asked, at the row's time, on the machine that wrote the row.  A *subject* the
        statuslog's rows hold beside the status (``JobSpec.computingSite``) is asked over the window they bound, on
        every machine (``W.value_held``).  *reach* is where the rows of the statements selecting the value *before*
        go (late, M2): a writer it leaves out did not move the row on from it, and is no candidate -- unless it leaves
        out every one, which it then says.
        """
        # H1
        self.start("H1", "candidates")
        readings = self._writers(subject, value, key)
        self.say(f"{len(readings)} reading(s) from the map for {subject}={value}")
        if reach is not None and (out := [r for r in readings if reach.leaves_out(r.owner)]):
            names = [f"{Path(r.file).name}::{r.owner.split('::')[-1]}" for r in out]
            if len(out) == len(readings):
                self.say(f"no read of {before!r}'s rows ({', '.join(reach.reads)}) reaches any of them: all kept (M2)")
            else:
                readings = [r for r in readings if r not in out]
                self.say(f"{len(out)} left out, no read of {before!r}'s rows ({', '.join(reach.reads)}) reaching them, "
                         f"only reads of other values, and no API endpoint (M2): {', '.join(names)}")
                self.result["left_out_by_reach"] = {"before": before, "reads": list(reach.reads), "owners": names}
        readings = self._handed_other(subject, value, readings)
        attr = subject.split(".")[-1]

        # H2
        self.start("H2", "which ran")
        holding = PT.history_holding(subject) if state is None else None
        if holding is not None:
            # a value the history's rows hold beside the status: they say by when it was there, not which run wrote it
            asking = W.value_held(self.data["statuslog"]["states"], attr, value, self._created(holding))
            found = self.need(T.Said("a-statuslog-time-for", {"value": value, "asking_why": asking.why}), asking.window)
            if isinstance(found, str):  # a time from --hand
                asking = W.Asking((found, found), None, T.Said("the-time-handed-for", {"value": value}))
            tries = [(state, asking, None)]
        elif state is None and PT.history(subject) is None:
            # an attribute no history records or holds: its writers' arms say when it took the value (K37)
            tries = self._attribute_tries(subject, attr, value, readings)
        else:
            if state is None:
                states = self.data["statuslog"]["states"]
                state = self.need(T.Said("a-statuslog-time-for-2", {"value": value}), next((s for s in states if s["status"] == value), None))
                if isinstance(state, str):  # a time from --hand
                    state = {"status": value, "time": state}
            tries = [(state, W.at_transition(state), None)]
        self._stand_ins: dict[tuple[str, int], list[StandIn]] = {}
        cands = self._h2_candidates(subject, readings) if readings else []
        names = self._writer_logs(subject, readings, cands)
        if len(tries) > 1 or any(_f3(state) for state, _, _ in tries):
            # every row's asks in one round, for the recording step to answer together, each of the logs of the
            # writers that can write it, for the candidates' marks alone, each whole (KNOWN_ISSUES.md K13); the rows
            # are then read newest first, and the first whose pass a writer is shown to write in ends the scan.  A
            # birth's row (no asking) asks nothing
            unanswered = []
            for _, asking, own in tries:
                if asking is None:
                    continue
                mine = cands if own is None else self._h2_candidates(subject, own)
                logs = names if own is None else self._writer_logs(subject, own, mine)
                unanswered += self._ask(key, logs, asking, owners_of([c.owner for c in mine]), whole=True)[1]
            self._stop_unanswered(unanswered)
            return self._scan(subject, attr, value, key, readings, tries)
        state, asking, own = tries[0]
        # H-5 (P43, bundle H, tmaeno's OK 2026-11-17 17:1x): a value held's window ends at the statuslog row that first
        # shows it; where no run of the key is there, that row is read as the scan reads one, its writers asked by
        # another key of the row their marks hold (K67's ``_recorded``): NR-J cloud's JobGenerator, by the task's
        closing = (next((s for s in self.data["statuslog"]["states"] if s["time"] == asking.window[1]), None)
                   if holding is not None and state is None and asking.window is not None
                   and self.hop not in self.hand else None)
        # a statuslog row the judge chooses no writer at is read as the scan reads a row (H2-1); a window with no row (a
        # value held, a time) has none to read so, and a time handed is the hand's
        wrote = self._wrote_at(subject, attr, value, key, readings if own is None else own, state, asking,
                               state is not None and self.hop not in self.hand, no_run=closing is not None)
        if wrote is None and closing is not None:
            # with no row of its own (*row* False), _wrote_at gives None only where no run of the key is there
            self.say(f"no run of {key} in the window: the statuslog's {closing['status']!r} at {closing['time']}, which "
                     "ends it, is read as one row of the scan, each writer asked as there (H-5)")
            return self._scan(subject, attr, value, key, readings, [(closing, asking, own)])
        if wrote is None:
            self.say("the judge chose no writer at the row: it is read as one row of the scan, each arm that wrote or "
                     "may have walking on (KNOWN_ISSUES.md K38)")
            return self._scan(subject, attr, value, key, readings, tries)
        return self._stood(subject, value, wrote, state)

    def _stood(self, subject: str, value: str, wrote: Wrote, state: dict | None) -> Wrote | OneOf:
        """A single window's pick, where it stands in for writers that print nothing (:class:`StandIn`): the arm shown
        to write, read on the pick's run (:meth:`_answer`), or the arms left (K38)."""
        if (wrote.chosen.candidate.owner, wrote.chosen.line) not in self._stand_ins:
            return wrote
        left = self._possibles(wrote.chosen, state, wrote.env, wrote)
        if (answer := self._answer(left)) is not None:
            return answer
        return OneOf(left, 1)

    def _h2_candidates(self, subject: str, readings: list) -> list[Candidate]:
        """H2's candidates, one per function: each reading's, where its runs are; and for a writer that prints nothing,
        whose arm a history pair places (``attr_time.arms``), the pair's caller at the call on the way to it, read on
        the caller's run (:class:`StandIn`, kept in ``_stand_ins`` by the caller's owner and line).  Only where the map
        places the prints (``rows.own_log_files``): else the readings' logs stand, as before."""
        silent = [r for r in readings if prints_nothing(r.owner, r.lines)] if R.ROWS_FROM == "graph" else []
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
                guards = tuple(pathcond.arm_condition(stmt)) if stmt is not None else ()
                for caller, point, level in sorted({(p.caller, p.point, p.level) for p in a.placed}):
                    rel, _, qual = caller.partition("::")
                    name = qual.rpartition(".")[2]
                    add(rel, name, point, self.files_of(f"{rel}::{name}", list(r.log_files), [point]))
                    stand = self._stand_ins.setdefault((f"{rel}::{name}", point), [])
                    if (s := StandIn(f"{r.owner}:{line}", level, guards)) not in stand:
                        stand.append(s)
        return list(found.values())

    def _guard_dropped(self, verdicts: list) -> frozenset[str]:
        """The arms *verdicts* say did not write by a guard (``_GUARD_REASONS``): a candidate's own, or, where it stands in
        for writers that print nothing (``_stand_ins``), theirs -- its call did not run them."""
        out: set[str] = set()
        for v in verdicts:
            if v.kind == "excluded" and any(r.startswith(_GUARD_REASONS) for r in v.reasons):
                stands = self._stand_ins.get((v.candidate.owner, v.line))
                out |= {_arm_name(s.arm) for s in stands} if stands else {_arm_name(f"{v.candidate.rel}::{v.candidate.func}:{v.line}")}
        return frozenset(out)

    def _possibles(self, v, state: dict | None, env: E.Env, on: Wrote | None = None) -> list[Possible]:
        """The arms verdict *v* says wrote, or may have, in the pass of the row *state*: its own line, or the arms of the
        writers that print nothing it stands in for, less those a guard read false shows did not.  *on* is *v* as H2
        read it, on its run (None where no run of it is at hand): each arm carries it as the one a fork walks on, and
        one shown to write as its answer; a stand-in's with the writer's arm chosen, read at the caller's call
        (``Wrote.via``).  Its run not showing the line ran is not in the data, unless its logs were not searched whole."""
        kept = [] if v.kind == "positive" else [Premise(T.Said("its-run-does-not-show", {"reason": _said(v.reasons[-1]) if v.reasons else v.kind}),
                                                        NOT_RECORDED if on is not None and on.unsearched else NOT_IN_DATA)]
        answer = on if on is not None and on.question is not None and not kept else None
        stands = self._stand_ins.get((v.candidate.owner, v.line))
        if not stands:
            return [Possible(f"{Path(v.candidate.rel).name}::{v.candidate.func}:{v.line}", state, not kept, kept, answer, branch=on)]
        out = []
        for s in stands:
            left = s.left(env)
            if left is None:
                self._dropped_by_guard = self._dropped_by_guard | {_arm_name(s.arm)}
                continue
            premises = kept + left
            owner, _, line = s.arm.rpartition(":")
            rel, _, func = owner.partition("::")
            arm = Verdict(Candidate(rel, func, [int(line)]), int(line), v.kind, list(v.reasons))
            stood = replace(on, chosen=arm, via=v) if on is not None else None
            out.append(Possible(f"{Path(rel).name}::{func}:{line}", state, not premises, premises,
                                stood if answer is not None and not premises else None,
                                f"{Path(v.candidate.rel).name}::{v.candidate.func}:{v.line}", stood))
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

    def _scan(self, subject: str, attr: str, value: str, key: str, readings: list, tries: list) -> Wrote | OneOf:
        """H2 over the statuslog rows the writers' passes can write, newest first (attr-time-shape.md 5): a row whose pass
        a writer is shown to write in ends the scan; one shown to have no writer's pass goes; one that may have stays.
        The answer is a writer that prints shown to write, with nothing newer that may have; else the arms left
        (:class:`OneOf`).  A row is read on the writers that can write it, where the try names them.  F3's life
        is read run by run (:meth:`_runs_as_rows`); a birth's row asks the run that made the row (:meth:`_birth_said`):
        its arms may have written there, as the row was made (KNOWN_ISSUES.md K37).  A row's arms that write as the row
        is made are read at the birth (:meth:`_births_apart`)."""
        left: list[Possible] = []
        tries = self._births_apart(subject, readings, tries)
        named = self._named(subject, key, readings, tries)
        tries = self._runs_as_rows(subject, key, tries)
        if not self._scan_tries(subject, attr, value, key, readings, tries, named, left) and (
                life := self._placed_life(subject, key, readings)) is not None:
            # no row shows a pass of a writer: the life of the ones on history rows that mark their runs by the key (Q2)
            cands = self._h2_candidates(subject, life[2])
            self._stop_unanswered(self._ask(key, list(dict.fromkeys(f for c in cands for f in c.log_files)), life[1],
                                            owners_of([c.owner for c in cands]), whole=True)[1])
            more = self._runs_as_rows(subject, key, [life])
            self._scan_tries(subject, attr, value, key, readings, more, named | self._named(subject, key, readings, [life]),
                             left)
            tries = tries + more
        if (answer := self._answer(left)) is not None:
            self.question = answer.question
            return answer
        return OneOf(left, len(tries))

    def _births_apart(self, subject: str, readings: list, tries: list) -> list:
        """P43 (the investigator's u10): on a subject no history records as its own -- a value its rows hold beside the
        status, or one none holds -- a row's arms on no history row that write it as the row is made
        (:func:`birth_arm`) are read at the row's birth, a try of its own after the rows, as
        :meth:`_attribute_tries` puts a birth's: the statuslog row shows the value, not their write.  JobGenerator's
        doGenerate:1250 sets cloud on the job TaskBuffer.storeJobs hands to insertNewJob's INSERT (NR-J cloud).
        *tries* as they were where the subject has a history, the row's making is not known, or a birth's try is
        there already."""
        if PT.history(subject) is not None or any(_f3(s) == "made" for s, _, _ in tries):
            return tries
        made = self._made_at(subject.partition(".")[0])
        if made is None:
            return tries
        fragment = mapdump.fragment()
        out, births = [], []
        for state, asking, own in tries:
            mine = readings if own is None else own
            arms = AT.arms(fragment, subject, mine, born=lambda o, b: birth_arm(o, b, subject)) if not _f3(state) else []
            moved = [a for a in arms if a.born and a.unplaced]
            if not moved:
                out.append((state, asking, own))
                continue
            births += [a for a in moved if a not in births]
            if rest := [a for a in arms if a not in moved]:
                out.append((state, asking, AT.readings_of(mine, rest)))
        if not births:
            return tries
        names = ", ".join(f"{a.owner.split('::')[-1]}:{a.line}" for a in births)
        self.say(f"P43: {names} write it as the row is made: read at the row's making, {made}")
        return out + [({"status": None, "time": made, "f3": "made"}, None, AT.readings_of(readings, births))]

    def _birth_said(self, subject: str, key: str, state: dict, own: list) -> Premise:
        """P43: a birth's try said with the run that made the row: the logs of the functions of the INSERTs the arms'
        writes are born by (:func:`birth_statements`) asked for the key over the row's making less and plus EDGE, on
        every machine, each whole; a run of theirs of the key there is that run (insertNewJob's ``inserted <PandaID>``
        under its ``<JediTaskID=... idPool=...>``).  An ask no record answers is kept for the asks file, with no stop;
        where none answers so, or the time is not known, the birth is said as before, no run read."""
        plain = Premise(T.Said("a-birth-no-run", {"time": state["time"]}), NOT_READ)
        inserts = sorted({s for r in own or [] for line in r.lines for s in birth_statements(r.owner, _ArmLine(line), subject)},
                         key=str)
        if not inserts or not state.get("time"):
            return plain
        at = W.when(state["time"])
        names = list(dict.fromkeys(o.rpartition("::")[2] for o, _ in inserts))
        lo, hi = (at - EDGE).isoformat(" "), (at + EDGE).isoformat(" ")
        ask = W.Asking((lo, hi), None, T.Said("for-the-birth", {"funcs": names, "time": state["time"]}))
        files = list(dict.fromkeys(f for o, line in inserts for f in self.files_of(o, [], [line])))
        logs, unanswered = self._logs_at(key, files, ask, owners_of([o for o, _ in inserts]))
        if unanswered:
            self._keep_unanswered(unanswered)
            return plain
        runs = {(r.first, r.machine): r for log in logs.values() for r in runs_for(log, key, self._other_of(key, log))
                if any(of_mover(r, o) for o, _ in inserts) and r.last[:19] >= lo[:19] and r.first[:19] <= hi[:19]}
        if not runs:
            return plain
        shown = [runs[k].said() for k in sorted(runs)[:3]]
        return Premise(T.Said("a-birth-its-run", {"time": state["time"], "funcs": names, "runs": shown,
                                                  "more": T.Said("more-runs", {"n": len(runs) - 3}) if len(runs) > 3 else ""}),
                       NOT_READ)

    def _placed_life(self, subject: str, key: str, readings: list) -> tuple[dict, W.Asking, list] | None:
        """Q2 (P38): where no statuslog row shows a pass of a writer of an attribute no history records, F3's life for
        the writers that are on history rows (``attr_time.arms``' placed) and mark their runs by the object's key
        (:func:`key_marks`): a pass of theirs may have written between the rows, writing none -- setScoutJobData_JEDI's
        1217 (mergeWalltime) in prepareTasksToBeFinished_JEDI's sqlEA 'reset in running' pass, which writes no status
        (NR-T-53639338-mergeWalltime).  Read run by run as F3's life is (:meth:`_runs_as_rows`), each run judged as a
        row is, where the printed bind map tells the value it wrote (``_printed_write``).  None where no writer is so,
        the subject has a history, or the life's ends are not known (said)."""
        if PT.history(subject) is not None:
            return None
        fragment = mapdump.fragment()
        cls = subject.partition(".")[0]
        # an arm on a history row holds its pairs, which hash not: kept as pairs, not as keys
        marks = [(a, m) for a in AT.arms(fragment, subject, readings) if a.placed and (m := key_marks(a.owner, a.line, cls, key))]
        if not marks:
            return None
        names = ", ".join(f"{a.owner.split('::')[-1]}:{a.line}" for a, _ in marks)
        life = W.lifetime(self._made_at(cls), self._read_at(subject), tuple(sorted({s for _, m in marks for s in m})))
        if life.window is None:
            self.say(f"Q2: {names} not asked over the object's life: {T.text(life.why)}")
            return None
        self.say(f"Q2: no row shows a pass of the writers; {names}, on history rows and marking their runs by the "
                 f"object's key, asked over {T.text(life.why)} (P38)")
        return {"status": None, "time": life.window[1], "f3": "span"}, life, AT.readings_of(readings, [a for a, _ in marks])

    def _outside_life(self, ran: datetime.datetime, subject: str) -> str:
        """Q3 (P42): why a write of *subject* run at *ran* is outside the object's life -- after the value was read
        (:meth:`_read_at`), or before its row was made (:meth:`_made_at`), by more than SLACK either way, the clocks
        being the database's and the machines' -- so it did not write the value; "" where it is not, or an end is not
        known.  A write of ``CURRENT_DATE-1`` of a task's modificationTime read the morning after it would have run at
        the value plus a day, after the read (NR-T-53655760-modificationTime)."""
        read = self._read_at(subject)
        if read and ran - W.SLACK > W.when(read):
            return f"after the value was read at {read}"
        made = self._made_at(subject.partition(".")[0])
        if made and ran + W.SLACK < W.when(made):
            return f"before the row was made at {made}"
        return ""

    def _scan_tries(self, subject: str, attr: str, value: str, key: str, readings: list, tries: list, named: set[str],
                    left: list[Possible]) -> bool:
        """:meth:`_scan`'s reading of *tries* in order, each one's arms added to *left*, until one shows a writer's pass
        wrote: whether one did."""
        for n, (state, asking, own) in enumerate(tries, 1):
            kind = _f3(state)
            if kind in ("made", "open"):
                why = (self._birth_said(subject, key, state, own) if kind == "made" else
                       Premise(T.Said("logs-not-searched-whole-over", {"why": asking.why}), NOT_RECORDED))
                self.say(f"-- row {n} of {len(tries)}: {_row_said(state)}")
                # walked on, if a fork does, on no run: H3 reads its guards in the rows' values at the row
                found = [Possible(f"{Path(r.owner.partition('::')[0]).name}::{r.owner.split('::')[-1]}:{line}", state, False, [why],
                                  branch=Wrote(own, None, [], None, Verdict(Candidate(r.file, r.owner.split("::")[-1], [line]), line,
                                                                           "unexamined", [why]),
                                               row=state, window=asking.window if asking is not None else None))
                         for r in own for line in r.lines]
                for p in found:
                    self.say(f"  may have written: {p.arm} ({why})")
            else:
                self.say(f"-- row {n} of {len(tries)}: {asking.why}")
                found = self._row_at(subject, attr, value, key, readings if own is None else own, state, asking, named)
            left += found
            if any(p.wrote for p in found):
                return True
        return False

    def _runs_as_rows(self, subject: str, key: str, tries: list) -> list:
        """*tries* with F3's life (``_f3`` "span": the logs of writers on no history row of the object that mark their
        runs by its key, asked for those marks) read run by run: each run the records hold in it whose mark holds the
        key becomes a try at its first record, on every machine, for the same marks, read as a row is (KNOWN_ISSUES.md
        K37), and all are put newest first, a birth's last among those of its time.  A run naming the key in its lines
        alone is another's -- getTasksToBeProcessed_JEDI's call listing the task under its own mark -- and is no pass
        of these writers.  A life with no run stays one try: no run there is no pass of those writers only in logs
        shown to name the key (:meth:`_row_at`).  One whose logs were not searched whole stays "open", before the
        rows: its writers may have written in what was not searched."""
        out = []
        pattern = keyed(key)
        for state, asking, own in tries:
            if _f3(state) != "span":
                out.append((state, asking, own))
                continue
            cands = self._h2_candidates(subject, own)
            runs, _, holes = self._ask(key, list(dict.fromkeys(f for c in cands for f in c.log_files)), asking,
                                       owners_of([c.owner for c in cands]), whole=True)
            lo, hi = asking.window
            found = sorted({(r.first[:19], r.last[:19]) for rs in runs.values() for r in rs
                            if pattern.search(r.token or "") and r.first[:19] <= hi and r.last[:19] >= lo}, reverse=True)
            self.say(f"F3: {len(found)} run(s) of {key} in {', '.join(dict.fromkeys(f for c in cands for f in c.log_files))} "
                     f"over {asking.why}" + (f"; not searched whole: {', '.join(holes)}" if holes else ""))
            if holes:
                out.append(({**state, "f3": "open"}, asking, own))
            elif not found:
                out.append((state, asking, own))
            out += [({"status": None, "time": first, "f3": "run"}, W.Asking((first, last), None, T.Said(
                "a-run-of-the-key-f3", {"key": key, "first": first, "last": last}), asking.strings), own) for first, last in found]
        return sorted(out, key=lambda t: t[0]["time"] if t[0] else "", reverse=True)

    def _named(self, subject: str, key: str, readings: list, tries: list) -> set[str]:
        """The writers' logs some run of *key* is in, in the records of any row of *tries*: a log shown to name its runs
        by the key, where no run at a row is no pass of its writers (:meth:`_row_at`).  JobGenerator's pass marks its
        lines <jediTaskID=... datasetID=...> and prints no PandaID: its log holds no run of a job it made (the harness's
        to-querier-62)."""
        out: set[str] = set()
        for _, asking, own in tries:
            if asking is None:  # a birth's row: nothing asked
                continue
            cands = self._h2_candidates(subject, readings if own is None else own)
            owners = owners_of([c.owner for c in cands])
            for name in self.logs.names(list(dict.fromkeys(f for c in cands for f in c.log_files))):
                try:
                    log = self.logs.log(name, key, self.window(asking), machine=asking.machine, owners=owners,
                                        strings=asking.strings, whole=True)
                except Unrecorded:
                    continue
                if runs_for(log, key, self._other_of(key, log)):
                    out.add(name)
        return out

    def _marked_by_key(self, c: Candidate, subject: str) -> bool:
        """G-4 (P33): whether every run the map records for the function of each of *c*'s arms is marked by a key
        holding the class's identity (``FunctionPrintsNode.runs``, as :func:`key_marks` reads one): it names its runs by
        the key whenever it runs.  TaskCommando's ``<jediTaskID=...>``; not a poller under ``label=user``."""
        ident = set(_identity_attrs(subject.partition(".")[0]))
        if not ident or not c.lines:
            return False
        for line in c.lines:
            found = R._prints_of(c.owner, [line])
            runs = list(getattr(found, "runs", None) or [])
            if not runs or not all(any(ident & set(k.columns or []) for k in run.key or []) for run in runs):
                return False
        return True

    def _unshown(self, readings: list, cands: list[Candidate], key: str, state: dict, env: E.Env,
                 window: tuple[str, str]) -> list[Possible]:
        """The arms of *cands*, writers none of whose logs is shown to name its runs by *key* (:meth:`_named`): no run
        of the key at the row is not their absence, so each may have written -- a stand-in's arm less one its condition
        read false rules out.  Each is walked on, if a fork does, on no run: its runs are in the data, not read."""
        out = []
        for c in cands:
            why = Premise(T.Said("no-line-holds-the-key", {"logs": list(c.log_files), "key": key}), NOT_READ)
            for line in c.lines:
                v = Verdict(c, line, "unexamined", [why])
                on = Wrote(readings, None, cands, env, v, row=state, window=window)
                stands = self._stand_ins.get((c.owner, line))
                if not stands:
                    out.append(Possible(f"{Path(c.rel).name}::{c.func}:{line}", state, False, [why], branch=on))
                for s in stands or []:
                    if (left := s.left(env)) is not None:
                        owner, _, at = s.arm.rpartition(":")
                        rel, _, func = owner.partition("::")
                        arm = Verdict(Candidate(rel, func, [int(at)]), int(at), "unexamined", [why])
                        out.append(Possible(f"{Path(s.arm.split('::')[0]).name}::{s.arm.split('::')[-1]}", state, False, [why, *left],
                                            branch=replace(on, chosen=arm, via=v)))
        return out

    def _recorded(self, subject: str, key: str, state: dict, asking: W.Asking, left: list[Possible]) -> list[Possible]:
        """*left*, the arms :meth:`_unshown` says may have written at the row *state* -- no line of their logs holds the
        key -- read on the runs that recorded the row (KNOWN_ISSUES.md K67).  For each call of the arm's pass that
        records the row after the arm (:func:`recorders`), its methods' logs are asked on the row's machine for the key
        near the row (:meth:`_recorder_read`).  Where none of those calls' methods ran the key there, the arm's pass did
        not write the row, and the arm goes from it; where they did, inside a call of a pass of the call's function
        at the row, that pass is said in place of the key not being named.  Else it stays as it was.  A stand-in, read
        at its caller's call on the way to it (``Wrote.via``), is read on that caller's recording calls only: another
        caller's pass is not its pass."""
        if not left:
            return left
        out = []
        for p in left:
            arm = p.branch.chosen if p.branch is not None else None
            via = cast("Verdict | None", p.branch.via) if p.branch is not None else None
            recs = recorders(subject, f"{arm.candidate.rel}::{arm.candidate.func}", arm.line) if arm is not None else []
            if via is not None:
                recs = [r for r in recs if _bare(r.caller) == via.candidate.owner and via.line in r.points]
            read = [self._recorder_read(r, subject, key, state, asking)[:2] for r in recs]
            if read and all(kind == "none" for kind, _ in read):
                at = f", read at {Path(via.candidate.rel).name}::{via.candidate.func}:{via.line}" if via is not None else ""
                self.say(f"  not its row: {p.arm}{at} -- "
                         + "; ".join(dict.fromkeys(T.text(said) for _, said in read if said is not None)))
                continue
            shown = next((said for kind, said in read if kind == "pass" and said is not None), None)
            if shown is not None:
                p = replace(p, premises=[Premise(shown, NOT_READ) if _premise_kind(x) == "no-line-holds-the-key" else x
                                         for x in p.premises])
            out.append(p)
        return out

    def _recorder_read(self, rec: Recorder, subject: str, key: str, state: dict,
                       asking: W.Asking) -> tuple[str, T.Said | None, Run | None, list[Run]]:
        """One call of :meth:`_recorded`: ``("none", why)`` where its methods' logs on the row's machine, one there at
        least and each read whole there, hold no run of theirs of the key within EDGE of the row -- each prints the key
        before it records the row, so no pass through the call wrote it; ``("pass", what)`` where every such run ran
        inside the call of a pass of the call's function at the row (``which_ran.recorded_in_pass``), its log asked on
        the machine for the marks the object's row fills (:meth:`_pass_marks`), with that pass; ``("", None, None)``
        where the records do not say.  Last, the methods' runs at the row that were read (each cut to its call there).
        An ask no record answers is kept for the asks file, with no stop.

        P43's redo (out/gm/qk-design.md, the section on P43's redo; the investigator's u11 NR-J cloud 1, gshare 1): a row
        no machine is known for (the row's creation to its first status, NR-J's cloud) has its methods' runs of the key
        near it asked of every machine first; the one machine they ran on is the host, read as above
        (:meth:`_recorder_anywhere`)."""
        if not asking.machine:
            return self._recorder_anywhere(rec, subject, key, state, asking)
        host = str(asking.machine)
        at = datetime.datetime.fromisoformat(state["time"][:19])
        lo, hi = (at - EDGE).isoformat(" "), (at + EDGE).isoformat(" ")
        methods = [(_bare(m), list(lines)) for m, lines in rec.methods]
        names = [m.rpartition("::")[2] for m, _ in methods]
        files = list(dict.fromkeys(f for m, lines in methods for f in self.files_of(m, [], lines)))
        # asked over the row less and plus EDGE, so that "read whole" (``Log.holes``) is said of all the runs read near it
        ask = W.Asking((lo, hi), asking.machine, T.Said("for-the-rows-recorder", {
            "methods": names, "status": state.get("status"), "time": state["time"], "machine": host}))
        logs, unanswered = self._logs_at(key, files, ask, owners_of([m for m, _ in methods]))
        caller = _bare(rec.caller)
        rel, _, func = caller.partition("::")
        pass_files = self.files_of(caller, [], [rec.call])

        def passes(values: dict[str, str]) -> list[tuple[str, W.Asking]]:
            return [(value, W.Asking(asking.window, asking.machine, T.Said("for-the-recorders-pass", {
                "func": func, "call": rec.call, "marks": list(marks)}), marks)) for value, marks in _pass_marks(caller, rec.call, values)]

        if unanswered:
            # the pass's asks go in the same round as the methods', for the values the rows file holds now: the
            # reading takes them from the methods' runs (``mark_columns``), at the row's time
            for value, fask in passes(self._row_values(subject)):
                unanswered += self._logs_at(value, pass_files, fask, owners_of([caller]))[1]
            self._keep_unanswered(unanswered)
            return "", None, None, []
        there = {name: log for name, log in logs.items() if host not in log.without}
        # the host's runs: a log the host is not listed for comes asked of every machine (K13), and another machine's
        # run is not the host's (a run of a file handed in names no machine)
        inner = [r for log in there.values() for r in runs_for(log, key, self._other_of(key, log))
                 if any(of_mover(r, m) for m, _ in methods) and r.last[:19] >= lo and r.first[:19] <= hi
                 and r.machine in (host, "")]
        # a method's run under its own mark holds every call of it (updateTask_JEDI's 21 records over 15 hours, for
        # MT-passed-late's 'passed'): one running on past the row's EDGE is cut at its entry rows, and the call at the
        # row is the run that recorded it; where no call is cut there, the run stays as it was
        cut: list = []
        for r in inner:
            if r.first[:19] >= lo and r.last[:19] <= hi:
                cut.append(r)
                continue
            parts = [p for m, lines in methods if of_mover(r, m)
                     for p in passes_of(R.function(m.partition("::")[0], m.partition("::")[2], lines), r, self.mode)[0]
                     if p is not r]
            cut += [p for p in parts if at_window(p, (state["time"], state["time"]))] or [r]
        inner = cut
        if not inner:
            columns = set(self._key_columns(subject, key))
            # a log so asked says nothing of a file of it on the host: not read whole there
            if (there and not any(log.holes or host in log.unlisted for log in there.values())
                    and all(prints_key_before(m, lines, columns) for m, lines in methods)):
                return "none", T.Said("no-recorder-run-at-the-row", {"methods": names, "key": key, "logs": sorted(there),
                                                                    "machine": host.split(".")[0]}), None, []
            return "", None, None, []
        fn = R.function(rel, func, [rec.call])
        sites = list(dict.fromkeys(s for m, lines in methods
                                   for s in self._called_at(fn, Candidate(m.partition("::")[0], m.rpartition("::")[2], lines))))
        # what the methods' runs' marks hold, by column, where they all agree
        held: dict[str, set[str]] = {}
        for r in inner:
            for m, lines in methods:
                if of_mover(r, m):
                    for col, v in mark_columns(m, lines, r).items():
                        held.setdefault(col, set()).add(v)
        for value, fask in passes({col: vs.pop() for col, vs in held.items() if len(vs) == 1}):
            found, unanswered = self._logs_at(value, pass_files, fask, owners_of([caller]))
            if unanswered:
                self._keep_unanswered(unanswered)
                continue
            got = recorded_in_pass(fn, [r for log in found.values() for r in runs_for(log, value)], (state["time"], state["time"]),
                                   inner, sites, self.mode)
            if got is not None:
                part, site = got
                return "pass", T.Said("its-pass-recorded-the-row", {"run": part.said(), "methods": names, "key": key, "site": site}), part, inner
        return "", None, None, inner

    def _recorder_anywhere(self, rec: Recorder, subject: str, key: str, state: dict,
                           asking: W.Asking) -> tuple[str, T.Said | None, Run | None, list[Run]]:
        """:meth:`_recorder_read` where no row says the machine: the recording methods' logs asked of every machine for
        the key over the row less and plus EDGE, each whole.  Runs of theirs on one machine alone make it the host, and
        the reading goes on there; none, in logs read whole, each method printing the key before it records the row,
        is ``"none"`` as on a host; else nothing is said.  JobGenerator's arms writing a new job's cloud, recorded by
        insertNewJob's "inserted <PandaID>" in storeJobs, on the machine that made the job."""
        at = datetime.datetime.fromisoformat(state["time"][:19])
        lo, hi = (at - EDGE).isoformat(" "), (at + EDGE).isoformat(" ")
        methods = [(_bare(m), list(lines)) for m, lines in rec.methods]
        names = [m.rpartition("::")[2] for m, _ in methods]
        files = list(dict.fromkeys(f for m, lines in methods for f in self.files_of(m, [], lines)))
        ask = W.Asking((lo, hi), None, T.Said("for-the-rows-recorder-anywhere", {
            "methods": names, "status": state.get("status"), "time": state["time"]}))
        logs, unanswered = self._logs_at(key, files, ask, owners_of([m for m, _ in methods]))
        if unanswered:
            self._keep_unanswered(unanswered)
            return "", None, None, []
        inner = [r for log in logs.values() for r in runs_for(log, key, self._other_of(key, log))
                 if any(of_mover(r, m) for m, _ in methods) and r.last[:19] >= lo and r.first[:19] <= hi]
        hosts = {r.machine for r in inner if r.machine}
        if len(hosts) == 1 and all(r.machine for r in inner):
            host = hosts.pop()
            self.say(f"  {', '.join(names)}: the runs of {key} near the row's {state['time']} are on "
                     f"{host.split('.')[0]} alone: the row's recorder is read there (P43)")
            return self._recorder_read(rec, subject, key, state, replace(asking, machine=host))
        if (not inner and logs and not any(log.holes for log in logs.values())
                and all(prints_key_before(m, lines, set(self._key_columns(subject, key))) for m, lines in methods)):
            return "none", T.Said("no-recorder-run-at-the-row", {"methods": names, "key": key, "logs": sorted(logs),
                                                                "machine": "any machine"}), None, []
        return "", None, None, inner

    def _row_values(self, subject: str) -> dict[str, str]:
        """The rows file's values of *subject*'s object, by column (``JobSpec.jediTaskID``), as read, at any time: what a
        pass's marks are first asked for (:meth:`_recorder_read`), not what a step reads."""
        cls = subject.partition(".")[0]
        entity = next((e for e, c in (self.data.get("classes") or {}).items() if c == cls and e != "about"), None)
        row = (self.data.get("rows") or {}).get(entity) or {} if entity else {}
        return {f"{cls}.{attr}": str(v["value"]) for attr, v in row.items()
                if isinstance(v, dict) and isinstance(v.get("value"), (str, int)) and v.get("value") != ""}

    def _logs_at(self, key: str, names: list[str], asking: W.Asking, owners: Owners | None) -> tuple[dict[str, Log], list]:
        """``(logs, unanswered)``: *names* asked over *asking*, each whole, as :meth:`_ask` does, as the logs (what a
        listing says the machine holds no file of is in ``Log.without``)."""
        window = self.window(asking)
        out: dict[str, Log] = {}
        unanswered: list = []
        for name in self.logs.names(names):
            try:
                out[name] = self.logs.log(name, key, window, machine=asking.machine, owners=owners, strings=asking.strings, whole=True)
            except Unrecorded as e:
                unanswered += [(ask, why or T.text(asking.why)) for ask, why in e.asks]
        return out, unanswered

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
        self._mark_row_after(cands, subject, readings, state)
        # P62: a row at the value's own time holds its write; a line writing its clock plus an offset did not write it
        if (written := _time_value(value)) is not None and abs(W.when(hi) - written) <= W.SLACK:
            cands = self._offset_out(cands, subject, written)
        runs, _, holes = self._ask(key, list(dict.fromkeys(f for c in cands for f in c.log_files)), asking,
                                   owners_of([c.owner for c in cands]), whole=True)
        at = {name: held for name, rs in runs.items() if (held := [r for r in rs if spans(r, (lo, hi))])}
        handed = self._batched(cands, at, runs, holes, state, hi, subject)
        at.update(self._ended_before(cands, at, handed, runs, (lo, hi)))
        env = self.base_env(at=state["time"])
        unnamed = [c for c in cands if not any(f in named or f in holes for f in c.log_files)]
        # G-4 (P33, bundle G, tmaeno's OK 2026-11-17 16:0x): a writer whose every arm's run is marked by the key names
        # its runs by it whenever it runs: its logs searched whole at the row with no run there is no pass of it
        marked = [c for c in unnamed if c.log_files and self._marked_by_key(c, subject)]
        for c in marked:
            self.say(f"  no pass of {c.owner.split('::')[-1]} at {when}: its marks hold the key, and {list(c.log_files)} "
                     "searched whole there show no run of it (G-4)")
        maybe = self._unshown(readings, [c for c in unnamed if c not in marked], key, state, env, (lo, hi))
        maybe = self._recorded(subject, key, state, asking, maybe)
        if not at and not holes and not handed:
            self.say(f"  no run of {key} at {when}: no writer's pass" + (" of a log that names it" if maybe else ""))
            for p in maybe:
                self.say(f"  may have written: {p.arm} ({'; '.join(p.premises)})")
            return maybe
        # a batched writer's run ended before the row: a run of its own log, judged as the others are
        at = {**at, **{r.log.name: [r] for r in handed.values()}}
        question = _question_run(at)
        if question is not None:
            self.say(f"the run: {question.describe()}")
        at_hand = [question] if question is not None else [r for rs in at.values() for r in rs]
        for c in cands:
            if any(name in c.log_files for name in at) and c.owner not in handed:
                self.warn_split(c.owner, c.lines)
        stable_holes([R.seen(R.function(c.rel, c.func, c.lines), run, self.mode)
                      for c in cands for run in at_hand if run.log.name in c.log_files], env)
        j = judge(
            cands, runs, mode=self.mode, decide=self.decide, window=(lo, hi), subject=attr, observed=value,
            question_run=question, env=env, called_at=self._called_at, keeps=self._keeps(subject),
        )
        self.say(j.report(question))
        self._dropped_by_guard = self._dropped_by_guard | self._guard_dropped(j.verdicts)
        out = []
        for v in j.verdicts:
            if v.kind == "excluded" or (v.kind == "unexamined" and not any(f in holes for f in v.candidate.log_files)):
                continue
            run = question if question is not None and question.log.name in v.candidate.log_files else j.runs.get(v.candidate.owner)
            on = Wrote(readings, run, cands, env, v, row=state, window=(lo, hi),
                       unsearched={f: holes[f] for f in v.candidate.log_files if f in holes})
            if v.kind == "unexamined":
                v.reasons.append(f"not searched whole: {', '.join(f for f in v.candidate.log_files if f in holes)}")
            out += self._possibles(v, state, env, on)
        out += maybe
        for p in out:
            self.say(f"  {'wrote' if p.wrote else 'may have written'}: {p.arm}" + (f" ({'; '.join(p.premises)})" if p.premises else ""))
        return out

    def _possibilities(self, left: list[Possible]) -> dict[str, list[Possible]]:
        """The arms that wrote last or may have, each with its rows, newest first (result.json's ``possibilities``)."""
        arms: dict[str, list[Possible]] = {}
        for p in left:
            arms.setdefault(p.arm, []).append(p)
        self.result["possibilities"] = [
            {"arm": arm, "rows": [{"status": p.row["status"] if p.row else None, "time": p.row["time"] if p.row else None,
                                   **({"f3": _f3(p.row)} if _f3(p.row) else {}), "wrote": p.wrote, "premises": [_premise_json(x) for x in p.premises]}
                                  for p in ps]}
            for arm, ps in arms.items()
        ]
        return arms

    def _one_of(self, subject: str, value: str, one_of: OneOf) -> Stop:
        """H2's stop naming the arms that wrote *value* last or may have, each with its rows, where the steps after H2
        are not walked for each -- L1's (KNOWN_ISSUES.md K38) -- or none is left."""
        arms = self._possibilities(one_of.left)
        if not arms:
            return Stop(T.at(self.hop, T.Said("a-pass-that-wrote-at-the-statuslog", {"subject": subject, "value": value, "of_rows": one_of.rows})), "one-of")
        said = [T.Said("arm-label", {"arm": arm, "rows": _at_rows(ps)}) for arm, ps in arms.items()]
        return Stop(T.at(self.hop, T.Said("one-arm-that-wrote-last-of-left", {"subject": subject, "value": value, "n_arms": len(arms), "said": said})), "one-of")

    def _the_arm_left(self, subject: str, value: str, one_of: OneOf) -> Wrote:
        """A late question's mover where H2 leaves one arm that may have written the next value (:class:`OneOf`,
        KNOWN_ISSUES.md K38): that arm, on the run H2 read it on, with what it rests on -- its conditions not read, its
        run not showing it ran -- named as what every END after rests on (``rests``).  More than one arm left, or none,
        stops as before: which of them moved it on is not walked for each."""
        arms = self._possibilities(one_of.left)
        branch = next(iter(arms.values()))[0].branch if len(arms) == 1 else None
        if branch is None:
            raise self._one_of(subject, value, one_of)
        (arm, ps), = arms.items()
        said = T.Said("the-one-arm-left-moved-it-on", {"arm": arm, "premises": [_said(p) for p in ps[0].premises]})
        self.say(f"{said}: the late question goes on with it")
        self.rests = (*self.rests, said)
        return branch

    def _writer_logs(self, subject: str, readings: list, cands: list | None = None) -> list[str]:
        """The logs H2 looks for a run of *subject*'s writers in (*cands*, ``_candidates`` of *readings* where not
        given).  A writer that prints nothing of its own, whose reading names no log and whose arm no history pair
        places, has no run a log holds (KNOWN_ISSUES.md K25): where no writer has a log, H2 stops on that -- not on the
        logs holding no run of the key, which none was asked for; and with no writer at all, on that.  A time handed
        for H2 gives no log."""
        if not readings:
            raise Stop(T.at(self.hop, T.Said("a-writer-of-in-the-map", {"subject": subject})))
        cands = self._candidates(readings) if cands is None else cands
        names = list(dict.fromkeys(f for c in cands for f in c.log_files))
        if not names:
            funcs = ", ".join(dict.fromkeys(r.owner.split("::")[-1] for r in readings))
            raise Stop(T.at(self.hop, T.Said("a-log-to-look-for-a-run", {"subject": subject, "funcs": funcs})))
        return names

    def _attribute_tries(self, subject: str, attr: str, value: str, readings: list) -> list[tuple[dict | None, W.Asking | None, list | None]]:
        """H2's windows for an attribute no history records (``attr_time``, KNOWN_ISSUES.md K37): the value itself where
        every writer writes the time it runs, else the statuslog rows the writers' passes can write, newest first, each
        with the readings of the writers that can write it (``attr_time.row_writers``), whose logs alone it is asked of.
        A writer on no history row of the object is read where the map says more of it: a birth (``born``) wrote as the
        row was made, a try at the row's creation that asks nothing, after the rows; one that marks its runs by the
        object's key (``key_marks``) is asked over the object's life, from its creation to when the value was read
        (``windows.lifetime``, F3), a try before the rows that :meth:`_scan` reads run by run -- or, where its write is
        the time it runs (``attr_time.NOW``), around the value alone (``windows.near_value``).  Any other stops."""
        fragment = mapdump.fragment()
        # P43: a value set on an object a birth then persists is a birth too (JobGenerator's gshare at 1301)
        found = AT.arms(fragment, subject, readings, born=lambda owner, b: birth_arm(owner, b, subject))
        self.need(T.Said("a-writer-of-in-the-map-to", {"subject": subject, "value": value}), found)
        if AT.now_valued(found):
            asking = W.value_now(attr, value)
            got = self.need(T.Said("a-time-for", {"value": value, "asking_why": asking.why}), asking.window)
            if isinstance(got, str):  # a time from --hand
                asking = W.Asking((got, got), None, T.Said("the-time-handed-for", {"value": value}))
            self.say(f"{len(found)} writer line(s), each writing the time it runs: {asking.why}")
            return [(None, asking, None)]
        cls = subject.partition(".")[0]
        births = [a for a in found if a.unplaced and a.born]
        marks = {a: key_marks(a.owner, a.line, cls, self.args.key) for a in found if a.unplaced and not a.born}
        # one whose runs no mark holds the key of, but which prints the key after its write on every way, is asked by
        # that print as a marked one is by its mark (Q5)
        printing = set()
        for a, m in marks.items():
            if not m and (got := after_marks(a, cls, self.args.key)):
                marks[a] = got
                printing.add(id(a))
        marked = [a for a, m in marks.items() if m]
        for a in found:
            self.say(f"  {a.owner.split('::')[-1]}:{a.line} " + (f"is on no history row of the object: {a.unplaced}" if a.unplaced else
                     "writes in a pass that writes the object's history row (" + ", ".join(sorted({f"{p.caller.split('::')[-1]}:{p.call} "
                     f"{p.callee.split('.')[-1]} {p.order}" for p in a.placed})) + ")") +
                     ("; a birth: it wrote as the row was made" if a in births else
                      "; it prints the object's key after its write on every way" if id(a) in printing else
                      "; its runs are marked by the object's key" if a in marked else ""))
        unplaced = [a for a in found if a.unplaced and a not in births and a not in marked]
        # Q3 (P42): one whose write is its clock plus a settled offset ran at the value less the offset, if it wrote it:
        # none where that is outside the object's life (before its row was made, after the value was read, by more than
        # SLACK either way: the clocks are the database's and the machines'), or the selection its key comes from
        # returns no row of the object then; else asked around then
        clocked: list[tuple[AT.Arm, W.Asking]] = []
        for a in list(unplaced):
            off = _written_offset(a)
            around = W.near_ran(attr, value, off) if off is not None else None
            if off is None or around is None:
                continue
            unplaced.remove(a)
            ran = W.when(value) - off
            if outside := self._outside_life(ran, subject):
                self.say(f"  {a.owner.split('::')[-1]}:{a.line}: its write is its clock plus {int(off.total_seconds())} s, so it "
                         f"ran at {ran} if it wrote {value!r}, {outside}: it did not write it (P42)")
                continue
            if (why := self._selection_false(a.owner, a.line, subject, ran)) is not None:
                self.say(f"  {a.owner.split('::')[-1]}:{a.line}: its write is its clock plus {int(off.total_seconds())} s, so it "
                         f"ran at {ran} if it wrote {value!r}, when the selection its key comes from returns no row of the "
                         f"object: {why} (P42)")
                continue
            clocked.append((a, around))
        self.need(T.Said("every-writer-of-on-a-history-row", {"subject": subject, "value": value, "n_unplaced": len(unplaced), "n_found": len(found)}), not unplaced)
        states = self.data["statuslog"]["states"]
        rows = AT.row_writers(found, states, AT.recorded_values(fragment))
        self.say(f"the writers' passes can write {len(rows)} of the {len(states)} statuslog row(s): asked newest first, "
                 "each of the logs of the writers that can write it")
        if len(found) > 1:
            for a in found:
                self.say(f"  {a.owner.split('::')[-1]}:{a.line}: {sum(a in arms for _, arms in rows)} row(s)")
        tries: list[tuple[dict | None, W.Asking | None, list | None]] = [
            (s, W.at_transition(s), AT.readings_of(readings, arms)) for s, arms in rows]
        made = self._made_at(cls) if births or marked else None
        # one whose write is the time it runs ran at the value, if it wrote it: asked around the value, not over the life
        # (windows.near_value; tmaeno's OK 2026-11-15 14:44:05)
        near = {a: w for a in marked if a.outcome in AT.NOW
                if (w := W.near_value(attr, value, tuple(sorted(marks[a])))) is not None}
        spanned = [a for a in marked if a not in near]
        if spanned:
            life = W.lifetime(made, self._read_at(subject), tuple(sorted({s for a in spanned for s in marks[a]})))
            names = ", ".join(f"{a.owner.split('::')[-1]}:{a.line}" for a in spanned)
            self.need(T.Said("the-object-s-life-to-ask-over", {"names": names, "life_why": life.why}), life.window)
            self.say(f"F3: {names} asked over {life.why}")
            tries.insert(0, ({"status": None, "time": life.window[1], "f3": "span"}, life, AT.readings_of(readings, spanned)))
        for a, asking in near.items():
            self.say(f"F3: {a.owner.split('::')[-1]}:{a.line} writes the time it runs: asked over {asking.why}")
            end = asking.window[1] if asking.window else value
            tries.insert(0, ({"status": None, "time": end, "f3": "span"}, asking, AT.readings_of(readings, [a])))
        for a, asking in clocked:
            self.say(f"Q3: {a.owner.split('::')[-1]}:{a.line} asked over {asking.why} (P42)")
            tries.insert(0, ({"status": None, "time": asking.window[1] if asking.window else value, "f3": "span"}, asking,
                             AT.readings_of(readings, [a])))
        if births:
            names = ", ".join(f"{a.owner.split('::')[-1]}:{a.line}" for a in births)
            self.need(T.Said("when-the-row-was-made-for-the", {"names": names}), made)
            tries.append(({"status": None, "time": made, "f3": "made"}, None, AT.readings_of(readings, births)))
        got = self.need(T.Said("a-statuslog-row-the-writers-of-can", {"subject": subject}), tries)
        if isinstance(got, str):  # a time from --hand
            return [({"status": value, "time": got}, W.Asking((got, got), None, T.Said("the-time-handed-for", {"value": value})), None)]
        return tries

    def _wrote_at(self, subject: str, attr: str, value: str, key: str, readings: list, state: dict | None,
                  asking: W.Asking, row: bool, no_run: bool = False) -> Wrote | None:
        """H2 at one window: the run of the key there and which candidate ran.  None where the judge chooses none at
        a statuslog row (*row*), before its report: the caller reads the row as :meth:`_scan` reads one, whose judge
        is this one -- the same candidates, runs and window -- and reports it (H2-1, KNOWN_ISSUES.md K38).
        MF-7396944927's 'failed' at 14:46:00 left process_job_report:416 and 441 and archiveJob:1813, each of which
        may have written it last in the adder's pass, and stopped here.  None too, with *no_run*, where no run of the
        key is in the window: a value held's, whose closing row the caller reads so (H-5)."""
        readings, said = self._unborn(readings, subject, state)
        if said:
            self.say(said)
        lo, hi = asking.window
        when = hi if lo == hi else f"{lo}..{hi}"
        cands = self._h2_candidates(subject, readings)
        self._mark_row_after(cands, subject, readings, state)
        if asking.holds_write:
            cands = self._selected_out(cands, subject, W.when(hi))
        runs, unanswered, holes = self._ask(key, list(dict.fromkeys(f for c in cands for f in c.log_files)), asking,
                                            owners_of([c.owner for c in cands]), whole=True)
        self._stop_unanswered(unanswered)
        at = {name: held for name, rs in runs.items() if (held := [r for r in rs if spans(r, (lo, hi))])}
        handed = self._batched(cands, at, runs, holes, state, hi, subject)
        at.update(self._ended_before(cands, at, handed, runs, (lo, hi)))
        if no_run and not at and not handed and not holes:
            return None
        self.need(T.Said("a-run-of-at", {"key": key, "when": when}), at or handed)
        # a batched writer's run ended before the row: a run of its own log, judged as the others are
        at = {**at, **{r.log.name: [r] for r in handed.values()}}
        question = _question_run(at)
        if question is not None:
            self.say(f"the run: {question.describe()}")
        elif len(at) > 1:
            self.say(f"runs of {key} at {when} in {len(at)} logs ({', '.join(at)}): each candidate is judged on its own run")
        else:
            self.say(f"{len(next(iter(at.values())))} runs of {key} at {when} in {next(iter(at))}: each candidate is judged on its own run")
        at_hand = [question] if question is not None else [r for rs in at.values() for r in rs]
        for c in cands:
            if any(name in c.log_files for name in at) and c.owner not in handed:
                self.warn_split(c.owner, c.lines)
        env = self.base_env(at=state["time"] if state else None)
        seens = [R.seen(R.function(c.rel, c.func, c.lines), run, self.mode)
                 for c in cands for run in at_hand if run.log.name in c.log_files]
        stable_holes(seens, env)
        j = judge(
            cands, runs, mode=self.mode, decide=self.decide, window=(lo, hi),
            subject=attr, observed=value, question_run=question, env=env, called_at=self._called_at,
            holds_write=asking.holds_write, keeps=self._keeps(subject),
        )
        if j.chosen is None and row:
            return None
        self.say(j.report(question))
        self._dropped_by_guard = self._dropped_by_guard | self._guard_dropped(j.verdicts)
        chosen = self.pick(T.Said("which-candidate-ran"), j)
        if j.pass_writes and chosen is j.chosen and chosen is not None:
            # the pass changed it, its write shown; which of its writes wrote it last is what the steps after rest on
            # (P34, bundle B's B-6)
            changed = T.Said("the-pass-changed-it", {"arm": f"{Path(chosen.candidate.rel).name}::{chosen.candidate.func}:{chosen.line}",
                                                     "others": j.pass_writes})
            self.say(f"{changed}")
            self.rests = (*self.rests, changed)
        if question is None and chosen.kind != "positive" and j.runs.get(chosen.candidate.owner) is None:
            # left by elimination with no pass of it at the window (KNOWN_ISSUES.md K47): no run is the question's
            self.say(f"no run: no pass of {chosen.candidate.func} at {when} shows it ran")
        elif question is None:
            question = self.need(T.Said("the-run-of-at", {"candidate_func": chosen.candidate.func, "when": when}), j.runs.get(chosen.candidate.owner))
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

    def _handed_other(self, subject: str, value: str, readings: list) -> list:
        """M2 (P40(a), out/gm/qk-design.md's H2-2 bundle; the map's ``ParameterFrom``, its answer 2026-11-17): *readings*
        less each line whose value is a parameter every call into its function hands as a literal other than *value*
        (:meth:`_calls_handing`); a reading left with no line goes.  A line one of whose calls hands a value not so read
        stays, the calls ruled out said.  entity_module's reassignShare:477 writes ``JobSpec.gshare = runtime(gshare)``:
        AtlasAnalWatchDog:717 hands 'Express Analysis' and AtlasProdWatchDog:196 'Express'; TaskCommando:345 and
        task_api:717 values made at run time, which stay (NR-J-7424203630-gshare)."""
        kept = []
        for r in readings:
            lines = []
            for line in r.lines:
                calls = self._calls_handing(subject, value, r.owner, line)
                if calls is None:
                    lines.append(line)
                    continue
                other = [f"{c} ({', '.join(o)})" for c, (o, k) in calls.items() if not k]
                left = [f"{c} ({', '.join(k)})" for c, (o, k) in calls.items() if k]
                where = f"  {r.owner.split('::')[-1]}:{line}, writing a parameter"
                if left:
                    lines.append(line)
                    if other:
                        self.say(f"{where}: its call(s) {'; '.join(other)} hand another value than {value!r}; left: "
                                 f"{', '.join(left)} (P40(a))")
                else:
                    self.say(f"{where}: every call into it hands another value than {value!r}: {'; '.join(other)}; "
                             f"no candidate (P40(a))")
            if lines == list(r.lines):
                kept.append(r)
            elif lines:
                kept.append(r.model_copy(update={"lines": lines}))
        return kept

    def _calls_handing(self, subject: str, value: str, owner: str, line: int) -> dict[str, tuple[list[str], list[str]]] | None:
        """The calls a value of *subject* written at *owner*'s *line* comes in by, where the map's branches there all
        write a parameter (``runtime(<name>)``, ``Branch.handed``), each with ``(others, kept)``: the parameter's ways
        (:func:`sources.param_ways`) by the call where they part -- their routes, read from where the value is made
        inward, less the calls every way goes by (``file:line``; reassignShare's TaskBuffer:2081) -- the literals other
        than *value* they end at, and why any other way of it is kept: it ends at *value* ("the value"), at a part of a
        value, at a literal only part of the value, or anywhere else (a value made at run time, the database, an API's
        parameter, a stop, by its kind).  None where the line writes no parameter or a way has no route."""
        branches = _branches_at(subject, owner, line)
        names = {str(getattr(b.handed, "name", "")).rpartition(".")[2] for b in branches if getattr(b, "handed", None)}
        if not branches or len(names) != 1 or not all(str(b.outcome or "").startswith("runtime(") and b.handed for b in branches):
            return None
        ways = SR.param_ways(_bare(owner), names.pop())
        if not ways or any(not w.route for w in ways):
            return None
        common = 0
        while all(len(w.route) > common and w.route[-1 - common] == ways[0].route[-1 - common] for w in ways):
            common += 1
        calls: dict[str, tuple[list[str], list[str]]] = {}
        for w in ways:
            at = w.route[-1 - common] if len(w.route) > common else w.route[0]
            others, kept = calls.setdefault(f"{Path(at.rpartition(':')[0]).name}:{at.rpartition(':')[2]}", ([], []))
            if w.end != SR.CONSTANT:
                why = f"{w.end}: {w.why}" if w.end == SR.STOP and w.why else w.end
            elif w.steps:
                why = "a part of a value"
            elif w.why in SR.NO_LITERAL:
                why = w.why
            elif SR.same_value(w.value, value):
                why = "the value"
            else:
                why = ""
                if w.value not in others:
                    others.append(w.value)
            if why and why not in kept:
                kept.append(why)
        return calls

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

    def _mark_row_after(self, cands: list[Candidate], subject: str, readings: list, state: dict | None) -> None:
        """P37: a candidate each of whose ways of writing the statuslog row *state* writes it by a call after its write
        (the map's history pairs, ``attr_time.after_on_row``) is read with the passes that ended before the row, where
        none is at it (``Candidate.row_after``: the time of the row before, "" where none).  Not for a window with no
        row (a value held, a time, F3's)."""
        if state is None or state.get("f3") or not state.get("status"):
            return
        fragment = mapdump.fragment()
        if self._recorded_values is None:
            self._recorded_values = AT.recorded_values(fragment)
        owners = AT.after_on_row(AT.arms(fragment, subject, readings), state, self._recorded_values)
        if not owners:
            return
        states = (self.data.get("statuslog") or {}).get("states") or []
        since = max((s["time"][:19] for s in states if s["time"][:19] < state["time"][:19]), default="")
        for c in cands:
            if c.owner in owners:
                c.row_after = since

    def _ended_before(self, cands: list[Candidate], at: dict[str, list], handed: dict[str, Run], runs: dict[str, list],
                      window: tuple[str, str]) -> dict[str, list]:
        """P37: for a candidate read with the passes that ended before the row (``Candidate.row_after``), the key's runs
        in each of its logs that hold none at the row and meet the stretch from the row before to it
        (``which_ran.before_stretch``), by log; said where taken.  A batched writer's run (:meth:`_batched`) stands."""
        out: dict[str, list] = {}
        for c in cands:
            if c.row_after is None or c.owner in handed:
                continue
            stretch = before_stretch(window, c.row_after)
            for name in c.log_files:
                if name in at or name in out or name not in runs:
                    continue
                if held := [r for r in runs[name] if spans(r, stretch)]:
                    out[name] = held
                    self.say(f"{c.func}: no run of the key in {name} at the row; the map has its history call after its "
                             f"write, so its runs from {stretch[0]} to the row are read (P37)")
        return out

    def _selection_false(self, owner: str, line: int, subject: str, at: datetime.datetime) -> str | None:
        """Q3 (P39, P42): why the selections *owner*'s write of *subject* at *line* draws its key from return no row of
        the object at *at*: each ``rows`` source of the write's key (the map's ``KeyFrom``, read in the writer itself)
        is a read site read as judgement 3 reads a route's record (:meth:`_route_refuses`: a refused pair it holds, a
        restricted subject whose values lack the row's, a condition false), at that one moment -- the object's status the
        statuslog's then, its other columns as judgement 3 takes them (the rows file's; KNOWN_ISSUES.md K17).  Every one
        false: the words; else None, and None where the key is drawn from no selection the map reads.
        getAchievedTasks_JEDI's 3303 asks ``goal IS NOT NULL``: a task whose row holds no goal is none of its rows
        (NR-T-53646395-assessmentTime).  I-4: where the key has no ``rows`` source in the writer but its parameters
        (an object made there with its key field from a parameter, the map's KMA: extractCommon's ``taskSpec.jediTaskID
        = jediTaskID``), each way the parameter comes in (``sources.param_ways``) must end at a statement's rows, and
        those selections are read so; one way ending anywhere else: None."""
        fragment = mapdump.fragment()
        history = PT.history_of_class(subject.partition(".")[0])
        if fragment is None or history is None:
            return None
        writes = [kf for kf in getattr(fragment, "key_froms", None) or []
                  if _bare(kf.owner) == _bare(owner) and line in (kf.writes.get(subject) or [])]
        sites = {rs.name: rs for rs in getattr(fragment, "read_sites", None) or []}
        names = [s.name for kf in writes for s in getattr(kf, "sources", None) or [] if s.kind == "rows" and not s.via]
        if not names:
            params = [(kf.owner, s.name) for kf in writes for s in getattr(kf, "sources", None) or []
                      if s.kind == KEY_PARAMETER and not s.via]
            ways = [w for owner_, p in params for w in SR.param_ways(owner_, p)]
            if not params or not ways or any(w.end != SR.DATABASE or not w.site for w in ways):
                return None
            names = list(dict.fromkeys(w.site for w in ways))
        if any(n not in sites for n in names):
            return None
        drawn = [sites[n] for n in names]
        states = [s for s in (self.data.get("statuslog") or {}).get("states") or [] if _when(s.get("time_extended") or s["time"]) <= at]
        if not states:
            return None
        then = f"at {at.isoformat(' ')[:19]}"
        q = Missed(history.subject, states[-1]["status"], "", at, at, [], "", None, end=then, start=then)
        said = []
        for site in {id(site): site for site in drawn}.values():
            why = self._route_refuses(site, q)
            if not why:
                return None
            said.append(f"@{site.anchor.line_start} " + "; ".join(why))
        return ", ".join(said)

    def _persisted_false(self, owner: str, line: int, subject: str, at: datetime.datetime) -> str | None:
        """K-4b (bundle K, out/gm/qk-design.md; the map's (A)(B)(C), codemap-xliii 637d38f): why no write persisting
        *owner*'s arm at *line* lands on the row at *at*, where the arm sets *subject* on an object it keeps in what it
        was handed (``KEY_KEPT``): the map's ``kept_hops`` name every caller's write persisting it (none open), and each
        such write -- a flush, read in the method it reaches (``flushed_in``, ``flushed_lines``), or the caller's own
        statement -- has a term on the class's history subject whose value, on every way the call hands it
        (``sources.param_ways``, each element where ``each_element``), is not the state the statuslog holds then: a
        literal of another state, or the rows of a selection whose selects lack it.  goal's arms (extractCommon 276,
        278) are persisted by TaskRefiner.runImpl's registerTaskInOneShot_JEDI (``status=:preStatus``) and seven
        updateTask_JEDI flushes (``status IN (:old___in)``), the value on getTasksToRefine_JEDI@1147's way the literal
        'registered' (1340), on @1369's its rows.  None where any of it is not read."""
        fragment = mapdump.fragment()
        history = PT.history_of_class(subject.partition(".")[0])
        states = [s for s in ((getattr(self, "data", None) or {}).get("statuslog") or {}).get("states") or []
                  if _when(s.get("time_extended") or s["time"]) <= at]
        if fragment is None or history is None or not states:
            return None
        state = states[-1].get("status")
        kfs = key_froms()
        kept = [kf for kf in kfs if _bare(kf.owner) == _bare(owner) and not getattr(kf, "term", "")
                and getattr(kf, "unread", "") == KEY_KEPT and line in (kf.writes.get(subject) or [])]
        hops = [h for kf in kept for h in getattr(kf, "kept_hops", None) or []]
        if not kept or not hops or any(getattr(h, "open", "") for h in hops):
            return None
        writes = list(dict.fromkeys((h.owner, w.statement, w.object) for h in hops for w in getattr(h, "persisted", None) or []))
        # the map's 3d30ddc follows the kept object down into callees too (``KeptHop.way`` handed): a callee's write
        # persisting another class's row there (appendDatasets_JEDI's 3776, a dataset: its terms' ``attr`` and its
        # writes are JediDatasetSpec's) persists no column of this one
        cls = subject.partition(".")[0]

        def of_class(caller: str, statement, obj) -> bool:
            own = [kf for kf in kfs if kf.owner == caller and getattr(kf, "statement", None) == statement
                   and getattr(kf, "object", "") == obj]
            return not own or any((getattr(kf, "attr", "") or "").partition(".")[0] == cls
                                  or any(c.partition(".")[0] == cls for c in kf.writes) for kf in own)

        writes = [w for w in writes if of_class(*w)]
        if not writes:
            return None
        sites = {rs.name: rs for rs in getattr(fragment, "read_sites", None) or []}
        takes: set[str] = set()
        for caller, statement, obj in writes:
            own = [kf for kf in kfs if kf.owner == caller and getattr(kf, "statement", None) == statement
                   and getattr(kf, "object", "") == obj]
            flushes = [kf for kf in own if getattr(kf, "flushed_in", "")]
            terms = [(kf, False) for kf in own if getattr(kf, "attr", "") == history.subject and getattr(kf, "term", "")] + [
                (ck, True) for fk in flushes for ck in kfs if ck.owner == fk.flushed_in
                and getattr(ck, "statement", None) in (getattr(fk, "flushed_lines", None) or [])
                and getattr(ck, "attr", "") == history.subject and getattr(ck, "term", "")]
            if not terms:
                return None
            call = f"{caller.partition('::')[0]}:{statement}"
            for term, flushed in terms:
                for s in term.sources:
                    if s.kind == KEY_CONSTANT and not getattr(s, "paths", None):
                        values = [s.name]
                    elif s.kind == KEY_PARAMETER and getattr(s, "site_owner", ""):
                        ways = SR.param_ways(s.site_owner, s.name, steps=("[*]",) if getattr(term, "each_element", False) else ())
                        ways = [w for w in ways if call in w.route] if flushed else ways
                        if not ways:
                            return None
                        values = []
                        for w in ways:
                            if w.end == SR.CONSTANT and w.why not in (SR.THROUGH, SR.MADE_EMPTY):
                                values.append(w.value)
                            elif w.end == SR.DATABASE and w.site in sites:
                                selected = [v.partition("=")[2] for v in getattr(sites[w.site], "selects", None) or []
                                            if v.partition("=")[0] == history.subject]
                                if not selected or state in selected:
                                    return None
                                takes |= {f"{w.site.rpartition('::')[2]} ({', '.join(selected)})"}
                            else:
                                return None
                    else:
                        return None
                    literals = {str(v).strip("'\"") for v in values}
                    if state in literals:
                        return None
                    takes |= {repr(v) for v in sorted(literals)}
        return (f"the {len(writes)} write(s) persisting it ({', '.join(sorted({c.rpartition('::')[2] for c, _, _ in writes}))}) "
                f"take the row in {', '.join(sorted(takes))} only, the statuslog holding {state!r} then")

    def _selected_out(self, cands: list[Candidate], subject: str, at: datetime.datetime) -> list[Candidate]:
        """Q3 (P39): *cands* less each line whose selection returns no row of the object at *at*, the time the write
        itself is in (a window holding the write, ``W.Asking.holds_write``), and a candidate left with no line; each
        said.  The judge then reads the rest as before: one left alone wrote there (KNOWN_ISSUES.md K47) --
        prepareTasksToBeFinished_JEDI's 3744, whose pass printed no key for a task with no finished job, once
        getAchievedTasks_JEDI's 3283 is out (NR-T-53646395-assessmentTime).

        Lines out by their offset first (:meth:`_offset_out`, P62)."""
        out = []
        for c in self._offset_out(cands, subject, at):
            kept = []
            for line in c.lines:
                if (why := self._selection_false(c.owner, line, subject, at)) is None:
                    kept.append(line)
                else:
                    self.say(f"  {c.func}:{line} did not write at {at}: the selection its key comes from returns no row "
                             f"of the object then: {why} (P39)")
            if kept:
                c.lines = kept
                out.append(c)
        return out

    def _offset_out(self, cands: list[Candidate], subject: str, at: datetime.datetime) -> list[Candidate]:
        """P62 (the investigator's u8 W11): *cands* less each line writing its clock plus a settled offset beyond SLACK
        (``_written_offset``, the map's ``Branch.amount``), for a window at *at* holding the write of the value at
        *at*: a pass then writes *at* plus the offset, not the value -- as P42 reads an unplaced arm.  A candidate left
        with no line is out; each said.  kickChildTasks_JEDI's 5181 writes ``CURRENT_DATE-1`` (child tasks'
        modificationTime), so its pass at 04:37:31 did not write the task's 04:37:31 (NR-T-53655760-modificationTime)."""
        offsets = {(_bare(j.owner), b.line): off for j in asyncio.run(self._writers_of(subject)) for b in j.branches
                   if (off := _written_offset(b)) is not None}
        out = []
        for c in cands:
            kept = []
            for line in c.lines:
                off = offsets.get((_bare(c.owner), line))
                if off is not None and abs(off) > W.SLACK:
                    self.say(f"  {c.func}:{line} did not write at {at}: its write is its clock plus "
                             f"{int(off.total_seconds())} s, so a pass then writes {at + off}, not the value (P62)")
                else:
                    kept.append(line)
            if kept:
                c.lines = kept
                out.append(c)
        return out

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
            runs, missed, holes = self._ask(key, names, asking, owners_of([c.owner for c in cands]), whole=True)
            unanswered += missed
            asked.append((state, readings, cands, asking, names, runs, holes, said))
        self._stop_unanswered(unanswered)
        chosen = 0
        for state, readings, cands, asking, names, runs, holes, said in asked:
            lo, hi = asking.window
            self.say(f"-- {state['status']!r} at {hi}: {len(readings)} reading(s) from the map, logs {names}")
            if said:
                self.say(f"   {said}")
            at_row = {name: held for name, rs in runs.items() if (held := [r for r in rs if spans(r, (lo, hi))])}
            handed = self._batched(cands, at_row, runs, holes, state, hi, history.subject)
            if not at_row and not handed:
                unsearched = "".join(f"; {name} not searched: {'; '.join(h)}" for name, h in holes.items())
                self.say(f"   no run of {key} at {hi} in {', '.join(runs) or 'no log asked'}{unsearched}: no writer chosen")
                continue
            # a batched writer's run ended before the row: a run of its own log, judged as the others are
            at_row = {**at_row, **{r.log.name: [r] for r in handed.values()}}
            # as H2: runs in more than one log are each judged on their own; in one log, the first is the row's
            question = None if len(at_row) > 1 else next(iter(at_row.values()))[0]
            at_hand = [question] if question is not None else [r for rs in at_row.values() for r in rs]
            env = self.base_env(at=state["time"])
            stable_holes([R.seen(R.function(c.rel, c.func, c.lines), run, self.mode)
                          for c in cands for run in at_hand if run.log.name in c.log_files], env)
            j = judge(
                cands, runs, mode=self.mode, decide=self.decide, window=(lo, hi),
                subject=attr, observed=state["status"], question_run=question, env=env, called_at=self._called_at,
                keeps=self._keeps(history.subject),
            )
            self.say(j.report(question))
            chosen += j.chosen is not None
        self.say(f"F1: a writer chosen for {chosen} of {len(states)} row(s); which of them can change the value, and the "
                 f"writes that leave no row, are not read yet (KNOWN_ISSUES.md K22)")
        self.hop = hop

    # ------------------------------------------------------------ Raised: what it was handed
    def raised(self, q: Raised):
        self.start("H5", "what it was handed")
        fn, question = q.fn, q.question
        chains = [
            (r, ch) for r in question.records for ch in r.chains()
            if any(f.rel == fn.rel and f.func == fn.name for f in ch.frames)
        ]
        record = self.need(T.Said("a-traceback-through-in-the-run", {"fn_name": fn.shown}), chains[0][0] if chains else None)
        if record.cut:
            self.need(T.Said("the-whole-traceback-of-the-search-cut", {"record_span": record.span}), None)
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
            ch, last = self.need(T.Said("a-frame-at-the-corpus-boundary-or"), inside[-1] if inside else None)
            self.say(f"raised inside the corpus: {last.rel}:{last.line} {last.func} ({last.source})")
            self.say(f"exception: {ch.exc_type}: {ch.exc_message.splitlines()[-1] if ch.exc_message else ''}")
            if not last.same_source:
                if last.source:
                    held = f"at lines {', '.join(map(str, last.printed_at))}" if last.printed_at else "at no line"
                    self.say(f"the pin's line {last.line} holds other text; its {last.func} holds the printed source"
                             f" {held} (the run's code was another version; KNOWN_ISSUES.md K33)")
                self.need(T.Said("the-frame-s-source-line-matches-the"), None)
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
        obj = self.need(T.Said("the-object-the-exception-names-seen-earlier"), names[-1] if names else None)
        missing = bool(re.search(r"NotFound|not found|does not exist", ch.exc_type + " " + ch.exc_message))
        self.say(f"object: {obj}  ({'the outside says it is missing' if missing else 'no word of it missing'})")
        self.need(T.Said("the-outside-says-the-object-is-missing"), missing)
        return Missing(obj, inner, question)

    # ------------------------------------------------------------ Missing: who makes it
    def missing(self, q: Missing):
        # H6: timed by the job's own rows, each on its host, in turn (W.rows_until; KNOWN_ISSUES.md K6)
        self.start("H6", "who makes it")
        obj, inner, question = q.obj, q.inner, q.question
        states = self.data["statuslog"]["states"]
        receivers = O.instances_in(inner.rel)
        receiver = self.need(T.Said("the-client-builds", {"inner_rel": inner.rel}), receivers[0] if receivers else None)
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
            held = [r for rs in runs.values() for r in rs if spans(r, (lo, hi))]
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
        self.need(T.Said("a-make-call-whose-function-printed-the"), makers)
        if len({(x.rel, x.func) for x, *_ in makers}) != 1:
            self.need(T.Said("one-function-among-the-makers"), None)
        x, f2, r2, _ = makers[0]
        make_calls = [m for m, *_ in makers if m.rel == x.rel and m.func == x.func]
        self.say(f"make calls in {x.func}: {[(m.line, m.method) for m in make_calls]}")
        timeline = ", ".join(f"{st['status']} {st['time']}" for st in states)
        self.say(f"the run {r2.first}..{r2.last}; statuslog {timeline}")

        # H7
        self.start("H7", "which way it went")
        make = self.need(T.Said("the-make-call-of-the-object"), next((m for m in make_calls if m.first_arg), None))
        fn7 = R.function(make.rel, make.func, within=[make.line])
        call_stmt = R.statement_at(fn7.node, make.line)
        loop = next(
            (a2 for a2 in pathcond.ancestors(call_stmt) if isinstance(a2, (ast.For, ast.AsyncFor)) and ast.unparse(a2.target) == make.first_arg),
            None,
        )
        loop = self.need(T.Said("the-loop-that-binds", {"make_first_arg": make.first_arg}), loop)
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
            # read at the test: another function's hole only where the map hands its object here (KNOWN_ISSUES.md K63)
            at7 = env7.reading(fn7.owner, i.lineno)
            value = E.evaluate(ast.unparse(i.test), at7)
            env7.used.update(at7.shared())
            used = {k: v for k, v in {**env7.used, **at7.used}.items() if k in ast.unparse(i.test)}
            unread = env7.unread_in(ast.unparse(i.test))
            used |= {k: f"unknown: {why}" for k, why in unread.items()}
            self.say(f"  implicit else of {i.lineno}: test {ast.unparse(i.test)!r} is {value!r} {used}" + (" -> it did not run" if value is True else ""))
            if value is not True:
                if unread:
                    self.rows_between(env7.at)
                if file_row is None:
                    self.say(f"  the rows file has no file row whose destinationDBlock is {obj!r}: its values are not read")
                self.need(T.Said("the-implicit-else-of", {"i_lineno": i.lineno}), None)
        cand7 = Candidate(make.rel, make.func, sorted(set(arms)), run=r2)
        j = judge([cand7], {r2.log.name: [r2]}, mode=self.mode, decide=self.decide, question_run=r2, env=env7)
        self.say(j.report(r2))
        exit7 = self.pick(T.Said("which-way-it-went"), j)
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
        self.need(T.Said("a-settled-comparison-of-two-values"), settled)
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
            self.say(f"  {atom!r} ({truth} at {fn7.shown}:{exit7.line}): {len(found)} occurrence(s)")
            for o in found:
                if (o.fn.owner, o.line) in seen_at:
                    continue
                seen_at.add((o.fn.owner, o.line))
                self.say(f"    {o.fn.shown}:{o.line} -> {o.value!r}  {dict(o.env.used)}")
                if o.value is (not truth):
                    pairs.append((atom, truth, o))
        self.need(T.Said("exactly-one-test-that-came-out-the"), len({(a2, o.fn.owner, o.line) for a2, _, o in pairs}) == 1)
        atom, truth, other = pairs[0]
        self.say(f"contradiction: {fn7.shown}:{exit7.line} has {atom} {truth}; {other.fn.shown}:{other.line} has it {other.value}")
        a_side = V.Side(f"{fn7.shown}", q.steps, q.env, fn7.owner)
        b_side = V.Side(f"{other.fn.shown}", V.walk_steps(other.fn, other.line), other.env, other.fn.owner)
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
                    end = T.Said("stops-at-a-terminal", {
                        "label": side.label, "line": t.step.line, "terminal": t.step.terminal, "detail": t.step.detail,
                        "loose": T.Said("loose-not-looked-up", {"loose": loose}) if loose else "",
                    })
                    ends.append((end, self.looked_up_with(side, t.step)))
        for end, how in ends:
            self.end(end)
            if how:
                self.say("    looked up with:")
                for x in how:
                    self.say(f"      {x}")
        # an input that differs by value and ends in no stop has not said where
        # the two sides part; the two sides being unknown is not an answer either
        self.need(T.Said("an-input-the-two-sides-part-on"), ends)
        return None

    # ------------------------------------------------------------ FalseAtom: an exception as a false guard
    def false_atom(self, q: FalseAtom):
        # R1
        self.start("R1", "what raised")
        frame, ch, question = q.frame, q.chain, q.question
        fn = R.function(frame.rel, frame.func, within=[frame.line])
        stmt = R.statement_at(fn.node, frame.line)
        self.need(T.Said("the-statement-at", {"frame_func": frame.func, "frame_line": frame.line}), stmt)
        self.say(f"{Path(frame.rel).name}:{frame.line} {frame.func}: {ast.unparse(stmt).splitlines()[0]}")
        read = X.read(stmt, ch.exc_type, ch.exc_message)
        for g in read:
            self.say(f"  {g.describe()}")
        if len(read) != 1:
            self.need(T.Said("one-expression-of-the-line-that-raises", {"ch_exc_type": ch.exc_type}), None)
        fg = read[0]
        if fg.kind != "in":
            self.need(T.Said("a-way-to-compare-the-operands-of", {"fg_kind": fg.kind}), None)

        # R2
        self.start("R2", "what the operands held")
        env = self.base_env(at=question.first if question is not None else None)
        cls = class_of(fn)
        # a plugin's own attributes (``self.add_to_top_only``) printed by its other methods in the run
        stable_holes([R.seen(f, q2, self.mode) for f, q2 in self.touched if q2 is question and class_of(f) == cls], env, cls)
        steps = A.walk(fn, frame.line)
        self.say(f"walk from {fn.shown}:{frame.line}: asked by question" + (f"; {steps.note}" if steps.note else ""))
        ran = [(f"{f.rel}::{f.func}", f.line) for f in ch.frames if f.in_corpus]
        walk = G.Walk(steps, env, self._collections(env), self.data, ran, ask=self.ask_collection)
        if walk.held:
            self.say(f"held (the traceback's frames ran past them): {sorted(f'{w}: {t}' for _, w, t in walk.held)}")
        drawn = walk.element_of(fg.operand, fn.owner)
        self.say(f"{fg.operand} = {fg.printed!r} (the message)" + (f", an element of the keys of {drawn[0]}" if drawn else ""))
        key_side = self.need(T.Said("the-container-is-drawn-from", {"fg_operand": fg.operand}), drawn)
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
            self.need(T.Said("the-key-set-the-walk-gives-consistent"), None)
        if not c_keys.complete():
            self.need(T.Said("a-complete-key-set-for", {"fg_other": fg.other}), None)

        # R3
        self.start("R3", "where they part")
        self.need(T.Said("one-binding-of-the-key-for-the"), parts)
        if len({(p.alternative.step.line, p.decided_by) for p in parts}) != 1:
            self.need(T.Said("one-alternative-and-one-guard-against-it"), None)
        p = parts[0]
        write, took = p.taken
        self.say(f"took effect: {took.where()} = {took.value!r}" + (f"  [guards {took.step.guards}]" if took.step.guards else ""))
        self.say(f"instead of:  {p.alternative.where()} = {p.alternative.value!r}, which is among {fg.other}'s keys")
        used = {k: v for k, v in p.env.used.items() if k in p.decided_by}
        self.say(f"the guard against it: {p.decided_by!r} is False for {p.alternative.element.label if p.alternative.element else 'this call'}  {used}")
        context, rest = G.inputs(p.decided_by, p.alternative.element)
        self.say(f"its inputs: {rest}; the loop's row: {context}")
        self.need(T.Said("one-input-of-the-guard-besides-the"), len(rest) == 1)
        expr = rest[0]
        # read at the alternative's binding: another function's hole only where the map hands its object (K63)
        step = p.alternative.step
        step_owner: str = getattr(step, "owner", "")
        step_line: int = getattr(step, "line", 0)
        at = p.env.reading(step_owner, step_line) if step_owner else p.env
        value = E.evaluate(expr, at)
        p.env.used.update(at.shared())
        self.say(f"  {expr} = {value!r} ({at.used.get(expr, p.env.used.get(expr, 'unknown'))})")
        how, leaves = G.derivation(expr, walk)
        for line in how:
            self.say(f"    {line}")
        leaf = self.need(T.Said("what-is-computed-from-in-the-walk", {"expr": expr}), leaves)
        self.need(T.Said("one-thing-is-computed-from", {"expr": expr}), len(leaf) == 1)
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
            self.say(f"{fn.shown}({', '.join(params)}) {label}:")
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
            self.need(T.Said("the-rebuilt-inputs-raise-what-the-log"), None)
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
        found = self.need(T.Said("the-value-of-a-field-of-the", {"what": what, "fact": m[1] if m else '?', "why": why}),
                          None if value is E.UNKNOWN else repr(value))
        self.end(T.Said("said", {"what": what, "found": found}))
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
        follow = self.need(T.Said("the-map-s-answer-to-what-moves"), strategy.follow_up)
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
        owners = owners_of([*movers, *keyed])
        wakes = wakers_of(follow, writers, q.subject, q.observed)
        # M2: where the selecting statements' rows go; a writer by its key they do not reach is left out (value_reach)
        value_reach = value_reach_of(follow, q.subject, q.observed, movers)
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
        rest = W.rest(self.data.get("statuslog", {}), q.observed)
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
            entry: dict = {"owner": owner, "reached": way,
                           "statements": [s for s, _ in mine if s], "verdict": None, "reasons": [],
                           "readings": [{"statement": s or None, "verdict": v.replace(" ", "-") if v else v} for s, v in mine],
                           "premises": None, **({"gaps": _gaps_json(gaps[owner])} if gaps.get(owner) else {})}
            entries.append(entry)
            if owner in keyed and value_reach.leaves_out(owner):
                told = T.Said("not-reached-from-the-value-s-reads", {"observed": q.observed, "reads": list(value_reach.reads)})
                self.say(f"  {owner.split('::')[-1]}: {T.text(told)}")
                entry["verdict"] = "not-reached"
                entry["reasons"].append({"code": "M2", "record": told})
                continue
            if owner in elsewhere:
                to_key, said, reach = elsewhere[owner]
                entry["reasons"].append({"code": "K16" if to_key else "K14", "record": said})
                if reach is not None:
                    entry["reach"] = reach
            if any(v == "refused" for _, v in mine):
                entry["reasons"].append({"code": "K17", "record": T.Said("refused-at-the-moment-read")})
            if made := births_of(writers, owner, q.subject, q.observed):
                told = T.Said("births-not-arms", {"made": made, "births": birth_why(owner, q.subject, made)})
                self.say(f"  {owner.split('::')[-1]}: {told}")
                entry["births"] = made
                entry["reasons"].append({"code": "K29", "record": told})
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
                dropped, reach_line = reach_said(reach, hidden)
                if reach is not None:
                    entry["reach"] = reach
                if dropped:
                    self.say(f"  {func}: {reach_line}")
                    entry["verdict"] = "reaches-no-write"
                    entry["reasons"].append({"code": "reach", "record": reach_line})
                    continue
                kept_said = T.Said("no-arm-kept", {"kept": kept})
                self.say(f"  {func}: {kept_said}")
                entry["reasons"].append({"code": "K14", "record": kept_said})
                if reach_line:
                    self.say(f"  {func}: {reach_line}")
            for w in wakes.get(owner, []):
                self.say(f"  {owner.split('::')[-1]}: {wake_said(w)}")
                entry.setdefault("wakers", []).append({"waker": w.owner, "line": w.line, "writes": w.writes,
                                                       "reader": w.reader, "term": w.term, "candidate": list(w.candidate)})
            if gaps.get(owner):
                entry["reasons"].append({"code": "map-gap", "record": T.Said("map-could-not-read-all", {"gaps": entry["gaps"]})})
            passes, missed = self._absence(owner, arms, files or follow.reader_log_files, entry, writers, q.subject, q.key,
                                           asking, rest, way, owners)
            unanswered += missed
            if not passes and entry["verdict"] in ("did-not-run", "no-run-found"):
                # P25 (i)'s rest, stuck side: no run of an arm of the value, but a pass that took the row the row's own
                # time column names
                for took in self._stuck_value_takes(q, owner, [rs for rs in follow.reader_statements
                                                               if asked in rs.selects and rs.owner == owner], writers):
                    self.say(f"  {owner.split('::')[-1]}: {T.text(took)}")
                    entry["reasons"].append({"code": "takes-by-value", "record": took})
            if passes:
                ran.append(passes)
            elif entry["verdict"] == "not-searched":
                unsearched.append(owner.split("::")[-1])
        # a mover dropped as reaching no write may be what the movers kept wait on (the map's wakers, ix's 6 b)
        woken = woken_of(entries, wakes)
        for entry in entries:
            if entry["owner"] in woken:
                woken_line = woken_said(woken[entry["owner"]])
                self.say(f"  {entry['owner'].split('::')[-1]}: {woken_line}")
                entry["reasons"].append({"code": "K39", "record": woken_line})
        self.result["movers"] = entries
        self._stop_unanswered(unanswered)
        # every mover that ran and left the value is part of the answer: each goes on alone (tmaeno, 2026-10-13)
        self.need(
            T.Said("a-mover-with-a-run-to-hand", {
                "window": T.Said("in-parens", {"what": asking.why}) if self.logs.needs_window else "",
                "unsearched": T.Said("not-searched-all-the-window", {"unsearched": unsearched}) if unsearched else "",
            }),
            ran,
        )
        if unsearched:
            self.say(f"movers that may have run where the search did not reach: {unsearched}")
        if len(ran) == 1:
            return ran[0]
        return Fork(ran, [T.Said("mover-label", {"owner": p.owner}) for p in ran])

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
        own (:func:`map_reach.by_call`: from a copy of *caller*'s row holding that call alone, by the map's own rules,
        the classes a call holds for and enters as).  A way that comes back into *caller* and on through its other calls
        counts as inside the call, as it runs there."""
        record = MR.graph()
        if record is None or not caller.node.body or not callee.lines:
            return []
        start = record.function_at(caller.rel, caller.node.body[0].lineno)
        end = record.function_at(callee.rel, callee.lines[0])
        if start is None or end is None or start == end:
            return []
        if (start, end) not in self._calls_into:
            file = start.partition("::")[0]
            lines = {ln for call, reached in MR.by_call(start) if end in reached for ln in call.lines or [call.line]}
            self._calls_into[(start, end)] = [f"{file}:{line}" for line in sorted(lines)]
        return self._calls_into[(start, end)]

    def _keeps(self, subject: str) -> Callable[[R.Function, ast.AST, str], int | None]:
        """Rule R's reading of a call on the way of a value it carries (``changed.breaks``), by the map's reach: whether
        the calls in a statement's *node* of *fn* keep *subject*'s field at *value* in the object they change -- from
        those calls alone, as ``_called_at`` asks a call (a copy of the row holding them, with its holes at their
        lines), the reach goes on to no arm assigning the field another value in memory
        (``reach.moving_off_in_memory``; births left out as ``unborn_junctions`` does) and leaves no hole.  The answer
        is how many calls it closed by their name, or None, as where the dump has no reach_graph."""

        def keeps(fn: R.Function, node: ast.AST, value: str) -> int | None:
            graph = self._graph()
            if graph is None:
                return None
            RE, record, rows = graph
            if (subject, value) not in self._in_memory:
                self._in_memory[subject, value] = RE.moving_off_in_memory(
                    unborn_junctions(mapdump.fragment().junctions, subject), record, subject, value)
            destination = self._in_memory[subject, value]
            start = record.function_at(fn.rel, fn.node.body[0].lineno) if fn.node.body else None
            first, last = getattr(node, "lineno", None), getattr(node, "end_lineno", None)
            # a junction the reach cannot place may be the arm
            if start is None or first is None or destination.unmatched:
                return None
            if (start, first, last or first, subject, value) not in self._kept:
                lines = set(range(first, (last or first) + 1))
                row = record.rows[start]
                one = row.model_copy(update={
                    "owner": f"{start}#keeps{first}",
                    "calls": [c for c in row.calls if lines & set(c.lines or [c.line])],
                    "holes": [h for h in row.holes if h.line in lines],
                })
                got = RE.ReachGraph([*rows, one]).reach([one.owner], destination)
                self._kept[start, first, last or first, subject, value] = (
                    len(got.closed) if got.writes == [] and not got.holes else None)
            return self._kept[start, first, last or first, subject, value]

        return keeps

    def _batched(self, cands: list[Candidate], at: dict[str, list], runs: dict[str, list], holes: dict[str, list[str]],
                 state: dict | None, hi: str, subject: str) -> dict[str, Run]:
        """W1a' (D3, KNOWN_ISSUES.md K52): a candidate with no run of the key at the row's second (*at*) that writes in
        a batch (:func:`batched`) is handed, as its run, the key's latest run in its logs held between the row before
        *state*'s and the row, the log searched with no hole -- the pass whose batch the row's write ended, where no
        other pass of it printed the key between.  None held: it stays as it is, said so, and nothing is asked.  The
        candidates handed a run, by owner."""
        out: dict[str, Run] = {}
        if state is None:
            return out
        before = None
        for c in cands:
            if any(name in at for name in c.log_files) or (found := batched(c.rel, c.func, c.lines, subject)) is None:
                continue
            if before is None:
                # the row before, read once a batched writer is a candidate: "" where none is
                states = (self.data.get("statuslog") or {}).get("states") or []
                before = max((s["time"][:19] for s in states if s["time"][:19] < state["time"][:19]), default="")
            if not before:
                break
            line, kf, loop = found
            facts = {"func": c.func, "line": line, "loop": loop, "statement": kf.statement,
                     "call": ", ".join(h.split("::")[-1] for h in kf.handed_to) or "a call", "before": before, "when": hi}
            held = [r for name in c.log_files if name in runs and name not in holes
                    for r in runs[name] if before < r.first[:19] and r.last[:19] < hi[:19]]
            if not held:
                self.say(T.text(T.Said("writes-in-a-batch-no-run-held", facts)))
                continue
            c.run = out[c.owner] = max(held, key=lambda r: r.last)
            self.say(T.text(T.Said("writes-in-a-batch-its-run", {**facts, "first": c.run.first, "last": c.run.last,
                                                                "log": c.run.log.name})))
        return out

    def _rows_elsewhere(self, q: Stuck, follow, writers: list, movers: dict, gaps: dict) -> dict[str, tuple[bool, T.Said, dict | None]]:
        """For each mover whose statements' rows reach none of its arms (``elsewhere_of``), what the map's reach says of
        it as of a mover with no arm of its own, those arms left out of where it ends (``_reach_of``'s *without*).
        Where it reaches no write of another value with nothing left open, and no gap of the map's may hide where its
        rows go (``reach_said``), the rows it selects stay with it unwritten, and its arms are those of a writer that
        takes a row by its key (KNOWN_ISSUES.md K16): ``(True, the line saying so, the reach)``.  Else it stays a mover,
        its arms as before, said why: ``(False, ...)``."""
        out: dict[str, tuple[bool, T.Said, dict | None]] = {}
        asked = f"{q.subject}={q.observed}"
        for owner, arms in elsewhere_of(follow, writers, q.subject, q.observed).items():
            if owner not in movers:
                continue
            mine = [rs for rs in follow.reader_statements if rs.owner == owner and asked in rs.selects and rs.result != RESULT_AGGREGATE]
            lines = [rs.line for rs in mine]
            rel, func = owner.split("::", 1)
            reach = self._reach_of(q, owner, lines, without=arms) if defs_holding(rel, func, lines) else None
            dropped, said = reach_said(reach, hides_where(gaps.get(owner, [])))
            whose = T.Said("rows-select-reach-none", {"selects": [f"{func}@{n}" for n in lines], "arms": arms})
            if dropped:
                assert reach is not None  # reach_said drops only on the reach's answer
                out[owner] = (True, T.Said("reach-writes-no-other-by-key", {"whose": whose, "closure": reach["closure"],
                                                                            "assumptions": sorted(reach["assumptions"])}), reach)
            else:
                out[owner] = (False, T.Said("kept-by-reach", {"whose": whose, "said": said or T.Said("the-reach-was-not-asked")}), reach)
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

    def _absence(self, owner: str, arms: dict, today: list[str], entry: dict, writers: list, subject: str, key: str,
                 asking: W.Asking, rest: W.Asking | None, way: str, owners: Owners | None) -> tuple[Passes | None, list]:
        """Judgement 3 for one mover with arms of its own, S1's and L2's alike (KNOWN_ISSUES.md K19): its logs asked for
        a run of *key* over *asking*, the window's last part, and over *rest*, the part before it, where the last part
        holds none and the mover is not reached by its key (*way*, ``_ask_mover``); a run not found rules it out only
        where the three premises hold (``_premises``).  *today*: the logs the map's reading names, where it places none
        of the prints.  Sets *entry*'s ``verdict`` (``prints-nothing``, ``no-log``, ``ran``, ``not-searched``,
        ``asks-unanswered``, ``did-not-run``, ``no-run-found``), ``premises`` and ``reasons``; returns the passes of a run
        found, or None, and the asks no record answered."""
        name, lines = owner.split("::")[-1], list(arms)
        if prints_nothing(owner, lines):
            said = T.Said("prints-nothing-not-followed", {"n": len(arms), "arms": [(a.line, a.outcomes) for a in arms.values()]})
            self.say(f"  {name}: {said}")
            entry["verdict"] = "prints-nothing"
            entry["reasons"].append({"code": "K25", "record": said})
            return None, []
        files = self.files_of(owner, list(dict.fromkeys(today)), lines)
        to_hand = self.logs.names(files)
        if to_hand:
            self.warn_split(owner, lines)
        self.say(f"  {name}: {len(arms)} arm(s) that write another value "
                 f"{[(a.line, a.outcomes) for a in arms.values()]}; logs {files} ({'to hand: ' + str(to_hand) if to_hand else 'none to hand'})")
        if not to_hand:
            entry["verdict"] = "no-log"
            entry["premises"] = self._premises(subject, key, owner, lines, to_hand, asking, asking, [], {}, writers, rest)
            return None, []
        found, missed, holes, searched = self._ask_mover(key, owner, to_hand, asking, rest if way != "key" else None, owners)
        logs = self._runs_of(found, owner, searched)
        if logs:
            entry["verdict"] = "ran"
            return Passes(owner, logs, list(arms.values()), key, searched, owners, subject=subject,
                          entries=_entries_of(writers, owner)), missed
        if way == "key" and rest is not None and rest.window is not None and self.logs.needs_window and not holes:
            # G-3 (P12, bundle G, tmaeno's OK 2026-11-17 16:0x): a writer taking the row by its key may have run as the
            # call that began the stay did -- setupper's activateJob at NJ-defined-late's start: the stay's start is
            # asked too, from the row less SLACK to SLACK after it.  A run there is read; none says no more than before,
            # and nor does an ask no record answers: it is kept for a records round, and the step goes on as before
            # (merged into the step's asks, it stopped NJ-transferring-stuck's S1 ahead of the branch bundle D ended)
            lo = rest.window[0]
            hi = min(rest.window[1], W._down(W.when(lo) + 2 * W.SLACK))
            head = W.Asking((lo, hi), None, T.Said("the-stay-s-start-by-key", {"start": lo, "end": hi}))
            hfound, hmissed, _, _ = self._ask_mover(key, owner, to_hand, head, None, owners)
            if hmissed:
                self._keep_unanswered(hmissed)
            if hlogs := self._runs_of(hfound, owner, head):
                self.say(f"  {name}: a run of its own at the stay's start, {lo} .. {hi} (G-3)")
                entry["verdict"] = "ran"
                return Passes(owner, hlogs, list(arms.values()), key, head, owners, subject=subject,
                              entries=_entries_of(writers, owner)), missed
            self.say(f"  {name}: {'no run of its own at' if not hmissed else 'not recorded at'} the stay's start, "
                     f"{lo} .. {hi}, either (G-3)")
        entry["premises"] = self._premises(subject, key, owner, lines, to_hand, asking, searched, missed, holes, writers, rest)
        if holes:
            # not found is not "did not run" where the search did not reach
            self.say(f"  {name}: no run of its own found, but {', '.join(holes)} were not searched all the window: it may have run there")
            entry["verdict"] = "not-searched"
        elif missed:
            entry["verdict"] = "asks-unanswered"
        else:
            # not followed; ruled out only where judgement 3's premises all hold (build order 2)
            over: T.Said | str = T.Said("over-the-whole-window") if searched is not asking else ""
            if way == "key" and self.logs.needs_window and rest is not None:
                over = T.Said("from-the-last-part-by-key", {"start": asking.window[0]})
            if all(p["held"] for p in entry["premises"].values()):
                said = T.Said("did-not-run-judgement-3", {"to_hand": to_hand, "over": over})
                entry["verdict"] = "did-not-run"
                entry["reasons"].append({"code": "judgement-3", "record": said})
            else:
                said = T.Said("no-run-found-not-ruled-out", {"to_hand": to_hand, "over": over})
                entry["verdict"] = "no-run-found"
                entry["reasons"].append({"code": "K19", "record": said})
            self.say(f"  {name}: {said}")
        return None, missed

    def _ask_mover(self, key: str, owner: str, to_hand: list[str], asking: W.Asking, rest: W.Asking | None,
                   owners: Owners | None = None):
        """Judgement 3's asks for one mover: the window's last part first (*asking*, ``STUCK_SPAN``), and the part
        before it (*rest*, ``W.rest``) only where every ask of the last part was answered whole and held no run of its
        own -- no pass in six hours is not no pass in the window, and a mover that ran and stopped is the one a stuck
        value most needs named (build order 2a; tmaeno, 2026-10-16).  No *rest* for a writer that takes a row by its
        key: its logs are mostly the DB proxy's, grepped whole over days (KNOWN_ISSUES.md K16; tmaeno, 2026-10-26).  The
        door over files reads each file whole and asks once.

        Returns what ``_ask`` does, and the asking the runs were looked for over."""
        found, missed, holes = self._ask(key, to_hand, asking, owners)
        if rest is None or missed or holes or not self.logs.needs_window or self._runs_of(found, owner, asking):
            return found, missed, holes, asking
        self.say(f"  {owner.split('::')[-1]}: no run of its own from {asking.window[0]}: asking the rest of the window")
        return *self._ask(key, to_hand, rest, owners), rest

    def _runs_of(self, found: dict[str, list], owner: str, asking: W.Asking) -> list[str]:
        """The logs among *found* holding a run of *owner* in *asking*'s window (any run, for the door over files)."""
        window = asking.window if self.logs.needs_window else None
        return [name for name, rs in found.items() if any((window is None or spans(r, window)) and of_mover(r, owner) for r in rs)]

    def _premises(self, subject: str, key: str, owner: str, lines, to_hand: list[str], asking: W.Asking,
                  searched: W.Asking, missed: list, holes: dict, writers: list, rest: W.Asking | None) -> dict:
        """Judgement 3's premises for a mover with no run of its own found: absence rules it out only where all three
        hold (build order 2a; KNOWN_ISSUES.md K19).  ``window``: the whole window was searched; ``logs``: every log its
        runs print to was asked (``every_log_asked``); ``prints``: every run that reaches an arm prints the key the asks
        search by (``prints_naming``, over *writers*).  Each is ``{"held": True | False | None, "why": ...}``, None where
        it is not known.

        *asking* is the window's last part (``W.last_part``), *rest* the part before it, *searched* the part
        ``_ask_mover`` asked last, and *missed* and *holes* what that ask returned."""
        return {"window": self._searched_whole(to_hand, asking, searched, missed, holes, rest),
                "logs": every_log_asked(owner, lines, to_hand),
                "prints": prints_naming(list(writers), owner, lines, self._key_columns(subject, key))}

    def _key_columns(self, subject: str, key: str) -> set[str]:
        """The columns of *subject*'s class whose value in the rows file is *key* -- the row's key, as ``held.split``
        reads it -- spelled ``Class.attribute`` as the map's names are."""
        cls = subject.partition(".")[0]
        return {f"{cls}.{c}" for c, cell in self._row_of(cls).items() if isinstance(cell, dict) and str(cell.get("value")) == key}

    def _searched_whole(self, to_hand: list[str], asking: W.Asking, searched: W.Asking, missed: list, holes: dict,
                        rest: W.Asking | None) -> dict:
        """Judgement 3's first premise: the logs were searched over the whole window, from the value's row less SLACK
        to its end (stuck's ``as_of``, late's move), every ask answered and nothing of it left unsearched
        (``Log.holes``); *rest*, the part before the last, asked too where there is one.  Not known through the door
        over files: a file handed in does not say what of the window it holds."""
        if not to_hand:
            return {"held": False, "why": T.Said("window-no-log-asked")}
        if not self.logs.needs_window:
            return {"held": None, "why": T.Said("window-door-over-files")}
        # a hole first: asking again does not search better what a log's answer did not reach
        if holes:
            return {"held": False, "why": T.Said("window-not-searched-all", {"logs": list(holes)})}
        if missed:
            return {"held": None, "why": T.Said("window-asks-not-answered", {"n": len(missed)})}
        if searched is asking and rest is not None:
            return {"held": False, "why": T.Said("window-searched-from-only", {"start": asking.window[0]})}
        return {"held": True, "why": T.Said("window-searched", {"start": searched.window[0], "end": asking.window[1], "to_hand": to_hand})}

    async def _writers_of(self, subject: str) -> list:
        writers = await mapdump.code_map("rerun").writers_of(subject)
        R.check_blobs(writers, f"the map's writers of {subject}")
        return on_rows(writers)

    # ------------------------------------------------------------ Passes: which way each pass went
    def passes(self, q: Passes):
        self.start("S2", "which way each pass went")
        rel, func = q.owner.split("::", 1)
        # the arms S2 and S3 judge are the mover's own; a mover that writes through what it calls has none here
        self.need(T.Said("an-arm-of-s-own-that-writes", {"func": func}), q.arms)
        fn = R.function(rel, func, [a.line for a in q.arms])
        runs = self.runs(q.key, q.logs, q.asking, q.owners)
        found = []
        whole: set[int] = set()
        for name in q.logs:
            for run in runs.get(name, []):
                if not of_mover(run, q.owner):
                    continue
                parts, how = passes_of(fn, run, self.mode)
                self.say(f"{run.describe()}: {how}")
                found += parts
                whole |= {id(p) for p in parts if p is run}
        # the runs come machine by machine; the passes are read, and the first and the last taken, in time
        found.sort(key=lambda p: p.first)
        states = self.data.get("statuslog", {}).get("states", [])
        # the stint the value is stuck in began at its last row; a late value's at the stay L1 read
        since = q.since or next((s["time"] for s in reversed(states) if s["status"] == self.args.observed), None)
        moved_on: list = []
        if since is None:
            # a log's coverage stands in for the window only in a file handed in: a door that searches has no other
            self.need(T.Said("a-statuslog-row-for-to-start-the", {"observed": self.args.observed}), not self.logs.needs_window)
            for name in q.logs:
                first, last = self.logs.log(name, q.key).covers
                self.say(f"window: the log's coverage {first} .. {last} (the statuslog has no row for {self.args.observed!r})")
        else:
            if self.logs.needs_window and q.asking.window[0] > since:
                since = q.asking.window[0]
                self.say(f"window: from {since} ({q.asking.why})")
            else:
                self.say(f"window: from {since}, when the statuslog shows {self.args.observed!r}")
            found, moved_on = self._stay_edges(fn, q.owner, [a.line for a in q.arms], q.subject or self.args.subject,
                                               q.key, found, since, whole)
        if self.logs.needs_window:
            # a search returns its files' records whatever the window; the statuslog says nothing after its as_of
            # (the file door keeps them: KNOWN_ISSUES.md K4)
            later = [p for p in found if p.first[:19] > q.asking.window[1]]
            if later:
                found = [p for p in found if p.first[:19] <= q.asking.window[1]]
                self.say(f"{len(later)} pass(es) after {q.asking.window[1]}, the end of the window, not read")
        # a pass with no record of the key, nor a run of it inside a call, went its way for other rows: a poller's
        # day sits under one mark with no key (getTasksToExecCommand_JEDI's 1149 passes, none of the key's; K47)
        keyed = [p for p in found if holds_key(fn, p, runs, self.mode, self._called_at)]
        unkeyed, found = len(found) - len(keyed), keyed
        if unkeyed:
            self.say(f"{unkeyed} pass(es) of {func} in the window print no record of the key, nor run it inside a call: not read")
        if q.stay is not None and not found:
            # a mover L2 follows for a late value whose runs fall outside the stay: it did not take the row there
            raise Excluded(T.at(self.hop, T.Said("no-pass-of-in-the-stay", {"func": func, "since": q.stay[0], "moved": q.stay[1]})))
        if q.before is not None:
            # judgement 4: the pass that moved the value on is not why it moved only then; the ones before it are.  The
            # one whose calls recorded the move, where the records show it (BP'); else those ending at its second or later
            if not moved_on:
                moving = [p for p in found if p.last[:19] >= q.before]
                found = [p for p in found if p.last[:19] < q.before]
                self.say(f"{len(moving)} pass(es) ending at {q.before} or later, the one that moved it on among them, not read")
            if not found and since is not None and (self._waker_turn(q, since, func) or self._hop_turn(
                    q.owner, q.subject or self.args.subject, since, q.before, None, func) or self._count_turn(
                    q.owner, q.subject or self.args.subject, since, q.before, func)):
                return None
            if not found and (up := self._callers_of_mover(q, since, "before")) is not None:
                return up
            if q.unsure:
                self.need(T.Said("a-pass-of-before-the-one-that-2", {"func": func, "q_before": q.before, "since": since, "asking_why": q.asking.why, "q_unsure": q.unsure}), found, "K45")
            self.need(T.Said("a-pass-of-before-the-one-that", {"func": func, "q_before": q.before, "since": since, "asking_why": q.asking.why, "func_2": func}), found, "K45")
        if not found and since is not None and q.before is None:
            # a stuck value's stay: A2 ends the branch where no row the mover takes it by came and the table is written
            # here alone (A2s), and stops it where another system writes it too; H stops it where the other rows its
            # selection waits on had not moved on; else U1 reads on
            if self._waker_turn(q, since, func):
                return None
            self._hop_turn(q.owner, q.subject or self.args.subject, since, None,
                           q.asking.window[1] if q.asking.window is not None else None, func)
        if not found and (up := self._callers_of_mover(q, since, "window")) is not None:
            return up
        self.need(T.Said("a-pass-in-the-window-that-holds-the-key", {"func": func, "unkeyed": unkeyed}) if unkeyed
                  else T.Said("a-pass-of-the-mover-in-the-window"), found)
        blocked = self._which_way(fn, rel, func, found, q.arms)
        if q.before is None and q.stay is None and since is not None and (up := self._callers_after(q, found, blocked)) is not None:
            return up
        return blocked

    def _callers_after(self, q: Passes, found: list, blocked: Blocked) -> Callers | None:
        """I-2 (D-4b, out/gm/qk-design.md's bundle I second half, tmaeno's OK 2026-11-17 20:3x): a stuck value's mover
        whose arms' writes take the row by what its callers hand it alone (the map's KeyFrom sources, each a parameter)
        runs on the row again only where a caller hands it the row.  After its last pass on the row (*found*'s, the way
        S2 read them *blocked*) to the window's end, the statuslog's as_of, its callers' passes are read (U1, kind
        ``after``), in place of S3 on the guard that pass left by: activateJob's 21:32:49 pass left by 151, then
        ActivatorWithRuleThr's turns of the key each left before the append (D-4; NJ-assigned-stuck).  None where the
        map lists no caller, the window is not known (the door over files), nothing is left of it, or an arm's key comes
        from elsewhere (a selection of its own: the stay is its selection's, S1 and H read it)."""
        if not q.entries or not found or not self.logs.needs_window:
            return None
        window = W.stuck(self.data.get("statuslog", {}), self.args.observed).window
        last = found[-1].last
        if window is None or W.when(last) >= W.when(window[1]):
            return None
        subject = q.subject or self.args.subject
        if not taken_as_handed(getattr(mapdump.fragment(), "key_froms", None) or [], q.owner, subject, {a.line for a in q.arms}):
            return None
        func = q.owner.split("::")[-1]
        n = len(found)
        way = (T.Said("passes-left-by", {"n": n, "exit": f"{func}:{blocked.exit_line}"}) if blocked.exit_line is not None
               else T.Said("passes-ran-no-arm", {"n": n, "left_by": list(blocked.left_by)}) if blocked.left_by
               else T.Said("passes-took-none", {"n": n}))
        self.say(f"{func} runs on the row only as its callers hand it (the keys of its arms' writes are parameters, by the "
                 f"map's KeyFrom), and no pass of it is on the row from {last[:23]} to {window[1]}: its callers' passes "
                 "after it are read, not the guard its last pass left by (I-2)")
        return Callers(q.owner, q.arms, subject, q.key, (last[:19], window[1]), "after", q.entries, after=last[:23],
                       left=T.Said("the-mover-s-last-pass", {"mover": func, "last": last[:23], "way": way}))

    def _callers_of_mover(self, q: Passes, since: str | None, kind: str) -> Callers | None:
        """U1's question where no pass of the mover *q* asks of took the row in the window: its callers' passes there,
        the delay or the stay being in what calls it on the row (KNOWN_ISSUES.md K55).  None where the map lists no
        caller of it, or the window is not known (the door over files)."""
        if not q.entries or since is None or not self.logs.needs_window or q.asking.window is None:
            return None
        end = q.before or q.asking.window[1]
        self.say(f"no pass of {q.owner.split('::')[-1]} took the row from {since} to {end}: its callers' passes there are read (KNOWN_ISSUES.md K55)")
        return Callers(q.owner, q.arms, q.subject or self.args.subject, q.key, (since, end), kind, q.entries, before=q.before)

    def _waker_turn(self, q: Passes, since: str, func: str) -> bool:
        """A2 (out/gm/qk-design.md, design A, tmaeno's OK 2026-11-15 10:42:25): where no pass of the mover *q* took
        the row in the stay (before the one that moved it on, for a late value), and the key every write of its arms
        takes the row by comes only from the rows of selections on another table (the map's KeyFrom sources, ``rows``)
        that the map's ``wakers`` write -- getTasksToExecCommand_JEDI's 4483 by 4339's ``prodsys_comm.comm_task``, a
        command row sendCommandTaskPanda inserts at 4038 -- the mover runs on the row when a waker wrote a row for its
        key.  The wakers' runs of the key are asked over the stay on every machine; a run wrote where a row of it
        prints only on a path through the waker's line (``implying_rows``, H2's own test).  The last write in the stay
        before the move ends the branch (True); none in the stay stops it; anything else -- another source of the key,
        a log not searched whole -- leaves it to U1 (False).  A table another system writes too (the map's inbound
        boundary, DEFT's PRODSYS_COMM) is named as a premise: its writes leave no log here.

        A stuck value's stay is read the same way up to the window's end, the statuslog's as_of: no write in it ends
        the branch where the table is written here alone (A2s: no row the mover takes it by came; MT-paused-stuck's
        command rows, DEFT writing none into PRODSYS_COMM) and stops it where another system writes it too; a write
        in it with the value not moved on is not read here and leaves it to U1 (False)."""
        # one rule for late and stuck: only the stay's end differs, the move or the window's end
        if not self.logs.needs_window:
            return False
        if q.before is not None:
            until = q.before
        elif q.asking.window is not None:
            until = q.asking.window[1]
        else:
            return False
        subject = q.subject or self.args.subject
        fragment = mapdump.fragment()
        cls = subject.partition(".")[0]
        row_tables = {t.name.lower() for t in getattr(fragment, "tables", None) or [] if t.rows_of == cls}
        found = AG.waker_readers(getattr(fragment, "key_froms", None) or [], getattr(fragment, "read_sites", None) or [],
                                 row_tables, q.owner, subject, {a.line for a in q.arms}, KEY_ROWS)
        via: list[str] = []
        if found is None:
            # P45: writes carrying no key of their own take it from the callers, whose rows a selection on another
            # table hands on (the map's crosses_to)
            crossed = AG.crossed_readers(getattr(fragment, "key_froms", None) or [], getattr(fragment, "read_sites", None) or [],
                                         getattr(fragment, "reach_graph", None) or [], row_tables, q.owner, subject,
                                         {a.line for a in q.arms}, KEY_KEPT,
                                         {i.rpartition(".")[2].lower() for i in _identity_attrs(cls)})
            if crossed is None:
                return False
            readers, wakers, via = crossed
            found = (readers, wakers)
        readers, wakers = found
        reader, column = readers[0]
        said_column: T.Said | str = (T.Said("handed-on-by", {"callers": [c.split("::")[-1] for c in via]}) if via else column)
        where = f"{reader.owner.split('::')[-1]}@{reader.anchor.line_start}"
        values = {w.value for w in wakers}
        writes, missed, unread = self._waker_writes(wakers, q.key, since, until, where, values)
        # every waker's logs asked before the branch stops: one record round answers them all
        self._stop_unanswered(missed)
        if unread is not None:
            self.say(unread)
            return False
        start, end = W.when(since), W.when(until)
        # the move to the microsecond where its statuslog row has it, as BP' reads the edges
        moved = next((W.when(s.get("time_extended") or str(s["time"])) for s in self.data.get("statuslog", {}).get("states", [])
                      if q.before is not None and str(s["time"])[:19] == q.before[:19] and s["status"] != self.args.observed), end)
        stay = sorted((w for w in writes if start <= w[0] <= max(end, moved)), key=lambda w: w[0])
        outside = [T.Said("an-outside-writer-of-the-table", {"system": b.system, "table": b.interface})
                   for b in getattr(fragment, "boundaries", None) or []
                   if getattr(b, "direction", "") == "inbound" and any(
                       str(b.interface).lower().rpartition(".")[2] == t.lower() for r, _ in readers for t in r.tables)]
        wakers_said = sorted({f"{w.owner.split('::')[-1].rpartition('.')[2]}:{w.line}" for w in wakers})
        told = T.Said("written-outside-too", {"writers": outside}) if outside else T.Said("written-here-alone")
        if not stay:
            if q.before is None and not outside:
                # A2s (out/gm/qk-design.md "A2s", delegated design; tmaeno's fact through the harness, 2026-11-18: DEFT
                # writes no command into PRODSYS_COMM, the map's DFC): a stuck value whose mover takes the row only by
                # the rows its wakers write, none of which was written for the key in the window, every waker's log
                # searched whole and the table written here alone -- no row it takes came, and the branch ends
                self.end(T.Said("no-row-its-wakers-wrote-in-the-window", {
                    "mover": func, "reader": where, "column": said_column, "wakers": wakers_said, "since": since,
                    "until": until, "premises": [T.Said("the-wakers-write-every-row-it-takes", {"reader": where}),
                                                 *([T.Said("the-key-from-the-callers", {"reader": where, "callers": [
                                                     c.split("::")[-1] for c in via]})] if via else [])]}))
                return True
            if q.before is None:
                self.need(T.Said("a-waker-s-write-in-the-window", {
                    "mover": func, "reader": where, "wakers": wakers_said, "since": since, "until": until, "outside": told}), None, "K55")
            self.need(T.Said("a-waker-s-write-in-the-stay", {
                "mover": func, "reader": where, "wakers": wakers_said, "since": since, "before": q.before,
                "outside": told}), None, "K55")
        at, p, owner, line, printed = stay[-1]
        if q.before is None:
            self.say(f"{owner.split('::')[-1]}:{line} wrote a row {where} selects for the key at {at:%Y-%m-%d %H:%M:%S} "
                     f"and the value has not moved on by {until}: why {func} did not take it is not read here (A2)")
            return False
        routes = {tuple(r) for w in wakers if w.owner == owner and w.line == line and (not printed or w.value in printed)
                  for r in w.routes}
        entries = sorted({r[-1] for r in routes if r})
        # P45's endpoint part: every way in enters from an endpoint the server runs for a request (the map's
        # endpoints): an outside request made the row -- updateWorkers' harvester and pilot ways
        requests = _entry_requests(entries, getattr(fragment, "endpoints", None) or [])
        through = (T.Said("through-requests", {"routes": requests}) if entries and len(requests) == len(entries)
                   else T.Said("through-the-entry", {"entries": entries}) if len(entries) == 1 else T.Said("through-no-one-entry"))
        what = (T.Said("a-row-holding", {"subject": next(w.subject for w in wakers if w.value == printed[0]), "value": printed[0]})
                if len(printed) == 1 else T.Said("a-row-for-it"))
        earlier = (T.Said("earlier-writes-in-the-stay", {"n": len(stay) - 1, "at": [f"{w[0]:%Y-%m-%d %H:%M:%S}" for w in stay[:-1]]})
                   if len(stay) > 1 else T.Said("no-earlier-write-in-the-stay", {"since": since}))
        self.end(T.Said("ran-on-a-row-its-wakers-wrote", {
            "mover": func, "after": f"{(moved - at).total_seconds():.0f}", "waker": owner.split("::")[-1].rpartition(".")[2],
            "line": line, "what": what, "at": f"{at:%Y-%m-%d %H:%M:%S}", "machine": p.machine.split(".")[0],
            "through": through, "reader": where, "column": said_column, "earlier": earlier,
            "premises": [*outside, T.Said("its-first-pass-after-the-write", {"mover": func}),
                         *([T.Said("the-key-from-the-callers", {"reader": where, "callers": [c.split("::")[-1] for c in via]})]
                           if via else [])]}))
        return True

    def _waker_writes(self, wakers: list, key: str, since: str, until: str, where: str,
                      values: set, quiet: bool = False, why: T.Said | None = None) -> tuple[list, list, str | None]:
        """A2's reading of *wakers*' writes for *key* from *since* to *until* (:meth:`_waker_turn`): each waker's logs
        asked by the key over that span on every machine, each whole; a run wrote where a row of it prints only on a
        path through the waker's line (``implying_rows``), the printed value one of *values*.  ``(writes, the asks no
        record answers, why not read)``: each write ``(when, pass, owner, line, values shown)``; the reason a waker's
        runs are not all read, said, with no writes.  *quiet*: each waker's ask not said (P14's many wakers); *why*: the
        asks' reason, where not the stay's."""
        writes: list = []
        asked = []
        for owner in sorted({w.owner for w in wakers}):
            wlines = sorted({w.line for w in wakers if w.owner == owner})
            rel, wfunc = _bare(owner).split("::", 1)
            prints = R._prints_of(owner, wlines)
            files = self.files_of(owner, sorted({f for st in prints.statements for f in st.log_files}) if prints else [], wlines)
            to_hand = self.logs.names(files)
            asking = W.Asking((since, until), None, why or T.Said("over-the-stay-for-its-wakers", {"start": since, "end": until, "reader": where}))
            if not quiet:
                self.say(f"{wfunc}, a waker of {where}: logs {files}, asked {asking.why}")
            runs, missed, holes = self._ask(key, to_hand, asking, quiet=quiet)
            asked.append((owner, wlines, rel, wfunc, to_hand, asking, runs, missed, holes))
        missed_all = [m for a in asked for m in a[7]]
        if missed_all:
            return [], missed_all, None
        for owner, wlines, rel, wfunc, to_hand, asking, runs, _, holes in asked:
            whole = self._searched_whole(to_hand, asking, asking, [], holes, None)
            every = every_log_asked(owner, wlines, to_hand)
            if not whole["held"] or not every["held"]:
                return [], [], (f"{wfunc}'s runs of the key in the stay are not all read ({T.text(whole['why'])}; "
                                f"{T.text(every['why'])}): its writes are not read (A2)")
            fn = R.function(rel, wfunc, wlines)
            for name in to_hand:
                for run in runs.get(name, []):
                    if not of_mover(run, owner):
                        continue
                    for p in passes_of(fn, run, self.mode)[0]:
                        sn = R.claim([R.seen(fn, p, self.mode)])[0]
                        for line in wlines:
                            stmt = R.statement_at(fn.node, line)
                            implying = [r for s in (R.implying_rows(fn.node, stmt) if stmt is not None else [])
                                        if (r := fn.row_at(s)) is not None and sn.proven(r)]
                            if not implying:
                                continue
                            hits = [(i, r) for r in implying for i, _, _ in sn.hits.get(r.line, ())]
                            i, r = min(hits, key=lambda h: h[0])
                            shown = {v.strip().strip("'\"") for v in r.capture(p.records[i]).values()} & values
                            writes.append((W.when(p.records[i].stamp), p, owner, line, sorted(shown)))
        return writes, [], None

    def _hop_turn(self, owner: str, subject: str, since: str, before: str | None, until: str | None, func: str) -> bool:
        """H (out/gm/qk-design.md "H", tmaeno's OK 2026-11-15 13:27:03; KNOWN_ISSUES.md K77): where the mover *owner*
        took the row only at the move (*before*; O1) or no pass of it took the row in the stay (S2), and one of its
        statements selecting the value holds a ``NOT EXISTS`` term on
        other rows that the map structures (:func:`hop.subquery_terms`), the term held once the other objects whose
        transitions write its columns had all moved on (:func:`hop.carriers`): read from the rows file's ``linked``
        rows of such an object, by a column the term correlates on, and their ``statuslogs`` (:func:`hop.read_rows`).
        Late: where every such row had reached a value its carriers record by the move, and the last did inside the
        stay, the stay waited on them -- an END naming the last row and the next question (why it moved on only then),
        True.  Stuck, one rule to the window's end: rows not moved on by then stop the branch, named.  Anything else
        (no such term, no carrier, no linked rows, the term holding from the stay's start, rows not moved on at a late
        value's move) is said and leaves the branch as it was (False)."""
        if not self.logs.needs_window:
            return False
        until = before or until
        if until is None:
            return False
        fragment = mapdump.fragment()
        terms = self._hop_terms(owner, subject)
        if not terms:
            return False
        start, end = W.when(since), W.when(until)
        moved = next((W.when(s.get("time_extended") or str(s["time"])) for s in self.data.get("statuslog", {}).get("states", [])
                      if before is not None and str(s["time"])[:19] == before[:19] and s["status"] != self.args.observed), end)
        linked = self.data.get("linked") or []
        statuslogs = self.data.get("statuslogs") or {}
        for term in terms:
            inner_cls = sorted({s.partition(".")[0] for s in term.inner})
            outers = {o for _, o in term.correlation}
            entries = [e for e in linked if str((e.get("via") or {}).get("from", "")) in outers
                       and str(self._row_value(str(e["via"]["from"]))) in {str(a) for a in e.get("asked") or []}]
            if not entries:
                self.say(f"H: {term.where}'s {term.polarity} term on {', '.join(inner_cls)}: the rows file links no rows of the row "
                         f"by {', '.join(sorted(outers))}: not read")
            for cls in sorted({str(e.get("class", "")) for e in entries}):
                hist = PT.history_of_class(cls)
                log = statuslogs.get(cls) or {}
                if hist is None or not log.get("by_key"):
                    self.say(f"H: {term.where}'s {term.polarity} term on {', '.join(inner_cls)}: the rows file links {cls}'s rows "
                             "but holds no history of them: not read")
                    continue
                car = HOP.carriers(getattr(fragment, "junctions", None) or [], getattr(fragment, "reach_graph", None) or [],
                                   term.tested, hist.subject)
                if car is None:
                    self.say(f"H: {term.where}'s {term.polarity} term on {', '.join(inner_cls)}: no function within {HOP.DEPTH} "
                             f"calls above its columns' writers records {hist.subject}: not read")
                    continue
                via = next(e["via"] for e in entries if e.get("class") == cls)
                key_attr = str(log.get("key", "")).partition(".")[2].lower()
                made: dict[str, datetime.datetime | None] = {}
                own_rows: dict[str, dict] = {}
                for e in entries:
                    if e.get("class") != cls:
                        continue
                    for r in e.get("rows") or []:
                        low = {k.lower(): v for k, v in r.items()}
                        if low.get(key_attr) is None:
                            continue
                        born = low.get(hist.created.lower()) if hist.created else None
                        made[str(low[key_attr])] = W.when(str(born).replace("T", " ")) if born else None
                        own_rows[str(low[key_attr])] = low
                history = {k: HOP.states_of(log["by_key"].get(k)) for k in made}
                reading = HOP.read_rows(history, made, car.values, moved)
                carriers = [f"{o.split('::')[-1]} ({', '.join(sorted(vs))})" for o, vs in car.carriers]
                values = sorted(car.values)
                # P19: column by column, the carriers' values under which the writers' accumulations move it, over the
                # linked rows (a fine-grained job's 'fg_done' moves nFilesFinished whatever its status); said where fewer
                by_column = []
                for col in sorted(term.tested):
                    col_moving = frozenset().union(*(self._carried_values(fragment, car, {col}, hist.subject, own_rows.get(k))[0]
                                                     for k in made)) if made else frozenset(car.values)
                    if col_moving < frozenset(car.values):
                        by_column.append(f"{col.partition('.')[2]} by {', '.join(sorted(col_moving)) or 'none of them'}, not by "
                                         f"{', '.join(sorted(frozenset(car.values) - col_moving))}")
                said_via = f"{via['from']} -> {via['to']}"
                premises = [T.Said("h-carried-by", {"inner": sorted(term.tested), "writers": [w.split("::")[-1] for w in car.writers],
                                                    "carriers": carriers, "status": hist.subject}),
                            *([T.Said("h-moved-by-column", {"columns": by_column})] if by_column else []),
                            T.Said("h-rows-stand-for-the-term-s", {"cls": cls, "via": said_via, "inner": ", ".join(inner_cls)})]
                if before is None:
                    if not reading.open:
                        self.say(f"H: {term.where}'s {term.polarity} term held by {until} ({len(reading.reached)} {cls} row(s) by "
                                 f"{said_via} in {values}) and the value has not moved on: why {func} did not take it is not read here")
                        return False
                    shown = [f"{k} {v or 'no row'}" for k, v in reading.open[:6]] + (["..."] if len(reading.open) > 6 else [])
                    first = reading.open[0][0]
                    nxt = HOP.current_stay(history.get(first, []), end)
                    self.need(T.Said("h-rows-it-waits-on", {
                        "where": term.where, "polarity": term.polarity, "inner": ", ".join(inner_cls), "until": until,
                        "n": len(reading.open), "total": len(made) - reading.made_later, "cls": cls, "via": said_via, "values": values,
                        "open": shown,
                        "next": T.Said("h-next-stuck", {"status": hist.subject, "cls": cls, "key": first, "value": nxt[0], "since": nxt[1][:23]})
                        if nxt else T.Said("h-next-none", {"cls": cls, "key": first}),
                        "premises": premises}), None, "K77")
                if reading.open:
                    shown = [f"{k} {v or 'no row'}" for k, v in reading.open[:6]]
                    self.say(f"H: {term.where}'s {term.polarity} term: {len(reading.open)} {cls} row(s) by {said_via} not in {values} at "
                             f"the move ({', '.join(shown)}): the term is not read as having held then")
                    continue
                turn = reading.turn
                if turn is None:
                    continue
                at, key, value, host = turn
                if at <= start:
                    self.say(f"H: {term.where}'s {term.polarity} term held from the stay's start: the last {cls} row by {said_via} "
                             f"reached {value} at {at:%Y-%m-%d %H:%M:%S} ({key}), before {since}")
                    continue
                stay = HOP.longest_stay(history.get(key, []), start, at)
                born = made.get(key)
                delta = (moved - at).total_seconds()
                premises.append(T.Said("h-a-later-write-not-read", {"delta": f"{delta:.1f}"}))
                self.end(T.Said("h-ran-after-other-rows-moved-on", {
                    "mover": func, "delta": f"{delta:.1f}", "where": term.where, "polarity": term.polarity,
                    "inner": ", ".join(inner_cls), "at": f"{at:%Y-%m-%d %H:%M:%S.%f}"[:23],
                    "n": len(reading.reached), "cls": cls, "via": said_via, "values": values, "key": key, "value": value,
                    "machine": host.split(".")[0] or "an unnamed host", "since": since[:19],
                    "later": T.Said("h-made-later", {"n": reading.made_later, "cls": cls}) if reading.made_later else "",
                    "next": T.Said("h-next-late", {"status": hist.subject, "cls": cls, "key": key, "held": stay[0], "since": stay[1],
                                                   "until": stay[2],
                                                   "made": T.Said("h-made-at", {"at": f"{born:%Y-%m-%d %H:%M:%S}"})
                                                   if born is not None and born > start else ""})
                    if stay else T.Said("h-next-none", {"cls": cls, "key": key}),
                    "premises": premises}))
                if born is not None and born > start:
                    # P14's first form: the row it waited on, made inside the stay -- its birth and its life
                    for said in self._waited_birth(cls, key, born, history.get(key, []), str(self._row_value(str(via["from"]))),
                                                   str(via["from"]), start):
                        self.say(f"  {T.text(said)}")
                return True
        return False

    def _waited_birth(self, cls: str, key: str, born: datetime.datetime, states: list[dict], link: str,
                      link_column: str = "", start: datetime.datetime | None = None) -> list[T.Said]:
        """P14, its first form (out/gm/qk-design.md, the section on P14's first form, delegated design; the
        investigator's u11 NT-scouting-late 4): a row O1's END says the stay waited on, made inside the stay (the
        merge job JobGenerator made at 17:45:57), read for its birth and its life.  Its birth: the runs of the map's
        INSERTs making a row of its class (``KeyFrom.new_row``: insertNewJob) holding its key, over its creation less
        and plus EDGE on every machine, each whole -- the first record's time and machine.  They are asked by the value
        that links the row to the asked one (*link*, the task's ID) first, as the maker's mark holds it (insertNewJob's
        ``<JediTaskID=... idPool=...>``: asked by the writer's mark, the records the walk already holds answer), then
        by the row's own key where that finds none.  Its life: its own
        history rows (*states*), each value with its time and machine.  What is said; an ask the records lack kept.
        Why its maker made it then: :meth:`_waited_trigger`, where *link_column* and the stay's *start* are given."""
        fragment = mapdump.fragment()
        tables = {str(t.name).lower() for t in getattr(fragment, "tables", None) or [] if getattr(t, "rows_of", "") == cls}
        makers: dict[str, list[int]] = {}
        for k in key_froms():
            if getattr(k, "new_row", False) and str(getattr(k, "table", "")).lower().rpartition(".")[2] in tables:
                makers.setdefault(_bare(k.owner), []).append(int(getattr(k, "new_row_line", 0) or 0))
        out: list[T.Said] = []
        if makers:
            files = list(dict.fromkeys(f for o, lines in sorted(makers.items()) for f in self.files_of(o, [], lines)))
            lo, hi = W._down(born - EDGE), W._down(born + EDGE)
            names = [o.rpartition("::")[2] for o in sorted(makers)]
            ask = W.Asking((lo, hi), None, T.Said("for-the-waited-rows-birth", {"cls": cls, "key": key, "makers": names,
                                                                                "at": f"{born:%Y-%m-%d %H:%M:%S}"}))
            runs: list = []
            unanswered: list = []
            for asked_by in dict.fromkeys(v for v in (link, key) if v and v != "None"):
                logs, unanswered = self._logs_at(asked_by, files, ask, owners_of(sorted(makers)))
                runs = sorted((r for log in logs.values() for r in runs_for(log, key, self._other_of(key, log))
                               if any(of_mover(r, o) for o in makers) and r.last[:19] >= lo and r.first[:19] <= hi),
                              key=lambda r: r.first) if not unanswered else []
                if runs:
                    break
            if unanswered and not runs:
                self._keep_unanswered(unanswered)
                out.append(T.Said("waited-birth-asked", {"cls": cls, "key": key, "makers": names, "n": len({a.key for a, _ in unanswered})}))
            else:
                if runs:
                    r = runs[0]
                    out.append(T.Said("waited-birth-run", {"cls": cls, "key": key, "maker": next(
                        o.rpartition("::")[2] for o in sorted(makers) if of_mover(r, o)), "first": r.first[:23],
                        "machine": (r.machine or "").split(".")[0] or "an unnamed host"}))
                else:
                    out.append(T.Said("waited-birth-unread", {"cls": cls, "key": key, "makers": names}))
        if makers and link_column and start is not None:
            out += self._waited_trigger(cls, [k for k in key_froms() if getattr(k, "new_row", False) and _bare(k.owner) in makers],
                                        link, link_column, start, born)
        life = [f"{s.get('status')} {str(s.get('time_extended') or s.get('time'))[:23]}"
                + (f" ({str(s.get('modificationhost') or s.get('host')).split('.')[0]})" if s.get("modificationhost") or s.get("host") else "")
                for s in sorted(states, key=lambda s: str(s.get("time_extended") or s.get("time")))]
        if life:
            out.append(T.Said("waited-life", {"cls": cls, "key": key, "life": life}))
        return out

    def _waited_trigger(self, cls: str, makers: list, link: str, link_column: str, start: datetime.datetime,
                        born: datetime.datetime) -> list[T.Said]:
        """P14's first form, its second step: why the row's maker made it then.  The maker's functions: the one holding
        each of *makers*' ``new_row_at`` (TaskBuffer.storeJobs) and those reaching it within two calls
        (JobGeneratorThread.runImpl).  Its selections: the map's read sites with wakers whose rows the map's
        ``crosses_to`` hands to a maker's function and whose keys hold *link_column* (JobGenerator's input selection
        ``_get_tasks_datasets_with_unprocessed_inputs@1872``, keyed by the task).  Their wakers' writes for *link* over
        the birth less EDGE (from the stay's *start* where later) are read as A2 reads them (:meth:`_waker_writes`):
        the last before the birth is said (updateUnmergedDatasets' 'ready' at 17:44:56 on aipanda099).  EDGE, not the
        stay: a maker that polls takes a row a waker wrote within its round, and a stay a day long asked whole by the
        wakers' marks is 32 asks (NT-scouting-late).  An ask the records lack kept; nothing said where no selection is
        found."""
        functions: set[str] = set()
        for k in makers:
            rel, _, line = str(getattr(k, "new_row_at", "") or "").rpartition(":")
            node = R.def_holding(rel, int(line)) if rel and line.isdigit() else None
            if node is None:
                continue
            owner = R.owner_of(rel, node)
            record = MR.graph()
            for named in (MR.owners(owner, record) if record is not None else []) or [owner]:
                functions |= {_plain(named), *(_plain(o) for o in MR.reaching(named, 2))}
        if not functions:
            return []
        readers = [rs for rs in getattr(mapdump.fragment(), "read_sites", None) or []
                   if getattr(rs, "wakers", None) and link_column in (getattr(rs, "keys", None) or [])
                   and any(_plain(str(c)) in functions for c in getattr(rs, "crosses_to", None) or [])]
        if not readers:
            return []
        wakers = [w for rs in readers for w in rs.wakers]
        where = ", ".join(f"{str(rs.owner).split('::')[-1]}@{rs.anchor.line_start}" for rs in readers)
        since, until = f"{max(start, born - EDGE):%Y-%m-%d %H:%M:%S}", f"{born:%Y-%m-%d %H:%M:%S}"
        writes, missed, unread = self._waker_writes(wakers, link, since, until, where, {w.value for w in wakers}, quiet=True,
                                                    why=T.Said("before-the-birth-for-its-wakers", {
                                                        "cls": cls, "start": since, "end": until, "reader": where}))
        if missed:
            self._keep_unanswered(missed)
            return [T.Said("waited-trigger-asked", {"where": where, "n": len({a.key for a, _ in missed}), "since": since, "until": until})]
        if unread is not None:
            return [T.Said("waited-trigger-unread", {"where": where, "why": unread})]
        before = sorted((w for w in writes if w[0] <= born), key=lambda w: w[0])
        if not before:
            return [T.Said("waited-trigger-none", {"where": where, "since": since, "until": until})]
        at, p, owner, line, shown = before[-1]
        return [T.Said("waited-trigger", {"where": where, "waker": owner.split("::")[-1].rpartition(".")[2], "line": line,
                                          "values": shown, "at": f"{at:%Y-%m-%d %H:%M:%S}", "machine": p.machine.split(".")[0],
                                          "cls": cls})]

    def _count_turn(self, owner: str, subject: str, since: str, before: str | None, func: str) -> bool:
        """P25 (i) (the investigator's u3/u10; the map's AGC, codemap-xliv bbc723f, and ``Branch.condition_atoms``):
        where the mover *owner* took the row at the move (*before*) by a selection whose loop printed it
        (:meth:`_took_by`), and a guard in that loop testing a count the selection groups (a guard source's
        ``RowAggregate``, ``COUNT`` of a column holding a value) holds at the counts the take printed and fails at one
        fewer, the count turned when the k-th of the rows it counts got the value.  Those rows are read on the rows
        file's ``linked`` rows of a class with a history, linked by the row's own class, each at the first status under
        which an arm writing the counted value may write it (:meth:`_counted_states`); the rows that got it by the move
        must be the printed k, else nothing is said.  MT-scouting-late: sqlEA printed 5/10 'for early avalanche', its
        3759 (``totFinished >= totJobs * minSuccessScouts / 10``, 5 at 3707) holds at 5 and not at 4, and the 5th scout
        job finished at 12:45:15.  True with an END; False (said where it reads) else."""
        took = getattr(self, "_took", None)
        if (before is None or took is None or took.record is None or took.row is None
                or _bare(str(getattr(took.statement, "owner", ""))) != _bare(owner)):
            return False
        fragment = mapdump.fragment()
        printed = self._printed_counts(took)
        turned = []
        for (line, test), names in sorted(self._count_guards(fragment, owner, took.statement.line).items()):
            counted = [(name, agg) for name, srcs in names.items() for src in srcs
                       if (agg := getattr(src, "aggregate", None)) is not None and getattr(agg, "fn", "") == "COUNT"
                       and getattr(agg, "value", "") and isinstance(printed.get(name), int)]
            if len({name for name, _ in counted}) != 1:
                continue
            name, agg = counted[0]
            k = printed[name]
            if (self._count_guard_value(test, names, printed, subject, {}) is True
                    and self._count_guard_value(test, names, printed, subject, {name: k - 1}) is False):
                turned.append((line, test, name, agg, k))
        if len(turned) != 1:
            return False
        line, test, name, agg, k = turned[0]
        where = f"{func}@{took.statement.line}"
        counted_attr = self._class_attr(fragment, str(agg.column))
        try:
            value = str(ast.literal_eval(str(agg.value)))
        except (ValueError, SyntaxError):
            value = str(agg.value).strip("'\"")
        if not counted_attr:
            self.say(f"H: {where}'s count at {line} ({test}) turned at the take's {name}={k}, but the map places "
                     f"{agg.column} in no class: when is not read")
            return False
        cls = subject.partition(".")[0]
        start = W.when(since)
        moved = next((W.when(s.get("time_extended") or str(s["time"])) for s in self.data.get("statuslog", {}).get("states", [])
                      if str(s["time"])[:19] == before[:19] and s["status"] != self.args.observed), W.when(before))
        statuslogs = self.data.get("statuslogs") or {}
        entries = [e for e in self.data.get("linked") or []
                   if str((e.get("via") or {}).get("from", "")).partition(".")[0] == cls
                   and str(self._row_value(str(e["via"]["from"]))) in {str(a) for a in e.get("asked") or []}]
        for other in sorted({str(e.get("class", "")) for e in entries}):
            hist = PT.history_of_class(other)
            log = statuslogs.get(other) or {}
            if hist is None or not log.get("by_key"):
                continue
            car = HOP.carriers(getattr(fragment, "junctions", None) or [], getattr(fragment, "reach_graph", None) or [],
                              {counted_attr}, hist.subject)
            if car is None:
                continue
            key_attr = str(log.get("key", "")).partition(".")[2].lower()
            via = next(e["via"] for e in entries if e.get("class") == other)
            reached: list[tuple[datetime.datetime, str, str, str, list[str]]] = []
            total = later = 0
            for e in entries:
                if e.get("class") != other:
                    continue
                for row in e.get("rows") or []:
                    low = {c.lower(): v for c, v in row.items()}
                    if low.get(key_attr) is None:
                        continue
                    born = low.get(hist.created.lower()) if hist.created else None
                    if born and W.when(str(born).replace("T", " ")) > moved:
                        # made after the move: in no count the take read (the avalanche's jobs)
                        later += 1
                        continue
                    total += 1
                    values, open_ = self._counted_states(fragment, car, counted_attr, value, hist.subject, low)
                    for st in sorted(HOP.states_of(log["by_key"].get(str(low[key_attr]))), key=lambda x: str(x.get("time"))):
                        if st.get("status") in values:
                            at = W.when(str(st.get("time_extended") or st["time"]))
                            if at <= moved:
                                reached.append((at, str(low[key_attr]), str(st["status"]), str(st.get("host") or st.get("modificationhost") or ""), open_))
                            break
            reached.sort()
            said_via = f"{via['from']} -> {via['to']}"
            if len(reached) != k:
                self.say(f"H: {where}'s count at {line} ({test}) turned at the take's {name}={k}; {len(reached)} of the row's "
                         f"{total} {other} row(s) by {said_via} got a status under which {counted_attr}={value!r} is written by "
                         f"the move: the count is not read from them")
                return False
            at, key, status, host, open_ = reached[k - 1]
            if at <= start:
                self.say(f"H: {where}'s count at {line} ({test}) reached {k} before the stay began ({key} at {at:%Y-%m-%d %H:%M:%S}): "
                         "why it took the row only then is not read here")
                return False
            delta = (moved - at).total_seconds()
            premises = [T.Said("h-carried-by", {"inner": [counted_attr], "writers": [w.split("::")[-1] for w in car.writers],
                                                "carriers": [f"{o.split('::')[-1]} ({', '.join(sorted(vs))})" for o, vs in car.carriers],
                                                "status": hist.subject}),
                        T.Said("h-rows-stand-for-the-count", {"cls": other, "via": said_via, "counted": counted_attr})]
            if open_:
                premises.append(T.Said("h-count-ways-not-read", {"atoms": open_}))
            self.end(T.Said("h-count-turned", {
                "mover": func, "delta": f"{delta:.1f}", "where": where, "line": line, "test": test, "name": name, "k": k,
                "less": k - 1, "printed": took.text, "n": total, "cls": other, "via": said_via, "counted": counted_attr,
                "value": value, "key": key, "status": status, "at": f"{at:%Y-%m-%d %H:%M:%S.%f}"[:23],
                "machine": host.split(".")[0] or "an unnamed host",
                "later": T.Said("h-made-later", {"n": later, "cls": other}) if later else "", "premises": premises}))
            return True
        self.say(f"H: {where}'s count at {line} ({test}) turned at the take's {name}={k}; the rows file links no rows with a "
                 f"history whose moves write {counted_attr}: when is not read")
        return False

    def _printed_counts(self, took: TookBy) -> dict[str, Any]:
        """P25 (i): what the taking print's holes printed in the moving pass, by expression; digits as ints."""
        out: dict[str, Any] = {}
        for m in getattr(took.row, "map_rows", None) or []:
            for h in getattr(m, "holes", None) or []:
                if h.expression in out:
                    continue
                got = hole_values(took.row, took.record, SimpleNamespace(part=MUST_PART_MESSAGE, expression=h.expression, bind=""))
                if got:
                    out[h.expression] = int(got[0]) if got[0].lstrip("-").isdigit() else got[0]
        return out

    @staticmethod
    def _count_guards(fragment, owner: str, line: int) -> dict[tuple[int, str], dict[str, list]]:
        """P25 (i): the guards in the loops over the rows of *owner*'s selection at *line* (the read site's
        ``row_loops``), by (test line, test), with each name's sources (``guard_froms``)."""
        site = next((rs for rs in getattr(fragment, "read_sites", None) or []
                     if _plain(str(rs.owner)) == _bare(owner) and getattr(getattr(rs, "anchor", None), "line_start", None) == line), None)
        loops = [tuple(getattr(lp, "lines", None) or ()) for lp in getattr(site, "row_loops", None) or []] if site is not None else []
        out: dict[tuple[int, str], dict[str, list]] = {}
        for g in getattr(fragment, "guard_froms", None) or []:
            at = int(getattr(g, "test_line", 0) or 0)
            if _bare(str(g.owner)) != _bare(owner) or not any(len(ls) == 2 and ls[0] <= at <= ls[1] for ls in loops):
                continue
            out.setdefault((at, str(g.test)), {})[str(g.name)] = list(getattr(g, "sources", None) or [])
        return out

    def _count_guard_value(self, test: str, names: dict[str, list], printed: dict[str, Any], subject: str,
                           override: dict[str, Any]) -> Any:
        """P25 (i): *test* evaluated (the map's evaluator) with its names bound as the take printed them (or
        *override*), a constant source's literal, and a source of the row's own status column its value in the stay."""
        env = E.shared.Env()
        cls, _, attr = subject.partition(".")
        tables = {str(t.name).lower() for t in getattr(mapdump.fragment(), "tables", None) or [] if getattr(t, "rows_of", "") == cls}
        for name, srcs in names.items():
            if name in override:
                env.put(name, override[name], "one fewer")
            elif name in printed:
                env.put(name, printed[name], "printed by the take")
            elif len(srcs) == 1 and srcs[0].kind == KEY_CONSTANT and not getattr(srcs[0], "changes", None):
                try:
                    env.put(name, ast.literal_eval(str(srcs[0].name)), "a constant")
                except (ValueError, SyntaxError):
                    continue
            elif srcs and all(getattr(x, "kind", "") == "rows" and str(getattr(x, "column", "")).lower()
                              in {f"{t}.{attr.lower()}" for t in tables} for x in srcs):
                env.put(name, self.args.observed, "the row's value in the stay")
        return E.shared.evaluate(test, env)

    @staticmethod
    def _class_attr(fragment, column: str) -> str:
        """``Class.attr`` of a ``table.column`` (a guard source's spelling): the class the table's rows are of, and the
        attribute of it some junction writes under that column's name; "" where either is not known."""
        table, _, col = column.lower().partition(".")
        classes = {str(t.rows_of) for t in getattr(fragment, "tables", None) or []
                   if str(t.name).lower() == table and bool(getattr(t, "rows_of", None))}
        attrs = {str(j.subject) for j in getattr(fragment, "junctions", None) or []
                 if str(j.subject).partition(".")[0] in classes and str(j.subject).partition(".")[2].lower() == col}
        return next(iter(attrs)) if len(attrs) == 1 else ""

    @staticmethod
    def _carried_values(fragment, car, tested: set[str], status: str,
                        row: dict | None = None) -> tuple[frozenset[str], list[str]]:
        """P19 (the investigator's u11 NT-scouting-late 3, MT-scouting-stuck 1; the map's ACC, codemap-xliv 2616f94): of
        the values of *status* the carriers record (``car.values``), those under which a writer's accumulation into a
        *tested* column may move the count -- an increment (op ``+``) of a counter write (``Branch.accumulated``), its
        written atoms (``ConditionAtom.written``: propagateResultToJEDI's ``fileStatus``, JediFileSpec.status) set to
        each value the same owner's junction of that subject writes while *status* holds the candidate, its path read
        with :func:`_entries_possible`.  A value that junction writes at run time is the candidate where it is *status*
        itself (4006's ``jobSpec.jobStatus``), else left open to the two paths read together (4008's ``fileSpec.status``
        under ``== 'nooutput'``, which the accumulation's ``not in ('nooutput')`` refuses).  *row*: a linked row's columns (lower case), its
        own atoms read from it as :meth:`_counted_states` reads them (a fine-grained job's jobSubStatus 'fg_done' makes
        its files 'finished' whatever its status).  ``(moving, left out)``; every value, none left out, where no counter
        write carries accumulations.  A merging job's input files are written 'ready' (3998), so
        'merging' moves no nFilesFinished, which counts 'finished' and 'lost'."""
        values = frozenset(str(v) for v in car.values)
        writers = {_bare(str(w)) for w in car.writers}
        junctions = list(getattr(fragment, "junctions", None) or [])
        counters = [(j, b) for j in junctions if str(getattr(j, "subject", "")) in tested and _bare(str(getattr(j, "owner", ""))) in writers
                    for b in getattr(j, "branches", None) or [] if getattr(b, "accumulated", None)]
        if not counters:
            return values, []
        attr = status.partition(".")[2]
        moving: set[str] = set()
        for v in sorted(values):
            for j, b in counters:
                if v in moving:
                    break
                for acc in b.accumulated:
                    if str(getattr(acc, "op", "")) != "+":
                        continue
                    written = [a for a in getattr(acc, "condition_atoms", None) or [] if getattr(a, "written", False)]
                    if not written:
                        moving.add(v)
                        break
                    subject = str(written[0].subject)
                    writes = [fb for fj in junctions if str(getattr(fj, "subject", "")) == subject
                              and _bare(str(getattr(fj, "owner", ""))) == _bare(str(j.owner)) for fb in getattr(fj, "branches", None) or []]
                    if not writes:
                        moving.add(v)
                        break
                    for fb in writes:
                        if not _entries_possible(list(getattr(fb, "path_condition", None) or []),
                                                 list(getattr(fb, "condition_atoms", None) or []), {status: v}, row):
                            continue
                        outcome = str(getattr(fb, "outcome", "") or "").strip("'\"")
                        fixed = {status: v, subject: outcome}
                        if outcome.startswith(SV.RUN_TIME):
                            # the candidate itself where it writes the status (4006); else what it writes is left to
                            # the two paths together (4008's fileSpec.status, 'nooutput' there, which no accumulation takes)
                            inner = outcome.partition("(")[2].rstrip(")")
                            fixed = {status: v, subject: v} if inner.rpartition(".")[2] == attr else {status: v}
                        entries = [*(getattr(fb, "path_condition", None) or []), *(getattr(acc, "path_condition", None) or [])]
                        atoms = [*(getattr(fb, "condition_atoms", None) or []), *(getattr(acc, "condition_atoms", None) or [])]
                        if _entries_possible(entries, atoms, fixed, row):
                            moving.add(v)
                            break
                    if v in moving:
                        break
        if not moving:
            return values, []
        return frozenset(moving), sorted(values - moving)

    @staticmethod
    def _counted_states(fragment, car, counted_attr: str, value: str, status: str, row: dict[str, Any]) -> tuple[frozenset[str], list[str]]:
        """P25 (i): the values of *status* (the linked row's history) under which an arm of the carriers' writers
        writing *counted_attr* = *value* may write it, for the linked *row* (its columns, lowercased): each candidate
        (the carriers' recorded values) set in the arm's path condition's atoms on *status*, the atoms on the row's
        other columns from the row, the rest unknown (the map's ``Branch.condition_atoms`` say which atom is which;
        none read by name); a candidate goes where an entry of the condition is false.  With the atoms left open on
        the ways kept (said as premises)."""
        cls = status.partition(".")[0]
        branches = [(j, b) for j in getattr(fragment, "junctions", None) or [] if str(j.subject) == counted_attr
                    and _bare(str(j.owner)) in {_bare(w) for w in car.writers}
                    for b in j.branches if str(getattr(b, "outcome", "")).strip("'\"") == value]
        kept: set[str] = set()
        open_: list[str] = []
        for _, b in branches:
            atoms = list(getattr(b, "condition_atoms", None) or [])
            if not atoms:
                continue
            # an entry read elsewhere ([at N], a later write's test; [in f()], a callee's) is no condition of this write
            entries = []
            for e in getattr(b, "path_condition", None) or []:
                if E.shared.read_elsewhere(e):
                    continue
                try:
                    entries.append(ast.parse(E.shared.strip_annotation(e), mode="eval").body)
                except SyntaxError:
                    continue
            for v in sorted(car.values):
                fixed: dict[str, bool] = {}
                unknown: list[str] = []
                for a in atoms:
                    text = _atom_text(str(a.text))
                    vals = {str(x).strip("'\"") for x in getattr(a, "values", None) or []}
                    subj = str(getattr(a, "subject", "") or "")
                    op = str(getattr(a, "op", ""))
                    col = subj.partition(".")[2].lower()
                    # the row's own column, where the rows file holds it: NULL is a value it holds (None), not unknown
                    own = subj.partition(".")[0] == cls and subj != status and col in row
                    if (subj != status and not own) or op not in ("==", "!=", "in", "not in"):
                        unknown.append(str(a.text))
                        continue
                    held = v if subj == status else row[col]
                    inside = held is not None and str(held) in vals
                    fixed[text] = inside if op in ("==", "in") else not inside
                verdicts = [E.shared.truth(e, fixed, E.shared.Env()) for e in entries]
                if any(x is False for x in verdicts):
                    continue
                kept.add(v)
                open_ += [u for u in unknown if u not in open_]
        return frozenset(kept), open_

    def _stay_edges(self, fn: R.Function, owner: str, arms: list[int], subject: str, key: str, found: list,
                    since: str, whole: frozenset[int] | set[int] = frozenset()) -> tuple[list, list]:
        """BP' (analyzer handover qk-design.md "BP'"): of *found*, a mover's passes (or a caller's) in the window from
        *since*, the ones of the value's time, and the ones that moved it on.  The time opens at the statuslog row of
        the value *since* names (the one --since names, else the last of that second) and ends at the next row, the
        move; each compared to the microsecond where the row has its time_extended: a pass over before the row, or
        begun after the move, ran outside it -- MT-scouted-late's setScoutJobData_JEDI, over 0.08 s before 'scouted',
        and MT-exhausted-late's runImpl, begun 4 ms after the move, were kept by the second.

        A pass whose calls recorded the row at an edge is the pass the value moved in, or moved on in, not one of its
        time: read on the run of the method that recorded the row, inside a call of a pass of the function the map's
        history pair names after one of *arms* (K67's :meth:`_recorder_read`), the pass being that one or inside it --
        MT-passed-late's TaskCommando pass, whose call of updateTask_JEDI at 201 recorded 'passed' (the mover's own log
        alone did not show it).  A row writing the value again over the same value (activateJob's 'defined' over
        'defined', MJ-defined-late) is no edge: its pass is read.  Where the records do not show a row's recorder the
        pass is read as before, the asks kept.  A run its function's entry row did not cut (*whole*, ids) may hold
        several executions -- JobGenerator's ``<jediTaskID=... datasetID=...>`` on aipanda088 from 08:42 to 14:46,
        MT-scouting-stuck -- and is not decided here: read as before, and said (the analyzer handover's qk-design.md
        "BP'", fix 2 (ii)).  ``(kept, the passes that recorded the move)``; where *since* names no row of the value,
        the passes ending in or after its second, as before."""
        states = self.data.get("statuslog", {}).get("states", [])
        observed = self.args.observed
        rows = self._stay_rows(since)
        if rows is None:
            return [p for p in found if p.last[:19] >= since], []
        n, row, nxt = rows
        opens = row.get("time_extended") or str(row["time"])
        off, said = self._off_the_stay([(p.first, p.last) for p in found], since)
        for words in said:
            self.say(words)
        kept = [p for i, p in enumerate(found) if i not in off]
        moved_in = n == 0 or states[n - 1]["status"] != observed
        if not moved_in:
            self.say(f"the row of {observed!r} at {opens} writes it again over the row before: no edge, the pass "
                     "that wrote it is read")
        moving: list = []
        for kind, edge in [*([("in", row)] if moved_in else []), *([("on", nxt)] if nxt is not None else [])]:
            edge_at = W.when(edge.get("time_extended") or str(edge["time"]))
            if uncut := [p for p in kept if id(p) in whole and W.when(p.first) <= edge_at <= W.when(p.last)]:
                self.say(f"{len(uncut)} run(s) not cut into passes hold the row of {edge['status']!r} at "
                         f"{edge.get('time_extended') or edge['time']}: whether one holds the pass the value moved "
                         f"{'in' if kind == 'in' else 'on in'} is not decided (KNOWN_ISSUES.md K76)")
            got = self._recording_pass(owner, arms, subject, key, edge, [p for p in kept if id(p) not in whole],
                                       entered=kind == "in")
            if got is None:
                continue
            p, said = got
            kept = [x for x in kept if x is not p]
            moving += [p] if kind == "on" else []
            what = "moved in" if kind == "in" else "moved on in"
            self.say(f"pass {p.first} .. {p.last} recorded the row of {edge['status']!r} at "
                     f"{edge.get('time_extended') or edge['time']} ({said}): the pass the value {what}, "
                     "not one of its time")
        return kept, moving

    def _stay_rows(self, since: str) -> tuple[int, dict, dict | None] | None:
        """The stay's rows as BP' reads them (:meth:`_stay_edges`): ``(index, the statuslog row of the observed value
        *since* names -- the one --since names, else the last of that second --, the next row, the move, or None)``;
        None where no row of the value is at *since*'s second."""
        states = self.data.get("statuslog", {}).get("states", [])
        observed = self.args.observed
        at = [n for n, st in enumerate(states) if st["status"] == observed and str(st["time"])[:19] == since[:19]]
        if not at:
            return None
        exact = [n for n in at if (states[n].get("time_extended") or "") == (getattr(self.args, "since", None) or "-")]
        n = (exact or at)[-1]
        return n, states[n], (states[n + 1] if n + 1 < len(states) else None)

    def _off_the_stay(self, spans: list[tuple[str, str]], since: str) -> tuple[set[int], list[str]]:
        """Of *spans* (``(first, last)`` record times), the indexes BP' reads as off the stay from *since*, and what it
        says of them -- the edges of :meth:`_stay_edges`, and of the caller's passes :meth:`_call_tied` reads by its
        callees' runs: over before the row of the value, to the microsecond where the row has its time_extended (those in
        its second said), else before its second; or begun after the row of the move, where that row has its
        time_extended.  Nothing where no row of the value is at *since*'s second."""
        rows = self._stay_rows(since)
        if rows is None:
            return set(), []
        _, row, nxt = rows
        said = []
        if opens := row.get("time_extended"):
            off = {i for i, (_, last) in enumerate(spans) if W.when(last) < W.when(opens)}
            if sneaked := [i for i in off if spans[i][1][:19] >= since[:19]]:
                said.append(f"{len(sneaked)} pass(es) over before the row of {row['status']!r} at {opens}, in its second: not read")
        else:
            off = {i for i, (_, last) in enumerate(spans) if last[:19] < since[:19]}
        if nxt is not None and (closes := nxt.get("time_extended")):
            if after := {i for i, (first, _) in enumerate(spans) if i not in off and W.when(first) > W.when(closes)}:
                off |= after
                said.append(f"{len(after)} pass(es) begun after the row of {nxt['status']!r} at {closes}, the move: not read")
        return off, said

    def _recording_pass(self, owner: str, arms: list[int], subject: str, key: str, row: dict, found: list,
                        entered: bool = False):
        """``(the pass of *found*, what says so)`` whose calls recorded *row*, as :meth:`_stay_edges` reads it, or None
        where the records do not show one.  The run of the method recording the row, cut to its call there, is placed
        inside a call of the pass (``recorded_in_pass``), the pass being that one or inside it; where the pass's prints
        cannot bracket the call -- TaskCommando's "done with" after updateTask_JEDI is printed by three rows, 202, 217
        and 232 -- inside the pass's own first and last records on its machine, the map's history pair saying that
        call records the row after one of *arms* (KNOWN_ISSUES.md K76).

        A1 (*entered*: the row that began the value's time): where none of *arms* recorded it, another writer's arm
        that may write the value did, inside the pass, in a function the map's calls reach from *owner*'s
        (:meth:`_entering_recorders`) -- MT-exhausted-late's runImpl pass at 09:33:01, inside which retryTask_JEDI,
        called on the retry command, recorded 'exhausted' (out/gm/qk-design.md, design A, tmaeno's OK 2026-11-15
        10:42:25)."""
        at = W.when(row.get("time_extended") or str(row["time"]))
        # only a pass running at the row's time can hold the call that recorded it: none, and nothing is asked
        found = [p for p in found if W.when(p.first) <= at <= W.when(p.last)]
        if not row.get("host") or not self.logs.needs_window or not found:
            return None
        recs = list(dict.fromkeys(r for line in arms for r in recorders(subject, owner, line)))
        asking = W.at_transition(row)
        # the pairs whose write is the mover's own call: each placed in its call first, and only where none is, inside
        # the pass -- retryTask_JEDI records 'exhausted' at 5353 and at 5365, in two branches, and 5365's ran
        own = [(rec, callee) for rec in recs if (callee := self._own_call(rec)) is not None]
        for placed in (True, False):
            for rec, callee in own:
                if (got := self._own_call_pass(rec, callee, key, row, asking, found, at, placed)) is not None:
                    return got
        for rec in recs:
            if any(rec is r for r, _ in own):
                continue
            kept = len(self.unanswered)
            kind, said, part, inner = self._recorder_read(rec, subject, key, row, asking)
            if kind == "pass" and part is not None and said is not None:
                for p in found:
                    if (p.log.name == part.log.name and p.machine == part.machine and _during(inner, p, at)
                            and W.when(part.first) <= W.when(p.first) and W.when(p.last) <= W.when(part.last)):
                        return p, T.text(said)
            elif kind == "" and inner:
                for p in found:
                    if _during(inner, p, at):
                        # placed without the caller's pass: the asks for it are not read, and not kept
                        del self.unanswered[kept:]
                        names = sorted({m.token for m in inner})
                        return p, f"{', '.join(names)} recorded it inside the pass, {rec.caller.split('::')[-1]}:{rec.call} recording it after the arm"
        if entered:
            for rec in self._entering_recorders(owner, arms, subject, row):
                got = None
                where = f"{rec.caller.split('::')[-1].rpartition('.')[2]}:{rec.call}"
                mover = owner.split("::")[-1].rpartition(".")[2]
                if (callee := self._own_call(rec)) is not None:
                    got = self._own_call_pass(rec, callee, key, row, asking, found, at, placed=False)
                    if got is None:
                        crel, _, cfunc = callee.partition("::")
                        defs = R.functions(crel, cfunc)
                        silent = (len(defs) == 1 and bool(defs[0].body)
                                  and not getattr(R._prints_of(callee, [defs[0].body[0].lineno]), "statements", None))
                        self.say(f"{where}, reached from {mover} by the map's calls: " + (
                            f"{cfunc} prints nothing, so no run of it places the row inside a pass (A1)" if silent else
                            f"the records place no run of {cfunc} at the row inside a pass of {mover} (A1)"))
                else:
                    kept = len(self.unanswered)
                    kind, _said, _part, inner = self._recorder_read(rec, subject, key, row, asking)
                    dropped = len(self.unanswered) > kept
                    del self.unanswered[kept:]
                    p = next((p for p in found if kind == "" and inner and _during(inner, p, at)), None)
                    if p is not None:
                        names = sorted({m.token for m in inner})
                        got = p, f"{', '.join(names)} recorded it inside the pass, {rec.caller.split('::')[-1]}:{rec.call} recording it"
                    else:
                        methods = ", ".join(sorted({_bare(m).rpartition("::")[2].rpartition(".")[2] for m, _ in rec.methods}))
                        self.say(f"{where}, reached from {mover} by the map's calls: the records place no run of {methods} at "
                                 f"the row inside a pass of {mover}" + (" (their logs there are not in the records; A1 asks none)"
                                                                        if dropped else "") + " (A1)")
                if got is not None:
                    return got[0], (f"{got[1]}; {owner.split('::')[-1].rpartition('.')[2]} reaches "
                                    f"{rec.caller.split('::')[-1].rpartition('.')[2]} by the map's calls (A1)")
        return None

    def _entering_recorders(self, owner: str, arms: list[int], subject: str, row: dict) -> list[Recorder]:
        """A1: the recorders (:func:`recorders`) of the arms other than *owner*'s own *arms* that may write the value
        *row* records -- the value itself, or one known only at run time (retryTask_JEDI's ``runtime(taskOldStatus)``)
        -- in a function the map's calls reach from *owner*'s (:meth:`_reached`)."""
        value = str(row.get("status"))
        own = {(_bare(owner), a) for a in arms}
        out: list[Recorder] = []
        for j in self._subject_writers(subject):
            if getattr(j, "subject", subject) != subject:
                continue
            for b in j.branches:
                outcome = b.outcome or ""
                if (_bare(j.owner), b.line) in own or not (outcome.startswith(SV.RUN_TIME) or outcome.strip("'\"") == value):
                    continue
                if not self._reached(owner, j.owner):
                    continue
                out += [r for r in recorders(subject, j.owner, b.line) if r not in out]
        return out

    def _subject_writers(self, subject: str) -> list:
        """The map's writers of *subject* (:meth:`_writers_of`), read once."""
        cache: dict[str, list] = self.__dict__.setdefault("_subject_writers_read", {})
        if subject not in cache:
            cache[subject] = asyncio.run(self._writers_of(subject))
        return cache[subject]

    def _reached(self, owner: str, target: str) -> bool:
        """Whether the map's reach goes from *owner*'s function into *target*'s, by calls alone (``by_name`` off: a call
        closed by its name adds no way) -- TaskCommando's runImpl into retryTask_JEDI through the task buffer."""
        cache: dict[tuple[str, str], bool] = self.__dict__.setdefault("_reached_read", {})
        if (owner, target) not in cache:
            cache[(owner, target)] = False
            graph = self._graph()
            if graph is not None:
                RE, record, _rows = graph

                def at(o: str) -> str | None:
                    rel, _, func = _bare(o).partition("::")
                    defs = R.functions(rel, func)
                    return record.function_at(rel, defs[0].body[0].lineno) if len(defs) == 1 and defs[0].body else None

                start, end = at(owner), at(target)
                if start is not None and end is not None and start != end:
                    reached = record.reach([start], RE.Destination({end: ["called"]}, []), by_name=False)
                    cache[(owner, target)] = any(w.function == end for w in reached.writes or [])
        return cache[(owner, target)]

    def _own_call(self, rec: Recorder) -> str | None:
        """The function *rec*'s call enters, ``file::function``, where the map's history pair names the caller's own call
        as the write (level 0: ``writes`` is ``rec.caller:rec.call``, getTasksToExecCommand_JEDI's
        ``self.record_task_status_change(jediTaskID)`` at 4508): the one function the map's reach goes into from that
        line.  None for a pair whose recording method is another function (runImpl's updateTask_JEDI, K67's), or where the
        reach names none or more than one (KNOWN_ISSUES.md K76)."""
        if [(_bare(m), tuple(lines)) for m, lines in rec.methods] != [(_bare(rec.caller), (rec.call,))]:
            return None
        graph = self._graph()
        row = graph[1].rows.get(rec.caller) if graph is not None else None
        to = {c.to for c in (row.calls if row is not None else []) if c.how == "call" and rec.call in (c.lines or [c.line])}
        return _bare(to.pop()) if len(to) == 1 else None

    def _own_call_pass(self, rec: Recorder, callee: str, key: str, row: dict, asking: W.Asking, found: list,
                       at: datetime.datetime, placed: bool = True) -> tuple[Run, str] | None:
        """``(the pass, what says so)`` of *found* whose own call at ``rec.call`` recorded *row*, for a pair whose write is
        that call (:meth:`_own_call`, *callee* the function it enters), or None (KNOWN_ISSUES.md K76).  The pass is in
        hand, so none is asked by its marks, nor its own mark whole.  A callee that prints (record_task_status_change,
        "start" and "done" under ``jediTaskID={}``, read in the caller's logs where the map places it in none): its run
        of the key on the row's host, cut to its call at the row, ran
        inside the pass's call (``recorded_in_pass``) while the row was written (:func:`_during`); or, not *placed*, where
        the pass's prints cannot bracket the call (a loop's next round prints between, as getTasksToExecCommand_JEDI's
        over its commands, or no print follows on every path), inside the pass, the call not named.  One that prints
        nothing (recordStatusChange): the pass's records just before and just after the call hold the row's time
        (``inside_call`` of the moment) -- the call's place, not a run of it, and said so."""
        host = str(asking.machine)
        rel, _, func = _bare(rec.caller).partition("::")
        fn = R.function(rel, func, [rec.call])
        site = f"{fn.rel}:{rec.call}"
        crel, _, cfunc = callee.partition("::")
        defs = R.functions(crel, cfunc)
        mine = [p for p in found if p.machine == host]
        if len(defs) != 1 or not defs[0].body or not mine:
            return None
        lines = [defs[0].body[0].lineno]
        if not getattr(R._prints_of(callee, lines), "statements", None):
            if not placed:
                return None
            moment = _stamp(at)
            for p in mine:
                if (got := inside_call(R.seen(fn, p, self.mode), Moment(moment, moment), site)) is not None:
                    return p, T.text(T.Said("its-call-at-the-row", {"func": func, "site": site, "callee": cfunc,
                                                                    "before": p.records[got[0]].stamp,
                                                                    "after": p.records[got[1]].stamp}))
            return None
        t = datetime.datetime.fromisoformat(str(row["time"])[:19])
        edge = ((t - EDGE).isoformat(" "), (t + EDGE).isoformat(" "))
        ask = W.Asking(edge, asking.machine, T.Said("for-the-rows-recorder", {
            "methods": [cfunc], "status": row.get("status"), "time": row["time"], "machine": host}))
        # a callee the map places in no log of its own -- BaseModule's, a mixin whose logger is its object's -- runs as
        # the caller's classes there (the map's reach, ``runs_as``), so its lines are read in the caller's logs
        files = self.files_of(callee, [], lines) or self.files_of(_bare(rec.caller), [], [rec.call])
        logs, unanswered = self._logs_at(key, list(dict.fromkeys(files)), ask, owners_of([callee]))
        if unanswered:
            # kept once, when the call is first read
            if placed:
                self._keep_unanswered(unanswered)
            return None
        # the callee's mark holds every call of it for the key (record_task_status_change's at 'staged' and at
        # 'defined', 17 minutes apart): each run is cut at its entry rows, and the calls at the row are read
        cfn, when = R.function(crel, cfunc, [], lines), (str(row["time"]), str(row["time"]))
        inner = [part for log in logs.values() if host not in log.without for r in runs_for(log, key, self._other_of(key, log))
                 if of_mover(r, callee) and r.machine == host and r.last[:19] >= edge[0] and r.first[:19] <= edge[1]
                 for part in passes_of(cfn, r, self.mode)[0] if at_window(part, when)]
        for p in mine:
            # each pass on the calls inside it: the callee is called again in the row's second by another pass
            # (MT-exhausted-late's 'toretry' at 11:14:23.613, its next call at .688 after the pass ended at .616)
            at_row = [m for m in inner if _during([m], p, at)]
            if not at_row:
                continue
            if placed and recorded_in_pass(fn, [p], when, at_row, [site], self.mode):
                return p, T.text(T.Said("its-pass-recorded-the-row", {"run": p.said(), "methods": [cfunc], "key": key, "site": site}))
            if not placed:
                names = sorted({m.token for m in at_row})
                return p, f"{', '.join(names)} recorded it inside the pass, {func} calling {cfunc}; its prints do not place the call"
        return None

    def _which_way(self, fn: R.Function, rel: str, func: str, found: list, arms: list[Arm]) -> Blocked:
        """Which way each pass of *found* went -- an arm of *arms*, an exit of *fn*, or none of them -- and the one way
        every pass took: S2's reading of a mover's passes, and U2's of a caller's, its calls into the mover the arms."""
        exits = sorted({n.lineno for n in R.own_nodes(fn.node) if isinstance(n, R.JUMPS)})
        lines = sorted({a.line for a in arms} | set(exits))
        self.say(f"candidates: arms {[a.line for a in arms]}, exits {exits}")
        chosen: list[Any] = []  # each pass's verdict, None where it ran none of the arms
        took: list[int | None] = []  # the exit a pass that ran no arm was shown to leave by, None where none is shown
        envs = []
        left_by: set[int] = set()  # the exits a pass that ran no arm may have left by (S-1, S-2a)
        unhanded: list[str] = []  # D-4: a pass every arm of which was a call handing no element of the key's turn
        arm_lines = {a.line for a in arms}
        for n, p in enumerate(found, 1):
            env = self.base_env(at=p.first)
            stable_holes([R.seen(fn, p, self.mode)], env, class_of(fn), local_to=fn.owner)
            self.say(f"-- pass {n}: {p.first} .. {p.last}, {len(p.records)} record(s); printed {self._printed(env)}")
            j = judge([Candidate(rel, func, lines, run=p)], {p.log.name: [p]}, mode=self.mode, decide=self.decide, question_run=p, env=env)
            self.say(j.report(p))
            mine = [v for v in j.verdicts if v.line in arm_lines]
            unhanded.append("; ".join(dict.fromkeys(v.unhanded for v in mine))
                            if mine and len(mine) == len(arm_lines) and all(v.unhanded for v in mine) else "")
            if j.chosen is None and (arm := _the_arm_taken(j, {a.line for a in arms})) is not None:
                j.chosen = arm
                self.say(f"  the way pass {n} went is arm {arm.line}: the positive exit(s) "
                         f"{sorted(v.line for v in j.verdicts if v.kind == 'positive' and v is not arm)} are where it left after")
            if took_none(j):
                self.say(f"  pass {n} took none of the candidates: no arm of {func} that moves the value ran in it")
                chosen.append(None)
                took.append(None)
            elif (way := ran_no_arm(j, {a.line for a in arms})) is not None:
                # S-2a: every arm excluded, so the pass ran none, the exit it took shown or not
                shown, left = way
                if shown is not None:
                    self.say(f"  pass {n} ran no arm of {func} that moves the value: every arm excluded; it left by {shown}")
                else:
                    by = f" ({j.chosen.line} chosen by elimination, which no row shows)" if j.chosen is not None else ""
                    self.say(f"  pass {n} ran no arm of {func} that moves the value: every arm excluded; it left by one of "
                             f"{left}{by} or ran on past them")
                left_by.update(left if shown is None else [shown])
                chosen.append(None)
                took.append(shown)
            else:
                chosen.append(self.pick(T.Said("which-way-pass-went", {"n": n}), j))
                took.append(None)
            envs.append(env)
        if all(v is None for v in chosen) and len(set(took)) == 1 and took[0] is not None:
            # every pass ran no arm and was shown to leave by the same exit: the way they went
            self.say(f"every pass ({len(found)}) left by {func}:{took[0]}")
            return Blocked(fn, took[0], found, envs, arms, unhanded=tuple(unhanded))
        taken = {v.line if v is not None else None for v in chosen}
        self.need(T.Said("one-exit-every-pass-took"), len(taken) == 1)
        exit_line = taken.pop()
        self.say(f"every pass ({len(found)}) left by {func}:{exit_line}" if exit_line is not None
                 else f"every pass ({len(found)}) of {func} took none of its candidates" if not left_by
                 else f"every pass ({len(found)}) of {func} ran none of its arms")
        return Blocked(fn, exit_line, found, envs, arms, tuple(sorted(left_by)),
                       tuple(took) if exit_line is None and any(x is not None for x in took) else (), tuple(unhanded))

    # ------------------------------------------------------------ Callers: what called the mover on the row
    def callers(self, q: Callers):
        """U1: the callers of a mover none of whose passes took the row in the window, each read on its own (U2), as S1
        forks over the movers (KNOWN_ISSUES.md K55)."""
        self.start("U1", "what called the mover on the row")
        func = q.mover.split("::")[-1]
        if self._own_cycles(q):
            return None
        cls = q.subject.partition(".")[0]
        if (other := self._mark_key(q)) is not None:
            self.say(f"the run that moved it is marked by another {cls}'s key ({other}), not the row's: its callers' "
                     "passes are asked by that key (H-4)")
            q = replace(q, key=other)
        found = callers_of(q.entries, cls, q.key, self._key_columns(q.subject, q.key))
        self.need(T.Said("a-caller-the-map-lists", {"func": func}), found, "K55")
        found = self._callers_by_printed(q, found)
        found = self._callers_by_service(q, found)
        ways = {"marks": "asked for its marks holding the key", "key": "asked for the key alone (a row of it prints it)",
                "none": "its runs print no key"}
        for c in found:
            through = f", through {', '.join(c.between)}" if c.between else ""
            across = f"; across the send at {', '.join(c.sent)}: {c.across}" if c.sent else ""
            self.say(f"  {c.owner.split('::')[-1]} at {c.lines} (hop {c.hop}{through}{across}): {ways[c.how]}")
        asks = [CallerPasses(c, q) for c in found]
        if len(asks) == 1:
            return asks[0]
        return Fork(asks, [T.Said("caller-branch", {"caller": c.owner.split("::")[-1], "lines": c.lines}) for c in found])

    def _callers_by_printed(self, q: Callers, found: list[Caller]) -> list[Caller]:
        """Bundle G, G-5 (P28, out/gm/qk-design.md, tmaeno's OK 2026-11-17 16:0x): of the callers U1 lists, those every
        way of a parameter the mover printed comes in by from them ends at a literal other than the value its moving run
        printed (the map's ParameterFrom, :func:`sources.param_ways`, the way's route holding one of the caller's call
        lines) did not hand that run its value.  killJob prints ``code=51`` at 840 before any rebinding of ``code``:
        copyArchive's calls handing '10' or '2' are not the one.  A caller with a way the map does not end at a literal,
        or at the value, stays; where every caller would go, none goes.  Only the mover's moving run of a late value.
        P9: of a caller kept, the call lines read the same way one by one: those every way through which hands another
        literal go, where one is left (copyArchive's main, 21 calls into killJobs, the rebrokerage's handing '51')."""
        q2 = getattr(self, "_missed_q", None)
        if q2 is None or q2.mover != q.mover or q2.run is None:
            return found
        printed = self._printed_parameters(q.mover, q2.run)
        if not printed:
            return found
        bare = _bare(q.mover)
        ways = {p: SR.param_ways(bare, p) for p in printed}

        def other_value(rel: str, lines: list[int]) -> tuple[str, set[str]] | None:
            # the parameter every way through *lines* hands another literal of, with those literals; None where none
            sites = {f"{Path(rel).name}:{line}" for line in lines}
            for p, value in printed.items():
                mine = [w for w in ways[p] if any(f"{Path(r.rpartition(':')[0]).name}:{r.rpartition(':')[2]}" in sites
                                                  for r in w.route)]
                if mine and all(w.end == SR.CONSTANT and not w.steps and w.why not in SR.NO_LITERAL
                                and not SR.same_value(w.value, value) for w in mine):
                    return p, {str(w.value) for w in mine}
            return None

        def words(found: list[tuple[str, set[str]]]) -> str:
            by: dict[str, set[str]] = {}
            for p, values in found:
                by.setdefault(p, set()).update(values)
            return "; ".join(f"{p} {sorted(v)}, none the value the run printed for it" for p, v in sorted(by.items()))

        kept: list[tuple[Caller, str]] = []
        dropped: list[tuple[Caller, str]] = []
        for c in found:
            other = other_value(c.owner.partition("::")[0], c.lines) if len(found) > 1 else None
            (dropped if other else kept).append((c, f"its calls hand {words([other])}" if other else ""))
        if not kept:
            kept, dropped = [(c, "") for c in found], []
        for c, why in dropped:
            self.say(f"  {c.owner.split('::')[-1]} at {c.lines}: not the caller of the run that moved it: {why} (G-5)")
        # P9 (the investigator's u9, NJ-defined-late and MJ-starting-late): of a caller's call lines, those every way
        # through which hands another literal did not make the run -- copyArchive's main calls killJobs from 21 lines,
        # the rebrokerage's handing code '51', the value killJob printed
        out: list[Caller] = []
        for c, _ in kept:
            rel = c.owner.partition("::")[0]
            off = {line: other for line in c.lines if (other := other_value(rel, [line]))} if len(c.lines) > 1 else {}
            left = [line for line in c.lines if line not in off]
            if off and left:
                self.say(f"  {c.owner.split('::')[-1]} at {c.lines}: its calls at {sorted(off)} hand "
                         f"{words(list(off.values()))}: the run's call is at {left} (P9, G-5)")
                c = replace(c, lines=left, across=[line for line in c.across if line in left],
                            outer=[(o, line) for o, line in c.outer if line in left])
            out.append(c)
        return out

    def _mark_key(self, q: Callers) -> str | None:
        """Bundle H, H-4 (P20, out/gm/qk-design.md, bundle H, tmaeno's OK 2026-11-17 17:1x): the one value the mover's
        moving run's mark holds in a hole the map names a key column of the asked class by, where it is not the row's
        key: the run wrote the row in a pass of that other object, which its callers' passes are asked by.
        NJ-merging-late's updateUnmergedJobs ran under ``updateUnmergedJobs < PandaID=<the merge job> >``, writing the
        unmerged job in the merge job's pass.  None where no such hole, two values, or not the moving run of a late
        value."""
        q2 = getattr(self, "_missed_q", None)
        if q2 is None or q2.mover != q.mover or q2.run is None or not getattr(q2.run, "token", ""):
            return None
        own = self._key_columns(q.subject, q.key)
        others: set[str] = set()
        for r in getattr(R._prints_of(q.mover, [a.line for a in q.arms]), "runs", None) or []:
            vs = mark_values(q2.run.token, r.mark)
            if vs is None or len(vs) != len(r.key or []):
                continue
            for k, v in zip(r.key, vs, strict=True):
                if set(getattr(k, "columns", None) or []) & own and str(v).strip() and str(v).strip() != q.key:
                    others.add(str(v).strip())
        return others.pop() if len(others) == 1 else None

    def _callers_by_service(self, q: Callers, found: list[Caller]) -> list[Caller]:
        """Bundle H, H-3 (P15, out/gm/qk-design.md, bundle H, tmaeno's OK 2026-11-17 17:1x): pandajedi's code runs in
        JEDI's processes alone (pandaserver does not import it), so a route whose outermost hop is in ``pandajedi/``
        runs on a JEDI machine; where the mover's moving run is on a machine the records name another service's, those
        routes did not make that run.  NJ-defined-late's killJob ran on aipanda097, a server machine: killJobs' routes
        from TaskCommando and AtlasProdWatchDog are not its.  A caller every route of which is so goes; one with other
        routes stays, the JEDI routes said.  A route entering in pandaserver is not placed (JEDI imports pandaserver);
        where every caller would go, none goes.  Only the mover's moving run of a late value, as G-5."""
        q2 = getattr(self, "_missed_q", None)
        if q2 is None or q2.mover != q.mover or q2.run is None:
            return found
        machine = getattr(q2.run, "machine", "") or ""
        service = (getattr(getattr(q2.run, "log", None), "services", None) or {}).get(machine)
        if not machine or service is None or service == "jedi":
            return found
        where = f"{machine.split('.')[0]}, a {service} machine as the records say"
        kept, said = [], []
        for c in found:
            jedi = sorted({o for o, _ in c.outer if o.startswith("pandajedi/")})
            if not jedi:
                kept.append(c)
            elif all(o.startswith("pandajedi/") for o, _ in c.outer):
                said.append(f"  {c.owner.split('::')[-1]} at {c.lines}: not the caller of the run that moved it: every "
                            f"route into it enters at {', '.join(jedi)}, in a JEDI process, and the run is on {where} (H-3)")
            else:
                kept.append(c)
                said.append(f"  {c.owner.split('::')[-1]} at {c.lines}: its routes entering at {', '.join(jedi)}, in a "
                            f"JEDI process, are not the run's: the run is on {where} (H-3)")
        if not kept:
            return found
        for line in said:
            self.say(line)
        return kept

    def _printed_parameters(self, mover: str, run) -> dict[str, str]:
        """G-5: the mover's parameters its moving *run* printed as handed -- a bare parameter name in a hole of a row
        no binding of that name in the function precedes -- each with the one value its records there hold.  A name
        printed with two values, or bound before the print, is left out."""
        rel, _, func = _bare(mover).partition("::")
        try:
            fn = R.function(rel, func, [])
        except (LookupError, ValueError, SyntaxError):
            return {}
        node = fn.node
        args = node.args
        params = {a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)} - {"self", "cls"}
        bound: dict[str, int] = {}
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store) and n.id in params:
                bound[n.id] = min(bound.get(n.id, n.lineno), n.lineno)
        sn = R.seen(fn, run, self.mode)
        values: dict[str, set[str]] = {}
        for row in fn.rows:
            for i, _, _ in sn.hits.get(row.line, ()):
                for expr, text in row.capture(run.records[i]).items():
                    if expr in params and text is not None and row.line < bound.get(expr, row.line + 1):
                        values.setdefault(expr, set()).add(str(text).strip())
        return {p: v.pop() for p, v in values.items() if len(v) == 1}

    def caller(self, q: CallerPasses):
        """U2: which way each pass of one caller went on the row in the window -- its calls into the mover are its arms,
        S2's reading -- and so what kept it from calling: S3 on the call's guard.  No pass of it on the row, where the
        window was searched whole and every line of its runs holds the key: it did not run on the row, and U1 goes up
        to its own callers once; else it stops, saying what is not read (KNOWN_ISSUES.md K55)."""
        c, up = q.caller, q.callers
        self.start("U2", "which way the caller's passes went on the row")
        rel, func = c.owner.split("::", 1)
        mover = up.mover.split("::")[-1]
        start, end = up.window
        # Q1 part 2: a caller whose runs print no key, or whose logs are not to hand, is read by its callees' runs
        if c.how == "none" and self._call_tied(c, up):
            return None
        self.need(T.Said("a-caller-s-run-printing-the-key", {"caller": func, "lines": c.lines}), c.how != "none", "K55")
        prints = R._prints_of(c.owner, c.lines)
        files = self.files_of(c.owner, sorted({f for st in prints.statements for f in st.log_files}) if prints else [], c.lines)
        if not self.logs.names(files) and self._call_tied(c, up):
            return None
        to_hand = self.need(T.Said("a-log-of-to-hand", {"func": func, "files": files}), self.logs.names(files))
        how = T.Said("for-marks", {"marks": list(c.marks)}) if c.marks else T.Said("for-the-key-alone")
        asking = W.Asking(up.window, None, T.Said("over-the-movers-window", {"start": start, "end": end, "how": how}), c.marks)
        self.say(f"{func}: logs {files}, asked {asking.why}")
        runs, missed, holes = self._ask(up.key, to_hand, asking)
        if missed and self._call_tied(c, up):
            # its own logs are asked for the next round; what its callees show is said now
            self._keep_unanswered(missed)
            return None
        self._stop_unanswered(missed)
        fn = R.function(rel, func, c.lines)
        found = []
        whole: set[int] = set()
        for name in to_hand:
            for run in runs.get(name, []):
                if of_mover(run, c.owner):
                    parts, said = passes_of(fn, run, self.mode)
                    self.say(f"{run.describe()}: {said}")
                    found += parts
                    whole |= {id(p) for p in parts if p is run}
        found = sorted((p for p in found if p.first[:19] <= end), key=lambda p: p.first)
        if up.after is not None:
            # I-2: the passes begun after the mover's last on the row; the window opens at no statuslog row
            found = [p for p in found if W.when(p.first) > W.when(up.after)]
            moved_on: list = []
        else:
            found, moved_on = self._stay_edges(fn, c.owner, c.lines, up.subject, up.key, found, start, whole)
        keyed = [p for p in found if holds_key(fn, p, runs, self.mode, self._called_at)]
        if len(found) > len(keyed):
            self.say(f"{len(found) - len(keyed)} pass(es) of {func} in the window print no record of the key, nor run it inside a call: not read")
        moving: list = moved_on
        if up.before is not None:
            if not moved_on:
                moving = [p for p in keyed if p.last[:19] >= up.before]
                keyed = [p for p in keyed if p.last[:19] < up.before]
                self.say(f"{len(moving)} pass(es) ending at {up.before} or later, the one that called {mover} to move it on among them, not read")
            if c.across and keyed:
                keyed, moved = self._moved_across(fn, rel, func, keyed, c, up.before)
                # the pass set aside as the one that moved it on is a moving pass (P16: M3(a)-2 reads its time)
                moving = [*moving, *(p for p in moved if not any(p is m for m in moving))]
                if moved and not keyed:
                    # no pass before the one that moved it: the caller is read as absent up to that pass
                    up = replace(up, window=(start, moved[0].first))
        if not keyed:
            return self._caller_absent(c, up, to_hand, asking, holes, moving)
        outcomes = sorted({o for a in up.arms for o in a.outcomes})
        arms = [Arm(line, outcomes, self._call_guards(rel, line)) for line in c.lines]
        blocked = self._which_way(fn, rel, func, keyed, arms)
        if self._caller_handed_none(c, up, fn, keyed, blocked, to_hand, asking, holes):
            return None
        self.need(T.Said("caller-called-mover-not-found", {"caller": func, "line": blocked.exit_line, "mover": mover}),
                  blocked.exit_line not in c.lines, "K55")
        return blocked

    def _caller_handed_none(self, c: Caller, up: Callers, fn: R.Function, keyed: list, blocked: Blocked, to_hand: list[str],
                     asking: W.Asking, holes: dict) -> bool:
        """I-2 (out/gm/qk-design.md's bundle I second half, tmaeno's OK 2026-11-17 20:3x), one rule for U2 whatever the
        window's end (a late value's move, a stuck one's as_of): where every pass of the caller *c* on the row in the
        window had each of its calls into the mover hand a list without the key's element (D-4, ``Blocked.unhanded``),
        no pass of it handed the mover the row there -- where every pass of it on the row is read: the window searched
        whole, every log of it asked, and a pass handing the key's element printing a record of it (the rows every way
        to the append prints begin with the element's key, :func:`which_ran.handing_rows`).  An END and True; False,
        said why, where a premise does not hold (S3 reads on)."""
        if not keyed or len(blocked.unhanded) != len(keyed) or not all(blocked.unhanded):
            return False
        func = c.owner.split("::")[-1]
        rows = handing_rows(fn, c.lines)
        premises = {"window": self._searched_whole(to_hand, asking, asking, [], holes, None),
                    "logs": every_log_asked(c.owner, c.lines, to_hand),
                    "prints": {"held": bool(rows), "why": T.Said("a-handing-prints-the-element", {"rows": rows, "lines": c.lines})
                               if rows else T.Said("a-handing-prints-no-element", {"lines": c.lines})}}
        failing = [p["why"] for p in premises.values() if not p["held"]]
        if failing:
            self.say(f"every pass of {func} on the row handed {up.mover.split('::')[-1]} no element of the key's turn (D-4), "
                     f"but not every pass of it on the row is read: {'; '.join(T.text(w) for w in failing)}")
            return False
        self.end(T.Said("no-pass-of-the-caller-handed-the-row", {
            "caller": func, "lines": c.lines, "mover": up.mover.split("::")[-1], "n": len(keyed),
            "start": up.after or up.window[0], "end": up.window[1], "why": list(dict.fromkeys(blocked.unhanded)),
            "after": up.left or "", "premises": [p["why"] for p in premises.values()]}))
        return True

    def _call_tied(self, c: Caller, up: Callers) -> bool:
        """Q1 part 2 (call-tied runs; out/gm/qk-design.md, the Q1 design summary, tmaeno's OK 2026-11-15 23:06; P22,
        P8): where the caller's own runs print no key, or its logs are not in the records, its passes on the row are
        read from the runs of the functions it calls that print the key (the map's ``reach_graph``,
        :func:`_callees_by_line`), asked for those functions alone over the window.  Each such run, under a mark that
        names its method, is a pass of the caller on its machine at its time that went through the caller's calls
        reaching that function: the TaskBroker thread's failed brokerage is its updateTask_JEDI run (post_process_for_
        error's calls), the one that moved the row its setCloudToTasks_JEDI run.  That the run is the caller's call, and
        not another caller's of the same function, is named, with how many others call it.

        Where the run of the mover's own (*up*'s) is one pass, and a run before it on its machine came through a call the
        caller makes before its calls into the mover, ending at most windows.SLACK before it, the mover took the row in
        the step after that one in one pass of the caller (P8): PostProcessor.start's prepareTasksToBeFinished_JEDI,
        then its getTasksToBeFinished_JEDI.  Otherwise the passes on the row are said, each with its way: those the stay's
        edges leave, read as S2 reads a mover's (:meth:`_off_the_stay`, C-2).  An END and True; False where no such run is
        read, where none but the one that moved the row is on it, or where a log of them is not in the records (the asks
        kept)."""
        rel, func = c.owner.split("::", 1)
        try:
            fn = R.function(rel, func, c.lines)
        except (LookupError, ValueError, SyntaxError):
            return False
        qual = f"{rel}::{pathcond.qualified_name(fn.node)}"
        reached = _callees_by_line(qual)
        columns = self._key_columns(up.subject, up.key)
        mover = _bare(up.mover)
        # each callee that prints the key: the lines of the caller's calls reaching it, the map's name, its logs
        printing: dict[str, tuple[set[int], str, set[str]]] = {}
        # E-1 (W8): a callee whose runs are marked by a column that references the key's class's identity, with that
        # column and the foreign keys it goes through -- feed_contents_to_tasks' insertFilesForDataset_JEDI
        referenced: dict[str, list[str]] = {}
        cls = up.subject.partition(".")[0]
        for o, (lines, _) in reached.items():
            bare = _bare(o)
            try:
                found = R._prints_of(o)
            except LookupError:
                continue
            if found is None or bare in printing:
                continue
            if not (bare == mover or prints_key(o, (), columns) or key_marks(o, 0, cls, up.key)):
                marks, refs = ref_marks(o, 0, cls, up.key)
                if not marks:
                    continue
                referenced[bare] = refs
            printing[bare] = (lines, o, {f for st in found.statements for f in st.log_files})
        files = {f for _, _, fs in printing.values() for f in fs}
        to_hand = self.logs.names(sorted(files))
        if not printing or not to_hand:
            self.say(f"{func}'s callees: none prints the key in a log to hand")
            return False
        start, end = up.window[0], up.before or up.window[1]
        asking = W.Asking((start, end), None, T.Said("for-the-callers-callees", {"caller": func, "start": start, "end": end,
                                                                                 "n": len(printing)}))
        runs, missed, _holes = self._ask(up.key, to_hand, asking, {}, quiet=True)
        if missed:
            # a callee's log not in the records may hold a pass of the caller on the row: nothing is read without it
            # (NT-assigning-late's panda-AtlasProdTaskBroker.log); the asks are kept and U2 goes on its own way, whose
            # asks come in the same round (NT-throttled-late's marks near the key's runs)
            self._keep_unanswered(missed)
            self.say(f"{func}'s callees: {len({ask.key for ask, _ in missed})} ask(s) of their logs over the window are "
                     "not in the records: their passes are not read")
            return False
        passes = []
        lo, hi = W.when(start), W.when(end) + datetime.timedelta(seconds=1)
        for name in to_hand:
            for run in runs.get(name, []):
                m = re.match(r"([A-Za-z_]\w*) <", run.token)
                if m is None:
                    continue
                for bare, (lines, o, fs) in printing.items():
                    if bare.rpartition("::")[2].rpartition(".")[2] != m[1] or name not in fs:
                        continue
                    # a mark's run glues every call of it: its records of the key in the window, cut where they stop
                    # for a minute, each a call
                    stamps = sorted(r.stamp for r in run.found_by if lo <= W.when(r.stamp) <= hi)
                    for a, b in _bursts(stamps, datetime.timedelta(minutes=1)):
                        passes.append((a, b, run.machine, bare, lines, o))
        # W14 (the investigator's u11, P68): a run of the mover inside the run that moved the row (L1's), whose function
        # is another caller of the mover, is that caller's call, not this one's -- NJ-merging-late's archiveJob at
        # 05:05:11 on aipanda099, inside updateUnmergedJobs's run, read for killJobs
        moving = (self.result.get("moved_by") or {}) if isinstance(getattr(self, "result", None), dict) else {}
        mrun = moving.get("run") or {}
        mowner = _bare(str(moving.get("owner", "")))
        if mrun.get("machine") and mowner and mowner != _bare(c.owner):
            lo_m, hi_m = W.when(str(mrun["first"])), W.when(str(mrun["last"]))
            # the mover's name as the map's closure keys it: the pass's own (its sixth item)
            inside = [p for p in passes if p[3] == mover and p[2] == mrun["machine"]
                      and mowner in {_bare(o) for o in MR.reaching(p[5], 1)}
                      and W.when(p[0]) <= hi_m + datetime.timedelta(seconds=1)
                      and W.when(p[1]) >= lo_m - datetime.timedelta(seconds=1)]
            if inside:
                passes = [p for p in passes if p not in inside]
                shown = ", ".join(p[0][:23] for p in inside[:3]) + (f" and {len(inside) - 3} more" if len(inside) > 3 else "")
                self.say(f"{len(inside)} run(s) of {mover.rpartition('::')[2]} on {mrun['machine'].split('.')[0]} at "
                         f"{shown} lie in the run that moved the row, "
                         f"{mowner.rpartition('::')[2]}'s ({mrun['first'][:23]} .. {mrun['last'][:23]}), another caller of "
                         f"it: its calls, not {func}'s (W14)")
        # a callee reached through the calls into the mover and through others alike does not tell the way
        told = sorted((p for p in passes if p[3] == mover or not set(p[4]) & set(c.lines)), key=lambda p: (p[2], p[0]))
        # the caller's passes: a machine's runs of its callees a minute or less apart
        grouped: list[list] = []
        for p in told:
            if grouped and grouped[-1][0][2] == p[2] and W.when(p[0]) - W.when(max(q[1] for q in grouped[-1])) < datetime.timedelta(minutes=1):
                grouped[-1].append(p)
            else:
                grouped.append([p])
        grouped.sort(key=lambda g: min(q[0] for q in g))
        if all(any(q[3] == mover for q in g) for g in grouped):
            # only the pass that moved it: nothing read of the passes before it
            self.say(f"{func}'s callees: no pass on the row but the one that moved it")
            return False
        mine = {qual, *reached}

        def callee_premises(groups: list[list]) -> list[T.Said]:
            # the mover's own runs too (W14): another caller of it may have run it on the row then
            return [T.Said("a-callee-run-is-the-caller-s", {
                "caller": func, "callee": o.rpartition("::")[2].rpartition(".")[2], "n": _others_reaching(o, mine)})
                for o in sorted({q[5] for g in groups for q in g})] + [
                T.Said("a-callee-marked-by-a-reference", {
                    "callee": b.rpartition("::")[2].rpartition(".")[2], "attr": referenced[b][0], "cls": cls,
                    "keys": referenced[b][1:]})
                for b in sorted({q[3] for g in groups for q in g if q[3] in referenced})]

        premises = callee_premises(grouped)

        def words(g: list) -> T.Said:
            callees = list(dict.fromkeys(f"{q[3].rpartition('::')[2]} {sorted(q[4])}" for q in g))
            return T.Said("a-caller-pass-by-its-callees", {
                "machine": g[0][2].split(".")[0], "first": min(q[0] for q in g)[:23], "last": max(q[1] for q in g)[:23],
                "callees": callees, "way": T.Said("it-called-the-mover" if any(q[3] == mover for q in g) else "not-into-the-mover")})

        said = [words(g) for g in grouped]
        began = up.window[0]
        # P8: the stay began in a pass of a call the caller makes before its calls into the mover, and the mover's run
        # is the next on that machine
        moving = [g for g in grouped if any(q[3] == mover for q in g)]
        if moving:
            mv = moving[-1]
            at = min(q[0] for q in mv if q[3] == mover)
            prior = [g for g in grouped if g is not mv and g[0][2] == mv[0][2] and max(q[1] for q in g) <= at]
            prev = prior[-1] if prior else None
            # the step: the function the caller calls nearest, of those reached only through calls before the mover's
            early = sorted((q for q in prev or [] if max(q[4]) < min(c.lines)), key=lambda q: (reached[q[5]][1], min(q[4])))
            if prev is not None and early and min(q[0] for q in prev)[:19] <= began <= max(q[1] for q in prev)[:23]:
                step = early[0]
                last = self._step_end(step[3], printing[step[3]][2], step[0], at, up.key)
                premises.append(T.Said("one-process-a-machine", {"caller": func, "machine": step[2].split(".")[0]}))
                self.end(T.Said("took-it-in-the-next-step", {
                    "mover": mover.rpartition("::")[2], "at": at[:23], "machine": step[2].split(".")[0], "caller": func,
                    "callee": step[3].rpartition("::")[2], "lines": sorted(step[4]), "mover_lines": c.lines, "began": began,
                    "rest": T.Said("the-step-ended-at", {"last": last[:23], "rest": f"{(W.when(last) - W.when(began)).total_seconds():.1f}"})
                    if last else T.Said("the-step-s-end-not-read", {"callee": step[3].rpartition("::")[2]}),
                    "passes": said, "premises": premises}))
                return True
        # C-2 (bundle C, W5; out/gm/qk-design.md, tmaeno's OK 2026-11-16 23:06): the passes said to be on the row are read
        # at the stay's edges as S2 and U2 read a mover's or a caller's own (BP', :meth:`_off_the_stay`), to the
        # microsecond where the rows have it.  NT-assigning-late's getTaskWithID_JEDI run on aipanda085 at 07:41:00,150 ..
        # ,175 -- ContentsFeeder's own call, before its pass recorded 'assigning' at 07:41:00.667322 -- is no pass of
        # TaskBroker's runImpl on the row.  P8 above reads the step the stay began in, which may end before the row
        off, off_said = self._off_the_stay([(min(q[0] for q in g), max(q[1] for q in g)) for g in grouped], start)
        for line in off_said:
            self.say(f"{func}'s callees: {line}")
        on_row = [g for i, g in enumerate(grouped) if i not in off]
        if all(any(q[3] == mover for q in g) for g in on_row):
            self.say(f"{func}'s callees: no pass on the row but the one that moved it")
            return False
        self.end(T.Said("the-caller-s-passes-by-its-callees", {
            "caller": func, "passes": [words(g) for g in on_row], "premises": callee_premises(on_row),
            "unknown": T.Said("why-each-pass-took-its-way", {"caller": func, "files": self.files_of(c.owner, [], c.lines)})}))
        return True

    def _step_end(self, bare: str, files: set[str], since: str, until: str, key: str) -> str | None:
        """The last record of the pass of the callee *bare* holding the key's record at *since* (P8's step), its run read
        whole (its own mark asked for, ``owners_of``) and cut into passes as S2 cuts a mover's (``passes_of``): the
        pass of PostProcessor's prepareTasksToBeFinished_JEDI went on to 14:35:06,032 over other tasks, though the key's
        last record in it is at 14:34:02.  None where its run is not in the records (the ask is kept for the next
        round)."""
        rel, _, name = bare.partition("::")
        try:
            fn = R.function(rel, name)
        except (LookupError, ValueError, SyntaxError):
            return None
        lo = W._down(W.when(since) - datetime.timedelta(minutes=1))
        asking = W.Asking((lo, until[:19]), None, T.Said("for-the-step-s-run", {"callee": name, "start": lo, "end": until[:19]}))
        runs, missed, _holes = self._ask(key, self.logs.names(sorted(files)), asking, owners_of([bare]), quiet=True)
        self._keep_unanswered(missed)
        for rs in runs.values():
            for run in rs:
                if not of_mover(run, bare):
                    continue
                for p in passes_of(fn, run, self.mode)[0]:
                    if p.first[:23] <= since[:23] <= p.last[:23]:
                        return p.last
        return None

    def _moved_across(self, fn: R.Function, rel: str, func: str, keyed: list, c: Caller, before: str) -> tuple[list, list]:
        """U2X (B) (out/gm/qk-design.md U2X): a caller across a send (:func:`across_sends`) whose call moved the row ends
        its pass before the move, by the send's time (copyArchive's pass at 10:33:04,808, killJob's at 10:33:15,725), so
        ending before *before* does not make a pass one before the move.  Of *keyed*, a pass ending within windows.SLACK
        of *before* whose way (S2's judgement over the caller's calls into the mover and its exits) is a call across the
        send is the one that moved it on: set aside, and said.  By its way, not its time alone: a pass in that span that
        took another way is still read as one before the move.  The passes kept and the ones set aside."""
        edge = W.when(before) - W.SLACK
        exits = sorted({n.lineno for n in R.own_nodes(fn.node) if isinstance(n, R.JUMPS)})
        lines = sorted(set(c.lines) | set(exits))
        kept, moved = [], []
        for p in keyed:
            if W.when(p.last) < edge:
                kept.append(p)
                continue
            env = self.base_env(at=p.first)
            stable_holes([R.seen(fn, p, self.mode)], env, class_of(fn), local_to=fn.owner)
            j = judge([Candidate(rel, func, lines, run=p)], {p.log.name: [p]}, mode=self.mode, decide=self.decide, question_run=p, env=env)
            took = j.chosen if j.chosen is not None else _the_arm_taken(j, set(c.lines))
            line = took.line if took is not None else None
            if line in c.across:
                self.say(f"pass {p.first} .. {p.last} took {func}:{line} ({took.kind}), a call across the send at {', '.join(c.sent)}, and ended "
                         f"within {int(W.SLACK.total_seconds() // 60)} min of the move at {before}: the one that moved it on, not read as a pass before it")
                moved.append(p)
            else:
                kept.append(p)
        return kept, moved

    def _call_guards(self, rel: str, line: int) -> list[str]:
        """The tests reaching the call at *rel*:*line* runs under (``pathcond.arm_condition``: its ``if``s and the early
        exits above it), of the statement the map's ``call_statements`` picks for the hop (``called_from``'s own
        reader); two statements on the line that differ are not read."""
        stmts = call_statements(R.CORPUS, f"{rel}:{line}")
        conds = {tuple(pathcond.arm_condition(s)) for s in stmts}
        self.need(T.Said("one-call-statement-at", {"site": f"{rel}:{line}", "n": len(stmts)}), len(conds) == 1, "K55")
        return list(conds.pop())

    def arm_caller(self, q: ArmCaller):
        """U3: an arm its run shows ran, with no guard of its own, ran because a call of its function's ran it.  The
        run's mark holds what that call handed it -- updateUnmergedJobs < PandaID=... > is the merge job's, not the
        row's -- and each call ``called_from`` lists hands the mark's parameter a name of the caller's (the map's
        ``call_bindings``): a run of the caller marked by that name holds the same value.  Those runs are asked over the
        arm's run on its machine; the innermost holding it whole is the call that ran it, and the call's guard held
        there (K54): where its values came from is read in that run, as H3 reads an arm's (KNOWN_ISSUES.md K56)."""
        self.start("U3", "which call ran the arm")
        rel, func = q.owner.split("::", 1)
        run = q.run
        ident = self.need(T.Said("the-arm-s-run-mark", {"fn": func, "token": run.token}), mark_identities(q.owner, q.line, run), "K56")
        self.say(f"{func}'s run {run.token!r}: its mark holds {ident}")
        writers = asyncio.run(self._writers_of(q.asked.subject))
        sites = list(dict.fromkeys(s for e in _entries_of(writers, q.owner) if (s := _site(e.split(" -> ")[-1]))))
        tied = self._tie_calls(q.owner, ident, sites)
        self.need(T.Said("a-call-handing-the-mark-on", {"fn": func, "sites": sites}), tied, "K56")
        files = self._tied_files(tied)
        to_hand = self.need(T.Said("a-log-of-to-hand", {"func": func, "files": files}), self.logs.names(files))
        holding, found, missed = self._callers_holding(tied, run, to_hand)
        self._stop_unanswered(missed)
        self.need(T.Said("a-caller-s-run-holding-the-arm-s", {"fn": func, "callers": sorted({c.split("::")[-1] for c, _ in tied}),
                                                              "start": run.first, "end": run.last, "n": len(found)}), holding, "K56")
        cowner, _, crun = max(holding, key=lambda h: h[2].first)
        lines = sorted({line for c, line, r in holding if c == cowner and r is crun})
        crel, cname = cowner.split("::", 1)
        self.need(T.Said("which-call-of-the-caller", {"caller": cname, "fn": func, "lines": lines}), len(lines) == 1, "K56")
        line = lines[0]
        self.say(f"{func}:{q.line} ran in {run.describe()}, inside {cname}'s call at {line}: {crun.describe()}")
        guards = self._call_guards(crel, line)
        self.say(f"guards of {cname}:{line}: {guards}")
        self.need(T.Said("the-call-has-no-guard", {"caller": cname, "line": line, "fn": func}), guards, "K56")
        cfn = R.function(crel, cname, [line])
        at = R.statement_at(cfn.node, line)
        stmt = self.need(T.Said("one-call-statement-at", {"site": f"{crel}:{line}", "n": 0 if at is None else 1}),
                         at if at is not None and pathcond.arm_condition(at) == guards else None, "K56")
        held_on, read_false = self._held_on(cfn, stmt, guards, crun)
        what = (T.Said("printed-values-read-false", {"guard": read_false[0], "on": read_false[1]}) if read_false is not None
                else T.Said("values-held-on", {"held": held_on}))
        self.need(T.Said("the-call-that-ran-it", {"fn": func, "line": q.line, "caller": cname, "call_line": line, "what": what}), None, "K56")

    def _tie_calls(self, owner: str, ident: list[tuple[str, str]], sites: list[str]) -> dict[tuple[str, int], dict[str, tuple[str, ...]]]:
        """Each call at *sites* (``file:line``) into *owner* whose caller's run marks hold the row the arm's run mark
        holds (*ident*: the class and the value of its identity, :func:`mark_identities`), in a hole of the same
        identity column (:func:`key_marks`): ``{(caller, call line): {value: the caller's marks holding it}}`` (U3,
        F1).  Not by the name the call hands: ``getEventRanges`` marks ``pandaID`` and hands ``tmpJobSpec``, whose
        ``PandaID`` it set from it."""
        tied: dict[tuple[str, int], dict[str, tuple[str, ...]]] = {}
        for site in sites:
            crel, line = site.rsplit(":", 1)[0], int(site.rsplit(":", 1)[1])
            node = R.def_holding(crel, line)
            if node is None:
                continue
            cowner = f"{crel}::{node.name}"
            for cls, value in ident:
                marks = key_marks(cowner, line, cls, value)
                self.say(f"  {site} ({node.name}): its run marks holding {cls} {value}: {list(marks)}")
                if marks:
                    tied.setdefault((cowner, line), {})[value] = marks
        return tied

    def _tied_files(self, tied: dict[tuple[str, int], dict[str, tuple[str, ...]]]) -> list[str]:
        """The logs the tied callers print their calls' rows to."""
        return sorted({f for cowner, line in tied for f in self.files_of(cowner, sorted(
            {f for st in getattr(R._prints_of(cowner, [line]), "statements", None) or [] for f in st.log_files}), [line])})

    def _callers_holding(self, tied: dict[tuple[str, int], dict[str, tuple[str, ...]]], run: Any,
                         to_hand: list[str]) -> tuple[list[tuple[str, int, Any]], list, list]:
        """The tied callers' runs, asked by each value with the marks holding it over *run* on its machine, that hold
        *run* whole -- ``(caller, call line, caller's run)``: where the call ran it, the innermost of them is the call's
        -- with every run found and the asks not answered (U3, F1)."""
        window = (run.first[:19], run.last[:19])
        found, missed = [], []
        for value in sorted({v for marks in tied.values() for v in marks}):
            marks = tuple(sorted({m for by in tied.values() for m in by.get(value, ())}))
            asking = W.Asking(window, run.machine, T.Said("over-the-arm-s-run", {"start": window[0], "end": window[1], "marks": list(marks)}), marks)
            got, unanswered, _ = self._ask(value, to_hand, asking)
            missed += unanswered
            found += [r for runs in got.values() for r in runs]
        holding = [(cowner, line, r) for cowner, line in tied for r in found
                   if of_mover(r, cowner) and r.machine == run.machine and r.first <= run.first and r.last >= run.last]
        return holding, found, missed

    def _caller_absent(self, c: Caller, up: Callers, to_hand: list[str], asking: W.Asking, holes: dict, moving: list | None = None):
        """U2 where no pass of the caller *c* took the row in the window: judgement 3 for it -- it did not run on the row
        only where the window was searched whole, every log of it was asked and every line of its runs holds the key
        (its marks do) -- then up once to the callers ``called_from`` lists for it (hop 1's, at hop 2); else, where its
        passes that moved the row (*moving*) came from an agent's cycles, why they ran when they did ((C),
        :meth:`_agent_cycles`); else it stops."""
        func = c.owner.split("::")[-1]
        start, end = up.window
        premises: dict[str, dict[str, Any]] = {"window": self._searched_whole(to_hand, asking, asking, [], holes, None),
                    "logs": every_log_asked(c.owner, c.lines, to_hand),
                    "prints": {"held": True, "why": T.Said("caller-marks-hold-the-key", {"marks": list(c.marks)})} if c.marks
                    else {"held": None, "why": T.Said("caller-asked-by-the-key-alone")}}
        failing = [T.Said("premise-why", {"premise": k, "why": p["why"]}) for k, p in premises.items() if not p["held"]]
        if premises["window"]["held"] and premises["logs"]["held"] and len(c.lines) > 1:
            # P9 (the investigator's u9): a call every way to which prints the row's key, none of its lines holding the
            # key in the window searched whole, was not the call -- copyArchive's 'reassignJobs for ... ([PandaIDs])'
            # before its Client.kill_jobs calls, the rebrokerage's killJobs at 1179 printing none
            off = self._calls_printing_key(c, self._key_columns(up.subject, up.key))
            left = [line for line in c.lines if line not in off]
            if off and left:
                self.say(f"  {func}'s calls at {off} print the row's key on every way to them, and no line of its "
                         f"holding the key is in the window searched whole: none of them is the call; it is at {left} (P9)")
        if [k for k, p in premises.items() if not p["held"]] == ["prints"] and self._keyed_turn(c, up, moving):
            return None
        self.need(T.Said("a-pass-of-the-caller-on-the-row", {"caller": func, "start": start, "end": end, "failing": failing}),
                  not failing, "K55")
        self.say(f"{func} did not run on the row from {start} to {end}: {'; '.join(T.text(p['why']) for p in premises.values())}")
        sites = {f"{c.owner.partition('::')[0]}:{line}" for line in c.lines}
        outer = [e.split(" -> ")[0] for e in up.entries if " -> " in e and _site(e.split(" -> ")[1]) in sites]
        if c.hop == 1 and up.up is None and outer:
            self.say(f"up to {func}'s callers: {outer}")
            return Callers(c.owner, up.arms, up.subject, up.key, up.window, up.kind, outer, up.before, up=c.owner,
                           after=up.after, left=up.left)
        if moving and up.before is not None and self._agent_cycles(c, up, moving, absent=True):
            return None
        self.need(T.Said("why-the-caller-ran-on-the-row-only", {"caller": func, "start": start, "end": end, "mover": up.mover.split("::")[-1],
                                                                 "hop": c.hop}), None, "K55")

    def _calls_printing_key(self, c: Caller, keys: set[str]) -> list[int]:
        """P9: the call lines of *c* every way to which prints a hole the map names a column of *keys* by (a list of
        them too: ``listed``), as the judge reads a must row (:func:`rows.must_rows`, before the call)."""
        fragment = mapdump.fragment()
        rel, _, func = _bare(c.owner).partition("::")
        if fragment is None or not keys:
            return []
        try:
            fn = R.function(rel, func, c.lines)
        except (LookupError, ValueError, SyntaxError):
            return []
        holes = _key_hole_lines(id(fragment)).get((rel.rpartition(":")[2], func.rpartition(".")[2]), {})
        out = []
        for line in c.lines:
            stmt = R.statement_at(fn.node, line)
            if stmt is None:
                continue
            before, _ = R.must_rows(fn.node, stmt, None)
            if any(holes.get(row.lineno, set()) & keys for row in before):
                out.append(line)
        return out

    def _keyed_turn(self, c: Caller, up: Callers, moving: list | None) -> bool:
        """M3(a)-2 step 2 (out/gm/qk-design.md "M3(a)-2", tmaeno's OK 2026-11-15 09:34:56): where the caller *c* was
        asked by the key alone and its runs' lines without the key are not seen, but the late question's statement it
        selects with has no cap and its loop over the rows prints each row's key before any exit
        (:meth:`_row_key_print`): from the statement's turn (:meth:`_turn`) every pass of it that reaches the
        statement selects the row and prints the key, so none printing it there says none reached it.  Late: its
        first pass after the turn moved it on -- an END and True; stuck: none to the window's end stops it.  False,
        said why, where it does not read -- copyArchive main's 345 ``jobStatus='sent' AND modificationTime<now-30 min``
        and 354 ``Sent Watcher {id}``."""
        q = getattr(self, "_missed_q", None)
        if q is None or q.subject != up.subject:
            return False
        sts = [st for st in q.statements if _bare(st.owner) == _bare(c.owner) and st.result != RESULT_AGGREGATE
               and AG.cap_rows(st.limits) is None]
        if len(sts) != 1:
            return False
        st = sts[0]
        func = c.owner.split("::")[-1]
        where = f"{func.rpartition('.')[2]}@{st.line}"
        line = self._row_key_print(c.owner, st, up.subject)
        if line is None:
            self.say(f"{where} returns every row it holds (no cap), but no print in its loop over the rows names the row's "
                     "key before any exit, by the map: a pass reaching it is not told by the key (M3(a)-2)")
            return False
        end = up.window[1]
        first = min((p.first for p in moving or []), default=None)
        # late: the statement returns the row all the way from its turn to the moving pass; stuck, to the window's end
        turn, assumed, printed = self._turn(st, q, W.when(first if up.before is not None and first else end))
        if turn is None:
            self.say(f"{where}: when it began to return the row up to {first or end} is not read (M3(a)-2)")
            return False
        common = {"caller": func, "where": where, "turn": f"{turn:%Y-%m-%d %H:%M:%S}", "line": line}
        stopped = T.Said("a-pass-stopped-before-it-not-told", {"where": where})
        if up.before is None:
            self.need(T.Said("a-pass-reaching-it-after-its-turn", {
                **common, "end": end, "premises": [stopped, *self._held_premises(assumed, printed, end)]}), None, "K55")
        if first is None or W.when(first) < turn:
            self.say(f"{where}: the pass that moved it on ({first}) is not after its turn {turn:%Y-%m-%d %H:%M:%S} (M3(a)-2)")
            return False
        self.end(T.Said("ran-at-the-first-pass-reaching-it-after-its-turn", {
            **common, "first": first[:23], "premises": [stopped, *self._held_premises(assumed, printed, first)]}))
        return True

    def _row_key_print(self, owner: str, st, subject: str, under_ifs: bool = False) -> int | None:
        """The line of a print in the statement *st*'s loop over its rows (its ``rows_handled``) whose hole the map says
        holds the row's key, a statement of the loop's own body with none of the statement's exits at or before it: a
        pass prints it for each row it selects.  None where there is none.  *under_ifs* (bundle A, A-5): a print
        under an ``if`` of the loop's body too, which a pass prints for a row it selects and keeps (its absence says
        nothing): prepareTasksToBeFinished_JEDI's 3768 ``got jediTaskID=... finished for early avalanche``.  For it a
        fall (``fall@<line>``, the map's part B 5197de0: an ``if`` whose other way ends the row's turn handing nothing on)
        is no exit: a print past it is printed for a row handed on -- read as an exit, sqlEA's falls at 3755 and 3764
        hid its 3766 on pin d, and MT-scouting-late lost which statement took the row (measure bD to bE).

        P16 (the map's HOA and RL, tmaeno's OK 2026-11-18 00:26): a later loop the map reads as taking every row one by
        one (``RowLoop.each_row``: getTasksToBeFinished_JEDI's over ``taskStatList``, the copy of its rows) is read as
        the first is, with the exits in it before the print; and a print after the first loop of every element of a
        list it puts each row into (``ListPrint``: copyArchive main's 885 ``str(jobs[iJob:iJob + nJob])`` in the
        slicing ``while``, 864 ``str(jobs)``), its hole holding the list naming the key, is printed for every row the
        pass selects, by the map's reading of every way on from the loop."""
        handled = list(getattr(st, "rows_handled", None) or [])
        if len(handled) != 2:
            return None
        keys = self._key_columns(subject, self.args.key)
        rel, _, name = _bare(owner).partition("::")
        try:
            fn = R.function(rel, name, [handled[0]])
        except (LookupError, ValueError, SyntaxError):
            return None

        def for_at(line: int) -> ast.For | ast.AsyncFor | None:
            return next((n for n in ast.walk(fn.node) if isinstance(n, ast.For | ast.AsyncFor) and n.lineno == line), None)

        loop = for_at(handled[0])
        found = R._prints_of(owner, [handled[0]])
        if loop is None or found is None:
            return None
        site = self._read_site_of(st)
        later = [(x, rl.lines[0]) for rl in getattr(site, "row_loops", None) or []
                 if rl.each_row and rl.lines and rl.lines[0] != handled[0] and (x := for_at(rl.lines[0])) is not None]
        listed = {lp.line: set(lp.holes) for lp in getattr(site, "list_prints", None) or []}
        exits = [e.line for e in getattr(st, "exits", None) or [] if not (under_ifs and getattr(e, "kind", "") == "fall")]

        def names_key(pst, holes: set[str] | None = None) -> bool:
            return any(keys & set(h.columns or []) for r in pst.rows or [] for h in [*(r.holes or []), *(r.prefix_holes or [])]
                       if holes is None or h.expression in holes)

        def in_body(stmt, around) -> bool:
            body = [n for s in around.body for n in ast.walk(s)] if under_ifs else around.body
            return any(stmt is s for s in body) and not (
                under_ifs and any(isinstance(a, R.LOOPS) and a is not around for a in pathcond.ancestors(stmt)
                                  if any(a is n for n in ast.walk(around))))

        for pst in getattr(found, "statements", None) or []:
            if pst.line in listed:
                if names_key(pst, listed[pst.line]):
                    return int(pst.line)
                continue
            if not handled[0] < pst.line <= handled[1] or not names_key(pst):
                continue
            stmt = R.statement_at(fn.node, pst.line)
            if stmt is None:
                continue
            if not any(x <= pst.line for x in exits) and in_body(stmt, loop):
                return int(pst.line)
            if any(start < pst.line <= (x.end_lineno or start) and not any(start <= e <= pst.line for e in exits)
                   and in_body(stmt, x) for x, start in later):
                return int(pst.line)
        return None

    def _read_site_of(self, st):
        """The map's read site of the statement reading *st* (``ReadSiteNode`` by its owner and line); None where the
        map has none."""
        owner, line = getattr(st, "owner", ""), getattr(st, "line", 0)
        return next((r for r in getattr(mapdump.fragment(), "read_sites", None) or []
                     if r.owner == owner and r.anchor.line_start == line), None)

    def _together_lines(self, owner: str, a: int, b: int) -> bool:
        """Whether one turn of *owner* on a row can run its statements at lines *a* and *b* (the map's
        ``flow.together``, the loops around both read as going round once: the row's turn): ``stay_value.tied``'s
        statement tie (W9).  reactivatePendingTasks_JEDI's loop over its tasks runs 4723 for one task and 4728 for
        another in one call, never both for one.  True where either statement is not read, as the tie stood."""
        rel, _, name = _bare(owner).partition("::")
        try:
            fn = R.function(rel, name, [a, b])
        except (LookupError, ValueError, SyntaxError):
            return True
        sa, sb = R.statement_at(fn.node, a), R.statement_at(fn.node, b)
        if sa is None or sb is None:
            return True
        around = {id(x) for x in pathcond.ancestors(sb)}
        once = [x for x in pathcond.ancestors(sa) if isinstance(x, R.LOOPS) and id(x) in around]
        return flow.together(sa, sb, once=once)

    def _took_by(self, q: Missed) -> TookBy | None:
        """Bundle A, A-5 (P25, out/gm/qk-design.md, tmaeno's OK 2026-11-16): which of the mover's selecting statements
        took the row in the pass that moved it on, where it has more than one: the one whose loop over its rows printed
        the row's key in that pass (:meth:`_row_key_print`, under the loop's ifs too) --
        prepareTasksToBeFinished_JEDI's sqlEA (3741) prints ``got jediTaskID=... finished for early avalanche`` or
        ``... for reset in {taskstatus}`` (3768) for a row it takes, its sqlFA (3789) ``... to force avalanche`` (3796).
        ``(the statement, the print's line, the print's text the record matched)``; None where it has one, or not
        exactly one printed the key in that pass (a print under an ``if`` says nothing where it is absent).

        Where none printed it, the others are ruled out by their hand-ons (:meth:`_handed_none`, the P25 reader,
        tmaeno's OK 2026-11-17): the one left took it, the line None and the text why the others did not."""
        own = [st for st in q.statements if st.owner == q.mover]
        if len(own) < 2 or q.run is None:
            return None
        rel, _, name = _bare(q.mover).partition("::")
        try:
            fn = R.function(rel, name, [st.line for st in own])
        except (LookupError, ValueError, SyntaxError):
            return None
        parts, _ = cut_passes(fn, cast(Any, q.run), self.mode)
        begun = [p for p in parts if p.records and W.when(p.records[0].stamp) <= q.moved + datetime.timedelta(seconds=1)]
        if not begun:
            return None
        moving = begun[-1]
        keys = self._key_columns(q.subject, self.args.key)
        found: list[tuple[Any, int, str, Any, Any]] = []
        for st in own:
            line = self._row_key_print(q.mover, st, q.subject, under_ifs=True)
            stmt = R.statement_at(fn.node, line) if line is not None else None
            row = fn.row_at(stmt) if stmt is not None else None
            if row is None or row.line != line:
                continue
            exprs = {h.expression for m in row.map_rows for h in m.holes or [] if keys & set(h.columns or [])}
            for i, _, _ in R.seen(fn, moving, self.mode).hits.get(row.line, ()):
                record = moving.records[i]
                if any(_hole_holds(row, record, SimpleNamespace(part=MUST_PART_MESSAGE, expression=e, bind=""),
                                   self.args.key) for e in exprs):
                    text = next((m.text for m in row.map_rows
                                 if trace.hole_values(m, record.text, record.token, whole=True) is not None), "")
                    text = text[1:] if text.startswith(("f'", "f\"")) else text
                    text = text.strip("'\"")
                    found.append((st, int(line), text, record, row))
                    break
        if found:
            if len({id(st) for st, *_ in found}) != 1:
                return None
            st, line, text, record, row = found[0]
            return TookBy(st, line, text, record=record, row=row)
        ruled = self._handed_none(q, fn, own, moving)
        left = [st for st in own if id(st) not in ruled]
        if len(left) != 1:
            return None
        out = tuple((st.line, ruled[id(st)]) for st in own if id(st) in ruled)
        return TookBy(left[0], None, "; ".join(why for _, why in out), out)

    def _handed_none(self, q: Missed, fn: R.Function, own: list, moving: Any) -> dict[int, str]:
        """The P25 reader (out/gm/qk-design.md, the P25 reader part, tmaeno's OK 2026-11-17 02:2x; the map's
        ``ReadSiteNode.hand_ons``, codemap-xl 5197de0, and ``KeySource.through_hand_ons``, d475eb9): of the mover's
        selecting statements, those that handed no row on to its writes of the subject in the pass that moved it.  A
        statement is one where every source of a write's key it is (the write's ``KeyFrom``, a ``rows`` source read in
        the mover itself) draws the key through hand-ons, and every one of those hand-ons prints the key on every way
        through it in the turn (``HandOn.sure_prints``, its holes holding the key) while the pass shows no such line.
        prepareTasksToBeFinished_JEDI's writes take jediTaskID from 3696's rows through 3702, from 3741's through 3767
        (3768 prints ``got jediTaskID=...`` on every way past it) and from 3789's through 3794 (3796): a pass with
        neither line moved a row 3696 took.  A key filled in the row loop itself, or drawn through no hand-on the map
        reads, rules nothing out.  The sure prints are read for a run that entered no handler: a pass holding a row of
        a handler of the mover's, or a traceback naming it, rules none out.  By ``id`` of the statement, the words."""
        fragment = mapdump.fragment()
        if fragment is None or R.raised_in(fn, cast(Any, moving))[0]:
            return {}
        hits = R.seen(fn, moving, self.mode).hits
        if any(R._handler_of(fn, r.stmt) is not None for r in fn.rows if r.line in hits):
            return {}
        mover = _bare(q.mover)
        writes = [kf for kf in getattr(fragment, "key_froms", None) or [] if _bare(kf.owner) == mover and kf.writes.get(q.subject)]
        sites = {rs.name: rs for rs in getattr(fragment, "read_sites", None) or []}
        out: dict[int, str] = {}
        for st in own:
            drawn = [s for kf in writes for s in kf.sources if s.kind == "rows" and s.site_line == st.line and not s.via]
            if not drawn or any(not getattr(s, "through_hand_ons", None) for s in drawn):
                continue
            site = sites.get(drawn[0].name)
            hand_ons = {h.line: h for h in getattr(site, "hand_ons", None) or []}
            said: list[str] = []
            for line in sorted({h for s in drawn for h in s.through_hand_ons}):
                sure = [p for p in getattr(hand_ons.get(line), "sure_prints", None) or [] if p.holes]
                if not sure or any(self._printed_key(fn, moving, hits, p) for p in sure):
                    break
                said.append(f"{line} (printing {' or '.join(str(p.line) for p in sure)})")
            else:
                out[id(st)] = (f"@{st.line} handed no row on in that pass: the hand-on(s) the write's key is drawn "
                               f"through, {', '.join(said)}, print the key every way through, and the pass shows none")
        return out

    def _printed_key(self, fn: R.Function, moving: Any, hits: dict, p: Any) -> bool:
        """Whether *moving* holds a record of *fn*'s print at ``p.line`` (a ``HandOnPrint``) whose hole of the hand-on's
        key (``p.holes``) holds the question's key."""
        stmt = R.statement_at(fn.node, p.line)
        row = fn.row_at(stmt) if stmt is not None else None
        if row is None:
            return False
        return any(_hole_holds(row, moving.records[i], SimpleNamespace(part=MUST_PART_MESSAGE, expression=e, bind=""),
                               self.args.key)
                   for i, _, _ in hits.get(row.line, ()) for e in p.holes)

    def _own_cycles(self, q: Callers) -> bool:
        """Q1 part 3, P6 (out/gm/qk-design.md, the Q1 design summary, tmaeno's OK 2026-11-15 23:06): a mover that takes
        its rows itself from a worker's list (``self.<field>.get()``: TaskCommando's runImpl) is read by the agent's
        cycles (:meth:`_agent_cycles`) before its callers, whose runs (the thread's run) print no key; the pass that moved
        the row is L1's run.  True where that ENDs."""
        mq = getattr(self, "_missed_q", None)
        if mq is None or mq.mover != q.mover or q.before is None or getattr(mq, "run", None) is None:
            return False
        rel, func = q.mover.split("::", 1)
        lines = sorted({a.line for a in q.arms})
        try:
            fn = R.function(rel, func, lines)
        except (LookupError, ValueError, SyntaxError):
            return False
        if AG._takes(fn.node) is None:
            return False
        self.say(f"{func} takes its rows from a worker's list itself: its agent's cycles are read first (P6)")
        return self._agent_cycles(Caller(q.mover, lines, 0, "none"), q, [mq.run])

    def _agent_cycles(self, c: Caller, up: Callers, moving: list, absent: bool = False) -> bool:
        """(C) (out/gm/qk-design.md): why the passes of the caller *c* that moved the row on (*moving*) ran when they
        did, where the parts its worker is handed come from an agent's cycles (:func:`agents.selection_of`).  The
        agent's lines around its selecting call hold no key: they are asked by the literal they share
        (:func:`agents.ask_strings`) on every machine, from the row's becoming selectable less windows.SLACK to the
        move.  Each pass belongs to its machine's last selection for the row's label before it, where that label's
        cycles there follow one another (one process); the selections between two takes are read against the selecting
        function's lock (:func:`agents.kept_off`), and so are those after a selection that took the row with no pass of
        the worker after it (P17, :meth:`_unrun_takes`).  *absent*: judgement 3 showed the caller ran on the row in the
        window only in *moving* (U2), so such a selection is said to have had none after it; else that is a premise.
        An END and True; False where it does not read, said why (the stop K55 stands); asks no record answers stop."""
        rel, func = c.owner.split("::", 1)
        try:
            fn = R.function(rel, func, c.lines)
        except (LookupError, ValueError, SyntaxError):
            return False
        sel, why = AG.selection_of(c.owner, fn.node.lineno)
        if sel is None:
            self.say(f"(C) not read: {why}")
            return False
        crows = AG.cycle_rows(sel)
        strings = AG.ask_strings(list(crows.rows)) if crows is not None else ()
        files = sorted({f for r in (crows.rows if crows is not None else ()) for f in (r.log_files or [])})
        if crows is None or not strings or not files:
            self.say(f"(C) not read: {sel.agent_name}'s call to {sel.callee} at {sel.rel}:{sel.call} has no rows around it to ask its cycles by")
            return False
        self.say(f"(C) {func} is handed parts of {sel.worker.split('::')[-1]}.{sel.field}, made at {sel.rel}:{sel.made} of what "
                 f"{sel.agent_name} got from {sel.callee} at {sel.call}")
        since = up.window[0]
        assert up.before is not None
        asking = W.Asking((W._down(W.when(since) - W.SLACK), up.before[:19]), None,
                          T.Said("for-an-agent-s-cycles", {"agent": sel.agent_name, "callee": sel.callee, "strings": list(strings)}),
                          strings=strings)
        unanswered: list = []
        window = self.window(asking)
        picks: dict[str, list] = {}
        spans: dict[str, dict] = {}
        n_lines = 0
        for name in self.logs.names(files):
            try:
                records = self.logs.log(name, self.args.key, window, machine=None, strings=strings).records
            except Unrecorded as e:
                unanswered += [(ask, why or T.text(asking.why)) for ask, why in e.asks]
                continue
            n_lines += len(records)
            got, at = AG.selections(records, crows, self.mode, name)
            for m, qs in got.items():
                picks.setdefault(m, []).extend(qs)
                spans.setdefault(m, {}).update(at[m])
        self._stop_unanswered(unanswered)
        for qs in picks.values():
            qs.sort(key=lambda q: q.at)
        if not picks:
            self.say(f"(C) not read: of {n_lines} line(s) of {sel.agent_name}'s, none is a cycle's selection under a mark "
                     "(its logger writes no mark there, or none was searched)")
            return False
        label = self._label_of(up.subject, crows.label)
        takes = []
        for p in sorted(moving, key=lambda p: p.first):
            m = p.machine
            chain = [q for q in picks.get(m, []) if not label or q.label == label]
            pick, why = AG.batch_of(chain, p.first, label, since)
            if pick is None:
                self.say(f"(C) not read: the pass on {m.split('.')[0]} from {p.first}: {why}")
                return False
            # a selection whose label line lies outside what was searched may be the label's
            unknown = [q for q in picks.get(m, []) if label and not q.label and pick.at < q.at <= p.first]
            if unknown:
                self.say(f"(C) not read: on {m.split('.')[0]}, {len(unknown)} selection(s) from {pick.at} to {p.first} print "
                         "no label in what was searched: one may be the row's")
                return False
            if not AG.one_process(chain, spans.get(m, {})):
                # C-3 (bundle C, W3; out/gm/qk-design.md, tmaeno's choice (b) 2026-11-17): with no label on the lines,
                # agents.cap_take reads each process's width in turn as the row's and keeps the one the cap does not
                # contradict -- NT-registered-late's aipanda081, where TaskRefiner's 4-label process took the row with a
                # full 999 and its 1-label one returned 1 row just before the pass
                if self._cap_take(c, sel, picks, label, moving, up):
                    return True
                self.say(f"(C) not read: {sel.agent_name}'s cycles{' for ' + label if label else ''} on {m.split('.')[0]} overlap: "
                         "more than one process selects there, and which one took the row is not told by its lines")
                return False
            takes.append((pick, p))
        # a selection that took the row with no pass on it after (P17): its lock keeps the others off as a take's does
        lock = AG.lock_of(sel)
        unrun = self._unrun_takes(sel, picks, spans, label, up, lock, takes)
        takes += [(pick, None) for pick, _ in unrun]
        # the selections that could have taken the row before the first take did and did not: nothing here says why
        first = min((pick for pick, _ in takes), key=lambda q: q.at)
        passed = AG.passed_over(picks, first, label, since)
        refused: T.Said | str = ""
        refused_premises: list[T.Said] = []
        left_unknown: list[T.Said] = []
        if passed:
            if self._cap_take(c, sel, picks, label, moving, up):
                return True
            refused, refused_premises = self._passed_refused(sel, passed, up)
            if not refused:
                refused, refused_premises, left_unknown = self._passed_left(sel, passed, up, label)
            if not refused:
                self.say(f"(C) not read: {sel.agent_name}'s cycles took the row first at {first.at} ({first.machine.split('.')[0]}), and "
                         f"the {len(passed)} selection(s) on {len({q.machine for q in passed})} machine(s) from {since} to "
                         f"{max(first.after or first.at, since)} did not take it: why ({sel.callee}'s lock by another, its count's "
                         "cap, a bound on the key) is not read")
                return False
        host = self._move_host(up.before)
        left = {id(pick): w for pick, w in unrun}
        said = [T.Said("an-agent-s-take", {"machine": pick.machine.split(".")[0], "mark": pick.mark, "callee": sel.callee,
                                           "after": max(pick.after or pick.at, since)[:23], "at": pick.at[:23], "count": pick.count,
                                           "first": p.first[:23],
                                           "moved": T.Said("and-it-moved-it-on", {}) if host is not None and p.machine == host else ""})
                if p is not None else self._unrun_said(sel, func, pick, left[id(pick)], moving, up)
                for pick, p in sorted(takes, key=lambda t: t[0].at)]
        kept: list[T.Said] = []
        ordered = sorted(takes, key=lambda t: t[0].at)
        read = {id(q) for q in passed} | {id(pick) for pick, _ in takes}
        for (take, _), (nxt, _) in zip(ordered, ordered[1:], strict=False):
            between = [q for qs in picks.values() for q in qs if take.at < q.at < (nxt.after or nxt.at)
                       and (not label or q.label == label) and q is not nxt]
            read |= {id(q) for q in between}
            if lock is None:
                self.say(f"(C) {len(between)} selection(s) from {take.at} to {nxt.after or nxt.at} did not take the row; "
                         f"{sel.callee}'s lock is not read")
                return False
            ok, why = AG.kept_off(take, nxt, between, lock, since)
            if not ok:
                self.say(f"(C) not read: the selections between the takes at {take.at} and {nxt.at}: {why}")
                return False
            kept.append(T.Said("kept-off-by-the-lock", {"n": len(between), "machines": len({q.machine for q in between}),
                                                        "start": take.at[:23], "end": (nxt.after or nxt.at)[:23], "callee": sel.callee,
                                                        "span": str(lock.span), "term": lock.term, "bound": lock.bound, "sets": lock.sets}))
        premises = [T.Said("one-process-a-label", {"agent": sel.agent_name, "label": label or "every label",
                                                  "machines": sorted({pick.machine.split(".")[0] for pick, _ in takes})})]
        premises += refused_premises
        # a selection that ran while a take did is neither before the first take nor between two: the take came first
        during = [q for qs in picks.values() for q in qs if id(q) not in read and (not label or q.label == label)
                  and q.at[:23] >= since[:23]
                  and any((q.after or q.at)[:23] <= t.at[:23] and q.at[:23] >= (t.after or t.at)[:23] for t, _ in takes)]
        if during:
            premises.append(T.Said("ran-while-a-take-did", {"items": [
                T.Said("a-selection-s-span", {"machine": q.machine.split(".")[0], "start": (q.after or q.at)[:23], "end": q.at[:23]})
                for q in sorted(during, key=lambda q: q.at)]}))
        not_known: list[T.Said] = list(left_unknown)
        for pick, w in unrun:
            premises.append(T.Said("a-cycle-s-lock-on-the-row" if w.how != TL.RAN else "a-cycle-s-lock-changed-it", {
                "machine": w.machine, "callee": sel.callee, "after": (pick.after or pick.at)[:23], "at": pick.at[:23],
                "sets": lock.sets if lock is not None else "", "where": w.where}))
            if not absent:
                premises.append(T.Said("no-pass-after-a-cycle-s-take", {"caller": func, "machine": w.machine}))
            not_known.append(T.Said("why-a-cycle-left-it", {"machine": w.machine, "files": files, "at": pick.at[:19]}))
        if sel.others:
            premises.append(T.Said("another-way-in-not-read", {"worker": sel.worker.split("::")[-1], "at": list(sel.others)}))
        self.end(T.Said("ran-when-an-agent-s-cycle-took-it", {"caller": func, "agent": sel.agent_name, "takes": said, "kept": kept,
                                                             "refused": refused, "premises": premises, "unknown": not_known}))
        return True

    def _unrun_takes(self, sel, picks: dict, spans: dict, label: str, up: Callers, lock, takes: list) -> list:
        """Q1 P17 (out/gm/qk-design.md, the Q1 design summary, tmaeno's OK 2026-11-15 23:06): the agent's selections
        that took the row with no pass of the worker after: a write of the column the selecting function's lock sets
        (C4, *lock*) by that function itself, read on the column's timeline (:meth:`_timeline`) inside one of its
        machine's selections for the row's label that began after the row could be selected (the window's start) and
        before the move -- NT-prepared-late's aipanda088 07:09:15.636 .. .809, whose select print named the row among
        its targets (:meth:`_column_writes`).  ``(pick, write)`` each, not one of *takes*; none where the lock or the
        timeline is not read, or where that machine's cycles for the label overlap (more than one process: which one
        selected is not told)."""
        q = getattr(self, "_missed_q", None)
        if lock is None or q is None or q.subject != up.subject or up.before is None:
            return []
        column = lock.sets.partition("=")[0].strip()
        tl = self._timeline(f"{up.subject.partition('.')[0]}.{column}", q)
        if tl is None:
            return []
        since = up.window[0]
        mine = {id(pick) for pick, _ in takes}
        out = []
        for w in tl.writes:
            if w.at >= W.when(up.before) or not any(a.rpartition(":")[0].rpartition(".")[2] == sel.callee for a in w.arms):
                continue
            for m, qs in picks.items():
                if m.split(".")[0] != w.machine:
                    continue
                chain = [p for p in qs if not label or p.label == label]
                # a selection before the row could be selected did not return it (NT-breaking-late's 03:58:05 named it
                # among its targets while its status was another)
                pick = next((p for p in chain if W.when(since) <= W.when(p.after or p.at) <= w.at <= W.when(p.at)), None)
                if pick is not None and id(pick) not in mine and AG.one_process(chain, spans.get(m, {})):
                    out.append((pick, w))
                    mine.add(id(pick))
        return out

    def _unrun_said(self, sel, func: str, pick, w: TL.Write, moving: list, up: Callers) -> T.Said:
        """The words for a selection that took the row with no pass on it after (:meth:`_unrun_takes`): until the next
        pass on its machine that moved it on, or the move."""
        after = [p.first for p in moving if p.machine.split(".")[0] == w.machine and W.when(p.first) > w.at]
        return T.Said("an-agent-s-unrun-take", {"machine": w.machine, "mark": pick.mark, "callee": sel.callee,
                                                "after": (pick.after or pick.at)[:23], "at": pick.at[:23], "count": pick.count,
                                                "where": w.where, "caller": func,
                                                "until": min(after)[:23] if after else (up.before or "")[:23]})

    def _passed_refused(self, sel, passed: list, up: Callers) -> tuple[T.Said | str, list[T.Said]]:
        """Q1 part 3, P5: why the agent's selections *passed* (:func:`agents.passed_over`) did not take the row, read on
        the selecting statement's terms at each one's time: every statement of the late question selecting through the
        agent's call (*sel*'s callee) has, for each selection, a term every build requires that is surely false from the
        selection's start to its count (:meth:`_false_between`) -- NT-tobroken's ``modificationtime<now-5 min`` on the
        stay's start until 20:34:16.  The arms no record reads are named as having not written the columns
        (``_assuming``, as a turn names them).  ``(said, premises)``; ``("", [])`` where one selection is not so read."""
        q = getattr(self, "_missed_q", None)
        if q is None or q.subject != up.subject:
            return "", []
        sts = [self._filled(st, q) for st in q.statements if st.owner.rpartition("::")[2].rpartition(".")[2] == sel.callee]
        if not sts:
            return "", []
        kept = getattr(self, "_assuming", None)
        kept_used = self._premised_used
        self._assuming, self._premised_used = {}, {}
        terms: set[str] = set()
        try:
            for p in passed:
                a, b = W.when(p.after or p.at), W.when(p.at)
                for st in sts:
                    why = self._false_between(st, q, a, b)
                    if not why:
                        return "", []
                    terms.add(why)
        finally:
            assumed, self._assuming = self._assuming or {}, kept
            used, self._premised_used = self._premised_used or {}, kept_used
        said = T.Said("passed-over-refused", {
            "n": len(passed), "machines": len({p.machine for p in passed}), "start": min(p.after or p.at for p in passed)[:23],
            "end": max(p.at for p in passed)[:23], "callee": sel.callee, "terms": sorted(terms)})
        return said, [*self._held_premises({k: v for k, v in assumed.items() if v}, {}, max(p.at for p in passed)),
                      *_none_premises(used)]

    def _passed_left(self, sel, passed: list, up: Callers, label: str) -> tuple[T.Said | str, list[T.Said], list[T.Said]]:
        """Q1 batch 2 part 2 (out/gm/qk-design.md, the Q1 batch 2 design, tmaeno's OK 2026-11-16 08:49, with the later steps): why
        the agent's selections *passed* did not take the row, where P5 (:meth:`_passed_refused`) does not read every
        one.  Each is read on what could have left the row out of it: the selecting statement's terms every build
        requires (surely false then: refused, P5; surely true: not it), its terms no tree holds (:func:`key_terms`:
        another table's column, the values the agent hands the call), its cap, and the ways its later steps drop a row
        the SELECT returned (:func:`later_steps`) -- a way printing a line naming the row is read in the records: found
        in the selection's span on its machine, it is why; searched and found in none, not it.  What is left is (ii),
        what no log prints, or stops the reading: (i) a log the records lack (its asks), (iii) what is not read here
        (said).  ``(said, premises, unknown)``: an annotated END's words where each selection is refused, shown, or
        left to (ii) alone; ``("", [], [])`` where not."""
        q = getattr(self, "_missed_q", None)
        if q is None or q.subject != up.subject:
            return "", [], []
        sts = [st for st in q.statements if st.owner.rpartition("::")[2].rpartition(".")[2] == sel.callee]
        if len(sts) != 1:
            self.say(f"(C) part 2: {len(sts)} statement(s) of the question select through {sel.callee}: not read")
            return "", [], []
        st = self._filled(sts[0], q)
        terms = key_terms(getattr(mapdump.fragment(), "key_froms", None) or [], st.owner, st.line)
        if terms.other or (terms.either and not terms.joined):
            self.say(f"(C) part 2: {sel.callee}'s terms {', '.join(terms.other or terms.either)} are not read")
            return "", [], []
        fixed: dict[str, T.Said] = {}
        premises: list[T.Said] = []
        if terms.joined:
            fixed["joined"] = T.Said("left-by-a-joined-key", {"terms": [t for t, _ in terms.joined] + list(terms.either),
                                                             "sources": [s for _, s in terms.joined]})
        if terms.handed:
            if label:
                premises.append(T.Said("handed-keys-as-the-row-s", {"agent": sel.agent_name, "terms": list(terms.handed),
                                                                    "label": label}))
            else:
                fixed["handed"] = T.Said("left-by-a-handed-key", {"terms": list(terms.handed)})
        if st.limits:
            fixed["cap"] = T.Said("left-by-the-cap", {"limits": list(st.limits)})
        self._attach_exit_prints()
        keys = self._key_columns(q.subject, self.args.key)
        lines = exit_print_lines(st)
        printed = (getattr(R._prints_of(_plain(st.owner), lines), "statements", None) or []) if lines else []
        keyed = {pst.line for pst in printed if any(keys & set(h.columns or []) for r in pst.rows or []
                                                    for h in [*(r.holes or []), *(r.prefix_holes or [])])}
        steps = later_steps(st, keyed.__contains__)
        fixed.update({s.way: T.Said("left-by-a-silent-step", {"way": s.way, "decided": s.decided}) for s in steps if not s.prints})
        found = self._later_prints(st, [s for s in steps if s.prints], passed)
        if found is None:
            return "", [], []
        shown, unprinted = found
        kept, kept_used = getattr(self, "_assuming", None), self._premised_used
        self._assuming, self._premised_used = {}, {}
        refused: dict[str, int] = {}
        left: dict[str, T.Said] = {}
        left_entries: set[str] = set()
        true_in: dict[str, int] = {}
        n_left = 0
        try:
            for p in passed:
                a, b = W.when(p.after or p.at), W.when(p.at)
                if why := self._false_between(st, q, a, b):
                    refused[why] = refused.get(why, 0) + 1
                    continue
                if id(p) in shown:
                    continue
                n_left += 1
                mine = 0
                for reading in st.condition_trees:
                    if reading.entry not in st.conditions:
                        continue
                    kind, said = self._term_left(reading, q, a, b)
                    if kind == "true":
                        true_in[reading.entry] = true_in.get(reading.entry, 0) + 1
                    elif kind == "ii" and isinstance(said, T.Said):
                        left[T.text(said)] = said
                        left_entries.add(reading.entry)
                        mine += 1
                    else:
                        self.say(f"(C) part 2: the selection on {p.machine.split('.')[0]} at {p.at}: {said}")
                        return "", [], []
                if not mine and not fixed:
                    self.say(f"(C) part 2: the selection on {p.machine.split('.')[0]} at {p.at} did not take the row, and "
                             "nothing read here could have left it out")
                    return "", [], []
        finally:
            assumed, self._assuming = self._assuming or {}, kept
            used, self._premised_used = self._premised_used or {}, kept_used
        func = sel.callee
        parts: list[T.Said] = []
        if refused:
            parts.append(T.Said("passed-refused-part", {"n": sum(refused.values()), "callee": func, "terms": sorted(refused)}))
        for s in dict.fromkeys(v[0] for v in shown.values()):
            times = [v[1] for v in shown.values() if v[0] is s]
            parts.append(T.Said("passed-shown-part", {"n": len(times), "way": s.way, "prints": [f"{func}:{n}" for n in s.prints],
                                                      "times": times}))
        excluded = [T.Said("excluded-term-true", {"entry": e}) for e, n in true_in.items() if n == n_left and e not in left_entries]
        excluded += [T.Said("excluded-step-unprinted", {"way": s.way, "prints": [f"{func}:{n}" for n in s.prints]})
                     for s in unprinted]
        if excluded and n_left:
            parts.append(T.Said("passed-excluded-part", {"items": excluded, "rest": "for the rest, " if parts else ""}))
        said = T.Said("passed-over-left", {
            "n": len(passed), "machines": len({p.machine for p in passed}), "start": min(p.after or p.at for p in passed)[:23],
            "end": max(p.at for p in passed)[:23], "parts": parts})
        unknown = [T.Said("which-left-it-out", {"n": n_left, "alternatives": [*left.values(), *fixed.values()]})] if n_left else []
        premises += [*self._held_premises({k: v for k, v in assumed.items() if v}, {}, max(p.at for p in passed)),
                     *_none_premises(used)]
        return said, premises, unknown

    def _attach_exit_prints(self) -> None:
        """Part 2: what each way out of the question's selecting statements prints (``StatementExit.skeleton``), from
        the map's walk of them (:func:`strategy.attach_reader_traces`), once: the dump keeps the exits' places only."""
        strategy = getattr(self, "_strategy", None)
        if strategy is None or self.__dict__.get("_exits_walked"):
            return
        self._exits_walked = True
        strategy_mod.attach_reader_traces(strategy, R.ROOTS)

    def _later_prints(self, st, printing: list[LaterStep], passed: list) -> tuple[dict[int, tuple[LaterStep, str]], list[LaterStep]] | None:
        """Part 2: the selections of *passed* a printing later step dropped the row in -- its print naming the row's
        key in the selection's span on the selection's machine, ``{id(selection): (step, "machine HH:MM:SS.mmm")}`` --
        and the steps found in none.  The function's logs are asked by the key over the span of them all; judgement
        3's window and logs premises must hold (a log the records lack stops on its ask).  None where they do not, or
        a print is not read, said why."""
        if not printing:
            return {}, []
        owner = _plain(st.owner)
        rel, _, func = owner.partition("::")
        lines = sorted({n for s in printing for n in s.prints})
        try:
            fn = R.function(rel, func, lines)
        except (LookupError, ValueError, SyntaxError):
            self.say(f"(C) part 2: {func} is not read at {lines}")
            return None
        rows = {}
        for n in lines:
            stmt = R.statement_at(fn.node, n)
            row = fn.row_at(stmt) if stmt is not None else None
            if row is None:
                self.say(f"(C) part 2: {func}:{n}'s print is not read")
                return None
            rows[n] = row
        today = sorted({f for pst in getattr(R._prints_of(owner, lines), "statements", None) or [] for f in pst.log_files})
        to_hand = self.logs.names(self.files_of(owner, today, lines))
        if not to_hand:
            self.say(f"(C) part 2: no log of {func}'s prints at {lines} to hand")
            return None
        lo = min(W.when(p.after or p.at) for p in passed)
        hi = max(W.when(p.at) for p in passed) + datetime.timedelta(seconds=1)
        asking = W.Asking((W._down(lo), W._down(hi)), None, T.Said("for-the-later-steps", {
            "func": func, "prints": lines, "start": W._down(lo), "end": W._down(hi)}))
        runs, missed, holes = self._ask(self.args.key, to_hand, asking, {}, quiet=True)
        self._stop_unanswered(missed)
        for premise in (self._searched_whole(to_hand, asking, asking, missed, holes, None),
                        every_log_asked(owner, lines, to_hand)):
            if not premise["held"]:
                self.say(f"(C) part 2: {func}'s prints at {lines} are not searched over its selections: {T.text(premise['why'])}")
                return None
        key = re.compile(rf"(?<![0-9]){re.escape(self.args.key)}(?![0-9])")
        hits = []
        for name in to_hand:
            for run in runs.get(name, []):
                sn = R.seen(fn, run, self.mode)
                for n, row in rows.items():
                    for i, _, _ in sn.hits.get(row.line, ()):
                        record = run.records[i]
                        if key.search(record.message):
                            hits.append((W.when(record.stamp), (record.machine or "").split(".")[0], n, record.stamp))
        shown: dict[int, tuple[LaterStep, str]] = {}
        for p in passed:
            a, b, m = W.when(p.after or p.at), W.when(p.at), p.machine.split(".")[0]
            got = next(((n, stamp) for at, mm, n, stamp in sorted(hits) if mm == m and a <= at <= b), None)
            if got is not None:
                step = next(s for s in printing if got[0] in s.prints)
                shown[id(p)] = (step, f"{m} {got[1][11:23]}")
        unprinted = [s for s in printing if not any(v[0] is s for v in shown.values())]
        return shown, unprinted

    def _term_left(self, reading, q: Missed, a: datetime.datetime, b: datetime.datetime) -> tuple[str, T.Said | str]:
        """Part 2: a condition tree of the selecting statement for one selection [*a*, *b*] P5 does not read false:
        ``"true"`` where surely true then (:meth:`_true_spans`); ``"ii"`` where the columns its terms compare were
        written in the stay by arms no line names the row in, or it turned within the span (the SELECT's own moment is
        not printed); ``"i"`` where a log the records lack would say (its asks kept); ``"iii"`` where a term is not read
        here.  With the words."""
        if self._tree_true_between(reading.tree, q, a, b):
            return "true", ""
        arms: set[str] = set()
        reached: set[str] = set()
        for leaf in _tree_leaves(reading.tree):
            nulls = leaf.node in ("null", "not_null")
            clock = leaf.node == "compare" and leaf.settled and bool(leaf.values) and all(
                v.base in UTC_CLOCKS and v.offset_seconds is not None for v in leaf.values)
            if not (nulls or clock):
                if self._truth(leaf, q)[0] is True:
                    continue
                return "iii", f"{reading.entry}: {leaf.text} is not read for the row"
            tl = self._timeline(leaf.subject, q, nulls=nulls)
            if tl is None:
                if not nulls and self._held(leaf.subject, q)[0]:
                    continue
                return "iii", f"{reading.entry}: {leaf.subject} is not read over the stay"
            open_why = self.__dict__.get("_open_why", {}).get((leaf.subject, nulls), {})
            for o in tl.opens:
                w = open_why.get(o.arm, "unread")
                if w == "unasked":
                    return "i", f"{reading.entry}: {o.arm}'s writes of {leaf.subject} are in logs the records lack (asked)"
                if w != "unprinted":
                    return "iii", f"{reading.entry}: {o.arm}'s writes of {leaf.subject} are not read"
                arms.add(o.arm)
            mine = {arm for w in tl.writes if w.how != TL.RAN and w.at <= b for arm in w.arms}
            reached |= mine
            if nulls and tl.first[0] != tl.first[1] and not tl.opens and not mine:
                return "iii", f"{reading.entry}: {leaf.subject}'s state as the stay began is not read"
        why: list[T.Said] = []
        if arms:
            why.append(T.Said("term-open-arms", {"arms": sorted(arms)}))
        if reached:
            why.append(T.Said("term-reached-writes", {"arms": sorted(reached)}))
        if why:
            return "ii", T.Said("left-by-a-term", {"entry": reading.entry, "why": why})
        return "ii", T.Said("left-by-a-term-turning", {"entry": reading.entry})

    def _tree_true_between(self, node, q: Missed, a: datetime.datetime, b: datetime.datetime) -> bool:
        """Part 2: whether a condition tree is surely true for the row at every moment of [*a*, *b*]: ``all`` where
        every item is, ``any`` where one is; a term ``IS NULL`` or ``IS NOT NULL`` on its states' timeline
        (:meth:`_state_between`: inside a reading naming the open arms, as having not written it); any other term where
        :meth:`_true_spans` holds it all the span."""
        if node.node == "all":
            return all(self._tree_true_between(item, q, a, b) for item in node.items)
        if node.node == "any":
            return any(self._tree_true_between(item, q, a, b) for item in node.items)
        if node.node in ("null", "not_null"):
            return self._state_between(node, q, a, b, node.node == "null")
        return any(s <= a and b <= e for s, e in self._true_spans(node, q))

    def _state_between(self, node, q: Missed, a: datetime.datetime, b: datetime.datetime, null: bool) -> bool:
        """Q1 batch 2 part 1: whether the column a term ``IS NULL`` or ``IS NOT NULL`` tests surely is NULL (*null*), or
        surely holds a value, at every moment of [*a*, *b*], on its states' timeline (:meth:`_timeline`, *nulls*).
        Inside a reading that names the open arms (``_assuming``) they are named as having not written it; the arms
        read as writing a value they know only at run time are kept for the premises (``_premised_used``)."""
        tl = self._timeline(node.subject, q, nulls=True)
        if tl is None:
            return False
        assuming = getattr(self, "_assuming", None)
        if not TL.surely_state_between(null, tl.pieces(assume=assuming is not None), a, b):
            return False
        if assuming is not None and (tl.opens or tl.begun_assumed):
            assuming[node.subject] = [o.arm for o in tl.opens]
        used = self._premised_used
        if tl.premised and used is not None:
            used[node.subject] = list(tl.premised)
        return True

    def _false_between(self, st, q: Missed, a: datetime.datetime, b: datetime.datetime) -> str:
        """A term of *st*'s every build (a ``conditions`` entry's tree) surely false for the row at every moment of
        [*a*, *b*], as ``entry: term``; "" where none is."""
        for reading in st.condition_trees:
            if reading.entry not in st.conditions:
                continue
            if why := self._tree_false_between(reading.tree, q, a, b):
                return f"{reading.entry}: {why}"
        return ""

    def _tree_false_between(self, node, q: Missed, a: datetime.datetime, b: datetime.datetime) -> str:
        """Why a condition tree is surely false for the row at every moment of [*a*, *b*]: ``all`` where an item is,
        ``any`` where every item is; a term comparing a column with the clock on its timeline, or on the one value
        M3(a)-1 reads it held; any other term only where it is false all the window (``_leaf``).  "" where not."""
        if node.node == "all":
            return next((why for item in node.items if (why := self._tree_false_between(item, q, a, b))), "")
        if node.node == "any":
            whys = [self._tree_false_between(item, q, a, b) for item in node.items]
            return "; ".join(whys) if whys and all(whys) else ""
        if node.node in ("null", "not_null") and self._state_between(node, q, a, b, node.node == "not_null"):
            return f"{node.text} false then, on its writes the records show"
        if node.node == "compare" and node.settled and node.values and all(
                v.base in UTC_CLOCKS and v.offset_seconds is not None for v in node.values):
            offsets = [datetime.timedelta(seconds=v.offset_seconds) for v in node.values]
            tl = self._timeline(node.subject, q)
            if tl is not None and tl.read():
                assuming = getattr(self, "_assuming", None)
                if assuming is not None and (tl.opens or tl.begun_assumed):
                    assuming[node.subject] = [o.arm for o in tl.opens]
                pieces = tl.pieces(assume=assuming is not None)
                if all(TL.surely_false_between(node.op, off, pieces, a, b) for off in offsets):
                    return f"{node.text} false then, on its writes the records show"
                return ""
            known, held = self._held(node.subject, q)
            if known and held is not None:
                try:
                    turns = [_when(str(held)) - off for off in offsets]
                except ValueError:
                    return ""
                # col < now + off holds from held - off on, col > now + off until it
                if node.op in ("<", "<=") and all(b < turn for turn in turns):
                    return f"{node.text} false until {min(turns)}"
                if node.op in (">", ">=") and all(a > turn for turn in turns):
                    return f"{node.text} false from {max(turns)}"
            return ""
        return self._leaf(node, q) or ""

    def _false_spans(self, node, q: Missed) -> list:
        """Where in [since, moved] a condition tree is surely false for the row, as :meth:`_tree_false_between` reads one
        stretch: ``all`` where some item is, ``any`` where every item is; a term comparing a column with the clock on its
        timeline (its open arms named as having not written it inside ``_assuming``), or before or after the turn of the
        one value M3(a)-1 reads it held; a state term on its states' timeline; any other term where it is false all the
        window (``_leaf``)."""
        whole = [(q.since, q.moved)]
        if node.node in ("any", "all"):
            found = [self._false_spans(item, q) for item in node.items]
            if node.node == "all":
                return _union([span for spans in found for span in spans])
            out = whole
            for spans in found:
                out = _intersect(out, spans)
            return out
        assuming = getattr(self, "_assuming", None)
        if node.node in ("null", "not_null"):
            tl = self._timeline(node.subject, q, nulls=True)
            if tl is None or tl.premised:
                return []
            if assuming is not None and (tl.opens or tl.begun_assumed):
                assuming[node.subject] = [o.arm for o in tl.opens]
            return _intersect(TL.state_spans(node.node == "not_null", tl.pieces(assume=assuming is not None)), whole)
        if node.node == "compare" and node.settled and node.values and all(
                v.base in UTC_CLOCKS and v.offset_seconds is not None for v in node.values):
            offsets = [datetime.timedelta(seconds=v.offset_seconds) for v in node.values]
            tl = self._timeline(node.subject, q)
            if tl is not None and tl.read():
                if assuming is not None and (tl.opens or tl.begun_assumed):
                    assuming[node.subject] = [o.arm for o in tl.opens]
                pieces = tl.pieces(assume=assuming is not None)
                for off in offsets:
                    whole = _intersect(whole, TL.surely_false(node.op, off, pieces))
                return whole
            known, held = self._held(node.subject, q)
            if not known or held is None:
                return []
            try:
                turns = [_when(str(held)) - off for off in offsets]
            except ValueError:
                return []
            # col < now + off holds from held - off on, col > now + off until it
            if node.op in ("<", "<="):
                return _intersect([(q.since, min(turns))], whole)
            if node.op in (">", ">="):
                return _intersect([(max(turns), q.moved)], whole)
            return []
        return whole if self._leaf(node, q) else []

    def _turn_of(self, sts: list, q: Missed) -> tuple[tuple[datetime.datetime, datetime.datetime] | None,
                                                       dict[str, list[str]], list[str]]:
        """L4: the last stretch before the move where every one of *sts* was surely false for the row -- a tree of
        each's every build (a ``conditions`` entry's) false then (:meth:`_false_spans`), its amounts filled (P29) --,
        the arms that may write the columns compared, named as having not, and the trees false all the stretch, by
        their entries; None where there is no such stretch."""
        self._assuming = {}
        trees: list[tuple[str, list]] = []
        try:
            spans = [(q.since, q.moved)]
            for st in sts:
                st = self._filled(st, q)
                mine = [(f"{reading.entry}", self._false_spans(reading.tree, q))
                        for reading in st.condition_trees if reading.entry in st.conditions]
                trees += mine
                spans = _intersect(spans, _union([span for _, found in mine for span in found]))
        finally:
            assumed, self._assuming = self._assuming or {}, None
        before = [s for s in _union(spans) if s[1] < q.moved]
        if not before:
            return None, {}, []
        span = before[-1]
        # the terms that were false all the stretch, to say it by
        words = [entry for entry, found in trees if any(a <= span[0] and span[1] <= b for a, b in found)]
        return span, {col: arms for col, arms in assumed.items() if arms}, words

    def _turn(self, st, q: Missed, at: datetime.datetime | None = None) -> tuple[
            datetime.datetime | None, dict[str, list[str]], dict[str, tuple[datetime.datetime, T.Said]]]:
        """M3(a)-cap: when the statement *st* began to surely return the row (the first of ``_statement_true``'s
        spans; given *at*, the start of the one holding it, the statement returning the row all the way to it), a
        time column M3(a)-1 starts the stay with but cannot keep all of it read at its start: ``(turn,
        {column: its arms that may write it in the stay, named as having not}, {column: (until when a run of the row
        printed it at the stay's start, what says so)})`` -- the last M3(a)-2's (:meth:`_printed_held`); None where it
        never surely does.  Its keys (``vo``, ``prodSourceLabel``: the agent's loop chooses them each selection) are
        read as the row's: the selections read are the ones made with the row's values (:func:`agents.cap_take`)."""
        self._assuming = {}
        self._printed_read: dict[str, tuple[datetime.datetime, T.Said]] | None = {}
        try:
            spans = self._statement_true(st, q, keyed=True)
        finally:
            assumed, self._assuming = self._assuming or {}, None
            printed, self._printed_read = self._printed_read or {}, None
        if at is not None:
            return next((s for s, e in spans if s <= at <= e), None), assumed, printed
        return (min(s for s, _ in spans) if spans else None), assumed, printed

    def _held_premises(self, assumed: dict[str, list[str]], printed: dict[str, tuple[datetime.datetime, T.Said]],
                       until: str) -> list[T.Said]:
        """The premises of a turn's time columns (:meth:`_turn`) up to *until*, the selection that took the row (the
        window's end, stuck): a column a run of the row printed at the stay's start at or after it is said held so
        (M3(a)-2); else its arms that may write it are named as having not, the print, earlier, said too."""
        out: list[T.Said] = []
        timelines = {k[0]: tl for k, tl in self.__dict__.get("_timelines", {}).items() if tl is not None}
        for col, arms in assumed.items():
            tl = timelines.get(col)
            if tl is not None and tl.begun_assumed:
                out.append(T.Said("none-began-the-stay", {"arms": list(tl.begun_assumed)}))
            if not arms:
                continue
            if tl is not None and tl.read():
                out.append(T.Said("none-of-wrote-it-besides", {"column": col, "arms": arms, "n": len(tl.writes)}))
                continue
            got = printed.get(col)
            if got is not None and got[0] >= W.when(until):
                out.append(got[1])
                continue
            out.append(T.Said("held-from-the-stay-s-start", {"column": col, "arms": arms}))
            if got is not None:
                out.append(got[1])
        return out

    def _printed_held(self, subject: str, q: Missed, found: SV.Before, start: str) -> tuple[datetime.datetime, T.Said] | None:
        """M3(a)-2 step 1 (out/gm/qk-design.md "M3(a)-2", tmaeno's OK 2026-11-15 09:34:56): ``(until when, what says
        so)`` the time column *subject* held *start*, the time of the statuslog row that began the stay, where every
        arm M3(a)-1 leaves as may writing it in the stay (*found*'s ``left``) writes the clock (``attr_time.NOW``) and
        a run of the row printed it in the stay: a write there leaves a later value, so a print of *start* says none
        came before it.  The print is a hole of the map's prints reading ``.attr`` of an object every write of which
        fills it from the row by its key (:func:`which_ran.printed_row_column`, K71's reading) -- Watcher.run's
        ``lastmod:{str(job.modificationTime)}`` at 78, MJ-sent-late's 09:32:01 at 10:32:14,097; its runs asked by the
        key over the stay.  None where an arm writes something else, no such print ran in the stay, or one printed
        another value (said)."""
        cache: dict[tuple, tuple[datetime.datetime, T.Said] | None] = self.__dict__.setdefault("_printed_helds", {})
        ckey = (subject, start, q.moved)
        if ckey in cache:
            return cache[ckey]
        cache[ckey] = None
        attr = subject.partition(".")[2]
        left = list(found.left)
        outcomes = {SV._name(j.owner, b.line): b.outcome for j in self._subject_writers(subject) for b in j.branches}
        if not left or any(outcomes.get(n) not in SV.NOW for n in left):
            return None
        keys = {c.partition(".")[2].lower() for c in self._key_columns(q.subject, self.args.key)}
        spelled = re.compile(rf"\.{re.escape(attr)}\b")
        since, moved = W.when(start), q.moved
        seen: list[tuple[datetime.datetime, str, str, str]] = []
        for f in getattr(mapdump.fragment(), "function_prints", None) or []:
            for st in f.statements or []:
                if not any(subject in (h.columns or []) or spelled.search(h.expression or "")
                           for r in st.rows or [] for h in [*(r.holes or []), *(r.prefix_holes or [])]):
                    continue
                rel, _, func = _bare(f.owner).partition("::")
                where = f"{func.rpartition('.')[2]}:{st.line}"
                try:
                    fn = R.function(rel, func, [st.line])
                except (LookupError, ValueError, SyntaxError):
                    continue
                stmt = R.statement_at(fn.node, st.line)
                row = fn.row_at(stmt) if stmt is not None else None
                if row is None or not printed_row_column(fn, st.line, attr, keys):
                    continue
                files = self.files_of(f.owner, list(st.log_files or []), [st.line])
                to_hand = self.logs.names(files)
                asking = W.Asking((start[:19], f"{moved:%Y-%m-%d %H:%M:%S}"), None, T.Said("for-a-print-of-the-column", {
                    "start": start[:19], "end": f"{moved:%Y-%m-%d %H:%M:%S}", "where": where, "column": attr}))
                self.say(f"{where} prints {attr} of the row (M3(a)-2): logs {files}, asked {T.text(asking.why)}")
                runs, missed, _holes = self._ask(self.args.key, to_hand, asking)
                if missed:
                    self._keep_unanswered(missed)
                    continue
                for name in to_hand:
                    for run in runs.get(name, []):
                        if not of_mover(run, f.owner):
                            continue
                        for p in passes_of(fn, run, self.mode)[0]:
                            sn = R.claim([R.seen(fn, p, self.mode)])[0]
                            for i, _, _ in sn.hits.get(row.line, ()):
                                for expr, value in row.capture(p.records[i]).items():
                                    if spelled.search(expr):
                                        seen.append((W.when(p.records[i].stamp), value.strip().strip("'\""), where, p.machine))
        inside = sorted(s for s in seen if since <= s[0] <= moved)
        if not inside:
            return None
        if other := next((s for s in inside if s[1][:19] != start[:19]), None):
            self.say(f"{other[2]} printed {attr} {other[1]!r} at {other[0]:%Y-%m-%d %H:%M:%S.%f}, not the stay's start "
                     f"{start[:19]}: not closed by the print (M3(a)-2)")
            return None
        at, value, where, machine = inside[-1]
        cache[ckey] = (at, T.Said("held-as-printed", {"column": subject, "value": start[:19], "where": where,
                                                      "at": f"{at:%Y-%m-%d %H:%M:%S.%f}"[:23], "machine": machine.split(".")[0],
                                                      "arms": left}))
        return cache[ckey]

    def _cap_take(self, c: Caller, sel, picks: dict, label: str, moving: list, up: Callers) -> bool:
        """M3(a)-cap (out/gm/qk-design.md, tmaeno's OK 2026-11-15 01:16:35, its build 01:46:09): where the late
        question's statement of the selecting function returns at most a cap of rows, the first selection after the
        statement's turn that returned fewer took every row selectable then -- the row among them -- where it is the
        last of the moving pass's machine before the pass; those before it returned the cap or none.  An END and True;
        False, said why, where it does not read."""
        q = getattr(self, "_missed_q", None)
        if q is None or q.subject != up.subject:
            return False
        sts = [st for st in q.statements if st.owner.rpartition("::")[2].rpartition(".")[2] == sel.callee
               and st.result != RESULT_AGGREGATE and AG.cap_rows(st.limits) is not None]
        if len(sts) != 1:
            return False
        st = sts[0]
        cap = AG.cap_rows(st.limits)
        assert cap is not None
        where = f"{sel.callee}@{st.line}"
        turn, assumed, printed = self._turn(st, q)
        if turn is None:
            self.say(f"(C) {where} returns at most {cap} row(s); when it began to return the row is not read")
            return False
        host = self._move_host(up.before) if up.before is not None else None
        p = next((p for p in sorted(moving, key=lambda p: p.first) if p.machine == host), None) or min(moving, key=lambda p: p.first)
        found, why = AG.cap_take(picks, p.machine, p.first, turn, cap, label)
        if found is None:
            self.say(f"(C) not read by {where}'s cap ({cap} row(s)) from its turn {turn:%Y-%m-%d %H:%M:%S}: {why}")
            return False
        take = found.take
        premises: list = self._held_premises(assumed, printed, take.at)
        if found.empty:
            premises.append(T.Said("an-empty-selection-lost-the-rows", {"n": len(found.empty), "callee": sel.callee}))
        if found.place >= 0:
            # the row's values of the keys, as the rows file read them at as_of (set as the row is made)
            keys = [f"{k.rpartition('.')[2]}={self._label_of(q.subject, k.rpartition('.')[2]) or '?'}" for k in st.keys] \
                or ["the row's label"]
            premises.append(T.Said("the-row-s-label-by-the-take-s-place", {"place": found.place + 1, "width": found.width,
                                                                            "mark": take.mark, "agent": sel.agent_name,
                                                                            "keys": keys}))
        if found.others:
            premises.append(T.Said("another-width-another-label", {"n": len(found.others), "agent": sel.agent_name,
                                                                   "width": found.width}))
        self.end(T.Said("ran-when-the-first-selection-below-its-cap-took-it", {
            "caller": c.owner.split("::")[-1], "agent": sel.agent_name, "where": where, "turn": f"{turn:%Y-%m-%d %H:%M:%S}",
            "cap": cap, "full": len(found.full), "empty": len(found.empty),
            "machines": len({x.machine for x in (*found.full, *found.empty)}),
            "machine": take.machine.split(".")[0], "mark": take.mark, "at": take.at[:23],
            "count": take.count, "first": p.first[:23], "premises": premises}))
        return True

    def _label_of(self, subject: str, hole: str) -> str:
        """The row's value of the column the agent's loop element *hole* names (``prod_source_label`` names
        ``prodSourceLabel``), from the rows file's row of *subject*'s class; "" where none is."""
        if not hole:
            return ""
        cls = subject.partition(".")[0]
        entity = next((e for e, k in (self.data.get("classes") or {}).items() if k == cls and e != "about"), None)
        row = (self.data.get("rows") or {}).get(entity or "", {})
        want = hole.replace("_", "").lower()
        held = next((v for col, v in row.items() if col.replace("_", "").lower() == want), None)
        value = held.get("value") if isinstance(held, dict) else None
        return "" if value is None else str(value)

    def _move_host(self, before: str) -> str | None:
        """The host of the statuslog row the move at *before* is, or None."""
        states = (self.data.get("statuslog") or {}).get("states") or []
        hosts = {s.get("host") for s in states if str(s.get("time", ""))[:19] == before[:19]}
        return hosts.pop() if len(hosts) == 1 else None

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
    @staticmethod
    def _keeps_said(fn: R.Function, exits: list[int], arms: list[Arm]) -> tuple[str, str]:
        """S-2b (qk-design.md, tmaeno "OK" 2026-11-16 16:1x): what the map's ``ExitKeeps`` says of the exits a pass that
        ran no arm may have left by -- one keeping a run from none of the *arms* is, by the flow, no reason it ran none
        (a test on the way kept it off, whether it left there or ran on past), one keeping a run from some may have kept
        it from them, one not listed is not read -- and, where every one keeps a run from none, the clause saying so
        (empty otherwise).  A pass that is one turn of a loop reads each exit over that turn, one not in the loop off
        its way; a whole call's return or raise in loops, over the turn it left by, the rest said as past it.  The
        lists are the flow's, path-insensitive both ways: what a run's values allow is the guards' reading.  Not read
        where the dump reads none."""
        owner = getattr(fn, "owner", "")
        keeps = exit_keeps(owner)
        if keeps is None:
            return "(whether one keeps a run from the arms is not read)", ""
        moving = {a.line for a in arms}
        # M-g (the map's TurnKeeps and PrintLoops; tmaeno "OK" 2026-11-16 16:1x and 17:2x): a pass cut at an entry row
        # in a loop is one turn of it (R-g1); a pass that is the whole call and leaves by a return or a raise in loops
        # is read over the turn it left by, the rest said as past that turn: a later turn, or what follows the outermost
        # loop, reached past the turn whether or not another runs (the map, 2026-11-16 17:3x) (R-g2)
        entry = R.entry_row(fn) if getattr(fn, "rows", None) and getattr(fn, "node", None) is not None else None
        loops = print_loops(owner, entry.line) if entry is not None else None
        turn_of = loops[0] if loops else None
        clear, off, unread = [], [], []
        early: dict[int, list[int]] = {}
        later: dict[int, tuple[list[int], int]] = {}
        for x in exits:
            ek = keeps.get(x)
            if ek is None:
                unread.append(x)
                continue
            turns = {t.loop: t for t in getattr(ek, "turns", None) or []}
            if turn_of is not None:
                if turn_of not in turns:
                    off.append(x)
                    continue
                kept = _kept_lines(turns[turn_of]) & moving
            elif ek.exit.partition("@")[0] in ("return", "raise") and turns:
                outer = ek.turns[-1]
                kept = _kept_lines(outer) & moving
                if after := (_kept_lines(ek) & moving) - kept:
                    later[x] = (sorted(after), outer.loop)
            else:
                kept = _kept_lines(ek) & moving
            if kept:
                early[x] = sorted(kept)
            elif x not in later:
                clear.append(x)
        parts = ([f"{clear} keep a run from none of the arms"] if clear else []) + [
            f"{x} keeps a run from {a}: a pass may have left by it before them" for x, a in early.items()] + [
            f"{x} keeps a run from {a} only past the turn it left by -- a later turn of the loop at {loop}, or what "
            "follows it: in that turn, from none (whether the pass would have run on past it is not read)"
            for x, (a, loop) in later.items()] + (
            [f"{off} are not in the loop at {turn_of} a pass is one turn of: no way it left by"] if off else []) + (
            [f"it lists no {unread}: not read"] if unread else [])
        conclusion = ""
        if exits and not early and not unread:
            conclusion = ("; so no exit kept a pass from them by the flow" + (" in the turn it left by" if later else "")
                          + ": a test on the way did")
        return f"(the map's ExitKeeps: {'; '.join(parts)})", conclusion

    def _held_printed(self, fn: R.Function, run, env: E.Env, arms: list[Arm], guards: list[str]) -> list[str]:
        """S-3 (qk-design.md, tmaeno "OK" 2026-11-16 13:0x): the values of the names the arms' tests read that the
        map's ``GuardFrom`` reads back to one literal, or to a callee's print of what its call gives -- read off the
        callee's one print in the pass *run*, its log asked over the pass on the pass's machine -- put in *env*, and a
        dict's items under the texts the *guards* spell them by.  A name *env* already knows (the pass's own prints),
        one whose tests' ways disagree, one a way changes in place in the function: left.  Each one read, said."""
        owner = getattr(fn, "owner", "")
        moving = {a.line for a in arms}
        found = [g for g in guard_froms(owner) if set(g.arms) & moving]
        if not found or run is None or getattr(fn, "node", None) is None:
            return []
        rel = owner.partition("::")[0]
        lo, hi = fn.node.lineno, fn.node.end_lineno or fn.node.lineno
        readings: dict[str, set] = {}
        for g in found:
            reading = _held_reading(g, rel, lo, hi)
            if reading is None and (on_way := self._constant_printed_way(fn, run, g, rel, lo, hi)) is not None:
                reading = ("way", *on_way)
            readings.setdefault(g.name, set()).add(reading)
        said = []
        for name, got in sorted(readings.items()):
            if len(got) != 1 or None in got or env.knows(name):
                continue
            kind, *what = next(iter(got))
            if kind == "constant":
                value, where = what[0], f"the constant at {what[1]}"
            elif kind == "way":
                # F-2: the binding the printed way passed last before the test, in the key's turn
                value, where = what[0], f"the constant at {what[1]}, which the way to {what[2]}'s record passed last"
            else:
                read = self._callee_print(run, *what)
                if read is None:
                    continue
                value, where = read
            source = f"the map's GuardFrom: {where}"
            env.put(name, value, source)
            if isinstance(value, dict):
                items = [(text, value[key]) for text, key in _items_spelled(guards, name) if key in value]
                for text, item in items:
                    env.put(text, item, source)
                said += [f"{text} = {item!r} ({where})" for text, item in items]
            else:
                said.append(f"{name} = {value!r} ({where})")
        return said

    def _constant_printed_way(self, fn: R.Function, run, g, rel: str, lo: int, hi: int) -> tuple[Any, int, int] | None:
        """Bundle F, F-2: where the map's GuardFrom *g* reads its name back to more than one literal, the one the pass
        *run* took to the test in the key's turn (:func:`which_ran.constant_on_way`), read off its rows whose records
        there hold the key.  None where a way changes the object in place in the function, a source is no literal, or
        the printed rows do not say."""
        sources = list(getattr(g, "sources", None) or [])
        if len(sources) < 2 or any(s.kind != "constant" for s in sources) or getattr(fn, "node", None) is None:
            return None
        for s in sources:
            for c in getattr(s, "changes", None) or []:
                crel, _, cline = str(c.at).rpartition(":")
                if crel == rel and cline.isdigit() and lo <= int(cline) <= hi:
                    return None
        try:
            bindings = {int(s.line): ast.literal_eval(s.name) for s in sources}
        except (ValueError, SyntaxError, TypeError):
            return None
        if len({repr(v) for v in bindings.values()}) < 2 or not all(_hashable(v) for v in bindings.values()):
            return None
        lines = self._key_rows(fn, run)
        return constant_on_way(fn, lines, int(g.test_line), bindings) if lines else None

    def _key_rows(self, fn: R.Function, run, subject: str | None = None) -> list[int]:
        """The lines of *fn*'s rows with a record in *run* whose hole of the row's key column holds the question's key
        (the check :meth:`_took_by` makes of a selection's print)."""
        try:
            hits = R.seen(fn, run, self.mode).hits
        except (AttributeError, TypeError):
            return []
        keys = self._key_columns(subject or self.args.subject, self.args.key)
        out = []
        for line, found in hits.items():
            stmt = R.statement_at(fn.node, line)
            row = fn.row_at(stmt) if stmt is not None else None
            if row is None or row.line != line:
                continue
            exprs = {h.expression for m in row.map_rows for h in m.holes or [] if keys & set(h.columns or [])}
            if any(_hole_holds(row, run.records[i], SimpleNamespace(part=MUST_PART_MESSAGE, expression=e, bind=""),
                               self.args.key) for i, _, _ in found for e in exprs):
                out.append(line)
        return sorted(out)

    def _callee_print(self, run, at: str, statement: str, hole: int, expression: str) -> tuple[Any, str] | None:
        """``(value, where)``: what the callee's print at *statement* holds in its hole *expression*, its one record
        inside the pass *run* (the callee's runs of the key in its own logs, on the pass's machine, over the pass);
        None where none or more than one is, or its log is not in the records (the ask kept)."""
        crel, _, cline = statement.rpartition(":")
        if not cline.isdigit():
            return None
        line = int(cline)
        cdef = R.def_holding(crel, line)
        if cdef is None:
            return None
        callee = f"{crel}::{cdef.name}"
        try:
            cfn = R.function(crel, cdef.name, within=[line])
        except (LookupError, ValueError, SyntaxError):
            return None
        stmt = R.statement_at(cfn.node, line)
        row = cfn.row_at(stmt) if stmt is not None else None
        if row is None:
            return None
        # one print's records in one pass are read, asked and said once, whichever names its holes give
        read: dict = vars(self).setdefault("_callee_hits", {})
        once = (run.machine, run.first, run.last, statement)
        if once not in read:
            window = (W._down(W.when(run.first)), W._down(W.when(run.last) + datetime.timedelta(seconds=1)))
            ask = W.Asking(window, run.machine, T.Said("for-a-callee-s-print", {
                "callee": cdef.name, "line": line, "at": at, "first": run.first, "last": run.last}))
            logs, unanswered = self._logs_at(self.args.key, list(dict.fromkeys(self.files_of(callee, [], [line]))), ask,
                                             owners_of([callee]))
            found: list = []
            if unanswered:
                self._keep_unanswered(unanswered)
            for log in logs.values() if not unanswered else ():
                if run.machine in log.without:
                    continue
                for r in runs_for(log, self.args.key, self._other_of(self.args.key, log)):
                    if not of_mover(r, callee) or r.machine != run.machine:
                        continue
                    for i, _, _ in R.seen(cfn, r, self.mode).hits.get(row.line, ()):
                        if run.first <= r.records[i].stamp <= run.last:
                            found.append(r.records[i])
            if len(found) > 1:
                self.say(f"  {cdef.name}'s print at {line} is in the pass {len(found)} times: which its call at {at} gave "
                         "is not read")
            read[once] = found if not unanswered else []
        hits = read[once]
        if len(hits) != 1:
            return None
        text = row.capture(hits[0]).get(expression)
        value = _hole_value(text) if text is not None else E.UNKNOWN
        if value is E.UNKNOWN:
            return None
        return value, f"{cdef.name}'s print at {line}, its hole {expression}, at {hits[0].span}"

    def blocked(self, q: Blocked):
        self.start("S3", "what kept it from moving")
        fn = q.fn
        arm = next((a for a in q.arms if a.line == q.exit_line), None)
        if arm is not None:
            return self._arm_taken(q, arm)
        def path_of(line: int) -> list[str]:
            # what held on the way to the exit, the early exits above it too; one read at another line is not solved here
            return [g for g in pathcond.arm_condition(R.statement_at(fn.node, line)) if not E.read_elsewhere(g)]

        if q.exit_line is None:
            # every pass ran none of the arms: what settles a guard is the path of the exit a pass was shown to leave by,
            # and what the passes printed on the way
            left_ways = list(q.took) or [None] * len(q.passes)
            paths = [path_of(x) if x is not None else [] for x in left_ways]
            keeps, so = self._keeps_said(fn, list(q.left_by), q.arms)
            by_way = Counter(x for x in left_ways if x is not None)
            if not q.left_by:
                self.say(f"{fn.shown}: every pass took none of its candidates, so no exit's path settles a guard")
            elif not by_way:
                self.say(f"{fn.shown}: every pass ran none of its arms, so no exit's path settles a guard; a pass may have left "
                         f"by {list(q.left_by)} {keeps}{so}")
            else:
                went = "; ".join(f"{k} pass(es) left by {x}, reached when {path_of(x)}" for x, k in sorted(by_way.items()))
                rest = len(left_ways) - sum(by_way.values())
                self.say(f"{fn.shown}: every pass ran none of its arms: {went}"
                         + (f"; {rest} may have left by one of the others or ran on past them" if rest else "")
                         + f"; of the exits {list(q.left_by)} {keeps}{so}")
        else:
            path = path_of(q.exit_line)
            paths = [path] * len(q.passes)
            keeps, _ = self._keeps_said(fn, [q.exit_line], q.arms)
            self.say(f"{fn.shown}:{q.exit_line} is reached when {path}"
                     + (f" {keeps}" if keeps.startswith("(the map's") else ""))
        read = {a.line: arm_guards(fn.node, a) for a in q.arms}
        # an arm in an except handler is kept off where every pass's rows show its try raised nothing: no guard to read
        printed = [self._rows_of_one_call(fn, run) for run in q.passes]
        quiet: dict[int, tuple[int, list[int]]] = {}
        for a in q.arms:
            for u in read[a.line][1]:
                shown = [raised_nothing(fn.node, u.line, rows) for rows in printed]
                if shown and all(r is not None for r in shown):
                    quiet[a.line] = (u.line, sorted({r for r in shown if r is not None}))
                    break
        guarded = [a for a in q.arms if a.line not in quiet]
        # S-3: the names the guards read whose values the map's GuardFrom reads back to a literal or a callee's print
        tested = sorted({g for a in guarded for g in read[a.line][0] if not E.read_elsewhere(g)})
        for n, (run, env) in enumerate(zip(q.passes, q.envs, strict=True), 1):
            for line in self._held_printed(fn, run, env, guarded, tested):
                self.say(f"  pass {n}: {line}")
            # D-3 (P50, bundle D): what the rows every way to a guarded arm prints before it printed in the pass, put as
            # the values its tests read (``printed_before``): FinisherThr's 727 prints the Limit and End 728 compares
            if run is not None:
                sn = R.claim([R.seen(fn, run, self.mode)])[0]
                for a in guarded:
                    stmt = R.statement_at(fn.node, a.line)
                    if stmt is not None:
                        printed_before(fn, sn, stmt, R.must_rows(fn.node, stmt)[0], env)
        per_pass = []
        for run, env, path in zip(q.passes, q.envs, paths, strict=True):
            fixed: dict[str, bool] = {}
            for g in path:
                sol = E.solve(g, True, env)
                fixed.update(sol.known)
                fixed.update(sol.settled)
            by_exit = dict(fixed)
            ways = self._ways_printed(fn, run)
            for g in sorted({g for _, conds in ways for g in conds}):
                sol = E.solve(g, True, env)
                fixed.update(sol.known)
                fixed.update(sol.settled)
            blocking = {}
            for arm in guarded:
                blocking[arm.line] = [
                    g for g in read[arm.line][0]
                    if not E.read_elsewhere(g)
                    and E.truth(ast.parse(E.strip_annotation(g), mode="eval").body, fixed, env) is False
                ]
            per_pass.append((by_exit, fixed, ways, blocking))
        by_exit, fixed, ways, blocking = per_pass[0]
        self.need(T.Said("the-same-guards-against-the-arms-in"), all(b == blocking for *_, b in per_pass))
        self.say(f"settled by the exit taken: {by_exit}")
        if fixed != by_exit:
            self.say(f"settled by the rows the pass printed on the way {[line for line, _ in ways]}: "
                     f"{ {k: v for k, v in fixed.items() if k not in by_exit} }")
        for arm in q.arms:
            guards, handlers = read[arm.line]
            inside = "".join(f"; in an except handler at {u.line} ({u.detail})" for u in handlers)
            if arm.line in quiet:
                handler, rows = quiet[arm.line]
                self.say(f"  arm {arm.line} {arm.outcomes}: in the except handler at {handler}; every pass printed row(s) {rows} "
                         "with nothing after it in the try that may raise: kept off, nothing raised")
            else:
                self.say(f"  arm {arm.line} {arm.outcomes}: needs {guards}{inside}; false in the passes: {blocking[arm.line]}")
        # what S3 needs to go on, not a finding: an arm none of whose guards the passes were shown to make false
        unshown = [str(a.line) + (" (in an except handler)" if read[a.line][1] else "") for a in guarded if not blocking[a.line]]
        self.need(T.Said("a-guard-the-passes-made-false-against", {"unshown": list(unshown)}), not unshown)
        if not guarded:
            # kept off by no guard to follow: why nothing raised has no cause to walk, so the branch did not do it
            raise Excluded(T.at(self.hop, T.Said("every-arm-runs-only-in-an-except", {
                "fn_name": fn.shown, "handlers": sorted({h for h, _ in quiet.values()}),
                "rows": sorted({r for _, rows in quiet.values() for r in rows})})))
        fewest = min(len(v) for v in blocking.values())
        nearest = [a for a in guarded if len(blocking[a.line]) == fewest]
        guards = sorted({g for a in nearest for g in blocking[a.line]})
        self.need(T.Said("one-guard-against-the-nearest-arm-s"), len(guards) == 1)
        guard = guards[0]
        for a in guarded:
            if a not in nearest:
                self.say(f"  not followed: arm {a.line} also needs {[g for g in blocking[a.line] if g != guard]}")
        self.say(f"the nearest arm(s) {[a.line for a in nearest]} were kept off by: {guard!r} (rule 21)")
        arm_stmt = R.statement_at(fn.node, nearest[0].line)
        where = guard_if(arm_stmt, guard) or early_exit_if(fn.node, arm_stmt, guard)
        where = self.need(T.Said("the-if-whose-test-is", {"guard": guard}), where)
        return Input(fn, guard, where.lineno, q.passes, q.envs, sorted({o for a in nearest for o in a.outcomes}))

    def _rows_of_one_call(self, fn: R.Function, run) -> set[int]:
        """The rows of *fn* one call's pass *run* printed: none of a run of several calls, another of which may have
        raised (as :meth:`_ways_printed`)."""
        if run is None or not getattr(fn, "rows", None):
            return set()
        sn = R.claim([R.seen(fn, run, self.mode)])[0]
        return {row.line for row in fn.rows if sn.present(row)} if sn.single_call() else set()

    def _ways_printed(self, fn: R.Function, run) -> list[tuple[int, list[str]]]:
        """``(row line, the tests reaching it ran under)`` of each row one call's pass *run* printed (unambiguous; its
        ``pathcond.arm_condition`` but a guard read at another line), outside any loop
        the pass may have gone round more than once (``which_ran.once_loops_of``): the tests of the ``if``s around it
        held when it printed -- activateJob's 111, in the else of 70, says ``not (len(res) == 0 or allOK)``, though
        the exit the pass left by (153, ``return True``) is reached either way (KNOWN_ISSUES.md K46).  None of a run of
        several calls, nor where no run is handed."""
        if run is None or not getattr(fn, "rows", None):
            return []
        sn = R.claim([R.seen(fn, run, self.mode)])[0]
        if not sn.single_call():
            return []
        once = once_loops_of(fn, sn, run, self.mode)
        return [(row.line, conds) for row in fn.rows
                if sn.unambiguous.get(row.line) and R.loops_over(row.stmt, fn.node) <= set(once)
                and (conds := [g for g in pathcond.arm_condition(row.stmt) if not E.read_elsewhere(g)])]

    def _arm_taken(self, q: Blocked, arm: Arm) -> None:
        """S3 where every pass left by an arm that writes the subject: no guard kept the passes off it, so what kept the
        value is what the arm wrote -- a value known at run time (``runtime(<name>)``), read off a row after the arm in
        its block that prints it, nothing binding the name between (``lines_after``; KNOWN_ISSUES.md K26).  The value
        observed in every pass: the mover ran and wrote it again, an END (Fr2's updateJobStatus: "old=running
        new=running", the pilot's reports).  Anything else stops, saying what is not read."""
        fn = q.fn
        self.say(f"{fn.shown}:{arm.line} is an arm that writes {arm.outcomes}: every pass ({len(q.passes)}) went through it")
        names = {o[len("runtime("):-1] for o in arm.outcomes if o.startswith("runtime(") and o.endswith(")")}
        name = self.need(T.Said("what-arm-wrote-yet-the-value-stayed", {"arm_line": arm.line, "observed": self.args.observed, "arm_outcomes": arm.outcomes}), names.pop() if len(names) == 1 and len(arm.outcomes) == 1 else None)
        spans = lines_after(fn.node, R.statement_at(fn.node, arm.line), name)
        wrote = []
        for n, p in enumerate(q.passes, 1):
            seen = R.claim([R.seen(fn, p, self.mode)])[0]
            values = sorted({row.capture(p.records[i])[name].strip()
                             for row in fn.rows if any(a <= row.line <= b for a, b in spans)
                             for i, _, _ in seen.hits.get(row.line, ()) if name in row.capture(p.records[i])})
            self.say(f"  pass {n} ({p.first}): {name} printed after it {values or 'nowhere'}")
            wrote.append(values)
        self.need(T.Said("the-value-of-arm-wrote-in-every", {"name": name, "arm_line": arm.line, "name_2": name}), all(len(v) == 1 for v in wrote))
        values = sorted({v[0] for v in wrote})
        self.need(T.Said("arm-wrote-in-the-passes-not-the", {"arm_line": arm.line, "values": values, "observed": self.args.observed}), values == [self.args.observed])
        self.end(T.Said("ran-in-every-pass-and-wrote-the", {"fn_name": fn.shown, "arm_line": arm.line, "n_q_passes": len(q.passes), "name": name, "observed": self.args.observed}))
        return None

    # ------------------------------------------------------------ Input: where the guard's name was bound
    def input(self, q: Input):
        self.start("S4", "where the guard's name was bound")
        fn = q.fn
        test = ast.parse(E.strip_annotation(q.guard), mode="eval").body
        by = [false_by_values(test, env) for env in q.envs]
        if q.envs and all(b is not None for b in by):
            return self._false_on_values(fn, q, test, by)
        names = [n.id for n in ast.walk(test) if isinstance(n, ast.Name) and n.id != "self"]
        names = list(dict.fromkeys(n for n in names if all(E.ev(ast.Name(n), env) is E.UNKNOWN for env in q.envs)))
        self.say(f"{q.guard!r} at {q.line} reads {names} (unknown from the passes)")
        name = self.need(T.Said("one-name-the-guard-reads"), names[0] if len(names) == 1 else None)
        steps = A.walk(fn, q.line)
        self.say(f"walk from {fn.shown}:{q.line}: asked by question" + (f"; {steps.note}" if steps.note else ""))
        takens = [E.taken(steps, name, env, owner=fn.owner) for env in q.envs]
        t = takens[0]
        for step, false in t.dropped:
            self.say(f"  {name} @{step.line} = {step.value}: out, {false!r} is false")
        for step in t.kept:
            self.say(f"  {name} @{step.line} = {step.value}: kept  {step.guards}")
        self.need(T.Said("one-binding-of-in-every-pass", {"name": name}), t.step is not None and all(x.step is t.step for x in takens))
        step = t.step
        self.say(f"took effect: {t.describe()}")
        for n, p in enumerate(q.passes, 1):
            printed = self._printed_in(fn, p, f"len({name})")
            self.say(f"  pass {n} ({p.first}) printed len({name}): {printed or 'nothing'}")
        if not step.terminal:
            self.need(T.Said("a-way-on-from", {"name": name, "step_value": step.value}), None)
        self.say(f"stops: {step.terminal} -- {step.detail}")
        return Rows(fn, name, step, q.passes, q.envs, q.outcomes)

    def _false_on_values(self, fn: R.Function, q: Input, test: ast.AST, by: list) -> None:
        """S-3b: the guard S3 handed is false in every pass on the values read (:func:`false_by_values`), so no name it
        reads is followed: the parts that make it false are said, each value they read in each pass with where it was
        read, the parts no pass gives a value as ones it is false whatever they hold, and the branch ends.  Why the
        values were what they were (nNewJobs 0) is another question, not read here."""
        parts: dict[str, ast.AST] = {}
        for atom in E.atoms(test):
            parts.setdefault(ast.unparse(atom), atom)
        deciding = list(dict.fromkeys(text for b in by for text in b))
        unread = [text for text, atom in parts.items() if any(E.truth(atom, {}, env) is E.UNKNOWN for env in q.envs)]
        self.say(f"{q.guard!r} at {q.line}: false in every pass ({len(q.passes)}) on the values read, by {deciding}"
                 + (f", whatever {unread} held" if unread else ""))
        for n, (env, b) in enumerate(zip(q.envs, by, strict=True), 1):
            for text in b:
                view = copy.copy(env)
                view.used = {}
                value = E.ev(parts[text], view)
                read = "; ".join(f"{k} = {view.lookup(k)!r} ({source})" if view.knows(k) else f"{k}: {source}"
                                 for k, source in list(view.used.items()))
                self.say(f"  pass {n}: {text} is {value!r}: {read}")
        self.end(T.Said("false-on-the-values-read", {
            "fn_name": fn.shown, "line": q.line, "n": len(q.passes), "parts": deciding,
            "whatever": T.Said("whatever-they-held", {"unread": unread}) if unread else ""}))

    # ------------------------------------------------------------ Rows: what the call returned
    def rows(self, q: Rows):
        self.start("S5", "what the call returned")
        fn, step = q.fn, q.step
        call = ast.parse(step.value, mode="eval").body
        self.need(T.Said("a-call-behind", {"q_name": q.name}), isinstance(call, ast.Call))
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
            T.Said("a-facade-s-the-method-the-walk", {"callee": callee}),
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
            T.Said("every-proxy-method-reaches-at-this-call", {"callee": callee, "unread": unread, "fact": named_unread or 'none'}),
            not unread,
        )
        self.need(
            T.Said("the-same-tables", {
                "named": named or "none", "at_call": at_call,
                "unread": T.Said("and-read-every-proxy-method", {"named_unread": named_unread}) if named_unread else "",
            }),
            named and not named_unread and (set(named) <= set(at_call) if len(named) >= 6 else set(named) == set(at_call)),
        )
        jobs = self.data.get("job_rows", {}).get("rows")
        history = self.data.get("job_statuslog", {}).get("by_pandaid")
        lacking = [n for n, v in (("job_rows", jobs), ("job_statuslog", history), ("counts", self.data.get("counts", {}).get("rows"))) if not v]
        if lacking:
            self.ask_rows({"kind": "statement", "method": callee, "site": sites, "tables": at_call, "binds": binds,
                           "at": [p.first for p in q.passes]},
                          f"the rows {callee} returned in each pass, their history and the tables' counts (the rows file lacks "
                          f"{', '.join(lacking)})", "call")
        self.need(T.Said("the-rows-returned-in-each-pass-the", {"callee": callee}), jobs and history)
        self.say(f"rows: {self.data['job_rows']['source']}")
        counts = self.need(T.Said("the-tables-rows-of-the-key-by"),
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
        self.need(T.Said("reading-to-give-every-count-the-passes"), fits)
        # its assumption, checked where both sides are of one moment: the counts, by table and jobStatus
        self.say(f"at one moment ({moment}): {at_call} hold {sum(held.values())} row(s){f' {held}' if held else ''}; "
                 f"final among them: {final_in or 'none'}; not yet final outside them: {open_out or 'none'}")
        self.need(T.Said("reading-s-assumption-at-that-moment-no", {"at_call": at_call}),
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
            self.need(T.Said("a-statuslog-row-after-with-what-the", {"after": after, "q_outcomes": q.outcomes}), not later)
            as_of = self.need(T.Said("the-time-the-statuslog-was-read-its"),
                              self.data.get("statuslog", {}).get("as_of"))
            # the record door drops the passes after as_of (S2), the file door does not: a statuslog read before the
            # last pass says nothing of the time after it (e1, 2026-10-14)
            self.need(T.Said("the-statuslog-read-after-the-last-pass", {"after": after, "as_of": as_of}), as_of > after)
            # the jobs are named from the tables at the counts' moment: the job rows of that moment, after the last pass
            tables_at = self.need(T.Said("the-time-the-counts-were-read-their"),
                                  self.data["counts"].get("as_of"))
            rows_at = self.data["job_rows"].get("as_of")
            self.need(T.Said("the-job-rows-read-at-the-counts", {"tables_at": tables_at, "rows_at": rows_at}), rows_at == tables_at)
            self.need(T.Said("the-tables-read-after-the-last-pass", {"after": after, "tables_at": tables_at}), tables_at > after)
            n_held = sum(held.values())
            self.need(T.Said("a-job-in-at-the-value-not", {"at_call": at_call, "tables_at": tables_at, "as_of": as_of}), n_held)
            open_rows = sorted((str(r["pandaid"]), r["jobstatus"]) for r in jobs if (r.get("tbl") or "").lower() in read)
            self.need(T.Said("the-job-rows-to-hold-the-row", {"n_held": n_held, "at_call": at_call, "n_open_rows": len(open_rows)}),
                      len(open_rows) == n_held)
            named = [f"{pid} {status}" for pid, status in open_rows]
            self.end(T.Said("a-wait-still-open-no-statuslog-row", {"after": after, "as_of": as_of, "q_name": q.name, "n_named": len(named), "at_call": at_call, "tables_at": tables_at, "named_item": named[:6], "fact": ' ...' if len(named) > 6 else ''}))
            return None
        # every job made before the move and not yet final at the first pass, not only the last pass's:
        # a job made after the passes also holds the list open
        created = {pid: min(r["creationtime"] for r in jobs if str(r["pandaid"]) == pid) for pid in ids}
        first = q.passes[0].first[:19]
        waited = [pid for pid in ids if created[pid] < moved["time"] and (finals[pid] is None or finals[pid] > first)]
        if any(finals[pid] is None for pid in waited):
            self.need(T.Said("a-final-state-for-every-job-in", {"q_name": q.name}), None)
        last = max(finals[pid] for pid in waited)
        made_later = [pid for pid in waited if created[pid] > q.passes[-1].first[:19]]
        self.say(f"the arm kept off wrote {moved['status']!r} at {moved['time']}; of the {len(waited)} job(s) made before it and "
                 f"not final at the first pass ({len(made_later)} made after the last pass), the last reached a final state at {last}")
        if last > moved["time"]:
            self.need(T.Said("empty-before-the-value-moved", {"q_name": q.name}), None)
        self.end(T.Said("a-legitimate-wait-emptied-at-and-the", {"q_name": q.name, "last": last, "moved_time": moved['time']}))
        return None

    # ------------------------------------------------------------ Late: the value moved on, late
    def late(self, q: Late):
        self.start("L1", "what moved it on, and when")
        states = self.data.get("statuslog", {}).get("states", [])
        at, unmet = _stay_of(states, q.observed, q.since)
        self.need(unmet, at is not None)
        after = self.need(T.Said("a-statuslog-row-after", {"q_observed": q.observed}), states[at + 1] if at + 1 < len(states) else None)
        since = _when(states[at].get("time_extended") or states[at]["time"])
        moved = _when(after.get("time_extended") or after["time"])
        self.result["window"] = {"start": _iso(states[at].get("time_extended") or states[at]["time"]),
                                 "end": _iso(after.get("time_extended") or after["time"]), "end_kind": "moved", "end_from": "statuslog"}
        self.say(f"{q.observed!r} from {since}; {after['status']!r} at {moved}, "
                 f"{(moved - since).total_seconds() / 3600:.1f} h later")
        symptom = Symptom(subject=q.subject, observed=q.observed, task_id=q.key)
        inv, _ = asyncio.run(_investigate(mapdump.code_map("rerun"), symptom, None, q.key, None, 1, None))
        R.check_blobs(inv, "derive-strategy's readings")
        follow = self.need(T.Said("the-map-s-answer-to-what-moves"), inv.hops[0].strategy.follow_up)
        asked = f"{q.subject}={q.observed}"
        statements = [rs for rs in follow.reader_statements if asked in rs.selects]
        # M2: every mover in the window -- by the statements that select the value, by the writes that require it, and
        # every other writer of another value, which takes a row by its key; and where the selecting statements' rows
        # go, which leaves out a writer they do not reach (value_reach)
        strategy_mod.attach_mover_gaps(inv.hops[0].strategy, R.ROOTS)
        self._strategy = inv.hops[0].strategy
        gaps = map_gaps(follow)
        writers = asyncio.run(self._writers_of(q.subject))
        movers = movers_of(follow, writers, q.subject, q.observed)
        reach = value_reach_of(follow, q.subject, q.observed, movers)
        # what moved it on is who wrote the next value at the row's time: the question type Asked's H1 and H2.  The
        # mover need not ask on the observed value (the adder takes a job by its key from a report), and a run's
        # mark need not name it (the adder's is the PandaID, one run for every function of a pass)
        w = self._which_wrote(q.subject, after["status"], q.key, after, reach=reach, before=q.observed)
        if isinstance(w, OneOf):
            # the one arm H2 left goes on as the mover, what every END rests on; several are not walked for each (K38)
            w = self._the_arm_left(q.subject, after["status"], w)
        owner = w.reading.owner
        if w.question is None:
            # no run to hand: what moved it on is read in the mover's own pass (K47)
            raise Stop(T.at(self.hop, T.Said("a-pass-of-at-the-move-to", {"owner": owner, "chosen_line": w.chosen.line})))
        self.say(f"moved on by {owner}:{w.chosen.line}, in {w.question.describe()}")
        run = w.question
        self.result["moved_by"] = {"owner": owner, "line": w.chosen.line, "run": {
            "log": run.log.name, "machine": run.machine or None, "token": run.token, "first": run.first, "last": run.last},
            **({"rests": [T.to_json(s) for s in self.rests]} if self.rests else {})}
        # M2: the arm of a writer that prints nothing, read at its caller's call (a setter: JediTaskSpec.setOnHold at
        # TaskRefiner's 318), is the chain's write; the mover is the caller, whose run H2 read -- where the rows of the
        # value's reads reach it and it writes the subject itself
        if w.via is not None and VR.bare(caller := w.via.candidate.owner) in reach.reached \
                and arms_of(writers, caller, q.subject, q.observed):
            self.say(f"the mover is its caller {caller}:{w.via.line}, whose run H2 read: "
                     f"{', '.join(reach.reads)} reach it (M2)")
            owner = caller
            self.result["moved_by"]["mover"] = caller
        owners = list(dict.fromkeys(j.owner for j in writers))
        by_key = by_key_of(writers, movers, q.subject, q.observed)
        refusing = [o for o in owners if o not in movers and o not in by_key and refused_arms(writers, o, q.subject, q.observed)]
        return Missed(q.subject, q.observed, after["status"], since, moved, statements, owner, w.question,
                      movers=movers, paths=paths_of(follow, writers, q.subject, q.observed), by_key=by_key,
                      refusing=refusing, writers=writers, gaps=gaps, reader_log_files=list(follow.reader_log_files),
                      reach=reach)

    def missed(self, q: Missed):
        self.start("L2", "why the others passed it over")
        # the late question the mover's callers are read for (M3(a)-cap reads its statements' turn)
        self._missed_q = q
        cls = q.subject.split(".")[0]
        self.need(T.Said("the-rows-file-s-row-of-its", {"cls": cls}), self._row_of(cls))
        self.say(f"{len(q.statements)} statement(s) ask rows on {q.subject}={q.observed}, from {q.since} to {q.moved}")
        ordered = []
        kinds: dict[tuple[str, int], str] = {}
        # A-5 (P25): of the mover's statements, the one whose loop printed the row's key in the moving pass took it
        took_by = self._took_by(q)
        # P25 (i): O1 reads the counts the taking print printed
        self._took = took_by
        if took_by is not None:
            self.result["took_by"] = {"line": took_by.statement.line, "print": took_by.line, "text": took_by.text}
            if took_by.ruled_out:
                self.result["took_by"]["ruled_out"] = [line for line, _ in took_by.ruled_out]
        for st in q.statements:
            # the call that moved the value on returned the row: a term the row fails was not in that build
            took = st.owner == q.mover and (took_by is None or st is took_by.statement)
            kind, said = self._passed_over(st, q, took)
            if took and took_by is not None and took_by.line is None:
                # the P25 reader: the others handed no row on in that pass (their hand-ons' sure prints are absent)
                kind = f"took it at {q.moved}, the mover's other selections ruled out in that pass: {took_by.text}"
            elif took:
                kind = f"took it at {q.moved}" + (f", its loop printing the row at {took_by.line} in that pass: "
                                                  f"{took_by.text}" if took_by is not None else "")
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
        self._late_ran = ()
        self._movers_in_window(q, kinds)
        own = self._why_moved_only_then(q, ordered)
        if not self._late_ran:
            return own
        # every other mover that ran in the stay goes on alone beside it: why its passes did not move the value (M3)
        ran = list(self._late_ran)
        return Fork([own, *ran], [T.Said("mover-that-moved-it-label", {"owner": q.mover}),
                                  *(T.Said("mover-earlier-takes-label" if p.owner == q.mover else "mover-label",
                                           {"owner": p.owner}) for p in ran)])

    def _why_moved_only_then(self, q: Missed, ordered: list):
        """L2's question of the mover that moved the value on: why only then -- its passes before (judgement 4), its
        callers, or its statement's order."""
        if not any(st.owner == q.mover for st in q.statements):
            # judgement 4: a mover that takes the row by its key moved it only then for what its passes before did
            return self._late_passes(q)
        if (up := self._why_late(q, ordered)) is not None:
            return up
        mine = [st for st in ordered if st.owner == q.mover]
        self.need(T.Said("the-statement-that-moved-the-value-on"), len(mine) == 1)
        return Ordered(mine[0], q.run, q)

    def _stuck_value_takes(self, q: Stuck, owner: str, statements: list, writers: list) -> list[T.Said]:
        """P25 (i)'s rest, stuck side (out/gm/qk-design.md, the section on P25 (i)'s rest; the investigator's u11
        MT-scouting-stuck 2): a mover judgement 3 found no run of an arm of the value for may still have taken the row
        in the window, by a statement whose time column only its own arms write there -- MT-scouting-stuck's sqlEA wrote
        assessmentTime at 13:54:18.  Read as F-4 reads a late value's (:meth:`_row_value_takes`), over the stay from the
        statuslog row that began it to the moment the rows file read the row (``_reading_moment``): M3(a)-1's
        timelines of the columns the mover's *statements* compare with the clock (the other writers left out by the map
        as there, getAchievedTasks_JEDI's selection refusing 'scouting'), and each take's pass with what it printed
        (:meth:`_unprinted_take`); a pass the records lack is asked by its mark.  What is said; none where
        nothing is read."""
        states = self.data.get("statuslog", {}).get("states", [])
        rows = [s for s in states if s["status"] == q.observed]
        as_of = self.data.get("statuslog", {}).get("as_of")
        if not rows or not as_of or not statements or not self.logs.needs_window:
            return []
        at, words, _ = self._reading_moment(q.subject, as_of)
        since = _when(str(rows[-1].get("time_extended") or rows[-1]["time"]))
        if since >= at:
            return []
        mq = Missed(q.subject, q.observed, "", since, at, statements, owner, None, end=words, start=words,
                    writers=writers, began=since)
        # the time columns its terms compare with the clock, as the late reading's (``_term_left``) read them
        trees = [r.tree for st in statements for r in [*st.condition_trees, *(t for c in st.call_readings for t in c.condition_trees)]]
        for column in sorted({leaf.subject for tree in trees for leaf in _tree_leaves(tree)
                              if leaf.node == "compare" and leaf.settled and leaf.values
                              and all(v.base in UTC_CLOCKS and v.offset_seconds is not None for v in leaf.values)}):
            self._timeline(column, mq)
        rel, _, func = owner.partition("::")
        try:
            fn = R.function(rel, func, [st.line for st in statements])
        except (LookupError, ValueError, SyntaxError):
            return []
        return self._row_value_takes(mq, fn, at, [])

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
        rows = [s for s in self.data.get("statuslog", {}).get("states", []) if s["status"] == q.observed]
        mq = Missed(q.subject, q.observed, "", at, at, [rs for rs in follow.reader_statements if asked in rs.selects], "", None,
                    end=words, start=words, at_end=True, amounts=W.stuck(self.data.get("statuslog", {}), q.observed),
                    began=_when(str(rows[-1].get("time_extended") or rows[-1]["time"])) if rows else None)
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
        ``by key`` (a writer, which takes a row by its key, its logs not asked: a run not found would not rule it out,
        KNOWN_ISSUES.md K15); judgement 3's verdicts for a writer whose logs were asked (``_late_absence``: ``ran``,
        ``did not run``, ``no run found``, ``not searched``, ``asks unanswered``, ``no log``, ``prints nothing``);
        ``undecided`` (what is missing is said).  ``(owner, verdict)`` for each, in the order printed."""
        whole = (q.since, q.moved)
        guarded = {o for o, why in q.movers.items() if why == GUARDED}
        listed = list(dict.fromkeys([*q.movers, *q.by_key, *q.refusing] + ([q.mover] if q.mover not in q.movers and q.mover not in q.by_key else [])))
        self.say(f"movers in the window ({q.since} .. {q.moved}): {len(listed)}")
        # judgement 3's window, S1's rule with the move for as_of, and the movers it may ask (the door's owners, made
        # at the first ask: their marks' heads are read off the corpus)
        since, end = f"{q.since:%Y-%m-%d %H:%M:%S}", f"{q.moved:%Y-%m-%d %H:%M:%S}"
        asking = W.last_part(q.observed, since, end, "the-move-less", T.Said("the-move-at", {"move": end}))
        rest = W.rest_of(q.observed, since, end, asking)
        owners = functools.cache(lambda: owners_of([o for o in listed if o != q.mover and o not in q.refusing
                                                    and arms_of(q.writers, o, q.subject, q.observed)]))
        out = []
        entries = []
        for owner in listed:
            rel, func = owner.split("::", 1)
            name = f"{Path(rel).name}::{func}"
            armed = arms_of(q.writers, owner, q.subject, q.observed)
            paths = q.paths.get(owner, [])
            said: list[T.Said | str] = []
            spans = {}
            for st in paths:
                where = f"{st.owner.split('::')[-1]}@{st.line}"
                kind = kinds.get((st.owner, st.line), "not read")
                true = self._statement_true(st, q)
                if true:
                    spans[where] = true
                said.append(T.Said("statement-kind", {"where": where, "kind": kind, "holds": T.Said(
                    "holds-for-the-row", {"spans": _spans_words(true, whole)}) if true else ""}))
            reasons = []
            premises = None
            if made := births_of(q.writers, owner, q.subject, q.observed):
                said.append(T.Said("births-not-arms", {"made": made, "births": birth_why(owner, q.subject, made)}))
                reasons.append({"code": "K29", "record": said[-1]})
            if owner == q.mover:
                verdict = "moved it"
                said.insert(0, f"at {q.moved} (L1)")
            elif owner in q.refusing:
                verdict = "write refuses"
                said = [T.Said("writes-refuse-a-row", {"observed": q.observed,
                                                       "arms": refused_arms(q.writers, owner, q.subject, q.observed)})]
                reasons.append({"code": "write-refuses", "record": said[0]})
            elif owner not in q.movers and q.reach is not None and q.reach.leaves_out(owner):
                # M2: no read of the value's rows reaches it, only reads of other values, and no API endpoint: not a
                # mover off the value (value_reach)
                verdict = "not reached"
                said = [T.Said("not-reached-from-the-value-s-reads", {"observed": q.observed, "reads": list(q.reach.reads)})]
                reasons.append({"code": "M2", "record": said[0]})
            elif not armed and owner not in guarded:
                if paths and all(kinds.get((st.owner, st.line)) == "refused" for st in paths):
                    verdict = "refused"
                    reasons.append({"code": "refused", "record": T.Said("statements-refused", {"said": list(said)})})
                else:
                    verdict = "could have" if spans else "undecided"
                    said.append(T.Said("no-arm-through-calls"))
                    reasons.append({"code": "K14", "record": said[-1]})
            elif armed:
                verdict, premises = self._late_absence(q, owner, armed, "could have" if spans else "by key", said, reasons,
                                                       asking, rest, owners)
            else:
                verdict = "could have" if spans else "by key"
                said.append(T.Said("taken-by-its-key"))
                reasons.append({"code": "K15", "record": said[-1]})
            gaps = _gaps_json(q.gaps.get(owner) or [])
            if gaps:
                gap = T.Said("map-could-not-read-all", {"gaps": gaps})
                said.append(gap)
                reasons.append({"code": "map-gap", "record": gap})
            if verdict not in ("moved it", "write refuses", "refused", "not reached"):
                # what keeps each statement from a verdict: a value the rows file does not hold, a run-time key
                for st in paths:
                    where = f"{st.owner.split('::')[-1]}@{st.line}"
                    if st.keys:
                        reasons.append({"code": "K18", "record": T.Said("asks-rows-by-run-time-value",
                                                                        {"where": where, "keys": list(st.keys)})})
                    lacked = []
                    for subject in sorted(_statement_subjects(st)):
                        if why := self._unanswered_why(subject, q):
                            reasons.append({"code": why[0], "record": T.Said("not-compared", {"where": where, "subject": subject,
                                                                                              "why": why[1]})})
                            lacked += [subject] if why[0] == "rows-no-row" else []
                    if kinds.get((st.owner, st.line)) != "refused":
                        said += [f"{where}: {note}" for note in self.ask_linked(st, lacked, q)]
            self.say(f"  {name}: {verdict} -- {'; '.join(map(T.text, said)) if said else q.movers.get(owner, 'a writer of another value')}")
            out.append((owner, verdict))
            entries.append({"owner": owner, "reached": "select" if paths else "guard" if owner in guarded else "key",
                            "statements": [f"{st.owner.split('::')[-1]}@{st.line}" for st in paths],
                            "verdict": verdict.replace(" ", "-"), "reasons": reasons, "premises": premises,
                            **({"gaps": gaps} if gaps else {})})
        self.result["movers"] = entries
        left = [o for o, v in out if v not in ("moved it", "write refuses", "refused", "not reached", "did not run")]
        self.say(f"not ruled out: {len(left)} mover(s)")
        return out

    def _late_absence(self, q: Missed, owner: str, arms: dict, verdict: str, said: list, reasons: list,
                      asking: W.Asking, rest: W.Asking | None, owners: Callable[[], Owners]) -> tuple[str, dict | None]:
        """L2's judgement 3 for a mover with arms of its own, S1's part (``_absence``) over the window's last part to
        the move.  Its logs are asked only where the two premises the map decides hold -- every log its runs print to
        is to hand, and every run that reaches an arm prints the key -- since a run not found rules nothing out
        otherwise; one with no prints of its own says so, as in S1.  The asks no record answers go to the asks file and
        the walk goes on (tmaeno, 2026-11-01): L2's movers are a late answer's context.  A mover that ran is kept in
        ``_late_ran``, with the stay, for ``missed`` to follow on its own: which way its passes in the stay went, and why
        they did not move the value (KNOWN_ISSUES.md K15).  Returns the verdict -- *verdict* where its logs are not
        asked -- and the premises read."""
        lines, key = list(arms), self.args.key
        today = [f for j in q.writers if j.owner == owner for f in j.log_files] or q.reader_log_files
        way = "select" if q.paths.get(owner) else "guard" if q.movers.get(owner) == GUARDED else "key"
        said.append(f"a writer {sorted(arms)}, which takes a row by its key too")
        if not prints_nothing(owner, lines):
            to_hand = self.logs.names(self.files_of(owner, list(dict.fromkeys(today)), lines))
            static = {"logs": every_log_asked(owner, lines, to_hand),
                      "prints": prints_naming(q.writers, owner, lines, self._key_columns(q.subject, key))}
            if failing := {k: p for k, p in static.items() if not p["held"]}:
                said.append(T.Said("logs-not-asked-not-ruled-out", {"failing": [
                    T.Said("premise-why", {"premise": k, "why": p["why"]}) for k, p in failing.items()]}))
                reasons.append({"code": "K15", "record": said[-1]})
                return verdict, {"window": {"held": None, "why": T.Said("window-not-asked")}, **static}
        entry: dict = {"verdict": None, "reasons": [], "premises": None}
        passes, missed = self._absence(owner, arms, today, entry, q.writers, q.subject, key, asking, rest, way, owners())
        self._keep_unanswered(missed)
        reasons += entry["reasons"]
        if passes:
            # a mover that ran is followed on its own, as S1 follows a stuck value's: why its passes in the stay did not
            # move the value (analyzer handover qk-design.md (M), M3)
            stay = (f"{q.since:%Y-%m-%d %H:%M:%S}", f"{q.moved:%Y-%m-%d %H:%M:%S}")
            self._late_ran = (*self._late_ran, replace(passes, since=stay[0], stay=stay))
            said.append(T.Said("ran-followed-for-late", {"logs": passes.logs, "window": list(passes.asking.window or ())}))
            reasons.append({"code": "K15", "record": said[-1]})
        elif entry["verdict"] == "asks-unanswered":
            said.append(T.Said("asks-of-its-logs-not-recorded", {"n": len({ask.key for ask, _ in missed}), "why": asking.why}))
            reasons.append({"code": "asks", "record": said[-1]})
        return entry["verdict"].replace("-", " "), entry["premises"]

    def _late_passes(self, q: Missed) -> Passes:
        """Judgement 4: the mover that moved the value on takes the row by its key, so it moved it only then for what
        its own passes before did (KNOWN_ISSUES.md K15).  Its logs are asked from the stay's start to the move for its
        marks holding the key (``key_marks``): a bare key over hours takes every listing of a DB proxy's log (tmaeno,
        10-26).  S2 then reads the passes ending before the second it moved on, as it reads a stuck mover's."""
        rel, func = q.mover.split("::", 1)
        arms = arms_of(q.writers, q.mover, q.subject, q.observed)
        self.need(T.Said("an-arm-of-that-writes-another-value", {"func": func, "q_observed": q.observed}), arms)
        cls = q.subject.partition(".")[0]
        marks = tuple(sorted({m for line in arms for m in key_marks(q.mover, line, cls, self.args.key)}))
        files = self.files_of(q.mover, list(dict.fromkeys(f for j in q.writers if j.owner == q.mover for f in j.log_files)), list(arms))
        to_hand = self.need(T.Said("a-log-of-to-hand", {"func": func, "files": files}), self.logs.names(files))
        since, before = f"{q.since:%Y-%m-%d %H:%M:%S}", f"{q.moved:%Y-%m-%d %H:%M:%S}"
        how = T.Said("for-marks", {"marks": list(marks)}) if marks else T.Said("for-the-key-alone")
        asking = W.Asking((since, before), None, T.Said("from-the-stays-start", {"how": how}), marks)
        self.say(f"  {func}: {len(arms)} arm(s) that write another value {[(a.line, a.outcomes) for a in arms.values()]}; "
                 f"logs {files}, asked {asking.why}")
        found, missed, holes = self._ask(self.args.key, to_hand, asking)
        self._stop_unanswered(missed)
        # no pass found says none ran only where every line of its runs holds the key and the window was searched whole
        unasked = [f for f in files if f not in to_hand]
        unsure = [*([] if marks else [T.Said("a-pass-printing-no-key-not-found")]),
                  *([T.Said("files-not-searched-all-the-window", {"files": list(holes)})] if holes else []),
                  *([T.Said("files-not-to-hand", {"files": unasked})] if unasked else [])]
        if holes:
            self.say(f"  {func}: {', '.join(holes)} not searched all the window: a pass there is not read")
        return Passes(q.mover, self._runs_of(found, q.mover, asking), list(arms.values()), self.args.key, asking,
                      since=since, before=before, unsure=unsure, subject=q.subject, entries=_entries_of(q.writers, q.mover))

    def _why_late(self, q: Missed, ordered: list) -> Callers | OtherRows | Turned | None:
        """Judgement 1: why the mover that moved the value on did it only then, where a statement of its own took the
        row.  One that returns the first rows of an order goes on to L3 (lost on order); one that took the row in no
        such order goes to its callers' passes before the move (U1: what it took the row by is what they handed it,
        KNOWN_ISSUES.md K55), or stops, saying what is not read yet, where the map lists none or the window is not
        known.  One that takes the row by its key goes to judgement 4 (``_late_passes``)."""
        rel, func = q.mover.split("::", 1)
        name = f"{Path(rel).name}::{func}"
        took = [st for st in q.statements if st.owner == q.mover]
        if any(st.owner == q.mover for st in ordered):
            return None
        spans = _union([span for st in took for span in self._statement_true(st, q)])
        where = ", ".join(f"{st.owner.split('::')[-1]}@{st.line}" for st in took)
        held: str | T.Said = f"held for the row {_spans_words(spans, (q.since, q.moved))}" if spans else "held for the row at times not known"
        if not spans:
            # Q1 part 1: where it holds with the arms no record reads named as having not written what it compares
            if_spans, assumed = self._assumed_true(took, q)
            if if_spans and assumed:
                held = T.Said("held-for-the-row-if", {"spans": _spans_words(if_spans, (q.since, q.moved)),
                                                      "premises": self._held_premises(assumed, {}, f"{q.moved}")})
            else:
                # F-1: its selection took the row in a pass before, which did not move it: that pass's way is read
                takes, earlier = self._earlier_takes(q)
                if takes:
                    held = T.Said("held-at-times-not-known-taken-before" if earlier
                                  else "held-at-times-not-known-taken-before-unread", {"takes": takes})
                self._late_ran = (*self._late_ran, *earlier)
        said = T.Said("why-took-it-only-at-and-it", {"name": name, "q_moved": q.moved, "where": where, "held": held})
        entries = _entries_of(q.writers, q.mover)
        since, before = f"{q.since:%Y-%m-%d %H:%M:%S}", f"{q.moved:%Y-%m-%d %H:%M:%S}"
        then: Callers | T.Said = said
        if entries and self.logs.needs_window:
            arms = list(arms_of(q.writers, q.mover, q.subject, q.observed).values())
            then = Callers(q.mover, arms, q.subject, self.args.key, (since, before), "statement", entries, before=before)
        # L4: when its statements turned for the row answers first, where they were surely false for it before the move
        span, assumed, false = self._turn_of(took, q) if self.logs.needs_window else (None, {}, [])
        if span is not None:
            after: OtherRows | Callers | T.Said = then
            if self._hop_terms(q.mover, q.subject):
                after = OtherRows(q.mover, q.subject, since, before, before, then)
            self.say(f"{said}: when its statement turned for the row is read first")
            return Turned(q, took, span, assumed, false, after)
        # H: a statement of its waiting on other rows answers first (KNOWN_ISSUES.md K77)
        if self._hop_terms(q.mover, q.subject) and self.logs.needs_window:
            self.say(f"{said}: what other rows it waited on is read first")
            return OtherRows(q.mover, q.subject, since, before, before, then)
        if isinstance(then, Callers):
            self.say(f"{said}: its callers' passes before are read (KNOWN_ISSUES.md K55)")
            return then
        self.need(said, None)
        return None

    def _earlier_takes(self, q: Missed) -> tuple[list[T.Said], list[Passes]]:
        """Bundle F, F-1 (out/gm/qk-design.md, bundle F, tmaeno's OK 2026-11-17 15:2x; P25 (ii)): where none of the
        mover's selecting statements is read as holding for the row in the stay, its passes before the one that moved
        it on whose selection's loop printed the row's key (:meth:`_row_key_print`, under the loop's ifs too) -- each
        took the row then and did not move it.  They are read from the key's records alone, as U2 reads a caller's
        callees (the key in the prints' logs over the stay to the moving pass, no marks; a pass printing no key is not
        found, and nothing is said of one).  Each take's pass is then read from the mover's marks' heads on its
        machine (:meth:`_take_pass`, H-1) and goes to S2 as judgement 4's do: which way it went.  MT-scouting-late's
        sqlEA (3739) took the row at 09:24:32.303 on aipanda085, printing 3766's ``... finished for reset in
        scouting``, and that pass wrote scout data only.  ``(what is said of each take, the passes for S2)``; ``([],
        [])`` where none is read, the asks of a log not in the records kept."""
        own = [st for st in q.statements if st.owner == q.mover]
        if not own or q.run is None or not self.logs.needs_window:
            return [], []
        rel, _, name = _bare(q.mover).partition("::")
        try:
            fn = R.function(rel, name, [st.line for st in own])
        except (LookupError, ValueError, SyntaxError):
            return [], []
        parts, _ = cut_passes(fn, cast(Any, q.run), self.mode)
        begun = [p for p in parts if p.records and W.when(p.records[0].stamp) <= q.moved + datetime.timedelta(seconds=1)]
        start = W.when(begun[-1].records[0].stamp) if begun else q.moved
        prints = {line: st for st in own if (line := self._row_key_print(q.mover, st, q.subject, under_ifs=True)) is not None}
        found = R._prints_of(q.mover, sorted(prints)) if prints else None
        files = sorted({f for pst in getattr(found, "statements", None) or [] if pst.line in prints for f in pst.log_files})
        to_hand = self.logs.names(files)
        arms = list(arms_of(q.writers, q.mover, q.subject, q.observed).values())
        if not to_hand or not arms or start <= q.since:
            return [], []
        bare = name.rpartition(".")[2]
        since, before = f"{q.since:%Y-%m-%d %H:%M:%S}", f"{start:%Y-%m-%d %H:%M:%S}"
        asking = W.Asking((since, before), None, T.Said("for-the-key-before-the-moving-pass", {
            "func": bare, "since": since, "before": before}))
        runs, missed, _ = self._ask(self.args.key, to_hand, asking, {}, quiet=True)
        if missed:
            self._keep_unanswered(missed)
            self.say(f"{bare}: {len({ask.key for ask, _ in missed})} ask(s) of its logs from the stay's start to its "
                     "moving pass are not in the records: its passes before are not read")
            return [], []
        takes: dict[tuple[str, str], list[tuple[int, int]]] = {}
        uncut: set[tuple[str, str]] = set()
        tokens: dict[tuple[str, str], str] = {}
        for log_name in to_hand:
            for run in runs.get(log_name, []):
                if not of_mover(run, q.mover):
                    continue
                for p in cut_passes(fn, run, self.mode)[0]:
                    if not p.records or W.when(p.records[0].stamp) >= start:
                        continue
                    lines = sorted(set(self._key_rows(fn, p, q.subject)) & set(prints))
                    if lines and W.when(p.records[-1].stamp) >= start:
                        # a run of the key's records alone that holds no record of the entry row is not cut into
                        # passes (MT-scouting-late's on aipanda085, 09:24:32 .. 18:29:47): the take is the print's own
                        # record before the moving pass, and which way its pass went is not read
                        hits = R.seen(fn, p, self.mode).hits
                        before_start = {line: min(st) for line in lines
                                        if (st := [p.records[i].stamp for i, _, _ in hits.get(line, ())
                                                   if W.when(p.records[i].stamp) < start])}
                        if not before_start:
                            continue
                        at = min(before_start.values())
                        lines = sorted(before_start)
                        uncut.add((at, run.machine))
                    else:
                        at = p.records[0].stamp
                    for line in lines:
                        takes.setdefault((at, run.machine), []).append((prints[line].line, line))
                    tokens.setdefault((at, run.machine), getattr(run, "token", "") or "")
        by_value = self._row_value_takes(q, fn, start, [W.when(at) for at, _ in takes])
        if not takes:
            return by_value, []
        said, branches, result = [], [], []
        for (at, machine), ways in sorted(takes.items()):
            how, passes, where = self._take_pass(q, at, machine, start, arms, since, before, tokens.get((at, machine), ""))
            words = {"func": bare, "at": at[:23], "machine": machine.split(".")[0],
                     "lines": [f"@{st} (printing the row at {line})" for st, line in sorted(set(ways))], **where}
            said.append(T.Said({"read": "took-it-before", "asked": "took-it-before-heads-asked",
                                "heads-uncut": "took-it-before-heads-uncut"}.get(how, "took-it-before-uncut"), words))
            branches += [passes] if passes is not None else []
            result.append({"at": at, "machine": machine, "statements": sorted({st for st, _ in ways}),
                           "cut": (at, machine) not in uncut, "pass": how})
        self.result["earlier_takes"] = result
        return said + by_value, branches

    def _take_pass(self, q: Missed, at: str, machine: str, start: datetime.datetime, arms: list, since: str,
                   before: str, token: str = "") -> tuple[str, Passes | None, dict]:
        """Bundle H, H-1 (F-1b, out/gm/qk-design.md, bundle H, tmaeno's OK 2026-11-17 17:1x): the pass of a take F-1
        found, asked by the mover's marks' heads on its machine, ``windows.SLACK`` each side of the take (F-4's ask),
        so the records hold its entry row and its prints without the key: the key's records alone hold neither
        (MT-scouting-late's sqlEA pass at 09:24:32 on aipanda085, its 3641 'start' not among them).  It goes to S2 as
        judgement 4's passes do where S2's own cut (``passes_of``) holds a pass around the take ending before the moving
        pass; else nothing is handed, and why is said.  The run's own mark (*token*, the key's records hold it) is
        asked where known: the heads fill their holes with the row's values, and the agent's call may not
        (prepareTasksToBeFinished_JEDI's runs are marked ``vo=None``, the task's row holding 'atlas').  ``(how,
        passes, where)``: ``read``, ``asked`` (the asks kept), ``heads-uncut``, or ``no-heads``."""
        rel, _, func = _bare(q.mover).partition("::")
        lines = sorted({a.line for a in arms} | {st.line for st in q.statements if st.owner == q.mover})
        heads = (token,) if token else self._row_mark_heads(q.mover, lines)[0]
        names = self.logs.names(self.files_of(q.mover, [], lines))
        t = W.when(at)
        lo, hi = W._down(max(q.since, t - W.SLACK)), W._down(min(start, t + W.SLACK))
        where = {"first": lo, "last": hi}
        if not heads or not names:
            return "no-heads", None, where
        owners = owners_of([q.mover])
        asking = W.Asking((lo, hi), machine, T.Said("the-take-s-pass-by-heads", {
            "func": func.rpartition(".")[2], "at": at[:23], "machine": machine.split(".")[0], "first": lo, "last": hi}), heads)
        runs, missed, _ = self._ask(self.args.key, names, asking, owners, quiet=True)
        if missed:
            self._keep_unanswered(missed)
            return "asked", None, {**where, "n": len({ask.key for ask, _ in missed})}
        try:
            fn = R.function(rel, func, [a.line for a in arms])
        except (LookupError, ValueError, SyntaxError):
            return "heads-uncut", None, where
        for name in names:
            for run in runs.get(name, []):
                if not of_mover(run, q.mover):
                    continue
                parts, how = passes_of(fn, run, self.mode)
                if not how.startswith("cut at"):
                    continue
                if any(p.records and W.when(p.records[0].stamp) <= t <= W.when(p.records[-1].stamp) < start for p in parts):
                    return "read", Passes(q.mover, names, arms, self.args.key, asking, owners, since=since, before=before,
                                          subject=q.subject, entries=_entries_of(q.writers, q.mover),
                                          stay=(since, before)), where
        return "heads-uncut", None, where

    def _row_value_takes(self, q: Missed, fn: R.Function, start: datetime.datetime,
                         read: list[datetime.datetime]) -> list[T.Said]:
        """Bundle F, F-4 (out/gm/qk-design.md, bundle F widened, tmaeno's OK 2026-11-17 15:2x; P25 (iii)): a time
        column the mover's selection terms read -- a timeline already built for the question, no new one -- whose row
        value only the mover's own arms wrote (F-3's narrowing), inside the stay before its moving pass: its pass then
        took the row.  One the key's records hold a pass of at that second (*read*, F-1's) is said there.  Otherwise
        its passes around the write are read by its marks' heads on every machine its log is on, ``windows.SLACK``
        each side, as L4 reads them; a log not in the records keeps its asks.  NT-scouting-late's assessmentTime,
        19:53:41, by prepareTasksToBeFinished_JEDI:3744 (sqlEA's lock of the rows it took): that pass printed no key.
        What is said of each."""
        own = SV._name(q.mover, 0).rpartition(":")[0]
        rel, _, func = _bare(q.mover).partition("::")
        bare = func.rpartition(".")[2]
        out: list[T.Said] = []
        cache: dict = self.__dict__.get("_timelines", {})
        seen: set[tuple[str, datetime.datetime]] = set()
        for key, tl in sorted(cache.items(), key=lambda x: str(x[0])):
            # the key as :meth:`_timeline` makes it; a reading that asked nothing (its last item) is read alike
            col, subject, observed, since, moved, nulls = key[:6]
            if (nulls or tl is None or not tl.row_value
                    or (subject, observed, since, moved) != (q.subject, q.observed, q.since, q.moved)):
                continue
            at, arms = tl.row_value[0], tuple(tl.row_value[2])
            # a row value of a DATE column names its second: the write fell in [at, at + 1 s), so it is before the
            # moving pass only where that second ended by the pass's start (MT-scouting-late's 12:47:03 was the
            # moving pass's own write at 12:47:03.942)
            if (not arms or any(a.rpartition(":")[0] != own for a in arms)
                    or not (q.since < at and at + datetime.timedelta(seconds=1) <= start)):
                continue
            if any(abs(t - at) <= datetime.timedelta(seconds=1) for t in read):
                continue
            # one take a column's value names, however many readings of the column the cache keeps (one that asked
            # nothing beside one that asked: NT-scouting-late's said its 19:53:41 take twice)
            if (col, at) in seen:
                continue
            seen.add((col, at))
            lines = sorted({int(a.rpartition(":")[2]) for a in arms})
            # the moving run's own mark, where it is the mover's: its passes before ran under it too (the heads fill
            # the holes with the row's values, and prepareTasksToBeFinished_JEDI's runs are marked vo=None)
            token = getattr(q.run, "token", "") if q.run is not None and of_mover(cast(Any, q.run), q.mover) else ""
            heads = (token,) if token else self._row_mark_heads(q.mover, lines)[0]
            names = self.logs.names(self.files_of(q.mover, [], lines))
            stamp = f"{at:%Y-%m-%d %H:%M:%S}"
            where = {"func": bare, "column": col.partition(".")[2], "at": stamp, "arms": list(arms)}
            if not heads or not names:
                out.append(T.Said("took-it-by-the-row-value-unread", where))
                continue
            window = (W._down(at - W.SLACK), W._down(at + W.SLACK))
            because = (f"which pass of {bare} wrote {col} at {stamp}, the row's value, taking the row: its mark holds no "
                       "key, so it is asked by its mark (F-4)")
            owners = owners_of([q.mover])
            missed: list = []
            passes: list = []
            for name in names:
                try:
                    listed = self.logs.log(name, self.args.key, window, owners=owners)
                except Unrecorded as e:
                    missed += [(ask, because) for ask, _ in e.asks]
                    continue
                for machine in sorted(listed.machines - listed.without):
                    try:
                        log = self.logs.log(name, self.args.key, window, machine=machine, owners=owners, strings=heads)
                    except Unrecorded as e:
                        missed += [(ask, because) for ask, _ in e.asks]
                        continue
                    for run in runs_for(log, bare):
                        if of_mover(run, bare):
                            passes += [p for p in passes_of(fn, run, self.mode)[0]
                                       if W.when(p.first) - datetime.timedelta(seconds=1) <= at <= W.when(p.last) + datetime.timedelta(seconds=1)]
            if missed:
                self._keep_unanswered(missed)
                out.append(T.Said("took-it-by-the-row-value-asked", {**where, "n": len({ask.key for ask, _ in missed}),
                                                                     "first": window[0], "last": window[1]}))
            elif passes:
                for p in passes:
                    out.append(T.Said("took-it-by-the-row-value", {**where, "machine": p.machine.split(".")[0],
                                                                   "first": p.first[:23], "last": p.last[:23]}))
                    # P25 (i)'s rest: a take that printed no key at its selection's print
                    if (unprinted := self._unprinted_take(q, fn, p, lines)) is not None:
                        out.append(unprinted)
            else:
                out.append(T.Said("took-it-by-the-row-value-unread", where))
        return out

    def _unprinted_take(self, q: Missed, fn: R.Function, p, lines: list[int]) -> T.Said | None:
        """P25 (i), its rest (out/gm/qk-design.md, the section on P25 (i)'s rest, delegated design; the investigator's u11
        NT-scouting-late u3:4): a pass F-4 read as taking the row -- writing the time column a selecting term reads, by
        *lines*, keyed by the rows of one of the mover's selections (the map's KeyFrom sources) -- that printed no key
        at that selection's print for a row it selects and keeps (:meth:`_row_key_print` under its ifs): the print's
        condition was false for the row then.  The condition is the print's tests (``pathcond.arm_condition``) less
        those the key's line in the loop ran under, a flag's test read as its true bindings' tests
        (:meth:`_flag_tests`).  Its names are bound as :meth:`_unprinted_env` reads them; the atoms of the pass's own
        values alone (a rowcount, a container, a parameter: sources neither the row's nor constants) are put the way
        the print needs them where that way is one, and said as premises; the rest are tried both ways
        (``evaluate.truth``).  Where the print would hold for every count of the name counting a value (``COUNT`` of a
        column holding it) from 1 to a bound past every number the condition holds, the count was 0 -- the hand's
        MT-scouting-stuck reading: walltimeUnit NULL, so a finished scout prints 'reset in scouting'.  Else the
        condition was false, its open atoms said.  None where no such print or no one selection."""
        fragment = mapdump.fragment()
        keyed = [k for k in getattr(fragment, "key_froms", None) or [] if _bare(str(k.owner)) == _bare(q.mover)
                 and any(set(ls) & set(lines) for ls in (getattr(k, "writes", None) or {}).values())]
        sources = [s for k in keyed for s in getattr(k, "sources", None) or [] if getattr(s, "kind", "") == KEY_ROWS]
        if not sources:
            return None
        own = [st for st in getattr(q, "statements", None) or []
               if st.owner == q.mover and st.line in {getattr(s, "site_line", None) for s in sources}]
        if len(own) != 1:
            return None
        st = own[0]
        printed_at = self._row_key_print(q.mover, st, q.subject, under_ifs=True)
        if printed_at is None or printed_at in self._key_rows(fn, p, q.subject):
            return None
        pstmt = R.statement_at(fn.node, printed_at)
        if pstmt is None:
            return None
        # the tests the key's line in the loop ran under held: the pass took the row there
        ran = {c for s in sources if getattr(s, "site_line", None) == st.line
               and (node := R.statement_at(fn.node, int(getattr(s, "line", 0) or 0))) is not None
               for c in pathcond.arm_condition(node)}
        tests = [self._flag_tests(fn, c, pstmt) for c in pathcond.arm_condition(pstmt)
                 if c not in ran and not E.shared.read_elsewhere(c)]
        if not tests:
            return None
        try:
            tree = ast.parse(" and ".join(f"({E.shared.strip_annotation(t)})" for t in tests), mode="eval").body
        except SyntaxError:
            return None
        names: dict[str, list] = {}
        for by in self._count_guards(fragment, q.mover, st.line).values():
            for name, srcs in by.items():
                names.setdefault(name, srcs)
        bound, said, counted = self._unprinted_env(names, q, p.first[:19])
        own_values = {n for n, srcs in names.items() if srcs and all(getattr(x, "kind", "") not in (KEY_ROWS, KEY_CONSTANT) for x in srcs)}
        zero, way, unknown, top = _unprinted_count(tree, bound, counted, own_values)
        where = {"func": q.mover.split("::")[-1].rpartition(".")[2], "line": printed_at, "condition": ast.unparse(tree),
                 "read": said, "premises": [T.Said("as-the-print-needs", {"atom": t, "held": v}) for t, v in way.items()]}
        if zero and counted is not None:
            return T.Said("unprinted-take-count-zero", {**where, "name": counted, "n": top})
        return T.Said("unprinted-take", {**where, "open": unknown})

    @staticmethod
    def _flag_tests(fn: R.Function, test: str, at: ast.AST) -> str:
        """P25 (i)'s rest: *test*, a print's, read as what sets it where it is a flag -- a bare name every binding of
        which in *fn* is ``True`` or ``False``, each ``False`` one run on every way to *at* (the reset before the
        tests): the tests its ``True`` bindings run under beyond *at*'s others, or-ed (sqlEA's ``to_trigger``: 3759's
        or 3762's, the elif's own with the if's negation).  The map's ``guard_froms`` list a flag's constant sources by
        value (3761's ``True`` and not 3764's), so the bindings are read with the map's shared part.  *test* where it
        is no flag."""
        try:
            node = ast.parse(test.strip(), mode="eval").body
        except SyntaxError:
            return test
        if not isinstance(node, ast.Name):
            return test
        binds = pathcond.bindings(fn.node, node.id)
        if not binds or any(b.nested or not isinstance(b.value, ast.Constant) or not isinstance(b.value.value, bool)
                            for b in binds):
            return test
        others = set(pathcond.arm_condition(at)) - {test}
        if any(not set(pathcond.arm_condition(b.site)) <= others for b in binds
               if isinstance(b.value, ast.Constant) and b.value.value is False):
            return test
        ways = [" and ".join(f"({E.shared.strip_annotation(c)})" for c in pathcond.arm_condition(b.site)
                             if c not in others and not E.shared.read_elsewhere(c)) or "True"
                for b in binds if isinstance(b.value, ast.Constant) and b.value.value is True]
        return " or ".join(f"({w})" for w in ways) if ways else "False"

    def _unprinted_env(self, names: dict[str, list], q: Missed, at: str) -> tuple[dict[str, Any], list[T.Said], str | None]:
        """P25 (i)'s rest: the print's names as the take's pass held them, where known, each said: a constant source
        its literal; the row's own status column its value in the stay; another column of the row the rows file's
        value where no statuslog row lies between the pass (*at*) and the file's read (K22: a write that leaves no row
        is not read).  And the one name counting a value (a ``COUNT`` of a column holding it), left for the counts
        tried.  ``(values, said, counted)``."""
        cls, _, attr = q.subject.partition(".")
        fragment = mapdump.fragment()
        tables = {str(t.name).lower() for t in getattr(fragment, "tables", None) or [] if getattr(t, "rows_of", "") == cls}
        entity = next((e for e, c in self.data.get("classes", {}).items() if c == cls), None)
        row = {str(c).lower(): cell for c, cell in (self.data.get("rows", {}).get(entity) or {}).items()} if entity else {}
        states = (self.data.get("statuslog") or {}).get("states") or []
        values: dict[str, Any] = {}
        said: list[T.Said] = []
        counted = sorted({n for n, srcs in names.items() for s in srcs
                          if (agg := getattr(s, "aggregate", None)) is not None and getattr(agg, "fn", "") == "COUNT"
                          and getattr(agg, "value", "")})
        for name, srcs in sorted(names.items()):
            if len(srcs) != 1 or name in counted:
                continue
            s = srcs[0]
            table, _, col = str(getattr(s, "column", "") or "").lower().partition(".")
            if s.kind == KEY_CONSTANT and not getattr(s, "changes", None):
                try:
                    values[name] = ast.literal_eval(str(s.name))
                except (ValueError, SyntaxError):
                    continue
            elif s.kind == KEY_ROWS and table in tables and getattr(s, "aggregate", None) is None:
                if col == attr.lower():
                    values[name] = q.observed
                    said.append(T.Said("unprinted-read", {"name": name, "value": q.observed, "why": T.Said("unprinted-the-stay")}))
                elif (isinstance(cell := row.get(col), dict) and "value" in cell
                      and (not cell.get("as_of") or not HD.rows_after(states, at, str(cell["as_of"])))):
                    values[name] = cell["value"]
                    said.append(T.Said("unprinted-read", {"name": name, "value": cell["value"], "why": T.Said(
                        "unprinted-the-rows-file", {"as_of": str(cell.get("as_of") or "")})}))
        return values, said, (counted[0] if len(counted) == 1 else None)

    def other_rows(self, q: OtherRows):
        """O1 (design H, KNOWN_ISSUES.md K77): :meth:`_hop_turn` for the mover's statements; where it does not hold,
        what *q* says is read next."""
        self.start("O1", "what other rows its selection waited on")
        func = q.owner.split("::")[-1]
        if self._hop_turn(q.owner, q.subject, q.since, q.before, q.until, func):
            return None
        # P25 (i): a count the taking selection groups, turned at the k-th of the rows it counts
        if q.since is not None and self._count_turn(q.owner, q.subject, q.since, q.before, func):
            return None
        if isinstance(q.then, Callers):
            self.say(f"{func}: its callers' passes before are read (KNOWN_ISSUES.md K55)")
            return q.then
        self.need(q.then, None)
        return None

    def turned(self, t: Turned):
        """L4 (out/gm/qk-design.md, next 2, tmaeno's OK 2026-11-16 14:27): why the mover took the row only at the move,
        read on the false side.  Its statements that took it were surely false for the row from *t*'s span's start to
        its end, the turn, so a pass of it that ended before the turn did not take the row.  Its passes are read from
        the runs of its marks: they hold no key, so they are asked by their heads, each hole the row's values fill
        filled (:meth:`_row_mark_heads`), over [turn, move] on each machine the key's records list its log on; a run of
        another row's values is left out (B-2).
        Where the span reaches back to the stay's start and no pass of it on any machine began between the turn and the
        pass that took the row -- the last holding the key that began by the move -- that pass was its first after the
        turn: an END.  A pass between is said, and *t*'s ``then`` is read (why it took nothing then is not read); a
        machine whose ask no record answers stops the walk on the ask (``asks``)."""
        self.start("L4", "when its statement turned for the row")
        q = t.q
        start, turn = t.span
        rel, func = q.mover.split("::", 1)
        bare = func.rpartition(".")[2]
        lines = [st.line for st in t.took]
        where = ", ".join(f"{bare}@{line}" for line in lines)
        stamp = f"{turn:%Y-%m-%d %H:%M:%S.%f}"[:23]
        words = ", ".join(t.words)
        premises = self._held_premises(t.assumed, {}, f"{q.moved}")
        self.say(f"{where}: surely false for the row from {start} to {stamp} ({words or 'its terms, by turns'})"
                 + (f", where {'; '.join(map(T.text, premises))}" if premises else ""))
        if start > q.since:
            self.say(f"  not read as false from the stay's start {q.since}: its passes before {start} are not read here")
            return self._turned_then(t)
        names = self.logs.names(self.files_of(q.mover, [], lines))
        heads, used = self._row_mark_heads(q.mover, lines)
        if used:
            premises = [*premises, T.Said("its-marks-hold-the-row-s-values", {
                "pairs": [f"{c.rpartition('.')[2]}={v}" for c, v in sorted(used.items())]})]
        if not names or not heads:
            self.say(f"  {bare}'s passes are not read: " + ("no log of it is to hand" if not names else
                                                           "no mark of it has a fixed head to ask it by"))
            return self._turned_then(t)
        window = (f"{turn:%Y-%m-%d %H:%M:%S}", f"{q.moved:%Y-%m-%d %H:%M:%S}")
        key, owners = self.args.key, owners_of([q.mover])
        fn = R.function(rel, func, lines)
        pattern = keyed(key)
        because = (f"whether {bare} began a pass between its statement's turn for the row at {stamp} and the move at "
                   f"{q.moved}: its mark holds no key, so it is asked on its own (L4)")
        missed: list = []
        read: list[str] = []
        holes: list[str] = []
        others = 0
        passes: list[tuple[datetime.datetime, datetime.datetime, str, bool]] = []
        for name in names:
            try:
                listed = self.logs.log(name, key, window, owners=owners)
            except Unrecorded as e:
                missed += [(ask, because) for ask, _ in e.asks]
                continue
            holes += [f"{name}: {h}" for h in listed.holes]
            for machine in sorted(listed.machines - listed.without):
                try:
                    log = self.logs.log(name, key, window, machine=machine, owners=owners, strings=heads)
                except Unrecorded as e:
                    missed += [(ask, because) for ask, _ in e.asks]
                    continue
                read.append(f"{name}@{machine.split('.')[0]}")
                for run in runs_for(log, bare):
                    if not of_mover(run, bare):
                        continue
                    if self._another_rows_run(q.mover, lines, run, q):
                        others += 1
                        continue
                    passes += [(W.when(p.first), W.when(p.last), p.machine, any(pattern.search(r.message) for r in p.records))
                               for p in passes_of(fn, run, self.mode)[0]]
        passes = sorted(set(passes))
        took = max((p for p in passes if p[3] and p[0] <= q.moved), key=lambda p: p[0], default=None)
        between = [p for p in passes if p != took and p[1] >= turn and p[0] < (took[0] if took else q.moved)]
        before = [p for p in passes if start <= p[0] and p[1] < turn]
        left = f" ({others} run(s) of another row's values left out)" if others else ""

        def at(p: tuple) -> str:
            return f"{p[2].split('.')[0]} from {p[0]:%Y-%m-%d %H:%M:%S.%f}"[:-3]

        self.say(f"  {bare}'s passes read on {read or 'no machine'}{left}: {len(before)} ended before the turn, "
                 f"{len(between)} began between it and the pass that took the row"
                 + (f" (on {at(took)})" if took else ", which is not read"))
        if between:
            self.say(f"  a pass on {at(between[0])} after the turn did not take the row: why it did not is not read here")
            return self._turned_then(t)
        if holes:
            self.say(f"  not searched over the turn to the move: {holes}")
            return self._turned_then(t)
        if missed:
            self.say(f"  {bare}'s passes between the turn and the move are not recorded on every machine: asked by its "
                     f"marks' heads {list(heads)}")
            self._stop_unanswered(missed)
        if took is None:
            return self._turned_then(t)
        self.end(T.Said("took-it-at-its-first-pass-after-the-turn", {
            "func": bare, "where": where, "since": f"{start}", "turn": stamp, "false": words or "its terms, by turns",
            "before": len(before), "on": sorted({p[2].split(".")[0] for p in before}),
            "machine": took[2].split(".")[0], "first": at(took).partition(" from ")[2], "moved": f"{q.moved}",
            "n": len(read), "logs": read, "premises": premises or ["none"]}))
        return None

    def _turned_then(self, t: Turned):
        """What L4 reads where it does not answer: *t*'s ``then``."""
        if isinstance(t.then, OtherRows):
            self.say("what other rows its selection waited on is read next")
            return t.then
        if isinstance(t.then, Callers):
            self.say("its callers' passes before are read (KNOWN_ISSUES.md K55)")
            return t.then
        self.need(t.then, None)
        return None

    def _hop_terms(self, owner: str, subject: str) -> list[HOP.Term]:
        """The ``NOT EXISTS`` terms the map structures in *owner*'s statements selecting *subject* at the observed value."""
        sites = getattr(mapdump.fragment(), "read_sites", None) or []
        if not sites:
            return []
        return [t for t in HOP.subquery_terms(sites, owner, subject, self.args.observed) if t.polarity == "NOT EXISTS"]

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
        return self._held_once(found.subject.partition(".")[0], found.created)

    def _held_once(self, cls: str, attr: str) -> str | None:
        """The rows file's value of *cls*'s *attr*, a column set once as the row is made, from its row or its ``after``
        (:meth:`_created`); None where it lacks it."""
        value = self._row_of(cls).get(attr, {}).get("value")
        if value is None:
            value = self._read()[1].get(attr)
            value = value.get("value") if isinstance(value, dict) else value
        return str(value) if value else None

    def _made_at(self, cls: str) -> str | None:
        """When the row of *cls* was made: its creation column by the map's tables (``attr_time.creation_of``,
        ``JediTaskSpec.creationDate``), as the rows file holds it, to the second; None where either says none."""
        attr = AT.creation_of(mapdump.fragment(), cls)
        made = self._held_once(cls, attr) if attr else None
        return f"{W.when(made):%Y-%m-%d %H:%M:%S}" if made else None

    def _read_at(self, subject: str) -> str | None:
        """When the rows file read *subject*'s value: its cell's ``as_of``, or ``after``'s where the value went there
        (``held.split``), else when it read the statuslog; None where none says."""
        cls, _, attr = subject.partition(".")
        cell = self._row_of(cls).get(attr)
        if isinstance(cell, dict) and cell.get("as_of"):
            return cell["as_of"]
        after = self._read()[1]
        return (after.get("as_of") if attr in after else None) or (self.data.get("statuslog") or {}).get("as_of")

    def _row_value(self, subject: str, q: Missed | None = None):
        """The rows file's value of ``Class.attr`` for this instance, or None; under *q*, its ``after`` too where that
        answers (``_after``: at the window's end, or a column only the row's birth writes)."""
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
        ``after`` was read after the window and answers nothing, but for a column only the row's birth writes
        (:meth:`_born_only`), or only passes whose selections return no row of the object from the window on
        (:meth:`_written_only_before`, I-4).  ``source``, ``as_of`` and ``clock`` say where and when, and are no
        column."""
        cls, _, attr = subject.partition(".")
        after = self._read()[1]
        if cls != q.subject.partition(".")[0] or attr in ("source", "as_of", "clock") or attr not in after:
            return False, None
        if not q.at_end and not self._born_only(subject, q) and not self._written_only_before(subject, q):
            return False, None
        return True, after[attr]

    def _written_only_before(self, subject: str, q: Missed) -> list[tuple[str, str]]:
        """I-4 (Q6b, out/gm/qk-design.md, tmaeno's OK 2026-11-17 20:3x; built on the map's KMA): ``(arm, why)`` for
        each arm writing *subject*, a column late's ``after`` holds of the asked row, where every one of them takes the
        row by a key drawn from selections (:meth:`_selection_false`) that return no row of the object at *q*'s start
        nor at any statuslog row after it up to the rows file's reading: no pass of theirs took the row from then on,
        so the value read after the move is the one it held over the window.  goal's arms (TaskRefinerBase.
        extractCommon 276, 278) take the task by ``jediTaskID``, handed down from getTasksToRefine_JEDI's rows, whose
        status terms the task left before 02:06:38 (NR-T-53646395-assessmentTime, getAchievedTasks_JEDI's ``goal IS NOT
        NULL``).  Empty where an arm's selection may return the row at one of those moments, or cannot be read; a
        column read while it is being read (a selection's own terms) is empty too."""
        cls, _, attr = subject.partition(".")
        after = self._read()[1]
        if q.at_end or cls != q.subject.partition(".")[0] or attr not in after or mapdump.fragment() is None:
            return []
        reading: set[str] = self.__dict__.setdefault("_written_only_reading", set())
        cache: dict[tuple[str, datetime.datetime], list[tuple[str, str]]] = self.__dict__.setdefault("_written_only_before_of", {})
        if (subject, q.since) in cache:
            return cache[subject, q.since]
        if subject in reading:
            return []
        reading.add(subject)
        # a reading tried, not a step: what its selections said and asked is not kept (the sentence says why it read),
        # nor the marks of what is said once (a window, a timeline), so a step after it says them as it would have
        said_lines: list = self.__dict__.setdefault("lines", [])
        unanswered: list = self.__dict__.setdefault("unanswered", [])
        lines, asks = len(said_lines), len(unanswered)
        once = [(s, set(s)) for s in (getattr(self, "windows_said", None), self.__dict__.get("_timelines_told"))
                if s is not None]
        told_before = "_timelines_told" in self.__dict__
        said: list[tuple[str, str]] = []
        try:
            arms = [(j.owner, b.line) for j in asyncio.run(self._writers_of(subject)) for b in j.branches]
            states = list((self.data.get("statuslog") or {}).get("states") or [])
            moments = [q.since, *(_when(s.get("time_extended") or s["time"])
                                  for s in HD.rows_after(states, q.since.isoformat(" "), after.get("as_of")))]
            for owner, line in arms:
                # K-4b: where the selections are not read at a moment, the writes persisting the arm's object
                whys = [self._selection_false(owner, line, subject, t) or self._persisted_false(owner, line, subject, t)
                        for t in moments]
                if any(w is None for w in whys):
                    said = []
                    break
                said.append((SV._name(owner, line), str(whys[0])))
        except Stop:
            said = []
        finally:
            reading.discard(subject)
            del said_lines[lines:]
            del unanswered[asks:]
            for marks, before in once:
                marks.intersection_update(before)
            if not told_before:
                self.__dict__.pop("_timelines_told", None)
        cache[subject, q.since] = said
        return said

    def _born_only(self, subject: str, q: Missed) -> list[str]:
        """Q1 P10: the arms writing *subject*, a column late's ``after`` holds of the asked row, where every one of
        them makes the row (``born``): nothing writes it after the birth, so the value read after the move is the one
        it held all the stay.  JobSpec.creationTime is written by insertNewJob 3073, ppEventServiceJob 5111 and
        makeFakeCoJumbo 4540, each as it inserts the job.  Empty where any arm is no birth, or the map has none."""
        cls, _, attr = subject.partition(".")
        if q.at_end or cls != q.subject.partition(".")[0] or attr not in self._read()[1] or mapdump.fragment() is None:
            return []
        cache: dict[str, list[str]] = self.__dict__.setdefault("_born_only_of", {})
        if subject not in cache:
            arms = [(j.owner, b) for j in asyncio.run(self._writers_of(subject)) for b in j.branches]
            # P10: an arm writing it on an object a birth persists counts as a birth (one rule, P43's birth_arm)
            every = bool(arms) and all(birth_arm(owner, b, subject) for owner, b in arms)
            cache[subject] = [SV._name(owner, b.line) for owner, b in arms] if every else []
        return cache[subject]

    def _held(self, subject: str, q: Missed) -> tuple[bool, object]:
        """``(known, value)`` of *subject* for the row over the window: the asked subject holds the observed value;
        another, what the rows file holds (``None`` there is SQL's NULL), its ``after`` only at the window's end
        (``_after``), or, late, the value the map says it held all the stay (``_before``); a column it lacks is not
        known."""
        if subject == q.subject:
            return True, q.observed
        cls, _, attr = subject.partition(".")
        row = self._row_of(cls)
        if attr in row:
            return True, row[attr]["value"]
        known, value = self._after(subject, q)
        if known:
            return known, value
        found = self._before(subject, q)
        if found is not None and found[0].known:
            return True, found[1]
        # M3(a)-cap's turn: the value every arm writing the observed value began the stay with, the column's arms that
        # may write it while the value holds (M3(a)-1's (2)) named as having not
        assuming = getattr(self, "_assuming", None)
        if assuming is not None and found is not None and found[0].starts and not found[0].lacking:
            assuming[subject] = list(found[0].left)
            # M3(a)-2 step 1: a run of the row printing the column in the stay closes those arms up to the print
            printed = getattr(self, "_printed_read", None)
            if found[0].left and printed is not None and subject not in printed:
                if (got := self._printed_held(subject, q, found[0], found[1])) is not None:
                    printed[subject] = got
            return True, found[1]
        return False, None

    def _held_off(self, other: str, reading: str, q: Missed) -> tuple[bool, object]:
        """P24 (the investigator's u11 MJ-assigned-late 4): *other*, a column of the row a writer's selecting statement
        restricts, as the row held it all the stay, for M3(a)-1 reading *reading* (``stay_value._term_off``): what
        :meth:`_held` knows, else :meth:`_held_since_birth`.  A column read on the way in (one reading another) is not
        known, so no reading waits on itself."""
        busy: set[str] = self.__dict__.setdefault("_held_off_busy", set())
        if other in busy or other == reading:
            return False, None
        busy.add(reading)
        try:
            known, value = self._held(other, q)
            return (known, value) if known else self._held_since_birth(other, q)
        finally:
            busy.discard(reading)

    def _held_since_birth(self, subject: str, q: Missed) -> tuple[bool, object]:
        """``(known, value)``: *subject*, a column the rows file read of the asked row after the move, held that value
        from the row's birth where no arm but a birth (P43's ``birth_arm``) can write it -- every other arm writes a
        literal of another value (JobSpec.eventService: the job's NULL, which set_fine_grained's 6, setEventServiceMerge's
        2 and makeJumboJobs' 4 are not).  Not known where an arm writes a value known at run time only, or the value."""
        cls, _, attr = subject.partition(".")
        after = self._read()[1]
        if q.at_end or cls != q.subject.partition(".")[0] or attr not in after or mapdump.fragment() is None:
            return False, None
        value = after[attr]
        arms = [(j.owner, b) for j in asyncio.run(self._writers_of(subject)) for b in j.branches]
        if not arms:
            return False, None
        for owner, b in arms:
            if birth_arm(owner, b, subject):
                continue
            outcome = str(getattr(b, "outcome", "") or "")
            if outcome.startswith(SV.RUN_TIME) or outcome == "" or outcome.strip("'\"") == str(value):
                return False, None
        return True, value

    def _before(self, subject: str, q: Missed, nulls: bool = False) -> tuple[SV.Before, str] | None:
        """M3(a)-1 (``stay_value``): what the map says of *subject*, a column of the asked row the rows file holds only
        after the move (late's ``after``), all the stay at the observed value, with the time of the statuslog row
        that began the stay -- the value it held where the map says ``known``; None where it does not apply: stuck
        (``after`` is read at the window's end), another class, a column the row holds, a history whose rows' time
        part H does not state is the write's (``History.at_write``), no such statuslog row.  Each is read once and
        kept in the result (``held_before``; *nulls*, a column read as a term ``IS NULL`` reads it, ``states_before``:
        ``stay_value.before``'s *nulls*)."""
        cls, _, attr = subject.partition(".")
        if q.at_end or cls != q.subject.partition(".")[0] or attr in self._row_of(cls) or attr not in self._read()[1]:
            return None
        if not getattr(PT.history(q.subject), "at_write", False):
            return None
        cache: dict[tuple, tuple[SV.Before, str] | None] = self.__dict__.setdefault("_befores", {})
        key = (subject, q.subject, q.observed, q.since, q.moved, nulls)
        if key in cache:
            return cache[key]
        states = [s for s in self.data.get("statuslog", {}).get("states", [])
                  if s.get("status") == q.observed and _when(s.get("time_extended") or s["time"]) <= q.since]
        if not states:
            cache[key] = None
            return None
        start = states[-1]["time"]
        # A-3: the row's history holds a state before the stay's, so it was there before the stay began
        existed = any(_when(s.get("time_extended") or s["time"]) < _when(start)
                      for s in self.data.get("statuslog", {}).get("states", []))
        fragment = mapdump.fragment()
        sites = {(r.owner, r.anchor.line_start): r for r in getattr(fragment, "read_sites", None) or []}
        row_tables = {t.name.lower() for t in getattr(fragment, "tables", None) or [] if t.rows_of == cls}
        found = SV.before(q.writers, asyncio.run(self._writers_of(subject)), q.subject, q.observed, subject,
                          getattr(fragment, "key_froms", None) or [], lambda kf: not never_runs(kf),
                          lambda owner, line: sites.get((owner, line)), row_tables,
                          lambda owner, b: born(b, birth_marks(owner, subject)), nulls=nulls, existed=existed,
                          status_born=lambda owner, b: born(b, birth_marks(owner, q.subject)),
                          offset_of=lambda b: o.total_seconds() if (o := _written_offset(b)) is not None else None,
                          together=self._together_lines, held=lambda other: self._held_off(other, subject, q))
        if found is None:
            cache[key] = None
            return None
        cache[key] = (found, start)
        self.result.setdefault("states_before" if nulls else "held_before", {})[subject] = {
            "known": found.known, "value": start if found.known else None, "starts": list(found.starts),
            "lacking": list(found.lacking), "runtime": list(found.runtime), "left": list(found.left),
            "out": [list(x) for x in found.out], "births": list(found.births),
            "offsets": [list(x) for x in found.offsets], "runtime_starts": list(found.runtime_starts)}
        return cache[key]

    def _timeline(self, subject: str, q: Missed, nulls: bool = False) -> TL.Timeline | None:
        """Q1 part 1 (:mod:`timeline`; out/gm/qk-design.md, the Q1 design summary, tmaeno's OK 2026-11-15 23:06):
        *subject*, a time column M3(a)-1 (:meth:`_before`) leaves open, read over the stay from the records.  Each arm
        M3(a)-1 leaves as may writing it in the stay (``left``): its writes where its must prints name the row's key
        (:meth:`_column_writes`), from the stay's start less ``windows.SLACK`` to the move; one with none is shown not to
        have written where judgement 3's three premises hold, and one whose premises do not hold is open, read or not.
        The bounds as the stay began: :func:`timeline.first_bounds`.  None where M3(a)-1 does not apply or settles the
        column.  Kept in the result (``timelines``).

        *nulls* (Q1 batch 2 part 1, out/gm/qk-design.md, tmaeno's OK 2026-11-16): *subject* is a column a selecting
        statement tests ``IS NULL``, its states read the same way (:data:`timeline.NULL`, :func:`_null_states`), the
        stay begun with the states of the column's arms tied to the arm of the value that began it (``starts``' mates);
        an arm read as writing a value it knows only at run time is named (``premised``).  Kept as ``states``.  None
        while I-4 tries a selection (:meth:`_written_only_before`): a try asks no log, so a step after it reads and says
        the timeline as it would have."""
        if self.__dict__.get("_written_only_reading"):
            return None
        found = self._before(subject, q, nulls)
        if found is None or found[0].known:
            return None
        cache: dict[tuple, TL.Timeline | None] = self.__dict__.setdefault("_timelines", {})
        # a reading that asks nothing (``_no_asks``) keeps its own: a later one that asks keeps its asks
        ckey = (subject, q.subject, q.observed, q.since, q.moved, nulls, bool(self.__dict__.get("_no_asks")))
        if ckey in cache:
            return cache[ckey]
        cache[ckey] = None
        before, start = found
        # the stay's start to the microsecond where the question has it (the statuslog row's time_extended)
        begun = q.since if abs(q.since - _when(start)) < datetime.timedelta(seconds=1) else _when(start)
        since = begun - W.SLACK
        writers = self._subject_writers(subject)
        keys = self._key_columns(q.subject, self.args.key)
        arms_by_name = {SV._name(j.owner, b.line): b for j in writers for b in j.branches}
        owner_by_name = {SV._name(j.owner, b.line): j.owner for j in writers for b in j.branches}
        shown: dict[tuple[str, datetime.datetime, str, str], list[tuple[str, str, Any]]] = {}
        opens: list[TL.Opening] = []
        absent: list[str] = []
        premised: set[str] = set()
        not_run: list[tuple[datetime.datetime, str, str, Any]] = []  # A-1: writes their pass's guard read false
        confirmed: list[tuple[datetime.datetime, str, Any]] = []  # K-2 (a): reached writes read as run
        seen: list[tuple[str, str, datetime.datetime, str, str]] = []  # K-3 (b): the row's columns the prints show
        tied: list[tuple[datetime.datetime, str, Any]] = []  # K-3 (c): reached writes a tied column's print shows run
        # why each open arm is: no must print names the row it writes ("unprinted"), a log of it the records lack
        # ("unasked"), or judgement 3's premises not held otherwise ("unread") -- Q1 batch 2 part 2's (ii), (i), (iii)
        open_why: dict[str, str] = self.__dict__.setdefault("_open_why", {}).setdefault((subject, nulls), {})
        for j in writers:
            for b in j.branches:
                name = SV._name(j.owner, b.line)
                if name not in before.left:
                    continue
                missed: list = []
                got, whole = self._column_writes(j, b, keys, (since, q.moved), writers, missed, subject, not_run_out=not_run,
                                                 seen_out=seen)
                # P67: an arm whose every statement is a flush of the object it sets is read off the flush's printed
                # binds (JobGenerator's runImpl unlocks, in updateTask_JEDI's 1483)
                flushed, fwhole = self._flushed_writes(j, b, keys, (since, q.moved), subject)
                if fwhole is not None:
                    got, whole = [*got, *flushed], whole or fwhole
                if not whole:
                    printed = any(n.row == MUST_ROW_WRITTEN and n.column in keys for n in getattr(b, "must_names", None) or [])
                    open_why[name] = "unprinted" if not printed else "unasked" if missed else "unread"
                value: Any
                if nulls:
                    value = _null_states(b)
                    opening = TL.state_opening(name, value)
                    if _none_premised(b, j.owner):
                        premised.add(name)
                else:
                    value = _written_offset(b)
                    opening = TL.Opening(name, value, value)
                for at, machine, where, how, bound in got:
                    if how == TL.REACHED:
                        # K-2 (b): a write whose key's selection takes other states than the row held then is none of
                        # the row's (getTasksToBeFinished_JEDI's lock at 1690, the task in tobroken)
                        if (other := self._selection_takes_other_states(j.owner, b.line, subject, at)) is not None:
                            not_run.append((at, name, machine.split(".")[0], [T.Said("its-selection-takes-other-states", other)]))
                            continue
                        # K-2 (a): one whose statement's own status value the statuslog shows then ran
                        if (by := self._statement_ran(j.owner, b.line, subject, at, q)) is not None:
                            how = TL.RAN
                            confirmed.append((at, name, by))
                        # K-3 (c): one whose statement's write of another column of the row a later print shows ran
                        elif (by := self._tied_witness(j.owner, b.line, subject, at, machine, q, nulls)) is not None:
                            how = TL.RAN
                            tied.append((at, name, by))
                    # a value its statement's printed bind map holds stands for the arm's own (the bind-print part)
                    read = _printed_bounds(bound[0], nulls) if bound else None
                    shown.setdefault((j.owner, at, machine, where), []).append((name, how, value if read is None else read))
                if not whole:
                    opens.append(opening)
                elif not got:
                    absent.append(name)
        writes = []
        for (_, at, machine, where), arms in sorted(shown.items(), key=lambda x: (x[0][1], x[0][3])):
            vals = [_write_bounds(v, at) for _, _, v in arms]
            lo, hi = min(v[0] for v in vals), max(v[1] for v in vals)
            ran = all(how == TL.RAN for _, how, _ in arms) and (len(arms) == 1 or (len(set(vals)) == 1 and vals[0] != (TL.LOW, TL.HIGH)))
            writes.append(TL.Write(at, lo, hi, TL.RAN if ran else TL.REACHED, tuple(n for n, _, _ in arms), where,
                                   machine.split(".")[0]))
        if not nulls and (birth := self._birth_write(subject, q, before, arms_by_name, since, begun)) is not None:
            # F-5: the row made in the window before the stay began, its birth arms writing the clock then
            writes = sorted([*writes, birth], key=lambda w: (w.at, w.where))
        # K-3 (b): a value of the column a print shows, its write the records show ran and held it to the print
        witnessed = [] if nulls else self._printed_witnesses(subject, seen, writes)
        matched = {id(w): hits for *_, w, hits in witnessed}
        writes = [replace(w, how=TL.RAN) if id(w) in matched and len(matched[id(w)]) == len(w.arms) else w
                  for w in writes]

        def start_of(kept: list[str]) -> tuple[datetime.datetime, datetime.datetime] | None:
            # the states the column's arms tied to the kept arms of the value write (*nulls*); the clock's time with
            # the offset of an arm the map ties to one (A-2: ContFeeder's 1373, six hours before); None, the clock's
            if not kept:
                return None
            if not nulls:
                shifts = [{**dict(before.offsets), **dict(before.runtime_offsets)}.get(s, 0.0) for s in kept]
                if not any(shifts):
                    return None
                return (begun + datetime.timedelta(seconds=min(shifts)), begun + datetime.timedelta(seconds=max(shifts)))
            mates = [f"{s.partition(' with ')[0].rpartition(':')[0]}:{line}" for s in kept
                     for line in s.partition(" with ")[2].split(",")]
            vals = [_null_states(arms_by_name[m]) if m in arms_by_name else TL.EITHER for m in mates]
            premised.update(m for m in mates if m in arms_by_name and _none_premised(arms_by_name[m], owner_by_name[m]))
            return min(v[0] for v in vals), max(v[1] for v in vals)

        # A-3: an arm writing the status from a value known only at run time whose write the records show as the stay
        # began may have begun it, as one writing the value alone, read as writing the observed value (said)
        began_rt = self._runtime_began(q, before, begun) if before.runtime else []
        # E-2 (bundle E, out/gm/qk-design.md, tmaeno's OK 2026-11-17 13:42): such an arm whose pass's column arm the map
        # ties to it (M3(a)-1's ``runtime_starts``, A-2's tie) wrote the column as it began the stay, as a start does --
        # updateJobStatus:240 'starting' with 238 ``modificationTime=CURRENT_DATE`` in its UPDATE at 241
        rt_starts = [s for s in before.runtime_starts if s.partition(" with ")[0] in began_rt]
        tied_rt = {s.partition(" with ")[0] for s in rt_starts}
        all_starts = [*before.starts, *rt_starts]
        lacking_all = [*before.lacking, *(r for r in began_rt if r not in tied_rt)]
        starts, lacking = bool(all_starts), bool(lacking_all)
        a_starts, a_lacking = starts, lacking
        kept_s = a_kept_s = list(all_starts)
        begun_by: tuple[str, ...] = ()
        begun_assumed: tuple[str, ...] = ()
        # P3: which kind of arm began the stay matters only where both may have
        if starts and lacking:
            absent_b, unread_b = self._stay_begun_by(q, before, begun, began_rt)

            def left(out: set[str]) -> tuple[list[str], list[str]]:
                return ([s for s in all_starts if s.partition(" with ")[0] not in out],
                        [s for s in lacking_all if s not in out])

            if absent_b:
                ls, ll = left(absent_b)
                starts, lacking = bool(ls), bool(ll)
                kept_s = a_kept_s = ls
                begun_by = (*ls, *ll)
            if unread_b:
                ls, ll = left(absent_b | unread_b)
                a_starts, a_lacking = bool(ls), bool(ll)
                a_kept_s = ls
                begun_assumed = tuple(sorted(unread_b))
            elif absent_b:
                a_starts, a_lacking = starts, lacking
        row_value = None if nulls else self._row_value_write(subject, q, writes)
        if row_value is not None:
            row_value = _left_writers(row_value, before, begun)
        if row_value is not None and row_value[0] <= begun:
            # none wrote it from the row's value's write on: no arm is open in the stay (NT-pending-late's
            # assessmentTime, last written on 11-14 17:06:01 by getAchievedTasks_JEDI:3283 or its 3744)
            opens = []
        # K-3 (c): open arms whose statements write another column of the row a print shows held a value then
        quiet = self._tied_quiet(subject, q, opens, arms_by_name, owner_by_name, nulls) if opens else []
        tl = TL.Timeline(begun, q.moved, TL.first_bounds(begun, since, starts, lacking, writes, opens, start_of(kept_s)),
                         TL.first_bounds(begun, since, a_starts, a_lacking, writes, [], start_of(a_kept_s)), tuple(writes),
                         tuple(opens), tuple(absent), begun_by, begun_assumed, tuple(sorted(premised)),
                         tuple(sorted({(x[0], x[1], x[2]): x for x in not_run}.values(), key=lambda x: (x[0], x[1]))),
                         tuple(began_rt),
                         tuple(r for r in _runtime_apart(before) if r not in began_rt) if begun_by else (),
                         row_value, tuple(confirmed), tuple(x[:7] for x in witnessed), tuple(quiet), tuple(tied))
        cache[ckey] = tl

        def said(lo: datetime.datetime, hi: datetime.datetime) -> str | None:
            if lo != hi:
                return None
            return ("NULL" if lo == TL.NULL else "a value") if nulls else _iso(str(lo))

        self.result.setdefault("states" if nulls else "timelines", {})[subject] = {
            "start": _iso(str(begun)), "end": _iso(str(q.moved)),
            "writes": [{"at": _iso(str(w.at)), "how": w.how, "arms": list(w.arms), "where": w.where, "machine": w.machine,
                        "value": said(w.lo, w.hi)} for w in writes],
            "absent": absent, "open": [o.arm for o in opens], "begun_by": list(begun_by),
            "not_run": [{"at": _iso(str(at)), "arm": arm, "machine": machine} for at, arm, machine, _ in tl.not_run],
            "begun_runtime": list(tl.begun_runtime),
            "confirmed": [{"at": _iso(str(at)), "arm": arm, "by": by.get("by", "")} for at, arm, by in tl.confirmed],
            "runtime_unread": list(tl.runtime_unread),
            "row_value": {"at": _iso(str(tl.row_value[0])), "value": _iso(str(tl.row_value[1])), "arms": list(tl.row_value[2]),
                          "as_of": tl.row_value[3],
                          **({"apart": [a for a, _ in apart]} if (apart := next(iter(tl.row_value[4:]), ())) else {}),
                          **({"named": list(tl.row_value[5])} if len(tl.row_value) > 5 else {})}
            if tl.row_value else None,
            "begun_assumed": list(begun_assumed), **({"first": said(*tl.first), "premised": list(tl.premised)} if nulls else {})}
        return tl

    def _birth_write(self, subject: str, q: Missed, before: SV.Before, arms_by_name: dict, since: datetime.datetime,
                     begun: datetime.datetime) -> TL.Write | None:
        """Bundle F, F-5 (out/gm/qk-design.md, bundle F widened, tmaeno's OK 2026-11-17 15:2x; P1 (2)(b)): the row's
        birth as a write of the time column *subject*, where the row was made inside the timeline's window (from the
        stay's start less ``windows.SLACK`` to the start) and the arms M3(a)-1 puts out as births write the clock as it
        is made -- the clock itself, or the class's creation column (``attr_time.creation_of``), itself the clock then.
        Its time is that column's value in the rows file; ``RAN``.  MJ-assigned-late's prodDBUpdateTime: insertNewJob
        sets it to the job's creationTime, 09:05:55, fourteen seconds before the stay; lockJobsForActivator's term
        (NULL, or older than an hour) holds from 10:05:55.  None otherwise."""
        cls = subject.partition(".")[0]
        created = AT.creation_of(mapdump.fragment(), cls)
        births = [a for a, why in before.out if why.startswith("a birth") and a in arms_by_name]
        if not created or not births:
            return None
        try:
            born = _when(str(self._row_value(f"{cls}.{created}", q)))
        except (TypeError, ValueError):
            return None
        if not since <= born <= begun:
            return None
        clock = sorted(a for a in births if arms_by_name[a].outcome in AT.NOW
                       or arms_by_name[a].outcome == f"passthrough({cls}.{created})")
        if not clock:
            return None
        return TL.Write(born, born, born, TL.RAN, tuple(clock), f"its {created}", "")

    def _row_value_write(self, subject: str, q: Missed, writes: Sequence[TL.Write] = ()) -> tuple | None:
        """Bundle A, A-4 (P24, out/gm/qk-design.md, tmaeno's OK 2026-11-16): a time column's value as the rows file
        read it (late's ``after``, read at its ``as_of``), where every arm of the column writes the clock with one
        offset (:func:`_written_offset`): its last write was at the value less that offset, and none wrote it from then
        to when the rows file read it -- a later write would hold a later time.  ``(written at, the value, the arms,
        as_of)`` where that write fell before the move; None otherwise.  NT-scouting-late's assessmentTime 2026-11-14
        19:53:41, read on 11-15: written then by prepareTasksToBeFinished_JEDI:3744 or getAchievedTasks_JEDI:3283, each
        ``CURRENT_DATE``.

        K-1 (bundle K, out/gm/qk-design.md): where the column's arms write other values too, the last of the *writes*
        the records show (the timeline's) by an arm writing the clock with an offset whose clock then is the value, to
        its second (:data:`CLOCKS_APART`), ran or reached, with no write the records show as run after it: that write
        left the value.  A later write by an arm writing the clock would hold a later time; one writing another value
        is named as having not written the same value again (``(written at, the value, the arms, as_of, (), named)``).
        NJ-holding-late's endTime 15:35:35, read on 11-15, was written by updateJobStatus:233 (``CURRENT_DATE``) at
        15:35:35.751 as the job went holding; process_job_report:450 and the others writing the clock would have
        written a later time, and updateJobStatus:67 (the pilot's endTime) and the copies of modificationTime are
        named."""
        cls, _, attr = subject.partition(".")
        after = self._read()[1]
        if cls != q.subject.partition(".")[0] or attr not in after or not after.get("as_of"):
            return None
        raw = after[attr]
        raw = raw.get("value") if isinstance(raw, dict) else raw
        try:
            value = _when(str(raw))
            read_at = _when(str(after["as_of"]))
        except (TypeError, ValueError):
            return None
        arms = [(j.owner, b) for j in self._subject_writers(subject) for b in j.branches]
        offsets = {_written_offset(b) for _, b in arms}
        offset = next(iter(offsets)) if len(offsets) == 1 else None
        if not arms:
            return None
        if offset is not None:
            at = value - offset
            if at > q.moved or at > read_at:
                return None
            return at, value, tuple(sorted(SV._name(o, b.line) for o, b in arms)), str(after["as_of"])
        by_name = {SV._name(o, b.line): _written_offset(b) for o, b in arms}
        named = tuple(sorted(n for n, off in by_name.items() if off is None))
        found = _clock_write(value, writes, by_name)
        if found is None:
            return None
        w, hits = found
        if w.at > q.moved or w.at > read_at:
            return None
        return w.at, value, hits, str(after["as_of"]), (), named

    def _printed_witnesses(self, subject: str, seen: list, writes: Sequence[TL.Write]) -> list[tuple]:
        """K-3 (b) (bundle K, out/gm/qk-design.md, the delegated design): each value of the time column *subject* a print
        of a write the records show holds (*seen*, :meth:`_column_writes`' ``seen_out``), earlier than the print by more
        than a second and :data:`CLOCKS_APART` (not the print's own clock), matched to the last of *writes* before the
        print whose clock it is (:func:`_clock_write`, K-1's match): that write ran, and nothing wrote the column from
        it to the print -- a later write would hold another value (a later clock, NULL).  The pass read the value
        before the print, at a moment its records do not hold: said at the print.  ``(since, until, value, arms, print
        "func:line", the print's machine, the write's, the write, the arms matched)`` each, the latest print of each
        write's value.
        NT-defined-late: unlockTasks_JEDI:897 printed lockedTime 14:56:41 on aipanda081 at 15:57:05.666, written by
        getTaskWithID_JEDI:1818 at 14:56:41.571 on aipanda082."""
        offsets: dict[str, datetime.timedelta | None] | None = None
        latest: dict[tuple[int, datetime.datetime], tuple] = {}
        for column, raw, at, machine, where in sorted(set(seen), key=lambda s: (s[2], s[4], s[0], s[1])):
            if column != subject:
                continue
            try:
                value = _when(raw)
            except (TypeError, ValueError):
                continue
            if value > at - CLOCKS_APART - datetime.timedelta(seconds=1):
                continue
            if offsets is None:
                offsets = {SV._name(j.owner, b.line): _written_offset(b) for j in self._subject_writers(subject)
                           for b in j.branches}
            found = _clock_write(value, [w for w in writes if w.at < at], offsets)
            if found is None:
                continue
            w, hits = found
            latest[(id(w), value)] = (w.at, at, value, hits, where, machine.split(".")[0], w.machine, w, hits)
        return sorted(latest.values(), key=lambda x: (x[0], x[1]))

    def _tied_witness(self, owner: str, line: int, subject: str, at: datetime.datetime, machine: str,
                      q: Missed, nulls: bool = False) -> dict | None:
        """K-3 (c) (bundle K, out/gm/qk-design.md, the delegated design): a write of *subject* by *owner*'s arm at *line*,
        a print reaching it at *at* on *machine*, ran where a statement of it (the map's ``KeyFrom.writes``) writes
        another column of the row whose write at that print a later print of that column shows ran (its timeline's
        *witnessed*, K-3 (b)): the one statement wrote both.  getTaskWithID_JEDI's 1849 writes lockedBy by 1847 and
        lockedTime by 1818 (NT-defined-late, 14:56:41.571 on aipanda082).  Reading *subject*'s states (*nulls*), its
        own time timeline counts too.  The words' fields, or None."""
        cls = subject.partition(".")[0]
        host = machine.split(".")[0]
        for kf in key_froms():
            if _bare(kf.owner) != _bare(owner) or line not in (kf.writes.get(subject) or []):
                continue
            for column, lines in sorted(kf.writes.items()):
                if (column == subject and not nulls) or column.partition(".")[0] != cls or not lines:
                    continue
                tl = self._timeline(column, q)
                if tl is None:
                    continue
                names = {SV._name(owner, x) for x in lines}
                for since, until, value, arms, where, m, wm in tl.witnessed:
                    if since == at and wm == host and names & set(arms):
                        return {"column": column, "arms": sorted(names & set(arms)), "value": f"{value:%Y-%m-%d %H:%M:%S}",
                                "where": where, "machine": m, "printed": f"{until:%H:%M:%S.%f}"[:12]}
        return None

    def _tied_quiet(self, subject: str, q: Missed, opens: list, arms_by_name: dict, owner_by_name: dict,
                    nulls: bool = False) -> list[tuple]:
        """K-3 (c) (bundle K, out/gm/qk-design.md, the delegated design): the stretches where open arms of *subject* did
        not write it.  One whose every statement (the map's ``KeyFrom.writes``) writes another column of the row too,
        by arms writing NULL or the clock with an offset, did not run while a print shows that column held a value
        (its timeline's *witnessed*, K-3 (b)): it would have left another value -- save near where its clock is the
        value.  ``(since, until, arms, that column)`` each.  NT-defined-late's lockedBy: unlockSingleTask_JEDI:2833 and
        the other unlocks null lockedTime too, which held 14:56:41 from 14:56:41.571 to 15:57:05.666.  Reading
        *subject*'s states (*nulls*), its own time timeline counts too."""
        cls = subject.partition(".")[0]
        slack = CLOCKS_APART + datetime.timedelta(seconds=1)
        found: dict[tuple[datetime.datetime, datetime.datetime, str], list[str]] = {}
        for o in opens:
            b, owner = arms_by_name.get(o.arm), owner_by_name.get(o.arm)
            if b is None or owner is None:
                continue
            statements = [kf for kf in key_froms()
                          if _bare(kf.owner) == _bare(owner) and b.line in (kf.writes.get(subject) or [])]
            if not statements:
                continue
            columns = set.intersection(*({c for c, ls in kf.writes.items()
                                          if ls and (c != subject or nulls) and c.partition(".")[0] == cls}
                                         for kf in statements))
            for column in sorted(columns):
                tl = self._timeline(column, q)
                if tl is None or not tl.witnessed:
                    continue
                arms = {SV._name(j.owner, x.line): x for j in self._subject_writers(column) for x in j.branches}
                theirs = [arms.get(SV._name(owner, x)) for kf in statements for x in kf.writes.get(column) or []]
                if not theirs or any(x is None for x in theirs):
                    continue
                for since, until, value, *_ in tl.witnessed:
                    start: datetime.datetime | None = since
                    for x in theirs:
                        if _null_states(x) == TL.NULL_ONLY:
                            continue
                        off = _written_offset(x)
                        if off is None or start is None:
                            start = None
                            break
                        # a run at t writes t + off: the value only for t near value - off
                        lo, hi = value - off - slack, value - off + slack
                        if hi < since or lo >= until:
                            continue
                        start = max(start, hi) if lo <= since else None
                    if start is not None and start < until:
                        found.setdefault((start, until, column), []).append(o.arm)
        return [(a, z, tuple(sorted(set(arms))), column) for (a, z, column), arms in sorted(found.items())]

    def _selection_takes_other_states(self, owner: str, line: int, subject: str, at: datetime.datetime) -> dict | None:
        """K-2 (b) (bundle K, out/gm/qk-design.md): a write of *subject* by *owner*'s arm at *line*, a print reaching it at
        *at*, is none of the row's where the write's key comes only from selections' rows (the map's ``KeyFrom``
        ``rows`` sources) and each selection's selects of the class's history subject lack the state the statuslog
        holds then.  Only that state is read at that moment; the selections' other terms are not (they would be read
        in the rows file's later values, KNOWN_ISSUES.md K17).  getTasksToBeFinished_JEDI's lock at 1690 at 03:58:05
        takes rows of its 1644 (prepared, scouted, breaking, toabort, passed) while the task was tobroken
        (NT-breaking-late).  The words' fields, or None."""
        fragment = mapdump.fragment()
        history = PT.history_of_class(subject.partition(".")[0])
        states = [s for s in ((getattr(self, "data", None) or {}).get("statuslog") or {}).get("states") or []
                  if _when(s.get("time_extended") or s["time"]) <= at]
        if fragment is None or history is None or not states:
            return None
        writes = [kf for kf in key_froms() if _bare(kf.owner) == _bare(owner) and kf.term
                  and line in (kf.writes.get(subject) or [])]
        sources = [s for kf in writes for s in getattr(kf, "sources", None) or []]
        if not sources or any(s.kind != "rows" or getattr(s, "via", None) for s in sources):
            return None
        sites = {rs.name: rs for rs in getattr(fragment, "read_sites", None) or []}
        state = states[-1].get("status")
        takes: list[str] = []
        names: list[str] = []
        for name in dict.fromkeys(s.name for s in sources):
            site = sites.get(name)
            values = [v.partition("=")[2] for v in getattr(site, "selects", None) or [] if v.partition("=")[0] == history.subject]
            if site is None or not values or state in values:
                return None
            takes += [v for v in values if v not in takes]
            names.append(name.rpartition("::")[2])
        return {"site": ", ".join(names), "takes": takes, "state": repr(state)}

    def _statement_ran(self, owner: str, line: int, subject: str, at: datetime.datetime, q: Missed) -> dict | None:
        """K-2 (a) (bundle K, out/gm/qk-design.md): a write of *subject* by *owner*'s arm at *line*, a print reaching it at
        *at*, ran where a status arm of the same function the map ties to it (``stay_value.tied``: ``same_pass``, or
        the one arm of the column at the status arm's statement, one call running both) writes the value the statuslog
        shows within a second and ``CLOCKS_APART`` of *at*: that row is the statement's run, and the column went with
        it.  prepareTasksToBeFinished_JEDI's UPDATE at 4075 writes the status (3971 'tobroken') and lockedBy only by 3839
        (NULL); the statuslog's tobroken row is at 03:57:57.7, the print at .600 (NT-breaking-late).  updateJobStatus's
        241 writes endTime by 233 or 67, neither tied: a 'running' row says nothing of endTime.  Another writer of the
        same status in that second is not told apart.  The words' fields, or None."""
        history = PT.history_of_class(subject.partition(".")[0])
        states = ((getattr(self, "data", None) or {}).get("statuslog") or {}).get("states") or []
        if history is None or not states or subject == history.subject:
            return None
        status = history.subject
        def mine(subj: str) -> list[tuple[str, Any]]:
            return [(j.owner, b) for j in self._subject_writers(subj) if _bare(j.owner) == _bare(owner) for b in j.branches]

        columns = [c for _, c in mine(subject)]
        kfs = key_froms()
        near = CLOCKS_APART + datetime.timedelta(seconds=1)
        for o, sb in mine(status):
            value = (sb.outcome or "").strip("'\"") if isinstance(sb.outcome, str) else None
            then = next((t for s in states if value and s.get("status") == value
                         and abs((t := _when(s.get("time_extended") or s["time"])) - at) <= near), None)
            if then is None or not any(c.line == line for c in SV.tied(sb, o, status, subject, columns, kfs, self._together_lines)):
                continue
            stmt = next((kf.statement for kf in kfs if kf.owner == o and set(kf.writes.get(status) or []) & SV._lines(sb)
                         and line in (kf.writes.get(subject) or [])), None)
            return {"by": SV._name(o, sb.line), "value": repr(value),
                    "statement": f"in the same statement, {_bare(o).rpartition('::')[2]}:{stmt}" if stmt else "in the same pass",
                    "then": f"{then:%H:%M:%S.%f}"[:12]}
        return None

    def _runtime_began(self, q: Missed, before: SV.Before, begun: datetime.datetime) -> list[str]:
        """Bundle A, A-3 (out/gm/qk-design.md, tmaeno's OK 2026-11-16): the arms writing the status from a value known
        only at run time (M3(a)-1's ``runtime``) whose write of the row the records show as the stay began
        (:meth:`_column_writes`, from ``windows.SLACK`` before the start -- not before the history's state before it --
        to a second after it): each may have begun it, read as writing the observed value.  NT-registered-late's
        reactivatePendingTasks_JEDI ``SET status=oldStatus`` at 03:06:37.929, where M3(a)-1 read only the birth of the
        row as writing 'registered'."""
        names = set(_runtime_apart(before))
        keys = self._key_columns(q.subject, self.args.key)
        earlier = [_when(s.get("time_extended") or s["time"]) for s in self.data.get("statuslog", {}).get("states", [])]
        prior = max((t for t in earlier if t < begun - datetime.timedelta(seconds=1)), default=None)
        lo = begun - W.SLACK if prior is None else max(begun - W.SLACK, prior + datetime.timedelta(seconds=1))
        window = (lo, begun + datetime.timedelta(seconds=1))
        out: list[str] = []
        for j in q.writers:
            for b in j.branches:
                name = SV._name(j.owner, b.line)
                if name in names and name not in out and self._column_writes(j, b, keys, window, q.writers)[0]:
                    out.append(name)
        return sorted(out)

    def _stay_begun_by(self, q: Missed, before: SV.Before, begun: datetime.datetime,
                       extra: list[str] | None = None) -> tuple[set[str], set[str]]:
        """Q1 P3 (out/gm/qk-design.md, the Q1 design summary, tmaeno's OK 2026-11-15 23:06): which of the arms M3(a)-1
        says may have begun the stay -- writing the value and the column (``starts``, "func:line with line") or the
        value alone (``lacking``) -- did, read as the timeline reads a column's writes (:meth:`_column_writes`): each
        arm's writes of the asked subject from ``windows.SLACK`` before the stay's start to a second after it.
        ``(absent, unread)``: those shown not to write it then (judgement 3's premises held, no write read), which did
        not begin it; and, where another's write is read there, those of which none is read and none shown, their
        logs there all in the records (one not in them stays, its ask kept: NT-assigning-late's doActionForReassign:309
        in panda-AtlasProdWatchDog.log) -- as H2
        reads a row's writer, the one whose run printed the row then, they are named, where the open arms are, as
        having not begun it.  Empty where every one would be left out.  NT-tobroken-late's
        prepareTasksToBeFinished_JEDI:3971 (modificationTime=CURRENT_DATE in the same pass) printed the row at
        20:29:15-16; feed_contents_to_tasks:742 and runImpl:577 print nothing naming it.  *extra*: the arms writing a
        value known only at run time read as having written as it began (:meth:`_runtime_began`), read with them."""
        names = {s.partition(" with ")[0] for s in before.starts} | set(before.lacking) | set(extra or [])
        keys = self._key_columns(q.subject, self.args.key)
        window = (begun - W.SLACK, begun + datetime.timedelta(seconds=1))
        absent: set[str] = set()
        read: set[str] = set()
        pending: set[str] = set()
        seen: set[str] = set()
        for j in q.writers:
            for b in j.branches:
                name = SV._name(j.owner, b.line)
                if name not in names or name in seen:
                    continue
                seen.add(name)
                missed: list = []
                got, whole = self._column_writes(j, b, keys, window, q.writers, missed_out=missed)
                if got:
                    read.add(name)
                elif missed:
                    # a log not in the records may hold its print: it stays, and the ask with it
                    pending.add(name)
                elif whole:
                    absent.add(name)
        if absent >= names:
            return set(), set()
        unread = seen - absent - read - pending if read else set()
        return absent, unread

    def _column_writes(self, j, b, keys: set[str], window: tuple[datetime.datetime, datetime.datetime],
                       writers: list, missed_out: list | None = None, subject: str = "",
                       not_run_out: list | None = None,
                       seen_out: list | None = None) -> tuple[list[tuple[datetime.datetime, str, str, str, tuple]], bool]:
        """The writes of the arm *b* of *j* the records show over *window*, ``(at, machine, print "func:line", how,
        printed)`` each: a record of one of its must prints whose hole the map names as the written row's key
        (``MustName.row`` ``written``, its column among *keys*) holding the key, under any mark (:func:`_hole_holds`) --
        the statement's bind map printed ran it (``timeline.RAN``), its message or mark reached it (``REACHED``).  Where
        the map's ``WritePrint`` says the statement's text and bind map print in the call, the value of *subject* it ran
        with is read off that print (:func:`_printed_write`): the write is at that print, *printed* ``(value,)``; one
        whose object's changed fields leave the column out wrote none of it -- not a write, though its run was read.
        *printed* is ``()`` where none is read.  A bind map print the map
        cannot show holds the value the write counts (``row`` ''), one holding the key among its values, reached it: the
        statement it prints named the row, and the write that takes its rows may have run on it (P17,
        getTasksToBeFinished_JEDI's select print 1643 naming the row among its targets).  A record another print of the
        function reads at least as well is not taken.  And whether these are all: judgement 3's three premises (window,
        logs, prints: :meth:`_premises`) held over the window, so no run reached the arm unseen -- never where only
        such a print names the row.  The logs are asked for no method's runs whole (``owners`` empty): a mark that does
        not hold the key is read from the key's records, which hold every must print the reading needs.

        Bundle A, A-1 (out/gm/qk-design.md, tmaeno's OK 2026-11-16): a write whose statement's guard the pass holding
        the record reads false on the values it printed did not run there (:meth:`_not_run_in_pass`) -- not a write,
        though its run was read, each put in *not_run_out* ``(at, "func:line", machine, the guard)``.

        Bundle A, A-3's reading through the map's part 1 (the count prints' key holes, tmaeno's OK to ask for it
        2026-11-16 20:49): a record of a count print after the write that holds the key (:func:`_counted_names`) is a
        write of the row the statement ran (``timeline.RAN``) -- NT-registered-late's reactivatePendingTasks_JEDI:4728
        on aipanda088 at 03:06:37,930.  Such prints say nothing of a write absent, so they do not make the window whole;
        they are read only where no must print names the row (there they confirm its write, :func:`_landed`), and a
        call's records of them are one write, at the first (unlockTasks_JEDI's 919, once for each dataset of the task).

        K-3 (b): *seen_out* gets, from each must or count print read as a write, the row's other columns of
        *subject*'s class the print shows (:func:`_printed_columns`), ``(column, value, at, machine, print
        "func:line")``."""
        key = self.args.key
        shown = [n for n in b.must_names or [] if n.row == MUST_ROW_WRITTEN and n.column in keys]
        named = [n for n in b.must_names or [] if n.row == "" and n.part == MUST_PART_BIND and n.column in keys]
        # where a must print names the row, its count prints confirm that write (:func:`_landed`), not another
        counted = [] if shown else _counted_names(b, keys)
        names = shown + named + counted
        rel, _, func = _bare(j.owner).partition("::")
        to_hand = self.logs.names(self.files_of(j.owner, list(j.log_files or []), [b.line]))
        if not names or not to_hand:
            return [], False
        lo, hi = W._down(window[0]), W._down(window[1])
        asking = W.Asking((lo, hi), None, T.Said("for-the-column-s-writes", {
            "start": lo, "end": hi, "where": f"{func.rpartition('.')[2]}:{b.line}"}))
        runs, missed, holes = self._ask(key, to_hand, asking, {}, quiet=True)
        self._keep_unanswered(missed)
        if missed_out is not None:
            missed_out += missed
        found: list[tuple[datetime.datetime, str, str, str, tuple]] = []
        unwritten = False  # a run read whose statement's printed map shows it did not write the column
        # a count print's records of the row in one call of the function are one write, at the first: unlockTasks_JEDI
        # prints 919 for each dataset of the task it unlocked once
        by_call: dict[tuple[int, int], tuple[datetime.datetime, str, str, str, tuple]] = {}
        for line in sorted({n.line for n in names}):
            try:
                fn = R.function(rel, func, within=[line])
            except (LookupError, ValueError, SyntaxError):
                continue
            stmt = R.statement_at(fn.node, line)
            row = fn.row_at(stmt) if stmt is not None else None
            if row is None:
                continue
            for name in to_hand:
                for run in runs.get(name, []):
                    sn = R.seen(fn, run, self.mode)
                    for i, literal, _ in sn.hits.get(row.line, ()):
                        record = run.records[i]
                        at = W.when(record.stamp)
                        if not window[0] <= at <= window[1] or any(
                                other != row.line and any(k == i and lit >= literal for k, lit, _ in hs)
                                for other, hs in sn.hits.items()):
                            continue
                        if any(n.line == line and _hole_holds(row, record, n, key) for n in counted):
                            # the statement's count print names the row: it changed it there (bundle A's remaining
                            # reader, the map's part 1)
                            call = (id(run), sum(1 for e in R.entries(fn, run, self.mode) if e <= i))
                            if call not in by_call or at < by_call[call][0]:
                                by_call[call] = (at, record.machine or "", f"{func.rpartition('.')[2]}:{line}", TL.RAN, ())
                            if seen_out is not None and subject:
                                # K-3 (b): the row's other columns the count print shows, as the pass read them
                                seen_out += [(c, v, at, record.machine or "", f"{func.rpartition('.')[2]}:{line}")
                                             for c, v in _printed_columns(j.owner, line, row, record,
                                                                          subject.partition(".")[0], keys)]
                            continue
                        held = [n for n in shown if n.line == line and _hole_holds(row, record, n, key)]
                        if held:
                            # R: the statement's count print in the same run says whether the write landed
                            landed, counted_at = _landed(fn, sn, i, row.line, b, j.owner, j.subject)
                            if landed is False:
                                # E-4: the run changed no row -- not a write, and read as not having written it, as
                                # A-1 reads a write whose guard its pass read false
                                unwritten = True
                                if not_run_out is not None:
                                    not_run_out.append((at, SV._name(j.owner, b.line), (record.machine or "").split(".")[0],
                                                        [T.Said("a-printed-count-of-none", {"line": counted_at})]))
                                continue
                            if (guard := self._not_run_in_pass(fn, j, b, run, i)) is not None:
                                unwritten = True
                                if not_run_out is not None:
                                    not_run_out.append((at, SV._name(j.owner, b.line), (record.machine or "").split(".")[0], guard))
                                continue
                            # K-3 (e): the judge's rules on the pass: a must row of the arm not in it, it did not write
                            # there; a row printing only on a path through it, it ran
                            judged = self._judged_in_pass(fn, j, b, run, i)
                            if judged is not None and judged[0] == "excluded":
                                unwritten = True
                                if not_run_out is not None:
                                    not_run_out.append((at, SV._name(j.owner, b.line), (record.machine or "").split(".")[0],
                                                        [T.Said("its-must-rows-not-in-the-pass", {"why": "; ".join(judged[1])})]))
                                continue
                            if seen_out is not None and subject:
                                # K-3 (b): the row's other columns the print shows, as the pass read them
                                seen_out += [(c, v, at, record.machine or "", f"{func.rpartition('.')[2]}:{line}")
                                             for c, v in _printed_columns(j.owner, line, row, record,
                                                                          subject.partition(".")[0], keys)]
                            # the judge's positive shows the write at the rows it was shown by, not at the earlier ones,
                            # and only of a statement picking its row by the key alone: one with a term on another column
                            # (lockTask_JEDI's ``lockedBy=:lockedBy``) ran without the run showing it took the row
                            how = (TL.RAN if landed or any(n.part == MUST_PART_BIND for n in held)
                                   or (judged is not None and judged[0] == "positive" and line in judged[2]
                                       and not _picks_beyond_key(j.owner, b.line, j.subject, keys)) else TL.REACHED)
                            # the bind-print part: the value its statement ran with, off the print of its bind map
                            printed = _printed_write(fn, sn, run, i, row.line, j.owner, b, subject) if subject else None
                            if printed is not None and printed[1] is UNWRITTEN:
                                unwritten = True
                                continue
                            if printed is not None and _hashable(printed[1]):
                                found.append((printed[0], record.machine or "", f"{func.rpartition('.')[2]}:{printed[2]}",
                                              how, (printed[1],)))
                                continue
                            found.append((at, record.machine or "", f"{func.rpartition('.')[2]}:{line}", how, ()))
                        elif any(n.line == line for n in named) and _binds_name(record, key):
                            if (guard := self._not_run_in_pass(fn, j, b, run, i)) is not None:
                                unwritten = True
                                if not_run_out is not None:
                                    not_run_out.append((at, SV._name(j.owner, b.line), (record.machine or "").split(".")[0], guard))
                                continue
                            found.append((at, record.machine or "", f"{func.rpartition('.')[2]}:{line}", TL.REACHED, ()))
        found += by_call.values()
        premises = {"window": self._searched_whole(to_hand, asking, asking, missed, holes, None),
                    "logs": every_log_asked(j.owner, [b.line], to_hand),
                    "prints": prints_naming(writers, j.owner, [b.line], keys)}
        # a run of its own in the window whose must prints are not read in it: a reading missed, not a run absent
        own = any(of_mover(r, j.owner) and spans(r, (lo, hi)) for rs in runs.values() for r in rs)
        return (sorted(set(found), key=lambda f: f[:4]),
                bool(shown) and all(p["held"] for p in premises.values()) and (bool(found) or unwritten or not own))

    def _flushed_writes(self, j, b, keys: set[str], window: tuple[datetime.datetime, datetime.datetime],
                        subject: str) -> tuple[list[tuple[datetime.datetime, str, str, str, tuple]], bool | None]:
        """P67 (the investigator's u10; out/gm/qk-design.md, the W13/P67 section, the delegated design): the writes of the arm
        *b* of *j* a flush of the object it sets shows, where every statement writing *subject* at *b* is a flush (the
        map's ``KeyFrom.flushed_in`` and ``flushed_lines``).  The flush's WritePrint there (``object_map``: the object's
        changed fields) prints the statement's text and bind map before it runs, in a run of the row's key; a record of
        it over *window* whose bind map holds *subject*'s bind, with the value *b* writes (None for an arm writing None,
        another for one writing a value), is a write of it then (``ran``), unless the statement's count print after it
        says no row changed (:func:`_landed`, read on an arm of the flush's function counted at that statement).
        ``(writes, whole)`` as :meth:`_column_writes` gives them, *whole* where those logs were searched whole over the
        window with every log asked; ``([], None)`` where *b* is no flush's or the map gives no such print.  Which
        caller's pass ran the flush is not read: every arm writing that value by a flush of the same statement reads the
        record (JobGenerator's runImpl:941 and doPreCheck:299 alike for ``:lockedBy None``)."""
        statements = [kf for kf in key_froms()
                      if _bare(kf.owner) == _bare(j.owner) and b.line in (kf.writes.get(subject) or [])]
        if not statements or not all(getattr(kf, "flushed_in", "") for kf in statements):
            return [], None
        states = _null_states(b)
        key = self.args.key
        lo, hi = W._down(window[0]), W._down(window[1])
        found: list[tuple[datetime.datetime, str, str, str, tuple]] = []
        whole = True
        for flush, lines in sorted({(kf.flushed_in, tuple(getattr(kf, "flushed_lines", None) or ())) for kf in statements}):
            wps = [wp for wp in write_prints(flush) if int(wp.statement) in lines and bool(getattr(wp, "object_map", None))]
            if not wps:
                return [], None
            rel, _, func = _bare(flush).partition("::")
            name = func.rpartition(".")[2]
            for wp in wps:
                binds = {w.bind for w in wp.binds if w.column == subject}
                if len(binds) != 1:
                    return [], None
                bind = binds.pop()
                counter = next((x for jj in self._subject_writers_any(flush) for x in jj.branches
                                if any(getattr(c, "run", 0) == int(wp.statement)
                                       for c in [*(getattr(x, "count_prints", None) or []),
                                                 *(getattr(x, "printed_counts", None) or [])])), None)
                for p in wp.printed_by:
                    to_hand = self.logs.names(self.files_of(flush, [], [p.line]))
                    if not to_hand:
                        whole = False
                        continue
                    asking = W.Asking((lo, hi), None, T.Said("for-the-column-s-writes", {
                        "start": lo, "end": hi, "where": f"{name}:{p.line}"}))
                    runs, missed, holes = self._ask(key, to_hand, asking, {}, quiet=True)
                    self._keep_unanswered(missed)
                    whole = whole and bool(self._searched_whole(to_hand, asking, asking, missed, holes, None)["held"]) \
                        and bool(every_log_asked(flush, [p.line], to_hand)["held"])
                    try:
                        fn = R.function(rel, func, within=[p.line])
                    except (LookupError, ValueError, SyntaxError):
                        return [], None
                    stmt = R.statement_at(fn.node, p.line)
                    row = fn.row_at(stmt) if stmt is not None else None
                    if row is None:
                        return [], None
                    for log in to_hand:
                        for run in runs.get(log, []):
                            sn = R.seen(fn, run, self.mode)
                            for i, _, _ in sn.hits.get(row.line, ()):
                                record = run.records[i]
                                at = W.when(record.stamp)
                                got = sqltext.cut(record.text, p.lead, p.gap) if window[0] <= at <= window[1] else None
                                if got is None or bind not in got[1]:
                                    continue
                                value = got[1][bind]
                                if (states == TL.NULL_ONLY and value is not None) or (states == TL.SET_ONLY and value is None):
                                    continue
                                if counter is not None and _landed(fn, sn, i, row.line, counter, flush,
                                                                   self._subject_of_branch(flush, counter))[0] is False:
                                    continue
                                found.append((at, record.machine or "", f"{name}:{p.line}", TL.RAN, (value,)))
        return found, whole

    def _subject_writers_any(self, owner: str) -> list:
        """The map's junctions of *owner* for any subject (P67: an arm of a flush's function counted at its statement)."""
        fragment = mapdump.fragment()
        return [j for j in getattr(fragment, "junctions", None) or [] if _bare(j.owner) == _bare(owner)]

    def _subject_of_branch(self, owner: str, branch) -> str:
        """The subject of the junction of *owner* holding *branch*."""
        return next((j.subject for j in self._subject_writers_any(owner) if any(x is branch for x in j.branches)), "")

    def _judged_in_pass(self, fn: R.Function, j, b, run: Any, i: int) -> tuple[str, list[str], list[int]] | None:
        """K-3 (e) (bundle K, out/gm/qk-design.md, the delegated design): the judge's verdict on the arm *b* of *j* alone
        in the pass of *fn* that holds record *i* (:func:`which_ran.judge`, one candidate on that pass): ``("positive",
        why, the rows it was shown by)`` where a row printing only on a path through the arm is in the pass,
        ``("excluded", why, [])`` where a must row of it is not (rule 4); None otherwise -- a guard read false is A-1's,
        read before (:meth:`_not_run_in_pass`) -- or where no pass holds *i*.  propagateResultToJEDI's 4404 ``SET
        walltimeUnit=NULL`` under ``if numDone in [5, 100]:``: its 4406 'trigger recalculation' printed only in the
        17:20:30 pass of NT-running-late's task, which ran it; the passes printing its 4390 without it did not."""
        cache: dict[tuple[int, int], list] = self.__dict__.setdefault("_passes_cut", {})
        if (id(fn), id(run)) not in cache:
            cache[(id(fn), id(run))] = [fn, run, cut_passes(fn, run, self.mode)[0]]
        record = run.records[i]
        part = next((p for p in cache[(id(fn), id(run))][2] if any(r is record for r in p.records)), None)
        if part is None:
            return None
        rel, _, func = _bare(j.owner).partition("::")
        name = run.log.name
        try:
            got = judge([Candidate(rel, func, [b.line], log_files=[name], run=part)], {name: [part]}, mode=self.mode,
                        decide="positive")
        except (LookupError, ValueError, SyntaxError):
            return None
        v = next((v for v in got.verdicts if v.line == b.line), None)
        if v is None:
            return None
        if v.kind == "positive":
            return "positive", list(v.reasons), list(v.proven)
        if v.kind == "excluded" and any(r.startswith("rule 4") for r in v.reasons):
            return "excluded", [r for r in v.reasons if r.startswith("rule 4")], []
        return None

    def _not_run_in_pass(self, fn: R.Function, j, b, run: Any, i: int) -> list[T.Said] | None:
        """Bundle A, A-1 (out/gm/qk-design.md, tmaeno's OK 2026-11-16): the guard of each statement the arm *b* of *j*
        writes at -- the map's ``KeyFrom.statement`` for its line (getTaskWithID_JEDI's 1818 builds the text
        ``SET lockedBy=:lockedBy,lockedTime=CURRENT_DATE``, which 1849 runs under ``if lockTask:``), else the arm's own
        line -- read false by :meth:`_held_on` on the values printed in the pass of *fn* that holds record *i*
        (:func:`rows.passes`, the run cut at *fn*'s entry row: a querier run can hold several calls, NT-toabort-late's
        seven of getTaskWithID_JEDI, lockTask True in one and False in six).  Each statement's guard with its values,
        where every one is read false; None otherwise (a guard not read, a statement with none, no pass holding *i*)."""
        lines = sorted({kf.statement for kf in key_froms() if kf.owner == j.owner
                        and set(kf.writes.get(j.subject, [])) & {b.line, *(getattr(b, "other_lines", None) or [])}})
        cache: dict[tuple[int, int], list] = self.__dict__.setdefault("_passes_cut", {})
        if (id(fn), id(run)) not in cache:
            cache[(id(fn), id(run))] = [fn, run, cut_passes(fn, run, self.mode)[0]]
        record = run.records[i]
        part = next((p for p in cache[(id(fn), id(run))][2] if any(r is record for r in p.records)), None)
        if part is None:
            return None
        said: list[T.Said] = []
        for line in lines or [b.line]:
            stmt = R.statement_at(fn.node, line)
            guards = pathcond.arm_condition(stmt) if stmt is not None else []
            if stmt is None or not guards:
                return None
            _held, false = self._held_on(fn, stmt, guards, part, quiet=True)
            if false is None:
                return None
            said.append(T.Said("guard-read-false-on", {"guard": false[0], "on": false[1]}))
        return said

    def _timeline_spans(self, t, q: Missed) -> list | None:
        """Where a term comparing a time column with the clock surely holds for the row, on the column's timeline
        (:meth:`_timeline`): None where it has none, or reads no write and shows no arm absent -- M3(a)-1 and -2 read
        it as before.  Inside a turn (``_assuming``, M3(a)-cap) its open arms are named as having not written it."""
        tl = self._timeline(t.subject, q)
        if tl is None or not tl.read():
            return None
        assuming = getattr(self, "_assuming", None)
        if assuming is not None and (tl.opens or tl.begun_assumed):
            assuming[t.subject] = [o.arm for o in tl.opens]
        pieces = tl.pieces(assume=assuming is not None)
        out = [(q.since, q.moved)]
        for v in t.values:
            if not v.base or v.base not in UTC_CLOCKS or v.offset_seconds is None:
                return []
            out = _intersect(out, TL.surely_true(t.op, datetime.timedelta(seconds=v.offset_seconds), pieces))
        return _union(out)

    def _states_at_hand(self, subject: str, q: Missed) -> TL.Timeline | None:
        """K-3 (a): *subject*'s states' timeline (:meth:`_timeline`, *nulls*) read from the records at hand, asking
        nothing: a note on a term ``IS NULL`` asks no log its question does not (``_no_asks``)."""
        kept = self.__dict__.get("_no_asks")
        self.__dict__["_no_asks"] = True
        try:
            return self._timeline(subject, q, nulls=True)
        finally:
            self.__dict__["_no_asks"] = kept

    def _timeline_state_term(self, t, q: Missed) -> T.Said | None:
        """K-3 (a) (bundle K, out/gm/qk-design.md, the delegated design): a term ``IS NULL`` or ``IS NOT NULL``, read on
        its column's states (:meth:`_timeline`, *nulls*): where it surely held for the row and where it surely failed
        (:func:`timeline.state_spans`) -- and, where neither is read, the same with the arms no record reads named as
        having not written the column.  The arms read as writing a value known only at run time as not None are said.
        None where the column has no states read, or nothing is sure either way.  NT-defined-late's
        getDatasetsToFeedContents_JEDI@130 ``tabT.lockedBy IS NULL``: false from 14:56:41.571 (aipanda082's lock) to
        15:57:05.666 (aipanda081's unlockTasks_JEDI)."""
        tl = self._states_at_hand(t.subject, q)
        if tl is None or not tl.read():
            return None
        whole = (q.since, q.moved)
        null = t.node == "null"
        named: list[T.Said] = []
        held, failed = TL.state_spans(null, tl.pieces()), TL.state_spans(not null, tl.pieces())
        if not held and not failed:
            held, failed = TL.state_spans(null, tl.pieces(assume=True)), TL.state_spans(not null, tl.pieces(assume=True))
            if not held and not failed:
                return None
            if tl.opens:
                named.append(T.Said("none-of-wrote-it", {"arms": [o.arm for o in tl.opens], "column": t.subject}))
            if tl.begun_assumed:
                named.append(T.Said("none-of-began-it", {"arms": list(tl.begun_assumed)}))
        if tl.premised:
            named.append(T.Said("a-run-time-value-not-none", {"arms": list(tl.premised), "column": t.subject}))
        parts = [*([T.Said("state-term-held", {"spans": _spans_words(held, whole)})] if held else []),
                 *([T.Said("state-term-failed", {"spans": _spans_words(failed, whole)})] if failed else [])]
        return T.Said("timeline-state-term", {"term": t.text, "parts": parts,
                                              "where": T.Said("state-term-where", {"named": named}) if named else ""})

    def _timeline_terms(self, st, q: Missed) -> list[T.Said]:
        """Q1 part 1: each term of *st*'s condition trees comparing a column read on its timeline with the clock, where
        it surely held for the row (:meth:`_timeline_spans`) -- and, where nowhere, where it did with the arms no record
        reads named as having not written the column.  K-3 (a): a term ``IS NULL`` or ``IS NOT NULL`` on its column's
        states (:meth:`_timeline_state_term`); (d) where a term comparing with the clock surely failed, read the same
        way (:meth:`_timeline_failed_spans`)."""
        out: list[T.Said] = []
        whole = (q.since, q.moved)
        seen: set[str] = set()

        def leaves(node):
            if node.node in ("any", "all"):
                for item in node.items:
                    yield from leaves(item)
            else:
                yield node

        for reading in st.condition_trees:
            for t in leaves(reading.tree):
                if t.node in ("null", "not_null") and t.text not in seen:
                    seen.add(t.text)
                    if (said := self._timeline_state_term(t, q)) is not None:
                        out.append(said)
                    continue
                if t.node == "compare" and not t.settled and t.text not in seen and (
                        said := self._timeline_false_if(t, q)) is not None:
                    seen.add(t.text)
                    out.append(said)
                    continue
                if t.node != "compare" or not t.settled or t.text in seen:
                    continue
                strict = self._timeline_spans(t, q)
                if strict is None:
                    continue
                seen.add(t.text)
                kept = getattr(self, "_assuming", None)
                self._assuming = {}
                try:
                    assumed = self._timeline_spans(t, q) or []
                finally:
                    arms, self._assuming = self._assuming or {}, kept
                # K-3 (d): where it surely failed, and, where nowhere, where it did with the open arms named
                failed = self._timeline_failed_spans(t, q) or []
                a_failed: list = []
                f_arms: dict[str, list[str]] = {}
                if not failed:
                    self._assuming = {}
                    try:
                        a_failed = self._timeline_failed_spans(t, q) or []
                    finally:
                        f_arms, self._assuming = self._assuming or {}, kept
                tl = self._timeline(t.subject, q)
                if strict:
                    out.append(T.Said("timeline-term-held", {"term": t.text, "spans": _spans_words(strict, whole)}))
                elif assumed and t.subject in arms:
                    out.append(T.Said("timeline-term-held-if", {"term": t.text, "spans": _spans_words(assumed, whole),
                                                                "named": _named_open(t.subject, arms, tl)}))
                elif not failed and not (a_failed and t.subject in f_arms):
                    out.append(T.Said("timeline-term-not-held", {"term": t.text}))
                if failed:
                    out.append(T.Said("timeline-term-failed", {"term": t.text, "spans": _spans_words(failed, whole)}))
                elif a_failed and t.subject in f_arms:
                    out.append(T.Said("timeline-term-failed-if", {"term": t.text, "spans": _spans_words(a_failed, whole),
                                                                  "named": _named_open(t.subject, f_arms, tl)}))
        return out

    def _timeline_failed_spans(self, t, q: Missed) -> list | None:
        """K-3 (d) (bundle K, out/gm/qk-design.md, the delegated design): where a term comparing a time column with the
        clock surely failed for the row, on the column's timeline, as :meth:`_timeline_spans` reads where it held
        (:func:`timeline.surely_false` at each of its offsets).  NT-exhausted-late's kickExhaustedTasks_JEDI@1968
        ``modificationTime<:timeLimit``, 240 h as its marks print it, over an eleven-hour stay."""
        tl = self._timeline(t.subject, q)
        if tl is None or not tl.read():
            return None
        assuming = getattr(self, "_assuming", None)
        if assuming is not None and (tl.opens or tl.begun_assumed):
            assuming[t.subject] = [o.arm for o in tl.opens]
        pieces = tl.pieces(assume=assuming is not None)
        out = [(q.since, q.moved)]
        for v in t.values:
            if not v.base or v.base not in UTC_CLOCKS or v.offset_seconds is None:
                return []
            out = _intersect(out, TL.surely_false(t.op, datetime.timedelta(seconds=v.offset_seconds), pieces))
        return _union(out)

    def _timeline_false_if(self, t, q: Missed) -> T.Said | None:
        """E-5 (out/gm/qk-design.md's bundle E, tmaeno's OK 2026-11-17 13:42; the map's ``ConditionValue`` *base* and
        *scale_seconds* beside *said*): a term ``col < clock - X units`` whose X the map does not settle (Watcher's
        main@309, ``naive_utcnow() - datetime.timedelta(hours=workflow_timeout_map['analysis'])``: base naive_utcnow,
        scale -3600) is false at every moment of the stay where X units are at least the most the column's age reaches
        there on its timeline (:meth:`_timeline`; an open arm writes at any moment, and one whose least is not read leaves
        no such bound).  That least X is said, not X itself (a setting).  None for any other term, a column with no
        timeline read, or no bound."""
        if t.op not in ("<", "<=") or len(t.values) != 1:
            return None
        v = t.values[0]
        scale = getattr(v, "scale_seconds", None)
        if (v.base not in UTC_CLOCKS or v.offset_seconds is not None or not scale or scale >= 0
                or not getattr(v, "said", "") or getattr(v, "needs", None)):
            return None
        tl = self._timeline(t.subject, q)
        if tl is None or not tl.read():
            return None
        ages = []
        for p in tl.pieces():
            lo, _ = TL.bounds_at(p, p[1])
            if lo in (TL.LOW, TL.HIGH) or not isinstance(lo, datetime.datetime):
                return None
            ages.append(p[1] - lo)
        if not ages:
            return None
        age = max(ages)
        # the least X, rounded up: the term is false all the stay from it on
        least = f"{math.ceil(age.total_seconds() / -scale * 100) / 100:.2f}".rstrip("0").rstrip(".")
        return T.Said("timeline-term-false-if", {
            "term": t.text, "said": v.said, "least": least, "unit": int(-scale),
            "age": f"{age.total_seconds() / 60:.1f} min", "opens": len(tl.opens), **_timeline_counts(tl)})

    def _timeline_false(self, t, q: Missed) -> str | None:
        """Why a term comparing a time column with the clock is surely false for the row all the window on the
        column's timeline, its open arms writing at any moment (no premise); "" where it is not; None where the
        timeline reads nothing (:meth:`_timeline_spans`)."""
        tl = self._timeline(t.subject, q)
        if tl is None or not tl.read():
            return None
        pieces = tl.pieces()
        for v in t.values:
            if not v.base or v.base not in UTC_CLOCKS or v.offset_seconds is None:
                return ""
            if not TL.surely_false_all(t.op, datetime.timedelta(seconds=v.offset_seconds), pieces):
                return ""
        return T.text(T.Said("false-on-the-timeline", {**_timeline_counts(tl), "absent": len(tl.absent)}))

    def _assumed_true(self, sts: list, q: Missed) -> tuple[list, dict[str, list[str]]]:
        """Where one of *sts* surely returns the row with the arms no record reads named as having not written the
        columns it compares (``_assuming``, as a turn reads them), and those arms by column."""
        self._assuming = {}
        try:
            spans = _union([span for st in sts for span in self._statement_true(st, q)])
        finally:
            assumed, self._assuming = self._assuming or {}, None
        return spans, {col: arms for col, arms in assumed.items() if arms}

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

    def _unanswered(self, subject: str, q: Missed) -> T.Said | None:
        """``_unanswered_why``'s words."""
        why = self._unanswered_why(subject, q)
        return why[1] if why else None

    def _unanswered_why(self, subject: str, q: Missed) -> tuple[str, T.Said] | None:
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
                before = self._before(subject, q)
                if before is None:
                    return "after-only", T.Said("after-only-why")
                if before[0].known:
                    return None
                tl = self._timeline(subject, q)
                if tl is not None and tl.read():
                    if not tl.opens:
                        return None
                    return "after-only", T.Said("after-only-before-why", {"value": q.observed, "why": _timeline_gaps(tl)})
                return "after-only", T.Said("after-only-before-why", {"value": q.observed, "why": _before_gaps(before[0])})
            return "rows-no-column", T.Said("rows-no-column-why", {"attr": attr})
        found = PT.history_by_table(owner)
        if found is None or found.subject.partition(".")[0] != q.subject.partition(".")[0]:
            return "rows-no-row", T.Said("rows-no-row-why", {"owner": owner})
        return None if PT.history_column(found, attr) else ("history-no-column", T.Said("history-no-column-why", {"attr": attr}))

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
        if t.node == "compare" and (got := self._timeline_false(t, q)) is not None:
            return got or None
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
        if t.node == "compare" and (spans := self._timeline_spans(t, q)) is not None:
            return spans
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

    def _statement_true(self, st, q: Missed, keyed: bool = False) -> list:
        """Where in [since, moved] the statement surely returns the row: it holds a value of each subject the map says
        every build restricts and none it refuses, every condition tree is true -- each build's and those some build
        adds, as which build ran is not known -- and so is one running route of calls into it, by that route's own
        reading (a route read as the statement adds nothing); a statement whose every route is not run returns
        nothing, one with no route read has no route to hold.  A statement that asks rows by a value chosen at run
        time (the map's ``keys``: ``WHERE jediTaskID=:jediTaskID``) returns the row only when called with the row's
        value, which is not known here: never surely (KNOWN_ISSUES.md K18) -- unless *keyed*: read as called with the
        row's values (M3(a)-cap, whose selections are the row's label's)."""
        if st.keys and not keyed:
            return []
        st = self._filled(st, q)
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

    def _route_refuses(self, route, q: Missed, terms: set[str] | None = None) -> list[str]:
        """What refuses the row on one route of calls, read from its record alone: a refused pair it holds, a
        restricted subject whose values lack the row's (the asked one compared too: a call that never asks for the
        observed value did not pass the row over), a condition false for it all the window.  *terms*, where handed,
        gets the subjects of the false conditions' terms."""
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
        false = self._trees_false(route.condition_trees, q, said_on)
        if terms is not None:
            terms.update(leaf.subject for _, leaves in false for leaf in leaves)
        said += [self._tree_words(entry, leaves) for entry, leaves in false]
        return said

    def _stamps_itself(self, st, q: Missed, subjects: set[str]) -> list[str]:
        """Bundle D, D-5 (P24, out/gm/qk-design.md, tmaeno's OK 2026-11-17): of the columns *subjects* whose terms keep
        the selection *st* from the row, those its own function writes with the clock (the map's arm with
        ``_written_offset`` 0): lockJobsForFinisher's 895 ``SET modificationTime=CURRENT_DATE`` on each row its 892
        takes, lockJobsForActivator's 941 ``prodDBUpdateTime``.  Where the row's value of one is a time inside the stay
        (after its start, by the time the window ends), every pass that takes the row stamps it again: the selection
        takes the row no more often than the gap its term reads.  Which write put the row's value is not read: a pass
        of the selection, or another write after it (NJ-transferring-stuck's 09:39:56, 24 s after the last Finisher
        round's prints, by the updateJob that follows the round)."""
        out = []
        own = _bare(st.owner)
        began = q.began if q.began is not None else q.since
        for subject in sorted(subjects):
            # the row's value first: the map's writers are read only for a time in the stay
            known, held = self._held(subject, q)
            try:
                value = _when(str(held)) if known and held is not None else None
            except (TypeError, ValueError):
                value = None
            if value is None or not began < value <= q.moved:
                continue
            arms = sorted({b.line for j in self._subject_writers(subject) if _bare(j.owner) == own for b in j.branches
                           if _written_offset(b) == datetime.timedelta(0)})
            if not arms:
                continue
            out.append(f"{own.split('::')[-1]} writes {subject} with the clock itself on the rows it takes (at {arms}), "
                       f"and the row's {held} falls in the stay, after {began} (which write put it there is not read): "
                       "each pass of it that takes the row stamps it again, so it takes the row no more often than "
                       "its term's gap")
        return out

    def _stamps_by_writes(self, st, q: Missed) -> list[T.Said]:
        """Bundle H, H-2 (P5, out/gm/qk-design.md, bundle H, tmaeno's OK 2026-11-17 17:1x): D-5's reading
        (:meth:`_stamps_itself`) where the row's value is no time in the stay, from the writes the records show
        instead: a column of *st*'s terms whose every write read in the stay (its timeline, two or more) is by an arm
        of *st*'s own function writing the clock itself (``_written_offset`` 0) -- getTasksToExecCommand_JEDI's 4571
        ``SET modificationtime=CURRENT_DATE``, 93 times in MT-finishing-late's stay before the move.  Each pass that
        took the row stamped it again, so it took the row no more often than its term's gap: on the writes the records
        show."""
        own = _bare(st.owner)
        func = own.split("::")[-1].rpartition(".")[2]
        out = []
        for subject in sorted(_statement_subjects(st)):
            tl = self._timeline(subject, q)
            # the writes of the stay before the move's second: one in it is the moving pass's (MT-finishing-late's
            # updateTask_JEDI at 07:49:38.704), not a take's
            moved = q.moved.replace(microsecond=0)
            writes = [w for w in (tl.writes if tl is not None else ()) if q.since <= w.at < moved]
            if len(writes) < 2:
                continue
            arms = sorted({f"{func}:{b.line}" for j in self._subject_writers(subject) if _bare(j.owner) == own
                           for b in j.branches if _written_offset(b) == datetime.timedelta(0)})
            if not arms or not all(w.arms and {a.rpartition(".")[2] for a in w.arms} <= set(arms) for w in writes):
                continue
            out.append(T.Said("stamps-itself-by-writes", {"func": func, "subject": subject, "arms": arms, "n": len(writes),
                                                         "first": f"{writes[0].at:%H:%M:%S}", "last": f"{writes[-1].at:%H:%M:%S}"}))
        return out

    def _routes(self, st, q: Missed) -> tuple[str, list[str]]:
        """L2's verdict from the routes of calls into *st* (the map's ``call_readings``): ``refused`` when the map
        places every call (``callers_complete``) and every route that runs the statement refuses the row;
        ``refused on a call`` for the routes that do; else nothing.  A route reading as the statement refuses only
        what the statement does, and that was said before the routes are asked.  A refusing term on a column the
        selection stamps with the clock itself is said so (:meth:`_stamps_itself`)."""
        lines, running, refusing = [], 0, 0
        terms: set[str] = set()
        for route in st.call_readings:
            if route.not_run:
                continue
            running += 1
            why = [] if route.as_statement else self._route_refuses(route, q, terms)
            if not why:
                continue
            refusing += 1
            constants = ", ".join(f"{k}={v}" for k, v in sorted(route.constants.items()))
            via = f" via {', '.join(route.chain[:-1])}" if len(route.chain) > 1 else ""
            lines.append(f"called from {route.chain[-1]}{via} ({constants}): " + "; ".join(why))
        if not refusing:
            return "not settled", []
        lines += self._stamps_itself(st, q, terms)
        if st.callers_complete and refusing == running:
            return "refused", [f"every call refuses it: {running} route(s), and the map places every call", *lines]
        stop = "; ".join(st.callers_stop) or "the map does not say it places every call"
        return "refused on a call", [*lines, f"{refusing} of {running} route(s) refuse it; {stop}"]

    def _amounts_printed(self, owner: str, line: int, asking: W.Asking, q: Missed) -> tuple[dict[str, set[str]], list[str], str]:
        """What the marks of *owner*'s runs of the row over *asking*'s window print for each parameter the map names a
        hole of them (:func:`mark_parameters`): the values by parameter, the runs read (``log@machine``), and why none
        was read where none was.  Asked as S1 asks a mover's logs, over the step's own window, the logs' notes not
        said; an ask of them with no record is not kept.

        A run is the row's unless a hole of its mark whose columns the map names (``RunKey.columns``) holds another
        value than the row held (:meth:`_held`): ``kickExhaustedTasks_JEDI < vo=atlas label=managed ... >`` selects no
        task of another label.  Where no run of the row's is read, the function's mark itself is asked -- it holds no
        key, so the door asks it only near the key's records (KNOWN_ISSUES.md K60) -- by its fixed head as the map
        reads it (``kickExhaustedTasks_JEDI < vo=``), over the window, on one machine: the last one the key's own
        records in that log are on, else the log's first JEDI machine; the ask goes to the asks file for the next
        round (B, P29)."""
        cache: dict[tuple, tuple[dict[str, set[str]], list[str], str]] = self.__dict__.setdefault("_amounts_cache", {})
        window = asking.window
        if (owner, window) in cache:
            return cache[(owner, window)]
        func = owner.rpartition("::")[2].rpartition(".")[2]
        values: dict[str, set[str]] = {}
        read: list[str] = []
        others = 0

        def take(name: str, runs: list, where: str = "") -> None:
            nonlocal others
            for run in runs:
                if not of_mover(run, func) or (window is not None and not spans(run, window)):
                    continue
                got = mark_parameters(owner, [line], run)
                if not got:
                    continue
                if self._another_rows_run(owner, [line], run, q):
                    others += 1
                    continue
                read.append(f"{name}@{run.machine}{where}")
                for parameter, value in got.items():
                    values.setdefault(parameter, set()).add(value)

        names = self.logs.names(self.files_of(owner, [], [line])) if window is not None or not self.logs.needs_window else []
        owners = owners_of([owner])
        found, missed, _holes = self._ask(self.args.key, names, asking, owners, quiet=True) if names else ({}, [], {})
        for name, runs in found.items():
            take(name, runs)
        why = "an ask of its logs has no record" if missed else "no run of it in the records holds its mark"
        heads = print_heads(owner, [line])
        if not read and not missed and names and heads:
            asked = []
            for name in names:
                keyed_runs = found.get(name) or []
                if keyed_runs:
                    machine: str | None = max(keyed_runs, key=lambda r: r.last).machine
                else:
                    log = self.logs.log(name, self.args.key, window, machine=asking.machine, owners=owners, strings=asking.strings)
                    machine = min(log.jedi) if log.jedi else None
                on = f" on {machine}" if machine else ""
                try:
                    log = self.logs.log(name, self.args.key, window, machine=machine, owners=owners, strings=heads)
                except Unrecorded as e:
                    because = (f"what {func}'s runs' marks print for the amount a statement of it waits for: its mark "
                               f"holds no key, so it is asked on its own (P29)")
                    self._keep_unanswered([(ask, because) for ask, _ in e.asks])
                    asked.append(f"{name}{on}")
                    continue
                take(name, runs_for(log, func), " by its mark asked alone")
            if asked:
                why = f"no run of it in the records holds its mark; its mark {list(heads)} is asked of {', '.join(asked)}"
            elif not read:
                why = f"no run of it in the records holds its mark, nor does the answer to its mark {list(heads)} asked alone"
        if others and not read:
            why += f" ({others} run(s) of another row's values left out)"
        cache[(owner, window)] = (values, sorted(set(read)), why)
        return cache[(owner, window)]

    def _another_rows_run(self, owner: str, lines: list[int], run: Any, q: Missed) -> bool:
        """Whether a hole of *run*'s mark whose columns the map names (``RunKey.columns``) holds another value than the
        row held (:meth:`_held`): ``kickExhaustedTasks_JEDI < vo=atlas label=managed ... >`` selects no task of another
        label (P29's B-2)."""
        return any(known and held is not None and str(held) != value
                   for column, value in mark_columns(owner, lines, run).items()
                   for known, held in [self._held(column, q)])

    def _row_mark_heads(self, owner: str, lines: list[int]) -> tuple[tuple[str, ...], dict[str, str]]:
        """The heads of *owner*'s marks for the row: each printed prefix (:func:`print_heads`) taken on past a hole while
        the map names a column of it (``RunKey.columns``) the rows file's row holds -- ``reactivatePendingTasks_JEDI <
        vo=atlas label=managed limit=`` -- so an ask by it returns the passes that may have selected the row, with every
        value of the parameters after; and the columns so read, with their values.  The agent's loop chooses them each
        selection, and they are read as the row's, as M3(a)-cap reads its keys (:meth:`_label_of`): a premise."""
        out = set()
        used: dict[str, str] = {}
        found = R._prints_of(owner, lines)
        for st in getattr(found, "statements", None) or []:
            for r in st.rows or []:
                if not getattr(r, "prefix", None):
                    continue
                parts = [re.sub(r"\\(.)", r"\1", p) for p in r.prefix.split(".*?")]
                text = parts[0]
                # the holes in the mark's order: a run's key of as many holes, else the print's own
                holes = next((run.key for run in getattr(found, "runs", None) or [] if len(run.key or []) == len(parts) - 1),
                             r.prefix_holes or [])
                for hole, after in zip(holes, parts[1:], strict=False):
                    held = next(((c, v) for c in hole.columns or [] if (v := self._label_of(c, c.partition(".")[2]))), None)
                    if held is None:
                        break
                    used[held[0]] = held[1]
                    text += held[1] + after
                out.add(text)
        return tuple(sorted(out)), used

    def _fill(self, st, q: Missed) -> tuple[Any, list[str]]:
        """*st* with each amount that waits for a parameter of its own function filled (P29): a value the map settles
        but for one parameter times a scale (``ConditionValue.needs``, ``scale_seconds``; its batch 4 S), where a hole
        of the function's runs' marks holds that parameter as handed (``RunKey.parameter``, batch 5 M-c) and the runs
        over the window (*q*'s ``amounts``, else the part of the stay judgement 3 asks first) print one value: the
        amount is the scale times it, and a term none of whose values then waits is read as settled.  Runs printing
        several values, or none read, leave it waiting.  With the words that say where each value came from, or why
        it was not read.  ``reactivatePendingTasks_JEDI``'s ``modificationTime<:timeLimit``, ``limit=20``: now less
        1200 s."""
        waits: set[str] = set()
        for reading in getattr(st, "condition_trees", None) or []:
            for leaf in _tree_leaves(reading.tree):
                for v in getattr(leaf, "values", None) or []:
                    needs = getattr(v, "needs", None) or []
                    if len(needs) == 1 and getattr(v, "scale_seconds", None) is not None and not getattr(v, "said", ""):
                        waits.add(needs[0])
        if not waits:
            return st, []
        # its own function's parameters, a hole of whose marks the map says holds one
        func = st.owner.rpartition("::")[2].rpartition(".")[2]
        waits = {n for n in waits if n.partition(".")[0] == func}
        if not waits or not (waits := waits & printed_parameters(st.owner, [st.line])):
            return st, []
        since, end = f"{q.since:%Y-%m-%d %H:%M:%S}", f"{q.moved:%Y-%m-%d %H:%M:%S}"
        asking = q.amounts or W.last_part(q.observed, since, end, "the-move-less", T.Said("the-move-at", {"move": end}))
        key = (id(st), asking.window, since, end)
        cache: dict[tuple, tuple[Any, list[str]]] = self.__dict__.setdefault("_fill_cache", {})
        if key in cache:
            return cache[key]
        values, read, none_why = self._amounts_printed(st.owner, st.line, asking, q)
        window = asking.window or (since, end)
        fill: dict[str, float] = {}
        words = []
        for parameter in sorted(waits):
            name = parameter.partition(".")[2]
            got = sorted(values.get(parameter, set()))
            number = _number(got[0]) if len(got) == 1 else None
            if number is not None:
                fill[parameter] = number
                words.append(f"{name} = {got[0]}, as the marks of {func}'s runs over {window[0]} .. {window[1]} print it "
                             f"({', '.join(read)})")
            elif len(got) > 1:
                words.append(f"{name} not read: {func}'s runs over {window[0]} .. {window[1]} print {got} ({', '.join(read)})")
            else:
                words.append(f"{name} not read: {none_why} over {window[0]} .. {window[1]}")
        out = copy.deepcopy(st) if fill else st
        for reading in out.condition_trees if fill else []:
            for leaf in _tree_leaves(reading.tree):
                filled = False
                for v in getattr(leaf, "values", None) or []:
                    needs = getattr(v, "needs", None) or []
                    if len(needs) == 1 and needs[0] in fill and v.scale_seconds is not None and not v.said:
                        v.offset_seconds = (v.offset_seconds or 0) + v.scale_seconds * fill[needs[0]]
                        v.needs = []
                        filled = True
                if filled and not leaf.settled and leaf.node != "other" and all(_settled(v) for v in leaf.values):
                    leaf.settled = True
        cache[key] = (out, words)
        return out, words

    def _filled(self, st, q: Missed):
        """*st* as :meth:`_fill` fills it, without the words."""
        return self._fill(st, q)[0]

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
        self.need(T.Said("the-map-s-condition-trees-and-routes"),
                  hasattr(st, "condition_trees") and hasattr(st, "call_readings"))
        st, amounts = self._fill(st, q)
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
        unanswered = {x: found for x in sorted(_statement_subjects(st)) if (found := self._unanswered_why(x, q))}
        if unanswered:
            said.append("not compared: " + "; ".join(f"{x} ({why})" for x, (_, why) in unanswered.items()))
        told = self.__dict__.setdefault("_timelines_told", set())
        for x in sorted(_statement_subjects(st)):
            if (before := self._before(x, q)) is not None and before[0].known:
                said.append(T.text(_held_before_said(x, q.observed, *before)))
            elif arms := self._born_only(x, q):
                said.append(T.text(T.Said("born-only-held", {"subject": x, "arms": arms})))
            elif before_ := self._written_only_before(x, q):
                said.append(T.text(T.Said("written-only-before-held", {
                    "subject": x, "since": f"{q.since:%Y-%m-%d %H:%M:%S}",
                    "arms": [f"{arm} ({why})" for arm, why in before_]})))
            elif (tl := self._timeline(x, q)) is not None and tl.read() and x not in told:
                # Q1 part 1: what the records show of it in the stay, said once
                told.add(x)
                said.append(T.text(_timeline_said(x, tl)))
        for x, terms in sorted(_state_terms(st).items()):
            # K-3 (a): a column tested IS NULL, said no other way: what the records show of its states, said once, where
            # they show a write of it in the stay or a term on it reads something sure (:meth:`_timeline_state_term`)
            if x not in told and f"{x} states" not in told and (
                    tl := self._states_at_hand(x, q)) is not None and tl.read() and (
                    any(tl.start < w.at <= tl.end for w in tl.writes)
                    or any(self._timeline_state_term(t, q) is not None for t in terms)):
                told.add(f"{x} states")
                said.append(T.text(_timeline_said(x, tl)))
        said += amounts
        said += [T.text(x) for x in self._timeline_terms(st, q)]
        if kind == "refused":
            return kind, said
        by_call, calls = self._routes(st, q)
        # H-2: D-5's own sentence, where the routes said it, is not said again from the writes
        said += [x for x in map(T.text, self._stamps_by_writes(st, q))
                 if not any(c.startswith(x.partition(" with the clock itself")[0]) for c in calls)]
        kind = min(kind, by_call, key=KINDS.index)
        if kind != "refused":
            said += self.ask_linked(st, [x for x, (code, _) in unanswered.items() if code == "rows-no-row"], q)
        return kind, calls + said

    def ordered(self, q: Ordered):
        self.start("L3", "what it lost to")
        st = q.statement
        where = f"{st.owner.split('::')[-1]}@{st.line}"
        self.say(f"{where} returns the first rows ({', '.join(st.limits)}) of one of {len(st.order_by)} order(s)")
        self.need(T.Said("the-map-s-prints-of-a-statement"),
                  hasattr(sqltext, "cut") and hasattr(sqltext, "unfilled") and hasattr(st, "printed_by"))
        self.need(T.Said("a-print-of-s-text-and-binds", {"where": where}), st.printed_by)
        # the call's own SQL: the records of the print the map names, cut into the text and the binds it ran with,
        # the text read as the map reads statements; the query an order of the statement's sorted
        rel, func = st.owner.split("::", 1)
        fn = R.function(rel, func, within=[st.line])
        used = {}
        for printed in st.printed_by:
            row = self.need(T.Said("s-print-at-among-its-rows", {"func": func, "printed_line": printed.line}), next((r for r in fn.rows if r.line == printed.line), None))
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
        self.need(T.Said("one-order-the-call-that-moved-it"), len(used) == 1)
        (took, (record, read, query, binds)), = used.items()
        others = [order for order in st.order_by if order != took]
        self.need(T.Said("one-other-order-the-statement-can-be"), len(others) == 1)
        other = others[0]
        cls = self.args.subject.split(".")[0]
        row = {k.lower(): v["value"] for k, v in self._row_of(cls).items()}
        key_col = next((k for k, v in row.items() if str(v) == self.args.key), None)
        # the WHERE and the binds of the query that sorted
        unfilled = sorted({b for t in query.where for b in sqltext.unfilled(t.text, binds)})
        self.need(T.Said("the-run-s-value-of-every-bind", {"unfilled": unfilled}), not unfilled)
        filled = [sqltext.filled(t.text, binds) for t in query.where]
        cut = " AND ".join(filled) if filled else None
        self.need(T.Said("the-where-and-binds-of-the-call", {"failure": T.Said("in-parens", {"what": read.failure}) if read.failure else ""}),
                  cut and key_col)
        self.say(f"  the call's WHERE: {cut}")
        # the order, one rule: its items before the key group the rows, the key ranks them within a group, and the
        # items after it decide nothing (the key is unique); an order of the key alone is one group.  A blanked item
        # the map's order_holes says ranks by a column of the row (getJobs' (CASE ...) by gshare) is grouped by, its
        # ranking computed at run time -- a row of another value is neither ahead nor behind here
        def sort_of(order: str) -> tuple[list[tuple[str, str]], list[str], list[tuple[str, str, Any]], str | None]:
            holes = {h.item: h for h in getattr(st, "order_holes", None) or [] if h.order == order}
            cols: list[tuple[str, str]] = []
            blank: list[str] = []
            ranked: list[tuple[str, str, Any]] = []
            key_order = None
            for i, item in enumerate(sqltext.order_items(order)):
                hole = holes.get(i)
                column = item.expression.split(".")[-1].lower() if not item.blanked and item.columns == (item.expression,) else None
                if column is not None and column == key_col:
                    key_order = "desc" if item.descending else "asc"
                    break
                if column is not None:
                    cols.append((column, "desc" if item.descending else "asc"))
                elif hole and hole.subjects and all(x.partition(".")[0] == cls for x in hole.subjects):
                    ranked += [(item.expression, x.partition(".")[2].lower(), hole.computed_at) for x in hole.subjects]
                    cols += [(x.partition(".")[2].lower(), "ranked") for x in hole.subjects]
                else:
                    blank.append(item.expression)
            return cols, blank, ranked, key_order

        sorts = {took: sort_of(took), other: sort_of(other)}
        for order, whose in ((took, "the call that took it sorted by it"), (other, "no call is shown to have sorted by it")):
            cols, blank, _, key_order = sorts[order]
            self.say(f"  under {order!r} ({whose}) the row sorts by "
                     + "".join(f"{c} {'(ranked at run time)' if d == 'ranked' else d} -> {row.get(c, '?')}; " for c, d in cols)
                     + (f"{'then its key' if cols else 'its key alone,'} {key_col} {key_order} -> {row.get(key_col)}" if key_order else
                        f"the order does not hold its key {key_col}: its ties are in an order the database picks")
                     + (f"; the map leaves {blank} blank" if blank else ""))
        history = self.need(T.Said("a-history-of-to-ask-the-window", {"subject": self.args.subject}), PT.history(self.args.subject))
        # the call's terms: on the subject, the history's value stands for them; on the columns the history holds, read
        # at the moment (the move, or the row each left); the others on the row as it is now
        attr = self.args.subject.split(".")[1].lower()
        held, now = [], []
        for t, f in zip(query.where, filled, strict=True):
            named = {c.split(".")[-1].lower() for c in t.columns}
            if attr in named:
                self.need(T.Said("a-term-of-the-call-s-where", {"subject": self.args.subject, "f": f, "fact": sorted(named - {attr})}), named == {attr})
                continue
            (held if named and named <= history.held else now).append(f)
        miss = q.q
        mine = row.get(key_col)
        lookback = miss.moved - L3_LOOKBACK

        def questions_of(order: str, named: bool = True) -> dict[str, Any]:
            """The two questions under *order*, by the names they are asked and answered under; *named* False gives
            the names L3 gave them when it counted the statement's other order alone (before 2026-11-15), the order
            then left out of the name."""
            cols, _, _, key_order = sorts[order]
            under = f", under {order!r}" if named else ""
            return {
                f"{where} held {miss.observed!r} at the move{under}, by the order's items before the key": PT.HeldAt(
                    history, miss.observed, miss.moved, lookback, held, now, cols, key_col, mine, key_order
                ),
                f"{where} left {miss.observed!r} in the window{under}, by the value it went to and the key": PT.MovedOn(
                    history, miss.observed, miss.since, miss.moved, held, now, cols, key_col, mine, key_order
                ),
            }

        answers = self.data.get("answers", {})

        def number(value):
            try:
                return float(value)
            except (TypeError, ValueError):
                return value

        def count(rs: list, column: str) -> int:
            return sum(int(r.get(column) or 0) for r in rs)

        def tally(order: str, qs: dict[str, Any]) -> tuple[dict[str, Any], T.Said | None]:
            """The rows ahead of the row under *order*, from the rows file's answers to its two questions *qs*: what
            the END would say of them, or what of the answers does not hold (None the first where they do)."""
            cols, _, ranked, key_order = sorts[order]
            (at_ask, at_q), (window_ask, window_q) = qs.items()
            moment, window = answers[at_ask], answers[window_ask]
            for ask, pq, answer in ((at_ask, at_q, moment), (window_ask, window_q, window)):
                # an answer does not carry the SQL it answers: it must at least hold every column the question's SQL names
                got = {k.lower() for r in answer["rows"] for k in r}
                if lacking := [c for c in PT.columns(pq) if c not in got]:
                    return {}, T.Said("the-answer-to-holding-the-column-s", {"ask": ask, "lacking": lacking, "held": sorted(got)})

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

            rows_at = [{k.lower(): v for k, v in r.items()} for r in moment["rows"]]
            own = [r for r in rows_at if same(r)]
            if count(own, "n_self") != 1:
                return {}, T.Said("the-row-itself-once-in-its-own", {"miss_observed": miss.observed, "fact": count(own, 'n_self'),
                                                                     "fact_2": count(rows_at, 'n_self'), "fact_3": now or 'none'})
            n_groups, n_own = count([r for r in rows_at if ahead(r)], "n"), count(own, "n")
            n_key = count(own, "n_key_ahead") if key_order else 0
            n_other = count([r for r in rows_at if other_rank(r)], "n")
            firsts = sorted(str(r["first_entered"])[:19] for r in rows_at if r.get("first_entered"))
            names = [c for c, d in cols if d == "ranked"]
            mine_rank = f" of its {', '.join(f'{c} {row.get(c)!r}' for c in names)}" if names else ""
            others = (f", {n_other} of another {'/'.join(names)}, placed by the ranking computed at run time"
                      f" ({'; '.join(at for *_, at in ranked)})") if names else ""
            how = ((f"{n_groups} in groups ahead, " if cols else "")
                   + (f"{n_key} of its own group's {n_own} with a key ahead" if key_order else
                      f"its own group's {n_own}, their order among them not known"))
            total = n_groups + n_key
            self.say(f"    at the move ({miss.moved}): {total} row(s){mine_rank} ahead of it ({how}){others}; the earliest "
                     f"entry counted {firsts[0] if firsts else 'none'}, the lookback {lookback} ({moment['source']})")
            rows_win = [{k.lower(): v for k, v in r.items()} for r in window["rows"]]
            self.say(f"    left {miss.observed!r} in the window, by the value it went to ({window['source']}):")

            def place(r: dict) -> list:
                return [str(r.get(c)) if d == "ranked" else number(r.get(c)) for c, d in cols]

            for nxt in sorted({str(r.get("next_value")) for r in rows_win}):
                for r in sorted((r for r in rows_win if str(r.get("next_value")) == nxt), key=place, reverse=True):
                    mark = "its group" if same(r) else "other" if other_rank(r) else "ahead" if ahead(r) else "behind"
                    self.say(f"      {mark:9} {' '.join(f'{k}={v}' for k, v in r.items())}")
            mine_left = [r for r in rows_win if same(r)]
            if not mine_left:
                return {}, T.Said("its-own-group-among-the-rows-that", {"miss_observed": miss.observed})
            return {
                "total": total, "mine_rank": mine_rank, "how": how,
                "at_least": bool(firsts) and _when(firsts[0]) - lookback < datetime.timedelta(hours=1), "first": firsts[0] if firsts else None,
                "left": "; ".join(f"{r.get('n')} to {r.get('next_value')!s} after {r.get('min_h')}-{r.get('max_h')} h"
                                  + (f" ({r.get('n_key_ahead')} with a key ahead)" if key_order else "") for r in mine_left),
                "as_now": [*now, *(f"the order's {c}" for c, _ in cols)],
            }, None

        # the rows ahead are asked under the order the call that took the row sorted by: an order no call is shown to
        # have used says nothing of why the row waited (KNOWN_ISSUES.md K48)
        questions = questions_of(took)
        asks = {ask: PT.draw(pq) for ask, pq in questions.items()}
        for ask, sql in asks.items():
            self.say(f"-- {ask}{' (answered in the rows file)' if ask in answers else ''}")
            for line in sql.splitlines():
                self.say(f"   {line}")
            if ask not in answers:
                # no db_read form: both are windows over the history; the structure lets a fetch answer by the parts
                self.ask_rows({"kind": "sql", "name": ask, "sql": sql, "binds": {}, "question": PT.asked(questions[ask])},
                              f"L3's answer {ask!r}, not in the rows file", "answers")
        # the statement's other order beside it, counted where the rows file answers its questions -- under their names
        # or the ones L3 gave them before -- and never asked
        beside = next((qs for qs in (questions_of(other), questions_of(other, named=False)) if all(a in answers for a in qs)), None)
        if beside is None:
            self.say(f"  under {other!r}: not counted, nor asked (no answer in the rows file; no call is shown to have sorted by it)")
        else:
            self.say(f"  under {other!r}, beside (no call is shown to have sorted by it), by the rows file's answers:")
            facts, problem = tally(other, beside)
            if problem is not None:
                self.say(f"    not counted: {T.text(problem)}")
            else:
                at_least = T.text(T.Said("at-least-within-an-hour", {"first": facts["first"], "lookback": lookback})) if facts["at_least"] else ""
                as_now = T.text(T.Said("read-as-now", {"as_now": facts["as_now"]})) if facts["as_now"] else ""
                self.say(f"    at the move {facts['total']} row(s){facts['mine_rank']} ahead of it ({facts['how']}){at_least}; its group "
                         f"left {miss.observed!r}: {facts['left']}{as_now} -- not why it waited: no call is shown to have sorted by it")
        self.need(T.Said("the-answers-to-in-the-rows-file", {"asks": list(asks)}), all(ask in answers for ask in asks))
        self.need(T.Said("a-count-under-the-order-that-took-it", {"took": took}), False)
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
    ap.add_argument("--since", default=None, help="with --late, the stay asked: the time_extended (or, in a row with none, the time) of "
                                                  "the statuslog row that begins it; needed where the value has more than one stay")
    args = ap.parse_args()
    if args.since and not args.late:
        ap.error("--since names a stay of --late")
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
        outcome = {"kind": "end", "step": r.ends[-1][0] if r.ends else r.hop, "record": _ends(r.ends), "code": r.end_code}
    except Stop as stop:
        r.say()
        for line in A.summary():
            r.say(line)
        r.say(f"STOP {stop}; HAND used {len(r.hand_used)} time(s) {r.hand_used}")
        # rows asked and no other rule naming the stop: answering them may move it -- not "asks", which the harness
        # records the logs for
        outcome = {"kind": "stop", "step": r.hop, "record": stop.said, "code": stop.code or r.rows_code()}
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
