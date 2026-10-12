"""A case run as one loop, with no one choosing between the steps.

    python -m bamboo.querier.rerun --subject JobSpec.jobStatus --observed failed --key <PandaID> \
        --log panda-adder.log --log panda-setupper.log \
        --rows rows.json [--hand hand.json] [--patterns map|outside] [--decide positive|strict]

Every step takes its arguments from the symptom, the logs, the rows, or the
step before it.  No line number, function or file of a case is written here:
where a step cannot find what the next one needs, it looks in ``--hand`` and
prints ``HAND``, and without an entry there it prints ``STOP`` and ends.  Run
with no hand file, the steps before the first ``STOP`` are the ones that ran
with no judgement.

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
                                        (``proxy_reading``), and the counts the passes printed
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
from dataclasses import dataclass
from pathlib import Path

from bamboo.codemap import strategy as strategy_mod
from bamboo.codemap import trace
from bamboo.codemap.calls import Types
from bamboo.codemap.lookup import CodeMap
from bamboo.codemap.models import Symptom
from bamboo.codemap.panda import pathcond, sqltext
from bamboo.codemap.panda import sql as sql_mod
from bamboo.codemap.panda.provenance import classify, facade_calls
from bamboo.codemap.panda.recognizers.sqlwrite import _flag_parameters
from bamboo.querier import census as C
from bamboo.querier import evaluate as E
from bamboo.querier import false_guard as G
from bamboo.querier import outbound as O
from bamboo.querier import parts as PT
from bamboo.querier import raised as X
from bamboo.querier import reproduce as P
from bamboo.querier import rows as R
from bamboo.querier import value_diff as V
from bamboo.querier.records import CORPUS, read_log, runs_for
from bamboo.querier.which_ran import Candidate, class_of, judge, stable_holes
from bamboo.scripts.derive_strategy import _investigate

BAMBOO = Path(R.BAMBOO_TREE)
#: where the assumption of reading 2 in S5 puts a job out of the active tables -- PanDA's,
#: written here because the map does not say which rows a table holds
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


def argument_said(param: str | None, arg: ast.expr, value: object) -> str:
    """``onlyActive = True`` for a literal, ``jediTaskID = 12345`` where the argument is the parameter's name,
    ``param = <argument> = value`` otherwise."""
    if param is None or (isinstance(arg, ast.Name) and arg.id == param):
        return f"{ast.unparse(arg)} = {value!r}"
    if isinstance(arg, ast.Constant):
        return f"{param} = {value!r}"
    return f"{param} = {ast.unparse(arg)} = {value!r}"


class Stop(Exception):
    pass


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
    """Which way each pass of a mover went."""

    owner: str
    log: str
    arms: list[Arm]
    key: str


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


@dataclass
class Ordered:
    """A statement returning only the first rows of an order, and the call of it that moved the value on."""

    statement: object
    run: object
    q: Missed


#: the clocks the map names that read one UTC clock: PanDA's Oracle session runs in UTC (WrappedCursor.initialize,
#: ``ALTER SESSION SET TIME_ZONE='UTC'``); SYSDATE, SYSTIMESTAMP and datetime.now read a host's zone, unknown here
UTC_CLOCKS = {"naive_utcnow", "datetime.utcnow", "datetime.now(utc)", "CURRENT_DATE", "CURRENT_TIMESTAMP"}
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


class Rerun:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.mode, self.decide = args.patterns, args.decide
        self.data = json.loads(Path(args.rows).read_text())
        self.hand = json.loads(Path(args.hand).read_text()) if args.hand else {}
        self.hand_used: list[str] = []
        self.logs = {}
        for path in args.log:
            log = read_log(path)
            if log.name in self.logs:
                raise SystemExit(f"{path} and {self.logs[log.name].path} are both {log.name}: give one file per log")
            self.logs[log.name] = log
        self.lines: list[str] = []
        self.hop = ""
        self.touched: list[tuple[R.Function, object]] = []  # (function, run) this investigation read
        #: what the runs read so far printed about a row, with where and when -- a value
        #: printed at one time and used for a decision at another says so in its source
        self.row_facts: dict[str, tuple[str, str]] = {}
        self.question = None  # the run the symptom's value changed in

    # ------------------------------------------------------------ plumbing
    def say(self, text: str = "") -> None:
        self.lines.append(text)
        print(text, flush=True)

    def need(self, what: str, found):
        if found:
            return found
        if self.hop in self.hand:
            self.say(f"HAND {self.hop}: {what} <- {self.hand[self.hop]!r}")
            self.hand_used.append(self.hop)
            return self.hand[self.hop]
        raise Stop(f"{self.hop}: {what}")

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
        self.say()
        self.say(f"== {hop} {title}")

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

    def base_env(self, file_row: dict | None = None) -> E.Env:
        sp = self.data["spellings"]
        env = E.Env(rows={s: entity for entity, spellings in sp.items() if entity != "about" for s in spellings})
        for entity, row in self.data.get("rows", {}).items():
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

    def runs(self, key: str) -> dict[str, list]:
        return {name: runs_for(log, key) for name, log in self.logs.items()}

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
        self.say(f"# {when}: bamboo {head}, panda-server {corpus}")
        dump = os.environ.get("MAP_DUMP")
        self.say(f"# map: dump {dump} (sha256 {hashlib.sha256(Path(dump).read_bytes()).hexdigest()[:16]})"
                 if dump else "# map: the one stored in Neo4j")
        if corpus.endswith("-dirty"):
            self.say("# the corpus is dirty: this run is void")
        if head.endswith("-dirty"):
            self.say("# the bamboo tree is dirty (another session may be editing it): this run does not measure a commit")

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
        while q is not None:
            q = steps[type(q)](q)

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

    def _which_wrote(self, subject: str, value: str, key: str, when: str | None = None) -> Wrote:
        """H1 and H2: the map's writers of ``subject=value``, and which of them the run of the key at *when* shows ran.

        *when* is the statuslog time of *value* unless given: ``--late`` asks it at the row after the observed one.
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
        if when is None:
            states = self.data["statuslog"]["states"]
            when = self.need(f"a statuslog time for {value}", next((s["time"] for s in states if s["status"] == value), None))
        runs = self.runs(key)
        question = self.need(
            f"a run of {key} at {when}",
            next((r for rs in runs.values() for r in rs if r.first[:19] <= when <= r.last[:19]), None),
        )
        self.say(f"the run: {question.describe()}")
        self.question = question
        cands = [Candidate(r.file, r.owner.split("::")[-1], r.lines, self.files_of(r.owner, r.log_files, r.lines)) for r in readings]
        for r, c in zip(readings, cands, strict=False):
            if question.log.name in c.log_files:
                self.warn_split(r.owner, r.lines)
        env = self.base_env()
        seens = [R.seen(R.function(c.rel, c.func, c.lines), question, self.mode) for c in cands if question.log.name in c.log_files]
        stable_holes(seens, env)
        j = judge(
            cands, runs, mode=self.mode, decide=self.decide, window=(when, when),
            subject=attr, observed=value, question_run=question, env=env,
        )
        self.say(j.report(question))
        chosen = self.pick("which candidate ran", j)
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
        obj_runs = self.runs(obj)
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
        env7 = self.base_env(file_row)
        env7.put(make.first_arg, obj, "the object's name, bound by the loop over names")
        seens7 = [R.seen(fn7, r2, self.mode)]
        stable_holes(seens7, env7)
        self.say(f"candidates: the make call {make.line}, and the branches around it {arms[1:]} + implicit else of {[i.lineno for i in implicit]}")
        # an implicit else has no statement to read rows from: its test is evaluated instead
        for i in implicit:
            value = E.evaluate(ast.unparse(i.test), env7)
            used = {k: v for k, v in env7.used.items() if k in ast.unparse(i.test)}
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
        other_env = self.base_env()
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
            self.say(f"END {end}")
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
        env = self.base_env()
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
        found = self.need(f"the value of {what} (a field of the {m[1] if m else '?'} row: the rows file does not have it)",
                          None if value is E.UNKNOWN else repr(value))
        self.say(f"END {what} = {found}")
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
        symptom = Symptom(subject=q.subject, observed=q.observed, task_id=q.key)
        inv, _ = asyncio.run(_investigate("panda", None, symptom, None, q.key, None, 1, None))
        R.check_blobs(inv, "derive-strategy's readings")
        strategy = inv.hops[0].strategy
        follow = self.need("the map's answer to what moves the value on", strategy.follow_up)
        strategy_mod.attach_mover_gaps(strategy, R.ROOTS)
        movers: dict[str, str] = {}
        for rs in follow.reader_statements:
            reader = f"{rs.owner.split('::')[-1]}@{rs.line}"
            for owner in rs.crosses_to or [rs.owner]:
                movers.setdefault(owner, f"{reader} hands the rows it selects to it" if rs.crosses_to else f"{reader} selects the value")
        for g in follow.guarded_by:
            movers.setdefault(g.owner, "its write requires the value")
        self.say(f"selected by {follow.selected_by}; guarded by {[g.owner for g in follow.guarded_by]}")
        for owner, why in movers.items():
            self.say(f"  mover {owner}: {why}")
        gaps = {**follow.reader_gaps, **{g.owner: g.gaps for g in follow.guarded_by if g.gaps}}
        self.say(f"where the map could not read them: {gaps or 'nowhere'}" + (f" ({follow.gaps_note})" if follow.gaps_note else ""))
        writers = asyncio.run(self._writers_of(q.subject))
        chosen = []
        for owner in movers:
            arms: dict[int, Arm] = {}
            files: list[str] = []
            for j in writers:
                if j.owner != owner:
                    continue
                files += j.log_files
                for b in j.branches:
                    # an arm moves the value on unless it can only write the value observed
                    if b.line is None or b.outcome == q.observed:
                        continue
                    arm = arms.setdefault(b.line, Arm(b.line, [], list(b.path_condition)))
                    arm.outcomes.append(b.outcome)
            files = self.files_of(owner, list(dict.fromkeys(files or follow.reader_log_files)), list(arms))
            to_hand = [f for f in files if f in self.logs]
            if to_hand:
                self.warn_split(owner, list(arms))
            self.say(f"  {owner.split('::')[-1]}: {len(arms)} arm(s) that write another value "
                     f"{[(a.line, a.outcomes) for a in arms.values()]}; logs {files} ({'to hand: ' + str(to_hand) if to_hand else 'none to hand'})")
            if to_hand and any(runs_for(self.logs[f], q.key) for f in to_hand):
                chosen.append(Passes(owner, to_hand[0], list(arms.values()), q.key))
        self.need("one mover with a run of the key in a log to hand", len(chosen) == 1)
        return chosen[0]

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
        fn = R.function(rel, func, [a.line for a in q.arms])
        log = self.logs[q.log]
        found = []
        for run in runs_for(log, q.key):
            parts, how = R.passes(fn, run, self.mode)
            self.say(f"{run.describe()}: {how}")
            found += parts
        states = self.data.get("statuslog", {}).get("states", [])
        since = next((s["time"] for s in states if s["status"] == self.args.observed), None)
        if since is None:
            first, last = log.covers
            self.say(f"window: the log's coverage {first} .. {last} (the statuslog has no row for {self.args.observed!r})")
        else:
            self.say(f"window: from {since}, when the statuslog shows {self.args.observed!r}")
            found = [p for p in found if p.last[:19] >= since]
        self.need("a pass of the mover in the window", found)
        exits = sorted({n.lineno for n in R.own_nodes(fn.node) if isinstance(n, R.JUMPS)})
        lines = sorted({a.line for a in q.arms} | set(exits))
        self.say(f"candidates: arms {[a.line for a in q.arms]}, exits {exits}")
        chosen, envs = [], []
        for n, p in enumerate(found, 1):
            env = self.base_env()
            stable_holes([R.seen(fn, p, self.mode)], env, class_of(fn), local_to=fn.owner)
            self.say(f"-- pass {n}: {p.first} .. {p.last}, {len(p.records)} record(s); printed {self._printed(env)}")
            j = judge([Candidate(rel, func, lines, run=p)], {log.name: [p]}, mode=self.mode, decide=self.decide, question_run=p, env=env)
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
        ids = sorted({str(r["pandaid"]) for r in jobs})
        finals = {pid: min((r["t_final"] for r in jobs if str(r["pandaid"]) == pid and r.get("t_final")), default=None) for pid in ids}
        by_table: dict[str, int] = {}
        for r in self.data.get("counts", {}).get("rows", []):
            by_table[r["table"]] = by_table.get(r["table"], 0) + r["n"]
        # reading 1 and 2 are of the tables this call reads (proxy_reading), not of every table the statement can
        read = {t.lower() for t in at_call}
        in_call = {t: n for t, n in by_table.items() if t.lower() in read}
        if by_table:
            at_least = max(in_call.values(), default=None)
        else:
            at_least = len(ids) if at_call == statement else None

        def state_at(pid: str, when: str) -> str | None:
            before = [h for h in history.get(pid, []) if h["time_extended"] < when]
            return before[-1]["status"] if before else None

        if at_least is not None:
            self.say(f"reading 1, the map's tables at this call: every job of the key -- at least {at_least} (one table alone, {in_call or 'the job rows'})")
        else:
            self.say(f"reading 1, the map's tables at this call: every job of the key -- no bound: the rows file holds no row "
                     f"of the key in {at_call} ({by_table or 'no counts'})")
        uncovered = [t for t in at_call if t.lower() not in {a.lower() for a in ACTIVE_TABLES}]
        if not uncovered:
            self.say(f"reading 2, the call's tables {at_call} hold the jobs not yet in {list(FINAL_STATES)} -- "
                     f"an assumption about where PanDA keeps a row (not the map's)")
        else:
            self.say(f"reading 2, an assumption about where PanDA keeps a row (not the map's): the jobs not yet in "
                     f"{list(FINAL_STATES)} -- of {list(ACTIVE_TABLES)} only; this call also reads {uncovered}")
        fits = {1: at_least is not None, 2: not uncovered}
        active_by_pass = []
        for n, p in enumerate(q.passes, 1):
            when = p.first.replace(",", ".") + "000"
            states = {pid: state_at(pid, when) for pid in ids}
            active = [pid for pid in ids if states[pid] is not None and states[pid] not in FINAL_STATES]
            active_by_pass.append((p, when, active, states))
            printed = self._printed_in(fn, p, f"len({q.name})")
            value = int(re.match(r"'(\d+)'", printed[0])[1]) if printed else None
            fits[1] &= value is not None and at_least is not None and at_least <= value
            fits[2] &= value == len(active)
            groups = Counter(states[pid] for pid in active)
            one = f">= {at_least}" if at_least is not None else "none"
            self.say(f"  pass {n} at {when}: printed {value}; reading 1 {one}; reading 2 {len(active)} {dict(groups)}")
        self.say(f"readings that give every printed count: {[k for k, v in fits.items() if v] or 'none'}")
        self.need("one reading of the tables that gives every count the passes printed", [k for k, v in fits.items() if v] == [2])
        last_pass, _, active, states = active_by_pass[-1]
        for status, _members in sorted(Counter(states[pid] for pid in active).items()):
            pids = [pid for pid in active if states[pid] == status]
            self.say(f"  group at the last pass: {status} {len(pids)}: {pids[:6]}{' ...' if len(pids) > 6 else ''}")
        # criterion 4: did the value move later, and had the input emptied by then?
        # the arm writes one of its outcomes per command; a row after the last pass is the move
        after = last_pass.first[:19]
        moved = next((st for st in self.data.get("statuslog", {}).get("states", []) if st["status"] in q.outcomes and st["time"] > after), None)
        if moved is None:
            self.need(f"a statuslog row after {after} with what the kept-off arm writes {q.outcomes}", None)
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
        self.say(f"END a legitimate wait: {q.name} emptied at {last} and the value moved "
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
        w = self._which_wrote(q.subject, after["status"], q.key, after["time"])
        owner = w.reading.owner
        self.say(f"moved on by {owner}:{w.chosen.line}, in {w.question.describe()}")
        return Missed(q.subject, q.observed, after["status"], since, moved, statements, owner, w.question)

    def missed(self, q: Missed):
        self.start("L2", "why the others passed it over")
        cls = q.subject.split(".")[0]
        self.need(f"the rows file's row of {cls} (its \"classes\" say which row is which class)", self._row_of(cls))
        self.say(f"{len(q.statements)} statement(s) ask rows on {q.subject}={q.observed}, from {q.since} to {q.moved}")
        ordered = []
        for st in q.statements:
            # the call that moved the value on returned the row: a term the row fails was not in that build
            took = st.owner == q.mover
            kind, said = self._passed_over(st, q, took)
            if took:
                kind = f"took it at {q.moved}"
            rows_to = ", ".join(c.split("::")[-1] for c in st.crosses_to) or "itself"
            self.say(f"  {st.owner.split('::')[-1]}@{st.line}: {kind} (rows to {rows_to})")
            for line in said:
                self.say(f"      {line}")
            if (kind in ("not settled", "refused on a call", "refused when") or took) and st.order_by and st.limits:
                self.say(f"      returns the first rows ({', '.join(st.limits)}) of {st.order_by}")
                ordered.append(st)
        mine = [st for st in ordered if st.owner == q.mover]
        self.need("the statement that moved the value on returns the first rows of an order", len(mine) == 1)
        return Ordered(mine[0], q.run, q)

    def _row_of(self, cls: str) -> dict:
        """The rows file's row of the class: the one its ``classes`` says is of it (A1 answers a row of a class)."""
        entity = next((e for e, c in self.data.get("classes", {}).items() if c == cls), None)
        return self.data.get("rows", {}).get(entity, {}) if entity else {}

    def _row_value(self, subject: str):
        """The rows file's value of ``Class.attr`` for this instance, or None."""
        cls, _, attr = subject.partition(".")
        row = self._row_of(cls)
        return row[attr]["value"] if attr in row else None

    def _row_holds(self, pair: str) -> bool:
        subject, _, value = pair.partition("=")
        held = self._row_value(subject)
        return held is not None and str(held) == value

    def _held(self, subject: str, q: Missed) -> tuple[bool, object]:
        """``(known, value)`` of *subject* for the row over the window: the asked subject holds the observed value;
        another, what the rows file holds (``None`` there is SQL's NULL); a column it lacks is not known."""
        if subject == q.subject:
            return True, q.observed
        cls, _, attr = subject.partition(".")
        row = self._row_of(cls)
        return (True, row[attr]["value"]) if attr in row else (False, None)

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
        """Why the row's values cannot answer *subject*: no row of its owner in the rows file and no history it reads,
        a log column the history by key does not carry, or a column the owner's row lacks -- kept under the rows
        file's ``after`` (the moved row's columns written at the move or later, so not what the statement read before
        it) or not fetched at all; None when they can.  Said, not
        skipped in silence."""
        owner, _, attr = subject.partition(".")
        if subject == q.subject:
            return None
        row = self._row_of(owner)
        if row:
            if attr in row:
                return None
            if owner == q.subject.partition(".")[0] and attr in self.data.get("after", {}):
                return "written at the move or later; its row here holds only the value after"
            return f"its row here holds no {attr}"
        found = PT.history_by_table(owner)
        if found is None or found.subject.partition(".")[0] != q.subject.partition(".")[0]:
            return f"no row of {owner}"
        return None if PT.history_column(found, attr) else f"its status history here holds no {attr}"

    def _history_leaf(self, t, rows: list, q: Missed) -> str | None:
        """A term on a joined log, false all the window: no row of the history satisfies it at any moment it exists in
        the window (the join passes where some row does)."""
        never = "no row of its status history before it moved on"
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
                return f"every row of its status history before it moved on holds {next(iter(values))!r}"
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
            return f"no row of its status history before it moved on holds one ({sorted({str(v) for v, _ in rows})})"
        known, held = self._held(subject, q)
        return f"the row holds {held!r}" if known and held is not None and str(held) not in values else None

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
        settled offset: true only outside the window; a number: false for every value."""
        if t.node == "other" or not t.settled:
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
                return f"with {_value_words(v)}: true for the row from {turn}, after it moved on"
            if op in (">", ">=") and turn < q.since:
                return f"with {_value_words(v)}: true for the row until {turn}, before the window"
            return None
        a, b = _number(held), _number(v.constant)
        if v.kind == "column" or a is None or b is None:
            return None
        true = {"<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b}.get(op)
        return f"with {_value_words(v)}: the row holds {held!r}" if true is False else None

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
            if self._row_holds(pair):
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
            if self._row_holds(pair):
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
                if self._row_holds(pair):
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
        subjects = {p.partition("=")[0] for p in [*st.selects, *st.excludes, *st.refuses, *st.restricts]}
        subjects |= {leaf for r in st.condition_trees for leaf in _subjects_of(r.tree)}
        unanswered = {x: why for x in sorted(subjects) if (why := self._unanswered(x, q))}
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
        self.say(f"END lost on order: {where} returns the first rows of {other!r} or of {took!r}; under the first, "
                 f"{n_ahead} row(s){mine} at the call's WHERE sort ahead of it now; its group left {miss.observed!r}: {left} "
                 f"(this row to {miss.next_value!r}); the call that took it used {took!r}")
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--observed", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--log", action="append", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--hand", default="")
    ap.add_argument("--patterns", default="outside", choices=R.PATTERN_MODES)
    ap.add_argument("--decide", default="positive", choices=("positive", "strict"))
    ap.add_argument("--out", default="")
    ap.add_argument("--stuck", action="store_true", help="ask why nothing moved the observed value on")
    ap.add_argument("--late", action="store_true", help="ask why the observed value was moved on so late")
    args = ap.parse_args()
    r = Rerun(args)
    try:
        r.run()
        r.say()
        r.say(f"== reached the end; HAND used {len(r.hand_used)} time(s) {r.hand_used}")
    except Stop as stop:
        r.say()
        r.say(f"STOP {stop}; HAND used {len(r.hand_used)} time(s) {r.hand_used}")
    r.stamp("end")
    if args.out:
        Path(args.out).write_text("\n".join(r.lines) + "\n")


if __name__ == "__main__":
    main()
