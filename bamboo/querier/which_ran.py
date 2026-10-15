"""Step 2 of the loop: which of the candidate sites ran, from one run's records.

Every candidate comes out as one of four, with the rule that put it there:

* **excluded** --
  a row every path through the site prints is missing from the run (rule 4;
  only when the run printed something at that level, rule 8);
  a row that path can no longer reach after its ``return`` is in the run;
  a printed row's ``arms`` leave the site out (outside a shared loop);
  the site sits after the record that already showed the observed value, or
  before the one that still showed the old value (rule 3);
  the site is reached only when the value is already the observed one
  (rule 6, solved from the guards on the way to it, the caller's included);
* **positive** -- every row that must print does, one of them says enough to
  prove it, and a hole of the subject in them holds the observed value; or,
  that failing, the rows printed around the ``if`` chain it sits under leave no
  other way through (P, :mod:`bamboo.querier.printed`);
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

    def count(self, kind: str) -> int:
        return sum(1 for v in self.verdicts if v.kind == kind)

    def report(self, run: Run | None = None) -> str:
        out = []
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


def rewriter(fn: R.Function, stmt: ast.stmt, attr: str, observed: str, env: E.Env) -> str:
    """Why *stmt* can only be reached once ``<x>.<attr>`` already is *observed*, or ''."""
    guards = [(g, f"{fn.name}:{stmt.lineno}") for g in pathcond.path_condition(stmt)] + callers_guards(fn)
    for guard, where in guards:
        try:
            solved = E.solve(guard, True, env)
        except SyntaxError:
            continue
        for atom, truth in solved.settled.items():
            m = re.fullmatch(rf"([\w.]+\.{re.escape(attr)}) (==|!=) ['\"]{re.escape(observed)}['\"]", atom)
            if m and ((m[2] == "==" and truth) or (m[2] == "!=" and not truth)):
                used = {k: v for k, v in env.used.items() if k in guard}
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
    raised_at: dict[str, set[int]] | None = None,
    env: E.Env | None = None,
    complete: bool = True,
) -> Judgement:
    raised_at = raised_at or {}
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
            # each log's reason replaces the one before (KNOWN_ISSUES.md K12)
            for name in to_hand:
                found = [r for r in logs[name] if window is None or (r.first[:19] <= window[1] and r.last[:19] >= window[0])]
                if found:
                    run = run_holding(c, found, mode)
                    if run is not None:
                        break
                    why_not[c.owner] = f"none of the key's {len(found)} runs in {name} at {window[0] if window else 'any time'} holds a row of it"
                    continue
                covering = logs[name][0].log if logs[name] else None
                if covering is not None and covering.records[0].machine:
                    # a recorded search says what it searched by machine (Log.holes), not by its first and last record
                    why_not[c.owner] = f"{name}: no run of the key at {window}" + (f"; not searched: {'; '.join(covering.holes)}" if covering.holes else "")
                elif covering is not None and window is not None and not (
                    covering.covers[0][:19] <= window[1] and covering.covers[1][:19] >= window[0]
                ):
                    why_not[c.owner] = f"{name} covers {covering.covers[0]}..{covering.covers[1]}, not {window[0]}..{window[1]}"
                else:
                    why_not[c.owner] = f"{name}: no run of the key at {window}"
        runs_of[c.owner] = run
        if run is not None:
            seens[c.owner] = R.seen(R.function(c.rel, c.func, c.lines), run, mode)
    # a record counts as printed only by the rows that read it best (rows.claim)
    seens = dict(zip(seens, R.claim(list(seens.values())), strict=False))

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
        for line in c.lines:
            stmt = R.statement_at(fn.node, line)
            v = Verdict(c, line, "kept")
            if stmt is None:
                v.reasons.append("no statement on this line")
                verdicts.append(v)
                continue
            before, after = R.must_rows(fn.node, stmt, raised_at.get(c.owner, set()))
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
                    if trust[1] not in ("", "unreadable"):
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
            if subject and observed and env is not None:
                own = E.Env(dict(env.values), dict(env.rows), env.table, dict(env.stale))
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
                else:
                    v.reasons.append(
                        f"must rows: printed {shown}, unreadable {neutral}; none prints only through it"
                        if must
                        else "no row every path through it prints"
                    )
            if raised_at.get(c.owner):
                v.reasons.append(f"a traceback frame raised at {sorted(raised_at[c.owner])}")
            where = f"{Path(c.rel).name}::{c.func}:{line}"
            if proof.kind == "positive" and v.kind == "excluded":
                p_notes.append(f"P twin: {where} excluded ({'; '.join(v.reasons)}), but {proof.reason}")
            elif proof.kind == "inconsistent":
                p_notes.append(f"P inconsistent: {where}: {proof.reason}")
            verdicts.append(v)

    positives = [v for v in verdicts if v.kind == "positive"]
    left = [v for v in verdicts if v.kind in ("positive", "kept", "unexamined")]
    unexamined = [v for v in verdicts if v.kind == "unexamined"]
    in_run = [v for v in verdicts if v.kind in ("positive", "kept") and runs_of[v.candidate.owner] is question_run]
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
    return Judgement(verdicts, chosen, how, bracket, [f"MAP: {m}" for m in sorted(map_findings)] + p_notes, runs_of)

