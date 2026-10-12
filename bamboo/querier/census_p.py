"""Which status arms the P rules could prove from the code alone (:func:`bamboo.querier.printed.census`).

    PANDA_PIN=<pin> PYTHONPATH=<tree> python -m bamboo.querier.census_p <status_arms list.json> [--hand SAMPLE] [--only REL::FUNC:LINE ...]
                                                          [--calls-change-nothing] [--call-values-known] [--no-render-rows]
                                                          [--no-handler-rule] [--jobs N] [--record FILE] [--out FILE]

The question per arm: if it ran, would the rows its path prints, and their
holes, rule out every other way through some S* (printed.py)?  ``positive``
-- yes for at least one way the arm can be reached (``proven/arm paths``
says for how many of them); ``soft`` -- only a call raising is left;
``none`` / ``no anchor`` / ``too many`` / ``not under an if`` -- no.

``--hand`` reads the class letters of a hand-read sample of the same list
and sets them beside each arm,
so the rule can be scored against the reading.

``--calls-change-nothing`` takes every call to bind nothing it is handed
(``printed.mutating_roots`` empty: no hole, fact or target value is lost at a
call; a call may still raise).  It is the most following a call to the writers it reaches
could prove, since that can only remove such losses.

``--call-values-known`` decides, the way the arm's path took it, a test that
reads a call's value -- a call in the test itself, or a local every binding
of which in the function is a call's result -- as if the callee's rows had
told it (``parse_job_output_report end`` before ``return 0``).  Also an
upper bound: it does not ask whether the callee prints such a row.

``--no-handler-rule`` turns off E6 (``printed.HANDLER_RULE``, on by default): an arm path that leaves by an exception whose handler prints a
traceback at a line after the arm is proven by that frame; one whose handler
prints any other row of the run, and a soft path whose missing paths all do,
is ``by handler`` -- the log decides it, and it does not keep census looking
at outer ifs.

``--no-render-rows`` turns off E5 (``printed.ROW_RENDER``, on by default): a path whose own rendering of a row cannot be the arm path's is
dropped.
"""
from __future__ import annotations

import argparse

# No map store is read here: run with every graph client refused, all 184 arms of a status-arm list gave the
# same tally and 0 connections.
# A census depends on BAMBOO_TREE and PANDA_PIN only, so neither MAP_DUMP nor the stored map's commit moves it.
import ast  # noqa: E402
import collections
import functools  # noqa: E402
import json
import re
import sys
import time
from pathlib import Path

from bamboo.querier import handlers as H  # noqa: E402
from bamboo.querier import printed as P  # noqa: E402
from bamboo.querier import rows as R  # noqa: E402
from bamboo.querier.records import CORPUS  # noqa: E402


@functools.lru_cache(maxsize=None)
def _call_bound(func: ast.AST) -> frozenset[str]:
    """Locals of *func* every binding of which is the result of a call (not a reader builtin, not a logger)."""
    seen: dict[str, bool] = {}
    for n in R.own_nodes(func):
        for t, _vnode in P.binds(n):
            if not t.isidentifier():
                continue
            ok = isinstance(n, ast.Assign) and isinstance(n.value, ast.Call) and _is_value_call(n.value)
            seen[t] = seen.get(t, True) and ok
    return frozenset(t for t, ok in seen.items() if ok)


def _is_value_call(node: ast.Call) -> bool:
    return not R.is_log_call(node) and not (isinstance(node.func, ast.Name) and node.func.id in P.READERS)


def _covered_with_calls(node: ast.AST, avail: set[str], state: dict) -> bool:
    """printed.covered, and a call's value counts as known."""
    if isinstance(node, ast.Call) and _is_value_call(node):
        return True
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return all(_covered_with_calls(e, avail, state) for e in node.elts)
    if isinstance(node, ast.BoolOp):
        return all(_covered_with_calls(v, avail, state) for v in node.values)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return _covered_with_calls(node.operand, avail, state)
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        return _covered_with_calls(node.left, avail, state) and _covered_with_calls(node.comparators[0], avail, state)
    return _real_covered(node, avail, state)


_real_covered = P.covered


def _containing(fns: list, line: int):
    for fn in fns:
        if fn.lineno <= line <= (fn.end_lineno or fn.lineno):
            return fn
    return None


def hand_classes(path: str) -> dict[tuple[str, int, str], str]:
    out = {}
    for raw in Path(path).read_text().splitlines():
        m = re.match(r"\s*(\d+)\s+(\S+)\s+(\S+\.py):(\d+) (\w+)", raw)
        if m:
            out[(m[3], int(m[4]), m[5])] = f"#{m[1]} {m[2]}"
    return out


def _apply_flags(
    no_render_rows: bool, calls_change_nothing: bool, call_values_known: bool, no_handler_rule: bool = False, no_reuse: bool = False
) -> None:
    """The run's switches, in this process (and in each worker of ``--jobs``)."""
    if no_reuse:
        P.REUSE_WALKS = False
    if no_render_rows:
        P.ROW_RENDER = False
    if no_handler_rule:
        P.HANDLER_RULE = False
    if calls_change_nothing:
        P.mutating_roots = lambda node: set()
    if call_values_known:
        real_available = P.Walker.available
        P.Walker.available = lambda w, p: real_available(w, p) | _call_bound(w.func)
        P.covered = _covered_with_calls


LABELS = (
    "positive every", "positive some", "by handler every", "by handler some",
    "soft", "inconsistent", "none", "no anchor", "too many", "not under an if",
)


def label(res: P.Census) -> str:
    """The arm's kind -- its best arm path's -- and, for positive and by handler, whether every arm path has it.

    A run takes one way through the arm, and which is not known before the log
    is read: ``positive some`` says the log proves the arm only on some of them.
    """
    if res.kind in ("positive", "by handler"):
        return f"{res.kind} {'every' if res.every else 'some'}"
    return res.kind


def _one(arm: dict, tag: str, record: bool) -> dict:
    """One arm: its output line, its tally key and, with *record*, every S* tried."""
    fn = _containing(R.functions(arm["rel"], arm["func"]), arm["line"])
    stmt = R.statement_at(fn, arm["line"]) if fn is not None else None
    levels: list | None = [] if record else None
    started = time.time()
    if stmt is None:
        res = P.Census("not under an if", note="statement not found")
    else:
        res = P.census(fn, stmt, levels)
    took = time.time() - started
    silent = "S" if arm["silent"] else " "
    kind = label(res)
    mech = ",".join(sorted(set(res.cuts))) if res.cuts else ""
    drops = ",".join(f"{k}:{v}" for k, v in sorted(res.dropped.items()))
    row = (
        f"{silent} {kind:16s} {Path(arm['rel']).name}:{arm['line']} {arm['func']}"
        f"  S*={res.s_star} {res.proven_paths}/{res.arm_paths}  [{res.note}] cut[{mech}] drop[{drops}]"
        f"{'  ' + tag if tag else ''}{f'  ({took:.1f}s)' if took > 5 else ''}"
    )
    rec = None
    if record:
        for level in levels or []:
            for p in level["paths"]:
                if "left" in p:
                    cls, line = H.leave_class(fn, p["left"])
                    p["class"] = cls
                    p["after_arm"] = bool(line) and cls != "NOTEXC" and H.after_arm(fn, stmt, line)
                if p.get("missing"):
                    p["missing"] = [[left, H.leave_class(fn, left)[0]] for left in p["missing"]]
        rec = {
            "rel": arm["rel"], "func": arm["func"], "line": arm["line"], "silent": arm["silent"], "kind": res.kind,
            "every": res.every, "s_star": res.s_star, "proven": res.proven_paths, "arm_paths": res.arm_paths, "seconds": round(took, 2),
            "levels": levels or [],
        }
    return {"row": row, "kind": kind, "silent": arm["silent"], "tag": tag, "record": rec, "took": took}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("arms")
    ap.add_argument("--hand")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--out")
    ap.add_argument("--record", help="one JSON line per arm: every S* tried, each sampled arm path, the walks' seconds")
    ap.add_argument("--jobs", type=int, default=1, help="arms walked in this many processes at once")
    ap.add_argument("--calls-change-nothing", action="store_true")
    ap.add_argument("--call-values-known", action="store_true")
    ap.add_argument("--no-render-rows", action="store_true")
    ap.add_argument("--no-handler-rule", action="store_true")
    ap.add_argument("--no-reuse", action="store_true", help="walk every arm path again (printed.REUSE_WALKS off); the output must not move")
    a = ap.parse_args()
    flags = (a.no_render_rows, a.calls_change_nothing, a.call_values_known, a.no_handler_rule, a.no_reuse)
    _apply_flags(*flags)
    arms = json.loads(Path(a.arms).read_text())
    hand = hand_classes(a.hand) if a.hand else {}
    if a.only:
        wanted = {tuple(s.rsplit(":", 1)) for s in a.only}
        arms = [x for x in arms if (f"{x['rel']}::{x['func']}", str(x["line"])) in wanted]
    lines = [
        f"# census_p: {a.arms}, corpus {CORPUS}, bamboo {R.BAMBOO_TREE}"
        + (" -- calls change nothing (upper bound)" if a.calls_change_nothing else "")
        + (" -- call values known (upper bound)" if a.call_values_known else "")
        + ("" if P.ROW_RENDER else " -- E5 off")
        + ("" if P.HANDLER_RULE else " -- E6 off")
        + ("" if P.REUSE_WALKS else " -- reuse off")
        + ("" if P.CALLS_RAISE else " -- no call raises (P_CALLS_RAISE=0, upper bound)")
        + ("" if H.FOLLOW_OUT else " -- OUT not followed (P_FOLLOW_OUT=0)")
        + ("" if P.LEVEL_BY == "share" else f" -- S* chosen by {P.LEVEL_BY} (P_LEVEL_BY)")
        + ("" if P.WRITTEN_BACK else " -- an execute rewrites its whole map (P_WRITTEN_BACK=0)"),
        f"# {len(arms)} arm statement(s)",
    ]
    tags = [hand.get((Path(x["rel"]).name, x["line"], x["func"]), "") for x in arms]
    record = bool(a.record)
    started = time.time()
    results: list = [None] * len(arms)

    def done(i: int, r: dict) -> None:
        results[i] = r
        n = sum(x is not None for x in results)
        print(f"[{n}/{len(arms)}] {time.time() - started:6.0f}s  {r['row'][:110]}", file=sys.stderr, flush=True)

    if a.jobs > 1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor, as_completed

        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=a.jobs, mp_context=ctx, initializer=_apply_flags, initargs=flags) as ex:
            futures = {ex.submit(_one, arm, tag, record): i for i, (arm, tag) in enumerate(zip(arms, tags, strict=False))}
            for f in as_completed(futures):
                done(futures[f], f.result())
    else:
        for i, (arm, tag) in enumerate(zip(arms, tags, strict=False)):
            done(i, _one(arm, tag, record))
    tally: collections.Counter = collections.Counter()
    by_hand: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for r in results:
        tally[(r["kind"], r["silent"])] += 1
        if r["tag"]:
            by_hand[r["tag"].split()[1]][r["kind"]] += 1
    lines += [r["row"] for r in results]
    lines.append("")
    lines.append("== tally (kind: silent / not silent)")
    for kind in LABELS:
        s, n = tally[(kind, True)], tally[(kind, False)]
        if s or n:
            lines.append(f"  {kind:16s} {s:4d} / {n:4d}")
    lines.append(f"  {'all':16s} {sum(v for (k, s), v in tally.items() if s):4d} / {sum(v for (k, s), v in tally.items() if not s):4d}")
    if by_hand:
        lines.append("== against the hand classes")
        for cls in sorted(by_hand):
            lines.append(f"  {cls:4s} " + ", ".join(f"{k} {v}" for k, v in by_hand[cls].most_common()))
    text = "\n".join(lines) + "\n"
    if a.out:
        Path(a.out).write_text(text)
    if record:
        with open(a.record, "w") as out:
            for r in results:
                out.write(json.dumps(r["record"]) + "\n")
    print(text)


if __name__ == "__main__":
    main()
