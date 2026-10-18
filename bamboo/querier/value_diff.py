"""Norm 1: find the guard that says the same thing and came out the other way, then diff by value.

    (a library for :mod:`bamboo.querier.rerun`)

**Finding the pair.**  An atom settled for this instance (``tmp_src_ddm ==
tmp_dst_ddm`` true at setupper 805) is looked for, by its text, in the guards
of every function the investigation has a run for.  Each occurrence is
evaluated with the holes that function printed above it in that run -- a
hole counts only when nothing binds its name again between the print and
the guard.  An occurrence that comes out the other way is the contradiction.

**Diffing by value, not by derivation.**  The structural diff of the two
walks reports every leaf one side has and the
other lacks, and the map's walks put false leaves on both sides (a
SiteMapper stop drops the site key on one side only; the loop's own
context rides along).  Here each input of the predicate is given its value
on both sides first.  **An input with the same value on both sides is not
where they part.**  For the ones that differ, the binding that took effect on
each side is chosen with the instance's values, and only those two are
shown, with the stop and the leaves each ends in.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

from bamboo.codemap import trace  # noqa: E402
from bamboo.codemap.panda.provenance import classify  # noqa: E402
from bamboo.querier import evaluate as E  # noqa: E402
from bamboo.querier import rows as R  # noqa: E402
from bamboo.querier.records import Run  # noqa: E402

FIELD = re.compile(r"\b((?:self\.)?[a-z_]+)\.([A-Za-z_]+)\b(?!\()")


def local_env(fn: R.Function, run: Run, mode: str, line: int, base: E.Env) -> E.Env:
    """*base* plus the holes *fn* printed above *line* in *run*, unbound since."""
    env = E.Env(dict(base.values), dict(base.rows), base.table, dict(base.stale), unread=base.unread)
    sn = R.seen(fn, run, mode)
    for row in fn.rows:
        if row.line >= line:
            continue
        for i, _, _ in sn.hits.get(row.line, ()):
            for expr, value in row.capture(run.records[i]).items():
                rebound = [b.lineno for b in R._binding_sites(fn.node, expr, line) if b.lineno > row.line]
                if rebound:
                    continue
                env.put_printed(expr, value.strip(), f"hole of {fn.name}:{row.line} at {run.records[i].span}")
    return env


@dataclass
class Occurrence:
    fn: R.Function
    run: Run
    line: int  # the if/while whose test holds the atom
    value: object
    env: E.Env


def occurrences(atom: str, functions: list[tuple[R.Function, Run]], mode: str, base: E.Env) -> list[Occurrence]:
    out = []
    for fn, run in functions:
        for node in R.own_nodes(fn.node):
            if not isinstance(node, (ast.If, ast.While)):
                continue
            if not any(ast.unparse(a) == atom for a in E.atoms(node.test)):
                continue
            env = local_env(fn, run, mode, node.lineno, base)
            out.append(Occurrence(fn, run, node.lineno, E.evaluate(atom, env), env))
    return out


@dataclass
class Side:
    label: str
    steps: list
    env: E.Env
    owner: str  # ``module::function`` whose names the steps bind
    values: dict[str, tuple[object, str]] = field(default_factory=dict)


def names_of(atom: str) -> list[str]:
    tree = ast.parse(atom, mode="eval").body
    return [ast.unparse(n) for n in ([tree.left, *tree.comparators] if isinstance(tree, ast.Compare) else [tree])]


def walk_steps(fn: R.Function, line: int) -> list:
    return trace.walk(R.ROOTS, file=fn.rel, owner=fn.owner, lines=[line], classify=classify, reaching=False).steps


def value_of(name: str, side: Side) -> tuple[object, str]:
    if name in side.values:
        return side.values[name]
    v = side.env.lookup(name)
    if v is not E.UNKNOWN:
        return v, side.env.used.get(name, "env")
    t = E.taken(side.steps, name, side.env, owner=side.owner)
    if t.step is not None and t.value is not E.UNKNOWN:
        return t.value, f"binding @{t.step.line} {t.step.value}"
    return E.UNKNOWN, t.describe()


def leaves(step, env: E.Env) -> list[str]:
    out = []
    for obj, attr in FIELD.findall(step.value):
        text = f"{obj}.{attr}"
        v = env.lookup(text)
        out.append(f"{text} = {v!r} ({env.used.get(text, 'unknown')})" if v is not E.UNKNOWN else f"{text} (unknown)")
    return out


def diff(atom: str, a: Side, b: Side) -> list[str]:
    lines = [f"atom {atom!r}"]
    for name in names_of(atom):
        va, sa = value_of(name, a)
        vb, sb = value_of(name, b)
        lines.append(f"  {name}: {a.label} {va!r} [{sa}]  |  {b.label} {vb!r} [{sb}]")
        if va is E.UNKNOWN or vb is E.UNKNOWN:
            lines.append("    one side unknown: cannot compare by value")
            continue
        if va == vb:
            lines.append("    same value on both sides: not where they part")
            continue
        lines.append("    DIFFERS -- the bindings that took effect:")
        for side in (a, b):
            t = E.taken(side.steps, name, side.env, owner=side.owner)
            if t.step is None:
                lines.append(f"      {side.label}: {t.describe()}")
                continue
            s = t.step
            stop = f"  STOP {s.terminal}: {s.detail}" if s.terminal else ""
            lines.append(f"      {side.label}: @{s.line} {name} {s.operator} {s.value}{stop}")
            for leaf in leaves(s, side.env):
                lines.append(f"        leaf {leaf}")
            for step, guard in t.dropped:
                lines.append(f"        not @{step.line} ({step.value}): guard {guard!r} is false")
    return lines
