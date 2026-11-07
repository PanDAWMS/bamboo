"""A guard read in a column the rows file holds, way by way (KNOWN_ISSUES.md K49; the analyzer handover's
out/tpm/q0/q0-step0.md, D1).

The map says what a name a test reads holds, way by way (``GuardFrom``, ``ParameterFrom``): ``'transPath' in
taskParamMap`` at ``TaskRefinerBase.extractCommon`` reads the task's parameters through four routes (TaskRefiner 211
straight, or each refiner's own extractCommon) to ``jedi_taskparams.taskparams``, the column the rows file holds
decoded as ``task_params``.  On each way it says what may change the object in place before the read (``changes``),
and after it, before a write gives the column back (``after``).  The rows file holds the column as fetched, after the
pass, so every answer rests on the column being read as the pass left it: no later write of it is read.

A test of a key's presence (``'K' in m``) is read in that value by the direction of each change.  A store, an inner
change or a ``*`` store adds a key or needs it there; a delete, a method, an escape or a call not read may remove one
(:data:`ADDS`, :data:`REMOVES`):

- the rows hold K: it was there at the read on a way, unless something may add K after the read on it, or remove it
  before;
- the rows lack K: it was absent, unless something may add K before the read, or remove it after.

Whether the pass wrote the column back is not read: either way these hold.  A test of a key's value (``m['K']``) is
read as the rows hold it where nothing changes K on the way, before or after.  A change counts unless a test on its
way reads false: the tests of each call of its ``via`` and of its ``at``, early returns included (``pathcond``), each
read by the map's GuardFrom of that test where there is one, the same way, :data:`DEPTH` deep.  A change after the read
is counted on every way of the guard, whichever route its *via* goes through: more counted, never fewer.

A way is one route.  The run shows a route by a line only that route's own functions print (the functions holding
its calls that not every route goes through): then it alone is read.  Else every route is read, and a key is decided
where they agree.  What it says is records (:mod:`render`'s ``held-*``), rendered where they are read.
"""
from __future__ import annotations

import ast
import functools
from dataclasses import dataclass, field

from bamboo.codemap.panda import pathcond
from bamboo.codemap.reading import containing_function
from bamboo.querier import evaluate as E
from bamboo.querier import mapdump
from bamboo.querier import render as T
from bamboo.querier import rows as R

#: the columns the rows file holds decoded, and the section holding each (fieldtest 5b6d2e9)
HELD = {"jedi_taskparams.taskparams": "task_params"}
#: how a change (``HeldChange.kind``) may move a key's presence: may add it, may remove it
ADDS = ("store", "method", "escaped", "unread")
REMOVES = ("delete", "method", "escaped", "unread")
#: a stop of a way that is no way: the read raises there (``''`` and None through ``json.loads``, the map's VK)
NO_WAY = ("raises",)
#: a write the changes after the read were not read for, that is not this pass's: it and the read never meet
NOT_THIS_PASS = ("no_meeting",)
#: how deep a change's conditions are read through the map's GuardFroms of the tests on its way
DEPTH = 2


def _short(at: str) -> str:
    return at.rsplit("/", 1)[-1]


@dataclass
class Route:
    """One way of a test's name to its end: the calls it goes through (*calls*, innermost first), those not every
    route goes through (*mine*) and the functions holding them (*own*), what may change the object before the read and
    after it, and why the way is not read (*open*, a record, or None)."""

    calls: list[str]
    column: str
    before: list
    after: list
    open: T.Said | None = None
    mine: tuple[str, ...] = ()
    own: tuple[str, ...] = ()

    @property
    def label(self) -> str:
        return ", ".join(map(_short, self.mine)) if self.mine else _short(self.calls[0]) if self.calls else "(no call)"


@dataclass
class Reading:
    """A guard read in the held column: True, False or ``E.UNKNOWN``; each key's reading, routes alike grouped
    (``keys``, records); the column's section and when it was fetched; the route the run showed (or "")."""

    value: object
    keys: list[T.Said] = field(default_factory=list)
    section: str = ""
    as_of: str = ""
    shown: str = ""

    def premise(self, guards: list[str]) -> T.Said:
        """What an answer from it rests on (K49)."""
        return T.Said("held-column-read", {
            "guards": list(guards), "section": self.section, "as_of": self.as_of,
            "shown": T.Said("held-route-shown", {"route": self.shown}) if self.shown else "",
        })


@functools.cache
def _index(fragment_id: int) -> tuple[dict, dict]:
    fragment = mapdump.fragment()
    guards: dict[str, list] = {}
    for g in getattr(fragment, "guard_froms", None) or []:
        guards.setdefault(g.owner, []).append(g)
    params = {(p.owner, p.parameter): p for p in getattr(fragment, "parameter_froms", None) or []}
    return guards, params


def _maps() -> tuple[dict, dict]:
    fragment = mapdump.fragment()
    return _index(id(fragment)) if fragment is not None else ({}, {})


def _key(change) -> str:
    """The item a change names, as a str (``'transPath'`` -> ``transPath``), ``*``, or an attribute as written."""
    key = str(change.key)
    if key[:1] in "'\"":
        try:
            return str(ast.literal_eval(key))
        except (ValueError, SyntaxError):
            return key
    return key


def _change(c) -> T.Said:
    return T.Said("held-change", {"kind": c.kind, "key": c.key, "at": _short(c.at)})


def _routes(g) -> list[Route]:
    """Each way of *g*'s name to its end, a ParameterFrom chain followed through; its changes before the read are every
    hop's, its changes after are the guard's way's."""
    _, params = _maps()
    out: list[Route] = []

    def follow(s, calls: list[str], before: list, after: list, opened: T.Said | None, seen: frozenset) -> None:
        calls = calls + list(reversed(s.route))
        before = before + list(s.changes)
        if any(step for step in s.steps):
            out.append(Route(calls, s.column, before, after, T.Said("held-way-part", {"steps": list(s.steps)})))
        elif s.kind == "parameter" and s.site_owner:
            pf = params.get((s.site_owner, s.name))
            if pf is None or (s.site_owner, s.name) in seen:
                out.append(Route(calls, "", before, after, T.Said("held-way-handed", {"owner": s.site_owner, "name": s.name})))
                return
            for stop in pf.stops:
                if stop.kind not in NO_WAY:
                    out.append(Route(calls, "", before, after, T.Said("held-way-stopped", {"kind": stop.kind})))
            for t in pf.sources:
                follow(t, calls, before, after, opened, seen | {(s.site_owner, s.name)})
        elif s.kind == "rows" and s.column in HELD:
            out.append(Route(calls, s.column, before, after, opened))
        else:
            out.append(Route(calls, s.column, before, after, T.Said("held-way-ends", {"kind": s.kind, "name": s.name})))

    for s in g.sources:
        stopped = [x.kind for x in s.after_stops if x.kind not in NOT_THIS_PASS]
        opened = T.Said("held-way-after-not-read", {"stops": stopped}) if stopped else None
        follow(s, [], [], list(s.after), opened, frozenset())
    shared = set.intersection(*(set(r.calls) for r in out)) if out else set()
    for r in out:
        r.mine = tuple(dict.fromkeys(c for c in r.calls if c not in shared))
        r.own = tuple(dict.fromkeys(_owner_at(c) for c in r.mine))
    return out


@functools.cache
def _tree(rel: str) -> ast.Module | None:
    try:
        return R.tree(rel)
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None


def _place(at: str) -> tuple[str, ast.AST, ast.stmt] | None:
    """``file:line`` as (the owner of the function holding it, the function, the statement)."""
    rel, _, line = at.rpartition(":")
    tree = _tree(rel)
    func = containing_function(tree, int(line), "") if tree is not None else None
    if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    node = R.statement_at(func, int(line))
    if node is None:
        return None
    return f"{rel}::{func.name}", func, node


def _owner_at(at: str) -> str:
    placed = _place(at)
    return placed[0] if placed else at


def _shown(routes: list[Route], run, mode: str) -> Route | None:
    """The route a record of *run* shows: one its own functions print and no other route's own functions do."""
    if run is None:
        return None
    hits: dict[int, set[int]] = {}
    for n, r in enumerate(routes):
        for owner in r.own:
            rel, _, name = owner.partition("::")
            if not R.functions(rel, name):
                continue
            for found in R.seen(R.function(rel, name), run, mode).hits.values():
                hits.setdefault(n, set()).update(i for i, _, _ in found)
    shown = [n for n, mine in hits.items() if mine - set().union(*(o for m, o in hits.items() if m != n))]
    return routes[shown[0]] if len(shown) == 1 else None


class _Reader:
    def __init__(self, data: dict, run, mode: str, env: object):
        self.data, self.run, self.mode = data, run, mode
        self.env = env if isinstance(env, E.Env) else E.Env()

    def held(self, column: str) -> tuple[dict | None, str]:
        section = self.data.get(HELD.get(column, ""))
        if not isinstance(section, dict) or not isinstance(section.get("value"), dict):
            return None, ""
        return section["value"], section.get("as_of", "")

    def ruled_out(self, change, depth: int) -> T.Said | None:
        """The test on *change*'s way read false, as a record, or None."""
        if depth >= DEPTH:
            return None
        for at in [*change.via, change.at]:
            placed = _place(at)
            if placed is None:
                continue
            owner, func, node = placed
            # pathcond's own reading of the tests deciding a node, with the arm each keeps it on (private there;
            # ruled_out_by reads parameters' values, not a test read otherwise)
            for test, keeps in pathcond._tests_on(node, func):
                got = self.test(owner, ast.unparse(test), test.lineno, None, depth + 1)
                if got is not None and got.value in (True, False) and got.value is not keeps:
                    return T.Said("held-change-did-not-run", {
                        "change": _change(change), "test": ast.unparse(test), "at": f"{_short(owner.partition('::')[0])}:{test.lineno}",
                        "read": got.value})
        return None

    def key(self, r: Route, key: str, presence: bool, depth: int) -> tuple[object, T.Said]:
        """*key*'s presence (or value) at the read on *r*, and why."""
        if r.open is not None:
            return E.UNKNOWN, r.open
        value, _ = self.held(r.column)
        if value is None:
            return E.UNKNOWN, T.Said("held-no-section", {"section": HELD[r.column]})
        hold = key in value

        def mine(changes: list, kinds: tuple | None) -> list:
            return [c for c in changes if _key(c) in (key, "*") and (kinds is None or c.kind in kinds)]

        if presence:
            risky = (mine(r.after, ADDS) + mine(r.before, REMOVES)) if hold else (mine(r.before, ADDS) + mine(r.after, REMOVES))
        else:
            risky = mine(r.before, None) + mine(r.after, None)
        live, gone = [], []
        for c in risky:
            why = self.ruled_out(c, depth)
            (gone.append(why) if why is not None else live.append(_change(c)))
        if live:
            return E.UNKNOWN, T.Said("held-may-be-changed", {"changes": live})
        why = T.Said("held-as-fetched" if hold else "not-held-as-fetched", {"gone": T.Said("held-gone", {"gone": gone}) if gone else ""})
        if not presence and not hold:
            return E.UNKNOWN, why
        return (hold if presence else value[key]), why

    def test(self, owner: str, text: str, line: int | None, arm: int | None, depth: int = 0) -> Reading | None:
        """*text*, a test or a guard of *owner*, read through the map's GuardFroms of the tests in it; None where none
        is of a name whose ways reach a held column."""
        guards, _ = _maps()
        try:
            tree = ast.parse(E.strip_annotation(text), mode="eval")
        except SyntaxError:
            return None
        inside = {ast.dump(n) for n in ast.walk(tree)}
        by_name: dict[str, list] = {}
        for g in guards.get(owner, []):
            if (line is not None and g.test_line != line) or (arm is not None and line is None and arm not in g.arms):
                continue
            try:
                if ast.dump(ast.parse(g.test, mode="eval").body) in inside:
                    by_name.setdefault(g.name, []).append(g)
            except SyntaxError:
                continue
        reading = Reading(E.UNKNOWN)
        decided: dict[str, dict[tuple[str, bool], object]] = {}
        for name, gs in by_name.items():
            routes = [r for g in gs for r in _routes(g)]
            if not any(r.column in HELD for r in routes):
                continue
            shown = _shown(routes, self.run, self.mode)
            if shown is not None:
                routes, reading.shown = [shown], shown.label
            column = next(r.column for r in routes if r.column in HELD) if any(r.column in HELD for r in routes) else ""
            reading.section, reading.as_of = HELD.get(column, ""), self.held(column)[1] if column else ""
            for key, presence in _uses(tree, name):
                got = [(r, *self.key(r, key, presence, depth)) for r in routes]
                agreed = got[0][1] if len({repr(v) for _, v, _ in got}) == 1 else E.UNKNOWN
                decided.setdefault(name, {})[(key, presence)] = agreed
                use = f"{key!r} in {name}" if presence else f"{name}[{key!r}]"
                groups: dict[tuple[str, str], list[str]] = {}
                for r, v, why in got:
                    groups.setdefault((repr(v), repr(T.to_json(why))), []).append(r.label)
                for (_, _), labels in groups.items():
                    r, v, why = next(x for x in got if x[0].label == labels[0])
                    reading.keys.append(T.Said("held-key" if v is not E.UNKNOWN else "held-key-not-read", {
                        "use": use, "value": v if v is not E.UNKNOWN else None, "routes": labels, "why": why}))
        if not decided:
            return None
        # read at the test (or the arm): another function's hole only where the map hands its object here (K63)
        at = line if line is not None else arm
        reading.value = E.evaluate(
            _put(tree, decided),
            self.env.reading(owner, at, arm=(owner, arm) if arm is not None else None) if at is not None else self.env,
        )
        return reading


def _uses(tree: ast.AST, name: str) -> list[tuple[str, bool]]:
    """The keys of *name* the test reads: ``'K' in name`` (presence) and ``name['K']`` (value)."""
    found = []
    for n in ast.walk(tree):
        if (isinstance(n, ast.Compare) and len(n.ops) == 1 and isinstance(n.ops[0], (ast.In, ast.NotIn))
                and isinstance(n.left, ast.Constant) and isinstance(n.left.value, str)
                and isinstance(n.comparators[0], ast.Name) and n.comparators[0].id == name):
            found.append((n.left.value, True))
        elif (isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id == name
              and isinstance(n.slice, ast.Constant) and isinstance(n.slice.value, str) and isinstance(n.ctx, ast.Load)):
            found.append((n.slice.value, False))
    return list(dict.fromkeys(found))


def _put(tree: ast.Expression, decided: dict[str, dict[tuple[str, bool], object]]) -> ast.AST:
    """*tree*'s body with each decided use of a name replaced by its value."""

    class Put(ast.NodeTransformer):
        def visit_Compare(self, n: ast.Compare) -> ast.AST:
            self.generic_visit(n)
            if (len(n.ops) == 1 and isinstance(n.ops[0], (ast.In, ast.NotIn)) and isinstance(n.left, ast.Constant)
                    and isinstance(n.left.value, str) and isinstance(n.comparators[0], ast.Name)):
                got = decided.get(n.comparators[0].id, {}).get((n.left.value, True), E.UNKNOWN)
                if got in (True, False):
                    return ast.Constant(got if isinstance(n.ops[0], ast.In) else not got)
            return n

        def visit_Subscript(self, n: ast.Subscript) -> ast.AST:
            self.generic_visit(n)
            if isinstance(n.value, ast.Name) and isinstance(n.slice, ast.Constant) and isinstance(n.slice.value, str):
                got = decided.get(n.value.id, {}).get((n.slice.value, False), E.UNKNOWN)
                # a literal only: a container read as fetched stays a term the evaluator does not read
                if isinstance(got, (str, bytes, bool, int, float, complex)) or got is None:
                    return ast.Constant(got)
            return n

    put = Put().visit(ast.parse(ast.unparse(tree), mode="eval"))
    assert isinstance(put, ast.Expression)
    return ast.fix_missing_locations(put.body)


def read(owner: str, guard: str, arm: int | None, data: dict, run=None, mode: str = "", env: object = None) -> Reading | None:
    """*guard* of *owner*'s arm at line *arm*, read in the held columns of the rows file *data* (K49); None where it
    reads no name whose ways reach one.  *run*, where given, may show a route; *env*, an ``E.Env`` where given, reads
    the guard's other terms."""
    return _Reader(data, run, mode, env).test(owner, guard, None, arm)
