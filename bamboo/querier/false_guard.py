"""A guard that came out false, its two operands compared by value (a library for rerun.py).

:mod:`bamboo.querier.raised` turns an exception raised inside the corpus into the guard
that was false (``tmp_dataset in sub_to_dataset_map``).  This module takes
the map's walk from the raising line and gives each operand its value for
this instance:

* **the operand the message prints** has its value already; what is asked
  is which of its origins produced it.  Each origin is a binding in the walk,
  evaluated for every row of the loop it sits in (``loop at 279: for file in
  self.job.Files`` in the step's ``unseen``), and the one whose value equals
  the printed one is the origin that took effect -- rule 9, value match, the
  same selection H4 makes among ``set_*`` calls.
* **the other operand** of ``in`` is a container, and what the atom tested
  is its key set: the keys every write to it put in (``d[k] = ...`` read
  off the step's own statement, because the walk's step carries the value
  and not the key), for every row that reached the write past its guards and
  early exits.  The set is complete only when every write is one of those
  and every row's reach is known.

**Where they part.**  The printed origin took effect for one row; the other
bindings of the same name for that row are what the code could have used
instead.  One whose value would have made the atom true is the alternative
that mattered, and the guard that chose between the two is the decision to
follow -- its inputs, less the loop's own row (context, not a cause), are
the next question.  This is the second half of norm 1 (H10) with the
two sides being the two operands of one atom rather than one atom at two
sites.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

from bamboo.querier import evaluate as E  # noqa: E402
from bamboo.querier import rows as R  # noqa: E402

LOOP = re.compile(r"loop at (\d+): for (.+?) in (.+)$")
EARLY = re.compile(r"early-exit at (\d+): (.+)$")
ELEMENT = re.compile(r"an element of (.+)$")


@dataclass
class Element:
    """One row a loop in the walk ran for."""

    var: str
    entity: str
    row: dict
    label: str


@dataclass
class Origin:
    """One way a value reached a name or a container's key: a binding, the row it ran for, its value there."""

    step: object  # the TraceStep
    expr: str  # what was bound (the key for a container write)
    element: Element | None
    value: object
    reached: object  # True, False or UNKNOWN: its guards and early exits for that row
    why: str
    via: list["Origin"] = field(default_factory=list)  # the origins of a name *expr* is
    failed: tuple[str, str] | None = None  # (what, test) that was false, when *reached* is False

    def where(self) -> str:
        row = f" for {self.element.label}" if self.element else ""
        return f"@{self.step.line} {self.expr}{row}"


@dataclass
class Keys:
    """A container's key set, from the writes the walk names."""

    origins: list[Origin]
    opaque: list[str]  # writes that do not say their keys, and why

    def reached(self) -> list[Origin]:
        return [o for o in self.origins if o.reached is True]

    def certain(self) -> set:
        """Keys that went in: the write was reached and its key is known."""
        return {o.value for o in self.reached() if o.value is not E.UNKNOWN}

    def possible(self) -> set:
        """Keys that may have gone in: every write not ruled out."""
        return {o.value for o in self.origins if o.reached is not False and o.value is not E.UNKNOWN}

    def complete(self) -> bool:
        return not self.opaque and all(o.reached is not E.UNKNOWN and o.value is not E.UNKNOWN for o in self.origins)


class Walk:
    """The map's walk from a line, read with this instance's values."""

    def __init__(
        self, steps: list, env: E.Env, collections: dict[str, dict], rows: dict[str, list[dict]], ran: list[tuple[str, int]] = (),
    ) -> None:
        self.steps = steps
        self.env = env
        #: canonical iterable (``<job>.Files``) -> {"rows": key in *rows*, "entity": name}
        self.collections = collections
        self.rows = rows
        #: (owner, "early-exit at N", test) that held: an early exit on the path to a line a
        #: traceback frame shows running was not taken, so nothing below it was cut off by it.
        #: Only early exits: they carry their line, and a guard's text alone does not say which ``if``.
        self.held: set[tuple[str, str, str]] = set()
        for owner, line in ran:
            for s in self.steps:
                if s.owner == owner and s.line == line:
                    self.held.update((owner, what, t) for t, what in self._tests(s) if what != "guard")
        #: (owner, what, test) a value match showed to read a row value from after the decision:
        #: unknown from then on, whatever the row says
        self.stale: set[tuple[str, str, str]] = set()

    # ------------------------------------------------------------ the walk's steps
    def defs(self, name: str, owner: str) -> list:
        # a handover step is the caller's, but binds the callee's parameter
        return [s for s in self.steps if s.name == name and s.owner == owner and s.kind not in ("unbound", "handover")]

    def handed(self, name: str, owner: str) -> list:
        """The caller's side of a parameter: the handover steps that pass *name* in."""
        unbound = [s for s in self.steps if s.name == name and s.owner == owner and s.kind == "unbound"]
        if not unbound:
            return []
        return [s for s in self.steps if s.kind == "handover" and s.name == name and s.owner != owner]

    # ------------------------------------------------------------ one row of a loop
    def elements(self, step) -> list[Element | None]:
        """The rows the loops around *step* ran for; ``[None]`` when it is in no loop."""
        loops = [m for u in step.unseen if (m := LOOP.match(u))]
        if not loops:
            return [None]
        _, var, iterable = loops[-1].groups()
        found = self.collections.get(self.env.canonical(iterable.strip()))
        if found is None:
            return []
        entity = found["entity"]
        out = []
        for row in self.rows[found["rows"]]:
            ident = next((row[k]["value"] for k in ("lfn", "name", "PandaID") if k in row), "?")
            out.append(Element(var.strip(), entity, row, f"{entity} {ident}"))
        return out

    def env_for(self, element: Element | None) -> E.Env:
        if element is None:
            return self.env
        # the element's row is read as the collection holds it, so what the subject's row left unread is not its
        unread = {k: v for k, v in self.env.unread.items() if not k.startswith(f"<{element.entity}>.")}
        env = E.Env(dict(self.env.values), dict(self.env.rows), dict(self.env.table), dict(self.env.stale), unread=unread, at=self.env.at)
        env.rows[element.var] = element.entity
        env.table[element.entity] = {k: E.Value(v["value"], v["source"]) for k, v in element.row.items()}
        return env

    @staticmethod
    def _tests(step) -> list[tuple[str, str]]:
        return [(g, "guard") for g in step.guards] + [(m[2], f"early-exit at {m[1]}") for u in step.unseen if (m := EARLY.match(u))]

    def reach(self, step, env: E.Env) -> tuple[object, str, tuple[str, str] | None]:
        """Whether the row got to *step*: its path condition and the early exits above it."""
        pending = []
        for test, what in self._tests(step):
            if (step.owner, what, test) in self.held:
                continue
            if (step.owner, what, test) in self.stale:
                pending.append(f"{what} {test!r} (the row's value is from after the decision)")
                continue
            v = E.evaluate(test, env)
            if v is False:
                return False, f"{what} {test!r} is false", (what, test)
            if v is E.UNKNOWN:
                pending.append(f"{what} {test!r}")
        return (E.UNKNOWN, "unknown: " + "; ".join(pending), None) if pending else (True, "", None)

    # ------------------------------------------------------------ what a name held
    def origins(self, name: str, owner: str, element: Element | None = None) -> list[Origin]:
        """Each binding of *name* in *owner* with its value, for *element* or for every row of its loop."""
        out = []
        defs = self.defs(name, owner)
        if not defs:
            for h in self.handed(name, owner):
                out.extend(self.origins(h.value, h.owner, element) if h.value.isidentifier() else [])
            return out
        for s in defs:
            for el in [element] if element is not None else self.elements(s):
                env = self.env_for(el)
                reached, why, failed = self.reach(s, env)
                o = Origin(s, s.value, el, E.UNKNOWN, reached, why, failed=failed)
                if s.kind == "loop-target":
                    o.value = E.UNKNOWN
                else:
                    o.value = E.evaluate(s.value, env) if _parses(s.value) else E.UNKNOWN
                    if o.value is E.UNKNOWN and s.value.isidentifier():
                        o.via = self.origins(s.value, s.owner, el)
                out.append(o)
        return out

    def keys(self, container: str, owner: str) -> Keys:
        """The keys put into *container*, following it back through the call that handed it in."""
        found = Keys([], [])
        defs = self.defs(container, owner)
        if not defs:
            handed = self.handed(container, owner)
            if not handed:
                found.opaque.append(f"{container} in {owner.split('::')[-1]}: no binding and no caller in the walk")
            for h in handed:
                if not h.value.isidentifier():
                    found.opaque.append(f"@{h.line} passes {h.value}")
                    continue
                more = self.keys(h.value, h.owner)
                found.origins += more.origins
                found.opaque += more.opaque
            return found
        for s in defs:
            key = _written_key(s)
            if key is None:
                if _puts_no_key(s):
                    continue
                found.opaque.append(f"@{s.line} {_shown(container, s)} does not say its keys")
                continue
            elements = self.elements(s)
            if not elements:
                found.opaque.append(f"@{s.line} runs in a loop over rows this investigation does not have")
            for el in elements:
                env = self.env_for(el)
                reached, why, failed = self.reach(s, env)
                value = E.evaluate(key, env)
                o = Origin(s, key, el, value, reached, why, failed=failed)
                if value is E.UNKNOWN and key.isidentifier():
                    o.via = self.origins(key, s.owner, el)
                found.origins.append(o)
        return found

    def element_of(self, name: str, owner: str) -> tuple[str, str] | None:
        """``(container, owner)`` when *name* is a loop's element drawn from a container's keys."""
        for s in self.defs(name, owner):
            m = ELEMENT.match(s.value) if s.kind == "loop-target" else None
            if not m:
                continue
            it = m[1].strip()
            inner = re.fullmatch(r"(?:list|sorted|set)\((\w+)\)|(\w+)(?:\.keys\(\))?", it)
            if inner:
                return (inner[1] or inner[2]), s.owner
        return None


    # ------------------------------------------------------------ a traced container's value
    def build(self, container: str, owner: str, prefer: dict[str, int] | None = None) -> tuple[object, list[str]]:
        """*container*'s value for this instance, rebuilt from the writes the walk names.

        Returns ``(value, notes)``.  The value is ``None`` when it cannot be rebuilt,
        and the notes say why; otherwise they list what was assumed.  A write whose
        reach is unknown is taken as run; a value the instance does not give is
        left opaque.  Only what the walk names can be rebuilt, so the function that
        fills the container is checked against its own AST first: a write there the
        walk does not name makes the value unknown rather than short.  The
        container built so far, and the key just chosen, are bound while a write's
        guards are read, so ``k not in d`` is answered as the code answered it.
        *prefer* (name -> line) takes that binding of a name wherever the row has
        it: R3's alternative, as a counterfactual.
        """
        from bamboo.querier.reproduce import Opaque

        prefer = prefer or {}
        name, fill, handed_at = container, owner, None
        while not self.defs(name, fill):
            handed = [h for h in self.handed(name, fill) if h.value.isidentifier()]
            if len(handed) != 1:
                return None, [f"{name}: {len(handed)} call(s) in the walk pass it in by name"]
            name, fill, handed_at = handed[0].value, handed[0].owner, handed[0].line
        defs = sorted(self.defs(name, fill), key=lambda s: s.line)
        unnamed = sorted(set(_writes(fill, name, handed_at)) - {s.line for s in defs})
        if unnamed:
            return None, [f"{name} is also written or passed on at {unnamed} in {fill.split('::')[-1]}, which the walk does not name"]
        value: dict = {}
        assumed: dict[str, int] = {}
        for s in defs:
            key_expr = _written_key(s)
            if key_expr is None:
                if _puts_no_key(s):
                    continue
                return None, [f"@{s.line} {_shown(name, s)} is not a keyed write"]
            elements = self.elements(s)
            if not elements:
                return None, [f"@{s.line} runs over rows this investigation does not have"]
            for el in elements:
                env = self.env_for(el)
                key = E.evaluate(key_expr, env)
                if key is E.UNKNOWN and key_expr.isidentifier():
                    key, how = self._choose(key_expr, s.owner, el, prefer)
                    if how:
                        assumed[how] = assumed.get(how, 0) + 1
                if key is E.UNKNOWN:
                    reached, _, _ = self.reach(s, env)
                    if reached is False:
                        continue
                    return None, [f"@{s.line} the key {key_expr} for {el.label if el else 'the call'} is not known"]
                env.put(key_expr, key, "chosen for this row")
                env.put(name, dict(value), "as built so far")
                reached, why, _ = self.reach(s, env)
                if reached is False:
                    continue
                if reached is E.UNKNOWN:
                    note = f"@{s.line} taken as run: {why}"
                    assumed[note] = assumed.get(note, 0) + 1
                label = el.label if el else "the call"
                if _operator(s) == ".setdefault":
                    if not _setdefault_alone(s):
                        value[key] = Opaque(f"what @{s.line} makes of {name}[{key!r}] for {label}")
                        continue
                    if key in value:
                        continue  # setdefault leaves a key that is there as it is
                v = E.evaluate(s.value, env) if _parses(s.value) else E.UNKNOWN
                value[key] = Opaque(f"{s.value} for {label}") if v is E.UNKNOWN else v
        notes = [f"{how} ({n} row(s))" for how, n in assumed.items()]
        notes.append(f"order: the rows file's order of {name}'s loop rows is taken as the code's")
        return value, notes

    def _choose(self, name: str, owner: str, element: Element | None, prefer: dict[str, int]) -> tuple[object, str]:
        """The value *name* was bound to for the row, choosing among its bindings by their guards."""
        found = self.origins(name, owner, element)
        forced = [o for o in found if prefer.get(name) == o.step.line]
        if forced:
            return forced[0].value, ""
        env = self.env_for(element)
        live = [o for o in found if not any(E.evaluate(g, env) is False for g in o.step.guards)]
        certain = [o for o in live if all(E.evaluate(g, env) is True for g in o.step.guards)]
        if len(certain) == 1 and len(live) == 1:
            return certain[0].value, ""
        if len(live) == 1:
            return live[0].value, f"@{live[0].step.line} {name} = {live[0].step.value} taken, its guards unknown"
        return E.UNKNOWN, ""


MUTATORS = {"update", "setdefault", "pop", "popitem", "clear", "append", "extend", "insert", "remove", "add", "discard"}
#: builtins that read a container they are given and do not keep or change it
READERS = {"len", "str", "repr", "list", "sorted", "set", "dict", "tuple", "bool", "any", "all", "isinstance", "min", "max", "sum", "enumerate", "zip", "iter"}


def _held_in(node: ast.expr) -> str | None:
    """The name whose container *node* is a place in -- ``d`` for ``d``, ``d[k][j]``, ``d.setdefault(k, [])`` and
    ``d.get(k)`` --, or None."""
    while True:
        if isinstance(node, ast.Subscript):
            node = node.value
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("setdefault", "get"):
            node = node.func.value
        else:
            return node.id if isinstance(node, ast.Name) else None


def _writes(owner: str, name: str, before: int | None) -> list[int]:
    """Lines of *owner* above *before* that write *name*, change it in place, or hand it to a call -- *name* itself or
    a place in it at any depth (``d[k][j] = v``, ``d[k].append(x)``, ``d.setdefault(k, []).append(x)``, ``f(d[k])``),
    since a value rebuilt from the walk's writes of *name* alone is short of what those put there."""
    rel, func = owner.split("::", 1)
    try:
        fn = R.function(rel, func)
    except LookupError:
        return []
    out = []
    for n in R.own_nodes(fn.node):
        line = getattr(n, "lineno", 0)
        if before is not None and line >= before:
            continue
        if isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Delete)):
            targets = n.targets if isinstance(n, (ast.Assign, ast.Delete)) else [n.target]
            for t in targets:
                if _held_in(t) == name and not (isinstance(n, ast.AnnAssign) and n.value is None):
                    out.append(line)
        elif isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Attribute) and _held_in(f.value) == name and f.attr in MUTATORS:
                out.append(line)
            elif any(_held_in(a) == name for a in [*n.args, *(k.value for k in n.keywords)]):
                if not (isinstance(f, ast.Name) and f.id in READERS):
                    out.append(line)
    return out


def _parses(text: str) -> bool:
    try:
        ast.parse(text, mode="eval")
        return True
    except SyntaxError:
        return False


def _operator(step) -> str:
    """How the step's value reaches its name, as the map writes it: ``=``, ``+=``, or the filling method (``.setdefault``)."""
    return getattr(step, "operator", "") or "="


def _puts_no_key(step) -> bool:
    """``d = {}`` or ``d.update({})``: a write that puts no key.  ``d.setdefault(k, {})`` carries the same value and
    puts *k*, so the form is read, not the value alone."""
    return step.value.strip() == "{}" and _operator(step) in ("=", ".update")


def _shown(name: str, step) -> str:
    op = _operator(step)
    if op == ".setdefault":
        return f"{name}.setdefault(..., {step.value})"
    return f"{name}{op}({step.value})" if op.startswith(".") else f"{name} {op} {step.value}"


def _statement(step) -> ast.stmt | None:
    rel, func = step.owner.split("::", 1)
    try:
        fn = R.function(rel, func)
    except LookupError:
        return None
    return R.statement_at(fn.node, step.line)


def _setdefault_of(step, stmt: ast.stmt | None) -> ast.Call | None:
    """The ``d.setdefault(k, v)`` call in *stmt* the step stands for: on the step's name, with the step's value."""
    if stmt is None or _operator(step) != ".setdefault":
        return None
    for n in ast.walk(stmt):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "setdefault" and len(n.args) == 2
                and ast.unparse(n.func.value) == step.name and ast.unparse(n.args[1]) == step.value):
            return n
    return None


def _written_key(step) -> str | None:
    """The key of ``d[k] = ...`` / ``d[k] op= ...`` or of ``d.setdefault(k, ...)`` on the step's line, read off the
    statement."""
    stmt = _statement(step)
    if isinstance(stmt, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
        targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
        for t in targets:
            if isinstance(t, ast.Subscript) and ast.unparse(t.value) == step.name:
                return ast.unparse(t.slice)
    call = _setdefault_of(step, stmt)
    return ast.unparse(call.args[0]) if call is not None else None


def _setdefault_alone(step) -> bool:
    """Whether the statement is the setdefault call and nothing else: in ``d.setdefault(k, []).append(x)`` or
    ``x = d.setdefault(k, [])`` the statement goes on to change, or to keep a name for, what *k* holds."""
    stmt = _statement(step)
    call = _setdefault_of(step, stmt)
    return call is not None and isinstance(stmt, ast.Expr) and stmt.value is call


def flat(origins: list[Origin]) -> list[tuple[Origin, Origin]]:
    """``(outer, leaf)`` pairs: a key written as a name, with each binding that name had for the same row."""
    out = []
    for o in origins:
        if o.via:
            out.extend((o, leaf) for leaf in o.via)
        else:
            out.append((o, o))
    return out


@dataclass
class Part:
    """Where the printed operand and the other one part."""

    taken: tuple[Origin, Origin]  # (the write, the binding) that produced the printed value
    alternative: Origin  # a binding of the same name, same row, whose value would have held
    decided_by: str  # the guard of the alternative, false for this row
    env: E.Env


def part(printed: object, key_side: list[Origin], keys: Keys, walk: Walk) -> tuple[list[str], list[Part]]:
    """Choose the origin of the printed key by value; find the alternatives that were in the key set."""
    lines, parts = [], []
    matched = [(o, leaf) for o, leaf in flat(key_side) if leaf.value == printed]
    for o, leaf in flat(key_side):
        mark = "  <- the message's value" if leaf.value == printed else ""
        lines.append(f"    {o.where()} via {leaf.where()} = {leaf.value!r}{mark}")
    if len({(o.step.line, leaf.step.line, leaf.element.label if leaf.element else "") for o, leaf in matched}) != 1:
        return lines, parts
    o, leaf = matched[0]
    for got in (o, leaf):
        if got.reached is False:
            # the value came out of this write, so it was reached: a row value the test
            # read is not the one the code saw (fetched after the decision)
            walk.stale.add((got.step.owner, *got.failed))
            lines.append(f"    {got.where()} reads as not reached ({got.why}), but the message's value came out of it:"
                         " a value that test reads is not what the code saw")
    for alt in o.via:
        if alt is leaf or alt.value is E.UNKNOWN or alt.value not in keys.certain():
            continue
        env = walk.env_for(alt.element)
        false = [g for g in alt.step.guards if E.evaluate(g, env) is False]
        for g in false:
            parts.append(Part((o, leaf), alt, g, env))
    return lines, parts


def inputs(guard: str, element: Element | None) -> tuple[list[str], list[str]]:
    """The operands of the guard's atoms, split into the loop's own row (context) and the rest."""
    context, rest = [], []
    for atom in E.atoms(ast.parse(E.strip_annotation(guard), mode="eval").body):
        operands = [atom.left, *atom.comparators] if isinstance(atom, ast.Compare) else [atom]
        for n in operands:
            if isinstance(n, ast.Constant):
                continue
            text = ast.unparse(n)
            (context if element is not None and text.split(".", 1)[0] == element.var else rest).append(text)
    return context, rest


def returns_of(expr: str, walk: Walk) -> list:
    """The walk's return steps of the method *expr* calls (``x.m()``): every step named ``m()``, of any owner.

    By name, so two classes' ``m()`` the walk entered come back together, and a
    call placed open may also return from a function the walk did not enter
    (``TraceStep.open``) -- the caller says which it is.
    """
    m = re.fullmatch(r"(.+)\.(\w+)\(\)", expr)
    if not m:
        return []
    return [s for s in walk.steps if s.name == f"{m[2]}()" and s.kind == "return"]


def computed_by(expr: str, walk: Walk) -> tuple[str, list[str], list[str]]:
    """The ``rel::function`` that computes *expr*, for recomputing it on its own class; "" when the walk cannot say.

    Returns (the function, the owners of its return steps, the reasons a call
    of it was placed open).  One owner and no mark: that one.  Two owners are
    two classes' ``m()`` pooled by name; a mark says the call may also land
    where the walk did not go.  Either way recomputing on one class would
    answer for a class the run may not have used.
    """
    returns = returns_of(expr, walk)
    owners = sorted({s.owner for s in returns})
    opened = sorted({s.open for s in returns if getattr(s, "open", "")})
    return (owners[0] if len(owners) == 1 and not opened else ""), owners, opened


def derivation(expr: str, walk: Walk, depth: int = 6) -> tuple[list[str], list[str]]:
    """How the walk says *expr* (a method call ``x.m()``) was computed, down to what it reads from outside.

    Returns (the lines, the leaves at the stops), a leaf spelled at the call
    site: ``self.jobMetrics`` inside ``JobSpec.altStgOutFileList`` is
    ``self.job.jobMetrics`` for the call ``self.job.altStgOutFileList()``.
    A return entered through a call placed open says "possibly more": the
    leaves are those of the functions the walk entered.
    """
    m = re.fullmatch(r"(.+)\.(\w+)\(\)", expr)
    if not m:
        return [], []
    receiver = m[1]
    lines, leaves = [], []
    frontier = returns_of(expr, walk)
    seen: set[tuple[str, str, int]] = set()
    while frontier and depth > 0:
        depth -= 1
        nxt = []
        for s in frontier:
            key = (s.owner, s.name, s.line)
            if key in seen:
                continue
            seen.add(key)
            guards = f"  [guards {s.guards}]" if s.guards else ""
            lines.append(f"@{s.line} {s.name} = {s.value}{guards}  ({s.owner.split('::')[-1]})")
            if getattr(s, "open", ""):
                lines.append(f"    possibly more: {s.open}")
            if s.terminal:
                leaf = s.name
                if leaf == "self" or leaf.startswith("self."):
                    leaf = receiver + leaf[4:]
                leaves.append(leaf)
                lines.append(f"    stop: {s.terminal} ({s.detail})")
                continue
            for r in s.reads:
                nxt += [t for t in walk.steps if t.name == r and t.owner == s.owner]
        frontier = nxt
    return lines, list(dict.fromkeys(leaves))
