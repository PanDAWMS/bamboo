"""Evaluate a guard for one instance, pick the binding that took effect, and solve a guard backwards.

The stand-in for the evaluation API the design settled on: a guard is evaluated with PanDA's own code where it calls a helper,
and only simple predicates are computed here.  Every value carries where it
came from.

* **three values** -- ``True``, ``False`` and :data:`UNKNOWN`, with ``and``/``or``
  short-circuiting the way Python does and unknown only where it has to be;
* **helpers** are called from the local panda-server tree only when they are on
  :data:`PURE`, a list of functions read one by one and found to have no
  effect (``getSerialNumber`` looks like a read and advances a sequence);
* **the binding that took effect** is the one whose guards are not false,
  when exactly one is;
* **solving backwards** -- a guard whose outcome is known (the exit ran)
  with some atoms unknown: every assignment of the unknown atoms is tried,
  and an atom that comes out the same in every assignment that reproduces
  the outcome is settled.  A settled ``a == b`` hands the known side's value
  to the other;
* **a value written after the decision** -- a row field that the same
  function writes further down is not what the guard saw, so it is unknown
  here however the row reads now (setupper writes ``file.destinationDBlock``
  with the ``_sub`` name after it has decided on it).
"""
from __future__ import annotations

import ast
import itertools
import re
import sys
from dataclasses import dataclass, field
from typing import Any

from .records import CORPUS

# importing from the pinned corpus must not write __pycache__ into it: the pin's manifest
# is how the corpus is checked not to have moved
sys.dont_write_bytecode = True
sys.path.insert(0, str(CORPUS))


class _Unknown:
    def __repr__(self) -> str:
        return "UNKNOWN"

    def __bool__(self) -> bool:
        raise TypeError("an unknown value has no truth value")


UNKNOWN: Any = _Unknown()


def _pure() -> dict[str, Any]:
    from pandaserver.dataservice import DataServiceUtils
    from pandaserver.taskbuffer import EventServiceUtils

    return {
        "DataServiceUtils.getDestinationSE": DataServiceUtils.getDestinationSE,
        "DataServiceUtils.getDistributedDestination": DataServiceUtils.getDistributedDestination,
        "DataServiceUtils.is_sub_dataset": DataServiceUtils.is_sub_dataset,
        "EventServiceUtils.isMergeAtOS": EventServiceUtils.isMergeAtOS,
    }


PURE: dict[str, Any] = {}


@dataclass
class Value:
    value: Any
    source: str


@dataclass
class Env:
    """What is known about one instance at one decision, each value with its source."""

    values: dict[str, Value] = field(default_factory=dict)
    #: variable spelling -> entity row name, e.g. "self.job" -> "job"
    rows: dict[str, str] = field(default_factory=dict)
    table: dict[str, dict[str, Value]] = field(default_factory=dict)
    #: expressions known not to hold what the guard saw, with the reason
    stale: dict[str, str] = field(default_factory=dict)
    used: dict[str, str] = field(default_factory=dict)
    #: row fields the rows file holds as read at another time than the step's, so not in ``table``, with why
    #: (``held.at_run``): "<job>.destinationSE" -> why
    unread: dict[str, str] = field(default_factory=dict)
    #: when the run the values are read for began (``Rerun.base_env``); None where no run is named
    at: str | None = None

    def canonical(self, text: str) -> str:
        """*text* with every spelling of a row variable replaced by the row's name.

        ``self.job`` in the adder and ``job`` in the task buffer are the same
        row when the question binds both to it; a printed
        ``isJumbo={EventServiceUtils.isJumboJob(self.job)}`` then answers the
        guard ``EventServiceUtils.isJumboJob(job)``.
        """
        if not self.rows:
            return text
        spellings = "|".join(re.escape(s) for s in sorted(self.rows, key=len, reverse=True))
        return re.sub(rf"(?<![\w.<])({spellings})(?!\w)", lambda m: f"<{self.rows[m[1]]}>", text)

    def put(self, text: str, value: Any, source: str) -> None:
        self.values[self.canonical(text)] = Value(value, source)

    def put_printed(self, text: str, printed: str, source: str) -> None:
        """A value read out of a log line, as Python reads its repr where it can.

        A hole ``{str(x)}`` that reads back as a literal other than a string
        says what ``x`` was, too (``str(self.job.altStgOutFileList())``).
        """
        try:
            value = ast.literal_eval(printed)
        except (ValueError, SyntaxError):
            value = printed
        self.put(text, value, source)
        m = re.fullmatch(r"str\((.+)\)", text)
        if m and not isinstance(value, str) and self.canonical(m[1]) not in self.values:
            self.put(m[1], value, f"{source}, printed as str()")

    def lookup(self, text: str) -> Any:
        key = self.canonical(text)
        for stale in (text, key):
            if stale in self.stale:
                self.used[text] = f"unknown: {self.stale[stale]}"
                return UNKNOWN
        if key in self.values:
            self.used[text] = self.values[key].source
            return self.values[key].value
        m = re.fullmatch(r"<(\w+)>\.(\w+)", key)
        if m:
            row = self.table.get(m[1], {})
            if m[2] in row:
                self.used[text] = row[m[2]].source
                return row[m[2]].value
        return UNKNOWN

    def unread_in(self, text: str) -> dict[str, str]:
        """The row fields *text* reads that nothing here holds and the rows file holds at another time, with why."""
        out = {}
        for term in re.findall(r"(?<![\w.])[A-Za-z_][\w.]*\.\w+", text):
            key = self.canonical(term)
            m = re.fullmatch(r"<(\w+)>\.(\w+)", key)
            if not m or key not in self.unread or key in self.values or term in self.stale or key in self.stale:
                continue
            if m[2] not in self.table.get(m[1], {}):
                out[term] = self.unread[key]
        return out


def strip_annotation(text: str) -> str:
    """Drop the ``  [name := expr]`` a path condition carries for a bare local."""
    return re.sub(r"\s+\[\w+ := .*\]$", "", text)


def in_callee(text: str) -> bool:
    """Whether a path condition is tested in a method the function calls: the map marks it ``  [in f()]``
    (``alias._produced``).  Its names are the callee's, so the caller's passes cannot say it."""
    return re.search(r"\s+\[in \w+\(\)\]$", text) is not None


def _call(node: ast.Call, env: Env) -> Any:
    if not PURE:
        PURE.update(_pure())
    name = ast.unparse(node.func)
    fn = PURE.get(name)
    if fn is None:
        return env.lookup(ast.unparse(node))
    args = [ev(a, env) for a in node.args]
    kwargs = {k.arg: ev(k.value, env) for k in node.keywords if k.arg}
    if any(a is UNKNOWN for a in [*args, *kwargs.values()]):
        return UNKNOWN
    result = fn(*args, **kwargs)
    env.used[ast.unparse(node)] = f"called {name} from panda-server"
    return result


def ev(node: ast.AST, env: Env) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        items = [ev(e, env) for e in node.elts]
        return UNKNOWN if any(i is UNKNOWN for i in items) else items
    if isinstance(node, ast.BoolOp):
        is_and = isinstance(node.op, ast.And)
        pending = False
        for value in node.values:
            v = ev(value, env)
            if v is UNKNOWN:
                pending = True
                continue
            if is_and and not v:
                return v
            if not is_and and v:
                return v
        return UNKNOWN if pending else (True if is_and else False)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        v = ev(node.operand, env)
        return UNKNOWN if v is UNKNOWN else (not v)
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        left, right = ev(node.left, env), ev(node.comparators[0], env)
        op = node.ops[0]
        # "x is None" needs only x; everything else needs both sides
        if left is UNKNOWN or right is UNKNOWN:
            return UNKNOWN
        table = {
            ast.Eq: lambda: left == right,
            ast.NotEq: lambda: left != right,
            ast.In: lambda: left in right,
            ast.NotIn: lambda: left not in right,
            ast.Is: lambda: left is right,
            ast.IsNot: lambda: left is not right,
        }
        if type(op) in table:
            return table[type(op)]()
        return UNKNOWN
    if isinstance(node, ast.Call):
        return _call(node, env)
    if isinstance(node, (ast.Name, ast.Attribute, ast.Subscript)):
        return env.lookup(ast.unparse(node))
    return UNKNOWN


def evaluate(text: str, env: Env) -> Any:
    return ev(ast.parse(strip_annotation(text), mode="eval").body, env)


# ---------------------------------------------------------------- which binding took effect


@dataclass
class Taken:
    name: str
    step: Any = None  # the TraceStep, when exactly one survived
    value: Any = UNKNOWN
    kept: list = field(default_factory=list)
    dropped: list = field(default_factory=list)  # (step, the guard that was false)
    foreign: list = field(default_factory=list)  # a field's steps in other methods whose class is not known, not read
    other_class: list = field(default_factory=list)  # a field's steps in a class outside the walked one's lineage

    def describe(self) -> str:
        if self.step is None:
            lines = sorted({s.line for s in self.kept})
            unread = f"; {len(self.foreign)} binding(s) in other methods not read" if self.foreign else ""
            return f"{self.name}: {len(self.kept)} binding(s) not ruled out {lines}{unread}"
        return f"{self.name} = {self.step.value}  @{self.step.line}  -> {self.value!r}"


def taken(steps: list, name: str, env: Env, *, owner: str) -> Taken:
    """The binding of *name* among a walk's *steps* that the instance's values leave standing.

    Only the steps of *owner* (``module::function``) bind its names.  A walk
    that crosses into a callee or out to a caller carries their steps too,
    and a local of another function spelled the same -- ``tmp_dst_ddm``,
    ``job`` -- is not a binding of this one's.  Required, so a caller that
    does not say whose name it asks about fails loudly rather than mixing them.

    A ``self.<field>`` is the object's, not the function's: ``__init__`` and
    the other methods of the class and its bases bind it too, and the map
    walks them for that reason.  The owner names a function, not its class
    (``TaskCommando.py::__init__`` is two classes'), so the class comes from
    the steps: each carries ``cls`` (``file::Class``, the map's ``TraceStep.cls``
    since f9b501c), and the walked function's own steps say its class.  A
    field's step in another method is kept when its class is in that class's
    lineage (``calls.Types.lineage``) and set aside when it is not -- another
    object's field.  Its guards and its value are that method's frame, so they
    are not read with *env*: the value only when it is a literal.  Where the
    class cannot be told (a tree before f9b501c, or own steps of two classes),
    such a step is ``foreign`` and no binding is taken, as before.
    """
    # imported here: evaluate reads no corpus, and asked's rows reads it as it is imported
    from bamboo.querier import asked as A

    A.bindings(steps, owner, name)
    out = Taken(name)
    walked = {s.cls for s in steps if s.owner == owner and getattr(s, "cls", "")}
    lineage = _lineage(next(iter(walked))) if len(walked) == 1 else None
    for step in steps:
        if step.name != name:
            continue
        if getattr(step, "kind", "") == "handover" and step.owner == owner:
            # the owner's side of a call: it binds the callee's parameter of that name, not the owner's own
            # (setup_destination 768 hands file.destinationDBlock to make_sub_dataset_name's original_name)
            continue
        if step.owner and step.owner != owner:
            if name.startswith("self."):
                if lineage is None or not getattr(step, "cls", ""):
                    out.foreign.append(step)
                elif step.cls in lineage:
                    out.kept.append(step)
                else:
                    out.other_class.append(step)
            continue
        false = next((g for g in step.guards if evaluate(g, env) is False), None)
        if false is not None:
            out.dropped.append((step, false))
        else:
            out.kept.append(step)
    # the same text bound twice (789/791 for tmp_src_ddm) is one answer
    if len({(s.value) for s in out.kept}) == 1 and out.kept and not out.foreign:
        out.step = out.kept[0]
        if out.step.terminal:
            out.value = UNKNOWN
        elif out.step.owner and out.step.owner != owner:
            out.value = _literal(out.step.value)
        else:
            out.value = _value_of(out.step.value, env)
    return out


def _literal(text: str) -> Any:
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return UNKNOWN


_LINEAGES: dict[str, frozenset[str] | None] = {}


def _lineage(cls: str) -> frozenset[str] | None:
    """``file::Class`` of *cls* and its bases, as the map's ``calls.Types.lineage`` follows them; None if unread."""
    if cls not in _LINEAGES:
        from bamboo.codemap.calls import Types

        from .rows import ROOTS

        file, name = cls.split("::", 1)
        types = Types(ROOTS, {})
        tree = types.tree(file)
        found = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == name] if tree is not None else []
        _LINEAGES[cls] = (
            frozenset(f"{where}::{holder.name}" for where, holder in types.lineage(file, found[0])) if len(found) == 1 else None
        )
    return _LINEAGES[cls]


def _value_of(text: str, env: Env) -> Any:
    try:
        return evaluate(text, env)
    except SyntaxError:
        return UNKNOWN


# ---------------------------------------------------------------- solving backwards


def atoms(node: ast.AST) -> list[ast.AST]:
    if isinstance(node, ast.BoolOp):
        return [a for v in node.values for a in atoms(v)]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return atoms(node.operand)
    return [node]


def _truth(node: ast.AST, fixed: dict[str, bool], env: Env) -> Any:
    text = ast.unparse(node)
    if text in fixed:
        return fixed[text]
    if isinstance(node, ast.BoolOp):
        vals = [_truth(v, fixed, env) for v in node.values]
        if isinstance(node.op, ast.And):
            if any(v is False for v in vals):
                return False
            return UNKNOWN if any(v is UNKNOWN for v in vals) else True
        if any(v is True for v in vals):
            return True
        return UNKNOWN if any(v is UNKNOWN for v in vals) else False
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        v = _truth(node.operand, fixed, env)
        return UNKNOWN if v is UNKNOWN else (not v)
    v = ev(node, env)
    return UNKNOWN if v is UNKNOWN else bool(v)


@dataclass
class Solved:
    test: str
    outcome: bool
    known: dict[str, bool]
    settled: dict[str, bool]
    open: list[str]
    derived: dict[str, tuple[Any, str]]  # expression -> (value, why)
    consistent: bool


def solve(test: str, outcome: bool, env: Env) -> Solved:
    tree = ast.parse(strip_annotation(test), mode="eval").body
    found: dict[str, ast.AST] = {}
    for atom in atoms(tree):
        found.setdefault(ast.unparse(atom), atom)
    known: dict[str, bool] = {}
    unknown: list[str] = []
    for text, atom in found.items():
        v = ev(atom, env)
        if v is UNKNOWN:
            unknown.append(text)
        else:
            known[text] = bool(v)
    fits = []
    for values in itertools.product((True, False), repeat=len(unknown)):
        fixed = {**known, **dict(zip(unknown, values, strict=False))}
        if _truth(tree, fixed, env) is outcome:
            fits.append(dict(zip(unknown, values, strict=False)))
    settled = {u: fits[0][u] for u in unknown if fits and all(f[u] == fits[0][u] for f in fits)}
    derived: dict[str, tuple[Any, str]] = {}
    for text, truth in settled.items():
        atom = found[text]
        if isinstance(atom, ast.Compare) and len(atom.ops) == 1:
            op = atom.ops[0]
            equal = (isinstance(op, ast.Eq) and truth) or (isinstance(op, ast.NotEq) and not truth)
            if equal:
                left, right = atom.left, atom.comparators[0]
                lv, rv = ev(left, env), ev(right, env)
                if lv is UNKNOWN and rv is not UNKNOWN:
                    derived[ast.unparse(left)] = (rv, f"{text} is {truth} and {ast.unparse(right)} = {rv!r}")
                elif rv is UNKNOWN and lv is not UNKNOWN:
                    derived[ast.unparse(right)] = (lv, f"{text} is {truth} and {ast.unparse(left)} = {lv!r}")
    return Solved(
        test,
        outcome,
        known,
        settled,
        [u for u in unknown if u not in settled],
        derived,
        bool(fits),
    )


# ---------------------------------------------------------------- values written after the decision


def written_after(func: ast.AST, line: int, env: Env) -> dict[str, str]:
    """Row fields the function writes below *line*, marked stale in *env*."""
    from .rows import own_nodes

    marked: dict[str, str] = {}
    for node in own_nodes(func):
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)) and node.lineno > line:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                text = ast.unparse(t)
                if re.fullmatch(r"<\w+>\.\w+", env.canonical(text)):
                    marked[env.canonical(text)] = f"{text} written at {node.lineno}, after the decision at {line}"
    env.stale.update(marked)
    return marked
