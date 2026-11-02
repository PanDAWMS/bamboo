"""What one run knows at a decision, for the map's evaluator; the binding that took effect; values written after it.

The evaluator is the map's (:mod:`bamboo.codemap.panda.evaluate`): three values, literals folded, a call the
reader's, solving backwards.  The querier reads it as ``E.ev``, ``E.solve`` and the like, and hands it what a run
says:

* **the rows a run read**, under the names its question binds (:class:`Env`): ``self.job`` in the adder and
  ``job`` in the task buffer are one row, and a field the rows file holds answers a guard that reads it;
* **helpers** are called from the local panda-server tree only when they are on
  :data:`PURE`, a list of functions read one by one and found to have no
  effect (``getSerialNumber`` looks like a read and advances a sequence);
* **the binding that took effect** is the one whose guards are not false,
  when exactly one is;
* **a value written after the decision** -- a row field that the same
  function writes further down is not what the guard saw, so it is unknown
  here however the row reads now (setupper writes ``file.destinationDBlock``
  with the ``_sub`` name after it has decided on it).
"""
from __future__ import annotations

import ast
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from bamboo.codemap.panda import evaluate as shared
from bamboo.codemap.panda.evaluate import (  # noqa: F401 -- the evaluator, read here as E.<name>
    UNKNOWN,
    Solved,
    Value,
    atoms,
    ev,
    evaluate,
    read_elsewhere,
    solve,
    strip_annotation,
    truth,
)

from .records import CORPUS

# importing from the pinned corpus must not write __pycache__ into it: the pin's manifest
# is how the corpus is checked not to have moved
sys.dont_write_bytecode = True
sys.path.insert(0, str(CORPUS))


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


def pure(name: str) -> Callable[..., Any] | None:
    """The helper of :data:`PURE` a call names, or None: what :class:`Env` hands the evaluator to run."""
    if not PURE:
        PURE.update(_pure())
    return PURE.get(name)


@dataclass
class Env(shared.Env):
    """What is known about one instance at one decision, each value with its source: the map's, with the rows the
    run read."""

    #: variable spelling -> entity row name, e.g. "self.job" -> "job"
    rows: dict[str, str] = field(default_factory=dict)
    table: dict[str, dict[str, Value]] = field(default_factory=dict)
    #: row fields the rows file holds as read at another time than the step's, so not in ``table``, with why
    #: (``held.at_run``): "<job>.destinationSE" -> why
    unread: dict[str, str] = field(default_factory=dict)
    #: when the run the values are read for began (``Rerun.base_env``); None where no run is named
    at: str | None = None
    call: Callable[[str], Callable[..., Any] | None] | None = pure

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

    def _field(self, text: str) -> Value | None:
        """The row field *text* reads, where nothing put and nothing stale says otherwise."""
        key = self.canonical(text)
        if key in self.values or text in self.stale or key in self.stale:
            return None
        m = re.fullmatch(r"<(\w+)>\.(\w+)", key)
        return self.table.get(m[1], {}).get(m[2]) if m else None

    def lookup(self, text: str) -> Any:
        held = self._field(text)
        if held is None:
            return super().lookup(text)
        self.used[text] = held.source
        return held.value

    def knows(self, text: str) -> bool:
        return super().knows(text) or self._field(text) is not None

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
