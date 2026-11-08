"""What one run knows at a decision, for the map's evaluator; the binding that took effect; values written after it.

The evaluator is the map's (:mod:`bamboo.codemap.panda.evaluate`): three values, literals folded, a call the
reader's, solving backwards.  The querier reads it as ``E.ev``, ``E.solve`` and the like, and hands it what a run
says:

* **the rows a run read**, under the names its question binds (:class:`Env`): ``self.job`` in the adder and
  ``job`` in the task buffer are one row, and a field the rows file holds answers a guard that reads it.  A hole of
  the question's own column one function printed answers another function's statement only where the map's records
  hold one object (:mod:`same_object`, KNOWN_ISSUES.md K63);
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
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from typing import Any

from bamboo.codemap.panda import evaluate as shared
from bamboo.codemap.panda.evaluate import (  # noqa: F401 -- the evaluator, read here as E.<name>
    UNKNOWN,
    Solved,
    Value,
    atoms,
    decidable,
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
    #: which function printed each hole about a row, as what, with the hole's source: "<job>.jobStatus" ->
    #: {("pandaserver/dataservice/adder_gen.py::process_job_report", "self.job.jobStatus", "hole of ...")}
    #: (:meth:`put_hole`)
    printed_in: dict[str, set[tuple[str, str, str]]] = field(default_factory=dict)
    #: the function whose statement is read, and the statement's line (:meth:`reading`); None where no one statement is
    reader: tuple[str, int] | None = None
    #: what this reading did not read of another function's holes (:meth:`_hidden`): true of this statement only
    hidden_here: set[str] = field(default_factory=set)
    #: the question's own column under its row's name ("<job>.jobStatus", ``Rerun.base_env``): the one the statuslog
    #: shows changing at the question's time; "" where no question is read
    subject: str = ""
    #: the question's subject as the map names it ("JobSpec.jobStatus"), whose arms :meth:`for_arm` binds names for
    written: str = ""
    #: a row's collection attribute -> its element rows' name ("Files" -> "file"), from the rows file
    elements: dict[str, str] = field(default_factory=dict)
    #: the arm whose guards this env reads (:meth:`for_arm`): its function and line; None where it reads no one arm's
    arm: tuple[str, int] | None = None
    #: the rows file's spellings (:attr:`rows` before :meth:`for_arm` narrowed it): how a printing function's names are
    #: read, since a print is the run's object as its function spells it
    spellings: dict[str, str] | None = None
    #: the row's key columns ("PandaID"): its class's columns whose value in the rows file is the question's key
    #: (``Rerun._key_columns``).  A print of one under a name bound to the object an arm writes says which object the
    #: arm's pass wrote (``which_ran.other_objects``)
    keys: frozenset[str] = frozenset()
    #: the rows file's statuslog rows of the question's key (``Rerun.base_env``): the state the question's column was in
    #: when a pass read it (``which_ran.row_state``, H2 step 2 (iii)); empty where none is read
    statuslog: list[dict] = field(default_factory=list)

    def put_hole(self, text: str, printed: str, source: str, owners: Iterable[str]) -> None:
        """:meth:`put_printed`, with the functions that printed it where it is about a row."""
        keys = [self.canonical(text)]
        inner = re.fullmatch(r"str\((.+)\)", text)
        if inner:
            keys.append(self.canonical(inner[1]))
        self.put_printed(text, printed, source)
        for key in keys:
            if "<" in key and key in self.values and self.values[key].source.startswith(source):
                self.printed_in.setdefault(key, set()).update((owner, text, source) for owner in owners)

    def reading(self, owner: str, line: int, arm: tuple[str, int] | None = None) -> Env:
        """This env, reading *owner*'s statement at *line*: a hole another function printed about a row answers only a
        name the map's records hand that object to (:meth:`_hidden`).  With *arm*, for that arm's guards
        (:meth:`for_arm`)."""
        env = replace(self, values=dict(self.values), stale=dict(self.stale), used={}, hidden_here=set(),
                      printed_in={k: set(v) for k, v in self.printed_in.items()}, reader=(owner, line))
        return env.for_arm(*arm) if arm is not None else env

    def for_arm(self, owner: str, line: int) -> Env:
        """This env, reading the guards of *owner*'s arm at *line*: the names that hold the question's row are the ones
        the map's records bind to the object the arm writes (:mod:`row_names`, KNOWN_ISSUES.md K64), not every name the
        rows file spells it with.  The rows file's other rows keep their spellings."""
        m = re.fullmatch(r"<(\w+)>\.\w+", self.subject)
        if m is None or not self.written or self.arm == (owner, line):
            return self
        # imported here: row_names reads the map's records through rows, which imports this module
        from bamboo.querier import row_names

        entity = m[1]
        bound = row_names.names(owner, line, self.written, entity, self.elements)
        spellings = self.spellings if self.spellings is not None else self.rows
        if bound is None:
            # the map records nothing the arm writes (a call on another object, ``self.result.set_fatal()``): no
            # record to narrow by, so the rows file's spellings
            return replace(self, rows=dict(spellings), arm=(owner, line), spellings=spellings)
        rows = {s: e for s, e in spellings.items() if e != entity and e not in self.elements.values()}
        rows.update(bound)
        return replace(self, rows=rows, arm=(owner, line), spellings=spellings)

    def shared(self) -> dict[str, str]:
        """:attr:`used` less what this reading did not read of another function's holes: what the env it was read
        from may keep."""
        return {k: v for k, v in self.used.items() if k not in self.hidden_here}

    def spelled(self, text: str) -> tuple[str, ...]:
        """The spellings of a row variable *text* holds (``job``, ``self.job``)."""
        if not self.rows:
            return ()
        spellings = "|".join(re.escape(s) for s in sorted(self.rows, key=len, reverse=True))
        return tuple(sorted(set(re.findall(rf"(?<![\w.<])({spellings})(?!\w)", text))))

    def _hidden(self, text: str) -> bool:
        """Whether *text* reads the question's own column (:attr:`subject`) from a hole printed only by functions whose
        object the map's records do not show is the one *text*'s names hold in the :attr:`reader`'s statement.

        Only the question's column: the statuslog shows it change at the question's time, so another object's print
        of it is not the value at the statement.  A column no write changes in a job's life (``prodSourceLabel``) is
        the row's value whichever object printed it.  Telling the other changing columns needs the writes between the
        print and the read (KNOWN_ISSUES.md K63, *Revisit*)."""
        if self.reader is None or not self.subject:
            return False
        key = self.canonical(text)
        if not re.search(rf"{re.escape(self.subject)}(?!\w)", key):
            return False
        held = self.values.get(key)
        found = [(o, e) for o, e, s in self.printed_in.get(key, ()) if held is not None and held.source.startswith(s)]
        if not found:
            return False
        # imported here: heldcol, which same_object reads, imports this module
        from bamboo.querier import same_object

        owner, line = self.reader
        names = self.spelled(text)
        printed = replace(self, rows=self.spellings) if self.spellings is not None else self
        return not any(same_object.same(owner, line, names, printer, printed.spelled(expr)) for printer, expr in found)

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

    def _field(self, text: str, hidden: bool = False) -> Value | None:
        """The row field *text* reads, where nothing put (*hidden* aside) and nothing stale says otherwise."""
        key = self.canonical(text)
        if (key in self.values and not hidden) or text in self.stale or key in self.stale:
            return None
        m = re.fullmatch(r"<(\w+)>\.(\w+)", key)
        return self.table.get(m[1], {}).get(m[2]) if m else None

    def lookup(self, text: str) -> Any:
        hidden = self._hidden(text)
        held = self._field(text, hidden)
        if held is not None:
            self.used[text] = held.source
            if hidden:
                self.hidden_here.add(text)
            return held.value
        if not hidden:
            return super().lookup(text)
        key = self.canonical(text)
        stale = self.stale.get(text, self.stale.get(key))
        self.used[text] = f"unknown: {stale}" if stale else f"unknown: {self.values[key].source} is another object's"
        self.hidden_here.add(text)
        return UNKNOWN

    def knows(self, text: str) -> bool:
        hidden = self._hidden(text)
        return (super().knows(text) and not hidden) or self._field(text, hidden) is not None

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

    A name the function does not bind is a free variable of it: a def that
    holds the function binds it (datasetManager's ``main`` binds
    TRANSFER_TIMEOUT_HI_PRIORITY at 84, which its FinisherThr.run reads), and
    the map's trace answers it with that def's step.  The innermost def around
    the function that binds the name is kept (:func:`_enclosing`), read as
    another method's field is: its value only when it is a literal.
    """
    # imported here: evaluate reads no corpus, and asked's rows reads it as it is imported
    from bamboo.querier import asked as A

    A.bindings(steps, owner, name)
    out = Taken(name)
    walked = {s.cls for s in steps if s.owner == owner and getattr(s, "cls", "")}
    lineage = _lineage(next(iter(walked))) if len(walked) == 1 else None
    binds = [s for s in steps if s.name == name and getattr(s, "kind", "") != "handover"]
    binder = None
    if not name.startswith("self.") and not any(s.owner == owner for s in binds):
        # a name the function does not bind is free: the innermost def around it that binds it does
        bound = {s.owner for s in binds}
        binder = next((o for o in _enclosing(owner, next(iter(walked)) if len(walked) == 1 else None) if o in bound), None)
    for step in steps:
        if step.name != name:
            continue
        if getattr(step, "kind", "") == "handover" and step.owner == owner:
            # the owner's side of a call: it binds the callee's parameter of that name, not the owner's own
            # (setup_destination 768 hands file.destinationDBlock to make_sub_dataset_name's original_name)
            continue
        if step.owner and step.owner != owner:
            if step.owner == binder and getattr(step, "kind", "") != "handover":
                # the enclosing def's frame: its guards and its value are not read with *env*
                out.kept.append(step)
            elif name.startswith("self."):
                if lineage is None or not getattr(step, "cls", ""):
                    out.foreign.append(step)
                elif step.cls in lineage:
                    out.kept.append(step)
                else:
                    out.other_class.append(step)
            continue
        # read at the step's statement: another function's hole only where the map hands its object here (K63)
        at = env.reading(owner, step.line) if getattr(step, "line", 0) else env
        false = next((g for g in step.guards if evaluate(g, at) is False), None)
        env.used.update(at.shared())
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
            at = env.reading(owner, out.step.line) if getattr(out.step, "line", 0) else env
            out.value = _value_of(out.step.value, at)
            env.used.update(at.shared())
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


_ENCLOSING: dict[tuple[str, str | None], tuple[str, ...]] = {}


def _enclosing(owner: str, cls: str | None) -> tuple[str, ...]:
    """``file::function`` of each def around *owner*'s function, innermost first: the scopes a free variable of it is
    read in (a class body is none).  *cls* (``file::Class``, the walked steps') tells same-named methods apart; () where
    the function is not one def of the file."""
    if (owner, cls) not in _ENCLOSING:
        from bamboo.codemap.calls import Types

        from .rows import ROOTS

        file, func = owner.split("::", 1)
        tree = Types(ROOTS, {}).tree(file)
        found: list[list[ast.AST]] = []

        def visit(node: ast.AST, around: list[ast.AST]) -> None:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if (not isinstance(child, ast.ClassDef) and child.name == func.rpartition(".")[2]
                            and (cls is None or (around and isinstance(around[-1], ast.ClassDef) and f"{file}::{around[-1].name}" == cls))):
                        found.append(around)
                    visit(child, [*around, child])
                else:
                    visit(child, around)

        if tree is not None:
            visit(tree, [])
        _ENCLOSING[(owner, cls)] = (
            tuple(f"{file}::{d.name}" for d in reversed(found[0]) if isinstance(d, (ast.FunctionDef, ast.AsyncFunctionDef)))
            if len(found) == 1 else ()
        )
    return _ENCLOSING[(owner, cls)]


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
