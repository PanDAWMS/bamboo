"""Write-alias recognizer -- a method that settles a subject, seen from its callers.

``JediTaskSpec.setOnHold`` is four lines and one of them writes the most
contended value in the system::

    def setOnHold(self):
        if self.status in ["ready", "running", "merging", ...]:
            self.oldStatus = self.status
            self.status = "pending"

The attribute slice already records that write, and records it exactly once,
under ``JediTaskSpec.py::setOnHold``.  That is a true statement and a useless
one for an investigation: asked why a task is in ``pending``, the map would
answer "because ``setOnHold`` was called" and stop.  The fourteen callers -- ten
in ``JobGenerator``, three in ``TaskRefiner``, one in ``ContentsFeeder`` -- each
call it under a different condition, and *those* are the candidate causes.

So the call sites are junctions too.  A branch's path condition is the caller's
conjunction followed by the alias's own guard, marked with where it came from:
reaching ``pending`` requires both, and a reader who cannot tell them apart
cannot tell a caller that never ran from a guard that rejected the status.

**Only literal writes make a method an alias.**  Widened to any write to a
declared attribute, the same rule matches 36 methods and 600 call sites, led by
``__init__``, ``pack``, ``__setattr__`` and ``setErrDiag`` -- serialization and
parameter passing, which settle nothing.  Narrowed to literals it matches five
methods, and the criteria mark the one that matters.  A literal the method goes
on to add to -- ``self.attributes = ""``, then ``self.attributes += attr`` --
never leaves it as written, so its calls write what the method computes,
``runtime(setDatasetAttribute())``: still a write of the field at each call,
and not of ``""``.

**The same method, seen the other way round.**  A helper can settle a subject by
*returning* the value instead of writing it, and the caller assigns it::

    taskSpec.status = self.getFinalTaskStatus(taskSpec, update_error_dialog=True)

The attribute slice takes only literal right-hand sides, so a write like this
was not on the map at all -- and production put 104 tasks into ``aborted``,
which nothing in the map produced, because the only place that value is decided
is inside that helper.  ``getFinalTaskStatus`` is a chain of eight guarded
assignments to a local followed by ``return status``, so the values and their
conditions are there to be read; what it needs is one hop, the same hop the path
condition extractor already makes for a predicate that is a same-class helper's
return.  Recorded like an alias's, with the caller's conditions first and the
helper's own marked with where they came from.
"""

from __future__ import annotations

import ast
from typing import Callable, Optional

from bamboo.codemap.astwalk import walk
from bamboo.codemap.models import (
    Anchor,
    Branch,
    CoverageStat,
    JunctionNode,
    SourceModule,
    SubjectNode,
)
from bamboo.codemap.panda import sql
from bamboo.codemap.panda.attribution import CERTAIN, SpecAttributor
from bamboo.codemap.panda.pathcond import (
    Callee,
    Replacement,
    additions,
    attach_parents,
    enclosing_class,
    enclosing_function,
    entries,
    field_replacements,
    field_writes,
    functions_with_owner,
    literal_values,
    path_condition,
    replacements,
    site_of,
    targets_of,
    unmarked,
    written_value,
)

SLICE_NAME = "write-alias"
# The same relationship read the other way: a helper that returns the value.
PRODUCER_SLICE = "return-alias"


class Alias:
    """A method whose body settles a subject to a literal."""

    def __init__(self, spec_class: str, method: str) -> None:
        self.spec_class = spec_class
        self.method = method
        # (attribute, value, the conditions inside the method, tier): a literal, or what the method adds to one
        self.writes: list[tuple[str, str, list[str], int]] = []


def _literal_self_writes(
    func: ast.FunctionDef | ast.AsyncFunctionDef, declared: set[str]
) -> list[tuple[str, str, list[str], int]]:
    """Return ``(attribute, value, inner conditions, tier)`` for each ``self.attr = "lit"`` the method returns with.

    A literal the method goes on to add to does not leave it as written: ``setDatasetAttribute`` sets
    ``self.attributes = ""`` and then ``self.attributes += attr``, so what its callers' objects hold is the sum.
    Each ``+=`` that adds to the literal and whose own value leaves is then a write of the method, said
    ``runtime(setDatasetAttribute())`` (tier 2) under its own tests: the callers still write the field.  Whether the
    literal leaves, and on what, is :func:`pathcond.replacements`' reading of the ``+=``: one it surely meets ends
    it, so none of the callers writes ``""``; one that may run leaves it where that one does not.
    """
    found: list[tuple[str, str, list[str], int]] = []
    for node in walk(func):
        value = written_value(node)
        if not (isinstance(value, ast.Constant) and isinstance(value.value, str)):
            continue
        for target in targets_of(node):
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
                and target.attr in declared
            ):
                left, adding = _added_to(func, node, target, value) if enclosing_function(node) is func else ([], [])
                if left is not None:
                    inner = path_condition(node)
                    found.append((target.attr, value.value, inner + entries(left, inner), 1))
                for one in adding:
                    write = (target.attr, f"runtime({func.name}())", path_condition(one), 2)
                    if write not in found:
                        found.append(write)
    return found


def _added_to(
    func: ast.FunctionDef | ast.AsyncFunctionDef, write: ast.AST, target: ast.Attribute, value: ast.expr
) -> tuple[Optional[list[Replacement]], list[ast.AugAssign]]:
    """What the method's ``+=`` of *target* do to the literal *write* puts there.

    ``(left, adding)``: :func:`pathcond.replacements` of the literal against the ``+=`` -- ``None`` where it never
    leaves as written; the field's other writes overwrite, as killers, since what is asked here is only what the
    ``+=`` do -- and the ``+=`` that add to it and leave the method (:func:`pathcond.additions`).
    """
    writes, added, _calls = field_writes(func, target.value, target.attr)
    plain = [one for one in writes if one is not write and not isinstance(one, ast.AugAssign)]
    place = ast.unparse(target)
    left = replacements(
        write, added, [], place=place, value=value, killers=plain, rereads=added, returning=True
    )
    return left, additions(write, writes, [], place=place, returning=True)


def find_aliases(
    modules: list[SourceModule], declarations: dict[str, set[str]]
) -> dict[str, list[Alias]]:
    """Return ``{method name: [alias, ...]}`` over the whole corpus.

    Keyed by method name rather than by class because that is what a call site
    offers: ``taskSpec.setOnHold()`` names the method and leaves the class to be
    worked out.  Where exactly one class defines the name, the call site needs
    no further evidence -- the same "only one declaration" argument the
    attribute slice uses, one level up.
    """
    aliases: dict[str, list[Alias]] = {}
    for module in modules:
        attach_parents(module.tree)
        for func, owner in functions_with_owner(module.tree):
            declared = declarations.get(owner or "", set())
            if not declared:
                continue
            writes = _literal_self_writes(func, declared)
            if not writes:
                continue
            alias = Alias(owner, func.name)  # type: ignore[arg-type]
            alias.writes = writes
            aliases.setdefault(func.name, []).append(alias)
    return aliases


def _returned_values(
    func: ast.FunctionDef | ast.AsyncFunctionDef, hops: int = 0
) -> tuple[list[tuple[str, list[str]]], bool]:
    """``([(literal, conditions)], whether anything is unresolved)`` for *func*'s returns.

    Three shapes are read: a literal returned outright; a local returned after
    a guarded chain assigned it, which is what :func:`literal_values` answers;
    and a call returned outright, followed into what *it* returns (see
    :func:`_produced`) -- ``return self.other(...)`` hands back that method's
    literals under that method's guards.  Anything else -- a computed
    expression, ``True`` -- is counted as unresolved so the junction can say the
    value may be something other than what was listed, rather than implying the
    list is closed.

    Each condition is marked with the method it is tested in.  A reader who
    cannot tell the caller's guard from the helper's cannot tell a caller that
    never ran from a helper that decided otherwise, and the same holds one
    helper further in.
    """
    mark = f"  [in {func.name}()]"

    def marked(test: str) -> str:
        # One mark an entry: the callee's says the test is read where the caller cannot.
        return unmarked(test) + mark

    values: list[tuple[str, list[str]]] = []
    opaque = False
    for node in walk(func):
        if not isinstance(node, ast.Return) or node.value is None:
            continue
        if enclosing_function(node) is not func:
            continue
        returning = path_condition(node)
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            values.append((node.value.value, [marked(test) for test in returning]))
        elif isinstance(node.value, ast.Name):
            resolved = literal_values(func, node.value.id, at=node)
            if resolved:
                # The return's own guards as well: a chain assigning the local
                # can sit inside a branch of the function.
                values.extend(
                    (
                        literal,
                        [marked(test) for test in returning + [c for c in conditions if c not in returning]],
                    )
                    for literal, conditions, _line in resolved
                )
            else:
                opaque = True
        elif isinstance(node.value, ast.Call):
            followed = _produced(func, node.value, hops + 1)
            if followed is None:
                opaque = True
                continue
            deeper, deeper_opaque = followed
            values.extend(
                (literal, [marked(test) for test in returning] + inner) for literal, inner in deeper
            )
            opaque = opaque or deeper_opaque
        else:
            opaque = True
    return values, opaque


def _produced(
    func: ast.AST, call: ast.Call, hops: int = 0
) -> Optional[tuple[list[tuple[str, list[str]]], bool]]:
    """What *call* hands back, read in each function it lands in; ``None`` where it lands nowhere.

    Placed by the corpus's calls, so the helper is the one the receiver's
    class declares or inherits.  Found by name instead, a method defined once
    anywhere in the corpus answered for every ``self.<name>()``, whatever
    ``self`` was, and a name two classes declared answered for none unless the
    caller's family held one of them.  A call that may land somewhere the
    calls cannot name leaves the list open.
    """
    if hops > sql.RETURN_HOPS:
        return None
    where = sql.placed(func, call)
    if where is None or not where.targets:
        return None
    values: list[tuple[str, list[str]]] = []
    opaque = bool(where.unplaced)
    for target in where.targets:
        found, unresolved = _returned_values(target.func, hops)  # type: ignore[arg-type]
        values.extend(found)
        opaque = opaque or unresolved
    return values, opaque


def extract_producers(
    modules: list[SourceModule],
    map_id: str,
    derived_from: str,
    declarations: dict[str, set[str]],
    attributor: SpecAttributor,
) -> tuple[list[SubjectNode], list[JunctionNode], list[CoverageStat]]:
    """Extract a junction per ``spec.attr = self.helper(...)`` write.

    Restricted to a bare ``self`` receiver.  Widened to any call, the shape
    matches forty-two sites of which two touch a subject, and the rest are
    plugin lookups and config reads -- the same ratio that got the naming
    heuristic deleted.  Restricted, it reads the two that decide a status.

    **This slice owns the shape outright**, including the writes whose helper it
    cannot follow: those get a run-time outcome rather than nothing, so the
    attribute slice can stay out of the shape entirely instead of adding a
    second, weaker reading of the two writes resolved here.  ``explained``
    still counts only the resolved ones, so the coverage figure keeps saying how
    much of the shape was actually followed.
    """
    junctions: dict[str, JunctionNode] = {}
    coverage: list[CoverageStat] = []
    attributed: set[tuple[str, str]] = set()
    aliases = find_aliases(modules, declarations)
    memo: dict = {}

    for module in modules:
        attach_parents(module.tree)
        candidates = 0
        explained = 0
        for node in walk(module.tree):
            call = written_value(node)
            if not isinstance(call, ast.Call):
                continue
            if not (
                isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "self"
            ):
                continue
            targets = [t for t in targets_of(node) if isinstance(t, ast.Attribute)]
            if not targets:
                continue
            func = enclosing_function(node)
            owner_class = enclosing_class(node)
            for target in targets:
                spec_class = attributor.class_of(target.value, func, owner_class)
                if spec_class is None or target.attr not in declarations.get(spec_class, set()):
                    continue
                candidates += 1
                attributed.add((spec_class, target.attr))
                caller = path_condition(node)
                # What the field still holds where the object leaves: no branch
                # where it never does, an entry per later write that can replace it.
                replaced = field_replacements(
                    node, target.value, target.attr, call, setters_of(aliases, attributor, target.attr), memo
                )
                if replaced is None:
                    continue
                later = entries(replaced, caller)
                followed = _produced(func, call) if func is not None else None
                if followed is None or not followed[0]:
                    # The helper returns something this slice cannot follow --
                    # a computed value, or another call.  The writer is still
                    # the answer to "who set this", which is what localize and
                    # prune work from, so it is recorded as a run-time outcome.
                    _record(
                        junctions,
                        map_id=map_id,
                        derived_from=derived_from,
                        module=module,
                        node=call,
                        owner=f"{module.rel_path}::{func.name if func else '<module>'}",
                        spec_class=spec_class,
                        attribute=target.attr,
                        literal=f"runtime({call.func.attr}())",
                        condition=caller + later,
                        tier=2,
                    )
                    continue
                explained += 1
                produced, opaque = followed
                for literal, inner in produced:
                    _record(
                        junctions,
                        map_id=map_id,
                        derived_from=derived_from,
                        module=module,
                        node=call,
                        owner=f"{module.rel_path}::{func.name if func else '<module>'}",
                        spec_class=spec_class,
                        attribute=target.attr,
                        literal=literal,
                        condition=caller + inner + [entry for entry in later if entry not in inner],
                    )
                if opaque:
                    # Says the list is not closed rather than letting it read as
                    # complete: the helper has a return this slice cannot follow.
                    _record(
                        junctions,
                        map_id=map_id,
                        derived_from=derived_from,
                        module=module,
                        node=call,
                        owner=f"{module.rel_path}::{func.name if func else '<module>'}",
                        spec_class=spec_class,
                        attribute=target.attr,
                        literal=f"runtime({call.func.attr}())",
                        condition=caller + later,
                        tier=2,
                    )
        if candidates:
            coverage.append(
                CoverageStat(
                    slice_name=PRODUCER_SLICE,
                    file=module.rel_path,
                    candidates=candidates,
                    explained=explained,
                )
            )

    subjects = [
        SubjectNode(
            map_id=map_id,
            derived_from=derived_from,
            name=SubjectNode.make_name(spec_class, attribute),
            spec_class=spec_class,
            attribute=attribute,
            criteria=["return-alias"],
        )
        for spec_class, attribute in sorted(attributed)
    ]
    return subjects, list(junctions.values()), coverage


def setters_of(
    aliases: dict[str, list[Alias]], attributor: SpecAttributor, attribute: str
) -> Callable[[ast.Call], Optional[Callee]]:
    """What a call's method does to *attribute*: its writes of it, each with the method's tests on it.

    How a write of a field reads a setter call on the same object for the
    value it may replace (:func:`pathcond.field_replacements`).
    """

    def callee(call: ast.Call) -> Optional[Callee]:
        alias = _alias_at(aliases, call, attributor, enclosing_function(call), enclosing_class(call))
        if alias is None:
            return None
        writes = tuple(tuple(inner) for written, _value, inner, _tier in alias.writes if written == attribute)
        return Callee(alias.method, writes) if writes else None

    return callee


def _alias_at(
    aliases: dict[str, list[Alias]],
    call: ast.Call,
    attributor: SpecAttributor,
    func: Optional[ast.FunctionDef | ast.AsyncFunctionDef],
    owner_class: Optional[str],
) -> Optional[Alias]:
    """Return the alias a call resolves to, or ``None`` when the class is open."""
    if not isinstance(call.func, ast.Attribute):
        return None
    candidates = aliases.get(call.func.attr)
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    spec_class = attributor.class_of(call.func.value, func, owner_class)
    for alias in candidates:
        if alias.spec_class == spec_class:
            return alias
    return None


def extract(
    modules: list[SourceModule],
    map_id: str,
    derived_from: str,
    declarations: dict[str, set[str]],
    attributor: SpecAttributor,
) -> tuple[list[SubjectNode], list[JunctionNode], list[CoverageStat]]:
    """Extract a junction per call site of a write alias."""
    aliases = find_aliases(modules, declarations)
    junctions: dict[str, JunctionNode] = {}
    coverage: list[CoverageStat] = []
    attributed: set[tuple[str, str]] = set()
    memo: dict = {}

    for module in modules:
        attach_parents(module.tree)
        candidates = 0
        explained = 0
        for node in walk(module.tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr not in aliases:
                continue
            func = enclosing_function(node)
            candidates += 1
            alias = _alias_at(aliases, node, attributor, func, enclosing_class(node))
            if alias is None:
                continue
            explained += 1
            caller = path_condition(node)
            for attribute, value, inner, tier in alias.writes:
                attributed.add((alias.spec_class, attribute))
                # The call writes the field as an assignment would: no branch
                # where what it writes never leaves the object, an entry per
                # later write that can replace it.
                replaced = field_replacements(
                    site_of(node),
                    node.func.value,
                    attribute,
                    ast.Constant(value=value) if tier == 1 else None,
                    setters_of(aliases, attributor, attribute),
                    memo,
                )
                if replaced is None:
                    continue
                marked = [f"{test}  [in {alias.method}()]" for test in inner]
                _record(
                    junctions,
                    map_id=map_id,
                    derived_from=derived_from,
                    module=module,
                    node=node,
                    owner=f"{module.rel_path}::{func.name if func else '<module>'}",
                    spec_class=alias.spec_class,
                    attribute=attribute,
                    literal=value,
                    # The caller's conditions decide whether the alias runs; the
                    # alias's own decide whether running it changes anything.
                    # Both are required and they fail differently, so the second
                    # set says where it lives.
                    condition=caller + marked + [entry for entry in entries(replaced, caller) if entry not in marked],
                    tier=tier,
                )
        if candidates:
            coverage.append(
                CoverageStat(
                    slice_name=SLICE_NAME,
                    file=module.rel_path,
                    candidates=candidates,
                    explained=explained,
                )
            )

    subjects = [
        SubjectNode(
            map_id=map_id,
            derived_from=derived_from,
            name=SubjectNode.make_name(spec_class, attribute),
            spec_class=spec_class,
            attribute=attribute,
            criteria=["write-alias"],
        )
        for spec_class, attribute in sorted(attributed)
    ]
    return subjects, list(junctions.values()), coverage


def _record(
    junctions: dict[str, JunctionNode],
    *,
    map_id: str,
    derived_from: str,
    module: SourceModule,
    node: ast.Call,
    owner: str,
    spec_class: str,
    attribute: str,
    literal: str,
    condition: list[str],
    tier: int = 1,
) -> None:
    """Add one branch for what this call site settles."""
    subject = SubjectNode.make_name(spec_class, attribute)
    name = JunctionNode.make_name(map_id, subject, owner)
    junction = junctions.get(name)
    if junction is None:
        junction = JunctionNode(
            map_id=map_id,
            derived_from=derived_from,
            name=name,
            subject=subject,
            owner=owner,
            attribution=CERTAIN,
            anchor=Anchor(
                package=module.package,
                file=module.rel_path,
                line_start=node.lineno,
                line_end=node.end_lineno,
                blob_sha=module.blob_sha,
            ),
        )
        junctions[name] = junction
    known = {(b.outcome, tuple(b.path_condition)) for b in junction.branches}
    if (literal, tuple(condition)) in known:
        return
    junction.branches.append(
        Branch(
            outcome=literal,
            path_condition=condition,
            order=len(junction.branches),
            tier=tier,
            # The same ``node`` the anchor above is built from.  Left out until
            # now, and nothing said so: a positionless arm is still counted,
            # stored, offered as a candidate and compared by ``diff-map``,
            # because none of those read a line.  What does read it is the
            # derivation -- ``reading.lines`` is filled from here and nowhere
            # else, and a reading that ends up with none is not walked at all,
            # so the arm came back with no steps and no skeleton either.
            line=node.lineno,
        )
    )
