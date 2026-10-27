"""The write that happens somewhere other than where the value was decided.

JEDI is a read-modify-write system, and most of its writes are spelled the way
that implies::

    taskSpec.status = self.getFinalTaskStatus(taskSpec)   # decided here
    tmpLog.info(f"set task_status={taskSpec.status}")      # announced here
    self.taskBufferIF.updateTask_JEDI(taskSpec, {"jediTaskID": ...})   # written here

The proxy end of that is a generic flush::

    sqlU = f"UPDATE {schema}.JEDI_Tasks SET {taskSpec.bindUpdateChangesExpression()} "

whose ``SET`` clause the map cannot read, because there is nothing to read: the
columns are ``_changedAttrs``, which ``JediTaskSpec.__setattr__`` filled from
the caller's assignments.  So the write is not invisible, it is *displaced* --
the attribute slice already holds every decision at full coverage, and what was
missing is that the flush is where they land.

Two things follow, and only the second needs recording.

The decision is already a junction, so no new node is warranted here.  What is
not already recorded is the flush's own ``WHERE``: ``updateTask_JEDI`` appends
``status IN (:old_...)`` when its caller supplies ``oldStatus``, and twelve of
its twenty-two call sites do.  That is the same compare-and-set the SQL slice
records on a statement it can read whole, arriving in a different spelling --
and letting the two spellings disagree about one fact is the failure this map
has had to correct more than once.

Three readings keep it from over-claiming, each answering a way the naive rule
is wrong:

*Which columns the flush writes* is what the caller changed, since that is what
``_changedAttrs`` holds -- so a guard on a column this caller never touched is
row selection, not a race.

*Minus what the flush resets.*  ``updateTask_JEDI`` opens with
``taskSpec.resetChangedAttr("jediTaskID")``, which is the method saying outright
that it does not write the column its ``WHERE`` identifies rows by.

*Minus guards the caller did not ask for.*  The ``status IN`` fragment is
appended under ``if oldStatus is not None``, a path condition on a parameter,
so whether it applies is settled one step up at the call site -- the one hop
the backward-slicing rules allow for a formal argument.  Without it the ten
callers that pass no ``oldStatus`` would be told their write races when it
does not, and a caveat invented is worse here than a caveat missed: it is the
confirmed candidates it weakens.
"""

from __future__ import annotations

import ast
import re
from typing import NamedTuple, Optional

from bamboo.codemap.models import (
    Emit,
    JunctionNode,
    RowKeySource,
    SourceModule,
    SubjectNode,
)
from bamboo.codemap.panda import sql, sqltext, values
from bamboo.codemap.panda.attribution import SpecAttributor
from bamboo.codemap.panda.pathcond import (
    attach_parents,
    functions_with_owner,
    targets_of,
)
from bamboo.codemap.panda.recognizers import logfile, sqlwrite
from bamboo.codemap.panda.recognizers.emit import row_count_lines
from bamboo.codemap.panda.recognizers.trigger import sole_definitions

#: The spec classes declare this to build a ``SET`` clause out of whatever was
#: assigned to them.  Its presence is the method saying it is a flush.
FLUSH_EXPRESSION = "bindUpdateChangesExpression"

#: And this to take a column back out of that clause.
RESET_CHANGED = "resetChangedAttr"


class Flush(NamedTuple):
    """A method that persists whatever its caller changed on a spec."""

    spec_classes: frozenset[str]
    #: column -> (the predicate term, the parameters whose presence appends it)
    guards: dict[str, tuple[str, frozenset[str]]]
    #: columns the method takes back out of the clause before writing
    excluded: frozenset[str]
    parameters: tuple[str, ...]
    #: lines the method writes about how many rows its write changed
    rows_emits: tuple[Emit, ...] = ()


def _parameters(func: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[str, ...]:
    """Positional parameters in order, ``self`` included so indices line up."""
    return tuple(arg.arg for arg in func.args.args) + tuple(
        arg.arg for arg in func.args.kwonlyargs
    )


def _is_flush_call(node: object) -> bool:
    """Whether *node* is a ``spec.bindUpdateChangesExpression()`` call."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == FLUSH_EXPRESSION
    )


def _names_flush_call(source: str) -> bool:
    """Whether a hole's *source* -- the expression the run did not follow -- is a flush call."""
    try:
        return _is_flush_call(ast.parse(source, mode="eval").body)
    except SyntaxError:
        return False


def _flushed_set(run: sql.Execution, match: re.Match) -> bool:
    """Whether the ``SET`` of *match* is what a ``bindUpdateChangesExpression()`` call returned, as the run says.

    Its text came from the call -- a piece whose way passes it, or the hole
    that stands for it where the run did not follow it -- and nothing but
    whitespace came from anywhere else.  Where the build follows the call, the
    clause is the callee's loop: a hole where an attribute changed, nothing
    where none did.  The text alone cannot tell that nothing from a clause the
    method left empty: ``updateJob`` appends the ``SET`` of its
    ``JEDI_Datasets`` update under ``if cType == "hold"`` / ``elif cType ==
    "touse"`` with no ``else``, and on the third path the clause is empty --
    whose ``WHERE`` was then read as the flush's.
    """
    start = run.sql.upper().rfind("SET", match.start(), match.start(2)) + len("SET")
    end = match.end(2)
    rest = list(run.sql[start:end])
    called = False

    def came_from_the_call(begin: int, finish: int) -> None:
        nonlocal called
        called = True
        for at in range(max(begin, start), min(finish, end)):
            rest[at - start] = " "

    for at, hole in run.holes:
        if start <= at < end and _names_flush_call(hole.source):
            came_from_the_call(at, at + len("{}"))
    for begin, finish, way in run.pieces:
        if begin < end and start < finish and any(_is_flush_call(node) for node in way):
            came_from_the_call(begin, finish)
    return called and not "".join(rest).strip()


def _gating_parameters(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    run: sql.Execution,
    span: tuple[int, int],
    parameters: tuple[str, ...],
) -> Optional[frozenset[str]]:
    """Parameters whose value decides whether the term at *span* of *run* is appended, or ``None`` if unknown.

    The tests are those on the way of each piece holding the term
    (:attr:`sql.Execution.pieces`), as the SQL slice reads the guards of a
    statement it can read whole (``sqlwrite._guard_conditions``).  Looking for
    the term's text among the method's literals instead tied it to nothing
    where the method formats it -- ``updateTask_JEDI`` appends
    ``f"status IN ({old_status_var_names_str}) AND "`` -- so every caller was
    told the guard; and to a parameter tested above the same text elsewhere --
    ``registerTaskInOneShot_JEDI``'s ``jediTaskID=:jediTaskID``, unconditional
    in its flush, to ``taskParams``.

    ``None`` where no piece holds the term, or one was written under a test of
    another function, whose names say nothing here: the term is then not said,
    as the SQL slice leaves such a column out.
    """
    start, end = span
    ways = [way for begin, finish, way in run.pieces if begin < end and start < finish]
    said = [sqlwrite._tests_on_way(func, way) for way in ways]
    if not ways or any(one is None for one in said):
        return None
    return frozenset(
        name
        for name in parameters
        if name != "self"
        for tests in said
        if any(re.search(rf"\b{re.escape(name)}\b", test) for test in tests or ())
    )


def flush_methods(
    modules: list[SourceModule], attributor: SpecAttributor
) -> dict[str, Flush]:
    """Return ``{method name: what it flushes}`` for the generic-update idiom.

    Keyed by name, so restricted to names one module implements -- the same
    limit the trigger slice puts on a cross-module hop, for the same reason.
    """
    only_here = sole_definitions(modules)
    # The same resolver the SQL slice builds, built again here rather than
    # threaded through ``attach``: both halves of a ``row_precondition`` have
    # to be settled by one rule, and the rule is ``sql.settled_precondition``
    # -- what each caller supplies is only the table of declared constants.
    settle = values.resolver()
    declared = logfile.declared_files(modules)
    inherited = logfile.inherited_files(modules, declared)
    found: dict[str, Flush] = {}
    for module in modules:
        # Whether a fragment is appended under a test is half of what this
        # reader needs, and that is unreadable without parent links.  Attached
        # here rather than relied on from an earlier slice: a recognizer that
        # works only when another ran first fails silently when it does not.
        attach_parents(module.tree)
        for func, owner in functions_with_owner(module.tree):
            if only_here.get(func.name) != module.rel_path:
                continue
            classes: set[str] = set()
            excluded: set[str] = set()
            for node in ast.walk(func):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                    continue
                if node.func.attr == FLUSH_EXPRESSION:
                    spec = attributor.class_of(node.func.value, func, owner)
                    if spec:
                        classes.add(spec)
                elif node.func.attr == RESET_CHANGED and node.args:
                    argument = node.args[0]
                    if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                        excluded.add(argument.value)
            if not classes:
                continue
            parameters = _parameters(func)
            guards: dict[str, tuple[str, frozenset[str]]] = {}
            for run in sql.executions(func):
                for match in sqltext._UPDATE.finditer(run.sql):
                    if not _flushed_set(run, match):
                        # A readable ``SET`` is the SQL slice's business; this
                        # reader exists for the clause the spec writes.
                        continue
                    for term_match in sqltext._TERM.finditer(run.sql, match.end(2)):
                        raw, column = term_match.group(1), term_match.group(2)
                        if column in guards:
                            continue
                        gating = _gating_parameters(func, run, term_match.span(1), parameters)
                        if gating is None:
                            continue
                        term = re.sub(r"\s+", " ", raw).strip()
                        # Settled here for the same reason the SQL slice
                        # settles its own: these terms land in the *same*
                        # ``row_precondition`` list, and one half spelling the
                        # bind while the other spells the value would leave the
                        # list unreadable and the gate comparing it against a
                        # reader's values half blind.
                        settled, _done = sql.settled_precondition(
                            term, func, run.varmap, run.window, settle, at=run
                        )
                        guards[column] = (settled, gating)
            found[func.name] = Flush(
                spec_classes=frozenset(classes),
                guards=guards,
                excluded=frozenset(excluded),
                parameters=parameters,
                rows_emits=tuple(
                    row_count_lines(
                        func, logfile.files_of(module.rel_path, declared, inherited)
                    )
                ),
            )
    return found


def _supplied(call: ast.Call, parameter: str, parameters: tuple[str, ...]) -> bool:
    """Whether this call passes *parameter* something other than ``None``."""
    for keyword in call.keywords:
        if keyword.arg == parameter:
            return not (
                isinstance(keyword.value, ast.Constant) and keyword.value.value is None
            )
        if keyword.arg is None:
            # ``**kwargs``: what it holds is not readable here, so the guard
            # is left in rather than ruled out.
            return True
    if parameter in parameters:
        # ``self`` is in the signature and never in the call's arguments.
        index = parameters.index(parameter) - (1 if parameters and parameters[0] == "self" else 0)
        if 0 <= index < len(call.args):
            argument = call.args[index]
            return not (isinstance(argument, ast.Constant) and argument.value is None)
    return False


def _changed_attributes(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    owner: Optional[str],
    attributor: SpecAttributor,
    spec_classes: frozenset[str],
) -> dict[str, set[str]]:
    """``{spec class: attributes assigned on it}`` inside *func*."""
    changed: dict[str, set[str]] = {}
    for node in ast.walk(func):
        for target in targets_of(node):
            if not isinstance(target, ast.Attribute):
                continue
            spec = attributor.class_of(target.value, func, owner)
            if spec in spec_classes:
                changed.setdefault(spec, set()).add(target.attr)
    return changed


def attach(
    junctions: list[JunctionNode],
    modules: list[SourceModule],
    attributor: SpecAttributor,
    row_keys: Optional[list[RowKeySource]] = None,
    key_fields: Optional[dict[str, frozenset[str]]] = None,
) -> int:
    """Give each caller's branches the row its flush needed.  Returns how many.

    *row_keys*, where given, gets one :class:`RowKeySource` per guarded flush:
    the row a flush writes is the spec it is handed, so where that spec came
    from -- a parameter, a field, or this function's own query -- is who named
    the row.  ``TaskBuffer.updateJobs`` flushes each of the ``jobs`` its caller
    passed, and its guard requires ``defined`` of rows it never selected.
    *key_fields* says which fields of each spec name its row (:func:`row_key_fields`).

    Applied to every branch of a matching junction rather than to the writes
    that provably reach the flush call: a method that changes a spec and then
    flushes it flushes what it changed, and pairing each assignment with the
    call that dominates it would be a dominance analysis for a distinction the
    corpus does not appear to make.  Stated because it over-reads: a function
    that flushes on one path and not another has the caveat on both.
    """
    flushes = flush_methods(modules, attributor)
    if not flushes:
        return 0
    by_owner: dict[str, list[JunctionNode]] = {}
    for junction in junctions:
        by_owner.setdefault(junction.owner, []).append(junction)

    annotated = 0
    for module in modules:
        for func, owner in functions_with_owner(module.tree):
            here = by_owner.get(f"{module.rel_path}::{func.name}")
            if not here:
                continue
            for node in ast.walk(func):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                    continue
                flush = flushes.get(node.func.attr)
                if flush is None:
                    continue
                changed = _changed_attributes(func, owner, attributor, flush.spec_classes)
                for spec, attributes in changed.items():
                    written = attributes - flush.excluded
                    guards = sorted(
                        term
                        for column, (term, gating) in flush.guards.items()
                        if column in written
                        and all(_supplied(node, p, flush.parameters) for p in gating)
                    )
                    if not guards and not flush.rows_emits:
                        continue
                    if row_keys is not None and guards:
                        row_keys.append(
                            _row_key_source(node, func, owner, spec, flush, guards, module, attributor, key_fields or {})
                        )
                    for junction in here:
                        if junction.subject not in {
                            SubjectNode.make_name(spec, attribute) for attribute in written
                        }:
                            continue
                        for branch in junction.branches:
                            missing = [g for g in guards if g not in branch.row_precondition]
                            if missing:
                                branch.row_precondition = sorted(
                                    branch.row_precondition + missing
                                )
                                annotated += 1
                            # The line saying whether the row actually moved is
                            # written by the flush, in the flush's own log, and
                            # the code that chose the value never sees it.
                            for line in flush.rows_emits:
                                if any(e.template == line.template for e in branch.emits):
                                    continue
                                branch.emits.append(line)
    return annotated


def _row_key_source(
    call: ast.Call,
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    owner: Optional[str],
    spec: str,
    flush: Flush,
    guards: list[str],
    module: SourceModule,
    attributor: SpecAttributor,
    key_fields: dict[str, frozenset[str]],
) -> RowKeySource:
    """The spec a guarded flush is handed; whether the caller handed it here is read after the reach record
    (:func:`sqlwrite.attach_key_sources`)."""
    passed: list[ast.expr] = [
        argument
        for argument in (*call.args, *(keyword.value for keyword in call.keywords))
        if attributor.class_of(argument, func, owner) == spec
    ]
    source = RowKeySource(
        owner=f"{module.rel_path}::{func.name}",
        file=module.rel_path,
        statement=call.lineno,
        columns=sorted(column for column, (term, _gating) in flush.guards.items() if term in guards),
        keys=[ast.unparse(argument) for argument in passed],
    )
    # What names the spec's row: the object itself, or a field holding its table's key (``.jediTaskID``) -- or any
    # field, where something fills the object by name (``taskSpec.pack(row)``).
    steps = key_fields.get(spec, frozenset()) | {sqlwrite.ANY_FIELD}
    source._filled = (func, module.rel_path, [(argument, call, steps) for argument in passed])
    return source


def row_key_fields(attributor: SpecAttributor) -> dict[str, frozenset[str]]:
    """``{spec class: the steps of its fields that name its row}`` -- its tables' key columns as the class spells
    them, ``{"JediTaskSpec": {".jediTaskID"}}``.

    Off an attributor that learned which tables hold which class (:meth:`SpecAttributor.learn_table_classes`) and
    inside :func:`sqltext.with_schema` (:func:`sqltext.table_key`).  The one :func:`attach` reads callers with does
    not learn the tables, and a spec with none learned names its row by no field.
    """
    found: dict[str, frozenset[str]] = {}
    for spec in attributor.declared_classes():
        keyed = set().union(*(sqltext.table_key(table) for table in attributor.tables_of(spec)))
        steps = frozenset("." + attribute for attribute in attributor.declared_attributes(spec) if attribute.lower() in keyed)
        if steps:
            found[spec] = steps
    return found
