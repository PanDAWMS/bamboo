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

from bamboo.codemap.astwalk import walk
from bamboo.codemap.calls import own_nodes
from bamboo.codemap.models import (
    KEY_IN_CALLEE,
    KEY_NO_BIND,
    Emit,
    JunctionNode,
    KeyFrom,
    RowKeySource,
    SourceModule,
    SubjectNode,
)
from bamboo.codemap.panda import sql, sqltext, values
from bamboo.codemap.panda.attribution import SpecAttributor
from bamboo.codemap.panda.pathcond import (
    attach_parents,
    functions_with_owner,
    site_of,
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
    #: the method, ``file::name``
    owner: str = ""
    #: ``(class, line, table)`` of each statement flushing a spec of the class: whose own records name its rows
    flushed: frozenset[tuple[str, int, str]] = frozenset()
    #: the dict parameter whose items make those statements' ``WHERE`` (``updateTask_JEDI``'s ``criteria``), or ""
    criteria: str = ""
    #: column -> (the term as written, the method, {bind: its values in the method}) for a guard the method does not
    #: settle itself -- ``status IN (:old___in)`` from its ``oldStatus`` -- settled at each call (:func:`_guard_at`)
    guard_binds: dict = {}


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


def _flushed_classes(
    run: sql.Execution,
    match: re.Match,
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    owner: Optional[str],
    attributor: SpecAttributor,
) -> set[str]:
    """The classes of the specs whose ``bindUpdateChangesExpression()`` the ``SET`` of *match* is: the call's object, in
    the hole standing for it or on the way of the piece it gave."""
    start = run.sql.upper().rfind("SET", match.start(), match.start(2))
    end = match.end(2)
    made = [
        ast.parse(hole.source, mode="eval").body.func  # type: ignore[attr-defined]
        for at, hole in run.holes
        if start <= at < end and _names_flush_call(hole.source)
    ]
    made += [
        node.func
        for begin, finish, way in run.pieces
        if begin < end and start < finish
        for node in way
        if _is_flush_call(node)
    ]
    return {spec for spelled in made if (spec := attributor.class_of(spelled.value, func, owner))}


def _criteria(run: sql.Execution, match: re.Match, parameters: tuple[str, ...], dicts: frozenset[str]) -> str:
    """The dict parameter whose items the ``WHERE`` of *match* is made of, or ``""``: a hole of the ``WHERE`` written
    in a loop over it -- ``updateTask_JEDI``'s ``for tmpKey, tmpVal in criteria.items(): sql += f"{tmpKey}={crKey}
    AND "``, its values bound by the keys.  The row is then the one the dict its caller hands it names, whatever the
    spec beside it holds."""
    for at, hole in run.holes:
        if at < match.end(2):
            continue
        node = hole.stmt
        while isinstance(node, ast.AST) and not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if isinstance(node, (ast.For, ast.AsyncFor)):
                looped, known = node.iter, dicts
                if (
                    isinstance(looped, ast.Call)
                    and isinstance(looped.func, ast.Attribute)
                    and looped.func.attr in ("items", "keys")
                ):
                    looped, known = looped.func.value, frozenset(parameters)
                if isinstance(looped, ast.Name) and looped.id in parameters and looped.id in known:
                    return looped.id
            node = getattr(node, "parent", None)
    return ""


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
            for node in walk(func):
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
            guard_binds: dict[str, tuple] = {}
            flushed: set[tuple[str, int, str]] = set()
            criteria = ""
            dicts = sqlwrite._dict_names(func)
            for run in sql.executions(func):
                for match in sqltext._UPDATE.finditer(run.sql):
                    if not _flushed_set(run, match):
                        # A readable ``SET`` is the SQL slice's business; this
                        # reader exists for the clause the spec writes.
                        continue
                    table = match.group(1).split(".")[-1]
                    flushed |= {
                        (spec, run.call.lineno, table) for spec in _flushed_classes(run, match, func, owner, attributor)
                    }
                    criteria = criteria or _criteria(run, match, parameters, dicts)
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
                        settled, done = sql.settled_precondition(
                            term, func, run.varmap, run.window, settle, at=run
                        )
                        guards[column] = (settled, gating)
                        if not done and run.varmap is not None:
                            guard_binds[column] = (
                                term,
                                func,
                                {
                                    key: tuple(sql.bound_values(func, run.varmap, key, run.window, at=run))
                                    for key in sql.precondition_binds(term)
                                },
                            )
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
                owner=f"{module.rel_path}::{func.name}",
                flushed=frozenset(flushed),
                criteria=criteria,
                guard_binds=guard_binds,
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
    for node in walk(func):
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
    key_froms: Optional[list[KeyFrom]] = None,
) -> int:
    """Give each caller's branches the row its flush needed.  Returns how many.

    *row_keys*, where given, gets one :class:`RowKeySource` per guarded flush:
    the row a flush writes is the one its call names -- the spec it is handed,
    or a value beside it the method takes the key from -- so where that came
    from -- a parameter, a field, or this function's own query -- is who named
    the row.  ``TaskBuffer.updateJobs`` flushes each of the ``jobs`` its caller
    passed, and its guard requires ``defined`` of rows it never selected.
    *key_froms*, where given, gets the :class:`KeyFrom` s of every flush call
    (:func:`_call_key_froms`), which the RowKeySource is read from, and must
    already hold the flush methods' own (:func:`sqlwrite.extract`).

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

    # The calls' records first, every function's: a call is read in the function it is made in, and the annotation
    # below walks a function's nested ones too.
    owned: dict[tuple[str, int], list[KeyFrom]] = {}
    for record in key_froms or ():
        owned.setdefault((record.owner, record.statement), []).append(record)
    called: dict[int, list[KeyFrom]] = {}
    for module in modules:
        for func, owner in functions_with_owner(module.tree) if key_froms is not None else ():
            for call in sqlwrite._own_calls(func):
                flush = flushes.get(call.func.attr) if isinstance(call.func, ast.Attribute) else None
                if flush is not None:
                    called[id(call)] = _call_key_froms(call, func, owner, module.rel_path, flush, owned, attributor)
                    key_froms.extend(called[id(call)])  # type: ignore[union-attr]

    # A guard's binds as the SQL slice settles its own (values.bind_settler), read at each call.
    settle = values.bind_settler(sql._CALLS)
    annotated = 0
    for module in modules:
        for func, owner in functions_with_owner(module.tree):
            here = by_owner.get(f"{module.rel_path}::{func.name}")
            if not here:
                continue
            uses = []
            for node in walk(func):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                    continue
                flush = flushes.get(node.func.attr)
                if flush is None:
                    continue
                changed = _changed_attributes(func, owner, attributor, flush.spec_classes)
                for spec, attributes in changed.items():
                    written = attributes - flush.excluded
                    applied = [
                        (column, term, _guard_at(node, func, flush, column, term, settle))
                        for column, (term, gating) in sorted(flush.guards.items())
                        if column in written
                        and all(_supplied(node, p, flush.parameters) for p in gating)
                    ]
                    uses.append((node, flush, spec, written, applied))
            # A guard settled at a call is said where every guarded flush call of the function settles the column
            # alike, and on the branches that call writes: the calls are alternatives, and a branch's guards a
            # conjunction.  TaskRefiner.runImpl hands registerTaskInOneShot_JEDI a status tested to be registered or
            # staged, and updateTask_JEDI the status it read (`[taskStatus]`), and either guard on every branch
            # would drop the other call's arms; read as the method has it, the bind drops none.
            at_calls: dict[str, set[str]] = {}
            for _node, _flush, _spec, _written, applied in uses:
                for column, _term, settled in applied:
                    at_calls.setdefault(column, set()).add(settled)
            for node, flush, spec, written, applied in uses:
                guards = sorted(term for _column, term, _settled in applied)
                if not guards and not flush.rows_emits:
                    continue
                if row_keys is not None and guards:
                    row_keys.append(
                        _row_key_source(
                            node, func, spec, [column for column, _t, _s in applied], module, called.get(id(node), [])
                        )
                    )
                alike = {
                    column: settled
                    for column, term, settled in applied
                    if settled != term and at_calls.get(column) == {settled}
                }
                reached = _reached_lines(node, func, owner, spec, written, attributor) if alike else set()
                for junction in here:
                    if junction.subject not in {
                        SubjectNode.make_name(spec, attribute) for attribute in written
                    }:
                        continue
                    attribute = junction.subject.rpartition(".")[2]
                    for branch in junction.branches:
                        mine = any(
                            (attribute, line) in reached
                            for line in (branch.line, *branch.other_lines)
                            if line is not None
                        )
                        said = sorted(
                            alike[column] if mine and column in alike else term for column, term, _settled in applied
                        )
                        missing = [g for g in said if g not in branch.row_precondition]
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


def _reached_lines(
    call: ast.Call,
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    owner: Optional[str],
    spec: str,
    attributes: set[str],
    attributor: SpecAttributor,
) -> set[tuple[str, int]]:
    """``(attribute, line)`` of each assignment on a spec *call* is handed that reaches it -- what the call's
    records write (:func:`_call_key_froms`, :func:`sqlwrite._fields_at`), read here because those are filled after
    the reach record."""
    found: set[tuple[str, int]] = set()
    for obj in (
        argument
        for argument in (*call.args, *(keyword.value for keyword in call.keywords))
        if not isinstance(argument, ast.Starred) and attributor.class_of(argument, func, owner) == spec
    ):
        for attribute in attributes:
            found |= {(attribute, node.lineno) for node in sqlwrite._fields_at(func, obj, attribute, [call], spec)}
    return found


def _guard_at(
    call: ast.Call,
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    flush: Flush,
    column: str,
    term: str,
    settle,
) -> str:
    """A flush's guard as *call* has it: one the method settles itself as it is, else its binds filled by the call --
    each a parameter's part read as the same part of the call's argument (:func:`_rerooted`), settled in the caller
    (*settle*, and a test above the call saying what a name equals: :func:`sql.tested_values`); the method's own value
    as the method settles it.  ``JobGenerator.runImpl`` hands ``updateTask_JEDI`` ``oldStatus=JediTaskSpec.
    statusForJobGenerator() + ["pending"]``: its guard is that list, not the bind -- and another caller's list is not
    its.  One bind not settled leaves the guard as the method has it (``TaskRefiner.runImpl``'s ``[taskStatus]``, the
    status it read off the row: a compare-and-set)."""
    held = flush.guard_binds.get(column)
    if held is None:
        return term
    raw, method, binds = held
    at = site_of(call)
    plain = values.resolver()

    def held_by(key: str) -> Optional[set[str]]:
        found: set[str] = set()
        for bind in binds.get(key, ()):
            root = sqlwrite._root(bind.value)
            handed = (
                _argument(call, root.id, flush.parameters)
                if root is not None and root.id in flush.parameters[1:]
                else None
            )
            if handed is not None:
                moved = _rerooted(bind.value, root.id, handed)  # type: ignore[union-attr]
                got = (settle(moved, func) or sql.tested_values(moved, at, func)) if moved is not None else []
            else:
                got = plain(bind.value, method)
            if not got:
                return None
            found.update(got)
        return found or None

    settled, done = sql.settled_term(raw, held_by)
    return settled if done else term


def _argument(call: ast.Call, parameter: str, parameters: tuple[str, ...]) -> Optional[ast.expr]:
    """What this call passes *parameter*, or ``None`` -- as :func:`_supplied` finds it."""
    for keyword in call.keywords:
        if keyword.arg == parameter:
            return keyword.value
    if parameter in parameters:
        index = parameters.index(parameter) - (1 if parameters and parameters[0] == "self" else 0)
        if 0 <= index < len(call.args) and not isinstance(call.args[index], ast.Starred):
            return call.args[index]
    return None


def _rerooted(value: ast.expr, parameter: str, argument: ast.expr) -> Optional[ast.expr]:
    """*value*, a part of the method's *parameter* (``job.PandaID``), as the caller holds it: the same part of the
    *argument* the call hands it (``jobSpec.PandaID``) -- the argument's elements, where the part is the whole of a
    list written into the call (``oldStatus=["assigning"]``, opened into an ``IN`` list) -- or ``None`` where
    *value* is no part of *parameter*."""
    steps: list[ast.expr] = []
    node: ast.expr = value
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        steps.append(node)
        node = node.value
    if not (isinstance(node, ast.Name) and node.id == parameter):
        return None
    built = argument
    for step in reversed(steps):
        if isinstance(step, ast.Attribute):
            built = ast.copy_location(ast.Attribute(value=built, attr=step.attr, ctx=ast.Load()), argument)
        else:
            built = ast.copy_location(ast.Subscript(value=built, slice=step.slice, ctx=ast.Load()), argument)
        built.parent = getattr(argument, "parent", None)  # type: ignore[attr-defined]
    return built


def _call_key_froms(
    call: ast.Call,
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    owner: Optional[str],
    rel_path: str,
    flush: Flush,
    owned: dict[tuple[str, int], list[KeyFrom]],
    attributor: SpecAttributor,
) -> list[KeyFrom]:
    """The :class:`KeyFrom` s of a flush call, one per term naming the row it writes for each spec it is handed: the
    method's own records of the statement it flushes the spec's class into (*owned*), each bind of a parameter's
    part read as the same part of the call's argument for it (:func:`_rerooted`), on a column the caller does not
    write (a term on one is its compare-and-set) and where the call appends the term; and, where the method's
    ``WHERE`` is made of a dict parameter's items, one per key of the dict the call hands it.  A term the method
    fills only with values of its own -- its rows, its ``self`` -- is :data:`KEY_IN_CALLEE`.  Its object is the spec,
    and what it writes is read after the reach record: the caller's assignments on the spec that reach the call."""
    caller = f"{rel_path}::{func.name}"
    found: dict[tuple[str, str, str], KeyFrom] = {}

    def put(table: str, term: str, spelled: str, spec: str, binds: list, unread: str) -> None:
        record = found.get((table, term, spelled))
        if record is None:
            record = found[(table, term, spelled)] = KeyFrom(
                owner=caller, file=rel_path, statement=call.lineno, table=table, term=term, object=spelled
            )
            record._binds = (func, rel_path, [])
            record._object = (spelled, spec, [call], flush.excluded)
        record._binds[2].extend(binds)
        record.unread = "" if record._binds[2] else unread

    at = site_of(call)
    for spec in sorted(flush.spec_classes):
        statements = sorted((line, table) for one, line, table in flush.flushed if one == spec)
        for obj in (
            argument
            for argument in (*call.args, *(keyword.value for keyword in call.keywords))
            if not isinstance(argument, ast.Starred) and attributor.class_of(argument, func, owner) == spec
        ):
            spelled = ast.unparse(obj)
            assigned = {
                target.attr
                for node in own_nodes(func)
                for target in targets_of(node)
                if isinstance(target, ast.Attribute) and ast.unparse(target.value) == spelled
            }
            written = {
                attribute.lower()
                for attribute in assigned - flush.excluded
                if sqlwrite._fields_at(func, obj, attribute, [call], spec)
            }
            given = _argument(call, flush.criteria, flush.parameters) if flush.criteria else None
            for _line, table in statements[:1] if given is not None else ():
                if not isinstance(given, ast.Dict):
                    put(table, f"{flush.criteria}={ast.unparse(given)}", spelled, spec, [sql.Bind(given, at)], "")
                    continue
                for key, value in zip(given.keys, given.values, strict=True):
                    named = key.value if isinstance(key, ast.Constant) and isinstance(key.value, str) else None
                    if named is not None and named.lower() not in written:
                        put(table, f'{named}={flush.criteria}["{named}"]', spelled, spec, [sql.Bind(value, at)], "")
            for line, table in statements:
                for record in owned.get((flush.owner, line), ()):
                    if record._insert is not None or record.table.lower() != table.lower():
                        continue
                    column = re.match(r"\s*(\w+)", record.term)
                    if column is None or column.group(1).lower() in written:
                        continue
                    gating = next(
                        (
                            gating
                            for name, (_term, gating) in flush.guards.items()
                            if name.lower() == column.group(1).lower()
                        ),
                        frozenset(),
                    )
                    if not all(_supplied(call, parameter, flush.parameters) for parameter in gating):
                        continue
                    binds, own = [], False
                    for bind in (record._binds or (None, None, []))[2]:
                        root = sqlwrite._root(bind.value)
                        handed = (
                            _argument(call, root.id, flush.parameters)
                            if root is not None and root.id in flush.parameters[1:]
                            else None
                        )
                        moved = (
                            _rerooted(bind.value, root.id, handed)  # type: ignore[union-attr]
                            if handed is not None
                            else None
                        )
                        if moved is None:
                            own = True
                            continue
                        values = moved.elts if isinstance(moved, (ast.List, ast.Tuple, ast.Set)) else [moved]
                        binds.extend(sql.Bind(one, at) for one in values if not isinstance(one, ast.Starred))
                    unread = KEY_IN_CALLEE if own else record.unread or KEY_NO_BIND
                    put(record.table, record.term, spelled, spec, binds, unread)
    return list(found.values())


def _row_key_source(
    call: ast.Call,
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    spec: str,
    columns: list[str],
    module: SourceModule,
    records: list[KeyFrom],
) -> RowKeySource:
    """A guarded flush's row keys, the terms of its call's records for *spec* (:func:`_call_key_froms`); whether the
    caller handed it them is read after the reach record (:func:`sqlwrite.attach_key_sources`), from the same
    values: ``updateTask_JEDI`` names its row by the criteria it is handed, not by the spec beside them."""
    mine = [record for record in records if record._object is not None and record._object[1] == spec]
    source = RowKeySource(
        owner=f"{module.rel_path}::{func.name}",
        file=module.rel_path,
        statement=call.lineno,
        columns=sorted(columns),
        keys=[record.term for record in mine],
    )
    # The key's value itself, as the records hold it: a field of the spec handed in (``jobSpec.PandaID``), a value of
    # the criteria (``taskSpec.jediTaskID``).
    source._filled = (
        mine[0]._binds[0] if mine else func,
        module.rel_path,
        [(bind.value, call, frozenset()) for record in mine for bind in record._binds[2]],
    )
    return source
