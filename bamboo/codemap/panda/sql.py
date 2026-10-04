"""Reading SQL out of Python.

PanDA writes most of its state through SQL, not attribute assignment, so the
progress slice is mostly here.  The shape is regular enough to read but not
regular enough to read naively::

    sqlU = f"UPDATE {panda_config.schemaJEDI}.JEDI_Tasks "
    sqlU += "SET status=:status,modificationTime=:updateTime,"
    sqlU += "lockedBy=NULL,frozenTime=:frozenTime "
    sqlU += " WHERE jediTaskID=:jediTaskID "
    ...
    varMap[":status"] = taskStatus
    self.cur.execute(sqlU + comment, varMap)

Four things follow, each of which cost a wrong assumption to learn:

**The statement has to be reassembled.**  It is built by ``=`` then a run of
``+=``, often with an f-string holding the schema, and interleaved with other
statements being built in the same function.  Reassembly is per variable name.

**Reassembly is the wrong unit for reading a branch.**  The fragments are
appended conditionally -- ``getTasksToExecCommand_JEDI`` appends either ``SET
status=:status`` or ``SET status=oldStatus`` -- so the ``if`` that explains the
value is on the fragment, not on the statement.  ``fragments`` keeps them.

**The bind key does not say whether it is a write.**  ``:status`` appears in
``SET status=:status`` *and* in another statement's ``WHERE status=:status``
in the same function -- the flagship ``updateTaskStatusByContFeeder_JEDI`` does
exactly this.  Only the ``SET`` clause distinguishes a write from a predicate,
so the statement is required; the bind key alone is not enough.

**Which spec a table holds is not declared anywhere**, so it is inferred from
the column names -- see ``SpecAttributor.learn_table_classes``.

What the reassembled text says -- tables, columns, values, order -- is read
by :mod:`~bamboo.codemap.panda.sqltext`.
"""

from __future__ import annotations

import ast
import contextlib
import itertools
import logging
import math
import re
import string
from typing import TYPE_CHECKING, Iterator, NamedTuple, Optional

from bamboo.codemap.panda import sqltext
from bamboo.codemap.panda.pathcond import (
    BIND_ANNOTATED,
    BIND_ASSIGN,
    BIND_FOR,
    UNDECIDED,
    ancestors,
    attach_parents,
    bindings,
    can_reach,
    decided,
    enclosing_function,
    loops_over,
    parted,
    path_condition,
    ruled_out_by,
    surviving_writes,
    targets_of,
    written_value,
)
from bamboo.codemap.panda.values import blank_fields, rendered_text

if TYPE_CHECKING:
    from pathlib import Path

    from bamboo.codemap.calls import Calls, Placed

logger = logging.getLogger(__name__)

# A run with this many branch combinations is folded rather than split.  Set
# well above the corpus's worst case (12, in ``propagateResultToJEDI``) so it is
# a guard against a pathological method rather than a working limit.
_MAX_BRANCH_VARIANTS = 24


class Span(NamedTuple):
    """Where one fragment landed in the statement it helped build.

    The unit an inline value is anchored at.  A statement is assembled from
    fragments and each fragment carries its own path condition, so the fragment
    holding the value is the one that says *why* it was written -- but which
    fragment that is cannot be decided fragment by fragment, because an
    ``INSERT``'s column list and value list are two parenthesised runs and the
    correspondence between them is positional.  Reading the assembled variant
    and mapping the match back here answers both at once.
    """

    node: ast.stmt
    start: int
    end: int


class Stop(NamedTuple):
    """Somewhere the reading of one statement stopped, and what it stopped on.

    The reassembly is regex over rendered strings and always had places it
    could not follow; what it did not have was a way to say so.  A fragment
    with no text of its own was dropped and the statement read on without it,
    which is how ``getJobs``'s ``WHERE`` -- every condition the pilot's job
    request is answered by -- came to be missing from the map with nothing
    anywhere saying it was.

    *source* is the expression, unparsed, not a class of expression.  A
    consumer told "a fragment does not render" knows only that something is
    wrong; told ``build_where_clause()`` or
    ``get_sql_IN_bind_variables(statList, prefix=':stat')`` it knows which
    function to open, and that is the whole point of saying it out loud.
    """

    line: int
    source: str
    reason: str


#: A ``+=`` whose right-hand side says nothing about its own text.  Two shapes
#: in the corpus: a name another function returned (``sqlP += sql_where_clause``)
#: and a call that builds the fragment (``sql2 += JobSpec.bindValuesExpression()``).
FRAGMENT_NOT_RENDERED = "a fragment of the statement has no text of its own"

#: ``AND jobStatus IN ({})``.  The column is readable and the placeholders are
#: not, because the code spells them with :func:`get_sql_IN_bind_variables`
#: over a list.  The statement is not truncated -- it is there, and it selects
#: on values nobody can name from this function alone.
BINDS_ARE_A_HOLE = "the WHERE names its bind variables through an interpolation"

#: ``WHERE {}=:{}`` -- ``queryDatasetWithMap`` builds the column name from the
#: key of a mapping it was handed.  A different answer from the one above: the
#: reader has no column to go looking for, let alone a value.
COLUMN_IS_A_HOLE = "the WHERE names its column through an interpolation"

#: Where the binds go, as opposed to where the schema does.  ``{}.JEDI_Tasks``
#: is a hole too and is not a gap: reassembly blanks the schema on purpose.
_HOLE = re.compile(r"\{\d*\}")


def _hole_stops(sql: str, line: int) -> tuple[Stop, ...]:
    """One :class:`Stop` per ``WHERE`` term the reassembly left a hole in.

    Read off the assembled text rather than off a fragment, because that is
    where it is decidable, and the three holes a ``WHERE`` can carry are three
    different answers:

    * **The schema**, ``{}.JEDI_Datasets`` -- blanked on purpose, and not a
      gap.  Stripped first, and stripping it is what separates the other two
      from the 19 statements whose only hole is a schema inside a subquery.
    * **The placeholders**, ``jobStatus IN ({})`` -- the column is readable
      and the values are at whichever call site supplied the list.
    * **The column**, ``{}=:{}`` -- the statement names its column from a
      mapping key, which :func:`key_column_census` counts from the other end.

    The term is kept whole, ``AND``/``OR`` to ``AND``/``OR``, so the consumer
    is told which column it is about.
    """
    clause = _SCHEMA_HOLE.sub("", sqltext.where_clause(sql))
    if not _HOLE.search(clause):
        return ()
    found: set[Stop] = set()
    for term in _CLAUSE_JOIN.split(clause):
        if not _HOLE.search(term):
            continue
        text = _balanced(" ".join(term.split()))
        found.add(
            Stop(
                line,
                text,
                COLUMN_IS_A_HOLE if _HOLE.match(text) else BINDS_ARE_A_HOLE,
            )
        )
    return tuple(sorted(found))


def _balanced(term: str) -> str:
    """Trim the parentheses splitting on ``AND`` left unmatched, and only those.

    ``(cloud IS NULL OR cloud IN ({}))`` splits into a leading ``(cloud IS
    NULL`` and a trailing ``cloud IN ({}))``, and a blanket ``strip("()")``
    would take the ``)`` that closes ``IN`` as well -- the term would read
    ``jobStatus IN ({}``, which is not what the code says and not something a
    reader can search the source for.
    """
    while term.startswith("(") and term.count("(") > term.count(")"):
        term = term[1:].lstrip()
    while term.endswith(")") and term.count(")") > term.count("("):
        term = term[:-1].rstrip()
    return term


#: ``{}.`` -- a hole that qualifies a name is the schema, wherever it appears.
_SCHEMA_HOLE = re.compile(r"\{\d*\}\.")

#: What separates one term of a ``WHERE`` from the next, at any depth.  Depth
#: is not tracked: a term of a subquery is still a term whose hole the reader
#: cannot fill, and splitting it out names the column rather than the subquery.
_CLAUSE_JOIN = re.compile(r"\b(?:AND|OR)\b", re.IGNORECASE)


#: The helper every ``IN`` list in the corpus is built with: it returns the
#: placeholders as one string and their values as a map, and both are spliced
#: in -- ``f"jobStatus IN ({names})"`` and ``varMap.update(values)``.  It is
#: pandacommon's, outside the corpus, and what it does is stated here once.
IN_LIST_HELPER = "get_sql_IN_bind_variables"


class InList(NamedTuple):
    """One ``IN`` list: the placeholder it stands for, and what fills it."""

    names: str
    values_map: str
    values: ast.expr
    bind: str
    site: ast.stmt


def in_lists(func: ast.AST) -> dict[str, InList]:
    """``{name holding the placeholders: InList}`` for each list *func* builds.

    ``stat_var_names_str, stat_var_map = get_sql_IN_bind_variables(statList,
    prefix=":stat")``: the statement spells the list as one placeholder,
    ``:stat__in``, whose values are ``statList`` -- so ``jobStatus IN
    ({stat_var_names_str})`` reads as a predicate on a bind like any other,
    where it was a hole.
    """
    found = dict(_own_lists(func))
    if _CALLS is not None:
        for name, held in _imported_lists(func).items():
            found.setdefault(name, held)
    return found


def _own_lists(scope: ast.AST) -> dict[str, InList]:
    """The lists *scope* builds itself; a module's, at its top level."""
    held = getattr(scope, "_in_lists", None)
    if held is not None:
        return held
    found: dict[str, InList] = {}
    nodes = scope.body if isinstance(scope, ast.Module) else ast.walk(scope)
    for node in nodes:
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Tuple)
            and len(node.targets[0].elts) == 2
            and all(isinstance(one, ast.Name) for one in node.targets[0].elts)
            and isinstance(node.value, ast.Call)
            and node.value.args
        ):
            continue
        called = node.value.func
        if (called.attr if isinstance(called, ast.Attribute) else getattr(called, "id", "")) != IN_LIST_HELPER:
            continue
        prefix = next(
            (
                keyword.value.value
                for keyword in node.value.keywords
                if keyword.arg == "prefix" and isinstance(keyword.value, ast.Constant)
            ),
            None,
        )
        if prefix is None and len(node.value.args) > 1 and isinstance(node.value.args[1], ast.Constant):
            prefix = node.value.args[1].value
        prefix = prefix if isinstance(prefix, str) and prefix else ":in"
        names, values_map = (one.id for one in node.targets[0].elts)  # type: ignore[attr-defined]
        found[names] = InList(names, values_map, node.value.args[0], f"{prefix}__in", node)
    scope._in_lists = found  # type: ignore[attr-defined]
    return found


def _imported_lists(func: ast.AST) -> dict[str, InList]:
    """The lists another module builds at its top level and *func*'s module imports.

    ``INPUT_TYPES_var_str, INPUT_TYPES_var_map = get_sql_IN_bind_variables(
    JediDatasetSpec.getInputTypes(), ...)`` is written once in
    ``JediDatasetSpec.py`` and spliced into statements across the proxy
    modules, under the names they import it as.
    """
    assert _CALLS is not None
    cached = getattr(func, "_imported_lists", None)
    if cached is not None and cached[0] == id(_CALLS):
        return cached[1]
    file = _CALLS.types.file_of(func)
    imports = _CALLS.types.imports(file) if file is not None else {}
    local = {held: name for name, held in imports.items()}
    found: dict[str, InList] = {}
    for name, (module, attribute) in imports.items():
        where = _CALLS.types.module_file(module)
        tree = _CALLS.types.tree(where) if where else None
        held = _own_lists(tree).get(attribute) if tree is not None else None
        if held is None:
            continue
        values_map = local.get((module, held.values_map))
        if values_map is None:
            continue
        found[name] = held._replace(names=name, values_map=values_map)
    func._imported_lists = (id(_CALLS), found)  # type: ignore[attr-defined]
    return found


def _with_lists(node: ast.expr, lists: dict[str, InList]) -> ast.expr:
    """*node* with each hole holding an ``IN`` list's placeholders spelled as its bind."""
    if not lists or not isinstance(node, ast.JoinedStr):
        return node
    values: list[ast.expr] = []
    changed = False
    for part in node.values:
        if (
            isinstance(part, ast.FormattedValue)
            and isinstance(part.value, ast.Name)
            and part.value.id in lists
        ):
            values.append(ast.Constant(value=lists[part.value.id].bind))
            changed = True
        else:
            values.append(part)
    return ast.JoinedStr(values=values) if changed else node


#: The corpus's calls, while a build reads statements through them.
_CALLS: Optional["Calls"] = None

#: How many calls out a statement's text or binds are followed.  ``getJobs``
#: is two: its ``WHERE`` is a closure's return, which is
#: ``construct_where_clause``'s.
RETURN_HOPS = 3


@contextlib.contextmanager
def across_calls(calls: Optional["Calls"]) -> Iterator[None]:
    """Read statements, for the length of the block, through the calls that hand their pieces in.

    ``sqlP += sql_where_clause`` appends what ``build_where_clause()``
    returned, and the text is only there.  Set for a whole build rather than
    passed to each reader, because every slice that reads a statement has to
    read the same one: a read side seeing ``getJobs``'s ``WHERE`` and a write
    side not seeing it would be the map describing one statement twice.
    """
    global _CALLS
    previous, _CALLS = _CALLS, calls
    try:
        yield
    finally:
        _CALLS = previous


def _slot(target: ast.expr, name: str) -> Optional[int]:
    """``-1`` where *target* is *name*, its index where *target* unpacks into it, else ``None``."""
    if isinstance(target, ast.Name):
        return -1 if target.id == name else None
    if isinstance(target, (ast.Tuple, ast.List)):
        for index, element in enumerate(target.elts):
            if isinstance(element, ast.Name) and element.id == name:
                return index
    return None


def returned(
    func: ast.FunctionDef | ast.AsyncFunctionDef, name: str
) -> list[tuple[ast.FunctionDef | ast.AsyncFunctionDef, ast.expr]]:
    """``(callee, expression)`` for what each call *name* is bound from hands back.

    ``a, b = f()`` binds ``b`` to the second element of each tuple ``f``
    returns; a return that is not a tuple of that length says nothing about it
    and is passed over.  Empty outside :func:`across_calls`, and for a call the
    corpus's calls do not place.
    """
    if _CALLS is None:
        return []
    found: list[tuple[ast.FunctionDef | ast.AsyncFunctionDef, ast.expr]] = []
    for node in ast.walk(func):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)) or not isinstance(node.value, ast.Call):
            continue
        if enclosing_function(node) is not func:
            continue
        for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
            slot = _slot(target, name)
            if slot is not None:
                found.extend(_called(func, node.value, slot, 0))
    unique: dict[tuple[int, int], tuple[ast.FunctionDef | ast.AsyncFunctionDef, ast.expr]] = {}
    for callee, value in found:
        unique.setdefault((id(callee), id(value)), (callee, value))
    return list(unique.values())


def _called(
    func: ast.AST, call: ast.Call, slot: int, hops: int
) -> list[tuple[ast.FunctionDef | ast.AsyncFunctionDef, ast.expr]]:
    """What each function *call* lands in returns, at *slot* (``-1`` for all of it)."""
    if _CALLS is None or hops >= RETURN_HOPS:
        return []
    file = _CALLS.types.file_of(func)
    if file is None:
        return []
    found: list[tuple[ast.FunctionDef | ast.AsyncFunctionDef, ast.expr]] = []
    for callee in _CALLS.callees(file, func, call).targets:
        for value in _returns(callee.func):
            found.extend(_at_slot(callee.func, value, slot, hops + 1))  # type: ignore[arg-type]
    return found


def _at_slot(
    func: ast.FunctionDef | ast.AsyncFunctionDef, value: ast.expr, slot: int, hops: int
) -> list[tuple[ast.FunctionDef | ast.AsyncFunctionDef, ast.expr]]:
    """*value*, returned by *func*, at *slot*: through a returned call, and a returned name bound to a tuple.

    ``return self.getCriteriaByAge(...)`` hands back what that call does, and
    ``return ret_empty`` what ``ret_empty = "", {}`` wrote.
    """
    if isinstance(value, ast.Call):
        return _called(func, value, slot, hops)
    if slot < 0:
        return [(func, value)]
    if isinstance(value, ast.Tuple):
        return [(func, value.elts[slot])] if slot < len(value.elts) else []
    if isinstance(value, ast.Name):
        return [
            (func, element)
            for node in ast.walk(func)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            and node.value is not None
            and enclosing_function(node) is func
            and any(
                isinstance(target, ast.Name) and target.id == value.id
                for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
            )
            and isinstance(node.value, ast.Tuple)
            and slot < len(node.value.elts)
            for element in [node.value.elts[slot]]
        ]
    return []


def _returns(func: ast.AST) -> list[ast.expr]:
    from bamboo.codemap.calls import returns

    return returns(func)


def placed(func: ast.AST, call: ast.Call) -> Optional["Placed"]:
    """Where *call*, made in *func*, lands; ``None`` outside :func:`across_calls`.

    For a slice that reads a callee its own way.  The build's calls are held
    here so that every slice asking where a call goes gets the same answer.
    """
    if _CALLS is None:
        return None
    file = _CALLS.types.file_of(func)
    if file is None:
        return None
    return _CALLS.callees(file, func, call)


def every_return(
    func: ast.AST, call: ast.Call, hops: int = 0
) -> Optional[list[tuple[ast.FunctionDef | ast.AsyncFunctionDef, ast.expr]]]:
    """``(callee, value)`` for every value *call* can hand back; ``None`` if one cannot be followed.

    :func:`returned` reads a statement's text, where a piece that cannot be
    followed stays a hole and the rest is still worth reading.  A value set is
    used differently -- to eliminate -- so one short of a member is a false
    one: a helper returning a literal on one branch and an unplaced call on
    another would read as returning the literal alone.  So every function the
    call can land in, every ``return`` in each, and every call such a return
    makes are followed, or nothing is claimed.

    Empty outside :func:`across_calls` is ``None`` too: no calls, no claim.
    """
    if _CALLS is None or hops >= RETURN_HOPS:
        return None
    file = _CALLS.types.file_of(func)
    if file is None:
        return None
    placed = _CALLS.callees(file, func, call)
    if not placed.targets or placed.unplaced:
        return None
    found: list[tuple[ast.FunctionDef | ast.AsyncFunctionDef, ast.expr]] = []
    for target in placed.targets:
        values = _returns(target.func)
        if not values:
            return None
        for value in values:
            if isinstance(value, ast.Call):
                deeper = every_return(target.func, value, hops + 1)
                if deeper is None:
                    return None
                found.extend(deeper)
            else:
                found.append((target.func, value))  # type: ignore[arg-type]
    return found


def _handed_texts(
    func: ast.FunctionDef | ast.AsyncFunctionDef, name: str, seen: frozenset[str]
) -> list[str]:
    """The statements local *name* holds, where a call hands it: one per variant the callee builds."""
    if sum(":" in one for one in seen) >= RETURN_HOPS:
        return []
    texts: list[str] = []
    for callee, value in returned(func, name):
        if isinstance(value, ast.Name):
            found = variants(callee, value.id, seen | {f"{id(callee)}:{value.id}"})
            found = found or _handed_texts(callee, value.id, seen | {f"{id(callee)}:{value.id}"})
        else:
            rendered = rendered_text(value)
            found = [rendered] if rendered is not None else []
        # An empty one is left out: ``ret_empty = "", {}`` is what a callee
        # hands back when there is nothing to append, and the fragment
        # appending it sits under ``if sorting_sql:`` -- a run appending the
        # empty text is a run that skipped the fragment.
        texts.extend(text for text in found if text and text not in texts)
    return texts[:_MAX_BRANCH_VARIANTS]


#: The function being read as one call site runs it, and the constants that
#: call hands its parameters -- see :func:`at_call`.
_AT_CALL: Optional[tuple[ast.AST, dict[str, object]]] = None


@contextlib.contextmanager
def at_call(func: ast.AST, values: dict[str, object]) -> Iterator[None]:
    """Read *func*, for the length of the block, as one call site runs it.

    *values* are constants the call hands parameters *func* never rebinds.
    A fragment, a table list's member or a bind under an ``if`` those
    constants decide against is not part of what that call runs:
    ``TaskCommando.py:143`` calls ``getPandaIDsWithTask_JEDI(jediTaskID,
    True)``, and the two archive tables are appended under ``if not
    onlyActive``.  Only *func*'s own nodes are ruled on: a callee the reader
    follows has parameters of its own, whatever they are called.
    """
    global _AT_CALL
    previous, _AT_CALL = _AT_CALL, (func, values)
    try:
        yield
    finally:
        _AT_CALL = previous


def ruled_out(node: ast.AST) -> bool:
    """Whether the call site being read (:func:`at_call`) never runs *node*."""
    if not _AT_CALL:
        return False
    func, values = _AT_CALL
    return ruled_out_by(node, func, values)


def decided_here(test: ast.expr) -> object:
    """What the call site being read (:func:`at_call`) makes of *test*: ``True``, ``False`` or ``UNDECIDED``."""
    if not _AT_CALL:
        return UNDECIDED
    func, values = _AT_CALL
    if enclosing_function(test) is not func:
        return UNDECIDED
    return decided(test, values)


def _parts(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    name: str,
    seen: frozenset[str] = frozenset(),
    stops: Optional[list[Stop]] = None,
) -> list[tuple[int, str, str, ast.stmt]]:
    """Return ``(line, operator, text, node)`` for each statement building *name*.

    *stops* is an out-parameter, filled with what this walk could not render.
    Out rather than returned because every caller but one wants the fragments
    alone, and because the account has to come from the walk that failed: a
    second pass asking "why is this statement short" would be a different
    reading of the same nodes, free to disagree with this one.  The same
    reason :class:`BindScan` travels with its binds.
    """
    if name in seen:
        return []
    seen = seen | {name}
    lists = in_lists(func)
    parts: list[tuple[int, str, str, ast.stmt]] = []
    for node in ast.walk(func):
        if _AT_CALL and ruled_out(node):
            continue
        if targets := targets_of(node):
            for target in targets:
                if not (isinstance(target, ast.Name) and target.id == name):
                    continue
                trimmed = _trimmed(_with_lists(node.value, lists), name)
                if trimmed is not None:
                    # ``sql = sql[:-4]`` cuts the ``AND `` a loop left after
                    # its last term.  Left unread, the statement ends in
                    # ``AND`` -- a text PanDA never runs.
                    _count, rest = trimmed
                    parts.append((node.lineno, TRIM, "", node))
                    if rest:
                        parts.append((node.lineno, "+=", _joined(rest), node))
                    continue
                wrapped = _wrapping(_with_lists(node.value, lists), name)
                if wrapped is not None:
                    # ``sqlP = "SELECT * FROM (" + sqlP`` puts text around
                    # what the name holds; read as a new statement, it is
                    # one that starts with ``SELECT * FROM (`` and has no
                    # table.
                    before, after = wrapped
                    if before:
                        parts.append((node.lineno, PREFIX, before, node))
                    if after:
                        parts.append((node.lineno, "+=", after, node))
                    continue
                text = rendered_text(_with_lists(node.value, lists))
                if text is None and not isinstance(node.value, ast.Name):
                    composed = _composed_texts(func, node.value, seen)
                    if composed:
                        # ``sql = sqlH + sqlL`` and ``re.sub(p, r, sql_y)``:
                        # each statement the names hold, put together or
                        # rewritten, as alternatives in this one place.
                        parts.extend((node.lineno, "=", one, node) for one in composed)
                        continue
                if text is None and isinstance(node.value, ast.Name):
                    # ``sql = sqlTU`` -- the statement was built under another
                    # name and *chosen* here.  The choosing assignment is the
                    # better anchor of the two: in ``reactivatePendingTasks_JEDI``
                    # the release statement is appended unconditionally and
                    # picked in the ``else`` of the timeout test, so the reason
                    # is on the alias and nowhere else.
                    text = _fold(_parts(func, node.value.id, seen))
                    if not text:
                        # ``sql = where_clause`` where a call handed the
                        # name its text: one fragment per statement the
                        # callee builds, as alternatives in one place.
                        parts.extend(
                            (node.lineno, "=", one, node)
                            for one in _handed_texts(func, node.value.id, seen)
                        )
                        continue
                if text:
                    parts.append((node.lineno, "=", text, node))
        elif (
            isinstance(node, ast.AugAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
            and isinstance(node.op, ast.Add)
        ):
            text = rendered_text(_with_lists(node.value, lists))
            if text is None and isinstance(node.value, ast.Name):
                # ``sql += scout_filter`` appends text the function wrote
                # down under another name; ``sqlP += sql_where_clause`` text a
                # call handed it.  Either way each statement the name can
                # hold is one alternative, appended in this one place.
                held = variants(func, node.value.id, seen) or _handed_texts(
                    func, node.value.id, seen
                )
                if held:
                    parts.extend((node.lineno, "+=", one, node) for one in held)
                    continue
            if text is not None:
                parts.append((node.lineno, "+=", text, node))
            elif stops is not None:
                # Only ``+=``.  A ``=`` that renders nothing starts no
                # statement, so nothing was dropped from one; and
                # ``sql = sql[:-1]``, once most of them, is read above as a
                # cut.
                stops.append(
                    Stop(node.lineno, ast.unparse(node.value), FRAGMENT_NOT_RENDERED)
                )
    parts.sort(key=lambda part: part[0])
    return parts


#: The operator of a fragment written in front of what the name holds.
PREFIX = "=+"

#: The operator of a fragment that cuts the end off what the name holds.  Its
#: text is empty and the count is read off the node -- see :func:`_cut_of`.
TRIM = "[:-]"


def _trimmed(value: ast.expr, name: str) -> Optional[tuple[int, list[ast.expr]]]:
    """``(count, rest)`` where *value* is *name* with its last *count* characters cut, then *rest*.

    The two spellings the corpus runs a statement through, 15 times over 14
    functions, each cutting what a loop left after its last item -- a comma,
    or ``AND ``::

        sql = sql[:-4]
        sql_to_get_ids = sql_to_get_ids[:-1] + ") "
    """
    operands = _concatenated(value)
    head = operands[0]
    if not (
        isinstance(head, ast.Subscript)
        and isinstance(head.value, ast.Name)
        and head.value.id == name
        and isinstance(head.slice, ast.Slice)
        and head.slice.lower is None
        and head.slice.step is None
    ):
        return None
    upper = head.slice.upper
    if (
        isinstance(upper, ast.UnaryOp)
        and isinstance(upper.op, ast.USub)
        and isinstance(upper.operand, ast.Constant)
        and type(upper.operand.value) is int
        and upper.operand.value > 0
    ):
        return upper.operand.value, operands[1:]
    return None


def _cut_of(node: ast.stmt) -> int:
    """How many characters a :data:`TRIM` fragment cuts."""
    value = written_value(node)
    for target in targets_of(node):
        if isinstance(target, ast.Name) and value is not None:
            trimmed = _trimmed(value, target.id)
            if trimmed is not None:
                return trimmed[0]
    return 0


def _joined(pieces: list[ast.expr]) -> str:
    """The text of a run of ``+`` operands, a piece that renders nothing being a hole."""
    rendered = [rendered_text(piece) for piece in pieces]
    return "".join(one if one is not None else _BARE_FIELD for one in rendered)


def _wrapping(value: ast.expr, name: str) -> Optional[tuple[str, str]]:
    """``(before, after)`` where *value* concatenates text around *name* itself, once.

    An operand that renders nothing is a hole, as it is at the call.
    """
    operands = _concatenated(value)
    at = [
        index
        for index, operand in enumerate(operands)
        if isinstance(operand, ast.Name) and operand.id == name
    ]
    if len(at) != 1 or len(operands) < 2:
        return None
    return _joined(operands[: at[0]]), _joined(operands[at[0] + 1 :])


def _composed_texts(
    func: ast.FunctionDef | ast.AsyncFunctionDef, value: ast.expr, seen: frozenset[str]
) -> list[str]:
    """The statements *value* builds out of names this function holds statements under.

    Two spellings: a ``+`` of names and literals, each name standing for the
    one statement it holds (a piece that renders nothing is a hole); and
    ``re.sub(<literal>, <literal>, <name>)``, run on each statement the name
    holds -- the pattern and the replacement are written down, so the result is
    computed, not guessed (see :func:`_rewritten_values`).
    """
    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr == "sub"
        and len(value.args) >= 3
        and _is_literal(value.args[0])
        and _is_literal(value.args[1])
        and isinstance(value.args[2], ast.Name)
    ):
        try:
            return [
                re.sub(value.args[0].value, value.args[1].value, text)
                for text in variants(func, value.args[2].id, seen)
            ][:_MAX_BRANCH_VARIANTS]
        except re.error:
            return []
    operands = _concatenated(value)
    if len(operands) < 2 or not any(isinstance(operand, ast.Name) for operand in operands):
        return []
    pieces: list[list[str]] = []
    for operand in operands:
        if isinstance(operand, ast.Name):
            held = variants(func, operand.id, seen)
            if len(held) > 1:
                # Two names each built several ways are not crossed: which
                # header goes with which tail is decided by the branches that
                # built them, and pairing every one with every other makes
                # statements no run sends -- a header with no VALUES.
                return []
            pieces.append(held or [_BARE_FIELD])
        elif isinstance(operand, ast.BinOp) and isinstance(operand.op, ast.Mod):
            pieces.append(_percent_texts(func, operand, seen) or [_BARE_FIELD])
        else:
            rendered = rendered_text(operand)
            pieces.append([rendered if rendered is not None else _BARE_FIELD])
    if all(piece == [_BARE_FIELD] for piece in pieces):
        return []
    if math.prod(len(piece) for piece in pieces) > _MAX_BRANCH_VARIANTS:
        return []
    return ["".join(combination) for combination in itertools.product(*pieces)]


def _percent_texts(
    func: ast.FunctionDef | ast.AsyncFunctionDef, value: ast.BinOp, seen: frozenset[str]
) -> list[str]:
    """``template % (a, b)``: the template's statements, each ``%s`` filled from what its argument holds.

    An argument the source does not write out -- ``JobSpec.columnNames()`` --
    stays a hole; a table name a loop or a list supplies fills it once per
    name, as :func:`_table_choices` does for an f-string.
    """
    head = _concatenated(value.left)
    templates: list[str] = [""]
    for operand in head:
        if isinstance(operand, ast.Name):
            held = variants(func, operand.id, seen)
        else:
            rendered = rendered_text(operand)
            held = [rendered] if rendered is not None else []
        if not held:
            return []
        templates = [one + more for one in templates for more in held]
    arguments = value.right.elts if isinstance(value.right, ast.Tuple) else [value.right]
    fills = [_literal_values(func, argument) or [_BARE_FIELD] for argument in arguments]
    found: list[str] = []
    for template in templates:
        if template.count(_PERCENT_S) != len(fills):
            found.append(template.replace(_PERCENT_S, _BARE_FIELD))
            continue
        for combination in itertools.product(*fills):
            text = template
            for fill in combination:
                text = text.replace(_PERCENT_S, fill, 1)
            found.append(text)
    return found[:_MAX_BRANCH_VARIANTS]


def _through_name(func: ast.FunctionDef | ast.AsyncFunctionDef, expression: ast.expr) -> ast.expr:
    """*expression*, or the ``%``/``.format`` expression the name at its head was bound to once.

    ``sql_exe = (sql_jt + comment) % table`` and then ``execute(sql_exe,
    ...)`` is ``execute((sql_jt + comment) % table, ...)`` read one line
    apart: the table is filled in by the ``%``, which :func:`_call_site_fill`
    reads where it is written into the call.
    """
    head = _concatenated(expression)[0]
    if not isinstance(head, ast.Name) or variants(func, head.id):
        return expression
    bound = [
        node.value
        for node in ast.walk(func)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and node.value is not None
        and enclosing_function(node) is func
        and any(isinstance(target, ast.Name) and target.id == head.id for target in targets_of(node))
    ]
    if len(bound) != 1:
        return expression
    value = bound[0]
    percent = isinstance(value, ast.BinOp) and isinstance(value.op, ast.Mod)
    formatted = (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr == "format"
    )
    return value if percent or formatted else expression


def _fold(parts: list[tuple[int, str, str, ast.stmt]]) -> str:
    """Concatenate one ``=``-started run of fragments."""
    assembled = ""
    for _line, operator, text, node in parts:
        if operator == "=":
            assembled = text
        elif operator == PREFIX:
            assembled = text + assembled
        elif operator == TRIM:
            assembled = assembled[: max(len(assembled) - _cut_of(node), 0)]
        else:
            assembled += text
    return assembled


def _compatible_sets(
    parts: list[tuple[int, str, str, ast.stmt]], start: int
) -> list[frozenset[int]]:
    """Return the fragment sets a single run of the code can append, largest first.

    One set per statement a reader can meet.  The test between two fragments is
    :func:`pathcond.parted`, reading the negations the path condition already
    carries as the function runs; a statement is then a set no two of whose
    members contradict, and
    which nothing else can be added to -- a maximal clique of the compatibility
    graph, enumerated with Bron-Kerbosch.

    **Connected components were the wrong shape, and the corpus says where.**
    Grouping the exclusive pairs and taking one member per group is right for
    an ``if``/``elif``/``else``, whose arms are pairwise exclusive, and wrong
    the moment two unrelated ``if``s touch.  ``insertTaskParams_JEDI`` builds
    one INSERT across a backend test and a ``parent_tid`` test, and the inner
    ``if`` repeats the outer one -- so one arm of each is exclusive with one arm
    of the other and all five fragments collapse into a single alternative.
    Exactly one of the five then survives per statement, and every statement
    comes out with an opening parenthesis and no closing one, or the reverse.
    ``writes`` reads none of them: the map reported six statements it could name
    no column of, and ``ATLAS_DEFT.T_TASK`` lost ``priority``.

    A fragment under no test is compatible with everything, so it lands in every
    set without being special-cased.
    """
    indices = list(range(start, len(parts)))
    # Two fragments one statement appends in one place are alternatives: the
    # statements a name handed in can hold, of which a run appends one.
    agrees = {
        index: {
            other
            for other in indices
            if other != index
            and not (parts[other][3] is parts[index][3] and parts[other][1] == parts[index][1])
            and not parted(parts[index][3], parts[other][3], exits=False)
        }
        for index in indices
    }
    if all(len(agrees[index]) == len(indices) - 1 for index in indices):
        # Nothing contradicts anything: one statement, which is most runs.
        return [frozenset(indices)]

    found: list[frozenset[int]] = []

    def expand(chosen: set[int], candidates: set[int], refused: set[int]) -> None:
        if len(found) > _MAX_BRANCH_VARIANTS:
            # The caller folds past the cap; stopping here keeps a pathological
            # run from costing more than the answer it would be given.
            return
        if not candidates and not refused:
            found.append(frozenset(chosen))
            return
        pivot = max(candidates | refused, key=lambda one: len(agrees[one] & candidates))
        for one in sorted(candidates - agrees[pivot]):
            expand(chosen | {one}, candidates & agrees[one], refused & agrees[one])
            candidates = candidates - {one}
            refused = refused | {one}

    expand(set(), set(indices), set())
    return sorted(found, key=lambda members: sorted(members))


#: A hole standing where a whole table reference goes.  The keyword settles it
#: from the left and the character after the hole settles it from the right:
#: ``FROM {schemaJEDI}.JEDI_Tasks`` interpolates the *schema* and is followed by
#: a dot, which is why reassembly blanks interpolations in the first place.
#:
#: The schema may itself be a hole, so the keyword need not be the hole's
#: immediate neighbour: ``INSERT INTO {schema}.{table}`` puts a filled-in
#: ``{}.`` between them.  Requiring adjacency lost the two statements that
#: create a job row -- ``INSERT INTO {}.{} (...) RETURNING PandaID`` -- and so
#: the map could not say where the system's most-read row comes into existence.
_BEFORE_TABLE = re.compile(
    r"\b(?:FROM|UPDATE|INTO|JOIN)\s+(?:(?:\{\}|[A-Za-z_][\w$]*)\.)?$", re.IGNORECASE
)


def _table_holes(text: str) -> set[int]:
    """Which of *text*'s ``{}`` holes stand for a whole table name.

    Position, not resolvability, is what this decides, and it decides it
    narrowly on purpose.  :func:`rendered_text` serves two readers that want
    opposite things from the same ``{}``: this one wants the hole *filled*,
    because a statement needs its table named, while ``DiagnosticTemplate``
    wants it *kept*, because the unfilled frame is the part a line observed in
    production can be matched against.  Filling every hole a value could be
    recovered for fills ninety-nine more across the corpus, nearly all of them
    in log messages and dataset names -- and a template expanded over its own
    values no longer matches the line that carried a different one.
    """
    found: set[int] = set()
    position = index = 0
    while (hole := text.find(_BARE_FIELD, position)) >= 0:
        if text[hole + 2 : hole + 3] != "." and _BEFORE_TABLE.search(text[:hole]):
            found.add(index)
        position, index = hole + 2, index + 1
    return found


# A field marker whose value is derived from an argument rather than being one
# -- ``{0.attr}``, ``{0[1]}``, ``{p:.3%}``, or an index the call passes nothing
# for.  It holds a position so the markers and the ``{}``s stay in step, and
# resolves to nothing, which is what a hole nobody can read is worth.
_DERIVED_FIELD = ast.Constant(value=None)


def _format_arguments(node: ast.Call) -> Optional[list[ast.expr]]:
    """The expression behind each field marker of ``<text>.format(...)``.

    One entry per marker in *source order, with repeats*, not one per distinct
    argument: ``"FROM {0}.{1} j, {0}.Datasets d"`` has three markers and two
    arguments, and :func:`_table_choices` measures this list against the number
    of ``{}`` the rendered text carries.  A list of distinct arguments is one
    short, and a fragment whose count does not match is dropped rather than
    truncated -- silently, which is how a table name goes missing.

    The indices are read back from the receiver, *before* :func:`rendered_text`
    flattens every marker to ``{}``.  Leaving them in the rendered text instead
    was measured and rejected: a ``{0}`` that survives is a ``{}`` that stopped
    working, because every reader downstream keys on the two literal
    characters, and the one that does not invented a boundary named ``{1}``.
    """
    text = rendered_text(node.func.value)
    if text is None or "{{" in text or "}}" in text:
        # An escaped brace is not a field, but ``_FIELD`` matches the ``{}``
        # inside ``{{}}`` -- so the markers and the holes stop counting the
        # same things, and nothing downstream would say which is which.
        return None
    if _hole_expressions(node.func.value) != []:
        # The receiver has to be text the source wrote, whole.  An f-string, or
        # a concatenation holding one, renders ``{}`` as well, and afterwards
        # there is no telling which hole came from where.
        return None
    if any(isinstance(argument, ast.Starred) for argument in node.args) or any(
        keyword.arg is None for keyword in node.keywords
    ):
        # ``*names`` / ``**names``: the call does not say how many, or which.
        return None
    named = {keyword.arg: keyword.value for keyword in node.keywords}
    found: list[ast.expr] = []
    automatic = 0
    for _literal, field, spec, conversion in string.Formatter().parse(text):
        if field is None:
            continue
        if spec or conversion or any(access in field for access in ".["):
            found.append(_DERIVED_FIELD)
        elif field == "":
            found.append(node.args[automatic] if automatic < len(node.args) else _DERIVED_FIELD)
            automatic += 1
        elif field.isdigit():
            index = int(field)
            found.append(node.args[index] if index < len(node.args) else _DERIVED_FIELD)
        else:
            found.append(named.get(field, _DERIVED_FIELD))
    # The same measurement :func:`_table_choices` will make, made here as well:
    # composed under a ``+``, a list of the wrong length can be summed with a
    # correct one into a total that matches and is aligned with nothing.
    rendered = rendered_text(node)
    if rendered is None or len(found) != rendered.count(_BARE_FIELD):
        return None
    return found


def _hole_expressions(node: ast.expr) -> Optional[list[ast.expr]]:
    """The expressions behind each ``{}`` :func:`rendered_text` left, in order.

    ``None`` when the alignment is not assured -- a hole filled with the wrong
    expression's value would put an invented table name in the statement with
    the same confidence as one the code wrote.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return []
    if isinstance(node, ast.JoinedStr):
        found: list[ast.expr] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                continue
            if not isinstance(value, ast.FormattedValue):
                return None
            found.append(value.value)
        return found
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _hole_expressions(node.left), _hole_expressions(node.right)
        return None if left is None or right is None else left + right
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "format"
    ):
        return _format_arguments(node)
    return None


def _fill(text: str, values: dict[int, str]) -> str:
    """Replace the numbered holes of *text*, leaving the rest as they are."""
    out: list[str] = []
    position = index = 0
    while (hole := text.find(_BARE_FIELD, position)) >= 0:
        out.append(text[position:hole])
        # A filled name carries its own schema hole -- ``{}.jobsActive4`` -- so
        # the scan continues past the original hole rather than over what
        # replaced it.
        out.append(values.get(index, _BARE_FIELD))
        position, index = hole + 2, index + 1
    out.append(text[position:])
    return "".join(out)


def _table_choices(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    parts: list[tuple[int, str, str, ast.stmt]],
) -> tuple[list[tuple[list[ast.expr], set[int]]], dict[str, list[str]]]:
    """Return each fragment's table holes, and the names each one can hold.

    Keyed by the expression rather than by the hole, because one loop variable
    interpolated into two fragments of a statement takes *one* value per pass;
    treating the holes independently would cross them and read a union of two
    tables as a single statement over both.
    """
    per_part: list[tuple[list[ast.expr], set[int]]] = []
    choices: dict[str, list[str]] = {}
    for _line, _operator, text, node in parts:
        value = written_value(node)
        expressions = _hole_expressions(value) if value is not None else None
        if expressions is None or len(expressions) != text.count(_BARE_FIELD):
            per_part.append(([], set()))
            continue
        holes = {
            hole for hole in _table_holes(text) if _literal_values(func, expressions[hole])
        }
        per_part.append((expressions, holes))
        for hole in holes:
            choices.setdefault(
                ast.unparse(expressions[hole]), _literal_values(func, expressions[hole])
            )
    return per_part, choices


def _assignments(choices: dict[str, list[str]]) -> list[dict[str, str]]:
    """One mapping of expression to table name per combination the code allows."""
    if not choices:
        return [{}]
    names = sorted(choices)
    return [
        dict(zip(names, combination, strict=True))
        for combination in itertools.product(*(choices[name] for name in names))
    ]


def _fold_filled(
    parts: list[tuple[int, str, str, ast.stmt]],
    per_part: list[tuple[list[ast.expr], set[int]]],
    assignment: dict[str, str],
    dropped: set[int],
    key_fills: Optional[list[Optional[str]]] = None,
) -> tuple[str, list[Span]]:
    """:func:`_fold`, with each fragment's table holes filled from *assignment*.

    Returns the statement and where in it each fragment landed.  The spans are
    taken here rather than recomputed because they have to describe the text
    *after* filling: ``_fill`` changes the length of any fragment carrying a
    hole, so offsets measured on the unfilled text point at the wrong column as
    soon as a table name is interpolated.

    *key_fills* is the other kind of filling -- see :func:`_key_fills`.  A
    fragment appended once per key of a mapping stands for all of them, so its
    replacement is already the whole run and the table holes have nothing left
    to say about it.
    """
    assembled = ""
    spans: list[Span] = []
    for index, (_line, operator, text, node) in enumerate(parts):
        if index in dropped:
            continue
        if operator == TRIM:
            # Cut from every fragment it reaches into, so no value is placed
            # in a stretch the statement no longer has.
            kept = max(len(assembled) - _cut_of(node), 0)
            assembled = assembled[:kept]
            spans = [Span(span.node, min(span.start, kept), min(span.end, kept)) for span in spans]
            continue
        if key_fills is not None and key_fills[index] is not None:
            text = key_fills[index]
            spans.append(Span(node, len(assembled), len(assembled) + len(text)))
            assembled += text
            continue
        expressions, holes = per_part[index]
        if holes and assignment:
            text = _fill(
                text,
                {
                    hole: assignment[ast.unparse(expressions[hole])]
                    for hole in holes
                    if ast.unparse(expressions[hole]) in assignment
                },
            )
        if operator == "=":
            assembled = ""
            spans = []
        if operator == PREFIX:
            spans = [Span(node, 0, len(text))] + [
                Span(span.node, span.start + len(text), span.end + len(text)) for span in spans
            ]
            assembled = text + assembled
            continue
        spans.append(Span(node, len(assembled), len(assembled) + len(text)))
        assembled += text
    return assembled, spans


def _accumulated(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    parts: list[tuple[int, str, str, ast.stmt]],
    name: str,
) -> list[tuple[int, str, str, ast.stmt]]:
    """*parts*, with a loop that builds one statement over a list of tables written out.

    A loop over table names reads as one statement per table when the
    statement runs inside it.  When nothing in the loop runs a statement, the
    loop is building one::

        sqlP = ""
        for tableName in tables:
            if sqlP != "":
                sqlP += "UNION ALL "
            sqlP += f"SELECT PandaID FROM {tableName} WHERE jediTaskID=:jediTaskID "

    and split per table, every pass kept the ``UNION ALL``: the test reads
    what earlier passes appended, which no single pass can say, so each one
    read as a statement beginning ``UNION ALL`` that PanDA never sends.
    Written out pass by pass, that test is answered by what is assembled so
    far, and a test on the table by :func:`_test_value`.  One neither can
    decide keeps its fragment, as an optional fragment is kept everywhere
    else.
    """
    per_part, choices = _table_choices(func, parts)
    for (_line, _operator, _text, node), (expressions, holes) in zip(parts, per_part, strict=True):
        for hole in holes:
            variable = ast.unparse(expressions[hole])
            loop = next(
                (
                    one
                    for one in ancestors(node)
                    if isinstance(one, ast.For)
                    and isinstance(one.target, ast.Name)
                    and one.target.id == variable
                ),
                None,
            )
            if loop is None or any(
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr in RUNNERS
                for call in ast.walk(loop)
            ):
                continue
            body = [
                index
                for index, part in enumerate(parts)
                if any(one is loop for one in ancestors(part[3]))
            ]
            # Only a run of ``+=`` the loop owns outright: a ``=`` inside it
            # starts a statement per pass, and that is the split's case.
            if body != list(range(body[0], body[-1] + 1)) or any(parts[i][1] != "+=" for i in body):
                return parts
            assembled = _fold(parts[: body[0]])
            text = ""
            for value in choices[variable]:
                for index in body:
                    _at, _op, fragment, held = parts[index]
                    if _guarded(held, loop, {variable: value, name: assembled + text}) is False:
                        continue
                    fragment_expressions, fragment_holes = per_part[index]
                    text += _fill(
                        fragment,
                        {
                            hole: value
                            for hole in fragment_holes
                            if ast.unparse(fragment_expressions[hole]) == variable
                        },
                    )
            return _accumulated(
                func, parts[: body[0]] + [(loop.lineno, "+=", text, loop)] + parts[body[-1] + 1 :], name
            )
    return parts


def _guarded(node: ast.AST, loop: ast.For, bound: dict[str, str]) -> Optional[bool]:
    """Whether the ``if``\\ s between *node* and *loop* let it through for *bound*; ``None`` if undecided."""
    decided: Optional[bool] = True
    child: ast.AST = node
    for one in ancestors(node):
        if one is loop:
            break
        if isinstance(one, ast.If):
            value = _test_value(one.test, bound, loop)
            taken = any(statement is child for statement in one.body)
            if value is None:
                decided = None
            elif value is not taken:
                return False
        child = one
    return decided


def _run_variants(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    parts: list[tuple[int, str, str, ast.stmt]],
    name: str,
) -> list[tuple[str, list[Span]]]:
    """Return the statements one ``=``-started run of fragments can produce.

    Each comes with the spans saying which fragment contributed which stretch
    of it -- see :func:`variant_spans` for why that is the interesting part.

    Fragments appended under mutually exclusive branches are *alternatives*, not
    parts of one statement, and folding them together builds a statement that
    cannot exist::

        UPDATE {}.JEDI_Tasks SET status=:status,SET status=oldStatus,...

    which is not merely unreadable -- the second ``SET`` overwrote the first in
    the column map, so the map lost the bind arm of the write entirely and, in
    another method, gained a column no statement ever assigns.

    Only mutual exclusion splits.  A fragment appended under an ``if`` with no
    ``else`` is left in every variant: splitting on optional fragments too would
    double the count for each one, and the resulting over-read is the safe
    direction -- an extra ``AND`` clause is read, not a value invented.

    The head of the run is never dropped.  It is what the statement *is*; the
    alternatives are among the fragments appended to it.

    **A table name the loop supplies splits the run the same way.**  A hole
    standing for a whole table reference is not the schema -- ``FROM
    {tableName}`` where ``tableName`` comes from a local list of names the
    source writes out is four statements, one per table, and read as one it is
    a statement about a table called ``{}``.  That is what hid the helper
    fetching a task's jobs: the rows it selects belong to a table the map had
    no name for.  That holds where the statement runs inside the loop; where
    nothing in the loop runs one, the loop is building a single statement,
    and :func:`_accumulated` writes it out pass by pass first.
    """
    parts = _accumulated(func, parts, name)
    heads = [index for index, part in enumerate(parts) if part[1] == "="]
    # One head is the statement and is never dropped.  Several are the arms of
    # one ``if``/``else`` (see :func:`_runs`), and each set takes one of them.
    start = 1 if heads == [0] else 0
    statements = _compatible_sets(parts, start)
    if len(heads) > 1:
        statements = [members for members in statements if members & set(heads)]
    per_part, choices = _table_choices(func, parts)
    key_fills = _key_fills(func, parts)
    assignments = _assignments(choices)
    total = len(statements) * len(assignments)
    if total > _MAX_BRANCH_VARIANTS:
        # Folded as before rather than split into an arbitrary subset: a
        # truncated list of statements reads as the complete one.  Said out
        # loud, because a silent cap is indistinguishable from full coverage.
        logger.warning(
            "%s at line %d is built across %d combinations of branch and table, over "
            "the cap of %d: read as one folded statement, so a conditionally assembled "
            "column or an interpolated table name may be misread here",
            name,
            parts[0][0],
            total,
            _MAX_BRANCH_VARIANTS,
        )
        return [_fold_filled(parts, per_part, {}, set(), key_fills)]

    every = set(range(start, len(parts)))
    found: list[tuple[str, list[Span]]] = []
    # Per arm, as when each arm was a run of its own: two arms writing the
    # same text are two places the statement is built, and a call in the
    # second arm is reached only by the second's fragments.
    seen: set[tuple[Optional[int], str]] = set()
    for members in statements:
        dropped = every - members
        arm = min(members & set(heads)) if len(heads) > 1 else None
        for assignment in assignments:
            text, spans = _fold_filled(parts, per_part, assignment, dropped, key_fills)
            if text and (arm, text) not in seen:
                seen.add((arm, text))
                found.append((text, spans))
    return found


def variants(
    func: ast.FunctionDef | ast.AsyncFunctionDef, name: str, seen: frozenset[str] = frozenset()
) -> list[str]:
    """Return every statement local *name* can hold.

    Two things make one name hold several statements, and both were learned by
    getting them wrong.

    **Reassignment.**  A name given a new ``=`` mid-function holds a different
    statement from then on, and reading only the last quietly drops the others:
    ``reactivatePendingTasks_JEDI`` picks between a timeout statement and a
    release statement through one variable, so keeping one of the two loses half
    of what the method does.

    **Mutually exclusive fragments.**  A statement assembled across an
    ``if``/``else`` is as much two statements as a reassigned one is -- see
    :func:`_run_variants`.

    Either way they are returned separately rather than concatenated, because
    concatenating runs one statement's ``SET`` clause into the next one's, which
    the column regexes cannot tell from a wider ``SET``.
    """
    return [text for run in _runs(_parts(func, name, seen)) for text, _spans in _run_variants(func, run, name) if text]


def _runs(parts: list[tuple[int, str, str, ast.stmt]]) -> list[list[tuple[int, str, str, ast.stmt]]]:
    """*parts* cut into the runs one statement each is built by, at each ``=``.

    **Except a ``=`` that is another arm of the same ``if``.**  ``getNumberJobsUser``
    writes::

        if workingGroup is not None:
            sql = "SELECT ... AND workingGroup=:workingGroup "
        else:
            sql = "SELECT ... AND workingGroup IS NULL "
        sql += "AND NOT jobStatus IN (:failed,:merging) "

    and cut at each ``=``, the shared ``+=`` went to the ``else`` arm alone:
    the ``if`` arm's statement never refused ``failed``.  A ``=`` exclusive
    with everything the current run holds opens an alternative of it rather
    than a new run, and :func:`_run_variants` gives each set one arm.
    """
    runs: list[list[tuple[int, str, str, ast.stmt]]] = []
    for part in parts:
        if part[1] == "=" and runs and any(one[1] == "=" for one in runs[-1]):
            if all(parted(one[3], part[3], exits=False) for one in runs[-1]):
                runs[-1].append(part)
                continue
        if part[1] == "=" or not runs:
            runs.append([])
        runs[-1].append(part)
    return runs


def variant_spans(
    func: ast.FunctionDef | ast.AsyncFunctionDef, name: str
) -> list[tuple[str, list[Span]]]:
    """:func:`variants`, with each statement's fragments and where they landed.

    **The variant is the unit, not the fragment and not the whole function.**
    A fragment is where a path condition lives, so it is tempting to read one
    at a time; that is what ``assigns_in`` was doing, and it cannot see an
    ``INSERT`` whose columns and values were appended on different lines.  It
    is equally tempting to group fragments that share a guard, but::

        sql  = "SELECT ... "          # no test above it
        sql += "WHERE a=:a "          # no test above it
        if wanted:
            sql += "AND b=:b "        # under ``wanted``

    has two statements a reader can meet -- with and without the third
    fragment -- and the group ``["AND b=:b "]`` is neither of them.  It is not
    a statement at all; nothing ever runs it alone.

    :func:`_run_variants` already enumerates the statements properly, splitting
    on mutual exclusion and keeping an ``if`` with no ``else`` in every variant.
    All that was missing is which fragment contributed which stretch, so this
    returns that rather than inventing a second decomposition.
    """
    return [
        (text, spans)
        for run in _runs(_parts(func, name))
        for text, spans in _run_variants(func, run, name)
        if text
    ]


def reconstruct(func: ast.FunctionDef | ast.AsyncFunctionDef, name: str) -> str:
    """Reassemble the SQL string held by local *name*, as last assigned.

    Fragments are ordered by line, with ``=`` restarting the string and ``+=``
    extending it.  This is not flow-sensitive: a statement built differently in
    two branches comes back as one concatenation.  That over-reads rather than
    under-reads, and the table and column names -- the parts used here -- are
    the same in both branches when it happens.

    Where the branches disagree about the *value*, ``fragments`` recovers what
    this cannot: ``getTasksToExecCommand_JEDI`` appends ``SET status=:status``
    or ``SET status=oldStatus`` depending on the command, and both appear here.
    """
    return _fold(_parts(func, name))


def fragments(
    func: ast.FunctionDef | ast.AsyncFunctionDef, name: str
) -> list[tuple[ast.stmt, str]]:
    """Return the statements building *name*, each with the text it contributes.

    Reassembly deliberately flattens a conditionally built statement, which is
    right for reading the table and the columns and wrong for reading *why*.
    A fragment appended under an ``else`` carries that ``else`` in its own path
    condition, so the fragment is the anchor for any value written inline --
    a literal or a copied column, neither of which has a bind assignment to
    point at instead.
    """
    return [(node, text) for _line, _operator, text, node in _parts(func, name)]


class Execution(NamedTuple):
    """One statement run against the database, in either of the two forms.

    ``cursor.execute(<statement>, <binds>)`` is the proxy's spelling.  The
    daemon layer speaks through the task buffer instead --
    ``taskBuffer.querySQLS(sql, var_map)`` -- and ``window`` is what the
    second form needs that the first does not.
    """

    sql: str
    varmap: Optional[str]
    variable: str
    call: ast.Call
    window: Optional[tuple[int, int]] = None
    """Lines within which this run's binds were filled, for the forwarded form.

    ``None`` for ``execute``.  The window is not guessed -- ``copyArchive.main``
    writes ``var_map = {}`` 31 times, so the code declares its own block
    boundaries and this only reads them -- but it is no longer what pairs a
    bind with the run: every reader passes the call to :func:`bound_values`,
    which keeps the binds the run's map can still hold, in both forms.  What
    is left for the window is where a walk should start.
    """

    unresolved: tuple[Stop, ...] = ()
    """Where reading *this* run stopped, if it stopped anywhere.

    Empty for most of them, and empty is an answer: it says the text in
    ``sql`` is the whole statement.  A consumer that reports a query without
    reading this field reports a truncated ``WHERE`` as a complete one, which
    is the failure the field exists for.
    """

    calls: tuple[ast.Call, ...] = ()
    """Every call of the function that runs this statement with this map.

    The readers keep one run per statement and map -- one read site, one set
    of columns counted -- and read the binds for all of these at once: a
    statement run from two places binds what either place fills.  Before the
    binds were paired with the run this was free, because every run read every
    bind of the key; ``resetStatus`` running one ``UPDATE`` inside an ``if``
    and again below it would otherwise lose the first run's value to the
    second's.  See :attr:`run_at`.
    """

    @property
    def run_at(self) -> tuple[ast.Call, ...]:
        """The calls to read this run's binds at -- :attr:`calls`, or its own."""
        return self.calls or (self.call,)


def interpolations(func: ast.FunctionDef | ast.AsyncFunctionDef, name: str) -> list[str]:
    """Return the expressions interpolated into the fragments building *name*.

    Reassembly blanks them, which is right for reading columns and wrong for
    reading *whose* table this is: the schema is interpolated into the head of
    the statement (``f"UPDATE {panda_config.schemaDEFT}.T_TASK "``) while the
    columns arrive in later ``+=`` fragments, so neither half can be read
    without the other.
    """
    found: list[str] = []
    for _line, _operator, _text, node in _parts(func, name):
        value = written_value(node)
        if not isinstance(value, ast.JoinedStr):
            continue
        found.extend(
            ast.unparse(part.value)
            for part in value.values
            if isinstance(part, ast.FormattedValue)
        )
    return found


def _is_literal(node: ast.expr) -> bool:
    """Whether *node* is a plain string constant."""
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def _gluing_fstring(node: ast.expr) -> Optional[list[ast.expr]]:
    """The operands of an f-string that only joins a statement to its tag.

    ``execute(f"{sql} {comment}")`` is ``execute(sql + " " + comment)`` written
    another way.  Read as one opaque expression it has no head that names a
    statement, so the execution was dropped whole rather than truncated -- and
    a statement that is never read leaves nothing behind for a graph invariant
    to catch.  Five went that way, one of them the only place
    ``JOBSDEFINED_SHARE_STATS`` is named.

    Recognised by its literal parts being nothing but whitespace.  An f-string
    that carries text of its own *is* the statement, and splitting that one
    would keep only the piece before the first hole -- a truncation no reader
    could tell from a short statement.
    """
    if not isinstance(node, ast.JoinedStr):
        return None
    if not any(isinstance(value, ast.FormattedValue) for value in node.values):
        return None
    if any(
        isinstance(value, ast.Constant)
        and isinstance(value.value, str)
        and value.value.strip()
        for value in node.values
    ):
        return None
    return [
        value.value if isinstance(value, ast.FormattedValue) else value
        for value in node.values
    ]


def _concatenated(node: ast.expr) -> list[ast.expr]:
    """Flatten a chain of ``+`` into its operands, left to right."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _concatenated(node.left) + _concatenated(node.right)
    glued = _gluing_fstring(node)
    if glued is not None:
        return [operand for value in glued for operand in _concatenated(value)]
    return [node]


#: ``UPDATE %s SET`` and ``UPDATE ATLAS_PANDA.{0} SET`` -- a table name the call
#: supplies rather than the statement.  Both markers are unambiguous: rendered
#: text spells an f-string interpolation ``{}``, never either of these.
_PERCENT_S = "%s"
_BARE_FIELD = "{}"


def _interpolates(node: ast.expr) -> bool:
    """Whether any brace this expression renders could be a hole, not a brace."""
    return any(
        isinstance(child, ast.JoinedStr)
        or (
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "format"
        )
        for child in ast.walk(node)
    )


def _braces_are_literal(sources: list[ast.stmt]) -> bool:
    """Whether every ``{}`` the fragments in *sources* wrote is a placeholder.

    Rendered text spells an f-string interpolation ``{}`` as well, so a bare
    brace is either the hole a schema went into or a field a ``.format`` at the
    call fills.  The fragments settle it without a guess: if none of them
    interpolates, every brace came from a literal.

    **The fragments of this statement, not every assignment to the name.**
    Asked across the whole function it is a different question, and a bigger
    one than the answer can bear: ``copyArchive.main`` assigns ``sql``
    forty-four times and exactly one of those is an f-string, which disqualified
    the other forty-three.  Among them is the query whose table the loop
    supplies, so ``JOBS_SHARE_STATS`` and ``JOBSDEFINED_SHARE_STATS`` were read
    as ``FROM ATLAS_PANDA.{}`` -- naming no table at all.
    """
    return not any(
        (value := written_value(node)) is not None and _interpolates(value)
        for node in sources
    )


def _elements(node: ast.expr) -> Optional[list[str]]:
    """The strings a literal sequence holds, or ``None`` if one is not written out.

    Rendered rather than required to be constant: the corpus spells a table
    name ``f"{panda_config.schemaPANDA}.jobsActive4"``, and the schema half is
    exactly the part no reader here needs -- :func:`~bamboo.codemap.panda.sqltext._table_of`
    drops it.  One
    element nobody can read sinks the whole sequence, because half a list of
    tables read as the list is a statement about tables the code never names.
    """
    if isinstance(node, ast.Dict):
        # The same list, written so each table can carry what to ask of it:
        # ``{"jobsDefined4": ["defined", "assigned"]}``.  The keys are the
        # sequence.  A ``**other`` entry is a whole key set from somewhere this
        # cannot see, and ``node.keys`` spells it ``None``.
        if any(key is None for key in node.keys):
            return None
        node = ast.List(elts=list(node.keys))
    if not isinstance(node, (ast.Tuple, ast.List)):
        return None
    rendered = [rendered_text(element) for element in node.elts]
    return None if any(text is None for text in rendered) else [str(t) for t in rendered]


def _sequence_values(
    func: ast.FunctionDef | ast.AsyncFunctionDef, name: str
) -> Optional[list[str]]:
    """The strings local sequence *name* holds, across ``=``, ``+=`` and keys.

    ``None`` rather than a partial answer wherever the sequence is built some
    other way -- an ``append``, or a sequence that is not a literal.  A missing
    element is not a smaller answer here: it is a statement the map says the
    code runs over a set of tables it does not.

    Which is why the dict spelling needs both halves.  ``getDispatchDatasets
    PerUser`` writes one table in the literal and adds two more by subscript
    under a flag, so reading the literal alone answers ``jobsDefined4`` -- one
    table of three, and nothing about it says it is one of three.
    """
    found: list[str] = []
    assigned = False
    for node in ast.walk(func):
        if _AT_CALL and ruled_out(node):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == name
            for target in targets_of(node)
        ):
            elements = _elements(node.value)
            if elements is None:
                return None
            found.extend(elements)
            assigned = True
        elif keys := [
            target.slice
            for target in targets_of(node)
            if isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and target.value.id == name
        ]:
            rendered = [rendered_text(key) for key in keys]
            if any(text is None for text in rendered):
                return None
            found.extend(str(text) for text in rendered)
        elif (
            isinstance(node, ast.AugAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
        ):
            elements = _elements(node.value) if isinstance(node.op, ast.Add) else None
            if elements is None:
                return None
            found.extend(elements)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            # ``update`` and ``setdefault`` add keys from somewhere this cannot
            # see; ``pop`` and ``clear`` take them away again.  Read the same
            # way as ``append`` on a list, and for the same reason.
            and node.func.attr
            in {"append", "extend", "insert", "update", "setdefault", "pop", "clear"}
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == name
        ):
            return None
    return found if assigned else None


def _rewritten_values(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    node: ast.expr,
    seen: frozenset[str],
) -> list[str]:
    """The strings ``re.sub(<pattern>, <replacement>, <name>)`` can produce.

    Computed, not inferred.  ``peekJobLog`` derives three archive table names
    from one -- ``re.sub("jobsArchived", "metaTable_ARCH", table)`` -- and all
    three parts are written down: the pattern, the replacement, and the values
    the third argument already resolves to.  Running the substitution on those
    values is the same reading the rest of this function does, not a new guess.

    All three have to be readable.  A pattern or replacement the map cannot see
    would make the result invented, and an invented table name is attributed
    with exactly the confidence of one the code states.
    """
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "sub"
        and len(node.args) >= 3
    ):
        return []
    pattern, replacement = node.args[0], node.args[1]
    if not (_is_literal(pattern) and _is_literal(replacement)):
        return []
    try:
        return [
            re.sub(pattern.value, replacement.value, value)
            for value in _literal_values(func, node.args[2], seen)
        ]
    except re.error:
        # A pattern that will not compile says nothing about the name, and
        # raising here would sink every statement in the function.
        return []


def _default_values(
    func: ast.FunctionDef | ast.AsyncFunctionDef, name: str
) -> list[str]:
    """The string a parameter falls back to, where the signature states it.

    ``insertDataset(self, dataset, tablename="ATLAS_PANDA.Datasets")`` names
    the table it writes, and names it in the function's own signature -- the
    only caller passes nothing.  Reading a parameter with no default would
    need the call graph, which this does not have; reading the default needs
    only the function already in hand.
    """
    arguments = func.args.posonlyargs + func.args.args
    paired = list(
        zip(
            arguments[len(arguments) - len(func.args.defaults) :],
            func.args.defaults,
            strict=True,
        )
    )
    paired += [
        (argument, default)
        for argument, default in zip(
            func.args.kwonlyargs, func.args.kw_defaults, strict=True
        )
        if default is not None
    ]
    return [
        default.value
        for argument, default in paired
        if argument.arg == name and _is_literal(default)
    ]


def _literal_values(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    expression: ast.expr,
    seen: frozenset[str] = frozenset(),
) -> list[str]:
    """Return the strings *expression* can hold, where the source writes them out.

    Five forms, all of which put the names in plain sight: the argument is a
    literal; it is the target of a ``for`` over a literal sequence --
    ``for table in ("ATLAS_PANDA.jobsDefined4", "ATLAS_PANDA.jobsActive4")``;
    or over a local list assembled from literals, which is how the same loop is
    written when a flag decides whether the archive tables are in it; it is
    rewritten from such a value by :func:`_rewritten_values`; or it is a
    parameter whose default the signature states.
    Anything else returns nothing, and the placeholder is left as written: a
    table name invented here would be attributed to a spec class with the same
    confidence as one the code states.

    *seen* stops a name that is defined in terms of itself -- ``x = re.sub(p,
    r, x)`` -- from recurring forever.
    """
    if isinstance(expression, ast.Constant) and isinstance(expression.value, str):
        return [expression.value]
    rewritten = _rewritten_values(func, expression, seen)
    if rewritten:
        return rewritten
    if not isinstance(expression, ast.Name) or expression.id in seen:
        return []
    deeper = seen | {expression.id}
    found: list[str] = _default_values(func, expression.id)
    for node in ast.walk(func):
        if (
            isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id == expression.id
        ):
            elements = (
                _sequence_values(func, node.iter.id)
                if isinstance(node.iter, ast.Name)
                else _elements(node.iter)
            )
            if elements is not None:
                found.extend(elements)
        elif any(
            isinstance(target, ast.Name) and target.id == expression.id
            for target in targets_of(node)
        ):
            if _is_literal(node.value):
                found.append(node.value.value)
            else:
                found.extend(_rewritten_values(func, node.value, deeper))
    return found


#: How far a key substitution follows a name bound from another name.
#: ``tmpScoutKey = f":{scoutKey}"`` is one hop and the corpus asks for no more.
_KEY_HOPS = 2


def _mapping_iterated(node: ast.For) -> tuple[str, Optional[ast.expr]]:
    """The name bound to the key by ``for k, v in d.items()``, and ``d``.

    Only a loop that plainly walks a mapping.  ``for table in (...)`` is a
    sequence and :func:`_literal_values` already answers for it; reading a
    sequence element as a column name would put the *values* of a list into a
    ``SET`` clause.
    """
    over = node.iter
    if not (
        isinstance(over, ast.Call)
        and isinstance(over.func, ast.Attribute)
        and over.func.attr in {"items", "keys"}
        and not over.args
    ):
        return "", None
    target = node.target
    if isinstance(target, ast.Name):
        return target.id, over.func.value
    if (
        isinstance(target, ast.Tuple)
        and target.elts
        and isinstance(target.elts[0], ast.Name)
    ):
        return target.elts[0].id, over.func.value
    return "", None


def _literal_keys(func: ast.FunctionDef | ast.AsyncFunctionDef, name: str) -> list[str]:
    """The keys *func* writes into the mapping it binds to *name*, in source order.

    Source order because that is the order the mapping is built in, and so the
    order the columns appear in the statement.  ``ast.walk`` is breadth-first,
    which would put a key set inside an ``if`` after one set below it.
    """
    found: dict[str, int] = {}
    for node in ast.walk(func):
        for target in targets_of(node):
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == name
                and isinstance(target.slice, ast.Constant)
                and isinstance(target.slice.value, str)
            ):
                line = getattr(node, "lineno", 0)
                found[target.slice.value] = min(
                    found.get(target.slice.value, line), line
                )
    return sorted(found, key=lambda key: (found[key], key))


def _mapping_keys(
    func: ast.FunctionDef | ast.AsyncFunctionDef, mapping: ast.expr
) -> list[str]:
    """The keys the mapping behind *mapping* can hold, read where a call hands it back.

    ``setScoutJobData_JEDI`` builds its ``SET`` clause from the map the scout
    returned, so the columns it writes are stated in ``getScoutJobData_JEDI``
    and nowhere else -- which is why the map had the scout writing no column
    of ``JediTaskSpec`` at all.

    Only where the source says all of it: the mapping is bound in *func* from
    a call, the call lands where the corpus's calls place it (:func:`returned`
    -- the class the receiver is, not a same-named function of the same file),
    and the name returned at the slot the mapping is unpacked from is
    subscripted with literal strings.  A mapping that arrives as a
    **parameter** is not resolved here -- its keys are the caller's to state,
    and five of the corpus's seven mapping loops are that shape.  Returning
    nothing there is the map saying it does not know the columns, which is
    true; filling them in from one caller would make it say something false
    about the others.
    """
    if not isinstance(mapping, ast.Name):
        return []
    for callee, value in returned(func, mapping.id):
        if isinstance(value, ast.Name) and (keys := _literal_keys(callee, value.id)):
            return keys
    return []


def _string_value(
    node: ast.expr, bound: dict[str, str], scope: ast.AST, hops: int = 0
) -> Optional[str]:
    """*node* rendered with the loop's key bound, or ``None`` if it cannot be.

    ``None`` is not "empty": it says this piece of the column name is decided
    at run time, and a hole filled with a guess reads exactly like one the code
    wrote out.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in bound:
            return bound[node.id]
        if hops >= _KEY_HOPS:
            return None
        for other in ast.walk(scope):
            if isinstance(other, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == node.id
                for target in other.targets
            ):
                return _string_value(other.value, bound, scope, hops + 1)
        return None
    if isinstance(node, ast.JoinedStr):
        parts = [
            part.value
            if isinstance(part, ast.Constant) and isinstance(part.value, str)
            else _string_value(part.value, bound, scope, hops)
            if isinstance(part, ast.FormattedValue)
            else None
            for part in node.values
        ]
        return None if any(part is None for part in parts) else "".join(parts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _string_value(node.left, bound, scope, hops)
        right = _string_value(node.right, bound, scope, hops)
        return None if left is None or right is None else left + right
    return None


def _constant_strings(node: ast.expr) -> Optional[list[str]]:
    """The strings of a literal sequence, or ``None`` if it is not one."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if not isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return None
    found = [
        element.value
        for element in node.elts
        if isinstance(element, ast.Constant) and isinstance(element.value, str)
    ]
    return found if len(found) == len(node.elts) else None


def _test_value(
    test: ast.expr, bound: dict[str, str], scope: ast.AST
) -> Optional[bool]:
    """Whether *test* holds for this key, or ``None`` where it cannot be decided.

    The corpus writes the same skip two ways -- ``if scoutKey in ["newNG"]``
    and ``if not k.startswith("walltime") and not k.startswith("ram")`` -- so
    reading only the first leaves the merge loop writing sixteen columns where
    it writes four.  Anything outside these forms answers ``None``, and the
    caller then declines the whole loop rather than assuming the skip never
    fires.
    """
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        inner = _test_value(test.operand, bound, scope)
        return None if inner is None else not inner
    if isinstance(test, ast.BoolOp):
        values = [_test_value(one, bound, scope) for one in test.values]
        if any(one is None for one in values):
            return None
        return all(values) if isinstance(test.op, ast.And) else any(values)
    if isinstance(test, ast.Compare) and len(test.ops) == len(test.comparators) == 1:
        left = _string_value(test.left, bound, scope)
        if left is None:
            return None
        operator, right = test.ops[0], test.comparators[0]
        if isinstance(operator, (ast.In, ast.NotIn)):
            members = _constant_strings(right)
            if members is None:
                return None
            return (left in members) is isinstance(operator, ast.In)
        if isinstance(operator, (ast.Eq, ast.NotEq)):
            other = _string_value(right, bound, scope)
            if other is None:
                return None
            return (left == other) is isinstance(operator, ast.Eq)
        return None
    if (
        isinstance(test, ast.Call)
        and isinstance(test.func, ast.Attribute)
        and test.func.attr in {"startswith", "endswith"}
        and len(test.args) == 1
        and not test.keywords
    ):
        subject = _string_value(test.func.value, bound, scope)
        prefixes = _constant_strings(test.args[0])
        if subject is None or prefixes is None:
            return None
        method = subject.startswith if test.func.attr == "startswith" else subject.endswith
        return any(method(one) for one in prefixes)
    return None


def _kept_keys(loop: ast.For, key: str, keys: list[str]) -> Optional[list[str]]:
    """Of *keys*, the ones the loop does not ``continue`` past.

    ``None`` when the loop skips on something this cannot read.  A skip that is
    missed does not shorten the answer, it falsifies it: the column would be
    reported as written on every pass when the code writes it on none.
    """
    guards: list[ast.expr] = []
    for node in ast.walk(loop):
        if isinstance(node, ast.For) and node is not loop:
            # Its ``continue`` belongs to the inner loop and says nothing
            # about which keys reach the fragment.
            continue
        if isinstance(node, ast.If) and len(node.body) == 1 and isinstance(node.body[0], ast.Continue):
            guards.append(node.test)
        elif isinstance(node, ast.Continue) and not any(
            isinstance(one, ast.If) and one.body == [node] for one in ast.walk(loop)
        ):
            return None
    kept: list[str] = []
    for one in keys:
        bound = {key: one}
        decided = [_test_value(guard, bound, loop) for guard in guards]
        if any(value is None for value in decided):
            return None
        if not any(decided):
            kept.append(one)
    return kept


def _key_fills(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    parts: list[tuple[int, str, str, ast.stmt]],
) -> list[Optional[str]]:
    """For each fragment, the text the loop around it actually appends.

    A loop over a mapping appends *one* fragment per key to *one* statement, so
    the expansion belongs here and not in :func:`_run_variants`: a table name
    the loop supplies makes four statements, and a column name it supplies
    makes one statement four columns wider.  ``None`` leaves the fragment as
    reassembly already reads it.
    """
    fills: list[Optional[str]] = []
    for _line, _operator, text, node in parts:
        value = written_value(node)
        expressions = _hole_expressions(value) if value is not None else None
        loop = next(
            (one for one in ancestors(node) if isinstance(one, ast.For)), None
        )
        if expressions is None or not expressions or loop is None:
            fills.append(None)
            continue
        if len(expressions) != text.count(_BARE_FIELD):
            fills.append(None)
            continue
        key, mapping = _mapping_iterated(loop)
        if not key or mapping is None or not any(
            _reads(expression, key) or isinstance(expression, ast.Name)
            for expression in expressions
        ):
            fills.append(None)
            continue
        keys = _mapping_keys(func, mapping)
        kept = _kept_keys(loop, key, keys) if keys else None
        if not kept:
            fills.append(None)
            continue
        pieces: list[str] = []
        for one in kept:
            bound = {key: one}
            filled = [_string_value(expression, bound, loop) for expression in expressions]
            if any(part is None for part in filled):
                pieces = []
                break
            pieces.append(_fill(text, dict(enumerate(filled))))
        fills.append("".join(pieces) if pieces else None)
    return fills


def spells_key(
    func: ast.FunctionDef | ast.AsyncFunctionDef, node: ast.AST, spelled: ast.expr, key: str
) -> bool:
    """Whether ``<varmap>[<spelled>]`` fills *key* on one pass of its loop.

    The bind side of :func:`_key_fills`.  ``setScoutJobData_JEDI`` writes
    ``varMap[tmpScoutKey] = scoutVal`` beside the fragment that names the
    column, so a statement whose ``SET`` clause now spells ``:cpuTime`` still
    had no value for it: the placeholder was read as computed and passed over.
    Both halves come from the same key, so both are answered by the same
    substitution or neither is.
    """
    loop = next((one for one in ancestors(node) if isinstance(one, ast.For)), None)
    if loop is None:
        return False
    name, mapping = _mapping_iterated(loop)
    if not name or mapping is None:
        return False
    keys = _mapping_keys(func, mapping)
    kept = _kept_keys(loop, name, keys) if keys else None
    if not kept:
        return False
    return any(_string_value(spelled, {name: one}, loop) == key for one in kept)


class KeyColumnCensus(NamedTuple):
    """How many statements name a column from a mapping key, and how many resolve."""

    loops: int
    resolved: int
    from_parameter: int


def key_column_census(func: ast.FunctionDef | ast.AsyncFunctionDef) -> KeyColumnCensus:
    """Count the fragments :func:`_key_fills` is asked about, and how it answered.

    An unresolved one leaves the statement naming no column at all, which looks
    exactly like a statement that writes none.  The count is reported so that
    the two are not read as the same answer, and so that the five the corpus
    leaves to a caller are a number rather than a silence.
    """
    loops = resolved = from_parameter = 0
    arguments = {argument.arg for argument in func.args.args}
    counted: set[int] = set()
    for name in {run.variable for run in executions(func) if run.variable}:
        parts = _parts(func, name)
        for (_line, _operator, _text, node), fill in zip(
            parts, _key_fills(func, parts), strict=True
        ):
            loop = next((one for one in ancestors(node) if isinstance(one, ast.For)), None)
            if loop is None or id(node) in counted:
                continue
            key, mapping = _mapping_iterated(loop)
            if not key or mapping is None:
                continue
            value = written_value(node)
            holes = _hole_expressions(value) if value is not None else None
            # Fed by the key, asked by substituting two of them: a fragment
            # whose only hole is the schema sits inside the loop without the
            # loop having anything to do with its text.
            if not holes or not any(
                _string_value(hole, {key: "\x00one"}, loop)
                != _string_value(hole, {key: "\x00two"}, loop)
                for hole in holes
            ):
                continue
            counted.add(id(node))
            loops += 1
            if fill is not None:
                resolved += 1
            elif isinstance(mapping, ast.Name) and mapping.id in arguments:
                from_parameter += 1
    return KeyColumnCensus(loops, resolved, from_parameter)


def _reads(expression: ast.expr, name: str) -> bool:
    """Whether *expression* reads *name* -- editing what is there, not replacing it."""
    return any(
        isinstance(node, ast.Name) and node.id == name for node in ast.walk(expression)
    )


def _runs_before(site: ast.stmt, call: ast.AST) -> bool:
    """Whether reaching *call* means having run *site* on the same pass.

    Position is not enough.  A rebinding under an arm the call is not in never
    runs for that call, and one inside a loop the call is outside leaves the
    first pass -- where the earlier statement is still what the name holds.
    Both are read as containment: every condition guarding the rebinding also
    guards the call, and every loop holding it holds the call.
    """
    return set(path_condition(site)) <= set(path_condition(call)) and loops_over(
        site
    ) <= loops_over(call)


def _discarded(discards: list[ast.stmt], built: list[ast.stmt], call: ast.AST) -> bool:
    """Whether the statement *built* was thrown away before *call* ran.

    Reachability is positional and path-sensitive and still leaves
    ``copyArchive.main`` claiming nine statements at each of its calls: a
    fragment written at line 532 does reach a call hundreds of lines below,
    and what stops it is the plain ``sql = ...`` in between, each of which
    discards the statement before it.

    From the last fragment, because that is when the statement exists.
    """
    after = max(part.lineno for part in built)
    return any(
        after < part.lineno < getattr(call, "lineno", 0) and _runs_before(part, call)
        for part in discards
    )


#: How the call fills in the statement it runs, as :func:`_call_site_fill` reads it.
_BY_PERCENT = "%"
_BY_FORMAT = "format"


def _substituted(
    text: str,
    sources: list[ast.stmt],
    fillers: list[list[str]],
    spelling: str,
) -> list[str]:
    """Return *text* with the call's arguments filled in, or as it stands."""
    if spelling == _BY_PERCENT:
        return [text.replace(_PERCENT_S, value) for value in fillers[0]] or [text]
    if spelling != _BY_FORMAT:
        # Not a template: a brace here is the statement's own text.
        return [text]
    filled = [text]
    for index, values in enumerate(fillers):
        marker = f"{{{index}}}"
        if not values:
            continue
        if marker in filled[0]:
            filled = [one.replace(marker, value) for one in filled for value in values]
        elif index == 0 and _BARE_FIELD in filled[0] and _braces_are_literal(sources):
            filled = [
                one.replace(_BARE_FIELD, value) for one in filled for value in values
            ]
    # A field no argument filled is a hole like any other, and spelled like
    # one.  Left as ``{0}``, the table in ``getUserJobMetadata``'s ``FROM {0}
    # j,{1} m`` -- filled from a loop over pairs this cannot read -- was taken
    # for a table named ``{0}``, and then for a boundary to a system that keeps
    # it.  Named fields too, which no argument here fills: ``jobStatus IN
    # {q_status_list_str}`` was neither a condition nor a stop, and dropped
    # out of the statement silently.  After the loop, because the markers are
    # what it fills by.
    return [blank_fields(one) for one in filled]


def _call_site_fill(
    func: ast.FunctionDef | ast.AsyncFunctionDef, expression: ast.expr
) -> tuple[ast.expr, list[list[str]], str]:
    """Split ``execute(...)``'s first argument into statement and substitution.

    The third value says how the call fills the statement in: ``_BY_PERCENT``,
    ``_BY_FORMAT``, or ``""`` where it does not.

    Three spellings put the table name at the call rather than in the
    statement, and until now all three were read as though the name were part
    of the text -- which for ``%s`` meant no statement at all, because the
    table pattern does not accept a ``%``.
    """
    glued = _gluing_fstring(expression)
    if glued is not None and sum(1 for value in glued if not _is_literal(value)) > 1:
        # Same rule as the ``+`` spelling below: the last operand is the
        # tracing tag the call appends, so the statement is everything before
        # it.  Only when there is something to drop -- ``f"{sql}"`` carries no
        # tag, and taking one off would leave nothing.
        last = max(index for index, value in enumerate(glued) if not _is_literal(value))
        expression = ast.JoinedStr(
            values=[
                value if _is_literal(value) else ast.FormattedValue(
                    value=value, conversion=-1, format_spec=None
                )
                for value in glued[:last]
            ]
        )
    if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Mod):
        return expression.left, [_literal_values(func, expression.right)], _BY_PERCENT
    inner = (
        expression.left
        if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Add)
        else expression
    )
    if (
        isinstance(inner, ast.Call)
        and isinstance(inner.func, ast.Attribute)
        and inner.func.attr == "format"
    ):
        return (
            inner.func.value,
            [_literal_values(func, argument) for argument in inner.args],
            _BY_FORMAT,
        )
    # ``execute(sqlU + comment, varMap)`` -- the comment is a tracing tag
    # appended at the call, so the statement is everything to its left.
    if isinstance(expression, ast.BinOp):
        return expression.left, [], ""
    return expression, [], ""


def executions(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    skipped: Optional[list["Skip"]] = None,
) -> list[Execution]:
    """Return one :class:`Execution` per cursor execution in *func*.

    The call is what pairs a statement with its binds.  Reading them separately
    -- every statement in the function against every bind in the function --
    would attach a task's status write to a dataset's statement whenever both
    appear in one method, which in ``db_proxy_mods`` is most of them.

    **The statement can arrive in more than two pieces.**  Reading only the
    operand to the left of the comment is right for ``execute(sqlU + comment)``
    and wrong for ``execute(sqlU + sql + comment)``, where the second name holds
    the ``WHERE`` clause -- and it did not merely truncate those statements, it
    dropped them, because the left operand was itself an addition rather than a
    name.  That cost the ``jobStatus`` write in ``updateJobStatus`` and the
    whole of ``updateTask_JEDI``, silently: a statement that is never read
    leaves nothing behind for a graph invariant to catch.

    An operand that says nothing about its own text becomes ``{}``, the same
    hole an f-string interpolation leaves, rather than sinking the statement.
    The head is still required to be a readable name: it is what the statement
    *is*, and without it there is no anchor to hang the execution on.

    **A statement is not run by every call in the function.**  The variants of
    a name are read across the whole function, and pairing each with every
    execution in it is a cross product, not a reading.  ``copyArchive.main``
    assigns ``sql`` forty-four times and runs it twenty-three: 21 of the 31
    statements this build reported as having a table name supplied at run time
    were *one* statement claimed at 21 call sites that never run it, and its
    table is resolved at the call that does.  A fragment below the call, or
    under an arm the call is not in, cannot be part of what the call runs --
    :func:`pathcond.can_reach` is the predicate the walk already uses to decide
    whether a binding can be what an arm read, and this is the same question
    asked of a different pair.

    **A call it reads nothing out of is said, where *skipped* is given.**  Every
    ``continue`` below used to be silent, and the statements behind them were
    in no node, no stop and no count: ``queryTasksToBePending_JEDI`` runs the
    ``SELECT`` its caller hands it, and the one function that moves ``ready``
    tasks to ``pending`` looked like one whose write picks its own rows.  Each
    is a :class:`Skip` with the reason, so a consumer can tell "read in full"
    from "never read" -- see :data:`SKIP_REASONS`.
    """
    # ``_compatible_sets`` reads the arms a fragment sits under off the ancestor
    # chain, and two of this function's callers -- ``boundary`` and ``trigger``
    # -- never attached one.  A tree without parents does not raise: every
    # fragment reads as unconditional, so the two arms of one ``if`` are
    # concatenated into a statement nobody runs and the two the code does run
    # are never seen.  Idempotent, so attaching again costs a walk and nothing
    # else.
    attach_parents(func)
    found: list[Execution] = []
    for node in ast.walk(func):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        forwarded = node.func.attr in _FORWARDED
        if not forwarded and node.func.attr not in {"execute", "executemany"}:
            continue
        if _AT_CALL and ruled_out(node):
            continue
        if not node.args:
            continue
        if len(node.args) < 2:
            if skipped is not None:
                skip = _skip_without_binds(func, node)
                if skip is not None:
                    skipped.append(skip)
            continue
        expression = node.args[0]
        # Filled by the reassembly below, and handed to every Execution this
        # call yields: the variants of one name are variants of one statement,
        # and a fragment the reader could not follow was missing from all of
        # them.
        stops: list[Stop] = []
        statement, fillers, spelling = _call_site_fill(func, _through_name(func, expression))
        operands = _concatenated(statement)
        sources: dict[str, list[ast.stmt]] = {}
        reaching: dict[str, list[ast.stmt]] = {}
        discards: list[ast.stmt] = []
        if isinstance(operands[0], ast.Name):
            base = operands[0]
            head = []
            for text, spans in variant_spans(func, base.id):
                if not text:
                    continue
                head.append(text)
                # Which fragments built this variant, so that filling its holes
                # asks about them and not about every other statement the name
                # has held.
                sources.setdefault(text, [span.node for span in spans])
                # The same fragments, pooled across variants instead of kept
                # for the first.  Two arms of one ``if`` often build the same
                # statement, and asking the first arm's copy whether it reaches
                # a call in the second answers no: ``activateJob`` loses
                # ``sqlF`` and ``sqlJob`` that way, four statements over the
                # corpus.  Filling cannot pool them -- ``_braces_are_literal``
                # is answered per statement, for the reason its docstring
                # gives -- so this is a second reading and not a wider one.
                reaching.setdefault(text, []).extend(span.node for span in spans)
            held = base.id
            # Where the name is thrown away and given a new statement.  A
            # statement that reads its own name is not one of these:
            # ``sql_to_get_ids = sql_to_get_ids[:-1] + ") "`` closes a bracket
            # on what is already there, and counting it cost three real read
            # edges, measured.  It reaches ``_parts`` as a :data:`TRIM` and a
            # ``+=``, not as a ``=``; the rest of that spelling only where it
            # renders -- ``sql = f"{sql} AND x"`` does.
            built = _parts(func, base.id, stops=stops)
            discards = [
                part
                for _line, operator, _text, part in built
                if operator == "=" and not _reads(part.value, base.id)
            ]
        elif forwarded:
            # The daemons write the statement into the call.  ``execute`` is
            # left requiring a name on purpose: widening it would move the
            # stored map and make ``diff-map`` report condition drift where
            # there is none.  Measured, that costs 36 statements on the
            # ``execute`` side and is a deliberate debt, not an oversight.
            rendered = rendered_text(operands[0])
            head = [rendered] if rendered is not None else []
            held = ""
        else:
            if skipped is not None:
                rendered = rendered_text(operands[0])
                skipped.append(
                    Skip(
                        node.lineno,
                        SKIP_INLINE,
                        _clipped(rendered if rendered is not None else _unparsed(operands[0])),
                        (rendered,) if rendered else (),
                    )
                )
            continue
        if not head:
            if skipped is not None:
                skip = _skip_without_text(func, node, operands[0], forwarded)
                if skip is not None:
                    skipped.append(skip)
            continue
        pieces: list[list[str]] = [head]
        for operand in operands[1:]:
            if isinstance(operand, ast.Name):
                pieces.append(
                    variants(func, operand.id) or _handed_texts(func, operand.id, frozenset()) or ["{}"]
                )
                continue
            rendered = rendered_text(operand)
            pieces.append([rendered if rendered is not None else "{}"])
        total = math.prod(len(choices) for choices in pieces)
        if total > _MAX_BRANCH_VARIANTS:
            # Said out loud, for the reason in ``_run_variants``: a silent cap
            # reads exactly like full coverage.  Falling back to the head alone
            # is what this function did for every statement until now.
            logger.warning(
                "the statement executed at line %d joins %d variants of %d name(s), "
                "over the cap of %d: read as %s alone, so its trailing clauses are "
                "not read here",
                node.lineno,
                total,
                len(pieces),
                _MAX_BRANCH_VARIANTS,
                base.id,
            )
            pieces = [head]
        varmap = node.args[1].id if isinstance(node.args[1], ast.Name) else None
        window = (
            binding_window(func, varmap, node.lineno)
            if forwarded and varmap is not None
            else None
        )
        produced = len(found)
        for combination in itertools.product(*pieces):
            built_by = reaching.get(combination[0], [])
            if built_by and not any(can_reach(fragment, node, current=True) for fragment in built_by):
                # Nothing that built this statement can be part of what this
                # call runs.  Guarded on having a fragment at all, because the
                # daemon layer writes the statement into the call and has none:
                # "no fragment reaches here" and "there is no fragment" are
                # different answers, and reading the second as the first would
                # drop 30 pairs.
                continue
            if built_by and _discarded(discards, built_by, node):
                continue
            assembled = "".join(combination)
            for text in _substituted(
                assembled, sources.get(combination[0], []), fillers, spelling
            ):
                found.append(
                    Execution(
                        sql=text,
                        varmap=varmap,
                        variable=held,
                        call=node,
                        window=window,
                        unresolved=tuple(stops) + _hole_stops(text, node.lineno),
                    )
                )
        if skipped is not None and len(found) == produced:
            # Every statement the name held was built somewhere this call
            # cannot be reached from, or thrown away before it: the call runs
            # something, and nothing here says what.
            skipped.append(Skip(node.lineno, SKIP_UNREACHED, held, tuple(head)))
    together: dict[tuple, dict[ast.Call, None]] = {}
    for run in found:
        together.setdefault((run.variable, run.sql, run.varmap, run.window), {})[run.call] = None
    return [
        run._replace(calls=tuple(together[(run.variable, run.sql, run.varmap, run.window)]))
        for run in found
    ]


class Skip(NamedTuple):
    """A cursor execution :func:`executions` saw and read no statement out of.

    ``texts`` is whatever text the statement can be read to hold -- the
    variants of its name, or what it renders to -- so a consumer can still say
    which tables it touches.  Empty when there is none, and then *which rows it
    reads* is unknown rather than "none".
    """

    line: int
    reason: str
    detail: str
    texts: tuple[str, ...] = ()


#: The ways an execution is seen and not read.  Spelled for a reader of the
#: build's stop list, beside the reasons a statement that *was* read stopped.
SKIP_NO_VARMAP = (
    "the statement runs with no bind map, and this build reads a statement only"
    " with its binds"
)
SKIP_HANDED_IN = "the statement is handed in by the caller, so its text is not in this function"
SKIP_UNBOUND = "the statement is a name this function does not bind"
SKIP_UNREAD_NAME = "the statement is a name bound here to text this build cannot render"
SKIP_INLINE = "the statement is written into the call, and this build reads it there only from a name"
SKIP_UNRENDERED = "the statement written into the forwarded call does not render"
SKIP_UNREACHED = "no statement this function builds reaches this call"
SKIP_REASONS = (
    SKIP_NO_VARMAP,
    SKIP_HANDED_IN,
    SKIP_UNBOUND,
    SKIP_UNREAD_NAME,
    SKIP_INLINE,
    SKIP_UNRENDERED,
    SKIP_UNREACHED,
)

#: Functions whose own statement parameter this reader reads at their callers
#: instead: the cursor's own ``execute`` wrapped, and the forwarded form.  The
#: same names :func:`executions` accepts, so a statement handed to one of them
#: is read where it is written, and handing it on is not a gap.
_READ_AT_THE_CALLER = frozenset({"execute", "executemany"}) | frozenset({"querySQL", "querySQLS"})


def _unparsed(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:  # noqa: BLE001 -- unparse fails on synthesised nodes
        return "..."


def _clipped(text: str, width: int = 72) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: width - 1] + "\u2026"


def _head_texts(
    func: ast.FunctionDef | ast.AsyncFunctionDef, head: ast.expr
) -> tuple[str, ...]:
    """What the head of a statement can be read to hold, without pairing binds."""
    if isinstance(head, ast.Name):
        return tuple(text for text in variants(func, head.id) if text)
    rendered = rendered_text(head)
    return (rendered,) if rendered else ()


def _skip_without_text(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    call: ast.Call,
    head: ast.expr,
    forwarded: bool,
) -> Optional[Skip]:
    """Why a call whose statement has no text here was not read, or ``None``.

    ``None`` for a function this reader already reads at its callers --
    ``WrappedCursor.execute(sql, ...)`` hands on a statement every caller
    wrote, and the callers are where it is read.
    """
    if isinstance(head, ast.Name):
        if _is_parameter(func, head.id):
            if func.name in _READ_AT_THE_CALLER:
                return None
            return Skip(call.lineno, SKIP_HANDED_IN, head.id)
        # ``sql = sql1_0 % (JobSpec.columnNames(), table) + sql1_1`` binds the
        # name to something no rendering here follows -- a different fact from
        # a name that came from nowhere in this function, and a different
        # place to go and read.
        bound = any(
            isinstance(node, ast.Name) and node.id == head.id and isinstance(node.ctx, ast.Store)
            for node in ast.walk(func)
        )
        return Skip(call.lineno, SKIP_UNREAD_NAME if bound else SKIP_UNBOUND, head.id)
    if forwarded:
        return Skip(call.lineno, SKIP_UNRENDERED, _clipped(_unparsed(head)))
    return Skip(call.lineno, SKIP_INLINE, _clipped(_unparsed(head)))


def _skip_without_binds(
    func: ast.FunctionDef | ast.AsyncFunctionDef, call: ast.Call
) -> Optional[Skip]:
    """The skip for ``execute(sql)``, which names no bind map to pair it with.

    ``None`` on the same terms as :func:`_skip_without_text`: a wrapper of the
    cursor handing on its caller's statement.
    """
    statement, _fillers, _spelling = _call_site_fill(func, call.args[0])
    head = _concatenated(statement)[0]
    texts = _head_texts(func, head)
    if not texts:
        # No text either: a statement handed in is handed in whether or not a
        # map comes with it, and saying so is the more useful of the two.
        skip = _skip_without_text(func, call, head, call.func.attr in _FORWARDED)
        if skip is None or skip.reason == SKIP_HANDED_IN:
            return skip
    detail = head.id if isinstance(head, ast.Name) else _clipped(texts[0] if texts else _unparsed(head))
    return Skip(call.lineno, SKIP_NO_VARMAP, detail, texts)


def _is_parameter(func: ast.AST, name: str) -> bool:
    args = getattr(func, "args", None)
    if args is None:
        return False
    every = [*args.posonlyargs, *args.args, *args.kwonlyargs]
    every += [one for one in (args.vararg, args.kwarg) if one is not None]
    return any(argument.arg == name for argument in every)


#: How the daemon layer runs a statement.  A second execution form rather than
#: a second reading: the statement, the binds and the pairing between them are
#: the same three things, reached through the task buffer instead of a cursor.
_FORWARDED = frozenset({"querySQL", "querySQLS"})

#: Every call form :func:`executions` will read a statement out of.  Exposed so
#: that a reader reporting which form it *declined* names the same set this one
#: accepts -- a second copy of the list would drift, and the whole point of the
#: read side's denominator is to catch a form nobody added.
RUNNERS = frozenset({"execute", "executemany"}) | _FORWARDED

#: ``RETURNING <column> INTO <bind>``, either side a comma list: the binds the
#: database writes into the map the statement ran with.
_RETURNING_INTO = re.compile(
    r"\bRETURNING\s+[\w.$]+(?:\s*,\s*[\w.$]+)*\s+INTO\s+(:\w+(?:\s*,\s*:\w+)*)", re.IGNORECASE
)


def written_back(
    func: ast.FunctionDef | ast.AsyncFunctionDef, call: ast.Call
) -> Optional[tuple[str, ...]]:
    """The keys of its map that running *call* writes, spelt as the map spells them.

    A reader that cannot see inside the cursor has to take a map handed to
    ``execute`` as rewritten, and then the value an arm put in it is not the
    one printed after the call: ``kickExhaustedTasks_JEDI`` sets
    ``varMap[":newStatus"]`` to ``exhausted`` or ``finishing`` under an ``if``,
    runs the ``UPDATE``, and prints the key where the two arms have joined.
    PanDA's ``WrappedCursor.execute`` touches no key of the map but the binds a
    ``RETURNING ... INTO`` names -- Oracle fills the variable bound there, the
    mysql branch deletes the key and writes it again -- and the statement is
    what names them.

    ``()`` when every statement the call runs was read and none returns into a
    bind.  ``None`` when this cannot say: no statement was read for the call,
    or one is missing a piece that has no text of its own
    (:data:`FRAGMENT_NOT_RENDERED`), which could have held the clause.  A hole
    the statement keeps for a schema name, a value or an ``IN`` list cannot,
    and is read past.
    """
    runs = [run for run in executions(func) if any(one is call for one in run.run_at)]
    if not runs:
        return None
    keys = {
        key.strip()
        for run in runs
        for match in _RETURNING_INTO.finditer(run.sql)
        for key in match.group(1).split(",")
    }
    if keys:
        return tuple(sorted(keys))
    if any(stop.reason == FRAGMENT_NOT_RENDERED for run in runs for stop in run.unresolved):
        return None
    return ()


def written_back_at(path: Path, line: int) -> Optional[tuple[str, ...]]:
    """:func:`written_back` for the ``execute`` or ``executemany`` call on *line* of *path*.

    For a reader holding a file and a line and no tree.  It reads the tree the
    rest of the map reads (:func:`~bamboo.codemap.calls.parsed`), so the call is
    the node the statement was paired with.  ``None`` as well where the file
    does not parse or no such call starts on the line.
    """
    from bamboo.codemap.calls import parsed

    held = parsed(path)
    if held is None:
        return None
    for node in ast.walk(held[1]):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"execute", "executemany"}
            and node.lineno == line
        ):
            func = enclosing_function(node)
            return written_back(func, node) if func is not None else None
    return None


def binding_window(
    func: ast.FunctionDef | ast.AsyncFunctionDef, varmap: str, line: int
) -> tuple[int, int]:
    """Lines whose ``<varmap>[...]`` assignments belong to the run at *line*.

    Read off the code rather than guessed.  ``var_map = {}`` restarts the map,
    so the nearest one above the call is the block boundary the author wrote;
    ``copyArchive.main`` writes it 31 times for 23 statements.  With no such
    assignment the window opens at the top of the function, which is the same
    answer the unwindowed reading gives.

    Public because *where a walk should start* is the question it answers,
    and the answer is as good for one call form as the other.  It no longer
    pairs a bind with a run -- :func:`bound_values` does that from the call,
    and a map rebuilt in each arm of an ``if`` is why: the window opens at the
    second arm and loses the first.  ``execute`` was once thought to need no
    pairing at all, on the argument that it resolves its binds by key; one
    function binds a key for statements on several tables.  Measured
    on the corpus, 216 of the 217 ``execute`` statements get a window here,
    median 6 lines and never the whole function, the same shape as the 47 the
    forwarded form already had.
    """
    start = func.lineno
    for node in ast.walk(func):
        # ``targets`` first: it is what rules out the nodes with no line number
        # at all, so testing it before ``lineno`` is what keeps this safe.
        targets = targets_of(node)
        if not targets or node.lineno >= line:
            continue
        if not isinstance(node.value, (ast.Dict, ast.DictComp)) and not (
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "dict"
        ):
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id == varmap:
                start = max(start, node.lineno)
    return start, line


#: ``jobStatus=:oldJobStatus`` -- the only shape of precondition whose value
#: is in Python.  ``commandToPilot IS NULL`` and ``nFiles>nFilesUsed`` name no
#: bind, and there is nothing to settle in them.
_PRECONDITION_BIND = re.compile(r"^([\w.]+)\s*(=|<>|!=)\s*(:\w+)\s*$")


def settled_precondition(
    term: str,
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    varmap: Optional[str],
    window: Optional[tuple[int, int]],
    settle,
    beside: Optional[ast.stmt] = None,
    at: Optional[ast.AST | tuple[ast.AST, ...]] = None,
) -> tuple[str, bool]:
    """*term* with its bind replaced by the value, and whether that happened.

    ``jobStatus=:oldJobStatus`` is what the statement says and ``activated``
    is what it means, and only the second can be compared with what a reader
    selects on.  ``getJobs`` -- the one query that moves a job from
    ``activated`` to ``sent``, the whole of the pilot's job request -- records
    the first, so the map holds its guard and cannot tell you which value it
    guards.

    The raw term is returned unchanged when nothing settles, and that is the
    honest answer rather than a failure: the bind is filled by an argument or
    by a callee, and ``jobStatus=:oldJobStatus`` still says truly that this
    column is being tested.  The caller is told which of the two it got so it
    can say so.

    Several values is not ambiguity -- a method that runs one statement twice
    binds two -- so they are rendered as the ``IN`` the statement would have
    spelled had they been literals, and a negated test as ``NOT IN``.

    *beside* is one branch's deciding statement: only the binds that can run
    with it are read.  ``getTasksToExecCommand_JEDI`` resumes a job
    (``throttled`` -> ``activated``) in one arm and pauses it in the other,
    and one ``IN (activated,throttled)`` for both said the resume could move
    an ``activated`` row.

    *at* is the execution, and only its own binds are read -- see
    :func:`bound_values`.
    """
    match = _PRECONDITION_BIND.match(term.strip())
    if match is None or varmap is None:
        return term, False
    column, operator, key = match.groups()
    found: set[str] = set()
    for bind in bound_values(func, varmap, key, window, at=at):
        if beside is not None and parted(bind.site, beside, exits=False):
            continue
        found.update(settle(bind.value, func))
    if not found:
        return term, False
    negated = operator in ("<>", "!=")
    if len(found) == 1:
        value = next(iter(found))
        return f"{column}{'<>' if negated else '='}{value}", True
    inside = ",".join(sorted(found))
    return f"{column} {'NOT IN' if negated else 'IN'} ({inside})", True


def fragment_holding(spans: list[Span], offset: int) -> Optional[ast.stmt]:
    """The fragment whose stretch of the statement contains *offset*."""
    for span in spans:
        if span.start < offset <= span.end:
            return span.node
    return None


# ``f"INSERT INTO ATLAS_PANDA.filesTable4 ({FileSpec.columnNames()}) "`` -- the
# statement names the spec class that owns the row, in the interpolation.
_COLUMN_NAMES = re.compile(r"^(\w+)\.column_?[Nn]ames\s*\(")


def declared_row_classes(
    func: ast.FunctionDef | ast.AsyncFunctionDef, spec_classes: set[str]
) -> dict[str, str]:
    """Return ``{table: spec class}`` the statements in *func* state outright.

    Stronger than inferring a table's class from its column names, and it
    settles the case inference cannot: ``filesTable4``'s widest ``UPDATE`` sets
    four columns that ``FileSpec`` and ``JediFileSpec`` both declare, so no
    column set ever separates them -- while its ``INSERT`` names ``FileSpec``
    in the f-string.

    Read here rather than during reassembly because reassembly deliberately
    blanks interpolations: they are usually the schema, and keeping them would
    make every statement unique.  This looks at the one interpolation that
    carries meaning instead of at all of them.
    """
    found: dict[str, str] = {}
    for node in ast.walk(func):
        if not isinstance(node, ast.JoinedStr):
            continue
        text = ""
        classes: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                text += value.value
                continue
            text += "{}"
            if isinstance(value, ast.FormattedValue):
                match = _COLUMN_NAMES.match(ast.unparse(value.value))
                if match and match.group(1) in spec_classes:
                    classes.append(match.group(1))
        if len(classes) != 1:
            continue
        named = sqltext.tables_named(text)
        if not named:
            continue
        table = named[0]
        # ``{}`` means the table name was itself interpolated, so the statement
        # does not say which table this is.
        if table and table != "{}":
            found[table] = classes[0]
    return found


class Bind(NamedTuple):
    """One value filling one placeholder, and the statement that put it there.

    ``value`` is what the placeholder receives; ``site`` is the statement to
    ask for the guards that reached it.  They are the same node for
    ``varMap[":x"] = v`` and different ones for a dict literal, which is the
    whole reason this is a pair rather than the assignment.
    """

    value: ast.expr
    site: ast.stmt


def bound_values(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    varmap: str,
    key: str,
    window: Optional[tuple[int, int]] = None,
    across: bool = False,
    at: Optional[ast.AST | tuple[ast.AST, ...]] = None,
) -> list[Bind]:
    """Return the assignments filling ``<varmap>[<key>]`` in *func*.

    Several are normal and correct: a method that runs the same statement twice
    with different values has two, and both are outcomes of that write.

    *at* is the execution the binds are read for -- a call, or the calls that
    run one statement (:attr:`Execution.run_at`) -- and only the binds the map
    it runs with can still hold are returned; see :func:`_live_at`.  Every
    reader of a run asks this way, in both call forms.  Reading every bind of
    the key instead was once thought safe for ``execute``, on the argument that
    the binds reachable by key are exactly the values that write site can
    produce.  They are not: one function binds ``:status`` for statements on
    several tables, and ``retryTask_JEDI`` gave ``JEDI_Tasks``,
    ``JEDI_Datasets`` and ``T_TASK`` the same 32 status writes, among them the
    dataset's ``ready`` and ``done`` as the task's.

    *window*, with no *at*, keeps the binds on lines inside it.  The bounds are
    the ones :func:`binding_window` reads off the code; they say where a walk
    should start, and are too coarse to pair a bind with a run -- a map rebuilt
    in each arm of an ``if`` opens the window at the second arm and loses the
    first.

    **Two spellings, one meaning.**  ``varMap[":status"] = v`` and ``var_map =
    {":status": v}`` bind the same placeholder to the same value, and reading
    only the first left 176 of the corpus's 1261 written columns unexplained
    -- 56% of everything this slice could not account for, and the whole of
    the harvester, worker and data-carousel side, which writes its binds as
    dict literals throughout.  A :class:`Bind` rather than the assignment
    itself because the two forms do not share a node: what the caller needs is
    the value, and the statement it sits in for the guards that reached it.
    """
    return scan_binds(func, varmap, key, window, across, at=at).binds


class BindScan(NamedTuple):
    """The binds one pass found, and what that same pass saw on the way.

    The census travels with the binds so that "nothing found" can say *which*
    of several reasons it was, from the walk that failed rather than from a
    second walk asking why.  A separate reader of the same nodes is free to
    disagree with this one, and this corpus already holds that shape: a call to
    ``reaching_modules`` with the flags and another without, under a comment
    warning that a second reading would drift.

    Everything but :attr:`binds` ignores *window*: the question those fields
    answer is about the function, not about one execution's slice of it.
    """

    binds: list[Bind]
    varmap_assigned: bool = False
    """The varmap name is assigned somewhere in this function.

    False means the map arrived from outside -- a parameter, or an element of
    something iterated, as ``hs_scrapers`` does with a row of a parsed table.
    Nothing here decided those values.
    """

    computed_keys: int = 0
    """Subscript assignments into the varmap whose key is not a constant.

    ``var_map[f":jtid{i}"] = jedi_task_id``.  The placeholder is spelled by an
    expression, so pairing it with one the statement spells needs the loop read
    as well as the assignment.
    """

    constant_keys: frozenset[str] = frozenset()
    """Placeholders this function does bind by a spelled-out name."""

    outside_window: int = 0
    """Binds for this key that the run's map cannot hold, or that lay outside *window*.

    Not a gap: one function binds a key for several statements, and without
    the pairing each statement claims every value in it.  Counted so that a
    bind excluded on purpose is not reported as one nobody wrote.
    """


def scan_binds(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    varmap: str,
    key: str,
    window: Optional[tuple[int, int]] = None,
    across: bool = False,
    hops: int = 0,
    at: Optional[ast.AST | tuple[ast.AST, ...]] = None,
) -> BindScan:
    """:func:`bound_values`, with the census of what the walk passed over.

    One walk, so the binds and the account of why there are none cannot
    disagree.  See :class:`BindScan` for what is counted and what ignores
    *window* and *at*.

    *across* also reads a varmap this function copies from another name
    (``getValMap = copy.copy(getValMapOrig)``) and one a call hands it
    (``..., getValMapOrig = build_where_clause()``, inside
    :func:`across_calls`).  A bind found that way sits in the other function,
    so its ``site`` is not a line of *func*: a caller anchoring on the site
    asks for it only where it reads the value and nothing else.
    """
    found: list[Bind] = []
    assigned = False
    computed = 0
    spelled_keys: set[str] = set()
    excluded = 0
    # Each bind with the write that decides whether the run's map holds it:
    # its own assignment, or the copy that brought it into *varmap*.  Kept
    # apart from *found* because which of them survive is asked of all of
    # them at once -- a later write is what cancels an earlier one.
    gated: list[tuple[Bind, ast.AST]] = []

    def keep(bind: Bind, gate: ast.AST) -> None:
        nonlocal excluded
        if at is not None:
            gated.append((bind, gate))
        elif window is None or window[0] <= gate.lineno <= window[1]:
            found.append(bind)
        else:
            excluded += 1

    lists = {one.values_map: one for one in in_lists(func).values()}
    for node in ast.walk(func):
        if _AT_CALL and ruled_out(node):
            continue
        if (
            lists
            and isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "update"
            and isinstance(node.value.func.value, ast.Name)
            and node.value.func.value.id == varmap
            and len(node.value.args) == 1
            and isinstance(node.value.args[0], ast.Name)
            and node.value.args[0].id in lists
        ):
            # ``varMap.update(stat_var_map)``: the list's values, under the
            # one placeholder the statement spells it as.
            held = lists[node.value.args[0].id]
            assigned = True
            spelled_keys.add(held.bind)
            if held.bind == key:
                keep(Bind(held.values, node), node)
            continue
        targets = targets_of(node)
        if not targets:
            continue
        for target in targets:
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == varmap
            ):
                assigned = True
                if not isinstance(target.slice, ast.Constant):
                    # A key the loop computes -- ``varMap[tmpScoutKey] = val``.
                    # Resolvable exactly where the statement's own column is:
                    # see :func:`spells_key`.  Where it is not, it names a
                    # placeholder the statement does not spell either, so there
                    # is nothing to pair it with.
                    if not spells_key(func, node, target.slice, key):
                        computed += 1
                        continue
                    spelled_keys.add(key)
                    keep(Bind(node.value, node), node)
                    continue
                spelled_keys.add(target.slice.value)
                if target.slice.value != key:
                    continue
                keep(Bind(node.value, node), node)
            elif isinstance(target, ast.Name) and target.id == varmap:
                assigned = True
                copied = _copied_from(node.value) if across else None
                if copied is not None and hops < RETURN_HOPS:
                    # What the copy holds is what the original held when the
                    # copy was taken, so the original's binds are paired with
                    # the copy rather than with the run.
                    inner = scan_binds(
                        func, copied, key, None, across, hops + 1, at=node if at is not None else None
                    )
                    spelled_keys |= inner.constant_keys
                    for bind in inner.binds:
                        keep(bind, node)
                    continue
                # The dict literal form.  Only constant keys: a computed one
                # (``var_map[f":{column}"] = val``, which the worker modules
                # also use) names a placeholder the statement itself does not
                # spell either, so there is nothing to pair it with.
                if not isinstance(node.value, ast.Dict):
                    continue
                for spelled, value in zip(
                    node.value.keys, node.value.values, strict=True
                ):
                    if not isinstance(spelled, ast.Constant):
                        computed += 1
                        continue
                    spelled_keys.add(spelled.value)
                    if spelled.value != key:
                        continue
                    keep(Bind(value, node), node)

    if gated:
        writes = [gate for _bind, gate in gated]
        live: set[int] = set()
        for one in at if isinstance(at, tuple) else (at,):
            live |= _live_at(func, varmap, writes, one)
        for bind, gate in gated:
            if id(gate) in live:
                found.append(bind)
            else:
                excluded += 1

    if across and hops < RETURN_HOPS:
        for callee, value in returned(func, varmap):
            if not isinstance(value, ast.Name):
                continue
            assigned = True
            inner = scan_binds(callee, value.id, key, None, across, hops + 1)
            spelled_keys |= inner.constant_keys
            found.extend(inner.binds)
    if across:
        # One callee's bind reaches here once per call that hands its varmap
        # over, and once per copy: ``getJobs`` builds its ``WHERE`` in two
        # places from one closure.
        unique: dict[tuple[int, int], Bind] = {}
        for bind in found:
            unique.setdefault((id(bind.value), id(bind.site)), bind)
        found = list(unique.values())

    return BindScan(
        binds=found,
        varmap_assigned=assigned,
        computed_keys=computed,
        constant_keys=frozenset(spelled_keys),
        outside_window=excluded,
    )


def _renews(func: ast.FunctionDef | ast.AsyncFunctionDef, varmap: str) -> list[ast.stmt]:
    """The statements of *func* that give *varmap* a new map.

    ``varMap = {}`` and every other assignment to the name, and a loop over
    maps -- ``for varMap in varMapList:`` hands the body a different one each
    time round.  A subscript assignment is not one: it changes the map it
    finds, and neither is ``+=``, which adds to it.
    """
    found: list[ast.stmt] = []
    for binding in bindings(func, varmap):
        if binding.kind in (BIND_ASSIGN, BIND_ANNOTATED, BIND_FOR) and all(
            binding.site is not one for one in found
        ):
            found.append(binding.site)  # type: ignore[arg-type]
    return found


def _appended(func: ast.AST, listing: str, varmap: str) -> list[ast.stmt]:
    """The ``<listing>.append(<varmap>)`` statements of *func*."""
    return [
        node
        for node in ast.walk(func)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "append"
        and isinstance(node.value.func.value, ast.Name)
        and node.value.func.value.id == listing
        and len(node.value.args) == 1
        and isinstance(node.value.args[0], ast.Name)
        and node.value.args[0].id == varmap
    ]


def _live_at(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    varmap: str,
    writes: list[ast.AST],
    at: ast.AST,
) -> set[int]:
    """The ``id`` of each of *writes* whose bind the map *at* runs with can still hold.

    The same reading :func:`pathcond.surviving_writes` makes of a local, with
    the map's own renewals among the writes: ``varMap = {}`` cancels every
    bind above it on the way to the run, and a later bind of the same key
    cancels an earlier one.  Two arms that each rebuild the map are both still
    live after the ``if``, which is what a window opening at the nearest
    rebuild could not say.  So is the loop's way round: ``copyArchive.main``
    starts its loop body with ``var_map = {}`` and binds ``:site`` further
    down, and the bind from the last pass does not reach the first run of the
    next.

    Read as the function runs (``as_run``), as the walk reads it.  Read as
    written, a test on a name reassigned since describes a different value,
    and ``prepareTasksToBeFinished_JEDI`` -- under ``if not toSkip:``, then
    in the ``else`` of a second ``if not toSkip:`` after ``toSkip = True`` --
    had its unlock's own ``:status`` bind ruled unable to reach the unlock.

    A run inside ``for varMap in varMapList:`` gets its map from the list, so
    its binds are the ones live where a map was appended to that list --
    ``prepareTasksToBeFinished_JEDI`` fills four statuses for the input
    datasets that way and runs them after the loop that built them.
    """
    renewals = _renews(func, varmap)
    sites = list({id(site): site for site in [*writes, *renewals]}.values())
    rereads = [write for write in writes if _rereads(write, varmap)]

    def surviving(run: ast.AST) -> set[int]:
        return {id(site) for site in surviving_writes(sites, run, rereads, as_run=True)}

    live = surviving(at)
    for loop in renewals:
        if (
            isinstance(loop, (ast.For, ast.AsyncFor))
            and id(loop) in live
            and isinstance(loop.iter, ast.Name)
            and any(ancestor is loop for ancestor in ancestors(at))
        ):
            for append in _appended(func, loop.iter.id, varmap):
                live |= surviving(append)
    return live


def _rereads(write: ast.AST, varmap: str) -> bool:
    """Whether *write* fills a key of *varmap* from what that key held.

    ``varMap[":diag"] = JobSpec.truncateStringAttr("supErrorDiag",
    varMap[":diag"])`` keeps the text above it, shortened, so it cancels
    nothing -- as ``x += y`` does in :mod:`~bamboo.codemap.panda.reaching`.  Read as
    a replacement it took ``updateJobStatus``'s diagnostic template with it.
    """
    targets = targets_of(write)
    if len(targets) != 1 or not isinstance(targets[0], ast.Subscript):
        return False
    key = targets[0].slice
    if not isinstance(key, ast.Constant):
        return False
    return any(
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == varmap
        and isinstance(node.slice, ast.Constant)
        and node.slice.value == key.value
        for node in ast.walk(write.value)  # type: ignore[attr-defined]
    )


def inline_binds(call: ast.Call, key: Optional[str] = None) -> list[Bind]:
    """The binds a dict literal written into the call fills: ``querySQLS(sql, {":name": name})``.

    All of them, or those for *key*.  The site is the statement holding the
    call, which is where the value is chosen.
    """
    if len(call.args) < 2 or not isinstance(call.args[1], ast.Dict):
        return []
    site = call
    for ancestor in ancestors(call):
        if isinstance(ancestor, ast.stmt):
            site = ancestor
            break
    return [
        Bind(value, site)  # type: ignore[arg-type]
        for spelled, value in zip(call.args[1].keys, call.args[1].values, strict=True)
        if isinstance(spelled, ast.Constant) and (key is None or spelled.value == key)
    ]


def _copied_from(value: ast.expr) -> Optional[str]:
    """The name a varmap is a copy of: ``copy.copy(x)``, ``copy.deepcopy(x)``, ``dict(x)``, ``x.copy()``."""
    if not isinstance(value, ast.Call):
        return None
    called = value.func
    if (
        isinstance(called, ast.Attribute)
        and called.attr == "copy"
        and not value.args
        and isinstance(called.value, ast.Name)
        and called.value.id != "copy"
    ):
        return called.value.id
    spelled = ast.unparse(called)
    if spelled in {"copy.copy", "copy.deepcopy", "dict"} and len(value.args) == 1:
        if isinstance(value.args[0], ast.Name):
            return value.args[0].id
    return None
