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

**The statement has to be put together from the Python.**  It is built by
``=`` then a run of ``+=``, often with an f-string holding the schema, through
``%``, ``.format``, loops and the calls that hand pieces in, and interleaved
with other statements being built in the same function.  It is read by running
the function on every path (:mod:`~bamboo.codemap.panda.sqleval`): what
reaches a call that runs a statement is what that call runs.

**The statement is the wrong unit for reading a branch.**  The fragments are
appended conditionally -- ``getTasksToExecCommand_JEDI`` appends either ``SET
status=:status`` or ``SET status=oldStatus`` -- so the ``if`` that explains the
value is on the fragment, not on the statement.  Each stretch of a statement
keeps the way it was written (:attr:`Execution.pieces`).

**The bind key does not say whether it is a write.**  ``:status`` appears in
``SET status=:status`` *and* in another statement's ``WHERE status=:status``
in the same function -- the flagship ``updateTaskStatusByContFeeder_JEDI`` does
exactly this.  Only the ``SET`` clause distinguishes a write from a predicate,
so the statement is required; the bind key alone is not enough.

**Which spec a table holds is not declared anywhere**, so it is inferred from
the column names -- see ``SpecAttributor.learn_table_classes``.

What the text says -- tables, columns, values, order -- is read by
:mod:`~bamboo.codemap.panda.sqltext`.
"""

from __future__ import annotations

import ast
import collections
import contextlib
import logging
import re
import string
from typing import TYPE_CHECKING, Iterator, NamedTuple, Optional, Sequence

from bamboo.codemap.panda import sqleval, sqltext
from bamboo.codemap.panda.pathcond import (
    BIND_ANNOTATED,
    BIND_ASSIGN,
    BIND_FOR,
    Replacement,
    ancestors,
    attach_parents,
    bindings,
    carried_round,
    enclosing_function,
    only_through,
    parted,
    replacements,
    ruled_out_by,
    site_of,
    surviving_writes,
    targets_of,
    written_value,
)
from bamboo.codemap.panda.values import rendered_text

if TYPE_CHECKING:
    from pathlib import Path

    from bamboo.codemap.calls import Calls, Placed

logger = logging.getLogger(__name__)


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

    A reading always has places it cannot follow; what it needs is a way to
    say so.  A fragment with no text of its own was once dropped and the
    statement read on without it,
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

#: ``AND {withCriteria}`` -- ``updateDataset`` appends whatever condition its
#: caller wrote; ``do_preassign`` three of its own, built apart.  Neither a
#: column nor a value: the condition itself is not in the text.
CONDITION_IS_A_HOLE = "the WHERE takes whole conditions through an interpolation"

#: Where the binds go, as opposed to where the schema does.  ``{}.JEDI_Tasks``
#: is a hole too and is not a gap: the reading blanks the schema on purpose.
_HOLE = re.compile(r"\{\d*\}")


def _hole_stops(sql: str, line: int) -> tuple[Stop, ...]:
    """One :class:`Stop` per ``WHERE`` term the reading left a hole in.

    Read off the assembled text rather than off a fragment, because that is
    where it is decidable, and the four holes a ``WHERE`` can carry are four
    different answers:

    * **The schema**, ``{}.JEDI_Datasets`` -- blanked on purpose, and not a
      gap.  Stripped first, and stripping it is what separates the other two
      from the 19 statements whose only hole is a schema inside a subquery.
    * **The placeholders**, ``jobStatus IN ({})`` -- the column is readable
      and the values are at whichever call site supplied the list.
    * **The column**, ``{}=:{}`` -- the statement names its column from a
      mapping key, which :func:`key_column_census` counts from the other end.
    * **The condition**, a term that is nothing but holes -- ``AND {}`` --
      where the code interpolates a whole condition, column and value alike.

    The term is kept whole, ``AND``/``OR`` to ``AND``/``OR``, so the consumer
    is told which column it is about.  Every ``WHERE`` of the statement is
    read -- each branch of a ``UNION``, the clause around a derived table
    (:func:`sqltext.where_clauses`) -- and a term read in two clauses, a
    subquery's inside its outer clause and on its own, is one stop.  A term
    ends with its own clause, and at the parenthesis closing the subquery it
    sits in (:func:`sqltext.own_clause_terms`).
    """
    found: set[Stop] = set()
    terms = [
        piece
        for clause in sqltext.where_clauses(sql)
        for term in _CLAUSE_JOIN.split(_SCHEMA_HOLE.sub("", clause))
        for piece in sqltext.own_clause_terms(term)
    ]
    for term in terms:
        if not _HOLE.search(term):
            continue
        text = _balanced(" ".join(term.split()))
        if not _HOLE.sub("", text).strip(" ()"):
            reason = CONDITION_IS_A_HOLE
        elif _HOLE.match(text):
            reason = COLUMN_IS_A_HOLE
        else:
            reason = BINDS_ARE_A_HOLE
        found.add(Stop(line, text, reason))
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
        found[names] = InList(names, values_map, node.value.args[0], f"{prefix}{sqltext.IN_LIST_SUFFIX}", node)
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
    held = sqleval.use_runners(frozenset(runners()))
    try:
        yield
    finally:
        _CALLS = previous
        sqleval.use_runners(held)


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
    corpus's calls do not place -- or place open, where what comes back may be
    another function's: an empty answer is the one every reader here takes
    for "not known", and a partial one would read as all of it.
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
    placed = _CALLS.callees(file, func, call)
    if placed.unplaced or placed.returns_none:
        return []
    found: list[tuple[ast.FunctionDef | ast.AsyncFunctionDef, ast.expr]] = []
    for callee in placed.targets:
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
    if not placed.targets or placed.unplaced or placed.returns_none:
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


# A field marker whose value is derived from an argument rather than being one
# -- ``{0.attr}``, ``{0[1]}``, ``{p:.3%}``, or an index the call passes nothing
# for.  It holds a position so the markers and the ``{}``s stay in step, and
# resolves to nothing, which is what a hole nobody can read is worth.
_DERIVED_FIELD = ast.Constant(value=None)


def _format_arguments(node: ast.Call) -> Optional[list[ast.expr]]:
    """The expression behind each field marker of ``<text>.format(...)``.

    One entry per marker in *source order, with repeats*, not one per distinct
    argument: ``"FROM {0}.{1} j, {0}.Datasets d"`` has three markers and two
    arguments, and the list is measured against the number of ``{}`` the
    rendered text carries.  A list of distinct arguments is one short, and a
    fragment whose count does not match is dropped rather than truncated --
    silently, which is how a table name goes missing.

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
    # Measured here and not only on the whole statement: composed under a
    # ``+``, a list of the wrong length can be summed with a
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


def variant_spans(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    name: str,
    at: ast.Call,
    text: Optional[str] = None,
    read: bool = False,
) -> list[tuple[str, list[Span]]]:
    """Every statement *name* holds at *at*, with each statement's fragments and where they landed.

    *at* is the call that runs *name*.  The answer is what *name* holds there
    -- the texts that reach it, each with the statement that put each stretch
    into *name* -- and *text*, where given, is the one statement asked
    about: ``updateSiteData`` copies ``sqlIn`` into ``sql`` in two loops, and
    read by name the call in the first was handed the copy in the second.

    With *read*, each statement is given as a reader takes it
    (:func:`reading_text`), its spans still positions in the statement's own
    text: what sqltext reads off the one, it places in the other.
    """
    return _evaluated_spans(func, name, at, text, read)


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

    head: str = ""
    """The text of :attr:`variable` this run was built from, before the call adds to it.

    One of the name's variants, verbatim.  ``execute((sql + comment) % table,
    ...)`` runs it with the comment after it and ``FROM {}`` filled in, so
    :attr:`sql` is no variant's text, and a reader asking which fragments
    built the run would otherwise have to guess by comparing spellings.
    """

    pieces: tuple = ()
    """Each stretch of :attr:`sql` with its way to the run: ``(start, end, way)``.

    *way* is :func:`sqleval.way_of`: where the stretch was written, then each
    statement that put it into a name, each callee ``return`` and each call it
    came back from.
    """

    holes: tuple = ()
    """Each ``{}`` of :attr:`sql` in order: ``(offset, sqleval.HoleAt)``.

    Where a loop of unknown length appended, the hole keeps the loop and what
    one pass appends (:attr:`sqleval.HoleAt.repeats`): ``getJobs``' ``(CASE {}
    ELSE {} END)``, filled with one ``WHEN gshare=:leave{} THEN {}`` per share.
    """

    unread_columns: tuple[Stop, ...] = ()
    """Each column this run writes that :func:`sqltext.writes` does not return, and why (:func:`sqltext.unread_columns`).

    Kept apart from :attr:`unresolved`, which says where reading the rows the
    statement picks stopped: a read site carries those, and a gate takes a
    statement with one as read only in part.  A ``SET`` naming its column at
    run time picks no fewer rows.  Both are stops on the statement
    (``sqlwrite._run_stops``).  The line is the statement's, *source* the item
    as the statement spells it -- ``SET {}=:{}``, ``({})`` -- so a reading of
    one file and the build's say the same.
    """

    @property
    def run_at(self) -> tuple[ast.Call, ...]:
        """The calls to read this run's binds at -- :attr:`calls`, or its own."""
        return self.calls or (self.call,)

    @property
    def reading(self) -> str:
        """:attr:`sql` as a reader of the statement takes it: each part a loop of unknown length repeats as one
        pass of each text the loop appends (:func:`reading_text`).  :attr:`sql` itself where nothing repeats."""
        return reading_text(self.sql, self.holes)

    @property
    def reading_holes(self) -> tuple:
        """Each hole of :attr:`reading` in order, as sqltext numbers them (``__h1`` the first): the
        :class:`sqleval.HoleAt` it is, a repeated part's own holes included (:attr:`sqleval.HoleAt.inner`), else
        ``None``."""
        return reading_holes(self.sql, self.holes)


def reading_holes(text: str, holes: tuple) -> tuple:
    """:func:`reading_text`'s holes, each ``{}`` of it in order: the :class:`sqleval.HoleAt` it is, else ``None``.

    Out of a repeated part, the hole *holes* has at its offset; in one, the hole
    the text a pass appends has there (:attr:`sqleval.HoleAt.inner`).  Counted
    as sqltext counts the holes it names -- every ``{}`` and ``{0}`` of the
    reading, a literal one too, which is a hole no one wrote.
    """
    out, at_offset = _reading(text, holes)
    return tuple(at_offset.get(match.start()) for match in _READING_HOLE.finditer(out))


def reading_text(text: str, holes: tuple) -> str:
    """*text* with each ``{}`` a loop of unknown length repeats (:data:`sqleval.REPEATED`) as one pass of each text
    the loop appends, between :data:`sqltext.REPEAT_OPEN` and :data:`sqltext.REPEAT_CLOSE`.

    *holes* are *text*'s ``(offset, sqleval.HoleAt)``.  The passes are run
    together, or joined as the list holding them was, less the end the hole
    leaves after it (:attr:`sqleval.HoleAt.end`): ``jediTaskID=:jediTaskID
    {}`` reads ``jediTaskID=:jediTaskID AND {}=:{}``, two terms, where the
    ``{}`` glued the first one to the hole.  sqltext reads the statement with
    them and says what it read in *text*'s terms (:data:`sqltext.REPEAT_OPEN`).
    """
    return _reading(text, holes)[0]


#: What sqltext reads as a hole of a statement (``sqltext._HOLE``).
_READING_HOLE = re.compile(r"\{\d*\}")


def _reading(text: str, holes: tuple) -> tuple[str, dict]:
    """:func:`reading_text`, and the hole at each offset of it."""
    out = text
    at_offset: dict[int, object] = {}
    shift = 0
    # Front to back: each repeated part moves what follows it on by what it put in place of its ``{}``.
    for at, hole in sorted(holes, key=lambda one: one[0]):
        where = at + shift
        if hole.reason != sqleval.REPEATED or not hole.repeats or out[where : where + 2] != "{}":
            at_offset[where] = hole
            continue
        passes, inner = _passes(hole)
        out = out[:where] + sqltext.REPEAT_OPEN + passes + sqltext.REPEAT_CLOSE + out[where + 2 :]
        for offset, one in inner:
            at_offset[where + 1 + offset] = one
        shift += len(passes)
    return out, at_offset


def _passes(hole) -> tuple[str, tuple]:
    """What a repeated part holds in a reading -- one pass of each text, run together or joined, less its end --
    and its holes, ``(offset, HoleAt)``."""
    inner = hole.inner if len(hole.inner) == len(hole.repeats) else ((),) * len(hole.repeats)
    joiner = (hole.joiner or "") if hole.listed else ""
    passes = ""
    found: list = []
    for index, (text, held) in enumerate(zip(hole.repeats, inner, strict=True)):
        if index:
            passes += joiner
        found += [(len(passes) + offset, one) for offset, one in held]
        passes += text
    if hole.end and passes.endswith(hole.end):
        passes = passes[: -len(hole.end)]
    return passes, tuple((offset, one) for offset, one in found if offset < len(passes))


def interpolations(func: ast.FunctionDef | ast.AsyncFunctionDef, name: str) -> list[str]:
    """Return the expressions interpolated into the statements the calls holding *name* run.

    The text blanks them, which is right for reading columns and wrong for
    reading *whose* table this is: the schema is interpolated into the head of
    the statement (``f"UPDATE {panda_config.schemaDEFT}.T_TASK "``) while the
    columns arrive in later ``+=`` fragments, so neither half can be read
    without the other.  They are the holes of the runs: what each was written
    as (``panda_config.schemaDEFT``).
    """
    return [
        hole.source
        for run in executions(func)
        if run.variable == name
        for _at, hole in run.holes
        if hole.source
    ]


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


_BARE_FIELD = "{}"


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
    """The strings local sequence *name* holds, across ``=``, ``+=``, ``append``, ``extend`` and keys.

    ``None`` rather than a partial answer wherever the sequence is built some
    other way -- an ``insert`` or a ``pop``, or an element that is not a literal.
    A missing element is not a smaller answer here: it is a statement the map
    says the code runs over a set of tables it does not.

    ``tables.append(x)`` is ``tables += [x]`` spelled as a call, and refusing it
    left ``peekJob`` -- which builds its list of job tables one ``append`` per
    flag -- reading a table named ``{}``.  The element still has to be written
    out: an ``append`` of a value read from somewhere else refuses the sequence
    exactly as a ``+=`` of one does.

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
            and node.func.attr in {"append", "extend"}
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == name
        ):
            added = (
                ast.List(elts=list(node.args)) if node.func.attr == "append" else node.args[0]
            ) if len(node.args) == 1 and not node.keywords else None
            elements = _elements(added) if added is not None else None
            if elements is None:
                return None
            found.extend(elements)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            # ``insert`` puts an element where this does not follow; ``update``
            # and ``setdefault`` add keys from somewhere this cannot see; ``pop``
            # and ``clear`` take them away again.
            and node.func.attr in {"insert", "update", "setdefault", "pop", "clear"}
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

    Only a loop that plainly walks a mapping: over ``d.items()`` or
    ``d.keys()``, or over ``d`` or a builtin's copy of it -- ``list(d)``,
    ``sorted(d)`` -- where the loop reads ``d[k]``: ``updateJobStatus``' ``for
    key in list(param): ... param[key]``.  ``for table in (...)`` is a
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
        return _keys_walked(node)
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


def _keys_walked(node: ast.For) -> tuple[str, Optional[ast.expr]]:
    """:func:`_mapping_iterated` of ``for k in d`` or ``for k in list(d)``: ``("k", d)`` where the loop reads ``d[k]``."""
    over, target = node.iter, node.target
    if (
        isinstance(over, ast.Call)
        and isinstance(over.func, ast.Name)
        and over.func.id in ("list", "tuple", "sorted", "set")
        and len(over.args) == 1
        and not over.keywords
    ):
        over = over.args[0]
    if not (isinstance(over, ast.Name) and isinstance(target, ast.Name)):
        return "", None
    if any(
        isinstance(one, ast.Subscript)
        and isinstance(one.value, ast.Name)
        and one.value.id == over.id
        and isinstance(one.slice, ast.Name)
        and one.slice.id == target.id
        for one in ast.walk(node)
    ):
        return target.id, over
    return "", None


def kept_map_keys(column: object, keys: list[str]) -> Optional[list[str]]:
    """Of *keys*, the ones a pass reaches *column* with -- the hole a loop over a mapping's keys fills with its key
    (:func:`map_key`) --, read as the loop's ``continue`` tests take them (:func:`_kept_keys`); ``None`` where they
    are not read so."""
    if type(column) is not sqleval.HoleAt:
        return None
    loop = next((one for one in ancestors(column.stmt) if isinstance(one, ast.For)), None)
    if loop is None:
        return None
    key, mapping = _mapping_iterated(loop)
    if not key or mapping is None or column.source != key:
        return None
    return _kept_keys(loop, key, keys)


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
            value = written_value(other)
            if value is not None and any(
                isinstance(target, ast.Name) and target.id == node.id
                for target in targets_of(other)
            ):
                return _string_value(value, bound, scope, hops + 1)
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


def spells_key(
    func: ast.FunctionDef | ast.AsyncFunctionDef, node: ast.AST, spelled: ast.expr, key: str
) -> bool:
    """Whether ``<varmap>[<spelled>]`` fills *key* on one pass of its loop.

    The bind side of a column named from a key.  ``setScoutJobData_JEDI`` writes
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


def map_key(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    run: Execution,
    column: object,
    bind: str,
    bind_holes: tuple,
) -> Optional[ast.expr]:
    """The mapping a term of *run* is named by the key of, where it is one: ``d`` for ``{k}=:cr_{k}`` written in
    ``for k, v in d.items()`` with ``varMap[f":cr_{k}"] = v`` in the loop.

    The term asks the row for each key the mapping holds, with the mapping's
    value at it -- read where the caller says the keys.  *column* is the hole
    the term's column is and *bind_holes* the holes of its bind
    (:attr:`Execution.reading_holes`), *bind* the bind as the statement spells
    it, ``{}`` for each hole (``:cr_{}``).  Each is the loop's key, as written
    there (:attr:`sqleval.HoleAt.source`), and the run's map gets the bind in
    the loop -- the bind spelt with the key -- valued by the loop's value or
    ``d[k]``, and still holds it at a call of the run (:func:`_live_at`).  The
    keys of a mapping the function itself is handed back are read by putting
    each in (:func:`spells_key`): no hole is left.
    """
    if type(column) is not sqleval.HoleAt or not run.varmap or not bind_holes:
        return None
    loop = next((one for one in ancestors(column.stmt) if isinstance(one, ast.For)), None)
    if loop is None:
        return None
    key, mapping = _mapping_iterated(loop)
    if not key or mapping is None or column.source != key:
        return None
    for hole in bind_holes:
        if type(hole) is not sqleval.HoleAt or hole.source != key or not any(one is loop for one in ancestors(hole.stmt)):
            return None
    target = loop.target
    value_name = (
        target.elts[1].id
        if isinstance(loop.iter, ast.Call)
        and isinstance(loop.iter.func, ast.Attribute)
        and loop.iter.func.attr == "items"
        and isinstance(target, ast.Tuple)
        and len(target.elts) == 2
        and isinstance(target.elts[1], ast.Name)
        else None
    )
    probe = "\x00key"
    spelled = bind.replace("{}", probe)
    for node in ast.walk(loop):
        for stored in targets_of(node):
            if not (
                isinstance(stored, ast.Subscript)
                and isinstance(stored.value, ast.Name)
                and stored.value.id == run.varmap
                and _string_value(stored.slice, {key: probe}, loop) == spelled
            ):
                continue
            value = written_value(node)
            by_value = value_name is not None and isinstance(value, ast.Name) and value.id == value_name
            by_key = (
                isinstance(value, ast.Subscript)
                and ast.dump(value.value) == ast.dump(mapping)
                and isinstance(value.slice, ast.Name)
                and value.slice.id == key
            )
            if (by_value or by_key) and any(
                id(node) in _live_at(func, run.varmap, [node], call) for call in run.run_at
            ):
                return mapping
    return None


class KeyColumnCensus(NamedTuple):
    """How many statements name a column from a mapping key, and how many resolve."""

    loops: int
    resolved: int
    from_parameter: int


def key_column_census(func: ast.FunctionDef | ast.AsyncFunctionDef) -> KeyColumnCensus:
    """Count the fragments written from a mapping's keys, and how many the run read.

    An unresolved one leaves the statement naming no column at all, which looks
    exactly like a statement that writes none.  The count is reported so that
    the two are not read as the same answer, and so that the five the corpus
    leaves to a caller are a number rather than a silence.
    """
    loops = resolved = from_parameter = 0
    arguments = {argument.arg for argument in func.args.args}
    counted: set[int] = set()
    # The fragments are where each stretch of what the calls run was written,
    # and the ones a hole stands for; read when no hole is left.
    fragments, unread = _run_fragments(func)
    asked = [(node, id(node) not in unread) for node in fragments]
    for node, read in asked:
        fed = _fed_by_key(node)
        if fed is None or id(node) in counted:
            continue
        counted.add(id(node))
        loops += 1
        if read:
            resolved += 1
        elif isinstance(fed, ast.Name) and fed.id in arguments:
            from_parameter += 1
    return KeyColumnCensus(loops, resolved, from_parameter)


def _fed_by_key(node: ast.stmt) -> Optional[ast.expr]:
    """The mapping whose keys the text *node* writes is made of, inside ``for k, v in d.items()``; ``None`` where
    it is not."""
    loop = next((one for one in ancestors(node) if isinstance(one, ast.For)), None)
    if loop is None:
        return None
    key, mapping = _mapping_iterated(loop)
    if not key or mapping is None:
        return None
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
        return None
    return mapping


def _run_fragments(func: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[list[ast.stmt], set[int]]:
    """The statements that wrote what *func*'s calls run, and those of them a hole stands for.

    A part a loop of unknown length repeats is a hole at the loop: what the
    loop appends to the name is the fragment, and unread.
    """
    fragments: dict[int, ast.stmt] = {}
    unread: set[int] = set()
    for run in executions(func):
        for _start, _end, way in run.pieces:
            if way and isinstance(way[0], ast.stmt):
                fragments.setdefault(id(way[0]), way[0])
        for _at, hole in run.holes:
            if isinstance(hole.stmt, (ast.For, ast.AsyncFor, ast.While)):
                inside = [
                    node
                    for node in ast.walk(hole.stmt)
                    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign))
                    and any(
                        isinstance(t, ast.Name) and t.id == hole.source
                        for t in (
                            [node.target] if isinstance(node, ast.AugAssign) else targets_of(node)
                        )
                    )
                ]
            else:
                inside = [hole.stmt] if isinstance(hole.stmt, ast.stmt) else []
            for node in inside:
                fragments.setdefault(id(node), node)
                unread.add(id(node))
    return list(fragments.values()), unread


#: How the call fills in the statement it runs, as :func:`_call_site_fill` reads it.
_BY_PERCENT = "%"
_BY_FORMAT = "format"


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
    """Return one :class:`Execution` per statement each runner call in *func* runs (:func:`runners`).

    A call by a runner's name -- the cursor's ``execute``, or a function that
    runs a statement it is handed, ``getClobObj(sql, varMap)`` -- runs the
    statement its first argument holds, with the binds its second does; one
    with no second runs it with none.

    The call is what pairs a statement with its binds.  Reading them separately
    -- every statement in the function against every bind in the function --
    would attach a task's status write to a dataset's statement whenever both
    appear in one method, which in ``db_proxy_mods`` is most of them.

    What a call runs is read by running the function (:mod:`sqleval`): each
    covering statement of what reaches the call, every path at once, with the
    text of the name it runs.  A statement is not run by every call in the
    function -- ``copyArchive.main`` assigns ``sql`` forty-four times and runs
    it twenty-three -- and each call is given only what reaches it.  A
    statement written into the call is read as any other.

    **A call it reads nothing out of is said, where *skipped* is given.**  A
    call passed over in silence leaves its statements in no node, no stop and
    no count: ``queryTasksToBePending_JEDI`` runs the ``SELECT`` its caller
    hands it, and the one function that moves ``ready`` tasks to ``pending``
    looked like one whose write picks its own rows.  Each is a :class:`Skip`
    with the reason, so a consumer can tell "read in full" from "never read"
    -- see :data:`SKIP_REASONS`; a runner handing on its own statement says so
    too (:data:`SKIP_CALLERS`): that ``SELECT`` is read where the callers run it.

    Kept per reading context (:func:`reading_context`), as the run is.
    """
    attach_parents(func)
    key = reading_context()
    held = getattr(func, "_sqleval_runs", None)
    if held is None:
        held = func._sqleval_runs = {}  # type: ignore[union-attr]
    if key not in held:
        held[key] = _executions_read(func)
    found, skips = held[key]
    if skipped is not None:
        skipped.extend(skips)
    return list(found)


class Skip(NamedTuple):
    """A runner call :func:`executions` saw and read no statement out of: its text is not here, so *which rows it
    reads* is unknown rather than "none"."""

    line: int
    reason: str
    detail: str


#: The ways an execution is seen and not read.  Spelled for a reader of the
#: build's stop list, beside the reasons a statement that *was* read stopped.
#: A statement handed in is said only by a reading with no corpus: in one,
#: the function handing it on is a runner (:func:`runners`), and the statement
#: is read at its callers (:data:`SKIP_CALLERS`).
SKIP_HANDED_IN = "the statement is handed in by the caller, so its text is not in this function"
SKIP_UNBOUND = "the statement is a name this function does not bind"
SKIP_UNREAD_NAME = "the statement is a name bound here to text this build cannot render"
SKIP_UNRENDERED = "the statement written into the call does not render"
SKIP_UNREACHED = "no statement this function builds reaches this call"
SKIP_REASONS = (
    SKIP_HANDED_IN,
    SKIP_UNBOUND,
    SKIP_UNREAD_NAME,
    SKIP_UNRENDERED,
    SKIP_UNREACHED,
)
#: A runner handing on the statement its function was handed: not read here, and no gap -- the text is the
#: callers', read where they run it.  Said all the same, so a reader of this function knows a statement runs
#: here whose rows it does not see: ``queryTasksToBePending_JEDI`` updates the tasks the ``SELECT`` it is handed
#: picks.
SKIP_CALLERS = "the statement is its callers', read where they run it"


def _unparsed(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:  # noqa: BLE001 -- unparse fails on synthesised nodes
        return "..."


def _clipped(text: str, width: int = 72) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: width - 1] + "\u2026"


def _skip_without_text(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    call: ast.Call,
    head: ast.expr,
) -> Optional[Skip]:
    """Why a call whose statement has no text here was not read.

    A runner (:func:`runners`) handing on its own statement --
    ``WrappedCursor.execute(sql, ...)``, ``getClobObj(sql, varMap)`` -- runs
    what its callers wrote, and they are where it is read (:data:`SKIP_CALLERS`).
    """
    if isinstance(head, ast.Name):
        if _is_parameter(func, head.id):
            return Skip(call.lineno, SKIP_CALLERS if func.name in runners() else SKIP_HANDED_IN, head.id)
        # ``sql = sql1_0 % (JobSpec.columnNames(), table) + sql1_1`` binds the
        # name to something no rendering here follows -- a different fact from
        # a name that came from nowhere in this function, and a different
        # place to go and read.
        bound = any(
            isinstance(node, ast.Name) and node.id == head.id and isinstance(node.ctx, ast.Store)
            for node in ast.walk(func)
        )
        return Skip(call.lineno, SKIP_UNREAD_NAME if bound else SKIP_UNBOUND, head.id)
    return Skip(call.lineno, SKIP_UNRENDERED, _clipped(_unparsed(head)))


def _is_parameter(func: ast.AST, name: str) -> bool:
    args = getattr(func, "args", None)
    if args is None:
        return False
    every = [*args.posonlyargs, *args.args, *args.kwonlyargs]
    every += [one for one in (args.vararg, args.kwarg) if one is not None]
    return any(argument.arg == name for argument in every)


class Runner(NamedTuple):
    """How a call by a runner's name runs its statement (:func:`runners`)."""

    #: Its binds are a list of maps, one per row: ``executemany``, and what hands its own on to it.
    many: bool
    #: A function of this name runs statements of its own as well as the one it is handed:
    #: ``queryTasksToBePending_JEDI`` runs the ``SELECT`` its caller hands it and updates the tasks it picks.
    own: bool


#: The cursor's own forms: the driver, which runs the statement and nothing of the corpus.
_CURSOR = {"execute": Runner(many=False, own=False), "executemany": Runner(many=True, own=False)}
#: Their names: a call by one lands in the driver, never in a function of the corpus of that name.
CURSOR_RUNNERS = frozenset(_CURSOR)

#: The runners of a reading with no corpus: the cursor's and the task buffer's (``taskBuffer.querySQLS(sql,
#: var_map)``, the daemon layer's form).  A corpus says its own (:func:`runners`).
_NO_CORPUS = {**_CURSOR, "querySQL": Runner(many=False, own=False), "querySQLS": Runner(many=False, own=False)}
RUNNERS = frozenset(_NO_CORPUS)

#: :func:`runners` of each corpus, by its roots.
_RUNNERS_OF: dict[tuple, dict[str, Runner]] = {}


def runners() -> dict[str, Runner]:
    """Every name a call runs a statement by, in the corpus read through (:func:`across_calls`).

    The cursor's ``execute`` and ``executemany``, and every function that runs,
    as its statement, one of its own parameters -- ``getClobObj(sql, varMap)``
    runs ``self.cur.execute(sql, varMap)``, ``TaskBuffer.querySQLS`` hands its
    own to the proxy's -- to a fixed point.  A call by one of these names runs
    the statement its caller writes, and is read there (:func:`executions`);
    the function handing it on is no gap (:data:`SKIP_CALLERS`).  A function
    sharing a runner's name is taken to be one, as ``execute`` always was.

    Read from the whole corpus whichever file is read, as :meth:`Calls.every_call`
    is, so a reading of one file reads the runners the build does.
    """
    if _CALLS is None:
        return _NO_CORPUS
    types = _CALLS.types
    key = tuple(sorted((package, str(root)) for package, root in types.roots.items()))
    if key not in _RUNNERS_OF:
        _RUNNERS_OF[key] = _corpus_runners(types)
    return _RUNNERS_OF[key]


def _corpus_runners(types) -> dict[str, Runner]:
    from bamboo.codemap.calls import own_nodes

    # Each function's calls by a name, and whether each runs one of the function's parameters as its statement.
    calling: list[tuple[str, list[tuple[str, bool]]]] = []
    for file in types.corpus_files():
        tree = types.tree(file)
        if tree is None:
            continue
        for func in ast.walk(tree):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            parameters = {
                argument.arg
                for argument in (*func.args.posonlyargs, *func.args.args, *func.args.kwonlyargs)
            }
            made = []
            for node in own_nodes(func):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.args:
                    named = any(isinstance(one, ast.Name) and one.id in parameters for one in ast.walk(node.args[0]))
                    made.append((node.func.attr, named and _runs_a_parameter(func, node)))
            if made:
                calling.append((func.name, made))
    found = dict(_CURSOR)
    grown = True
    while grown:
        grown = False
        for name, made in calling:
            if name in found:
                continue
            handed = [called for called, hands in made if hands and called in found]
            if handed:
                found[name] = Runner(many=any(found[called].many for called in handed), own=False)
                grown = True
    for name, made in calling:
        if name in found and name not in _CURSOR:
            ran = [hands for called, hands in made if called in found]
            if any(ran) and not all(ran):
                found[name] = found[name]._replace(own=True)
    return found


def _runs_a_parameter(func: ast.AST, call: ast.Call) -> bool:
    """Whether *call* hands on, as its statement, a parameter of *func* -- the head :func:`executions` reads."""
    statement, _fillers, _spelling = _call_site_fill(func, call.args[0])  # type: ignore[arg-type]
    head = _concatenated(statement)[0]
    return isinstance(head, ast.Name) and _is_parameter(func, head.id)


#: ``RETURNING <column> INTO <bind>``, either side a comma list: the binds the
#: database writes into the map the statement ran with.
_RETURNING_INTO = re.compile(
    r"\bRETURNING\s+[\w.$]+(?:\s*,\s*[\w.$]+)*\s+INTO\s+(:\w+(?:\s*,\s*:\w+)*)", re.IGNORECASE
)

#: What a statement starts with past blanks, comments and a loop's marks: its verb, or a hole.
_LEADING = re.compile(
    r"(?:\s|/\*.*?\*/|--[^\n]*|[" + sqltext.REPEAT_OPEN + sqltext.REPEAT_CLOSE + r"])*(\{\}|[A-Za-z]+)",
    re.DOTALL,
)
#: The statements a ``RETURNING ... INTO`` ends, and the ones that return into nothing.
_RETURNING_VERBS = frozenset({"INSERT", "UPDATE", "DELETE", "MERGE"})
_READING_VERBS = frozenset({"SELECT", "WITH"})
#: What may stand after the clause's hole: blanks, comments, other holes, a loop's marks.
_NOTHING_AFTER = re.compile(
    r"(?:\s|;|\{\}|/\*.*?\*/|--[^\n]*|[" + sqltext.REPEAT_OPEN + sqltext.REPEAT_CLOSE + r"])*", re.DOTALL
)


def _could_end_unread(text: str) -> bool:
    """Whether a hole of *text* stands where a ``RETURNING ... INTO`` could.

    The clause is the last of an ``INSERT``, ``UPDATE``, ``DELETE`` or
    ``MERGE``, so a hole inside parentheses or quotes, or with any text after
    it, cannot hold it: ``IN ({})``, ``VALUES ({})``, ``SET {} WHERE``.  Nor
    can one that goes on a bind's name: ``updateTask_JEDI`` names a bind
    ``:cr_{}`` after each key of the criteria it is handed, and the last ends
    the statement.  ``updateDataset``'s ``AND {withCriteria}`` can -- the
    caller's condition is the statement's end.  A ``SELECT`` returns into
    nothing; any other statement, a PL/SQL block or one whose verb is a hole,
    could hold the clause at any of its holes.
    """
    leading = _LEADING.match(text)
    verb = leading.group(1).upper() if leading else ""
    if verb in _READING_VERBS:
        return False
    if verb not in _RETURNING_VERBS:
        return "{}" in text
    depth, index = 0, 0
    while index < len(text):
        if text.startswith("/*", index):
            closing = text.find("*/", index + 2)
            index = len(text) if closing < 0 else closing + 2
            continue
        if text.startswith("--", index):
            closing = text.find("\n", index + 2)
            index = len(text) if closing < 0 else closing + 1
            continue
        char = text[index]
        if char == "'":
            closing = text.find("'", index + 1)
            index = len(text) if closing < 0 else closing
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif text.startswith("{}", index):
            ended = _NOTHING_AFTER.fullmatch(text, index + 2) is not None
            named = re.search(r":\w*$", text[:index].rstrip(sqltext.REPEAT_OPEN + sqltext.REPEAT_CLOSE))
            if depth == 0 and ended and named is None:
                return True
            index += 1
        index += 1
    return False


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
    one is missing a piece that has no text of its own
    (:data:`FRAGMENT_NOT_RENDERED`), or a hole stands where the clause could
    (:func:`_could_end_unread`) -- either could have held it.  A hole the
    statement keeps for a schema name, a value, an ``IN`` list or a bind's
    name cannot, and is read past.
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
    if any(_could_end_unread(run.reading) for run in runs):
        return None
    return ()


def reading_context() -> tuple:
    """What a reading of a function can be kept against.

    The calls the statements are read across (:func:`across_calls`), and
    under :func:`at_call` the function read and the constants handed to it:
    every call handing the same constants is read the same way, and the
    reader reads each one.
    """
    if _AT_CALL is None:
        return (id(_CALLS),)
    func, values = _AT_CALL
    return (
        id(_CALLS),
        id(func),
        tuple(sorted((name, repr(value)) for name, value in values.items())),
    )


def written_back_at(path: Path, line: int) -> Optional[tuple[str, ...]]:
    """:func:`written_back` for the ``execute`` or ``executemany`` call on *line* of *path*.

    For a reader holding a file and a line and no tree.  It reads the tree the
    rest of the map reads (:func:`~bamboo.codemap.calls.parsed`), so the call is
    the node the statement was paired with, and through the calls the build
    reads through (:func:`across_calls`) unless the reader set its own:
    ``sql += JobSpec.bindValuesExpression()`` is a ``VALUES`` list only in the
    callee, and without them 23 ``INSERT``s the build reads as writing nothing
    back answered ``None``.  ``None`` as well where the file does not parse or
    no such call starts on the line.
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
            if func is None:
                return None
            with across_calls(_CALLS or _corpus_calls(path, held[1])):
                return written_back(func, node)
    return None


#: The calls of each corpus a reader holding a file asks about (:func:`_corpus_calls`), by its directory.
_CORPUS_CALLS: dict[str, "Calls"] = {}


def _corpus_calls(path: Path, tree: ast.Module) -> Optional["Calls"]:
    """The calls of the corpus *path* is in, over every module's tree as the build holds them.

    The corpus is the directory holding the package *path* is in, spelt as
    *path* spells it, so that *tree*, the one read for the call, is the one
    handed in.  ``None`` outside a package the map reads.  The trees are
    handed in as the build hands them
    (:func:`~bamboo.codemap.panda.recognizers.sqlwrite.corpus_calls`): a
    function in a module the reader was not handed has no file, and none of
    its calls is placed -- so where *tree* is not the one held for its file,
    the file changed since, and the corpus is read again.  A test module is
    not held, as the build holds none.
    """
    from bamboo.codemap.calls import Calls, Types, parsed
    from bamboo.codemap.gitsource import blob_sha
    from bamboo.codemap.panda.plugin import PACKAGES

    package = next((one for one in path.parents if one.name in PACKAGES), None)
    if package is None:
        return None
    corpus = package.parent
    held = _CORPUS_CALLS.get(str(corpus))
    walked_tree = held.types.walked.get(str(path.relative_to(corpus)), (None, tree))[1] if held else None
    if held is None or walked_tree is not tree:
        roots = {name: corpus / name for name in PACKAGES if (corpus / name).is_dir()}
        walked = {}
        for file in Types(roots, {}).corpus_files():
            read = parsed(corpus / file)
            if read is not None:
                walked[file] = (blob_sha(read[0]), read[1])
        held = _CORPUS_CALLS[str(corpus)] = Calls(Types(roots, walked))
    return held


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


#: Where a statement's rows arrive.  The ``execute`` form binds nothing at the
#: call -- ``self.cur.execute(sql, varMap)`` and then ``res =
#: self.cur.fetchall()`` -- so reading the call line alone finds the rows for
#: 48 of the corpus's 264 statements and misses every exit taken because none
#: came back.  A fetch reads the rows of the run whose region holds it (:func:`regions`).
FETCHES = ("fetchall", "fetchone", "fetchmany")


def regions(
    func: ast.FunctionDef | ast.AsyncFunctionDef, executions: list[Execution]
) -> dict[int, tuple[int, int]]:
    """The stretch of *func* each run owns, by its call's line: ``(first, last)``.

    From where the run's binds start being filled to just before the next
    run's do, so a ``fetch*`` in it reads this run's rows.  Every run of the
    function is laid out: the statement that ends a region may be one no
    reader keeps, and stopping at the next *kept* one would hand a reader the
    following query's rows as well.

    Where two statements share one ``var_map = {}`` the later one's binds
    start filling *above* the earlier one's call, so the next run's window
    is no edge at all: it would end a region before the statement it
    belongs to ran, taking that statement's own ``fetchall`` with it.  A
    statement's territory cannot close at or before it ran, so where the
    windows overlap the next *call* is the edge.

    Keyed by the call line because a branchy statement is several runs at one
    ``execute``: without the key the region ends at the statement's own second
    variant, which is itself.
    """
    starts: dict[int, int] = {}
    for run in executions:
        line = run.call.lineno
        window = run.window
        if window is None and run.varmap:
            window = binding_window(func, run.varmap, line)
        begins = window[0] if window else line
        starts[line] = min(starts.get(line, begins), begins)
    last = func.end_lineno or func.lineno
    ordered = sorted(starts.items())
    found: dict[int, tuple[int, int]] = {}
    for index, (line, begins) in enumerate(ordered):
        if index + 1 < len(ordered):
            next_line, next_begins = ordered[index + 1]
            following = next_begins if next_begins > line else next_line
        else:
            following = last + 1
        found[line] = (begins, max(line, following - 1))
    return found


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
    at: Optional[ast.AST | Execution] = None,
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

    Read here rather than off the statement's text because the text
    deliberately blanks interpolations: they are usually the schema, and
    keeping them would make every statement unique.  This looks at the one interpolation that
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
    at: Optional[ast.AST | Execution] = None,
) -> list[Bind]:
    """Return the assignments filling ``<varmap>[<key>]`` in *func*.

    Several are normal and correct: a method that runs the same statement twice
    with different values has two, and both are outcomes of that write.

    *at* is the execution the binds are read for -- the run itself, or a bare
    call -- and only the binds the map it runs with can still hold are
    returned; see :func:`_live_at`.  Every reader of a run asks this way, in
    both call forms.  Given the run, they are also its own branch's: one text
    of the statement is built on some passes and another on others, and a bind
    only a pass building another text fills is not this one's
    (:func:`_not_its_text`), nor one every pass building this text writes
    again before the call (:func:`_not_on_its_passes`).  Reading every bind of
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
    at: Optional[ast.AST | Execution] = None,
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
        run = at if isinstance(at, Execution) else None
        for one in run.run_at if run is not None else (at,):
            live |= _live_at(func, varmap, writes, one)
        if run is not None:
            live -= _not_its_text(func, varmap, run, [gate for gate in writes if id(gate) in live], writes)
            live -= _not_on_its_passes(func, varmap, run, [gate for gate in writes if id(gate) in live], writes)
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
    for loop, appends in _listed_maps(func, varmap, renewals, at):
        if id(loop) in live:
            for append in appends:
                live |= surviving(append)
    return live


def bind_replacements(
    func: ast.FunctionDef | ast.AsyncFunctionDef, run: Execution, key: str, binds: Sequence[Bind]
) -> dict[int, list[Replacement]]:
    """What the run's other binds of *key* do to each of *binds* *func* writes, by its site's ``id``.

    The route a statement map's key takes to its write (:func:`pathcond.replacements`):
    the binds of the key the run can still hold (:func:`scan_binds` at the
    run) are asked about; every bind of the key in *func* is a write of the
    place -- the ones only another text of the statement is built with, too
    -- the calls running the statement are its reads, the map's renewals what
    gives it a new map, and a bind filling the key from what it held a
    reread.  ``updateTaskStatusByContFeeder_JEDI`` fills ``:errorDialog`` with
    None and fills it again from the task where it is given one, which builds
    another text.  Which binds the run holds is :func:`scan_binds`' to say, so
    a bind it keeps is kept here too, with no entry where the reading finds it
    reaching none.
    """
    if run.varmap is None:
        return {}
    asked = list({id(bind.site): bind.site for bind in binds if enclosing_function(bind.site) is func}.values())
    every = [bind for bind in scan_binds(func, run.varmap, key).binds if enclosing_function(bind.site) is func]
    sites = list({id(one): one for one in [*asked, *(bind.site for bind in every)]}.values())
    held = {id(site) for site in sites}
    rereads = [site for site in sites if _rereads(site, run.varmap)]
    killers = [one for one in _renews(func, run.varmap) if id(one) not in held]
    reads = list({id(one): one for one in (site_of(call) for call in run.run_at)}.values())
    place = f"{run.varmap}[{key!r}]"
    values = {id(bind.site): bind.value for bind in [*every, *binds]}
    found: dict[int, list[Replacement]] = {}
    for site in asked:
        replaced = replacements(
            site,
            [one for one in sites if one is not site],
            reads,
            place=place,
            value=values.get(id(site)),
            killers=killers,
            rereads=rereads,
        )
        found[id(site)] = replaced or []
    return found


def _listed_maps(
    func: ast.FunctionDef | ast.AsyncFunctionDef, varmap: str, renewals: list[ast.stmt], at: ast.AST
) -> list[tuple[ast.stmt, list[ast.stmt]]]:
    """Each ``for <varmap> in <list>:`` among *renewals* holding *at*, with the statements appending a map to the list."""
    return [
        (loop, _appended(func, loop.iter.id, varmap))
        for loop in renewals
        if isinstance(loop, (ast.For, ast.AsyncFor))
        and isinstance(loop.iter, ast.Name)
        and any(ancestor is loop for ancestor in ancestors(at))
    ]


def _not_its_text(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    varmap: str,
    run: Execution,
    live: list[ast.AST],
    writes: list[ast.AST],
) -> set[int]:
    """The ``id`` of each of *live* that no pass building *run*'s text fills.

    :func:`_live_at` reads the binds the map can hold at the call, whatever
    the text; one call can run several texts of one statement, and each is
    built on some passes and not others.  ``updateTaskStatusByContFeeder_JEDI``
    adds ``,oldStatus=:oldStatus,splitRule=:splitRule`` to its ``UPDATE`` only
    when it is given a task, fills ``:status`` from the task in that case and
    ``'assigning'`` or ``'ready'`` otherwise, and runs whichever it built at
    one call: read by the call alone, the text with the clause wrote
    ``assigning``.  ``insertNewJob`` fills a file's ``:status`` and appends
    its ``is_waiting`` clause under two tests of one flag.

    A bind is not the text's where it cannot be on one pass with a statement
    that put a piece of the text there (:func:`pathcond.parted`) -- unless a
    loop carries it to the call from an earlier pass that the conditions let
    differ (:func:`pathcond.carried_round`) -- or where it runs only through
    the arm of a piece another text at the call adds to all of this one's
    (:func:`pathcond.only_through`): the pass that built this text skipped
    that arm.  A text written twice the same -- ``archiveJob`` writes
    ``sqlFMod`` before its loop and again in one arm -- is built two ways and
    no statement is on both (:func:`_text_reading`), so it parts nothing.
    """
    calls, mine, added = _text_reading(func, run)
    if not mine:
        return set()
    held = _pairs_read(func)
    renewals = _renews(func, varmap)
    sites = list({id(site): site for site in [*writes, *renewals]}.values())
    rereads = [write for write in writes if _rereads(write, varmap)]
    found: set[int] = set()
    for write in live:
        if enclosing_function(write) is not func:
            continue
        if any(
            piece is not write
            and held(parted, piece, write)
            and not any(carried_round(write, call, piece, sites, rereads) for call in calls)
            for piece in mine
        ) or any(held(only_through, write, piece) for piece in added):
            found.add(id(write))
    return found


def _not_on_its_passes(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    varmap: str,
    run: Execution,
    live: list[ast.AST],
    writes: list[ast.AST],
) -> set[int]:
    """The ``id`` of each of *live* that no pass building *run*'s text carries to a call of it.

    :func:`_not_its_text` reads where a bind is written; a pass writing it can
    build the text and still write the key again before the call.
    ``updateTaskStatusByContFeeder_JEDI`` sets ``:updateTime`` to now before
    its ``if``, and the two arms building the text without the ``oldStatus``
    clause each set it six hours back: the one pass keeping now builds the
    text with the clause.  So the statements no pass building the text runs
    (:func:`_not_building`) are cut from the way, as writes that are no
    answer, and a bind is the text's where it still reaches a call.

    A loop holding the call reads each pass afresh: an earlier pass can be one
    not building the text and a later one one that does, so what the map
    carries into a pass is read apart -- kept where it reaches the top of the
    innermost loop holding the call and a value there reaches the call, cut
    so.  A ``for`` writes its target at the top of each pass, where
    :func:`~bamboo.codemap.panda.reaching.reaching` puts it, so the loop
    stands for that value; a ``while`` has no such write, and what it carries
    is kept.  A map taken from a list is the one built where it was appended
    (:func:`_live_at`): its binds are kept where that ``for`` reaches the call.
    """
    killers = [one for one in _not_building(func, run) if all(one is not write for write in writes)]
    if not killers:
        return set()
    renewals = _renews(func, varmap)
    sites = list({id(site): site for site in [*writes, *renewals]}.values())
    rereads = [write for write in writes if _rereads(write, varmap)]

    def reaching_ids(targets: list[ast.AST], at: ast.AST, cut: list[ast.AST]) -> set[int]:
        return {id(site) for site in surviving_writes(targets, at, rereads, as_run=True, killers=cut)}

    kept: set[int] = set()
    for call in (site_of(one) for one in run.run_at):
        loop = _innermost_loop(func, call)
        marked = sites
        if isinstance(loop, (ast.For, ast.AsyncFor)) and all(loop is not site for site in sites):
            marked = [*sites, loop]
        reached = reaching_ids(marked, call, killers)
        kept |= reached
        if isinstance(loop, ast.While) or (loop is not None and id(loop) in reached):
            kept |= reaching_ids(sites, loop.body[0], [])
        for listed, appends in _listed_maps(func, varmap, renewals, call):
            if id(listed) in reached:
                for append in appends:
                    kept |= reaching_ids(sites, append, [])
    return {id(write) for write in live if enclosing_function(write) is func and id(write) not in kept}


def _not_building(func: ast.FunctionDef | ast.AsyncFunctionDef, run: Execution) -> list[ast.stmt]:
    """The simple statements of *func* no pass building *run*'s text runs -- read once per run.

    :func:`_not_its_text`'s two readings, of a statement on the way rather
    than of a bind: parted from a statement every way of building the text
    runs, or run only through the arm of a piece another text at its calls
    adds to all of this one's.
    """
    store = _held_on(func, "_sql_not_building")
    key = (reading_context(), run.sql, tuple(id(call) for call in run.run_at), _piece_ids(run))
    if key not in store:
        _calls, mine, added = _text_reading(func, run)
        held = _pairs_read(func)
        store[key] = [
            node
            for node in (_simple_statements(func) if mine else ())
            if any(piece is not node and held(parted, piece, node) for piece in mine)
            or any(held(only_through, node, piece) for piece in added)
        ]
    return store[key]


def _simple_statements(func: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.stmt]:
    """The statements of *func* itself holding no body: each runs whole, where it stands."""
    store = _held_on(func, "_sql_simple_statements")
    if not store:
        store[0] = [
            node
            for node in ast.walk(func)
            if isinstance(node, ast.stmt)
            and "body" not in node._fields
            and "cases" not in node._fields
            and enclosing_function(node) is func
        ]
    return store[0]


def _innermost_loop(func: ast.AST, node: ast.AST) -> Optional[ast.For | ast.AsyncFor | ast.While]:
    """The innermost loop of *func* whose body holds *node*; None in none."""
    previous = node
    for ancestor in ancestors(node):
        if ancestor is func:
            return None
        if isinstance(ancestor, (ast.For, ast.AsyncFor, ast.While)) and any(previous is one for one in ancestor.body):
            return ancestor
        previous = ancestor
    return None


def _text_reading(
    func: ast.FunctionDef | ast.AsyncFunctionDef, run: Execution
) -> tuple[list[ast.AST], list[ast.AST], list[ast.AST]]:
    """*run*'s calls as statements, the statements every way of building its text runs, and the pieces other texts at
    those calls add to all of its own -- read once per run, as every key of the run asks.

    A text is built by every way that renders it (:func:`_text_ways`), and a
    statement only some of them run decides nothing: the same text written in
    each arm of an ``if`` and run once after it is one run, its pieces naming
    one arm.  Where the ways are not all known, no statement is taken.
    """
    store = _held_on(func, "_sql_text_reading")
    key = (reading_context(), run.sql, tuple(id(call) for call in run.run_at), _piece_ids(run))
    if key not in store:
        calls = [site_of(call) for call in run.run_at]
        ways = _text_ways(func, run, calls)
        mine = _every_way(ways)
        anywhere = {one for way in ways or () for one in way}
        own = {id(one) for one in mine}
        added: list[ast.AST] = []
        for other in executions(func) if mine else ():
            if not any(call is one for call in other.run_at for one in run.run_at):
                continue
            theirs = _every_way(_text_ways(func, other, calls))
            if not own <= {id(one) for one in theirs}:
                continue
            added += [
                one for one in theirs if id(one) not in anywhere and not any(only_through(held, one) for held in mine)
            ]
        store[key] = (calls, mine, added)
    return store[key]


#: How many ways of building one text are read; past them, the ways are not all known.
TEXT_WAYS_CAP = 16


def _text_ways(
    func: ast.FunctionDef | ast.AsyncFunctionDef, run: Execution, calls: list[ast.AST]
) -> Optional[list[dict[int, ast.AST]]]:
    """Each way *run*'s text is built at its calls, as the statements of *func* putting its pieces; None where the ways
    are not all known -- more than :data:`TEXT_WAYS_CAP`, or the text not rendered again from what reaches a call."""
    store = _held_on(func, "_sql_text_ways")
    key = (reading_context(), run.sql, tuple(id(call) for call in run.run_at), _piece_ids(run))
    if key not in store:
        reading = _evaluation(func)
        drawn = [run.pieces]
        for call in run.run_at:
            found = sqleval.rendered_as(list(reading.values.get(call.lineno, {})), run.sql, limit=TEXT_WAYS_CAP + 1)
            if not found or len(found) > TEXT_WAYS_CAP:
                drawn = []
                break
            drawn += [one.pieces for one in found]
        store[key] = [_text_statements(func, pieces, calls) for pieces in drawn] or None
    return store[key]


def _every_way(ways: Optional[list[dict[int, ast.AST]]]) -> list[ast.AST]:
    """The statements every one of *ways* runs."""
    if not ways:
        return []
    return [node for key, node in ways[0].items() if all(key in way for way in ways[1:])]


def _piece_ids(run: Execution) -> tuple[int, ...]:
    return tuple(id(one) for _start, _end, way in run.pieces for one in way if isinstance(one, ast.AST))


def _held_on(func: ast.AST, name: str) -> dict:
    store = getattr(func, name, None)
    if store is None:
        store = {}
        setattr(func, name, store)
    return store


def _pairs_read(func: ast.FunctionDef | ast.AsyncFunctionDef):
    """A reader of a test on two statements of *func* (``parted``, ``only_through``), each pair read once."""
    store = _held_on(func, "_sql_pairs_read")

    def held(test, one: ast.AST, other: ast.AST) -> bool:
        key = (test.__name__, id(one), id(other))
        if key not in store:
            store[key] = test(one, other)
        return store[key]

    return held


def _text_statements(
    func: ast.FunctionDef | ast.AsyncFunctionDef, pieces: tuple, calls: list[ast.AST]
) -> dict[int, ast.AST]:
    """The statements of *func* that put one of *pieces*, by ``id``, where one can be on a pass through *calls*.

    One run holds every call that runs its text, and two calls can run one
    text from two assignments in two arms -- ``activateJob`` writes ``sqlJob``
    for its call in each arm -- while the pieces name one of them.
    """
    found: dict[int, ast.AST] = {}
    for _start, _end, way in pieces:
        for one in way:
            if not isinstance(one, ast.AST):
                continue
            node = one if isinstance(one, ast.stmt) else site_of(one)
            if enclosing_function(node) is not func or all(parted(node, call) for call in calls):
                continue
            found.setdefault(id(node), node)
    return found


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


# ---------------------------------------------------------------- read by running the function


def _evaluation(func: ast.FunctionDef | ast.AsyncFunctionDef) -> "sqleval.Reading":
    """*func* run on every path (:mod:`sqleval`), kept on it per reading context.

    The context is what the run is given: the calls it follows into
    (:func:`across_calls`, the map's own placing of each call) and, read as one
    call site runs it (:func:`at_call`), that call's constants.  A build asks
    about a function many times in one context -- 43,706 readings of 3,446
    (function, constants) pairs -- and running it is the whole cost.
    """
    constants = dict(_AT_CALL[1]) if _AT_CALL is not None and _AT_CALL[0] is func else {}
    key = (id(_CALLS), tuple(sorted((name, repr(value)) for name, value in constants.items())))
    held = getattr(func, "_sqleval_held", None)
    if held is None:
        held = func._sqleval_held = {}  # type: ignore[union-attr]
    if key not in held:
        # Values are interned per run; one kept from an earlier run stays
        # valid, but is never the same object as an equal one made later.
        sqleval.clear_memo()
        held[key] = sqleval.evaluate(func, constants, _placed_callees(), _conventions(func))
    return held[key]


def _conventions(func: ast.FunctionDef | ast.AsyncFunctionDef) -> dict:
    """What a run of *func* is told besides the code: the in-list helper, the lists it builds, the tracing tag."""
    lists = in_lists(func)
    return {
        "in_list_helper": IN_LIST_HELPER,
        "globals": {name: sqleval.lit(one.bind, one.site) for name, one in lists.items()},
        # ``execute(sqlU + comment)``: the tag the call appends is not the statement (:func:`_call_site_fill`).
        "tracing_tag": True,
    }


def _text_branches(value) -> list:
    """*value*'s texts: itself, or each branch of a choice that is one -- a name bound on some paths only is a
    choice between its text and nothing where the paths meet.  One level: a branch that is itself such a choice
    is left out rather than taken apart."""
    if sqleval.is_text(value):
        return [value]
    if type(value) is sqleval.Choice:
        return [branch for branch in value.branches if sqleval.is_text(branch)]
    return []


def names_carried(runs: list["Execution"]) -> set[str]:
    """The local names the text of *runs* went through on its way to the run (:attr:`Execution.pieces`): every name a
    statement on a stretch's way wrote it into or carried it on in."""
    found: set[str] = set()
    for run in runs:
        for _start, _end, way in run.pieces:
            for at in way:
                if isinstance(at, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                    targets = at.targets if isinstance(at, ast.Assign) else [at.target]
                    found |= {target.id for target in targets if isinstance(target, ast.Name)}
    return found


def assembled_texts(func: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, list[str]]:
    """Every statement *func* assembles, by the local holding it, read apart from the calls that run statements.

    What each name the function writes statement text into holds where the
    code uses it -- runs it, hands it to a call, formats it into another name,
    returns it -- or where the function ends (:func:`sqleval.assembled`).  The
    sql-read slice's denominator: a statement no accepted call runs is in it
    and not among :func:`executions`, which is what it is for.
    """
    constants = dict(_AT_CALL[1]) if _AT_CALL is not None and _AT_CALL[0] is func else {}
    key = (id(_CALLS), tuple(sorted((name, repr(value)) for name, value in constants.items())))
    held = getattr(func, "_sqleval_assembled", None)
    if held is None:
        held = func._sqleval_assembled = {}  # type: ignore[union-attr]
    if key not in held:
        sqleval.clear_memo()
        run = sqleval.assembled(func, constants, _placed_callees(), _conventions(func))
        found: dict[str, list[str]] = {}
        for name, values in run.used.items():
            texts: list[str] = []
            for one in sqleval.cover([text for value in values for text in _text_branches(value)])[0]:
                text = _fragments_out(one, func).text
                if not _says_nothing(text) and text not in texts:
                    texts.append(text)
            if texts:
                found[name] = texts
        held[key] = found
    return held[key]


def _placed_callees():
    """The calls a run follows: the map's own placing of each (:func:`across_calls`), every target, nothing open,
    nothing whose value is not what the targets return (:attr:`calls.Placed.returns_none`)."""
    calls = _CALLS
    if calls is None:
        return None

    def resolve(func, call):
        file = calls.types.file_of(func)
        if file is None:
            return None
        placed = calls.callees(file, func, call)
        if placed.unplaced or not placed.targets or placed.returns_none:
            return None
        return [target.func for target in placed.targets]

    return resolve


def _says_nothing(text: str) -> bool:
    """A statement that is holes and nothing else: what a parameter handed in renders to."""
    return not _HOLE.sub("", text).strip()


def _statement_of(node: ast.AST) -> ast.AST:
    while not isinstance(node, ast.stmt) and getattr(node, "parent", None) is not None:
        node = node.parent  # type: ignore[attr-defined]
    return node


def _spans_of(drawn: "sqleval.Rendered", anchor: ast.AST) -> list[Span]:
    """Each stretch of *drawn* with the statement that put it into its name.

    A piece no assignment put there -- a parameter run as handed in, or a
    route's constant -- is anchored at *anchor*, as a statement written into
    its call is.
    """
    return [
        Span(where if isinstance(where, ast.AST) else anchor, start, end)
        for where, start, end in drawn.carried
    ]


def _evaluated_spans(
    func: ast.FunctionDef | ast.AsyncFunctionDef, name: str, at: ast.Call, text: Optional[str], read: bool = False
) -> list[tuple[str, list[Span]]]:
    """:func:`variant_spans`, run: what *name* holds at *at*."""
    reading = _evaluation(func)
    found = [_fragments_out(one, func) for one in reading.named(at.lineno, name, text)]
    anchor = _statement_of(at)
    return [
        (reading_text(one.text, one.holes) if read else one.text, _spans_of(one, anchor)) for one in found if one.text
    ]


#: How many statements of one call :func:`every_statement` draws, at most.
#: ``lockJobsForReassign`` runs 384, ``propagateResultToJEDI``'s file update
#: 324; past them, ``getJobs`` runs 24,576 and ``getTasksWithCriteria_JEDI``
#: 9,984.
EVERY_CAP = 512


def every_statement(func: ast.FunctionDef | ast.AsyncFunctionDef, line: int) -> Optional[list[str]]:
    """Every statement the call at *line* runs, each as a reader takes it
    (:attr:`Execution.reading`); ``None`` past :data:`EVERY_CAP` of them.

    :func:`executions` hands a covering set -- every branch shows up, not
    every combination of branches -- which is all a reader of what some run
    asks needs.  A reader of what *every* run asks needs them all:
    ``getJobs`` appends a dozen terms each under its own ``if``, and no
    statement of its covering set appends none of them, so read as the ways a
    run can go, the set required one of its combinations.
    """
    drawn = every_rendering(func, line)
    return None if drawn is None else [reading_text(one.text, one.holes) for one in drawn]


def every_rendering(func: ast.FunctionDef | ast.AsyncFunctionDef, line: int) -> Optional[list["sqleval.Rendered"]]:
    """:func:`every_statement`, each with the pieces it was built from as :func:`executions` gives them.

    A reader asking whether some statement the call runs holds a guard, and which tests its pieces passed, needs
    them all: a covering set showed ``getEventRanges`` narrowing its jumbo subquery for one limit and not for the
    other.
    """
    reading = _evaluation(func)
    held = getattr(reading, "_every", None)
    if held is None:
        held = reading._every = {}  # type: ignore[attr-defined]
    if line not in held:
        held[line] = _every_statement(func, reading, line)
    return held[line]


def _every_statement(
    func: ast.FunctionDef | ast.AsyncFunctionDef, reading: "sqleval.Reading", line: int
) -> Optional[list["sqleval.Rendered"]]:
    if line in reading.full_capped:
        return None
    full = reading.texts.get(line) or {}
    if len(full) > EVERY_CAP:
        return None
    drawn = reading.cover(line)[0]
    if not set(full) <= {one.text for one in drawn}:
        values = list(reading.values.get(line, {}))
        drawn = []
        for text in full:
            found = sqleval.rendered_as(values, text, limit=1)
            if not found:
                return None
            drawn.append(found[0])
    texts: list[str] = []
    kept: list["sqleval.Rendered"] = []
    for one in drawn:
        out = _fragments_out(one, func)
        if not _says_nothing(out.text) and out.text not in texts:
            texts.append(out.text)
            kept.append(out)
    return kept


def _unrendered(hole: "sqleval.HoleAt", func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """A hole that is a whole ``+=`` fragment with no text of its own (:data:`FRAGMENT_NOT_RENDERED`), appended
    by *func* itself.

    One a callee appends -- ``ret += attr`` in ``JobSpec.columnNames()`` -- is a piece of the value the call
    returns, not a fragment of the statement, and stays ``{}``.
    """
    if not isinstance(hole.stmt, ast.AugAssign) or hole.source != ast.unparse(hole.stmt.value):
        return False
    own = getattr(func, "_own_augassigns", None)
    if own is None:
        own = func._own_augassigns = frozenset(  # type: ignore[union-attr]
            id(node) for node in ast.walk(func) if isinstance(node, ast.AugAssign)
        )
    return id(hole.stmt) in own


def _fragments_out(
    drawn: "sqleval.Rendered", func: ast.FunctionDef | ast.AsyncFunctionDef
) -> "sqleval.Rendered":
    """*drawn* without the ``+=`` fragments *func* appends that have no text of its own.

    ``sqlP += sql_where_clause`` with nothing known of the name: the
    reading leaves the fragment out of the statement and says so, once
    (:func:`_fragment_stops`).  Kept in as ``{}`` it would be said a second
    time, as a ``WHERE`` that takes whole conditions through an interpolation,
    and a head and the statement it starts would disagree on every position
    after it.
    """
    cut = sorted(at for at, hole in drawn.holes if _unrendered(hole, func))
    if not cut:
        return drawn
    width = len("{}")

    def moved(position: int) -> int:
        return position - width * sum(1 for at in cut if at < position)

    text = drawn.text
    for at in reversed(cut):
        text = text[:at] + text[at + width :]

    def spans(found):
        out = []
        for where, start, end in found:
            start, end = moved(start), moved(end)
            if end > start:
                out.append((where, start, end))
        return tuple(out)

    return drawn._replace(
        text=text,
        spans=spans(drawn.spans),
        holes=tuple((moved(at), hole) for at, hole in drawn.holes if at not in cut),
        carried=spans(drawn.carried),
        pieces=tuple(
            (moved(start), moved(end), way)
            for start, end, way in drawn.pieces
            if moved(end) > moved(start)
        ),
    )


def _fragment_stops(
    drawn: "sqleval.Rendered", func: ast.FunctionDef | ast.AsyncFunctionDef
) -> tuple[Stop, ...]:
    """A ``+=`` *func* appends whose right-hand side has no text of its own: :data:`FRAGMENT_NOT_RENDERED`."""
    found = {
        Stop(hole.stmt.lineno, hole.source, FRAGMENT_NOT_RENDERED)
        for _at, hole in drawn.holes
        if _unrendered(hole, func)
    }
    return tuple(sorted(found))


def _column_stops(reading: str, line: int, unresolved: tuple[Stop, ...]) -> tuple[Stop, ...]:
    """The columns the statement *reading* writes and :func:`sqltext.writes` does not return (:attr:`Execution.unread_columns`).

    A table no column of which is read (:data:`sqltext.COLUMNS_NOT_READ`) is
    left out where a fragment of the statement has no text of its own: that
    stop already says the statement is not all here, and names the expression
    to open -- ``JobSpec.bindValuesExpression()``, where a reading of one file
    cannot follow the call into the spec.
    """
    unrendered = any(stop.reason == FRAGMENT_NOT_RENDERED for stop in unresolved)
    return tuple(
        sorted(
            {
                Stop(line, what, why)
                for what, why in sqltext.unread_columns(reading)
                if not (unrendered and why == sqltext.COLUMNS_NOT_READ)
            }
        )
    )


def _own_walk(func: ast.AST) -> Iterator[ast.AST]:
    """*func*'s nodes in :func:`ast.walk`'s order, without the bodies of the functions and classes it defines.

    Those run elsewhere and are read as their own owners (:func:`bamboo.codemap.calls.own_nodes` reads the same
    nodes, depth first).  Walked into, ``datasetManager.main``'s thread classes made each of their statements a stop
    of ``main`` -- "no statement this function builds reaches this call" -- and a gap under ``main`` in the answer.
    The order is ``ast.walk``'s, because a statement run on two lines is named by the first run read.
    """
    todo = collections.deque([func])
    while todo:
        node = todo.popleft()
        todo.extend(
            child
            for child in ast.iter_child_nodes(node)
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef))
        )
        yield node


def _executions_read(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[list[Execution], list["Skip"]]:
    """:func:`executions` and its skips, read off one run of *func*."""
    reading = _evaluation(func)
    found: list[Execution] = []
    skips: list[Skip] = []
    names = runners()
    for node in _own_walk(func):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in names:
            continue
        forwarded = node.func.attr not in _CURSOR
        if _AT_CALL and ruled_out(node):
            continue
        if not node.args:
            continue
        # The name the call runs, as written: ``sql_exe = (sql_jt + comment) %
        # table`` is read where ``sql_exe`` runs.
        statement, _fillers, _spelling = _call_site_fill(func, node.args[0])
        operands = _concatenated(statement)
        held = operands[0].id if isinstance(operands[0], ast.Name) else ""
        line = node.lineno
        if line not in reading.reached:
            skips.append(Skip(line, SKIP_UNREACHED, held or _clipped(_unparsed(operands[0]))))
            continue
        if held:
            drawn = reading.heads(line, held)
        else:
            drawn = [(one, []) for one in reading.cover(line)[0]]
        drawn = [
            (one, [_fragments_out(head, func) for head in heads])
            for one, heads in drawn
            if not _says_nothing(_fragments_out(one, func).text)
        ]
        if not drawn:
            skip = _skip_without_text(func, node, operands[0])
            if skip is not None:
                skips.append(skip)
            continue
        # ``execute(sql)`` runs with no map: a run with no binds, as one whose map is written into the call has
        # none by a name.
        varmap = node.args[1].id if len(node.args) > 1 and isinstance(node.args[1], ast.Name) else None
        window = binding_window(func, varmap, line) if forwarded and varmap is not None else None
        for one, heads in drawn:
            run = _fragments_out(one, func)
            read_as = reading_text(run.text, run.holes)
            # What a stop names is what the code wrote: a repeated part as its passes.
            unresolved = _fragment_stops(one, func) + _hole_stops(sqltext.unmarked(read_as), line)
            found.append(
                Execution(
                    sql=run.text,
                    varmap=varmap,
                    variable=held,
                    call=node,
                    window=window,
                    unresolved=unresolved,
                    head=heads[0].text if heads else "",
                    pieces=run.pieces,
                    holes=run.holes,
                    unread_columns=_column_stops(read_as, line, unresolved),
                )
            )
    # A stop is on the statement, so a table no column of which one build reads
    # is said only where no build of that statement read a column of it or said
    # why: the build of a ``SET {}`` in which no attribute changed, or of
    # ``updateJob``'s ``if cType == "hold": ... elif cType == "touse": ...``
    # past both arms, writes nothing the others do not.
    said: set[int] = set()
    read: dict[int, set[str]] = {}
    for run in found:
        if any(stop.reason != sqltext.COLUMNS_NOT_READ for stop in run.unread_columns):
            said.add(id(run.call))
        if any(stop.reason == sqltext.COLUMNS_NOT_READ for stop in run.unread_columns):
            continue
        read.setdefault(id(run.call), set()).update(write.table.lower() for write in sqltext.writes(run.reading))
    found = [
        run._replace(
            unread_columns=tuple(
                stop
                for stop in run.unread_columns
                if stop.reason != sqltext.COLUMNS_NOT_READ
                or not (id(run.call) in said or stop.source.lower() in read.get(id(run.call), set()))
            )
        )
        for run in found
    ]
    together: dict[tuple, dict[ast.Call, None]] = {}
    for run in found:
        together.setdefault((run.variable, run.sql, run.varmap, run.window), {})[run.call] = None
    return (
        [
            run._replace(calls=tuple(together[(run.variable, run.sql, run.varmap, run.window)]))
            for run in found
        ],
        skips,
    )
