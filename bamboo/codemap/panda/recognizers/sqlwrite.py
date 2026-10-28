"""SQL write recognizer -- where the code settles a subject's value in the database.

The dominant write form in PanDA, and the one the attribute slice does not see.
A knight decides and a ``db_proxy_mods`` method writes, so most task and job
state changes reach the database as a bind rather than as an attribute
assignment::

    sqlU = f"UPDATE {panda_config.schemaJEDI}.JEDI_Tasks "
    sqlU += "SET status=:status,modificationTime=:updateTime "
    sqlU += " WHERE jediTaskID=:jediTaskID "
    varMap[":status"] = "tobroken"
    self.cur.execute(sqlU + comment, varMap)

Two things had to be settled before this could be read at all, and both are in
``sql`` and ``attribution`` rather than here:

* which statement a bind belongs to -- the ``execute`` call pairs them, and
  the ``SET`` clause is what distinguishes a write from a ``WHERE`` predicate
  using the identically named bind;
* which spec class a table holds -- nothing declares it, so it is inferred
  from the column names and pooled per table.

The junction is anchored where the value is *decided*, not at the ``execute``:
that is where the surrounding ``if`` says why, which is what a reader following
the map back needs to see.  Which statement that is depends on the form -- a
bind is decided at the Python assignment filling it, a value written inline at
the fragment that appended it to the statement.

Three forms reach a subject, and the third was the plan's named blind spot::

    SET status=:status      # decided in Python
    SET status='ready'      # decided in the statement
    SET status=oldStatus    # carried from another column

The last one is how a task comes back out of ``pending``.  Missing it did not
merely lose a write: it removed "the release path never fired" from the
candidate causes of a task stuck in ``pending``, which is one of the two
symptoms this map exists to explain.

``read_side`` reads the same statements the other way, at two resolutions.  A
``WHERE`` clause says which rows were asked for, and a status nothing selects
on is a status nothing moves a task out of; the table it names says *whose*
rows, which is the only thing a query predicated on nothing but a join key has
to say -- and the thing that makes "the finish is waiting on jobs" a fact the
map holds rather than one it has no room for.
"""

from __future__ import annotations

import ast
import dataclasses
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, NamedTuple, Optional

from bamboo.codemap.astwalk import walk
from bamboo.codemap.calls import (
    FILLERS,
    UNPLACED_NO_METHOD,
    UNSET,
    Calls,
    CallSite,
    Placed,
    Recorded,
    Target,
    Types,
    own_nodes,
    receiver_parameter,
)
from bamboo.codemap.models import (
    CALL_READING_ROUTE,
    KEY_ATTRIBUTE,
    KEY_BOUND,
    KEY_COLUMN_EXPRESSION,
    KEY_COLUMN_HOLE,
    KEY_COLUMN_NONE,
    KEY_COLUMN_SEVERAL,
    KEY_COLUMN_STAR,
    KEY_COLUMN_TEXTS,
    KEY_COLUMN_UNREAD,
    KEY_CONSTANT,
    KEY_HANDED,
    KEY_KEPT,
    KEY_NO_BIND,
    KEY_NO_MAP,
    KEY_NO_USE,
    KEY_PARAMETER,
    KEY_RETURNED,
    KEY_ROWS,
    KEY_SELF_FIELD,
    KEY_SEQUENCE,
    KEY_UNKNOWN,
    NEW_ROW_COMPUTED,
    NEW_ROW_HANDED,
    NEW_ROW_LITERAL,
    NEW_ROW_MOVED,
    NEW_ROW_NO_KEY,
    NEW_ROW_OTHER_CLASS,
    NEW_ROW_RETURNING,
    NEW_ROW_REWRITTEN,
    NEW_ROW_SEQUENCE,
    NEW_ROW_UNREAD,
    NEW_ROW_UPSERT,
    RESULT_AGGREGATE,
    RESULT_ROWS,
    RESULT_VARIES,
    AddedTerm,
    Anchor,
    Branch,
    CallReading,
    ConditionalGuard,
    ConditionNode,
    ConditionReading,
    ConditionValue,
    CoverageStat,
    DiagnosticTemplate,
    EntityNode,
    GuardRead,
    JoinPair,
    JunctionNode,
    KeyFrom,
    KeySource,
    OpenCrossing,
    OrderHole,
    PrintedBy,
    ReadingStop,
    ReadSiteNode,
    RowKeySource,
    SourceModule,
    SubjectNode,
    TermGuard,
    TermPlace,
    pooled_frames,
)
from bamboo.codemap.panda import pathcond, sql, sqleval, sqltext, values
from bamboo.codemap.panda.attribution import SpecAttributor, loop_only_names
from bamboo.codemap.panda.pathcond import (
    ancestors,
    attach_parents,
    compatible,
    enclosing_function,
    entries,
    functions_with_owner,
    literal_values,
    names_bound_by,
    path_condition,
    reaching_definitions,
    site_of,
    surviving_writes,
    targets_of,
)

# The block a decision records about itself reads the same way whichever slice
# found the write, so it is read in one place and imported rather than copied.
# Both halves of the map disagreeing about a fact is the failure this codebase
# has spent the most effort undoing.
from bamboo.codemap.panda.recognizers import progress

if TYPE_CHECKING:
    from bamboo.codemap.reach import ReachGraph

SLICE_NAME = "sql-write"

#: Why a written column produced no outcome, named at the branch that decided
#: it rather than inferred afterwards.  Counting is not enough here: the slice
#: went five rounds with a ratio that said 89% and nothing that said which
#: tenth, and a shape that should be read reads exactly like a value this code
#: does not decide.
#:
#: ``ACCOUNTED_FOR`` holds the reasons that are **not** gaps in the extraction
#: -- the same distinction ``uncovered_tables`` already draws for a table that
#: holds no spec.  Kept as a set beside the strings so a reader adding a reason
#: has to decide which side it is on.
NO_VARMAP = "the execute call names no varmap"
VARMAP_FROM_OUTSIDE = "the varmap is not built in this function"
BOUND_OUTSIDE_WINDOW = "bound outside the window this execution owns"
PLACEHOLDER_COMPUTED = "the placeholder is spelled by an expression"
VARMAP_IS_A_LIST = "the varmap is one of a list handed to executemany"
NOTHING_BINDS_IT = "nothing in this function binds this placeholder"
NO_DECIDING_FRAGMENT = "the fragment that wrote this value could not be located"

#: The read side counts statements, not columns, so its reasons are about how
#: a query is run rather than where a value came from.  ``sql.executions``
#: accepts a call by a runner's name (``sql.runners``: the cursor's
#: ``execute`` / ``executemany`` and the functions handing their statement on
#: to one); each reason below is one of the ways a statement this slice
#: assembled fails to reach it.
RUN_BUT_TEXT_DIFFERS = "run through an accepted call, but the call is never handed this text"
RUN_THROUGH_OTHER_FORM = "run through a call form the reader does not accept"
NOT_RUN_HERE = "assembled here and not run here"

#: Reasons that are not gaps in the extraction.  ``NOT_RUN_HERE`` is a statement
#: handed to a caller to run, so the function that assembled it is not where it
#: could have been read.  ``BOUND_OUTSIDE_WINDOW`` is the window doing its job;
#: see :func:`sql.bound_values`.  ``VARMAP_FROM_OUTSIDE`` is here for the shape
#: where the map really does arrive as a parameter, and on this corpus it fires
#: **nowhere** -- which is a correction worth keeping: 51 columns were first read
#: that way, and the varmap turned out to be a list handed to ``executemany``
#: with every value filled in the same function.  A reason that sounds like "not
#: our business" is the one to check against the call form before believing.
#:
#: What is deliberately *not* here: ``hs_scrapers`` writes five of seven columns
#: from a row of a parsed HTML table, and those land on the gap side under
#: ``NOTHING_BINDS_IT``.  The walk has a terminal for exactly that ("a value
#: another system supplies"), but nothing here can yet tell such a row from a
#: helper filling the same map, and guessing would put a judgement where a
#: measurement belongs.
#:
#: **Being on this side does not take a candidate out of the denominator.**  The
#: ratio still counts it, because changing what counts as a candidate would
#: move ``slice coverage`` and hide the judgement inside a number that looks
#: like a measurement.
ACCOUNTED_FOR = frozenset(
    {VARMAP_FROM_OUTSIDE, BOUND_OUTSIDE_WINDOW, NOT_RUN_HERE}
)


#: A row precondition naming a bind nothing in the function fills.  Kept as
#: the raw term rather than dropped -- ``jobStatus=:oldJobStatus`` still says
#: truly which column is tested -- and counted, because the value is what makes
#: the guard comparable with what a reader selects on.
PRECONDITION_BIND_UNSETTLED = "the row precondition names a bind nothing here fills"


class Unexplained(NamedTuple):
    """One counted column the slice could not account for, and why.

    Carries the junction name it *would* have joined so that a reader can tell
    the two severities apart: a column whose junction exists is a way of
    setting a value missing from a junction that looks complete, and the map's
    enumeration is the thing being relied on.  A column whose junction does not
    exist is a value the map cannot offer at all.
    """

    reason: str
    file: str
    owner: str
    table: str
    column: str
    junction: str
    detail: str = ""
    """What the failing branch had in hand, for a reader who has to go look.

    The reason names a class; this names the instance.  Without it "nothing in
    this function binds this placeholder" sends a reader to a function that
    plainly does bind it, and the disagreement is in which varmap or which
    spelling -- neither of which the class can carry.
    """


def _declared_spelling(
    attributor: SpecAttributor, spec_class: str, column: str
) -> Optional[str]:
    """Return the attribute as the spec declares it, matching case-insensitively.

    SQL and Python disagree on case for the same field -- ``modificationTime``
    is written ``modificationtime`` in statements -- and a subject keyed on the
    SQL spelling would never join with one keyed on the attribute spelling,
    quietly splitting one subject in two.
    """
    declared = attributor.declared_attributes(spec_class)
    lowered = column.lower()
    for attribute in declared:
        if attribute.lower() == lowered:
            return attribute
    return None


def _subject_of(
    attributor: SpecAttributor,
    spec_class: Optional[str],
    table: str,
    column: str,
) -> tuple[str, str, str]:
    """Return ``(qualifier, attribute, qualifier kind)`` for a written column.

    A subject's key needs a qualifier that disambiguates -- ``status`` means
    eight different things unqualified -- and a *table* qualifies as well as a
    class does.  For a SQL write it is arguably the better name: the class is
    inferred from the columns, while the table is what the statement says.

    **The class stays canonical wherever one exists**, though, because
    ``jobsActive4``, ``jobsDefined4`` and ``jobsArchived4`` are one
    ``JobSpec.jobStatus`` split across a job's lifetime.  Keying on the table
    would make three subjects out of one, and would file an attribute write and
    a SQL write to the same field under different names.

    Where no class exists the table qualifies instead, which is how
    ``ddm_endpoint.blacklisted`` -- the writer behind a blacklisted RSE, and a
    root cause the map is supposed to reach -- becomes reachable at all.  What
    tells the bookkeeping apart is the criteria, not the absence of a spec class.
    """
    if spec_class is not None:
        attribute = _declared_spelling(attributor, spec_class, column)
        if attribute is not None:
            return spec_class, attribute, "spec"
        # The table holds this spec but the column is not a declared attribute
        # -- a join key or a housekeeping column.  It is still a column of a
        # real table, so it is qualified by the table like any other.
    return table, column, "table"


def _term_subject(attributor: SpecAttributor, term: "sqltext.Term") -> str:
    """The subject a statement's term is about, named as a write to that column would be."""
    return _column_subject(attributor, term.table, term.column)


def _column_subject(attributor: SpecAttributor, table: str, column: str) -> str:
    """The subject *table*'s *column* is, named as a write to that column would be."""
    qualifier, attribute, _kind = _subject_of(attributor, attributor.class_for_table(table), table, column)
    return SubjectNode.make_name(qualifier, attribute)


def _run_stops(run: sql.Execution, rel_path: str, owner: str) -> list[ReadingStop]:
    """Where reading one statement that *was* read stopped.

    With the tables the statement reads, so a stop can be told apart by
    whether it is in a query at all.  A gap in the ``VALUES`` of an ``INSERT``
    and a gap in a ``SELECT``'s ``WHERE`` are the same reason and different
    answers to "which rows did this function pick up".  The columns a write
    names that were not read (:attr:`sql.Execution.unread_columns`) are stops
    too, and a plain ``UPDATE`` reads no table, so they are not about rows.
    """
    read_tables = sorted({table for verb, table in _table_verbs(run.reading) if verb == "read"})
    return [
        ReadingStop(
            reason=stop.reason,
            detail=stop.source,
            file=rel_path,
            owner=owner,
            line=stop.line,
            statement=run.call.lineno,
            reads=read_tables,
        )
        for stop in (*run.unresolved, *run.unread_columns)
    ]


def _skip_stops(skipped: list[sql.Skip], rel_path: str, owner: str) -> list[ReadingStop]:
    """An execution nothing was read out of, as a stop on the statement it runs.

    Its text is not known, so its tables are not either, and the stop says so
    rather than reading as a write.
    """
    return [
        ReadingStop(
            reason=skip.reason,
            detail=skip.detail,
            file=rel_path,
            owner=owner,
            line=skip.line,
            statement=skip.line,
            reads=[],
            reads_known=False,
        )
        for skip in skipped
    ]


def distinct_stops(stops: list[ReadingStop]) -> list[ReadingStop]:
    """One stop per place the reading stopped, the tables of every variant merged.

    A statement is read once per variant -- ``lockJobsForReassign`` builds
    one per table it may read -- and each variant carried the same stop, so
    a fragment the build could not render was counted as many times as the
    statement had shapes.  The count was of shapes, the list of places.
    """
    merged: dict[tuple, ReadingStop] = {}
    for stop in stops:
        key = (stop.reason, stop.detail, stop.file, stop.owner, stop.line, stop.statement)
        held = merged.get(key)
        if held is None:
            merged[key] = stop.model_copy(update={"reads": list(stop.reads)})
            continue
        held.reads = sorted(set(held.reads) | set(stop.reads))
        held.reads_known = held.reads_known and stop.reads_known
    return list(merged.values())


def statement_stops(
    func: ast.FunctionDef | ast.AsyncFunctionDef, rel_path: str
) -> list[ReadingStop]:
    """Every place reading *func*'s statements stopped, the build's way.

    The read half of what :func:`extract` records, for a caller holding one
    function rather than the corpus: ``derive-strategy`` says under each
    function that moves a value which of its statements the map could not
    read, and a second reading of the same source is how two counts of one
    thing come to disagree.  The precondition stops are the write's and are
    left to ``extract``.
    """
    owner = f"{rel_path}::{func.name}"
    found: list[ReadingStop] = []
    seen: set[tuple[str, str, Optional[str], Optional[tuple[int, int]]]] = set()
    skipped: list[sql.Skip] = []
    for run in sql.executions(func, skipped):
        key = (run.variable, run.sql, run.varmap, run.window)
        if key in seen:
            continue
        seen.add(key)
        found.extend(_run_stops(run, rel_path, owner))
    found.extend(_skip_stops(skipped, rel_path, owner))
    return distinct_stops(found)


def _row_key_source(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    run: sql.Execution,
    statement: sqltext.SqlWrite,
    rel_path: str,
) -> RowKeySource:
    """Where a guarded write's row keys were filled from -- see :class:`RowKeySource`.

    The values only: who handed them is read after the reach record, by the carry (:func:`attach_key_sources`).
    """
    values = []
    if run.varmap is not None:
        for _column, bind in sorted(statement.keys.items()):
            values.extend(
                (filled.value, filled.site, frozenset())
                for filled in sql.bound_values(func, run.varmap, bind, run.window, at=run)
            )
    source = RowKeySource(
        owner=f"{rel_path}::{func.name}",
        file=rel_path,
        statement=run.call.lineno,
        columns=sorted(statement.preconditions),
        keys=[f"{column}={bind}" for column, bind in sorted(statement.keys.items())],
    )
    source._filled = (func, rel_path, values)
    return source


def _key_froms(
    keyed: dict[tuple[str, int, str, str], KeyFrom],
    key_froms: list[KeyFrom],
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    run: sql.Execution,
    statement: sqltext.SqlWrite,
    rel_path: str,
    written: dict[str, set[int]],
) -> None:
    """Add *run*'s row-naming terms of *statement* to their :class:`KeyFrom` s, one per term however many texts run it.

    The binds only -- where their values came from is read after the reach record (:func:`attach_key_sources`).
    A term none of whose runs has a bind map is :data:`KEY_NO_MAP`; one whose runs fill none, :data:`KEY_NO_BIND`.
    """
    owner = f"{rel_path}::{func.name}"
    terms = [(f"{column}={bind}", bind) for column, bind in sorted(statement.keys.items())]
    terms += [(f"{column} IN ({bind})", bind) for column, bind in sorted(statement.key_lists.items())]
    for term, bind in terms:
        name = (owner, run.call.lineno, statement.table, term)
        record = keyed.get(name)
        if record is None:
            record = keyed[name] = KeyFrom(
                owner=owner, file=rel_path, statement=run.call.lineno, table=statement.table, term=term, unread=KEY_NO_MAP
            )
            record._binds = (func, rel_path, [])
            key_froms.append(record)
        _ran_by(record, func, run)
        for subject, lines in written.items():
            record.writes[subject] = sorted({*record.writes.get(subject, ()), *lines})
        # A map written into the call -- ``execute(sql, {":vo": vo})`` -- is a bind map as a named one is.
        found = (
            sql.bound_values(func, run.varmap, bind, run.window, at=run)
            if run.varmap is not None
            else sql.inline_binds(run.call, bind)
        )
        if run.varmap is None and not found:
            continue
        binds = record._binds[2]
        binds.extend(found)
        record.unread = "" if binds else KEY_NO_BIND


def _ran_by(record: KeyFrom, func: ast.FunctionDef | ast.AsyncFunctionDef, run: sql.Execution) -> None:
    """Add the calls running *run*'s text with its map to where *record*'s write runs from (``KeyFrom._runs``)."""
    if record._runs is None:
        record._runs = (func, [])
    places = record._runs[1]
    places.extend(call for call in (run.calls or (run.call,)) if not any(call is one for one in places))
    if not any(run is one for one in record._executions):
        record._executions.append(run)


_INSERT_INTO = re.compile(r"^\s*INSERT\s+(?:/\*.*?\*/\s*)?INTO\s+([\w.{}$]+)", re.IGNORECASE | re.DOTALL)
#: A statement selecting a sequence's next values -- ``SELECT ...JEDI_DATASET_CONT_FILEID_SEQ.nextval FROM (SELECT
#: level FROM dual CONNECT BY level<=:nIDs)`` -- and the sequence.
_SELECTS_SEQUENCE = re.compile(r"^\s*SELECT\s+(?:/\*.*?\*/\s*)?([\w.{}$]+)\.nextval\b", re.IGNORECASE | re.DOTALL)
_RETURNING = re.compile(r"\bRETURNING\s+(.*?)\s+INTO\b", re.IGNORECASE | re.DOTALL)
#: Calls making one value of another without making it new: ``int(row[0])`` is still the row's.
_CONVERSIONS = frozenset({"int", "str", "float", "long"})


@dataclasses.dataclass
class _Insert:
    """What decides an ``INSERT`` KeyFrom's :attr:`KeyFrom.new_row` beside its sources (:func:`_new_row`).

    *value* is how the statement fills the key column: settled in it (``literal``, ``expression``, ``sequence``
    -- a ``nextval`` -- or ``returning``, the column ``RETURNING ... INTO`` hands back), by a bind or an object's
    field read by the carry (``bind`` / ``object``), or not at all (``absent``, and ``none`` for a table with no
    key).  *binds* is each bind value as written: ``(literal | computed | returned | plain, line, the class of its
    root object)`` -- ``returned`` is a cursor variable read back (``varMap[":newPandaID"].getvalue()``).
    """

    table_class: str
    value: str
    text: str = ""
    binds: list = dataclasses.field(default_factory=list)
    #: The line of a ``DELETE`` in the function naming the same table's whole key: the row is written again.
    rewritten: Optional[int] = None
    #: The function's statements returning a key (a sequence's next values are a source of their own, KEY_SEQUENCE).
    returning_lines: frozenset = frozenset()


@dataclasses.dataclass
class _Writing:
    """A function's statements as an ``INSERT``'s mark reads them: which return a key, which delete rows by which
    columns (``{table: [(columns, call)]}``), and which calls update a table (``{(line, table)}``) -- one call
    running an ``UPDATE`` text or an ``INSERT`` one is an upsert.  A run's lines are all of its calls'
    (:attr:`sql.Execution.run_at`)."""

    returning_lines: frozenset
    deletes: dict
    updated: frozenset = frozenset()

    @classmethod
    def of(cls, runs: list[sql.Execution]) -> "_Writing":
        returning, deletes, updated = set(), {}, set()
        for run in runs:
            lines = {call.lineno for call in run.run_at}
            text = run.sql or ""
            updated |= {
                (at, s.table.lower()) for s in sqltext.writes(run.reading) if s.kind == "update" for at in lines
            }
            if _RETURNING.search(text):
                returning |= lines
            if re.search(r"\bDELETE\s+FROM\b", text, re.IGNORECASE):
                columns: dict[str, set[str]] = {}
                for term in sqltext.terms(text).terms:
                    if term.verb == "deleted" and not term.excluding:
                        columns.setdefault(term.table.lower(), set()).add(term.column.lower())
                for table, named in columns.items():
                    deletes.setdefault(table, []).extend((frozenset(named), call) for call in run.run_at)
        return cls(frozenset(returning), deletes, frozenset(updated))

    def rewritten(self, table: str, identity: Iterable[str], run: sql.Execution) -> Optional[int]:
        """The line of a ``DELETE`` of *table* naming its whole key that runs first: above a call of *run* and
        reaching it -- ``lockProcess_JEDI`` deletes its lock and inserts it again.  One that runs after the
        ``INSERT`` deletes other rows (``updateWorkers``' relations the worker no longer has) or a broken one the
        failed ``INSERT`` found (``getTasksToRefine_JEDI``'s ``except``), and the row it makes is still new."""
        named = set(identity)
        if not named:
            return None
        for columns, call in self.deletes.get(table.lower(), ()):
            if named <= columns and any(
                call.lineno < at.lineno and surviving_writes([site_of(call)], site_of(at), as_run=True)
                for at in run.run_at
            ):
                return call.lineno
        return None

    def upserts(self, run: sql.Execution, table: str) -> bool:
        """Whether a call of *run* also updates *table*: ``addStdOut``'s ``sqlU`` or ``sqlI``."""
        return any((call.lineno, table.lower()) in self.updated for call in run.run_at)


def _returned(text: str) -> set[str]:
    """The columns ``RETURNING ... INTO`` hands back, lower-cased."""
    return {column.strip().lower() for column in m.group(1).split(",")} if (m := _RETURNING.search(text)) else set()


def _plain(value: ast.AST) -> bool:
    """A copy of one value: a name, an attribute or a subscript of one, or a conversion of such."""
    while isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id in _CONVERSIONS:
        if len(value.args) != 1:
            return False
        value = value.args[0]
    while isinstance(value, (ast.Attribute, ast.Subscript)):
        value = value.value
    return isinstance(value, ast.Name)


def _root(value: ast.AST) -> Optional[ast.Name]:
    while (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id in _CONVERSIONS
        and value.args
    ):
        value = value.args[0]
    while isinstance(value, (ast.Attribute, ast.Subscript)):
        value = value.value
    return value if isinstance(value, ast.Name) else None


def _bind_value(
    value: ast.AST, line: int, func: ast.AST, owner: Optional[str], attributor: SpecAttributor
) -> tuple[str, int, Optional[str]]:
    if isinstance(value, ast.Constant):
        return ("literal", line, None)
    if any(isinstance(n, ast.Call) and _callee_name(n) == "getvalue" for n in walk(value)):
        return ("returned", line, None)
    if not _plain(value):
        return ("computed", line, None)
    root = _root(value)
    return ("plain", line, attributor.class_of(root, func, owner) if root is not None else None)


def _sequenced_attributes(modules: list[SourceModule]) -> dict[str, frozenset[str]]:
    """``{spec class: the attributes its _seqAttrMap fills from a sequence}`` -- ``JobSpec``'s ``PandaID``, put in
    its ``INSERT`` as the sequence's ``nextval`` where the class is written with ``useSeq=True``."""
    found: dict[str, frozenset[str]] = {}
    for module in modules:
        for node in walk(module.tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for statement in node.body:
                if (
                    isinstance(statement, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "_seqAttrMap" for t in statement.targets)
                    and isinstance(statement.value, ast.Dict)
                ):
                    found[node.name] = frozenset(
                        k.value
                        for k in statement.value.keys
                        if isinstance(k, ast.Constant) and isinstance(k.value, str)
                    )
    return found


def _values_map(node: Optional[ast.AST]) -> Optional[tuple[ast.expr, ast.Call]]:
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "valuesMap":
        return node.func.value, node
    return None


def _objects_in(func: ast.AST, name: str, at: ast.AST) -> list[tuple[ast.expr, ast.Call]]:
    """The objects the map *name* is made of where *at* runs: each ``name = obj.valuesMap(...)`` that reaches it."""
    made = [
        node
        for node in own_nodes(func)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
        and _values_map(node.value) is not None
    ]
    if not made:
        return []
    live = sql._live_at(func, name, made, at)  # type: ignore[arg-type]
    return [_values_map(node.value) for node in made if id(node) in live]  # type: ignore[misc]


def _objects_of(run: sql.Execution, func: ast.AST) -> list[tuple[ast.expr, ast.Call]]:
    """The objects a run's bind maps are made of, each with its ``valuesMap`` call -- ``job`` of ``execute(sql,
    job.valuesMap())``, of a map named ``varMap = job.valuesMap(useSeq=True)`` that reaches a call of the run, or of
    the maps appended to the list ``executemany`` takes -- the first a call hands over directly, then those by name.
    Read at every call of the run (:attr:`sql.Execution.run_at`): ``retryJob``'s ``sql1`` holds its ``INSERT`` at the
    ``SELECT``'s call too, round the retry loop, and the object's map reaches only the second; and
    ``registerTaskInOneShot_JEDI`` runs one ``INSERT`` of a dataset from five places, each with its own map."""
    found: list[tuple[ast.expr, ast.Call]] = []
    for call in run.run_at:
        one = _values_map(call.args[1] if len(call.args) > 1 else None)
        if one is not None:
            found.append(one)
    if run.varmap:
        for call in run.run_at:
            found += _objects_in(func, run.varmap, call)
        appends = [
            node
            for node in own_nodes(func)
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "append"
            and isinstance(node.value.func.value, ast.Name)
            and node.value.func.value.id == run.varmap
            and len(node.value.args) == 1
        ]
        # Only the appends the list a call runs with still holds, as the bind reader takes them: getRandomSeed_JEDI's
        # ``var_maps = []`` before its new seeds' maps renews the list its UPDATE of the old seeds ran with.
        live: set[int] = set()
        for call in run.run_at if appends else ():
            live |= sql._live_at(func, run.varmap, appends, call)
        for node in appends:
            if id(node) not in live:
                continue
            element = node.value.args[0]  # type: ignore[attr-defined]
            one = _values_map(element)
            if one is not None:
                found.append(one)
            elif isinstance(element, ast.Name):
                found += _objects_in(func, element.id, node)
    return list({id(made): (obj, made) for obj, made in found}.values())


def _maps_of(found: list[tuple[ast.expr, ast.Call]]) -> list[ast.Call]:
    """The ``valuesMap`` calls of the first object's spelling among *found*: the maps one record's object is in."""
    spelled = ast.unparse(found[0][0]) if found else None
    return [made for obj, made in found if ast.unparse(obj) == spelled]


def _insert_from(
    keyed: dict[tuple[str, int, str, str], KeyFrom],
    key_froms: list[KeyFrom],
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    run: sql.Execution,
    table: str,
    term: str,
    rel_path: str,
    written: dict[str, set[int]],
    insert: _Insert,
    binds: list[sql.Bind],
    obj: str = "",
    obj_class: Optional[str] = None,
    mades: Optional[list[ast.Call]] = None,
) -> KeyFrom:
    owner = f"{rel_path}::{func.name}"
    # Apart from a WHERE term of the same spelling: a run that updates or inserts the row (``addStdOut``'s
    # ``sqlU`` or ``sqlI``) has both, and they are not one record.
    name = (owner, run.call.lineno, table, "\0insert " + term)
    record = keyed.get(name)
    if record is None:
        record = keyed[name] = KeyFrom(
            owner=owner, file=rel_path, statement=run.call.lineno, table=table, term=term, object=obj
        )
        record._binds = (func, rel_path, [])
        record._insert = insert
        if obj:
            record._object = (obj, obj_class, mades or [], frozenset())
        key_froms.append(record)
    else:
        record._insert.binds.extend(insert.binds)
    _ran_by(record, func, run)
    for subject, lines in written.items():
        record.writes[subject] = sorted({*record.writes.get(subject, ()), *lines})
    record._binds[2].extend(binds)
    if insert.value in ("bind", "object") and not record._binds[2]:
        record.unread = KEY_NO_BIND if run.varmap is not None or obj else KEY_NO_MAP
    else:
        record.unread = ""
    return record


def _insert_froms(
    keyed: dict[tuple[str, int, str, str], KeyFrom],
    key_froms: list[KeyFrom],
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    owner_class: Optional[str],
    run: sql.Execution,
    statement: sqltext.SqlWrite,
    rel_path: str,
    written: dict[str, set[int]],
    attributor: SpecAttributor,
    writing: _Writing,
    upsert: bool = False,
) -> None:
    """An ``INSERT`` whose text spells its columns: a :class:`KeyFrom` per column of its table's key, dated ones
    left out (:func:`sqltext.table_identity`) -- the value the statement gives it, and what decides whether it
    makes the row (:attr:`KeyFrom.new_row`).  A table with no key gets one, of no term, to say so."""
    table = statement.table
    table_class = attributor.class_for_table(table) or f"table:{table.lower()}"
    identity = sorted(sqltext.table_identity(table))
    returned = _returned(run.sql or "")
    spelled = {column.lower(): (column, supplied) for column, supplied in statement.columns.items()}
    common = {
        "table_class": table_class,
        "rewritten": writing.rewritten(table, identity, run),
        "returning_lines": writing.returning_lines,
    }
    if not identity:
        _insert_from(keyed, key_froms, func, run, table, "", rel_path, written, _Insert(value="none", **common), [])
        return
    for column in identity:
        name, supplied = spelled.get(column, (column, None))
        binds: list[sql.Bind] = []
        if upsert:
            insert, term = _Insert(value="upsert", **common), f"{name}={supplied.text}" if supplied else name
        elif column in returned:
            insert, term = _Insert(value="returning", **common), f"{name} RETURNING"
        elif supplied is None:
            insert, term = _Insert(value="absent", **common), name
        elif supplied.kind == "literal":
            insert, term = _Insert(value="literal", text=supplied.text, **common), f"{name}={supplied.text}"
        elif supplied.kind in ("expression", "column"):
            value = "sequence" if "nextval" in supplied.text.lower() else "expression"
            insert, term = _Insert(value=value, text=supplied.text, **common), f"{name}={supplied.text}"
        else:
            binds = (
                sql.bound_values(func, run.varmap, supplied.text, run.window, at=run)
                if run.varmap
                else sql.inline_binds(run.call, supplied.text)
            )
            insert = _Insert(
                value="bind",
                binds=[_bind_value(b.value, b.site.lineno, func, owner_class, attributor) for b in binds],
                **common,
            )
            term = f"{name}={supplied.text}"
        _insert_from(keyed, key_froms, func, run, table, term, rel_path, written, insert, binds)


def _fields_at(
    func: ast.AST, obj: ast.expr, attribute: str, mades: list[ast.Call], obj_class: Optional[str]
) -> list[ast.stmt]:
    """The assignments of ``obj.<attribute>`` that reach one of *mades*, the object's ``valuesMap()`` calls: one after
    the map is made is not in it (getRandomSeed_JEDI's ``tmpFileSpec.status = "ready"``), nor one from a pass before
    where the name is given a new object of its class on the way (``tmpFileSpec = JediFileSpec()``).  The name taken
    again from a list or a loop may be the same object -- insertFilesForDataset_JEDI fills its specs, keeps them, and
    inserts them from ``for fileID, fileSpec in zip(newFileIDs, fileSpecsForInsert)`` -- and cancels nothing."""
    spelled = ast.unparse(obj)
    assigned = [
        node
        for node in own_nodes(func)
        for target in targets_of(node)
        if isinstance(target, ast.Attribute)
        and target.attr == attribute
        and ast.unparse(target.value) == spelled
        and getattr(node, "value", None) is not None
    ]
    if not assigned or not mades:
        return []
    root = _root(obj)
    renewed: list[ast.AST] = []
    if root is not None and obj_class is not None:
        made_anew = obj_class.rsplit(".", 1)[-1]
        renewed = [
            node
            for node in own_nodes(func)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            and isinstance(node.value, ast.Call)
            and _callee_name(node.value) == made_anew
            and any(
                name == root.id for target in targets_of(node) for name, _slot, _unpacked in names_bound_by(target)
            )
        ]
    found: dict[int, ast.stmt] = {}
    for made in mades:
        for node in surviving_writes(assigned, site_of(made), as_run=True, killers=renewed):
            found[id(node)] = node  # type: ignore[assignment]
    return [node for node in assigned if id(node) in found]  # type: ignore[misc]


def _unspelled_froms(
    keyed: dict[tuple[str, int, str, str], KeyFrom],
    key_froms: list[KeyFrom],
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    owner_class: Optional[str],
    run: sql.Execution,
    found: list[tuple[ast.expr, ast.Call]],
    rel_path: str,
    attributor: SpecAttributor,
    writing: _Writing,
    sequenced: dict[str, frozenset[str]],
) -> None:
    """An ``INSERT`` whose text does not spell its columns: a :class:`KeyFrom` per key column of its table (of the
    object's class's tables where the text names it by a variable).  Run with an object's map --
    ``INSERT INTO jobsArchived4 ({JobSpec.columnNames()}) ...`` with ``job.valuesMap()`` -- a column's value is the
    object's field: a sequence's where ``useSeq=True`` fills it, else what the function assigns to it where the map
    is made, else the field as the object holds it.  A column ``RETURNING`` hands back is new; one no field of the
    object holds, or any of a map built otherwise (``copy_file_records``' ``VALUES (:col, ...)`` from a cursor's
    description), is not read."""
    obj, made = found[0] if found else (None, None)
    mades = _maps_of(found)
    matched = _INSERT_INTO.match(run.sql or "")
    if matched is None:
        return
    named = matched.group(1).split(".")[-1]
    table = "" if "{" in named or "$" in named else named
    obj_class = attributor.class_of(obj, func, owner_class) if obj is not None else None
    if not table and obj_class is None:
        return
    # The table's class as the subjects map it, else the object's: a table a JobSpec's valuesMap is written into
    # holds JobSpec rows.
    table_class = (attributor.class_for_table(table) if table else None) or obj_class or f"table:{table.lower()}"
    identity = sorted(
        sqltext.table_identity(table)
        if table
        else set().union(
            *(sqltext.table_identity(t) for t in attributor.tables_of(obj_class))  # type: ignore[arg-type]
        )
    )
    use_seq = made is not None and any(
        k.arg == "useSeq" and isinstance(k.value, ast.Constant) and k.value.value for k in made.keywords
    )
    attributes = {a.lower(): a for a in attributor.declared_attributes(obj_class)} if obj_class else {}
    spelled_obj = ast.unparse(obj) if obj is not None else ""
    returned = _returned(run.sql or "")
    upsert = bool(table) and writing.upserts(run, table)
    common = {
        "table_class": table_class,
        "rewritten": writing.rewritten(table, identity, run) if table else None,
        "returning_lines": writing.returning_lines,
    }
    shown = table or named

    def put(term: str, insert: _Insert, binds: list[sql.Bind]) -> None:
        _insert_from(
            keyed, key_froms, func, run, shown, term, rel_path, {}, insert, binds, spelled_obj, obj_class, mades
        )

    if not identity:
        put("", _Insert(value="none", **common), [])
        return
    for column in identity:
        attribute = attributes.get(column)
        term = f"{attribute}={spelled_obj}.{attribute}" if attribute else column
        if upsert:
            put(term, _Insert(value="upsert", **common), [])
        elif attribute and use_seq and attribute in sequenced.get(obj_class, ()):  # type: ignore[arg-type]
            put(term, _Insert(value="sequence", **common), [])
        elif column in returned:
            put(f"{attribute or column} RETURNING", _Insert(value="returning", **common), [])
        elif attribute is None or obj is None or made is None:
            put(term, _Insert(value="absent", **common), [])
        else:
            # What the function puts in the field where the map is made: insertNewJob's ``file.row_ID = None`` is
            # replaced by ``fileIDPool[...]`` before its valuesMap(), and insertFilesForDataset_JEDI's fileID is a
            # sequence's, read through the name it was assigned from, not the field.
            assigned = _fields_at(func, obj, attribute, mades, obj_class)
            values = [_bind_value(node.value, node.lineno, func, owner_class, attributor) for node in assigned]
            binds = [sql.Bind(node.value, site_of(node.value)) for node in assigned]
            if not assigned:
                field = ast.copy_location(ast.Attribute(value=obj, attr=attribute, ctx=ast.Load()), obj)
                field.parent = getattr(obj, "parent", None)  # type: ignore[attr-defined]
                values, binds = [("plain", run.call.lineno, obj_class)], [sql.Bind(field, site_of(run.call))]
            put(term, _Insert(value="object", binds=values, **common), binds)


def _mark_object(
    keyed: dict[tuple[str, int, str, str], KeyFrom],
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    owner_class: Optional[str],
    run: sql.Execution,
    statement: sqltext.SqlWrite,
    rel_path: str,
    found: list[tuple[ast.expr, ast.Call]],
    attributor: SpecAttributor,
) -> None:
    """A statement whose bind map is an object's ``valuesMap`` -- an ``UPDATE`` setting what the function changed on
    it (``SET {job.bindUpdateChangesExpression()}``): its terms' KeyFroms persist that object, and a term no bind of
    the map fills is the object's field of the bind's name (``:PandaID``, ``job.PandaID``)."""
    obj, _made = found[0]
    obj_class = attributor.class_of(obj, func, owner_class)
    if obj_class is None:
        return
    spelled_obj = ast.unparse(obj)
    attributes = {a.lower(): a for a in attributor.declared_attributes(obj_class)}
    owner = f"{rel_path}::{func.name}"
    terms = [(f"{column}={bind}", bind) for column, bind in statement.keys.items()]
    terms += [(f"{column} IN ({bind})", bind) for column, bind in statement.key_lists.items()]
    for term, bind in terms:
        record = keyed.get((owner, run.call.lineno, statement.table, term))
        if record is None:
            continue
        record.object = spelled_obj
        record._object = (spelled_obj, obj_class, _maps_of(found), frozenset())
        attribute = attributes.get(bind.lstrip(":").lower())
        if not record._binds[2] and attribute is not None:
            field = ast.copy_location(ast.Attribute(value=obj, attr=attribute, ctx=ast.Load()), obj)
            record._binds[2].append(sql.Bind(field, site_of(run.call)))
            record.unread = ""


def extract(
    modules: list[SourceModule],
    map_id: str,
    derived_from: str,
    attributor: SpecAttributor,
    row_keys: Optional[list[RowKeySource]] = None,
    key_froms: Optional[list[KeyFrom]] = None,
) -> tuple[
    list[SubjectNode],
    list[JunctionNode],
    list[CoverageStat],
    set[str],
    list[DiagnosticTemplate],
    list[Unexplained],
    list[ReadingStop],
]:
    """Extract junctions for SQL bind writes.

    Returns the tables holding no spec alongside the usual three, so the
    build can say what the map does not cover instead of burying it in a
    coverage ratio, and the diagnostic templates bound into statements -- which
    belong here rather than in a scan of their own because the column a bind
    lands in is the join this slice has already made.

    And the columns it counted and could not account for, each with the reason
    the branch that gave up recorded.  For the same argument as ``uncovered``,
    one step finer: a ratio at 89% says a tenth is missing and not which tenth,
    and this slice held 140 of those for five rounds while every number beside
    it stayed still.  Scanning for them afterwards was tried and is what this
    replaces -- a second reader of the same source disagreed with this one by 77
    columns.

    And, last, where the reading of a statement stopped -- a fragment with no
    text of its own, a ``WHERE`` whose placeholders are an interpolation, a row
    precondition whose bind nothing here fills.  Collected on this walk because
    this walk already visits every execution in the corpus, and a second one
    asking the same question is how the two counts would come to disagree.

    *attributor* is passed in already taught: the table map is learned from the
    whole corpus, so it cannot be built from the module in hand.

    *row_keys*, where given, is filled with where each guarded write's row keys
    came from.  A list to fill rather than an eighth return value: one caller
    asks for it, and the seven above are unpacked in many places.  *key_froms*
    likewise, with each write's row-naming terms (:class:`KeyFrom`), one per
    term of a statement however many texts it runs with; who filled them is
    read after the reach record (:func:`attach_key_sources`).
    """
    junctions: dict[str, JunctionNode] = {}
    carriers: dict[int, tuple[Branch, list[_Carrier]]] = {}
    keyed: dict[tuple[str, int, str, str], KeyFrom] = {}
    sequenced = _sequenced_attributes(modules) if key_froms is not None else {}
    stopped: list[ReadingStop] = []
    coverage: list[CoverageStat] = []
    attributed: set[tuple[str, str, str]] = set()
    uncovered: set[str] = set()
    diagnostics: list[DiagnosticTemplate] = []
    unexplained: list[Unexplained] = []
    settle = values.resolver()
    # A guard's binds as the read side settles its own (values.bind_settler): a list's members, what a parameter's
    # callers hand it, all of them or none.
    guard_settle = values.bind_settler(sql._CALLS)
    # The same census the attribute slice makes, for the same reading: which
    # call persists a message into a declared column.  Asked of the attributor
    # rather than recomputed from the source, so the two halves of the map
    # cannot disagree about what a setter is.
    setters = progress.spec_setters(
        modules,
        {cls: attributor.declared_attributes(cls) for cls in attributor.declared_classes()},
    )

    for module in modules:
        attach_parents(module.tree)
        candidates = 0
        explained = 0
        for func, _owner in functions_with_owner(module.tree):
            # A method that runs one statement from two places yields the same
            # run twice.  Counting both would inflate the coverage denominator
            # with duplicates and make the slice look worse than it reads.
            seen: set[tuple[str, str, Optional[str], Optional[tuple[int, int]]]] = set()
            owner = f"{module.rel_path}::{func.name}"
            skipped: list[sql.Skip] = []
            writing: Optional[_Writing] = None
            for run in sql.executions(func, skipped):
                key = (run.variable, run.sql, run.varmap, run.window)
                if key in seen:
                    continue
                seen.add(key)
                stopped.extend(_run_stops(run, module.rel_path, owner))
                statements = sqltext.writes(run.reading)
                made_of = _objects_of(run, func) if key_froms is not None else []
                inserting = _INSERT_INTO.match(run.sql or "") is not None
                if key_froms is not None and (made_of or inserting):
                    writing = writing or _Writing.of(list(sql.executions(func)))
                if writing is not None and inserting and not any(s.kind == "insert" for s in statements):
                    _unspelled_froms(
                        keyed, key_froms, func, _owner, run, made_of, module.rel_path, attributor, writing, sequenced
                    )
                for statement in statements:
                    spec_class = attributor.class_for_table(statement.table)
                    # Settled once per statement, not once per column: every
                    # column of one ``SET`` is bounded by the same ``WHERE``,
                    # and settling inside the column loop would count each
                    # unsettled bind as many times as the statement has
                    # columns.  Each test once: one testing two written columns
                    # (a lock's ``(locked_by IS NULL OR lock_time < :t)``) is
                    # filed under both.
                    terms = sorted(set(statement.preconditions.values()))
                    preconditions = []
                    for term in terms:
                        settled, done = sql.settled_precondition(
                            term, func, run.varmap, run.window, guard_settle, at=run
                        )
                        preconditions.append(settled)
                        if not done and ":" in term:
                            # ``reads`` left empty whatever else the text
                            # holds: this is the write's own guard, and a
                            # subquery beside it is not what failed to settle.
                            stopped.append(
                                ReadingStop(
                                    reason=PRECONDITION_BIND_UNSETTLED,
                                    detail=term,
                                    file=module.rel_path,
                                    owner=f"{module.rel_path}::{func.name}",
                                    line=run.call.lineno,
                                    statement=run.call.lineno,
                                )
                            )
                    # When this run carries each guard, beside every term in
                    # the order ``preconditions`` has them -- see
                    # :func:`_guard_conditions`.
                    carried_when = _guard_conditions(func, run, statement)
                    term_when: dict[str, list[str]] = {}
                    for column, term in sorted(statement.preconditions.items()):
                        if column in carried_when:
                            term_when.setdefault(term, carried_when[column])
                    whens = [term_when.get(term) for term in terms]
                    if row_keys is not None and statement.preconditions:
                        row_keys.append(_row_key_source(func, run, statement, module.rel_path))
                    # A name nobody could read is not a table that holds no
                    # spec.  Listed together they read as one finding, and only
                    # one of the two is a gap in the extraction -- see
                    # ``ReadSide.unreadable``, which counts these instead.
                    if spec_class is None and not _UNREAD_TABLE.fullmatch(statement.table):
                        uncovered.add(statement.table)
                    # Each subject this statement writes, and the lines its values are written on: its branches.
                    written: dict[str, set[int]] = {}
                    for column, supplied in statement.columns.items():
                        candidates += 1
                        qualifier, attribute, kind = _subject_of(
                            attributor, spec_class, statement.table, column
                        )
                        templates: list[tuple[str, ast.stmt]] = []
                        why: list[str] = []
                        outcomes = _outcomes(
                            attributor,
                            func=func,
                            run=run,
                            statement=statement,
                            column=column,
                            supplied=supplied,
                            spec_class=spec_class,
                            settle=settle,
                            templates=templates,
                            why=why,
                        )
                        diagnostics.extend(
                            DiagnosticTemplate(
                                map_id=map_id,
                                derived_from=derived_from,
                                template=template,
                                field=SubjectNode.make_name(qualifier, attribute),
                                form="bind",
                                anchor=Anchor(
                                    package=module.package,
                                    file=module.rel_path,
                                    line_start=node.lineno,
                                    line_end=node.end_lineno,
                                    blob_sha=module.blob_sha,
                                ),
                            )
                            for template, node in templates
                        )
                        if not outcomes:
                            unexplained.append(
                                Unexplained(
                                    reason=why[0] if why else "no reason given",
                                    detail=why[1] if len(why) > 1 else "",
                                    file=module.rel_path,
                                    owner=f"{module.rel_path}::{func.name}",
                                    table=statement.table,
                                    column=column,
                                    # The name ``_record`` would have used, so
                                    # that resolving it against the built
                                    # junctions asks the recogniser's own
                                    # question rather than a second one.
                                    junction=JunctionNode.make_name(
                                        map_id,
                                        SubjectNode.make_name(qualifier, attribute),
                                        f"{module.rel_path}::{func.name}",
                                    ),
                                )
                            )
                            continue
                        explained += 1
                        attributed.add((qualifier, attribute, kind))
                        _record(
                            junctions,
                            map_id=map_id,
                            derived_from=derived_from,
                            module=module,
                            func=func,
                            spec_class=qualifier,
                            attribute=attribute,
                            outcomes=outcomes,
                            # Every predicate the statement makes about a column
                            # it also writes, not only the one for this column:
                            # any of them failing leaves the row untouched, so
                            # they bound this write as much as its own does.
                            row_precondition=preconditions,
                            whens=whens,
                            row_precondition_spelled=terms,
                            call=run.call,
                            column=column,
                            carriers=carriers,
                            setters=setters,
                            # Called by _record before the loop moves on; the defaults keep this pass's run if it ever is not.
                            beside=lambda node, terms=terms, func=func, run=run: [
                                sql.settled_precondition(
                                    term, func, run.varmap, run.window, guard_settle, beside=node, at=run
                                )[0]
                                for term in terms
                            ],
                        )
                        written.setdefault(SubjectNode.make_name(qualifier, attribute), set()).update(
                            found.decided_at.lineno for found in outcomes
                        )
                    if key_froms is not None:
                        _key_froms(keyed, key_froms, func, run, statement, module.rel_path, written)
                        if statement.kind == "insert" and writing is not None:
                            _insert_froms(
                                keyed, key_froms, func, _owner, run, statement, module.rel_path, written,
                                attributor, writing,
                                upsert=writing.upserts(run, statement.table),
                            )
                        elif made_of:
                            _mark_object(keyed, func, _owner, run, statement, module.rel_path, made_of, attributor)
            # The executions nothing was read out of, last: they have no run
            # to hang on, and saying them is what tells "read in full" from
            # "never read".
            stopped.extend(_skip_stops(skipped, module.rel_path, owner))
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
            name=SubjectNode.make_name(qualifier, attribute),
            spec_class=qualifier,
            qualifier_kind=kind,
            attribute=attribute,
            criteria=["sql-write"],
        )
        for qualifier, attribute, kind in sorted(attributed)
    ]
    _settle_conditional_guards(carriers.values())
    # ``unexplained`` and ``stopped`` last: they are the two that are about
    # what this pass could *not* do, and a caller ignoring them still builds a
    # map.  Two lists rather than one because they answer different questions
    # -- a column with no outcome, and a statement read only in part.
    return (
        subjects,
        list(junctions.values()),
        coverage,
        uncovered,
        diagnostics,
        unexplained,
        stopped,
    )


def _runs_many(run: sql.Execution) -> bool:
    """Is this execution ``executemany``, or a runner handing its map on to it, whose varmap argument is a sequence?

    Asked of the call the reader already paired with the statement, so the
    answer cannot disagree with the pairing.
    """
    call = getattr(run, "call", None)
    if call is None or not isinstance(getattr(call, "func", None), ast.Attribute):
        return False
    runner = sql.runners().get(call.func.attr)
    return bool(runner is not None and runner.many)


def _deciding_fragment(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    run: sql.Execution,
    column: str,
    supplied: sqltext.ColumnValue,
    kind: str,
) -> Optional[ast.stmt]:
    """Return the statement fragment that put ``<column>=<value>`` into the SQL.

    For a value written inline there is no bind assignment to anchor at, and
    anchoring at the ``execute`` instead would throw away the reason: the two
    forms of ``getTasksToExecCommand_JEDI``'s update are appended in the two
    arms of one ``if``, so the fragment carries the condition and the statement
    does not.

    **The match is made on the assembled variant and mapped back.**  Reading a
    fragment at a time is what ``assigns_in`` did, and it answers only for
    ``SET col=value``, where the pair is adjacent.  An ``INSERT`` names its
    columns in one parenthesised run and its values in another, so the pair
    straddles whatever line breaks the author chose -- sixty-eight columns went
    unanchored for nothing but that, and the same reading lost three ``UPDATE``
    values written as multi-line ``CASE`` expressions.  Two statements spelling
    one thing two ways must not get two answers.

    Grouping fragments by the guard they share does not work either.  Given a
    head, a second fragment, and a third appended under an ``if``, a reader can
    meet the statement with the third or without it; the group holding the
    third alone is neither, and nothing ever runs it.  :func:`sql.variant_spans`
    enumerates the statements that exist and says which fragment contributed
    which stretch, so the guard falls out of the mapping rather than being the
    thing split on.

    **A statement with no local name is anchored at its call.**  The daemons
    write the text into ``querySQLS`` directly, so there are no fragments and
    the call is where the text is -- the opposite of the case above rather than
    an exception to it.
    """
    if not run.variable:
        return run.call
    for statement, spans in sql.variant_spans(func, run.variable, at=run.call, text=run.head or None, read=True):
        at = sqltext.supplied_at(statement, kind, column, supplied)
        if at is not None:
            return sql.fragment_holding(spans, at)
    return None


class _Outcome(NamedTuple):
    """One value a column's write settles to -- see :func:`_outcomes`."""

    outcome: str
    tier: int
    node: ast.stmt
    extra: list[str]
    decided_at: ast.stmt
    #: Every text a run-time value can be -- see ``Branch.value_frames``.
    frames: tuple[str, ...] = ()


def _outcomes(
    attributor: SpecAttributor,
    *,
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    run: sql.Execution,
    statement: sqltext.SqlWrite,
    column: str,
    supplied: sqltext.ColumnValue,
    spec_class: Optional[str],
    settle,
    templates: list[tuple[str, ast.stmt]],
    why: list[str],
) -> list[_Outcome]:
    """Return ``(outcome, tier, node, extra conditions, decided at, frames)`` for a column.

    The node is where the value was decided, which differs by form: a bind is
    decided at the Python assignment filling it, and a literal or a copied
    column at the fragment that put it in the statement.

    The extra conditions are for the one form where the deciding node is not the
    whole story.  A bind filled from a local is decided twice over -- the guards
    that reached ``varMap[":status"] = newTaskStatus`` say the write happened,
    and the guards on the assignment that gave the local its value say which
    value.  Both are needed, and neither is derivable from the other's node.
    And a bind of the key the run holds can be filled again before the run:
    each later one that can replace it adds its entry
    (:func:`sql.bind_replacements`).

    *why* is appended to, and only when the return is empty: the reason a
    column produced no outcome is decided at the branch that gave up, and
    nowhere else can tell the four bind cases apart.  An out-parameter for the
    same reason *templates* is one -- it is not an outcome.

    Diagnostic templates are appended to *templates* rather than returned: they
    are not outcomes, and the caller files them against the column rather than
    the subject.  Collected here because *which column the text lands in* is the
    join this slice has already made -- on its own the bind is called
    ``:errDiag`` and could belong to any statement in the method.
    """
    if supplied.kind == "bind":
        if run.varmap is None:
            why.append(NO_VARMAP)
            return []
        scan = sql.scan_binds(func, run.varmap, supplied.text, run.window, at=run)
        # A later bind of the key the run still holds replaces this one where it runs.
        replaced = sql.bind_replacements(func, run, supplied.text, scan.binds)
        found = []
        for bind in scan.binds:
            value, site = bind.value, bind.site
            later = entries(replaced.get(id(site), ()), path_condition(site))
            settled = settle(value, func)
            if settled:
                found.extend(_Outcome(outcome, 1, site, list(later), site) for outcome in settled)
                continue
            template = values.diagnostic_template(value)
            if template:
                templates.append((template, site))
            if isinstance(value, ast.Name):
                # Reaching definitions, the same reading the attribute slice
                # makes of a local: 19% of the corpus's writes fill the bind
                # from a variable a guarded chain assigned above it.
                reached = literal_values(func, value.id, settle, at=site)
                if reached:
                    dominating = path_condition(site)
                    found.extend(
                        _Outcome(
                            outcome,
                            1,
                            site,
                            [c for c in conditions if c not in dominating]
                            + [entry for entry in later if entry not in conditions],
                            # Where the value was chosen, which is not where it
                            # was bound: ``newTaskStatus = "exhausted"`` sits in
                            # the block that also records why, and the bind is
                            # further down, after the chain has closed.
                            _statement_at(func, line) or site,
                        )
                        for outcome, conditions, line in reached
                    )
                    continue
            # The writer is known, the value is not until run time.
            # Recorded rather than dropped: localize and prune read
            # observed values, so they work from the writer alone.
            found.append(
                _Outcome(
                    f"runtime({ast.unparse(value)})",
                    2,
                    site,
                    list(later),
                    site,
                    tuple(values.value_frames(value, func)),
                )
            )
        if not found:
            # Which of the several reasons, from the walk that failed rather
            # than from a second walk asking why -- see ``sql.BindScan``.
            if _runs_many(run):
                # ``executemany(sql, varMaps)`` is handed a *sequence* of maps,
                # and none the scan reads its maps from was appended to it --
                # one a helper cuts (``for shard in create_shards(maps, 100)``),
                # one ``list(map(lambda ...))`` builds.  Read off the call form
                # rather than off the list, which is what makes it exact.
                why.append(VARMAP_IS_A_LIST)
            elif not scan.varmap_assigned:
                why.append(VARMAP_FROM_OUTSIDE)
            elif scan.outside_window:
                why.append(BOUND_OUTSIDE_WINDOW)
            elif scan.computed_keys and supplied.text not in scan.constant_keys:
                why.append(PLACEHOLDER_COMPUTED)
            else:
                why.append(NOTHING_BINDS_IT)
            why.append(
                f"varmap={run.varmap} window={run.window} "
                f"keys={len(scan.constant_keys)} computed={scan.computed_keys}"
            )
        return found

    node = _deciding_fragment(func, run, column, supplied, statement.kind)
    if node is None:
        why.append(NO_DECIDING_FRAGMENT)
        return []
    if supplied.kind == "literal":
        return [_Outcome(supplied.text, 1, node, [], node)]
    if supplied.kind == "expression":
        # ``stateChangeTime=CURRENT_DATE``, ``nFiles=nFiles+:iFiles``: the
        # database decides, from the clock or from the row's own prior
        # contents.  Tier 2 for the reason a bind filled at run time is --
        # the writer is known and that is what localize and prune read -- and
        # dropping it instead was not neutral.  Promotion weighs a subject's
        # literal writes against all of them, so a column the source only ever
        # increments looked, from its one ``= 0``, like a field with a closed
        # vocabulary of one.
        return [_Outcome(f"runtime({supplied.text})", 2, node, [], node)]

    # A copied column.  The source is named as a subject rather than as a bare
    # column so the edge joins: ``passthrough(JediTaskSpec.oldStatus)`` points
    # at a node the backward walk can continue from, ``passthrough(oldStatus)``
    # at a string.
    source, attribute, _kind = _subject_of(
        attributor, spec_class, statement.table, supplied.text
    )
    outcome = f"passthrough({SubjectNode.make_name(source, attribute)})"
    return [_Outcome(outcome, 2, node, [], node)]


def _statement_at(
    func: ast.FunctionDef | ast.AsyncFunctionDef, line: int
) -> Optional[ast.stmt]:
    """The statement of *func* that starts on *line*.

    ``literal_values`` answers with a line rather than a node, and the block
    around that line is what says why the value was chosen.  Looked up rather
    than threaded through because the resolver is shared with the attribute
    slice, which has the node in hand and needs no line at all.
    """
    for node in walk(func):
        if isinstance(node, ast.stmt) and node.lineno == line:
            return node
    return None


def _record(
    junctions: dict[str, JunctionNode],
    *,
    map_id: str,
    derived_from: str,
    module: SourceModule,
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    spec_class: str,
    attribute: str,
    outcomes: list[_Outcome],
    row_precondition: list[str],
    setters: dict[str, object],
    beside=None,
    whens: Optional[list[Optional[list[str]]]] = None,
    row_precondition_spelled: Optional[list[str]] = None,
    call: Optional[ast.AST] = None,
    column: str = "",
    carriers: Optional[dict[int, tuple[Branch, list[_Carrier]]]] = None,
) -> None:
    """Add one branch per decided value to this write site's junction.

    *beside*, where given, settles the row precondition for one branch from
    the binds that can run with the statement deciding its value -- see
    :func:`sql.settled_precondition`.

    A branch another statement already made keeps only the terms this one
    gives it too: the write lands under every statement that writes the value,
    so a guard one of them leaves out is not one the value needs.  Kept from
    the first statement, the answer followed the order statements were read
    in -- ``ppEventServiceJob`` writes ``ready`` by an ``UPDATE`` guarded on
    the event's status and by two ``INSERT ... SELECT`` that guard nothing,
    and said the guard for all three.

    *whens*, beside the terms in the order *beside* gives them, says when this
    statement carries each (``None`` where that cannot be said).  What each
    statement writing a branch's value into *column* carries, run by *call*,
    goes into *carriers*, for :func:`_settle_conditional_guards` to say which
    guards some of the statements the call runs carry and when.
    """
    subject = SubjectNode.make_name(spec_class, attribute)
    owner = f"{module.rel_path}::{func.name}"
    name = JunctionNode.make_name(map_id, subject, owner)

    junction = junctions.get(name)
    if junction is None:
        junction = JunctionNode(
            map_id=map_id,
            derived_from=derived_from,
            name=name,
            subject=subject,
            owner=owner,
            # The table settled the class, and a table holds one kind of row.
            attribution="certain",
            anchor=Anchor(
                package=module.package,
                file=module.rel_path,
                # Over all the outcomes, not the first and last the scan
                # happened to visit: ``outcomes`` is in discovery order, so an
                # arm above the first one gave seventeen junctions a span that
                # ends before it starts.  Nothing complained, because no
                # production reader looks at ``line_end`` (see ``Anchor``).
                line_start=min(found.node.lineno for found in outcomes),
                line_end=max(
                    found.node.end_lineno or found.node.lineno for found in outcomes
                ),
                blob_sha=module.blob_sha,
            ),
        )
        junctions[name] = junction
    else:
        # And over every statement's: one text of a statement writes only the
        # outcomes its own branch binds (``sql.bound_values``), so the first
        # statement read is not every arm.
        junction.anchor.line_start = min(junction.anchor.line_start, *(found.node.lineno for found in outcomes))
        junction.anchor.line_end = max(
            junction.anchor.line_end or junction.anchor.line_start,
            *(found.node.end_lineno or found.node.lineno for found in outcomes),
        )

    known = {(branch.outcome, tuple(branch.path_condition)): branch for branch in junction.branches}
    for outcome, tier, node, extra, decided_at, frames in outcomes:
        condition = path_condition(node) + extra
        guards = beside(node) if beside is not None else row_precondition
        # The tests the branch's own condition already makes are not what tells
        # one statement from another.
        spelled = sorted(row_precondition_spelled or [])
        conditional = [
            (ConditionalGuard(term=term, when=[test for test in when if test not in condition]), raw)
            for term, when, raw in zip(guards, whens or [], spelled, strict=False)
            if when is not None
        ]
        carrier = _Carrier(func=func, call=call, column=column, conditional=conditional)
        held = known.get((outcome, tuple(condition)))
        if held is not None:
            held.value_frames = pooled_frames(held.value_frames, list(frames))
            held.row_precondition = [term for term in held.row_precondition if term in guards]
            if decided_at.lineno != held.line and decided_at.lineno not in held.other_lines:
                held.other_lines = sorted([*held.other_lines, decided_at.lineno])
            if carriers is not None:
                carriers[id(held)][1].append(carrier)
            continue
        # Read at the deciding statement rather than at the bind.  The two are
        # the same for a literal and far apart for a local: ``retryTask_JEDI``
        # picks the status and writes the reason in one block and binds it
        # after the whole chain has closed, so the bind's block says nothing
        # about which of five refusals this is.
        tags, messages = progress.recorded_signature(decided_at, func, setters)
        branch = Branch(
            outcome=outcome,
            path_condition=condition,
            row_precondition=guards,
            order=len(junction.branches),
            tier=tier,
            tags=tags,
            messages=messages,
            line=decided_at.lineno,
            value_frames=list(frames),
        )
        known[(outcome, tuple(condition))] = branch
        junction.branches.append(branch)
        if carriers is not None:
            carriers[id(branch)] = (branch, [carrier])


class _Carrier(NamedTuple):
    """One statement writing a branch's value into *column*, run by *call* in *func*, and when it carries each
    guard: the guard as the branch says it, beside the term as the statement spells it."""

    func: ast.FunctionDef | ast.AsyncFunctionDef
    call: Optional[ast.AST]
    column: str
    conditional: list[tuple[ConditionalGuard, str]]


class _Written(NamedTuple):
    """One statement a call runs, as :func:`_settle_conditional_guards` asks of it: the columns it writes, its
    row preconditions as spelled, and the tests it is known to have passed."""

    columns: frozenset[str]
    guards: frozenset[str]
    passed: frozenset[str]


def _every_written(
    func: ast.FunctionDef | ast.AsyncFunctionDef, call: ast.AST, held: dict[int, Optional[list[_Written]]]
) -> Optional[list[_Written]]:
    """Every statement *call* runs (:func:`sql.every_rendering`), ``None`` past the cap, held in *held* by call.

    The tests passed are the execute's and those on every piece's way here: only what the pieces show -- a piece
    left out passed no test this can see.
    """
    if id(call) not in held:
        drawn = sql.every_rendering(func, call.lineno)
        found: Optional[list[_Written]] = None
        if drawn is not None:
            found = []
            for one in drawn:
                tests = set(path_condition(call))
                for _start, _end, way in one.pieces:
                    tests.update(_tests_on_way(func, way) or ())
                found.extend(
                    _Written(
                        columns=frozenset(name.lower() for name in statement.columns),
                        guards=frozenset(statement.preconditions.values()),
                        passed=frozenset(tests),
                    )
                    for statement in sqltext.writes(sql.reading_text(one.text, one.holes))
                )
        held[id(call)] = found
    return held[id(call)]


def _guard_conditions(
    func: ast.FunctionDef | ast.AsyncFunctionDef, run: sql.Execution, statement: sqltext.SqlWrite
) -> dict[str, list[str]]:
    """When *run* carries each of *statement*'s row preconditions, by column.

    A term is a guard where the statement holds it and writes its column, so
    the tests are those over the pieces holding the term's text and the value
    the column is set to, and over the execute: ``updateJobStatus`` appends
    ``AND NOT jobStatus=:ngStatus`` under ``if jobStatus == "transferring"``;
    ``killJob``'s guard is in a text written once, run by the one execute that
    is not a force kill.  The pieces say where each stretch was written
    (:attr:`sql.Execution.pieces`).  A column is left out where a stretch has no
    piece, or a piece was written under a test of another function, whose
    names say nothing here.
    """
    if not run.pieces:
        return {}
    found: dict[str, list[str]] = {}
    for column, spans in statement.precondition_spans.items():
        supplied = statement.columns.get(column)
        end = sqltext.supplied_at(run.reading, statement.kind, column, supplied) if supplied is not None else None
        if end is None:
            continue
        tests = list(path_condition(run.call))
        for start, stop in [*spans, (end - 1, end)]:
            ways = [way for begin, finish, way in run.pieces if begin < stop and start < finish]
            said = [_tests_on_way(func, way) for way in ways]
            if not ways or any(one is None for one in said):
                break
            for one in said:
                tests.extend(one)
        else:
            found[column] = list(dict.fromkeys(tests))
    return found


def _tests_on_way(func: ast.FunctionDef | ast.AsyncFunctionDef, way: tuple) -> Optional[list[str]]:
    """The tests *way* passes in *func* -- where the stretch was written and each name it was put into on its way.

    ``None`` where a statement of another function on the way sits under a test.
    """
    tests: list[str] = []
    for node in way:
        if not isinstance(node, ast.stmt):
            continue
        if enclosing_function(node) is not func:
            if path_condition(node):
                return None
            continue
        tests.extend(path_condition(node))
    return tests


def _settle_conditional_guards(carried: Iterable[tuple[Branch, list[_Carrier]]]) -> None:
    """Say on each branch the guards only some statements writing its value carry, where it can be said when.

    A guard holds "when C" only if no statement writing the value without it
    passes C, asked of every statement each call runs, not of the covering
    set: ``getEventRanges`` narrows its jumbo subquery for one limit and not
    for the other, and the covering set held only one of the two.  A statement
    known to have passed a test contradicting C does not pass C.  One run by
    the same execute does not either, unless the tests its pieces show passed
    include all of C: C is where the guard's text was written and passed on
    its way, so without the guard C did not hold -- except where the guard's
    text is there and something appended inside it made another term of it.
    One run by another execute, with nothing contradicting C, can run under
    C: ``ppEventServiceJob`` writes ``ready`` by a guarded ``UPDATE`` and by
    two ``INSERT ... SELECT`` that guard nothing, one after the other.  Past
    the cap the same execute cannot be asked, and nothing is said.  A guard
    every statement gives is in ``row_precondition`` already, and one with no
    test would read as always.
    """
    held: dict[int, Optional[list[_Written]]] = {}
    for branch, carriers in carried:
        said: list[ConditionalGuard] = []
        calls = {(id(carrier.call), carrier.column.lower()): carrier for carrier in carriers if carrier.call is not None}
        for carrier in carriers:
            for guard, spelled in carrier.conditional:
                if not guard.when or guard.term in branch.row_precondition or guard in said:
                    continue
                if all(
                    _none_without(guard, spelled, other, other.call is carrier.call, held)
                    for other in calls.values()
                ):
                    said.append(guard)
        branch.row_precondition_when = sorted(said, key=lambda guard: (guard.term, guard.when))


def _none_without(
    guard: ConditionalGuard, spelled: str, other: _Carrier, same: bool, held: dict[int, Optional[list[_Written]]]
) -> bool:
    """Whether no statement *other*'s call runs writes the value without *guard* (spelled *spelled*) and can pass
    its tests -- see :func:`_settle_conditional_guards`."""
    written = _every_written(other.func, other.call, held)
    if written is None:
        return not same and pathcond.exclusive(guard.when, path_condition(other.call))
    for one in written:
        if other.column.lower() not in one.columns or spelled in one.guards:
            continue
        if pathcond.exclusive(guard.when, sorted(one.passed)):
            continue
        if same and not set(guard.when) <= one.passed:
            continue
        return False
    return True


class EntityUse(NamedTuple):
    """Where one kind of row is read, made, moved and removed."""

    tables: set[str]
    read_by: set[str]
    created_by: set[str]
    updated_by: set[str]
    deleted_by: set[str]


class ValueUse(NamedTuple):
    """Who asks for a value, and who acts on rows that already hold it.

    One ``WHERE`` clause, two questions, and they part company here.  *Does
    anything move a row out of this state?* is answered by ``UPDATE ... SET
    status=:new WHERE status=:old`` as squarely as by a query, so both verbs
    count towards it.  *Who has to pick this row up, so who to ask why they did
    not?* is not: an update is the picking up, not a chance to have missed it.

    Pooling them said an update was a query for a fifth of the corpus -- 99 of
    524 ``(function, subject, value)`` claims came from nothing but a write's
    predicate.  The cost was not only wording: one hop out of a junction opens
    a descent on whatever the called helper selects, so a writer counted as a
    reader sent the walk forward while it said it was stepping down.
    """

    selected_by: set[str]
    updated_by: set[str]


class ReadSide(NamedTuple):
    """What the predicates and the row sources of the corpus's statements say.

    Two questions about the same statements, answered in one walk because the
    walk is the cost: which *column values* something asks for, and whose
    *rows* it asks for at all.  Keeping them apart as two passes would also
    have written the table-to-class join twice.
    """

    #: ``{subject: {value: ValueUse}}`` -- the values a predicate names, with
    #: the verb that brought the table in kept beside each function.
    values: dict[str, dict[str, ValueUse]]
    #: ``{spec class: EntityUse}``
    entities: dict[str, EntityUse]
    #: ``{owner: sets of kinds of row one of its queries reads together}``.
    #: The sound form of a join between two entities: co-residence in a
    #: function is not a relation -- 112 functions read more than one kind
    #: of row somewhere, and 51 read two in one statement -- but a single
    #: ``FROM`` list is the corpus stating the relation itself.
    joins: dict[str, set[frozenset[str]]]
    #: ``{owner: statements whose table reference is still a placeholder}``.
    #: Counted rather than left out: a statement whose table nobody could read
    #: is one whose rows belong to nothing, and an unreadable name looks
    #: exactly like a table that holds no spec once both end up in one list.
    unreadable: dict[str, int]
    #: ``{(owner, line): ReadStatement}`` -- the same records as ``values``,
    #: kept per statement instead of per function.  The walk already tells the
    #: statements apart, because ``(sql, window)`` is what it deduplicates on;
    #: until now ``record`` was handed the owner and the position was dropped,
    #: which made three queries in ``copyArchive.main`` one answer.
    statements: dict[tuple[str, int], "ReadStatement"]
    #: What the build watches now that every caller is read: the function with
    #: the most routes into it, ``(owner, routes)``, and the seconds spent
    #: settling bind values from their callers.  For the build's report.
    watch: dict[str, object] = {}


class ReadStatement(NamedTuple):
    """One run of one statement, and every value its predicates ask rows by.

    Per run, not per text: the forwarded form pairs one statement with several
    varmaps, and ``window`` is what tells those runs apart -- so two runs of the
    same text in one function are two of these, which is the distinction the
    whole record exists to keep.
    """

    owner: str
    package: str
    file: str
    blob_sha: str
    line: int
    #: The lines this run's binds are filled within.  Read from the code for
    #: both call forms -- see :func:`~bamboo.codemap.panda.sql.binding_window`
    #: -- and ``None`` only where the call names no varmap to read.
    window: Optional[tuple[int, int]]
    #: ``{(subject, value)}`` the predicates name, for the read verb only.
    selects: set[tuple[str, str]]
    #: ``{(subject, value)}`` the statement asks rows *not* to hold --
    #: ``<>``, ``!=``, ``NOT IN``.
    excludes: set[tuple[str, str]]
    #: ``{(subject, value)}`` a subquery of the statement checks rows by --
    #: ``EXISTS (SELECT 1 ... WHERE jobStatus=:st_merging)``.
    consults: set[tuple[str, str]]
    #: Subjects the statement asks rows by with a value chosen at run time --
    #: ``jediTaskID=:jediTaskID``, or a bind nothing here settles.
    keys: set[str]
    #: ``ORDER BY`` clauses, one per way the statement is built.
    order_by: set[str]
    #: How many rows come back at most: ``<= 10``, ``<= :njobs``.
    limits: set[str]
    tables: set[str]
    #: What the ``WHERE`` asks of a row beside those values, rendered, with
    #: ``@<line>`` on the binds a walk can be seeded at.
    conditions: list[str]
    #: ``[first, last]`` -- the stretch of the function this statement owns.
    #: Filled once the function's other runs are known, so it arrives by
    #: ``_replace`` rather than at construction.
    region: Optional[tuple[int, int]] = None
    #: ``"continue@390"`` / ``"no-rows:return@263"`` -- where a row this
    #: statement asked for stops being followed.  Read inside ``region``, so
    #: it arrives by ``_replace`` too.  A tuple, because a NamedTuple's
    #: default is shared by every instance that does not override it.
    exits: tuple[str, ...] = ()
    #: ``(first, last)`` -- the stretch where this run's rows are worked
    #: through, which is the loop over them and can outlive ``region``.
    #: ``None`` where nothing loops over them.
    rows_handled: Optional[tuple[int, int]] = None
    #: ``"jobStatus IN ({}) -- the WHERE names its bind variables through an
    #: interpolation"``.  What this run was read only in part on, from
    #: :attr:`sql.Execution.unresolved`, so that a consumer offered the
    #: statement is offered the expression to open next rather than a
    #: ``conditions`` list that is silently short.
    unresolved: tuple[str, ...] = ()
    #: What the statement reads on each route of calls -- see ``place_routes``.
    call_readings: tuple[CallReading, ...] = ()
    #: Whether the routes are every way the statement is run from the corpus,
    #: and where the list may stop short -- :attr:`ReadSiteNode.callers_complete`.
    callers_complete: bool = False
    callers_stop: tuple[str, ...] = ()
    #: ``{(subject, value)}`` no row this statement returns holds, however it
    #: is built -- the part of ``excludes`` that holds of every row.  Known
    #: once every run of the statement is, so it arrives by ``_replace``.
    refuses: frozenset[tuple[str, str]] = frozenset()
    #: ``{(subject, value)}`` such that every row this statement returns holds
    #: one of the subject's values here, however it is built -- the part of
    #: ``selects`` that holds of every row.  Arrives by ``_replace`` as
    #: ``refuses`` does.
    restricts: frozenset[tuple[str, str]] = frozenset()
    #: Every term the fields hold on some builds only -- see
    #: :attr:`ReadSiteNode.added_under`.  Known once every run is, so it
    #: arrives by ``_replace`` too.
    added_under: tuple[AddedTerm, ...] = ()
    #: Each ``conditions`` entry and ``added_under`` conditions term as a
    #: tree -- :attr:`ReadSiteNode.condition_trees`.  Arrives by ``_replace``.
    condition_trees: tuple[ConditionReading, ...] = ()
    #: The prints of this statement's text and map -- :attr:`ReadSiteNode.printed_by`.
    printed_by: tuple[PrintedBy, ...] = ()
    #: Each blanked order item and what builds it -- :attr:`ReadSiteNode.order_holes`.
    order_holes: tuple[OrderHole, ...] = ()
    #: Each equality joining two of its sources, every run's (:func:`sqltext.equalities`), as
    #: ``(a, b, a_source, b_source, a_attr, b_attr)`` -- :attr:`ReadSiteNode.joins`.  A set made with the record.
    joins: Optional[set[tuple[str, str, str, str, str, str]]] = None
    #: What the statement returns -- :attr:`ReadSiteNode.result`.  Read off
    #: every run, so it arrives by ``_replace``.
    result: str = ""


class _Hop(NamedTuple):
    """A function a statement's text passes through: used there at *at*, handed in by *call* from the next."""

    func: ast.FunctionDef | ast.AsyncFunctionDef
    at: ast.AST
    call: Optional[ast.Call] = None


class _Leaf(NamedTuple):
    """A fragment of an assembled statement: where it lies in the text, what appends it, and the hops above it."""

    start: int
    end: int
    node: ast.stmt
    hops: tuple[_Hop, ...]
    #: Whether every run of its variant appends it: not cut out, here or in
    #: any name it was appended through.
    kept: bool = True


class _Build(NamedTuple):
    """A run's statement, and what every run of it holds."""

    #: The run's text: what :attr:`leaves` index.
    variant: str
    #: The run's own text with what some paths append cut out.
    text: str
    #: Its fragments; empty where the variant was not read fragment by fragment.
    leaves: list[_Leaf]


def _run_builds(func: ast.FunctionDef | ast.AsyncFunctionDef, run: sql.Execution) -> Optional[list[_Build]]:
    """The build of *run*, read fragment by fragment: one, the statement as it stands, each stretch from the way
    it came (:attr:`sql.Execution.pieces`).

    Kept on *func* per reading context (:func:`sql.reading_context`): every
    field read off the builds asks for them -- ``refuses`` and ``restricts``,
    ``conditions``, ``added_under`` -- at every call site handing constants
    as well, and reading them again was most of the read side's time.
    """
    key = (sql.reading_context(), run.variable, run.head, run.sql, id(run.call), run.pieces, run.holes)
    held = getattr(func, "_run_builds_held", None)
    if held is None or held[0] != key[0][0]:
        held = (key[0][0], {})
        func._run_builds_held = held  # type: ignore[union-attr]
    if key not in held[1]:
        held[1][key] = _read_builds(func, run)
    return held[1][key]


def _read_builds(func: ast.FunctionDef | ast.AsyncFunctionDef, run: sql.Execution) -> Optional[list[_Build]]:
    """:func:`_run_builds`, read: the statement is one the paths build as it stands -- every stretch of it is
    appended on every path that builds it, so nothing is cut -- and each stretch says the way it came."""
    return [_Build(run.sql, run.reading, _evaluated_leaves(func, run))]


def _evaluated_leaves(func: ast.FunctionDef | ast.AsyncFunctionDef, run: sql.Execution) -> list[_Leaf]:
    """*run*'s stretches as fragments, from the way the evaluator says each came (:attr:`sql.Execution.pieces`).

    The fragment is the statement that wrote the stretch -- a value handed to a
    parameter, written by no statement here, is placed at the first statement
    that put it into a name -- and the hops are the rest of the way, out to
    the ``execute``: each statement that put it into a name, and a callee's
    ``return`` with the call it came back from.  ``getJobs``'s ``WHERE`` terms
    are appended in ``construct_where_clause``, a function the call runs: read
    by name instead, the way stopped at ``sqlP += sql_where_clause`` and every
    test above a term was said to be in ``getJobs``.
    """
    leaves: list[_Leaf] = []
    for start, end, way in run.pieces:
        nodes = [node for node in way if isinstance(node, ast.AST)]
        if not nodes:
            continue
        hops = [_Hop(func, run.call)]
        for node in reversed(nodes[1:]):
            if isinstance(node, ast.Call):
                hops[-1] = hops[-1]._replace(call=node)
            elif node is not hops[-1].at:
                hops.append(_Hop(enclosing_function(node) or func, node))
        leaves.append(_Leaf(start, end, _statement_of(nodes[0]), tuple(hops)))
    return leaves


def _parameters(func: ast.AST) -> list[str]:
    """*func*'s parameter names; none for a module's own body (a call there, :class:`CallSite`)."""
    args = getattr(func, "args", None)
    if args is None:
        return []
    return [one.arg for one in [*args.posonlyargs, *args.args, *args.kwonlyargs]]


def _nested_in(func: ast.AST) -> Optional[ast.FunctionDef | ast.AsyncFunctionDef]:
    node = getattr(func, "parent", None)
    while node is not None and not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        node = getattr(node, "parent", None)
    return node


def _statement_of(node: ast.AST) -> ast.AST:
    """*node*, or the statement holding it -- a ``return``'s value is its ``return``."""
    if isinstance(node, ast.stmt):
        return node
    return next((one for one in ancestors(node) if isinstance(one, ast.stmt)), node)


def _guards_between(node: ast.AST, below: ast.AST) -> list[tuple[str, str, int, Optional[ast.expr]]]:
    """``(kind, test, line, expression)`` for each ``if``, loop and handler above *node* and not above *below*, outermost first.

    *below* a function: every one above *node* in it.  An ``else`` -- and a
    later branch of an ``elif`` chain, through the ``else`` it hangs from --
    is the test not holding.
    """
    around = {id(one) for one in ancestors(below)} | {id(below)}
    found: list[tuple[str, str, int, Optional[ast.expr]]] = []
    previous: ast.AST = node
    for ancestor in ancestors(node):
        if id(ancestor) in around:
            break
        if isinstance(ancestor, ast.If):
            if any(previous is one for one in ancestor.body):
                found.append(("if", ast.unparse(ancestor.test), ancestor.lineno, ancestor.test))
            elif any(previous is one for one in ancestor.orelse):
                found.append(("else", f"not ({ast.unparse(ancestor.test)})", ancestor.lineno, ancestor.test))
        elif isinstance(ancestor, (ast.For, ast.AsyncFor, ast.While)):
            found.append(("loop", f"a loop at {ancestor.lineno} runs", ancestor.lineno, None))
        elif isinstance(ancestor, ast.ExceptHandler):
            found.append(("except", f"an except at {ancestor.lineno} runs", ancestor.lineno, None))
        previous = ancestor
    found.reverse()
    return found


def _stretches(leaf: _Leaf) -> list[tuple[int, ast.AST, ast.AST]]:
    """``(hop, from, to)`` -- where along the text's way from *leaf* to the ``execute`` a test can choose it.

    In the fragment's own function, between it and where the text is used;
    in a function the text was returned from, above that ``return``; in the
    function a call handed it to, between the call and where the text is used.
    """
    hops = leaf.hops
    last = len(hops) - 1
    found: list[tuple[int, ast.AST, ast.AST]] = [(last, leaf.node, hops[last].at)]
    for index in range(last - 1, -1, -1):
        hop, inner = hops[index], hops[index + 1]
        if hop.call is not None:
            found.append((index + 1, _statement_of(inner.at), inner.func))
            found.append((index, hop.call, hop.at))
        else:
            found.append((index, inner.at, hop.at))
    return found


def _locals_bound(func: ast.AST, name: str) -> list[str]:
    """``<expression>@<line>`` for each plain assignment of *name* in *func*."""
    found: list[str] = []
    for node in walk(func):
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None and enclosing_function(node) is func:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(
                isinstance(one, ast.Name) and one.id == name
                for target in targets
                for one in ([target] if isinstance(target, ast.Name) else getattr(target, "elts", []))
            ):
                found.append(f"{_clip(ast.unparse(node.value))}@{node.lineno}")
    return found[:4]


def _clip(text: str, width: int = 120) -> str:
    return text if len(text) <= width else text[: width - 3] + "..."


def _rebound_at(func: ast.AST, name: str) -> list[int]:
    """Lines where *func* itself -- not a function nested in it -- binds *name* again."""
    lines: list[int] = []
    for node in walk(func):
        if node is func or enclosing_function(node) is not func:
            continue
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign, ast.For, ast.AsyncFor, ast.NamedExpr)):
            targets = [node.target]
        elif isinstance(node, ast.withitem):
            targets = [node.optional_vars] if node.optional_vars is not None else []
        elif isinstance(node, ast.ExceptHandler):
            if node.name == name:
                lines.append(node.lineno)
            continue
        else:
            continue
        if any(isinstance(one, ast.Name) and one.id == name for target in targets for one in walk(target)):
            lines.append(getattr(node, "lineno", 0) or getattr(getattr(node, "context_expr", None), "lineno", 0))
    return sorted(set(lines))


def _guard_read(name: str, hop: int, hops: tuple[_Hop, ...], file_of) -> GuardRead:
    """Where *name*, read by a test in ``hops[hop]``'s function, gets its value: out to the read site's parameter.

    Only unchanged: where a function on the way binds the name again, the
    value at the test is not the argument, and ``at_site`` stays empty with
    ``via`` saying where.
    """
    func = hops[hop].func
    site = hops[0].func
    read = GuardRead(name=name)
    current, at, changed = name, hop, False

    def rebound(where: ast.AST, one: str) -> bool:
        lines = _rebound_at(where, one)
        if lines:
            read.via.append(
                f"{getattr(where, 'name', '?')}.{one} is reassigned at "
                + ", ".join(f"{file_of(where)}:{line}" for line in lines)
            )
        return bool(lines)

    while True:
        if current not in _parameters(func):
            outer = _nested_in(func)
            if outer is not None and not _rebound_at(func, current) and current in _parameters(outer):
                read.via.append(f"{func.name}: {current} is {outer.name}'s")
                func = outer
                continue
            read.bound = _locals_bound(func, current)
            return read
        if not read.parameter:
            read.parameter = f"{func.name}.{current}"
        changed = rebound(func, current) or changed
        if func is site:
            if not changed:
                read.at_site = f"{site.name}.{current}"
            return read
        # The call into *func*: the nearest hop above whose function is another.
        while at > 0 and hops[at - 1].func is func:
            at -= 1
        if at == 0 or hops[at - 1].call is None:
            read.via.append(f"{func.name}.{current}: no call into it on the way")
            return read
        call = hops[at - 1].call
        handed = next((one.value for one in call.keywords if one.arg == current), None)
        if handed is None:
            names = _parameters(func)
            position = names.index(current) - (1 if names and names[0] in ("self", "cls") else 0)
            if 0 <= position < len(call.args):
                handed = call.args[position]
        if handed is None:
            read.via.append(f"{func.name}.{current}: not passed at {call.lineno}, so its default")
            return read
        read.via.append(f"{func.name}.{current} <- {_clip(ast.unparse(handed))}@{call.lineno}")
        if not isinstance(handed, ast.Name):
            return read
        at -= 1
        func, current = hops[at].func, handed.id


def _place(leaf: _Leaf, term: str, file_of, decides: Optional[bool] = None) -> TermPlace:
    """Where *leaf* is appended, in which *term*, and every test along its way to the ``execute``."""
    tests: list[TermGuard] = []
    for hop, node, below in reversed(_stretches(leaf)):
        func = leaf.hops[hop].func
        for kind, test, line, expression in _guards_between(node, below):
            names = sorted({one.id for one in walk(expression) if isinstance(one, ast.Name)}) if expression is not None else []
            tests.append(
                TermGuard(
                    kind=kind,
                    test=test,
                    at=f"{file_of(func)}:{line}",
                    function=func.name,
                    reads=[_guard_read(name, hop, leaf.hops, file_of) for name in names],
                )
            )
    return TermPlace(
        fragment=f"{file_of(leaf.hops[-1].func)}:{getattr(leaf.node, 'lineno', 0)}",
        term=term,
        decides=decides,
        tests=tests,
    )


def _placed(leaves: list[_Leaf], term: str, file_of, decides: Optional[bool] = None) -> TermPlace:
    """Where a term whose text is made of *leaves* is appended: under every test above any of them.

    The statement holds the term where it holds every one, so the tests add
    up.  The place is named by the fragment with the most tests -- the one a
    branch chooses: ``jobStatus IN ('activated', 'sent', 'starting')`` is
    the template's ``jobStatus IN `` and the list written under ``if
    site_workflow ...``.
    """
    each = [_place(leaf, term, file_of, decides) for leaf in leaves]
    tests: list[TermGuard] = []
    for place in each:
        tests.extend(one for one in place.tests if one not in tests)
    lead = max(each, key=lambda place: len(place.tests))
    return TermPlace(fragment=lead.fragment, term=term, decides=decides, tests=tests)


def _in_loop(node: ast.AST, loops: set[int]) -> bool:
    """Whether *node* is one of *loops* (by ``id``) or in one's body."""
    return id(node) in loops or any(id(one) in loops for one in ancestors(node))


def _tests_of(place: TermPlace) -> tuple[tuple[str, str, str], ...]:
    """*place*'s tests, as what tells two places of one fragment apart."""
    return tuple((one.kind, one.test, one.at) for one in place.tests)


def _condition_seeds(
    func: ast.FunctionDef | ast.AsyncFunctionDef, run: sql.Execution, line: int, found
) -> dict[str, list[int]]:
    """Where each bind *found* conditions name is filled for *run* -- the ``@<line>`` :func:`_render_condition` writes."""
    window = run.window
    if window is None and run.varmap:
        window = sql.binding_window(func, run.varmap, line)
    seeds: dict[str, list[int]] = {}
    for condition in found:
        for key in condition.binds:
            if key in seeds:
                continue
            at_lines = sorted(
                {bind.site.lineno for bind in sql.bound_values(func, run.varmap or "", key, window, at=run)}
            )
            if at_lines:
                seeds[key] = at_lines
    return seeds


def _trail(leaf: _Leaf) -> list[ast.AST]:
    """Where a fragment's text is on its way to the ``execute``: appended, then each use and call above that."""
    found: list[ast.AST] = [leaf.node]
    for hop in leaf.hops:
        found.append(hop.at)
        if hop.call is not None:
            found.append(hop.call)
    return found


def _along(nodes: Iterable[ast.AST]) -> dict[int, list[ast.AST]]:
    """*nodes* by the function each is in -- what :func:`_meets` compares a node with."""
    found: dict[int, list[ast.AST]] = {}
    for node in nodes:
        found.setdefault(id(enclosing_function(node)), []).append(node)
    return found


def _meets(node: ast.AST, along: dict[int, list[ast.AST]]) -> bool:
    """Whether one run can pass *node* and every node of *along* in its function.

    :func:`pathcond.compatible` read as the function runs, the question asked
    of a fragment and the call it might reach: the two arms
    of one ``if`` never meet, and neither do two ``if``\\ s testing the same
    thing with no name of the test given a new value between them --
    ``getJobStatisticsForExtIF`` picks its statement under one ``if
    source_type == "analysis"`` and fills its binds under another.  Only in
    one function: a test in ``getJobs`` says nothing about one in
    ``construct_where_clause``.
    """
    return all(compatible(node, one, current=True) for one in along.get(id(enclosing_function(node)), ()))


def _squeezed(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def _names(piece: str, binds: Iterable[str], words: Iterable[str]) -> bool:
    """Whether a fragment's text names a term: every one of its binds, else every one of its words."""
    binds = [bind for bind in binds if bind]
    if binds:
        return all(re.search(re.escape(bind) + r"(?!\w)", piece) for bind in binds)
    words = [word for word in words if word]
    return bool(words) and all(re.search(r"(?<!\w)" + re.escape(word) + r"(?!\w)", piece, re.I) for word in words)


def _outright_terms(
    func: ast.FunctionDef | ast.AsyncFunctionDef, run: sql.Execution
) -> tuple[tuple["sqltext.Term", ...], ...]:
    """The terms every row of *run* meets, one tuple per way a row comes back in each statement the call runs.

    Read on every statement the call runs and not on the run's text alone: a
    fragment appending ``OR ...`` takes rows back that the shorter text leaves
    out, and a term a fragment appends is met only where the fragment is.  The
    caller combines the ways -- a value refused is refused in every one, a
    subject restricted is restricted in every one, to any of their values.
    """
    whole = sqltext.outright(run.reading)
    # Before the statements are asked for: they are the expensive half, and a
    # text nothing decides has no statement that does.
    if not any(whole):
        return whole
    # What every run asks is what every statement of the call asks, not every
    # statement of the covering set (:func:`sql.every_statement`).
    every = sql.every_statement(func, run.call.lineno)
    floors = every if every is not None else [run.reading]
    return tuple(way for floor in floors for way in sqltext.outright(floor))


def _render_condition(condition: "sqltext.Condition", seeds: dict[str, list[int]]) -> str:
    """One condition as the map stores it, with its walkable binds placed.

    ``@<line>`` marks a bind an assignment in this run's window fills, so a
    consumer can seed a walk there without re-reading the source; a bind with
    no assignment to point at, and every term that names no bind at all, is
    written plain -- which is the map saying the term is real and not
    walkable, rather than saying nothing.

    Branches are joined by ``OR`` because the list of conditions is the
    conjunction.  A branch holding more than one term is parenthesised: SQL
    binds ``AND`` tighter than ``OR`` and so does this, but a reader should
    not have to know that to see which terms go together.
    """
    parts: list[str] = []
    for branch in condition.branches:
        text = " AND ".join(branch.terms)
        for key in sorted(set(branch.binds), key=len, reverse=True):
            lines = seeds.get(key)
            if lines:
                placed = ",".join(str(line) for line in lines)
                text = re.sub(re.escape(key) + r"\b", f"{key}@{placed}", text)
        parts.append(f"({text})" if len(branch.terms) > 1 and len(condition.branches) > 1 else text)
    return " OR ".join(parts)


def _text_run(
    func: ast.FunctionDef | ast.AsyncFunctionDef, run: sql.Execution, drawn: "sqleval.Rendered"
) -> sql.Execution:
    """*run* holding *drawn*'s text, holes and pieces: one object per text, so what a reader holds by the run is read
    once -- a site's conditions are folded more than once, and ``bound_sites`` keeps what it read by the run."""
    store = getattr(func, "_sqlwrite_text_runs", None)
    if store is None:
        store = func._sqlwrite_text_runs = {}  # type: ignore[union-attr]
    held = store.get((id(run), id(drawn)))
    if held is None or held[0] is not run or held[1] is not drawn:
        held = store[(id(run), id(drawn))] = (run, drawn, run._replace(sql=drawn.text, holes=drawn.holes, pieces=drawn.pieces))
    return held[2]


def _folded(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    line: int,
    variants: list,
    tree_of: Optional[Callable[[sql.Execution, "sqltext.Twin"], ConditionNode]] = None,
) -> list[tuple[str, Optional[ConditionNode]]]:
    """What every one of a statement's *variants* requires, and what they require between them.

    The variants are folded as what they are, a disjunction: ``(A and B) or
    (A and C)`` is ``A``, required outright, and ``B or C`` as one condition.
    A variant that requires nothing beyond the common part makes that
    disjunction free, and then it is not recorded at all.

    **The variants are every statement the call runs** (:func:`sql.every_statement`),
    each holding what is appended on some paths only where that path does:
    ``getDatasetsToFeedContents_JEDI`` appends ``tabT.lockedTime IS NULL OR
    tabT.lockedTime<:lockTimeLimit`` under ``if not force_read:``, and the
    statement a run with ``force_read`` runs never asks it, so it is one side
    of a disjunction, not required.  Past the statements that can be counted,
    only a covering set: its statements are not every way a run can go, so
    each is read as it is and no disjunction of theirs is said.

    Each entry comes with its tree where *tree_of* reads one off a run's
    :func:`sqltext.condition_twins` -- the values of the same entry in two
    runs merged, and the disjunction folded here an ``any`` of the ``all``
    of each variant's remainder.
    """
    rendered: list[list[tuple[str, Optional[ConditionNode]]]] = []
    # Run, the statements are the call's every statement where there are few
    # enough (:func:`sql.every_statement`); past that only a covering set,
    # whose statements are not every way a run can go, so no disjunction of
    # theirs is said.
    open_ended = False
    # ``(run, its texts)``: each statement of the call read as a run of its
    # own text and pieces, since a bind is the text's (:func:`sql.bound_values`):
    # ``getJobStatisticsForExtIF`` fills ``:prodSourceLabel1`` with ``user`` for
    # the analysis text and ``managed`` for the production one.
    texts_of: list[tuple[sql.Execution, list[str]]] = []
    if variants:
        drawn = sql.every_rendering(func, line)
        if drawn is None:
            open_ended = True
            texts_of = [(run, [run.reading]) for run in variants]
        else:
            texts_of = [(_text_run(func, variants[0], one), [sql.reading_text(one.text, one.holes)]) for one in drawn]
    for run, texts in texts_of:
        window = run.window
        if window is None and run.varmap:
            window = sql.binding_window(func, run.varmap, line)
        for text in texts:
            found = list(zip(sqltext.conditions(text), sqltext.condition_twins(text), strict=True))
            seeds: dict[str, list[int]] = {}
            for condition, _twin in found:
                for key in condition.binds:
                    if key in seeds:
                        continue
                    at_lines = sorted(
                        {
                            bind.site.lineno
                            for bind in sql.bound_values(func, run.varmap or "", key, window, at=run)
                        }
                    )
                    if at_lines:
                        seeds[key] = at_lines
            rendered.append(
                [(_render_condition(c, seeds), tree_of(run, twin) if tree_of is not None else None) for c, twin in found]
            )
    trees: dict[str, Optional[ConditionNode]] = {}
    for one in rendered:
        for entry, tree in one:
            trees[entry] = _merged_tree(trees.get(entry), tree)
    entries = [[entry for entry, _tree in one] for one in rendered]
    common = set.intersection(*(set(one) for one in entries))
    folded = [(entry, trees[entry]) for entry in sorted(common)]
    remainders: list[tuple[str, ...]] = []
    for one in entries:
        shape = tuple(entry for entry in one if entry not in common)
        if shape not in remainders:
            remainders.append(shape)
    # ``A or (A and B)`` is ``A``: a way that asks all another asks and more
    # adds nothing to the disjunction.  ``lockJobsForReassign``'s 384
    # statements each add their own optional terms to one of two time tests.
    remainders = [one for one in remainders if not any(set(other) < set(one) for other in remainders)]
    if len(remainders) == 1:
        folded.extend((entry, trees[entry]) for entry in remainders[0])
    elif all(remainders) and not open_ended:
        tree = None
        if tree_of is not None:
            tree = ConditionNode(
                node="any",
                items=[ConditionNode(node="all", items=[trees[entry] for entry in shape]) for shape in remainders],
            )
        folded.append(
            (" OR ".join(f"({' AND '.join(shape)})" if len(shape) > 1 else shape[0] for shape in remainders), tree)
        )
    return folded


def _merged_tree(one: Optional[ConditionNode], other: Optional[ConditionNode]) -> Optional[ConditionNode]:
    """One entry's tree as two runs read it: the values either reads, settled where both are."""
    if one is None or other is None:
        return one if other is None else other
    if (one.node, one.text, len(one.items)) != (other.node, other.text, len(other.items)):
        return one
    return one.model_copy(
        update={
            "items": [_merged_tree(a, b) for a, b in zip(one.items, other.items, strict=True)],
            "values": one.values + [value for value in other.values if value not in one.values],
            "settled": one.settled and other.settled,
        }
    )


#: ``:key@412,415`` -- where a condition's bind is filled, which two readings
#: of one statement may place differently without reading it differently.
_SEEDED = re.compile(r"(:\w+)@[\d,]+")


#: The logger methods a print of a statement is made with.
_PRINTS = frozenset({"debug", "info", "warning", "error", "critical", "exception"})


def _message_parts(node: ast.expr) -> list[tuple[str, object]]:
    """A message as it is assembled: ``("text", str)`` and ``("expr", node)`` in order.

    ``sqlP + comment + str(getValMap)`` is three expressions, ``f"running
    sql: {sqlGD} {varMap}"`` text, an expression, text and an expression.
    ``str(x)`` and an f-string's ``{x}`` both print *x*, so both are *x*.
    """
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _message_parts(node.left) + _message_parts(node.right)
    if isinstance(node, ast.JoinedStr):
        parts: list[tuple[str, object]] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(("text", value.value))
            elif isinstance(value, ast.FormattedValue) and value.format_spec is None:
                parts.extend(_message_parts(value.value))
            else:
                parts.append(("expr", value))
        return parts
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [("text", node.value)]
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "str"
        and len(node.args) == 1
        and not node.keywords
    ):
        return [("expr", node.args[0])]
    return [("expr", node)]


def _comment_name(func: ast.AST, part: tuple[str, object], at: ast.AST) -> bool:
    """Whether *part* is a name holding only a ``/* ... */`` comment where *at* reads it."""
    if part[0] != "expr" or not isinstance(part[1], ast.Name):
        return False
    written = reaching_definitions(func, part[1].id, at)
    return bool(written) and all(
        isinstance(one, ast.Constant) and isinstance(one.value, str) and one.value.strip().startswith("/*")
        for one in written
    )


def _changed_between(func: ast.AST, names: set[str], low: int, high: int) -> bool:
    """Whether a statement strictly between lines *low* and *high* rebinds or changes one of *names*."""
    for node in walk(func):
        line = getattr(node, "lineno", 0)
        if not low < line < high or enclosing_function(node) is not func:
            continue
        targets: list[ast.AST] = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign, ast.For, ast.AsyncFor)):
            targets = [node.target]
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in FILLERS + (
            "pop", "popitem", "clear", "remove", "discard"
        ):
            targets = [node.func.value]
        for target in targets:
            for one in walk(target):
                base = one
                while isinstance(base, (ast.Subscript, ast.Attribute)):
                    base = base.value
                if isinstance(base, ast.Name) and base.id in names:
                    return True
    return False


def _printed_by(func: ast.FunctionDef | ast.AsyncFunctionDef, run: sql.Execution) -> list[PrintedBy]:
    """The logging calls in *func* that print *run*'s statement text and then the map it is executed with.

    Only where the record can be cut without guessing: the print writes the
    executed text -- the execute's own pieces, or those less a trailing name
    holding a ``/* ... */`` comment -- then text, then the executed map as
    its last piece, and nothing between the print and the ``execute``
    rebinds or changes either.  A print of another statement or another
    map is not this statement's, however alike it reads.
    """
    if not run.call.args or not run.varmap:
        return []
    executed = _message_parts(run.call.args[0])
    if any(kind != "expr" for kind, _value in executed):
        return []
    spelled = [ast.unparse(value) for _kind, value in executed]
    execute = _statement_of(run.call)
    whole = len(spelled)
    shorter = whole - 1 if whole > 1 and _comment_name(func, executed[-1], execute) else None
    found: list[PrintedBy] = []
    for node in walk(func):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _PRINTS
            and len(node.args) == 1
            and enclosing_function(node) is func
        ):
            continue
        parts = _message_parts(node.args[0])
        if not parts or parts[-1][0] != "expr" or not (
            isinstance(parts[-1][1], ast.Name) and parts[-1][1].id == run.varmap
        ):
            continue
        said = [ast.unparse(value) if kind == "expr" else None for kind, value in parts[:-1]]
        for length, text in ((whole, "whole"), (shorter, "without comment")):
            if not length:
                continue
            starts = [
                at
                for at in range(len(said) - length + 1)
                if said[at : at + length] == spelled[:length]
                and all(kind == "text" for kind, _value in parts[at + length : -1])
            ]
            if not starts:
                continue
            at = starts[-1]
            lead = ""
            for kind, value in reversed(parts[:at]):
                if kind != "text":
                    break
                lead = str(value) + lead
            printed = _statement_of(node)
            low, high = sorted((printed.lineno, execute.lineno))
            names = {one.id for _kind, value in executed for one in walk(value) if isinstance(one, ast.Name)}
            if _changed_between(func, names | {run.varmap}, low, high):
                break
            found.append(
                PrintedBy(
                    line=printed.lineno,
                    where="before" if printed.lineno < execute.lineno else "after",
                    text=text,
                    lead=lead,
                    gap="".join(str(value) for _kind, value in parts[at + length : -1]),
                )
            )
            break
    return sorted(found, key=lambda one: one.line)


def _loop_said(loop: ast.AST, func: ast.AST, file: str) -> str:
    """``file:line for x in xs (xs = f()@line)``: a loop, and what it walks where that is a name."""
    said = f"{file}:{loop.lineno} for {ast.unparse(loop.target)} in {ast.unparse(loop.iter)}"
    if isinstance(loop.iter, ast.Name):
        written = [f"{ast.unparse(one)}@{site_of(one).lineno}" for one in reaching_definitions(func, loop.iter.id, loop)]
        if written:
            said += f" ({loop.iter.id} = {', '.join(written)})"
    return said


def _number(amount: float) -> str:
    """``800`` for 800.0, ``0.5`` for 0.5: a number as Python spells the value the source wrote."""
    return str(int(amount)) if float(amount).is_integer() else repr(amount)


def _bind_amount(side: "sqltext.Side", value: "values.Stated") -> ConditionValue:
    """A bind the statement compares a column with, filled with *value*."""
    if value.said:
        return ConditionValue(kind="bind", text=side.text, said=value.said)
    if value.constant:
        return ConditionValue(kind="bind", text=side.text, constant=value.constant)
    if value.base:
        return ConditionValue(kind="bind", text=side.text, base=value.base, offset_seconds=value.offset, needs=list(value.needs))
    if value.offset is None:
        return ConditionValue(kind="bind", text=side.text, needs=list(value.needs))
    return ConditionValue(kind="bind", text=side.text, constant=_number(value.offset))


def _clock_amount(side: "sqltext.Side", value: "values.Stated") -> ConditionValue:
    """A database clock plus a bind amount -- ``CURRENT_DATE-:days`` -- with the bind filled with *value*."""
    held = ConditionValue(kind="clock", text=side.text, base=side.clock, scale_seconds=side.scale_seconds)
    if value.said or value.constant or value.duration:
        return held.model_copy(update={"said": value.said or f"{side.binds[0]} holds {value.constant or 'a duration'}"})
    if value.base or value.offset is None:
        needs = list(value.needs) + ([value.base] if value.base and not values.is_clock(value.base) else [])
        return held.model_copy(update={"needs": needs} if needs else {"said": f"{side.binds[0]} holds a clock"})
    return held.model_copy(update={"offset_seconds": (side.scale_seconds or 0.0) * value.offset})


@dataclasses.dataclass
class _CallReading:
    """What the runs one call site makes of a statement read and ask by."""

    tables: set[str]
    selects: set[tuple[str, str]] = dataclasses.field(default_factory=set)
    excludes: set[tuple[str, str]] = dataclasses.field(default_factory=set)
    consults: set[tuple[str, str]] = dataclasses.field(default_factory=set)

    def add(self, other: "_CallReading") -> None:
        self.tables |= other.tables
        self.selects |= other.selects
        self.excludes |= other.excludes
        self.consults |= other.consults


def _on_route(via: tuple[str, ...], chain: tuple[str, ...], into: frozenset[str]) -> bool:
    """Whether a value handed through the calls *via* reaches the statement along the route *chain*.

    *into* is every call into the statement's function.  A value handed
    through none of them -- written in the function, or in a callee whose
    calls never leave it -- is on every route; one handed through one of them
    is on the routes starting there, as far as both go: ``copyArchive``'s
    status list at line 774 is that call's, not the one at 727.
    """
    at = next((index for index, call in enumerate(via) if call in into), None)
    if at is None:
        return True
    rest = via[at:]
    shared = min(len(rest), len(chain))
    return rest[:shared] == chain[:shared]


def _route_handed(target: Target, sites: tuple, calls: Calls) -> dict[str, "values.Stated"]:
    """What the route *sites* hands each of *target*'s parameters, read as an amount in the caller (:func:`values.stated`).

    A parameter the call does not pass holds the default the function
    states; one passed is read where the caller writes it, through the
    caller's own parameters as the rest of the route hands them.  A
    parameter the call hands with ``*args`` is left out: its value is the
    function's own parameter, ``needs``.
    """
    site = sites[0]
    passed = calls.filled(site, target)
    outer: Optional[dict[str, values.Stated]] = None
    found: dict[str, values.Stated] = {}
    for name, expression in calls.handed(site, target).items():
        if expression is None:
            continue
        if name not in passed:
            found[name] = values.stated(expression, target.func, expression)
            continue
        if outer is None:
            outer = {}
            if len(sites) > 1:
                owned = calls.types.method_class(site.file, site.func)
                outer = _route_handed(Target(site.file, site.func, owned[1] if owned else None), sites[1:], calls)
        found[name] = values.stated(expression, site.func, _statement_of(site.call), handed=outer)
    return found


def _check_as_statement(
    recorded: list[tuple[tuple[str, int], tuple[str, ...], CallReading]],
    statements: dict[tuple[str, int], "ReadStatement"],
) -> None:
    """Stop the build where a route recorded as reading as the statement does not read what the statement's fields say.

    A route marked ``as_statement`` carries no fields: a reader takes the
    statement's.  The mark is decided against the statement read the route's
    way (``route_reading``); the fields stored are filed another way, run by
    run (``record``, ``refused``, ``restricted``, ``place_conditions``).  A
    statement whose own fields name no value -- a call's value not known --
    beside a route that reads one would lose the route's value without a
    word.  So the two are compared on every field the mark stands for; the
    statement's condition trees are its conditions' (``added_under``'s are
    not a route's to say).
    """
    def pairs(found) -> list[str]:
        return sorted(f"{subject}={value}" for subject, value in found)

    def trees(readings, entries=None) -> dict[str, object]:
        return {
            one.entry: one.tree.model_dump() if one.tree is not None else None
            for one in readings
            if entries is None or one.entry in entries
        }

    wrong: list[str] = []
    for site, chain, read in recorded:
        at = statements.get(site)
        if at is None:
            continue
        stored = {
            "tables": sorted(at.tables), "selects": pairs(at.selects), "excludes": pairs(at.excludes),
            "consults": pairs(at.consults), "refuses": pairs(at.refuses), "restricts": pairs(at.restricts),
        }
        moved = [name for name, value in stored.items() if sorted(getattr(read, name)) != value]
        if trees(read.condition_trees) != trees(at.condition_trees, set(at.conditions)):
            moved.append("condition_trees")
        if moved:
            wrong.append(f"{site[0]}@{site[1]} via {list(chain)}: {', '.join(moved)}")
    if wrong:
        raise RuntimeError(
            f"{len(wrong)} route(s) recorded as reading as the statement read otherwise -- "
            + "; ".join(wrong[:5])
            + (f"; … {len(wrong) - 5} more" if len(wrong) > 5 else "")
        )


def _reads_alike(route: CallReading, statement: Optional[CallReading]) -> bool:
    """Whether a route reads what the statement does: every field :class:`CallReading` has, the route's own aside."""
    if statement is None:
        return False
    return all(
        getattr(route, name) == getattr(statement, name)
        for name in CallReading.model_fields
        if name not in CALL_READING_ROUTE
    )


def _restricted_both(one: dict[str, set[str]], other: dict[str, set[str]]) -> dict[str, set[str]]:
    """What two ways a row comes back both restrict: the subjects both do, to any value either asks."""
    return {subject: one[subject] | other[subject] for subject in one.keys() & other.keys()}


def _restricted_pairs(restricted: dict[str, set[str]]) -> set[tuple[str, str]]:
    return {(subject, value) for subject, values_of in restricted.items() for value in values_of}


READ_SLICE_NAME = "sql-read"


def read_coverage(modules: list[SourceModule]) -> list[CoverageStat]:
    """How many of the corpus's queries the read side actually reads.

    The denominator that did not exist.  ``sql-write`` has one, so a write the
    slice cannot read shows up as a ratio falling; the read side had none, and
    508 statements could stop being read without a single number moving.  That
    is P1-25's rule 6 -- a slice with no denominator passes silently even when
    it is missing entirely.

    **Counted by a different reading from the one it measures.**  The
    numerator comes from :func:`sql.executions`, which recognises the call
    forms; the denominator must not, or the measure would define its
    population as whatever the extractor already sees and could never report
    a form it does not know.  So candidates are found from the *statements* --
    every local a function assembles SQL into, whatever executes it -- and a
    query reachable only through a call form ``executions`` does not accept
    lands in the denominator and not the numerator, which is the whole point.

    Per file, like every other slice, so the build's lowest-coverage listing
    can name where the gap is.
    """
    found: list[CoverageStat] = []
    unexplained: list[Unexplained] = []
    for module in modules:
        candidates = explained = 0
        for func, _owner in functions_with_owner(module.tree):
            assembled = _assembled_queries(func)
            candidates += len(assembled)
            executions = sql.executions(func)
            run = {execution.sql for execution in executions}
            # In the spelling the runs use.  A ``.format`` field the call could
            # not fill is ``{}`` in the run and still ``{0}`` in the statement
            # assembled, and comparing them as written counted
            # ``getUserJobMetadata``'s query as one nothing runs.
            seen = {text for text in assembled if values.blank_fields(text) in run}
            # A field the call did fill is the table or the ordering in the
            # run, and the template still has the hole: ``sqlR.format(table)``
            # runs over jobsDefined4 and jobsActive4 and matched neither, which
            # listed thirteen queries the reader does read as ones it cannot
            # follow.  Matched as a template, against the runs of its own name
            # only: the holes fit anything, and ``getOriginalConsumers``' sqlJ
            # fits sqlC2's statement as well as its own.
            seen |= {
                text
                for text, name in assembled.items()
                if text not in seen
                and any(
                    execution.variable == name
                    and values.template_matches(values.blank_fields(text), execution.sql)
                    for execution in executions
                )
            }
            # Run under another name: the name's text reached a run through a
            # statement putting it into another (``sql_exe = (sql_jt + comment)
            # % table``, a loop over ``(sql, varmap)`` pairs) or as a part of one
            # (a subquery), which the way of each stretch says.
            carried = sql.names_carried(executions)
            seen |= {text for text, name in assembled.items() if name in carried}
            explained += len(seen)
            for text, name in sorted(assembled.items()):
                if text in seen:
                    continue
                unexplained.append(
                    Unexplained(
                        reason=_why_not_run(func, name),
                        file=module.rel_path,
                        owner=f"{module.rel_path}::{func.name}",
                        # The variable, not a table: this slice counts
                        # statements, and which tables one names is what it
                        # could not get to.
                        table=name,
                        column="",
                        junction="",
                    )
                )
        if candidates:
            found.append(
                CoverageStat(
                    slice_name=READ_SLICE_NAME,
                    file=module.rel_path,
                    candidates=candidates,
                    explained=explained,
                )
            )
    return found, unexplained


def _why_not_run(func: ast.FunctionDef | ast.AsyncFunctionDef, name: str) -> str:
    """Why a statement this function assembled is not one the reader saw run.

    Asked of the calls that take *name* as their first argument, which is the
    same place :func:`sql.executions` looks and declines.  Naming the form it
    declined is the point: the denominator exists so that a query form the
    slice does not recognise cannot go missing silently, and a count alone says
    a form is missing without saying which one to add.
    """
    others: set[str] = set()
    for node in walk(func):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        first = node.args[0]
        mentions = isinstance(first, ast.Name) and first.id == name
        if not mentions:
            # ``execute(sql + comment, varMap)`` -- the reader fills and
            # concatenates before matching, so a statement reaching a call
            # inside an expression still counts as reaching it.
            mentions = any(
                isinstance(inner, ast.Name) and inner.id == name
                for inner in walk(first)
            )
        if not mentions:
            continue
        attr = node.func.attr if isinstance(node.func, ast.Attribute) else (
            node.func.id if isinstance(node.func, ast.Name) else "?"
        )
        if attr in sql.runners():
            # Accepted form, yet no run there holds the text: what the name
            # holds where it is used is not what reaches the call, which is a
            # different finding from an unrecognised call.
            return RUN_BUT_TEXT_DIFFERS
        others.add(attr)
    if others:
        return f"{RUN_THROUGH_OTHER_FORM}: {', '.join(sorted(others))}"
    return NOT_RUN_HERE


def _assembled_queries(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
) -> dict[str, str]:
    """Every statement *func* builds that asks for rows, however it is run.

    What each name the function writes statement text into holds where the
    code uses it or where the function ends (:func:`sql.assembled_texts`),
    rather than what the executions run.  A name is a candidate when what it
    holds names a table under a read verb; the call that runs it is not
    consulted, which is what keeps this independent of the reading it is the
    denominator for.

    Maps each statement to a local that holds it, so that one left out of the
    numerator can be traced back to the variable -- which is what a reader needs
    in order to see how it is run.  **Keyed by the text**: two locals assembling
    the same statement are one candidate, because the denominator counts
    statements.
    """
    held = sql.assembled_texts(func)
    return {
        text: name
        for name in sorted(held, reverse=True)
        for text in held[name]
        if any(verb == "read" for verb, _table in _table_verbs(text))
    }


#: ``ExceptHandler`` is not an ``ast.stmt``, which is why it is named rather
#: than reached through one.
_EXITS = (ast.Continue, ast.Break, ast.Return, ast.Raise, ast.ExceptHandler)
#: Spelled as the code spells them, so a reader can grep the line.
_EXIT_NAMES = {
    ast.Continue: "continue",
    ast.Break: "break",
    ast.Return: "return",
    ast.Raise: "raise",
    ast.ExceptHandler: "except",
}
#: The four ways a row stops being followed on its own.  ``ExceptHandler`` is
#: not one of them: it is a place where one of these may be written.
_LEAVES = (ast.Return, ast.Continue, ast.Break, ast.Raise)


#: ``while`` carries no ``iter``; its test is where the rows would be named,
#: and either way it is the names that decide.
_LOOPS = (ast.For, ast.AsyncFor, ast.While)


def _enclosing_loops(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
) -> dict[int, ast.AST]:
    """``{id(node): the loop that most closely encloses it}``, function-wide.

    An exit belongs to this statement only when the loop it is taken in is the
    loop over this statement's rows.  Judged by line range instead, a
    ``continue`` four levels down reads as the row being skipped:
    ``copyArchive@206`` picks up ``continue@245``, which is about one of a
    job's files.  Over the corpus that is 19 exits filed against the wrong
    question.
    """
    found: dict[int, ast.AST] = {}

    def descend(node: ast.AST, loop: Optional[ast.AST]) -> None:
        for child in ast.iter_child_nodes(node):
            if loop is not None:
                found[id(child)] = loop
            descend(child, child if isinstance(child, _LOOPS) else loop)

    descend(func, None)
    return found


def _leaves(handler: ast.ExceptHandler) -> bool:
    """Whether catching here actually stops the row.

    ``except Exception: pass`` is written so the row is *not* stopped, and a
    handler assigning a fallback is the same answer at more length.  Counting
    every handler made 2 of the 3 statements the map credited with an
    ``except`` say the opposite of what their code does.
    """
    return any(isinstance(inner, _LEAVES) for inner in walk(handler))


def _assigned_names(node: ast.AST) -> set[str]:
    return {name for target in targets_of(node) for name, _slot, _unpacked in names_bound_by(target)}


def _rows_written(
    func: ast.FunctionDef | ast.AsyncFunctionDef, line: int, last: int
) -> list[ast.Assign | ast.AnnAssign | ast.For | ast.AsyncFor]:
    """Where this statement's rows land, within the stretch it owns.

    An assignment on the call line, for the forwarded form that returns them;
    one in the stretch that fetches them or reads ``rowcount``, however its
    value computes from them -- ``res = self.cur.fetchall()``, ``has_merging =
    self.cur.fetchone() is not None``, ``nRow = self.cur.rowcount``; and a loop
    over the cursor that ran the statement (``for (pandaID,) in self.cur:``,
    the receiver of the call line's ``execute``), whose target takes each row.
    Reading only the call would leave most of the corpus with no rows to test
    an exit against -- see :data:`sql.FETCHES`.

    A fetch was read only as the whole value, and only where the call line
    bound nothing: ``has_merging`` and ``nRow`` held no rows, so ``if not
    has_merging: return`` was no exit and nothing they decide or are handed
    to a crossing, and a loop over the cursor was no loop over the rows.
    """
    cursors = {
        ast.dump(node.func.value)
        for node in walk(func)
        if isinstance(node, ast.Call)
        and node.lineno == line
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in ("execute", "executemany")
    }
    found: list[ast.Assign | ast.AnnAssign | ast.For | ast.AsyncFor] = []
    for node in walk(func):
        if not line <= getattr(node, "lineno", 0) <= last:
            continue
        if isinstance(node, (ast.For, ast.AsyncFor)):
            if ast.dump(node.iter) in cursors:
                found.append(node)
        # ``res: list = self.cur.fetchall()`` lands the rows as ``res = ...`` does.
        elif isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None and _assigned_names(node):
            if node.lineno == line or any(_fetches(inner) for inner in walk(node.value)):
                found.append(node)
    return found


def _fetches(node: ast.AST) -> bool:
    """Whether *node* fetches a statement's rows or reads how many there were: ``x.fetchall()``, ``x.rowcount``."""
    if isinstance(node, ast.Call):
        return isinstance(node.func, ast.Attribute) and node.func.attr in sql.FETCHES
    return isinstance(node, ast.Attribute) and node.attr == "rowcount"


def _route_constants(calls: Calls, target: Target, sites: tuple[CallSite, ...], tested: set[str]) -> dict[str, object]:
    """The constants the route *sites* (the call into *target* first) hands *target*'s parameters *tested*."""
    return {name: value for name in sorted(tested) if (value := calls.route_constant(name, target, sites)) is not UNSET}


def _flag_parameters(func: ast.AST) -> set[str]:
    """The parameters *func* tests in an ``if`` and never rebinds: the ones a call's constant can decide."""
    args = func.args  # type: ignore[attr-defined]
    parameters = {a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)} - {"self", "cls"}
    tested = {
        name.id
        for node in walk(func)
        if isinstance(node, ast.If) and enclosing_function(node) is func
        for name in walk(node.test)
        if isinstance(name, ast.Name) and name.id in parameters
    }
    rebound = {
        node.id
        for node in walk(func)
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del))
    }
    return tested - rebound


def _module_of(node: ast.AST) -> Optional[ast.Module]:
    while getattr(node, "parent", None) is not None:
        node = node.parent  # type: ignore[attr-defined]
    return node if isinstance(node, ast.Module) else None


def corpus_calls(
    modules: list[SourceModule], roots: Optional[dict[str, Path]], recorded: Optional[Recorded] = None
) -> Optional[Calls]:
    """The corpus's calls, read over the build's own trees; ``None`` without *roots*.

    The trees are the modules' so that a caller :meth:`Calls.callers` finds is
    a function the build has already parsed and attached parents to.
    *recorded* is the instance's record, where the build has one (:func:`~bamboo.codemap.calls.read_recorded`).
    """
    if not roots:
        return None
    walked = {}
    for module in modules:
        attach_parents(module.tree)
        walked[module.rel_path] = (module.blob_sha, module.tree)
    return Calls(Types(roots, walked, recorded))


def read_side(
    modules: list[SourceModule],
    attributor: SpecAttributor,
    calls: Optional[Calls] = None,
) -> ReadSide:
    """Return what the corpus's statements ask for: which values, and whose rows.

    **The rows are half of it, and the half the map had nowhere to put.**  A
    vocabulary of ``(spec class, attribute)`` pairs can only say something
    about a column, so a query selecting a task's jobs on the join key alone
    -- ``SELECT PandaID FROM jobsActive4 WHERE jediTaskID=:jediTaskID`` --
    named no promoted subject and read as observing nothing.  It observes jobs.
    The class was already being computed here and dropped whenever no predicate
    landed on a promoted attribute.

    Verbs are kept apart.  A function that selects a task's jobs and one that
    updates them make different claims, and pooling them is the same
    conflation that let an ``UPDATE ... WHERE`` be reported as a query that
    selects on a value.

    The same statements read the other way.  A write says what a value becomes;
    a predicate says which rows were asked for, and only both together make a
    state machine out of a pile of writes -- a status nothing ever selects on
    is a status nothing ever moves a task out of, which is the shape of "stuck"
    that no branch table can show.

    **The asking function is kept, not only the value.**  It is in hand here --
    the walk is over functions -- and dropping it left "something selects
    ``finishing``" as the whole answer when exactly one query does, which is
    the difference between a fact and a direction to look in.  Measured over
    the corpus: 134 ``(subject, value)`` pairs have a named reader and 69 of
    them have exactly one.  Downstream, ``FollowUp`` had been pooling triggers
    over the subject's *writers* as a stand-in for the reader's -- an
    approximation its own docstring had to name.

    Attributed exactly like a write, so ``JediTaskSpec.status`` means the same
    thing on both sides and the two can be compared at all.

    **Bound values are settled the same way the write side settles them.**  Both
    sides read the same declared mappings, and taking only a literal here left
    the read side blind to every status a command passes through:
    ``varMap[":status"] = taskStatusMap["doing"]`` is the orphan-rescue query in
    ``getTasksToExecCommand_JEDI``, and without it ``aborting``, ``finishing``,
    ``toretry``, ``toincexec`` and ``toreassign`` all looked like values no
    query ever asks for -- that is, like states a task can enter and never
    leave.  The write side already resolved the same mapping through the other
    spelling, so the two slices were disagreeing about one fact.

    That mapping's ``dummy`` sentinel comes along with them, and the loop
    skips it (``if varMap[":status"] in ["dummy", "paused"]: continue``) before
    the statement runs.  Left in rather than modelled: the guard is a
    ``continue`` above the execution, which is not the dominating-``if`` shape
    the path conditions read, and building that for one site would be a
    mechanism for one site.  It costs nothing measurable here: ``dummy`` was
    not among the values reported as written-but-unselected for this subject,
    so nothing is hidden by it.
    """
    handed = values.handed(values.resolver(), calls)
    # Every caller of a parameter is read now, however many: how long that
    # takes, and the function with the most routes, go to the build's report.
    watch: dict[str, object] = {"settle_seconds": 0.0, "most_routes": ("", 0)}

    def settle(expression, func, file):
        started = time.perf_counter()
        try:
            return handed(expression, func, file)
        finally:
            watch["settle_seconds"] += time.perf_counter() - started  # type: ignore[operator]

    found: dict[str, dict[str, ValueUse]] = {}
    rows: dict[str, EntityUse] = {}
    unreadable: dict[str, int] = {}
    joins: dict[str, set[frozenset[str]]] = {}

    statements: dict[tuple[str, int], ReadStatement] = {}
    # The route of calls the read side is reading a function as, while it
    # reads one (``place_routes``): its chain, and what it hands the
    # function's parameters.
    routing: dict[str, object] = {"chain": None, "into": frozenset(), "func": None, "handed": None}
    # What every run of a statement so far passes over, by (owner, line) --
    # kept for runs that file nothing too, since a run that refuses nothing
    # is the run that returns the row.
    refused: dict[tuple[str, int], set[tuple[str, str]]] = {}
    # And what every run so far restricts, the same way.
    restricted: dict[tuple[str, int], dict[str, set[str]]] = {}
    # Each route recorded as reading as the statement, with what it read:
    # checked against the statement's own fields once they are final.
    as_statement: list[tuple[tuple[str, int], tuple[str, ...], CallReading]] = []

    def record(
        subject: str,
        value: str,
        owner: str,
        verb: str,
        at: Optional[ReadStatement] = None,
    ) -> None:
        use = found.setdefault(subject, {}).setdefault(value, ValueUse(set(), set()))
        (use.selected_by if verb == "read" else use.updated_by).add(owner)
        if verb != "read" or at is None:
            return
        # Filed against the statement as well as the function, and only kept
        # when a pair actually lands: a run whose predicates name no subject
        # is not somewhere to send a reader, so it gets no row.
        kept = statements.setdefault((at.owner, at.line), at)
        kept.selects.add((subject, value))
        # Unioned across the variants, not taken from whichever recorded
        # first.  ``conditions`` is deliberately *not* unioned -- see
        # ``place_conditions``, where a union reads as a conjunction and
        # over-states what the row must be -- but a stop is not a requirement:
        # it says the reader could not follow a fragment, and a fragment one
        # variant could not follow is one the statement has.  Keeping the
        # first variant's left ``lockJobsForReassign`` unmarked, because the
        # two-column variant with no ``WHERE`` at all recorded before the one
        # carrying ``jobStatus IN ({})``.
        if at is not kept and at.unresolved:
            statements[(at.owner, at.line)] = kept._replace(
                unresolved=tuple(sorted(set(kept.unresolved) | set(at.unresolved)))
            )

    def exclude(subject: str, value: str, at: ReadStatement) -> None:
        """File a value the statement passes over.  Not a use of the value:
        ``status<>'done'`` is no reader of ``done`` rows."""
        kept = statements.setdefault((at.owner, at.line), at)
        kept.excludes.add((subject, value))

    def consult(subject: str, value: str, at: ReadStatement) -> None:
        """File a value a subquery of the statement checks rows by.  Not a use
        of the value either: ``EXISTS (... jobStatus=:st_merging)`` asks whether
        a merging job exists and hands none of them on."""
        statements.setdefault((at.owner, at.line), at).consults.add((subject, value))

    def bound(func, run, key: str, file: str) -> list[str]:
        return [value for value, _site in bound_sites(func, run, key, file)]

    def file_run(func, run, at: ReadStatement, read_tables: set[str], file: str) -> None:
        """Onto the record kept for *run*'s line: *at*'s stops, and the tables, orders and limits *run* reads.

        Every run of the line, the ones whose predicates file no pair too: the
        run of ``get_job_statistics_per_site_label_resource``'s loop over four
        statements that has no ``WHERE`` still reads ``jobsDefined4``.
        """
        site = (at.owner, at.line)
        kept = statements[site]
        if kept is not at and at.unresolved:
            kept = kept._replace(unresolved=tuple(sorted(set(kept.unresolved) | set(at.unresolved))))
            statements[site] = kept
        kept.tables.update(read_tables)
        if kept.joins is not None:
            kept.joins.update(
                (
                    f"{one.table}.{one.column}",
                    f"{one.other_table}.{one.other_column}",
                    one.source,
                    one.other_source,
                    _spec_attr(attributor, one.table, one.column),
                    _spec_attr(attributor, one.other_table, one.other_column),
                )
                for one in sqltext.equalities(run.reading)
            )
        orders, limits = sqltext.ordering(run.reading)
        kept.order_by.update(orders)
        for bound_to, strict in limits:
            values_of = bound(func, run, bound_to, file) if bound_to.startswith(":") else [bound_to]
            kept.limits.update(f"{'<' if strict else '<='} {value}" for value in values_of or [bound_to])

    # What each run's bind is filled with, every route's values with the
    # calls they came through: a route only keeps some of them, and reading
    # them again for each of a function's routes is the cost of reading
    # routes at all.  The run is kept beside its values so that its ``id``
    # is not another run's while this read lasts.
    settled: dict[tuple[int, str, str], tuple[object, list[tuple[str, ast.stmt, tuple[str, ...], bool]]]] = {}

    def bound_sites(func, run, key: str, file: str) -> list[tuple[str, ast.stmt]]:
        """The values *key* is filled with for *run*, each with the statement putting it there -- on the route being read, if one is.

        Read as every run any call makes, none where a call's value is not
        known (:attr:`values.Handed.unknown`): the rest are not every value
        the bind is filled with.  On a route, what the route's calls handed.
        """
        held = settled.get((id(run), key, file))
        if held is None or held[0] is not run:
            found: list[tuple[str, ast.stmt, tuple[str, ...], bool]] = []
            binds = sql.bound_values(func, run.varmap or "", key, run.window, across=True, at=run)
            if not run.varmap:
                binds = binds + sql.inline_binds(run.call, key)
            for bind in binds:
                # Read where the value is written: one a call handed the varmap
                # sits in the callee, an imported ``IN`` list at the top of
                # another module, and their names are theirs.
                where = enclosing_function(bind.value) or _module_of(bind.value) or func
                at_file = file
                if where is not func and calls is not None:
                    at_file = calls.types.file_of(where) or file
                found.extend(
                    (one.value, bind.site, one.via, one.unknown) for one in settle(bind.value, where, at_file)
                )
            held = settled[(id(run), key, file)] = (run, found)
        chain = routing["chain"]
        if chain is None:
            if any(unknown for _value, _site, _via, unknown in held[1]):
                return []
            return [(value, site) for value, site, _via, _unknown in held[1]]
        return [
            (value, site)
            for value, site, via, unknown in held[1]
            if not unknown and _on_route(via, chain, routing["into"])
        ]

    def tree_of(func, run, file: str, twin: "sqltext.Twin") -> ConditionNode:
        """*twin* with each value its terms compare with read off *run*: :attr:`ReadSiteNode.condition_trees`."""
        if twin.node != "test":
            return ConditionNode(node=twin.node, items=[tree_of(func, run, file, one) for one in twin.items])
        test = twin.test
        if test is None or test.kind == "other":
            return ConditionNode(node="other", text=test.text if test is not None else "")
        found: list[ConditionValue] = []
        settled = True
        for side in test.sides:
            for value in side_values(func, run, file, side, test.kind == "compare"):
                settled = settled and not (value.needs or value.said or value.kind == "other")
                if value not in found:
                    found.append(value)
        return ConditionNode(
            node=test.kind,
            text=test.text,
            subject=_column_subject(attributor, test.table, test.column),
            op=test.op,
            values=found,
            settled=settled,
        )

    def side_values(func, run, file: str, side: "sqltext.Side", amount: bool) -> list[ConditionValue]:
        """What one side of a term holds on *run*: one value per way a run fills it.

        *amount*: a ``<``/``>`` comparison, whose bind is read as an amount
        (:func:`values.stated`, through the writes reaching it); an ``=`` or
        ``IN`` reads its bind's values as ``selects`` does.
        """
        if side.kind == "constant":
            return [ConditionValue(kind="constant", text=side.text, constant=side.constant)]
        if side.kind == "column":
            return [ConditionValue(kind="column", text=side.text, subject=_column_subject(attributor, side.table, side.column))]
        if side.kind == "other":
            return [ConditionValue(kind="other", text=side.text, said=side.text)]
        if side.kind == "clock" and not side.binds:
            return [ConditionValue(kind="clock", text=side.text, base=side.clock, offset_seconds=side.offset_seconds)]
        key = side.binds[0]
        if side.kind == "bind" and not amount:
            settled_values = bound(func, run, key, file)
            if not settled_values:
                return [ConditionValue(kind="bind", text=side.text, said=f"nothing in the map settles {key}")]
            return [ConditionValue(kind="bind", text=side.text, constant=value) for value in settled_values]
        binds = sql.bound_values(func, run.varmap or "", key, run.window, across=True, at=run)
        if not run.varmap:
            binds = binds + sql.inline_binds(run.call, key)
        if not binds:
            return [ConditionValue(kind=side.kind, text=side.text, base=side.clock, scale_seconds=side.scale_seconds,
                                   said=f"nothing in the map fills {key}")]
        found: list[ConditionValue] = []
        for bind in binds:
            where = enclosing_function(bind.value) or func
            handed = routing["handed"] if where is routing["func"] else None
            value = values.stated(bind.value, where, bind.site, handed=handed)
            if side.kind == "clock":
                found.append(_clock_amount(side, value))
            else:
                found.append(_bind_amount(side, value))
        return found

    def reading_of(func, run, file: str) -> _CallReading:
        """What one run reads and asks by -- what a call site can narrow."""
        reading = _CallReading({table for verb, table in _table_verbs(run.reading) if verb == "read"})
        # The same terms the statement's own fields are filled from, read the
        # same way, or a call site would differ from the statement by the
        # reading alone.
        for term in sqltext.terms(run.reading).terms:
            if term.verb != "read" or term.scope not in (sqltext.ROW, sqltext.CONSULT):
                continue
            subject = _term_subject(attributor, term)
            if term.scope == sqltext.CONSULT:
                if not term.excluding:
                    for key in term.binds:
                        reading.consults.update((subject, value) for value in bound(func, run, key, file))
                    reading.consults.update((subject, value) for value in term.literals)
                continue
            into = reading.excludes if term.excluding else reading.selects
            unsaid = term.undecided
            for key in term.binds:
                values_of = bound(func, run, key, file)
                unsaid = unsaid or not values_of
                into.update((subject, value) for value in values_of)
            into.update((subject, value) for value in term.literals)
            if term.excluding and unsaid:
                into.add((subject, "{}"))
        return reading

    def run_refuses(func, run, file: str, ways) -> Optional[set[tuple[str, str]]]:
        """What no row of *run* holds, however it was built; ``None`` where nothing is read.

        *ways* is :func:`_outright_terms` of *run*, read once for this and
        :func:`run_restricts`: reading the builds is the expensive half.

        Narrower than ``excludes``, which files every value a term names: here
        only what every row of every branch is without, and only a value the
        run settles -- ``jobStatus<>:jobStatus`` bound to ``dummy`` on one path
        and ``running`` on another passes over neither on every run.  An
        ``IN`` list the function states is the exception (:func:`stated_list`):
        all its members are passed over at once.
        """
        refuses_here: Optional[set[tuple[str, str]]] = None
        lists = {one.bind for one in sql.in_lists(func).values()}
        for alternative in ways:
            passed_over: set[tuple[str, str]] = set()
            for term in alternative:
                if not term.excluding or term.verb != "read" or term.scope != sqltext.ROW:
                    continue
                subject = _term_subject(attributor, term)
                for key in term.binds:
                    members = stated_list(func, run, key, file) if key in lists else None
                    if members:
                        passed_over.update((subject, value) for value in members)
                        continue
                    values_of = bound(func, run, key, file)
                    if len(set(values_of)) == 1:
                        passed_over.add((subject, values_of[0]))
                passed_over.update((subject, value) for value in term.literals)
            refuses_here = passed_over if refuses_here is None else refuses_here & passed_over
        return refuses_here

    def run_restricts(func, run, file: str, ways) -> Optional[dict[str, set[str]]]:
        """``{subject: values}`` every row of *run* holds one of, however it was built; ``None`` where nothing is read.

        The ``selects`` counterpart of :func:`run_refuses`: a subject counts
        where every way a row comes back, in every build, asks it for values
        outright (:attr:`sqltext.Term.decided`), and its values are all of
        theirs -- a row holding none of them comes back from no way.  Read as
        ``selects`` reads them, so they are some of its pairs.  A subject one
        of whose deciding terms has a bind nothing here fills is dropped from
        that way: the row may hold the value no one could name.
        """
        restricts_here: Optional[dict[str, set[str]]] = None
        for alternative in ways:
            asked: dict[str, set[str]] = {}
            unnamed: set[str] = set()
            for term in alternative:
                if term.excluding or term.verb != "read" or term.scope != sqltext.ROW:
                    continue
                subject = _term_subject(attributor, term)
                values_of = set(term.literals)
                for key in term.binds:
                    filled = bound(func, run, key, file)
                    if not filled:
                        unnamed.add(subject)
                    values_of.update(filled)
                asked.setdefault(subject, set()).update(values_of)
            for subject in unnamed:
                asked.pop(subject, None)
            restricts_here = asked if restricts_here is None else _restricted_both(restricts_here, asked)
        return restricts_here

    def stated_list(func, run, key: str, file: str) -> Optional[list[str]]:
        """The members of the ``IN`` list bound to *key*, where every run holds all of them.

        ``get_task_ids_with_dataset_attributes`` asks ``tabT.status NOT IN
        ({task_var_names_str})`` of ``JediTaskSpec.statusToRejectExtChange()``:
        one placeholder for ten values, every one passed over on each run.
        That holds only for a list the function states -- one assignment, of
        a literal, a declaration or a name written once -- and not for one a
        caller hands, where different calls hand different lists and the
        union is no run's.
        """
        binds = sql.bound_values(func, run.varmap or "", key, run.window, across=True, at=run)
        if len(binds) != 1:
            return None
        value = binds[0].value
        where = enclosing_function(value) or _module_of(value) or func
        if isinstance(value, ast.Name):
            written = [
                node
                for node in walk(where)
                if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign))
                and any(
                    isinstance(target, ast.Name) and target.id == value.id
                    for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
                )
            ]
            if len(written) != 1 or isinstance(written[0], ast.AugAssign):
                return None
        at_file = file
        if where is not func and calls is not None:
            at_file = calls.types.file_of(where) or file
        handed = settle(value, where, at_file)
        if not handed or any(one.via or one.unknown for one in handed):
            return None
        return [one.value for one in handed]

    def place_routes(func, owner: str, file: str, executions: list) -> None:
        """Say, per statement, what it reads on each route of calls into *func* -- :class:`CallReading`.

        The statement's own reading is every run any call can make.  A route
        runs fewer: ``TaskCommando.py:143`` calls ``getPandaIDsWithTask_JEDI
        (jediTaskID, True)``, and under ``onlyActive`` the loop reads two
        tables, not four; ``copyArchive`` hands ``lockJobsForReassign`` a
        status list of its own at each of four calls.  Each route is read as
        the statement is -- the same functions, with the route's constants
        deciding what runs (:func:`sql.at_call`) and its values in the binds
        -- and compared with the statement's reading read the same way,
        field by field off :class:`CallReading`: equal, it says so and
        carries nothing; otherwise it carries every field.
        """
        lines = sorted({run.call.lineno for run in executions if (owner, run.call.lineno) in statements})
        if calls is None or not lines:
            return
        owned = calls.types.method_class(file, func)
        target = Target(file, func, owned[1] if owned else None)
        found, stops = calls.routes(target)
        if len(found) > watch["most_routes"][1]:  # type: ignore[index]
            watch["most_routes"] = (owner, len(found))
        into = frozenset(f"{site.file}:{site.call.lineno}" for site in calls.callers(target))
        tested = _flag_parameters(func)
        own = route_reading(func, file)
        records: dict[int, list[CallReading]] = {line: [] for line in lines}
        for route in found:
            constants = _route_constants(calls, target, route.sites, tested)
            routing.update(chain=route.chain, into=into, func=func, handed=_route_handed(target, route.sites, calls))
            try:
                if constants:
                    with sql.at_call(func, constants):
                        here = route_reading(func, file)
                else:
                    here = route_reading(func, file)
            finally:
                routing.update(chain=None, into=frozenset(), func=None, handed=None)
            named = {name: repr(value) for name, value in constants.items()}
            opened = [f"{site.file}:{site.call.lineno} -- {why}" for site in route.sites if (why := calls.opened(site))]
            for line in lines:
                if line not in here:
                    records[line].append(CallReading(chain=list(route.chain), constants=named, not_run=True, open=opened))
                    continue
                read = here[line].model_copy(update={"chain": list(route.chain), "constants": named, "open": opened})
                if _reads_alike(read, own.get(line)):
                    as_statement.append(((owner, line), tuple(route.chain), read))
                    read = CallReading(chain=list(route.chain), constants=named, as_statement=True, open=opened)
                records[line].append(read)
        for line in lines:
            kept = statements[(owner, line)]
            statements[(owner, line)] = kept._replace(
                call_readings=tuple(records[line]), callers_complete=not stops, callers_stop=tuple(stops)
            )

    def route_reading(func, file: str) -> dict[int, CallReading]:
        """What each statement of *func* reads, per line, as the read side is reading it now -- a route's or every run's."""
        per_line: dict[int, _CallReading] = {}
        variants: dict[int, list] = {}
        refuses_at: dict[int, set[tuple[str, str]]] = {}
        restricts_at: dict[int, dict[str, set[str]]] = {}
        for run in sql.executions(func):
            line = run.call.lineno
            per_line.setdefault(line, _CallReading(set())).add(reading_of(func, run, file))
            if all(run.sql != other.sql for other in variants.setdefault(line, [])):
                variants[line].append(run)
            ways = _outright_terms(func, run)
            here = run_refuses(func, run, file, ways) or set()
            refuses_at[line] = refuses_at[line] & here if line in refuses_at else here
            asked = run_restricts(func, run, file, ways) or {}
            restricts_at[line] = _restricted_both(restricts_at[line], asked) if line in restricts_at else asked
        readings: dict[int, CallReading] = {}
        for line, one in per_line.items():
            folded = _folded(func, line, variants[line], lambda run, twin: tree_of(func, run, file, twin))
            readings[line] = CallReading(
                chain=[],
                tables=sorted(one.tables),
                selects=sorted(f"{subject}={value}" for subject, value in one.selects),
                excludes=sorted(f"{subject}={value}" for subject, value in one.excludes),
                consults=sorted(f"{subject}={value}" for subject, value in one.consults),
                refuses=sorted(f"{subject}={value}" for subject, value in refuses_at[line]),
                restricts=sorted(f"{subject}={value}" for subject, value in _restricted_pairs(restricts_at[line])),
                condition_trees=sorted(
                    (ConditionReading(entry=entry, tree=tree) for entry, tree in folded if tree is not None),
                    key=lambda reading: reading.entry,
                ),
            )
        return readings

    def touch(table: str, owner: str, verb: str) -> None:
        use = rows.setdefault(table.lower(), EntityUse(set(), set(), set(), set(), set()))
        use.tables.add(table)
        getattr(use, f"{verb}_by").add(owner)

    def place_orders(func: ast.FunctionDef | ast.AsyncFunctionDef, owner: str, file: str, executions: list) -> None:
        """Say, per statement, what builds each blanked item of its ``ORDER BY`` -- :class:`OrderHole`.

        The item's hole is the evaluator's (:attr:`sql.Execution.holes`): a
        part a loop of unknown length appends keeps the loop and what one
        pass appends.  The template, written in once, says which columns the
        item ranks by; the members' order is the loop's, computed at run time.
        """

        def file_of(one: ast.AST) -> str:
            return (calls.types.file_of(one) if calls is not None else None) or file

        def said(
            run: sql.Execution, order: str, index: int, item, templates: list[str], loops: list, where, reading: str
        ) -> OrderHole:
            # The columns the template adds to the order: read with the loop's passes in the hole
            # (:func:`sql.reading_text`).
            before = set(sqltext.order_owners(run.sql))
            owners = sqltext.order_owners(reading)
            subjects = sorted({_column_subject(attributor, table, column) for table, column in owners if (table, column) not in before})
            return OrderHole(
                order=order,
                item=index,
                subjects=subjects,
                templates=templates,
                computed_at="; ".join(_loop_said(loop, where, file_of(where)) for loop in loops),
            )

        for run in executions:
            kept = statements.get((owner, run.call.lineno))
            if kept is None or not any("{}" in order for order in kept.order_by):
                continue
            found = list(kept.order_holes)
            # The hole is the evaluator's: a loop of unknown length's part keeps the loop and what one
            # pass appends (``sqleval.HoleAt.repeats``).  Its holes are the text's ``{}`` in order.
            if len(run.holes) != run.sql.count("{}"):
                continue
            squeezed = _squeezed(run.sql)
            for order in kept.order_by:
                start = squeezed.find(_squeezed(f"ORDER BY {order}"))
                if "{}" not in order or start < 0:
                    continue
                at = squeezed.count("{}", 0, start)
                for index, item in enumerate(sqltext.order_items(order)):
                    held = run.holes[at] if item.blanked and at < len(run.holes) else None
                    at += item.expression.count("{}")
                    hole = held[1] if held is not None else None
                    if hole is None or not hole.repeats:
                        continue
                    where = enclosing_function(hole.stmt) or func
                    hole_read = said(
                        run, order, index, item, list(hole.repeats), [hole.stmt], where, sql.reading_text(run.sql, (held,))
                    )
                    if hole_read not in found:
                        found.append(hole_read)
            statements[(owner, run.call.lineno)] = kept._replace(order_holes=tuple(found))

    def place_prints(func: ast.FunctionDef | ast.AsyncFunctionDef, owner: str, executions: list) -> None:
        """Say, per statement, which logging calls print its text and the map it runs with: :func:`_printed_by`."""
        for run in executions:
            kept = statements.get((owner, run.call.lineno))
            if kept is None:
                continue
            found = [one for one in _printed_by(func, run) if one not in kept.printed_by]
            if found:
                statements[(owner, run.call.lineno)] = kept._replace(printed_by=(*kept.printed_by, *found))

    def place_conditions(
        func: ast.FunctionDef | ast.AsyncFunctionDef, owner: str, file: str, executions: list
    ) -> None:
        """Say what each of *func*'s statements requires of a row besides the value.

        A branchy statement is emitted as several variants at one ``execute``,
        and exactly one of them ran.  Unioning their terms -- which is what
        ``selects`` does, and what this did first -- reads as a conjunction and
        so **over-states** the requirement: 24 of the corpus's statements have
        variants that disagree, over 66 terms, and a reader ruling a row out
        against a term the text that ran never carried is the failure this
        field exists to prevent.  So the variants are folded as what they are,
        a disjunction -- :func:`_folded`, which a call site's reading folds by
        too.
        """
        runs: dict[int, list] = {}
        for run in executions:
            seen_text = runs.setdefault(run.call.lineno, [])
            if all(run.sql != other.sql for other in seen_text):
                seen_text.append(run)
        for line, variants in runs.items():
            kept = statements.get((owner, line))
            if kept is None:
                continue
            folded = _folded(func, line, variants, lambda run, twin: tree_of(func, run, file, twin))
            kept.conditions.extend(entry for entry, _tree in folded)
            statements[(owner, line)] = kept._replace(
                condition_trees=(
                    *kept.condition_trees,
                    *(ConditionReading(entry=entry, tree=tree) for entry, tree in folded if tree is not None),
                )
            )

    def place_added_under(
        func: ast.FunctionDef | ast.AsyncFunctionDef, owner: str, file: str, executions: list
    ) -> None:
        """Say, per statement, which terms of its fields some builds of it do not hold, and what adds each.

        A build is one variant of the statement with what some paths append
        cut out (:func:`_run_builds`), through every name its text passed
        through.  A term every build holds is not said; one some build lacks
        is, with each fragment that carries it and the tests above that
        fragment (:func:`_place`).  ``selects`` and ``excludes`` are compared
        pair by pair, so that a pair one build asks in an ``IN`` list and
        another in a plain ``=`` is in both.

        **Each variant is read on its own, the values in its binds too.**
        ``construct_where_clause``'s branches reuse their bind names --
        ``:prodSourceLabel2`` is ``panda`` under ``prod_source_label ==
        "user"`` and ``test`` under ``"managed"`` -- and a varmap is read by
        key, not by path.  Read across variants, every branch asked for every
        label, so ``panda`` was in every build and read as required, and a
        term was placed at every fragment naming its binds.  So a value counts
        in a variant only where the statement putting it into the bind can run
        with what the variant appends (:func:`_meets`), and a term is looked
        for among that variant's own fragments, where a bind is in one place.
        """
        runs: dict[int, list] = {}
        for run in executions:
            held = runs.setdefault(run.call.lineno, [])
            if all(run.sql != other.sql for other in held):
                held.append(run)
        for line, variants in runs.items():
            kept = statements.get((owner, line))
            if kept is not None:
                added, read_trees = terms_added(func, file, line, variants, kept)
                held = {one.entry for one in kept.condition_trees}
                statements[(owner, line)] = kept._replace(
                    added_under=added,
                    condition_trees=(*kept.condition_trees, *(one for one in read_trees if one.entry not in held)),
                )

    def terms_added(
        func: ast.FunctionDef | ast.AsyncFunctionDef, file: str, line: int, variants: list, kept: ReadStatement
    ) -> tuple[tuple[AddedTerm, ...], tuple[ConditionReading, ...]]:
        """:func:`place_added_under` for the statement *variants* run at *line*, and each added conditions term's tree."""

        def file_of(one: ast.AST) -> str:
            return (calls.types.file_of(one) if calls is not None else None) or file

        def identity(term) -> tuple:
            return (term.table.lower(), term.column.lower(), term.excluding, term.binds, term.literals)

        def plain(condition) -> str:
            return _SEEDED.sub(r"\1", _render_condition(condition, {}))

        filled: dict[tuple[int, str], list[tuple[str, ast.stmt]]] = {}
        row_terms: dict[int, list] = {}
        orders: dict[int, list[str]] = {}
        # ``(plain, binds)`` per run, and each plain spelling as ``conditions`` seeds it.
        own_conditions: dict[int, list[tuple[str, tuple[str, ...]]]] = {}
        spelled: dict[tuple[str, str], str] = {}
        # Each conditions term's tree, the runs holding it merged.
        trees: dict[tuple[str, str], Optional[ConditionNode]] = {}
        # ``(spelling, plain, binds)`` of each term the run's ``WHERE`` ``AND``\ s
        # together -- what a pair enters the statement in (:func:`sqltext.conjoined`).
        conjuncts: dict[int, list[tuple[str, str, tuple[str, ...]]]] = {}
        # Where each conditions term, order and row term is in a run's statement
        # (``run.pieces``): each place it is read off, as stretches.
        located: dict[int, list[tuple[str, Any, tuple[tuple[tuple[int, int], ...], ...]]]] = {}
        for run in variants:
            try:
                whole = sqltext.terms(run.reading).terms
                whole_at = sqltext.term_spans(run.reading)
            except Exception:  # noqa: BLE001 -- a text sqlglot will not scope is read as nothing
                whole, whole_at = (), ()
            row_terms[id(run)] = [term for term in whole if term.verb == "read" and term.scope == sqltext.ROW]
            orders[id(run)] = sqltext.ordering(run.reading)[0]
            found = sqltext.conditions(run.reading)
            seeds = _condition_seeds(func, run, line, found)
            own_conditions[id(run)] = [(plain(one), tuple(dict.fromkeys(one.binds))) for one in found]
            for one, twin in zip(found, sqltext.condition_twins(run.reading), strict=True):
                spelled.setdefault(("conditions", plain(one)), _render_condition(one, seeds))
                key = ("conditions", plain(one))
                trees[key] = _merged_tree(trees.get(key), tree_of(func, run, file, twin))
            joined = sqltext.conjoined(run.reading)
            seeds = _condition_seeds(func, run, line, joined)
            conjuncts[id(run)] = [
                (_render_condition(one, seeds), plain(one), tuple(dict.fromkeys(one.binds))) for one in joined
            ]
            found_at = sqltext.condition_spans(run.reading)
            orders_at = sqltext.ordering_spans(run.reading)
            located[id(run)] = (
                [("conditions", plain(one), at) for one, at in zip(found, found_at, strict=True)]
                + [("order_by", order, at) for order, at in zip(orders[id(run)], orders_at, strict=True)]
                + [
                    ("term", term, ((at,),))
                    for term, at in zip(whole, whole_at, strict=True)
                    if term.verb == "read" and term.scope == sqltext.ROW
                ]
            )

        def values(run, key: str, along: Optional[dict[int, list[ast.AST]]]) -> list[str]:
            """What *key* is filled with in a run passing *along*; every value where *along* is ``None``."""
            if (id(run), key) not in filled:
                filled[(id(run), key)] = bound_sites(func, run, key, file)
            return [value for value, site in filled[(id(run), key)] if along is None or _meets(site, along)]

        def pairs(run, term, along) -> list[tuple[tuple[str, str], tuple[str, ...], tuple[str, ...]]]:
            """``(pair, binds, words)``: what *term* asks for in a run passing *along*, and what in the text carries it."""
            subject = _term_subject(attributor, term)
            field = "excludes" if term.excluding else "selects"
            found = [((field, f"{subject}={value}"), (), (term.column, f"'{value}'")) for value in term.literals]
            unsaid = term.undecided
            for key in term.binds:
                values_of = values(run, key, along)
                unsaid = unsaid or not values_of
                found.extend(((field, f"{subject}={value}"), (key,), ()) for value in values_of)
            if term.excluding and unsaid:
                found.append(((field, f"{subject}={{}}"), term.binds, (term.column,)))
            return found

        def holds(run, build: _Build) -> Optional[set[tuple[str, str]]]:
            """Every ``(field, term)`` each run of *build* has; ``None`` where its text will not read."""
            along = _along(node for leaf in build.leaves if leaf.kept for node in _trail(leaf))
            try:
                present = {identity(term) for term in sqltext.terms(build.text).terms}
                here = {("conditions", plain(one)) for one in sqltext.conditions(build.text)}
                here |= {("order_by", order) for order in sqltext.ordering(build.text)[0]}
            except Exception:  # noqa: BLE001 -- a build whose text cannot be read decides nothing
                return None
            for term in row_terms[id(run)]:
                if identity(term) in present:
                    here.update(pair for pair, _binds, _words in pairs(run, term, along))
            return here

        def within(run, binds: tuple[str, ...], words: tuple[str, ...]) -> str:
            """The term of *run*'s ``WHERE`` that holds a pair -- an ``OR`` with another column whole -- and nothing
            where none does: the fragment's own text can hold other terms' ``OR``\\ s."""
            return next(
                (
                    spelling
                    for spelling, text, holding in conjuncts[id(run)]
                    if (set(binds) <= set(holding) if binds else _names(text, (), words))
                ),
                "",
            )

        def at(
            run, build: _Build, along: Optional[dict[int, list[ast.AST]]]
        ) -> dict[tuple[str, str], list[tuple[list[_Leaf], str, Optional[bool]]]]:
            """``{(field, term): [(the fragments its text is made of, the term it enters in, whether that decides it)]}``.

            For a run the evaluator read (:attr:`sql.Execution.pieces`), off
            where each term is in the statement: every fragment its text
            overlaps is one the statement needs to hold it.  ``jobStatus IN
            ('activated', 'sent', 'starting')`` is a template's words around a
            list one branch writes, and asking which fragment's text names the
            term found neither.  What a loop's passes write, where the
            statement holds them as one repeated part (``getJobs``' ``(CASE {}
            ELSE {} END)``), is not: any number of passes holds the term, and
            it is placed by them only where they are all of it.
            """
            looped = {id(hole.stmt) for _at, hole in run.holes or () if hole.reason == sqleval.REPEATED}
            found: dict[tuple[str, str], list[tuple[list[_Leaf], str, Optional[bool]]]] = {}
            for kind, what, places_of in located[id(run)]:
                for spans in places_of:
                    leaves = [leaf for leaf in build.leaves if any(leaf.start < b and a < leaf.end for a, b in spans)]
                    leaves = [leaf for leaf in leaves if not _in_loop(leaf.node, looped)] or leaves
                    if not leaves:
                        continue
                    if kind == "conditions":
                        found.setdefault(("conditions", what), []).append((leaves, spelled[("conditions", what)], None))
                    elif kind == "order_by":
                        found.setdefault(("order_by", what), []).append((leaves, what, None))
                    else:
                        for pair, binds, words in pairs(run, what, along):
                            found.setdefault(pair, []).append((leaves, within(run, binds, words), what.decided))
            return found

        read = [
            (run, build, holds(run, build))
            for run in variants
            for build in _run_builds(func, run) or [_Build(run.sql, run.reading, [])]
        ]
        readable = [held for _run, _build, held in read if held is not None]
        if not readable:
            return (), ()
        every = set.intersection(*readable)
        fields = (
            [("selects", f"{subject}={value}") for subject, value in kept.selects]
            + [("excludes", f"{subject}={value}") for subject, value in kept.excludes]
            + list(spelled)
            + [("order_by", order) for order in kept.order_by]
        )
        missing = sorted(set(fields) - every)
        places: dict[tuple[str, str], dict[tuple, TermPlace]] = {key: {} for key in missing}

        def hold(key: tuple[str, str], place: TermPlace, by_position: bool) -> None:
            # Placed by where the term is, one fragment names places with other tests too: what else of the term's
            # text a statement holds.  Placed by text, a fragment has its tests, and the first is kept.
            held = places[key].setdefault(
                (place.fragment, place.term, _tests_of(place) if by_position else ()), place
            )
            # Two terms of one place naming the pair: either one ruling it out is enough.
            if place.decides and not held.decides:
                held.decides = True

        for run, build, _held in read:
            kept_nodes = [node for leaf in build.leaves if leaf.kept for node in _trail(leaf)]
            for key, ways in at(run, build, _along(kept_nodes)).items():
                for leaves, term, decides in ways if key in places else ():
                    hold(key, _placed(leaves, term, file_of, decides), True)

        def argued(text: str, term: str) -> list[TermPlace]:
            """Where a value the ``execute`` formats into the statement is written as *text*, with the tests that choose it.

            ``sql_read_files.format(order_by_policy, ...)``: the order is no
            fragment of the statement, it is ``order_by_policy``, and each
            of its writes under an ``if`` -- ``"lumiBlockNr,lfn"``,
            ``"boundaryID"`` -- is a build of its own.  Run, the value is a
            fragment, but the statements read are one per text: three
            branches writing ``"fileID"`` are one statement, and the other
            two ways it gets in are found here.
            """
            found: dict[tuple[str, str], TermPlace] = {}
            for run in variants:
                if not run.call.args:
                    continue
                execute = _statement_of(run.call)
                for node in walk(run.call.args[0]):
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "format":
                        handed = [*node.args, *(one.value for one in node.keywords)]
                    elif isinstance(node, ast.JoinedStr):
                        handed = [one.value for one in node.values if isinstance(one, ast.FormattedValue)]
                    else:
                        continue
                    for name in handed:
                        if not isinstance(name, ast.Name):
                            continue
                        for value in reaching_definitions(func, name.id, execute):
                            if not (isinstance(value, ast.Constant) and isinstance(value.value, str)):
                                continue
                            if _squeezed(value.value) != _squeezed(text):
                                continue
                            leaf = _Leaf(0, 0, site_of(value), (_Hop(func, run.call),))
                            place = _place(leaf, term, file_of)
                            if place.tests:
                                found.setdefault((place.fragment, place.term), place)
            return list(found.values())

        def named(key: tuple[str, str]) -> bool:
            """Whether a fragment appends *key*'s term with its values read across every branch."""
            return any(key in at(run, build, None) for run, build, _held in read)

        added: list[AddedTerm] = []
        for key in missing:
            field, said = key
            term = spelled.get(key, said)
            if field in ("order_by", "conditions"):
                for place in argued(said, term):
                    hold(key, place, True)
            chosen = [place for place in places[key].values() if place.tests]
            if chosen:
                added.append(AddedTerm(field=field, term=term, places=chosen))
            elif places[key]:
                added.append(
                    AddedTerm(
                        field=field,
                        term=term,
                        unplaced=(
                            "no test chooses it: the map reads several statements at this call, at "
                            + ", ".join(sorted({held[0] for held in places[key]}))
                        ),
                    )
                )
            elif named(key):
                added.append(
                    AddedTerm(
                        field=field,
                        term=term,
                        unplaced=(
                            "no build holds it: every statement putting the value into its bind "
                            "is on a branch the fragment naming the bind is not"
                        ),
                    )
                )
            elif field in ("order_by", "conditions") and (handed := argued(said, term)):
                added.append(AddedTerm(field=field, term=term, places=handed))
            else:
                added.append(AddedTerm(field=field, term=term, unplaced="the fragment appending it was not found"))
        read_trees = tuple(
            ConditionReading(entry=spelled[key], tree=trees[key])
            for key in missing
            if key[0] == "conditions" and trees.get(key) is not None
        )
        return tuple(added), read_trees

    def place_regions(
        func: ast.FunctionDef | ast.AsyncFunctionDef, owner: str, executions: list
    ) -> None:
        """Give each of *func*'s statements the stretch of it that it owns (:func:`sql.regions`).

        Every run of the function is laid out, not only the stored ones: the
        statement that ends a region may be one no read site was built from,
        and stopping at the next *stored* one would hand a reader the
        following query's logging as well.
        """
        for line, region in sql.regions(func, executions).items():
            kept = statements.get((owner, line))
            if kept is not None:
                statements[(owner, line)] = kept._replace(region=region)

    def place_exits(func: ast.FunctionDef | ast.AsyncFunctionDef, owner: str) -> None:
        """Give each statement the ways a row it asked for stops being followed.

        Two kinds, told apart by a prefix because they are different answers.
        A ``continue`` in the loop over the rows says *this row* was dropped
        here -- the request's own example, ``if int(id) in xmlIDs: continue``.
        A ``return`` under a test on the result says **nothing came back**,
        which is the answer when the query is not the thing that is wrong.

        Only the loop over *this statement's* rows, and only the exits taken
        in it.  Counting every loop inside the region instead files
        ``insertFilesForDataset_JEDI``'s 14 exits as "this row was skipped"
        when its query owns 363 lines and loops over its rows in 7 of them;
        counting by line range inside the right loop adds 19 more that belong
        to a loop nested in it, such as ``copyArchive@206``'s ``continue@245``
        about one of a job's files.

        The function's own exits are left out for the same reason the JOIN
        terms are left out of ``conditions``: an answer that names every error
        path in the function does not say where the row went.
        """
        for (own, line), kept in list(statements.items()):
            if own != owner or kept.region is None:
                continue
            _first, last = kept.region
            seeds = _rows_written(func, line, last)
            names = {name for node in seeds for name in _assigned_names(node)}
            # A loop over the cursor is over the rows itself; its target is a
            # row, and a loop over that is one nested in it.
            cursor_loops = {id(node) for node in seeds if isinstance(node, (ast.For, ast.AsyncFor))}
            enclosing = _enclosing_loops(func)
            spans: list[tuple[int, int]] = []
            over_rows: set[int] = set()
            for node in walk(func):
                if not isinstance(node, _LOOPS):
                    continue
                # The loop must **open** inside the region; only its end is
                # followed.  Searching the function for every loop over the
                # same names gives 174 exits over 71 statements against 90
                # over 45, because ``res`` is reused: ``copyArchive::main``'s
                # first query picks up the ``continue`` that belongs to its
                # query on line 381, and one handler on line 1364 lands on 19
                # statements.  The region is what makes ``res`` mean *this*
                # ``res``, and keeping the opening inside it keeps that.
                if not line < node.lineno <= last:
                    continue
                header = (
                    node.iter if isinstance(node, (ast.For, ast.AsyncFor)) else node.test
                )
                over = {n.id for n in walk(header) if isinstance(n, ast.Name)}
                if over & names or id(node) in cursor_loops:
                    spans.append((node.lineno, node.end_lineno or node.lineno))
                    over_rows.add(id(node))
            # Where the rows are worked through.  Kept as a field of its own
            # because it is not ``region``: a loop opened inside the region
            # runs to its own end, and for 64 of the corpus's 151 looping
            # statements that end is past where the next statement opens.
            handled = (
                (min(low for low, _ in spans), max(high for _, high in spans))
                if spans
                else None
            )
            found: set[str] = set()
            for node in walk(func):
                if not isinstance(node, _EXITS):
                    continue
                # The loop it is *taken in*, not the lines it falls between:
                # see :func:`_enclosing_loops`.  A row loop opens after this
                # statement ran, so membership already places the exit after
                # ``line`` -- and past ``last`` is exactly the case the
                # region alone was losing.
                taken_in = enclosing.get(id(node))
                if taken_in is None or id(taken_in) not in over_rows:
                    continue
                if isinstance(node, ast.ExceptHandler) and not _leaves(node):
                    continue
                found.add(f"{_EXIT_NAMES[type(node)]}@{node.lineno}")
            for node in walk(func):
                if not isinstance(node, ast.If) or not line < node.lineno <= last:
                    continue
                tested = {n.id for n in walk(node.test) if isinstance(n, ast.Name)}
                if not tested & names:
                    continue
                # The guarded branch only.  A test on the result splits the
                # code in two, and the ``else`` is the branch where the rows
                # *did* come back -- ``if resF is None or len(resF) == 0:``
                # logs "no files" and its ``else`` works through them, ending
                # in a ``break`` that means the opposite of this field.
                for inner in node.body:
                    if not isinstance(inner, (ast.Return, ast.Continue, ast.Break, ast.Raise)):
                        continue
                    # And inside the region, like every other exit: the guard
                    # can be in this statement's stretch while what it guards
                    # runs on into the next one's.
                    if not line <= inner.lineno <= last:
                        continue
                    found.add(f"no-rows:{_EXIT_NAMES[type(inner)]}@{inner.lineno}")
            statements[(owner, line)] = kept._replace(
                exits=tuple(sorted(found)), rows_handled=handled
            )

    def place_result(func: ast.FunctionDef | ast.AsyncFunctionDef, owner: str) -> None:
        """Say what each of *func*'s statements returns, read off every way it is built (:func:`sql.every_rendering`).

        One answer of every way and arm (:func:`sqltext.aggregated`), else
        ``varies``; none where a way is not read -- past the cap, or a text
        that does not parse as a ``SELECT``: a way not read could be either.
        """
        for (own, line), kept in list(statements.items()):
            if own != owner:
                continue
            drawn = sql.every_rendering(func, line)
            arms = [sqltext.aggregated(one.text) for one in drawn] if drawn else [None]
            if any(one is None for one in arms):
                continue
            got = frozenset().union(*arms)  # type: ignore[arg-type]
            result = RESULT_VARIES if len(got) > 1 else RESULT_AGGREGATE if True in got else RESULT_ROWS
            statements[(own, line)] = kept._replace(result=result)

    for module in modules:
        for func, _owner in functions_with_owner(module.tree):
            owner = f"{module.rel_path}::{func.name}"
            # Keyed by the window as well as the text.  A statement is emitted
            # once per call site, so deduplicating on the text alone keeps the
            # first pairing and throws the rest away -- which is exactly the
            # pairing the forwarded form exists to make.  ``copyArchive.main``
            # runs one ``jobsActive4`` query from three places with
            # ``:jobStatus`` bound to ``holding`` at each, and the text-only
            # key reported none of them.
            seen: set[tuple[str, Optional[tuple[int, int]]]] = set()
            # Read once and handed on.  ``executions`` is the most expensive
            # call in this walk, and the two passes below want the same list:
            # calling it again for each of them tripled the build.
            executions = sql.executions(func)
            # Runs that came before their line had a record, filed once every
            # run is read: whether a run's tables count must not hang on the
            # order the runs come in.
            unfiled: list[tuple[sql.Execution, ReadStatement, set[str]]] = []
            for run in executions:
                if (run.sql, run.window) in seen:
                    continue
                seen.add((run.sql, run.window))
                together: set[str] = set()
                for verb, table in _rows_touched(run.reading):
                    if _UNREAD_TABLE.fullmatch(table):
                        unreadable[owner] = unreadable.get(owner, 0) + 1
                        continue
                    touch(table, owner, verb)
                    if verb == "read":
                        # Joined entities stay keyed by the spec class: that
                        # reading asks which *other kind of thing* a decision
                        # looked at in the same statement, and there the three
                        # job tables really are one kind of thing.
                        spec_class = attributor.class_for_table(table)
                        if spec_class is not None:
                            together.add(spec_class)
                if len(together) > 1:
                    joins.setdefault(owner, set()).add(frozenset(together))
                # Built before the verbs so ``record`` has somewhere to file a
                # pair, and thrown away again if none lands.  The line is the
                # one the statement *runs* on: the text is assembled further up
                # and the binds are not in scope there.
                # Read for both call forms: where a walk should start.  The
                # binds are paired with the run by the call, not by this
                # window -- see ``sql.bound_values``.
                window = run.window
                if window is None and run.varmap:
                    window = sql.binding_window(func, run.varmap, run.call.lineno)
                if window is None and sql.inline_binds(run.call):
                    # ``querySQLS(sql, {":name": name})``: the binds are
                    # filled in the call itself.
                    window = (run.call.lineno, run.call.end_lineno or run.call.lineno)
                reading = sqltext.terms(run.reading)
                at = ReadStatement(
                    owner=owner,
                    package=module.package,
                    file=module.rel_path,
                    blob_sha=module.blob_sha,
                    line=run.call.lineno,
                    window=window,
                    selects=set(),
                    excludes=set(),
                    consults=set(),
                    keys=set(),
                    order_by=set(),
                    limits=set(),
                    tables=set(),
                    conditions=[],
                    joins=set(),
                    unresolved=tuple(
                        sorted(
                            {f"{stop.source} -- {stop.reason}" for stop in run.unresolved}
                            | set(reading.unresolved)
                        )
                    ),
                )
                read_tables = {table for verb, table in _table_verbs(run.reading) if verb == "read"}
                # Each term once, on the table it belongs to and under the verb
                # whose rows it picks.  Read off the text, one term was filed
                # against every table the statement named and under every verb:
                # ``tabD.status`` came out as a reader of task statuses, and a
                # ``NOT EXISTS`` subquery's value as one the statement asks for.
                for term in reading.terms:
                    subject = _term_subject(attributor, term)
                    if term.scope in (sqltext.NEGATED, sqltext.PICKS):
                        continue
                    if term.scope == sqltext.CONSULT:
                        if not term.excluding:
                            for key in term.binds:
                                for value in bound(func, run, key, module.rel_path):
                                    consult(subject, value, at)
                            for value in term.literals:
                                consult(subject, value, at)
                        continue
                    if term.excluding:
                        if term.verb == "read":
                            unsaid = term.undecided
                            for key in term.binds:
                                values_of = bound(func, run, key, module.rel_path)
                                unsaid = unsaid or not values_of
                                for value in values_of:
                                    exclude(subject, value, at)
                            for value in term.literals:
                                exclude(subject, value, at)
                            # ``NOT IN (SELECT ...)``, a hole, a bind nothing
                            # here fills: the statement passes over values the
                            # map cannot name, and a reader asking whether its
                            # value is one of them has to be told it might be.
                            if unsaid:
                                exclude(subject, "{}", at)
                        continue
                    for key in term.binds:
                        values_of = bound(func, run, key, module.rel_path)
                        for value in values_of:
                            record(subject, value, owner, term.verb, at)
                        if not values_of and term.verb == "read":
                            statements.setdefault((at.owner, at.line), at).keys.add(subject)
                    for value in term.literals:
                        record(subject, value, owner, term.verb, at)
                ways = _outright_terms(func, run)
                refuses_here = run_refuses(func, run, module.rel_path, ways)
                site = (owner, run.call.lineno)
                refused[site] = (
                    (refuses_here or set()) & refused[site]
                    if site in refused
                    else (refuses_here or set())
                )
                asked = run_restricts(func, run, module.rel_path, ways) or {}
                restricted[site] = _restricted_both(restricted[site], asked) if site in restricted else asked
                if not reading.parsed and read_tables:
                    # Not read some other way, and not dropped either: a
                    # statement whose text does not parse still reads these
                    # tables, and ``unresolved`` says why nothing else is known.
                    statements.setdefault((at.owner, at.line), at)
                # After the terms, and onto whichever record was kept: a join
                # reads two tables in one statement, and the second one would
                # otherwise land on the copy ``setdefault`` discarded.
                if site in statements:
                    file_run(func, run, at, read_tables, module.rel_path)
                else:
                    unfiled.append((run, at, read_tables))
            for run, at, read_tables in unfiled:
                if (at.owner, at.line) in statements:
                    file_run(func, run, at, read_tables, module.rel_path)
            place_conditions(func, owner, module.rel_path, executions)
            place_prints(func, owner, executions)
            place_orders(func, owner, module.rel_path, executions)
            place_added_under(func, owner, module.rel_path, executions)
            # Regions first: the exits are read inside the stretch it settles.
            place_regions(func, owner, executions)
            place_exits(func, owner)
            place_result(func, owner)
            place_routes(func, owner, module.rel_path, executions)
    for site, at in statements.items():
        statements[site] = at._replace(
            refuses=frozenset(refused.get(site, ())),
            restricts=frozenset(_restricted_pairs(restricted.get(site, {}))),
        )
    _check_as_statement(as_statement, statements)
    return ReadSide(
        values=found,
        entities=rows,
        unreadable=unreadable,
        joins=joins,
        statements=statements,
        watch=watch,
    )


def read_site_nodes(
    statements: dict[tuple[str, int], "ReadStatement"],
    map_id: str,
    derived_from: str,
) -> list[ReadSiteNode]:
    """Turn the statements :func:`read_side` saw into one node each.

    No promotion and no criterion.  A statement that asks for rows by a value
    the map already holds a subject for is somewhere an investigation can be
    sent, and there is nothing further to judge -- which is the difference
    between this and :func:`entity_nodes`, where "is this worth asking about"
    is a question about a column.

    ``selects`` is rendered rather than kept as pairs because it is read, not
    matched: the pair is already in ``SubjectNode.selected_by``, and what this
    row adds is which statement, in what company.
    """
    nodes: list[ReadSiteNode] = []
    for (owner, line), at in sorted(statements.items()):
        nodes.append(
            ReadSiteNode(
                map_id=map_id,
                derived_from=derived_from,
                name=ReadSiteNode.make_name(map_id, owner, line),
                owner=owner,
                selects=sorted(f"{subject}={value}" for subject, value in at.selects),
                excludes=sorted(f"{subject}={value}" for subject, value in at.excludes),
                refuses=sorted(f"{subject}={value}" for subject, value in at.refuses),
                restricts=sorted(f"{subject}={value}" for subject, value in at.restricts),
                consults=sorted(f"{subject}={value}" for subject, value in at.consults),
                keys=sorted(at.keys),
                order_by=sorted(at.order_by),
                limits=sorted(at.limits),
                call_readings=list(at.call_readings),
                callers_complete=at.callers_complete,
                callers_stop=list(at.callers_stop),
                tables=sorted(at.tables),
                joins=[
                    JoinPair(a=a, b=b, a_source=a_source, b_source=b_source, a_attr=a_attr, b_attr=b_attr)
                    for a, b, a_source, b_source, a_attr, b_attr in sorted(at.joins or ())
                ],
                # Empty only where the call names no varmap to read -- one
                # statement in the corpus, ``querySQLS(sql.format(t), {})``,
                # whose values are all literals in the text.  There is no
                # assignment anywhere to place, so an empty window is the
                # true answer rather than a gap.
                bind_window=list(at.window) if at.window else [],
                conditions=sorted(at.conditions),
                added_under=list(at.added_under),
                condition_trees=sorted(at.condition_trees, key=lambda one: one.entry),
                printed_by=list(at.printed_by),
                order_holes=list(at.order_holes),
                unresolved=list(at.unresolved),
                region=list(at.region) if at.region else [],
                exits=list(at.exits),
                rows_handled=list(at.rows_handled) if at.rows_handled else [],
                result=at.result,
                anchor=Anchor(
                    package=at.package,
                    file=at.file,
                    line_start=line,
                    line_end=line,
                    blob_sha=at.blob_sha,
                ),
            )
        )
    return nodes


class CrossingCensus(NamedTuple):
    """What :func:`attach_crossings` resolved, and what it could not.

    ``handing_back`` counts the statements whose rows reach a ``return``, and
    ``resolved`` the ones where the caller then does something the map holds a
    junction for.  The difference is the shape the map can see and cannot
    answer -- a number worth printing, because an empty ``crosses_to`` on a
    statement that plainly hands its rows on reads as "nowhere" otherwise.
    ``opened`` counts the statements with a call on the way, or a caller
    back, this reading cannot place (``crosses_open``).
    """

    stored: int
    handing_back: int
    resolved: int
    opened: int


#: How a row leaves the name the statement first bound it to.  Assignment and
#: a loop target are the obvious two; the rest are how the corpus writes "put
#: this row in the bucket it belongs to" before handing the bucket back.
_CARRIERS = ("append", "add", "update", "extend", "setdefault")




#: How many steps of an access path a holding tells apart -- ``res[0].Files[*].lfn`` is four.  A longer path is cut
#: to its first steps and :data:`_MORE`, any part below them.
_DEPTH = 5
#: The steps of an access path: ``.f`` a field, ``[0]`` / ``['k']`` an index or key a literal names, and
#: ``[*]`` any element of a list, a set or a dict's values, ``[key]`` a dict's keys, ``.*`` any field (a field
#: named by a value: ``setattr(x, n, v)``), ``...`` anything below the cut.
_ALL, _KEY, _MORE, _ANY_FIELD = "[*]", "[key]", "...", ".*"
#: The step for any field of an object: what a call filling it by name (``spec.pack(row)``) puts its value at.
ANY_FIELD = _ANY_FIELD
_Paths = frozenset[tuple[str, ...]]
#: The value itself holds the rows.
_WHOLE: _Paths = frozenset({()})
_NOTHING: _Paths = frozenset()

#: Calls a value goes into without being used: a log line.
_LOG_METHODS = frozenset({"debug", "info", "warning", "warn", "error", "exception", "critical", "sendMsg", "keepMsg", "log"})
#: Calls making a logger whose tag is the value: the tag is a log line's.  ``create_tagged_logger`` tests ``if tag is
#: None`` and ``MsgWrapper.__init__`` its token, which read as using it.
_LOG_MAKERS = frozenset({"create_tagged_logger", "MsgWrapper", "LogWrapper"})
#: What a ``+`` is of, as its operand is written (U1): a list, a number, a text -- or not written (``None``).
_SEQUENCE, _NUMBER, _TEXT = "sequence", "number", "text"
#: Calls reading a value again out of a text -- a column's content parsed, decoded: a value of its own (U1).
_READ_AGAIN = frozenset(
    {"loads", "load", "literal_eval", "safe_load", "b64decode", "decode", "unquote", "fromisoformat", "strptime"}
)
#: Builtins and modules computing a number from what they are handed (U1).
_COMPUTING = frozenset(
    {"len", "sum", "round", "abs", "divmod", "pow", "min", "max", "any", "all", "ord", "chr", "hash", "bool", "range"}
)
_COMPUTING_MODULES = frozenset({"math", "statistics", "numpy", "np"})


def _unread_call(call: ast.Call, name: str) -> bool:
    """Whether *call*, which no reading places, makes a value the map cannot read (U1's 4): another system's answer, a
    function of the corpus or a library -- not a builtin or a module computing a number, a value read again out of a
    text, a text made by a str method."""
    if name in _READ_AGAIN or name in _COMPUTING or name in ("join", "format"):
        return False
    if isinstance(call.func, ast.Attribute):
        receiver = call.func.value
        if isinstance(receiver, ast.Constant):
            return False
        if isinstance(receiver, ast.Name) and receiver.id in _COMPUTING_MODULES:
            return False
    return True


def _operand(expression: ast.AST, names: Optional[dict[str, str]] = None) -> Optional[str]:
    """What an operand of ``+`` is, as written: a list or tuple made here, a number, a text, a name the function says
    one of them of (*names*: :func:`_name_kinds`) -- ``None`` where the code does not say (a field, a call's value)."""
    if isinstance(expression, ast.Name):
        return (names or {}).get(expression.id)
    if isinstance(expression, ast.Subscript) and isinstance(expression.value, ast.Name):
        return (names or {}).get(expression.value.id + _ELEMENT)
    if isinstance(expression, (ast.List, ast.Tuple, ast.ListComp, ast.GeneratorExp)):
        return _SEQUENCE
    if isinstance(expression, ast.Constant):
        if isinstance(expression.value, str):
            return _TEXT
        if isinstance(expression.value, (int, float)) and not isinstance(expression.value, bool):
            return _NUMBER
        return None
    if isinstance(expression, ast.JoinedStr):
        return _TEXT
    if isinstance(expression, ast.Call):
        name = _callee_name(expression)
        if name in ("str", "repr", "format", "join"):
            return _TEXT
        if name in ("int", "float", "len", "sum", "round", "abs"):
            return _NUMBER
        if name in ("list", "tuple", "sorted"):
            return _SEQUENCE
    if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Add):
        kinds = {_operand(expression.left, names), _operand(expression.right, names)}
        return next((kind for kind in (_SEQUENCE, _TEXT, _NUMBER) if kind in kinds), None)
    return None


def _summands(expression: ast.BinOp) -> list[ast.expr]:
    """The operands of a chain of ``+`` (``a + b + c``: ``[a, b, c]``)."""
    found: list[ast.expr] = []
    for side in (expression.left, expression.right):
        if isinstance(side, ast.BinOp) and isinstance(side.op, ast.Add):
            found.extend(_summands(side))
        else:
            found.append(side)
    return found


def _annotation_kind(annotation: Optional[ast.AST]) -> Optional[str]:
    """What an annotation says a value is, as :func:`_operand` names it: ``list[...]``, ``int``, ``Optional[str]``."""
    if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
        sides = [
            side
            for side in (annotation.left, annotation.right)
            if not (isinstance(side, ast.Constant) and side.value is None)
        ]
        return _annotation_kind(sides[0]) if len(sides) == 1 else None
    head = annotation.value if isinstance(annotation, ast.Subscript) else annotation
    name = head.attr if isinstance(head, ast.Attribute) else getattr(head, "id", None)
    if name == "Optional" and isinstance(annotation, ast.Subscript):
        return _annotation_kind(annotation.slice)
    if name in ("list", "List", "tuple", "Tuple", "Sequence"):
        return _SEQUENCE
    if name in ("int", "float"):
        return _NUMBER
    return _TEXT if name == "str" else None


#: The key :func:`_name_kinds` says what a name's elements are under: ``d[k] = []`` makes ``d[k]`` a list.
_ELEMENT = "[]"


def _name_kinds(func: ast.AST) -> dict[str, str]:
    """What each name of *func* is, for a ``+`` of it (U1): what its annotation, or the values it is bound to where
    the code says, are -- one of them, every one that says agreeing (``files = []`` then ``files = self.get()`` is a
    list); a loop over ``range(...)`` binds numbers; ``d[k] = []`` says what ``d``'s elements are.  A name bound to
    two kinds is none."""
    said: dict[str, set[Optional[str]]] = {}
    made: dict[str, list[ast.AST]] = {}
    arguments = getattr(func, "args", None)
    if arguments is not None:
        for one in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs):
            said.setdefault(one.arg, set()).add(_annotation_kind(one.annotation))
    for node in own_nodes(func):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    made.setdefault(target.id, []).append(node.value)
                elif isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                    made.setdefault(target.value.id + _ELEMENT, []).append(node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            said.setdefault(node.target.id, set()).add(_annotation_kind(node.annotation))
            if node.value is not None:
                made.setdefault(node.target.id, []).append(node.value)
        elif (
            isinstance(node, (ast.For, ast.AsyncFor))
            and isinstance(node.target, ast.Name)
            and isinstance(node.iter, ast.Call)
            and _callee_name(node.iter) == "range"
        ):
            said.setdefault(node.target.id, set()).add(_NUMBER)
    kinds: dict[str, str] = {}
    for _ in range(len(made) + 1):  # each round settles at least one name more, or the kinds stand
        found = {
            name: (said.get(name, set()) | {_operand(value, kinds) for value in made.get(name, [])}) - {None}
            for name in set(said) | set(made)
        }
        settled = {name: next(iter(kind)) for name, kind in found.items() if len(kind) == 1}
        if settled == kinds:
            break
        kinds = settled
    return kinds


#: Builtins whose value is what they are handed, elements and all.
_KEEPING = frozenset({"list", "tuple", "set", "frozenset", "sorted", "reversed", "iter", "dict"})
#: Builtins handing on what they are handed, or its elements: read where their value goes.
_PASSING_ON = _KEEPING | {"next", "enumerate", "zip", "filter", "map"}
#: Builtins that tell nothing of the value: its type, whether it has a part.
_TELLING_NOTHING = frozenset({"isinstance", "issubclass", "hasattr", "callable", "id", "type"})
#: ``copy.copy`` / ``copy.deepcopy``: the value again, not a use of it.
_COPIES = frozenset({"copy", "deepcopy"})
#: Calls that write a value out whole -- every part of it read.
_SERIALIZING = frozenset({"dumps", "str", "repr"})
#: A builtin container's methods handing back the container, and one of its elements.
_CONTAINER_KEEPING = frozenset({"copy"})
_CONTAINER_ELEMENT = frozenset({"get", "pop", "popitem", "setdefault"})
#: Iterator helpers outside the corpus (itertools', pandacommon's ``batched``)
#: whose value is made of the first argument's elements -- in chunks (``batched(items, chunk_size)``), or as they
#: come (``islice``) -- and, for ``chain``, of every argument's; a count handed beside them makes nothing.
_CHUNKING = frozenset({"batched"})
_SLICING = frozenset({"islice"})
_CHAINING = frozenset({"chain"})


def _cut(path: tuple[str, ...]) -> tuple[str, ...]:
    return path if len(path) <= _DEPTH else path[:_DEPTH] + (_MORE,)


def _bare(path: tuple[str, ...]) -> tuple[str, ...]:
    """A path cut at its first step: the whole value, any part of which may hold them."""
    return () if path and path[0] == _MORE else path


def _below(steps: Iterable[str], paths: Iterable[tuple[str, ...]]) -> _Paths:
    """What a value holds where it is put at *steps* of another, it holding *paths*."""
    return frozenset(_cut(tuple(steps) + _bare(path)) for path in paths)


def _same_step(head: str, step: str) -> bool:
    if head == step:
        return True
    if _ANY_FIELD in (head, step):
        return head.startswith(".") and step.startswith(".")
    if _KEY in (head, step):
        return False
    return (head == _ALL and step.startswith("[")) or (step == _ALL and head.startswith("["))


def _under(step: str, paths: Iterable[tuple[str, ...]]) -> _Paths:
    """What the part *step* of a value holds, the value holding *paths*: the whole value holds every part."""
    found = set()
    for path in paths:
        path = _bare(path)
        if not path:
            found.add(())
        elif _same_step(path[0], step):
            found.add(_bare(path[1:]))
    return frozenset(found)


def _elements(paths: _Paths) -> _Paths:
    """What an element a loop or a comprehension takes holds, of a value not known as a dict (:func:`_keys`): of a list
    or a set, or a dict's key."""
    return _under(_ALL, paths) | _under(_KEY, paths)


def _keys(paths: _Paths) -> _Paths:
    """What an element of a value known as a dict holds: its key, not what its values hold (``[*]`` is both a
    list's element and the value under a key not known before it runs)."""
    return _under(_KEY, paths)


#: Calls making a dict.
_DICT_MAKERS = frozenset({"dict", "defaultdict", "OrderedDict", "Counter"})
#: Annotations naming a dict.
_DICT_TYPES = _DICT_MAKERS | {"Dict", "DefaultDict", "Mapping", "MutableMapping"}


def _dict_type(annotation: Optional[ast.AST]) -> bool:
    """Whether *annotation* names a dict -- or a dict or ``None``."""
    if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
        sides = [
            side
            for side in (annotation.left, annotation.right)
            if not (isinstance(side, ast.Constant) and side.value is None)
        ]
        return len(sides) == 1 and _dict_type(sides[0])
    head = annotation.value if isinstance(annotation, ast.Subscript) else annotation
    name = head.attr if isinstance(head, ast.Attribute) else getattr(head, "id", None)
    if name == "Optional" and isinstance(annotation, ast.Subscript):
        return _dict_type(annotation.slice)
    return name in _DICT_TYPES


def _is_dict(expression: Optional[ast.AST], dicts: frozenset[str]) -> bool:
    """Whether *expression*'s value is a dict: made here, or a name of *dicts* (:func:`_dict_names`)."""
    if isinstance(expression, (ast.Dict, ast.DictComp)):
        return True
    if isinstance(expression, ast.Name):
        return expression.id in dicts
    if isinstance(expression, ast.Call):
        name = _callee_name(expression)
        if name in _DICT_MAKERS:
            return True
        return name == "copy" and isinstance(expression.func, ast.Attribute) and _is_dict(expression.func.value, dicts)
    if isinstance(expression, ast.IfExp):
        return _is_dict(expression.body, dicts) and _is_dict(expression.orelse, dicts)
    if isinstance(expression, ast.BoolOp):
        return all(_is_dict(value, dicts) for value in expression.values)
    return False


def _loop_iterables(func: ast.AST, dicts: frozenset[str]) -> dict[tuple[int, str], ast.expr]:
    """``{(id(statement), name): iterable}``: for each assignment inside a loop whose target *name* only loops bind
    in *func* (:func:`attribution.loop_only_names`), the innermost such loop's iterable -- not one known as a dict,
    whose element is a key."""
    names = loop_only_names(func)
    found: dict[tuple[int, str], ast.expr] = {}
    for loop in walk(func):
        if (
            not isinstance(loop, (ast.For, ast.AsyncFor))
            or not isinstance(loop.target, ast.Name)
            or loop.target.id not in names
            or _is_dict(loop.iter, dicts)
        ):
            continue
        for statement in loop.body:
            for node in walk(statement):
                if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                    found[(id(node), loop.target.id)] = loop.iter
    return found


def _dict_names(func: ast.AST) -> frozenset[str]:
    """The names *func* holds dicts in: annotated as one (``**kwargs`` among them), or bound only to values making
    one."""
    known: set[str] = set()
    bound: dict[str, list[Optional[ast.AST]]] = {}

    def binds(target: ast.AST, value: Optional[ast.AST]) -> None:
        if isinstance(target, ast.Name):
            bound.setdefault(target.id, []).append(value)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                binds(element, None)
        elif isinstance(target, ast.Starred):
            binds(target.value, None)

    arguments = getattr(func, "args", None)
    if arguments is not None:
        for one in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs, arguments.vararg):
            if one is not None:
                bound.setdefault(one.arg, []).append(None)
                if one is not arguments.vararg and _dict_type(one.annotation):
                    known.add(one.arg)
        if arguments.kwarg is not None:
            known.add(arguments.kwarg.arg)
    for node in own_nodes(func):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                binds(target, node.value)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and _dict_type(node.annotation):
                known.add(node.target.id)
            if node.value is not None:
                binds(node.target, node.value)
        elif isinstance(node, ast.NamedExpr):
            binds(node.target, node.value)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            binds(node.target, None)
        elif isinstance(node, ast.withitem) and node.optional_vars is not None:
            binds(node.optional_vars, None)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.setdefault(node.name, []).append(None)
    growing = True
    while growing:
        growing = False
        for name, made in bound.items():
            if name not in known and all(_is_dict(value, frozenset(known)) for value in made):
                known.add(name)
                growing = True
    return frozenset(known)


def _index_step(node: ast.AST) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return f"[{node.value}]"
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return f"[{node.value!r}]"
    return _ALL


def _in_elements(paths: _Paths) -> bool:
    """Whether a value holds the rows only in its elements -- a list or tuple of them, never a number or a text."""
    return bool(paths) and all(path and path[0] == _ALL for path in paths)


def _whole(paths: Iterable[tuple[str, ...]]) -> bool:
    """Whether the value itself holds the rows -- not only a part of it."""
    return any(not _bare(path) for path in paths)


def _any_field(paths: _Paths) -> _Paths:
    """What some one field of a value holds: ``getattr(x, name)``, the name not a literal."""
    fields = {_bare(path[1:]) for path in map(_bare, paths) if path and path[0].startswith(".")}
    return frozenset(fields | ({()} if _whole(paths) else set()))


def _access_path(expression: Optional[ast.AST]) -> Optional[tuple[str, tuple[str, ...]]]:
    """``(name, steps)`` of ``x.a[k].b``, or ``None`` where it is rooted at no name."""
    steps = []
    while True:
        if isinstance(expression, ast.Attribute):
            steps.append("." + expression.attr)
            expression = expression.value
        elif isinstance(expression, ast.Subscript) and not isinstance(expression.slice, ast.Slice):
            steps.append(_index_step(expression.slice))
            expression = expression.value
        else:
            break
    return (expression.id, tuple(reversed(steps))) if isinstance(expression, ast.Name) else None


def _callee_name(call: ast.Call) -> Optional[str]:
    callee = call.func
    return callee.id if isinstance(callee, ast.Name) else callee.attr if isinstance(callee, ast.Attribute) else None


def _setattr_parts(call: ast.Call) -> Optional[tuple[ast.expr, ast.expr]]:
    """The object and the value of ``setattr(x, n, v)`` / ``object.__setattr__(x, n, v)``."""
    name = _callee_name(call)
    if name not in ("setattr", "__setattr__") or len(call.args) != 3:
        return None
    return call.args[0], call.args[2]


def _own_calls(func: ast.AST) -> list[ast.Call]:
    """The calls *func* makes itself; a function, lambda or class nested in it runs elsewhere."""
    return [node for node in own_nodes(func) if isinstance(node, ast.Call)]


def _same(paths: _Paths) -> _Paths:
    return paths


class _Write:
    """One write of a name in a function: the steps it writes at, where the value comes from and how it is taken
    apart (*take*: a target unpacked, a loop's element), or the call keeping something in what the name holds."""

    __slots__ = ("name", "node", "steps", "source", "take", "adds", "call", "forced", "paths", "mentions", "inner_calls")

    def __init__(
        self,
        name: str,
        node: ast.AST,
        steps: tuple[str, ...] = (),
        source: Optional[ast.AST] = None,
        take: Optional[Callable[[_Paths], _Paths]] = None,
        adds: bool = False,
        call: Optional[ast.Call] = None,
    ):
        self.name, self.node, self.steps, self.source, self.take = name, node, tuple(steps), source, take
        self.adds, self.call = adds, call
        #: What the write holds whatever it is handed: a parameter given the rows, a statement's own seed.
        self.forced: _Paths = _NOTHING
        self.paths: _Paths = _NOTHING
        looked = call if call is not None else source
        self.mentions = (
            frozenset(n.id for n in walk(looked) if isinstance(n, ast.Name)) if looked is not None else frozenset()
        )
        self.inner_calls = (
            frozenset(id(n) for n in walk(looked) if isinstance(n, ast.Call)) if looked is not None else frozenset()
        )

    def fresh(self) -> "_Write":
        """The write again, holding nothing: one function read for another way in."""
        other = _Write.__new__(_Write)
        for slot in _Write.__slots__:
            setattr(other, slot, getattr(self, slot))
        other.forced, other.paths = _NOTHING, _NOTHING
        return other


class _Summary(NamedTuple):
    """What a definition does with what a call hands it: the parts of its value that hold the rows, the parts of
    what each parameter names it puts them into, and the carry it was read with (``None`` while it is read)."""

    returns: _Paths
    kept: dict[str, _Paths]
    carry: Optional["_PathCarry"]


#: Where a select item sits, ``{}`` its position, in what a call giving a statement's rows gives: one row's of a
#: ``fetchone``, each row's of a ``fetchall`` / ``fetchmany`` and of the task buffer's ``querySQL``, the rows' of
#: ``querySQLS``'s ``(status, rows)``.
_ITEM_AT: dict[str, tuple[str, ...]] = {
    "fetchone": ("[{}]",),
    "fetchall": (_ALL, "[{}]"),
    "fetchmany": (_ALL, "[{}]"),
    "querySQL": (_ALL, "[{}]"),
    "querySQLS": ("[1]", _ALL, "[{}]"),
}


def _item_seed(write: _Write, item: int) -> _Paths:
    """Where select item *item* of a statement's rows sits in what its seed *write* holds: a loop over the cursor
    takes a row each pass, an assignment what its value gives (:func:`_item_in`), each taken apart as its target
    is."""
    if isinstance(write.node, (ast.For, ast.AsyncFor)):
        placed: _Paths = frozenset({(_ALL, f"[{item}]")})
    else:
        placed = _item_in(write.source, item)
    if not placed:
        return _NOTHING
    return write.take(placed) if write.take is not None else placed


def _item_in(value: Optional[ast.AST], item: int) -> _Paths:
    """Where select item *item* sits in *value*'s value: a fetch's (:data:`_ITEM_AT`), a part of one, a number or a
    text made of one (``int(self.cur.fetchone()[0])``); nowhere in any other value."""
    while (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id in ("int", "str", "float")
        and len(value.args) == 1
        and not value.keywords
    ):
        value = value.args[0]
    if isinstance(value, ast.Subscript) and not isinstance(value.slice, ast.Slice):
        return _under(_index_step(value.slice), _item_in(value.value, item))
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) and value.func.attr in _ITEM_AT:
        return frozenset({tuple(step.format(item) for step in _ITEM_AT[value.func.attr])})
    return _NOTHING


class _PathCarry:
    """Which parts of which names hold the rows in one function, as it runs: a holding is a name and an access path.

    The rows come in through the statement's own seeds (*start*: a write holds them whole), the parameters a call
    hands parts of them to (*seeds*), or the value of a call the function made (*call_values*: the way back).  A
    write holds what its value holds, put at its steps -- ``x.a[k] = v`` puts what ``v`` holds at ``.a[*]`` of ``x``
    -- and taken apart as its target is: ``a, b = v`` gives ``a`` what ``v[0]`` holds, a loop's target what an
    element does, and an element of a value known as a dict is its key (:func:`_dict_names`), wherever one is taken:
    ``for k in sorted(d)`` takes what ``d``'s keys hold, not what its values do.  A read takes the part it names:
    ``x.a`` holds what ``.a`` of ``x`` holds, and the whole value holds every part.  Which writes of a name reach a
    read is ``pathcond.surviving_writes`` as the function runs; a write that puts something into a name (``x.a = v``,
    ``d[k] = v``, ``x.append(v)``, ``x += v``) adds to it, and a key it puts in is one of the keys there -- each
    subscript written through (``d[a][b] = v`` puts ``a`` and ``b``), ``d[k].append(v)``, ``d.setdefault(k, v)``.

    A call's value is read from where it lands (:class:`_Reading`): what the definition returns of what it is
    handed, what a constructor keeps in its object.  A call no reading places is a builtin's or a hole: a container
    method takes the part it names, a builtin of :data:`_KEEPING` keeps its argument, an iterator helper outside the
    corpus (``batched``, ``islice``, ``chain``) is made of its iterables' elements -- not of the count beside them --
    and any other call's value holds the rows where something it is handed holds them whole -- a part of them
    handed to a hole makes nothing of them.  What the reach record reads as the database gives nothing back of this
    statement's rows: its answer is another run's (a call inside it, a bind's ``int(row[0])``, is read as any
    other).  A call placed in the corpus can put what it is handed into an object named here (its summary's
    *kept*): ``fileSpec.pack(row)`` fills ``fileSpec``, ``job.addFile(fileSpec)`` appends it to ``job.Files``.  So
    does a method the object's class gets from a base outside the corpus, with all it is handed:
    ``dc_req_spec.pack(row)`` on pandacommon's ``SpecBase`` fills ``dc_req_spec``.

    With *item*, the statement's seeds hold its select item of that position where it sits in what they are given,
    not the rows whole (:func:`_item_seed`): a value held whole is that column of a row.
    """

    def __init__(
        self,
        reading: "_Reading",
        file: str,
        func: ast.AST,
        bound: tuple[tuple[str, ast.ClassDef], ...] = (),
        seeds: Optional[dict[str, _Paths]] = None,
        start: Iterable[ast.AST] = (),
        call_values: Optional[dict[int, _Paths]] = None,
        deciding: bool = False,
        item: Optional[int] = None,
    ):
        self.reading, self.file, self.func, self.bound = reading, file, func, bound
        self.call_values = call_values or {}
        #: Read for what the rows decide rather than what carries them (U1): a value computed from them -- a
        #: comparison, a count, a flag -- is made from them, so a test on it reads them (:meth:`_Reading.reads_in`).
        self.deciding = deciding
        #: A value the map cannot read (U1's 4 and 5: a call no reading places, a + of operands whose type is not
        #: written) held as before U1 -- what the rows reach through one is said apart (:attr:`_Reading.unread_kept`).
        self.unread_kept = reading.unread_kept
        self.names = reading.names_of(func)
        self._made = (seeds, tuple(start))
        self.dicts = reading.dicts_of(func)
        self.writes: list[_Write] = []
        self.by_name: dict[str, list[_Write]] = {}
        for write in reading.writes_of(func, deciding):
            self._add(write.fresh())
        arguments = getattr(func, "args", None)
        seeds = seeds or {}
        start_ids = {id(node) for node in start}
        for write in self.writes:
            if write.node is arguments and write.source is None and write.call is None and write.name in seeds:
                write.forced = frozenset(seeds[write.name])
            elif (
                id(write.node) in start_ids
                and write.call is None
                # A ``+=`` started from writes its field with what the reader follows, though it holds no rows.
                and (write.source is not None or isinstance(write.node, ast.AugAssign))
            ):
                write.forced = _below(write.steps, _WHOLE if item is None else _item_seed(write, item))
        self._grow()

    def _add(self, write: _Write) -> None:
        self.writes.append(write)
        self.by_name.setdefault(write.name, []).append(write)

    # -- the writes ----------------------------------------------------------

    @staticmethod
    def read_writes(
        func: ast.AST,
        dicts: frozenset[str] = frozenset(),
        deciding: bool = False,
        unread_kept: bool = False,
        names: Optional[dict[str, str]] = None,
    ) -> list[_Write]:
        """Every write of a name in *func*, its parameters last, holding nothing yet; *dicts* the names it holds
        dicts in."""
        carry = _PathCarry.__new__(_PathCarry)
        carry.writes, carry.by_name, carry.dicts, carry.deciding = [], {}, dicts, deciding
        carry.unread_kept, carry.names = unread_kept, names if names is not None else _name_kinds(func)
        carry._read(func)
        arguments = getattr(func, "args", None)
        if arguments is not None:
            for one in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs, arguments.vararg, arguments.kwarg):
                if one is not None:
                    carry._add(_Write(one.arg, arguments))
        return carry.writes

    def _target(
        self, target: ast.AST, node: ast.AST, source: ast.AST, take: Callable[[_Paths], _Paths], adds: bool
    ) -> None:
        if isinstance(target, ast.Name):
            self._add(_Write(target.id, node, (), source, take, adds))
        elif isinstance(target, (ast.Tuple, ast.List)):
            for index, element in enumerate(target.elts):
                if isinstance(element, ast.Starred):
                    self._target(element.value, node, source, lambda p, t=take: _below((_ALL,), _under(_ALL, t(p))), adds)
                else:
                    self._target(element, node, source, lambda p, t=take, s=f"[{index}]": _under(s, t(p)), adds)
        elif isinstance(target, (ast.Attribute, ast.Subscript)):
            found = _access_path(target)
            if found is not None:
                base, steps = found
                self._add(_Write(base, node, steps, source, take, True))
                self._into_container(base, steps, node, source, take, {base})
            self._keys_put(target, node)

    def _into_container(
        self,
        name: str,
        steps: tuple[str, ...],
        node: ast.AST,
        source: ast.AST,
        take: Callable[[_Paths], _Paths],
        seen: set[str],
    ) -> None:
        """A field written on an element a loop took is written in the loop's iterable too -- the element is the
        container's: ``for f in job.Files: f.status = v`` puts what ``v`` holds at ``.Files[*].status`` of ``job``,
        and on up where ``job`` is a loop's element as well.  Only for a name loops alone bind
        (:func:`attribution.loop_only_names`), of an iterable not known as a dict that is a name or an access
        path of one."""
        found = _access_path(self._iterables.get((id(node), name)))
        if found is None or found[0] in seen:
            return
        base, inner = found
        steps = inner + (_ALL,) + steps
        self._add(_Write(base, node, steps, source, take, True))
        self._into_container(base, steps, node, source, take, seen | {base})

    def _augmented_kinds(self, node: ast.AugAssign) -> set[Optional[str]]:
        """What the two sides of ``x += v`` are (:func:`_operand`)."""
        return {_operand(node.target, self.names), _operand(node.value, self.names)}

    @staticmethod
    def _target_writes(target: ast.AST, node: ast.AST) -> list[_Write]:
        """The writes of an augmented assignment computing a new value: holding nothing, a name's replacing it."""
        if isinstance(target, ast.Name):
            return [_Write(target.id, node)]
        found = _access_path(target)
        return [_Write(found[0], node, found[1], None, None, True)] if found is not None else []

    def _keys_put(self, expression: ast.AST, node: ast.AST) -> None:
        """What each key of the access *expression* written through holds is one of its container's keys:
        ``d[k] = v``, ``d[a][b] = v``, ``d[k].append(v)``."""
        while isinstance(expression, (ast.Attribute, ast.Subscript)):
            if isinstance(expression, ast.Subscript) and not isinstance(expression.slice, ast.Slice):
                inner = _access_path(expression.value)
                if inner is not None:
                    base, steps = inner
                    self._add(_Write(base, node, steps + (_KEY,), expression.slice, _same, True))
            expression = expression.value

    def _read(self, func: ast.AST) -> None:
        # A comprehension's target is its own scope's, not the function's.
        inner = {
            id(name)
            for node in walk(func)
            if isinstance(node, ast.comprehension)
            for name in walk(node.target)
        }
        self._iterables = _loop_iterables(func, self.dicts)
        for node in own_nodes(func):
            if (
                isinstance(node, ast.AugAssign)
                and not self.deciding
                and not (isinstance(node.op, ast.Add) and _SEQUENCE in self._augmented_kinds(node))
                and not (self.unread_kept and isinstance(node.op, ast.Add) and self._augmented_kinds(node) == {None})
            ):
                # ``total += row[2]``, ``x /= 2``, ``msg += f"..."``: a value computed from the old holds none of the
                # rows (U1) -- a name's replaces it, a field of an object is written with nothing.
                for write in self._target_writes(node.target, node):
                    self._add(write)
            elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)) and node.value is not None:
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    self._target(target, node, node.value, _same, isinstance(node, ast.AugAssign))
            elif isinstance(node, (ast.For, ast.AsyncFor)):
                take = _keys if _is_dict(node.iter, self.dicts) else _elements
                self._target(node.target, node, node.iter, take, False)
            elif isinstance(node, ast.withitem) and node.optional_vars is not None:
                for name in walk(node.optional_vars):
                    if isinstance(name, ast.Name):
                        self._add(_Write(name.id, site_of(node.optional_vars)))
            elif isinstance(node, ast.ExceptHandler) and node.name:
                self._add(_Write(node.name, node))
            elif isinstance(node, ast.Delete):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self._add(_Write(target.id, node))
            elif isinstance(node, ast.NamedExpr) and id(node.target) not in inner:
                self._add(_Write(node.target.id, site_of(node), (), node.value, _same, False))
            elif isinstance(node, ast.Call):
                self._call_writes(node)

    def _call_writes(self, call: ast.Call) -> None:
        statement = site_of(call)
        name = _callee_name(call)
        if isinstance(call.func, ast.Attribute):
            found = _access_path(call.func.value)
            made = call.func.value
            if (
                found is None
                and isinstance(made, ast.Call)
                and isinstance(made.func, ast.Attribute)
                and made.func.attr == "setdefault"
                and (inner := _access_path(made.func.value)) is not None
            ):
                # ``d.setdefault(k, []).append(v)``: the element setdefault hands back is one ``d`` holds.
                found = (inner[0], inner[1] + (_ALL,))
            if found is not None and name in _CARRIERS + ("insert",):
                base, steps = found
                if name in ("append", "add") and call.args:
                    self._add(_Write(base, statement, steps + (_ALL,), call.args[0], _same, True))
                elif name == "insert" and len(call.args) >= 2:
                    self._add(_Write(base, statement, steps + (_ALL,), call.args[1], _same, True))
                elif name in ("extend", "update") and call.args:
                    self._add(_Write(base, statement, steps, call.args[0], _same, True))
                elif name == "setdefault" and len(call.args) >= 2:
                    self._add(_Write(base, statement, steps + (_ALL,), call.args[1], _same, True))
                if name == "setdefault" and call.args:
                    self._add(_Write(base, statement, steps + (_KEY,), call.args[0], _same, True))
                self._keys_put(made, statement)
                return
        parts = _setattr_parts(call)
        if parts is not None:
            holder, value = parts
            found = _access_path(holder)
            if found is not None:
                base, steps = found
                attr = call.args[1]
                field = (
                    ("." + attr.value,)
                    if isinstance(attr, ast.Constant) and isinstance(attr.value, str)
                    else (_ANY_FIELD,)
                )
                self._add(_Write(base, statement, steps + field, value, _same, True))
            return
        # A call whose definition may keep what it is handed in an object named here (its summary's *kept*).
        for expression in [*call.args, *(keyword.value for keyword in call.keywords)] + (
            [call.func.value] if isinstance(call.func, ast.Attribute) else []
        ):
            found = _access_path(expression)
            if found is not None:
                base, steps = found
                self._add(_Write(base, statement, steps, None, None, True, call))

    def _grow(self) -> None:
        # Which names can hold at all, read by name: only their writes are worked out.
        names = {write.name for write in self.writes if write.forced}
        values = set(self.call_values)
        changed = True
        while changed:
            changed = False
            for write in self.writes:
                if write.name not in names and (write.mentions & names or write.inner_calls & values):
                    names.add(write.name)
                    changed = True
        live = [
            write
            for write in self.writes
            if write.forced or (write.name in names and (write.mentions & names or write.inner_calls & values))
        ]
        growing = True
        while growing:
            growing = False
            for write in live:
                new = write.forced | self._value(write)
                if not new <= write.paths:
                    write.paths = write.paths | new
                    growing = True

    def _value(self, write: _Write) -> _Paths:
        if write.call is not None:
            kept = self.reading.kept_by(self, write.call)
            return frozenset(
                path
                for (base, steps), paths in kept.items()
                if base == write.name and steps == write.steps
                for path in _below(steps, paths)
            )
        if write.source is None or write.take is None:
            return _NOTHING
        return _below(write.steps, write.take(self.held(write.source, write.node)))

    # -- the reads -----------------------------------------------------------

    def _of_name(self, name: str, at: ast.AST) -> _Paths:
        writes = self.by_name.get(name)
        if not writes or not any(write.paths for write in writes):
            return _NOTHING
        statement = at if isinstance(at, (ast.stmt, ast.ExceptHandler)) else site_of(at)
        reached = self.reading.surviving(self.func, name, statement, writes)
        return frozenset(path for write in writes if id(write.node) in reached for path in write.paths)

    def _comprehension(self, name: str, node: ast.AST) -> Optional[_Paths]:
        scope = getattr(node, "parent", None)
        while scope is not None and scope is not self.func:
            if isinstance(scope, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
                for generator in scope.generators:
                    bound = {n.id for n in walk(generator.target) if isinstance(n, ast.Name)}
                    if name in bound:
                        return self._unpack_into(
                            generator.target, name, _elements(self._as_list(generator.iter, generator.iter))
                        )
            scope = getattr(scope, "parent", None)
        return None

    def _as_list(self, expression: ast.AST, at: ast.AST) -> _Paths:
        """What *expression* holds as what it is iterated for, its elements at ``[*]``: a dict's are its keys."""
        held = self.held(expression, at)
        return _below((_ALL,), _keys(held)) if _is_dict(expression, self.dicts) else held

    def _unpack_into(self, target: ast.AST, name: str, paths: _Paths) -> Optional[_Paths]:
        if isinstance(target, ast.Name):
            return paths if target.id == name else None
        if isinstance(target, (ast.Tuple, ast.List)):
            for index, element in enumerate(target.elts):
                one = element.value if isinstance(element, ast.Starred) else element
                got = self._unpack_into(one, name, _under(f"[{index}]", paths))
                if got is not None:
                    return got
        return None

    def held(self, expression: Optional[ast.AST], at: Optional[ast.AST] = None) -> _Paths:
        """The parts of *expression*'s value that hold the rows where *at* runs."""
        if expression is None:
            return _NOTHING
        at = at if at is not None else expression
        held = self.held
        if isinstance(expression, ast.Name):
            got = self._comprehension(expression.id, expression)
            return got if got is not None else self._of_name(expression.id, at)
        if isinstance(expression, ast.Attribute):
            return _under("." + expression.attr, held(expression.value, at))
        if isinstance(expression, ast.Subscript):
            if isinstance(expression.slice, ast.Slice):
                return held(expression.value, at)
            return _under(_index_step(expression.slice), held(expression.value, at))
        if isinstance(expression, ast.Tuple):
            return frozenset(
                path for index, one in enumerate(expression.elts) for path in _below((f"[{index}]",), held(one, at))
            )
        if isinstance(expression, (ast.List, ast.Set)):
            return frozenset(
                path
                for one in expression.elts
                for path in _below((_ALL,), held(one.value if isinstance(one, ast.Starred) else one, at))
            )
        if isinstance(expression, ast.Dict):
            found: set[tuple[str, ...]] = set()
            for key, value in zip(expression.keys, expression.values, strict=True):
                if key is None:
                    found |= held(value, at)
                else:
                    found |= _below((_index_step(key),), held(value, at)) | _below((_KEY,), held(key, at))
            return frozenset(found)
        if isinstance(expression, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            return _below((_ALL,), held(expression.elt, at))
        if isinstance(expression, ast.DictComp):
            return _below((_ALL,), held(expression.value, at)) | _below((_KEY,), held(expression.key, at))
        if isinstance(expression, ast.BoolOp):
            return frozenset(path for value in expression.values for path in held(value, at))
        if isinstance(expression, ast.IfExp):
            return held(expression.body, at) | held(expression.orelse, at)
        if self.deciding and isinstance(expression, (ast.BinOp, ast.UnaryOp, ast.Compare, ast.JoinedStr)):
            # Made from the rows, for what they decide: any operand holding them whole.
            parts = [held(one, at) for one in ast.iter_child_nodes(expression) if isinstance(one, ast.expr)]
            if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Add):
                return parts[0] | parts[1]
            if isinstance(expression, ast.JoinedStr):
                return _WHOLE if any(parts) else _NOTHING
            return _WHOLE if any(_whole(part) for part in parts) else _NOTHING
        # A value computed from the rows holds none of them (U1): arithmetic, a sign, a comparison, a message of
        # several values -- what reaches a call or a key from it is a number or a text, not a row nor what names one.
        if isinstance(expression, ast.BinOp):
            if not isinstance(expression.op, ast.Add):
                return _NOTHING
            # The chain ``a + b + c`` as one: a list anywhere in it makes every + of it a list's.
            operands = _summands(expression)
            parts = [held(one, at) for one in operands]
            joined = frozenset(path for part in parts for path in part)
            kinds = {_operand(one, self.names) for one in operands}
            if _SEQUENCE in kinds or any(_in_elements(part) for part in parts):
                return joined  # lists concatenated: still the rows
            if _TEXT in kinds and _NUMBER not in kinds:
                # One value made into a text carries it (``"x" + str(pandaID)``); two make a message.
                return _WHOLE if sum(1 for part in parts if part) == 1 else _NOTHING
            if self.unread_kept and kinds == {None}:
                return joined  # operands whose type is not written: held as before U1 (the crossing pass)
            return _NOTHING  # a sum of numbers, or of operands whose type is not written: not read
        if isinstance(expression, (ast.UnaryOp, ast.Compare)):
            return _NOTHING
        if isinstance(expression, ast.JoinedStr):
            formatted = [value for value in expression.values if isinstance(value, ast.FormattedValue)]
            return _WHOLE if len(formatted) == 1 and held(formatted[0], at) else _NOTHING
        if isinstance(expression, ast.FormattedValue):
            return _WHOLE if held(expression.value, at) else _NOTHING
        if isinstance(expression, (ast.Await, ast.NamedExpr, ast.Starred)):
            return held(expression.value, at)
        if isinstance(expression, ast.Call):
            return self._call_value(expression, at)
        return _NOTHING

    def _call_value(self, call: ast.Call, at: ast.AST) -> _Paths:
        if id(call) in self.call_values:
            return self.call_values[id(call)]
        name = _callee_name(call)
        if name in _LOG_MAKERS:
            return _NOTHING
        held = self.held
        args = [one.value if isinstance(one, ast.Starred) else one for one in call.args]
        if name in _SERIALIZING and args and (isinstance(call.func, ast.Name) or name == "dumps"):
            return _WHOLE if held(args[0], at) else _NOTHING
        if isinstance(call.func, ast.Name):
            if name in _TELLING_NOTHING:
                return _NOTHING
            as_list = self._as_list
            if name in _KEEPING:
                return (held(args[0], at) if name == "dict" else as_list(args[0], at)) if args else _NOTHING
            if name == "next":
                return _elements(as_list(args[0], at)) if args else _NOTHING
            if name in ("filter", "map") and len(args) >= 2:
                return as_list(args[1], at)  # the elements kept, or what is made of each (read as the element)
            if name in ("min", "max") and len(args) == 1:
                return _under(_ALL, as_list(args[0], at))
            if name == "enumerate" and args:
                return _below((_ALL, "[1]"), _under(_ALL, as_list(args[0], at)))
            if name == "zip":
                return frozenset(
                    path
                    for index, one in enumerate(args)
                    for path in _below((_ALL, f"[{index}]"), _under(_ALL, as_list(one, at)))
                )
            if name == "getattr" and len(args) >= 2:
                attr = args[1]
                if isinstance(attr, ast.Constant) and isinstance(attr.value, str):
                    return _under("." + attr.value, held(args[0], at))
                return _any_field(held(args[0], at))
        if isinstance(call.func, ast.Attribute):
            if name in _COPIES and isinstance(call.func.value, ast.Name) and call.func.value.id == "copy":
                return held(args[0], at) if args else _NOTHING
            receiver = held(call.func.value, at)
            placed = self.reading.place(self, call) if receiver else None
            if not receiver and name in _CONTAINER_ELEMENT and self.handed_any(call, at):
                # ``d.get(k)``, ``k`` holding them: the key chooses the value; a default handed beside it may be it.
                placed = self.reading.place(self, call)
                if not (placed is not None and placed.targets):
                    return frozenset(path for one in args[1:] for path in held(one, at))
            if receiver and not (placed is not None and placed.targets):
                # A method of a builtin container, which no reading places: dict.get, .values(), .items(), ...
                if name == "values":
                    return _below((_ALL,), _under(_ALL, receiver))
                if name == "keys":
                    return _below((_ALL,), _under(_KEY, receiver))
                if name == "items":
                    return _below((_ALL, "[0]"), _under(_KEY, receiver)) | _below((_ALL, "[1]"), _under(_ALL, receiver))
                if name in _CONTAINER_ELEMENT:
                    return _under(_ALL, receiver)
                if name in _CONTAINER_KEEPING:
                    return receiver
        if not self.handed_any(call, at):
            return _NOTHING
        if self.reading.record.is_database(self.file, call) and self.reading.outermost(call):
            return _NOTHING
        placed = self.reading.place(self, call)
        if placed is not None and placed.targets:
            found: set[tuple[str, ...]] = set()
            for target in placed.targets:
                given = self.reading.given(self, call, placed, target)
                if not given:
                    continue
                summary = self.reading.summary(target, given)
                if target.func.name == "__init__":  # type: ignore[attr-defined]
                    found |= summary.kept.get(receiver_parameter(target) or "", _NOTHING)
                elif not placed.thread:
                    found |= summary.returns
            return frozenset(found)
        if args and name in _CHUNKING | _SLICING | _CHAINING:
            # An iterator helper no reading places: the first argument's elements, or every one's -- not the
            # count beside them, which the rule below would read as the whole value.
            taken = args if name in _CHAINING else args[:1]
            elements = frozenset(path for one in taken for path in _elements(self._as_list(one, at)))
            return _below((_ALL, _ALL) if name in _CHUNKING else (_ALL,), elements)
        # A call no reading places -- a builtin, a str method, a hole -- computes a value from what it is handed, and
        # it holds none of the rows (U1): a count, a sum, a value read again (``json.loads`` of a column), another
        # system's answer (a hole handed them is said where it is handed them: ``crosses_open``).  What carries them
        # is a conversion of one value (``int(row[0])``), a held value reshaped by its own method
        # (``tmpLFN.strip()``), one column's values joined (``",".join(ids)``) or one value formatted
        # (``"{}".format(pandaID)``).
        def holding(part: _Paths) -> bool:
            return _whole(part) or _whole(_under(_ALL, part))

        if self.deciding:
            # Made from what it is handed, for what the rows decide.
            parts = [held(one, at) for one in args] + [held(keyword.value, at) for keyword in call.keywords]
            if isinstance(call.func, ast.Attribute):
                parts.append(held(call.func.value, at))
            return _WHOLE if any(holding(part) for part in parts) else _NOTHING
        if isinstance(call.func, ast.Name) and name in _CONVERSIONS:
            return _WHOLE if args and holding(held(args[0], at)) else _NOTHING
        if isinstance(call.func, ast.Attribute) and name not in _READ_AGAIN:
            if holding(held(call.func.value, at)):
                return _WHOLE
            if name in ("sub", "subn") and args and len(args) <= 3:
                # ``re.sub(pattern, repl, value)`` / ``pattern.sub(repl, value)``: one value reshaped, as
                # ``value.replace(...)`` is, where it alone holds them.
                handed = [held(one, at) for one in args]
                if holding(handed[-1]) and not any(handed[:-1]):
                    return _WHOLE
            if name in ("join", "format"):
                # A text made by a separator or a template, written here or held in a name.
                handed = [held(one, at) for one in args] + [held(keyword.value, at) for keyword in call.keywords]
                holders = [part for part in handed if holding(part)]
                if len(handed) == 1 or (name == "join" and len(holders) == 1):
                    return _WHOLE if holders else _NOTHING  # one value, or one column's values ("," .join(ids))
                return _NOTHING  # a text of several values: os.path.join(a, b), "{} {}".format(a, b)
        if self.unread_kept and _unread_call(call, name) and not self.reading.record.is_database(self.file, call):
            parts = [held(one, at) for one in args] + [held(keyword.value, at) for keyword in call.keywords]
            if isinstance(call.func, ast.Attribute):
                parts.append(held(call.func.value, at))
            return _WHOLE if any(holding(part) for part in parts) else _NOTHING
        return _NOTHING

    def handed_any(self, call: ast.Call, at: Optional[ast.AST] = None) -> bool:
        """Whether an argument of *call*, or the object it is made on, holds a part of the rows."""
        at = at if at is not None else call
        if any(self.held(one.value if isinstance(one, ast.Starred) else one, at) for one in call.args):
            return True
        if any(self.held(keyword.value, at) for keyword in call.keywords):
            return True
        return isinstance(call.func, ast.Attribute) and bool(self.held(call.func.value, at))

    def returns(self) -> _Paths:
        """What a call of the function gives back of the rows: its returns, and a generator's yields as elements."""
        found: set[tuple[str, ...]] = set()
        for node in own_nodes(self.func):
            if isinstance(node, ast.Return) and node.value is not None:
                found |= self.held(node.value, node)
            elif isinstance(node, ast.Yield) and node.value is not None:
                found |= _below((_ALL,), self.held(node.value, node))
            elif isinstance(node, ast.YieldFrom):
                found |= self.held(node.value, node)
        return frozenset(found)

    def kept(self) -> dict[str, _Paths]:
        """``{parameter: paths}`` its writes put into what a parameter (the receiver among them) names: a write at
        steps of it, or a call keeping something in it."""
        arguments = getattr(self.func, "args", None)
        names = set()
        if arguments is not None:
            names = {one.arg for one in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs)}
        found: dict[str, _Paths] = {}
        for write in self.writes:
            if write.name in names and (write.steps or write.call is not None) and write.adds and write.paths:
                found[write.name] = found.get(write.name, _NOTHING) | write.paths
        return found


class _Reading:
    """What each definition does with what a call hands it, read once a build: a summary per (definition, the class
    it is bound to, the parts each parameter is given), and whether it reads them.

    A function **reads** what it is handed where its value decides something there: a test on it whole (``if
    rows:``), an index it chooses, a call that uses it -- the database's (a bind), a method of the value
    (``job.valuesMap()``), a function owning no junction that reads it in its body; else what a call makes of it is
    read where that goes.  Not a read: a log line (:data:`_LOG_METHODS`, :data:`_LOG_MAKERS`), a builtin telling
    nothing of it, a copy, a value put into a container or handed on -- to a name, a loop, a ``return``, the caller.
    A definition is read while it is read: a call of itself gets nothing back and reads nothing.
    """

    def __init__(self, calls: Calls, owners: set[str], record: "ReachGraph", unread_kept: bool = False):
        self.calls, self.owners, self.record = calls, owners, record
        self.unread_kept = unread_kept
        #: The names junction owners' functions have: where a hole spelled ``x.f(...)`` could land at one.
        self.owner_names = {owner.rsplit("::", 1)[-1] for owner in owners}
        self._summaries: dict[tuple, _Summary] = {}
        self._reads: dict[tuple, bool] = {}
        self._placed: dict[tuple, Optional[Placed]] = {}
        #: Keyed by node ids, each holding the node, so that no id is reused while the build runs.
        self._writes: dict[int, tuple[ast.AST, list[_Write]]] = {}
        self._dicts: dict[int, tuple[ast.AST, frozenset[str]]] = {}
        self._names: dict[int, tuple[ast.AST, dict[str, str]]] = {}
        self._surviving: dict[tuple, set[int]] = {}
        self._nested: dict[int, tuple[ast.AST, set[int]]] = {}

    def writes_of(self, func: ast.AST, deciding: bool = False) -> list[_Write]:
        held = self._writes.get((id(func), deciding))
        if held is None:
            held = self._writes[(id(func), deciding)] = (
                func, _PathCarry.read_writes(func, self.dicts_of(func), deciding, self.unread_kept, self.names_of(func))
            )
        return held[1]

    def names_of(self, func: ast.AST) -> dict[str, str]:
        """What *func*'s names are, for a ``+`` of them (:func:`_name_kinds`)."""
        held = self._names.get(id(func))
        if held is None:
            held = self._names[id(func)] = (func, _name_kinds(func))
        return held[1]

    def dicts_of(self, func: ast.AST) -> frozenset[str]:
        """The names *func* holds dicts in (:func:`_dict_names`)."""
        held = self._dicts.get(id(func))
        if held is None:
            held = self._dicts[id(func)] = (func, _dict_names(func))
        return held[1]

    def surviving(self, func: ast.AST, name: str, statement: ast.AST, writes: list[_Write]) -> set[int]:
        """The ids of *name*'s writes that reach *statement* as *func* runs."""
        key = (id(func), name, id(statement))
        reached = self._surviving.get(key)
        if reached is None:
            nodes = {id(write.node): write.node for write in writes if write.node is not None}
            adds = {id(write.node): write.node for write in writes if write.adds and write.node is not None}
            reached = {
                id(node) for node in surviving_writes(list(nodes.values()), statement, list(adds.values()), as_run=True)
            }
            self._surviving[key] = reached
        return reached

    def outermost(self, call: ast.Call) -> bool:
        """Whether *call* is not inside another call of its statement (its arguments, keywords or callee)."""
        statement = site_of(call)
        nested = self._nested.get(id(statement))
        if nested is None:
            nested = self._nested[id(statement)] = (
                statement,
                {
                    id(inner)
                    for outer in walk(statement)
                    if isinstance(outer, ast.Call)
                    for inner in walk(outer)
                    if inner is not outer and isinstance(inner, ast.Call)
                },
            )
        return id(call) not in nested[1]

    def place(self, carry: _PathCarry, call: ast.Call) -> Optional[Placed]:
        """Where *call* lands (:meth:`Calls.callees`); ``None`` for the database's call or one no reading places."""
        key = (id(call), carry.bound)
        if key not in self._placed:
            if self.record.is_database(carry.file, call):
                self._placed[key] = None
            else:
                try:
                    self._placed[key] = self.calls.callees(carry.file, carry.func, call, carry.bound)
                except Exception:  # noqa: BLE001 -- the record has the call as a hole
                    self._placed[key] = None
        return self._placed[key]

    def given(
        self, carry: _PathCarry, call: ast.Call, placed: Placed, target: Target, everything: bool = False
    ) -> dict[str, _Paths]:
        """``{parameter: paths}`` *call* hands *target* of what *carry* holds -- all of it, *everything*: the call
        running the statement, its rows what the call makes."""
        try:
            handed = self.calls.handed(CallSite(carry.file, carry.func, call), target)
        except Exception:  # noqa: BLE001 -- the record has the call as a hole
            return {}
        got: dict[str, _Paths] = {}
        for name, value in handed.items():
            if value is None:
                continue
            paths = _WHOLE if everything else carry.held(value, call)
            if paths:
                got[name] = paths
        spread = any(isinstance(one, ast.Starred) for one in call.args) or any(
            keyword.arg is None for keyword in call.keywords
        )
        if spread and (
            everything
            or any(carry.held(one, call) for one in call.args)
            or any(carry.held(keyword.value, call) for keyword in call.keywords)
        ):
            for name, value in handed.items():
                if value is None:
                    got[name] = _WHOLE
        if not placed.on_class and isinstance(call.func, ast.Attribute):
            receiver = receiver_parameter(target)
            paths = _WHOLE if everything else carry.held(call.func.value, call)
            if receiver is not None and paths:
                got[receiver] = paths
        return got

    def _key(self, target: Target, given: dict[str, _Paths]) -> tuple:
        return (target.file, target.func.lineno, target.bound, frozenset(given.items()))  # type: ignore[attr-defined]

    def carry(self, target: Target, given: dict[str, _Paths]) -> _PathCarry:
        return _PathCarry(self, target.file, target.func, target.bound, seeds=given)

    def summary(self, target: Target, given: dict[str, _Paths]) -> _Summary:
        key = self._key(target, given)
        got = self._summaries.get(key)
        if got is not None:
            return got
        self._summaries[key] = _Summary(_NOTHING, {}, None)
        carry = self.carry(target, given)
        got = _Summary(carry.returns(), carry.kept(), carry)
        self._summaries[key] = got
        return got

    def kept_by(self, carry: _PathCarry, call: ast.Call) -> dict[tuple[str, tuple[str, ...]], _Paths]:
        """``{(name, steps): paths}`` the definitions *call* lands at put into the objects *carry*'s function names.

        A method the corpus does not define, on an object whose class gets it from a base outside the corpus, keeps
        all it is handed in that object (:meth:`_kept_outside`)."""
        name = _callee_name(call)
        if name in _LOG_METHODS or name in _LOG_MAKERS:
            return {}
        if not carry.handed_any(call):
            return {}
        placed = self.place(carry, call)
        if placed is None:
            return {}
        if not placed.targets:
            return self._kept_outside(carry, call, placed)
        found: dict[tuple[str, tuple[str, ...]], _Paths] = {}
        for target in placed.targets:
            given = self.given(carry, call, placed, target)
            if not given:
                continue
            summary = self.summary(target, given)
            if not summary.kept:
                continue
            try:
                handed = dict(self.calls.handed(CallSite(carry.file, carry.func, call), target))
            except Exception:  # noqa: BLE001 -- the record has the call as a hole
                continue
            receiver = receiver_parameter(target)
            if receiver is not None and not placed.on_class and isinstance(call.func, ast.Attribute):
                handed[receiver] = call.func.value
            for parameter, paths in summary.kept.items():
                value = handed.get(parameter)
                where = _access_path(value) if value is not None else None
                if where is not None:
                    found[where] = found.get(where, _NOTHING) | paths
        return found

    def _kept_outside(
        self, carry: _PathCarry, call: ast.Call, placed: Placed
    ) -> dict[tuple[str, tuple[str, ...]], _Paths]:
        """What a method no class of the corpus defines keeps of what *call* hands it: all of it, in the object it is
        made on, where every class that object can be is built on a base outside the corpus -- the base supplies the
        method and nothing here says what it does with its arguments.  ``dc_req_spec.pack(row)`` fills a
        ``DataCarouselRequestSpec``, a ``SpecBase`` of pandacommon's, with the row; a class with no such base has no
        such method, and nothing is kept."""
        if placed.unplaced != UNPLACED_NO_METHOD or not isinstance(call.func, ast.Attribute):
            return {}
        where = _access_path(call.func.value)
        if where is None:
            return {}
        types = self.calls.types
        try:
            declared = types.declared(carry.file, carry.func, call.func.value)
        except Exception:  # noqa: BLE001 -- the record has the call as a hole
            return {}
        classes = [one for each in declared or () for one in self.calls.objects_of(each)]
        if not classes or not all(types.built_outside(*one) for one in classes):
            return {}
        held = _NOTHING
        for one in [*call.args, *(keyword.value for keyword in call.keywords)]:
            held |= carry.held(one.value if isinstance(one, ast.Starred) else one, call)
        return {where: held} if held else {}

    # -- whether a definition reads what it is handed ----------------------------

    def reads(self, target: Target, given: dict[str, _Paths]) -> bool:
        key = self._key(target, given)
        if key in self._reads:
            return self._reads[key]
        self._reads[key] = False
        got = self.reads_in(_PathCarry(self, target.file, target.func, target.bound, seeds=given, deciding=True))
        self._reads[key] = got
        return got

    def reads_in(self, carry: _PathCarry) -> bool:
        """Whether *carry*'s function reads a part of the rows it holds -- read for what they decide
        (:attr:`_PathCarry.deciding`): a test on a value computed from them reads them."""
        if not carry.deciding:
            seeds, start = carry._made
            carry = _PathCarry(
                self, carry.file, carry.func, carry.bound, seeds=seeds, start=start, call_values=carry.call_values,
                deciding=True,
            )
        for node in own_nodes(carry.func):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                # ``x.m(...)`` on what holds them: ``x.m`` itself is no part they are held in (job.valuesMap()).
                if carry.held(node.func.value, node) and self._method_reads(carry, node):
                    return True
                continue
            if not isinstance(node, (ast.Name, ast.Attribute, ast.Subscript)) or not isinstance(
                getattr(node, "ctx", None), ast.Load
            ):
                continue
            parent = getattr(node, "parent", None)
            if (
                isinstance(parent, ast.Attribute)
                and isinstance(getattr(parent, "parent", None), ast.Call)
                and parent.parent.func is parent
            ):
                continue  # a method's receiver: read above, at its call
            if isinstance(parent, (ast.Attribute, ast.Subscript)) and parent.value is node:
                continue  # judged at the longer path
            if carry.held(node, node) and self._use_reads(carry, node):
                return True
        return False

    def _use_reads(self, carry: _PathCarry, node: ast.AST) -> bool:
        """Whether the use *node* is put to reads it: up through what forms a value, to where it goes."""
        child, parent = node, getattr(node, "parent", None)
        while parent is not None:
            if isinstance(parent, ast.Call):
                if child is parent.func:
                    return self._method_reads(carry, parent)
                return self._argument_reads(carry, parent, child)
            if isinstance(parent, ast.Attribute) and parent.value is child:
                if isinstance(getattr(parent, "parent", None), ast.Call) and parent.parent.func is parent:
                    return self._method_reads(carry, parent.parent)
                return False  # a field the path does not hold (held() read it as such)
            if isinstance(parent, ast.Subscript):
                if parent.slice is child:
                    return _whole(carry.held(child, child))  # an index: the rows choose
                child, parent = parent, getattr(parent, "parent", None)
                continue
            if isinstance(parent, (ast.If, ast.While, ast.Assert, ast.IfExp)) and child is parent.test:
                return _whole(carry.held(child, child))
            if isinstance(parent, ast.comprehension) and child in parent.ifs:
                return _whole(carry.held(child, child))
            if isinstance(parent, ast.comprehension) and child is parent.iter:
                child, parent = parent, getattr(parent, "parent", None)
                continue
            if isinstance(
                parent,
                (
                    ast.Compare,
                    ast.UnaryOp,
                    ast.BinOp,
                    ast.BoolOp,
                    ast.IfExp,
                    ast.Tuple,
                    ast.List,
                    ast.Set,
                    ast.Dict,
                    ast.Starred,
                    ast.JoinedStr,
                    ast.FormattedValue,
                    ast.Await,
                    ast.NamedExpr,
                    ast.keyword,
                    ast.ListComp,
                    ast.SetComp,
                    ast.GeneratorExp,
                    ast.DictComp,
                ),
            ):
                child, parent = parent, getattr(parent, "parent", None)
                continue
            # Handed on -- to a name, the loop, the caller -- or dropped: read where that is used.
            return False
        return False

    def _argument_reads(self, carry: _PathCarry, call: ast.Call, child: ast.AST) -> bool:
        name = _callee_name(call)
        if name in _LOG_METHODS or name in _LOG_MAKERS:
            return False
        if isinstance(call.func, ast.Name) and name in _TELLING_NOTHING:
            return False
        if isinstance(call.func, ast.Attribute) and name in _CARRIERS + ("insert",):
            return False  # put into a container: read where that is used
        if (
            name in _COPIES
            and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "copy"
        ):
            return False
        if isinstance(call.func, ast.Name) and name in _PASSING_ON:
            return self._use_reads(carry, call)
        return self._call_reads(carry, call)

    def _method_reads(self, carry: _PathCarry, call: ast.Call) -> bool:
        name = _callee_name(call)
        if name in _LOG_METHODS or name in _LOG_MAKERS:
            return False
        placed = self.place(carry, call)
        if placed is not None and placed.targets:
            return self._call_reads(carry, call)
        if name in {"values", "keys", "items"} | _CONTAINER_ELEMENT | _CONTAINER_KEEPING:
            return self._use_reads(carry, call)
        if name in _CARRIERS + ("insert",):
            return False
        return self._call_reads(carry, call)

    def _call_reads(self, carry: _PathCarry, call: ast.Call) -> bool:
        """A call handed what holds them: the database reads it (a bind is a use); a function owning no junction
        reads it where its body does; otherwise what the call makes of it is read where that goes."""
        if self.record.is_database(carry.file, call):
            return True
        placed = self.place(carry, call)
        if placed is not None and placed.targets:
            for target in placed.targets:
                given = self.given(carry, call, placed, target)
                if given and target.owner not in self.owners and self.reads(target, given):
                    return True
        return bool(carry.held(call, call)) and self._use_reads(carry, call)


@dataclasses.dataclass
class _Landing:
    """Where the walk carrying a statement's rows (:class:`_Crossings`) reached a call the reach record reads as the
    database through calls: the call and its function, the way there, and where the rows sit in what it binds."""

    file: str
    func: ast.AST
    call: ast.Call
    #: ``(kind, file, func, call)`` per step, the first in the statement's own function: ``"into"`` a call followed
    #: down, ``"back"`` a caller's call the rows are returned to.
    way: tuple
    #: ``{bind key without its colon: the paths of the rows under it}`` (:func:`_bind_holdings`).
    held: dict[str, _Paths]
    #: The same, per select item by its position, from the walks seeded with each (:func:`_item_landings`).
    items: dict[int, dict[str, _Paths]] = dataclasses.field(default_factory=dict)
    #: The function of the statement whose rows these are: the way's first step is a caller's where the rows were
    #: returned before they went down.
    origin: Optional[ast.AST] = None

    def key(self) -> tuple:
        """The same landing in another walk of the statement: its call by the same way."""
        return id(self.call), tuple((kind, id(func), id(call)) for kind, _file, func, call in self.way)


def _bind_holdings(carry: _PathCarry, call: ast.Call) -> dict[str, _Paths]:
    """Where *carry*'s rows sit in each bind *call* hands the database, the text (its first argument) aside: ``{key
    without its colon: the paths under it}`` of a map, ``""`` for one held whole; each map of a list alike."""
    found: dict[str, set[tuple[str, ...]]] = {}
    handed = [one.value if isinstance(one, ast.Starred) else one for one in call.args[1:]]
    for value in [*handed, *(keyword.value for keyword in call.keywords)]:
        for path in carry.held(value, call):
            steps = [step for step in path if step != _ALL] if path[:1] == (_ALL,) else list(path)
            if not steps:
                found.setdefault("", set()).add(())
            elif steps[0].startswith("['") and steps[0].endswith("']"):
                found.setdefault(steps[0][2:-2].lstrip(":"), set()).add(tuple(steps[1:]))
    return {key: frozenset(paths) for key, paths in found.items()}


class _Crossings:
    """Where the rows one statement keeps reach a junction that reads them, read through the corpus's calls.

    A call lands where :meth:`Calls.callees` places it and a function is called from where :meth:`Calls.callers`
    finds it -- the reading the reach record and the walk have, not a name.  A call is followed where it is handed a
    part of the rows (:meth:`_PathCarry.handed_any`), the call running the statement with all of it.  **A junction's
    owner it lands at is a crossing where it reads the part it is handed** (:meth:`_Reading.reads`); one that does
    not -- it logs it, stores it, hands it on -- is gone through as any other function is: the parameters given
    parts hold them there, the object it is made on among them, and each call it makes with them is read the same
    way.  Each definition is entered once a statement for what it is handed (the class it is bound to and the parts
    each parameter is given), and a call the record reads as the database is not followed: what a statement hands
    the database is another run's input.

    Back to a caller: the rows a function returns are what the call there gives, and the caller is read with that
    value -- a junction's owner reading it is a crossing (it is still followed on), the calls handed a part of it
    are followed, and where it returns them its callers are read in turn.  ``TaskBuffer.getConfigValue`` returns
    what ``proxy.getConfigValue`` returned, and ``AtlasAnalJobBroker`` is where the value goes.

    A call on the way that this reading cannot place -- a hole of the reach record -- is kept in :attr:`open`, as
    the record names it, where it could land at a junction: ``x.f(...)`` lands at a definition named ``f``, if in
    the corpus at all, so one whose name no junction owner's function has is no way to one -- ``tmpLogger.error``,
    ``self.logger.debug``.  A plain name called is a hole whatever it spells -- the value of a local or a parameter
    -- and is kept.  So is a caller this reading cannot place, on the way back: a hole spelled ``x.f(...)``, ``f``
    the returning function's name, may be one more -- ``insert_step_task``'s task id goes back through
    ``TaskBuffer`` to ``self.tbif.insert_step_task(...)`` in ``PandaTaskStepHandler.submit_target``, a receiver
    declared ``Any``.
    """

    def __init__(self, reading: _Reading, landing: bool = False):
        self.reading = reading
        #: Whether to keep where the rows reach the database through calls (:attr:`landings`).
        self.landing = landing
        #: The junction owners the rows reach and read them; those found on the way back too.
        self.found: set[str] = set()
        self.found_back: set[str] = set()
        #: ``(owner, line, callee)`` of each hole on the way, as the reach record spells it.
        self.open: dict[tuple[str, int, str], None] = {}
        self._entered: set[tuple] = set()
        self._returned: set[tuple] = set()
        self._backing = 0
        #: Where the rows reached the database through calls (:class:`_Landing`), and the way the walk is on now.
        self.landings: list[_Landing] = []
        self._way: list[tuple[str, str, ast.AST, ast.Call]] = []

    def _cross(self, owner: str) -> None:
        self.found.add(owner)
        if self._backing:
            self.found_back.add(owner)

    def into(self, carry: _PathCarry, call: ast.Call, everything: bool = False) -> None:
        """Follow *call*, made in *carry*'s function, handed what *carry* holds there -- all of it, *everything*."""
        reading = self.reading
        if reading.record.is_database(carry.file, call):
            if self.landing and self._way:
                self.landings.append(
                    _Landing(carry.file, carry.func, call, tuple(self._way), _bind_holdings(carry, call))
                )
            return
        hole = reading.record.hole_at(carry.file, call)
        if hole is not None and (not isinstance(call.func, ast.Attribute) or call.func.attr in reading.owner_names):
            self.open[hole] = None
        placed = reading.place(carry, call)
        if placed is None:
            return
        for target in placed.targets:
            given = reading.given(carry, call, placed, target, everything)
            if not given:
                continue
            self._way.append(("into", carry.file, carry.func, call))
            try:
                if target.owner in reading.owners and reading.reads(target, given):
                    self._cross(target.owner)
                    if self.landing:
                        self._land_in(target, given)
                    continue
                self._through(target, given)
            finally:
                self._way.pop()

    def _land_in(self, target: Target, given: dict[str, _Paths]) -> None:
        """Where what a junction's owner crossed to is given reaches the database in its own statements: not gone
        through, the owner's other calls are not followed, but a statement it runs with them in a bind is a
        :class:`_Landing` as one in a function gone through is."""
        key = ("landed", self.reading._key(target, given))
        if key in self._entered:
            return
        self._entered.add(key)
        carry = self.reading.summary(target, given).carry or self.reading.carry(target, given)
        for call in _own_calls(target.func):
            if self.reading.record.is_database(target.file, call) and carry.handed_any(call):
                self.landings.append(
                    _Landing(target.file, target.func, call, tuple(self._way), _bind_holdings(carry, call))
                )

    def _through(self, target: Target, given: dict[str, _Paths]) -> None:
        key = self.reading._key(target, given)
        if key in self._entered:
            return
        self._entered.add(key)
        carry = self.reading.summary(target, given).carry or self.reading.carry(target, given)
        for call in _own_calls(target.func):
            if carry.handed_any(call):
                self.into(carry, call)

    def back(self, file: str, func: ast.AST, paths: _Paths) -> None:
        """Follow the parts *paths* of the rows *func* returns into each of its callers."""
        key = (file, func.lineno, paths)  # type: ignore[attr-defined]
        if key in self._returned:
            return
        self._returned.add(key)
        reading = self.reading
        for hole in reading.record.holes_spelled(func.name):  # type: ignore[attr-defined]
            self.open[hole] = None
        held = reading.calls.types.method_class(file, func)
        self._backing += 1
        for site in reading.calls.callers(Target(file, func, held[1] if held is not None else None)):
            # A call in a module's own body has no caller to hand the rows on to, and a function calling itself is
            # not where they go next.
            if site.in_module or (site.file == file and site.func is func):
                continue
            self._way.append(("back", site.file, site.func, site.call))
            try:
                carry = _PathCarry(reading, site.file, site.func, (), call_values={id(site.call): paths})
                owner = f"{site.file}::{site.func.name}"  # type: ignore[attr-defined]
                if owner in reading.owners and reading.reads_in(carry):
                    self._cross(owner)
                for call in _own_calls(site.func):
                    if call is not site.call and carry.handed_any(call):
                        self.into(carry, call)
                returned = carry.returns()
                if returned:
                    self.back(site.file, site.func, returned)
            finally:
                self._way.pop()
        self._backing -= 1


def _walk(
    reading: _Reading,
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    site: ReadSiteNode,
    item: Optional[int] = None,
    landing: bool = False,
) -> tuple[_Crossings, _Paths]:
    """Carry *site*'s rows on from its statement in *func* (:func:`attach_crossings`) -- with *item*, that select item
    of them alone (:class:`_PathCarry`), with *landing* keeping where they reach the database through calls: the walk,
    and what *func* returns of them."""
    file, line = site.anchor.file, site.anchor.line_start
    rows = _rows_written(func, line, site.region[1])
    carry = _PathCarry(reading, file, func, (), start=rows, item=item)
    crossings = _Crossings(reading, landing)
    for call in _own_calls(func):
        if call.lineno == line:
            # The call running the statement, followed with all it hands:
            # ``do_preassign`` hands its statement to ``JediTaskBuffer.
            # queryTasksToPreassign_JEDI``, which hands it to the proxy that
            # updates the tasks it picks.
            crossings.into(carry, call, everything=True)
        elif rows and carry.handed_any(call):
            crossings.into(carry, call)
    returned = carry.returns() if rows else _NOTHING
    if returned:
        crossings.back(file, func, returned)
    return crossings, returned


def _own_run(landing: _Landing, func: ast.AST, site: ReadSiteNode) -> bool:
    """Whether *landing* is *site*'s statement's own run, handed on by the call running it."""
    kind, _file, first, call = landing.way[0]
    return kind == "into" and first is func and call.lineno == site.anchor.line_start


def _item_landings(
    reading: _Reading, func: ast.FunctionDef | ast.AsyncFunctionDef, site: ReadSiteNode, found: list[_Landing]
) -> None:
    """Fill each of *found*'s :attr:`_Landing.items`: *site*'s rows walked again once per select item, each seeded
    alone, where one holds a bind by key."""
    if not any(key for landing in found for key in landing.held):
        return
    for number in _positions(_select_lists(func, site)):
        crossings, _returned = _walk(reading, func, site, number, landing=True)
        by = {landing.key(): landing.held for landing in crossings.landings}
        for landing in found:
            held = by.get(landing.key())
            if held is not None:
                landing.items[number] = held


def _select_lists(func: ast.AST, site: ReadSiteNode) -> tuple[tuple[sqltext.SelectItem, ...], ...]:
    """*site*'s select lists (:func:`sqltext.select_items`): each distinct one the texts its run has give where read
    as a tree, in the runs' order.  One line's texts can differ -- a hole spelled ``{}`` on some passes and ``{},{}``
    on others, a table chosen by an ``if`` -- and a value's column is read on each."""
    line = site.anchor.line_start
    runs = [run for run in sql.executions(func) if run.call.lineno == line]  # type: ignore[arg-type]
    return tuple(dict.fromkeys(items for run in runs if (items := sqltext.select_items(run.reading)) is not None))


def _positions(lists: tuple[tuple[sqltext.SelectItem, ...], ...]) -> list[int]:
    """The places in a row at which one of *lists* has an item not after a hole: each is carried on its own."""
    return [
        number
        for number in range(max(map(len, lists), default=0))
        if any(number < len(items) and items[number].kind != sqltext.ITEM_AFTER_HOLE for items in lists)
    ]


def _column_choice(
    lists: tuple[tuple[sqltext.SelectItem, ...], ...],
    held: list[int],
    value: Optional[ast.AST],
    site: ReadSiteNode,
    subject_of: Callable[[str], str],
) -> tuple[str, str, str]:
    """``(table.column, its subject, "")`` of the select item of *site*'s rows a bind's *value* is, the places whose
    carry holds it whole being *held* -- chosen on each of its select *lists* and said where all agree, the subject by
    *subject_of*; where they give columns of one subject, ``("", it, KEY_COLUMN_TEXTS)`` (a hole read per table:
    ``jobsActive4``'s or ``jobsDefined4``'s ``PandaID``, ``JobSpec.PandaID``); or ``("", "", why not)``
    (:data:`KEY_COLUMN_STAR` ...)."""
    if not lists:
        return "", "", KEY_COLUMN_UNREAD
    chosen = {
        _item_choice(items, [number for number in held if number < len(items)], value, site) for items in lists
    }
    if len(chosen) == 1:
        column, why = chosen.pop()
        return column, subject_of(column), why
    subjects = {subject_of(column) for column, _why in chosen}
    return "", subjects.pop() if len(subjects) == 1 else "", KEY_COLUMN_TEXTS


def _item_choice(
    items: tuple[sqltext.SelectItem, ...], held: list[int], value: Optional[ast.AST], site: ReadSiteNode
) -> tuple[str, str]:
    """:func:`_column_choice` on one select list.  A field of an object several items fill (``taskSpec.pack(row)``)
    is the item of its name, or the hole's or ``*``'s: the column of its name where the table has one."""
    held = [number for number in held if items[number].kind != sqltext.ITEM_AFTER_HOLE]
    field = value.attr.lower() if isinstance(value, ast.Attribute) else None
    if len(held) > 1 and field is not None:
        named = [number for number in held if items[number].column == field]
        spread = [number for number in held if items[number].kind in (sqltext.ITEM_HOLE, sqltext.ITEM_STAR)]
        held = named or spread or held
    if not held:
        unread = any(item.kind in (sqltext.ITEM_HOLE, sqltext.ITEM_AFTER_HOLE) for item in items)
        return "", KEY_COLUMN_HOLE if unread else KEY_COLUMN_NONE
    if len(held) > 1:
        return "", KEY_COLUMN_SEVERAL
    item = items[held[0]]
    if item.kind == sqltext.ITEM_COLUMN:
        return f"{item.table}.{item.column}", ""
    if item.kind in (sqltext.ITEM_HOLE, sqltext.ITEM_STAR):
        tables = {table.lower() for table in site.tables}
        table = item.table or (next(iter(tables)) if len(tables) == 1 else "")
        if field is not None and table and field in sqltext.table_columns(table):
            return f"{table}.{field}", ""
        if field is not None and table and sqltext.table_columns(table):
            # ``fileSpec.fileID`` of ``job.Files``, ``job`` packed from ``SELECT {} FROM jobsDefined4``: a part of
            # the rows, and no column of theirs.
            return "", KEY_COLUMN_NONE
        return "", KEY_COLUMN_HOLE if item.kind == sqltext.ITEM_HOLE else KEY_COLUMN_STAR
    return "", KEY_COLUMN_EXPRESSION


def attach_crossings(
    read_sites: list[ReadSiteNode],
    modules: list[SourceModule],
    junctions: list[JunctionNode],
    calls: Optional[Calls],
    record: Optional["ReachGraph"],
    unread_kept: bool = True,
    landings: Optional[dict[str, list[_Landing]]] = None,
) -> CrossingCensus:
    """Say which junction a row handled here is carried on to.

    The request's sixth point: a reader wants to be able to pick a *statement*
    as the lead and be told where the investigation goes next.  A statement
    already says which rows it asked for and which of them it drops; what it
    could not say is that the ones it keeps are handed to something the map
    already holds an answer for.

    Two shapes, both stored, every call placed as :class:`_Crossings` reads
    it (*calls*, and *record* for the holes and the database's calls):

    * the call running the statement, and each call the function hands a part
      of the rows to (:class:`_PathCarry`, an object they were put into among
      them), lands in a function the map owns a junction in and that reads
      them, or goes through functions to one -- ``copyArchive.py::main``'s
      ``holding`` query hands each job to ``Watcher(...)`` and calls
      ``.start()``, so the next thing that happens to a held job is
      ``Watcher.run``;
    * the rows are handed back to a caller, and what it does with them is
      read the same way -- the flagship, ``getTasksToExecCommand_JEDI@4591``,
      returns its tasks to the ``JediTaskBuffer`` layer and ``TaskCommando``
      starts a ``TaskCommandoThread`` with them, whose ``runImpl`` owns one.

    The seeds are the writes the statement's rows go into on its line and to
    the end of its region (:func:`_rows_written`), as the exits are read: a
    fetch after the next ``execute`` in the loop over the rows takes another
    statement's rows.

    **The second was sized against the wrong shape once.**  It used to be
    read off "a ``return`` inside the region", which is 139 returns over 73
    statements and does not include ``getTasksToExecCommand_JEDI@4591`` at
    all: that statement owns 4573-4603, works its rows through 4600-4621, and
    the function returns on 4647, the rows having travelled ``resList`` ->
    ``retTaskIDs`` -> ``retTaskList`` on the way.

    A crossing names a junction, never the caller's own function: naming the
    caller would satisfy the question and not the purpose.  Where a call on
    the way could not be placed, or a hole may be a caller the rows go back
    to, the statement says so (``crosses_open``): what ``crosses_to`` names is
    then what this reading knows, not all of it.

    **A call the rows only decide is not followed**: one under a test on
    them, in a loop over them, in a later operand of an ``and`` / ``or`` or
    after an exit they choose, but handed nothing they hold.  Read so, a
    write a test on the rows chose would hold them too, and the reading
    grows with every caller: 2136 crossings, 1181 at the statement's own
    function alone, against 824 -- the decision is in the junction's own
    guards, and no reader asks a statement for it.

    **A value computed from the rows carries none of them** (U1, the
    querier's five kinds): a number, a comparison, a count, a text of several
    values, a column's content read again (``json.loads``) reach the callee,
    not a row nor what names one -- getTaskParamsWithID_JEDI 1402's parsed
    taskParams, getScoutJobData_JEDI's medians.  A value the map cannot read
    -- another system's answer, a call no reading places, a ``+`` whose
    operands' types the code does not say -- is held as before (*unread_kept*,
    the build's way): the callee is kept, and the reader's words "hands the
    rows" say more than the map knows there.  What the rows decide is read
    with the rule before U1 throughout (:attr:`_PathCarry.deciding`).

    A lead has to name somewhere the map will answer for, and read by name it
    did not: before promotion filtered the junctions, every
    ``self.cur.execute(...)`` in a region resolved to the ``execute`` of an
    adder plugin promotion then dropped -- 520 crossings over 239 statements
    where 71 over 38 are real -- and over every candidate, 98 statements went
    to ``WorkQueue.pack`` from ``datasetSpec.pack(...)``.  A placed call goes
    where it lands, and promotion now drops nothing.

    Returns the census: how many crossings were stored, how many statements
    hand their rows back at all, how many of those named a junction on the way
    back, and how many statements have a call on the way, or a caller back,
    this reading cannot place.  The gap between the second and third is the
    honest size of what is still unresolved, and it is reported rather than
    silently left as an empty field.  Without *calls* or *record* nothing is
    placed and nothing stored.

    With *landings*, where the walk reached the database through calls with
    the rows in a statement's binds goes into it by the statement's read site
    (:class:`_Landing`, each select item's too): the key a statement of another
    function names its rows by, which the KeyFroms read
    (:func:`attach_key_sources`).  Not the statement's own run handed on.
    """
    if calls is None or record is None:
        return CrossingCensus(0, 0, 0, 0)
    owners = {junction.owner for junction in junctions}
    trees = {module.rel_path: module.tree for module in modules}
    reading = _Reading(calls, owners, record, unread_kept)
    stored = handing_back = resolved = opened = 0
    for site in read_sites:
        if not site.region:
            continue
        func = _site_function(trees, site)
        if func is None:
            continue
        crossings, returned = _walk(reading, func, site, landing=landings is not None)
        if landings is not None:
            found = [one for one in crossings.landings if not _own_run(one, func, site)]
            for landing in found:
                landing.origin = func
            if found:
                _item_landings(reading, func, site, found)
                landings[site.name] = found
        # Not itself: a statement is not somewhere its own rows go next.  Done
        # to the handed-back answer before it is counted as well as to the
        # whole, or the census reports two statements as resolved whose only
        # lead was back to themselves.
        crossings.found.discard(site.owner)
        crossings.found_back.discard(site.owner)
        handing_back += 1 if returned else 0
        resolved += 1 if crossings.found_back else 0
        site.crosses_to = sorted(crossings.found)
        site.crosses_open = [
            OpenCrossing(owner=owner, line=line, callee=callee) for owner, line, callee in sorted(crossings.open)
        ]
        stored += len(site.crosses_to)
        opened += 1 if site.crosses_open else 0
    return CrossingCensus(stored, handing_back, resolved, opened)


class _Filled:
    """Which of a function's statements' rows, parameters and fields of ``self`` a value it binds was filled from, as the
    carry reads it (:class:`KeySource`).

    Each way in is seeded on its own -- a parameter whole, ``self`` at one field's step -- and the value is filled
    from each whose carry reaches it, by the rule every value is carried by: the writes of a name that survive to
    where it is read (:func:`pathcond.surviving_writes`).  A name was once followed by its spelling over every loop
    of the function, the nested scopes' among them: ``datasetManager.main`` bound ``:name`` from its own rows (``for
    (name,) in resReset:``) and was said to be handed it by ``self.datasets``, a loop of a class it defines; and
    ``lockJobOutputReport``'s ``for panda_id, attempt_nr in resGL:`` rebinds the parameters' names with its own rows,
    which read as the caller's ``panda_id``.

    A field of ``self`` is an attribute of it the function reads and does not call: ``self.cur.rowcount`` is the
    cursor's, ``self.getNext(jediTaskID)`` hands on what it is given.  A call the reach record reads as the database
    gives nothing back of what it is handed (:meth:`_PathCarry._call_value`), so a fetch from ``self.cur`` is not
    the field's.  A statement's rows are seeded as :func:`attach_crossings` seeds them, to its region's end; so are
    a statement's that selects a sequence's next values, which reads no table and is no read site
    (:data:`KEY_SEQUENCE`).

    Which column of a statement's rows a value is: each select item (:func:`sqltext.select_items`) carried on its
    own (:class:`_PathCarry`'s *item*), the value that item's where its carry holds it whole; a field of an object
    several items fill (``taskSpec.pack(row)``) is the item of its name, and where the item is a hole or ``*`` the
    column of its name where the table has one.  And for which of the values a statement selects a statement here
    never runs with the value: a test on the way to each call handed it on a name holding that column
    (:meth:`never_at`).
    """

    def __init__(
        self,
        reading: _Reading,
        file: str,
        func: ast.FunctionDef | ast.AsyncFunctionDef,
        sites: Iterable[ReadSiteNode] = (),
        attributor: Optional[SpecAttributor] = None,
        class_of: Optional[Callable[[str], Optional[str]]] = None,
        owner_class: Optional[str] = None,
    ):
        self.reading, self.file, self.func, self.attributor = reading, file, func, attributor
        self.class_of, self.owner_class = class_of, owner_class
        self.sites = sorted((site for site in sites if site.region), key=lambda site: site.anchor.line_start)
        runs = [run for run in sql.executions(func) if _SELECTS_SEQUENCE.match(run.sql or "")]
        laid = sql.regions(func, list(sql.executions(func))) if runs else {}
        #: ``(the sequence, the line its statement runs on, its region's last line)``.
        self.sequences = sorted(
            {
                (m.group(1).split(".")[-1], call.lineno, laid[call.lineno][1])
                for run in runs
                if (m := _SELECTS_SEQUENCE.match(run.sql or ""))
                for call in run.run_at
                if call.lineno in laid
            },
            key=lambda one: one[1],
        )
        arguments = func.args
        given = [one.arg for one in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs)]
        self.parameters = [name for name in given if name not in ("self", "cls")]
        called = {id(node.func) for node in own_nodes(func) if isinstance(node, ast.Call)}
        self.fields = (
            sorted(
                {
                    node.attr
                    for node in own_nodes(func)
                    if isinstance(node, ast.Attribute)
                    and isinstance(node.ctx, ast.Load)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "self"
                    and id(node) not in called
                }
            )
            if given[:1] == ["self"]
            else []
        )
        self._carries: dict[str, _PathCarry] = {}
        self._items: dict[str, tuple[tuple[sqltext.SelectItem, ...], ...]] = {}

    def _carry(self, name: str, seeds: dict[str, _Paths]) -> _PathCarry:
        carry = self._carries.get(name)
        if carry is None:
            carry = self._carries[name] = _PathCarry(self.reading, self.file, self.func, (), seeds=seeds)
        return carry

    def handed(self, value: ast.AST, at: ast.AST, steps: frozenset[str] = frozenset()) -> list[str]:
        """The parameters, then the ``self.<field>`` s, that fill *value* itself where *at* runs -- or, for an
        object, one of *steps* of it: the fields naming its row.

        Not one that fills only a part: ``tmpTaskSpec`` in ``TaskRefinerThread.runImpl`` holds
        ``self.data_carousel_interface``'s message at ``.errorDialog`` and ``self.taskList``'s task at
        ``.jediTaskID``, and only the second names the row its flush writes.
        """
        return [
            name
            for _kind, name, paths in self._handed(value, at)
            if _whole(paths) or any(_bare(path)[:1] and _bare(path)[0] in steps for path in paths)
        ]

    def _handed(self, value: ast.AST, at: ast.AST) -> list[tuple[str, str, _Paths]]:
        found = [
            (KEY_PARAMETER, name, paths)
            for name in self.parameters
            if (paths := self._carry(name, {name: _WHOLE}).held(value, at))
        ]
        found += [
            (KEY_SELF_FIELD, f"self.{field}", paths)
            for field in self.fields
            if (paths := self._carry(f"self.{field}", {"self": frozenset({(f".{field}",)})}).held(value, at))
        ]
        return found

    def _rows(self, site: ReadSiteNode) -> Optional[_PathCarry]:
        key = f"rows {site.name}"
        if key not in self._carries:
            seeds = _rows_written(self.func, site.anchor.line_start, site.region[1])
            self._carries[key] = _PathCarry(self.reading, self.file, self.func, (), start=seeds) if seeds else None
        return self._carries[key]

    def _sequence(self, line: int, last: int) -> Optional[_PathCarry]:
        key = f"sequence {line}"
        if key not in self._carries:
            seeds = _rows_written(self.func, line, last)
            self._carries[key] = _PathCarry(self.reading, self.file, self.func, (), start=seeds) if seeds else None
        return self._carries[key]

    def sources(self, value: ast.AST, at: ast.AST) -> list[KeySource]:
        """Where *value*, filled at *at*, came from: each statement's rows, sequence's next values, parameter and
        field of ``self`` that reaches it, else what it is -- a literal, an attribute of something else, or a value
        not followed."""
        line = getattr(at, "lineno", None)
        found = []
        for site in self.sites:
            carry = self._rows(site)
            paths = carry.held(value, at) if carry is not None else _NOTHING
            if paths:
                column, attr, unread = self.column(site, value, at)
                found.append(
                    KeySource(
                        kind=KEY_ROWS,
                        line=line,
                        name=site.name,
                        site_owner=site.owner,
                        site_line=site.anchor.line_start,
                        paths=_spelled_paths(paths),
                        column=column,
                        attr=attr,
                        column_unread=unread,
                    )
                )
        for sequence, run_line, last in self.sequences:
            carry = self._sequence(run_line, last)
            paths = carry.held(value, at) if carry is not None else _NOTHING
            if paths:
                found.append(
                    KeySource(
                        kind=KEY_SEQUENCE,
                        line=line,
                        name=sequence,
                        site_owner=f"{self.file}::{self.func.name}",
                        site_line=run_line,
                        paths=_spelled_paths(paths),
                    )
                )
        found += [
            KeySource(kind=kind, line=line, name=name, paths=_spelled_paths(paths), attr=self.subject(value))
            for kind, name, paths in self._handed(value, at)
        ]
        if found:
            return found
        if isinstance(value, ast.Constant):
            return [KeySource(kind=KEY_CONSTANT, line=line, name=repr(value.value))]
        kind = KEY_ATTRIBUTE if isinstance(value, ast.Attribute) else KEY_UNKNOWN
        return [KeySource(kind=kind, line=line, name=ast.unparse(value), attr=self.subject(value))]

    def subject(self, value: ast.AST) -> str:
        """The subject *value* is where it is a field of an object whose spec class the build tells -- ``x.f`` as
        ``Class.f``, the attribute as the class declares it (``archiveJob``'s ``tmpFile.dispatchDBlock``,
        ``FileSpec.dispatchDBlock``) -- else ``""``: the key's value, whichever parameter or field it came by."""
        if not isinstance(value, ast.Attribute) or self.attributor is None:
            return ""
        spec_class = self.attributor.class_of(value.value, self.func, self.owner_class)
        spelled = _declared_spelling(self.attributor, spec_class, value.attr) if spec_class else None
        return f"{spec_class}.{spelled}" if spelled else ""

    def items(self, site: ReadSiteNode) -> tuple[tuple[sqltext.SelectItem, ...], ...]:
        """*site*'s select lists, one per distinct list its run's texts give (:func:`_select_lists`)."""
        if site.name not in self._items:
            self._items[site.name] = _select_lists(self.func, site)
        return self._items[site.name]

    def _item(self, site: ReadSiteNode, item: int) -> _PathCarry:
        key = f"item {item} {site.name}"
        if key not in self._carries:
            seeds = _rows_written(self.func, site.anchor.line_start, site.region[1])
            self._carries[key] = _PathCarry(self.reading, self.file, self.func, (), start=seeds, item=item)
        return self._carries[key]

    def column(self, site: ReadSiteNode, value: ast.AST, at: ast.AST) -> tuple[str, str, str]:
        """``(table.column, its subject, "")`` of the select item of *site*'s rows *value* is where *at* runs, or
        ``("", the subject or "", why not)`` (:func:`_column_choice`)."""
        lists = self.items(site)
        held = [number for number in _positions(lists) if _whole(self._item(site, number).held(value, at))]
        return _column_choice(lists, held, value, site, self.attr)

    def attr(self, column: str) -> str:
        """*column* (``table.column``) as a subject where its table's spec class has it, else ``""``."""
        if not column or self.attributor is None:
            return ""
        return _spec_attr(self.attributor, *column.split(".", 1), self.class_of)

    def never_at(self, site: ReadSiteNode, places: list[ast.AST]) -> list[str]:
        """The values *site* selects (its *selects*) for which no node of *places* runs: a test on the way to each
        (``pathcond.ruled_out_at``) on a name holding the column of that subject in *site*'s rows."""
        lists = self.items(site)
        if not places or not site.selects or self.attributor is None:
            return []
        subjects = {}
        for number in _positions(lists):
            named = {
                self.attr(f"{items[number].table}.{items[number].column}")
                if number < len(items) and items[number].kind == sqltext.ITEM_COLUMN
                else ""
                for items in lists
            }
            if len(named) == 1 and (subject := named.pop()):
                subjects[number] = subject
        chosen: dict[str, list[str]] = {}
        for selected in site.selects:
            chosen.setdefault(selected.partition("=")[0], []).append(selected)
        wanted = {number: subject for number, subject in subjects.items() if subject in chosen}
        if not wanted:
            return []
        tested: list[dict[str, set[str]]] = []
        for place in places:
            names: dict[str, set[str]] = {}
            for test in pathcond.tests_on(place, self.func):
                for node in walk(test):
                    if isinstance(node, ast.Name):
                        for number, subject in wanted.items():
                            if _whole(self._item(site, number).held(node, node)):
                                names.setdefault(subject, set()).add(node.id)
            tested.append(names)
        return sorted(
            selected
            for subject, listed in chosen.items()
            for selected in listed
            if all(
                any(
                    pathcond.ruled_out_at(place, self.func, name, _selected_value(selected))
                    for name in sorted(names.get(subject, ()))
                )
                for place, names in zip(places, tested, strict=True)
            )
        )


def _spec_attr(
    attributor: SpecAttributor, table: str, column: str, class_of: Optional[Callable[[str], Optional[str]]] = None
) -> str:
    """*table*'s *column* as a subject where the table's spec class has it (``JediTaskSpec.jediTaskID``), else ``""``:
    the class by *class_of* where given (the attributor that learned the tables), else *attributor*'s."""
    spec_class = class_of(table) if class_of is not None else attributor.class_for_table(table)
    qualifier, attribute, kind = _subject_of(attributor, spec_class, table, column)
    return SubjectNode.make_name(qualifier, attribute) if kind == "spec" else ""


def _selected_value(selected: str) -> object:
    """The value of a read site's *selects* entry (``JobSpec.jobStatus=activated``) as the code compares it: a number
    where it is one."""
    text = selected.partition("=")[2]
    try:
        return int(text)
    except ValueError:
        return text


def _term_column(term: str) -> str:
    """The column a KeyFrom's *term* compares: ``jediTaskID`` of ``jediTaskID=:jediTaskID`` or of
    ``jediTaskID IN (:a, :b)``."""
    return term.split(" IN ", 1)[0].split("=", 1)[0].strip()


def _site_function(
    trees: dict[str, ast.AST], site: ReadSiteNode
) -> Optional[ast.FunctionDef | ast.AsyncFunctionDef]:
    """The function *site*'s statement runs in: the first of its owner's name holding its line."""
    tree = trees.get(site.anchor.file)
    if tree is None:
        return None
    wanted = site.owner.rsplit("::", 1)[-1]
    return next(
        (
            node
            for node in walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == wanted
            and node.lineno <= site.anchor.line_start <= (node.end_lineno or node.lineno)
        ),
        None,
    )


def _read_key_froms(sites: Iterable[ReadSiteNode], trees: dict[str, ast.AST]) -> list[KeyFrom]:
    """A KeyFrom (:attr:`KeyFrom.reads`) for each row-naming term of each read site's run: a term on the statement's
    own rows (:data:`sqltext.ROW`) comparing a column with binds, not excluding, every row meeting it -- as
    :func:`_key_froms` makes a write's, its binds read the same way.  Its statement is the read site's line, the run's
    first; an ``IN`` list of several binds is one term.
    """
    found: dict[tuple[str, int, str, str], KeyFrom] = {}
    for site in sorted(sites, key=lambda one: (one.owner, one.anchor.line_start)):
        func = _site_function(trees, site)
        if func is None:
            continue
        line, file = site.anchor.line_start, site.anchor.file
        for run in sql.executions(func):
            if run.call.lineno != line:
                continue
            for term in sqltext.terms(run.reading).terms:
                if term.verb != "read" or term.scope != sqltext.ROW or term.excluding or not term.decided:
                    continue
                # A bind spelt with a hole is no term here: sqltext.terms reads none.
                binds = list(term.binds)
                if not binds:
                    continue
                if len(binds) == 1 and not binds[0].endswith(sqltext.IN_LIST_SUFFIX):
                    spelled = f"{term.column}={binds[0]}"
                else:
                    spelled = f"{term.column} IN ({', '.join(binds)})"
                record = found.get((site.owner, line, term.table, spelled))
                if record is None:
                    record = found[(site.owner, line, term.table, spelled)] = KeyFrom(
                        owner=site.owner,
                        file=file,
                        statement=line,
                        table=term.table,
                        term=spelled,
                        unread=KEY_NO_MAP,
                        reads=True,
                    )
                    record._binds = (func, file, [])
                _ran_by(record, func, run)
                bound = [
                    filled
                    for bind in binds
                    for filled in (
                        sql.bound_values(func, run.varmap, bind, run.window, at=run)
                        if run.varmap is not None
                        else sql.inline_binds(run.call, bind)
                    )
                ]
                if run.varmap is None and not bound:
                    continue
                record._binds[2].extend(bound)
                record.unread = "" if record._binds[2] else KEY_NO_BIND
    return list(found.values())


def _handed_at(
    func: ast.FunctionDef | ast.AsyncFunctionDef, key_from: KeyFrom, value: Optional[ast.AST]
) -> list[ast.AST]:
    """The calls of *key_from*'s runs that run its statement with *value* under one of its term's binds: those at
    which the map the run is given can still hold it (:func:`sql.bound_values` at the call) --
    ``restartTasksForContentsUpdate_JEDI`` runs ``sqlTD`` in both arms of ``if taskStatus == "defined":`` and in a
    loop over another statement's rows, each with its own bind.  Every place it runs from where *value* is bound at
    none (an object's field)."""
    places = list(key_from._runs[1]) if key_from._runs is not None else []
    keys = re.findall(r":\w+", key_from.term)
    found: list[ast.AST] = []
    for run in key_from._executions:
        for call in run.calls or (run.call,):
            live = (
                bind
                for key in keys
                for bind in (
                    sql.bound_values(func, run.varmap, key, at=call)
                    if run.varmap is not None
                    else sql.inline_binds(call, key)
                )
            )
            if any(bind.value is value for bind in live) and not any(call is one for one in found):
                found.append(call)
    return found or places


def _landed_sources(
    froms: list[KeyFrom],
    landings: dict[str, list[_Landing]],
    sites: dict[str, ReadSiteNode],
    attributor: Optional[SpecAttributor],
    class_of: Optional[Callable[[str], Optional[str]]] = None,
) -> None:
    """Add to each KeyFrom a ``rows`` source of each statement whose rows the walk carried through calls to a run of
    its statement with them under one of its term's binds (:class:`_Landing`): the calls on the way
    (:attr:`KeySource.via`) and the column, by the walks of each select item (:func:`_column_choice`).  One per way."""
    by_call: dict[int, list[KeyFrom]] = {}
    for key_from in froms:
        if key_from.term and key_from._runs is not None:
            for node in key_from._runs[1]:
                by_call.setdefault(id(node), []).append(key_from)
    for name, found in landings.items():
        site = sites.get(name)
        if site is None:
            continue
        lists = _select_lists(found[0].origin, site)
        for landing in found:
            for key_from in by_call.get(id(landing.call), ()):
                hit = [key.lstrip(":") for key in re.findall(r":\w+", key_from.term) if key.lstrip(":") in landing.held]
                if not hit:
                    continue
                via = [f"{file}:{call.lineno}" for _kind, file, _func, call in landing.way]
                if any(one.kind == KEY_ROWS and one.name == site.name and one.via == via for one in key_from.sources):
                    continue
                held = sorted(
                    number
                    for number, holdings in landing.items.items()
                    if any(_whole(holdings.get(key, _NOTHING)) for key in hit)
                )
                bound = key_from._binds[2] if key_from._binds is not None else []
                column, attr, unread = _column_choice(
                    lists,
                    held,
                    bound[0].value if bound else None,
                    site,
                    lambda column: (
                        _spec_attr(attributor, *column.split(".", 1), class_of)
                        if column and attributor is not None
                        else ""
                    ),
                )
                source = KeySource(
                    kind=KEY_ROWS,
                    line=getattr(bound[0].site, "lineno", None) if bound else landing.call.lineno,
                    name=site.name,
                    site_owner=site.owner,
                    site_line=site.anchor.line_start,
                    paths=_spelled_paths(frozenset(path for key in hit for path in landing.held[key])),
                    column=column,
                    attr=attr,
                    column_unread=unread,
                    via=via,
                )
                source._way = landing.way
                key_from.sources.append(source)


def _attach_reaches(froms: list[KeyFrom], sites: dict[str, ReadSiteNode], reading: _Reading) -> None:
    """Fill :attr:`KeySource.reaches` / :attr:`KeySource.reaches_open` of each *same_rows* source of a read through
    calls (:func:`_reaches`), *reading* the crossings' own (:func:`attach_crossings`)."""
    at = {(site.owner, site.anchor.line_start): site for site in sites.values()}
    held: dict[tuple, tuple[list[str], list[OpenCrossing]]] = {}
    for key_from in froms:
        site = at.get((key_from.owner, key_from.statement))
        if not key_from.reads or site is None or key_from._binds is None:
            continue
        for source in key_from.sources:
            if not (source.same_rows and source.via and source._way):
                continue
            key = (site.name, tuple((kind, id(func), id(call)) for kind, _file, func, call in source._way))
            if key not in held:
                held[key] = _reaches(reading, site, key_from._binds[0], source._way)
            source.reaches, source.reaches_open = held[key]


def _reaches(
    reading: _Reading, site: ReadSiteNode, func: ast.AST, way: tuple
) -> tuple[list[str], list[OpenCrossing]]:
    """Where the rows of *site*, a read in *func* the walk reached by *way*, go next, going back up *way* alone: the
    calls *func* hands them to, then in each function up the way what its call there gives back and what it hands
    on -- to a caller the rows were returned to, and from the source's own function, every caller, as
    :meth:`_Crossings.back` reads one.  *site*'s owner aside."""
    crossings = _Crossings(reading)
    line = site.anchor.line_start
    rows = _rows_written(func, line, site.region[1]) if site.region else []  # type: ignore[arg-type]
    carry = _PathCarry(reading, site.anchor.file, func, (), start=rows)
    for call in _own_calls(func):
        if call.lineno != line and rows and carry.handed_any(call):
            crossings.into(carry, call)
    returned = carry.returns() if rows else _NOTHING
    for kind, file, step, call in reversed(way):
        if not returned:
            break
        if kind == "back":
            crossings.back(file, step, returned)
            returned = _NOTHING
            break
        carry = _PathCarry(reading, file, step, (), call_values={id(call): returned})
        owner = f"{file}::{step.name}"
        if owner in reading.owners and reading.reads_in(carry):
            crossings._cross(owner)
        for other in _own_calls(step):
            if other is not call and carry.handed_any(other):
                crossings.into(carry, other)
        returned = carry.returns()
    if returned:
        _kind, file, step, _call = way[0]
        crossings.back(file, step, returned)
    crossings.found.discard(site.owner)
    return sorted(crossings.found), [
        OpenCrossing(owner=owner, line=at, callee=callee) for owner, at, callee in sorted(crossings.open)
    ]


def _same_rows(froms: list[KeyFrom], sites: dict[str, ReadSiteNode]) -> None:
    """Mark each ``rows`` source of a statement that names its source statement's own rows
    (:attr:`KeySource.same_rows`): the statement's KeyFroms on one of the source's tables fill every column naming a
    row of it (:func:`sqltext.table_identity`: its key, dated columns aside) from that column of the source's rows."""
    together: dict[tuple[str, int], list[KeyFrom]] = {}
    for key_from in froms:
        if key_from.term and key_from.table:
            together.setdefault((key_from.owner, key_from.statement), []).append(key_from)
    for listed in together.values():
        for table in sorted({key_from.table.lower() for key_from in listed}):
            identity = sqltext.table_identity(table)
            if not identity:
                continue
            by_source: dict[str, dict[str, list[KeySource]]] = {}
            for key_from in listed:
                column = _term_column(key_from.term).lower()
                if key_from.table.lower() != table or column not in identity:
                    continue
                for source in key_from.sources:
                    site = sites.get(source.name) if source.kind == KEY_ROWS else None
                    if (
                        site is not None
                        and source.column == f"{table}.{column}"
                        and table in {one.lower() for one in site.tables}
                    ):
                        by_source.setdefault(source.name, {}).setdefault(column, []).append(source)
            for columns in by_source.values():
                if set(columns) == identity:
                    for marked in columns.values():
                        for source in marked:
                            source.same_rows = True


def _spelled_paths(paths: _Paths) -> list[str]:
    """*paths* as :attr:`KeySource.paths` spells them: ``()`` for the value itself."""
    return sorted("".join(path) or "()" for path in paths)


def attach_key_sources(
    row_keys: list[RowKeySource],
    junctions: list[JunctionNode],
    calls: Optional[Calls],
    record: Optional["ReachGraph"],
    key_froms: Optional[list[KeyFrom]] = None,
    read_sites: Iterable[ReadSiteNode] = (),
    table_class: Optional[Callable[[str], Optional[str]]] = None,
    modules: Iterable[SourceModule] = (),
    attributor: Optional[SpecAttributor] = None,
    landings: Optional[dict[str, list[_Landing]]] = None,
) -> int:
    """Say where each write's row keys were filled from: :attr:`KeyFrom.sources`, and which parameter or field of
    ``self`` handed each guarded write its keys (:attr:`RowKeySource.handed_by`) -- with *modules*, a read
    statement's too, a KeyFrom for each row-naming term of each read site's run added to *key_froms*
    (:func:`_read_key_froms`); of a ``rows`` source, the column of the rows it is (:attr:`KeySource.column`), the
    values the rows' statement selects at which the term's statement never runs (:attr:`KeySource.never_at`) and
    whether the statement names those very rows (:attr:`KeySource.same_rows`) -- with *landings*
    (:func:`attach_crossings`'), a source too of each statement whose rows the walk carried through calls into the
    term's binds, by the calls on the way (:attr:`KeySource.via`), and where a read of the same rows so reached
    hands them on back up that way alone (:attr:`KeySource.reaches`); then, by those sources, whether each
    ``INSERT`` makes its row (:attr:`KeyFrom.new_row`, *table_class* saying whose rows a read site's tables hold),
    and which subjects a write persisting an object writes (:attr:`KeyFrom.writes`, the function's assignments on it);
    last, with *modules* and *attributor* (the one the attribute slice reads assignments with), a KeyFrom for each
    object of *modules* whose assigned fields no write persists, added to *key_froms* (:func:`_unpersisted`).

    After the reach record, because the carry reads a call by where it lands and by what the record reads as the
    database, and over the values the write's reader kept (``KeyFrom._binds``, ``RowKeySource._filled``), by one
    :class:`_Filled` per function, so the two records read one value alike.  *handed_by* is the first, in the keys'
    order, that a parameter or a field fills -- the key's value itself, or the field of a flushed spec naming its
    row (:meth:`_Filled.handed`); a KeyFrom lists every source and the part of the value each fills.  Returns how
    many records were read; without *calls* or *record* none is, every ``handed_by`` stays empty and no KeyFrom has
    a source.
    """
    if calls is None or record is None:
        return 0
    reading = _Reading(calls, {junction.owner for junction in junctions}, record)
    modules = list(modules)
    sites_of: dict[str, list[ReadSiteNode]] = {}
    for site in read_sites:
        sites_of.setdefault(site.owner, []).append(site)
    sites = {site.name: site for listed in sites_of.values() for site in listed}
    owners = {id(func): owner for module in modules for func, owner in functions_with_owner(module.tree)}
    filled: dict[int, tuple[ast.AST, _Filled]] = {}

    def of(func: ast.FunctionDef | ast.AsyncFunctionDef, file: str) -> _Filled:
        held = filled.get(id(func))
        if held is None:
            first, last = func.lineno, func.end_lineno or func.lineno
            sites = [
                site
                for site in sites_of.get(f"{file}::{func.name}", ())
                if first <= site.anchor.line_start <= last
            ]
            held = filled[id(func)] = (
                func,
                _Filled(reading, file, func, sites, attributor, table_class, owners.get(id(func))),
            )
        return held[1]

    read = 0
    froms = list(key_froms or ())
    if key_froms is not None and modules:
        reads = _read_key_froms(sites.values(), {module.rel_path: module.tree for module in modules})
        key_froms.extend(reads)
        froms.extend(reads)
    handed: dict[int, ast.AST] = {}
    for key_from in froms:
        if key_from._binds is None:
            continue
        func, file, binds = key_from._binds
        reader = of(func, file)
        # A statement run with two texts can be given one bind by both.
        unique = {id(bind.value): bind for bind in binds}.values()
        key_from.sources = []
        for bind in unique:
            for source in reader.sources(bind.value, bind.site):
                handed[id(source)] = bind.value
                key_from.sources.append(source)
        read += 1
    for key_from in froms:
        if attributor is not None and key_from.table and key_from.term:
            key_from.attr = _spec_attr(attributor, key_from.table, _term_column(key_from.term), table_class)
        if key_from._binds is None or key_from._runs is None:
            continue
        func, file, _values = key_from._binds
        reader = of(func, file)
        for source in key_from.sources:
            if source.kind == KEY_ROWS and source.name in sites and not source.via and sites[source.name].selects:
                places = _handed_at(func, key_from, handed.get(id(source)))
                source.never_at = reader.never_at(sites[source.name], places)
    if landings:
        _landed_sources(froms, landings, sites, attributor, table_class)
    _same_rows(froms, sites)
    if landings:
        _attach_reaches(froms, sites, _Reading(calls, {junction.owner for junction in junctions}, record, True))
    _object_writes(froms, junctions)
    statements: dict[tuple[str, int, str], list[KeyFrom]] = {}
    for key_from in froms:
        if key_from._insert is not None:
            statements.setdefault((key_from.owner, key_from.statement, key_from.table), []).append(key_from)
    for together in statements.values():
        new_row, why, line = _new_row(together, sites, table_class or (lambda _table: None))
        for key_from in together:
            key_from.new_row, key_from.new_row_by, key_from.new_row_line = new_row, why, line
    for source in row_keys:
        if source._filled is None:
            continue
        func, file, values = source._filled
        reader = of(func, file)
        source.handed_by = next(
            (names[0] for value, at, steps in values if (names := reader.handed(value, at, steps))), ""
        )
        read += 1
    if attributor is not None and key_froms is not None:
        key_froms.extend(_unpersisted(froms, junctions, modules, attributor, reading))
    return read


def attach_routes(key_froms: Iterable[KeyFrom], calls: Optional[Calls]) -> None:
    """Fill each KeyFrom's :attr:`~KeyFrom.routes`, :attr:`~KeyFrom.routes_open` and :attr:`~KeyFrom.out_on`: the
    routes of calls into its function, whether a call by the function's name is placed nowhere, and the routes on
    which no place its write runs from runs -- each ruled out by the constants the route hands the function's flag
    parameters, as a read site's route says its statement is not run (``CallReading.not_run``, :func:`sql.at_call`).
    Without *calls*, nothing is filled."""
    if calls is None:
        return
    held: dict[int, tuple[ast.AST, list[tuple[list[str], dict[str, object]]], bool]] = {}
    for record in key_froms:
        if record._runs is None:
            continue
        func, places = record._runs
        found = held.get(id(func))
        if found is None or found[0] is not func:
            owned = calls.types.method_class(record.file, func)
            target = Target(record.file, func, owned[1] if owned else None)
            routes, stops = calls.routes(target)
            tested = _flag_parameters(func)
            found = held[id(func)] = (
                func,
                [(list(route.chain), _route_constants(calls, target, route.sites, tested)) for route in routes],
                bool(stops),
            )
        _func, routes, opened = found
        record.routes = len(routes)
        record.routes_open = opened
        record.out_on = [
            chain
            for chain, constants in routes
            if constants and places and all(pathcond.ruled_out_by(place, func, constants) for place in places)
        ]


def _object_writes(key_froms: list[KeyFrom], junctions: list[JunctionNode]) -> None:
    """Add to a KeyFrom persisting an object the subjects the function assigns on that object, at the
    assignments' lines -- ``killJob``'s ``job.jobStatus = ...`` written by its ``INSERT`` of ``job.valuesMap()``.
    The assignments are the attribute slice's junctions, so their lines are the branches a reader joins on; only
    those that reach the object's ``valuesMap()``, or the flush call handed it, are in what it writes
    (:func:`_fields_at`), and not one the write takes back out (``updateTask_JEDI``'s
    ``resetChangedAttr("jediTaskID")``)."""
    by_owner: dict[str, list[JunctionNode]] = {}
    for junction in junctions:
        by_owner.setdefault(junction.owner, []).append(junction)
    for key_from in key_froms:
        if key_from._object is None or key_from._binds is None:
            continue
        spelled, obj_class, mades, excluded = key_from._object
        func = key_from._binds[0]
        statements: dict[int, list[ast.stmt]] = {}
        for node in own_nodes(func):
            if isinstance(node, ast.stmt):
                statements.setdefault(node.lineno, []).append(node)
        obj = ast.parse(spelled, mode="eval").body
        for junction in by_owner.get(key_from.owner, ()):
            spec, _dot, attribute = junction.subject.rpartition(".")
            if spec != obj_class or attribute in excluded:
                continue
            reaching = {id(node) for node in _fields_at(func, obj, attribute, mades, obj_class)} if mades else None
            lines = sorted(
                {
                    line
                    for branch in junction.branches
                    for line in [branch.line, *branch.other_lines]
                    if line is not None
                    and any(
                        isinstance(target, ast.Attribute)
                        and target.attr == attribute
                        and ast.unparse(target.value) == spelled
                        and (reaching is None or id(statement) in reaching)
                        for statement in statements.get(line, ())
                        for target in targets_of(statement)
                    )
                }
            )
            if lines:
                key_from.writes[junction.subject] = sorted({*key_from.writes.get(junction.subject, ()), *lines})


def _unpersisted(
    key_froms: list[KeyFrom],
    junctions: list[JunctionNode],
    modules: Iterable[SourceModule],
    attributor: SpecAttributor,
    reading: _Reading,
) -> list[KeyFrom]:
    """A KeyFrom for each object of a junction owner whose fields it assigns at an arm no write writes -- no
    statement's record, nor a flush call's, has the arm's line for its subject -- one per what those fields reach
    next (:func:`_uses`), with the arms it reaches as *writes*: a reader taking the arms whose write names its row by
    a statement's rows would otherwise read these as known not to be.  getRandomSeed_JEDI's
    ``tmpFileSpec.status = "ready"``, set after the spec's ``valuesMap()`` is taken, is returned; datasetManager's
    ``job.jobStatus = "failed"`` is handed to ``TaskBuffer.updateJobs``, which persists it a call away.  An arm is a
    junction's line assigning its subject's attribute on an object of its class, by *attributor*."""
    written = {
        (key_from.owner, subject, line)
        for key_from in key_froms
        for subject, lines in key_from.writes.items()
        for line in lines
    }
    by_owner: dict[str, list[JunctionNode]] = {}
    for junction in junctions:
        by_owner.setdefault(junction.owner, []).append(junction)
    made: list[KeyFrom] = []
    for module in modules:
        for func, owner_class in functions_with_owner(module.tree):
            owner = f"{module.rel_path}::{func.name}"
            here = by_owner.get(owner)
            if not here:
                continue
            first, last = func.lineno, func.end_lineno or func.lineno
            statements: dict[int, list[ast.stmt]] = {}
            for node in own_nodes(func):
                if isinstance(node, ast.stmt):
                    statements.setdefault(node.lineno, []).append(node)
            used: dict[int, list[tuple[str, int, frozenset[str]]]] = {}
            found: dict[tuple[str, str], tuple[list[int], set[str], dict[str, set[int]]]] = {}
            for junction in here:
                spec, _dot, attribute = junction.subject.rpartition(".")
                lines = {
                    line
                    for branch in junction.branches
                    for line in (branch.line, *branch.other_lines)
                    if line is not None and first <= line <= last and (owner, junction.subject, line) not in written
                }
                for line in sorted(lines):
                    # A ``+=`` that adds to a value is an arm of its own (the attribute slice's), as an assignment is.
                    assigned = [
                        (statement, target.value)
                        for statement in statements.get(line, ())
                        for target in (
                            [statement.target] if isinstance(statement, ast.AugAssign) else targets_of(statement)
                        )
                        if isinstance(target, ast.Attribute)
                        and target.attr == attribute
                        and attributor.class_of(target.value, func, owner_class) == spec
                    ]
                    if not assigned:
                        continue
                    statement, obj = assigned[0]
                    if id(statement) not in used:
                        used[id(statement)] = _uses(reading, module.rel_path, func, statement)
                    for kind, at, callees in used[id(statement)]:
                        ats, handed_to, writes = found.setdefault((ast.unparse(obj), kind), ([], set(), {}))
                        ats.append(at)
                        handed_to |= callees
                        writes.setdefault(junction.subject, set()).add(line)
            for (spelled, kind), (ats, handed_to, writes) in sorted(found.items()):
                record = KeyFrom(
                    owner=owner,
                    file=module.rel_path,
                    statement=min(ats),
                    term="",
                    object=spelled,
                    unread=kind,
                    handed_to=sorted(handed_to),
                    writes={subject: sorted(lines) for subject, lines in sorted(writes.items())},
                )
                record._runs = (func, [statement for at in sorted(set(ats)) for statement in statements.get(at, ())])
                made.append(record)
    return made


def _uses(
    reading: _Reading, file: str, func: ast.FunctionDef | ast.AsyncFunctionDef, statement: ast.stmt
) -> list[tuple[str, int, frozenset[str]]]:
    """What the field *statement* assigns reaches next, as the carry reads it from there: each kind of use --
    :data:`KEY_HANDED` (calls of the corpus handed a part of it, or made on what holds it, and the functions they land
    in), :data:`KEY_BOUND` (the database's calls), :data:`KEY_RETURNED`, :data:`KEY_KEPT` (put in what a parameter,
    ``self`` among them, names: the caller holds it after -- adder's ``self.job.ddmErrorCode = ...`` before it
    returns) -- with the first line it is used on (the assignment's, kept), else :data:`KEY_NO_USE` at the
    assignment.  A call no reading places (a log line, a builtin, a hole) is no use."""
    carry = _PathCarry(reading, file, func, (), start=[statement])
    handed: list[int] = []
    callees: set[str] = set()
    bound: list[int] = []
    for call in _own_calls(func):
        if not carry.handed_any(call):
            continue
        if reading.record.is_database(file, call):
            bound.append(call.lineno)
            continue
        placed = reading.place(carry, call)
        if placed is not None and placed.targets:
            handed.append(call.lineno)
            callees |= {f"{target.file}::{target.func.name}" for target in placed.targets}  # type: ignore[attr-defined]
    returned = [
        node.lineno
        for node in own_nodes(func)
        if isinstance(node, (ast.Return, ast.Yield, ast.YieldFrom))
        and node.value is not None
        and carry.held(node.value, site_of(node))
    ]
    arguments = func.args
    handed_in = {one.arg for one in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs)}
    kept = [statement.lineno] if handed_in & set(carry.kept()) else []
    found = [
        (kind, min(lines), frozenset(names))
        for kind, lines, names in (
            (KEY_HANDED, handed, callees),
            (KEY_BOUND, bound, set()),
            (KEY_RETURNED, returned, set()),
            (KEY_KEPT, kept, set()),
        )
        if lines
    ]
    return found or [(KEY_NO_USE, statement.lineno, frozenset())]


#: The order a statement's why is told in where several of its key columns are new.
_NEW_ROW_ORDER = (NEW_ROW_SEQUENCE, NEW_ROW_RETURNING, NEW_ROW_LITERAL, NEW_ROW_COMPUTED)


def _key_values(
    key_from: KeyFrom, sites: dict[str, ReadSiteNode], table_class: Callable[[str], Optional[str]]
) -> list[tuple[str, str, Optional[int]]]:
    """What one key column of an ``INSERT`` is given: ``(new | carried | handed | other | unread, why, line)`` each."""
    insert: _Insert = key_from._insert
    statement = key_from.statement
    settled = {
        "returning": NEW_ROW_RETURNING,
        "sequence": NEW_ROW_SEQUENCE,
        "literal": NEW_ROW_LITERAL,
        "expression": NEW_ROW_COMPUTED,
    }
    if insert.value in settled:
        return [("new", settled[insert.value], statement)]
    if insert.value in ("absent", "none"):
        return [("unread", NEW_ROW_UNREAD, statement)]
    if insert.value == "upsert":
        return [("upsert", NEW_ROW_UPSERT, statement)]
    found: list[tuple[str, str, Optional[int]]] = []
    for kind, line, _class in insert.binds:
        if kind == "literal":
            found.append(("new", NEW_ROW_LITERAL, line))
        elif kind == "computed":
            found.append(("new", NEW_ROW_COMPUTED, line))
        elif kind == "returned":
            found.append(("new", NEW_ROW_RETURNING, line))
    held = {one_class for kind, _line, one_class in insert.binds if kind == "plain"}
    if not any(kind == "plain" for kind, _line, _class in insert.binds):
        return found or [("unread", NEW_ROW_UNREAD, statement)]
    if key_from.unread or not key_from.sources:
        found.append(("unread", NEW_ROW_UNREAD, statement))
    for source in key_from.sources:
        if source.kind == KEY_ROWS:
            if source.site_line in insert.returning_lines:
                found.append(("new", NEW_ROW_RETURNING, source.site_line))
            else:
                site = sites.get(source.name)
                classes = {table_class(t) or f"table:{t.lower()}" for t in (site.tables if site is not None else ())}
                found.append(("carried" if insert.table_class in classes else "other", NEW_ROW_MOVED, source.site_line))
        elif source.kind == KEY_SEQUENCE:
            found.append(("new", NEW_ROW_SEQUENCE, source.site_line))
        elif source.kind in (KEY_PARAMETER, KEY_SELF_FIELD):
            found.append(("handed" if insert.table_class in held else "other", NEW_ROW_HANDED, source.line))
        elif source.kind == KEY_CONSTANT:
            found.append(("new", NEW_ROW_LITERAL, source.line))
        else:
            found.append(("unread", NEW_ROW_UNREAD, source.line))
    return found


def _new_row(
    key_froms: list[KeyFrom], sites: dict[str, ReadSiteNode], table_class: Callable[[str], Optional[str]]
) -> tuple[Optional[bool], str, Optional[int]]:
    """An ``INSERT``'s mark, over its key columns (:class:`_Insert`): written again where a ``DELETE`` of the same
    table names its whole key; new where a column gets a new value -- a sequence's, one ``RETURNING`` hands back, a
    literal, one computed here (a conversion of one carried value is not); not said where a value is not read or
    is carried from an object of the table's class handed in (who made its key is the caller's); moved where every
    one is carried from a row of the table's class read here; else new, its key carried from something not of the
    table's class -- a scalar handed in, a parent's row.  A table no class maps is a class of its own."""
    first = key_froms[0]
    insert: _Insert = first._insert
    if insert.value == "none":
        return None, NEW_ROW_NO_KEY, first.statement
    if insert.rewritten is not None:
        return False, NEW_ROW_REWRITTEN, insert.rewritten
    values = [value for key_from in key_froms for value in _key_values(key_from, sites, table_class)]
    new = [value for value in values if value[0] == "new"]
    if new:
        why = min(new, key=lambda value: (_NEW_ROW_ORDER.index(value[1]), value[2] or 0))
        return True, why[1], why[2]
    for kind, why in (("upsert", NEW_ROW_UPSERT), ("unread", NEW_ROW_UNREAD), ("handed", NEW_ROW_HANDED)):
        found = [value for value in values if value[0] == kind]
        if found:
            return None, why, found[0][2]
    other = [value for value in values if value[0] == "other"]
    if not other:
        return False, NEW_ROW_MOVED, values[0][2] if values else first.statement
    return True, NEW_ROW_OTHER_CLASS, other[0][2]


def entity_nodes(
    uses: dict[str, EntityUse],
    map_id: str,
    derived_from: str,
    attributor: Optional[SpecAttributor] = None,
) -> list[EntityNode]:
    """Turn what :func:`read_side` saw into nodes, one per kind of row.

    No promotion.  The criteria decide whether asking "why is this attribute
    this value?" is a question worth having, which is a question about a
    column; an entity is not a candidate for it and would be judged by rules
    that read an attribute it does not have.

    **A table this corpus only reads is left out.**  It already has a node, and
    a better-fitting one: ``tables_never_written`` finds it and
    ``extract_read_only_tables`` makes it an unbound boundary -- a dependency on
    something outside, which is what a table nothing here maintains is.
    Recording it here as well would put one fact in two node types, the shape
    that ``log_files`` and ``opened_by`` have already cost this map twice.  The
    two readings agree on the installed corpus: the tables with no writing verb
    are exactly ``tables_never_written``'s thirty-nine.

    What is left is sixty-seven kinds of row where keying on the spec class
    gave thirteen, and the fifty-four that appear are the ones a stalled
    command is about.
    """
    return [
        EntityNode(
            name=table,
            map_id=map_id,
            derived_from=derived_from,
            spec_class=attributor.class_for_table(table) if attributor else None,
            tables=sorted(use.tables),
            read_by=sorted(use.read_by),
            created_by=sorted(use.created_by),
            updated_by=sorted(use.updated_by),
            deleted_by=sorted(use.deleted_by),
        )
        for table, use in sorted(uses.items())
        if use.created_by or use.updated_by or use.deleted_by
    ]


def selection_gates(
    modules: list[SourceModule], attributor: SpecAttributor, never_written: dict[str, str]
) -> dict[str, set[str]]:
    """Return ``{subject: tables bounding the queries that select on it}``.

    Read from the same statements as :func:`read_side` and kept beside it
    because the two are halves of one answer.  That one says a query asks for
    this value; this one says what limits which rows the query can see, and a
    task can be invisible for the second reason while the first is satisfied.
    That is not hypothetical: a task sat in ``finishing`` while the query that
    rescues it ran every cycle, because ``JEDI_AUX_Status_MinTaskID`` had
    stopped being updated and its watermark was above the task's id.

    Only tables *nothing in this map writes* count.  A join to a table the
    corpus maintains is a step in a query; a join to one it only ever reads is
    a dependency on something outside, and only the second can go stale in a
    way the map cannot account for.

    The membership test folds case, because :func:`tables_never_written` keys
    on the folded name: the corpus reads ``jobs_statuslog`` and writes
    ``jobs_StatusLog``, and comparing spellings made twenty subjects -- among
    them ``JobSpec.jobStatus`` -- carry a gate on a table PanDA maintains.
    """
    found: dict[str, set[str]] = {}
    for module in modules:
        for func, _owner in functions_with_owner(module.tree):
            seen: set[str] = set()
            for run in sql.executions(func):
                if run.sql in seen:
                    continue
                seen.add(run.sql)
                # Folded for the lookup and shown as ``never_written`` spells
                # it, so one table does not reach a reader under two names.
                gates = {
                    never_written[t.lower()]
                    for t in sqltext.joins(run.reading)
                    if t.lower() in never_written
                }
                if not gates:
                    continue
                for term in sqltext.terms(run.reading).terms:
                    if term.scope != sqltext.ROW or not term.binds:
                        continue
                    # A table does not bound itself.  The statement's gates are
                    # read once and offered to every term in it, so a subject
                    # sitting on the gating table got its own table back as the
                    # thing limiting which of its rows can be seen --
                    # ``JEDI_AUX_Status_MinTaskID.status`` bounded by
                    # ``JEDI_AUX_Status_MinTaskID``.  Seventeen of the
                    # fifty-eight gated subjects were that, and the verdict
                    # they produce tells a reader to go and find out who
                    # maintains a table they are already looking at.
                    bounds = {gate for gate in gates if gate.lower() != term.table.lower()}
                    if bounds:
                        found.setdefault(_term_subject(attributor, term), set()).update(bounds)
    return found


#: A table reference that is nothing but holes -- ``{}`` or ``{0}``.  What is
#: left when a name the source supplies at run time could not be resolved.
_UNREAD_TABLE = re.compile(r"(?:\{\d*\})+")


def _rows_touched(statement: str) -> list[tuple[str, str]]:
    """``[("read" | "created" | "updated" | "deleted", table)]`` for *statement*.

    "Who reads a task's jobs" cares about the verb and nothing else, and
    folding the verbs is how an ``UPDATE ... WHERE`` came to be reported as a
    query that selects on a value.
    """
    touched: list[tuple[str, str]] = [
        ("read", table) for table, _columns in sqltext.reads(statement)
    ]
    # The rest of the ``FROM`` list too.  :func:`sqltext.reads` keeps naming the
    # leading table because it answers *where the row came from*; the question
    # here is which kinds of row the statement touches at all, and a join
    # partner's rows are read as surely as the first table's.  Sixty-one
    # (function, entity) readings were missing for the difference, sixteen of
    # them on datasets and sixteen on files -- the two a task waits for.
    touched.extend(("read", table) for table in sqltext.joins(statement))
    # The three writing verbs stay apart.  Making a row, changing one and
    # removing one are three different claims about a kind of row, and pooling
    # them is why nothing downstream could ask where an object enters the
    # system.  ``sqltext.deletes``'s own docstring already said what that costs: a
    # DELETE followed by an INSERT on a command table means a second command
    # silently replaced one that was never picked up, and the map had nowhere
    # to put it.
    touched.extend(("created", table) for table in sqltext.creates(statement))
    touched.extend(("updated", table) for table in sqltext.updates(statement))
    touched.extend(("deleted", table) for table in sqltext.deletes(statement))
    return touched


def _table_verbs(statement: str) -> list[tuple[str, str]]:
    """``(verb, table)`` for each way *statement* touches a table, deduplicated.

    A statement naming one table under both verbs -- ``INSERT INTO a SELECT
    FROM a`` -- yields both, which is what it does.
    """
    return sorted(set(_rows_touched(statement)))
