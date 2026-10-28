"""Reading SQL text.

:mod:`~bamboo.codemap.panda.sql` reads a statement out of the Python that
builds it; this reads the text -- which tables it touches, which
columns it writes and with what, which values it asks rows by, and in what
order and how many come back.  Nothing here looks at Python: every function
takes the statement as a string.

Two readings, for now.  What a reader asks of rows -- the values
(:func:`terms`), what else a row had to be (:func:`conditions`), the order and
the limit (:func:`ordering`) -- and the tables a statement touches
(:func:`reads`, :func:`joins`, :func:`creates` ...) are read off sqlglot's
tree (:func:`parsed`), because only the tree says which table a column belongs
to and which terms are a subquery's.  What a statement writes is still regexes
over the text, enough for ``UPDATE ... SET`` and ``INSERT INTO ... VALUES`` and
nothing more.  Either way, what cannot be read is reported rather than guessed
at.
"""

from __future__ import annotations

import ast
import contextlib
import datetime
import functools
import itertools
import re
from pathlib import Path
from typing import Iterator, NamedTuple, Optional

import sqlglot
from pydantic import BaseModel, Field
from sqlglot import exp
from sqlglot.errors import ErrorLevel, ParseError, SqlglotError
from sqlglot.optimizer.scope import ScopeType, traverse_scope

# ``UPDATE [/*+ hint */] <schema>.<table> [alias] SET <assignments> [WHERE ...]``.
# The schema is usually an f-string placeholder, which the reading leaves as
# ``{}``.  The optimizer hint and the table alias are both optional and both
# occur: ``UPDATE /*+ index(tab ...) */ ATLAS_PANDA.filesTable4 tab SET
# status='ready'`` is one statement, and without either allowance it reads as no
# statement at all.
_UPDATE = re.compile(
    r"\bUPDATE\s+(?:/\*.*?\*/\s*)?([\w{}.]+)(?:\s+(?!SET\b)\w+)?\s+SET\s+(.*?)(?:\bWHERE\b|$)",
    re.IGNORECASE | re.DOTALL,
)
_INSERT = re.compile(
    r"\bINSERT\s+INTO\s+([\w{}.]+)\s*\(([^)]*)\)\s*VALUES\s*\(([^)]*)\)",
    re.IGNORECASE | re.DOTALL,
)
_ASSIGNMENT = re.compile(r"([A-Za-z_]\w*)\s*=\s*(:?[A-Za-z_]\w*|[^,]+?)(?=\s*,|\s*$)")
_IDENTIFIER = re.compile(r"[A-Za-z_]\w*")
_WHERE = re.compile(r"\bWHERE\b", re.IGNORECASE)

_BARE = re.compile(r"^[A-Za-z_]\w*$")
_NUMBER = re.compile(r"^[+-]?\d+(\.\d+)?$")
# Bare words that are SQL, not columns.  This has to stay a list of *keywords*
# rather than of names that look like plumbing: ``T_TASK`` really does have a
# column called ``timeStamp``, and ``JEDI_Events`` one called ``event_offset``,
# both of which are copied into other columns.  ``NULL`` is settled earlier as
# a literal and never reaches here; it stays listed because what the set means
# is "not a column name", which is true of it however it is read.
_KEYWORDS = frozenset(
    {"NULL", "CURRENT_DATE", "CURRENT_TIMESTAMP", "SYSDATE", "SYSTIMESTAMP", "DEFAULT", "TRUE", "FALSE"}
)


class ColumnValue(BaseModel):
    """Where one column's new value comes from.

    The four forms are four different kinds of branch, which is why the
    distinction is drawn here rather than left to the caller:

    ``bind``
        ``SET status=:status`` -- the value is decided in Python, at the
        assignment filling the bind, and that is where the ``if`` explaining it
        is too.
    ``literal``
        ``SET status='ready'``, ``SET oldStatus=NULL``, ``SET coreCount=0`` --
        decided in the statement itself.  Quoting is a property of the type,
        not of how settled the value is, and reading only the quoted ones was
        the same mistake the attribute slice made in only reading string
        right-hand sides.
    ``column``
        ``SET status=oldStatus`` -- carried from another column of the same
        row.  This is the form the recognizers were blind to, and it is not a
        curiosity: it is how a task returns from ``pending``, so without it the
        map cannot offer "the release did not fire" as a candidate at all.
    ``expression``
        ``CURRENT_DATE``, ``nFiles+1``, a subquery.  A write whose value the
        statement does not settle: the clock and the row's own prior contents
        are read when it runs.
    """

    kind: str = Field(..., description="bind | literal | column | expression")
    text: str = Field(
        ...,
        description="Bind key, literal value, source column, or the raw SQL, per ``kind``.",
    )


def classify(value: str) -> ColumnValue:
    """Classify the right-hand side of one SQL column assignment.

    ``NULL`` is reported as Python's ``None`` because the two spell one fact,
    and the column it clears is one subject: ``JediTaskSpec.oldStatus`` is
    written ``= None`` through the attribute slice and ``=NULL`` through this
    one, and a branch table offering both spellings would read as two outcomes
    where the source has one.  Python's is the spelling that wins for the same
    reason ``runtime(...)`` holds unparsed Python -- everything downstream that
    compares an outcome to an observed value is reading PanDA through Python.
    """
    value = value.strip()
    if value.startswith(":"):
        return ColumnValue(kind="bind", text=value)
    quoted = _QUOTED.match(value)
    if quoted is not None:
        return ColumnValue(kind="literal", text=quoted.group(1))
    if value.upper() == "NULL":
        return ColumnValue(kind="literal", text="None")
    if _NUMBER.match(value):
        return ColumnValue(kind="literal", text=value)
    if _BARE.match(value) and value.upper() not in _KEYWORDS:
        return ColumnValue(kind="column", text=value)
    return ColumnValue(kind="expression", text=value)


def assigns_in(fragment: str) -> list[tuple[str, ColumnValue]]:
    """Return the ``column = value`` pairs a SQL *fragment* sets.

    Everything from the first ``WHERE`` on is dropped: a predicate uses the
    same ``column = value`` spelling as an assignment, so a fragment carrying
    both would report its selection criteria as writes.
    """
    head = _WHERE.split(fragment, maxsplit=1)[0]
    return [(column, classify(value)) for column, value in _ASSIGNMENT.findall(head)]


class SqlWrite(BaseModel):
    """One ``UPDATE ... SET`` or ``INSERT``, as far as it could be read."""

    kind: str = Field(..., description="update | insert")
    table: str = Field(..., description="Table name, schema stripped.")
    columns: dict[str, ColumnValue] = Field(
        default_factory=dict,
        description=(
            "Column -> where its new value comes from.  Columns written to an "
            "expression are kept even though they carry nothing to trace: they "
            "are part of what identifies the table."
        ),
    )
    preconditions: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Column -> the predicate testing it, for columns this statement "
            "also writes.  That combination is a compare-and-set: whoever moved "
            "the row first wins and this write changes no rows, returning a "
            "count the caller usually discards.  Predicates on columns the "
            "statement does not write are row identity, not a race."
        ),
    )
    keys: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Column -> the bind it is equated to, for the ``WHERE`` terms on "
            "columns this statement does not write: ``PandaID=:PandaID``.  The "
            "row identity the preconditions leave out, and where to look to "
            "say who named the rows -- see ``sqlwrite``'s ``RowKeySource``."
        ),
    )
    key_lists: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Column -> the one bind an ``IN`` list is spelt as, for the ``WHERE`` terms on columns this statement "
            "does not write that name its rows by a list a helper builds: ``datasetID IN (:dataset__in)`` "
            "(``sql.in_lists``).  Kept out of *keys*, whose readers take an equality: a list names several rows.  "
            "Read off the tree only."
        ),
    )
    precondition_spans: dict[str, list[tuple[int, int]]] = Field(
        default_factory=dict,
        description=(
            "Column -> where each conjunct of its precondition sits in the text "
            "read, as start and end offsets, so a caller holding the pieces "
            "the text was built from can say which wrote it.  Empty where the "
            "statement was read off its text rather than its tree."
        ),
    )
    key_holes: list[tuple[int, str, tuple[int, ...]]] = Field(
        default_factory=list,
        description=(
            "The ``WHERE`` terms ``{}=:cr_{}`` -- a column that is a hole, equated to a bind whose name holds one: "
            "the column's hole number (``__h3`` is 3), the bind spelt with ``{}`` for its holes (``:cr_{}``), and "
            "their numbers.  Kept out of *keys*, which name a column: what the holes are says whether the term "
            "names its row (``sql.map_key``), and that is the run's to read."
        ),
    )
    column_holes: list[tuple[int, str, tuple[int, ...]]] = Field(
        default_factory=list,
        description=(
            "The ``SET`` items ``{}=:{}`` -- a column that is a hole, given a bind whose name holds one -- as "
            "*key_holes* spells a term: ``updateJobStatus``' ``,{key}=:{key}`` for each key of ``param``.  Not in "
            "*columns*, and each still a :data:`SET_COLUMN_IS_A_HOLE`: which columns the holes are is the run's "
            "to read."
        ),
    )


#: The end of the one bind an ``IN`` list built by ``get_sql_IN_bind_variables`` is spelt as: ``:stat__in`` for the
#: prefix ``:stat`` (``sql.in_lists``).
IN_LIST_SUFFIX = "__in"


#: A single ``WHERE``/``AND``/``OR`` term, kept whole so the guard reads as the
#: source wrote it.  ``NOT`` is part of the term rather than a separate one:
#: ``updateJobStatus`` guards with ``AND NOT jobStatus=:ngStatus``, and dropping
#: the negation would report the opposite condition.
_TERM = re.compile(
    r"(?:WHERE|AND|OR)\s+((?:NOT\s+)?(?:\w+\.)?([A-Za-z_]\w*)\s*"
    r"(?:(?:=|<>|!=|<=|>=|<|>)\s*[^\s)]+|(?:NOT\s+)?IN\s*\([^)]*\)|IS\s+(?:NOT\s+)?NULL))",
    re.IGNORECASE,
)


def _preconditions(clause: str, columns: set[str]) -> dict[str, str]:
    """Return the terms of *clause* testing a column in *columns*.

    A statement that tests a column it also assigns is doing a compare-and-set,
    and losing that race is silent: no exception, no log, just a row count the
    caller discards.  The map has no other way to say that seeing a value
    *decided* is not the same as seeing the row take it.
    """
    found: dict[str, str] = {}
    lowered = {column.lower(): column for column in columns}
    for term, column in _TERM.findall(clause):
        owner = lowered.get(column.lower())
        if owner is not None and owner not in found:
            found[owner] = re.sub(r"\s+", " ", term).strip()
    return found


#: ``PandaID=:PandaID`` -- a term naming one column by one bind.
_KEY_TERM = re.compile(r"^(?:\w+\.)?([A-Za-z_]\w*)\s*=\s*(:\w+)$")


def _keys(clause: str, columns: set[str]) -> dict[str, str]:
    """``{column: bind}`` for the terms of *clause* on columns not in *columns*.

    Read with the same term pattern :func:`_preconditions` uses, so the two
    split one ``WHERE`` between them and cannot both claim a term.  A negated
    term and a list are not a key: neither names *the* row.
    """
    found: dict[str, str] = {}
    lowered = {column.lower() for column in columns}
    for term, column in _TERM.findall(clause):
        if column.lower() in lowered:
            continue
        match = _KEY_TERM.match(re.sub(r"\s+", " ", term).strip())
        if match is not None and match.group(1) not in found:
            found[match.group(1)] = match.group(2)
    return found


def _writes_in_text(sql: str) -> list[SqlWrite]:
    """:func:`writes` for a statement that does not parse, off its text."""
    found: list[SqlWrite] = []
    for match in _UPDATE.finditer(sql):
        columns = {
            column: classify(value) for column, value in _ASSIGNMENT.findall(match.group(2))
        }
        if columns:
            found.append(
                SqlWrite(
                    kind="update",
                    table=_table_of(match.group(1)),
                    columns=columns,
                    preconditions=_preconditions(sql[match.end(2) :], set(columns)),
                    keys=_keys(sql[match.end(2) :], set(columns)),
                )
            )
    for match in _INSERT.finditer(sql):
        columns = {name: supplied for name, supplied, _at in _insert_supplies(match)}
        if columns:
            found.append(
                SqlWrite(kind="insert", table=_table_of(match.group(1)), columns=columns)
            )
    return found


def _insert_supplies(match: re.Match) -> list[tuple[str, ColumnValue, tuple[int, int]]]:
    """``(column, value, where the value sits)`` for one ``INSERT`` match.

    One reading for the two questions asked of an ``INSERT``: which columns it
    writes, and where in the statement each value was written.  Kept together
    because pairing a column with a value here is positional -- the names are
    in one parenthesised run and the values in another -- and a second reading
    that paired them its own way would be free to pair them differently.

    The offsets are into the same string the match was taken from, so a caller
    that matched on an assembled variant can map a value back to the fragment
    that contributed it.
    """
    names = [c.strip().split(".")[-1] for c in match.group(2).split(",")]
    supplies: list[tuple[str, ColumnValue, tuple[int, int]]] = []
    at = match.start(3)
    for name, value in zip(names, match.group(3).split(","), strict=False):
        start, end = at, at + len(value)
        at = end + 1  # the comma the split consumed
        name = name.strip()
        if not _IDENTIFIER.fullmatch(name):
            continue
        supplied = classify(value)
        if supplied.kind == "column":
            # An INSERT has no prior row to copy from, so a bare word in its
            # VALUES list is a sequence or a function, not a source column.
            # Calling it one would invent a passthrough edge.
            supplied = ColumnValue(kind="expression", text=supplied.text)
        supplies.append((name, supplied, (start, end)))
    return supplies


def _supplied_at_in_text(
    statement: str, kind: str, column: str, supplied: ColumnValue
) -> Optional[int]:
    """:func:`supplied_at` for a statement that does not parse, off its text."""
    if kind == "insert":
        for match in _INSERT.finditer(statement):
            for name, value, (_start, end) in _insert_supplies(match):
                if name == column and value == supplied:
                    return end
        return None
    # Everything from the first ``WHERE`` on is a predicate, which uses the
    # same ``column = value`` spelling as an assignment -- see ``assigns_in``.
    where = _WHERE.search(statement)
    head = statement[: where.start()] if where else statement
    for match in _ASSIGNMENT.finditer(head):
        if match.group(1) == column and classify(match.group(2)) == supplied:
            return match.end()
    return None


def _table_of(reference: str) -> str:
    """Strip the schema: ``{}.JEDI_Tasks`` and ``ATLAS_PANDA.jobsActive4``."""
    return reference.split(".")[-1]


# The whole FROM list, aliases and all: ``FROM {0}.JEDI_Tasks tabT,
# {0}.JEDI_AUX_Status_MinTaskID tabA``.
_FROM_LIST = re.compile(
    r"\bFROM\s+((?:[\w{}.]+(?:\s+\w+)?\s*,\s*)*[\w{}.]+(?:\s+\w+)?)", re.IGNORECASE
)

# ``WITH tmpTab AS (SELECT ...)`` -- a name the statement defines for itself,
# and the second and later ones after a comma.  The word boundary goes before
# ``WITH`` only: a comma is not a word character, so ``\b,`` demands one to its
# left and ``), wq_results AS (`` -- the spelling every multi-clause WITH in
# this corpus uses -- did not match.
_CTE = re.compile(r"(?:\bWITH|,)\s+(\w+)\s+AS\s*\(", re.IGNORECASE)

# ``JOIN {0}.JEDI_WORK_QUEUE jwq ON ...`` -- the other spelling of a join
# partner, which ``_FROM_LIST`` cannot see because it is not in the comma list.
_JOIN = re.compile(r"\bJOIN\s+([\w{}.]+)", re.IGNORECASE)

#: Oracle's one-row pseudo-table.  ``SELECT <seq>.nextval FROM dual`` asks a
#: sequence for a number; ``dual`` sits where a table goes and holds none of
#: PanDA's rows.  ``WrappedCursor`` deletes ``FROM dual`` outright when the
#: backend is not Oracle, which settles what it is.
_PSEUDO_TABLES = frozenset({"dual"})

#: SQL's own words, written where a column goes.  ``recoverLostFiles`` pads a
#: ``UNION`` with ``SELECT fileID,datasetID,lfn,NULL`` so its projection lines up
#: with the branch that selects ``outPandaID`` in that position, and
#: ``closeTaskAttempt`` asks the database for the time with ``SELECT
#: CURRENT_DATE,status``.  Neither names anything the row carries -- and the
#: first is worse than absent, since it stands where a real column stands.
#:
#: **Kept to the words this corpus writes there.** Dropping SQL's reserved words
#: as a class would delete real columns: ``get_shares`` selects twelve from
#: ``GLOBAL_SHARES`` and ``NAME``, ``VALUE`` and ``PARENT`` are three of them,
#: spelled in upper case exactly like the literals are.  So case cannot be the
#: filter and a longer list is a liability rather than insurance.
_PSEUDO_COLUMNS = frozenset({"null", "current_date", "current_timestamp"})


def _names_a_table(name: str, defined: set[str]) -> bool:
    """Whether *name*, written where a table goes, names one that stores rows.

    Two ways it can fail to, and both belong wherever a table position is
    read: the statement may have introduced the name itself in a ``WITH``
    clause, and it may be Oracle's ``dual``.  Either one counted as a table is
    read by the corpus and written by none of it, which is the exact shape
    :func:`~bamboo.codemap.panda.recognizers.boundary.tables_never_written`
    turns into a claim that some other system maintains it.

    An interpolated name -- a bare ``{}`` -- is *not* decided here, because the
    two callers want opposite things from it.  :func:`joins` drops it, since a
    partner nobody can name cannot be said to bound anything; :func:`reads`
    keeps it, since the reader downstream counts those and the build report
    says how many statements and which functions.  Dropping it there would
    turn a counted gap into silence.
    """
    folded = name.lower()
    return bool(name) and folded not in defined and folded not in _PSEUDO_TABLES


_SELECT = re.compile(r"\bSELECT\s+(?:DISTINCT\s+)?(.*?)\s+FROM\s+([\w{}.]+)", re.IGNORECASE | re.DOTALL)
_DELETE = re.compile(r"\bDELETE\s+FROM\s+([\w{}.]+)", re.IGNORECASE)


def _reads_in_text(sql: str) -> list[tuple[str, list[str]]]:
    """:func:`reads` for a statement that does not parse, off its text.

    Fifteen statements of the corpus do not, five of them reading a table --
    ``getJobs``'s ``jobsActive4`` among them -- and dropping their tables
    would take readers off entities and could turn a table into one nothing
    writes.
    """
    defined = {match.group(1).lower() for match in _CTE.finditer(sql)}
    found: list[tuple[str, list[str]]] = []
    for match in _SELECT.finditer(sql):
        table = _table_of(match.group(2))
        if not _names_a_table(table, defined):
            continue
        columns = [
            name
            for name in (
                part.strip().split(".")[-1] for part in match.group(1).split(",")
            )
            if _IDENTIFIER.fullmatch(name) and name.lower() not in _PSEUDO_COLUMNS
        ]
        found.append((table, columns))
    return found


def _joins_in_text(sql: str) -> list[str]:
    """:func:`joins` for a statement that does not parse, off its text."""
    defined = {match.group(1).lower() for match in _CTE.finditer(sql)}
    found: list[str] = []

    def offer(reference: str) -> None:
        table = _table_of(reference)
        # ``{}`` means the name was interpolated, so the statement does not say
        # which table this is, and a partner nobody can name bounds nothing.
        if table == "{}" or not _names_a_table(table, defined):
            return
        if table not in found:
            found.append(table)

    for match in _FROM_LIST.finditer(sql):
        parts = [part.strip() for part in match.group(1).split(",")]
        for part in parts[1:]:
            offer(part.split()[0] if part.split() else "")
    for match in _JOIN.finditer(sql):
        offer(match.group(1))
    return found


def _joined_columns_in_text(sql: str, table: str) -> list[str]:
    """:func:`joined_columns` for a statement that does not parse, off its text."""
    found: dict[str, str] = {}
    for match in _FROM_LIST.finditer(sql):
        for part in match.group(1).split(","):
            words = part.strip().split()
            if len(words) != 2 or _table_of(words[0]) != table:
                continue
            for column in re.finditer(rf"\b{re.escape(words[1])}\.(\w+)", sql):
                found.setdefault(column.group(1).lower(), column.group(1))
    return sorted(found.values())


#: ``INSERT INTO <table>`` with nothing required after it.  ``_INSERT`` wants a
#: column list and a ``VALUES`` list because it is answering which columns get
#: which values; this answers only which kind of row is being made, and the
#: corpus has five statements where the first question has no answer and the
#: second does -- two ``INSERT ... SELECT``, one positional insert with no
#: column list, and two whose column list is interpolated.
_INSERT_HEAD = re.compile(r"\bINSERT\s+INTO\s+([\w{}.]+)", re.IGNORECASE)


def _creates_in_text(sql: str) -> list[str]:
    """:func:`creates` for a statement that does not parse, off its text."""
    return [_table_of(match.group(1)) for match in _INSERT_HEAD.finditer(sql)]


def _updates_in_text(sql: str) -> list[str]:
    """:func:`updates` for a statement that does not parse, off its text."""
    return [_table_of(match.group(1)) for match in _UPDATE.finditer(sql)]


def _deletes_in_text(sql: str) -> list[str]:
    """:func:`deletes` for a statement that does not parse, off its text."""
    return [_table_of(match.group(1)) for match in _DELETE.finditer(sql)]


def _named_table(node, defined: set[str]) -> Optional[str]:
    """*node*'s table as a statement's tables are spelled, if it names one that stores rows."""
    if not isinstance(node, exp.Table):
        return None
    table = _table_name(node)
    return table if _names_a_table(table, defined) else None


def _tables_tree(sql: str):
    """*sql*'s tree and the names its ``WITH`` clauses define, or ``None`` where it does not parse."""
    tree = parsed(sql).tree
    if tree is None:
        return None
    return tree, {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}


def _selects(tree: exp.Expression) -> list[exp.Select]:
    """Every ``SELECT`` in *tree*, in the order the text writes them."""
    return list(tree.find_all(exp.Select, bfs=False))


def _sources(select: exp.Select) -> list[exp.Expression]:
    """What a ``SELECT`` reads rows from: its ``FROM``, then each join, comma or ``JOIN``."""
    head = select.args.get("from_")
    return ([head.this] if head is not None else []) + [join.this for join in select.args.get("joins") or []]


def reads(sql: str) -> list[tuple[str, list[str]]]:
    """Return ``(table, selected columns)`` for each ``SELECT`` in *sql*.

    The table is the one the ``SELECT`` names first in its ``FROM``; one whose
    ``FROM`` is a subquery reads that subquery's rows, and the subquery is a
    ``SELECT`` of its own.  The text reading this replaced took the first name
    after the next ``FROM`` it found, which for ``SELECT t_status ... FROM
    (SELECT tabT.status AS t_status ... FROM JEDI_Tasks tabT ...)`` filed the
    outer projection under ``JEDI_Tasks``.

    Only the columns that are plain names are kept.  A projection built from
    expressions (``COUNT(1)``, ``CASE WHEN ...``, ``COALESCE(t.currentPriority,
    ...)``) says what the query computes rather than what the row carries, and
    a boundary is about the latter.  A bare word that is one of SQL's own --
    see :data:`_PSEUDO_COLUMNS` -- is dropped for that same reason, written
    without the parentheses that would have given it away.  A hint before the
    first column (``SELECT /*+ INDEX(...) */ PandaID``) is a comment, and the
    text reading lost the column behind it.

    An interpolated column is no name.  A statement that does not parse is
    read off its text, as before -- see :func:`_reads_in_text`.
    """
    read = _tables_tree(sql)
    if read is None:
        return _reads_in_text(plain(sql))
    tree, defined = read
    found: list[tuple[str, list[str]]] = []
    for select in _selects(tree):
        sources = _sources(select)
        table = _named_table(sources[0], defined) if sources else None
        if table is None:
            continue
        columns = [
            column.name
            for column in select.expressions
            if isinstance(column, exp.Column)
            and not column.name.startswith(_HOLE_NAME)
            and _IDENTIFIER.fullmatch(column.name)
            and column.name.lower() not in _PSEUDO_COLUMNS
        ]
        found.append((table, columns))
    return found


def aggregated(sql: str) -> Optional[frozenset[bool]]:
    """Whether each arm of *sql*'s outermost ``SELECT`` aggregates: ``{True}``, ``{False}``, both, or ``None``.

    An arm aggregates where an expression it projects calls an aggregate --
    ``COUNT``, ``MAX``, ``SUM`` ... -- not under ``OVER`` and not in a
    subquery: what it returns is computed from the rows, one per group or one
    in all, and is no row of a table.  ``SELECT ramCount, count(*) ... GROUP BY
    ramCount`` aggregates; ``SELECT status ... GROUP BY status`` and ``MAX(x)
    OVER (...)`` do not.  Each arm a row can come from is read (:func:`_arms`), so
    one text can answer both.  ``None`` where *sql* does not parse or is not a ``SELECT``.

    Nor does an arm each of whose rows stands for one row of a table it reads
    (:func:`_one_row_each`), inside :func:`with_schema` with keys: its
    aggregates are values beside that row.  ``throttleTasks_JEDI`` groups
    ``JEDI_Tasks`` by ``jediTaskID`` with the files' ``AVG(failedAttempt)``,
    and throttles each task by its ID.
    """
    tree = parsed(sql).tree
    arms = _arms(tree) if tree is not None else []
    if not arms:
        return None
    return frozenset(
        any(_aggregates(column) for column in arm.expressions) and not _one_row_each(arm) for arm in arms
    )


def _one_row_each(arm: exp.Select) -> bool:
    """Whether every column of the key of a table *arm* reads is grouped or fixed to one value by its ``WHERE``.

    Fixed: an ``AND`` of the ``WHERE`` is ``column = :bind`` or ``column =
    'literal'`` -- ``prepareTasksToBeFinished_JEDI`` counts a task's files per
    ``tabD.datasetID`` with ``tabD.jediTaskID=:jediTaskID``, one row per
    dataset of ``JEDI_Datasets``.  A column is the table's where it names the
    table's alias or name, or no table: then the table is the one that has it,
    a key column being one of its own.
    """
    group = arm.args.get("group")
    if _KEYS is None or group is None:
        return False
    where = arm.args.get("where")
    conjuncts = list(where.this.flatten()) if where is not None and isinstance(where.this, exp.And) else []
    if where is not None and not conjuncts:
        conjuncts = [where.this]
    fixed = [
        one.this
        for one in conjuncts
        if isinstance(one, exp.EQ)
        and isinstance(one.this, exp.Column)
        and isinstance(one.expression, (exp.Placeholder, exp.Literal))
    ]
    held = [one for one in group.expressions if isinstance(one, exp.Column)] + fixed
    for source in _sources(arm):
        key = _KEYS.get(source.name.lower()) if isinstance(source, exp.Table) else None
        if not key:
            continue
        spelled = {source.name.lower(), source.alias_or_name.lower()}
        if key <= {column.name.lower() for column in held if not column.table or column.table.lower() in spelled}:
            return True
    return False


def _arms(tree: exp.Expression) -> list[exp.Select]:
    """The ``SELECT`` s whose rows a set operation returns, each arm in parentheses or not; ``[]`` where one is
    something else.  A row of ``A EXCEPT B`` (``MINUS``) or ``A INTERSECT B`` is a row of ``A``; of a ``UNION``, of
    either arm."""
    if isinstance(tree, exp.SetOperation):
        left, right = _arms(tree.left), _arms(tree.right)
        if not (left and right):
            return []
        return left + right if isinstance(tree, exp.Union) else left
    if isinstance(tree, exp.Subquery):
        return _arms(tree.this)
    return [tree] if isinstance(tree, exp.Select) else []


def _aggregates(node: exp.Expression) -> bool:
    """Whether *node* calls an aggregate, not looking under ``OVER`` or into a subquery."""
    if isinstance(node, (exp.Select, exp.Subquery, exp.Window)):
        return False
    if isinstance(node, exp.AggFunc):
        return True
    return any(_aggregates(child) for child in node.iter_expressions())


def joins(sql: str) -> list[str]:
    """Return the tables *sql* reads beside the first of a ``FROM``.

    Separate from :func:`reads`, which answers *where the row came from* and so
    keeps naming the leading table.  This answers a different question -- *what
    bounds which rows can be seen at all* -- and a join partner is the usual
    way that bound is written::

        FROM {0}.JEDI_Tasks tabT,{0}.JEDI_AUX_Status_MinTaskID tabA
        WHERE tabT.status=tabA.status AND tabT.jediTaskID>=tabA.min_jediTaskID

    A task below ``min_jediTaskID`` is invisible to that query no matter what
    its status is, so a stale ``JEDI_AUX_Status_MinTaskID`` silently narrows
    thirty-one functions at once.  A comma and a ``JOIN`` are the same partner.

    A name the statement defines for itself is not one of these
    (``checkDuplication_JEDI`` joins its own ``tmpTab``), and neither is an
    interpolated name: a partner nobody can name bounds nothing.
    """
    read = _tables_tree(sql)
    if read is None:
        return _joins_in_text(plain(sql))
    tree, defined = read
    found: list[str] = []
    for select in _selects(tree):
        for source in _sources(select)[1:]:
            table = _named_table(source, defined)
            if table is not None and table != "{}" and table not in found:
                found.append(table)
    return found


def joined_columns(sql: str, table: str) -> list[str]:
    """Columns of *table* the statement names, found through the alias it joins it by.

    The alias is how a join predicate refers to a table --
    ``tabA.min_jediTaskID`` -- so without resolving it the most useful thing
    about a join, *which column bounds the rows*, cannot be read.  Only the
    aliases the table is read by: ``DELETE FROM Task_Evaluation te WHERE ...
    IN (SELECT ... FROM JEDI_Tasks jt, Task_Evaluation tez ...)`` joins it as
    ``tez``, and ``te``'s columns are the rows being deleted.
    """
    read = _tables_tree(sql)
    if read is None:
        return _joined_columns_in_text(plain(sql), table)
    tree, defined = read
    aliases = {
        source.alias
        for select in _selects(tree)
        for source in _sources(select)
        if source.alias and _named_table(source, defined) == table
    }
    found: dict[str, str] = {}
    for column in tree.find_all(exp.Column, bfs=False):
        if column.table and column.table in aliases:
            found.setdefault(column.name.lower(), column.name)
    return sorted(found.values())


def _targets(sql: str, kind: type) -> Optional[list[str]]:
    """The tables *sql*'s statements of *kind* write to, or ``None`` where it does not parse."""
    read = _tables_tree(sql)
    if read is None:
        return None
    tree, defined = read
    found: list[str] = []
    for node in tree.find_all(kind, bfs=False):
        target = node.this.this if isinstance(node.this, exp.Schema) else node.this
        table = _named_table(target, defined)
        if table is not None:
            found.append(table)
    return found


def creates(sql: str) -> list[str]:
    """Return the tables *sql* inserts rows into.

    Separate from :func:`writes` on purpose.  *Which columns get which values*
    and *which kind of row is being made* are two questions, and deriving the
    second from the first made a table look externally maintained whenever the
    first could not be answered: ``TASK_ATTEMPTS`` is filled by an
    ``INSERT ... SELECT`` in ``log_task_attempt_start``.  A hint between
    ``INSERT`` and ``INTO`` hid ``CARBON_REGION_EMISSIONS`` from the text
    reading the same way.
    """
    found = _targets(sql, exp.Insert)
    return _creates_in_text(plain(sql)) if found is None else found


def updates(sql: str) -> list[str]:
    """Return the tables *sql* changes rows in.  ``SELECT ... FOR UPDATE`` changes none."""
    found = _targets(sql, exp.Update)
    return _updates_in_text(plain(sql)) if found is None else found


def deletes(sql: str) -> list[str]:
    """Return the tables *sql* deletes rows from.

    Worth reading separately because a ``DELETE`` immediately followed by an
    ``INSERT`` on a command table is not bookkeeping: it means a second command
    silently replaces one that was never picked up.
    """
    found = _targets(sql, exp.Delete)
    return _deletes_in_text(plain(sql)) if found is None else found


def _statement(sql: str):
    """*sql*'s ``UPDATE`` or ``INSERT`` node and its tokens, or ``None`` for any other statement.

    The tokens are of *sql* itself, holes and all -- its repeated parts'
    passes in place (:func:`_source`), which :func:`_at_plain` maps back --
    so an offset taken from them is an offset into what the caller holds --
    the assembled variant a fragment's span is measured in -- and not into
    the text the tree was parsed from, where every hole is a name of another
    length.
    """
    tree = parsed(sql).tree
    while isinstance(tree, exp.Subquery):
        tree = tree.this
    if not isinstance(tree, (exp.Update, exp.Insert)):
        return None
    return tree, _tokens(_source(sql)[0])


def _spelled(sql: str, tokens: list, lo: int, hi: int, written: bool = False) -> str:
    """``tokens[lo:hi]`` as *sql* spells them, an ``IN`` list's repeated items once (:func:`_one_of_each`) and a
    repeated part ``{}`` (:data:`REPEAT_OPEN`) -- or with *written* as its passes wrote it: the value a column
    takes on each pass that writes the column.  The tokens are of :func:`_source`'s text."""
    text, regions = _source(sql)
    if hi <= lo:
        text = ""
    elif written:
        text = text[tokens[lo].start : tokens[hi - 1].end + 1].strip()
    else:
        text = _spell(text, regions, tokens[lo].start, tokens[hi - 1].end + 1).strip()
    return _IN_LIST.sub(_listed_once, text) if "," in text else text


_IN_LIST = re.compile(r"(\bIN\s*\()([^()]*)(\))", re.IGNORECASE)


def _listed_once(match: re.Match) -> str:
    items = match.group(2).split(",")
    kept: dict[str, str] = {}
    for item in items:
        kept.setdefault(_NAMED_HOLE.sub("{}", item.strip()), item)
    if len(kept) == len(items):
        return match.group(0)
    return match.group(1) + ",".join(kept.values()) + match.group(3)


def _at_depth(tokens: list, start: int, kinds, stop: Optional[int] = None) -> Optional[int]:
    """The first token from *start* of a type in *kinds*, outside any parentheses opened after it."""
    depth = 0
    for index in range(start, len(tokens) if stop is None else stop):
        kind = tokens[index].token_type
        if kind in _OPENS:
            depth += 1
        elif kind in _CLOSES:
            depth -= 1
        elif depth == 0 and kind in kinds:
            return index
    return None


def _items(tokens: list, lo: int, hi: int) -> list[tuple[int, int]]:
    """``tokens[lo:hi]`` cut at its own commas, as ``(lo, hi)`` per item."""
    bounds = [lo - 1, *_cuts(tokens, lo, hi, _T.COMMA), hi]
    return [(a + 1, b) for a, b in zip(bounds, bounds[1:], strict=False)]


def _supplied(node: exp.Expression, spelled: str, insert: bool) -> ColumnValue:
    """What *node*, spelled *spelled*, writes into a column.

    The kind is :func:`classify`'s, read off the spelling, unless the tree
    says the value is not one of the simple four: ``'a' || :b`` starts with a
    quoted value and is an expression.  An ``INSERT`` has no prior row to copy
    from, so a bare word in it is a sequence or a function, not a source
    column -- calling it one would invent a passthrough edge.
    """
    value = classify(spelled)
    simple = (
        isinstance(node, (exp.Placeholder, exp.Literal, exp.Null))
        or (isinstance(node, exp.Neg) and isinstance(node.this, exp.Literal) and not node.this.is_string)
        or (isinstance(node, exp.Column) and not node.table)
    )
    if not simple or (insert and value.kind == "column"):
        value = ColumnValue(kind="expression", text=spelled)
    return value


def _update_pairs(
    node: exp.Update,
    sql: str,
    tokens: list,
    passed: Optional[list[tuple[str, str]]] = None,
    holes: Optional[list[tuple[int, str, tuple[int, ...]]]] = None,
) -> Optional[list[tuple[str, int, ColumnValue]]]:
    """``(column, end offset, value)`` for each ``SET`` item of *node*, or ``None`` where the two readings disagree.

    An item it cannot pair goes into *passed*, where given, as ``(the item as
    the statement spells it, why)`` -- see :func:`unread_columns`; one whose
    column is a hole and whose bind holds one goes into *holes* as well, where
    given (:attr:`SqlWrite.column_holes`).
    """
    update = _at_depth(tokens, 0, {_T.UPDATE})
    assign = _at_depth(tokens, update + 1, {_T.SET}) if update is not None else None
    if assign is None:
        return None
    end = _at_depth(tokens, assign + 1, _ENDS_A_CLAUSE | {_T.WHERE})
    spans = _items(tokens, assign + 1, len(tokens) if end is None else end)
    if len(spans) != len(node.expressions):
        return None
    found: list[tuple[str, int, ColumnValue]] = []
    for (lo, hi), item in zip(spans, node.expressions, strict=True):
        if not isinstance(item, exp.EQ) or not isinstance(item.this, exp.Column):
            if passed is not None:
                # ``SET {}`` -- a spec's bindUpdateChangesExpression(): the
                # assignments themselves are Python's.
                whole = isinstance(item, exp.Column) and _NAMED_HOLE.search(item.name)
                passed.append(
                    ("SET " + _spelled(sql, tokens, lo, hi, written=True), SET_ITEM_IS_A_HOLE if whole else SET_ITEM_NOT_READ)
                )
            continue
        # ``SET merge{}={}`` -- a column Python names at run time.
        name = item.this.name
        equals = _at_depth(tokens, lo, {_T.EQ}, hi)
        if _NAMED_HOLE.search(name) or equals is None:
            if passed is not None:
                passed.append(
                    ("SET " + _spelled(sql, tokens, lo, hi, written=True), SET_COLUMN_IS_A_HOLE if equals is not None else SET_ITEM_NOT_READ)
                )
            if holes is not None and equals is not None and (held := _key_hole(item)) is not None:
                holes.append(held)
            continue
        found.append((name, tokens[hi - 1].end + 1, _supplied(item.expression, _spelled(sql, tokens, equals + 1, hi, written=True), False)))
    return found


def _insert_pairs(
    node: exp.Insert, sql: str, tokens: list, passed: Optional[list[tuple[str, str]]] = None
) -> Optional[list[tuple[str, int, ColumnValue]]]:
    """``(column, end offset, value)`` for each column *node* names, or ``None`` where it names none.

    Paired by position, the names in one parenthesised run and the values in
    another.  ``INSERT ... SELECT`` pairs a column with the select list's item
    in its place: a bind or a literal there is the value, and a column there
    is copied from the row selected, which is read when the statement runs.
    ``ppEventServiceJob`` inserts ``:newStatus`` into ``JEDI_Events.status``
    that way.

    A column it cannot pair goes into *passed*, where given: the column list
    once, with its holes, where Python names a column (``({})``, a spec's
    ``columnNames()``), and the names past the values where the two lists do
    not pair -- see :func:`unread_columns`.  The list is said before the
    values are looked at, since ``INSERT INTO {}.workflows ({}) {}`` takes
    its values from Python too, and a run of holes is one ``{}``: the reading
    unrolls ``columnNames()``' loop once or twice, and ``({},{})`` is the
    same list as ``({})``.
    """
    target = node.this
    if not isinstance(target, exp.Schema):
        return None
    names = [column.name for column in target.expressions]
    if passed is not None and any(_NAMED_HOLE.search(name) for name in names):
        listed = "(" + ",".join(_NAMED_HOLE.sub("{}", name) for name in names) + ")"
        passed.append((_HOLE_RUN.sub("{}", listed), INSERT_COLUMN_IS_A_HOLE))
    insert = _at_depth(tokens, 0, {_T.INSERT})
    source = node.expression
    spans: list[tuple[int, int]] = []
    values: list[exp.Expression] = []
    if isinstance(source, exp.Values) and source.expressions and isinstance(source.expressions[0], exp.Tuple):
        given = _at_depth(tokens, insert + 1, {_T.VALUES}) if insert is not None else None
        if given is None or given + 1 >= len(tokens) or tokens[given + 1].token_type != _T.L_PAREN:
            return None
        closing = _closing(tokens, given + 1, len(tokens))
        spans = _items(tokens, given + 2, closing if closing > 0 else len(tokens))
        values = list(source.expressions[0].expressions)
    elif isinstance(source, exp.Select):
        chosen = _at_depth(tokens, insert + 1, {_T.SELECT}) if insert is not None else None
        stop = _at_depth(tokens, chosen + 1, {_T.FROM}) if chosen is not None else None
        if chosen is None:
            return None
        spans = _items(tokens, chosen + 1, len(tokens) if stop is None else stop)
        values = [item.unalias() for item in source.expressions]
    else:
        return None
    found: list[tuple[str, int, ColumnValue]] = []
    for name, (lo, hi), value in zip(names, spans, values, strict=False):
        if not _IDENTIFIER.fullmatch(name) or _NAMED_HOLE.search(name):
            continue
        found.append((name, tokens[hi - 1].end + 1, _supplied(value, _spelled(sql, tokens, lo, hi, written=True), True)))
    if passed is not None:
        paired = min(len(spans), len(values))
        unpaired = [name for name in names[paired:] if _IDENTIFIER.fullmatch(name) and not _NAMED_HOLE.search(name)]
        if unpaired:
            passed.append((",".join(unpaired), INSERT_VALUE_NOT_PAIRED))
    return found


def _conjunct_spans(tokens: list, lo: int, hi: int) -> list[tuple[int, int]]:
    """``tokens[lo:hi]`` cut at its own ``AND``\\ s, through parentheses that hold nothing but those.

    The tree's :func:`_conjuncts` does the same to the clause, so the two
    readings line up term for term.
    """
    while hi - lo >= 2 and tokens[lo].token_type == _T.L_PAREN and _closing(tokens, lo, hi) == hi - 1:
        if not _cuts(tokens, lo + 1, hi - 1, _T.AND) or _cuts(tokens, lo + 1, hi - 1, _T.OR):
            break
        lo, hi = lo + 1, hi - 1
    # ``AND`` binds tighter: ``(A) OR (B) AND c AND d`` is one term, an ``OR``
    # whose second branch is ``B AND c AND d`` -- ``get_workers_to_synchronize``
    # spells its time limits that way.
    if _cuts(tokens, lo, hi, _T.OR):
        return [(lo, hi)]
    cuts = _cuts(tokens, lo, hi, _T.AND)
    if not cuts:
        return [(lo, hi)]
    bounds = [lo - 1, *cuts, hi]
    return [span for a, b in zip(bounds, bounds[1:], strict=False) for span in _conjunct_spans(tokens, a + 1, b)]


def _guards(
    node: exp.Update, sql: str, tokens: list, written: set[str]
) -> Optional[
    tuple[
        dict[str, str],
        dict[str, str],
        dict[str, list[tuple[int, int]]],
        list[tuple[int, str, tuple[int, ...]]],
        dict[str, str],
    ]
]:
    """``(preconditions, keys, spans, key_holes, key_lists)`` of an ``UPDATE``, off its tree and in its own spelling.

    A term counts only where every row has to meet it -- a conjunct of the
    ``WHERE`` -- and only its own columns do: the text reading took a term
    inside a subquery for one on the row (``queue_name=:queueName`` of a
    ``SET workqueue_id=(SELECT ...)``), and the branch after an ``OR`` for the
    whole of the test.  A conjunct naming a column the statement writes is a
    precondition, kept whole -- ``(lockedBy IS NULL OR lockedTime<:t)`` is the
    lock a compare-and-set takes, and neither half of it is.  ``None`` where
    the two readings of the clause disagree on its terms.  *spans* says where
    in *sql* each precondition's conjuncts are.  A term naming its column and
    its bind by holes, ``{}=:cr_{}``, is in *key_holes* (:attr:`SqlWrite.key_holes`),
    a loop's passes' or not: what the holes are says whether it names the row.
    A term naming its rows by an ``IN`` list a helper builds, ``datasetID IN (:dataset__in)``, is in *key_lists*.
    """
    where = node.args.get("where")
    preconditions: dict[str, str] = {}
    keys: dict[str, str] = {}
    spans_of: dict[str, list[tuple[int, int]]] = {}
    key_holes: list[tuple[int, str, tuple[int, ...]]] = []
    key_lists: dict[str, str] = {}
    if where is None:
        return preconditions, keys, spans_of, key_holes, key_lists
    at = _at_depth(tokens, (_at_depth(tokens, 0, {_T.SET}) or 0) + 1, {_T.WHERE})
    if at is None:
        return None
    lo, hi = _clause(tokens, at)
    spans = _conjunct_spans(tokens, lo, hi)
    terms = list(_conjuncts(where.this))
    if len(spans) != len(terms):
        return None
    lowered = {column.lower(): column for column in written}
    testing: dict[str, list[str]] = {}
    repeated = _source(sql)[1]
    for (a, b), term in zip(spans, terms, strict=True):
        held = _key_hole(term)
        if held is not None:
            key_holes.append(held)
            continue
        # A term only a loop's passes append is on no row the loop gave no pass: required of none.
        if any(lo <= tokens[a].start and tokens[b - 1].end < hi for lo, hi in repeated):
            continue
        spelled = re.sub(r"\s+", " ", _spelled(sql, tokens, a, b)).strip()
        own = [column for column in term.find_all(exp.Column) if column.find_ancestor(exp.Select) is None]
        hits = list(dict.fromkeys(lowered[column.name.lower()] for column in own if column.name.lower() in lowered))
        # Every test of the column, joined: ``(ramCount IS NOT NULL AND
        # ramCount>:ramCount)`` is two conjuncts on one column, and keeping
        # the first dropped the one that says which rows are moved.
        for owner in hits:
            testing.setdefault(owner, []).append(spelled)
            spans_of.setdefault(owner, []).append((tokens[a].start, tokens[b - 1].end + 1))
        if (
            not hits
            and isinstance(term, exp.EQ)
            and isinstance(term.this, exp.Column)
            and isinstance(term.expression, exp.Placeholder)
            and not _NAMED_HOLE.search(term.expression.name)
        ):
            keys.setdefault(term.this.name, f":{term.expression.name}")
        elif (
            not hits
            and isinstance(term, exp.In)
            and isinstance(term.this, exp.Column)
            and len(term.expressions) == 1
            and isinstance(term.expressions[0], exp.Placeholder)
            and term.expressions[0].name.endswith(IN_LIST_SUFFIX)
        ):
            key_lists.setdefault(term.this.name, f":{term.expressions[0].name}")
    preconditions.update((owner, " AND ".join(dict.fromkeys(said))) for owner, said in testing.items())
    return preconditions, keys, spans_of, key_holes, key_lists


def _key_hole(term: exp.Expression) -> Optional[tuple[int, str, tuple[int, ...]]]:
    """``(column hole, bind, bind holes)`` of a term ``{}=:cr_{}`` (:attr:`SqlWrite.key_holes`); ``None`` for any other."""
    if not (isinstance(term, exp.EQ) and _is_hole(term.this) and isinstance(term.expression, exp.Placeholder)):
        return None
    column = _HOLE_NUMBER.fullmatch(term.this.name)
    numbers = tuple(int(number) for number in _HOLE_NUMBER.findall(term.expression.name))
    if column is None or not numbers:
        return None
    return int(column.group(1)), ":" + _NAMED_HOLE.sub("{}", term.expression.name), numbers


#: Why a column a statement writes is not in what :func:`writes` returns.  Each
#: is said as a stop on the statement (``sql.Execution.unread_columns``): a
#: column passed over in silence reads as one the statement does not write, and
#: ``updateJobStatus`` -- the pilot's every update -- wrote ``jobStatus`` alone
#: in the map while its ``SET`` takes a column for each key of ``param``.
#:
#: ``SET {}=:{}`` -- ``updateJobStatus``'s ``param``, ``harvesterIsAlive``'s
#: ``data``: the column is named from a key at run time.
SET_COLUMN_IS_A_HOLE = "the SET names its column through an interpolation"
#: ``SET {}`` -- a spec's ``bindUpdateChangesExpression()``: the columns are
#: the object's changed attributes, which Python decides.
SET_ITEM_IS_A_HOLE = "the SET takes whole assignments through an interpolation"
#: ``SET (endtime, endstatus) = (SELECT ...)`` -- no hole, but not ``column =
#: value`` either.
SET_ITEM_NOT_READ = "a SET item is not column = value"
#: ``INSERT INTO jobsArchived4 ({}) VALUES({})`` -- a spec's ``columnNames()``,
#: every column of the class in Python's order.
INSERT_COLUMN_IS_A_HOLE = "the INSERT names its columns through an interpolation"
#: Columns the ``VALUES`` or ``SELECT`` list has no item for in their place.
INSERT_VALUE_NOT_PAIRED = "the INSERT's values do not pair with its columns"
#: What is left: the statement writes a table, by its own target, and no column
#: of it was read and no item said why -- an ``INSERT`` naming no columns
#: (``INSERT INTO Metrics VALUES (...)``), one the text reading could not take.
#: Judged per text here; a run keeps it only where no other build of the same
#: statement said why (``sql._executions_read``): a ``SET {}``'s build in which
#: no attribute changed writes nothing the others do not.
COLUMNS_NOT_READ = "the statement writes a table and no column of it is read"
#: Holes side by side in a column list, with or without a comma between.
_HOLE_RUN = re.compile(r"\{\}(?:,?\{\})+")


def writes(sql: str) -> list[SqlWrite]:
    """Return the write statements in *sql*, as far as they can be read.

    Off the tree, with each value spelled as *sql* spells it.  The text
    reading this replaced cut a ``SET`` at every comma (``NVL(b, 0)`` came
    apart), read no ``INSERT ... SELECT`` and no ``INSERT`` whose values
    nested parentheses, and took a subquery's terms for the row's.  A
    statement that does not parse, or whose two readings disagree on its
    parts, is read off its text as before -- see :func:`_writes_in_text`.

    An ``UPDATE`` whose ``SET`` gives no column it can read -- ``SET {}``, a
    spec's ``bindUpdateChangesExpression()`` -- is returned with no columns
    and its ``WHERE``'s keys: the row it writes is read whether or not what
    it writes there is (:data:`SET_ITEM_IS_A_HOLE` says the columns are not).
    """
    read = _statement(sql)
    if read is None:
        return _writes_in_text(plain(sql)) if parsed(sql).tree is None else []
    node, tokens = read
    table = _named_table(node.this.this if isinstance(node.this, exp.Schema) else node.this, set())
    if isinstance(node, exp.Update):
        column_holes: list[tuple[int, str, tuple[int, ...]]] = []
        pairs = _update_pairs(node, sql, tokens, holes=column_holes)
        if pairs is None:
            return _writes_in_text(plain(sql))
        columns = {name: value for name, _end, value in pairs}
        if table is None:
            return []
        guarded = _guards(node, sql, tokens, set(columns))
        if guarded is None:
            return _writes_in_text(plain(sql)) if columns else []
        preconditions, keys, spans, key_holes, key_lists = guarded
        return [
            SqlWrite(
                kind="update",
                table=table,
                columns=columns,
                preconditions=preconditions,
                keys=keys,
                key_holes=key_holes,
                key_lists=key_lists,
                column_holes=column_holes,
                precondition_spans={
                    owner: [(_at_plain(sql, a), _at_plain(sql, b, end=True)) for a, b in found]
                    for owner, found in spans.items()
                },
            )
        ]
    pairs = _insert_pairs(node, sql, tokens)
    if pairs is None:
        return _writes_in_text(plain(sql))
    columns = {name: value for name, _end, value in pairs}
    return [SqlWrite(kind="insert", table=table, columns=columns)] if columns and table is not None else []


def unread_columns(sql: str) -> list[tuple[str, str]]:
    """``(what, why)`` for each column *sql* writes that :func:`writes` does not return, each once.

    Two readings, so that a column cannot drop out in silence.  The items the
    pairing passes over are said where it passes them (:func:`_update_pairs`,
    :func:`_insert_pairs`).  Then the tables the statement writes by its own
    target (:func:`creates`, :func:`updates` -- the reading the entities are
    made from) are held against the tables :func:`writes` returns columns of:
    one with none, where no item said why, is :data:`COLUMNS_NOT_READ`, *what*
    the table.  A ``DELETE`` writes no column; a ``MERGE`` is neither reading's.
    """
    found: dict[tuple[str, str], None] = {}
    read = _statement(sql)
    if read is not None:
        node, tokens = read
        passed: list[tuple[str, str]] = []
        if isinstance(node, exp.Update):
            _update_pairs(node, sql, tokens, passed)
        else:
            _insert_pairs(node, sql, tokens, passed)
        found.update(dict.fromkeys(passed))
    if not found:
        written = {write.table.lower() for write in writes(sql) if write.columns}
        for table in [*creates(sql), *updates(sql)]:
            if table.lower() not in written:
                found[(table, COLUMNS_NOT_READ)] = None
    return list(found)


def supplied_at(
    statement: str, kind: str, column: str, supplied: ColumnValue
) -> Optional[int]:
    """Where *statement* writes *supplied* into *column*, as an end offset.

    ``None`` when this statement does not write that value into that column.
    The offset is into *statement* as given -- the tokens are its own -- so a
    caller holding an assembled variant maps it to the fragment that wrote it.

    **The reading is chosen by kind rather than tried in turn.**  An UPSERT
    builds its ``INSERT`` and its ``UPDATE`` through one local -- seven
    functions here do -- and both spell the same column with the same value, so
    a reading that accepted either would answer for the ``INSERT`` when asked
    about the ``UPDATE``.  That is the same mis-answer this whole path exists
    to remove, arriving from the other side.
    """
    read = _statement(statement)
    if read is None:
        return _supplied_at_in_text(plain(statement), kind, column, supplied) if parsed(statement).tree is None else None
    node, tokens = read
    if kind == "insert":
        pairs = _insert_pairs(node, statement, tokens) if isinstance(node, exp.Insert) else []
    else:
        pairs = _update_pairs(node, statement, tokens) if isinstance(node, exp.Update) else []
    if pairs is None:
        return _supplied_at_in_text(plain(statement), kind, column, supplied)
    for name, end, value in pairs:
        if name == column and value == supplied:
            return _at_plain(statement, end, end=True)
    return None


#: ``INSERT INTO t``, ``UPDATE t``, ``FROM t`` -- where a statement names a
#: table, for :func:`tables_named` to fall back on.
_TABLE_REFERENCE = re.compile(r"\b(?:INSERT\s+INTO|UPDATE|FROM)\s+([\w.{}]+)", re.IGNORECASE)


def tables_named(sql: str) -> list[str]:
    """Every table *sql* names, whatever it does with it, in the order it names them.

    For what wants the tables and not the verbs: the class an ``INSERT``'s
    f-string declares for its table, and the tables a proxy method's step is
    said to read.  ``{}`` where the name is interpolated -- the caller decides
    whether a table nobody can name is worth saying.  Read off the text where
    the statement, or the fragment of one, does not parse.
    """
    read = _tables_tree(sql)
    if read is None:
        return list(dict.fromkeys(_table_of(name) for name in _TABLE_REFERENCE.findall(plain(sql))))
    tree, defined = read
    # By where the text names them: a tree puts a ``WITH`` clause after the
    # query that uses it.
    placed = sorted(tree.find_all(exp.Table), key=lambda table: table.this.meta.get("start", 0))
    found = (_named_table(table, defined) for table in placed)
    return list(dict.fromkeys(table for table in found if table is not None))


_BIND_KEY = re.compile(r":\w+")
_QUOTED = re.compile(r"'([^']*)'")


#: Where the ``WHERE`` clause ends.  Everything after is about the rows that
#: did come back, not about which ones do.
_CLAUSE_END = re.compile(
    r"\b(?:GROUP\s+BY|ORDER\s+BY|HAVING|FOR\s+UPDATE|UNION|MINUS|INTERSECT|FETCH\s+FIRST)\b", re.IGNORECASE
)


def where_clauses(sql: str) -> list[str]:
    """The text of each ``WHERE``'s clause: from it to whatever ends the clause, in the order written.

    Asked by the readers of holes and keys -- whether the placeholders a
    statement filters on are readable, and which keys it is asked by.  Every
    ``WHERE``, not the first: ``checkJobStatus`` asks ``jobsArchived4`` by
    ``:timeLimit`` in the branch after its ``UNION``, ``copyArchive`` limits
    rows with ``rownum<:nRows`` in the ``WHERE`` around a derived table, and a
    ``MINUS`` branch has keys of its own.  A clause ends at a set operator, an
    ``ORDER BY`` or the like at its own depth, or at the parenthesis closing
    the subquery it is in; a subquery inside it is part of it, and is read
    again as a clause of its own.  What the clause requires of a row is read
    off the tree, by :func:`conditions`.  A repeated part (:data:`REPEAT_OPEN`)
    is read as its passes and spelled ``{}``.
    """
    found: list[str] = []
    for opened in _WHERE.finditer(sql):
        rest = sql[opened.end() :]
        depth = 0
        end = len(rest)
        for index, char in enumerate(rest):
            if char == "(":
                depth += 1
            elif char == ")":
                if depth == 0:
                    end = index
                    break
                depth -= 1
            elif depth == 0 and _CLAUSE_END.match(rest, index):
                end = index
                break
        found.append(plain(rest[:end]))
    return found


def own_clause_terms(term: str) -> list[str]:
    """The terms *term* holds of the clause it is in, each ending where it does.

    A ``WHERE`` split into terms at any depth runs its subquery's last term
    on past the subquery: ``getEventRanges`` read ``attemptNr>:minAttemptNr
    ORDER BY jediTaskID,PandaID,fileID ) WHERE rownum<={}`` as one term, and
    named the derived table's ``rownum`` hole as ``attemptNr``'s.  What ends
    a clause in :func:`where_clauses` ends the term, and so does the next
    ``WHERE``; what follows is another clause, read as its own.

    A parenthesis the term did not open ends the term too, and what follows
    it is a term of the level it closes into: ``do_preassign``'s ``dl.rse IN
    ({}) ) {} {} {}`` is the subquery's last term, then three conditions of
    the outer ``WHERE`` interpolated whole.  Where the parenthesis closes a
    group -- ``cloud IN ({}))`` -- nothing follows it.
    """
    found: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(term):
        if char == "(":
            depth += 1
        elif char == ")":
            if depth:
                depth -= 1
                continue
            found.append(term[start:index])
            start = index + 1
        elif depth == 0 and (_CLAUSE_END.match(term, index) or _WHERE.match(term, index)):
            found.append(term[start:index])
            start = len(term)
            break
    found.append(term[start:])
    return [one for one in found if one.strip()]


#: A hole the reading left where the source interpolates -- ``{}``,
#: ``{0}``.  Read as a name, which is what most of them are: the schema, a
#: table, a column.
_HOLE = re.compile(r"\{\d*\}")
_HOLE_NAME = "__h"
_NAMED_HOLE = re.compile(rf"{_HOLE_NAME}\d+")
_HOLE_NUMBER = re.compile(rf"{_HOLE_NAME}(\d+)")

#: Where a loop of unknown length repeats a part, the statement holds a ``{}``.
#: A reader handed ``sql.Execution.reading`` gets one pass of each text the
#: loop appends there instead, between these two characters, and reads the
#: statement with them: ``jediTaskID=:jediTaskID {}`` glued a whole term to
#: the hole, ``jediTaskID=:jediTaskID AND {}=:{}`` is two.  What it says, it
#: says of the statement: a stretch made only of what is between the marks is
#: spelled ``{}``, and an offset is one into the statement's own text.
REPEAT_OPEN, REPEAT_CLOSE = "\ue000", "\ue001"
_MARKS = REPEAT_OPEN + REPEAT_CLOSE
_REPEATED = re.compile(f"{REPEAT_OPEN}[^{_MARKS}]*{REPEAT_CLOSE}")


def plain(sql: str) -> str:
    """*sql* as the statement holds it: each repeated part (:data:`REPEAT_OPEN`) a ``{}``.

    A part a clause's end cut in two (:func:`where_clauses`) is a ``{}`` for
    what of it the clause holds.
    """
    if REPEAT_OPEN not in sql and REPEAT_CLOSE not in sql:
        return sql
    text = _REPEATED.sub("{}", sql)
    closing = text.find(REPEAT_CLOSE)
    if closing >= 0:
        text = "{}" + text[closing + 1 :]
    opening = text.find(REPEAT_OPEN)
    if opening >= 0:
        text = text[:opening] + "{}"
    return text


def unmarked(sql: str) -> str:
    """*sql* with one pass of each text a loop repeats in place and the marks gone: what a stop names."""
    return sql.replace(REPEAT_OPEN, "").replace(REPEAT_CLOSE, "")


@functools.lru_cache(maxsize=None)
def _source(sql: str) -> tuple[str, tuple[tuple[int, int], ...]]:
    """*sql* without its marks, and where each repeated part is in what is left."""
    if REPEAT_OPEN not in sql:
        return sql, ()
    parts: list[str] = []
    regions: list[tuple[int, int]] = []
    length = at = 0
    for match in _REPEATED.finditer(sql):
        parts.append(sql[at : match.start()])
        length += match.start() - at
        body = match.group(0)[1:-1]
        regions.append((length, length + len(body)))
        parts.append(body)
        length += len(body)
        at = match.end()
    parts.append(sql[at:])
    return unmarked("".join(parts)), tuple(regions)


def _spell(text: str, regions: tuple, start: int, end: int) -> str:
    """``text[start:end]`` with what of it lies in a repeated part spelled ``{}``."""
    out: list[str] = []
    at = start
    for a, b in regions:
        # A part every pass of which was its end is empty here, and a ``{}`` in the statement.
        inside = a < end and b > start if a < b else start < a < end
        if not inside:
            continue
        out.append(text[at : max(a, start)])
        out.append("{}")
        at = max(at, min(b, end))
    out.append(text[at:end])
    return "".join(out)


def _at_plain(sql: str, offset: int, end: bool = False) -> int:
    """An offset into :func:`_source`'s text of *sql* as one into :func:`plain`'s.  One inside a repeated part is
    the ``{}``'s start, or with *end* its end."""
    shift = 0
    for a, b in _source(sql)[1]:
        if offset <= a:
            break
        if offset < b or (end and offset == b):
            return a - shift + (2 if end else 0)
        shift += (b - a) - 2
    return offset - shift


def _after_hole(sql: str, end: int) -> str:
    """The character after a hole ending at *end*, past the marks: a word glued to it is spaced off."""
    return sql[end : end + 4].lstrip(_MARKS)[:1]


def _plain_at(read: "Parsed", sql: str, offset: int, end: bool = False) -> int:
    """An offset into ``read.text`` as one into :func:`plain`'s text of *sql*.

    The parsed text names each hole (:func:`_named_holes`) and drops one read
    as empty; each goes back to the hole it was, and a repeated part to its
    ``{}`` (:func:`_at_plain`).  One inside a name is the hole's start, or
    with *end* its end.
    """
    source = _source(sql)[0]
    kept = {int(number) for number in _HOLE_NUMBER.findall(read.text)}
    shift = 0
    for number, hole in enumerate(_HOLE.finditer(source), 1):
        length = (len(f"{_HOLE_NAME}{number}") if number in kept else 0) + _after_hole(source, hole.end()).isalpha()
        at = hole.start() + shift
        if offset <= at:
            break
        if offset < at + length:
            return _at_plain(sql, hole.end() if end else hole.start(), end=end)
        shift += length - (hole.end() - hole.start())
    return _at_plain(sql, offset - shift, end=end)


def _plain_span(read: "Parsed", sql: str, start: int, end: int) -> tuple[int, int]:
    """``read.text[start:end]`` as a stretch of :func:`plain`'s text of *sql* (:func:`_plain_at`)."""
    return _plain_at(read, sql, start), _plain_at(read, sql, end, end=True)

#: sqlglot's word for a part the grammar requires and the text lacks.
_REQUIRED = "Required keyword"

#: Oracle's pseudo-columns -- a row's number, its depth in a tree, its address.
#: Compared with a value they limit or walk the rows; no row holds them.
_PSEUDO_NAMES = frozenset({"rownum", "level", "rowid"})

NOT_PARSED = "the statement does not parse"
NO_SCHEMA = "no schema was given to say which of the statement's tables holds the column"
OWNER_UNDECIDED = "the schema does not decide which of the statement's tables holds the column"
JSON_PATH = "a path into a JSON column, which this build does not read"
UNKNOWN_QUALIFIER = "the qualifier names no table of the statement"

#: Where a compared column sits.  ``row``: the statement's own rows -- its
#: ``WHERE``, a ``WITH`` clause, a derived table, a ``UNION`` branch.
#: ``consult``: a subquery the statement asks *about* rows through --
#: ``EXISTS (SELECT 1 ... WHERE jobStatus=:st_merging)`` checks that a merging
#: job exists and returns none of them.  ``negated``: such a subquery under an
#: odd number of ``NOT``, where the value is one the rows must *not* have
#: company in, and reading it as asked for is reading it backwards.
#: ``picks``: the subquery an ``UPDATE`` or ``DELETE`` chooses its own rows by
#: (:func:`_picks_own_rows`) -- neither a question about other rows nor the
#: rows a reader is handed.
ROW, CONSULT, NEGATED, PICKS = "row", "consult", "negated", "picks"

_SCHEMA: Optional[dict[str, frozenset[str]]] = None
#: ``{table: its primary key's columns}``, both lowered -- :func:`read_keys`.
_KEYS: Optional[dict[str, frozenset[str]]] = None
#: ``{table: its DATE / TIMESTAMP columns}``, both lowered -- :func:`read_dated`.
_DATED: Optional[dict[str, frozenset[str]]] = None


class Parsed(NamedTuple):
    """A statement as a tree, or why it is not one."""

    tree: Optional[exp.Expression]
    failure: str
    #: The text around where the parser stopped, holes spelled ``{}`` again.
    near: str = ""
    #: The text the tree was parsed from: holes named, and any read as empty
    #: gone.  The tree's positions are offsets into this.
    text: str = ""
    #: Where each repeated part (:data:`REPEAT_OPEN`) is in :attr:`text`, and
    #: the numbers of the named holes inside one.
    regions: tuple[tuple[int, int], ...] = ()
    repeated: frozenset[int] = frozenset()


class Term(NamedTuple):
    """One column a statement compares with a value."""

    #: The table the column belongs to, schema stripped; ``{}`` for a table
    #: name the reading could not render.
    table: str
    column: str
    #: Whose rows the term picks: ``read``, ``updated`` or ``deleted``.  An
    #: ``INSERT ... SELECT`` picks rows to read.
    verb: str
    #: :data:`ROW`, :data:`CONSULT`, :data:`NEGATED` or :data:`PICKS`.
    scope: str
    #: ``<>``, ``!=``, ``NOT IN``: rows without the value.
    excluding: bool
    #: The bind keys and the quoted values it is compared with, as written.
    binds: tuple[str, ...]
    literals: tuple[str, ...]
    #: Compared with something whose value the statement does not say -- a
    #: hole, or a subquery.
    undecided: bool
    #: Whether every row the term's clause lets through fails what the term
    #: rules out (:func:`_decides`).  ``processingType=:pt OR prodSourceLabel
    #: IN (...)`` names both columns and rules out neither: a row passes on
    #: the other one.
    decided: bool


class Reading(NamedTuple):
    """What a statement compares, and where reading it stopped."""

    terms: tuple[Term, ...]
    #: ``<expression> -- <reason>``, the spelling ``ReadSiteNode.unresolved`` uses.
    unresolved: tuple[str, ...]
    parsed: bool


@contextlib.contextmanager
def with_schema(
    columns: Optional[dict[str, frozenset[str]]],
    keys: Optional[dict[str, frozenset[str]]] = None,
    dated: Optional[dict[str, frozenset[str]]] = None,
) -> Iterator[None]:
    """Decide unqualified columns by *columns*, ``{table: columns}`` lowered, inside, and read *keys* and *dated*.

    A statement joining two tables may name a column without saying whose it
    is -- valid only because one of them has it, and which one is not in the
    text.  ``read_schema`` reads the answer from panda-database, and
    ``read_keys`` each table's key, which says whether a grouped statement
    returns one row per row of a table (:func:`aggregated`); ``read_dated``
    its dated columns, which a key holds to time a version of a row, not to
    name it (:func:`table_identity`).
    """
    global _SCHEMA, _KEYS, _DATED
    before, _SCHEMA = _SCHEMA, columns
    keys_before, _KEYS = _KEYS, keys
    dated_before, _DATED = _DATED, dated
    terms.cache_clear()
    conditions.cache_clear()
    conjoined.cache_clear()
    refusals.cache_clear()
    outright.cache_clear()
    condition_twins.cache_clear()
    condition_spans.cache_clear()
    _readings.cache_clear()
    term_spans.cache_clear()
    _terms_read.cache_clear()
    select_items.cache_clear()
    equalities.cache_clear()
    try:
        yield
    finally:
        _SCHEMA = before
        _KEYS = keys_before
        _DATED = dated_before
        terms.cache_clear()
        conditions.cache_clear()
        conjoined.cache_clear()
        refusals.cache_clear()
        outright.cache_clear()
        condition_twins.cache_clear()
        condition_spans.cache_clear()
        _readings.cache_clear()
        term_spans.cache_clear()
        _terms_read.cache_clear()
        select_items.cache_clear()
        equalities.cache_clear()


_CREATE_TABLE = re.compile(r'CREATE\s+(?:GLOBAL\s+TEMPORARY\s+)?TABLE\s+"(\w+)"\."(\w+)"\s*\(', re.IGNORECASE)
_COLUMN_DEFINITION = re.compile(r'^\s*"(\w+)"\s+[A-Za-z]')
_COLUMN_TYPE = re.compile(r'^\s*"(\w+)"\s+([A-Za-z]\w*)')


def read_schema(directory: Path) -> dict[str, frozenset[str]]:
    """``{table: columns}``, both lowered, from the ``CREATE TABLE`` statements under *directory*.

    panda-database's ``schema/oracle`` is one DDL file per schema owner.  The
    body of each table is read by paren depth rather than up to a closing
    pattern: a column's type has parentheses of its own -- ``NUMBER(11,0)`` --
    and a pattern looking for the ``)`` that ends the body ran one table into
    the next, which lost ``JOBSARCHIVED4`` and four others.
    """
    found: dict[str, set[str]] = {}
    for path in sorted(Path(directory).glob("*.sql")):
        text = path.read_text()
        for match in _CREATE_TABLE.finditer(text):
            columns = found.setdefault(match.group(2).lower(), set())
            for part in _top_level_parts(text, match.end()):
                column = _COLUMN_DEFINITION.match(part)
                if column is not None:
                    columns.add(column.group(1).lower())
    return {table: frozenset(columns) for table, columns in found.items()}


_PRIMARY_KEY = re.compile(
    r'ALTER\s+TABLE\s+"(\w+)"\."(\w+)"\s+ADD\s+(?:CONSTRAINT\s+"?\w+"?\s+)?PRIMARY\s+KEY\s*\(([^)]*)\)', re.IGNORECASE
)
_INLINE_PRIMARY_KEY = re.compile(r'^\s*(?:CONSTRAINT\s+"?\w+"?\s+)?PRIMARY\s+KEY\s*\(([^)]*)\)', re.IGNORECASE)
_COLUMN_PRIMARY_KEY = re.compile(r"\bPRIMARY\s+KEY\b(?!\s*\()", re.IGNORECASE)


def table_columns(table: str) -> frozenset[str]:
    """*table*'s columns, lowered, as :func:`with_schema` was given them; empty outside it."""
    return (_SCHEMA or {}).get(table.lower(), frozenset())


def table_key(table: str) -> frozenset[str]:
    """*table*'s primary key's columns, lowered, as :func:`with_schema` was given them; empty outside it."""
    return (_KEYS or {}).get(table.lower(), frozenset())


def table_identity(table: str) -> frozenset[str]:
    """*table*'s key without its dated columns: what names a row rather than a version of it --
    ``jobsArchived4``'s key is ``(PANDAID, MODIFICATIONTIME)``, and an archived job is the job its ``PandaID``
    names, whenever it was archived.  Empty outside :func:`with_schema`."""
    return table_key(table) - (_DATED or {}).get(table.lower(), frozenset())


def read_dated(directory: Path) -> dict[str, frozenset[str]]:
    """``{table: its DATE / TIMESTAMP columns}``, both lowered, from the DDL :func:`read_schema` reads."""
    found: dict[str, set[str]] = {}
    for path in sorted(Path(directory).glob("*.sql")):
        text = path.read_text()
        for match in _CREATE_TABLE.finditer(text):
            dated = found.setdefault(match.group(2).lower(), set())
            for part in _top_level_parts(text, match.end()):
                typed = _COLUMN_TYPE.match(part)
                if typed is not None and typed.group(2).upper().startswith(("DATE", "TIMESTAMP")):
                    dated.add(typed.group(1).lower())
    return {table: frozenset(columns) for table, columns in found.items()}


_PARTITION_BY = re.compile(r'PARTITION\s+BY\s+(RANGE|LIST|HASH)\s*\(([^)]*)\)', re.IGNORECASE)


def read_partitions(directory: Path) -> dict[str, tuple[str, tuple[str, ...]]]:
    """``{table: (kind, its partitioning columns)}``, lowered, from the DDL :func:`read_schema` reads: the
    ``PARTITION BY RANGE ("MODIFICATIONTIME")`` after a table's body, before the next table -- ``jobsArchived4``
    and ATLAS_PANDAARCH's ``jobsArchived`` by ``modificationtime``, the JEDI tables by ``jeditaskid``.  A table
    partitioned by nothing is absent."""
    found: dict[str, tuple[str, tuple[str, ...]]] = {}
    for path in sorted(Path(directory).glob("*.sql")):
        text = path.read_text()
        tables = list(_CREATE_TABLE.finditer(text))
        for number, match in enumerate(tables):
            span = text[match.end() : tables[number + 1].start() if number + 1 < len(tables) else len(text)]
            partitioned = _PARTITION_BY.search(span)
            if partitioned is not None:
                columns = tuple(one.strip().strip('"').lower() for one in partitioned.group(2).split(","))
                found[match.group(2).lower()] = (partitioned.group(1).lower(), columns)
    return found


def read_declaring(directory: Path) -> dict[str, tuple[str, ...]]:
    """``{table: the schemas whose DDL creates it}``, the table lowered: ``jobsarchived`` is ATLAS_PANDAARCH's,
    ``jobsarchived4`` ATLAS_PANDA's."""
    found: dict[str, set[str]] = {}
    for path in sorted(Path(directory).glob("*.sql")):
        for match in _CREATE_TABLE.finditer(path.read_text()):
            found.setdefault(match.group(2).lower(), set()).add(match.group(1))
    return {table: tuple(sorted(schemas)) for table, schemas in found.items()}


def read_keys(directory: Path) -> dict[str, frozenset[str]]:
    """``{table: its primary key's columns}``, both lowered, from the DDL :func:`read_schema` reads.

    panda-database states a key three ways: ``ALTER TABLE "ATLAS_PANDA"."JEDI_TASKS" ADD CONSTRAINT
    "JEDI_TASKS_PK" PRIMARY KEY ("JEDITASKID")`` after the table, a ``PRIMARY KEY (...)`` part of its body,
    or a column's own ``PRIMARY KEY`` (``ERROR_DESCRIPTIONS``' ``"ID" NUMBER GENERATED ... PRIMARY KEY``);
    the constraint named, quoted or not, or unnamed (``GLOBAL_SHARES``, ``SECRETS``, ``TASK_EVALUATION``).
    A unique index that is no constraint is not read.
    """
    found: dict[str, frozenset[str]] = {}

    def columns(listed: str) -> frozenset[str]:
        return frozenset(one.strip().strip('"').lower() for one in listed.split(","))

    for path in sorted(Path(directory).glob("*.sql")):
        text = path.read_text()
        for match in _CREATE_TABLE.finditer(text):
            for part in _top_level_parts(text, match.end()):
                inline = _INLINE_PRIMARY_KEY.match(part)
                column = _COLUMN_DEFINITION.match(part)
                if inline is not None:
                    found[match.group(2).lower()] = columns(inline.group(1))
                elif column is not None and _COLUMN_PRIMARY_KEY.search(part):
                    found[match.group(2).lower()] = frozenset({column.group(1).lower()})
        for match in _PRIMARY_KEY.finditer(text):
            found[match.group(2).lower()] = columns(match.group(3))
    return found


def _top_level_parts(text: str, start: int) -> list[str]:
    """The comma-separated parts of the parenthesised body opening before *start*."""
    parts: list[str] = []
    depth, index, begun = 1, start, start
    while index < len(text) and depth:
        char = text[index]
        if char == "'":
            closing = text.find("'", index + 1)
            index = len(text) if closing < 0 else closing
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 1:
            parts.append(text[begun:index])
            begun = index + 1
        index += 1
    parts.append(text[begun : index - 1])
    return parts


def _named_holes(sql: str) -> str:
    """*sql* with each hole a name, ``__h1``, ``__h2`` ..., spaced off a word it was glued to."""
    count = itertools.count(1)

    def name(match: re.Match) -> str:
        after = _after_hole(sql, match.end())
        # ``SET {}WHERE`` -- the fragment after the hole began without a space.
        return f"{_HOLE_NAME}{next(count)}" + (" " if after.isalpha() else "")

    return _HOLE.sub(name, sql)


def _is_hole(node: exp.Expression) -> bool:
    return isinstance(node, exp.Column) and not node.table and node.name.startswith(_HOLE_NAME)


def _holes_stand_for_what_is_missing(tree: exp.Expression) -> bool:
    """Whether every node missing a required part has a hole as a direct part.

    ``CASE {} ELSE {} END`` -- ``getJobs`` orders by a ``CASE`` whose ``WHEN``
    arms Python joins from a list -- parses into a ``CASE`` with no arms and
    the hole where they go.  ``... AND status IN (:s) AND`` parses into an
    ``AND`` with nothing on its right and no hole beside it: that statement is
    cut short, and taking it would hide what cut it.
    """
    unfinished = [
        node
        for node in tree.walk()
        if any(required and node.args.get(key) in (None, []) for key, required in node.arg_types.items())
    ]
    return bool(unfinished) and all(
        any(
            _is_hole(value)
            for values in node.args.values()
            for value in (values if isinstance(values, list) else [values])
            if isinstance(value, exp.Expression)
        )
        for node in unfinished
    )


def _one_of_each(tree: exp.Expression) -> exp.Expression:
    """*tree* with an ``IN`` list's repeated items said once: ``x IN (a, a)`` asks what ``x IN (a)`` does.

    A list a loop of unknown length builds is run once and twice
    (:mod:`sqleval`), ``site IN (:site_{})`` and ``site IN (:site_{},:site_{})``,
    each hole a bind the loop names: one requirement, which two spellings made
    a disjunction of the two lengths.
    """
    for node in tree.find_all(exp.In):
        items = node.expressions
        kept: dict[str, exp.Expression] = {}
        for item in items:
            kept.setdefault(_NAMED_HOLE.sub("{}", item.sql()), item)
        if len(kept) < len(items):
            node.set("expressions", list(kept.values()))
    return tree


@functools.lru_cache(maxsize=None)
def parsed(sql: str) -> Parsed:
    """*sql* as a tree, its holes read as names.

    Two holes are not names, and each is let through on one condition.  A hole
    standing for a part the grammar requires leaves a node with that part
    missing, and the tree is kept when the hole is a direct part of every such
    node -- see :func:`_holes_stand_for_what_is_missing`.  A hole the parser
    stops at is read as empty and the statement parsed again:
    ``do_preassign``'s ``) {} {} {} AND EXISTS`` is three ``.format`` fields
    holding either nothing or ``AND ...``.

    Nothing else is forgiven.  A statement that does not parse is not read
    some other way; it is reported, see :data:`NOT_PARSED`.

    A repeated part between marks (:data:`REPEAT_OPEN`) is parsed as the
    passes it holds; :attr:`Parsed.regions` says where they are.
    """
    marked = _named_holes(sql)

    def failed(failure: str, near: str) -> Parsed:
        return Parsed(None, failure, " ".join(_NAMED_HOLE.sub("{}", unmarked(near)).split()))

    for _attempt in range(len(_NAMED_HOLE.findall(marked)) + 1):
        text, regions = _source(marked)
        repeated = frozenset(
            int(number) for part in _REPEATED.finditer(marked) for number in _HOLE_NUMBER.findall(part.group(0))
        )
        try:
            return Parsed(_one_of_each(sqlglot.parse_one(text, read="oracle")), "", "", text, regions, repeated)
        except ParseError as error:
            said = [one.get("description") or "" for one in error.errors]
            if said and all(_REQUIRED in one for one in said):
                tree = sqlglot.parse_one(text, read="oracle", error_level=ErrorLevel.IGNORE)
                if tree is not None and _holes_stand_for_what_is_missing(tree):
                    return Parsed(_one_of_each(tree), "", "", text, regions, repeated)
                return failed(said[0], " ".join(text.split()[-8:]))
            first = error.errors[0] if error.errors else {}
            hole = _NAMED_HOLE.match(first.get("highlight") or "")
            if hole is None:
                before = " ".join((first.get("start_context") or "").split()[-4:])
                near = f"{before} {first.get('highlight') or ''}".strip()
                return failed(said[0] if said else str(error), near or " ".join(text.split()[:8]))
            marked = re.sub(rf"\b{hole.group(0)}\b", "", marked, count=1)
        except SqlglotError as error:
            return failed(str(error).splitlines()[0], " ".join(text.split()[:8]))
    return failed("every hole was read as empty and it still does not parse", " ".join(unmarked(marked).split()[-8:]))


def _table_name(table: exp.Table) -> str:
    return "{}" if table.name.startswith(_HOLE_NAME) else table.name


class _Root(NamedTuple):
    """An ``UPDATE``'s or ``DELETE``'s own rows: sqlglot builds no scope for them."""

    expression: exp.Expression
    sources: dict
    parent: None = None
    scope_type: None = None


def _kind(scope) -> str:
    """:data:`ROW`, or the kind of subquery *scope* is inside, by the ``NOT``\\ s above it."""
    negations, inside, picks = 0, False, False
    while scope is not None:
        if scope.scope_type == ScopeType.SUBQUERY:
            inside = True
            node = scope.expression.parent
            while node is not None and not isinstance(node, (exp.Where, exp.Select, exp.Update, exp.Delete)):
                if isinstance(node, exp.Not):
                    negations += 1
                picks = picks or _picks_own_rows(node)
                node = node.parent
        scope = scope.parent
    if not inside:
        return ROW
    if negations % 2:
        return NEGATED
    return PICKS if picks else CONSULT


def _picks_own_rows(node: exp.Expression) -> bool:
    """Whether *node* is the ``key IN (SELECT ...)`` an ``UPDATE`` or ``DELETE`` picks its own rows by.

    ``getEventRanges`` reserves events with ``UPDATE JEDI_Events SET ... WHERE
    (jediTaskID,PandaID,...) IN (SELECT jediTaskID,PandaID,... FROM (SELECT
    ... FROM JEDI_Events WHERE status=:eventStatus ...) WHERE rownum<=N)``:
    the subquery reads the very rows the statement then changes, to take the
    first ``N``.  It asks nothing about other rows, which is what a consult
    does.  The key is the statement's own columns and the subquery's rows
    come from the same table, through any derived tables but no join.
    """
    if not isinstance(node, exp.In) or node.args.get("query") is None:
        return False
    statement = node.find_ancestor(exp.Select, exp.Update, exp.Delete)
    if not isinstance(statement, (exp.Update, exp.Delete)) or not isinstance(statement.this, exp.Table):
        return False
    target = statement.this
    names = {target.name.lower(), target.alias_or_name.lower()}
    keys = list(node.this.expressions) if isinstance(node.this, exp.Tuple) else [node.this]
    if not all(isinstance(key, exp.Column) and (key.table or target.name).lower() in names for key in keys):
        return False
    query = node.args["query"]
    select = query.this if isinstance(query, exp.Subquery) else query
    while isinstance(select, exp.Select) and not select.args.get("joins"):
        source = select.args.get("from_")
        rows = source.this if source is not None else None
        if isinstance(rows, exp.Subquery):
            select = rows.this
            continue
        return isinstance(rows, exp.Table) and rows.name.lower() == target.name.lower()
    return False


def _compared(column: exp.Column) -> Optional[tuple[bool, list[exp.Expression], bool]]:
    """``(excluding, values, undecided)`` when *column* is compared with a value.

    ``None`` for a column compared with another column -- a join says how two
    tables meet, not which rows were asked for -- and for anything but ``=``,
    ``<>``/``!=`` and ``IN``, the comparisons a value can be read off.
    """
    parent = column.parent
    if isinstance(parent, exp.In) and parent.this is column:
        if parent.args.get("query") is not None:
            return False, [], True
        values = list(parent.expressions)
        return False, values, any(_is_hole(value) for value in values)
    if isinstance(parent, (exp.EQ, exp.NEQ)):
        other = parent.expression if parent.this is column else parent.this
        if isinstance(other, exp.Column) and not _is_hole(other):
            return None
        return isinstance(parent, exp.NEQ), [other], _is_hole(other)
    return None


def _negations(node: exp.Expression) -> Optional[int]:
    """The ``NOT``\\ s between *node* and the clause it is a term of.

    ``None`` when one covers an ``AND`` or ``OR`` above the comparison:
    ``NOT (a=:x AND b=:y)`` admits rows with ``a=:x``, so it neither asks for
    the value nor excludes it.
    """
    count, compound = 0, False
    walk = node.parent
    while walk is not None and not isinstance(walk, (exp.Where, exp.Join, exp.Select)):
        if isinstance(walk, (exp.And, exp.Or)):
            compound = True
        elif isinstance(walk, exp.Not):
            if compound:
                return None
            count += 1
        walk = walk.parent
    return count


def _chain(scope, root) -> list:
    found = []
    while scope is not None:
        found.append(scope)
        scope = scope.parent
    if root is not None and root not in found:
        found.append(root)
    return found


def _projected(name: str, source) -> Optional[tuple[exp.Column, object]]:
    """The column a derived table or ``WITH`` clause hands out as *name*, and its scope."""
    for projection in source.expression.selects:
        if projection.alias_or_name.lower() == name and isinstance(projection.unalias(), exp.Column):
            return projection.unalias(), source
    return None


def _owner(column: exp.Column, scope, root, depth: int = 0) -> tuple[tuple[str, ...], str]:
    """The table *column* belongs to, as ``((table,), "")``, or the candidates and why not."""
    name = column.name.lower()
    qualifier = (column.table or "").lower()
    chain = _chain(scope, root)
    if qualifier:
        for one in chain:
            for key, source in one.sources.items():
                if key.lower() != qualifier:
                    continue
                if isinstance(source, exp.Table):
                    return (_table_name(source),), ""
                projected = _projected(name, source)
                if projected is not None and depth < 8:
                    return _owner(projected[0], projected[1], None, depth + 1)
                return (), OWNER_UNDECIDED
        if column.args.get("db") is not None:
            return (), JSON_PATH
        return (), UNKNOWN_QUALIFIER
    for one in chain:
        tables = [source for source in one.sources.values() if isinstance(source, exp.Table)]
        derived = [source for source in one.sources.values() if not isinstance(source, exp.Table)]
        if one is chain[0] and len(tables) == 1 and not derived:
            return (_table_name(tables[0]),), ""
        if _SCHEMA is None:
            if tables or derived:
                return tuple(sorted({_table_name(t) for t in tables})), NO_SCHEMA
            continue
        having = [t for t in tables if name in _SCHEMA.get(t.name.lower(), ())]
        handing = [source for source in derived if _projected(name, source) is not None]
        unknown = [t for t in tables if t.name.lower() not in _SCHEMA]
        if len(having) + len(handing) == 1 and not unknown:
            if having:
                return (_table_name(having[0]),), ""
            projected = _projected(name, handing[0])
            if depth < 8:
                return _owner(projected[0], projected[1], None, depth + 1)
        if having or handing or unknown:
            return tuple(sorted({_table_name(t) for t in tables})), OWNER_UNDECIDED
        # No table here has it: a correlated reference to a scope outside.
    return (), OWNER_UNDECIDED


def _scopes(tree: exp.Expression) -> tuple[list[tuple[object, str, str]], Optional[_Root]]:
    """Every scope of *tree* with the verb whose rows it picks and its kind, and the ``UPDATE``/``DELETE`` root."""
    root: Optional[_Root] = None
    found: list[tuple[object, str, str]] = []
    if isinstance(tree, (exp.Update, exp.Delete)):
        target = tree.this
        sources = {target.alias_or_name: target} if isinstance(target, exp.Table) else {}
        root = _Root(tree, sources)
        found.append((root, "updated" if isinstance(tree, exp.Update) else "deleted", ROW))
    scopes = list(traverse_scope(tree))
    found.extend((scope, "read", _kind(scope)) for scope in scopes)
    if isinstance(tree, exp.Insert) and not scopes:
        # ``INSERT ... VALUES (:a, (SELECT datasetID FROM ... WHERE ...))``:
        # sqlglot builds no scope below a ``VALUES`` list.  Each subquery
        # there looks rows up for a value, which is a consult.
        for select in tree.find_all(exp.Select):
            if select.find_ancestor(exp.Select) is None:
                found.extend(
                    (scope, "read", CONSULT if _kind(scope) == ROW else _kind(scope))
                    for scope in traverse_scope(select)
                )
    return found, root


def _own_columns(scope) -> list[exp.Column]:
    if isinstance(scope, _Root):
        where = scope.expression.args.get("where")
        return [c for c in (where.find_all(exp.Column) if where is not None else []) if c.find_ancestor(exp.Select) is None]
    # A scope's columns include those of the subqueries in its ``WHERE``; each
    # of those is read in its own scope.
    return [c for c in scope.columns if c.find_ancestor(exp.Select) is scope.expression]


def _filtering(column: exp.Column) -> Optional[exp.Expression]:
    """The ``WHERE`` or inner join ``ON`` *column* is a term of, or ``None``."""
    holder = column.find_ancestor(exp.Where, exp.Join, exp.Select, exp.Update, exp.Delete)
    if isinstance(holder, exp.Where):
        return holder
    if isinstance(holder, exp.Join) and not holder.side and column.find_ancestor(exp.Where) is None:
        # ``JOIN ... ON a.id=b.id AND b.status=:s`` drops the rows an inner
        # join cannot match; an outer join keeps them.
        return holder.args.get("on")
    return None


@functools.lru_cache(maxsize=None)
def terms(sql: str) -> Reading:
    """The columns *sql* compares with a value, each with its table and whose rows it picks.

    Read off the tree, and only because reading the text could not say three
    things: which table an aliased column belongs to (``tabD.status`` was filed
    against every table the statement reads), which one an unqualified column
    belongs to when the statement reads several, and whether a term is the
    statement's own or a subquery's (``NOT EXISTS (... status=:s)`` was filed
    as asking for ``s``).  The values are read as the text readers read them --
    bind keys and quoted literals, ``=`` and ``IN`` asking, ``<>``, ``!=`` and
    ``NOT IN`` excluding.  A number is not read.
    """
    return _terms_read(sql)[0]


@functools.lru_cache(maxsize=None)
def term_spans(sql: str) -> tuple[tuple[int, int], ...]:
    """Where each of :func:`terms`' terms is in *sql*'s statement, in the same order, as offsets into
    :func:`plain`'s text: the term of its ``WHERE`` the column is compared in -- ``jobStatus IN ('activated',
    'sent')``, whatever fragments of the text its column and its values fall in -- or the column alone where it
    is in none."""
    return _terms_read(sql)[1]


#: What one item of a select list is (:class:`SelectItem`): a column of a table, every column (``*``), a hole the
#: code fills with columns (``{JobSpec.columnNames()}``), an expression -- or an item after a hole, at a position the
#: hole's columns move.
ITEM_COLUMN, ITEM_STAR, ITEM_HOLE, ITEM_EXPRESSION, ITEM_AFTER_HOLE = "column", "star", "hole", "expression", "after hole"


class SelectItem(NamedTuple):
    """One item of a statement's select list, at its position in each row the statement returns."""

    kind: str
    #: The table it is a column of, schema stripped and lowered; ``""`` where none is read.
    table: str = ""
    #: Its column, lowered; ``""`` but for :data:`ITEM_COLUMN`.
    column: str = ""


@functools.lru_cache(maxsize=None)
def select_items(sql: str) -> Optional[tuple[SelectItem, ...]]:
    """The items of the select list of *sql*'s first arm, in order -- what each row it returns holds where; ``None``
    where the statement is not read as a tree or returns no ``SELECT``'s rows.

    A column's table is read as a term's is (:func:`_owner`): ``tabT.jediTaskID`` is ``jedi_tasks``'s.  A hole
    stands for the columns the code fills it with, so it is one item -- a name holding one too (``{}{}``,
    ``tabT.{}``) -- and the items after it are at positions it moves.  A bare ``*`` over one derived table and no
    join (``SELECT * FROM (SELECT PandaID, ... ) WHERE rownum<=:n``) returns that table's rows: its items.
    """
    tree = parsed(sql).tree
    if tree is None:
        return None
    arms = _arms(tree)
    if not arms:
        return None
    select = arms[0]
    while (
        len(select.selects) == 1
        and isinstance(select.selects[0], exp.Star)
        and not select.args.get("joins")
        and isinstance(source := select.args.get("from_"), exp.From)
        and isinstance(source.this, exp.Subquery)
        and isinstance(source.this.this, exp.Select)
    ):
        select = source.this.this
    scopes, root = _scopes(tree)
    scope = next((one for one, _verb, _kind in scopes if one.expression is select), None)
    items: list[SelectItem] = []
    for projection in select.selects:
        column = projection.unalias()
        if items and items[-1].kind in (ITEM_HOLE, ITEM_AFTER_HOLE):
            items.append(SelectItem(ITEM_AFTER_HOLE))
        elif isinstance(column, exp.Star) or (isinstance(column, exp.Column) and isinstance(column.this, exp.Star)):
            items.append(SelectItem(ITEM_STAR))
        elif isinstance(column, exp.Column) and scope is not None:
            owned, why = _owner(column, scope, root)
            table = owned[0].lower() if owned and not why else ""
            if _NAMED_HOLE.search(column.name):
                items.append(SelectItem(ITEM_HOLE, table))
            elif table:
                items.append(SelectItem(ITEM_COLUMN, table, column.name.lower()))
            else:
                items.append(SelectItem(ITEM_EXPRESSION))
        else:
            items.append(SelectItem(ITEM_EXPRESSION))
    return tuple(items)


def is_hole(name: str) -> bool:
    """Whether *name* holds a hole the code fills (``__h1``): no table or column the text names."""
    return bool(_NAMED_HOLE.search(name))


@functools.lru_cache(maxsize=None)
def union_places(sql: str) -> tuple[tuple[str, str], ...]:
    """Each pair of distinct tables two arms of one of *sql*'s ``UNION``\\ s read in one place, lowered and sorted:
    the sources of the same alias, or the one source of each arm -- ``SELECT ... FROM jobsArchived4 UNION SELECT
    ... FROM ATLAS_PANDAARCH.jobsArchived``.  An arm joining two tables pairs each with the other arm's source of
    its alias (``filesTable4 f, jobsActive4 j`` against ``filesTable4 f, jobsDefined4 j``: the jobs tables), not
    with every table of the other arm.  A ``WITH`` name and a hole are no table; ``()`` where *sql* is not read as
    a tree."""
    tree = parsed(sql).tree
    if tree is None:
        return ()
    named = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    found: set[tuple[str, str]] = set()
    for union in tree.find_all(exp.Union):
        arms = [_arm_sources(arm) for arm in _arms(union)]
        for number, left in enumerate(arms):
            for right in arms[number + 1 :]:
                if len(left) == 1 and len(right) == 1:
                    pairs = [(left[0][1], right[0][1])]
                else:
                    pairs = [(a, b) for alias, a in left for other, b in right if alias == other]
                found.update(
                    (min(a, b), max(a, b))
                    for a, b in pairs
                    if a != b and not ({a, b} & named) and not any(is_hole(t) for t in (a, b))
                )
    return tuple(sorted(found))


def _arm_sources(select: exp.Select) -> list[tuple[str, str]]:
    """``(alias, table)`` of the sources one arm reads its rows from, lowered: its ``FROM`` and joins, a derived
    table's own arms' under the derived table's alias."""
    head = select.args.get("from_")
    sources = ([head.this] if head is not None else []) + [join.this for join in select.args.get("joins") or []]
    found: list[tuple[str, str]] = []
    for source in sources:
        if isinstance(source, exp.Table):
            found.append((source.alias_or_name.lower(), source.name.lower()))
        elif isinstance(source, exp.Subquery):
            for arm in _arms(source.this):
                found.extend((source.alias_or_name.lower() or alias, table) for alias, table in _arm_sources(arm))
    return found


class Equality(NamedTuple):
    """A column of one source of a statement compared equal with a column of another: how two tables meet."""

    #: The first side's table, schema stripped and lowered, its column lowered, and the source it is read from as the
    #: statement names it -- its alias, or the table where it has none -- lowered.
    table: str
    column: str
    source: str
    #: The other side, the same way; the two sides sorted.
    other_table: str
    other_column: str
    other_source: str


@functools.lru_cache(maxsize=None)
def equalities(sql: str) -> tuple[Equality, ...]:
    """Each ``a.x = b.y`` of *sql* between two of its sources -- two tables, or two aliases of one -- in a ``WHERE`` or
    an ``ON``, inner or outer, a subquery's among them; each column's table read as a term's is (:func:`_owner`).
    Sorted.  A composite key is one statement's equalities sharing one pair of sources and their tables (as
    :class:`~bamboo.codemap.models.JoinPair` says).

    Not a ``SET a = b``, nor a ``MERGE``'s, nor a side naming a hole, a pseudo column or a table not read, nor a column
    compared with one of its own source (``a.x = a.y``); ``x IN (SELECT y ...)`` is no equality.  ``()`` where the
    statement is not read as a tree: a text :func:`joins` reads only by its words gives none.
    """
    tree = parsed(sql).tree
    if tree is None:
        return ()
    scopes, root = _scopes(tree)
    found: set[Equality] = set()
    for node in tree.find_all(exp.EQ):
        left, right = node.this, node.expression
        if not (isinstance(left, exp.Column) and isinstance(right, exp.Column)):
            continue
        if isinstance(node.parent, exp.Update) or node.find_ancestor(exp.Merge) is not None:
            continue
        holder = node.find_ancestor(exp.Select, exp.Update, exp.Delete)
        if isinstance(holder, (exp.Update, exp.Delete)):
            scope = root
        else:
            scope = next((one for one, _verb, _kind in scopes if one.expression is holder), None)
        if scope is None:
            continue
        sides = []
        for column in (left, right):
            name = column.name.lower()
            if _NAMED_HOLE.search(column.name) or name in _PSEUDO_NAMES:
                break
            owned, why = _owner(column, scope, root)
            if why or not owned:
                break
            sides.append((owned[0].lower(), name, (column.table or owned[0]).lower()))
        else:
            if sides[0][2] != sides[1][2]:
                first, second = sorted(sides)
                found.add(Equality(*first, *second))
    return tuple(sorted(found))


@functools.lru_cache(maxsize=None)
def _terms_read(sql: str) -> tuple[Reading, tuple[tuple[int, int], ...]]:
    """:func:`terms` and :func:`term_spans`."""
    read = parsed(sql)
    tree = read.tree
    if tree is None:
        return Reading((), (f"{read.near} -- {NOT_PARSED}",), False), ()
    tokens = _tokens(read.text)
    # The terms each ``WHERE`` ``AND``\ s and ``OR``\ s together: a subquery's are inside its own.
    pieces: list[tuple[int, int]] = []
    for where in tree.find_all(exp.Where):
        clause = _where_span(where, tokens)
        if clause is not None and clause[0] < clause[1]:
            pieces.extend(_leaves(_split(tokens, *clause), tokens))
    found: list[Term] = []
    spans: list[tuple[int, int]] = []
    unresolved: set[str] = set()
    scopes, root = _scopes(tree)
    for scope, verb, kind in scopes:
        for column in _own_columns(scope):
            term = _term(column, scope, root, verb, kind, read.regions)
            if isinstance(term, Term):
                found.append(term)
                meta = column.this.meta if isinstance(column.this, exp.Identifier) else {}
                start = meta.get("start", 0)
                around = [(a, b) for a, b in pieces if a <= start < b]
                span = min(around, key=lambda one: one[1] - one[0]) if around else (start, meta.get("end", start) + 1)
                spans.append(_plain_span(read, sql, *span))
            elif term is not None:
                unresolved.add(term)
    return Reading(tuple(found), tuple(sorted(unresolved)), True), tuple(spans)


def _leaves(node, tokens: list) -> list[tuple[int, int]]:
    """The stretches of the parsed text a :func:`_split` tree's leaves are."""
    if node[0] == "leaf":
        _leaf, a, b = node
        return [(tokens[a].start, tokens[b - 1].end + 1)] if a < b else []
    return [span for child in node[1] for span in _leaves(child, tokens)]


def _term(column: exp.Column, scope, root, verb: str, kind: str, regions: tuple = ()) -> Optional[Term | str]:
    """The term *column* is compared in, why its table is not known, or ``None``.

    A column in a repeated part (*regions*, :attr:`Parsed.regions`) is asked by
    on the passes that append it, and decides nothing: a run whose loop gave
    no such pass does not ask it.
    """
    # ``tabT.{0}=:{0}`` -- a column Python names at run time, which the
    # reading already reports.  Read as a column called ``__h4``, it
    # came out as a key no table has.
    if _NAMED_HOLE.search(column.name) or column.name.lower() in _PSEUDO_NAMES:
        return None
    holder = _filtering(column)
    if holder is None:
        return None
    compared = _compared(column)
    if compared is None:
        return None
    excluding, values, undecided = compared
    # ``status=:{}`` -- a bind whose name is interpolated has no value
    # to look up, so it is a value the statement does not decide.
    placeholders = [value for value in values if isinstance(value, exp.Placeholder)]
    binds = tuple(f":{value.name}" for value in placeholders if not _NAMED_HOLE.search(value.name))
    undecided = undecided or len(binds) < len(placeholders)
    negations = _negations(column.parent)
    if negations is None:
        return None
    owners, why = _owner(column, scope, root)
    if why:
        spelled = column.sql(dialect="oracle")
        among = f" ({', '.join(owners)})" if owners else ""
        return f"{spelled}{among} -- {why}"
    excluding = excluding != bool(negations % 2)
    return Term(
        table=owners[0],
        column=column.name,
        verb=verb,
        scope=kind,
        excluding=excluding,
        binds=binds,
        literals=tuple(
            value.this for value in values if isinstance(value, exp.Literal) and value.is_string
        ),
        undecided=undecided,
        decided=_decides(column, holder, scope, root, excluding, undecided, values) and not _in_region(column, regions),
    )


def _in_region(column: exp.Column, regions: tuple) -> bool:
    """Whether *column*'s name is written in one of *regions* of the parsed text."""
    start = column.this.meta.get("start") if isinstance(column.this, exp.Identifier) else None
    return start is not None and any(a <= start < b for a, b in regions)


def _decides(
    column: exp.Column, holder: exp.Expression, scope, root, excluding: bool, undecided: bool, values: list
) -> bool:
    """Whether every row *holder* lets through fails what *column*'s term rules out.

    For a term asking for values, that is a row holding none of the values
    the clause asks this column for; for one excluding them, a row holding
    them.  Read on the conjunct of *holder* the term is in: one branch of an
    ``AND`` deciding is enough, an ``OR`` needs every branch to.
    ``(modificationTime<:t AND jobStatus=:a) OR (stateChangeTime<:t AND
    jobStatus=:b)`` decides ``jobStatus`` -- no row with another status
    passes either branch -- and ``processingType=:pt OR prodSourceLabel IN
    (...)`` decides neither column.  A test of the column in every branch is
    not enough: ``status<>:a OR status<>:b`` passes the rows holding ``a``
    on its second branch.

    Asking needs every value said: ``status=:{}`` asks for something the
    statement does not name, so a row holding none of the named values can
    still be the one asked for.
    """
    if not excluding and undecided:
        return False
    top = holder.this if isinstance(holder, exp.Where) else holder
    conjunct = next(
        (one for one in _conjuncts(top) if any(node is column for node in one.find_all(exp.Column))), None
    )
    if conjunct is None:
        return False
    owner = _owner(column, scope, root)[0]
    ruled = {value.sql() for value in values}
    quoted = all(isinstance(value, exp.Literal) and value.is_string for value in values)

    def same(other: Optional[exp.Expression]) -> bool:
        if not isinstance(other, exp.Column) or other.name.lower() != column.name.lower():
            return False
        return other is column or _owner(other, scope, root) == (owner, "")

    def rules_out(node: exp.Expression) -> bool:
        negated = False
        while isinstance(node, (exp.Paren, exp.Not)):
            negated ^= isinstance(node, exp.Not)
            node = node.this
        if isinstance(node, (exp.And, exp.Or)):
            if negated:
                return False  # ``NOT (a OR b)`` -- read only as the term it is
            sides = (rules_out(node.this), rules_out(node.expression))
            return any(sides) if isinstance(node, exp.And) else all(sides)
        if isinstance(node, exp.Is) and isinstance(node.expression, exp.Null):
            # ``col IS NULL`` holds no value at all, which is every value excluded.
            return excluding and not negated and same(node.this)
        if not isinstance(node, (exp.EQ, exp.NEQ, exp.In)):
            return False
        sides = [node.this] if isinstance(node, exp.In) else [node.this, node.expression]
        tested = next((side for side in sides if same(side)), None)
        compared = _compared(tested) if tested is not None else None
        if compared is None:
            return False
        excludes, said, open_ = compared
        excludes ^= negated
        if not excluding:
            return not excludes and not open_ and bool(said)
        if excludes:
            # ``NOT IN (SELECT ...)`` names no value to find in another
            # branch: only its own test passes them over.
            return tested is column if undecided else ruled <= {value.sql() for value in said}
        # ``col IN ('a','b')`` holds no row with ``'c'``.
        return (
            quoted
            and not open_
            and all(isinstance(value, exp.Literal) and value.is_string for value in said)
            and not ruled & {value.sql() for value in said}
        )

    return rules_out(conjunct)


class Branch(NamedTuple):
    """One alternative of a condition: the terms a row can satisfy together."""

    #: The terms, joined by ``AND`` as the statement writes them.
    terms: tuple[str, ...]
    #: Every bind key named anywhere in ``terms``, in the order they appear.
    binds: tuple[str, ...]


class Condition(NamedTuple):
    """One thing a statement requires of a row, beside the values it selects on.

    A conjunction of branches is not what this is: the *list* of conditions is
    the conjunction, and each condition is a disjunction of :class:`Branch`.
    One branch is the ordinary case -- ``modificationTime<:modificationTime``
    is required outright -- and several is an ``OR``, which has to stay one
    condition or the list stops reading as "all of these must hold".
    """

    branches: tuple[Branch, ...]

    @property
    def binds(self) -> tuple[str, ...]:
        """Every bind key this condition names, across all its branches."""
        return tuple(key for branch in self.branches for key in branch.binds)


class Side(NamedTuple):
    """What a term compares its column with, as the statement says it."""

    #: ``bind`` | ``constant`` | ``column`` | ``clock`` | ``other``.
    kind: str
    #: As the statement spells it: ``:timeLimit``, ``800``, ``CURRENT_DATE-3``.
    text: str
    #: The bind keys in it, in order.
    binds: tuple[str, ...] = ()
    #: A column's table, as :attr:`Term.table`, and its name.
    table: str = ""
    column: str = ""
    #: A constant's value as ``selects`` spells values: ``800``, ``pending``.
    constant: str = ""
    #: A clock the database reads: ``CURRENT_DATE``, ``SYSDATE``, ``CURRENT_TIMESTAMP``, ``SYSTIMESTAMP``.
    clock: str = ""
    #: Seconds added to the clock where the statement writes the amount:
    #: ``CURRENT_DATE-3`` is ``-259200``, Oracle counting a ``DATE``'s
    #: arithmetic in days.
    offset_seconds: Optional[float] = None
    #: Seconds per unit of a bind amount added to the clock:
    #: ``CURRENT_DATE-:days`` is ``-86400`` times the bind's value.
    scale_seconds: Optional[float] = None


class Test(NamedTuple):
    """One term of a condition, read as what it asks of the row's column."""

    #: ``compare`` (``< <= > >=``), ``equal`` (``=``, ``IN``), ``unequal``
    #: (``<>``, ``NOT IN``), ``null``, ``not_null``, or ``other``.
    kind: str
    #: The term as the statement writes it.
    text: str
    #: The column the term tests, on the left, as :attr:`Term.table` and its name.
    table: str = ""
    column: str = ""
    #: ``<``, ``<=``, ``>``, ``>=`` for ``compare`` (turned round where the
    #: statement puts the column on the right), ``=``/``IN``, ``<>``/``NOT IN``.
    op: str = ""
    #: The other side: one for a comparison, each listed value for ``IN``.
    sides: tuple[Side, ...] = ()


class Twin(NamedTuple):
    """A condition read as a tree a row can be evaluated against: :func:`condition_twins`.

    ``any`` holds where one of its *items* does, ``all`` where every one
    does, and ``test`` is one term.  An entry of :func:`conditions` is an
    ``any`` of its branches, each an ``all`` of its terms; a term holding an
    ``OR`` of its own nests, and so does what :func:`_either` folds across
    the branches of a ``UNION``.
    """

    node: str
    items: tuple["Twin", ...] = ()
    test: Optional[Test] = None


_TOKENIZE = sqlglot.Dialect.get_or_raise("oracle").tokenize
_T = sqlglot.TokenType
#: What ends a ``WHERE`` clause at its own depth.  ``START WITH`` comes out of
#: the tokenizer as ``BEGIN`` and ``WITH``.
_ENDS_A_CLAUSE = frozenset(
    {
        _T.GROUP_BY,
        _T.ORDER_BY,
        _T.HAVING,
        _T.UNION,
        _T.EXCEPT,
        _T.INTERSECT,
        _T.FOR,
        _T.CONNECT_BY,
        _T.BEGIN,
        _T.FETCH,
        _T.LIMIT,
        _T.OFFSET,
        _T.RETURNING,
        _T.SEMICOLON,
    }
)
_OPENS = frozenset({_T.L_PAREN, _T.CASE})
_CLOSES = frozenset({_T.R_PAREN, _T.END})
#: What a term is to the conditions list.  ``_KEEP`` is a reason a row was
#: passed over; ``_HELD`` is a column compared with one bind, which
#: ``selects``, ``excludes`` or ``keys`` already carries; ``_DROP`` is not a
#: requirement on the row at all.
_KEEP, _HELD, _DROP = "keep", "held", "drop"
_COMPARISONS = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE)


class _Token(NamedTuple):
    token_type: object
    start: int
    end: int


def _tokens(text: str) -> list[_Token]:
    """*text*'s tokens, a bind one token with its name.

    The tokenizer reads ``:limit`` as a colon and the keyword ``LIMIT``, and
    ``:end`` as one that closes a ``CASE``.
    """
    found: list[_Token] = []
    for token in _TOKENIZE(text):
        if found and found[-1].token_type == _T.COLON and token.start == found[-1].end + 1:
            found[-1] = _Token(_T.PLACEHOLDER, found[-1].start, token.end)
        else:
            found.append(_Token(token.token_type, token.start, token.end))
    return found


def _clause(tokens: list, where: int) -> tuple[int, int]:
    """The tokens of the ``WHERE`` at *where*, up to what ends it at its depth."""
    depth = 0
    for index in range(where + 1, len(tokens)):
        kind = tokens[index].token_type
        if kind == _T.L_PAREN:
            depth += 1
        elif kind == _T.R_PAREN:
            if depth == 0:
                return where + 1, index
            depth -= 1
        elif depth == 0 and kind in _ENDS_A_CLAUSE:
            return where + 1, index
    return where + 1, len(tokens)


def _closing(tokens: list, opened: int, end: int) -> int:
    depth = 0
    for index in range(opened, end):
        kind = tokens[index].token_type
        if kind == _T.L_PAREN:
            depth += 1
        elif kind == _T.R_PAREN:
            depth -= 1
            if depth == 0:
                return index
    return -1


def _cuts(tokens: list, lo: int, hi: int, keyword) -> list[int]:
    """Where *keyword* joins terms at the depth of ``tokens[lo:hi]``.

    Not inside parentheses or a ``CASE``, and not the ``AND`` of a ``BETWEEN``.
    """
    depth, between, found = 0, False, []
    for index in range(lo, hi):
        kind = tokens[index].token_type
        if kind in _OPENS:
            depth += 1
        elif kind in _CLOSES:
            depth -= 1
        elif depth == 0:
            if kind == _T.BETWEEN:
                between = True
            elif kind == _T.AND and between:
                between = False
            elif kind == keyword:
                found.append(index)
    return found


def _split(tokens: list, lo: int, hi: int):
    """``tokens[lo:hi]`` as ``("or"|"and", [child, ...])`` or ``("leaf", lo, hi)``.

    ``AND`` binds tighter than ``OR``, so the ``OR`` split happens first:
    ``(A AND B) OR (C AND D) AND E`` is two alternatives, ``E`` part of the
    second.
    """
    while hi - lo >= 2 and tokens[lo].token_type == _T.L_PAREN and _closing(tokens, lo, hi) == hi - 1:
        lo, hi = lo + 1, hi - 1
    for kind, keyword in (("or", _T.OR), ("and", _T.AND)):
        cuts = _cuts(tokens, lo, hi, keyword)
        if cuts:
            bounds = [lo - 1, *cuts, hi]
            return (kind, [_split(tokens, a + 1, b) for a, b in zip(bounds, bounds[1:], strict=False)])
    return ("leaf", lo, hi)


def _connectives(where: exp.Where) -> set[int]:
    """The ``AND``, ``OR`` and grouping parentheses of *where*'s own clause."""
    found: set[int] = set()
    stack = [where.this]
    while stack:
        node = stack.pop()
        if isinstance(node, (exp.And, exp.Or)):
            found.add(id(node))
            stack.extend((node.this, node.expression))
        elif isinstance(node, exp.Paren) and isinstance(node.this, (exp.And, exp.Or, exp.Paren)):
            found.add(id(node))
            stack.append(node.this)
    return found


def _term_holding(node: exp.Expression, where: exp.Where, joins: set[int]) -> exp.Expression:
    """The term of *where*'s clause that *node* is part of, its brackets off."""
    while node.parent is not None and node.parent is not where and id(node.parent) not in joins:
        node = node.parent
    while isinstance(node, exp.Paren):
        node = node.this
    return node


def _own(node: exp.Expression, select: Optional[exp.Expression]) -> list[exp.Column]:
    return [
        column
        for column in node.find_all(exp.Column)
        if column.find_ancestor(exp.Select) is select
        and not _NAMED_HOLE.search(column.name)
        and column.name.lower() not in _PSEUDO_NAMES
    ]


def _term_kind(term: exp.Expression, scope, root) -> str:
    """:data:`_KEEP`, :data:`_HELD` or :data:`_DROP`, read off the tree."""
    node, negations = term, 0
    while isinstance(node, (exp.Not, exp.Paren)):
        negations += isinstance(node, exp.Not)
        node = node.this
    if isinstance(node, exp.Exists) or (isinstance(node, exp.In) and node.args.get("query") is not None):
        # ``EXISTS (...)`` looks rows up and is ``consults``; ``NOT EXISTS``
        # turns the subquery's terms round, and only the term as a whole says
        # what it asks.  So does the subquery an ``UPDATE`` picks its own
        # rows by, which no other field holds.
        return _KEEP if negations % 2 or _picks_own_rows(node) else _DROP
    select = None if isinstance(scope, _Root) else scope.expression
    own = _own(node, select)
    if not own:
        return _DROP  # ``ROWNUM <= :n`` is ``limits``; ``1=1`` asks nothing
    if isinstance(node, _COMPARISONS):
        sides = (node.this, node.expression)
        if all(isinstance(side, exp.Column) and not _NAMED_HOLE.search(side.name) for side in sides):
            if (sides[0].table or "").lower() != (sides[1].table or "").lower():
                return _DROP  # a join says how two tables meet, not why a row was skipped
    if isinstance(node, (exp.EQ, exp.NEQ)):
        # ``status=:s`` and ``status<>:s``, the shape the text reading held
        # back.  ``IN`` and a quoted value are kept as that reading kept them:
        # a bind written in a kept condition is where a reader's walk starts.
        for column in own:
            if column.parent is not node or _compared(column) is None:
                continue
            other = node.expression if node.this is column else node.this
            if not isinstance(other, exp.Placeholder) or _NAMED_HOLE.search(other.name):
                continue
            if not _owner(column, scope, root)[1]:
                return _HELD
    return _KEEP


def _respelled(fragment: str, sql: str) -> str:
    """*fragment* of the parsed text with each named hole spelled as *sql* has it."""
    holes = [(match.group(0), _after_hole(sql, match.end()).isalpha()) for match in _HOLE.finditer(sql)]

    def back(match: re.Match) -> str:
        number = int(match.group(1)) - 1
        if number >= len(holes):
            return match.group(0)
        spelled, padded = holes[number]
        return spelled if padded or not match.group(2) else spelled + match.group(2)

    return re.sub(rf"{_HOLE_NAME}(\d+)( ?)", back, fragment)


def _spelled_read(read: Parsed, sql: str, start: int, end: int) -> str:
    """``read.text[start:end]`` as *sql*'s statement spells it: each hole as written, a repeated part ``{}``."""
    return _respelled(_spell(read.text, read.regions, start, end), sql)


def _repeated_only(node: exp.Expression, read: Parsed) -> bool:
    """Whether all of *node* lies in a repeated part, by what of it says where it is: the words and literals the
    parser placed, and the holes its names carry -- a bind ``:pt___h3`` is placed by neither."""
    inside = [
        any(a <= leaf.meta["start"] < b for a, b in read.regions)
        for leaf in node.find_all(exp.Identifier, exp.Literal)
        if "start" in leaf.meta
    ]
    inside += [int(number) in read.repeated for number in _HOLE_NUMBER.findall(node.sql(dialect="oracle"))]
    return bool(inside) and all(inside)


def _where_span(where: exp.Where, tokens: list) -> Optional[tuple[int, int]]:
    """The tokens of *where*'s clause: from the last ``WHERE`` before its first term to what ends it."""
    placed = [node.meta["start"] for node in where.this.find_all(exp.Identifier, exp.Literal) if "start" in node.meta]
    if not placed:
        return None
    first = min(placed)
    opened = [index for index, token in enumerate(tokens) if token.token_type == _T.WHERE and token.start < first]
    if not opened:
        return None
    return _clause(tokens, opened[-1])


def _where_tree(where: exp.Where, scope, root, read: Parsed, sql: str, tokens: list):
    """*where*'s clause as a tree of ``("term", text, kind, node, span)``, or ``None``.

    Split on the tokens, so each term keeps the statement's own spelling --
    a tree regenerated by sqlglot respells it (``endTime IS NOT NULL`` comes
    back as ``NOT endTime IS NULL``).  Each piece is then matched to the term
    of the tree it spans, which says what it is.  ``None`` when the two do not
    agree on where a term begins and ends.  *span* is where the term is in
    the statement (:func:`_plain_span`).
    """
    placed = [
        (node.meta["start"], node)
        for node in where.this.find_all(exp.Identifier, exp.Literal)
        if "start" in node.meta
    ]
    span = _where_span(where, tokens)
    if span is None:
        return None
    lo, hi = span
    if hi <= lo or any(not tokens[lo].start <= start <= tokens[hi - 1].end for start, _node in placed):
        return None
    joins = _connectives(where)
    agreed = True

    def build(node):
        nonlocal agreed
        if node[0] != "leaf":
            return (node[0], [build(child) for child in node[1]])
        _leaf, a, b = node
        if a >= b:
            agreed = False
            return ("term", "", _DROP, None, None)
        start, end = tokens[a].start, tokens[b - 1].end
        span = _plain_span(read, sql, start, end + 1)
        held = {id(_term_holding(n, where, joins)): _term_holding(n, where, joins) for s, n in placed if start <= s <= end}
        text = " ".join(_spelled_read(read, sql, start, end + 1).split())
        if "," in text:
            text = _IN_LIST.sub(_listed_once, text)
        if len(held) > 1:
            agreed = False
        # A term only a loop's passes append is on no row the loop gave no pass: required of none.
        if len(held) != 1 or any(lo <= start and end < hi for lo, hi in read.regions):
            return ("term", text, _DROP, None, span)
        node = next(iter(held.values()))
        return ("term", text, _term_kind(node, scope, root), node, span)

    tree = build(_split(tokens, lo, hi))
    return tree if agreed else None


def _prune(node, in_or: bool, held: bool = False):
    """Drop what is not an answer, and drop a disjunction that lost a branch.

    An ``OR`` with one branch gone is satisfied by that branch whatever the
    row does, so recording the rest would state a requirement the statement
    does not make -- the direction that gets a row ruled out for failing a
    test it never faced.  A term ``selects`` already carries is dropped only
    outside a disjunction: inside one it is not the same fact, since
    ``selects`` says the statement asks by the value outright.  *held* keeps
    it outside one too (:func:`conjoined`).
    """
    if node[0] == "term":
        kind = node[2]
        return node if kind == _KEEP or (kind == _HELD and (in_or or held)) else None
    kind, body = node
    children = [_prune(child, in_or or kind == "or", held) for child in body]
    if kind == "or":
        return (kind, children) if all(children) else None
    kept = [child for child in children if child]
    return (kind, kept) if kept else None


def _terms_of(node) -> tuple[str, ...]:
    """A branch's terms, with a nested disjunction kept whole as one term.

    Parenthesised, because the terms around it are joined by ``AND`` and
    ``AND`` binds tighter: ``X AND a OR b`` reads as ``(X AND a) OR b``, which
    is not what the statement said and is the looser of the two -- a row that
    fails ``X`` would be reported as still able to come back.
    """
    if node[0] == "term":
        return (node[1],)
    kind, body = node
    if kind == "and":
        return tuple(term for child in body for term in _terms_of(child))
    return (f"({' OR '.join(_render_branch(child) for child in body)})",)


def _render_branch(node) -> str:
    terms = _terms_of(node)
    return f"({' AND '.join(terms)})" if len(terms) > 1 else terms[0]


def _branch(terms: tuple[str, ...]) -> Branch:
    return Branch(terms, tuple(key for term in terms for key in _BIND_KEY.findall(term)))


def _as_condition(node) -> Condition:
    parts = node[1] if node[0] == "or" else [node]
    return Condition(tuple(_branch(_terms_of(part)) for part in parts))


def _as_terms(condition: Condition) -> tuple[str, ...]:
    """*condition* as terms of a branch: its own, or its disjunction as one."""
    if len(condition.branches) == 1:
        return condition.branches[0].terms
    rendered = [
        f"({' AND '.join(branch.terms)})" if len(branch.terms) > 1 else branch.terms[0]
        for branch in condition.branches
    ]
    return (f"({' OR '.join(rendered)})",)


_CLOCKS = {exp.CurrentDate: "CURRENT_DATE", exp.Systimestamp: "SYSTIMESTAMP"}
#: Seconds in each unit an ``INTERVAL`` names.
_INTERVAL_UNITS = {"DAY": 86400.0, "HOUR": 3600.0, "MINUTE": 60.0, "SECOND": 1.0}
#: How :class:`Test` turns a comparison round to put its column on the left.
_TURNED = {"<": ">", "<=": ">=", ">": "<", ">=": "<="}
_ORDERS = {exp.LT: "<", exp.LTE: "<=", exp.GT: ">", exp.GTE: ">="}


def _clock_name(node: exp.Expression) -> str:
    if isinstance(node, exp.CurrentTimestamp):
        return "SYSDATE" if node.args.get("sysdate") else "CURRENT_TIMESTAMP"
    return _CLOCKS.get(type(node), "")


def _days(node: exp.Expression) -> Optional[float]:
    """A constant amount of days: ``3``, ``2/24``."""
    while isinstance(node, exp.Paren):
        node = node.this
    if isinstance(node, exp.Literal) and not node.is_string:
        try:
            return float(node.this)
        except ValueError:
            return None
    if isinstance(node, (exp.Div, exp.Mul)):
        left, right = _days(node.this), _days(node.expression)
        if left is None or right is None:
            return None
        if isinstance(node, exp.Mul):
            return left * right
        return left / right if right else None
    return None


def _clock_side(node: exp.Expression, text: str) -> Optional[Side]:
    """``CURRENT_DATE``, ``SYSDATE-1``, ``CURRENT_DATE-:days``, ``CURRENT_TIMESTAMP - INTERVAL '7' DAY``."""
    while isinstance(node, exp.Paren):
        node = node.this
    name = _clock_name(node)
    if name:
        return Side("clock", text, clock=name, offset_seconds=0.0)
    if not isinstance(node, (exp.Add, exp.Sub)):
        return None
    name = _clock_name(node.this)
    if not name:
        return None
    sign = -1.0 if isinstance(node, exp.Sub) else 1.0
    amount = node.expression
    while isinstance(amount, exp.Paren):
        amount = amount.this
    if isinstance(amount, exp.Placeholder) and not _NAMED_HOLE.search(amount.name):
        return Side("clock", text, binds=(f":{amount.name}",), clock=name, scale_seconds=sign * 86400.0)
    if isinstance(amount, exp.Interval):
        unit = (amount.args.get("unit").name if amount.args.get("unit") is not None else "").upper()
        try:
            count = float(amount.this.name)
        except (AttributeError, ValueError):
            return None
        if unit not in _INTERVAL_UNITS:
            return None
        return Side("clock", text, clock=name, offset_seconds=sign * count * _INTERVAL_UNITS[unit])
    days = _days(amount)
    if days is None:
        return None
    return Side("clock", text, clock=name, offset_seconds=sign * days * 86400.0)


def _side(node: exp.Expression, scope, root, sql: str) -> Side:
    """What *node*, the side a term compares its column with, is."""
    read = parsed(sql)
    text = "{}" if read.regions and _repeated_only(node, read) else _respelled(node.sql(dialect="oracle"), sql)
    while isinstance(node, exp.Paren):
        node = node.this
    if isinstance(node, exp.Placeholder):
        if _NAMED_HOLE.search(node.name):
            return Side("other", text)
        return Side("bind", text, binds=(f":{node.name}",))
    if isinstance(node, exp.Literal):
        return Side("constant", text, constant=node.this)
    if isinstance(node, exp.Null):
        return Side("constant", text, constant="None")
    if isinstance(node, exp.Column):
        owner = _column_owner(node, scope, root)
        if owner is None:
            return Side("other", text)
        return Side("column", text, table=owner, column=node.name)
    clock = _clock_side(node, text)
    if clock is not None:
        return clock
    return Side("other", text, binds=tuple(f":{one.name}" for one in node.find_all(exp.Placeholder)))


def _column_owner(column: exp.Expression, scope, root) -> Optional[str]:
    """The table a plain column of the clause belongs to, or ``None``."""
    if not isinstance(column, exp.Column) or _NAMED_HOLE.search(column.name):
        return None
    if column.name.lower() in _PSEUDO_NAMES:
        return None
    owners, why = _owner(column, scope, root)
    return owners[0] if owners and not why else None


def _test(node: Optional[exp.Expression], text: str, scope, root, sql: str) -> Twin:
    """The term *node* of a clause as a :class:`Test`, or the ``all`` of a ``BETWEEN``'s two."""
    other = Twin("test", test=Test("other", text))
    if node is None:
        return other
    negated = False
    while isinstance(node, (exp.Not, exp.Paren)):
        negated ^= isinstance(node, exp.Not)
        node = node.this
    if isinstance(node, exp.Is) and isinstance(node.expression, exp.Null):
        owner = _column_owner(node.this, scope, root)
        if owner is None:
            return other
        return Twin("test", test=Test("not_null" if negated else "null", text, owner, node.this.name))
    if isinstance(node, exp.In) and node.args.get("query") is None:
        owner = _column_owner(node.this, scope, root)
        if owner is None or any(_is_hole(one) for one in node.expressions):
            return other
        sides = tuple(_side(one, scope, root, sql) for one in node.expressions)
        kind, op = ("unequal", "NOT IN") if negated else ("equal", "IN")
        return Twin("test", test=Test(kind, text, owner, node.this.name, op, sides))
    if negated:
        return other
    if isinstance(node, exp.Between):
        owner = _column_owner(node.this, scope, root)
        if owner is None:
            return other
        return Twin(
            "all",
            (
                Twin("test", test=Test("compare", text, owner, node.this.name, ">=", (_side(node.args["low"], scope, root, sql),))),
                Twin("test", test=Test("compare", text, owner, node.this.name, "<=", (_side(node.args["high"], scope, root, sql),))),
            ),
        )
    if isinstance(node, (exp.EQ, exp.NEQ, *_ORDERS)):
        left, right = node.this, node.expression
        op = "=" if isinstance(node, exp.EQ) else "<>" if isinstance(node, exp.NEQ) else _ORDERS[type(node)]
        if _column_owner(left, scope, root) is None and _column_owner(right, scope, root) is not None:
            left, right, op = right, left, _TURNED.get(op, op)
        owner = _column_owner(left, scope, root)
        if owner is None:
            return other
        kind = "equal" if op == "=" else "unequal" if op == "<>" else "compare"
        return Twin("test", test=Test(kind, text, owner, left.name, op, (_side(right, scope, root, sql),)))
    return other


def _twin(node, scope, root, sql: str) -> Twin:
    """A pruned clause tree as a :class:`Twin`: ``or`` an ``any``, ``and`` an ``all``."""
    if node[0] == "term":
        return _test(node[3], node[1], scope, root, sql)
    kind, body = node
    return Twin("any" if kind == "or" else "all", tuple(_twin(child, scope, root, sql) for child in body))


def _as_twin(node, scope, root, sql: str) -> Twin:
    """The :class:`Twin` of the condition :func:`_as_condition` makes of *node*: branches, then terms."""
    parts = node[1] if node[0] == "or" else [node]
    return Twin("any", tuple(Twin("all", tuple(_twin(term, scope, root, sql) for term in _anded(part))) for part in parts))


def _anded(node) -> list:
    """The terms *node* ``AND``\\ s together, a nested ``AND`` opened up as :func:`_terms_of` opens it."""
    if node[0] == "and":
        return [term for child in node[1] for term in _anded(child)]
    return [node]


#: A condition, its :class:`Twin`, and where it is in the statement: each
#: place it is read off, as the stretches of the terms it is made of
#: (:func:`condition_spans`).
_Read = tuple[Condition, Twin, tuple[tuple[tuple[int, int], ...], ...]]


def _either(alternatives: list[list[_Read]]) -> list[_Read]:
    """What a row from any one of *alternatives* is required to satisfy, each with its :class:`Twin` and where it is.

    ``(A and B) or (A and C)`` is ``A``, required outright, and ``B or C`` as
    one condition; an alternative that requires nothing beyond the common
    part makes that disjunction free, and then it is not recorded.  A
    condition read off several branches is where each of them is: the
    statement holds it as long as it holds them all.
    """

    def where(condition: Condition, among: list[list[_Read]]) -> tuple[tuple[tuple[int, int], ...], ...]:
        return (tuple(span for one in among for read in one if read[0] == condition for at in read[2] for span in at),)

    common = [
        (condition, twin, where(condition, alternatives))
        for condition, twin, _at in alternatives[0]
        if all(condition in [read[0] for read in other] for other in alternatives[1:])
    ]
    held = [read[0] for read in common]
    remainders: list[list[_Read]] = []
    for alternative in alternatives:
        rest = [read for read in alternative if read[0] not in held]
        same = [other for other in remainders if [read[0] for read in other] == [read[0] for read in rest]]
        if same:
            same[0][:] = [(one, twin, where(one, [same[0], rest])) for one, twin, _at in same[0]]
        else:
            remainders.append(rest)
    if len(remainders) == 1:
        return common + remainders[0]
    if not all(remainders):
        return common
    folded = Condition(
        tuple(_branch(tuple(term for read in rest for term in _as_terms(read[0]))) for rest in remainders)
    )
    twin = Twin("any", tuple(Twin("all", tuple(read[1] for read in rest)) for rest in remainders))
    return common + [(folded, twin, (tuple(span for rest in remainders for read in rest for at in read[2] for span in at),))]


def _set_operands(node: exp.Expression, kind: type) -> list[exp.Expression]:
    if isinstance(node, kind) and type(node) is kind:
        return [*_set_operands(node.this, kind), *_set_operands(node.expression, kind)]
    return [node]


@functools.lru_cache(maxsize=None)
def conditions(sql: str) -> tuple[Condition, ...]:
    """Return what *sql* asks of a row beyond the values it selects on.

    Beside :func:`terms` rather than inside it, and the two answer different
    questions.  ``terms`` says *which value* a statement asks rows by -- that
    is what a subject's vocabulary is made of.  This says *what else had to
    be true*, which is the question a row that holds the value and never came
    back raises: ``copyArchive.py``'s ``holding`` query also requires
    ``modificationTime<:modificationTime``, and the map had no way to say so.

    **The list reads as a conjunction**, so a disjunction is one entry with
    its alternatives joined -- split up, ``(status=:running OR
    status=:scouting)`` asks for a row whose status is both, which no row can
    be.  The rows' own ``WHERE``\\ s are read: the statement's, each branch of
    a ``UNION`` (folded as the alternatives they are, see :func:`_either`),
    and those of the derived tables and ``WITH`` clauses the rows come from,
    which hold as well as the outer one.

    What is left out, and why:

    * **A column compared with one bind** -- ``status=:s``, ``status<>:s``
      -- where it stands alone: ``selects``, ``excludes`` or ``keys`` has it,
      and one fact is one row.  Inside a disjunction it is kept, see
      :func:`_prune`.  ``IN`` and a quoted value are kept, as the text
      reading kept them.
    * **Join terms.**  ``c1.jediTaskID=d.jediTaskID`` says how two tables
      meet, not why a row was passed over.
    * **A term that names none of the rows' columns**: ``ROWNUM <= :n`` is
      ``limits``, ``EXISTS (...)`` is ``consults``.  ``NOT EXISTS (...)`` is
      kept whole: what it asks is the subquery turned round.  So is the
      ``key IN (SELECT ...)`` an ``UPDATE`` picks its own rows by.
    * **A disjunction that lost a branch to any of those.**  It no longer
      constrains anything, and recording what is left would over-state it.
    * **A statement that does not parse**, which ``unresolved`` reports.

    What is kept and cannot be walked is still returned: an ``IS NULL`` names
    no value to go looking for, and saying the term is there is the point.
    The caller decides what to do with :attr:`Branch.binds`, which is empty
    for exactly those.
    """
    return _required(sql, held=False)


@functools.lru_cache(maxsize=None)
def conjoined(sql: str) -> tuple[Condition, ...]:
    """Every term the rows' ``WHERE``\\ s ``AND`` together, read as :func:`conditions` reads them, with a column compared with one bind kept.

    Where a value enters the statement.  ``conditions`` leaves
    ``status=:status`` out, since ``selects`` has it, and a reader asking
    which term a pair of ``selects`` is asked in was handed the fragment the
    term was appended in instead -- ``status=:status AND (maxAttempt IS NULL
    OR attemptNr<maxAttempt)``, whose ``OR`` is another term's.
    """
    return _required(sql, held=True)


def _required(sql: str, held: bool) -> tuple[Condition, ...]:
    """:func:`conditions`, and with *held* :func:`conjoined`."""
    return tuple(read[0] for read in _readings(sql, held))


def returns_rows(sql: str) -> Optional[bool]:
    """Whether *sql* is a query, whose rows a ``fetch*`` reads: ``None`` where it does not parse."""
    tree = parsed(sql).tree
    return None if tree is None else isinstance(tree, exp.Query)


@functools.lru_cache(maxsize=None)
def row_tests(sql: str) -> tuple[Test, ...]:
    """The tests every row *sql* returns holds: each term of an entry of :func:`conjoined` with one branch.

    A row a query returns met its ``WHERE``, so ``PandaID=:PandaID`` holds
    of each one, and its ``PandaID`` is what the bind was filled with.  An
    entry of two branches or more, an ``OR``, holds of no one column, and
    neither does an ``OR`` nested in a term.  A statement that is not a
    query (:func:`returns_rows`) returns no rows and has none.
    """
    if not returns_rows(sql):
        return ()

    def held(twin: Twin) -> list[Test]:
        if twin.node == "test":
            return [twin.test] if twin.test is not None else []
        if twin.node == "all" or len(twin.items) == 1:
            return [test for item in twin.items for test in held(item)]
        return []

    return tuple(test for _condition, twin, _at in _readings(sql, True) for test in held(twin))


@functools.lru_cache(maxsize=None)
def condition_twins(sql: str) -> tuple[Twin, ...]:
    """Each entry of :func:`conditions` as a tree a row can be evaluated against, in the same order.

    Read off the same clause the entry is, so the two cannot drift: an
    entry is an ``any`` of its branches and each branch an ``all`` of its
    terms, and each term says what it asks of which table's column --
    ``modificationTime<:modificationTime`` compares ``jobsActive4``'s
    ``modificationTime`` with a bind, ``creationTime>CURRENT_DATE-3`` with
    the database's clock less three days.  What does not read as one column
    tested against a value -- a ``NOT EXISTS``, an expression on both sides
    -- is ``other``, never left out: an entry false for a row is false
    because of its terms, and dropping one would make it look decided.
    """
    return tuple(read[1] for read in _readings(sql, False))


@functools.lru_cache(maxsize=None)
def condition_spans(sql: str) -> tuple[tuple[tuple[tuple[int, int], ...], ...], ...]:
    """Where each entry of :func:`conditions` is in *sql*'s statement, in the same order.

    Each place the entry is read off -- a ``WHERE`` of the statement, of a
    derived table -- as the stretches of the terms it is made of, offsets
    into :func:`plain`'s text.  The statement holds the entry where it holds
    every stretch of one place: what puts a term there is what puts that text
    there, whichever fragments of the text the term's own words fall in.
    """
    return tuple(read[2] for read in _readings(sql, False))


@functools.lru_cache(maxsize=None)
def _readings(sql: str, held: bool) -> tuple[_Read, ...]:
    """:func:`_required` with each condition's :class:`Twin` and where it is."""
    read = parsed(sql)
    if read.tree is None:
        return ()
    tokens = _tokens(read.text)
    scopes, root = _scopes(read.tree)
    scope_of = {id(scope.expression): scope for scope, _verb, _kind in scopes}
    ctes: dict[str, exp.Expression] = {}
    entered: set[int] = set()

    def required(node: Optional[exp.Expression]) -> list[_Read]:
        if node is None or id(node) in entered:
            return []
        entered.add(id(node))
        if isinstance(node, exp.Subquery):
            return required(node.this)
        if isinstance(node, exp.Union):
            return _either([required(one) for one in _set_operands(node, type(node))])
        if isinstance(node, exp.Intersect):
            return [one for operand in _set_operands(node, exp.Intersect) for one in required(operand)]
        if isinstance(node, exp.Except):
            # ``A MINUS B``: what came back met ``A``; ``B`` only took rows away.
            return required(node.this)
        if isinstance(node, exp.Insert):
            return required(node.expression) if isinstance(node.expression, (exp.Query, exp.Subquery)) else []
        if not isinstance(node, (exp.Select, exp.Update, exp.Delete)):
            return []
        for cte in (node.args.get("with_") or exp.With()).expressions:
            ctes[cte.alias_or_name.lower()] = cte.this
        found: list[_Read] = []
        where = node.args.get("where")
        scope = root if isinstance(node, (exp.Update, exp.Delete)) else scope_of.get(id(node))
        if where is not None and scope is not None:
            clause = _where_tree(where, scope, root, read, sql, tokens)
            kept = _prune(clause, False, held) if clause is not None else None
            if kept is not None:
                found.extend(
                    (_as_condition(child), _as_twin(child, scope, root, sql), (_spans_of(child),))
                    for child in (kept[1] if kept[0] == "and" else [kept])
                )
        sources = [node.args.get("from_")] + list(node.args.get("joins") or [])
        for source in (one.this for one in sources if one is not None):
            if isinstance(source, exp.Subquery):
                found.extend(required(source))
            elif isinstance(source, exp.Table) and source.name.lower() in ctes:
                found.extend(required(ctes[source.name.lower()]))
        return found

    if isinstance(read.tree, exp.Query):
        for cte in (read.tree.args.get("with_") or exp.With()).expressions:
            ctes[cte.alias_or_name.lower()] = cte.this
    found: list[_Read] = []
    for one, twin, at in required(read.tree):
        same = [index for index, read in enumerate(found) if read[0] == one]
        if same:
            # Read off two clauses -- the statement's and a derived table's: either one holding it is enough.
            found[same[0]] = (one, found[same[0]][1], found[same[0]][2] + at)
        else:
            found.append((one, twin, at))
    return tuple(found)


def _spans_of(node) -> tuple[tuple[int, int], ...]:
    """The stretches of the statement a pruned clause tree's terms are."""
    if node[0] == "term":
        return (node[4],) if node[4] is not None else ()
    return tuple(span for child in node[1] for span in _spans_of(child))


def _conjuncts(node: exp.Expression) -> Iterator[exp.Expression]:
    """The terms *node* ``AND``\\ s together, through parentheses."""
    if isinstance(node, exp.Paren):
        yield from _conjuncts(node.this)
    elif isinstance(node, exp.And):
        yield from _conjuncts(node.this)
        yield from _conjuncts(node.expression)
    else:
        yield node


def _disjuncts(node: exp.Expression) -> list[exp.Expression]:
    """The terms *node* ``OR``\\ s together, through parentheses."""
    if isinstance(node, exp.Paren):
        return _disjuncts(node.this)
    if isinstance(node, exp.Or):
        return _disjuncts(node.this) + _disjuncts(node.expression)
    return [node]


def _outright(where: exp.Where, scope, root, verb: str, kind: str, regions: tuple = ()) -> tuple[Term, ...]:
    """The terms of *where* every row it lets through meets -- :attr:`Term.decided`."""
    found: list[Term] = []
    own = {id(column) for column in _own_columns(scope)}
    for conjunct in _conjuncts(where.this):
        for column in conjunct.find_all(exp.Column):
            if id(column) not in own:
                continue
            term = _term(column, scope, root, verb, kind, regions)
            if isinstance(term, Term) and term.decided:
                found.append(term)
    return tuple(found)


@functools.lru_cache(maxsize=None)
def refusals(sql: str) -> tuple[tuple[Term, ...], ...]:
    """The exclusions every row *sql* returns satisfies, one tuple per way a row comes back.

    Narrower than the excluding :func:`terms`, which say a statement *names* a
    value to pass over.  ``getFailureCountsForTask_JEDI`` names
    ``relocationFlag<>:relThrottled`` inside one branch of an ``OR``, and its
    other branches return throttled jobs all the same; a reader told it leaves
    them out would stop looking at a query that picks them up.  The excluding
    half of :func:`outright`.
    """
    return tuple(tuple(term for term in way if term.excluding) for way in outright(sql))


@functools.lru_cache(maxsize=None)
def outright(sql: str) -> tuple[tuple[Term, ...], ...]:
    """The terms every row *sql* returns meets, one tuple per way a row comes back.

    A term counts only where every row has to meet it (:attr:`Term.decided`):
    a conjunct of the ``WHERE``, or an ``OR`` every branch of which rules out
    what the term does, of the statement or of the derived table or ``WITH``
    clause the rows come from.  A ``UNION`` returns rows from any of its
    branches, and each branch is a tuple of its own -- the caller names the
    values and decides how the branches combine, since which subject a
    table's column is, is not this module's to say.  ``()`` for a statement
    that does not parse.
    """
    read = parsed(sql)
    if read.tree is None:
        return ()
    scopes, root = _scopes(read.tree)
    scope_of = {id(scope.expression): (scope, verb, kind) for scope, verb, kind in scopes}
    ctes: dict[str, exp.Expression] = {}
    entered: set[int] = set()

    def rows(node: Optional[exp.Expression]) -> list[tuple[Term, ...]]:
        if node is None or id(node) in entered:
            return [()]
        entered.add(id(node))
        if isinstance(node, exp.Subquery):
            return rows(node.this)
        if isinstance(node, exp.Union):
            return [one for operand in _set_operands(node, type(node)) for one in rows(operand)]
        if isinstance(node, exp.Intersect):
            return [
                tuple(term for part in parts for term in part)
                for parts in itertools.product(*(rows(one) for one in _set_operands(node, exp.Intersect)))
            ]
        if isinstance(node, exp.Except):
            return rows(node.this)
        if isinstance(node, exp.Insert):
            return rows(node.expression) if isinstance(node.expression, (exp.Query, exp.Subquery)) else [()]
        if not isinstance(node, (exp.Select, exp.Update, exp.Delete)):
            return [()]
        for cte in (node.args.get("with_") or exp.With()).expressions:
            ctes[cte.alias_or_name.lower()] = cte.this
        where = node.args.get("where")
        if isinstance(node, (exp.Update, exp.Delete)):
            entry = (root, "updated" if isinstance(node, exp.Update) else "deleted", ROW)
        else:
            entry = scope_of.get(id(node))
        own: tuple[Term, ...] = ()
        if where is not None and entry is not None and entry[0] is not None:
            scope, verb, kind = entry
            own = _outright(where, scope, root, verb, kind, read.regions)
        found = [own]
        sources = [node.args.get("from_")] + list(node.args.get("joins") or [])
        for source in (one.this for one in sources if one is not None):
            if isinstance(source, exp.Subquery):
                inner = rows(source)
            elif isinstance(source, exp.Table) and source.name.lower() in ctes:
                inner = rows(ctes[source.name.lower()])
            else:
                continue
            found = [mine + theirs for mine in found for theirs in inner]
        return found

    if isinstance(read.tree, exp.Query):
        for cte in (read.tree.args.get("with_") or exp.With()).expressions:
            ctes[cte.alias_or_name.lower()] = cte.this
    return tuple(rows(read.tree))


#: What ends an ``ORDER BY`` list at its own depth.
_ENDS_AN_ORDER = frozenset(
    {_T.FOR, _T.FETCH, _T.LIMIT, _T.OFFSET, _T.UNION, _T.EXCEPT, _T.INTERSECT, _T.SEMICOLON}
)


def _rows_from(node: Optional[exp.Expression], ctes: dict[str, exp.Expression], entered: set[int]) -> Iterator[exp.Expression]:
    """Every query whose rows *node* returns: itself, a set operation's sides, its derived tables and ``WITH`` clauses."""
    if node is None or id(node) in entered:
        return
    entered.add(id(node))
    if isinstance(node, exp.Query):
        for cte in (node.args.get("with_") or exp.With()).expressions:
            ctes[cte.alias_or_name.lower()] = cte.this
    if isinstance(node, exp.Subquery):
        yield from _rows_from(node.this, ctes, entered)
        return
    if isinstance(node, exp.Insert):
        yield from _rows_from(node.expression, ctes, entered)
        return
    yield node
    if isinstance(node, exp.SetOperation):
        yield from _rows_from(node.this, ctes, entered)
        if not isinstance(node, exp.Except):
            yield from _rows_from(node.expression, ctes, entered)
        return
    if isinstance(node, exp.Select):
        sources = [node.args.get("from_")] + list(node.args.get("joins") or [])
        for source in (one.this for one in sources if one is not None):
            if isinstance(source, exp.Subquery):
                yield from _rows_from(source, ctes, entered)
            elif isinstance(source, exp.Table) and source.name.lower() in ctes:
                yield from _rows_from(ctes[source.name.lower()], ctes, entered)


def _bound(node: exp.Expression, sql: str) -> Optional[str]:
    """A row limit as written: a bind key, a number, or the hole it is spelled through."""
    read = parsed(sql)
    if read.regions and _repeated_only(node, read):
        return "{}"
    if isinstance(node, exp.Placeholder):
        return _respelled(f":{node.name}", sql)
    if isinstance(node, exp.Literal) and not node.is_string:
        return node.this
    if _is_hole(node):
        return _respelled(node.name, sql)
    return None


def _order_text(query: exp.Expression, order: exp.Order, read: Parsed, sql: str, tokens: list) -> str:
    """*order*'s list as the statement spells it."""
    span = _order_span(query, tokens)
    return " ".join(_spelled_read(read, sql, *span).split()) if span is not None else ""


def _order_span(query: exp.Expression, tokens: list) -> Optional[tuple[int, int]]:
    """Where *query*'s ``ORDER BY`` list is in the parsed text, if it has one.

    Found as the first ``ORDER BY`` after the rest of *query* that comes
    before it, not by the list's own positions: ``DBMS_RANDOM.value`` is read
    as a ``RAND()`` with none.
    """
    before = [
        node.meta["start"]
        for part in ("expressions", "from_", "joins", "where", "group", "having", "this", "expression")
        for value in ([query.args.get(part)] if not isinstance(query.args.get(part), list) else query.args.get(part))
        if isinstance(value, exp.Expression)
        for node in value.find_all(exp.Identifier, exp.Literal)
        if "start" in node.meta
    ]
    after = max(before) if before else -1
    opened = [index for index, token in enumerate(tokens) if token.token_type == _T.ORDER_BY and token.start > after][:1]
    if not opened:
        return None
    depth, last = 0, len(tokens)
    for index in range(opened[-1] + 1, len(tokens)):
        kind = tokens[index].token_type
        if kind == _T.L_PAREN:
            depth += 1
        elif kind == _T.R_PAREN:
            if depth == 0:
                last = index
                break
            depth -= 1
        elif depth == 0 and kind in _ENDS_AN_ORDER:
            last = index
            break
    if last <= opened[-1] + 1:
        return None
    return tokens[opened[-1] + 1].start, tokens[last - 1].end + 1


def ordering(sql: str) -> tuple[list[str], list[tuple[str, bool]]]:
    """``ORDER BY`` clauses and row limits: which rows come back first, and how many.

    A row that passes the ``WHERE`` can still lose on order and not come back:
    ``getJobs`` sorts by share, priority and ``PandaID`` and keeps
    ``ROWNUM <= :njobs``.  Limits are ``(bind or number, strict)`` -- ``<``
    keeps one fewer than ``<=``.

    Read off the tree, from the queries whose rows come back -- see
    :func:`_rows_from`.  The text reading took every ``ORDER BY`` and
    ``ROWNUM`` in the statement, a subquery's and an ``OVER (...)``'s too.
    A statement that does not parse has neither, and ``unresolved`` says so.
    """
    read = parsed(sql)
    if read.tree is None:
        return [], []
    tokens = _tokens(read.text)
    orders: list[str] = []
    limits: list[tuple[str, bool]] = []
    for query in _rows_from(read.tree, {}, set()):
        order = query.args.get("order")
        if isinstance(order, exp.Order):
            text = _order_text(query, order, read, sql, tokens)
            if text and text not in orders:
                orders.append(text)
        limit = query.args.get("limit")
        if isinstance(limit, (exp.Limit, exp.Fetch)):
            counted = limit.args.get("count") if isinstance(limit, exp.Fetch) else limit.expression
            value = _bound(counted, sql) if counted is not None else None
            if value is not None:
                limits.append((value, False))
        where = query.args.get("where")
        if where is None:
            continue
        # ``ROWNUM = 1`` keeps the first row, which is ``<= 1``: Oracle numbers
        # a row only once it has passed, so no other ``=`` returns anything.
        for comparison in where.find_all(exp.LT, exp.LTE, exp.GT, exp.GTE, exp.EQ):
            if comparison.find_ancestor(exp.Select, exp.Update, exp.Delete) is not query:
                continue
            left, right = comparison.this, comparison.expression
            if isinstance(left, exp.Column) and left.name.lower() == "rownum" and isinstance(comparison, (exp.LT, exp.LTE, exp.EQ)):
                value = _bound(right, sql)
            elif isinstance(right, exp.Column) and right.name.lower() == "rownum" and isinstance(comparison, (exp.GT, exp.GTE, exp.EQ)):
                value = _bound(left, sql)
            else:
                continue
            if value is not None:
                limits.append((value, isinstance(comparison, (exp.LT, exp.GT))))
    return orders, limits


def ordering_spans(sql: str) -> tuple[tuple[tuple[tuple[int, int], ...], ...], ...]:
    """Where each ``ORDER BY`` list of :func:`ordering` is in *sql*'s statement, in the same order: each place
    it is read off, as its stretch of :func:`plain`'s text (the shape :func:`condition_spans` has)."""
    read = parsed(sql)
    if read.tree is None:
        return ()
    tokens = _tokens(read.text)
    orders: list[str] = []
    places: list[list[tuple[tuple[int, int], ...]]] = []
    for query in _rows_from(read.tree, {}, set()):
        order = query.args.get("order")
        span = _order_span(query, tokens) if isinstance(order, exp.Order) else None
        text = " ".join(_spelled_read(read, sql, *span).split()) if span is not None else ""
        if not text:
            continue
        if text not in orders:
            orders.append(text)
            places.append([])
        places[orders.index(text)].append((_plain_span(read, sql, *span),))
    return tuple(tuple(one) for one in places)


class OrderItem(NamedTuple):
    """One item of an ``ORDER BY`` list."""

    #: The expression sorted on, as the list spells it, direction off.
    expression: str
    #: The columns it names, as spelled -- ``gshare`` of a ``CASE WHEN
    #: gshare=:leave0 THEN 0 ... END``; none where it is ``blanked``.
    columns: tuple[str, ...]
    descending: bool
    #: Built at run time, and spelled ``{}`` where the map reads the source:
    #: ``getJobs``' share ranking is ``(CASE {} ELSE {} END)``.  Any item a
    #: run prints in its place is that item.
    blanked: bool


_DIRECTION = re.compile(r"^(?P<expression>.*?)(?:\s+(?P<direction>ASC|DESC))?(?:\s+NULLS\s+(?:FIRST|LAST))?$", re.I | re.S)


def _spelled_table(table: exp.Table) -> str:
    """*table* as the statement names it: catalog and schema kept, alias off, a hole ``{}``."""
    parts = [table.args.get(key) for key in ("catalog", "db", "this")]
    return ".".join(
        "{}" if part.name.startswith(_HOLE_NAME) else part.sql(dialect="oracle")
        for part in parts
        if isinstance(part, exp.Identifier)
    )


def _spelled_columns(columns: Iterator[exp.Column] | list[exp.Column]) -> tuple[str, ...]:
    """*columns* as spelled -- ``tab.jobStatus`` -- once each, in the order the text has them."""
    placed = sorted(columns, key=lambda column: column.this.meta.get("start", 0) if isinstance(column.this, exp.Identifier) else 0)
    return tuple(dict.fromkeys(".".join(part for part in (column.table, column.name) if part) for column in placed))


def order_items(order: str) -> tuple[OrderItem, ...]:
    """*order* -- an ``ORDER BY`` list as :attr:`ReadSiteNode.order_by` or :func:`ordering` spells it -- item by item.

    The same reading for the map's spelling and for the text a run printed,
    so the two compare item for item: the map's ``(CASE {} ELSE {} END),
    currentpriority desc, pandaid asc`` and a ``getJobs`` debug line's
    ``ORDER BY (CASE WHEN gshare=:leave0 THEN 0 ... END), currentpriority
    desc, pandaid asc`` are three items each, the first blanked in the map.
    Split at its own commas -- not inside parentheses or a ``CASE``.
    """
    tokens = _tokens(order)
    if not tokens:
        return ()
    bounds = [-1, *_cuts(tokens, 0, len(tokens), _T.COMMA), len(tokens)]
    items: list[OrderItem] = []
    for a, b in zip(bounds, bounds[1:], strict=False):
        text = " ".join(_spelled(order, tokens, a + 1, b).split())
        match = _DIRECTION.match(text)
        expression = match.group("expression") if match else text
        blanked = "{}" in expression
        columns: tuple[str, ...] = ()
        if not blanked:
            try:
                tree = sqlglot.parse_one(expression, read="oracle")
            except SqlglotError:
                tree = None
            if tree is not None:
                columns = _spelled_columns(tree.find_all(exp.Column))
        items.append(
            OrderItem(
                expression=expression,
                columns=columns,
                descending=bool(match and (match.group("direction") or "").upper() == "DESC"),
                blanked=blanked,
            )
        )
    return tuple(items)


class PrintedTerm(NamedTuple):
    """One conjunct of a ``WHERE`` a run printed."""

    #: As the statement spells it.
    text: str
    #: The columns of the query it filters that it names, as spelled
    #: (``tab.jobStatus``); a subquery's own are not among them.
    columns: tuple[str, ...]
    #: Every bind it names, ``:name``, in order -- a subquery's too, since
    #: the printed map fills them all.
    binds: tuple[str, ...]


class PrintedQuery(NamedTuple):
    """One query whose rows a printed statement returns."""

    #: The tables its ``FROM`` and joins read, schema off; a derived table or
    #: ``WITH`` clause is not one -- it is a query of its own.
    tables: tuple[str, ...]
    #: The same tables, in the same order, as the statement spells them:
    #: schema kept, alias off -- ``ATLAS_PANDA.jobsActive4``, what a query
    #: run by hand against the database has to name.  A part the map reads
    #: as a hole is ``{}``.
    spelled: tuple[str, ...]
    #: Its ``WHERE``'s conjuncts.  Empty where it has none, or where the
    #: clause could not be cut term for term (``failure`` says which).
    where: tuple[PrintedTerm, ...]
    #: Its ``ORDER BY`` list, item by item (:func:`order_items`).
    order: tuple[OrderItem, ...]


class Printed(NamedTuple):
    """What a statement a run printed asks, read as the map reads statements."""

    #: The query whose rows come back first, then each it takes rows from:
    #: ``getJobs``' ``SELECT * FROM (SELECT ... FROM jobsActive4 tab WHERE
    #: ... ORDER BY ...) WHERE ROWNUM<=:njobs`` is the outer query, which
    #: reads no table and keeps the first rows, then the inner one, which
    #: reads ``jobsActive4`` and filters it.
    queries: tuple[PrintedQuery, ...]
    #: ``(bind or number, strict)``, as :func:`ordering` reads them.
    limits: tuple[tuple[str, bool], ...]
    #: Why the text was not read, or a part of it was not; ``""`` where it was.
    failure: str


def printed(sql: str) -> Printed:
    """Read *sql*, a statement a run printed, as the map reads the statements it stores.

    For a reader holding a log line rather than a tree: the debug record
    ``getJobs`` writes before its ``execute`` is the statement that ran,
    with the ``ORDER BY`` the run chose and the share ranking it built.  The
    bind map printed after it is the run's, not the statement's, and is
    the caller's to read; a ``/* comment */`` between them is read past.

    This is the entry kept stable for such a reader.  It is made of the
    readings the map uses -- the queries rows come back from
    (:func:`ordering`), each ``WHERE`` cut at its own ``AND``\\ s as the map
    cuts one (:func:`_conjunct_spans`), each ``ORDER BY`` item by item
    (:func:`order_items`) -- and those may change under it.
    """
    read = parsed(sql)
    if read.tree is None:
        return Printed((), (), f"{read.near} -- {NOT_PARSED}" if read.near else NOT_PARSED)
    tokens = _tokens(read.text)
    queries: list[PrintedQuery] = []
    failures: list[str] = []
    for query in _rows_from(read.tree, {}, set()):
        if not isinstance(query, (exp.Select, exp.Update, exp.Delete)):
            continue
        sources = [query.args.get("from_"), *(query.args.get("joins") or [])] if isinstance(query, exp.Select) else [query]
        named = list(
            dict.fromkeys(
                (_table_name(source.this), _spelled_table(source.this))
                for source in sources
                if source is not None and isinstance(source.this, exp.Table)
            )
        )
        tables = tuple(name for name, _spelled in named)
        where: list[PrintedTerm] = []
        clause = query.args.get("where")
        if clause is not None:
            span = _where_span(clause, tokens)
            spans = _conjunct_spans(tokens, *span) if span is not None else []
            terms = list(_conjuncts(clause.this))
            if len(spans) != len(terms):
                failures.append(f"{len(spans)} conjunct(s) spelled, {len(terms)} read, in a WHERE of {', '.join(tables) or 'a derived table'}")
            else:
                for (a, b), term in zip(spans, terms, strict=True):
                    text = " ".join(_respelled(read.text[tokens[a].start : tokens[b - 1].end + 1], sql).split())
                    own = [column for column in term.find_all(exp.Column) if column.find_ancestor(exp.Select, exp.Update, exp.Delete) is query]
                    where.append(
                        PrintedTerm(
                            text=text,
                            columns=_spelled_columns(own),
                            # A quoted value -- ``'12:30'`` -- holds no bind.
                            binds=tuple(dict.fromkeys(_BIND_KEY.findall(_QUOTED.sub("''", text)))),
                        )
                    )
        order = query.args.get("order")
        items = order_items(_order_text(query, order, read, sql, tokens)) if isinstance(order, exp.Order) else ()
        queries.append(PrintedQuery(tables=tables, spelled=tuple(spelled for _name, spelled in named), where=tuple(where), order=items))
    return Printed(tuple(queries), tuple(ordering(sql)[1]), "; ".join(failures))


#: A quoted value, or a bind -- found together so that a bind is never looked
#: for inside a value: ``TO_DATE(:t, 'HH24:MI:SS')`` has one bind, not three.
_BIND_OR_QUOTED = re.compile(r"'(?:[^']|'')*'|:\w+")


def _literal(value: object) -> Optional[str]:
    """*value* as the SQL literal it stands for; ``None`` for a value with no one spelling."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    return None


def filled(text: str, values: dict[str, object]) -> str:
    """*text* -- a statement or a :class:`PrintedTerm`'s text -- with the binds *values* gives written in.

    *values* is keyed as a printed bind map keys it, ``:name`` (or ``name``),
    and holds Python values: a string is quoted, its quotes doubled, a
    number written as it is, ``None`` as ``NULL``.  What is not given, and a
    value with no one SQL spelling (a ``datetime``, a ``bool``), stays a
    bind, to be filled by whoever runs the result.  Nothing inside quotes is
    touched.
    """

    def put(match: re.Match) -> str:
        spelled = match.group(0)
        if spelled.startswith("'"):
            return spelled
        key = spelled if spelled in values else spelled[1:]
        if key not in values:
            return spelled
        literal = _literal(values[key])
        return spelled if literal is None else literal

    return _BIND_OR_QUOTED.sub(put, text)


def unfilled(text: str, values: dict[str, object]) -> tuple[str, ...]:
    """The binds of *text* :func:`filled` leaves as ``:name`` with *values*, in order, each once.

    A bind *values* does not give, and one whose value has no one SQL
    spelling -- a ``datetime``, a ``bool`` -- so that a reader drawing the
    filled text is told which binds it still holds, rather than finding a
    bare ``:timeLimit`` where a time was printed.
    """
    left: list[str] = []
    for match in _BIND_OR_QUOTED.finditer(text):
        spelled = match.group(0)
        if spelled.startswith("'") or spelled in left:
            continue
        key = spelled if spelled in values else spelled[1:]
        if key not in values or _literal(values[key]) is None:
            left.append(spelled)
    return tuple(left)


#: The ``datetime`` constructors a printed bind map spells its values with.
_PRINTED_CALLS = {
    "datetime.datetime": datetime.datetime,
    "datetime": datetime.datetime,
    "datetime.date": datetime.date,
    "date": datetime.date,
    "datetime.timedelta": datetime.timedelta,
    "timedelta": datetime.timedelta,
}


def _printed_value(node: ast.expr) -> object:
    """One value of a printed bind map: a literal, a ``datetime`` of constants, or its text."""
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError):
        pass
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        items = [_printed_value(one) for one in node.elts]
        return items if isinstance(node, ast.List) else tuple(items) if isinstance(node, ast.Tuple) else set(items)
    if isinstance(node, ast.Call) and ast.unparse(node.func) in _PRINTED_CALLS:
        try:
            args = [ast.literal_eval(one) for one in node.args]
            keywords = {one.arg: ast.literal_eval(one.value) for one in node.keywords if one.arg}
            return _PRINTED_CALLS[ast.unparse(node.func)](*args, **keywords)
        except (ValueError, SyntaxError, TypeError, OverflowError):
            pass
    return ast.unparse(node)


def _quoted_reprs(text: str) -> str:
    """*text* with each ``<...>`` outside a string literal written as a string of itself.

    ``repr``'s convention for a value no expression spells -- a
    ``cx_Oracle.Var``, a function, a lock -- is ``<...some description...>``,
    which no parser reads.  Outside quotes a ``<`` opens one, a printed
    container holding no comparison, and its ``>`` is found by depth, as
    ``<function f.<locals>.g at 0x1>`` nests.  An unclosed one is left as it
    is, and the map with it.
    """
    out: list[str] = []
    at = 0
    quote = ""
    while at < len(text):
        char = text[at]
        if quote:
            out.append(text[at : at + 2] if char == "\\" else char)
            if char == quote:
                quote = ""
            at += 2 if char == "\\" else 1
            continue
        if char in "'\"":
            quote = char
        elif char == "<":
            depth = 0
            for end in range(at, len(text)):
                depth += {"<": 1, ">": -1}.get(text[end], 0)
                if depth == 0:
                    out.append(repr(text[at : end + 1]))
                    at = end + 1
                    break
            else:
                out.append(text[at:])
                break
            continue
        out.append(char)
        at += 1
    return "".join(out)


def _printed_map(text: str) -> Optional[dict[object, object]]:
    """*text*, all of it, as the bind map ``str(varMap)`` printed; ``None`` where it is not one."""
    try:
        tree = ast.parse(_quoted_reprs(text.strip()), mode="eval")
    except SyntaxError:
        return None
    if not isinstance(tree.body, ast.Dict) or any(key is None for key in tree.body.keys):
        return None
    return {_printed_value(key): _printed_value(value) for key, value in zip(tree.body.keys, tree.body.values, strict=True)}


def cut(message: str, lead: str = "", gap: str = "") -> Optional[tuple[str, dict[object, object]]]:
    """A logged statement and the bind map logged after it: the text and the map, out of *message*.

    *message* is the line after its logger's prefix; *lead* and *gap* are a
    :attr:`~bamboo.codemap.models.PrintedBy`'s -- what the print writes
    before the statement, and between it and the map.  The statement starts
    at the first *lead*, and the map at the first ``{`` after it, behind
    *gap*, from which the rest of the line reads as a dict: a value
    ``str()`` spells as ``datetime.datetime(2026, 10, 4, 18, 38, 7)`` is
    read as that time, and one with no literal spelling -- ``Decimal('1.5')``,
    or ``repr``'s ``<...>`` -- is kept as its text, so one value never voids
    the map.  ``None`` where no ``{`` does.
    """
    start = message.find(lead) if lead else 0
    if start < 0:
        return None
    start += len(lead)
    at = message.find("{", start)
    while at >= 0:
        if at - len(gap) >= start and message[at - len(gap) : at] == gap:
            binds = _printed_map(message[at:])
            if binds is not None:
                return message[start : at - len(gap)], binds
        at = message.find("{", at + 1)
    return None


def order_owners(sql: str) -> tuple[tuple[str, str], ...]:
    """``(table, column)`` for each column the ``ORDER BY`` of the rows *sql* returns sorts by, its table read as :func:`terms` reads one.

    For a list the map spells blanked and a reader fills in:
    ``getJobs``' ``(CASE {} ELSE {} END)`` with one ``WHEN gshare=:leave{}
    THEN {}`` written in is sorted by ``jobsActive4``'s ``gshare``.  A column
    whose table the statement does not decide is left out.
    """
    read = parsed(sql)
    if read.tree is None:
        return ()
    scopes, root = _scopes(read.tree)
    scope_of = {id(scope.expression): scope for scope, _verb, _kind in scopes}
    found: list[tuple[str, str]] = []
    for query in _rows_from(read.tree, {}, set()):
        order = query.args.get("order")
        scope = scope_of.get(id(query))
        if not isinstance(order, exp.Order) or scope is None:
            continue
        for column in order.find_all(exp.Column):
            if _NAMED_HOLE.search(column.name) or column.name.lower() in _PSEUDO_NAMES:
                continue
            owners, why = _owner(column, scope, root)
            if owners and not why and (owners[0], column.name) not in found:
                found.append((owners[0], column.name))
    return tuple(found)
