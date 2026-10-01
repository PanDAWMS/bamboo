"""Reading SQL text.

:mod:`~bamboo.codemap.panda.sql` reassembles a statement from the Python that
builds it; this reads the reassembled text -- which tables it touches, which
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

import contextlib
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
# The schema is usually an f-string placeholder, which reassembly leaves as
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


def writes(sql: str) -> list[SqlWrite]:
    """Return the write statements in *sql*, as far as they can be read."""
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


def supplied_at(
    statement: str, kind: str, column: str, supplied: ColumnValue
) -> Optional[int]:
    """Where *statement* writes *supplied* into *column*, as an end offset.

    ``None`` when this statement does not write that value into that column.

    **The reading is chosen by kind rather than tried in turn.**  An UPSERT
    builds its ``INSERT`` and its ``UPDATE`` through one local -- seven
    functions here do -- and both spell the same column with the same value, so
    a reading that accepted either would answer for the ``INSERT`` when asked
    about the ``UPDATE``.  That is the same mis-answer this whole path exists
    to remove, arriving from the other side.
    """
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
        return _reads_in_text(sql)
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
        return _joins_in_text(sql)
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
        return _joined_columns_in_text(sql, table)
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
    return _creates_in_text(sql) if found is None else found


def updates(sql: str) -> list[str]:
    """Return the tables *sql* changes rows in.  ``SELECT ... FOR UPDATE`` changes none."""
    found = _targets(sql, exp.Update)
    return _updates_in_text(sql) if found is None else found


def deletes(sql: str) -> list[str]:
    """Return the tables *sql* deletes rows from.

    Worth reading separately because a ``DELETE`` immediately followed by an
    ``INSERT`` on a command table is not bookkeeping: it means a second command
    silently replaces one that was never picked up.
    """
    found = _targets(sql, exp.Delete)
    return _deletes_in_text(sql) if found is None else found


_BIND_KEY = re.compile(r":\w+")
_QUOTED = re.compile(r"'([^']*)'")


#: Where the ``WHERE`` clause ends.  Everything after is about the rows that
#: did come back, not about which ones do.
_CLAUSE_END = re.compile(
    r"\b(?:GROUP\s+BY|ORDER\s+BY|HAVING|FOR\s+UPDATE|UNION|FETCH\s+FIRST)\b", re.IGNORECASE
)


def where_clause(sql: str) -> str:
    """The text between ``WHERE`` and whatever ends the clause, or ``""``.

    Asked by the readers of holes -- whether the placeholders a statement
    filters on are readable at all.  What the clause requires of a row is
    read off the tree, by :func:`conditions`.
    """
    opened = _WHERE.search(sql)
    if opened is None:
        return ""
    rest = sql[opened.end() :]
    closed = _CLAUSE_END.search(rest)
    return rest[: closed.start()] if closed else rest


#: A hole the reassembly left where the source interpolates -- ``{}``,
#: ``{0}``.  Read as a name, which is what most of them are: the schema, a
#: table, a column.
_HOLE = re.compile(r"\{\d*\}")
_HOLE_NAME = "__h"
_NAMED_HOLE = re.compile(rf"{_HOLE_NAME}\d+")

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
ROW, CONSULT, NEGATED = "row", "consult", "negated"

_SCHEMA: Optional[dict[str, frozenset[str]]] = None


class Parsed(NamedTuple):
    """A statement as a tree, or why it is not one."""

    tree: Optional[exp.Expression]
    failure: str
    #: The text around where the parser stopped, holes spelled ``{}`` again.
    near: str = ""
    #: The text the tree was parsed from: holes named, and any read as empty
    #: gone.  The tree's positions are offsets into this.
    text: str = ""


class Term(NamedTuple):
    """One column a statement compares with a value."""

    #: The table the column belongs to, schema stripped; ``{}`` for a table
    #: name the reassembly could not render.
    table: str
    column: str
    #: Whose rows the term picks: ``read``, ``updated`` or ``deleted``.  An
    #: ``INSERT ... SELECT`` picks rows to read.
    verb: str
    #: :data:`ROW`, :data:`CONSULT` or :data:`NEGATED`.
    scope: str
    #: ``<>``, ``!=``, ``NOT IN``: rows without the value.
    excluding: bool
    #: The bind keys and the quoted values it is compared with, as written.
    binds: tuple[str, ...]
    literals: tuple[str, ...]
    #: Compared with something whose value the statement does not say -- a
    #: hole, or a subquery.
    undecided: bool


class Reading(NamedTuple):
    """What a statement compares, and where reading it stopped."""

    terms: tuple[Term, ...]
    #: ``<expression> -- <reason>``, the spelling ``ReadSiteNode.unresolved`` uses.
    unresolved: tuple[str, ...]
    parsed: bool


@contextlib.contextmanager
def with_schema(columns: Optional[dict[str, frozenset[str]]]) -> Iterator[None]:
    """Decide unqualified columns by *columns*, ``{table: columns}`` lowered, inside.

    A statement joining two tables may name a column without saying whose it
    is -- valid only because one of them has it, and which one is not in the
    text.  ``read_schema`` reads the answer from panda-database.
    """
    global _SCHEMA
    before, _SCHEMA = _SCHEMA, columns
    terms.cache_clear()
    conditions.cache_clear()
    refusals.cache_clear()
    try:
        yield
    finally:
        _SCHEMA = before
        terms.cache_clear()
        conditions.cache_clear()
        refusals.cache_clear()


_CREATE_TABLE = re.compile(r'CREATE\s+(?:GLOBAL\s+TEMPORARY\s+)?TABLE\s+"(\w+)"\."(\w+)"\s*\(', re.IGNORECASE)
_COLUMN_DEFINITION = re.compile(r'^\s*"(\w+)"\s+[A-Za-z]')


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
        after = sql[match.end() : match.end() + 1]
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
    """
    text = _named_holes(sql)

    def failed(failure: str, near: str) -> Parsed:
        return Parsed(None, failure, " ".join(_NAMED_HOLE.sub("{}", near).split()))

    for _attempt in range(len(_NAMED_HOLE.findall(text)) + 1):
        try:
            return Parsed(sqlglot.parse_one(text, read="oracle"), "", "", text)
        except ParseError as error:
            said = [one.get("description") or "" for one in error.errors]
            if said and all(_REQUIRED in one for one in said):
                tree = sqlglot.parse_one(text, read="oracle", error_level=ErrorLevel.IGNORE)
                if tree is not None and _holes_stand_for_what_is_missing(tree):
                    return Parsed(tree, "", "", text)
                return failed(said[0], " ".join(text.split()[-8:]))
            first = error.errors[0] if error.errors else {}
            hole = _NAMED_HOLE.match(first.get("highlight") or "")
            if hole is None:
                before = " ".join((first.get("start_context") or "").split()[-4:])
                near = f"{before} {first.get('highlight') or ''}".strip()
                return failed(said[0] if said else str(error), near or " ".join(text.split()[:8]))
            text = re.sub(rf"\b{hole.group(0)}\b", "", text, count=1)
        except SqlglotError as error:
            return failed(str(error).splitlines()[0], " ".join(text.split()[:8]))
    return failed("every hole was read as empty and it still does not parse", " ".join(text.split()[-8:]))


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
    negations, inside = 0, False
    while scope is not None:
        if scope.scope_type == ScopeType.SUBQUERY:
            inside = True
            node = scope.expression.parent
            while node is not None and not isinstance(node, (exp.Where, exp.Select, exp.Update, exp.Delete)):
                if isinstance(node, exp.Not):
                    negations += 1
                node = node.parent
        scope = scope.parent
    if not inside:
        return ROW
    return NEGATED if negations % 2 else CONSULT


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
    read = parsed(sql)
    tree = read.tree
    if tree is None:
        return Reading((), (f"{read.near} -- {NOT_PARSED}",), False)
    found: list[Term] = []
    unresolved: set[str] = set()
    scopes, root = _scopes(tree)
    for scope, verb, kind in scopes:
        for column in _own_columns(scope):
            term = _term(column, scope, root, verb, kind)
            if isinstance(term, Term):
                found.append(term)
            elif term is not None:
                unresolved.add(term)
    return Reading(tuple(found), tuple(sorted(unresolved)), True)


def _term(column: exp.Column, scope, root, verb: str, kind: str) -> Optional[Term | str]:
    """The term *column* is compared in, why its table is not known, or ``None``."""
    # ``tabT.{0}=:{0}`` -- a column Python names at run time, which the
    # reassembly already reports.  Read as a column called ``__h4``, it
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
    return Term(
        table=owners[0],
        column=column.name,
        verb=verb,
        scope=kind,
        excluding=excluding != bool(negations % 2),
        binds=binds,
        literals=tuple(
            value.this for value in values if isinstance(value, exp.Literal) and value.is_string
        ),
        undecided=undecided,
    )


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
            return (kind, [_split(tokens, a + 1, b) for a, b in zip(bounds, bounds[1:])])
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
        # what it asks.
        return _KEEP if negations % 2 else _DROP
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
    holes = [(match.group(0), sql[match.end() : match.end() + 1].isalpha()) for match in _HOLE.finditer(sql)]

    def back(match: re.Match) -> str:
        number = int(match.group(1)) - 1
        if number >= len(holes):
            return match.group(0)
        spelled, padded = holes[number]
        return spelled if padded or not match.group(2) else spelled + match.group(2)

    return re.sub(rf"{_HOLE_NAME}(\d+)( ?)", back, fragment)


def _where_tree(where: exp.Where, scope, root, read: Parsed, sql: str, tokens: list):
    """*where*'s clause as a tree of ``("term", text, kind)``, or ``None``.

    Split on the tokens, so each term keeps the statement's own spelling --
    a tree regenerated by sqlglot respells it (``endTime IS NOT NULL`` comes
    back as ``NOT endTime IS NULL``).  Each piece is then matched to the term
    of the tree it spans, which says what it is.  ``None`` when the two do not
    agree on where a term begins and ends.
    """
    placed = [
        (node.meta["start"], node)
        for node in where.this.find_all(exp.Identifier, exp.Literal)
        if "start" in node.meta
    ]
    if not placed:
        return None
    first = min(start for start, _node in placed)
    opened = [index for index, token in enumerate(tokens) if token.token_type == _T.WHERE and token.start < first]
    if not opened:
        return None
    lo, hi = _clause(tokens, opened[-1])
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
            return ("term", "", _DROP)
        start, end = tokens[a].start, tokens[b - 1].end
        held = {id(_term_holding(n, where, joins)): _term_holding(n, where, joins) for s, n in placed if start <= s <= end}
        text = " ".join(_respelled(read.text[start : end + 1], sql).split())
        if len(held) > 1:
            agreed = False
        if len(held) != 1:
            return ("term", text, _DROP)
        return ("term", text, _term_kind(next(iter(held.values())), scope, root))

    tree = build(_split(tokens, lo, hi))
    return tree if agreed else None


def _prune(node, in_or: bool):
    """Drop what is not an answer, and drop a disjunction that lost a branch.

    An ``OR`` with one branch gone is satisfied by that branch whatever the
    row does, so recording the rest would state a requirement the statement
    does not make -- the direction that gets a row ruled out for failing a
    test it never faced.  A term ``selects`` already carries is dropped only
    outside a disjunction: inside one it is not the same fact, since
    ``selects`` says the statement asks by the value outright.
    """
    if node[0] == "term":
        _term, _text, kind = node
        return node if kind == _KEEP or (kind == _HELD and in_or) else None
    kind, body = node
    children = [_prune(child, in_or or kind == "or") for child in body]
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


def _either(alternatives: list[list[Condition]]) -> list[Condition]:
    """What a row from any one of *alternatives* is required to satisfy.

    ``(A and B) or (A and C)`` is ``A``, required outright, and ``B or C`` as
    one condition; an alternative that requires nothing beyond the common
    part makes that disjunction free, and then it is not recorded.
    """
    common = [one for one in alternatives[0] if all(one in other for other in alternatives[1:])]
    remainders: list[tuple[Condition, ...]] = []
    for alternative in alternatives:
        rest = tuple(one for one in alternative if one not in common)
        if rest not in remainders:
            remainders.append(rest)
    if len(remainders) == 1:
        return common + list(remainders[0])
    if not all(remainders):
        return common
    return common + [
        Condition(tuple(_branch(tuple(term for one in rest for term in _as_terms(one))) for rest in remainders))
    ]


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
      kept whole: what it asks is the subquery turned round.
    * **A disjunction that lost a branch to any of those.**  It no longer
      constrains anything, and recording what is left would over-state it.
    * **A statement that does not parse**, which ``unresolved`` reports.

    What is kept and cannot be walked is still returned: an ``IS NULL`` names
    no value to go looking for, and saying the term is there is the point.
    The caller decides what to do with :attr:`Branch.binds`, which is empty
    for exactly those.
    """
    read = parsed(sql)
    if read.tree is None:
        return ()
    tokens = _tokens(read.text)
    scopes, root = _scopes(read.tree)
    scope_of = {id(scope.expression): scope for scope, _verb, _kind in scopes}
    ctes: dict[str, exp.Expression] = {}
    entered: set[int] = set()

    def required(node: Optional[exp.Expression]) -> list[Condition]:
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
        found: list[Condition] = []
        where = node.args.get("where")
        scope = root if isinstance(node, (exp.Update, exp.Delete)) else scope_of.get(id(node))
        if where is not None and scope is not None:
            clause = _where_tree(where, scope, root, read, sql, tokens)
            kept = _prune(clause, False) if clause is not None else None
            if kept is not None:
                found.extend(_as_condition(child) for child in (kept[1] if kept[0] == "and" else [kept]))
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
    found: list[Condition] = []
    for one in required(read.tree):
        if one not in found:
            found.append(one)
    return tuple(found)


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


def _null_safe(conjunct: exp.Expression) -> Optional[exp.Column]:
    """The column of ``(col IS NULL OR col<>:v)``, or ``None`` for any other ``OR``.

    The one disjunction that still leaves a value out of every row: a row
    holding ``v`` is not null and fails the other branch.  Five statements in
    the corpus spell "not this value" that way, ``lockJobsForReassign``'s
    ``relocationFlag`` among them, for the rows that never had one.
    """
    def spelled(column: exp.Column) -> tuple[str, str]:
        return (column.table or "").lower(), column.name.lower()

    branches = _disjuncts(conjunct)
    nulls = [b for b in branches if isinstance(b, exp.Is) and isinstance(b.expression, exp.Null)]
    rest = [b for b in branches if not any(b is n for n in nulls)]
    if not nulls or len(rest) != 1 or not all(isinstance(n.this, exp.Column) for n in nulls):
        return None
    columns = list(rest[0].find_all(exp.Column))
    if len(columns) != 1 or {spelled(n.this) for n in nulls} != {spelled(columns[0])}:
        return None
    return columns[0]


def _outright(where: exp.Where, scope, root, verb: str, kind: str) -> tuple[Term, ...]:
    """The exclusions *where* makes of every row: its own terms outside any ``OR``.

    And inside one where it is ``col IS NULL OR`` the exclusion -- see
    :func:`_null_safe`.
    """
    found: list[Term] = []
    own = {id(column) for column in _own_columns(scope)}
    for conjunct in _conjuncts(where.this):
        guarded = _null_safe(conjunct) if isinstance(conjunct, (exp.Or, exp.Paren)) else None
        if guarded is not None and id(guarded) in own:
            term = _term(guarded, scope, root, verb, kind)
            if isinstance(term, Term) and term.excluding:
                found.append(term)
            continue
        for column in conjunct.find_all(exp.Column):
            if id(column) not in own:
                continue
            walk, inside = column.parent, False
            while walk is not None and walk is not conjunct.parent:
                inside = inside or isinstance(walk, (exp.And, exp.Or))
                walk = walk.parent
            if inside:
                continue
            term = _term(column, scope, root, verb, kind)
            if isinstance(term, Term) and term.excluding:
                found.append(term)
    return tuple(found)


@functools.lru_cache(maxsize=None)
def refusals(sql: str) -> tuple[tuple[Term, ...], ...]:
    """The exclusions every row *sql* returns satisfies, one tuple per way a row comes back.

    Narrower than the excluding :func:`terms`, which say a statement *names* a
    value to pass over.  ``getFailureCountsForTask_JEDI`` names
    ``relocationFlag<>:relThrottled`` inside one branch of an ``OR``, and its
    other branches return throttled jobs all the same; a reader told it leaves
    them out would stop looking at a query that picks them up.

    So a term counts only where every row has to meet it: a conjunct of the
    ``WHERE`` outside any ``OR``, of the statement or of the derived table or
    ``WITH`` clause the rows come from.  A ``UNION`` returns rows from any of
    its branches, and each branch is a tuple of its own -- the caller names
    the values and keeps the ones every branch passes over, since which
    subject a table's column is, is not this module's to say.  ``()`` for a
    statement that does not parse.
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
            own = _outright(where, scope, root, verb, kind)
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
    if isinstance(node, exp.Placeholder):
        return _respelled(f":{node.name}", sql)
    if isinstance(node, exp.Literal) and not node.is_string:
        return node.this
    if _is_hole(node):
        return _respelled(node.name, sql)
    return None


def _order_text(query: exp.Expression, order: exp.Order, read: Parsed, sql: str, tokens: list) -> str:
    """*order*'s list as the statement spells it.

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
        return ""
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
        return ""
    start, end = tokens[opened[-1] + 1].start, tokens[last - 1].end
    return " ".join(_respelled(read.text[start : end + 1], sql).split())


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
