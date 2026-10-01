"""Reading SQL text.

:mod:`~bamboo.codemap.panda.sql` reassembles a statement from the Python that
builds it; this reads the reassembled text -- which tables it touches, which
columns it writes and with what, which values it asks rows by, and in what
order and how many come back.  Nothing here looks at Python: every function
takes the statement as a string.

Not attempted: parsing SQL properly.  These are regexes over reassembled
strings, which is enough for ``UPDATE ... SET`` and ``INSERT INTO ... VALUES``
and nothing more.  Anything they cannot read is reported as unexplained rather
than guessed at, which is what the coverage matrix is for.
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


def reads(sql: str) -> list[tuple[str, list[str]]]:
    """Return ``(table, selected columns)`` for each ``SELECT`` in *sql*.

    Only the columns that are plain names are kept.  A projection built from
    expressions (``COUNT(1)``, ``CASE WHEN ...``) says what the query computes
    rather than what the row carries, and a boundary is about the latter.  A
    bare word that is one of SQL's own -- see :data:`_PSEUDO_COLUMNS` -- is
    dropped for that same reason, written without the parentheses that would
    have given it away.

    An interpolated projection keeps no column at all.  ``SELECT {}, FROM ...``
    is what the assembler sees when a statement builds its column list in a
    loop and then slices the trailing comma off, and the empty part after that
    comma is not a name either.
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


def joins(sql: str) -> list[str]:
    """Return the tables *sql* names in a ``FROM`` beyond the first.

    Separate from :func:`reads`, which answers *where the row came from* and so
    keeps naming the leading table.  This answers a different question -- *what
    bounds which rows can be seen at all* -- and a join partner is the usual
    way that bound is written::

        FROM {0}.JEDI_Tasks tabT,{0}.JEDI_AUX_Status_MinTaskID tabA
        WHERE tabT.status=tabA.status AND tabT.jediTaskID>=tabA.min_jediTaskID

    A task below ``min_jediTaskID`` is invisible to that query no matter what
    its status is, so a stale ``JEDI_AUX_Status_MinTaskID`` silently narrows
    thirty-one functions at once -- which is a real stall this map could not
    explain, because ``reads`` stopped at the first table and the auxiliary one
    had never been seen.

    A name the statement defines for itself is not one of these.
    ``checkDuplication_JEDI`` builds ``WITH tmpTab AS (...)`` and then joins
    ``tmpTab`` to itself, which is a step in the query rather than a dependency
    on anything outside it.
    """
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


def joined_columns(sql: str, table: str) -> list[str]:
    """Columns of *table* the statement names, found through its alias.

    The alias is how a join predicate refers to a table --
    ``tabA.min_jediTaskID`` -- so without resolving it the most useful thing
    about a join, *which column bounds the rows*, cannot be read.
    """
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


def creates(sql: str) -> list[str]:
    """Return the tables *sql* inserts rows into.

    Separate from :func:`writes` on purpose.  *Which columns get which values*
    and *which kind of row is being made* are two questions, and deriving the
    second from the first made a table look externally maintained whenever the
    first could not be answered: ``TASK_ATTEMPTS`` is filled by an
    ``INSERT ... SELECT`` in ``log_task_attempt_start``, and the map said
    something outside PanDA keeps it current.
    """
    return [_table_of(match.group(1)) for match in _INSERT_HEAD.finditer(sql)]


def updates(sql: str) -> list[str]:
    """Return the tables *sql* changes rows in.

    Matched with :data:`_UPDATE`, which requires the ``SET``, rather than with
    a bare ``UPDATE <name>``: ``SELECT ... FOR UPDATE NOWAIT`` otherwise reads
    as a statement updating a table called ``nowait``.
    """
    return [_table_of(match.group(1)) for match in _UPDATE.finditer(sql)]


def deletes(sql: str) -> list[str]:
    """Return the tables *sql* deletes rows from.

    Worth reading separately because a ``DELETE`` immediately followed by an
    ``INSERT`` on a command table is not bookkeeping: it means a second command
    silently replaces one that was never picked up.
    """
    return [_table_of(match.group(1)) for match in _DELETE.finditer(sql)]


_PREDICATE_BIND = re.compile(
    r"(?:WHERE|AND|OR)\s+(?:\w+\.)?([A-Za-z_]\w*)\s*=\s*(:\w+)", re.IGNORECASE
)
#: ``status<>:s`` / ``status!='done'``: a row is asked for by *not* having
#: the value.
_EXCLUDING_BIND = re.compile(
    r"(?:WHERE|AND|OR)\s+\(*\s*(?:\w+\.)?([A-Za-z_]\w*)\s*(?:<>|!=)\s*(:\w+)", re.IGNORECASE
)
_PREDICATE_IN = re.compile(
    r"(?:WHERE|AND|OR)\s+(?:\w+\.)?([A-Za-z_]\w*)\s+IN\s*\(([^)]*)\)", re.IGNORECASE
)
_BIND_KEY = re.compile(r":\w+")
_QUOTED = re.compile(r"'([^']*)'")


def predicates(sql: str) -> list[tuple[str, str]]:
    """Return ``(column, bind key)`` for each predicate testing a bind.

    What :func:`conditions` holds back as already said in ``selects``.  The
    values themselves -- which rows a statement asks for, on which table --
    are read off the tree by :func:`terms`; this is the text reading of the
    same ``=`` and ``IN``, kept until ``conditions`` is read off the tree too.

    Equality and ``IN`` only.  ``status<>:s`` asks for every row *but* the
    value, and reading it here filed 36 predicates of the corpus as asking
    for the one value they pass over.
    """
    found = [(column, key) for column, key in _PREDICATE_BIND.findall(sql)]
    for column, inner in _PREDICATE_IN.findall(sql):
        found.extend((column, key) for key in _BIND_KEY.findall(inner))
    return found


_ORDER_BY = re.compile(r"\bORDER\s+BY\s+", re.IGNORECASE)
#: Where an ``ORDER BY`` ends at its own depth.
_ORDER_END = re.compile(r"\b(?:FOR\s+UPDATE|FETCH|LIMIT|OFFSET)\b", re.IGNORECASE)


def _order_clause(sql: str, start: int) -> str:
    """The ``ORDER BY`` list from *start*: to the paren closing the query it is in, or a keyword at its depth.

    Counted, not matched: ``ORDER BY (CASE ... END), currentpriority desc``
    holds a paren of its own.
    """
    depth = 0
    index = start
    while index < len(sql):
        char = sql[index]
        if char == "(":
            depth += 1
        elif char == ")":
            if depth == 0:
                break
            depth -= 1
        elif depth == 0 and _ORDER_END.match(sql, index):
            break
        index += 1
    return re.sub(r"\s+", " ", sql[start:index]).strip()
_LIMIT = re.compile(
    r"\bROWNUM\s*(<=?)\s*(:\w+|\d+|\{\d*\})"
    r"|\bFETCH\s+FIRST\s+(:\w+|\d+|\{\d*\})\s+ROWS?\s+ONLY"
    r"|\bLIMIT\s+(:\w+|\d+|\{\d*\})",
    re.IGNORECASE,
)


def ordering(sql: str) -> tuple[list[str], list[tuple[str, bool]]]:
    """``ORDER BY`` clauses and row limits: which rows come back first, and how many.

    A row that passes the ``WHERE`` can still lose on order and not come back:
    ``getJobs`` sorts by share, priority and ``PandaID`` and keeps
    ``ROWNUM <= :njobs``.  Limits are ``(bind or number, strict)`` -- ``<``
    keeps one fewer than ``<=``.
    """
    orders = [_order_clause(sql, match.end()) for match in _ORDER_BY.finditer(sql)]
    limits: list[tuple[str, bool]] = []
    for match in _LIMIT.finditer(sql):
        if match.group(2):
            limits.append((match.group(2), match.group(1) == "<"))
        else:
            limits.append((match.group(3) or match.group(4), False))
    return [order for order in orders if order], limits


#: One ``WHERE`` term, whole.  Used to recognise a leaf of the clause rather
#: than to find one: the splitting is done by :func:`_parse_clause`, which
#: reads the ``AND``/``OR`` structure the parens declare.  ``predicates``
#: cannot use this -- it answers "which value does this select on", and
#: ``modificationTime<:limit`` names no value.
_CONDITION_TERM = re.compile(
    r"^(?:NOT\s+)?(?:\w+\.)?[A-Za-z_]\w*\s*"
    r"(?:(?:=|<>|!=|<=|>=|<|>)\s*.+"
    r"|(?:NOT\s+)?IN\s*\(.*\)|IS\s+(?:NOT\s+)?NULL)$",
    re.IGNORECASE | re.DOTALL,
)
#: A term that is nothing but ``<column> <equality> <bind>`` -- the shape
#: :func:`predicates` returns, used to tell "already in ``selects``" from
#: "said nowhere".
_CONDITION_EQUALITY = re.compile(r"^[^=<>!]*(?:=|<>|!=)\s*(:\w+)\s*$")
#: ``<column> <op> <column>``.  Qualified on either side it is a join;
#: unqualified it is the statement comparing two of a row's own columns, which
#: is a reason a row was passed over and nothing else in the map records.
_CONDITION_COMPARISON = re.compile(
    r"^(?:NOT\s+)?((?:\w+\.)?\w+)\s*(?:=|<>|!=|<=|>=|<|>)\s*((?:\w+\.)?\w+)\s*$",
    re.IGNORECASE,
)
#: Where the ``WHERE`` clause ends.  Everything after is about the rows that
#: did come back, not about which ones do.
_CLAUSE_END = re.compile(
    r"\b(?:GROUP\s+BY|ORDER\s+BY|HAVING|FOR\s+UPDATE|UNION|FETCH\s+FIRST)\b", re.IGNORECASE
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


def _top_level(text: str, keyword: str) -> list[tuple[int, int]]:
    """Spans of *text* between its ``keyword``s at paren depth zero."""
    spans: list[tuple[int, int]] = []
    start = 0
    depth = 0
    for token in re.finditer(r"\(|\)|\b" + keyword + r"\b", text, re.IGNORECASE):
        found = token.group(0)
        if found == "(":
            depth += 1
        elif found == ")":
            depth -= 1
        elif depth == 0:
            spans.append((start, token.start()))
            start = token.end()
    spans.append((start, len(text)))
    return spans


def _parse_clause(text: str):
    """The ``AND``/``OR`` tree of a ``WHERE`` clause, as nested lists.

    ``("or", [child, …])``, ``("and", [child, …])`` or ``("term", text)``.
    Read rather than pattern-matched because the two readings that came before
    it were both wrong in ways a regex cannot see: ``AND (`` hides a term from
    anything that wants whitespace, and looking outward for an enclosing paren
    misses ``(A AND B) OR (C AND D) AND E``, where the disjunction is at the
    top and has no parens of its own.

    ``AND`` binds tighter than ``OR``, so the ``OR`` split happens first --
    which is what makes ``E`` above part of the second alternative and not a
    requirement in its own right.
    """
    text = text.strip()
    while text.startswith("(") and text.endswith(")"):
        depth = 0
        for index, char in enumerate(text):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0 and index < len(text) - 1:
                    return _parse_clause_split(text)
        text = text[1:-1].strip()
    return _parse_clause_split(text)


def _parse_clause_split(text: str):
    for keyword, kind in (("OR", "or"), ("AND", "and")):
        spans = _top_level(text, keyword)
        if len(spans) > 1:
            return (kind, [_parse_clause(text[lo:hi]) for lo, hi in spans])
    return ("term", re.sub(r"\s+", " ", text).strip())


def _keep_term(term: str, held: set[str], in_or: bool) -> bool:
    """Whether *term* is a reason a row was passed over that the map lacks."""
    if not _CONDITION_TERM.match(term):
        return False
    equality = _CONDITION_EQUALITY.match(term)
    # Dropped as "already in ``selects``" only outside a disjunction.  Inside
    # one it is not the same fact: ``selects`` says the statement asks by the
    # value outright, and the alternative says it accepts that *or* something
    # else.  ``predicates`` also picks up a group's later branches while the
    # paren hides its first, so dropping them hands a reader half a
    # disjunction and no sign that the rest exists.
    if not in_or and equality and equality.group(1) in held:
        return False
    if not _BIND_KEY.search(term):
        sides = _CONDITION_COMPARISON.match(term)
        if sides and ("." in sides.group(1) or "." in sides.group(2)):
            return False  # a join says how two tables meet, not why a row was skipped
    return True


def _prune(node, held: set[str], in_or: bool):
    """Drop what is not an answer, and drop a disjunction that lost a branch.

    An ``OR`` with one branch gone is satisfied by that branch whatever the
    row does, so recording the rest would state a requirement the statement
    does not make -- the direction that gets a row ruled out for failing a
    test it never faced.
    """
    kind, body = node
    if kind == "term":
        return node if _keep_term(body, held, in_or) else None
    children = [_prune(child, held, in_or or kind == "or") for child in body]
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
    kind, body = node
    if kind == "term":
        return (body,)
    if kind == "and":
        return tuple(term for child in body for term in _terms_of(child))
    return (f"({' OR '.join(_render_branch(child) for child in body)})",)


def _render_branch(node) -> str:
    terms = _terms_of(node)
    return f"({' AND '.join(terms)})" if len(terms) > 1 else terms[0]


def _as_condition(node) -> Condition:
    kind, body = node
    parts = body if kind == "or" else [node]
    return Condition(
        tuple(
            Branch(
                _terms_of(part),
                tuple(key for term in _terms_of(part) for key in _BIND_KEY.findall(term)),
            )
            for part in parts
        )
    )


def conditions(sql: str) -> list[Condition]:
    """Return what *sql* asks of a row beyond the values it selects on.

    Beside :func:`predicates` rather than inside it, and the two answer
    different questions.  ``predicates`` says *which value* a statement asks
    rows by, so it reads equalities on a bind and nothing else -- that is what
    a subject's vocabulary is made of.  This says *what else had to be true*,
    which is the question a row that holds the value and never came back
    raises: ``copyArchive.py``'s ``holding`` query also requires
    ``modificationTime<:modificationTime``, and the map had no way to say so.

    **The list reads as a conjunction**, so a disjunction is one entry with
    its alternatives joined -- split up, ``(status=:running OR
    status=:scouting)`` asks for a row whose status is both, which no row can
    be.  That is why the clause is parsed rather than scanned for terms.

    What is left out, and why:

    * **Terms ``predicates`` already returns, where they stand alone.**  Those
      are the ``selects`` list; repeating them would make one fact two rows.
      Inside a disjunction they are kept -- see :func:`_keep_term`.
    * **Join terms.**  ``c1.jediTaskID=d.jediTaskID`` says how two tables meet,
      not why a row was passed over.  182 of the corpus's terms are joins, and
      left in they bury the ones that are answers -- one statement in
      ``entity_module.py`` carries six.
    * **A disjunction that lost a branch to either of those.**  It no longer
      constrains anything, and recording what is left would over-state it.

    What is kept and cannot be walked is still returned: an ``IS NULL`` and a
    column compared against another column name no value to go looking for,
    and saying the term is there is the point.  The caller decides what to do
    with :attr:`Branch.binds`, which is empty for exactly those.
    """
    clause = where_clause(sql)
    if not clause:
        return []
    # An excluding term is read into ``excludes`` the way an equality is read
    # into ``selects``, so it is held the same way.
    held = {key for _column, key in [*predicates(sql), *_EXCLUDING_BIND.findall(sql)]}
    tree = _prune(_parse_clause(clause), held, False)
    if tree is None:
        return []
    kind, body = tree
    return [_as_condition(child) for child in body] if kind == "and" else [_as_condition(tree)]


def where_clause(sql: str) -> str:
    """The text between ``WHERE`` and whatever ends the clause, or ``""``.

    One spelling, because two readers want it for opposite questions -- which
    terms the statement makes, and whether the placeholders in them are
    readable at all -- and a second copy of "where the clause ends" is how the
    two would come to disagree about a statement with a ``UNION`` in it.
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
    try:
        yield
    finally:
        _SCHEMA = before
        terms.cache_clear()


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
            return Parsed(sqlglot.parse_one(text, read="oracle"), "")
        except ParseError as error:
            said = [one.get("description") or "" for one in error.errors]
            if said and all(_REQUIRED in one for one in said):
                tree = sqlglot.parse_one(text, read="oracle", error_level=ErrorLevel.IGNORE)
                if tree is not None and _holes_stand_for_what_is_missing(tree):
                    return Parsed(tree, "")
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
            if _is_hole(column) or column.name.lower() in _PSEUDO_NAMES:
                continue
            holder = _filtering(column)
            if holder is None:
                continue
            compared = _compared(column)
            if compared is None:
                continue
            excluding, values, undecided = compared
            negations = _negations(column.parent)
            if negations is None:
                continue
            owners, why = _owner(column, scope, root)
            if why:
                spelled = column.sql(dialect="oracle")
                among = f" ({', '.join(owners)})" if owners else ""
                unresolved.add(f"{spelled}{among} -- {why}")
                continue
            found.append(
                Term(
                    table=owners[0],
                    column=column.name,
                    verb=verb,
                    scope=kind,
                    excluding=excluding != bool(negations % 2),
                    binds=tuple(f":{value.name}" for value in values if isinstance(value, exp.Placeholder)),
                    literals=tuple(
                        value.this for value in values if isinstance(value, exp.Literal) and value.is_string
                    ),
                    undecided=undecided,
                )
            )
    return Reading(tuple(found), tuple(sorted(unresolved)), True)
