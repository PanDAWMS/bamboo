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
from typing import Callable, Iterable, NamedTuple, Optional

from bamboo.codemap.calls import FILLERS, Calls, Route, Target, Types, bind
from bamboo.codemap.calls import said as passed_by
from bamboo.codemap.models import (
    CALL_READING_ROUTE,
    AddedTerm,
    Anchor,
    Branch,
    CallReading,
    ConditionNode,
    ConditionReading,
    ConditionValue,
    CoverageStat,
    DiagnosticTemplate,
    EntityNode,
    GuardRead,
    JunctionNode,
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
from bamboo.codemap.panda import sql, sqltext, values
from bamboo.codemap.panda.attribution import SpecAttributor
from bamboo.codemap.panda.pathcond import (
    UNDECIDED,
    ancestors,
    attach_parents,
    compatible,
    containers_changed,
    enclosing_function,
    functions_with_owner,
    handed_in,
    literal_values,
    names_bound_by,
    path_condition,
    reaching_definitions,
    site_of,
    surviving_writes,
    targets_of,
    written_value,
)

# The block a decision records about itself reads the same way whichever slice
# found the write, so it is read in one place and imported rather than copied.
# Both halves of the map disagreeing about a fact is the failure this codebase
# has spent the most effort undoing.
from bamboo.codemap.panda.recognizers import progress

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
PLACEHOLDER_MISSING_COLON = "the placeholder is bound without its leading colon"
NOTHING_BINDS_IT = "nothing in this function binds this placeholder"
NO_DECIDING_FRAGMENT = "the fragment that wrote this value could not be located"

#: The read side counts statements, not columns, so its reasons are about how
#: a query is run rather than where a value came from.  ``sql.executions``
#: accepts ``execute``/``executemany``/``querySQL``/``querySQLS`` with at least
#: two arguments; each reason below is one of the ways a statement this slice
#: assembled fails to reach it.
RUN_WITHOUT_VARMAP = "run through an accepted call with no varmap argument"
RUN_BUT_TEXT_DIFFERS = "run through an accepted call, but the two reassemblies disagree"
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
    keeps the bookkeeping out is promotion, not the absence of a spec class.
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
    answers to "which rows did this function pick up".
    """
    read_tables = sorted({table for verb, table in _table_verbs(run.sql) if verb == "read"})
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
        for stop in run.unresolved
    ]


def _skip_stops(skipped: list[sql.Skip], rel_path: str, owner: str) -> list[ReadingStop]:
    """An execution nothing was read out of, as a stop on the statement it runs.

    Left out where the text is known and names no table: ``ALTER SESSION`` and
    ``SET`` run through the same cursor and are about no rows.  Where the text
    is not known at all the tables are not either, and the stop says so rather
    than reading as a write.
    """
    found: list[ReadingStop] = []
    for skip in skipped:
        verbs = [pair for text in skip.texts for pair in _table_verbs(text)]
        if skip.texts and not verbs:
            continue
        found.append(
            ReadingStop(
                reason=skip.reason,
                detail=skip.detail,
                file=rel_path,
                owner=owner,
                line=skip.line,
                statement=skip.line,
                reads=sorted({table for verb, table in verbs if verb == "read"}),
                reads_known=bool(skip.texts),
            )
        )
    return found


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
    """Where a guarded write's row keys were filled from -- see :class:`RowKeySource`."""
    handed = ""
    if run.varmap is not None:
        for _column, bind in sorted(statement.keys.items()):
            for filled in sql.bound_values(func, run.varmap, bind, run.window, at=run.run_at):
                handed = handed_in(func, filled.value)
                if handed:
                    break
            if handed:
                break
    return RowKeySource(
        owner=f"{rel_path}::{func.name}",
        file=rel_path,
        statement=run.call.lineno,
        columns=sorted(statement.preconditions),
        keys=[f"{column}={bind}" for column, bind in sorted(statement.keys.items())],
        handed_by=handed,
    )


def extract(
    modules: list[SourceModule],
    map_id: str,
    derived_from: str,
    attributor: SpecAttributor,
    row_keys: Optional[list[RowKeySource]] = None,
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
    asks for it, and the seven above are unpacked in many places.
    """
    junctions: dict[str, JunctionNode] = {}
    stopped: list[ReadingStop] = []
    coverage: list[CoverageStat] = []
    attributed: set[tuple[str, str, str]] = set()
    uncovered: set[str] = set()
    diagnostics: list[DiagnosticTemplate] = []
    unexplained: list[Unexplained] = []
    settle = values.resolver()
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
            for run in sql.executions(func, skipped):
                key = (run.variable, run.sql, run.varmap, run.window)
                if key in seen:
                    continue
                seen.add(key)
                stopped.extend(_run_stops(run, module.rel_path, owner))
                for statement in sqltext.writes(run.sql):
                    spec_class = attributor.class_for_table(statement.table)
                    # Settled once per statement, not once per column: every
                    # column of one ``SET`` is bounded by the same ``WHERE``,
                    # and settling inside the column loop would count each
                    # unsettled bind as many times as the statement has
                    # columns.
                    preconditions = []
                    for term in sorted(statement.preconditions.values()):
                        settled, done = sql.settled_precondition(
                            term, func, run.varmap, run.window, settle, at=run.run_at
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
                    if row_keys is not None and statement.preconditions:
                        row_keys.append(_row_key_source(func, run, statement, module.rel_path))
                    # A name nobody could read is not a table that holds no
                    # spec.  Listed together they read as one finding, and only
                    # one of the two is a gap in the extraction -- see
                    # ``ReadSide.unreadable``, which counts these instead.
                    if spec_class is None and not _UNREAD_TABLE.fullmatch(statement.table):
                        uncovered.add(statement.table)
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
                            setters=setters,
                            beside=lambda node, terms=sorted(statement.preconditions.values()): [
                                sql.settled_precondition(
                                    term, func, run.varmap, run.window, settle, beside=node, at=run.run_at
                                )[0]
                                for term in terms
                            ],
                        )
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
    """Is this execution ``executemany``, whose varmap argument is a sequence?

    Asked of the call the reader already paired with the statement, so the
    answer cannot disagree with the pairing.
    """
    call = getattr(run, "call", None)
    return bool(
        call is not None
        and isinstance(getattr(call, "func", None), ast.Attribute)
        and call.func.attr == "executemany"
    )


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
    for statement, spans in sql.variant_spans(func, run.variable):
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
        scan = sql.scan_binds(func, run.varmap, supplied.text, run.window, at=run.run_at)
        found = []
        for bind in scan.binds:
            value, site = bind.value, bind.site
            settled = settle(value, func)
            if settled:
                found.extend(_Outcome(outcome, 1, site, [], site) for outcome in settled)
                continue
            template = values.diagnostic_template(value)
            if template:
                templates.append((template, site))
            if isinstance(value, ast.Name):
                # Reaching definitions, the same reading the attribute slice
                # makes of a local: 19% of the corpus's writes fill the bind
                # from a variable a guarded chain assigned above it.
                reached = literal_values(func, value.id, settle)
                if reached:
                    dominating = path_condition(site)
                    found.extend(
                        _Outcome(
                            outcome,
                            1,
                            site,
                            [c for c in conditions if c not in dominating],
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
                    [],
                    site,
                    tuple(values.value_frames(value, func)),
                )
            )
        if not found:
            # Which of the several reasons, from the walk that failed rather
            # than from a second walk asking why -- see ``sql.BindScan``.
            if _runs_many(run):
                # ``executemany(sql, varMaps)`` is handed a *sequence* of maps,
                # each appended after being filled, so the name in the call
                # never carries a placeholder as a key.  Read off the call form
                # rather than off the list, which is what makes it exact.
                why.append(VARMAP_IS_A_LIST)
            elif not scan.varmap_assigned:
                why.append(VARMAP_FROM_OUTSIDE)
            elif scan.outside_window:
                why.append(BOUND_OUTSIDE_WINDOW)
            elif supplied.text.lstrip(":") in scan.constant_keys:
                # ``hs_scrapers`` writes ``r["source"] = url`` against
                # ``VALUES (:source)``.  The key is there and the colon is not,
                # so a reader matching the spelled placeholder misses a value
                # that was decided right here.
                why.append(PLACEHOLDER_MISSING_COLON)
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
    for node in ast.walk(func):
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
) -> None:
    """Add one branch per decided value to this write site's junction.

    *beside*, where given, settles the row precondition for one branch from
    the binds that can run with the statement deciding its value -- see
    :func:`sql.settled_precondition`.
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

    known = {(branch.outcome, tuple(branch.path_condition)): branch for branch in junction.branches}
    for outcome, tier, node, extra, decided_at, frames in outcomes:
        condition = path_condition(node) + extra
        held = known.get((outcome, tuple(condition)))
        if held is not None:
            held.value_frames = pooled_frames(held.value_frames, list(frames))
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
            row_precondition=beside(node) if beside is not None else row_precondition,
            order=len(junction.branches),
            tier=tier,
            tags=tags,
            messages=messages,
            line=decided_at.lineno,
            value_frames=list(frames),
        )
        known[(outcome, tuple(condition))] = branch
        junction.branches.append(branch)



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


def _decided_chain(node: ast.If) -> bool:
    """Whether an ``if``/``elif`` chain ends in an ``else``, so one of its branches always runs."""
    while node.orelse:
        if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
            node = node.orelse[0]
            continue
        return True
    return False


def _on_some_paths(fragment: ast.stmt, call: ast.Call) -> bool:
    """Whether *fragment* is appended on some of the paths to *call* only.

    The reassembly keeps such a fragment in every variant -- reading an extra
    ``AND`` was the safe direction for what it was built for -- so a statement
    read off one names it as though every run carried it.  Under an ``if``
    whose chain has no ``else``, in a loop that may not run, or in an
    ``except`` it is not.  Branches of a chain that does end in an ``else``
    are variants of their own, and in each the fragment is certain.  So is a
    branch the call being read (:func:`sql.at_call`) takes for certain:
    ``copyArchive`` hands ``lockJobsForReassign`` ``onlyReassignable=True``,
    and every run of that call appends the ``relocationFlag`` test.
    """
    around_call = {id(node) for node in ancestors(call)}
    previous: ast.AST = fragment
    for node in ancestors(fragment):
        if id(node) in around_call:
            return False
        if isinstance(node, ast.If):
            taken = sql.decided_here(node.test)
            if taken is not UNDECIDED and taken == any(previous is one for one in node.body):
                previous = node
                continue
            head = node
            while isinstance(getattr(head, "parent", None), ast.If) and head.parent.orelse == [head]:
                head = head.parent
            if not _decided_chain(head):
                return True
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.While, ast.Match)):
            return True
        elif isinstance(node, ast.Try) and previous not in node.body and previous not in node.finalbody:
            return True
        previous = node
    return False


#: A hole of the reassembly, wherever a variant has one: ``{}`` for a
#: ``.format`` or f-string, ``%s`` for ``sql % tableName``.
_VARIANT_HOLE = re.compile(r"\{[^{}]*\}|%(?:\([^)]*\))?[sdr]")


def _filled(variant: str, statement: str) -> Optional[tuple[list[str], str]]:
    """How *statement* spells *variant*: what fills each of its holes, and what the call adds after it.

    ``execute((sql + comment) % table, ...)`` runs ``sql``'s variant with
    ``FROM {}`` turned into ``FROM ATLAS_PANDA.jobsActive4`` and the comment
    after it.  ``None`` where *statement* does not start with *variant* so filled.
    """
    pieces = [re.escape(" ".join(piece.split())) for piece in _VARIANT_HOLE.split(variant)]
    match = re.fullmatch(r"\s*(.*?)\s*".join(pieces) + r"(.*)", " ".join(statement.split()))
    if match is None:
        return None
    *fillings, after = match.groups()
    return fillings, after


def _cut_text(variant: str, cuts: list[tuple[int, int]], statement: str) -> Optional[str]:
    """*statement*, run from *variant*, with the stretches *cuts* of the variant taken out.

    What the database is sent, less what some paths do not append: a hole
    the cut leaves is filled as the run fills it, and what the call adds
    after the variant stays.  ``None`` where the run does not spell the
    variant.
    """
    if not cuts:
        return statement
    filled = _filled(variant, statement)
    if filled is None:
        return None
    fillings, after = filled
    edits = [(start, end, " ") for start, end in cuts]
    for hole, filling in zip(_VARIANT_HOLE.finditer(variant), fillings, strict=True):
        if not any(start <= hole.start() and hole.end() <= end for start, end in cuts):
            edits.append((hole.start(), hole.end(), filling))
    text = variant
    for start, end, replacement in sorted(edits, reverse=True):
        text = text[:start] + replacement + text[end:]
    return f"{text} {after}"


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
    """One variant a run's statement can be, and what every run of that variant holds."""

    #: The text of the run's name, as the reassembly reads it: what
    #: :attr:`leaves` index.
    variant: str
    #: The run's own text with what some paths append cut out.
    text: str
    #: Its fragments; empty where the variant was not read fragment by fragment.
    leaves: list[_Leaf]


def _named_text(node: ast.AST) -> Optional[str]:
    """The name whose text *node* appends or assigns -- ``sql += where``, ``sql = where`` -- if it does."""
    if isinstance(node, (ast.AugAssign, ast.Assign)) and isinstance(node.value, ast.Name):
        return node.value.id
    return None


def _binding_calls(func: ast.FunctionDef | ast.AsyncFunctionDef, name: str) -> list[tuple[ast.Call, int]]:
    """``(call, slot)`` for each call *func* binds *name* from -- ``where, var_map = build_where_clause()`` is slot 0."""
    found: list[tuple[ast.Call, int]] = []
    for node in ast.walk(func):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)) or not isinstance(node.value, ast.Call):
            continue
        if enclosing_function(node) is not func:
            continue
        for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
            if isinstance(target, ast.Name) and target.id == name:
                found.append((node.value, -1))
            for slot, one in enumerate(getattr(target, "elts", [])):
                if isinstance(one, ast.Name) and one.id == name:
                    found.append((node.value, slot))
    return found


def _handed_paths(
    func: ast.FunctionDef | ast.AsyncFunctionDef, name: str
) -> list[tuple[Optional[ast.Call], tuple[_Hop, ...], ast.FunctionDef | ast.AsyncFunctionDef, ast.Name]]:
    """``(call, through, callee, value)`` for each name a call binding *name* in *func* hands back.

    :func:`sql.returned`, with the functions it passes through kept: ``return
    self.getCriteriaByAge(...)`` in ``getSortingCriteria`` is where the
    statement's ``ORDER BY`` is chosen -- under ``if random_number <=
    sloppy_ratio:`` -- and a path that skipped the forwarding function lost the
    test.  Only names: a literal returned is not a fragment anyone builds.
    """
    found: list = []

    def through(call: ast.Call, slot: int, at_func, at_call: ast.Call, passed: tuple[_Hop, ...], hops: int) -> None:
        placed = sql.placed(at_func, at_call)
        # As :func:`sql.returned`: a call placed open hands back what this
        # reading cannot name as well, so no path is read through it.
        if placed is None or placed.unplaced or hops >= sql.RETURN_HOPS:
            return
        for target in placed.targets:
            for value in sql._returns(target.func):
                if isinstance(value, ast.Call):
                    through(call, slot, target.func, value, (*passed, _Hop(target.func, value, value)), hops + 1)
                    continue
                if slot >= 0:
                    if not (isinstance(value, ast.Tuple) and len(value.elts) > slot):
                        continue
                    value = value.elts[slot]
                if isinstance(value, ast.Name):
                    found.append((call, passed, target.func, value))

    for call, slot in _binding_calls(func, name):
        through(call, slot, func, call, (), 0)
    return found


#: How many names deep a statement's text is cut -- a local built from
#: another, or a callee's return, each one more.
_CUT_DEPTH = 6


def _cut_name(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    name: str,
    text: str,
    at: ast.AST,
    hops: tuple[_Hop, ...] = (),
    depth: int = 0,
) -> Optional[tuple[list[tuple[int, int]], list[_Leaf]]]:
    """*text* -- one statement *name* holds in *func*, used at *at* -- read fragment by fragment.

    ``(cuts, leaves)``: the stretches of *text* some paths do not append, and
    its fragments, both by position in *text*; ``None`` where no way *func*
    builds *name* is *text* verbatim, which positions need.
    A fragment appended on some of the paths to *at* only is cut
    (:func:`_on_some_paths`).  A fragment that is itself a name's text is cut
    the same way, in that name's terms: ``sql += constituent_filter`` appends
    a local that is ``""`` unless ``skip_constituent_ids``, and ``sqlP +=
    sql_where_clause`` appends what ``construct_where_clause`` returns, six of
    whose terms are each appended under an ``if`` there.  Read as one piece,
    every one of them was in every run of ``getJobs``.
    """
    if depth > _CUT_DEPTH:
        return None
    here = (*hops, _Hop(func, at))
    for variant, spans in sql.variant_spans(func, name):
        if variant != text:
            continue
        cuts: list[tuple[int, int]] = []
        leaves: list[_Leaf] = []
        for span in spans:
            piece = variant[span.start : span.end]
            inner = _named_text(span.node)
            held = _cut_name(func, inner, piece, span.node, here, depth + 1) if inner else None
            if held is None:
                held = ([], [_Leaf(0, len(piece), span.node, here)])
            optional = _on_some_paths(span.node, at)
            leaves.extend(
                leaf._replace(start=leaf.start + span.start, end=leaf.end + span.start, kept=leaf.kept and not optional)
                for leaf in held[1]
            )
            cuts.extend(
                [(span.start, span.end)]
                if optional
                else [(start + span.start, end + span.start) for start, end in held[0]]
            )
        return sorted(cuts), sorted(leaves, key=lambda leaf: leaf.start)
    for call, passed, callee, value in _handed_paths(func, name):
        found = _cut_name(callee, value.id, text, value, (*hops, _Hop(func, at, call), *passed), depth + 1)
        if found is not None:
            return found
    return None


def _run_builds(func: ast.FunctionDef | ast.AsyncFunctionDef, run: sql.Execution) -> Optional[list[_Build]]:
    """Each variant that could have built *run*, read fragment by fragment; ``None`` where none can be told.

    The variants are the ones the run was built from (:attr:`sql.Execution.head`):
    one build each, so two branches of an ``if``/``else`` that append the same
    text are two, because what each fills the binds with is not the same.  A
    run with no name for its text, or whose name holds no such variant and
    nothing optional, is one build, itself, with no fragments.

    Kept on *func* per reading context (:func:`sql.reading_context`): every
    field read off the builds asks for them -- ``refuses`` and ``restricts``,
    ``conditions``, ``added_under`` -- at every call site handing constants
    as well, and reading them again was most of the read side's time.
    """
    key = (sql.reading_context(), run.variable, run.head, run.sql, id(run.call))
    held = getattr(func, "_run_builds_held", None)
    if held is None or held[0] != key[0][0]:
        held = (key[0][0], {})
        func._run_builds_held = held  # type: ignore[union-attr]
    if key not in held[1]:
        held[1][key] = _read_builds(func, run)
    return held[1][key]


def _read_builds(func: ast.FunctionDef | ast.AsyncFunctionDef, run: sql.Execution) -> Optional[list[_Build]]:
    """:func:`_run_builds`, read."""
    if not run.variable:
        return [_Build(run.sql, run.sql, [])]
    variants = sql.variant_spans(func, run.variable)
    matching = [one for one in variants if one[0] == run.head]
    if not matching:
        if any(_on_some_paths(span.node, run.call) for _text, spans in variants for span in spans):
            return None
        return [_Build(run.sql, run.sql, [])]
    builds: list[_Build] = []
    for text, _spans in matching:
        found = _cut_name(func, run.variable, text, run.call)
        if found is None:
            builds.append(_Build(text, run.sql, []))
            continue
        cut = _cut_text(text, found[0], run.sql)
        if cut is None:
            return None
        builds.append(_Build(text, cut, found[1]))
    return builds


def _every_run_texts(func: ast.FunctionDef | ast.AsyncFunctionDef, run: sql.Execution) -> Optional[list[str]]:
    """*run*'s statement with each fragment appended on some paths only cut out.

    ``[run.sql]`` itself where nothing is; one text per variant that could
    have built the run where more than one could; ``None`` where a fragment
    is optional and no variant is the run's -- a run whose text was rewritten
    past recognising.  What a name appended to the statement holds is cut
    too, through as many names as built it (:func:`_cut_name`).
    """
    builds = _run_builds(func, run)
    if builds is None:
        return None
    if len(builds) == 1 and builds[0].text == run.sql:
        return [run.sql]
    return [build.text for build in builds]


def _parameters(func: ast.AST) -> list[str]:
    args = func.args  # type: ignore[attr-defined]
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
    for node in ast.walk(func):
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
    for node in ast.walk(func):
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
        if any(isinstance(one, ast.Name) and one.id == name for target in targets for one in ast.walk(target)):
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
            names = sorted({one.id for one in ast.walk(expression) if isinstance(one, ast.Name)}) if expression is not None else []
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
                {bind.site.lineno for bind in sql.bound_values(func, run.varmap or "", key, window, at=run.run_at)}
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

    :func:`pathcond.compatible` read as the function runs, the question the
    reassembly asks of a fragment and the call it might reach: the two arms
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


def _words(text: str) -> list[str]:
    """The names, quoted values and numbers *text* is spelled with."""
    return re.findall(r"'[^']*'|[A-Za-z_][\w$#]*|\d+(?:\.\d+)?", text)


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
    """The terms every row of *run* meets, one tuple per way a row comes back in each of its builds.

    Read on every build (:func:`_every_run_texts`) and not on the run's text
    alone: a fragment appending ``OR ...`` takes rows back that the shorter
    text leaves out, and a term a fragment appends is met only where the
    fragment is.  The caller combines the ways -- a value refused is refused
    in every one, a subject restricted is restricted in every one, to any of
    their values.
    """
    whole = sqltext.outright(run.sql)
    # Before the builds are asked for: they are the expensive half, and a
    # text nothing decides has no build that does.
    if not any(whole):
        return whole
    floors = _every_run_texts(func, run)
    if floors is None:
        return ()
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


def _folded_conditions(
    func: ast.FunctionDef | ast.AsyncFunctionDef, line: int, variants: list
) -> list[str]:
    """What every one of a statement's *variants* requires, and what they require between them.

    :func:`_folded` without the trees.
    """
    return [entry for entry, _tree in _folded(func, line, variants)]


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

    **A variant is read with what it appends on some paths only cut out**
    (:func:`_every_run_texts`).  The reassembly keeps such a fragment in every
    variant, and a fragment in every variant folds as required:
    ``getDatasetsToFeedContents_JEDI`` appends ``tabT.lockedTime IS NULL OR
    tabT.lockedTime<:lockTimeLimit`` under ``if not force_read:``, and a run
    with ``force_read`` never asked it.  What is cut is not recorded at all,
    even where it travels with the value a reader asks by -- saying less is
    the direction this field can afford.  A run whose variant cannot be told
    is read as it is.

    Each entry comes with its tree where *tree_of* reads one off a run's
    :func:`sqltext.condition_twins` -- the values of the same entry in two
    runs merged, and the disjunction folded here an ``any`` of the ``all``
    of each variant's remainder.
    """
    rendered: list[list[tuple[str, Optional[ConditionNode]]]] = []
    for run in variants:
        window = run.window
        if window is None and run.varmap:
            window = sql.binding_window(func, run.varmap, line)
        whole = set(sqltext.conditions(run.sql))
        for text in _every_run_texts(func, run) or [run.sql]:
            # Only what the run itself requires: a cut inside a term leaves
            # one the statement never has -- ``computingSite IN ( )`` once the
            # loop filling the list is cut out.
            found = [
                (condition, twin)
                for condition, twin in zip(sqltext.conditions(text), sqltext.condition_twins(text))
                if condition in whole
            ]
            seeds: dict[str, list[int]] = {}
            for condition, _twin in found:
                for key in condition.binds:
                    if key in seeds:
                        continue
                    at_lines = sorted(
                        {
                            bind.site.lineno
                            for bind in sql.bound_values(func, run.varmap or "", key, window, at=run.run_at)
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
    if len(remainders) == 1:
        folded.extend((entry, trees[entry]) for entry in remainders[0])
    elif all(remainders):
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
            "items": [_merged_tree(a, b) for a, b in zip(one.items, other.items)],
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
    for node in ast.walk(func):
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
            for one in ast.walk(target):
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
    for node in ast.walk(func):
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
            names = {one.id for _kind, value in executed for one in ast.walk(value) if isinstance(one, ast.Name)}
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


def _joined_members(name: ast.Name, func: ast.AST, at: ast.AST) -> Optional[tuple[list[str], list[ast.AST]]]:
    """The templates a ``"<sep>".join(members)`` written into *name* repeats, and the loops appending them.

    ``leave_bindings = " ".join(tmp_list)``, ``tmp_list`` appended one
    ``WHEN gshare=:leave{i} THEN {i}`` per share in a ``for`` -- the text is
    as long as the loop runs, and only its template is in the source.
    """
    for value in reaching_definitions(func, name.id, at):
        if not (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == "join"
            and isinstance(value.func.value, ast.Constant)
            and len(value.args) == 1
            and isinstance(value.args[0], ast.Name)
        ):
            continue
        members = value.args[0].id
        templates: list[str] = []
        loops: list[ast.AST] = []
        for node in ast.walk(func):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "append"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == members
                and len(node.args) == 1
                and enclosing_function(node) is func
            ):
                continue
            text = values.rendered_text(node.args[0])
            if text and text not in templates:
                templates.append(text)
            loop = next((one for one in ancestors(node) if isinstance(one, (ast.For, ast.AsyncFor))), None)
            if loop is not None and loop not in loops:
                loops.append(loop)
        if templates:
            return templates, loops
    return None


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


#: A parameter a route fixes no constant for.
_UNSET = object()


def _route_constant(name: str, target: Target, sites: tuple, calls: Calls) -> object:
    """The constant the route *sites* hands *target*'s parameter *name*, or :data:`_UNSET`.

    Followed out through each forwarder that hands the parameter on
    unchanged; a call that does not pass it hands the default it states.  A
    value the route's outermost caller takes as its own parameter is not
    fixed: that caller's callers are not asked.
    """
    site = sites[0]
    handed = bind(site.call, target).get(name)
    if isinstance(handed, ast.Constant):
        return handed.value
    if (
        isinstance(handed, ast.Name)
        and name in passed_by(site.call, target)
        and len(sites) > 1
        and handed.id in _parameters(site.func)
    ):
        owned = calls.types.method_class(site.file, site.func)
        return _route_constant(handed.id, Target(site.file, site.func, owned[1] if owned else None), sites[1:], calls)
    return _UNSET


def _route_handed(target: Target, sites: tuple, calls: Calls) -> dict[str, "values.Stated"]:
    """What the route *sites* hands each of *target*'s parameters, read as an amount in the caller (:func:`values.stated`).

    A parameter the call does not pass holds the default the function
    states; one passed is read where the caller writes it, through the
    caller's own parameters as the rest of the route hands them.  A
    parameter the call hands with ``*args`` is left out: its value is the
    function's own parameter, ``needs``.
    """
    site = sites[0]
    passed = passed_by(site.call, target)
    outer: Optional[dict[str, values.Stated]] = None
    found: dict[str, values.Stated] = {}
    for name, expression in bind(site.call, target).items():
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
    for node in ast.walk(func):
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
                for inner in ast.walk(first)
            )
        if not mentions:
            continue
        attr = node.func.attr if isinstance(node.func, ast.Attribute) else (
            node.func.id if isinstance(node.func, ast.Name) else "?"
        )
        if attr in sql.RUNNERS:
            if len(node.args) < 2:
                return RUN_WITHOUT_VARMAP
            # Accepted form, yet the text did not match: the reassembly this
            # slice did and the one ``executions`` did disagree, which is a
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

    Read off the assignments rather than off the executions.  A name is a
    candidate when what it holds names a table under a read verb; the call
    that runs it is not consulted, which is what keeps this independent of the
    reading it is the denominator for.

    Maps each statement to a local that holds it, so that one left out of the
    numerator can be traced back to the variable -- which is what a reader needs
    in order to see how it is run.  **Keyed by the text**, exactly as the set it
    replaces was: two locals assembling the same statement were one candidate
    and stay one, because the denominator counts statements.  Keying by the pair
    instead would have moved ``sql-read``'s candidate count.
    """
    names = {
        target.id
        for node in ast.walk(func)
        for target in (
            [node.target] if isinstance(node, ast.AugAssign) else targets_of(node)
        )
        if isinstance(target, ast.Name)
    }
    return {
        text: name
        for name in sorted(names, reverse=True)
        for text in sql.variants(func, name)
        if any(verb == "read" for verb, _table in _table_verbs(text))
    }


#: Where a statement's rows arrive.  The ``execute`` form binds nothing at the
#: call -- ``self.cur.execute(sql, varMap)`` and then ``res =
#: self.cur.fetchall()`` -- so reading the call line alone finds the rows for
#: 48 of the corpus's 264 statements and misses every exit taken because none
#: came back.
_FETCH = ("fetchall", "fetchone", "fetchmany")
#: An exit that says nothing is a different answer from one that logs: the
#: reader looking for why a row stopped here has no line to grep for.
_SAYS_SOMETHING = ("debug", "info", "warning", "error", "critical", "exception")
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
    return any(isinstance(inner, _LEAVES) for inner in ast.walk(handler))


def _assigned_names(node: ast.Assign) -> set[str]:
    return {name for target in node.targets for name, _slot, _unpacked in names_bound_by(target)}


def _rows_written(
    func: ast.FunctionDef | ast.AsyncFunctionDef, line: int, last: int
) -> list[ast.Assign]:
    """The assignments this statement's rows land in, within the stretch it owns.

    The call line first, for the forwarded form that returns them; otherwise
    the first ``fetch*`` under it.  Reading only the call would leave most of
    the corpus with no rows to test an exit against -- see :data:`_FETCH`.
    """
    named: list[ast.Assign] = []
    fetched: list[ast.Assign] = []
    for node in ast.walk(func):
        if not isinstance(node, ast.Assign) or not _assigned_names(node):
            continue
        if node.lineno == line:
            named.append(node)
        elif line < node.lineno <= last and isinstance(node.value, ast.Call):
            callee = node.value.func
            if isinstance(callee, ast.Attribute) and callee.attr in _FETCH:
                fetched.append(node)
    return named or fetched


def _rows_of(
    func: ast.FunctionDef | ast.AsyncFunctionDef, line: int, last: int
) -> set[str]:
    """The names :func:`_rows_written` binds."""
    return {name for node in _rows_written(func, line, last) for name in _assigned_names(node)}


def _flag_parameters(func: ast.AST) -> set[str]:
    """The parameters *func* tests in an ``if`` and never rebinds: the ones a call's constant can decide."""
    args = func.args  # type: ignore[attr-defined]
    parameters = {a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)} - {"self", "cls"}
    tested = {
        name.id
        for node in ast.walk(func)
        if isinstance(node, ast.If) and enclosing_function(node) is func
        for name in ast.walk(node.test)
        if isinstance(name, ast.Name) and name.id in parameters
    }
    rebound = {
        node.id
        for node in ast.walk(func)
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del))
    }
    return tested - rebound


def _module_of(node: ast.AST) -> Optional[ast.Module]:
    while getattr(node, "parent", None) is not None:
        node = node.parent  # type: ignore[attr-defined]
    return node if isinstance(node, ast.Module) else None


def corpus_calls(modules: list[SourceModule], roots: Optional[dict[str, Path]]) -> Optional[Calls]:
    """The corpus's calls, read over the build's own trees; ``None`` without *roots*.

    The trees are the modules' so that a caller :meth:`Calls.callers` finds is
    a function the build has already parsed and attached parents to.
    """
    if not roots:
        return None
    walked = {}
    for module in modules:
        attach_parents(module.tree)
        walked[module.rel_path] = (module.blob_sha, module.tree)
    return Calls(Types(roots, walked))


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
        # when a pair actually lands: a run whose predicates name nothing the
        # map promoted is not somewhere to send a reader, so it gets no row.
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
            binds = sql.bound_values(func, run.varmap or "", key, run.window, across=True, at=run.run_at)
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
        binds = sql.bound_values(func, run.varmap or "", key, run.window, across=True, at=run.run_at)
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
        reading = _CallReading({table for verb, table in _table_verbs(run.sql) if verb == "read"})
        # The same terms the statement's own fields are filled from, read the
        # same way, or a call site would differ from the statement by the
        # reading alone.
        for term in sqltext.terms(run.sql).terms:
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
        binds = sql.bound_values(func, run.varmap or "", key, run.window, across=True, at=run.run_at)
        if len(binds) != 1:
            return None
        value = binds[0].value
        where = enclosing_function(value) or _module_of(value) or func
        if isinstance(value, ast.Name):
            written = [
                node
                for node in ast.walk(where)
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
            constants = {
                name: value for name in sorted(tested) if (value := _route_constant(name, target, route.sites, calls)) is not _UNSET
            }
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

        The fragment writing the order is found as ``added_under`` finds it,
        by the text it appends; the item's hole is the f-string's value at
        that place, and a value joined from members a loop appends is read
        as its template and the loop.  The template, written in once, says
        which columns the item ranks by; the members' order is the loop's,
        computed at run time.
        """

        def file_of(one: ast.AST) -> str:
            return (calls.types.file_of(one) if calls is not None else None) or file

        for run in executions:
            kept = statements.get((owner, run.call.lineno))
            if kept is None or not any("{}" in order for order in kept.order_by):
                continue
            found = list(kept.order_holes)
            for build in _run_builds(func, run) or []:
                for leaf in build.leaves:
                    piece = _squeezed(build.variant[leaf.start : leaf.end])
                    value = written_value(leaf.node)
                    if not isinstance(value, ast.JoinedStr):
                        continue
                    rendered = _squeezed(values.rendered_text(value) or "")
                    holes = [one.value for one in value.values if isinstance(one, ast.FormattedValue)]
                    where = leaf.hops[-1].func
                    for order in kept.order_by:
                        spelled = _squeezed(f"ORDER BY {order}")
                        if "{}" not in order or spelled not in piece or spelled not in rendered:
                            continue
                        at = rendered.count("{}", 0, rendered.find(spelled))
                        for index, item in enumerate(sqltext.order_items(order)):
                            count = item.expression.count("{}")
                            hole = holes[at] if item.blanked and at < len(holes) else None
                            at += count
                            if not isinstance(hole, ast.Name):
                                continue
                            joined = _joined_members(hole, where, leaf.node)
                            if joined is None:
                                continue
                            templates, loops = joined
                            # The columns the template adds to the order, written in once.
                            filled = item.expression.replace("{}", templates[-1], 1)
                            before = set(sqltext.order_owners(run.sql))
                            owners = sqltext.order_owners(run.sql.replace(item.expression, filled, 1))
                            subjects = sorted(
                                {_column_subject(attributor, table, column) for table, column in owners if (table, column) not in before}
                            )
                            hole_read = OrderHole(
                                order=order,
                                item=index,
                                subjects=subjects,
                                templates=templates,
                                computed_at="; ".join(_loop_said(loop, where, file_of(where)) for loop in loops),
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
        a disjunction -- :func:`_folded_conditions`, which a call site's
        reading folds by too.
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
        for run in variants:
            try:
                whole = sqltext.terms(run.sql).terms
            except Exception:  # noqa: BLE001 -- a text sqlglot will not scope is read as nothing
                whole = ()
            row_terms[id(run)] = [term for term in whole if term.verb == "read" and term.scope == sqltext.ROW]
            orders[id(run)] = sqltext.ordering(run.sql)[0]
            found = sqltext.conditions(run.sql)
            seeds = _condition_seeds(func, run, line, found)
            own_conditions[id(run)] = [(plain(one), tuple(dict.fromkeys(one.binds))) for one in found]
            for one, twin in zip(found, sqltext.condition_twins(run.sql)):
                spelled.setdefault(("conditions", plain(one)), _render_condition(one, seeds))
                key = ("conditions", plain(one))
                trees[key] = _merged_tree(trees.get(key), tree_of(func, run, file, twin))
            joined = sqltext.conjoined(run.sql)
            seeds = _condition_seeds(func, run, line, joined)
            conjuncts[id(run)] = [
                (_render_condition(one, seeds), plain(one), tuple(dict.fromkeys(one.binds))) for one in joined
            ]

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

        def carried(
            run, piece: str, along: Optional[dict[int, list[ast.AST]]]
        ) -> dict[tuple[str, str], list[tuple[str, Optional[bool]]]]:
            """``{(field, term): [(the term it enters in, whether that decides it)]}`` for what a fragment reading *piece* appends."""
            found: dict[tuple[str, str], list[tuple[str, Optional[bool]]]] = {}
            for order in orders[id(run)]:
                if _squeezed(f"ORDER BY {order}") in _squeezed(piece):
                    found.setdefault(("order_by", order), []).append((order, None))
            for text, binds in own_conditions[id(run)]:
                if _names(piece, binds, () if binds else _words(text)):
                    found.setdefault(("conditions", text), []).append((spelled[("conditions", text)], None))
            for term in row_terms[id(run)]:
                for pair, binds, words in pairs(run, term, along):
                    if not _names(piece, binds, words):
                        continue
                    # The term of this run's ``WHERE`` that holds it -- an ``OR``
                    # with another column whole -- and nothing where none does:
                    # the fragment's own text can hold other terms' ``OR``\ s.
                    within = next(
                        (
                            spelling
                            for spelling, text, holding in conjuncts[id(run)]
                            if (set(binds) <= set(holding) if binds else _names(text, (), words))
                        ),
                        "",
                    )
                    found.setdefault(pair, []).append((within, term.decided))
            return found

        read = [
            (run, build, holds(run, build))
            for run in variants
            for build in _run_builds(func, run) or [_Build(run.sql, run.sql, [])]
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
        places: dict[tuple[str, str], dict[tuple[str, str], TermPlace]] = {key: {} for key in missing}
        for run, build, _held in read:
            kept_nodes = [node for leaf in build.leaves if leaf.kept for node in _trail(leaf)]
            for leaf in build.leaves:
                piece = build.variant[leaf.start : leaf.end]
                for key, terms in carried(run, piece, _along([*kept_nodes, *_trail(leaf)])).items():
                    for term, decides in terms if key in places else ():
                        place = _place(leaf, term, file_of, decides)
                        held = places[key].setdefault((place.fragment, place.term), place)
                        # Two terms of one place naming the pair: either one ruling it out is enough.
                        if decides and not held.decides:
                            held.decides = True

        def argued(text: str, term: str) -> list[TermPlace]:
            """Where a value the ``execute`` formats into the statement is written as *text*, with the tests that choose it.

            ``sql_read_files.format(order_by_policy, ...)``: the order is no
            fragment of the statement, it is ``order_by_policy``, and each
            of its writes under an ``if`` -- ``"lumiBlockNr,lfn"``,
            ``"boundaryID"`` -- is a build of its own.
            """
            found: dict[tuple[str, str], TermPlace] = {}
            for run in variants:
                if not run.call.args:
                    continue
                execute = _statement_of(run.call)
                for node in ast.walk(run.call.args[0]):
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
            return any(
                key in carried(run, build.variant[leaf.start : leaf.end], None)
                for run, build, _held in read
                for leaf in build.leaves
            )

        added: list[AddedTerm] = []
        for key in missing:
            field, said = key
            term = spelled.get(key, said)
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
                            + ", ".join(sorted({fragment for fragment, _term in places[key]}))
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
        """Give each of *func*'s statements the stretch of it that it owns.

        From where this run's binds start being filled to just before the next
        run's do.  Every run of the function is laid out, not only the stored
        ones: the statement that ends a region may be one no read site was
        built from, and stopping at the next *stored* one would hand a reader
        the following query's logging as well.

        Where two statements share one ``var_map = {}`` the later one's binds
        start filling *above* the earlier one's call, so the next run's window
        is no edge at all: it would end a region before the statement it
        belongs to ran, taking that statement's own ``fetchall`` with it.  A
        statement's territory cannot close at or before it ran, so where the
        windows overlap the next *call* is the edge.

        Keyed by the call line because a branchy statement is emitted as
        several runs at one ``execute``, and a read site is one ``(owner,
        line)``: without the key the region ends at the statement's own second
        variant, which is itself.
        """
        starts: dict[int, int] = {}
        for run in executions:
            line = run.call.lineno
            window = run.window
            if window is None and run.varmap:
                window = sql.binding_window(func, run.varmap, line)
            begins = window[0] if window else line
            starts[line] = min(starts.get(line, begins), begins)
        last = func.end_lineno or func.lineno
        ordered = sorted(starts.items())
        for index, (line, begins) in enumerate(ordered):
            kept = statements.get((owner, line))
            if kept is None:
                continue
            if index + 1 < len(ordered):
                next_line, next_begins = ordered[index + 1]
                following = next_begins if next_begins > line else next_line
            else:
                following = last + 1
            statements[(owner, line)] = kept._replace(
                region=(begins, max(line, following - 1))
            )

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
            names = _rows_of(func, line, last)
            enclosing = _enclosing_loops(func)
            spans: list[tuple[int, int]] = []
            over_rows: set[int] = set()
            for node in ast.walk(func):
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
                over = {n.id for n in ast.walk(header) if isinstance(n, ast.Name)}
                if over & names:
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
            for node in ast.walk(func):
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
            for node in ast.walk(func):
                if not isinstance(node, ast.If) or not line < node.lineno <= last:
                    continue
                tested = {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
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
            for run in executions:
                if (run.sql, run.window) in seen:
                    continue
                seen.add((run.sql, run.window))
                together: set[str] = set()
                for verb, table in _rows_touched(run.sql):
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
                reading = sqltext.terms(run.sql)
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
                    unresolved=tuple(
                        sorted(
                            {f"{stop.source} -- {stop.reason}" for stop in run.unresolved}
                            | set(reading.unresolved)
                        )
                    ),
                )
                read_tables = {table for verb, table in _table_verbs(run.sql) if verb == "read"}
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
                kept = statements.get((owner, run.call.lineno))
                if kept is not None and kept is not at and at.unresolved:
                    kept = kept._replace(
                        unresolved=tuple(sorted(set(kept.unresolved) | set(at.unresolved)))
                    )
                    statements[(owner, run.call.lineno)] = kept
                if kept is not None:
                    kept.tables.update(read_tables)
                    orders, limits = sqltext.ordering(run.sql)
                    kept.order_by.update(orders)
                    for bound_to, strict in limits:
                        values_of = (
                            bound(func, run, bound_to, module.rel_path)
                            if bound_to.startswith(":")
                            else [bound_to]
                        )
                        kept.limits.update(
                            f"{'<' if strict else '<='} {value}" for value in values_of or [bound_to]
                        )
            place_conditions(func, owner, module.rel_path, executions)
            place_prints(func, owner, executions)
            place_orders(func, owner, module.rel_path, executions)
            place_added_under(func, owner, module.rel_path, executions)
            # Regions first: the exits are read inside the stretch it settles.
            place_regions(func, owner, executions)
            place_exits(func, owner)
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
    """

    stored: int
    handing_back: int
    resolved: int


#: How a row leaves the name the statement first bound it to.  Assignment and
#: a loop target are the obvious two; the rest are how the corpus writes "put
#: this row in the bucket it belongs to" before handing the bucket back.
_CARRIERS = ("append", "add", "update", "extend", "setdefault")


def _bound_by(node: ast.AST) -> set[str]:
    """Names an assignment binds, counting ``d[key] = row`` as binding ``d``.

    :func:`_assigned_names` deliberately does not: there it is asked which
    names hold *this statement's rows*, and a dict the rows are filed into
    holds other statements' rows too.  Here the question is the looser one --
    could a row reach the return -- and ``retTaskIDs[jediTaskID] = {...}`` is
    exactly how ``getTasksToExecCommand_JEDI`` builds what it returns.
    """
    targets: list[ast.expr]
    if isinstance(node, ast.Assign):
        targets = list(node.targets)
    elif isinstance(node, ast.AugAssign) or (isinstance(node, ast.AnnAssign) and node.value is not None):
        targets = [node.target]
    else:
        return set()
    found: set[str] = set()
    for target in targets:
        found.update(name for name, _slot, _unpacked in names_bound_by(target))
        # ``retMap[site][status] += cnt`` files rows two keys down.
        found.update(containers_changed(target))
    return found


def _keys_filed_under(target: ast.AST) -> list[ast.AST]:
    """The keys of ``d[k] = v`` and ``d[k][j] = v`` -- a dict rooted at a name."""
    if isinstance(target, (ast.Tuple, ast.List)):
        return [key for element in target.elts for key in _keys_filed_under(element)]
    keys: list[ast.AST] = []
    while isinstance(target, ast.Subscript):
        keys.append(target.slice)
        target = target.value
    return keys if isinstance(target, ast.Name) else []


class _Carry:
    """Which writes in one function hold the rows some of its statements wrote.

    The writes followed are the ones that move rows on: an assignment
    (:func:`_bound_by`, so ``d[k] = row`` fills ``d``), a loop target, and a
    :data:`_CARRIERS` call on a name.  A write holds the rows when what it
    takes from -- the value, the loop's iterable, the call's arguments, and a
    key it files under -- reads a name one of whose writes holding them
    reaches it.  Which writes reach is ``pathcond.surviving_writes`` as the
    function runs: the walk's rule.  Every other binding of a name
    (``with ... as``, ``except ... as``, ``del``) overwrites and holds nothing;
    ``d[k] = v``, ``x += v`` and ``n.append(v)`` add and overwrite nothing.

    **It replaced a set of names.**  Any write anywhere in the function that
    mentioned a held name added its targets, whatever the order, which made
    the ``return`` generous and the caller's leads wrong.  Of ce1's 295
    crossings, 25 named somewhere these rows never go -- 8 through a name
    overwritten before the use, 8 through a link written before the rows
    arrived (``AtlasAnalJobBroker.doBrokerage``'s ``setErrDiag`` at 496, from
    ``failureCounts`` read at 388), 9 through rows only ever used to choose or
    log, reaching the ``return`` through a loop name used again further down.
    A caller was asked whether the arriving write still held at the use with
    every seed name's writes at once, so ``ret_val`` rebound by the next call
    dropped the rows the first had put in ``analy_sites_class_dict``; carried
    from the arriving write, each name keeps its own.

    The key is new: ``retMap[jediTaskID] = set()`` is how
    ``throttleJobsInPausedTasks_JEDI`` hands its tasks back, and the caller
    loops over ``.items()``.  The names never saw it -- 6 crossings over 3
    statements.

    Not followed, here or before: rows put into an object by a method and the
    object into an element of what is returned --
    ``_get_datasets_with_unprocessed_inputs`` packs ``tmp_res`` into
    ``dataset_spec`` and adds it to an ``input_chunk`` of the list it returns.
    The names reached that ``return`` only because ``tmp_res`` was used again;
    following every method call would carry rows into ``tmp_log``.
    """

    def __init__(self, func: ast.AST, start: Iterable[ast.AST]):
        self._writes: dict[str, dict[int, ast.AST]] = {}
        self._adds: dict[str, dict[int, ast.AST]] = {}
        self._takes: list[tuple[ast.AST, list[ast.AST]]] = []
        self._read(func)
        self._holding = {id(node) for node in start}
        self._grow()

    def _write(self, name: str, node: ast.AST, adds: bool) -> None:
        self._writes.setdefault(name, {})[id(node)] = node
        if adds:
            self._adds.setdefault(name, {})[id(node)] = node

    def _read(self, func: ast.AST) -> None:
        # A comprehension's target is its own scope's, not the function's.
        inner = {
            id(name)
            for node in ast.walk(func)
            if isinstance(node, ast.comprehension)
            for name in ast.walk(node.target)
        }
        for node in ast.walk(func):
            if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                if id(node) not in inner:
                    statement = site_of(node)
                    self._write(node.id, statement, isinstance(statement, ast.AugAssign))
            elif isinstance(node, ast.ExceptHandler) and node.name:
                self._write(node.name, node, False)
            elif isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    for name in containers_changed(target):
                        self._write(name, node, True)
                if node.value is not None and _bound_by(node):
                    keys = [key for target in targets for key in _keys_filed_under(target)]
                    self._takes.append((node, [node.value, *keys]))
            elif isinstance(node, (ast.For, ast.AsyncFor)):
                self._takes.append((node, [node.iter]))
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in _CARRIERS
                and isinstance(node.func.value, ast.Name)
            ):
                statement = site_of(node)
                self._write(node.func.value.id, statement, True)
                self._takes.append((statement, list(node.args)))

    def _holds(self, name: str, statement: ast.AST) -> bool:
        writes = self._writes.get(name)
        if not writes or not any(key in self._holding for key in writes):
            return False
        reached = surviving_writes(
            list(writes.values()),
            statement,
            list(self._adds.get(name, {}).values()),
            as_run=True,
        )
        return any(id(write) in self._holding for write in reached)

    def reaches(self, expression: ast.AST, at: ast.AST) -> bool:
        """Whether a name *expression* reads holds the rows where *at* runs."""
        statement = at if isinstance(at, (ast.stmt, ast.ExceptHandler)) else site_of(at)
        names = {node.id for node in ast.walk(expression) if isinstance(node, ast.Name)}
        return any(self._holds(name, statement) for name in names)

    def _grow(self) -> None:
        growing = True
        while growing:
            growing = False
            for statement, sources in self._takes:
                if id(statement) in self._holding:
                    continue
                if any(self.reaches(source, statement) for source in sources):
                    self._holding.add(id(statement))
                    growing = True


def _hands_rows_back(func: ast.AST, carry: _Carry) -> bool:
    """Whether a ``return`` in *func* returns something *carry* holds there."""
    return any(
        isinstance(node, ast.Return) and node.value is not None and carry.reaches(node.value, node)
        for node in ast.walk(func)
    )


def _enclosing(node: ast.AST, kinds: tuple) -> Optional[ast.AST]:
    while node is not None and not isinstance(node, kinds):
        node = getattr(node, "parent", None)
    return node


def _consumers_of(
    name: str,
    calls: dict[str, list[tuple[str, ast.Call, ast.AST]]],
    where: str,
    definition: ast.AST,
) -> list[tuple[str, ast.Call, ast.AST]]:
    """The call sites of *name* that are somewhere an investigation would go.

    Two kinds are dropped.  The function calling itself, which is not a
    crossing.  And the proxy layer: ``JediTaskBuffer.getTasksToExecCommand_JEDI``
    consists of ``return proxy.getTasksToExecCommand_JEDI(...)``, so naming it
    as where the rows went would stop the reader one hop short of the code
    that does anything with them.  A caller whose own function has the same
    name and hands the call straight back is that layer, by construction.

    It is what makes the rest answerable rather than a refinement: 12 of the
    corpus's statements had exactly one call site before this, 52 after.
    """
    found = []
    for rel, call, caller in calls.get(name, []):
        if caller is None or (rel == where and caller is definition):
            continue
        if caller.name == name and isinstance(
            _enclosing(call, (ast.stmt,)), ast.Return
        ):
            continue
        found.append((rel, call, caller))
    return found


def _reached_by(
    caller: ast.AST,
    arrived: ast.stmt,
    by_func: dict[str, set[str]],
    by_class: dict[str, set[str]],
) -> set[str]:
    """The junctions the rows that *arrived* reach in *caller*, by the two rules already used.

    The same pair :func:`attach_crossings` applies inside a region -- a call
    the map owns a junction for, and a class built here and started as a
    thread -- with one thing added that the region form does not need.  There
    the stretch is the statement's own and everything in it is about these
    rows.  Here it is somebody else's whole function, so the call has to take
    something that holds these rows where it runs -- carried from the line
    they arrived on, see :class:`_Carry`.

    That is not caution.  ``datasetManager.py::main`` runs seven queries into
    the same ``res`` and hands each to a different thread class, so carried by
    name one statement's rows resolve to all seven constructions -- the reuse
    P1-54 met on the exits, met again a function further out.  At the
    construction on line 906 the ``res`` in scope is the one from line 896,
    and the six others were overwritten or had not run.
    """
    carry = _Carry(caller, [arrived])
    reached: set[str] = set()
    built: dict[str, ast.AST] = {}
    for node in ast.walk(caller):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        callee = node.value.func
        made = callee.id if isinstance(callee, ast.Name) else getattr(callee, "attr", "")
        if made in by_class and any(carry.reaches(a, node) for a in node.value.args):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    built[target.id] = made
    for node in ast.walk(caller):
        if not isinstance(node, ast.Call):
            continue
        callee = node.func
        if isinstance(callee, ast.Attribute):
            if (
                callee.attr == "start"
                and isinstance(callee.value, ast.Name)
                and callee.value.id in built
            ):
                reached |= by_class[built[callee.value.id]]
                continue
            called = callee.attr
        elif isinstance(callee, ast.Name):
            called = callee.id
        else:
            continue
        if any(carry.reaches(a, node) for a in node.args):
            reached |= by_func.get(called, set())
    return reached


def attach_crossings(
    read_sites: list[ReadSiteNode],
    modules: list[SourceModule],
    junctions: list[JunctionNode],
) -> int:
    """Say which junction a row handled here is carried on to.

    The request's sixth point: a reader wants to be able to pick a *statement*
    as the lead and be told where the investigation goes next.  A statement
    already says which rows it asked for and which of them it drops; what it
    could not say is that the ones it keeps are handed to something the map
    already holds an answer for.

    Two shapes resolve by name and both are stored:

    * the region calls a function the map owns a junction in -- 65 of them;
    * the region builds a class the map owns a junction in and starts it as a
      thread -- 6, and one of those is the statement this whole slice is
      measured on.  ``copyArchive.py::main``'s ``holding`` query hands each
      job to ``Watcher(...)`` and calls ``.start()``, so the next thing that
      happens to a held job is ``Watcher.run``, which is a junction.  The
      ``start`` in ``thr.start()`` is not itself a junction owner, which is
      why the class the name was built from has to be carried.

    * the rows are handed back to the caller and the caller does one of those
      two things with them -- 51 more, and the flagship is one.

    **The third one was sized against the wrong shape.**  It used to be read
    off "a ``return`` inside the region", which is 139 returns over 73
    statements and does not include ``getTasksToExecCommand_JEDI@4591`` at
    all: that statement owns 4573-4603, works its rows through 4600-4621, and
    the function returns on 4647, the rows having travelled ``resList`` ->
    ``retTaskIDs`` -> ``retTaskList`` on the way.  Followed as the function
    runs instead -- see :class:`_Carry` -- 240 statements hand their rows back.

    **A crossing is only stored where all three of these hold**, and 174 of
    those 240 fail the last one and keep an empty ``crosses_to``:

    1. the rows reach a ``return``;
    2. the owner has a call site that is not the proxy layer -- see
       :func:`_consumers_of`;
    3. at that call site the returned value is bound to a name from which the
       caller does something the map holds a junction for.

    Naming the caller's function and stopping there would satisfy 2 and not 3,
    and a lead that does not name a junction is a lead pointing at whoever
    happened to be grepped first.  Where several call sites resolve, all of
    them are stored: the map's junctions are indexed by bare name throughout
    this function, so one statement reaching five is the same ambiguity the
    other two shapes already carry, not a new one.

    Run after promotion, not merely after the merge.  A crossing is a lead,
    and a lead has to name somewhere the map will answer for -- resolved
    against the candidate junctions instead, every ``self.cur.execute(...)``
    in a region resolved to the ``execute`` of an adder plugin that promotion
    then dropped: 520 crossings over 239 statements where 71 over 38 are real,
    with 176 statements handed the same two leads to nowhere.

    Returns the census: how many crossings were stored, how many statements
    hand their rows back at all, and how many of those named a junction.  The
    gap between the last two is the honest size of what is still unresolved,
    and it is reported rather than silently left as an empty field.
    """
    owners = {junction.owner for junction in junctions}
    calls: dict[str, list[tuple[str, ast.Call, ast.AST]]] = {}
    by_func: dict[str, set[str]] = {}
    for owner in owners:
        name = owner.rsplit("::", 1)[-1]
        # ``self.cur.execute(sql, varMap)`` names the database, not a function
        # of this corpus, so a junction living in a *method* called ``execute``
        # must not answer for it.  The adder plugins have one, and without this
        # every statement in the corpus crosses to both of them: 222 read sites
        # handed the same two false leads, which is the shape the crossing pass
        # was written to avoid and had been avoiding by accident, because until
        # ``JobSpec.ddmErrorDiag`` was promoted no ``execute`` owned a junction.
        if name in sql.RUNNERS:
            continue
        by_func.setdefault(name, set()).add(owner)
    trees = {module.rel_path: module.tree for module in modules}
    by_class: dict[str, set[str]] = {}
    for rel, tree in trees.items():
        # The call index the return shape resolves against, and the parent
        # links it walks upwards through.  Built once over the whole corpus
        # because the question is the reverse of the other two shapes': not
        # what this function calls, but who calls it.
        attach_parents(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                callee = node.func
                called = (
                    callee.attr
                    if isinstance(callee, ast.Attribute)
                    else (callee.id if isinstance(callee, ast.Name) else None)
                )
                if called:
                    calls.setdefault(called, []).append(
                        (
                            rel,
                            node,
                            _enclosing(node, (ast.FunctionDef, ast.AsyncFunctionDef)),
                        )
                    )
            if not isinstance(node, ast.ClassDef):
                continue
            for item in node.body:
                if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                name = f"{rel}::{item.name}"
                if name in owners:
                    by_class.setdefault(node.name, set()).add(name)

    stored = 0
    handing_back = 0
    resolved = 0
    for site in read_sites:
        if not site.region:
            continue
        tree = trees.get(site.anchor.file)
        if tree is None:
            continue
        wanted = site.owner.rsplit("::", 1)[-1]
        func = next(
            (
                node
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == wanted
                and node.lineno <= site.anchor.line_start <= (node.end_lineno or node.lineno)
            ),
            None,
        )
        if func is None:
            continue
        first, last = site.region
        reached: set[str] = set()
        # The name a thread was built from, so ``.start()`` can be resolved.
        built: dict[str, str] = {}
        for node in ast.walk(func):
            if not isinstance(node, ast.stmt) or not first <= node.lineno <= last:
                continue
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                callee = node.value.func
                made = callee.id if isinstance(callee, ast.Name) else getattr(callee, "attr", "")
                if made in by_class:
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            built[target.id] = made
            for inner in ast.walk(node):
                if not isinstance(inner, ast.Call):
                    continue
                if isinstance(inner.func, ast.Attribute):
                    receiver = inner.func.value
                    if (
                        inner.func.attr == "start"
                        and isinstance(receiver, ast.Name)
                        and receiver.id in built
                    ):
                        reached |= by_class[built[receiver.id]]
                        continue
                    called = inner.func.attr
                elif isinstance(inner.func, ast.Name):
                    called = inner.func.id
                else:
                    continue
                reached |= by_func.get(called, set())
        back, hands = _handed_back(site, func, wanted, calls, by_func, by_class)
        # Not itself: a statement is not somewhere its own rows go next.  Done
        # to the return shape's answer before it is counted as well as to the
        # whole, or the census reports two statements as resolved whose only
        # lead was back to themselves.
        back.discard(site.owner)
        reached |= back
        handing_back += 1 if hands else 0
        resolved += 1 if back else 0
        reached.discard(site.owner)
        site.crosses_to = sorted(reached)
        stored += len(site.crosses_to)
    return CrossingCensus(stored, handing_back, resolved)


def _handed_back(
    site: ReadSiteNode,
    func: ast.AST,
    wanted: str,
    calls: dict[str, list[tuple[str, ast.Call, ast.AST]]],
    by_func: dict[str, set[str]],
    by_class: dict[str, set[str]],
) -> tuple[set[str], bool]:
    """The junctions reached by giving these rows back, and whether it is given.

    The third shape of :func:`attach_crossings`, kept apart because it is the
    only one that leaves the statement's own file.  Nothing here is new
    machinery: the rows are followed to the ``return`` by name, the callers
    are read off the corpus-wide call index with the proxy layer removed, and
    what happens to them there is decided by the same two rules the region
    scan uses.
    """
    first, last = site.region
    handled = site.rows_handled
    through = max(last, handled[1] if len(handled) >= 2 else last)
    rows = _rows_written(func, site.anchor.line_start, through)
    if not rows or not _hands_rows_back(func, _Carry(func, rows)):
        return set(), False
    reached: set[str] = set()
    for _rel, call, caller in _consumers_of(wanted, calls, site.anchor.file, func):
        statement = _enclosing(call, (ast.stmt,))
        if not isinstance(statement, ast.Assign):
            # Returned on again, or thrown away.  The first is the proxy shape
            # under another name and the second has no rows to follow.
            continue
        if _bound_by(statement):
            reached |= _reached_by(caller, statement, by_func, by_class)
    return reached, True


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
                    for t in sqltext.joins(run.sql)
                    if t.lower() in never_written
                }
                if not gates:
                    continue
                for term in sqltext.terms(run.sql).terms:
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
