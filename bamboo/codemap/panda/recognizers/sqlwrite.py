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
import re
from pathlib import Path
from typing import NamedTuple, Optional

from bamboo.codemap.calls import Calls, Target, Types
from bamboo.codemap.models import (
    Anchor,
    Branch,
    CoverageStat,
    DiagnosticTemplate,
    EntityNode,
    JunctionNode,
    ReadingStop,
    ReadSiteNode,
    RowKeySource,
    SourceModule,
    SubjectNode,
)
from bamboo.codemap.panda import sql, sqltext, values
from bamboo.codemap.panda.attribution import SpecAttributor
from bamboo.codemap.panda.pathcond import (
    attach_parents,
    enclosing_function,
    functions_with_owner,
    handed_in,
    literal_values,
    path_condition,
    site_of,
    surviving_writes,
    targets_of,
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
            for filled in sql.bound_values(func, run.varmap, bind, run.window):
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
                            term, func, run.varmap, run.window, settle
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
                                    term, func, run.varmap, run.window, settle, beside=node
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
) -> list[tuple[str, int, ast.stmt, list[str], ast.stmt]]:
    """Return ``(outcome, tier, node, extra conditions, decided at)`` for a column.

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
        scan = sql.scan_binds(func, run.varmap, supplied.text, run.window)
        found = []
        for bind in scan.binds:
            value, site = bind.value, bind.site
            settled = settle(value, func)
            if settled:
                found.extend((outcome, 1, site, [], site) for outcome in settled)
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
                        (
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
            found.append((f"runtime({ast.unparse(value)})", 2, site, [], site))
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
        return [(supplied.text, 1, node, [], node)]
    if supplied.kind == "expression":
        # ``stateChangeTime=CURRENT_DATE``, ``nFiles=nFiles+:iFiles``: the
        # database decides, from the clock or from the row's own prior
        # contents.  Tier 2 for the reason a bind filled at run time is --
        # the writer is known and that is what localize and prune read -- and
        # dropping it instead was not neutral.  Promotion weighs a subject's
        # literal writes against all of them, so a column the source only ever
        # increments looked, from its one ``= 0``, like a field with a closed
        # vocabulary of one.
        return [(f"runtime({supplied.text})", 2, node, [], node)]

    # A copied column.  The source is named as a subject rather than as a bare
    # column so the edge joins: ``passthrough(JediTaskSpec.oldStatus)`` points
    # at a node the backward walk can continue from, ``passthrough(oldStatus)``
    # at a string.
    source, attribute, _kind = _subject_of(
        attributor, spec_class, statement.table, supplied.text
    )
    outcome = f"passthrough({SubjectNode.make_name(source, attribute)})"
    return [(outcome, 2, node, [], node)]


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
    outcomes: list[tuple[str, int, ast.stmt, list[str], ast.stmt]],
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
                line_start=min(node.lineno for _o, _t, node, _e, _d in outcomes),
                line_end=max(
                    node.end_lineno or node.lineno for _o, _t, node, _e, _d in outcomes
                ),
                blob_sha=module.blob_sha,
            ),
        )
        junctions[name] = junction

    known = {(branch.outcome, tuple(branch.path_condition)) for branch in junction.branches}
    for outcome, tier, node, extra, decided_at in outcomes:
        condition = path_condition(node) + extra
        if (outcome, tuple(condition)) in known:
            continue
        known.add((outcome, tuple(condition)))
        # Read at the deciding statement rather than at the bind.  The two are
        # the same for a literal and far apart for a local: ``retryTask_JEDI``
        # picks the status and writes the reason in one block and binds it
        # after the whole chain has closed, so the bind's block says nothing
        # about which of five refusals this is.
        tags, messages = progress.recorded_signature(decided_at, func, setters)
        junction.branches.append(
            Branch(
                outcome=outcome,
                path_condition=condition,
                row_precondition=beside(node) if beside is not None else row_precondition,
                order=len(junction.branches),
                tier=tier,
                tags=tags,
                messages=messages,
                line=decided_at.lineno,
            )
        )



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
    #: ``"<call> <- <call> (flag=True): reads a, b"`` for each call site whose
    #: constants narrow what the statement reads -- see ``place_call_sites``.
    call_sites: tuple[str, ...] = ()


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
    found: set[str] = set()
    for target in node.targets:
        parts = target.elts if isinstance(target, (ast.Tuple, ast.List)) else [target]
        found |= {part.id for part in parts if isinstance(part, ast.Name)}
    return found


def _rows_of(
    func: ast.FunctionDef | ast.AsyncFunctionDef, line: int, last: int
) -> set[str]:
    """The names this statement's rows land in, within the stretch it owns.

    The call line first, for the forwarded form that returns them; otherwise
    the first ``fetch*`` under it.  Reading only the call would leave most of
    the corpus with no rows to test an exit against -- see :data:`_FETCH`.
    """
    names: set[str] = set()
    fetched: set[str] = set()
    for node in ast.walk(func):
        if not isinstance(node, ast.Assign):
            continue
        if node.lineno == line:
            names |= _assigned_names(node)
        elif line < node.lineno <= last and isinstance(node.value, ast.Call):
            callee = node.value.func
            if isinstance(callee, ast.Attribute) and callee.attr in _FETCH:
                fetched |= _assigned_names(node)
    return names or fetched


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
    mechanism for one site.  Over-crediting is the direction this function
    already takes -- see :func:`_tables_of` -- and it costs nothing measurable
    here: ``dummy`` was not among the values reported as written-but-unselected
    for this subject, so nothing is hidden by it.
    """
    settle = values.handed(values.resolver(), calls)
    found: dict[str, dict[str, ValueUse]] = {}
    rows: dict[str, EntityUse] = {}
    unreadable: dict[str, int] = {}
    joins: dict[str, set[frozenset[str]]] = {}

    statements: dict[tuple[str, int], ReadStatement] = {}

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

    def bound(func, run, key: str, file: str) -> list[str]:
        values_of: list[str] = []
        binds = sql.bound_values(func, run.varmap or "", key, run.window, across=True)
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
            values_of.extend(one.value for one in settle(bind.value, where, at_file))
        return values_of

    def reading_of(func, run, file: str) -> tuple[set[str], set[tuple[str, str]]]:
        """The tables one run reads and the pairs it selects by -- what a call site can narrow."""
        tables: set[str] = set()
        pairs: set[tuple[str, str]] = set()
        for verb, table in _table_verbs(run.sql):
            if verb != "read":
                continue
            tables.add(table)
            spec_class = attributor.class_for_table(table)
            for column, key in sqltext.predicates(run.sql):
                qualifier, attribute, _kind = _subject_of(attributor, spec_class, table, column)
                subject = SubjectNode.make_name(qualifier, attribute)
                pairs.update((subject, value) for value in bound(func, run, key, file))
            for column, value in sqltext.selected_literals(run.sql):
                qualifier, attribute, _kind = _subject_of(attributor, spec_class, table, column)
                pairs.add((SubjectNode.make_name(qualifier, attribute), value))
        return tables, pairs

    def place_call_sites(func, owner: str, file: str, executions: list) -> None:
        """Say, per call site that narrows a statement, what it reads there.

        The statement's own reading is every run any call can make.  A call
        handing a flag parameter a constant runs fewer: ``TaskCommando.py:143``
        calls ``getPandaIDsWithTask_JEDI(jediTaskID, True)``, and under
        ``onlyActive`` the loop reads two tables, not four.  Asked only for a
        parameter the function tests and never rebinds, and recorded only
        where the call's reading differs from the statement's.
        """
        if calls is None or not any((owner, run.call.lineno) in statements for run in executions):
            return
        tested = _flag_parameters(func)
        if not tested:
            return
        owned = calls.types.method_class(file, func)
        target = Target(file, func, owned[1] if owned else None)
        for via, handed in calls.handed_constants(target, tested):
            per_line: dict[int, tuple[set[str], set[tuple[str, str]]]] = {}
            with sql.at_call(func, handed):
                for run in sql.executions(func):
                    tables, pairs = reading_of(func, run, file)
                    held = per_line.setdefault(run.call.lineno, (set(), set()))
                    held[0].update(tables)
                    held[1].update(pairs)
            said = ", ".join(f"{name}={value!r}" for name, value in sorted(handed.items()))
            route = " <- ".join(reversed(via))
            for (own, line), kept in list(statements.items()):
                if own != owner:
                    continue
                tables, pairs = per_line.get(line, (set(), set()))
                if line not in per_line:
                    entry = f"{route} ({said}): not run"
                elif tables != kept.tables or pairs != kept.selects:
                    # What differs from the statement's own reading, not the
                    # whole reading again.
                    said_here: list[str] = []
                    if tables != kept.tables:
                        said_here.append("reads " + (", ".join(sorted(tables)) or "no table"))
                    for label, held in (
                        ("does not select", sorted(kept.selects - pairs)),
                        ("also selects", sorted(pairs - kept.selects)),
                    ):
                        if held:
                            shown = ", ".join(f"{subject}={value}" for subject, value in held[:6])
                            more = f" and {len(held) - 6} more" if len(held) > 6 else ""
                            said_here.append(f"{label} {shown}{more}")
                    entry = f"{route} ({said}): " + "; ".join(said_here)
                else:
                    continue
                statements[(owner, line)] = kept._replace(call_sites=(*kept.call_sites, entry))

    def touch(table: str, owner: str, verb: str) -> None:
        use = rows.setdefault(table.lower(), EntityUse(set(), set(), set(), set(), set()))
        use.tables.add(table)
        getattr(use, f"{verb}_by").add(owner)

    def place_conditions(
        func: ast.FunctionDef | ast.AsyncFunctionDef, owner: str, executions: list
    ) -> None:
        """Say what each of *func*'s statements requires of a row besides the value.

        A branchy statement is emitted as several variants at one ``execute``,
        and exactly one of them ran.  Unioning their terms -- which is what
        ``selects`` does, and what this did first -- reads as a conjunction and
        so **over-states** the requirement: 24 of the corpus's statements have
        variants that disagree, over 66 terms, and a reader ruling a row out
        against a term the text that ran never carried is the failure this
        field exists to prevent.

        So the variants are folded as what they are, a disjunction:
        ``(A and B) or (A and C)`` is ``A``, required outright, and ``B or C``
        as one condition.  A variant that requires nothing beyond the common
        part makes that disjunction free, and then it is not recorded at all.
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
            rendered: list[list[str]] = []
            for run in variants:
                window = run.window
                if window is None and run.varmap:
                    window = sql.binding_window(func, run.varmap, line)
                found = sqltext.conditions(run.sql)
                seeds: dict[str, list[int]] = {}
                for condition in found:
                    for key in condition.binds:
                        if key in seeds:
                            continue
                        at_lines = sorted(
                            {
                                bind.site.lineno
                                for bind in sql.bound_values(
                                    func, run.varmap or "", key, window
                                )
                            }
                        )
                        if at_lines:
                            seeds[key] = at_lines
                rendered.append([_render_condition(c, seeds) for c in found])
            common = set.intersection(*(set(one) for one in rendered))
            kept.conditions.extend(sorted(common))
            remainders: list[tuple[str, ...]] = []
            for one in rendered:
                shape = tuple(entry for entry in one if entry not in common)
                if shape not in remainders:
                    remainders.append(shape)
            if len(remainders) == 1:
                kept.conditions.extend(remainders[0])
            elif all(remainders):
                kept.conditions.append(
                    " OR ".join(
                        f"({' AND '.join(shape)})" if len(shape) > 1 else shape[0]
                        for shape in remainders
                    )
                )

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
                # Read for both call forms.  ``run.window`` is left alone
                # below, where it pairs a statement with its varmap: narrowing
                # the ``execute`` form's bind lookup would drop values from
                # ``selects``, which is a different question from where a walk
                # should start.
                window = run.window
                if window is None and run.varmap:
                    window = sql.binding_window(func, run.varmap, run.call.lineno)
                if window is None and sql.inline_binds(run.call):
                    # ``querySQLS(sql, {":name": name})``: the binds are
                    # filled in the call itself.
                    window = (run.call.lineno, run.call.end_lineno or run.call.lineno)
                at = ReadStatement(
                    owner=owner,
                    package=module.package,
                    file=module.rel_path,
                    blob_sha=module.blob_sha,
                    line=run.call.lineno,
                    window=window,
                    selects=set(),
                    excludes=set(),
                    keys=set(),
                    order_by=set(),
                    limits=set(),
                    tables=set(),
                    conditions=[],
                    unresolved=tuple(
                        sorted(
                            {
                                f"{stop.source} -- {stop.reason}"
                                for stop in run.unresolved
                            }
                        )
                    ),
                )
                read_tables: set[str] = set()
                for verb, table in _table_verbs(run.sql):
                    spec_class = attributor.class_for_table(table)
                    if verb == "read":
                        read_tables.add(table)
                    for column, key in sqltext.predicates(run.sql):
                        qualifier, attribute, _kind = _subject_of(
                            attributor, spec_class, table, column
                        )
                        subject = SubjectNode.make_name(qualifier, attribute)
                        values_of = bound(func, run, key, module.rel_path)
                        for value in values_of:
                            record(subject, value, owner, verb, at)
                        if not values_of and verb == "read":
                            statements.setdefault((at.owner, at.line), at).keys.add(subject)
                    if verb == "read":
                        for column, key in sqltext.exclusions(run.sql):
                            qualifier, attribute, _kind = _subject_of(
                                attributor, spec_class, table, column
                            )
                            subject = SubjectNode.make_name(qualifier, attribute)
                            for value in bound(func, run, key, module.rel_path):
                                exclude(subject, value, at)
                        for column, value in sqltext.excluded_literals(run.sql):
                            qualifier, attribute, _kind = _subject_of(
                                attributor, spec_class, table, column
                            )
                            exclude(SubjectNode.make_name(qualifier, attribute), value, at)
                    for column, value in sqltext.selected_literals(run.sql):
                        qualifier, attribute, _kind = _subject_of(
                            attributor, spec_class, table, column
                        )
                        record(
                            SubjectNode.make_name(qualifier, attribute),
                            value,
                            owner,
                            verb,
                            at,
                        )
                # After the verbs, and onto whichever record was kept: a join
                # reads two tables in one statement, and the second one would
                # otherwise land on the copy ``setdefault`` discarded.
                kept = statements.get((owner, run.call.lineno))
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
            place_conditions(func, owner, executions)
            # Regions first: the exits are read inside the stretch it settles.
            place_regions(func, owner, executions)
            place_exits(func, owner)
            place_call_sites(func, owner, module.rel_path, executions)
    return ReadSide(
        values=found,
        entities=rows,
        unreadable=unreadable,
        joins=joins,
        statements=statements,
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
                keys=sorted(at.keys),
                order_by=sorted(at.order_by),
                limits=sorted(at.limits),
                call_sites=list(at.call_sites),
                tables=sorted(at.tables),
                # Empty only where the call names no varmap to read -- one
                # statement in the corpus, ``querySQLS(sql.format(t), {})``,
                # whose values are all literals in the text.  There is no
                # assignment anywhere to place, so an empty window is the
                # true answer rather than a gap.
                bind_window=list(at.window) if at.window else [],
                conditions=sorted(at.conditions),
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
    elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
        targets = [node.target]
    else:
        return set()
    found: set[str] = set()
    for target in targets:
        parts = target.elts if isinstance(target, (ast.Tuple, ast.List)) else [target]
        for part in parts:
            if isinstance(part, ast.Name):
                found.add(part.id)
            elif isinstance(part, ast.Subscript) and isinstance(part.value, ast.Name):
                found.add(part.value.id)
    return found


def _mentions(expression: ast.AST, held: set[str]) -> bool:
    return any(isinstance(n, ast.Name) and n.id in held for n in ast.walk(expression))


def _carried_into(func: ast.AST, seed: set[str]) -> set[str]:
    """The names *seed*'s rows reach, by assignment, loop target or carrier.

    Deliberately generous, because of what it decides: not *where* the rows go
    but only whether the function's ``return`` is one of the places.  The
    flagship's closure picks up ``varMap`` along the way and the answer --
    ``retTaskList`` is returned -- is right anyway.  A name it should not have
    taken costs a statement that resolves to nothing, not a wrong lead: the
    caller still has to do something the map holds a junction for.
    """
    held = set(seed)
    growing = True
    while growing:
        growing = False
        for node in ast.walk(func):
            if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                if node.value is None or not _mentions(node.value, held):
                    continue
                fresh = _bound_by(node) - held
            elif isinstance(node, (ast.For, ast.AsyncFor)):
                if not _mentions(node.iter, held):
                    continue
                target = node.target
                parts = target.elts if isinstance(target, (ast.Tuple, ast.List)) else [target]
                fresh = {p.id for p in parts if isinstance(p, ast.Name)} - held
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if (
                    node.func.attr not in _CARRIERS
                    or not isinstance(node.func.value, ast.Name)
                    or not any(_mentions(a, held) for a in node.args)
                ):
                    continue
                fresh = {node.func.value.id} - held
            else:
                continue
            if fresh:
                held |= fresh
                growing = True
    return held


def _hands_rows_back(func: ast.AST, held: set[str]) -> bool:
    """Whether a ``return`` in *func* returns something built from *held*."""
    return any(
        isinstance(node, ast.Return) and node.value is not None and _mentions(node.value, held)
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
    seed: set[str],
    held: set[str],
    by_func: dict[str, set[str]],
    by_class: dict[str, set[str]],
) -> set[str]:
    """The junctions *held* reaches in *caller*, by the two rules already used.

    The same pair :func:`attach_crossings` applies inside a region -- a call
    the map owns a junction for, and a class built here and started as a
    thread -- with two things added that the region form does not need.  There
    the stretch is the statement's own and everything in it is about these
    rows.  Here it is somebody else's whole function, so the call has to
    actually *take* one of the names the rows arrived in, **and** the line
    they arrived on has to still be what that name holds there.

    The second is not caution.  ``datasetManager.py::main`` runs seven queries
    into the same ``res`` and hands each to a different thread class, so
    without it one statement's rows resolve to all seven constructions -- the
    reuse P1-54 met on the exits, met again a function further out.
    ``pathcond.surviving_writes`` is the same answer here as it was there: at
    the construction on line 906 the ``res`` in scope is the one from line
    896, and the six others were overwritten or had not run.
    """
    writes = [
        node
        for node in ast.walk(caller)
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign))
        and _bound_by(node) & seed
    ]

    def theirs(use: ast.AST) -> bool:
        """Whether the rows that reached *use* are the ones that arrived here."""
        site = use if isinstance(use, ast.stmt) else site_of(use)
        return arrived in surviving_writes(writes, site)

    reached: set[str] = set()
    built: dict[str, ast.AST] = {}
    for node in ast.walk(caller):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        callee = node.value.func
        made = callee.id if isinstance(callee, ast.Name) else getattr(callee, "attr", "")
        if made in by_class and any(_mentions(a, held) for a in node.value.args) and theirs(node):
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
        if any(_mentions(a, held) for a in node.args) and theirs(node):
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
    ``retTaskIDs`` -> ``retTaskList`` on the way.  Followed by name instead --
    see :func:`_carried_into` -- 106 statements hand their rows back.

    **A crossing is only stored where all three of these hold**, and 76 of
    those 106 fail the last one and keep an empty ``crosses_to``:

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
    rows = _rows_of(func, site.anchor.line_start, through)
    if not rows:
        return set(), False
    held = _carried_into(func, rows)
    if not _hands_rows_back(func, held):
        return set(), False
    reached: set[str] = set()
    for _rel, call, caller in _consumers_of(wanted, calls, site.anchor.file, func):
        statement = _enclosing(call, (ast.stmt,))
        if not isinstance(statement, ast.Assign):
            # Returned on again, or thrown away.  The first is the proxy shape
            # under another name and the second has no rows to follow.
            continue
        seed = _bound_by(statement)
        if seed:
            reached |= _reached_by(
                caller,
                statement,
                seed,
                _carried_into(caller, seed),
                by_func,
                by_class,
            )
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
                for table in _tables_of(run.sql):
                    # A table does not bound itself.  The statement's gates are
                    # read once and offered to every table in it, so a subject
                    # sitting on the gating table got its own table back as the
                    # thing limiting which of its rows can be seen --
                    # ``JEDI_AUX_Status_MinTaskID.status`` bounded by
                    # ``JEDI_AUX_Status_MinTaskID``.  Seventeen of the
                    # fifty-eight gated subjects were that, and the verdict
                    # they produce tells a reader to go and find out who
                    # maintains a table they are already looking at.
                    bounds = {gate for gate in gates if gate.lower() != table.lower()}
                    if not bounds:
                        continue
                    spec_class = attributor.class_for_table(table)
                    for column, _key in [*sqltext.predicates(run.sql), *sqltext.exclusions(run.sql)]:
                        qualifier, attribute, _kind = _subject_of(
                            attributor, spec_class, table, column
                        )
                        found.setdefault(
                            SubjectNode.make_name(qualifier, attribute), set()
                        ).update(bounds)
    return found


#: A table reference that is nothing but holes -- ``{}`` or ``{0}``.  What is
#: left when a name the source supplies at run time could not be resolved.
_UNREAD_TABLE = re.compile(r"(?:\{\d*\})+")


def _rows_touched(statement: str) -> list[tuple[str, str]]:
    """``[("read" | "created" | "updated" | "deleted", table)]`` for *statement*.

    The same three readings :func:`_tables_of` folds together, kept apart.  A
    predicate does not care which verb brought the table in -- an ``UPDATE``'s
    ``WHERE`` says which rows it was willing to act on as much as a query's
    does -- but "who reads a task's jobs" cares about nothing else.  Folding
    the verbs is how an ``UPDATE ... WHERE`` came to be reported as a query
    that selects on a value.
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

    :func:`_tables_of` with the verb kept, for the reading that needs it.  A
    statement naming one table under both verbs -- ``INSERT INTO a SELECT FROM
    a`` -- yields both, which is what it does.
    """
    return sorted(set(_rows_touched(statement)))


def _tables_of(statement: str) -> list[str]:
    """Return the tables a statement names, so a predicate can be qualified.

    All four verbs, not just ``SELECT``: most of the interesting predicates are
    on an ``UPDATE`` -- ``SET status=:status WHERE status=:oldStatus`` is one
    statement that both writes a status and says which one it is willing to
    move away from, and reading only the queries loses every one of them.

    A statement joining two tables qualifies its predicates against both, which
    over-reads: ``status`` in a task/dataset join is attributed to each.  That
    is the safe direction here -- the values answer whether *anything* selects
    on a status, so one credited too widely weakens a report while a missing
    one would invent a dead end.
    """
    return sorted({table for _verb, table in _rows_touched(statement)})
