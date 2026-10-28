"""Code Map node models.

The Code Map is a machine-derived view of a target system's source: where the
code *determines* the value of a subject, under what conditions, and where it
hands off to another system.  It is stored in the same Neo4j database as the
incident graph but under its own labels, because the two have opposite
lifecycles -- the Code Map is regenerated from source at will, the incident
graph is human-validated and irreplaceable.

The node kinds:

``SubjectNode``
    An attribute the code writes -- a declared spec attribute or a table's
    column -- with the census criteria that say its value is worth asking
    "why is it this?" about (``criteria``; none where none fires).
``JunctionNode``
    A place where the code settles a subject's value.  Its branches are the
    possible outcomes, each with the path condition that selects it.  A
    junction does not *judge*; the branch taken follows deterministically from
    the conditions, which is why it is not called a decision point (that term
    is reserved for the constrained points where an LLM or a human chooses).
``FilterStageNode``
    One reason a candidate was dropped on the way to a selection.  Separate
    from a junction because every stage of a chain runs and each removes some
    candidates, where a junction's branches are alternatives and one wins.
``BoundaryNode``
    Where causation crosses into a system this map does not cover.  Modelled
    explicitly rather than left as an absence so that adding the other
    system's map later is a *binding* operation instead of a re-derivation.
``ValueEnumNode``
    One ``NAME = value`` constant, so an observed code can be decoded.

Identity is a semantic signature, never a file position.  Positions move --
between ``panda-server-source`` 0.8.1 and 1.0.2 the pilot boundary moved file
entirely (``jobdispatcher/JobDispatcher.py`` disappeared; the entry is now
``api/v1/pilot_api.py::update_job``) while remaining the same boundary.  The
anchor records where the evidence was found; the signature says what it *is*.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Optional

from pydantic import BaseModel, Field, PrivateAttr, computed_field, model_validator

from bamboo.models.graph_element import BaseNode, NodeType


class SourceModule(BaseModel):
    """One parsed source file, shared by every recognizer in a build.

    Parsing and hashing happen once per module rather than once per
    recognizer: with several slices reading the same tree, re-reading it per
    slice would multiply the build's cost for nothing.
    """

    model_config = {"arbitrary_types_allowed": True}

    package: str
    rel_path: str  # package-prefixed, e.g. "pandajedi/jediorder/JobGenerator.py"
    tree: Any  # ast.Module
    source: str
    blob_sha: str


class Anchor(BaseModel):
    """Where in the source a Code Map fact was found.

    Provenance for a human following the map back to the code -- deliberately
    *not* an identity.  ``blob_sha`` lets a rebuild tell whether the enclosing
    file changed at all; ``line_span`` is only ever as good as the version the
    map was derived from.
    """

    package: str
    file: str
    line_start: int
    line_end: Optional[int] = Field(
        default=None,
        description=(
            "End of the span, and today nothing in production reads it -- "
            "``reading.region_in`` derives the region from the containing "
            "function instead, and ``diff.py`` leaves the whole anchor out of "
            "its comparison on purpose.  So a wrong one is silent, which is "
            "how seventeen junctions came to carry a span that ends above its "
            "own start.  An analyser reading the span is the natural first use "
            "of it, and that is when a wrong one would be believed."
        ),
    )
    blob_sha: Optional[str] = None

    def as_ref(self) -> str:
        """Return a ``file:line`` reference for display."""
        return f"{self.file}:{self.line_start}"


class SubjectNode(BaseNode):
    """An attribute the reasoning may start from, with the promotion criteria that fire on it -- none where none does.

    ``name`` is qualified as ``"<spec_class>.<attribute>"`` because the same
    attribute name means different things in different classes -- ``FileSpec``
    and ``JediFileSpec`` both declare ``status``.  The same shape recurs for
    error codes (``(namespace, code)``) and for maps (``map_id``), so every
    Code Map identifier is qualified as a rule.
    """

    node_type: NodeType = NodeType.SUBJECT
    map_id: str
    derived_from: str
    spec_class: str = Field(
        ...,
        description=(
            "What qualifies the attribute: a spec class, or a table where no "
            "spec holds that row.  Named for the common case; ``qualifier_kind`` "
            "says which it is."
        ),
    )
    qualifier_kind: str = Field(
        default="spec",
        description=(
            "spec | table.  Recorded so a reader can tell ``JEDI_Events.status`` "
            "from ``JediTaskSpec.status`` -- one names a table, the other a "
            "class, and they are different kinds of claim."
        ),
    )
    attribute: str
    criteria: list[str] = Field(
        default_factory=list,
        description=(
            "Promotion criteria satisfied, e.g. ['1:state-gate-in-where']; empty "
            "where none is -- the subject is kept, and nothing says it decides anything."
        ),
    )
    vocabulary: list[str] = Field(
        default_factory=list,
        description="Declared or observed value set, when the code states one.",
    )
    selected_values: list[str] = Field(
        default_factory=list,
        description=(
            "Values something acts on rows by -- a query's ``WHERE`` or an "
            "update's.  The counterpart of the outcomes the junctions write: "
            "together they make a state machine out of a pile of writes, since "
            "a value nothing acts on is one nothing ever moves away from.  Both "
            "verbs count here, because an ``UPDATE ... WHERE status=:old`` "
            "moves a task out of that status as squarely as a query does."
        ),
    )
    selected_by: dict[str, list[str]] = Field(
        default_factory=dict,
        description=(
            "``{value: the functions whose *query* selects rows on it}``.  Kept "
            "per value rather than per subject because the two answer different "
            "questions: ``JediTaskSpec.status`` is selected on thirty-one values "
            "by dozens of functions, and pooling them says only that the subject "
            "is read.  Per value it says who to ask -- ``finishing`` is selected "
            "by exactly one query in the whole corpus.  Queries only; see "
            "``updated_by`` for the other verb and why the two are apart."
        ),
    )
    updated_by: dict[str, list[str]] = Field(
        default_factory=dict,
        description=(
            "``{value: the functions whose UPDATE or DELETE acts on rows "
            "already holding it}``.  A different claim from ``selected_by``, "
            "and folding them said an update was a query for a fifth of the "
            "corpus.  The distinction is what the reading is for: asked why a "
            "row was not picked up, a query is somewhere that could have "
            "missed it, while an update is the picking up itself -- so the "
            "first names a place to look and the second names what already "
            "happened."
        ),
    )
    selection_gates: list[str] = Field(
        default_factory=list,
        description=(
            "Tables that bound which rows the queries selecting this subject "
            "can see at all, and that nothing in this map writes.  A different "
            "reason for a row not to be picked up than any value in "
            "``selected_values``: ``JEDI_AUX_Status_MinTaskID`` is joined by "
            "thirty-one functions on ``jediTaskID >= min_jediTaskID``, so when "
            "it goes stale a task is invisible to all of them whatever its "
            "status is.  Named here because a status the map says is selected "
            "is only half an answer without what bounds the selecting."
        ),
    )
    truncated_to: Optional[int] = Field(
        default=None,
        description=(
            "How many characters the spec cuts this field to before it is "
            "stored, where the class declares it in ``_limitLength`` and "
            "cuts by it.  ``JobSpec`` keeps 500 of a ``ddmErrorDiag``, and "
            "``JediTaskSpec`` 510 of an ``errorDialog`` the moment it is set. "
            "A value read back at exactly that length may have lost its "
            "tail, so an arm is ruled out by its frame's head alone -- see "
            ":func:`branch_excludes`."
        ),
    )

    @staticmethod
    def make_name(spec_class: str, attribute: str) -> str:
        return f"{spec_class}.{attribute}"


class EntityNode(BaseNode):
    """A kind of row, and which functions read it and which write it.

    The map's vocabulary was pairs -- ``(spec class, attribute)`` -- and a pair
    can only say something about a *column*.  ``getPandaIDsWithTask_JEDI``
    selects a task's jobs on nothing but the join key, so it named no promoted
    subject and the map read that as observing nothing.  It observes jobs.  The
    fact had nowhere to go: ``class_for_table`` was already computed beside the
    predicate loop and thrown away when no predicate landed on a promoted
    attribute.

    **Not folded into** :class:`SubjectNode`.  That model is load-bearing for
    promotion and for three gates -- ``declared-status-is-written``, the
    written-but-never-selected report and ``structural-attribution-agrees`` --
    all of which read ``attribute``.  Rows with no attribute would flow into
    every one of them, which is the "one field answering two questions" shape
    this map has already paid for twice (``log_files`` and ``opened_by``).

    **Stored, not merely reported.**  ``DiagnosticTemplate`` and
    ``EnumerationWrite`` are indexes kept on the fragment with no label, and
    nothing in an investigation can reach them -- the database holds none of
    their rows.  An entity is read by ``derive-strategy``, so it is a node.

    **All four verbs stay apart.**  Selecting a task's jobs and updating them
    are different claims, and pooling them is exactly the conflation that let a
    function's ``UPDATE ... WHERE`` be reported as a query that selects on a
    value.  The same is true one level down: an ``INSERT`` is an object
    entering the system, an ``UPDATE`` is one already here moving, and a
    ``DELETE`` is one leaving.  Reported as one "written" they could not be
    asked apart, so the map had no way to answer where a kind of row comes into
    existence -- while telling a reader to go and ask whether a command
    arrived, which is exactly a row appearing in a table.

    **Keyed by the table, not by the spec class.**  Most of this corpus's
    tables have no spec: forty-one of the forty-eight it inserts into, and they
    are the ones where arrival means something -- ``HARVESTER_COMMANDS``,
    ``PRODSYS_COMM``, ``SQL_QUEUE``, ``async_requests``, ``Job_Output_Report``,
    ``users``, ``SiteData``, ``T_TASK``, ``jobs_StatusLog``.  Keying on the
    class answered for seven.

    The class would be the wrong key even where there is one.  ``SubjectNode``
    keys on it because ``jobsActive4``, ``jobsDefined4`` and ``jobsArchived4``
    are one ``JobSpec.jobStatus``, and splitting that would make three subjects
    out of one fact.  A row's *creation* is the opposite: which of those tables
    it was created in is the fact, and folding them loses it.  So the class
    rides along as an attribute instead of being the signature.

    The name is the table folded to lower case, because SQL identifiers are
    case-insensitive and this corpus spells one table several ways.
    """

    node_type: NodeType = NodeType.ENTITY
    map_id: str
    derived_from: str
    spec_class: Optional[str] = Field(
        default=None,
        description=(
            "The spec class this table holds, where one was learned.  An "
            "attribute rather than the key -- see the class docstring -- and "
            "``None`` for the majority of tables, which declare none."
        ),
    )
    tables: list[str] = Field(
        default_factory=list,
        description=(
            "The spellings the corpus uses for this table.  More than one is "
            "ordinary: ``reassignShare`` loops over ``jobsactive4`` where "
            "everything else writes ``jobsActive4``."
        ),
    )
    read_by: list[str] = Field(
        default_factory=list,
        description="``module::method`` of every function whose SELECT names this table.",
    )
    created_by: list[str] = Field(
        default_factory=list,
        description=(
            "The same for INSERT: where a row of this kind comes into "
            "existence.  The question a value's branch table cannot answer, "
            "and the one a stalled command needs -- a command arriving is a "
            "row appearing here."
        ),
    )
    updated_by: list[str] = Field(
        default_factory=list,
        description="The same for UPDATE: where a row already here is moved.",
    )
    deleted_by: list[str] = Field(
        default_factory=list,
        description=(
            "The same for DELETE.  Worth its own list because of what it means "
            "on a command table: a DELETE followed by an INSERT is a second "
            "command silently replacing one that was never picked up."
        ),
    )


#: What a line is about.  A junction leaves two kinds and they are not
#: interchangeable: one carries the value, so a probe can be built from it by
#: substituting the value observed; the other carries the row count the write
#: returned, which is the only trace a compare-and-set leaves when it loses.
#: Reading the second as the first would build a pattern for a value that
#: never appears in it.
REPORTS_DECISION = "decision"
REPORTS_ROWS_CHANGED = "rows_changed"


class Emit(BaseModel):
    """A log line the code leaves when it settles a value, and where it lands.

    The file belongs to the emit rather than to the node, because one junction
    leaves two kinds of line in two different places.  ``updateTask_JEDI``
    writes ``updated N rows`` through the proxy's own inherited logger, while
    the ``set task_status=`` line about that same junction is written by the
    knight that called it.  ``JunctionNode.owns_logger`` answers that with one
    bit for the whole node, which was right for the question it was added for
    and is the wrong shape for this one.

    ``log_files`` empty means the module writes through a wrapper its caller
    supplied and the map cannot say where that goes.  Left empty rather than
    filled in: a probe against a guessed file comes back silent, and silence is
    what the eliminator reads.
    """

    template: str = Field(
        ...,
        description=(
            "The line as the source frames it, with run-time parts left as "
            "``{}`` -- the literal frame is what a production log is matched on."
        ),
    )
    log_level: Optional[str] = Field(
        default=None,
        description=(
            "Level it is written at, or None where one hop could not settle it. "
            "A DEBUG line the map offers as an observable does not exist if the "
            "deployment runs at INFO, which only production can say."
        ),
    )
    log_files: list[str] = Field(
        default_factory=list, description="Files this particular line lands in."
    )
    reports: str = Field(
        default=REPORTS_DECISION,
        description=(
            f"{REPORTS_DECISION} | {REPORTS_ROWS_CHANGED} -- whether the line "
            "carries the value the code settled, or the number of rows the "
            "write actually changed.  Both are evidence and they answer "
            "different questions: the first says the code decided, the second "
            "says the row took it."
        ),
    )


#: ``passthrough(JediTaskSpec.oldStatus)`` -> ``JediTaskSpec.oldStatus``.  Kept
#: beside the field whose spelling it reads, so the grammar of an outcome has
#: one definition: the readers of it are in two modules and a second copy of the
#: pattern is a second thing free to drift from what the extractor writes.
PASSTHROUGH_OUTCOME = re.compile(r"^passthrough\((?P<field>[^)]+)\)$")


class ConditionalGuard(BaseModel):
    """A row precondition some writes of a branch's value carry, and when they do.

    ``updateJobStatus`` adds ``AND NOT jobStatus=:ngStatus`` only when the new
    status is ``transferring``: a ``holding`` job may not move there, and may
    move anywhere else.  ``row_precondition`` holds what every write of the
    value requires, so it cannot say this, and saying it there would claim the
    guard for every other status the same branch writes.
    """

    term: str
    when: list[str] = Field(
        ...,
        description=(
            "The tests, beside the branch's own path condition, under which a "
            "write carries the term: where its text, the value its column is "
            "set to (the term guards only a column the statement writes), and "
            "the execute were written.  Each test is necessary, as a path "
            "condition is; together they are taken as enough, which a "
            "``return`` between them would break.  Two guards with one term "
            "are alternatives."
        ),
    )


#: :attr:`MustName.row` for a hole holding a column of the row the arm writes.
MUST_ROW_WRITTEN = "written"


class MustName(BaseModel):
    """A hole of a must print that holds a column of the row the arm writes.

    ``tmp_log.debug(f"PandaID={job.PandaID} ...")`` on every path to
    ``job.jobStatus = "failed"``: the line that says ``PandaID=123`` names the
    row the arm wrote, so finding it among a run's lines says that run is the
    one that wrote it.  The hole has to be read off the object the arm writes
    to, and the same spelling is not the same object: ``job`` rebound between
    the print and the arm -- a loop's next item, ``job = self.job`` on one
    path -- prints one job and writes another.  So a name is said only where,
    on every path into the arm, the print ran after the object and the
    attribute were last bound, and no call between was made on it or handed
    it.  The value is the column as it was when printed: for the column the
    arm writes, the value before the write.  A hole that spells the written
    object's attribute and is not known to hold it is kept, with why, so that
    its absence from the names is not read as a print that names nothing.
    """

    line: int = Field(description="The print's line: a statement of the owner's :class:`FunctionPrintsNode`.")
    expression: str = Field(description="What fills the hole, as :attr:`SkeletonHole.expression` spells it.")
    column: str = Field(description="The column it holds, as the entity's class spells it: ``JobSpec.PandaID``.")
    row: str = Field(
        default="",
        description=(
            f"{MUST_ROW_WRITTEN!r}: the row the arm writes.  Empty when the "
            "hole is not known to hold it, with ``why``."
        ),
    )
    why: str = Field(default="", description="Why the hole is not known to hold the written row's column.")


#: :attr:`HistoryPair.row` where the history row is the arm's row: the writer
#: is handed an attribute of the object the arm writes, bound alike at both.
HISTORY_ROW_SAME = "same"
#: :attr:`HistoryPair.row` where the history row can be the arm's row: the
#: arm's object, or its key, followed into the call reaches the writer's key.
HISTORY_ROW_CAN = "can"
#: :attr:`HistoryPair.order`: where the history call runs from the point.
HISTORY_AFTER = "after"
HISTORY_BEFORE = "before"
HISTORY_EITHER = "either"


class HistoryRekey(BaseModel):
    """Where a pair's object is given its key again between the point and the call (:attr:`HistoryPair.rekeyed`)."""

    at: str = Field(description="``owner:line`` of the assignment: the caller, or a function on the pair's way.")
    path: str = Field(description="The key assigned, in that function's names: ``job@3917.PandaID``.")
    value: str = Field(
        default="",
        description="What it is given, where that is a literal -- ``None`` --; empty where the value is not read.",
    )
    kept_in: list[str] = Field(
        default_factory=list,
        description=(
            "The object's attributes given the old key before the new one, in that function: ``parentID`` from "
            "``job.parentID = job.PandaID`` on a way to the assignment, the object not bound again and the attribute "
            "not given another value between.  The row written whole holds the old key there."
        ),
    )
    printed_at: list[int] = Field(
        default_factory=list,
        description=(
            "The lines of that function's prints that print the new key: on a way from the assignment, the object "
            "not bound again between, a hole spelling the key's path.  The line prints in its statement's runs, "
            "whose mark was made with the old key where the run started before.  With ``kept_in``, empty where "
            "nothing names the old key and the new together."
        ),
    )


class HistoryPair(BaseModel):
    """A call that writes a history row in the same pass of a function as an arm.

    The question is bamboo-68's (h2-value-time-design.md §3): a value the map
    sees set has no time of its own, and PanDA keeps a job's and a task's
    past statuses in tables of their own -- ``jobs_StatusLog``,
    ``TASKS_STATUSLOG``, one row per change, stamped when it was written.  A
    history row written in the same pass as the arm gives the arm a time,
    and the run that pass is in -- a logger bound with a key -- says which
    lines of the log are that pass's.

    ``level`` 0 is the arm's own function and ``point`` the arm's line; level
    n+1 is a caller of a level-n function and ``point`` the line of its call
    on the way to the arm.  The call at ``call`` is in one pass with the
    point: one runs after the other with no run started between them and no
    turn of a loop holding both, read by
    :func:`~bamboo.codemap.panda.reaching.reaching`.  The walk up stops on
    each way at the first level with a pair, and does not take a way the
    values its calls hand rule out (``setErrDiag(diag)`` leaves ``append``
    False, so its append arm pairs only with callers handing it True); the
    caller's own parameters are read as any value.

    **Reachable, not taken.**  A call made on some paths only --
    ``updateTask_JEDI`` records a task's status only when it changed -- pairs
    as if made on every path, and a function running one of several actions
    per pass pairs each action's arms with the others' history calls.
    ``writes`` names every history write the call can lead to, not the one it
    made.  Whether the row is the arm's own (``row``) is read in place where
    both are in one function and the arm writes an attribute; otherwise the
    arm's object -- or its key, for an SQL arm -- is followed from the caller
    into the call (:func:`bamboo.codemap.objects.follow`), and ``can`` says it
    reaches the writer's key: on some way, an element of a list among the
    others, the bindings read without order and without what the calls on the
    way down hand -- a row the pass can write, not one it wrote.  Where the
    object is given a new key between the point and the call (``retryJob``'s
    new ``PandaID``), the row is the new key's while the run's mark holds the
    old one: ``rekeyed`` says where.
    """

    level: int = Field(description="0 for the arm's own function; n+1 for a caller of a level-n one.")
    caller: str = Field(description="The function the point and the call are in: a :attr:`ReachNode.owner`.")
    point: int = Field(description="The arm's line at level 0; else the line of the call on the way to the arm.")
    call: int = Field(description="The line of the call that leads to a history row.")
    callee: str = Field(description="The called expression, as source, cut at 80 characters.")
    order: str = Field(
        description=(
            f"{HISTORY_AFTER!r}, {HISTORY_BEFORE!r} or {HISTORY_EITHER!r}: "
            "whether the call runs after the point, before it, or either, in the pass."
        )
    )
    history: list[str] = Field(
        default_factory=list,
        description="The subjects whose history the call can write: ``JobSpec.jobStatus``, ``JediTaskSpec.status``.",
    )
    writes: list[str] = Field(
        default_factory=list,
        description=(
            "``owner:line`` of each history write the call can lead to: a "
            ":class:`HistoryCall` on that owner's :class:`ReachNode`, which "
            "holds the values its row can record.  The call itself where it "
            "is the writer's."
        ),
    )
    row: str = Field(
        default="",
        description=(
            f"{HISTORY_ROW_SAME!r} read in place; {HISTORY_ROW_CAN!r} where the "
            "arm's object followed into the call reaches the writer's key by a "
            "way that runs in one pass (``object``, ``row_via``) -- each "
            "interval of it, between two events of the walk, held by a run "
            "with the path the object is held by not bound again; where the "
            "object reaches the key by no such way, empty with ``row_why`` "
            "naming an interval; ``another: <expression>`` where the "
            "writer is handed its key as another expression the object is not "
            "followed to -- ``other.PandaID``, or a job's ``job.jediTaskID`` "
            "for the task's history -- whose row is not read.  Empty with "
            "``row_why``."
        ),
    )
    row_why: str = Field(default="", description="Why ``row`` is empty.")
    object: str = Field(
        default="",
        description=(
            "The path of the caller holding the arm's object -- ``self.job`` "
            "in the adder's ``process_job_report`` -- or, for an SQL arm, the "
            "row's key: what a reader asks of the pass's run (may_change's "
            "path).  Set with ``same`` and ``can``."
        ),
    )
    row_via: list[str] = Field(
        default_factory=list,
        description=(
            "With ``can``, one way that runs in one pass, by which the object "
            "goes from the caller's "
            "``object`` to the writer's key, a step each: ``owner:line how "
            "path`` for a binding of a function it passes (``element job`` "
            "for ``for job in jobs``, ``fill updatedJobList[]`` for an "
            "append, ``... back`` from a name to what is bound to it), "
            "``caller:line -> callee path`` for a call it is handed into, "
            "``owner:line <- callee path`` for a call that hands it back "
            "(kept on the callee's ``self``, or returned), and last "
            "``owner:line key`` -- the history write and the path it is "
            "handed as its key."
        ),
    )
    rekeyed: list[HistoryRekey] = Field(
        default_factory=list,
        description=(
            "With ``same`` and ``can``, each assignment of the object's key "
            "that runs between the point and the call, either way round, with "
            "the path the object is held by not bound again on the way: in "
            "the caller, in each function on ``row_via``, or in a call made "
            "between that is handed the object.  Only an assignment on a path "
            "the record follows the object by is seen, as with ``can`` itself."
            "  The entries are in the order of their ``at`` texts, compared as "
            "strings, not the order they run in."
        ),
    )
    runs: list[int] = Field(
        default_factory=list,
        description=(
            "The lines of the runs (:attr:`FunctionPrintsNode.runs`) the point "
            "is in: the caller's, or ``runs_in``'s.  Empty with ``run_hole``."
        ),
    )
    runs_in: str = Field(
        default="",
        description=(
            "The function whose runs ``runs`` are, where the caller prints "
            "through a logger it is handed: the one that builds it -- "
            "``PostProcessorThread.post_process_tasks`` for the ``tmpLog`` "
            "``doPostProcess`` is handed.  Empty for the caller's own."
        ),
    )
    run_hole: str = Field(default="", description="Why ``runs`` is empty.")


class Branch(BaseModel):
    """One possible outcome of a junction and the condition that selects it.

    ``outcome`` is per-branch, not a property of the junction: a single writer
    reaches several values, and "no transition at all" is itself an outcome
    that matters (a task sitting in ``pending`` because a lock was held is not
    the same as one that took a different branch).

    ``path_condition`` is the conjunction of the ``if``/``elif``/``else`` tests
    dominating the write, and of one entry for every other write of the same
    place that can replace the value before it is read
    (:func:`~bamboo.codemap.panda.pathcond.replacements`): ``not (<the tests
    that write is under and this one is not>)``, ``not any(<them> for <target>
    in <iter>)`` where it sits in a loop this one is not in, ending in ``  [in
    <method>()]`` where a setter's own tests decide it, or in ``  [at <line>]``
    where a test reads what may have changed since this write -- its names as
    they are when that line's test runs.  A write whose value is
    never read -- replaced on every way to its write -- is no branch.  It is
    not solved symbolically -- the values come from observation and are
    substituted in, which answers "which branch actually fired" rather than
    "which branch could fire".

    ``row_precondition`` is the other kind of condition, and the two are not
    interchangeable: the path condition is evaluated by the code, before the
    write, and decides whether it is attempted; the row precondition is
    evaluated by the database, during the write, against the row as it stands,
    and decides whether it lands.  Only the first is visible in a log.
    """

    outcome: str = Field(
        ...,
        description=(
            "Resulting value, ``passthrough(<attr>)`` when carried from "
            "elsewhere, or ``NO_TRANSITION``."
        ),
    )
    path_condition: list[str] = Field(default_factory=list)
    row_precondition: list[str] = Field(
        default_factory=list,
        description=(
            "What the row already had to say for this write to change it, when "
            "the statement tests a column it also writes.  Losing that race is "
            "silent -- zero rows changed, no exception, a count the caller "
            "usually discards -- so a branch carrying one is a branch whose "
            "outcome the code can announce without the row ever taking it."
        ),
    )
    row_precondition_when: list[ConditionalGuard] = Field(
        default_factory=list,
        description=(
            "The guards only some writes of this value carry, each with when.  "
            "Apart from ``row_precondition`` so that a reader taking every "
            "guard there as unconditional stays right; read where the "
            "statement was read by running the function, whose pieces say "
            "where each part of the text was written."
        ),
    )
    tags: list[str] = Field(
        default_factory=list,
        description=(
            "Machine-readable tags the code names when this branch fires, e.g. "
            "``action=set_exhausted`` and ``reason=low_efficiency``.  A list "
            "because the signature is a *set*: PanDA writes an action and a "
            "reason as separate tokens and puts words between them (``action="
            "set_exhausted since reason=many_shorter_jobs``), so a single "
            "joined string would match no message production ever writes.  "
            "Read where the write is read, from the block that records the "
            "line, and the same tokens land in the record itself when the "
            "block also calls a message setter -- which is what lets an "
            "observed ``errorDialog`` name one branch out of six."
        ),
    )
    messages: list[str] = Field(
        default_factory=list,
        description=(
            "The frames the block records when this branch fires, holes left "
            "open.  The untagged half of the same signature, and the one "
            "production mostly writes: of thirty tasks found in ``exhausted`` "
            "with a message on the record, none carried a tag -- they were "
            "retry refusals and the goal check, which write prose.  Weaker "
            "evidence than a tag, since wording drifts between releases and "
            "two arms can share a frame, so a match names an arm only where it "
            "is the only one."
        ),
    )
    line: Optional[int] = Field(
        default=None,
        description=(
            "Where this branch's write is.  The junction's anchor is the first "
            "of them, and six branches of one function sharing it answers "
            "*which branch* with a line that belongs to another; an index that "
            "names a branch has to be able to say where to read.  Not part of "
            "the content hash: a branch's identity is its outcome, and a line "
            "moves whenever anything above it is edited."
        ),
    )
    other_lines: list[int] = Field(
        default_factory=list,
        description=(
            "Where else this branch's value is written.  Two writes of one "
            "outcome under one path condition are one branch, and ``line`` is "
            "the first of them: ``increaseAttemptNrPanda`` binds "
            "``maxAttempt`` on lines 846, 851 and 856 under the same tests, and "
            "the two ``run`` s of ``datasetManager.py`` that delete a dataset "
            "are one junction's.  A fact read at ``line`` alone -- which "
            "prints come before it -- holds for this branch only if it holds "
            "at these too."
        ),
    )
    emits: list[Emit] = Field(
        default_factory=list,
        description=(
            "Lines the code writes when this branch settles the value, each "
            "with the file it lands in.  Withdrawn once, when the only "
            "templates that survived promotion were identifier frames rather "
            "than diagnostics; raised again now that a consumer needs them -- "
            "the walk renders them into the skeleton, one row per line the "
            "code prints, and while they were empty the report fell back to "
            "one hard-coded shape covering one subject."
        ),
    )
    log_level: Optional[str] = Field(
        default=None,
        description=(
            "Level of the diagnostic emit.  Recorded because a DEBUG line the "
            "map promises as an observable does not exist if production runs "
            "at INFO -- only comparing against real logs settles that."
        ),
    )
    order: int = 0
    tier: int = Field(
        default=1,
        description=(
            "1 when the outcome is statically resolved; 2 when the writer is "
            "known but the value is only determined at run time.  Tier 2 still "
            "supports localize/prune, which read observed output rather than "
            "the static outcome."
        ),
    )
    value_frames: list[str] = Field(
        default_factory=list,
        description=(
            "Every text a tier-2 arm's value can be, holes as ``{}``, read "
            "back through the locals it is built from -- "
            "``err_msg + extracted_err_str`` is ``Could not add files to DDM: "
            "{}``.  **All of them or none**: empty whenever one part of one "
            "of them cannot be bounded, because an observed value that fits "
            "none of them rules the arm out.  Not ``messages``, which is what "
            "the block records about the decision rather than what it writes."
        ),
    )
    must_prints: Optional[list[int]] = Field(
        default=None,
        description=(
            "The lines of the owner's print statements that run before the "
            "statement at ``line`` on every path from the function's entry, "
            "exceptions included -- see :mod:`bamboo.codemap.panda.mustprints`.  "
            "A run without one of these lines did not reach the arm.  The "
            "level, files and rows are the owner's :class:`FunctionPrintsNode` "
            "statement at the line.  ``None`` when not read, with "
            "``must_hole``; ``[]`` when read and none is."
        ),
    )
    must_names: Optional[list[MustName]] = Field(
        default=None,
        description=(
            "The holes of the must prints that read an attribute of the object "
            "the arm writes, each saying whether it holds the written row's "
            "column.  ``None`` when not read, with ``must_hole``: the names of "
            "an arm that does not write the subject as an object's attribute "
            "-- an SQL statement's binds -- are not read yet."
        ),
    )
    must_hole: str = Field(
        default="",
        description="Why ``must_prints`` or ``must_names`` is ``None``.",
    )
    history: Optional[list[HistoryPair]] = Field(
        default=None,
        description=(
            "The calls that write a history row in the same pass as this "
            "branch's write, at its function or at the first caller on each "
            "way up that has one -- see :mod:`bamboo.codemap.panda.runs`.  "
            "``[]`` when read and none is; ``None`` when not read, with "
            "``history_hole``: a subject no history table records."
        ),
    )
    history_hole: str = Field(
        default="",
        description=(
            "Why ``history`` is ``None``; beside a list, where the walk up "
            "stopped short of an end, so the list may hold less than every pair; "
            "beside an empty list, also a way up it did not take because what "
            "the calls on it hand rule it out."
        ),
    )


#: A hole in a frame -- see :attr:`Branch.value_frames`.
_HOLE = "{}"


def pooled_frames(one: list[str], other: list[str]) -> list[str]:
    """The frames of two branches kept as one: both sets, or none if either has none.

    Two writes with the same outcome text and conditions are one branch, but
    the same text read at two sites need not hold the same locals, and an
    empty list means the value is unbounded, not that it has no text.  Keeping
    the first branch's frames would rule the arm out by the other's value.
    """
    if not one or not other:
        return []
    return sorted(set(one) | set(other))


def branch_excludes(branch: Branch, value: str, truncated_to: Optional[int] = None) -> bool:
    """Whether a tier-2 branch's own frames prove it cannot have produced *value*.

    Tier 2 says the writer is known and the value is settled at run time, and
    everywhere else that is read as "this one could have" -- the honest answer
    for ``runtime(newStatus)``, where nothing bounds what the name holds.  It
    is not the honest answer for ``runtime(f'merge_{s}')``: whatever ``s``
    turns out to be, the result begins with ``merge_``, so a row reading
    ``es_inaction`` was written somewhere else.  Nor for ``runtime(err_msg +
    extracted_err_str)`` where ``err_msg`` was set to ``"Could not add files to
    DDM: "`` two lines up: of nine writers offered for a ``ddmErrorDiag``, five
    fall to their own text once the locals are read.

    **Syntactic proof only.**  The frames are the branch's own value, read back
    through the locals that build it, and nothing here consults another node:
    an elimination resting on some *other* fact being complete is the shape
    that turns an incomplete candidate set into a confident wrong one.
    Copying a value from elsewhere -- ``passthrough(JobSpec.jobStatus)`` --
    looks like it bounds the value too, and does not: it would need that
    subject's value set to be closed, and two of this corpus's hundred and
    eight subjects have one.  An arm with no frames is kept.

    **Whole, unless the value may have lost its tail.**  A spec that cuts a
    field cuts it to exactly *truncated_to* characters, so a value of any
    other length is whole and has to fit a frame end to end, and one of
    exactly that length may be the head of a longer text, which only a
    frame's head can rule out.  Reading every value as whole rules out the arm
    that wrote a long ``ddmErrorDiag`` -- ``JobSpec`` keeps 500 characters and
    the frame's tail is in what it dropped.  Reading every value as a head
    gives up 26 eliminations the map makes, ``es_merge`` against
    ``es_merge_{}`` among them.
    """
    if branch.tier != 2 or not branch.value_frames:
        return False
    return not fitting_frames(branch, value, truncated_to)


def fitting_frames(branch: Branch, value: str, truncated_to: Optional[int] = None) -> list[str]:
    """The frames of a tier-2 *branch* that *value* fits, read as :func:`branch_excludes` reads them."""
    if branch.tier != 2:
        return []
    cut = truncated_to is not None and len(value) == truncated_to
    return [frame for frame in branch.value_frames if _fits(frame.split(_HOLE), value, cut)]


def _fits(literals: list[str], value: str, cut: bool) -> bool:
    """Whether *value* is what a frame of these literal chunks renders, or its head when *cut*."""
    if len(literals) == 1:
        return literals[0][: len(value)] == value if cut else literals[0] == value
    head, *middle, tail = literals
    if cut and len(head) >= len(value):
        return head[: len(value)] == value
    if not value.startswith(head):
        return False
    if cut:
        return True
    # A hole is any text, the empty one and line breaks included, so the
    # leftmost place each chunk fits is as good as any.
    cursor, end = len(head), len(value) - len(tail)
    if end < cursor or not value.endswith(tail):
        return False
    for chunk in middle:
        found = value.find(chunk, cursor, end)
        if found < 0:
            return False
        cursor = found + len(chunk)
    return True


#: ``jobStatus=activated``, ``status='ready'`` -- a settled equality.
_REQUIRES_ONE = re.compile(r"^([\w.]+)\s*=\s*('?)([^:\s'(){},]+)\2$")
#: ``jobStatus IN (activated,throttled)`` -- a settled list, quoted or not.
_REQUIRES_ANY = re.compile(r"^([\w.]+)\s+IN\s*\(([^():{}]+)\)$", re.IGNORECASE)


def required_values(term: str) -> Optional[tuple[str, list[str]]]:
    """``(column, values)`` a row precondition requires the row to hold, or ``None``.

    One reading of a guard for everything that compares guards with anything,
    because two readings drift: the gate matched ``col=value`` alone, so
    ``getTasksToExecCommand_JEDI``'s ``jobStatus IN (activated,throttled)`` --
    a write that moves activated jobs on a command -- was not among the movers
    of ``activated`` anywhere, and nothing said so.

    The column comes back bare and lowered (``tab.jobStatus`` -> ``jobstatus``),
    which is how a subject's attribute is matched.  Values lose their quotes.

    **``None`` for everything that does not require a value**, and that
    includes a negation: ``status<>missing`` lands on a row holding almost
    anything, so it says which value the write refuses rather than which one it
    moves the row off.  A bind nobody settled, a null test, a comparison and a
    term spelled through an interpolation are ``None`` for the plainer reason
    that no value can be read off them.
    """
    text = term.strip()
    match = _REQUIRES_ONE.match(text)
    if match:
        return match.group(1).rsplit(".", 1)[-1].lower(), [match.group(3)]
    match = _REQUIRES_ANY.match(text)
    if match:
        values = [part.strip().strip("'") for part in match.group(2).split(",")]
        if values and all(value and ":" not in value for value in values):
            return match.group(1).rsplit(".", 1)[-1].lower(), values
    return None


#: What a read site files for an exclusion whose values the map cannot name --
#: ``NOT IN (SELECT ...)``, a hole, a bind nothing settles.
UNNAMED_EXCLUSION = "{}"


def admits_by_exclusion(site: "ReadSiteNode", subject: str, value: str) -> bool:
    """Whether *site* returns rows holding *value* by passing over other values.

    ``getQueuedPandaIDsWithTask_JEDI`` asks for ``jobStatus NOT IN (holding,
    running, transferring)``: it never names ``defined`` and picks up every
    defined job of the task.  It is not a reader of ``defined`` -- the map's
    readers name the value, and counting this one would make every value but
    three selected by it -- so it is answered beside the readers, not among them.

    True only when the statement says nothing else about the subject:

    * it excludes some value of the subject, and none of those is *value*;
    * none of them is :data:`UNNAMED_EXCLUSION` -- a ``NOT IN (SELECT ...)``
      may pass over *value* too, and "it is not excluded" would then be a
      confident wrong answer;
    * it does not select on the subject, by a value or by a key settled at run
      time (``keys``): ``jobStatus<>:a AND jobStatus=:b`` returns the rows
      ``:b`` names, which nothing here says is *value*.

    Not a claim that a row holding *value* comes back.  The rest of the
    ``WHERE`` still applies, and so does the table: ``jobsArchived4`` holds no
    ``defined`` job whatever its ``NOT IN`` passes over.
    """
    prefix = f"{subject}="
    passed_over = [pair[len(prefix):] for pair in site.excludes if pair.startswith(prefix)]
    if not passed_over or value in passed_over or UNNAMED_EXCLUSION in passed_over:
        return False
    if any(pair.startswith(prefix) for pair in site.selects):
        return False
    return subject not in site.keys


# How work arrives at a junction.  Declared with the field that carries it
# rather than in the recognizer that assigns it, because both halves of the map
# read them: the extraction to label an entry, and an investigation to decide
# whether a stalled value will come back on its own.
POLLED = "polled"
COMMAND = "command"
MESSAGE = "message"
REQUEST = "request"

#: Triggers that re-evaluate.  A subject only these can reach recovers by
#: itself, so waiting is a plan; a subject only the others can reach is
#: consumed once, and a missed one stays missed.
SELF_REPAIRING_TRIGGERS = frozenset({POLLED})


#: How control got in.  Three shapes, found three ways: a call is resolved by
#: method name, a dispatch is a constructed worker being started, and a door is
#: a facade method forwarding to a borrowed proxy.  Their ``arg_binding`` keys
#: come from different namespaces -- and a door's are not read at all -- so the
#: set is what makes each comparable only with its own kind.
ARRIVES_BY_CALL = "call"
ARRIVES_BY_DISPATCH = "dispatch"

#: Through a facade.  ``api/v1/job_api`` calls ``TaskBuffer.storeJobs``, which
#: calls ``proxy.insertNewJob`` -- the edge is real and the arguments are not
#: this entry's, since the door rewrites them.  Kept apart from ``call`` so
#: that "nothing was read here" is never compared against "nothing was passed".
ARRIVES_THROUGH_DOOR = "door"


class EntryPoint(BaseModel):
    """One way control reaches a junction, and what it hands over on the way.

    Plural on purpose.  ``getTasksToBeProcessed_JEDI`` is called by
    ``JobGenerator`` with ``minPriority`` and ``maxNumJobs`` and by the message
    processor without them, so the throttle guard cannot fire on the message
    path at all -- the set of reasons a task was not picked up differs by entry,
    which makes the entry part of the structure rather than context.
    """

    trigger: str = Field(
        ...,
        description=(
            "polled | command | message.  How work arrives, which decides "
            "whether missing it repairs itself: a loop re-evaluates, a command "
            "row and a broker message are each consumed once."
        ),
    )
    entry: str = Field(..., description="Module that carries the trigger.")
    via: Optional[str] = Field(
        default=None,
        description="Method through which the entry reaches this junction, if not its own.",
    )
    arg_binding: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "What this entry hands over, keyed by the name the callee's body "
            "reads it under -- the keyword for a call, the field for a "
            "constructed worker.  Comparable only within one ``reached_by``: "
            "the two are different namespaces and an empty set means "
            "'positional, not bound' on one side and 'handed nothing' on the "
            "other.  **One site's bindings, and ``sites`` says how many there "
            "were**: an entry is keyed on the door, so several constructions in "
            "one function fold into this row and the values here are the last "
            "one read."
        ),
    )
    sites: int = Field(
        default=1,
        description=(
            "How many places hand over this way.  The key is ``(trigger, entry, "
            "via, reached_by)`` because that is a property of the door, so "
            "``copyArchive.py::main`` building a ``Watcher`` eight times with "
            "four different ``sleepTime`` values is one entry -- and until this "
            "field the row presented the last one read as though it were the "
            "handover.  Above one means ``arg_binding`` is a sample: which site "
            "goes with which statement is a question about the statement, and "
            "the read side is where that is answered."
        ),
    )
    reached_by: str = Field(
        default=ARRIVES_BY_CALL,
        description=(
            "``call``, ``dispatch`` or ``door``.  A call is resolved by "
            "method name; a dispatch is a knight constructing a worker and "
            "starting it, which no name match reaches; a door is a facade "
            "method forwarding to a borrowed proxy, whose arguments are the "
            "door's rather than this entry's and are left unread.  Recorded "
            "because the three bind their arguments differently, and comparing "
            "one against another reports a difference in spelling -- or an "
            "unread list -- as a difference in what the path can do."
        ),
    )


class DispatchFanout(BaseModel):
    """The classes a run-time dispatch could have chosen, and how to tell.

    Not an attribution and never one.  P1-11 refused to say *which* spec class
    a run-time receiver is, because a single wrong answer invents a junction;
    this is the other shape -- "one of these three" -- which the map already
    trades in everywhere else.  The set is a candidate set and the run is
    settled by the log line beside it.

    Read from what the code declares rather than from a naming convention.
    The dispatch asks a config for a class and falls back to a concrete one it
    imports; that concrete class names the interface through its base, and the
    base's subclasses in the corpus are the alternatives.  Measured: the
    structural shape alone -- assign a call, then ``if x is None: x = y`` --
    matches 29 sites and most of them are ordinary defaults
    (``maxHS06sec``, ``coreCount``, ``newScanList``).  Requiring the fallback
    to be a *class the corpus declares* leaves 3, which are the three
    ``getPlugin`` sites.
    """

    at: Optional[int] = Field(default=None, description="Line of the dispatch.")
    selector: str = Field(
        default="", description="The call that asks for a class, as written."
    )
    default: str = Field(
        default="", description="The concrete class the source falls back to."
    )
    base: str = Field(
        default="",
        description=(
            "Its base, which is what names the interface.  Empty when the "
            "default declares none -- the fan-out is then that one class, "
            "which is a true answer and a narrow one."
        ),
    )
    candidates: list[str] = Field(
        default_factory=list,
        description="The base's subclasses in this corpus, the default included.",
    )
    announced_by: str = Field(
        default="",
        description=(
            "The literal text of the line that prints the chosen class, when "
            "the code prints one.  This is what turns a candidate set into an "
            "answer at use time, and it is the third leg the plan asked for: "
            "candidates at build, run-time attribute at use, log for proof."
        ),
    )


class JunctionNode(BaseNode):
    """A place where the code settles a subject's value.

    ``name`` is the semantic signature, which must survive refactoring: the
    same junction keeps its identity when its file, line, or enclosing module
    changes.  Positions live in ``anchor``.
    """

    node_type: NodeType = NodeType.JUNCTION_POINT
    map_id: str
    derived_from: str
    subject: str = Field(..., description="Qualified subject name this junction writes.")
    owner: str = Field(..., description="module::qualname that contains the write.")
    log_files: list[str] = Field(
        default_factory=list,
        description=(
            "Log files the module holding this junction writes to, e.g. "
            "'panda-ContentsFeeder.log'.  More than one when the code runs in "
            "more than one process: the proxy mixins land in "
            "'panda-DBProxy.log' under the server and 'panda-JediDBProxy.log' "
            "under a knight, and which one is a runtime fact, so both are named."
            "  This is where the code *lives*, which is not always where a line "
            "about it firing appears -- see ``caller_log_files``."
        ),
    )
    caller_log_files: list[str] = Field(
        default_factory=list,
        description=(
            "Log files belonging to the code that reaches this junction.  A "
            "separate field rather than more entries in ``log_files`` because "
            "it answers a different question, and merging them would repeat "
            "the mistake it exists to correct: a ``db_proxy_mods`` method "
            "declares no logger of its own, so ``log_files`` names the mixin's "
            "two proxy files while the diagnostic -- 'set task_status=' -- is "
            "written by the knight that called it.  Production confirms the "
            "split: of 33 files asked, that line is in exactly five -- "
            "ContentsFeeder, JobGenerator, PostProcessor, TaskCommando and "
            "TaskRefiner -- and in neither proxy file."
        ),
    )
    owns_logger: bool = Field(
        default=True,
        description=(
            "Whether the module holding this junction declares its own logger. "
            "False means ``log_files`` are the files of the classes that mix it "
            "in -- true for where its own output lands, since the SQL comment "
            "trace does appear there, but *not* a place a line about this "
            "junction firing can appear, because the code that writes such a "
            "line is the caller.  The difference decides whether an empty grep "
            "may be read as 'it did not fire': asking ``panda-DBProxy.log`` for "
            "``set task_status=`` returns nothing however often the junction "
            "runs, so an eliminator that did not know this would rule out every "
            "proxy candidate at once and confidently keep the wrong one."
        ),
    )
    attribution: str = Field(
        default="certain",
        description=(
            "How the subject's class was settled: ``certain`` (the code states "
            "the type), ``structural`` (only one class declares every "
            "attribute the code touches on the object), ``heuristic`` (the "
            "variable's name, narrowed by the module's imports), or "
            "``unresolved``.  A separate axis from ``Branch.tier`` on purpose "
            "-- tier is about whether the *outcome* is statically known, and a "
            "write can state its outcome as a literal while leaving the class "
            "it wrote to open.  Recorded so a guessed attribution is visible "
            "rather than indistinguishable from a stated one."
        ),
    )
    structural_subject: Optional[str] = Field(
        default=None,
        description=(
            "What the attributes touched on the object imply, derived "
            "independently of how ``subject`` was settled.  Kept even when it "
            "merely agrees, because agreement is the point: the declaration "
            "and the usage are two expressions of one fact, and comparing them "
            "is a gate that needs no production data."
        ),
    )
    branches: list[Branch] = Field(default_factory=list)
    entry_points: list[EntryPoint] = Field(
        default_factory=list,
        description=(
            "How this junction is reached.  Empty means nothing the map "
            "recognises starts it -- reported rather than defaulted, since "
            "assuming a loop would make the self-repair property unusable."
        ),
    )
    anchor: Optional[Anchor] = None
    calls: list[str] = Field(
        default_factory=list,
        description=(
            "What the owner consults, as ``module::method`` targets.  The one "
            "edge out of a junction that reaches the rows an arm decided on: "
            "``setScoutJobData_JEDI`` writes ``exhausted`` and "
            "``getScoutJobData_JEDI`` selects on finished jobs, with nothing "
            "but the call between them.  Qualified because a bare name is not "
            "an identity here and the reader joins on this -- which module a "
            "call lands in is settled at build time, where a name is followed "
            "only if it means one thing or the caller imports the module it "
            "names.  Recorded rather than joined at read time through the "
            "shared owner, because ``selected_by`` is per function and a "
            "method handling several commands reads one thing in one arm and "
            "writes another in another; treating that as a relation is the "
            "mistake this corpus has already charged for twice."
        ),
    )
    called_from: list[str] = Field(
        default_factory=list,
        description=(
            "``file:line`` of each call placed at the owner, sorted -- and, "
            "where that call is made by a function that is itself called, "
            "``outer file:line -> inner file:line``, one hop further.  "
            "``activateJob`` is called only by ``TaskBuffer.activateJobs``, and "
            "what moves a job to ``activated`` is who calls that: the setupper, "
            "the activator, the dataset manager.  Placed by what the receiver "
            "is declared to be (``calls.Calls.callers``), not by name.  Every "
            "chain is listed, so a function found in none does not call the "
            "owner within these two hops; one further out is not listed."
        ),
    )
    joined_entities: list[str] = Field(
        default_factory=list,
        description=(
            "Kinds of row one of this owner's queries reads *in the same "
            "statement* as the rows it decides about.  The sound form of the "
            "join that is unsound at owner granularity: sharing a function "
            "proves nothing, but a single ``FROM`` list is the corpus stating "
            "the relation itself, so ``prepareTasksToBeFinished_JEDI`` "
            "selecting tasks against their datasets really does say which "
            "datasets that task waits on.  The difference is the whole point "
            "-- 113 junctions would gain an entity from co-residence in the "
            "function and 31 do from a shared statement."
        ),
    )
    dispatch: list[DispatchFanout] = Field(
        default_factory=list,
        description=(
            "Run-time class choices made inside this junction's class.  A "
            "property of the class rather than of the arm, the same scope "
            "``self.<field>`` has: ``AdderGen`` picks its plugin in "
            "``get_plugin_class`` and runs it from ``process_job_report``, "
            "which is where the arm is."
        ),
    )
    gloss_key: str = Field(
        default="",
        description=(
            "Hash of the enclosing function's text -- the key of the reading "
            "that explains it.  Computed at build time so the input a reader "
            "would be given is chosen by the map and not by whoever asks; the "
            "reading itself is filled in lazily.  Empty when the function "
            "could not be located in the snapshot."
        ),
    )

    def observable_log_files(self) -> list[str]:
        """Files where a line *about this junction firing* can appear.

        The caller's, plus its own only where its module declares a logger.  An
        inherited file is deliberately left out: it is a true statement about
        where the code's own output lands -- the SQL comment trace really is in
        ``panda-DBProxy.log`` -- and a false one about where a line saying the
        junction fired appears, because the code writing that line is the
        caller.  Production settles it: of thirty-three files asked,
        ``set task_status=`` is in exactly five, all of them callers, and in
        neither proxy file.

        A method on the node rather than a rule in one reader, because two
        readers need it and they would drift.  Deriving an investigation asks
        it to know where to grep; the gate that reports dead code paths asks it
        to know whether a junction has any surviving log at all, and answering
        that from ``log_files`` alone credits a proxy method with the liveness
        of the mixin that inherits it.
        """
        files = set(self.caller_log_files)
        if self.owns_logger:
            files.update(self.log_files)
        return sorted(files)

    @staticmethod
    def make_name(map_id: str, subject: str, signature: str) -> str:
        """Build the semantic signature used as the merge key."""
        return f"{map_id}:{subject}:{signature}"

    def content_hash(self) -> str:
        """Hash of the junction's *meaning*, used to key derived artefacts.

        Deliberately excludes the anchor.  A gloss cached against a file
        position would be regenerated every time an unrelated edit shifts the
        line numbers; keyed on content it survives until the logic itself
        changes.
        """
        payload = {
            "subject": self.subject,
            "branches": [
                {
                    "outcome": b.outcome,
                    "path_condition": b.path_condition,
                    "tags": b.tags,
                    "emits": [e.model_dump() for e in b.emits],
                }
                for b in self.branches
            ],
        }
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


class LogSiteNode(BaseNode):
    """Where one function's diagnostics land, for a function that owns no junction.

    The map already answers this for a junction, and a junction is not who is
    asked.  Two thirds of the owners the map names as reading a value own no
    junction at all -- ``getPandaIDsWithTask_JEDI`` selects a task's jobs and
    settles nothing -- so ``_follow_up`` looks the reader up among the writers,
    misses, and says nothing about which log will show the query running.
    Measured on the installed corpus: 279 reader and writer owners, 94 of them
    junctions, and all 185 of the rest resolve to a file.

    **Only for owners that are not junctions.**  One fact in two places is how
    ``log_files`` came to answer both "where does this code live" and "whose
    log mentions it"; a reader checks junctions first and these second, so
    nothing is stored twice.  Since promotion marks subjects and drops none,
    every function writing a subject owns a junction: on pin k 328 rows became
    228, each owner of the 100 now a junction's, with the same ``log_files``.

    **Stored, not merely reported.**  ``DiagnosticTemplate`` and
    ``EnumerationWrite`` are built on every run, carry no label and reach
    nobody -- the database holds none of their rows.  An investigation reads
    this to decide where to grep, so it is a node.

    Scoped to the owners the map names.  Every function in a module with a
    resolvable logger could have a row, but a row for a function nothing in
    the map points at is vocabulary added before a reader asked for it.
    """

    node_type: NodeType = NodeType.LOG_SITE
    map_id: str
    derived_from: str
    owner: str = Field(
        ...,
        description=(
            "``module::function``, the same spelling a junction's ``owner`` "
            "and a subject's ``selected_by`` use, because that is what the "
            "reader joins on.  Also the node's ``name``: here the owner is the "
            "identity, there being nothing else to be."
        ),
    )
    triggers: list[str] = Field(
        default_factory=list,
        description=(
            "How this reader's module is started, from the same classification "
            "a junction's entry points carry.  Without it ``_follow_up`` falls "
            "back to the *writers'* triggers to answer whether anything will "
            "re-evaluate the row -- and for a value whose only re-evaluating "
            "reader is a daemon script that settles nothing, that turns "
            "``polled`` into ``command, request`` and sends the reader to ask "
            "whether a command arrived.  A cadence, not an entry point: a log "
            "site has no arm for an argument to be bound at."
        ),
    )
    log_files: list[str] = Field(
        default_factory=list,
        description=(
            "Files this module's own output can reach, declared first and "
            "otherwise inherited through the classes that mix it in.  A list "
            "because the answer is genuinely two for a proxy method: the "
            "server's log when the server calls it and JEDI's when a knight "
            "does."
        ),
    )
    caller_log_files: list[str] = Field(
        default_factory=list,
        description=(
            "Files belonging to the modules that reach this function.  For a "
            "proxy method this is where a line about the call actually appears "
            "-- the mixin declares no logger, and the knight that called it "
            "does."
        ),
    )
    owns_logger: bool = Field(
        default=False,
        description=(
            "Whether ``log_files`` is declared by this module or inherited.  A "
            "reader that cannot tell the two apart has no way to know that "
            "asking the proxy files for a caller's line always returns "
            "nothing, and would read that nothing as 'it never ran'."
        ),
    )

    def observable_log_files(self) -> list[str]:
        """Files where a line about this function running can appear.

        **Its own file counts here, and on a junction it does not.**  The two
        rules differ because the two nodes are asked about different lines.  A
        junction is asked where the line saying it *settled a value* appears,
        and for a proxy method that line is the knight's -- production settles
        it, ``set task_status=`` is in five files and neither proxy file is one
        of them.  A reader is asked where the line saying its *query ran*
        appears, and that one it writes itself: of the 53 owners here whose
        only file is inherited, 51 emit into it, up to ten calls apiece
        (``getScoutJobData_JEDI`` six, ``toEnableJumbo_JEDI`` eight).  Applying
        the junction's rule said "nothing about this can be seen anywhere"
        about code that logs on every line of its body.

        Both inherited candidates are named.  Which of the two a line landed in
        depends on which process ran the code, which is a run-time fact, and
        naming both is the true answer rather than an ambiguity to resolve.
        """
        return sorted(set(self.log_files) | set(self.caller_log_files))


class GuardRead(BaseModel):
    """A name a guard reads, and where a caller's value for it comes from."""

    name: str
    parameter: str = Field(
        default="",
        description=(
            "``<function>.<parameter>`` where *name* is a parameter of the function "
            "the guard is in, or of the function it is nested in; empty for a local."
        ),
    )
    at_site: str = Field(
        default="",
        description=(
            "``<function>.<parameter>`` of the read site's own function the value "
            "arrives through unchanged -- what a call site's constants or a printed "
            "run settle -- or empty where it does not: another expression is "
            "handed on the way, or a function on the way reassigns the name "
            "(``via`` says which)."
        ),
    )
    via: list[str] = Field(
        default_factory=list,
        description=(
            "The hops from the guard's function out: ``construct_where_clause."
            "prod_source_label <- prodSourceLabel@2329``, ``build_where_clause: "
            "prodSourceLabel is getJobs's``; the last says why it stops where it "
            "stops short of the read site."
        ),
    )
    bound: list[str] = Field(
        default_factory=list,
        description=(
            "For a local, what assigns it: ``random.randrange(100)@839`` -- a value "
            "computed at run time, said rather than settled."
        ),
    )


class TermGuard(BaseModel):
    """One test a term's fragment is appended under."""

    kind: str = Field(
        ...,
        description=(
            "``if`` -- the test holds; ``else`` -- it does not (an ``else``, or an "
            "earlier branch of the ``elif`` chain a branch hangs from); ``loop`` / "
            "``except`` -- the loop body or the handler runs."
        ),
    )
    test: str = Field(
        ...,
        description=(
            "As the source writes it; ``not (<test>)`` for an ``else`` or a later "
            "``elif`` -- each earlier branch of a chain is its own entry; ``a loop "
            "at <line> runs`` / ``an except at <line> runs`` for those."
        ),
    )
    at: str = Field(..., description="``<file>:<line>`` of the ``if``, loop or handler.")
    function: str
    reads: list[GuardRead] = Field(default_factory=list)


class TermPlace(BaseModel):
    """One way a term gets into a statement: the fragment appending it and every test above it.

    A term whose text several fragments make -- ``jobStatus IN ('activated',
    'sent', 'starting')``, a template's words and a list one branch writes --
    is there where every one of them is, so it is under every test above any
    of them.  What a loop's passes write where the statement holds them as
    one repeated part is not among them: any number of passes holds the term.
    """

    fragment: str = Field(
        ...,
        description=(
            "``<file>:<line>`` of the statement that appends the text; of several, "
            "the one the most tests are above -- the one a branch chooses."
        ),
    )
    term: str = Field(
        default="",
        description=(
            "The term the entry enters the statement in -- one of the terms the "
            "``WHERE`` ``AND``\\ s together -- spelled as ``conditions`` spells it: "
            "``processingType=:processingType1 OR prodSourceLabel IN (...)`` for a "
            "pair one branch of an ``OR`` with another column holds; "
            "``status=:status`` for a pair ``conditions`` leaves to ``selects``.  "
            "Empty where the ``WHERE`` does not read term by term: not known, which "
            "is not the same as outside an ``OR``."
        ),
    )
    decides: Optional[bool] = Field(
        default=None,
        description=(
            "For a ``selects`` or ``excludes`` pair, whether ``term`` rules out every "
            "row the pair's field does -- for ``selects`` a row holding none of the "
            "subject's values, for ``excludes`` a row holding the value.  ``AND`` "
            "decides where one side does, ``OR`` where every branch does: "
            "``(modificationTime<:t AND jobStatus=:a) OR (stateChangeTime<:t AND "
            "jobStatus=:b)`` decides ``jobStatus``, ``processingType=:pt OR "
            "prodSourceLabel IN (...)`` decides neither column, and ``relocationFlag "
            "IS NULL OR relocationFlag<>:r`` decides ``relocationFlag``.  ``False`` "
            "is a pair the builds through this place only *allow*.  Read off the "
            "statement whatever ``term`` could be spelled as.  ``None`` for "
            "``conditions`` and ``order_by``, which are not pairs."
        ),
    )
    tests: list[TermGuard] = Field(
        default_factory=list,
        description=(
            "Outermost first, all of them holding: the ones between where the "
            "statement's text is used and the ``execute`` first, then each function "
            "the text passed through, out to in, ending in the fragment's own.  A "
            "guard above the ``execute`` holds for every build and is not here.  "
            "Of a term several fragments make, each one's, in the order they are "
            "in the statement, each test once."
        ),
    )


class AddedTerm(BaseModel):
    """A term a statement's fields hold on some builds of it only."""

    field: str = Field(..., description="``selects``, ``excludes``, ``conditions`` or ``order_by``.")
    term: str = Field(..., description="As that field spells it.")
    places: list[TermPlace] = Field(default_factory=list)
    unplaced: str = Field(
        default="",
        description=(
            "Why no place is given, when none is: the term is said rather than "
            "dropped.  ``no test chooses it`` -- the call runs statements whose "
            "difference no test above a fragment makes, such as a text passed "
            "through ``re.sub``; ``no build holds it`` -- every statement putting "
            "the value into its bind is on a branch the fragment naming the bind "
            "is not, so the pair is in the field and in no build the map can see; "
            "``the fragment appending it was not found``."
        ),
    )


class ConditionValue(BaseModel):
    """One value a term compares its column with, as far as the map settles it.

    A time is a *base* -- the clock -- plus an offset: ``modificationTime
    <:timeLimit`` with ``timeLimit = naive_utcnow() - timedelta(days=7)`` at
    the line that reaches the bind is the base ``naive_utcnow`` and
    ``-604800``.  Where the source does not settle the value here, ``needs``
    names the parameters the offset waits for -- a call's record fills them --
    or ``said`` gives the expression, as :attr:`GuardRead.bound` does.
    """

    kind: str = Field(
        ...,
        description=(
            "``bind`` -- the value a bind is filled with; ``constant`` -- one the "
            "statement writes; ``column`` -- another column of the row; ``clock`` "
            "-- the database's own clock, ``CURRENT_DATE-3``; ``other``."
        ),
    )
    text: str = Field(..., description="This side as the statement spells it: ``:timeLimit``, ``CURRENT_DATE-3``.")
    base: str = Field(
        default="",
        description=(
            "What the offset is added to: a clock -- the database's "
            "(``CURRENT_DATE``, ``SYSDATE``, ``CURRENT_TIMESTAMP``, "
            "``SYSTIMESTAMP``) or Python's, named by what the call resolves to "
            "through the module's imports (``naive_utcnow``, ``datetime.utcnow``, "
            "``datetime.now``, ``datetime.now(utc)``) -- or "
            "``<function>.<parameter>``, a value the call hands in.  Empty for a "
            "plain constant.  A call reading any other clock is ``said``.  Clocks "
            "are named, not equated: PanDA's Oracle session runs in UTC "
            "(``WrappedCursor.initialize``), ``SYSDATE`` in the database host's "
            "zone, ``datetime.now`` in the server's."
        ),
    )
    offset_seconds: Optional[float] = Field(
        default=None,
        description=(
            "Added to ``base``: seconds for a time, the amount itself for a "
            "number handed in as a parameter.  ``None`` where it is not settled."
        ),
    )
    constant: str = Field(
        default="",
        description="A settled value with no base, spelled as ``selects`` spells values: ``800``, ``activated``, ``None``.",
    )
    scale_seconds: Optional[float] = Field(
        default=None,
        description=(
            "Seconds per unit of a bind amount the statement adds to a database "
            "clock: ``CURRENT_DATE-:days`` is ``-86400`` times the bind's value."
        ),
    )
    subject: str = Field(default="", description="A ``column`` side's subject, spelled as ``ConditionNode.subject``.")
    needs: list[str] = Field(
        default_factory=list,
        description="``<function>.<parameter>`` for each value the offset waits for: ``lockJobOutputReport.time_limit``.",
    )
    said: str = Field(
        default="",
        description=(
            "``<expression>@<line>`` where the source does not settle the value: "
            "a configuration lookup, several writes reaching the bind, a bind "
            "nothing in the map fills."
        ),
    )


class ConditionNode(BaseModel):
    """One node of a condition read as a tree: :attr:`ReadSiteNode.condition_trees`."""

    node: str = Field(
        ...,
        description=(
            "``any`` -- holds where one of ``items`` does; ``all`` -- where every "
            "one does; or a term: ``compare`` (``< <= > >=``), ``equal`` (``=``, "
            "``IN``), ``unequal`` (``<>``, ``NOT IN``), ``null``, ``not_null``, "
            "``other`` -- what does not read as one column tested against values "
            "(an expression on the left, a ``NOT EXISTS``), kept so that the "
            "entry is never read as decided without it."
        ),
    )
    items: list["ConditionNode"] = Field(default_factory=list)
    text: str = Field(default="", description="A term as the statement writes it.")
    subject: str = Field(
        default="",
        description=(
            "The column the term tests, as ``selects`` spells subjects: "
            "``JobSpec.modificationTime``, or ``jobs_StatusLog.modificationTime`` "
            "for a table no spec holds -- the alias never: a joined row is not "
            "the job's."
        ),
    )
    op: str = Field(
        default="",
        description=(
            "``<``, ``<=``, ``>``, ``>=`` with the column on the left (turned "
            "round where the statement writes ``:t > col``), ``=``/``IN``, "
            "``<>``/``NOT IN``."
        ),
    )
    values: list[ConditionValue] = Field(
        default_factory=list,
        description=(
            "What the column is compared with: each value a run can use, so a "
            "bind filled differently on two paths is two, and an ``IN`` list "
            "one per member and value."
        ),
    )
    settled: bool = Field(
        default=False,
        description="Every bind of the term is filled with a value the map settles, and nothing in it is ``other``.",
    )


class ConditionReading(BaseModel):
    """One ``conditions`` entry, or ``added_under`` conditions term, and its tree."""

    entry: str = Field(..., description="The entry exactly as ``conditions`` or ``added_under`` spells it.")
    tree: ConditionNode


class PrintedBy(BaseModel):
    """A logging call that prints this statement's text and the bind map it runs with.

    ``getJobs`` logs ``sqlP + comment + str(getValMap)`` the line before it
    executes ``sqlP + comment`` with ``getValMap``: the record is the
    statement that ran, then its binds.  Said only where that holds -- the
    print's text is the executed text (or it less the trailing comment), its
    map is the executed map, and neither is rebound between the two -- so a
    reader never takes another statement's print for this one's.
    """

    line: int = Field(..., description="The logging statement's line: :attr:`PrintStatement.line` of the function's prints.")
    where: str = Field(..., description="``before`` or ``after`` the ``execute``.")
    text: str = Field(
        ...,
        description="``whole`` -- the executed text; ``without comment`` -- it less the trailing ``/* ... */`` comment.",
    )
    lead: str = Field(
        default="",
        description=(
            "What the message writes just before the statement: ``running sql: ``.  "
            "The statement starts at its first occurrence -- :func:`sqltext.cut`."
        ),
    )
    gap: str = Field(default="", description="What the message writes between the statement and the map: ``\" \"``, ``\" var:\"``.")


class OrderHole(BaseModel):
    """A blanked item of an ``ORDER BY`` -- one the source builds at run time -- and what builds it.

    ``getJobs`` sorts by ``(CASE {} ELSE {} END)``: ``getCriteriaForGlobalShares``
    joins one ``WHEN gshare=:leave{i} THEN {i}`` per share, in the order
    ``get_sorted_leaves()`` returns them.  Which column the ranking is over
    is in the source; the ranking itself is not.
    """

    order: str = Field(..., description="The ``order_by`` entry, as that field spells it.")
    item: int = Field(..., description="The blanked item's place in the list, from 0 -- :func:`sqltext.order_items`.")
    subjects: list[str] = Field(
        default_factory=list,
        description="The columns the item ranks by, as ``selects`` spells subjects: ``JobSpec.gshare``.",
    )
    templates: list[str] = Field(
        default_factory=list,
        description=(
            "What fills the hole, once per member, with its own holes blanked: "
            "``WHEN gshare=:leave{} THEN {}``, and ``... THEN 0`` for the share "
            "the source puts first."
        ),
    )
    computed_at: str = Field(
        default="",
        description=(
            "Where the members, and so the ranking, come from at run time: the "
            "loop building them and what it walks -- ``entity_module.py:874 "
            "for leave in sorted_leaves (sorted_leaves = self.get_sorted_leaves()@867)``."
        ),
    )


class OpenCrossing(BaseModel):
    """A call on the way a statement's rows are carried, or back, that the corpus's calls cannot place: a hole of the reach record.

    Named as the record names the hole -- the function's ``owner`` in
    ``reach_graph``, the ``line`` and the ``callee`` -- so the why is read
    there: ``taskSpec.setErrDiag(...)`` in ``ContentsFeeder.feed_contents_to_tasks``,
    whose ``taskSpec`` is unpacked from a call's result and declared nowhere;
    ``self.tbif.insert_step_task(...)`` in ``PandaTaskStepHandler.submit_target``,
    which may be a caller ``insert_step_task``'s rows go back to, its receiver
    declared ``Any``.
    """

    owner: str = Field(..., description="The ``reach_graph`` row holding the call: ``file::Class.method``.")
    line: int
    callee: str = Field(..., description="As :attr:`ReachHole.callee` spells it.")


class JoinPair(BaseModel):
    """A column of one source of a statement compared equal with a column of another (:func:`sqltext.equalities`):
    how two tables meet in it -- ``tabD.jediTaskID=tabT.jediTaskID``.  Sides sorted by *a* / *b*.  A composite key is
    the pairs of one statement sharing one (*a_source*, *b_source*) and one pair of tables (*a*'s and *b*'s): two
    aliases of one table joined to another twice are two keys, and so are one pair of aliases over other tables in
    another text or arm (``getUserJobMetadata``'s ``j`` / ``m``, ``jobsArchived4`` with ``metatable`` and
    ``jobsArchived`` with ``metatable_arch``; a ``UNION``'s arms).  The pairs of a line's texts and arms are one list,
    so two arms joining the same two tables under the same aliases by other columns would read as one key: none on
    pin k."""

    a: str = Field(..., description="``table.column``, lowered: ``jedi_datasets.jeditaskid``.")
    b: str
    a_source: str = Field(
        ..., description="The source *a* is read from as the statement names it -- its alias, or the table -- lowered."
    )
    b_source: str
    a_attr: str = Field(default="", description="*a* as a subject where its table's spec class has it, else empty.")
    b_attr: str = ""


class CallReading(BaseModel):
    """What one statement reads when one route of calls runs it: :attr:`ReadSiteNode.call_readings`.

    A route is the call into the statement's function and, outward, each
    call that hands it its caller's own parameter unchanged, ending where the
    values are written down.  Read the way the statement's own fields are
    read, with what the route fixes: the constants that decide which
    fragments and binds run, the values it hands the parameters -- a list,
    a time limit -- and a callee's default where the route's call does not
    pass the argument.  A parameter of the route's outermost function is not
    fixed: its callers are not asked, and the value stays ``needs``.
    """

    chain: list[str] = Field(..., description="``file:line`` of each call of the route, the call into the statement's function first.")
    constants: dict[str, str] = Field(
        default_factory=dict,
        description="The constants the route hands the parameters the function tests, as Python spells them -- what names the call.",
    )
    not_run: bool = Field(default=False, description="The route's constants rule the statement out: it never runs on this route.")
    open: list[str] = Field(
        default_factory=list,
        description=(
            "``file:line -- reason`` for each call of the route placed open, in "
            "``chain``'s order: it lands where the route goes, or somewhere this "
            "reading cannot name -- ``self.taskBufferIF.X(...)`` where the "
            "interface forwards -- so the route may not run the statement.  "
            "Empty where every call of the route is placed closed."
        ),
    )
    as_statement: bool = Field(
        default=False,
        description=(
            "The route's fields, read in full, are the statement's own -- each "
            "field below compared, none copied.  Every route reading as the "
            "statement does carries no fields of its own."
        ),
    )
    tables: list[str] = Field(default_factory=list)
    selects: list[str] = Field(default_factory=list)
    excludes: list[str] = Field(default_factory=list)
    consults: list[str] = Field(default_factory=list)
    refuses: list[str] = Field(default_factory=list)
    restricts: list[str] = Field(default_factory=list)
    condition_trees: list[ConditionReading] = Field(
        default_factory=list,
        description="The conditions every build of the route requires, as trees with the route's values in them.",
    )


#: :class:`CallReading`'s fields that say which route, not what it reads.
CALL_READING_ROUTE = frozenset({"chain", "constants", "not_run", "as_statement", "open"})

#: :attr:`ReadSiteNode.result`: rows of the statement's tables, values computed from them, or the one and the other.
RESULT_ROWS = "rows"
RESULT_AGGREGATE = "aggregate"
RESULT_VARIES = "varies"


class ReadSiteNode(BaseNode):
    """One statement that asks the database for rows by a value.

    The other half of a state machine, and the half that answers a row which is
    not moving.  A junction says what a value becomes; this says which statement
    would pick the row up and carry it on, and *where that statement is*.
    Until now the map held the reader as a function name only --
    ``SubjectNode.selected_by[value]`` -- which is not enough to send a reader
    anywhere: ``copyArchive.py::main`` selects ``jobStatus='holding'`` from
    three statements with three different time limits, and
    ``getTasksToExecCommand_JEDI`` runs the orphan rescue beside the query it
    exists for.  Named by the function, those are one answer; they are three and
    two.

    **One row per statement, not per subject.**  A statement selects on
    everything its ``WHERE`` names -- measured on the corpus, 189 of the 264
    read sites select on more than one ``(subject, value)`` pair and one names
    twelve subjects -- so keying per subject would put several nodes at one
    ``(owner, line)`` and ``map-identities-are-distinct`` would be right to
    fail.  What the statement asks for is a list on the row.

    **No log files here.**  The owner already has them, on its junction or on
    its :class:`LogSiteNode`, and a reader joins on ``owner``; one fact in two
    places is the mistake ``LogSiteNode`` records having made.

    Not the whole of what a reader needs.  The exits -- which rows the
    predicate refuses, what the code does with the ones it gets, and which of
    those paths print nothing -- are a separate question, and two of the three
    need the walk scoped to this statement rather than to its function.  This
    is the unit that makes asking them possible.
    """

    node_type: NodeType = NodeType.READ_SITE
    map_id: str
    derived_from: str
    owner: str = Field(
        ...,
        description=(
            "``module::function``, the same spelling a junction's ``owner``, a "
            "subject's ``selected_by`` and a log site's ``owner`` use, because "
            "that is what the reader joins on to find the log."
        ),
    )
    selects: list[str] = Field(
        default_factory=list,
        description=(
            "``subject=value`` for every pair this statement asks rows by, "
            "sorted.  Several because one ``WHERE`` names several columns, and "
            "the same subject twice because ``jobStatus IN ('running', "
            "'starting')`` asks for both."
        ),
    )
    excludes: list[str] = Field(
        default_factory=list,
        description=(
            "``subject=value`` for every pair this statement asks rows *not* "
            "to hold -- ``<>``, ``!=``, ``NOT IN`` -- sorted.  Kept apart from "
            "``selects``: ``status<>'done'`` picks up every row but the done ones.  "
            "``subject={}`` where some of the values it passes over cannot be "
            "named -- ``NOT IN (SELECT ...)``, a hole, a bind nothing here fills "
            "-- so that the statement does not read as passing over nothing."
        ),
    )
    refuses: list[str] = Field(
        default_factory=list,
        description=(
            "The part of ``excludes`` no row this statement returns holds, "
            "however it is built, sorted: a conjunct outside any ``OR``, in "
            "every branch of a ``UNION`` and every way the text is assembled, "
            "with a value every run settles.  ``excludes`` says what a term "
            "names; ``relocationFlag<>:relThrottled`` in one branch of an "
            "``OR`` is there and not here, since the other branches return "
            "those rows.  An ``OR`` every branch of which rules the value out "
            "counts: ``relocationFlag IS NULL OR relocationFlag<>:r``."
        ),
    )
    restricts: list[str] = Field(
        default_factory=list,
        description=(
            "The ``selects`` counterpart of ``refuses``, sorted: per subject, "
            "every row this statement returns holds one of the values listed "
            "for it here -- in every branch of a ``UNION``, every way the text "
            "is assembled and every run at the call, the subject is asked for "
            "outright, by a conjunct or by an ``OR`` every branch of which asks "
            "it (:attr:`TermPlace.decides`), and its values are all of theirs.  "
            "A row holding none of them is not returned.  ``selects`` says "
            "what a term names: ``processingType=:pt OR prodSourceLabel IN "
            "(...)`` is there and not here, since a row passes on either "
            "column.  A subject one of whose asking terms has a bind nothing "
            "fills is left out -- the row may hold the value no one named -- "
            "and is in ``keys``."
        ),
    )
    consults: list[str] = Field(
        default_factory=list,
        description=(
            "``subject=value`` for every pair a subquery of this statement checks "
            "rows by -- ``EXISTS (SELECT 1 ... WHERE jobStatus=:st_merging)`` -- "
            "sorted.  Kept out of ``selects``: the rows it looks at are not the "
            "rows it returns, and nothing here picks them up."
        ),
    )
    keys: list[str] = Field(
        default_factory=list,
        description=(
            "Subjects this statement asks rows by with a value chosen at run "
            "time, sorted: ``JobSpec.jediTaskID`` for ``WHERE "
            "jediTaskID=:jediTaskID``, or a bind nothing in the map settles.  "
            "A statement asking by these alone is still a reader of the rows."
        ),
    )
    order_by: list[str] = Field(
        default_factory=list,
        description=(
            "The ``ORDER BY`` clauses the statement can be built with, sorted.  "
            "A row that passes the ``WHERE`` can still lose on order and not "
            "come back when a limit is set."
        ),
    )
    limits: list[str] = Field(
        default_factory=list,
        description=(
            "``<= 10`` / ``<= :njobs`` -- how many rows come back at most, with "
            "a bind's value where the map settles it."
        ),
    )
    result: str = Field(
        default="",
        description=(
            f"``{RESULT_ROWS}`` | ``{RESULT_AGGREGATE}`` | ``{RESULT_VARIES}`` | "
            "empty -- what the statement returns, read off every way it is "
            "built and each arm of a ``UNION``: rows of its tables, or values "
            "computed from them -- an expression it projects calls an "
            "aggregate (``COUNT``, ``MAX`` ..., not under ``OVER``, not in a "
            "subquery), one row per group or one in all.  ``varies``: some "
            "ways aggregate and some do not.  Empty where a way is not read: "
            "past the count of ways every one is drawn for, or a text that "
            "does not parse as a ``SELECT``.\n\n"
            "Said beside ``crosses_to``, not instead of it: an aggregate's "
            "value goes on -- ``increaseRamLimitJobJEDI``'s ``SELECT ramCount, "
            "count(*) ... GROUP BY`` hands the largest ``ramCount`` to "
            "``increaseRamLimitJEDI``, which writes it -- and where it goes is "
            "still the next place to look.  What an aggregate cannot hand on "
            "is a row of a table."
        ),
    )
    call_readings: list[CallReading] = Field(
        default_factory=list,
        description=(
            "What the statement reads on each route of calls into its function "
            "(:class:`CallReading`), every route the map places, each complete "
            "in the statement's own forms or marked as reading as the statement "
            "does.  The statement's own fields are every run any call can make; "
            "``copyArchive`` hands ``lockJobsForReassign`` four status lists at "
            "four calls, and only a route's record says which call asks for "
            "which, and which of them refuse ``relocationFlag=2``.  Whether the "
            "routes are every way in is ``callers_complete``."
        ),
    )
    callers_complete: bool = Field(
        default=False,
        description=(
            "Every call spelled with the name of the statement's function, and "
            "of each forwarder a route passes through, is placed -- so "
            "``call_readings`` is every way the statement is run from the "
            "corpus.  A call placed open is in the list, its route marked "
            "(:attr:`CallReading.open`).  ``False`` can be "
            "said of a complete list, a call by the same name to another "
            "function counting against it; ``True`` is never said of an "
            "incomplete one.  The corpus is the one the map reads: a test "
            "module is not read, so a call from one neither makes a route "
            "nor counts against the list."
        ),
    )
    callers_stop: list[str] = Field(
        default_factory=list,
        description=(
            "Where the list of routes may stop short, when it may: the function, "
            "how many calls by its name are not placed and why, by receiver "
            "(``self.taskBufferIF 3``, the first five receivers and the rest "
            "counted), how many are placed elsewhere and open beside it, or "
            "that no call in the corpus names it -- an endpoint, reached from "
            "outside."
        ),
    )
    tables: list[str] = Field(
        default_factory=list,
        description="The tables the statement reads, as the source spells them.",
    )
    bind_window: list[int] = Field(
        default_factory=list,
        description=(
            "``[first, last]`` -- the lines the forwarded form fills this run's "
            "binds within, and empty for ``execute``, where the binds are the "
            "whole function's.  47 of the corpus's 264 read sites have one, "
            "and ``copyArchive.py::main``'s three ``holding`` queries have "
            "``[374, 381]``, ``[400, 405]`` and ``[429, 433]``.\n\n"
            "Its own field rather than the anchor's span, because the two say "
            "different things: the anchor is where to send a reader, and this "
            "is the region a walk would have to be scoped to -- seeded at the "
            "bind assignments inside it rather than at the statement, whose "
            "value is the SQL text and reads no names at all."
        ),
    )
    conditions: list[str] = Field(
        default_factory=list,
        description=(
            "Every other term the ``WHERE`` requires of a row, sorted -- what "
            "``selects`` does not say.  A row can hold the value the map names "
            "and still not come back, and until now the map had no way to say "
            "why: ``copyArchive.py``'s ``holding`` query also requires "
            "``modificationTime<:modificationTime``, and that is the answer a "
            "reader came for.\n\n"
            "Written as the source writes the term, with ``@<line>`` appended "
            "when a bind fills it -- ``modificationTime<:modificationTime@375`` "
            "-- and nothing appended when none can, which is the map saying the "
            "term is real but not walkable (a ``NULL`` check, a column compared "
            "against another column).\n\n"
            "**The list reads as a conjunction, so an ``OR`` group is one "
            "entry** with its branches joined by `` OR ``.  Split into separate "
            "entries, ``(tabT.status=:running OR tabT.status=:scouting)`` asks "
            "for a row whose status is both, which no row can be; 124 of the "
            "corpus's 473 terms are branches of such a group.\n\n"
            "Join terms are left out.  ``c1.jediTaskID=d.jediTaskID`` says how "
            "two tables meet, not why a row was passed over, and the 182 of "
            "them would bury the terms that do."
        ),
    )
    order_holes: list[OrderHole] = Field(
        default_factory=list,
        description=(
            "Each blanked ``order_by`` item the source builds at run time, with "
            "the column it ranks by and where the ranking is computed.  A reader "
            "comparing a row with the rows sorted ahead of it groups by these "
            "subjects; without them ``getJobs``' share ranking read as an order "
            "by priority alone."
        ),
    )
    printed_by: list[PrintedBy] = Field(
        default_factory=list,
        description=(
            "The logging calls in the statement's own function that print its "
            "text and then the map it is executed with -- where to find the "
            "statement that ran, and its binds, in a log.  50 read statements "
            "of the corpus print ``<sql> + comment + str(<map>)``; most print "
            "nothing of their SQL, and this is empty for them."
        ),
    )
    condition_trees: list[ConditionReading] = Field(
        default_factory=list,
        description=(
            "Each ``conditions`` entry, and each ``conditions`` term of "
            "``added_under``, as a tree a row can be evaluated against: an "
            "``any`` of its branches, each an ``all`` of its terms, each term "
            "saying which subject it tests and against what.  The entries are "
            "text a reader would otherwise parse; ``tabT.modificationTime<"
            ":timeLimit@4143`` and ``lockedTime IS NULL OR lockedTime<"
            ":timeLimit@1822`` were out of reach of a pattern that took "
            "``col<:bind@line``, and so was the time the bind holds.\n\n"
            "A bind's value is read through the writes that reach it: "
            "``copyArchive.py::main`` writes ``timeLimit`` twenty-six times, and "
            "the statement at 381 reads the one at 375.  A parameter's value is "
            "the call's, named in ``needs`` and filled in that call's record; "
            "its default is never taken as fixed, since a caller may pass "
            "another."
        ),
    )
    added_under: Optional[list[AddedTerm]] = Field(
        default=None,
        description=(
            "Every term ``selects``, ``excludes``, ``conditions`` and ``order_by`` "
            "hold on some builds of the statement only, with the tests that add it; "
            "a term not here is in every build.\n\n"
            "``getJobs``' ``WHERE`` is ``construct_where_clause``'s return, and six of "
            "its terms are each appended under an ``if`` there -- ``mem not in [0, "
            "'0']`` for ``minRamCount``.  Read as one piece, every run required them, "
            "and a reader ruled a job out for failing a term its build never "
            "asked.  Those leave ``conditions`` and are here, each with its test, "
            "and each name the test reads that arrives as a parameter followed out "
            "to ``getJobs``' own, so that a call's constants or a printed run can "
            "settle it.  ``selects`` and ``excludes`` keep every pair any build asks "
            "by; this says which of them some builds do not.\n\n"
            "Each build is read on its own, the values in its binds too: a value "
            "counts only where the statement filling the bind can be on one run "
            "with the build's fragments.  ``construct_where_clause``'s branches "
            "reuse ``:prodSourceLabel1..N`` with other labels, and read by key every "
            "branch asked for every label.\n\n"
            "``None`` -- an older build -- is not computed, and not \"every term in "
            "every build\": a consumer that read the two alike would rule rows out "
            "against terms nobody checked."
        ),
    )
    unresolved: list[str] = Field(
        default_factory=list,
        description=(
            "``<expression> -- <reason>`` for every part of this statement the "
            "extraction could not follow, sorted.  Empty is the usual answer "
            "and it is an answer: it says ``conditions`` is the whole "
            "``WHERE``.\n\n"
            "What it is for is that the opposite case looked identical. "
            "``lockJobsForReassign`` selects the jobs a two-day reassign picks "
            "up; its ``WHERE`` reads ``jobStatus IN ({})`` because the states "
            "arrive as an argument, so the map stored the statement with an "
            "empty ``conditions`` and nothing anywhere said the statement was "
            "read only in part. A consumer cannot tell a query that filters on "
            "nothing from one whose filter nobody could read, and the second is "
            "where it should go and read the caller.\n\n"
            "The expression, not a class of expression: ``build_where_clause()``"
            " and ``jobStatus IN ({})`` name what to open next, which is the "
            "whole point of saying it out loud."
        ),
    )
    region: list[int] = Field(
        default_factory=list,
        description=(
            "``[first, last]`` -- the stretch of the function this statement "
            "owns, from where its binds start being filled to just before the "
            "next statement starts filling its own.\n\n"
            "Not the same claim as ``bind_window``, which ends at the call: "
            "what a reader came for is usually what the code *prints after* "
            "the rows come back, and in ``copyArchive.py::main`` every logging "
            "call for the ``holding`` query sits outside the window. The "
            "region is what a skeleton is scoped to, and it is why the three "
            "``holding`` statements stop returning the same 251 rows: they are "
            "``[374, 399]``, ``[400, 428]`` and ``[429, 453]``. Median 21 "
            "lines against the enclosing function's 86, and 363 at worst "
            "against 1353."
        ),
    )
    exits: list[str] = Field(
        default_factory=list,
        description=(
            "Where a row this statement asked for stops being followed, "
            "sorted -- ``\"continue@390\"``, ``\"except@1301\"``, "
            "``\"no-rows:return@263\"``.\n\n"
            "**The prefix separates two different answers.** Without it, a "
            "``continue`` inside the loop over the rows says *this row* was "
            "dropped here -- ``copyArchive.py``'s ``holding`` query hands its "
            "rows to a loop that skips every job whose ID is already in the "
            "XML, which is why a job the map says was selected never reaches "
            "the ``Watcher``. With ``no-rows:``, the exit is under a test on "
            "the result itself and the answer is that nothing came back at "
            "all, which points at the ``conditions`` rather than at the "
            "handling.\n\n"
            "The rows are where the statement's run lands them: an assignment "
            "on its line, one that fetches them or reads ``rowcount`` however "
            "it computes from them (``has_merging = self.cur.fetchone() is not "
            "None``), and a loop over the cursor that ran it (``for (pandaID,) "
            "in self.cur:``), which is a loop over its rows.\n\n"
            "**An ``except`` counts only when the handler leaves.** "
            "``except Exception: pass`` is written so the row is *not* "
            "stopped, and a handler assigning a fallback says the same at "
            "more length; a handler is an exit when it contains a "
            "``continue``, ``break``, ``return`` or ``raise``.\n\n"
            "Read in the loop over *this* statement's rows, which opens "
            "inside ``region`` and runs to its own end -- see "
            "``rows_handled``. Only the exits taken **in that loop**: judged "
            "by line range instead, 19 more come in from loops nested in it, "
            "such as ``copyArchive@206``'s ``continue@245``, which is about "
            "one of a job's files. The function's own control flow is left "
            "out, because an answer naming every error path in the function "
            "does not say where the row went.\n\n"
            "The corpus holds 103 over 57 statements: 90 per row (43 "
            "``continue``, 27 ``raise``, 11 ``break``, 7 ``except``, 2 "
            "``return``) and 13 because nothing came back, 6 of which print "
            "nothing at all. Empty is an answer -- most statements have no "
            "exit of their own -- so there is no coverage gate on this field, "
            "only the invariant that every line named falls inside ``region`` "
            "or ``rows_handled``."
        ),
    )
    rows_handled: list[int] = Field(
        default_factory=list,
        description=(
            "``[first, last]`` -- the stretch where this statement's rows are "
            "worked through, which is the loop over them. Empty where "
            "nothing loops over them; 151 of the corpus's 264 statements "
            "have one, a median of 11 lines and 238 at worst.\n\n"
            "**Not a second ``region``, and it can outlive one.** ``region`` "
            "ends where the next statement's binds start being filled, and "
            "for 64 of those 151 the row loop opened inside the region runs "
            "on past that point: ``getTasksToExecCommand_JEDI``'s query on "
            "line 4591 owns 4585-4603 and works its rows through 4600-4621, "
            "because another query opens on 4606 *inside the loop*. Reading "
            "exits in ``region`` alone loses the ``except`` that gives up the "
            "lock and the ``continue`` that skips the task -- which was the "
            "one thing the analyser asked for that P1-53 did not deliver.\n\n"
            "The loop must **open** inside ``region``; only its end is "
            "followed. Searching the function for every loop over the same "
            "names instead gives 174 exits over 71 statements against 90 over "
            "45, because ``res`` is reused: ``copyArchive.py::main``'s first "
            "query picks up the ``continue`` belonging to its query on line "
            "381, and one handler on line 1364 lands on 19 statements. The "
            "region is what makes ``res`` mean *this* ``res``."
        ),
    )
    crosses_to: list[str] = Field(
        default_factory=list,
        description=(
            "Junction owners a row kept by this statement is carried on to, "
            "sorted -- where the investigation goes next when the statement "
            "itself is the lead.\n\n"
            "Every call is placed by the corpus's calls, as the reach record "
            "places it, not by its name: a call of the region landing in a "
            "function the map owns a junction in, and the rows handed back to "
            "a caller and what it does with them.  Rows reaching a function "
            "that owns none are followed through it -- into the calls it hands "
            "them to (``JediTaskBuffer.updateTask_JEDI`` handing ``taskSpec`` "
            "to ``proxy.updateTask_JEDI``; a started thread's inherited "
            "``run`` calling ``self.runImpl()`` for the class it was built "
            "as), and back to its callers where it returns them "
            "(``TaskBuffer.getConfigValue``).  ``copyArchive.py::main``'s "
            "``holding`` query hands each job to ``Watcher(...)`` and calls "
            "``.start()``, so the next thing that happens to a held job is "
            "``Watcher.run``.  A call the record reads as the database is not "
            "followed.\n\n"
            "Never the caller's own function: naming it would satisfy the "
            "question and not the purpose.  Where a call on the way could not "
            "be placed, :attr:`crosses_open` names it, and this is what is "
            "known rather than all."
        ),
    )
    crosses_open: list[OpenCrossing] = Field(
        default_factory=list,
        description=(
            "The calls on the way the rows are carried that the corpus's calls "
            "cannot place, and the ones that may be a caller they go back to "
            "(spelled with the returning function's name) -- each a hole of the "
            "reach record, its why there.  Empty: every call on the way was "
            "placed, and no hole may be a caller back."
        ),
    )
    joins: list[JoinPair] = Field(
        default_factory=list,
        description=(
            "Each column = column equality between two of the statement's "
            "sources, in a ``WHERE`` or an ``ON``, a subquery's among them -- "
            "read off the statement's tree, so a statement not read as one "
            "gives none; sorted.  Not ``x IN (SELECT y ...)``."
        ),
    )
    anchor: Anchor = Field(
        ...,
        description=(
            "``line_start`` is the line the statement *runs* on -- the "
            "``execute`` / ``querySQLS`` call -- not the line the text was "
            "assembled on, which is where a reader looking for the query would "
            "start but not where the binds are in scope.  ``line_end`` is the "
            "same line: a statement is one line of execution, and the region "
            "around it is ``bind_window``, which is a different claim."
        ),
    )

    @staticmethod
    def make_name(map_id: str, owner: str, line: int) -> str:
        """Identity is the statement: one owner runs many.

        The line rather than the text, because two runs of the *same* text in
        one function are two sites -- that is the whole reason this node
        exists -- and because the text is not a name anybody can look up.
        """
        return f"{map_id}:{owner}@{line}"


class BoundaryNode(BaseNode):
    """Where causation crosses into a system this map does not cover.

    An unbound boundary is a complete answer in itself ("the pilot reported
    1099") *and* a concrete work item ("bind this boundary"), the same shape as
    a recorded capability gap.

    ``kind`` splits the two diagnostic modes.  A system that *reports state*
    (the pilot posting an error code) leaves its value in the record, so
    non-arrival is visible.  A system that *transports causation* (a message
    broker) leaves nothing on the consumer side when a message is lost, so the
    question flips from "which condition blocked it" to "was it published at
    all", which the consumer alone cannot answer.
    """

    node_type: NodeType = NodeType.BOUNDARY
    map_id: str
    derived_from: str
    system: str
    kind: str = Field(default="reports_state", description="reports_state | transports_causation")
    transport: str = Field(
        default="http",
        description=(
            "http | shared_table.  A separate axis from ``kind``: that one says "
            "whether evidence survives a failure to cross, this one says where "
            "to go looking.  An endpoint is investigated through the arrival "
            "log; a shared table is investigated by querying it, and the row "
            "either exists or it does not."
        ),
    )
    interface: str = Field(
        ...,
        description=(
            "Receiving-side function, or ``schema.table`` for a shared table; for "
            "an outbound boundary, the PanDA method that makes the call out."
        ),
    )
    direction: str = Field(
        default="inbound",
        description=(
            "``inbound`` -- another system reaches PanDA here -- or ``outbound``: "
            "PanDA calls another system here (``RucioAPI.register_dataset``).  "
            "The inbound fields below say nothing about an outbound one, which "
            "carries ``verbs``, ``calls_out`` and ``call_sites`` instead."
        ),
    )
    verbs: list[str] = Field(
        default_factory=list,
        description=(
            "Outbound: what the call out does -- creates, attaches, changes, "
            "removes, asks -- read from the outside methods' names, which are the "
            "far side's API and still names."
        ),
    )
    calls_out: list[str] = Field(
        default_factory=list,
        description="Outbound: the outside methods called, ``rucio.client.Client.add_dataset``.",
    )
    call_sites: list[str] = Field(
        default_factory=list,
        description="Outbound: ``file:line`` of each call in PanDA placed at this method.",
    )
    carried_values: list[str] = Field(
        default_factory=list,
        description="Names the far side supplies, in the receiving side's spelling.",
    )
    handed_over: list[str] = Field(
        default_factory=list,
        description=(
            "Names this map's code writes for the far side to read.  Empty for "
            "an endpoint, which is inbound only; a shared table is a channel in "
            "both directions, and which direction is broken is the first "
            "question to ask about one."
        ),
    )
    operations: list[str] = Field(
        default_factory=list,
        description=(
            "SQL verbs used against a shared table.  Recorded because the set "
            "itself is a finding: a ``DELETE`` alongside an ``INSERT`` on a "
            "command table means a second command silently replaces one that "
            "was never picked up."
        ),
    )
    access_conditions: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Preconditions the boundary itself enforces before any junction "
            "sees the request -- transport security, caller role, HTTP method, "
            "ownership.  These are the *first* place a request can be rejected, "
            "so a command that appears to have vanished may never have been "
            "accepted here."
        ),
    )
    observable_values: list[str] = Field(
        default_factory=list,
        description=(
            "Carried values the receiving code logs on arrival.  What is not "
            "logged cannot be recovered afterwards, so this is the difference "
            "between a boundary that can be investigated and one that can only "
            "be guessed at."
        ),
    )
    accepts_arbitrary: bool = Field(
        default=False,
        description=(
            "The endpoint takes ``**kwargs``, so what crosses cannot be "
            "enumerated from the signature.  Recorded rather than ignored: "
            "listing no carried values for such a boundary would claim nothing "
            "crosses it, which is the opposite of the truth."
        ),
    )
    version_binding: list[str] = Field(
        default_factory=list,
        description=(
            "Fields that pin the far side's version at run time.  Needed "
            "because a system whose version moves independently cannot share "
            "this map's version stamp."
        ),
    )
    resolution: str = Field(default="unresolved", description="unresolved | bound")
    anchor: Optional[Anchor] = None

    @staticmethod
    def make_name(map_id: str, system: str, interface: str) -> str:
        """Identity is the interface, not one value crossing it.

        An endpoint carries many values at once, so keying per value would
        split one boundary into dozens that all move together.
        """
        return f"{map_id}:{system}:{interface}"


class ValueEnumNode(BaseNode):
    """One ``NAME = value`` constant from a declared enumeration.

    Kept separate from a *declared value set* (``FINAL_TASK_STATUSES = [...]``)
    even though both are module-level constants: an enumeration entry is an
    index entry -- code ``100`` in namespace ``taskbuffer`` means ``EC_Kill`` --
    whereas a value set is a vocabulary oracle used to check extraction.
    Counting them together made an ErrorCode module and a config module look
    like the same kind of file.

    ``name`` is ``"<namespace>.<constant>"`` and the reverse index is keyed on
    ``(namespace, value)``.  Numbers are reused across namespaces --
    ``taskbuffer.EC_Kill``, ``dataservice.EC_Setupper`` and
    ``jobdispatcher.EC_Watcher`` are all ``100`` -- so decoding an observed
    code requires knowing which field it came from.
    """

    node_type: NodeType = NodeType.VALUE_ENUM
    map_id: str
    derived_from: str
    namespace: str
    constant: str
    value: Any
    comment: Optional[str] = None
    anchor: Optional[Anchor] = None
    references: int = Field(
        default=0,
        description=(
            "Number of sites referencing this constant.  Zero means it is dead "
            "or was mis-classified, so it does not belong in a decoding index."
        ),
    )

    @staticmethod
    def make_name(namespace: str, constant: str) -> str:
        return f"{namespace}.{constant}"


#: What a count of :class:`ReadingStop` covers, said wherever one is printed:
#: a count read as "where the map stops" is taken for all of it, and it is the
#: SQL reader's alone.
SCOPE_OF_STOPS = (
    "(the SQL reader's stops only: the calls the map cannot place are a list of "
    "their own, and a method's writes and SiteMapper are in neither)"
)


class CallStop(BaseModel):
    """One call the map cannot place, and why: the other list of where it stops.

    :class:`ReadingStop` is the SQL reader's, and its count was read as where
    the map stops at all.  A call ``calls`` cannot place is a stop of its own
    kind -- the walk does not enter it and no reader follows a value through
    it -- and a third of the corpus's calls to its own functions are one.
    Every one, with its reason, because a ratio says how much and the reason
    says what would place it.  Not a node, for :class:`ReadingStop`'s reason.
    """

    reason: str
    file: str
    line: int
    owner: str
    callee: str = Field(default="", description="The called expression as written, cut to 80 characters.")


class ReadingStop(BaseModel):
    """One place the reading of one statement stopped, and what it stopped on.

    Beside :class:`CoverageStat` rather than inside it, because it answers a
    different question.  Coverage asks whether a statement was read at all; a
    statement can be counted as read, carry its table, its subject and its
    binds, and still be missing the half of its ``WHERE`` that decides which
    rows come back.  Until this record existed those two cases were the same
    row in the same ratio, which is why ``getJobs`` -- the query the pilot's
    whole job request is answered by -- sat in the map as a ``SELECT`` that
    filters on nothing, with no number anywhere moving.

    Not a node.  It is what the build could not say, and a map that stored its
    own gaps as nodes would count them among what it covers.  For the same
    reason ``diff`` does not compare these: it compares what the map claims,
    and a build that stops somewhere new says so in its own report.
    """

    reason: str
    detail: str = Field(
        default="",
        description=(
            "The expression, unparsed -- ``build_where_clause()``, "
            "``jobStatus IN ({})``. What to open next, not what class of thing "
            "stopped the reading."
        ),
    )
    file: str = ""
    owner: str = ""
    line: int = Field(
        default=0,
        description=(
            "Where the thing that stopped the reading is written -- the "
            "fragment's line, which for ``getJobs`` is 2380, not the line the "
            "statement runs on.  See ``statement``."
        ),
    )
    statement: int = Field(
        default=0,
        description=(
            "The line the statement runs on, which is the line a read site is "
            "anchored at.  Without it a stop names a function and not a "
            "statement, and a function-level answer is how "
            "``movers-of-a-value-agree`` came to excuse ``getJobs`` as 'read "
            "only in part' on the strength of a stop in a different statement "
            "from the one it was asked about -- seven of the eight functions it "
            "excused that way stopped only in a write."
        ),
    )
    reads: list[str] = Field(
        default_factory=list,
        description=(
            "The tables the statement reads rows of, as the source spells them. "
            "Empty for a statement that only writes: a precondition whose bind "
            "nothing fills, or an ``INSERT``'s ``bindValuesExpression()``, is a "
            "gap in reading a write, and says nothing about which rows a "
            "function picked up."
        ),
    )
    reads_known: bool = Field(
        default=True,
        description=(
            "Whether ``reads`` is known at all.  False for a statement the "
            "build has no text of -- one handed in by the caller, or a name "
            "this function does not bind -- where an empty ``reads`` would "
            "read as \"only writes\" and is really \"any rows\"."
        ),
    )


class RowKeySource(BaseModel):
    """How one guarded write names the rows it lands on.

    Not a node, beside :class:`ReadingStop` and for the same reason.  A write
    whose ``WHERE`` requires a value -- ``status=:oldStatus`` -- lands on rows
    its other terms name: ``jediTaskID=:jediTaskID``.  Where that key was
    filled from a parameter or a field, the caller chose the rows and is where
    a reader goes next; where it was filled here, from this function's own
    query, it is not.  ``movers-of-a-value-agree`` needs that to split the
    functions none of whose queries read the rows.
    """

    owner: str
    file: str = ""
    statement: int = Field(default=0, description="The line the write runs on.")
    columns: list[str] = Field(
        default_factory=list, description="The columns its guard tests, as SQL spells them."
    )
    keys: list[str] = Field(
        default_factory=list,
        description="The row-naming terms it was judged from, ``jediTaskID=:jediTaskID``.",
    )
    handed_by: str = Field(
        default="",
        description=(
            "The parameter or ``self`` field a key was filled from, or empty "
            "when every key was filled here.  Read as the carry reads any "
            "value, each parameter and each field of ``self`` seeded on its "
            "own, after the reach record: ``for panda_id, attempt_nr in "
            "resGL:`` rebinds the parameters' names with the function's own "
            "rows, and a key bound from ``panda_id`` below is not the "
            "caller's.  Empty, too, where the build had no corpus calls to "
            "read it with."
        ),
    )
    #: The key values to read *handed_by* from, ``(func, file, [(value, statement, steps)])`` in the keys' order --
    #: *steps* the fields of an object that name its row; the step
    #: after the reach record reads them (:func:`~bamboo.codemap.panda.recognizers.sqlwrite.attach_key_sources`).
    _filled: Optional[tuple] = PrivateAttr(default=None)


#: Where a value filling a write's row-naming bind came from (:attr:`KeySource.kind`): the rows of a statement of
#: the same function, a sequence's next values a statement of it selects (``SELECT ...FILEID_SEQ.nextval FROM
#: dual``, which reads no table and so is no read site), a parameter, a field of ``self`` -- each read by the carry
#: -- or, where none of those reaches it, a literal, an attribute of something else, or a value the map does not
#: follow.
KEY_ROWS = "rows"
KEY_SEQUENCE = "sequence"
KEY_PARAMETER = "parameter"
KEY_SELF_FIELD = "self_field"
KEY_CONSTANT = "constant"
KEY_ATTRIBUTE = "attribute"
KEY_UNKNOWN = "unknown"
KEY_KINDS = (KEY_ROWS, KEY_SEQUENCE, KEY_PARAMETER, KEY_SELF_FIELD, KEY_CONSTANT, KEY_ATTRIBUTE, KEY_UNKNOWN)
#: Why a row-naming term's values were not read (:attr:`KeyFrom.unread`); the last, a flush call's term the method
#: fills with values no argument of the call brings -- its own rows, a field of its own ``self``.
KEY_NO_MAP = "no bind map"
KEY_NO_BIND = "no bind found"
KEY_IN_CALLEE = "filled in the callee"
#: An object the function assigns fields on that no write here persists (:attr:`KeyFrom.unread`, of no term): what
#: an assigned field reaches next -- a call of the corpus it is handed to or made on, a statement here it is bound
#: in whose own record does not write it, a return, what the function was handed (a parameter's object, ``self``'s:
#: the caller holds it after) -- or nothing the map reads.
KEY_HANDED = "handed to a call"
KEY_BOUND = "bound in a statement"
KEY_RETURNED = "returned"
KEY_KEPT = "kept in what it was handed"
KEY_NO_USE = "no use found"
#: Why an ``INSERT`` makes a row or writes one that was a row of its class before (:attr:`KeyFrom.new_row_by`).
#: New: a key column filled from a sequence, by ``RETURNING``, with a literal, with a value computed here, or with
#: one carried from something not of the table's class (a scalar handed in, a parent's row).
NEW_ROW_SEQUENCE = "sequence"
NEW_ROW_RETURNING = "returning"
NEW_ROW_LITERAL = "literal"
NEW_ROW_COMPUTED = "computed"
NEW_ROW_OTHER_CLASS = "not of its class"
#: Not new: every key column carried from a row of the table's class read here, or the function deletes the row
#: its whole key names from the same table first.
NEW_ROW_MOVED = "moved within its class"
NEW_ROW_REWRITTEN = "rewritten"
#: Not said: a key column carried from an object of the table's class handed in (who made its key is the
#: caller's), the table has no key in the schema (or only dated columns), or a key column's value is not read.
NEW_ROW_HANDED = "handed in as its class"
NEW_ROW_NO_KEY = "no key"
NEW_ROW_UNREAD = "not read"
#: Not said either: the run updates the row or inserts it -- one text or the other of one statement
#: (``addStdOut``'s ``sqlU`` / ``sqlI``) -- so the row it writes may have been there.
NEW_ROW_UPSERT = "updates or inserts"
#: Why a ``rows`` source names no column of its statement's rows (:attr:`KeySource.column_unread`): the select list
#: is ``*``, the value is at a place a hole's columns fill (``SELECT {JobSpec.columnNames()}`` read by position),
#: the item is an expression, no item or several hold the value (no item: also a field of an object packed from a
#: hole or ``*`` that the table has no column of), the statement's texts give it different columns
#: (``getCoJumboJobsToBeFinished``'s ``jobsActive4`` or ``jobsDefined4``), or no select list was read.
KEY_COLUMN_STAR = "select *"
KEY_COLUMN_HOLE = "in a hole"
KEY_COLUMN_EXPRESSION = "an expression"
KEY_COLUMN_NONE = "no item"
KEY_COLUMN_SEVERAL = "several items"
KEY_COLUMN_TEXTS = "texts differ"
KEY_COLUMN_UNREAD = "select list not read"


class KeySource(BaseModel):
    """One place a value filling a write's row-naming bind came from -- see :class:`KeyFrom`."""

    kind: str = Field(description="One of :data:`KEY_KINDS`.")
    line: Optional[int] = Field(default=None, description="Where the bind is filled.")
    name: str = Field(
        default="",
        description=(
            "``rows``: the :class:`ReadSiteNode`'s name (``panda:<owner>@<line>``, also as *site_owner* and "
            "*site_line*); ``sequence``: the sequence's (``JEDI_DATASET_CONT_FILEID_SEQ``); ``parameter``: its "
            "name; ``self_field``: ``self.<field>``; ``constant``: its repr; ``attribute`` / ``unknown``: the value "
            "as written."
        ),
    )
    site_owner: str = Field(default="", description="``rows`` / ``sequence``: the statement's owner.")
    site_line: Optional[int] = Field(
        default=None, description="``rows`` / ``sequence``: the line the statement runs on."
    )
    paths: list[str] = Field(
        default_factory=list,
        description=(
            "``rows`` / ``sequence`` / ``parameter`` / ``self_field``: where in the bind's value they sit -- ``()`` "
            "the value itself, else its steps (``[0]``, ``.f``, ``[*]``).  A part of them reads as them here; which "
            "column of a row fills the key is *column*, and which field of a parameter or of ``self`` is *attr* "
            "(``archiveJob``'s ``:name`` at 1718, ``tmpFile.dispatchDBlock`` of a file of its ``job``, is ``job`` "
            "here and ``FileSpec.dispatchDBlock`` there)."
        ),
    )
    column: str = Field(
        default="",
        description=(
            "``rows``: the statement's select item the value is, as ``table.column`` lowered "
            "(``jedi_tasks.jeditaskid``) -- each item carried on its own from where it sits in what the statement's "
            "fetch gives; a field of an object filled by name (``taskSpec.pack(row)``) is the item of its name.  "
            "Read on each select list the statement's texts give, and said where all agree.  Empty where "
            "*column_unread* says why."
        ),
    )
    attr: str = Field(
        default="",
        description=(
            "``rows``: *column* as a subject, ``JediTaskSpec.jediTaskID``, where its table's spec class has it; where "
            f"the statement's texts give columns of several tables that are one subject (*column_unread* "
            f"``{KEY_COLUMN_TEXTS}``: a hole read per table, ``jobsActive4``'s or ``jobsDefined4``'s ``PandaID``), "
            "that subject.  ``parameter`` / ``self_field`` / ``attribute``: the subject of the value the key is "
            "filled with, where that value is a field of an object whose spec class the build tells "
            "(``archiveJob`` 1718's ``tmpFile.dispatchDBlock`` of a file of its parameter ``job``, "
            "``FileSpec.dispatchDBlock``); empty for the parameter or field itself (what the callers hand)."
        ),
    )
    column_unread: str = Field(
        default="",
        description=(
            f"``rows`` with no *column*: ``{KEY_COLUMN_STAR}``, ``{KEY_COLUMN_HOLE}``, ``{KEY_COLUMN_EXPRESSION}``, "
            f"``{KEY_COLUMN_NONE}``, ``{KEY_COLUMN_SEVERAL}``, ``{KEY_COLUMN_TEXTS}``, ``{KEY_COLUMN_UNREAD}``."
        ),
    )
    same_rows: bool = Field(
        default=False,
        description=(
            "``rows``: the term's statement names the source statement's own rows -- one of its tables, by every "
            "column of the table's primary key, each filled from that *column* of the source's rows: a read reads "
            "them again (``getTasksToBeFinished_JEDI`` reads 1676's tasks again at 1731 by ``jediTaskID``), a write "
            "writes them.  Not where it names other rows by them: another table's (``JEDI_Datasets`` by the task's "
            "``jediTaskID``), a part of a key, another column (``checkParentTask_JEDI``'s parent by ``parent_tid``)."
        ),
    )
    never_at: list[str] = Field(
        default_factory=list,
        description=(
            "``rows``: the values the source statement selects (``JediTaskSpec.status=scouted``, as its read site's "
            "*selects*) for which the term's statement never runs with this value -- a test on the way to each call "
            "handed it (``pathcond.ruled_out_at``) on a name holding that column of the source's rows: 1731 is in the "
            "``else`` of ``if taskStatus == \"scouted\":``, ``taskStatus`` 1676's ``status``.  Read in the source's "
            "own function only: empty where the rows come through calls (*via*)."
        ),
    )
    via: list[str] = Field(
        default_factory=list,
        description=(
            "``rows`` reaching the bind through calls, as the walk carrying the statement's rows "
            "(``attach_crossings``) took them: each call on the way, ``file:line`` -- one handing them down, or one "
            "in a caller they are returned to -- from the source's function to the term's; empty in the source's own "
            "function.  The term's statement may be in a junction's owner the walk crosses to and does not go "
            "through: its statements are read with what it is handed."
        ),
    )
    reaches: list[str] = Field(
        default_factory=list,
        description=(
            "On a *same_rows* source of a read through calls (*via*): the junction owners the read's rows reach and "
            "read (as ``ReadSiteNode.crosses_to`` names them), going back up only by the calls of *via* -- not every "
            "caller of the read's function, as the read site's own ``crosses_to`` is walked; the read's own owner "
            "aside, the source's own may be one."
        ),
    )
    reaches_open: list[OpenCrossing] = Field(
        default_factory=list,
        description="On that way, the calls the walk could not place, as ``ReadSiteNode.crosses_open``.",
    )
    #: The walk's way to the bind, ``(kind, file, func, call)`` per step, for *reaches* (``sqlwrite._Landing``).
    _way: Optional[tuple] = PrivateAttr(default=None)


class KeyFrom(BaseModel):
    """Where one row-naming term of a statement was filled from: the rows it lands on, and who named them.

    Not a node, beside :class:`RowKeySource`.  One per term of a write's ``WHERE`` that names its rows --
    ``jediTaskID=:jediTaskID``, or an ``IN`` list a helper builds, ``datasetID IN (:dataset__in)`` -- on a column
    the statement does not write, for every write statement, guarded or not; and one per term a read statement (a
    :class:`ReadSiteNode`'s) compares its own rows' column with a bind in, every row meeting it (*reads*).  Its *sources* are where the bind's
    values came from, each read by the carry: the rows of a statement of the same function
    (``prepareTasksToBeFinished_JEDI`` keys the runs of its later loop by the ``jediTaskID`` its statements at 3723
    and 3768 select), a parameter, a field of ``self``.  A reader asking which of a function's arms the rows of one of its statements
    reach takes the KeyFroms with a ``rows`` source of that statement, and the branches at *writes*' lines; one asking
    where they go after a statement reads them again takes the read KeyFroms with a ``rows`` source of it that is
    :attr:`KeySource.same_rows`, less those whose :attr:`KeySource.never_at` holds the value, and that read site's
    own way on (its ``crosses_to``, its KeyFroms' sources): joined by the reader, which can test the value there.

    Read in the same step as :attr:`RowKeySource.handed_by`, after the reach record and by the same carries, so the
    two never disagree.  A statement run with several texts is one KeyFrom per term, its sources every text's.
    Not followed here: rows reaching the bind through a call (a parameter is where this function was handed them),
    and the values of the columns the statement writes.

    Two more kinds of record, of an object whose fields the function assigns (*object*).  A call of a method that
    persists what its caller changed on a spec (a flush: ``self.taskBufferIF.updateJob(job, ...)``) is a write at
    the call, one KeyFrom per term of the statement the method flushes the object's class into, as the method fills
    it: each bind of the method's own record of the term whose value is a parameter, or a part of one, read as the
    call's argument for it -- ``updateJob``'s ``PandaID=:PandaID`` is its ``job``'s field, and
    ``registerTaskInOneShot_JEDI``'s ``jediTaskID=:jediTaskID`` is its parameter ``jediTaskID``, not the spec's --
    on a column the caller does not write, where the call appends the term; and, where the method makes its
    ``WHERE`` in a loop over a dict parameter's items (``updateTask_JEDI``'s *criteria*), one per key of the dict the
    call hands it, ``jediTaskID=criteria["jediTaskID"]``.  And an object no write here persists gets one record per
    what its assigned fields reach next (*unread*: :data:`KEY_HANDED`, :data:`KEY_BOUND`, :data:`KEY_RETURNED`,
    :data:`KEY_KEPT`, :data:`KEY_NO_USE`), of no term and no source: with none, its arms would read as known not to be
    on a statement's rows.
    """

    owner: str
    file: str = ""
    statement: int = Field(
        default=0,
        description=(
            "The line the write runs on -- the first, where the function runs one text with one bind map from "
            "several places (``setNoRetry`` at 2604 and 2608): one run, its binds every place's "
            "(``sql.Execution.calls``); a flush's, the call.  For an object no write persists, the first line its "
            "fields reach the use on (the call, the statement, the return), else the first assignment's."
        ),
    )
    table: str = Field(
        default="", description="The table written, or read (*reads*); empty for an object no write persists."
    )
    term: str = Field(
        description=(
            "As the map reads it: ``jediTaskID=:jediTaskID``, ``datasetID IN (:dataset__in)``; empty for an object no "
            "write persists, and for an ``INSERT`` into a table with no key."
        )
    )
    writes: dict[str, list[int]] = Field(
        default_factory=dict,
        description=(
            "Each subject the statement writes (``JobSpec.jobStatus``, as a junction's subject), with the lines it "
            "writes that subject's values on -- a junction's :attr:`Branch.line` or one of its "
            ":attr:`Branch.other_lines`: a junction is one per subject and owner, its branches gathered from every "
            "statement writing it, and these say which are this statement's."
        ),
    )
    unread: str = Field(
        default="",
        description=(
            f"Empty, or why the term's values were not read: ``{KEY_NO_MAP}`` / ``{KEY_NO_BIND}`` / "
            f"``{KEY_IN_CALLEE}``; for an object no write persists, what its fields reach: ``{KEY_HANDED}`` (see "
            f"*handed_to*) / ``{KEY_BOUND}`` / ``{KEY_RETURNED}`` / ``{KEY_KEPT}`` / ``{KEY_NO_USE}``."
        ),
    )
    sources: list[KeySource] = Field(default_factory=list)
    reads: bool = Field(
        default=False,
        description=(
            "A term of a statement that reads rows (its read site's run): *writes* empty, no *object* nor *new_row*; "
            "*routes* as a write's.  Its sources say which rows the statement is asked for by."
        ),
    )
    attr: str = Field(
        default="",
        description="The term's column as a subject, ``JediTaskSpec.jediTaskID``, where *table*'s spec class has it.",
    )
    object: str = Field(
        default="",
        description=(
            "The object whose fields the write persists -- ``job`` of ``execute(sql, job.valuesMap())``, of "
            "``updateJob(job, ...)`` -- or none does; empty for a term of a bind map.  *writes* has the subjects the "
            "function assigns on it that reach the write (the map, the call), or the use *unread* names."
        ),
    )
    handed_to: list[str] = Field(
        default_factory=list,
        description=(
            f"Where *unread* is ``{KEY_HANDED}``: the functions of the corpus the calls land in (``file::name``, as a "
            "junction's owner) -- where the object may be persisted."
        ),
    )
    new_row: Optional[bool] = Field(
        default=None,
        description=(
            "On every KeyFrom of an ``INSERT`` (one per key column of its table, dated ones left out): ``True`` where "
            "the statement makes the row, ``False`` where it writes one that was a row of the table's class before "
            "(moved within its class, or rewritten), ``None`` where that is not read; always ``None`` for any "
            "other statement.  Its why is *new_row_by*."
        ),
    )
    new_row_by: str = Field(
        default="",
        description=(
            f"``{NEW_ROW_SEQUENCE}`` / ``{NEW_ROW_RETURNING}`` / ``{NEW_ROW_LITERAL}`` / ``{NEW_ROW_COMPUTED}`` / "
            f"``{NEW_ROW_OTHER_CLASS}`` (new); ``{NEW_ROW_MOVED}`` / ``{NEW_ROW_REWRITTEN}`` (not new); "
            f"``{NEW_ROW_HANDED}`` / ``{NEW_ROW_UPSERT}`` / ``{NEW_ROW_NO_KEY}`` / ``{NEW_ROW_UNREAD}`` (not said)."
        ),
    )
    new_row_line: Optional[int] = Field(
        default=None,
        description=(
            "Where *new_row_by* is read: the statement selecting the sequence, the one returning the key, the value "
            "computed or carried, the delete -- or the ``INSERT`` itself."
        ),
    )
    routes: int = Field(
        default=0,
        description=(
            "How many routes of calls the corpus's calls place into the function (``Calls.routes``).  0 where no call "
            "is placed at it -- an entry point, a thread's ``run``, a daemon's ``main`` started by configuration, a "
            "class read as a value -- and *routes_open* is then true: it says nothing of whether the write runs."
        ),
    )
    routes_open: bool = Field(
        default=False,
        description=(
            "Whether a route may be missing from *routes*: a call by the function's name placed nowhere, or the "
            "function reached from outside the corpus (``Calls.missed``)."
        ),
    )
    out_on: list[list[str]] = Field(
        default_factory=list,
        description=(
            "The routes on which the write never runs: each the calls of one route as ``file:line``, the call into "
            "the function first, where every place the write runs from (each call running the text with its map, a "
            "flush's call, an object's uses) is ruled out by the constants the route hands the function's flag "
            "parameters -- an ``if`` holding it, an ``if`` before it that leaves.  A flag the function binds again "
            "rules nothing out.  All of *routes* with *routes_open* false: no placed caller runs it (``retryJob``'s "
            "old-row ``UPDATE`` under ``if not getNewPandaID``, every caller handing ``True``)."
        ),
    )
    #: The binds to read *sources* from, ``(func, file, [sql.Bind])``; the step after the reach record reads them.
    _binds: Optional[tuple] = PrivateAttr(default=None)
    #: ``(func, [node, ...])``: where the write runs from -- the calls running its text with its map, a flush's call,
    #: an object's uses -- for *routes* / *out_on* (``sqlwrite.attach_routes``).
    _runs: Optional[tuple] = PrivateAttr(default=None)
    #: The runs of its statement (``sql.Execution``) a term's record was read from: which of *_runs*' calls a source's
    #: value is handed to (``never_at``).
    _executions: list = PrivateAttr(default_factory=list)
    #: An ``INSERT``'s, what decides *new_row* beside the sources: see :class:`~bamboo.codemap.panda.recognizers.
    #: sqlwrite._Insert`.
    _insert: Optional[Any] = PrivateAttr(default=None)
    #: ``(object as spelled, its class, the calls taking it -- its valuesMap() calls, or the flush call --, the
    #: attributes the write takes back out)`` where a write persists *object*: the step after the reach record adds to
    #: *writes* the subjects the function assigns on it that reach one of those calls.
    _object: Optional[tuple] = PrivateAttr(default=None)


class CoverageStat(BaseModel):
    """Per-file extraction coverage for one slice.

    Reported per file rather than as a slice average: the filter-chain idiom
    ranges from 26 occurrences to none across sibling broker files, and a mean
    would bury that variance under the slices whose recognizers depend only on
    Python syntax and so match near-perfectly everywhere.
    """

    slice_name: str
    file: str
    candidates: int = 0
    explained: int = 0

    @property
    def ratio(self) -> float:
        return self.explained / self.candidates if self.candidates else 1.0


class MapFragment(BaseModel):
    """What one recognizer contributes to a build.

    The plugin contract is shaped around this rather than around how the
    extraction is done, so a recognizer implemented with a different tool
    still composes -- fragments merge on the semantic signature.
    """

    map_id: str
    derived_from: str
    #: The instance record the calls were read with -- ``<path>: <instance>, <n> value(s), <digest>``
    #: (:func:`~bamboo.codemap.calls.read_recorded`) -- or ``""``, the code alone.  A reader computing at query
    #: time reads module names the same way only with the same record.
    instance_record: str = ""
    subjects: list[SubjectNode] = Field(default_factory=list)
    entities: list[EntityNode] = Field(default_factory=list)
    junctions: list[JunctionNode] = Field(default_factory=list)
    boundaries: list[BoundaryNode] = Field(default_factory=list)
    value_enums: list[ValueEnumNode] = Field(default_factory=list)
    filter_stages: list["FilterStageNode"] = Field(default_factory=list)
    loop_cuts: list["LoopCutNode"] = Field(default_factory=list)
    log_sites: list[LogSiteNode] = Field(default_factory=list)
    read_sites: list["ReadSiteNode"] = Field(default_factory=list)
    function_prints: list["FunctionPrintsNode"] = Field(default_factory=list)
    reach_graph: list["ReachNode"] = Field(default_factory=list)
    print_calls: dict[str, list[int]] = Field(
        default_factory=dict,
        description=(
            "``owner`` -> the line of every logging call the function makes "
            "itself, read off the syntax tree alone.  The other reading of what "
            "``FunctionPrintsNode.statements`` holds, carried so that "
            "``prints-are-counted`` can fail on a call the skeleton left out."
        ),
    )
    diagnostics: list["DiagnosticTemplate"] = Field(default_factory=list)
    enumeration_writes: list["EnumerationWrite"] = Field(default_factory=list)
    coverage: list[CoverageStat] = Field(default_factory=list)
    reading_stops: list[ReadingStop] = Field(default_factory=list)
    call_stops: list[CallStop] = Field(default_factory=list)
    open_calls: list[CallStop] = Field(
        default_factory=list,
        description=(
            "The build's calls to its own functions placed open: at the "
            "definitions they name, and possibly somewhere this reading cannot "
            "name, the reason saying why.  Not stops -- the walk enters what "
            "they name -- and counted from the same calls as :attr:`call_stops`."
        ),
    )
    calls_considered: int = Field(
        default=0,
        description=(
            "How many calls :attr:`call_stops` was drawn from: those naming a "
            "function or class this corpus defines, a logger's methods left out."
        ),
    )
    row_key_sources: list[RowKeySource] = Field(default_factory=list)
    key_froms: list[KeyFrom] = Field(default_factory=list)
    declaration_yields: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "``Class (file)`` -> how many names the extraction read from the "
            "column declaration there.  Every declaration is listed, including "
            "the ones that yielded nothing, which is the point: a form nobody "
            "reads looks exactly like a class with nothing to declare once the "
            "names have been merged away.  Carried on the fragment rather than "
            "logged so that ``spec-declarations-are-read`` can fail on it."
        ),
    )
    annotation_readings: list["AnnotationAudit"] = Field(
        default_factory=list,
        description=(
            "One row per container element annotation the map depends on -- what "
            "it states, what the code actually stores there, and whether "
            "removing it would change any write.  A bare ``Dict`` says nothing, "
            "so these element types are facts the map takes on trust; these rows "
            "are what let two gates check them instead."
        ),
    )
    spec_annotation_forms: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "``file:line`` -> the spec class the extraction read from the "
            "annotation there, empty when it read none.  Listed only where the "
            "annotation's text names a class the corpus declares, so an empty "
            "value means the two readings disagree and the form is one the "
            "extraction does not understand.  Same shape as "
            "``declaration_yields`` and for the same reason: counting what came "
            "back is the only way an unreadable form differs from an absent one."
        ),
    )

    def extend(self, other: "MapFragment") -> None:
        """Merge *other* into this fragment in place.

        Every part but the three that name the build and its inputs, read off
        the model rather than listed here: a list of fields written out by hand
        is a second copy of the model, and the part added next is the one it
        drops.
        """
        for part in type(self).model_fields:
            if part in ("map_id", "derived_from", "instance_record"):
                continue
            held = getattr(self, part)
            if isinstance(held, dict):
                held.update(getattr(other, part))
            elif isinstance(held, int):
                setattr(self, part, held + getattr(other, part))  # a count of what was looked at
            else:
                held.extend(getattr(other, part))


#: The parts of a :class:`MapFragment` that hold the map's nodes, in the order
#: the reader takes them (``lookup._fragment_nodes``).  One tuple for the reader
#: and for ``diff``, because the store and ``diff`` once kept two lists and
#: ``diff`` held seven of the eight kinds the database stored: every log site
#: compared as absent from both builds.  Every other part is the build's own
#: report -- see ``diff.NOT_COMPARED_PARTS``.
MAP_NODE_GROUPS: tuple[str, ...] = (
    "value_enums",
    "subjects",
    "entities",
    "junctions",
    "boundaries",
    "filter_stages",
    "loop_cuts",
    "log_sites",
    "read_sites",
    "function_prints",
    "reach_graph",
)


class AnnotationAudit(BaseModel):
    """Two readings of one annotation the map has to take on trust, side by side.

    Not a node.  This records how much the map trusts a statement PanDA makes
    about a class it cannot otherwise read, which is a fact about the extraction
    rather than about the system, so it belongs in the report and the gates and
    not in the graph.

    Two shapes qualify, and both are trusted for the same reason -- the class
    they name is stated nowhere else in the expression that reaches the write.
    A container's element type (``Dict[str, DatasetSpec]``) has a second
    reading in what the code stores; a context manager's yield type
    (``Iterator[WorkflowSpec | None]``) has one only when the object's usage
    happens to be distinctive, which for two of PanDA's three workflow locks it
    is not.
    """

    where: str = Field(description="``file:line`` of the annotation")
    kind: str = Field(
        default="container",
        description=(
            "``container`` for an element type, ``yield`` for a context "
            "manager's.  Only the first has a stored second reading, so the "
            "agreement gate is about that one and the ablation gate is about "
            "both."
        ),
    )
    container: str = Field(description="the annotated name, for the report")
    stated: str = Field(description="the spec class the annotation names")
    put_in: list[str] = Field(
        default_factory=list,
        description=(
            "Spec classes the code stores in that container, where they resolve "
            "on their own.  Empty means no independent reading was available, "
            "which is not evidence against the annotation -- it is the case the "
            "annotation was asked for."
        ),
    )
    read: bool = Field(
        description=(
            "Whether removing the annotation would change any write in its "
            "scope, in class or in basis.  False means the map does not depend "
            "on it: either it is redundant or a read form is missing."
        )
    )
    audited: bool = Field(
        default=True,
        description=(
            "Whether the ablation had anything to compare.  False where no "
            "spec attribute is written in the annotation's scope, which makes "
            "``read`` vacuous -- the row is kept because its other reading, "
            "what the code puts in the container, is unaffected by that."
        ),
    )
    unsettled: list[str] = Field(
        default_factory=list,
        description=(
            "``line:attribute`` for each write in scope of an attribute the "
            "stated class declares that still resolves by inference or not at "
            "all.  This is what separates the two readings of ``read=False``: "
            "with nothing unsettled the map simply reaches the class another "
            "way, which an annotated tree makes the normal case."
        ),
    )


class DiagnosticTemplate(BaseModel):
    """One place the code assembles text, and the frame it assembles.

    **Deliberately not a node, and deliberately outside promotion.**  Promotion
    answers "is this field worth asking why about?", which for a free-text field
    is the wrong question -- ``ddmErrorDiag`` has no value set to enumerate, and
    a subject node for it would invite a reader to expect a branch table.  The
    question an investigation actually asks of it is the other way round: *this
    message was seen, who wrote it?*  That is an index from a template to an
    anchor, and an index makes no claim about the field, so it needs none of the
    machinery -- no promotion criterion, no threshold, and no rule for telling a
    message from an identifier.

    That last point is what settled the design.  Both structural readings that
    look like they could separate the two were measured and both misclassify:
    the template reading as prose calls ``panda.pp.in.{}.{}`` a message and
    misses ``errorDialog`` (whose other writes go through ``setErrDiag``), and
    "nothing ever compares this field" also matches ``lfn`` and ``jobName``.
    Indexing all of them costs nothing and misleads nobody.

    Only *assembled* text is indexed.  A bare literal is not a template, and an
    exact message can be found by searching the source for itself.
    """

    map_id: str
    derived_from: str
    template: str = Field(
        ...,
        description=(
            "The literal frame, run-time parts as ``{}`` -- what a message "
            "observed in production is matched against."
        ),
    )
    field: str = Field(
        ...,
        description=(
            "Qualified name of the field the text lands in.  Often not a "
            "subject: that is the point."
        ),
    )
    form: str = Field(..., description="attribute | bind")
    anchor: Anchor


class EnumerationWrite(BaseModel):
    """One place the code puts a named enumeration constant into a field.

    The binding between a field and the enumeration that decodes it, which is
    what a reverse index needs and cannot get from the constants alone.  Numbers
    are reused across enumerations by design -- ``taskbuffer.ErrorCode.EC_Kill``
    and ``jobdispatcher.ErrorCode.EC_Watcher`` are both ``100`` -- so the index
    is keyed on ``(namespace, value)``, and decoding a code seen in a record is
    impossible without knowing which field it came from.

    ``errorcode`` calls that binding "separate and explicit, not recoverable
    from the constant's location", and it is right about the location.  It is
    recoverable from the *write*: ``jobSpec.taskBufferErrorCode =
    ErrorCode.EC_Kill`` states the field and the constant in one statement, and
    the corpus does that 79 times over five fields -- three of which are the
    three ``ErrorCode`` modules, one field each.

    Not restricted to error codes, and deliberately: the rule is "a declared
    field assigned a constant the value-enum slice extracted", which also
    catches ``JediTaskSpec.eventService`` and ``JobSpec.job_label``.  Telling
    an error code from any other enumeration would need a classifier, and an
    index makes no claim that would justify one -- the same reason
    :class:`DiagnosticTemplate` indexes every assembled string.

    **Outside promotion**, also for that class's reason: no criterion fires on
    these fields -- every write is tier 2, the right-hand side being a module
    constant rather than a literal -- so their subjects carry none, and rightly.
    "Why is this field 100?" is not the question anyone asks of it; "what does
    100 mean here?" is, and that is an index.
    """

    map_id: str
    derived_from: str
    field: str = Field(
        ..., description="Qualified name of the field written, e.g. JobSpec.taskBufferErrorCode."
    )
    constant: str = Field(..., description="The constant's bare name, e.g. EC_Kill.")
    namespace: str = Field(
        ...,
        description=(
            "The enumeration the constant belongs to, as the value-enum index "
            "keys it.  This is the half that makes an observed value decodable."
        ),
    )
    anchor: Anchor


class FilterStageNode(BaseNode):
    """One reason a candidate was dropped on the way to a selection.

    Brokerage does not pick a site; it narrows a list, twenty-odd times in a
    row, and "the distribution is wrong" means asking *which step* threw the
    candidates away.  That is not a branch table -- every step runs, and each
    removes some -- so it is its own node rather than a junction whose
    branches happen to be cumulative.

    ``criteria_tag`` is the identity, because it is the semantic signature and
    the log line at once: the code emits ``criteria=-diskIO`` per rejected site,
    so a stage keyed on it can be counted directly from production logs and
    survives any refactoring that keeps the tag.  ``funnel_label`` is the
    coarser step the summary counter reports (``diskIO check``), which several
    tags can share -- ``-lowmemory`` and ``-highmemory`` are both
    ``memory check``.  Both appear in logs and they are read differently, so
    both are kept.
    """

    node_type: NodeType = NodeType.FILTER_STAGE
    map_id: str
    derived_from: str
    owner: str = Field(..., description="module::function holding the chain.")
    criteria_tag: str = Field(
        default="",
        description="The tag the code emits, e.g. '-diskIO'.  Empty when the step is unnamed.",
    )
    funnel_label: str = Field(
        default="",
        description="Step name the candidate counter reports, e.g. 'diskIO check'.",
    )
    order: int = Field(default=0, description="Position in the chain, by source order.")
    conditions: list[str] = Field(
        default_factory=list,
        description=(
            "Guards under which a candidate is dropped.  Not solved: the values "
            "come from observation and are substituted in, which is what turns "
            "'120 sites became 3' into a reason."
        ),
    )
    inputs: list[str] = Field(
        default_factory=list,
        description="Identifiers the conditions read -- where the backward walk continues.",
    )
    emits: list[str] = Field(
        default_factory=list, description="Message templates the stage logs when it drops one."
    )
    log_level: Optional[str] = Field(default=None, description="Level of those messages.")
    log_files: list[str] = Field(
        default_factory=list,
        description=(
            "Log files this stage's messages can land in, e.g. "
            "'panda-AtlasProdJobBroker.log'.  Part of the answer rather than a "
            "fetching detail: the map exists to say which component's log to "
            "read.  Empty when the module declares no logger and is mixed into "
            "nothing, so its output goes to a caller's file that the source "
            "does not name."
        ),
    )
    anchor: Optional[Anchor] = None
    gloss_key: str = Field(
        default="",
        description=(
            "Hash of the enclosing function's text -- the key of the reading "
            "that explains it.  Computed at build time so the input a reader "
            "would be given is chosen by the map and not by whoever asks; the "
            "reading itself is filled in lazily.  Empty when the function "
            "could not be located in the snapshot."
        ),
    )

    @staticmethod
    def make_name(map_id: str, owner: str, signature: str) -> str:
        return f"{map_id}:{owner}:{signature}"


class LoopCutNode(BaseNode):
    """A guard inside a loop that drops the candidate and says why in prose.

    The same cut as :class:`FilterStageNode` and a different kind of node,
    because the two are evidenced differently and every consumer of a stage
    reads the difference.  Brokerage announces its cuts twice in machine-
    readable form -- ``criteria=-diskIO`` per rejected site and
    ``N candidates passed disk check`` per step -- and the whole of ``localize``
    is built on those two.  Everywhere else in the corpus a loop drops its
    candidate with ``continue`` under a guard and a sentence::

        for workQueue in workQueueList:
            for resource_type in resource_types:
                ...
                if not flagLocked:
                    tmpLog_inner.debug("skip since locked by another process")
                    continue

    No tag, no counter, and the sentence is the only name the step has.  Poured
    into ``FilterStageNode`` these arrive with ``criteria_tag`` and
    ``funnel_label`` both empty, which is two fields answering a question they
    were not asked: ``vocabulary`` would offer 75 new chains described in
    brokerage words, ``_leading``'s fallback would open one of them, and
    ``check-map`` would silently add 36 files to what it asks production --
    a footprint decision that was deliberately made once and by hand.

    ``message`` is the identity, for the reason ``criteria_tag`` is one over
    there: it is the semantic signature and the log line at once, so a cut keyed
    on it can be counted from production and survives any edit that keeps the
    wording.  Position is not identity -- the loop moves when anything above it
    does.

    Measured over ``panda-server-source 1.0.2``: 797 of 834 ``continue``
    statements sit inside a loop under a guard, 200 have a logged line in the
    same block, five of those already carry a ``criteria=`` tag and 37 more are
    inside the five brokerage functions the stage recognizer already reads.
    The remaining 153 are this node, over 75 functions in 36 files.
    """

    node_type: NodeType = NodeType.LOOP_CUT
    map_id: str
    derived_from: str
    owner: str = Field(..., description="module::function holding the loop.")
    message: str = Field(
        ...,
        description=(
            "The line logged before the candidate is dropped, with "
            "interpolations rendered as '{}'.  The step's only name."
        ),
    )
    search_key: str = Field(
        default="",
        description=(
            "The longest fixed run of the message -- what to grep production "
            "for.  Not guaranteed to select only this cut: 'skip {}' leaves "
            "'skip'.  ``build-map`` reports the keys that also match another "
            "cut in the same file, which is the threshold-free form of that "
            "worry, and a capped answer says the rest."
        ),
    )
    scope_prefix: str = Field(
        default="",
        description=(
            "The prefix the logger this cut writes through puts on every line, "
            "when the enclosing function builds one -- 'vo={} queue={} cloud={} "
            "pid={} {}'.  Which candidate a line is about is in the prefix, not "
            "the sentence, so this is the key that narrows the question to one "
            "queue or one task.  Empty when the module logs through a bare "
            "logger."
        ),
    )
    order: int = Field(default=0, description="Position within the owner, by source order.")
    loop_line: int = Field(
        default=0, description="Line of the loop header whose iteration this drops."
    )
    conditions: list[str] = Field(
        default_factory=list,
        description="Guards under which the candidate is dropped.",
    )
    inputs: list[str] = Field(
        default_factory=list,
        description="Identifiers the conditions read -- where the backward walk continues.",
    )
    log_level: Optional[str] = Field(default=None, description="Level the message is logged at.")
    log_files: list[str] = Field(
        default_factory=list,
        description="Files the message can land in.  Empty when the module declares no logger.",
    )
    anchor: Optional[Anchor] = None

    @staticmethod
    def make_name(map_id: str, owner: str, signature: str) -> str:
        return f"{map_id}:{owner}:{signature}"


class GlossNode(BaseNode):
    """A reading of one function, and the exact text that reading was given.

    The unit is a function because that is what a reader needs, measured
    rather than assumed.  An arm's own guard is a median of eight lines, but
    only 35 of 927 arms refer to nothing outside it: 96% load names they do not
    bind, a median of four each, and 52% sit in a loop whose header is outside
    their guard.  Following those references -- the backward slice the map
    already names as its method -- reaches a median of 37% of the enclosing
    function and, at the ninetieth percentile, 96% of it.  Slicing would build
    a mechanism to arrive at very nearly the function.

    Sharing is why it is a node at all rather than a field on each junction:
    497 junctions sit in 213 functions, 376 of them share one with another
    junction, and the 109 filter stages sit in five.  Read once per function
    and every arm in it is explained together -- 213 readings rather than 1046.

    ``shown`` is the text that was actually handed over, kept for the same
    reason grep output is written to an evidence file: when a reading is wrong,
    it separates *the input was insufficient* from *the reader misread it*, and
    those want opposite fixes.  It is evidence, never the record -- the source
    tree stays canonical.  Storing the mapped functions instead would not even
    save space: the whole corpus is 5.6 MB of Python, the owner functions alone
    are 1.65 MB of it, and adding the callers a reader might ask for next takes
    that to 63% and then 90%.  The closure does not converge on a subset.  It
    would also cost the oracle, since ``check-map`` rebuilds the fragment from
    source on every run and the offline gates work by comparing two statements
    the *code* makes.

    Identity is ``gloss_key``, the hash of the text.  A reading is only about
    the text it was given, so a changed function is a different node rather
    than a stale field -- which is also why this does not share
    :meth:`JunctionNode.content_hash`: that one hashes a junction's *meaning*
    and deliberately ignores the surrounding source, which is the right key for
    an artefact derived from the branch table and the wrong one for an artefact
    derived from the file.
    """

    node_type: NodeType = NodeType.GLOSS
    map_id: str
    derived_from: str
    gloss_key: str = Field(..., description="Hash of `shown`; the node's identity.")
    owner: str = Field(..., description="module::qualname of the function read.")
    anchor: Optional[Anchor] = None
    shown: str = Field(
        default="",
        description="The exact text the reader was given.  Evidence, not the record.",
    )
    explanation: str = Field(
        default="",
        description=(
            "What the reader made of it.  Empty until something reads it: the "
            "map is built eagerly and this is filled in lazily, when an "
            "investigation actually touches this function."
        ),
    )

    @staticmethod
    def make_name(map_id: str, gloss_key: str) -> str:
        return f"{map_id}:gloss:{gloss_key}"


# ---------------------------------------------------------------------------
# What one investigation makes of the map
# ---------------------------------------------------------------------------
#
# None of these are nodes.  A strategy is the product of asking the map one
# question about one incident; storing it would put an answer into the
# vocabulary the answers are drawn from, and the next build would either
# overwrite it or leave it behind pointing at junctions that have moved.  They
# carry ``derived_from`` for the same reason every node does -- a strategy is a
# statement about one version of the source, and reading one against another
# deployment is exactly the skew the version stamp exists to make visible.


#: What an observation settled about a candidate.
SEEN = "seen"
ELIMINATED = "eliminated"
UNSETTLED = "unsettled"
UNASKABLE = "unaskable"

#: What production answered for one log file.
ANSWER_SEEN = "seen"
ANSWER_ABSENT = "absent"
ANSWER_NO_FILE = "no_file"
ANSWER_INCONCLUSIVE = "inconclusive"
ANSWER_NOT_ASKED = "not_asked"


#: What kind of question a symptom is, which decides which derivation answers it.
#: Not a flag anyone types -- a symptom is resolved out of the map's own
#: vocabulary, and the vocabulary entry carries the kind.  See :class:`MapTerm`.
SYMPTOM_VALUE = "value"
SYMPTOM_DISTRIBUTION = "distribution"


class Symptom(BaseModel):
    """What was observed, in the map's own vocabulary.

    Deliberately not free text.  Turning "the task is stuck in pending" into
    grep terms and ranking thirty files is the retrieval problem the map exists
    to remove; naming a subject and a value instead makes the lookup exact, and
    the quality of the answer stops depending on how well the question was
    phrased.

    **Two kinds, because the system settles values two different ways.**  A
    value is settled by one branch of one junction, so the question is *which
    arm*.  A site list is not settled at all -- brokerage starts with every
    candidate and narrows about twenty-five times, every step runs, and the
    question is *which step threw them away*.  One shape cannot carry both
    without one of them lying, and the kind is what the derivation dispatches
    on.  It is deliberately not a command-line flag: adding an option per
    symptom would put the retrieval problem back in the caller's hands, one
    layer up.
    """

    kind: str = Field(
        default=SYMPTOM_VALUE,
        description=f"{SYMPTOM_VALUE} | {SYMPTOM_DISTRIBUTION}.",
    )
    subject: str = Field(default="", description="Qualified subject, e.g. JediTaskSpec.status.")
    observed: str = Field(default="", description="The value the record actually holds.")
    focus: str = Field(
        default="",
        description=(
            "For a distribution, what the description landed on -- a chain, a "
            "``criteria=-`` tag or a funnel step.  It decides what the answer "
            "*leads* with, never what it contains: the whole chain is read "
            "either way, because a description naming one cut is a guess about "
            "which cut mattered and the evidence is what settles that."
        ),
    )
    task_id: Optional[str] = Field(
        default=None,
        description=(
            "The entity this is about, when there is one.  Without it the "
            "evidence can still say which writers are live, but not which one "
            "wrote *this* row -- a different and much weaker claim, so it is "
            "recorded rather than defaulted."
        ),
    )
    observed_diag: Optional[str] = Field(
        default=None,
        description=(
            "The message the record carries, when it carries one.  The cheapest "
            "evidence in the system and the only kind with no window: it is a "
            "column, so one API call returns it whole, where a log line has to "
            "be found in a rotation under a byte cap.  What it buys is the arm "
            "rather than the junction, because PanDA writes the reason into the "
            "same message it logs."
        ),
    )

    @model_validator(mode="after")
    def _the_kind_has_what_it_needs(self) -> "Symptom":
        """Refuse a symptom its own kind cannot be asked.

        The fields are optional per-field because the two kinds use different
        ones, and leaving it there would let a distribution with no focus reach
        the derivation and come back empty -- an answer shaped like "nothing is
        wrong".  Checked here so the failure is at the point the question is
        formed, where the caller still knows what it meant.
        """
        if self.kind == SYMPTOM_VALUE and not (self.subject and self.observed):
            raise ValueError("a value symptom needs both a subject and an observed value")
        if self.kind == SYMPTOM_DISTRIBUTION and not self.focus:
            raise ValueError("a distribution symptom needs a focus -- a chain, a tag or a step")
        if self.kind not in (SYMPTOM_VALUE, SYMPTOM_DISTRIBUTION):
            raise ValueError(f"{self.kind!r} is not a symptom kind")
        return self


#: What part of the map a vocabulary entry names.  Distinct from
#: ``Symptom.kind``, which says which derivation answers it: several term kinds
#: resolve to one derivation, because a chain, one of its cuts and one of its
#: steps are three ways of pointing at the same brokerage question.
TERM_VALUE = "value"
TERM_CUT = "cut"
TERM_STEP = "step"
TERM_CHAIN = "chain"


class MapTerm(BaseModel):
    """One thing the map can be asked about, and the symptom it resolves to.

    The entry that makes free text tractable.  Asking a source navigator means
    turning a description into grep terms and ranking thirty files; asking the
    map means picking from a set that is *closed and enumerable* -- measured at
    420 entries for PanDA: 296 ``(subject, value)`` pairs, 70 cut tags, 49
    funnel steps and 5 chains.  That number is what "the retrieval problem is
    removed" means quantitatively, and it is small enough to match against
    deterministically and small enough to hand an LLM whole.

    Each entry carries the :class:`Symptom` it resolves to, so resolution has
    nothing to decide beyond which entry the words meant.  Building the symptom
    at the far end instead would put the map's vocabulary into the resolver,
    where it would be a second copy free to drift.
    """

    kind: str = Field(..., description=f"{TERM_VALUE} | {TERM_CUT} | {TERM_STEP} | {TERM_CHAIN}")
    key: str = Field(..., description="How the map spells it, e.g. 'JediTaskSpec.status=exhausted'.")
    words: list[str] = Field(
        default_factory=list,
        description=(
            "The text a description is matched against -- the key split into "
            "words, plus whatever else names the same thing.  A cut carries its "
            "step's words too, because production spells it ``-lowmemory`` and "
            "people say 'memory'."
        ),
    )
    symptom: Symptom


class Match(BaseModel):
    """One vocabulary entry a description could have meant, and how well.

    Ranked rather than chosen.  A description that names two things is a fact
    about the description, and picking one of them silently is the failure mode
    the map was built to remove -- ``too_many_candidates`` answered as though it
    were one candidate.
    """

    term: MapTerm
    score: float = Field(
        default=0.0,
        description=(
            "Share of the entry's own weight the description accounted for. "
            "Words are weighted by how rare they are across the vocabulary, "
            "computed from the vocabulary itself rather than tuned: 'check' "
            "appears in most steps and settles nothing, 'lowmemory' appears in "
            "one and settles it."
        ),
    )
    words: list[str] = Field(
        default_factory=list, description="Which of the entry's words the description said."
    )
    exact: bool = Field(
        default=False,
        description=(
            "The description was this entry's key, spelled the same way.  Kept "
            "as a fact rather than folded into the score: the word weighting "
            "cannot express it -- an entry that carries its step's words too "
            "has a larger denominator and loses to a near-homograph that "
            "carries fewer, which is how ``-t1_weight`` came third to "
            "``-t1weight`` when ``-t1_weight`` was what was typed."
        ),
    )
    tied_with: list[str] = Field(
        default_factory=list,
        description=(
            "Entries scoring exactly the same.  A tie broken by sorting is a "
            "tie decided by how the two keys happen to be spelled -- '-' sorts "
            "below 'T', and nothing about the question said so.  Reported so "
            "the reader breaks it, since the resolver has nothing to break it "
            "with."
        ),
    )


class CandidateBranch(BaseModel):
    """One way a candidate reaches the observed value.

    Junctions were the unit until a symptom came along whose whole question is
    *which arm*: ``setScoutJobData_JEDI`` sends a task to ``exhausted`` from six
    arms, one per reason, and pooling their conditions into one list per
    candidate answers "why" with the union of six answers.  Flattening was
    right for ``pending``, where fourteen of eighteen candidates have no
    condition at all -- and it hid the question here entirely.
    """

    outcome: str
    tier: int = 1
    line: Optional[int] = Field(
        default=None, description="Where the write is, so a reader can go and look."
    )
    tags: list[str] = Field(
        default_factory=list,
        description=(
            "What the code calls this decision -- ``action=set_exhausted`` and "
            "``reason=low_efficiency``.  Both a search key in production's logs "
            "and, where the same message is persisted, the part of the record "
            "that names this arm."
        ),
    )
    messages: list[str] = Field(
        default_factory=list,
        description=(
            "The frames the block records, holes left open.  What names the arm "
            "where no tag does, which in production is most of the time."
        ),
    )
    conditions: list[str] = Field(default_factory=list)
    row_precondition: list[str] = Field(default_factory=list)
    row_precondition_when: list[ConditionalGuard] = Field(default_factory=list)
    matched: bool = Field(
        default=False,
        description=(
            "Whether the message the record carries names this arm.  One "
            "direction only: a match proves this arm decided, and a record that "
            "names none proves nothing, because the field holds the last message "
            "written to it and a later junction may have overwritten it."
        ),
    )


class Handover(BaseModel):
    """What one entry passed in on its way to this candidate.

    Carried on the candidate because the arm's own body is where a trace
    starts and the body reads ``self.taskList``, not ``taskList``.  Naming the
    function that filled it, and with what, is the one step forward from the
    anchor that cannot be taken by reading the arm's own module: the knight
    hands its rows to a worker through a constructor and a thread, so there is
    no call to follow.
    """

    entry: str = Field(..., description="Module the control came from.")
    via: str = Field(..., description="Function there that made the handover.")
    reached_by: str = Field(
        ...,
        description=(
            "``call`` or ``dispatch`` -- which tells a reader what the keys "
            "of ``fields`` are, a parameter or a worker's own attribute."
        ),
    )
    fields: dict[str, str] = Field(
        default_factory=dict,
        description="Name the arm's body reads it under -> the expression supplied.",
    )


class Candidate(BaseModel):
    """One junction that could have produced the observed value.

    The set is the system's real fan-out, not an artefact of how the map was
    built: an expert asked why a task is ``pending`` faces the same eighteen.
    What the map adds is that the enumeration is complete, precomputed, and
    carries the log file to read for each one.
    """

    owner: str
    file: str = Field(default="", description="Where the junction is, for a reader.")
    blob_sha: str = Field(
        default="",
        description=(
            "Hash of the file the map read.  Carried so that reading the code "
            "against a different snapshot is noticed: line drift is one of the "
            "two skew symptoms no gate catches."
        ),
    )
    gloss_key: str = Field(
        default="",
        description=(
            "Key of the reading of its enclosing function.  Carried on the "
            "candidate because eliminating a candidate has to take its reading "
            "with it: the point of the enumeration is that what the evidence "
            "rules out stops being offered."
        ),
    )
    tier: int = Field(
        default=1,
        description=(
            "1 when a branch states this value outright, 2 when the writer is "
            "known and the value is only settled at run time.  A tier-2 "
            "candidate is not eliminated by the value, because 'this one could "
            "have' is the honest answer for it -- unless the branch's own "
            "frames are ones the value does not fit, which is the one case "
            "where the arm rules itself out.  See ``branch_excludes``."
        ),
    )
    dispatch: list[DispatchFanout] = Field(
        default_factory=list,
        description=(
            "Classes a run-time choice inside this junction's class could "
            "have picked.  Carried to the report because a candidate set "
            "nobody can check is three guesses wearing a bracket -- the line "
            "that prints the chosen class travels with it."
        ),
    )
    log_files: list[str] = Field(
        default_factory=list,
        description=(
            "Where a line about this junction firing can appear -- the caller's "
            "files where it has callers, its own only where it declares a "
            "logger.  Empty means no file can be asked about it at all, which "
            "is a finding rather than a reason to guess one."
        ),
    )
    branches: list[CandidateBranch] = Field(
        default_factory=list,
        description="The arms that reach this outcome, each with its own reason.",
    )
    triggers: list[str] = Field(default_factory=list)
    entries: list[str] = Field(default_factory=list)
    called_from: list[str] = Field(
        default_factory=list,
        description=(
            "``file:line`` of each call placed at the owner -- "
            ":attr:`JunctionNode.called_from`.  What moves a job to "
            "``activated`` is whoever calls ``activateJob``, and the map held "
            "that on the junction without the answer ever saying it."
        ),
    )
    handovers: list[Handover] = Field(
        default_factory=list,
        description=(
            "Entries that passed something in, with what.  Only the ones that "
            "did: an entry handing over nothing says nothing, and listing it "
            "would make 'this path omits the argument' and 'this reader does "
            "not bind positionals' look the same."
        ),
    )
    verdict: str = Field(
        default=UNSETTLED,
        description=f"{SEEN} | {ELIMINATED} | {UNSETTLED} | {UNASKABLE}",
    )
    because: str = Field(default="", description="Why the verdict, in one clause.")

    @property
    def conditions(self) -> list[str]:
        """Every condition any of the arms is under, in order, deduplicated.

        The summary the arms compose to.  Derived rather than stored so that the
        two cannot say different things -- which is the failure the arms were
        introduced to fix, one level up.
        """
        seen: list[str] = []
        for branch in self.branches:
            for condition in branch.conditions:
                if condition not in seen:
                    seen.append(condition)
        return seen

    @property
    def row_precondition(self) -> list[str]:
        """What the row had to already say for any of these arms to land.

        Kept apart from the conditions because the two fail differently: an
        unmet path condition means the code never got here and the log is
        silent; an unmet row precondition means the code got here, said so, and
        changed nothing.
        """
        seen: list[str] = []
        for branch in self.branches:
            for guard in branch.row_precondition:
                if guard not in seen:
                    seen.append(guard)
        return seen

    @property
    def row_precondition_when(self) -> list[ConditionalGuard]:
        """The guards only some of these arms' writes carry, each with when (:class:`ConditionalGuard`)."""
        seen: list[ConditionalGuard] = []
        for branch in self.branches:
            for guard in branch.row_precondition_when:
                if guard not in seen:
                    seen.append(guard)
        return seen

    @property
    def named(self) -> list[CandidateBranch]:
        """The arms the record's message names, if it names any."""
        return [branch for branch in self.branches if branch.matched]


class Observation(BaseModel):
    """One question put to production, and what its answer would settle.

    Asked of both machine groups, not of the one the package suggests.  JEDI
    opens its own TaskBuffer, so ``pandaserver`` code called by a knight runs in
    the JEDI process and logs there; the package is wrong precisely where most
    junctions are.  A service that does not have the file answers "No such file
    or directory", which is cheap and is itself distinguishable from an empty
    match -- so the union costs one extra query and removes a guess.
    """

    log_file: str
    pattern: str
    role: str = Field(
        default="probe",
        description=(
            "``probe`` asks whether this row was moved here; ``control`` asks "
            "whether the file carries that kind of line at all.  Without the "
            "second, a file that never speaks the sentence is indistinguishable "
            "from one whose writer did not fire, and the map's largest group of "
            "junctions is reached through files of exactly that kind."
        ),
    )
    services: list[str] = Field(default_factory=list)
    settles: list[str] = Field(
        default_factory=list, description="Owners of the candidates this can settle."
    )
    control_for: Optional[str] = Field(
        default=None,
        description=(
            "For a control, the probe pattern it controls.  Paired by pattern "
            "rather than by file because one file can carry two probes that are "
            "different sentences -- the value line every writer shares, and the "
            "tagged line one branch names itself with -- and a control answers "
            "for exactly one of them."
        ),
    )
    verdict: str = Field(
        default=ANSWER_NOT_ASKED,
        description=(
            f"{ANSWER_SEEN} | {ANSWER_ABSENT} | {ANSWER_NO_FILE} | "
            f"{ANSWER_INCONCLUSIVE} | {ANSWER_NOT_ASKED}.  Four ways of not "
            "matching and only two of them are answers, which is the whole "
            "reason this is not a count."
        ),
    )
    keep_lines: Optional[int] = Field(
        default=None,
        description=(
            "How many matched lines this question needs kept, where the role's "
            "default is wrong for it.  A probe asking *whether* a writer fired "
            "wants a handful for the report; a probe whose answer **is** the "
            "lines -- every candidate a chain dropped -- wants all of them, and "
            "a trimmed answer there is not a smaller sample but a different one."
        ),
    )
    matched: int = 0
    sample: list[str] = Field(
        default_factory=list, description="A few matched lines, for the report."
    )


#: What the reader a value's follow-up names can do to the row.  Three states,
#: because "a query selects this value" and "something will move this row on"
#: are not the same claim and the map was answering the second with the first:
#: ``analy_pmerge_jobs_wait_time`` selects ``cancelled`` jobs to average a wait
#: time and changes nothing, and the verdict told a reader to go and ask a
#: metrics daemon why it had not picked their row up.
#:
#: The split is by whether the reader is started by something of its own, not
#: by where it lives.  A getter settles nothing and writes nothing either, and
#: saying the row will not move would be just as wrong in the other direction
#: -- what acts is its caller, which the line does not name.
ACTS = "acts"
READS_ONLY_AT_TOP = "reads only, and nothing starts it but its own trigger"
READS_ONLY_FOR_A_CALLER = "reads only, on behalf of whoever called it"


class FollowUp(BaseModel):
    """Whether anything will move the value on, and what to ask if not.

    The other half of the question.  Which junction wrote ``pending`` says how
    the row got where it is; this says whether it is going to leave, and a task
    can be stuck for a reason that has nothing to do with the writer.
    """

    selected: bool = Field(
        ...,
        description=(
            "Whether anything in the map acts on rows by this value -- a "
            "query's predicate or an update's.  Both answer 'will the row be "
            "moved on at all'; ``selected_by`` and ``updated_by`` say which."
        ),
    )
    selection_gates: list[str] = Field(
        default_factory=list,
        description=(
            "Tables bounding what those queries can see, none of which anything "
            "in the map writes.  The reason a row can fail to be picked up while "
            "passing every condition on its own status."
        ),
    )
    selected_by: list[str] = Field(
        default_factory=list,
        description=(
            "The functions whose query selects rows on the observed value.  "
            "Where the row has to be picked up, so where to ask why it was not."
        ),
    )
    reader_acts: str = Field(
        default=ACTS,
        description=(
            "What those readers can do to the row: ``ACTS`` where one settles a "
            "value or writes a row, and one of the two reads-only states where "
            "none does.  Separate from ``selected``, which stays true either "
            "way -- a query really does select on the value; the question this "
            "answers is whether being selected leads anywhere."
        ),
    )
    updated_by: list[str] = Field(
        default_factory=list,
        description=(
            "The functions whose update or delete acts on rows already holding "
            "the observed value.  Not a place the row could have been missed --"
            " an update is the picking up -- so it is named separately and only "
            "answers 'what will move this row' where no query does."
        ),
    )
    creates_rows: bool = Field(
        default=False,
        description=(
            "Whether anything in this map creates rows of the subject's kind.  "
            "Separate from ``created_by`` being empty, which would otherwise "
            "read as 'the map did not look': three kinds of row here are "
            "changed by this corpus and created outside it."
        ),
    )
    created_by: list[str] = Field(
        default_factory=list,
        description=(
            "The functions whose INSERT brings a row of this kind into "
            "existence.  The question a branch table cannot answer, because a "
            "branch is about a value a row already has.  Where the verdict is "
            "'ask whether the command arrived', this is where arriving happens."
        ),
    )
    reader_log_files: list[str] = Field(
        default_factory=list,
        description=(
            "Where those readers' diagnostics land.  Read from the reader's "
            "junction where it settles something and from its "
            ":class:`LogSiteNode` where it does not -- an earlier wording here "
            "said a function that only selects \"has no node to hang a file "
            "on\", which was the state of things the log site node was added to "
            "fix.  Empty is a finding, not a licence to guess a file."
        ),
    )
    reader_statements: list[StatementReading] = Field(
        default_factory=list,
        description=(
            "One entry per statement that asks for the observed value: where "
            "it is, what it asks rows by, what else it requires of them, and "
            "the stretch of its function it owns.  The log file says which log "
            "to read and this says where in the code: ``copyArchive.py::main`` "
            "asks for ``jobStatus='holding'`` from three statements with three "
            "different time limits, and named by the function alone those are "
            "one answer.\n\n"
            "Rendered text was what this held until P1-52, which is why the "
            "conditions and the walk had nowhere to go: a string cannot carry "
            "a skeleton."
        ),
    )
    reader_statements_elsewhere: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "``{reader: how many of its other statements ask on other "
            "values}``.  Counted rather than dropped -- a reader asking what "
            "else this daemon does is asking something real, and a silent "
            "filter would make the listed ones look like the whole of it."
        ),
    )
    guarded_by: list[GuardReading] = Field(
        default_factory=list,
        description=(
            "The functions whose write requires the observed value of the row "
            "-- the movers the queries above cannot show.  With "
            "``reader_statements`` this is the whole of 'what moves the row "
            "on' as the map holds it, and each entry says which half it came "
            "from.  Not ``updated_by``: that also holds writes that act on "
            "rows holding the value and change something else -- a priority, "
            "a share -- and leave the value where it is."
        ),
    )
    returned_by_exclusion: list[StatementReading] = Field(
        default_factory=list,
        description=(
            "Statements that never ask for the observed value and return rows "
            "holding it anyway, because what they pass over is other values -- "
            "see :func:`admits_by_exclusion`.  Not readers: counted among them, "
            "``status NOT IN ('done')`` would select every value but one.  "
            "Without them, ``no query selects on 'defined'`` was the whole "
            "answer for a value a ``NOT IN`` picks up every cycle."
        ),
    )
    refused_by: list[StatementReading] = Field(
        default_factory=list,
        description=(
            "Statements no row of which holds the observed value: they pass "
            "over it outright, in every branch and every way they are built "
            "-- :attr:`ReadSiteNode.refuses`.  The answer to a row that sits "
            "on a value: ``copyArchive``'s reassign and ``lockJobsForReassign`` "
            "both say ``relocationFlag<>2``, so a job holding 2 is one they "
            "leave alone."
        ),
    )
    consulted_by: list[StatementReading] = Field(
        default_factory=list,
        description=(
            "Statements whose subquery checks whether rows holding the observed "
            "value exist -- ``EXISTS (SELECT 1 ... WHERE jobStatus=:st)`` -- and "
            "pick none of them up.  Not readers either: the rows they return "
            "are other rows, and whether those move depends on these."
        ),
    )
    triggers: list[str] = Field(
        default_factory=list,
        description=(
            "Triggers of the query that selects the observed value, where the "
            "map names a reader that is also a junction; otherwise pooled over "
            "the subject's writers, which is the older approximation and stays "
            "as the fallback rather than leaving the field empty."
        ),
    )
    self_repairing: bool = False
    carried_from: list[str] = Field(
        default_factory=list,
        description=(
            "Fields this subject's value is copied from.  Where to take the "
            "question when nothing selects the observed value: the row is not "
            "going anywhere, so the useful question is who wrote the step before."
        ),
    )
    question: str = Field(default="", description="What to ask next, in one sentence.")
    assembled_as: list[str] = Field(
        default_factory=list,
        description=(
            "The frames the observed value fits, of the arms that assemble it "
            "at run time -- set only where nothing in the map selects, guards, "
            "updates or checks rows by it.  Then the rest of this block has "
            "nothing to say: no reader to name, and the triggers and the places "
            "rows are made answer how a row is picked up by a value nothing "
            "picks it up by.  Decided by the observation, not the subject: "
            "``Could not add files to DDM: {}`` is a text an arm assembles, "
            "and the map does not say whether a field holds messages."
        ),
    )
    reader_gaps: dict[str, list[ReaderGap]] = Field(
        default_factory=dict,
        description=(
            "``{reader: where the map could not read its statements on these "
            "rows}``, a :class:`ReaderGap` per statement -- the same judgement "
            ":attr:`GuardReading.gaps` carries for the other half.  Only readers "
            "with one are listed; :func:`bamboo.codemap.strategy.gap_texts` says "
            "them."
        ),
    )
    gaps_note: str = Field(
        default="",
        description="Why no function above carries its gaps, when none could be read.",
    )
    # What ``attach_mover_gaps`` judges with, gathered while the map is open and
    # left out of what is printed or saved: it is the map's, not the answer's.
    mover_sites: dict[str, list["ReadSiteNode"]] = Field(default_factory=dict, exclude=True)
    mover_files: dict[str, tuple[str, str]] = Field(default_factory=dict, exclude=True)
    table_kinds: dict[str, Optional[str]] = Field(default_factory=dict, exclude=True)
    spec_class: str = Field(default="", exclude=True)


class StageCut(BaseModel):
    """One filter stage, and what it actually removed for this entity.

    The unit of the brokerage answer.  A stage is not a candidate explanation
    the way a junction is -- every stage runs, on every pass -- so what has to
    be measured is not *whether* it fired but *how much of the list it took*.

    Counted in distinct sites rather than in log lines.  Brokerage re-runs for
    a task many times over the window, so a line count is a count of passes
    multiplied by an effect, and the ranking it produces is about how often the
    task was brokered.  The same weight-not-count reading that made one branch
    the answer for ``'cleanup'``.
    """

    tag: str = Field(default="", description="The ``criteria=-`` token, e.g. '-lowmemory'.")
    funnel_label: str = Field(default="", description="The coarser step, e.g. 'memory check'.")
    owner: str = Field(default="", description="The chain this stage belongs to.")
    order: int = Field(default=0, description="Its position in that chain.")
    line: Optional[int] = Field(default=None, description="Where to go and read.")
    conditions: list[str] = Field(default_factory=list)
    inputs: list[str] = Field(
        default_factory=list, description="What those conditions read -- where a backward walk goes."
    )
    reads: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Attribute name -> the subject the map says that read names.  "
            "Resolved while the map is open, because the rejection line that "
            "supplies the *value* arrives later and by then there is nothing to "
            "resolve it against.  Names several subjects declare and the map "
            "cannot separate are absent here and reported as leads instead."
        ),
    )
    log_files: list[str] = Field(
        default_factory=list,
        description=(
            "Where this stage's own module writes.  What attributes an observed "
            "tag to a chain: two chains emit ``-status`` and the file the line "
            "landed in is the only thing that separates them.  Empty for a "
            "helper that declares no logger, whose lines surface in whichever "
            "broker called it."
        ),
    )
    sites: list[str] = Field(
        default_factory=list, description="The distinct candidates this cut removed."
    )
    reasons: list[str] = Field(
        default_factory=list,
        description=(
            "What the log line said as it dropped each one, most frequent first "
            "-- ``status=offline (31)``.  These carry the measured values that "
            "made the condition true, which is what turns a condition into a "
            "reason and is why the line is better evidence than re-reading the "
            "value now."
        ),
    )
    verdict: str = Field(default=UNSETTLED, description=f"{SEEN} | {ELIMINATED} | {UNSETTLED}")
    because: str = Field(default="")


class FunnelStep(BaseModel):
    """How many candidates were left after one step, as production counted them.

    Reported in the *map's* order, and aggregated rather than split into
    passes.  Splitting was measured and refused: over 51,128 adjacent pairs
    within one ``(file, machine, run key)``, 11.0% of them show the count going
    *up*, and the largest single offender is inside a pass rather than at its
    boundary.  So "the count only falls, therefore a rise is a new pass" is a
    property the code looks like it has and does not, and every way of guessing
    a boundary this check has tried cost it a false finding.
    """

    label: str
    order: Optional[int] = Field(
        default=None, description="Position in the chain, or None where the map has no step for it."
    )
    most: int = Field(default=0, description="The largest count seen after this step.")
    fewest: int = Field(default=0, description="The smallest.")
    seen: int = Field(default=0, description="How many times the step reported at all.")


class Localization(BaseModel):
    """Which step of a chain threw the candidates away.

    The answer to the second symptom class.  ``follow_up`` has no counterpart
    here: a filter stage carries no entry points, and pooling the chain's
    triggers over the module's junctions is the same over-approximation that
    was reverted once already, so what would have gone in it is reported as a
    gap instead.
    """

    chain: str = Field(default="", description="module::function holding the chain.")
    chains_sharing_the_file: list[str] = Field(
        default_factory=list,
        description=(
            "Other chains whose lines land in the same file.  A task broker calls "
            "a job broker and passes its own log slot, so one file holds two "
            "chains and a tag can belong to either.  Named rather than chosen "
            "between: the map records where a stage's own module writes, not who "
            "delegated to it."
        ),
    )
    log_files: list[str] = Field(default_factory=list)
    cuts: list[StageCut] = Field(
        default_factory=list, description="Ranked by how much of the list each took."
    )
    describes: str = Field(
        default="",
        description=(
            "What the description resolved to, when the map holds a stage for "
            "it: the ``criteria=-`` tag production writes per rejected "
            "candidate, or the funnel label it prints per count.  Empty for a "
            "chain, whose key is an owner and names no single stage."
        ),
    )
    emitted_by: list[StageCut] = Field(
        default_factory=list,
        description=(
            "The stages the map says write that tag.  Beside ``cuts`` rather "
            "than inside it because the two are known at different times and a "
            "reader must not confuse them: this is what the map can say before "
            "anything is asked, while a cut's rank is what production measured. "
            "Without it a described cut was never named at all -- the chain was "
            "printed and the stage inside it was left to be found among a "
            "hundred and nine, as deep as forty-seventh."
        ),
    )
    funnel: list[FunnelStep] = Field(default_factory=list)
    entered: Optional[int] = Field(
        default=None, description="Candidates the first step of the chain counted."
    )
    left: Optional[int] = Field(default=None, description="Candidates the last step counted.")
    passes: int = Field(
        default=0, description="How often the chain reported -- not used to split, only stated."
    )
    sample: str = Field(
        default="partial",
        description=(
            "complete | partial.  Only a complete sample licenses saying a stage "
            "removed nothing: an answer cut off at a byte or match bound is "
            "indistinguishable from a stage that never fired."
        ),
    )


#: Why a backward walk stops instead of continuing.  Each is a terminal
#: category -- a place the value came from that this map does not explain --
#: and naming which one is the answer, not the absence of one.
#:
#: The first six are reached by a :class:`Lead`, hop by hop through the map.
#: The last four are reached by the use-time trace, which walks the source of
#: one arm and gets further down into an expression than the map's own edges
#: do.  One vocabulary for both, because it is one question and two spellings
#: of it drift.  Measured on a prototype walk over the map's 983 locatable
#: arms: the configuration terminal fires 545 times and the external ones 34,
#: neither of which had anywhere to be recorded before this.
STOP_NO_WRITER = "nothing in the map writes that field"
STOP_SHARED_TABLE = "a table the map only ever reads"
STOP_NEEDS_VALUE = "read by the condition, but its value was not observed"
STOP_AMBIGUOUS = "several subjects declare that name and the map cannot say which"
STOP_DESCENT = "another entity's rows, which this derivation does not census"
#: The read side's terminal, and the one the analyser asked for.  A statement
#: that keeps a row hands it to something the map holds a junction for -- the
#: ``holding`` query builds a ``Watcher`` and starts it -- so the map can name
#: exactly where to look next and cannot say what the value became there,
#: which is an observation rather than a derivation.
STOP_HANDED_ON = "handed to a junction the map holds, whose outcome was not observed"
STOP_CONFIG = "a configuration value, set outside the code"
STOP_EXTERNAL = "a value another system supplies"
STOP_UPSTREAM = "a value the database supplies, whose writers the map names"
STOP_PARAMETER = "chosen by whoever called or built this"


#: Where a lead came from, kept because the three are not equally trustworthy
#: and a trace that cannot tell them apart makes the map's coverage read better
#: than it is.  ``map`` is an edge the extraction recorded; ``callee`` is one
#: hop along a ``self.<method>()`` call, deterministic but assembled here;
#: ``gloss`` is a reading's proposal, a hypothesis the next hop's evidence
#: checks; ``reader`` is a statement that selects a row and hands it on, which
#: is an edge the extraction recorded like ``map`` but from the read side --
#: kept apart because the three map-side suppliers all answer "who wrote this"
#: and this one answers "who picked it up".  ``gloss`` alone does not exist yet.
LEAD_MAP = "map"
LEAD_CALLEE = "callee"
LEAD_GLOSS = "gloss"
LEAD_READER = "reader"


class Lead(BaseModel):
    """Where the answer continues after this one, or the reason it stops.

    **The map is asked once per hop, not once per investigation.**  A symptom is
    not one question but a path: "the task sits in ``finishing``" opens the
    read, the modify and the write, and learning that the modify stalled opens a
    different question -- which job it waits on -- that nobody typed and that
    the evidence, not the description, chose.  The vocabulary being closed is
    what makes each hop a selection out of a known set; the set of *paths* is
    not closed and does not need to be.

    So this deliberately does not recurse.  A hop needs evidence before the next
    one is worth taking, and evidence collection is a separate cadence -- a
    ``derive`` that walked would have to reach production from inside the
    derivation, which is the split the two phases exist to keep.  The loop
    belongs to whoever holds the history: enumeration is the map's job and
    selection is the reasoning's, and that seam is where an LLM can sit while
    still being checkable.

    A lead that continues carries a whole :class:`Symptom`, so the caller can
    put it straight back in.  That is the contract, and it is why a field read
    without a value *stops*: naming something to go and observe is a true and
    useful answer, but it is not yet a question the map can be asked.
    """

    field: str = Field(..., description="What was carried or read, qualified where known.")
    symptom: Optional[Symptom] = Field(
        default=None,
        description="The next question, ready to ask.  None when the walk stops here.",
    )
    stop: str = Field(
        default="",
        description="Why it stops, when it does -- one of the STOP_* categories.",
    )
    why: str = Field(default="", description="What opened this, in one clause.")
    opened_by: str = Field(
        default="",
        description=(
            "The candidate or cut that opened it.  Kept so that eliminating a "
            "candidate takes its leads with it: a walk that keeps descending "
            "from a branch the evidence ruled out is following a path the "
            "system did not take."
        ),
    )
    source: str = Field(
        default=LEAD_MAP,
        description=(
            "Which of the LEAD_* suppliers proposed it.  Separate from "
            "``opened_by``, which names *whose* survival it depends on and is "
            "what the eliminator filters by; this names *how sure* it is."
        ),
    )


#: What a :class:`TraceStep` is a step of.  ``write`` is the arm itself;
#: ``binding`` and ``loop-target`` are reaching definitions inside one
#: function; ``handover`` is the one crossing the map supplies, from a
#: worker's own attribute to the expression the knight built it with; and
#: ``unbound`` is a name this function does not bind, which is where the walk
#: either crosses or stops.  ``print`` is a logging call a reader started
#: from, with the names its message reads; ``return`` is below.
TRACE_WRITE = "write"
TRACE_PRINT = "print"
#: What a called function hands back, where the walk went into the call to
#: explain a value bound from it.  Named after the call, not after the name it
#: was bound to: its expression is in the callee's terms.
TRACE_RETURN = "return"
TRACE_BINDING = "binding"
TRACE_LOOP = "loop-target"
TRACE_HANDOVER = "handover"
TRACE_UNBOUND = "unbound"

#: :attr:`TraceStep.role` of a step that says which row a loop was on.
ROLE_ITERATION = "iteration"


#: What a skeleton row is.  ``branch`` is a compound statement's header and
#: carries no pattern; ``print`` is a line production writes; ``arm`` is one of
#: the writes the map sent the reader here for.
SKELETON_BRANCH = "branch"
SKELETON_PRINT = "print"
SKELETON_ARM = "arm"


class SkeletonHole(BaseModel):
    """One hole of a printed line, and the expression the source fills it with.

    The pattern says where a value goes and how far it can run; this says
    *which* value.  A reader who has the line in hand reads each hole back
    against its expression -- ``prodSourceLabel`` out of ``{str(e)} :
    {self.job.prodSourceLabel} in {self.job.jobStatus}`` -- which the pattern
    alone cannot tell it.
    """

    expression: str = Field(description="What fills the hole, as the source spells it.")
    prints: str = Field(
        default="token",
        description="token | text | number -- how the value prints, read off the syntax.",
    )
    appended: bool = Field(
        default=False,
        description=(
            "Whether this hole is what ``+=`` adds after the message was first "
            "written.  Always the last hole, and left out of the pattern."
        ),
    )


class SkeletonLine(BaseModel):
    """One row of what a function prints, in source order, with the arms in place.

    The artefact a reader aligns a grepped log region against.  The map's
    ``log_pattern`` is one sentence for a whole function, settled at build time
    by majority over every writer of the subject; matched against a region it
    says the function ran and nothing more.  A skeleton says *which branch*,
    because the lines either side of an arm are in the region too and the
    nesting says which of them can have been printed together.

    Deliberately not one row per arm.  A line that reports nothing about the
    value -- ``log.debug(blah1)`` before a write and ``log.debug(blah2)`` after
    it -- still proves the write ran when both turn up, and a shape that
    insists every row name the value cannot express that at all.  ``value`` is
    the stronger reading kept beside the weaker one: where the observed value
    does land in a hole, the pattern with it filled in is narrower than the
    pattern without, so both are rendered and the row carries each.

    Not written into the map as it stands.  Like the rest of the walk it is a
    reading of the tree the map was built from, and is refused outright when
    that is not the tree at hand.  The print rows' fields that no question
    changes are stored, once per function -- see :class:`FunctionPrintsNode`.
    """

    kind: str = Field(description=f"{SKELETON_BRANCH} | {SKELETON_PRINT} | {SKELETON_ARM}")
    line: int = Field(description="Where in the file this row is.")
    depth: int = Field(default=0, description="How deep the nesting is at this row.")
    text: str = Field(
        default="",
        description="The branch's header or the arm's statement, as the source spells it.",
    )
    pattern: str = Field(
        default="",
        description="The line as a pattern, with every hole left open.  Empty when refused.",
    )
    refused: str = Field(
        default="",
        description="Why no pattern was rendered, when none was -- shown in its place.",
    )
    keyed: bool = Field(
        default=False,
        description=(
            "Whether the pattern is a question only together with the entity's "
            "key: too little literal to ask on its own -- ``Try:`` -- and "
            "exactly the question among one job's lines."
        ),
    )
    holes: list[SkeletonHole] = Field(
        default_factory=list,
        description="Every hole of the pattern, in order, with what fills it.",
    )
    prefix: str = Field(
        default="",
        description=(
            "What the logger writes before the message, as a pattern: the "
            "prefix a ``LogWrapper`` was built with here, or by the function "
            "that built it -- ``getPandaIDsWithTask_JEDI\\ <\\ .*?\\ >\\ `` "
            "for a proxy method's ``create_tagged_logger``.  Apart from "
            "``pattern`` because the line holds it before the message, and it "
            "names the writer where the message alone -- ``start``, ``done`` "
            "-- is shared by many.  Empty where none is read, and then "
            "``prefix_hole`` says why when the logger may write one."
        ),
    )
    prefix_hole: str = Field(
        default="",
        description=(
            "Why ``prefix`` is empty where the logger may write one: the "
            "row has no pattern, or the logger is not a local this function "
            "builds by a ``LogWrapper`` whose prefix folds -- ``self.logger`` "
            "built in ``__init__``, a parameter, the module's logger, a "
            "``MsgWrapper``, a prefix of ``None`` (a time stamp).  Empty "
            "where ``prefix`` is the whole reading: said, or a "
            "``LogWrapper`` built here with none.  Apart from ``prefix``, "
            "which a reader narrows records by and reads empty as not "
            "narrowing: an empty ``prefix`` alone does not say the line "
            "starts with the message."
        ),
    )
    arms: list[int] = Field(
        default_factory=list,
        description=(
            "The arms this row can have been printed alongside.  One of them "
            "means the row names which arm ran; several mean it does not."
        ),
    )
    value: str = Field(
        default="",
        description="The same line with the observed value in its hole, where one lands.",
    )
    hole: str = Field(
        default="", description="The expression whose place the observed value takes."
    )
    because: str = Field(
        default="", description="Why that hole is the one the value fills."
    )


class PrintRow(BaseModel):
    """One way a logging call can print: a :class:`SkeletonLine` print row less the question.

    The fields are the skeleton's, read the same way and meaning the same
    thing.  What is left out is what an asking changes -- the arms a row was
    printed alongside, and the observed value in its hole.
    """

    text: str = Field(description="The message as the source spells it, or as a binding wrote it.")
    pattern: str = Field(default="", description="As :attr:`SkeletonLine.pattern`.")
    refused: str = Field(default="", description="As :attr:`SkeletonLine.refused`.")
    keyed: bool = Field(default=False, description="As :attr:`SkeletonLine.keyed`.")
    holes: list[SkeletonHole] = Field(
        default_factory=list, description="As :attr:`SkeletonLine.holes`."
    )
    prefix: str = Field(default="", description="As :attr:`SkeletonLine.prefix`.")
    prefix_hole: str = Field(default="", description="As :attr:`SkeletonLine.prefix_hole`.")


#: How a print statement's ``log_files`` was found -- see
#: :func:`~bamboo.codemap.panda.recognizers.logfile.statement_files`.
#: ``logger``: the logger the call is made on was followed, on every path, to
#: the files it opens, so a line absent from all of them was not printed.
#: ``partly``: some paths were followed and some were not, and the module's own
#: files stand in for the rest -- the list can name a file the line never
#: reaches.  ``module``: nothing was followed, and the list is the module's.
LOG_FILES_FOLLOWED = "logger"
LOG_FILES_PARTLY = "partly"
LOG_FILES_OF_MODULE = "module"


class PrintStatement(BaseModel):
    """One logging call and every way it can print, in the order the skeleton gives them."""

    line: int = Field(description="The statement's first line, where the skeleton puts its rows.")
    level: str = Field(description="The logger method called: debug, info, warning, error, critical.")
    rows: list[PrintRow] = Field(
        default_factory=list,
        description=(
            "One per way the call can print, at least one: a message that "
            "cannot be written out still has a row, with the reason."
        ),
    )
    log_files: list[str] = Field(
        default_factory=list,
        description=(
            "The files this call's lines land in: those of the logger it is "
            "made on, followed to where that logger was opened.  A logger the "
            "caller handed in lands in the caller's file."
        ),
    )
    log_files_basis: str = Field(
        default="",
        description=(
            f"How ``log_files`` was found: {LOG_FILES_FOLLOWED!r}, "
            f"{LOG_FILES_PARTLY!r} or {LOG_FILES_OF_MODULE!r}."
        ),
    )
    runs: list[int] = Field(
        default_factory=list,
        description=(
            "The lines of the owner's runs (:attr:`FunctionPrintsNode.runs`) "
            "whose logger this call can print through.  Empty where its "
            "logger is none of them -- a parameter, the module's logger -- "
            "and then the rows' ``prefix_hole`` says how it is bound."
        ),
    )


#: :attr:`RunKey.columns` for a mark the logger stamps with the time it was made.
RUN_KEY_TIME = "time"

#: Each :attr:`KeyStop.kind`, and how :meth:`KeyStop.said` says it.  The
#: wording is here rather than in the dump: a reader gets this module's, and
#: rewording it rebuilds no map.
KEY_STOPS: dict[str, str] = {
    "parameter": "{name} is a parameter of {at}: what the callers hand is not read",
    "outside": "{name} is bound outside {at}",
    "bound": "{name} is {what} at {at}",
    "element": "{name} is an element of {what} at {at}",
    "value": "it is {what}",
    "undeclared": "the class of {name} is not declared at {at}",
    "several": "{name} is declared as several classes at {at}: {what}",
    "declaration_failed": "reading the declaration of {name} failed: {what}",
    "no_row": "{name} holds no row",
    "spread": "handed to *args or **kwargs at {at}",
    "hole": "handed to a call not placed at {at}",
    "cut": "following it from {at} stopped at its cap on {what}, so it can be used further than read",
    "capped": "a path holding it, {name}, has too many parts to be followed, at {at}",
    "no_class": "keys {name}, a table no class holds, at {at}",
    "not_declared": "keys {name}, which {what} does not declare, at {at}",
    "no_use": "no use of it in {at} or the calls it is handed to names a column",
    "failed": "reading its uses failed: {what}",
}


class KeyStop(BaseModel):
    """Where one reading of a run key's value ended without a column of a class holding rows (:attr:`RunKey.stops`).

    ``pandaID`` handed to ``Watcher.__init__`` is a parameter there, and what
    the callers hand is not read: ``KeyStop(kind="parameter", name="pandaID",
    at=["Watcher.__init__"])``.  One stop said at several sites is one, with
    each site.
    """

    kind: str = Field(description=f"What ended it: one of {', '.join(KEY_STOPS)}.")
    name: str = Field(
        default="",
        description=(
            "What it is said of: the value read there (``pandaID``, "
            "``self.masterIndexName``), the class (``AdderGen``), or the "
            "table's column (``T_TASK.taskID``)."
        ),
    )
    what: str = Field(
        default="",
        description=(
            "What that is there: ``what destination_blocks.items() returns``, "
            "``a literal``; the classes or the class a column is not declared "
            "by; an error's name."
        ),
    )
    at: list[str] = Field(
        default_factory=list,
        description="``owner:line`` of each site, or the function, in the order met.",
    )

    def said(self) -> str:
        return KEY_STOPS[self.kind].format(name=self.name, what=self.what, at=", ".join(self.at))


class RunKey(BaseModel):
    """One hole of a run's mark, and the columns the value it holds is.

    A reader matching a history row to a run (the querier's F1) asks which
    column of the row's class the run's key is.  A declaration says it where
    the hole is an attribute of a name declared as one class.  Elsewhere the
    value is read where it is used -- the key of a row a statement writes, a
    term a statement picks its rows by, in the function and wherever it is
    followed from there (:func:`bamboo.codemap.objects.follow`: the calls it
    is handed to, whole or inside a list) -- and where a binding in the
    function gives it an attribute of a declared name
    (:mod:`bamboo.codemap.panda.keycolumns`).  ``TaskRefinerThread.runImpl``'s
    ``jediTaskID`` is the key of the rows ``getTaskParamsWithID_JEDI`` reads;
    ``killJob``'s ``pandaID`` the key of the job rows it updates.

    **Each column is one use, not every one.**  A column is what the value
    is at the use it was read at, so the columns can name two attributes of
    one class -- ``retryTask_JEDI``'s ``jediTaskID`` keys the task and picks
    its children by ``parent_tid`` -- and a value used nowhere the reading
    goes may be a column it does not name.  What the callers hand is not
    read.
    """

    expression: str = Field(description="What fills the hole, as source; empty for a time stamp.")
    columns: list[str] = Field(
        default_factory=list,
        description=(
            "``Class.attribute`` for each column the value is read to be, sorted: "
            "the declared attribute, then those its uses say, spelled as the "
            "class declares the attribute; a table no class holds gives none.  "
            f"[{RUN_KEY_TIME!r}] for a time stamp."
        ),
    )
    stops: list[KeyStop] = Field(
        default_factory=list,
        description=(
            "Where ``columns`` names no column of a class holding rows, where "
            "each reading stopped: what the value is bound to first, then its "
            "uses.  Empty wherever one is named, and for a time stamp."
        ),
    )

    @property
    def why(self) -> str:
        """Why ``columns`` names no column of a class holding rows: ``stops``, said (:data:`KEY_STOPS`)."""
        return "; ".join(stop.said() for stop in self.stops)


class RunStart(BaseModel):
    """Where a run starts: a logger bound with the mark every line printed through it carries.

    ``tmpLog = MsgWrapper(logger, f"<jediTaskID={taskSpec.jediTaskID}>")``
    starts one.  The lines printed through ``tmpLog`` until it is bound again
    are that run's, and the mark says which task they are about -- the key a
    reader picks one run's lines out of a log by.  The builders, and which
    argument each takes its mark from, are PanDA's logger factories
    (:data:`bamboo.codemap.panda.runs.BUILDERS`).  An object's run is a
    logger the class's ``__init__`` binds to ``self`` -- ``Watcher`` and
    ``AdderGen`` keep one per job -- and holds the lines of every method.
    """

    line: int = Field(description="The line of the statement that binds the logger.")
    at: str = Field(
        description=(
            "The function that statement is in: the owner itself, or the "
            "class's ``__init__`` for an object's run."
        )
    )
    logger: str = Field(description="The name bound: ``tmpLog``, ``self.logger``.")
    builder: str = Field(description="The factory called: ``LogWrapper``, ``MsgWrapper``, ``create_tagged_logger``, ``make_logger``.")
    mark: str = Field(
        default="",
        description=(
            "The argument the mark is made from, holes as ``{}``: "
            "``<jediTaskID={}>``, ``PandaID={}``.  Empty for none and for a "
            "time stamp."
        ),
    )
    key: list[RunKey] = Field(default_factory=list, description="One per hole of the mark, in order; a time stamp's one.")
    object: bool = Field(default=False, description="Whether it is an object's run, bound to ``self`` in ``__init__``.")


class FunctionPrintsNode(BaseNode):
    """Every line one function can print, for every function in the corpus that prints.

    The querier lays these rows over a log region to say which code ran, and
    until this existed it got them by walking: ``trace.walk`` for each
    candidate function, keeping the skeleton's print rows and throwing the
    rest away.  In the querier's case A that was 210 of 293 seconds.  The rows
    themselves cost the build 8.7 seconds for all 1093 functions, and 27 of
    them sampled came out the same as the walk's field for field: nothing in
    a print row but the arms and the observed value depends on what was asked.

    **One node per function, every function that prints.**  ``LogSiteNode``
    stops at the owners the map names because no reader had asked for more;
    this one has a reader, and the functions it reads come from tracebacks,
    callers and movers as much as from the map.  Per function rather than per
    call because a function is what the reader asks for, all of it at once.

    ``count`` is what lets a reader trust an absence: a line that should be
    there and is not means the code did not pass it, which is only true when
    no call is missing.  The build checks it against the calls themselves
    (``prints-are-counted``).

    ``owner`` spells the nesting: ``file::Class.method``,
    ``file::function``, ``file::outer.inner``.  The bare name collides for
    seventeen of these functions -- ``datasetManager.py`` has seven named ``run``.

    ``log_files`` is where the rows land, so a reader knows which log to lay
    them over.  ``JobComplexModule.archiveJob`` writes ``start status=...``
    into ``panda-DBProxy.log`` or ``panda-JediDBProxy.log``, and the only
    files a reader had been given for it were its junction's
    ``observable_log_files`` -- its callers', the right answer to where a line
    saying the value was settled appears and the wrong one here.

    **Per statement, because the logger is.**  Each statement carries the files
    of the logger it is called on, and the function's list is their union.
    The module's own file was the first answer, and for 4,735 of the 5,159
    calls in modules that have one it is still the answer.  It is not for the
    rest.  ``TypicalWatchDogBase.pre_action`` prints all 30 of its lines
    through the ``tmpLog`` that ``WatchDog`` hands it, so they land in
    ``panda-WatchDog.log`` and in none of the three files its module's
    subclasses declare.  ``AtlasProdJobBroker.doBrokerage`` also writes into
    ``panda-AtlasProdTaskBroker.log`` whenever the task broker passes its own
    logger as ``glLog``.  One function can do both: ``pilot_api.acquire_jobs``
    prints 12 lines through ``_logger`` and 2 through ``pilot_logger``, and
    those two loggers are different files.  A wrong file is worse than none,
    because nothing found reads as "not printed".  So each statement also says
    how its files were found (``log_files_basis``).
    """

    node_type: NodeType = NodeType.FUNCTION_PRINTS
    map_id: str
    derived_from: str
    owner: str = Field(..., description="``file::qualified.name``, the nesting spelled out.")
    cls: str = Field(
        default="",
        description=(
            "``file::Class`` for a method, as :attr:`TraceStep.cls` spells it; "
            "empty for a function no class holds directly."
        ),
    )
    function: str = Field(..., description="The function's own name.")
    count: int = Field(..., description="How many logging calls the function makes; one statement each.")
    statements: list[PrintStatement] = Field(default_factory=list)
    log_files: list[str] = Field(
        default_factory=list,
        description=(
            "Every file one of the function's statements lands in: the union "
            "of the statements' ``log_files``."
        ),
    )
    runs: list[RunStart] = Field(
        default_factory=list,
        description=(
            "The runs the function's lines can be in: each logger it binds by "
            "a factory, and its object's run -- see :class:`RunStart`."
        ),
    )
    anchor: Optional[Anchor] = None

    @staticmethod
    def make_name(map_id: str, owner: str) -> str:
        return f"{map_id}:{owner}"


class ReachCall(BaseModel):
    """One way a function goes on into another: a call, or a function handed to one.

    ``for_classes`` and ``runs_as`` are what a method found by inheritance
    needs (:attr:`calls.Target.bound`).  ``self.runImpl()`` in
    ``WorkerThread.run`` is placed once for each class ``self`` can be there,
    and lands at ``JobGeneratorThread.runImpl`` only for a
    ``JobGeneratorThread``: that edge is ``for_classes`` = ``runs_as`` =
    ``[JobGeneratorThread]``.  A call that does not read ``self`` holds for
    every class (``for_classes`` empty), and a function entered with no class
    said (``runs_as`` empty) runs as each class its ``self`` can be.

    One row stands for every call of the function that goes the same way
    -- same :attr:`to`, :attr:`how`, classes and :attr:`thread` -- and
    :attr:`lines` holds each of their lines.  Where the closure can go
    does not depend on which of them ran, but a reader placing a call in
    the function's flow does: ``insertNewJob`` calls ``recordStatusChange``
    at 3615, inside a ``try``, and at 3620, on the other branch.
    """

    line: int = Field(
        ...,
        description=(
            "One of :attr:`lines`: the first call met reading the function "
            "depth first from its last statement back, so mostly the last in "
            "the source, not the first."
        ),
    )
    lines: list[int] = Field(
        default_factory=list,
        description=(
            "The line of every call going this way, in source order; :attr:`line` among them.  "
            "Empty in a dump built before they were recorded, where :attr:`line` is one of the calls only."
        ),
    )
    callee: str = Field(..., description="The called or handed expression, as source, cut at 80 characters.")
    to: str = Field(..., description="The :attr:`ReachNode.owner` it goes into.")
    how: str = Field(
        default="call",
        description=(
            "``call``; ``handed``, a function or method passed as an argument "
            "-- ``Thread(target=self.run)``, ``executor.submit(fn)`` -- taken "
            "as called; ``lambda``, a call in a lambda's body, taken as made "
            "by the function holding the lambda; ``setattr``, an attribute "
            "store running its object's ``__setattr__`` -- ``job.jobStatus = "
            "'closed'`` -- or ``setattr(x, name, v)``."
        ),
    )
    for_classes: list[str] = Field(
        default_factory=list,
        description="The classes ``self`` is, in this function, for which the call goes here; empty for every class.",
    )
    runs_as: list[str] = Field(
        default_factory=list,
        description="The classes ``self`` is in :attr:`to` when it is entered this way; empty for every class it can be.",
    )
    thread: bool = Field(
        default=False,
        description="``thr.start()`` running ``run`` on another thread or process (:attr:`calls.Placed.thread`).",
    )


class ReachHole(BaseModel):
    """A call whose way on this reading cannot close: it may go somewhere no edge says.

    A reach answer with a hole in its closure and no write reached is no
    answer (``None``), not "reaches none" -- the hole may be the way to it.
    """

    line: int
    callee: str = Field(..., description="The called expression, as source, cut at 80 characters.")
    why: str
    for_classes: list[str] = Field(
        default_factory=list,
        description="As :attr:`ReachCall.for_classes`: the classes of ``self`` it is a hole for; empty for every class.",
    )


class HistoryCall(BaseModel):
    """A call of a history writer, and the values the row it writes can record.

    Read once per call: every :class:`HistoryPair` whose call can lead here
    names it in ``writes``.  ``values`` are the outcomes of the function's
    arms of ``subject`` whose write can reach the call; ``passed`` is a way
    to the call on which no such arm writes, so the row can record the value
    the object came in with or was given elsewhere.  ``TASKS_STATUSLOG``
    copies the task row's status as it stands (``INSERT ... SELECT``), and a
    task arm written by SQL has the line of the statement that decides it,
    not of the execute: there the values are as near as the arms' lines are.
    """

    line: int = Field(description="The call's line.")
    subject: str = Field(description="The subject whose history the writer records: ``JobSpec.jobStatus``.")
    values: list[str] = Field(default_factory=list, description="The outcomes of the arms that reach the call, sorted.")
    passed: bool = Field(default=False, description="Whether a way to the call passes no arm of the subject.")
    hole: str = Field(default="", description="Why ``values`` is not read; empty where it is.")


#: How :class:`PathBinding` ties a name to a path: ``x = p`` (or ``self.f = p``), ``for x in p``,
#: ``x.append(p)``, ``x = f(...)`` and ``return p``.  A label of the statement: what the binding means is
#: in its name and path, which say where each holds the object.
BIND_ALIAS = "alias"
BIND_ELEMENT = "element"
BIND_FILL = "fill"
BIND_RESULT = "result"
BIND_RETURN = "return"


class PathAssign(BaseModel):
    """An attribute assigned on an object a function names by a path -- read by :mod:`bamboo.codemap.objects`.

    A path is a name of the function -- a parameter, ``self`` or a local --
    then attributes and subscripts: ``[]`` for an element at a position not
    known, ``[i]`` for a tuple's i-th, ``['k']`` for the value under a
    literal key: ``self.job``, ``jobs[]``, ``varMap[':PandaID']``.  What a
    call returns where no name takes it is named by the call,
    ``<result@line:col>``.
    Only an attribute some spec declares a column of is kept (``attrs``),
    and a write whose name is computed -- ``setattr(job, name, v)``, a
    spec's own ``pack`` -- carries the role :mod:`bamboo.codemap.panda.specwrites`
    gives it.
    """

    line: int
    path: str = Field(description="The object assigned on, in this function's names.")
    attrs: list[str] = Field(description="The attributes it can set; ``*`` for any the class declares.")
    how: str = Field(
        description=(
            "``=`` or ``+=``; for a computed name the specwrites role -- ``row_read`` sets the row's own "
            "values, ``construct`` a new object's first ones."
        )
    )


class PathBinding(BaseModel):
    """A name of a function that holds what a path holds: ``name`` holds, where it says, what ``path`` holds where it says.

    ``updated = [job]`` is ``updated[]`` holding ``job``; ``for job in jobs``
    is ``job`` holding ``jobs[]``; ``status, job = f()`` is ``job`` holding
    what is at ``[1]`` of what ``f`` returns.
    """

    line: int
    name: str = Field(
        description=(
            "The name bound, and where in it: a local, or a path written to (``self.job`` in ``self.job = job``); "
            "``<return>`` for what the function returns."
        )
    )
    path: str = Field(
        description=(
            "The path it holds; for :data:`BIND_RESULT` where in what the call returns: empty for all of it, "
            "``[1]``, ``[]``, ``.job``."
        )
    )
    how: str = Field(
        description=(
            ":data:`BIND_ALIAS` -- an assignment; :data:`BIND_ELEMENT` -- a loop over path; :data:`BIND_FILL` -- "
            "the statement puts path into name (a literal, ``append``); :data:`BIND_RESULT` -- name is what the "
            "call at the line returns; :data:`BIND_RETURN` -- the function returns path."
        )
    )
    to: list[str] = Field(
        default_factory=list,
        description=(
            "For :data:`BIND_RESULT`: the :attr:`ReachNode.owner` of each function the call lands in; empty for "
            "one outside the corpus."
        ),
    )


class CallBinding(BaseModel):
    """A path a call hands on, and the parameter of the function it lands in that takes it."""

    line: int
    to: str = Field(description="The :attr:`ReachNode.owner` the call lands in; empty where it is a hole.")
    parameter: str = Field(
        description="``self`` for the receiver; empty where the call does not say which parameter takes it (``*args``)."
    )
    path: str
    at: str = Field(
        default="",
        description=(
            "Where in what the parameter takes the path is: empty for all of it; ``[]`` for an element of a list "
            "handed, ``['jediTaskID']`` for a dict's value."
        ),
    )
    why: str = Field(default="", description="Why a hole is one, where :attr:`ReachNode.holes` does not say.")


#: The column of a key a loop over a mapping names by the mapping's key: ``WHERE {k}=:cr_{k}`` for each ``k`` of
#: ``criteria``, its value ``criteria[k]``.  Its :attr:`WrittenValue.path` is the mapping's element, ``criteria[]``:
#: the row is named, for each key the mapping holds, by the column of that name.
MAP_KEY = "[]"


class WrittenValue(BaseModel):
    """Where one column a statement writes takes its value from, or one key it picks its row by."""

    column: str = Field(description="The column; for a key, :data:`MAP_KEY` where the column is a mapping's key.")
    path: str = Field(
        default="",
        description="The value as a path of the function -- ``job.jobStatus``, ``pandaID`` -- where it is one.",
    )
    value: str = Field(
        default="",
        description="Otherwise the value as the code or the statement spells it: ``literal:'failed'``, ``expression:CURRENT_DATE``.",
    )
    why: str = Field(default="", description="Why neither is read; empty where one is.")


class DbWrite(BaseModel):
    """A statement a function runs that writes rows: the columns it writes from what, and the row it picks.

    The record :func:`bamboo.codemap.objects.may_change` reads beside the
    objects': a write takes its values from an object the function holds --
    the whole row of one (``job.valuesMap()``), its changed attributes
    (``onlyChanged=True``), one attribute bound by name -- or from another
    value, and lands on the row its keys name.
    """

    lines: list[int] = Field(
        description=(
            "The lines of the calls that run it with these values and keys: a statement run from two places"
            " with other ones is two writes."
        )
    )
    table: str
    kind: str = Field(description="``insert`` or ``update``.")
    columns: list[WrittenValue] = Field(default_factory=list, description="Each column it writes by name, a bind read once per value.")
    whole: str = Field(
        default="",
        description="The path whose ``valuesMap()`` is the bind map: every column of the row comes from that object.",
    )
    changed: bool = Field(default=False, description="The map holds only the attributes changed since the object was read.")
    unread: bool = Field(
        default=False, description="It names columns at run time that are not read: which of the row's columns it writes is not known."
    )
    keys: list[WrittenValue] = Field(
        default_factory=list, description="The ``WHERE`` terms that name its row: ``PandaID=:PandaID``."
    )
    new_row: bool = Field(
        default=False, description="An ``INSERT`` whose row is new: its key comes from a sequence (``useSeq=True``)."
    )


#: The lines :class:`ClearedLines` reads a function's entry and its normal exit as.
ENTRY_LINE = 0
EXIT_LINE = -1


class LoopLines(BaseModel):
    """A loop of a function whose body holds lines of its record: the line of its header and its body's first and last."""

    line: int
    start: int
    end: int
    again: bool = Field(default=False, description="A ``while``: its test runs again after each pass of its body.")


class LineStart(BaseModel):
    """A line of the record whose statement starts on another: a call on the second line of a statement."""

    line: int
    start: int


class LinesApart(BaseModel):
    """A line of a function and lines of others the order of its lines says it can run before, and it cannot."""

    line: int
    lines: list[int]


class Renewed(BaseModel):
    """A name a loop binds to another object between a line and a later pass's line, on every way.

    Another object: an element of what the loop or a ``with`` holds, a
    call's result, a literal, or a path rooted at a name the loop binds so
    -- ``TaskRefinerThread.runImpl`` binds ``impl`` to a new refiner for each
    task, and ``tmpTaskSpec`` to its ``impl.taskSpec``; ``job = self.job``
    gives ``job`` the object every pass holds.  A way from a line to a line
    of a later pass goes round the loop's top, so every way passes such a
    binding where every way from the first line to the top does
    (:attr:`after`) or every way from the top to the second does
    (:attr:`before`).
    """

    loop: int = Field(description="The :attr:`LoopLines.line` of the loop.")
    name: str = Field(description="The name, without its version: ``impl`` of ``impl@166``.")
    after: list[int] = Field(default_factory=list)
    before: list[int] = Field(default_factory=list)


class LineOrder(BaseModel):
    """Which lines of a function's assignments, writes and calls can run before which: :attr:`ReachNode.order`.

    Before on one run: on one pass, by a way through the function with no
    jump away from the second line (``continue``, ``return``, ``break``) and
    no test parting the two (:func:`bamboo.codemap.panda.pathcond.parted`);
    or on a later pass of a loop holding both, by a way round it whose
    passes no test parts.  Most of it the lines' order says, and only the
    rest is here.  A line can run before another on one pass where its
    statement starts no later; one starting later can run before it only on
    a later pass of a loop holding both, or before a ``while``'s test on its
    next pass.  :attr:`apart`, :attr:`rounds` and :attr:`before` are the
    pairs where that is not so: ``retryJob`` writes the job back at 4146 under ``if not
    getNewPandaID`` and sets its ``parentID`` at 4171 in the ``else`` (4171
    apart from 4146); ``TaskRefinerThread.runImpl`` updates the task at 226
    and goes on to the next task, so its register at 597 follows only on a
    later pass (226 round to 597).
    """

    loops: list[LoopLines] = Field(default_factory=list)
    starts: list[LineStart] = Field(default_factory=list)
    apart: list[LinesApart] = Field(
        default_factory=list, description="The lines it cannot run before on one run, although their order says it can."
    )
    rounds: list[LinesApart] = Field(
        default_factory=list,
        description="The lines it runs before only on a later pass of a loop holding both, although it starts no later.",
    )
    before: list[LinesApart] = Field(
        default_factory=list,
        description="The lines it can run before on one pass, although their order says it cannot (none expected).",
    )
    renewed: list[Renewed] = Field(default_factory=list)


class ClearedLines(BaseModel):
    """Lines of a function every way between which passes a reset of what changed on an object.

    A spec's changed attributes are what ``bindUpdateChangesExpression`` and
    ``valuesMap(onlyChanged=True)`` write, and ``resetChangedAttr`` /
    ``resetChangedList`` take them back out.  ``registerTaskInOneShot_JEDI``
    resets ``jediTaskID`` on ``taskSpec`` at 5800, before its ``UPDATE`` at
    5832: every way from its entry to 5832 passes it.  A call of a function
    that resets what it is handed on every way to its normal exit is a reset
    where it is called (``resetRefinedAttrs``).
    """

    path: str = Field(description="The object reset, as a path of the function.")
    columns: list[str] = Field(
        description="The attributes reset, each by these resets; ``*`` for all of them (``resetChangedList``)."
    )
    line: int = Field(description=f"Where the change was made: a line, or {ENTRY_LINE} for before the function was entered.")
    lines: list[int] = Field(description=f"Each line, or {EXIT_LINE} for its normal exit, every way to which from :attr:`line` passes the reset.")


class ReachNode(BaseNode):
    """Where one function can go on to, for every function in the corpus: the record ``reach`` is answered from.

    The question is bamboo-8f's (10-14): does a function that moves a
    subject off a value reach, through calls and hand-offs, a write that
    moves it off again.  Its answer is a closure over every function below
    the one asked, and the facts per function do not depend on the subject
    or the value -- only where the closure ends does, and the junctions say
    that.  So the facts are recorded once per build and the question is
    answered from the record (:func:`bamboo.codemap.reach.reach`), not by
    placing every call again for each of the hundreds of movers one run asks
    about.

    Every call the function makes is one of four things.  It goes on into a
    function of the corpus (:attr:`calls`); it is the database
    (:attr:`database` -- a statement the map reads, or the driver -- whose
    writes are read where it runs, so the closure goes no further); it is
    closed by a rule (outside the corpus, a builtin's method) and counted in
    :attr:`settled` where the rule is an assumption a reader must be able
    to name; or it is a :attr:`holes` entry.  A call recorded nowhere was
    closed by one of the premises every answer shares
    (:data:`bamboo.codemap.reach.PREMISES`).

    ``owner`` spells the nesting as :attr:`FunctionPrintsNode.owner` does,
    and a second definition of one name in a file -- a property's setter
    beside its getter -- is told apart by ``#2``.
    """

    node_type: NodeType = NodeType.REACH
    map_id: str
    derived_from: str
    owner: str = Field(..., description="``file::qualified.name``, ``#n`` after a repeated one.")
    cls: str = Field(default="", description="``file::Class`` holding the method; empty for a function.")
    function: str = Field(..., description="The function's own name.")
    classes: list[str] = Field(
        default_factory=list,
        description=(
            "Every class ``self`` can be here: the class and each of its "
            "subclasses in the corpus (:meth:`calls.Calls.objects_of`).  "
            "Empty for a function."
        ),
    )
    calls: list[ReachCall] = Field(default_factory=list)
    database: list[int] = Field(
        default_factory=list,
        description="The lines of the calls that are the database: a statement the map reads at that call, or the driver.",
    )
    holes: list[ReachHole] = Field(default_factory=list)
    settled: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "How many calls each assumption closed here, by its name in "
            ":data:`bamboo.codemap.reach.ASSUMPTIONS`.  An answer lists the "
            "ones its closure relied on, so a reader dropping a mover can "
            "say what that rests on."
        ),
    )
    history: list[HistoryCall] = Field(
        default_factory=list,
        description=(
            "The calls it makes of a function that writes a history row, each "
            "with the values the row can record -- see "
            ":mod:`bamboo.codemap.panda.runs`.  Not where the closure goes: "
            "the calls are in :attr:`calls` too."
        ),
    )
    assigns: list[PathAssign] = Field(
        default_factory=list,
        description="The columns it assigns on an object it names by a path (:mod:`bamboo.codemap.objects`).",
    )
    bindings: list[PathBinding] = Field(
        default_factory=list, description="Its names that hold what a path holds."
    )
    call_bindings: list[CallBinding] = Field(
        default_factory=list,
        description=(
            "The paths its calls hand on, receivers included, and the parameter each lands in.  "
            "A call into code outside the corpus is not here: that code assigns no attribute of what it is handed "
            "(:data:`bamboo.codemap.objects.PREMISES`)."
        ),
    )
    writes: list[DbWrite] = Field(
        default_factory=list,
        description="The statements it runs that write rows, with where each value and key comes from.",
    )
    order: Optional[LineOrder] = Field(
        default=None,
        description=(
            "Which lines of its assignments, writes and calls can run before which: read by "
            ":func:`bamboo.codemap.objects.may_change`, an assignment changes a row at a write only before it.  "
            "None for a function with fewer than two."
        ),
    )
    cleared: list[ClearedLines] = Field(
        default_factory=list,
        description="The lines every way between which passes a reset of an object's changed attributes.",
    )
    anchor: Optional[Anchor] = None

    @staticmethod
    def make_name(map_id: str, owner: str) -> str:
        return f"{map_id}:{owner}"


class ReachWrite(BaseModel):
    """A function in the closure holding arms the question is about, and how it is reached."""

    function: str = Field(..., description="The :attr:`ReachNode.owner`.")
    arms: list[str] = Field(default_factory=list, description="``outcome@line`` of each arm.")
    chain: list[str] = Field(
        default_factory=list,
        description=(
            "``file:line`` of each call from the start to it, the first call first; empty for the start itself.  "
            "A hop's line is its :attr:`ReachCall.line`; the other calls going that way are in :attr:`ReachCall.lines`."
        ),
    )


class Reach(BaseModel):
    """What :func:`bamboo.codemap.reach.reach` answers for one start.

    Computed at use time from :class:`ReachNode` rows; not stored.

    ``writes`` is ``None`` where no write was reached and the closure has a
    hole -- the answer is not known -- and ``[]`` where no write was reached
    and there is no hole.  Those two must not be read alike: only ``[]``
    says the start cannot move the subject, and it says so under the
    premises every answer shares and the :attr:`assumptions` listed.
    """

    writes: Optional[list[ReachWrite]] = None
    holes: list[str] = Field(
        default_factory=list,
        description="``owner:line callee -- why`` for each hole in the closure.",
    )
    assumptions: dict[str, int] = Field(
        default_factory=dict,
        description="Each assumption the closure relied on, with how many calls it closed there.",
    )
    closure: int = Field(default=0, description="How many functions the closure holds, the start included.")


class TraceStep(BaseModel):
    """One thing the source says about why an arm ran with the value it did.

    Computed at use time from the tree, not stored in the map.  The analysis
    is the same one the build already does -- reaching definitions and
    dominating guards -- and the cadence is the whole difference: storing it
    forces one condition string per branch, and that is what made the eager
    version need a loop rule, a length cap and a recursion rule before it
    could be written down at all.  A use-time answer may be a *set*, so none
    of those constraints arise; four sites setting one flag are four steps
    because four of them are the answer.

    ``guards`` is what the path condition says and ``unseen`` is what it
    cannot: only ``ast.If`` contributes to a path condition, so a step inside
    a handler, a loop or a ``try`` body carries an empty ``guards`` and is not
    unconditional.  86% of the map's arms have at least one of those on the
    way to them, which is why the two are separate fields rather than one
    list -- a reader that cannot tell them apart reads silence as certainty.

    ``value`` is the right-hand side alone, so ``operator`` carries the rest
    of the assignment: printing ``timeOutVal = 60`` for ``timeOutVal *= 60``
    states a value the line never produced.  A reader that joins the two gets
    the statement back.
    """

    kind: str = Field(..., description="One of the TRACE_* kinds.")
    name: str = Field(default="", description="The name this step explains.")
    owner: str = Field(default="", description="module::function the site is in.")
    cls: str = Field(
        default="",
        description=(
            "``file::Class`` whose method holds the step's statement; empty at "
            "module level or where the walk cannot tell.  ``owner`` names the "
            "function alone, and one file can define a method of that name in "
            "several classes -- ``TaskBroker.py``'s three ``__init__`` -- so a "
            "reader keeping the bindings of ``self.<field>`` that belong to one "
            "object asks this, with the class's bases, and not ``owner``."
        ),
    )
    file: str = Field(default="")
    line: int = Field(default=0)
    value: str = Field(default="", description="The expression, as source.")
    operator: str = Field(
        default="=",
        description="How the value reaches the name: '=', or '+=' and friends.",
    )
    guards: list[str] = Field(
        default_factory=list, description="The path condition of this site."
    )
    unseen: list[str] = Field(
        default_factory=list,
        description="What constrains reaching it that the path condition omits.",
    )
    reads: list[str] = Field(
        default_factory=list, description="Names this step's value depends on."
    )
    terminal: str = Field(
        default="",
        description="Empty while the walk continues, else one of the STOP_* categories.",
    )
    detail: str = Field(default="", description="Which interface, key or caller.")
    depth: int = Field(default=0, description="Hops from the arm.")
    role: str = Field(
        default="",
        description=(
            "``iteration`` for a step that says which row or file a loop was "
            "on -- the element and what it was drawn from -- rather than how a "
            "value the guard reads was derived.  A reader comparing two "
            "guards' derivations leaves these out, or the loop's own rows read "
            "as a difference between them."
        ),
    )
    printed: list[str] = Field(
        default_factory=list,
        description=(
            "Patterns of the lines the same block prints, where the step is a "
            "place something was set: which of thirteen ``set_fatal()`` calls "
            "ran is told by what printed beside it."
        ),
    )
    open: str = Field(
        default="",
        description=(
            "On a ``return`` step: why some call that entered this function was "
            "placed open -- here, and possibly somewhere this reading cannot "
            "name, such as wherever ``JediTaskBufferInterface`` forwards a "
            "call -- so the value may instead come from a function the walk "
            "did not enter.  Said of the function, not of one call: every "
            "return step of it carries it, whichever call the walk came in by.  "
            "On a ``handover`` step from a call into the function -- a "
            "parameter's caller, or a call of the setter that sets the field -- "
            "why that call was placed open: it may land elsewhere, and then it "
            "is not the call that handed the value.  "
            "Empty where every call that entered it was placed closed."
        ),
    )


class StatementExit(BaseModel):
    """One way a row this statement asked for stops being followed, and what it says.

    The map stores the coordinate; this is the reading of it.  Two of them
    answer different questions and the reader has to be told which they are
    looking at: a ``continue`` in the loop over the rows means the row *came
    back* and was passed over, so the query is fine and the handling is the
    thing to read; ``no_rows`` means nothing came back at all, which sends
    them to the statement's ``conditions`` instead.

    ``skeleton`` is what the block prints on the way out -- the line to grep
    the log for.  **Empty is the finding**, not a gap: six of the corpus's
    fourteen no-rows exits print nothing at all, and a row that vanishes in
    silence is precisely the case a reader cannot resolve from a log and has
    been asking the map about.
    """

    kind: str = Field(description="``continue`` | ``break`` | ``return`` | ``raise`` | ``except``")
    line: int = Field(default=0, description="Where the exit is taken.")
    no_rows: bool = Field(
        default=False,
        description=(
            "True when the exit is guarded by a test on the result rather "
            "than taken inside the loop over the rows -- nothing came back."
        ),
    )
    decided_at: int = Field(
        default=0, description="The line of the ``if`` or ``except`` that decides it."
    )
    skeleton: list[SkeletonLine] = Field(
        default_factory=list,
        description="What the block holding this exit prints; empty means it is silent.",
    )
    trace: list[TraceStep] = Field(
        default_factory=list,
        description=(
            "Why the exit fired, walked from the line that decides it -- the "
            "answer to 'the row came back and was dropped, by what?'"
        ),
    )
    trace_note: str = Field(default="", description="Why the walk is short, where it is.")


#: The kinds of :class:`ReaderGap`.  A statement of the function: read in no
#: node, handed in by the caller, or read only in part.  The whole function, not
#: read from the answer's tree: outside it, unreadable there, or another blob
#: than the map read.  In this order a reader is told them.
GAP_UNNODED = "unnoded"
GAP_HANDED_IN = "handed_in"
GAP_PARTIAL = "partial"
GAP_OUTSIDE = "outside"
GAP_UNREADABLE = "unreadable"
GAP_OTHER_TREE = "other_tree"
GAP_KINDS = (GAP_UNNODED, GAP_HANDED_IN, GAP_PARTIAL, GAP_OUTSIDE, GAP_UNREADABLE, GAP_OTHER_TREE)


class ReaderGap(BaseModel):
    """One statement of a mover the map could not read on these rows -- or, with no line, the whole function.

    A record, not a sentence: the kind of gap, where it is, and what the
    build's reading said there; the text a reader prints is made from it when
    read (:func:`bamboo.codemap.strategy.gap_texts`).  One per statement, so a
    function with a statement read in no node and another read in part says
    both: a sentence per function named the first and hid the second.
    """

    kind: str = Field(description="One of :data:`GAP_KINDS`.")
    file: str = ""
    line: Optional[int] = Field(
        default=None,
        description="The line the statement runs on; none where the whole function was not read.",
    )
    verbs: list[tuple[str, str]] = Field(
        default_factory=list,
        description="``(verb, table)`` of what runs on the line; empty where no text of it was read.",
    )
    stops: list[tuple[str, str]] = Field(
        default_factory=list,
        description="``(reason, detail)`` of each place the build's reading stopped on the line.",
    )
    unresolved: list[str] = Field(
        default_factory=list,
        description="What the read site on the line could not read (:attr:`ReadSiteNode.unresolved`).",
    )

    @computed_field
    @property
    def hides_where(self) -> bool:
        """Whether the gap can hide where the rows go.

        Nothing of the statement, or of the function, was read, so its rows'
        way on is not either.  A statement read in part keeps its way read and
        hides only which rows it takes: a term of its ``WHERE`` (an ``IN`` list
        of binds named by an interpolation, a path into a JSON column).
        """
        return self.kind != GAP_PARTIAL


class GuardReading(BaseModel):
    """One function whose write requires the observed value of the row it lands on.

    The other half of "what moves this row on".  :class:`StatementReading` is
    the half the map always answered with -- a query whose ``WHERE`` selects
    the value -- and it cannot see a write that picks its rows some other way
    and only *requires* the value: ``getJobs`` moves every ``activated`` job
    the pilot is sent, and its ``SELECT`` is one the map could not read, so the
    answer for ``activated`` never named it although its ``UPDATE`` said
    ``jobStatus=activated`` in plain text.

    Per function rather than per junction: one ``UPDATE`` is several junctions
    -- a status, a time, a counter -- and each carries the same guard.
    """

    owner: str
    file: str = Field(default="", description="Where the write is.")
    writes: list[str] = Field(
        default_factory=list,
        description=(
            "The subjects the guarded write settles, the asked subject first "
            "when it is one of them."
        ),
    )
    lines: list[int] = Field(
        default_factory=list,
        description="Where the guarded branches are decided, in source order.",
    )
    outcomes: list[str] = Field(
        default_factory=list,
        description=(
            "What those branches move the asked subject to -- where the row "
            "goes next.  Empty when the guard rides on a write of another "
            "subject only."
        ),
    )
    guards: list[str] = Field(
        default_factory=list,
        description="The row preconditions, as the map stores them, that require the value.",
    )
    log_files: list[str] = Field(default_factory=list)
    also_selects: bool = Field(
        default=False,
        description=(
            "Whether a query of this function selects the value too, so it is "
            "listed on both halves of the answer."
        ),
    )
    gaps: list[ReaderGap] = Field(
        default_factory=list,
        description=(
            "Where the map could not read this function's statements on these "
            "rows, as ``movers-of-a-value-agree`` judges it, a :class:`ReaderGap` "
            "per statement (said as ``@2391 sql_where_clause, sorting_sql -- "
            "<why>`` by :func:`bamboo.codemap.strategy.gap_texts`).  ``getJobs`` "
            "is on this half because its ``SELECT`` is one the map could not "
            "read, and without this the answer named it and did not say so.  "
            "Filled at use time from the source, like the traces -- see "
            ":func:`bamboo.codemap.strategy.attach_mover_gaps`."
        ),
    )


class StatementReading(BaseModel):
    """One statement a reader runs, what else it asks of a row, and its stretch.

    The read side's counterpart to :class:`Reading`, and per statement rather
    than per function for the reason the read site node exists: ``copyArchive.py
    ::main`` asks for ``jobStatus='holding'`` three times with three different
    time limits, and one answer for the function is the wrong number of
    answers.

    Carries its own trace and skeleton rather than borrowing the function's.
    A reading walks the arms of a junction and is scoped to the whole
    function, which is right there -- the junction owns it.  A statement owns
    only ``region``, and handed the function it answers three questions with
    the same 251 rows.
    """

    owner: str
    file: str = Field(default="", description="Where the statement is.")
    blob_sha: str = Field(
        default="", description="Hash of the file the map read it from."
    )
    line: int = Field(default=0, description="The line the statement runs on.")
    selects: list[str] = Field(
        default_factory=list, description="``subject=value`` this statement asks rows by."
    )
    keys: list[str] = Field(
        default_factory=list,
        description=(
            "Subjects the statement asks rows by with a value chosen at run "
            "time -- :attr:`ReadSiteNode.keys`.  Carried because a statement "
            "asking by these alone printed an empty ``asks on``, which reads "
            "as a query that filters on nothing."
        ),
    )
    excludes: list[str] = Field(
        default_factory=list,
        description="``subject=value`` the statement asks rows *not* to hold -- :attr:`ReadSiteNode.excludes`.",
    )
    refuses: list[str] = Field(
        default_factory=list,
        description="``subject=value`` no row the statement returns holds -- :attr:`ReadSiteNode.refuses`.",
    )
    restricts: list[str] = Field(
        default_factory=list,
        description=(
            "``subject=value``, per subject the values every row the statement "
            "returns holds one of -- :attr:`ReadSiteNode.restricts`."
        ),
    )
    consults: list[str] = Field(
        default_factory=list,
        description=(
            "``subject=value`` a subquery of the statement checks rows by -- "
            ":attr:`ReadSiteNode.consults`."
        ),
    )
    tables: list[str] = Field(
        default_factory=list,
        description="The tables the statement reads -- :attr:`ReadSiteNode.tables`.",
    )
    order_by: list[str] = Field(
        default_factory=list,
        description=(
            "The ``ORDER BY`` clauses it can be built with -- "
            ":attr:`ReadSiteNode.order_by`.  With ``limits``, why a row that "
            "passes the ``WHERE`` can still not come back."
        ),
    )
    condition_trees: list[ConditionReading] = Field(
        default_factory=list,
        description="Each conditions entry as a tree -- :attr:`ReadSiteNode.condition_trees`.",
    )
    printed_by: list[PrintedBy] = Field(
        default_factory=list,
        description="The prints of the statement's text and map -- :attr:`ReadSiteNode.printed_by`.",
    )
    order_holes: list[OrderHole] = Field(
        default_factory=list,
        description="Each blanked order item and what builds it -- :attr:`ReadSiteNode.order_holes`.",
    )
    added_under: Optional[list[AddedTerm]] = Field(
        default=None,
        description=(
            "The terms some builds of the statement do not hold, with the tests "
            "that add them -- :attr:`ReadSiteNode.added_under`.  ``None`` where the "
            "map did not compute it, which is not the same answer as ``[]``."
        ),
    )
    limits: list[str] = Field(
        default_factory=list,
        description="How many rows come back at most -- :attr:`ReadSiteNode.limits`.",
    )
    call_readings: list[CallReading] = Field(
        default_factory=list,
        description="What the statement reads on each route of calls -- :attr:`ReadSiteNode.call_readings`.",
    )
    callers_complete: bool = Field(default=False, description=":attr:`ReadSiteNode.callers_complete`.")
    callers_stop: list[str] = Field(default_factory=list, description=":attr:`ReadSiteNode.callers_stop`.")
    conditions: list[str] = Field(
        default_factory=list,
        description=(
            "What else the ``WHERE`` requires, read as a conjunction -- the "
            "answer to a row that holds the value and never came back."
        ),
    )
    unresolved: list[str] = Field(
        default_factory=list,
        description=(
            "``<expression> -- <reason>`` for the parts of this statement the "
            "extraction could not follow. Carried from the node so a reader "
            "shown the statement is told where ``conditions`` stops being the "
            "whole ``WHERE``."
        ),
    )
    bind_window: list[int] = Field(
        default_factory=list, description="``[first, last]`` the binds are filled within."
    )
    region: list[int] = Field(
        default_factory=list,
        description="``[first, last]`` of the function this statement owns.",
    )
    rows_handled: list[int] = Field(
        default_factory=list,
        description=(
            "``[first, last]`` where the rows are worked through -- the loop "
            "over them, which can run past ``region``. Empty where nothing "
            "loops over them."
        ),
    )
    trace: list[TraceStep] = Field(
        default_factory=list,
        description=(
            "Why the conditions hold the values they do, walked from the "
            "assignments that fill their binds.  Empty without a source tree, "
            "and empty on purpose when the tree is not the one mapped."
        ),
    )
    trace_note: str = Field(
        default="", description="Why the walk is short, where it is."
    )
    skeleton: list[SkeletonLine] = Field(
        default_factory=list,
        description="What this statement's region prints, in source order.",
    )
    exits: list[StatementExit] = Field(
        default_factory=list,
        description=(
            "Where a row this statement asked for stops being followed, and "
            "what the code says on the way out.  Empty for most statements, "
            "which is an answer."
        ),
    )
    crosses_to: list[str] = Field(
        default_factory=list,
        description=(
            "Junction owners a row this statement keeps is carried on to -- "
            "where the investigation goes next when the statement is the lead: "
            ":attr:`ReadSiteNode.crosses_to`, every call placed by the corpus's "
            "calls, through functions that own no junction.  Empty is not "
            "'nowhere': a statement whose rows are handed back may lead to a "
            "caller that does nothing the map answers for, which the build "
            "reports rather than hides, and a call on the way may not be "
            "placed (``crosses_open``)."
        ),
    )
    crosses_open: list[OpenCrossing] = Field(
        default_factory=list,
        description=":attr:`ReadSiteNode.crosses_open`: the calls on the way, and the callers back, this reading cannot place.",
    )
    result: str = Field(
        default="",
        description=(
            ":attr:`ReadSiteNode.result`: rows of its tables, values computed "
            "from them, or the one and the other; empty where a way it is "
            "built is not read."
        ),
    )


class Reading(BaseModel):
    """One function to read, the arms it covers, and the line to read it against.

    Per function rather than per arm because that is where the sharing is: the
    arms that survive a question routinely sit together -- one function holds
    up to eleven junctions -- and asking about each separately reads the same
    text over again and invites two answers about one piece of code.

    ``log_pattern`` is empty for an arm the map records no line for.  That is
    not a defect to be papered over: of 1046 branches, 988 can be pointed at in
    the source and 484 carry a line that production would print, so 43% can be
    put side by side and the rest are honestly *read this, there is nothing to
    match it against*.  Filling the gap with a guessed pattern would turn a
    known silence into an empty query, and an empty query is what this design
    reads as evidence.
    """

    owner: str
    file: str = Field(default="", description="Where the function is.")
    blob_sha: str = Field(
        default="", description="Hash of the file the map read it from."
    )
    gloss_key: str = Field(
        default="",
        description="Key of the reading of this function, from the map.",
    )
    lines: list[int] = Field(
        default_factory=list,
        description="The arms' own lines -- what to mark in the text handed over.",
    )
    outcomes: list[str] = Field(
        default_factory=list, description="The values those arms write."
    )
    dispatch: list[DispatchFanout] = Field(default_factory=list)
    log_files: list[str] = Field(default_factory=list)
    log_pattern: str = Field(
        default="",
        description="What production would print for these arms.  Empty means silent.",
    )

    trace: list[TraceStep] = Field(
        default_factory=list,
        description=(
            "Why these arms ran with the value they did, walked from the "
            "source at use time.  Empty without a source tree, and empty on "
            "purpose when the tree is not the one the map was built from."
        ),
    )
    trace_note: str = Field(
        default="",
        description="Why the trace is empty or short, when it is.",
    )
    skeleton: list[SkeletonLine] = Field(
        default_factory=list,
        description=(
            "What this function prints, in source order, with the arms in "
            "place -- computed from the tree at use time.  Beside "
            "``log_pattern`` rather than instead of it: that one is the map's "
            "shared sentence and is what the questions already asked are "
            "built from, and swapping them would make every answer already "
            "collected read as unasked."
        ),
    )

    @property
    def silent(self) -> bool:
        """True when the map records no production line for any of these arms."""
        return not self.log_pattern


class Strategy(BaseModel):
    """What the map has to say about one symptom.

    Two phases in one object: everything above ``observations`` comes from the
    map alone and needs no network, and the verdicts are filled in afterwards
    from an evidence file.  Separating them is what lets the derivation be
    tested against a fixture, re-evaluated offline, and inspected before a
    single query is put to production -- the same split ``check-map`` makes, and
    for the same reason: first contact turns up the errors in the model.
    """

    symptom: Symptom
    map_id: str
    derived_from: str
    candidates: list[Candidate] = Field(default_factory=list)
    observations: list[Observation] = Field(default_factory=list)
    follow_up: Optional[FollowUp] = None
    localization: Optional[Localization] = Field(
        default=None,
        description=(
            "Filled in for a distribution symptom instead of ``candidates``.  "
            "Both live on one model because everything around them is shared -- "
            "the observations, the evidence file, the two-phase split -- and "
            "only what the map says in the middle differs."
        ),
    )
    leads: list[Lead] = Field(
        default_factory=list,
        description=(
            "Where the answer goes next, and where it stops.  An investigation "
            "is a path through the map rather than one lookup, and this is the "
            "one hop this derivation can license -- filtered to the surviving "
            "candidates once the evidence is in."
        ),
    )
    readings: list[Reading] = Field(
        default_factory=list,
        description=(
            "The code to read, one entry per function, filtered to the "
            "surviving candidates once the evidence is in.  The map chooses "
            "the input; what reads it is not this module's business."
        ),
    )
    findings: list[str] = Field(
        default_factory=list,
        description="Facts about the map itself that this question turned up.",
    )
    gaps: list[str] = Field(
        default_factory=list,
        description=(
            "What the map would have to record for this to go further.  A "
            "capability gap is an answer of its own -- it names the next thing "
            "to build instead of being absorbed as a weaker conclusion."
        ),
    )
#: Why a walk stopped.  Each is a fact about the investigation rather than an
#: error: a walk that ran out of map, one that came back to a question it had
#: already asked, and one that spent its budget are three different answers and
#: the reader has to be able to tell them apart.
WALK_TERMINAL = "every lead is a terminal -- the map has nothing further to open"
WALK_ASKED = "every question left had already been asked"
WALK_BUDGET = "the hop budget was spent"


class Hop(BaseModel):
    """One question asked of the map, and what opened it.

    The unit of a path.  Keeping the whole strategy rather than a summary is
    deliberate: a trace exists so the reasoning can be checked afterwards, and
    checking it means seeing which candidates were live at each step, not only
    which field the walk moved to.
    """

    number: int = Field(..., description="0 for the question that was asked, then 1, 2 …")
    symptom: Symptom
    opened: str = Field(
        default="",
        description="The lead's field.  Empty at hop 0, which nothing opened.",
    )
    source: str = Field(
        default=LEAD_MAP,
        description=(
            "Which supplier proposed the lead that opened this hop.  What makes "
            "the trace auditable: a path that went three hops on hypotheses is "
            "not the same claim as one that went three hops on recorded edges."
        ),
    )
    strategy: Strategy


class Investigation(BaseModel):
    """A path through the map, and why it ended.

    The product of an investigation is not the last hop but the path, the
    terminal it reached and which category that terminal is in.  A single
    :class:`Strategy` cannot hold that: it is one hop's worth by construction,
    because evidence has to be collected between hops and collecting it is a
    separate cadence.  So the history lives here and the loop lives in whoever
    owns this object -- the map enumerates, the evidence selects, and neither of
    those is a thing to be done from inside a derivation.

    Not a node.  It is the product of one investigation, not part of the
    vocabulary the map is written in, and storing it would make a question
    somebody asked once look like a fact about the code.
    """

    hops: list[Hop] = Field(default_factory=list)
    visited: list[str] = Field(
        default_factory=list,
        description=(
            "Questions already asked, as ``subject=value``.  ``status`` and "
            "``oldStatus`` copy from each other and ``jobStatus`` copies from "
            "itself, so a walk without this goes round for ever -- and one that "
            "dropped the repeat silently would report a loop as a dead end."
        ),
    )
    cycles: list[str] = Field(
        default_factory=list,
        description="Questions that came round again.  A property of the system, so it is reported.",
    )
    budget: int = Field(default=0, description="Hops allowed.")
    stopped: str = Field(default="", description="One of the WALK_* reasons.")
