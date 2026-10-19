"""Where a function's runs start, and the history rows written in the same pass as an arm.

bamboo-68's questions 1 and 2 (h2-value-time-design.md §3-§4): for each arm of
a job's or a task's junction, which calls write a history row in the same
pass, and which run -- a logger bound with a key -- that pass is in.  A value
the map sees set has no time of its own; a history row written in the same
pass gives it one, and the run's mark says which lines of the log are that
pass's.  Three records answer it, each read once, at build time:

- :attr:`FunctionPrintsNode.runs`: where a run starts -- a logger bound by one
  of PanDA's factories (:data:`BUILDERS`), its mark, and the column each hole
  of the mark holds where a declaration says (:class:`RunStart`).  Each print
  statement says which of them its logger can be (:attr:`PrintStatement.runs`).
- :attr:`ReachNode.history`: each call of a history writer, and the values
  the row it writes can record (:class:`HistoryCall`).  A history writer is a
  function whose SQL inserts into a history table (:data:`HISTORY_TABLES`),
  found by the map's reading of its statements, not by its name.
- :attr:`Branch.history`: the pairs (:class:`HistoryPair`).

**One pass.**  A call H and a point S of one function are in one pass when
one reaches the other -- ``reaching(H, [S])`` or ``reaching(S, [H])`` -- with
every logger the function binds by a factory and the header of every loop
holding both as killers: a new run, or the next turn of a loop over jobs, is
another row's pass.  H is a call whose closure writes a history row, followed
through plain calls only.  A function handed on (``Thread(target=...)``,
``submit``), a lambda's body and a thread's ``run`` are not followed, since
what they write is not written in this pass; a function handed to one that
calls it at once is left out with them.

**Up from the arm.**  Level 0 is the arm's own function, the point its
statement; level n+1 is each caller of a level-n function, the point each line
it calls it from.  Callers are read off the reach record, plain calls only,
classes ignored.  A way up stops at the first level with a pair, at a function
nothing calls (the top of its pass: an entry, or a thread's ``run``), at depth
:data:`MAX_DEPTH`, or once :data:`MAX_FUNCTIONS` functions are seen for the
arm; the last two are said (:attr:`Branch.history_hole`).

**What the calls on a way hand.**  A way up is the arm's pass only where the
values its calls hand do not rule out a statement on it
(:func:`pathcond.ruled_out_by`: an ``if`` holding the statement, or an early
exit before it, that the values decide against it).  Each function on the
way reads the constants its route hands the parameters its tests read -- the
call into it, then outward through each forwarder (:meth:`Calls.route_constant`)
-- so ``setErrDiag(diag)`` leaves ``append`` False and ``setErrDiag``'s append
arm is not on that caller's pass, and the adder's ``peekJobs(...,
forAnal=True)`` keeps ``peekJob`` from its ``job.jobStatus = "unknown"``.  A
way ruled out neither pairs nor goes on up; a branch with no pair and such a
way says so.  The pair's caller's own parameters are read as any value: what
is above the pair is not walked.

**The arm's row.**  Whether a history row is the arm's own
(:attr:`HistoryPair.row`) is read in place where both are in one function and
the arm writes an attribute: the writer handed an attribute of the object the
arm writes, the object not bound again between the two.  Elsewhere the arm's
object is followed (:func:`bamboo.codemap.objects.follow`, on the records
:mod:`bamboo.codemap.panda.objects` keeps) from the pair's caller into the
pair's call, with its key (:data:`HISTORY_KEYS`), and ``can`` says it reaches
the key the writer is handed:

- the object is the path the arm assigns on, the receiver of a method of its
  class it runs (``taskSpec.setOnHold()``), or for an SQL arm the key of the
  row the statement it fills writes;
- at a caller of the arm's function, the object is the path of the caller
  whose follow through the call on the way holds the arm's object there --
  a path the call is handed, a path below one (``self.job`` for
  ``self.execute_plugin()``), an element of either, or a name bound to what
  the call returns;
- ``archiveJob``'s ``updatedJobList.append(job)`` and ``for tmpJob in
  updatedJobList`` make the row of ``tmpJob.PandaID`` one the arm's can be:
  an element among the others, read without the statements' order.

So ``can`` is a row the pass can write for the arm's object, not the one it
wrote; ``row_via`` says one way the object goes there.  An object no path the
caller hands reaches, or that does not reach the writer, says so.

**Not read.**  What the calls on the way down to the writer hand: the
object is followed without the values calls hand
(:data:`bamboo.codemap.objects.PREMISES` ``handed_values_not_read``), so a
branch a constant rules out there is read as one that runs.  A key the object
is given again on the way: ``retryJob``'s ``job.PandaID = int(val)`` after its
``INSERT ... RETURNING``, ``ppEventServiceJob``'s clone given a new
``PandaID`` -- the row then is the new key's, the arm's values in it, and
``can`` holds for that row, while the run's mark carries the old key.  A key
the run holds by another name than the object's is tied to it only where a
binding says so, and one way -- what a function is handed is not handed back
through its parameters: the adder's ``self.job_id`` reaches
``self.job.PandaID`` through ``peekJob``'s ``job.PandaID = pandaID``, a
branch the adder's own ``forAnal=True`` rules out, read because follow does
not read what calls hand; a row a ``SELECT`` reads by its key is not read as
holding it.  The values of one container are one path (``taskParamMap[]``),
so a path into a dict can stand for the key it holds among the others.
"""

from __future__ import annotations

import ast
import string
from collections import defaultdict, deque
from typing import Iterator, NamedTuple, Optional, Union

from bamboo.codemap import calls as C
from bamboo.codemap import objects, reach
from bamboo.codemap.models import (
    BIND_RESULT,
    HISTORY_AFTER,
    HISTORY_BEFORE,
    HISTORY_EITHER,
    HISTORY_ROW_CAN,
    HISTORY_ROW_SAME,
    RUN_KEY_TIME,
    Branch,
    HistoryCall,
    HistoryPair,
    JunctionNode,
    MapFragment,
    ReachCall,
    RunKey,
    RunStart,
    SourceModule,
)
from bamboo.codemap.panda import mustprints, pathcond, sql, sqltext
from bamboo.codemap.panda.reaching import reaching

#: The tables PanDA keeps a subject's past values in, lower-cased, each with
#: the subject it records: one row per change, stamped when it is written.
HISTORY_TABLES: dict[str, str] = {
    "jobs_statuslog": "JobSpec.jobStatus",
    "tasks_statuslog": "JediTaskSpec.status",
}
#: The attribute a history row is keyed by, per class: the row of the object's own key.
HISTORY_KEYS: dict[str, str] = {"JobSpec": "PandaID", "JediTaskSpec": "jediTaskID"}


class Builder(NamedTuple):
    """Where a logger factory takes its mark from, and what it marks lines with when given none."""

    #: The mark's position among the call's arguments.
    position: int
    keyword: str
    #: For no argument: :data:`RUN_KEY_TIME` for a time stamp, ``""`` for no mark.
    absent: str
    #: For ``None`` given.
    none: str


#: PanDA's logger factories: a logger bound to one of these starts a run.
BUILDERS: dict[str, Builder] = {
    # pandacommon LogWrapper(log, prefix="", ...): a prefix of None is the time.
    "LogWrapper": Builder(1, "prefix", "", RUN_KEY_TIME),
    # pandajedi MsgWrapper(logger, token=None, ...): "<time>" for none.
    "MsgWrapper": Builder(1, "token", RUN_KEY_TIME, RUN_KEY_TIME),
    # DBProxy.create_tagged_logger(comment, tag=None): "method < tag >", the time for none.
    "create_tagged_logger": Builder(1, "tag", RUN_KEY_TIME, RUN_KEY_TIME),
    # pandacommon make_logger(tmp_log, token=None, ...): "method <token>", "method :" for none.
    "make_logger": Builder(1, "token", "", ""),
}

MAX_DEPTH = 6
MAX_FUNCTIONS = 400

HISTORY_NOT_RECORDED = "not read: no history table records {cls}"
HISTORY_NO_FUNCTION = "the arm's function is not a row of the reach record"
HISTORY_DEPTH = f"a way up reached depth {MAX_DEPTH} with no pair"
HISTORY_WIDE = f"the walk up met {MAX_FUNCTIONS} functions and went no further"
HISTORY_RULED_OUT = "a way up from the arm is ruled out by what the calls on it hand"

ROW_ANOTHER = "another: "
ROW_NO_START = (
    "the record holds no path of the object the arm writes,"
    " nor a key of the row its statement writes"
)
ROW_NOT_HANDED = (
    "no path the caller hands into the call on the way to the arm is followed"
    " to the object the arm writes"
)
ROW_NOT_TO_WRITER = (
    "the record follows the object to no key of a history write through this call"
)
ROW_HOLE = "the object reaches a call the record could not follow on the way: {why}"
ROW_NO_KEY = "the writer is handed no key this reading finds"
ROW_NOT_A_NAME = "the object or the key is not a name or attributes on one"
ROW_REBOUND = "the object can be bound again between the arm and the call"

RUN_NO_PRINTS = "the function prints nothing, so its runs are not recorded"
RUN_NONE = (
    "no logger a factory builds here holds the point, and the class binds none to self"
)
RUN_OBJECT_REBOUND = (
    "the logger {logger} its class binds in __init__ is bound again in {where}"
)

KEY_PARAMETER = "a parameter: what the caller passes is not followed"
KEY_LOCAL = "a local name: what it is bound to is not followed"
KEY_NOT_ATTRIBUTE = "not an attribute of a name"
KEY_UNDECLARED = "the class of {name} is not declared"
KEY_SEVERAL = "{name} is declared as several classes: {classes}"
KEY_FAILED = "reading the declaration of {name} failed: {error}"

VALUES_NO_STATEMENT = "no statement of the function holds the call's line"

_LOOPS = (ast.For, ast.AsyncFor, ast.While)
_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
#: Calls that hand a value on unchanged as far as a key is concerned: ``str(pandaID)``.
_CONVERSIONS = frozenset({"str", "int", "repr"})

_Piece = Union[str, ast.expr]


def _holder(held, path: str) -> Optional[str]:
    """The held path *path* is, or is below -- ``job`` for ``job.PandaID`` -- the nearest one; ``None`` where none."""
    found = [one for one in held if objects._under(one, path) is not None]
    return max(found, key=lambda one: (len(one), one)) if found else None


class _Arm(NamedTuple):
    """What an arm writes, as paths of its function: the object, and the key of its row."""

    owner: str
    objects: frozenset[str]
    keys: frozenset[str]


class _Step(NamedTuple):
    """A function the walk up from an arm has reached, and the way it came by."""

    owner: str
    func: ast.AST
    #: the statement of the function on the way: the arm's, or the call into the function below
    point: ast.stmt
    level: int
    #: each function from the arm's out to this one, with its statement on the way
    way: tuple[tuple[str, ast.stmt], ...]
    #: the call into each function of *way* but this one, from the next one out; None where none is placed
    into: tuple[Optional[C.CallSite], ...]


class _Bound(NamedTuple):
    """A statement binding a name or ``self.<attr>`` to a logger a factory builds."""

    statement: ast.stmt
    target: ast.expr
    call: ast.Call
    builder: str


def attach(
    fragment: MapFragment, modules: list[SourceModule], calls: C.Calls
) -> tuple[int, int, int]:
    """Read the runs, the history calls and the pairs into *fragment*'s nodes and branches.

    Returns how many functions have a run, how many history calls there are,
    and how many branches have a pair.
    """
    reader = _Reader(fragment, modules, calls)
    return reader.attach_runs(), reader.attach_history_calls(), reader.attach_pairs()


def history_writers(functions: list[tuple[str, ast.AST, str]]) -> dict[str, str]:
    """``owner -> subject`` for every function whose SQL inserts a row into a history table."""
    found: dict[str, str] = {}
    for _file, func, owner in functions:
        for run in sql.executions(func):  # type: ignore[arg-type]
            for table in sqltext.creates(run.reading):
                subject = HISTORY_TABLES.get(table.rpartition(".")[2].lower())
                if subject is not None:
                    found[owner] = subject
    return found


class _Reader:
    def __init__(
        self, fragment: MapFragment, modules: list[SourceModule], calls: C.Calls
    ):
        self.fragment = fragment
        self.calls = calls
        self.types = calls.types
        self.trees = {module.rel_path: module.tree for module in modules}
        self.names = reach._Names(calls.types)
        self.funcs: dict[str, tuple[str, ast.AST]] = {
            owner: (file, func) for file, func, owner in self.names.functions
        }
        self.rows = {row.owner: row for row in fragment.reach_graph}
        self.prints = {node.owner: node for node in fragment.function_prints}
        self.writers = history_writers(self.names.functions)
        self._built: dict[int, list[_Bound]] = {}
        self._follows: dict[tuple[str, str], objects.ObjectReach] = {}
        self._objects: dict[int, tuple[list[RunStart], str]] = {}
        self._runs: dict[str, list[RunStart]] = {}
        self._kills: dict[int, list[ast.stmt]] = {}
        self._targets: dict[str, C.Target] = {}
        self._callers: dict[str, list[C.CallSite]] = {}
        # Plain calls only, as one pass follows them.
        self.callers: dict[str, dict[str, list[int]]] = defaultdict(dict)
        for row in fragment.reach_graph:
            for call in row.calls:
                if call.how == "call" and not call.thread:
                    lines = self.callers[call.to].setdefault(row.owner, [])
                    for line in call.lines or [call.line]:
                        if line not in lines:
                            lines.append(line)
        self.below: dict[str, set[str]] = {}
        self.subject_of: dict[str, str] = {}

    # -- runs ----------------------------------------------------------------- #

    def attach_runs(self) -> int:
        """Every printing function's runs, and each print statement's."""
        found = 0
        for node in self.fragment.function_prints:
            node.runs = []
            for statement in node.statements:
                statement.runs = []
            held = self.funcs.get(node.owner)
            if held is None:
                continue
            file, func = held
            node.runs = self.runs_of(node.owner)
            found += bool(node.runs)
            for statement in node.statements:
                site = mustprints._statement_at(func, statement.line)
                if site is not None:
                    statement.runs = self._printed_through(
                        file, func, site, statement.level, node.runs
                    )
        return found

    def runs_of(self, owner: str) -> list[RunStart]:
        """The runs *owner*'s lines can be in: its own, then its object's."""
        if owner not in self._runs:
            file, func = self.funcs[owner]
            own = [
                self._run_start(bound, owner, file, func, False)
                for bound in self.built(func)
                if isinstance(bound.target, ast.Name)
            ]
            self._runs[owner] = own + self._object_runs(file, func)[0]
        return self._runs[owner]

    def built(self, func: ast.AST) -> list[_Bound]:
        """Every statement of *func* itself binding one target to a factory's logger, in source order."""
        key = id(func)
        if key not in self._built:
            found = []
            for node in _own_nodes(func):
                if isinstance(node, ast.Assign) and len(node.targets) == 1:
                    target, value = node.targets[0], node.value
                elif isinstance(node, ast.AnnAssign) and node.value is not None:
                    target, value = node.target, node.value
                else:
                    continue
                if (
                    isinstance(value, ast.Call)
                    and _called(value) in BUILDERS
                    and (isinstance(target, ast.Name) or _on_self(target))
                ):
                    found.append(_Bound(node, target, value, _called(value)))
            self._built[key] = sorted(found, key=lambda one: one.statement.lineno)
        return self._built[key]

    def _object_runs(self, file: str, func: ast.AST) -> tuple[list[RunStart], str]:
        """The loggers *func*'s class binds to ``self`` in its ``__init__``, and why one is left out."""
        owned = self.types.method_class(file, func)
        if owned is None:
            return [], ""
        key = id(owned[1])
        if key not in self._objects:
            runs: list[RunStart] = []
            why = ""
            found = self.types.method(owned[0], owned[1], "__init__")
            if found is not None:
                where, init = found
                for bound in self.built(init):
                    if not _on_self(bound.target):
                        continue
                    attribute = bound.target.attr  # type: ignore[attr-defined]
                    again = self._bound_again(owned, attribute, bound.statement)
                    if again:
                        why = RUN_OBJECT_REBOUND.format(
                            logger=f"self.{attribute}", where=again
                        )
                        continue
                    runs.append(
                        self._run_start(
                            bound, self.names.of(init) or "", where, init, True
                        )
                    )
            self._objects[key] = (runs, why)
        return self._objects[key]

    def _bound_again(
        self, owned: tuple[str, ast.ClassDef], attribute: str, kept: ast.stmt
    ) -> str:
        """Where else ``self.<attribute>`` is stored, in any class ``self`` can be or a base of one; ``""`` for nowhere."""
        seen: set[int] = set()
        for one in self.calls.objects_of(owned):
            for where, holder in self.types.lineage(one[0], one[1]):
                if id(holder) in seen:
                    continue
                seen.add(id(holder))
                for node in ast.walk(holder):
                    if (
                        isinstance(node, ast.Attribute)
                        and isinstance(node.ctx, ast.Store)
                        and node.attr == attribute
                        and isinstance(node.value, ast.Name)
                        and node.value.id == "self"
                        and mustprints._statement_of(node) is not kept
                    ):
                        method = _enclosing_function(node)
                        return f"{where}::{holder.name}.{getattr(method, 'name', '?')}"
        return ""

    def _run_start(
        self, bound: _Bound, at: str, file: str, func: ast.AST, is_object: bool
    ) -> RunStart:
        spec = BUILDERS[bound.builder]
        given, expression = _given(bound.call, spec)
        common = {
            "line": bound.statement.lineno,
            "at": at,
            "logger": ast.unparse(bound.target),
            "builder": bound.builder,
            "object": is_object,
        }
        if not given or (
            isinstance(expression, ast.Constant) and expression.value is None
        ):
            stamp = spec.none if given else spec.absent
            key = [RunKey(expression="", column=RUN_KEY_TIME)] if stamp else []
            return RunStart(**common, mark="", key=key)
        pieces = _pieces(expression, func)  # type: ignore[arg-type]
        mark = "".join(piece if isinstance(piece, str) else "{}" for piece in pieces)
        key = [
            self._key(piece, file, func)
            for piece in pieces
            if not isinstance(piece, str)
        ]
        return RunStart(**common, mark=mark, key=key)

    def _key(self, expression: ast.expr, file: str, func: ast.AST) -> RunKey:
        """The column one hole of a mark holds: ``JediTaskSpec.jediTaskID`` for ``taskSpec.jediTaskID`` declared so."""
        spelled = ast.unparse(expression)
        inner = expression
        while (
            isinstance(inner, ast.Call)
            and isinstance(inner.func, ast.Name)
            and inner.func.id in _CONVERSIONS
            and len(inner.args) == 1
            and not inner.keywords
        ):
            inner = inner.args[0]
        if isinstance(inner, ast.Name):
            why = KEY_PARAMETER if _is_parameter(func, inner.id) else KEY_LOCAL
            return RunKey(expression=spelled, why=why)
        if not isinstance(inner, ast.Attribute):
            return RunKey(expression=spelled, why=KEY_NOT_ATTRIBUTE)
        owner = inner.value
        name = ast.unparse(owner)
        try:
            if isinstance(owner, ast.Name) and owner.id == "self":
                held = self.types.method_class(file, func)
                declared = frozenset({held}) if held is not None else None
            else:
                declared = self.types.declared(file, func, owner)
        except Exception as error:  # noqa: BLE001 -- a key unread, not the build's end
            return RunKey(
                expression=spelled,
                why=KEY_FAILED.format(name=name, error=type(error).__name__),
            )
        if not declared:
            return RunKey(expression=spelled, why=KEY_UNDECLARED.format(name=name))
        classes = sorted({cls.name for _where, cls in declared})
        if len(classes) > 1:
            return RunKey(
                expression=spelled,
                why=KEY_SEVERAL.format(name=name, classes=", ".join(classes)),
            )
        return RunKey(expression=spelled, column=f"{classes[0]}.{inner.attr}")

    def _printed_through(
        self, file: str, func: ast.AST, site: ast.stmt, level: str, runs: list[RunStart]
    ) -> list[int]:
        """The lines of *runs* whose logger the logging call at *site* can be made on."""
        call = next(
            (
                node
                for node in ast.walk(site)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == level
            ),
            None,
        )
        if call is None:
            return []
        receiver = call.func.value  # type: ignore[attr-defined]
        if _on_self(receiver):
            spelled = ast.unparse(receiver)
            return [run.line for run in runs if run.object and run.logger == spelled]
        if not isinstance(receiver, ast.Name):
            return []
        return self._holding(func, site, receiver.id)

    def _holding(
        self, func: ast.AST, site: ast.AST, name: Optional[str] = None
    ) -> list[int]:
        """The lines of *func*'s own runs (of the logger *name*, else of every one) whose binding reaches *site*."""
        by_name: dict[str, list[ast.stmt]] = defaultdict(list)
        for bound in self.built(func):
            if isinstance(bound.target, ast.Name) and (
                name is None or bound.target.id == name
            ):
                by_name[bound.target.id].append(bound.statement)
        lines: list[int] = []
        for held, sites in by_name.items():
            killers = [
                one
                for binding in pathcond.bindings(func, held)
                if not binding.nested
                and (one := _site_statement(binding.site)) is not None
                and not any(one is site_ for site_ in sites)
            ]
            lines.extend(found.lineno for found in reaching(site, sites, killers))  # type: ignore[attr-defined]
        return sorted(set(lines))

    # -- history calls -------------------------------------------------------- #

    def attach_history_calls(self) -> int:
        """Each call of a history writer, on the caller's reach row, with the values its row can record."""
        arms: dict[tuple[str, str], list[tuple[ast.stmt, str]]] = defaultdict(list)
        for junction in self.fragment.junctions:
            if junction.subject not in HISTORY_TABLES.values():
                continue
            for branch in junction.branches:
                sites = mustprints._sites(junction, branch, self.trees)
                if isinstance(sites, str):
                    continue
                for site in sites:
                    owner = self.names.of(site.func)
                    if owner is not None:
                        arms[(owner, junction.subject)].append(
                            (site.statement, branch.outcome)
                        )
        count = 0
        for row in self.fragment.reach_graph:
            row.history = []
            lines: dict[int, str] = {}
            for call in row.calls:
                if call.how == "call" and not call.thread and call.to in self.writers:
                    for line in call.lines or [call.line]:
                        lines.setdefault(line, self.writers[call.to])
            if not lines or row.owner not in self.funcs:
                continue
            _file, func = self.funcs[row.owner]
            for line, subject in sorted(lines.items()):
                row.history.append(
                    self._values(
                        func, line, subject, arms.get((row.owner, subject), [])
                    )
                )
                self.subject_of[f"{row.owner}:{line}"] = subject
            count += len(row.history)
        # What each function's closure can write, through plain calls: its own
        # history calls and every one below.
        below: dict[str, set[str]] = defaultdict(set)
        pending: deque[str] = deque()
        for row in self.fragment.reach_graph:
            if row.history:
                below[row.owner] = {f"{row.owner}:{held.line}" for held in row.history}
                pending.append(row.owner)
        while pending:
            owner = pending.popleft()
            for caller in self.callers.get(owner, {}):
                if not below[owner] <= below[caller]:
                    below[caller] |= below[owner]
                    pending.append(caller)
        self.below = dict(below)
        return count

    def _values(
        self, func: ast.AST, line: int, subject: str, arms: list[tuple[ast.stmt, str]]
    ) -> HistoryCall:
        site = mustprints._statement_at(func, line)
        if site is None:
            return HistoryCall(line=line, subject=subject, hole=VALUES_NO_STATEMENT)
        statements: list[ast.stmt] = []
        for statement, _outcome in arms:
            if not any(statement is held for held in statements):
                statements.append(statement)
        entry = func.args  # type: ignore[attr-defined]
        reached = reaching(site, [*statements, entry])
        values = sorted(
            {
                outcome
                for statement, outcome in arms
                if any(statement is one for one in reached)
            }
        )
        return HistoryCall(
            line=line,
            subject=subject,
            values=values,
            passed=any(one is entry for one in reached),
        )

    # -- pairs ---------------------------------------------------------------- #

    def attach_pairs(self) -> int:
        """Every branch's history pairs, or why it has none."""
        recorded = sorted(
            {subject.rpartition(".")[0] for subject in HISTORY_TABLES.values()}
        )
        found = 0
        for junction in self.fragment.junctions:
            cls = junction.subject.rpartition(".")[0]
            for branch in junction.branches:
                branch.history = None
                branch.history_hole = ""
                if cls not in recorded:
                    branch.history_hole = HISTORY_NOT_RECORDED.format(cls=cls)
                    continue
                self._pair(junction, branch)
                found += bool(branch.history)
        return found

    def _pair(self, junction: JunctionNode, branch: Branch) -> None:
        sites = mustprints._sites(junction, branch, self.trees)
        if isinstance(sites, str):
            branch.history_hole = sites
            return
        pairs: dict[tuple[int, str, int, int], HistoryPair] = {}
        holes: list[str] = []
        ruled = False
        for site in sites:
            owner = self.names.of(site.func)
            if owner is None or owner not in self.rows:
                branch.history_hole = HISTORY_NO_FUNCTION
                return
            written = mustprints._written_object(site.statement, junction.subject)
            arm = self._arm(owner, site.func, site.statement, junction.subject)
            seen = {owner}
            # Each entry carries its way: the functions from the arm's out to this one, each with its statement on
            # the way, and the call into each but this one (None where the call record places none).
            frontier: deque[_Step] = deque(
                [
                    _Step(
                        owner,
                        site.func,
                        site.statement,
                        0,
                        ((owner, site.statement),),
                        (),
                    )
                ]
            )
            while frontier:
                here, func, point, level, way, into = frontier.popleft()
                found = self._pairs_at(
                    here, func, point, level, junction.subject, written, arm
                )
                if found:
                    for pair in found:
                        pairs.setdefault(
                            (pair.level, pair.caller, pair.point, pair.call), pair
                        )
                    continue
                if level >= MAX_DEPTH:
                    if HISTORY_DEPTH not in holes:
                        holes.append(HISTORY_DEPTH)
                    continue
                for caller, lines in self.callers.get(here, {}).items():
                    if caller in seen:
                        continue
                    if len(seen) >= MAX_FUNCTIONS:
                        if HISTORY_WIDE not in holes:
                            holes.append(HISTORY_WIDE)
                        continue
                    held = self.funcs.get(caller)
                    if held is None:
                        seen.add(caller)
                        continue
                    steps: list[tuple[ast.stmt, Optional[C.CallSite]]] = []
                    out = False
                    for line in lines:
                        up = mustprints._statement_at(held[1], line)
                        if up is None:
                            continue
                        placed = self._sites_into(here, caller, line)
                        kept = next(
                            (
                                one
                                for one in placed
                                if not self._ruled_out(way, (*into, one))
                            ),
                            None,
                        )
                        if placed and kept is None:
                            out = True
                            continue
                        steps.append((up, kept))
                    ruled = ruled or out
                    if out and not steps:
                        # A way the calls rule out is not this arm's pass; another way may still reach the caller.
                        continue
                    seen.add(caller)
                    for up, kept in steps:
                        frontier.append(
                            _Step(
                                caller,
                                held[1],
                                up,
                                level + 1,
                                (*way, (caller, up)),
                                (*into, kept),
                            )
                        )
        branch.history = sorted(
            pairs.values(), key=lambda one: (one.level, one.caller, one.point, one.call)
        )
        if ruled and not pairs:
            holes.append(HISTORY_RULED_OUT)
        branch.history_hole = "; ".join(holes)

    def _pairs_at(
        self,
        owner: str,
        func: ast.AST,
        point: ast.stmt,
        level: int,
        subject: str,
        written: Optional[ast.expr],
        arm: "_Arm",
    ) -> list[HistoryPair]:
        """The calls of *owner* that lead to a history row in one pass with *point*."""
        row = self.rows.get(owner)
        if row is None:
            return []
        candidates: dict[int, list[ReachCall]] = {}
        for call in row.calls:
            if call.how != "call" or call.thread:
                continue
            if call.to in self.writers or self.below.get(call.to):
                for line in call.lines or [call.line]:
                    candidates.setdefault(line, []).append(call)
        if not candidates:
            return []
        if id(func) not in self._kills:
            self._kills[id(func)] = [bound.statement for bound in self.built(func)]
        kills = self._kills[id(func)]
        around = _loops(point, func)
        pairs: list[HistoryPair] = []
        for line, through in sorted(candidates.items()):
            site = mustprints._statement_at(func, line)
            if site is None or site is point:
                continue
            both = [
                loop
                for loop in _loops(site, func)
                if any(loop is one for one in around)
            ]
            killers = [*kills, *both]
            after = bool(reaching(site, [point], killers))
            before = bool(reaching(point, [site], killers))
            if not (after or before):
                continue
            writes: set[str] = set()
            for call in through:
                writes |= (
                    {f"{owner}:{line}"}
                    if call.to in self.writers
                    else self.below.get(call.to, set())
                )
            row_said, row_why, held, via = self._row(
                level,
                owner,
                func,
                point,
                site,
                line,
                through,
                subject,
                written,
                arm,
                writes,
            )
            runs, run_hole = self._runs_at(owner, func, point)
            pairs.append(
                HistoryPair(
                    level=level,
                    caller=owner,
                    point=point.lineno,
                    call=line,
                    callee=through[0].callee,
                    order=HISTORY_EITHER
                    if after and before
                    else HISTORY_AFTER
                    if after
                    else HISTORY_BEFORE,
                    history=sorted(
                        {
                            self.subject_of[write]
                            for write in writes
                            if write in self.subject_of
                        }
                    ),
                    writes=sorted(writes),
                    row=row_said,
                    row_why=row_why,
                    object=held,
                    row_via=via,
                    runs=runs,
                    run_hole=run_hole,
                )
            )
        return pairs

    def _row(
        self,
        level: int,
        owner: str,
        func: ast.AST,
        point: ast.stmt,
        site: ast.stmt,
        line: int,
        through: list[ReachCall],
        subject: str,
        written: Optional[ast.expr],
        arm: "_Arm",
        writes: set[str],
    ) -> tuple[str, str, str, list[str]]:
        """Whether the history row is the arm's row: ``(row, why, object, row_via)``."""
        direct = [call for call in through if call.to in self.writers]
        if level == 0 and direct and written is not None:
            row, why = self._in_place(func, point, site, line, direct, subject, written)
            if row == HISTORY_ROW_SAME:
                return row, why, ast.unparse(written), []
            if not row.startswith(ROW_ANOTHER):
                return row, why, "", []
            found = self._followed(level, owner, point, line, subject, arm, writes)
            return found if found[0] == HISTORY_ROW_CAN else (row, why, "", [])
        return self._followed(level, owner, point, line, subject, arm, writes)

    def _in_place(
        self,
        func: ast.AST,
        point: ast.stmt,
        site: ast.stmt,
        line: int,
        direct: list[ReachCall],
        subject: str,
        written: ast.expr,
    ) -> tuple[str, str]:
        """The row read where the arm and the writer's call are in one function: ``(row, why)``."""
        named = self.rows[direct[0].to].function if direct[0].to in self.rows else ""
        call = next(
            (
                node
                for node in ast.walk(site)
                if isinstance(node, ast.Call)
                and node.lineno == line
                and _called(node) == named
            ),
            None,
        )
        first = call.args[0] if call is not None and call.args else None
        if first is None or isinstance(first, ast.Starred):
            return "", ROW_NO_KEY
        root, base = _root(written), _root(first)
        if not (isinstance(root, ast.Name) and isinstance(base, ast.Name)):
            return "", ROW_NOT_A_NAME
        recorded = self.writers[direct[0].to].rpartition(".")[0]
        if root.id != base.id or recorded != subject.rpartition(".")[0]:
            return f"{ROW_ANOTHER}{ast.unparse(first)}", ""
        sites: list[ast.AST] = [func.args]  # type: ignore[attr-defined]
        for binding in pathcond.bindings(func, root.id):
            if (
                not binding.nested
                and (one := _site_statement(binding.site)) is not None
            ):
                sites.append(one)
        if reaching(point, sites) == reaching(site, sites):
            return HISTORY_ROW_SAME, ""
        return "", ROW_REBOUND

    # -- the arm's object followed into the pair's call ----------------------------------------------------------

    def _arm(
        self, owner: str, func: ast.AST, statement: ast.stmt, subject: str
    ) -> "_Arm":
        """What the arm at *statement* writes, as paths of its function: the object, else the row's key."""
        row = self.rows.get(owner)
        cls, _sep, attr = subject.rpartition(".")
        key = HISTORY_KEYS.get(cls, "")
        if row is None or not key:
            return _Arm(owner, frozenset(), frozenset())
        lines = range(statement.lineno, (statement.end_lineno or statement.lineno) + 1)
        held = {
            one.path for one in row.assigns if one.line in lines and attr in one.attrs
        }
        if not held:
            # A method of the object's class run on it: ``taskSpec.setOnHold()``.
            held = {
                call.path
                for call in row.call_bindings
                if call.line in lines
                and call.parameter == "self"
                and (callee := self.rows.get(call.to)) is not None
                and any(one.rpartition("::")[2] == cls for one in callee.classes)
            }
        if held:
            return _Arm(
                owner, frozenset(held), frozenset(f"{one}.{key}" for one in held)
            )
        # An SQL arm: the key of the row each statement it can fill writes.
        keys: set[str] = set()
        for write in row.writes:
            if not any(value.column.lower() == attr.lower() for value in write.columns):
                continue
            ran = [mustprints._statement_at(func, one) for one in write.lines]
            if not any(
                one is not None and (one is statement or reaching(one, [statement]))
                for one in ran
            ):
                continue
            named = [*write.keys, *(write.columns if write.kind == "insert" else [])]
            keys |= {
                value.path
                for value in named
                if value.column.lower() == key.lower() and value.path
            }
        return _Arm(owner, frozenset(), frozenset(keys))

    # -- what the calls on a way up hand ------------------------------------------------------------------------

    def _target(self, owner: str) -> C.Target:
        if owner not in self._targets:
            file, func = self.funcs[owner]
            owned = self.types.method_class(file, func)
            self._targets[owner] = C.Target(file, func, owned[1] if owned else None)
        return self._targets[owner]

    def _sites_into(self, owner: str, caller: str, line: int) -> list[C.CallSite]:
        """The calls of *caller* on *line* that the call record places at *owner*."""
        if owner not in self._callers:
            self._callers[owner] = self.calls.callers(self._target(owner))
        func = self.funcs[caller][1]
        return [
            site
            for site in self._callers[owner]
            if site.func is func
            and site.call.lineno <= line <= (site.call.end_lineno or site.call.lineno)
        ]

    def _ruled_out(
        self,
        way: tuple[tuple[str, ast.stmt], ...],
        into: tuple[Optional[C.CallSite], ...],
    ) -> bool:
        """Whether what the calls *into* hand rules out a function's statement on *way* (:func:`pathcond.ruled_out_by`).

        Each function's parameters its tests read take the constant its route
        hands -- the call into it, then outward through each forwarder, as far
        as the calls are placed (:meth:`Calls.route_constant`).  A parameter
        the function binds again is not ruled on: what it was handed may not
        be what the test reads.
        """
        for index, (owner, statement) in enumerate(way[: len(into)]):
            route: list[C.CallSite] = []
            for one in into[index:]:
                if one is None:
                    break
                route.append(one)
            if not route:
                continue
            func = self.funcs[owner][1]
            tested = sorted(
                {
                    name.id
                    for test in pathcond.tests_on(statement, func)
                    for name in ast.walk(test)
                    if isinstance(name, ast.Name)
                    and _is_parameter(func, name.id)
                    and not C._rebinds(func, name.id)
                }
            )
            if not tested:
                continue
            target = self._target(owner)
            constants = {
                name: value
                for name in tested
                if (value := self.calls.route_constant(name, target, tuple(route)))
                is not C.UNSET
            }
            if constants and pathcond.ruled_out_by(statement, func, constants):
                return True
        return False

    def _follow(self, owner: str, path: str) -> objects.ObjectReach:
        if (owner, path) not in self._follows:
            self._follows[(owner, path)] = objects.follow(self.rows, owner, path)
        return self._follows[(owner, path)]

    def _key_parameter(self, writer: str) -> str:
        """The parameter a history writer takes its row's key as: its first after ``self``."""
        held = self.funcs.get(writer)
        if held is None:
            return ""
        arguments = held[1].args  # type: ignore[attr-defined]
        names = [one.arg for one in [*arguments.posonlyargs, *arguments.args]]
        row = self.rows.get(writer)
        if row is not None and row.cls:
            names = names[1:]
        return names[0] if names else ""

    def _handed(
        self, owner: str, point: ast.stmt, arm: "_Arm"
    ) -> tuple[dict[str, int], dict[str, int]]:
        """The caller's paths whose follow through the call at *point* holds the arm's object, and its key.

        Each with 0 where it is a path the call is handed or a name bound to
        what it returns, 1 where it is below one or an element of one.
        """
        here = self.rows[owner]
        lines = range(point.lineno, (point.end_lineno or point.lineno) + 1)
        at = [
            call
            for call in here.call_bindings
            if call.line in lines
            and call.to
            and call.parameter
            and not call.parameter.startswith("*")
        ]
        handed = {call.path for call in at}
        recorded = (
            {one.path for one in here.assigns}
            | {one.path for one in here.bindings}
            | {one.name for one in here.bindings}
            | {one.path for one in here.call_bindings}
        )
        below = {
            one
            for one in recorded
            if any(objects._under(path, one) is not None for path in handed)
        }
        held_objects: dict[str, int] = {}
        held_keys: dict[str, int] = {}

        def found(reach: objects.ObjectReach, start: str, rank: int) -> None:
            for path in reach.held.get(arm.owner, ()):
                for target, into in (
                    (arm.objects, held_objects),
                    (arm.keys, held_keys),
                ):
                    for one in target:
                        rest = objects._under(path, one)
                        if rest is not None:
                            into[start + rest] = min(into.get(start + rest, rank), rank)

        for start in sorted(handed | below | {one + "[]" for one in handed | below}):
            if objects._parts(start) > objects.MAX_PARTS:
                continue
            held, _spent = objects._held(here, {start})
            for call in at:
                for one in held:
                    rest = objects._under(call.path, one)
                    if rest is not None:
                        found(
                            self._follow(call.to, call.parameter + call.at + rest),
                            start,
                            0 if start in handed else 1,
                        )
        for binding in here.bindings:
            if binding.how == BIND_RESULT and binding.line in lines:
                found(self._follow(owner, binding.name), binding.name, 0)
        return held_objects, held_keys

    def _followed(
        self,
        level: int,
        owner: str,
        point: ast.stmt,
        line: int,
        subject: str,
        arm: "_Arm",
        writes: set[str],
    ) -> tuple[str, str, str, list[str]]:
        """The arm's object followed from the caller into the call at *line*: ``(row, why, object, row_via)``."""
        cls = subject.rpartition(".")[0]
        key = HISTORY_KEYS.get(cls, "")
        here = self.rows.get(owner)
        if here is None or not (arm.objects or arm.keys):
            return "", ROW_NO_START, "", []
        if level == 0:
            held_objects = dict.fromkeys(arm.objects, 0)
            held_keys = dict.fromkeys(arm.keys, 0)
        else:
            held_objects, held_keys = self._handed(owner, point, arm)
            if not (held_objects or held_keys):
                return "", ROW_NOT_HANDED, "", []

        def order(held: dict[str, int]) -> list[str]:
            # The path the call is handed before one below it or an element of it, then the shorter, then the
            # wider: self.jobs[] -- any job -- before self.jobs[0].
            return sorted(
                held,
                key=lambda one: (
                    held[one],
                    objects._parts(one),
                    not objects._plain(one),
                    one,
                ),
            )

        starts: dict[str, str] = {}  # a path of the caller -> the object it says
        for one in order(held_objects):
            starts[one] = one
            starts.setdefault(f"{one}.{key}", one)
        for one in order(held_keys):
            starts.setdefault(one, one)
        another = ""
        hole = ""
        for start, said in starts.items():
            # What the caller holds the object by: its own bindings, and what the calls it makes hand back.
            whole = self._follow(owner, start)
            held = set(whole.held.get(owner, ())) | {start}
            for call in here.call_bindings:
                if call.line != line:
                    continue
                if call.to in self.writers and call.parameter == self._key_parameter(
                    call.to
                ):
                    holder = _holder(held, call.path)
                    if (
                        holder is not None
                        and self.writers[call.to].rpartition(".")[0] == cls
                    ):
                        return (
                            HISTORY_ROW_CAN,
                            "",
                            said,
                            [
                                *self._steps(owner, start, holder, whole),
                                f"{owner}:{line} {call.path}",
                            ],
                        )
                    another = another or call.path
                    continue
                for one in sorted(held):
                    rest = objects._under(call.path, one)
                    if rest is None:
                        continue
                    if not call.to:
                        hole = hole or call.why or "a hole"
                        continue
                    if not call.parameter or call.parameter.startswith("*"):
                        hole = hole or objects.NO_PARAMETER
                        continue
                    taken = call.parameter + call.at + rest
                    reach = self._follow(call.to, taken)
                    hole = hole or (reach.holes[0].how if reach.holes else reach.cut)
                    for write in sorted(writes):
                        target, _sep, at = write.rpartition(":")
                        paths = reach.held.get(target)
                        if not paths:
                            continue
                        for keyed in self.rows[target].call_bindings:
                            if (
                                str(keyed.line) != at
                                or keyed.to not in self.writers
                                or keyed.parameter != self._key_parameter(keyed.to)
                            ):
                                continue
                            holder = _holder(paths, keyed.path)
                            if (
                                holder is not None
                                and self.writers[keyed.to].rpartition(".")[0] == cls
                            ):
                                return (
                                    HISTORY_ROW_CAN,
                                    "",
                                    said,
                                    self._way(
                                        owner,
                                        line,
                                        start,
                                        one,
                                        call.to,
                                        taken,
                                        whole,
                                        reach,
                                        target,
                                        holder,
                                    )
                                    + [f"{write} {keyed.path}"],
                                )
                            another = another or keyed.path
        if another:
            return f"{ROW_ANOTHER}{another}", "", "", []
        if hole:
            return "", ROW_HOLE.format(why=hole), "", []
        return "", ROW_NOT_TO_WRITER, "", []

    def _steps(
        self, owner: str, start: str, path: str, reach: objects.ObjectReach
    ) -> list[str]:
        """How *owner* takes *start*'s object to *path*: its bindings (``owner:line how path``), from a call that
        handed the object back (``owner:line <- callee path``) where the bindings alone do not."""
        row = self.rows[owner]
        direct = objects.steps_to(row, start, path)
        if direct or path == start:
            return [f"{owner}:{line} {how} {held}" for line, how, held in direct]
        for back, (at, callee) in sorted(reach.returned.get(owner, {}).items()):
            tail = objects.steps_to(row, back, path)
            if back == path or tail:
                return [
                    f"{owner}:{at} <- {callee} {back}",
                    *(f"{owner}:{line} {how} {held}" for line, how, held in tail),
                ]
        return []

    def _way(
        self,
        owner: str,
        line: int,
        start: str,
        handed: str,
        callee: str,
        taken: str,
        whole: objects.ObjectReach,
        reach: objects.ObjectReach,
        target: str,
        holder: str,
    ) -> list[str]:
        """Each step the object takes from the caller's *start* to *holder* in *target*: bindings and calls."""
        way = [
            *self._steps(owner, start, handed, whole),
            f"{owner}:{line} -> {callee} {taken}",
        ]
        entry = {callee: taken}
        for caller, at, into, path, given in objects.way_to(reach, target):
            way += self._steps(caller, entry.get(caller, ""), given, reach)
            way.append(f"{caller}:{at} -> {into} {path}")
            entry[into] = path
        return way + self._steps(target, entry.get(target, taken), holder, reach)

    def _runs_at(
        self, owner: str, func: ast.AST, point: ast.stmt
    ) -> tuple[list[int], str]:
        """The caller's runs the point is in: its own loggers holding it, else its object's."""
        if owner not in self.prints:
            return [], RUN_NO_PRINTS
        held = self._holding(func, point)
        if held:
            return held, ""
        file = self.funcs[owner][0]
        objects, why = self._object_runs(file, func)
        if objects:
            return sorted(run.line for run in objects), ""
        return [], why or RUN_NONE


# -- syntax ------------------------------------------------------------------- #


def _called(call: ast.Call) -> str:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return ""


def _on_self(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    )


def _own_nodes(func: ast.AST) -> Iterator[ast.AST]:
    """The nodes of *func* outside the functions, lambdas and classes it holds."""
    pending = list(ast.iter_child_nodes(func))
    while pending:
        node = pending.pop()
        if isinstance(node, _SCOPES):
            continue
        yield node
        pending.extend(ast.iter_child_nodes(node))


def _enclosing_function(node: ast.AST) -> Optional[ast.AST]:
    for held in pathcond.ancestors(node):
        if isinstance(held, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return held
    return None


def _site_statement(site: ast.AST) -> Optional[ast.AST]:
    """A binding's site as :func:`reaching` takes it: a statement, or an ``except ... as``."""
    if isinstance(site, (ast.stmt, ast.ExceptHandler)):
        return site
    return mustprints._statement_of(site)


def _loops(node: ast.AST, func: ast.AST) -> list[ast.AST]:
    """The loops of *func* holding *node*."""
    held = []
    for one in pathcond.ancestors(node):
        if one is func:
            break
        if isinstance(one, _LOOPS):
            held.append(one)
    return held


def _root(node: ast.expr) -> ast.expr:
    while isinstance(node, ast.Attribute):
        node = node.value
    return node


def _is_parameter(func: ast.AST, name: str) -> bool:
    arguments = getattr(func, "args", None)
    if arguments is None:
        return False
    held = [
        *arguments.posonlyargs,
        *arguments.args,
        *arguments.kwonlyargs,
        arguments.vararg,
        arguments.kwarg,
    ]
    return any(one is not None and one.arg == name for one in held)


def _given(call: ast.Call, spec: Builder) -> tuple[bool, Optional[ast.expr]]:
    """Whether the factory call is given its mark, and the expression it is given."""
    for keyword in call.keywords:
        if keyword.arg == spec.keyword:
            return True, keyword.value
    leading = call.args[: spec.position + 1]
    if len(call.args) > spec.position and not any(
        isinstance(one, ast.Starred) for one in leading
    ):
        return True, call.args[spec.position]
    return False, None


def _pieces(expression: ast.expr, func: ast.AST, depth: int = 0) -> list[_Piece]:
    """*expression* as text and holes: literal parts as strings, every other part as its expression.

    Read through an f-string, ``+``, ``"...".format(...)`` and a local bound
    once; any other expression is one hole.
    """
    if isinstance(expression, ast.Constant) and isinstance(expression.value, str):
        return [expression.value]
    if isinstance(expression, ast.JoinedStr):
        out: list[_Piece] = []
        for part in expression.values:
            if isinstance(part, ast.Constant):
                out.append(str(part.value))
            elif isinstance(part, ast.FormattedValue):
                out.append(part.value)
        return out
    if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Add):
        return _pieces(expression.left, func, depth) + _pieces(
            expression.right, func, depth
        )
    if (
        isinstance(expression, ast.Call)
        and isinstance(expression.func, ast.Attribute)
        and expression.func.attr == "format"
        and isinstance(expression.func.value, ast.Constant)
        and isinstance(expression.func.value.value, str)
        and not expression.keywords
        and not any(isinstance(one, ast.Starred) for one in expression.args)
    ):
        formatted = _formatted(expression.func.value.value, expression.args)
        if formatted is not None:
            return formatted
    if (
        isinstance(expression, ast.Name)
        and depth < 2
        and isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef))
    ):
        bound = pathcond.single_definition(func, expression.id)
        if bound is not None and not _is_parameter(func, expression.id):
            return _pieces(bound, func, depth + 1)
    return [expression]


def _formatted(template: str, args: list[ast.expr]) -> Optional[list[_Piece]]:
    """``"<jediTaskID={0}>".format(x)`` as pieces; ``None`` for a field this reading does not place."""
    out: list[_Piece] = []
    automatic = 0
    try:
        parsed = list(string.Formatter().parse(template))
    except ValueError:
        return None
    for literal, field, _spec, _conversion in parsed:
        if literal:
            out.append(literal)
        if field is None:
            continue
        if field == "":
            index = automatic
            automatic += 1
        elif field.isdigit():
            index = int(field)
        else:
            return None
        if index >= len(args):
            return None
        out.append(args[index])
    return out
