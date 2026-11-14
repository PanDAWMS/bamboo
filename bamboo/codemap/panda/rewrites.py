"""The other columns an arm's block writes on its row, and the writes that can run after the arm in its pass.

A reader matching a row to the arms that could have written it -- the
querier's H2 on a run's rows file, the harness's production signatures --
reads more of the row than the arm's own column: ``parse_job_output_report``
sets ``jobStatus = "failed"`` beside ``ddmErrorCode = EC_Adder`` and
``ddmErrorDiag = f"expected output {lfn} is missing in pilot JSON"``, and a
row holding neither code nor text is one that arm did not leave.  Unless the
pass wrote them again after it: ``execute_plugin``'s ``except`` sets
``ddmErrorDiag = "AdderPlugin failure"`` later in the same pass, and the row
keeps the last.  Two records answer it, read once, at build time:

- :attr:`Branch.same_pass`: the other columns the block of the arm's write
  writes on its row (:class:`SamePassWrite`) -- another attribute of the
  object an attribute arm assigns on, or another key an SQL arm's map binds
  that the statement sets (``varMap[":jobSubStatus"]`` beside
  ``varMap[":jobStatus"]``), its row-naming keys left out.  A write with an
  early exit between it and the arm's is under another condition and is left
  out (:func:`pathcond.exits_above`).
- :attr:`Branch.rewrites`: each write of the arm's column, and of each
  ``same_pass`` column, that can run after the arm in its pass on the arm's
  object or its row (:class:`Rewrite`), with what it writes; and
  :attr:`Branch.rewrites_open`, where that was not read.

**Where.**  The walk up is the history pairs' (:meth:`runs._Reader.walk_up`):
level 0 is the arm's function, the point its write; level n+1 a caller, the
point its call on the way; a way the calls on it rule out is not the arm's
pass.  It goes up while the caller holds the arm's object
(:meth:`runs._Reader._handed`), not past :data:`runs.MAX_DEPTH`.  At each level,
in one pass after the point (:meth:`runs._Reader._ordered`: the next turn of a
loop is another pass), the object held by every path of the function its
bindings give it (:func:`objects._held`: a name bound to it, the element of
a list it is put in -- ``upJobs.append(job)`` before ``updateJobs(upJobs)``
--, read without the statements' order):

- a store: at level 0, a write of the column on the arm's holder that
  :func:`pathcond.field_replacements` finds can replace the value -- one it
  writes an entry for, or one it cannot (``left before``: the value may be
  read first, which is no reason the row keeps it); any other assignment of
  the column on a path holding the object;
- a call handed the object: what its run can write
  (:func:`bamboo.codemap.objects.may_change`) -- its assignments of the
  column on the object, each said written or not by a write of the run, and
  its statements writing the column on the object's row by its key -- read
  as that call runs the function it lands in: the constants the call hands
  (:meth:`Calls.route_constant`: a literal, or a default it leaves) to
  parameters the function's ``if`` tests read and it never binds again rule
  out what those tests keep from running (:func:`pathcond.ruled_out_by`),
  and what is on those lines -- calls, bindings, writes, holes,
  assignments -- is not read.  JobGenerator's ``storeJobs(...,
  toPending=True, bulk_job_insert=True)`` does not run the direct
  ``insertNewJob`` nor the Setupper.  One level: the function's own callees
  are read for any call.
- a row: a statement there writing the column on a row its key names by a
  path holding the object.

**What it does not read.**  Whether a store above level 0 comes before the
write that saves the row: one after it is listed too.  A ``same_pass``
column's later writes are read from the arm's point, not its own line.  The
later writes of an arm that assigns no attribute -- an SQL arm, a method run
on the object, a value carried -- (:data:`REWRITE_NOT_ATTRIBUTE`).  A column
the reach record keeps no assignment of (:data:`REWRITE_COLUMNS_UNRECORDED`): its calls cannot
be followed for it.  What a call's reading leaves open is said at the call,
once for each reason (:class:`RewriteOpen`): where inside its run is what
:func:`objects.may_change` says again, and an arm's function repeats a call's
for each arm.  An empty ``rewrites`` beside an empty ``rewrites_open`` is
none found, not none.
"""

from __future__ import annotations

import ast
import re
from functools import partial
from typing import TYPE_CHECKING, Optional

from bamboo.codemap import calls as C
from bamboo.codemap import objects, pathcond, values
from bamboo.codemap.models import (
    REWRITE_CALL,
    REWRITE_COLUMNS_UNRECORDED,
    REWRITE_CUT,
    REWRITE_DEPTH,
    REWRITE_NOT_ATTRIBUTE,
    REWRITE_ROW,
    REWRITE_ROW_KEY,
    REWRITE_RUN_TIME_COLUMNS,
    REWRITE_STATEMENT_UNREAD,
    REWRITE_STORE,
    REWRITE_UNPLACED,
    REWRITE_WIDE,
    Branch,
    DbWrite,
    JunctionNode,
    ReachNode,
    Rewrite,
    RewriteOpen,
    RewriteSite,
    SamePassWrite,
    SourceModule,
)
from bamboo.codemap.panda import mustprints, runs
from bamboo.codemap.panda import objects as panda_objects
from bamboo.codemap.panda.recognizers import progress

if TYPE_CHECKING:
    from bamboo.codemap.objects import Change

#: The replacements of a value :func:`pathcond.field_replacements` finds that are later writes of the row's column:
#: each can run after the value's write with the value still there, and leads on to where the value leaves.
LATER = frozenset(
    {
        pathcond.REPLACES,
        pathcond.SURELY,
        pathcond.LEFT_BEFORE,
        pathcond.SKIPPED,
        pathcond.NOT_ON_ITS_TESTS,
        pathcond.NO_TEST,
        pathcond.UNREADABLE,
    }
)

#: The name :attr:`RewriteSite.of_prior` gives the value a store replaces.
PRIOR = "PRIOR"

#: The column a class's row is named by, for :func:`objects.may_change`'s key: a history table's, and a file's.
ROW_KEYS: dict[str, str] = {**runs.HISTORY_KEYS, "FileSpec": "row_ID"}

#: Each reason :func:`objects.may_change` leaves an answer open for (:attr:`objects.Change.open`), by its template.
_OPEN_KINDS = (
    (objects.NOT_FOLLOWED_KEY, REWRITE_ROW_KEY),
    (objects.NOT_READ_HERE, REWRITE_ROW_KEY),
    (objects.UNREAD_KEY, REWRITE_ROW_KEY),
    (objects.NAMED_AT_RUN_TIME, REWRITE_RUN_TIME_COLUMNS),
    (objects.HOLE_NAMED, REWRITE_UNPLACED),
    (objects.HOLE_UNNAMED, REWRITE_UNPLACED),
)
_OPEN_PATTERNS = [
    (
        re.compile(
            ".*".join(re.escape(part) for part in re.split(r"\{\w+\}", template)), re.S
        ),
        kind,
    )
    for template, kind in _OPEN_KINDS
]
#: A hole the reach record holds for a statement it does not read (:attr:`ReachNode.holes`), as its why begins.
_NOT_READ = "the statement is not read"


def attach(reader: "runs._Reader", modules: list[SourceModule]) -> int:
    """Every branch's ``same_pass``, ``rewrites`` and ``rewrites_open``; returns how many branches have a rewrite."""
    later = _Later(reader, modules)
    found = 0
    for junction in reader.fragment.junctions:
        for branch in junction.branches:
            later.read(junction, branch)
            found += bool(branch.rewrites)
    return found


def _statements_by_line(func: ast.AST) -> dict[int, ast.stmt]:
    """Each line of *func*'s own body, with the innermost of its statements holding it."""
    found: dict[int, ast.stmt] = {}
    for node in C.own_nodes(func):
        if not isinstance(node, ast.stmt):
            continue
        end = node.end_lineno or node.lineno
        for line in range(node.lineno, end + 1):
            there = found.get(line)
            if (
                there is None
                or (there.end_lineno or there.lineno) - there.lineno
                >= end - node.lineno
            ):
                found[line] = node
    return found


def open_kind(how: str) -> str:
    """The ``REWRITE_*`` reason of what :func:`objects.may_change` leaves open (:attr:`objects.Change.open`)."""
    for pattern, kind in _OPEN_PATTERNS:
        if pattern.fullmatch(how):
            return kind
    return REWRITE_STATEMENT_UNREAD


class _Later:
    def __init__(self, reader: "runs._Reader", modules: list[SourceModule]):
        self.reader = reader
        self.rows: dict[str, ReachNode] = reader.rows
        self.tables = objects.table_classes(reader.fragment.entities)
        declared = progress.spec_attributes(modules)
        classes = panda_objects.column_classes(self.tables)
        #: The columns the reach record keeps assignments of (:func:`panda_objects.column_classes`).
        self.recorded: frozenset[str] = frozenset().union(
            *(declared.get(name, set()) for name in classes)
        )
        self._changes: dict[tuple, "Change"] = {}
        #: The constants a call hands its callee's tested parameters, by (caller, line, callee).
        self._constants: dict[tuple[str, int, str], tuple[tuple[str, object], ...]] = {}
        #: The record and may_change's memo with a callee's row read as a call hands it (:meth:`_as_called`).
        self._called: dict[tuple, tuple[dict[str, ReachNode], dict]] = {}
        self._replaced: dict = {}
        self._helds: dict[tuple[str, frozenset[str]], tuple[set[str], bool]] = {}
        self._names = objects._function_names(self.rows)
        #: :func:`objects.may_change`'s memo of what it reads of the whole record.
        self._record: dict = {}
        self._afters: dict[tuple[int, int, int], Optional[ast.stmt]] = {}

    # -- one branch ----------------------------------------------------------------------------------------------

    def read(self, junction: JunctionNode, branch: Branch) -> None:
        branch.same_pass, branch.rewrites, branch.rewrites_open = None, None, []
        sites = mustprints._sites(junction, branch, self.reader.trees)
        if isinstance(sites, str):
            return
        cls, _sep, attribute = junction.subject.rpartition(".")
        same: dict[tuple[str, int], SamePassWrite] = {}
        found: dict[tuple, Rewrite] = {}
        opened: dict[tuple, RewriteOpen] = {}
        read_any = False
        for site in sites:
            owner = self.reader.names.of(site.func)
            if owner is None or owner not in self.rows:
                return
            statement = site.statement
            written = mustprints._written_object(statement, junction.subject)
            if written is None:
                for one in self._bound_beside(owner, site.func, statement, attribute):
                    same.setdefault((one.column, one.line), one)
                _open(
                    opened,
                    RewriteOpen(
                        level=0,
                        caller=owner,
                        line=statement.lineno,
                        why=REWRITE_NOT_ATTRIBUTE,
                    ),
                )
                continue
            read_any = True
            beside = self._assigned_beside(site.func, statement, written, attribute)
            for one, _write in beside:
                same.setdefault((one.column, one.line), one)
            writes = [
                (attribute, statement),
                *((one.column, write) for one, write in beside),
            ]
            for column, write in writes:
                self._replacing(owner, statement, column, write, found)
                if column not in self.recorded:
                    _open(
                        opened,
                        RewriteOpen(
                            column=column,
                            level=0,
                            caller=owner,
                            line=statement.lineno,
                            why=REWRITE_COLUMNS_UNRECORDED,
                        ),
                    )
            lines = range(
                statement.lineno, (statement.end_lineno or statement.lineno) + 1
            )
            held = {
                one.path
                for one in self.rows[owner].assigns
                if one.line in lines and attribute in one.attrs
            }
            if not held:
                continue
            arm = runs._Arm(owner, frozenset(held), frozenset())
            columns = sorted(
                {column for column, _write in writes if column in self.recorded}
            )
            visit = partial(self._visit, cls, columns, arm, found, opened)
            met, _ruled = self.reader.walk_up(owner, site.func, statement, visit)
            for hole in met:
                why = REWRITE_DEPTH if hole == runs.HISTORY_DEPTH else REWRITE_WIDE
                _open(
                    opened,
                    RewriteOpen(level=0, caller=owner, line=statement.lineno, why=why),
                )
        branch.same_pass = sorted(same.values(), key=lambda one: (one.line, one.column))
        branch.rewrites = (
            sorted(
                found.values(),
                key=lambda one: (
                    one.column,
                    one.level,
                    one.caller,
                    one.point,
                    one.line,
                    one.how,
                ),
            )
            if read_any
            else None
        )
        branch.rewrites_open = sorted(
            opened.values(),
            key=lambda one: (
                one.column,
                one.level,
                one.caller,
                one.line,
                one.why,
            ),
        )

    # -- what the block writes beside the arm ----------------------------------------------------------------------

    def _assigned_beside(
        self, func: ast.AST, statement: ast.stmt, written: ast.expr, attribute: str
    ) -> list[tuple[SamePassWrite, ast.stmt]]:
        """Each other attribute of the arm's object the arm's block assigns, with no early exit between, and the
        statement assigning it."""
        holder = ast.unparse(written)
        found = []
        exits = None
        for one in _block(statement):
            target = _attribute_target(one)
            if (
                one is statement
                or target is None
                or target.attr == attribute
                or ast.unparse(target.value) != holder
            ):
                continue
            if exits is None:
                exits = pathcond.exits_above(statement)
            if pathcond.exits_above(one) != exits:
                continue
            got, frames = self._written_value(one, func)
            found.append(
                (
                    SamePassWrite(
                        column=target.attr, line=one.lineno, values=got, frames=frames
                    ),
                    one,
                )
            )
        return found

    def _bound_beside(
        self, owner: str, func: ast.AST, statement: ast.stmt, attribute: str
    ) -> list[SamePassWrite]:
        """Each other key of the map an SQL arm's bind fills, filled in the arm's block with no early exit between, that
        a statement the arm gives its value to sets: the column it sets, not a key naming the row."""
        keyed = _bind_target(statement)
        if keyed is None:
            return []
        name, _key = keyed
        lowered = attribute.lower()
        sets: dict[str, str] = {}
        for write in self.rows[owner].writes:
            if not any(value.column.lower() == lowered for value in write.columns):
                continue
            if not any(
                self.reader._gives(func, statement, line, attribute)
                for line in write.lines
            ):
                continue
            naming = {key.column.lower() for key in write.keys}
            sets.update(
                {
                    value.column.lower(): value.column
                    for value in write.columns
                    if value.column.lower() not in naming
                }
            )
        found = []
        exits = None
        for one in _block(statement):
            other = _bind_target(one) if one is not statement else None
            if other is None or other[0] != name:
                continue
            column = sets.get(other[1].lstrip(":").lower())
            if column is None or column.lower() == lowered:
                continue
            if exits is None:
                exits = pathcond.exits_above(statement)
            if pathcond.exits_above(one) != exits:
                continue
            got, frames = self._value(one.value, func)  # type: ignore[attr-defined]
            found.append(
                SamePassWrite(column=column, line=one.lineno, values=got, frames=frames)
            )
        return found

    # -- later writes ------------------------------------------------------------------------------------------------

    def _replacing(
        self,
        owner: str,
        arm: ast.stmt,
        column: str,
        write: ast.stmt,
        found: dict[tuple, Rewrite],
    ) -> None:
        """The level-0 stores: the writes of *column* on the holder of *write* that can replace its value."""
        target = _attribute_target(write)
        if target is None:
            return
        replaced = pathcond.field_replacements(
            write,
            target.value,
            column,
            getattr(write, "value", None),
            None,
            self._replaced,
        )
        func = pathcond.enclosing_function(write)
        for one in replaced or ():
            if one.why not in LATER or not isinstance(one.site, ast.stmt):
                continue
            got, frames = self._written_value(one.site, func, column)
            line = one.site.lineno
            site = RewriteSite(
                owner=owner,
                line=line,
                how=REWRITE_STORE,
                values=got,
                frames=frames,
                of_prior=_of_prior(one.site, func, column),
            )
            _found(
                found,
                Rewrite(
                    column=column,
                    level=0,
                    caller=owner,
                    point=arm.lineno,
                    line=line,
                    how=REWRITE_STORE,
                    sites=[site],
                ),
            )

    def _visit(
        self,
        cls: str,
        columns: list[str],
        arm: "runs._Arm",
        found: dict[tuple, Rewrite],
        opened: dict[tuple, RewriteOpen],
        step: "runs._Step",
    ) -> bool:
        """:meth:`runs._Reader.walk_up`'s visit: the later writes at *step*; ``True`` (go no higher) where the caller
        does not hold the arm's object."""
        if step.level == 0:
            own = set(arm.objects)
            held = own
        else:
            own = set()
            held = set(self.reader._handed(step.owner, step.point, arm)[0])
            if not held:
                return True
        row = self.rows[step.owner]
        func, point = step.func, step.point
        # Every path of the function holding the object: a name bound to it, the element of a list it is put in
        # (``upJobs.append(job)`` before ``updateJobs(upJobs)``), read without the statements' order.
        held, spent = self._held_in(step.owner, held)
        if spent:
            for column in columns:
                _open(
                    opened,
                    RewriteOpen(
                        column=column,
                        level=step.level,
                        caller=step.owner,
                        line=point.lineno,
                        why=REWRITE_CUT,
                    ),
                )

        after = partial(self._after, func, point)
        calls = self._handed_calls(row, held)
        for column in columns:
            # At the arm's function its own holder's stores are the replacements read already; another name's are not.
            for one in row.assigns:
                if (
                    one.path in held
                    and one.path not in own
                    and column in one.attrs
                    and (statement := after(one.line)) is not None
                ):
                    got, frames = self._written_value(statement, func, column)
                    site = RewriteSite(
                        owner=step.owner,
                        line=one.line,
                        how=REWRITE_STORE,
                        values=got,
                        frames=frames,
                        of_prior=_of_prior(statement, func, column),
                    )
                    _found(
                        found,
                        Rewrite(
                            column=column,
                            level=step.level,
                            caller=step.owner,
                            point=point.lineno,
                            line=one.line,
                            how=REWRITE_STORE,
                            sites=[site],
                        ),
                    )
            for write in row.writes:
                line = write.lines[0]
                by = self._row_values(write, column, held, cls)
                if by is not None and after(line) is not None:
                    site = RewriteSite(
                        owner=step.owner, line=line, how=REWRITE_ROW, values=by
                    )
                    _found(
                        found,
                        Rewrite(
                            column=column,
                            level=step.level,
                            caller=step.owner,
                            point=point.lineno,
                            line=line,
                            how=REWRITE_ROW,
                            sites=[site],
                        ),
                    )
            self._calls(cls, column, calls, step, after, found, opened)
        for hole in row.holes:
            whys = [
                (column, why)
                for column in columns
                if (why := self._unplaced(hole.callee, hole.why, column, cls))
            ]
            if not whys or after(hole.line) is None:
                continue
            for column, why in whys:
                _open(
                    opened,
                    RewriteOpen(
                        column=column,
                        level=step.level,
                        caller=step.owner,
                        line=hole.line,
                        why=why,
                    ),
                )
        return False

    def _after(self, func: ast.AST, point: ast.stmt, line: int) -> Optional[ast.stmt]:
        """The statement of *func* at *line* where it can run after *point* in one pass, else ``None``."""
        key = (id(func), id(point), line)
        if key not in self._afters:
            statement = self.reader._statement(func, line)
            self._afters[key] = (
                None
                if statement is None
                or statement is point
                or not self.reader._after(func, point, statement)
                else statement
            )
        return self._afters[key]

    def _handed_calls(
        self, row: ReachNode, held: set[str]
    ) -> dict[tuple[int, str], set[str]]:
        """The plain calls of *row* handed the object *held* holds: each ``(line, callee)``, and the callee's paths
        holding it."""
        plain = {
            (call.line, call.to)
            for call in row.calls
            if call.how == "call" and not call.thread
        }
        handed: dict[tuple[int, str], set[str]] = {}
        for binding in row.call_bindings:
            if (
                (binding.line, binding.to) not in plain
                or not binding.parameter
                or binding.parameter.startswith("*")
            ):
                continue
            for path in held:
                rest = objects._under(binding.path, path)
                if rest is not None:
                    handed.setdefault((binding.line, binding.to), set()).add(
                        binding.parameter + binding.at + rest
                    )
        return handed

    def _calls(self, cls, column, handed, step, after, found, opened) -> None:
        """The later writes of *column* in the runs of the calls at *step* after its point that are handed the
        object (:meth:`_handed_calls`)."""
        for (line, to), paths in sorted(handed.items()):
            if to not in self.rows or after(line) is None:
                continue
            sites: dict[tuple[str, int, str], RewriteSite] = {}
            for path in sorted(paths):
                change = self._change(to, path, column, cls, step.owner, line)
                for one in change.because:
                    site = self._site(one, column, True)
                    if site is not None:
                        sites.setdefault((site.owner, site.line, site.how), site)
                for one in change.unwritten:
                    site = self._site(one, column, False)
                    if site is not None:
                        sites.setdefault((site.owner, site.line, site.how), site)
                for one in change.open:
                    _open(
                        opened,
                        RewriteOpen(
                            column=column,
                            level=step.level,
                            caller=step.owner,
                            line=line,
                            why=open_kind(one.how),
                        ),
                    )
                reach = self.reader._follow(to, path)
                if reach.cut or reach.capped:
                    _open(
                        opened,
                        RewriteOpen(
                            column=column,
                            level=step.level,
                            caller=step.owner,
                            line=line,
                            why=REWRITE_CUT,
                        ),
                    )
            if sites:
                _found(
                    found,
                    Rewrite(
                        column=column,
                        level=step.level,
                        caller=step.owner,
                        point=step.point.lineno,
                        line=line,
                        how=REWRITE_CALL,
                        sites=sorted(
                            sites.values(),
                            key=lambda one: (one.owner, one.line, one.how),
                        ),
                    ),
                )

    def _held_in(self, owner: str, held: set[str]) -> tuple[set[str], bool]:
        """Every path of *owner* holding what *held* holds (:func:`objects._held`), and whether its cap was met."""
        key = (owner, frozenset(held))
        if key not in self._helds:
            found, spent = objects._held(self.rows[owner], set(held))
            self._helds[key] = (found | set(held), spent)
        return self._helds[key]

    def _change(
        self,
        owner: str,
        path: str,
        column: str,
        cls: str,
        caller: str = "",
        line: int = 0,
    ) -> "Change":
        """:func:`objects.may_change` of *owner*'s run, as the call at *caller*:*line* runs it (:meth:`_as_called`)."""
        handed = self._handed_constants(caller, line, owner) if caller else ()
        key = (owner, path, column, cls, handed)
        if key not in self._changes:
            rows, memo = self._as_called(owner, handed)
            self._changes[key] = objects.may_change(
                rows,
                owner,
                path,
                column,
                key_column=ROW_KEYS.get(cls, "PandaID"),
                spec_class=cls,
                table_classes=self.tables,
                memo=memo,
            )
        return self._changes[key]

    def _handed_constants(
        self, caller: str, line: int, owner: str
    ) -> tuple[tuple[str, object], ...]:
        """The constants the one call at *caller*:*line* into *owner* hands it (:func:`panda_objects.route_constants`).
        Empty where the line holds no call into it or more than one."""
        key = (caller, line, owner)
        if key not in self._constants:
            found: dict[str, object] = {}
            held = self.reader.funcs.get(owner)
            sites = (
                self.reader._sites_into(owner, caller, line)
                if held is not None and caller in self.reader.funcs
                else []
            )
            if len(sites) == 1:
                found = panda_objects.route_constants(
                    self.reader.calls, self.reader._target(owner), (sites[0],)
                )
            self._constants[key] = tuple(sorted(found.items(), key=lambda one: one[0]))
        return self._constants[key]

    def _as_called(
        self, owner: str, handed: tuple[tuple[str, object], ...]
    ) -> tuple[dict[str, ReachNode], dict]:
        """The record, and a memo for :func:`objects.may_change`, with *owner*'s row read as a call handing *handed*
        runs it: what is on a line the constants rule out (:func:`pathcond.ruled_out_by`) is not read -- a call all of
        whose lines are, a call binding, a write all of whose lines are, a hole, a binding, an assignment.  The
        record itself where none is ruled out."""
        key = (owner, handed)
        if key not in self._called:
            self._called[key] = (self.rows, self._record)
            held = self.reader.funcs.get(owner)
            row = self.rows.get(owner)
            if handed and held is not None and row is not None:
                func, values = held[1], dict(handed)
                dead = {
                    line
                    for line, statement in _statements_by_line(func).items()
                    if pathcond.ruled_out_by(statement, func, values)
                }
                if dead:
                    rows = dict(self.rows)
                    rows[owner] = row.model_copy(
                        update={
                            "calls": [
                                one
                                for one in row.calls
                                if not set(one.lines or [one.line]) <= dead
                            ],
                            "call_bindings": [
                                one for one in row.call_bindings if one.line not in dead
                            ],
                            "writes": [
                                one for one in row.writes if not set(one.lines) <= dead
                            ],
                            "holes": [one for one in row.holes if one.line not in dead],
                            "bindings": [
                                one for one in row.bindings if one.line not in dead
                            ],
                            "assigns": [
                                one for one in row.assigns if one.line not in dead
                            ],
                        }
                    )
                    self._called[key] = (rows, {})
        return self._called[key]

    def _site(
        self, one: objects.Site, column: str, written: bool
    ) -> Optional[RewriteSite]:
        """A store or a row write of a call's run, from :func:`objects.may_change`'s site; ``None`` for a write that
        saves the object."""
        row = self.rows.get(one.owner)
        if row is not None:
            for write in row.writes:
                if write.lines[0] != one.line:
                    continue
                if one.how == objects.ROW_WRITE.format(
                    kind=write.kind, table=write.table, column=column
                ):
                    got = self._row_values(write, column, None, "")
                    return RewriteSite(
                        owner=one.owner,
                        line=one.line,
                        how=REWRITE_ROW,
                        values=got or [],
                    )
                if one.how == f"{write.kind} {write.table}":
                    return None
        held = self.reader.funcs.get(one.owner)
        statement = (
            self.reader._statement(held[1], one.line) if held is not None else None
        )
        if statement is None:
            return RewriteSite(
                owner=one.owner,
                line=one.line,
                how=REWRITE_STORE,
                values=["runtime(?)"],
                written=written,
            )
        got, frames = self._written_value(statement, held[1], column)  # type: ignore[index]
        return RewriteSite(
            owner=one.owner,
            line=one.line,
            how=REWRITE_STORE,
            values=got,
            frames=frames,
            written=written,
            of_prior=_of_prior(statement, held[1], column),  # type: ignore[index]
        )

    def _row_values(
        self, write: DbWrite, column: str, held: Optional[set[str]], cls: str
    ) -> Optional[list[str]]:
        """What *write* writes *column* by, where it writes it by a value of its own on a row named by a path *held*
        holds the object by (any, where *held* is None); else ``None``."""
        lowered = column.lower()
        if write.whole:
            return None
        if objects.other_class(write, cls, self.tables):
            return None
        if held is not None and not any(
            key.path
            and any(objects._under(path, key.path) is not None for path in held)
            for key in write.keys
        ):
            return None
        got = []
        for value in write.columns:
            if value.column.lower() != lowered:
                continue
            if (
                held is not None
                and value.path
                and any(objects._under(path, value.path) is not None for path in held)
            ):
                continue
            got.append(
                _spelled(value.value)
                if value.value
                else f"runtime({value.path or '?'})"
            )
        return sorted(set(got)) or None

    def _unplaced(self, callee: str, why: str, column: str, cls: str) -> str:
        """Why a hole of the reach record may hide a write of *column*, or ``""`` where its name says it does not."""
        if why.startswith(_NOT_READ):
            return REWRITE_STATEMENT_UNREAD
        name = objects._called_name(callee)
        if name is None:
            return REWRITE_UNPLACED
        up = objects.writers_of(self.rows, column, cls, self.tables, self._record)
        return REWRITE_UNPLACED if self._names.get(name, set()) & up else ""

    # -- values ------------------------------------------------------------------------------------------------------

    def _written_value(
        self, statement: ast.stmt, func: Optional[ast.AST], column: str = ""
    ) -> tuple[list[str], list[str]]:
        """What *statement* writes: its value where it assigns one (to *column*'s attribute where named)."""
        if isinstance(statement, ast.AugAssign):
            return [
                progress._runtime(
                    ast.BinOp(statement.target, statement.op, statement.value)
                )
            ], []
        if (
            isinstance(statement, (ast.Assign, ast.AnnAssign))
            and statement.value is not None
        ):
            targets = (
                statement.targets
                if isinstance(statement, ast.Assign)
                else [statement.target]
            )
            if not column or any(
                isinstance(one, ast.Attribute) and one.attr == column for one in targets
            ):
                return self._value(statement.value, func)
        return ["runtime(?)"], []

    def _value(
        self, expression: ast.expr, func: Optional[ast.AST]
    ) -> tuple[list[str], list[str]]:
        """Each literal *expression* settles to, else ``runtime(<expression>)`` and the texts it can be."""
        settled = self.reader._settle(expression, func)
        if settled:
            return sorted(set(settled)), []
        reading = values.value_reading(expression, func)  # type: ignore[arg-type]
        return [progress._runtime(expression)], list(reading.frames)


def _of_prior(statement: ast.stmt, func: Optional[ast.AST], column: str) -> str:
    """What *statement*, a store of *column*, writes as an expression of the value it replaces, that value spelled
    :data:`PRIOR`, where it reads it -- the same holder's *column* -- in its value (``x.col = f(x.col)``, ``x.col +=
    v``) or in the value of the one binding reaching the local its value is (``newJobName = re.sub(..., job.jobName)``
    before ``job.jobName = newJobName``); else ``""``."""
    if isinstance(statement, ast.AugAssign):
        augmented = statement.target
        if not (isinstance(augmented, ast.Attribute) and augmented.attr == column):
            return ""
        prior = ast.Name(id=PRIOR, ctx=ast.Load())
        return ast.unparse(
            ast.BinOp(left=prior, op=statement.op, right=statement.value)
        )
    if (
        not isinstance(statement, (ast.Assign, ast.AnnAssign))
        or statement.value is None
    ):
        return ""
    targets = (
        statement.targets if isinstance(statement, ast.Assign) else [statement.target]
    )
    target = next(
        (
            one
            for one in targets
            if isinstance(one, ast.Attribute) and one.attr == column
        ),
        None,
    )
    if target is None:
        return ""
    holder = ast.unparse(target.value)
    value = statement.value
    spelled = _with_prior(value, holder, column)
    if spelled or not isinstance(value, ast.Name):
        return spelled
    if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return ""
    sites = pathcond.held_at(func, value.id, statement).sites
    if len(sites) != 1:
        return ""
    (site,) = sites
    if (
        isinstance(site, ast.Assign)
        and len(site.targets) == 1
        and isinstance(site.targets[0], ast.Name)
    ) or (isinstance(site, ast.AnnAssign) and isinstance(site.target, ast.Name)):
        return _with_prior(site.value, holder, column) if site.value is not None else ""
    return ""


def _with_prior(value: ast.expr, holder: str, column: str) -> str:
    """*value* with :data:`PRIOR` for each read of *holder*'s *column* (``holder.column``, by its text); ``""`` where
    it reads none."""
    fresh = ast.parse(ast.unparse(value), mode="eval").body
    found = False

    class _Prior(ast.NodeTransformer):
        def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
            nonlocal found
            if node.attr == column and ast.unparse(node.value) == holder:
                found = True
                return ast.Name(id=PRIOR, ctx=ast.Load())
            return self.generic_visit(node)

    spelled = _Prior().visit(fresh)
    return ast.unparse(spelled) if found else ""


def _spelled(text: str) -> str:
    """A statement's recorded literal (:attr:`WrittenValue.value`, as Python writes it) spelled as a branch's outcome
    is: a string's text, any other value as Python spells it."""
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text
    return value if isinstance(value, str) else repr(value)


def _found(found: dict[tuple, Rewrite], rewrite: Rewrite) -> None:
    key = (
        rewrite.column,
        rewrite.level,
        rewrite.caller,
        rewrite.point,
        rewrite.line,
        rewrite.how,
    )
    held = found.get(key)
    if held is None:
        found[key] = rewrite
        return
    sites = {(one.owner, one.line, one.how): one for one in held.sites}
    for one in rewrite.sites:
        sites.setdefault((one.owner, one.line, one.how), one)
    held.sites = sorted(sites.values(), key=lambda one: (one.owner, one.line, one.how))


def _open(opened: dict[tuple, RewriteOpen], one: RewriteOpen) -> None:
    opened.setdefault((one.column, one.level, one.caller, one.line, one.why), one)


def _block(statement: ast.stmt) -> list[ast.stmt]:
    """The statements of the block holding *statement*."""
    parent = getattr(statement, "parent", None)
    for field in ("body", "orelse", "finalbody"):
        block = getattr(parent, field, None)
        if isinstance(block, list) and any(one is statement for one in block):
            return block
    return [statement]


def _attribute_target(statement: ast.AST) -> Optional[ast.Attribute]:
    """The attribute *statement* assigns on a name or attributes on one, where it assigns one alone."""
    if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
        target = statement.targets[0]
    elif isinstance(statement, (ast.AnnAssign, ast.AugAssign)):
        target = statement.target
    else:
        return None
    if isinstance(target, ast.Attribute) and mustprints._chain(target.value):
        return target
    return None


def _bind_target(statement: ast.AST) -> Optional[tuple[str, str]]:
    """``(map, key)`` where *statement* fills a literal key of a map named by a name: ``varMap[":status"] = v``."""
    if not (isinstance(statement, ast.Assign) and len(statement.targets) == 1):
        return None
    target = statement.targets[0]
    if (
        isinstance(target, ast.Subscript)
        and isinstance(target.value, ast.Name)
        and isinstance(target.slice, ast.Constant)
        and isinstance(target.slice.value, str)
    ):
        return target.value.id, target.slice.value
    return None
