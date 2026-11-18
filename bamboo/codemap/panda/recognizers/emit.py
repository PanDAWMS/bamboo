"""The line a junction leaves behind when it settles a value.

Filter stages have carried their message templates from the start -- 94 of 109
have one and 104 have a level -- and junctions never did: 0 of 1066 branches.
The mechanism existed and was simply never pointed at the other node kind, and
what that cost is visible in the consumer.  The report had to keep a
hard-coded dictionary of one entry, ``set task_status={value}``, so
``derive-strategy`` could observe exactly one subject and had to report every
other as a capability gap.  The skeleton reads these now, and the pooled
question that dictionary stood in for is gone.

Written as an attach pass rather than inside the slices that build junctions,
because two of them do -- the attribute slice and the SQL slice -- and reading
the same functions twice is how the two ended up disagreeing about a fact
before.  It runs over the merged junctions, keyed by owner, the way
:mod:`bamboo.codemap.panda.recognizers.flush` does.

Three readings, each answering a way the naive rule is wrong.

**A message is usually built before it is logged.**  ``tmpMsg = f"set
task_status={taskSpec.status}"`` then ``tmpLog.info(tmpMsg)`` accounts for 99
of the 265 matching emits in the corpus, and without following it every one of
the four knights that writes ``set task_status=`` is missed -- which is every
writer the transition gate reads.  The same shape gate 1 found when
``criteria = "-link_unusable"`` was assigned before being interpolated.

**Whether a line is about this junction is not a substring test.**  The first
estimate of this harvest matched the attribute name against the rendered text,
which counted ``status`` inside ``task_status`` and would have counted every
line mentioning a status anywhere.  What ties a line to a junction is that it
interpolates the value: either an attribute of the right spec class, or a local
holding one of the values this junction writes.  ``TaskRefiner`` logs the local
and is only reachable by the second.  A value the branch could only spell
(``runtime(<expr>)``) is tied by a binding the line and the write share, not
by the spelling: the same text in another place is another value, and a line
the branch's run prints with another value hands the reader a wrong one --
which no width argument below covers.

**A line under one arm is not evidence for the other.**  ``ContentsFeeder``
writes ``set task_status=`` in three arms of one function.  Attaching all three
to every branch would lose which arm; :func:`pathcond.exclusive` filters them,
and it is conservative in the direction that costs least here -- an emit
attached too widely means a grep that comes back empty, which the surrounding
design already treats as proving nothing, while one attached too narrowly means
no probe can be built at all.
"""

from __future__ import annotations

import ast
import re
from typing import NamedTuple, Optional

from bamboo.codemap import values
from bamboo.codemap.astwalk import walk
from bamboo.codemap.calls import def_name
from bamboo.codemap.models import (
    REPORTS_DECISION,
    REPORTS_ROWS_CHANGED,
    Branch,
    Emit,
    JunctionNode,
    SourceModule,
)
from bamboo.codemap.panda import sql
from bamboo.codemap.panda.attribution import SpecAttributor
from bamboo.codemap.panda.recognizers import logfile
from bamboo.codemap.panda.recognizers.selection import log_level
from bamboo.codemap.pathcond import (
    ancestors,
    assigned_expressions,
    attach_parents,
    bindings,
    exclusive,
    functions_with_owner,
    held_at,
    literal_values,
    path_condition,
    reaching_values,
    targets_of,
    written_value,
)

#: ``runtime(<expr>)`` -- how a branch records an outcome it cannot value.
#: The expression is the connection to a line that interpolates the same one.
_RUNTIME = re.compile(r"^runtime\((.*)\)$", re.DOTALL)


def _logged_arguments(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[tuple[ast.expr, ast.expr]]:
    """Every ``(argument, message expression)`` a logging call in *func* writes.

    The message is resolved one hop through a local, which is how most of the
    corpus writes it -- see the module docstring.  Both come back: the argument
    sits at the call, which is where the level and the dominating conditions
    are read from, and the expression carries the text -- and for the local
    spelling those are in two different places.
    """
    assignments = assigned_expressions(func)

    found: list[tuple[ast.expr, ast.expr]] = []
    for argument in logging_arguments(func):
        if isinstance(argument, ast.Name):
            found.extend((argument, built) for built in assignments.get(argument.id, ()))
        else:
            found.append((argument, argument))
    return found


def logging_arguments(func: ast.AST) -> list[ast.expr]:
    """The message argument of every logging call in *func*, resolved or not.

    Split out of :func:`_logged_arguments` because the two questions differ and
    only one of them used to be answerable.  That function pairs a call with
    the messages it could be printing, and a bare name it can resolve to
    nothing yields no pair at all -- so the call vanishes, where a call holding
    an expression it cannot render at least leaves a refusal.  Sixty-five calls
    in this corpus go that way, and a reader lays the skeleton over a grep of
    the log region: a line present there and absent here reads as certainty.

    A call is a print only as a statement of its own, its value thrown away.
    The level is read off the method's name, and ``raise socket.error("empty
    packet")`` is a method named ``error`` whose value is the exception: read
    as a print, it put two rows in ``ConBridge.bridge_recv``'s skeleton that
    no log can hold.  Those two are the only calls in the corpus the rule
    removes.
    """
    found: list[ast.expr] = []
    for node in walk(func):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if not node.args:
            continue
        if not isinstance(getattr(node, "parent", None), ast.Expr):
            continue
        argument = node.args[0]
        # ``log_level`` reads the level off the call *enclosing* a node, so it
        # is asked about the argument rather than about the call.
        if log_level(argument) is None:
            continue
        found.append(argument)
    return found


def own_logging_arguments(func: ast.AST) -> list[ast.expr]:
    """:func:`logging_arguments` less the calls a ``def`` or ``class`` nested in *func* makes.

    ``ast.walk`` goes into a nested function, and its lines are its own: they
    print when it is called, under its own name.  125 calls in this corpus sit
    in twenty nested functions.
    """
    found: list[ast.expr] = []
    for argument in logging_arguments(func):
        scope = next(
            (
                ancestor
                for ancestor in ancestors(argument)
                if isinstance(
                    ancestor, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
                )
            ),
            None,
        )
        if scope is func:
            found.append(argument)
    return found


def _row_counts(func: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Locals holding what a write returned -- ``nRows = self.cur.rowcount``.

    A line interpolating one of these is about the write landing rather than
    about the value, and that is the whole of what the map can observe of a
    compare-and-set losing: the count comes back, most callers discard it, and
    the one place it is written down is a line like ``updated 0 rows``.
    """
    found: set[str] = set()
    for node in walk(func):
        value = written_value(node)
        if not (isinstance(value, ast.Attribute) and value.attr == "rowcount"):
            continue
        found |= {t.id for t in targets_of(node) if isinstance(t, ast.Name)}
    return found


def _reports_rows(message: ast.expr, counts: set[str]) -> bool:
    """Whether *message* interpolates a row count this function took."""
    if not isinstance(message, ast.JoinedStr) or not counts:
        return False
    return any(
        isinstance(part, ast.FormattedValue) and ast.unparse(part.value) in counts
        for part in message.values
    )


def row_count_lines(
    func: ast.FunctionDef | ast.AsyncFunctionDef, files: list[str]
) -> list[Emit]:
    """The lines *func* writes about how many rows its write changed.

    Public because the flush pass needs the same reading: a knight decides in
    memory and a proxy persists it, so the line saying whether the row moved is
    written in a different function from the one that chose the value, and the
    caller's branches are where an investigation looks for it.
    """
    counts = _row_counts(func)
    if not counts:
        return []
    found: list[Emit] = []
    for at_call, message in _logged_arguments(func):
        if not _reports_rows(message, counts):
            continue
        template = values.rendered_text(message)
        if not template or not values.has_literal_text(template):
            continue
        if any(e.template == template for e in found):
            continue
        found.append(
            Emit(
                template=template,
                log_level=log_level(at_call),
                log_files=files,
                reports=REPORTS_ROWS_CHANGED,
            )
        )
    return found


def _names_the_value(
    message: ast.expr,
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    owner: Optional[str],
    attributor: SpecAttributor,
    spec_class: str,
    attribute: str,
    branches: list[Branch],
    settle,
    held: _Held,
) -> bool:
    """Whether *message* interpolates the value this junction settles.

    Three spellings, and the corpus needs all three: the line reads the spec
    back (``{taskSpec.status}``); it logs the local that was just assigned to
    it, tied by the values that local can hold rather than by its name; or the
    branch could not value the local at all and recorded ``runtime(<expr>)``,
    in which case the line names that value where it holds what the branch
    reads, by a binding the two share and never by the text
    (:func:`_holds_the_read_value`).

    The third is not a weaker version of the second -- it is what is left when
    the second fails, and it is exactly where the line is most worth having,
    since the map has nothing else to say about that branch's value.
    *held* holds the bindings read so far, a function's lines being asked
    about at every junction it writes.
    """
    if not isinstance(message, ast.JoinedStr):
        return False
    outcomes = {b.outcome for b in branches}
    spelled: dict[str, list[Optional[int]]] = {}
    for branch in branches:
        if m := _RUNTIME.match(branch.outcome):
            spelled.setdefault(m.group(1), []).append(branch.line)
    at = next((one for one in ancestors(message) if isinstance(one, ast.stmt)), None)
    for part in message.values:
        if not isinstance(part, ast.FormattedValue):
            continue
        shown = part.value
        if isinstance(shown, ast.Attribute) and shown.attr == attribute:
            if attributor.class_of(shown.value, func, owner) == spec_class:
                return True
        if isinstance(shown, ast.Name):
            reached = {value for value, _conditions, _line in literal_values(func, shown.id, settle)}
            if reached & outcomes:
                return True
        if at is not None and spelled and _holds_the_read_value(shown, at, spelled, held):
            return True
    return False


def _holds_the_read_value(
    shown: ast.expr, at: ast.stmt, spelled: dict[str, list[Optional[int]]], held: _Held
) -> bool:
    """Whether *shown*, interpolated at *at*, holds what a branch recorded as ``runtime(<S>)`` reads where it reads
    it (:meth:`_Held.read_at`): tied by a binding the two share, as the same text elsewhere is another value.

    A local the line names holds it where a binding of the local reaching the line is one that holds the value
    there: ``killJob``'s ``currentTime = naive_utcnow()``.  ``upsert_workflow_entities`` writes the ``workflow_id``
    it bound from the insert, and its last line, past the merge with the parameter, logs that binding; the lines
    above the insert log the parameter, which the write never reads.  ``getOutputFiles_JEDI``'s ``fileDatasetID =
    int(val)`` is not the ``int(val)`` its file ID is written from, ``updateWorkers``' ``panda_id`` of the rows it
    reads back is not the one of the loop it writes in, and ``insertFilesForDataset_JEDI``'s ``dsStateInDB`` is
    another slot of the one unpacking that binds the ``nFilesToUseDS`` it writes.  An expression holds it where
    every local it reads shares a binding with the read; one that reads no local -- a call -- is another
    evaluation, never the value read.
    """
    if isinstance(shown, ast.Name):
        here = held.sites(shown.id, at)
        for spelling, lines in spelled.items():
            for line in lines:
                read = held.read_at(spelling, line)
                if read is not None and _share(here, read.holders.get(shown.id, ())):
                    return True
        return False
    text = ast.unparse(shown)
    if text not in spelled:
        return False
    names = sorted({one.id for one in walk(shown) if isinstance(one, ast.Name) and isinstance(one.ctx, ast.Load)})
    names = [name for name in names if held.is_local(name)]
    if not names:
        return False
    for line in spelled[text]:
        read = held.read_at(text, line)
        if read is not None and all(_share(held.sites(name, at), held.sites(name, read.place)) for name in names):
            return True
    return False


def _share(one: tuple[ast.AST, ...], other: tuple[ast.AST, ...]) -> bool:
    return any(site is seen for site in one for seen in other)


def _own_expressions(statement: ast.stmt) -> list[ast.expr]:
    """The expressions *statement* itself reads, not those of the statements in its body nor its targets."""
    found: list[ast.expr] = []
    todo = [child for child in ast.iter_child_nodes(statement) if not isinstance(child, ast.stmt)]
    while todo:
        node = todo.pop()
        if isinstance(node, ast.expr) and not isinstance(getattr(node, "ctx", None), (ast.Store, ast.Del)):
            found.append(node)
        todo.extend(child for child in ast.iter_child_nodes(node) if not isinstance(child, ast.stmt))
    return found


class _Read(NamedTuple):
    """Where a branch recorded as ``runtime(<S>)`` reads its value (:meth:`_Held.read_at`)."""

    #: the statement that reads *S* itself: the branch's own, or the one binding that binds a local it reads whole
    #: to *S*
    place: ast.AST
    #: each local that holds the value there, and the bindings of it that do
    holders: dict[str, tuple[ast.AST, ...]]


class _Held:
    """What a function's lines hold, read once per function: the bindings of a local that reach a statement
    (:func:`pathcond.held_at`, the function's entry among them), and where a branch reads the value it spells."""

    def __init__(self, func: ast.FunctionDef | ast.AsyncFunctionDef, settle) -> None:
        self.func = func
        self.settle = settle
        self._sites: dict[tuple[str, int], tuple[ast.AST, ...]] = {}
        self._reads: dict[tuple[str, Optional[int]], Optional[_Read]] = {}
        self._lines: Optional[dict[int, ast.stmt]] = None
        self._locals: Optional[set[str]] = None

    def sites(self, name: str, at: ast.AST) -> tuple[ast.AST, ...]:
        """The bindings of *name* that can be what *at* reads."""
        key = (name, id(at))
        if key not in self._sites:
            self._sites[key] = held_at(self.func, name, at).sites
        return self._sites[key]

    def is_local(self, name: str) -> bool:
        """Whether *name* is a parameter of the function or bound in it."""
        if self._locals is None:
            arguments = self.func.args
            self._locals = {one.arg for one in [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]}
            self._locals |= {one.arg for one in (arguments.vararg, arguments.kwarg) if one is not None}
        return name in self._locals or any(not one.nested for one in bindings(self.func, name))

    def read_at(self, spelled: str, line: Optional[int]) -> Optional[_Read]:
        """Where a branch on *line* recorded as ``runtime(<spelled>)`` reads its value, and which locals hold it there.

        The statement on *line*, where it binds the name *spelled* -- an unpacking's slot, a loop's target: that
        binding holds it -- or reads *spelled* itself: a name read there is held by its bindings reaching it, an
        expression by the names the statement binds whole to it.  Else the one binding reaching it of a local it
        reads, where that binding binds the local whole to *spelled* -- the binding the branch was spelled through
        (:func:`pathcond.reaching_values`).  The text is matched within that one statement, where two expressions
        alike read the same bindings.  None where neither."""
        key = (spelled, line)
        if key not in self._reads:
            self._reads[key] = self._read_at(spelled, line)
        return self._reads[key]

    def _read_at(self, spelled: str, line: Optional[int]) -> Optional[_Read]:
        if self._lines is None:
            self._lines = {}
            for node in walk(self.func):
                if isinstance(node, ast.stmt):
                    self._lines.setdefault(node.lineno, node)
        statement = self._lines.get(line) if line is not None else None
        if statement is None:
            return None
        expressions = _own_expressions(statement)
        named = spelled.isidentifier()
        if named and any(one.site is statement and not one.nested for one in bindings(self.func, spelled)):
            return _Read(statement, {spelled: (statement,)})
        if named and any(isinstance(one, ast.Name) and one.id == spelled for one in expressions):
            return _Read(statement, {spelled: self.sites(spelled, statement)})
        if not named and any(ast.unparse(one) == spelled for one in expressions):
            value = written_value(statement)
            whole = value is not None and ast.unparse(value) == spelled
            targets = [one.id for one in targets_of(statement) if isinstance(one, ast.Name)] if whole else []
            return _Read(statement, dict.fromkeys(targets, (statement,)))
        for name in sorted({one.id for one in expressions if isinstance(one, ast.Name)}):
            reached = reaching_values(self.func, name, self.settle, statement)
            if (
                len(reached) == 1
                and not reached[0].values
                and reached[0].expression is not None
                and reached[0].site is not None
                and ast.unparse(reached[0].expression) == spelled
            ):
                site = reached[0].site
                holders: dict[str, tuple[ast.AST, ...]] = {name: (site,)}
                if named:
                    holders[spelled] = self.sites(spelled, site)
                return _Read(site, holders)
        return None


def _about(branch, named: set[str], names_value: bool, reports: str) -> bool:
    """Whether a line is about *branch*, given what it names.

    The fourth reading, and the one the others cannot make.  Six arms of
    ``setScoutJobData_JEDI`` write ``exhausted`` and none of their lines
    interpolates the value -- they say ``action=set_exhausted reason=
    scout_cpuTime`` and let the tag carry the meaning -- so ``_names_the_value``
    finds nothing and the deciding junction ends up with no line to ask
    production for at all.

    **The branch's tags decide, not the line's.**  A tagged branch takes only a
    line covering its tags, which is what keeps the six arms apart; an untagged
    branch is judged the way it always was.  Reading it the other way round --
    a tagged *line* belongs only to a tagged branch -- cost three branches of
    ``doActionForReassign`` their line, because ``#ATM #KV label=managed
    action=trigger_new_brokerage by setting task_status={}`` does both at once
    and those branches carry no tag of their own (the block settles two
    subjects, so the tag reading declines to say which).

    A row count is about the write landing rather than about which branch chose
    it, so it is exempt.
    """
    if reports == REPORTS_ROWS_CHANGED:
        return True
    if branch.tags:
        return set(branch.tags) <= named
    return names_value


def attach(
    fragment,
    modules: list[SourceModule],
    attributor: SpecAttributor,
) -> int:
    """Give each branch the line the code writes when it fires.  Returns how many.

    Runs after :func:`logfile.attach`, which is what resolves a module's
    loggers -- and the file an emit lands in is the one of the module that
    writes the line, not the one of the node it is about.
    """
    declared = logfile.declared_files(modules)
    inherited = logfile.inherited_files(modules, declared)
    settle = values.resolver(sql._CALLS)

    by_owner: dict[str, list[JunctionNode]] = {}
    for junction in fragment.junctions:
        by_owner.setdefault(junction.owner, []).append(junction)

    attached = 0
    for module in modules:
        attach_parents(module.tree)
        files = logfile.files_of(module.rel_path, declared, inherited)
        for func, owner in functions_with_owner(module.tree):
            here = by_owner.get(f"{module.rel_path}::{def_name(func)}")
            if not here:
                continue
            logged = _logged_arguments(func)
            if not logged:
                continue
            counts = _row_counts(func)
            held = _Held(func, settle)
            for junction in here:
                spec_class, _, attribute = junction.subject.rpartition(".")
                for at_call, message in logged:
                    template = values.rendered_text(message)
                    named = values.decision_tags(template) if template else set()
                    names_value = _names_the_value(
                        message, func, owner, attributor, spec_class, attribute,
                        junction.branches, settle, held,
                    )
                    if _reports_rows(message, counts):
                        reports = REPORTS_ROWS_CHANGED
                    elif named or names_value:
                        reports = REPORTS_DECISION
                    else:
                        continue
                    if not template or not values.has_literal_text(template):
                        # A frame with no literal text is not something a
                        # production line can be matched on -- it says only
                        # that something was interpolated.
                        continue
                    emit = Emit(
                        template=template,
                        log_level=log_level(at_call),
                        log_files=files,
                        reports=reports,
                    )
                    where = path_condition(at_call)
                    for branch in junction.branches:
                        if not _about(branch, named, names_value, reports):
                            continue
                        # Read as written, not as the function runs: the
                        # branch's conditions are where its value is decided,
                        # not where it is written.  ``prepareTasksToBeFinished_JEDI``
                        # decides the status under one ``not toSkip`` and logs
                        # the unlock that replaces the write under the ``else``
                        # of a later one; read as run, the two meet, and the
                        # line saying the write did not happen went to the branch.
                        if exclusive(branch.path_condition, where):
                            continue
                        if any(e.template == template for e in branch.emits):
                            continue
                        branch.emits.append(emit)
                        attached += 1
    return attached
