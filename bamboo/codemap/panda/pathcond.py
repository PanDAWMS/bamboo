"""Where a write sits in the code: its enclosing scope and its path condition.

Shared by the recognizers rather than owned by one of them.  Both the attribute
slice and the SQL slice need the same two answers -- which function contains
this write, and which conditions had to hold to reach it -- and a recognizer
importing another recognizer to get them is the kind of dependency that turns
into a cycle the first time either grows.

**A path condition records what decides, not merely that something decided.**
An ``else`` contributes the negation of its ``if``, and a condition written as
a bare local name carries the expression that produced it -- otherwise a branch
guarded by ``if not allowed:`` reads as "some condition held", which cannot be
checked against anything.  PanDA moved a command's acceptance test into exactly
such a helper between two releases.
"""

from __future__ import annotations

import ast
from typing import Callable, Iterator, NamedTuple, Optional, Sequence


def functions_with_owner(
    node: ast.AST, owner: Optional[str] = None
) -> Iterator[tuple[ast.FunctionDef | ast.AsyncFunctionDef, Optional[str]]]:
    """Yield every function in *node* paired with the class enclosing it.

    Carried down the walk rather than read back from a ``parent`` link, so
    this works on a bare tree -- the element-type pass runs before the
    recognizer attaches parents.
    """
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.ClassDef):
            yield from functions_with_owner(child, child.name)
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield child, owner
            yield from functions_with_owner(child, owner)


def attach_parents(tree: ast.AST) -> None:
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child.parent = parent  # type: ignore[attr-defined]


def ancestors(node: ast.AST) -> Iterator[ast.AST]:
    current = getattr(node, "parent", None)
    while current is not None:
        yield current
        current = getattr(current, "parent", None)


def enclosing_function(node: ast.AST) -> Optional[ast.FunctionDef | ast.AsyncFunctionDef]:
    for ancestor in ancestors(node):
        if isinstance(ancestor, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return ancestor
    return None


def targets_of(node: ast.AST) -> list[ast.expr]:
    """What an ``=`` statement writes to, or ``[]`` if *node* is not one.

    ``x: dict[str, Any] = {...}`` is the same write as ``x = {...}`` and the
    corpus spells it that way 914 times in a function body, against five in the
    release the map was last built from.  Reading only :class:`ast.Assign` cost
    the SQL slice eleven facts -- three tables their reader, three values their
    selector, a criterion, two junctions and the whole of
    ``harvester_workers.pilotStatus`` -- and left one statement's table
    unresolved.  None of that was a change in PanDA: both versions of
    ``getDispatchDatasetsPerUser`` are identical but for the annotation.

    Here rather than in one recognizer because the same reading was found and
    fixed three separate times, each time at the single site that had just been
    caught.  A reader asking "is this an assignment" should get one answer.

    An ``AnnAssign`` with no value -- ``found: list[str]`` -- states a type and
    writes nothing, so it has no targets here: reading one as a write would have
    the name hold whatever the *next* statement assigns.

    ``AugAssign`` is deliberately absent.  Every caller that wants ``+=`` treats
    it as a different operator, and folding it in here would make ``sql +=
    " AND x=1"`` look like the statement rather than a fragment of it.
    """
    if isinstance(node, ast.Assign):
        return node.targets
    if isinstance(node, ast.AnnAssign) and node.value is not None:
        return [node.target]
    return []


def written_value(node: ast.AST) -> Optional[ast.expr]:
    """What an assignment writes, across all three spellings of one.

    Unlike :func:`targets_of` this does fold in ``AugAssign``, because its
    callers are reassembling a statement from the ``=`` and ``+=`` fragments
    that build it and need the right-hand side of both.
    """
    if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
        return node.value
    return None


def single_definition(
    func: ast.FunctionDef | ast.AsyncFunctionDef, name: str
) -> Optional[ast.expr]:
    """Return the expression assigned to *name*, when exactly one assigns it.

    A condition written as a bare local name says nothing on its own: ``if not
    allowed:`` names no predicate.  Substituting the single expression that
    produced it recovers which check decides -- typically a call such as
    ``self._check_command_allowed(...)``, whose own branches are a deeper
    expansion than this slice attempts.

    Returns ``None`` when several statements assign the name.  That is the
    fan-out case (a flag set from many places), where one expression would
    misrepresent the branch rather than explain it.

    Counting the annotated spelling is what makes that refusal honest.  The
    corpus's habit is to declare the initial value -- ``decoded = None``,
    ``res: Sequence[Any] = self.cur.fetchone()`` -- and overwrite it under a
    condition, so a reading blind to the annotation sees the *overwrite* alone
    and calls it the single definition.  Ten guards in the corpus were annotated
    that way, and nine of them named a statement that runs after the test or
    inside the test's own ``else``: ``if res:`` came out as ``res  [res := []]``,
    where ``res = []`` is what the failing branch does about it.  Withdrawing
    those is the contract, not a loss.  Six hundred and twenty-two guards gain a
    definition in exchange.

    The node rather than its text, because the other caller resolves what the
    expression *evaluates to* -- a local holding a declared mapping, which needs
    the tree.
    """
    found = _index(func).definitions().get(name, ())
    return found[0] if len(found) == 1 else None


class _Index:
    """What one pass over a scope finds, kept on the scope and asked many times.

    Each question used to walk the whole scope: the walk resolves a name at
    every call it classifies, and ``insertFilesForDataset_JEDI`` is 1100
    lines, so one arm spent 26 of 91 profiled seconds in
    :func:`single_definition` alone.  A parsed tree is not changed after it is
    read, so each pass is made once.  The entries are tuples and every caller
    gets its own list, so a reader that edits what it was given edits only that.

    Separate passes rather than one, because the questions do not walk the
    same nodes: :func:`single_definition` and :func:`assigned_expressions`
    count a nested function's writes and :func:`attribute_expressions` does not.
    """

    __slots__ = ("scope", "_definitions", "_assigned", "_attributes", "_bindings")

    def __init__(self, scope: ast.AST) -> None:
        self.scope = scope
        self._definitions: Optional[dict[str, tuple[ast.expr, ...]]] = None
        self._assigned: Optional[dict[str, tuple[ast.expr, ...]]] = None
        self._attributes: Optional[dict[str, tuple[ast.expr, ...]]] = None
        self._bindings: Optional[dict[str, tuple[Binding, ...]]] = None

    def definitions(self) -> dict[str, tuple[ast.expr, ...]]:
        """name -> the values an ``=`` writes to it, an unpacked element included."""
        if self._definitions is None:
            found: dict[str, list[ast.expr]] = {}
            for node in ast.walk(self.scope):
                value = written_value(node)
                if value is None:
                    continue
                for target in targets_of(node):
                    for name, _slot, _unpacked in names_bound_by(target):
                        found.setdefault(name, []).append(value)
            self._definitions = {name: tuple(values) for name, values in found.items()}
        return self._definitions

    def assigned(self) -> dict[str, tuple[ast.expr, ...]]:
        """name -> every value bound to it, ``+=`` included (:func:`assigned_expressions`)."""
        if self._assigned is None:
            found: dict[str, list[ast.expr]] = {}
            for node in ast.walk(self.scope):
                if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
                    names = [node.target]
                else:
                    names = [t for t in targets_of(node) if isinstance(t, ast.Name)]
                value = written_value(node)
                if value is None:
                    continue
                for name in names:
                    found.setdefault(name.id, []).append(value)
            self._assigned = {name: tuple(values) for name, values in found.items()}
        return self._assigned

    def attributes(self) -> dict[str, tuple[ast.expr, ...]]:
        """field -> every value bound to ``self.<field>``, nested functions not entered."""
        if self._attributes is None:
            found: dict[str, list[ast.expr]] = {}
            for node in _own_body(self.scope):
                if isinstance(node, ast.Assign):
                    targets = node.targets
                elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                    targets = [node.target]
                else:
                    continue
                for target in targets:
                    for bound in _bound_names(target):
                        if node.value is not None:
                            found.setdefault(bound, []).append(node.value)
            self._attributes = {field: tuple(values) for field, values in found.items()}
        return self._attributes

    def bindings(self) -> dict[str, tuple[Binding, ...]]:
        """name -> every :class:`Binding` of it in the scope, in ``ast.walk`` order."""
        if self._bindings is None:
            inner: set[int] = set()
            for node in ast.walk(self.scope):
                if node is not self.scope and isinstance(node, _SCOPES):
                    inner.update(id(held) for held in ast.walk(node) if held is not node)
            found: dict[str, list[Binding]] = {}
            for node in ast.walk(self.scope):
                for binding in _bindings_at(node, id(node) in inner):
                    found.setdefault(binding.name, []).append(binding)
            self._bindings = {name: tuple(held) for name, held in found.items()}
        return self._bindings


#: The nodes that open a scope of their own: a name bound inside one is not
#: the enclosing function's.
_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)

#: How a :class:`Binding` gives its name a value.
BIND_ASSIGN = "="
BIND_ANNOTATED = "annotated"
BIND_AUGMENTED = "augmented"
BIND_FOR = "for"
BIND_WITH = "with"
BIND_WALRUS = "walrus"
BIND_EXCEPT = "except"


class Binding(NamedTuple):
    """One place a scope gives a plain name a value.

    *site* is the statement that does it -- the ``for`` or ``with``, the
    handler for ``except ... as`` -- or the ``:=`` expression.  *value* is what
    the site writes as a whole: the right-hand side, the context expression of
    a ``with``, the walrus's value; ``None`` for a loop target or a caught
    exception, whose value no expression here spells.  An unpacked name gets
    the whole right-hand side and its *slot* in a one-level unpacking --
    ``None`` in a nested one or beside a ``*rest``, where the position is not
    the name's to say.  *nested* is whether the site sits in a function,
    lambda or class inside the scope, which binds a name of its own.

    The one reading of the syntax.  The walk asked six questions of it, each
    with its own traversal, and they disagreed: a nested tuple, a ``*rest``
    and a ``for (i, (x, j)) in`` target were bindings to the audit and not to
    the walk.  ``tests/test_codemap_bindings.py`` holds every reader to it.
    """

    name: str
    site: ast.AST
    kind: str
    value: Optional[ast.expr]
    slot: Optional[int]
    unpacked: bool
    nested: bool


def names_bound_by(target: ast.AST) -> list[tuple[str, Optional[int], bool]]:
    """``(name, slot, unpacked)`` for every plain name an assignment target binds.

    How a target unpacks, for every reader that takes one apart: a name, a
    tuple or list of targets to any depth, a ``*rest``.  The slot is the
    name's position in a one-level unpacking, ``None`` where a ``*rest`` or a
    nested unpacking leaves it unsaid.  Attribute and subscript targets bind
    no name.
    """
    if isinstance(target, ast.Name):
        return [(target.id, None, False)]
    if isinstance(target, ast.Starred):
        return [(name, None, True) for name, _slot, _unpacked in names_bound_by(target.value)]
    if not isinstance(target, (ast.Tuple, ast.List)):
        return []
    starred = any(isinstance(element, ast.Starred) for element in target.elts)
    found: list[tuple[str, Optional[int], bool]] = []
    for slot, element in enumerate(target.elts):
        if isinstance(element, ast.Name):
            found.append((element.id, None if starred else slot, True))
        else:
            found.extend((name, None, True) for name, _slot, _unpacked in names_bound_by(element))
    return found


def containers_changed(target: ast.AST) -> list[str]:
    """The names an assignment target changes in place: ``x`` for ``x[k] = v`` and ``x[k][j] = v``.

    Down the chain of subscripts only.  ``d[e[k]] = v`` reads ``e`` and
    writes ``d``; walking the whole target counted ``bw_map`` as written in
    ``site_class_n_site_dict[bw_map['class']] += 1``.
    """
    if isinstance(target, (ast.Tuple, ast.List)):
        return [name for element in target.elts for name in containers_changed(element)]
    if isinstance(target, ast.Starred):
        return containers_changed(target.value)
    if not isinstance(target, ast.Subscript):
        return []
    base = target.value
    while isinstance(base, ast.Subscript):
        base = base.value
    return [base.id] if isinstance(base, ast.Name) else []


def _bindings_at(node: ast.AST, nested: bool) -> list[Binding]:
    """The :class:`Binding` s *node* itself makes."""
    found: list[Binding] = []

    def add(target: ast.AST, kind: str, value: Optional[ast.expr], site: ast.AST = node) -> None:
        for name, slot, unpacked in names_bound_by(target):
            found.append(Binding(name, site, kind, value, slot, unpacked, nested))

    if isinstance(node, ast.Assign):
        for target in node.targets:
            add(target, BIND_ASSIGN, node.value)
    elif isinstance(node, ast.AnnAssign) and node.value is not None:
        add(node.target, BIND_ANNOTATED, node.value)
    elif isinstance(node, ast.AugAssign):
        add(node.target, BIND_AUGMENTED, node.value)
    elif isinstance(node, (ast.For, ast.AsyncFor)):
        add(node.target, BIND_FOR, None)
    elif isinstance(node, (ast.With, ast.AsyncWith)):
        for item in node.items:
            if item.optional_vars is not None:
                add(item.optional_vars, BIND_WITH, item.context_expr)
    elif isinstance(node, ast.NamedExpr):
        add(node.target, BIND_WALRUS, node.value)
    elif isinstance(node, ast.ExceptHandler) and node.name:
        found.append(Binding(node.name, node, BIND_EXCEPT, None, None, False, nested))
    return found


def bindings(scope: ast.AST, name: str) -> list[Binding]:
    """Every :class:`Binding` of *name* in *scope*, nested scopes' included, in ``ast.walk`` order.

    Read once per scope and kept on it, as :class:`_Index` keeps its other
    answers: the walk asks this of every name it explains.
    """
    return list(_index(scope).bindings().get(name, ()))


def _index(scope: ast.AST) -> _Index:
    held = getattr(scope, "_pathcond_index", None)
    if held is None:
        held = _Index(scope)
        scope._pathcond_index = held  # type: ignore[attr-defined]
    return held


def assigned_expressions(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
) -> dict[str, list[ast.expr]]:
    """Every expression bound to each local name in *func*, ``+=`` included.

    The unrestricted sibling of :func:`single_definition`, for the readers that
    want *all* the values a name can hold rather than the one that explains a
    condition.  Both of them are the same hop: a message is written into a
    local and logged or persisted a few lines later, and following it is what
    the emit pass needs to find ``set task_status=`` at all and what the tag
    reading needs to find the reason a branch names for itself.

    An augmented assignment contributes its own right-hand side rather than the
    concatenation.  ``errMsg += ...`` under a condition may not run, so the
    pieces are what the reading establishes and the whole is not.

    Only names, still: an attribute target belongs to :func:`attribute_expressions`
    and putting one here would attach a spec field's history to a local's.  But
    a name is a name however it is declared, and ``error_message: str | None =
    f"method {method_name} is forbidden"`` is the write the emit pass has to
    follow to give the 403 any text at all.
    """
    return {name: list(values) for name, values in _index(func).assigned().items()}


def attribute_expressions(
    scope: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef, field: str
) -> list[ast.expr]:
    """Every expression bound to ``self.<field>`` inside *scope*.

    The ``ast.Attribute`` sibling of :func:`assigned_expressions`, and a
    separate function rather than a widening of it.  That one is shared with
    the build, where its targets being bare names is load-bearing: the write
    slice reads it to find the value a bind was filled from, and an attribute
    target there would attach a spec field's history to a local's.  Widening
    it would move the map; this is read at use time only.

    Scoped to one field rather than returning a mapping, because the caller
    arrives with the field in hand and the classes this is asked about carry
    hundreds of attributes between them -- ``JobSpec`` alone declares 126.

    Nested functions are not entered.  A closure assigning ``self.x`` runs
    when it is called and not where it is written, so its guards are not the
    ones that reached the arm; the walk would report a condition that never
    dominated the write.
    """
    return list(_index(scope).attributes().get(field, ()))


def _bound_names(target: ast.AST) -> list[str]:
    """The ``self.<field>`` names an assignment target binds."""
    if (
        isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
    ):
        return [target.attr]
    if isinstance(target, (ast.Tuple, ast.List)):
        return [name for element in target.elts for name in _bound_names(element)]
    return []


def _own_body(scope: ast.AST) -> Iterator[ast.AST]:
    """Every node under *scope* that belongs to it, nested functions excluded."""
    for child in ast.iter_child_nodes(scope):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        yield child
        yield from _own_body(child)


def methods_of(cls: ast.ClassDef) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    """The methods *cls* declares, in source order."""
    return [
        node
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def class_of(tree: ast.AST, func: ast.AST) -> Optional[ast.ClassDef]:
    """The class *func* is a method of, without needing parent links.

    ``attribution`` answers the same question by walking ``parent`` back, which
    it can because the recognizer attaches them.  The walk parses files of its
    own at use time and has no such guarantee, so this descends instead -- the
    same rule, read from the other end.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and any(
            method is func for method in methods_of(node)
        ):
            return node
    return None


def bases_in(tree: ast.AST, cls: ast.ClassDef) -> list[ast.ClassDef]:
    """*cls*'s bases that this module declares, nearest first.

    The same order ``attribution._class_and_ancestors`` produces, cut where
    this reading's evidence stops: it has one parsed module and no corpus
    index, so a base declared elsewhere is not followed.  Measured before the
    cut was made -- of 292 classes in the corpus, 8 have all their bases in
    their own module and 141 have at least one elsewhere, and the 83 steps
    this whole reading exists for need none of them.  The limit is real and
    it is not in the way.
    """
    declared = {
        node.name: node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
    }
    order: list[ast.ClassDef] = []
    seen = {cls.name}
    queue = [_base_name(base) for base in cls.bases]
    while queue:
        name = queue.pop(0)
        if not name or name in seen:
            continue
        seen.add(name)
        node = declared.get(name)
        if node is None:
            continue
        order.append(node)
        queue.extend(_base_name(base) for base in node.bases)
    return order


def _base_name(base: ast.expr) -> Optional[str]:
    if isinstance(base, ast.Name):
        return base.id
    if isinstance(base, ast.Attribute):
        return base.attr
    return None


def path_condition(node: ast.AST) -> list[str]:
    """Return the conjunction of tests dominating *node*, outermost first.

    Walks the ancestor chain rather than the call stack: an ``else`` branch
    contributes ``not <test>``, because "the condition did not hold" is as much
    a reason for the outcome as the condition holding.
    """
    conditions: list[str] = []
    func = enclosing_function(node)
    previous = node
    for ancestor in ancestors(node):
        if isinstance(ancestor, ast.If):
            try:
                test = ast.unparse(ancestor.test)
            except Exception:  # noqa: BLE001
                previous = ancestor
                continue
            if previous in ancestor.orelse:
                test = f"not ({test})"
            elif previous not in ancestor.body:
                previous = ancestor
                continue
            if func is not None:
                test = _substitute_bare_name(test, ancestor.test, func)
            conditions.append(test)
        previous = ancestor
    conditions.reverse()
    return conditions


#: What :func:`enclosing_guards` found around a site.  Each kind is a way the
#: code constrains reaching a line that ``path_condition`` is structurally
#: blind to, because only ``ast.If`` contributes to a path condition.
UNSEEN_TRY = "try"
UNSEEN_EXCEPT = "except"
UNSEEN_LOOP = "loop"
UNSEEN_WITH = "with"
UNSEEN_EARLY_EXIT = "early-exit"


class Unseen(NamedTuple):
    """One constraint on reaching a line that its path condition omits."""

    kind: str
    detail: str
    line: int


def _text(node: Optional[ast.AST]) -> str:
    if node is None:
        return "anything"
    try:
        return ast.unparse(node)
    except Exception:  # noqa: BLE001 -- unparse fails on synthesised nodes
        return "..."


def _jumps(body: list[ast.stmt]) -> bool:
    """Whether entering *body* means leaving the block it is in.

    The last statement, not any statement: control reaching the end of a block
    has run it, so a trailing ``continue`` is unconditional for anyone who
    entered.  Read from the tree rather than from a flow graph, which is the
    whole reason this is cheap enough to be sound.
    """
    return bool(body) and isinstance(
        body[-1], (ast.Continue, ast.Break, ast.Return, ast.Raise)
    )


def _implied_by(statement: ast.stmt) -> Optional[str]:
    """The condition a preceding ``if ...: continue`` puts on what follows it."""
    if not isinstance(statement, ast.If):
        return None
    test = _text(statement.test)
    if test == "...":
        return None
    if _jumps(statement.body):
        return f"not ({test})"
    if statement.orelse and _jumps(statement.orelse):
        return test
    return None


def _early_exits(ancestor: ast.AST, previous: ast.AST) -> list[Unseen]:
    for field in ("body", "orelse", "finalbody"):
        block = getattr(ancestor, field, None)
        if not isinstance(block, list) or previous not in block:
            continue
        return [
            Unseen(UNSEEN_EARLY_EXIT, implied, statement.lineno)
            for statement in block[: block.index(previous)]
            if (implied := _implied_by(statement))
        ]
    return []


def _structure(ancestor: ast.AST, previous: ast.AST) -> list[Unseen]:
    if isinstance(ancestor, ast.Try) and previous in ancestor.body and ancestor.handlers:
        caught = ", ".join(_text(handler.type) for handler in ancestor.handlers)
        return [Unseen(UNSEEN_TRY, f"nothing above raised {caught}", ancestor.lineno)]
    if isinstance(ancestor, ast.ExceptHandler):
        return [
            Unseen(
                UNSEEN_EXCEPT,
                f"something above raised {_text(ancestor.type)}",
                ancestor.lineno,
            )
        ]
    if isinstance(ancestor, (ast.For, ast.AsyncFor)) and previous in ancestor.body:
        target, over = _text(ancestor.target), _text(ancestor.iter)
        return [Unseen(UNSEEN_LOOP, f"for {target} in {over}", ancestor.lineno)]
    if isinstance(ancestor, ast.While) and previous in ancestor.body:
        return [Unseen(UNSEEN_LOOP, f"while {_text(ancestor.test)}", ancestor.lineno)]
    if isinstance(ancestor, (ast.With, ast.AsyncWith)):
        held = ", ".join(_text(item.context_expr) for item in ancestor.items)
        return [Unseen(UNSEEN_WITH, f"with {held}", ancestor.lineno)]
    return []


def enclosing_guards(node: ast.AST) -> list[Unseen]:
    """What had to hold to reach *node* that its path condition cannot say.

    **A path condition is a necessary condition, not a sufficient one**, and
    this is the part that makes the difference nameable.  Only ``ast.If``
    contributes to :func:`path_condition`, so a site inside a ``try`` body, a
    handler, a loop or a ``with`` carries whatever tests are above it and
    nothing about the structure itself.  Measured on the installed corpus, of
    18018 reaching-definition sites 10619 are in a loop, 6515 in a ``try`` body
    and 433 in a handler -- and 3058, 2180 and 111 of those have an *empty*
    path condition, which reads as "unconditional" and is not.  On the map's
    own arms the exposure is higher still: 86% have at least one of these on
    the way to them.

    Four of the five kinds come from the ancestor chain.  The fifth is the
    early exit -- ``if X: continue`` above the site in the same block -- which
    is sound without any dataflow, because statements in a block run in order
    and a trailing jump is unconditional for whoever entered it.  14% of the
    map's arms have one on their walk, which is not thin enough to leave to a
    report.

    **A ``try`` body names its handlers, not the statements before it.**  Both
    are true constraints, but the handler set is bounded and usually logs,
    where the preceding statements have a median of twelve and a maximum of
    135 -- a list that long is noise, and it is the handler that says which
    external call could have diverted control here.

    Nothing above the enclosing function is read: module-scope statements run
    at import, so counting them as preceding would report a reason that never
    applied to this call.
    """
    found: list[Unseen] = []
    previous = node
    for ancestor in ancestors(node):
        found.extend(_early_exits(ancestor, previous))
        if isinstance(
            ancestor, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)
        ):
            break
        found.extend(_structure(ancestor, previous))
        previous = ancestor
    found.sort(key=lambda entry: entry.line)
    return found


def own_test(node: ast.AST) -> Optional[str]:
    """The test of the innermost ``if``/``elif`` whose body contains *node*.

    Read from the tree rather than taken as the last entry of
    :func:`path_condition`, whose entries may carry the ``[name := expr]``
    annotation -- wrapping that in ``not (...)`` produces text nothing can read
    back.
    """
    previous = node
    for ancestor in ancestors(node):
        if isinstance(ancestor, ast.If) and previous in ancestor.body:
            try:
                return ast.unparse(ancestor.test)
            except Exception:  # noqa: BLE001
                return None
        previous = ancestor
    return None


def exclusive(one: list[str], other: list[str]) -> bool:
    """Whether two path conditions cannot both hold.

    Detected from the negations :func:`path_condition` already writes down: an
    ``elif`` branch carries ``not (<the test before it>)``, so two branches of
    one chain each hold a negation of something the other asserts.

    Deliberately conservative -- an annotated test will not match textually, so
    some exclusive pairs read as compatible.  That is the safe direction for
    both callers.  Deciding whether a later write overwrites an earlier one, a
    missed exclusion adds a condition that is true but redundant where a missed
    *overlap* would drop a required one; splitting a SQL statement built across
    branches, a missed exclusion leaves the statement folded as it was before,
    where a missed overlap would split a statement that is really one.
    """
    return any(f"not ({test})" in other for test in one) or any(
        f"not ({test})" in one for test in other
    )


#: A test the constants at hand do not settle.
UNDECIDED = object()


def decided(test: ast.expr, values: dict[str, object]) -> object:
    """``True`` or ``False`` where *values* settle *test*, else :data:`UNDECIDED`.

    A name's truth, ``not``, ``and``/``or``, and one comparison of a name
    with a literal -- the shapes a flag parameter is tested in.
    """
    if isinstance(test, ast.Name):
        return bool(values[test.id]) if test.id in values else UNDECIDED
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        inner = decided(test.operand, values)
        return inner if inner is UNDECIDED else not inner
    if isinstance(test, ast.BoolOp):
        results = [decided(value, values) for value in test.values]
        stop = isinstance(test.op, ast.Or)
        if any(result is stop for result in results):
            return stop
        return (not stop) if all(result is (not stop) for result in results) else UNDECIDED
    if (
        isinstance(test, ast.Compare)
        and len(test.ops) == 1
        and isinstance(test.left, ast.Name)
        and test.left.id in values
    ):
        left = values[test.left.id]
        try:
            right = ast.literal_eval(test.comparators[0])
        except (ValueError, SyntaxError, TypeError):
            return UNDECIDED
        compare = {
            ast.Is: lambda a, b: a is b,
            ast.IsNot: lambda a, b: a is not b,
            ast.Eq: lambda a, b: a == b,
            ast.NotEq: lambda a, b: a != b,
            ast.In: lambda a, b: a in b,
            ast.NotIn: lambda a, b: a not in b,
        }.get(type(test.ops[0]))
        if compare is None:
            return UNDECIDED
        try:
            return bool(compare(left, right))
        except TypeError:
            return UNDECIDED
    return UNDECIDED


def ruled_out_by(node: ast.AST, func: ast.AST, values: dict[str, object]) -> bool:
    """Whether *func* never runs *node* when its parameters hold *values*.

    Only *func*'s own ``if`` statements are ruled on: a nested function or
    lambda has parameters of its own, whatever they are called.
    """
    if enclosing_function(node) is not func:
        return False
    child = node
    for ancestor in ancestors(node):
        if isinstance(ancestor, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            break
        if isinstance(ancestor, ast.If):
            result = decided(ancestor.test, values)
            if result is False and any(child is one for one in ancestor.body):
                return True
            if result is True and any(child is one for one in ancestor.orelse):
                return True
        child = ancestor
    return False


# ---------------------------------------------------------------------------
# Whether what one site writes can be what another site reads
# ---------------------------------------------------------------------------
#
# Written for the walk, which offers a reader every binding of a name it could
# not value, and wrong there in one direction only: a site below the arm never
# ran before it.  The build asks the same question of a different pair -- can
# the fragments that built a statement reach the call that runs it -- and was
# wrong in the same direction, attributing one statement to every call site in
# the function.
#
# Here rather than in either caller because the build must not import the
# walk: one makes the stored map and one reads it, and the separation is what
# keeps a use-time convenience out of the stored facts.  One spelling, so the
# two cannot drift into disagreeing about which writes a read can see.


def site_of(expression: ast.AST) -> ast.AST:
    """The statement *expression* is part of, or the expression itself.

    :func:`assigned_expressions` hands back the right-hand side, and everything
    positional -- the line, the guards above it, whether a loop holds it -- is
    a property of the statement it sits in.
    """
    node: Optional[ast.AST] = expression
    while node is not None and not isinstance(node, ast.stmt):
        node = getattr(node, "parent", None)
    return node if node is not None else expression


def loops_over(node: ast.AST) -> set[int]:
    """The header line of every loop whose *body* holds *node*."""
    found: set[int] = set()
    previous = node
    for ancestor in ancestors(node):
        if isinstance(
            ancestor, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)
        ):
            break
        if isinstance(ancestor, (ast.For, ast.AsyncFor, ast.While)) and previous in ancestor.body:
            found.add(ancestor.lineno)
        previous = ancestor
    return found


def handlers_over(node: ast.AST) -> set[int]:
    """The header line of every ``except`` handler whose *body* holds *node*.

    :func:`loops_over`'s counterpart for the other construct a path condition
    cannot see.  :func:`reachable_when` renders ``ast.If`` tests and nothing
    else, so an assignment in a handler reads as one that always ran -- but a
    handler runs only when something below the ``try`` raised, which is a
    condition no ``if`` in the function states.
    """
    found: set[int] = set()
    previous = node
    for ancestor in ancestors(node):
        if isinstance(
            ancestor, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)
        ):
            break
        if isinstance(ancestor, ast.ExceptHandler) and previous in ancestor.body:
            found.add(ancestor.lineno)
        previous = ancestor
    return found


def not_below(site: ast.AST, target: ast.AST) -> bool:
    """Whether *site* could have run before *target* on one pass.

    A write below the read cannot have run before it -- unless a loop holds
    them both, where the next iteration reaches it.  That second half is not a
    refinement: without it the rule is unsound, and a message assigned at the
    foot of a loop body and logged at its head is dropped although every
    iteration but the first reaches it.
    """
    return getattr(site, "lineno", 0) <= getattr(target, "lineno", 0) or bool(
        loops_over(site) & loops_over(target)
    )


def reaches_one(site: ast.AST, targets: Sequence[ast.AST]) -> bool:
    """Whether *site* is positioned to reach any of *targets*."""
    return any(not_below(site, target) for target in targets)


def reachable_when(node: ast.AST) -> list[str]:
    """The conditions on reaching *node*, including the ones an ``if`` cannot say.

    :func:`path_condition` sees only ``ast.If``, so an ``if X: continue`` above
    a site leaves the site reading as unconditional.  That blindness is not
    harmless here: a log call inside the skipped block and a write after it are
    on paths that cannot both run, and comparing their path conditions alone
    says they are compatible.  The early exit puts ``not (X)`` on the write and
    the block puts ``X`` on the call, which is exactly the shape
    :func:`exclusive` was built to detect.
    """
    return path_condition(node) + [
        unseen.detail
        for unseen in enclosing_guards(node)
        if unseen.kind == UNSEEN_EARLY_EXIT
    ]


def _names_in(expression: ast.AST) -> set[str]:
    return {node.id for node in ast.walk(expression) if isinstance(node, ast.Name)}


def _rebinds(func: ast.AST) -> dict[str, list[ast.AST]]:
    """``{name: every node in *func* that gives it a new value}``, found once per function.

    Every :class:`Binding` outside a nested scope, and the two ways a
    container changes in place -- ``x[k] = v`` and ``x.add(v)`` -- because a
    test on ``x`` read before any of those described a different ``x``.  Wide
    on purpose: the one reader of this, :func:`parted`, reads a contradiction
    away for it, and that only makes two sites look more able to meet.
    """
    held = getattr(func, "_rebinds", None)
    if held is not None:
        return held
    found: dict[str, list[ast.AST]] = {}
    for name, sites in _index(func).bindings().items():
        own = [binding.site for binding in sites if not binding.nested]
        if own:
            found[name] = own
    for node in ast.walk(func):
        if node is func or enclosing_function(node) is not func:
            continue
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                for name in containers_changed(target):
                    found.setdefault(name, []).append(node)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
        ):
            found.setdefault(node.func.value.id, []).append(node)
    try:
        func._rebinds = found  # type: ignore[attr-defined]
    except AttributeError:
        pass
    return found


def _dominating_ifs(node: ast.AST) -> list[ast.If]:
    """The ``if`` statements :func:`path_condition` reads a condition from, outermost first."""
    found: list[ast.If] = []
    previous = node
    for ancestor in ancestors(node):
        if isinstance(ancestor, ast.If) and (
            previous in ancestor.body or previous in ancestor.orelse
        ):
            found.append(ancestor)
        previous = ancestor
    found.reverse()
    return found


def _early_exit_ifs(node: ast.AST) -> list[tuple[str, ast.stmt]]:
    """:func:`enclosing_guards`' early exits, each with the statement that implies it."""
    found: list[tuple[str, ast.stmt]] = []
    previous = node
    for ancestor in ancestors(node):
        for field in ("body", "orelse", "finalbody"):
            block = getattr(ancestor, field, None)
            if not isinstance(block, list) or previous not in block:
                continue
            found.extend(
                (implied, statement)
                for statement in block[: block.index(previous)]
                if (implied := _implied_by(statement))
            )
            break
        if isinstance(
            ancestor, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)
        ):
            break
        previous = ancestor
    return found


def _tested_by(node: ast.AST, exits: bool) -> list[tuple[str, Optional[ast.stmt]]]:
    """*node*'s conditions, each with the statement whose test it renders.

    :func:`path_condition`'s, and with *exits* :func:`reachable_when`'s early
    exits too.  A condition ``path_condition`` could not pair with its ``if``
    -- it skipped a test it could not render -- carries ``None``, and
    :func:`parted` then reads it as written.  Held on the node, since the
    reassembly asks it of every pair of a statement's fragments.
    """
    held: Optional[dict[bool, list[tuple[str, Optional[ast.stmt]]]]] = getattr(
        node, "_tested_by", None
    )
    if held is not None and exits in held:
        return held[exits]
    ifs = _dominating_ifs(node)
    written = path_condition(node)
    found: list[tuple[str, Optional[ast.stmt]]] = (
        list(zip(written, ifs, strict=True)) if len(ifs) == len(written) else [(text, None) for text in written]
    )
    if exits:
        found.extend(_early_exit_ifs(node))
    try:
        if held is None:
            held = {}
            node._tested_by = held  # type: ignore[attr-defined]
        held[exits] = found
    except AttributeError:
        pass
    return found


def _apart(one: ast.AST, other: ast.AST) -> bool:
    """Whether an ``if`` holds *one* in one arm and *other* in the other -- one pass runs one of them."""
    previous = one
    for ancestor in ancestors(one):
        if isinstance(
            ancestor,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef, ast.Module),
        ):
            break
        if isinstance(ancestor, ast.If):
            if previous in ancestor.body:
                arm = ancestor.orelse
            elif previous in ancestor.orelse:
                arm = ancestor.body
            else:
                arm = []
            if any(_within(other, statement) for statement in arm):
                return True
        previous = ancestor
    return False


def _given_between(
    func: ast.AST, earlier: ast.stmt, later: ast.stmt, ends: tuple[ast.AST, ...] = ()
) -> bool:
    """Whether a name *earlier*'s test reads is given a new value on the way to *later*'s test.

    On the way: below the end of *earlier*'s test and above *later*, and in no
    arm a pass through either test, or through either of *ends*, cannot enter.
    ``prepareTasksToBeFinished_JEDI`` tests ``not toSkip`` at 4044, sets
    ``toSkip = True`` at 4092 in the ``else`` of the ``if`` whose body holds
    4044, and tests ``not toSkip`` again at 4095: a pass that tested 4044 never
    ran 4092.
    """
    start = getattr(earlier.test, "end_lineno", None) or earlier.lineno  # type: ignore[attr-defined]
    rebinds = _rebinds(func)
    return any(
        start < site.lineno < later.lineno  # type: ignore[attr-defined]
        and not any(_apart(site, on) for on in (earlier, later, *ends))
        for name in _names_in(earlier.test)  # type: ignore[attr-defined]
        for site in rebinds.get(name, ())
    )


def parted(one: ast.AST, other: ast.AST, exits: bool = True) -> bool:
    """Whether *one* and *other* cannot both be on one run, read as the function runs.

    :func:`exclusive` compares conditions as text, and a condition describes
    its name as it was when its test ran.  ``AtlasProdWatchDog.doActionForReassign``
    skips a task with ``if tmpStat is not True: continue`` and, further down
    the same loop, sets ``tmpStat`` afresh and fails the task under ``if
    tmpStat is not True``.  Compared as text, everything below the skip was
    "not (tmpStat is not True)" and the failure "tmpStat is not True", so the
    two could never meet -- and a walk narrowed to what reaches ``if isOK``
    dropped the one ``isOK = False`` that says why a task was not reassigned.

    So a contradiction counts by the statements the two conditions come from.
    From one statement -- the two arms of an ``if`` -- it always holds: one
    execution of the ``if`` enters one arm, whatever either arm then assigns.
    From two statements testing the same thing, it holds unless a name the
    test reads is given a new value between them.  ``getTasksWithCriteria_JEDI``
    tests ``isinstance(tmpVal, list)`` in a loop over the task criteria and
    again in a loop over the dataset criteria, and the second ``for`` gives
    ``tmpVal`` its new value: a list among the task criteria and a scalar among
    the dataset criteria go into one statement, which text alone split apart.

    **Dropping the stale condition from each side was the wrong shape.**  The
    rule this replaces did that, and dropped too much for the build: the arms
    of one ``if`` lost each other as soon as either arm assigned the name or
    called a method on it (``elif workingGroup is not None:`` ...
    ``workingGroup.upper()``), and a multi-line test counted the calls inside
    it.  Over the corpus's reassembly, 98.5% of the "cannot meet" answers come
    from one statement's arms; that rule turned 334 of them into "can meet" and
    would have built statements no run sends.  Read by statement, 12 pairs
    change there, all in ``getTasksWithCriteria_JEDI``.

    *exits* reads :func:`reachable_when`'s early exits as well as the ``if``
    tests, as :func:`compatible` does; the reassembly reads the tests alone,
    as it always has.
    """
    func = enclosing_function(one)
    mine, theirs = _tested_by(one, exits), _tested_by(other, exits)
    for text, statement in mine:
        for other_text, other_statement in theirs:
            if other_text != f"not ({text})" and text != f"not ({other_text})":
                continue
            if statement is None or other_statement is None or statement is other_statement:
                return True
            earlier, later = sorted((statement, other_statement), key=lambda held: held.lineno)
            if func is None or not _given_between(func, earlier, later, (one, other)):
                return True
    return False


def compatible(one: ast.AST, other: ast.AST, current: bool = False) -> bool:
    """Whether both nodes can be on one run through the function.

    A path condition is a necessary condition, so this is conservative in the
    direction that is safe: it separates two sites only where one's condition
    contradicts the other's.

    *current* reads the function as it runs -- :func:`parted` -- rather than
    comparing each side's conditions as text.  Only for deciding whether two
    sites *can* meet.  Not for deciding whether one write *replaced* another,
    where a contradiction read away would make replacing easier -- the unsafe
    direction.
    """
    if current:
        return not parted(one, other)
    return not exclusive(reachable_when(one), reachable_when(other))


def can_reach(site: ast.AST, target: ast.AST, current: bool = False) -> bool:
    """Whether what *site* writes can be what *target* reads."""
    return not_below(site, target) and compatible(site, target, current)


def _within(node: ast.AST, container: ast.AST) -> bool:
    return node is container or any(ancestor is container for ancestor in ancestors(node))


def _skipped_by_a_raise(other: ast.AST, target: ast.AST, live: Sequence[ast.AST]) -> bool:
    """Whether an exception can carry control past *other*'s write and on to *target*.

    A write in a ``try`` body runs unless something above it in the body
    raised, and that is the path this asks about: a handler that neither
    leaves nor writes the name again sends control on with the value from
    before the ``try``.  ``TaskRefiner.runImpl`` sets ``dataset_did = None``,
    tries ``rucioAPI.get_did_str()``, passes on any exception and then tests
    ``if dataset_did is not None`` -- the ``None`` is what that test sees
    whenever Rucio failed, and reading the ``try``'s write as having replaced
    it dropped exactly that answer.

    A handler holding another of the *live* writes is taken to write the name,
    which is the case ``except: limit = 8`` below a ``try: limit = parse()``
    is: there the ``try``'s write does replace the one above it on every path
    that reaches the read.
    """
    previous = other
    for ancestor in ancestors(other):
        if isinstance(
            ancestor, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)
        ):
            break
        if (
            isinstance(ancestor, ast.Try)
            and previous in ancestor.body
            and not _within(target, ancestor)
        ):
            for handler in ancestor.handlers:
                if _jumps(handler.body):
                    continue
                if any(site is not other and _within(site, handler) for site in live):
                    continue
                return True
        previous = ancestor
    return False


def _cancels(
    site: ast.AST,
    other: ast.AST,
    target: ast.AST,
    rereads: Sequence[ast.AST] = (),
    live: Optional[Sequence[ast.AST]] = None,
) -> bool:
    """Whether *other* overwrites *site* before *target* reads it.

    Three things have to hold.  *other* sits between them, so it runs after the
    write it cancels; it is held by no loop *target* is outside of, or it would
    be an iteration's write rather than one on the way; and every condition it
    is under is one *target* is under too, so a path that reaches the read
    passed through it.

    The conditions are compared as the strings :func:`reachable_when` renders,
    which is why the third test is a subset rather than an implication.  Two
    conditions that mean the same thing and are written differently do not
    match, and the rule then keeps a binding it could have cancelled -- short
    of the truth in the direction that leaves the reader more to look at
    rather than less.  That holds for a condition the renderer spells
    differently; it does **not** hold for one it cannot see at all, which is
    why the ``except`` test below is a separate rule rather than a refinement.
    A condition that goes uncounted makes the rendered set *smaller*, the
    subset test easier to pass, and the cancelling too eager.

    An ``except`` body is such a condition.  It runs only when something
    raised, so a write in one cannot be assumed to have replaced the write
    above it::

        try:    nRanges = int(nRanges)     # 64
        except: nRanges = 8                # 66

    and the map said the request always asks for eight.  Measured over the
    corpus's read sites, 199 firings across 28 (name, write, killer) triples
    are of this shape -- ``objectstores`` read as always empty,
    ``retryJob``'s ``attemptNr`` as always ``-999``, and
    ``assigned_harvester_id`` as always unset.

    The killed write's own handlers are subtracted as well as the target's,
    and that is not a loosening: a write inside the handler is live only on a
    path where the handler ran, and on that path the later write in the same
    handler ran too.  ``getTasksToRefine_JEDI`` rebuilds ``varMap`` three
    times inside one ``except``, and without this the rule would hand a
    reader all three.  Subtracting only the target's would give up 51
    firings over 3 triples that are right.

    An ``ast.AugAssign`` cancels nothing, because it reads the value it
    replaces: ``timeOutVal *= 60`` needs whatever line 423 or 425 put there,
    so both are still answers to the read below it.  Treating it as an
    overwrite cost five of the corpus's thirty-seven narrowings a *wrong*
    answer rather than a short one -- ``copyArchive.py``'s holding limit read
    as sixty minutes instead of the configured value or forty-eight hours, and
    two SQL strings assembled with ``+=`` read as their last clause with the
    ``SELECT`` dropped.  The reverse stays: a later *plain* assignment does
    cancel an ``AugAssign``, since it needs nothing of what was there.

    The rule stops at ``AugAssign`` rather than at "reads the name it binds".
    ``x = f(x)`` reads it too and is still the whole answer once it has run;
    the difference is that :func:`assigned_expressions` returns the right-hand
    side alone, which reconstructs ``f(x)`` and not ``x * 60``.

    *rereads* are the other writes of that kind -- ``keyList.add(tmpKey)``,
    which the caller knows is a mutation and this function cannot tell from a
    statement.  A set after ``add`` still holds everything it held before, so
    ``copyArchive.py``'s ``if tmpKey not in keyList`` read the ``set()`` and
    the first loop's keys as well as the second's, and cancelling them left
    the test explained by its own loop alone.
    """
    if isinstance(other, ast.AugAssign) or any(other is one for one in rereads):
        return False
    # *live* is given where the caller reads the function as it runs -- see
    # :func:`_skipped_by_a_raise`.
    if live is not None and _skipped_by_a_raise(other, target, live):
        return False
    if not site.lineno < other.lineno <= getattr(target, "lineno", 0):
        return False
    if loops_over(other) - loops_over(target):
        return False
    if handlers_over(other) - handlers_over(target) - handlers_over(site):
        return False
    return set(reachable_when(other)) <= set(reachable_when(target))


def reaching_definitions(
    func: ast.FunctionDef | ast.AsyncFunctionDef, name: str, target: ast.AST
) -> list[ast.expr]:
    """The expressions bound to *name* that can still be what *target* reads.

    :func:`assigned_expressions` answers "everything this name ever holds",
    which is the right question for the build and the wrong one for a reader
    asking why *this* line saw what it saw.  ``copyArchive.py::main`` assigns
    ``timeLimit`` twenty-six times in 1353 lines; the statement on line 381
    reads the one on line 360 and no other, because lines 397 onwards had not
    run yet and 298, 321 and 341 were overwritten by 360 on the way.

    Two steps, and the second is what the first cannot do.  :func:`can_reach`
    drops the writes that are positioned below the read or on a path that
    excludes it -- five of the twenty-six survive that for line 381.  It does
    not drop a write that *did* run and was then replaced, because reaching
    the read is not the question there.  :func:`_cancels` is.

    Measured on the corpus by walking every read statement: of 62 (statement,
    bind name) pairs where the name has more than one assignment, 34 narrow
    and none is emptied.  An emptied name would mean the rule had cancelled
    the write the read actually saw, so nothing coming back is a signal that
    it is wrong rather than an answer -- but only that one.  It cannot see a
    narrowing that kept the *wrong* line, which is what ``AugAssign`` was
    doing to five of these pairs until the rule stopped treating it as an
    overwrite; that needs the lines themselves compared.

    Returns the expressions in source order -- ``assigned_expressions`` walks
    breadth-first, so a nested default comes back after the statement below
    it -- and a caller that wants positions takes :func:`site_of` as it
    already does for :func:`assigned_expressions`.
    """
    bound = assigned_expressions(func).get(name, [])
    kept = surviving_writes([site_of(value) for value in bound], target)
    alive = {id(site) for site in kept}
    return sorted(
        (value for value in bound if id(site_of(value)) in alive),
        key=lambda value: site_of(value).lineno,
    )


def surviving_writes(
    sites: Sequence[ast.AST],
    target: ast.AST,
    rereads: Sequence[ast.AST] = (),
    as_run: bool = False,
) -> list[ast.AST]:
    """Of *sites*, the writes that can still be what *target* reads.

    The body of :func:`reaching_definitions`, taken apart so that a caller
    holding writes from several shapes can ask about all of them at once.  The
    walk is that caller: a name arrives at a read through a plain assignment,
    a tuple unpacking, a ``with`` or a ``for`` target, and cancelling only
    within one of those shapes would keep a tuple-bound ``res`` alive against
    the assignment that replaced it.

    *rereads* are writes that keep what was there -- see :func:`_cancels`.

    *as_run* reads the function the way it runs rather than the way it is
    written, in the two places those differ: a test describes its name as it
    was when the test ran (:func:`parted`), and an exception can carry control
    past a ``try``'s write (:func:`_skipped_by_a_raise`).  The walk and the
    build both ask for it.
    """
    live = [site for site in sites if can_reach(site, target, as_run)]
    return [
        site
        for site in live
        if not any(
            _cancels(site, other, target, rereads, live if as_run else None)
            for other in live
            if other is not site
        )
    ]


def literal_values(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    name: str,
    resolve: Optional[Callable[[ast.expr, ast.AST], list[str]]] = None,
) -> list[tuple[str, list[str], int]]:
    """``(value, conditions, line)`` for every settled ``name = ...`` in *func*.

    The line is the assignment's, because that is where the value is decided --
    an anchor pointing at the use would send a reader to the place that merely
    passes it on.

    Reaching definitions for one local, which is what a value assigned to a
    variable before it is used needs -- the tag a broker interpolates into its
    rejection message, or the status a post-processor's helper returns.

    **Reassignment is the part dominating-guard analysis cannot see.**  The
    conditions on an assignment are necessary and, on their own, not sufficient:
    a later write reaching the same name replaces it.  So each assignment also
    carries the negation of every later one that is not mutually exclusive with
    it -- ``criteria = "-link_unusable"`` sits above the ``elif`` chain that
    replaces it, and ``status = "aborted"`` sits above two rechecks at the end
    of the function that can replace it whatever the chain decided.

    Exclusive siblings are skipped, or every branch of a chain would carry the
    negation of every other: ``-dest_blacklisted`` would come out requiring
    ``not (totalQueued >= limit)``, a condition with nothing to do with it.
    Exclusive as the function runs (:func:`parted`): ``JobGenerator.runImpl``
    tests ``tmpStat != SC_SUCCEEDED`` three times with a new ``tmpStat``
    between each, so a later ``tmpErrStr`` can replace an earlier one although
    the texts contradict.

    What counts as settled is the caller's to widen.  By default a string
    literal, and nothing here knows anything else; *resolve* lets a caller that
    does -- one holding the corpus's declared mappings -- settle
    ``newTaskStatus = commandStatusMap[commandStr]["doing"]`` to the six statuses
    it can hold.  One assignment may then contribute several values, all under
    the same guards, since the guards are what reached the assignment and the
    mapping is what chose among its entries.
    """

    def literal_only(expression: ast.expr, _func: ast.AST) -> list[str]:
        if isinstance(expression, ast.Constant) and isinstance(expression.value, str):
            return [expression.value]
        return []

    settle = resolve or literal_only
    found: list[tuple[ast.stmt, list[str]]] = []
    for node in ast.walk(func):
        value = written_value(node)
        if value is None:
            continue
        if not any(isinstance(t, ast.Name) and t.id == name for t in targets_of(node)):
            continue
        if enclosing_function(node) is not func:
            continue
        settled = settle(value, func)
        if settled:
            found.append((node, settled))
    found.sort(key=lambda pair: pair[0].lineno)
    conditions = {node: path_condition(node) for node, _ in found}
    values: list[tuple[str, list[str], int]] = []
    for assignment, settled in found:
        guards = list(conditions[assignment])
        for other, _ in found:
            if other.lineno <= assignment.lineno:
                continue
            if parted(assignment, other, exits=False):
                continue
            test = own_test(other)
            if test and f"not ({test})" not in guards:
                guards.append(f"not ({test})")
        # A copy per value: two branches sharing one condition list is a
        # mutation away from one of them rewriting the other's reason.
        values.extend((value, list(guards), assignment.lineno) for value in settled)
    return values


def _substitute_bare_name(
    rendered: str, test: ast.expr, func: ast.FunctionDef | ast.AsyncFunctionDef
) -> str:
    """Annotate a test that is a bare local name with the expression behind it."""
    target = test.operand if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not) else test
    if not isinstance(target, ast.Name):
        return rendered
    node = single_definition(func, target.id)
    if node is None:
        return rendered
    try:
        definition = ast.unparse(node)
    except Exception:  # noqa: BLE001 -- unparse fails on synthesised nodes
        return rendered
    if definition == target.id:
        return rendered
    return f"{rendered}  [{target.id} := {definition}]"


def enclosing_class(node: ast.AST) -> Optional[str]:
    for ancestor in ancestors(node):
        if isinstance(ancestor, ast.ClassDef):
            return ancestor.name
    return None


def handed_in(
    func: ast.FunctionDef | ast.AsyncFunctionDef, expression: ast.AST, hops: int = 1
) -> str:
    """The parameter or ``self`` field *expression* takes its value from, or ``""``.

    Asked of a write's row key -- the bind a ``WHERE jediTaskID=:jediTaskID``
    is filled with -- to say whether the rows it lands on were named by this
    function or handed to it.  ``setCloudToTasks_JEDI`` binds the key from the
    ``taskCloudMap`` its caller passed; ``rescuePickedFiles_JEDI`` binds it from
    the rows of its own query.  The first sends a reader to the caller, the
    second does not.

    *hops* through a local, counted the way the walk counts them: one local's
    single definition or one loop over something.  A method's result is the
    callee's and not handed in -- ``self.cur.fetchall()`` is the query's rows,
    not the cursor field -- but a method called *on* something handed in is
    reading it: ``taskCloudMap.items()``.
    """
    arguments = getattr(func, "args", None)
    handed = (
        {a.arg for a in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs)}
        - {"self", "cls"}
        if arguments is not None
        else set()
    )

    def root(node: ast.AST) -> ast.AST:
        while isinstance(node, (ast.Attribute, ast.Subscript)):
            node = node.value
        return node

    def visit(node: ast.AST, left: int) -> str:
        if isinstance(node, ast.Call):
            for argument in (*node.args, *(keyword.value for keyword in node.keywords)):
                found = visit(argument, left)
                if found:
                    return found
            called = node.func
            if isinstance(called, ast.Attribute):
                base = root(called.value)
                if not (isinstance(base, ast.Name) and base.id == "self"):
                    return visit(called.value, left)
            return ""
        if isinstance(node, (ast.Attribute, ast.Subscript)):
            base = root(node)
            if isinstance(base, ast.Name) and base.id == "self":
                chain = node
                while isinstance(chain, (ast.Attribute, ast.Subscript)):
                    if (
                        isinstance(chain, ast.Attribute)
                        and isinstance(chain.value, ast.Name)
                        and chain.value.id == "self"
                    ):
                        return f"self.{chain.attr}"
                    chain = chain.value
                return ""
            found = visit(node.value, left)
            if found or not isinstance(node, ast.Subscript):
                return found
            return visit(node.slice, left)
        if isinstance(node, ast.Name):
            if node.id in handed:
                return node.id
            if left <= 0:
                return ""
            for loop in ast.walk(func):
                if isinstance(loop, (ast.For, ast.AsyncFor)) and node.id in _loop_names(
                    loop.target
                ):
                    found = visit(loop.iter, left - 1)
                    if found:
                        return found
            bound = single_definition(func, node.id)
            return visit(bound, left - 1) if bound is not None else ""
        for child in ast.iter_child_nodes(node):
            found = visit(child, left)
            if found:
                return found
        return ""

    return visit(expression, hops)


def _loop_names(target: ast.AST) -> set[str]:
    """The local names a loop target binds, through any unpacking."""
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        return {name for element in target.elts for name in _loop_names(element)}
    if isinstance(target, ast.Starred):
        return _loop_names(target.value)
    return set()
