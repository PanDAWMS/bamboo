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
import functools
from typing import Callable, Iterator, NamedTuple, Optional, Sequence

from bamboo.codemap.astwalk import walk
from bamboo.codemap.panda import sqleval
from bamboo.codemap.panda.reaching import reaching, reaching_exit


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


def qualified_name(func: ast.AST) -> str:
    """``Class.method``, ``function`` or ``outer.inner``: *func*'s name with what encloses it.

    Read off ``parent`` links, so the tree must have them.  The bare name is
    no identity -- ``datasetManager.py`` defines seven methods named ``run``,
    one in each class its ``main`` declares -- and spelling every enclosing
    ``def`` and ``class`` is what tells them apart.
    """
    names = [getattr(func, "name", "")]
    for ancestor in ancestors(func):
        if isinstance(ancestor, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(ancestor.name)
    return ".".join(reversed(names))


def attach_parents(tree: ast.AST) -> None:
    for parent in walk(tree):
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
    callers read what the ``=`` and ``+=`` fragments of a value write and need
    the right-hand side of both.
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
            for node in walk(self.scope):
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
            for node in walk(self.scope):
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
            for node in walk(self.scope):
                if node is not self.scope and isinstance(node, _SCOPES):
                    inner.update(id(held) for held in walk(node) if held is not node)
            found: dict[str, list[Binding]] = {}
            for node in walk(self.scope):
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
    for node in walk(tree):
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
        node.name: node for node in walk(tree) if isinstance(node, ast.ClassDef)
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


def unsubscripted(base: ast.expr) -> ast.expr:
    """A base clause without its type arguments: ``FactoryBase[WatchDogBase]`` -> ``FactoryBase``.

    Eight of JEDI's factories derive from ``FactoryBase[...]``, and a reader
    that looks for a name and finds a subscript drops the base without a
    word -- the class reads as deriving from one thing fewer.  The argument
    is not a base (see :func:`~bamboo.codemap.panda.attribution.base_name`),
    so only the subscripted value is kept.
    """
    while isinstance(base, ast.Subscript):
        base = base.value
    return base


def _base_name(base: ast.expr) -> Optional[str]:
    base = unsubscripted(base)
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
    one chain each hold a negation of something the other asserts.  A test and
    its negation are matched by :func:`negates`, as they mean, not as they are
    spelled: ``if noNewJob:`` and a later ``if not noNewJob:`` write
    ``noNewJob`` and ``not noNewJob``, which no ``not (...)`` spelled.

    Conservative where a missed exclusion is the safe direction for both
    callers.  Deciding whether a later write overwrites an earlier one, a
    missed exclusion adds a condition that is true but redundant where a missed
    *overlap* would drop a required one; splitting a SQL statement built across
    branches, a missed exclusion leaves the statement folded as it was before,
    where a missed overlap would split a statement that is really one.
    """
    return any(negates(test, held) for test in one for held in other) or any(
        negates(test, held) for test in other for held in one
    )


def negates(test: str, other: str) -> bool:
    """Whether the condition *other* is the negation of *test*, both as :func:`path_condition` writes them.

    Read by the AST, so ``not noNewJob``, ``not (noNewJob)`` and ``not
    (noNewJob)  [noNewJob := ...]`` are one negation of ``noNewJob``: a bare
    name's annotation (:func:`_substitute_bare_name`) says what the name is
    bound to and is left out.  A comparison is the negation of its opposite
    (:func:`sqleval.positive`): ``setNumSlotsForWP`` builds its ``DELETE``
    under ``if numSlots == -1`` and runs it under the ``else`` of ``if
    numSlots != -1``.  A condition that does not parse is compared as written.
    """
    return _read_test(other) == _read_test(f"not ({_unannotated(test)})")


@functools.lru_cache(maxsize=None)
def _read_test(condition: str) -> str:
    text = _unannotated(condition)
    try:
        return ast.dump(_negations_read(ast.parse(text, mode="eval").body))
    except SyntaxError:
        return text


def _negations_read(test: ast.expr) -> ast.expr:
    """*test* with a comparison's opposite read as ``not`` of it, and ``not not`` as nothing."""
    form, opposite = sqleval.positive(test)
    if opposite:
        return ast.UnaryOp(op=ast.Not(), operand=form)
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        inner = _negations_read(test.operand)
        if isinstance(inner, ast.UnaryOp) and isinstance(inner.op, ast.Not):
            return inner.operand
        return ast.UnaryOp(op=ast.Not(), operand=inner)
    return test


def _unannotated(condition: str) -> str:
    """*condition* without the ``  [name := definition]`` :func:`_substitute_bare_name` puts after a bare name."""
    rendered, sep, rest = condition.partition("  [")
    name, colon, _definition = rest.partition(" := ")
    if sep and colon and name.isidentifier() and rest.endswith("]"):
        return rendered
    return condition


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

    Two ways a test keeps *node* from running: an ``if`` holding it whose
    test the values decide against the arm it is in, and an ``if`` before it
    in a block holding it that leaves the block on the arm the values decide
    -- ``if forAnal: return None`` above ``job.jobStatus = "unknown"``.  Only
    *func*'s own statements are ruled on: a nested function or lambda has
    parameters of its own, whatever they are called.
    """
    for test, keeps in _tests_on(node, func):
        result = decided(test, values)
        if result is not UNDECIDED and result is not keeps:
            return True
    return False


def tests_on(node: ast.AST, func: ast.AST) -> list[ast.expr]:
    """The tests :func:`ruled_out_by` reads for *node* in *func*: what its parameters must settle to rule on it."""
    return [test for test, _keeps in _tests_on(node, func)]


def _tests_on(node: ast.AST, func: ast.AST) -> list[tuple[ast.expr, bool]]:
    """Each test of *func* that decides whether *node* runs, with the value it runs on.

    Kept on *node*, as its ``parent`` is: a route reading asks it of every
    node of a function, once per route.
    """
    held = getattr(node, "_tests_on", None)
    if held is not None and held[0] is func:
        return held[1]
    found = _read_tests_on(node, func)
    node._tests_on = (func, found)  # type: ignore[attr-defined]
    return found


def _read_tests_on(node: ast.AST, func: ast.AST) -> list[tuple[ast.expr, bool]]:
    if enclosing_function(node) is not func:
        return []
    found: list[tuple[ast.expr, bool]] = []
    child = node
    for ancestor in ancestors(node):
        for field in ("body", "orelse", "finalbody"):
            block = getattr(ancestor, field, None)
            if not isinstance(block, list) or not any(child is one for one in block):
                continue
            for statement in block[: next(i for i, one in enumerate(block) if one is child)]:
                if isinstance(statement, ast.If):
                    if _jumps(statement.body):
                        found.append((statement.test, False))
                    elif statement.orelse and _jumps(statement.orelse):
                        found.append((statement.test, True))
        if isinstance(ancestor, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            break
        if isinstance(ancestor, ast.If):
            if any(child is one for one in ancestor.body):
                found.append((ancestor.test, True))
            elif any(child is one for one in ancestor.orelse):
                found.append((ancestor.test, False))
        child = ancestor
    return found


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
    return {loop.lineno for loop in _loops_holding(node)}


def _loops_holding(node: ast.AST) -> list[ast.For | ast.AsyncFor | ast.While]:
    """Every loop whose *body* holds *node*, innermost first."""
    found: list[ast.For | ast.AsyncFor | ast.While] = []
    previous = node
    for ancestor in ancestors(node):
        if isinstance(
            ancestor, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)
        ):
            break
        if isinstance(ancestor, (ast.For, ast.AsyncFor, ast.While)) and previous in ancestor.body:
            found.append(ancestor)
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
    return {node.id for node in walk(expression) if isinstance(node, ast.Name)}


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
    for node in walk(func):
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
    :func:`parted` then reads it as written.  Held on the node, since it is
    asked of many pairs.
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


def parted(
    one: ast.AST, other: ast.AST, exits: bool = True, around: Optional[ast.AST] = None
) -> bool:
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
    it.  When statements were reassembled, 98.5% of the "cannot meet" answers came
    from one statement's arms; that rule turned 334 of them into "can meet" and
    would have built statements no run sends.  Read by statement, 12 pairs
    change there, all in ``getTasksWithCriteria_JEDI``.

    *exits* reads :func:`reachable_when`'s early exits as well as the ``if``
    tests, as :func:`compatible` does; without it, the tests alone.

    *around* reads the contradiction across the passes of that loop rather
    than within one: one pass runs one arm and a later pass the other, so it
    holds only where the loop gives no name a test reads a new value -- a
    ``for`` target, a binding or a call on it, as :func:`_rebinds` reads them,
    anywhere in the loop (:func:`_given_in`).  ``parse_condition_string``
    replaces ``cond_string`` in the ``else`` of ``if not item_list``, goes
    round its ``while``, binds ``item_list`` afresh, and hands the replaced
    string on from the ``if``.  A call counts here as it does between two
    tests, and the whole loop is between: ``doGenerate`` makes 68 calls on
    ``taskSpec`` in its loop, so ``taskSpec.usePrePro()`` reads as able to
    change from one pass to the next, which it is not -- a value offered in
    excess, the safe side.
    """
    func = enclosing_function(one)
    mine, theirs = _tested_by(one, exits), _tested_by(other, exits)
    for text, statement in mine:
        for other_text, other_statement in theirs:
            if not (negates(text, other_text) or negates(other_text, text)):
                continue
            if statement is None or other_statement is None or func is None:
                return True
            if around is not None:
                if _given_in(func, around, (statement, other_statement)):
                    continue
                return True
            if statement is other_statement:
                return True
            earlier, later = sorted((statement, other_statement), key=lambda held: held.lineno)
            if not _given_between(func, earlier, later, (one, other)):
                return True
    return False


def _given_in(func: ast.AST, loop: ast.AST, statements: tuple[ast.stmt, ...]) -> bool:
    """Whether *loop* gives a name one of the *statements*' tests reads a new value."""
    names: set[str] = set()
    for statement in statements:
        names |= _names_in(statement.test)  # type: ignore[attr-defined]
    rebinds = _rebinds(func)
    return any(_within(site, loop) for name in names for site in rebinds.get(name, ()))


def arm_of(node: ast.AST) -> Optional[tuple[ast.If, bool]]:
    """The innermost ``if`` arm holding *node*: the ``if``, and whether the arm is its body.  None outside any."""
    previous = node
    for ancestor in ancestors(node):
        if isinstance(ancestor, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            return None
        if isinstance(ancestor, ast.If):
            if any(previous is one for one in ancestor.body):
                return ancestor, True
            if any(previous is one for one in ancestor.orelse):
                return ancestor, False
        previous = ancestor
    return None


def only_through(site: ast.AST, piece: ast.AST) -> bool:
    """Whether *site* runs only on a pass through the innermost ``if`` arm holding *piece*.

    On that arm, or under a test meaning the arm's condition -- matched by the
    AST as :func:`negates` matches, the same way round -- with no name the test
    reads given a new value between the two, as :func:`parted` reads two
    statements testing one thing.  ``updateTaskStatusByContFeeder_JEDI`` adds
    ``,oldStatus=:oldStatus,splitRule=:splitRule`` to its ``UPDATE`` under ``if
    taskSpec is not None`` and fills ``:status`` from ``taskSpec`` under a
    second ``if taskSpec is not None`` further down: a pass that built the
    statement without the clause never ran that fill.
    """
    arm = arm_of(piece)
    if arm is None:
        return False
    branch, body = arm
    if any(_within(site, one) for one in (branch.body if body else branch.orelse)):
        return True
    func = enclosing_function(branch)
    condition = next((text for text, held in _tested_by(piece, False) if held is branch), None)
    if func is None or condition is None:
        return False
    for text, held in _tested_by(site, False):
        if held is None or held is branch or not negates(text, f"not ({_unannotated(condition)})"):
            continue
        earlier, later = sorted((branch, held), key=lambda one: one.lineno)
        if not _given_between(func, earlier, later, (piece, site)):
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
    :func:`surviving_writes` is that answer.

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
    *,
    killers: Sequence[ast.AST] = (),
) -> list[ast.AST]:
    """Of *sites*, the writes that can still be what *target* reads.

    The body of :func:`reaching_definitions`, taken apart so that a caller
    holding writes from several shapes can ask about all of them at once.  The
    walk is that caller: a name arrives at a read through a plain assignment,
    a tuple unpacking, a ``with`` or a ``for`` target, and cancelling only
    within one of those shapes would keep a tuple-bound ``res`` alive against
    the assignment that replaced it.

    Two steps.  :func:`~bamboo.codemap.panda.reaching.reaching` follows the
    function the way it runs and keeps the writes that some path brings to
    the read with nothing overwriting them on the way; *rereads* add to what
    is there and overwrite nothing, *killers* overwrite and are never an
    answer.  The flow takes every branch both ways, so :func:`can_reach` then
    drops the writes whose conditions contradict the read's.

    *as_run* reads a test as describing its name as it was when the test ran
    (:func:`parted`).  The walk and the build both ask for it.

    With *as_run*, a write the conditions drop is kept where it reaches the
    read round a loop that holds both (:func:`_round_a_loop`): a contradiction
    is one pass's, and the next pass tests afresh.  ``insertFilesForDataset_JEDI``
    adds ``nRemEvents`` to ``splitStartEvent`` in the ``else`` of ``if
    splitStartEvent + nRemEvents >= numEventsInFile``, and the next pass's
    ``if`` starts the last chunk there.  Only that way round:
    ``_read_unprocessed_inputs`` tests ``tmp_dataset_spec.isMaster()`` twice in
    one pass, and the first test's ``max_files_to_read_for_this_dataset`` does
    not reach the second's ``else`` -- within the pass the contradiction holds,
    and round the loop every arm binds the name again.  The flow takes every
    branch both ways, so a value a literal rules out still comes round:
    ``get_origin_datasets`` sets ``dataset_names = None`` and breaks, and the
    ``append`` under ``dataset_names is not None`` is offered it.
    """
    reached = reaching(target, sites, killers, rereads)
    kept = [site for site in reached if site is target or can_reach(site, target, as_run)]
    if as_run and len(kept) < len(reached):
        held = {id(site) for site in kept}
        extra = {
            id(site)
            for site in reached
            if id(site) not in held and _round_a_loop(site, target, sites, killers, rereads)
        }
        if extra:
            kept = [site for site in reached if id(site) in held or id(site) in extra]
    return kept


def _round_a_loop(
    site: ast.AST,
    target: ast.AST,
    sites: Sequence[ast.AST],
    killers: Sequence[ast.AST],
    rereads: Sequence[ast.AST],
    beside: Optional[ast.AST] = None,
) -> bool:
    """Whether *site* reaches *target* from one pass of a loop holding both to a later pass.

    The conditions do not part them across the passes (:func:`parted`
    *around* the loop), *site* is live at the top of the next pass, and a
    write at the top of a pass -- the loop itself, as a site, is one -- reaches
    *target* with nothing overwriting it.  With *beside*, the loop holds it too
    and does not part it from *site* either: the later pass runs it.
    """
    theirs = {id(loop) for loop in _loops_holding(target)}
    if beside is not None:
        theirs &= {id(loop) for loop in _loops_holding(beside)}
    for loop in _loops_holding(site):
        if id(loop) not in theirs or parted(site, target, around=loop):
            continue
        if beside is not None and parted(beside, site, around=loop):
            continue
        if not any(site is one for one in reaching(loop.body[0], sites, killers, rereads)):
            continue
        if any(loop is one for one in reaching(target, [*sites, loop], killers, rereads)):
            return True
    return False


def carried_round(
    site: ast.AST, target: ast.AST, beside: ast.AST, sites: Sequence[ast.AST], rereads: Sequence[ast.AST] = ()
) -> bool:
    """Whether *site*, parted from *beside* on one pass, is still what *target* reads on a pass running *beside*.

    It is where a loop holds all three and carries *site* to *target* from
    one pass to a later one that runs *beside* (:func:`_round_a_loop`), the
    map's renewals among *sites*.  A map made afresh in the loop carries
    nothing: ``insertNewJob`` makes ``varMap = {}`` for each file, so the
    other arm's ``:status`` of the file before is gone.
    """
    return _round_a_loop(site, target, sites, (), rereads, beside)


def literal_values(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    name: str,
    resolve: Optional[Callable[[ast.expr, ast.AST], list[str]]] = None,
    at: Optional[ast.AST] = None,
) -> list[tuple[str, list[str], int]]:
    """``(value, conditions, line)`` for every settled ``name = ...`` in *func*, or every one *at* reads.

    The line is the assignment's, because that is where the value is decided --
    an anchor pointing at the use would send a reader to the place that merely
    passes it on.

    Reaching definitions for one local, which is what a value assigned to a
    variable before it is used needs -- the tag a broker interpolates into its
    rejection message, or the status a post-processor's helper returns.

    **Reassignment is the part dominating-guard analysis cannot see.**  The
    conditions on an assignment are necessary and, on their own, not sufficient:
    a later write reaching the same name replaces it.  So with *at* -- the
    statement that puts the name into a write, the one a branch is about -- an
    assignment is a value only where it can still be what *at* reads, and it
    carries an entry for every other binding of the name that replaces it on
    the way (:func:`replacements`): ``criteria = "-link_unusable"`` sits above
    the ``elif`` chain that replaces it, and ``status = "aborted"`` sits above
    two rechecks at the end of the function that can replace it whatever the
    chain decided.  A binding the value computes counts as one a literal does:
    ``reset_resource_type_task`` replaces ``resource_name = "Undefined"`` in a
    loop with the resource spec's name.  ``retryTask_JEDI`` sets ``newStatus =
    None`` where no bind reads it, and that is no value of the bind.  And an
    augmented binding replaces it as well: after ``tmpNumFiles += ...`` in
    ``updateUnmergedDatasets``' loop the bind reads another number than the
    ``0`` above it, so ``0`` is a value only where no pass added to it; one
    the value surely meets leaves it no value of the bind past it, and a test
    past one does not read the name as the value.

    Exclusive siblings replace nothing, or every branch of a chain would carry
    the negation of every other: ``-dest_blacklisted`` would come out requiring
    ``not (totalQueued >= limit)``, a condition with nothing to do with it.
    Exclusive as the function runs (:func:`parted`): ``JobGenerator.runImpl``
    tests ``tmpStat != SC_SUCCEEDED`` three times with a new ``tmpStat``
    between each, so a later ``tmpErrStr`` can replace an earlier one although
    the texts contradict.

    Without *at*, every settled assignment under its own guards: what a caller
    asking which values the name can hold anywhere reads.

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
    for node in walk(func):
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
    own = [binding for binding in bindings(func, name) if not binding.nested]
    sites = _unique([binding.site for binding in own])
    # The name's ``+=``: each keeps what was there and makes another value of it.
    added = [site for site in sites if isinstance(site, ast.AugAssign)]
    values: list[tuple[str, list[str], int]] = []
    for assignment, settled in found:
        guards = list(path_condition(assignment))
        if at is not None:
            replaced = replacements(
                assignment,
                [site for site in sites if site is not assignment],
                [at],
                place=name,
                value=written_value(assignment),
                rereads=added,
            )
            if replaced is None:
                continue
            # Each value the assignment is settled to meets the tests on the way as a literal does in
            # replacements: getTasksToExecCommand_JEDI binds the status only under `if newTaskStatus != "dummy"`.
            settled = [one for one in settled if not _refused(at, assignment, func, name, one, added, (), texts=True)]
            if not settled:
                continue
            guards.extend(entries(replaced, guards))
        # A copy per value: two branches sharing one condition list is a
        # mutation away from one of them rewriting the other's reason.
        values.extend((value, list(guards), assignment.lineno) for value in settled)
    return values


def _unique(nodes: Sequence[ast.AST]) -> list[ast.AST]:
    return list({id(node): node for node in nodes}.values())


# ---------------------------------------------------------------------------
# A value's way from the write to where it is read
# ---------------------------------------------------------------------------
#
# A branch's condition is the condition its value reaches its write under: the
# guards on the statement that puts it in its place, and, for every other write
# of the place that can replace it before the place is read, that the other did
# not run.  Three routes, one rule -- a local to the statement putting it into
# a write (:func:`literal_values`), a statement map's key to the call running
# the statement, an object's field to where the object leaves the function
# (:func:`field_replacements`).  A write whose value reaches none of them is
# no branch.

#: :class:`Replacement` kinds: an entry is written ...
REPLACES = "replaces"
#: ... or none, because the other write is not below this one, or the two cannot both run;
NOT_AFTER = "not after"
#: the value is not there when the other write runs (it was replaced or its holder renewed first);
NOT_REACHED = "not reached"
#: the other write leads to no read the value reaches;
OFF_THE_WAY = "off the way"
#: the value settles the other write's test against it, so the other never runs after it;
NEVER_AFTER = "never after"
#: the value settles every test of the other write for it, so the other runs wherever the value gets to
#: it -- the value's way past it is cut;
SURELY = "surely"
#: the value may have been read before the other write runs;
LEFT_BEFORE = "left before"
#: a return, raise, continue or break on the way can skip the other write with its tests holding;
SKIPPED = "skipped"
#: the other write is in a handler, a loop's or try's ``else``, a ``while``, an ``async for`` or a
#: ``match`` case the value's write is not in, whose tests are not enough for it to run;
NOT_ON_ITS_TESTS = "not on its tests"
#: the other write has no test the value's write is not under -- it runs wherever the value's pass
#: gets to it, and the value reaches a read past it only where something raises -- past a ``+=``, not as
#: itself at all;
NO_TEST = "no test"
#: or the entry would not parse.
UNREADABLE = "unreadable"


class Replacement(NamedTuple):
    """Another write of a place, read for what it does to a value written there before it.

    *entry* is the condition the value needs for the other write not to have
    replaced it -- ``not (<the tests the other write is under and this one is
    not>)`` -- where *why* is :data:`REPLACES`; else ``None``, and *why* says
    why no entry can be written.
    """

    site: ast.AST
    why: str
    entry: Optional[str] = None


class Callee(NamedTuple):
    """A method a call runs that writes the field: its name, and each of its writes' conditions there.

    The conditions are the method's own, as :func:`path_condition` writes
    them inside it, on ``self``.
    """

    method: str
    writes: tuple[tuple[str, ...], ...]


def entries(replaced: Sequence[Replacement], held: Sequence[str] = ()) -> list[str]:
    """The entries of *replaced*, in order, less any *held* already has."""
    found: list[str] = []
    for one in replaced:
        if one.entry is not None and one.entry not in held and one.entry not in found:
            found.append(one.entry)
    return found


def replacements(
    write: ast.AST,
    others: Sequence[ast.AST],
    reads: Sequence[ast.AST],
    *,
    place: str,
    value: Optional[ast.expr] = None,
    killers: Sequence[ast.AST] = (),
    rereads: Sequence[ast.AST] = (),
    partial: Sequence[ast.AST] = (),
    returning: bool = False,
    callee: Optional[Callable[[ast.AST], Optional[Callee]]] = None,
) -> Optional[list[Replacement]]:
    """What each of *others* does to the value *write* puts in *place*; ``None`` where the value reaches no read.

    *others* are the other writes of the place, *rereads* those of them that
    add to it and replace nothing (``x += y``), *killers* what gives the place
    a new holder (a ``for``'s next pass, ``varMap = {}``) -- the map's
    reaching reads them all at once (:func:`surviving_writes`, as the function
    runs).  *partial* are writes of others that may leave the value there --
    a setter call writing the field only under the method's tests -- read as
    rereads by the flow and still asked about.  *reads* are the statements the
    value can leave the place at;
    *returning* adds the function's normal end.  *value* is what *write*
    writes, and *callee* says which of *others* are calls of a method writing
    the place, with the method's own tests.

    Another write replaces the value -- and the value needs it not to have
    run -- where it is below *write*, can run on one pass with it, the value
    is still there when it runs, it leads on to a read the value reaches, and
    no such read can come between them.  Its entry is the tests it is under
    and *write* is not, as one conjunction: ``not (doMerging and tmpMatch is
    not None)`` for ``ppEventServiceJob``'s rewrite two tests deeper.  Each
    test is as written, unannotated, and reads its names as they are when it
    runs, as every test in a path condition does.

    A ``+=`` of the place is asked too, a reread or not: past it the place
    holds another value.  Where the value surely meets one (:data:`NO_TEST`)
    it goes no further as itself -- ``setDatasetAttribute`` sets
    ``self.attributes = ""`` and then ``self.attributes += attr``, and the
    object never leaves holding ``""`` -- and where one may run, the value
    needs it not to have run, as for any other write.  It stays a reread for
    the tests on the way: one past it does not read the place as the value.

    A read is one the value reaches only where no test on its way refuses it:
    a test of an ``if`` holding the read and not *write*, reading *place*
    while it still holds the literal *value* -- ``retryJob``'s ``longSite =
    None`` never gets to ``job.computingSite = longSite`` under ``if longSite
    is not None``.

    The entry has to be false only where the other write surely ran, so:

    - a test that reads *place* itself is read with *value* where that is a
      literal settling it and no reread on the way has made something else
      of it -- ``killJob``'s ``if job.jobStatus != "closed"`` after
      ``job.jobStatus = "closed"`` never runs, and a test the literal makes
      true says nothing;
    - in a ``for`` the value's write is not in, the other write ran on some
      pass: ``not any(<its tests in the loop> for <target> in <iter> ...)``,
      outermost loop first;
    - a callee's write ran under the method's tests, which the caller's names
      cannot read: the entry ends in one ``  [in <method>()]``, read as a
      whole;
    - a test reading *place* or a name given a new value after *write* reads
      it as it is then, not as it was at *write*: the entry ends in one ``
      [at <line>]``, the line of the last such test, read as a whole --
      ``remAttribute``'s ``not (self.attributes == '')  [at 391]``, where
      ``self.attributes`` is what the rewrites before made of it;
    - no entry where a jump on the way, a handler, an ``else`` of a loop or
      ``try``, a ``while``, an ``async for`` or a ``match`` case can keep the
      other write from running with its tests holding.

    What it does not read: a call raising between the two (as nowhere in a
    path condition), and a write of the place through another name for its
    holder.
    """
    func = enclosing_function(write)
    if func is None:
        return []
    sites = _unique([write, *others])
    keeping = _unique([*rereads, *partial])
    kept = _literal(value)
    reached = [
        read
        for read in reads
        if read is not write
        and any(site is write for site in surviving_writes(sites, read, keeping, as_run=True, killers=killers))
        and not (kept is not None and _refused(read, write, func, place, kept[0], keeping, killers))
    ]
    exits = returning and any(site is write for site in reaching_exit(func, sites, killers, keeping))
    if not reached and not exits:
        return None
    asked = [
        other
        for other in others
        if other is not write and (isinstance(other, ast.AugAssign) or not any(other is one for one in rereads))
    ]
    found = [
        _replacement(write, other, func, reached, exits, sites, killers, keeping, place, kept, callee)
        for other in asked
    ]
    cut = [held for one, held in found if one.why == SURELY]
    cut += [one.site for one, _held in found if one.why == NO_TEST and isinstance(one.site, ast.AugAssign)]
    if cut:
        # Where the value always meets another write, it goes no further.
        killers = _unique([*killers, *cut])
        keeping = [one for one in keeping if not any(one is held for held in cut)]
        reached = [
            read
            for read in reached
            if any(site is write for site in surviving_writes(sites, read, keeping, as_run=True, killers=killers))
        ]
        exits = exits and any(site is write for site in reaching_exit(func, sites, killers, keeping))
        if not reached and not exits:
            return None
        found = [
            _replacement(write, other, func, reached, exits, sites, killers, keeping, place, kept, callee)
            for other in asked
        ]
    return [one for one, _held in found]


def _literal(value: Optional[ast.expr]) -> Optional[tuple[object]]:
    """``(v,)`` where *value* is a literal ``v``, else ``None``."""
    if value is None:
        return None
    try:
        return (ast.literal_eval(value),)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return None


def _replacement(
    write: ast.AST,
    other: ast.AST,
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    reached: list[ast.AST],
    exits: bool,
    sites: list[ast.AST],
    killers: Sequence[ast.AST],
    rereads: Sequence[ast.AST],
    place: str,
    kept: Optional[tuple[object]],
    callee: Optional[Callable[[ast.AST], Optional[Callee]]],
) -> tuple[Replacement, Optional[ast.AST]]:
    if getattr(other, "lineno", 0) <= getattr(write, "lineno", 0) or parted(write, other, exits=False):
        return Replacement(other, NOT_AFTER), None
    if not any(
        site is write
        for site in surviving_writes(sites, other, rereads, as_run=True, killers=killers)
    ):
        return Replacement(other, NOT_REACHED), None
    if not (
        any(_leads_to(other, read) for read in reached)
        or (exits and bool(reaching_exit(func, [other])))
    ):
        return Replacement(other, OFF_THE_WAY), None
    # Between them on one way that keeps the value there: a read on an earlier
    # pass, with the write run again since, read that pass's value, and a
    # renewal leaves the other write another object.
    if any(
        _leads_to(read, other, (write, *killers))
        for read in reached
        if not any(read is one for one in killers)
    ):
        return Replacement(other, LEFT_BEFORE), None
    if _skipped(write, other, func, reached, sites, killers, rereads):
        return Replacement(other, SKIPPED), None
    structure = _tests_after(write, other, func)
    if structure is None:
        return Replacement(other, NOT_ON_ITS_TESTS), None
    outside, loops, inside = structure
    parts_outside: list[str] = []
    parts_inside: list[str] = []
    bound = {name for loop in loops for name, _slot, _unpacked in names_bound_by(loop.target)}
    # The lines of the tests whose names may not be what they were at the write.
    after = [loop.lineno for loop in loops if _read_after(loop.iter, write, other, func, place, bound)]
    # The tests the value makes true, outermost first: where they are all there is, the
    # other write runs wherever the value gets to the first of them.
    settled: list[ast.AST] = []
    for tests, parts in ((outside, parts_outside), (inside, parts_inside)):
        for test, holds in tests:
            if kept is not None and _holding(write, test.parent, rereads, killers):  # type: ignore[attr-defined]
                result = _decided_with(test, place, kept[0])
                if result is not UNDECIDED:
                    if result is not holds:
                        return Replacement(other, NEVER_AFTER), None
                    settled.append(test.parent)  # type: ignore[attr-defined]
                    continue
            try:
                text = ast.unparse(test)
            except Exception:  # noqa: BLE001 -- unparse fails on synthesised nodes
                return Replacement(other, UNREADABLE), None
            parts.append(text if holds else f"not ({text})")
            if _read_after(test, write, other, func, place, bound):
                after.append(test.parent.lineno)  # type: ignore[attr-defined]
    method = None
    called = callee(other) if callee is not None else None
    if called is not None and kept is not None and not _holding(write, other, rereads, killers):
        kept = None
    if called is not None:
        alternatives: list[str] = []
        surely = False
        for conditions in called.writes:
            held: list[str] = []
            ruled_out = False
            for condition in conditions:
                text = _unannotated(condition)
                try:
                    test = ast.parse(text, mode="eval").body
                except SyntaxError:
                    return Replacement(other, UNREADABLE), None
                if kept is not None:
                    result = _decided_with(test, f"self.{place.rsplit('.', 1)[-1]}", kept[0])
                    if result is False:
                        ruled_out = True
                        break
                    if result is True:
                        continue
                held.append(text)
            if ruled_out:
                continue
            if not held:
                # A write the method makes whenever it runs: the call writes the field.
                surely = True
                if conditions:
                    settled.append(other)
                break
            alternatives.append(_joined(held))
        if not surely:
            if not alternatives:
                return Replacement(other, NEVER_AFTER), None
            parts_inside.append(
                " or ".join(_wrapped(one) for one in alternatives) if len(alternatives) > 1 else alternatives[0]
            )
            method = called.method
    if loops:
        clauses = " ".join(
            f"for {ast.unparse(loop.target)} in {ast.unparse(loop.iter)}" for loop in loops
        )
        condition = _joined(parts_inside) if parts_inside else "True"
        parts_outside.append(f"any({condition} {clauses})")
    else:
        parts_outside.extend(parts_inside)
    if not parts_outside:
        if settled:
            return Replacement(other, SURELY), settled[0]
        return Replacement(other, NO_TEST), None
    if loops and len(parts_outside) == 1:
        entry = f"not {parts_outside[0]}"
    else:
        entry = f"not ({_joined(parts_outside)})"
    try:
        ast.parse(entry, mode="eval")
    except SyntaxError:
        return Replacement(other, UNREADABLE), None
    if method is not None:
        entry = f"{entry}  [in {method}()]"
    elif after:
        entry = f"{entry}  [at {max(after)}]"
    return Replacement(other, REPLACES, entry), None


def _read_after(
    expression: ast.expr,
    write: ast.AST,
    other: ast.AST,
    func: ast.AST,
    place: str,
    bound: set[str],
) -> bool:
    """Whether *expression* -- a test, or a loop's iterable, on *other*'s way -- reads what may differ from the write's time.

    *place* itself, which the write gave the value it tests, or a name or
    field given a new value between the write and the statement *expression*
    is in, on a way to *other*: a binding, a call on it, an assignment to it.
    *bound* are names the entry's ``any()`` binds itself.
    """
    start, end = getattr(write, "lineno", 0), getattr(getattr(expression, "parent", None), "lineno", 0)
    texts: set[str] = set()
    names: set[str] = set()
    for node in walk(expression):
        if isinstance(node, (ast.Name, ast.Attribute, ast.Subscript)) and isinstance(getattr(node, "ctx", None), ast.Load):
            texts.add(_text(node))
            if isinstance(node, ast.Name):
                names.add(node.id)
    if place in texts:
        return True
    rebinds = _rebinds(func)
    for name in names - bound:
        for site in rebinds.get(name, ()):
            if start < getattr(site, "lineno", 0) < end and not _apart(site, other):
                return True
    for statement in _own_statements(func):
        if not start < statement.lineno < end or _apart(statement, other):
            continue
        targets = [statement.target] if isinstance(statement, ast.AugAssign) else targets_of(statement)
        if any(_text(target) in texts for target in targets):
            return True
    return False


def unmarked(condition: str) -> str:
    """*condition* without the ``  [at <line>]`` :func:`replacements` ends an entry in, where it has one."""
    rendered, sep, rest = condition.rpartition("  [at ")
    if sep and rest.endswith("]") and rest[:-1].isdigit():
        return rendered
    return condition


def _wrapped(text: str) -> str:
    """*text* as one operand of ``and``: parenthesized where it is a boolean operation or lower."""
    try:
        node = ast.parse(text, mode="eval").body
    except SyntaxError:
        return f"({text})"
    if isinstance(node, (ast.BoolOp, ast.IfExp, ast.Lambda, ast.NamedExpr)):
        return f"({text})"
    return text


def _joined(parts: Sequence[str]) -> str:
    """*parts* as one conjunction; one part as it is."""
    if len(parts) == 1:
        return parts[0]
    return " and ".join(_wrapped(part) for part in parts)


def _leads_to(start: ast.AST, target: ast.AST, through: Sequence[ast.AST] = ()) -> bool:
    """Whether a pass can go from *start* to *target* without running any of *through*.

    From a compound statement's header into its blocks, too, where none of
    *through* is in them.
    """
    if target is not start and _within(target, start) and not any(_within(one, start) for one in through):
        return True
    return any(
        site is start
        for site in surviving_writes([start], target, as_run=True, killers=[one for one in through if one is not start])
    )


def _jump_statements(func: ast.AST) -> list[ast.stmt]:
    held = getattr(func, "_jump_statements", None)
    if held is None:
        held = [
            node
            for node in walk(func)
            if isinstance(node, (ast.Return, ast.Raise, ast.Continue, ast.Break))
            and enclosing_function(node) is func
        ]
        try:
            func._jump_statements = held  # type: ignore[attr-defined]
        except AttributeError:
            pass
    return held


def _skipped(
    write: ast.AST,
    other: ast.AST,
    func: ast.AST,
    reached: list[ast.AST],
    sites: list[ast.AST],
    killers: Sequence[ast.AST],
    keeping: Sequence[ast.AST],
) -> bool:
    """Whether a jump between *write* and *other* can keep *other* from running with its tests holding, and the value still land.

    Between by line, on one pass with both: a ``return`` or ``raise``, or a
    ``continue`` or ``break`` of a loop holding *other* -- with the value
    still there when it runs, and a read the value reaches being the jump
    itself or after it, on a way not through *other*.  ``storeJobs`` returns
    early where a user is banned, between ``userCountry = None`` and the loop
    that may replace it, and nothing it returns to reads the name: on a run
    where the value lands, the return did not run.
    """
    start, end = getattr(write, "lineno", 0), getattr(other, "lineno", 0)
    for jump in _jump_statements(func):
        if not start < jump.lineno < end:
            continue
        if parted(write, jump, exits=False) or parted(jump, other, exits=False):
            continue
        if isinstance(jump, (ast.Continue, ast.Break)):
            loops = _loops_holding(jump)
            if not loops or not _within(other, loops[0]):
                continue
        if not any(site is write for site in surviving_writes(sites, jump, keeping, as_run=True, killers=killers)):
            continue
        if any(jump is read for read in reached) or any(
            _leads_to(jump, read, (write, other, *killers)) for read in reached
        ):
            return True
    return False


def _tests_after(
    write: ast.AST, other: ast.AST, func: ast.AST
) -> Optional[tuple[list[tuple[ast.expr, bool]], list[ast.For], list[tuple[ast.expr, bool]]]]:
    """``(tests outside its loops, its loops, tests inside them)`` *other* is under and *write* is not.

    Each test with whether *other* runs where it holds; the loops outermost
    first.  ``None`` where something *write* is not in can keep *other* from
    running with all of them holding (:data:`NOT_ON_ITS_TESTS`).
    """
    mine = {id(ancestor) for ancestor in ancestors(write)}
    found: list[tuple[str, object]] = []
    previous = other
    for ancestor in ancestors(other):
        if ancestor is func:
            break
        if id(ancestor) not in mine:
            if isinstance(ancestor, (ast.ExceptHandler, ast.match_case)):
                return None
            if isinstance(ancestor, (ast.For, ast.AsyncFor, ast.While, ast.Try)) and any(
                previous is one for one in ancestor.orelse
            ):
                return None
            if isinstance(ancestor, ast.If):
                if any(previous is one for one in ancestor.body):
                    found.append(("test", (ancestor.test, True)))
                elif any(previous is one for one in ancestor.orelse):
                    found.append(("test", (ancestor.test, False)))
            elif isinstance(ancestor, (ast.AsyncFor, ast.While)):
                if any(previous is one for one in ancestor.body):
                    return None
            elif isinstance(ancestor, ast.For) and any(previous is one for one in ancestor.body):
                found.append(("loop", ancestor))
        previous = ancestor
    found.reverse()
    outside: list[tuple[ast.expr, bool]] = []
    loops: list[ast.For] = []
    inside: list[tuple[ast.expr, bool]] = []
    for kind, held in found:
        if kind == "loop":
            loops.append(held)  # type: ignore[arg-type]
        elif loops:
            inside.append(held)  # type: ignore[arg-type]
        else:
            outside.append(held)  # type: ignore[arg-type]
    return outside, loops, inside


class _AsKept(ast.NodeTransformer):
    """A test with every reading of one place spelled as one name."""

    def __init__(self, place: str) -> None:
        self.place = place

    def visit(self, node: ast.AST) -> ast.AST:
        if isinstance(node, (ast.Name, ast.Attribute, ast.Subscript)) and isinstance(
            getattr(node, "ctx", None), ast.Load
        ):
            try:
                if ast.unparse(node) == self.place:
                    return ast.Name(id=_KEPT_NAME, ctx=ast.Load())
            except Exception:  # noqa: BLE001
                pass
        return self.generic_visit(node)


_KEPT_NAME = " kept"


def _decided_with(test: ast.expr, place: str, kept: object) -> object:
    """:func:`decided` for *test* with *place* holding *kept*; :data:`UNDECIDED` where it does not read it."""
    try:
        fresh = ast.parse(ast.unparse(test), mode="eval").body
    except Exception:  # noqa: BLE001
        return UNDECIDED
    spelled = _AsKept(place).visit(fresh)
    if not any(isinstance(node, ast.Name) and node.id == _KEPT_NAME for node in walk(spelled)):
        return UNDECIDED
    return decided(spelled, {_KEPT_NAME: kept})


def _holding(write: ast.AST, statement: ast.AST, rereads: Sequence[ast.AST], killers: Sequence[ast.AST]) -> bool:
    """Whether the place still holds what *write* put there when *statement* runs: no reread on a way between."""
    return not any(
        one is not statement
        and one is not write
        and _leads_to(write, one, killers)
        and _leads_to(one, statement, (write, *killers))
        for one in rereads
    )


def _refused(
    read: ast.AST,
    write: ast.AST,
    func: ast.AST,
    place: str,
    kept: object,
    rereads: Sequence[ast.AST],
    killers: Sequence[ast.AST],
    texts: bool = False,
) -> bool:
    """Whether a test on the way to *read* -- an ``if`` holding it and not *write* -- is settled by *kept* against
    its arm.

    With *texts*, *kept* is a value as a settler spells it -- ``"dummy"``, and ``"0"`` or ``"None"`` for what a
    literal of another type spells -- so a test comparing *place* with anything but a text is not read with it:
    ``"0" == 0`` would refuse a value the code writes.
    """
    child = read
    for ancestor in ancestors(read):
        if ancestor is func:
            return False
        if isinstance(ancestor, ast.If) and not _within(write, ancestor):
            arm: Optional[bool] = None
            if any(child is one for one in ancestor.body):
                arm = True
            elif any(child is one for one in ancestor.orelse):
                arm = False
            if texts and not _compared_with_texts(ancestor.test, place):
                arm = None
            if arm is not None and _holding(write, ancestor, rereads, killers):
                result = _decided_with(ancestor.test, place, kept)
                if result is not UNDECIDED and result is not arm:
                    return True
        child = ancestor
    return False


def _compared_with_texts(test: ast.expr, place: str) -> bool:
    """Whether every constant *test* compares *place* with is a text (a truth test of it compares with none)."""
    for node in walk(test):
        if not isinstance(node, ast.Compare):
            continue
        operands = [node.left, *node.comparators]
        if not any(ast.unparse(operand) == place for operand in operands):
            continue
        for operand in operands:
            for part in walk(operand):
                if isinstance(part, ast.Constant) and not isinstance(part.value, str):
                    return False
    return True


def holder_texts(holder: ast.expr) -> set[str]:
    """*holder*'s text and the text of each object that holds it: ``impl.taskSpec`` and ``impl``."""
    found: set[str] = set()
    node: ast.AST = holder
    while isinstance(node, (ast.Attribute, ast.Subscript, ast.Name)):
        found.add(ast.unparse(node))
        if isinstance(node, ast.Name):
            break
        node = node.value
    return found


def _own_statements(func: ast.AST) -> list[ast.stmt]:
    held = getattr(func, "_own_statements", None)
    if held is None:
        held = [
            node
            for node in walk(func)
            if isinstance(node, ast.stmt) and node is not func and enclosing_function(node) is func
        ]
        try:
            func._own_statements = held  # type: ignore[attr-defined]
        except AttributeError:
            pass
    return held


def _evaluated(statement: ast.stmt) -> Iterator[ast.AST]:
    """The nodes *statement* evaluates itself: a compound statement's header, any other statement whole.

    A nested ``def`` or ``class`` whole, since what it closes over leaves with it.
    """
    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        yield from walk(statement)
        return
    stack = [
        child
        for child in ast.iter_child_nodes(statement)
        if not isinstance(child, (ast.stmt, ast.ExceptHandler, ast.match_case))
    ]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(ast.iter_child_nodes(node))


def ways_out(
    func: ast.FunctionDef | ast.AsyncFunctionDef, holder: ast.expr, writes: Sequence[ast.AST] = ()
) -> list[ast.stmt]:
    """The statements of *func* where an object's field can be seen outside it, *writes* aside.

    Where the object -- or what holds it, ``impl`` for ``impl.taskSpec`` -- is
    handed to a call, assigned, put in a container or returned, where a method
    is called on it, and at a ``return``, ``yield`` or ``raise``.  A field of
    it read (``h.x``) or written is not one.  Read wide: a statement that may
    let the value be seen counts, so a write is dropped or given an entry only
    where nothing can have seen it.  The function's normal end is the other
    way out (:func:`replacements`' *returning*).
    """
    held = holder_texts(holder)
    roots = {text for text in held if text.isidentifier()}
    mine = {id(one) for one in writes}
    found: list[ast.stmt] = []
    for statement in _own_statements(func):
        if id(statement) in mine:
            continue
        if isinstance(statement, (ast.Return, ast.Raise)):
            found.append(statement)
            continue
        for node in _evaluated(statement):
            if isinstance(node, (ast.Yield, ast.YieldFrom)):
                found.append(statement)
                break
            if not isinstance(node, (ast.Name, ast.Attribute, ast.Subscript)):
                continue
            if not isinstance(getattr(node, "ctx", None), ast.Load):
                continue
            root = node
            while isinstance(root, (ast.Attribute, ast.Subscript)):
                root = root.value
            if not (isinstance(root, ast.Name) and root.id in roots) or ast.unparse(node) not in held:
                continue
            parent = getattr(node, "parent", None)
            if isinstance(parent, (ast.Attribute, ast.Subscript)) and parent.value is node:
                called = getattr(parent, "parent", None)
                if not (isinstance(parent, ast.Attribute) and isinstance(called, ast.Call) and called.func is parent):
                    # A field or an item of it read, or written: not the object itself.
                    continue
            found.append(statement)
            break
    return found


def renewals(func: ast.FunctionDef | ast.AsyncFunctionDef, holder: ast.expr) -> list[ast.AST]:
    """What gives *holder* another object: a binding of a name it is read through, an assignment to it or to what holds it.

    ``for file in ...:`` on its next pass, ``job = ...``, ``self.job = ...``
    for ``self.job``, and ``tmp_name``'s next value for
    ``self.dataset_map[tmp_name]``.  ``+=`` keeps the object.
    """
    texts = holder_texts(holder) - {name.id for name in walk(holder) if isinstance(name, ast.Name)}
    found: list[ast.AST] = []
    for name in {node.id for node in walk(holder) if isinstance(node, ast.Name)}:
        found.extend(
            binding.site
            for binding in bindings(func, name)
            if not binding.nested and binding.kind != BIND_AUGMENTED
        )
    for statement in _own_statements(func):
        if any(ast.unparse(target) in texts for target in targets_of(statement)):
            found.append(statement)
    return _unique(found)


def field_writes(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    holder: ast.expr,
    attribute: str,
    callee: Optional[Callable[[ast.Call], Optional[Callee]]] = None,
) -> tuple[list[ast.stmt], list[ast.stmt], dict[int, Callee]]:
    """``(writes, rereads, calls)``: the statements of *func* writing ``<holder>.<attribute>``.

    ``h.f = v`` and ``h.f += v`` (a reread, which keeps what was there), and
    a call on the holder of a method *callee* says writes the field -- those
    are in *calls*, by statement, with what it says.
    """
    place = f"{ast.unparse(holder)}.{attribute}"
    text = ast.unparse(holder)
    writes: list[ast.stmt] = []
    rereads: list[ast.stmt] = []
    calls: dict[int, Callee] = {}
    for statement in _own_statements(func):
        if isinstance(statement, ast.AugAssign):
            if ast.unparse(statement.target) == place:
                writes.append(statement)
                rereads.append(statement)
            continue
        if any(ast.unparse(target) == place for target in targets_of(statement)):
            writes.append(statement)
            continue
        if callee is None:
            continue
        for node in _evaluated(statement):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and ast.unparse(node.func.value) == text
            ):
                called = callee(node)
                if called is not None and called.writes:
                    writes.append(statement)
                    calls[id(statement)] = called
                    break
    return writes, rereads, calls


def field_replacements(
    write: ast.AST,
    holder: ast.expr,
    attribute: str,
    value: Optional[ast.expr] = None,
    callee: Optional[Callable[[ast.Call], Optional[Callee]]] = None,
    memo: Optional[dict] = None,
) -> Optional[list[Replacement]]:
    """:func:`replacements` for a write of an object's field: the route to where the object leaves.

    The other writes of the field on the same holder, by its text, its setter
    calls among them (*callee*), are the place's writes; the holder's
    :func:`ways_out` and the function's end its reads; its :func:`renewals`
    give it another object -- a write after one is to that other object --
    and are reads too, since the object it held may be held elsewhere: a
    parameter, an element of the list a ``for`` takes it from.  At module level, or on a holder that is no chain
    of names, fields and items, ``[]``: read as before.  *memo* keeps what one
    function's holder and field give, for a caller asking it of each write.
    """
    held = _field_held(write, holder, attribute, callee, memo)
    if held is None:
        return []
    writes, rereads, calls, out, renewed = held
    return replacements(
        write,
        [one for one in writes if one is not write],
        _unique([*out, *renewed]),
        place=f"{ast.unparse(holder)}.{attribute}",
        value=value,
        killers=renewed,
        rereads=rereads,
        # A setter call writing the field only under the method's tests may leave it as it was.
        partial=[one for one in writes if id(one) in calls and all(calls[id(one)].writes)],
        returning=True,
        callee=lambda site: calls.get(id(site)),
    )


def _field_held(
    write: ast.AST,
    holder: ast.expr,
    attribute: str,
    callee: Optional[Callable[[ast.Call], Optional[Callee]]],
    memo: Optional[dict],
) -> Optional[tuple[list[ast.stmt], list[ast.stmt], dict[int, Callee], list[ast.AST], list[ast.AST]]]:
    """``(writes, rereads, calls, ways out, renewals)`` of the field *write* writes, or ``None`` at module level or
    on a holder that is no chain of names, fields and items."""
    func = enclosing_function(write)
    if func is None:
        return None
    root = holder
    while isinstance(root, (ast.Attribute, ast.Subscript)):
        root = root.value
    if not isinstance(root, ast.Name):
        return None
    key = (id(func), ast.unparse(holder), attribute)
    held = memo.get(key) if memo is not None else None
    if held is None:
        writes, rereads, calls = field_writes(func, holder, attribute, callee)
        if not any(one is write for one in writes):
            writes.append(write)  # type: ignore[arg-type]
        held = (writes, rereads, calls, ways_out(func, holder, writes), renewals(func, holder))
        if memo is not None:
            memo[key] = held
    return held


def additions(
    write: ast.AST,
    writes: Sequence[ast.AST],
    reads: Sequence[ast.AST],
    *,
    place: str,
    killers: Sequence[ast.AST] = (),
    returning: bool = False,
) -> list[ast.AugAssign]:
    """The ``+=`` of *place* among *writes* that add to what *write* puts there and whose own value leaves.

    Each is a write of what the function makes of the value: past a ``+=`` the place holds another value
    (:func:`replacements`), and where the value is still there when one runs and what that one makes reaches a
    read -- one of *reads*, or with *returning* the function's end -- that is what leaves.  ``setDatasetAttribute``
    sets ``self.attributes = ""`` and then ``self.attributes += attr``: the method writes the sum.  *writes* are the
    place's writes; the ``+=`` among them keep what was there, the others overwrite.
    """
    added = [one for one in writes if isinstance(one, ast.AugAssign) and one is not write]
    sites = _unique([write, *writes])
    return [
        one
        for one in added
        if any(site is write for site in surviving_writes(sites, one, added, as_run=True, killers=killers))
        and replacements(
            one,
            [other for other in writes if other is not one],
            reads,
            place=place,
            killers=killers,
            rereads=added,
            returning=returning,
        )
        is not None
    ]


def field_additions(
    write: ast.AST,
    holder: ast.expr,
    attribute: str,
    callee: Optional[Callable[[ast.Call], Optional[Callee]]] = None,
    memo: Optional[dict] = None,
) -> list[ast.AugAssign]:
    """:func:`additions` for a write of an object's field, the field's ``+=`` read on the route
    :func:`field_replacements` reads: to where the object leaves."""
    held = _field_held(write, holder, attribute, callee, memo)
    if held is None:
        return []
    writes, _rereads, _calls, out, renewed = held
    return additions(
        write,
        writes,
        _unique([*out, *renewed]),
        place=f"{ast.unparse(holder)}.{attribute}",
        killers=renewed,
        returning=True,
    )

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
