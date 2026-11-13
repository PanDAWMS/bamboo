"""Every way one call of a function can take through a region of its statements, one statement at a time.

The walker the querier's P reads a run's rows with, moved into the map so that its readers walk one way.  A region
is an ``if`` with the statements after it in its block (:func:`region_of`), and a *path* (:class:`Path`) is one
way through it with what it knows: the values it bound, the facts the tests it took settled, the rows it passed
and how it left.

* **Branches.**  An ``if`` goes both ways, or the ways the reader's *decide* leaves; a way taken keeps what its
  test settles of the atoms the walk reads (:meth:`Walker.settle`).
* **Leaving.**  A ``return``, ``raise``, ``break`` or ``continue`` ends the path (``left``), and a statement that
  can raise, as the map reads exceptions (:func:`reaching.may_raise`: any expression but a literal, a name or a
  display of them, an unpacking, an ``assert``, ``del`` or ``import`` -- a logger's call too), forks a copy that
  leaves there, ``left`` saying ``call at <line>`` (:meth:`Walker.raises`).  Handed *raised*, the handlers whose
  marks a run shows, a statement leaves so only where its raise reaches one of them, whatever it holds
  (:func:`reaching.catching_try`): a run that shows none ran every ``try`` to its end.
* **Passed whole.**  A loop, a ``try`` or a ``match`` is one step: what it binds becomes unknown, the rows in it
  are passed in their order, and a copy leaves where it can divert.  A ``with`` is entered.
* **Calls.**  A call changes what it is handed, as the map reads the call (:func:`changes.call_changes`): a value
  or a fact the path holds of a chain whose name the call is handed or called on (:func:`mutating_roots`) is dropped
  where the call may change that name's object at the chain's first step -- at any key for a bare name, and for the
  receiver of a method a test calls (``task.useScout()``).  It stays where the object cannot change (its stated
  type, a value or a fact the path holds) and where the test reads only the chain's identity (:func:`readings`): a
  change inside the chain's own value leaves that, and no call binds its caller's name anew.  What a call may
  change marks the path as a binding does (:func:`change_mark`), by the same reading: ``r.a`` where it stores
  ``r.a``, ``r.a()`` where it may change what ``r.a`` holds, ``r()`` where it may change any of ``r``; a call that
  changes nothing of a name marks nothing.
* **Rows.**  The reader's *order* -- the rows it asks of, by line, each with its rank -- is passed in that order: a
  path passing a row out of it, or missing one where *require_all*, is dropped, and the reader's *check_row* may
  drop it at each.
* **Many paths.**  After each statement a path keeps only the facts a statement walked after it can read
  (:meth:`Walker.drop_unread`): a later test looks it up, or a later call may be asked about the name it says
  cannot change.  Equal paths are one (:func:`merge`), so paths that differ only in what nobody reads are one.
  Beyond :data:`WIDEN_AT`, paths that agree on what the
  reader reads are joined into one that knows only what all of them knew (:func:`widen`): joining can lose a
  verdict, never make one.  Beyond :data:`MAX_PATHS` the walk stops (:class:`TooMany`).

Values are read with :mod:`bamboo.codemap.evaluate`.  A reader adds its own rules by subclassing
:class:`Walker` and :class:`Path` at the methods marked as hooks: the querier adds its printed holes, a target
held as an expression, rows rendered by the path and the calls it reads against a run.  A value a path knows only
in part is a :class:`Partial`.

The statements walked must be of one function of the corpus the walker's :class:`~bamboo.codemap.calls.Calls`
reads, on the tree it reads (:func:`bamboo.codemap.calls.parsed`, parents attached): the map finds a definition by
its node.
"""

from __future__ import annotations

import ast
import copy
import functools
import weakref
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Collection, Iterable, Optional

from bamboo.codemap import callfacts, evaluate, pathcond, reaching
from bamboo.codemap.callfacts import LOG_LEVELS
from bamboo.codemap.models import (
    CHANGE_DELETE,
    CHANGE_INNER,
    CHANGE_STORE,
    CHANGE_UNREAD,
)
from bamboo.codemap.panda import changes
from bamboo.codemap.panda.recognizers.selection import log_level

if TYPE_CHECKING:
    from bamboo.codemap.calls import Calls

#: The walk stops where one statement leaves more paths than this (:class:`TooMany`).
MAX_PATHS = 3000
#: Beyond this many paths after one statement, paths that agree on what the reader reads are joined (:func:`widen`).
WIDEN_AT = 400
#: The statements passed whole, as one step.
OPAQUE = (ast.For, ast.AsyncFor, ast.While, ast.Try, ast.Match) + (
    (ast.TryStar,) if hasattr(ast, "TryStar") else ()
)
NAMED_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
#: The nodes whose body is not the walked function's own.
DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
LOOPS = (ast.For, ast.AsyncFor, ast.While)
JUMPS = (ast.Return, ast.Raise, ast.Continue, ast.Break)
#: Builtins that read their arguments and change nothing (a call of one still may raise).
READERS = frozenset(
    {
        "str",
        "len",
        "int",
        "float",
        "bool",
        "repr",
        "sorted",
        "list",
        "dict",
        "set",
        "tuple",
        "min",
        "max",
        "sum",
        "any",
        "all",
        "isinstance",
        "type",
        "format",
        "abs",
        "round",
        "enumerate",
        "zip",
        "range",
        "hasattr",
        "frozenset",
    }
)
#: The source of a value the path bound itself, in the walk's :class:`~bamboo.codemap.evaluate.Env`.
BOUND_ON_PATH = "bound on the path"
#: How a test reads a chain (:func:`readings`): by its identity -- ``is``, against a literal, its type, as a key --,
#: which only binding the chain anew changes; or by what it holds, which a change in place changes too.
IDENTITY, CONTENT = "identity", "content"
#: The builtins a test reads its arguments' identity or type with -- ``isinstance(x, str)``.
IDENTITY_TESTS = frozenset({"isinstance", "issubclass", "type", "id", "callable"})
#: The types of a value the path knows that no change in place can touch.
_IMMUTABLE_VALUES = (str, int, float, bool, bytes, complex, type(None))


class _NotBound:
    def __repr__(self) -> str:
        return "NOT_BOUND"


#: What a tracked row records of the target where the path has not bound it.
NOT_BOUND: Any = _NotBound()


class TooMany(Exception):
    """A statement left more than :data:`MAX_PATHS` paths."""


class Partial:
    """A value a path knows only in part: read as not known where a test or a value reads it."""


# ---------------------------------------------------------------- the function's own statements


def own_nodes(root: ast.AST) -> Iterable[ast.AST]:
    """``ast.walk`` that does not go into nested functions, classes or lambdas."""
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, DEFS):
                stack.append(child)


def contains(stmt: ast.AST, kinds: tuple) -> bool:
    return any(isinstance(n, kinds) for n in own_nodes(stmt))


def block_of(parent: ast.AST, child: ast.AST) -> Optional[tuple[str, list]]:
    """The block of *parent* holding *child*, and its name."""
    for name in ("body", "orelse", "finalbody", "handlers"):
        block = getattr(parent, name, None)
        if isinstance(block, list) and child in block:
            return name, block
    return None


def is_log_call(node: ast.AST) -> bool:
    """Whether *node* is a logger's call writing one line at a level (:data:`~bamboo.codemap.callfacts.LOG_LEVELS`)."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in LOG_LEVELS
        and bool(node.args)
        and log_level(node.args[0]) is not None
    )


def may_divert(
    stmt: ast.stmt, raised: Optional[Collection[ast.ExceptHandler]] = None
) -> bool:
    """Whether control can leave *stmt* other than by falling through it: a jump in it, or a statement in it that
    can raise (:func:`can_raise`) -- handed *raised*, the handlers a run entered, one whose raise reaches one of
    them in a ``try`` around *stmt* (:func:`reaching.catching_try`)."""
    if isinstance(stmt, JUMPS) or contains(stmt, JUMPS):
        return True
    if raised is None:
        return any(isinstance(n, ast.stmt) and can_raise(n) for n in own_nodes(stmt))
    around = {id(node) for node in pathcond.ancestors(stmt)}
    for node in own_nodes(stmt):
        if isinstance(node, ast.stmt):
            caught = reaching.catching_try(node, raised)
            if caught is not None and id(caught) in around:
                return True
    return False


# ---------------------------------------------------------------- reading expressions


@functools.lru_cache(maxsize=None)
def chains(text: str) -> frozenset[str]:
    """The name / attribute / subscript chains *text* reads, each as long as it goes."""
    try:
        node = ast.parse(text, mode="eval").body
    except SyntaxError:
        return frozenset()
    return frozenset(node_chains(node))


def node_chains(node: ast.AST) -> set[str]:
    """The chains *node* reads (:func:`chains`)."""
    out: set[str] = set()
    if isinstance(node, (ast.Name, ast.Attribute, ast.Subscript)):
        out.add(ast.unparse(node))
        cur: ast.AST = node
        while True:
            if isinstance(cur, ast.Attribute):
                cur = cur.value
            elif isinstance(cur, ast.Subscript):
                out |= node_chains(cur.slice)
                cur = cur.value
            elif isinstance(cur, ast.Call):
                for a in [*cur.args, *(k.value for k in cur.keywords)]:
                    out |= node_chains(a)
                cur = cur.func
            else:
                break
        if not isinstance(cur, ast.Name):
            out |= node_chains(cur)
        return out
    for child in ast.iter_child_nodes(node):
        out |= node_chains(child)
    return out


@functools.lru_cache(maxsize=None)
def root(chain: str) -> str:
    """The name *chain* starts at; empty where it starts at no name."""
    try:
        node = ast.parse(chain, mode="eval").body
    except SyntaxError:
        return ""
    while isinstance(node, (ast.Attribute, ast.Subscript, ast.Call)):
        node = node.func if isinstance(node, ast.Call) else node.value
    return node.id if isinstance(node, ast.Name) else ""


def affects(bound: str, read: frozenset[str] | set[str]) -> bool:
    """Whether binding *bound* changes what any of the chains *read* holds."""
    return any(
        c == bound
        or c.startswith((bound + ".", bound + "["))
        or bound.startswith((c + ".", c + "["))
        for c in read
    )


def change_mark(name: str, change: changes.ObjectChange) -> str:
    """The mark a call's *change* of the object *name* holds puts on a path, beside the texts a binding marks:
    ``name.a`` (``name['k']``) where the call stores or deletes that key, ``name.a()`` where it may change what the
    key holds, ``name()`` where it may change the object at any key."""
    if change.key == changes.EVERY:
        return f"{name}()"
    step = change.key if change.key.startswith(".") else f"[{change.key}]"
    if change.kind in (CHANGE_STORE, CHANGE_DELETE):
        return name + step
    return f"{name}{step}()"


def _flat(target: ast.expr) -> list[ast.expr]:
    if isinstance(target, (ast.Tuple, ast.List)):
        return [t for e in target.elts for t in _flat(e)]
    if isinstance(target, ast.Starred):
        return _flat(target.value)
    return [target]


def binds(stmt: ast.AST) -> list[tuple[str, Optional[ast.expr]]]:
    """``(text bound, value node or None)`` for what this one node binds, not what is nested in it."""
    out: list[tuple[str, Optional[ast.expr]]] = []
    if isinstance(stmt, ast.Assign):
        for t in stmt.targets:
            single = not isinstance(t, (ast.Tuple, ast.List))
            out += [(ast.unparse(x), stmt.value if single else None) for x in _flat(t)]
    elif isinstance(stmt, ast.AugAssign):
        out.append((ast.unparse(stmt.target), None))
    elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
        out.append((ast.unparse(stmt.target), stmt.value))
    elif isinstance(stmt, ast.Delete):
        out += [(ast.unparse(x), None) for t in stmt.targets for x in _flat(t)]
    elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
        out += [((a.asname or a.name).split(".")[0], None) for a in stmt.names]
    elif isinstance(stmt, NAMED_DEFS):
        out.append((stmt.name, None))
    elif isinstance(stmt, (ast.For, ast.AsyncFor)):
        out += [(ast.unparse(x), None) for x in _flat(stmt.target)]
    elif isinstance(stmt, (ast.With, ast.AsyncWith)):
        out += [
            (ast.unparse(x), None)
            for i in stmt.items
            if i.optional_vars is not None
            for x in _flat(i.optional_vars)
        ]
    elif isinstance(stmt, ast.ExceptHandler) and stmt.name:
        out.append((stmt.name, None))
    elif isinstance(stmt, ast.NamedExpr):
        out.append((ast.unparse(stmt.target), None))
    return out


def own_binds(stmt: ast.AST) -> list[tuple[str, Optional[ast.expr]]]:
    """What a simple statement binds, walrus included."""
    out = binds(stmt)
    out += [
        (ast.unparse(n.target), None)
        for n in own_nodes(stmt)
        if isinstance(n, ast.NamedExpr)
    ]
    return out


def deep_binds(stmt: ast.AST) -> set[str]:
    """What *stmt* binds, what is nested in it included (not in a nested definition)."""
    return {t for n in own_nodes(stmt) for t, _ in binds(n)}


def text_of(node: ast.AST) -> str:
    """``ast.unparse`` of *node*, kept on it: every path through a test reads it again."""
    held = getattr(node, "_walker_text", None)
    if held is None:
        held = ast.unparse(node)
        node._walker_text = held  # type: ignore[attr-defined]
    return held


def can_raise(node: ast.AST) -> bool:
    """Whether *node* -- a statement, or an expression the walk reads alone (a test, a context expression) -- can
    raise before it is done, as the map reads exceptions (:func:`reaching.may_raise`)."""
    return reaching.may_raise(node)


def mutating_roots(
    node: ast.AST, kept: Optional[Callable[[ast.Call], set[int]]] = None
) -> set[str]:
    """Names a call in *node* is handed or called on -- what it could change.  *kept*: the ids of the ``Name``
    nodes of a call that the call only reads."""
    out: set[str] = set()
    for n in own_nodes(node):
        if isinstance(n, ast.Call) and not is_log_call(n):
            if isinstance(n.func, ast.Name) and n.func.id in READERS:
                continue
            read_only = kept(n) if kept is not None else set()
            out |= {
                m.id
                for m in ast.walk(n)
                if isinstance(m, ast.Name) and id(m) not in read_only
            }
    return out


@functools.lru_cache(maxsize=None)
def expression(text: str) -> Optional[ast.expr]:
    """*text* parsed as an expression, ``None`` where it is not one: read once per text."""
    try:
        return ast.parse(text, mode="eval").body
    except SyntaxError:
        return None


@functools.lru_cache(maxsize=None)
def readings(text: str) -> tuple[tuple[ast.expr, str], ...]:
    """``(chain, how)`` for each chain rooted at a name that the test *text* reads, *how* :data:`IDENTITY` or
    :data:`CONTENT`: ``job.status == 'a'`` reads the identity of ``job.status``, ``if items:`` the content of
    ``items``, and a method the test calls -- ``task.useScout()`` -- its receiver's content.  A function called by
    its name reads nothing of its own."""
    body = expression(text)
    if body is None:
        return ()
    parents = {
        child: node for node in ast.walk(body) for child in ast.iter_child_nodes(node)
    }
    out: list[tuple[ast.expr, str]] = []
    for node in ast.walk(body):
        if not isinstance(node, (ast.Name, ast.Attribute, ast.Subscript)):
            continue
        up = parents.get(node)
        if isinstance(up, (ast.Attribute, ast.Subscript)) and up.value is node:
            continue
        if changes.chain_steps(node) is None:
            continue
        if isinstance(up, ast.Call) and up.func is node:
            if isinstance(node, ast.Attribute):
                out.append((node.value, CONTENT))
            elif isinstance(node, ast.Subscript):
                out.append((node, CONTENT))
            continue
        out.append((node, _how(node, up)))
    return tuple(out)


def _how(node: ast.expr, up: Optional[ast.AST]) -> str:
    """How the expression *up* reads *node*, one of its operands (:func:`readings`)."""
    if isinstance(up, ast.Compare):
        operands = [up.left, *up.comparators]
        at = next(i for i, one in enumerate(operands) if one is node)
        around = [(up.ops[at - 1], operands[at - 1], False)] if at else []
        around += [(up.ops[at], operands[at + 1], True)] if at < len(up.ops) else []
        for op, other, left in around:
            if isinstance(op, (ast.Is, ast.IsNot)):
                continue
            if isinstance(op, (ast.In, ast.NotIn)):
                if left and _literals(other):
                    continue
                return CONTENT
            if not _literal(other):
                return CONTENT
        return IDENTITY
    if (
        isinstance(up, ast.Call)
        and isinstance(up.func, ast.Name)
        and up.func.id in IDENTITY_TESTS
    ):
        return IDENTITY
    if isinstance(up, ast.Subscript) and up.slice is node:
        return IDENTITY
    return CONTENT


def _literal(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) or (
        isinstance(node, ast.UnaryOp) and isinstance(node.operand, ast.Constant)
    )


def _literals(node: ast.expr) -> bool:
    return isinstance(node, (ast.Tuple, ast.List, ast.Set)) and all(
        _literal(one) for one in node.elts
    )


def value_reading(t: str, v: Any) -> tuple[tuple[ast.expr, str], ...]:
    """``(chain, how)`` for the chain *t* holding *v* on a path: a container read by what it holds, a value not
    known too, any other value by its identity."""
    node = expression(t)
    if node is None:
        return ()
    how = (
        CONTENT
        if v is evaluate.UNKNOWN or isinstance(v, (list, dict, set))
        else IDENTITY
    )
    return ((node, how),) if changes.chain_steps(node) is not None else ()


def stated_immutable(
    calls: "Calls", file: str, func: ast.AST, name: str, bound: tuple = ()
) -> bool:
    """Whether the code states that *name* holds, everywhere in *func* of *file*, a value no change in place can touch:
    a parameter annotated so (:func:`callfacts.immutable_annotation`), and each binding of it annotated so, a
    literal, an f-string, a comparison, a ``not``, a builtin giving a text or a number
    (:data:`callfacts.BUILTIN_FUNCTIONS`), or a call of definitions each annotated to return one."""
    said: list[bool] = []
    arguments: Optional[ast.arguments] = getattr(func, "args", None)
    if arguments is not None:
        named = [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]
        parameter = next((one for one in named if one.arg == name), None)
        if parameter is not None:
            said.append(callfacts.immutable_annotation(parameter.annotation))
        if any(
            one is not None and one.arg == name
            for one in (arguments.vararg, arguments.kwarg)
        ):
            said.append(False)
    for binding in pathcond.bindings(func, name):
        if binding.nested:
            continue
        if isinstance(binding.site, ast.AnnAssign):
            said.append(callfacts.immutable_annotation(binding.site.annotation))
        elif binding.value is None or binding.slot is not None:
            said.append(False)
        else:
            said.append(_gives_immutable(calls, file, func, binding.value, bound))
    return bool(said) and all(said)


def _gives_immutable(
    calls: "Calls", file: str, func: ast.AST, value: ast.expr, bound: tuple
) -> bool:
    if isinstance(value, (ast.Constant, ast.JoinedStr, ast.Compare)):
        return True
    if isinstance(value, ast.UnaryOp):
        return isinstance(value.op, ast.Not) or _gives_immutable(
            calls, file, func, value.operand, bound
        )
    if isinstance(value, ast.BoolOp):
        return all(
            _gives_immutable(calls, file, func, one, bound) for one in value.values
        )
    if isinstance(value, ast.IfExp):
        return all(
            _gives_immutable(calls, file, func, one, bound)
            for one in (value.body, value.orelse)
        )
    if not isinstance(value, ast.Call):
        return False
    function = callfacts.builtin_function(calls.types, file, func, value)
    if function is not None:
        return function.result in callfacts.SCALARS
    try:
        placed = calls.callees(file, func, value, bound)
    except Exception:  # noqa: BLE001 -- a placing that fails says nothing
        return False
    targets = list(placed.targets)
    return (
        bool(targets)
        and not placed.unplaced
        and all(
            callfacts.immutable_annotation(getattr(one.func, "returns", None))
            for one in targets
        )
    )


def _immutable_value(v: Any) -> bool:
    if isinstance(v, _IMMUTABLE_VALUES):
        return True
    return isinstance(v, (tuple, frozenset)) and all(_immutable_value(one) for one in v)


def _fact_immutable(name: str, facts: dict[str, bool]) -> bool:
    """Whether a fact says *name* holds a value no change in place can touch: ``n == 'a'``, ``n is None``, ``n in
    ('a', 'b')`` held, or ``isinstance(n, str)``."""
    for atom, truth in facts.items():
        body = expression(atom)
        if (
            isinstance(body, ast.Compare)
            and len(body.ops) == 1
            and isinstance(body.left, ast.Name)
            and body.left.id == name
        ):
            op, other = body.ops[0], body.comparators[0]
            positive = isinstance(op, (ast.Eq, ast.Is, ast.In))
            holds = truth is True if positive else truth is False
            if (
                holds
                and isinstance(op, (ast.Eq, ast.NotEq, ast.Is, ast.IsNot))
                and _literal(other)
            ):
                return True
            if holds and isinstance(op, (ast.In, ast.NotIn)) and _literals(other):
                return True
        if (
            isinstance(body, ast.Call)
            and isinstance(body.func, ast.Name)
            and body.func.id == "isinstance"
            and truth is True
            and len(body.args) == 2
            and isinstance(body.args[0], ast.Name)
            and body.args[0].id == name
            and callfacts.immutable_annotation(body.args[1])
        ):
            return True
    return False


def lookups(test: ast.AST) -> set[str]:
    """The texts :func:`evaluate.truth` looks *test* up by among a path's facts: the test itself, the values of an
    ``and`` / ``or`` and the operand of a ``not``, through each other."""
    found = {evaluate.text_of(test)}
    if isinstance(test, ast.BoolOp):
        for one in test.values:
            found |= lookups(one)
    elif isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        found |= lookups(test.operand)
    return found


@functools.lru_cache(maxsize=None)
def witnessed(atom: str, truth: bool) -> frozenset[str]:
    """The names a fact *atom* holding *truth* says hold a value no change in place can touch
    (:func:`_fact_immutable`)."""
    body = expression(atom)
    if body is None:
        return frozenset()
    return frozenset(
        n.id
        for n in ast.walk(body)
        if isinstance(n, ast.Name) and _fact_immutable(n.id, {atom: truth})
    )


class _Asked:
    """What the map said of a corpus's calls and names, kept while the corpus's calls live: each answer with the
    node it was asked of, so that its id is not reused."""

    def __init__(self) -> None:
        self.changes: dict[tuple, tuple[ast.AST, list[changes.ObjectChange]]] = {}
        self.stated: dict[tuple, tuple[ast.AST, bool]] = {}


_ASKED: "weakref.WeakKeyDictionary[Calls, _Asked]" = weakref.WeakKeyDictionary()


def _asked(calls: "Calls") -> _Asked:
    if calls not in _ASKED:
        _ASKED[calls] = _Asked()
    return _ASKED[calls]


def below(t: str, v: Any, handed: str) -> bool:
    """Whether a call handed *handed* can change what *t* holds (*v*, None where not known): a chain below it, or
    *handed* itself holding a container -- an attribute chain whose value is not known counts as one."""
    if t.startswith((handed + ".", handed + "[")):
        return True
    if t != handed:
        return False
    return (
        isinstance(v, (list, dict, set)) if v is not None else not handed.isidentifier()
    )


# ---------------------------------------------------------------- the region


def chain_of_arm(func: ast.AST, stmt: ast.stmt) -> list[ast.If]:
    """The ``if``s the arm sits under through bodies and ``else``s only, innermost first."""
    found = []
    node: ast.AST = stmt
    while node is not func:
        parent = node.parent  # type: ignore[attr-defined]
        if isinstance(parent, ast.If) and (
            node in parent.body or node in parent.orelse
        ):
            found.append(parent)
        elif not (
            isinstance(parent, (ast.With, ast.AsyncWith)) and node in parent.body
        ):
            break
        node = parent
    return found


def region_of(s_star: ast.stmt) -> list[ast.stmt]:
    """*s_star* and the statements after it in its block."""
    placed = block_of(s_star.parent, s_star)  # type: ignore[attr-defined]
    if placed is None:
        return []
    _, block = placed
    return block[block.index(s_star) :]


def in_region(node: ast.AST, region: list[ast.stmt]) -> Optional[ast.stmt]:
    """The region statement *node* is in, or None."""
    ids = {id(s): s for s in region}
    for a in [node, *pathcond.ancestors(node)]:
        if id(a) in ids:
            return ids[id(a)]
    return None


def _place(node: ast.AST, region: list[ast.stmt]) -> int:
    """The position in *region* of the statement *node* is in; ValueError where it is in none."""
    found = in_region(node, region)
    if found is None:
        raise ValueError(f"{ast.unparse(node)[:60]!r} is not in the region")
    return region.index(found)


def target_of(stmt: ast.stmt) -> str:
    """What *stmt* assigns, where it assigns one name, attribute or item; empty otherwise."""
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
        t = stmt.targets[0]
    elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
        t = stmt.target
    else:
        return ""
    return (
        ast.unparse(t)
        if isinstance(t, (ast.Name, ast.Attribute, ast.Subscript))
        else ""
    )


def unbound_before(func: ast.AST, s_star: ast.stmt, name: str) -> bool:
    """Whether a bare local can hold nothing at S*: only the function's entry reaches it, and the name is no
    parameter (:func:`~bamboo.codemap.pathcond.held_at`)."""
    if not name.isidentifier():
        return False
    held = pathcond.held_at(func, name, s_star)  # type: ignore[arg-type]
    return held.kind == pathcond.HELD_UNBOUND


def entry_state(func: ast.AST, s_star: ast.stmt, names: set[str]) -> dict[str, Any]:
    """Constants bare locals hold at S*: the one literal every way to it binds last
    (:func:`~bamboo.codemap.pathcond.held_at`).

    ``isOK = True`` four lines above ``if not toSkip: if isOK:`` holds at the
    inner ``if`` whatever ran before, and a binding before a loop around S*
    holds as long as nothing in the loop binds the name again.
    """
    out: dict[str, Any] = {}
    for name in names:
        if name.isidentifier():
            held = pathcond.held_at(func, name, s_star)  # type: ignore[arg-type]
            if held.kind == pathcond.HELD_CONSTANT:
                out[name] = held.value
    return out


# ---------------------------------------------------------------- paths


@dataclass
class Path:
    """One way through the region and what it knows.  A reader holding more on a path subclasses it, and extends
    :meth:`key`'s, :meth:`agree`'s and :meth:`join`'s readings with what it adds."""

    #: texts bound on the path -> value (:data:`~bamboo.codemap.evaluate.UNKNOWN` when not known)
    state: dict[str, Any] = field(default_factory=dict)
    arm: bool = False
    left: str = ""
    seen: tuple[int, ...] = ()
    #: row line -> what was bound or handed to a call after it; a reader may keep other entries beside the rows'
    since: dict[int, frozenset[str]] = field(default_factory=dict)
    marks: frozenset[str] = frozenset()
    #: (test, outcome, state then) for tests that read the target while the path has not bound it
    prior: tuple = ()
    #: a call was handed the target (an attribute chain) since the path bound it
    target_called: bool = False
    tests: tuple = ()
    at_row: tuple = ()
    #: atom -> truth a branch taken earlier on the path settled (``not toSkip`` false: ``toSkip`` true)
    facts: dict[str, bool] = field(default_factory=dict)

    def copy(self) -> "Path":
        q = copy.copy(self)
        q.state = dict(self.state)
        q.since = dict(self.since)
        q.facts = dict(self.facts)
        return q

    def key(self) -> str:
        """Equal for equal paths (:func:`merge`)."""
        return repr(self.key_parts())

    def key_parts(self) -> tuple:
        # since's sets are sorted: a set prints in an order that depends on the hash seed and on how it was
        # built, and two equal paths whose sets printed differently did not merge (PYTHONHASHSEED moved
        # census_p's path counts).  Values go through value_key; the rest is repr, which is fast.
        return (
            self.arm,
            bool(self.left),
            self.left.split(" ")[0] if self.left else "",
            self.seen,
            sorted((k, value_key(v)) for k, v in self.state.items()),
            sorted((line, sorted(m)) for line, m in self.since.items()),
            sorted(self.marks),
            self.prior,
            self.target_called,
            self.at_row,
            sorted(self.facts.items()),
        )

    def rough_key(self) -> tuple:
        """A part of :meth:`key_parts`, cheap to read: paths it tells apart are not equal.  A reader's
        :meth:`key_parts` extends the base's, so this stays a part of it."""
        return (
            self.arm,
            self.left.split(" ")[0] if self.left else "",
            self.seen,
            len(self.state),
            len(self.facts),
            len(self.marks),
            self.target_called,
        )

    def agree(self, target: str) -> tuple:
        """What paths :func:`widen` joins must agree on: the arm, the rows passed, how they left and the target."""
        return (
            self.arm,
            self.left.split(" ")[0] if self.left else "",
            self.seen,
            self.at_row,
            target in self.state,
        )

    def join(self, q: "Path") -> None:
        """Keep what this path and *q* both know: a value that differs becomes unknown, a fact or a constraint only
        one of them had is dropped, a binding or a call any of them made counts."""
        keys = set(self.state) | set(q.state)
        self.state = {
            k: self.state[k]
            if k in self.state
            and k in q.state
            and value_key(self.state[k]) == value_key(q.state[k])
            else evaluate.UNKNOWN
            for k in keys
        }
        self.since = {
            line: self.since.get(line, frozenset()) | q.since.get(line, frozenset())
            for line in set(self.since) | set(q.since)
        }
        self.marks |= q.marks
        self.target_called = self.target_called or q.target_called
        self.facts = {k: v for k, v in self.facts.items() if q.facts.get(k) == v}
        self.prior = tuple(x for x in self.prior if x in q.prior)


def canon(v: Any) -> str:
    """``repr`` with sets and dicts in sorted order: equal values give one text.

    A set iterates in an order that depends on the hash seed and on how it was
    built, so ``repr`` of two equal sets can differ -- and two equal paths
    would not merge (PYTHONHASHSEED moved census_p's path counts).
    """
    if isinstance(v, (set, frozenset)):
        return f"{type(v).__name__}({sorted(canon(x) for x in v)})"
    if isinstance(v, dict):
        return (
            "{"
            + ", ".join(sorted(f"{canon(k)}: {canon(x)}" for k, x in v.items()))
            + "}"
        )
    if isinstance(v, (list, tuple)):
        inner = ", ".join(canon(x) for x in v)
        return (
            f"[{inner}]"
            if isinstance(v, list)
            else f"({inner}{',' if len(v) == 1 else ''})"
        )
    return repr(v)


def value_key(v: Any) -> str:
    """``repr`` of a value, through :func:`canon` only when it is a container (one may hold a set)."""
    return canon(v) if isinstance(v, (set, frozenset, dict, list, tuple)) else repr(v)


def merge(paths: list[Path]) -> list[Path]:
    """One of each set of equal paths, the first.  Paths are told apart first by what is cheap to read of them
    (:meth:`Path.rough_key`), and only paths alike in that are compared whole (:meth:`Path.key`)."""
    if len(paths) < 2:
        return paths
    rough = [p.rough_key() for p in paths]
    alike = Counter(rough)
    kept: dict[tuple, Path] = {}
    for p, r in zip(paths, rough, strict=True):
        kept.setdefault((r, p.key()) if alike[r] > 1 else (r,), p)
    return list(kept.values())


def widen(paths: list[Path], target: str) -> list[Path]:
    """Join paths that agree on what :meth:`Path.agree` reads, keeping what all of them know (:meth:`Path.join`).

    A joined path knows less, so it can only be dropped by less: joining can
    lose a verdict, never make one.
    """
    groups: dict[str, list[Path]] = {}
    for p in paths:
        groups.setdefault(canon(p.agree(target)), []).append(p)
    out = []
    for group in groups.values():
        base = group[0].copy()
        for q in group[1:]:
            base.join(q)
        out.append(base)
    return out


# ---------------------------------------------------------------- the walk


class Walker:
    """Every path through a region, with the reader's rows as the evidence.

    *target*: what the arm assigns, followed along each path; *relevant*: the
    chains the walk keeps values and facts of; *order*: row line -> (its
    statement, its rank); *decide*: the ways an ``if`` goes on a path;
    *check_row*: why a path cannot pass a row, or ''; *corpus*, *file* and
    *bound*: the corpus's calls, the file of *func* and the classes ``self``
    can be in it (:attr:`calls.Target.bound`; empty for every class), what
    the map reads a call by (:meth:`call`); *track*: record the tests taken
    and the target at each row; *require_all*: a path must pass every row of
    *order*.
    """

    #: The class of the path a walk starts with.
    path_type: type[Path] = Path

    def __init__(
        self,
        func: ast.AST,
        arm: ast.stmt,
        *,
        target: str,
        relevant: set[str],
        order: dict[int, tuple[ast.stmt, int]],
        decide: Callable[[Any, ast.If, Any], list[bool]],
        check_row: Callable[[Any, int, Any], str],
        corpus: "Calls",
        file: str,
        bound: tuple = (),
        track: bool = False,
        require_all: bool = True,
        raised: Optional[Collection[ast.ExceptHandler]] = None,
    ) -> None:
        self.func = func
        #: the handlers whose marks the run shows (:func:`reaching.catching_try`); None reads (a)
        self.raised = raised
        self.corpus = corpus
        self.file = file
        self.bound_classes = bound
        self._bound_key = tuple((where, id(cls)) for where, cls in bound)
        self.arm = arm
        self.target = target
        self.target_attr = bool(target) and not target.isidentifier()
        self.target_root = root(target) if target else ""
        self.relevant = set(relevant) | ({target} if target else set())
        self.rel_roots = {root(c) for c in self.relevant} - {""}
        self.order = order
        self.decide = decide
        self.check_row = check_row
        self.track = track
        self.require_all = require_all
        self.dropped: Counter[str] = Counter()
        self.cuts: Counter[str] = Counter()
        self.widened = 0
        #: the most paths any statement left
        self.most = 0
        self.facts: dict[str, str] = {}
        self.region: list[ast.stmt] = []
        #: each statement of the region the walk steps into: (line, what its test looks facts up by, the names it
        #: hands a call) (:meth:`read_after`)
        self.steps: list[tuple[int, set[str], set[str]]] = []
        self._after: dict[int, tuple[set[str], set[str]]] = {}

    # -- hooks: what a reader adds

    def make_env(self) -> evaluate.Env:
        """The environment a path's values are read in."""
        return evaluate.Env()

    def put_holes(self, env: evaluate.Env, p: Path) -> None:
        """Put in *env* what the reader knows on *p* beyond what the path bound."""

    def raises(self, node: ast.AST) -> bool:
        """Whether a path may leave at *node* by an exception (:func:`can_raise`); handed *raised*, whether a raise
        at *node* reaches one of them (:func:`reaching.catching_try`)."""
        if self.raised is None:
            return can_raise(node)
        return reaching.catching_try(node, self.raised) is not None

    def roots(self, node: ast.AST) -> set[str]:
        """The names the calls in *node* are handed or called on (:func:`mutating_roots`)."""
        return mutating_roots(node)

    def self_calls(self, q: Path, node: ast.AST, roots: set[str]) -> set[str]:
        """Read what the calls in *node* change of the roots the reader can say more of; the roots left to
        :meth:`call`."""
        return roots

    def handed_target(self, q: Path, node: ast.AST) -> None:
        """A call in *node* was handed the target."""

    def value_of(self, q: Path, t: str, vnode: Optional[ast.expr]) -> Any:
        """The value *t* is bound to by *vnode* on *q*."""
        value: Any = evaluate.UNKNOWN
        if isinstance(vnode, ast.Constant):
            value = vnode.value
        elif vnode is not None:
            try:
                value = evaluate.ev(vnode, self.env(q))
            except Exception:
                value = evaluate.UNKNOWN
        return value

    def bound(self, q: Path, t: str, value: Any) -> None:
        """*t* was bound to *value* on *q*."""

    def appended(self, s: ast.stmt, q: Path) -> Any:
        """What a ``name += ...`` *s* makes the name hold where the walk cannot read it; None to leave it unknown."""
        return None

    def row_rule(self, q: Path, line: int) -> str:
        """Why *q* cannot pass the row at *line*, after *check_row*; '' where it can."""
        return ""

    # -- environment

    def env(self, p: Path, with_holes: bool = True) -> evaluate.Env:
        env = self.make_env()
        if with_holes:
            self.put_holes(env, p)
        for t, v in p.state.items():
            if v is evaluate.UNKNOWN or isinstance(v, Partial):
                env.stale[t] = BOUND_ON_PATH
            else:
                env.put(t, v, BOUND_ON_PATH)
        return env

    # -- walking

    def run(self, region: list[ast.stmt]) -> list[Path]:
        if self.require_all and self.order:
            # nothing after the last row can drop a path or change whether it passed the arm
            last = max(_place(stmt, region) for stmt, _ in self.order.values())
            region = region[: last + 1]
        self.region = region
        self.steps = self.stepped(region)
        self._after = {}
        start = self.path_type(state=entry_state(self.func, region[0], self.relevant))
        paths = self.block(region, [start])
        if not self.require_all:
            return paths
        out = []
        for p in paths:
            if set(self.order) - set(p.seen):
                self.dropped["E2 missed"] += 1
            else:
                out.append(p)
        return out

    def block(self, stmts: list[ast.stmt], paths: list[Path]) -> list[Path]:
        for s in stmts:
            nxt: list[Path] = []
            for p in paths:
                nxt.extend([p] if p.left else self.stmt(s, p))
            self.drop_unread(s, nxt)
            paths = merge(nxt)
            if len(paths) > WIDEN_AT:
                paths = widen(paths, self.target)
                self.widened += 1
            if len(paths) > self.most:
                self.most = len(paths)
            if len(paths) > MAX_PATHS:
                raise TooMany(len(paths))
        return paths

    def stepped(self, region: list[ast.stmt]) -> list[tuple[int, set[str], set[str]]]:
        """Each statement of *region* the walk steps into -- an ``if``'s body and ``else`` and a ``with``'s body
        are entered, the rest is one step -- by its line, with the texts its test looks facts up by
        (:func:`lookups`, an ``if``'s) and the names it hands a call (:meth:`roots`: of an ``if``'s test, of a
        ``with``'s items, of any other statement whole)."""
        out: list[tuple[int, set[str], set[str]]] = []

        def visit(stmts: list[ast.stmt]) -> None:
            for s in stmts:
                if isinstance(s, ast.If):
                    out.append((s.lineno, lookups(s.test), self.roots(s.test)))
                    visit(s.body)
                    visit(s.orelse)
                elif isinstance(s, (ast.With, ast.AsyncWith)):
                    handed = set().union(*(self.roots(i.context_expr) for i in s.items))
                    out.append((s.lineno, set(), handed))
                    visit(s.body)
                else:
                    out.append((s.lineno, set(), self.roots(s)))

        visit(region)
        return out

    def read_after(self, s: ast.stmt) -> tuple[set[str], set[str]]:
        """What the statements walked after *s* can read of a path's facts: the texts their tests look facts up
        by, and the names they hand a call.  After is by line: a loop, a ``try`` or a ``match`` is one step, so
        no way leads back, and another arm's statements count too, so it holds more than a path reads."""
        end = s.end_lineno or s.lineno
        held = self._after.get(end)
        if held is None:
            texts: set[str] = set()
            handed: set[str] = set()
            for line, read, names in self.steps:
                if line > end:
                    texts |= read
                    handed |= names
            held = self._after[end] = (texts, handed)
        return held

    def drop_unread(self, s: ast.stmt, paths: list[Path]) -> None:
        """Drop from *paths* the facts nothing walked after *s* can read, so that paths which differ only in them
        merge.  A fact is read by :meth:`truth`, by its text (:func:`lookups`), and by :meth:`immutable` as a
        witness that a name holds a value no call can change (:func:`witnessed`), which only :meth:`call` asks, of
        a name a statement hands a call.  So a fact stays while a later test looks it up, or, as a witness of a
        name, while a later statement hands the name to a call, the name roots the target, or the path holds a
        value at a chain under the name."""
        texts, handed = self.read_after(s)
        for q in paths:
            if any(k not in texts for k in q.facts):
                q.facts = {
                    k: v
                    for k, v in q.facts.items()
                    if k in texts or self.witness_asked(k, v, handed, q)
                }

    def witness_asked(self, atom: str, truth: bool, handed: set[str], q: Path) -> bool:
        """Whether *atom* holding *truth*, a witness of a name (:func:`witnessed`), can still be asked on *q*: a
        later statement hands the name to a call (*handed*), the name roots the target, or *q* holds a value at a
        chain under it."""
        return any(
            name in handed
            or name == self.target_root
            or any(t != name and root(t) == name for t in q.state)
            for name in witnessed(atom, truth)
        )

    def leave(self, p: Path, why: str) -> Path:
        q = p.copy()
        q.left = why
        return q

    def stmt(self, s: ast.stmt, p: Path) -> list[Path]:
        if isinstance(s, ast.If):
            return self.branch(s, p)
        if isinstance(s, (ast.With, ast.AsyncWith)):
            out = []
            if any(self.raises(i.context_expr) for i in s.items):
                out.append(self.leave(p, f"call at {s.lineno}"))
            q = p.copy()
            for i in s.items:
                self.calls(q, i.context_expr)
            self.bind(q, binds(s))
            return out + self.block(s.body, [q])
        if isinstance(s, JUMPS):
            return [self.leave(p, f"{type(s).__name__.lower()} at {s.lineno}")]
        if isinstance(s, OPAQUE):
            return self.opaque(s, p)
        return self.simple(s, p)

    def branch(self, s: ast.If, p: Path) -> list[Path]:
        out = []
        if self.raises(s.test):
            out.append(self.leave(p, f"call at {s.lineno}"))
        q0 = p.copy()
        self.calls(q0, s.test)
        for outcome in self.decide(self, s, q0):
            q = q0.copy()
            if self.track:
                q.tests += ((s.lineno, outcome),)
            if (
                self.target
                and self.target not in q.state
                and affects(self.target, chains(text_of(s.test)))
            ):
                q.prior += (
                    (
                        text_of(s.test),
                        outcome,
                        tuple(sorted(q.state.items(), key=lambda kv: kv[0])),
                    ),
                )
            self.settle(q, s.test, outcome)
            out.extend(self.block(s.body if outcome else s.orelse, [q]))
        return out

    def settle(self, q: Path, test: ast.expr, outcome: bool) -> None:
        """Keep what taking this branch says about the test's atoms the path did not know."""
        found = evaluate.atoms(test)
        if len(found) > 10:
            return
        try:
            solved = evaluate.solve(
                text_of(test), outcome, self.env(q, with_holes=False)
            )
        except Exception:
            return
        for atom, truth in solved.settled.items():
            if chains(atom) and all(
                c in self.relevant or any(affects(c, {r}) for r in self.relevant)
                for c in chains(atom)
            ):
                q.facts[atom] = truth

    def truth(self, test: ast.expr, p: Path, env: evaluate.Env) -> Any:
        """The test's value from the path's facts and *env*; UNKNOWN when they do not decide it."""
        try:
            return evaluate.truth(test, p.facts, env)
        except Exception:
            return evaluate.UNKNOWN

    def simple(self, s: ast.stmt, p: Path) -> list[Path]:
        out = []
        if self.raises(s):
            out.append(self.leave(p, f"call at {s.lineno}"))
        q = p.copy()
        if s is self.arm:
            q.arm = True
        appended = self.appended(s, q)
        self.calls(q, s)
        self.bind(q, own_binds(s))
        if appended is not None and s.target.id in q.state:  # type: ignore[attr-defined]
            q.state[s.target.id] = appended  # type: ignore[attr-defined]
        if s.lineno in self.order and self.order[s.lineno][0] is s:
            if not self.pass_row(q, s.lineno):
                return out
        out.append(q)
        return out

    def opaque(self, s: ast.stmt, p: Path) -> list[Path]:
        out = []
        if may_divert(s, self.raised):
            jumps = contains(s, (ast.Return, ast.Raise))
            out.append(
                self.leave(
                    p,
                    f"{'jump' if jumps else 'call'} in {type(s).__name__.lower()} at {s.lineno}",
                )
            )
        q = p.copy()
        self.calls(q, s)
        self.bind(q, [(t, None) for t in deep_binds(s)])
        inside = sorted(
            (
                line
                for line, (stmt, _) in self.order.items()
                if s in pathcond.ancestors(stmt)
            ),
            key=lambda line: self.order[line][1],
        )
        for line in inside:
            if not self.pass_row(q, line):
                return out
        out.append(q)
        return out

    def calls(self, q: Path, node: ast.AST) -> None:
        """What the calls in *node* -- a statement, a test, a context expression -- can change."""
        roots = self.self_calls(q, node, self.roots(node))
        self.call(q, roots, node)

    def call(self, q: Path, roots: set[str], node: ast.AST) -> None:
        """Drop what the calls in *node* may change of what the path holds of a chain rooted at one of *roots*: the
        values it bound there, save the target's (read through ``target_called``), and the facts reading them; and
        mark what they may change (:func:`change_mark`)."""
        roots = roots & self.rel_roots
        if not roots:
            return
        made = [
            n for n in own_nodes(node) if isinstance(n, ast.Call) and not is_log_call(n)
        ]
        marks = frozenset(
            change_mark(r, change)
            for r in roots
            for call in made
            for change in self.call_changes(call, r)
        )
        q.marks |= marks
        q.since = {line: m | marks for line, m in q.since.items()}
        facts, state = dict(q.facts), dict(q.state)

        def changed(read: Iterable[tuple[ast.expr, str]]) -> bool:
            return any(
                self.changed(made, roots, chain, how, facts, state)
                for chain, how in read
            )

        for t, v in state.items():
            if (
                t != self.target
                and v is not evaluate.UNKNOWN
                and root(t) in roots
                and changed(value_reading(t, v))
            ):
                q.state[t] = evaluate.UNKNOWN
        for atom in facts:
            if any(root(c) in roots for c in chains(atom)) and changed(readings(atom)):
                del q.facts[atom]
        if (
            self.target_attr
            and self.target_root in roots
            and changed(
                value_reading(self.target, state.get(self.target, evaluate.UNKNOWN))
            )
        ):
            q.target_called = True
            self.handed_target(q, node)

    def changed(
        self,
        made: list[ast.Call],
        roots: set[str],
        chain: ast.expr,
        how: str,
        facts: dict[str, bool],
        state: dict[str, Any],
    ) -> bool:
        """Whether one of the calls *made* may change what a path holding *facts* and *state* reads of *chain*,
        read as *how* says (:data:`IDENTITY` or :data:`CONTENT`): the call may change the object the chain's name
        holds at the chain's first step, or at any key for a bare name read by its content -- its identity no call
        changes.  Where the chain is one step and read by its identity, a change inside the value it holds does not
        count.  An object no change in place can touch changes nothing (:meth:`immutable`)."""
        steps = changes.chain_steps(chain)
        if steps is None:
            return False
        name, keys = steps
        if name not in roots or (not keys and how == IDENTITY):
            return False
        first = keys[0] if keys else changes.EVERY
        reached = any(
            first == changes.EVERY or change.key in (first, changes.EVERY)
            for call in made
            for change in self.call_changes(call, name)
            if not (how == IDENTITY and len(keys) == 1 and change.kind == CHANGE_INNER)
        )
        return reached and not self.immutable(name, facts, state)

    def immutable(
        self, name: str, facts: dict[str, bool], state: dict[str, Any]
    ) -> bool:
        """Whether *name* holds a value no change in place can touch: a value the path knows, else a fact it holds
        or what the code states (:func:`stated_immutable`)."""
        value = state.get(name, evaluate.UNKNOWN)
        if value is not evaluate.UNKNOWN and not isinstance(value, Partial):
            return _immutable_value(value)
        if _fact_immutable(name, facts):
            return True
        asked = _asked(self.corpus).stated
        key = (id(self.func), name, self._bound_key)
        if key not in asked:
            asked[key] = (
                self.func,
                stated_immutable(
                    self.corpus, self.file, self.func, name, self.bound_classes
                ),
            )
        return asked[key][1]

    def call_changes(self, call: ast.Call, name: str) -> list[changes.ObjectChange]:
        """What *call* may change in place of the object *name* holds (:func:`changes.call_changes`); a change of
        any key where the reading fails."""
        asked = _asked(self.corpus).changes
        key = (id(call), name, self._bound_key)
        if key not in asked:
            try:
                found = changes.call_changes(
                    self.corpus, self.file, self.func, call, name, self.bound_classes
                )
            except Exception:  # noqa: BLE001 -- a reading that fails may be of anything
                found = [
                    changes.ObjectChange(
                        self.file, call.lineno, CHANGE_UNREAD, changes.EVERY, "not read"
                    )
                ]
            asked[key] = (call, found)
        return asked[key][1]

    def forget(self, q: Path, changed: Callable[[str, Any], bool]) -> None:
        """Drop what the path knew of what a call changed: the values it bound there (the target's stays, read
        through ``target_called``) and the facts reading them."""
        for t, v in list(q.state.items()):
            if t != self.target and v is not evaluate.UNKNOWN and changed(t, v):
                q.state[t] = evaluate.UNKNOWN
        for atom in list(q.facts):
            if any(changed(c, None) for c in chains(atom)):
                del q.facts[atom]

    def bind(self, q: Path, pairs: list[tuple[str, Optional[ast.expr]]]) -> None:
        for t, vnode in pairs:
            hit = [c for c in self.relevant if affects(t, {c})]
            if not hit:
                continue
            value = self.value_of(q, t, vnode)
            for c in hit:
                if c != t:
                    q.state[c] = evaluate.UNKNOWN
            q.state[t] = value
            for atom in list(q.facts):
                if affects(t, chains(atom)):
                    del q.facts[atom]
            q.marks |= {t}
            q.since = {line: m | {t} for line, m in q.since.items()}
            self.bound(q, t, value)
            if t == self.target:
                q.target_called = False

    def pass_row(self, q: Path, line: int) -> bool:
        rank = self.order[line][1]
        if q.seen and self.order[q.seen[-1]][1] > rank:
            self.dropped["E2 order"] += 1
            return False
        q.seen += (line,)
        q.since = {**q.since, line: frozenset()}
        if self.track and self.target:
            value = q.state.get(self.target, NOT_BOUND)
            if self.target_attr and q.target_called:
                value = evaluate.UNKNOWN
            q.at_row += ((line, value),)
        why = self.check_row(self, line, q)
        if not why:
            why = self.row_rule(q, line)
        if why:
            self.dropped[why] += 1
            return False
        return True
