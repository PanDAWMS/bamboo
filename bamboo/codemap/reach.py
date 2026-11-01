"""Whether a function reaches, through calls and hand-offs, a write: recorded once per build, answered from the record.

bamboo-8f's question (10-14), with bamboo-c9's ends: from a function that
moves a subject off a value -- a *mover* -- does any function the mover can
go on into hold an arm that moves the subject off it again?  A mover that
cannot is one a reader may drop, and only where the answer says so for
certain.  So every answer keeps three things apart: the writes reached; no
write reached and nothing left open (``[]``); and no write reached with a
call somewhere below whose way on is not known (``None``).

:func:`graph` records, per function, where each call it makes goes
(:class:`ReachNode`); :class:`ReachGraph` answers from those rows alone.
Nothing in a row depends on the subject or the value: those decide only
where a closure ends, and the junctions say that (:func:`moving_off`).

A call goes on into the definitions :meth:`calls.Calls.callees` places it
at.  A method found by inheritance runs as the object's class
(:attr:`calls.Target.bound`): a call that reads ``self`` is placed once for
each class ``self`` can be, and a closure is taken over ``(function,
class)``, so ``WorkerThread.run`` entered as ``JobGeneratorThread``'s goes on
into ``JobGeneratorThread.runImpl`` and into no other worker's.  A function
handed as an argument is taken as called (:meth:`calls.Calls.named_callables`), and
so is a call in a lambda's body.

A call placed nowhere is closed where a rule says it cannot go on into the
corpus, and is a hole where none does.  Some rules are facts under the
premises every answer shares (:data:`PREMISES`); the rest are assumptions,
counted per function (:data:`ASSUMPTIONS`), and every answer lists those its
closure relied on.

Functions only: a module's own body runs at import, and no mover is one.
"""

from __future__ import annotations

import ast
import builtins
import re
from collections import Counter, deque
from typing import Callable, Iterable, Iterator, Optional

from bamboo.codemap import callfacts
from bamboo.codemap import calls as C
from bamboo.codemap.astwalk import walk
from bamboo.codemap.models import (
    Anchor,
    JunctionNode,
    Reach,
    ReachCall,
    ReachHole,
    ReachNode,
    ReachWrite,
    required_values,
)
from bamboo.codemap.panda import pathcond
from bamboo.codemap.panda.reaching import reaching

#: What every answer rests on.  Said once, to every reader, rather than
#: counted per answer: no answer could be given without them.
PREMISES: dict[str, str] = {
    "declared_types_hold": (
        "A value declared as a class is an instance of it or of one of its "
        "subclasses in the corpus."
    ),
    "outside_calls_back_only_handed": (
        "Code outside the corpus calls code of the corpus only through a "
        "callable it was handed, ``start()`` running ``run``, or an implicit "
        "dunder (``with``, ``for``, an operator; an attribute store's "
        "``__setattr__`` is an edge) -- so a call on a builtin or outside "
        "object goes on into nothing of the corpus."
    ),
    "cross_run_not_reach": (
        "What a run hands on through the database, a command or a message is "
        "another run's input, not this run's reach."
    ),
    "path_insensitive": "A closure is every function that can be entered, by any path: reaches means can reach.",
}

#: The assumptions a closure can rely on, each one closing calls a premise
#: does not: a reader dropping a mover names the ones its answer lists.
ASSUMPTIONS: dict[str, str] = {
    "outside_returns_outside": (
        "Code outside the corpus returns no object of the corpus it was not "
        "handed: a call on what it returns goes on into nothing of the corpus."
    ),
    "no_runtime_methods": (
        "No method is added to a class at run time: a call of a name no class "
        "of the corpus defines lands in no corpus code."
    ),
    "runner_runs_only_sql": "A call the map reads a SQL statement at runs the statement and nothing of the corpus.",
    "runner_runs_callers_sql": (
        "A call running the statement its function was handed runs its callers' statement, read where they "
        "run it, and nothing of the corpus."
    ),
    "driver_talks_only_to_db": "A call on the database driver -- a cursor, a connection -- goes only to the database.",
    "closed_by_name": (
        "A call left open lands, if anywhere in the corpus, at a function of the corpus of its name: where none of "
        "those, nor anything they reach, writes the destination, the call writes none of it (MV 4, O1)."
    ),
    "queue_given_back": (
        "A value put back into the queue it was taken from hands no work on: whoever takes it next runs its own "
        "(DBProxyPool's proxies, a child process kept in a queue, a lock's token)."
    ),
}

#: What :meth:`ReachGraph.closure` reached: ``(owner, class)`` -> the node and the call it was reached from, ``None``
#: for a start.
Came = dict[tuple[str, str], Optional[tuple[tuple[str, str], ReachCall]]]

#: A database leaf or a hole for a call, as a build's plugin reads it:
#: ``("database", assumption)`` or ``("hole", why)``.  ``None`` where the call
#: is not the database's.  See :func:`graph`.
Leaves = Callable[[str, ast.AST, ast.Call, C.Placed], Optional[tuple[str, str]]]

HOLE_LOCAL_CALLED = "a local or a parameter is called"
HOLE_QUEUE = "hands a value to another thread or process through a queue"
HOLE_FAILED = "placing it failed"
HOLE_OUTSIDE_START = "start() of a class built outside the corpus may run its run"
#: How a plugin's leaves say a database call whose statement the build could not read: it may be a write.
HOLE_STATEMENT_NOT_READ = "the statement is not read"
#: Holes no name closes (:class:`_ByName`): whoever takes a queue's value, a ``run`` a ``start()`` of a class built
#: outside the corpus may run, a statement not read.
_NOT_BY_NAME = (HOLE_QUEUE, HOLE_OUTSIDE_START, HOLE_STATEMENT_NOT_READ)
_HOLE_TEXT = re.compile(r"^(?P<owner>.*?):(?P<line>\d+) (?P<callee>.*?) -- (?P<why>.*)$")

#: The library classes whose ``put()`` hands a value to whoever ``get()``s it (:data:`callfacts.QUEUES`).
_QUEUES, _PUTS, _GETS = callfacts.QUEUES, callfacts.QUEUE_PUTS, callfacts.QUEUE_GETS
_BUILTINS = set(dir(builtins))
#: The builtin types and the typing and collections names a builtin value is annotated with: the scalars, and the
#: containers holding keys, elements or positions (:data:`callfacts.CONTAINERS`).
_BUILTIN_ANNOTATIONS = {
    "str", "int", "float", "bool", "bytes", "bytearray", "complex", "range", "None", "NoneType",
} | callfacts.containers(callfacts.MAPPING, callfacts.COLLECTION, callfacts.RECORD)  # fmt: skip
_SCALARS = {"str", "int", "float", "bool", "bytes", "complex", "NoneType"}
#: The dunders each operator calls (:data:`callfacts.OPERATOR_DUNDERS`).
_OPERATORS = callfacts.OPERATOR_DUNDERS

#: The dunders Python runs with no call naming them (:data:`callfacts.IMPLICIT_DUNDERS`) -- ``with``, ``for``,
#: ``in``, ``x[k]``, ``str(x)``, an operator, comparing, hashing, pickling
#: and copying.  No edge says when they run, so the premise
#: ``outside_calls_back_only_handed`` takes them to reach no write, and the
#: build checks it (``implicit-dunders-reach-no-write``) -- but for
#: ``__setattr__``, whose edges are recorded (:data:`HOW_SETATTR`):
#: ``JobSpec.__setattr__`` sets ``stateChangeTime`` when ``jobStatus``
#: changes.  Those edges are the reach record's alone -- :meth:`calls.Calls.callers`,
#: and the walk and the trace with it, do not see the call, so a junction in a
#: ``__setattr__`` has no ``called_from`` -- and a store on an object no
#: reading says anything of has none (167 stores on pin k, 9 of them of
#: ``jobStatus``).
IMPLICIT_DUNDERS = callfacts.IMPLICIT_DUNDERS

#: How a reach edge says an attribute store runs its object's ``__setattr__`` (:meth:`calls.Calls.stored`).
HOW_SETATTR = "setattr"

_BUILTIN, _OUTSIDE, _QUEUE, _UNKNOWN = "builtin", "outside", "queue", "unknown"
#: A builtin's detail for a value ``json.loads`` makes (:data:`callfacts.JSON_MAKERS`): a builtin, and every value
#: inside it, and every method's result, a builtin too.
_JSON = "a value json makes"
#: What a binding read inside its own reading adds: nothing.  ``diag =
#: re.sub(..., diag)`` beside ``diag: str`` is a str; the second binding
#: says no more than the first.  Never an answer: a reading that finds
#: nothing else is ``unknown``.
_NOTHING = ("nothing", "")
_IN_PROGRESS = ("", "")


def _cut(node: ast.AST) -> str:
    text = ast.unparse(node)
    return text if len(text) <= 80 else text[:79] + "…"


class _Names:
    """The id of every function and class of the corpus: ``file::Outer.inner``, ``#n`` after a repeated one.

    Of the definitions one scope repeats a name with, the one the name is bound to -- the one the calls resolve it to
    (:func:`calls.binding`) -- is spelled without ``#n``, as every reader spelling an owner by its qualified name
    means it, and the others ``#2``, ``#3`` in source order: ``CoreUtils.clean_user_id``'s ``@overload`` stubs come
    before the body that runs, and a property's setter after its getter.
    """

    def __init__(self, types: C.Types):
        self.functions: list[tuple[str, ast.AST, str]] = []
        self._of: dict[int, str] = {}
        self._keep: list[ast.AST] = []
        for file in types.corpus_files():
            tree = types.tree(file)
            if tree is None:
                continue
            seen: Counter = Counter()
            self._walk(file, tree, [], seen)

    def _walk(self, file: str, scope: ast.AST, names: list[str], seen: Counter) -> None:
        defs = list(_scope_definitions(scope))
        bound = {name: C.binding(defs, name) for name in {node.name for node in defs}}
        spelled_as: dict[int, str] = {}
        for node in sorted(defs, key=lambda one: one is not bound[one.name]):
            spelled = f"{file}::{'.'.join([*names, node.name])}"
            seen[spelled] += 1
            if seen[spelled] > 1:
                spelled = f"{spelled}#{seen[spelled]}"
            spelled_as[id(node)] = spelled
        for node in defs:
            self._of[id(node)] = spelled_as[id(node)]
            self._keep.append(node)
            if not isinstance(node, ast.ClassDef):
                self.functions.append((file, node, spelled_as[id(node)]))
            self._walk(file, node, [*names, node.name], seen)

    def of(self, node: ast.AST) -> Optional[str]:
        return self._of.get(id(node))


def _scope_definitions(scope: ast.AST) -> Iterator[ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef]:
    """The ``def`` and ``class`` statements of *scope* itself, in source order: inside its ``if`` and ``try`` blocks
    too, not inside another definition."""
    for node in ast.iter_child_nodes(scope):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield node
        else:
            yield from _scope_definitions(node)


def _annotation_parts(annotation: ast.expr) -> Iterator[ast.expr]:
    """The members of a union annotation, ``None`` left out: ``Optional[X]``, ``Union[X, Y]``, ``X | Y``, a string."""
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        try:
            annotation = ast.parse(annotation.value, mode="eval").body
        except SyntaxError:
            yield annotation
            return
    if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
        yield from _annotation_parts(annotation.left)
        yield from _annotation_parts(annotation.right)
        return
    if isinstance(annotation, ast.Subscript):
        head = annotation.value
        named = (
            head.attr if isinstance(head, ast.Attribute) else getattr(head, "id", "")
        )
        if named in ("Optional", "Union"):
            inner = annotation.slice
            for one in inner.elts if isinstance(inner, ast.Tuple) else [inner]:
                yield from _annotation_parts(one)
            return
    if isinstance(annotation, ast.Constant) and annotation.value is None:
        return
    yield annotation


class _Receivers:
    """Why the object a call is made on is no object of the corpus, where it can be said: ``(kind, detail)``.

    Asked only of a receiver :meth:`calls.Calls.callees` could not name.
    *kind* is ``builtin`` -- a builtin type's value, whose methods are the
    builtin's (a premise) --, ``outside`` -- made by code outside the corpus,
    which returns no corpus object it was not handed (an assumption) --,
    ``queue``, or ``unknown``, *detail* saying why.  Read from the root of
    the receiver: a literal, a parameter's annotation, what a corpus call
    is declared to return, a field's declaration where it names a class
    outside the corpus, what a name or a field is bound to.

    Once per expression: a name bound several ways is read once for each
    binding, and without the memo a function rebinding its locals from one
    another costs the product of their bindings.  A binding read again
    while it is being read adds nothing (:data:`_NOTHING`), and what is read
    under it is not kept until the binding it waited on is done -- read
    alone it would miss what that binding adds -- as Tarjan's lowlink keeps
    a component open until its root closes.
    """

    DEPTH = 12

    def __init__(self, calls: C.Calls, leaves: Optional[Leaves]):
        self.calls = calls
        self.types = calls.types
        self.leaves = leaves
        self._memo: dict[tuple[str, int, int], tuple[str, str]] = {}
        self._open: dict[tuple[str, int, int], int] = {}
        self._low: list[int] = []
        self._operators: dict[type, Optional[tuple[str, str]]] = {}
        self._held: dict[
            tuple[int, str, tuple[Optional[str], ...]],
            Optional[list[tuple[str, ast.AST, ast.expr, int]]],
        ] = {}
        self._scopes: dict[int, _Scope] = {}
        self._as_values: Optional[set[str]] = None
        self._keys: dict[tuple[int, str], Optional[frozenset[str]]] = {}

    def of(
        self, file: str, func: ast.AST, expression: ast.expr, depth: int = 0
    ) -> tuple[str, str]:
        key = (file, id(func), id(expression))
        held = self._memo.get(key)
        if held is _IN_PROGRESS:
            self._low[-1] = min(self._low[-1], self._open[key])
            return _NOTHING
        if held is not None:
            return held
        index = len(self._low)
        self._memo[key] = _IN_PROGRESS
        self._open[key] = index
        self._low.append(index)
        try:
            found = self._of(file, func, expression, depth)
        finally:
            low = self._low.pop()
            del self._open[key]
            del self._memo[key]
        if low < index:
            # Read under a binding still open: kept by none, told to it.
            self._low[-1] = min(self._low[-1], low)
            return found
        if found == _NOTHING:
            found = _UNKNOWN, "a binding that reads only itself"
        self._memo[key] = found
        return found

    def annotation(self, file: str, annotation: Optional[ast.expr]) -> tuple[str, str]:
        """What a value annotated *annotation* in *file* is."""
        if annotation is None:
            return _UNKNOWN, "nothing declares it"
        kinds = [self._annotated(file, part) for part in _annotation_parts(annotation)]
        return _joined(kinds) if kinds else (_BUILTIN, "NoneType")

    def _annotated(self, file: str, part: ast.expr) -> tuple[str, str]:
        head = part.value if isinstance(part, ast.Subscript) else part
        named = (
            head.attr if isinstance(head, ast.Attribute) else getattr(head, "id", "")
        )
        if named in ("Any", "object"):
            return _UNKNOWN, f"declared {named}"
        library = (
            self.types.library_class(file, head)
            if isinstance(head, (ast.Name, ast.Attribute))
            else None
        )
        if library in _QUEUES:
            return _QUEUE, ".".join(library)
        if named in _BUILTIN_ANNOTATIONS and (
            library is None or library[0].split(".")[0] in ("typing", "collections")
        ):
            held = callfacts.CONTAINERS.get(named)
            return _BUILTIN, (held.builtin if held is not None and held.builtin else named)
        if self.types.annotated_all(file, head):
            return _UNKNOWN, f"declared {ast.unparse(head)}, a class of the corpus"
        if library is not None:
            return _OUTSIDE, f"declared {'.'.join(library)}"
        root = head
        while isinstance(root, ast.Attribute):
            root = root.value
        if isinstance(root, ast.Name) and self.types.imported_from_outside(
            file, root.id
        ):
            return _OUTSIDE, f"declared {ast.unparse(head)}"
        return _UNKNOWN, f"declared {ast.unparse(head)}, not read"

    def _of(
        self, file: str, func: ast.AST, expression: ast.expr, depth: int
    ) -> tuple[str, str]:
        if depth > self.DEPTH:
            return _UNKNOWN, "read too deep"
        literal = self.types._builtin_instance(file, func, expression)
        if literal:
            return _BUILTIN, literal
        if isinstance(expression, (ast.Compare, ast.GeneratorExp)):
            return _BUILTIN, "bool" if isinstance(
                expression, ast.Compare
            ) else "iterator"
        if isinstance(expression, ast.UnaryOp) and isinstance(expression.op, ast.Not):
            return _BUILTIN, "bool"
        if isinstance(expression, (ast.BinOp, ast.UnaryOp)):
            return self._operated(file, func, expression, depth)
        if isinstance(expression, ast.BoolOp):
            return _joined(
                [self.of(file, func, one, depth + 1) for one in expression.values]
            )
        if isinstance(expression, ast.IfExp):
            return _joined(
                [
                    self.of(file, func, one, depth + 1)
                    for one in (expression.body, expression.orelse)
                ]
            )
        if isinstance(expression, (ast.NamedExpr, ast.Await)):
            return self.of(file, func, expression.value, depth + 1)
        if isinstance(expression, ast.Subscript):
            filled = self._filled(file, func, expression, depth)
            if filled is not None:
                return filled
            return _element(self.of(file, func, expression.value, depth + 1))
        if isinstance(expression, ast.Call):
            return self._called(file, func, expression, depth)
        if isinstance(expression, ast.Name):
            return self._named(file, func, expression.id, depth)
        if isinstance(expression, ast.Attribute):
            return self._attribute(file, func, expression, depth)
        return _UNKNOWN, _a(type(expression).__name__)

    def _operated(
        self, file: str, func: ast.AST, expression: ast.BinOp | ast.UnaryOp, depth: int
    ) -> tuple[str, str]:
        """What an operation is: a builtin's on builtins; else, where an operand may be a corpus object, what its dunder returns.

        ``a * b`` on an object of the corpus runs that class's ``__mul__``,
        ``__rmul__`` or ``__imul__`` -- ``GlobalShares``' return a float, a
        float and a ``Share`` -- so it is a builtin only where every
        definition of them is declared to return one.  Where none is
        defined, no corpus class takes part.
        """
        operands = (
            [expression.left, expression.right]
            if isinstance(expression, ast.BinOp)
            else [expression.operand]
        )
        kinds = _joined([self.of(file, func, one, depth + 1) for one in operands])
        if kinds == _NOTHING or kinds[0] in (_BUILTIN, _OUTSIDE):
            return kinds if kinds != _NOTHING else _NOTHING
        returned = self._operator_returns(type(expression.op))
        if returned is None:
            return _OUTSIDE, "an operation no class of the corpus defines"
        if returned[0] == _BUILTIN:
            return returned
        return _UNKNOWN, f"an operation a class of the corpus defines ({kinds[1]})"

    def _operator_returns(self, op: type) -> Optional[tuple[str, str]]:
        """What the corpus's definitions of *op*'s dunders are declared to return, joined; ``None`` where there is none."""
        if op not in self._operators:
            names = set(_OPERATORS.get(op, ()))
            found = [
                self.annotation(file, node.returns)
                for file in self.types.corpus_files()
                for node in walk(
                    self.types.tree(file) or ast.Module(body=[], type_ignores=[])
                )
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name in names
            ]
            self._operators[op] = _joined(found) if found else None
        return self._operators[op]

    def _slot(
        self,
        file: str,
        func: ast.AST,
        value: Optional[ast.expr],
        slot: Optional[int],
        depth: int,
    ) -> tuple[str, str]:
        """What a name unpacked from *value* at *slot* is: the literal's element there, the slot of the ``tuple[...]``
        the call's targets are declared to return, else one of what *value* holds -- ``siteid, queue_data_json, ... =
        resTmp``, a row the database returned."""
        if value is None:
            return _UNKNOWN, "an unpacking of nothing read"
        if (
            isinstance(value, (ast.Tuple, ast.List))
            and slot is not None
            and slot < len(value.elts)
            and not any(isinstance(one, ast.Starred) for one in value.elts)
        ):
            return self.of(file, func, value.elts[slot], depth + 1)
        if isinstance(value, ast.Call) and slot is not None:
            try:
                placed = self.calls.callees(file, func, value)
            except Exception:  # noqa: BLE001 -- a reading that fails says nothing
                placed = C.Placed()
            if placed.targets and not placed.unplaced and not placed.returns_none:
                kinds = []
                for target in placed.targets:
                    returns = getattr(target.func, "returns", None)
                    held = C._slot_of(returns, slot) if returns is not None else None
                    kinds.append(
                        self.annotation(target.file, held)
                        if held is not None
                        else _element(self.annotation(target.file, returns))
                    )
                return _joined(kinds)
        return _element(self.of(file, func, value, depth + 1))

    def _handed_only_outside(
        self, file: str, func: ast.AST, call: ast.Call, depth: int
    ) -> Optional[str]:
        """Why *call*'s arguments may be objects of the corpus, or ``None`` where every one is builtin or outside."""
        for argument in [*call.args, *(keyword.value for keyword in call.keywords)]:
            value = argument.value if isinstance(argument, ast.Starred) else argument
            kind, detail = self.of(file, func, value, depth + 1)
            if kind not in (_BUILTIN, _OUTSIDE, _NOTHING[0]):
                return f"handed {_cut(value)}, which may be an object of the corpus ({detail})"
        return None

    def _called(
        self, file: str, func: ast.AST, call: ast.Call, depth: int
    ) -> tuple[str, str]:
        if self.leaves is not None:
            leaf = self.leaves(file, func, call, C.Placed())
            if leaf is not None and leaf[0] == "database":
                return _OUTSIDE, "what the database returns"
        try:
            placed = self.calls.callees(file, func, call)
        except Exception as error:  # noqa: BLE001 -- a reading that fails says nothing
            return _UNKNOWN, f"placing it failed: {type(error).__name__}"
        if placed.returns_none:
            return _BUILTIN, "NoneType"
        if placed.targets:
            if placed.unplaced:
                return _UNKNOWN, placed.unplaced
            return _joined(
                [
                    self.annotation(target.file, getattr(target.func, "returns", None))
                    for target in placed.targets
                ]
            )
        called = call.func
        if placed.unplaced == C.UNPLACED_OUTSIDE:
            library = (
                self.types.library_class(file, called)
                if isinstance(called, (ast.Name, ast.Attribute))
                else None
            )
            if library in _QUEUES:
                return _QUEUE, ".".join(library)
            if library in callfacts.JSON_MAKERS and not any(one.arg in callfacts.JSON_HOOKS for one in call.keywords):
                return _BUILTIN, _JSON
            if isinstance(called, ast.Attribute):
                receiver = self.of(file, func, called.value, depth + 1)
                if receiver[0] in (_BUILTIN, _QUEUE):
                    return self._method_result(receiver, called.attr)
            handed = self._handed_only_outside(file, func, call, depth)
            return (
                (_UNKNOWN, handed) if handed else (_OUTSIDE, "built outside the corpus")
            )
        if (
            placed.unplaced == C.UNPLACED_NAME
            and isinstance(called, ast.Name)
            and self._is_builtin(file, func, called.id)
        ):
            function = callfacts.BUILTIN_FUNCTIONS.get(called.id)
            result = function.result if function is not None else None
            if result is None:
                return _UNKNOWN, f"what {called.id}() returns"
            return _BUILTIN, result
        if placed.unplaced == C.UNPLACED_RECEIVER and isinstance(called, ast.Attribute):
            receiver = self.of(file, func, called.value, depth + 1)
            if receiver == _NOTHING:
                return _NOTHING
            if receiver[0] == _OUTSIDE:
                handed = self._handed_only_outside(file, func, call, depth)
                return (_UNKNOWN, handed) if handed else (_OUTSIDE, receiver[1])
            if receiver[0] in (_BUILTIN, _QUEUE):
                return self._method_result(receiver, called.attr)
            return receiver
        return _UNKNOWN, placed.unplaced or "placed nowhere"

    @staticmethod
    def _method_result(receiver: tuple[str, str], method: str) -> tuple[str, str]:
        """What a builtin's or a queue's method returns: one of what it holds, another of its type (a copy), or a builtin."""
        kind, held = receiver
        if held == _JSON:
            return receiver
        if kind == _QUEUE:
            return (
                (_UNKNOWN, f"what {_a(held)} holds")
                if method in _GETS
                else (_BUILTIN, "")
            )
        if held in _SCALARS:
            return _BUILTIN, ""
        gives = callfacts.taken(method, held)
        if gives is not None and gives.part == callfacts.SELF:
            return _BUILTIN, held
        if gives is not None and gives.part != callfacts.KEYS:
            return _UNKNOWN, f"what {_a(held or 'builtin container')} holds"
        return _BUILTIN, ""

    def _filled(
        self, file: str, func: ast.AST, expression: ast.Subscript, depth: int
    ) -> Optional[tuple[str, str]]:
        """What ``d[k]...`` is, read from what is put at that place of the container *d* holds: the join of the values
        every holder of it puts there (:class:`_Holders`), read down the literals put (``file_list[block] = {"lfns":
        [], ...}`` then ``file_list[block]["lfns"].append(lfn)``: a list) and, for a value held at a slot, the slot
        (:meth:`_slot`: ``(fetched[site], ...) = self.getOutputFiles_JEDI(...)``, the first of the tuple it is declared
        to return).

        ``None`` -- the element of what *d* is, as before -- where *d* is read at a module's scope, is a scalar, an
        outside object or a queue, where a move of the walk is not read, or where nothing is put at that place.
        """
        root, keys = _subscripted(expression)
        if root is None or isinstance(func, ast.Module):
            return None
        container = self.of(file, func, root, depth + 1)
        if container[0] in (_OUTSIDE, _QUEUE) or (
            container[0] == _BUILTIN and container[1] in _SCALARS
        ):
            return None
        asked = (id(func), root.id, _spelled(tuple(keys)))
        if asked not in self._held:
            self._held[asked] = _Holders(self, tuple(keys)).values(file, func, root.id)
        values = self._held[asked]
        if not values:
            return None
        kinds = []
        for where, scope, value, path in values:
            if path:
                kind = self._slot(where, scope, value, _position(path[0]), depth)
            else:
                kind = self.of(where, scope, value, depth + 1)
            for _key in path[1:]:
                kind = _element(kind)
            kinds.append(kind)
        return _joined(kinds)

    def _scope(self, func: ast.AST) -> "_Scope":
        held = self._scopes.get(id(func))
        if held is None:
            held = self._scopes[id(func)] = _Scope(func)
        return held

    def _target(self, file: str, func: ast.AST) -> C.Target:
        """*func* as a call lands at it: a method with the class defining it, or a function."""
        owned = self.types.method_class(file, func)
        cls = owned[1] if owned is not None and func in owned[1].body else None
        return C.Target(file, func, cls, ())

    def keys(self, file: str, func: ast.AST, name: str) -> Optional[frozenset[str]]:
        """The keys the mapping *name* holds in *func* can be at: the texts every put into it puts it at, read from
        every name holding it (:class:`_Keys`) -- ``param`` in ``updateJobStatus``, filled by ``pilot_api.update_job``;
        ``None`` where one is not read."""
        asked = (id(func), name)
        if asked not in self._keys:
            self._keys[asked] = _Keys(self).read_keys(file, func, name)
        return self._keys[asked]

    def _read_as_value(self, func: ast.AST) -> bool:
        """Whether a name spelt as *func*'s is read anywhere in the corpus other than as what a call calls: handed as
        a callback, a thread's target, held -- called then with what no call here says.  Not a read to build an
        object whose every call hands the function its arguments (:meth:`Calls.forwarded_reads`): those calls are
        placed at it."""
        if self._as_values is None:
            found: set[str] = set()
            forwarded = self.calls.forwarded_reads()
            for file in self.types.corpus_files():
                for node in walk(
                    self.types.tree(file) or ast.Module(body=[], type_ignores=[])
                ):
                    if id(node) in forwarded:
                        continue
                    if isinstance(node, ast.Attribute) and isinstance(
                        node.ctx, ast.Load
                    ):
                        spelled = node.attr
                    elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                        spelled = node.id
                    else:
                        continue
                    parent = getattr(node, "parent", None)
                    if not (isinstance(parent, ast.Call) and parent.func is node):
                        found.add(spelled)
            self._as_values = found
        return func.name in self._as_values  # type: ignore[attr-defined]

    def _is_builtin(self, file: str, func: ast.AST, name: str) -> bool:
        return self.types.is_builtin(file, func, name)

    def _named(
        self, file: str, func: ast.AST, name: str, depth: int
    ) -> tuple[str, str]:
        if isinstance(func, ast.Module):
            return self._module_named(file, func, name, depth)
        kinds: list[tuple[str, str]] = []
        parameters = _parameters(func)
        if name in parameters:
            kinds.append(self.annotation(file, parameters[name]))
        for source, entered in _loop_sources(func, name):
            held = self.of(file, func, source, depth + 1)
            kinds.append(_entered(held) if entered else _element(held))
        for value in self.types.bound(func).get(name, ()):
            kinds.append(self.of(file, func, value, depth + 1))
        for one in pathcond.bindings(func, name):
            if (
                not one.nested
                and one.unpacked
                and one.kind in (pathcond.BIND_ASSIGN, pathcond.BIND_ANNOTATED)
            ):
                kinds.append(self._slot(file, func, one.value, one.slot, depth))
        if kinds:
            return _joined(kinds)
        enclosing = self.types.enclosing_function(file, func)
        if enclosing is not None:
            return self._named(file, enclosing, name, depth + 1)
        tree = self.types.tree(file)
        return (
            self._module_named(file, tree, name, depth)
            if tree is not None
            else (_UNKNOWN, "no binding found")
        )

    def _module_named(
        self, file: str, tree: ast.Module, name: str, depth: int
    ) -> tuple[str, str]:
        values = _module_values(tree, name)
        if values:
            return _joined(
                [
                    self.annotation(file, value[1])
                    if isinstance(value, tuple)
                    else self.of(file, tree, value, depth + 1)
                    for value in values
                ]
            )
        if name in self.types.imports(file):
            if self.types.imported_from_outside(file, name):
                return _OUTSIDE, f"{name}, imported from outside the corpus"
            return _UNKNOWN, f"{name}, imported from the corpus"
        if name in _BUILTINS:
            return _BUILTIN, ""
        return _UNKNOWN, "no binding found"

    def _attribute(
        self, file: str, func: ast.AST, expression: ast.Attribute, depth: int
    ) -> tuple[str, str]:
        root = expression.value
        if (
            isinstance(root, ast.Name)
            and root.id in ("self", "cls")
            and not isinstance(func, ast.Module)
        ):
            owned = self.types.method_class(file, func)
            if owned is None:
                return _UNKNOWN, "self outside a method"
            # A field declared as a class outside the corpus -- ``jedi_mb_proxy_dict: "PassiveProxies | None"``,
            # ``logger: "LogWrapper"`` -- holds that class's object, whatever its assignments read as: a callable
            # field's call, a value a loop installs.  The declaration is the boundary.  Any other declaration (``Any``,
            # a class of the corpus) says less than the assignments, which are read as before.
            declared = self.types._field_annotation(owned, expression.attr)
            if declared is not None:
                said = self.annotation(*declared)
                if said[0] in (_OUTSIDE, _QUEUE):
                    return said
            kinds = []
            seen: set[int] = set()
            for one in self.calls.objects_of(owned):
                for where, holder in self.types.lineage(*one):
                    if id(holder) in seen:
                        continue
                    seen.add(id(holder))
                    for method in pathcond.methods_of(holder):
                        for value in pathcond.attribute_expressions(
                            method, expression.attr
                        ):
                            if isinstance(value, ast.Constant) and value.value is None:
                                continue
                            kinds.append(self.of(where, method, value, depth + 1))
            return (
                _joined(kinds)
                if kinds
                else (_UNKNOWN, f"a field nothing assigns: {expression.attr}")
            )
        module = self.types.module_named(file, root)
        if module is not None:
            tree = self.types.tree(module)
            return (
                self._module_named(module, tree, expression.attr, depth + 1)
                if tree is not None
                else (_UNKNOWN, "")
            )
        owner = self.of(file, func, root, depth + 1)
        if owner == _NOTHING:
            return _NOTHING
        if owner[0] == _OUTSIDE or (owner[0] == _BUILTIN and owner[1] in _SCALARS):
            return owner[0], f"an attribute of {owner[1]}"
        return _UNKNOWN, f"an attribute of another object ({owner[1]})"


def _joined(kinds: list[tuple[str, str]]) -> tuple[str, str]:
    """What a value bound several ways is: unknown if any way is, outside if any is, else builtin.

    A way that adds nothing (:data:`_NOTHING`) is left out; where every way
    is one, the value adds nothing either.
    """
    kinds = [one for one in kinds if one != _NOTHING]
    if not kinds:
        return _NOTHING
    for kind in (_UNKNOWN, _QUEUE, _OUTSIDE):
        found = [detail for one, detail in kinds if one == kind]
        if found:
            return kind, found[0]
    held = {detail for _kind, detail in kinds}
    return _BUILTIN, held.pop() if len(held) == 1 else ""


def _element(container: tuple[str, str]) -> tuple[str, str]:
    """What one of what *container* holds is."""
    if container == _NOTHING:
        return _NOTHING
    kind, detail = container
    if kind == _OUTSIDE:
        return _OUTSIDE, f"an element of {detail}"
    if detail == _JSON:
        return container
    if kind == _BUILTIN and detail in ("str", "bytes", "range"):
        return _BUILTIN, "str" if detail == "str" else ""
    if kind == _UNKNOWN:
        return container
    return _UNKNOWN, f"an element of {_a(detail or kind)}"


def _a(held: str) -> str:
    """*held* with its article -- ``a list``, ``an iterator`` --; a phrase that has one (``an attribute of NoneType``)
    as it is."""
    if held.startswith(("a ", "an ", "what ", "the ")):
        return held
    return f"an {held}" if held[:1].lower() in tuple("aeiou") else f"a {held}"


def _entered(manager: tuple[str, str]) -> tuple[str, str]:
    """What ``with <manager> as x`` binds: a builtin's or an outside object's own ``__enter__`` result."""
    if manager == _NOTHING:
        return _NOTHING
    return (
        manager
        if manager[0] in (_BUILTIN, _OUTSIDE)
        else (_UNKNOWN, f"what entering {manager[1]} yields")
    )


#: A place in a container: its keys from the top, ``None`` for a position not said (an ``append``, a loop's element).
_Path = tuple[Optional[ast.expr], ...]

#: How many keys a path of :class:`_Holders` is read to; a longer one is not read.
_PATH_CUT = 8
#: How many holders one ask of :class:`_Holders` reads; more are not read.
_HOLDERS_CUT = 300
#: The slot of an item's value -- ``for key, value in d.items()``, ``for index, one in enumerate(xs)``.
_SECOND = ast.Constant(value=1)


def _part(gives: callfacts.Give, args: list[ast.expr]) -> Optional[tuple[_Path, _Path]]:
    """Where a builtin container's method's value is a part of its receiver (:class:`callfacts.Give`): (the place in
    the value, the receiver's place it holds) -- ``d.items()``: each element's second is one of ``d``'s values;
    ``d.get(k)`` is ``d``'s value at ``k``, ``xs.pop()`` one of its elements.  ``None`` for a view of the keys, which
    holds none of what it holds."""
    if gives.part == callfacts.SELF:
        return (), ()
    if gives.part == callfacts.VALUES:
        return (None,), (None,)
    if gives.part == callfacts.ITEMS:
        return (None, _SECOND), (None,)
    if gives.part == callfacts.ITEM:
        return (_SECOND,), (None,)
    if gives.part == callfacts.AT:
        return ((), (args[gives.key],)) if gives.key is not None and gives.key < len(args) else ((), (None,))
    return None


def _subscripted(expression: ast.expr) -> tuple[Optional[ast.Name], list[ast.expr]]:
    """The name ``d[k1][k2]`` is taken out of, and its keys in order; no name for another root, or a slice."""
    keys: list[ast.expr] = []
    while isinstance(expression, ast.Subscript):
        if isinstance(expression.slice, ast.Slice):
            return None, []
        keys.append(expression.slice)
        expression = expression.value
    return (expression if isinstance(expression, ast.Name) else None), keys[::-1]


def _same_key(one: Optional[ast.expr], other: Optional[ast.expr]) -> bool:
    """Whether two keys can be the same: two constants by value; any other pair can."""
    if isinstance(one, ast.Constant) and isinstance(other, ast.Constant):
        return bool(one.value == other.value)
    return True


def _position(key: Optional[ast.expr]) -> Optional[int]:
    """The slot a key names -- a literal integer from the start --, ``None`` for any other key."""
    if isinstance(key, ast.Constant) and type(key.value) is int and key.value >= 0:
        return key.value
    return None


def _matched(one: _Path, other: _Path) -> bool:
    """Whether two paths can be the same as far as the shorter goes."""
    return all(_same_key(a, b) for a, b in zip(one, other, strict=False))


def _spelled(path: _Path) -> tuple[Optional[str], ...]:
    return tuple(None if key is None else ast.dump(key) for key in path)


def _composed(
    ys: _Path, xs: _Path, at: _Path, of: _Path
) -> Optional[tuple[_Path, _Path]]:
    """Y at *ys* is X at *xs*, and X at *at* is the container at *of*: where Y holds the container's -- (Y's path, the
    container's) --, or ``None`` where the two are other places."""
    if not _matched(xs, at):
        return None
    if len(xs) <= len(at):
        return ys + at[len(xs) :], of
    return ys, of + xs[len(at) :]


def _chain(expression: ast.AST) -> Optional[tuple[ast.Name, _Path]]:
    """The name *expression* is a part of, and the keys down to it: ``d[k]``, ``d.get(k)``, ``d.setdefault(k, v)``,
    ``d.pop(k)``, ``d.copy()`` (the same keys); ``None`` for another root, or a slice."""
    keys: list[Optional[ast.expr]] = []
    while True:
        if isinstance(expression, ast.Subscript) and not isinstance(
            expression.slice, ast.Slice
        ):
            keys.append(expression.slice)
            expression = expression.value
        elif (
            isinstance(expression, ast.Call)
            and isinstance(expression.func, ast.Attribute)
            and not expression.keywords
            and not any(isinstance(one, ast.Starred) for one in expression.args)
            and (
                (
                    expression.func.attr in ("get", "setdefault", "pop")
                    and expression.args
                )
                or (expression.func.attr == "copy" and not expression.args)
            )
        ):
            if expression.func.attr != "copy":
                keys.append(expression.args[0])
            expression = expression.func.value
        else:
            break
    return (expression, tuple(keys[::-1])) if isinstance(expression, ast.Name) else None


def _mapping_key(key: Optional[ast.expr]) -> bool:
    """Whether *key* is one only a mapping is read at: a text.  No list, tuple or set holds anything there -- and none
    is read as a mapping: a merge is read only from a mapping or a set (:meth:`_Holders._merged`), so a list of pairs
    ``dict(pairs)`` or ``d.update(pairs)`` reads is not taken for one."""
    return isinstance(key, ast.Constant) and isinstance(key.value, str)


class _Scope:
    """One function's names, for :class:`_Holders`: every ``Name`` node by its id, and the names bound otherwise --
    ``except ... as``, an import, ``global`` / ``nonlocal``, a ``match`` capture, a nested ``def`` or class."""

    __slots__ = ("func", "names", "others")

    def __init__(self, func: ast.AST) -> None:
        self.func = func
        self.names: dict[str, list[ast.Name]] = {}
        self.others: set[str] = set()
        for node in walk(func):
            if isinstance(node, ast.Name):
                self.names.setdefault(node.id, []).append(node)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                self.others.add(node.name)
            elif isinstance(node, (ast.Global, ast.Nonlocal)):
                self.others.update(node.names)
            elif isinstance(node, ast.alias):
                self.others.add(node.asname or node.name.split(".")[0])
            elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
                self.others.add(node.name)
            elif isinstance(node, ast.MatchMapping) and node.rest:
                self.others.add(node.rest)
            elif node is not func and isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                self.others.add(node.name)

    def nested(self, node: ast.AST) -> bool:
        """Whether *node* is in a function, lambda or class nested in this one."""
        above = getattr(node, "parent", None)
        while above is not None and above is not self.func:
            if isinstance(
                above, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
            ):
                return True
            above = getattr(above, "parent", None)
        return False


class _Holders:
    """The values put at one place of a container, read from every name that holds it (R6).

    A holder is ``(file, function, name, at, of)``: the name's value at the keys *at* is the container's at *of*
    (:data:`_Path`).  Asked of ``d[k1]...[kn]`` in F, the walk starts from ``(F, d, (), ())`` and moves, each move
    read with the map's readers and its paths composed (:func:`_composed`):

    * up through a parameter, to what every call placed at the function hands it (:meth:`Calls.callers`, with
      :meth:`Calls.missed` empty and the function's name read nowhere as a value): a name, a literal, the default,
      or a ``**`` spread's entry at the parameter's name;
    * through the name's own bindings: an assignment, a slot it is unpacked from, a loop over a value (an element;
      of ``values()``, of ``items()``'s second slot, of a builtin's copy);
    * through the puts into a holder -- ``y[k] = v``, ``append`` / ``add`` / ``insert`` / ``extend`` / ``setdefault``
      / ``update``, ``+=``, ``|=`` -- on the name or on a part of it (``y[k]``, ``y.get(k)``, ``y.setdefault(k, [])``);
    * and wherever a holder's value or a part of it goes: bound to a name, put into a local's container, held in a
      literal, iterated, picked or copied by a builtin (a copy holds the same elements), ``or`` / ``if`` ``else``,
      handed to a call placed in the corpus -- its parameter; ``**`` to the parameter of the key's name.

    A place bears on the ask where its path on the container is a prefix of the asked keys: a strict one, or the
    asked place itself held inside another value -- that holder is walked; the asked place itself, reached up from
    a value bound or put there, is an answer.  Where a part that bears goes anywhere else -- returned, yielded,
    stored on an attribute, read in a nested function, handed to a call not placed, placed open, placed outside the
    corpus or on another thread, to a builtin not read here -- other code can put into it, and the ask is not read
    (``None``): no premise says what outside code puts into what it is handed.  But a return from the function the
    ask is in hands the container to its caller when the call ends, after every read of the call, and goes nowhere
    those reads see -- ``collect_existing_files`` returns the ``dataset_file_map`` it fills; a caller that fills
    what was returned and hands it in again does so by a name the walk reads up through the parameter
    (``getOutputFiles_JEDI``'s ``parallelOutMap``).  A callee's return is the container under a name of its caller
    the walk does not read, and is not read.  So is a binding not read here
    (``with``, ``except``, ``global``), a name the function does not bind, a module's name, ``self``, a merge from a
    value not read as a mapping or a set, a path longer than :data:`_PATH_CUT` keys, and more than
    :data:`_HOLDERS_CUT` holders.

    Flow-insensitive, and both ways: a name holding the container at one place holds whatever any value bound or put
    there holds, at any time -- more than the code may put there, never less.
    """

    def __init__(self, receivers: _Receivers, keys: tuple[ast.expr, ...]):
        self.receivers = receivers
        self.calls = receivers.calls
        self.keys: _Path = keys
        self.pending: list[tuple[str, ast.AST, str, _Path, _Path]] = []
        self.seen: set[
            tuple[int, str, tuple[Optional[str], ...], tuple[Optional[str], ...]]
        ] = set()
        self.found: list[tuple[str, ast.AST, ast.expr, _Path]] = []
        #: The function the ask is in, whose return goes nowhere its reads see (:meth:`_goes`).
        self.start: Optional[ast.AST] = None
        #: Holders a put on a value of a class not read needs to be among those read (:meth:`_put_into`).
        self.required: set[
            tuple[int, str, tuple[Optional[str], ...], tuple[Optional[str], ...]]
        ] = set()
        self.failed = False

    def values(
        self, file: str, func: ast.AST, name: str
    ) -> Optional[list[tuple[str, ast.AST, ast.expr, _Path]]]:
        """``(file, function, value, path)`` for each value at the asked place -- what *value* holds at *path* --;
        ``None`` where a move is not read."""
        self.start = func
        self._hold(file, func, name, (), ())
        while self.pending and not self.failed:
            self._read(*self.pending.pop())
        if self.failed or not self.required <= self.seen:
            return None
        return self.found

    # -- what a place is to the ask ---------------------------------------------

    def _bears(self, at: _Path, of: _Path) -> str:
        """``holder`` where a value at *at* of a holder at *of* is walked, ``value`` where it is the asked place, ``""``
        where it is another place."""
        if len(of) > len(self.keys) or not _matched(of, self.keys):
            return ""
        return "value" if len(of) == len(self.keys) and not at else "holder"

    def _unread(self, at: _Path, of: _Path) -> None:
        """A part at *of* goes where this walk does not follow: the ask is not read where it bears."""
        if self._bears(at, of) == "holder":
            self.failed = True

    def _hold(self, file: str, func: ast.AST, name: str, at: _Path, of: _Path) -> None:
        if self.failed or self._bears(at, of) != "holder":
            return
        if len(at) > _PATH_CUT or len(of) > _PATH_CUT:
            self.failed = True
            return
        key = (id(func), name, _spelled(at), _spelled(of))
        if key in self.seen:
            return
        self.seen.add(key)
        if len(self.seen) > _HOLDERS_CUT:
            self.failed = True
            return
        self.pending.append((file, func, name, at, of))

    # -- a value bound or put at a holder's place --------------------------------

    def _place(
        self, file: str, func: ast.AST, value: ast.expr, at: _Path, of: _Path
    ) -> None:
        """*value*, read in *func*, holds at *at* what the container holds at *of*."""
        bears = self._bears(at, of)
        if not bears or self.failed:
            return
        if bears == "value":
            self.found.append((file, func, value, ()))
            return
        if isinstance(value, ast.Constant) and value.value is None:
            return  # holds nothing
        if isinstance(value, ast.BoolOp):
            for one in value.values:
                self._place(file, func, one, at, of)
            return
        if isinstance(value, ast.IfExp):
            for one in (value.body, value.orelse):
                self._place(file, func, one, at, of)
            return
        if isinstance(value, ast.NamedExpr):
            self._place(file, func, value.value, at, of)
            return
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and (gives := callfacts.taken(value.func.attr)) is not None
            and gives.default is not None
            and len(value.args) > gives.default
        ):
            self._place(file, func, value.args[gives.default], at, of)  # the default
        chain = _chain(value)
        if chain is not None:
            root, path = chain
            self._hold(file, func, root.id, path + at, of)
            return
        moved = self._copied(file, func, value)
        if moved is not None:
            for source, ys, xs in moved:
                composed = _composed(ys, xs, at, of)
                if composed is not None:
                    self._place(file, func, source, *composed)
            return
        if at:
            self._read_down(file, func, value, at, of)
        else:
            self._entries(file, func, value, of)

    def _copied(
        self, file: str, func: ast.AST, value: ast.expr
    ) -> Optional[list[tuple[ast.expr, _Path, _Path]]]:
        """Where *value* is made of another value's parts: ``(source, ys, xs)``, the source at *ys* being the value
        at *xs* -- a builtin's copy or pick, a view, a slice, a sum."""
        if isinstance(value, ast.Subscript) and isinstance(value.slice, ast.Slice):
            return [(value.value, (None,), (None,))]
        if isinstance(value, ast.BinOp) and isinstance(value.op, (ast.Add, ast.Mult)):
            return [
                (one, (None,), (None,))
                for one in (value.left, value.right)
                if not isinstance(one, ast.Constant)
            ]
        if (
            not isinstance(value, ast.Call)
            or value.keywords
            and not (isinstance(value.func, ast.Name) and value.func.id == "sorted")
        ):
            return None
        called = value.func
        if (
            isinstance(called, ast.Attribute)
            and not value.args
            and (gives := callfacts.taken(called.attr)) is not None
        ):
            part = _part(gives, [])
            return [] if part is None else [(called.value, part[1], part[0])]
        if (
            isinstance(called, ast.Name)
            and len(value.args) == 1
            and not isinstance(value.args[0], ast.Starred)
            and (function := callfacts.builtin_function(self.calls.types, file, func, value)) is not None
        ):
            if function.gives == callfacts.COPY:
                return [(value.args[0], (None,), (None,))]
            if function.gives == callfacts.NUMBERED:
                return [(value.args[0], (None,), (None, _SECOND))]
            if function.gives == callfacts.ONE:
                return [(value.args[0], (None,), ())]
        return None

    def _read_down(
        self, file: str, func: ast.AST, value: ast.expr, at: _Path, of: _Path
    ) -> None:
        """A literal holding at *at* what the container holds at *of*: what it holds there."""
        key, below = at[0], at[1:]
        if isinstance(value, ast.Dict):
            for one, held in zip(value.keys, value.values, strict=True):
                if one is None:
                    self._place(file, func, held, at, of)  # {**held}
                elif _same_key(one, key):
                    self._place(file, func, held, below, of)
        elif isinstance(value, (ast.List, ast.Tuple, ast.Set)):
            if _mapping_key(key):
                return  # nothing: a sequence holds nothing at a text
            fixed = (
                _position(key)
                if isinstance(value, ast.Tuple)
                and not any(isinstance(one, ast.Starred) for one in value.elts)
                else None
            )
            for index, one in enumerate(value.elts):
                if isinstance(one, ast.Starred):
                    self._place(file, func, one.value, (None, *below), of)
                elif fixed is None or fixed == index:
                    self._place(file, func, one, below, of)
        elif isinstance(value, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            if _mapping_key(key):
                return  # nothing: a sequence holds nothing at a text
            self._place(file, func, value.elt, below, of)
        elif isinstance(value, ast.DictComp):
            self._place(file, func, value.value, below, of)
        elif self._constructed(value):
            for keyword in value.keywords:  # type: ignore[attr-defined]
                if keyword.arg is None:
                    self._place(file, func, keyword.value, at, of)
                elif _same_key(ast.Constant(value=keyword.arg), key):
                    self._place(file, func, keyword.value, below, of)
        else:
            self.found.append((file, func, value, (*at, *self.keys[len(of) :])))

    def _entries(self, file: str, func: ast.AST, value: ast.expr, of: _Path) -> None:
        """A value that is the container at *of*, a strict prefix of the asked keys: each value it holds."""
        if isinstance(value, ast.Dict):
            for one, held in zip(value.keys, value.values, strict=True):
                self._place(file, func, held, (), of if one is None else (*of, one))
        elif isinstance(value, (ast.List, ast.Tuple, ast.Set)):
            for one in value.elts:
                if isinstance(one, ast.Starred):
                    self._place(file, func, one.value, (None,), (*of, None))
                else:
                    self._place(file, func, one, (), (*of, None))
        elif isinstance(value, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            self._place(file, func, value.elt, (), (*of, None))
        elif isinstance(value, ast.DictComp):
            self._place(file, func, value.value, (), (*of, None))
        elif self._constructed(value):
            for keyword in value.keywords:  # type: ignore[attr-defined]
                self._place(
                    file,
                    func,
                    keyword.value,
                    (),
                    of
                    if keyword.arg is None
                    else (*of, ast.Constant(value=keyword.arg)),
                )
        else:
            self.found.append((file, func, value, self.keys[len(of) :]))

    def _constructed(self, value: ast.expr) -> bool:
        """``dict()``, ``list()``, ``set()``, ``dict(k=v, ...)``: a builtin container from keywords alone."""
        return (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id in ("dict", "list", "set")
            and not value.args
            and (value.func.id == "dict" or not value.keywords)
        )

    def _merged(
        self, file: str, func: ast.AST, source: ast.expr, at: _Path, of: _Path
    ) -> None:
        """*source*'s entries merged at the same keys into a value holding at *at* what the container holds at *of*:
        read where *source* is a mapping or a set -- a literal, or so read --; a list of pairs is not read."""
        if self._bears(at, of) != "holder":
            return  # the asked place's own content
        if isinstance(
            source, (ast.Dict, ast.DictComp, ast.Set, ast.SetComp)
        ) or self._constructed(source):
            self._place(file, func, source, at, of)
            return
        kind = self.receivers.of(file, func, source)
        if kind[0] == _BUILTIN and kind[1] in ("dict", "set", "frozenset"):
            self._place(file, func, source, at, of)
        else:
            self.failed = True

    # -- one holder ----------------------------------------------------------------

    def _read(self, file: str, func: ast.AST, name: str, at: _Path, of: _Path) -> None:
        if isinstance(func, ast.Module) or name in ("self", "cls"):
            self.failed = True
            return
        scope = self.receivers._scope(func)
        nodes = scope.names.get(name, [])
        parameters = _parameters(func)
        if name in scope.others or (
            name not in parameters
            and not any(isinstance(node.ctx, ast.Store) for node in nodes)
        ):
            self.failed = True
            return
        if name in parameters:
            self._up(file, func, name, at, of)
        for node in nodes:
            if self.failed:
                return
            if scope.nested(node):
                self.failed = True
            elif isinstance(node.ctx, ast.Store):
                self._bound(file, func, node, at, of)
            elif isinstance(node.ctx, ast.Load):
                self._goes(file, func, node, at, of)

    def _up(self, file: str, func: ast.AST, name: str, at: _Path, of: _Path) -> None:
        """A parameter: what every call placed at *func* hands it."""
        arguments = func.args  # type: ignore[attr-defined]
        if any(
            one is not None and one.arg == name
            for one in (arguments.vararg, arguments.kwarg)
        ):
            self.failed = True
            return
        target = self.receivers._target(file, func)
        try:
            sites = self.calls.callers(target)
            missed = self.calls.missed(target)
        except Exception:  # noqa: BLE001 -- a call not read is a move not read
            self.failed = True
            return
        if not sites or missed or self.receivers._read_as_value(func):
            self.failed = True
            return
        tree = self.receivers.types.tree(target.file)
        for site in sites:
            try:
                placed = self.calls.callees(site.file, site.func, site.call)
            except Exception:  # noqa: BLE001
                self.failed = True
                return
            handed = (
                None if placed.thread else _handed(site, target, placed.on_class, name)
            )
            if handed is None:
                self.failed = True
                return
            for value, path, default in handed:
                where, scope = (
                    (target.file, tree) if default else (site.file, site.func)
                )
                if scope is None:
                    self.failed = True
                    return
                self._place(where, scope, value, path + at, of)

    def _bound(
        self, file: str, func: ast.AST, node: ast.expr, at: _Path, of: _Path
    ) -> None:
        """A binding of a holder's name, or a put at a holder's place (``y[k] = ...``): *node*, a target, holds at
        *at* what the container holds at *of*."""
        slots: list[Optional[ast.expr]] = []
        here: ast.AST = node
        parent = getattr(here, "parent", None)
        while isinstance(parent, (ast.Tuple, ast.List)):
            if any(isinstance(one, ast.Starred) for one in parent.elts):
                slots.insert(0, None)
            else:
                slots.insert(0, ast.Constant(value=parent.elts.index(here)))  # type: ignore[arg-type]
            here = parent
            parent = getattr(here, "parent", None)
        if isinstance(parent, ast.Starred):
            self._unread(at, of)
            return
        path = tuple(slots)
        if isinstance(parent, ast.Assign) or (
            isinstance(parent, ast.AnnAssign) and parent.target is here
        ):
            if parent.value is not None:
                self._place(file, func, parent.value, path + at, of)
        elif isinstance(parent, ast.AugAssign) and parent.target is here:
            if isinstance(parent.op, ast.Add):
                moved = _composed((None,), (None,), at, of)
                if moved is not None:
                    self._place(file, func, parent.value, *moved)
            elif isinstance(parent.op, ast.BitOr):
                self._merged(file, func, parent.value, at, of)
        elif (
            isinstance(parent, (ast.For, ast.AsyncFor, ast.comprehension))
            and parent.target is here
        ):
            self._place(file, func, parent.iter, (None, *path, *at), of)
        elif isinstance(parent, ast.NamedExpr) and parent.target is here:
            self._place(file, func, parent.value, at, of)
        else:
            self._unread(at, of)

    # -- where a holder's value goes ------------------------------------------------

    def _goes(
        self, file: str, func: ast.AST, node: ast.Name, at: _Path, of: _Path
    ) -> None:
        """A read of a holder's name, followed up the expression it is in: *here* at *at* is the container at *of*."""
        here: ast.AST = node
        while not self.failed and self._bears(at, of):
            parent = getattr(here, "parent", None)
            step: Optional[tuple[_Path, _Path]] = None  # the parent at ys is here at xs
            if isinstance(parent, ast.Subscript) and parent.value is here:
                sliced = isinstance(parent.slice, ast.Slice)
                if isinstance(parent.ctx, ast.Store):
                    moved = None if sliced else _composed((), (parent.slice,), at, of)
                    if sliced:
                        self._unread(at, of)
                    elif moved is not None:
                        self._bound(file, func, parent, *moved)
                    return
                if not isinstance(parent.ctx, ast.Load):
                    return
                step = ((None,), (None,)) if sliced else ((), (parent.slice,))
            elif isinstance(parent, ast.Attribute) and parent.value is here:
                call = getattr(parent, "parent", None)
                if not (isinstance(call, ast.Call) and call.func is parent):
                    return  # an attribute of it, read or set
                step = self._method(file, func, call, parent.attr, at, of)
                if step is None:
                    return
                parent = call
            elif isinstance(parent, ast.Call):
                if parent.func is here:
                    return
                step = self._handed_to(file, func, parent, here, at, of)
                if step is None:
                    return
            elif isinstance(parent, ast.Starred):
                call = getattr(parent, "parent", None)
                if isinstance(call, ast.Call):
                    step = self._handed_to(file, func, call, parent, at, of)
                    if step is None:
                        return
                    parent = call
                elif isinstance(call, (ast.List, ast.Tuple, ast.Set)):
                    step = ((None,), (None,))
                    parent = call
                else:
                    self._unread(at, of)
                    return
            elif isinstance(parent, ast.keyword):
                call = getattr(parent, "parent", None)
                if not isinstance(call, ast.Call):
                    self._unread(at, of)
                    return
                step = self._handed_to(file, func, call, parent, at, of)
                if step is None:
                    return
                parent = call
            elif isinstance(parent, ast.BoolOp):
                step = ((), ())
            elif isinstance(parent, ast.IfExp):
                if parent.test is here:
                    return
                step = ((), ())
            elif isinstance(parent, ast.NamedExpr) and parent.value is here:
                self._hold(file, func, parent.target.id, at, of)
                step = ((), ())
            elif isinstance(parent, (ast.List, ast.Tuple, ast.Set)):
                step = ((None,), ())
            elif isinstance(parent, ast.Dict):
                if any(one is here for one in parent.keys):
                    return
                index = next(i for i, one in enumerate(parent.values) if one is here)
                held = parent.keys[index]
                step = ((), ()) if held is None else ((held,), ())
            elif isinstance(parent, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
                step = ((None,), ())
            elif isinstance(parent, ast.DictComp):
                if parent.value is not here:
                    return
                step = ((None,), ())
            elif isinstance(parent, (ast.comprehension, ast.For, ast.AsyncFor)):
                if parent.iter is here:
                    self._assigned(file, func, parent.target, (), (None,), at, of)
                return
            elif (
                isinstance(parent, (ast.Assign, ast.AnnAssign)) and parent.value is here
            ):
                targets = (
                    parent.targets
                    if isinstance(parent, ast.Assign)
                    else [parent.target]
                )
                for target in targets:
                    self._assigned(file, func, target, (), (), at, of)
                return
            elif isinstance(parent, ast.AugAssign) and parent.value is here:
                if isinstance(parent.op, ast.Add):
                    moved = _composed((None,), (None,), at, of)
                    if moved is not None:
                        self._into(file, func, parent.target, *moved)
                elif isinstance(parent.op, ast.BitOr):
                    self._unread(at, of)
                return
            elif isinstance(parent, ast.BinOp):
                if isinstance(parent.op, ast.Mod):
                    return  # a text formatted
                step = ((None,), (None,))
            elif isinstance(parent, ast.Subscript) and parent.slice is here:
                return  # a key
            elif isinstance(
                parent,
                (
                    ast.Expr, ast.Compare, ast.UnaryOp, ast.FormattedValue, ast.JoinedStr, ast.If, ast.While,
                    ast.Assert, ast.Delete, ast.Slice,
                ),
            ):  # fmt: skip
                return
            elif isinstance(parent, ast.Return) and func is self.start:
                return
            else:
                self._unread(at, of)
                return
            moved = _composed(step[0], step[1], at, of)
            if moved is None:
                return
            at, of = moved
            here = parent

    def _assigned(
        self,
        file: str,
        func: ast.AST,
        target: ast.expr,
        ys: _Path,
        xs: _Path,
        at: _Path,
        of: _Path,
    ) -> None:
        """*target* at *ys* is bound to the value at *xs* below one holding at *at* what the container holds at *of*:
        a name, a slot it is unpacked into, ``*rest`` (its elements), a place put at."""
        if isinstance(target, (ast.Tuple, ast.List)):
            if ys:
                self._unread(at, of)
                return
            starred = any(isinstance(one, ast.Starred) for one in target.elts)
            for index, one in enumerate(target.elts):
                if isinstance(one, ast.Starred):
                    self._assigned(file, func, one.value, (None,), (*xs, None), at, of)
                else:
                    slot = None if starred else ast.Constant(value=index)
                    self._assigned(file, func, one, (), (*xs, slot), at, of)
            return
        moved = _composed(ys, xs, at, of)
        if moved is None:
            return
        if isinstance(target, ast.Name):
            self._hold(file, func, target.id, *moved)
        else:
            self._into(file, func, target, *moved)

    def _into(
        self, file: str, func: ast.AST, target: ast.expr, at: _Path, of: _Path
    ) -> None:
        """A value put at *target* (``y[k]``, ``self.x``) holds at *at* what the container holds at *of*."""
        chain = _chain(target)
        if chain is None:
            self._unread(at, of)  # stored on an attribute
            return
        root, path = chain
        self._hold(file, func, root.id, path + at, of)

    def _method(
        self,
        file: str,
        func: ast.AST,
        call: ast.Call,
        method: str,
        at: _Path,
        of: _Path,
    ) -> Optional[tuple[_Path, _Path]]:
        """``here.method(...)``, *here* at *at* the container at *of*: its puts, and where its value is a part of it
        (the step from it, as :meth:`_goes` takes one); ``None`` where its value holds nothing of it."""
        args = call.args
        if any(isinstance(one, ast.Starred) for one in args):
            self._unread(at, of)
            return None
        if (method in ("append", "add") and len(args) == 1) or (
            method == "insert" and len(args) == 2
        ):
            moved = _composed((), (None,), at, of)
            if moved is not None:
                self._place(file, func, args[-1], *moved)
        elif method == "extend" and len(args) == 1:
            moved = _composed((None,), (None,), at, of)
            if moved is not None:
                self._place(file, func, args[0], *moved)
        elif method == "setdefault" and args:
            if len(args) == 2:
                moved = _composed((), (args[0],), at, of)
                if moved is not None:
                    self._place(file, func, args[1], *moved)
            return (), (args[0],)
        elif method == "update":
            for one in args:
                self._merged(file, func, one, at, of)
            for keyword in call.keywords:
                if keyword.arg is None:
                    self._merged(file, func, keyword.value, at, of)
                else:
                    moved = _composed((), (ast.Constant(value=keyword.arg),), at, of)
                    if moved is not None:
                        self._place(file, func, keyword.value, *moved)
        elif (gives := callfacts.taken(method)) is not None and gives.part != callfacts.KEYS:
            return _part(gives, args)
        elif method.startswith("__"):
            self._unread(at, of)
        return None

    def _handed_to(
        self,
        file: str,
        func: ast.AST,
        call: ast.Call,
        argument: ast.AST,
        at: _Path,
        of: _Path,
    ) -> Optional[tuple[_Path, _Path]]:
        """*argument* -- the value, ``*value`` or ``name=value`` -- handed to *call*, the value at *at* being the
        container at *of*: the parameters it binds, a put into another container; the step from it where the call's
        value is made of it (a builtin's copy or pick); ``None`` where it holds nothing of it."""
        called = call.func
        if isinstance(called, ast.Name) and self.receivers._is_builtin(
            file, func, called.id
        ):
            function = callfacts.BUILTIN_FUNCTIONS.get(called.id)
            gives = function.gives if function is not None else None
            if gives in callfacts.HOLDING_NONE:
                return None  # a text, a number, a count or a test of it
            if function is not None and gives == callfacts.ATTRIBUTE:
                # getattr's default may be the value; an attribute of a builtin container is a method of it.
                default = function.default
                kept = default is not None and len(call.args) > default and argument is call.args[default]
                return ((), ()) if kept else None
            if isinstance(argument, (ast.keyword, ast.Starred)):
                self._unread(at, of)
                return None
            if function is not None and gives == callfacts.ONE:
                elements, whole = callfacts.picked(function, call)
                if any(argument is one for one in whole):
                    return (), ()  # max(a, b), next(it, default): one of them
                if any(argument is one for one in elements):
                    return (), (None,)
            if len(call.args) == 1 and gives == callfacts.COPY:
                return (None,), (None,)
            if len(call.args) == 1 and gives == callfacts.NUMBERED:
                return (None, _SECOND), (None,)
            self._unread(at, of)
            return None
        try:
            placed = self.calls.callees(file, func, call)
        except Exception:  # noqa: BLE001
            self._unread(at, of)
            return None
        if placed.targets and not placed.unplaced and not placed.thread:
            for target in placed.targets:
                self._parameters(call, target, placed.on_class, argument, at, of)
            return None
        if isinstance(called, ast.Attribute) and not placed.targets:
            receiver = self.receivers.of(file, func, called.value)
            if receiver[0] == _BUILTIN and receiver[1] in _SCALARS:
                return None  # a text's method: ", ".join(xs)
            if receiver[0] == _QUEUE:
                self._unread(at, of)  # handed to whoever gets it: HOLE_QUEUE
                return None
            meant = callfacts.meaning(self.calls, file, func, call, placed)
            if meant.kind in (callfacts.LOGGER, callfacts.CLIENT):
                return None  # a line written, another system handed it: it holds nothing of it here
            if (
                receiver[0] == _BUILTIN
                and not called.attr.startswith("__")
                and meant.kind != callfacts.LIBRARY
            ):
                # The builtin this reading says the receiver is: its type's method.
                method = callfacts.builtin_method(receiver[1], called.attr)
                if method is not None and method.puts:
                    self._put_into(file, func, call, argument, at, of, method)
                return None  # a key, a test: y.index(x), y.remove(x)
            if (
                meant.kind
                in (callfacts.LIBRARY, callfacts.BUILTIN, callfacts.ASSUMED)
                and meant.method is not None
                and meant.method.puts
            ):
                assumed = meant.kind == callfacts.ASSUMED
                self._put_into(
                    file, func, call, argument, at, of, meant.method, held=assumed
                )
                return None
        self._unread(at, of)
        return None

    def _put_into(
        self,
        file: str,
        func: ast.AST,
        call: ast.Call,
        argument: ast.AST,
        at: _Path,
        of: _Path,
        method: callfacts.Method,
        held: bool = False,
    ) -> None:
        """``y.append(here)`` and the like: *y* holds it where *method* puts it (:class:`callfacts.Put`) -- one
        element, its elements as elements, the value under the key handed beside it, *y* itself merged or keeping it
        whole; a key holds nothing of it.

        *held*: *y*'s class is not read (``kwargs.update(new_kwargs)``, ``kwargs`` one of a list declared ``Any``).
        The put is read only where *y* is a holder of the same place anyway, reached by another move: its own walk
        reads what it is bound to, and any binding not a literal makes the answer what it holds, not a builtin.
        Asked at the walk's end (:meth:`values`), so the order the holders are read in does not decide it."""
        called = call.func
        if isinstance(argument, ast.Starred):
            self._unread(at, of)
            return
        value = argument.value if isinstance(argument, ast.keyword) else argument
        links: list[tuple[_Path, _Path]] = []
        for one, put, key, keyword in callfacts.put_arguments(method, call):
            if one is not value:
                continue
            if put.into == callfacts.ELEMENT:
                links.append(((None,), ()))
            elif put.into == callfacts.ELEMENTS:
                links.append(((None,), (None,)))
            elif put.into == callfacts.VALUE:
                under = key if key is not None else ast.Constant(value=keyword) if keyword else None
                links.append(((under,), ()))
            elif put.into in (callfacts.MERGE, callfacts.WHOLE):
                links.append(((), ()))
        for link in links:
            moved = _composed(link[0], link[1], at, of)
            if moved is None:
                continue
            if not held:
                self._into(file, func, called.value, *moved)  # type: ignore[attr-defined]
                continue
            chain = _chain(called.value)  # type: ignore[attr-defined]
            if chain is None:
                self._unread(*moved)
            elif self._bears(chain[1] + moved[0], moved[1]) == "holder":
                self.required.add(
                    (
                        id(func),
                        chain[0].id,
                        _spelled(chain[1] + moved[0]),
                        _spelled(moved[1]),
                    )
                )

    def _parameters(
        self,
        call: ast.Call,
        target: C.Target,
        on_class: bool,
        argument: ast.AST,
        at: _Path,
        of: _Path,
    ) -> None:
        """The parameters of *target* that *argument* of *call* binds hold at *at* what the container holds at *of*."""
        arguments = target.func.args  # type: ignore[attr-defined]
        positional = [one.arg for one in C._positional(target, on_class)]
        keyword_only = [one.arg for one in arguments.kwonlyargs]
        named = {one.arg for one in call.keywords if one.arg}
        stars = [i for i, one in enumerate(call.args) if isinstance(one, ast.Starred)]
        first_star = stars[0] if stars else len(call.args)
        file, func = target.file, target.func
        if isinstance(argument, ast.keyword):
            if argument.arg is None:  # **value
                for name in [*positional[first_star:], *keyword_only]:
                    if name not in named:
                        moved = _composed((), (ast.Constant(value=name),), at, of)
                        if moved is not None:
                            self._hold(file, func, name, *moved)
                if arguments.kwarg is not None:
                    self._hold(file, func, arguments.kwarg.arg, at, of)
            elif argument.arg in positional or argument.arg in keyword_only:
                self._hold(file, func, argument.arg, at, of)
            elif arguments.kwarg is not None:
                moved = _composed((ast.Constant(value=argument.arg),), (), at, of)
                if moved is not None:
                    self._hold(file, func, arguments.kwarg.arg, *moved)
            return
        if isinstance(argument, ast.Starred):  # *value
            for name in positional[first_star:]:
                if name not in named:
                    moved = _composed((), (None,), at, of)
                    if moved is not None:
                        self._hold(file, func, name, *moved)
            if arguments.vararg is not None:
                self._unread(at, of)
            return
        index = next(i for i, one in enumerate(call.args) if one is argument)
        if index > first_star:
            self._unread(at, of)
        elif index < len(positional):
            self._hold(file, func, positional[index], at, of)
        elif arguments.vararg is not None:
            self._unread(at, of)


class _Keys(_Holders):
    """The keys a mapping may hold, read from every name that holds it (E15): :class:`_Holders`' walk, asked of the
    mapping itself.

    A holder is a name holding the mapping whole, at a path or not; a value put at one of its keys -- ``d[k] = v``,
    ``setdefault(k, ...)``, a dict literal's entry, ``update`` with a literal or keywords -- is an answer, and its key
    is read (:func:`_key_texts`).  What is read out of the mapping, and where it goes, is no move: ``varMap[f":{k}"] =
    param[k]`` puts a value, not the mapping.  ``None`` where a move is not read, as for :class:`_Holders`, where the
    mapping is a value not read as one (a call's result), and where a key is not read as texts.
    """

    def __init__(self, receivers: "_Receivers"):
        super().__init__(receivers, (None,))
        #: ``(file, function, key)`` of each put; ``None`` for a key not said (a comprehension's).
        self.keyed: list[tuple[str, ast.AST, Optional[ast.expr]]] = []

    def read_keys(
        self, file: str, func: ast.AST, name: str
    ) -> Optional[frozenset[str]]:
        found = self.values(file, func, name)
        if found is None or any(path for _file, _func, _value, path in found):
            return None
        holders: dict[int, set[str]] = {}
        for held, spelled, at, of in self.seen:
            if not at and not of:
                holders.setdefault(held, set()).add(spelled)
        texts: set[str] = set()
        for _file, scope, key in self.keyed:
            read = (
                _key_texts(scope, key, holders.get(id(scope), set()))
                if key is not None
                else None
            )
            if read is None:
                return None
            texts |= read
        return frozenset(texts)

    def _bears(self, at: _Path, of: _Path) -> str:
        if not of:
            return "holder"
        return "value" if len(of) == 1 and not at else ""

    def _place(
        self, file: str, func: ast.AST, value: ast.expr, at: _Path, of: _Path
    ) -> None:
        if not self.failed and self._bears(at, of) == "value":
            self.keyed.append((file, func, of[0]))
        super()._place(file, func, value, at, of)


def _key_texts(func: ast.AST, key: ast.expr, holders: set[str]) -> Optional[set[str]]:
    """The texts *key*, a key put into a mapping in *func*, can be: a literal, or a name bound only to literals -- by
    assignment, or as a loop's target, whole or at a slot, over a literal or a local bound only to literals.  A key
    taken from the mapping itself (``for k in list(d): d[k] = ...``, *holders* the names holding it whole there) adds
    none.  ``None`` where one is not read so."""
    if isinstance(key, ast.Constant):
        return {key.value} if isinstance(key.value, str) else None
    if not isinstance(key, ast.Name) or key.id in _parameters(func):
        return None
    stores = [
        node
        for node in C.own_nodes(func)
        if isinstance(node, ast.Name)
        and node.id == key.id
        and not isinstance(node.ctx, ast.Load)
    ]
    if not stores or any(
        isinstance(node, (ast.Global, ast.Nonlocal)) and key.id in node.names
        for node in walk(func)
    ):
        return None
    texts: set[str] = set()
    for node in stores:
        parent = getattr(node, "parent", None)
        if isinstance(parent, (ast.Assign, ast.AnnAssign)) and node is not parent.value:
            value = parent.value
            if not (isinstance(value, ast.Constant) and isinstance(value.value, str)):
                return None
            texts.add(value.value)
            continue
        slot: Optional[int] = None
        if isinstance(parent, ast.Tuple) and not any(
            isinstance(one, ast.Starred) for one in parent.elts
        ):
            slot = next(i for i, one in enumerate(parent.elts) if one is node)
            node, parent = parent, getattr(parent, "parent", None)
        if not (isinstance(parent, (ast.For, ast.AsyncFor)) and parent.target is node):
            return None
        read = _iterated_texts(func, parent.iter, slot, holders)
        if read is None:
            return None
        texts |= read
    return texts


def _iterated_texts(
    func: ast.AST, over: ast.expr, slot: Optional[int], holders: set[str]
) -> Optional[set[str]]:
    """:func:`_key_texts` of a loop's target over *over*, at *slot* of each element (the element where ``None``)."""
    taken = ""
    if isinstance(over, ast.Name) and slot is None:
        taken = over.id  # for k in d
    elif isinstance(over, ast.Call) and not over.keywords:
        called = over.func
        if (
            isinstance(called, ast.Attribute)
            and isinstance(called.value, ast.Name)
            and not over.args
            and called.attr
            == ("keys" if slot is None else "items" if slot == 0 else None)
        ):
            taken = called.value.id  # for k in d.keys(), for k, v in d.items()
        elif (
            isinstance(called, ast.Name)
            and called.id in ("list", "tuple", "sorted", "set")
            and slot is None
            and len(over.args) == 1
            and isinstance(over.args[0], ast.Name)
        ):
            taken = over.args[0].id  # for k in list(d)
    if taken and taken in holders:
        return set()  # the mapping's own keys
    sequences: list[ast.expr]
    if isinstance(over, (ast.List, ast.Tuple, ast.Set)):
        sequences = [over]
    elif isinstance(over, ast.Name) and over.id not in _parameters(func):
        uses = [
            node
            for node in walk(func)
            if isinstance(node, ast.Name) and node.id == over.id
        ]
        bound = [
            getattr(node, "parent", None)
            for node in uses
            if isinstance(node.ctx, ast.Store)
        ]
        read = [node for node in uses if isinstance(node.ctx, ast.Load)]
        if (
            not bound
            or not all(
                isinstance(one, (ast.Assign, ast.AnnAssign))
                and isinstance(one.value, (ast.List, ast.Tuple))
                for one in bound
            )
            or not all(
                isinstance(getattr(node, "parent", None), (ast.For, ast.AsyncFor))
                and node.parent.iter is node  # type: ignore[attr-defined]
                for node in read
            )
        ):
            return None
        sequences = [one.value for one in bound]  # type: ignore[union-attr]
    else:
        return None
    texts: set[str] = set()
    for sequence in sequences:
        for element in sequence.elts:  # type: ignore[attr-defined]
            if slot is not None:
                if not (
                    isinstance(element, (ast.Tuple, ast.List))
                    and len(element.elts) > slot
                    and not any(isinstance(one, ast.Starred) for one in element.elts)
                ):
                    return None
                element = element.elts[slot]
            if not (
                isinstance(element, ast.Constant) and isinstance(element.value, str)
            ):
                return None
            texts.add(element.value)
    return texts


def _handed(
    site: C.CallSite, target: C.Target, on_class: bool, name: str
) -> Optional[list[tuple[ast.expr, _Path, bool]]]:
    """What the call at *site* hands *target*'s parameter *name*: ``(value, path, default)`` -- the value, or the value
    at *path* below it (a ``**`` spread's entry at the parameter's name), *default* where it is the parameter's
    default, read in the module --; ``None`` where a ``*`` spread may fill it, or it is the object the call is made on.

    A ``*`` spread does not fill a parameter a ``**`` spread at the call holds on every way to it
    (:func:`_always_given`): Python would raise, given it twice -- ``self.insertNewJob(*args, **kwargs)`` after
    ``kwargs.update(new_kwargs)``, ``new_kwargs`` spelling ``extracted_sqls``."""
    call = site.call
    arguments = target.func.args  # type: ignore[attr-defined]
    positional = [one.arg for one in C._positional(target, on_class)]
    if name == C.receiver_parameter(target) and name not in positional:
        return None
    for keyword in call.keywords:
        if keyword.arg == name:
            return [(keyword.value, (), False)]
    stars = [i for i, one in enumerate(call.args) if isinstance(one, ast.Starred)]
    first_star = stars[0] if stars else len(call.args)
    if name in positional:
        index = positional.index(name)
        if index < first_star:
            return [(call.args[index], (), False)]
        if stars and not any(
            _always_given(site.func, call, keyword.value, name)
            for keyword in call.keywords
            if keyword.arg is None
        ):
            return None
    found: list[tuple[ast.expr, _Path, bool]] = [
        (keyword.value, (ast.Constant(value=name),), False)
        for keyword in call.keywords
        if keyword.arg is None
    ]
    defaults = dict(
        zip(
            [
                *[one.arg for one in arguments.posonlyargs],
                *[one.arg for one in arguments.args],
            ][
                len(arguments.posonlyargs)
                + len(arguments.args)
                - len(arguments.defaults) :
            ],
            arguments.defaults,
            strict=True,
        )
    )
    defaults.update(
        (one.arg, default)
        for one, default in zip(
            arguments.kwonlyargs, arguments.kw_defaults, strict=True
        )
        if default is not None
    )
    if name in defaults:
        found.append((defaults[name], (), True))
    return found


def _always_given(func: ast.AST, call: ast.Call, spread: ast.expr, key: str) -> bool:
    """Whether the name *spread*, spread by ``**`` at *call* in *func*, holds *key* on every way to the call.

    A put of the key -- ``s[key] = v``, ``s.update(key=v)``, ``s.setdefault(key, v)``, ``s.update(d)`` with *d* a
    dict literal spelling it or a name bound only to such literals and never emptied, a binding of *s* to such a
    literal -- after every other binding of the name and every removal from it (``pop``, ``popitem``, ``clear``,
    ``del``), read by :func:`reaching.reaching` with the puts as killers.  No where the name is not bound in *func*,
    or a nested function reads it.
    """
    if not isinstance(spread, ast.Name) or isinstance(func, ast.Module):
        return False
    name = spread.id
    statement = _statement(call)
    puts: list[ast.AST] = []
    sites: list[ast.AST] = []
    if name in _parameters(func):
        sites.append(func.args)  # type: ignore[attr-defined]
    for node in walk(func):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
            and node is not func
        ):
            if any(
                isinstance(one, ast.Name) and one.id == name for one in walk(node)
            ):
                return False
        if (
            isinstance(node, ast.Name)
            and node.id == name
            and isinstance(node.ctx, ast.Store)
        ):
            binder = _statement(node)
            if (
                isinstance(binder, ast.Assign)
                and node in binder.targets
                and _spells(func, binder.value, key)
            ):
                puts.append(binder)
            elif binder is not None:
                sites.append(binder)
        elif isinstance(node, ast.Assign) and any(
            isinstance(one, ast.Subscript)
            and isinstance(one.value, ast.Name)
            and one.value.id == name
            and isinstance(one.slice, ast.Constant)
            and one.slice.value == key
            for one in node.targets
        ):
            puts.append(node)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == name
        ):
            method, args = node.func.attr, node.args
            put = (
                (method == "update" and any(one.arg == key for one in node.keywords))
                or (
                    method == "update"
                    and len(args) == 1
                    and _spells(func, args[0], key)
                )
                or (
                    method == "setdefault"
                    and args
                    and isinstance(args[0], ast.Constant)
                    and args[0].value == key
                )
            )
            held = _statement(node)
            if put and held is not None:
                puts.append(held)
            elif (
                method in ("pop", "popitem", "clear", "__delitem__")
                and held is not None
            ):
                sites.append(held)
        elif isinstance(node, ast.Delete) and any(
            name in {one.id for one in walk(target) if isinstance(one, ast.Name)}
            for target in node.targets
        ):
            sites.append(node)
    if statement is None or not puts or not sites:
        return False
    return not reaching(statement, sites, puts)


def _spells(func: ast.AST, value: ast.expr, key: str) -> bool:
    """Whether *value* is a dict literal spelling *key*, or a name *func* binds only to such literals and never
    empties or takes the key out of."""
    if isinstance(value, ast.Dict):
        return any(
            isinstance(one, ast.Constant) and one.value == key for one in value.keys
        )
    if not isinstance(value, ast.Name) or isinstance(func, ast.Module):
        return False
    if value.id in _parameters(func):
        return False
    bound = False
    for node in walk(func):
        if (
            isinstance(node, ast.Name)
            and node.id == value.id
            and not isinstance(node.ctx, ast.Load)
        ):
            binder = _statement(node)
            if not (
                isinstance(binder, ast.Assign)
                and node in binder.targets
                and isinstance(binder.value, ast.Dict)
                and _spells(func, binder.value, key)
            ):
                return False
            bound = True
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == value.id
            and node.func.attr in ("pop", "popitem", "clear", "__delitem__")
        ):
            return False
        elif isinstance(node, ast.Delete) and any(
            isinstance(one, ast.Subscript)
            and isinstance(one.value, ast.Name)
            and one.value.id == value.id
            for one in node.targets
        ):
            return False
    return bound


def _statement(node: ast.AST) -> Optional[ast.stmt]:
    """The statement *node* is in."""
    current: Optional[ast.AST] = node
    while current is not None and not isinstance(current, ast.stmt):
        current = getattr(current, "parent", None)
    return current  # type: ignore[return-value]


def _parameters(func: ast.AST) -> dict[str, Optional[ast.expr]]:
    arguments = getattr(func, "args", None)
    if arguments is None:
        return {}
    every = [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]
    every += [one for one in (arguments.vararg, arguments.kwarg) if one is not None]
    return {one.arg: one.annotation for one in every}


def _loop_sources(func: ast.AST, name: str) -> list[tuple[ast.expr, bool]]:
    """``(source, entered)`` for each loop, comprehension or ``with`` in *func* that binds *name*."""
    found: list[tuple[ast.expr, bool]] = []
    for node in walk(func):
        if isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
            if any(
                isinstance(leaf, ast.Name) and leaf.id == name
                for leaf in walk(node.target)
            ):
                found.append((node.iter, False))
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None and any(
                    isinstance(leaf, ast.Name) and leaf.id == name
                    for leaf in walk(item.optional_vars)
                ):
                    found.append((item.context_expr, True))
    return found


def _module_values(
    tree: ast.Module, name: str
) -> list[ast.expr | tuple[str, ast.expr]]:
    """Every value the module's own body binds to *name*, ``("annotation", node)`` where only an annotation does."""
    found: list[ast.expr | tuple[str, ast.expr]] = []
    pending: list[ast.AST] = list(tree.body)
    while pending:
        node = pending.pop(0)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(node, ast.Assign) and any(
            isinstance(one, ast.Name) and one.id == name for one in node.targets
        ):
            found.append(node.value)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
        ):
            found.append(
                node.value
                if node.value is not None
                else ("annotation", node.annotation)
            )
        pending.extend(
            child for child in ast.iter_child_nodes(node) if isinstance(child, ast.stmt)
        )
    return found


def _calls_in(func: ast.AST) -> Iterator[tuple[ast.Call, str]]:
    """``(call, how)`` for every call *func* makes itself, a lambda's body included (``how`` ``lambda``)."""
    pending: list[tuple[ast.AST, str]] = [
        (child, "call") for child in ast.iter_child_nodes(func)
    ]
    while pending:
        node, how = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(node, ast.Lambda):
            how = "lambda"
        if isinstance(node, ast.Call):
            yield node, how
        pending.extend((child, how) for child in ast.iter_child_nodes(node))


def _stores_in(func: ast.AST) -> Iterator[tuple[ast.Attribute | ast.Call, ast.expr, Optional[ast.expr]]]:
    """``(store, name, value)`` for each attribute *func* stores itself -- ``x.a = v``, ``x.a += v``, ``x.a: T = v``,
    an ``x.a`` an unpacking or a ``for`` binds (*value* ``None``), and ``setattr(x, name, v)``: where Python calls
    ``type(x).__setattr__`` (:meth:`calls.Calls.stored`)."""
    for node in C.own_nodes(func):
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
            statement = getattr(node, "parent", None)
            value = (
                statement.value
                if isinstance(statement, (ast.Assign, ast.AugAssign, ast.AnnAssign)) and node in _targets(statement)
                else None
            )
            yield node, ast.Constant(value=node.attr), value
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "setattr"
            and len(node.args) == 3
            and not node.keywords
        ):
            yield node, node.args[1], node.args[2]


def _targets(statement: ast.Assign | ast.AugAssign | ast.AnnAssign) -> list[ast.expr]:
    return list(statement.targets) if isinstance(statement, ast.Assign) else [statement.target]


def _handed_values(call: ast.Call) -> Iterator[ast.expr]:
    for argument in [*call.args, *(keyword.value for keyword in call.keywords)]:
        value = argument.value if isinstance(argument, ast.Starred) else argument
        if isinstance(value, (ast.Name, ast.Attribute)):
            yield value


def _reads_self(value: ast.expr) -> bool:
    """Whether *value*, a value handed, is read off ``self``, ``cls`` or ``super()``."""
    if not isinstance(value, ast.Attribute):
        return False
    receiver = value.value
    return (isinstance(receiver, ast.Name) and receiver.id in ("self", "cls")) or (
        isinstance(receiver, ast.Call)
        and isinstance(receiver.func, ast.Name)
        and receiver.func.id == "super"
    )


class _Row:
    """One function's record while it is read: one row per way a call goes, with every line; holes deduplicated, assumptions per call."""

    def __init__(self) -> None:
        self.calls: dict[tuple, ReachCall] = {}
        self.database: set[int] = set()
        self.holes: dict[tuple, ReachHole] = {}
        self.settled: dict[tuple[int, int, str], None] = {}

    def edge(
        self,
        call: ast.AST,
        callee: ast.AST,
        to: str,
        how: str,
        for_classes: list[str],
        runs_as: list[str],
        thread: bool,
    ) -> None:
        key = (to, how, tuple(for_classes), tuple(runs_as), thread)
        held = self.calls.get(key)
        if held is None:
            self.calls[key] = ReachCall(
                line=call.lineno,
                lines=[call.lineno],
                callee=_cut(callee),
                to=to,
                how=how,
                for_classes=for_classes,
                runs_as=runs_as,
                thread=thread,
            )
        elif call.lineno not in held.lines:
            held.lines.append(call.lineno)
            held.lines.sort()

    def hole(self, call: ast.Call, why: str, for_classes: list[str]) -> None:
        key = (call.lineno, _cut(call.func), why, tuple(for_classes))
        if key not in self.holes:
            self.holes[key] = ReachHole(
                line=call.lineno,
                callee=_cut(call.func),
                why=why,
                for_classes=for_classes,
            )

    def settle(self, call: ast.Call, assumption: str) -> None:
        self.settled[(call.lineno, call.col_offset, assumption)] = None


def graph(
    calls: C.Calls,
    map_id: str,
    derived_from: str,
    leaves: Optional[Leaves] = None,
    shas: Optional[dict[str, str]] = None,
) -> list[ReachNode]:
    """One :class:`ReachNode` per function of the corpus *calls* reads.

    *leaves* is the plugin's reading of the database: for a call, already
    placed, ``("database", assumption)`` where the call is the database --
    the closure goes no further, and its statements are read where it runs
    -- or ``("hole", why)`` where it may be and cannot be read.  Without it
    no call is the database's.
    """
    types = calls.types
    names = _Names(types)
    receivers = _Receivers(calls, leaves)
    defined = {
        method.name
        for file in types.corpus_files()
        for node in walk(types.tree(file) or ast.Module(body=[], type_ignores=[]))
        if isinstance(node, ast.ClassDef)
        for method in pathcond.methods_of(node)
    }
    nodes: list[ReachNode] = []
    for file, func, owner in names.functions:
        owned = types.method_class(file, func)
        objects = calls.objects_of(owned) if owned is not None else []
        classes = [
            spelled for one in objects if (spelled := names.of(one[1])) is not None
        ]
        row = _Row()
        for call, how in _calls_in(func):
            try:
                placed = calls.callees(file, func, call)
            except Exception as error:  # noqa: BLE001 -- the call is a hole, not the build's end
                row.hole(call, f"{HOLE_FAILED}: {type(error).__name__}", [])
                continue
            leaf = leaves(file, func, call, placed) if leaves is not None else None
            if leaf is not None:
                kind, said = leaf
                if kind == "database":
                    row.database.add(call.lineno)
                    row.settle(call, said)
                else:
                    row.hole(call, said, [])
                continue
            for for_classes, bound in _groups(objects, classes, C._reads_self(call)):
                if bound:
                    try:
                        placed = calls.callees(file, func, call, bound)
                    except Exception as error:  # noqa: BLE001
                        row.hole(
                            call, f"{HOLE_FAILED}: {type(error).__name__}", for_classes
                        )
                        continue
                for target in placed.targets:
                    to = names.of(target.func)
                    if to is not None:
                        runs_as = [
                            spelled
                            for one in target.bound
                            if (spelled := names.of(one[1])) is not None
                        ]
                        row.edge(
                            call,
                            call.func,
                            to,
                            how,
                            for_classes,
                            runs_as,
                            placed.thread,
                        )
                _close(
                    row,
                    receivers,
                    calls,
                    defined,
                    file,
                    func,
                    call,
                    placed,
                    list(bound) or objects,
                    for_classes,
                )
            function = callfacts.builtin_function(receivers.types, file, func, call)
            for value in _handed_values(call):
                if function is not None and callfacts.builds_nothing(function, call, value):
                    continue  # isinstance(x, C), hasattr(C, name): no constructor run
                for for_classes, bound in _groups(objects, classes, _reads_self(value)):
                    for target in calls.named_callables(file, func, value, bound):
                        to = names.of(target.func)
                        if to is not None:
                            runs_as = [
                                spelled
                                for one in target.bound
                                if (spelled := names.of(one[1])) is not None
                            ]
                            row.edge(
                                value, value, to, "handed", for_classes, runs_as, False
                            )
        # An attribute store runs its object's ``__setattr__`` with no call
        # naming it: ``job.jobStatus = "closed"`` runs ``JobSpec.__setattr__``,
        # which sets ``stateChangeTime``.  Placed as the call Python makes, where
        # the object can be a class of the corpus that has one; a class taking
        # ``object``'s, or an object no reading says anything of, has no edge
        # and no hole (:data:`IMPLICIT_DUNDERS`).
        for store, name, value in _stores_in(func):
            implicit = calls.stored(store, name, value)
            for for_classes, bound in _groups(objects, classes, C._reads_self(implicit)):
                try:
                    placed = calls.callees(file, func, implicit, bound)
                except Exception:  # noqa: BLE001 -- no edge, as for an object nothing says
                    continue
                for target in placed.targets:
                    to = names.of(target.func)
                    if to is not None and target.func.name == "__setattr__":  # type: ignore[attr-defined]
                        runs_as = [
                            spelled for one in target.bound if (spelled := names.of(one[1])) is not None
                        ]
                        row.edge(implicit, store, to, HOW_SETATTR, for_classes, runs_as, False)
        settled = Counter(assumption for _line, _column, assumption in row.settled)
        nodes.append(
            ReachNode(
                name=ReachNode.make_name(map_id, owner),
                map_id=map_id,
                derived_from=derived_from,
                owner=owner,
                cls=(names.of(owned[1]) or "") if owned is not None else "",
                function=func.name,  # type: ignore[attr-defined]
                classes=classes,
                calls=sorted(
                    row.calls.values(),
                    key=lambda one: (one.line, one.to, one.how, one.for_classes),
                ),
                database=sorted(row.database),
                holes=sorted(
                    row.holes.values(),
                    key=lambda one: (one.line, one.callee, one.why, one.for_classes),
                ),
                settled=dict(sorted(settled.items())),
                anchor=_anchor(file, func, shas),
            )
        )
    return nodes


def _anchor(file: str, func: ast.AST, shas: Optional[dict[str, str]]) -> Anchor:
    return Anchor(
        package=file.split("/", 1)[0],
        file=file,
        line_start=func.lineno,  # type: ignore[attr-defined]
        line_end=getattr(func, "end_lineno", None),
        blob_sha=(shas or {}).get(file),
    )


def _groups(
    objects: list[tuple[str, ast.ClassDef]], classes: list[str], reads_self: bool
) -> Iterator[tuple[list[str], tuple[tuple[str, ast.ClassDef], ...]]]:
    """``(for_classes, bound)``: once for every class where nothing is read off ``self``, else once per class ``self`` can be."""
    if not reads_self or not objects or len(objects) != len(classes):
        yield [], ()
        return
    for one, spelled in zip(objects, classes, strict=True):
        yield [spelled], (one,)



def _given_back(calls: C.Calls, file: str, func: ast.AST, call: ast.Call) -> bool:
    """Whether the value *call* puts into a queue is one taken from that same queue (``queue_given_back``).

    Taken in the same function -- ``i = self.childlock.get()`` ... ``self.childlock.put(i)`` -- or handed in by
    every call of the function, each a value its caller took through a method of the same class that returns a get
    of the queue: ``DBProxyPool.get()`` takes ``proxy = self.getProxy()``, which returns ``self.proxyList.get()``,
    and gives it back with ``self.putProxy(proxy)``.  A queue the value came into some other way -- a process just
    started, a message read, a request -- may carry work to whoever takes it, and is a hole (:data:`HOLE_QUEUE`).
    """
    queue = ast.unparse(call.func.value)  # type: ignore[attr-defined]
    value = call.args[0] if call.args else next((one.value for one in call.keywords if one.arg == "item"), None)
    if not isinstance(value, ast.Name):
        return False
    if value.id not in _parameters(func):
        return _taken_from(func, value.id, queue)
    owned = calls.types.method_class(file, func)
    if owned is None or not queue.startswith("self."):
        return False
    target = C.Target(file, func, owned[1])
    try:
        sites = calls.callers(target)
    except Exception:  # noqa: BLE001 -- no caller read is no caller known
        return False
    return bool(sites) and all(_handed_back(calls, site, target, value.id, owned, queue) for site in sites)


def _handed_back(
    calls: C.Calls, site: C.CallSite, target: C.Target, parameter: str, owned: tuple[str, ast.ClassDef], queue: str
) -> bool:
    """Whether *site* hands *parameter* a value its function took through a method of *owned* returning a get of
    *queue*."""
    try:
        handed = calls.handed(site, target).get(parameter)
    except Exception:  # noqa: BLE001
        return False
    if not isinstance(handed, ast.Name) or handed.id in _parameters(site.func):
        return False
    found = [one for one in pathcond.bindings(site.func, handed.id) if not one.nested]
    if not found or any(not isinstance(one.value, ast.Call) or one.slot is not None for one in found):
        return False
    for one in found:
        try:
            placed = calls.callees(site.file, site.func, one.value)  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001
            return False
        if placed.unplaced or not placed.targets:
            return False
        for getter in placed.targets:
            if getter.cls is not owned[1] or not _returns_taken(getter.func, queue):
                return False
    return True


def _returns_taken(func: ast.AST, queue: str) -> bool:
    """Whether every ``return`` of *func* gives back a value taken from *queue*."""
    returns = [node for node in walk(func) if isinstance(node, ast.Return)]
    return bool(returns) and all(
        _is_get(one.value, queue) or (isinstance(one.value, ast.Name) and _taken_from(func, one.value.id, queue))
        for one in returns
    )


def _taken_from(func: ast.AST, name: str, queue: str) -> bool:
    """Whether every binding of *name* in *func* is a get of *queue*."""
    found = [one for one in pathcond.bindings(func, name) if not one.nested]
    return bool(found) and all(one.slot is None and _is_get(one.value, queue) for one in found)


def _is_get(value: Optional[ast.expr], queue: str) -> bool:
    return (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr in _GETS
        and ast.unparse(value.func.value) == queue
    )


def _close(
    row: _Row,
    receivers: _Receivers,
    calls: C.Calls,
    defined: set[str],
    file: str,
    func: ast.AST,
    call: ast.Call,
    placed: C.Placed,
    selfs: list[tuple[str, ast.ClassDef]],
    for_classes: list[str],
) -> None:
    """Close a call's ways on that no target says, or record the hole: see :data:`PREMISES` and :data:`ASSUMPTIONS`.

    *selfs* are the classes ``self`` is taken to be for this placement.
    """
    reason = placed.unplaced
    called = call.func
    if placed.targets:
        if reason:
            row.hole(call, reason, for_classes)
        return
    reads_receiver = reason == C.UNPLACED_RECEIVER or (
        reason == C.UNPLACED_OUTSIDE and getattr(called, "attr", "") in _PUTS
    )
    if reads_receiver and isinstance(called, ast.Attribute):
        kind, detail = receivers.of(file, func, called.value)
        if kind == _QUEUE and called.attr in _PUTS:
            if _given_back(calls, file, func, call):
                row.settle(call, "queue_given_back")
            else:
                row.hole(call, HOLE_QUEUE, for_classes)
            return
        if reason == C.UNPLACED_OUTSIDE or kind in (_BUILTIN, _QUEUE):
            return
        if kind == _OUTSIDE:
            row.settle(call, "outside_returns_outside")
            return
        if called.attr not in defined:
            row.settle(call, "no_runtime_methods")
            return
        row.hole(call, f"{reason}: {detail}", for_classes)
        return
    if reason in (C.UNPLACED_OUTSIDE, C.UNPLACED_NO_INIT):
        return
    if reason == C.UNPLACED_NAME:
        if isinstance(called, ast.Name) and receivers._is_builtin(
            file, func, called.id
        ):
            return
        row.hole(call, HOLE_LOCAL_CALLED, for_classes)
        return
    if reason == C.UNPLACED_NO_METHOD:
        asked = _asked(receivers.calls, file, func, call, selfs)
        types = receivers.types
        if (
            called.attr == callfacts.START
            and any(types.method(*one, callfacts.RUN) is not None for one in asked)
            and any(types.built_outside(*one) for one in asked)
        ):
            row.hole(call, HOLE_OUTSIDE_START, for_classes)
            return
        if _outside_base(types, asked):
            return
        row.settle(call, "no_runtime_methods")
        return
    row.hole(call, reason or "placed nowhere", for_classes)


def _asked(
    calls: C.Calls,
    file: str,
    func: ast.AST,
    call: ast.Call,
    selfs: list[tuple[str, ast.ClassDef]],
) -> list[tuple[str, ast.ClassDef]]:
    """The classes a call placed at no method was looked up on: ``self``'s, or what its receiver is declared as."""
    called = call.func
    receiver = called.value if isinstance(called, ast.Attribute) else None
    if isinstance(receiver, ast.Name) and receiver.id in ("self", "cls"):
        return selfs
    declared = (
        calls.types.declared(file, func, receiver) if receiver is not None else None
    )
    return [one for each in declared or () for one in calls.objects_of(each)]


def _outside_base(types: C.Types, classes: Iterable[tuple[str, ast.ClassDef]]) -> bool:
    """Whether every one of *classes* is built on a class outside the corpus, whose methods this reading does not see."""
    asked = list(classes)
    return bool(asked) and all(types.built_outside(*one) for one in asked)


class Destination:
    """The arms a question ends at, per :attr:`ReachNode.owner`, and the writes no row was found for."""

    def __init__(self, arms: dict[str, list[str]], unmatched: list[str]):
        self.arms = arms
        self.unmatched = unmatched


def _named_call(hole: str) -> Optional[str]:
    """The name *hole*'s call is made with -- ``mb_proxy.send`` gives ``send`` --, or ``None`` where no name can close
    it: a hole of :data:`_NOT_BY_NAME`, a called local or parameter, a callee cut short, no call (a write no function
    holds)."""
    found = _HOLE_TEXT.match(hole)
    if found is None or found.group("why").startswith(_NOT_BY_NAME):
        return None
    callee = found.group("callee")
    _head, dot, name = callee.rpartition(".")
    return name if dot and not callee.endswith("…") and name.isidentifier() else None


class _ByName:
    """Which call names can reach a write of one destination: the least fixpoint over names (MV 4, O1).

    A name can where a function of the corpus of that name reaches a write of the destination, leaves a hole no name
    closes, or makes a call whose name can.  A name no function of the corpus has lands in no corpus code
    (``no_runtime_methods``), and cannot."""

    def __init__(self, graph: "ReachGraph", destination: Destination):
        self.graph, self.destination = graph, destination
        self._read: dict[str, tuple[bool, set[Optional[str]]]] = {}
        self._can: set[str] = set()

    def can(self, name: str) -> bool:
        todo = [name]
        while todo:
            one = todo.pop()
            if one in self._read:
                continue
            owners = self.graph.functions_named(one)
            if not owners:
                self._read[one] = (False, set())
                continue
            answer = self.graph.reach(owners, self.destination, by_name=False)
            inner = {_named_call(hole) for hole in answer.holes}
            self._read[one] = (bool(answer.writes), inner)
            todo.extend(other for other in inner if other is not None and other not in self._read)
        changed = True
        while changed:
            changed = False
            for one, (writes, inner) in self._read.items():
                if one not in self._can and (writes or None in inner or inner & self._can):
                    self._can.add(one)
                    changed = True
        return name in self._can


class ReachGraph:
    """The record :func:`graph` built, read: which function holds a line, which a junction owner names, what a start reaches."""

    def __init__(self, rows: list[ReachNode]):
        self.rows = {row.owner: row for row in rows}
        self._by_file: dict[str, list[ReachNode]] = {}
        self._spelled: dict[str, list[tuple[str, int, str]]] = {}
        self._functions: dict[str, list[str]] = {}
        # Per destination, kept with it so its id is not reused (a destination is built per question).
        self._closers: dict[int, tuple[Destination, _ByName]] = {}
        for row in rows:
            self._functions.setdefault(row.function, []).append(row.owner)
            if row.anchor is not None:
                self._by_file.setdefault(row.anchor.file, []).append(row)
            for hole in row.holes:
                _head, dot, name = hole.callee.rpartition(".")
                if dot:
                    self._spelled.setdefault(name, []).append((row.owner, hole.line, hole.callee))

    def function_at(self, file: str, line: int) -> Optional[str]:
        """The innermost function of *file* holding *line*."""
        best: Optional[ReachNode] = None
        for row in self._by_file.get(file, ()):
            anchor = row.anchor
            if anchor is not None and anchor.line_start <= line <= (
                anchor.line_end or anchor.line_start
            ):
                if best is None or anchor.line_start > best.anchor.line_start:  # type: ignore[union-attr]
                    best = row
        return best.owner if best is not None else None

    def hole_at(self, file: str, call: ast.Call) -> Optional[tuple[str, int, str]]:
        """``(owner, line, callee)`` where the record holds *call* as a hole (:class:`ReachHole`), else ``None``."""
        owner = self.function_at(file, call.lineno)
        row = self.rows.get(owner) if owner is not None else None
        if row is None:
            return None
        callee = _cut(call.func)
        if any(hole.line == call.lineno and hole.callee == callee for hole in row.holes):
            return owner, call.lineno, callee  # type: ignore[return-value]
        return None

    def holes_spelled(self, name: str) -> list[tuple[str, int, str]]:
        """``(owner, line, callee)`` of each hole spelled ``x.<name>(...)``: a call no reading placed that, if it lands
        in the corpus at all, lands at a definition named *name*."""
        return self._spelled.get(name, [])

    def is_database(self, file: str, call: ast.Call) -> bool:
        """Whether the record reads the call on *call*'s line as PanDA's database (:attr:`ReachNode.database`)."""
        owner = self.function_at(file, call.lineno)
        row = self.rows.get(owner) if owner is not None else None
        return row is not None and call.lineno in row.database

    def named(self, owner: str) -> list[str]:
        """Every function a junction owner's ``file::name`` can mean: each definition of that name in the file."""
        file, _, name = owner.partition("::")
        bare = name.rsplit(".", 1)[-1]
        return [
            row.owner for row in self._by_file.get(file, ()) if row.function == bare
        ]

    def functions_named(self, name: str) -> list[str]:
        """The owner of every function of the record named *name*: where a call of that name can land."""
        return self._functions.get(name, [])

    def closure(
        self, starts: list[str], cut: Optional[Callable[[tuple[str, str], ReachCall], bool]] = None
    ) -> Came:
        """Every function *starts* reach through their calls, each as every class ``self`` can be there, and how.

        Each start as each of its :attr:`ReachNode.classes` (or none); from a function as a class, every call it
        makes for that class (:attr:`ReachCall.for_classes`), into the callee as the classes it runs as
        (:attr:`ReachCall.runs_as`, else each it can be).  Breadth first, so each is reached by the fewest calls;
        every start must be a row of the record.  A call *cut* says of the function and class making it is not
        followed."""
        start_nodes = [
            (owner, one) for owner in starts for one in self.rows[owner].classes or [""]
        ]
        came: Came = dict.fromkeys(start_nodes)
        pending = deque(start_nodes)
        while pending:
            node = pending.popleft()
            owner, cls = node
            for call in self.rows[owner].calls:
                if call.for_classes and cls not in call.for_classes:
                    continue
                if cut is not None and cut(node, call):
                    continue
                entered = self.rows.get(call.to)
                if entered is None:
                    continue
                for one in call.runs_as or entered.classes or [""]:
                    following = (call.to, one)
                    if following not in came:
                        came[following] = (node, call)
                        pending.append(following)
        return came

    def reach(self, starts: list[str], destination: Destination, by_name: bool = True) -> Reach:
        """What *starts* reach: the writes, or whether none can be said (see :class:`Reach`).

        Where no write is reached, a hole whose call's name no function of the corpus reaching the destination has
        (:class:`_ByName`) is closed by ``closed_by_name`` and listed in :attr:`Reach.closed`; *by_name* off, none
        is."""
        for owner in starts:
            if owner not in self.rows:
                return Reach(
                    writes=None, holes=[f"{owner} -- no such function in the record"]
                )
        came = self.closure(starts)
        owners = list(dict.fromkeys(owner for owner, _cls in came))
        holes: list[str] = []
        for owner, cls in came:
            for hole in self.rows[owner].holes:
                if not hole.for_classes or cls in hole.for_classes:
                    holes.append(f"{owner}:{hole.line} {hole.callee} -- {hole.why}")
        holes = list(dict.fromkeys(holes))
        holes.extend(
            f"{owner} -- a write the record has no function for"
            for owner in destination.unmatched
        )
        assumptions: Counter = Counter()
        for owner in owners:
            assumptions.update(self.rows[owner].settled)
        closed: list[str] = []
        writes = []
        for node in came:
            owner = node[0]
            if owner in destination.arms and all(
                write.function != owner for write in writes
            ):
                writes.append(
                    ReachWrite(
                        function=owner,
                        arms=destination.arms[owner],
                        chain=way(came, node),
                    )
                )
        if by_name and not writes and holes:
            held = self._closers.get(id(destination))
            if held is None or held[0] is not destination:
                held = self._closers[id(destination)] = (destination, _ByName(self, destination))
            names = held[1]
            kept: list[str] = []
            for hole in holes:
                called = _named_call(hole)
                (closed if called is not None and not names.can(called) else kept).append(hole)
            holes = kept
            if closed:
                assumptions["closed_by_name"] += len(closed)
        return Reach(
            writes=writes if writes or not holes else None,
            holes=holes,
            closed=closed,
            assumptions=dict(sorted(assumptions.items())),
            closure=len(owners),
        )


def way(came: Came, node: tuple[str, str]) -> list[str]:
    """``file:line`` of each call :meth:`ReachGraph.closure` took from a start to *node*, the first call first."""
    chain: list[str] = []
    at = came[node]
    while at is not None:
        caller, call = at
        chain.append(f"{caller[0].partition('::')[0]}:{call.line}")
        at = came[caller]
    return list(reversed(chain))


def every_write(junctions: list[JunctionNode], record: ReachGraph) -> Destination:
    """Every function of the record that holds an arm of any junction: where any write is."""
    arms: dict[str, list[str]] = {}
    unmatched: list[str] = []
    for junction in junctions:
        if not junction.branches:
            continue
        owner = _junction_function(junction, record)
        if owner is None:
            unmatched.append(junction.owner)
            continue
        arms.setdefault(owner, []).extend(
            f"{branch.outcome}@{branch.line}" for branch in junction.branches
        )
    return Destination(arms, sorted(set(unmatched)))


def moving_off(
    junctions: list[JunctionNode], record: ReachGraph, subject: str, value: str
) -> Destination:
    """The arms that can move *subject* off *value*, per function of the record.

    An arm writing another value whose row precondition does not require
    the row to hold something else (:func:`models.required_values`).  An arm
    whose outcome is decided at run time, or passed through, writes another
    value as far as this can say, and is kept.  An arm whose object the
    records prove is written to no row (:attr:`models.Branch.off_row`) moves
    nothing.
    """
    column = subject.rsplit(".", 1)[-1].lower()
    arms: dict[str, list[str]] = {}
    unmatched: list[str] = []
    for junction in junctions:
        if junction.subject != subject:
            continue
        kept = []
        for branch in junction.branches:
            if branch.outcome == value or branch.off_row is not None:
                continue
            if any(
                (required := required_values(term)) is not None
                and required[0] == column
                and value not in required[1]
                for term in branch.row_precondition
            ):
                continue
            kept.append(f"{branch.outcome}@{branch.line}")
        if not kept:
            continue
        owner = _junction_function(junction, record)
        if owner is None:
            unmatched.append(junction.owner)
            continue
        arms.setdefault(owner, []).extend(kept)
    return Destination(arms, sorted(set(unmatched)))


def _junction_function(junction: JunctionNode, record: ReachGraph) -> Optional[str]:
    """The function of the record a junction is in: named by its owner, holding its anchor."""
    named = record.named(junction.owner)
    if junction.anchor is None:
        return named[0] if len(named) == 1 else None
    holding = record.function_at(junction.anchor.file, junction.anchor.line_start)
    return holding if holding in named else None
