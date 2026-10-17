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
from collections import Counter, deque
from typing import Callable, Iterable, Iterator, Optional

from bamboo.codemap import calls as C
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
        "dunder (``with``, ``for``, an operator) -- so a call on a builtin or "
        "outside object goes on into nothing of the corpus."
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
    "driver_talks_only_to_db": "A call on the database driver -- a cursor, a connection -- goes only to the database.",
}

#: A database leaf or a hole for a call, as a build's plugin reads it:
#: ``("database", assumption)`` or ``("hole", why)``.  ``None`` where the call
#: is not the database's.  See :func:`graph`.
Leaves = Callable[[str, ast.AST, ast.Call, C.Placed], Optional[tuple[str, str]]]

HOLE_LOCAL_CALLED = "a local or a parameter is called"
HOLE_QUEUE = "hands a value to another thread or process through a queue"
HOLE_FAILED = "placing it failed"
HOLE_OUTSIDE_START = "start() of a class built outside the corpus may run its run"

#: The library classes whose ``put()`` hands a value to whoever ``get()``s it
#: -- another thread, another process, another task.
_QUEUES = {
    ("queue", "Queue"),
    ("queue", "LifoQueue"),
    ("queue", "PriorityQueue"),
    ("queue", "SimpleQueue"),
    ("multiprocessing", "Queue"),
    ("multiprocessing", "JoinableQueue"),
    ("multiprocessing", "SimpleQueue"),
    ("asyncio", "Queue"),
    ("asyncio", "LifoQueue"),
    ("asyncio", "PriorityQueue"),
}
_PUTS = {"put", "put_nowait"}
_GETS = {"get", "get_nowait"}
#: Builtins that never call what they are handed: a class passed to them is no constructor run.
_NOT_CALLING = {
    "isinstance", "issubclass", "super", "getattr", "hasattr", "delattr", "type", "id",
    "len", "repr", "str", "print", "callable", "vars", "dir",
}  # fmt: skip

_BUILTINS = set(dir(builtins))
#: The builtin types and the typing and collections names a builtin value is annotated with.
_BUILTIN_ANNOTATIONS = {
    "str", "int", "float", "bool", "bytes", "bytearray", "complex", "dict", "list", "set", "tuple",
    "frozenset", "range", "None", "NoneType", "Dict", "List", "Set", "Tuple", "FrozenSet", "Iterable",
    "Iterator", "Sequence", "Mapping", "MutableMapping", "Generator", "defaultdict", "OrderedDict",
    "Counter", "deque", "DefaultDict",
}  # fmt: skip
_SCALARS = {"str", "int", "float", "bool", "bytes", "complex", "NoneType"}
#: Builtin functions whose value is a builtin whatever they are handed, and of which type.
_BUILTIN_RESULTS = {
    "len": "int", "str": "str", "int": "int", "float": "float", "bool": "bool", "bytes": "bytes",
    "repr": "str", "hash": "int", "id": "int", "abs": "", "round": "", "ord": "int", "chr": "str",
    "hex": "str", "oct": "str", "bin": "str", "format": "str", "isinstance": "bool", "issubclass": "bool",
    "hasattr": "bool", "callable": "bool", "any": "bool", "all": "bool", "divmod": "tuple", "range": "range",
    "list": "list", "dict": "dict", "set": "set", "tuple": "tuple", "frozenset": "frozenset",
    "sorted": "list", "enumerate": "iterator", "zip": "iterator", "map": "iterator", "filter": "iterator",
    "reversed": "iterator", "iter": "iterator", "open": "file", "input": "str", "print": "NoneType",
    "sum": "", "pow": "",
}  # fmt: skip
#: Methods of a builtin container whose value is one of what it holds.
_ELEMENT_METHODS = {
    "get",
    "pop",
    "popitem",
    "setdefault",
    "values",
    "items",
    "copy",
    "popleft",
    "__getitem__",
}
#: The dunders each operator calls: what ``a * b`` is, where a class of the corpus defines one.
_OPERATORS: dict[type, tuple[str, ...]] = {
    op: (f"__{name}__", f"__r{name}__", f"__i{name}__")
    for op, name in (
        (ast.Add, "add"), (ast.Sub, "sub"), (ast.Mult, "mul"), (ast.MatMult, "matmul"), (ast.Div, "truediv"),
        (ast.FloorDiv, "floordiv"), (ast.Mod, "mod"), (ast.Pow, "pow"), (ast.LShift, "lshift"),
        (ast.RShift, "rshift"), (ast.BitAnd, "and"), (ast.BitXor, "xor"), (ast.BitOr, "or"),
    )
} | {ast.USub: ("__neg__",), ast.UAdd: ("__pos__",), ast.Invert: ("__invert__",)}  # fmt: skip

#: The dunders Python runs with no call naming them -- ``with``, ``for``,
#: ``in``, ``x[k]``, ``str(x)``, an operator, comparing, hashing, pickling
#: and copying.  No edge says when they run, so the premise
#: ``outside_calls_back_only_handed`` takes them to reach no write, and the
#: build checks it (``implicit-dunders-reach-no-write``).
IMPLICIT_DUNDERS = frozenset(
    {
        "__enter__", "__exit__", "__aenter__", "__aexit__", "__iter__", "__next__", "__aiter__", "__anext__",
        "__getitem__", "__setitem__", "__delitem__", "__contains__", "__len__", "__bool__", "__eq__", "__ne__",
        "__lt__", "__le__", "__gt__", "__ge__", "__hash__", "__str__", "__repr__", "__format__", "__del__",
        "__call__", "__getattr__", "__getattribute__", "__setattr__", "__delattr__", "__reduce__",
        "__reduce_ex__", "__getstate__", "__setstate__", "__copy__", "__deepcopy__", "__new__", "__post_init__",
        "__await__", "__index__", "__int__", "__float__", "__init_subclass__",
    }
    | {name for names in _OPERATORS.values() for name in names}
)  # fmt: skip

_BUILTIN, _OUTSIDE, _QUEUE, _UNKNOWN = "builtin", "outside", "queue", "unknown"
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
    """The id of every function and class of the corpus: ``file::Outer.inner``, ``#n`` after a repeated one."""

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
        for node in ast.iter_child_nodes(scope):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                spelled = f"{file}::{'.'.join([*names, node.name])}"
                seen[spelled] += 1
                if seen[spelled] > 1:
                    spelled = f"{spelled}#{seen[spelled]}"
                self._of[id(node)] = spelled
                self._keep.append(node)
                if not isinstance(node, ast.ClassDef):
                    self.functions.append((file, node, spelled))
                self._walk(file, node, [*names, node.name], seen)
            else:
                self._walk(file, node, names, seen)

    def of(self, node: ast.AST) -> Optional[str]:
        return self._of.get(id(node))


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
    is declared to return, what a name or a field is bound to.

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
            library is None or library[0] in ("typing", "collections")
        ):
            return _BUILTIN, {
                "Dict": "dict",
                "List": "list",
                "Set": "set",
                "Tuple": "tuple",
            }.get(named, named)
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
            return _element(self.of(file, func, expression.value, depth + 1))
        if isinstance(expression, ast.Call):
            return self._called(file, func, expression, depth)
        if isinstance(expression, ast.Name):
            return self._named(file, func, expression.id, depth)
        if isinstance(expression, ast.Attribute):
            return self._attribute(file, func, expression, depth)
        return _UNKNOWN, f"a {type(expression).__name__}"

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
                for node in ast.walk(
                    self.types.tree(file) or ast.Module(body=[], type_ignores=[])
                )
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name in names
            ]
            self._operators[op] = _joined(found) if found else None
        return self._operators[op]

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
        if placed.thread:
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
            result = _BUILTIN_RESULTS.get(called.id)
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
        """What a builtin's or a queue's method returns: one of what it holds, or a builtin."""
        kind, held = receiver
        if kind == _QUEUE:
            return (
                (_UNKNOWN, f"what a {held} holds")
                if method in _GETS
                else (_BUILTIN, "")
            )
        if held in _SCALARS:
            return _BUILTIN, ""
        if method in _ELEMENT_METHODS:
            return _UNKNOWN, f"what a {held or 'builtin container'} holds"
        return _BUILTIN, ""

    def _is_builtin(self, file: str, func: ast.AST, name: str) -> bool:
        if name not in _BUILTINS:
            return False
        if not isinstance(func, ast.Module) and (
            name in _parameters(func) or name in self.types.assigned(func)
        ):
            return False
        return name not in self.types._module_bound(file)

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
        for value in self.types.assigned(func).get(name, ()):
            kinds.append(self.of(file, func, value, depth + 1))
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
    if kind == _BUILTIN and detail in ("str", "bytes", "range"):
        return _BUILTIN, "str" if detail == "str" else ""
    if kind == _UNKNOWN:
        return container
    return _UNKNOWN, f"an element of a {detail or kind}"


def _entered(manager: tuple[str, str]) -> tuple[str, str]:
    """What ``with <manager> as x`` binds: a builtin's or an outside object's own ``__enter__`` result."""
    if manager == _NOTHING:
        return _NOTHING
    return (
        manager
        if manager[0] in (_BUILTIN, _OUTSIDE)
        else (_UNKNOWN, f"what entering {manager[1]} yields")
    )


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
    for node in ast.walk(func):
        if isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
            if any(
                isinstance(leaf, ast.Name) and leaf.id == name
                for leaf in ast.walk(node.target)
            ):
                found.append((node.iter, False))
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None and any(
                    isinstance(leaf, ast.Name) and leaf.id == name
                    for leaf in ast.walk(item.optional_vars)
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
        for node in ast.walk(types.tree(file) or ast.Module(body=[], type_ignores=[]))
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
                    defined,
                    file,
                    func,
                    call,
                    placed,
                    list(bound) or objects,
                    for_classes,
                )
            if (
                isinstance(call.func, ast.Name)
                and call.func.id in _NOT_CALLING
                and receivers._is_builtin(file, func, call.func.id)
            ):
                continue
            for value in _handed_values(call):
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


def _close(
    row: _Row,
    receivers: _Receivers,
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
            called.attr == "start"
            and any(types.method(*one, "run") is not None for one in asked)
            and any(_built_outside(types, one) for one in asked)
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
    return bool(asked) and all(_built_outside(types, one) for one in asked)


def _built_outside(types: C.Types, one: tuple[str, ast.ClassDef]) -> bool:
    """Whether a class in *one*'s order has a base outside the corpus that may supply methods.

    A base is read as :meth:`calls.Types.lineage` reads it, a subscript
    taken off -- ``FactoryBase[JobBrokerBase]`` is FactoryBase.
    """
    for where, holder in types.mro(*one):
        for base in holder.bases:
            head = pathcond.unsubscripted(base)
            if _supplies_nothing(head) or types.class_named(where, head) is not None:
                continue
            return True
    return False


def _supplies_nothing(base: ast.expr) -> bool:
    """``object`` and the typing bases a class is declared generic or a protocol with: no method comes from them."""
    named = base.attr if isinstance(base, ast.Attribute) else getattr(base, "id", "")
    return named in ("object", "Generic", "Protocol")


class Destination:
    """The arms a question ends at, per :attr:`ReachNode.owner`, and the writes no row was found for."""

    def __init__(self, arms: dict[str, list[str]], unmatched: list[str]):
        self.arms = arms
        self.unmatched = unmatched


class ReachGraph:
    """The record :func:`graph` built, read: which function holds a line, which a junction owner names, what a start reaches."""

    def __init__(self, rows: list[ReachNode]):
        self.rows = {row.owner: row for row in rows}
        self._by_file: dict[str, list[ReachNode]] = {}
        for row in rows:
            if row.anchor is not None:
                self._by_file.setdefault(row.anchor.file, []).append(row)

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

    def named(self, owner: str) -> list[str]:
        """Every function a junction owner's ``file::name`` can mean: each definition of that name in the file."""
        file, _, name = owner.partition("::")
        bare = name.rsplit(".", 1)[-1]
        return [
            row.owner for row in self._by_file.get(file, ()) if row.function == bare
        ]

    def reach(self, starts: list[str], destination: Destination) -> Reach:
        """What *starts* reach: the writes, or whether none can be said (see :class:`Reach`)."""
        start_nodes = []
        for owner in starts:
            row = self.rows.get(owner)
            if row is None:
                return Reach(
                    writes=None, holes=[f"{owner} -- no such function in the record"]
                )
            start_nodes.extend((owner, one) for one in row.classes or [""])
        came: dict[tuple[str, str], Optional[tuple[tuple[str, str], str]]] = (
            dict.fromkeys(start_nodes)
        )
        pending = deque(start_nodes)
        while pending:
            node = pending.popleft()
            owner, cls = node
            file = owner.partition("::")[0]
            for call in self.rows[owner].calls:
                if call.for_classes and cls not in call.for_classes:
                    continue
                entered = self.rows.get(call.to)
                if entered is None:
                    continue
                for one in call.runs_as or entered.classes or [""]:
                    following = (call.to, one)
                    if following not in came:
                        came[following] = (node, f"{file}:{call.line}")
                        pending.append(following)
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
        writes = []
        for node in came:
            owner = node[0]
            if owner in destination.arms and all(
                write.function != owner for write in writes
            ):
                chain: list[str] = []
                at: Optional[tuple[str, str]] = node
                while at is not None and came[at] is not None:
                    at, line = came[at]  # type: ignore[misc]
                    chain.append(line)
                writes.append(
                    ReachWrite(
                        function=owner,
                        arms=destination.arms[owner],
                        chain=list(reversed(chain)),
                    )
                )
        return Reach(
            writes=writes if writes or not holes else None,
            holes=holes,
            assumptions=dict(sorted(assumptions.items())),
            closure=len(owners),
        )


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
    value as far as this can say, and is kept.
    """
    column = subject.rsplit(".", 1)[-1].lower()
    arms: dict[str, list[str]] = {}
    unmatched: list[str] = []
    for junction in junctions:
        if junction.subject != subject:
            continue
        kept = []
        for branch in junction.branches:
            if branch.outcome == value:
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
