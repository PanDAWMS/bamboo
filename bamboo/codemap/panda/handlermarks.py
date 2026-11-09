"""What each ``except`` handler prints when it is entered, and the handlers a run entered by the marks it shows.

A reader of a run hands the map's flow and walker *raised*, the handlers the run entered
(:func:`~bamboo.codemap.panda.reaching.catching_try`): a statement raises only where its raise reaches one of
them, and a ``try`` whose handlers the run shows no mark of ran its body to the end.  It knows them by their
marks (:class:`~bamboo.codemap.models.HandlerMark`): a statement of the handler that prints, a row of the
owner's :class:`~bamboo.codemap.models.FunctionPrintsNode`, or a call each of whose placed definitions passes
one of its own prints on every way -- ``self.dump_error_message(tmp_log)`` in a DB proxy's method.

*Every way* is read by :func:`~bamboo.codemap.panda.reaching.reaching_each`, from the handler's entry (or the
callee's) to its exits: a ``return`` or ``raise`` of its own, a ``break`` or ``continue`` of no loop inside it,
and the end of its body.  The end is a statement a reader can reach only if one is put there, so the function
is read on a copy from a parse of its own, a ``pass`` after the last statement of each handler; the build's
tree is shared and stays as it is.  A raise the map's rule reads inside the handler leads to no exit counted
here.

A handler that can reach a ``raise`` of its own without passing a mark (:attr:`HandlerMark.reraises`) passes a
raise on without a word: ``except Exception as e: if self.is_no_wait_exception(e): tmpLog.debug(...) else:
raise``, inside a ``try`` whose handler prints.  A run that shows the outer mark may have entered it, so
:func:`entered` adds it.

A reader of one key's run can read only the owner's own prints absent, not a call's callee's: whether every
way passes one of them alone is asked again where a call is among the marks
(:attr:`HandlerMark.prints_on_every_way`).
"""

from __future__ import annotations

import ast
from collections.abc import Collection, Iterator, Sequence
from typing import Optional, TypeVar

from bamboo.codemap import calls as C
from bamboo.codemap.models import (
    HANDLER_CONDITIONAL,
    HANDLER_MARKED,
    HANDLER_SILENT,
    HANDLER_UNPLACED,
    HandlerMark,
    MapFragment,
    MarkCall,
    MarkCallee,
    SourceModule,
)
from bamboo.codemap.panda import pathcond
from bamboo.codemap.panda.reaching import catches_all, catching_try, reaching_each
from bamboo.codemap.panda.walker import is_log_call

_FUNCS = (ast.FunctionDef, ast.AsyncFunctionDef)
_NESTED = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
_LOOPS = (ast.For, ast.AsyncFor, ast.While)
#: Placed outside the corpus: a library's, or a class whose ``__init__`` is a library's (``changes._OUTSIDE``).
_OUTSIDE = frozenset({C.UNPLACED_OUTSIDE, C.UNPLACED_NO_INIT})
#: The builtins that write what a run shows: a handler calling one may print with no mark the map knows.
_WRITES = frozenset({"print"})

_Node = TypeVar("_Node", bound=ast.AST)


def _copy(node: _Node) -> _Node:
    """*node* and the nodes under it, new, with their fields and positions only.

    Not ``copy.deepcopy``, which follows every attribute: the context objects (``Load``, ``Store``, ``Del``)
    are one of each in the interpreter, shared by every tree parsed, and a tree given parents leaves one of
    its nodes on them.  A deep copy of a function from a parse of its own walked from them into the build's
    tree, and stopped at a value there that cannot be copied.  They are shared here too."""
    if isinstance(node, ast.expr_context):
        return node
    fields = {}
    for name, value in ast.iter_fields(node):
        if isinstance(value, list):
            value = [_copy(one) if isinstance(one, ast.AST) else one for one in value]
        elif isinstance(value, ast.AST):
            value = _copy(value)
        fields[name] = value
    for name in node._attributes:
        if hasattr(node, name):
            fields[name] = getattr(node, name)
    return type(node)(**fields)


def attach(fragment: MapFragment, modules: list[SourceModule], calls: C.Calls) -> int:
    """Set :attr:`MapFragment.handler_marks`, the prints read off *fragment*'s function prints; returns how many
    handlers are marked."""
    prints = {
        node.owner: {statement.line for statement in node.statements}
        for node in fragment.function_prints
    }
    fragment.handler_marks = handler_marks(modules, calls, prints)
    return sum(one.kind == HANDLER_MARKED for one in fragment.handler_marks)


def handler_marks(
    modules: list[SourceModule], calls: C.Calls, prints: dict[str, set[int]]
) -> list[HandlerMark]:
    """A record for every ``except`` handler in a function of *modules*, by owner and line.  *prints*: each
    function's print statements' lines, by owner."""
    reader = _Reader(modules, calls, prints)
    found: list[HandlerMark] = []
    for module in modules:
        for func in ast.walk(module.tree):
            if isinstance(func, _FUNCS):
                found.extend(reader.marks(module.rel_path, func))
    return sorted(found, key=lambda one: (one.owner, one.line))


def entered(
    func: ast.AST, marks: Sequence[HandlerMark], seen: Collection[int]
) -> list[ast.ExceptHandler]:
    """The handlers of *func* a run entered, as a reader hands them to the flow and the walker (*raised*).

    *seen*: the lines of the handlers whose marks the run shows.  *marks*: the records of *func*'s handlers.
    To those seen, each handler that passes a raise on without a mark (:attr:`HandlerMark.reraises`) is added
    where a ``raise`` of its own reaches one of them (:func:`~bamboo.codemap.panda.reaching.catching_try`),
    until none is added.
    """
    handlers = {
        node.lineno: node for node in _own(func) if isinstance(node, ast.ExceptHandler)
    }
    out = {line: handlers[line] for line in seen if line in handlers}
    passing = [
        handlers[one.line] for one in marks if one.reraises and one.line in handlers
    ]
    grown = True
    while grown:
        grown = False
        for handler in passing:
            if handler.lineno in out:
                continue
            raises = [node for node in _own(handler) if isinstance(node, ast.Raise)]
            if any(catching_try(one, list(out.values())) is not None for one in raises):
                out[handler.lineno] = handler
                grown = True
    return sorted(out.values(), key=lambda one: one.lineno)


def _own(node: ast.AST) -> Iterator[ast.AST]:
    """The nodes under *node*, a nested definition's body left out."""
    stack = list(ast.iter_child_nodes(node))
    while stack:
        one = stack.pop()
        yield one
        if not isinstance(one, _NESTED):
            stack.extend(ast.iter_child_nodes(one))


def _key(node: ast.AST) -> tuple[str, int, int]:
    return (
        type(node).__name__,
        getattr(node, "lineno", 0),
        getattr(node, "col_offset", 0),
    )


def _end(block: list[ast.stmt]) -> ast.Pass:
    """A ``pass`` after the last statement of *block*: the statement a way off its end reaches."""
    last = block[-1].end_lineno or block[-1].lineno
    return ast.Pass(lineno=last, col_offset=0, end_lineno=last, end_col_offset=0)


def _is_print(statement: ast.AST, lines: set[int]) -> bool:
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and (statement.lineno in lines or statement.value.lineno in lines)
    )


def _statement_call(statement: ast.AST) -> Optional[ast.Call]:
    """The call a statement makes whole -- ``f()``, ``x = f()``, ``return f()`` -- or None."""
    if isinstance(
        statement, (ast.Expr, ast.Assign, ast.AnnAssign, ast.Return)
    ) and isinstance(statement.value, ast.Call):
        return statement.value
    return None


def _every_way(
    body_owner: ast.AST, entry: ast.AST, marks: Sequence[ast.AST], end: ast.stmt
) -> bool:
    """Whether every way from *entry* to an exit of *body_owner* -- a ``return`` or ``raise`` of its own, a
    ``break`` or ``continue`` of no loop inside it, *end* -- passes one of *marks*."""
    exits: list[ast.AST] = [
        node for node in _own(body_owner) if isinstance(node, (ast.Return, ast.Raise))
    ]
    if isinstance(body_owner, ast.ExceptHandler):
        for node in _own(body_owner):
            if isinstance(node, (ast.Break, ast.Continue)):
                here = getattr(node, "parent", None)
                while (
                    here is not None
                    and here is not body_owner
                    and not isinstance(here, _LOOPS)
                ):
                    here = getattr(here, "parent", None)
                if here is body_owner:
                    exits.append(node)
    exits.append(end)
    return not any(reached for reached in reaching_each(exits, [entry], marks))


class _Reader:
    def __init__(
        self, modules: list[SourceModule], calls: C.Calls, prints: dict[str, set[int]]
    ) -> None:
        self.calls = calls
        self.prints = prints
        self.sources = {module.rel_path: module.source for module in modules}
        self._fresh: dict[
            str, dict[tuple[int, int], ast.FunctionDef | ast.AsyncFunctionDef]
        ] = {}
        self._callees: dict[str, Optional[MarkCallee]] = {}

    def fresh(self, file: str, func: ast.AST) -> ast.FunctionDef | ast.AsyncFunctionDef:
        """*func* of *file* from a parse of its own, copied (:func:`_copy`): nothing in it links into the build's
        tree."""
        if file not in self._fresh:
            tree = ast.parse(self.sources[file])
            self._fresh[file] = {
                (n.lineno, n.col_offset): n
                for n in ast.walk(tree)
                if isinstance(n, _FUNCS)
            }
        at = (getattr(func, "lineno", 0), getattr(func, "col_offset", 0))
        return _copy(self._fresh[file][at])

    def may_print(self, file: str, func: ast.AST, call: ast.Call) -> bool:
        """Whether *call*, no mark, may print all the same: a builtin that writes, or a call placed nowhere -- not
        in the corpus, and neither a builtin nor a library's."""
        called = call.func
        if isinstance(called, ast.Name) and self.calls.types.is_builtin(
            file, func, called.id
        ):
            return called.id in _WRITES
        placed = self.calls.callees(file, func, call)
        return not placed.targets and placed.unplaced not in _OUTSIDE

    def callee(self, target: C.Target) -> Optional[MarkCallee]:
        """*target* with its prints where every way through it passes one of them; None where one does not."""
        owner = f"{target.file}::{pathcond.qualified_name(target.func)}"
        if owner not in self._callees:
            self._callees[owner] = None
            lines = self.prints.get(owner, set())
            if lines and target.file in self.sources:
                func = self.fresh(target.file, target.func)
                end = _end(func.body)
                func.body.append(end)
                pathcond.attach_parents(func)
                marks = [
                    node
                    for node in _own(func)
                    if isinstance(node, ast.Expr) and _is_print(node, lines)
                ]
                if marks and _every_way(func, func.args, marks, end):
                    self._callees[owner] = MarkCallee(
                        owner=owner, prints=sorted({m.lineno for m in marks})
                    )
        return self._callees[owner]

    def marks(self, file: str, func: ast.AST) -> list[HandlerMark]:
        """A record for each handler in *func*'s own body."""
        handlers = [node for node in _own(func) if isinstance(node, ast.ExceptHandler)]
        if not handlers:
            return []
        owner = f"{file}::{pathcond.qualified_name(func)}"
        lines = self.prints.get(owner, set())
        read: list[
            tuple[
                ast.ExceptHandler, list[ast.stmt], list[MarkCall], list[ast.stmt], bool
            ]
        ] = []
        for handler in handlers:
            own_prints: list[ast.stmt] = []
            marking: list[ast.stmt] = []
            calls: list[MarkCall] = []
            unplaced = False
            printed = set()
            for node in _own(handler):
                if isinstance(node, ast.Expr) and _is_print(node, lines):
                    own_prints.append(node)
                    printed.add(id(node.value))
            for node in _own(handler):
                call = _statement_call(node)
                if (
                    isinstance(node, ast.stmt)
                    and call is not None
                    and id(call) not in printed
                ):
                    placed = self.calls.callees(file, func, call)
                    callees = [self.callee(target) for target in placed.targets]
                    if callees and all(one is not None for one in callees):
                        marking.append(node)
                        held = {one.owner: one for one in callees if one is not None}
                        calls.append(
                            MarkCall(
                                line=node.lineno,
                                callees=[held[k] for k in sorted(held)],
                            )
                        )
                if (
                    isinstance(node, ast.Call)
                    and id(node) not in printed
                    and not is_log_call(node)
                ):
                    unplaced = unplaced or self.may_print(file, func, node)
            read.append(
                (
                    handler,
                    own_prints,
                    sorted(calls, key=lambda one: one.line),
                    marking,
                    unplaced,
                )
            )
        copied = self.fresh(file, func)
        ends: dict[tuple[str, int, int], tuple[ast.ExceptHandler, ast.Pass]] = {}
        for node in list(_own(copied)):
            if isinstance(node, ast.ExceptHandler):
                end = _end(node.body)
                ends[_key(node)] = (node, end)
                node.body.append(end)
        pathcond.attach_parents(copied)
        by_key = {
            _key(node): node for node in _own(copied) if isinstance(node, ast.stmt)
        }
        out = []
        for handler, own_prints, calls, marking, unplaced in read:
            mine, end = ends[_key(handler)]
            marks = [by_key[_key(node)] for node in own_prints + marking]
            if marks:
                kind = (
                    HANDLER_MARKED
                    if _every_way(mine, mine, marks, end)
                    else HANDLER_CONDITIONAL
                )
            else:
                kind = HANDLER_UNPLACED if unplaced else HANDLER_SILENT
            # Its own prints alone, the calls aside: asked again only where calls are among the marks.
            own = [by_key[_key(node)] for node in own_prints]
            covered = kind == HANDLER_MARKED and (
                not marking or _every_way(mine, mine, own, end)
            )
            raises = [node for node in _own(mine) if isinstance(node, ast.Raise)]
            # the try the handler guards (a ``try`` or a ``try`` with ``except*``), by its parent link
            body = getattr(getattr(handler, "parent", None), "body", None)
            if not isinstance(body, list) or not body:
                raise ValueError(
                    f"{owner}:{handler.lineno}: the tree has no parent links"
                )
            out.append(
                HandlerMark(
                    owner=owner,
                    line=handler.lineno,
                    catches=ast.unparse(handler.type)
                    if handler.type is not None
                    else "",
                    catch_all=catches_all(handler),
                    try_first=body[0].lineno,
                    try_last=body[-1].end_lineno or body[-1].lineno,
                    kind=kind,
                    prints=sorted({node.lineno for node in own_prints}),
                    calls=calls,
                    reraises=bool(raises)
                    and any(
                        reached for reached in reaching_each(raises, [mine], marks)
                    ),
                    prints_on_every_way=covered,
                )
            )
        return out
