"""Which statements one call of a function can run after which: one question on the map's flow.

A reader of a run asks it of the statements around what the run printed: can the call have run this statement
between those two records, did every way between them pass it, can this print and that statement both be in one
call.  The map reads values by one flow (:mod:`bamboo.codemap.panda.reaching`), so it answers these as well: the
question is :func:`runs_after`, and the others are made of it.

* **A statement has run once it is entered.**  A call that raised ran as far as the raise -- its writes, a
  callee's prints -- so a statement asked of counts from its start, its own raise to a handler included, and one
  a way must pass is passed once it is entered: a ``for`` whose iterable was read, with no pass.  A ``for``'s
  iterable is read once, as the loop is entered, not on each pass.
* **A print has run once it printed.**  A reader asks of prints whose records it read, and one that raised
  printed nothing: so a print counts from its end, and a way through its raise passes it -- the *stops*, and a
  *first* said *printed*.
* **The ways are the flow's.**  Every branch both ways; a loop any number of times, a ``while True`` leaving by
  its ``break`` only; a statement that can raise, to each handler around it and on out -- any attribute,
  subscript, call or format, a print's own logger call included; ``finally`` on every way out.
* **What a run adds.**  *once*: loops the run went round exactly once (a reader's ``once_loops``) have no way
  back and run their body, as a ``for`` over ``range(1)`` does.  *raised*: the handlers whose marks the run
  shows; handed them, a raise is read by where it is, not by what raises -- any statement of a ``try`` holding
  one, before a catch-all not among them -- and only they are entered (:func:`reaching.catching_try`).  A run
  with no mark hands none: every ``try`` ran to its end.
* **Not read.**  A ``with`` whose manager swallows what its body raised, and a call that never returns.  The
  first is watched: :func:`swallowing_managers` lists each such manager of the corpus into the dump, and the
  ``with-exits-are-read`` gate fails on one (none in PanDA).

The statements asked of must be of one function, on a tree with parent links
(:func:`bamboo.codemap.panda.pathcond.attach_parents`).
"""

from __future__ import annotations

import ast
from typing import Iterable, Iterator, Optional, Sequence

from bamboo.codemap.models import SwallowingManager
from bamboo.codemap.panda.reaching import reaching, reaching_end

#: The nodes whose body is a scope of its own, as :mod:`~bamboo.codemap.panda.reaching` reads them.
_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)
_NESTED = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)

#: How a context manager may swallow what its ``with`` body raised (:attr:`SwallowingManager.kind`).
SWALLOW_CONTEXTMANAGER = "contextmanager"
SWALLOW_EXIT = "exit"
SWALLOW_SUPPRESS = "suppress"


def _scope(node: ast.AST) -> Optional[ast.AST]:
    current = getattr(node, "parent", None)
    while current is not None and not isinstance(current, _SCOPES):
        current = getattr(current, "parent", None)
    return current


def _one_function(*statements: ast.AST) -> None:
    scopes = set()
    for statement in statements:
        if getattr(statement, "parent", None) is None:
            raise ValueError(
                f"the flow needs parent links on the tree holding line {getattr(statement, 'lineno', '?')}"
            )
        scopes.add(id(_scope(statement)))
    if len(scopes) > 1:
        raise ValueError("the flow answers of one function's statements")


def runs_after(
    first: ast.stmt,
    then: ast.stmt,
    stops: Sequence[ast.stmt] = (),
    *,
    once: Sequence[ast.AST] = (),
    printed: bool = False,
    raised: Optional[Sequence[ast.ExceptHandler]] = None,
) -> bool:
    """Whether one call can run *then* after *first* has started -- or, *printed*, after the print *first* printed
    -- with none of *stops* (prints, passed once they printed) run between."""
    _one_function(first, then)
    killers = [stop for stop in stops if stop is not first]
    return bool(
        reaching(
            then,
            [first],
            killers,
            entered=[] if printed else [first],
            once=once,
            raised=raised,
        )
    )


def runs_between(
    first: ast.stmt,
    last: ast.stmt,
    site: ast.stmt,
    prints: Sequence[ast.stmt],
    *,
    once: Sequence[ast.AST] = (),
    raised: Optional[Sequence[ast.ExceptHandler]] = None,
) -> bool:
    """Whether *site* can run after the print *first* and before the print *last*, with none of *prints* between.

    A run's two records in a row of a function's prints then hold the call's pass by *site*, though no print
    every way runs is next to it: ``feed_contents_to_tasks`` calls ``updateTaskStatusByContFeeder_JEDI`` at the
    end of an arm, and its next print is in another."""
    if any(site is one for one in prints):
        return False
    return runs_after(
        first, site, prints, once=once, printed=True, raised=raised
    ) and runs_after(site, last, prints, once=once, raised=raised)


def passes_through(
    first: ast.stmt,
    last: ast.stmt,
    site: ast.stmt,
    prints: Sequence[ast.stmt],
    *,
    once: Sequence[ast.AST] = (),
    raised: Optional[Sequence[ast.ExceptHandler]] = None,
) -> bool:
    """Whether every way from the print *first* to the print *last* with none of *prints* between passes *site*,
    so two records in a row say the call ran it: ``TaskRefinerThread.runImpl`` prints at 205 and 563 around its
    call of ``extractCommon`` at 211, which prints nothing."""
    if not runs_between(first, last, site, prints, once=once, raised=raised):
        return False
    stops = [stop for stop in prints if stop is not first]
    return not reaching(
        last, [first], [*stops, site], entered=[site], once=once, raised=raised
    )


def passes_before(
    printed: ast.stmt,
    sites: Sequence[ast.stmt],
    *,
    once: Sequence[ast.AST] = (),
    raised: Optional[Sequence[ast.ExceptHandler]] = None,
) -> bool:
    """Whether every way from the entry of the function holding the print *printed* to its end passes one of
    *sites*, so its record says the call ran one of them: ``setupper_atlas_plugin.run`` prints ``end`` at 206
    only after calling ``setup_jumbo_jobs`` at 202, which prints nothing.

    Each of *sites* is passed once entered, a raise inside it to a handler included, and a site the print holds
    (``log.debug(f"{fn()}")``) is passed before it prints.  Hand every site of the question at once: after
    ``if a: fn(1)`` / ``else: fn(2)`` each way passes one, and neither alone."""
    _one_function(printed, *sites)
    if any(site is printed for site in sites):
        return True
    func = _scope(printed)
    if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
        raise ValueError("the flow answers from a function's entry")
    return not reaching(
        printed, [func.args], sites, entered=sites, once=once, raised=raised
    )


def together(
    printed: ast.stmt,
    site: ast.stmt,
    *,
    once: Sequence[ast.AST] = (),
    raised: Optional[Sequence[ast.ExceptHandler]] = None,
) -> bool:
    """Whether one call can print *printed* and run *site*, in either order.

    Where it cannot, a call that printed it did not run *site*: below the ``return`` every way from the print
    ends at, in the other arm of an ``if`` no loop holds with it, in another arm of a pass of a loop *once* says
    went round once."""
    return runs_after(
        printed, site, once=once, printed=True, raised=raised
    ) or runs_after(site, printed, once=once, raised=raised)


def ends_after(
    first: ast.stmt | ast.arguments,
    stops: Sequence[ast.stmt] = (),
    *,
    passed: Sequence[ast.stmt] = (),
    once: Sequence[ast.AST] = (),
    raised: Optional[Sequence[ast.ExceptHandler]] = None,
) -> bool:
    """Whether one call can end its pass after the print *first* -- print *first* again, or run off the end of
    the function -- with none of *stops* (prints, passed once they printed) or *passed* (statements, passed once
    entered) on the way.  *first* the function's parameters (its ``args``) asks from the call's entry.

    A reader that cuts a run at each record of a print every pass starts with reads a pass as running to the
    next one, and chooses the way it went among *passed* -- arms, the function's own ``return`` statements; a
    way that ends past all of them is one more.  ``check_output_file_metadata`` leaves by a ``return`` or after
    its loop over the files with none taken, and no print is on every way there.  A ``return`` and a raise out
    of the function are not its end: the reader asks of them as statements."""
    _one_function(first, *stops, *passed)
    func = _scope(first)
    if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
        raise ValueError("the flow answers of a function's pass")
    killers = [*(stop for stop in stops if stop is not first), *passed]
    if isinstance(first, ast.stmt) and reaching(
        first, [first], killers, entered=passed, once=once, raised=raised
    ):
        return True
    return bool(
        reaching_end(func, [first], killers, entered=passed, once=once, raised=raised)
    )


# -- the premise: no with goes on past a raise in its body ------------------------------------------------------ #


def _own(node: ast.AST) -> Iterator[ast.AST]:
    """``ast.walk`` below *node* that does not go into a nested function, class or lambda."""
    stack = list(ast.iter_child_nodes(node))
    while stack:
        one = stack.pop()
        yield one
        if not isinstance(one, _NESTED):
            stack.extend(ast.iter_child_nodes(one))


def _always_raises(block: list[ast.stmt]) -> bool:
    if not block:
        return False
    last = block[-1]
    if isinstance(last, ast.Raise):
        return True
    return (
        isinstance(last, ast.If)
        and _always_raises(last.body)
        and _always_raises(last.orelse)
    )


def _swallows_around_yield(func: ast.AST) -> Optional[int]:
    """The line where a ``contextmanager`` function may swallow what its ``with`` body raised: a handler around
    its ``yield`` that may end without raising, or a ``return`` in a ``finally`` around it."""
    for node in _own(func):
        if not isinstance(node, ast.Try) or not any(
            isinstance(one, (ast.Yield, ast.YieldFrom))
            for part in node.body
            for one in [part, *_own(part)]
        ):
            continue
        for handler in node.handlers:
            if not _always_raises(handler.body):
                return handler.lineno
        for part in node.finalbody:
            for one in [part, *_own(part)]:
                if isinstance(one, ast.Return):
                    return one.lineno
    return None


def _exit_swallows(method: ast.AST) -> Optional[int]:
    """The line of a ``return`` of an ``__exit__`` that may hand back what is not None or False."""
    for node in _own(method):
        if isinstance(node, ast.Return) and node.value is not None:
            if not (
                isinstance(node.value, ast.Constant)
                and node.value.value in (None, False)
            ):
                return node.lineno
    return None


def swallowing_managers(
    modules: Iterable[tuple[str, ast.AST]],
) -> tuple[int, list[SwallowingManager]]:
    """``(read, swallowing)``: how many context managers *modules* (each its path and tree) define or name, and
    those that may swallow what a ``with`` body raised -- the flow's stated limit, checked over the corpus.

    A manager from outside the corpus is read as not swallowing, ``contextlib.suppress`` aside."""
    read = 0
    found: list[SwallowingManager] = []
    for path, tree in modules:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names = {ast.unparse(one).split(".")[-1] for one in node.decorator_list}
                if names & {"contextmanager", "asynccontextmanager"}:
                    read += 1
                    line = _swallows_around_yield(node)
                    if line is not None:
                        found.append(
                            SwallowingManager(
                                file=path,
                                line=line,
                                name=node.name,
                                kind=SWALLOW_CONTEXTMANAGER,
                            )
                        )
            elif isinstance(node, ast.ClassDef):
                for method in node.body:
                    if isinstance(
                        method, (ast.FunctionDef, ast.AsyncFunctionDef)
                    ) and method.name in (
                        "__exit__",
                        "__aexit__",
                    ):
                        read += 1
                        line = _exit_swallows(method)
                        if line is not None:
                            found.append(
                                SwallowingManager(
                                    file=path,
                                    line=line,
                                    name=f"{node.name}.{method.name}",
                                    kind=SWALLOW_EXIT,
                                )
                            )
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                for item in node.items:
                    expression = item.context_expr
                    called = (
                        expression.func
                        if isinstance(expression, ast.Call)
                        else expression
                    )
                    if ast.unparse(called).split(".")[-1] == "suppress":
                        read += 1
                        found.append(
                            SwallowingManager(
                                file=path,
                                line=node.lineno,
                                name=ast.unparse(expression)[:80],
                                kind=SWALLOW_SUPPRESS,
                            )
                        )
    return read, found
