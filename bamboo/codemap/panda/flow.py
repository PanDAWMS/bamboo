"""Which statements one call of a function can run after which: one question on the map's flow.

A reader of a run asks it of the statements around what the run printed: can the call have run this statement
between those two records, did every way between them pass it, can this print and that statement both be in one
call.  The map reads values by one flow (:mod:`bamboo.codemap.reaching`), so it answers these as well: the
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
* **Where a run would have gone.**  :func:`kept_from` reads an exit as a ``pass`` too (:func:`reaching`'s
  *passing*): what a run leaving there no longer reaches and would have, the pass's or the turn's.
* **What a way's own literals rule out.**  The ways are path-insensitive, but a name a statement's way binds to one
  literal decides a test after it: :func:`ruled_out_after`.
* **Not read.**  A ``with`` whose manager swallows what its body raised, and a call that never returns.  The
  first is watched: :func:`swallowing_managers` lists each such manager of the corpus into the dump, and the
  ``with-exits-are-read`` gate fails on one (none in PanDA).

The statements asked of must be of one function, on a tree with parent links
(:func:`bamboo.codemap.pathcond.attach_parents`).
"""

from __future__ import annotations

import ast
from typing import Iterable, Iterator, Optional, Sequence

from bamboo.codemap import pathcond
from bamboo.codemap.models import SwallowingManager
from bamboo.codemap.reaching import reaching, reaching_each, reaching_end

#: The nodes whose body is a scope of its own, as :mod:`~bamboo.codemap.reaching` reads them.
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


def passes_after(first: ast.stmt, then: ast.stmt, *, raised: Sequence[ast.ExceptHandler] = ()) -> bool:
    """Whether every way a call goes on by after *first* passes *then* before the turn of the innermost loop around
    *first* ends -- before the call ends where no loop holds it: ``TaskCommando.runImpl`` prints ``set
    task_status=...`` after its ``tmpTaskSpec.status`` arm on every way.  A *then* outside that turn -- past the
    loop, in its ``else`` -- is passed in no turn of it.

    Read for a run that entered none of the handlers but *raised* (none by default), as a reader of a run with no
    handler's mark reads it: a statement raises only into them (:func:`reaching`'s *raised*), and a ``raise``
    statement ends its way short of *then*.  With every statement able to raise, almost no print would follow every
    way."""
    _one_function(first, then)
    func = _scope(first)
    if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
        raise ValueError("the flow answers of a function's pass")
    loops = loops_around(first)
    if loops and not any(then is one or _under(then, one) for one in loops[0].body):  # type: ignore[attr-defined]
        return False
    ends = [node for node in _own(func) if isinstance(node, (ast.Return, ast.Raise))]
    if any(reaching_each(ends, [first], [then], entered=[first], once=loops, raised=raised)):
        return False
    return not reaching_end(func, [first], [then], entered=[first], once=loops, raised=raised)


def turn_loops(first: ast.stmt, then: ast.stmt) -> list[ast.AST]:
    """The loops around *first* whose body holds *then* too, inside out: a way from *first* to *then* stays in one
    turn of each, and goes round any other as often as it may -- a print past the loop follows the arm's turn and
    every later one."""
    return [loop for loop in loops_around(first) if any(then is one or _under(then, one) for one in loop.body)]  # type: ignore[attr-defined]


def changed_between(first: ast.stmt, then: ast.stmt, sites: Sequence[ast.AST]) -> bool:
    """Whether one of *sites* -- statements that bind or change a value -- can run on a way from *first* to *then*,
    in one turn of each loop holding both (:func:`turn_loops`): after an arm and before a print after it.

    A site reaching *then* on a way that never ran *first* is not between them -- ``TaskCommando.runImpl``'s
    ``tmpTaskSpec.status = oldStatus`` at 181, on the way that skips the arm at 197.  A site that is the last write
    before *then* on some way and that *first* reaches is: the two ways make one."""
    _one_function(first, then)
    others = [site for site in sites if site is not first and site is not then]
    if not others:
        return False
    once = turn_loops(first, then)
    last = reaching(then, [first, *others], once=once)
    return any(site is not first and reaching(site, [first], entered=[first], once=once) for site in last)


def ruled_out_after(first: ast.stmt, then: ast.stmt) -> bool:
    """Whether a call that ran *first* cannot run *then* after it in the same turn of each loop around *first* that
    holds *then* (:func:`turn_loops`), by the values *first*'s way gives: a test on *then*'s way reads a name that
    holds one literal where *first* runs (:func:`pathcond.held_at`), nothing binds it again on a way from *first* to
    *then* in that turn, and the literal decides the test against *then* (:func:`pathcond.ruled_out_at`).

    ``reactivatePendingTasks_JEDI``'s ``sql = sqlTU`` arm runs with the ``timeoutFlag = False`` its turn starts
    with, so ``if timeoutFlag:`` keeps it from the ``timeout`` print after the run, which its ``sqlTO`` arm, beside
    ``timeoutFlag = True``, reaches.  False where no test is so decided: a test on a value computed, or on a name
    bound to different literals on the ways to *first*, rules on nothing."""
    _one_function(first, then)
    func = _scope(first)
    if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    once = turn_loops(first, then)
    names = sorted(
        {node.id for test in pathcond.tests_on(then, func) for node in ast.walk(test) if isinstance(node, ast.Name)}
    )
    for name in names:
        held = pathcond.held_at(func, name, first)
        if held.kind != pathcond.HELD_CONSTANT:
            continue
        # A parameter is bound as the call starts, before any turn.
        sites = [
            one.site if isinstance(one.site, (ast.stmt, ast.ExceptHandler)) else pathcond.site_of(one.site)
            for one in pathcond.bindings(func, name)
            if not one.nested and not isinstance(one.site, ast.arg)
        ]
        if any(
            reaching(site, [first], entered=[first], once=once) and reaching(then, [site], once=once)
            for site in sites
        ):
            continue
        if pathcond.ruled_out_at(then, func, name, held.value):
            return True
    return False


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


def kept_from(
    exit_: ast.Return | ast.Raise | ast.Continue | ast.Break,
    statements: Sequence[ast.stmt],
    over: Optional[ast.AST] = None,
) -> list[ast.stmt]:
    """Of *statements*, the ones a run leaving by *exit_* can no longer reach and could have reached had it gone on
    past it as a ``pass``, in the order given: what the exit keeps the run from.

    A ``return`` or a raise ends the call's pass: a ``finally`` it runs and a handler catching the raise -- and what
    follows them -- are still reached.  A ``continue`` or a ``break`` ends one turn of its loop: the loop and every
    loop around it are read as gone round once, and only statements in the loop's body or ``else`` count (a
    ``break`` keeps the run from the ``else``; a ``continue`` does not).  So ``getTasksToBeFinished_JEDI``'s
    ``continue`` at 1667 keeps a run from the turn's writes at 1673, 1676 and 1690, and its ``raise`` at 1666,
    caught by the ``except`` around the loop that returns, from the count print at 1717 after the loop.

    *over*, a loop whose body holds *exit_*, asks the same over one turn of it, whatever the exit: that loop and
    every loop around it gone round once, every loop inside it as often as it may, and only statements in its body
    or ``else`` counted.  ``TaskCommandoThread.runImpl``'s ``break`` at 211, over a turn of the ``for`` at 123 around
    its own loop, keeps a run from the arms another turn of its own loop reaches.

    The flow is path-insensitive on both sides: a statement only an impossible way after the exit reaches is read as
    reached, so an answer may leave out what the exit does keep the run from; and one only an impossible way past the
    exit reaches is read as reachable, so an answer may hold what no run going on past it could reach."""
    _one_function(exit_, *statements)
    once: list[ast.AST] = []
    asked = list(statements)
    loops = loops_around(exit_)
    if over is None and isinstance(exit_, (ast.Continue, ast.Break)):
        if not loops:
            return []
        over = loops[0]
    if over is not None:
        index = next((number for number, loop in enumerate(loops) if loop is over), None)
        if index is None:
            raise ValueError(f"line {getattr(over, 'lineno', '?')} is no loop whose body holds the exit")
        once = loops[index:]
        asked = [statement for statement in statements if _under(statement, over)]
    if not asked:
        return []
    after = reaching_each(asked, [exit_], entered=[exit_], once=once)
    passing = reaching_each(asked, [exit_], entered=[exit_], once=once, passing=[exit_])
    return [statement for statement, left, went in zip(asked, after, passing, strict=True) if went and not left]


def loops_around(node: ast.AST) -> list[ast.AST]:
    """The loops whose body holds *node* in its function, inside out: a ``continue`` or ``break`` in a loop's
    ``else`` is the loop around it's."""
    found: list[ast.AST] = []
    child: ast.AST = node
    current = getattr(node, "parent", None)
    while current is not None and not isinstance(current, _SCOPES):
        if isinstance(current, (ast.For, ast.AsyncFor, ast.While)) and any(one is child for one in current.body):
            found.append(current)
        child, current = current, getattr(current, "parent", None)
    return found


def _under(node: ast.AST, ancestor: ast.AST) -> bool:
    current = getattr(node, "parent", None)
    while current is not None:
        if current is ancestor:
            return True
        current = getattr(current, "parent", None)
    return False


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
