"""A statement's row count, and what an ``if`` testing it says of the rows the statement changed.

``_read_task_with_unprocessed_inputs`` locks a task::

    self.cur.execute(sql_to_lock_task + comment, var_map)
    n_row = self.cur.rowcount
    if n_row != 1:
        tmp_log.debug(f"failed to lock jediTaskID={jedi_task_id}")

The print runs only where the ``UPDATE`` changed no row (or more than one): it says how the statement's run ended,
not which value an arm of the function chose.  So it is no frame of the branches the block decides
(:attr:`~bamboo.codemap.models.Branch.messages`); it is a print of the run, with what it says of the rows
(:class:`~bamboo.codemap.models.CountPrint`), on each branch whose write lands at the statement.

**The count.**  ``x.rowcount`` itself, or a name every binding of which reaching the read binds it to ``x.rowcount``.
Its run is the one statement run (``execute`` / ``executemany``) reaching the place the count is read with no other
run between -- the binding for a name, the test for ``x.rowcount`` itself; a count two runs can reach is no one run's.

**The test.**  The count against 0 or 1 -- ``== 0``, ``< 1``, ``<= 0``, ``!= 0``, ``> 0``, ``>= 1``, ``== 1``,
``!= 1``, either side -- or the count alone, or ``not`` it; and those as one operand of an ``and``, whose body holds
the comparison's reading and whose ``else`` holds nothing, since another operand can fail there.
"""

from __future__ import annotations

import ast
from typing import Optional

from bamboo.codemap import calls as C
from bamboo.codemap import pathcond
from bamboo.codemap.models import ROWS_NONE, ROWS_NOT_ONE, ROWS_ONE, ROWS_SOME
from bamboo.codemap.reaching import reaching

#: The calls that run a statement on a cursor.
_RUNS = frozenset({"execute", "executemany"})

#: ``(operator, constant)`` with the count on the left -> what the body and the else say of the rows.
_COMPARED: dict[tuple[type, int], tuple[str, str]] = {
    (ast.Eq, 0): (ROWS_NONE, ROWS_SOME),
    (ast.LtE, 0): (ROWS_NONE, ROWS_SOME),
    (ast.Lt, 1): (ROWS_NONE, ROWS_SOME),
    (ast.NotEq, 0): (ROWS_SOME, ROWS_NONE),
    (ast.Gt, 0): (ROWS_SOME, ROWS_NONE),
    (ast.GtE, 1): (ROWS_SOME, ROWS_NONE),
    (ast.Eq, 1): (ROWS_ONE, ROWS_NOT_ONE),
    (ast.NotEq, 1): (ROWS_NOT_ONE, ROWS_ONE),
}

#: The operator that says the same with the sides swapped: ``0 == n`` is ``n == 0``, ``0 < n`` is ``n > 0``.
_SWAPPED: dict[type, type] = {
    ast.Eq: ast.Eq,
    ast.NotEq: ast.NotEq,
    ast.Lt: ast.Gt,
    ast.Gt: ast.Lt,
    ast.LtE: ast.GtE,
    ast.GtE: ast.LtE,
}


def _counted(binding: pathcond.Binding) -> bool:
    """Whether *binding* binds its name, whole, to a cursor's ``rowcount``."""
    return (
        binding.kind in (pathcond.BIND_ASSIGN, pathcond.BIND_ANNOTATED)
        and not binding.unpacked
        and isinstance(binding.value, ast.Attribute)
        and binding.value.attr == "rowcount"
    )


def _read_at(node: ast.expr, func: ast.AST) -> Optional[list[ast.AST]]:
    """Where the count *node* is read from its run: the test's statement for ``x.rowcount``, each binding reaching it
    for a name bound only to one; ``None`` where *node* is no count."""
    if isinstance(node, ast.Attribute) and node.attr == "rowcount":
        return [pathcond.site_of(node)]
    if not isinstance(node, ast.Name):
        return None
    own = [one for one in pathcond.bindings(func, node.id) if not one.nested]
    if not own:
        return None
    reached = {id(site) for site in reaching(pathcond.site_of(node), [one.site for one in own])}
    held = [one for one in own if id(one.site) in reached]
    if not held or not all(_counted(one) for one in held):
        return None
    return [one.site for one in held]


def _compared(test: ast.expr, func: ast.AST) -> Optional[tuple[tuple[str, str], ast.expr]]:
    """``((body, else), count)`` for a test of a count alone, ``not`` it, or it against 0 or 1."""
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not) and _read_at(test.operand, func) is not None:
        return (ROWS_NONE, ROWS_SOME), test.operand
    if _read_at(test, func) is not None:
        return (ROWS_SOME, ROWS_NONE), test
    if not (isinstance(test, ast.Compare) and len(test.ops) == 1):
        return None
    left, right, op = test.left, test.comparators[0], type(test.ops[0])
    if isinstance(left, ast.Constant) and op in _SWAPPED:
        left, right, op = right, left, _SWAPPED[op]
    if not (isinstance(right, ast.Constant) and type(right.value) is int) or _read_at(left, func) is None:
        return None
    reading = _COMPARED.get((op, right.value))
    return (reading, left) if reading is not None else None


def arms(test: ast.expr, func: ast.AST) -> Optional[tuple[Optional[str], Optional[str]]]:
    """What the body and the ``else`` of an ``if`` on *test* say of the rows a run changed, where *test* tests a row
    count (see the module); ``None`` where it does not."""
    found = _compared(test, func)
    if found is not None:
        return found[0]
    if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And):
        each = [one for one in (_compared(value, func) for value in test.values) if one is not None]
        if len(each) == 1:
            return each[0][0][0], None
    return None


def _count_of(test: ast.expr, func: ast.AST) -> Optional[ast.expr]:
    """The count *test* compares, as :func:`arms` reads it."""
    found = _compared(test, func)
    if found is not None:
        return found[1]
    if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And):
        each = [one for one in (_compared(value, func) for value in test.values) if one is not None]
        if len(each) == 1:
            return each[0][1]
    return None


#: Held on a function: ``(runs, {id(if): (if, body, else, the run its count is of)})``, read once.
_HELD = "_rowcount_tests"

_Test = tuple[ast.If, Optional[str], Optional[str], Optional[ast.AST]]
_Tests = tuple[list[ast.AST], dict[int, _Test]]


def _tests(func: ast.AST) -> _Tests:
    """The statement runs of *func*, and each ``if`` of it testing a row count with its reading and its run -- none
    where *func* reads no ``rowcount``, which is most of them."""
    cached: Optional[_Tests] = getattr(func, _HELD, None)
    if cached is not None:
        return cached
    runs: list[ast.AST] = []
    tests: dict[int, _Test] = {}
    nodes = list(C.own_nodes(func))
    if any(isinstance(node, ast.Attribute) and node.attr == "rowcount" for node in nodes):
        runs = [
            pathcond.site_of(node)
            for node in nodes
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in _RUNS
        ]
        for node in nodes:
            if not isinstance(node, ast.If):
                continue
            reading = arms(node.test, func)
            count = _count_of(node.test, func)
            if reading is None or count is None:
                continue
            read_at = _read_at(count, func) or []
            ran = {id(one): one for one in (_one_run(site, runs) for site in read_at)}
            run = next(iter(ran.values())) if len(ran) == 1 else None
            tests[id(node)] = (node, reading[0], reading[1], run)
    held = (runs, tests)
    setattr(func, _HELD, held)
    return held


def is_count_test(node: ast.If, func: ast.AST) -> bool:
    """Whether the ``if`` *node* of *func* tests a row count (:func:`arms`), whoever's."""
    return id(node) in _tests(func)[1]


def tests_of(func: ast.AST, call: ast.AST) -> list[tuple[ast.If, Optional[str], Optional[str]]]:
    """``(if, body, else)`` for each ``if`` of *func* testing the row count of the run *call* makes, with what its
    body and its ``else`` say of the rows (:func:`arms`), in line order."""
    _runs, tests = _tests(func)
    mine = pathcond.site_of(call)
    found = [(node, body, other) for node, body, other, run in tests.values() if run is not None and run is mine]
    return sorted(found, key=lambda one: one[0].lineno)


def count_of_run(node: ast.expr, func: ast.AST, call: ast.AST) -> bool:
    """Whether *node* is the row count of the run *call* makes, read as an ``if`` reads one (see the module): the
    count itself, or a name every binding of which reaching *node* binds it to the count, and that run the one
    reaching each read with no other between -- ``updateJobStatus`` prints ``nUp={nUp}`` after ``nUp =
    self.cur.rowcount`` and its ``execute``."""
    read_at = _read_at(node, func)
    if not read_at:
        return False
    runs = _tests(func)[0]
    mine = pathcond.site_of(call)
    return all(_one_run(site, runs) is mine for site in read_at)


def _one_run(site: ast.AST, runs: list[ast.AST]) -> Optional[ast.AST]:
    """The one run reaching *site* with no other between, or ``None``."""
    reached = reaching(site, runs)
    return reached[0] if len(reached) == 1 else None
