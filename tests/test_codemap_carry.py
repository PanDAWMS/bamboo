"""Which writes hold a statement's rows, carried through the function as it runs.

Each shape marks the statements the rows arrive in ``# rows`` and a statement
whose call arguments (or returned value) should or should not hold a part of
them ``# holds`` / ``# not``.  ``sqlwrite._PathCarry`` is asked, starting from
the marked statements, as the crossing pass asks it -- over no corpus, so no
call is placed and none keeps or returns anything of what it is handed.
"""

from __future__ import annotations

import ast
import textwrap

import pytest

from bamboo.codemap import reach
from bamboo.codemap.calls import Calls, Types
from bamboo.codemap.panda.pathcond import attach_parents
from bamboo.codemap.panda.recognizers.sqlwrite import _PathCarry, _Reading

SHAPES = {
    "a link written before the rows arrived": """
        def f(self, res):
            y = res
            res = self.cur.fetchall()  # rows
            g(y)  # not
            h(res)  # holds
    """,
    "overwritten before the use": """
        def f(self):
            res = self.cur.fetchall()  # rows
            res = other()
            return res  # not
    """,
    "overwritten on one branch only": """
        def f(self, c):
            res = self.cur.fetchall()  # rows
            if c:
                res = other()
            return res  # holds
    """,
    "carried round the loop": """
        def f(self):
            y = None
            for i in range(3):
                g(y)  # holds
                res = self.cur.fetchall()  # rows
                y = res
    """,
    "filed into a dict": """
        def f(self):
            res = self.cur.fetchall()  # rows
            d = {}
            for r in res:
                d[r[0]] = r
            return d  # holds
    """,
    "filed as a dict's keys": """
        def f(self):
            res = self.cur.fetchall()  # rows
            d = {}
            for (task_id,) in res:
                d[task_id] = set()
            return d  # holds
    """,
    "appended": """
        def f(self):
            out = []
            res = self.cur.fetchall()  # rows
            for r in res:
                out.append(r)
            return out  # holds
    """,
    "appended, then the list replaced": """
        def f(self):
            out = []
            res = self.cur.fetchall()  # rows
            for r in res:
                out.append(r)
            out = []
            return out  # not
    """,
    "a status name the next call reuses": """
        def f(self):
            ret_val, a = self.proxy.get()  # rows
            ret_val, b = self.proxy.other()
            use(a)  # holds
            use(ret_val)  # not
    """,
    "rebound from its own value": """
        def f(self):
            limit = self.proxy.get()  # rows
            limit = max(limit, 100)
            use(limit)  # holds
    """,
    "used only to choose, under a loop name used again": """
        def f(self):
            res = self.cur.fetchall()  # rows
            seen = set()
            for (task_id,) in res:
                seen.add(task_id)
            out = []
            for task_id in self.candidates:
                if task_id in seen:
                    out.append(task_id)
            return out  # not
    """,
    "added up": """
        def f(self):
            total = 0
            res = self.cur.fetchall()  # rows
            for r in res:
                total += r[1]
            return total  # holds
    """,
    "rebound in the loop after the use": """
        def f(self):
            res = self.cur.fetchall()  # rows
            for i in range(3):
                g(res)  # holds
                res = other()
    """,
    "caught under the same name": """
        def f(self):
            try:
                res = self.cur.fetchall()  # rows
            except Exception as res:
                return res  # not
            return res  # holds
    """,
    "read by a nested function": """
        def f(self):
            res = self.cur.fetchall()  # rows
            def inner():
                return g(res)  # holds
            return inner
    """,
    "with ... as the same name": """
        def f(self, p):
            res = self.cur.fetchall()  # rows
            with open(p) as res:
                g(res)  # not
    """,
    "a comprehension's own target": """
        def f(self):
            res = self.cur.fetchall()  # rows
            names = [res for res in range(3)]
            g(res)  # holds
    """,
    "returned before the rows arrive": """
        def f(self, c):
            out = []
            if c:
                return out  # not
            res = self.cur.fetchall()  # rows
            for r in res:
                out.append(r)
            return out  # holds
    """,
    # Over no corpus no method is placed, so what ``pack`` keeps of its
    # argument is not read (``_get_datasets_with_unprocessed_inputs``; placed,
    # it is -- test_codemap's rows put into an object).
    "packed into an element by a method": """
        def f(self, chunks):
            for chunk in chunks:
                res = self.cur.fetchone()  # rows
                spec = Spec()
                spec.pack(res)
                chunk.addMasterDS(spec)
            return chunks  # not
    """,
}


def _cases() -> list[tuple[str, int]]:
    return [
        (title, number)
        for title, source in SHAPES.items()
        for number, line in enumerate(textwrap.dedent(source).splitlines(), 1)
        if line.rstrip().endswith(("# holds", "# not"))
    ]


def _taken(statement: ast.stmt) -> list[ast.AST]:
    if isinstance(statement, ast.Return):
        return [statement.value]
    assert isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)
    return list(statement.value.args)


@pytest.mark.parametrize(("title", "line"), _cases(), ids=[f"{t}@{n}" for t, n in _cases()])
def test_a_write_holds_the_rows_where_some_run_brings_them(title, line):
    text = textwrap.dedent(SHAPES[title])
    tree = ast.parse(text)
    attach_parents(tree)
    lines = text.splitlines()
    statements = {node.lineno: node for node in ast.walk(tree) if isinstance(node, ast.stmt)}
    rows = [statements[n] for n, text_line in enumerate(lines, 1) if text_line.rstrip().endswith("# rows")]
    reading = _Reading(Calls(Types({}, {})), set(), reach.ReachGraph([]))
    carry = _PathCarry(reading, "f.py", tree.body[0], start=rows)

    use = statements[line]
    held = any(carry.held(taken, use) for taken in _taken(use))

    assert held == lines[line - 1].rstrip().endswith("# holds")
