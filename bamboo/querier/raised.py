"""An exception raised inside the corpus, read as a guard that came out false.

``d[k]`` raising ``KeyError`` says ``k in d`` was false at that line, as
surely as an ``if k in d:`` that took its else.  What each expression
requires in order not to raise is Python's semantics, not PanDA's, so the
table is closed: it does not grow with the cases, only with the language.

=================  ====================  =========================================
expression         exception             the guard that was false
=================  ====================  =========================================
``d[k]``, ``d.pop(k)``  ``KeyError``     ``k in d``
``s[i]``           ``IndexError``        ``-len(s) <= i < len(s)``
``x.a``            ``AttributeError``    ``x is not None`` (the message says ``'NoneType'``)
``a / b``, ``//``, ``%``  ``ZeroDivisionError``  ``b != 0``
``int(x)``, ``float(x)``  ``ValueError``   ``x`` reads as a number
``a, b = x``       ``ValueError``        ``len(x) == 2``
``raise ...``      (any)                 the guard of the ``raise`` (the path condition)
=================  ====================  =========================================

An exception raised outside the corpus is not read here: its frame at the
corpus boundary is the external object's.  The message is read as
the printed value of the operand it names -- ``KeyError`` prints ``repr(k)``,
``int()`` prints ``repr(x)`` -- so rule 9 (value match) can choose among the
operand's bindings afterwards.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any

UNPRINTED = object()


@dataclass
class FalseGuard:
    expr: str  # the raising expression, as source
    atom: str  # the guard that was false, as source
    kind: str  # "in" | "index" | "not None" | "nonzero" | "number" | "length" | "raise"
    operand: str  # the operand the message speaks of
    other: str  # the operand it was tested against ("" when there is none)
    printed: Any = UNPRINTED  # the operand's value, read from the message

    def describe(self) -> str:
        said = "" if self.printed is UNPRINTED else f"; the message says {self.operand} = {self.printed!r}"
        return f"{self.expr} raised: {self.atom!r} was false{said}"


def _literal(text: str) -> Any:
    try:
        return ast.literal_eval(text.strip())
    except (ValueError, SyntaxError):
        return UNPRINTED


def _loads(stmt: ast.stmt):
    for node in ast.walk(stmt):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)) and node is not stmt:
            continue
        yield node


def read(stmt: ast.stmt, exc_type: str, message: str) -> list[FalseGuard]:
    """Every expression of *stmt* that raises *exc_type*, with the guard it says was false.

    *exc_type* is the traceback's last line up to the colon (``KeyError``,
    ``builtins.KeyError``); *message* is the rest.  More than one entry means
    the line alone does not say which expression raised; the caller then
    chooses by value or stops.
    """
    exc = exc_type.rsplit(".", 1)[-1].strip("<>' ")
    exc = re.sub(r"^class '(?:builtins\.)?", "", exc)
    out: list[FalseGuard] = []
    nodes = list(_loads(stmt))
    if isinstance(stmt, ast.Raise):
        return [FalseGuard(ast.unparse(stmt), "", "raise", "", "")]
    if exc == "KeyError":
        for n in nodes:
            if isinstance(n, ast.Subscript) and isinstance(n.ctx, ast.Load) and not isinstance(n.slice, ast.Slice):
                k, d = ast.unparse(n.slice), ast.unparse(n.value)
                out.append(FalseGuard(ast.unparse(n), f"{k} in {d}", "in", k, d, _literal(message)))
            elif (
                isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "pop"
                and len(n.args) == 1 and not n.keywords
            ):
                k, d = ast.unparse(n.args[0]), ast.unparse(n.func.value)
                out.append(FalseGuard(ast.unparse(n), f"{k} in {d}", "in", k, d, _literal(message)))
    elif exc == "IndexError":
        for n in nodes:
            if isinstance(n, ast.Subscript) and isinstance(n.ctx, ast.Load) and not isinstance(n.slice, ast.Slice):
                i, s = ast.unparse(n.slice), ast.unparse(n.value)
                out.append(FalseGuard(ast.unparse(n), f"-len({s}) <= {i} < len({s})", "index", i, s))
    elif exc == "AttributeError":
        m = re.search(r"'NoneType' object has no attribute '(\w+)'", message)
        if m:
            for n in nodes:
                if isinstance(n, ast.Attribute) and n.attr == m[1]:
                    x = ast.unparse(n.value)
                    out.append(FalseGuard(ast.unparse(n), f"{x} is not None", "not None", x, "", None))
    elif exc == "ZeroDivisionError":
        for n in nodes:
            if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Div, ast.FloorDiv, ast.Mod)):
                b = ast.unparse(n.right)
                out.append(FalseGuard(ast.unparse(n), f"{b} != 0", "nonzero", b, "", 0))
    elif exc == "ValueError":
        m = re.search(r"(?:too many|not enough) values to unpack \(expected (\d+)", message)
        if m:
            for n in nodes:
                if isinstance(n, (ast.Assign, ast.For)):
                    targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                    source = n.value if isinstance(n, ast.Assign) else n.iter
                    for t in targets:
                        if isinstance(t, (ast.Tuple, ast.List)) and len(t.elts) == int(m[1]):
                            x = ast.unparse(source)
                            out.append(FalseGuard(ast.unparse(n).splitlines()[0], f"len({x}) == {m[1]}", "length", x, ""))
        m = re.search(r"(?:invalid literal for (int|float)\(\)[^:]*|could not convert string to (float)): (.*)$", message, re.S)
        if m:
            fn = m[1] or m[2]
            for n in nodes:
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == fn and n.args:
                    x = ast.unparse(n.args[0])
                    out.append(FalseGuard(ast.unparse(n), f"{x} reads as {fn}", "number", x, "", _literal(m[3])))
    return out
