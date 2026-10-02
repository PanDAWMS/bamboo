"""Every reader of the binding syntax reads it the same way.

A rule written twice moves once.  The walk asked six questions of the syntax,
each with its own traversal, and the audit that checks the walk holds a
seventh on purpose -- a shared traversal is a shared blind spot.  So the
readers stay apart and this holds them to one answer: for each way Python
gives a name a value, every reader that asks "which statements bind ``x``
here" names the same lines, or the difference is listed below with its reason.

The forms are written out rather than drawn from the corpus, so a form PanDA
does not use today is held to the rule as well as one it does.
"""

import ast
import textwrap

import pytest

from bamboo.codemap import calls, trace
from bamboo.codemap.panda import pathcond, sql
from bamboo.codemap.panda.recognizers import sqlwrite
from bamboo.scripts.panda import score_trace

#: form -> source binding (or not binding) ``x`` once, inside ``def f``.
FORMS = {
    "assign": "x = 1",
    "annotated": "x: int = 1",
    "annotated, no value": "x: int",
    "augmented": "x += 1",
    "tuple": "x, y = g()",
    "nested tuple": "(a, (x, b)) = g()",
    "starred": "*x, y = g()",
    "list target": "[x, y] = g()",
    "chained": "y = x = 1",
    "for": "for x in L:\n    pass",
    "for tuple": "for i, x in enumerate(L):\n    pass",
    "for nested tuple": "for (i, (x, j)) in L:\n    pass",
    "with": "with g() as x:\n    pass",
    "with tuple": "with g() as (x, y):\n    pass",
    "with second item": "with g() as a, h() as x:\n    pass",
    "walrus": "if (x := g()):\n    pass",
    "except as": "try:\n    pass\nexcept E as x:\n    pass",
    "import": "import x",
    "def": "def x():\n    pass",
    "subscript store": "x[0] = 1",
    "attribute store": "x.attr = 1",
    "method call": "x.add(1)",
    "del": "del x",
    "comprehension": "z = [x for x in L]",
    "lambda parameter": "z = lambda x: x",
    "nested def": "def inner():\n    x = 2",
}

#: The forms that give ``f``'s own ``x`` a value.  ``import`` and ``def``
#: bind a module or a function, which no reader is asked about.
BINDS = {
    "assign", "annotated", "augmented", "tuple", "nested tuple", "starred",
    "list target", "chained", "for", "for tuple", "for nested tuple", "with",
    "with tuple", "with second item", "walrus", "except as",
}  # fmt: skip


def _func(form: str) -> ast.FunctionDef:
    tree = ast.parse("def f(L, E, g, h):\n" + textwrap.indent(FORMS[form], "    ") + "\n")
    pathcond.attach_parents(tree)
    return tree.body[0]  # type: ignore[return-value]


def _line(node: ast.AST) -> int:
    return getattr(pathcond.site_of(node), "lineno", 0)


def _walk_lines(func: ast.AST) -> set[int]:
    """The lines ``trace`` offers as binding ``x``, as ``explain`` gathers them."""
    values = list(pathcond.assigned_expressions(func).get("x", []))
    values += [node.value for node, _slot in trace._tuple_bindings(func, "x")]
    values += trace._with_bindings(func, "x")
    values += trace._walrus_bindings(func, "x")
    lines = {_line(value) for value in values}
    lines |= {node.lineno for node in trace._loop_bindings(func, "x")}
    lines |= {node.lineno for node in trace._caught_as(func, "x")}
    return lines


def _own_lines(func: ast.AST) -> set[int]:
    return {b.site.lineno for b in pathcond.bindings(func, "x") if not b.nested}  # type: ignore[attr-defined]


@pytest.mark.parametrize("form", sorted(FORMS))
def test_pathcond_reads_every_form_that_gives_a_name_a_value(form):
    assert bool(_own_lines(_func(form))) == (form in BINDS)


#: Where the audit and the walk differ, and why it is right that they do.
AUDIT_ONLY = {
    # The walk offers ``x.add(v)`` only where every assignment is an empty
    # container (``trace._mutated_bindings``); the audit counts it always, so
    # that it never calls such a step a fabrication.
    "method call",
}


@pytest.mark.parametrize("form", sorted(FORMS))
def test_the_walk_and_the_audit_name_the_same_binding_lines(form):
    func = _func(form)
    walk = _walk_lines(func)
    audit = set(score_trace._binding_lines(func, "x"))

    if form in AUDIT_ONLY:
        assert walk == set() and audit
    else:
        assert walk == audit


@pytest.mark.parametrize("form", sorted(FORMS))
def test_the_walk_reads_its_own_bindings_and_a_nested_scope_s_alike(form):
    """``explain`` keeps a nested function's binding and says where it came from (``_nested_scope``)."""
    func = _func(form)
    nested = {b.site.lineno for b in pathcond.bindings(func, "x") if b.nested}  # type: ignore[attr-defined]

    assert _walk_lines(func) == _own_lines(func) | nested


@pytest.mark.parametrize("form", sorted(FORMS))
def test_a_message_s_bindings_are_read_with_the_same_syntax(form):
    """``_message_sites`` reads the function's own body only, as a log call's message is its own.

    A caught exception is anchored at the handler's first statement there,
    which is where the name first holds it; the handler is the binding.
    """
    plain, appended, opaque = trace._message_sites(_func(form), "x")
    lines = {getattr(node, "lineno", 0) for node in [*plain, *appended, *opaque]}
    if form == "except as":
        lines = {line - 1 for line in lines}

    assert lines == _own_lines(_func(form))


@pytest.mark.parametrize("form", sorted(FORMS))
def test_a_new_value_that_parts_two_tests_is_read_with_the_same_syntax(form):
    """``pathcond._rebinds`` adds the in-place changes on purpose -- see there."""
    func = _func(form)
    rebinds = {node.lineno for node in pathcond._rebinds(func).get("x", ())}  # type: ignore[attr-defined]
    in_place = {"subscript store", "method call"}

    assert rebinds == (_own_lines(func) | ({2} if form in in_place else set()))


@pytest.mark.parametrize("form", sorted(FORMS))
def test_a_parameter_given_a_new_value_is_read_with_the_same_syntax(form):
    """``calls._rebinds`` asks whether the value handed in may not be the one read; ``del`` ends it too."""
    assert calls._rebinds(_func(form), "x") == (form in BINDS or form == "del")


@pytest.mark.parametrize("form", sorted(FORMS))
def test_a_map_is_renewed_by_the_bindings_that_replace_it(form):
    """``sql._renews``: ``+=`` adds to the map, and a ``with`` / ``:=`` / ``except`` never holds one."""
    func = _func(form)
    renewed = {node.lineno for node in sql._renews(func, "x")}
    kinds = {pathcond.BIND_ASSIGN, pathcond.BIND_ANNOTATED, pathcond.BIND_FOR}
    expected = {b.site.lineno for b in pathcond.bindings(func, "x") if b.kind in kinds}  # type: ignore[attr-defined]

    assert renewed == expected


@pytest.mark.parametrize("form", sorted(FORMS))
def test_the_build_takes_a_target_apart_the_same_way(form):
    """``_Index.definitions``, ``sqlwrite._assigned_names`` and ``sqlwrite._bound_by`` on ``=``."""
    func = _func(form)
    statement = func.body[0]
    kinds = {pathcond.BIND_ASSIGN, pathcond.BIND_ANNOTATED}
    by_syntax = any(b.kind in kinds and not b.nested for b in pathcond.bindings(func, "x"))

    assert ("x" in pathcond._index(func).definitions()) == (
        by_syntax or form == "nested def"
    )  # ``single_definition`` counts a nested function's writes, as it always has
    if isinstance(statement, ast.Assign):
        assert ("x" in sqlwrite._assigned_names(statement)) == by_syntax
    bound = "x" in sqlwrite._bound_by(statement)
    assert bound == (by_syntax or form in ("augmented", "subscript store"))
