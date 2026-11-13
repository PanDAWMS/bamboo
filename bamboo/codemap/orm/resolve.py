"""What value an ORM write gives its field, read with the shared value machinery.

The ladder, for the expression a write puts into a field:

1. **Settled:** a literal, a module or class constant, a Django enumeration
   member (``Task.Status.DONE``), or what the shared ``values.resolver()``
   settles (declared mappings and constants across the corpus).  Tier 1.
2. **A conditional expression** ``'a' if ok else 'b'``: each arm, under its
   test.
3. **A copy of a column** ``F('other')``: ``passthrough(Model.other)``.  Tier 2.
4. **A local narrowed by a membership guard:** ``if s not in STATES: raise``
   (or ``return`` / ``continue``) above the write, with ``s`` not bound again
   in between, says ``s`` is one of ``STATES`` where the write runs; so does an
   enclosing ``if s in STATES:``.  Each member is a value, tier 1, under
   ``s == <member>``.  This is how request handlers set statuses: the value
   comes from the request, and the code checks it against the declared list
   before it writes.
5. **A local:** every binding ``pathcond.reaching_values`` says can reach the
   write, each under its own conditions, with a binding's expression read
   again through this ladder.
6. Otherwise ``runtime(<expression>)``, tier 2: the value is decided as the
   code runs.

Steps 1, 5 and the conditions are the shared machinery's; 2 to 4 are what an
ORM code base needed in addition, and none is ORM-specific -- they would read a
PanDA write the same way.
"""

from __future__ import annotations

import ast
from typing import Callable, NamedTuple, Optional

from bamboo.codemap.orm.schema import DjangoModel, choice_values, dotted
from bamboo.codemap.panda import pathcond, values

Settle = Callable[[ast.expr, Optional[ast.AST]], list[str]]

#: How far a local's binding is followed through other locals.
MAX_DEPTH = 3


class Resolved(NamedTuple):
    """One value a write can give its field."""

    outcome: str
    #: 1 settled from the source, 2 decided at run time
    tier: int
    #: what must hold for this value, beyond the write's own path condition
    conditions: list[str]
    #: where the value is decided (a binding's line), None where it is the write's own
    line: Optional[int] = None


def settler(models: dict[str, DjangoModel], constants: dict[str, list[str]]) -> Settle:
    """A ``(expression, func) -> [value, ...]`` for one module: its own constants and the models'
    enumerations first, then the shared :func:`bamboo.codemap.panda.values.resolver`.

    The models' first because the shared resolver reads a ``(value, label)`` list as both strings.
    Call the shared one inside ``sql.across_calls(calls)`` so it can read constants other modules declare.
    """
    shared = values.resolver()
    model_constants = {
        f"{m}.{k}": v for m, info in models.items() for k, v in info.choice_sets.items()
    }

    def settle(expression: ast.expr, func: Optional[ast.AST]) -> list[str]:
        if isinstance(expression, ast.Constant):
            return (
                [expression.value]
                if isinstance(expression.value, str)
                else [ast.unparse(expression)]
            )
        name = dotted(expression)
        if name is not None:
            if name.endswith(".value"):
                name = name[: -len(".value")]
            head = name.split(".")[0]
            local = func is not None and (
                _is_parameter(func, head) or bool(pathcond.bindings(func, head))
            )
            if not local:
                if name in model_constants:
                    return list(model_constants[name])
                if name in constants:
                    return list(constants[name])
        return shared(expression, func)  # type: ignore[arg-type]

    return settle


def _is_parameter(func: ast.AST, name: str) -> bool:
    args = getattr(func, "args", None)
    if args is None:
        return False
    return name in {
        a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)
    } or name in {a.arg for a in (args.vararg, args.kwarg) if a is not None}


# --------------------------------------------------------------------------- narrowing


def _members(
    container: ast.expr,
    func: ast.AST,
    settle: Settle,
    models: dict[str, DjangoModel],
    depth: int,
) -> Optional[list[str]]:
    """The values a membership test's container holds, where the source says them."""
    if isinstance(container, (ast.List, ast.Tuple, ast.Set)):
        return choice_values(container, {})
    if (
        isinstance(container, (ast.SetComp, ast.ListComp, ast.GeneratorExp))
        and len(container.generators) == 1
    ):
        # {c for c, _ in Model.STATUS_CHOICES}: the first slot of each pair
        gen = container.generators[0]
        if isinstance(container.elt, ast.Name) and not gen.ifs:
            first = (
                gen.target.elts[0]
                if isinstance(gen.target, ast.Tuple) and gen.target.elts
                else gen.target
            )
            if isinstance(first, ast.Name) and first.id == container.elt.id:
                name = dotted(gen.iter)
                if name is not None:
                    model, _, attr = name.partition(".")
                    if model in models and attr in models[model].choice_sets:
                        return list(models[model].choice_sets[attr])
                return choice_values(gen.iter, {}) or None
        return None
    if (
        isinstance(container, ast.Name)
        and depth < MAX_DEPTH
        and pathcond.bindings(func, container.id)
    ):
        # A local container counts where the function binds it once, whole: `states = {c for c, _ in ...}`.
        bound = [b for b in pathcond.bindings(func, container.id) if not b.nested]
        if (
            len(bound) != 1
            or bound[0].value is None
            or bound[0].unpacked
            or bound[0].kind not in (pathcond.BIND_ASSIGN, pathcond.BIND_ANNOTATED)
        ):
            return None
        return _members(bound[0].value, func, settle, models, depth + 1)
    name = dotted(container)
    if name is not None:
        model, _, attr = name.partition(".")
        if model in models and attr in models[model].choice_sets:
            return list(models[model].choice_sets[attr])
    got = settle(container, func)
    return got or None


def _membership(test: ast.expr, name: str, holds: bool) -> Optional[ast.expr]:
    """The container *test* says *name* is in, where *test* evaluating to *holds* says so; else None."""
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        return _membership(test.operand, name, not holds)
    if isinstance(test, ast.BoolOp):
        # A true `and` makes every term true; a false `or` makes every term false.
        if (holds and isinstance(test.op, ast.And)) or (
            not holds and isinstance(test.op, ast.Or)
        ):
            for term in test.values:
                found = _membership(term, name, holds)
                if found is not None:
                    return found
        return None
    if (
        isinstance(test, ast.Compare)
        and len(test.ops) == 1
        and isinstance(test.left, ast.Name)
        and test.left.id == name
    ):
        op, right = test.ops[0], test.comparators[0]
        if (holds and isinstance(op, ast.In)) or (
            not holds and isinstance(op, ast.NotIn)
        ):
            return right
        if (holds and isinstance(op, ast.Eq)) or (
            not holds and isinstance(op, ast.NotEq)
        ):
            return ast.Tuple(elts=[right], ctx=ast.Load())
    return None


def narrowed(
    name: str,
    at: ast.AST,
    func: ast.AST,
    settle: Settle,
    models: dict[str, DjangoModel],
) -> Optional[list[str]]:
    """The values a membership guard above *at* says the local *name* holds there, or None.

    An early exit (``if name not in S: raise``) counts where nothing binds *name* between it and *at*; an
    enclosing ``if name in S:`` counts for its body.  Several guards intersect.
    """
    found: Optional[set[str]] = None
    ordered: list[str] = []

    def take(values_: Optional[list[str]]) -> None:
        nonlocal found, ordered
        if values_ is None:
            return
        found = set(values_) if found is None else found & set(values_)
        ordered = [v for v in (ordered or values_) if v in found]

    statement = _statement_of(at)
    for _implied, exit_if in pathcond._early_exit_ifs(at):
        if pathcond._given_between(func, exit_if, statement):  # type: ignore[arg-type]
            continue
        container = _membership(exit_if.test, name, holds=False)  # type: ignore[attr-defined]
        if container is not None:
            take(_members(container, func, settle, models, 0))
    previous: ast.AST = at
    for ancestor in pathcond.ancestors(at):
        if ancestor is func:
            break
        if isinstance(ancestor, ast.If):
            in_body = previous in ancestor.body
            if in_body or previous in ancestor.orelse:
                rebound = [
                    b
                    for b in pathcond.bindings(func, name)
                    if _between(b.site, ancestor, at)
                ]
                if not rebound:
                    container = _membership(ancestor.test, name, holds=in_body)
                    if container is not None:
                        take(_members(container, func, settle, models, 0))
        previous = ancestor
    return ordered or None


def _statement_of(node: ast.AST) -> ast.AST:
    """The statement *node* is part of."""
    current = node
    while not isinstance(current, ast.stmt):
        parent = getattr(current, "parent", None)
        if parent is None:
            return node
        current = parent
    return current


def _between(site: Optional[ast.AST], guard: ast.AST, at: ast.AST) -> bool:
    line = getattr(site, "lineno", None)
    return line is not None and guard.lineno < line < getattr(at, "lineno", 0)  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- the ladder


def resolve(
    value: Optional[ast.expr],
    at: ast.AST,
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    settle: Settle,
    models: dict[str, DjangoModel],
    *,
    model: Optional[str] = None,
    depth: int = 0,
) -> list[Resolved]:
    """Every value *value*, written at *at*, can give its field (see the module docstring).

    The trees must have parents attached (:func:`bamboo.codemap.panda.pathcond.attach_parents`).
    """
    if value is None:
        return [Resolved("runtime(<objects' own values>)", 2, [])]
    got = settle(value, func)
    if got:
        return [Resolved(v, 1, []) for v in got]
    if isinstance(value, ast.IfExp):
        test = ast.unparse(value.test)
        return [
            r._replace(conditions=[test, *r.conditions])
            for r in resolve(
                value.body, at, func, settle, models, model=model, depth=depth
            )
        ] + [
            r._replace(conditions=[f"not ({test})", *r.conditions])
            for r in resolve(
                value.orelse, at, func, settle, models, model=model, depth=depth
            )
        ]
    if (
        isinstance(value, ast.Call)
        and (dotted(value.func) or "").split(".")[-1] == "F"
        and value.args
        and isinstance(value.args[0], ast.Constant)
    ):
        return [Resolved(f"passthrough({model or '?'}.{value.args[0].value})", 2, [])]
    if isinstance(value, ast.Name):
        members = narrowed(value.id, at, func, settle, models)
        if members:
            return [Resolved(v, 1, [f"{value.id} == {v!r}"]) for v in members]
        if depth < MAX_DEPTH and (
            pathcond.bindings(func, value.id) or not _is_parameter(func, value.id)
        ):
            held = pathcond.path_condition(at) + pathcond.exits_above(at)
            out: list[Resolved] = []
            reached = pathcond.reaching_values(func, value.id, settle, at)  # type: ignore[arg-type]
            for r in reached:
                conditions = pathcond.binding_tests(r.conditions, held, r.site, at)
                if r.values:
                    out.extend(
                        Resolved(v, 1, list(conditions), r.line) for v in r.values
                    )
                elif r.expression is not None:
                    site = r.site if r.site is not None else at
                    for sub in resolve(
                        r.expression,
                        site,
                        func,
                        settle,
                        models,
                        model=model,
                        depth=depth + 1,
                    ):
                        out.append(
                            sub._replace(
                                conditions=[*conditions, *sub.conditions],
                                line=sub.line or r.line,
                            )
                        )
                else:
                    out.append(
                        Resolved(f"runtime({value.id})", 2, list(conditions), r.line)
                    )
            if out:
                return out
    return [Resolved(f"runtime({ast.unparse(value)[:80]})", 2, [])]
