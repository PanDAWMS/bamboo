"""A test's value from what is known: one evaluator for the map and its readers.

The map decides a test with the constants its code fixes --
:func:`bamboo.codemap.panda.pathcond.decided` with the literal a caller
hands a flag parameter -- and a reader of a run decides it with what the run
printed and the rows it read.  The question is one, so the evaluator is one:
the values are the caller's, and each carries where it came from.

* **Three values.**  ``True``, ``False`` and :data:`UNKNOWN`, with ``and`` /
  ``or`` short-circuiting the way Python does and unknown only where it has
  to be: ``taskSpec.status == "running" or useScout`` is ``True`` with the
  status alone.
* **Literals fold.**  ``-1`` is a value, as :func:`ast.literal_eval` reads
  it; a list, tuple or set of known values is the list of them.
* **A comparison Python cannot make is unknown**, not an error:
  ``None in 3`` says nothing of the run.
* **A call is the reader's.**  The map runs no code of the corpus.  A call is
  looked up by its text, as a printed hole of it would be put
  (``str(self.job.altStgOutFileList())``), unless the reader hands a function
  for its name (:attr:`Env.call`): a reader that has read a helper as pure
  -- the querier's ``DataServiceUtils.getDestinationSE`` -- runs it there.
* **Solving backwards** (:func:`solve`).  A test whose outcome is known with
  some atoms unknown: every assignment of the unknown atoms is tried, and an
  atom that comes out the same in every assignment that reproduces the
  outcome is settled.  A settled ``a == b`` hands the known side's value to
  the other.
"""

from __future__ import annotations

import ast
import itertools
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union


class _Unknown:
    def __repr__(self) -> str:
        return "UNKNOWN"

    def __bool__(self) -> bool:
        raise TypeError("an unknown value has no truth value")


#: What a value or a test is where what is known does not say.
UNKNOWN: Any = _Unknown()

#: The comparisons a test is decided by.
_COMPARE: dict[type, Callable[[Any, Any], Any]] = {
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
    ast.Is: lambda a, b: a is b,
    ast.IsNot: lambda a, b: a is not b,
}


@dataclass
class Value:
    """One known value and where it came from."""

    value: Any
    source: str


@dataclass
class Env:
    """What is known at one decision, each value with its source.

    A reader with more to look up -- the rows a run read, under the names
    its question binds -- extends :meth:`canonical` and :meth:`lookup`.
    """

    values: dict[str, Value] = field(default_factory=dict)
    #: expressions known not to hold what the test saw, with why
    stale: dict[str, str] = field(default_factory=dict)
    #: each expression a value was taken for, with its source (or why it was unknown)
    used: dict[str, str] = field(default_factory=dict)
    #: the function to run for a call's name, or None to look the call up by its text
    call: Optional[Callable[[str], Optional[Callable[..., Any]]]] = None

    def canonical(self, text: str) -> str:
        """The key *text* is held under."""
        return text

    def put(self, text: str, value: Any, source: str) -> None:
        self.values[self.canonical(text)] = Value(value, source)

    def put_printed(self, text: str, printed: str, source: str) -> None:
        """A value read out of a log line, as Python reads its repr where it can.

        A hole ``{str(x)}`` that reads back as a literal other than a string
        says what ``x`` was, too (``str(self.job.altStgOutFileList())``).
        """
        try:
            value = ast.literal_eval(printed)
        except (ValueError, SyntaxError):
            value = printed
        self.put(text, value, source)
        inner = re.fullmatch(r"str\((.+)\)", text)
        if (
            inner
            and not isinstance(value, str)
            and self.canonical(inner[1]) not in self.values
        ):
            self.put(inner[1], value, f"{source}, printed as str()")

    def lookup(self, text: str) -> Any:
        key = self.canonical(text)
        for stale in (text, key):
            if stale in self.stale:
                self.used[text] = f"unknown: {self.stale[stale]}"
                return UNKNOWN
        if key in self.values:
            self.used[text] = self.values[key].source
            return self.values[key].value
        return UNKNOWN


def strip_annotation(text: str) -> str:
    """Drop the ``  [name := expr]`` a path condition carries for a bare local."""
    return re.sub(r"\s+\[\w+ := .*\]$", "", text)


def read_elsewhere(text: str) -> bool:
    """Whether a path condition is read where the function's passes cannot say it: ``  [in f()]``, tested in a
    method the function calls, its names the callee's; ``  [at N]``, a later write's test, its names as they are
    when line N's test runs, after the branch's write."""
    return re.search(r"\s+\[(in \w+\(\)|at \d+)\]$", text) is not None


def text_of(node: ast.AST) -> str:
    """``ast.unparse`` of *node*, kept on it: a test is read again for each set of values, and unparsing it was most
    of :func:`pathcond.decided`'s time."""
    held = getattr(node, "_evaluate_text", None)
    if held is None:
        held = ast.unparse(node)
        node._evaluate_text = held  # type: ignore[attr-defined]
    return held


def _call(node: ast.Call, env: Env) -> Any:
    name = text_of(node.func)
    run = env.call(name) if env.call is not None else None
    if run is None:
        return env.lookup(text_of(node))
    args = [ev(one, env) for one in node.args]
    kwargs = {one.arg: ev(one.value, env) for one in node.keywords if one.arg}
    if any(one is UNKNOWN for one in [*args, *kwargs.values()]):
        return UNKNOWN
    result = run(*args, **kwargs)
    env.used[text_of(node)] = f"called {name}"
    return result


def _literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)  # type: ignore[arg-type]
    except (ValueError, SyntaxError, TypeError):
        return UNKNOWN


def ev(node: ast.AST, env: Env) -> Any:
    """What *node* holds by what *env* knows, or :data:`UNKNOWN`."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and not isinstance(node.op, ast.Not):
        folded = _literal(node)
        if folded is not UNKNOWN:
            return folded
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        items = [ev(one, env) for one in node.elts]
        return UNKNOWN if any(one is UNKNOWN for one in items) else items
    if isinstance(node, ast.BoolOp):
        both = isinstance(node.op, ast.And)
        pending = False
        for one in node.values:
            value = ev(one, env)
            if value is UNKNOWN:
                pending = True
                continue
            if both and not value:
                return value
            if not both and value:
                return value
        return UNKNOWN if pending else both
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        value = ev(node.operand, env)
        return UNKNOWN if value is UNKNOWN else (not value)
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        compare = _COMPARE.get(type(node.ops[0]))
        left, right = ev(node.left, env), ev(node.comparators[0], env)
        if compare is None or left is UNKNOWN or right is UNKNOWN:
            return UNKNOWN
        try:
            return compare(left, right)
        except TypeError:
            return UNKNOWN
    if isinstance(node, ast.Call):
        return _call(node, env)
    if isinstance(node, (ast.Name, ast.Attribute, ast.Subscript)):
        return env.lookup(text_of(node))
    return UNKNOWN


def evaluate(test: Union[str, ast.AST], env: Env) -> Any:
    """:func:`ev` of a test, given as source (a path condition's text, its annotation dropped) or as a node."""
    node = (
        ast.parse(strip_annotation(test), mode="eval").body
        if isinstance(test, str)
        else test
    )
    return ev(node, env)


# -- solving backwards ------------------------------------------------------- #


def atoms(node: ast.AST) -> list[ast.AST]:
    """The atoms of a test: what ``and``, ``or`` and ``not`` combine."""
    if isinstance(node, ast.BoolOp):
        return [one for value in node.values for one in atoms(value)]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return atoms(node.operand)
    return [node]


def truth(node: ast.AST, fixed: dict[str, bool], env: Env) -> Any:
    """Whether *node* holds, its atoms read from *fixed* (by their text) before *env*; :data:`UNKNOWN` where
    neither decides it."""
    text = text_of(node)
    if text in fixed:
        return fixed[text]
    if isinstance(node, ast.BoolOp):
        values = [truth(one, fixed, env) for one in node.values]
        stop = isinstance(node.op, ast.Or)
        if any(value is stop for value in values):
            return stop
        return UNKNOWN if any(value is UNKNOWN for value in values) else (not stop)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        value = truth(node.operand, fixed, env)
        return UNKNOWN if value is UNKNOWN else (not value)
    value = ev(node, env)
    if value is UNKNOWN:
        return UNKNOWN
    try:
        return bool(value)
    except TypeError:
        return UNKNOWN


@dataclass
class Solved:
    """What a test's known outcome says of its atoms (:func:`solve`)."""

    test: str
    outcome: bool
    #: atom -> its truth by what was known
    known: dict[str, bool]
    #: atom -> its truth in every assignment of the unknown atoms that gives the outcome
    settled: dict[str, bool]
    #: the unknown atoms the outcome leaves open
    open: list[str]
    #: expression -> (its value, why): the other side of a settled equality
    derived: dict[str, tuple[Any, str]]
    #: whether any assignment gives the outcome
    consistent: bool


def solve(test: str, outcome: bool, env: Env) -> Solved:
    """What *test* coming out *outcome* says of its unknown atoms."""
    tree = ast.parse(strip_annotation(test), mode="eval").body
    found: dict[str, ast.AST] = {}
    for atom in atoms(tree):
        found.setdefault(ast.unparse(atom), atom)
    known: dict[str, bool] = {}
    unknown: list[str] = []
    for text, atom in found.items():
        value = truth(atom, {}, env)
        if value is UNKNOWN:
            unknown.append(text)
        else:
            known[text] = value
    fits = []
    for values in itertools.product((True, False), repeat=len(unknown)):
        fixed = {**known, **dict(zip(unknown, values, strict=True))}
        if truth(tree, fixed, env) is outcome:
            fits.append(dict(zip(unknown, values, strict=True)))
    settled = {
        one: fits[0][one]
        for one in unknown
        if fits and all(fit[one] == fits[0][one] for fit in fits)
    }
    derived: dict[str, tuple[Any, str]] = {}
    for text, held in settled.items():
        atom = found[text]
        if not (isinstance(atom, ast.Compare) and len(atom.ops) == 1):
            continue
        operator = atom.ops[0]
        if not (
            (isinstance(operator, ast.Eq) and held)
            or (isinstance(operator, ast.NotEq) and not held)
        ):
            continue
        left, right = atom.left, atom.comparators[0]
        left_value, right_value = ev(left, env), ev(right, env)
        if left_value is UNKNOWN and right_value is not UNKNOWN:
            derived[ast.unparse(left)] = (
                right_value,
                f"{text} is {held} and {ast.unparse(right)} = {right_value!r}",
            )
        elif right_value is UNKNOWN and left_value is not UNKNOWN:
            derived[ast.unparse(right)] = (
                left_value,
                f"{text} is {held} and {ast.unparse(left)} = {left_value!r}",
            )
    return Solved(
        test,
        outcome,
        known,
        settled,
        [one for one in unknown if one not in settled],
        derived,
        bool(fits),
    )
