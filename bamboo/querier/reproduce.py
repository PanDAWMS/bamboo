"""Run the one function that raised, with the inputs this investigation rebuilt (a library for rerun.py).

Importing a server module is not generally possible -- ``adder_atlas_plugin``
reads the server's config file at import -- and it is not needed: what raised
is one function, and its source is in the pinned corpus.  Its ``FunctionDef``
alone is compiled, under the corpus file's name so its line numbers are the
real ones, in a namespace with no module globals.  Annotations are dropped,
because they name types the namespace does not have.

**Inputs.**  A parameter the walk traced back to the caller is rebuilt there
(``false_guard.Walk.build``); every other name -- the rest of the parameters,
``self``, a module global -- is an *opaque* value.  An opaque value is
hashable and equal only to itself; using it in any other way (an attribute,
a call, iteration, truth, formatting) stops the run with the name it stands
for.  A comparison with it is answered as "different" and listed as an
assumption, since the run cannot know it.

**Verdict.**  Reproduced when the same exception type is raised at the same
line with the same message.  Stopped on an opaque value is not a failure of
the reading, only a limit of what was rebuilt; any other outcome says the
reading does not account for what the log shows.

The child runs in its own process with the network refused, without
``open``/``__import__``/``exec``/``eval``, and under a time limit.
"""
from __future__ import annotations

import ast
import builtins
import copy
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TIMEOUT = 30
UNSAFE = {"open", "__import__", "exec", "eval", "compile", "input", "breakpoint", "exit", "quit"}


@dataclass
class Opaque:
    """A value this investigation does not have, standing in for *label*."""

    label: str


@dataclass
class Outcome:
    raised: str = ""  # exception type name
    message: str = ""  # str(exception)
    line: int = 0  # the line of the corpus function it came from
    needed: str = ""  # the opaque value the run stopped on, and how it was used
    returned: str = ""
    assumptions: list[str] = field(default_factory=list)
    error: str = ""  # the child itself failed

    def describe(self) -> str:
        if self.error:
            return f"the run itself failed: {self.error}"
        if self.needed:
            return f"stopped at line {self.line}: needs {self.needed}"
        if self.raised:
            return f"raised {self.raised}: {self.message} at line {self.line}"
        return f"returned {self.returned}"


# ---------------------------------------------------------------- the parent side


def encode(value: Any) -> Any:
    if isinstance(value, Opaque):
        return {"__opaque__": value.label}
    if isinstance(value, dict):
        return {"__dict__": [[encode(k), encode(v)] for k, v in value.items()]}
    if isinstance(value, (set, frozenset)):
        return {"__set__": [encode(v) for v in value]}
    if isinstance(value, (list, tuple)):
        return [encode(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {"__opaque__": f"a {type(value).__name__} value"}


def run(corpus: str, rel: str, func: str, args: dict[str, Any], line: int | None = None) -> Outcome:
    """Call *func* of *rel* with *args* (parameter name -> value; missing ones are opaque) in a child process.

    *line* is a line of the def, for a name its file defines more than once."""
    from bamboo.querier import rows as R

    node = R.def_at(rel, func, [line] if line is not None else [])
    params = [a.arg for a in [*node.args.posonlyargs, *node.args.args]]
    spec = {
        "path": str(Path(corpus) / rel),
        "rel": rel,
        "func": func,
        "line": node.lineno,
        "args": [encode(args.get(p, Opaque(p))) for p in params],
    }
    try:
        done = subprocess.run(
            [sys.executable, __file__, "--child"], input=json.dumps(spec), capture_output=True, text=True, timeout=TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return Outcome(error=f"no answer in {TIMEOUT}s")
    if done.returncode != 0 or not done.stdout.strip():
        return Outcome(error=(done.stderr.strip().splitlines() or ["no output"])[-1])
    return Outcome(**json.loads(done.stdout))


def _stripped(node: ast.FunctionDef) -> ast.FunctionDef:
    node = copy.deepcopy(node)
    node.decorator_list = []
    node.returns = None
    for a in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs, node.args.vararg, node.args.kwarg]:
        if a is not None:
            a.annotation = None

    class Strip(ast.NodeTransformer):
        def visit_AnnAssign(self, n: ast.AnnAssign):
            if n.value is None:
                return ast.copy_location(ast.Pass(), n)
            return ast.copy_location(ast.Assign(targets=[n.target], value=n.value), n)

    return ast.fix_missing_locations(Strip().visit(node))


# ---------------------------------------------------------------- the child side


class Needed(Exception):
    pass


def _child() -> None:
    import socket

    def refused(*a, **k):
        raise Needed("the network (refused in a reproduction)")

    socket.socket.connect = refused  # type: ignore[method-assign]
    socket.create_connection = refused  # type: ignore[assignment]

    spec = json.loads(sys.stdin.read())
    filename = spec["rel"]
    assumptions: list[str] = []

    class Stand:
        """An opaque value in the child."""

        def __init__(self, label: str) -> None:
            object.__setattr__(self, "_label", label)

        def _use(self, how: str):
            raise Needed(f"{object.__getattribute__(self, '_label')} ({how})")

        def __getattr__(self, name):
            self._use(f".{name}")

        def __setattr__(self, name, value):
            self._use(f".{name} =")

        def __eq__(self, other):
            if other is self:
                return True
            assumptions.append(f"{object.__getattribute__(self, '_label')} != {other!r}")
            return False

        def __ne__(self, other):
            return not self.__eq__(other)

        def __hash__(self):
            return id(self)

        def __repr__(self):
            return f"<{object.__getattribute__(self, '_label')}>"

    for dunder in ("__bool__", "__iter__", "__len__", "__call__", "__getitem__", "__setitem__", "__contains__",
                   "__str__", "__format__", "__lt__", "__le__", "__gt__", "__ge__", "__add__", "__int__", "__float__"):
        setattr(Stand, dunder, (lambda d: lambda self, *a, **k: self._use(d))(dunder))

    def decode(v):
        if isinstance(v, dict):
            if "__opaque__" in v:
                return Stand(v["__opaque__"])
            if "__dict__" in v:
                return {decode(k): decode(x) for k, x in v["__dict__"]}
            if "__set__" in v:
                return {decode(x) for x in v["__set__"]}
        if isinstance(v, list):
            return [decode(x) for x in v]
        return v

    safe = {k: v for k, v in vars(builtins).items() if k not in UNSAFE}
    # the definition from the corpus file's own AST, so the line numbers are the real ones
    tree = ast.parse(Path(spec["path"]).read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                and n.name == spec["func"] and n.lineno == spec["line"])
    module = ast.Module(body=[_stripped(node)], type_ignores=[])
    namespace: dict[str, Any] = {"__builtins__": safe, "__name__": "reproduction"}

    class Globals(dict):
        # a dict subclass as globals is asked before the builtins are, so they are looked up here
        def __missing__(self, key):
            if key in safe:
                return safe[key]
            if key in UNSAFE:
                raise Needed(f"{key} (not allowed in a reproduction)")
            return Stand(f"the module global {key}")

    namespace = Globals(namespace)
    code = compile(module, filename, "exec")
    exec(code, namespace)  # noqa: S102 -- the corpus function's own definition
    fn = namespace[spec["func"]]
    args = [decode(a) for a in spec["args"]]
    out: dict[str, Any] = {"assumptions": assumptions}
    try:
        value = fn(*args)
        out["returned"] = repr(value)[:300]
    except Needed as e:
        out["needed"] = str(e)
        out["line"] = _line(e, filename)
    except Exception as e:  # noqa: BLE001 -- the exception is the answer
        out["raised"] = type(e).__name__
        out["message"] = str(e)
        out["line"] = _line(e, filename)
    print(json.dumps(out))


def _line(e: BaseException, filename: str) -> int:
    import traceback

    frames = [f for f in traceback.extract_tb(e.__traceback__) if f.filename == filename]
    return frames[-1].lineno if frames else 0


if __name__ == "__main__" and sys.argv[1:] == ["--child"]:
    _child()
