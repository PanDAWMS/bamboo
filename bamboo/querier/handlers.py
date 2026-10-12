"""Where a path that leaves by an exception would show in the log: the handler that catches it.

A walk (:mod:`bamboo.querier.printed`) leaves a path at every statement that may raise
(``call at N``, ``raise at N``, ``call in try at N`` ...).  Whether the log can
say that path happened depends on what catches the exception inside the same
function -- the innermost ``try`` whose body holds line N and that has a
handler catching anything (bare, ``Exception``, ``BaseException``):

* ``TB`` -- the handler prints a traceback (``dump_error_message``,
  ``traceback.format_exc``/``print_exc``, ``logger.exception``,
  ``exc_info=``) through the run's logger: its frame names line N;
* ``LOG`` -- it prints a row without one through the run's logger: the row says
  an exception came, not from where;
* ``UNTAGGED`` -- it prints only through another logger (a module's
  ``logger``): the row carries no instance mark, so no run holds it, and its
  absence from a run says nothing (TaskRefiner, JobGenerator);
* ``SILENT`` -- it prints nothing;
* ``OUT`` -- no such ``try``, or its handler only raises again: the exception
  leaves the function.  With ``P_FOLLOW_OUT`` (the default) it is followed to
  the callers (:mod:`bamboo.querier.callers`): ``TB``, with no handler in this function,
  when every caller prints its traceback into this function's run;
* ``NOTEXC`` -- the path left by ``return``, a jump, or fell through.

A handler that prints only under an ``if`` and otherwise raises again
(``if self.is_deadlock_exception(e): tmpLog.warning(...); continue`` then
``raise``) has two ways out, so its class is the weaker of its own and the
next catching handler's outwards (:data:`STRENGTH`): both must print for the
log to decide.  ``P_CONDITIONAL_HANDLERS=0`` makes it its own, LOG, whatever
is outside.

The run's loggers (:func:`run_loggers`) are the receivers of the function's
own rows that are its parameters or names it binds -- ``tmp_log =
self.create_tagged_logger(...)``, ``tmpLog = MsgWrapper(...)`` -- the loggers
that put the token a run is made of (:mod:`bamboo.querier.records`), and
an instance attribute its class binds once to a wrapper that puts
a mark (:func:`attribute_loggers`: AdderGen's and Watcher's ``self.logger``;
``P_ATTRIBUTE_LOGGERS=0`` leaves them out).  By name, not by binding.  Of the 75 functions holding census_p's 184 arms, one binds a run
logger's name twice:
JobGenerator.runImpl, ``tmpLog = MsgWrapper(self.logger, monToken=prefix)`` (no
token) at 601 and the tagged ``MsgWrapper(self.logger, f"<jediTaskID=...>", ...)``
at 621.  601 is followed by ``return`` at 604, so its handlers (690, 746, 773)
see only 621's.  (TaskRefiner's untagged ``tmpLog`` at 58 is in ``start``, not
in ``runImpl``, which binds it once, at 137.)

Both rules together changed the class of 8 handlers and moved no census line.

Of the no-anchor arm paths a census found, 64% of the exception leaves are ``TB`` (before
``UNTAGGED`` was split off).
"""
from __future__ import annotations

import ast
import functools
import os
import re

from bamboo.codemap.panda import pathcond

from . import rows as R

CATCH_ALL = {"Exception", "BaseException"}
TRACEBACK_CALLS = {"dump_error_message", "format_exc", "print_exc", "format_exception", "print_exception", "exception"}
LEAVE = re.compile(r"(call|raise|jump|return)(?: in (\w+))? at (\d+)")
WRAPPERS = {"LogWrapper", "MsgWrapper"}
#: how much a class says about a path that left into it; a conditional handler takes the weaker of two
STRENGTH = {"TB": 3, "LOG": 2, "UNTAGGED": 1, "SILENT": 0, "OUT": 0}
#: P_ATTRIBUTE_LOGGERS=0 leaves instance attribute loggers out of
#: run_loggers, P_CONDITIONAL_HANDLERS=0 classes a handler that prints only under an if by itself, as before
ATTRIBUTE_LOGGERS = os.environ.get("P_ATTRIBUTE_LOGGERS", "1") != "0"
CONDITIONAL_HANDLERS = os.environ.get("P_CONDITIONAL_HANDLERS", "1") != "0"
#: P_FOLLOW_OUT=1 (the default) classes an exception that leaves
#: the function TB when every caller prints its traceback into the function's run (:func:`bamboo.querier.callers.out_class`)
FOLLOW_OUT = os.environ.get("P_FOLLOW_OUT", "1") != "0"


def catches_all(h: ast.ExceptHandler) -> bool:
    if h.type is None:
        return True
    names = h.type.elts if isinstance(h.type, ast.Tuple) else [h.type]
    return any(
        (isinstance(n, ast.Name) and n.id in CATCH_ALL) or (isinstance(n, ast.Attribute) and n.attr in CATCH_ALL) for n in names
    )


@functools.lru_cache(maxsize=None)
def run_loggers(func: ast.AST) -> frozenset[str]:
    """Receivers of *func*'s own rows that are its parameters or names it binds, and :func:`attribute_loggers`."""
    args = func.args if isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)) else None
    params = {a.arg for a in [*args.posonlyargs, *args.args, *args.kwonlyargs]} if args else set()
    bound = {t.id for n in R.own_nodes(func) if isinstance(n, (ast.Assign, ast.AnnAssign)) for t in _targets(n) if isinstance(t, ast.Name)}
    out = set()
    for n in R.own_nodes(func):
        if R.is_print(n) and isinstance(n.value.func.value, ast.Name) and n.value.func.value.id in params | bound:
            out.add(n.value.func.value.id)
    if ATTRIBUTE_LOGGERS:
        out |= attribute_loggers(func)
    return frozenset(out)


def _self_attr(e: ast.expr) -> str:
    return ast.unparse(e) if isinstance(e, ast.Attribute) and isinstance(e.value, ast.Name) and e.value.id == "self" else ""


def puts_mark(v: ast.expr) -> bool:
    """*v* is a LogWrapper/MsgWrapper that puts a mark on every row: a token, or a stamp (LogWrapper's prefix
    ``None``, MsgWrapper's token ``None`` or absent).  ``LogWrapper(x)`` has prefix ``""``: no mark."""
    if not isinstance(v, ast.Call):
        return False
    f = v.func
    name = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else "")
    if name not in WRAPPERS:
        return False
    second = v.args[1] if len(v.args) > 1 else next((k.value for k in v.keywords if k.arg in ("prefix", "token")), None)
    if second is None:
        return name == "MsgWrapper"
    return not (isinstance(second, ast.Constant) and second.value == "")


@functools.lru_cache(maxsize=None)
def _class_bindings(cls: ast.ClassDef) -> dict[str, list[ast.expr]]:
    out: dict[str, list[ast.expr]] = {}
    for n in ast.walk(cls):
        if isinstance(n, (ast.Assign, ast.AnnAssign)) and n.value is not None:
            for t in _targets(n):
                if _self_attr(t):
                    out.setdefault(_self_attr(t), []).append(n.value)
    return out


def attribute_loggers(func: ast.AST) -> frozenset[str]:
    """``self.X`` receivers of *func*'s own rows that its class binds once, to a wrapper that puts a mark
    (:func:`puts_mark`): every row of the object carries it, so the object's rows are one run (AdderGen's
    ``self.logger = LogWrapper(_logger, str(self.job_id))``).  A binding in a base class elsewhere is not followed."""
    cls = next((a for a in pathcond.ancestors(func) if isinstance(a, ast.ClassDef)), None)
    if cls is None:
        return frozenset()
    out = set()
    for n in R.own_nodes(func):
        if R.is_print(n) and _self_attr(n.value.func.value):
            name = _self_attr(n.value.func.value)
            values = _class_bindings(cls).get(name, [])
            if len(values) == 1 and puts_mark(values[0]):
                out.add(name)
    return frozenset(out)


def _targets(n: ast.AST) -> list[ast.expr]:
    ts = n.targets if isinstance(n, ast.Assign) else [n.target]
    return [e for t in ts for e in (t.elts if isinstance(t, ast.Tuple) else [t])]


def _logger_of(c: ast.Call) -> str:
    """The logger a printing call writes through: a log call's receiver, ``dump_error_message``'s argument."""
    f = c.func
    name = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else "")
    if name == "dump_error_message":
        return ast.unparse(c.args[0]) if c.args else ""
    if R.is_log_call(c) or name == "exception":
        return ast.unparse(f.value) if isinstance(f, ast.Attribute) else ""
    return ""


def handler_class(h: ast.ExceptHandler, loggers: frozenset[str] = frozenset()) -> str:
    nodes = [n for s in h.body for n in R.own_nodes(s)]
    calls = [n for n in nodes if isinstance(n, ast.Call)]

    def name(c: ast.Call) -> str:
        return c.func.attr if isinstance(c.func, ast.Attribute) else (c.func.id if isinstance(c.func, ast.Name) else "")

    printing = [c for c in calls if _logger_of(c)]
    ours = [c for c in printing if _logger_of(c) in loggers]
    traceback = any(name(c) in TRACEBACK_CALLS for c in calls) or any(k.arg == "exc_info" for c in calls for k in c.keywords)
    if ours:
        return "TB" if traceback else "LOG"
    if printing:
        return "UNTAGGED"
    if any(isinstance(n, ast.Raise) for n in nodes):
        return "RERAISE"
    return "SILENT"


def leave_class(func: ast.AST, left: str) -> tuple[str, int | None]:
    """``(class, line)`` of a walk's leave text (``Path.left``)."""
    cls, line, _, _ = leave(func, left)
    return cls, line


def leave(func: ast.AST, left: str) -> tuple[str, int | None, ast.ExceptHandler | None, ast.stmt | None]:
    """``(class, line, the handler, the statement that raises)`` of a walk's leave text."""
    m = LEAVE.match(left or "")
    if not m:
        return "NOTEXC", None, None, None
    kind, inside, line = m[1], m[2], int(m[3])
    if kind in ("return", "jump"):
        return "NOTEXC", line, None, None
    stmt = R.statement_at(func, line)
    if stmt is None:
        return "?", line, None, None
    chain = [stmt, *pathcond.ancestors(stmt)]
    # "call in try at N": something in the try at N raised
    trys = [stmt] if inside == "try" and isinstance(stmt, ast.Try) else []
    for child, parent in zip(chain, chain[1:], strict=False):
        if parent is func:
            break
        if isinstance(parent, ast.Try) and any(child is b for b in parent.body):
            trys.append(parent)
    loggers = run_loggers(func)
    held: str | None = None  # the class of a conditional handler passed on the way out
    for t in trys:
        for h in t.handlers:
            if catches_all(h):
                c = handler_class(h, loggers)
                if c == "RERAISE":
                    break  # out of this try; the next one outwards decides
                if CONDITIONAL_HANDLERS and c in ("TB", "LOG") and conditional(h, loggers):
                    held = c if held is None else min(held, c, key=STRENGTH.get)
                    break  # its other way raises again: the next one outwards decides that way
                if held is not None and STRENGTH[held] < STRENGTH[c]:
                    # both ways must print for the log to decide; a LOG carries no handler whose rows could be
                    # read as missing, since the conditional way does not print them
                    return held, line, None if held == "LOG" else h, stmt
                return c, line, h, stmt
    c = "OUT"
    if FOLLOW_OUT:
        from . import callers

        c = callers.out_class(func)  # TB: every caller prints its traceback into this run (no handler here)
    if held is not None and STRENGTH[held] < STRENGTH[c]:
        return held, line, None, stmt
    return c, line, None, stmt


def conditional(h: ast.ExceptHandler, loggers: frozenset[str]) -> bool:
    """*h* prints through a run logger only under an ``if``, and raises again straight in its body: ``except
    Exception as e: if self.is_deadlock_exception(e): tmpLog.warning(...); continue`` then ``raise``."""
    top_raise = any(isinstance(s, ast.Raise) for s in h.body)
    top_print = any(
        isinstance(n, ast.Call) and _logger_of(n) in loggers
        for s in h.body if not isinstance(s, ast.If) for n in R.own_nodes(s)
    )
    return top_raise and not top_print


def after_arm(func: ast.AST, arm: ast.stmt, line: int) -> bool:
    """Reaching *line* passes *arm*: it is *arm*, or inside a later statement of *arm*'s own block."""
    target = R.statement_at(func, line)
    if target is None:
        return False
    if target is arm:
        return True
    parent = next(iter(pathcond.ancestors(arm)), None)
    found = R.block_of(parent, arm) if parent is not None else None
    if not found:
        return False
    block = found[1]
    at = next(i for i, s in enumerate(block) if s is arm)
    for e in [target, *pathcond.ancestors(target)]:
        for i, s in enumerate(block):
            if s is e:
                return i > at
    return False
