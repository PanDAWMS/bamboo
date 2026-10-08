"""Why one arm ran, walked from the source at the moment it is asked.

**What the map cannot answer and this can.**  The map is a reverse index: it
turns a value into the arms that can write it, an arm into the log that would
mention it, and a field into whoever reads it -- facts that need the whole
corpus and cannot be reached by walking forward from anywhere.  Everything
*reachable* from an anchor is a different job, and doing it here rather than
in the build is the point:

  A fact you can reach forward from the anchor belongs to the trace.  Only
  the facts you cannot reach belong to the map.

**Cadence, not analysis, is what went wrong the last time.**  The same walk was
built into ``build-map`` once and reverted: it cost half the build again, it
made one extra name visible in production, and four of its nine rejection
rules only appeared once it was running.  The cause was storage, not the
reading -- a stored branch has to fold its reasons into one condition string,
and that is where the loop rule, the length cap and the recursion rule all came
from.  At use time an answer may be a **set**, so a flag set in four places is
four steps because four of them are the answer, and none of those rules is
needed.

**Three components, because one arm is rarely explained by one of them.**  A
guard (71% of the map's arms have an ``if`` chain), where the value came from
(44% have a right-hand side that is not a literal), and what the enclosing
function was given.  Guards and values are walked together: a guard's names
are followed exactly like a value's, which is how ``taskSpec.status =
"tobroken"`` -- a constant with no provenance of its own -- reaches the DDM
call whose failure set the flag above it.

**Necessary, not sufficient.**  Only ``ast.If`` contributes to a path
condition, so :func:`pathcond.enclosing_guards` runs beside it and every step
carries what it could not say.  86% of the map's arms have a ``try``, a loop,
a handler or an early exit on the way to them; a step with empty ``guards`` is
one the walk found no ``if`` for, not one that is unconditional.

**Demand-driven is not a closure.**  Measured over the 930 arms of the 481
junctions the installed map can locate: a median of one file parsed and a
maximum of four, a median of one function visited and a maximum of four, and
a median of four steps with 24 at the ninetieth percentile.  478 of the 481
walks finish on their own and three reach the step bound, which they say.
The downward closure of the same corpus reaches 1566 functions at one hop and
90% of the tree at two, so the difference is the whole point of asking only
for what a condition names.

Nothing here is written into the map but what :func:`function_prints` reads,
the print rows no question changes, which the build stores.
"""

from __future__ import annotations

import ast
import builtins
import itertools
import logging
import re
import string
import sys
from collections import deque
from pathlib import Path
from typing import Any, Callable, Iterator, NamedTuple, Optional, Sequence

from bamboo.codemap import calls as calls_mod
from bamboo.codemap.calls import Types as _Types
from bamboo.codemap.gitsource import blob_sha as git_blob_sha
from bamboo.codemap.models import (
    ARRIVES_BY_CALL,
    ARRIVES_BY_DISPATCH,
    ROLE_ITERATION,
    SKELETON_ARM,
    SKELETON_BRANCH,
    SKELETON_PRINT,
    STOP_CONFIG,
    STOP_EXTERNAL,
    STOP_PARAMETER,
    STOP_UPSTREAM,
    TRACE_BINDING,
    TRACE_HANDOVER,
    TRACE_LOOP,
    TRACE_PRINT,
    TRACE_RETURN,
    TRACE_UNBOUND,
    TRACE_WRITE,
    Handover,
    PrintRow,
    PrintStatement,
    SkeletonHole,
    SkeletonLine,
    TraceStep,
)
from bamboo.codemap.panda import pathcond

# ``logging_arguments`` is the build's own spelling of which calls are log
# calls, reused rather than restated: the map's question and the trace's
# prediction have to be about the same calls or they cannot be compared at
# all.  What a call *prints* is read here, by :meth:`_Walk.messages`, and more
# widely than the build's ``_logged_arguments`` reads it -- a message built
# with ``+``, ``%`` or ``.format``, or into an attribute, is written out here
# and is still one hole there.  Two readings of one thing; the build's is the
# narrower, and stored ``Branch.messages`` still come from it.
from bamboo.codemap.panda.recognizers.emit import (
    logging_arguments,
    own_logging_arguments,
)
from bamboo.codemap.panda.recognizers.selection import log_level
from bamboo.codemap.reading import containing_function

logger = logging.getLogger(__name__)

_BUILTINS = frozenset(dir(builtins))


class Budget(NamedTuple):
    """Bounds on the walk, sized well above what the corpus asks for.

    Not a tuning knob: the measurement says the walk stops on its own, so
    these exist to keep a pathological function from turning an investigation
    into a scan.  When one bites it is said out loud -- a silently truncated
    answer is the failure this whole design is built to avoid.

    ``depth`` is set where the walk converges rather than where it is
    comfortable: when eight was chosen, six still cut two of 481 walks short
    and eight cut one, and the terminals the whole corpus reaches were the same
    as six's but for two steps.

    Re-measured since, over the derivation for every term in the vocabulary
    rather than that one sample -- 1567 readings, all of them walked.  A bite
    is counted per reading, so one awkward function accounts for many:
    ``steps`` bites 49 times and ``depth`` 15, over **four functions in all**
    -- ``insertFilesForDataset_JEDI`` (43), ``closer.py::run`` (12),
    ``JobGenerator.py::runImpl`` (5) and ``datasetManager.py::run`` (4).  The
    population is not the old one, so read the shape and not the difference:
    four functions out of 1567 readings is the design working.  The numbers are
    not a tuning knob, and "none are" was written when the corpus was smaller.
    """

    files: int = 12
    functions: int = 40
    steps: int = 120
    depth: int = 8
    #: What the walk may spend beyond the function it was asked about --
    #: inside the calls a value was bound from, and at the calls that passed a
    #: parameter in.  Its own allowance, spent only after the function's own
    #: answer is complete, so going into a call can never cut short what the
    #: walk said about the function before it could.
    crossed_files: int = 24
    crossed_functions: int = 60
    crossed_steps: int = 480
    #: Depth is counted from the arm, and a call adds a hop for the return
    #: and one for each argument: ``scoutVal`` to the percentile behind it is
    #: nine hops, six of them inside ``getScoutJobData_JEDI``.
    crossed_depth: int = 8


DEFAULT_BUDGET = Budget()


def _budgets(names: set[str]) -> str:
    """``depth budget``, ``functions and depth budgets``: in :class:`Budget`'s order, so one set reads one way."""
    order = [one for one in Budget._fields if one in names] + sorted(names - set(Budget._fields))
    if len(order) == 1:
        return f"{order[0]} budget"
    return f"{', '.join(order[:-1])} and {order[-1]} budgets"

#: What a step's terminal is, when the classifier is not supplied.  Injected
#: rather than imported so that the walk stays about Python and the naming of
#: interfaces stays with the map that has them.
Classifier = Callable[..., tuple[str, str]]


def _nothing(_expression: ast.expr, _resolve=None, _typed=None) -> tuple[str, str]:
    return "", ""


#: Terminals the walk records and still walks *into*.  A value another system
#: supplies is answered by that system for a key, and the key is this code's:
#: ``src_site_spec = self.siteMapper.getSite(self.job.computingSite)`` is CRIC's
#: answer about ``job.computingSite``, and stopping at "CRIC" dropped which site
#: was asked -- the one difference between the adder and the setupper deciding
#: the same ``tmp_src_ddm == tmp_dst_ddm`` two ways.  So the step keeps its
#: terminal, and the names its value and its guards read are explained as for
#: any other step.
#:
#: The database's and a configuration row's are not here.  Both are calls into
#: this corpus -- ``getConfigValue`` is a ``SELECT`` on ``ATLAS_PANDA.CONFIG`` --
#: so their arguments flow into statements the map reads, which is the crossing
#: of a call and its arguments, and a different change from this one.
_WALKED_THROUGH = frozenset({STOP_EXTERNAL})


#: Where a binding the walk reports actually lives, when that is not the
#: function the arm is in.  Both are notes on ``TraceStep.unseen`` rather than
#: new kinds: the row is still a binding, and what changes is whether the
#: reader may take it as a step on the path to the arm.
MODULE_SCOPE = "module-scope"
NESTED_SCOPE = "nested-def"


class _Frame(NamedTuple):
    file: str
    owner: str
    func: ast.AST
    handovers: tuple[Handover, ...]
    imports: frozenset[str] = frozenset()


def _with_class(file: str, tree: ast.Module, func: ast.AST) -> str:
    """``file::Class.method``: the owner of *func* told apart from another class's method of its name."""
    cls = pathcond.class_of(tree, func)
    return f"{file}::{cls.name if cls is not None else 'module'}.{getattr(func, 'name', '')}"


def _text(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:  # noqa: BLE001 -- unparse fails on synthesised nodes
        return "..."


def _operator(site: ast.AST) -> str:
    """How a binding's value reaches the name -- ``=``, or an augmented form.

    The step carries the right-hand side alone, so without this a reader is
    shown ``timeOutVal = 60`` for ``timeOutVal *= 60`` and takes sixty for the
    value.  ``ast.unparse`` renders the operator through a throwaway ``BinOp``
    rather than a table, so an operator the grammar gains is not one this has
    to be told about.

    A container filled by :func:`_mutated_bindings` says which method filled
    it.  ``xmlIDs = int(panda_id)`` would be a wrong answer twice over: the
    name is not assigned there, and what it holds is every such value rather
    than only this one.
    """
    if isinstance(site, ast.Expr) and isinstance(site.value, ast.Call):
        called = site.value.func
        if isinstance(called, ast.Attribute) and called.attr in _FILLERS:
            return f".{called.attr}"
    if not isinstance(site, ast.AugAssign):
        return "="
    blank = ast.Name(id="_", ctx=ast.Load())
    rendered = _text(ast.BinOp(left=blank, op=site.op, right=blank))
    return f"{rendered[1:-1].strip()}=" if rendered.startswith("_") else "="


def _add(out: list[str], name: str) -> None:
    if name not in out:
        out.append(name)


def _reads(expression: ast.AST) -> list[str]:
    """The names an expression depends on, in the two forms the walk resolves.

    A bare local (which reaching definitions answer) and ``self.<field>``
    (which only the handover the map recorded can answer).  Builtins are
    dropped: resolving ``len`` would produce an unbound step per call and say
    nothing.
    """
    # ``self.helper(...)`` reads no attribute: the value is what the call
    # returns, and the method's body is a hop downwards that this walk does
    # not take.  Counting it as a field put 30 steps on names like
    # ``self.getClobObj`` and said the caller chose them.
    called = {
        node.func
        for node in ast.walk(expression)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    out: list[str] = []
    for node in ast.walk(expression):
        if isinstance(node, ast.Attribute):
            base: ast.AST = node
            nearest = ""
            depth = 0
            while isinstance(base, ast.Attribute):
                nearest = base.attr
                base = base.value
                depth += 1
            if depth == 1 and node in called:
                continue
            if isinstance(base, ast.Name) and base.id == "self":
                _add(out, f"self.{nearest}")
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id != "self" and node.id not in _BUILTINS:
                _add(out, node.id)
    return out


def _guard_reads(node: ast.AST) -> list[str]:
    """The names the ``if`` tests dominating *node* depend on.

    Walked as well as the value, which is what makes a constant right-hand
    side explainable at all: the reason ``status = "tobroken"`` ran is the
    flag above it, not the string.
    """
    out: list[str] = []
    previous = node
    for ancestor in pathcond.ancestors(node):
        if isinstance(
            ancestor, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)
        ):
            break
        if isinstance(ancestor, ast.If) and (
            previous in ancestor.body or previous in ancestor.orelse
        ):
            for name in _reads(ancestor.test):
                _add(out, name)
        previous = ancestor
    return out


def _guard_tests(node: ast.AST) -> list[tuple[ast.If, list[str]]]:
    """:func:`_guard_reads` with the ``if`` each name is read at.

    Where a walk counts a name's writes from the statement that read it, the
    statement that read a guard's names is the ``if``, not the line under it:
    ``if tmpKey not in keyList: keyList.add(tmpKey)`` read the set before the
    ``add`` below the test ran.
    """
    found: list[tuple[ast.If, list[str]]] = []
    previous = node
    for ancestor in pathcond.ancestors(node):
        if isinstance(
            ancestor, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)
        ):
            break
        if isinstance(ancestor, ast.If) and (
            previous in ancestor.body or previous in ancestor.orelse
        ):
            found.append((ancestor, _reads(ancestor.test)))
        previous = ancestor
    return found


def _unseen(node: ast.AST) -> list[str]:
    return [
        f"{entry.kind} at {entry.line}: {entry.detail}"
        for entry in pathcond.enclosing_guards(node)
    ]


def _statement_at(func: ast.AST, line: int) -> Optional[ast.stmt]:
    """The statement the map's line addresses, innermost first."""
    exact = [
        node
        for node in ast.walk(func)
        if isinstance(node, ast.stmt) and node.lineno == line
    ]
    if exact:
        return min(exact, key=lambda node: (node.end_lineno or node.lineno) - node.lineno)
    holding = [
        node
        for node in ast.walk(func)
        if isinstance(node, ast.stmt)
        and node.lineno <= line <= (node.end_lineno or node.lineno)
    ]
    if not holding:
        return None
    return min(holding, key=lambda node: (node.end_lineno or node.lineno) - node.lineno)


def _walked_key(node: ast.AST) -> tuple[int, int, str]:
    """A binding's identity across two askings of one name: where it is, and what."""
    return (getattr(node, "lineno", 0), getattr(node, "col_offset", 0), type(node).__name__)


def _value_of(statement: ast.stmt) -> Optional[ast.expr]:
    if isinstance(statement, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
        return statement.value
    if isinstance(statement, ast.Expr):
        return statement.value
    if isinstance(statement, ast.Return):
        return statement.value
    return None


def _written_texts(statement: ast.stmt) -> list[str]:
    """How the arm spells what it writes, and what it writes it to.

    Both, because a log line reports either one.  ``taskSpec.status = newStatus``
    is followed by a line interpolating ``newStatus`` as often as by one
    interpolating ``taskSpec.status``, and the two are the same fact said from
    the two ends of the assignment.
    """
    targets: list[ast.expr] = []
    if isinstance(statement, ast.Assign):
        targets = list(statement.targets)
    elif isinstance(statement, (ast.AugAssign, ast.AnnAssign)):
        targets = [statement.target]
    texts = [_text(target) for target in targets]
    value = _value_of(statement)
    if value is not None:
        texts.append(_text(value))
    return [text for text in texts if text and text != "..."]


# How much literal text a pattern needs before it can be asked on its own.
#
# **Below it a pattern is marked, not refused.**  The reader asks every line
# together with the entity's key -- the PandaID a line carries -- and ``Try:``
# is four characters that tell a temporary failure from a fatal one in
# ``register_files``, once the job is fixed.  Refused, the row said nothing at
# all; marked, it says what to grep and that it is only a question with the key.
# A message with no literal text at all is still refused: there is nothing to
# ask with.  What follows is why the mark sits at 14.
#
# How much literal text a pattern needs before it is a question rather than a
# sieve.  Set from two measurements, neither of which found a clean answer.
#
# Put against the production lines already in the evidence file, the patterns
# a lower bar would admit match the *wrong* message at 13 characters and below
# -- ``IO intensity `` finds ``candidates passed max IO intensity check``, and
# ``registering`` finds ``registering <dataset> with location=...``.  From 14
# up, none of the admitted patterns matched a line it did not mean.
#
# The second measurement is the one to be honest about.  Asking how often a
# message's literal run sits inside *another* message in the corpus gives 89%
# at 0-4 characters, 65% at 5-9, 41% at 10-14, 37% at 15-19 -- and **11% at 40
# and above**.  It decays and never reaches zero, so no length makes a pattern
# safe and this guard only removes the worst of them.  A reader who takes a
# long pattern for a reliable one is the next version of this bug.
MIN_LITERAL = 14

# What has to *not* follow a value for the pattern to be about that value.
# ``newPrio=100`` matches ``newPrio=1000``: without a literal after the hole
# there is nothing to say the number ended, and a question that cannot miss is
# read by the eliminator as a question that was answered.
NOT_A_WORD = r"(?![\w.-])"


def _neighbouring_literal(message: ast.JoinedStr, hole: int) -> bool:
    """Whether the hole at *hole* has literal text to anchor on beside it.

    The requirement the map's shared sentence cannot make.  A pattern with no
    literal next to the value is ``.*finished.*``, which matches every line in
    the file that mentions the word -- and a question that cannot miss is read
    by the eliminator as a question that was answered.
    """
    for side in (hole - 1, hole + 1):
        if not 0 <= side < len(message.values):
            continue
        part = message.values[side]
        if isinstance(part, ast.Constant) and isinstance(part.value, str) and part.value.strip():
            return True
    return False


def _literal_text(part: ast.expr) -> str:
    """*part*'s text if it is a literal piece of the message, else ``""``."""
    if isinstance(part, ast.Constant) and isinstance(part.value, str):
        return part.value
    return ""


def literal_length(message: ast.JoinedStr) -> int:
    """How many characters of literal text the message carries.

    What is left once the holes are taken out is the whole of what a pattern
    can anchor on, so it is the only honest measure of whether the pattern is a
    sentence.  Stripped, because a hole sitting between two spaces contributes
    two characters that match anything with a space on either side.
    """
    return sum(len(_literal_text(part).strip()) for part in message.values)


#: Calls whose result prints as a Python ``repr`` -- a dict, a list, an
#: exception's text -- and so may hold spaces, commas and brackets.
_PRINTS_TEXT = frozenset(
    {"str", "repr", "sorted", "list", "dict", "set", "tuple", "format_exc", "join", "keys", "values", "items"}
)
#: Calls whose result prints as a number.
_PRINTS_NUMBER = frozenset({"len", "int", "round", "sum", "count"})
#: The type letter a format spec ends with when it prints a number.
_NUMBER_SPEC = frozenset("bcdeEfFgGnoxX%")


def _hole_prints(part: ast.FormattedValue) -> str:
    """How the value in a hole prints: ``"number"``, ``"text"``, or ``""``.

    ``""`` is a token -- a status, an id, a site name -- and the default,
    closed against the character after it as before.  The two others are the
    shapes that closing got wrong: ``registerFilesInDatasets {str(dest_id_map)}
    zip=`` prints a dict with spaces in it, so a hole that stopped at the first
    space never reached ``zip=``, and none of the job's lines matched.
    Read off the syntax only; a name whose value is a dict prints like one too,
    and is left a token rather than guessed at.
    """
    if getattr(part, "appended", False):
        return "text"
    spec = part.format_spec
    if isinstance(spec, ast.JoinedStr) and all(
        isinstance(value, ast.Constant) for value in spec.values
    ):
        written = "".join(str(value.value) for value in spec.values)
        if written and written[-1] in _NUMBER_SPEC:
            return "number"
    if part.conversion == ord("r"):
        return "text"
    value = part.value
    if isinstance(
        value,
        (ast.Dict, ast.List, ast.Set, ast.Tuple, ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp),
    ):
        return "text"
    if isinstance(value, ast.Call):
        called = value.func
        name = called.attr if isinstance(called, ast.Attribute) else getattr(called, "id", "")
        if name in _PRINTS_TEXT:
            return "text"
        if name in _PRINTS_NUMBER:
            return "number"
    return ""


def _bound(message: ast.JoinedStr, hole: int) -> str:
    """A pattern for an unfilled hole, closed against the literal after it.

    ``.*`` is unbounded, and unbounded is how ``set\\ .*=None`` came to match
    ``set task_status=pending oldTask=False with (True, 1) ...``: the run
    crossed the field name, its value and two more fields to reach an ``=None``
    that belonged to nothing in this statement.  The literal after the hole
    says where the hole has to stop, and excluding its first character is what
    stops it there.

    Falls back to ``.*`` with nothing to close against -- a trailing hole, or
    one butted straight against another.

    **Closed by what the hole prints, where the syntax says.**  A hole that
    prints text -- ``str(dest_id_map)``, a list, a ``!r`` -- may hold the very
    character that closes a token, so it runs, lazily, up to the whole literal
    after it; a number prints as one.  See :func:`_hole_prints`.
    """
    following = _literal_text(message.values[hole + 1]) if hole + 1 < len(message.values) else ""
    part = message.values[hole]
    prints = _hole_prints(part) if isinstance(part, ast.FormattedValue) else ""
    if prints == "number":
        return r"[-+]?[\d.]+(?:[eE][-+]?\d+)?%?"
    if prints == "text":
        return ".*?" if following else ".*"
    if following:
        return f"[^{re.escape(following[0])}]*"
    return ".*"


def _pattern(message: ast.JoinedStr, hole: Optional[int], observed: str) -> str:
    """The message as a regular expression, with *hole* filled by *observed*.

    *hole* is ``None`` for the skeleton's own line, where no value is put in
    and the pattern says only *this line was printed here*.  That is the whole
    difference between the two kinds this module now renders, and both go
    through here so that neither can drift into being the looser one.

    Three things keep the result a question that can come back empty.  Every
    other hole is closed against the literal that follows it rather than left
    as ``.*``; a filled hole with no literal after it gets :data:`NOT_A_WORD`,
    so the value has to end where the pattern says it ends; and a message with
    no literal text at all is refused, because there is nothing to ask with.
    One with less than :data:`MIN_LITERAL` characters is rendered, and the row
    carries :func:`keyed` for it.

    Leading and trailing holes are dropped.  The search is unanchored, so a
    run at either end asks for nothing the rest does not already ask for, and
    printing it invites a reader to think it does.
    """
    if literal_length(message) == 0:
        return ""
    pieces: list[tuple[str, bool]] = []
    for position, part in enumerate(message.values):
        if position == hole:
            pieces.append((re.escape(observed), False))
            after = (
                _literal_text(message.values[position + 1])
                if position + 1 < len(message.values)
                else ""
            )
            if not after:
                pieces.append((NOT_A_WORD, False))
            continue
        literal = _literal_text(part)
        if literal:
            pieces.append((re.escape(literal), False))
        else:
            pieces.append((_bound(message, position), True))
    while pieces and pieces[0][1]:
        pieces.pop(0)
    while pieces and pieces[-1][1]:
        pieces.pop()
    out: list[str] = []
    for text, is_hole in pieces:
        if is_hole and out and out[-1] == text:
            continue
        out.append(text)
    return "".join(out)


def _rendered(message: ast.JoinedStr, hole: int, observed: str) -> str:
    """The value pattern: the observed value put into the hole it fills."""
    return _pattern(message, hole, observed)


def line_pattern(message: ast.JoinedStr) -> str:
    """The line pattern: every hole left open, so it names the line only."""
    return _pattern(message, None, "")


def _enclosing_statement(node: ast.AST) -> Optional[ast.stmt]:
    """The statement *node* sits in, so a call is placed where the source puts it.

    A call's first argument carries the line number the walk has in hand, and
    for a message split over four lines that is not the line the reader sees
    the call on.  The statement is.
    """
    if isinstance(node, ast.stmt):
        return node
    for ancestor in pathcond.ancestors(node):
        if isinstance(ancestor, ast.stmt):
            return ancestor
    return None


def _blocks_of(statement: ast.stmt) -> list[tuple[str, list[ast.stmt]]]:
    """``(header, body)`` for each block a compound statement opens.

    The nesting is the whole reason a skeleton beats a list of lines: two rows
    under one ``if`` were printed together or not at all, and two rows either
    side of an ``else`` cannot both have been.  Returning the headers in source
    order is what lets the reader see that without being told it.

    ``elif`` comes back as ``else:`` wrapping an ``if``, which is what it is.
    A ``match`` comes back as its subject, a block holding nothing of its own,
    and then each ``case`` as a block beside it -- the cases are siblings the
    way an ``if`` and its ``else`` are.  While this returned nothing for one,
    the fifteen lines ``workflow_core.py`` prints inside a ``case`` were in no
    skeleton at all.
    """
    if isinstance(statement, ast.If):
        blocks = [(f"if {_clip(_text(statement.test))}:", statement.body)]
        if statement.orelse:
            blocks.append(("else:", statement.orelse))
        return blocks
    if isinstance(statement, (ast.For, ast.AsyncFor)):
        head = f"for {_clip(_text(statement.target))} in {_clip(_text(statement.iter))}:"
        blocks = [(head, statement.body)]
        if statement.orelse:
            blocks.append(("else:", statement.orelse))
        return blocks
    if isinstance(statement, ast.While):
        blocks = [(f"while {_clip(_text(statement.test))}:", statement.body)]
        if statement.orelse:
            blocks.append(("else:", statement.orelse))
        return blocks
    if isinstance(statement, ast.Try):
        blocks = [("try:", statement.body)]
        for handler in statement.handlers:
            caught = _clip(_text(handler.type)) if handler.type else ""
            blocks.append((f"except {caught}:" if caught else "except:", handler.body))
        if statement.orelse:
            blocks.append(("else:", statement.orelse))
        if statement.finalbody:
            blocks.append(("finally:", statement.finalbody))
        return blocks
    if isinstance(statement, (ast.With, ast.AsyncWith)):
        held = ", ".join(_clip(_text(item.context_expr)) for item in statement.items)
        return [(f"with {held}:", statement.body)]
    if isinstance(statement, ast.Match):
        blocks = [(f"match {_clip(_text(statement.subject))}:", [])]
        for case in statement.cases:
            guard = f" if {_clip(_text(case.guard))}" if case.guard else ""
            blocks.append((f"case {_clip(_text(case.pattern))}{guard}:", case.body))
        return blocks
    return []


def _of_interest(
    statement: ast.stmt,
    printed: dict[int, list[tuple[ast.expr, ast.expr]]],
    armed: dict[int, ast.stmt],
) -> bool:
    """Whether anything under *statement* is a line printed or an arm written.

    The rule that keeps a skeleton a skeleton.  Without it a 540-line function
    comes back whole, and burying the rows a reader came for is the same
    failure this round is fixing in the report's other half.
    """
    for node in ast.walk(statement):
        if isinstance(node, ast.stmt):
            if id(node) in printed or getattr(node, "lineno", 0) in armed:
                return True
    return False


def _constant_pattern(message: ast.expr) -> str:
    """A pattern for a message with no holes in it at all.

    Worth rendering: a constant line says nothing about the value but proves
    the branch it sits in ran, and that is what a reader aligning a region
    needs from the rows around an arm.
    """
    if isinstance(message, ast.Constant) and isinstance(message.value, str):
        text = message.value
        return re.escape(text) if text.strip() else ""
    return ""


def _called_name(call: ast.Call) -> str:
    called = call.func
    return called.attr if isinstance(called, ast.Attribute) else getattr(called, "id", "")


def _holes_of(message: ast.expr) -> list[SkeletonHole]:
    """Every hole of a rendered message, with the expression that fills it."""
    if not isinstance(message, ast.JoinedStr):
        return []
    return [
        SkeletonHole(
            expression=_text(part.value),
            prints=_hole_prints(part) or "token",
            appended=bool(getattr(part, "appended", False)),
        )
        for part in message.values
        if isinstance(part, ast.FormattedValue)
    ]


def keyed(message: ast.expr) -> bool:
    """Whether a rendered *message* is a question only together with the entity's key.

    Below :data:`MIN_LITERAL` characters of literal the pattern matches lines
    it does not mean -- ``Try:`` is in more than one message -- and narrowed to
    one job's lines it is exactly the question: the reader always narrows.
    """
    if isinstance(message, ast.JoinedStr):
        return literal_length(message) < MIN_LITERAL
    if isinstance(message, ast.Constant) and isinstance(message.value, str):
        return len(message.value.strip()) < MIN_LITERAL
    return False


#: How many ways one message may be written out before the rest are left.
_MAX_EXPANDED = 4


def _merged(pieces: list[ast.expr]) -> list[ast.expr]:
    """Adjacent literal pieces as one, so a written-out hole reads as its text."""
    out: list[ast.expr] = []
    for piece in pieces:
        if (
            out
            and isinstance(piece, ast.Constant)
            and isinstance(out[-1], ast.Constant)
            and isinstance(piece.value, str)
            and isinstance(out[-1].value, str)
        ):
            out[-1] = ast.Constant(value=out[-1].value + piece.value)
            continue
        out.append(piece)
    return out


#: Why a printed line has no pattern, one reason per way of failing.  Said
#: rather than left blank: a row that silently loses its pattern reads as a
#: line production does not print, and the reader would go looking for it.
#: The last three are where the message comes from another function, which
#: the walk does not follow yet; they are separate so a count of them is the
#: size of that work.
REFUSED_NO_LITERAL = "no literal text to ask with"
REFUSED_UNRENDERED = "the message is not a literal this walk can render"
REFUSED_HANDED_IN = "the message is handed in by the caller"
REFUSED_SET_ELSEWHERE = "the message is a value set outside this function"
REFUSED_RETURNED = "the message is what a call returns"


def _refusal(message: ast.expr) -> str:
    """Why no pattern was rendered, for a message that reached rendering."""
    if isinstance(message, ast.JoinedStr) or (
        isinstance(message, ast.Constant) and isinstance(message.value, str)
    ):
        return REFUSED_NO_LITERAL
    return REFUSED_UNRENDERED


class _Message(NamedTuple):
    """One way a logging call can print: a message to render, or why not."""

    expr: ast.expr
    refused: str = ""


#: How deep a message is followed through the names it is built from.
_MESSAGE_DEPTH = 4

#: A ``%`` conversion, as ``printf`` spells one.
_PRINTF = re.compile(
    r"%(?:\((?P<key>[^)]*)\))?[-#0 +]*(?P<width>\*|\d+)?(?:\.(?P<precision>\*|\d+))?"
    r"[hlL]?(?P<kind>[diouxXeEfFgGcrsa%])"
)
#: The ``%`` conversions that print a number, spelled as the format spec
#: :func:`_hole_prints` reads.
_PRINTF_NUMBER = {"d": "d", "i": "d", "u": "d", "o": "o", "x": "x", "X": "X",
                  "e": "e", "E": "E", "f": "f", "F": "F", "g": "g", "G": "G"}


def _hole(value: ast.expr, spec: str = "", conversion: int = -1) -> ast.FormattedValue:
    """A hole holding *value*, as an f-string would write it."""
    return ast.FormattedValue(
        value=value,
        conversion=conversion,
        format_spec=ast.JoinedStr(values=[ast.Constant(value=spec)]) if spec else None,
    )


def _appended(name: ast.expr) -> ast.FormattedValue:
    """The hole for what ``+=`` adds to a message after it was first written.

    Text of any shape -- ``traceback.format_exc()`` is the common one -- and
    always at the end, which is where a trailing hole is dropped from the
    pattern: the line starts the way the first binding wrote it.
    """
    hole = _hole(name)
    hole.appended = True  # type: ignore[attr-defined]
    return hole


def _constant_text(pieces: Sequence[ast.expr]) -> Optional[str]:
    """The text of *pieces* when every one is literal, else ``None``."""
    if all(isinstance(p, ast.Constant) and isinstance(p.value, str) for p in pieces):
        return "".join(p.value for p in pieces)  # type: ignore[attr-defined]
    return None


def _has_literal(pieces: Sequence[ast.expr]) -> bool:
    return any(_literal_text(piece).strip() for piece in pieces)


def _is_token(pieces: Sequence[ast.expr]) -> bool:
    """Whether *pieces*, less what ``+=`` adds, are one word with no space in it."""
    own = [piece for piece in pieces if not getattr(piece, "appended", False)]
    return (
        len(own) == 1
        and isinstance(own[0], ast.Constant)
        and isinstance(own[0].value, str)
        and bool(own[0].value)
        and not any(character.isspace() for character in own[0].value)
    )


def _printf_pieces(template: str, values: ast.expr) -> Optional[list[ast.expr]]:
    """``template % values`` as literal pieces and holes, or ``None`` if unreadable."""
    elements = list(values.elts) if isinstance(values, ast.Tuple) else None
    pieces: list[ast.expr] = []
    position = used = 0
    for match in _PRINTF.finditer(template):
        pieces.append(ast.Constant(value=template[position : match.start()]))
        position = match.end()
        kind = match["kind"]
        if kind == "%":
            pieces.append(ast.Constant(value="%"))
            continue
        if match["width"] == "*" or match["precision"] == "*":
            return None
        if match["key"] is not None:
            value: ast.expr = ast.Subscript(value=values, slice=ast.Constant(value=match["key"]))
        elif elements is not None:
            if used >= len(elements):
                return None
            value = elements[used]
            used += 1
        else:
            if used:
                return None
            value = values
            used += 1
        pieces.append(
            _hole(
                value,
                spec=_PRINTF_NUMBER.get(kind, ""),
                conversion=ord(kind) if kind in "ra" else -1,
            )
        )
    pieces.append(ast.Constant(value=template[position:]))
    return pieces


def _format_pieces(template: str, call: ast.Call) -> Optional[list[ast.expr]]:
    """``template.format(...)`` as literal pieces and holes, or ``None`` if unreadable."""
    try:
        parsed = list(string.Formatter().parse(template))
    except ValueError:
        return None
    named = {keyword.arg: keyword.value for keyword in call.keywords if keyword.arg}
    spread = [keyword.value for keyword in call.keywords if keyword.arg is None]
    pieces: list[ast.expr] = []
    automatic = 0
    for literal, field, spec, conversion in parsed:
        if literal:
            pieces.append(ast.Constant(value=literal))
        if field is None:
            continue
        head = re.match(r"[^.\[]*", field).group(0)  # type: ignore[union-attr]
        value: Optional[ast.expr]
        if head == "":
            value = call.args[automatic] if automatic < len(call.args) else None
            automatic += 1
        elif head.isdigit():
            value = call.args[int(head)] if int(head) < len(call.args) else None
        elif head in named:
            value = named[head]
        elif spread:
            value = ast.Subscript(value=spread[0], slice=ast.Constant(value=head))
        else:
            value = None
        if value is None:
            return None
        if field[len(head) :]:
            try:
                value = ast.parse(_text(value) + field[len(head) :], mode="eval").body
            except SyntaxError:
                return None
        pieces.append(
            _hole(
                value,
                spec="" if (spec is None or "{" in spec) else spec,
                conversion=ord(conversion) if conversion else -1,
            )
        )
    return pieces


class _Unknown(NamedTuple):
    """A part of a folded string this reading cannot say: a hole."""

    expression: str


#: Methods of ``str`` a constant can be folded through.
_FOLDABLE = frozenset({"strip", "lstrip", "rstrip", "lower", "upper"})


def _fold(expression: Optional[ast.expr], env: dict[str, list], depth: int = 0) -> Optional[list]:
    """*expression* as a string: literal text, and ``_Unknown`` holes; ``None`` if it is not one.

    Straight arithmetic on strings only -- constants, names in *env*,
    f-strings, ``+``, and ``split(sep)[i]`` or ``strip()`` on a string that is
    all literal.  Enough to say what ``comment.split(" ")[-2].split(".")[-1]``
    is for ``comment = " /* DBProxy.activateJob */"``, and no more.
    """
    if expression is None or depth > 8:
        return None
    if isinstance(expression, ast.Constant):
        return [expression.value] if isinstance(expression.value, str) else None
    if isinstance(expression, ast.Name):
        return env.get(expression.id)
    if isinstance(expression, ast.JoinedStr):
        out: list = []
        for part in expression.values:
            if isinstance(part, ast.Constant):
                out.append(str(part.value))
                continue
            inner = _fold(part.value, env, depth + 1) if part.format_spec is None else None
            out.extend(inner if inner is not None else [_Unknown(_text(part.value))])
        return out
    if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Add):
        left = _fold(expression.left, env, depth + 1)
        right = _fold(expression.right, env, depth + 1)
        return left + right if left is not None and right is not None else None
    index = _integer(expression.slice) if isinstance(expression, ast.Subscript) else None
    if isinstance(expression, ast.Subscript) and index is not None:
        called = expression.value
        if (
            isinstance(called, ast.Call)
            and isinstance(called.func, ast.Attribute)
            and called.func.attr == "split"
        ):
            text = _literal_only(_fold(called.func.value, env, depth + 1))
            separator = _literal_only(_fold(called.args[0], env, depth + 1)) if called.args else None
            if text is None or (called.args and separator is None):
                return None
            pieces = text.split(separator) if called.args else text.split()
            return [pieces[index]] if -len(pieces) <= index < len(pieces) else None
        return None
    if (
        isinstance(expression, ast.Call)
        and isinstance(expression.func, ast.Attribute)
        and expression.func.attr in _FOLDABLE
        and not expression.args
    ):
        text = _literal_only(_fold(expression.func.value, env, depth + 1))
        return [getattr(text, expression.func.attr)()] if text is not None else None
    return None


def _integer(node: ast.AST) -> Optional[int]:
    """An integer written as a constant, ``-2`` included (a unary minus in the tree)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, int)
    ):
        return -node.operand.value
    return None


def _literal_only(pieces: Optional[list]) -> Optional[str]:
    if pieces is None or any(isinstance(piece, _Unknown) for piece in pieces):
        return None
    return "".join(pieces)


def _folded_pattern(pieces: list) -> str:
    """A pattern for folded text: literal escaped, each hole lazy."""
    out = ""
    for piece in pieces:
        if isinstance(piece, _Unknown):
            if not out.endswith(".*?"):
                out += ".*?"
        else:
            out += re.escape(piece)
    return out


def _dominates(site: ast.AST, target: ast.AST) -> bool:
    """Whether statement *site* runs on every path to *target*.

    Read off the nesting: *site* sits in a block that holds *target*, above
    it.  Two branches of an ``if``, a ``try`` body and its handler, are
    different blocks, so a binding in one does not dominate a call in the
    other -- which is the case that matters.
    """
    below = target
    for ancestor in [target, *pathcond.ancestors(target)]:
        for _header, body in _blocks_of(ancestor) if isinstance(ancestor, ast.stmt) else []:
            if below in body and site in body:
                return body.index(site) < body.index(below)
        if isinstance(ancestor, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return site in ancestor.body and below in ancestor.body and (
                ancestor.body.index(site) < ancestor.body.index(below)
            )
        if isinstance(ancestor, ast.stmt):
            below = ancestor
    return False


#: How many of a function's call sites a parameter is followed to.
_MAX_CALLERS = 8
#: How many calls of a method that sets a field are listed.  More than the
#: parameters get: each one is a different place the value was set, and
#: ``set_fatal`` alone is called from nine.
_MAX_MUTATIONS = 24


def _same_node(scope: ast.AST, node: ast.AST) -> Optional[ast.AST]:
    """The node of *scope*'s own tree at *node*'s position, *node* being from another parse."""
    for held in ast.walk(scope):
        if (
            type(held) is type(node)
            and getattr(held, "lineno", None) == getattr(node, "lineno", None)
            and getattr(held, "col_offset", None) == getattr(node, "col_offset", None)
        ):
            return held
    return None


def _keyed_reads(expression: ast.AST) -> list[str]:
    """``x['k']`` for every item of a local read by a constant key in *expression*."""
    found: list[str] = []
    for node in ast.walk(expression):
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, (str, int))
            and isinstance(getattr(node, "ctx", ast.Load()), ast.Load)
        ):
            _add(found, f"{node.value.id}[{node.slice.value!r}]")
    return found


#: How many callers out an ``__init__`` parameter's object is looked for, and
#: how many calls one of them may have before it is not.  Two reaches the
#: knight that builds a worker's list: the worker's ``__init__`` and the knight.
_BUILT_HOPS = 2
_BUILT_CALLERS = 16


class _Narrowed:
    """Calls left out because they were not made on the objects the walk came in on.

    *made* is where those objects were built and *left* the calls not
    followed.  When another object comes in later, *matches* picks the left
    calls that are its and *follow* walks them, as many as *room* allows;
    *step* says what was followed.
    """

    __slots__ = ("made", "left", "matches", "follow", "room", "step", "said")

    def __init__(self, made, left, matches, follow, room, step) -> None:
        self.made: set[tuple[str, int, int]] = set(made)
        self.left: list = left
        self.matches: Callable[[Any, set], bool] = matches
        self.follow: Callable[[list], Any] = follow
        self.room: int = room
        self.step: "TraceStep" = step
        self.said: set[tuple[str, int]] = set()


class _ByClass:
    """Calls placed at an ``__init__`` left out because they cannot build an object of the classes that asked.

    *classes* are the classes whose methods asked a field the ``__init__``
    assigns; *left* the calls not followed.  When a method of another class
    asks later, the left calls that can build its objects are walked by
    *follow*, as many as *room* allows; *step* says so.  See
    :meth:`_Walk._asked_into`.
    """

    __slots__ = ("classes", "left", "follow", "room", "step")

    def __init__(self, classes, left, follow, room, step) -> None:
        self.classes: list[tuple[str, ast.ClassDef]] = list(classes)
        self.left: list = left
        self.follow: Callable[[list], Any] = follow
        self.room: int = room
        self.step: "TraceStep" = step


def _objects_of(classes: list[tuple[str, ast.ClassDef]]) -> str:
    names = " or ".join(sorted({cls.name for _file, cls in classes}))
    return f"an object of {names}, the class{'es' if ' or ' in names else ''} the walk read the field in"


def _site_key(site) -> tuple[str, int, int]:
    return (site.file, site.call.lineno, site.call.col_offset)


def _is_super(expression: ast.AST) -> bool:
    return isinstance(expression, ast.Call) and isinstance(expression.func, ast.Name) and expression.func.id == "super"


def _own_object(func: ast.AST) -> Optional[str]:
    """The parameter a method's own object is in -- ``self`` -- or ``None`` where it takes none."""
    args = getattr(func, "args", None)
    positional = [*args.posonlyargs, *args.args] if args is not None else []
    return positional[0].arg if positional else None


def _built_detail(frame: "_Frame", followed: int, every: int) -> str:
    which = "that built" if getattr(frame.func, "name", "") == "__init__" else "made on"
    return (
        f"a parameter; followed to the {followed} of {every} call(s) placed at "
        f"{getattr(frame.func, 'name', '')}: the ones {which} the object the walk came in on"
    )


def _parameters(func: ast.AST) -> set[str]:
    """*func*'s parameter names, ``*args`` and ``**kwargs`` included; none for a module's own body."""
    args = getattr(func, "args", None)
    if args is None:
        return set()
    names = {one.arg for one in [*args.posonlyargs, *args.args, *args.kwonlyargs]}
    return names | {one.arg for one in (args.vararg, args.kwarg) if one is not None}


def _plain_bindings(func: ast.AST, name: str) -> Optional[list[ast.expr]]:
    """What ``name = ...`` gives *name* in *func*; ``None`` where anything else binds it."""
    values: list[ast.expr] = []

    def names(target: ast.AST) -> bool:
        return any(isinstance(node, ast.Name) and node.id == name for node in ast.walk(target))

    for node in calls_mod.own_nodes(func):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    values.append(node.value)
                elif names(target):
                    return None
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == name and node.value is not None:
                values.append(node.value)
        elif isinstance(node, (ast.AugAssign, ast.For, ast.AsyncFor, ast.comprehension, ast.NamedExpr)):
            if names(node.target):
                return None
        elif isinstance(node, ast.withitem):
            if node.optional_vars is not None and names(node.optional_vars):
                return None
        elif isinstance(node, ast.ExceptHandler):
            if node.name == name:
                return None
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            if name in node.names:
                return None
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            if any((alias.asname or alias.name.split(".")[0]) == name for alias in node.names):
                return None
    return values


def _self_fills(method: ast.AST, field: str) -> list[ast.expr]:
    """What *method* puts into ``self.<field>``: ``append``, ``add`` and the like, and ``[k] =``."""
    found: list[ast.expr] = []
    for node in _own_nodes(method):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Attribute)
                    and isinstance(target.value.value, ast.Name)
                    and target.value.value.id == "self"
                    and target.value.attr == field
                ):
                    found.append(node.value)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _FILLERS
            and isinstance(node.func.value, ast.Attribute)
            and isinstance(node.func.value.value, ast.Name)
            and node.func.value.value.id == "self"
            and node.func.value.attr == field
            and node.args
        ):
            found.append(node.args[-1])
    return found


def _projected(value: ast.expr, key: Optional[object]) -> ast.expr:
    """The part of a returned *value* a caller took: a tuple's slot, a literal dict's key."""
    if key is None:
        return value
    if isinstance(key, int) and isinstance(value, (ast.Tuple, ast.List)) and -len(value.elts) <= key < len(value.elts):
        return value.elts[key]
    if isinstance(value, ast.Dict):
        for held, item in zip(value.keys, value.values, strict=True):
            if isinstance(held, ast.Constant) and held.value == key:
                return item
    return value


def _own_nodes(func: ast.AST):
    """Every node of *func*'s own body; nested functions and classes run elsewhere."""
    for child in ast.iter_child_nodes(func):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        yield child
        yield from _own_nodes(child)


def _message_sites(
    func: ast.AST, spelled: str
) -> tuple[list[ast.expr], list[ast.expr], list[ast.AST]]:
    """``(plain, appended, opaque)`` bindings of the name or place spelled *spelled*.

    *plain* are values assigned outright, *appended* the right-hand sides of
    ``+=``, and *opaque* the sites that bind it to something no text is
    written for here: a loop target, ``with ... as``, ``except ... as``, a
    slot of an unpacked call.
    """
    plain: list[ast.expr] = []
    appended: list[ast.expr] = []
    opaque: list[ast.AST] = []

    def slot(target: ast.expr, value: Optional[ast.expr]) -> None:
        if _text(target) == spelled:
            if value is None:
                opaque.append(target)
            else:
                plain.append(value)
        elif isinstance(target, ast.Starred):
            # ``*rest`` takes however many elements are left, which no
            # element of the value spells.
            slot(target.value, None)
        elif isinstance(target, (ast.Tuple, ast.List)):
            # A slot of a literal tuple is that element; a slot of a call is
            # what the call returns, which is the call's to say.
            if isinstance(value, (ast.Tuple, ast.List)) and len(value.elts) == len(target.elts):
                elements: list[Optional[ast.expr]] = list(value.elts)
            elif isinstance(value, ast.Call):
                elements = [value] * len(target.elts)
            else:
                elements = [None] * len(target.elts)
            for inner, held in zip(target.elts, elements, strict=True):
                slot(inner, held)

    for node in _own_nodes(func):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                slot(target, node.value)
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            slot(node.target, node.value)
        elif isinstance(node, ast.NamedExpr):
            slot(node.target, node.value)
        elif isinstance(node, ast.AugAssign) and _text(node.target) == spelled:
            if isinstance(node.op, ast.Add):
                appended.append(node.value)
            else:
                opaque.append(node.target)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            slot(node.target, None)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None:
                    slot(item.optional_vars, None)
        elif isinstance(node, ast.ExceptHandler) and node.name == spelled and node.body:
            opaque.append(node.body[0])
    return plain, appended, opaque


def _still_written(
    plain: list[ast.expr],
    appended: list[ast.expr],
    opaque: list[ast.AST],
    at: ast.AST,
    spelled: str,
) -> tuple[list[ast.expr], list[ast.expr], list[ast.AST]]:
    """Of :func:`_message_sites`' bindings, the ones that can still be what *at* prints.

    Lying above the call was not enough.  A binding the function overwrites
    on every way to the call, or leaves the function after, is not a way the
    line is written: ``AtlasDDMClient.getAvailableFiles`` sets
    ``error_message`` in its ``try`` and again in the handler just before
    printing it, and ``TaskCommando.runImpl``'s ``tmpMsg``, written further
    down the loop, was offered to the line that prints ``executing ...`` --
    four lines said to print ``set task_status=ready``, where one does.  The
    build's printed statements carried 978 rows that no run can print, on 305
    statements.

    So it is the walk's own reading (:func:`pathcond.surviving_writes`): what
    ``+=`` adds keeps what was there, and ``except ... as`` binds on entering
    the handler, as Python does.  A binding inside the call's own statement
    (``log(msg := ...)``) is kept: the flow asks what reaches a statement's
    entry, and that one is written on the way through.
    """
    def written_at(node: ast.AST, kind: str) -> ast.AST:
        handler = getattr(node, "parent", None)
        if (
            kind == "opaque"
            and isinstance(handler, ast.ExceptHandler)
            and handler.name == spelled
            and handler.body
            and handler.body[0] is node
        ):
            return handler
        return pathcond.site_of(node)

    pairs = [
        (node, written_at(node, kind), kind)
        for kind, nodes in (("plain", plain), ("appended", appended), ("opaque", opaque))
        for node in nodes
    ]
    kept = pathcond.surviving_writes(
        [site for _node, site, _kind in pairs],
        at,
        rereads=[site for _node, site, kind in pairs if kind == "appended"],
        as_run=True,
    )
    alive = {id(site) for site in kept} | {id(pathcond.site_of(at))}

    def still(kind: str) -> list:
        return [node for node, site, one in pairs if one == kind and id(site) in alive]

    return still("plain"), still("appended"), still("opaque")


def _clip(text: str, width: int = 72) -> str:
    """*text* on one line, cut to *width* with an ellipsis when it is longer."""
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: width - 1] + "\u2026"


def _loop_bindings(func: ast.AST, name: str) -> list[ast.For | ast.AsyncFor]:
    """Loops whose target binds *name*.

    Reaching definitions do not cover a loop target, and 7.7% of the names
    conditions read are one.  The chain that matters most runs straight
    through one: a command is read out of a map that a ``for`` took out of the
    list the knight handed over.
    """
    found: list[ast.For | ast.AsyncFor] = []
    for binding in pathcond.bindings(func, name):
        if binding.kind == pathcond.BIND_FOR and all(binding.site is not one for one in found):
            found.append(binding.site)  # type: ignore[arg-type]
    return found


def _tuple_bindings(func: ast.AST, name: str) -> list[tuple[ast.Assign, Optional[int]]]:
    """Assignments that bind *name* as one element of an unpacking, and which.

    ``pathcond.assigned_expressions`` deliberately reads only ``ast.Name``
    targets, and it is shared with the build, so the form is picked up here
    instead of widening it -- the map's output must not move for a change to
    the walk.  The form matters: ``tmpStat, taskSpec = getTaskWithID_JEDI(...)``
    is how a spec arrives, and without it a guard reading ``taskSpec`` looks
    like a name nothing binds.

    The slot comes back with it because the site is only half the answer.
    ``tmpStat, taskSpec = get(...)`` rendered as ``taskSpec = get(...)`` says
    the call returns the spec, when it returns a pair whose second element is
    the spec -- and the first is the status that decides whether the second
    means anything.  38 of 218 binding steps over the 41-case sample are this
    shape.  A ``*rest`` in the target makes the position of everything after it
    depend on the length of the value, so the slot is left unsaid there rather
    than guessed, and so is a slot inside a nested unpacking.
    """
    return [
        (binding.site, binding.slot)  # type: ignore[misc]
        for binding in pathcond.bindings(func, name)
        if binding.kind == pathcond.BIND_ASSIGN and binding.unpacked
    ]


def _imported_names(tree: ast.Module) -> frozenset[str]:
    """Names that stand for modules, classes and functions rather than values.

    Dropped from the walk rather than reported as unresolved: ``JediTaskSpec``
    and ``Interaction`` appear in half the conditions in this corpus and the
    walk has nothing to say about either, so keeping them produces one
    unexplained step per mention and hides the ones that matter.

    A ``def`` or ``class`` at module scope is the same kind of name as an
    import -- ``_compFunc`` passed as a sort key is a function, not a value --
    so it is dropped here too.  Module-level *assignments* are not: those hold
    values, and :func:`_module_bindings` reads them.
    """
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                found.add(alias.asname or alias.name.split(".")[0])
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            found.add(node.name)
    return frozenset(found)


def _with_bindings(func: ast.AST, name: str) -> list[ast.expr]:
    """The context expressions a ``with ... as <name>`` binds.

    Reaching definitions do not cover the form, and it is how every
    ``TaskBuffer`` method reaches the database -- ``with self.proxyPool.get()
    as proxy`` -- so without it the walk said a function does not bind a name
    it plainly binds.
    """
    found: list[ast.expr] = []
    for binding in pathcond.bindings(func, name):
        if binding.kind == pathcond.BIND_WITH and binding.value is not None:
            if all(binding.value is not one for one in found):
                found.append(binding.value)
    return found


def _walrus_bindings(func: ast.AST, name: str) -> list[ast.expr]:
    """The values an assignment expression ``(<name> := ...)`` binds."""
    return [
        binding.value
        for binding in pathcond.bindings(func, name)
        if binding.kind == pathcond.BIND_WALRUS and binding.value is not None
    ]


#: How a name that is never re-assigned still gets its members.  The same set
#: :mod:`~bamboo.codemap.panda.recognizers.sqlwrite` carries a row on, less
#: the removals -- those take a value out and so answer no question about
#: where one came from.
_FILLERS = ("append", "add", "update", "extend", "setdefault")


#: The constructors an empty container is spelled with.
_EMPTY_CALLS = ("set", "dict", "list", "tuple", "defaultdict", "OrderedDict")


def _is_empty_container(node: ast.expr) -> bool:
    """Whether *node* is a container literal with nothing in it."""
    if isinstance(node, (ast.List, ast.Set, ast.Tuple)):
        return not node.elts
    if isinstance(node, ast.Dict):
        return not node.keys
    if isinstance(node, ast.Call) and not node.args and not node.keywords:
        return isinstance(node.func, ast.Name) and node.func.id in _EMPTY_CALLS
    return False


def _mutated_bindings(func: ast.AST, name: str) -> list[ast.expr]:
    """What ``<name>.add(x)`` and its kin put into a container.

    A name filled this way is assigned exactly once, to an empty container,
    and answering a reader with ``xmlIDs = set()`` is honest and useless.
    ``copyArchive.py::main`` skips every job already in ``xmlIDs``, and what
    is in it comes from ``xmlIDs.add(int(panda_id))`` over the rows of
    ``taskBuffer.listJobOutputReport()`` -- which is the answer to why the
    row was skipped.

    Twelve (function, name) pairs in the corpus are reached only this way --
    ``xmlIDs`` and ``sitesToDisableReassign`` in the same function,
    ``checkedDS``, ``umCheckedIDs``, ``workers_queued``, ``statsPerShare``.
    Kept narrow on purpose: a container that is assigned a *value* has that
    value to answer with, and listing every ``append`` beside it would bury
    it, so this is asked only where every assignment is an empty container.
    """
    found: list[ast.expr] = []
    for node in ast.walk(func):
        # ``returnMap["ramCount"] = memVal`` puts a value in as much as
        # ``xmlIDs.add(x)`` does, and it is how the scout's result is built.
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == name
                ):
                    found.append(node.value)
            continue
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        held = node.func.value
        if node.func.attr not in _FILLERS or not isinstance(held, ast.Name):
            continue
        if held.id != name or not node.args:
            continue
        # ``d.setdefault(key)`` names a key and no value; its last argument is
        # the key, and reporting that as what the container holds would be a
        # wrong answer rather than a short one.
        if node.func.attr == "setdefault" and len(node.args) < 2:
            continue
        found.append(node.args[-1])
    return found


def _module_bindings(tree: ast.Module, name: str) -> list[ast.expr]:
    """Module-scope assignments to *name*, nothing nested.

    The last place to look before saying a name is not bound here, and the
    trace's own rule says to look: a fact reachable forward from the anchor
    belongs to the trace, and a constant in the same file is as reachable as
    one a line above the arm.  ``skipBrokerageProTypes = ["prod_test"]``
    is the shape -- a value the reader wants, previously reported as a name
    nothing binds.

    Only the module's own body, so a name assigned inside some unrelated
    function of the same file is not offered as this one's.
    """
    found: list[ast.expr] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
        else:
            continue
        for target in targets:
            elements = target.elts if isinstance(target, (ast.Tuple, ast.List)) else [target]
            if any(isinstance(e, ast.Name) and e.id == name for e in elements):
                found.append(node.value)
    return found


def _is_main_test(test: ast.expr) -> bool:
    """Whether *test* is ``__name__ == "__main__"``, either way round."""
    return (
        isinstance(test, ast.Compare)
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and {ast.unparse(test.left), ast.unparse(test.comparators[0])} == {"__name__", "'__main__'"}
    )


def _caught_as(func: ast.AST, name: str) -> list[ast.ExceptHandler]:
    """Handlers that bind *name* as the exception they caught."""
    return [
        binding.site  # type: ignore[misc]
        for binding in pathcond.bindings(func, name)
        if binding.kind == pathcond.BIND_EXCEPT
    ]


def _nested_scope(func: ast.AST, site: ast.AST) -> Optional[ast.AST]:
    """The inner ``def`` or ``class`` between *site* and *func*, if any.

    ``pathcond.assigned_expressions`` walks the whole subtree, so a name
    assigned inside a nested function comes back as though *func* bound it.
    Its ``self.<field>`` sibling refuses to enter nested functions for the
    reason this note carries -- a closure runs when it is called, not where it
    is written -- but that one is read only at use time while this one is
    shared with the build, where the wider reading is load-bearing.  So the
    row stays and says where it came from instead.

    Keeping it matters: a daemon whose work lives in inner functions has its
    only answer there, and dropping the row would turn an incomplete
    explanation into a confident wrong one.
    """
    held = getattr(site, "parent", None)
    while held is not None and held is not func:
        if isinstance(held, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            return held
        held = getattr(held, "parent", None)
    return None


def _is_parameter(func: ast.AST, name: str) -> bool:
    args = getattr(func, "args", None)
    if args is None:
        return False
    every = list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)
    if args.vararg:
        every.append(args.vararg)
    if args.kwarg:
        every.append(args.kwarg)
    return any(argument.arg == name for argument in every)


def _raised_in(node: ast.AST, classify: Classifier, resolve, typed=None) -> tuple[str, str]:
    """The terminal of what the ``try`` body above this handler called.

    A handler ran because something above it raised, so the interface that
    could raise is where the value came from.  This is the only route from an
    arm whose own right-hand side is ``True`` to "DDM said so", and it is
    bounded -- the paired body's calls, not every statement before the site.
    """
    for ancestor in pathcond.ancestors(node):
        if isinstance(ancestor, (ast.FunctionDef, ast.AsyncFunctionDef)):
            break
        if not isinstance(ancestor, ast.ExceptHandler):
            continue
        block = getattr(ancestor, "parent", None)
        if not isinstance(block, ast.Try):
            break
        # Every call in the body that could have raised, not the first: the
        # handler in ``AdderAtlasPlugin.execute`` was put down to CRIC because
        # ``getSite`` is the first classifiable call in a body that also
        # registers files with Rucio.  The category stays the first one's; the
        # detail names each, so a traceback's frame can be matched to one.
        found: dict[str, tuple[str, str]] = {}
        for statement in block.body:
            for sub in ast.walk(statement):
                if isinstance(sub, ast.Call):
                    terminal, detail = classify(sub, resolve, typed)
                    if terminal:
                        found.setdefault(_text(sub.func), (terminal, detail))
        if len(found) == 1:
            return next(iter(found.values()))
        if found:
            first = next(iter(found.values()))[0]
            return first, "raised by one of: " + "; ".join(
                f"{called} ({detail})" for called, (_terminal, detail) in found.items()
            )
        break
    return "", ""


class _Walk:
    def __init__(
        self,
        roots: dict[str, Path],
        classify: Classifier,
        budget: Budget,
        observed: str = "",
        within: Optional[tuple[str, str, int, int]] = None,
        reading_at: Optional[tuple[str, str, int]] = None,
        narrowed: Optional[tuple[str, str]] = None,
        guards_above: bool = True,
    ) -> None:
        self.roots = roots
        #: Whether an arm's walk explains the tests above it as well as its own.
        #: See :func:`walk`.
        self.guards_above = guards_above
        self.classify = classify
        self.budget = budget
        self.observed = observed
        #: ``(file, owner, first, last)`` -- where in the starting function a
        #: name's bindings are looked for first.  See :meth:`_within`.
        self.within = within
        #: ``(file, owner, line)`` -- the line in the starting function whose
        #: reading is being explained.  See :meth:`_reaching`.
        self.reading_at = reading_at
        #: ``(file, owner)`` of the starting function where a name is narrowed
        #: to the writes that reach the statement that read it, or ``None``.
        self.narrowed = narrowed or (reading_at[:2] if reading_at else None)
        self.parsed: dict[str, tuple[str, ast.Module]] = {}
        self.frames: dict[tuple[str, str], Optional[_Frame]] = {}
        self._classes: dict[tuple[str, int, str], str] = {}
        self.steps: list[TraceStep] = []
        self.armed: dict[int, ast.stmt] = {}
        self.seen: set[tuple] = set()
        #: ``{(function, name): the bindings already walked}``, where a name is
        #: asked again from each statement that reads it -- so the second
        #: asking adds what *its* statement reaches and nothing twice.
        self.explained: dict[tuple[str, str], set[tuple[int, int, str]]] = {}
        self._scopes: dict[tuple[str, str], str] = {}
        self._logged: dict[tuple[str, str], list[tuple[ast.expr, _Message]]] = {}
        #: The budgets that ran out in the function's own answer, and beyond
        #: it -- every one, as :attr:`note` and :attr:`crossing_note` say them.
        self.ran_out: set[str] = set()
        self.ran_out_crossing: set[str] = set()
        self.queue: deque[tuple[_Frame, str, int, Optional[ast.AST]]] = deque()
        #: What waits until the function's own answer is done: going into a
        #: call, and the names asked in the functions reached that way.
        self.beyond: deque[Callable[[], None]] = deque()
        #: Whether what is being recorded now is spent from the crossing's
        #: allowance -- see ``Budget.crossed_steps``.
        self.crossing = False
        self.crossed = 0
        #: ``{(file, line of a method): [frames that asked a field it assigns]}``
        #: -- the ``__init__`` and the setters whose calls are a question about
        #: the object those frames came in on.  See :meth:`_object_made`.
        self.asked_of: dict[tuple[str, int], list[_Frame]] = {}
        #: ``{(file, owner): [what was narrowed]}`` of each frame that asked:
        #: another call into it can bring another object.  See :meth:`_came_later`.
        self.watched: dict[tuple[str, str], list[_Narrowed]] = {}
        #: ``{(file, line of a method): [what was narrowed]}``, for a frame
        #: that asks after the narrowing was done.
        self.narrowings: dict[tuple[str, int], list[_Narrowed]] = {}
        #: ``{(file, line of an __init__): [calls left out by class]}``, for a
        #: method of another class that asks later.  See :meth:`_asked_into`.
        self.by_class: dict[tuple[str, int], list[_ByClass]] = {}
        #: :meth:`_made_at` per (file, function, expression).
        self._made: dict[tuple[str, int, str], Optional[set[tuple[str, int, int]]]] = {}
        #: ``_message_sites`` per (function, spelling): a function with a
        #: hundred logging calls asks for the same local a hundred times.
        self._sites: dict[tuple[int, str], tuple] = {}
        self._asked: dict[tuple[str, str], str] = {}
        self._prefixes: dict[tuple[str, str, str], str] = {}
        #: The role the name being explained was asked in -- see
        #: ``TraceStep.role`` -- and the role each queued asking carries.
        self._role = ""
        self._roles: dict[tuple, str] = {}
        self._constants: dict[int, dict[str, list]] = {}
        self.types = _Types(roots, self.parsed)
        self._declared: dict[tuple[str, str, str], Optional[str]] = {}
        self.calls = calls_mod.Calls(self.types)
        #: ``{(file, owner): [(caller frame, the call, what it binds)]}`` for
        #: every function the walk went into, so a parameter read there is
        #: answered by the call that brought the walk in.
        self.passed: dict[tuple[str, str], list[tuple[_Frame, ast.Call, dict]]] = {}
        #: Functions the walk went into: narrowed like the starting one, each
        #: name answered with the writes that reach the statement reading it.
        self.entered: set[tuple[str, str]] = set()
        #: ``(file, owner, projection)`` already gone into for its returns.
        self.returned: set[tuple] = set()
        #: ``{(file, owner): reason}`` for each function some call placed open
        #: entered -- :attr:`TraceStep.open` of its return steps.
        self.entered_open: dict[tuple[str, str], str] = {}

    # -- source ------------------------------------------------------------

    def module(self, file: str) -> Optional[tuple[str, ast.Module]]:
        if file in self.parsed:
            return self.parsed[file]
        limit = self.budget.files + (self.budget.crossed_files if self.crossing else 0)
        if len(self.parsed) >= limit:
            self._exhausted("files")
            return None
        package, _, relative = file.partition("/")
        root = self.roots.get(package)
        if root is None or not relative:
            # Outside the corpus the map covers, which is the walk's stop rule
            # rather than a failure: a name that comes from somewhere else is
            # exactly what a boundary is for.
            return None
        # The process's one tree for the file, never a parse of the walk's
        # own: the caller index and the declared-type reading hold that tree's
        # nodes -- see ``calls.parsed``.  What this walk has read is still its
        # own, and that is what the file budget counts.
        held = calls_mod.parsed(root / relative)
        if held is None:
            logger.debug("trace: %s unreadable", file)
            return None
        self.parsed[file] = held
        return held

    def frame(
        self,
        file: str,
        owner: str,
        *,
        line: Optional[int] = None,
        handovers: Sequence[Handover] = (),
    ) -> Optional[_Frame]:
        key = (file, owner)
        if key in self.frames:
            held = self.frames[key]
            if held is None or line is None or held.func.lineno <= line <= (held.func.end_lineno or held.func.lineno):
                return held
            # Two classes of one file with a method of one name, and the key
            # holds the other one's: ``datasetManager.py`` defines ``run`` in
            # each of its threads, and the line asked about is not in the
            # ``run`` the walk entered first.  Spelled with the class, as
            # ``_callee_frame`` spells a call's.
            func = containing_function(self.parsed[file][1], line, owner) if file in self.parsed else None
            if func is None or "." in owner.rsplit("::", 1)[-1]:
                return held
            return self.frame(file, _with_class(file, self.parsed[file][1], func), line=line, handovers=handovers)
        if len(self.frames) >= self._functions_limit():
            self._exhausted("functions")
            return None
        self.frames[key] = None
        parsed = self.module(file)
        if parsed is None:
            return None
        _source, tree = parsed
        wanted = owner.rsplit("::", 1)[-1].rsplit(".", 1)[-1]
        if line is not None:
            func = containing_function(tree, line, owner)
        else:
            func = next(
                (
                    node
                    for node, _cls in pathcond.functions_with_owner(tree)
                    if node.name == wanted
                ),
                None,
            )
        if func is None:
            return None
        self.frames[key] = _Frame(
            file, owner, func, tuple(handovers), _imported_names(tree)
        )
        return self.frames[key]

    def _functions_limit(self) -> int:
        return self.budget.functions + (self.budget.crossed_functions if self.crossing else 0)

    # -- steps -------------------------------------------------------------

    def _exhausted(self, what: str) -> None:
        (self.ran_out_crossing if self.crossing else self.ran_out).add(what)

    @property
    def note(self) -> str:
        """Which budgets the function's own answer ran out of, or ``""``.

        Every one, not the first: ``setCloudToTasks_JEDI``'s walk ran out of
        depth and then of steps beyond the function, and saying depth alone
        hid the 37 steps the steps budget left out (X8).
        """
        if not self.ran_out:
            return ""
        return f"the walk's {_budgets(self.ran_out)} ran out; what follows is part of the answer"

    @property
    def crossing_note(self) -> str:
        """:attr:`note` for what the walk spent beyond the function."""
        if not self.ran_out_crossing:
            return ""
        return (
            f"the walk's {_budgets(self.ran_out_crossing)} beyond this function ran out; "
            "what the calls hand back is part of the answer"
        )

    def database_asked(self, method: str, frame: Optional[_Frame] = None, call: Optional[ast.Call] = None) -> str:
        """``reads jobsDefined4, jobsActive4 by :jediTaskID`` for what the task buffer's *method* asks the proxy, or ``""``.

        The step that stops at "a value the database supplies" named the
        method and nothing else: which table the value came from, and by what
        key, were one proxy method away and unsaid.  Read from the statements
        the proxy's methods run, the way the build reads them.

        The proxy's methods are the ones the facade's *method* calls
        (:func:`~bamboo.codemap.panda.provenance.facade_calls`), not the
        proxy's method of the same name: ``peekJobs`` reads what ``peekJob``
        reads, and ``querySQL`` what ``querySQLS`` does.  A name no facade
        has says nothing.  A call the facade makes under an ``if`` the call's
        constants decide against is not read:
        ``getDatasetsWithJediTaskID_JEDI`` reads the files only when it is
        handed ``getFiles=True``, and its default is ``False``.

        Read at *call*, the call in *frame* that went to the database, where a
        route of calls into a proxy method passes through it
        (:meth:`Calls.routes`): with the constants each such route hands the
        parameters the method tests, as the build reads a route
        (``sqlwrite._route_constant``), so ``getPandaIDsWithTask_JEDI(jediTaskID,
        True)`` reads the two tables ``onlyActive`` leaves, not four.  The
        tables and keys of every route through the call together.  Where no
        route passes through it -- the call is not placed, or the facade does
        not hand the method its own parameters -- every run the method can
        make, as before.
        """
        from bamboo.codemap.panda import sql as sql_mod
        from bamboo.codemap.panda import sqltext
        from bamboo.codemap.panda.provenance import facade_calls

        at = f"{frame.file}:{call.lineno}" if frame is not None and call is not None else ""
        if (method, at) in self._asked:
            return self._asked[(method, at)]
        from bamboo.codemap.panda.recognizers import sqlwrite

        tables: list[str] = []
        keys: list[str] = []
        read: set[tuple[str, int]] = set()
        for made in facade_calls(self.types, method):
            if at and self._ruled_out_at(made, at):
                continue
            where, func = made.proxy_file, made.proxy_method
            if (where, func.lineno) in read:
                continue
            read.add((where, func.lineno))
            readings = [None]
            if at:
                owned = self.types.method_class(where, func)
                target = calls_mod.Target(where, func, owned[1] if owned else None)
                through = [route for route in self.calls.routes(target)[0] if at in route.chain]
                if through:
                    flags = sorted(sqlwrite._flag_parameters(func))
                    readings = [
                        {
                            name: value
                            for name in flags
                            if (value := sqlwrite._route_constant(name, target, route.sites, self.calls))
                            is not sqlwrite._UNSET
                        }
                        for route in through
                    ]
            for constants in readings:
                if constants:
                    with sql_mod.at_call(func, constants):
                        runs = list(sql_mod.executions(func))
                else:
                    runs = list(sql_mod.executions(func))
                for run in runs:
                    for table in sqltext.tables_named(run.sql):
                        _add(tables, table)
                    for clause in sqltext.where_clauses(run.sql):
                        for key in re.findall(r":(\w+)", clause):
                            _add(keys, f":{key}")
        said = ""
        if tables:
            said = f"reads {', '.join(tables[:6])}" + (
                f" by {', '.join(keys[:6])}" if keys else ""
            )
        self._asked[(method, at)] = said
        return said

    def _ruled_out_at(self, made, at: str) -> bool:
        """Whether every route through *at* into the facade's method hands it constants under which it never makes *made*'s call."""
        from bamboo.codemap.panda.recognizers import sqlwrite

        owned = self.types.method_class(made.file, made.method)
        facade = calls_mod.Target(made.file, made.method, owned[1] if owned else None)
        through = [route for route in self.calls.routes(facade)[0] if at in route.chain]
        if not through:
            return False
        flags = sorted(sqlwrite._flag_parameters(made.method))
        return all(
            pathcond.ruled_out_by(
                made.call,
                made.method,
                {
                    name: value
                    for name in flags
                    if (value := sqlwrite._route_constant(name, facade, route.sites, self.calls))
                    is not sqlwrite._UNSET
                },
            )
            for route in through
        )

    def _classified(
        self, frame: _Frame, expression: ast.expr, resolve=None, typed=None
    ) -> tuple[str, str]:
        """``(terminal, detail)`` for *expression* in *frame*; a value the database supplies says what it is read from at the call."""
        resolve = resolve if resolve is not None else self.resolver(frame)
        typed = typed if typed is not None else self.typer(frame)
        terminal, detail = self.classify(expression, resolve, typed)
        if terminal == STOP_UPSTREAM and detail and " " not in detail:
            from bamboo.codemap.panda.provenance import database_call

            asked = self.database_asked(detail, frame, database_call(expression, resolve, typed))
            if asked:
                detail = f"{detail} -- {asked}"
        return terminal, detail

    def record(self, step: TraceStep) -> bool:
        if self._role and not step.role:
            step.role = self._role
        if not step.cls:
            step.cls = self._class_at(step)
        if step.terminal == STOP_UPSTREAM and step.detail and " " not in step.detail:
            asked = self.database_asked(step.detail)
            if asked:
                step.detail = f"{step.detail} -- {asked}"
        if self.crossing:
            if self.crossed >= self.budget.crossed_steps:
                self._exhausted("steps")
                return False
            self.crossed += 1
            self.steps.append(step)
            return True
        if len(self.steps) - self.crossed >= self.budget.steps:
            self._exhausted("steps")
            return False
        self.steps.append(step)
        return True

    def _class_at(self, step: TraceStep) -> str:
        """``file::Class`` for the method holding *step*'s statement, or ``""``.

        Read off modules the walk has already parsed and nothing else: parsing
        one here would spend the walk's file budget on a label.  A step with no
        line -- a stop -- takes the class of the one frame its owner names.
        """
        held = self.parsed.get(step.file)
        if held is None:
            return ""
        key = (step.file, step.line, step.owner)
        if key in self._classes:
            return self._classes[key]
        tree = held[1]
        func = containing_function(tree, step.line, step.owner) if step.line else None
        if func is None:
            frames = {
                id(one.func): one.func
                for one in self.frames.values()
                if one is not None and one.file == step.file and one.owner == step.owner
            }
            func = next(iter(frames.values())) if len(frames) == 1 else None
        cls = pathcond.class_of(tree, func) if func is not None else None
        self._classes[key] = f"{step.file}::{cls.name}" if cls is not None else ""
        return self._classes[key]

    def drain(self) -> None:
        """Explain everything asked: the function's own names first, then beyond it.

        Once the walk has gone beyond the function it stays there for the
        budget's sake: a name of the function asked again from a call that
        came back to it -- the argument that filled a parameter -- is spent
        from the crossing's allowance, not from what the function's own answer
        was given.
        """
        self.crossing = False
        while self.queue:
            held, name, depth, at = self.queue.popleft()
            self._explain_as(self._roles.get((held.file, held.owner, name, id(at)), ""), held, name, depth, at)
        self.crossing = True
        while self.queue or self.beyond:
            if self.queue:
                held, name, depth, at = self.queue.popleft()
                self._explain_as(self._roles.get((held.file, held.owner, name, id(at)), ""), held, name, depth, at)
            else:
                self._role = ""
                self.beyond.popleft()()
        self.crossing = False
        self._role = ""

    def _explain_as(self, role: str, frame: _Frame, name: str, depth: int, at) -> None:
        self._role = role
        try:
            self.explain(frame, name, depth, at)
        finally:
            self._role = ""

    def notes(self) -> str:
        """The note, and the crossing's own where it ran out."""
        return "; ".join(one for one in (self.note, self.crossing_note) if one)

    def resolver(self, frame: _Frame):
        return lambda name: pathcond.single_definition(frame.func, name)

    def typer(self, frame: _Frame):
        """expression -> "<file>::<Class>" it is declared to be in *frame*, or None."""

        def declared(expression: ast.expr) -> Optional[str]:
            key = (frame.file, frame.owner, _text(expression))
            if key not in self._declared:
                found = self.types.of_expression(frame.file, frame.func, expression)
                self._declared[key] = f"{found[0]}::{found[1].name}" if found else None
            return self._declared[key]

        return declared

    def terminal_of(self, frame: _Frame, expression: ast.expr) -> tuple[str, str]:
        """(terminal, detail) for *expression* read in *frame*, the walk's way."""
        return self._classified(frame, expression)

    def scope_key(self, frame: _Frame) -> str:
        """What a name asked in *frame* is unique within.

        A local belongs to its function.  ``self.<field>`` belongs to the
        object, so the class is its scope and not the method that happened to
        read it -- and once stage three answers a field from every method of
        the class, asking again from a sibling frame would replay the same
        assignments under a second heading.  Keyed by the class where there is
        one, so the two kinds of name cannot collide.
        """
        held = self._scopes.get((frame.file, frame.owner))
        if held is not None:
            return held
        parsed = self.module(frame.file)
        cls = None
        if parsed is not None:
            cls = pathcond.class_of(parsed[1], frame.func)
        key = f"{frame.file}::{cls.name}" if cls is not None else f"{frame.file}::{frame.owner}"
        self._scopes[(frame.file, frame.owner)] = key
        return key

    def narrows(self, frame: _Frame) -> bool:
        """Whether names in *frame* are narrowed to the writes that reach their read.

        The starting function where the walk asks for it, and every function
        it went into through a call: there the reading point is the
        ``return`` or the call, never a line of the starting function.
        """
        key = (frame.file, frame.owner)
        return (self.narrowed is not None and key == self.narrowed) or key in self.entered

    def want(
        self,
        frame: _Frame,
        names: Sequence[str],
        depth: int,
        at: Optional[ast.AST] = None,
    ) -> None:
        """Queue *names* for explaining, as read by the statement *at*.

        **Once per statement that reads a name, where the walk is narrowed.**
        Once per name is right for a walk that answers with every assignment
        in the function, since a second asking would find the same ones.  It
        is wrong for a walk that answers with the writes reaching *the read*,
        because a name read in two places has two sets of those -- and
        ``getScoutJobData_JEDI`` reads ``memVal`` four times on the way to
        line 1019: ``memVal = int(memVal)`` on 1007 needs the ``/= 1024`` on
        1006, which needs the margin on 1005, which needs the percentile on
        1002.  Asked once, from 1019, the walk kept 1007 and never asked
        again, so the percentile and both scout settings behind it were cut.
        """
        if depth > self.budget.depth + (self.budget.crossed_depth if self.crossing else 0):
            self._exhausted("depth")
            return
        for name in names:
            if name in frame.imports:
                continue
            if name.startswith("self."):
                key: tuple = (self.scope_key(frame), name)
            else:
                key = (f"{frame.file}::{frame.owner}", name)
                if at is not None and self.narrows(frame):
                    key = (*key, id(at))
            if key in self.seen:
                continue
            self.seen.add(key)
            role = self._role
            if (frame.file, frame.owner) in self.entered:
                self.beyond.append(
                    lambda held=frame, one=name, deep=depth, where=at, part=role: self._explain_as(
                        part, held, one, deep, where
                    )
                )
            else:
                self._roles[(frame.file, frame.owner, name, id(at))] = role
                self.queue.append((frame, name, depth, at))

    # -- the walk ----------------------------------------------------------

    def arm(self, frame: _Frame, line: int) -> None:
        statement = _statement_at(frame.func, line)
        if statement is None:
            return
        value = _value_of(statement)
        reads = _reads(value) if value is not None else []
        own: list[str] = []
        # An arm is usually an assignment, and then its value is what to
        # follow.  An exit is not: it is armed at the ``if`` that decides it,
        # and ``_value_of`` has no value for one.  ``_guard_reads`` answers
        # for the tests *above* a statement and correctly leaves this one out,
        # so ``copyArchive.py``'s ``if int(id) in xmlIDs: continue`` was
        # explained by the rows it was iterating and never by ``xmlIDs`` --
        # the name that decided it.  Seventy of the corpus's exits were short
        # of a hundred and two such names.
        if isinstance(statement, (ast.If, ast.While)):
            for name in _reads(statement.test):
                _add(reads, name)
        own = list(reads)
        for name in _guard_reads(statement) if self.guards_above else ():
            _add(reads, name)
        terminal, detail = (
            self.terminal_of(frame, value) if value is not None else ("", "")
        )
        step = TraceStep(
            kind=TRACE_WRITE,
            owner=frame.owner,
            file=frame.file,
            line=line,
            value=_text(value) if value is not None else _text(statement),
            guards=pathcond.path_condition(statement),
            unseen=_unseen(statement),
            reads=reads,
            terminal=terminal,
            detail=detail,
        )
        if not self.record(step):
            return
        # Kept for the skeleton, which is built once for the whole function
        # after the walk rather than once per arm: the lines a function prints
        # do not change with which arm is being explained, and rendering them
        # per arm is what made the old shape repeat itself.
        self.armed[line] = statement
        if not terminal or terminal in _WALKED_THROUGH:
            # Read at the line the caller named, where it named one: an exit
            # is decided at its ``if`` and a statement answers for the binds
            # filled on the way to it, and the arm is not always that line.
            if not self.narrows(frame):
                self.want(frame, reads, 1)
                return
            at: ast.AST = statement
            if self.reading_at is not None:
                at = _statement_at(frame.func, self.reading_at[2]) or statement
            self.want(frame, own, 1, at)
            for test, names in _guard_tests(statement) if self.guards_above else ():
                self.want(frame, names, 1, test)
        self._enter_tests(frame, statement, 1, above=self.guards_above)
        if value is not None:
            self._ask_keyed(frame, value, 1, statement)
            if not terminal or terminal == STOP_EXTERNAL:
                self.beyond.append(lambda: self._enter(frame, step, value, None, 1))

    # -- skeleton ----------------------------------------------------------

    def messages(self, frame: _Frame) -> list[tuple[ast.expr, _Message]]:
        """Every way each logging call in this frame can print, or why it cannot be said.

        One entry per way, and at least one per call: a call whose message
        cannot be written out still gets a row with the reason, because the
        reader lays the skeleton over a grep of the log region and a line that
        is there and not here reads as certainty rather than as a gap.

        Only the bindings that can reach the call count -- one below it, or one
        its path condition contradicts, cannot have run on the way to it.  That
        took 2188 (call, message) pairs to 1843 when it was first applied.
        """
        key = (frame.file, frame.owner)
        if key in self._logged:
            return self._logged[key]
        out: list[tuple[ast.expr, _Message]] = []
        if isinstance(frame.func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for argument in logging_arguments(frame.func):
                if isinstance(argument, (ast.JoinedStr, ast.Constant)):
                    # Written at the call: rendered as it always was.
                    out.append((argument, _Message(argument)))
                    continue
                ways, refused = self._message_ways(frame, argument, argument, 0, frozenset())
                for pieces in ways:
                    if len(pieces) == 1 and isinstance(pieces[0], ast.Constant):
                        message: ast.expr = pieces[0]
                    else:
                        message = ast.JoinedStr(values=_merged(pieces))
                    out.append((argument, _Message(message)))
                for reason in dict.fromkeys(refused):
                    out.append((argument, _Message(argument, reason)))
        self._logged[key] = out
        return out

    def _message_ways(
        self,
        frame: _Frame,
        expr: ast.expr,
        at: ast.AST,
        depth: int,
        seen: frozenset[str],
    ) -> tuple[list[list[ast.expr]], list[str]]:
        """``(ways, refusals)``: each way *expr* can print as pieces, and why the rest cannot.

        A piece is literal text (``ast.Constant``) or a hole
        (``ast.FormattedValue``) -- an f-string's own parts, so every way comes
        out as one and renders the way an f-string always has.  A hole is also
        what a part of the message becomes when its text cannot be written out
        here and the rest can: ``error_message + traceback.format_exc()`` is
        the message, a hole, and nothing is lost by saying so.  Only where the
        *whole* message comes from somewhere this walk does not look is there
        a refusal, with the reason.
        """
        if depth > _MESSAGE_DEPTH:
            return [[_hole(expr)]], []
        if isinstance(expr, ast.Constant):
            return ([[expr]] if isinstance(expr.value, str) else [[_hole(expr)]]), []
        if isinstance(expr, ast.JoinedStr):
            return [list(expr.values)], []
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
            left = self._operand(frame, expr.left, at, depth, seen)
            right = self._operand(frame, expr.right, at, depth, seen)
            if len(left) * len(right) > _MAX_EXPANDED:
                if len(left) >= len(right):
                    left = [[_hole(expr.left)]]
                else:
                    right = [[_hole(expr.right)]]
            return [one + other for one in left for other in right], []
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Mod):
            templates, _refused = self._message_ways(frame, expr.left, at, depth + 1, seen)
            texts = [_constant_text(template) for template in templates]
            ways = [_printf_pieces(text, expr.right) for text in texts if text is not None]
            if templates and len(ways) == len(templates) and all(w is not None for w in ways):
                return ways, []  # type: ignore[return-value]
            return [[_hole(expr)]], []
        if isinstance(expr, ast.Call):
            called = expr.func
            if isinstance(called, ast.Attribute) and called.attr == "format":
                templates, _refused = self._message_ways(frame, called.value, at, depth + 1, seen)
                texts = [_constant_text(template) for template in templates]
                ways = [_format_pieces(text, expr) for text in texts if text is not None]
                if templates and len(ways) == len(templates) and all(w is not None for w in ways):
                    return ways, []  # type: ignore[return-value]
                return [[_hole(expr)]], []
            name = called.attr if isinstance(called, ast.Attribute) else getattr(called, "id", "")
            if (
                name in _PRINTS_TEXT
                or name in _PRINTS_NUMBER
                or (isinstance(called, ast.Name) and called.id in _BUILTINS)
            ):
                return [[_hole(expr)]], []
            return [], [REFUSED_RETURNED]
        if isinstance(expr, ast.IfExp):
            ways: list[list[ast.expr]] = []
            for branch in (expr.body, expr.orelse):
                ways.extend(self._operand(frame, branch, at, depth, seen))
            return ways, []
        if isinstance(expr, (ast.Name, ast.Attribute, ast.Subscript)):
            return self._bound_message(frame, expr, at, depth, seen)
        return [[_hole(expr)]], []

    def _operand(
        self, frame: _Frame, expr: ast.expr, at: ast.AST, depth: int, seen: frozenset[str]
    ) -> list[list[ast.expr]]:
        """The ways a *part* of a message prints; one hole where it cannot be said."""
        ways, refused = self._message_ways(frame, expr, at, depth + 1, seen)
        if not ways or refused or len(ways) > _MAX_EXPANDED:
            return [[_hole(expr)]]
        return ways

    def _bound_message(
        self, frame: _Frame, expr: ast.expr, at: ast.AST, depth: int, seen: frozenset[str]
    ) -> tuple[list[list[ast.expr]], list[str]]:
        """The ways a name, an attribute or an item printed here was written.

        What this function writes into it before the call: every outright
        binding that can still be what it prints (:func:`_still_written`),
        each a way of its own, and what ``+=`` adds
        after as a hole at the end.  What ``+=`` adds is a way of its own too,
        where it has literal text -- the line then *contains* it, wherever the
        first binding left off.  ``err_str += traceback.format_exc()`` has
        none, and adds no row: the line is the one the first binding wrote.
        A short one is kept and marked like any other -- ``for merge`` and
        ``for retry`` added to one message are what says which branch ran.

        One row per binding however many there are, as the build's own
        reading of the same call always gave; only a message nested inside
        another is capped, and then left one hole rather than dropped.

        A name bound nowhere in the function is a module constant, a parameter
        or a value from elsewhere, and only the first is read here.
        """
        spelled = _text(expr)
        if spelled in seen:
            return [[_hole(expr)]], []
        seen = seen | {spelled}
        held = self._sites.get((id(frame.func), spelled))
        if held is None:
            held = self._sites[(id(frame.func), spelled)] = _message_sites(frame.func, spelled)
        plain, appended, opaque = _still_written(*held, at, spelled)
        if not plain and not opaque:
            if not isinstance(expr, ast.Name):
                return [], [REFUSED_SET_ELSEWHERE]
            if _is_parameter(frame.func, expr.id):
                return [], [REFUSED_HANDED_IN]
            parsed = self.module(frame.file)
            outer = _module_bindings(parsed[1], expr.id) if parsed is not None else []
            if expr.id in frame.imports or not outer:
                return [], [REFUSED_SET_ELSEWHERE]
            plain = outer
        ways: list[list[ast.expr]] = []
        refused: list[str] = []
        # A parameter, a field or an item keeps whatever it held on the way in
        # unless a binding here runs on every path to the call -- and a way
        # that holds the caller's value is one this walk cannot write out.  A
        # local has no such way: read before it is bound, it raises.
        outside = (
            REFUSED_HANDED_IN
            if isinstance(expr, ast.Name) and _is_parameter(frame.func, expr.id)
            else "" if isinstance(expr, ast.Name) else REFUSED_SET_ELSEWHERE
        )
        if outside and not any(_dominates(pathcond.site_of(value), at) for value in plain):
            refused.append(outside)
        tail = [_appended(expr)] if appended else []
        for value in plain:
            found, why = self._message_ways(frame, value, value, depth + 1, seen)
            ways.extend(way + tail for way in found)
            refused.extend(why)
        for value in appended:
            found, _why = self._message_ways(frame, value, value, depth + 1, seen)
            ways.extend(tail + way + tail for way in found if _has_literal(way))
        if opaque:
            ways.append([_hole(expr)])
        if len(ways) + len(refused) > _MAX_EXPANDED and depth:
            return [[_hole(expr)]], []
        return ways, refused

    def prefix_of(self, frame: _Frame, argument: ast.expr) -> str:
        """The pattern a logging call's logger writes before *argument*, or ``""``.

        ``LogWrapper(logger, prefix)`` writes ``prefix + " " + message``.  A
        logger built here says its prefix; one built by a call -- the proxy's
        ``create_tagged_logger(comment, tag)`` -- says it in the function it
        is built in, read with the call's arguments folded in.  Nothing else
        is guessed at: a logger built outside the function has a prefix this
        call site does not state.
        """
        call = getattr(argument, "parent", None)
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)):
            return ""
        receiver = call.func.value
        if not isinstance(receiver, ast.Name):
            return ""
        key = (frame.file, frame.owner, receiver.id)
        if key in self._prefixes:
            return self._prefixes[key]
        said = ""
        bound = pathcond.single_definition(frame.func, receiver.id)
        if isinstance(bound, ast.Call):
            pieces = self._logger_prefix(frame, bound)
            if pieces:
                said = _folded_pattern(pieces) + r"\ "
        self._prefixes[key] = said
        return said

    def _logger_prefix(self, frame: _Frame, built: ast.Call, depth: int = 0) -> Optional[list]:
        """The folded prefix of the ``LogWrapper`` *built* makes, directly or by returning one."""
        env = self._constant_env(frame.func)
        if _called_name(built) == "LogWrapper" and len(built.args) >= 2:
            return _fold(built.args[1], env)
        if depth > 1:
            return None
        placed = self.calls.callees(frame.file, frame.func, built)
        # One definition, placed closed: placed open, the logger may be
        # built by a function this reading cannot name.
        if len(placed.targets) != 1 or placed.unplaced:
            return None
        (target,) = placed.targets
        bound = calls_mod.bind(built, target, placed.on_class)
        callee_env = {
            name: (_fold(value, env) if value is not None else None)
            for name, value in bound.items()
        }
        callee_env = {name: value for name, value in callee_env.items() if value is not None}
        loggers: dict[str, Optional[list]] = {}
        for statement in target.func.body:  # type: ignore[attr-defined]
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
                value = statement.value
                if isinstance(value, ast.Call) and _called_name(value) == "LogWrapper" and len(value.args) >= 2:
                    loggers[statement.targets[0].id] = _fold(value.args[1], callee_env)
                    continue
                folded = _fold(value, callee_env)
                callee_env[statement.targets[0].id] = folded if folded is not None else [_Unknown(statement.targets[0].id)]
            elif isinstance(statement, ast.AugAssign) and isinstance(statement.target, ast.Name) and isinstance(statement.op, ast.Add):
                held = callee_env.get(statement.target.id)
                folded = _fold(statement.value, callee_env)
                callee_env[statement.target.id] = (held or [_Unknown(statement.target.id)]) + (folded or [_Unknown("...")])
            elif isinstance(statement, ast.If):
                # Whatever a branch assigns is not known to have run.
                for node in ast.walk(statement):
                    if isinstance(node, ast.Assign):
                        for held in node.targets:
                            if isinstance(held, ast.Name):
                                callee_env[held.id] = [_Unknown(held.id)]
            elif isinstance(statement, ast.Return):
                returned = statement.value
                if isinstance(returned, ast.Name):
                    return loggers.get(returned.id)
                if isinstance(returned, ast.Call) and _called_name(returned) == "LogWrapper" and len(returned.args) >= 2:
                    return _fold(returned.args[1], callee_env)
                return None
        return None

    def _constant_env(self, func: ast.AST) -> dict[str, list]:
        """Locals bound once, to a string this reading can fold."""
        key = id(func)
        if key not in self._constants:
            env: dict[str, list] = {}
            for name, values in pathcond.assigned_expressions(func).items():
                if len(values) == 1:
                    folded = _fold(values[0], {})
                    if folded is not None:
                        env[name] = folded
            self._constants[key] = env
        return self._constants[key]

    def _value_hole(
        self, frame: _Frame, message: ast.JoinedStr, arms: Sequence[ast.stmt]
    ) -> tuple[str, str, str]:
        """``(pattern, hole, because)`` for the hole the observed value lands in.

        The one thing the map's shared sentence cannot pick.  The arm's own
        statement and the value actually observed are both in hand here, so
        the hole can be chosen by what it *spells* rather than by where it
        sits -- the map anchors on the text before the first hole, and for 80
        of the 182 subjects that get a probe at all that is not the hole the
        value fills.

        First choice is a hole one of *arms* names, which is a fact about a
        write.  Second is a hole holding a name a reaching definition settles
        to the observed value, which is only a fact about the function.  Empty
        when neither, and nothing is guessed to fill the gap: an arm no line
        reports keeps its silence, which is the honest answer.

        *arms* is every arm this row could have been printed alongside, and
        the names they write are taken together.  A row is in the skeleton
        once, so a value pattern that holds for any of them belongs on it --
        and which ones those are is on the row, in ``arms``, rather than
        implied by there being a value at all.
        """
        if not self.observed:
            return "", "", ""
        written: set[str] = set()
        for arm in arms:
            written.update(_written_texts(arm))
        holes = [
            (position, part)
            for position, part in enumerate(message.values)
            if isinstance(part, ast.FormattedValue)
        ]
        fallback = ("", "", "")
        for position, part in holes:
            if not _neighbouring_literal(message, position):
                continue
            spelled = _text(part.value)
            rendered = _rendered(message, position, self.observed)
            if not rendered:
                continue
            if spelled in written:
                return rendered, spelled, "the arm writes it"
            if not fallback[0] and isinstance(part.value, ast.Name):
                if self._holds_observed(frame, part.value.id):
                    fallback = (rendered, spelled, f"{spelled} is set to this value above")
        return fallback

    def skeleton(
        self, frame: _Frame, region: Optional[tuple[int, int]] = None
    ) -> list[SkeletonLine]:
        """What this function prints, in source order, with the arms in place.

        Built from the whole function rather than from the arms outward, and
        that is the point: a line that says nothing about the value still
        proves the code between two of them ran, so the rows that carry no
        hole are as much of the answer as the rows that do.

        Only the parts of the tree that lead to a printed line or an arm are
        walked into.  A skeleton is not the source again -- a 540-line
        function whose every branch is reproduced buries the handful of rows a
        reader came for.

        *region* is how the read side says which part of the function is its
        own.  ``copyArchive.py::main`` runs three ``holding`` queries and is
        1353 lines: without a region all three answer with the same 251 rows,
        and the reader cannot tell which limit rejected their row.  A
        statement outside the region is not printed, and a header above one
        inside it still is -- the nesting is what a reader is here for, so it
        survives by enclosing something shown rather than by its own line.
        """
        func = frame.func
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return []

        def within(statement: ast.stmt) -> bool:
            if region is None:
                return True
            return region[0] <= statement.lineno <= region[1]

        printed: dict[int, list[tuple[ast.expr, _Message]]] = {}
        for argument, message in self.messages(frame):
            statement = _enclosing_statement(argument)
            if statement is not None and within(statement):
                printed.setdefault(id(statement), []).append((argument, message))
        rows: list[SkeletonLine] = []
        self._emit(frame, func.body, 0, printed, rows)
        return rows

    def _emit(
        self,
        frame: _Frame,
        body: Sequence[ast.stmt],
        depth: int,
        printed: dict[int, list[tuple[ast.expr, _Message]]],
        rows: list[SkeletonLine],
    ) -> None:
        for statement in body:
            if not _of_interest(statement, printed, self.armed):
                continue
            blocks = _blocks_of(statement)
            if blocks:
                # A call in the header itself -- ``if log.warning(x):`` is
                # absurd but ``with open(name(log.debug(x))):`` is not, and
                # ``_enclosing_statement`` hands both back keyed on the
                # compound.  Emitted before the header so the row is not lost
                # to a branch that only ever recurses.
                for row in self._rows_for(frame, statement, depth, printed):
                    if row.kind == SKELETON_PRINT:
                        rows.append(row)
                # Every header of a statement that is worth showing at all,
                # even where that block holds nothing.  Dropping the empty
                # ones produced an ``except Exception:`` with no ``try:`` over
                # it and an ``else:`` whose condition was nowhere on the page
                # -- and the nesting is the thing a reader is here for, so a
                # header that explains a shown block is not decoration.
                for header, inner in blocks:
                    rows.append(
                        SkeletonLine(
                            kind=SKELETON_BRANCH,
                            line=getattr(statement, "lineno", 0),
                            depth=depth,
                            text=header,
                        )
                    )
                    if any(_of_interest(one, printed, self.armed) for one in inner):
                        self._emit(frame, inner, depth + 1, printed, rows)
                continue
            for row in self._rows_for(frame, statement, depth, printed):
                rows.append(row)

    def _rows_for(
        self,
        frame: _Frame,
        statement: ast.stmt,
        depth: int,
        printed: dict[int, list[tuple[ast.expr, _Message]]],
    ) -> list[SkeletonLine]:
        line = getattr(statement, "lineno", 0)
        rows: list[SkeletonLine] = []
        for argument, written in printed.get(id(statement), ()):
            # Compatibility only, not :func:`pathcond.can_reach`: a line printed after
            # the arm is printed alongside it just as much as one before, and
            # the position rule would drop exactly the lines that say what the
            # arm went on to do.
            arms = [
                arm
                for arm, held in self.armed.items()
                if pathcond.compatible(held, argument, current=True)
            ]
            if written.refused:
                if not any(
                    (row.refused, row.text) == (written.refused, _text(written.expr))
                    for row in rows
                ):
                    rows.append(
                        SkeletonLine(
                            kind=SKELETON_PRINT,
                            line=line,
                            depth=depth,
                            text=_text(written.expr),
                            refused=written.refused,
                            arms=sorted(arms),
                        )
                    )
                continue
            for message in self._expanded(frame, written.expr, argument):
                if isinstance(message, ast.JoinedStr):
                    pattern = line_pattern(message)
                    value, hole, because = self._value_hole(
                        frame, message, [self.armed[arm] for arm in arms]
                    )
                else:
                    pattern = _constant_pattern(message)
                    value, hole, because = "", "", ""
                refused = "" if pattern else _refusal(message)
                # Deduped on what the row would *say*, refusals included: one
                # call can resolve to several messages that render the same,
                # and three identical refusals on one line tell a reader
                # nothing three times.
                if any(
                    (row.pattern, row.text) == (pattern, _text(message)) for row in rows
                ):
                    continue
                rows.append(
                    SkeletonLine(
                        kind=SKELETON_PRINT,
                        line=line,
                        depth=depth,
                        text=_text(message),
                        pattern=pattern,
                        refused=refused,
                        keyed=bool(pattern) and keyed(message),
                        holes=_holes_of(message),
                        prefix=self.prefix_of(frame, argument) if pattern else "",
                        arms=sorted(arms),
                        value=value,
                        hole=hole,
                        because=because,
                    )
                )
        if line in self.armed:
            rows.append(
                SkeletonLine(
                    kind=SKELETON_ARM,
                    line=line,
                    depth=depth,
                    text=_text(statement),
                )
            )
        return rows

    def _expanded(self, frame: _Frame, message: ast.expr, at: ast.AST) -> list[ast.expr]:
        """*message* with each hole a local fills with text written out.

        ``reg_msg_str`` is assigned an f-string on 748 and another on 750, and
        ``f"{reg_msg_str} took {...} sec"`` on 775 rendered as eight characters
        of literal -- refused -- where either line production prints is forty.
        Each way the local was written is a variant of its own, since which of
        them ran is what the line says -- read by :meth:`_bound_message`, the
        same reading a message passed bare gets, so ``f"{error_message}"`` and
        ``error_message`` print the same rows.

        A hole is written out only where every way it was written can be, and
        there are few; and not where one of them is a bare token --
        ``newStatus = "exhausted"`` -- which the hole already matches and which
        is the value the row's ``value`` pattern names.  What ``+=`` added is
        never written out: it is the tail of a message, not a local of its own.
        """
        if not isinstance(message, ast.JoinedStr):
            return [message]
        choices: list[list[list[ast.expr]]] = []
        changed = False
        for part in message.values:
            if (
                isinstance(part, ast.FormattedValue)
                and isinstance(part.value, ast.Name)
                and part.conversion == -1
                and part.format_spec is None
                and not getattr(part, "appended", False)
            ):
                ways, refused = self._bound_message(frame, part.value, at, 1, frozenset())
                if (
                    ways
                    and not refused
                    and len(ways) <= _MAX_EXPANDED
                    and not any(_is_token(way) for way in ways)
                ):
                    choices.append(ways)
                    changed = True
                    continue
            choices.append([[part]])
        if not changed:
            return [message]
        variants: list[ast.expr] = []
        for combination in itertools.product(*choices):
            pieces = [piece for text in combination for piece in text]
            variants.append(ast.JoinedStr(values=_merged(pieces)))
            if len(variants) >= _MAX_EXPANDED:
                break
        return variants

    def _holds_observed(self, frame: _Frame, name: str) -> bool:
        """Whether a reaching definition in this function settles *name* to it."""
        if not isinstance(frame.func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return False
        return any(
            value == self.observed
            for value, _conditions, _line in pathcond.literal_values(frame.func, name)
        )

    def _within(
        self,
        frame: _Frame,
        bindings: list,
        loops: list,
        caught: list,
        fills: Sequence[ast.AST] = (),
    ) -> tuple[list, list, list]:
        """Keep only the bindings inside the caller's stretch, when any are.

        ``assigned_expressions`` hands back every assignment of the name in
        the function, and a read side statement owns one part of it.
        ``copyArchive.py::main`` assigns ``res = taskBuffer.querySQLS(...)``
        nine times in 1353 lines, so the exit taken on line 390 was explained
        with the assignments on lines 263 and 284 -- other queries, whose
        rows the row that stopped here never was.

        Inside *or* nothing, rather than inside and then the rest: a name with
        no binding in the stretch is one the statement did not fill, and there
        the whole function is the right place to look.  ``xmlIDs`` above the
        same exit is filled far from the statement and still explains it.

        *fills* -- ``seen[k] = []``, ``seen.add(k)`` -- are not evidence that
        the statement filled the name.  A fill adds to what the name holds and
        binds nothing anew, so one inside the stretch left ``archiveJob``'s
        ``toBeClosedSubList = {}``, built above the loop, cut away from the
        test that reads it first.

        The starting function only.  Deeper frames are other files, where
        these line numbers mean something else.
        """
        if self.within is None:
            return bindings, loops, caught
        file, owner, low, high = self.within
        if frame.file != file or frame.owner != owner:
            return bindings, loops, caught

        def here(node: ast.AST) -> bool:
            site = (
                node
                if isinstance(node, (ast.stmt, ast.ExceptHandler))
                else pathcond.site_of(node)
            )
            return low <= getattr(site, "lineno", 0) <= high

        filled = {id(node) for node in fills}
        if not any(
            here(node) for node in (*bindings, *loops, *caught) if id(node) not in filled
        ):
            return bindings, loops, caught
        return (
            [node for node in bindings if here(node)],
            [node for node in loops if here(node)],
            [node for node in caught if here(node)],
        )

    def _reaching(
        self,
        frame: _Frame,
        bindings: list,
        loops: list,
        caught: list,
        at: Optional[ast.AST] = None,
        rereads: Sequence[ast.AST] = (),
        cut: Sequence[ast.AST] = (),
    ) -> tuple[list, list, list]:
        """Drop the writes a later one overwrote before the read.

        :meth:`_within` cuts by position -- which part of the function this
        statement owns -- and that is the wrong instrument for a name the
        function reuses *in order*.  ``copyArchive.py::main`` builds
        ``timeLimit`` afresh before each of its queries, so the query on line
        381 is answered by line 360 and by nothing else; the span rule gets
        that one right and gets the query on 758 wrong, because its own
        ``timeLimit`` is on 753, one line above where its region opens.
        Execution order has no such edge.

        The two compose rather than replace one another.  Measured over the
        corpus's exits with the rule ``pathcond.surviving_writes`` used before
        it followed the function's structure, the span was the narrower answer
        34 times and the order 657 times, so neither contains the other.  *cut*
        is what the span left out: those writes still ran, so they still
        overwrite what was there before them, and are never an answer.

        The starting frame only, and the same reason as :meth:`_within`: a
        deeper frame is another function, where this line number belongs to
        another read.

        **The read is the statement that read the name** -- *at* -- and not the
        line the walk started from.  Cancelling against the starting line at
        every depth answered "which writes reach line 1019" for a name read on
        line 1007, and ``memVal = int(memVal)`` then cancelled the very writes
        it reads.  The statement itself is not its own answer, because it reads
        before it writes -- unless a loop brings its last pass's value round
        to it with nothing overwriting that on the way.
        """
        if not self.narrows(frame):
            return bindings, loops, caught
        target = at
        if target is None and self.reading_at is not None:
            target = _statement_at(frame.func, self.reading_at[2])
        if target is None:
            return bindings, loops, caught

        def site(node: ast.AST) -> ast.AST:
            if isinstance(node, (ast.stmt, ast.ExceptHandler)):
                return node
            return pathcond.site_of(node)

        sites = {id(node): site(node) for node in (*bindings, *loops, *caught)}
        # ``as_run``: a test on a name the function has since rebound says
        # nothing about the rebound value -- see ``pathcond.parted``.
        alive = {
            id(one)
            for one in pathcond.surviving_writes(
                list(sites.values()),
                target,
                rereads,
                as_run=True,
                killers=[site(node) for node in cut],
            )
        }

        def lives(node: ast.AST) -> bool:
            return id(sites[id(node)]) in alive

        kept = (
            [node for node in bindings if lives(node)],
            [node for node in loops if lives(node)],
            [node for node in caught if lives(node)],
        )
        # Never down to nothing.  A name the read saw was written by something,
        # so an empty answer is the rule having cancelled the write that
        # actually ran -- a name read above every assignment to it, or a loop
        # this walk cannot see back into.  Saying "bound nowhere" there would
        # be false, and the honest fallback is the unfiltered set.
        if not any(kept):
            return bindings, loops, caught
        return kept

    def explain(
        self, frame: _Frame, name: str, depth: int, at: Optional[ast.AST] = None
    ) -> None:
        if name.startswith("self."):
            self._field(frame, name, depth)
            return
        if name.endswith("]"):
            self._keyed(frame, name, depth, at)
            return
        bindings = list(pathcond.assigned_expressions(frame.func).get(name, []))
        unpacked = _tuple_bindings(frame.func, name)
        slots = {id(node.value): slot for node, slot in unpacked}
        bindings.extend(node.value for node, _slot in unpacked)
        bindings.extend(_with_bindings(frame.func, name))
        bindings.extend(_walrus_bindings(frame.func, name))
        # Last, and only where the assignments leave the reader nowhere: a
        # container that is assigned a value has that value to answer with,
        # and the members would be noise beside it.
        mutated: list[ast.expr] = []
        if bindings and all(_is_empty_container(node) for node in bindings):
            mutated = _mutated_bindings(frame.func, name)
            bindings.extend(mutated)
        loops = _loop_bindings(frame.func, name)
        caught = _caught_as(frame.func, name)
        every = (*bindings, *loops, *caught)
        bindings, loops, caught = self._within(frame, bindings, loops, caught, mutated)
        inside = {id(node) for node in (*bindings, *loops, *caught)}
        bindings, loops, caught = self._reaching(
            frame,
            bindings,
            loops,
            caught,
            at,
            [pathcond.site_of(node) for node in mutated],
            [node for node in every if id(node) not in inside],
        )
        held = bool(bindings or loops or caught)
        # Asked before, from another statement: only what this one reaches and
        # the first did not.  The fallbacks below answered the first asking
        # and would say the same thing again.
        done = self.explained.get((f"{frame.file}::{frame.owner}", name))
        again = done is not None
        if depth == 1 and self.armed:
            # ``assigned_expressions`` hands back every assignment in the
            # function and the arms are at one point in it.  Pruning is by
            # elimination, so a site that cannot have run on the way to any of
            # them does not merely add noise: it dilutes "one of these three"
            # into "one of these four" and sends a reader down a path the code
            # did not take.  Measured on the 41-case sample, eight of 287
            # steps.
            #
            # Depth one only, and that is a limit rather than a choice.  A
            # name is explained once for every place that reads it, so deeper
            # than this the reading point is whichever want arrived first and
            # filtering against it would drop sites the other readers can
            # reach.  At depth one the readers are the arms, all of them
            # armed before the queue drains, and a site is kept if it reaches
            # any.
            arms = list(self.armed.values())
            bindings = [b for b in bindings if pathcond.reaches_one(pathcond.site_of(b), arms)]
            loops = [loop for loop in loops if pathcond.reaches_one(loop, arms)]
            caught = [one for one in caught if pathcond.reaches_one(one, arms)]
            if held and not (bindings or loops or caught):
                # Bound here, and by nothing that could have run first.
                # "Not bound in this function" would be false and silence
                # would leave the guard unexplained, so it says which.
                if not again:
                    self._unbound(frame, name, depth, only_below=True)
                return
        if again:
            bindings = [node for node in bindings if _walked_key(node) not in done]
            loops = [node for node in loops if _walked_key(node) not in done]
            caught = [node for node in caught if _walked_key(node) not in done]
            if not (bindings or loops or caught):
                return
        else:
            done = self.explained.setdefault((f"{frame.file}::{frame.owner}", name), set())
        done.update(_walked_key(node) for node in (*bindings, *loops, *caught))
        if not bindings and not loops and not caught:
            if any(
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
                for node in ast.walk(frame.func)
                if node is not frame.func
            ):
                # A function defined here is a name for code, not a value --
                # the same reason imports are dropped.
                return
            # A closure reads what the function it is nested in bound:
            # ``build_where_clause`` is ``getJobs``'s, and its ``mem`` is the
            # parameter ``getJobs`` was called with, not a module name.
            if not _is_parameter(frame.func, name):
                enclosing = self.types.enclosing_function(frame.file, frame.func)
                if enclosing is not None:
                    outer_frame = self.frame(
                        frame.file,
                        f"{frame.file}::{getattr(enclosing, 'name', '')}",
                        line=enclosing.lineno,
                    )
                    if outer_frame is not None:
                        self.want(outer_frame, [name], depth)
                        return
            # The file's own body is the last place to look.  Asked only here,
            # after the function has had its say, so a local always wins over a
            # module constant of the same name.
            parsed = self.module(frame.file)
            outer = _module_bindings(parsed[1], name) if parsed is not None else []
            if outer:
                for expression in outer:
                    self._module_binding(frame, name, expression, depth)
                return
            self._unbound(frame, name, depth)
            return
        # In the order the file states them.  ``ast.walk`` is breadth-first, so
        # unsorted it prints ``copyArchive.py``'s ``timeOutVal *= 60`` before
        # the ``timeOutVal = 48`` nested one level in above it, and the default
        # reads as if it were applied after the multiply.
        for expression in sorted(
            bindings, key=lambda node: getattr(pathcond.site_of(node), "lineno", 0)
        ):
            self._binding(frame, name, expression, depth, slots.get(id(expression)))
        for loop in loops:
            self._loop(frame, name, loop, depth)
        for handler in caught:
            self._caught(frame, name, handler, depth)

    def _binding(
        self,
        frame: _Frame,
        name: str,
        expression: ast.expr,
        depth: int,
        slot: Optional[int] = None,
    ) -> None:
        site = pathcond.site_of(expression)
        resolve = self.resolver(frame)
        typed = self.typer(frame)
        terminal, detail = self._classified(frame, expression, resolve, typed)
        unseen = _unseen(site)
        inner = _nested_scope(frame.func, site)
        if inner is not None:
            unseen.append(
                f"{NESTED_SCOPE} at {inner.lineno}: inside "
                f"{getattr(inner, 'name', 'a lambda')}, which runs when it is "
                f"called and not on the way here"
            )
        if not terminal and any(
            entry.startswith(pathcond.UNSEEN_EXCEPT) for entry in unseen
        ):
            terminal, detail = _raised_in(site, self.classify, resolve, typed)
        reads = _reads(expression)
        own = list(reads)
        for read in _guard_reads(site):
            _add(reads, read)
        if (
            not detail
            and isinstance(expression, ast.Call)
            and isinstance(expression.func, ast.Name)
            and expression.func.id in ("max", "min")
            and len(expression.args) > 1
        ):
            # ``max(task.ramCount, ram_count)``: either can be the value, and
            # which one was is the question a sieve that rejected a site asks.
            detail = (
                f"the {'larger' if expression.func.id == 'max' else 'smaller'} of "
                f"{len(expression.args)} values, and each is walked"
            )
        step = TraceStep(
            kind=TRACE_BINDING,
            name=name,
            owner=frame.owner,
            file=frame.file,
            line=getattr(site, "lineno", 0),
            value=_text(expression) if slot is None else f"{_text(expression)}[{slot}]",
            operator=_operator(site),
            guards=pathcond.path_condition(site),
            unseen=unseen,
            reads=reads,
            terminal=terminal,
            detail=detail,
            depth=depth,
        )
        if not self.record(step):
            return
        onwards = not terminal or terminal in _WALKED_THROUGH
        if onwards and not self.narrows(frame):
            self.want(frame, reads, depth + 1)
        elif onwards:
            # Each name from the statement that read it: the value's from
            # this line, a guard's from its ``if``.
            self.want(frame, own, depth + 1, site)
            for test, names in _guard_tests(site):
                self.want(frame, names, depth + 1, test)
        if not terminal or terminal == STOP_EXTERNAL:
            self.beyond.append(
                lambda: self._enter(frame, step, expression, slot, depth + 1)
            )
        if onwards:
            self._enter_tests(frame, site, depth + 1)
            self._ask_keyed(frame, expression, depth + 1, site)
        if (
            isinstance(site, ast.AugAssign)
            and isinstance(site.target, ast.Name)
            and site.target.id == name
            and self.narrows(frame)
        ):
            # ``memVal /= 1024`` reads the ``memVal`` it replaces, and the
            # expression this step was built from is ``1024`` alone -- so the
            # read is asked for here, or the write above it is never reached.
            # Whatever the right-hand side resolved to: that stops the walk
            # along *it*, not along the value being divided.
            self.want(frame, [name], depth + 1, site)

    def _ask_keyed(self, frame: _Frame, expression: ast.AST, depth: int, at: ast.AST) -> None:
        """Ask, beyond the function's own answer, for each item read by a constant key.

        ``scoutData["ramCount"]`` is one entry of what ``getScoutJobData_JEDI``
        returned, and the name ``scoutData`` answers it only as "the whole
        map": the entry is asked for as itself, after the name has been.
        """
        for keyed in _keyed_reads(expression):
            self.beyond.append(
                lambda one=keyed: self.want(frame, [one], depth, at)
            )

    def _keyed(self, frame: _Frame, name: str, depth: int, at: Optional[ast.AST]) -> None:
        """Explain ``x['k']``: what this function puts there, or what the call ``x`` came from returns there."""
        try:
            node = ast.parse(name, mode="eval").body
        except SyntaxError:
            return
        if not (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)):
            return
        base, key = node.value.id, getattr(node.slice, "value", None)
        written = [
            one.value
            for one in _own_nodes(frame.func)
            if isinstance(one, (ast.Assign, ast.AugAssign, ast.AnnAssign))
            and one.value is not None
            and any(
                _text(target) == name
                for target in (one.targets if isinstance(one, ast.Assign) else [one.target])
            )
        ]
        written, _loops, _caught = self._reaching(frame, written, [], [], at)
        for value in sorted(written, key=lambda v: getattr(pathcond.site_of(v), "lineno", 0)):
            self._binding(frame, name, value, depth)
        if written:
            return
        # Nothing here writes the entry: the whole of ``x`` came from a call, so
        # the entry is what that call returns under the key.
        for value in pathcond.assigned_expressions(frame.func).get(base, []):
            if isinstance(value, ast.Call):
                self.beyond.append(lambda call=value: self._enter(frame, None, call, (key,), depth))
            elif isinstance(value, ast.Dict):
                for held, item in zip(value.keys, value.values, strict=True):
                    if isinstance(held, ast.Constant) and held.value == key:
                        self._binding(frame, name, item, depth)
        for assign, slot in _tuple_bindings(frame.func, base):
            if slot is not None and isinstance(assign.value, ast.Call):
                self.beyond.append(
                    lambda call=assign.value, where=slot: self._enter(frame, None, call, (where, key), depth)
                )

    def _enter_tests(self, frame: _Frame, site: ast.AST, depth: int, above: bool = True) -> None:
        """Go into the calls a guard above *site* makes, for the value they return.

        ``if self.add_result is None or not self.add_result.is_succeeded():``
        decides the job fails, and what decides ``is_succeeded`` is inside it.
        The names the test reads are walked where the test reads them; the
        calls it makes are entered, after the function's own answer, like a
        value bound from a call.

        ``_guard_tests`` hands back each ``if`` itself, the statement its names
        are read at.  Its test is what decides: walking the whole ``if`` entered
        every call in its body as though the guard made it -- over fifty
        thousand entries for one arm of ``insertFilesForDataset_JEDI``, whose
        returns then filled the allowance beyond the function.
        """
        tests = [guard.test for guard, _names in _guard_tests(site)] if above else []
        if isinstance(site, (ast.If, ast.While)):
            tests.append(site.test)
        for test in tests:
            for node in ast.walk(test):
                if isinstance(node, ast.Call) and not (
                    isinstance(node.func, ast.Name) and node.func.id in _BUILTINS
                ):
                    self.beyond.append(
                        lambda call=node: self._enter(frame, None, call, None, depth)
                    )

    def _block_prints(self, frame: _Frame, statement: ast.AST) -> list[str]:
        """Patterns of the lines printed on the way to *statement*, nearest first.

        The block that holds it, both sides of it; where that prints nothing,
        the enclosing blocks, but only what they print *before* the statement
        that leads here -- a line after it may belong to the path that did not
        come this way.  ``register_files`` calls ``set_fatal()`` three blocks
        below the ``self.logger.error(f"{error_message}")`` that says why.
        """
        rendered: dict[int, list[str]] = {}
        for argument, message in self.messages(frame):
            enclosing = _enclosing_statement(argument)
            if enclosing is None or message.refused:
                continue
            # Written out the way the skeleton writes it, holes filled from
            # the locals that fill them.
            for one in self._expanded(frame, message.expr, argument):
                pattern = (
                    line_pattern(one)
                    if isinstance(one, ast.JoinedStr)
                    else _constant_pattern(one)
                )
                if pattern:
                    rendered.setdefault(id(enclosing), []).append(pattern)
        below = statement
        first = True
        while below is not None and not isinstance(below, (ast.FunctionDef, ast.AsyncFunctionDef)):
            parent = getattr(below, "parent", None)
            if parent is None:
                break
            if isinstance(parent, ast.ExceptHandler):
                blocks = [("except", parent.body)]
            elif isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
                blocks = [("def", parent.body)]
            elif isinstance(parent, ast.stmt):
                blocks = _blocks_of(parent)
            else:
                below = parent
                continue
            for _header, body in blocks:
                if below not in body:
                    continue
                index = body.index(below)
                candidates = body if first else body[:index]
                found: list[str] = []
                for one in candidates:
                    for pattern in rendered.get(id(one), []):
                        if pattern not in found:
                            found.append(pattern)
                if found:
                    return found
            first = False
            below = parent
        return []

    def _mutated_at(
        self, method_frame: _Frame, field: str, depth: int, said: Optional[TraceStep] = None
    ) -> None:
        """Where a method that sets ``self.<field>`` was called, and why it was.

        ``status_code`` is set in ``set_fatal``, ``set_temporary`` and
        ``set_succeeded`` -- true, and not the answer: the answer is which of
        the thirteen ``self.result.set_fatal()`` calls in the plugin ran.  Each
        call placed at the method is a step, and the guards above it are
        walked in the function that makes it.

        Only the calls made on the object the walk came in on, where that is
        known (:meth:`_on_the_object`): ``ListWithLock.append`` sets the
        ``dataList`` a ``TaskRefinerThread`` reads, and its two calls append to
        a broker's ``inputListWorld`` and to ``JobGeneratorThread``'s
        ``brokerageLockIDs`` -- other lists, whose callers the walk followed
        into ``JobGeneratorThread.runImpl``.  The calls left out are counted;
        where none is left, *said* -- the step of what the method assigns --
        says so.

        Measured on the corpus walks against f82008d, constructors and setters
        together: 266 of 6,647 change and 14,957 steps go, each behind a call
        left out -- most of them the way into ``JobGeneratorThread.runImpl``
        through those two appends; none of the 429 silent-scan reads changes,
        and the six answers keep their conclusions.
        """
        cls = self.types.method_class(method_frame.file, method_frame.func)
        target = calls_mod.Target(
            method_frame.file, method_frame.func, cls[1] if cls else None
        )
        sites = self.calls.callers(target)
        sites, left, made = self._on_the_object(method_frame, sites)
        before = len(self.steps)
        if not self._mutations(method_frame, field, depth, sites[:_MAX_MUTATIONS]):
            return
        name = getattr(method_frame.func, "name", "")
        if left:
            step = said
            if len(self.steps) > before:
                step = self.steps[-1]
                step.detail += f"; {len(left)} call(s) of {name} on other objects not walked"
            elif step is not None:
                step.detail += f"; none of the {len(left)} call(s) of {name} is made on the object the walk came in on"
            if step is not None:
                self._narrowed(
                    method_frame, made, step, left,
                    lambda more: self._mutations(method_frame, field, depth, more),
                    _MAX_MUTATIONS - len(sites),
                )
        if len(sites) > _MAX_MUTATIONS and self.steps:
            self.steps[-1].detail += f"; {len(sites) - _MAX_MUTATIONS} more call(s) of {name} not walked"

    def _mutations(self, method_frame: _Frame, field: str, depth: int, sites: list) -> bool:
        """A handover step for each call in *sites* of the method that sets *field*; ``False`` once out of steps."""
        for site in sites:
            if site.in_module:
                caller = self._module_frame(site.file)
            else:
                caller = self.frame(
                    site.file, f"{site.file}::{site.func.name}", line=site.func.lineno  # type: ignore[attr-defined]
                )
            if caller is None:
                continue
            call = _same_node(caller.func, site.call)
            if not isinstance(call, ast.Call):
                continue
            statement = _enclosing_statement(call) or call
            guards = _guard_tests(statement)
            step = TraceStep(
                kind=TRACE_HANDOVER,
                name=f"self.{field}",
                owner=caller.owner,
                file=caller.file,
                line=call.lineno,
                value=_text(call),
                guards=pathcond.path_condition(statement),
                unseen=_unseen(statement),
                reads=[name for _test, names in guards for name in names],
                detail=(
                    f"calls {getattr(method_frame.func, 'name', '')} here, "
                    f"which sets {field}"
                ),
                depth=depth,
                printed=[] if site.in_module else self._block_prints(caller, statement),
                open=self.calls.opened(site),
            )
            if site.in_module:
                self._at_module(caller, step, statement, None)
            if not self.record(step):
                return False
            if site.in_module:
                continue
            self.entered.add((caller.file, caller.owner))
            for test, names in guards:
                self.want(caller, names, depth + 1, test)
        return True

    def _enter(
        self,
        frame: _Frame,
        step: Optional[TraceStep],
        expression: ast.expr,
        slot: Optional[int],
        depth: int,
    ) -> None:
        """Go into the call a value was bound from, and explain what it returns.

        The walk stopped at ``pandaIDs = self.tb.getPandaIDsWithTask_JEDI(...)``
        and at ``tmp_where_clause, tmp_val_map = self.construct_where_clause(...)``
        as though the call were the value.  The value is what the call hands
        back: each ``return`` of each function the call is placed at is a step,
        in that function's terms, and the names it reads are asked there --
        its parameters answered by the call that brought the walk in.

        Where the call cannot be placed the step says why, so a walk that ends
        here says what would carry it further.  Where it is placed open -- at
        these functions and possibly somewhere this reading cannot name -- each
        is entered all the same, and every return step of it says so
        (:attr:`TraceStep.open`): the value may come from somewhere else.
        """
        call = expression
        path: tuple = slot if isinstance(slot, tuple) else ((slot,) if slot is not None else ())
        if isinstance(call, ast.Subscript) and isinstance(call.slice, ast.Constant):
            path, call = (call.slice.value, *path), call.value
        if not isinstance(call, ast.Call):
            return
        if isinstance(call.func, ast.Name) and call.func.id in _BUILTINS:
            return
        if self.terminal_of(frame, call)[0] in (STOP_CONFIG, STOP_UPSTREAM):
            # The database's or a configuration's: where the walk stops is the
            # answer, and inside the wrapper is not -- the statement's table and
            # key are said on the step instead.  Another system's interface is
            # still PanDA's code on this side: ``SiteMapper.getSite`` resolves
            # ``nucleus:UNI-FREIBURG`` to one of its sites before CRIC's
            # declaration is read, and that resolution is gone into.
            return
        placed = self.calls.callees(frame.file, frame.func, call)
        if not placed.targets:
            if placed.unplaced and step is not None and not step.terminal:
                step.detail = f"not entered: {placed.unplaced}"
            return
        entered: list[str] = []
        for target in placed.targets:
            callee = self._callee_frame(target)
            if callee is None:
                continue
            if placed.unplaced:
                self._entered_open(callee, placed.unplaced)
            # One entry per call, however many ways the walk arrived at it: the
            # ten statements under one ``if`` each enter the test's
            # ``JediTaskSpec.statusToIncexec()``, and a parameter read there was
            # handed over once for each of them.
            held = self.passed.setdefault((callee.file, callee.owner), [])
            if not any(one is call for _caller, one, _bound in held):
                held.append((frame, call, calls_mod.bind(call, target, placed.on_class)))
                for narrowed in self.watched.get((callee.file, callee.owner), ()):
                    self._came_later(narrowed, frame, call)
            self.entered.add((callee.file, callee.owner))
            entered.append(callee.owner)
            # Once per function and way of taking its value: a second call of
            # ``percentile`` hands its arguments in beside the first's, and
            # what it returns is the same statements either way.
            taken = (callee.file, callee.owner, path)
            if taken in self.returned:
                continue
            self.returned.add(taken)
            for returned in calls_mod.returns(callee.func):
                value, pending = returned, list(path)
                while pending:
                    taken = _projected(value, pending[0])
                    if taken is value:
                        break
                    value, pending = taken, pending[1:]
                statement = pathcond.site_of(returned) if hasattr(returned, "parent") else callee.func
                terminal, detail = self.terminal_of(callee, value)
                text = _text(value)
                if pending:
                    text = f"({text})" + "".join(f"[{one!r}]" for one in pending)
                reads = _reads(value)
                if not self.record(
                    TraceStep(
                        kind=TRACE_RETURN,
                        name=f"{getattr(callee.func, 'name', 'call')}()",
                        owner=callee.owner,
                        file=callee.file,
                        line=getattr(statement, "lineno", 0),
                        value=text,
                        guards=pathcond.path_condition(statement),
                        unseen=_unseen(statement),
                        reads=reads,
                        terminal=terminal,
                        detail=detail,
                        depth=depth,
                        open=self.entered_open.get((callee.file, callee.owner), ""),
                    )
                ):
                    return
                if not terminal or terminal in _WALKED_THROUGH:
                    self.want(callee, reads, depth + 1, statement)
                    for test, names in _guard_tests(statement):
                        self.want(callee, names, depth + 1, test)
                    if pending and isinstance(value, ast.Name):
                        # ``return True, returnMap, extraInfo`` taken at slot 1
                        # and then read at ``["ramCount"]``: the entry.
                        self.want(callee, [f"{value.id}[{pending[0]!r}]"], depth + 1, statement)
                    self._ask_keyed(callee, value, depth + 1, statement)
        if entered and step is not None and not step.terminal:
            step.detail = f"returns from {', '.join(entered)}" + (f"; {placed.unplaced}" if placed.unplaced else "")

    def _entered_open(self, callee: _Frame, reason: str) -> None:
        """Say on each return step of *callee* that some call that entered it was placed open.

        Of the function, whichever call the walk came in by: its returns are
        recorded once, by the first call to enter it, and that may be a call
        placed closed -- so the steps already recorded are marked as well,
        and the marks do not depend on the order the walk took.
        """
        key = (callee.file, callee.owner)
        if key in self.entered_open:
            return
        self.entered_open[key] = reason
        for one in self.steps:
            if one.kind == TRACE_RETURN and (one.file, one.owner) == key:
                one.open = reason

    def _callee_frame(self, target: calls_mod.Target) -> Optional[_Frame]:
        """This walk's own frame for a function a call was placed at."""
        owner = f"{target.file}::{target.func.name}"  # type: ignore[attr-defined]
        held = self.frames.get((target.file, owner))
        line = target.func.lineno  # type: ignore[attr-defined]
        if held is not None and held.func.lineno != line:
            # Two functions of one file with one name: a method of each of two
            # classes.  Spelled with the class so the second is not the first.
            owner = f"{target.file}::{target.cls.name if target.cls else 'module'}.{target.func.name}"  # type: ignore[attr-defined]
        return self.frame(target.file, owner, line=line)

    def _passed_in(self, frame: _Frame, name: str, depth: int) -> bool:
        """Answer a parameter from the call that brought the walk into *frame*."""
        held = self.passed.get((frame.file, frame.owner))
        if not held:
            return False
        cls = self.types.method_class(frame.file, frame.func)
        target = calls_mod.Target(frame.file, frame.func, cls[1] if cls else None)
        for caller, call, bound in held:
            step, statement = self._handed_step(caller, call, target, name, bound, depth)
            if not self.record(step):
                return True
            if step.reads:
                self.want(caller, step.reads, depth + 1, statement)
        return True

    def _handed_step(
        self,
        caller: _Frame,
        call: ast.Call,
        target: "calls_mod.Target",
        name: str,
        bound: dict,
        depth: int,
    ) -> tuple[TraceStep, ast.AST]:
        """The step saying what *call* hands the parameter *name*, and the statement to walk from.

        Four answers, and only the first is "passed here": an argument the
        call writes -- ``self`` too, where the method is read off its class,
        ``AdderPluginBase.__init__(self, job, ...)``; the default the
        signature holds, which the call leaves alone and whose names are the
        callee's module's, not the caller's; the object a method is called
        on, for ``self`` and ``cls`` -- the caller's own for
        ``super().__init__(...)``, not ``super()``; and an argument hidden in
        ``*``/``**``.
        """
        callee = getattr(target.func, "name", "")
        statement = _enclosing_statement(call) or call
        argument = bound.get(name)
        on_class = self.calls.callees(caller.file, caller.func, call).on_class
        if argument is not None and name in calls_mod.said(call, target, on_class):
            value, reads, detail = _text(argument), _reads(argument), f"passed to {callee} here"
        elif argument is not None:
            # Evaluated where the ``def`` is, once: nothing on the path to
            # this call chose it, so there is nothing further to walk.
            value, reads, detail = _text(argument), [], f"left to the default {callee}'s signature gives it"
        elif name == calls_mod.receiver_parameter(target) and isinstance(call.func, ast.Attribute):
            receiver = call.func.value
            own = _own_object(caller.func)
            if _is_super(receiver) and own is not None:
                receiver = ast.Name(id=own, ctx=ast.Load())
            value = _text(receiver)
            reads = _reads(receiver)
            detail = f"the object {callee} is called on"
        else:
            value, reads, detail = name, [], "passed through * or ** here, so which argument it is is not said"
        step = TraceStep(
            kind=TRACE_HANDOVER,
            name=name,
            owner=caller.owner,
            file=caller.file,
            line=getattr(call, "lineno", 0),
            value=value,
            guards=pathcond.path_condition(statement),
            unseen=_unseen(statement),
            reads=reads,
            detail=detail,
            depth=depth,
        )
        return step, statement

    def _callers_beside(self, frame: _Frame, spanned: frozenset[str]) -> list:
        """The call sites ``calls`` places at *frame*'s function outside the functions named *spanned*."""
        cls = self.types.method_class(frame.file, frame.func)
        target = calls_mod.Target(frame.file, frame.func, cls[1] if cls else None)
        return [
            site
            for site in self.calls.callers(target) or []
            if getattr(site.func, "name", "") not in spanned
        ]

    def _called_from(
        self,
        frame: _Frame,
        name: str,
        depth: int,
        stopped: TraceStep,
        spanned: frozenset[str] = frozenset(),
    ) -> bool:
        """Answer a parameter from the calls placed at this function, where nothing brought the walk in.

        The parameter is chosen at the call site, and until now that was the
        whole answer.  ``calls.callers`` names the sites; each is a step with
        the expression it hands over, walked in the caller.  At most
        :data:`_MAX_CALLERS` of them, and the rest counted, so a helper called
        from everywhere does not turn one question into sixty.

        *spanned* are the facade methods a map handover already crossed to
        answer this parameter; their calls are that handover's, not others.

        Whatever the step says, it says too why the calls placed here may not
        be every call -- :meth:`Calls.missed`, each after ``"; "`` -- so "a
        parameter, so its value is chosen at the call site" with no caller
        says whether the function is an endpoint or called where nothing
        places the call, and "followed to the 3 call(s)" says when more calls
        by the name were not placed.
        """
        cls = self.types.method_class(frame.file, frame.func)
        target = calls_mod.Target(frame.file, frame.func, cls[1] if cls else None)
        missing = "".join(f"; {one}" for one in self.calls.missed(target))
        sites = self._callers_beside(frame, spanned)
        if not sites:
            stopped.detail += missing
            return False
        every = sites
        made: Optional[set[tuple[str, int, int]]] = None
        left: list = []
        asking: list[tuple[str, ast.ClassDef]] = []
        unbuilt: list = []
        if not spanned and getattr(frame.func, "name", "") == "__init__":
            # The object is one of the classes whose methods read the field,
            # or of a subclass: the chain from a sibling's constructor builds
            # another object.
            classes = self._asking_classes(frame)
            if classes is not None:
                kept = [site for site in sites if self._may_build(site, classes)]
                if len(kept) < len(sites):
                    asking, unbuilt, sites = classes, [site for site in sites if site not in kept], kept
            made = self._object_made(frame)
            if made is not None:
                kept = [site for site in sites if _site_key(site) in made]
                if kept and len(kept) < len(sites) and len(kept) <= _MAX_CALLERS:
                    left = [site for site in sites if _site_key(site) not in made]
                    sites = kept
        elif not spanned and (frame.file, getattr(frame.func, "lineno", 0)) in self.asked_of:
            # A setter's parameter is chosen by the calls made on the object.
            sites, left, made = self._on_the_object(frame, sites)
        called = getattr(frame.func, "name", "")
        if left and not sites:
            stopped.detail = (
                f"a parameter; none of the {len(every)} call(s) placed at "
                f"{called} is made on the object the walk came in on"
            )
        elif unbuilt and not sites:
            stopped.detail = (
                f"a parameter; none of the {len(every)} call(s) placed at {called} can build {_objects_of(asking)}"
            )
        elif len(sites) > _MAX_CALLERS:
            # A function built or called from everywhere -- ``SiteMapper`` is
            # constructed by every daemon -- has no one caller to answer with,
            # and the first eight would be as arbitrary as any.  Said, not walked.
            stopped.detail = (
                f"a parameter; {len(sites)} of the {len(every)} calls placed at {called} "
                f"can build {_objects_of(asking)}, too many to follow"
                if unbuilt
                else f"a parameter; {len(sites)} calls are placed at {called}, too many to follow"
            ) + missing
            return True
        elif left:
            stopped.detail = _built_detail(frame, len(sites), len(every))
        elif unbuilt:
            stopped.detail = (
                f"a parameter; followed to the {len(sites)} of {len(every)} call(s) placed at "
                f"{called}: the ones that can build {_objects_of(asking)}"
            )
        else:
            stopped.detail = (
                f"a parameter; followed to the {len(sites)} call(s) placed at "
                f"{getattr(frame.func, 'name', '')}"
                + (f" besides the handover through {', '.join(sorted(spanned))}" if spanned else "")
            )
        stopped.detail += missing
        if left:
            self._narrowed(
                frame, made, stopped, left,
                lambda more: self._follow_callers(target, name, depth, more, stopped),
                _MAX_CALLERS - len(sites),
            )
        if unbuilt:
            self.by_class.setdefault((frame.file, getattr(frame.func, "lineno", 0)), []).append(
                _ByClass(
                    asking, unbuilt,
                    lambda more: self._follow_callers(target, name, depth, more, stopped),
                    _MAX_CALLERS - len(sites), stopped,
                )
            )
        self._follow_callers(target, name, depth, sites, stopped)
        return True

    def _asking_classes(self, frame: _Frame) -> Optional[list[tuple[str, ast.ClassDef]]]:
        """The classes whose methods asked a field *frame* assigns; ``None`` unless every one of them is known."""
        askers = self.asked_of.get((frame.file, getattr(frame.func, "lineno", 0)))
        if not askers:
            return None
        found: list[tuple[str, ast.ClassDef]] = []
        for asker in askers:
            held = self.types.method_class(asker.file, asker.func)
            if held is None:
                return None
            if not any(one[1] is held[1] for one in found):
                found.append(held)
        return found

    def _may_build(self, site, classes: list[tuple[str, ast.ClassDef]]) -> bool:
        """Whether *site*, a call placed at an ``__init__``, can be building an object of one of *classes*.

        A construction builds the class it names, and that is one of
        *classes* or a subclass of one.  A chain from a constructor --
        ``super().__init__(...)``, ``Base.__init__(self, ...)`` -- runs on an
        object whose class has both the class the chain is made in and one of
        *classes* among its bases: ``AdderAtlasPlugin``'s chain to
        ``AdderPluginBase`` builds no ``AdderDummyPlugin``.  A call whose
        class is not known is kept.
        """
        called = site.call.func
        if isinstance(called, ast.Attribute) and called.attr == "__init__":
            held = self.types.method_class(site.file, site.func)
            return held is None or any(self._one_object(held, one) for one in classes)
        built = self.types.class_named(site.file, called)
        return built is None or any(
            any(cls is one[1] for _where, cls in self.types.lineage(*built)) for one in classes
        )

    def _one_object(self, one: tuple[str, ast.ClassDef], other: tuple[str, ast.ClassDef]) -> bool:
        """Whether a corpus class has both *one* and *other* among its lineage, itself included."""
        return any(
            any(cls is other[1] for _where, cls in self.types.lineage(file, held))
            for file, held in [one, *self.calls.subclasses(*one)]
        )

    def _follow_callers(
        self, target: "calls_mod.Target", name: str, depth: int, sites: list, stopped: TraceStep
    ) -> None:
        """A step for each of *sites* saying what it hands *name*, and the names that reads asked in its caller.

        A site the walk does not take is said on *stopped*, whose "followed to
        the N call(s)" counted it: where the walk's budget ran out before the
        caller's frame was made or its step recorded, or the caller could not
        be read.

        A call in a module's own body is where the value is written: its step
        says so, as a binding at module scope does (:meth:`_module_binding`),
        and what the argument reads is not walked.
        """
        unwalked: dict[str, list[str]] = {}
        for index, site in enumerate(sites):
            if site.in_module:
                spent = False
                caller = self._module_frame(site.file)
            else:
                owner = f"{site.file}::{site.func.name}"  # type: ignore[attr-defined]
                spent = (site.file, owner) not in self.frames and len(self.frames) >= self._functions_limit()
                caller = self.frame(site.file, owner, line=site.func.lineno)  # type: ignore[attr-defined]
            call = _same_node(caller.func, site.call) if caller is not None else None
            if not isinstance(call, ast.Call):
                why = (
                    "the walk's functions budget ran out" if caller is None and spent
                    else "its function is not read" if caller is None
                    else "the call is not found in the function the walk read"
                )
                unwalked.setdefault(why, []).append(f"{site.file}:{site.call.lineno}")
                continue
            bound = calls_mod.bind(call, target, self.calls.callees(caller.file, caller.func, call).on_class)
            step, statement = self._handed_step(caller, call, target, name, bound, depth)
            step.open = self.calls.opened(site)
            if site.in_module:
                self._at_module(caller, step, statement, bound.get(name))
            if not self.record(step):
                unwalked.setdefault("the walk's steps budget ran out", []).extend(
                    f"{one.file}:{one.call.lineno}" for one in sites[index:]
                )
                break
            if step.reads and not site.in_module:
                self.entered.add((caller.file, caller.owner))
                self.want(caller, step.reads, depth + 1, statement)
        for why, where in unwalked.items():
            stopped.detail += f"; {len(where)} of them not walked, {why}: {', '.join(where)}"

    def _module_frame(self, file: str) -> Optional[_Frame]:
        """The module's own body, to read a call in it: not entered, so it spends none of the functions budget.

        Its file is parsed as any caller's is, and a file not read yet spends
        one of the files budget: on the new pin, ``activateJob``'s walk ran out
        of files beyond the function once module-scope callers were read, and
        14 steps of ``killJob`` and ``archiveJob`` went (X8).
        """
        parsed = self.module(file)
        if parsed is None:
            return None
        tree = parsed[1]
        return _Frame(file, f"{file}::{MODULE_SCOPE}", tree, (), _imported_names(tree))

    def _at_module(self, module: _Frame, step: TraceStep, statement: ast.AST, argument: Optional[ast.expr]) -> None:
        """Mark *step*, made at a call in *module*'s own body, as written there and run at import or as a script.

        What the argument reads is the module's, and the walk does not go on
        into it -- as for a binding at module scope.  A value it can classify
        says so, as that binding's does.
        """
        script = any(
            isinstance(one, ast.If) and _is_main_test(one.test) for one in pathcond.ancestors(statement)
        )
        step.unseen.append(
            f"{MODULE_SCOPE} at {step.line}: "
            + ("run when the file is run as a script" if script else "run when the file is imported")
            + ", not on the way here"
        )
        if step.reads and argument is not None:
            step.terminal, detail = self.terminal_of(module, argument)
            if detail:
                step.detail += f" -- {detail}"

    def _narrowed(
        self,
        frame: _Frame,
        made: Optional[set[tuple[str, int, int]]],
        step: TraceStep,
        left: list,
        follow: Callable[[list], Any],
        room: int,
    ) -> None:
        """Watch the frames that asked into *frame* for objects that come in later; see :meth:`_came_later`."""
        if made is None:
            return
        if getattr(frame.func, "name", "") == "__init__":
            matches = lambda site, built: _site_key(site) in built  # noqa: E731
        else:
            matches = lambda site, built: bool(  # noqa: E731
                (self._made_at(site.file, site.func, site.call.func.value) or set()) & built
            )
        narrowed = _Narrowed(made, left, matches, follow, max(room, 0), step)
        key = (frame.file, getattr(frame.func, "lineno", 0))
        self.narrowings.setdefault(key, []).append(narrowed)
        for asker in self.asked_of.get(key, []):
            self._watch(narrowed, asker)

    def _watch(self, narrowed: _Narrowed, asker: _Frame) -> None:
        held = self.watched.setdefault((asker.file, asker.owner), [])
        if any(one is narrowed for one in held):
            return
        held.append(narrowed)
        for caller, call, _bound in self.passed.get((asker.file, asker.owner), []):
            self._came_later(narrowed, caller, call)

    def _came_later(self, narrowed: _Narrowed, frame: _Frame, call: ast.Call) -> None:
        """A call into a frame that asked brought another object: follow the left-out calls that are its.

        The walk keeps one frame per function, so ``Box.take`` entered on
        ``self.box`` and, from another method, on ``self.spare`` asks for
        ``self.items`` once, and the calls were chosen for the objects that
        had come in by then.  The later object's are followed as well.  Where
        its construction is not found, the step says so and nothing more is
        followed: any of the calls could be its.  Nothing is said for an
        object built where one already followed was -- ``setupper``'s
        ``self.site_mapper`` comes into ``SiteMapper.getSite`` again at line
        1701, and it is the one ``run`` built at line 116.

        9e5e0f7 followed later objects for constructors alone, and the object
        it followed had come in through a setter call made on another list
        (:meth:`_mutated_at`), which is what made it wrong.
        """
        where = (frame.file, getattr(call, "lineno", 0))
        if where in narrowed.said:
            return
        objects = self._call_objects(frame, call, 0)
        made: Optional[set[tuple[str, int, int]]] = None
        if objects is not None:
            made = set()
            for file, func, expression in objects:
                more = self._made_at(file, func, expression)
                if more is None:
                    made = None
                    break
                made |= more
        if made is not None and made <= narrowed.made:
            return
        narrowed.said.add(where)
        receiver = _text(call.func.value) if isinstance(call.func, ast.Attribute) else _text(call.func)
        came = f"; another object came in later on {receiver} at {frame.file}:{where[1]}"
        if made is None:
            if narrowed.left:
                narrowed.step.detail += (
                    f"{came}, whose construction is not found, so {len(narrowed.left)} more call(s) are not followed"
                )
            return
        narrowed.made |= made
        extra = [site for site in narrowed.left if narrowed.matches(site, made)]
        if not extra:
            return
        if len(extra) > narrowed.room:
            narrowed.step.detail += f"{came}, with {len(extra)} more call(s), too many to follow"
            return
        narrowed.room -= len(extra)
        narrowed.left = [site for site in narrowed.left if not any(site is one for one in extra)]
        narrowed.step.detail += f"{came}, and its {len(extra)} call(s) are followed too"
        self.beyond.append(lambda: narrowed.follow(extra))

    def _object_made(self, frame: _Frame) -> Optional[set[tuple[str, int, int]]]:
        """Where the objects were built that the frames which asked into *frame* came in on.

        A parameter of ``__init__`` is chosen where the object is built, and
        every class built in more than one place gave the walk every one of
        them: ``ListWithLock`` is built by twelve knights and message
        processors, and the ``dataList`` a ``TaskRefinerThread`` reads through
        ``self.taskList.get(nTasks)`` is the one ``TaskRefiner.start`` built at
        line 72 -- the other eleven are other workers' lists, and twelve was
        too many to follow at all.  A setter's calls are the same question:
        the ones that matter are made on that object.  The walk came into
        ``get`` on ``self.taskList``, so that object's construction is found
        the way the walk finds anything: backwards, through what was assigned
        to it (:meth:`_constructions`).

        ``None`` -- every call, as before -- unless each frame that asked was
        entered by calls on objects whose constructions are all found.  A field
        assigned from outside its class (``gen_thr.inputList = None``) is not
        seen here, as it is not by :meth:`_assigned_field`.
        """
        askers = self.asked_of.get((frame.file, getattr(frame.func, "lineno", 0)))
        if not askers:
            return None
        found: set[tuple[str, int, int]] = set()
        for asker in askers:
            objects = self._entered_on(asker, 0)
            if objects is None:
                return None
            for file, func, expression in objects:
                made = self._made_at(file, func, expression)
                if made is None:
                    return None
                found |= made
        return found or None

    def _on_the_object(
        self, frame: _Frame, sites: list
    ) -> tuple[list, list, Optional[set[tuple[str, int, int]]]]:
        """``(kept, left out, where the object was built)`` of the calls *sites* placed at a setter.

        Left out: a call on an object whose construction is found and is none
        of the walk's object's.  Kept: the rest -- a call on ``self``, where the
        object is the caller's own and the walk may have come in through it,
        and a call on an object whose construction is not found.
        """
        made = self._object_made(frame)
        if made is None:
            return sites, [], None
        kept, left = [], []
        for site in sites:
            call = site.call
            receiver = call.func.value if isinstance(call.func, ast.Attribute) else None
            if receiver is None or (isinstance(receiver, ast.Name) and receiver.id == "self"):
                kept.append(site)
                continue
            other = self._made_at(site.file, site.func, receiver)
            (kept if other is None or other & made else left).append(site)
        return kept, left, made

    def _made_at(self, file: str, func: ast.AST, expression: ast.expr) -> Optional[set[tuple[str, int, int]]]:
        """:meth:`_constructions` of *expression* in *func*, once per walk; ``None`` where it builds nothing."""
        key = (file, getattr(func, "lineno", 0), ast.dump(expression))
        if key not in self._made:
            self._made[key] = self._constructions(file, func, expression, _BUILT_HOPS, set()) or None
        return self._made[key]

    def _entered_on(self, frame: _Frame, depth: int) -> Optional[list[tuple[str, ast.AST, ast.expr]]]:
        """``(file, function, object)`` for each call that brought the walk into *frame*.

        A call on ``self`` is the caller's own object, so the calls that brought
        the walk into the caller answer it, up to four deep.  ``None`` where a
        call is not made on an object, or *frame* was not entered by a call.
        """
        held = self.passed.get((frame.file, frame.owner))
        if not held:
            return None
        objects: list[tuple[str, ast.AST, ast.expr]] = []
        for caller, call, _bound in held:
            more = self._call_objects(caller, call, depth)
            if more is None:
                return None
            objects.extend(more)
        return objects

    def _call_objects(
        self, caller: _Frame, call: ast.Call, depth: int
    ) -> Optional[list[tuple[str, ast.AST, ast.expr]]]:
        """``(file, function, object)`` *call*, made in *caller*, is made on; see :meth:`_entered_on`."""
        if not isinstance(call.func, ast.Attribute):
            return None
        receiver = call.func.value
        if isinstance(receiver, ast.Name) and receiver.id == "self":
            if depth >= 4:
                return None
            return self._entered_on(caller, depth + 1)
        return [(caller.file, caller.func, receiver)]

    def _constructions(
        self,
        file: str,
        func: ast.AST,
        expression: ast.expr,
        hops: int,
        seen: set,
    ) -> Optional[set[tuple[str, int, int]]]:
        """The constructor calls that built what *expression* holds in *func*.

        ``self.<field>`` is what the class assigns to it, in any of its methods
        -- an ``append`` into it changes what the object holds, not which
        object it is; a local is what ``=`` gives it, and a name the function
        neither binds nor takes is the module's; a parameter is what each
        caller hands it, *hops* callers out.  A constant builds nothing --
        ``JobGeneratorThread(None, ...)`` in a test script.  ``None`` as soon as
        one way is not known: a field a base class elsewhere assigns, a name
        bound by a loop or an import, a value that is any other call --
        ``self.get_datasets_list()``.

        The files are read outside the walk's budget: which object a call is
        made on does not depend on how much the walk has read, and a walk out
        of files had made ``JobGeneratorThread``'s ``brokerageLockIDs`` an
        object of unknown construction.
        """
        if isinstance(expression, ast.Constant):
            return set()
        key = (file, getattr(func, "lineno", 0), ast.dump(expression), hops)
        if key in seen:
            return set()
        seen.add(key)
        values: list[tuple[ast.AST, Optional[ast.expr]]] = []
        if (
            isinstance(expression, ast.Attribute)
            and isinstance(expression.value, ast.Name)
            and expression.value.id == "self"
        ):
            tree = self.types.tree(file)
            if tree is None:
                return None
            cls = pathcond.class_of(tree, func)
            if cls is None:
                return None
            for where, holder in self.types.lineage(file, cls)[1:]:
                if where != file and any(
                    pathcond.attribute_expressions(method, expression.attr)
                    for method in pathcond.methods_of(holder)
                ):
                    return None
            for holder in [cls, *pathcond.bases_in(tree, cls)]:
                for method in pathcond.methods_of(holder):
                    values.extend((method, value) for value in pathcond.attribute_expressions(method, expression.attr))
        elif isinstance(expression, ast.Name):
            bound = _plain_bindings(func, expression.id)
            if bound is None:
                return None
            values = [(func, value) for value in bound]
            if expression.id in _parameters(func):
                values.append((func, None))
            elif not bound and not isinstance(func, ast.Module):
                tree = self.types.tree(file)
                bound = _plain_bindings(tree, expression.id) if tree is not None else None
                if not bound:
                    return None
                values = [(tree, value) for value in bound]
        else:
            return None
        if not values:
            return None
        found: set[tuple[str, int, int]] = set()
        for holder, value in values:
            if value is None:
                # The parameter itself, as each caller of *holder* hands it.
                if hops <= 0:
                    return None
                owned = self.types.method_class(file, holder)
                called = calls_mod.Target(file, holder, owned[1] if owned else None)
                sites = self.calls.callers(called)
                if not sites or len(sites) > _BUILT_CALLERS:
                    return None
                for site in sites:
                    argument = self.calls.handed(site, called).get(expression.id)
                    if argument is None:
                        return None
                    more = self._constructions(site.file, site.func, argument, hops - 1, seen)
                    if more is None:
                        return None
                    found |= more
            elif isinstance(value, ast.Call):
                if self.types.class_named(file, value.func) is None and (
                    isinstance(holder, ast.Module)
                    or not any(
                        getattr(one.func, "name", "") == "__init__"
                        for one in self.calls.callees(file, holder, value).targets
                    )
                ):
                    return None
                found.add((file, value.lineno, value.col_offset))
            elif isinstance(value, ast.Name):
                more = self._constructions(file, holder, value, hops, seen)
                if more is None:
                    return None
                found |= more
            elif not isinstance(value, ast.Constant):
                return None
        return found

    def _module_binding(
        self, frame: _Frame, name: str, expression: ast.expr, depth: int
    ) -> None:
        """A value the file binds once, when it is imported.

        Recorded as a binding because that is what it is, and marked because
        the reader must not read it as a step on the path: it ran at import,
        not on the way to this arm, so no guard of this function dominated it.

        The walk stops here rather than following what the expression reads.
        Those names live in the module's scope, and resolving them inside this
        function's frame would answer with whatever local happened to share the
        name -- the mistake the scope note below exists to prevent.
        """
        terminal, detail = self.terminal_of(frame, expression)
        line = getattr(expression, "lineno", 0)
        self.record(
            TraceStep(
                kind=TRACE_BINDING,
                name=name,
                owner=f"{frame.file}::{MODULE_SCOPE}",
                file=frame.file,
                line=line,
                value=_text(expression),
                operator=_operator(pathcond.site_of(expression)),
                unseen=[
                    f"{MODULE_SCOPE} at {line}: bound when the file is imported, "
                    f"not on the way here"
                ],
                terminal=terminal,
                detail=detail,
                depth=depth,
            )
        )

    def _loop(self, frame: _Frame, name: str, loop: ast.AST, depth: int) -> None:
        over = _text(loop.iter)
        reads = _reads(loop.iter)
        step = TraceStep(
            kind=TRACE_LOOP,
            name=name,
            owner=frame.owner,
            file=frame.file,
            line=loop.lineno,
            value=f"an element of {over}",
            guards=pathcond.path_condition(loop),
            unseen=_unseen(loop),
            reads=reads,
            depth=depth,
        )
        # What the loop was on is a fact about which row this pass was, not
        # about how the value was worked out: told apart, so two guards'
        # derivations can be compared without the rows each loop visited.
        held = self._role
        self._role = ROLE_ITERATION
        try:
            if self.record(step):
                self.want(frame, reads, depth + 1, loop)
        finally:
            self._role = held

    def _caught(
        self, frame: _Frame, name: str, handler: ast.ExceptHandler, depth: int
    ) -> None:
        """An ``except ... as e``: the value is whatever the body raised."""
        resolve = self.resolver(frame)
        terminal, detail = _raised_in(
            handler.body[0] if handler.body else handler,
            self.classify,
            resolve,
            self.typer(frame),
        )
        self.record(
            TraceStep(
                kind=TRACE_BINDING,
                name=name,
                owner=frame.owner,
                file=frame.file,
                line=handler.lineno,
                value=f"raised under {_text(handler.type)}",
                unseen=_unseen(handler),
                terminal=terminal,
                detail=detail,
                depth=depth,
            )
        )

    def _sibling(self, frame: _Frame, method: ast.AST) -> Optional[_Frame]:
        """A frame for another method of the same class.

        Its own frame, not the caller's, because the guards the walk reports
        come from the statement's ancestors: recording ``__init__``'s
        assignment inside the frame of the method that read it would print a
        condition that never dominated the write.  Handovers travel with it --
        they are a fact about how the object was constructed, which is as true
        in one of its methods as in another.
        """
        owner = f"{frame.owner.rsplit('::', 1)[0]}::{method.name}"
        key = (frame.file, owner)
        held = self.frames.get(key)
        if held is not None and held.func is not method and frame.file in self.parsed:
            # Another class's method of that name holds the key --
            # ``ThreadUtils.py`` has an ``__init__`` in each class -- and its
            # assignments are not this object's.
            owner = _with_class(frame.file, self.parsed[frame.file][1], method)
            key = (frame.file, owner)
            held = self.frames.get(key)
        if held is not None:
            return held
        if key in self.frames:
            return None
        if len(self.frames) >= self.budget.functions:
            self._exhausted("functions")
            self.frames[key] = None
            return None
        self.frames[key] = _Frame(
            frame.file, owner, method, frame.handovers, frame.imports
        )
        return self.frames[key]

    def _assigned_field(self, frame: _Frame, name: str, field: str, depth: int) -> bool:
        """Stages two and three: ``self.<field> = ...`` in scope.

        Ordered, and the order is the design.  ``classify()`` has already run
        above this -- it has to, because ``self.cur`` is assigned in an
        ``__init__`` like any other field and is still the database, and this
        reading would otherwise answer a question about a row with a line
        about a connection.

        Two before three because the nearer assignment is the one whose guards
        reached the arm, and both rather than either because the answer is a
        set: ``self.jobs`` is written in three methods of
        ``setupper_atlas_plugin`` and which one ran last is a fact about the
        run, not about the text.  Nothing is folded.
        """
        answered = False
        for expression in pathcond.attribute_expressions(frame.func, field):
            self._binding(frame, name, expression, depth)
            answered = True
        parsed = self.module(frame.file)
        if parsed is None:
            return answered
        _source, tree = parsed
        cls = pathcond.class_of(tree, frame.func)
        if cls is None:
            return answered
        for holder in [cls, *pathcond.bases_in(tree, cls)]:
            for method in pathcond.methods_of(holder):
                if method is frame.func:
                    continue
                other = None
                said: Optional[TraceStep] = None
                # What a sibling assigns, and what it puts into the field --
                # ``self.allPandaSites.append(siteName)`` in ``add`` is how a
                # nucleus learns its sites.
                for expression in [*pathcond.attribute_expressions(method, field), *_self_fills(method, field)]:
                    other = other or self._sibling(frame, method)
                    if other is None:
                        break
                    self._asked_into(other, frame)
                    before = len(self.steps)
                    self._binding(other, name, expression, depth)
                    said = said or next((step for step in self.steps[before:] if step.owner == other.owner), None)
                    answered = True
                if other is not None and method.name != "__init__":
                    self.beyond.append(
                        lambda held=other, step=said: self._mutated_at(held, field, depth + 1, step)
                    )
        # Bases declared in other files are read beyond the function's own
        # answer: ``AdderAtlasPlugin``'s ``self.result`` is assigned in
        # ``AdderPluginBase.__init__``, one file over.
        for where, holder in self.types.lineage(frame.file, cls)[1:]:
            if where == frame.file:
                continue
            for method in pathcond.methods_of(holder):
                if not pathcond.attribute_expressions(method, field):
                    continue
                self.beyond.append(
                    lambda place=where, one=method: self._inherited_field(
                        frame, place, one, name, field, depth
                    )
                )
                answered = True
        if not answered:
            # ``rc_fatal = 2`` in the class body: a constant every instance
            # reads until one assigns its own, and here none does.
            for holder in [cls, *pathcond.bases_in(tree, cls)]:
                for node in holder.body:
                    if (
                        isinstance(node, ast.Assign)
                        and any(isinstance(target, ast.Name) and target.id == field for target in node.targets)
                    ):
                        self.record(
                            TraceStep(
                                kind=TRACE_BINDING,
                                name=name,
                                owner=f"{frame.file}::{holder.name}",
                                file=frame.file,
                                line=node.lineno,
                                value=_text(node.value),
                                reads=_reads(node.value),
                                unseen=["class body: the value every instance reads until it assigns its own"],
                                depth=depth,
                            )
                        )
                        answered = True
        return answered

    def _asked_into(self, other: _Frame, frame: _Frame) -> None:
        """Record that *frame* asked into *other*, a method assigning a field of the object *frame* came in on.

        The calls placed at *other*, constructor or setter, are a question
        about that object (:meth:`_object_made`), and so is a narrowing
        already made there that a later object may widen (:meth:`_watch`).
        """
        key = (other.file, getattr(other.func, "lineno", 0))
        held = self.asked_of.setdefault(key, [])
        if not any(one is frame for one in held):
            held.append(frame)
            for narrowed in self.narrowings.get(key, ()):
                self._watch(narrowed, frame)
            for left in self.by_class.get(key, ()):
                self._asked_later(left, frame)

    def _asked_later(self, left: _ByClass, frame: _Frame) -> None:
        """A method of another class asks into an ``__init__`` already answered: follow the left calls that build its objects."""
        held = self.types.method_class(frame.file, frame.func)
        if held is not None and any(one[1] is held[1] for one in left.classes):
            return
        extra = [site for site in left.left if held is None or self._may_build(site, [held])]
        if held is not None:
            left.classes.append(held)
        if not extra:
            return
        asker = held[1].name if held is not None else frame.owner
        if len(extra) > left.room:
            left.step.detail += f"; asked again from {asker}, with {len(extra)} more call(s), too many to follow"
            return
        left.room -= len(extra)
        left.left = [site for site in left.left if site not in extra]
        left.step.detail += f"; asked again from {asker}, and its {len(extra)} call(s) are followed too"
        self.beyond.append(lambda: left.follow(extra))

    def _inherited_field(
        self, frame: _Frame, file: str, method: ast.AST, name: str, field: str, depth: int
    ) -> None:
        """``self.<field>`` as a base class in another file assigns it, asked by *frame* as a sibling in the file is."""
        other = self.frame(file, f"{file}::{getattr(method, 'name', '')}", line=method.lineno)
        if other is None:
            return
        self._asked_into(other, frame)
        self.entered.add((other.file, other.owner))
        for expression in pathcond.attribute_expressions(other.func, field):
            self._binding(other, name, expression, depth)
        if getattr(method, "name", "") != "__init__":
            self._mutated_at(other, field, depth + 1)

    def _field(self, frame: _Frame, name: str, depth: int) -> None:
        field = name.split(".", 1)[1]
        try:
            expression = ast.parse(name, mode="eval").body
        except SyntaxError:
            return
        terminal, detail = self.terminal_of(frame, expression)
        if terminal:
            self.record(
                TraceStep(
                    kind=TRACE_UNBOUND,
                    name=name,
                    owner=frame.owner,
                    file=frame.file,
                    value=name,
                    terminal=terminal,
                    detail=detail,
                    depth=depth,
                )
            )
            return
        answered = self._assigned_field(frame, name, field, depth)
        # Only a dispatch: its keys are the worker's own attributes, which is
        # what ``self.<field>`` is.  A call's keys are parameter names, and
        # letting those answer here would hand ``self.x`` whatever a caller
        # passed for a parameter that happens to be spelled ``x``.
        #
        # Every one of them, not the first.  ``AdderGen`` is built in three
        # places and the pilot's and the daemon's differ in what they pass;
        # stopping at whichever the map listed first would answer "who chose
        # this" with one of the callers and no sign that there were others.
        for handover in frame.handovers:
            if handover.reached_by == ARRIVES_BY_DISPATCH and field in handover.fields:
                self._handover(frame, name, handover, depth)
                answered = True
        if answered:
            # The assignments are the answer.  Saying "no handover names it"
            # underneath them would report an absence as though it were the
            # reason, which is the shape this whole change is about.
            return
        self.record(
            TraceStep(
                kind=TRACE_UNBOUND,
                name=name,
                owner=frame.owner,
                file=frame.file,
                value=name,
                terminal=STOP_PARAMETER,
                detail="an attribute of the worker, and the map names no handover for it",
                depth=depth,
            )
        )

    def _handover(
        self, frame: _Frame, name: str, handover: Handover, depth: int
    ) -> None:
        key = name.split(".", 1)[1] if name.startswith("self.") else name
        supplied = handover.fields[key]
        owner = f"{handover.entry}::{handover.via}"
        other = self.frame(handover.entry, owner)
        try:
            expression = ast.parse(supplied, mode="eval").body
        except SyntaxError:
            expression = None
        # ``reads`` is a promise that each name will get a step of its own, so
        # it is only made when there is a frame to make it in.  Six names in
        # the corpus were promised by a crossing whose far side would not
        # parse, and a promise nothing keeps reads as a hole in the walk.
        reads = _reads(expression) if expression is not None and other else []
        step = TraceStep(
            kind=TRACE_HANDOVER,
            name=name,
            owner=owner,
            file=handover.entry,
            line=getattr(other.func, "lineno", 0) if other else 0,
            value=supplied,
            reads=reads,
            detail=(
                f"handed over by {handover.via}, reached by {handover.reached_by}"
                if other
                else f"handed over by {handover.via}, which could not be read here"
            ),
            depth=depth,
        )
        if not self.record(step):
            return
        if other is not None:
            self.want(other, reads, depth + 1)

    def _unbound(
        self, frame: _Frame, name: str, depth: int, only_below: bool = False
    ) -> None:
        # A parameter is settled at the call site, and for the entries the map
        # reached by a plain call it recorded what was passed -- so the same
        # crossing that answers a worker's attribute answers this too.  Kept
        # to parameters of *this* function, since a dispatch's keys name the
        # worker's attributes and could collide with an unrelated local.
        #
        # The map's handover runs from an entry straight to here, across the
        # facade it was called through, so it is not the only call: ``killJobs``
        # retries an ES merge job through ``proxy.retryJob`` and the handover
        # from ``copyArchive`` hid it.  The calls ``calls`` places at this
        # function are asked as well, less the facade method the handover
        # already crosses (its ``via``); where that leaves none, the handover
        # is the whole answer, as it was.
        parameter = _is_parameter(frame.func, name)
        spanned: frozenset[str] = frozenset()
        if parameter:
            for handover in frame.handovers:
                if handover.reached_by == ARRIVES_BY_CALL and name in handover.fields:
                    self._handover(frame, name, handover, depth)
                    spanned = frozenset({handover.via})
                    break
            if spanned:
                if not self._callers_beside(frame, spanned):
                    return
            elif self._passed_in(frame, name, depth):
                return
        stopped = TraceStep(
                kind=TRACE_UNBOUND,
                name=name,
                owner=frame.owner,
                file=frame.file,
                value=name,
                terminal=STOP_PARAMETER if parameter else "",
                detail=(
                    "a parameter, so its value is chosen at the call site"
                    if parameter
                    else "bound in this function only below the arms, and in no loop that "
                    "holds both, so none of those can have set it"
                    if only_below
                    else "not bound in this function -- module scope, an import, or a form the walk does not read"
                ),
                depth=depth,
            )
        if self.record(stopped) and parameter:
            # The parameter's own answer is the stop, as it always was; where
            # it was chosen is looked for beyond this function, afterwards.
            self.beyond.append(lambda: self._called_from(frame, name, depth, stopped, spanned))


class Walked(NamedTuple):
    """What one walk found.

    A named tuple rather than a pair because the walk answers two questions
    about the same function -- why the arms ran, and what the function prints
    when they do -- and the second is empty whenever the first is refused.

    The two are siblings rather than one inside the other because they are
    ordered on different axes.  ``steps`` runs along data dependence and
    crosses into other functions; the skeleton runs in source order and stays
    in this one, because the thing it is laid against is a log region and a
    log region is in time order.  Folding the skeleton into ``steps`` would
    put rows that say nothing about the value into a list whose every row is
    about the value.
    """

    steps: list[TraceStep]
    note: str
    skeleton: list[SkeletonLine] = []


def exit_guard(
    roots: dict[str, Path],
    *,
    file: str,
    owner: str,
    line: int,
    expected_sha: str = "",
) -> Optional[tuple[int, int, int]]:
    """``(decided at, first, last)`` for the block an exit at *line* sits in.

    What the read side stores is where a row stops being followed; what a
    reader asks next is *what the code says when it stops there*, and the
    answer is the block the exit is in, not the lines before it.  Cutting the
    statement's skeleton at the exit line instead reads the other branch back
    to them: ``copyArchive.py``'s ``holding`` query logs one line if nothing
    came back and another if something did, and everything in between belongs
    to the branch the row did not take.

    The block, and not the guarding ``if`` alone, because the exit is only
    half of what happened -- ``if int(id) in xmlIDs:`` is followed by
    ``_logger.debug("   found XML -> skip")`` and *then* the ``continue``, and
    that line is the whole of what a reader can grep for.

    ``None`` when the tree is not the one the map was built from, on the same
    terms as :func:`walk`: a block read out of the wrong release is plausible
    code that is not the code.
    """
    state = _Walk(roots, _nothing, DEFAULT_BUDGET, "")
    parsed = state.module(file)
    if parsed is None:
        return None
    source, _tree = parsed
    if expected_sha and git_blob_sha(source) != expected_sha:
        return None
    frame = state.frame(file, owner, line=line)
    if frame is None or not isinstance(
        frame.func, (ast.FunctionDef, ast.AsyncFunctionDef)
    ):
        return None
    # An ``except`` is addressed by its own line and is not an ``ast.stmt``,
    # so it is looked for by hand before the statements are.
    for node in ast.walk(frame.func):
        if isinstance(node, ast.ExceptHandler) and node.lineno == line and node.body:
            return (line, node.body[0].lineno, node.body[-1].end_lineno or node.lineno)
    statement = _statement_at(frame.func, line)
    if statement is None:
        return None
    for ancestor in pathcond.ancestors(statement):
        if not isinstance(ancestor, (ast.stmt, ast.ExceptHandler)):
            continue
        for _header, body in (
            [("except", ancestor.body)]
            if isinstance(ancestor, ast.ExceptHandler)
            else _blocks_of(ancestor)
        ):
            if statement not in body:
                continue
            return (
                ancestor.lineno,
                body[0].lineno,
                max(item.end_lineno or item.lineno for item in body),
            )
    return None


def walk(
    roots: dict[str, Path],
    *,
    file: str,
    owner: str,
    lines: Sequence[int],
    observed: str = "",
    expected_sha: str = "",
    handovers: Sequence[Handover] = (),
    classify: Optional[Classifier] = None,
    budget: Budget = DEFAULT_BUDGET,
    region: Optional[tuple[int, int]] = None,
    bindings_within: Optional[tuple[int, int]] = None,
    reading_at: Optional[int] = None,
    reaching: bool = False,
    guards_above: bool = True,
) -> Walked:
    """Explain the arms at *lines*, and say why the answer is short when it is.

    *reaching* keeps its default for callers outside this package -- the
    querier's driver calls ``walk`` and leaves it out -- but every call in
    bamboo states it (``tests/test_codemap_trace.py`` checks): whether a
    binding must be able to reach the arm is a choice each caller makes for a
    reason of its own, and the default made it for two answer paths that never
    said.

    *lines* are the arms of one function, walked together so that a flag they
    share is explained once.  Returns the steps and a note -- empty when the
    walk finished on its own, which the measurement says is the usual case.

    Also returns the skeleton: what this function prints, in source order,
    with the arms in place -- see
    :class:`~bamboo.codemap.models.SkeletonLine`.  *observed* is the value the
    record actually holds, and given one the rows whose hole it lands in carry
    the narrower pattern with it filled in as well.  Without one the skeleton
    still comes back, because a line that reports nothing about the value
    still says the code around it ran.

    *region* narrows that skeleton to one part of the function, and is the
    read side's to give: a junction owns its whole function and passes none,
    a statement owns the stretch between its own binds and the next
    statement's.  The default is the whole function, so nothing that does not
    ask for a region sees a different answer.

    *bindings_within* says the same thing about the walk rather than the
    skeleton: a name assigned inside that stretch is explained from there
    only.  See :meth:`_Walk._within` -- and the same default, whole-function,
    for everything that does not ask.

    *reading_at* is the other way of narrowing the same thing, by execution
    order rather than by position: the line whose reading is being explained,
    against which a write that a later one overwrote is dropped.  See
    :meth:`_Walk._reaching`.  The two are given separately because neither
    contains the other.

    *reaching* asks for the same narrowing with no line to read at: each arm
    is its own reading point, and each name deeper in is read where the
    statement that asked for it read it.  What a junction's arms want, since a
    junction owns its whole function and has no one statement to answer for.
    Implied by *reading_at*.

    *guards_above* false explains an arm by its own test alone, and not by the
    tests that reach it.  What a question about one ``if`` wants: the silent
    decisions are listed by what their test reads, and ``AtlasAnalJobBroker``'s
    ``if minRamCount > site_maxmemory`` was listed as reading
    ``IO_INTENSITY_CUTOFF_USER`` because a guard at the head of
    ``doBrokerage`` does.  Deeper steps keep theirs: which binding a name the
    test reads got is decided by the tests above that binding.

    **A tree that is not the map's is refused rather than warned about.**  A
    trace is shaped like an answer, and reading line 4971 of the wrong release
    produces plausible code that is not the code -- which already happened
    once, to a report that only printed coordinates.  Computing from the wrong
    tree is worse than printing from it.
    """
    state = _Walk(
        roots,
        classify or _nothing,
        budget,
        observed,
        within=(file, owner, *bindings_within) if bindings_within else None,
        reading_at=(file, owner, reading_at) if reading_at else None,
        narrowed=(file, owner) if reaching or reading_at else None,
        guards_above=guards_above,
    )
    parsed = state.module(file)
    if parsed is None:
        return Walked([], "the map's file could not be read from this tree")
    source, _tree = parsed
    if expected_sha and git_blob_sha(source) != expected_sha:
        return Walked([], "the walk was not run: this tree is not the one that was mapped")
    if not lines:
        return Walked([], "the map records no line for these arms")
    frame = state.frame(file, owner, line=lines[0], handovers=handovers)
    if frame is None:
        return Walked([], "the map's function could not be located in this tree")
    for line in lines:
        state.arm(frame, line)
    state.drain()
    return Walked(state.steps, state.notes(), state.skeleton(frame, region))


def walk_from_print(
    roots: dict[str, Path],
    *,
    file: str,
    line: int,
    expected_sha: str = "",
    classify: Optional[Classifier] = None,
    budget: Budget = DEFAULT_BUDGET,
) -> Walked:
    """Explain what the logging call on *line* prints, from the names its message reads.

    The entry a reader with a log line in hand needs: ``waiting 26 jobs for
    soft finish`` is the line, and the question is where its ``26`` came from.
    Every other entry starts from a write or a guard the map recorded; this
    one starts from what production printed, which is the one thing the
    reader holds.

    The call is the first step and its message's names are what is walked --
    not the logger it is called on, which says nothing about the value.  A
    message built into a local first is walked through the local, the way any
    name is.  Narrowed as a statement's walk is: each name is answered with
    the writes that reach the statement that read it.

    The walk does not yet enter a call, so a value a call returns ends the
    step that binds it, as it does for every other entry.  Refused, like
    :func:`walk`, when the tree is not the one the map was built from.
    """
    state = _Walk(roots, classify or _nothing, budget)
    parsed = state.module(file)
    if parsed is None:
        return Walked([], "the file could not be read from this tree")
    source, tree = parsed
    if expected_sha and git_blob_sha(source) != expected_sha:
        return Walked([], "the walk was not run: this tree is not the one that was mapped")
    func = containing_function(tree, line, "")
    if func is None:
        return Walked([], "no function holds this line")
    owner = f"{file}::{func.name}"
    state.narrowed = (file, owner)
    frame = state.frame(file, owner, line=line)
    if frame is None:
        return Walked([], "the function could not be located in this tree")
    statement = _statement_at(frame.func, line)
    calls = logging_arguments(frame.func)
    arguments = [
        argument
        for argument in calls
        if statement is not None and _enclosing_statement(argument) is statement
    ]
    # The line a reader recognises is often where the message is written,
    # not where it is logged: ``tmpMsg = f"waiting {len(pandaIDs)} ..."`` on
    # one line and ``tmpLog.info(tmpMsg)`` on the next.  A binding a logging
    # call prints is the message as much as the call's own argument is.
    if not arguments and isinstance(statement, (ast.Assign, ast.AnnAssign)) and statement.value:
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        named = {_text(target) for target in targets}
        if any(
            _text(argument) in named
            and pathcond.can_reach(statement.value, argument, current=True)
            for argument in calls
        ):
            arguments = [statement.value]
    if not arguments:
        return Walked([], "no logging call prints what is on this line")
    for argument in arguments:
        reads = _reads(argument)
        state.record(
            TraceStep(
                kind=TRACE_PRINT,
                owner=owner,
                file=file,
                line=statement.lineno,
                value=_text(argument),
                guards=pathcond.path_condition(statement),
                unseen=_unseen(statement),
                reads=reads,
            )
        )
        state.want(frame, reads, 1, statement)
    state.drain()
    return Walked(state.steps, state.notes(), state.skeleton(frame))


def walk_from_line(
    roots: dict[str, Path],
    *,
    file: str,
    line: int,
    expected_sha: str = "",
    classify: Optional[Classifier] = None,
    budget: Budget = DEFAULT_BUDGET,
) -> Walked:
    """Explain the statement on *line*: what it prints if it prints, else what it computes.

    A printed line is walked from the names its message reads -- see
    :func:`walk_from_print`.  Any other statement is walked as an arm is,
    narrowed at itself: ``AtlasProdJobBroker.py:903``,
    ``origMinRamCount = inputChunk.getMaxRamCount()``, is the sieve's input,
    and what decides it is inside ``getMaxRamCount``.
    """
    walked = walk_from_print(
        roots, file=file, line=line, expected_sha=expected_sha, classify=classify, budget=budget
    )
    if walked.note != "no logging call prints what is on this line":
        return walked
    state = _Walk(roots, _nothing, budget)
    parsed = state.module(file)
    if parsed is None:
        return walked
    func = containing_function(parsed[1], line, "")
    if func is None:
        return Walked([], "no function holds this line")
    return walk(
        roots,
        file=file,
        owner=f"{file}::{func.name}",
        lines=[line],
        expected_sha=expected_sha,
        classify=classify,
        budget=budget,
        reaching=True,
    )


class FunctionPrints(NamedTuple):
    """What :func:`function_prints` hands back for one function."""

    file: str
    #: ``file::qualified.name`` -- see :func:`pathcond.qualified_name`.
    owner: str
    #: ``file::Class`` for a method, as ``TraceStep.cls`` spells it, else ``""``.
    cls: str
    func: ast.FunctionDef | ast.AsyncFunctionDef
    #: The hash of the text the rows were read from.
    blob_sha: str
    statements: list[PrintStatement]


def function_prints(roots: dict[str, Path], files: Sequence[str]) -> Iterator[FunctionPrints]:
    """Every function in *files* that makes a logging call, with the rows each call prints.

    The skeleton's print rows with nothing asked: no arm, so no row is placed
    beside one, and no observed value, so none is filled in.  What is left
    depends on the function alone, which is why the build can store it -- see
    :class:`~bamboo.codemap.models.FunctionPrintsNode`.

    One walk for every file, with the budget lifted: a budget bounds what one
    question may cost, and here nothing is asked.  The logger's prefix is the
    one reading that leaves the function, and a file budget spent half way
    through would empty the later prefixes without saying so.

    A function's own calls only.  ``logging_arguments`` walks into a nested
    ``def``, whose lines are that function's: the skeleton already leaves them
    out, and they come back under the nested function's own owner.  A call
    the skeleton gives no row has no statement here, so the count falls short
    of the calls and both the build's check and a reader's own count say so.
    """
    state = _Walk(
        roots, _nothing, DEFAULT_BUDGET._replace(files=sys.maxsize, functions=sys.maxsize)
    )
    for file in files:
        parsed = state.module(file)
        if parsed is None:
            continue
        source, tree = parsed
        imports = _imported_names(tree)
        for func, _cls in pathcond.functions_with_owner(tree):
            levels: dict[int, str] = {}
            for argument in own_logging_arguments(func):
                statement = _enclosing_statement(argument)
                if statement is not None:
                    levels.setdefault(statement.lineno, log_level(argument) or "")
            if not levels:
                continue
            owner = f"{file}::{pathcond.qualified_name(func)}"
            # The line keeps two functions of one name apart in the walk's
            # memo, which is keyed on the owner.
            frame = _Frame(file, f"{owner}@{func.lineno}", func, (), imports)
            rows: dict[int, list[PrintRow]] = {}
            for row in state.skeleton(frame):
                if row.kind != SKELETON_PRINT:
                    continue
                rows.setdefault(row.line, []).append(
                    PrintRow(
                        text=row.text,
                        pattern=row.pattern,
                        refused=row.refused,
                        keyed=row.keyed,
                        holes=row.holes,
                        prefix=row.prefix,
                    )
                )
            cls = pathcond.class_of(tree, func)
            yield FunctionPrints(
                file=file,
                owner=owner,
                cls=f"{file}::{cls.name}" if cls is not None else "",
                func=func,  # type: ignore[arg-type]
                blob_sha=git_blob_sha(source),
                statements=[
                    PrintStatement(line=line, level=levels.get(line, ""), rows=held)
                    for line, held in sorted(rows.items())
                ],
            )
