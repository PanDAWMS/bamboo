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

Nothing here is written into the map.
"""

from __future__ import annotations

import ast
import builtins
import itertools
import logging
import re
import string
from collections import deque
from pathlib import Path
from typing import Callable, NamedTuple, Optional, Sequence

from bamboo.codemap.gitsource import blob_sha as git_blob_sha
from bamboo.codemap.models import (
    ARRIVES_BY_CALL,
    ARRIVES_BY_DISPATCH,
    SKELETON_ARM,
    SKELETON_BRANCH,
    SKELETON_PRINT,
    STOP_EXTERNAL,
    STOP_PARAMETER,
    TRACE_BINDING,
    TRACE_HANDOVER,
    TRACE_LOOP,
    TRACE_UNBOUND,
    TRACE_WRITE,
    Handover,
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
from bamboo.codemap.panda.recognizers.emit import logging_arguments
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


DEFAULT_BUDGET = Budget()

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
    found = []
    for node in ast.walk(func):
        if not isinstance(node, (ast.For, ast.AsyncFor)):
            continue
        targets = [node.target]
        if isinstance(node.target, (ast.Tuple, ast.List)):
            targets = list(node.target.elts)
        if any(isinstance(t, ast.Name) and t.id == name for t in targets):
            found.append(node)
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
    than guessed.
    """
    found: list[tuple[ast.Assign, Optional[int]]] = []
    for node in ast.walk(func):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, (ast.Tuple, ast.List)):
                continue
            starred = any(isinstance(e, ast.Starred) for e in target.elts)
            for slot, element in enumerate(target.elts):
                if isinstance(element, ast.Name) and element.id == name:
                    found.append((node, None if starred else slot))
    return found


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
    it plainly binds.  Read here rather than in ``pathcond``: that module is
    shared with the build, and widening it would move the map.
    """
    found: list[ast.expr] = []
    for node in ast.walk(func):
        if not isinstance(node, (ast.With, ast.AsyncWith)):
            continue
        for item in node.items:
            held = item.optional_vars
            targets = [held]
            if isinstance(held, (ast.Tuple, ast.List)):
                targets = list(held.elts)
            if any(isinstance(t, ast.Name) and t.id == name for t in targets):
                found.append(item.context_expr)
    return found


def _walrus_bindings(func: ast.AST, name: str) -> list[ast.expr]:
    """The values an assignment expression ``(<name> := ...)`` binds."""
    return [
        node.value
        for node in ast.walk(func)
        if isinstance(node, ast.NamedExpr)
        and isinstance(node.target, ast.Name)
        and node.target.id == name
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


def _caught_as(func: ast.AST, name: str) -> list[ast.ExceptHandler]:
    """Handlers that bind *name* as the exception they caught."""
    return [
        node
        for node in ast.walk(func)
        if isinstance(node, ast.ExceptHandler) and node.name == name
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
        for statement in block.body:
            for sub in ast.walk(statement):
                if isinstance(sub, ast.Call):
                    terminal, detail = classify(sub, resolve, typed)
                    if terminal:
                        return terminal, detail
        break
    return "", ""


def _unquoted(annotation: ast.expr) -> ast.expr:
    """"SiteMapper" as the expression it quotes; anything else as it is."""
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        try:
            return ast.parse(annotation.value.strip(), mode="eval").body
        except SyntaxError:
            return annotation
    return annotation

class _Types:
    """What class the code *declares* a receiver to be.

    Read the way a type checker reads it -- an annotation, a constructor, the
    return annotation of the method called -- and nothing else.  The walk
    needs it for one question, which system is on the other side of a call:
    ``self.site_mapper.getSite(x)`` and ``self.siteMapper.getSite(x)`` are one
    interface spelled two ways, and matching the spelling classified one and
    walked straight through the other.  What the object *is* does not depend
    on what someone called it.

    Its own parse cache, apart from the walk's: reading a base class or the
    interface's own module to learn a type is not a step on the path, so it
    does not spend the walk's file budget.  The frame's own module is taken
    from the walk, because ``pathcond.class_of`` finds a method by identity.
    """

    #: How deep a declaration is followed through fields, locals and calls.
    #: JEDI's ``self.siteMapper`` is four links from what declares it -- the
    #: field's assignment, the call, ``self.taskBufferIF``'s assignment, the
    #: ``__init__`` parameter's annotation -- and each of those is two levels
    #: here, so four stopped one short of the answer.
    DEPTH = 8

    def __init__(self, roots: dict[str, Path], walked: dict[str, tuple[str, ast.Module]]):
        self.roots = roots
        self.walked = walked
        self.parsed: dict[str, Optional[ast.Module]] = {}
        self._imports: dict[str, dict[str, tuple[str, str]]] = {}
        self._assigned: dict[int, dict[str, list[ast.expr]]] = {}

    def assigned(self, func: ast.AST) -> dict[str, list[ast.expr]]:
        """``pathcond.assigned_expressions``, once per function."""
        if id(func) not in self._assigned:
            self._assigned[id(func)] = pathcond.assigned_expressions(func)
        return self._assigned[id(func)]

    # -- modules -----------------------------------------------------------

    def tree(self, file: str) -> Optional[ast.Module]:
        if file in self.walked:
            return self.walked[file][1]
        if file not in self.parsed:
            package, _, relative = file.partition("/")
            root = self.roots.get(package)
            tree = None
            if root is not None and relative:
                try:
                    tree = ast.parse((root / relative).read_text(errors="replace"))
                except (OSError, SyntaxError, ValueError):
                    tree = None
            self.parsed[file] = tree
        return self.parsed[file]

    def module_file(self, dotted: str) -> Optional[str]:
        """``pandaserver.brokerage.SiteMapper`` -> its file, when the tree holds it."""
        parts = dotted.split(".")
        if not parts or parts[0] not in self.roots:
            return None
        for candidate in ("/".join(parts) + ".py", "/".join(parts) + "/__init__.py"):
            if self.tree(candidate) is not None:
                return candidate
        return None

    def imports(self, file: str) -> dict[str, tuple[str, str]]:
        """``{local name: (module, attribute)}`` for every import in *file*.

        Every import, wherever it sits: ``if TYPE_CHECKING:`` is where PanDA
        puts the ones only an annotation needs, and those are the ones this
        reads.  A relative import is resolved against the file's own package.
        """
        if file in self._imports:
            return self._imports[file]
        found: dict[str, tuple[str, str]] = {}
        tree = self.tree(file)
        package = file[: -len(".py")].split("/")[:-1] if file.endswith(".py") else []
        for node in ast.walk(tree) if tree is not None else ():
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname:
                        found[alias.asname] = (alias.name, "")
                    else:
                        head = alias.name.split(".")[0]
                        found.setdefault(head, (head, ""))
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = package[: len(package) - (node.level - 1)] if node.level > 1 else package
                    module = ".".join([*base, *([node.module] if node.module else [])])
                else:
                    module = node.module or ""
                for alias in node.names:
                    found[alias.asname or alias.name] = (module, alias.name)
        self._imports[file] = found
        return found

    # -- classes -----------------------------------------------------------

    def class_named(self, file: str, expression: ast.expr, depth: int = 0) -> Optional[tuple[str, ast.ClassDef]]:
        """The class *expression* names in *file*: declared there, or imported."""
        if depth > self.DEPTH:
            return None
        tree = self.tree(file)
        if tree is None:
            return None
        if isinstance(expression, ast.Name):
            for node in tree.body:
                if isinstance(node, ast.ClassDef) and node.name == expression.id:
                    return file, node
            held = self.imports(file).get(expression.id)
            if held is None:
                return None
            module, attribute = held
            if not attribute:
                return None
            # ``from pkg.mod import Cls`` -- or ``from pkg import mod``, a module.
            target = self.module_file(module)
            if target is not None:
                return self.class_named(target, ast.Name(id=attribute, ctx=ast.Load()), depth + 1)
            return None
        if isinstance(expression, ast.Attribute) and isinstance(expression.value, ast.Name):
            # ``SiteMapper.SiteMapper`` -- a module imported, then its class.
            held = self.imports(file).get(expression.value.id)
            if held is None:
                return None
            module, attribute = held
            dotted = f"{module}.{attribute}" if attribute else module
            target = self.module_file(dotted)
            if target is None:
                return None
            return self.class_named(target, ast.Name(id=expression.attr, ctx=ast.Load()), depth + 1)
        return None

    def annotated(self, file: str, annotation: Optional[ast.expr]) -> Optional[tuple[str, ast.ClassDef]]:
        """The class an annotation states, through quotes and ``Optional``/``| None``."""
        if annotation is None:
            return None
        annotation = _unquoted(annotation)
        if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
            sides = [
                side for side in (annotation.left, annotation.right)
                if not (isinstance(side, ast.Constant) and side.value is None)
            ]
            found = {self.annotated(file, side) for side in sides} - {None}
            return found.pop() if len(found) == 1 else None
        if isinstance(annotation, ast.Subscript):
            wrapper = annotation.value
            name = wrapper.attr if isinstance(wrapper, ast.Attribute) else getattr(wrapper, "id", "")
            if name == "Optional":
                return self.annotated(file, annotation.slice)
            if name == "Union" and isinstance(annotation.slice, ast.Tuple):
                sides = [
                    side for side in annotation.slice.elts
                    if not (isinstance(side, ast.Constant) and side.value is None)
                ]
                found = {self.annotated(file, side) for side in sides} - {None}
                return found.pop() if len(found) == 1 else None
            return None
        return self.class_named(file, annotation)

    def lineage(self, file: str, cls: ast.ClassDef) -> list[tuple[str, ast.ClassDef]]:
        """*cls* and its bases, nearest first, followed across files through imports."""
        order: list[tuple[str, ast.ClassDef]] = [(file, cls)]
        queue = [(file, base) for base in cls.bases]
        seen = {(file, cls.name)}
        while queue and len(order) < 12:
            where, base = queue.pop(0)
            found = self.class_named(where, base)
            if found is None or (found[0], found[1].name) in seen:
                continue
            seen.add((found[0], found[1].name))
            order.append(found)
            queue.extend((found[0], one) for one in found[1].bases)
        return order

    def method(self, file: str, cls: ast.ClassDef, name: str):
        for where, holder in self.lineage(file, cls):
            for node in holder.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                    return where, node
        return None

    # -- declarations ------------------------------------------------------

    def of_field(self, file: str, cls: ast.ClassDef, field: str, depth: int) -> Optional[tuple[str, ast.ClassDef]]:
        """``self.<field>``'s class: its annotation in the class or a base, else what it is assigned."""
        lineage = self.lineage(file, cls)
        for where, holder in lineage:
            for node in holder.body:
                if (
                    isinstance(node, ast.AnnAssign)
                    and isinstance(node.target, ast.Name)
                    and node.target.id == field
                ):
                    return self.annotated(where, node.annotation)
            for method in pathcond.methods_of(holder):
                for node in ast.walk(method):
                    if (
                        isinstance(node, ast.AnnAssign)
                        and isinstance(node.target, ast.Attribute)
                        and isinstance(node.target.value, ast.Name)
                        and node.target.value.id == "self"
                        and node.target.attr == field
                    ):
                        return self.annotated(where, node.annotation)
        if depth >= self.DEPTH:
            return None
        found: set[tuple[str, str]] = set()
        held: dict[tuple[str, str], tuple[str, ast.ClassDef]] = {}
        for where, holder in lineage:
            for method in pathcond.methods_of(holder):
                for expression in pathcond.attribute_expressions(method, field):
                    if isinstance(expression, ast.Constant) and expression.value is None:
                        continue
                    one = self.of_expression(where, method, expression, depth + 1)
                    if one is None:
                        return None
                    found.add((one[0], one[1].name))
                    held[(one[0], one[1].name)] = one
        return held[found.pop()] if len(found) == 1 else None

    def of_expression(self, file: str, func: ast.AST, expression: ast.expr, depth: int = 0) -> Optional[tuple[str, ast.ClassDef]]:
        """The class *expression*, read in *func* of *file*, is declared to be."""
        if depth > self.DEPTH:
            return None
        tree = self.tree(file)
        if tree is None:
            return None
        if isinstance(expression, ast.Name):
            args = getattr(func, "args", None)
            for argument in (*args.posonlyargs, *args.args, *args.kwonlyargs) if args else ():
                if argument.arg == expression.id:
                    return self.annotated(file, argument.annotation)
            for node in ast.walk(func):
                if (
                    isinstance(node, ast.AnnAssign)
                    and isinstance(node.target, ast.Name)
                    and node.target.id == expression.id
                ):
                    return self.annotated(file, node.annotation)
            # Every assignment, less the ``None`` a name is cleared with:
            # ``siteMapper = SiteMapper(taskBuffer)`` and, at the end of the
            # daemon, ``siteMapper = None``.  The evidence for a local is all in
            # this function, so it is read rather than asked for; the answer
            # is a class only where every assignment says the same one.
            bound = [
                value
                for value in self.assigned(func).get(expression.id, [])
                if not (isinstance(value, ast.Constant) and value.value is None)
            ]
            found = {self.of_expression(file, func, value, depth + 1) for value in bound}
            return found.pop() if len(found) == 1 else None
        if (
            isinstance(expression, ast.Attribute)
            and isinstance(expression.value, ast.Name)
            and expression.value.id == "self"
        ):
            cls = pathcond.class_of(tree, func)
            return self.of_field(file, cls, expression.attr, depth + 1) if cls is not None else None
        if isinstance(expression, ast.Call):
            called = expression.func
            constructed = self.class_named(file, called) if isinstance(called, (ast.Name, ast.Attribute)) else None
            if constructed is not None:
                return constructed
            if isinstance(called, ast.Attribute):
                receiver = called.value
                if isinstance(receiver, ast.Name) and receiver.id == "self":
                    cls = pathcond.class_of(tree, func)
                    owner = (file, cls) if cls is not None else None
                else:
                    owner = self.of_expression(file, func, receiver, depth + 1)
                if owner is None:
                    return None
                found = self.method(owner[0], owner[1], called.attr)
                if found is None:
                    return None
                where, method = found
                return self.annotated(where, method.returns)
        return None


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
    ) -> None:
        self.roots = roots
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
        self.steps: list[TraceStep] = []
        self.armed: dict[int, ast.stmt] = {}
        self.seen: set[tuple] = set()
        #: ``{(function, name): the bindings already walked}``, where a name is
        #: asked again from each statement that reads it -- so the second
        #: asking adds what *its* statement reaches and nothing twice.
        self.explained: dict[tuple[str, str], set[tuple[int, int, str]]] = {}
        self._scopes: dict[tuple[str, str], str] = {}
        self._logged: dict[tuple[str, str], list[tuple[ast.expr, _Message]]] = {}
        self.note = ""
        self.queue: deque[tuple[_Frame, str, int, Optional[ast.AST]]] = deque()
        self.types = _Types(roots, self.parsed)
        self._declared: dict[tuple[str, str, str], Optional[str]] = {}

    # -- source ------------------------------------------------------------

    def module(self, file: str) -> Optional[tuple[str, ast.Module]]:
        if file in self.parsed:
            return self.parsed[file]
        if len(self.parsed) >= self.budget.files:
            self._exhausted("files")
            return None
        package, _, relative = file.partition("/")
        root = self.roots.get(package)
        if root is None or not relative:
            # Outside the corpus the map covers, which is the walk's stop rule
            # rather than a failure: a name that comes from somewhere else is
            # exactly what a boundary is for.
            return None
        try:
            source = (root / relative).read_text(errors="replace")
            tree = ast.parse(source, filename=str(root / relative))
        except (OSError, SyntaxError, ValueError) as exc:
            logger.debug("trace: %s unreadable (%s)", file, exc)
            return None
        pathcond.attach_parents(tree)
        self.parsed[file] = (source, tree)
        return self.parsed[file]

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
            return self.frames[key]
        if len(self.frames) >= self.budget.functions:
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

    # -- steps -------------------------------------------------------------

    def _exhausted(self, what: str) -> None:
        if not self.note:
            self.note = (
                f"the walk's {what} budget ran out; what follows is part of the answer"
            )

    def record(self, step: TraceStep) -> bool:
        if len(self.steps) >= self.budget.steps:
            self._exhausted("steps")
            return False
        self.steps.append(step)
        return True

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
        return self.classify(expression, self.resolver(frame), self.typer(frame))

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
        """Whether names in *frame* are narrowed to the writes that reach their read."""
        return self.narrowed is not None and (frame.file, frame.owner) == self.narrowed

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
        if depth > self.budget.depth:
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
        for name in _guard_reads(statement):
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
            for test, names in _guard_tests(statement):
                self.want(frame, names, 1, test)

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
        binding that can reach it, each a way of its own, and what ``+=`` adds
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
        plain, appended, opaque = _message_sites(frame.func, spelled)
        plain = [value for value in plain if pathcond.can_reach(value, at)]
        appended = [value for value in appended if pathcond.can_reach(value, at)]
        opaque = [site for site in opaque if pathcond.can_reach(site, at)]
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
                arm for arm, held in self.armed.items() if pathcond.compatible(held, argument)
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

        if not any(here(node) for node in (*bindings, *loops, *caught)):
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
        corpus's exits, the span is the narrower answer 34 times and the
        cancelling rule is narrower 657 times, so neither contains the other.

        The starting frame only, and the same reason as :meth:`_within`: a
        deeper frame is another function, where this line number belongs to
        another read.

        **The read is the statement that read the name** -- *at* -- and not the
        line the walk started from.  Cancelling against the starting line at
        every depth answered "which writes reach line 1019" for a name read on
        line 1007, and ``memVal = int(memVal)`` then cancelled the very writes
        it reads.  The statement itself is never its own answer, because it
        reads before it writes; inside a loop its previous iteration's value
        still can be, and is kept -- as an answer, never as a canceller.
        """
        if not self.narrows(frame):
            return bindings, loops, caught
        target = at
        if target is None and self.reading_at is not None:
            target = _statement_at(frame.func, self.reading_at[2])
        if target is None:
            return bindings, loops, caught
        sites: dict[int, ast.AST] = {}
        for node in (*bindings, *loops, *caught):
            sites[id(node)] = (
                node
                if isinstance(node, (ast.stmt, ast.ExceptHandler))
                else pathcond.site_of(node)
            )
        others = [site for site in sites.values() if site is not target]
        # ``as_run``: a test on a name the function has since rebound says
        # nothing about the rebound value, and a ``try``'s write is skipped by
        # the exception its handler swallows -- see ``pathcond.surviving_writes``.
        alive = {
            id(site)
            for site in pathcond.surviving_writes(others, target, rereads, as_run=True)
        }
        if pathcond.loops_over(target):
            alive.add(id(target))

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
        bindings, loops, caught = self._within(frame, bindings, loops, caught)
        bindings, loops, caught = self._reaching(
            frame, bindings, loops, caught, at, [pathcond.site_of(node) for node in mutated]
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
        terminal, detail = self.classify(expression, resolve, typed)
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
        if self.record(step):
            self.want(frame, reads, depth + 1, loop)

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
                for expression in pathcond.attribute_expressions(method, field):
                    other = other or self._sibling(frame, method)
                    if other is None:
                        break
                    self._binding(other, name, expression, depth)
                    answered = True
        return answered

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
        parameter = _is_parameter(frame.func, name)
        if parameter:
            for handover in frame.handovers:
                if handover.reached_by == ARRIVES_BY_CALL and name in handover.fields:
                    self._handover(frame, name, handover, depth)
                    return
        self.record(
            TraceStep(
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
        )


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
) -> Walked:
    """Explain the arms at *lines*, and say why the answer is short when it is.

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
    while state.queue:
        held, name, depth, at = state.queue.popleft()
        state.explain(held, name, depth, at)
    return Walked(state.steps, state.note, state.skeleton(frame, region))
