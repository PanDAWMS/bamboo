"""Where the values a status's arm's guards held on come from: each term's end on the map's ways (S, step 1).

A status's arm its run shows ran ran under its guards (G, KNOWN_ISSUES.md K54).  What its question asks next is where
the values those guards held on came from, and the answer ends only when every term does (D1's C2).  A term is a name
a guard's test reads, as the map's ``GuardFrom`` records it at the test's line in the arm's function; it ends where
every way of it ends at one of:

* the database: a statement's rows, a sequence, a fetch no statement of the function reads, or a value the map's
  trace says the database supplies (``self.cur.rowcount``, ``self.cur.fetchone()``);
* a constant;
* an object filled from one fetched row and not changed in place on the way (the map's ``fills.row_fills``, as
  KNOWN_ISSUES.md K71 reads it).

Any other end stops the term, named: a parameter (its callers are read in the next step), a field of ``self`` (the
map's FieldFrom, SFF), a module's name, a call not read, a value not followed.  So does a way with changes on it: the
map's changes the value may take between where the way ends and the read are on the way, its end kept, each of the
object it changes (out/gm/gm5-design.md G4): those that change what the read takes off the value, a method the test
calls on it taking what the map's ``Runnable`` says it reads of its receiver (S4c's scope 2).  Which of them made what
the read reads is the walk's to read on the run (S4c): what a store stores (:func:`stored`) against what the run
printed, and which ran by its must rows; the one chosen is read as an arm (:class:`Made`).

A way ending at a parameter is read up the callers (:func:`param_ways`) to a constant a call hands, a parameter an
API's caller hands, the database, or a clock the caller reads.  Two or more constants one call hands, which the
caller's run cannot tell apart, are told apart by the ``if`` their bindings sit in (:class:`Split`, S2's (a)).  A way
ending at a call the map's ``values.clock_of`` reads as a clock ends there, as a field's way does (S2's (b)).  A
constant way with *steps* or *through* (an item of ``range(3)``) ends at the literal, but its value is not the
literal's.  The function's entry ends where no corpus call enters it, at the call a
chosen way came in by, or at the call a caller's run shows entered it (:data:`CALLER`).

A source with *paths* says where parts of its end sit in the value read (``self.job`` kept at ``.job`` of the plugin
``self.add_result`` is read off), not that the value is its end: it is no way of the value.  A term with only such
sources stops, named.  The map drops ``()`` from *paths*, so a source that is both the value and a part reads as a
part: a stop where there may be an end, never an end where there is none.

A call in a test (M, 1b-i) is read by the map's ``Runnable`` record of the definition asked at the test's line: the
map makes one for each corpus definition a call in an arm's path condition can land at.  A runnable definition reads
its receiver's fields: the receiver's own term says where they come from.  One the map says is not runnable because
it reaches the database or another system gives that system's answer, and a name the test reads only as the receiver
of such calls ends there (``self`` in ``self._commit()``).  A call the map asks no definition of at the line (a
builtin, a library's) is read through its receiver and arguments.  No definition is run here.
"""

from __future__ import annotations

import ast
import functools
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from dataclasses import field as dc_field
from typing import Any

from bamboo.codemap import callfacts, trace
from bamboo.codemap import models as M
from bamboo.codemap.panda import fills, heldfrom

from . import asked, mapdump
from . import printed as PR
from . import rows as R
from . import which_ran as WR

DATABASE, ROW, CONSTANT, OUTSIDE, STOP = "database", "row", "constant", "outside", "stop"
#: a value an API's caller handed: a parameter no call of the corpus hands
API = "api"
#: a function no call of the corpus enters (a daemon's loop, an API's endpoint): its entry is where it starts
ENTRY = "entry"
#: the call a caller's run shows entered the arm's function (F1): why the caller made it is the caller's arm's question
CALLER = "caller"
#: a value read off a clock: a call the map's :func:`values.clock_of` reads as one (``naive_utcnow()``)
CLOCK = "clock"
#: a term its run's test did not read: an earlier operand of an ``and`` or ``or`` decided the test on the run
#: (out/gm/qk-design.md Q6's (c))
UNREAD = "unread"
#: the ends a term may end at; a term with any other end stops
ENDS = (DATABASE, ROW, CONSTANT, OUTSIDE, API, ENTRY, CALLER, CLOCK, UNREAD)
#: a way that ends at a parameter of the arm's function: read up its callers next
PARAMETER = M.KEY_PARAMETER
#: a term the map holds no GuardFrom for at its test's line
NO_RECORD = "no-record"
#: a name or parameter the map holds only sources with *paths* for: where parts of other values sit, no way of its own
PARTS_ONLY = "parts-only"
#: a constant way with *steps* or *through*: it ends at the literal, its value is something made of it
THROUGH = "constant-through"
#: a field of a class the map holds no FieldFrom for
NO_FIELD = "no-field-from"
#: a field whose FieldFrom gives no way: no binding, change or stop, or none S4b's choice took
NO_WAY = "no-way"
#: a container made empty (the map's ``made_empty``) a read takes an element of, where a put of it is not read (gone
#: where the map does not follow it, a call not read, a put with no value) or the step may take a mapping's key: the
#: values the puts the map reads put are ways, and the container stays one (out/gm/qk-design.md Q1)
CONTAINER = "container"
#: the empty start of a container made empty, read whole: a constant, but no literal a run prints (Q1's whole read)
MADE_EMPTY = "made-empty"
#: a statement putting into a container made empty that a test reads for its truth alone: what made it not empty,
#: a constant to the test, but no literal a run prints (out/gm/qk-design.md Q6's (a))
PUT = "put"
#: the constants no literal a run prints is: a part of one, a container's empty start, a put into it
NO_LITERAL = (THROUGH, MADE_EMPTY, PUT)


def element_kept(change: Any, steps: tuple[str, ...]) -> bool:
    """Whether *change* is on the way of a read taking *steps* off the value: a read of an element (its first step
    ``[...]``) reads a value some put put, which a change taking elements out or reordering them (``pop``, ``remove``,
    ``sort``, ``del x[k]``) does not make (out/gm/qk-design.md Q1); every change counts for any other read."""
    if not steps or not steps[0].startswith("["):
        return True
    kind, what = getattr(change, "kind", ""), getattr(change, "what", "") or ""
    if kind == M.CHANGE_METHOD and what in callfacts.CHANGING and what not in callfacts.PUTTING:
        return False
    return not (kind == M.CHANGE_DELETE and not str(getattr(change, "key", "") or "").startswith("."))


def _elements(s: Any, rest: tuple[str, ...]) -> tuple[list[Any], bool] | None:
    """Q1: where the source *s* ends at a container made empty, whole, and the read takes an element of it (*rest*'s
    first step) or the container whole (no step), the ways of the values its puts put that the step may take -- every
    value, for the whole -- (the map's ``heldfrom.held_elements``: each with the calls to its put and *s*'s route after
    its own, the changes that may change an element on it) and whether *s* stays a way too (a put not read; a mapping
    read by ``[*]`` or whole, whose keys no put's value holds); None where it does not, which reads *s* as before.  A
    whole read also has the container's empty start (:func:`_start`)."""
    if not getattr(s, "made_empty", "") or getattr(s, "steps", None) or getattr(s, "through", None):
        return None
    if not rest:
        values, stays = heldfrom.held_elements(s, "[*]")
        # nothing done to it: the empty start is all it holds
        return values, stays and bool(getattr(s, "changes", None))
    if not rest[0].startswith("["):
        return None
    return heldfrom.held_elements(s, rest[0])


def _put_read(change: Any) -> bool:
    """Whether *change* is a put whose value the map reads: Q1 reads it as a way of the container's contents."""
    return bool(getattr(change, "value", None)) and getattr(change, "kind", "") in (M.CHANGE_METHOD, M.CHANGE_STORE)


def _puts_into(change: Any) -> bool:
    """Q6's (a): whether *change*, of a container, puts into it -- an item's store, or a method that puts -- whether
    or not the map reads what it puts: a container made empty is not empty after one that put something."""
    kind = getattr(change, "kind", "")
    if kind == M.CHANGE_STORE:
        return not str(getattr(change, "key", "") or "").startswith(".")
    return kind == M.CHANGE_METHOD and (getattr(change, "what", "") or "") in callfacts.PUTTING


def _empties(change: Any) -> bool:
    """Q6's (a): whether *change*, of a container, may leave it empty after a put, or is not read: a method that takes
    out of it (``pop``, ``clear``), an item's delete, the container gone where it is not followed, a call not read."""
    kind = getattr(change, "kind", "")
    if kind in (M.CHANGE_ESCAPED, M.CHANGE_UNREAD):
        return True
    if kind == M.CHANGE_METHOD:
        what = getattr(change, "what", "") or ""
        return what in callfacts.CHANGING and what not in callfacts.PUTTING and what not in _REORDERING
    return kind == M.CHANGE_DELETE and not str(getattr(change, "key", "") or "").startswith(".")


#: a container's methods that change it by reordering what it holds, not by taking any out
_REORDERING = frozenset({"sort", "reverse"})


def _truth_ways(s: Any, route: tuple[str, ...], level: tuple[Changed, ...] = ()) -> list[Way] | None:
    """Q6's (a): the ways of a container made empty, *s* (its source, read whole), that a test reads for its truth
    alone (:func:`truth_only`): whether a put ran is what the test reads, not what it put.  Its empty start; each
    statement putting into it, of the map's changes of it and *level*'s (a field record's), placed at the put and
    through the calls it was made by, then *route*; and the container, where a change may take out of it or is not
    read.  None for any other source.  A change of what an element holds, or a store of an attribute, leaves it as
    empty as it was."""
    if not getattr(s, "made_empty", "") or getattr(s, "steps", None) or getattr(s, "through", None):
        return None
    changes = [*(getattr(s, "changes", None) or ()), *(c.record for c in level if c.record is not None)]
    here = next(iter(getattr(s, "at", None) or ()), "")
    ways = [Way(CONSTANT, "", MADE_EMPTY, route, end_at=here)]
    ways += [Way(CONSTANT, "", PUT, (*reversed(tuple(getattr(c, "via", None) or ())), *route), end_at=c.at)
             for c in changes if _puts_into(c)]
    if any(_empties(c) for c in changes):
        ways.append(Way(STOP, "", CONTAINER, route, end_at=here))
    return ways


def truth_only(test: ast.expr, name: str) -> bool:
    """Q6's (a): whether *test* reads the value *name* holds for its truth alone: every place it names it is the test
    itself, an operand of ``not``, ``and`` or ``or`` read so, the test of a conditional expression, ``bool(x)`` or
    ``len(x)`` read so, or ``len(x)`` compared with 0 by ``==``, ``!=`` or ``>`` (``0 <`` too).  ``len(x) > 3`` reads how
    many it holds, not whether it is empty (out/gm/qk-design.md Q6, counter-case 1)."""
    parents = _parents(test)

    def truth(node: ast.AST) -> bool:
        # a value read for its truth: up through not, and, or, to the test or a conditional expression's test
        while True:
            p = parents.get(id(node))
            if p is None:
                return True
            if isinstance(p, ast.IfExp):
                return p.test is node
            if not (isinstance(p, ast.BoolOp) or (isinstance(p, ast.UnaryOp) and isinstance(p.op, ast.Not))):
                return False
            node = p

    def zero(n: ast.AST) -> bool:
        return isinstance(n, ast.Constant) and n.value == 0 and not isinstance(n.value, bool)

    found = False
    for n in ast.walk(test):
        if not isinstance(n, (ast.Name, ast.Attribute)) or ast.unparse(n) != name:
            continue
        found = True
        read: ast.AST = n
        call = parents.get(id(n))
        if (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id in ("bool", "len")
                and len(call.args) == 1 and call.args[0] is n and not call.keywords):
            read = call
            cmp = parents.get(id(call))
            if call.func.id == "len" and isinstance(cmp, ast.Compare) and len(cmp.ops) == 1:
                op, left, right = cmp.ops[0], cmp.left, cmp.comparators[0]
                if not ((left is call and zero(right) and isinstance(op, (ast.Eq, ast.NotEq, ast.Gt)))
                        or (right is call and zero(left) and isinstance(op, (ast.Eq, ast.NotEq, ast.Lt)))):
                    return False
                read = cmp
        if not truth(read):
            return False
    return found


def _guard_ends(term: str, s: Any, rel: str, keys: frozenset[str] | None, line: int, depth: int = 0,
                rest: tuple[str, ...] = ()) -> list[End]:
    """The ends of a guard's source *s* (:func:`_source_end`), *rest* the steps the test takes past it (an element,
    :func:`item_step`).  For a container made empty (Q1): read whole, its empty start and the ends of each value its
    puts put, read in turn as parts of it; an element of it, the ends of each value a put put that the step may take,
    read with the steps after it -- and the container where a put is not read or it is a mapping, the changes left on
    them those that are no put read."""
    got = _elements(s, rest) if depth < 4 else None
    first = _source_end(term, s, rel, keys, line, read=rest)
    if got is None:
        return [first]
    values, stays = got
    left = tuple(c for c in first.changes if c.record is None or not _put_read(c.record))
    ends = [] if rest else [End(term, CONSTANT, MADE_EMPTY, first.at, changes=left)]
    for v in values:
        if not _cut(v):
            held = _guard_ends(term, v, rel, keys, line, depth + 1, rest[1:])
            ends += held if rest else [as_part(e) for e in held]
    return ends + ([replace(first, end=STOP, why=CONTAINER, changes=left)] if stays else [])


def as_part(w: Any) -> Any:
    """A :class:`Way` or :class:`End` reached through a value a container read whole holds: a part of the value read,
    so a literal there ends it as a constant whose value is not the literal's (:data:`THROUGH`), as an item of
    ``range(3)`` does (Q1's whole read)."""
    if w.end != CONSTANT or w.why in (THROUGH, MADE_EMPTY):
        return w
    return replace(w, value="", why=THROUGH) if isinstance(w, Way) else replace(w, why=THROUGH)


def _start(s: Any, route: tuple[str, ...], rest: tuple[str, ...]) -> list[Way]:
    """A container made empty read whole (no step, *rest* empty): its empty start, a constant no run prints (Q1)."""
    return [] if rest else [Way(CONSTANT, "", MADE_EMPTY, route, end_at=next(iter(getattr(s, "at", None) or ()), ""))]


def _ways_of_the_value(sources: list[Any]) -> tuple[list[Any], bool]:
    """The sources that are ways of the value itself (no *paths*), and whether any source said only where parts sit."""
    whole = [s for s in sources if not getattr(s, "paths", None)]
    return whole, len(whole) < len(sources)


@dataclass(frozen=True)
class Changed:
    """One change on a way, between where the way ends and the read (out/gm/gm5-design.md G4): the map's
    ``HeldChange`` -- its *at*, *kind* and *key*, *made_by* the calls it was made through in the order run (its
    *via*), and the record itself (what a store stores), which is not compared."""

    at: str
    kind: str
    key: str
    made_by: tuple[str, ...] = ()
    record: Any = dc_field(default=None, compare=False)
    #: the read its record's choice placed it before (``file:line``): the arm's test, or the binding a field of an
    #: object one level in is read at (S4c's V reads a print after that binding only where every change is before it)
    before: str = dc_field(default="", compare=False)


def _changed(c: Any, key: str | None = None, before: str = "") -> Changed:
    """The map's change *c* as a way holds it; *key* in place of its own (``*``: one handed in, G4 rule 2)."""
    return Changed(c.at, getattr(c, "kind", "") or "", key if key is not None else str(getattr(c, "key", "*") or "*"),
                   tuple(getattr(c, "via", None) or ()), c, before)


@dataclass(frozen=True)
class End:
    """Where one way of a term ends: *end* one of :data:`ENDS` or :data:`STOP`; *why* the map's kind of the end (or
    the reason it stops); *at* ``file:line`` where the way ends, where the map says it.  A way with *changes* left on
    it does not end there: what the read reads may be what one of them made (G4)."""

    term: str
    end: str
    why: str
    at: str = ""
    #: a parameter's way: the parameter's name (to read up its callers)
    param: str = ""
    #: the calls the value came in by, the outermost first (``file:line``): a parameter's chosen way
    route: tuple[str, ...] = ()
    #: a parameter's way reached through a field's binding: the method whose parameter it is (``file::name``); empty for
    #: the arm's own function
    of: str = ""
    #: a field's way: each binding it went through, ``file:line``, the read's first (S4a)
    via: tuple[str, ...] = ()
    #: a parameter's way reached through a field's binding: the calls the run's path entered *of* by (``file:line``,
    #: S4b's (e)); empty where any of its callers
    entered: tuple[str, ...] = ()
    #: the changes on the way between where it ends and the read, each of the object it changes (G4)
    changes: tuple[Changed, ...] = ()
    #: a way that came back through calls: each call, in the order the value went (``file:line``, the map's *route*,
    #: the last one in the function the way is read in; G2)
    returned: tuple[str, ...] = ()
    #: and the line the value came from, in the function the first of them calls (a file the map does not name)
    returned_line: int = 0
    #: S4c: the change its run shows made what the read reads, its site read as an arm (*changes* is then that one)
    made: Made | None = None
    #: the ``except`` handler (its line) the statement the way ends at is in, in the arm's function: S3's to read
    handler: int = 0
    #: S2's (a): a parameter's constants one call hands, told apart by the ``if`` their bindings sit in
    split: Split | None = None
    #: a parameter's way: the steps the read still takes off the parameter's value (``[*]``, ``[3]``), read on up its
    #: callers (out/gm/qk-design.md Q2's (c))
    steps: tuple[str, ...] = ()
    #: K: a way read off the name an ``except`` handler of the arm's function binds (the handler's line): a field of the
    #: exception caught, read up to the raise sites; the run's traceback names the raise (rerun)
    caught: int = 0
    #: Q6's (a): a parameter's way the test reads for its truth alone (:func:`truth_only`), read up its callers so
    truth: bool = False
    #: Q6's (c): a term its run's test did not read (:data:`UNREAD`): the operand whose value on the run decided it
    decided: str = ""
    #: Q7 (out/gm/qk-design.md): a way ending at an object a class's constructor made: that class (``file::Class``),
    #: whose ``__init__`` binds what the object holds of each part at its start
    made_of: str = ""
    #: Q7's (a): a way ending at an object filled from one fetched row, with other writes of what the test takes after
    #: the fill: the fill's statements (``file:line``) -- the row is the start, the other writes its changes
    filled: tuple[str, ...] = ()
    #: Q7's (c): a way ending at the statement that last wrote the part the test takes (*part*, ``.status_code``) on the
    #: run -- the change S4c chose (*made*), or the object's start -- of the object made at *made_at*
    part: str = ""
    made_at: str = ""


@dataclass(frozen=True)
class Made:
    """S4c (out/gm/gm5-design.md): of the changes on a way, the one the run's printed value and its run leave as what
    made what the read reads -- *how*: ``shown`` (its run shows it ran) or ``left`` (by elimination: every other shown
    not to have, its function's rows in the run) -- read as an arm is, at *site*, the deepest of the calls it was made
    through and its statement H2 left, in *fn* (``file::name``): its guards, each term's ends, and its entry."""

    change: Changed
    site: str
    fn: str
    how: str = "shown"
    guards: tuple[str, ...] = ()
    terms: tuple[Term, ...] = ()
    entry: End | None = None
    #: K's form, the arm in an ``except`` handler: the exception its run's traceback shows the handler took (its type
    #: as Python prints it), where it was raised and the handler (``file:line`` each); empty where not shown
    raised_type: str = ""
    raised_at: str = ""
    caught_at: str = ""
    #: where the pin does not hold the raising frame's source at *raised_at* (the run's code was another version): its
    #: function and the source the traceback printed
    raised_other: str = ""
    #: Q7's (b), a change left by elimination: the part it writes (``.status_code``) and what the object held of it at
    #: its start (:func:`part_start`'s literals), which the run's print differs from where the change wrote what the
    #: read reads (*printed_start*: one of them is what the run printed, so the run does not tell the change from the
    #: start); *start* empty where not read, and then *no_record* the field the map holds no FieldFrom of
    #: (``file::Class.field``; empty where the way's end is no object a class made)
    part: str = ""
    start: tuple[str, ...] = ()
    printed_start: bool = False
    no_record: str = ""


@dataclass(frozen=True)
class Split:
    """S2's (a) (out/gm/s4-design.md rule 6, gm5-design.md): the constants one call hands a parameter, which the
    caller's run cannot tell apart, told apart by the ``if`` statements their bindings sit in different arms of --
    *ifs* each one's ``file:line``, in *fn* (``file::name``), the call at *call*; *guards* as the bindings' path
    conditions spell them -- and each term of those tests with its ends, by the map's records alone: the caller's run
    is not asked for them, so no field choice, change reading (S4b, S4c) or entry is read there.  *asked*: what the
    caller's run said of which binding ran (Q5) -- :data:`NOT_ASKED`, or the kind of the answer that did not tell it
    (:data:`Q5_ANSWERS`)."""

    call: str
    fn: str
    ifs: tuple[str, ...]
    guards: tuple[str, ...] = ()
    terms: tuple[Term, ...] = ()
    asked: str = ""


#: Split.asked: Q5 not reached (another way left, the walk not asking, a binding not in the caller)
NOT_ASKED = ""
#: Split.asked, Q5 reached and not told: no log of the caller to hand; asked and not answered (a binding unexamined);
#: answered, none shown; answered, two or more shown
Q5_ANSWERS = ("no-log", "unexamined", "none-shown", "several-shown")


@dataclass(frozen=True)
class Way:
    """One way a parameter's value comes in, read up the map's ``ParameterFrom`` records: *end* :data:`CONSTANT`
    (*value* the literal as written), :data:`API`, :data:`DATABASE`, or :data:`STOP` (*why* the map's kind or stop);
    *route* each call on the way, the outermost first (``file:line``)."""

    end: str
    value: str
    why: str
    route: tuple[str, ...]
    #: a way ending at a parameter of a field binding's method (*value* the parameter): that method (``file::name``)
    of: str = ""
    #: and the calls the run's path entered that method by (``file:line``, S4b's (e)); empty where any of its callers
    entered: tuple[str, ...] = ()
    #: as :attr:`End.changes`, :attr:`End.returned` and :attr:`End.returned_line` (G4, G2)
    changes: tuple[Changed, ...] = ()
    returned: tuple[str, ...] = ()
    returned_line: int = 0
    #: a parameter's way ending at a constant bound in the function the first call of *route* is in: the line of that
    #: binding (S2's (a) reads the ``if`` it sits in); 0 where the map's line is in a callee the value came back from
    bound_at: int = 0
    #: a way ending at a parameter: the steps the read still takes off its value (Q2's (c))
    steps: tuple[str, ...] = ()
    #: where the way ends, ``file:line`` (the map's *at* of its source; SBG), where the map says it
    end_at: str = ""
    #: as :attr:`End.made_of` (Q7)
    made_of: str = ""
    #: a way that came across a send (the map's ``HeldSource.sent``, SND): the client's call that sent it, the last of
    #: that source's route (``file:line``), and the endpoint function the send entered (``file::name``); the send
    #: nearest the way's end where it came across more than one (:func:`param_ways`)
    sent: str = ""
    sent_into: str = ""


def _made_of(s: Any) -> str:
    """Q7: the class (``file::Class``) of the object the map's source *s* says a constructor made; empty for any other
    source."""
    return str(getattr(s, "site_owner", "") or "") if s.kind == M.KEY_OBJECT else ""


def _returned(s: Any) -> dict[str, Any]:
    """What a source that came back through calls holds of them (G2): its *route* and its own line; empty for one that
    did not."""
    route = tuple(getattr(s, "route", None) or ())
    return {"returned": route, "returned_line": getattr(s, "line", None) or 0} if route else {}


@dataclass(frozen=True)
class Term:
    """One term of a guard: the name (or call) the test reads and each way's end."""

    guard: str
    term: str
    ends: tuple[End, ...]
    #: the line of the test it is read at
    line: int = dc_field(default=0, compare=False)

    @property
    def ended(self) -> bool:
        return bool(self.ends) and all(e.end in ENDS and not e.changes and arm_ends(e.made) for e in self.ends)


def arm_ends(m: Made | None) -> bool:
    """Q7's (d): whether the arm of the change a way ends at (S4c's *made*) ends -- its terms and its entry ending, and
    no raise its run's traceback shows the handler took (K's form: the raise has terms of its own) -- or there is none."""
    return m is None or (not m.raised_at and m.entry is not None and all(u.ended for u in m.terms)
                         and m.entry.end in ENDS and not m.entry.changes)


@functools.cache
def _index(fragment_id: int) -> tuple[dict, dict]:
    fragment = mapdump.fragment()
    guards: dict[tuple[str, int], list] = {}
    for g in getattr(fragment, "guard_froms", None) or []:
        guards.setdefault((g.owner, g.test_line), []).append(g)
    runs: dict[str, list] = {}
    for r in getattr(fragment, "runnables", None) or []:
        runs.setdefault(r.owner.rpartition("::")[2], []).append(r)
    return guards, runs


def _maps() -> tuple[dict, dict]:
    fragment = mapdump.fragment()
    return _index(id(fragment)) if fragment is not None else ({}, {})


def call_end(rel: str, test_line: int, call: ast.Call) -> End | None:
    """The end of a call in a test at *test_line* of *rel* by the map's Runnable record of the definition asked there
    (M, 1b-i); None where the call is read through its receiver and arguments: one the map asks no definition of there
    (a builtin, a library's), or a runnable one."""
    _guards, runs = _maps()
    callee = ast.unparse(call.func)
    name = call.func.attr if isinstance(call.func, ast.Attribute) else call.func.id if isinstance(call.func, ast.Name) else ""
    sites = {f"{rel}:{test_line}", f"{rel}:{call.lineno}"}
    records = [r for r in runs.get(name, []) if sites & set(r.asked_at)]
    at = f"{rel}:{call.lineno}"
    if not records:
        return None
    if all(r.runnable for r in records):
        return None
    if any(r.runnable for r in records):
        return End(f"{callee}()", STOP, "runnable-in-part", at)
    kinds = {s.kind for r in records for s in r.stops}
    if M.RUN_DATABASE in kinds:
        return End(f"{callee}()", DATABASE, M.RUN_DATABASE, next(s.at for r in records for s in r.stops if s.kind == M.RUN_DATABASE))
    if M.RUN_OUTSIDE in kinds:
        return End(f"{callee}()", OUTSIDE, M.RUN_OUTSIDE, next(s.at for r in records for s in r.stops if s.kind == M.RUN_OUTSIDE))
    return End(f"{callee}()", STOP, ",".join(sorted(kinds)) or "not-runnable", at)


def _parents(node: ast.AST) -> dict[int, ast.AST]:
    out: dict[int, ast.AST] = {}
    for parent in ast.walk(node):
        for child in ast.iter_child_nodes(parent):
            out[id(child)] = parent
    return out


def _receiver_only(test: ast.expr, name: str, ended: dict[int, End]) -> End | None:
    """The end of the calls *name* is the receiver of, where the test reads it only as such and each is a system's
    answer (:func:`call_end`'s database or outside); else None."""
    parents = _parents(test)
    got: list[End] = []
    for node in ast.walk(test):
        if not isinstance(node, (ast.Name, ast.Attribute)) or ast.unparse(node) != name:
            continue
        attr = parents.get(id(node))
        call = parents.get(id(attr)) if isinstance(attr, ast.Attribute) and attr.value is node else None
        if not isinstance(call, ast.Call) or call.func is not attr or id(call) not in ended:
            return None
        got.append(ended[id(call)])
    return got[0] if got and all(e.end in (DATABASE, OUTSIDE) for e in got) else None


def _source_end(term: str, s: Any, rel: str, keys: frozenset[str] | None = None, line: int = 0,
                read: tuple[str, ...] = ()) -> End:
    """Where the way of a guard's source *s* ends, in *rel*, *read* the steps the test takes past it (an element,
    :func:`item_step`).  One that came back through calls is placed at the last of them, the call in *rel* (its own
    line is the callee's, in a file the record does not name), and holds them (G2).  The changes the map says may be
    made to it before the read (at *line*) are on its way, its own end kept (G4): those that change what the read takes
    off the value, *keys* (S4c's rule 1), and, for an element, none that only takes elements out or reorders them
    (:func:`element_kept`)."""
    back: dict[str, Any] = _returned(s)
    route = back.get("returned", ())
    at = (f"{s.site_owner.partition('::')[0]}:{s.site_line}" if getattr(s, "site_owner", "") and getattr(s, "site_line", None)
          else route[-1] if route else f"{rel}:{s.line}" if getattr(s, "line", None) else "")
    taken = tuple(getattr(s, "steps", None) or ())
    off = _keys_off(taken, keys)
    back["changes"] = tuple(_changed(c, before=f"{rel}:{line}" if line else "")
                            for c in getattr(s, "changes", None) or ()
                            if change_counts(c, off) and element_kept(c, (*taken, *read)))
    if s.kind in (M.KEY_ROWS, M.KEY_SEQUENCE, M.KEY_DATABASE):
        return End(term, DATABASE, s.kind, at, **back)
    if s.kind == M.KEY_CONSTANT:
        # an item of range(3) ends at the literal, its value not the literal's (as a parameter's and a field's way)
        return End(term, CONSTANT, THROUGH if taken or read or getattr(s, "through", None) else s.name, at, **back)
    if s.kind == M.KEY_PARAMETER:
        # read up the callers with what the read takes off it (Q2's (c))
        return End(term, STOP, s.kind, at, param=s.name, steps=(*taken, *read), **back)
    if s.kind == M.KEY_CALL and _clock_at(at, s.name or ""):
        # a call the map reads as a clock (timeNow = naive_utcnow()), as a field's way ends (S2's (b))
        return End(term, CLOCK, s.name, at, **back)
    return End(term, STOP, s.kind, at, made_of=_made_of(s), **back)


def caught_name(fn: R.Function, s: Any) -> ast.ExceptHandler | None:
    """K: the ``except`` handler of *fn* whose name the guard's source *s* is, read with a field step (``e.fatal``):
    the map's object of the class caught (its *site_owner*) placed at the handler's own line (EXF); None for any other
    source.  The handler is told by where the map places the object, not by its name's spelling."""
    steps = tuple(getattr(s, "steps", None) or ())
    if s.kind != M.KEY_OBJECT or not getattr(s, "site_owner", "") or not steps or not steps[0].startswith("."):
        return None
    rel, _, ln = next(iter(getattr(s, "at", None) or ()), "").rpartition(":")
    if rel != fn.rel or not ln.isdigit():
        return None
    return next((h for h in ast.walk(fn.node) if isinstance(h, ast.ExceptHandler) and h.name and h.lineno == int(ln)),
                None)


@functools.cache
def _objects_placed(fragment_id: int) -> dict[tuple[str, int], str]:
    fragment = mapdump.fragment()
    out: dict[tuple[str, int], str] = {}
    for g in getattr(fragment, "guard_froms", None) or ():
        for s in g.sources:
            if s.kind == M.KEY_OBJECT and getattr(s, "site_owner", ""):
                for at in getattr(s, "at", None) or ():
                    rel, _, ln = at.rpartition(":")
                    if ln.isdigit():
                        out.setdefault((rel, int(ln)), s.site_owner)
    return out


def handler_classes(fn: R.Function) -> dict[int, str]:
    """K: the class each ``except`` handler of *fn* catches (``file::Class``) by the handler's line, where the map holds
    an object of it placed at the handler (EXF: the handler's name some guard reads) -- the handler's fact, whichever
    guard's record holds it."""
    fragment = mapdump.fragment()
    placed = _objects_placed(id(fragment)) if fragment is not None else {}
    return {h.lineno: placed[(fn.rel, h.lineno)] for h in ast.walk(fn.node)
            if isinstance(h, ast.ExceptHandler) and (fn.rel, h.lineno) in placed}


def caught_ends(term: str, s: Any, h: ast.ExceptHandler, fn: R.Function, one: Any, choose: Choose | None) -> list[End]:
    """K: the ends of a field of the exception the handler *h* names (*s*, :func:`caught_name`): the field's ways
    (:func:`field_ways`), up to its class's ``__init__``'s parameter and the raises that hand it, each marked with the
    handler and the class caught (the run's traceback picks the raise, rerun); the changes the map says may be made to
    the object before the read on each."""
    steps = tuple(s.steps)
    keys = read_keys(one.test, term, fn.rel, one.lineno)
    on = _source_end(term, s, fn.rel, keys, one.lineno).changes
    return [replace(_field_end(term, w), caught=h.lineno, changes=(*w.changes, *on))
            for w in field_ways(s.site_owner, steps[0][1:], steps[1:], choose=choose, at=(fn.rel, one.lineno), keys=keys)]


@functools.cache
def _field_index(fragment_id: int) -> dict[tuple[str, str], Any]:
    fragment = mapdump.fragment()
    return {(r.owner, r.field): r for r in getattr(fragment, "field_froms", None) or []}


def _field_froms() -> dict[tuple[str, str], Any]:
    fragment = mapdump.fragment()
    return _field_index(id(fragment)) if fragment is not None else {}


def fields_read(test: ast.expr, source: Any, own: set[str]) -> list[str]:
    """The fields of ``self`` a way ending at *source* (``self_field``) is read for: ``self.job`` -> ``job``; plain
    ``self`` -> each ``self.<f>`` the test reads that it does not call and that has no GuardFrom of its own at the line
    (*own*: ``self.add_result`` at 440 has)."""
    name = getattr(source, "name", "") or ""
    if name.startswith("self."):
        return [name.split(".")[1].split("[")[0]]
    if name != "self":
        return []
    parents = _parents(test)
    out: list[str] = []
    # in the order the test is written
    reads = sorted((n for n in ast.walk(test) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "self"),
                   key=lambda n: (n.lineno, n.col_offset))
    for n in reads:
        call = parents.get(id(n))
        if isinstance(call, ast.Call) and call.func is n:
            continue
        if f"self.{n.attr}" not in own and n.attr not in out:
            out.append(n.attr)
    return out


def _clock_at(at: str, text: str) -> bool:
    """Whether the call *text* at *at* (``file:line``) reads a clock, by the map's :func:`values.clock_of`."""
    from bamboo.codemap import values

    rel, _, line = at.rpartition(":")
    try:
        module = R.tree(rel)
    except (OSError, SyntaxError, KeyError):
        return False
    return any(isinstance(n, ast.Call) and n.lineno == int(line) and ast.unparse(n.func) == text and values.clock_of(n)
               for n in ast.walk(module))


#: S4b's choice (``rerun`` binds it to the run): ``(the FieldFrom, rel, line read at, keys read, whether the field is
#: one of self) -> field_order.Choice``
Choose = Callable[[Any, str, int, "frozenset[str] | None", bool], Any]


def _receiver_reads(rel: str, line: int, call: ast.Call) -> frozenset[str] | None:
    """The fields of its receiver a method a test calls at *line* of *rel* reads, by the map's ``Runnable`` records of
    the definitions asked there (their *reads* of ``self``; S4c's scope 2, out/gm/gm5-design.md); None where the map
    asks none there or one is not runnable: the whole value.  The map's *reads* are the definition's own statements'."""
    if not rel or not isinstance(call.func, ast.Attribute):
        return None
    _guards, runs = _maps()
    sites = {f"{rel}:{line}", f"{rel}:{call.lineno}"}
    records = [r for r in runs.get(call.func.attr, []) if sites & set(r.asked_at)]
    if not records or not all(r.runnable for r in records):
        return None
    return frozenset(x.split(".")[1].split("[")[0] for r in records for x in r.reads if x.startswith("self."))


def parts_taken(test: ast.expr, name: str, rel: str = "", line: int = 0) -> frozenset[str] | None:
    """What *test* (at *line* of *rel*) takes off the value *name* holds, each one step: an attribute it reads off it
    (``.status_code``), those a method it calls on it reads (:func:`_receiver_reads`), an item (``[disp]``); empty where
    it tests only which object it is (``is None``), None where it reads the whole value (a method the map does not say
    it may run, the value handed on)."""
    parents = _parents(test)
    parts: set[str] = set()
    for n in ast.walk(test):
        if not isinstance(n, (ast.Name, ast.Attribute)) or ast.unparse(n) != name:
            continue
        p = parents.get(id(n))
        if isinstance(p, ast.Attribute) and p.value is n:
            call = parents.get(id(p))
            if isinstance(call, ast.Call) and call.func is p:
                if (reads := _receiver_reads(rel, line, call)) is None:
                    return None
                parts |= {f".{r}" for r in reads}
                continue
            parts.add(f".{p.attr}")
        elif isinstance(p, ast.Subscript) and p.value is n:
            parts.add(ast.unparse(p)[len(name):])
        elif isinstance(p, ast.Compare) and all(isinstance(o, (ast.Is, ast.IsNot)) for o in p.ops):
            continue
        else:
            return None
    return frozenset(parts)


def item_step(test: ast.expr, name: str, rel: str = "", line: int = 0) -> str | None:
    """The step an element read takes where *test* takes only items off the value *name* holds (``disp_error[disp]``),
    spelled as the map spells a key: a literal as written (``['total']``), ``[*]`` for any other key or several; None
    where it takes anything else, or nothing (:func:`parts_taken`)."""
    parts = parts_taken(test, name, rel, line)
    if not parts or any(not p.startswith("[") for p in parts):
        return None
    keys = {ast.unparse(n.slice) if isinstance(n.slice, ast.Constant) else "*" for n in ast.walk(test)
            if isinstance(n, ast.Subscript) and ast.unparse(n.value) == name}
    return f"[{keys.pop()}]" if len(keys) == 1 else "[*]"


def read_keys(test: ast.expr, name: str, rel: str = "", line: int = 0) -> frozenset[str] | None:
    """The attributes *test* takes off the value *name* holds (:func:`parts_taken`); None where it reads the whole value
    or an item of it."""
    parts = parts_taken(test, name, rel, line)
    if parts is None or any(not p.startswith(".") for p in parts):
        return None
    return frozenset(p[1:] for p in parts)


def change_counts(change: Any, keys: frozenset[str] | None) -> bool:
    """Whether *change* changes what a read taking *keys* off the value reads (None: the whole value; empty: which object
    it is, which no change in place makes another).  A change by a key it does not say (``*``) counts
    (out/gm/s4-design.md rule 4)."""
    if keys is not None and not keys:
        return False
    key = str(getattr(change, "key", "*") or "*")
    return keys is None or key == "*" or key.lstrip(".") in keys


def _keys_off(steps: tuple[str, ...], keys: frozenset[str] | None) -> frozenset[str] | None:
    """What a read taking *steps* off a value, then *keys* off what they reach, takes off the value itself."""
    if not steps:
        return keys
    return frozenset({steps[0][1:]}) if steps[0].startswith(".") else None


def _line_of(at: str) -> tuple[str, int]:
    rel, _, line = at.rpartition(":")
    return rel, int(line)


def _with_changes(ways: list[Way], changes: tuple[Changed, ...]) -> list[Way]:
    """*ways* with *changes* on each, after those already on it (G4)."""
    return [replace(w, changes=(*w.changes, *changes)) for w in ways] if changes else ways


def field_ways(cls: str, field: str, steps: tuple[str, ...] = (), seen: frozenset = frozenset(),
               choose: Choose | None = None, at: tuple[str, int] | None = None,
               keys: frozenset[str] | None = None, of_self: bool = False,
               handed: tuple[Changed, ...] = (), truth: bool = False) -> list[Way]:
    """Each way the field *field* of an object of *cls* (``file::Class``) is bound, read on through the map's
    ``FieldFrom`` records (S4a, out/gm/s4-design.md): a binding's value read as a way is (a field of ``self`` or of an
    object a corpus class builds read on in its own record, *steps* the parts of it the read takes), a parameter of the
    binding's method (read up its callers next), a constant, the database, a clock (:func:`_clock_at`), or a stop.
    With *choose* and *at* (the read's ``rel`` and line), only the bindings the read may have taken on its run's path
    and the changes that may have run between them and it (S4b, :mod:`field_order`; *keys* what the read takes off the
    value *steps* reach; *of_self* the field is one of ``self`` where it is read); without, every binding and change.
    Each of the record's stops (``computed``, ``method``, ``outside``) stops a way.  *route* starts with the binding's
    ``file:line``.

    The record's changes are on its ways, not ways of their own (out/gm/gm5-design.md G4): each is of the object the
    field holds.  *handed* are changes of that object made through a field of an object one level out (rule 2); one
    made at the statement of one of the record's own changes (its *at*, or a call of its *via*) is that change, which
    says what it changes and the record's choice places (rule 3); the others are this level's too.  *truth*: the read
    takes the value's truth alone (Q6's (a)): a container made empty is its empty start and the puts into it, the
    record's own among them."""
    ff = _field_froms().get((cls, field))
    if ff is None:
        return [Way(STOP, "", NO_FIELD, (f"{cls}.{field}",), changes=handed)]
    if (cls, field) in seen:
        return [Way(STOP, "", "cycle", (), changes=handed)]
    seen = seen | {(cls, field)}
    bindings, changes = list(ff.bindings), list(ff.changes)
    chosen = None
    if choose is not None and at is not None:
        chosen = choose(ff, at[0], at[1], _keys_off(steps, keys), of_self)
        bindings, changes = list(chosen.bindings), list(chosen.changes)
    changes = [c for c in changes if element_kept(c, steps)]
    made_here ={c.at for c in ff.changes} | {v for c in ff.changes for v in getattr(c, "via", None) or ()}
    level = (tuple(_changed(c, before=f"{at[0]}:{at[1]}" if at is not None else "") for c in changes)
             + tuple(h for h in handed if h.at not in made_here))
    ways: list[Way] = []
    for b in bindings:
        whole, parts = _ways_of_the_value(b.sources)
        if not whole and parts:
            ways += _with_changes([Way(STOP, "", PARTS_ONLY, (b.at,))], level)
        entered = chosen.entered_by(b.at) if chosen is not None else ()
        for s in whole:
            # a source read at the binding itself is placed there already
            ways += [w if w.route[:1] == (b.at,) else replace(w, route=(b.at, *w.route))
                     for w in _binding_ways(s, b, steps, seen, choose, at, keys, entered, level, truth)]
        ways += _with_changes([Way(STOP, "", st.kind, (b.at,)) for st in b.stops], level)
    # a record's stop at several sites is one, placed at all of them
    ways += _with_changes([Way(STOP, "", st.kind, (", ".join(st.at),) if st.at else ()) for st in ff.stops], level)
    # a record with no binding, change or stop, or a choice that took none: no way, not "never bound"
    return ways or [Way(STOP, "", NO_WAY, (f"{cls}.{field}",), changes=level)]


def _binding_ways(s: Any, b: Any, steps: tuple[str, ...], seen: frozenset, choose: Choose | None = None,
                  read_at: tuple[str, int] | None = None, keys: frozenset[str] | None = None,
                  entered: tuple[str, ...] = (), level: tuple[Changed, ...] = (), truth: bool = False) -> list[Way]:
    """The ways of one source *s* of the binding *b*, with *steps* still to take off its value.  A way that came back
    through calls is placed at the last of them (its line is the callee's, in a file the record does not name), and
    holds them and its line (G2).  A field of ``self`` the binding reads is read at the binding; a field of an object
    the binding holds, where the field holding it is read (*read_at*).  A parameter of the binding's own method is read
    up the calls the run's path entered it by (*entered*, S4b's (e)).

    On each way are the changes of the value: *level*, the field record's, and those the map says may be made to the
    source before it is bound, where the read reads what they change (S4b; without a choice, every one) -- G4.  Where
    the way reads a field ``.f`` of the value on in that field's own record, a change of what ``.f`` holds (any but a
    store or delete of ``.f``, which binds the field) is that object's: handed there, its key not said (rule 2).  A way
    a callee's test rules out on its call's constants is none (CRO (ii), :func:`_cut`)."""
    if _cut(s):
        return []
    route = tuple(getattr(s, "route", None) or ())
    at = route[-1] if route else f"{b.at.rpartition(':')[0]}:{s.line}" if getattr(s, "line", None) else b.at
    rest = (*tuple(getattr(s, "steps", None) or ()), *steps)
    # what a way that came back through calls holds of them (G2)
    back = _returned(s)
    on = (*(_changed(c, before=b.at) for c in getattr(s, "changes", None) or []
            if change_counts(c, _keys_off(rest, keys) if choose is not None else None) and element_kept(c, rest)), *level)
    truth = truth and not rest
    if truth and (put := _truth_ways(s, (at,), level)) is not None:
        # Q6's (a): a test reading the container's truth alone reads its empty start and its puts, each placed at the
        # put, through the binding (the record's own changes among them)
        return [replace(w, route=(*w.route[:-1], w.end_at)) if w.why == PUT else replace(w, **back) for w in put]
    # on the arm's run (*choose*), an element of a container made empty is read as before: its puts are changes on the
    # container's way, which S4c tells apart by the run (setupper 1098, chosen by elimination); Q1 reads the container
    # whole there, and both where no run is read
    elements = _elements(s, rest) if choose is None or not rest else None
    if elements is not None:
        # Q1: an element of a container made empty is a value one of its puts put; the container whole, its start too.
        # A put whose value is read is a way, not a change on one
        values, stays = elements
        left = tuple(c for c in on if c.record is None or not _put_read(c.record))
        ways = [w for v in values for w in _binding_ways(v, b, rest[1:], seen, choose, read_at, keys, entered, level)]
        return (_with_changes(_start(s, (at,), rest), left) + (ways if rest else [as_part(w) for w in ways])
                + ([Way(STOP, "", CONTAINER, (at,), changes=left, **back)] if stays else []))
    if (made := _returned_row(s, _keys_off(rest, keys))) is not None:
        # Q4: an object a callee filled from one row and returned
        return [Way(ROW, "", s.kind, (at,), changes=(*on, *made), **back)]
    if s.kind in (M.KEY_SELF_FIELD, M.KEY_OBJECT) and getattr(s, "site_owner", None):
        name = s.name or ""
        if s.kind == M.KEY_SELF_FIELD and name.startswith("self."):
            # another field of self: the same object, its changes these too
            return _with_changes(field_ways(s.site_owner, name.split(".")[1].split("[")[0], rest, seen, choose,
                                            _line_of(b.at), keys, of_self=True, truth=truth), on)
        if rest and rest[0].startswith("."):
            inner = [c for c in on if c.key == rest[0] and c.kind not in (M.CHANGE_STORE, M.CHANGE_DELETE)]
            handed = tuple(replace(c, key="*") for c in inner)
            return _with_changes(field_ways(s.site_owner, rest[0][1:], rest[1:], seen, choose, read_at, keys,
                                            handed=handed), tuple(c for c in on if c not in inner))
        return [Way(STOP, "", s.kind, (at,), changes=on, made_of=_made_of(s), **back)]
    if s.kind == M.KEY_PARAMETER:
        return [Way(STOP, s.name, PARAMETER, (at,), of=getattr(s, "site_owner", "") or b.owner,
                    entered=() if route else entered, changes=on, steps=rest, **back)]
    if s.kind == M.KEY_CONSTANT:
        return [Way(CONSTANT, "" if rest or getattr(s, "through", None) else s.name, THROUGH if rest else s.kind, (at,),
                    changes=on, **back)]
    if s.kind in (M.KEY_ROWS, M.KEY_SEQUENCE, M.KEY_DATABASE):
        return [Way(DATABASE, "", s.kind, (at,), changes=on, **back)]
    if s.kind == M.KEY_CALL and _clock_at(at, s.name or ""):
        return [Way(CLOCK, "", s.name, (at,), changes=on, **back)]
    return [Way(STOP, "", s.kind, (at,), changes=on, **back)]


@functools.cache
def _calls_in(fragment_id: int) -> tuple[dict, dict]:
    fragment = mapdump.fragment()
    params = {(p.owner, p.parameter): p for p in getattr(fragment, "parameter_froms", None) or []}
    entries = {(e.owner, e.line): e for e in getattr(fragment, "entered_froms", None) or []}
    return params, entries


def _callers() -> tuple[dict, dict]:
    fragment = mapdump.fragment()
    return _calls_in(id(fragment)) if fragment is not None else ({}, {})


@functools.cache
def _sends_in(fragment_id: int) -> tuple[dict[str, tuple[str, ...]], dict[tuple[str, str], list[tuple[str, tuple[int, ...]]]]]:
    fragment = mapdump.fragment()
    sends: dict[str, set[str]] = {}
    for p in getattr(fragment, "parameter_froms", None) or []:
        for s in p.sources:
            if getattr(s, "sent", False) and s.route:
                sends.setdefault(p.owner, set()).add(s.route[-1])
    calls: dict[tuple[str, str], list[tuple[str, tuple[int, ...]]]] = {}
    for n in getattr(fragment, "reach_graph", None) or []:
        for c in n.calls:
            rel, _, qual = c.to.split("#")[0].partition("::")
            calls.setdefault((rel, qual), []).append((n.owner.split("#")[0], tuple(sorted(set(c.lines or [c.line])))))
    return {o: tuple(sorted(v)) for o, v in sends.items()}, calls


def sends_into(owner: str) -> tuple[str, ...]:
    """The client's calls that send into the endpoint function *owner* (``file::name``): the last of the route of each
    source of its ``ParameterFrom`` records the map marks sent (``HeldSource.sent``, SND) -- job_api.kill's from
    Client.py:273, Client.kill_jobs's ``HttpClient.post``.  Empty where nothing is sent into it."""
    fragment = mapdump.fragment()
    return _sends_in(id(fragment))[0].get(owner, ()) if fragment is not None else ()


def def_line(owner: str) -> int:
    """The line of *owner*'s ``def`` as its ``ParameterFrom`` records hold it (``file::name``); 0 where it has none."""
    return next((p.line for (o, _), p in _callers()[0].items() if o == owner), 0)


def calls_into(rel: str, qual: str) -> list[tuple[str, tuple[int, ...]]]:
    """Each function whose calls go into the function of *rel* the map names *qual* (``kill_jobs``, ``Class.method``:
    the map's ``ReachNode.calls``), with the lines of those calls: into Client.kill_jobs, copyArchive's main at 19 lines
    and add_sub's main at 205."""
    fragment = mapdump.fragment()
    return list(_sends_in(id(fragment))[1].get((rel, qual), [])) if fragment is not None else []


def _by_the_path(sources: list, entered: tuple[str, ...]) -> tuple[list, list[str]]:
    """Of a ``ParameterFrom``'s *sources*, those that come in by one of the calls *entered* (``file:line``, the first
    of a source's route: S4b's (e)), and the call each of the others comes in by.  All, with none left out, where
    *entered* is empty or no source comes in by its calls (the map does not hold the path's call)."""
    if not entered:
        return list(sources), []
    by = [s for s in sources if not getattr(s, "route", None) or s.route[0] in entered]
    if not any(getattr(s, "route", None) for s in by):
        return list(sources), []
    return by, [s.route[0] for s in sources if getattr(s, "route", None) and s.route[0] not in entered]


def calls_off_the_path(owner: str, parameter: str, entered: tuple[str, ...]) -> list[str]:
    """The call each way of *owner*'s *parameter* comes in by that is not one of *entered* (:func:`param_ways`'s left
    out), once per way."""
    pf = _callers()[0].get((owner, parameter))
    return _by_the_path(pf.sources, entered)[1] if pf is not None else []


#: CRO (ii): each way a callee's test ruled out on its call's constants since rerun last said them (:func:`take_cuts`)
_CUTS: list[str] = []


def take_cuts() -> list[str]:
    """The ways :func:`_cut` ruled out since the last call, each once; forgotten once said."""
    got = list(dict.fromkeys(_CUTS))
    _CUTS.clear()
    return got


@functools.cache
def _guarded(fragment_id: int) -> dict[tuple[str, int], list]:
    """The map's GuardFroms by each line on the way to which their test is (``(file, line)``, from their *calls*)."""
    by: dict[tuple[str, int], list] = {}
    for (owner, _line), gs in _index(fragment_id)[0].items():
        for g in gs:
            for line in getattr(g, "calls", None) or ():
                by.setdefault((owner.partition("::")[0], line), []).append(g)
    return by


def _handed(owner: str, name: str, calls: tuple[str, ...], depth: int = 0) -> tuple[bool, Any]:
    """The one constant the call *calls[0]* hands *owner*'s parameter *name* -- a literal it hands, the parameter's
    default the map places by that call (its *route*, not its *at*), or what the next call (*calls[1:]*) hands the
    caller's own parameter it passes on -- as ``(True, value)``; ``(False, None)`` where it is not one constant."""
    pf = _callers()[0].get((owner, name))
    if pf is None or not calls or depth > 8:
        return False, None
    values: list[Any] = []
    for x in pf.sources:
        if list(getattr(x, "route", None) or ())[-1:] != [calls[0]]:
            continue
        if getattr(x, "paths", None) or getattr(x, "steps", None) or getattr(x, "through", None):
            return False, None
        if x.kind == M.KEY_CONSTANT:
            try:
                values.append(ast.literal_eval(x.name))
            except (ValueError, SyntaxError):
                return False, None
        elif x.kind == M.KEY_PARAMETER and getattr(x, "site_owner", ""):
            ok, value = _handed(x.site_owner, x.name, calls[1:], depth + 1)
            if not ok:
                return False, None
            values.append(value)
        else:
            return False, None
    if not values or any(repr(v) != repr(values[0]) for v in values):
        return False, None
    return True, values[0]


def _cut(s: Any) -> str:
    """CRO (ii) (the map's design, tmaeno "OK" 2026-11-11): where the way of the source *s* came back through calls (its
    *route*, the innermost first), the line it leaves each callee by -- its own line (*at*), then each call's -- and
    the tests on the way to it there (the map's GuardFroms whose *calls* hold that line; SBG), read with what the
    route's calls hand the parameters they read (:func:`_handed`): an arm condition of that line false there rules the
    way out, said (:func:`take_cuts`) and returned; '' where none is (a value not one constant keeps the way)."""
    route = tuple(getattr(s, "route", None) or ())
    at = next(iter(getattr(s, "at", None) or ()), "")
    fragment = mapdump.fragment()
    if not route or not at or fragment is None:
        return ""
    from bamboo.codemap import evaluate as EV
    from bamboo.codemap import pathcond

    guarded = _guarded(id(fragment))
    for k, place in enumerate((at, *route[:-1])):
        rel, _, ln = place.rpartition(":")
        if not ln.isdigit() or (rel, int(ln)) not in guarded:
            continue
        try:
            node = R.def_holding(rel, int(ln))
        except (OSError, SyntaxError):
            continue
        owner = f"{rel}::{node.name}" if node is not None else ""
        on = [g for g in guarded[(rel, int(ln))] if g.owner == owner]
        if not on or node is None:
            continue
        env = EV.Env()
        for g in on:
            whole, _parts = _ways_of_the_value(g.sources)
            got = [(True, ast.literal_eval(x.name)) if x.kind == M.KEY_CONSTANT and not getattr(x, "steps", None)
                   and not getattr(x, "through", None) and _literal(x.name)
                   else _handed(owner, x.name, route[k:]) if x.kind == M.KEY_PARAMETER else (False, None)
                   for x in whole]
            if got and all(ok for ok, _v in got) and all(repr(v) == repr(got[0][1]) for _ok, v in got):
                env.put(g.name, got[0][1], route[k])
        if not env.values:
            continue
        stmt = R.statement_at(node, int(ln))
        for guard in pathcond.arm_condition(stmt) if stmt is not None else []:
            text = re.sub(r"\s+\[at \d+\]$", "", guard)
            try:
                test = ast.parse(text, mode="eval").body
            except SyntaxError:
                continue
            if EV.truth(test, {}, env) is False:
                held = ", ".join(f"{n}={v.value!r}" for n, v in env.values.items())
                why = f"{text!r} false at {place} ({held} by {route[k]})"
                _CUTS.append(f"{at} by {' -> '.join(route)}: {why}")
                return why
    return ""


def identity_attrs(cls: str) -> list[str]:
    """The ``Class.attr`` the dump's tables name *cls*'s identity by; none where no table of it names one."""
    fragment = mapdump.fragment()
    return next((t.identity_attrs for t in (getattr(fragment, "tables", None) or []) if t.spec_class == cls and t.identity_attrs), [])


def _returned_row(s: Any, keys: frozenset[str] | None) -> tuple[Changed, ...] | None:
    """Q4 (out/gm/qk-design.md): where the source *s* is an object a callee built and returned (its *route*; the map's
    *at* the statement building it there, ``job = JobSpec()`` at job_standalone_module 518), whether the callee fills
    it from one fetched row for what the read takes off it -- each field in *keys*, or, read for which object it is
    (no key), the fields its class's table names its identity by (``JobSpec.PandaID``) -- at every return of the name
    it builds it in, by the map's ``fills.row_fills`` (as :func:`_row_fills` reads a fill in the arm's own function):
    then the changes the fills say are on the way after them (G4), else None.  None too where the read takes the
    value whole (*keys* None)."""
    route = getattr(s, "route", None)
    at = next(iter(getattr(s, "at", None) or ()), "")
    rel, _, ln = at.rpartition(":")
    if s.kind != M.KEY_OBJECT or not route or keys is None or not ln.isdigit():
        return None
    attrs = sorted(keys) or [a.partition(".")[2] for a in identity_attrs((getattr(s, "site_owner", "") or "").rpartition("::")[2])]
    if not attrs:
        return None
    from bamboo.codemap import pathcond

    try:
        types = asked._types()
        tree = types.tree(rel)
        if tree is None:
            return None
        holding = [f for f, _ in pathcond.functions_with_owner(tree) if f.lineno <= int(ln) <= (f.end_lineno or f.lineno)]
        func = min(holding, key=lambda f: (f.end_lineno or f.lineno) - f.lineno)
        stmt = R.statement_at(func, int(ln))
        calls = PR._calls_index(types)
    except (OSError, SyntaxError, ValueError, LookupError):
        return None
    if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name)):
        return None
    name = stmt.targets[0].id
    returns = [(n, n.value) for n in ast.walk(func) if isinstance(n, ast.Return) and isinstance(n.value, ast.Name)
               and n.value.id == name]
    if not returns:
        return None
    made: list[Changed] = []
    for attr in attrs:
        for ret, value in returns:
            # the field read where the callee hands the object back
            read = ast.Attribute(value=value, attr=attr, ctx=ast.Load())
            ast.copy_location(read, value)
            read.end_lineno, read.end_col_offset = value.end_lineno, value.end_col_offset
            read.parent = ret  # type: ignore[attr-defined]
            try:
                held = fills.row_fills(calls, rel, func, read, ())
            except ValueError:
                return None
            if held.others or not held.fills or not all(f.pick in (fills.PICK_ONE, fills.PICK_AT) and f.reads for f in held.fills):
                return None
            made += [Changed(f"{c.file}:{c.line}", c.kind, str(c.key), tuple(c.via)) for f in held.fills for c in f.changes]
    return tuple(dict.fromkeys(made))


def _literal(text: str) -> bool:
    try:
        ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return False
    return True


def param_ways(owner: str, parameter: str, seen: frozenset = frozenset(), entered: tuple[str, ...] = (),
               steps: tuple[str, ...] = (), truth: bool = False) -> list[Way]:
    """Each way *owner*'s *parameter* is handed in, read up the map's ``ParameterFrom`` records through the callers'
    own parameters to where it ends: a constant a call hands, a parameter no corpus call hands (an API's), the
    database, a clock a caller reads (``values.clock_of``), or a stop.  ``killJob``'s ``code`` comes in through ``killJobs`` (TaskBuffer 1190) from ``'60'`` at
    adder_gen 226, among others.  With *entered*, only the ways that come in by those calls (S4b's (e): the calls the
    run's path entered *owner* by).  *steps* are what the read still takes off the parameter's value (Q2's (c)): each
    source is read with them (:func:`_param_source_ways`).  *truth*: the read takes the value's truth alone (Q6's (a),
    :func:`truth_only`)."""
    params, _entries = _callers()
    pf = params.get((owner, parameter))
    if pf is None:
        return [Way(STOP, "", "no-parameter-from", ())]
    if (owner, parameter) in seen:
        return [Way(STOP, "", "cycle", ())]
    seen = seen | {(owner, parameter)}
    ways: list[Way] = []
    whole, parts = _ways_of_the_value(_by_the_path(pf.sources, entered)[0])
    for s in whole:
        got = _param_source_ways(s, steps, seen, truth)
        if getattr(s, "sent", False) and s.route:
            # SND: the value came across a send into *owner*, the client's call last on the source's route; a send
            # read further up (nearer the way's end) is kept
            got = [w if w.sent else replace(w, sent=s.route[-1], sent_into=owner) for w in got]
        ways += got
    kinds = sorted({st.kind for st in pf.stops})
    here = (f"{owner.partition('::')[0]}:{pf.line}",)
    if not whole and parts:
        ways.append(Way(STOP, "", ",".join([PARTS_ONLY, *kinds]), here))
    elif not whole:
        ways.append(Way(API, "", "no_caller", here) if kinds == ["no_caller"] else Way(STOP, "", ",".join(kinds) or "no-source", here))
    elif kinds:
        ways.append(Way(STOP, "", ",".join(kinds), here))
    return ways


def _param_source_ways(s: Any, steps: tuple[str, ...], seen: frozenset, truth: bool = False) -> list[Way]:
    """Where one source *s* of a ``ParameterFrom`` ends, with *steps* still to take off the parameter's value (Q2's
    (c)): a caller's own parameter, read up its callers in turn; an element of a container made empty, the values its
    puts put (Q1, :func:`_elements`); a field of an object the caller holds (``self.job_output_reports`` at add_main
    136), read on in the map's FieldFrom, and a parameter its binding takes up that method's callers (Q2's (a)); a
    constant -- its value the literal's only where no step is taken off it and no call outside the corpus made it --,
    the database, a clock the caller reads, or a stop.  Each way it ends at holds where (the map's *at*).  A way a
    callee's test rules out on its call's constants is none (CRO (ii), :func:`_cut`).  A value a container read whole
    holds is a part of the value read: a literal there is not its value (:func:`as_part`).  *truth*: the read takes the
    value's truth alone, so a container made empty is read by its empty start and its puts (Q6's (a),
    :func:`_truth_ways`)."""
    if _cut(s):
        return []
    route = tuple(s.route)
    rest = (*tuple(getattr(s, "steps", None) or ()), *steps)
    truth = truth and not rest
    here = next(iter(getattr(s, "at", None) or ()), "")
    if truth and (put := _truth_ways(s, route)) is not None:
        return put
    elements = _elements(s, rest)
    if elements is not None:
        values, stays = elements
        held = [w for v in values for w in _param_source_ways(v, rest[1:], seen)]
        return (_start(s, route, rest) + (held if rest else [as_part(w) for w in held])
                + ([Way(STOP, "", CONTAINER, route, end_at=here)] if stays else []))
    if s.kind == M.KEY_PARAMETER and s.site_owner:
        return [replace(w, route=w.route + route) for w in param_ways(s.site_owner, s.name, seen, steps=rest, truth=truth)]
    if (made := _returned_row(s, _keys_off(rest, None))) is not None:
        # Q4: an object a callee filled from one row and returned
        return [Way(ROW, "", s.kind, route, changes=made, end_at=here)]
    name = s.name or ""
    if s.kind == M.KEY_SELF_FIELD and getattr(s, "site_owner", "") and name.startswith("self."):
        out: list[Way] = []
        for w in field_ways(s.site_owner, name.split(".")[1].split("[")[0], rest, seen, truth=truth):
            if w.why == PARAMETER and w.of:
                out += [replace(x, route=x.route + w.route + route, changes=(*w.changes, *x.changes))
                        for x in param_ways(w.of, w.value, seen, steps=w.steps, truth=truth and not w.steps)]
            else:
                out.append(replace(w, route=w.route + route))
        return out
    if s.kind == M.KEY_CONSTANT and (getattr(s, "through", None) or rest):
        # a part of the literal, or what a call outside the corpus makes of it (an item of range(3)): it ends at the
        # constant, but its value is not the literal's
        return [Way(CONSTANT, "", THROUGH, route, end_at=here)]
    if s.kind == M.KEY_CONSTANT:
        # bound in the caller where the call handing it is its only call (else in the callee it came back from)
        return [Way(CONSTANT, s.name, s.kind, route, bound_at=(getattr(s, "line", None) or 0) if len(route) == 1 else 0,
                    end_at=here)]
    if s.kind in (M.KEY_ROWS, M.KEY_SEQUENCE, M.KEY_DATABASE):
        return [Way(DATABASE, "", s.kind, route, end_at=here)]
    if (s.kind == M.KEY_CALL and len(route) == 1 and getattr(s, "line", None)
            and _clock_at(f"{route[0].rpartition(':')[0]}:{s.line}", s.name or "")):
        # a call in the caller the map reads as a clock (S2's (b)); one in a callee the value came back from is in a
        # file the record does not name, and stops as before
        return [Way(CLOCK, "", s.name, route, end_at=here)]
    return [Way(STOP, "", s.kind, route, end_at=here)]


def same_value(literal: str, printed: str) -> bool:
    """Whether a constant as written (``'60'``, ``None``, ``2``) is what a run printed (``60``, ``None``, ``2``): a
    print shows a value's ``str``."""
    try:
        return str(ast.literal_eval(literal)) == printed
    except (ValueError, SyntaxError):
        return literal == printed


def stored(change: Any) -> tuple[str, ...] | None:
    """What a store the map records (``HeldChange``) stores, each literal as written, where every way of its value ends
    at a constant: a literal, or a field of an object of a corpus class its FieldFrom binds only to literals
    (``self.rc_fatal`` -> ``2``, S4c's V); None where any does not (a value made at run time, a parameter) or it is no
    store."""
    if getattr(change, "kind", "") != M.CHANGE_STORE or not getattr(change, "value", None) or getattr(change, "value_stops", None):
        return None
    out: list[str] = []
    for s in change.value:
        if getattr(s, "paths", None) or getattr(s, "steps", None) or getattr(s, "through", None) or getattr(s, "changes", None):
            return None
        name = s.name or ""
        if s.kind == M.KEY_CONSTANT:
            out.append(name)
        elif s.kind == M.KEY_SELF_FIELD and getattr(s, "site_owner", "") and re.fullmatch(r"self\.\w+", name):
            ways = field_ways(s.site_owner, name.split(".")[1])
            if not ways or any(w.end != CONSTANT or w.why == THROUGH or w.changes for w in ways):
                return None
            out += [w.value for w in ways]
        else:
            return None
    return tuple(dict.fromkeys(out)) or None


def entered(owner: str, line: int) -> Any:
    """The map's ``EnteredFrom`` of *owner*'s function whose ``def`` is at *line* (its corpus call sites, and the ways
    in it does not place), or None where the map holds none."""
    _params, entries = _callers()
    return entries.get((owner, line))


def _supplied_by_the_database(fn: R.Function, name: str, line: int) -> bool:
    """Whether the map's trace says every binding of *name* reaching *line* is a value the database supplies."""
    try:
        answer = asked.walk(fn, line).asker.ask(trace.Question(trace.ASK_BINDINGS, fn.rel, fn.owner, name, line))  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001 -- a trace that cannot answer leaves the term where it stops
        return False
    return bool(answer.steps) and all(s.terminal == M.STOP_UPSTREAM for s in answer.steps)


def part_start(cls: str, part: str) -> tuple[tuple[str, str], ...] | None:
    """Q7's (a): what an object of *cls* (``file::Class``) holds of *part* (one step, ``.status_code``) when its
    constructor made it -- each literal the class's ``__init__`` binds the field to, with the binding's ``file:line``,
    by the map's FieldFrom of the class (the class body's own binding where no ``__init__`` binds it); None where the
    map holds no FieldFrom of the field, or a binding there is no literal (a parameter, a call)."""
    if not cls or not part.startswith(".") or not part[1:].isidentifier():
        return None
    ff = _field_froms().get((cls, part[1:]))
    if ff is None:
        return None
    init = [b for b in ff.bindings if str(getattr(b, "owner", "")).endswith("::__init__")]
    out: list[tuple[str, str]] = []
    for b in init or [b for b in ff.bindings if getattr(b, "how", "") == "class"]:
        for s in b.sources:
            if s.kind != M.KEY_CONSTANT or any(getattr(s, k, None) for k in ("steps", "through", "paths", "changes")):
                return None
            out.append((s.name, b.at))
    return tuple(dict.fromkeys(out)) or None


def _row_fills(fn: R.Function, name: str, line: int, others: bool = False) -> frozenset[str] | None:
    """The statements (``file:line``) that fill what the test at *line* reads of *name* from one fetched row -- the
    field it takes off it (``job.jobStatus``, filled by ``job.pack(res[0])``), or the name itself where it reads it
    whole -- where every write of it is such a fill, with no in-place change on the way (the map's
    ``fills.row_fills``, as :func:`which_ran.row_state` reads it); None where any is not.  With *others*, the fills
    where other writes follow them too (Q7's (a): the row is the start, the other writes its changes)."""
    try:
        types = asked._types()
        tree = types.tree(fn.rel)
        if tree is None:
            return None
        from bamboo.codemap import pathcond

        func = next(f for f, _ in pathcond.functions_with_owner(tree) if f.lineno == fn.node.lineno and f.name == fn.name)
        node = R.statement_at(func, line)
        if not isinstance(node, (ast.If, ast.While)):
            return None
        parents = _parents(node.test)
        reads: list[ast.expr] = []
        for n in ast.walk(node.test):
            if isinstance(n, (ast.Name, ast.Attribute)) and ast.unparse(n) == name:
                # the field the test takes off it, not a method it calls on it (that reads the whole value)
                p = parents.get(id(n))
                taken = (isinstance(p, ast.Attribute) and p.value is n
                         and not (isinstance(c := parents.get(id(p)), ast.Call) and c.func is p))
                reads.append(p if taken and isinstance(p, ast.Attribute) else n)
        calls = PR._calls_index(types)
        owned = types.method_class(fn.rel, func)
        bound = tuple(calls.objects_of(owned)) if owned else ()
    except (StopIteration, LookupError, ValueError):
        return None
    sites: set[str] = set()
    for read in reads:
        held = WR._row_held(calls, fn.rel, func, read, bound)
        if held is None or (held.others and not others) or not held.fills or not all(
                fill.pick in (fills.PICK_ONE, fills.PICK_AT) and not fill.changes and fill.reads for fill in held.fills):
            return None
        sites |= {f"{fill.file or fn.rel}:{fill.site.lineno}" for fill in held.fills}
    return frozenset(sites) if reads else None


def terms(fn: R.Function, owner: str, guards: list[tuple[str, list[ast.If]]], choose: Choose | None = None) -> list[Term]:
    """Each term of each guard (its text and the ``if`` statements it is read at) of an arm of *fn* (*owner* as the
    map spells the arm's function), with where each way of it ends.  A field of ``self`` is read on in the map's
    FieldFrom, by *choose*'s bindings where it is given (S4b).  *choose* is given where the terms are read on the arm's
    run: there an element the test takes off a container made empty keeps its puts as changes on the container's way,
    for S4c to tell apart by the run; Q1 reads it where no run is read, and the container whole everywhere."""
    index, _runs = _maps()
    out: list[Term] = []
    for guard, ifs in guards:
        for one in ifs:
            ended = {id(c): e for c in ast.walk(one.test) if isinstance(c, ast.Call)
                     if (e := call_end(fn.rel, one.lineno, c)) is not None}
            calls_out = [e for e in ended.values() if e.end not in (DATABASE, OUTSIDE)]
            records = index.get((owner, one.lineno), [])
            if not records and not ended:
                out.append(Term(guard, ast.unparse(one.test), (End(ast.unparse(one.test), STOP, NO_RECORD, f"{fn.rel}:{one.lineno}"),),
                                one.lineno))
                continue
            for e in calls_out:
                out.append(Term(guard, e.term, (e,), one.lineno))
            for g in sorted(records, key=lambda g: g.name):
                if (by_call := _receiver_only(one.test, g.name, ended)) is not None:
                    out.append(Term(guard, g.name, (End(g.name, by_call.end, by_call.why, by_call.at),), one.lineno))
                    continue
                ends = []
                # a field of self read on in the map's FieldFrom (S4a): a term per field, a plain self's per field it reads
                fields: dict[str, list[End]] = {}
                read_on = False
                own = {r.name for r in records}
                whole, parts = _ways_of_the_value(g.sources)
                if not whole and parts:
                    ends.append(End(g.name, STOP, PARTS_ONLY, f"{fn.rel}:{one.lineno}"))
                for s in whole:
                    if _cut(s):
                        # a callee's test rules the way out on its call's constants (CRO (ii))
                        continue
                    if s.kind == M.KEY_SELF_FIELD and getattr(s, "site_owner", "") and not getattr(s, "changes", None) and _field_froms():
                        read = fields_read(one.test, s, own)
                        if read or g.name == "self":
                            read_on = True
                            for f in read:
                                name = f"self.{f}" if g.name == "self" else g.name
                                # the element the test takes off the field is a step the read takes
                                item = item_step(one.test, name, fn.rel, one.lineno)
                                truth = truth_only(one.test, name)
                                fields.setdefault(name, []).extend(
                                    _truth_param(_field_end(name, w), truth) for w in field_ways(
                                        s.site_owner, f, (*tuple(getattr(s, "steps", None) or ()), *((item,) if item else ())),
                                        choose=choose, at=(fn.rel, one.lineno),
                                        keys=read_keys(one.test, name, fn.rel, one.lineno), of_self=True, truth=truth))
                            continue
                    if (h := caught_name(fn, s)) is not None:
                        # K: a field of the exception a handler names is what the raise that made it handed -- the
                        # field's ways up to the raise sites, the run's traceback choosing among them (rerun)
                        ends += caught_ends(g.name, s, h, fn, one, choose)
                        continue
                    item = item_step(one.test, g.name, fn.rel, one.lineno)
                    taken = (item,) if item else ()
                    if not taken and truth_only(one.test, g.name) and (put := _truth_ways(s, ())) is not None:
                        # Q6's (a): a container made empty the test reads for its truth alone: its empty start and the
                        # statements putting into it, each placed at the put and the calls it was made through
                        first = _source_end(g.name, s, fn.rel, None, one.lineno)
                        ends += [End(g.name, w.end, w.why, w.end_at, route=w.route) if w.why == PUT
                                 else replace(first, end=w.end, why=w.why, changes=()) for w in put]
                        continue
                    if _elements(s, taken) is not None and (choose is None or not taken):
                        # a container made empty the test reads whole, or an element of it where no run is read (Q1)
                        ends += _guard_ends(g.name, s, fn.rel, read_keys(one.test, g.name, fn.rel, one.lineno), one.lineno,
                                            rest=taken)
                        continue
                    e = _source_end(g.name, s, fn.rel, read_keys(one.test, g.name, fn.rel, one.lineno), one.lineno,
                                    read=taken)
                    if e.end == STOP and s.kind == M.KEY_ATTRIBUTE and _supplied_by_the_database(fn, g.name, one.lineno):
                        e = replace(e, end=DATABASE, why=s.kind)
                    elif e.end == STOP and s.kind == M.KEY_OBJECT and (filled := _row_fills(fn, g.name, one.lineno)) is not None:
                        # the fill is the row's value, not a change after it: a change the map records at or through
                        # a fill's statement (killJob's job.pack(res[0]) at 998) is that fill (G4)
                        e = replace(e, end=ROW, why=s.kind,
                                    changes=tuple(c for c in e.changes if not ({c.at, *c.made_by} & filled)))
                    elif (e.end == STOP and s.kind == M.KEY_OBJECT and e.changes
                          and (filled := _row_fills(fn, g.name, one.lineno, others=True)) is not None):
                        # Q7's (a): other writes follow the fill (killJob's job.jobStatus = "closed" at 1028 ...): the
                        # row is what the test takes at its start, the others the changes after it, S4c's to read
                        e = replace(e, filled=tuple(sorted(filled)),
                                    changes=tuple(c for c in e.changes if not ({c.at, *c.made_by} & filled)))
                    elif e.end == STOP and (made := _returned_row(s, _keys_off(tuple(getattr(s, "steps", None) or ()),
                                                                              read_keys(one.test, g.name, fn.rel, one.lineno)))) is not None:
                        # Q4: an object a callee filled from one row and returned
                        e = replace(e, end=ROW, why=s.kind, changes=(*e.changes, *made))
                    ends.append(_truth_param(e, truth_only(one.test, g.name)))
                ends += [End(g.name, STOP, getattr(st, "kind", "stop") or "stop", "") for st in g.stops]
                # the term's own field (``self.job``): one term with its other ways
                ends += fields.pop(g.name, [])
                # a way-end whose every fact is another's is that one, said once (G2)
                if ends or not read_on:
                    out.append(Term(guard, g.name, tuple(dict.fromkeys(ends))
                                    or (End(g.name, STOP, NO_RECORD, f"{fn.rel}:{one.lineno}"),), one.lineno))
                out += [Term(guard, name, tuple(dict.fromkeys(fe)), one.lineno) for name, fe in fields.items()]
    return out


def _truth_param(e: End, truth: bool) -> End:
    """*e*, a parameter's way with no step left that a test reads for its truth alone (*truth*), marked so: it is read
    up its callers so (Q6's (a))."""
    return replace(e, truth=True) if truth and e.why == PARAMETER and e.param and not e.steps else e


def _field_end(term: str, w: Way) -> End:
    """A field's way (:func:`field_ways`) as a term's end: where it ends, the bindings it went through (*via*); a
    constant's *why* its literal, as a guard's own constant says it."""
    why = w.value if w.end == CONSTANT and w.value else w.why
    return End(term, w.end, why, w.route[-1] if w.route else "", param=w.value if w.why == PARAMETER else "", of=w.of,
               via=w.route[:-1], entered=w.entered, changes=w.changes, returned=w.returned,
               returned_line=w.returned_line, steps=w.steps, made_of=w.made_of)
