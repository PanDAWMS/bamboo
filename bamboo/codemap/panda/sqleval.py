"""Reading the SQL a function hands its cursor by running the function, on every path, over values.

:mod:`~bamboo.codemap.panda.sql` asks it what each call that runs a statement runs; this runs the Python.  A small
abstract interpreter over one function at a time:

- Every path is run.  An ``if`` whose test it cannot settle runs both arms, and two tests spelled the same, on names
  nothing rebinds in between, are taken the same way (facts).  A test that reads attributes (``self.backend ==
  'oracle'``) is one too, until something assigns the attribute or the object, or a call is handed the object (as its
  receiver or an argument): a call handed something else -- ``self.cur.execute(...)`` -- is taken not to change it.
- A loop over a sequence it knows is unrolled; over anything else, its body runs once or twice (:data:`UNKNOWN_EMPTY`).
  What such a loop appends to a name its body reads nowhere else is one part, however many passes run: a hole that
  keeps what one pass appends (:data:`REPEATED`, :attr:`HoleAt.repeats`), then what every pass ends with.
  ``getJobs`` orders by ``(CASE {} ELSE {} END)`` whichever shares the database holds, and ``ret[:-1]`` after
  ``JobSpec.columnNames``' loop cuts the comma every pass ends with, not the part.
- What it cannot know is a hole, :data:`HOLE` inside a text and ``{}`` when rendered, which carries the statement it
  entered the text at, the expression and why (:class:`HoleAt`).  A value with no text at all is ``UNK``.
- A name the function does not bind is unknown.  A parameter with a literal str default holds whatever a caller hands
  it, the default among it -- a choice of the default and a hole -- as a caller that leaves the argument out runs the
  default; a route's constants bind the parameters they name (``constants``).  Attributes, calls it does not model and
  subscripts it cannot settle are unknown.
- Only what flows into a runner's text, or into what decides whether a runner is reached, is tracked
  (:meth:`Ref._relevant`); a name nothing reads is dropped as soon as it is assigned.

What it keeps besides the texts, because the readers of the map ask for it:

- Each piece of a text with the statement that wrote it and, once a name holds it, the statement that put it into that
  name (:class:`Str`, :class:`Via`).  The second is the one whose path condition says why the piece is in a run --
  ``sql = sqlIn`` under ``if res is None`` in ``updateSiteData``, not the literal above the ``if`` -- and the values
  reaching the call decide it, so a name given the same text in two places names the one that reaches
  (``archiveJob``'s ``sqlFMod``: line 1700, not 1312).
- Paths that differ only in names that flow into a statement's text and into no test, loop or condition
  (:attr:`Ref.mergeable`) are one path whose values are :class:`Choice` s.  The names joined at one point share one
  :class:`Dim`, so a choice made for one is made for all (an ``INSERT``'s columns and its values).  An operation on a
  choice is done per branch and keeps the dimension (:func:`lift`); a test or a loop that meets a choice splits the
  path by it again (:meth:`Ref.concretize`).  ``getTasksWithCriteria_JEDI``'s tens of thousands of statements are one
  path's choices.
- A call's statements, two ways: every assignment of the dimensions (:func:`full_texts`) and a covering set in which
  every branch shows up at least once (:func:`cover`), which is what a consumer is handed; past :data:`COVER_CAP` it
  says so.
- A call the resolver places is run with its arguments' values, and what it returns is read (:data:`RETURN_HOPS`
  deep): ``getJobs``'s ``WHERE`` is ``construct_where_clause``'s return.
- What runs only on a path through an exception handler is read as any other path is (``retryJob``'s ``sql1``,
  rebound in the ``try`` and run again by the retry).
- A dict written at a key that is not known keeps "at least n entries" (:class:`Opn`); ``len`` of it is a lower bound
  (:class:`AtLeast`), so ``len(varMap) == 0`` reads False after the first write (``queryDatasetWithMap``).
- A bool a loop carries from one pass to the next is kept where a test deciding whether a call runs reads it
  (``propagateResultToJEDI``'s ``toUpdateFlag``).
- What the readers ask besides a call's text (:meth:`Reading.heads`, :meth:`Reading.named`): the text of the name a
  call runs, drawn as each covering statement was.

Limits, each said rather than hidden:

- Past :data:`MAX_ENVS` paths a run widens, coarser at each step until the paths fit: what the tests found; then the
  names no text reads whose values differ between paths (a name every path holds alike stays, as ``updateJob``'s
  route constant ``inJobsDefined`` must); then every name joined into one path (:func:`join_value`).  It never gives
  up on paths.  Joined dicts hold the union of their keys, the ones some paths lack uncertain: a loop over the dict
  counts them, so a statement built from the keys carries the union of its columns -- what a reader of columns needs,
  not every combination (``setScoutJobData_JEDI``'s ``SET`` from the dict ``getScoutJobData_JEDI`` returns).  A
  callee run widens at :data:`CALLEE_ENVS`; what a widened callee returns widens its caller; more values than
  :data:`MAX_VALUES` from one call are joined, not dropped.  A widened run's lines say so (``widened_at``); it is not
  run again call by call -- keeping the alike names gave the same statements on every line that would fix, at a third
  of the time.
- Every cap counts work (:data:`WORK_BUDGET`), never seconds, so a reading does not depend on the machine's load.
- A loop over a sequence whose length is not known runs at least once.  An assumption: PanDA builds column lists, IN
  lists and criteria from inputs its callers never pass empty (an empty one gives SQL no database takes, ``SELECT
  FROM``, ``IN ()``), and guards the ones that may be (``if skip_constituent_ids:``).  The rows a query returns may
  well be none: a statement only an empty loop leads to (a flag the loop sets, tested after it) is not read.

Approximations: values a loop carries between passes are forgotten (numbers always); tests that decide a guard are not
tracked; a dict is tracked by literal keys only where it is read by them; lists are values (no aliasing); nested
functions are not run unless called through the resolver.
"""

from __future__ import annotations

import ast
import itertools
import re
import string
from typing import Callable, NamedTuple, Optional

HOLE = "\x00"
MAX_ENVS = 512
CALLEE_ENVS = 64  # paths a callee run keeps before widening: it reads one value for the caller, on a smaller budget
MAX_VALUES = 64
MAX_UNROLL = 64
UNKNOWN_PASSES = 2
UNKNOWN_EMPTY = (
    False  # True: a for over a sequence of unknown length may also run no pass
)
MAX_FORK = 64  # paths one test may split a joined path into; past it the test is read undecided (widened)
MAX_LIFT = 4096  # branches one operation on a joined value may be applied to; past it the value is a hole
COVER_CAP = 64  # statements handed to a consumer per call, at most
FULL_CAP = 50_000  # statements in a full expansion, at most (comparison only)
RETURN_HOPS = 3
# Work, not seconds: one path through one statement is one step, one branch an operation
# on a joined value is applied to is one more.  A run past its cap says so (overflow);
# the run over every call goes per call past its cap. The same input gives the same
# answer on any machine at any load -- a cap in seconds read _read_unprocessed_inputs
# whole in 3.96 s once and cut it at 5.02 s another time, at the same 287,000 steps.
# Set from the S1 corpus (s1e calibration, about 100,000 steps a second): the largest
# run took 420,609 steps, the largest whole run 395,402, the largest callee run 140,666
# (getScoutJobData_JEDI; the others 812 at most).
WORK_BUDGET: Optional[int] = 2_000_000  # one call's run (was 20 s)
WHOLE_WORK: Optional[int] = (
    500_000  # the run over every call of a function; past it, each call on its own (was 5 s)
)
CALLEE_WORK: Optional[int] = 200_000  # one callee run (was 0.5 s)
RUNNERS = frozenset({"execute", "executemany", "querySQL", "querySQLS"})
_DICT_READS = frozenset(
    {"keys", "items", "values", "get", "copy"}
)  # methods that read a dict's structure


# ================================================================ values


class _Unknown:
    __slots__ = ()

    def __repr__(self) -> str:
        return "UNK"


UNK = _Unknown()
_MISSING = object()
_INTERNED: dict = {}  # Str, Choice and Cat by content: emptied by clear_memo() between functions
_NODIMS: frozenset = frozenset()


class Entry(NamedTuple):
    """Where a piece came from when no statement of the function wrote it: the value handed to parameter *name*."""

    name: str


class HoleAt(NamedTuple):
    """A piece whose text is not known: the statement it entered the text at, the expression, and why.

    For the part a loop of unknown length appends (:data:`REPEATED`), *stmt* is the loop, *source* the name it
    appends to, and *repeats* each text one pass appends, ``{}`` for a hole: ``WHEN gshare=:leave{} THEN {}``.  The
    hole holds what the passes appended less *end*, which each of them ends with and which stands after the hole
    where nothing cut it (``ret[:-1]`` cuts the comma of ``{},``).  Where the passes appended items to a list
    (*listed*), each text is an item, and *joiner* is what the text joins them with once it does (``" AND
    ".join(conditions)``, :func:`joined_by`)."""

    stmt: object
    source: str
    reason: str
    repeats: tuple = ()
    end: str = ""
    listed: bool = False
    joiner: Optional[str] = None


#: Why a hole stands where a loop of unknown length appended to a name: the passes'
#: part, as one.
REPEATED = "a part a loop of unknown length repeats"
#: The holes that stand, while such a loop runs, where its first pass appends and where
#: a later one does.
MARK = "where the first pass of a loop of unknown length appends"
MARK_LATER = "where a later pass of a loop of unknown length appends"
MAX_TEMPLATES = (
    64  # texts one pass may append, kept as a repeated part's templates; past it, none
)


class Via(NamedTuple):
    """A piece's origin once a text name holds it: where it was written (a statement, an :class:`Entry` or a
    :class:`HoleAt`), and the statement that put it into *name*.  :class:`sql.Span`'s ``node`` is the latter --
    ``sql = sqlIn`` under ``if res is None``, not the literal outside it -- and sqlwrite reads that statement's path
    condition and, when it copies a name, that name's own pieces.

    *prior* is the piece's origin before *at* put it into *name* -- the name it was copied from, or the callee's
    ``return`` and the call it came back from (:data:`RETURNED`, :data:`CALLED`) -- so the whole way from where it
    was written to the run is kept (:attr:`Rendered.pieces`)."""

    written: object
    name: str
    at: object
    prior: object = None


#: The name a callee's ``return`` puts its value into, and the one the call that ran the
#: callee does: the marks a piece carries out of a callee and into its caller
#: (:func:`marked`).
RETURNED = "<return>"
CALLED = "<call>"


def written_at(origin):
    """Where a piece was written, whichever name holds it."""
    return origin.written if type(origin) is Via else origin


class Str:
    """A text with no choice in it: the string (holes as HOLE) and its runs, ``(piece, origin)`` in order.

    The origin of a written piece is the statement that wrote it (or an :class:`Entry`); of a hole, a :class:`HoleAt`.
    Runs are not merged when texts are joined, so a ``+=`` stays a piece of its own (the join factors on them).

    Interned, like :class:`Choice` and :class:`Cat`: two equal values are one object, so equality is identity.
    Compared by structure, a joined text sharing its branches was walked as a tree on every lookup
    (_read_unprocessed_inputs: 29 million comparisons in 15 s).
    """

    __slots__ = ("s", "runs", "_r")

    def __new__(cls, s: str, runs: tuple):
        key = ("S", s, runs)
        held = _INTERNED.get(key)
        if held is None:
            held = object.__new__(cls)
            held.s = s
            held.runs = runs
            held._r = None
            _INTERNED[key] = held
        return held

    def rendered(self) -> str:
        r = self._r
        if r is None:
            r = self._r = self.s.replace(HOLE, "{}")
        return r

    def __repr__(self) -> str:
        return f"Str({self.s!r})"


EMPTY = Str("", ())


def lit(text: str, origin) -> Str:
    return Str(text, ((text, origin),)) if text else EMPTY


def hole_str(origin: HoleAt) -> Str:
    return Str(HOLE, ((HOLE, origin),))


class Dim:
    """One point where paths were joined: a choice of one of *n* branches, the same for every value joined there."""

    __slots__ = ("n", "id", "line")
    count = 0

    def __init__(self, n: int, line: int = 0):
        Dim.count += 1
        self.id = Dim.count
        self.n = n
        self.line = line

    def __repr__(self) -> str:
        return f"D{self.id}/{self.n}"


def _dim_id(dim: Dim) -> int:
    return dim.id


def dims(value) -> frozenset:
    t = type(value)
    if t is Choice or t is Cat or t is Lst or t is Tup or t is Dct:
        return value.dims
    return _NODIMS


def _dims_of(values) -> frozenset:
    found = _NODIMS
    for value in values:
        t = type(value)
        if t is Choice or t is Cat or t is Lst or t is Tup or t is Dct:
            d = value.dims
            if d:
                found = found | d if found else d
    return found


class Choice:
    """One of *branches*, the one :attr:`dim` takes.  Interned (see :class:`Str`)."""

    __slots__ = ("dim", "branches", "dims", "_text")

    def __new__(cls, dim: Dim, branches: tuple):
        key = ("C", dim, branches)
        held = _INTERNED.get(key)
        if held is None:
            held = object.__new__(cls)
            held.dim = dim
            held.branches = branches
            inner = _dims_of(branches)
            held.dims = inner | {dim} if inner else frozenset((dim,))
            held._text = None
            _INTERNED[key] = held
        return held

    def textlike(self) -> bool:
        t = self._text
        if t is None:
            t = self._text = all(is_text(branch) for branch in self.branches)
        return t

    def __repr__(self) -> str:
        return f"Choice({self.dim!r}, {list(self.branches)!r})"


class Cat:
    """A text with at least one choice in it: its parts in order, each a :class:`Str` or a text :class:`Choice`.
    Interned (see :class:`Str`)."""

    __slots__ = ("parts", "dims")

    def __new__(cls, parts: tuple):
        key = ("K", parts)
        held = _INTERNED.get(key)
        if held is None:
            held = object.__new__(cls)
            held.parts = parts
            held.dims = _dims_of(parts)
            _INTERNED[key] = held
        return held

    def __repr__(self) -> str:
        return f"Cat({list(self.parts)!r})"


class Lst:
    __slots__ = ("items", "dims", "_h")

    def __init__(self, items: tuple):
        self.items = items
        self.dims = _dims_of(items)
        self._h = None

    def __hash__(self) -> int:
        h = self._h
        if h is None:
            h = self._h = hash(("L", self.items))
        return h

    def __eq__(self, other) -> bool:
        return self is other or (type(other) is Lst and other.items == self.items)

    def __repr__(self) -> str:
        return f"Lst({list(self.items)!r})"


class Tup:
    __slots__ = ("items", "dims", "_h")

    def __init__(self, items: tuple):
        self.items = items
        self.dims = _dims_of(items)
        self._h = None

    def __hash__(self) -> int:
        h = self._h
        if h is None:
            h = self._h = hash(("T", self.items))
        return h

    def __eq__(self, other) -> bool:
        return self is other or (type(other) is Tup and other.items == self.items)

    def __repr__(self) -> str:
        return f"Tup({list(self.items)!r})"


class Dct:
    """A dict with known keys.  Keys are plain Python values (a text's string, never its origin).

    *maybe* holds the keys only some of the paths joined into this one had (:func:`join_value`): a loop over the
    dict counts them as there (the run is widened, so it may say a combination no path builds), a test ``k in d``
    on one is not decided, and ``len`` is the number of the others at least."""

    __slots__ = ("items", "maybe", "dims", "_h")

    def __init__(self, items: tuple, maybe: frozenset = _NODIMS):
        self.items = items
        self.maybe = maybe
        self.dims = _dims_of(v for _, v in items)
        self._h = None

    def __hash__(self) -> int:
        h = self._h
        if h is None:
            h = self._h = hash(("D", self.items, self.maybe))
        return h

    def __eq__(self, other) -> bool:
        return self is other or (
            type(other) is Dct
            and other.items == self.items
            and other.maybe == self.maybe
        )

    def again(self, items: tuple) -> "Dct":
        """A dict with *items* (the same keys, other values) and this one's uncertain keys."""
        return Dct(items, self.maybe)

    def lookup(self, key):
        for k, v in self.items:
            if k == key and type(k) is type(key):
                return v
        return _MISSING

    def keys(self) -> tuple:
        return tuple(k for k, _ in self.items)

    def put(self, key, value) -> "Dct":
        maybe = self.maybe - {key} if key in self.maybe else self.maybe
        for index, (k, _) in enumerate(self.items):
            if k == key and type(k) is type(key):
                return Dct(
                    self.items[:index] + ((k, value),) + self.items[index + 1 :], maybe
                )
        return Dct(self.items + ((key, value),), maybe)

    def __repr__(self) -> str:
        return (
            f"Dct({dict(self.items)!r}"
            + (f", maybe={sorted(map(str, self.maybe))}" if self.maybe else "")
            + ")"
        )


class Opn:
    """A dict written at a key that is not known: no key is known, but it holds at least *least* entries."""

    __slots__ = ("least",)

    def __init__(self, least: int):
        self.least = least

    def __hash__(self) -> int:
        return hash(("O", self.least))

    def __eq__(self, other) -> bool:
        return type(other) is Opn and other.least == self.least

    def __repr__(self) -> str:
        return f"Opn(>={self.least})"


class AtLeast:
    """An int known only to be at least *n*: ``len`` of an :class:`Opn`."""

    __slots__ = ("n",)

    def __init__(self, n: int):
        self.n = n

    def __hash__(self) -> int:
        return hash(("A", self.n))

    def __eq__(self, other) -> bool:
        return type(other) is AtLeast and other.n == self.n

    def __repr__(self) -> str:
        return f">={self.n}"


_SCALARS = (str, int, float, bool, type(None))


class Overflow(Exception):
    pass


class _LiftCap(Exception):
    pass


# ---------------------------------------------------------------- choices


def make_choice(dim: Dim, branches) -> object:
    branches = tuple(branches)
    first = branches[0]
    for branch in branches[1:]:
        if not (branch is first or branch == first):
            return Choice(dim, branches)
    return first


def make_cat(parts) -> object:
    """The text of *parts* in order -- :class:`Str`, plain ``str``, :class:`Cat` or a text :class:`Choice` -- normalized."""
    out: list = []
    for part in parts:
        t = type(part)
        if t is Cat:
            pieces = part.parts
        else:
            pieces = (part,)
        for piece in pieces:
            tp = type(piece)
            if tp is str:
                piece = lit(piece, None)
                tp = Str
            if tp is Str:
                if not piece.s:
                    continue
                if out and type(out[-1]) is Str:
                    out[-1] = str_concat(out[-1], piece)
                else:
                    out.append(piece)
            elif tp is Choice:
                out.append(piece)
            else:
                raise TypeError(f"not a text: {piece!r}")
    if not out:
        return EMPTY
    if len(out) == 1:
        return out[0]
    return Cat(tuple(out))


def is_text(value) -> bool:
    t = type(value)
    return t is Str or t is str or t is Cat or (t is Choice and value.textlike())


_CARRIED: dict = {}  # (text, name, statement) -> the text carried into the name there; emptied by clear_memo()
_HELD_BY: set = (
    set()
)  # (text, name): every piece already in the name -- carried anywhere, it is itself


def carried(value, name: str, at):
    """*value* put into the text name *name* by the statement *at*: each piece *name* did not already hold says so.
    A piece from *name*'s own value (``sql = sql[:-1]``, ``sql += x``) keeps the statement that first put it there,
    so a text whose every piece is in *name* is returned as it is, whatever *at*: a ``+=`` walks only what it adds."""
    t = type(value)
    if not (t is Str or t is Choice or t is Cat):
        return value
    if (value, name) in _HELD_BY:
        return value
    key = (value, name, at)
    held = _CARRIED.get(key)
    if held is None:
        if t is Str:
            runs = value.runs
            if all(
                type(origin) is Via and origin.name == name for _piece, origin in runs
            ):
                held = value
            else:
                held = Str(
                    value.s,
                    tuple(
                        (
                            piece,
                            origin
                            if type(origin) is Via and origin.name == name
                            else Via(
                                written_at(origin),
                                name,
                                at,
                                origin if type(origin) is Via else None,
                            ),
                        )
                        for piece, origin in runs
                    ),
                )
        elif t is Choice:
            held = make_choice(
                value.dim, [carried(branch, name, at) for branch in value.branches]
            )
        else:
            held = make_cat([carried(part, name, at) for part in value.parts])
        _CARRIED[key] = held
        _HELD_BY.add((held, name))
    return held


def marked(value, name: str, at):
    """:func:`carried`, into the texts a tuple, list or dict holds too: a callee's ``return sql, var_map``."""
    t = type(value)
    if t is Tup or t is Lst:
        items = tuple(marked(item, name, at) for item in value.items)
        return value if items == value.items else t(items)
    if t is Dct:
        items = tuple((k, marked(v, name, at)) for k, v in value.items)
        return value if items == value.items else value.again(items)
    return carried(value, name, at)


def subst(value, dim: Dim, index: int):
    """*value* with *dim* taking branch *index* everywhere in it."""
    t = type(value)
    if (
        not (t is Choice or t is Cat or t is Lst or t is Tup or t is Dct)
        or dim not in value.dims
    ):
        return value
    if t is Choice:
        if value.dim is dim:
            return subst(value.branches[index], dim, index)
        return make_choice(value.dim, (subst(b, dim, index) for b in value.branches))
    if t is Cat:
        return make_cat([subst(p, dim, index) for p in value.parts])
    if t is Lst:
        return Lst(tuple(subst(x, dim, index) for x in value.items))
    if t is Tup:
        return Tup(tuple(subst(x, dim, index) for x in value.items))
    return value.again(tuple((k, subst(x, dim, index)) for k, x in value.items))


def rename(value, mapping: dict):
    """*value* with each dimension in *mapping* replaced by its new one (a callee's choices, fresh for each call)."""
    t = type(value)
    if not (
        t is Choice or t is Cat or t is Lst or t is Tup or t is Dct
    ) or value.dims.isdisjoint(mapping):
        return value
    if t is Choice:
        return Choice(
            mapping.get(value.dim, value.dim),
            tuple(rename(b, mapping) for b in value.branches),
        )
    if t is Cat:
        return Cat(tuple(rename(p, mapping) for p in value.parts))
    if t is Lst:
        return Lst(tuple(rename(x, mapping) for x in value.items))
    if t is Tup:
        return Tup(tuple(rename(x, mapping) for x in value.items))
    return value.again(tuple((k, rename(x, mapping)) for k, x in value.items))


def distribute(f: Callable, value):
    """``f`` applied to each branch of the choice *value* is (recursively), the choices kept as they are.

    For an operation that handles what is inside a value by structure -- an append to a list whose items are joined
    texts -- where :func:`lift` would expand every choice inside it."""
    if type(value) is Choice:
        return make_choice(
            value.dim, [distribute(f, branch) for branch in value.branches]
        )
    return f(value)


def lift(
    f: Callable,
    args: list,
    over: Optional[list[int]] = None,
    budget: Optional[list] = None,
):
    """``f(*args)`` once per branch of each dimension in ``args[over]`` (all of *args* by default), as a choice on it.

    Every argument is given the same branch, so values joined at one point stay paired.  Arguments outside *over*
    keep their other choices, for an *f* that handles them by structure (a ``%`` template's arguments)."""
    if budget is None:
        budget = [MAX_LIFT]
    pool = args if over is None else [args[k] for k in over]
    for value in pool:
        found = dims(value)
        if found:
            dim = min(found, key=_dim_id)
            return make_choice(
                dim,
                [
                    lift(f, [subst(x, dim, i) for x in args], over, budget)
                    for i in range(dim.n)
                ],
            )
    budget[0] -= 1
    if budget[0] < 0:
        raise _LiftCap()
    return f(*args)


# ---------------------------------------------------------------- texts


def str_concat(a: Str, b: Str) -> Str:
    if not a.s:
        return b
    if not b.s:
        return a
    return Str(a.s + b.s, a.runs + b.runs)


def str_slice(a: Str, start: int, stop: int) -> Str:
    """``a[start:stop]`` for normalized bounds (0 <= start, stop <= len), keeping each piece's origin."""
    if stop <= start:
        return EMPTY
    if start == 0 and stop == len(a.s):
        return a
    out = []
    position = 0
    for piece, origin in a.runs:
        end = position + len(piece)
        lo = start if start > position else position
        hi = stop if stop < end else end
        if lo < hi:
            out.append((piece[lo - position : hi - position], origin))
        position = end
        if position >= stop:
            break
    return Str(a.s[start:stop], tuple(out))


def str_repeat(a: Str, n: int) -> Str:
    return Str(a.s * n, a.runs * n) if n > 0 else EMPTY


def str_map(a: Str, method: str) -> Str:
    """A length-keeping case change, piece by piece."""
    out = getattr(a.s, method)()
    if len(out) != len(a.s):
        first = a.runs[0][1] if a.runs else None
        return Str(out, ((out, first),)) if out else EMPTY
    return Str(
        out, tuple((getattr(piece, method)(), origin) for piece, origin in a.runs)
    )


def str_split(a: Str, sep: Optional[str]) -> list[Str]:
    s = a.s
    if sep is None:
        return [str_slice(a, m.start(), m.end()) for m in re.finditer(r"\S+", s)]
    out = []
    start = 0
    while True:
        found = s.find(sep, start)
        if found < 0:
            out.append(str_slice(a, start, len(s)))
            return out
        out.append(str_slice(a, start, found))
        start = found + len(sep)


def str_join(sep, items: list):
    parts: list = []
    for index, item in enumerate(items):
        if index:
            parts.append(sep)
        parts.append(item)
    return make_cat(parts)


def to_py(value):
    t = type(value)
    if t is Str:
        return value.s
    if t is Lst:
        return [to_py(item) for item in value.items]
    if t is Tup:
        return tuple(to_py(item) for item in value.items)
    if t is Dct:
        return {k: to_py(v) for k, v in value.items}
    return value


def plain(value):
    """*value* with no origins: what a name that never flows into a text holds (two paths whose values differ only in
    where a piece was written are one path, as in the oracle)."""
    t = type(value)
    if t is Str:
        return value.s
    if t is Lst:
        return Lst(tuple(plain(x) for x in value.items))
    if t is Tup:
        return Tup(tuple(plain(x) for x in value.items))
    if t is Dct:
        return value.again(tuple((k, plain(v)) for k, v in value.items))
    if t is Choice:
        return make_choice(value.dim, (plain(b) for b in value.branches))
    if t is Cat:
        return value  # a joined text in a name no text reads: keep it (it cannot be flattened without forking)
    return value


def from_py(value, origin=None):
    if isinstance(value, str):
        return lit(value, origin) if value else EMPTY
    if isinstance(value, list):
        return Lst(tuple(from_py(item, origin) for item in value))
    if isinstance(value, tuple):
        return Tup(tuple(from_py(item, origin) for item in value))
    if isinstance(value, dict):
        return Dct(
            tuple(
                (k, from_py(v, origin))
                for k, v in value.items()
                if isinstance(k, _SCALARS)
            )
        )
    if isinstance(value, (set, frozenset)):
        return UNK
    return value if isinstance(value, _SCALARS) else UNK


def known(value) -> bool:
    """Whether *value* is one fully known value: no UNK, no hole, no choice."""
    t = type(value)
    if value is UNK or t is Opn or t is AtLeast or t is Choice or t is Cat:
        return False
    if t is Str:
        return HOLE not in value.s
    if t is str:
        return HOLE not in value
    if t is Lst or t is Tup:
        return all(known(item) for item in value.items)
    if t is Dct:
        return not value.maybe and all(known(v) for _, v in value.items)
    return True


def key_of(value):
    """*value* as a dict key, or _MISSING when it is not one known key."""
    t = type(value)
    if t is Str:
        return value.s if HOLE not in value.s else _MISSING
    if t is str:
        return value if HOLE not in value else _MISSING
    if isinstance(value, (int, float, bool, type(None))):
        return value
    if t is Tup and known(value):
        return to_py(value)
    return _MISSING


def truth(value) -> Optional[bool]:
    t = type(value)
    if value is UNK:
        return None
    if t is Str or t is str:
        s = value.s if t is Str else value
        if HOLE not in s:
            return bool(s)
        return True if s.replace(HOLE, "") else None
    if t is Dct and value.maybe:
        return True if len(value.items) > len(value.maybe) else None
    if t is Lst or t is Tup or t is Dct:
        return bool(value.items)
    if t is Opn:
        return True if value.least >= 1 else None
    if t is AtLeast:
        return True if value.n >= 1 else None
    if t is Choice or t is Cat:
        try:
            got = lift(truth, [value])
        except _LiftCap:
            return None
        return got if got is True or got is False else None
    return bool(value)


def sequence(value) -> Optional[tuple]:
    t = type(value)
    if t is Lst or t is Tup:
        return value.items
    if t is Dct:
        return value.keys()
    return None


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def immutable(value) -> bool:
    t = type(value)
    if t is Str or t is Cat or isinstance(value, _SCALARS):
        return True
    if t is Choice:
        return all(immutable(b) for b in value.branches)
    return False


# ================================================================ static helpers

_PERCENT = re.compile(
    r"%(?:\((?P<key>[^)]*)\))?(?P<flags>[#0\- +]*)(?P<width>\*|\d+)?(?:\.(?P<prec>\*|\d+))?[hlL]?(?P<conv>[diouxXeEfFgGcrsa%])"
)
_FORMATTER = string.Formatter()
_MUTATORS = frozenset(
    {
        "append",
        "extend",
        "insert",
        "pop",
        "remove",
        "clear",
        "sort",
        "reverse",
        "update",
        "setdefault",
    }
)
_STR_METHODS = frozenset(
    {
        "format",
        "join",
        "replace",
        "strip",
        "lstrip",
        "rstrip",
        "upper",
        "lower",
        "removeprefix",
        "removesuffix",
        "split",
        "startswith",
        "endswith",
    }
)
_CONTAINER_METHODS = frozenset({"get", "keys", "values", "items", "copy"})
_BUILTINS = frozenset(
    {
        "str",
        "len",
        "range",
        "list",
        "tuple",
        "sorted",
        "enumerate",
        "zip",
        "int",
        "bool",
        "reversed",
    }
)


def _names(node: ast.AST) -> set[str]:
    found: set[str] = set()
    stack = [node]
    while stack:
        here = stack.pop()
        if isinstance(here, ast.Lambda):
            continue
        if isinstance(here, ast.Name):
            found.add(here.id)
        stack.extend(ast.iter_child_nodes(here))
    return found


def _own_nodes(func: ast.AST):
    """Every node of *func*'s own body: nested functions, classes and lambdas are not entered."""
    stack = list(func.body)
    while stack:
        node = stack.pop()
        yield node
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
        ):
            continue
        stack.extend(ast.iter_child_nodes(node))


_SQL_COMMENT = re.compile(r"/\*.*?\*/|--[^\n]*", re.S)


def _no_sql(value) -> bool:
    """Whether every text *value* can be is holes, comments and whitespace: a tracing tag, not part of a statement.
    A value with no text at all is one too -- what a tag the run cannot read renders to."""
    t = type(value)
    if t is Str:
        return not _SQL_COMMENT.sub("", value.s.replace(HOLE, "")).strip()
    if t is str:
        return not _SQL_COMMENT.sub("", value).strip()
    if t is Choice:
        return all(_no_sql(branch) for branch in value.branches)
    if t is Cat:
        return all(_no_sql(part) for part in value.parts)
    return not is_text(value)


def _expression_nodes(node: ast.AST):
    stack = [node]
    while stack:
        here = stack.pop()
        yield here
        if isinstance(here, ast.Lambda):
            continue
        stack.extend(ast.iter_child_nodes(here))


def runner_calls(func: ast.AST) -> list[ast.Call]:
    return [
        node
        for node in _own_nodes(func)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in RUNNERS
        and node.args
    ]


#: How a piece of statement text reads, and a log line does not: a verb in capitals anywhere; a statement that
#: starts with its verb, in any case (42 in the corpus are written in lower case); a fragment that starts with a
#: clause word in capitals (``"AND status=:s "``).  "failed to update the files" and "with taskName=" are not
#: statement text (574 and 55 literals of the corpus a looser rule took).
_STATEMENT_VERB = re.compile(r"\b(?:SELECT|INSERT|UPDATE|DELETE|MERGE)\b")
_STATEMENT_START = re.compile(
    r"^\s*\(?\s*(?:(?i:select|insert|update|delete|merge)|WITH)\b"
)
_STATEMENT_CLAUSE = re.compile(
    r"^\s*(?:AND|OR|WHERE|SET|FROM|VALUES|JOIN|UNION|MINUS|ORDER BY|GROUP BY)\b"
)


def _statement_literal(value: ast.AST) -> bool:
    """Whether *value* holds a literal piece of statement text (:data:`_STATEMENT_VERB`)."""
    return any(
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and bool(
            _STATEMENT_VERB.search(node.value)
            or _STATEMENT_START.search(node.value)
            or _STATEMENT_CLAUSE.search(node.value)
        )
        for node in ast.walk(value)
    )


#: Methods of a text whose result is the text again, filled in or trimmed: what ``sql = sqlR.format(table)`` makes
#: of a statement is a statement.
_TEXT_RESULT = frozenset(
    {"format", "replace", "join", "strip", "rstrip", "lstrip", "upper", "lower"}
)


def _text_reads(value: ast.AST) -> set[str]:
    """The names *value* makes text of: read as operands, items, fields or the receiver or arguments of a
    :data:`_TEXT_RESULT` method -- not as the argument of any other call, whose result is that call's."""
    found: set[str] = set()
    stack = [value]
    while stack:
        here = stack.pop()
        if isinstance(here, ast.Name) and isinstance(here.ctx, ast.Load):
            found.add(here.id)
        elif isinstance(here, ast.Call):
            if isinstance(here.func, ast.Attribute) and here.func.attr in _TEXT_RESULT:
                stack.append(here.func.value)
                stack.extend(here.args)
        elif not isinstance(here, ast.Lambda):
            stack.extend(ast.iter_child_nodes(here))
    return found


def _built_names(func: ast.AST) -> frozenset:
    """The local names holding statement text in *func*: one written a literal piece of it
    (:func:`_statement_literal`), one a runner runs, and one made of either -- ``sql = sqlX % table``,
    ``sql_exe = (sql_jt + comment) % table``, ``for sql, var_map in [(sql_defined, {}), ...]``."""
    found: set = {
        n.id
        for call in runner_calls(func)
        for n in ast.walk(call.args[0])
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }
    writes: list = []
    for node in _own_nodes(func):
        if (
            isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign))
            and node.value is not None
        ):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = node.value
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            targets, value = [node.target], node.iter
        else:
            continue
        names = {
            n.id
            for target in targets
            for n in ast.walk(target)
            if isinstance(n, ast.Name)
        }
        if not names:
            continue
        if not isinstance(node, (ast.For, ast.AsyncFor)) and _statement_literal(value):
            found |= names
        writes.append((names, _text_reads(value)))
    changed = True
    while changed:
        changed = False
        for names, reads in writes:
            if reads & found and not names <= found:
                found |= names
                changed = True
    return frozenset(found)


def _uses(func: ast.AST, names: frozenset) -> list[ast.Name]:
    """Where *func* reads one of *names* other than to go on building it: ``sql = sql + x`` and
    ``sql = sql.replace(...)`` build ``sql``; ``sql_exe = sql_jt % table`` and ``execute(sql)`` use it."""
    building: set = set()
    for node in _own_nodes(func):
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            written = {t.id for t in targets if isinstance(t, ast.Name)}
            building |= {
                id(n)
                for n in ast.walk(node.value)
                if isinstance(n, ast.Name)
                and isinstance(n.ctx, ast.Load)
                and n.id in written
            }
    return [
        node
        for node in _own_nodes(func)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id in names
        and id(node) not in building
    ]


_PURE: dict[int, tuple[ast.AST, bool]] = {}
_SPELLED: dict[int, tuple[ast.AST, str]] = {}
_SOURCE: dict[int, tuple[ast.AST, str]] = {}


def _pure(test: ast.expr) -> bool:
    held = _PURE.get(id(test))
    if held is not None and held[0] is test:
        return held[1]
    answer = all(
        isinstance(
            node,
            (
                ast.Name,
                ast.Constant,
                ast.Compare,
                ast.cmpop,
                ast.Load,
                ast.Tuple,
                ast.List,
                ast.BoolOp,
                ast.boolop,
                ast.UnaryOp,
                ast.Not,
            ),
        )
        for node in ast.walk(test)
    )
    _PURE[id(test)] = (test, answer)
    return answer


_FACTUAL: dict[int, tuple[ast.AST, bool]] = {}


def _factual(test: ast.expr) -> bool:
    """Whether a path can remember how it took *test*: :func:`_pure`, or pure but for attributes it reads.

    No call and no subscript: ``self.backend == 'oracle'`` is one, ``isinstance(v, list)`` is not.
    """
    held = _FACTUAL.get(id(test))
    if held is not None and held[0] is test:
        return held[1]
    answer = _pure(test) or all(
        isinstance(
            node,
            (
                ast.Name,
                ast.Attribute,
                ast.Constant,
                ast.Compare,
                ast.cmpop,
                ast.Load,
                ast.Tuple,
                ast.List,
                ast.BoolOp,
                ast.boolop,
                ast.UnaryOp,
                ast.Not,
            ),
        )
        for node in ast.walk(test)
    )
    _FACTUAL[id(test)] = (test, answer)
    return answer


def _touched_by(test: ast.expr) -> frozenset[str]:
    """What a fact on *test* hangs on: each attribute it reads and each object on the way, spelled.

    ``job.x.y == 1`` -> ``job.x.y``, ``job.x``, ``job``.  Empty for a test on names only.
    """
    found: set[str] = set()
    for node in ast.walk(test):
        if isinstance(node, ast.Attribute):
            found.add(_spelled(node))
            found.add(_spelled(node.value))
    return frozenset(found)


def _header(stmt: ast.stmt) -> list[ast.AST]:
    """The parts of *stmt* that run when it starts: a compound statement's test, iterable or context, else all of it."""
    if isinstance(stmt, (ast.If, ast.While)):
        return [stmt.test]
    if isinstance(stmt, (ast.For, ast.AsyncFor)):
        return [stmt.iter]
    if isinstance(stmt, (ast.With, ast.AsyncWith)):
        return [item.context_expr for item in stmt.items]
    if isinstance(stmt, ast.Match):
        return [stmt.subject]
    if isinstance(
        stmt,
        (
            ast.Try,
            getattr(ast, "TryStar", ast.Try),
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.ClassDef,
        ),
    ):
        return []
    return [stmt]


def _spelled(test: ast.expr) -> str:
    held = _SPELLED.get(id(test))
    if held is not None and held[0] is test:
        return held[1]
    text = ast.unparse(test)
    _SPELLED[id(test)] = (test, text)
    return text


def _source(node: ast.AST) -> str:
    held = _SOURCE.get(id(node))
    if held is not None and held[0] is node:
        return held[1]
    try:
        text = ast.unparse(node)
    except Exception:
        text = type(node).__name__
    _SOURCE[id(node)] = (node, text)
    return text


def _test_keys(test: ast.expr) -> list[ast.expr]:
    found = [test]
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        found += _test_keys(test.operand)
    elif isinstance(test, ast.BoolOp):
        for value in test.values:
            found += _test_keys(value)
    return found


def _modelled_call(call: ast.Call) -> bool:
    """Whether the evaluator computes *call*'s value from its arguments (a builtin, a str or container method,
    ``re.sub``); any other call's value is not known (M1) or is a callee's (M2, read where a test needs it)."""
    func = call.func
    if isinstance(func, ast.Name):
        return func.id in _BUILTINS
    if isinstance(func, ast.Attribute):
        if func.attr in _STR_METHODS or func.attr in _CONTAINER_METHODS:
            return True
        return (
            func.attr == "sub"
            and isinstance(func.value, ast.Name)
            and func.value.id == "re"
        )
    return False


def _pattern_names(pattern) -> list[str]:
    return [
        node.name
        for node in ast.walk(pattern)
        if getattr(node, "name", None) and isinstance(node.name, str)
    ]


def _product(choices: list[list]) -> list[tuple]:
    total = 1
    for choice in choices:
        total *= max(len(choice), 1)
        if total > MAX_VALUES * 4:
            raise Overflow(f"more than {MAX_VALUES * 4} combinations")
    return list(itertools.product(*choices))


def _distinct(values) -> list:
    kept: dict = {}
    for value in values:
        try:
            kept.setdefault((type(value), value), value)
        except TypeError:
            kept.setdefault((type(value), id(value)), UNK)
    return list(kept.values())


_STMTS: dict[int, tuple[ast.AST, dict]] = {}
_PARENTS: dict[int, tuple[ast.AST, dict]] = {}


def statement_map(func: ast.AST) -> dict[int, ast.AST]:
    """``id(expression node) -> the statement whose own expression it is``; a default or decorator maps to *func*."""
    held = _STMTS.get(id(func))
    if held is not None and held[0] is func:
        return held[1]
    table: dict[int, ast.AST] = {}

    def own(stmt: ast.AST, roots) -> None:
        stack = list(roots)
        while stack:
            node = stack.pop()
            table[id(node)] = stmt
            for child in ast.iter_child_nodes(node):
                if not isinstance(child, ast.stmt):
                    stack.append(child)

    todo = list(func.body)
    own(func, [n for n in ast.iter_child_nodes(func) if not isinstance(n, ast.stmt)])
    while todo:
        stmt = todo.pop()
        roots = []
        for child in ast.iter_child_nodes(stmt):
            if isinstance(child, ast.stmt):
                todo.append(child)
            elif isinstance(child, (ast.excepthandler, ast.match_case)):
                for inner in ast.iter_child_nodes(child):
                    if isinstance(inner, ast.stmt):
                        todo.append(inner)
                    else:
                        roots.append(inner)
            else:
                roots.append(child)
        own(stmt, roots)
    _STMTS[id(func)] = (func, table)
    return table


def parent_map(func: ast.AST) -> dict[int, ast.AST]:
    held = _PARENTS.get(id(func))
    if held is not None and held[0] is func:
        return held[1]
    table = {
        id(child): parent
        for parent in ast.walk(func)
        for child in ast.iter_child_nodes(parent)
    }
    _PARENTS[id(func)] = (func, table)
    return table


# ================================================================ environments


class Env:
    __slots__ = ("vars", "facts", "_key", "_skel")

    def __init__(self, vars: dict, facts: dict):
        self.vars = vars
        self.facts = facts
        self._key = None
        self._skel = None

    def key(self):
        if self._key is None:
            self._key = (frozenset(self.vars.items()), frozenset(self.facts.items()))
        return self._key

    def skeleton(self, mergeable: frozenset):
        """Everything but the joinable names: two paths with the same skeleton can be one."""
        if self._skel is None:
            self._skel = (
                frozenset((k, v) for k, v in self.vars.items() if k not in mergeable),
                frozenset(self.facts.items()),
            )
        return self._skel


def factor(dim: Dim, values: list):
    """``Choice(dim, values)``, with the pieces every value begins and ends with taken out of it."""
    if len(values) > 1 and all(type(v) is Str or type(v) is Cat for v in values):
        tokens = [_tokens(v) for v in values]
        shortest = min(len(t) for t in tokens)
        p = 0
        first = tokens[0]
        while p < shortest and all(t[p] == first[p] for t in tokens[1:]):
            p += 1
        q = 0
        while q < shortest - p and all(
            t[len(t) - 1 - q] == first[len(first) - 1 - q] for t in tokens[1:]
        ):
            q += 1
        if p or q:
            middles = [_from_tokens(t[p : len(t) - q]) for t in tokens]
            return make_cat(
                [
                    _from_tokens(first[:p]),
                    make_choice(dim, middles),
                    _from_tokens(first[len(first) - q :]),
                ]
            )
    return make_choice(dim, values)


_ABSENT = object()


def join_value(values: list, dim_for: Callable):
    """One value for *values* -- one per path, :data:`_ABSENT` where the path has no such name -- as widening joins
    every path into one: a value all of them hold stays; texts (or None) become a choice on a new dimension
    (``dim_for(n)``); dicts hold every key any of them holds, the ones some lack marked uncertain (:class:`Dct`), each
    key's values joined; tuples, and lists, of one length are joined item by item; anything else that differs is not
    known.  A path without the name holds it unknown (an unknown value is not kept in a path's names): joined with
    it, the name is unknown -- read over the other paths only, a row a fetch may return was lost to its ``else``
    default (getScoutJobData_JEDI's ``preRamUnit``, and with it the merge scout's ram columns)."""
    distinct: list = []
    for value in values:
        if value is _ABSENT:
            return UNK
        if not any(
            value is d or (type(value) is type(d) and value == d) for d in distinct
        ):
            distinct.append(value)
    if not distinct:
        return UNK
    if len(distinct) == 1:
        return distinct[0]
    if any(v is UNK for v in distinct):
        return UNK
    if all(v is None or is_text(v) for v in distinct):
        return factor(dim_for(len(distinct)), distinct)
    kinds = {type(v) for v in distinct}
    if kinds == {Dct}:
        order: dict = {}
        for d in distinct:
            for k, _ in d.items:
                order.setdefault(k, None)
        items, maybe = [], set()
        for k in order:
            held = []
            for d in distinct:
                found = d.lookup(k)
                if found is _MISSING or k in d.maybe:
                    maybe.add(k)
                if found is not _MISSING:
                    held.append(found)
            items.append((k, join_value(held, dim_for)))
        return Dct(tuple(items), frozenset(maybe))
    if kinds <= {Dct, Opn}:
        return Opn(
            min(
                v.least if type(v) is Opn else len(v.items) - len(v.maybe)
                for v in distinct
            )
        )
    if len(kinds) == 1 and kinds <= {Tup, Lst}:
        n = len(distinct[0].items)
        if all(len(v.items) == n for v in distinct):
            return distinct[0].__class__(
                tuple(
                    join_value([v.items[i] for v in distinct], dim_for)
                    for i in range(n)
                )
            )
    return UNK


def _alike(envs, keep: frozenset) -> set:
    """The names outside *keep* every one of *envs* holds, each with one value on all of them."""
    envs = list(envs)
    first = envs[0].vars
    out = set()
    for name, value in first.items():
        if name in keep:
            continue
        if all(
            name in env.vars and (env.vars[name] is value or env.vars[name] == value)
            for env in envs[1:]
        ):
            out.add(name)
    return out


def _tokens(value) -> list:
    if type(value) is Str:
        return list(value.runs)
    out: list = []
    for part in value.parts:
        if type(part) is Str:
            out.extend(part.runs)
        else:
            out.append(part)
    return out


def _from_tokens(tokens: list):
    parts: list = []
    runs: list = []
    for token in tokens:
        if type(token) is tuple:
            runs.append(token)
        else:
            if runs:
                parts.append(Str("".join(r[0] for r in runs), tuple(runs)))
                runs = []
            parts.append(token)
    if runs:
        parts.append(Str("".join(r[0] for r in runs), tuple(runs)))
    return make_cat(parts)


def _unify(a, b, mapping: dict, taken: dict) -> bool:
    """Whether *b* is *a* with its dimensions renamed, the renaming (b's -> a's) added to *mapping*."""
    if a is b:
        return True
    ta = type(a)
    if ta is not type(b):
        return False
    if ta is Choice:
        if a.dim.n != b.dim.n or len(a.branches) != len(b.branches):
            return False
        held = mapping.get(b.dim)
        if held is None:
            if a.dim in taken and taken[a.dim] is not b.dim:
                return False
            mapping[b.dim] = a.dim
            taken[a.dim] = b.dim
        elif held is not a.dim:
            return False
        return all(
            _unify(x, y, mapping, taken)
            for x, y in zip(a.branches, b.branches, strict=False)
        )
    if ta is Cat:
        return len(a.parts) == len(b.parts) and all(
            _unify(x, y, mapping, taken) for x, y in zip(a.parts, b.parts, strict=False)
        )
    if ta is Lst or ta is Tup:
        return len(a.items) == len(b.items) and all(
            _unify(x, y, mapping, taken) for x, y in zip(a.items, b.items, strict=False)
        )
    if ta is Dct:
        return (
            len(a.items) == len(b.items)
            and a.maybe == b.maybe
            and all(
                ka == kb and _unify(x, y, mapping, taken)
                for (ka, x), (kb, y) in zip(a.items, b.items, strict=False)
            )
        )
    return a == b


def _aligned(base: "Env", env: "Env", names) -> "Env":
    """*env* with its dimensions renamed to *base*'s where its values are base's but for the names of dimensions.

    A dimension only names a choice within one path, so renaming it in every value of a path changes nothing the
    path can be.  Without this, a join at the same statement in two passes of a loop made two names for one choice,
    and every later join nested them (_read_unprocessed_inputs)."""
    mapping: dict = {}
    taken: dict = {}
    for name in names:
        a, b = base.vars.get(name), env.vars.get(name)
        if a is None or b is None or a is b or not dims(b):
            continue
        trial, trial_taken = dict(mapping), dict(taken)
        if _unify(a, b, trial, trial_taken):
            mapping, taken = trial, trial_taken
    mapping = {old: new for old, new in mapping.items() if old is not new}
    if not mapping:
        return env
    present = _dims_of(env.vars.values())
    if any(new in present and new not in mapping for new in mapping.values()):
        return env  # a dimension of base's is already one of env's own: renaming would tie two choices
    return Env({k: rename(v, mapping) for k, v in env.vars.items()}, env.facts)


_RETURNS: dict = {}  # (callee, arguments) -> (callee, values or None, dims made in it)
_PLACED: dict = {}  # id(call) -> (call, the resolver's targets)
_DATABASE_READS = frozenset({"fetchall", "fetchone", "fetchmany"})
_NESTED: dict = {}  # id(func) -> (func, ids of the functions defined inside it)


def _nested_ids(func) -> frozenset:
    held = _NESTED.get(id(func))
    if held is None or held[0] is not func:
        held = (
            func,
            frozenset(
                id(n)
                for n in ast.walk(func)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                and n is not func
            ),
        )
        _NESTED[id(func)] = held
    return held[1]


_ANALYSES: dict = {}  # (id(func), mode, line) -> (func, the static reading of it a Ref needs)


def clear_memo() -> None:
    """Forget callee readings and interned values (between functions: a value read before stays valid, but a value
    made after is not the same object as an equal one made before)."""
    _RETURNS.clear()
    _PLACED.clear()
    _INTERNED.clear()
    _INTERNED[("S", "", ())] = EMPTY
    _CARRIED.clear()
    _HELD_BY.clear()
    _HOLDS.clear()
    _FACTUAL.clear()
    _JOINED.clear()


# ================================================================ the evaluator


class Ref:
    """Run *func* and record, per runner call line, the texts its first argument can hold (``mode="calls"``), or
    what it returns (``mode="return"``, a callee run by :meth:`follow`)."""

    def __init__(
        self,
        func,
        constants: Optional[dict] = None,
        line: Optional[int] = None,
        *,
        resolver: Optional[Callable] = None,
        mode: str = "calls",
        hops: int = 0,
        free: Optional[dict] = None,
        stack: frozenset = frozenset(),
        entry: bool = True,
        conventions: Optional[dict] = None,
        work_cap: Optional[int] = None,
    ):
        self.func = func
        # Declared readings the consumers share with the map's reader:
        # {"in_list_helper": name, "globals": {free name: value}}.  None reads the code
        # alone.
        self.conventions = conventions or {}
        self.constants = dict(constants or {})
        self.entry = entry
        self.values: dict[
            int, dict
        ] = {}  # line -> {value: None}, in the order the paths reached it
        # What a consumer asks besides the call's text.  line -> {call's value: {text
        # name in it: {its value}}} (Execution.head, drawn with the cover's own
        # choices).
        self.names_at: dict[int, dict] = {}
        self.no_text: dict[int, int] = {}
        self.reached: set[int] = set()
        self.overflow: Optional[str] = None
        self.defaults_used: set[str] = set()
        self.line = line
        self.work_cap = work_cap
        self.max_envs = CALLEE_ENVS if mode == "return" else MAX_ENVS
        self.resolver = resolver
        self.mode = mode
        self.hops = hops
        self.free = dict(free or {})
        self.stack = stack | {id(func)}
        self.returns: list = []
        self.built_names: frozenset = frozenset()
        self.used: dict[
            str, dict
        ] = {}  # mode "uses": name -> {value it held at a use or an exit}
        self._for_text = False  # evaluating what a text is built from: only there is a call followed (d)
        self._epoch = 0  # bumped at each pass of a loop: a call run again on one path gets fresh choices
        self._handed: dict = {}  # (call, arguments, epoch) -> the callee's values, its choices renamed for that call
        self.stats = {
            "merges": 0,
            "forks": 0,
            "fork_capped": 0,
            "lift_capped": 0,
            "lift_miss": 0,
            "callees": 0,
            "callee_failed": 0,
            "max_paths": 0,
            "joined_all": 0,
            "work": 0,
        }
        if mode == "return":
            returns = [n for n in _own_nodes(func) if isinstance(n, ast.Return)]
            self.runners: list = []
            self._targets = {id(n) for n in returns}
            self._last_target = max((n.lineno for n in returns), default=0)
            seeds = [n.value for n in returns if n.value is not None]
        elif mode == "uses":
            # What each name the function builds a statement into holds where the code uses it, and where the
            # function ends: the statements it assembles, whether a runner runs them or not
            # (sql.assembled_texts).  The names are seeds too, so one nothing uses is read to the end.
            self.runners = []
            self.built_names = _built_names(func)
            uses = _uses(func, self.built_names)
            self._targets = {id(n) for n in uses}
            self._last_target = func.end_lineno or 0
            seeds = list(uses) + [
                ast.Name(id=name, ctx=ast.Load()) for name in sorted(self.built_names)
            ]
        else:
            self.runners = [
                call
                for call in runner_calls(func)
                if line is None or call.lineno == line
            ]
            self._targets = {id(call) for call in self.runners}
            self._last_target = max((call.lineno for call in self.runners), default=0)
            seeds = [call.args[0] for call in self.runners]
        self._loop_names: dict = {}
        self._contains: dict[int, bool] = {}
        self._loads: dict = {}
        self._loops_with_target = 0
        self._catching: list[tuple[dict, frozenset]] = []
        self._parent = parent_map(func)
        self._stmt = statement_map(func)
        self._augmented: dict[int, ast.BinOp] = {}
        self._stated: dict[int, tuple] = {}
        self.key_only: dict[str, set] = {}
        self.text_names: set[str] = set()
        self._text_keyed: set = set()
        analysis_key = (id(func), mode, line)
        held = _ANALYSES.get(analysis_key)
        if held is not None and held[0] is func:
            (
                self.text_names,
                self._text_keyed,
                self._whole,
                self.relevant,
                self.key_only,
                self.decisions,
                self.mergeable,
                self.call_guards,
                self.repeated,
                self.fact_names,
                self.fact_attrs,
            ) = held[1]
        else:
            whole, keyed = self._relevant(seeds, guards=False)
            self.text_names, self._text_keyed = whole | {n for n, _ in keyed}, keyed
            whole, keyed = self._relevant(seeds, guards=True)
            self._whole = whole
            self.relevant = whole | {n for n, _ in keyed}
            # A dict read only by literal keys is tracked for those keys only
            # (getScoutJobData_JEDI).
            self.key_only = {
                name: {key for n, key in keyed if n == name}
                for name, _ in keyed
                if name not in self._whole
            }
            decided, decided_keyed = self._relevant(
                self._decision_seeds(), guards=False, value_flow=True
            )
            self.decisions = decided | {n for n, _ in decided_keyed}
            # Joined into one path where nothing else differs: a name only a text reads
            # (b).
            self.mergeable = frozenset(self.text_names - self.decisions)
            self.call_guards = self._call_guards()
            self.repeated, self.fact_names, self.fact_attrs = self._repeated_atoms()
            _ANALYSES[analysis_key] = (
                func,
                (
                    self.text_names,
                    self._text_keyed,
                    self._whole,
                    self.relevant,
                    self.key_only,
                    self.decisions,
                    self.mergeable,
                    self.call_guards,
                    self.repeated,
                    self.fact_names,
                    self.fact_attrs,
                ),
            )
        self.widened = False
        self.widened_lines: set[int] = (
            set()
        )  # call lines read after the run had widened (what was read before is exact)
        args = func.args
        self.params = {a.arg for a in args.posonlyargs + args.args + args.kwonlyargs}
        if args.vararg:
            self.params.add(args.vararg.arg)
        if args.kwarg:
            self.params.add(args.kwarg.arg)

    # ---------------------------------------------------------- set-up

    def _key_read(self, name: ast.Name) -> Optional[str]:
        parent = self._parent.get(id(name))
        if (
            isinstance(parent, ast.Subscript)
            and parent.value is name
            and isinstance(parent.ctx, ast.Load)
            and isinstance(parent.slice, ast.Constant)
            and isinstance(parent.slice.value, str)
        ):
            return parent.slice.value
        if (
            isinstance(parent, ast.Compare)
            and len(parent.ops) == 1
            and isinstance(parent.ops[0], (ast.In, ast.NotIn))
            and parent.comparators[0] is name
            and isinstance(parent.left, ast.Constant)
            and isinstance(parent.left.value, str)
        ):
            return parent.left.value
        if (
            isinstance(parent, ast.Attribute)
            and parent.attr == "get"
            and parent.value is name
        ):
            call = self._parent.get(id(parent))
            if (
                isinstance(call, ast.Call)
                and call.func is parent
                and call.args
                and isinstance(call.args[0], ast.Constant)
                and isinstance(call.args[0].value, str)
            ):
                return call.args[0].value
        return None

    def _reads(self, node: Optional[ast.AST]) -> tuple[set[str], set[tuple[str, str]]]:
        whole: set[str] = set()
        keyed: set[tuple[str, str]] = set()
        if node is None:
            return whole, keyed
        stack = [node]
        while stack:
            here = stack.pop()
            if isinstance(here, ast.Lambda):
                continue
            if isinstance(here, ast.Name):
                key = self._key_read(here)
                if key is None:
                    whole.add(here.id)
                else:
                    keyed.add((here.id, key))
            stack.extend(ast.iter_child_nodes(here))
        return whole, keyed

    def _value_reads(
        self, node: Optional[ast.AST]
    ) -> tuple[set[str], set[tuple[str, str]]]:
        """:meth:`_reads`, but not into a call this does not model: its value is not known whatever its arguments are,
        so they decide nothing through it (do_preassign: ``res = ...querySQL(dry_sql_query, ...)`` tested made every
        text name a deciding one, and no path was ever joined)."""
        whole: set[str] = set()
        keyed: set[tuple[str, str]] = set()
        if node is None:
            return whole, keyed
        stack = [node]
        while stack:
            here = stack.pop()
            if isinstance(here, ast.Lambda) or (
                isinstance(here, ast.Call) and not _modelled_call(here)
            ):
                continue
            if isinstance(here, ast.Name):
                key = self._key_read(here)
                if key is None:
                    whole.add(here.id)
                else:
                    keyed.add((here.id, key))
            stack.extend(ast.iter_child_nodes(here))
        return whole, keyed

    @staticmethod
    def _writes(target: ast.expr) -> list[tuple[str, Optional[str]]]:
        if isinstance(target, ast.Name):
            return [(target.id, None)]
        if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
            if isinstance(target.slice, ast.Constant) and isinstance(
                target.slice.value, str
            ):
                return [(target.value.id, target.slice.value)]
            return [(target.value.id, None)]
        if isinstance(target, (ast.Tuple, ast.List)):
            return [one for element in target.elts for one in Ref._writes(element)]
        if isinstance(target, ast.Starred):
            return Ref._writes(target.value)
        return []

    def _index_reads(self, target: ast.expr) -> tuple[set[str], set[tuple[str, str]]]:
        whole: set[str] = set()
        keyed: set[tuple[str, str]] = set()
        if isinstance(target, ast.Subscript):
            whole, keyed = self._reads(target.slice)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                w, k = self._index_reads(element)
                whole |= w
                keyed |= k
        elif isinstance(target, ast.Starred):
            return self._index_reads(target.value)
        return whole, keyed

    def _relevant(
        self, seeds: list, guards: bool, value_flow: bool = False
    ) -> tuple[set[str], set[tuple[str, str]]]:
        """The slice from *seeds* (a runner's text, or a return's value): what flows into them and, with
        *guards*, what decides whether one is reached or a text name is bound.  With *value_flow*, only what can
        change a value (:meth:`_value_reads`)."""
        whole: set[str] = set()
        keyed: set[tuple[str, str]] = set()
        reads_of = self._value_reads if value_flow else self._reads

        def add(reads) -> bool:
            names, pairs = reads
            new = (names - whole) or (pairs - keyed)
            whole.update(names)
            keyed.update(pairs)
            return bool(new)

        def hits(writes, names, pairs) -> bool:
            keyed_names = {n for n, _ in pairs}
            for name, key in writes:
                if (
                    name in names
                    or (key is None and name in keyed_names)
                    or (key is not None and (name, key) in pairs)
                ):
                    return True
            return False

        for seed in seeds:
            add(reads_of(seed))
        nodes = list(_own_nodes(self.func))
        bodies = []
        if guards:
            for node in nodes:
                if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                    inside = [
                        n
                        for part in (node.body, node.orelse)
                        for stmt in part
                        for n in _expression_nodes(stmt)
                    ]
                    runs = any(id(n) in self._targets for n in inside)
                    writes: list = []
                    for n in inside:
                        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                            writes.append((n.id, None))
                        elif isinstance(n, ast.Subscript) and isinstance(
                            n.ctx, ast.Store
                        ):
                            writes += self._writes(n)
                        elif (
                            isinstance(n, ast.Call)
                            and isinstance(n.func, ast.Attribute)
                            and n.func.attr in _MUTATORS
                            and isinstance(n.func.value, ast.Name)
                        ):
                            writes.append((n.func.value.id, None))
                    test = (
                        node.test
                        if isinstance(node, (ast.If, ast.While))
                        else node.iter
                    )
                    bodies.append((writes, runs, self._reads(test)))
        changed = True
        while changed:
            changed = False
            for node in nodes:
                reads = None
                if isinstance(node, ast.Assign):
                    if any(
                        hits(self._writes(target), whole, keyed)
                        for target in node.targets
                    ):
                        reads = reads_of(node.value)
                        for target in node.targets:
                            changed |= add(self._index_reads(target))
                elif (
                    isinstance(node, (ast.AugAssign, ast.AnnAssign))
                    and node.value is not None
                ):
                    if hits(self._writes(node.target), whole, keyed):
                        reads = reads_of(node.value)
                        changed |= add(self._index_reads(node.target))
                elif isinstance(node, (ast.For, ast.AsyncFor)):
                    if hits(self._writes(node.target), whole, keyed):
                        reads = reads_of(node.iter)
                elif (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in _MUTATORS
                    and isinstance(node.func.value, ast.Name)
                    and (
                        node.func.value.id in whole
                        or any(n == node.func.value.id for n, _ in keyed)
                    )
                ):
                    reads = (
                        set().union(
                            *(_names(arg) for arg in node.args),
                            *(_names(k.value) for k in node.keywords),
                        ),
                        set(),
                    )
                if reads is not None and add(reads):
                    changed = True
            for writes, runs, test in bodies:
                if (runs or hits(writes, self.text_names, self._text_keyed)) and add(
                    test
                ):
                    changed = True
        return whole, keyed

    def _decision_seeds(self) -> list:
        """Every expression whose value steers the run rather than flowing into a value: tests, a loop's sequence,
        the operands of ``and``/``or``."""
        seeds = []
        for node in _own_nodes(self.func):
            if isinstance(node, (ast.If, ast.While, ast.IfExp, ast.Assert)):
                seeds.append(node.test)
            elif isinstance(node, (ast.For, ast.AsyncFor)):
                seeds.append(node.iter)
            elif isinstance(node, ast.comprehension):
                seeds.append(node.iter)
                seeds.extend(node.ifs)
            elif isinstance(node, ast.BoolOp):
                seeds.extend(node.values)
            elif isinstance(node, ast.Match):
                seeds.append(node.subject)
        return seeds

    def _call_guards(self) -> frozenset:
        """Names in the test of an ``if`` or ``while`` that holds a call this reads (a flag a loop carries is kept for them)."""
        found: set[str] = set()
        for node in _own_nodes(self.func):
            if isinstance(node, (ast.If, ast.While)):
                inside = (
                    n
                    for part in (node.body, node.orelse)
                    for stmt in part
                    for n in _expression_nodes(stmt)
                )
                if any(id(n) in self._targets for n in inside):
                    found |= _names(node.test)
        return frozenset(found & self.relevant)

    def _repeated_atoms(self):
        """Tests spelled twice or more, the names each reads (a rebinding forgets it), and for one that reads
        attributes what else forgets it (:meth:`kills`)."""
        counts: dict[str, int] = {}
        names: dict[str, set[str]] = {}
        touched: dict[str, frozenset[str]] = {}
        for node in _own_nodes(self.func):
            if isinstance(node, (ast.If, ast.While, ast.IfExp)):
                for part in _test_keys(node.test):
                    if _factual(part):
                        key = _spelled(part)
                        counts[key] = counts.get(key, 0) + 1
                        names[key] = _names(part)
                        touched[key] = _touched_by(part)
        repeated = {
            key
            for key, count in counts.items()
            if count >= 2 and names[key] <= self.relevant
        }
        return (
            repeated,
            {key: names[key] for key in repeated},
            {key: touched[key] for key in repeated if touched[key]},
        )

    def initial(self) -> Env:
        vars: dict = {}
        args = self.func.args
        positional = args.posonlyargs + args.args
        defaults = dict(
            zip(
                (a.arg for a in positional[len(positional) - len(args.defaults) :]),
                args.defaults,
                strict=False,
            )
        )
        defaults.update(
            {
                a.arg: d
                for a, d in zip(args.kwonlyargs, args.kw_defaults, strict=False)
                if d is not None
            }
        )
        for name, default in defaults.items():
            if name in self.constants:
                continue
            if self.mode == "return":
                # A callee's parameter the call does not pass holds its default: that is
                # what Python does.
                try:
                    vars[name] = from_py(ast.literal_eval(default), self.func)
                except (ValueError, SyntaxError, TypeError):
                    pass
            elif isinstance(default, ast.Constant) and isinstance(default.value, str):
                # Read as the default alone, what a caller passes is not read:
                # get_shares's ``parents=""`` dropped the WHERE every corpus caller runs
                # (None or a share's name).  The other branch is unknown, not a hole: a
                # hole is a str, and ``parents is None`` would read False on it.
                vars[name] = make_choice(
                    Dim(2, self.func.lineno), (lit(default.value, self.func), UNK)
                )
                self.defaults_used.add(name)
        for name, value in self.conventions.get("globals", {}).items():
            if name not in self.relevant_assigned and name not in self.params:
                vars[name] = value
        vars.update(self.free)
        for name, value in self.constants.items():
            vars[name] = from_py(value, Entry(name)) if self.entry else value
        out = {}
        for name, value in vars.items():
            if value is UNK:
                continue
            out[name] = value if name in self.text_names else plain(value)
        return Env(out, {})

    # ---------------------------------------------------------- driving

    def run(self) -> "Ref":
        try:
            envs, _, _ = self.block(self.func.body, [self.initial()])
            if self.mode == "return" and envs:
                self.returns.append(None)  # the end of the body: ``return None``
            if self.mode == "uses":
                self.hold_exit(envs)
        except Overflow as error:
            self.overflow = str(error)
        return self

    def dedupe(self, envs: list[Env]) -> list[Env]:
        if self.work_cap is not None and self.stats["work"] > self.work_cap:
            raise Overflow(f"more than {self.work_cap} steps")
        kept: dict = {}
        for env in envs:
            kept.setdefault(env.key(), env)
        if len(kept) > 1 and self.mergeable:
            kept = self.join_paths(kept)
        if len(kept) > self.stats["max_paths"]:
            self.stats["max_paths"] = len(kept)
        if len(kept) > self.max_envs:
            self.widened = True
            # Widening, coarser at each step until the paths fit: (1) what the tests
            # found; (2) the names no text reads, where their value differs between
            # paths -- one every path holds alike stays (updateJob's ``inJobsDefined``,
            # a route's constant, separated no path and was what decided which table);
            # (3) every name, each joined to one value (:func:`join_value`): one path.
            for step in (1, 2, 3):
                if step == 3:
                    self.stats["joined_all"] += 1
                    kept = {None: self.joined_all(list(kept.values()))}
                    break
                if step == 2:
                    alike = _alike(kept.values(), self.text_names)
                merged: dict = {}
                for env in kept.values():
                    vars = (
                        env.vars
                        if step == 1
                        else {
                            k: v
                            for k, v in env.vars.items()
                            if k in self.text_names or k in alike
                        }
                    )
                    one = Env(vars, {})
                    merged.setdefault(one.key(), one)
                kept = merged
                if len(kept) > 1 and self.mergeable:
                    kept = self.join_paths(kept)
                if len(kept) <= self.max_envs:
                    break
        return list(kept.values())

    def joined_all(self, envs: list[Env]) -> Env:
        """*envs* as one path: each name joined over them (:func:`join_value`), a new dimension for each that differs."""
        names: dict = {}
        for env in envs:
            for name in env.vars:
                names[name] = None
        line = getattr(self, "_at_line", 0)
        vars = {}
        for name in names:
            value = join_value(
                [env.vars.get(name, _ABSENT) for env in envs], lambda n: Dim(n, line)
            )
            if value is not UNK:
                vars[name] = value
        return Env(vars, {})

    def join_paths(self, kept: dict) -> dict:
        """Paths that differ only in :attr:`mergeable` names, as one path each (b)."""
        groups: dict = {}
        for env in kept.values():
            groups.setdefault(env.skeleton(self.mergeable), []).append(env)
        if len(groups) == len(kept):
            return kept
        out: dict = {}
        made: dict = {}
        for group in groups.values():
            one = group[0] if len(group) == 1 else self.merged(group, made)
            out.setdefault(one.key(), one)
        return out

    def merged(self, group: list[Env], made: dict) -> Env:
        """One path for *group*, each name that differs a choice on one new dimension.

        Groups joined at the same time are different paths (they differ in a name a test reads), and where two of
        them join the same values they share the dimension: the paths an earlier ``if`` split ran the same loop and
        built the same choice, and two dimensions for it made every later join see two different values
        (_read_unprocessed_inputs: 14 dimensions, nested 8 deep, for a template with 2)."""
        vars = {k: v for k, v in group[0].vars.items() if k not in self.mergeable}
        names: dict = {}
        for env in group:
            for name in env.vars:
                if name in self.mergeable:
                    names[name] = None
        group = [group[0]] + [_aligned(group[0], env, names) for env in group[1:]]
        differing = []
        for name in names:
            values = [env.vars.get(name, UNK) for env in group]
            first = values[0]
            if all(v is first or v == first for v in values[1:]):
                if first is not UNK:
                    vars[name] = first
            else:
                differing.append((name, tuple(values)))
        if differing:
            try:
                key = tuple(differing)
                dim = made.get(key)
            except TypeError:
                key, dim = None, None
            if dim is None:
                dim = Dim(len(group), getattr(self, "_at_line", 0))
                if key is not None:
                    made[key] = dim
            for name, values in differing:
                vars[name] = factor(dim, list(values))
        self.stats["merges"] += 1
        return Env(vars, group[0].facts)

    def concretize(self, envs: list[Env], names: frozenset) -> list[Env]:
        """Each path whose *names* hold a choice, split into one path per branch of it -- every name the same branch."""
        out = []
        for env in envs:
            found: frozenset = _NODIMS
            for name in names:
                value = env.vars.get(name)
                if value is not None:
                    d = dims(value)
                    if d:
                        found = found | d
            if not found:
                out.append(env)
                continue
            order = sorted(found, key=_dim_id)
            total = 1
            for dim in order:
                total *= dim.n
            if total > MAX_FORK:
                self.stats["fork_capped"] += 1
                self.widened = True
                out.append(env)
                continue
            self.stats["forks"] += 1
            for chosen in itertools.product(*(range(dim.n) for dim in order)):
                vars = env.vars
                for dim, index in zip(order, chosen, strict=False):
                    vars = {k: subst(v, dim, index) for k, v in vars.items()}
                out.append(Env(vars, env.facts))
        return out

    def ctl_names(self, stmt: ast.stmt) -> frozenset:
        """The tracked names a statement reads to decide which way it goes (its test, its loop's sequence, a condition
        inside its expressions): a choice there splits the path first."""
        key = ("ctl", id(stmt))
        held = self._loads.get(key)
        if held is not None:
            return held
        found: set[str] = set()
        if isinstance(stmt, (ast.If, ast.While)):
            found = _names(stmt.test)
        elif isinstance(stmt, (ast.For, ast.AsyncFor)):
            found = _names(stmt.iter)
        elif isinstance(stmt, ast.Match):
            found = _names(stmt.subject)
        elif not isinstance(
            stmt,
            (
                ast.Try,
                ast.With,
                ast.AsyncWith,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
            ),
        ) and not isinstance(stmt, getattr(ast, "TryStar", ast.Try)):
            for node in ast.walk(stmt):
                if isinstance(node, ast.IfExp):
                    found |= _names(node.test)
                elif isinstance(node, ast.BoolOp):
                    found |= _names(node)
                elif isinstance(node, ast.comprehension):
                    found |= _names(node.iter)
                    for test in node.ifs:
                        found |= _names(test)
                elif isinstance(node, ast.Assert):
                    found |= _names(node.test)
        held = frozenset(found & self.relevant)
        self._loads[key] = held
        return held

    # ---------------------------------------------------------- environments

    def assign_name(self, env: Env, name: str, value, rebound: bool = True) -> Env:
        """*name* holding *value*.  Facts on tests that read it are forgotten -- those on its attributes only where
        it is *rebound*, not where a call it was handed may have changed its value (:meth:`lose`; :meth:`kills`
        decides those)."""
        facts = env.facts
        if any(
            name in self.fact_names[key] and (rebound or key not in self.fact_attrs)
            for key in facts
        ):
            facts = {
                key: b
                for key, b in facts.items()
                if name not in self.fact_names[key]
                or (not rebound and key in self.fact_attrs)
            }
        if name not in self.relevant:
            if name in env.vars:
                return Env({k: v for k, v in env.vars.items() if k != name}, facts)
            return env if facts is env.facts else Env(env.vars, facts)
        vars = dict(env.vars)
        if value is UNK:
            vars.pop(name, None)
        else:
            vars[name] = value if name in self.text_names else plain(value)
        return Env(vars, facts)

    def forget(self, env: Env, name: str) -> Env:
        if name not in env.vars and not any(
            name in self.fact_names[key] for key in env.facts
        ):
            return env
        vars = {k: v for k, v in env.vars.items() if k != name}
        facts = {
            key: b for key, b in env.facts.items() if name not in self.fact_names[key]
        }
        return Env(vars, facts)

    def bind(self, env: Env, target: ast.expr, value) -> list[Env]:
        if isinstance(target, ast.Name):
            return [self.assign_name(env, target.id, value)]
        if isinstance(target, (ast.Tuple, ast.List)):
            items = self._unpack(value, len(target.elts))
            if any(isinstance(element, ast.Starred) for element in target.elts):
                items = None
            if items is None:
                for element in target.elts:
                    env = self.bind_unknown(env, element)
                return [env]
            envs = [env]
            for element, item in zip(target.elts, items, strict=False):
                envs = [out for one in envs for out in self.bind(one, element, item)]
            return envs
        if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
            name = target.value.id
            if name not in self.relevant or self._unread_key(target):
                return [env]
            container = env.vars.get(name, UNK)
            out = []
            for key in self.ev(target.slice, env):
                new = self.lift(
                    lambda k: distribute(lambda c: _put(c, k, value), container), [key]
                )
                out.append(self.assign_name(env, name, new))
            return out
        if isinstance(target, ast.Starred):
            return [self.bind_unknown(env, target.value)]
        return [env]

    def _unpack(self, value, n: int) -> Optional[list]:
        if type(value) is Dct:
            return None
        items = sequence(value)
        if items is not None:
            return list(items) if len(items) == n else None
        if type(value) is Choice:

            def item(v, k):
                found = sequence(v) if type(v) is not Dct else None
                return found[k] if found is not None and len(found) == n else UNK

            return [distribute(lambda v, k=k: item(v, k), value) for k in range(n)]
        return None

    def bind_unknown(self, env: Env, target: ast.expr) -> Env:
        for name in (
            _names(target)
            if not isinstance(target, (ast.Attribute, ast.Subscript))
            else set()
        ):
            env = self.assign_name(env, name, UNK)
        if (
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and target.value.id in self.relevant
            and not self._unread_key(target)
        ):
            env = self.assign_name(
                env,
                target.value.id,
                self._unknown_key_write(env.vars.get(target.value.id, UNK)),
            )
        return env

    @staticmethod
    def _unknown_key_write(container):
        """A container after a write at a key that is not known: a dict holds at least one entry (g)."""
        if type(container) is Dct:
            return Opn(max(1, len(container.items)))
        if type(container) is Opn:
            return Opn(max(1, container.least))
        return UNK

    def _unread_key(self, target: ast.Subscript) -> bool:
        keys = (
            self.key_only.get(target.value.id)
            if isinstance(target.value, ast.Name)
            else None
        )
        return (
            keys is not None
            and isinstance(target.slice, ast.Constant)
            and isinstance(target.slice.value, str)
            and target.slice.value not in keys
        )

    def lift(
        self,
        f: Callable,
        args: list,
        over: Optional[list[int]] = None,
        node: Optional[ast.AST] = None,
    ):
        budget = [MAX_LIFT]
        try:
            return lift(f, args, over, budget)
        except _LiftCap:
            self.stats["lift_capped"] += 1
            return (
                self.hole(node, "too many variants to apply this to")
                if node is not None
                else UNK
            )
        finally:
            self.stats["work"] += MAX_LIFT - max(budget[0], 0)

    # ---------------------------------------------------------- tests

    def decide(self, test: ast.expr, env: Env) -> Optional[bool]:
        if _factual(test):
            key = _spelled(test)
            if key in env.facts:
                return env.facts[key]
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            inner = self.decide(test.operand, env)
            return None if inner is None else not inner
        if isinstance(test, ast.BoolOp):
            results = [self.decide(value, env) for value in test.values]
            stop = isinstance(test.op, ast.Or)
            if any(result is stop for result in results):
                return stop
            return (
                (not stop) if all(result is (not stop) for result in results) else None
            )
        values = self.ev(test, env)
        truths = set()
        for value in values:
            if dims(value):
                self.stats["lift_miss"] += 1
            truths.add(truth(value))
        return truths.pop() if len(truths) == 1 else None

    def assume(self, test: ast.expr, outcome: bool, env: Env) -> Env:
        if (
            _factual(test)
            and not isinstance(test, ast.Name)
            and not isinstance(test, ast.Compare)
        ):
            key = _spelled(test)
            if key in self.repeated:
                facts = dict(env.facts)
                facts[key] = outcome
                env = Env(env.vars, facts)
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            return self.assume(test.operand, not outcome, env)
        if isinstance(test, ast.BoolOp):
            if isinstance(test.op, ast.And) == outcome:
                for value in test.values:
                    env = self.assume(value, outcome, env)
            return env
        if _factual(test):
            key = _spelled(test)
            if key in self.repeated:
                facts = dict(env.facts)
                facts[key] = outcome
                return Env(env.vars, facts)
        return env

    def split(self, test: ast.expr, envs: list[Env]) -> tuple[list[Env], list[Env]]:
        yes, no = [], []
        for env in envs:
            result = self.decide(test, env)
            if result is not False:
                yes.append(self.assume(test, True, env) if result is None else env)
            if result is not True:
                no.append(self.assume(test, False, env) if result is None else env)
        return yes, no

    # ---------------------------------------------------------- statements

    def tracks(self, stmt: ast.stmt) -> bool:
        targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
        for target in targets:
            for node in ast.walk(target):
                if isinstance(node, ast.Name) and node.id in self.relevant:
                    return True
        return False

    def builds_text(self, stmt: ast.stmt) -> bool:
        """Whether an assignment binds a name a text is built from (where a call's return is worth reading)."""
        key = ("text", id(stmt))
        held = self._loads.get(key)
        if held is None:
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            held = any(
                name in self.text_names
                for target in targets
                for name, _ in self._writes(target)
            )
            self._loads[key] = held
        return held

    def hold_use(self, name: str, envs: list[Env]) -> None:
        held = self.used.setdefault(name, {})
        for env in envs:
            one = env.vars.get(name)
            if one is not None:
                held[one] = None

    def hold_exit(self, envs: list[Env]) -> None:
        """Mode ``uses``: what every built name holds where the function returns or ends."""
        for name in sorted(self.built_names):
            self.hold_use(name, envs)

    def record(self, node: Optional[ast.AST], envs: list[Env]) -> None:
        if self.mode == "uses":
            if node is not None and self.contains_target(node):
                for here in _expression_nodes(node):
                    if id(here) in self._targets:
                        self.hold_use(here.id, envs)
            return
        if node is None or self.mode == "return" or not self.contains_target(node):
            return
        calls = [
            here
            for here in _expression_nodes(node)
            if isinstance(here, ast.Call)
            and isinstance(here.func, ast.Attribute)
            and here.func.attr in RUNNERS
            and here.args
        ]
        for call in calls:
            line = call.lineno
            if self.line is not None and line != self.line:
                continue
            self.reached.add(line)
            held = self.values.setdefault(line, {})
            named = self.names_at.setdefault(line, {})
            inside = [name for name in _names(call.args[0]) if name in self.text_names]
            statement, tag = self.stated(call)
            self._for_text = True
            try:
                for env in envs:
                    run = call.args[0]
                    if tag is not None and all(
                        _no_sql(one) for one in self.ev(tag, env)
                    ):
                        run = statement
                    for value in self.ev(run, env):
                        held[value] = None
                        for name in inside:
                            one = env.vars.get(name)
                            if one is not None:
                                named.setdefault(value, {}).setdefault(name, {})[
                                    one
                                ] = None
            finally:
                self._for_text = False
            if self.widened:
                self.widened_lines.add(line)

    def stated(self, call: ast.Call) -> tuple[ast.expr, Optional[ast.expr]]:
        """What a runner's first argument runs, and the tracing tag it ends in.

        With ``conventions["tracing_tag"]``, as the map's readers take it:
        ``execute(sql + comment)`` runs ``sql`` tagged ``comment``, and so does
        ``f"{sql} {comment}"``, an f-string whose literal parts are only
        whitespace.  The tag is left out where it carries no SQL (:func:`_no_sql`)
        -- a reader of ``SET a=:a`` reads `` /* DBProxy.x */`` after it as part
        of the value.  Otherwise the whole argument, and no tag.
        """
        expression = call.args[0]
        if not self.conventions.get("tracing_tag"):
            return expression, None
        held = self._stated.get(id(call))
        if held is None:
            held = (expression, None)
            if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Add):
                held = (expression.left, expression.right)
            elif isinstance(expression, ast.JoinedStr) and not any(
                isinstance(value, ast.Constant)
                and isinstance(value.value, str)
                and value.value.strip()
                for value in expression.values
            ):
                interpolated = [
                    index
                    for index, value in enumerate(expression.values)
                    if isinstance(value, ast.FormattedValue)
                ]
                if len(interpolated) > 1:
                    last = interpolated[-1]
                    statement = ast.copy_location(
                        ast.JoinedStr(values=expression.values[:last]), expression
                    )
                    self._stmt[id(statement)] = self._stmt.get(id(expression))
                    held = (statement, expression.values[last].value)
            self._stated[id(call)] = held
        return held

    def contains_target(self, node: ast.AST) -> bool:
        key = id(node)
        if key not in self._contains:
            self._contains[key] = any(
                id(n) in self._targets for n in _expression_nodes(node)
            )
        return self._contains[key]

    def loads(self, node: ast.AST) -> frozenset:
        key = id(node)
        if key not in self._loads:
            found = set()
            for n in _expression_nodes(node):
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                    found.add(n.id)
                elif isinstance(n, ast.AugAssign):
                    found |= _names(n.target)
            found |= {"?" + test for test in self.tests(node)}
            self._loads[key] = frozenset(found)
        return self._loads[key]

    def tests(self, node: ast.AST) -> frozenset:
        key = ("tests", id(node))
        if key not in self._loads:
            found = set()
            for n in _expression_nodes(node):
                if isinstance(n, (ast.If, ast.While, ast.IfExp)):
                    for part in _test_keys(n.test):
                        if _factual(part):
                            spelled = _spelled(part)
                            if spelled in self.repeated:
                                found.add(spelled)
            self._loads[key] = frozenset(found)
        return self._loads[key]

    def kills(self, stmt: ast.stmt) -> frozenset:
        """The facts on attributes *stmt* forgets as it starts (:func:`_header`): a call handed an object or attribute a
        fact reads (as receiver or argument), and an assignment to one.  ``self.cur.execute(...)`` forgets
        ``self.backend == 'oracle'`` only if handed ``self`` or ``self.backend``; ``self.cur.arraysize = 100`` does not."""
        key = ("kills", id(stmt))
        held = self._loads.get(key)
        if held is None:
            found: set[str] = set()
            for root in _header(stmt):
                for node in _expression_nodes(root):
                    if isinstance(node, ast.Call):
                        handed = (
                            [node.func.value]
                            if isinstance(node.func, ast.Attribute)
                            else []
                        )
                        handed += [
                            a.value if isinstance(a, ast.Starred) else a
                            for a in node.args
                        ]
                        handed += [k.value for k in node.keywords]
                    elif isinstance(node, ast.Attribute) and isinstance(
                        node.ctx, (ast.Store, ast.Del)
                    ):
                        handed = [node]
                    elif isinstance(node, ast.Subscript) and isinstance(
                        node.ctx, (ast.Store, ast.Del)
                    ):
                        handed = [node.value]
                    else:
                        continue
                    spelled = {_spelled(one) for one in handed}
                    found |= {
                        fact
                        for fact, touched in self.fact_attrs.items()
                        if spelled & touched
                    }
            held = frozenset(found)
            self._loads[key] = held
        return held

    def forget_killed(self, stmt: ast.stmt, envs: list[Env]) -> list[Env]:
        """*envs* without the facts :meth:`kills` says *stmt* forgets."""
        if not self.fact_attrs:
            return envs
        killed = self.kills(stmt)
        if not killed or not any(key in killed for env in envs for key in env.facts):
            return envs
        return self.dedupe(
            [
                Env(env.vars, {k: b for k, b in env.facts.items() if k not in killed})
                for env in envs
            ]
        )

    def loads_of(self, stmts: list[ast.stmt]) -> frozenset:
        out: frozenset = frozenset()
        for stmt in stmts:
            out |= self.loads(stmt)
        return out

    def project(self, env: Env, live: frozenset) -> Env:
        if all(name in live for name in env.vars) and all(
            "?" + k in live for k in env.facts
        ):
            return env
        return Env(
            {k: v for k, v in env.vars.items() if k in live},
            {k: b for k, b in env.facts.items() if "?" + k in live},
        )

    def prune(self, envs: list[Env], live: frozenset) -> list[Env]:
        out = []
        for env in envs:
            dead = [name for name in env.vars if name not in live]
            stale = [key for key in env.facts if "?" + key not in live]
            if dead or stale:
                env = Env(
                    {k: v for k, v in env.vars.items() if k in live},
                    {k: b for k, b in env.facts.items() if "?" + k in live},
                )
            out.append(env)
        return self.dedupe(out)

    def block(
        self,
        stmts: list[ast.stmt],
        envs: list[Env],
        states: Optional[list[Env]] = None,
        after: frozenset = frozenset(),
    ):
        if (
            not self._loops_with_target
            and self.mode != "uses"  # it reads the names where the function ends, too
            and stmts
            and stmts[-1].end_lineno >= self._last_target
        ):
            holding = [i for i, stmt in enumerate(stmts) if self.contains_target(stmt)]
            if holding:
                stmts = stmts[: holding[-1] + 1]
        suffix = [frozenset()] * (len(stmts) + 1)
        for i in range(len(stmts) - 1, -1, -1):
            suffix[i] = suffix[i + 1] | self.loads(stmts[i])
        broken: list[Env] = []
        continued: list[Env] = []
        for i, stmt in enumerate(stmts):
            if not envs:
                break
            if states is not None:
                states.extend(envs)
            for seen, live in self._catching:
                for env in envs:
                    one = self.project(env, live)
                    seen.setdefault(one.key(), one)
            # Mode ``uses`` reads every built name where the function ends: live to the end.
            rest = after | suffix[i + 1] | self.built_names
            ctl = self.ctl_names(stmt)
            if ctl:
                envs = self.concretize(envs, ctl)
            self._at_line = stmt.lineno
            envs, b, c = self.stmt(stmt, envs, rest)
            envs = self.prune(envs, rest)
            broken += b
            continued += c
        if states is not None:
            states.extend(envs)
        return self.dedupe(envs), broken, continued

    def stmt(self, stmt: ast.stmt, envs: list[Env], after: frozenset = frozenset()):
        self.stats["work"] += len(envs)
        envs = self.forget_killed(stmt, envs)
        if isinstance(stmt, ast.Expr):
            self.record(stmt.value, envs)
            envs = self.lose(envs, self.escaped(stmt.value))
            return (
                self.dedupe(
                    [out for env in envs for out in self.effect(stmt.value, env)]
                ),
                [],
                [],
            )
        if isinstance(
            stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)
        ) and not self.tracks(stmt):
            self.record(getattr(stmt, "value", None), envs)
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            names = {
                n.id
                for target in targets
                for n in ast.walk(target)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
            }
            for name in names:
                envs = [self.assign_name(env, name, UNK) for env in envs]
            return self.dedupe(envs), [], []
        if isinstance(stmt, ast.Assign):
            self.record(stmt.value, envs)
            out = []
            lost = self.escaped(stmt.value)
            self._for_text = self.builds_text(stmt)
            try:
                evaluated = [(env, self.ev(stmt.value, env)) for env in envs]
            finally:
                self._for_text = False
            for env, values in evaluated:
                if lost:
                    env = self.lose([env], lost)[0]
                for value in values:
                    here = [env]
                    for target in stmt.targets:
                        one_value = (
                            carried(value, target.id, stmt)
                            if type(target) is ast.Name and target.id in self.text_names
                            else value
                        )
                        here = [
                            after
                            for one in here
                            for after in self.bind(one, target, one_value)
                        ]
                    out += here
            return self.dedupe(out), [], []
        if isinstance(stmt, ast.AnnAssign):
            if stmt.value is None:
                return envs, [], []
            self.record(stmt.value, envs)
            self._for_text = self.builds_text(stmt)
            try:
                evaluated = [(env, self.ev(stmt.value, env)) for env in envs]
            finally:
                self._for_text = False
            into = (
                stmt.target.id
                if type(stmt.target) is ast.Name and stmt.target.id in self.text_names
                else None
            )
            out = [
                after
                for env, values in evaluated
                for value in values
                for after in self.bind(
                    env, stmt.target, carried(value, into, stmt) if into else value
                )
            ]
            return self.dedupe(out), [], []
        if isinstance(stmt, ast.AugAssign):
            self.record(stmt.value, envs)
            if isinstance(stmt.target, ast.Name):
                expression = self._augmented.get(id(stmt))
                if expression is None:
                    load = ast.copy_location(
                        ast.Name(id=stmt.target.id, ctx=ast.Load()), stmt.target
                    )
                    expression = ast.copy_location(
                        ast.BinOp(left=load, op=stmt.op, right=stmt.value), stmt
                    )
                    self._augmented[id(stmt)] = expression
                    self._stmt[id(load)] = stmt
                    self._stmt[id(expression)] = stmt
                name = stmt.target.id
                self._for_text = name in self.text_names
                try:
                    out = [
                        self.assign_name(
                            env,
                            name,
                            carried(value, name, stmt) if self._for_text else value,
                        )
                        for env in envs
                        for value in self.ev(expression, env)
                    ]
                finally:
                    self._for_text = False
                return self.dedupe(out), [], []
            return (
                self.dedupe([self.bind_unknown(env, stmt.target) for env in envs]),
                [],
                [],
            )
        if isinstance(stmt, ast.If):
            self.record(stmt.test, envs)
            yes, no = self.split(stmt.test, envs)
            a = self.block(stmt.body, yes, after=after) if yes else ([], [], [])
            b = self.block(stmt.orelse, no, after=after) if no else ([], [], [])
            return self.dedupe(a[0] + b[0]), a[1] + b[1], a[2] + b[2]
        if isinstance(stmt, (ast.For, ast.AsyncFor, ast.While)):
            holds = self.contains_target(stmt)
            self._loops_with_target += holds
            try:
                if isinstance(stmt, ast.While):
                    return self.loop_while(stmt, envs, after | self.loads(stmt))
                return self.loop_for(stmt, envs, after | self.loads(stmt))
            finally:
                self._loops_with_target -= holds
        if isinstance(stmt, (ast.Try, getattr(ast, "TryStar", ast.Try))):
            return self.try_(stmt, envs, after)
        if isinstance(stmt, (ast.With, ast.AsyncWith)):
            for item in stmt.items:
                self.record(item.context_expr, envs)
                if item.optional_vars is not None:
                    envs = [self.bind_unknown(env, item.optional_vars) for env in envs]
            return self.block(stmt.body, self.dedupe(envs), after=after)
        if isinstance(stmt, ast.Return):
            if self.mode == "return":
                self._for_text = True
                try:
                    for env in envs:
                        self.returns.extend(
                            [
                                marked(value, RETURNED, stmt)
                                for value in self.ev(stmt.value, env)
                            ]
                            if stmt.value is not None
                            else [None]
                        )
                finally:
                    self._for_text = False
            else:
                self.record(stmt.value, envs)
                if self.mode == "uses":
                    self.hold_exit(envs)
            return [], [], []
        if isinstance(stmt, ast.Raise):
            self.record(stmt.exc, envs)
            return [], [], []
        if isinstance(stmt, ast.Break):
            return [], list(envs), []
        if isinstance(stmt, ast.Continue):
            return [], [], list(envs)
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return self.dedupe([self.forget(env, stmt.name) for env in envs]), [], []
        if isinstance(stmt, ast.Delete):
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    envs = [self.forget(env, target.id) for env in envs]
                else:
                    envs = [self.bind_unknown(env, target) for env in envs]
            return self.dedupe(envs), [], []
        if isinstance(stmt, ast.Match):
            self.record(stmt.subject, envs)
            out, broken, continued = list(envs), [], []
            for case in stmt.cases:
                here = [
                    self.bind_unknown(env, ast.Name(id=name, ctx=ast.Store()))
                    for env in envs
                    for name in _pattern_names(case.pattern)
                ] or list(envs)
                a = self.block(case.body, self.dedupe(here), after=after)
                out += a[0]
                broken += a[1]
                continued += a[2]
            return self.dedupe(out), broken, continued
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            for alias in stmt.names:
                name = (alias.asname or alias.name).split(".")[0]
                envs = [self.forget(env, name) for env in envs]
            return self.dedupe(envs), [], []
        if isinstance(stmt, ast.Assert):
            self.record(stmt.test, envs)
            return envs, [], []
        return envs, [], []

    def loop_names(self, stmt) -> frozenset:
        key = id(stmt)
        if key not in self._loop_names:
            nodes = [n for part in stmt.body for n in _expression_nodes(part)]
            if isinstance(stmt, (ast.For, ast.AsyncFor)):
                nodes += list(ast.walk(stmt.target))
            names = set()
            for n in nodes:
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                    names.add(n.id)
                elif (
                    isinstance(n, ast.Subscript)
                    and isinstance(n.ctx, ast.Store)
                    and isinstance(n.value, ast.Name)
                ):
                    if not self._unread_key(n):
                        names.add(n.value.id)
                elif (
                    isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute)
                    and n.func.attr in _MUTATORS
                    and isinstance(n.func.value, ast.Name)
                ):
                    names.add(n.func.value.id)
            self._loop_names[key] = frozenset(names & self.relevant)
        return self._loop_names[key]

    def loop_deciders(self, stmt) -> frozenset:
        key = ("deciders", id(stmt))
        if key not in self._loop_names:
            found: set[str] = set()
            for part in stmt.body:
                for n in _expression_nodes(part):
                    if isinstance(n, ast.If):
                        inside = [
                            m for s in n.body + n.orelse for m in _expression_nodes(s)
                        ]
                        if self._binds_text(inside):
                            found |= _names(n.test)
                    elif (
                        isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign))
                        and n.value is not None
                    ):
                        targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                        if any(
                            name in self.text_names
                            for t in targets
                            for name, _ in self._writes(t)
                        ):
                            found |= {
                                m
                                for e in ast.walk(n.value)
                                if isinstance(e, ast.IfExp)
                                for m in _names(e.test)
                            }
            self._loop_names[key] = frozenset(found & self.relevant)
        return self._loop_names[key]

    def _binds_text(self, nodes) -> bool:
        for n in nodes:
            if (
                isinstance(n, ast.Name)
                and isinstance(n.ctx, ast.Store)
                and n.id in self.text_names
            ):
                return True
            if (
                isinstance(n, ast.Subscript)
                and isinstance(n.ctx, ast.Store)
                and isinstance(n.value, ast.Name)
                and n.value.id in self.text_names
            ):
                return True
            if (
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr in _MUTATORS
                and isinstance(n.func.value, ast.Name)
                and n.func.value.id in self.text_names
            ):
                return True
        return False

    def grown_names(self, stmt) -> frozenset:
        """The text names a loop's body only appends to -- ``sql += ...``, ``parts.append(...)``,
        ``parts.extend(...)`` -- and reads nowhere else (nor a ``while``'s test): what a pass appends to one does not
        depend on what it holds, so the passes' part can be read as one (:meth:`restart`, :meth:`settled`)."""
        key = ("grown", id(stmt))
        if key not in self._loop_names:
            parts = list(stmt.body)
            if isinstance(stmt, ast.While):
                parts.append(stmt.test)
            else:
                parts.append(stmt.target)
            appended: set[str] = set()
            allowed: set[int] = set()
            for part in parts:
                for n in ast.walk(part):
                    if (
                        isinstance(n, ast.AugAssign)
                        and isinstance(n.op, ast.Add)
                        and isinstance(n.target, ast.Name)
                    ):
                        appended.add(n.target.id)
                        allowed.add(id(n.target))
                    elif (
                        isinstance(n, ast.Call)
                        and isinstance(n.func, ast.Attribute)
                        and n.func.attr in ("append", "extend")
                        and isinstance(n.func.value, ast.Name)
                        and len(n.args) == 1
                        and not n.keywords
                    ):
                        appended.add(n.func.value.id)
                        allowed.add(id(n.func.value))
            used: set[str] = set()
            for part in parts:
                for n in ast.walk(part):
                    if isinstance(n, ast.Name) and id(n) not in allowed:
                        used.add(n.id)
                    elif isinstance(n, (ast.Global, ast.Nonlocal)):
                        used.update(n.names)
            # Texts only: a list of objects the run does not keep between passes either
            # (:meth:`any_pass`).
            self._loop_names[key] = frozenset(
                (appended - used) & self.loop_names(stmt) & self.text_names
            )
        return self._loop_names[key]

    def marks(self, stmt) -> dict:
        """Name -> the marks of the first pass and of the later ones, for each name *stmt* only appends to."""
        return {
            name: (HoleAt(stmt, name, MARK), HoleAt(stmt, name, MARK_LATER))
            for name in self.grown_names(stmt)
        }

    def restart(self, env: Env, marks: dict, later: bool) -> Env:
        """*env* at the start of a pass of a loop of unknown length.  Each name the loop only appends to (*marks*:
        name -> the first pass's mark and the later passes') ends with this pass's mark, which what it appends
        follows: what the first pass appended stays before a later pass's mark, what passes between appended is
        cut -- :meth:`settled` reads the passes as one part."""
        vars = env.vars
        for name, both in marks.items():
            value = vars.get(name)
            if value is None:
                continue
            origin = both[1] if later else both[0]
            mark = hole_str(origin)

            def again(before, _after=None, mark=mark):
                if type(before) is Lst:
                    return Lst(before.items + (mark,))
                return make_cat([before, mark]) if is_text(before) else before

            if holds(value, origin):
                new = around(value, origin, again)
            elif later and holds(value, both[0]):
                # Where the first pass appended nothing, the next one starts where it
                # did: the same pass again (the loop stops there, as a pass that changed
                # nothing), not a later one.
                first = hole_str(both[0])

                def kept(before, after, first=first, again=again, both=both):
                    return lift(
                        lambda b, a: (
                            again(_concat(_followed(b, first), a))
                            if _appended_any(a, both)
                            else _followed(b, first)
                        ),
                        [before, after],
                        over=[1],
                    )

                new = around(value, both[0], kept)
            else:
                new = distribute(again, value)
            if new is not value:
                if vars is env.vars:
                    vars = dict(vars)
                vars[name] = new
        return env if vars is env.vars else Env(vars, env.facts)

    @staticmethod
    def appended(envs: list[Env], marks: dict, later: bool) -> dict:
        """What a pass appended to each name in *marks*, on each path: what follows the pass's mark."""
        out: dict = {}
        for name, both in marks.items():
            origin = both[1] if later else both[0]
            held: list = []

            def keep(before, after, held=held):
                held.append(after)
                return before

            for env in envs:
                value = env.vars.get(name)
                if value is not None and holds(value, origin):
                    around(value, origin, keep)
            out[name] = held
        return out

    def settled(
        self, loop, marks: dict, added: list[dict], *groups: list
    ) -> list[list[Env]]:
        """*groups* of paths leaving a loop of unknown length, each name in *marks* holding what the passes appended
        as one part (:func:`repeated_part`) on each path one of them appended something, else what it held before.

        *added* is what each pass appended (:meth:`appended`).  Where the first pass appends what the later ones do,
        the part is every pass's; where it does not -- ``WHERE`` first, ``AND`` after -- what the first appended is
        kept as it is and the part is the later passes'."""
        joins: dict = {}
        for name in marks:
            first = added[0].get(name, []) if added else []
            rest = added[1].get(name, []) if len(added) > 1 else []
            alike = not rest or _pass_texts(first) == _pass_texts(rest)
            joins[name] = (
                alike,
                repeated_part(loop, name, first + rest if alike else rest),
            )
        out = []
        for group in groups:
            settled = []
            for env in group:
                vars, facts = env.vars, env.facts
                for name, (first, later) in marks.items():
                    value = vars.get(name)
                    if value is None:
                        continue
                    alike, join = joins[name]
                    both = marks[name]
                    if alike and holds(value, first):
                        new = around(
                            value,
                            first,
                            lambda before, after, join=join, both=both: self._grown(
                                before, after, join, both
                            ),
                        )
                    elif holds(value, first):
                        # A branch a later pass appended to: the first pass's text and
                        # the later passes' part.
                        new = around(
                            value,
                            later,
                            lambda before, after, join=join, both=both: self._grown(
                                _without(before, both[0]), after, join, both
                            ),
                        )
                        # A branch left after the first pass, or after a pass that ran
                        # as the first.
                        if holds(new, first):
                            new = _without(new, first)
                    else:
                        continue
                    if is_text(new):
                        new = carried(new, name, loop)
                    if vars is env.vars:
                        vars = dict(vars)
                    vars[name] = new
                    facts = {
                        key: b
                        for key, b in facts.items()
                        if name not in self.fact_names[key]
                    }
                settled.append(env if vars is env.vars else Env(vars, facts))
            out.append(self.dedupe(settled))
        return out

    @staticmethod
    def _grown(before, after, join: Callable, marks: tuple):
        """*before* and the part (*join*) on each path *after* -- what the passes appended -- holds something."""
        try:
            return lift(
                lambda b, a: join(b) if _appended_any(a, marks) else b,
                [before, after],
                over=[1],
            )
        except _LiftCap:
            return make_choice(Dim(2), (before, join(before)))

    def any_pass(
        self,
        env: Env,
        names: frozenset,
        deciders: frozenset = frozenset(),
        marks: Optional[dict] = None,
    ) -> Env:
        """The start of any pass of a loop of unknown length.  A bool a test deciding whether a call runs
        reads is kept as the passes before left it (a flag set by the pass that added to the text).  A name
        holding its mark (*marks*, :meth:`restart`) is kept: what a pass appends follows the mark."""
        for name in names:
            value = env.vars.get(name, UNK)
            if marks and name in marks and any(holds(value, m) for m in marks[name]):
                continue
            number = (
                isinstance(value, (int, float)) and not isinstance(value, bool)
            ) or type(value) is AtLeast
            if (
                value is not UNK
                and not number
                and (name in self.text_names or name in deciders)
            ):
                continue
            if isinstance(value, bool) and name in self.call_guards:
                continue
            env = self.forget(env, name)
        return env

    def loop_for(self, stmt: ast.For, envs: list[Env], after: frozenset):
        self.record(stmt.iter, envs)
        finished: list[Env] = []
        broken: list[Env] = []
        unknown: list[Env] = []
        known_groups: list[tuple[tuple, list[Env]]] = []
        by_items: dict = {}
        for env in envs:
            for value in self.ev(stmt.iter, env):
                items = sequence(value)
                if items is not None and len(items) <= MAX_UNROLL:
                    try:
                        index = by_items.setdefault(items, len(known_groups))
                    except TypeError:
                        index = len(known_groups)
                    if index == len(known_groups):
                        known_groups.append((items, []))
                    known_groups[index][1].append(env)
                else:
                    unknown.append(env)
        for items, group in known_groups:
            current = self.dedupe(group)
            for item in items:
                self._epoch += 1
                entered = [
                    after
                    for one in current
                    for after in self.bind(one, stmt.target, item)
                ]
                normal, b, c = self.block(stmt.body, self.dedupe(entered), after=after)
                broken += b
                current = self.dedupe(normal + c)
                if not current:
                    break
            finished += current
        if unknown:
            current = self.dedupe(unknown)
            if UNKNOWN_EMPTY:
                finished += current
            names, deciders = self.loop_names(stmt), self.loop_deciders(stmt)
            marks = self.marks(stmt)
            began: set = set()
            ran: list[Env] = []
            stopped: list[Env] = []
            added: list[dict] = []
            for passes in range(UNKNOWN_PASSES):
                self._epoch += 1
                entered = self.dedupe(
                    [
                        self.bind_unknown(
                            self.any_pass(
                                self.restart(one, marks, passes > 0),
                                names,
                                deciders,
                                marks,
                            ),
                            stmt.target,
                        )
                        for one in current
                    ]
                )
                if all(one.key() in began for one in entered):
                    break
                began |= {one.key() for one in entered}
                normal, b, c = self.block(stmt.body, entered, after=after)
                stopped += b
                current = self.dedupe(normal + c)
                ran += current
                if marks:
                    added.append(self.appended(current + b, marks, passes > 0))
                if not current:
                    break
            if marks:
                ran, stopped = self.settled(stmt, marks, added, ran, stopped)
            finished += ran
            broken += stopped
        done, b2, c2 = (
            self.block(stmt.orelse, self.dedupe(finished), after=after)
            if stmt.orelse
            else (self.dedupe(finished), [], [])
        )
        return self.dedupe(done + broken), b2, c2

    def loop_while(self, stmt: ast.While, envs: list[Env], after: frozenset):
        self.record(stmt.test, envs)
        enter, leave = self.split(stmt.test, envs)
        finished = list(leave)
        broken = []
        names, deciders = self.loop_names(stmt), self.loop_deciders(stmt)
        marks = self.marks(stmt)
        began: set = set()
        ran: list[Env] = []
        stopped: list[Env] = []
        added: list[dict] = []
        for passes in range(UNKNOWN_PASSES):
            self._epoch += 1
            enter = self.dedupe(
                [
                    self.any_pass(
                        self.restart(one, marks, passes > 0), names, deciders, marks
                    )
                    for one in enter
                ]
            )
            if not enter or all(one.key() in began for one in enter):
                break
            began |= {one.key() for one in enter}
            normal, b, c = self.block(stmt.body, enter, after=after)
            stopped += b
            again = self.dedupe(normal + c)
            if marks:
                added.append(self.appended(again + b, marks, passes > 0))
            if passes == UNKNOWN_PASSES - 1:
                ran += again
                break
            again = self.concretize(again, self.ctl_names(stmt))
            again = self.forget_killed(stmt, again)
            enter, leave = self.split(stmt.test, again)
            ran += leave
        if marks:
            ran, stopped = self.settled(stmt, marks, added, ran, stopped)
        broken += stopped
        finished = self.dedupe(finished + ran)
        done, b2, c2 = (
            self.block(stmt.orelse, finished, after=after)
            if stmt.orelse
            else (finished, [], [])
        )
        return self.dedupe(done + broken), b2, c2

    def try_(self, stmt, envs: list[Env], after: frozenset):
        tail = after | self.loads_of(stmt.finalbody)
        handler_reads = tail | self.loads_of([n for h in stmt.handlers for n in h.body])
        handled = handler_reads | self.loads_of(stmt.orelse)
        seen: dict = {}
        if stmt.handlers:
            self._catching.append((seen, handler_reads))
        try:
            normal, broken, continued = self.block(stmt.body, envs, None, after=handled)
        finally:
            if stmt.handlers:
                self._catching.pop()
        out: list[Env] = []
        if stmt.handlers:
            for env in normal:
                one = self.project(env, handler_reads)
                seen.setdefault(one.key(), one)
            starts = list(seen.values())
            if self.mode == "uses":
                # A statement the body was still building when it raised is not one the function builds: the
                # handler starts without what the body writes into a built name.
                cut = self.built_names & {
                    n.id
                    for part in stmt.body
                    for n in ast.walk(part)
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
                }
                for name in sorted(cut):
                    starts = [self.forget(env, name) for env in starts]
                starts = self.dedupe(starts) if cut else starts
            for handler in stmt.handlers:
                here = starts
                if handler.name:
                    here = self.dedupe([self.forget(env, handler.name) for env in here])
                a = self.block(handler.body, here, after=tail)
                out += a[0]
                broken += a[1]
                continued += a[2]
        if stmt.orelse:
            a = self.block(stmt.orelse, normal, after=tail)
            out += a[0]
            broken += a[1]
            continued += a[2]
        else:
            out += normal
        out = self.dedupe(out)
        if stmt.finalbody:
            out, b, c = self.block(stmt.finalbody, out, after=after)
            broken += b
            continued += c
        return out, broken, continued

    def escaped(self, node: Optional[ast.AST]) -> set[str]:
        found: set[str] = set()
        if node is None:
            return found
        for call in _expression_nodes(node):
            if not isinstance(call, ast.Call):
                continue
            func = call.func
            direct = isinstance(func, ast.Attribute) and isinstance(
                func.value, ast.Name
            )
            modelled = (
                (isinstance(func, ast.Name) and func.id in _BUILTINS)
                or (
                    isinstance(func, ast.Attribute)
                    and func.attr in _STR_METHODS | _CONTAINER_METHODS
                )
                or (direct and func.attr in _MUTATORS)
                or (
                    isinstance(func, ast.Attribute)
                    and func.attr == "sub"
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "re"
                )
            )
            if not modelled:
                found |= {
                    arg.id
                    for arg in call.args
                    if isinstance(arg, ast.Name) and arg.id in self.relevant
                }
                found |= {
                    k.value.id
                    for k in call.keywords
                    if isinstance(k.value, ast.Name) and k.value.id in self.relevant
                }
            if isinstance(func, ast.Attribute) and not direct:
                root = func.value
                while isinstance(root, (ast.Attribute, ast.Call, ast.Subscript)):
                    root = root.func if isinstance(root, ast.Call) else root.value
                if (
                    isinstance(root, ast.Name)
                    and root.id in self.relevant
                    and func.attr not in _STR_METHODS | _CONTAINER_METHODS
                ):
                    found.add(root.id)
        return found

    def lose(self, envs: list[Env], names: set[str]) -> list[Env]:
        for name in names:
            envs = [
                env
                if immutable(env.vars.get(name))
                else self.assign_name(env, name, UNK, rebound=False)
                for env in envs
            ]
        return envs

    def effect(self, value: ast.expr, env: Env) -> list[Env]:
        if not (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and isinstance(value.func.value, ast.Name)
            and value.func.value.id in self.relevant
        ):
            return [env]
        name, method = value.func.value.id, value.func.attr
        container = env.vars.get(name, UNK)
        if container is UNK:
            return [env]
        if value.keywords or any(isinstance(arg, ast.Starred) for arg in value.args):
            return (
                [self.assign_name(env, name, UNK)]
                if method in _MUTATORS and type(container) not in (Str, str, Tup)
                else [env]
            )
        out = []
        for args in self.ev_args(value.args, env):
            new = distribute(
                lambda c, args=args: _effect_one(c, method, args), container
            )
            out.append(env if new is container else self.assign_name(env, name, new))
        return out

    # ---------------------------------------------------------- texts

    def stmt_of(self, node: Optional[ast.AST]):
        return self._stmt.get(id(node)) if node is not None else None

    def why(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            if node.id in self.params:
                return "a parameter whose value is not given"
            if node.id not in self.relevant_assigned:
                return "a name the function does not bind"
            return "a value not known on this path"
        if isinstance(node, ast.Attribute):
            return "an attribute"
        if isinstance(node, ast.Call):
            return (
                "a call not followed"
                if self.resolver is None
                else "a call whose return is not known"
            )
        if isinstance(node, ast.Subscript):
            return "a subscript not settled"
        return "an expression not modeled"

    def hole(self, node: ast.AST, reason: Optional[str] = None) -> Str:
        return hole_str(
            HoleAt(self.stmt_of(node), _source(node), reason or self.why(node))
        )

    def as_text(self, value, node: ast.AST):
        """*value*, a text, with a plain string's pieces written by the statement of *node*."""
        if type(value) is str:
            return lit(value, self.stmt_of(node))
        return value

    def text_of(self, value, node: ast.AST):
        """``str(value)``: a text as it is; a known value printed; anything else a hole."""
        if is_text(value):
            return self.as_text(value, node)
        if type(value) is Choice:
            return distribute(lambda v: self.text_of(v, node), value)
        if value is UNK or not known(value):
            return self.hole(node)
        return lit(str(to_py(value)), self.stmt_of(node))

    @property
    def relevant_assigned(self) -> set[str]:
        cached = getattr(self, "_assigned", None)
        if cached is None:
            cached = {
                node.id
                for node in _own_nodes(self.func)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
            }
            self._assigned = cached
        return cached

    # ---------------------------------------------------------- expressions

    def ev_args(self, args: list[ast.expr], env: Env) -> list[tuple]:
        return _product([self.ev(arg, env) for arg in args])

    def ev(self, node: ast.expr, env: Env) -> list:
        values = self._ev(node, env)
        if len(values) > MAX_VALUES:
            raise Overflow(
                f"more than {MAX_VALUES} values at line {getattr(node, 'lineno', '?')}"
            )
        return values

    def _ev(self, node: ast.expr, env: Env) -> list:
        t = type(node)
        if t is ast.Constant:
            value = node.value
            if isinstance(value, str):
                return [lit(value, self.stmt_of(node))]
            return [value if isinstance(value, _SCALARS) else UNK]
        if t is ast.Name:
            return [env.vars.get(node.id, UNK)]
        if t is ast.JoinedStr:
            parts = []
            for piece in node.values:
                if isinstance(piece, ast.Constant):
                    parts.append([lit(piece.value, self.stmt_of(piece))])
                else:
                    parts.append(self.formatted(piece, env))
            return _distinct(make_cat(chosen) for chosen in _product(parts))
        if t is ast.BinOp:
            pairs = _product([self.ev(node.left, env), self.ev(node.right, env)])
            return _distinct(self.binop(node, left, right) for left, right in pairs)
        if t is ast.BoolOp:
            return self.boolop(node, env)
        if t is ast.UnaryOp:
            return _distinct(
                self.lift(lambda v: _unary(node.op, v), [value])
                for value in self.ev(node.operand, env)
            )
        if t is ast.Compare:
            return self.compare(node, env)
        if t is ast.IfExp:
            result = self.decide(node.test, env)
            if result is True:
                return self.ev(node.body, env)
            if result is False:
                return self.ev(node.orelse, env)
            return _distinct(self.ev(node.body, env) + self.ev(node.orelse, env))
        if t is ast.List or t is ast.Tuple:
            pieces = []
            for element in node.elts:
                if isinstance(element, ast.Starred):
                    pieces.append(
                        [("*", value) for value in self.ev(element.value, env)]
                    )
                else:
                    pieces.append([("", value) for value in self.ev(element, env)])
            kind_of = Lst if t is ast.List else Tup
            out = []
            for chosen in _product(pieces):
                items: list = []
                ok = True
                for kind, value in chosen:
                    if kind == "*":
                        spread = sequence(value)
                        if spread is None:
                            ok = False
                            break
                        items.extend(spread)
                    else:
                        items.append(value)
                out.append(kind_of(tuple(items)) if ok else UNK)
            return _distinct(out)
        if t is ast.Dict:
            if any(key is None for key in node.keys):
                return [UNK]
            pieces = [
                _product([self.ev(k, env), self.ev(v, env)])
                for k, v in zip(node.keys, node.values, strict=False)
            ]
            out = []
            for chosen in _product(pieces):
                d = Dct(())
                for k, v in chosen:
                    key = key_of(k)
                    if key is _MISSING:
                        d = UNK
                        break
                    d = d.put(key, v)
                out.append(d)
            return _distinct(out)
        if t is ast.Subscript:
            return self.subscript(node, env)
        if t is ast.Call:
            return self.call(node, env)
        if t is ast.ListComp or t is ast.GeneratorExp:
            return self.comprehension(node, env)
        return [UNK]

    def formatted(self, piece: ast.FormattedValue, env: Env) -> list:
        if piece.format_spec is None:
            specs = [""]
        else:
            specs = [
                v.s
                if type(v) is Str and HOLE not in v.s
                else (v if type(v) is str else None)
                for v in self.ev(piece.format_spec, env)
            ]
        conversion = chr(piece.conversion) if piece.conversion != -1 else ""
        out = []
        for value in self.ev(piece.value, env):
            for spec in specs:
                out.append(self.format_value(value, conversion, spec, piece.value))
        return _distinct(out)

    def format_value(self, value, conversion: str, spec: Optional[str], node: ast.AST):
        """One formatted field: a text with no conversion and no spec goes in as it is, pieces and choices kept."""
        if spec is None:
            return self.hole(node)
        if not conversion and not spec and is_text(value):
            return self.as_text(value, node)
        if type(value) is Choice:
            return distribute(
                lambda v: self.format_value(v, conversion, spec, node), value
            )
        if value is UNK or not known(value):
            return self.hole(node)
        py = to_py(value)
        try:
            if conversion == "r":
                py = repr(py)
            elif conversion == "s":
                py = str(py)
            elif conversion == "a":
                py = ascii(py)
            return lit(format(py, spec), self.stmt_of(node))
        except Exception:
            return self.hole(node)

    def binop(self, node: ast.BinOp, left, right):
        op = node.op
        if isinstance(op, ast.Add):
            text_left, text_right = is_text(left), is_text(right)
            if text_left and text_right:
                return make_cat(
                    [self.as_text(left, node.left), self.as_text(right, node.right)]
                )
            if text_left and right is UNK:
                return make_cat([self.as_text(left, node.left), self.hole(node.right)])
            if left is UNK and text_right:
                return make_cat([self.hole(node.left), self.as_text(right, node.right)])
            if type(left) is Choice or type(right) is Choice:
                chosen = left if type(left) is Choice else right
                return make_choice(
                    chosen.dim,
                    [
                        self.binop(
                            node,
                            b if chosen is left else left,
                            b if chosen is right else right,
                        )
                        for b in chosen.branches
                    ],
                )
            if type(left) is Lst and type(right) is Lst:
                return Lst(left.items + right.items)
            if type(left) is Tup and type(right) is Tup:
                return Tup(left.items + right.items)
        if isinstance(op, ast.Mod) and is_text(left):
            return self.lift(
                lambda template: self.percent(node, template, right), [left], node=node
            )
        if dims(left) or dims(right):
            return self.lift(
                lambda a, b: self.binop(node, a, b), [left, right], node=node
            )
        return _binop(op, left, right)

    def percent(self, node: ast.BinOp, template, arg):
        """``template % arg``, the template's pieces and each argument's kept (a choice in an argument is a part)."""
        if type(arg) is Choice:
            return distribute(lambda a: self.percent(node, template, a), arg)
        if type(template) is str:
            template = lit(template, self.stmt_of(node.left))
        arg_nodes = (
            node.right.elts
            if isinstance(node.right, ast.Tuple)
            and not any(isinstance(e, ast.Starred) for e in node.right.elts)
            else None
        )
        if type(arg) is Tup:
            items, mapping = list(arg.items), None
        elif type(arg) is Dct:
            items, mapping = None, arg
        else:
            items, mapping = None, None
        s = template.s
        parts: list = []
        position = 0
        used = 0
        for match in _PERCENT.finditer(s):
            parts.append(str_slice(template, position, match.start()))
            position = match.end()
            if match["conv"] == "%":
                parts.append(str_slice(template, match.start(), match.start() + 1))
                continue
            if "*" in (match["width"] or "") + (match["prec"] or ""):
                return UNK
            value_node = node.right
            if match["key"] is not None:
                if mapping is not None:
                    value = mapping.lookup(match["key"])
                    value = UNK if value is _MISSING else value
                elif arg is UNK or type(arg) is Opn:
                    value = UNK
                else:
                    return UNK
            elif items is not None:
                if used >= len(items):
                    return UNK
                value = items[used]
                if arg_nodes is not None and used < len(arg_nodes):
                    value_node = arg_nodes[used]
                used += 1
            elif arg is UNK:
                value = UNK
            else:
                if used:
                    return UNK
                value = arg
                used += 1
            spec = (
                "%"
                + (match["flags"] or "")
                + (match["width"] or "")
                + ("." + match["prec"] if match["prec"] else "")
                + match["conv"]
            )
            parts.append(self.percent_value(value, spec, match["conv"], value_node))
        parts.append(str_slice(template, position, len(s)))
        if items is not None and used != len(items):
            return UNK
        return make_cat(parts)

    def percent_value(self, value, spec: str, conv: str, node: ast.AST):
        if spec == "%s" and is_text(value):
            return self.as_text(value, node)
        if type(value) is Choice:
            return distribute(lambda v: self.percent_value(v, spec, conv, node), value)
        if value is UNK or not known(value):
            return self.hole(node)
        try:
            return lit(spec % (to_py(value),), self.stmt_of(node))
        except Exception:
            return self.hole(node)

    def boolop(self, node: ast.BoolOp, env: Env) -> list:
        is_or = isinstance(node.op, ast.Or)
        results: list = []
        for index, operand in enumerate(node.values):
            last = index == len(node.values) - 1
            still = False
            for value in self.ev(operand, env):
                t = truth(value)
                if dims(value) and t is None:
                    self.stats["lift_miss"] += 1
                if last:
                    results.append(value)
                elif t is None:
                    results.append(value)
                    still = True
                elif t == is_or:
                    results.append(value)
                else:
                    still = True
            if not still:
                break
        return _distinct(results) or [UNK]

    def compare(self, node: ast.Compare, env: Env) -> list:
        out = []
        operands = [node.left] + list(node.comparators)
        for chosen in _product([self.ev(operand, env) for operand in operands]):
            out.append(
                self.lift(
                    lambda *values: _compare_chain(node.ops, values), list(chosen)
                )
            )
        return _distinct(out)

    def subscript(self, node: ast.Subscript, env: Env) -> list:
        out = []
        if isinstance(node.slice, ast.Slice):
            bounds = [
                self.ev(part, env) if part is not None else [None]
                for part in (node.slice.lower, node.slice.upper, node.slice.step)
            ]
            for base, lower, upper, step in _product(
                [self.ev(node.value, env)] + bounds
            ):
                ends = (
                    _ends(lower, upper, step)
                    if (type(base) is Cat or type(base) is Choice) and is_text(base)
                    else None
                )
                if ends is not None:
                    out.append(
                        drop_end(base, ends[1])
                        if ends[0] == 0
                        else drop_start(base, ends[0])
                    )
                else:
                    out.append(
                        self.lift(
                            lambda b, lo, hi, st: _sliced(b, lo, hi, st),
                            [base, lower, upper, step],
                            node=node,
                        )
                    )
            return _distinct(out)
        for base, key in _product([self.ev(node.value, env), self.ev(node.slice, env)]):
            out.append(self.lift(lambda b, k: _indexed(b, k), [base, key]))
        return _distinct(out)

    def call(self, node: ast.Call, env: Env) -> list:
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "sub"
            and isinstance(func.value, ast.Name)
            and func.value.id == "re"
            and "re" not in self.relevant_assigned
            and len(node.args) == 3
            and not node.keywords
        ):
            out = []
            for pattern, repl, text in self.ev_args(node.args, env):
                out.append(
                    self.lift(
                        lambda p, r, x: self._resub(p, r, x, node),
                        [pattern, repl, text],
                        node=node,
                    )
                )
            return _distinct(out)
        helper = self.conventions.get("in_list_helper")
        if (
            helper
            and (
                func.attr
                if isinstance(func, ast.Attribute)
                else getattr(func, "id", "")
            )
            == helper
        ):
            return [self.in_list(node)]
        if isinstance(func, ast.Attribute):
            method = func.attr
            if method not in _STR_METHODS | _CONTAINER_METHODS:
                return self.follow(node, env)
            out = []
            for receiver in self.ev(func.value, env):
                if method == "format" and is_text(receiver):
                    out += self.format_call(receiver, node, env)
                elif (
                    any(isinstance(arg, ast.Starred) for arg in node.args)
                    or node.keywords
                ):
                    out.append(UNK)
                elif method == "join" and is_text(receiver) and len(node.args) == 1:
                    for (items,) in self.ev_args(node.args, env):
                        out.append(
                            self.lift(
                                lambda sep, items=items: self.join(node, sep, items),
                                [receiver],
                                node=node,
                            )
                        )
                else:
                    # A read of a dict's own structure is not applied per branch of the
                    # choices inside it: the list ``items()`` makes holds them as they
                    # are.  Applied per branch, a dict joined by widening (its values
                    # choices) handed a loop a choice of lists, read as a loop of
                    # unknown length.
                    inside = type(receiver) is Dct and method in _DICT_READS
                    for args in self.ev_args(node.args, env):
                        over = list(range(1, len(args) + 1)) if inside else None
                        out.append(
                            self.lift(
                                lambda r, *a: self.method(node, r, method, a),
                                [receiver, *args],
                                over,
                                node=node,
                            )
                        )
            return _distinct(out)
        if (
            isinstance(func, ast.Name)
            and func.id in _BUILTINS
            and func.id not in env.vars
            and func.id not in self.relevant_assigned
        ):
            if node.keywords or any(isinstance(arg, ast.Starred) for arg in node.args):
                return [UNK]
            out = []
            for args in self.ev_args(node.args, env):
                if (
                    func.id == "str"
                    and len(args) == 1
                    and (is_text(args[0]) or type(args[0]) is Choice)
                ):
                    out.append(self.text_of(args[0], node.args[0]))
                else:
                    out.append(
                        self.lift(
                            lambda *a: self.builtin(node, func.id, a),
                            list(args),
                            node=node,
                        )
                    )
            return _distinct(out)
        return self.follow(node, env)

    def in_list(self, node: ast.Call):
        """``get_sql_IN_bind_variables(values, prefix=P)`` as the map reads it (sql.in_lists): one placeholder bind
        ``P__in`` standing for the list, and a map of binds whose keys are not known."""
        prefix = next(
            (
                k.value.value
                for k in node.keywords
                if k.arg == "prefix" and isinstance(k.value, ast.Constant)
            ),
            None,
        )
        if (
            prefix is None
            and len(node.args) > 1
            and isinstance(node.args[1], ast.Constant)
        ):
            prefix = node.args[1].value
        prefix = prefix if isinstance(prefix, str) and prefix else ":in"
        return Tup((lit(f"{prefix}__in", self.stmt_of(node)), Opn(0)))

    def _resub(self, pattern, repl, text, node):
        """``re.sub(pattern, repl, text)``: what the pattern does not touch keeps where it was written, and what it
        writes is the call's.  ``_read_unprocessed_inputs`` adds a hint to a statement whose terms were appended under
        ``if not simulation_with_file_stat``; read as one text the call wrote, the terms were under no test, and a
        loop's repeated part in it was text."""
        if (
            type(pattern) in (Str, str)
            and type(repl) in (Str, str)
            and known(pattern)
            and known(repl)
            and type(text) in (Str, str)
        ):
            try:
                compiled = re.compile(to_py(pattern))
                matches = list(compiled.finditer(to_py(text)))
                written = [match.expand(to_py(repl)) for match in matches]
            except (re.error, IndexError):
                return UNK
            if type(text) is not Str:
                return lit(compiled.sub(to_py(repl), text), self.stmt_of(node))
            out, at = EMPTY, 0
            for match, one in zip(matches, written, strict=True):
                kept = str_slice(text, at, match.start())
                out = str_concat(str_concat(out, kept), lit(one, self.stmt_of(node)))
                at = match.end()
            return str_concat(out, str_slice(text, at, len(text.s)))
        return UNK

    def join(self, node: ast.Call, sep, items):
        """``sep.join(items)``: each item's pieces kept, a choice inside an item a part of the text."""
        if type(items) is Choice:
            return distribute(lambda i: self.join(node, sep, i), items)
        found = sequence(items)
        if found is None:
            return self.hole(node.args[0])
        sep = self.as_text(sep, node.func)
        joined = [self.text_of(item, node.args[0]) for item in found]
        if type(sep) is Str and HOLE not in sep.s:
            # A loop of unknown length's items, one part: the passes are joined by *sep*.
            joined = [joined_by(item, sep.s) for item in joined]
        return str_join(sep, joined)

    def format_call(self, template, node: ast.Call, env: Env) -> list:
        positional_choices = []
        spread_unknown = False
        for arg in node.args:
            if isinstance(arg, ast.Starred):
                choices = []
                for value in self.ev(arg.value, env):
                    items = sequence(value)
                    if items is None:
                        spread_unknown = True
                        choices.append(())
                    else:
                        choices.append(tuple(items))
                positional_choices.append(("*", choices, arg))
            else:
                positional_choices.append(("", self.ev(arg, env), arg))
        keyword_choices = [
            (keyword.arg, self.ev(keyword.value, env), keyword.value)
            for keyword in node.keywords
        ]
        out = []
        for chosen in _product(
            [choices for _, choices, _ in positional_choices]
            + [choices for _, choices, _ in keyword_choices]
        ):
            positional: list = []
            for (kind, _, arg_node), value in zip(
                positional_choices, chosen, strict=False
            ):
                if kind == "*":
                    positional.extend((one, arg_node) for one in value)
                else:
                    positional.append((value, arg_node))
            named: dict = {}
            unknown_names = False
            for (name, _, value_node), value in zip(
                keyword_choices, chosen[len(positional_choices) :], strict=False
            ):
                if name is None:
                    if type(value) is Dct and all(
                        isinstance(k, str) for k in value.keys()
                    ):
                        named.update({k: (v, value_node) for k, v in value.items})
                    else:
                        unknown_names = True
                else:
                    named[name] = (value, value_node)
            filled = None
            if type(template) is Cat or type(template) is Choice:
                filled = self.format_parts(
                    node, template, positional, named, spread_unknown, unknown_names, {}
                )
            if filled is None:
                filled = self.lift(
                    lambda t, positional=positional, named=named, unknown_names=unknown_names: (
                        self.format_template(
                            node, t, positional, named, spread_unknown, unknown_names
                        )
                    ),
                    [template],
                    node=node,
                )
            out.append(filled)
        return _distinct(out)

    def format_parts(
        self,
        node: ast.Call,
        template,
        positional: list,
        named: dict,
        spread_unknown: bool,
        unknown_names: bool,
        done: dict,
    ):
        """``.format`` on a joined template piece by piece and branch by branch, without expanding it: right when every
        field is numbered or named and lies within one piece (``"ORDER BY {0}) "``).  ``None`` where that does not
        hold (a ``{}`` counts the fields before it, which differ by branch).  *done* holds each piece already filled:
        branches share pieces, and walked as a tree a template of 25 pieces was 469."""
        held = done.get(id(template))
        if held is not None:
            return held[1]
        t = type(template)
        out = None
        if t is Str or t is str:
            out = self.format_template(
                node,
                template,
                positional,
                named,
                spread_unknown,
                unknown_names,
                numbered_only=True,
            )
            out = None if out is None or not is_text(out) else out
        elif t is Choice:
            branches = []
            for branch in template.branches:
                one = self.format_parts(
                    node, branch, positional, named, spread_unknown, unknown_names, done
                )
                if one is None:
                    break
                branches.append(one)
            else:
                out = make_choice(template.dim, branches)
        elif t is Cat:
            parts = []
            for part in template.parts:
                one = self.format_parts(
                    node, part, positional, named, spread_unknown, unknown_names, done
                )
                if one is None:
                    break
                parts.append(one)
            else:
                out = make_cat(parts)
        done[id(template)] = (template, out)
        return out

    def format_template(
        self,
        node: ast.Call,
        template,
        positional: list,
        named: dict,
        spread_unknown: bool,
        unknown_names: bool,
        numbered_only: bool = False,
    ):
        if type(template) is str:
            template = lit(template, self.stmt_of(node))
        s = template.s
        try:
            parsed = list(_FORMATTER.parse(s))
        except ValueError:
            return None if numbered_only else UNK
        parts: list = []
        position = 0
        auto = 0
        for literal, field, spec, conversion in parsed:
            source_length = len(literal) + literal.count("{") + literal.count("}")
            piece = str_slice(template, position, position + source_length)
            if "{{" in piece.s or "}}" in piece.s:
                piece = _unescape(piece)
            parts.append(piece)
            position += source_length
            if field is None:
                continue
            end = _field_end(s, position)
            if end < 0:
                return UNK
            position = end
            if spec and "{" in spec:
                parts.append(self.hole(node))
                continue
            head = re.match(r"[^.\[]*", field).group(0)
            rest = field[len(head) :]
            if head == "" and numbered_only:
                return None
            if head == "":
                index: object = auto
                auto += 1
            elif head.isdigit():
                index = int(head)
            else:
                index = head
            if isinstance(index, int):
                if index < len(positional):
                    value, value_node = positional[index]
                else:
                    if not spread_unknown:
                        return UNK
                    value, value_node = UNK, node
            else:
                if index in named:
                    value, value_node = named[index]
                else:
                    if not unknown_names:
                        return UNK
                    value, value_node = UNK, node
            if rest:
                parts.append(self.hole(value_node))
                continue
            parts.append(
                self.format_value(value, conversion or "", spec or "", value_node)
            )
        return make_cat(parts)

    def method(self, node: ast.Call, receiver, method: str, args: tuple):
        t = type(receiver)
        if t is Str or t is str:
            text = receiver if t is Str else lit(receiver, self.stmt_of(node))
            plain_args = []
            for arg in args:
                if type(arg) is Str:
                    if HOLE in arg.s:
                        return UNK
                    plain_args.append(arg.s)
                elif isinstance(arg, (str, int, type(None))):
                    if isinstance(arg, str) and HOLE in arg:
                        return UNK
                    plain_args.append(arg)
                else:
                    return UNK
            s = text.s
            try:
                if method in ("startswith", "endswith"):
                    return UNK if HOLE in s else getattr(s, method)(*plain_args)
                if method == "split":
                    if HOLE in s:
                        return UNK
                    if not plain_args or (
                        len(plain_args) == 1
                        and (
                            plain_args[0] is None
                            or (isinstance(plain_args[0], str) and plain_args[0])
                        )
                    ):
                        return Lst(
                            tuple(
                                str_split(text, plain_args[0] if plain_args else None)
                            )
                        )
                    return Lst(
                        tuple(lit(x, self.stmt_of(node)) for x in s.split(*plain_args))
                    )
                if method in ("strip", "lstrip", "rstrip"):
                    out = getattr(s, method)(*plain_args)
                    start = (
                        len(s) - len(s.lstrip(*plain_args))
                        if method in ("strip", "lstrip")
                        else 0
                    )
                    return str_slice(text, start, start + len(out))
                if method in ("upper", "lower") and not plain_args:
                    return str_map(text, method)
                if (
                    method == "removeprefix"
                    and len(plain_args) == 1
                    and isinstance(plain_args[0], str)
                ):
                    return (
                        str_slice(text, len(plain_args[0]), len(s))
                        if s.startswith(plain_args[0])
                        else text
                    )
                if (
                    method == "removesuffix"
                    and len(plain_args) == 1
                    and isinstance(plain_args[0], str)
                ):
                    return (
                        str_slice(text, 0, len(s) - len(plain_args[0]))
                        if plain_args[0] and s.endswith(plain_args[0])
                        else text
                    )
                if (
                    method == "replace"
                    and len(plain_args) == 2
                    and isinstance(plain_args[0], str)
                    and plain_args[0]
                    and isinstance(plain_args[1], str)
                ):
                    new = (
                        args[1]
                        if type(args[1]) is Str
                        else lit(plain_args[1], self.stmt_of(node))
                    )
                    return str_join(new, str_split(text, plain_args[0]))
                out = getattr(s, method)(*plain_args)
            except Exception:
                return UNK
            return (
                lit(out, self.stmt_of(node))
                if isinstance(out, str)
                else (out if isinstance(out, _SCALARS) else UNK)
            )
        if t is Dct:
            if method == "get" and 1 <= len(args) <= 2:
                key = key_of(args[0])
                if key is _MISSING:
                    return UNK
                found = receiver.lookup(key)
                return (
                    (args[1] if len(args) == 2 else None)
                    if found is _MISSING
                    else found
                )
            if method == "keys" and not args:
                return Lst(receiver.keys())
            if method == "values" and not args:
                return Lst(tuple(v for _, v in receiver.items))
            if method == "items" and not args:
                return Lst(tuple(Tup((k, v)) for k, v in receiver.items))
            if method == "copy" and not args:
                return receiver
            return UNK
        if (t is Lst or t is Opn) and method == "copy" and not args:
            return receiver
        return UNK

    def builtin(self, node: ast.Call, name: str, args: tuple):
        if name == "str" and len(args) == 1:
            return self.text_of(args[0], node.args[0])
        if name == "len" and len(args) == 1:
            value = args[0]
            if type(value) is Str:
                return len(value.s) if HOLE not in value.s else UNK
            if type(value) is str:
                return len(value) if HOLE not in value else UNK
            if type(value) is Opn:
                return AtLeast(value.least)
            if type(value) is Dct and value.maybe:
                return AtLeast(len(value.items) - len(value.maybe))
            items = sequence(value)
            if items is not None and any(repeats(item) for item in items):
                # One item stands for what each pass of a loop of unknown length
                # appended.
                return AtLeast(len(items))
            return len(items) if items is not None else UNK
        if (
            name == "range"
            and 1 <= len(args) <= 3
            and all(isinstance(a, int) and not isinstance(a, bool) for a in args)
        ):
            spread = range(*args)
            return Lst(tuple(spread)) if len(spread) <= MAX_UNROLL else UNK
        if name in {"list", "tuple", "sorted", "reversed"} and len(args) == 1:
            items = sequence(args[0])
            if items is None:
                return UNK
            if name == "sorted":
                if not all(known(i) for i in items):
                    return UNK
                try:
                    items = tuple(sorted(items, key=to_py))
                except TypeError:
                    return UNK
            if name == "reversed":
                items = tuple(reversed(items))
            return Tup(tuple(items)) if name == "tuple" else Lst(tuple(items))
        if name == "enumerate" and len(args) == 1:
            items = sequence(args[0])
            return (
                Lst(tuple(Tup((i, item)) for i, item in enumerate(items)))
                if items is not None
                else UNK
            )
        if name == "zip" and args:
            seqs = [sequence(a) for a in args]
            if any(s is None for s in seqs):
                return UNK
            return Lst(tuple(Tup(tuple(row)) for row in zip(*seqs, strict=False)))
        if name == "int" and len(args) == 1:
            value = args[0]
            if isinstance(value, bool):
                return int(value)
            if isinstance(value, int):
                return value
            if type(value) in (Str, str) and HOLE not in to_py(value):
                try:
                    return int(to_py(value))
                except ValueError:
                    return UNK
            return UNK
        if name == "bool" and len(args) == 1:
            t = truth(args[0])
            return UNK if t is None else t
        return UNK

    # ---------------------------------------------------------- calls out (d)

    def follow(self, node: ast.Call, env: Env) -> list:
        """What a call the resolver places returns, run with its arguments' values; UNK when it cannot be read."""
        if self.resolver is None or self.hops >= RETURN_HOPS or not self._for_text:
            return [UNK]
        if any(isinstance(arg, ast.Starred) for arg in node.args) or any(
            k.arg is None for k in node.keywords
        ):
            return [UNK]
        name = (
            node.func.attr
            if isinstance(node.func, ast.Attribute)
            else getattr(node.func, "id", "")
        )
        if name in RUNNERS or name in _DATABASE_READS:
            return [UNK]  # what the database hands back
        held = _PLACED.get(id(node))
        if held is None or held[0] is not node:
            held = (node, self.resolver(self.func, node))
            _PLACED[id(node)] = held
        targets = held[1]
        if not targets or any(target.name == "__init__" for target in targets):
            return [
                UNK
            ]  # a class called: the value is the object made, not what __init__ returns
        try:
            combos = _product(
                [self.ev(arg, env) for arg in node.args]
                + [self.ev(k.value, env) for k in node.keywords]
            )
        except Overflow:
            return [UNK]
        out: list = []
        for combo in combos:
            positional = list(combo[: len(node.args)])
            named = {
                k.arg: v
                for k, v in zip(node.keywords, combo[len(node.args) :], strict=False)
            }
            for target in targets:
                got = self.returns_of(target, node, positional, named, env)
                if got is None:
                    self.stats["callee_failed"] += 1
                    return [UNK]
                out.extend(marked(value, CALLED, node) for value in got)
        out = _distinct(out)
        if len(out) > MAX_VALUES:
            # More values than a call hands on: joined into one, as widening joins paths
            # -- read as unknown, the dict getScoutJobData_JEDI returns lost the columns
            # setScoutJobData_JEDI writes from its keys.
            self.widened = True
            line = getattr(node, "lineno", 0)
            return [join_value(out, lambda n: Dim(n, line))]
        return out

    def returns_of(
        self, target, node: ast.Call, positional: list, named: dict, env: Env
    ) -> Optional[list]:
        if id(target) in self.stack or any(
            isinstance(n, (ast.Yield, ast.YieldFrom)) for n in _own_nodes(target)
        ):
            return None
        params = target.args
        names = [a.arg for a in params.posonlyargs + params.args]
        if (
            isinstance(node.func, ast.Attribute)
            and names
            and names[0] in ("self", "cls")
        ):
            names = names[1:]
        binding: dict = {}
        if len(positional) > len(names):
            if params.vararg is None:
                return None
            positional = positional[: len(names)]
        for name, value in zip(names, positional, strict=False):
            binding[name] = value
        keywords = {a.arg for a in params.kwonlyargs} | set(names)
        for name, value in named.items():
            if name in keywords:
                binding[name] = value
            elif params.kwarg is None:
                return None
        free: dict = {}
        if id(target) in _nested_ids(self.func):
            # A nested function sees the caller's names it does not bind (getJobs's
            # ``construct_where_clause``).
            local = (
                {
                    n.id
                    for n in ast.walk(target)
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
                }
                | set(names)
                | keywords
            )
            for name in _names(target):
                if name not in local and name in env.vars:
                    free[name] = env.vars[name]
        try:
            key = (
                id(target),
                tuple(sorted(binding.items(), key=lambda kv: kv[0])),
                tuple(sorted(free.items(), key=lambda kv: kv[0])),
            )
            hash(key)
        except TypeError:
            key = None
        held = _RETURNS.get(key) if key is not None else None
        if held is None or held[0] is not target:
            start = Dim.count
            one = Ref(
                target,
                binding,
                None,
                work_cap=CALLEE_WORK,
                resolver=self.resolver,
                mode="return",
                hops=self.hops + 1,
                free=free,
                stack=self.stack,
                entry=False,
                conventions={
                    k: v for k, v in self.conventions.items() if k != "globals"
                },
            )
            one.run()
            self.stats["callees"] += 1
            values = None if one.overflow else _distinct(one.returns)
            held = (target, values, start, Dim.count, one.widened)
            if key is not None:
                _RETURNS[key] = held
        _, values, start, end, widened = held
        if values is None:
            return None
        if widened:
            # The callee's paths were widened: what it returns may hold a value no path
            # through it does, so the caller's reading of it is widened too.
            self.widened = True
        # The callee's own choices are fresh for each call, and for each pass of a loop:
        # two calls, or one call run twice on one path, may take different paths through
        # it.  Paths running the call together share them.
        handed_key = (id(node), key, self._epoch) if key is not None else None
        if handed_key is not None and handed_key in self._handed:
            return self._handed[handed_key]
        made: dict = {}
        out = []
        for value in values:
            # Newest first, by number: a set of dimensions iterates by address, which
            # differs between runs, and the fresh dimensions' order decides which one an
            # operation splits first (:func:`lift` takes the lowest number) --
            # setScoutJobData_JEDI's steps moved 85,274 / 92,282 / 118,170 over three
            # runs. The newest are the outer joins: split first, the choices inside them
            # fall away on most branches (31,098 steps; oldest first, 493,050).
            for dim in sorted(dims(value), key=_dim_id, reverse=True):
                if start < dim.id <= end and dim not in made:
                    made[dim] = Dim(dim.n, dim.line)
            out.append(rename(value, made) if made else value)
        if handed_key is not None:
            self._handed[handed_key] = out
        return out

    # ---------------------------------------------------------- comprehensions

    def comprehension(self, node, env: Env) -> list:
        if len(node.generators) != 1:
            return [UNK]
        generator = node.generators[0]
        if generator.is_async:
            return [UNK]
        out = []
        for value in self.ev(generator.iter, env):
            items = sequence(value)
            if items is None or len(items) > MAX_UNROLL:
                out.append(UNK)
                continue
            element_choices = []
            ok = True
            for item in items:
                bound = self._bind_free(Env(dict(env.vars), {}), generator.target, item)
                if bound is None:
                    ok = False
                    break
                keep = True
                for test in generator.ifs:
                    t = self.decide(test, bound)
                    if t is None:
                        ok = False
                        break
                    keep = keep and t
                if not ok:
                    break
                if keep:
                    element_choices.append(self.ev(node.elt, bound))
            if not ok:
                out.append(UNK)
                continue
            out += [Lst(tuple(chosen)) for chosen in _product(element_choices)]
        return _distinct(out)

    def _bind_free(self, env: Env, target: ast.expr, value) -> Optional[Env]:
        if isinstance(target, ast.Name):
            vars = dict(env.vars)
            vars[target.id] = value
            return Env(vars, env.facts)
        if isinstance(target, (ast.Tuple, ast.List)):
            items = sequence(value) if type(value) is not Dct else None
            if items is None or len(items) != len(target.elts):
                return None
            for element, item in zip(target.elts, items, strict=False):
                env = self._bind_free(env, element, item)
                if env is None:
                    return None
            return env
        return None


# ================================================================ operations on
# concrete values


def _unescape(piece: Str) -> Str:
    out = piece
    for doubled in ("{{", "}}"):
        if doubled in out.s:
            out = str_join(
                str_slice(out, out.s.index(doubled), out.s.index(doubled) + 1),
                str_split(out, doubled),
            )
    return out


def _field_end(s: str, position: int) -> int:
    """The index just past the replacement field that begins at *position* (a ``{``), or -1."""
    if position >= len(s) or s[position] != "{":
        return -1
    depth = 0
    for index in range(position, len(s)):
        if s[index] == "{":
            depth += 1
        elif s[index] == "}":
            depth -= 1
            if depth == 0:
                return index + 1
    return -1


def _unary(op, value):
    if isinstance(op, ast.Not):
        t = truth(value)
        return UNK if t is None else (not t)
    if _number(value) and isinstance(op, ast.USub):
        return -value
    return UNK


def _binop(op: ast.operator, left, right):
    if isinstance(op, ast.Add):
        if _number(left) and _number(right):
            return left + right
        return UNK
    if isinstance(op, ast.Mod):
        if _number(left) and _number(right) and right:
            return left % right
        return UNK
    if isinstance(op, ast.Mult):
        if (
            type(left) is Str
            and isinstance(right, int)
            and not isinstance(right, bool)
            and 0 <= right <= MAX_UNROLL
        ):
            return str_repeat(left, right)
        if (
            type(left) in (Lst, Tup)
            and isinstance(right, int)
            and not isinstance(right, bool)
            and 0 <= right <= MAX_UNROLL
        ):
            return type(left)(left.items * right)
        if _number(left) and _number(right):
            return left * right
        return UNK
    if isinstance(op, ast.Sub) and _number(left) and _number(right):
        return left - right
    return UNK


def _sliced(base, lower, upper, step):
    if not all(
        b is None or (isinstance(b, int) and not isinstance(b, bool))
        for b in (lower, upper, step)
    ):
        return UNK
    t = type(base)
    if t is Str:
        if step in (None, 1):
            start, stop, _ = slice(lower, upper).indices(len(base.s))
            return str_slice(base, start, stop)
        out = base.s[lower:upper:step]
        return lit(out, base.runs[0][1] if base.runs else None)
    if t is str:
        return base[lower:upper:step]
    if t is Lst or t is Tup:
        return t(base.items[lower:upper:step])
    return UNK


def _indexed(base, key):
    t = type(base)
    if t is Dct:
        k = key_of(key)
        if k is _MISSING:
            return UNK
        found = base.lookup(k)
        return UNK if found is _MISSING else found
    if isinstance(key, int) and not isinstance(key, bool):
        if t is Str:
            if HOLE in base.s or not -len(base.s) <= key < len(base.s):
                return UNK
            index = key % len(base.s)
            return str_slice(base, index, index + 1)
        if t is str:
            return (
                base[key] if HOLE not in base and -len(base) <= key < len(base) else UNK
            )
        if t is Lst or t is Tup:
            return base.items[key] if -len(base.items) <= key < len(base.items) else UNK
    return UNK


def _put(container, key, value):
    t = type(container)
    if t is Dct:
        k = key_of(key)
        if k is _MISSING:
            return Opn(max(1, len(container.items)))
        return container.put(k, value)
    if t is Opn:
        return Opn(max(1, container.least))
    if (
        t is Lst
        and isinstance(key, int)
        and not isinstance(key, bool)
        and -len(container.items) <= key < len(container.items)
    ):
        items = list(container.items)
        items[key] = value
        return Lst(tuple(items))
    return UNK


def _effect_one(container, method: str, args: tuple):
    t = type(container)
    if t is Lst or t is Dct or t is Opn:
        return _mutated(container, method, args)
    if container is UNK:
        return UNK
    if method in _MUTATORS and not (t is Str or t is str or t is Tup):
        return UNK
    return container


def _mutated(container, method: str, args: tuple):
    t = type(container)
    if t is Lst:
        items = list(container.items)
        if method == "append" and len(args) == 1:
            return Lst(tuple(items + [args[0]]))
        if method == "extend" and len(args) == 1:
            more = sequence(args[0])
            return Lst(tuple(items + list(more))) if more is not None else UNK
        if method == "insert" and len(args) == 2 and isinstance(args[0], int):
            items.insert(args[0], args[1])
            return Lst(tuple(items))
        if method == "clear" and not args:
            return Lst(())
        if method == "pop" and not args and items:
            return Lst(tuple(items[:-1]))
        if (
            method == "remove"
            and len(args) == 1
            and known(args[0])
            and all(known(i) for i in items)
        ):
            target = to_py(args[0])
            for index, item in enumerate(items):
                if to_py(item) == target:
                    del items[index]
                    return Lst(tuple(items))
            return UNK
        if method == "reverse" and not args:
            return Lst(tuple(reversed(items)))
        if method == "sort" and not args and all(known(i) for i in items):
            try:
                return Lst(tuple(sorted(items, key=to_py)))
            except TypeError:
                return UNK
        return UNK if method in _MUTATORS else container
    if t is Dct:
        if method == "update" and len(args) == 1:
            if type(args[0]) is Dct:
                had = {k for k, _ in container.items} - container.maybe
                for k, v in args[0].items:
                    container = container.put(k, v)
                uncertain = args[0].maybe - had
                return (
                    Dct(container.items, container.maybe | uncertain)
                    if uncertain
                    else container
                )
            if type(args[0]) is Opn or args[0] is UNK:
                return Opn(
                    len(container.items)
                    if args[0] is UNK
                    else max(len(container.items), args[0].least)
                )
            return UNK
        if method == "setdefault" and len(args) == 2:
            k = key_of(args[0])
            if k is _MISSING:
                return Opn(max(1, len(container.items)))
            return (
                container
                if container.lookup(k) is not _MISSING
                else container.put(k, args[1])
            )
        if method == "clear" and not args:
            return Dct(())
        return UNK if method in _MUTATORS else container
    if t is Opn:
        if method == "clear" and not args:
            return Dct(())
        if method in ("update",) and len(args) == 1:
            return Opn(container.least)
        if method == "setdefault" and len(args) == 2:
            return Opn(max(1, container.least))
        return UNK if method in _MUTATORS else container
    return container


def _empty_text(value) -> bool:
    t = type(value)
    return (t is Str and value.s == "") or (t is str and value == "")


def _holds_a_letter(value, memo: dict) -> bool:
    """Whether every text *value* can be holds a written character (a hole can be ``""``)."""
    held = memo.get(id(value))
    if held is not None:
        return held
    t = type(value)
    if t is Str:
        found = any(ch != HOLE for ch in value.s)
    elif t is str:
        found = any(ch != HOLE for ch in value)
    elif t is Cat:
        found = any(_holds_a_letter(part, memo) for part in value.parts)
    elif t is Choice:
        found = all(_holds_a_letter(branch, memo) for branch in value.branches)
    else:
        found = False
    memo[id(value)] = found
    return found


def _compare_chain(ops, values):
    result: object = True
    for op, left, right in zip(ops, values, values[1:], strict=False):
        one = _compare_one(op, left, right)
        if one is UNK:
            return UNK
        if not one:
            return False
    return result


def _compare_one(op: ast.cmpop, left, right):
    if isinstance(op, (ast.Is, ast.IsNot)):
        if left is UNK or right is UNK:
            return UNK
        if right is None or left is None:
            same = left is None and right is None
            return same if isinstance(op, ast.Is) else not same
        return UNK
    if isinstance(op, (ast.In, ast.NotIn)):
        if (
            type(right) in (Lst, Tup, Dct)
            and not right.items
            and not (type(right) is Dct and right.maybe)
        ):
            # Nothing is in an empty container, whatever it is: ``attr in defaultVales``
            # with ``{}``.
            return isinstance(op, ast.NotIn)
        if left is UNK or not known(left):
            return UNK
        if type(right) in (Str, str) and type(left) in (Str, str):
            haystack = to_py(right)
            if HOLE in haystack:
                return UNK
            found = to_py(left) in haystack
        elif type(right) in (Lst, Tup, Dct):
            pool = right.keys() if type(right) is Dct else right.items
            wanted = to_py(left)
            if type(right) is Dct and right.maybe:
                try:
                    if wanted in right.maybe:
                        return UNK
                except TypeError:
                    pass
            if any(known(item) and to_py(item) == wanted for item in pool):
                found = True
            elif all(known(item) for item in pool):
                found = False
            else:
                return UNK
        else:
            return UNK
        return found if isinstance(op, ast.In) else not found
    if isinstance(op, (ast.Eq, ast.NotEq)) and (
        (_empty_text(left) and _holds_a_letter(right, {}))
        or (_empty_text(right) and _holds_a_letter(left, {}))
    ):
        # A text with a written letter is not "" whatever its holes are: ``if sqlP !=
        # "": sqlP += "UNION ALL "``.
        return isinstance(op, ast.NotEq)
    if type(left) is AtLeast or type(right) is AtLeast:
        return _bound_compare(op, left, right)
    if not known(left) or not known(right):
        return UNK
    a, b = to_py(left), to_py(right)
    try:
        if isinstance(op, ast.Eq):
            return a == b
        if isinstance(op, ast.NotEq):
            return a != b
        if isinstance(op, ast.Lt):
            return a < b
        if isinstance(op, ast.LtE):
            return a <= b
        if isinstance(op, ast.Gt):
            return a > b
        if isinstance(op, ast.GtE):
            return a >= b
    except TypeError:
        return UNK
    return UNK


_FLIP = {
    ast.Lt: ast.Gt,
    ast.Gt: ast.Lt,
    ast.LtE: ast.GtE,
    ast.GtE: ast.LtE,
    ast.Eq: ast.Eq,
    ast.NotEq: ast.NotEq,
}


def _bound_compare(op, left, right):
    """A comparison with a lower bound on one side (``len`` of a dict written at an unknown key)."""
    if type(right) is AtLeast and _number(left):
        flipped = _FLIP.get(type(op))
        if flipped is None:
            return UNK
        return _bound_compare(flipped(), right, left)
    if type(left) is not AtLeast or not _number(right):
        return UNK
    n, k = left.n, right
    if isinstance(op, ast.Eq):
        return False if k < n else UNK
    if isinstance(op, ast.NotEq):
        return True if k < n else UNK
    if isinstance(op, ast.Gt):
        return True if n > k else UNK
    if isinstance(op, ast.GtE):
        return True if n >= k else UNK
    if isinstance(op, ast.Lt):
        return False if n >= k else UNK
    if isinstance(op, ast.LtE):
        return False if n > k else UNK
    return UNK


def _ends(lower, upper, step) -> Optional[tuple[int, int]]:
    """``(k, 0)`` for ``[k:]`` and ``(0, k)`` for ``[:-k]``: a slice that cuts only at one end, done on a joined text
    without expanding it (``sqlRT = sqlRT[:-1]`` after two loops of appended columns)."""
    if step is not None or any(isinstance(b, bool) for b in (lower, upper)):
        return None
    if lower in (None, 0) and isinstance(upper, int) and upper < 0:
        return 0, -upper
    if upper is None and isinstance(lower, int) and lower > 0:
        return lower, 0
    return None


def drop_end(value, k: int):
    """*value* without its last *k* characters, a joined text cut branch by branch."""
    if k <= 0:
        return value
    t = type(value)
    if t is Str:
        return str_slice(value, 0, max(0, len(value.s) - k))
    if t is str:
        return value[:-k]
    if t is Choice:
        return make_choice(value.dim, [drop_end(b, k) for b in value.branches])
    if t is Cat:
        last = value.parts[-1]
        rest = make_cat(value.parts[:-1])
        if type(last) is Str:
            if len(last.s) >= k:
                return make_cat([rest, str_slice(last, 0, len(last.s) - k)])
            return drop_end(rest, k - len(last.s))
        return make_choice(
            last.dim, [drop_end(make_cat([rest, b]), k) for b in last.branches]
        )
    return UNK


def drop_start(value, k: int):
    """*value* without its first *k* characters."""
    if k <= 0:
        return value
    t = type(value)
    if t is Str:
        return str_slice(value, min(k, len(value.s)), len(value.s))
    if t is str:
        return value[k:]
    if t is Choice:
        return make_choice(value.dim, [drop_start(b, k) for b in value.branches])
    if t is Cat:
        first = value.parts[0]
        rest = make_cat(value.parts[1:])
        if type(first) is Str:
            if len(first.s) >= k:
                return make_cat([str_slice(first, k, len(first.s)), rest])
            return drop_start(rest, k - len(first.s))
        return make_choice(
            first.dim, [drop_start(make_cat([b, rest]), k) for b in first.branches]
        )
    return UNK


# ---------------------------------------------------------------- what a loop repeats

_HOLDS: dict = {}  # (value, origin) -> whether a piece written as origin is in value; emptied by clear_memo()


def _is_piece(value, origin) -> bool:
    """Whether *value* is the one piece written as *origin*: a mark a list holds as an item."""
    return (
        type(value) is Str
        and len(value.runs) == 1
        and written_at(value.runs[0][1]) == origin
    )


def holds(value, origin) -> bool:
    """Whether a piece written as *origin* is in *value*: a text, or a list (as an item)."""
    t = type(value)
    if not (t is Str or t is Cat or t is Choice or t is Lst):
        return False
    key = (value, origin)
    held = _HOLDS.get(key)
    if held is None:
        if t is Str:
            held = any(written_at(o) == origin for _piece, o in value.runs)
        elif t is Cat:
            held = any(holds(part, origin) for part in value.parts)
        elif t is Choice:
            held = any(holds(branch, origin) for branch in value.branches)
        else:
            held = any(_is_piece(item, origin) for item in value.items)
        _HOLDS[key] = held
    return held


def around(value, origin, f: Callable):
    """``f(before, after)``, what comes before and after the piece written as *origin* in *value* -- two texts, or
    two lists -- branch by branch.  A branch without the piece is kept as it is."""
    if not holds(value, origin):
        return value
    t = type(value)
    if t is Choice:
        return make_choice(
            value.dim, [around(branch, origin, f) for branch in value.branches]
        )
    if t is Str:
        at = 0
        for piece, o in value.runs:
            if written_at(o) == origin:
                return f(
                    str_slice(value, 0, at),
                    str_slice(value, at + len(piece), len(value.s)),
                )
            at += len(piece)
    if t is Cat:
        for index, part in enumerate(value.parts):
            if not holds(part, origin):
                continue
            head, tail = value.parts[:index], value.parts[index + 1 :]
            if type(part) is Choice:
                return make_choice(
                    part.dim,
                    [
                        around(make_cat([*head, branch, *tail]), origin, f)
                        for branch in part.branches
                    ],
                )
            return around(
                part,
                origin,
                lambda before, after, head=head, tail=tail: f(
                    make_cat([*head, before]), make_cat([after, *tail])
                ),
            )
    if t is Lst:
        for index, item in enumerate(value.items):
            if _is_piece(item, origin):
                return f(Lst(value.items[:index]), Lst(value.items[index + 1 :]))
    return value


def _renderings(value):
    """Each text *value* renders to with every choice made on its own, as a :class:`Str` with its pieces' origins."""
    t = type(value)
    if t is Str:
        yield value
    elif t is str:
        yield lit(value, None)
    elif t is Choice:
        for branch in value.branches:
            yield from _renderings(branch)
    elif t is Cat:
        yield from _rendered_parts(value.parts, 0, EMPTY)


def _rendered_parts(parts: tuple, index: int, done: Str):
    if index == len(parts):
        yield done
        return
    for one in _renderings(parts[index]):
        yield from _rendered_parts(parts, index + 1, str_concat(done, one))


def _common_end(texts) -> str:
    """What every text in *texts* (rendered) ends with after its last hole, from a space or a comma on: the
    separator a cut after the loop drops (``ret[:-1]``, ``sql[:-4]`` of ``AND ``), not the tail of an item the hole
    holds the rest of (``+1,`` of ``{k}={k}+1,`` is ``,``)."""
    ends = [text[text.rfind("{}") + 2 :] if "{}" in text else text for text in texts]
    if not ends:
        return ""
    common = ends[0]
    for end in ends[1:]:
        k = 0
        while k < len(common) and k < len(end) and common[-1 - k] == end[-1 - k]:
            k += 1
        common = common[len(common) - k :]
    at = next((k for k, c in enumerate(common) if c.isspace() or c == ","), len(common))
    return common[at:]


def _ending(values, text: str, origin) -> Str:
    """*text*, which the texts *values* render all end with, with the origins of one rendering's pieces there."""
    for value in values:
        for one in itertools.islice(_renderings(value), MAX_TEMPLATES):
            if one.s.endswith(text):
                return str_slice(one, len(one.s) - len(text), len(one.s))
    return lit(text, origin)


def repeated_part(loop, name: str, added: list) -> Callable:
    """How a loop of unknown length leaves *name* where it appended something: given what *name* held before, that
    and what the passes appended, as one part.  *added* is what one pass appended, on each path: texts, or lists of
    the items appended.

    The part is a hole that keeps each text a pass appends (:attr:`HoleAt.repeats`), then what all of them end
    with after their last hole -- a comma ``ret[:-1]`` cuts after the loop.  A list holds the part as one item."""
    listed = any(type(one) is Lst for one in added)
    if listed:
        pieces = [item for one in added if type(one) is Lst for item in one.items]
    else:
        pieces = [one for one in added if is_text(one)]
    texts, _no_text, capped = full_texts(pieces, cap=MAX_TEMPLATES)
    templates = () if capped else tuple(text for text in texts if text)
    if listed and pieces and not (templates or capped):  # items, none a text
        return lambda before: Lst(before.items + (UNK,))
    end = "" if capped else _common_end(templates)
    part = hole_str(HoleAt(loop, name, REPEATED, templates, end, listed))
    if end:
        part = make_cat([part, _ending(pieces, end, loop)])
    if listed:
        return lambda before: Lst(before.items + (part,))
    return lambda before: make_cat([before, part])


def _pass_texts(added: list) -> Optional[frozenset]:
    """What one pass of a loop appended, on every path, rendered -- to tell a first pass from the rest -- or None
    past :data:`MAX_TEMPLATES`."""
    out: set = set()
    for one in added:
        if type(one) is Lst:
            if not one.items:
                out.add(("list", None))
            texts, no_text, capped = full_texts(one.items, cap=MAX_TEMPLATES)
            out |= {("list", text) for text in texts}
            if no_text:
                out.add(("list", "?"))
        else:
            texts, _no_text, capped = full_texts([one], cap=MAX_TEMPLATES)
            out |= set(texts)
        if capped or len(out) > MAX_TEMPLATES:
            return None
    return frozenset(out)


def _concat(before, after):
    """*before* then *after*: two texts, or two lists."""
    if type(before) is Lst and type(after) is Lst:
        return Lst(before.items + after.items)
    return make_cat([before, after])


def _followed(before, piece: Str):
    """*before* and then *piece*: the text, or the list holding it as an item."""
    if type(before) is Lst:
        return Lst(before.items + (piece,))
    return make_cat([before, piece])


def _without(value, origin):
    """*value* without the piece written as *origin* (a mark)."""
    return around(value, origin, _concat)


def _appended_any(value, marks: tuple) -> bool:
    """Whether the text or list *value*, with no choice in it, holds anything but the pieces written as *marks*."""
    if type(value) is Lst:
        return any(not any(_is_piece(item, m) for m in marks) for item in value.items)
    if type(value) is Str:
        return any(piece and written_at(o) not in marks for piece, o in value.runs)
    return value is not EMPTY


def repeats(value) -> bool:
    """Whether the text *value* holds a part a loop of unknown length repeats (:data:`REPEATED`)."""
    if type(value) is not Str:
        return False
    for _piece, origin in value.runs:
        written = written_at(origin)
        if type(written) is HoleAt and written.reason == REPEATED:
            return True
    return False


_JOINED: dict = {}  # (value, sep) -> joined_by(value, sep); emptied by clear_memo()


def _unjoined(origin) -> bool:
    written = written_at(origin)
    return (
        type(written) is HoleAt
        and written.reason == REPEATED
        and written.listed
        and written.joiner is None
    )


def _rebased(origin, written):
    """*origin* written as *written* instead, every name that carried it kept."""
    if type(origin) is Via:
        prior = origin.prior
        return Via(
            written,
            origin.name,
            origin.at,
            None if prior is None else _rebased(prior, written),
        )
    return written


def joined_by(value, sep: str):
    """*value*, an item ``sep.join`` joins, with each part in it that a loop of unknown length appended to the list
    as items saying *sep* joins them (:attr:`HoleAt.joiner`): one pass's text is one item."""
    t = type(value)
    if not (t is Str or t is Cat or t is Choice):
        return value
    key = (value, sep)
    held = _JOINED.get(key)
    if held is None:
        if t is Str:
            if any(_unjoined(origin) for _piece, origin in value.runs):
                held = Str(
                    value.s,
                    tuple(
                        (
                            piece,
                            _rebased(origin, written_at(origin)._replace(joiner=sep))
                            if _unjoined(origin)
                            else origin,
                        )
                        for piece, origin in value.runs
                    ),
                )
            else:
                held = value
        elif t is Choice:
            held = make_choice(
                value.dim, [joined_by(branch, sep) for branch in value.branches]
            )
        else:
            held = make_cat([joined_by(part, sep) for part in value.parts])
        _JOINED[key] = held
    return held


# ================================================================ expansion


def _free(value, memo: dict, cap: int) -> set:
    """Every text *value* renders to if each occurrence of a dimension chose on its own (``None`` for no text): a
    superset of :func:`full_texts`, cheap because it is a set per node, not an assignment per path."""
    held = memo.get(id(value))
    if held is not None:
        return held[1]
    t = type(value)
    if t is Str:
        out = {value.rendered()}
    elif t is str:
        out = {value.replace(HOLE, "{}")}
    elif t is Choice:
        out = set()
        for branch in value.branches:
            out |= _free(branch, memo, cap)
            if len(out) > cap:
                raise Overflow("cap")
    elif t is Cat:
        out = {""}
        for part in value.parts:
            texts = _free(part, memo, cap)
            out = {a + b for a in out for b in texts if b is not None}
            if len(out) > cap:
                raise Overflow("cap")
    else:
        out = {None}
    memo[id(value)] = (value, out)
    return out


def _match(
    value, text: str, position: int, assign: dict, budget: Optional[list] = None
):
    """``(end, assignment)`` for each way *value* renders ``text[position:end]`` with every dimension choosing once.
    *budget* (``[steps]``) bounds the search: past it, Overflow."""
    if budget is not None:
        budget[0] -= 1
        if budget[0] < 0:
            raise Overflow("match budget")
    t = type(value)
    if t is Str or t is str:
        piece = value.rendered() if t is Str else value.replace(HOLE, "{}")
        if text.startswith(piece, position):
            yield position + len(piece), assign
        return
    if t is Choice:
        index = assign.get(value.dim)
        if index is not None:
            yield from _match(value.branches[index], text, position, assign, budget)
            return
        for index, branch in enumerate(value.branches):
            chosen = dict(assign)
            chosen[value.dim] = index
            yield from _match(branch, text, position, chosen, budget)
        return
    if t is Cat:
        yield from _match_parts(value.parts, 0, text, position, assign, budget)


def _match_parts(
    parts: tuple,
    index: int,
    text: str,
    position: int,
    assign: dict,
    budget: Optional[list] = None,
):
    if index == len(parts):
        yield position, assign
        return
    for end, chosen in _match(parts[index], text, position, assign, budget):
        yield from _match_parts(parts, index + 1, text, end, chosen, budget)


def _match_prefix(value, text: str, position: int = 0, assign: Optional[dict] = None):
    """``(end, assignment)`` for each way *value* renders a text that begins with ``text[position:]`` (comparison only:
    the reader's variant of a name against the core's text at the call, which goes on with ``+ comment``)."""
    assign = {} if assign is None else assign
    if position >= len(text):
        yield position, assign
        return
    t = type(value)
    if t is Str or t is str:
        piece = value.rendered() if t is Str else value.replace(HOLE, "{}")
        rest = text[position:]
        if rest.startswith(piece):
            yield position + len(piece), assign
        elif piece.startswith(rest):
            yield len(text), assign
        return
    if t is Choice:
        index = assign.get(value.dim)
        if index is not None:
            yield from _match_prefix(value.branches[index], text, position, assign)
            return
        for index, branch in enumerate(value.branches):
            chosen = dict(assign)
            chosen[value.dim] = index
            yield from _match_prefix(branch, text, position, chosen)
        return
    if t is Cat:
        yield from _match_prefix_parts(value.parts, 0, text, position, assign)


def _match_prefix_parts(
    parts: tuple, index: int, text: str, position: int, assign: dict
):
    if index == len(parts) or position >= len(text):
        yield position, assign
        return
    for end, chosen in _match_prefix(parts[index], text, position, assign):
        yield from _match_prefix_parts(parts, index + 1, text, end, chosen)


def renders(value, text: str) -> bool:
    """Whether some assignment of *value*'s dimensions renders *text*."""
    return any(end == len(text) for end, _ in _match(value, text, 0, {}))


def full_texts(values, cap: int = FULL_CAP) -> tuple[dict, bool, bool]:
    """Every text *values* can be, rendered (``{}`` for a hole): ``(texts, some value is no text, capped)``.

    The texts with each dimension choosing on its own, kept where one assignment of every dimension renders them."""
    texts: dict = {}
    no_text = False
    memo: dict = {}
    for value in values:
        try:
            free = _free(value, memo, cap)
        except Overflow:
            return texts, no_text, True
        joint = not dims(value)
        for text in sorted(free, key=lambda x: (x is None, x or "")):
            if text is None:
                no_text = True
            elif joint or renders(value, text):
                texts[text] = None
        if len(texts) > cap:
            return texts, no_text, True
    return texts, no_text, False


class Rendered(NamedTuple):
    """One statement handed to a consumer: its text, where each stretch was written, and its holes."""

    text: str
    spans: tuple  # (statement or Entry, start, end), offsets in *text*
    holes: tuple  # (offset, HoleAt)
    carried: tuple = ()  # (statement that put the stretch into its name, else where it was written, start, end)
    # (start, end, way): each stretch with its way to the run, from where it was written
    # (a statement or an Entry) through each statement that put it into a name, each
    # callee ``return`` and each call it came back from (:func:`way_of`).
    pieces: tuple = ()


def way_of(origin) -> tuple:
    """Where a piece was written, then every statement, ``return`` and call that carried it on, in that order."""
    ats = []
    while type(origin) is Via:
        ats.append(origin.at)
        if origin.prior is None:
            origin = origin.written
            break
        origin = origin.prior
    way = [origin.stmt if type(origin) is HoleAt else origin]
    for at in reversed(ats):
        # ``sql += "AND ..."`` writes the piece and puts it into ``sql`` at once: one
        # step, not two.
        if at is not way[-1]:
            way.append(at)
    return tuple(way)


def _same_origin(a, b) -> bool:
    return a is b or (type(a) is Entry and a == b)


def rendered(runs: list) -> Rendered:
    parts = []
    spans: list = []
    holes: list = []
    position = 0
    carried_spans: list = []
    pieces: list = []
    for piece, origin in runs:
        text = piece.replace(HOLE, "{}")
        written = written_at(origin)
        where = written.stmt if type(written) is HoleAt else written
        if type(written) is HoleAt:
            holes.append((position, written))
        if spans and _same_origin(spans[-1][0], where) and spans[-1][2] == position:
            spans[-1] = (where, spans[-1][1], position + len(text))
        else:
            spans.append((where, position, position + len(text)))
        into = _into(origin, where)
        if (
            carried_spans
            and _same_origin(carried_spans[-1][0], into)
            and carried_spans[-1][2] == position
        ):
            carried_spans[-1] = (into, carried_spans[-1][1], position + len(text))
        else:
            carried_spans.append((into, position, position + len(text)))
        way = way_of(origin)
        if pieces and pieces[-1][2] == way and pieces[-1][1] == position:
            pieces[-1] = (pieces[-1][0], position + len(text), way)
        else:
            pieces.append((position, position + len(text), way))
        position += len(text)
        parts.append(text)
    return Rendered(
        "".join(parts),
        tuple(spans),
        tuple(holes),
        tuple(carried_spans),
        tuple(pieces),
    )


def _into(origin, where):
    """The statement that last put a piece into a name -- past the ``return`` and the call a callee's piece came back
    through, which put it into no name -- else where it was written."""
    while type(origin) is Via and origin.name in (RETURNED, CALLED):
        origin = origin.prior if origin.prior is not None else origin.written
    return origin.at if type(origin) is Via else where


def _pick(value, assign: dict, covered: set, runs: list, fresh: list) -> bool:
    t = type(value)
    if t is Str:
        runs.extend(value.runs)
        return True
    if t is str:
        runs.append((value, None))
        return True
    if t is Cat:
        ok = True
        for part in value.parts:
            ok = _pick(part, assign, covered, runs, fresh) and ok
        return ok
    if t is Choice:
        index = assign.get(value.dim)
        if index is None:
            index = _choose(value, covered)
            assign[value.dim] = index
            if (value.dim, index) not in covered:
                covered.add((value.dim, index))
                fresh[0] += 1
        return _pick(value.branches[index], assign, covered, runs, fresh)
    return False


def _choose(choice: Choice, covered: set) -> int:
    dim = choice.dim
    for index in range(len(choice.branches)):
        if (dim, index) not in covered:
            return index
    for index, branch in enumerate(choice.branches):
        if any((d, j) not in covered for d in dims(branch) for j in range(d.n)):
            return index
    return 0


def cover(
    values, cap: int = COVER_CAP, picks: Optional[dict] = None
) -> tuple[list[Rendered], bool]:
    """Statements in which every branch of every choice in *values* shows up at least once: ``(statements, complete)``.

    Not every combination: a consumer reading tables, columns and binds needs every piece, and the combinations
    of independent pieces are what made getTasksWithCriteria_JEDI's tens of thousands.  *picks*, where given, gets
    each statement's text -> (the value it was drawn from, the choice each dimension made), so a name inside the
    value can be drawn the same way (:attr:`sql.Execution.head`)."""
    covered: set = set()
    out: list[Rendered] = []
    group: list[int] = []
    seen: set = set()
    complete = True
    for index, value in enumerate(values):
        rounds = 0
        while True:
            assign: dict = {}
            runs: list = []
            fresh = [0]
            ok = _pick(value, assign, covered, runs, fresh)
            rounds += 1
            if ok:
                one = rendered(runs)
                if one.text not in seen:
                    seen.add(one.text)
                    out.append(one)
                    group.append(index)
                    if picks is not None:
                        picks[one.text] = (value, dict(assign))
            if len(out) >= cap or (fresh[0] == 0 and rounds > 1) or not dims(value):
                break
            if fresh[0] == 0:
                break
        if any((d, j) not in covered for d in dims(value) for j in range(d.n)):
            complete = False
        if len(out) >= cap:
            complete = complete and all(
                not any((d, j) not in covered for d in dims(v) for j in range(d.n))
                for v in values
            )
            break
    # The values in the order the paths reached the call (a loop's passes in turn), each
    # value's statements in the order of their texts, not of the picks: the picks follow
    # the dimensions' numbers, which a run's history moves.
    order = sorted(range(len(out)), key=lambda i: (group[i], out[i].text))
    return [out[i] for i in order], complete


def drawn(value, assign: dict) -> Optional[Rendered]:
    """*value* with each dimension in *assign* taking the branch it says (the others their first): a name inside a
    covering statement, drawn as that statement was."""
    runs: list = []
    if not _pick(value, dict(assign), set(), runs, [0]):
        return None
    return rendered(runs)


def rendered_as(values, text: str, limit: int = 4) -> list[Rendered]:
    """*values* drawn so that they render exactly *text*, up to *limit* ways: the variant a consumer names by its text
    (sqlwrite cuts a run's head, then the piece a copied name holds, each by its text).  A bounded search: a value too
    large to list, or a match past its bound, gives fewer."""
    out: list[Rendered] = []
    for value in values:
        try:
            free = _free(value, {}, 20000)
        except Overflow:
            continue
        if text not in free:
            continue
        try:
            for end, assign in _match(value, text, 0, {}, [200000]):
                if end != len(text):
                    continue
                runs: list = []
                if _pick(value, dict(assign), set(), runs, [0]):
                    out.append(rendered(runs))
                    if len(out) >= limit:
                        return out
        except Overflow:
            continue
    return out


# ================================================================ readings


def _add_stats(total: dict, one: dict) -> None:
    for key, value in one.items():
        if key == "max_paths":
            total[key] = max(total.get(key, 0), value)
        else:
            total[key] = total.get(key, 0) + value


class Reading:
    """Every runner call of a function: one run over the whole function, or each call by its own run past WHOLE_WORK."""

    def __init__(
        self,
        func,
        constants: Optional[dict] = None,
        resolver: Optional[Callable] = None,
        line: Optional[int] = None,
        conventions: Optional[dict] = None,
    ):
        self.values: dict[int, dict] = {}
        self.names_at: dict[int, dict] = {}
        self.reached: set[int] = set()
        self.overflow_at: dict[int, str] = {}
        self.widened_at: set[int] = set()
        self.defaults_used: set[str] = set()
        self.stats: dict = {}
        self.mode = "function"
        self._texts = None
        lines = (
            [line]
            if line is not None
            else sorted({call.lineno for call in runner_calls(func)})
        )
        whole = (
            Ref(
                func,
                constants,
                None,
                resolver=resolver,
                conventions=conventions,
                work_cap=WHOLE_WORK,
            ).run()
            if len(lines) > 1
            else None
        )
        if whole is not None:
            _add_stats(self.stats, whole.stats)
        if whole is not None and not whole.overflow:
            # Widened or not, it stands: widening keeps the names alike on every path
            # (updateJob@538 reads one job table for inJobsDefined=False only because
            # inJobsDefined survives it).
            self.values, self.reached, self.defaults_used = (
                whole.values,
                whole.reached,
                whole.defaults_used,
            )
            self.names_at = whole.names_at
            self.widened_at |= whole.widened_lines
            return
        # Past the budget: each call by its own run, which tracks only the names that
        # call needs.
        self.mode = "per call"
        for one_line in lines:
            one = Ref(
                func,
                constants,
                one_line,
                resolver=resolver,
                conventions=conventions,
                work_cap=WORK_BUDGET,
            ).run()
            _add_stats(self.stats, one.stats)
            if one.overflow:
                self.overflow_at[one_line] = one.overflow
            self.widened_at |= one.widened_lines
            self.values.update(one.values)
            self.names_at.update(one.names_at)
            self.reached |= one.reached
            self.defaults_used |= one.defaults_used

    def _expand_all(self):
        texts, no_text, capped = {}, {}, set()
        for line, held in self.values.items():
            got, none, cut = full_texts(list(held))
            texts[line] = got
            if none:
                no_text[line] = 1
            if cut:
                capped.add(line)
        self._texts = (texts, no_text, capped)

    @property
    def texts(self) -> dict:
        if self._texts is None:
            self._expand_all()
        return self._texts[0]

    @property
    def no_text(self) -> dict:
        if self._texts is None:
            self._expand_all()
        return self._texts[1]

    @property
    def full_capped(self) -> set:
        if self._texts is None:
            self._expand_all()
        return self._texts[2]

    def cover(self, line: int, cap: int = COVER_CAP) -> tuple[list[Rendered], bool]:
        return cover(list(self.values.get(line, {})), cap)

    def heads(
        self, line: int, name: str, cap: int = COVER_CAP
    ) -> list[tuple[Rendered, list[Rendered]]]:
        """Each covering statement at *line*, with what the text name *name* in it held there, drawn with the
        statement's own choices (:attr:`sql.Execution.head`)."""
        picks: dict = {}
        shown, _complete = cover(list(self.values.get(line, {})), cap, picks)
        out = []
        for one in shown:
            value, assign = picks[one.text]
            held = self.names_at.get(line, {}).get(value, {}).get(name, ())
            out.append(
                (one, [d for d in (drawn(h, assign) for h in held) if d is not None])
            )
        return out

    def named(self, line: int, name: str, text: Optional[str] = None) -> list[Rendered]:
        """What the text name *name* held at the call at *line*: the ways it renders *text*, or without *text* each
        covering statement's drawing of it (:meth:`heads`), one per text."""
        held: dict = {}
        for by_name in self.names_at.get(line, {}).values():
            held.update(dict.fromkeys(by_name.get(name, ())))
        if text is not None:
            return rendered_as(list(held), text)
        out: dict = {}
        for _one, drawings in self.heads(line, name):
            for one in drawings:
                out.setdefault(one.text, one)
        return [out[t] for t in sorted(out)]

    @property
    def overflow(self) -> Optional[str]:
        return next(iter(self.overflow_at.values()), None)


def evaluate(
    func,
    constants: Optional[dict] = None,
    resolver: Optional[Callable] = None,
    conventions: Optional[dict] = None,
) -> Reading:
    return Reading(func, constants, resolver, conventions=conventions)


def assembled(
    func,
    constants: Optional[dict] = None,
    resolver: Optional[Callable] = None,
    conventions: Optional[dict] = None,
) -> "Ref":
    """What each name *func* builds a statement into holds where the code uses it or where *func* ends (mode
    ``uses``): every statement it assembles, run or not.  One run within :data:`WHOLE_WORK`; past it, ``overflow``
    says so and what was held up to there is all there is."""
    return Ref(
        func,
        constants,
        None,
        resolver=resolver,
        conventions=conventions,
        mode="uses",
        work_cap=WHOLE_WORK,
    ).run()


def evaluate_line(
    func, constants: Optional[dict], line: int, resolver: Optional[Callable] = None
) -> Reading:
    return Reading(func, constants, resolver, line=line)
