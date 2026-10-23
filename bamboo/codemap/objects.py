"""Which columns of an object a closure can assign, read from the record -- the object followed, not the closure.

The record is the build's, per function (:mod:`bamboo.codemap.panda.objects`):
the columns a function assigns on a path, the names that hold what a path
holds, and the paths its calls hand on.  :func:`follow` starts at one path of
one function -- ``self.job`` in the adder's ``process_job_report`` -- and goes
where that object goes:

* within a function, to every name holding it: an alias, a loop over a
  container holding it, a container it is put in, a field it is stored in --
  and back from a name only by the binding it came in by (:func:`_held_steps`);
* into a call, as the parameter that takes it (the receiver as ``self``),
  an element of a list or tuple literal handed as that parameter's element;
* into the call a name holding it was bound from, through what that call
  returns (a constructor's ``self``): the object came out of it;
* back out to the calls it came in by, when the function it came into keeps
  it on its own ``self`` -- ``AdderPluginBase.__init__``'s ``self.job = job``
  makes the plugin the caller builds hold the job -- or returns it.

Each (function, path) is read once, and again only when a call out of it
hands back a path that holds the object.  Where the object ends up, seen
from outside the function -- on its ``self``, or returned -- is that state's
summary, and is handed to every call that brought the object in, whatever
call brought that one in: the summary does not depend on the caller.

Where each assigned value comes from is not read.  A function that builds
the object from the database -- ``peekJob`` setting ``jobParameters`` from
``jobParamsTable`` -- assigns it as any other does, and is said so.

It answers what is assigned on the object there, and the holes it reaches: a
call it is handed to that the record could not place, or one that does not
say which parameter takes it.  A hole the object does not reach is not this
object's, which is why this answers where the closure's own reach does not.

What it rests on, beyond :data:`bamboo.codemap.reach.PREMISES`, is in
:data:`PREMISES`.  A local name is read at each use by the bindings that
reach it, but a field and what a container holds without the statements'
order, and what calls hand is not read (a branch a constant rules out is
read as run), which can only say an object reaches more than it does.  The other way: what the
record does not read (:mod:`bamboo.codemap.panda.objects` lists it) the
object is not followed through, and it can reach more than this says.  Nor
is it followed past its caps -- a path longer than :data:`MAX_PARTS` parts,
which is what stops a cycle (a job's file's owner is the job), more than
:data:`MAX_HELD` paths in one function, :data:`MAX_STATES` states --, each
said where it is met (:attr:`ObjectReach.capped`, :attr:`ObjectReach.cut`);
:func:`may_change` names ``follow_capped`` where one was.

A path's parts are matched one by one (:func:`_under`): ``[]``, an element
at a position not known, matches a tuple's ``[i]`` and a literal key's
``['k']`` as well as itself, so what was put under ``varMap[':PandaID']``
is read at ``varMap[':PandaID']`` and ``varMap[]``, not at
``varMap[':status']``.
"""

from __future__ import annotations

import re
from collections import defaultdict
from functools import lru_cache
from typing import NamedTuple, Optional, Sequence

from bamboo.codemap.models import (
    BIND_ALIAS,
    BIND_ELEMENT,
    BIND_FILL,
    BIND_RESULT,
    BIND_RETURN,
    ENTRY_LINE,
    EXIT_LINE,
    MAP_KEY,
    CallBinding,
    DbWrite,
    EntityNode,
    LoopLines,
    ReachNode,
    WrittenValue,
)

#: How many (function, path) states one answer reads before it stops and says so.
MAX_STATES = 5000
#: How many paths one function can hold the object by before the rest are dropped and said.
MAX_HELD = 200
#: The longest path followed, in parts after the root (:func:`_tokens`); where one longer is not, the answer says
#: so (:attr:`ObjectReach.capped`).  The record keeps every path; this is how far one is read.
MAX_PARTS = 6

PREMISES: dict[str, str] = {
    "outside_assigns_nothing": (
        "Code outside the corpus assigns no attribute of an object of the corpus it is handed, other than through "
        "the corpus's own methods -- which it calls only as reach's outside_calls_back_only_handed allows."
    ),
    "bindings_without_order": (
        "A local name bound at two or more statements -- a parameter's entry is one -- is a name for each, and a "
        "use is read as the ones that reach it (x += y adds to x and kills nothing); a field and what a container "
        "holds are read without the statements' order: bound to what a path holds anywhere in a function, they "
        "hold it everywhere in it."
    ),
    "handed_values_not_read": (
        "What a call hands its callee is not read: a statement the values it hands would rule out is read as one "
        "that runs, with its bindings and assignments."
    ),
    "literal_keys_apart": (
        "A literal str key names one entry of a container: what is put under one key is read under that key, or "
        "under one not spelled as a literal.  The corpus's classes with their own item access (MapWithLock, "
        "CachedObject) hand it to a dict."
    ),
    "rows_hold_their_where": (
        "A row a query returns holds, in each column its own WHERE sets equal to one bind, what the bind was "
        "filled with; a spec reading the row (pack) holds it there, as if assigned by hand."
    ),
    "follow_capped": (
        f"The object is followed to paths of at most {MAX_PARTS} parts, {MAX_HELD} paths in one function and "
        f"{MAX_STATES} states, and no further, so it can reach more than it says.  The parts cap is what stops a "
        "cycle: a job's file's owner is the job (JobSpec.addFile), and job.Files[]._owner.Files[]._owner names it "
        "again."
    ),
}

#: Why a hole the record keeps no reason for is one.
NO_PARAMETER = "the call does not say which parameter takes it (*args, **kwargs)"
STATES_SPENT = f"stopped after {MAX_STATES} (function, path) states"
HELD_SPENT = f"the object is held by more than {MAX_HELD} paths in one function; the rest are not followed"
PARTS_SPENT = f"a path longer than {MAX_PARTS} parts is not followed: {{path}}"

RETURN = "<return>"

#: The bindings a held path is stepped along (:func:`_held_steps`); a result's is read by :func:`follow`.
STEPPED = (BIND_ALIAS, BIND_ELEMENT, BIND_FILL, BIND_RETURN)

#: How :func:`_held_steps` steps along a binding: from what is bound to the name, or back from the name.
_FORWARD, _BACK = "forward", "back"

_State = tuple[str, str]


class Site(NamedTuple):
    owner: str
    line: int
    how: str


class ObjectReach(NamedTuple):
    """What :func:`follow` answers."""

    #: attribute -> where it is assigned on the object, and how
    assigned: dict[str, list[Site]]
    #: where the object reaches a call the record could not follow, and why
    holes: list[Site]
    #: where the start's object is returned or kept on self, with nothing read further up
    escapes: list[Site]
    #: where a path the object is held by is bound to what a call returns: where it, or another object in
    #: its place, comes from (how = the path)
    came_from: list[Site]
    functions: list[str]
    #: owner -> every path of that function the object is held by
    held: dict[str, list[str]]
    states: int
    #: why the follow stopped short -- :data:`STATES_SPENT`, :data:`HELD_SPENT` --; empty where it did not
    cut: str
    #: where a path longer than :data:`MAX_PARTS` is not followed: the line that would hold the object by it, and
    #: the path (:data:`PARTS_SPENT`)
    capped: list[Site]
    #: state -- a function and the path it holds the object by -- -> how the object first came into it: the
    #: caller's state, the line of the call, and the caller's path that was handed (:func:`way_to` reads a way back
    #: to the start from it)
    entered: dict[_State, tuple[_State, int, str]]
    #: state -> path -> the call that handed the object back to that state as the path: its line, the callee (on
    #: its ``self``, or returned), and the state's path that went into the call
    returned: dict[_State, dict[str, tuple[int, str, str]]]


class _Link(NamedTuple):
    """A call that brought the object into a state: the caller's state, the line, the function called, the
    caller's path it went in under (a call binding's), and the caller's path that held the object."""

    caller: _State
    line: int
    to: str
    via: str = ""
    handed: str = ""


def follow(rows: dict[str, ReachNode], owner: str, path: str) -> ObjectReach:
    """What is assigned on the object *owner* names *path*, and the holes it reaches; see the module docstring."""
    assigned: dict[str, dict[Site, None]] = defaultdict(dict)
    holes: dict[Site, None] = {}
    came_from: dict[Site, None] = {}
    capped: dict[Site, None] = {}
    functions: dict[str, None] = {}
    held_in: dict[str, set[str]] = defaultdict(set)
    extra: dict[_State, set[str]] = defaultdict(set)
    outputs: dict[_State, set[str]] = defaultdict(set)
    links: dict[_State, set[_Link]] = defaultdict(set)
    read_with: dict[
        _State, int
    ] = {}  # how many paths calls had handed each state when it was read
    root: _State = (owner, path)
    work: list[_State] = [root]
    cut = ""
    entered: dict[_State, tuple[_State, int, str]] = {}
    returned: dict[_State, dict[str, tuple[int, str, str]]] = defaultdict(dict)

    def into(child: _State, link: _Link) -> None:
        """Bring the object into *child* by *link*; what *child* already says of it goes back at once."""
        if _parts(child[1]) > MAX_PARTS:
            capped[
                Site(
                    link.caller[0],
                    link.line,
                    PARTS_SPENT.format(path=f"{child[1]} in {child[0]}"),
                )
            ] = None
            return
        if child != root:
            entered.setdefault(child, (link.caller, link.line, link.handed))
        if link not in links[child]:
            links[child].add(link)
            if _hand_back(rows, link, outputs[child], extra, returned, capped):
                work.append(link.caller)
        if child not in read_with:
            work.append(child)

    while work:
        state = work.pop()
        if read_with.get(state) == len(extra[state]):
            continue
        if len(read_with) >= MAX_STATES:
            cut = STATES_SPENT
            break
        read_with[state] = len(extra[state])
        at, start = state
        row = rows.get(at)
        if row is None:
            continue
        functions[at] = None
        steps, spent = _held_steps(row, {start}, extra[state], capped)
        held = set(steps)
        rooted: dict[str, list[str]] = defaultdict(
            list
        )  # held by root, each sorted (_under)
        for one in sorted(held):
            rooted[_root(one)].append(one)
        held_in[at] |= held
        if spent:
            cut = cut or HELD_SPENT
        for one in row.assigns:
            if _holds(held, one.path):
                for attr in one.attrs:
                    assigned[attr][Site(at, one.line, one.how)] = None
        for binding in row.bindings:
            if binding.how != BIND_RESULT:
                continue
            for one in rooted.get(_root(binding.name), ()):
                rest = _under(binding.name, one)
                if rest is None or _came_forward(steps[one]):
                    # What came into the name by a binding is not what the call bound to it returns.
                    continue
                # The object held here came out of the call: in from what it returns.
                came_from[Site(at, binding.line, binding.name)] = None
                for to in binding.to:
                    if to.rsplit(".", 1)[-1] == "__init__":
                        if binding.path:
                            continue  # a part of a new object is not the object
                        taken = "self" + rest
                    else:
                        taken = RETURN + binding.path + rest
                    into((to, taken), _Link(state, binding.line, to, handed=one))
        for call in row.call_bindings:
            for one in rooted.get(_root(call.path), ()):
                rest = _under(call.path, one)
                if rest is None:
                    continue
                if not call.to:
                    holes[Site(at, call.line, call.why or _hole_at(row, call.line))] = (
                        None
                    )
                elif not call.parameter or call.parameter.startswith("*"):
                    holes[Site(at, call.line, NO_PARAMETER)] = None
                else:
                    taken = call.parameter + call.at + rest
                    into(
                        (call.to, taken),
                        _Link(state, call.line, call.to, call.path, one),
                    )
        # Where the object ends up, seen from outside: on self, or returned.  New ends go back to every call
        # that brought it in.
        ends = {
            one
            for one in held
            if one.startswith(RETURN) or _under("self", one) is not None
        }
        if not ends <= outputs[state]:
            outputs[state] |= ends
            for link in sorted(links[state]):
                if _hand_back(rows, link, outputs[state], extra, returned, capped):
                    work.append(link.caller)
    escapes: dict[Site, None] = {}
    for end in sorted(outputs[root]):
        if end.startswith(RETURN):
            escapes[Site(owner, 0, "returned")] = None
        elif _under("self", path) is None:
            escapes[Site(owner, 0, f"kept on self as {end}")] = None
    return ObjectReach(
        assigned={attr: list(sites) for attr, sites in assigned.items()},
        holes=list(holes),
        escapes=list(escapes),
        came_from=list(came_from),
        functions=list(functions),
        held={owner: sorted(paths) for owner, paths in held_in.items()},
        states=len(read_with),
        cut=cut,
        capped=list(capped),
        entered=entered,
        returned=dict(returned),
    )


def way_to(reach: ObjectReach, state: _State) -> list[tuple[_State, int, _State, str]]:
    """One way the object went from the start into *state*, in order: per call, ``(the caller's state, the line,
    the state it brought the object into, the caller's path handed)``.

    Each state's first entry is kept, so this is the way :func:`follow`
    first came into that state by -- one of the ways, not every one.  It
    goes by states, not functions: ``updateJobs`` hands ``archiveJob`` the
    job and ``async_params`` = ``{"PandaID": job.PandaID, ...}`` at one
    call, and what ``archiveJob`` does with the job is not what it does with
    the dict.  An entry comes from a state reached before it, so the way
    ends at the start.
    """
    way: list[tuple[_State, int, _State, str]] = []
    seen = {state}
    while state in reach.entered:
        caller, line, handed = reach.entered[state]
        way.append((caller, line, state, handed))
        if caller in seen:
            break
        seen.add(caller)
        state = caller
    return list(reversed(way))


def _hand_back(
    rows: dict[str, ReachNode],
    link: _Link,
    ends: set[str],
    extra: dict[_State, set[str]],
    returned: dict[_State, dict[str, tuple[int, str, str]]],
    capped: dict[Site, None],
) -> bool:
    """Put where the object ends up in the callee into the caller's state; whether that added a path.  A path
    longer than :data:`MAX_PARTS` is not put, and goes into *capped*."""
    caller = rows.get(link.caller[0])
    if caller is None or not ends:
        return False
    added = False
    held = extra[link.caller]
    for end in sorted(ends):
        if end.startswith(RETURN):
            # ``x = f()[0]``: x is what is at [0] of what f returns, and the end is below that, or not x's.
            names = [
                result.name + below
                for result in caller.bindings
                if result.how == BIND_RESULT
                and result.line == link.line
                and link.to in result.to
                and (below := _under(RETURN + result.path, end)) is not None
            ]
        else:
            below = end[len("self") :]
            receivers = _receivers_of(caller, link)
            if link.via in receivers:
                # It went in as the receiver: what the callee keeps on self comes back to that one.
                receivers = [link.via]
            names = [receiver + below for receiver in receivers]
        for name in names:
            if name == link.caller[1] or name in held:
                continue
            if _parts(name) > MAX_PARTS:
                capped[
                    Site(link.caller[0], link.line, PARTS_SPENT.format(path=name))
                ] = None
                continue
            held.add(name)
            returned[link.caller].setdefault(name, (link.line, link.to, link.handed))
            added = True
    return added


def _held(row: ReachNode, start: set[str]) -> tuple[set[str], bool]:
    """Every path of *row*'s function that holds what *start* holds, by its bindings; see :func:`_held_steps`."""
    steps, spent = _held_steps(row, start)
    return set(steps), spent


def _held_steps(
    row: ReachNode,
    start: set[str],
    returned: set[str] = frozenset(),
    capped: Optional[dict[Site, None]] = None,
) -> tuple[dict[str, Optional[tuple[str, int, str]]], bool]:
    """Every path of *row*'s function that holds what *start* holds, each with the step it was first reached by.

    A step is ``(the path it came from, the binding's line, how)``; a start
    path has none.  A binding is stepped along both ways: forward, from what
    is bound to the name -- ``x = p``, ``for x in p``, ``x.append(p)``,
    ``return p`` -- and back, from the name to what is bound to it.

    **What came into a name by one binding is not read back by another.**  A
    name bound in several places holds what each binding gives it, and those
    are different objects: ``tag_job = self.jobs[0]`` and ``tag_job =
    self.jumbo_jobs[0]`` in two branches, or ``archiveJob``'s
    ``updatedJobList.append(job)`` and ``.append(dJob)``.  So a path reached
    forward is not stepped back: ``self.jobs[]`` reaches ``tag_job`` and not
    ``self.jumbo_jobs[]``.  Back through the binding it came by it only comes
    to where it came from -- read part by part, wider: ``any_one =
    var_map[k]``, reached from ``var_map[':PandaID']``, read back is
    ``var_map[]``, every key.
    A start path, or one reached back, steps back through every binding: it
    can be what any of them gives.  :func:`follow` keeps the same rule for a
    name a call's result is bound to: ``makeFakeCoJumbo``'s ``varMap`` holds
    the old job's ``PandaID`` it was filled with, not what the new job's
    ``valuesMap()`` returns when the name is bound again.

    *returned* are paths calls handed the object back by (kept on the
    callee's ``self``, or returned).  One the bindings already reach from
    *start* keeps how it was reached: ``TaskRefiner``'s
    ``tmpTaskSpec.jediTaskID``, reached through ``tmpTaskSpec =
    impl.taskSpec`` and handed to ``tmpTaskSpec.setErrDiag()``, is not read
    back to the ``jediTaskID`` the other branch gives it.  The others join
    as starts once the bindings are read.
    """
    held: dict[str, Optional[tuple[str, int, str]]] = {}
    # held's paths in the order reached, and where each root's are in it: a path is below another only with
    # its root (_under), so a binding is read against the held paths of its two sides' roots, in that order.
    order: list[str] = []
    by_root: dict[str, list[int]] = defaultdict(list)

    def reach(one: str, step: Optional[tuple[str, int, str]]) -> None:
        held[one] = step
        by_root[_root(one)].append(len(order))
        order.append(one)

    for one in sorted(start):
        reach(one, None)
    pending = sorted(set(returned) - set(start))
    entered: dict[str, set[int]] = defaultdict(
        set
    )  # path -> the bindings it was reached forward by
    changed = True
    while changed:
        changed = False
        for index, binding in enumerate(row.bindings):
            if binding.how not in STEPPED:
                continue
            # The name says where it holds the path: ``updatedJobList[]`` for ``updatedJobList.append(job)``.
            target = binding.name
            sides = {_root(binding.path), _root(target)}
            for one in [
                order[at] for at in sorted(at for root in sides for at in by_root[root])
            ]:
                for frm, to, way in (
                    (binding.path, target, _FORWARD),
                    (target, binding.path, _BACK),
                ):
                    rest = _under(frm, one)
                    if rest is None:
                        continue
                    if way == _BACK and entered.get(one):
                        continue
                    taken = to + rest
                    if way == _FORWARD and (taken in entered or taken not in held):
                        entered[taken].add(index)
                    if taken in held:
                        continue
                    if _parts(taken) > MAX_PARTS:
                        if capped is not None:
                            capped[
                                Site(
                                    row.owner,
                                    binding.line,
                                    PARTS_SPENT.format(path=taken),
                                )
                            ] = None
                        continue
                    if len(held) >= MAX_HELD:
                        return held, True
                    how = binding.how if way == _FORWARD else f"{binding.how} back"
                    reach(taken, (one, binding.line, how))
                    changed = True
        if not changed and pending:
            for one in pending:
                if one not in held:
                    reach(one, None)
                    changed = True
            pending = []
    return held, False


def _came_forward(step: Optional[tuple[str, int, str]]) -> bool:
    """Whether a path was reached forward through a binding: not a start, and not reached back."""
    return step is not None and not step[2].endswith(" back")


def steps_to(row: ReachNode, start: str, path: str) -> list[tuple[int, str, str]]:
    """The bindings of *row*'s function that take what *start* holds to *path*: ``(line, how, path)`` each, in order.

    Empty where *path* is *start*, or is not reached from it.
    """
    held, _spent = _held_steps(row, {start})
    way: list[tuple[int, str, str]] = []
    seen = {path}
    while (step := held.get(path)) is not None:
        came, line, how = step
        way.append((line, how, path))
        if came in seen:
            return []
        seen.add(came)
        path = came
    return list(reversed(way)) if path == start else []


def _under(base: str, path: str) -> Optional[str]:
    """What *path* goes on with below *base* -- ``.job`` for ``self`` and ``self.job`` -- or ``None`` where it is not below it.

    Part by part, where ``[]`` -- an element, at a position not known --
    matches ``[i]`` and ``['k']`` as well as itself: ``d['a'].x`` is below
    ``d[]`` and ``d[].x`` below ``d['a']``, and neither is below ``d['b']``.
    """
    if path == base:
        return ""
    if _plain(base) and _plain(path):
        if path.startswith(base) and path[len(base)] in ".[":
            return path[len(base) :]
        return None
    over, under = _tokens(base), _tokens(path)
    if len(under) < len(over) or over[0] != under[0]:
        return None
    for one, other in zip(over[1:], under[1:], strict=False):
        if one != other and not (
            (one == "[]" and other[0] == "[") or (other == "[]" and one[0] == "[")
        ):
            return None
    return "".join(under[len(over) :])


def _root(path: str) -> str:
    """*path*'s root: the part :func:`_under` matches first, so a path is below another only with the same root."""
    return _tokens(path)[0]


def _plain(path: str) -> bool:
    """Whether *path*'s only subscripts are ``[]``, so a string prefix reads it."""
    return path.count("[") == path.count("[]")


@lru_cache(maxsize=None)
def _tokens(path: str) -> tuple[str, ...]:
    """*path*'s root, then each of its parts: ``.attr``, ``[]``, ``[i]``, ``['k']``."""
    cut = min(
        (at for at in (path.find("."), path.find("[")) if at >= 0), default=len(path)
    )
    return (path[:cut], *_PART.findall(path, cut))


_PART = re.compile(r"\.[^.\[]+|\[[^\]]*\]")


def key_path(value: WrittenValue, column: str) -> str:
    """Where *value*, a key a write names its row by, holds the value it names the row by *column* with: its path
    where *column* is its column; the mapping's value at *column* where its column is the mapping's key
    (:data:`MAP_KEY`) -- ``criteria['jediTaskID']`` of ``criteria[]``; empty where neither."""
    if value.column.lower() == column.lower():
        return value.path
    if value.column == MAP_KEY and value.path.endswith("[]"):
        return f"{value.path[:-2]}['{column}']"
    return ""


def map_key_columns(path: str, held: set[str]) -> set[str]:
    """The columns a key whose column is the mapping's key (:data:`MAP_KEY`), the mapping's element at *path*
    (``criteria[]``), names the row by where *held* holds the value: the literal key each held path is at
    (``criteria['jediTaskID']``).  A value held at a key not known names no column."""
    found: set[str] = set()
    for one in held:
        if _under(path, one) == "" and (
            key := _LITERAL_KEY.fullmatch(_tokens(one)[-1])
        ):
            found.add(key.group(1))
    return found


_LITERAL_KEY = re.compile(r"\['(.+)'\]")


def _holds(held: set[str], path: str) -> bool:
    """Whether *path* is one of *held*, read part by part (:func:`_under`)."""
    if path in held:
        return True
    keyed = held if not _plain(path) else [one for one in held if not _plain(one)]
    return any(_under(one, path) == "" for one in keyed)


def _parts(path: str) -> int:
    """How many parts *path* has after its root, as :func:`_under` reads them (:func:`_tokens`)."""
    return len(_tokens(path)) - 1


def _receivers_of(caller: ReachNode, link: _Link) -> list[str]:
    """What the caller names the object the call at *link* runs on: its receiver, or the name a constructor's result is bound to."""
    found = [
        c.path
        for c in caller.call_bindings
        if c.line == link.line and c.to == link.to and c.parameter == "self"
    ]
    if link.to.rsplit(".", 1)[-1] == "__init__":
        found += [
            b.name
            for b in caller.bindings
            if b.how == BIND_RESULT
            and b.line == link.line
            and b.path == ""
            and link.to in b.to
        ]
    return sorted(set(found))


def _hole_at(row: ReachNode, line: int) -> str:
    whys = sorted({hole.why for hole in row.holes if hole.line == line})
    return "; ".join(whys) or "a hole"


# -- whether a run can change a column of the row it holds ---------------------------------------------------------

#: Roles of a computed-name write that set no value of the row's own: the row's values read back, a new
#: object's first ones, a pickle's, and the mechanism that is its caller's assignment.
#: How a spec's read of a row (``pack``) assigns its columns: the row's own values.
READS_ROW = "row_read"
NOT_A_CHANGE = frozenset({READS_ROW, "construct", "restore", "mechanism"})

#: What a ``may_change`` answer rests on, beyond :data:`PREMISES` and reach's.
CHANGE_PREMISES: dict[str, str] = {
    "written_back_as_read": (
        "A row written back from an object read from it carries what was read: a write another run made "
        "between the read and the write-back is overwritten, not kept (the lost update)."
    ),
    "holes_land_by_name": (
        "A call the record could not place lands at a function of its name or outside the corpus "
        "(reach's no_runtime_methods): one whose name no function writing the column has is not a write of it."
    ),
    "changes_in_order": (
        "An assignment changes the row at a write only where it can run before it on one run, by the record's "
        "calls and the order of each function's lines (a loop's later pass among them, unless it holds the object "
        "by names bound afresh), and, for a write of the changed attributes, where no reset of the change stands "
        "on every way between.  A call the record could not place is no way."
    ),
}

NO_FLUSH = (
    "assigned on the object, and no write of the object the answer reads takes it"
)
NOT_FOLLOWED_KEY = (
    "writes {column} on a row named by {key}, which is not followed to the object's"
)
UNREAD_KEY = "writes {column} on a row whose key is not read"
NOT_READ_HERE = "writes {column} as {object} holds it, which is not followed to where it was read, on a row named by {key}"
HOLE_NAMED = (
    "a call the record could not place, named {name}, as a function writing {column} is"
)
HOLE_UNNAMED = "a call the record could not place, whose name says nothing"


class Change(NamedTuple):
    """What :func:`may_change` answers: ``True`` the run can change the column, ``False`` it cannot, ``None`` not known."""

    verdict: Optional[bool]
    #: the assignments and the writes taking them, or the writes keyed to the object's row
    because: list[Site]
    #: what leaves the answer open: holes the object reaches, writes whose row is not read, calls not placed
    open: list[Site]
    premises: list[str]
    #: the column assigned on the object where no write of the object the answer reads takes it
    unwritten: list[Site] = []


def may_change(
    rows: dict[str, ReachNode],
    owner: str,
    path: str,
    column: str,
    keys: tuple[str, ...] = (),
    key_column: str = "PandaID",
    spec_class: str = "",
    table_classes: Optional[dict[str, Optional[str]]] = None,
) -> Change:
    """Whether the run *owner* starts can change *column* of the row the object at *path* is.

    Two ways write the row.  The object: *column* assigned on it
    (:func:`follow`) and a write taking it -- the whole object
    (``valuesMap()``), its changed attributes, or the attribute by name --
    where an ``UPDATE``'s key is the object's: held by the key's follow, or
    handed equal to the object's key by every call of the record handing the
    object (:func:`_on_its_row`); one naming its row by anything else is open
    on it, not a write of the object's row.  A
    key: a statement anywhere in the closure writing *column* by another
    value, on the row its ``WHERE`` -- or an ``INSERT``'s *key_column* --
    names, where that key is one of *keys* (paths of *owner* holding the
    row's key, followed as :func:`follow` follows an object) or the object's
    own *key_column*.  A new row (a sequence's key) is not the object's.
    A write by key of what an object holds -- the whole object, or its own
    attribute of the column (``varMap[":param"] = job.jobParameters``) --
    carries the row as that object holds it, the same rule as the object's
    own: the column changes where an object read from the row that holds it
    assigns it, and not otherwise (:func:`_carried_objects`,
    :func:`_row_objects`) -- so asking by the object or by its key answers
    alike.  Where no such object is found the column is open -- unless only
    the changed attributes are written and no change the object held before
    its way began reaches the write.

    Both ways, an assignment changes the row at a write only where it can run
    before the write on one run, on the same object, with no reset of the
    change on every way between for a write of the changed attributes
    (:class:`_Order`, from :attr:`ReachNode.order` and
    :attr:`ReachNode.cleared`): ``retryJob``'s ``else`` sets ``parentID`` on
    the job its ``if`` writes back, and that write never takes it.  The
    answer names ``changes_in_order`` where an assignment was dropped so; an
    assignment no write of the object takes is in ``unwritten``.

    What leaves the answer open is said: a hole the object reaches, a write
    of the column whose row is named by something not followed to the
    object's, and a call of the closure the record could not place whose name
    a function writing the column has (:data:`CHANGE_PREMISES`).

    *spec_class* and *table_classes* (:func:`table_classes`) keep to the
    tables that hold the object's class: ``filesTable4`` holds FileSpec, and a
    write of a file's row is not one of the job's.  A table the build knows
    holds the class learned for it, or none: ``jobs_StatusLog``'s ``INSERT``
    names columns no spec declares (``modiftime_extended``), and its row is a
    log's, not the job's.  A table it does not know -- a name the statement
    builds at run time -- may be the class's, and is kept.
    """
    lowered = column.lower()

    def other_class(write: DbWrite) -> bool:
        if not spec_class or table_classes is None:
            return False
        table = write.table.rpartition(".")[2].lower()
        return table in table_classes and table_classes[table] != spec_class

    obj = follow(rows, owner, path)
    closure = _closure(rows, owner)
    order = _Order(rows, closure)
    because: dict[Site, None] = {}
    opened: dict[Site, None] = {}
    premises = [
        "outside_assigns_nothing",
        "bindings_without_order",
        "handed_values_not_read",
        "literal_keys_apart",
        "rows_hold_their_where",
        "holes_land_by_name",
    ]
    assigned = [
        site
        for site in [*obj.assigned.get(column, []), *obj.assigned.get("*", [])]
        if site.how not in NOT_A_CHANGE
    ]
    # The key: every write of the column in the closure, by the row it names.
    # The row's key is followed as an object is: from *keys*, and from the object's own key attribute, which
    # goes on as a value -- ``recordStatusChange(job.PandaID, ...)``.
    keyed: dict[str, set[str]] = defaultdict(set)
    capped = bool(obj.capped or obj.cut)
    for one in (*keys, f"{path}.{key_column}"):
        key = follow(rows, owner, one)
        capped = capped or bool(key.capped or key.cut)
        for at, paths in key.held.items():
            keyed[at] |= set(paths)
    flushes: list[tuple[Site, DbWrite, bool]] = []
    # An update taking the object lands on its row only where a key of it is the object's key (:func:`_on_its_row`).
    off_row: dict[Site, None] = {}
    for at, paths in obj.held.items():
        row = rows.get(at)
        for write in row.writes if row is not None else []:
            if other_class(write):
                continue
            held = set(paths)
            attributes = {f"{one}.{column}" for one in held}
            whole = bool(write.whole and _holds(held, write.whole))
            taken = [write.whole] if whole else []
            taken += [
                value.path.rsplit(".", 1)[0]
                for value in write.columns
                if value.column.lower() == lowered
                and bool(value.path)
                and _holds(attributes, value.path)
            ]
            if taken and write.kind == "update":
                if not _on_its_row(rows, at, write, taken, keyed, key_column):
                    key_values = _key_values(write, key_column)
                    spelled = ", ".join(text for _held, text in key_values)
                    off_row[
                        Site(
                            at,
                            write.lines[0],
                            NOT_FOLLOWED_KEY.format(column=column, key=spelled)
                            if key_values
                            else UNREAD_KEY.format(column=column),
                        )
                    ] = None
                    continue
            if taken:
                flushes.append(
                    (
                        Site(at, write.lines[0], f"{write.kind} {write.table}"),
                        write,
                        whole and write.changed,
                    )
                )
    written: dict[Site, None] = {}
    taking: dict[Site, None] = {}
    for site in assigned:
        for flush, write, changed in flushes:
            if order.meets(site, flush.owner, write.lines, obj.held, column, changed):
                written[site] = None
                taking[flush] = None
    for site in [*written, *(flush for flush, _write, _changed in flushes)]:
        if site in written or site in taking:
            because[site] = None
    unwritten = [
        Site(site.owner, site.line, NO_FLUSH)
        for site in assigned
        if site not in written
    ]
    if flushes:
        premises.append("written_back_as_read")
    for site in obj.holes:
        opened[site] = None
    for site in off_row:
        opened[site] = None
    if capped:
        premises.append("follow_capped")
    handing: Optional[dict[str, list[tuple[str, CallBinding]]]] = None
    for at in closure:
        row = rows[at]
        own = set(obj.held.get(at, ()))
        attributes = {f"{o}.{column}" for o in own}
        for write in row.writes:
            if (
                write.new_row
                or (write.whole and _holds(own, write.whole))
                or other_class(write)
            ):
                continue
            named = [
                v
                for v in write.columns
                if v.column.lower() == lowered
                and not (v.path and _holds(attributes, v.path))
            ]
            if not named and not write.unread and not write.whole:
                continue
            site_line = write.lines[0]
            key_values = _key_values(write, key_column)
            if write.kind == "insert" and write.whole and not key_values:
                key_values = [
                    (f"{write.whole}.{key_column}", f"{write.whole}.{key_column}")
                ]
            carried = _carried_objects(write, named, column)
            if carried and any(
                held and _holds(keyed.get(at, set()), held)
                for held, _spelled in key_values
            ):
                if handing is None:
                    handing = _handing(rows, closure)
                # What a row object holds unassigned is the row's own value; an object no way reads from the
                # row changes the column where it is assigned, and is not known to otherwise -- unless only its
                # changed attributes are written and no change it held before its way began reaches the write.
                changes: list[Site] = []
                for one in carried:
                    found, unread = _row_objects(
                        rows, handing, keyed, key_column, at, one
                    )
                    if found and "written_back_as_read" not in premises:
                        premises.append("written_back_as_read")
                    changed = write.changed and one == write.whole
                    changes += [
                        site
                        for reach in found
                        for site in _written(order, reach, column, at, write, changed)
                    ]
                    for top, reach in unread:
                        assigned_there = _written(
                            order, reach, column, at, write, changed
                        )
                        changes += assigned_there
                        if not assigned_there and (
                            not changed
                            or order.before_entry_meets(
                                top, at, write.lines, reach.held, column
                            )
                        ):
                            spelled = ", ".join(text for _held, text in key_values)
                            opened[
                                Site(
                                    at,
                                    site_line,
                                    NOT_READ_HERE.format(
                                        column=column, object=one, key=spelled
                                    ),
                                )
                            ] = None
                for site in changes:
                    because[site] = None
                if changes:
                    because[Site(at, site_line, f"{write.kind} {write.table}")] = None
            elif any(
                held and _holds(keyed.get(at, set()), held)
                for held, _spelled in key_values
            ):
                because[
                    Site(
                        at,
                        site_line,
                        f"{write.kind} {write.table} writes {column} on the object's row",
                    )
                ] = None
            elif key_values:
                spelled = ", ".join(text for _held, text in key_values)
                opened[
                    Site(
                        at,
                        site_line,
                        NOT_FOLLOWED_KEY.format(column=column, key=spelled),
                    )
                ] = None
            else:
                opened[Site(at, site_line, UNREAD_KEY.format(column=column))] = None
    # Calls of the closure the record could not place: by name, a writer of the column or not.
    up = _writers_up(rows, column, other_class)
    names = _function_names(rows)
    for at in closure:
        for hole in rows[at].holes:
            if hole.why.startswith("the statement is not read"):
                opened[Site(at, hole.line, hole.why)] = None
                continue
            name = _called_name(hole.callee)
            if name is None:
                opened[Site(at, hole.line, HOLE_UNNAMED)] = None
            elif names.get(name, set()) & up:
                opened[
                    Site(at, hole.line, HOLE_NAMED.format(name=name, column=column))
                ] = None
    if order.dropped:
        premises.append("changes_in_order")
    verdict: Optional[bool] = True if because else (None if opened else False)
    return Change(
        verdict=verdict,
        because=list(because),
        open=list(opened),
        premises=premises,
        unwritten=unwritten,
    )


def _key_values(write: DbWrite, key_column: str) -> list[tuple[str, str]]:
    """``(path, spelled)`` of each value *write* names its row by on *key_column* -- an ``INSERT``'s column, an
    ``UPDATE``'s key: the path holding it (:func:`key_path`) and how an answer spells it."""
    named_by = write.columns if write.kind == "insert" else write.keys
    return [
        (held, v.path if v.column == MAP_KEY else held or v.value or v.why)
        for v in named_by
        if (held := key_path(v, key_column)) or v.column.lower() == key_column.lower()
    ]


def _on_its_row(
    rows: dict[str, ReachNode],
    at: str,
    write: DbWrite,
    taken: list[str],
    keyed: dict[str, set[str]],
    key_column: str,
) -> bool:
    """Whether the ``UPDATE`` *write* at *at*, taking the objects at *taken*, names its row by one of their keys.

    A key of the write on *key_column* is the object's where the asked key's
    follow holds it (*keyed*: the keys given and the object's own key
    attribute, which reads an assignment either way and a row read by the
    key), or where every call of the record hands it and the object equal
    (:func:`_by_callers`): ``registerTaskInOneShot_JEDI`` takes ``jediTaskID``
    and ``taskSpec`` apart and writes the spec's changes on the id's row.  A
    write naming its row by anything else -- another row's key, a key not
    read -- is not the object's row's.
    """
    for held, _spelled in _key_values(write, key_column):
        if not held:
            continue
        if _holds(keyed.get(at, set()), held):
            return True
        if any(_by_callers(rows, at, held, f"{one}.{key_column}", 0) for one in taken):
            return True
    return False


#: How many frames up :func:`_by_callers` reads the callers' word.
CALLER_FRAMES = 4

_CALLERS: dict[int, tuple[dict, dict, frozenset]] = {}


def _callers_of(
    rows: dict[str, ReachNode],
) -> tuple[dict[str, dict[tuple[str, int], list[CallBinding]]], frozenset]:
    """Each function's calls in the record -- caller and line, with every binding of the call, a spread one too --
    and the names of the calls it could not place; kept with *rows* itself, so a cache entry is never another
    record's."""
    held = _CALLERS.get(id(rows))
    if held is None or held[0] is not rows:
        calls: dict[str, dict[tuple[str, int], list[CallBinding]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for caller, row in rows.items():
            for call in row.call_bindings:
                if call.to:
                    calls[call.to][(caller, call.line)].append(call)
        names = frozenset(
            name
            for row in rows.values()
            for hole in row.holes
            if (name := _called_name(hole.callee)) is not None
        )
        _CALLERS.clear()
        held = _CALLERS[id(rows)] = (rows, calls, names)
    return held[1], held[2]


def _by_callers(
    rows: dict[str, ReachNode], at: str, value: str, key: str, frames: int
) -> bool:
    """Whether every call of the record handing *at* the object *key* is the key of hands *value* equal to it --
    paths of *at* rooted at its parameters: one held by the other's follow at the caller, or so at its callers in
    turn.

    The caller's word, read over the record rather than a run's closure: a
    function asked below its callers has none in the run.  A call not handing
    the object leaves it the parameter's default -- no object, no row of it --
    and is not asked (``updateTaskStatusByContFeeder_JEDI`` called with no
    taskSpec).  Not where a call the record could not place has the
    function's name (holes_land_by_name), where a call spreads its arguments
    (what lands where is not said), or where one hands the object and not the
    value; nor where no call hands the object.
    """
    calls, unplaced = _callers_of(rows)
    if at.rpartition(".")[2].rpartition(":")[2] in unplaced or frames >= CALLER_FRAMES:
        return False
    asked = 0
    for (caller, _line), bindings in calls.get(at, {}).items():
        if any(not one.parameter or one.parameter.startswith("*") for one in bindings):
            return False
        handed = {one.parameter + one.at: one.path for one in bindings if one.path}
        handed_value, handed_key = (
            next(
                (
                    path + rest
                    for name, path in handed.items()
                    if (rest := _under(name, one)) is not None
                ),
                None,
            )
            for one in (value, key)
        )
        if handed_key is None:
            continue
        asked += 1
        if handed_value is None or not (
            handed_value == handed_key
            or _holds(_held_in(rows, caller, handed_key), handed_value)
            or _holds(_held_in(rows, caller, handed_value), handed_key)
            or _by_callers(rows, caller, handed_value, handed_key, frames + 1)
        ):
            return False
    return asked > 0


def _held_in(rows: dict[str, ReachNode], at: str, path: str) -> set[str]:
    """The paths of *at* that hold what *path* holds there (:func:`follow`)."""
    return set(follow(rows, at, path).held.get(at, ()))


def _carried_objects(write: DbWrite, named: list, column: str) -> list[str]:
    """The objects *write* takes *column*'s value from as they hold it: the whole object written, where every value
    naming the column is a part of it, or each object whose own attribute of that name a value is
    (``varMap[":param"] = job.jobParameters``); none where a value is anything else -- another attribute
    (``:newStatus = taskSpec.oldStatus`` copies one column into another), a constant, a value made of one."""
    if write.whole and all(
        v.path and _under(write.whole, v.path) is not None for v in named
    ):
        return [write.whole]
    found: list[str] = []
    for value in named:
        base, _, attribute = value.path.rpartition(".")
        if not base or attribute.lower() != column.lower():
            return []
        found.append(base)
    return sorted(set(found))


def _handing(
    rows: dict[str, ReachNode], closure: list[str]
) -> dict[str, list[tuple[str, CallBinding]]]:
    """Each function of *closure* with the calls of *closure* handing it a value: ``(caller, binding)``."""
    found: dict[str, list[tuple[str, CallBinding]]] = defaultdict(list)
    for caller in closure:
        for call in rows[caller].call_bindings:
            if (
                call.to
                and call.path
                and call.parameter
                and not call.parameter.startswith("*")
            ):
                found[call.to].append((caller, call))
    return found


def _row_objects(
    rows: dict[str, ReachNode],
    handing: dict[str, list[tuple[str, CallBinding]]],
    keyed: dict[str, set[str]],
    key_column: str,
    at: str,
    whole: str,
) -> tuple[list[ObjectReach], list[tuple[str, ObjectReach]]]:
    """The follows of the objects read from the row that hold *whole* at *at*, one on each way the key comes by,
    and of the object at the top of each way that reaches none, with the function it is at.

    Found up from the write.  *whole*'s own follow reads the row where it
    reaches a spec's read of one (:data:`READS_ROW`, ``pack``) on the key
    column -- ``killJob`` packs ``job`` from its select and writes it whole.
    Otherwise each path it holds as a start, or back through a binding, whose
    root a call of the closure hands (*handing*) is followed at that caller
    in turn -- ``archiveJob``'s ``job`` from ``updateJobs``' ``jobs[]``, from
    the run's ``upJobs[]``, which ``peekJobs`` read --, on the ways where the
    key's follow (*keyed*) holds the handed object's *key_column*: an object
    handed with another key is another row's, and ``archiveJob`` writes
    ``Setupper``'s jobs too.  A path reached forward goes out, not in.  A way
    ends with no row object where nothing of the closure hands its object in
    -- a parameter of the run itself (``archiveJob``'s ``job``, asked of
    ``archiveJob``), a field.
    """
    found: list[ObjectReach] = []
    unread: list[tuple[str, ObjectReach]] = []
    work = [(at, whole)]
    seen: set[_State] = set()
    while work:
        state = work.pop()
        if state in seen:
            continue
        seen.add(state)
        function, path = state
        reach = follow(rows, function, path)
        if any(
            site.how == READS_ROW
            for site in [
                *reach.assigned.get(key_column, []),
                *reach.assigned.get("*", []),
            ]
        ):
            found.append(reach)
            continue
        steps, _spent = _held_steps(rows[function], {path})
        ups: list[_State] = []
        for held, step in steps.items():
            if _came_forward(step):
                continue
            for caller, call in handing.get(function, ()):
                rest = _under(call.parameter + call.at, held)
                if rest is None:
                    continue
                handed = call.path + rest
                if _holds(keyed.get(caller, set()), f"{handed}.{key_column}"):
                    ups.append((caller, handed))
        if not ups:
            unread.append((function, reach))
            continue
        work += ups
    return found, unread


def _written(
    order: "_Order",
    reach: ObjectReach,
    column: str,
    at: str,
    write: DbWrite,
    changed: bool,
) -> list[Site]:
    """The assignments of *column* on the object *reach* follows that *write*, in *at*, writes (:class:`_Order`)."""
    return [
        site
        for site in _changes(reach, column)
        if order.meets(site, at, write.lines, reach.held, column, changed)
    ]


def _changes(reach: ObjectReach, column: str) -> list[Site]:
    """The assignments of *column* on the object *reach* follows that change it (:data:`NOT_A_CHANGE` aside)."""
    return [
        site
        for site in [*reach.assigned.get(column, []), *reach.assigned.get("*", [])]
        if site.how not in NOT_A_CHANGE
    ]


def _closure(rows: dict[str, ReachNode], owner: str) -> list[str]:
    """Every function *owner* can enter by the record's calls, itself first."""
    found: dict[str, None] = {}
    todo = [owner]
    while todo:
        at = todo.pop()
        if at in found or at not in rows:
            continue
        found[at] = None
        todo.extend(call.to for call in rows[at].calls)
    return list(found)


#: What :func:`by_line_order` and :class:`~bamboo.codemap.models.LineOrder` say of one line running before another.
ONE_PASS, ROUND, APART = "one pass", "round", "apart"


def by_line_order(start: int, other_start: int, loops: Sequence[LoopLines]) -> str:
    """What the order of two lines says of the first running before the second, by the lines their statements
    start at (:class:`~bamboo.codemap.models.LineOrder`): on one pass where it starts no later, or before a
    ``while``'s test on its next pass; only on a later pass where a loop's body holds both; else not."""
    if start <= other_start:
        return ONE_PASS
    inside = [loop for loop in loops if loop.start <= start <= loop.end]
    if any(loop.again and loop.line == other_start for loop in inside):
        return ONE_PASS
    if any(loop.start <= other_start <= loop.end for loop in inside):
        return ROUND
    return APART


def line_paths(row: ReachNode) -> dict[int, set[str]]:
    """Each line of *row* with the paths it takes an object by: an assignment's object, what a call hands and the
    name its result is bound to, a write's whole and the objects of its values."""
    found: dict[int, set[str]] = defaultdict(set)
    for one in row.assigns:
        found[one.line].add(one.path)
    for call in row.call_bindings:
        found[call.line].add(call.path)
    for binding in row.bindings:
        if binding.how == BIND_RESULT:
            found[binding.line].add(binding.name)
    for write in row.writes:
        held = {write.whole} if write.whole else set()
        for value in write.columns:
            if value.path:
                held |= {value.path, value.path.rpartition(".")[0]} - {""}
        for line in write.lines:
            found[line] |= held
    return found


def root_name(path: str) -> str:
    """The name *path* starts at, without its version: ``impl`` of ``impl@166.taskSpec``; a call's result,
    ``<result@12:4>``, whole."""
    root = _root(path)
    return root if root.startswith("<") else root.split("@", 1)[0]


class _Lines:
    """One function's :attr:`ReachNode.order` and :attr:`ReachNode.cleared`, read for :class:`_Order`."""

    def __init__(self, row: ReachNode):
        order = row.order
        self.loops = order.loops if order is not None else []
        self.starts = {one.line: one.start for one in order.starts} if order else {}
        self.said: dict[tuple[int, int], str] = {}
        for kind, held in (
            (APART, order.apart if order else []),
            (ROUND, order.rounds if order else []),
            (ONE_PASS, order.before if order else []),
        ):
            for one in held:
                for other in one.lines:
                    self.said[(one.line, other)] = kind
        self.renewed = {
            (one.loop, one.name): (set(one.after), set(one.before))
            for one in (order.renewed if order else [])
        }
        self.paths = line_paths(row)
        tables: dict[tuple[str, tuple[str, ...]], dict[int, set[int]]] = defaultdict(
            dict
        )
        for one in row.cleared:
            tables[(one.path, tuple(one.columns))][one.line] = set(one.lines)
        self.cleared = [
            (path, set(columns), table) for (path, columns), table in tables.items()
        ]

    def kind(self, one: int, other: int) -> str:
        said = self.said.get((one, other))
        if said is not None:
            return said
        return by_line_order(
            self.starts.get(one, one), self.starts.get(other, other), self.loops
        )

    def renewed_between(self, one: int, other: int, name: str) -> bool:
        """Whether every loop holding both lines renews *name* on every way from *one* round to *other*."""
        start, other_start = self.starts.get(one, one), self.starts.get(other, other)
        common = [
            loop
            for loop in self.loops
            if loop.start <= start <= loop.end and loop.start <= other_start <= loop.end
        ]
        if not common:
            return False
        for loop in common:
            held = self.renewed.get((loop.line, name))
            if held is None or not (one in held[0] or other in held[1]):
                return False
        return True


class _Order:
    """The record's order over one run's closure: whether a change can be what a write writes.

    A change made at a line of one function and a write at a line of another
    meet where some function of the closure has a line leading to the change
    -- its own, or a call going down to it, the change surviving to the
    normal exit of each function on the way back up -- before a line leading
    to the write -- its own, or a call going down to it, reached from the
    entry of each function on the way down.  A call is a way for the object
    only where it hands the object, or what holds it, or binds what it
    returns to a path holding it (:func:`line_paths`): ``updateTask_JEDI(tmpTaskSpec, ...)``
    with a new spec is not a way to write the refined one.  The two lines
    can run one before the other (:attr:`ReachNode.order`), and where only on
    a later pass of a loop, not where every path the second line takes the
    object by is rooted at a name the loop binds to another object between
    (:class:`~bamboo.codemap.models.Renewed`; ``for job in jobs`` holds the
    object as ``jobs[]`` too, which names no one element); and, for a
    write of the changed attributes, no reset of the column on a path holding
    the object stands on every way between (:attr:`ReachNode.cleared`).  One
    line going down to the same functions both ways is read further down.  A
    function not in the closure is taken to meet.
    """

    def __init__(self, rows: dict[str, ReachNode], closure: list[str]):
        self.rows = rows
        self.inside = set(closure)
        self.callers: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for at in closure:
            for call in rows[at].calls:
                if call.to in self.inside:
                    for line in call.lines or [call.line]:
                        self.callers[call.to].append((at, line))
        self._lines: dict[str, _Lines] = {}
        #: Whether a pair was found not to meet: the answer rests on the order (``changes_in_order``).
        self.dropped = False

    def meets(
        self,
        change: Site,
        at: str,
        lines: list[int],
        held: dict[str, list[str]],
        column: str,
        changed: bool,
    ) -> bool:
        """Whether *change* can be what the write at *lines* of *at* writes, the object held as *held* says."""
        if change.owner not in self.inside or at not in self.inside:
            return True
        ups = self._ways(change.owner, [change.line], held, column, changed, True)
        downs = self._ways(at, lines, held, column, changed, False)
        for frame in sorted(ups.keys() & downs.keys()):
            paths = set(held.get(frame, ()))
            for one, those in ups[frame].items():
                for other, these in downs[frame].items():
                    if one == other:
                        if those != these or "" in those:
                            return True
                        continue
                    if self._before(frame, one, other, paths, column, changed):
                        return True
        self.dropped = True
        return False

    def before_entry_meets(
        self,
        top: str,
        at: str,
        lines: list[int],
        held: dict[str, list[str]],
        column: str,
    ) -> bool:
        """Whether a change of *column* the object held before *top* was entered can be what the write of its changed
        attributes at *lines* of *at* writes."""
        if top not in self.inside or at not in self.inside:
            return True
        downs = self._ways(at, lines, held, column, True, False)
        paths = set(held.get(top, ()))
        if any(
            self._before(top, ENTRY_LINE, other, paths, column, True)
            for other in downs.get(top, {})
        ):
            return True
        self.dropped = True
        return False

    def _ways(
        self,
        start: str,
        lines: list[int],
        held: dict[str, list[str]],
        column: str,
        changed: bool,
        up: bool,
    ) -> dict[str, dict[int, set[str]]]:
        """Each function with the lines at which what happens at *lines* of *start* happens there: its own, ``""``,
        or the calls going down to it carrying the object, the functions called.  *up*: a change, surviving to each
        exit on the way; else a write, reached from each entry."""
        found: dict[str, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
        for line in lines:
            found[start][line].add("")

        def passes(frame: str, line: int) -> bool:
            if not changed:
                return True
            paths = set(held.get(frame, ()))
            if up:
                return not self._is_cleared(frame, line, EXIT_LINE, paths, column)
            return not self._is_cleared(frame, ENTRY_LINE, line, paths, column)

        live = [start] if any(passes(start, line) for line in lines) else []
        seen = set(live)
        while live:
            callee = live.pop()
            for caller, line in self.callers.get(callee, ()):
                if not self._carries(caller, line, held):
                    continue
                found[caller][line].add(callee)
                if caller not in seen and passes(caller, line):
                    seen.add(caller)
                    live.append(caller)
        return found

    def _carries(self, frame: str, line: int, held: dict[str, list[str]]) -> bool:
        """Whether a path *line* of *frame* takes an object by holds the object or what holds it."""
        here = held.get(frame)
        if not here:
            return False
        return any(
            _under(path, one) is not None
            for path in self._lines_of(frame).paths.get(line, ())
            for one in here
        )

    def _before(
        self,
        frame: str,
        one: int,
        other: int,
        paths: set[str],
        column: str,
        changed: bool,
    ) -> bool:
        lines = self._lines_of(frame)
        kind = lines.kind(one, other)
        if kind == APART:
            return False
        if kind == ROUND:
            # What the later pass writes it takes by the second line's paths.
            taken = [
                path
                for path in lines.paths.get(other, ())
                if any(_under(path, held) is not None for held in paths)
            ]
            if taken and all(
                lines.renewed_between(one, other, root_name(path)) for path in taken
            ):
                return False
        return not (changed and self._is_cleared(frame, one, other, paths, column))

    def _is_cleared(
        self, frame: str, one: int, other: int, paths: set[str], column: str
    ) -> bool:
        if not paths:
            return False
        return any(
            (column in columns or "*" in columns)
            and other in table.get(one, ())
            and _holds(paths, path)
            for path, columns, table in self._lines_of(frame).cleared
        )

    def _lines_of(self, frame: str) -> _Lines:
        found = self._lines.get(frame)
        if found is None:
            found = self._lines[frame] = _Lines(self.rows[frame])
        return found


def _function_names(rows: dict[str, ReachNode]) -> dict[str, set[str]]:
    names: dict[str, set[str]] = defaultdict(set)
    for row in rows.values():
        names[row.function].add(row.owner)
    return names


def _called_name(callee: str) -> Optional[str]:
    """The method name a hole calls -- ``debug`` of ``self.logger.debug`` -- or ``None`` where its name says nothing.

    A plain name called is a local, a parameter or a function, and only the
    last is a name to read; a call's result or an element called names nothing.
    """
    text = callee.rstrip()
    if "." not in text:
        return None
    last = text.rsplit(".", 1)[1]
    return last if last.isidentifier() else None


def _writers_up(
    rows: dict[str, ReachNode], column: str, other_class=lambda write: False
) -> set[str]:
    """The functions that write *column* (any write that can), and every function that can reach one, holes by name."""
    lowered = column.lower()
    up = {
        row.owner
        for row in rows.values()
        if any(
            not other_class(write)
            and (
                write.unread
                or write.whole
                or any(v.column.lower() == lowered for v in write.columns)
            )
            for write in row.writes
        )
        or any(hole.why.startswith("the statement is not read") for hole in row.holes)
    }
    names = _function_names(rows)
    changed = True
    while changed:
        changed = False
        for row in rows.values():
            if row.owner in up:
                continue
            if any(call.to in up for call in row.calls) or any(
                (name := _called_name(hole.callee)) and names.get(name, set()) & up
                for hole in row.holes
            ):
                up.add(row.owner)
                changed = True
    return up


def table_classes(entities: list[EntityNode]) -> dict[str, Optional[str]]:
    """Each table, as every spelling of it folded to lower case, and the spec class it holds where one was learned."""
    found: dict[str, Optional[str]] = {}
    for entity in entities:
        for spelled in {entity.name.rsplit(":", 1)[-1], *entity.tables}:
            found[spelled.rsplit(".", 1)[-1].lower()] = entity.spec_class
    return found
