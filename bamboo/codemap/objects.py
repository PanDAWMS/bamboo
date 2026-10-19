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
:data:`PREMISES`; the bindings are read without the statements' order, so an
object can be said to reach more than it does, not less.
"""

from __future__ import annotations

from collections import defaultdict
from typing import NamedTuple, Optional

from bamboo.codemap.models import (
    BIND_ALIAS,
    BIND_ELEMENT,
    BIND_FILL,
    BIND_RESULT,
    BIND_RETURN,
    DbWrite,
    EntityNode,
    ReachNode,
)

PREMISES: dict[str, str] = {
    "outside_assigns_nothing": (
        "Code outside the corpus assigns no attribute of an object of the corpus it is handed, other than through "
        "the corpus's own methods -- which it calls only as reach's outside_calls_back_only_handed allows."
    ),
    "bindings_without_order": (
        "A name bound to what a path holds anywhere in a function holds it everywhere in the function."
    ),
}

#: How many (function, path) states one answer reads before it stops and says so.
MAX_STATES = 5000
#: How many paths one function can hold the object by before the rest are dropped and said.
MAX_HELD = 200
#: The longest path followed, in parts after the root.
MAX_PARTS = 6

#: Why a hole the record keeps no reason for is one.
NO_PARAMETER = "the call does not say which parameter takes it (*args, **kwargs)"
STATES_SPENT = f"stopped after {MAX_STATES} (function, path) states"
HELD_SPENT = f"the object is held by more than {MAX_HELD} paths in one function; the rest are not followed"

RETURN = "<return>"

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
    cut: str
    #: owner -> how the object first came into that function: the caller, the line of the call, the path it has
    #: there, and the caller's path that was handed (:func:`way_to` reads a way back to the start from it)
    entered: dict[str, tuple[str, int, str, str]]
    #: owner -> path -> the call that handed the object back to that function as the path: its line and the callee
    #: (on its ``self``, or returned)
    returned: dict[str, dict[str, tuple[int, str]]]


class _Link(NamedTuple):
    """A call that brought the object into a state: the caller's state, the line, and the function called."""

    caller: _State
    line: int
    to: str


def follow(rows: dict[str, ReachNode], owner: str, path: str) -> ObjectReach:
    """What is assigned on the object *owner* names *path*, and the holes it reaches; see the module docstring."""
    assigned: dict[str, dict[Site, None]] = defaultdict(dict)
    holes: dict[Site, None] = {}
    came_from: dict[Site, None] = {}
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
    entered: dict[str, tuple[str, int, str, str]] = {}
    returned: dict[str, dict[str, tuple[int, str]]] = defaultdict(dict)

    def into(child: _State, link: _Link, handed: str) -> None:
        """Bring the object into *child* by *link*; what *child* already says of it goes back at once."""
        if _parts(child[1]) > MAX_PARTS:
            return
        if child[0] != owner:
            entered.setdefault(child[0], (link.caller[0], link.line, child[1], handed))
        if link not in links[child]:
            links[child].add(link)
            if _hand_back(rows, link, outputs[child], extra, returned):
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
        steps, spent = _held_steps(row, {start}, extra[state])
        held = set(steps)
        held_in[at] |= held
        if spent:
            cut = cut or HELD_SPENT
        for one in row.assigns:
            if one.path in held:
                for attr in one.attrs:
                    assigned[attr][Site(at, one.line, one.how)] = None
        for binding in row.bindings:
            if binding.how != BIND_RESULT:
                continue
            for one in sorted(held):
                rest = _under(binding.name, one)
                if rest is None or _came_forward(steps[one]):
                    # What came into the name by a binding is not what the call bound to it returns.
                    continue
                # The object held here came out of the call: in from what it returns.
                came_from[Site(at, binding.line, binding.name)] = None
                for call in row.calls:
                    if binding.line not in (call.lines or [call.line]):
                        continue
                    if call.to.rsplit(".", 1)[-1] == "__init__":
                        if binding.path == "[]":
                            continue  # an element of a new object is not the object
                        taken = "self" + rest
                    else:
                        taken = RETURN + binding.path + rest
                    into((call.to, taken), _Link(state, binding.line, call.to), one)
        for call in row.call_bindings:
            for one in sorted(held):
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
                    taken = call.parameter + ("[]" if call.element else "") + rest
                    into((call.to, taken), _Link(state, call.line, call.to), one)
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
                if _hand_back(rows, link, outputs[state], extra, returned):
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
        entered=entered,
        returned=dict(returned),
    )


def way_to(reach: ObjectReach, owner: str) -> list[tuple[str, int, str, str, str]]:
    """One way the object went from the start to *owner*, in order: per call, ``(caller, line, callee, path in the
    callee, the caller's path handed)``.

    Each function's first entry is kept, so this is the way :func:`follow`
    first came in by -- one of the ways, not every one.
    """
    way: list[tuple[str, int, str, str, str]] = []
    seen = {owner}
    while owner in reach.entered:
        caller, line, path, handed = reach.entered[owner]
        way.append((caller, line, owner, path, handed))
        if caller in seen:
            break
        seen.add(caller)
        owner = caller
    return list(reversed(way))


def _hand_back(
    rows: dict[str, ReachNode],
    link: _Link,
    ends: set[str],
    extra: dict[_State, set[str]],
    returned: dict[str, dict[str, tuple[int, str]]],
) -> bool:
    """Put where the object ends up in the callee into the caller's state; whether that added a path."""
    caller = rows.get(link.caller[0])
    if caller is None or not ends:
        return False
    added = False
    held = extra[link.caller]
    for end in sorted(ends):
        if end.startswith(RETURN):
            back = end[len(RETURN) :]
            names = [
                result.name + below
                for result in caller.bindings
                if result.how == BIND_RESULT
                and result.line == link.line
                and (below := _element_back(result.path, back)) is not None
            ]
        else:
            below = end[len("self") :]
            names = [receiver + below for receiver in _receivers_of(caller, link)]
        for name in names:
            if (
                name != link.caller[1]
                and name not in held
                and _parts(name) <= MAX_PARTS
            ):
                held.add(name)
                returned[link.caller[0]].setdefault(name, (link.line, link.to))
                added = True
    return added


def _held(row: ReachNode, start: set[str]) -> tuple[set[str], bool]:
    """Every path of *row*'s function that holds what *start* holds, by its bindings; see :func:`_held_steps`."""
    steps, spent = _held_steps(row, start)
    return set(steps), spent


def _held_steps(
    row: ReachNode, start: set[str], returned: set[str] = frozenset()
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
    forward through one binding of a name is stepped back only through that
    binding: ``self.jobs[]`` reaches ``tag_job`` and not ``self.jumbo_jobs[]``.
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
    held: dict[str, Optional[tuple[str, int, str]]] = dict.fromkeys(sorted(start))
    pending = sorted(set(returned) - set(start))
    entered: dict[str, set[int]] = defaultdict(
        set
    )  # path -> the bindings it was reached forward by
    changed = True
    while changed:
        changed = False
        for index, binding in enumerate(row.bindings):
            if binding.how == BIND_FILL or (
                binding.how == BIND_ALIAS and binding.name.endswith("[]")
            ):
                # an element of the container: ``updatedJobList.append(job)``, ``varMap[":x"] = v``
                target = (
                    binding.name if binding.name.endswith("[]") else binding.name + "[]"
                )
            elif binding.how in (BIND_ALIAS, BIND_ELEMENT, BIND_RETURN):
                # an element's path is already ``container[]``
                target = binding.name
            else:
                continue
            for one in list(held):
                for frm, to, way in (
                    (binding.path, target, _FORWARD),
                    (target, binding.path, _BACK),
                ):
                    rest = _under(frm, one)
                    if rest is None:
                        continue
                    if way == _BACK and entered.get(one) and index not in entered[one]:
                        continue
                    taken = to + rest
                    if way == _FORWARD and (taken in entered or taken not in held):
                        entered[taken].add(index)
                    if taken not in held and _parts(taken) <= MAX_PARTS:
                        if len(held) >= MAX_HELD:
                            return held, True
                        how = binding.how if way == _FORWARD else f"{binding.how} back"
                        held[taken] = (one, binding.line, how)
                        changed = True
        if not changed and pending:
            for one in pending:
                if one not in held:
                    held[one] = None
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
    """What *path* goes on with below *base* -- ``.job`` for ``self`` and ``self.job`` -- or ``None`` where it is not below it."""
    if path == base:
        return ""
    if path.startswith(base) and path[len(base)] in ".[":
        return path[len(base) :]
    return None


def _parts(path: str) -> int:
    return path.count(".") + path.count("[]")


def _element_back(result: str, back: str) -> Optional[str]:
    """The path in the caller of what the callee returned, from the binding's ``[]`` and what is below the return."""
    if result == "[]":
        # ``x = f()[0]``: x is an element of what f returns.
        return back[2:] if back.startswith("[]") else None
    return back


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
            if b.how == BIND_RESULT and b.line == link.line and b.path == ""
        ]
    return sorted(set(found))


def _hole_at(row: ReachNode, line: int) -> str:
    whys = sorted({hole.why for hole in row.holes if hole.line == line})
    return "; ".join(whys) or "a hole"


# -- whether a run can change a column of the row it holds ---------------------------------------------------------

#: Roles of a computed-name write that set no value of the row's own: the row's values read back, a new
#: object's first ones, a pickle's, and the mechanism that is its caller's assignment.
NOT_A_CHANGE = frozenset({"row_read", "construct", "restore", "mechanism"})

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
}

NO_FLUSH = (
    "assigned on the object, and no write of the object the answer reads takes it"
)
NOT_FOLLOWED_KEY = (
    "writes {column} on a row named by {key}, which is not followed to the object's"
)
UNREAD_KEY = "writes {column} on a row whose key is not read"
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
    (``valuesMap()``), its changed attributes, or the attribute by name.  A
    key: a statement anywhere in the closure writing *column* by another
    value, on the row its ``WHERE`` -- or an ``INSERT``'s *key_column* --
    names, where that key is one of *keys* (paths of *owner* holding the
    row's key, followed as :func:`follow` follows an object) or the object's
    own *key_column*.  A new row (a sequence's key) is not the object's.

    What leaves the answer open is said: a hole the object reaches, a write
    of the column whose row is named by something not followed to the
    object's, and a call of the closure the record could not place whose name
    a function writing the column has (:data:`CHANGE_PREMISES`).

    *spec_class* and *table_classes* (:func:`table_classes`) keep to the
    tables that hold the object's class: ``filesTable4`` holds FileSpec, and a
    write of a file's row is not one of the job's.  A table whose class is
    not learned is kept.
    """
    lowered = column.lower()

    def other_class(write: DbWrite) -> bool:
        if not spec_class or table_classes is None:
            return False
        held = table_classes.get(write.table.lower())
        return held is not None and held != spec_class

    obj = follow(rows, owner, path)
    because: dict[Site, None] = {}
    opened: dict[Site, None] = {}
    premises = [
        "outside_assigns_nothing",
        "bindings_without_order",
        "holes_land_by_name",
    ]
    assigned = [
        site
        for site in [*obj.assigned.get(column, []), *obj.assigned.get("*", [])]
        if site.how not in NOT_A_CHANGE
    ]
    flushes: list[Site] = []
    for at, paths in obj.held.items():
        row = rows.get(at)
        for write in row.writes if row is not None else []:
            if other_class(write):
                continue
            held = set(paths)
            takes = write.whole in held or any(
                value.column.lower() == lowered
                and value.path in {f"{one}.{column}" for one in held}
                for value in write.columns
            )
            if takes:
                flushes.append(Site(at, write.lines[0], f"{write.kind} {write.table}"))
    if assigned and flushes:
        for site in [*assigned, *flushes]:
            because[site] = None
    unwritten = (
        [] if flushes else [Site(site.owner, site.line, NO_FLUSH) for site in assigned]
    )
    if flushes:
        premises.append("written_back_as_read")
    for site in obj.holes:
        opened[site] = None
    # The key: every write of the column in the closure, by the row it names.
    # The row's key is followed as an object is: from *keys*, and from the object's own key attribute, which
    # goes on as a value -- ``recordStatusChange(job.PandaID, ...)``.
    keyed: dict[str, set[str]] = defaultdict(set)
    for one in (*keys, f"{path}.{key_column}"):
        for at, paths in follow(rows, owner, one).held.items():
            keyed[at] |= set(paths)
    closure = _closure(rows, owner)
    for at in closure:
        row = rows[at]
        own = set(obj.held.get(at, ()))
        for write in row.writes:
            if write.new_row or write.whole in own or other_class(write):
                continue
            named = [
                v
                for v in write.columns
                if v.column.lower() == lowered
                and v.path not in {f"{o}.{column}" for o in own}
            ]
            if not named and not write.unread and not write.whole:
                continue
            site_line = write.lines[0]
            named_by = write.columns if write.kind == "insert" else write.keys
            key_values = [
                (v.path, v.path or v.value or v.why)
                for v in named_by
                if v.column.lower() == key_column.lower()
            ]
            if write.kind == "insert" and write.whole and not key_values:
                key_values = [
                    (f"{write.whole}.{key_column}", f"{write.whole}.{key_column}")
                ]
            if any(held and held in keyed.get(at, ()) for held, _spelled in key_values):
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
    verdict: Optional[bool] = True if because else (None if opened else False)
    return Change(
        verdict=verdict,
        because=list(because),
        open=list(opened),
        premises=premises,
        unwritten=unwritten,
    )


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
