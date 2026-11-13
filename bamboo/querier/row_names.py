"""Which names of an arm's function hold the row the arm writes, by the map's records (KNOWN_ISSUES.md K64).

A guard's names were read as the question's row by how they are spelled: the rows file names ``job`` and
``self.job``, and every such name in any arm's guard read the row's values.  The spelling is not the object.
``killEventServiceConsumers`` 818 writes ``dJob``, a consumer it packs from a SELECT, while its guard
``pandaID == job.PandaID`` reads ``job``, the parameter it was handed: another job.  And ``doGenerate``'s ``jobSpec``,
the object every arm there writes, was read as no row at all, since no rows file spells it.

So a name holds the row only where the map's records say it holds the object the arm writes (its
``WriteFrom`` of role ``object``), under the premise every reading of an arm's guards makes: this arm wrote the
question's row.

- **The written expression itself.**  A guard name that is the arm's written expression holds the row where the
  bindings that reach the guard's test are the ones that reach the arm (``pathcond.held_at``).  A field of ``self``
  is one place in the method.  ``for job in self.jobs + [None]`` binds ``job`` once.  In the iteration whose arm
  ran, ``job`` is the row, so its sentinel test ``job is None`` reads the row: not None.
- **The same object by the records.**  Each side has one way (one source).  Neither goes through a call outside the
  corpus (``through``: a copy, a conversion).  Both end at the same place (kind, name, and the function and line of
  a parameter or of a select's rows, or the line of an object built).  Their steps are equal.  An element step
  (``[*]``, ``[1]``) alone binds nothing: two names, each an element of one list, may hold different elements.
- **An element of the row's collection.**  A name whose way is the written object's plus an attribute the rows file
  holds as a collection (``job.Files``) and an element (``[*]``) holds one of the collection's rows (``file``).
- **An SQL arm.**  An arm that writes through a statement has no written object, and the statement's ``KeyFrom``
  stands in.  A name holds the row where the key has one source and the name holds the whole row the key's column
  is read off: the same select, with no column and no steps.

A name the rows file spells and the records do not bind is not read as the row: its value is unknown.
"""

from __future__ import annotations

import functools
from collections.abc import Iterable
from typing import Any

from bamboo.codemap import pathcond
from bamboo.querier import mapdump
from bamboo.querier import rows as R

#: the kinds of way ends that say which object a name holds; a constant or an unknown end says nothing of it
IDENTITY = frozenset({"object", "parameter", "rows", "self_field", "call", "global", "attribute", "database", "sequence"})


def names(owner: str, line: int, subject: str, entity: str, elements: dict[str, str]) -> dict[str, str] | None:
    """The names of *owner*'s arm at *line*, writing *subject* (``JobSpec.jobStatus``), that hold the row: name ->
    the row's entity (*entity*, or the element's for a collection, *elements* giving attribute -> entity).  None
    where the map records nothing the arm writes: no ``WriteFrom`` of role ``object`` and no ``KeyFrom``."""
    fragment = mapdump.fragment()
    if fragment is None:
        return None
    found = _names(id(fragment), owner, line, subject, entity, tuple(sorted(elements.items())))
    return None if found is None else dict(found)


@functools.cache
def _index(_fragment: int) -> tuple[dict, dict, dict]:
    fragment = mapdump.fragment()
    writes: dict[tuple, list] = {}
    for w in getattr(fragment, "write_froms", None) or []:
        if w.role == "object":
            writes.setdefault((w.owner, w.line, w.subject), []).append(w)
    guards: dict[str, list] = {}
    for g in getattr(fragment, "guard_froms", None) or []:
        guards.setdefault(g.owner, []).append(g)
    keys: dict[tuple, list] = {}
    for k in getattr(fragment, "key_froms", None) or []:
        for column, lines in (k.writes or {}).items():
            for n in lines:
                keys.setdefault((column, n), []).append(k)
    return writes, guards, keys


def _end(s: Any) -> tuple:
    """Where a way ends, as both sides' ends are compared: a field of ``self`` wherever the function reads it, an object
    at the line it is built.  *site_owner* is the statement or function a way ends at, but on a field of ``self`` or
    an object the map gives the class there (SFF): ``JediDatasetSpec()`` built at 549 and at 788 of one function are
    two objects."""
    column = getattr(s, "column", "") or ""
    if s.site_owner and s.kind not in ("self_field", "object"):
        return (s.kind, s.name, s.site_owner, s.site_line, column)
    if s.kind == "self_field":
        return (s.kind, s.name, None, column)
    return (s.kind, s.name, s.line, column)


def _steps(s: Any) -> tuple[str, ...]:
    return tuple(str(x) for x in getattr(s, "steps", None) or ())


def _one(sources: Iterable[Any]) -> Any | None:
    """The one way of *sources*, None where there are several ways or it goes through a call outside the corpus."""
    seen = {(_end(s), _steps(s), tuple(map(str, s.through or ()))): s for s in sources}
    if len(seen) != 1:
        return None
    (one,) = seen.values()
    return None if one.through else one


def _function(owner: str, line: int) -> Any | None:
    rel, _, name = owner.partition("::")
    try:
        found = R.functions(rel, name.split(".")[-1])
    except (OSError, SyntaxError):
        return None
    return next((f for f in found if f.lineno <= line <= (f.end_lineno or f.lineno)), None)


def same_binding(func: Any, name: str, tests: Iterable[int], arm: int) -> bool:
    """Whether the bindings of *name*'s root that reach each of *tests* are the ones that reach the arm at *arm*: a
    guard's test, or a row printing the name (``which_ran.other_objects``)."""
    root = name.split(".")[0]
    if root in ("self", "cls"):
        return True
    at = R.statement_at(func, arm)
    if at is None:
        return False
    sites = {id(s) for s in pathcond.held_at(func, root, at).sites}
    for n in tests:
        test = R.statement_at(func, n)
        if test is None or {id(s) for s in pathcond.held_at(func, root, test).sites} != sites:
            return False
    return True


@functools.lru_cache(maxsize=8192)
def _names(
    _fragment: int, owner: str, line: int, subject: str, entity: str, elements: tuple[tuple[str, str], ...]
) -> tuple[tuple[str, str], ...] | None:
    writes, guards, keys = _index(_fragment)
    at_arm: dict[str, list] = {}
    for g in guards.get(owner, []):
        if line in (g.arms or []):
            at_arm.setdefault(g.name, []).append(g)
    out: dict[str, str] = {}
    written = writes.get((owner, line, subject), [])
    if written:
        func = _function(owner, line)
        exprs = {w.expression for w in written}
        for expr in exprs:
            out[expr] = entity
        mine = _one(s for w in written for s in w.sources)
        by_attr = dict(elements)
        for name, gs in at_arm.items():
            if name in exprs:
                if func is not None and same_binding(func, name, [g.test_line for g in gs], line):
                    out[name] = entity
                else:
                    out.pop(name, None)
                continue
            theirs = _one(s for g in gs for s in g.sources)
            if mine is None or theirs is None or theirs.kind not in IDENTITY or _end(theirs) != _end(mine):
                continue
            g_steps, w_steps = _steps(theirs), _steps(mine)
            if g_steps == w_steps and not any(x.startswith("[") for x in g_steps):
                out[name] = entity
            elif (
                len(g_steps) == len(w_steps) + 2
                and g_steps[: len(w_steps)] == w_steps
                and g_steps[-1] == "[*]"
                and g_steps[-2].startswith(".")
                and g_steps[-2][1:] in by_attr
            ):
                out[name] = by_attr[g_steps[-2][1:]]
        return tuple(sorted(out.items()))
    statements = keys.get((subject, line), [])
    if not statements:
        return None
    # the key's sources in the statement's own function: a parameter of it (a KeySource names no function for one),
    # or a select it runs.  The others are where its callers' values came from, the same object followed up
    own = {
        (s.kind, s.name, s.site_owner or k.owner, s.site_line if s.site_owner else None)
        for k in statements
        for s in k.sources or []
        if (s.kind == "parameter" and not s.site_owner) or s.site_owner == k.owner
    }
    if len(own) != 1:
        return ()
    ((kind, name_, site, site_line),) = own
    for name, gs in at_arm.items():
        theirs = _one(s for g in gs for s in g.sources)
        if (
            theirs is not None
            and not (getattr(theirs, "column", "") or "")
            and not _steps(theirs)
            and theirs.kind == kind
            and theirs.name == name_
            and theirs.site_owner == site
            and (kind == "parameter" or theirs.site_line == site_line)
        ):
            out[name] = entity
    return tuple(sorted(out.items()))
