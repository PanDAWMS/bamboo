"""Whether a value one function printed is a value of the object a name holds in another, by the map's records
(KNOWN_ISSUES.md K63).

A hole another function printed about a row (``self.job.jobStatus`` in the adder) used to be read for any name the
rows file spells the row with (``job`` in the task buffer's ``killJob``): the same row by spelling.  The same row is
not the same object.  ``killJob`` packs a ``JobSpec`` of its own from a SELECT (job_complex_module 997-998) and reads
the database's ``holding``, while the adder's object holds the ``failed`` the pilot reported.  The adder printed that
after ``killJob`` had run.

So a hole printed in another function is read for a name only where the map's records say both hold one object.
Each side's ways are followed to the places they reach (:func:`ways`):
- the name itself: a parameter of its function, a field of ``self`` in its file, or a local;
- each source of its ``GuardFrom`` at the line, and of its ``ParameterFrom``;
- through the ``ParameterFrom`` of each parameter a way reaches, :data:`DEPTH` deep.
Both hold one object where they share a place.

Some examples:
- The adder's ``self.job`` is handed to ``updateJobs``'s ``jobs`` at adder_gen.py:472.  So 873's
  ``isJumboJob(self.job)`` is ``updateJobs``'s ``isJumboJob(job)``.
- ``archiveJob``'s ``job`` is handed from ``updateJobs``'s ``jobs`` at TaskBuffer.py:667.  So it is the ``job``
  ``updateJobs`` loops over.
- ``killJob``'s ``job`` is a ``JobSpec`` it builds at 997.  Nothing hands it the adder's.

Two limits.  An element of one list is read as the list's one object: the map does not say which element a loop is at.
A field of ``self`` is matched by its file and name.
"""

from __future__ import annotations

import functools

from bamboo.querier import heldcol, mapdump

#: how many ParameterFroms a way is followed through
DEPTH = 3


def _file(where: str) -> str:
    """The file of an owner (``file::name``) or of a call (``file:line``)."""
    return where.partition("::")[0] if "::" in where else where.rpartition(":")[0]


def _place(source, context: str) -> tuple:
    """Where a way ends, as a place both sides can share: a function's parameter, a field of ``self`` in a file, an
    object built at a line; *context* is the file of the function the way is read in, where the source has no call."""
    if source.kind == "parameter":
        return ("parameter", source.site_owner, source.name)
    file = _file(source.route[-1]) if source.route else context
    return (source.kind, file, source.name, getattr(source, "line", None) if source.kind != "self_field" else None)


def ways(owner: str, line: int | None, name: str) -> frozenset[tuple]:
    """The places *name* at *line* of *owner* reaches by the map's records."""
    return _ways(id(mapdump.fragment()), owner, line, name)


@functools.lru_cache(maxsize=8192)
def _ways(_fragment: int, owner: str, line: int | None, name: str) -> frozenset[tuple]:
    guards, params = heldcol._maps()
    here = _file(owner)
    places: set[tuple]
    if name.startswith("self."):
        places = {("self_field", here, name, None)}
    elif (owner, name) in params:
        places = {("parameter", owner, name)}
    else:
        places = {("local", owner, name)}
    frontier = [
        (s, here, 0)
        for g in guards.get(owner, [])
        if g.name == name and line is not None and (line in g.arms or g.test_line == line)
        for s in g.sources
    ]
    seen: set[tuple[str, str]] = set()
    if (owner, name) in params:
        seen.add((owner, name))
        frontier += [(s, here, 0) for s in params[(owner, name)].sources]
    while frontier:
        s, context, depth = frontier.pop()
        places.add(_place(s, context))
        if s.kind == "parameter" and depth < DEPTH:
            key = (s.site_owner, s.name)
            if key not in seen and key in params:
                seen.add(key)
                frontier.extend((s2, _file(s.site_owner), depth + 1) for s2 in params[key].sources)
    return frozenset(places)


def same(reader: str, line: int | None, names: tuple[str, ...], printer: str, printed: tuple[str, ...]) -> bool:
    """Whether *names* at *line* of *reader* hold the object *printed* names in *printer*, by the map's records."""
    if reader == printer:
        return True
    mine = set().union(*(ways(reader, line, n) for n in names)) if names else set()
    theirs = set().union(*(ways(printer, None, r) for r in printed)) if printed else set()
    return bool(mine & theirs)
