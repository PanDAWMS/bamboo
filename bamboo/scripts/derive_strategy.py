"""Follow the Code Map from one observed value, one derivation per hop: the investigation the querier runs.

This was also ``bamboo derive-strategy``, the command that printed the walk's
strategy for a reader and read the map from Neo4j.  The map is kept as dumps now,
read through the querier's in-memory backend (``bamboo.querier.mapdump``), and the
command had no user left, so it and its report are gone.  What stays is the walk:
:func:`_investigate` is what ``bamboo.querier.rerun`` calls, by this module's name.
"""

from __future__ import annotations

from typing import Callable, Optional

from bamboo.codemap import strategy as strategy_mod
from bamboo.codemap.lookup import CodeMap
from bamboo.codemap.models import (
    LEAD_MAP,
    WALK_ASKED,
    WALK_BUDGET,
    WALK_TERMINAL,
    Hop,
    Investigation,
    Strategy,
    Symptom,
)

#: Hops a walk takes before it stops of its own accord.  Small because the
#: measured walks are: the passthrough chains in this corpus are at most four
#: fields long, so anything deeper is a loop or a mistake -- and because each
#: hop reaches production once, the count is what stands in for the person who
#: would otherwise say "that's enough".
DEFAULT_HOPS = 3


async def walk(
    code_map: CodeMap,
    symptom: Symptom,
    budget: int = DEFAULT_HOPS,
    settle: Optional[Callable[[Strategy], Strategy]] = None,
) -> Investigation:
    """Follow the map from *symptom* until it stops, one derivation per hop.

    **The loop is here and not in the derivation, and that is the design.**  A
    hop needs evidence before the next one is worth taking, and collecting
    evidence is a separate cadence -- a ``derive`` that walked would have to
    reach production from inside itself, which is the split the two phases
    exist to keep.  So the map enumerates, *settle* selects, and this only
    decides whether to go round again.  That seam is also where a reasoner can
    sit without being able to invent anything: it may pick among the leads the
    map opened, and it may not add one.

    *settle* is what applies the evidence -- fetching it, loading it, or
    neither.  Handed in rather than built here so this is testable against a
    fixture and so an offline walk and a fetching one are the same code path.
    """
    investigation = Investigation(budget=budget)
    asked = symptom
    opened, source = "", LEAD_MAP
    while True:
        strategy = await strategy_mod.derive(code_map, asked)
        if settle is not None:
            strategy = settle(strategy)
        investigation.visited.append(strategy_mod.visit_key(asked))
        investigation.hops.append(
            Hop(
                number=len(investigation.hops),
                symptom=asked,
                opened=opened,
                source=source,
                strategy=strategy,
            )
        )
        lead, repeats = strategy_mod.next_question(strategy, set(investigation.visited))
        # Recorded even when a hop is available, because a walk can both find
        # somewhere to go and have come back past somewhere it has been.
        investigation.cycles += [r for r in repeats if r not in investigation.cycles]
        if lead is None or lead.symptom is None:
            investigation.stopped = WALK_ASKED if repeats else WALK_TERMINAL
            return investigation
        if len(investigation.hops) >= budget:
            # Checked after choosing, so the reason is "the budget ran out"
            # rather than "there was nothing left" -- those are opposite things
            # to do next, and a walk that conflated them would tell a reader to
            # stop looking when it had simply been told to stop.
            investigation.stopped = WALK_BUDGET
            return investigation
        asked, opened, source = lead.symptom, lead.field, lead.source


async def _investigate(
    map_id: str,
    version: Optional[str],
    symptom: Optional[Symptom],
    description: Optional[str],
    task_id: Optional[str],
    observed_diag: Optional[str],
    budget: int,
    settle: Optional[Callable[[Strategy], Strategy]],
) -> tuple[Investigation, list]:
    from bamboo.database.graph_database_client import GraphDatabaseClient

    graph_db = GraphDatabaseClient()
    await graph_db.connect()
    try:
        code_map = CodeMap(graph_db, map_id=map_id, version=version)
        matches: list = []
        if symptom is None:
            # Resolution needs the map, so it happens here rather than in the
            # caller: the vocabulary is the map's, and building a second copy
            # of it outside would be a second thing to keep current.
            matches = strategy_mod.resolve(description or "", await code_map.vocabulary())
            if not matches:
                raise LookupError(
                    "nothing in this map's vocabulary matches that description.  "
                    "Name a subject and a value instead, or say it in the map's own "
                    "words -- a status, a rejection reason, a brokerage step."
                )
            symptom = matches[0].term.symptom.model_copy(
                update={
                    "task_id": task_id or strategy_mod.entity_in(description or ""),
                    "observed_diag": observed_diag,
                }
            )
        # The description only ever chooses the first question.  Every hop after
        # it is chosen by what the map opened and the evidence left standing,
        # which is why resolution happens once and outside the walk.
        return await walk(code_map, symptom, budget=budget, settle=settle), matches
    finally:
        await graph_db.close()
