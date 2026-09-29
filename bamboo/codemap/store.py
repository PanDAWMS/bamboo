"""Persist a Code Map fragment to the graph database.

The Code Map shares Neo4j with the incident graph but occupies its own labels.
That separation is load-bearing rather than cosmetic: the Code Map is derived
and disposable, rebuilt whenever the source version moves, while the incident
graph holds human-validated knowledge that cannot be regenerated.  Rebuilding
one must never touch the other, which is why writes go through
``clear_map`` / ``merge_map_node`` instead of the wholesale ``clear_all``.
"""

from __future__ import annotations

import logging

from bamboo.codemap.models import MAP_NODE_GROUPS, MapFragment
from bamboo.database.graph_database_client import GraphDatabaseClient

logger = logging.getLogger(__name__)


async def store_fragment(
    fragment: MapFragment,
    graph_db: GraphDatabaseClient,
    replace_map: bool = True,
) -> dict[str, int]:
    """Write *fragment* to the graph database.

    The map is one build of one corpus.  Clearing only this build's own stamp
    was meant to let several versions coexist so an old incident could be
    explained against the code that was running then -- but nothing reads the
    map that way, and what it produced instead was a 54-node fragment of a
    different source tree sitting under the same ``map_id``, answering queries
    about a corpus it was never built from.  A build that cannot be told from
    its neighbours is worse than one that replaces them.

    Reading an old build back is still possible and still per-version: the
    ``version`` argument on :meth:`clear_map` and :meth:`find_map_nodes` is
    untouched, and ``valid_for`` still records every build a node survived.
    What changed is only which nodes a *write* is allowed to leave behind.

    Args:
        fragment:    What a plugin produced.
        graph_db:    Connected client.
        replace_map: Drop every node of this ``map_id`` first, whatever build
            wrote it, so the stored map is exactly one build of one corpus.
            ``False`` writes alongside what is there, which is for a caller
            assembling one map from several fragments.

    Returns:
        Counts of what was written, per node kind.
    """
    written = {"cleared": 0, **dict.fromkeys(MAP_NODE_GROUPS, 0)}

    if replace_map:
        removed = await graph_db.clear_map(fragment.map_id)
        logger.info(
            "store_fragment: cleared %d existing node(s) of map %s "
            "(any build) before writing %s",
            removed, fragment.map_id, fragment.derived_from,
        )
        written["cleared"] = removed

    for group in MAP_NODE_GROUPS:
        for node in getattr(fragment, group):
            await graph_db.merge_map_node(node)
            written[group] += 1

    logger.info("store_fragment: wrote %r", written)
    return written
