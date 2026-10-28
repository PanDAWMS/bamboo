"""Read a Code Map's dump, or load a fragment into a graph backend.

The map is kept as dumps (``build-map --dump``), one per source version, and is
read from the dump itself (:func:`load_dump`, handed to ``lookup.CodeMap``); no
reader asks a graph backend for it.  Neo4j holds the incident
graph only.  When the map was stored there it shared the database under its own
labels, and the separation was load-bearing: the Code Map is derived and
disposable, while the incident graph holds human-validated knowledge that cannot
be regenerated.  Rebuilding one must never touch the other, which is why writes
go through ``clear_map`` / ``merge_map_node`` instead of the wholesale
``clear_all`` -- and why ``clear_map`` is still how a map left in Neo4j is taken
out.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

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


def load_dump(path: Path, source_root: Optional[Path] = None) -> MapFragment:
    """The map ``build-map --dump`` wrote at *path*, for ``lookup.CodeMap``.

    Where *source_root* is given and the dump names the tree it was built from,
    a dump of another tree is refused -- the readers walk *source_root* beside
    the map, and a map of a different corpus answers about lines that are not
    there.  A build of a git checkout is stamped with its description instead
    (``git:...``), which says no path to compare, and is taken as it is.
    """
    fragment = MapFragment.model_validate_json(Path(path).read_text())
    built_on = fragment.derived_from.removeprefix("source-root:")
    if (
        source_root is not None
        and built_on != fragment.derived_from
        and Path(built_on).resolve() != Path(source_root).resolve()
    ):
        raise ValueError(
            f"the map dump {path} was built on {built_on}, not on {source_root}"
        )
    return fragment
