"""Read a Code Map's dump.

The map is kept as dumps (``build-map --dump``), one per source version, and is
read from the dump itself (:func:`load_dump`, handed to ``lookup.CodeMap``).  No
database holds it: Neo4j holds the incident graph only.  While the map was stored
there, beside the incident graph under labels of its own, every reading was a
lookup by label and property; a dump is read as it is.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from bamboo.codemap.models import MapFragment


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
