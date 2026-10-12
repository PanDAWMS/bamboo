"""The querier: one question about one PanDA object, answered from the code map, the logs and the database rows.

It reads the map through :mod:`bamboo.codemap` from the tree it is itself imported from.
Put that tree on ``PYTHONPATH``; ``BAMBOO_TREE``, when set, must name the same tree, and the
import fails when ``bamboo`` came from elsewhere.  The usual way that happens: ``python -m``
run from a checkout's root reads that checkout ahead of ``PYTHONPATH``.  A run thus never mixes
the map's code of one commit with the querier of another without saying so.

The PanDA source the map was built from (the corpus) is ``PANDA_PIN``; see
:mod:`bamboo.querier.records`.
"""

import os
import sys
from pathlib import Path

import bamboo

#: the root of the tree ``bamboo`` -- and so the map's code and this package -- was imported from
TREE = Path(bamboo.__file__).resolve().parents[1]

_named = os.environ.get("BAMBOO_TREE")
if _named and Path(_named).resolve() != TREE:
    raise ImportError(f"bamboo was imported from {TREE}, not BAMBOO_TREE={_named}")

# a tree is compared file for file against its git archive: write no __pycache__ into it
sys.dont_write_bytecode = True
