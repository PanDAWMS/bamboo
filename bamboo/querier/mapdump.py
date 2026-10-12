"""Read the map from a dump (``diff-map --dump``) instead of Neo4j, in this process.

``install(path)`` replaces ``graph_database_client.GraphDatabaseClient`` with the in-memory backend, filled with the
dump through the tree's own ``store_fragment`` on each connect.  Every reader looks the client class up at call time
(``rerun._writers_of``, ``derive_strategy._investigate``, ``rows._function_prints``), so the swap reaches them.

Each case is read against the map built from the corpus that wrote its log: its own ``PANDA_PIN`` and the dump of
that pin.  The swap is per process: a process started with ``spawn`` would have to call ``install_from_env()`` itself.  None needs to
today -- census_p's ``--jobs`` workers read no map store at all (census_p.py says how that was checked).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

_INSTALLED: dict[str, str] = {}


def install(path: str) -> None:
    """Make every graph client in this process read the dump at *path*.  A second call with the same path does nothing."""
    if _INSTALLED.get("path") == path:
        return
    if _INSTALLED:
        raise RuntimeError(f"a map dump is installed already ({_INSTALLED['path']}), not {path}")
    from bamboo.codemap.models import MapFragment
    from bamboo.codemap.store import store_fragment
    from bamboo.database import graph_database_client
    from bamboo.database.backends.examples.in_memory_backend import InMemoryGraphBackend

    raw = json.loads(Path(path).read_text())
    # The dump must be the map of the corpus the run reads: check_blobs compares only the files an answer names, and a
    # file no answer names still moves what the map says through placement (a map built on one pin passed it whole on a pin that
    # differs in one file).
    built_on = raw.get("derived_from", "").removeprefix("source-root:")
    pin = os.environ.get("PANDA_PIN", "")
    if not built_on or not pin or Path(built_on).resolve() != Path(pin).resolve():
        raise RuntimeError(f"the map dump {path} was built on {built_on or '(no derived_from)'}, the run reads PANDA_PIN={pin or '(unset)'}")
    fragment = MapFragment.model_validate(raw)

    class _FromDump(InMemoryGraphBackend):
        async def connect(self):
            await super().connect()
            await store_fragment(fragment, self)

        async def close(self):
            pass

    graph_database_client.GraphDatabaseClient = _FromDump
    _INSTALLED["path"] = path


def install_from_env() -> None:
    """``install($MAP_DUMP)`` where it is set and not empty; the map in Neo4j otherwise."""
    path = os.environ.get("MAP_DUMP")
    if path:
        install(path)


def described() -> str:
    """``dump <path> (sha256 <16>)`` or ``the one stored in Neo4j`` -- for an output's header."""
    path = os.environ.get("MAP_DUMP")
    if not path:
        return "the one stored in Neo4j"
    return f"dump {path} (sha256 {hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]})"
