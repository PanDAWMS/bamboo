"""Read the map from a dump (``diff-map --dump``) instead of Neo4j, in this process.

``install(path)`` replaces ``graph_database_client.GraphDatabaseClient`` with a client that carries the dump and
nothing else: a ``CodeMap`` on it reads the dump's own nodes, with the store's rules (``lookup._fragment_nodes``).
Nothing is stored, so the dump stays as it was read for the whole run; a reader that asks the client itself stops.
Every reader looks the client class up at call time (``rerun._writers_of``, ``derive_strategy._investigate``,
``rows._function_prints``), so the swap reaches them.

Each case is read against the map built from the corpus that wrote its log: its own ``PANDA_PIN`` and the dump of
that pin.  The swap is per process: a process started with ``spawn`` would have to call ``install_from_env()`` itself.  None needs to
today -- census_p's ``--jobs`` workers read no map store at all (census_p.py says how that was checked).

The entry that reads the map (``rerun``) calls ``install_from_env()`` first thing in ``main``, so ``MAP_DUMP`` means
the same run with or without a driver that installed it before it; ``described()`` says what was installed, not what
the environment asks for.

There is no map but a dump (2026-10-13: Neo4j keeps only the incident graph).  A run without ``MAP_DUMP`` stops, and
so does a reader in the package that would open a graph client with no dump installed (``require``): the client it
would get is Neo4j's, whose store holds no map, and an empty answer from it reads as "nothing writes it".
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

_INSTALLED: dict[str, str] = {}
_FRAGMENT: list = []  # the MapFragment installed, for the readers that take a section of it whole (``fragment``)


def install(path: str) -> None:
    """Make every graph client in this process read the dump at *path*.  A second call with the same path does nothing."""
    if _INSTALLED.get("path") == path:
        return
    if _INSTALLED:
        raise RuntimeError(f"a map dump is installed already ({_INSTALLED['path']}), not {path}")
    from bamboo.codemap.models import MapFragment
    from bamboo.database import graph_database_client

    data = Path(path).read_bytes()
    raw = json.loads(data)
    # The dump must be the map of the corpus the run reads: check_blobs compares only the files an answer names, and a
    # file no answer names still moves what the map says through placement (a map built on one pin passed it whole on a pin that
    # differs in one file).
    built_on = raw.get("derived_from", "").removeprefix("source-root:")
    pin = os.environ.get("PANDA_PIN", "")
    if not built_on or not pin or Path(built_on).resolve() != Path(pin).resolve():
        raise RuntimeError(f"the map dump {path} was built on {built_on or '(no derived_from)'}, the run reads PANDA_PIN={pin or '(unset)'}")
    fragment = MapFragment.model_validate(raw)

    class _FromDump:
        async def connect(self):
            pass

        async def close(self):
            pass

        def __getattr__(self, name):
            # a store of the dump would write its nodes (ids, valid_for), and Neo4j's client holds no map: neither
            # answers a reader of the client, so it stops rather than reading an empty map as "nothing writes it"
            if name.startswith("__"):
                raise AttributeError(name)
            raise RuntimeError(f"the map dump is read through CodeMap (lookup._fragment_nodes), not the graph client's {name}")

    _FromDump.fragment = fragment  # what CodeMap reads; set here, as the class body would read mapdump.fragment
    graph_database_client.GraphDatabaseClient = _FromDump
    _INSTALLED["path"] = path
    _INSTALLED["sha256"] = hashlib.sha256(data).hexdigest()
    _FRAGMENT[:] = [fragment]


def install_from_env() -> None:
    """``install($MAP_DUMP)``; a run without it stops."""
    path = os.environ.get("MAP_DUMP")
    if not path:
        raise SystemExit("set MAP_DUMP to the dump of the map built on PANDA_PIN: the map is read from a dump, never from Neo4j")
    install(path)


def require(what: str) -> None:
    """Stop before *what* opens a graph client with no dump installed: that client would be Neo4j's."""
    if not _INSTALLED:
        raise RuntimeError(f"{what} reads the map, and no map dump is installed: install_from_env() first (MAP_DUMP)")


def installed() -> dict[str, str]:
    """The installed dump's ``path`` and ``sha256`` (all 64 digits), or ``{}`` before any install."""
    return dict(_INSTALLED)


def fragment():
    """The installed dump as its ``MapFragment``, or ``None`` before any install: for a reader that takes a section whole
    rather than asking the graph client (``rerun._reach_of`` reads ``reach_graph`` and ``junctions``)."""
    return _FRAGMENT[0] if _FRAGMENT else None


def described() -> str:
    """``dump <path> (sha256 <16>)`` or ``the one stored in Neo4j`` -- for an output's header.

    It reads what :func:`install` put in place, with the hash of the bytes it read: a ``MAP_DUMP`` nobody installed is
    not the map a run read.  The second form names the state before any install; no reader gets past :func:`require`
    in it.
    """
    if not _INSTALLED:
        return "the one stored in Neo4j"
    return f"dump {_INSTALLED['path']} (sha256 {_INSTALLED['sha256'][:16]})"
