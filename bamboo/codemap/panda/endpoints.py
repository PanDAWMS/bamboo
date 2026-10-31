"""The API's endpoints, and the statements reading rows each one's calls reach (:class:`~bamboo.codemap.models.Endpoint`).

The live-case harness's EP (to-map-33, 10-28; to-map-44, 10-31): a reader asking PanDA for an object's rows through
its API, rather than the database, needs per endpoint the rows it reads -- their tables, the terms naming them, and
what of the request fills those.

An endpoint is a function the server runs for a URL ``/api/<version>/<module>/<method>``.  The server's routing table
names the module for each ``<module>`` -- ``module_mapping`` in :data:`SERVER`, a literal mapping per version, its
``"module"`` a name the server imports (``from pandaserver.api.v1 import credential_management_api as cred_api_v1``
under ``"creds"``) -- and every public function defined in that module is a method, but ``init_task_buffer``
(``extract_allowed_methods``): undecorated ones too, ``workflow/submit_workflow_raw_request``.  The legacy
``/server/panda/<method>``, run from the server's own globals, is not read.

From each, the reach record's closure (:meth:`~bamboo.codemap.reach.ReachGraph.closure`, as a mover's reach walks
it), and each read site whose statement a function of it holds -- placed by its anchor, so a method's owner spelled
with or without its class does not matter -- with its read terms (:attr:`KeyFrom.reads`).  A statement past a hole of
the closure is not there: :attr:`Endpoint.holes` says where one may be.
"""

from __future__ import annotations

import ast
import logging
from typing import Iterator, Optional

from bamboo.codemap import reach
from bamboo.codemap.models import (
    Endpoint,
    EndpointRead,
    EndpointTerm,
    KeyFrom,
    MapFragment,
    ReadSiteNode,
    SourceModule,
)

logger = logging.getLogger(__name__)

#: The file holding the server's routing table, and the function holding it.
SERVER = "pandaserver/server/panda.py"
ROUTING = "module_mapping"
#: A module's public functions the server does not route (``extract_allowed_methods``).
NOT_ROUTED = frozenset({"init_task_buffer"})


def routes(modules: list[SourceModule]) -> list[tuple[str, str, str]]:
    """``(version, route, owner)`` per endpoint, ``route`` ``<module>/<method>``; none where :data:`SERVER`'s
    routing table is not read."""
    by_path = {module.rel_path: module for module in modules}
    server = by_path.get(SERVER)
    if server is None:
        return []
    imported = _imported(server.tree)
    found = []
    for version, name, held in _table(server.tree):
        module = by_path.get(imported.get(held, ""))
        if module is None:
            continue
        for node in module.tree.body:
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and not node.name.startswith("_")
                and node.name not in NOT_ROUTED
            ):
                found.append(
                    (version, f"{name}/{node.name}", f"{module.rel_path}::{node.name}")
                )
    return list(dict.fromkeys(found))


def _imported(tree: ast.Module) -> dict[str, str]:
    """``{name: file}`` for each module the file imports by name: ``from a.b import c as d`` gives ``d: a/b/c.py``."""
    found = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for alias in node.names:
                found[alias.asname or alias.name] = (
                    f"{node.module}.{alias.name}".replace(".", "/") + ".py"
                )
    return found


def _table(tree: ast.Module) -> Iterator[tuple[str, str, str]]:
    """``(version, module name, the name its module is held by)`` per entry of :data:`ROUTING`'s mapping: the dict
    literal of versions, each a dict literal of module names, each a dict literal whose ``"module"`` is a name."""
    for node in tree.body:
        if not (isinstance(node, ast.FunctionDef) and node.name == ROUTING):
            continue
        for held in ast.walk(node):
            if not isinstance(held, ast.Dict):
                continue
            for version, entries in zip(held.keys, held.values, strict=True):
                if not (_text(version) and isinstance(entries, ast.Dict)):
                    continue
                for name, entry in zip(entries.keys, entries.values, strict=True):
                    module = _value(entry, "module") if _text(name) else None
                    if isinstance(module, ast.Name):
                        yield version.value, name.value, module.id  # type: ignore[union-attr]


def _text(node: Optional[ast.expr]) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def _value(node: ast.expr, key: str) -> Optional[ast.expr]:
    if isinstance(node, ast.Dict):
        for one, value in zip(node.keys, node.values, strict=True):
            if isinstance(one, ast.Constant) and one.value == key:
                return value
    return None


def attach(fragment: MapFragment, modules: list[SourceModule]) -> int:
    """Set *fragment*'s :attr:`MapFragment.endpoints`; returns how many statements they read, counted per endpoint."""
    record = reach.ReachGraph(fragment.reach_graph)
    sites: dict[str, list[ReadSiteNode]] = {}
    for site in fragment.read_sites:
        if site.anchor is None:
            continue
        holder = record.function_at(site.anchor.file, site.anchor.line_start)
        if holder is not None:
            sites.setdefault(holder, []).append(site)
    terms: dict[tuple[str, int], list[KeyFrom]] = {}
    for key_from in fragment.key_froms:
        if key_from.reads:
            terms.setdefault((key_from.owner, key_from.statement), []).append(key_from)
    found = []
    for version, route, owner in routes(modules):
        if owner not in record.rows:
            logger.warning(
                "endpoints: %s's function %s is no row of the reach record",
                route,
                owner,
            )
            continue
        came = record.closure([owner])
        holes, reads = [], {}
        for node in came:
            function, cls = node
            holes.extend(
                f"{function}:{hole.line}"
                for hole in record.rows[function].holes
                if not hole.for_classes or cls in hole.for_classes
            )
            for site in sites.get(function, ()):
                if site.name in reads:
                    continue
                reads[site.name] = EndpointRead(
                    site=site.name,
                    function=function,
                    tables=sorted({table.lower() for table in site.tables}),
                    terms=[
                        EndpointTerm(
                            term=key_from.term, table=key_from.table, attr=key_from.attr
                        )
                        for key_from in terms.get(
                            (site.owner, site.anchor.line_start), ()
                        )  # type: ignore[union-attr]
                    ],
                    way=reach.way(came, node),
                )
        found.append(
            Endpoint(
                route=route,
                version=version,
                function=owner,
                closure=len({function for function, _cls in came}),
                holes=list(dict.fromkeys(holes)),
                reads=list(reads.values()),
            )
        )
    fragment.endpoints = found
    return sum(len(endpoint.reads) for endpoint in found)
