"""swf-epicprod Code Map plugin, built on the shared tools for a Django code base.

swf-epicprod is a Django project, so where PanDA's recognizers read SQL text
and infer the row's class from how it is used, most writes here go through
the ORM and *name* their class.  Models, write forms, attribution, values and
conditions are :mod:`bamboo.codemap.orm`'s, and the walk over a checkout --
with raw ``UPDATE`` text, the rows other systems declare and the tables of
schemas they own -- is its
:class:`~bamboo.codemap.orm.plugin.DjangoCodeMapPlugin`.  This plugin states
what is swf-epicprod's own:

* which packages are read (:data:`PACKAGES`),
* what it is installed beside (:data:`NEIGHBOURS`): swf-monitor's Django
  project hosts it, and a write of one of swf-monitor's models is a boundary
  to it,
* whose tables it reads directly (:data:`SCHEMAS`): PanDA's,
* calls into PanDA (``pandaclient`` ``Client.*`` and ``requests`` to
  ``PANDA_URL``) as outbound boundaries, named by the client's method or the
  URL's path,
* its sends through the broker (:mod:`bamboo.codemap.msgdispatch`): the web
  code asks the production operations agent for what needs credentials by a
  message, and the agent is swf-monitor's.

Values are read with the whole corpus's calls handed to the shared resolver,
so a constant another module declares (``PROPAGATION_STATES`` imported from
``pcs.models``) settles the way a local one does.  No module of another map's
plugin is loaded: what this map states of the far side is in the fields every
map has, and :mod:`bamboo.codemap.bind` binds it.

Not done yet: the read side as statements (``ReadSiteNode``), log sites,
reach graph and entry points, writes through ``save(update_fields=...)``
helpers that take the field name as data, and calls of swf-monitor's functions
in the same process.

Scope: ``pcs/``, ``swf_epicprod/`` and ``tools/`` are read; ``scripts/``,
``pilot/``, migrations and tests are not, so PanDA use from those is not in
the map.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from bamboo.codemap import msgdispatch
from bamboo.codemap.orm.plugin import DjangoCodeMapPlugin
from bamboo.codemap.orm.schema import dotted

MAP_ID = "swf-epicprod"

#: Top-level packages analysed, in the order they are walked.
PACKAGES = ("pcs", "swf_epicprod", "tools")


@dataclass
class ModuleContext:
    """What a module says once."""

    #: the names the module imports ``pandaclient`` under
    panda_aliases: set[str]
    constants: dict[str, list[str]]
    queues: msgdispatch.Queues


class EpicprodCodeMapPlugin(DjangoCodeMapPlugin):
    """Builds the swf-epicprod fragment from a checkout."""

    MAP_ID = MAP_ID
    PACKAGES = {package: package for package in PACKAGES}
    #: swf-epicprod is installed into swf-monitor's Django project and writes some of its rows.
    NEIGHBOURS = {
        "monitor_app": "swf-monitor",
        "ai": "swf-monitor",
        "swf_alarms": "swf-monitor",
    }
    #: The PanDA database, which its code reads beside the server's API.
    SCHEMAS = {"doma_panda": "panda"}

    def module_context(self, tree: ast.Module, constants) -> ModuleContext:
        panda_aliases: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.startswith("pandaclient"):
                        panda_aliases.add(a.asname or a.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "pandaclient"
            ):
                panda_aliases.update(a.asname or a.name for a in node.names)
        return ModuleContext(panda_aliases, constants, msgdispatch.module_queues(tree))

    def visit_node(
        self, node, func, owner: str, rel: str, context: ModuleContext
    ) -> None:
        if not isinstance(node, ast.Call):
            return
        msgdispatch.visit_send(
            self, node, func, owner, rel, context.constants, context.queues
        )
        name = dotted(node.func) or ""
        head = name.split(".")[0]
        if head in context.panda_aliases or name.startswith("pandaclient."):
            method = name.split(".")[-1]
            if not method.startswith(
                "_"
            ):  # the client's own plumbing is not a PanDA interface
                self._panda(
                    owner, f"pandaclient.Client.{method}", "pandaclient", rel, node
                )
        elif name in ("requests.get", "requests.post", "requests.put") and node.args:
            url = node.args[0]
            if isinstance(url, ast.JoinedStr) and any(
                isinstance(v, ast.FormattedValue) and "PANDA" in ast.unparse(v.value)
                for v in url.values
            ):
                path = "".join(
                    str(v.value) for v in url.values if isinstance(v, ast.Constant)
                )
                self._panda(
                    owner,
                    path or "<panda url>",
                    "http",
                    rel,
                    node,
                    verb=name.split(".")[-1].upper(),
                )

    def finish(self) -> None:
        msgdispatch.close(self)

    def _panda(
        self,
        owner: str,
        interface: str,
        client: str,
        rel: str,
        call: ast.Call,
        verb: str = "",
    ) -> None:
        self.boundary(
            owner,
            "panda",
            interface,
            "http",
            rel,
            call.lineno,
            verb=verb,
            metadata={"client": client},
        )
