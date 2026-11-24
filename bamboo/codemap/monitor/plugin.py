"""swf-monitor Code Map plugin, built on the shared tools for a Django code base.

swf-monitor is the Django project that hosts the ePIC production
applications: its own apps (``monitor_app``, ``ai``), the alarm engine
(``swf_alarms``) and the agents that hold the credentials (``agents/``).
Models, write forms, values and conditions are :mod:`bamboo.codemap.orm`'s,
and the walk over a checkout -- with raw ``UPDATE`` text, the rows other
systems declare and the tables of schemas they own -- is its
:class:`~bamboo.codemap.orm.plugin.DjangoCodeMapPlugin`.  This plugin states
what is swf-monitor's own:

* which packages are read and where a checkout keeps them (:data:`PACKAGES`),
* what is installed into it (:data:`NEIGHBOURS`): the production applications
  of swf-epicprod, whose rows its code could write,
* whose tables it reads directly (:data:`SCHEMAS`): PanDA's and iDDS's, which
  the monitoring pages query beside the servers' APIs,
* messages through its broker connection, both ways
  (:mod:`bamboo.codemap.msgdispatch`): a send is an outbound boundary named
  by its destination, and an agent that hands a message to
  ``_handle_<msg_type>`` an inbound one named by its queue, with a receiver
  row per type.

No module of another map's plugin is loaded: what this map states of the far
side is in the fields every map has, and :mod:`bamboo.codemap.bind` binds it.

Not done yet: the read side as statements, log sites, reach graph and entry
points; the REST and MCP endpoints other systems reach this one through; the
message a caller builds and hands to the function that sends it.

Scope: ``src/monitor_app``, ``src/ai``, ``src/swf_monitor_project``,
``alarms/swf_alarms`` and ``agents`` are read; ``scripts/``, the management
commands outside ``src/``, migrations and tests are not.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from bamboo.codemap import msgdispatch
from bamboo.codemap.orm.plugin import DjangoCodeMapPlugin

MAP_ID = "swf-monitor"

#: Top-level package -> its directory in a checkout, in the order they are walked.
PACKAGES = {
    "monitor_app": "src/monitor_app",
    "ai": "src/ai",
    "swf_monitor_project": "src/swf_monitor_project",
    "swf_alarms": "alarms/swf_alarms",
    "agents": "agents",
}


@dataclass
class ModuleContext:
    """What a module says once."""

    constants: dict[str, list[str]]
    queues: msgdispatch.Queues


class MonitorCodeMapPlugin(DjangoCodeMapPlugin):
    """Builds the swf-monitor fragment from a checkout."""

    MAP_ID = MAP_ID
    PACKAGES = PACKAGES
    #: The production applications installed into this project, whose rows this code could write.
    NEIGHBOURS = {"pcs": "swf-epicprod", "swf_epicprod": "swf-epicprod"}
    #: The databases of the systems it monitors.
    SCHEMAS = {"doma_panda": "panda", "doma_idds": "idds"}

    def module_context(self, tree: ast.Module, constants) -> ModuleContext:
        return ModuleContext(constants, msgdispatch.module_queues(tree))

    def visit_module(self, tree: ast.Module, rel: str, context: ModuleContext) -> None:
        msgdispatch.visit_receivers(self, tree, rel, context.queues)

    def visit_node(
        self, node, func, owner: str, rel: str, context: ModuleContext
    ) -> None:
        msgdispatch.visit_send(
            self, node, func, owner, rel, context.constants, context.queues
        )

    def finish(self) -> None:
        msgdispatch.close(self)
