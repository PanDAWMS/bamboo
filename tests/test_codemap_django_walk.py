"""The walk a Django plugin makes (orm.plugin) and the typed-message recognizers (msgdispatch), on a small project."""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import pytest

from bamboo.codemap import gates, msgdispatch
from bamboo.codemap.orm.plugin import DjangoCodeMapPlugin


class Plugin(DjangoCodeMapPlugin):
    """A project with a package under src/, agents beside it, one neighbour and one schema it does not own."""

    MAP_ID = "shop"
    PACKAGES = {"shop": "src/shop", "agents": "agents"}
    NEIGHBOURS = {"ai": "assistant"}
    SCHEMAS = {"doma_panda": "panda"}

    def module_context(self, tree: ast.Module, constants):
        return constants, msgdispatch.module_queues(tree)

    def visit_module(self, tree, rel, context) -> None:
        msgdispatch.visit_receivers(self, tree, rel, context[1])

    def visit_node(self, node, func, owner, rel, context) -> None:
        msgdispatch.visit_send(self, node, func, owner, rel, *context)

    def finish(self) -> None:
        msgdispatch.close(self)


MODELS = """
from django.db import models


class Order(models.Model):
    STATUS_CHOICES = [('queued', 'Queued'), ('failed', 'Failed'), ('done', 'Done')]
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='queued')

    class Meta:
        db_table = 'shop_order'
"""

CONSTANTS = """
PANDA_SCHEMA = 'doma_panda'
OTHER_SCHEMA = 'elsewhere'
"""

OPERATIONS = """
import json
from django.contrib.auth.models import User
from shop.broker import ActiveMQConnectionManager
from .constants import PANDA_SCHEMA, OTHER_SCHEMA
from .models import Order

CONTROL = '/queue/control'


def request(pk):
    message = {'msg_type': 'ship', 'id': pk}
    sent = ActiveMQConnectionManager().send_message('/queue/shop.ops', json.dumps(message))
    if not sent:
        Order.objects.filter(pk=pk).update(status='failed')


def stop(username, topic):
    mq = ActiveMQConnectionManager()
    mq.send_message(CONTROL, json.dumps(dict(msg_type='stop')))
    mq.send_message(f'/queue/agent_control.{username}', '{}')
    mq.send_message(topic, '{}')


def forward(message):
    ActiveMQConnectionManager().send_message('/queue/shop.ops', json.dumps(message))


def mail(smtp, msg, channel):
    smtp.send_message(msg)
    channel.send_message('town-square', 'hello')


def raw(cursor, pk):
    cursor.execute(f"UPDATE shop_order SET status = 'done' WHERE id = {pk}")


def jobs(cursor, fields):
    cursor.execute(f'''
        SELECT "jobstatus", "pandaid" FROM "{PANDA_SCHEMA}"."jobsactive4"
        UNION ALL
        SELECT "jobstatus", "pandaid" FROM "{PANDA_SCHEMA}"."jobsarchived4"
    ''')
    cursor.execute("SELECT status FROM doma_panda.jedi_tasks t JOIN doma_panda.jedi_datasets d ON 1=1")
    cursor.execute(f'SELECT 1 FROM "{OTHER_SCHEMA}"."things"')
    cursor.execute(f'SELECT 1 FROM "{fields}"."things"')


def touch(cursor):
    cursor.execute(f"UPDATE {PANDA_SCHEMA}.jedi_tasks SET status = 'x'")
    cursor.execute("DELETE FROM doma_panda.jedi_tasks WHERE 1=0")
    cursor.execute("INSERT INTO doma_panda.jedi_tasks (status) VALUES ('y')")


def withdraw(pk):
    from ai.models import Proposal
    proposal = Proposal.objects.get(pk=pk)
    proposal.status = 'withdrawn'
    proposal.quality = 'poor'
    proposal.save()


def expire(pk):
    from ai.models import Proposal as P
    P.objects.filter(pk=pk).update(status='expired', executor='nobody')
    P.objects.create(status='open')


def deactivate(pk):
    user = User.objects.get(pk=pk)
    user.status = 'gone'
"""

AGENT = """
import os

OPS_QUEUE = os.environ.get('SHOP_OPS_QUEUE', '/queue/shop.ops')


class OpsAgent:
    def on_message(self, frame):
        handler = getattr(self, f'_handle_{frame.msg_type}', None)
        handler(frame)

    def _handle_ship(self, data):
        self.send_message('/topic/events', {'msg_type': 'shipped'})

    def _handle_restock(self, data):
        pass


class Bot:
    def _handle_event(self, raw):
        pass
"""


@pytest.fixture()
def fragment(tmp_path: Path):
    app = tmp_path / "src" / "shop"
    (app / "migrations").mkdir(parents=True)
    (app / "__init__.py").write_text("")
    (app / "models.py").write_text(textwrap.dedent(MODELS))
    (app / "constants.py").write_text(textwrap.dedent(CONSTANTS))
    (app / "operations.py").write_text(textwrap.dedent(OPERATIONS))
    (app / "migrations" / "0001.py").write_text("def f(t):\n    t.status = 'ignored'\n")
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents" / "ops_agent.py").write_text(textwrap.dedent(AGENT))
    plugin = Plugin()
    assert plugin.prepare(tmp_path).startswith("shop ")
    return plugin.run()


def _boundary(fragment, name):
    return next(b for b in fragment.boundaries if b.name == name)


def test_prepare_needs_a_checkout(tmp_path):
    with pytest.raises(FileNotFoundError):
        Plugin().prepare(None)
    with pytest.raises(FileNotFoundError):
        Plugin().prepare(tmp_path)


def test_files_are_spelled_from_their_package(fragment):
    junction = next(j for j in fragment.junctions if j.owner.endswith("::request"))
    assert junction.owner == "shop/operations.py::request"
    assert junction.anchor.file == "shop/operations.py"
    (branch,) = junction.branches
    assert branch.outcome == "failed"
    assert branch.path_condition[0].startswith("not sent")
    (subject,) = fragment.subjects
    assert subject.name == "Order.status"
    assert subject.vocabulary == ["queued", "failed", "done"]
    assert subject.metadata["table"] == "shop_order"
    assert not any("migrations" in j.owner for j in fragment.junctions)


def test_an_update_in_an_f_string_is_read_once(fragment):
    junction = next(j for j in fragment.junctions if j.owner.endswith("::raw"))
    assert [(b.outcome, b.tags) for b in junction.branches] == [
        ("done", ["sql.update"])
    ]


# ---------------------------------------------------------------- rows other systems declare


def test_a_write_of_a_neighbours_model_is_a_boundary_named_by_the_class(fragment):
    shared = _boundary(fragment, "shop->assistant::ai.models.Proposal")
    assert (shared.system, shared.transport, shared.kind, shared.direction) == (
        "assistant",
        "shared_table",
        "reports_state",
        "outbound",
    )
    # the class's own name, not the one it is imported under
    assert shared.interface == "ai.models.Proposal"
    # its fields are not known here: only one named like a state is taken
    assert shared.handed_over == ["status"]
    assert shared.operations == ["UPDATE", "INSERT"]
    assert shared.metadata["values"] == {"status": ["withdrawn", "expired", "open"]}
    assert shared.calls_out == [
        "shop/operations.py::withdraw",
        "shop/operations.py::expire",
    ]
    # neither a junction of this map nor, for a package that is no neighbour, a boundary
    assert not any("Proposal" in j.subject for j in fragment.junctions)
    assert not any("User" in b.name for b in fragment.boundaries)


def test_a_table_of_a_schema_another_system_owns_is_a_boundary(fragment):
    active = _boundary(fragment, "shop->panda::doma_panda.jobsactive4")
    assert (active.system, active.transport, active.kind) == (
        "panda",
        "shared_table",
        "reports_state",
    )
    assert active.operations == ["SELECT"]
    assert active.carried_values == ["jobstatus", "pandaid"]
    assert active.calls_out == ["shop/operations.py::jobs"]
    assert _boundary(fragment, "shop->panda::doma_panda.jobsarchived4")
    assert _boundary(fragment, "shop->panda::doma_panda.jedi_datasets")
    tasks = _boundary(fragment, "shop->panda::doma_panda.jedi_tasks")
    assert sorted(tasks.operations) == ["DELETE", "INSERT", "SELECT", "UPDATE"]
    assert tasks.calls_out == ["shop/operations.py::jobs", "shop/operations.py::touch"]
    # a schema nobody is said to own, and one the code fills at run time, are no boundary
    assert not any("things" in b.name for b in fragment.boundaries)


# ---------------------------------------------------------------- messages


def test_a_send_is_a_boundary_named_by_its_destination_and_an_edge_per_type(fragment):
    ops = _boundary(fragment, "shop->activemq::/queue/shop.ops")
    assert (ops.transport, ops.direction, ops.kind) == (
        "message",
        "outbound",
        "transports_causation",
    )
    assert ops.calls_out == [
        "shop/operations.py::request",
        "shop/operations.py::forward",
    ]
    assert ops.handed_over == ["ship"]  # forward's message is its caller's
    control = _boundary(fragment, "shop->activemq::/queue/control")
    assert control.handed_over == ["stop"]
    assert _boundary(fragment, "shop->activemq::/queue/agent_control.{}")
    sends = {(e.sender, e.msg_type) for e in fragment.message_edges}
    assert ("shop/operations.py::request", "ship") in sends
    assert not any(sender.endswith("::forward") for sender, _type in sends)


def test_an_unsettled_destination_is_kept_only_for_the_brokers_connection(fragment):
    runtime = _boundary(fragment, f"shop->activemq::{msgdispatch.RUNTIME_DESTINATION}")
    assert runtime.calls_out == ["shop/operations.py::stop"]
    assert not any(
        "::mail" in owner for b in fragment.boundaries for owner in b.calls_out
    )


def test_a_dispatching_class_receives_from_the_queue_its_module_declares(fragment):
    inbound = _boundary(fragment, "activemq->shop::/queue/shop.ops")
    assert (inbound.direction, inbound.transport) == ("inbound", "message")
    assert inbound.metadata["env"] == "SHOP_OPS_QUEUE"
    assert inbound.carried_values == ["restock", "ship"]
    assert inbound.calls_out == []
    rows = {r.msg_type: r for r in fragment.message_receivers}
    assert set(rows) == {"restock", "ship"}
    assert rows["ship"].receiver == "agents/ops_agent.py::OpsAgent"
    assert rows["ship"].key == "msg_type"
    assert rows["ship"].handled_by == ["agents/ops_agent.py::OpsAgent._handle_ship"]
    assert rows["ship"].boundary == inbound.name
    assert not any("Bot" in b.name for b in fragment.boundaries)


def test_a_send_and_its_receiver_in_one_map_are_tied(fragment):
    rows = {r.msg_type: r for r in fragment.message_receivers}
    (place,) = rows["ship"].senders
    assert place.startswith("shop/operations.py::request:")
    assert rows["restock"].senders == []
    edge = next(e for e in fragment.message_edges if e.msg_type == "ship")
    assert edge.receiver == "agents/ops_agent.py::OpsAgent._handle_ship"
    # sent to a topic nothing here receives from: an edge with no receiver
    shipped = next(e for e in fragment.message_edges if e.msg_type == "shipped")
    assert shipped.sender == "agents/ops_agent.py::OpsAgent._handle_ship"
    assert shipped.receiver == ""


def test_another_convention_is_read_by_the_same_rule(tmp_path):
    other = msgdispatch.Convention(
        broker="kafka", send="publish", type_key="kind", handler_prefix="on_"
    )

    class Other(Plugin):
        PACKAGES = {"svc": "svc"}

        def visit_module(self, tree, rel, context) -> None:
            msgdispatch.visit_receivers(self, tree, rel, context[1], other)

        def visit_node(self, node, func, owner, rel, context) -> None:
            msgdispatch.visit_send(self, node, func, owner, rel, *context, other)

    (tmp_path / "svc").mkdir()
    (tmp_path / "svc" / "worker.py").write_text(textwrap.dedent("""
            IN = '/queue/in'


            class Worker:
                def run(self, m):
                    getattr(self, f'on_{m.kind}')(m)

                def on_ping(self, m):
                    self.bus.publish('/queue/out', {'kind': 'pong'})
            """))
    plugin = Other()
    plugin.prepare(tmp_path)
    fragment = plugin.run()
    assert [(r.key, r.msg_type) for r in fragment.message_receivers] == [
        ("kind", "ping")
    ]
    assert _boundary(fragment, "shop->kafka::/queue/out").handed_over == ["pong"]


def test_generic_gates_pass(fragment):
    results = {r.gate: r for r in gates.run_all(fragment)}
    for name in (
        "map-references-resolve",
        "branch-positions-are-known",
        "map-identities-are-distinct",
        "structural-attribution-agrees",
    ):
        assert results[name].passed, results[name].failures
