"""swf-monitor Code Map plugin: packages under src/, messages both ways, PanDA's tables."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from bamboo.codemap import gates
from bamboo.codemap.factory import available_map_ids, get_code_map_plugin
from bamboo.codemap.monitor.plugin import MonitorCodeMapPlugin

MODELS = """
from django.db import models


class PandaTaskOperation(models.Model):
    STATUS_CHOICES = [('queued', 'Queued'), ('failed', 'Failed'), ('done', 'Done')]
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='queued')

    class Meta:
        db_table = 'swf_panda_task_operation'
"""

CONSTANTS = """
PANDA_SCHEMA = 'doma_panda'
"""

OPERATIONS = """
import json
from monitor_app.activemq_connection import ActiveMQConnectionManager
from .constants import PANDA_SCHEMA
from .models import PandaTaskOperation


def request(pk):
    message = {'msg_type': 'panda_task_operation', 'id': pk}
    sent = ActiveMQConnectionManager().send_message('/queue/epicprod.ops', json.dumps(message))
    if not sent:
        PandaTaskOperation.objects.filter(pk=pk).update(status='failed')


def jobs(cursor):
    cursor.execute(f'SELECT "jobstatus" FROM "{PANDA_SCHEMA}"."jobsactive4"')
    cursor.execute("SELECT request_id FROM doma_idds.requests")


def adopt(pk):
    from pcs.models import ProdTask
    ProdTask.objects.filter(pk=pk).update(status='adopted')
"""

AGENT = """
import os

OPS_QUEUE = os.environ.get('EPICPROD_OPS_QUEUE', '/queue/epicprod.ops')


class OpsAgent:
    def on_message(self, frame):
        msg_type = frame.headers['msg_type']
        handler = getattr(self, f'_handle_{msg_type}', None)
        handler(frame)

    def _handle_panda_task_operation(self, data):
        self.send_message('/topic/epictopic', {'msg_type': 'panda_task_operation_done'})

    def _handle_submit_task(self, data):
        pass
"""


@pytest.fixture()
def fragment(tmp_path: Path):
    app = tmp_path / "src" / "monitor_app"
    app.mkdir(parents=True)
    (app / "__init__.py").write_text("")
    (app / "models.py").write_text(textwrap.dedent(MODELS))
    (app / "constants.py").write_text(textwrap.dedent(CONSTANTS))
    (app / "operations.py").write_text(textwrap.dedent(OPERATIONS))
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents" / "ops_agent.py").write_text(textwrap.dedent(AGENT))
    plugin = MonitorCodeMapPlugin()
    assert plugin.prepare(tmp_path).startswith("swf-monitor ")
    return plugin.run()


def _boundary(fragment, name):
    return next(b for b in fragment.boundaries if b.name == name)


def test_registered():
    assert "swf-monitor" in available_map_ids()
    assert isinstance(get_code_map_plugin("swf-monitor"), MonitorCodeMapPlugin)


def test_prepare_needs_a_checkout(tmp_path):
    with pytest.raises(FileNotFoundError):
        MonitorCodeMapPlugin().prepare(None)
    with pytest.raises(FileNotFoundError):
        MonitorCodeMapPlugin().prepare(tmp_path)


def test_a_package_under_src_is_spelled_from_the_package(fragment):
    (junction,) = fragment.junctions
    assert junction.owner == "monitor_app/operations.py::request"
    (subject,) = fragment.subjects
    assert subject.name == "PandaTaskOperation.status"
    assert subject.metadata["table"] == "swf_panda_task_operation"


def test_a_send_and_the_agent_that_takes_it(fragment):
    sent = _boundary(fragment, "swf-monitor->activemq::/queue/epicprod.ops")
    assert sent.handed_over == ["panda_task_operation"]
    taken = _boundary(fragment, "activemq->swf-monitor::/queue/epicprod.ops")
    assert taken.carried_values == ["panda_task_operation", "submit_task"]
    assert taken.metadata["env"] == "EPICPROD_OPS_QUEUE"
    rows = {r.msg_type: r for r in fragment.message_receivers}
    assert rows["panda_task_operation"].handled_by == [
        "agents/ops_agent.py::OpsAgent._handle_panda_task_operation"
    ]
    assert (
        rows["panda_task_operation"]
        .senders[0]
        .startswith("monitor_app/operations.py::request:")
    )
    assert rows["submit_task"].senders == []
    # the agent's answer goes to a topic nothing in this map receives from
    answer = _boundary(fragment, "swf-monitor->activemq::/topic/epictopic")
    assert answer.calls_out == [
        "agents/ops_agent.py::OpsAgent._handle_panda_task_operation"
    ]


def test_the_tables_of_the_systems_it_monitors(fragment):
    panda = _boundary(fragment, "swf-monitor->panda::doma_panda.jobsactive4")
    assert (panda.transport, panda.operations, panda.carried_values) == (
        "shared_table",
        ["SELECT"],
        ["jobstatus"],
    )
    idds = _boundary(fragment, "swf-monitor->idds::doma_idds.requests")
    assert idds.system == "idds"


def test_a_write_of_an_installed_applications_model(fragment):
    shared = _boundary(fragment, "swf-monitor->swf-epicprod::pcs.models.ProdTask")
    assert shared.handed_over == ["status"]
    assert shared.calls_out == ["monitor_app/operations.py::adopt"]


def test_generic_gates_pass(fragment):
    results = {r.gate: r for r in gates.run_all(fragment)}
    for name in (
        "map-references-resolve",
        "branch-positions-are-known",
        "map-identities-are-distinct",
    ):
        assert results[name].passed, results[name].failures
