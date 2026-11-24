"""Binding maps: a server, a client library, and two applications, each built by hand from the fields a map has.

No plugin is run: binding reads maps, whatever built them.  ``server`` is shaped as PanDA's plugin shapes a map
(endpoints, inbound boundaries named ``module::function``, entities with qualified tables, receivers with no
queue), ``client`` as a library's, ``portal`` and ``console`` as an application's.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from bamboo.codemap import bind as B
from bamboo.codemap.models import (
    BoundaryNode,
    Endpoint,
    EntityNode,
    MapFragment,
    MessageEdge,
    MessageReceiver,
    SubjectNode,
    TableRecord,
)
from bamboo.scripts.bind_maps import main as bind_maps
from bamboo.scripts.build_map import write_dump


def _boundary(map_id: str, system: str, interface: str, **fields) -> BoundaryNode:
    return BoundaryNode(
        name=fields.pop("name", f"{map_id}:{system}:{interface}"),
        map_id=map_id,
        derived_from=f"{map_id} 1",
        system=system,
        interface=interface,
        **fields,
    )


def _subject(map_id: str, cls: str, attribute: str, **fields) -> SubjectNode:
    return SubjectNode(
        name=f"{cls}.{attribute}",
        map_id=map_id,
        derived_from=f"{map_id} 1",
        spec_class=cls,
        attribute=attribute,
        **fields,
    )


def server() -> MapFragment:
    receiver = "srv/msg/status.py::StatusPlugin.process"
    return MapFragment(
        map_id="server",
        derived_from="server 1",
        endpoints=[
            Endpoint(route="creds/get_access_token", version="v1", function="srv/api/v1/creds_api.py::get_access_token"),
            Endpoint(route="job/get_description", version="v1", function="srv/api/v1/job_api.py::get_description"),
        ],
        boundaries=[
            _boundary("server", "client", "creds_api::get_access_token", carried_values=["client_name"]),
            _boundary(
                "server", "rucio", "srv/ddm.py::RucioAPI.register_dataset",
                direction="outbound", calls_out=["rucio.client.Client.add_dataset"],
            ),
            _boundary("server", "deft", "ATLAS_DEFT.T_TASK", transport="shared_table", carried_values=["STATUS"]),
            _boundary("server", "", receiver, transport="message", kind="transports_causation", name="server::in"),
        ],
        entities=[
            EntityNode(
                name="server::TaskSpec", map_id="server", derived_from="server 1", spec_class="TaskSpec",
                tables=["ATLAS_PANDA.JEDI_Tasks"], updated_by=["srv/db.py::set_status"],
            ),
        ],
        subjects=[_subject("server", "TaskSpec", "status", vocabulary=["running", "done"])],
        tables=[TableRecord(name="jobsarchived4", rows_of="JobSpec", schema_names=["ATLAS_PANDAARCH"])],
        message_receivers=[
            MessageReceiver(receiver=receiver, key="msg_type", msg_type="task_status", boundary="server::in"),
            MessageReceiver(
                receiver=receiver, key="msg_type", msg_type="job_status", senders=["srv/proxy.py::update:806"]
            ),
        ],
    )  # fmt: skip


def client() -> MapFragment:
    return MapFragment(
        map_id="client",
        derived_from="client 1",
        boundaries=[
            _boundary(
                "client", "server", "clientlib/Client.py::get_job_status",
                direction="outbound", calls_out=["POST /api/v1/job/get_description"],
            ),
            _boundary(
                "client", "server", "clientlib/Client.py::kill",
                direction="outbound", calls_out=["https://server.example:25443/api/v1/job/kill?x=1"],
            ),
        ],
    )  # fmt: skip


def portal() -> MapFragment:
    """An application that names a boundary by the far interface and lists its own functions."""
    ops = "portal->broker::/queue/ops"
    return MapFragment(
        map_id="portal",
        derived_from="portal 1",
        boundaries=[
            _boundary(
                "portal", "server", "/api/v1/creds/get_access_token",
                direction="outbound", calls_out=["web/token.py::fetch"],
            ),
            _boundary(
                "portal", "server", "clientlib.Client.get_job_status",
                direction="outbound", calls_out=["web/jobs.py::status"],
            ),
            _boundary(
                "portal", "server", "clientlib.Client.gone", direction="outbound", calls_out=["web/jobs.py::old"]
            ),
            _boundary(
                "portal", "server", "doma_panda.jedi_tasks", transport="shared_table", direction="outbound",
                operations=["SELECT"], carried_values=["status", "jeditaskid"], calls_out=["web/tasks.py::rows"],
            ),
            _boundary(
                "portal", "server", "doma_panda.jobsarchived4", transport="shared_table", direction="outbound",
                calls_out=["web/jobs.py::rows"],
            ),
            _boundary(
                "portal", "server", "doma_panda.harvester_workers", transport="shared_table", direction="outbound",
                calls_out=["web/workers.py::rows"],
            ),
            _boundary(
                "portal", "console", "ai.models.Proposal", transport="shared_table", direction="outbound",
                handed_over=["status", "quality"], operations=["UPDATE"], calls_out=["web/ai.py::withdraw"],
                metadata={"values": {"status": ["withdrawn", "expired", "runtime(x)"]}},
            ),
            _boundary(
                "portal", "broker", "/queue/ops", transport="message", direction="outbound", name=ops,
                handed_over=["submit", "retire"],
                calls_out=["web/ops.py::submit", "web/ops.py::retire", "web/ops.py::forward"],
                call_sites=["web/ops.py:10", "web/ops.py:20", "web/ops.py:30"],
            ),
            _boundary(
                "portal", "broker", "/queue/reports", transport="message", direction="outbound",
                handed_over=["task_status"], calls_out=["web/report.py::send"],
            ),
            _boundary(
                "portal", "broker", "/topic/events", transport="message", direction="outbound",
                handed_over=["done"], calls_out=["web/ops.py::announce"],
            ),
        ],
        message_edges=[
            MessageEdge(sender="web/ops.py::submit", line=10, msg_type="submit", receiver=""),
            MessageEdge(sender="web/ops.py::retire", line=20, msg_type="retire", receiver=""),
        ],
    )  # fmt: skip


def console() -> MapFragment:
    """An application whose agent takes messages from a queue and whose models another application writes."""
    inbound = "broker->console::/queue/ops"
    agent = "agents/ops.py::Agent"
    return MapFragment(
        map_id="console",
        derived_from="console 1",
        boundaries=[
            _boundary(
                "console", "broker", "/queue/ops", transport="message", name=inbound,
                carried_values=["submit", "sweep", "restart"],
            ),
            _boundary(
                "console", "broker", "/queue/ops", transport="message", direction="outbound",
                name="console->broker::/queue/ops", handed_over=["restart"], calls_out=["ui/admin.py::restart"],
                call_sites=["ui/admin.py:5"],
            ),
        ],
        entities=[
            EntityNode(
                name="console::Proposal", map_id="console", derived_from="console 1", spec_class="Proposal",
                tables=["ai_proposal"],
            )
        ],
        subjects=[
            _subject(
                "console", "Proposal", "status", vocabulary=["open", "withdrawn"],
                metadata={"table": "ai_proposal", "module": "ai.models"},
            )
        ],
        message_edges=[MessageEdge(sender="ui/admin.py::restart", line=5, msg_type="restart", receiver="")],
        message_receivers=[
            MessageReceiver(receiver=agent, key="msg_type", msg_type=t, handled_by=[f"{agent}._handle_{t}"], boundary=inbound)
            for t in ("submit", "sweep", "restart")
        ],
    )  # fmt: skip


@pytest.fixture()
def found() -> B.Bindings:
    return B.bind([server(), client(), portal(), console()])


def _bound(found: B.Bindings, kind: str, source: str = "portal") -> list[B.Binding]:
    return [b for b in found.bound if b.kind == kind and b.source_map == source]


def _left(found: B.Bindings) -> dict[tuple[str, str], str]:
    return {(u.node, u.key or u.part): u.reason for u in found.unbound}


# ---------------------------------------------------------------- the keys


def test_keys_are_spelled_one_way():
    assert B.table_key('"ATLAS_PANDA"."jobsActive4"') == ("jobsactive4", "ATLAS_PANDA")
    assert B.table_key("jedi_tasks") == ("jedi_tasks", "")
    assert B.call_key("pkg/mod.py::Class.method") == "pkg.mod.Class.method"
    assert B.call_key("pkg/__init__.py::f") == "pkg.f"


def test_an_outbound_boundary_names_both_sides_either_way_round():
    by_own = [p for p in B.ports(server()) if p.role == B.USES and p.kind == B.CALL]
    assert [(p.key, p.functions) for p in by_own] == [
        ("rucio.client.Client.add_dataset", ["srv/ddm.py::RucioAPI.register_dataset"])
    ]
    by_far = [p for p in B.ports(portal()) if p.role == B.USES and p.kind == B.HTTP]
    assert [(p.key, p.functions) for p in by_far] == [
        ("/api/v1/creds/get_access_token", ["web/token.py::fetch"])
    ]


# ---------------------------------------------------------------- http and call


def test_a_request_binds_to_the_endpoint_and_the_boundary_it_arrives_by(found):
    (token,) = _bound(found, B.HTTP)
    assert token.key == "/api/v1/creds/get_access_token"
    assert token.source_functions == ["web/token.py::fetch"]
    assert token.target_map == "server"
    assert token.target_node == "srv/api/v1/creds_api.py::get_access_token"
    assert token.target_boundary == "server:client:creds_api::get_access_token"


def test_a_call_binds_to_the_library_and_the_library_to_the_server(found):
    (call,) = _bound(found, B.CALL)
    assert call.key == "clientlib.Client.get_job_status"
    assert (call.target_map, call.target_node) == (
        "client",
        "clientlib/Client.py::get_job_status",
    )
    (request,) = _bound(found, B.HTTP, source="client")
    assert (
        request.key == "/api/v1/job/get_description"
    )  # the verb is not part of the key
    assert request.source_functions == ["clientlib/Client.py::get_job_status"]
    assert request.target_node == "srv/api/v1/job_api.py::get_description"
    left = _left(found)
    # a host and a query are not part of the key either; the server has no such endpoint
    assert (
        left[
            (
                "client:server:clientlib/Client.py::kill",
                "https://server.example:25443/api/v1/job/kill?x=1",
            )
        ]
        == "the map of server has no node for it"
    )
    assert (
        left[("portal:server:clientlib.Client.gone", "clientlib.Client.gone")]
        == "the map of client has no node for it"
    )
    assert (
        left[
            (
                "server:rucio:srv/ddm.py::RucioAPI.register_dataset",
                "rucio.client.Client.add_dataset",
            )
        ]
        == "no map given covers rucio"
    )


# ---------------------------------------------------------------- tables


def test_a_table_binds_by_its_own_name_whatever_the_schema(found):
    tables = {b.key: b for b in _bound(found, B.TABLE) if not b.part}
    tasks = tables["jedi_tasks"]
    assert tasks.target_node == "server::TaskSpec"
    assert tasks.target_functions == ["srv/db.py::set_status"]
    assert "schema doma_panda here, ATLAS_PANDA there: the deployment's" in tasks.notes
    # a column with no subject is said on the table's binding, not left unbound
    assert "1 column(s) named here are no subject there: jeditaskid" in tasks.notes
    # a table only its record names binds to the class whose rows it holds
    assert tables["jobsarchived4"].target_node == "JobSpec"
    (status,) = [b for b in _bound(found, B.TABLE) if b.key == "jedi_tasks" and b.part]
    assert (status.part, status.target_node, status.notes) == (
        "status",
        "TaskSpec.status",
        ["read here"],
    )
    assert (
        _left(found)[
            (
                "portal:server:doma_panda.harvester_workers",
                "doma_panda.harvester_workers",
            )
        ]
        == "the map of server has no node for it"
    )


def test_a_model_binds_by_its_class_path_and_notes_an_undeclared_value(found):
    by_part = {
        b.part: b for b in _bound(found, B.TABLE) if b.key == "ai.models.Proposal"
    }
    assert by_part[""].target_node == "console::Proposal"
    assert by_part[""].notes == ["1 column(s) named here are no subject there: quality"]
    assert by_part["status"].target_node == "Proposal.status"
    assert by_part["status"].notes == [
        "written here",
        "writes 'expired', which Proposal.status does not declare",
    ]


def test_a_shared_table_of_the_server_itself_is_a_use_too(found):
    assert (
        _left(found)[("server:deft:ATLAS_DEFT.T_TASK", "ATLAS_DEFT.T_TASK")]
        == "no map given covers deft"
    )


# ---------------------------------------------------------------- messages


def test_a_type_binds_to_its_handler_where_the_destinations_agree(found):
    ops = [
        b
        for b in _bound(found, B.MESSAGE)
        if b.source_node == "portal->broker::/queue/ops"
    ]
    (submit,) = ops
    assert submit.key == "submit"
    assert submit.source_functions == ["web/ops.py::submit"]
    assert (submit.target_map, submit.target_node) == (
        "console",
        "agents/ops.py::Agent._handle_submit",
    )
    assert submit.target_boundary == "broker->console::/queue/ops"
    assert submit.notes == []
    left = _left(found)
    assert (
        left[("portal->broker::/queue/ops", "retire")]
        == "its receiver has no handler for this type"
    )
    assert (
        "does not write itself"
        in left[("portal->broker::/queue/ops", "web/ops.py::forward")]
    )
    assert (
        left[("portal:broker:/topic/events", "done")]
        == "no map given receives from /topic/events"
    )


def test_a_receiver_that_names_no_queue_binds_with_a_note(found):
    (report,) = [b for b in _bound(found, B.MESSAGE) if b.key == "task_status"]
    assert report.target_node == "srv/msg/status.py::StatusPlugin.process"
    assert report.notes == [
        "the receiver names no queue; whether it listens to /queue/reports is the deployment's"
    ]


def test_a_message_binds_inside_one_map(found):
    (restart,) = _bound(found, B.MESSAGE, source="console")
    assert (restart.target_map, restart.target_node) == (
        "console",
        "agents/ops.py::Agent._handle_restart",
    )


def test_a_type_taken_that_nobody_sends_is_listed_apart(found):
    # sweep has no sender; job_status has one its own map already places; task_status, submit, restart are bound
    assert [(u.map_id, u.key) for u in found.unreached] == [("console", "sweep")]


# ---------------------------------------------------------------- the whole


def test_resolution_of_a_boundary(found):
    assert found.resolution("portal:server:/api/v1/creds/get_access_token") == "bound"
    assert found.resolution("portal->broker::/queue/ops") == "partial"
    assert (
        found.resolution("portal:server:doma_panda.harvester_workers") == "unresolved"
    )


def test_binding_does_not_change_the_maps():
    fragments = [server(), client(), portal(), console()]
    before = [fragment.model_dump_json() for fragment in fragments]
    B.bind(fragments)
    assert [fragment.model_dump_json() for fragment in fragments] == before


def test_a_new_kind_of_crossing_is_a_reader(monkeypatch):
    monkeypatch.setattr(B, "_READERS", list(B._READERS))

    @B.register_reader
    def files(fragment):
        for boundary in fragment.boundaries:
            if boundary.transport == "file":
                yield B.Port(
                    map_id=fragment.map_id,
                    role=B.USES if boundary.direction == "outbound" else B.OFFERS,
                    kind="file",
                    key=boundary.interface,
                    node=boundary.name,
                )

    writer = MapFragment(
        map_id="writer", derived_from="writer 1",
        boundaries=[_boundary("writer", "reader", "/data/out.json", transport="file", direction="outbound")],
    )  # fmt: skip
    reader = MapFragment(
        map_id="reader", derived_from="reader 1",
        boundaries=[_boundary("reader", "writer", "/data/out.json", transport="file")],
    )  # fmt: skip
    (binding,) = [b for b in B.bind([writer, reader]).bound if b.kind == "file"]
    assert (binding.source_map, binding.target_map, binding.key) == (
        "writer",
        "reader",
        "/data/out.json",
    )


def test_bind_maps_reads_dumps_and_writes_the_bindings(tmp_path: Path):
    paths = []
    for fragment in (server(), client(), portal(), console()):
        paths.append(tmp_path / f"{fragment.map_id}.json")
        write_dump(fragment, paths[-1])
    out = tmp_path / "bindings.json"
    result = CliRunner().invoke(bind_maps, [*map(str, paths), "--dump", str(out)])
    assert result.exit_code == 0, result.output
    assert "portal -> server [http]: 1 bound" in result.output
    assert "portal -> server [table]: 2 bound, 1 column(s)" in result.output
    assert "portal -> console [message]: 1 bound" in result.output
    written = json.loads(out.read_text())
    assert set(written["maps"]) == {"server", "client", "portal", "console"}
    assert len(written["bound"]) == len(
        B.bind([server(), client(), portal(), console()]).bound
    )

    alone = CliRunner().invoke(bind_maps, [str(paths[0])])
    assert alone.exit_code != 0 and "at least two maps" in alone.output
    twice = CliRunner().invoke(bind_maps, [str(paths[0]), str(paths[0])])
    assert twice.exit_code != 0 and "more than once" in twice.output
