"""panda-client Code Map plugin: per function of the library, the requests it makes."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from bamboo.codemap.factory import available_map_ids, get_code_map_plugin
from bamboo.codemap.pandaclient.plugin import PandaClientCodeMapPlugin

CLIENT = """
server_base_path_ssl = f"{scheme}://{netloc}/api/v2"


def http_request_decorator(endpoint, method="post"):
    def decorator(func):
        return func
    return decorator


@http_request_decorator(endpoint="job/get_description", method="post", json_out=True)
def getJobStatus_internal(ids):
    return {"job_ids": ids}


def getJobStatus(ids, verbose=False):
    return getJobStatus_internal(ids)


@http_request_decorator("task/kill")
def killTask(task_id):
    return {"task_id": task_id}


@http_request_decorator(endpoint="system/is_alive", method="get")
def is_alive():
    return {}


def retire(task_id):
    getJobStatus([task_id])
    return killTask(task_id)


def hostname():
    return "localhost"
"""

BOOK = """
from . import Client


def show(ids):
    return Client.getJobStatus(ids)
"""


@pytest.fixture()
def fragment(tmp_path: Path):
    package = tmp_path / "pandaclient"
    package.mkdir()
    (package / "Client.py").write_text(textwrap.dedent(CLIENT))
    (package / "book.py").write_text(textwrap.dedent(BOOK))
    plugin = PandaClientCodeMapPlugin()
    assert plugin.prepare(tmp_path).startswith("panda-client ")
    return plugin.run()


def _requests(fragment) -> dict[str, list[str]]:
    return {b.interface: b.calls_out for b in fragment.boundaries}


def test_registered():
    assert "panda-client" in available_map_ids()
    assert isinstance(get_code_map_plugin("panda-client"), PandaClientCodeMapPlugin)


def test_prepare_needs_the_package(tmp_path):
    with pytest.raises(FileNotFoundError):
        PandaClientCodeMapPlugin().prepare(tmp_path)


def test_a_declared_request_is_the_functions_boundary(fragment):
    requests = _requests(fragment)
    # the base path is the one the module builds, the verb the decorator's, post where it gives none
    assert requests["pandaclient/Client.py::getJobStatus_internal"] == [
        "POST /api/v2/job/get_description"
    ]
    assert requests["pandaclient/Client.py::killTask"] == ["POST /api/v2/task/kill"]
    assert requests["pandaclient/Client.py::is_alive"] == [
        "GET /api/v2/system/is_alive"
    ]
    boundary = next(
        b for b in fragment.boundaries if b.interface.endswith("::is_alive")
    )
    assert (
        boundary.system,
        boundary.transport,
        boundary.direction,
        boundary.verbs,
    ) == (
        "panda",
        "http",
        "outbound",
        ["GET"],
    )
    assert boundary.anchor.file == "pandaclient/Client.py"


def test_a_wrapper_makes_the_requests_its_calls_reach(fragment):
    requests = _requests(fragment)
    assert requests["pandaclient/Client.py::getJobStatus"] == [
        "POST /api/v2/job/get_description"
    ]
    assert requests["pandaclient/Client.py::retire"] == [
        "POST /api/v2/job/get_description",
        "POST /api/v2/task/kill",
    ]
    # through another module of the package, by the name it imports the module under
    assert requests["pandaclient/book.py::show"] == ["POST /api/v2/job/get_description"]
    # a function that makes none is no boundary
    assert "pandaclient/Client.py::hostname" not in requests
    assert "pandaclient/Client.py::http_request_decorator" not in requests
