"""bamboo.mcp.external_mcp_client against both ``mcp`` SDK majors.

mcp 2.x renamed ``streamablehttp_client``, moved its headers onto the HTTP client it is handed, yields two
streams instead of three, and spells its fields in snake_case (``input_schema``, ``is_error``).  The fakes below
pin each version's shape; the round trips run a real server in a child process with whichever SDK is installed
(CI installs the newest).
"""

import socket
import subprocess
import sys
import textwrap
import time
import types
from contextlib import asynccontextmanager

import pytest

from bamboo.mcp import external_mcp_client as ext

# --- the transport and the fields, by version (fakes) ---------------------------------------------------------


class _FakeHttpClient:
    def __init__(self, headers):
        self.headers = headers

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _install(monkeypatch, transport_attrs: dict) -> list:
    """Fake ``mcp.client.streamable_http`` (and the SDK's HTTP client maker); returns the calls seen."""
    seen: list = []
    transport = types.ModuleType("mcp.client.streamable_http")
    for name, fn in transport_attrs.items():
        setattr(transport, name, fn(seen))
    client_mod = types.ModuleType("mcp.client")
    client_mod.streamable_http = transport
    httpx_utils = types.ModuleType("mcp.shared._httpx_utils")
    httpx_utils.create_mcp_http_client = lambda headers=None: _FakeHttpClient(headers)
    monkeypatch.setitem(sys.modules, "mcp.client", client_mod)
    monkeypatch.setitem(sys.modules, "mcp.client.streamable_http", transport)
    monkeypatch.setitem(sys.modules, "mcp.shared._httpx_utils", httpx_utils)
    return seen


def _current(n_items):
    def make(seen):
        @asynccontextmanager
        async def streamable_http_client(url, *, http_client=None):
            seen.append(("current", url, http_client.headers))
            yield ("read", "write", "session-id-getter")[:n_items]

        return streamable_http_client

    return make


def _older(seen):
    @asynccontextmanager
    async def streamablehttp_client(url, headers=None):
        seen.append(("older", url, headers))
        yield "read", "write", "session-id-getter"

    return streamablehttp_client


async def _open(url="http://h/mcp", headers=None):
    async with ext._streamable_http(url, headers or {"X-Probe": "1"}) as streams:
        return streams


async def test_mcp_2_current_client_two_streams_headers_on_the_http_client(monkeypatch):
    seen = _install(monkeypatch, {"streamable_http_client": _current(2)})
    assert await _open() == ("read", "write")
    assert seen == [("current", "http://h/mcp", {"X-Probe": "1"})]


async def test_mcp_1_26_current_client_three_items_first_two_kept(monkeypatch):
    seen = _install(
        monkeypatch,
        {"streamable_http_client": _current(3), "streamablehttp_client": _older},
    )
    assert await _open() == ("read", "write")
    assert [s[0] for s in seen] == ["current"]


async def test_an_older_mcp_1_gets_its_own_client_with_the_headers(monkeypatch):
    seen = _install(monkeypatch, {"streamablehttp_client": _older})
    assert await _open() == ("read", "write")
    assert seen == [("older", "http://h/mcp", {"X-Probe": "1"})]


async def test_no_known_client_is_an_import_error(monkeypatch):
    _install(monkeypatch, {})
    with pytest.raises(ImportError):
        await _open()


@pytest.mark.parametrize(
    "obj",
    [
        types.SimpleNamespace(
            input_schema={"type": "object"}, is_error=True
        ),  # mcp 2.x
        types.SimpleNamespace(inputSchema={"type": "object"}, isError=True),  # mcp 1.x
    ],
)
def test_fields_are_read_in_either_spelling(obj):
    assert ext._sdk_field(obj, "input_schema", "inputSchema") == {"type": "object"}
    assert ext._sdk_field(obj, "is_error", "isError") is True


# --- round trips with the installed SDK -------------------------------------------------------------------------

SERVER = textwrap.dedent(
    '''
    import sys

    try:
        from mcp.server.mcpserver import Context, MCPServer as Server  # mcp 2.x
        MAJOR = 2
    except ImportError:
        from mcp.server.fastmcp import Context, FastMCP as Server  # mcp 1.x
        MAJOR = 1

    mode = sys.argv[1]
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    if MAJOR == 1 and mode == "http":
        srv = Server("probe", host="127.0.0.1", port=port)
    else:
        srv = Server("probe")


    @srv.tool()
    def add(a: int, b: int) -> str:
        """Add two numbers."""
        return '{"sum": %d}' % (a + b)


    @srv.tool()
    def fail() -> str:
        """Always fails."""
        raise ValueError("boom")


    @srv.tool()
    def header(ctx: Context) -> str:
        """The x-probe request header, as the server saw it."""
        return ctx.request_context.request.headers.get("x-probe", "<none>")


    if mode == "stdio":
        srv.run()
    elif MAJOR == 2:
        srv.run(transport="streamable-http", host="127.0.0.1", port=port)
    else:
        srv.run(transport="streamable-http")
    '''
)


@pytest.fixture
def server_file(tmp_path):
    pytest.importorskip("mcp")
    path = tmp_path / "probe_server.py"
    path.write_text(SERVER)
    return path


async def _round_trip(client):
    await client.connect()
    try:
        tools = {t.name: t for t in client.list_tools()}
        assert set(tools) == {"add", "fail", "header"}
        assert set(tools["add"].parameters_schema["properties"]) == {"a", "b"}
        assert await client.execute("add", a=2, b=3) == {"sum": 5}
        with pytest.raises(RuntimeError, match="returned error"):
            await client.execute("fail")
        return client
    finally:
        await client.close()


async def test_stdio_round_trip(server_file):
    await _round_trip(
        ext.StdioMcpClient("probe", sys.executable, [str(server_file), "stdio"])
    )


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


async def test_streamable_http_round_trip_with_headers(server_file):
    pytest.importorskip("uvicorn")
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, str(server_file), "http", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 30
        while True:
            with socket.socket() as s:
                if s.connect_ex(("127.0.0.1", port)) == 0:
                    break
            assert proc.poll() is None, "the probe server exited"
            assert time.monotonic() < deadline, "the probe server did not listen"
            time.sleep(0.2)
        client = ext.ExternalMcpClient(
            "probe", f"http://127.0.0.1:{port}/mcp", headers={"X-Probe": "seen"}
        )
        await client.connect()
        try:
            assert {t.name for t in client.list_tools()} == {"add", "fail", "header"}
            assert await client.execute("add", a=2, b=3) == {"sum": 5}
            assert await client.execute("header") == "seen"
            with pytest.raises(RuntimeError, match="returned error"):
                await client.execute("fail")
        finally:
            await client.close()
    finally:
        proc.terminate()
        proc.wait(timeout=10)
