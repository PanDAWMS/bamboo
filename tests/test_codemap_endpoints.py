"""The API's endpoints and the statements each one's calls reach -- ``codemap.panda.endpoints``."""

from __future__ import annotations

import ast
from pathlib import Path

from bamboo.codemap import reach
from bamboo.codemap.gitsource import blob_sha
from bamboo.codemap.models import (
    FILL_CONSTANT,
    Anchor,
    EndpointFill,
    KeyFrom,
    KeySource,
    MapFragment,
    ReadSiteNode,
    SourceModule,
)
from bamboo.codemap.panda import endpoints, pathcond, sql
from bamboo.codemap.panda import objects as panda_objects
from bamboo.codemap.panda.plugin import _DatabaseLeaves
from bamboo.codemap.panda.recognizers import sqlwrite

PACKAGE = "pandaserver"

SERVER = """\
from pandaserver.api.v1 import credential_management_api as cred_api_v1
from pandaserver.api.v1 import job_api as job_api_v1


def module_mapping(version, api_module):
    mapping = {
        "v0": {"panda": {"module": None, "allowed_methods": []}},
        "v1": {
            "creds": {"module": cred_api_v1, "allowed_methods": []},
            "job": {"module": job_api_v1, "allowed_methods": []},
        },
    }
    return mapping[version][api_module]
"""

JOB_API = """\
from pandaserver.taskbuffer.proxy import Archiver, Peeker


def init_task_buffer(task_buffer):
    pass


def _helper(job_id):
    return Peeker().run(job_id)


def get_status(req, job_id):
    return Peeker().run(job_id)  # S


def get_archived(req, job_id):
    respond = req.respond
    respond()  # H
    return Archiver().run(job_id)


def get_task(req, task_id):
    task_id = int(task_id)
    return Peeker().run(task_id)


def get_many(req, job_ids):
    if not job_ids:
        job_ids = []
    job_ids = list(dict.fromkeys(job_ids))
    for job_id in job_ids:
        Peeker().run(job_id)


def get_fixed(req):
    return Peeker().run(123)  # F


def get_both(req, job_ids, more_ids):
    Peeker().run_each(job_ids + more_ids)
"""

CREDS_API = """\
def get_key(req):
    return None
"""

PROXY = """\
class Reader:
    def run(self, job_id):
        return self.impl(job_id)

    def run_each(self, job_ids):
        for job_id in job_ids:
            self.run(job_id)


class Peeker(Reader):
    def impl(self, job_id):
        self.cur.execute("SELECT jobStatus FROM ATLAS_PANDA.jobsActive4 WHERE PandaID=:PandaID", {":PandaID": job_id})  # P
        return self.cur.fetchall()


class Archiver(Reader):
    def impl(self, job_id):
        self.cur.execute("SELECT jobStatus FROM ATLAS_PANDAARCH.jobsArchived WHERE PandaID=:PandaID", {":PandaID": job_id})  # A
        return self.cur.fetchall()
"""

FILES = {
    "server/panda.py": SERVER,
    "api/v1/job_api.py": JOB_API,
    "api/v1/credential_management_api.py": CREDS_API,
    "taskbuffer/proxy.py": PROXY,
}


def _line(text: str, marker: str) -> int:
    return next(
        n for n, line in enumerate(text.splitlines(), 1) if line.endswith(f"# {marker}")
    )


def _site(marker: str, table: str, via: str) -> tuple[ReadSiteNode, KeyFrom]:
    """The read site and the read term of PROXY's statement at *marker*, as the build records them: owned by the
    function's bare name.  The term is filled by ``job_id``, and by the statement's own rows come by a call at
    *via*."""
    file, line = f"{PACKAGE}/taskbuffer/proxy.py", _line(PROXY, marker)
    owner = f"{file}::impl"
    site = ReadSiteNode(
        map_id="m",
        derived_from="v",
        name=f"panda:{owner}@{line}",
        owner=owner,
        tables=[table],
        anchor=Anchor(package=PACKAGE, file=file, line_start=line, line_end=line),
    )
    term = KeyFrom(
        owner=owner,
        file=file,
        statement=line,
        table=table,
        term="PandaID=:PandaID",
        reads=True,
        attr="JobSpec.PandaID",
        sources=[
            KeySource(kind="parameter", line=line, name="job_id", paths=["()"]),
            KeySource(kind="rows", line=line, name=site.name, paths=["()"], via=[via]),
        ],
    )
    return site, term


def _build(tmp_path: Path) -> MapFragment:
    root = tmp_path / PACKAGE
    modules = []
    for rel, text in FILES.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        tree = ast.parse(text)
        pathcond.attach_parents(tree)
        modules.append(
            SourceModule(
                package=PACKAGE,
                rel_path=f"{PACKAGE}/{rel}",
                tree=tree,
                source=text,
                blob_sha=blob_sha(text),
            )
        )
    for folder in ("", "server", "api", "api/v1", "taskbuffer"):
        (root / folder / "__init__.py").write_text("")
    calls = sqlwrite.corpus_calls(modules, {PACKAGE: root})
    peeked, archived = (
        _site("P", "jobsActive4", f"{PACKAGE}/taskbuffer/elsewhere.py:9"),
        _site("A", "ATLAS_PANDAARCH.jobsArchived", f"{PACKAGE}/taskbuffer/proxy.py:3"),
    )
    with sql.across_calls(calls):
        fragment = MapFragment(
            map_id="m",
            derived_from="v",
            read_sites=[peeked[0], archived[0]],
            key_froms=[peeked[1], archived[1]],
            reach_graph=reach.graph(calls, "m", "v", _DatabaseLeaves()),
        )
        panda_objects.attach(fragment, modules, calls)
        endpoints.attach(fragment, modules, calls)
    return fragment


def test_the_endpoints_are_the_routed_modules_public_functions(tmp_path: Path) -> None:
    """Every public function of each module the routing table names, by the table's name for it (``creds``), but
    ``init_task_buffer``; the legacy entry, whose module is ``None``, is no module."""
    found = {
        (one.version, one.route, one.function) for one in _build(tmp_path).endpoints
    }

    assert found == {
        (
            "v1",
            "creds/get_key",
            f"{PACKAGE}/api/v1/credential_management_api.py::get_key",
        ),
        *(
            ("v1", f"job/{name}", f"{PACKAGE}/api/v1/job_api.py::{name}")
            for name in (
                "get_status",
                "get_archived",
                "get_task",
                "get_many",
                "get_fixed",
                "get_both",
            )
        ),
    }


def test_an_endpoint_reads_the_statements_its_calls_reach_as_the_class_they_run_as(
    tmp_path: Path,
) -> None:
    """``Peeker().run`` reaches ``Reader.run`` as a ``Peeker``, whose ``self.impl`` is ``Peeker.impl``: the endpoint
    reads ``jobsActive4`` and not the archive ``Archiver.impl`` reads, by the way it took."""
    by_route = {one.route: one for one in _build(tmp_path).endpoints}
    proxy = f"{PACKAGE}/taskbuffer/proxy.py"

    status = by_route["job/get_status"]
    assert [(read.function, read.tables) for read in status.reads] == [
        (f"{proxy}::Peeker.impl", ["jobsactive4"])
    ]
    assert [(term.term, term.table, term.attr) for term in status.reads[0].terms] == [
        ("PandaID=:PandaID", "jobsActive4", "JobSpec.PandaID")
    ]
    assert status.reads[0].way == [
        f"{PACKAGE}/api/v1/job_api.py:{_line(JOB_API, 'S')}",
        f"{proxy}:3",
    ]
    assert status.holes == []

    archived = by_route["job/get_archived"]
    assert [(read.function, read.tables) for read in archived.reads] == [
        (f"{proxy}::Archiver.impl", ["atlas_pandaarch.jobsarchived"])
    ]
    assert by_route["creds/get_key"].reads == []


def test_an_endpoint_says_where_its_calls_may_go_where_no_edge_says(
    tmp_path: Path,
) -> None:
    """``respond()`` calls a local: a hole, so a statement may be read past it."""
    by_route = {one.route: one for one in _build(tmp_path).endpoints}

    assert by_route["job/get_archived"].holes == [
        f"{PACKAGE}/api/v1/job_api.py::get_archived:{_line(JOB_API, 'H')}"
    ]


def test_a_term_is_filled_by_what_the_endpoint_is_handed_followed_up_the_calls(
    tmp_path: Path,
) -> None:
    """``PandaID=:PandaID`` is ``impl``'s ``job_id``, ``run``'s, and so what each endpoint hands ``run``: its own
    ``job_id``; ``task_id`` converted by ``int()``; each of ``job_ids``, through ``dict.fromkeys`` and ``list`` and
    the loop -- the empty list beside it fills nothing --; a literal, which no request fills; each of two lists
    added and handed to a loop.  The statement's own rows fill it where they come by a call of the endpoint's closure (``run``'s, into
    ``Archiver.impl``), not by a call elsewhere."""
    by_route = {one.route: one for one in _build(tmp_path).endpoints}

    def fills(route: str) -> list[EndpointFill]:
        (read,) = by_route[route].reads
        (term,) = read.terms
        return term.fills

    archived = f"panda:{PACKAGE}/taskbuffer/proxy.py::impl@{_line(PROXY, 'A')}"
    assert fills("job/get_status") == [EndpointFill(parameter="job_id")]
    assert fills("job/get_archived") == [
        EndpointFill(rows=archived),
        EndpointFill(parameter="job_id"),
    ]
    assert fills("job/get_task") == [EndpointFill(parameter="task_id")]
    assert fills("job/get_many") == [EndpointFill(parameter="job_ids[*]")]
    assert fills("job/get_fixed") == [
        EndpointFill(
            why=FILL_CONSTANT, at=f"{PACKAGE}/api/v1/job_api.py:{_line(JOB_API, 'F')}"
        )
    ]
    assert fills("job/get_both") == [
        EndpointFill(parameter="job_ids[*]"),
        EndpointFill(parameter="more_ids[*]"),
    ]
