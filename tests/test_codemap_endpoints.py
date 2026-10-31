"""The API's endpoints and the statements each one's calls reach -- ``codemap.panda.endpoints``."""

from __future__ import annotations

import ast
from pathlib import Path

from bamboo.codemap import reach
from bamboo.codemap.gitsource import blob_sha
from bamboo.codemap.models import (
    FILL_CONSTANT,
    HOLE_CLOSED,
    HOLE_MAY_CHANGE,
    HOLE_OPEN,
    Anchor,
    EndpointFill,
    EndpointWrite,
    KeyFrom,
    KeySource,
    MapFragment,
    ReadSiteNode,
    SourceModule,
)
from bamboo.codemap.panda import attribution, endpoints, pathcond, sql, sqltext
from bamboo.codemap.panda import objects as panda_objects
from bamboo.codemap.panda.plugin import _DatabaseLeaves
from bamboo.codemap.panda.recognizers import progress, sqlwrite

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
from pandaserver.taskbuffer.proxy import Archiver, Layer, Peeker


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


def get_peek(req, job_id):
    return Peeker().peek(job_id)


def get_peek_files(req, job_id):
    return Peeker().peek(job_id, with_files=True)


def get_layered(req, job_id):
    return Layer().go(job_id)
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

    def peek(self, job_id, with_files=False):
        self.impl(job_id)
        if with_files:
            self.files(job_id)  # G

    def files(self, job_id):
        self.cur.execute("SELECT lfn FROM ATLAS_PANDA.filesTable4 WHERE PandaID=:PandaID", {":PandaID": job_id})  # L


class Layer:
    def go(self, job_id, with_files=False):
        return Peeker().peek(job_id, with_files)


class Archiver(Reader):
    def impl(self, job_id):
        self.cur.execute("SELECT jobStatus FROM ATLAS_PANDAARCH.jobsArchived WHERE PandaID=:PandaID", {":PandaID": job_id})  # A
        self.cur.execute("UPDATE ATLAS_PANDAARCH.jobsArchived SET jobStatus=:s WHERE PandaID=:PandaID", {":s": "closed", ":PandaID": job_id})  # W
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


def _site(
    marker: str, table: str, via: str, function: str = "impl"
) -> tuple[ReadSiteNode, KeyFrom]:
    """The read site and the read term of PROXY's statement at *marker*, as the build records them: owned by the
    function's bare name.  The term is filled by ``job_id``, and by the statement's own rows come by a call at
    *via*."""
    file, line = f"{PACKAGE}/taskbuffer/proxy.py", _line(PROXY, marker)
    owner = f"{file}::{function}"
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


def _build(
    tmp_path: Path,
    returned: dict[str, sqlwrite.Returned] | None = None,
    files: dict[str, str] = FILES,
) -> MapFragment:
    root = tmp_path / PACKAGE
    modules = []
    for rel, text in files.items():
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
    for folder in ("", "server", "api", "api/v1", "taskbuffer", "asyncprocess"):
        (root / folder).mkdir(parents=True, exist_ok=True)
        (root / folder / "__init__.py").write_text("")
    calls = sqlwrite.corpus_calls(modules, {PACKAGE: root})
    peeked, archived, listed = (
        _site("P", "jobsActive4", f"{PACKAGE}/taskbuffer/elsewhere.py:9"),
        _site("A", "ATLAS_PANDAARCH.jobsArchived", f"{PACKAGE}/taskbuffer/proxy.py:3"),
        _site("L", "filesTable4", f"{PACKAGE}/taskbuffer/elsewhere.py:9", "files"),
    )
    with sql.across_calls(calls):
        fragment = MapFragment(
            map_id="m",
            derived_from="v",
            read_sites=[peeked[0], archived[0], listed[0]],
            key_froms=[peeked[1], archived[1], listed[1]],
            reach_graph=reach.graph(calls, "m", "v", _DatabaseLeaves()),
        )
        panda_objects.attach(fragment, modules, calls)
        endpoints.attach(fragment, modules, calls, returned)
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
                "get_peek",
                "get_peek_files",
                "get_layered",
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


def test_a_read_says_where_its_rows_are_in_the_answer_as_the_walk_carried_them_back(
    tmp_path: Path,
) -> None:
    """What the crossings' walk carried back to each endpoint's function (``sqlwrite.Returned``) is where the read's
    rows are in that endpoint's answer, and none where it carried nothing there; a hole the walk met is said only
    where a function of the endpoint's closure makes it -- ``Archiver.impl``'s is no way to ``get_status``'s."""
    proxy, api = f"{PACKAGE}/taskbuffer/proxy.py", f"{PACKAGE}/api/v1/job_api.py"
    returned = {
        f"panda:{proxy}::impl@{_line(PROXY, 'P')}": sqlwrite.Returned(
            {f"{api}::get_status": ["()"], f"{api}::get_task": ["[*][0]"]},
            [
                (f"{proxy}::Peeker.impl", 12, "x.f"),
                (f"{proxy}::Archiver.impl", 30, "y.g"),
            ],
        )
    }
    by_route = {one.route: one for one in _build(tmp_path, returned).endpoints}

    assert [
        (read.returned, read.returned_open)
        for route in ("job/get_status", "job/get_task", "job/get_fixed")
        for read in by_route[route].reads
    ] == [
        (["()"], [f"{proxy}::Peeker.impl:12"]),
        (["[*][0]"], [f"{proxy}::Peeker.impl:12"]),
        ([], [f"{proxy}::Peeker.impl:12"]),
    ]
    (archived,) = by_route["job/get_archived"].reads
    assert (archived.returned, archived.returned_open) == ([], [])


OPS_API = """\
import os


def touch(req, name):
    os.utime(name)


def clean(req, keeper):
    keeper.tidy()
    keeper.peek()
    keeper()
"""

STORE = """\
class Store:
    def tidy(self):
        self.cur.execute("DELETE FROM ATLAS_PANDA.jobsActive4 WHERE PandaID=:PandaID", {":PandaID": 1})

    def peek(self):
        return 1
"""

PROCESSOR = """\
import subprocess

from pandaserver.asyncprocess import dc_handlers

READ = "read"


def _handle_grep(row, tb, log, machine):
    subprocess.run(["rg", row])


def _handle_read(row, tb, log, machine):
    pass


HANDLERS = {"grep": _handle_grep, READ: _handle_read, **dc_handlers.HANDLERS}
"""

DC_HANDLERS = """\
import functools

_NAMES = ("pin", "retire")


def _handle(name, row, tb, log, machine):
    pass


HANDLERS = {f"dc_{name}": functools.partial(_handle, name) for name in _NAMES}
"""

CHANGING = {
    **FILES,
    "server/panda.py": SERVER.replace(
        "from pandaserver.api.v1 import job_api as job_api_v1\n",
        "from pandaserver.api.v1 import job_api as job_api_v1\nfrom pandaserver.api.v1 import ops_api as ops_api_v1\n",
    ).replace(
        '"job": {"module": job_api_v1, "allowed_methods": []},\n',
        '"job": {"module": job_api_v1, "allowed_methods": []},\n'
        '            "ops": {"module": ops_api_v1, "allowed_methods": []},\n',
    ),
    "api/v1/ops_api.py": OPS_API,
    "taskbuffer/store.py": STORE,
    "asyncprocess/processor.py": PROCESSOR,
    "asyncprocess/dc_handlers.py": DC_HANDLERS,
}


def test_an_endpoint_says_what_it_changes_beside_the_tables_and_which_holes_may_hide_a_change(
    tmp_path: Path,
) -> None:
    """``touch`` changes a file's times (``callfacts.change``).  ``clean``'s three calls on a parameter no one
    declares are holes, bounded by their names as the reach record bounds a mover's: ``tidy`` may change rows,
    ``Store.tidy`` deletes them; no ``peek`` of the corpus changes anything; a parameter called has no name."""
    by_route = {one.route: one for one in _build(tmp_path, files=CHANGING).endpoints}
    api, store = f"{PACKAGE}/api/v1/ops_api.py", f"{PACKAGE}/taskbuffer/store.py"

    assert [
        (one.kind, one.callee, one.function, one.line)
        for one in by_route["ops/touch"].changes
    ] == [("file", "os.utime", f"{api}::touch", 5)]
    assert by_route["ops/touch"].hole_bounds == []
    assert [
        (one.at, one.callee, one.bound, one.through)
        for one in by_route["ops/clean"].hole_bounds
    ] == [
        (f"{api}::clean:9", "keeper.tidy", HOLE_MAY_CHANGE, [f"{store}::Store.tidy"]),
        (f"{api}::clean:10", "keeper.peek", HOLE_CLOSED, []),
        (f"{api}::clean:11", "keeper", HOLE_OPEN, []),
    ]
    assert by_route["ops/clean"].changes == by_route["job/get_status"].changes == []


def test_the_async_requests_handlers_are_read_off_the_daemons_table_as_endpoints_are(
    tmp_path: Path,
) -> None:
    """``processor.HANDLERS``: a literal key, a name the module binds to a text, and another module's table spread
    in, whose keys a comprehension makes over a tuple and whose values are ``functools.partial`` of one function.
    Each request type is a record of its own, read as an endpoint is."""
    handled = _build(tmp_path, files=CHANGING).handlers
    asynced = f"{PACKAGE}/asyncprocess"

    assert [(one.route, one.version, one.function) for one in handled] == [
        ("async/grep", "async", f"{asynced}/processor.py::_handle_grep"),
        ("async/read", "async", f"{asynced}/processor.py::_handle_read"),
        ("async/dc_pin", "async", f"{asynced}/dc_handlers.py::_handle"),
        ("async/dc_retire", "async", f"{asynced}/dc_handlers.py::_handle"),
    ]
    assert [(one.kind, one.callee) for one in handled[0].changes] == [
        ("process", "subprocess.run")
    ]


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


SPECS = """\
class FileSpec:
    _attributes = ("row_ID", "PandaID", "lfn")


class JobSpec:
    _attributes = ("PandaID", "jobStatus", "computingSite")
"""


def test_a_terms_column_is_a_subject_of_the_class_its_table_holds_or_its_history_is_of() -> (
    None
):
    """``filesTable_ARCH``, an archive no statement inserts into, has no class the build learns, and every attribute
    of ``FileSpec`` is one of its columns: its ``PandaID`` is ``FileSpec.PandaID``.  ``jobs_StatusLog`` holds a job's
    past statuses: its ``PandaID``, the column its rows are keyed by, is the job's, and its ``jobStatus`` is none of
    the job's attributes.  Outside the schema the archive's column is no subject."""
    tree = ast.parse(SPECS)
    pathcond.attach_parents(tree)
    modules = [
        SourceModule(
            package=PACKAGE,
            rel_path=f"{PACKAGE}/taskbuffer/Specs.py",
            tree=tree,
            source=SPECS,
            blob_sha=blob_sha(SPECS),
        )
    ]
    attributor = attribution.SpecAttributor(
        progress.spec_attributes(modules), attribution.class_bases(modules)
    )
    columns = {
        "filestable_arch": frozenset({"row_id", "pandaid", "lfn", "status"}),
        "jobs_statuslog": frozenset({"pandaid", "jobstatus", "modificationtime"}),
    }
    with sqltext.with_schema(columns):
        assert (
            sqlwrite._spec_attr(
                attributor, "ATLAS_PANDAARCH.filesTable_ARCH", "PandaID"
            )
            == "FileSpec.PandaID"
        )
        assert (
            sqlwrite._spec_attr(attributor, "ATLAS_PANDA.jobs_StatusLog", "PandaID")
            == "JobSpec.PandaID"
        )
        assert (
            sqlwrite._spec_attr(attributor, "ATLAS_PANDA.jobs_StatusLog", "jobStatus")
            == ""
        )
    assert (
        sqlwrite._spec_attr(attributor, "ATLAS_PANDAARCH.filesTable_ARCH", "PandaID")
        == ""
    )


def test_an_endpoint_says_which_statements_writing_rows_its_calls_reach(
    tmp_path: Path,
) -> None:
    """``Archiver.impl`` updates the archived job it reads; ``Peeker.impl`` writes nothing."""
    by_route = {one.route: one for one in _build(tmp_path).endpoints}

    assert by_route["job/get_archived"].writes == [
        EndpointWrite(
            function=f"{PACKAGE}/taskbuffer/proxy.py::Archiver.impl",
            line=_line(PROXY, "W"),
            kind="update",
            tables=["jobsarchived"],
        )
    ]
    assert by_route["job/get_status"].writes == []


def test_a_call_a_constant_flag_rules_out_on_every_way_in_is_left_out(
    tmp_path: Path,
) -> None:
    """``peek`` reads the files only ``if with_files:``.  ``get_peek`` leaves it to its default, ``False``, and so
    does ``Layer.go``, which hands ``peek`` its own parameter: the call is shut and the files are not read.
    ``get_peek_files`` hands ``True``: they are."""
    by_route = {one.route: one for one in _build(tmp_path).endpoints}
    proxy = f"{PACKAGE}/taskbuffer/proxy.py"
    files = f"panda:{proxy}::files@{_line(PROXY, 'L')}"

    for route in ("job/get_peek", "job/get_layered"):
        assert [
            read.site for read in by_route[route].reads if read.site == files
        ] == [], route
        assert by_route[route].shut == [f"{proxy}:{_line(PROXY, 'G')}"], route
    assert files in [read.site for read in by_route["job/get_peek_files"].reads]
    assert by_route["job/get_peek_files"].shut == []
