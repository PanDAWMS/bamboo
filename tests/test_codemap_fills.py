"""Which writes of a value a function reads filled it from a row the function fetched (bamboo.codemap.panda.fills)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from bamboo.codemap.gitsource import blob_sha
from bamboo.codemap.models import SourceModule
from bamboo.codemap.panda import fills, pathcond, sql
from bamboo.codemap.panda.recognizers import sqlwrite

PACKAGE = "pkg"

SPEC = """\
class JobSpec:
    _attributes = ("PandaID", "jobStatus", "endTime")

    def __init__(self):
        for attr in self._attributes:
            object.__setattr__(self, attr, None)

    def __setattr__(self, name, value):
        oldVal = getattr(self, name)
        object.__setattr__(self, name, value)

    def pack(self, values):
        for i in range(len(self._attributes)):
            attr = self._attributes[i]
            val = values[i]
            object.__setattr__(self, attr, val)

    @classmethod
    def columnNames(cls):
        return ",".join(cls._attributes)
"""

PROXY = """\
from pkg.spec import JobSpec


class Proxy:
    def killJob(self, pandaID, code):
        sql2 = f"SELECT {JobSpec.columnNames()} "
        sql2 += "FROM %s WHERE PandaID=:PandaID AND jobStatus<>:jobStatus"
        for table in ("ATLAS_PANDA.jobsDefined4", "ATLAS_PANDA.jobsActive4"):
            varMap = {}
            varMap[":PandaID"] = pandaID
            if code == "9":
                varMap[":jobStatus"] = "dummy"
            else:
                varMap[":jobStatus"] = "running"
            self.cur.execute(sql2 % table, varMap)
            res = self.cur.fetchall()
            if len(res) == 0:
                continue
            job = JobSpec()
            job.pack(res[0])
            if job.jobStatus != "failed":  # K1
                if code == "2":
                    job.jobStatus = "closed"
                if job.jobStatus != "closed":  # K2
                    job.jobStatus = "cancelled"
            self.touch(job)
            if job.jobStatus == "cancelled":  # K3
                return

    def touch(self, job):
        job.jobStatus = "touched"

    def updateJobStatus(self, pandaID):
        sql0 = "SELECT commandToPilot,jobStatus FROM ATLAS_PANDA.jobsActive4 WHERE PandaID=:PandaID "
        varMap0 = {}
        varMap0[":PandaID"] = pandaID
        self.cur.execute(sql0, varMap0)
        res = self.cur.fetchone()
        if res is not None:  # U1
            (
                commandToPilot,
                oldJobStatus,
            ) = res
            if oldJobStatus == "running":  # U2
                return

    def resetJob(self, pandaID):
        sqlF = "SELECT PandaID,jobStatus,endTime FROM ATLAS_PANDA.filesTable4 WHERE PandaID=:PandaID "
        self.cur.execute(sqlF, {":PandaID": pandaID})
        resFs = self.cur.fetchall()
        for resF in resFs:
            file = JobSpec()
            file.pack(resF)
            if file.jobStatus == "missing":  # E1
                continue

    def given(self, job, rows):
        if job.jobStatus == "running":  # A1
            return
        other = JobSpec()
        if other.jobStatus is None:  # A2
            return
        other.pack(rows[0])
        if other.jobStatus is None:  # A3
            return

    def peekJob(self, pandaID, fromActive):
        sql1_0 = "SELECT %s FROM %s "
        sql1_1 = "WHERE PandaID=:PandaID"
        tables = []
        if fromActive:
            tables.append("ATLAS_PANDA.jobsActive4")
        tables.append("ATLAS_PANDA.jobsArchived4")
        varMap = {}
        varMap[":PandaID"] = pandaID
        for table in tables:
            sql = sql1_0 % (JobSpec.columnNames(), table) + sql1_1
            self.cur.execute(sql, varMap)
            res = self.cur.fetchall()
            if len(res) != 0:
                job = JobSpec()
                job.pack(res[0])
                sqlFile = f"SELECT {JobSpec.columnNames()} FROM ATLAS_PANDA.filesTable4 "
                sqlFile += "WHERE PandaID=:PandaID"
                self.cur.execute(sqlFile, varMap)
                resFs = self.cur.fetchall()
                sqlMeta = "SELECT metaData FROM ATLAS_PANDA.metaTable WHERE PandaID=:PandaID"
                self.cur.execute(sqlMeta, varMap)
                for (clobMeta,) in self.cur:
                    pass
                if job.jobStatus == "finished":  # P1
                    return job
        return None
"""


BUFFER = """\
from contextlib import contextmanager
from typing import Generator

from pkg.proxy import Proxy


class Pool:
    @contextmanager
    def get(self) -> Generator[Proxy, None, None]:
        proxy = Proxy()
        yield proxy


class TaskBuffer:
    proxyPool: Pool

    def peekJobs(self, jobIDs, fromActive=True, use_json=False):
        with self.proxyPool.get() as proxy:
            retJobs = []
            for jobID in jobIDs:
                res = proxy.peekJob(jobID, fromActive)
                if res:
                    if use_json:
                        retJobs.append(res.to_dict())
                    else:
                        retJobs.append(res)
                else:
                    retJobs.append(None)
        return retJobs


class Adder:
    def __init__(self, taskBuffer: TaskBuffer, job_id):
        self.taskBuffer = taskBuffer
        self.job_id = job_id
        self.job = None

    def run(self):
        self.job = self.taskBuffer.peekJobs([self.job_id], fromActive=False)[0]
        self.process()
        self.unchecked()
        job = self.taskBuffer.peekJobs([self.job_id])[0]
        if job is not None:
            self.check(job)

    def process(self):
        if self.job is None:
            return
        if self.job.jobStatus not in ["transferring"]:  # M1
            self.job.jobStatus = "holding"

    def unchecked(self):
        if self.job.jobStatus == "running":  # M2
            return

    def check(self, job):
        if job.jobStatus == "failed":  # C1
            return


class Relay:
    def __init__(self, inner):
        self.inner = inner

    def __getattr__(self, name):
        return getattr(self.inner, name)


def watch(source: TaskBuffer | Relay, job_id):
    job = source.peekJobs([job_id])[0]
    if job is None:
        return
    if job.jobStatus == "running":  # O1
        return
"""

SOURCES = {"spec": SPEC, "proxy": PROXY, "buffer": BUFFER}


def _setup(tmp_path: Path):
    root = tmp_path / PACKAGE
    root.mkdir(parents=True, exist_ok=True)
    (root / "__init__.py").write_text("")
    modules = []
    for name, text in SOURCES.items():
        (root / f"{name}.py").write_text(text)
        tree = ast.parse(text)
        pathcond.attach_parents(tree)
        modules.append(
            SourceModule(
                package=PACKAGE,
                rel_path=f"{PACKAGE}/{name}.py",
                tree=tree,
                source=text,
                blob_sha=blob_sha(text),
            )
        )
    calls = sqlwrite.corpus_calls(modules, {PACKAGE: root})
    assert calls is not None
    return calls, {module.rel_path: module.tree for module in modules}


def _file(module: str) -> str:
    return f"{PACKAGE}/{module}.py"


FILE = _file("proxy")


def _ask(
    tmp_path: Path, function: str, marker: str, read: str, module: str = "proxy"
) -> fills.RowHeld:
    calls, trees = _setup(tmp_path)
    func = next(
        node
        for node in ast.walk(trees[_file(module)])
        if isinstance(node, ast.FunctionDef) and node.name == function
    )
    line = next(
        number
        for number, text in enumerate(SOURCES[module].splitlines(), 1)
        if text.endswith(f"  # {marker}")
    )
    statement = next(
        node
        for node in ast.walk(func)
        if isinstance(node, ast.If) and node.lineno == line
    )
    node = next(
        one
        for one in ast.walk(statement.test)
        if isinstance(one, ast.expr) and ast.unparse(one) == read
    )
    return fills.row_fills(calls, _file(module), func, node)


def _keys(fill: fills.Fill) -> set[tuple[str, str, str, tuple[str, ...]]]:
    return {
        (
            read.run.reading.split("FROM ")[1].split()[0],
            test.test.text,
            key,
            tuple(ast.unparse(one.value) for one in values),
        )
        for read in fill.reads
        for test in read.tests
        for key, values in test.binds
    }


def _source(node: ast.AST, module: str = "proxy") -> str:
    return SOURCES[module].splitlines()[node.lineno - 1].strip()  # type: ignore[attr-defined]


def _hops(fill: fills.Fill | fills.Other) -> list[tuple[str, str, str]]:
    return [
        (
            hop.way,
            hop.site.func.name,  # type: ignore[attr-defined]
            _source(hop.site.call, hop.site.file.split("/")[-1][: -len(".py")]),
        )
        for hop in fill.route
    ]


def test_a_column_a_row_reading_call_filled_is_said_with_the_run_and_its_keys(
    tmp_path,
) -> None:
    """``killJob`` 's guard reads what ``job.pack(res[0])`` filled from the rows ``fetchall`` gives: the run on each
    table, its rows' tests and the values filling their binds; no other write reaches, and nothing between changes
    it."""
    held = _ask(tmp_path, "killJob", "K1", "job.jobStatus")
    assert held.others == ()
    (fill,) = held.fills
    assert _source(fill.site) == "job.pack(res[0])"
    assert ast.unparse(fill.fetch) == "self.cur.fetchall()"
    assert fill.pick == fills.PICK_AT
    assert fill.changes == ()
    assert {read.column for read in fill.reads} == {"jobStatus"}
    assert _keys(fill) == {
        (table, text, key, values)
        for table in ("ATLAS_PANDA.jobsDefined4", "ATLAS_PANDA.jobsActive4")
        for text, key, values in (
            ("PandaID=:PandaID", ":PandaID", ("pandaID",)),
            ("jobStatus<>:jobStatus", ":jobStatus", ("'dummy'", "'running'")),
        )
    }


def test_a_store_reaching_beside_the_fill_is_another_write(tmp_path) -> None:
    """After an arm storing ``job.jobStatus``, the guard reads the store or the fill: both are said."""
    held = _ask(tmp_path, "killJob", "K2", "job.jobStatus")
    assert [_source(fill.site) for fill in held.fills] == ["job.pack(res[0])"]
    assert [(_source(one.site), one.kind) for one in held.others] == [
        ('job.jobStatus = "closed"', fills.OTHER_STORE)
    ]


def test_a_call_that_may_change_the_column_in_place_is_kept_on_the_fill(
    tmp_path,
) -> None:
    """``self.touch(job)`` stores ``job.jobStatus`` in the callee: the fill still reaches, with the change on it;
    the function's own stores are others, not changes."""
    held = _ask(tmp_path, "killJob", "K3", "job.jobStatus")
    (fill,) = held.fills
    assert [(change.kind, change.key) for change in fill.changes] == [
        ("store", ".jobStatus")
    ]
    assert {one.kind for one in held.others} == {fills.OTHER_STORE}


def test_a_local_unpacked_from_a_fetched_row_is_its_column(tmp_path) -> None:
    """``updateJobStatus`` 's ``oldJobStatus`` is the second item of ``fetchone`` 's row; the row whole is a fill
    with no column."""
    held = _ask(tmp_path, "updateJobStatus", "U2", "oldJobStatus")
    assert held.others == ()
    (fill,) = held.fills
    assert fill.pick == fills.PICK_ONE
    assert [read.column for read in fill.reads] == ["jobstatus"]
    assert _keys(fill) == {
        ("ATLAS_PANDA.jobsActive4", "PandaID=:PandaID", ":PandaID", ("pandaID",))
    }
    (whole,) = _ask(tmp_path, "updateJobStatus", "U1", "res").fills
    assert [read.column for read in whole.reads] == [""]


def test_a_row_a_loop_takes_of_the_fetched_rows_is_each_row(tmp_path) -> None:
    """``resetJob`` packs each file row of a run keyed by ``PandaID``, the bind written into the call."""
    held = _ask(tmp_path, "resetJob", "E1", "file.jobStatus")
    (fill,) = held.fills
    assert fill.pick == fills.PICK_EACH
    assert _keys(fill) == {
        ("ATLAS_PANDA.filesTable4", "PandaID=:PandaID", ":PandaID", ("pandaID",))
    }


def test_a_fill_takes_the_rows_of_its_own_run_where_later_runs_share_its_binds(
    tmp_path,
) -> None:
    """``peekJob`` fills one ``varMap`` for the job's select, its files' and its metadata's: the ``fetchall`` after
    the job's select is that select's rows, so the fill names its run on each table and not the files' or the
    metadata's, whose stretches start where the job select's ends (:func:`sql.regions`)."""
    held = _ask(tmp_path, "peekJob", "P1", "job.jobStatus")
    assert held.others == ()
    (fill,) = held.fills
    assert _source(fill.site) == "job.pack(res[0])"
    assert {key[0] for key in _keys(fill)} == {
        "ATLAS_PANDA.jobsActive4",
        "ATLAS_PANDA.jobsArchived4",
    }
    func = pathcond.enclosing_function(fill.site)
    assert func is not None
    stretches = sorted(sql.regions(func, sql.executions(func)).values())  # type: ignore[arg-type]
    assert len(stretches) == 3
    assert all(
        earlier[1] < later[0]
        for earlier, later in zip(stretches, stretches[1:], strict=False)
    )


def test_a_field_a_caller_stored_from_a_calls_value_is_read_through_the_calls(
    tmp_path,
) -> None:
    """``process`` reads ``self.job.jobStatus``; ``run`` stored ``self.job`` from ``peekJobs(...)[0]`` before calling
    it, ``peekJobs`` appends what ``proxy.peekJob`` returns, and that is the job it packed from its select.  The
    ``None`` s on the way do not reach: ``peekJobs`` appends ``res`` only ``if res:``, and ``process`` returns first
    ``if self.job is None``.  No ``use_json`` is passed, so ``res.to_dict()`` is not appended."""
    held = _ask(tmp_path, "process", "M1", "self.job.jobStatus", "buffer")
    assert held.others == ()
    (fill,) = held.fills
    assert (fill.file, _source(fill.site)) == (FILE, "job.pack(res[0])")
    assert _hops(fill) == [
        (fills.HOP_UP, "run", "self.process()"),
        (
            fills.HOP_DOWN,
            "run",
            "self.job = self.taskBuffer.peekJobs([self.job_id], fromActive=False)[0]",
        ),
        (fills.HOP_DOWN, "peekJobs", "res = proxy.peekJob(jobID, fromActive)"),
    ]
    assert _keys(fill) == {
        (table, "PandaID=:PandaID", ":PandaID", ("pandaID",))
        for table in ("ATLAS_PANDA.jobsActive4", "ATLAS_PANDA.jobsArchived4")
    }


def test_a_none_no_test_rules_out_reaches_beside_the_fill(tmp_path) -> None:
    """``unchecked`` reads ``self.job.jobStatus`` with no test of ``self.job`` on the way: the ``None`` that
    ``peekJobs`` appends for a job not found reaches, beside the job."""
    held = _ask(tmp_path, "unchecked", "M2", "self.job.jobStatus", "buffer")
    assert [_source(fill.site) for fill in held.fills] == ["job.pack(res[0])"]
    assert [
        (one.kind, one.file, _source(one.site, "buffer")) for one in held.others
    ] == [(fills.OTHER_BINDING, _file("buffer"), "retJobs.append(None)")]


def test_a_parameter_is_read_at_each_call_of_its_function(tmp_path) -> None:
    """``check`` reads its parameter ``job``; ``run`` hands it a job ``peekJobs`` returned, and calls it only
    ``if job is not None``."""
    held = _ask(tmp_path, "check", "C1", "job.jobStatus", "buffer")
    assert held.others == ()
    (fill,) = held.fills
    assert _hops(fill) == [
        (fills.HOP_UP, "run", "self.check(job)"),
        (fills.HOP_DOWN, "run", "job = self.taskBuffer.peekJobs([self.job_id])[0]"),
        (fills.HOP_DOWN, "peekJobs", "res = proxy.peekJob(jobID, fromActive)"),
    ]


def test_a_call_that_may_land_beyond_the_functions_read_is_also_an_other(
    tmp_path,
) -> None:
    """``watch`` 's ``source`` is a ``TaskBuffer`` or a ``Relay``, which hands every name on to what it wraps, so
    ``source.peekJobs`` may run somewhere this reading cannot name (:meth:`Calls.opened`): ``TaskBuffer.peekJobs``
    is read, and the call is an other beside its fill."""
    held = _ask(tmp_path, "watch", "O1", "job.jobStatus", "buffer")
    assert [_source(fill.site) for fill in held.fills] == ["job.pack(res[0])"]
    assert [
        (one.kind, one.file, _source(one.site, "buffer"), one.route)
        for one in held.others
    ] == [
        (
            fills.OTHER_BINDING,
            _file("buffer"),
            "job = source.peekJobs([job_id])[0]",
            (),
        )
    ]


def test_a_value_from_elsewhere_is_another_write(tmp_path) -> None:
    """A parameter is the entry; an object no row filled is its binding; a row the function was handed is not
    followed to a fetch."""
    (entry,) = _ask(tmp_path, "given", "A1", "job.jobStatus").others
    assert entry.kind == fills.OTHER_ENTRY
    (made,) = _ask(tmp_path, "given", "A2", "other.jobStatus").others
    assert (made.kind, _source(made.site)) == (fills.OTHER_BINDING, "other = JobSpec()")
    held = _ask(tmp_path, "given", "A3", "other.jobStatus")
    assert held.fills == ()
    assert [(one.kind, _source(one.site)) for one in held.others] == [
        (fills.OTHER_ROW_UNREAD, "other.pack(rows[0])")
    ]


def test_a_read_is_asked_on_a_tree_with_parent_links_and_of_a_chain_on_a_name(
    tmp_path,
) -> None:
    calls, trees = _setup(tmp_path)
    func = next(
        node
        for node in ast.walk(trees[FILE])
        if isinstance(node, ast.FunctionDef) and node.name == "killJob"
    )
    with pytest.raises(ValueError, match="parent links"):
        fills.row_fills(calls, FILE, func, ast.parse("job.jobStatus", mode="eval").body)
    subscript = next(node for node in ast.walk(func) if isinstance(node, ast.Subscript))
    with pytest.raises(ValueError, match="chain"):
        fills.row_fills(calls, FILE, func, subscript)
