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


def _setup(tmp_path: Path):
    root = tmp_path / PACKAGE
    root.mkdir(parents=True, exist_ok=True)
    (root / "__init__.py").write_text("")
    modules = []
    for name, text in {"spec": SPEC, "proxy": PROXY}.items():
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
    proxy = next(
        module.tree for module in modules if module.rel_path.endswith("proxy.py")
    )
    return calls, proxy


LINES = {
    text.rsplit("# ", 1)[1]: number
    for number, text in enumerate(PROXY.splitlines(), 1)
    if "  # " in text
}
FILE = f"{PACKAGE}/proxy.py"


def _ask(tmp_path: Path, function: str, marker: str, read: str) -> fills.RowHeld:
    calls, proxy = _setup(tmp_path)
    func = next(
        node
        for node in ast.walk(proxy)
        if isinstance(node, ast.FunctionDef) and node.name == function
    )
    statement = next(
        node
        for node in ast.walk(func)
        if isinstance(node, ast.If) and node.lineno == LINES[marker]
    )
    node = next(
        one
        for one in ast.walk(statement.test)
        if isinstance(one, ast.expr) and ast.unparse(one) == read
    )
    return fills.row_fills(calls, FILE, func, node)


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


def _source(node: ast.AST) -> str:
    return PROXY.splitlines()[node.lineno - 1].strip()  # type: ignore[attr-defined]


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
    calls, proxy = _setup(tmp_path)
    func = next(
        node
        for node in ast.walk(proxy)
        if isinstance(node, ast.FunctionDef) and node.name == "killJob"
    )
    with pytest.raises(ValueError, match="parent links"):
        fills.row_fills(calls, FILE, func, ast.parse("job.jobStatus", mode="eval").body)
    subscript = next(node for node in ast.walk(func) if isinstance(node, ast.Subscript))
    with pytest.raises(ValueError, match="chain"):
        fills.row_fills(calls, FILE, func, subscript)
