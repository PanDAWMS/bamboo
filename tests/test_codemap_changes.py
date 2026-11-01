"""Whether a call may change a value an attribute chain holds (bamboo.codemap.panda.changes)."""

from __future__ import annotations

import ast
from pathlib import Path

from bamboo.codemap.calls import UNPLACED_DYNAMIC
from bamboo.codemap.gitsource import blob_sha
from bamboo.codemap.models import SourceModule
from bamboo.codemap.panda import changes, pathcond
from bamboo.codemap.panda.recognizers import sqlwrite

PACKAGE = "pkg"

SPEC = """\
class JobSpec:
    PandaID: int
    jobStatus: str

    def valuesMap(self):
        ret = {}
        for attr in ("PandaID", "jobStatus"):
            ret[f":{attr}"] = getattr(self, attr)
        return ret

    def is_push_job(self):
        return self.check_special_handling("pushJob")

    def check_special_handling(self, key):
        return key in (self.specialHandling or "").split(",")

    def set_status(self, status):
        self.jobStatus = status

    def renumber(self, other):
        other.PandaID = 0

    def pack(self, values):
        for name, value in zip(self.names, values):
            setattr(self, name, value)

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        self._changed[name] = value
"""

PROXY = """\
import json

from pkg.spec import JobSpec


class Proxy:
    def push(self, job: JobSpec):
        tmp_log = self.make_logger()
        tmp_log.debug(json.dumps({"id": job.PandaID, "name": self.__class__.__name__}))
        self.cur.execute("UPDATE t SET x=:x", {":x": 1})
        raise RuntimeError("no broker")

    def again(self, job: JobSpec, n):
        if n:
            return self.again(job, n - 1)
        return job.is_push_job()

    def sort(self, jobs: list):
        return sorted(jobs, key=lambda one: one.set_status("sorted"))

    def unknown(self, job: JobSpec, thing):
        thing.renumber(job)

    def chosen(self, job: JobSpec, name):
        getattr(self, name)(job)

    def copy_status(self, job: JobSpec, other: JobSpec):
        other.jobStatus = job.jobStatus
        other.PandaID = job.PandaID

    def renew(self, job: JobSpec):
        new = JobSpec()
        new.PandaID = job.PandaID
        return new

    def keep(self, job: JobSpec):
        self.kept.append(job)
        self.reset_kept()

    def reset_kept(self):
        for one in self.kept:
            one.PandaID = 0

    def a(self, job: JobSpec):
        job.valuesMap()
        job.is_push_job()
        job.set_status("done")
        job.pack([1])
        self.push(job)
        self.again(job, 3)
        self.sort([job])
        self.unknown(job, object())
        vars(job)
        self.chosen(job, "x")
        self.copy_status(job, self.other)
        self.renew(job)
        self.keep(job)
"""


def _calls(tmp_path: Path):
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
    proxy = next(
        module.tree for module in modules if module.rel_path.endswith("proxy.py")
    )
    a = next(
        node
        for node in ast.walk(proxy)
        if isinstance(node, ast.FunctionDef) and node.name == "a"
    )
    made = {
        ast.unparse(statement.value): statement.value
        for statement in a.body
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)
    }
    return calls, a, made


def _asked(tmp_path: Path, attribute: str) -> dict[str, object]:
    calls, a, made = _calls(tmp_path)
    value = ast.parse(f"job.{attribute}", mode="eval").body
    found: dict[str, object] = {}
    for text, call in made.items():
        change = changes.may_change(calls, f"{PACKAGE}/proxy.py", a, call, value)
        found[text] = (
            None if change is None else (change.name, change.why, change.detail)
        )
    return found


def test_a_call_changes_a_value_only_where_what_it_runs_stores_its_attribute_on_the_object_handed(
    tmp_path,
):
    """The object is followed into what the call runs, under the names it is handed to: ``job.valuesMap()`` and
    ``is_push_job`` only read it, and ``self.again(job, 3)`` coming back to itself adds nothing.  A store on
    another object does not count -- ``other.PandaID = job.PandaID``, a ``JobSpec()`` built anew -- nor do a logger's,
    the driver's or a library's calls, nor a builtin class built.  ``thing.renumber(job)`` on a receiver of no known
    class lands in every ``renumber`` of the corpus, and hands ``job`` to ``other``, which it sets.  A name computed
    on the object and a builtin handing back its attributes may write every name; a callee chosen at run time loses
    it, and so does ``[job]``: then the call is read by the attribute's name."""
    found = _asked(tmp_path, "PandaID")

    assert found == {
        "job.valuesMap()": None,
        "job.is_push_job()": None,
        "job.set_status('done')": None,
        "job.pack([1])": (changes.EVERY, changes.COMPUTED, "setattr"),
        "self.push(job)": None,
        "self.again(job, 3)": None,
        "self.sort([job])": None,
        "self.unknown(job, object())": ("PandaID", changes.STORED, ""),
        "vars(job)": (changes.EVERY, changes.COMPUTED, "vars"),
        "self.chosen(job, 'x')": (changes.EVERY, changes.UNREAD, UNPLACED_DYNAMIC),
        "self.copy_status(job, self.other)": None,
        "self.renew(job)": None,
        "self.keep(job)": ("PandaID", changes.STORED, ""),
    }


def test_an_object_lost_is_read_by_the_attribute_s_name_and_a_spec_s_setattr_writes_the_name_assigned(
    tmp_path,
):
    """``sorted([job], key=lambda one: one.set_status(...))`` puts the object in a list: by the name, the lambda
    sets ``jobStatus``.  ``self.keep(job)`` keeps it in ``self.kept``, and ``reset_kept`` sets ``PandaID`` on what
    it holds.  A spec's ``__setattr__`` hands ``object.__setattr__`` the name its caller assigned, so read by the
    name it writes no other."""
    found = _asked(tmp_path, "jobStatus")

    assert found["self.sort([job])"] == ("jobStatus", changes.STORED, "")
    assert found["job.set_status('done')"] == ("jobStatus", changes.STORED, "")
    assert found["job.valuesMap()"] is None
    assert found["self.copy_status(job, self.other)"] is None
    assert found["self.keep(job)"] is None
