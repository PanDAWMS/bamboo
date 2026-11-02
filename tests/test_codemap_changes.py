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


MAPS = """\
import copy
import json


class Cache:
    def get(self, key):
        self.hits[key] = 1


class Spec:
    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        self._changed[name] = value


class Refiner:
    def fill(self, params):
        params["filled"] = True

    def keeps(self, params):
        return params

    def touch(self):
        self.saved["touched"] = 3

    def read(self, params):
        params.pop("old", None)
        params.update({"merged": 1}, extra=2)
        params["inner"]["deep"] = 4
        self.fill(params)
        alias = self.keeps(params)
        alias["aliased"] = 5
        text = json.dumps(params)
        clone = copy.deepcopy(params)
        clone["copied"] = text
        self.saved = params
        self.log.debug(params)
        if "filled" in params:
            params["late"] = 6
        for _ in range(2):
            if "again" in params:
                pass
            params["again"] = 7
        self.queue.append(params)

    def other(self, params, tasks):
        params.get("x")
        object.__setattr__(self, "saved", params)
        for task in tasks:
            params = json.loads(task)
            if "seen" in params:
                pass
            self.fill(params)

    def holders(self, params, other):
        varMap = {}
        varMap[":p"] = params
        self.cur.execute("SELECT 1", varMap)
        part = getattr(params, "x", None)
        part["k"] = 1
        best = max(params, other)
        best["b"] = 2
        ret = {}
        ret["p"] = params
        if "x" in params:
            pass
        return ret
"""


def _object_changes(
    tmp_path: Path, test: str, name: str = "read", bound_by_loop: bool = False
) -> list[tuple]:
    root = tmp_path / PACKAGE
    root.mkdir(parents=True, exist_ok=True)
    (root / "__init__.py").write_text("")
    (root / "maps.py").write_text(MAPS)
    tree = ast.parse(MAPS)
    pathcond.attach_parents(tree)
    module = SourceModule(
        package=PACKAGE,
        rel_path=f"{PACKAGE}/maps.py",
        tree=tree,
        source=MAPS,
        blob_sha=blob_sha(MAPS),
    )
    calls = sqlwrite.corpus_calls([module], {PACKAGE: root})
    assert calls is not None
    read = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    (last,) = [
        node
        for node in ast.walk(read)
        if (isinstance(node, ast.If) and ast.unparse(node.test) == test)
        or (isinstance(node, ast.For) and ast.unparse(node.iter) == test)
    ]
    line = {text.strip(): number for number, text in enumerate(MAPS.splitlines(), 1)}
    file = f"{PACKAGE}/maps.py"
    first: ast.AST = read.args
    if bound_by_loop:
        (first,) = [
            node
            for node in ast.walk(read)
            if isinstance(node, ast.Assign)
            and ast.unparse(node) == "params = json.loads(task)"
        ]
    return [
        (
            change.kind,
            change.key,
            change.what,
            [one.removeprefix(f"{file}:") for one in change.via],
            None if change.value is None else ast.unparse(change.value),
            next(text for text, number in line.items() if number == change.line),
        )
        for change in changes.object_changes(calls, file, read, "params", first, last)
    ]


def test_every_change_of_an_object_between_its_binding_and_a_read_is_said_by_its_key(
    tmp_path,
):
    """Before ``"filled" in params``: a builtin container's methods by the keys they name, a store inside an item,
    what ``fill`` the map is handed to stores, a store through a name a call hands it back as, and through the
    attribute it is put in, by the attribute's name (``touch``).  A library's function changes nothing it is handed
    -- what ``copy.deepcopy`` hands back is read as the map, ``json.dumps``'s too, and neither is stored on -- nor
    does a logger.  The store under the test and what follows it are after the read."""
    found = _object_changes(tmp_path, "'filled' in params")

    def at(text: str) -> str:
        return str(
            next(
                n for n, held in enumerate(MAPS.splitlines(), 1) if held.strip() == text
            )
        )

    assert found == [
        ("method", "'old'", "pop", [], None, 'params.pop("old", None)'),
        (
            "method",
            "'merged'",
            "update",
            [],
            "1",
            'params.update({"merged": 1}, extra=2)',
        ),
        (
            "method",
            "'extra'",
            "update",
            [],
            "2",
            'params.update({"merged": 1}, extra=2)',
        ),
        (
            "inner",
            "'inner'",
            "params['inner']['deep']",
            [],
            None,
            'params["inner"]["deep"] = 4',
        ),
        (
            "store",
            "'filled'",
            "",
            [at("self.fill(params)")],
            "True",
            'params["filled"] = True',
        ),
        ("store", "'aliased'", "", [], "5", 'alias["aliased"] = 5'),
        ("store", "'copied'", "", [], "text", 'clone["copied"] = text'),
        (
            "store",
            "'touched'",
            "",
            [at("self.saved = params")],
            "3",
            'self.saved["touched"] = 3',
        ),
    ]


def test_a_store_a_loop_runs_after_the_read_is_before_its_next_pass(tmp_path):
    """``params["again"] = 7`` follows ``"again" in params`` in the loop's body, and runs before the test's next
    pass; the store under the earlier test runs before the loop.  What follows the loop does not."""
    found = _object_changes(tmp_path, "'again' in params")

    assert [(kind, key) for kind, key, *_rest in found][-2:] == [
        ("store", "'late'"),
        ("store", "'again'"),
    ]
    assert "escaped" not in {kind for kind, *_rest in found}


def test_a_method_of_no_known_class_is_the_builtin_s_and_one_set_by_name_is_read_by_the_name(
    tmp_path,
):
    """``params.get("x")`` is the dict's, not every ``get`` of the corpus (``Cache.get`` stores on its own object);
    ``object.__setattr__(self, "saved", params)`` puts the map in ``saved``, through which ``touch`` stores, and no
    ``__setattr__`` of the corpus is followed -- all before the loop, which reads ``tasks``.  In the loop, a call
    after the read, before the name is bound again, changes an object the next pass does not read."""
    found = _object_changes(tmp_path, "tasks", "other")

    assert [(kind, key, via) for kind, key, _what, via, _value, _text in found] == [
        (
            "store",
            "'touched'",
            [
                str(
                    MAPS.splitlines().index(
                        '        object.__setattr__(self, "saved", params)'
                    )
                    + 1
                )
            ],
        ),
    ]
    assert (
        _object_changes(tmp_path, "'seen' in params", "other", bound_by_loop=True) == []
    )


def test_a_local_container_handing_nothing_out_keeps_no_change_and_a_builtin_hands_back_an_argument_only(
    tmp_path,
):
    """``varMap`` only takes the map and goes to the driver: nothing changes it through there.  ``ret``, handed
    back, may.  ``getattr`` hands back an attribute of the map, not the map; ``max`` one of its arguments."""
    found = _object_changes(tmp_path, "'x' in params", "holders")

    assert [(kind, key, what) for kind, key, what, *_rest in found] == [
        ("store", "'b'", ""),
        ("escaped", "*", "ret['p']"),
    ]
