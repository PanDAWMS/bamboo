"""Where a call lands, what it is handed, and what comes back -- ``codemap.calls``."""

from __future__ import annotations

import ast
from pathlib import Path

from bamboo.codemap import calls as calls_mod

PACKAGE = "pkg"


def _tree(tmp_path: Path, **files: str) -> dict[str, Path]:
    root = tmp_path / PACKAGE
    root.mkdir(parents=True, exist_ok=True)
    (root / "__init__.py").write_text("")
    for name, text in files.items():
        (root / f"{name}.py").write_text(text)
    return {PACKAGE: root}


def _calls(roots):
    return calls_mod.Calls(calls_mod.Types(roots, {}))


def _function(calls, file, name):
    tree = calls.types.tree(f"{PACKAGE}/{file}.py")
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    )


def _call_in(func, text):
    return next(
        node for node in ast.walk(func) if isinstance(node, ast.Call) and ast.unparse(node.func) == text
    )


def _placed(calls, file, func_name, call_text):
    func = _function(calls, file, func_name)
    return calls.callees(f"{PACKAGE}/{file}.py", func, _call_in(func, call_text))


def test_a_method_on_self_lands_in_the_class_or_a_base(tmp_path):
    roots = _tree(
        tmp_path,
        base="class Base:\n    def helper(self):\n        return 1\n",
        m="from pkg.base import Base\n\nclass C(Base):\n    def run(self):\n        return self.helper()\n",
    )
    calls = _calls(roots)

    placed = _placed(calls, "m", "run", "self.helper")

    (target,) = placed.targets
    assert (target.file, target.func.name, target.cls.name) == ("pkg/base.py", "helper", "Base")


def test_a_closure_calling_self_reads_the_enclosing_methods_class(tmp_path):
    """``getJobs`` builds its WHERE in a closure that calls ``self.construct_where_clause``."""
    roots = _tree(
        tmp_path,
        m='''\
class Proxy:
    def construct(self, site):
        return "WHERE site=:site", {":site": site}

    def get_jobs(self, site):
        def build():
            return self.construct(site)
        where, binds = build()
        return where
''',
    )
    calls = _calls(roots)

    (outer,) = _placed(calls, "m", "get_jobs", "build").targets
    (inner,) = _placed(calls, "m", "build", "self.construct").targets

    assert outer.func.name == "build"
    assert inner.func.name == "construct"


def test_a_receiver_is_placed_by_what_it_is_declared_to_be_not_by_its_name(tmp_path):
    roots = _tree(
        tmp_path,
        mapper="class SiteMapper:\n    def getSite(self, name):\n        return name\n",
        m='''\
from pkg.mapper import SiteMapper

def f(site_mapper: SiteMapper, other):
    site_mapper.getSite("a")
    other.getSite("b")
''',
    )
    calls = _calls(roots)

    assert _placed(calls, "m", "f", "site_mapper.getSite").targets
    unplaced = _placed(calls, "m", "f", "other.getSite")
    assert unplaced.targets == ()
    assert unplaced.unplaced == calls_mod.UNPLACED_RECEIVER


def test_a_module_instance_is_what_the_module_constructs(tmp_path):
    """``taskBuffer = TaskBuffer()`` and ``rucioAPI = RucioAPI()`` are what daemons import."""
    roots = _tree(
        tmp_path,
        ddm="class RucioAPI:\n    def register(self, name):\n        return name\n\nrucioAPI = RucioAPI()\n",
        m="from pkg.ddm import rucioAPI\n\ndef f():\n    return rucioAPI.register('x')\n",
    )
    calls = _calls(roots)

    (target,) = _placed(calls, "m", "f", "rucioAPI.register").targets

    assert (target.cls.name, target.func.name) == ("RucioAPI", "register")


def test_a_with_binding_is_the_element_its_context_manager_yields(tmp_path):
    """``with self.proxyPool.get() as proxy`` is a ``DBProxy``, not an iterator."""
    roots = _tree(
        tmp_path,
        proxy="class DBProxy:\n    def peek(self, i):\n        return i\n",
        pool='''\
from typing import Iterator
from pkg import proxy as DBProxy

class Pool:
    def get(self) -> Iterator["DBProxy.DBProxy"]:
        yield DBProxy.DBProxy()
''',
        m='''\
from pkg.pool import Pool

class Buffer:
    def __init__(self, pool: Pool):
        self.proxyPool = pool

    def peek(self, i):
        with self.proxyPool.get() as proxy:
            return proxy.peek(i)
''',
    )
    calls = _calls(roots)

    (target,) = _placed(calls, "m", "peek", "proxy.peek").targets

    assert target.cls.name == "DBProxy"


def test_a_facade_that_forwards_every_call_is_said_to_and_not_guessed_through(tmp_path):
    roots = _tree(
        tmp_path,
        m='''\
from typing import Any

class Interface:
    def __getattr__(self, name: str) -> Any:
        return getattr(self.remote, name)

def f(tb: Interface):
    return tb.getPandaIDs(1)
''',
    )
    calls = _calls(roots)

    placed = _placed(calls, "m", "f", "tb.getPandaIDs")

    assert placed.unplaced == calls_mod.UNPLACED_FORWARDED


def test_a_base_class_read_off_by_name_is_a_method_of_that_class(tmp_path):
    roots = _tree(
        tmp_path,
        m='''\
class Base:
    def __init__(self, x):
        self.x = x

class Child(Base):
    def __init__(self, x):
        Base.__init__(self, x)
''',
    )
    calls = _calls(roots)

    tree = calls.types.tree("pkg/m.py")
    child = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Child")
    init = child.body[0]
    (target,) = calls.callees("pkg/m.py", init, _call_in(init, "Base.__init__")).targets

    assert target.cls.name == "Base"


def test_a_call_binds_positional_keyword_and_default_arguments(tmp_path):
    roots = _tree(
        tmp_path,
        m='''\
class Proxy:
    def lock(self, limit, states, labels, only=False, *, sites=()):
        return limit

def f(proxy: Proxy, t):
    proxy.lock(t, ["activated"], labels=["managed"])
''',
    )
    calls = _calls(roots)
    func = _function(calls, "m", "f")
    call = _call_in(func, "proxy.lock")
    (target,) = calls.callees("pkg/m.py", func, call).targets

    bound = {name: ast.unparse(value) if value is not None else None for name, value in calls_mod.bind(call, target).items()}

    assert bound == {
        "limit": "t",
        "states": "['activated']",
        "labels": "['managed']",
        "only": "False",
        "sites": "()",
    }


def test_a_spread_argument_leaves_what_it_could_fill_unsaid(tmp_path):
    roots = _tree(
        tmp_path,
        m="def g(a, b=1):\n    return a\n\ndef f(args):\n    return g(*args)\n",
    )
    calls = _calls(roots)
    func = _function(calls, "m", "f")
    call = _call_in(func, "g")
    (target,) = calls.callees("pkg/m.py", func, call).targets

    assert calls_mod.bind(call, target) == {"a": None, "b": None}


def test_what_a_function_returns_is_every_return_in_its_own_body(tmp_path):
    roots = _tree(
        tmp_path,
        m='''\
def f(x):
    def inner():
        return "not mine"
    if x:
        return "a", 1
    return
''',
    )
    calls = _calls(roots)

    returned = [ast.unparse(value) for value in calls_mod.returns(_function(calls, "m", "f"))]

    assert returned == ["('a', 1)", "None"]


def test_a_method_that_sets_a_field_is_a_write_of_it(tmp_path):
    """``set_fatal()`` is ``status_code = self.rc_fatal`` at the call that makes it."""
    roots = _tree(
        tmp_path,
        m='''\
class Result:
    def __init__(self):
        self.rc_fatal = 2
        self.status_code = 0

    def set_fatal(self):
        self.status_code = self.rc_fatal

    def fail_hard(self):
        self.set_fatal()
        self.errors.append("hard")
''',
    )
    calls = _calls(roots)
    cls_file, cls = calls.types.class_named("pkg/m.py", ast.Name(id="Result", ctx=ast.Load()))
    target = calls_mod.Target("pkg/m.py", _function(calls, "m", "fail_hard"), cls)

    writes = [(field, ast.unparse(value)) for field, value, _site, _where in calls.writes(target)]

    assert writes == [("status_code", "self.rc_fatal"), ("errors", "'hard'")]


def test_the_callers_of_a_function_are_the_calls_placed_there_and_no_others(tmp_path):
    """A name is how the candidates are found, never how they are kept."""
    roots = _tree(
        tmp_path,
        a="class A:\n    def run(self):\n        return 1\n",
        b="class B:\n    def run(self):\n        return 2\n",
        m='''\
from pkg.a import A
from pkg.b import B

def f(one: A, two: B, other):
    one.run()
    two.run()
    other.run()
''',
    )
    calls = _calls(roots)
    file, cls = calls.types.class_named("pkg/a.py", ast.Name(id="A", ctx=ast.Load()))
    run = next(node for node in cls.body if isinstance(node, ast.FunctionDef))

    sites = calls.callers(calls_mod.Target(file, run, cls))

    assert [ast.unparse(site.call.func) for site in sites] == ["one.run"]


def test_a_function_is_what_its_return_annotation_says_it_returns(tmp_path):
    """``get_task_utils_module(self)``, declared ``-> TaskUtilsModule``."""
    roots = _tree(
        tmp_path,
        utils='''\
class TaskUtilsModule:
    def getScout(self, task_id):
        return task_id


def get_task_utils_module(base) -> TaskUtilsModule:
    return base.get_composite_module("task_utils")
''',
        m="from pkg.utils import get_task_utils_module\n\ndef f(self, t):\n    return get_task_utils_module(self).getScout(t)\n",
    )
    calls = _calls(roots)

    (target,) = _placed(calls, "m", "f", "get_task_utils_module(self).getScout").targets

    assert (target.cls.name, target.func.name) == ("TaskUtilsModule", "getScout")


def test_a_module_instance_declared_with_an_annotation_is_of_that_class(tmp_path):
    """``global_task_buffer: TaskBuffer = None``: the annotation, not the placeholder."""
    roots = _tree(
        tmp_path,
        tb="class TaskBuffer:\n    def peek(self, i):\n        return i\n",
        m="from pkg.tb import TaskBuffer\n\nglobal_task_buffer: TaskBuffer = None\n\ndef f():\n    return global_task_buffer.peek(1)\n",
    )
    calls = _calls(roots)

    (target,) = _placed(calls, "m", "f", "global_task_buffer.peek").targets

    assert target.cls.name == "TaskBuffer"


def test_an_item_of_a_declared_container_is_of_its_element_class_through_an_alias(tmp_path):
    """``target = self.nuclei`` then ``target[name].add(...)``, with
    ``self.nuclei: dict[str, NucleusSpec]``."""
    roots = _tree(
        tmp_path,
        nucleus="class NucleusSpec:\n    def add(self, site):\n        return site\n",
        m='''\
from pkg.nucleus import NucleusSpec


class SiteMapper:
    def __init__(self):
        self.nuclei: dict[str, NucleusSpec] = {}

    def collect(self, name, site):
        target = self.nuclei
        target[name].add(site)
''',
    )
    calls = _calls(roots)

    (target,) = _placed(calls, "m", "collect", "target[name].add").targets

    assert target.cls.name == "NucleusSpec"
