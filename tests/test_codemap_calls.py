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


def test_a_base_with_type_arguments_is_still_the_base_a_method_lands_in(tmp_path):
    """Eight JEDI factories are ``FactoryBase[...]``, and ``self.getImpl`` there is ``FactoryBase``'s.

    A subscript has no name, and the lineage dropped it: the class read as
    deriving from nothing, its inherited calls landed nowhere, and its
    ``FactoryBase.__init__(self, ...)`` call reached no constructor.
    """
    roots = _tree(
        tmp_path,
        base=(
            "class FactoryBase(Generic[T]):\n"
            "    def __init__(self, vos, logger):\n        self.logger = logger\n"
            "    def getImpl(self, vo):\n        return None\n"
        ),
        m=(
            "from pkg.base import FactoryBase\n\n"
            "class JobBroker(FactoryBase[JobBrokerBase]):\n"
            "    def run(self):\n        return self.getImpl('atlas')\n"
        ),
    )
    calls = _calls(roots)

    (target,) = _placed(calls, "m", "run", "self.getImpl").targets
    cls = next(n for n in calls.types.tree("pkg/m.py").body if isinstance(n, ast.ClassDef))

    assert (target.file, target.func.name) == ("pkg/base.py", "getImpl")
    assert [held.name for _where, held in calls.types.lineage("pkg/m.py", cls)] == ["JobBroker", "FactoryBase"]


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


def test_a_name_rebound_from_names_rebound_many_times_is_read_once_per_depth(tmp_path):
    """Each name read once, not once per path to it.

    ``JobGenerator`` rebinds locals from calls on locals that are themselves
    rebound, and reading every assignment of every name afresh made one
    receiver cost the product of their counts -- a census stopped in one file
    for hours.  Five names, six assignments each: the answer is unchanged and
    the reading is linear in them.
    """
    chain = ["def run(n0: Spec):"]
    for level in range(1, 5):
        chain += [f"    n{level} = n{level - 1}.step()"] * 6
    chain.append("    return n4.step()")
    roots = _tree(
        tmp_path,
        m="class Spec:\n    def step(self) -> 'Spec':\n        return self\n\n\n"
        + "\n".join(chain)
        + "\n",
    )
    calls = _calls(roots)
    types = calls.types
    asked = []
    read = types.of_expression

    def counted(*args, **kwargs):
        asked.append(args)
        return read(*args, **kwargs)

    types.of_expression = counted
    func = _function(calls, "m", "run")
    returned = next(node for node in ast.walk(func) if isinstance(node, ast.Return))

    (target,) = calls.callees("pkg/m.py", func, returned.value).targets

    assert target.cls.name == "Spec"
    assert len(asked) < 100


# -- statements and binds read through calls (``sql.across_calls``) ---------


def _runs(calls, file, name):
    from bamboo.codemap.panda import sql

    func = _function(calls, file, name)
    with sql.across_calls(calls):
        return func, sql.executions(func)


def test_an_in_list_helper_is_read_as_one_bind_holding_the_list(tmp_path):
    from bamboo.codemap.panda import sql

    roots = _tree(
        tmp_path,
        db=(
            "def lock(cur, statList):\n"
            "    varMap = {}\n"
            "    sql = 'SELECT PandaID FROM jobsActive4 WHERE a=1 '\n"
            "    names, stat_map = get_sql_IN_bind_variables(statList, prefix=':stat')\n"
            "    sql += f'AND jobStatus IN ({names}) '\n"
            "    varMap.update(stat_map)\n"
            "    cur.execute(sql, varMap)\n"
        ),
    )
    calls = _calls(roots)
    func, runs = _runs(calls, "db", "lock")

    (run,) = runs
    assert "jobStatus IN (:stat__in)" in run.sql
    assert run.unresolved == ()
    (bind,) = sql.bound_values(func, "varMap", ":stat__in")
    assert ast.unparse(bind.value) == "statList"


def test_a_value_is_settled_through_the_list_a_caller_hands_the_parameter(tmp_path):
    from bamboo.codemap.panda import values

    roots = _tree(
        tmp_path,
        db="def lock(statList):\n    return statList\n",
        tb="from pkg.db import lock\n\ndef forward(stats):\n    return lock(stats)\n",
        main="from pkg.tb import forward\n\ndef run():\n    forward(['activated', 'throttled'])\n",
    )
    calls = _calls(roots)
    func = _function(calls, "db", "lock")
    settle = values.handed(values.resolver(), calls)

    found = settle(ast.Name(id="statList"), func, "pkg/db.py")

    assert sorted(one.value for one in found) == ["activated", "throttled"]
    assert {one.via for one in found} == {("pkg/tb.py:4", "pkg/main.py:4")}


def test_a_value_a_test_around_the_read_rules_out_is_not_handed_there(tmp_path):
    """``getEventRanges`` binds ``:datasetID`` under ``if segment_id is not None:``; a call handing None never runs it."""
    from bamboo.codemap.panda import values

    roots = _tree(
        tmp_path,
        db=(
            "def get(segment_id):\n"
            "    vm = {}\n"
            "    if segment_id is not None:\n"
            "        vm[':datasetID'] = segment_id\n"
            "    return vm\n"
        ),
        tb="from pkg.db import get\n\ndef forward(segment_id):\n    return get(segment_id)\n",
        main=(
            "from pkg.db import get\n"
            "from pkg.tb import forward\n"
            "\n"
            "def run():\n"
            "    get(None)\n"
            "    forward(None)\n"
            "    get(7)\n"
        ),
    )
    calls = _calls(roots)
    func = _function(calls, "db", "get")
    settle = values.handed(values.resolver(), calls)
    (read,) = [
        node.value for node in ast.walk(func) if isinstance(node, ast.Assign) and isinstance(node.value, ast.Name)
    ]

    found = settle(read, func, "pkg/db.py")

    assert [(one.value, one.via) for one in found] == [("7", ("pkg/main.py:7",))]
    # Where nothing tests it, every call's value is the parameter's.
    assert sorted(one.value for one in settle(ast.Name(id="segment_id"), func, "pkg/db.py")) == ["7", "None", "None"]


def test_a_call_handing_a_flag_says_what_the_statement_refuses_and_requires_there(tmp_path):
    """``copyArchive`` hands ``lockJobsForReassign`` ``onlyReassignable=True`` at two calls of four.

    Only there is ``relocationFlag=2`` refused -- the statement's own reading,
    over every call, cannot say it -- and only there is the test required.
    """
    from bamboo.codemap.gitsource import blob_sha
    from bamboo.codemap.models import SourceModule
    from bamboo.codemap.panda import attribution, sql
    from bamboo.codemap.panda.recognizers import progress, sqlwrite

    db = (
        "def lock(cur, status, onlyReassignable):\n"
        "    varMap = {':status': status, ':relocationFlag': 2}\n"
        "    sql = 'SELECT PandaID FROM ATLAS_PANDA.jobsDefined4 WHERE jobStatus=:status '\n"
        "    if onlyReassignable:\n"
        "        sql += 'AND (relocationFlag IS NULL OR relocationFlag<>:relocationFlag) '\n"
        "    cur.execute(sql, varMap)\n"
    )
    main = "from pkg.db import lock\n\ndef run(cur):\n    lock(cur, 'defined', False)\n    lock(cur, 'activated', True)\n"
    roots = _tree(tmp_path, db=db, main=main)
    modules = [
        SourceModule(package=PACKAGE, rel_path=f"{PACKAGE}/{name}.py", tree=ast.parse(text), source=text, blob_sha=blob_sha(text))
        for name, text in (("db", db), ("main", main))
    ]
    attributor = attribution.SpecAttributor(progress.spec_attributes(modules), attribution.class_bases(modules))
    calls = sqlwrite.corpus_calls(modules, roots)

    with sql.across_calls(calls):
        side = sqlwrite.read_side(modules, attributor, calls)

    (statement,) = side.statements.values()
    # Over every call the test is appended on some paths only: neither
    # refused nor required.
    assert statement.refuses == frozenset()
    assert statement.conditions == []
    assert statement.callers_complete and statement.callers_stop == ()
    unreassignable, reassignable = statement.call_readings
    assert (unreassignable.chain, unreassignable.constants) == (["pkg/main.py:4"], {"onlyReassignable": "False"})
    assert (unreassignable.selects, unreassignable.excludes, unreassignable.refuses, unreassignable.condition_trees) == (
        ["jobsDefined4.jobStatus=defined"], [], [], []
    )
    assert (reassignable.chain, reassignable.constants) == (["pkg/main.py:5"], {"onlyReassignable": "True"})
    assert (reassignable.selects, reassignable.refuses, reassignable.restricts) == (
        ["jobsDefined4.jobStatus=activated"], ["jobsDefined4.relocationFlag=2"], ["jobsDefined4.jobStatus=activated"]
    )
    assert [one.entry for one in reassignable.condition_trees] == [
        "relocationFlag IS NULL OR relocationFlag<>:relocationFlag@2"
    ]


def test_a_route_reading_as_the_statement_says_so_and_one_settling_more_does_not(tmp_path):
    """Each route is read in full and compared with the statement, field by field off the model.

    The two calls of ``scan`` hand the same flag and nothing else, and read
    as the statement does.  The two of ``lock`` hand time limits of their own:
    the statement waits for the parameter, each route settles it, and so
    neither reads as the statement -- though no constant differs.
    """
    from bamboo.codemap.gitsource import blob_sha
    from bamboo.codemap.models import CALL_READING_ROUTE, CallReading, SourceModule
    from bamboo.codemap.panda import attribution, sql
    from bamboo.codemap.panda.recognizers import progress, sqlwrite

    db = (
        "def scan(cur, active):\n"
        "    sql = 'SELECT PandaID FROM ATLAS_PANDA.jobsActive4 WHERE jobStatus=:s'\n"
        "    varMap = {':s': 'running'}\n"
        "    cur.execute(sql, varMap)\n"
        "\n"
        "def lock(cur, minutes):\n"
        "    sql = 'SELECT PandaID FROM ATLAS_PANDA.jobsActive4 WHERE jobStatus=:s AND modificationTime<:t'\n"
        "    varMap = {':s': 'running', ':t': naive_utcnow() - datetime.timedelta(minutes=minutes)}\n"
        "    cur.execute(sql, varMap)\n"
    )
    main = (
        "from pkg.db import lock, scan\n\n"
        "def run(cur):\n    scan(cur, True)\n    scan(cur, True)\n    lock(cur, 30)\n    lock(cur, 60)\n"
    )
    roots = _tree(tmp_path, db=db, main=main)
    modules = [
        SourceModule(package=PACKAGE, rel_path=f"{PACKAGE}/{name}.py", tree=ast.parse(text), source=text, blob_sha=blob_sha(text))
        for name, text in (("db", db), ("main", main))
    ]
    attributor = attribution.SpecAttributor(progress.spec_attributes(modules), attribution.class_bases(modules))
    calls = sqlwrite.corpus_calls(modules, roots)

    with sql.across_calls(calls):
        side = sqlwrite.read_side(modules, attributor, calls)

    scanned = next(one for (owner, _line), one in side.statements.items() if owner.endswith("::scan"))
    locked = next(one for (owner, _line), one in side.statements.items() if owner.endswith("::lock"))
    assert [one.as_statement for one in scanned.call_readings] == [True, True]
    # A record reading as the statement carries nothing; expanded, it is the statement.
    assert all(one.selects == [] for one in scanned.call_readings)
    assert sorted(f"{s}={v}" for s, v in scanned.selects) == ["jobsActive4.jobStatus=running"]
    assert [one.as_statement for one in locked.call_readings] == [False, False]
    offsets = [
        one.condition_trees[0].tree.items[0].items[0].values[0].offset_seconds for one in locked.call_readings
    ]
    assert offsets == [-1800.0, -3600.0]
    (statement_tree,) = locked.condition_trees
    assert statement_tree.tree.items[0].items[0].values[0].needs == ["lock.minutes"]
    # The comparison is every field of the model but the route's own: one
    # added later is compared without anyone listing it.
    compared = {name for name in CallReading.model_fields if name not in CALL_READING_ROUTE}
    assert compared == {"tables", "selects", "excludes", "consults", "refuses", "restricts", "condition_trees"}
    assert CALL_READING_ROUTE == {"chain", "constants", "not_run", "as_statement", "open"}


def test_a_call_requiring_several_conditions_reads_each_as_a_tree_of_its_own(tmp_path):
    """``A AND B OR C`` is ``(A AND B) OR C`` to SQL: the ``OR`` term is one condition, its own tree."""
    from bamboo.codemap.gitsource import blob_sha
    from bamboo.codemap.models import SourceModule
    from bamboo.codemap.panda import attribution, sql
    from bamboo.codemap.panda.recognizers import progress, sqlwrite

    db = (
        "def lock(cur, status, onlyReassignable):\n"
        "    varMap = {':status': status, ':relocationFlag': 2, ':timeLimit': 1}\n"
        "    sql = 'SELECT PandaID FROM ATLAS_PANDA.jobsDefined4 WHERE jobStatus=:status '\n"
        "    if onlyReassignable:\n"
        "        sql += 'AND modificationTime<:timeLimit '\n"
        "        sql += 'AND (relocationFlag IS NULL OR relocationFlag<>:relocationFlag) '\n"
        "    cur.execute(sql, varMap)\n"
    )
    main = "from pkg.db import lock\n\ndef run(cur):\n    lock(cur, 'defined', False)\n    lock(cur, 'activated', True)\n"
    roots = _tree(tmp_path, db=db, main=main)
    modules = [
        SourceModule(package=PACKAGE, rel_path=f"{PACKAGE}/{name}.py", tree=ast.parse(text), source=text, blob_sha=blob_sha(text))
        for name, text in (("db", db), ("main", main))
    ]
    attributor = attribution.SpecAttributor(progress.spec_attributes(modules), attribution.class_bases(modules))
    calls = sqlwrite.corpus_calls(modules, roots)

    with sql.across_calls(calls):
        side = sqlwrite.read_side(modules, attributor, calls)

    (statement,) = side.statements.values()
    _, reassignable = statement.call_readings
    assert [(one.entry, one.tree.items[0].items[0].node if len(one.tree.items) == 1 else "any") for one in reassignable.condition_trees] == [
        ("modificationTime<:timeLimit@2", "compare"),
        ("relocationFlag IS NULL OR relocationFlag<>:relocationFlag@2", "any"),
    ]


def test_a_where_clause_a_call_returns_is_read_into_the_statement_with_its_binds(tmp_path):
    from bamboo.codemap.panda import sql

    roots = _tree(
        tmp_path,
        db=(
            "class DB:\n"
            "    def where(self, label):\n"
            "        vm = {':status': 'activated'}\n"
            "        clause = 'WHERE jobStatus=:status '\n"
            "        if label:\n"
            "            clause += 'AND label=:label '\n"
            "            vm[':label'] = label\n"
            "        return clause, vm\n"
            "\n"
            "    def get(self, cur, label):\n"
            "        def build():\n"
            "            c, v = self.where(label)\n"
            "            return c, v\n"
            "        clause, orig = build()\n"
            "        vm = dict(orig)\n"
            "        sql = 'SELECT PandaID FROM jobsActive4 '\n"
            "        sql += clause\n"
            "        cur.execute(sql, vm)\n"
        ),
    )
    calls = _calls(roots)
    func, runs = _runs(calls, "db", "get")

    (run,) = runs
    assert run.sql == "SELECT PandaID FROM jobsActive4 WHERE jobStatus=:status AND label=:label "
    assert run.unresolved == ()
    with sql.across_calls(calls):
        binds = sql.bound_values(func, "vm", ":status", across=True)
        assert sql.bound_values(func, "vm", ":status") == []
    assert [ast.unparse(bind.value) for bind in binds] == ["'activated'"]


def test_a_callee_building_two_statements_hands_over_two_alternatives_not_one(tmp_path):
    roots = _tree(
        tmp_path,
        db=(
            "def order(by_age):\n"
            "    if by_age:\n"
            "        text = 'ORDER BY age'\n"
            "    else:\n"
            "        text = 'ORDER BY share'\n"
            "    return text\n"
            "\n"
            "def get(cur, by_age):\n"
            "    tail = order(by_age)\n"
            "    sql = 'SELECT PandaID FROM jobsActive4 '\n"
            "    sql += tail\n"
            "    cur.execute(sql, {})\n"
        ),
    )
    calls = _calls(roots)
    _func, runs = _runs(calls, "db", "get")

    assert sorted(run.sql for run in runs) == [
        "SELECT PandaID FROM jobsActive4 ORDER BY age",
        "SELECT PandaID FROM jobsActive4 ORDER BY share",
    ]


def test_a_returned_call_and_a_returned_name_are_followed_to_their_text(tmp_path):
    roots = _tree(
        tmp_path,
        db=(
            "class DB:\n"
            "    def criteria(self, fast):\n"
            "        if fast:\n"
            "            return self.by_age()\n"
            "        return self.by_share()\n"
            "\n"
            "    def by_age(self):\n"
            "        empty = '', {}\n"
            "        if self.enabled:\n"
            "            return ' ORDER BY age', {}\n"
            "        return empty\n"
            "\n"
            "    def by_share(self):\n"
            "        return ' ORDER BY share', {}\n"
            "\n"
            "    def get(self, cur, fast):\n"
            "        sorting, _vm = self.criteria(fast)\n"
            "        sql = 'SELECT PandaID FROM jobsActive4'\n"
            "        if sorting:\n"
            "            sql += sorting\n"
            "        cur.execute(sql, {})\n"
        ),
    )
    calls = _calls(roots)
    _func, runs = _runs(calls, "db", "get")

    # The empty text ``by_age`` hands back when disabled is the run that skips
    # the ``if``, not a third alternative.
    assert sorted(run.sql for run in runs) == [
        "SELECT PandaID FROM jobsActive4 ORDER BY age",
        "SELECT PandaID FROM jobsActive4 ORDER BY share",
    ]


def test_a_statement_wrapped_around_itself_keeps_what_it_held(tmp_path):
    roots = _tree(
        tmp_path,
        db=(
            "def get(cur, limit):\n"
            "    sql = 'SELECT PandaID FROM jobsActive4 WHERE a=:a '\n"
            "    if limit:\n"
            "        sql = 'SELECT * FROM (' + sql\n"
            "        sql += ') WHERE ROWNUM <= :n'\n"
            "    cur.execute(sql, {':a': 1})\n"
        ),
    )
    calls = _calls(roots)
    _func, runs = _runs(calls, "db", "get")

    assert [run.sql for run in runs] == [
        "SELECT * FROM (SELECT PandaID FROM jobsActive4 WHERE a=:a ) WHERE ROWNUM <= :n"
    ]


def test_a_fragment_held_under_a_local_name_is_appended_as_its_text(tmp_path):
    roots = _tree(
        tmp_path,
        db=(
            "def get(cur, failed_only):\n"
            "    failed = \"AND jobStatus IN ('failed','closed') \"\n"
            "    sql = 'SELECT PandaID FROM jobsActive4 WHERE jediTaskID=:t '\n"
            "    if failed_only:\n"
            "        sql += failed\n"
            "    cur.execute(sql, {':t': 1})\n"
        ),
    )
    calls = _calls(roots)
    _func, runs = _runs(calls, "db", "get")

    (run,) = runs
    assert run.sql == "SELECT PandaID FROM jobsActive4 WHERE jediTaskID=:t AND jobStatus IN ('failed','closed') "
    assert run.unresolved == ()


def test_a_local_name_settles_only_where_every_binding_of_it_does(tmp_path):
    """``pandaID = 0`` before the loop that rebinds it from the rows is no value asked by."""
    from bamboo.codemap.panda import values

    roots = _tree(
        tmp_path,
        db=(
            "def get(cur, rows, labels):\n"
            "    pandaID = 0\n"
            "    for pandaID, other in rows:\n"
            "        break\n"
            "    status = 'activated'\n"
            "    if labels:\n"
            "        status = 'throttled'\n"
            "    return pandaID, status\n"
        ),
    )
    calls = _calls(roots)
    func = _function(calls, "db", "get")
    settle = values.handed(values.resolver(), calls)

    assert settle(ast.Name(id="pandaID"), func, "pkg/db.py") == []
    assert sorted(one.value for one in settle(ast.Name(id="status"), func, "pkg/db.py")) == [
        "activated",
        "throttled",
    ]


def test_an_imported_in_list_is_read_with_the_values_its_module_builds_it_from(tmp_path):
    from bamboo.codemap.panda import sql, values

    roots = _tree(
        tmp_path,
        spec=(
            "class DatasetSpec:\n"
            "    @classmethod\n"
            "    def input_types(cls):\n"
            "        return ['input', 'pseudo_input']\n"
            "\n"
            "    @classmethod\n"
            "    def process_types(cls):\n"
            "        return cls.input_types() + ['pp_input']\n"
            "\n"
            "TYPES_var_str, TYPES_var_map = get_sql_IN_bind_variables(DatasetSpec.process_types(), prefix=':type_')\n"
        ),
        db=(
            "from pkg.spec import TYPES_var_str, TYPES_var_map\n"
            "\n"
            "def get(cur):\n"
            "    varMap = {}\n"
            "    sql = 'SELECT datasetID FROM JEDI_Datasets '\n"
            "    sql += f'WHERE type IN ({TYPES_var_str}) '\n"
            "    varMap.update(TYPES_var_map)\n"
            "    cur.execute(sql, varMap)\n"
        ),
    )
    calls = _calls(roots)
    func, runs = _runs(calls, "db", "get")

    (run,) = runs
    assert run.sql == "SELECT datasetID FROM JEDI_Datasets WHERE type IN (:type___in) "
    with sql.across_calls(calls):
        (bind,) = sql.bound_values(func, "varMap", ":type___in", across=True)
    settle = values.handed(values.resolver(), calls)
    where = calls.types.tree("pkg/spec.py")
    assert sorted(one.value for one in settle(bind.value, where, "pkg/spec.py")) == [
        "input",
        "pp_input",
        "pseudo_input",
    ]


_FLAGGED = {
    "db": (
        "class Proxy:\n"
        "    def ids(self, cur, task_id, only_active):\n"
        "        tables = ['jobsDefined4', 'jobsActive4']\n"
        "        if not only_active:\n"
        "            tables += ['jobsArchived4']\n"
        "        for table in tables:\n"
        "            sql = f'SELECT PandaID FROM {table} WHERE jediTaskID=:t '\n"
        "            cur.execute(sql, {':t': task_id})\n"
    ),
    "tb": (
        "from pkg.db import Proxy\n"
        "\n"
        "class Buffer:\n"
        "    def __init__(self):\n"
        "        self.proxy = Proxy()\n"
        "\n"
        "    def ids(self, task_id, only_active):\n"
        "        return self.proxy.ids(None, task_id, only_active)\n"
    ),
    "cmd": (
        "from pkg.tb import Buffer\n"
        "\n"
        "def run(task_id):\n"
        "    Buffer().ids(task_id, True)\n"
        "    Buffer().ids(task_id, False)\n"
    ),
}


def test_a_flag_a_call_hands_is_followed_through_a_forwarding_method(tmp_path):
    roots = _tree(tmp_path, **_FLAGGED)
    calls = _calls(roots)
    func = _function(calls, "db", "ids")
    cls = calls.types.method_class("pkg/db.py", func)
    target = calls_mod.Target("pkg/db.py", func, cls[1])

    found = calls.handed_constants(target, {"only_active"})

    assert sorted(found) == [
        (("pkg/tb.py:8", "pkg/cmd.py:4"), {"only_active": True}),
        (("pkg/tb.py:8", "pkg/cmd.py:5"), {"only_active": False}),
    ]


def test_a_call_sites_constant_rules_out_what_its_flag_decides_against(tmp_path):
    from bamboo.codemap.panda import sql

    roots = _tree(tmp_path, **_FLAGGED)
    calls = _calls(roots)
    func = _function(calls, "db", "ids")

    with sql.across_calls(calls):
        every = sorted(run.sql for run in sql.executions(func))
        with sql.at_call(func, {"only_active": True}):
            active = sorted(run.sql for run in sql.executions(func))

    assert [text.split()[3] for text in every] == ["jobsActive4", "jobsArchived4", "jobsDefined4"]
    assert [text.split()[3] for text in active] == ["jobsActive4", "jobsDefined4"]


def test_order_and_limit_are_read_off_the_statement():
    from bamboo.codemap.panda import sqltext

    text = (
        "SELECT * FROM (SELECT PandaID FROM jobsActive4 WHERE jobStatus=:s "
        "ORDER BY currentPriority DESC, PandaID ASC) WHERE ROWNUM <= :njobs"
    )

    assert sqltext.ordering(text) == (["currentPriority DESC, PandaID ASC"], [(":njobs", False)])
    assert sqltext.ordering("SELECT a FROM t WHERE rownum<3 ORDER BY a FOR UPDATE") == (["a"], [("3", True)])
    nested = "SELECT * FROM (SELECT a FROM t ORDER BY (CASE {} ELSE {} END), b desc) WHERE ROWNUM <= :n"
    assert sqltext.ordering(nested)[0] == ["(CASE {} ELSE {} END), b desc"]


def test_a_statement_composed_under_a_name_is_read_where_it_runs(tmp_path):
    from bamboo.codemap.panda import sql

    roots = _tree(
        tmp_path,
        db=(
            "import re\n"
            "\n"
            "def stats(cur, flag):\n"
            "    sql_jt = 'SELECT computingSite FROM %s WHERE jobStatus=:s '\n"
            "    for table in ('ATLAS_PANDA.jobsActive4', 'ATLAS_PANDA.jobsDefined4'):\n"
            "        sql_exe = sql_jt % table\n"
            "        cur.execute(sql_exe, {':s': 'running'})\n"
            "\n"
            "def cached(cur):\n"
            "    base = 'SELECT a FROM t WHERE b=:b '\n"
            "    hinted = re.sub('SELECT ', 'SELECT /*+ RESULT_CACHE */ ', base)\n"
            "    cur.execute(hinted, {':b': 1})\n"
        ),
    )
    calls = _calls(roots)
    _func, runs = _runs(calls, "db", "stats")
    _func, cached = _runs(calls, "db", "cached")

    assert sorted(run.sql for run in runs) == [
        "SELECT computingSite FROM ATLAS_PANDA.jobsActive4 WHERE jobStatus=:s ",
        "SELECT computingSite FROM ATLAS_PANDA.jobsDefined4 WHERE jobStatus=:s ",
    ]
    assert [run.sql for run in cached] == ["SELECT /*+ RESULT_CACHE */ a FROM t WHERE b=:b "]


def test_a_format_field_the_call_cannot_fill_is_a_hole_like_any_other(tmp_path):
    from bamboo.codemap.panda import sqltext

    roots = _tree(
        tmp_path,
        db=(
            "def metadata(cur, pairs):\n"
            "    sqlC = 'SELECT j.PandaID FROM {0} j,{1} m '\n"
            "    sqlC += 'WHERE j.jediTaskID=:t AND m.PandaID=j.PandaID '\n"
            "    for a, m in pairs:\n"
            "        sql = sqlC.format(a, m)\n"
            "        cur.execute(sql, {':t': 1})\n"
            "\n"
            "def chunk(cur, n, m):\n"
            "    sql_read = 'SELECT * FROM (SELECT fileID FROM t WHERE f=:f ORDER BY {0}) WHERE rownum <= {1}'\n"
            "    cur.execute(sql_read.format('fileID', n - m), {':f': 1})\n"
        ),
    )
    calls = _calls(roots)
    _func, runs = _runs(calls, "db", "metadata")
    _func, chunk = _runs(calls, "db", "chunk")

    (run,) = runs
    assert run.sql == "SELECT j.PandaID FROM {} j,{} m WHERE j.jediTaskID=:t AND m.PandaID=j.PandaID "
    assert [table for table, _columns in sqltext.reads(run.sql)] == ["{}"]
    assert [run.sql for run in chunk] == [
        "SELECT * FROM (SELECT fileID FROM t WHERE f=:f ORDER BY fileID) WHERE rownum <= {}"
    ]


def test_a_function_is_said_to_be_called_through_the_door_that_forwards_to_it(tmp_path):
    roots = _tree(tmp_path, **_FLAGGED)
    calls = _calls(roots)
    func = _function(calls, "db", "ids")
    cls = calls.types.method_class("pkg/db.py", func)

    assert calls.called_from(calls_mod.Target("pkg/db.py", func, cls[1])) == [
        "pkg/cmd.py:4 -> pkg/tb.py:8",
        "pkg/cmd.py:5 -> pkg/tb.py:8",
    ]


def test_an_interface_method_calling_the_far_system_is_an_outbound_boundary(tmp_path, monkeypatch):
    from bamboo.codemap.panda.recognizers import outbound

    roots = _tree(
        tmp_path,
        ddm=(
            "from rucio.client import Client as RucioClient\n"
            "from pandacommon.log import Logger\n"
            "\n"
            "class API:\n"
            "    def _client(self) -> RucioClient:\n"
            "        return RucioClient()\n"
            "\n"
            "    def register(self, name):\n"
            "        client = self._client()\n"
            "        client.add_dataset(name=name)\n"
            "        client.add_files_to_dataset(name=name, files=[])\n"
            "\n"
            "    def scope(self, name, log: Logger):\n"
            "        log.debug(name)\n"
            "        return name.split(':')[0]\n"
            "\n"
            "api = API()\n"
        ),
        setup="from pkg.ddm import api\n\ndef run():\n    api.register('user.x')\n",
    )
    monkeypatch.setattr(outbound, "INTERFACE_CLASSES", {"pkg/ddm.py::API": "DDM"})
    monkeypatch.setattr(outbound, "OUTSIDE_PACKAGES", {"DDM": ("rucio",)})

    (node,) = outbound.extract([], _calls(roots), "m", "v")

    assert (node.interface, node.direction, node.kind) == ("pkg/ddm.py::API.register", "outbound", "transports_causation")
    assert node.verbs == ["attaches", "creates"]
    assert node.calls_out == ["rucio.client.Client.add_dataset", "rucio.client.Client.add_files_to_dataset"]
    assert node.call_sites == ["pkg/setup.py:4"]


def test_building_a_class_with_no_init_in_the_corpus_says_so(tmp_path):
    """``Interaction.JEDITemporaryError(...)``: nothing of the corpus runs, and the reason says why.

    Placed nowhere and given no reason, ninety-nine calls read in the census as
    a gap with no name -- the one class of stop nobody could act on.
    """
    roots = _tree(
        tmp_path,
        errors="class Retry(Exception):\n    pass\n\nclass Plain:\n    pass\n",
        m=(
            "from pkg import errors\nfrom pkg.errors import Plain\n\n"
            "def run():\n    errors.Retry('x')\n    Plain()\n"
        ),
    )
    calls = _calls(roots)

    assert _placed(calls, "m", "run", "errors.Retry").unplaced == calls_mod.UNPLACED_NO_INIT
    assert _placed(calls, "m", "run", "Plain").unplaced == calls_mod.UNPLACED_NO_INIT


def test_the_build_lists_every_call_to_its_own_functions_it_cannot_place(tmp_path):
    """A call to ``json.loads`` names nothing to enter, and a logger's lands in a file."""
    from bamboo.codemap.gitsource import blob_sha
    from bamboo.codemap.models import SourceModule
    from bamboo.codemap.panda import plugin

    texts = {
        "m": (
            "import json\n\n"
            "def helper():\n    return 1\n\n"
            "def debug():\n    return 2\n\n"
            "def run(spec, logger):\n"
            "    helper()\n"
            "    spec.helper()\n"
            "    logger.debug('x')\n"
            "    json.loads('1')\n"
        ),
    }
    roots = _tree(tmp_path, **texts)
    calls = _calls(roots)
    modules = [
        SourceModule(
            package=PACKAGE,
            rel_path=f"{PACKAGE}/{name}.py",
            tree=calls.types.tree(f"{PACKAGE}/{name}.py"),
            source=text,
            blob_sha=blob_sha(text),
        )
        for name, text in texts.items()
    ]

    considered, stops, opened = plugin._call_stops(calls, modules)
    assert opened == []

    assert considered == 2  # ``helper()`` and ``spec.helper()``
    assert [(stop.line, stop.callee, stop.reason) for stop in stops] == [
        (11, "spec.helper", calls_mod.UNPLACED_RECEIVER)
    ]
    assert stops[0].owner == f"{PACKAGE}/m.py::run"


def test_a_term_a_callee_or_a_local_appends_on_some_paths_is_not_required(tmp_path):
    """``getJobs`` appends what ``construct_where_clause`` returns, through a closure.

    Its ``minRamCount`` term is appended under ``if mem not in [0, "0"]:`` in
    the callee, and ``_read_unprocessed_inputs`` appends a local that holds
    its constituent filter only ``if skip_constituent_ids:``.  Read as one
    piece each, every run of the statement required them.
    """
    from bamboo.codemap.gitsource import blob_sha
    from bamboo.codemap.models import SourceModule
    from bamboo.codemap.panda import attribution, sql
    from bamboo.codemap.panda.recognizers import progress, sqlwrite

    db = (
        "class Jobs:\n"
        "    def construct_where_clause(self, mem):\n"
        "        clause = 'WHERE jobStatus=:st '\n"
        "        if mem:\n"
        "            clause += 'AND minRamCount<=:mem '\n"
        "        clause += 'AND maxCpuCount>:cpu '\n"
        "        return clause, {}\n"
        "\n"
        "    def getJobs(self, mem, skip):\n"
        "        def build_where_clause():\n"
        "            where, values = self.construct_where_clause(mem)\n"
        "            return where, values\n"
        "\n"
        "        sql_where_clause, var_map = build_where_clause()\n"
        "        extra = ''\n"
        "        if skip:\n"
        "            extra = 'AND jediTaskID NOT IN (:skip) '\n"
        "        sqlP = 'SELECT PandaID FROM ATLAS_PANDA.jobsActive4 '\n"
        "        sqlP += sql_where_clause\n"
        "        sqlP += extra\n"
        "        self.cur.execute(sqlP, {':st': 'activated', ':cpu': 1})\n"
    )
    roots = _tree(tmp_path, db=db)
    modules = [SourceModule(package=PACKAGE, rel_path=f"{PACKAGE}/db.py", tree=ast.parse(db), source=db, blob_sha=blob_sha(db))]
    attributor = attribution.SpecAttributor(progress.spec_attributes(modules), attribution.class_bases(modules))
    calls = sqlwrite.corpus_calls(modules, roots)

    with sql.across_calls(calls):
        side = sqlwrite.read_side(modules, attributor, calls)

    (statement,) = side.statements.values()
    assert statement.conditions == ["maxCpuCount>:cpu"]


def test_a_bind_reused_across_branches_is_read_per_branch(tmp_path):
    """``construct_where_clause``'s branches reuse ``:prodSourceLabel1..N`` with other values.

    Read by key, every branch asked for every label: ``managed`` -- asked
    under one branch only -- was in every build and read as required, and the
    ``IN`` list was placed at the fragment that puts the same binds inside an
    ``OR`` with ``processingType``.  ``test`` is asked by both branches and is
    required.
    """
    from bamboo.codemap.gitsource import blob_sha
    from bamboo.codemap.models import SourceModule
    from bamboo.codemap.panda import attribution, sql
    from bamboo.codemap.panda.recognizers import progress, sqlwrite

    db = (
        "class Jobs:\n"
        "    def getJobs(self, label):\n"
        "        var_map = {':site': 'X'}\n"
        "        sql = 'SELECT PandaID FROM ATLAS_PANDA.jobsActive4 WHERE computingSite=:site '\n"
        "        if label == 'managed':\n"
        "            sql += 'AND prodSourceLabel IN (:l1,:l2) '\n"
        "            var_map[':l1'] = 'managed'\n"
        "            var_map[':l2'] = 'test'\n"
        "        else:\n"
        "            sql += 'AND (processingType=:pt OR prodSourceLabel IN (:l1,:l2)) '\n"
        "            var_map[':pt'] = 'gangarobot'\n"
        "            var_map[':l1'] = 'prod_test'\n"
        "            var_map[':l2'] = 'test'\n"
        "        self.cur.execute(sql, var_map)\n"
    )
    roots = _tree(tmp_path, db=db)
    modules = [SourceModule(package=PACKAGE, rel_path=f"{PACKAGE}/db.py", tree=ast.parse(db), source=db, blob_sha=blob_sha(db))]
    attributor = attribution.SpecAttributor(progress.spec_attributes(modules), attribution.class_bases(modules))
    calls = sqlwrite.corpus_calls(modules, roots)

    with sql.across_calls(calls):
        side = sqlwrite.read_side(modules, attributor, calls)

    (statement,) = side.statements.values()
    added = {(one.field, one.term.split("=")[-1] if one.field == "selects" else one.term): one for one in statement.added_under}
    assert ("selects", "test") not in added
    for label, line in (("managed", 6), ("prod_test", 10), ("gangarobot", 10)):
        (place,) = added[("selects", label)].places
        assert place.fragment.endswith(f"db.py:{line}")
    # The IN list it is asked in, not the OR the other branch puts the same binds in.
    assert [place.term.split(" (")[0] for place in added[("selects", "managed")].places] == ["prodSourceLabel IN"]
    assert [place.term.split("=")[0] for place in added[("selects", "gangarobot")].places] == ["processingType"]
    (in_list,) = [one for key, one in added.items() if key[0] == "conditions" and key[1].startswith("prodSourceLabel")]
    assert [place.fragment.rsplit(":", 1)[1] for place in in_list.places] == ["6"]


def test_a_bind_filled_under_a_second_if_testing_the_same_thing_is_read_with_its_branch(tmp_path):
    """``getJobStatisticsForExtIF`` picks its statement under one ``if source_type == "analysis"`` and fills the binds under another.

    Two ``if`` statements, one decision: nothing between them gives
    ``source_type`` a new value.  ``managed`` is the production branch's
    label and was placed at the analysis statement, which names the same
    bind.
    """
    from bamboo.codemap.gitsource import blob_sha
    from bamboo.codemap.models import SourceModule
    from bamboo.codemap.panda import attribution, sql
    from bamboo.codemap.panda.recognizers import progress, sqlwrite

    db = (
        "class Jobs:\n"
        "    def stats(self, source_type):\n"
        "        if source_type == 'analysis':\n"
        "            sql = 'SELECT PandaID FROM ATLAS_PANDA.jobsActive4 WHERE prodSourceLabel IN (:l1,:l2) '\n"
        "        else:\n"
        "            sql = 'SELECT PandaID FROM ATLAS_PANDA.jobsActive4 WHERE prodSourceLabel IN (:l1) AND jobExecutionID=1 '\n"
        "        var_map = {}\n"
        "        if source_type == 'analysis':\n"
        "            var_map[':l1'] = 'user'\n"
        "            var_map[':l2'] = 'panda'\n"
        "        else:\n"
        "            var_map[':l1'] = 'managed'\n"
        "        self.cur.execute(sql, var_map)\n"
    )
    roots = _tree(tmp_path, db=db)
    modules = [SourceModule(package=PACKAGE, rel_path=f"{PACKAGE}/db.py", tree=ast.parse(db), source=db, blob_sha=blob_sha(db))]
    attributor = attribution.SpecAttributor(progress.spec_attributes(modules), attribution.class_bases(modules))
    calls = sqlwrite.corpus_calls(modules, roots)

    with sql.across_calls(calls):
        side = sqlwrite.read_side(modules, attributor, calls)

    (statement,) = side.statements.values()
    placed = {
        one.term.split("=")[-1]: sorted(place.fragment.rsplit(":", 1)[1] for place in one.places)
        for one in statement.added_under
        if one.field == "selects"
    }
    assert placed == {"user": ["4"], "panda": ["4"], "managed": ["6"]}


def test_a_pair_is_placed_in_its_own_term_not_in_the_fragment_that_appends_it(tmp_path):
    """``_read_unprocessed_inputs`` appends ``status=:status`` and an ``OR`` on attempts in one fragment.

    The pair's term is ``status=:status``.  Spelled as the fragment, the pair
    read as asked inside an ``OR`` -- another term's -- and so as allowed,
    not required.
    """
    from bamboo.codemap.gitsource import blob_sha
    from bamboo.codemap.models import SourceModule
    from bamboo.codemap.panda import attribution, sql
    from bamboo.codemap.panda.recognizers import progress, sqlwrite

    db = (
        "class Files:\n"
        "    def unprocessed(self, flag):\n"
        "        sql = 'SELECT PandaID FROM ATLAS_PANDA.jobsActive4 WHERE computingSite=:site '\n"
        "        if flag:\n"
        "            sql += 'AND jobStatus=:status AND (maxAttempt IS NULL OR attemptNr<maxAttempt) '\n"
        "        self.cur.execute(sql, {':site': 'X', ':status': 'activated'})\n"
    )
    roots = _tree(tmp_path, db=db)
    modules = [SourceModule(package=PACKAGE, rel_path=f"{PACKAGE}/db.py", tree=ast.parse(db), source=db, blob_sha=blob_sha(db))]
    attributor = attribution.SpecAttributor(progress.spec_attributes(modules), attribution.class_bases(modules))
    calls = sqlwrite.corpus_calls(modules, roots)

    with sql.across_calls(calls):
        side = sqlwrite.read_side(modules, attributor, calls)

    (statement,) = side.statements.values()
    (added,) = [one for one in statement.added_under if one.field == "selects"]
    assert added.term.endswith(".jobStatus=activated")
    assert [place.term.split("@")[0] for place in added.places] == ["jobStatus=:status"]


def test_a_place_says_whether_its_term_decides_the_pair(tmp_path):
    """``getJobs`` appends ``processingType OR prodSourceLabel`` under one ``if``; ``lockJobsForReassign`` a null-safe ``<>`` under another.

    The first term only allows its pairs -- a row passes on either column --
    and the second leaves the value out of every row of the builds it is in.
    The spelling cannot tell them apart: both are an ``OR``.
    """
    from bamboo.codemap.gitsource import blob_sha
    from bamboo.codemap.models import SourceModule
    from bamboo.codemap.panda import attribution, sql
    from bamboo.codemap.panda.recognizers import progress, sqlwrite

    db = (
        "class Jobs:\n"
        "    def pick(self, flag, only):\n"
        "        sql = 'SELECT PandaID FROM ATLAS_PANDA.jobsActive4 WHERE computingSite=:site '\n"
        "        if flag:\n"
        "            sql += 'AND (processingType=:pt OR prodSourceLabel=:psl) '\n"
        "        if only:\n"
        "            sql += 'AND (relocationFlag IS NULL OR relocationFlag<>:r) '\n"
        "        self.cur.execute(sql, {':site': 'X', ':pt': 'pmerge', ':psl': 'user', ':r': 'two'})\n"
    )
    roots = _tree(tmp_path, db=db)
    modules = [SourceModule(package=PACKAGE, rel_path=f"{PACKAGE}/db.py", tree=ast.parse(db), source=db, blob_sha=blob_sha(db))]
    attributor = attribution.SpecAttributor(progress.spec_attributes(modules), attribution.class_bases(modules))
    calls = sqlwrite.corpus_calls(modules, roots)

    with sql.across_calls(calls):
        side = sqlwrite.read_side(modules, attributor, calls)

    (statement,) = side.statements.values()
    decides = {
        (one.field, one.term.split(".")[-1]): [place.decides for place in one.places]
        for one in statement.added_under
        if one.field != "conditions"
    }
    assert decides == {
        ("selects", "processingType=pmerge"): [False],
        ("selects", "prodSourceLabel=user"): [False],
        ("excludes", "relocationFlag=two"): [True],
    }
    assert [pair for pair in statement.restricts if pair[0].endswith(".computingSite")] == [
        (next(s for s, _v in statement.selects if s.endswith(".computingSite")), "X")
    ]


def test_a_statement_says_its_conditions_as_trees_and_the_prints_of_its_text(tmp_path):
    """The time a bind holds, read through the write that reaches it, and the logging calls a record can be cut by.

    ``timeLimit`` is written twice and the bind reads the second; the locked
    time waits for the caller's ``time_limit``.  Of the three prints, two
    hold the executed text and map -- one whole, one less its comment and
    after the ``execute`` -- and the third prints another map.
    """
    from bamboo.codemap.gitsource import blob_sha
    from bamboo.codemap.models import SourceModule
    from bamboo.codemap.panda import attribution, sql
    from bamboo.codemap.panda.recognizers import progress, sqlwrite

    db = (
        "class Jobs:\n"
        "    def pick(self, time_limit, other):\n"
        "        comment = ' /* DBProxy.pick */'\n"
        "        varMap = {}\n"
        "        varMap[':st'] = 'activated'\n"
        "        timeLimit = naive_utcnow() - datetime.timedelta(hours=1)\n"
        "        timeLimit = naive_utcnow() - datetime.timedelta(days=7)\n"
        "        varMap[':t'] = timeLimit\n"
        "        varMap[':l'] = naive_utcnow() - datetime.timedelta(minutes=time_limit)\n"
        "        sql = 'SELECT PandaID FROM ATLAS_PANDA.jobsActive4 WHERE jobStatus=:st AND modificationTime<:t '\n"
        "        sql += 'AND (lockedTime IS NULL OR lockedTime<:l)'\n"
        "        self.log.debug(sql + comment + str(varMap))\n"
        "        self.cur.execute(sql + comment, varMap)\n"
        "        self.log.debug(f'running sql: {sql} {varMap}')\n"
        "        self.log.debug(sql + str(other))\n"
    )
    roots = _tree(tmp_path, db=db)
    modules = [SourceModule(package=PACKAGE, rel_path=f"{PACKAGE}/db.py", tree=ast.parse(db), source=db, blob_sha=blob_sha(db))]
    attributor = attribution.SpecAttributor(progress.spec_attributes(modules), attribution.class_bases(modules))
    calls = sqlwrite.corpus_calls(modules, roots)

    with sql.across_calls(calls):
        side = sqlwrite.read_side(modules, attributor, calls)

    (statement,) = side.statements.values()
    trees = {one.entry.split("@")[0]: one.tree for one in statement.condition_trees}
    assert set(trees) == {"modificationTime<:t", "lockedTime IS NULL OR lockedTime<:l"}
    (limit,) = trees["modificationTime<:t"].items[0].items
    (value,) = limit.values
    assert (limit.node, limit.op, value.base, value.offset_seconds) == ("compare", "<", "naive_utcnow", -604800.0)
    locked = trees["lockedTime IS NULL OR lockedTime<:l"]
    assert [branch.items[0].node for branch in locked.items] == ["null", "compare"]
    (waits,) = locked.items[1].items[0].values
    assert (waits.base, waits.offset_seconds, waits.needs) == ("naive_utcnow", None, ["pick.time_limit"])
    assert [(one.line, one.where, one.text, one.lead, one.gap) for one in statement.printed_by] == [
        (12, "before", "whole", "", ""),
        (14, "after", "without comment", "running sql: ", " "),
    ]


def test_an_order_a_loop_builds_says_its_column_and_an_order_a_format_hands_says_its_test(tmp_path):
    """``getJobs``' share ranking and ``_read_unprocessed_inputs``' ``ORDER BY {0}``.

    The ranking is blanked in the map -- its members are a loop's -- and the
    template the loop appends says it ranks by ``gshare``.  The other order
    is a value the ``execute`` formats in, written under an ``if``: that
    write is where the order is chosen.
    """
    from bamboo.codemap.gitsource import blob_sha
    from bamboo.codemap.models import SourceModule
    from bamboo.codemap.panda import attribution, sql
    from bamboo.codemap.panda.recognizers import progress, sqlwrite

    db = (
        "class Jobs:\n"
        "    def criteria(self, n):\n"
        "        var_map = {}\n"
        "        tmp_list = []\n"
        "        leaves = self.sorted_leaves()\n"
        "        i = 0\n"
        "        for leave in leaves:\n"
        "            var_map[f':leave{i}'] = leave\n"
        "            tmp_list.append('WHEN gshare=:leave{0} THEN {0}'.format(i))\n"
        "            i += 1\n"
        "        bindings = ' '.join(tmp_list)\n"
        "        ret_sql = f' ORDER BY (CASE {bindings} ELSE {len(leaves)} END), currentpriority desc'\n"
        "        return ret_sql, var_map\n"
        "\n"
        "    def pick(self, sorting):\n"
        "        sql = 'SELECT PandaID FROM ATLAS_PANDA.jobsActive4 WHERE jobStatus=:s '\n"
        "        varMap = {':s': 'activated'}\n"
        "        if sorting:\n"
        "            order, more = self.criteria(5)\n"
        "            sql += order\n"
        "            varMap.update(more)\n"
        "        self.cur.execute(sql, varMap)\n"
        "\n"
        "    def files(self, mode):\n"
        "        sql = 'SELECT fileID FROM ATLAS_PANDA.filesTable4 WHERE status=:s ORDER BY {0}'\n"
        "        if mode == 'lumi':\n"
        "            order_by_policy = 'lumiBlockNr,lfn'\n"
        "        else:\n"
        "            order_by_policy = 'fileID'\n"
        "        self.cur.execute(sql.format(order_by_policy), {':s': 'ready'})\n"
    )
    roots = _tree(tmp_path, db=db)
    modules = [SourceModule(package=PACKAGE, rel_path=f"{PACKAGE}/db.py", tree=ast.parse(db), source=db, blob_sha=blob_sha(db))]
    attributor = attribution.SpecAttributor(progress.spec_attributes(modules), attribution.class_bases(modules))
    calls = sqlwrite.corpus_calls(modules, roots)

    with sql.across_calls(calls):
        side = sqlwrite.read_side(modules, attributor, calls)

    picked = next(one for (owner, _line), one in side.statements.items() if owner.endswith("::pick"))
    (hole,) = picked.order_holes
    assert (hole.order, hole.item, hole.subjects) == ("(CASE {} ELSE {} END), currentpriority desc", 0, ["jobsActive4.gshare"])
    assert hole.templates == ["WHEN gshare=:leave{} THEN {}"]
    assert hole.computed_at == "pkg/db.py:7 for leave in leaves (leaves = self.sorted_leaves()@5)"
    files = next(one for (owner, _line), one in side.statements.items() if owner.endswith("::files"))
    chosen = {one.term: [test.test for place in one.places for test in place.tests] for one in files.added_under if one.field == "order_by"}
    assert chosen == {"fileID": ["not (mode == 'lumi')"], "lumiBlockNr,lfn": ["mode == 'lumi'"]}


# ---------------------------------------------------------------------------
# A class named through a module-level name, and an object declared as several
# ---------------------------------------------------------------------------


def _method(calls, file, cls_name, name):
    found = calls.types.class_named(f"{PACKAGE}/{file}.py", ast.Name(id=cls_name, ctx=ast.Load()))
    assert found is not None
    where, cls = found
    func = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
    return calls_mod.Target(where, func, cls)


def _where(placed):
    return [(target.cls.name, target.func.name) for target in placed.targets]


_BUFFER = '''\
class JediTaskBuffer:
    def getTask(self, x):
        return x
'''

_INTERFACE = '''\
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pkg.tb import JediTaskBuffer

    _Proxied: Any = JediTaskBuffer
else:
    _Proxied = object


class Interface(_Proxied):
    def __getattr__(self, name: str) -> Any:
        return getattr(self.remote, name)

    def own(self):
        return 1
'''


def test_a_base_bound_to_a_class_at_module_level_is_that_class_whatever_its_annotation(tmp_path):
    """``_Proxied: Any = JediTaskBuffer`` under ``TYPE_CHECKING``: the value, not ``Any``, and not ``object``."""
    roots = _tree(
        tmp_path,
        tb=_BUFFER,
        iface=_INTERFACE,
        m='''\
from pkg.iface import Interface

def f(tb: Interface):
    tb.getTask(1)
    tb.own()
    tb.nowhere()
''',
    )
    calls = _calls(roots)

    assert _where(_placed(calls, "m", "f", "tb.getTask")) == [("JediTaskBuffer", "getTask")]
    assert _placed(calls, "m", "f", "tb.getTask").unplaced == ""
    assert _where(_placed(calls, "m", "f", "tb.own")) == [("Interface", "own")]
    assert _placed(calls, "m", "f", "tb.nowhere").unplaced == calls_mod.UNPLACED_FORWARDED


def test_a_name_bound_to_two_corpus_classes_names_neither(tmp_path):
    roots = _tree(
        tmp_path,
        tb=_BUFFER,
        other="class Other:\n    def getTask(self, x):\n        return x\n",
        m='''\
import sys

from pkg.other import Other
from pkg.tb import JediTaskBuffer

if sys.version_info > (3,):
    Chosen = JediTaskBuffer
else:
    Chosen = Other


class C(Chosen):
    pass


def f(c: C):
    c.getTask(1)
''',
    )
    calls = _calls(roots)

    assert _placed(calls, "m", "f", "c.getTask").unplaced == calls_mod.UNPLACED_NO_METHOD


_CONNECTIONS = '''\
class Oracle:
    def __getattribute__(self, name):
        return object.__getattribute__(self, name)

    def cursor(self):
        return 1


class Postgres:
    def __getattribute__(self, name):
        return object.__getattribute__(self, name)

    def begin(self):
        pass

    def cursor(self):
        return 2


class Base:
    conn: "Oracle | Postgres"

    def run(self):
        self.conn.begin()
        self.conn.cursor()
        self.conn.close()


def one(conn: Oracle):
    conn.cursor()
    conn.close()
'''


def test_an_object_declared_as_several_classes_lands_in_each_that_has_the_method(tmp_path):
    """``self.conn: WrappedOracleConn | WrappedPostgresConn``: a class that may only forward the name opens the placement."""
    roots = _tree(tmp_path, m=_CONNECTIONS)
    calls = _calls(roots)

    begin = _placed(calls, "m", "run", "self.conn.begin")
    cursor = _placed(calls, "m", "run", "self.conn.cursor")
    close = _placed(calls, "m", "run", "self.conn.close")

    assert (_where(begin), begin.unplaced) == ([("Postgres", "begin")], calls_mod.UNPLACED_OPEN)
    assert (_where(cursor), cursor.unplaced) == ([("Oracle", "cursor"), ("Postgres", "cursor")], "")
    assert (_where(close), close.unplaced) == ([], calls_mod.UNPLACED_GETATTRIBUTE)


def test_a_name_getattribute_may_supply_is_said_so_and_a_method_the_class_defines_is_still_placed(tmp_path):
    roots = _tree(tmp_path, m=_CONNECTIONS)
    calls = _calls(roots)

    assert _where(_placed(calls, "m", "one", "conn.cursor")) == [("Oracle", "cursor")]
    assert _placed(calls, "m", "one", "conn.cursor").unplaced == ""
    assert _placed(calls, "m", "one", "conn.close").unplaced == calls_mod.UNPLACED_GETATTRIBUTE


_FEEDER = '''\
from typing import Any


class Interface:
    def __getattr__(self, name: str) -> Any:
        return getattr(self.remote, name)


class Buffer:
    def getTask(self, x):
        return x + 1


class Other:
    def getTask(self, x):
        return x


class Feeder:
    def __init__(self, taskbufferIF: "Interface | Buffer"):
        self.taskBufferIF = taskbufferIF

    def run(self, a):
        self.taskBufferIF.getTask(a)
        self.taskBufferIF.elsewhere(a)
'''


def test_a_union_is_read_through_the_field_its_parameter_is_assigned_to(tmp_path):
    """The declaration is two classes wherever a single one would be read; a single-class answer is still none."""
    roots = _tree(tmp_path, m=_FEEDER)
    calls = _calls(roots)

    got = _placed(calls, "m", "run", "self.taskBufferIF.getTask")
    forwarded = _placed(calls, "m", "run", "self.taskBufferIF.elsewhere")
    run = _function(calls, "m", "run")
    receiver = _call_in(run, "self.taskBufferIF.getTask").func.value

    assert (_where(got), got.unplaced) == ([("Buffer", "getTask")], calls_mod.UNPLACED_OPEN)
    assert (_where(forwarded), forwarded.unplaced) == ([], calls_mod.UNPLACED_FORWARDED)
    assert sorted(cls.name for _file, cls in calls.types.declared("pkg/m.py", run, receiver)) == ["Buffer", "Interface"]
    assert calls.types.of_expression("pkg/m.py", run, receiver) is None


def test_a_call_placed_open_at_another_definition_may_miss_this_one(tmp_path):
    """``Other.getTask`` is not where ``self.taskBufferIF.getTask`` lands -- unless it is where the interface forwards it."""
    roots = _tree(tmp_path, m=_FEEDER)
    calls = _calls(roots)

    other = _method(calls, "m", "Other", "getTask")
    buffer = _method(calls, "m", "Buffer", "getTask")

    assert calls.missed(other) == [
        "getTask: 1 call(s) by its name placed elsewhere -- "
        f"{calls_mod.UNPLACED_OPEN} (self.taskBufferIF 1)"
    ]
    assert [ast.unparse(site.call.func) for site in calls.callers(buffer)] == ["self.taskBufferIF.getTask"]
    assert calls.missed(buffer) == []


def test_a_value_built_as_a_class_that_only_forwards_opens_the_placement(tmp_path):
    """Read from the values as from a union: ``Interface()`` adds no definition, and says the call may go elsewhere."""
    roots = _tree(
        tmp_path,
        m=_FEEDER + '''

def f(flag, a):
    if flag:
        tb = Interface()
    else:
        tb = Buffer()
    tb.getTask(a)
    tb.elsewhere(a)
''',
    )
    calls = _calls(roots)

    got = _placed(calls, "m", "f", "tb.getTask")

    assert (_where(got), got.unplaced) == ([("Buffer", "getTask")], calls_mod.UNPLACED_OPEN)
    assert _placed(calls, "m", "f", "tb.elsewhere").unplaced == calls_mod.UNPLACED_FORWARDED


def test_what_a_call_placed_open_hands_back_is_not_known_where_a_statement_reads_it(tmp_path):
    """A partial answer would read as all of it: the readers of :func:`sql.returned` take empty for "not known"."""
    from bamboo.codemap.panda import sql

    roots = _tree(
        tmp_path,
        m='''\
from typing import Any


class Interface:
    def __getattr__(self, name: str) -> Any:
        return getattr(self.remote, name)


class Buffer:
    def where(self):
        return " AND x=1", {}


class Holder:
    def __init__(self, source: "Interface | Buffer", buffer: Buffer):
        self.source = source
        self.buffer = buffer

    def run(self):
        opened, _ = self.source.where()
        closed, _ = self.buffer.where()
''',
    )
    calls = _calls(roots)
    run = _function(calls, "m", "run")

    with sql.across_calls(calls):
        opened = sql.returned(run, "opened")
        closed = sql.returned(run, "closed")

    assert _placed(calls, "m", "run", "self.source.where").unplaced == calls_mod.UNPLACED_OPEN
    assert opened == []
    assert [(callee.name, ast.unparse(value)) for callee, value in closed] == [("where", "' AND x=1'")]


# -- X5: a call placed open, said where callers are listed ----------------------


def test_a_listing_of_callers_says_which_call_was_placed_open(tmp_path):
    """``called_from`` lists ``self.taskBufferIF.getTask(...)`` at ``Buffer.getTask`` and says it may land elsewhere."""
    roots = _tree(tmp_path, m=_FEEDER)
    calls = _calls(roots)
    buffer = _method(calls, "m", "Buffer", "getTask")
    (site,) = calls.callers(buffer)

    assert calls.opened(site) == calls_mod.UNPLACED_OPEN
    assert calls.called_from(buffer) == [f"pkg/m.py:{site.call.lineno} (placed open -- {calls_mod.UNPLACED_OPEN})"]


def test_the_build_lists_the_calls_it_places_open_apart_from_those_it_cannot_place(tmp_path):
    """A call placed open is entered, so it is no stop; it is listed on its own with its reason."""
    from bamboo.codemap.gitsource import blob_sha
    from bamboo.codemap.models import SourceModule
    from bamboo.codemap.panda import plugin

    roots = _tree(tmp_path, m=_FEEDER)
    calls = _calls(roots)
    modules = [
        SourceModule(
            package=PACKAGE, rel_path="pkg/m.py", tree=calls.types.tree("pkg/m.py"), source=_FEEDER, blob_sha=blob_sha(_FEEDER)
        )
    ]

    considered, stops, opened = plugin._call_stops(calls, modules)

    assert [(one.callee, one.reason) for one in opened] == [("self.taskBufferIF.getTask", calls_mod.UNPLACED_OPEN)]
    assert all(one.callee != "self.taskBufferIF.getTask" for one in stops)


_OPEN_ROUTE_DB = (
    "from typing import Any\n"
    "\n"
    "class Interface:\n"
    "    def __getattr__(self, name: str) -> Any:\n"
    "        return getattr(self.remote, name)\n"
    "\n"
    "class Buffer:\n"
    "    def scan(self, cur, label):\n"
    "        sql = 'SELECT PandaID FROM ATLAS_PANDA.jobsActive4 WHERE prodSourceLabel=:l'\n"
    "        varMap = {':l': label}\n"
    "        cur.execute(sql, varMap)\n"
)


def _read_side(tmp_path, **texts):
    from bamboo.codemap.gitsource import blob_sha
    from bamboo.codemap.models import SourceModule
    from bamboo.codemap.panda import attribution, sql
    from bamboo.codemap.panda.recognizers import progress, sqlwrite

    roots = _tree(tmp_path, **texts)
    modules = [
        SourceModule(package=PACKAGE, rel_path=f"{PACKAGE}/{name}.py", tree=ast.parse(text), source=text, blob_sha=blob_sha(text))
        for name, text in texts.items()
    ]
    attributor = attribution.SpecAttributor(progress.spec_attributes(modules), attribution.class_bases(modules))
    calls = sqlwrite.corpus_calls(modules, roots)
    with sql.across_calls(calls):
        return sqlwrite.read_side(modules, attributor, calls)


def test_a_route_through_a_call_placed_open_says_so(tmp_path):
    """The route is read like any other, and its record names the call that may land elsewhere and why."""
    main = (
        "from pkg.db import Buffer, Interface\n"
        "\n"
        "class Feeder:\n"
        "    def __init__(self, tb: 'Interface | Buffer'):\n"
        "        self.tb = tb\n"
        "\n"
        "    def run(self, cur):\n"
        "        self.tb.scan(cur, 'managed')\n"
    )
    side = _read_side(tmp_path, db=_OPEN_ROUTE_DB, main=main)

    (scanned,) = [one for (owner, _line), one in side.statements.items() if owner.endswith("::scan")]
    (route,) = scanned.call_readings
    assert route.chain == ["pkg/main.py:8"]
    assert route.open == [f"pkg/main.py:8 -- {calls_mod.UNPLACED_OPEN}"]
    # In the list, so the list is not short of it.
    assert scanned.callers_complete is True


def test_a_statement_says_no_value_where_one_of_its_calls_hands_one_not_known(tmp_path):
    """``getCache_JEDI``: one call hands ``main_key="AtlasSites"``, another an attribute.

    The statement's own fields, every run any call makes, name no value -- the
    union of the calls that settle is not every value.  Each route still reads
    what its own call hands, and the route whose call hands the attribute reads
    none; a route that read as the statement before now carries its own field.
    """
    db = (
        "def get_cache(cur, main_key):\n"
        "    sql = 'SELECT data FROM ATLAS_PANDA.JEDI_Cache WHERE main_key=:k'\n"
        "    varMap = {':k': main_key}\n"
        "    cur.execute(sql, varMap)\n"
    )
    main = (
        "from pkg.db import get_cache\n"
        "\n"
        "class Watchdog:\n"
        "    def __init__(self):\n"
        "        self.key = 'Filler'\n"
        "\n"
        "    def sites(self, cur):\n"
        "        get_cache(cur, 'AtlasSites')\n"
        "\n"
        "    def filler(self, cur):\n"
        "        get_cache(cur, self.key)\n"
    )
    side = _read_side(tmp_path, db=db, main=main)

    (cached,) = [one for (owner, _line), one in side.statements.items() if owner.endswith("::get_cache")]
    assert cached.selects == set()
    routes = {tuple(one.chain): one for one in cached.call_readings}
    assert routes[("pkg/main.py:8",)].as_statement is False
    assert routes[("pkg/main.py:8",)].selects == ["JEDI_Cache.main_key=AtlasSites"]
    assert routes[("pkg/main.py:11",)].selects == []


def test_a_value_settles_from_callers_only_where_every_call_is_known(tmp_path):
    """Handed marks, not silence: a call by the name placed nowhere, a value passed through ``*``, a value not settled."""
    from bamboo.codemap.panda import values

    roots = _tree(
        tmp_path,
        db="def lock(statList):\n    return statList\n",
        main=(
            "from pkg.db import lock\n"
            "\n"
            "def run(more, rows):\n"
            "    lock(['activated'])\n"
            "    lock(*more)\n"
            "    lock(rows[0])\n"
            "    unknown.lock(['held'])\n"
        ),
    )
    calls = _calls(roots)
    func = _function(calls, "db", "lock")
    settle = values.handed(values.resolver(), calls)

    found = settle(ast.Name(id="statList"), func, "pkg/db.py")

    assert [(one.value, one.via) for one in found if not one.unknown] == [("activated", ("pkg/main.py:4",))]
    assert sorted(one.via for one in found if one.unknown) == [(), ("pkg/main.py:5",), ("pkg/main.py:6",)]
    assert values.known(found) == []


def test_a_route_read_as_the_statement_is_checked_against_the_statements_fields():
    """The build stops where a route marked as the statement read otherwise -- every field the mark stands for."""
    import pytest

    from bamboo.codemap.models import CallReading
    from bamboo.codemap.panda.recognizers import sqlwrite

    at = sqlwrite.ReadStatement(
        owner="pkg/db.py::scan", package=PACKAGE, file="pkg/db.py", blob_sha="", line=4, window=None,
        selects={("jobsActive4.jobStatus", "running")}, excludes=set(), consults=set(), keys=set(),
        order_by=set(), limits=set(), tables={"jobsActive4"}, conditions=[], unresolved=(),
    )
    same = CallReading(chain=["pkg/main.py:4"], tables=["jobsActive4"], selects=["jobsActive4.jobStatus=running"])
    other = same.model_copy(update={"selects": []})

    sqlwrite._check_as_statement([(("pkg/db.py::scan", 4), ("pkg/main.py:4",), same)], {("pkg/db.py::scan", 4): at})
    with pytest.raises(RuntimeError, match="1 route.*selects"):
        sqlwrite._check_as_statement([(("pkg/db.py::scan", 4), ("pkg/main.py:4",), other)], {("pkg/db.py::scan", 4): at})


def test_every_caller_is_read_however_many_there_are(tmp_path):
    """No call count past which a function's routes or constants are not read; ``missed`` says only what is missing."""
    callers = "".join(f"    lock({i})\n" for i in range(20))
    roots = _tree(
        tmp_path,
        db="def lock(minutes):\n    return minutes\n",
        main=f"from pkg.db import lock\n\ndef run():\n{callers}",
    )
    calls = _calls(roots)
    target = calls_mod.Target("pkg/db.py", _function(calls, "db", "lock"), None)

    routes, stops = calls.routes(target)

    assert len(routes) == 20 and stops == []
    assert len(calls.handed_constants(target, {"minutes"})) == 20
    assert calls.missed(target) == []


def test_a_name_shared_with_every_container_names_five_receivers_and_counts_the_rest(tmp_path):
    """``update``'s 168 calls on ``varMap``, ``var_map``, ... would make one clause of seven hundred characters."""
    body = "".join(f"    d{i}.update(x)\n" for i in range(8))
    roots = _tree(
        tmp_path,
        m=f"class Mine:\n    def update(self, x):\n        return x\n\ndef run(x, {', '.join(f'd{i}' for i in range(8))}):\n{body}",
    )
    calls = _calls(roots)

    (why,) = calls.missed(_method(calls, "m", "Mine", "update"))

    assert why.endswith("(d0 1, d1 1, d2 1, d3 1, d4 1, … 3 more)")


# -- X5: a statement read at a call ---------------------------------------------


def test_a_fragment_under_a_test_no_pass_of_its_loop_decides_is_added_under_it_not_required(tmp_path):
    """``getPandaIDsWithTask_JEDI``: ``modificationTime>(CURRENT_DATE-30)`` is appended under a test of the table name.

    No pass can say whether ``{}.jobsArchived`` starts with the archive schema,
    so every pass keeps the fragment -- the statement's text is unchanged --
    but it is kept as appended under that ``if``, not as what every branch requires.
    """
    from bamboo.codemap.panda import sql

    db = (
        "from pandaserver.config import panda_config\n"
        "\n"
        "\n"
        "def get_ids(cur, jediTaskID, onlyActive):\n"
        "    tables = [\n"
        "        f\"{panda_config.schemaPANDA}.jobsDefined4\",\n"
        "        f\"{panda_config.schemaPANDA}.jobsActive4\",\n"
        "    ]\n"
        "    if not onlyActive:\n"
        "        tables += [f\"{panda_config.schemaPANDA}.jobsArchived4\", f\"{panda_config.schemaPANDAARCH}.jobsArchived\"]\n"
        "    sqlP = \"\"\n"
        "    for tableName in tables:\n"
        "        if sqlP != \"\":\n"
        "            sqlP += \"UNION ALL \"\n"
        "        sqlP += f\"SELECT PandaID FROM {tableName} WHERE jediTaskID=:jediTaskID \"\n"
        "        if tableName.startswith(panda_config.schemaPANDAARCH):\n"
        "            sqlP += \"AND modificationTime>(CURRENT_DATE-30) \"\n"
        "    varMap = {}\n"
        "    varMap[\":jediTaskID\"] = jediTaskID\n"
        "    cur.execute(sqlP, varMap)\n"
    )
    main = "from pkg.db import get_ids\n\ndef run(cur):\n    get_ids(cur, 1, True)\n    get_ids(cur, 2, False)\n"
    side = _read_side(tmp_path, db=db, main=main)

    (got,) = side.statements.values()
    (run,) = sql.executions(_function(_calls(_tree(tmp_path, db=db, main=main)), "db", "get_ids"))
    assert run.sql.count("AND modificationTime>(CURRENT_DATE-30)") == 4
    assert got.conditions == []
    (added,) = [one for one in got.added_under if one.field == "conditions"]
    assert added.term == "modificationTime>(CURRENT_DATE-30)"
    assert [test.test for test in added.places[0].tests] == [
        "a loop at 12 runs", "tableName.startswith(panda_config.schemaPANDAARCH)"
    ]
    assert all(route.condition_trees == [] for route in got.call_readings)


def test_a_route_constant_is_handed_through_a_forwarder_only_as_it_came(tmp_path):
    """Renamed and passed by position, it is the caller's; bound again on the route, it is no one's; a literal or a default is the forwarder's."""
    from bamboo.codemap.panda.recognizers import sqlwrite

    roots = _tree(
        tmp_path,
        db="def lock(flag, other=False, more=None):\n    if flag or other or more:\n        return 1\n",
        tb=(
            "from pkg.db import lock\n"
            "\n"
            "def renamed(f):\n    return lock(f)\n"
            "\n"
            "def rebound(flag, sim):\n    if sim:\n        flag = True\n    return lock(flag, sim)\n"
            "\n"
            "def maybe(flag, sim):\n    if sim:\n        flag = True\n    return lock(flag, sim)\n"
            "\n"
            "def fixed(flag):\n    return lock(True, flag)\n"
            "\n"
            "def spread(**kwargs):\n    return lock(False, **kwargs)\n"
        ),
        main=(
            "from pkg.tb import fixed, maybe, rebound, renamed, spread\n"
            "\n"
            "def run():\n    renamed(False)\n    rebound(False, [1])\n    fixed(False)\n    spread(other=True)\n    maybe(False, None)\n"
        ),
    )
    calls = _calls(roots)
    target = calls_mod.Target("pkg/db.py", _function(calls, "db", "lock"), None)
    routes, _stops = calls.routes(target)

    def constants(owner):
        (route,) = [one for one in routes if one.sites[0].func.name == owner]
        found = {name: sqlwrite._route_constant(name, target, route.sites, calls) for name in ("flag", "other", "more")}
        return {name: value for name, value in found.items() if value is not sqlwrite._UNSET}

    assert constants("renamed") == {"flag": False, "other": False, "more": None}
    # ``sim`` is handed on as it came, so the route goes on out to ``run``; ``flag``
    # is bound again first, and what ``run`` hands it is not what ``lock`` gets.
    assert constants("rebound") == {"more": None}
    # The same binding under a test the route's own constant rules out is not on the route.
    assert constants("maybe") == {"flag": False, "other": None, "more": None}
    # ``other`` is handed on as it came: the outer call's.
    assert constants("fixed") == {"flag": True, "other": False, "more": None}
    assert constants("spread") == {"flag": False}


# -- X6: a builtin type's method is not a corpus function's -----------------------

_LISTS = '''\
import collections
from collections import defaultdict


class ListWithLock:
    def append(self, x):
        return x

    def add(self, x):
        return x


def build(x, given):
    items = []
    items.append(x)
    seen = set()
    seen.add(x)
    named = [one for one in given]
    named.append(x)
    grouped = defaultdict(list)
    grouped.setdefault(x, [])
    counted = collections.Counter()
    counted.update(x)
    given.append(x)


def either(flag, x):
    held = [] if flag else None
    if flag:
        held = ListWithLock()
    else:
        held = []
    held.append(x)


def shadowed(x):
    set = ListWithLock
    mine = set()
    mine.add(x)
'''


def test_a_receiver_built_as_a_builtin_lands_outside_the_corpus(tmp_path):
    """``items = []``, ``set()``, a comprehension, a library container: a builtin's method, not ``ListWithLock``'s."""
    calls = _calls(_tree(tmp_path, m=_LISTS))

    for text in ("items.append", "seen.add", "named.append", "grouped.setdefault", "counted.update"):
        placed = _placed(calls, "m", "build", text)
        assert (placed.targets, placed.unplaced) == ((), calls_mod.UNPLACED_OUTSIDE), text
    given = _placed(calls, "m", "build", "given.append")
    assert (given.targets, given.unplaced) == ((), calls_mod.UNPLACED_RECEIVER)


def test_a_builtin_beside_a_corpus_class_leaves_the_call_open_as_before(tmp_path):
    """``held`` is a list on one branch and a ``ListWithLock`` on the other: either may run the call."""
    calls = _calls(_tree(tmp_path, m=_LISTS))

    placed = _placed(calls, "m", "either", "held.append")

    assert (_where(placed), placed.unplaced) == ([("ListWithLock", "append")], calls_mod.UNPLACED_OPEN)


def test_a_builtin_name_the_function_rebinds_is_not_the_builtin(tmp_path):
    """``set = ListWithLock``: ``set()`` builds what the name holds."""
    calls = _calls(_tree(tmp_path, m=_LISTS))

    placed = _placed(calls, "m", "shadowed", "mine.add")

    assert _where(placed) == [("ListWithLock", "add")]


def test_a_builtin_name_the_module_defines_is_not_the_builtin(tmp_path):
    calls = _calls(
        _tree(
            tmp_path,
            m="class Box:\n    def add(self, x):\n        return x\n\n"
            "def sorted(x):\n    return Box()\n\n"
            "def run(x):\n    held = sorted(x)\n    held.add(x)\n",
        )
    )

    placed = _placed(calls, "m", "run", "held.add")

    assert placed.unplaced != calls_mod.UNPLACED_OUTSIDE


def test_what_a_builtins_method_returns_is_not_read_so_the_call_stays_open(tmp_path):
    """``self.task_setupper_map = {}`` ... ``.get(key)``: the dict's values are not read here, so a class beside them is not all."""
    calls = _calls(
        _tree(
            tmp_path,
            m="class Setupper:\n    def initializeMods(self):\n        return 1\n\n"
            "class Processor:\n    def __init__(self):\n        self.task_setupper_map = {}\n\n"
            "    def run(self, key):\n        taskSetupper = self.task_setupper_map.get(key)\n"
            "        if taskSetupper is None:\n            taskSetupper = Setupper()\n"
            "        taskSetupper.initializeMods()\n",
        )
    )

    placed = _placed(calls, "m", "run", "taskSetupper.initializeMods")

    assert (_where(placed), placed.unplaced) == ([("Setupper", "initializeMods")], calls_mod.UNPLACED_OPEN)


def test_a_call_landing_outside_the_corpus_is_not_counted_as_a_call_the_list_misses(tmp_path):
    """``ListWithLock.append`` misses ``given.append`` -- its receiver's class is not known -- and not ``items.append``."""
    calls = _calls(_tree(tmp_path, m=_LISTS))

    (why,) = calls.missed(_method(calls, "m", "ListWithLock", "append"))

    assert why.startswith("append: 1 call(s) by its name not placed -- the receiver's class is not declared (given 1)")


def test_a_function_whose_calls_by_name_all_land_outside_says_so_and_not_that_nothing_names_it(tmp_path):
    """``WorkQueueMapper.dump``: ``json.dump(...)`` names it and lands in ``json``; ``in`` asks ``__contains__`` unread."""
    calls = _calls(
        _tree(
            tmp_path,
            m="import json\n\nclass Mapper:\n    def dump(self):\n        return 1\n\n    def unused(self):\n        return 2\n\n"
            "def save(x, f):\n    json.dump(x, f)\n    json.dump(x, f)\n",
        )
    )

    assert calls.missed(_method(calls, "m", "Mapper", "dump")) == [
        "dump: no call this reading places lands at it -- 2 call(s) by its name land outside the corpus (json 2)"
    ]
    assert calls.missed(_method(calls, "m", "Mapper", "unused")) == [
        "unused: no call in the corpus names it -- reached from outside"
    ]


def test_a_list_with_one_element_unsettled_settles_to_no_value_not_to_the_rest(tmp_path):
    """``[a, b]`` with ``b`` unknown is not ``{a}`` -- as a name with one unsettled write is not its other writes."""
    from bamboo.codemap.panda import values

    roots = _tree(
        tmp_path,
        db=(
            "def get(other, more):\n"
            "    known = ['activated', 'throttled']\n"
            "    partly = ['activated', other]\n"
            "    spread = ['activated', *more]\n"
            "    return known, partly, spread\n"
        ),
    )
    calls = _calls(roots)
    func = _function(calls, "db", "get")
    settle = values.handed(values.resolver(), calls)
    lists = {
        node.targets[0].id: node.value
        for node in ast.walk(func)
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.List)
    }

    assert sorted(one.value for one in settle(lists["known"], func, "pkg/db.py")) == ["activated", "throttled"]
    assert settle(lists["partly"], func, "pkg/db.py") == []
    assert settle(lists["spread"], func, "pkg/db.py") == []
