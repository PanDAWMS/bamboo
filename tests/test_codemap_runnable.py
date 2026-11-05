"""Which definitions a reader may run with a run's values -- ``codemap.panda.runnable`` -- on helpers shaped as
PanDA's: ``DataServiceUtils``' regular expressions, ``EventServiceUtils``' tokens read off a spec's class, a spec's
methods reading its fields, ``RefinerUtils.decodeJSON`` and its hook."""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import pytest

from bamboo.codemap import calls as calls_mod
from bamboo.codemap import reach
from bamboo.codemap.gitsource import blob_sha
from bamboo.codemap.models import (
    Anchor,
    Branch,
    JunctionNode,
    Runnable,
    RunStop,
    SourceModule,
)
from bamboo.codemap.panda import runnable, sql
from bamboo.codemap.panda.plugin import _DatabaseLeaves
from bamboo.codemap.panda.recognizers import sqlwrite

SPEC = "pandaserver/taskbuffer/TaskSpec.py"
UTILS = "pandaserver/taskbuffer/TaskUtils.py"
CONFIG = "pandaserver/config/panda_config.py"
DRIVER = "pandaserver/taskbuffer/Driver.py"

SOURCES = {
    CONFIG: """\
        limit = 3
        """,
    SPEC: """\
        class TaskSpec:
            _tags = {"cloning": "sc", "merge": "mo"}
            _inv = {1: "no"}
            _inv = {v: k for k, v in {"scout": 1}.items()}
            enum_scout = _inv["scout"]

            def __init__(self):
                self.splitRule = None
                self.status = None

            def useScout(self):
                return self.splitRule is not None and str(self.enum_scout) in self.splitRule.split(",")

            def setStatus(self, status):
                self.status = status
                return True
        """,
    UTILS: """\
        import datetime
        import json
        import re

        from pandacommon.pandalogger.PandaLogger import PandaLogger
        from pandaserver.config import panda_config
        from pandaserver.taskbuffer.TaskSpec import TaskSpec

        logger = PandaLogger().getLogger("TaskUtils")
        cloningToken = TaskSpec._tags["cloning"]
        _seen = {}

        def is_sub(name: str) -> bool:
            return re.search("_sub[0-9]+$", name) is not None

        def is_cloning(handling):
            if handling is None:
                return False
            logger.debug("checking")
            return cloningToken in handling.split(",")

        def is_late(task):
            return task.modificationTime < datetime.datetime.utcnow()

        def over_limit(n):
            return n > panda_config.limit

        def remember(name):
            _seen[name] = True
            return name in _seen

        def mark(params):
            params["seen"] = True
            return True

        def cloning_late(task):
            return is_cloning(task.specialHandling) and is_late(task)

        def unicodeConvert(input):
            if isinstance(input, dict):
                return {unicodeConvert(k): unicodeConvert(v) for k, v in input.items()}
            return input

        def decodeJSON(text):
            return json.loads(text, object_hook=unicodeConvert)

        def ping(n):
            return n <= 0 or pong(n - 1)

        def pong(n):
            return datetime.datetime.now() is not None and ping(n - 1)
        """,
    DRIVER: """\
        from pandaserver.taskbuffer import TaskUtils
        from pandaserver.taskbuffer.TaskSpec import TaskSpec

        class Driver:
            def run(self, task: TaskSpec, name, params, n):
                if TaskUtils.is_sub(name) or TaskUtils.is_cloning(task.specialHandling):  # A1
                    self.a = 1
                if task.useScout() and TaskUtils.is_late(task) and TaskUtils.over_limit(n):  # A2
                    self.b = 1
                if TaskUtils.remember(name) and TaskUtils.mark(params) and task.setStatus("x"):  # A3
                    self.c = 1
                if TaskUtils.cloning_late(task):  # A4
                    self.d = 1
                if TaskUtils.pong(n) and TaskUtils.ping(n):  # A5
                    self.e = 1
        """,
}


def _line(rel: str, marker: str) -> int:
    (found,) = [
        number
        for number, held in enumerate(textwrap.dedent(SOURCES[rel]).splitlines(), 1)
        if held.rstrip().endswith(f"# {marker}")
    ]
    return found


@pytest.fixture(scope="module")
def found(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Runnable]:
    """The records of the driver's arms, each arm's test its path condition, by the definition's name."""
    tmp_path: Path = tmp_path_factory.mktemp("runnable")
    modules = []
    for rel, text in SOURCES.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text))
        parsed = calls_mod.parsed(path)
        assert parsed is not None
        modules.append(
            SourceModule(
                package="pandaserver",
                rel_path=rel,
                tree=parsed[1],
                source=textwrap.dedent(text),
                blob_sha=blob_sha(text),
            )
        )
    calls = sqlwrite.corpus_calls(modules, {"pandaserver": tmp_path / "pandaserver"})
    assert calls is not None
    lines = textwrap.dedent(SOURCES[DRIVER]).splitlines()
    junctions = [
        JunctionNode(
            name=f"m:{marker}",
            map_id="m",
            derived_from="v",
            subject="TaskSpec.status",
            owner=f"{DRIVER}::run",
            anchor=Anchor(
                package="pandaserver", file=DRIVER, line_start=_line(DRIVER, marker) + 1
            ),
            branches=[
                Branch(
                    outcome=marker,
                    line=_line(DRIVER, marker) + 1,
                    path_condition=[
                        lines[_line(DRIVER, marker) - 1].strip()[3:].split(":  #")[0]
                    ],
                )
            ],
        )
        for marker in ("A1", "A2", "A3", "A4", "A5")
    ]
    trees = {module.rel_path: module.tree for module in modules}
    with sql.across_calls(calls):
        record = reach.ReachGraph(reach.graph(calls, "m", "v", _DatabaseLeaves()))
        records = runnable.runnables(junctions, trees, calls, record)
        judge = runnable._Judge(calls, record, trees)
        (decoder,) = runnable._defined(trees, f"{UTILS}::decodeJSON")
        (hook,) = runnable._defined(trees, f"{UTILS}::unicodeConvert")
        extra = {"decodeJSON": judge.of(decoder), "unicodeConvert": judge.of(hook)}
    held: dict[str, Runnable] = {one.owner.rpartition("::")[2]: one for one in records}
    for name, verdict in extra.items():
        held[name] = Runnable(
            owner=verdict.owner,
            line=verdict.line,
            runnable=verdict.runnable,
            reads=list(verdict.reads),
            stops=[
                RunStop(kind=one.kind, at=one.at, what=one.what)
                for one in verdict.stops
            ],
        )
    return held


def _kinds(one: Runnable) -> set[str]:
    return {stop.kind for stop in one.stops}


def test_a_helper_of_its_arguments_alone_is_runnable_and_names_the_fields_it_reads(
    found: dict[str, Runnable],
) -> None:
    """A regular expression of its argument; a token a module binds to an item of a class's dict literal, logged
    through the module's logger; a spec's method reading its field and its class's constant, bound twice in the class
    body and read at its last binding there."""
    assert found["is_sub"].runnable
    assert found["is_sub"].asked == ["condition"]
    assert found["is_sub"].asked_at == [f"{DRIVER}:{_line(DRIVER, 'A1')}"]
    assert found["is_cloning"].runnable
    assert found["useScout"].runnable
    assert found["useScout"].reads == ["self.splitRule"]
    assert found["useScout"].bound == [f"{SPEC}::TaskSpec"]


def test_a_helper_reading_the_clock_the_configuration_or_a_changed_name_is_not(
    found: dict[str, Runnable],
) -> None:
    assert _kinds(found["is_late"]) == {"clock"}
    assert found["is_late"].reads == ["task.modificationTime"]
    assert _kinds(found["over_limit"]) == {"config"}
    assert _kinds(found["remember"]) == {"changes", "global"}


def test_a_helper_changing_what_it_is_handed_or_calling_one_that_is_not_runnable_is_not(
    found: dict[str, Runnable],
) -> None:
    assert _kinds(found["mark"]) == {"changes"}
    assert _kinds(found["setStatus"]) == {"changes"}
    assert _kinds(found["cloning_late"]) == {"callee"}
    assert [stop.what for stop in found["cloning_late"].stops] == [f"{UTILS}::is_late"]


def test_a_decoder_and_its_hook_are_runnable(found: dict[str, Runnable]) -> None:
    """``json.loads`` runs alike everywhere; the hook rebuilds what it is handed and calls itself."""
    assert found["decodeJSON"].runnable
    assert found["unicodeConvert"].runnable


def test_definitions_calling_each_other_are_runnable_only_together(
    found: dict[str, Runnable],
) -> None:
    """``ping`` calls ``pong``, which reads the clock and calls ``ping`` back: neither is runnable, though ``pong`` is
    read first and reaches ``ping`` while it is read."""
    assert _kinds(found["pong"]) == {"clock", "callee"}
    assert _kinds(found["ping"]) == {"callee"}
    assert [stop.what for stop in found["ping"].stops] == [f"{UTILS}::pong"]


CLOCKS = "pandaserver/taskbuffer/Clocks.py"
CLOCK_CALLS = """\
    import datetime
    import time
    import uuid

    from pandacommon.pandautils.PandaUtils import naive_utcnow


    def helper(s):
        return s


    def f(s, text, log):
        a = naive_utcnow()
        b = time.gmtime()
        c = time.gmtime(s)
        d = time.strptime(text, "%Y")
        e = uuid.uuid4()
        g = uuid.uuid5(uuid.NAMESPACE_DNS, text)
        h = len(text)
        i = text.split(",")
        j = datetime.timedelta(minutes=s)
        time.sleep(s)
        setattr(log, "x", 1)
        k = helper(s)
        m = log.mystery(s)
    """


def test_one_reading_says_how_each_call_makes_its_value_and_a_runnable_stops_on_it(
    tmp_path: Path,
) -> None:
    """read_call is the one reading of a call that the runnable records and the trace's steps share: the time, a draw,
    a function of the time handed its time, what the same arguments always give, a builtin, a str's method, a span, a
    wait that the map does not say, a builtin that changes what it is handed, the corpus's own definition, and a method
    of a value of no known class."""
    text = textwrap.dedent(CLOCK_CALLS)
    path = tmp_path / CLOCKS
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    parsed = calls_mod.parsed(path)
    assert parsed is not None
    module = SourceModule(
        package="pandaserver",
        rel_path=CLOCKS,
        tree=parsed[1],
        source=text,
        blob_sha=blob_sha(text),
    )
    calls = sqlwrite.corpus_calls([module], {"pandaserver": tmp_path / "pandaserver"})
    assert calls is not None
    (func,) = [
        one
        for one in parsed[1].body
        if isinstance(one, ast.FunctionDef) and one.name == "f"
    ]
    made = {
        ast.unparse(node): runnable.read_call(calls, CLOCKS, func, node)
        for node in ast.walk(func)
        if isinstance(node, ast.Call)
    }
    assert {call: read.made for call, read in made.items()} == {
        "naive_utcnow()": "time",
        "time.gmtime()": "time",
        "time.gmtime(s)": "handed",
        "time.strptime(text, '%Y')": "handed",
        "uuid.uuid4()": "draw",
        "uuid.uuid5(uuid.NAMESPACE_DNS, text)": "handed",
        "len(text)": "handed",
        "text.split(',')": "handed",
        "datetime.timedelta(minutes=s)": "handed",
        "time.sleep(s)": "library",
        "setattr(log, 'x', 1)": "changes",
        "helper(s)": "corpus",
        "log.mystery(s)": "unplaced",
    }
    assert (
        made["naive_utcnow()"].what == "pandacommon.pandautils.PandaUtils.naive_utcnow"
    )
    assert [one.func.name for one in made["helper(s)"].targets] == ["helper"]
    stops = {call: runnable._STOP_OF.get(read.made) for call, read in made.items()}
    assert stops["naive_utcnow()"] == stops["uuid.uuid4()"] == "clock"
    assert stops["time.gmtime(s)"] is None and stops["helper(s)"] is None
    assert stops["time.sleep(s)"] == "library" and stops["log.mystery(s)"] == "unplaced"
