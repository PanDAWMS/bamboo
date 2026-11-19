"""A row born with a value a statement asks wakes it too -- ``codemap.panda.wakers``' inserts, and the two readings
that settle the value TaskCommando's statement asks: an ``IN`` list's map spread into a bind map, a dict's keys."""

from __future__ import annotations

import ast
from pathlib import Path

from bamboo.codemap import pathcond, values
from bamboo.codemap.gitsource import blob_sha
from bamboo.codemap.models import (
    KEY_PARAMETER,
    KEY_ROWS,
    Anchor,
    ConditionNode,
    ConditionReading,
    ConditionValue,
    DbWrite,
    KeyFrom,
    KeySource,
    MapFragment,
    ReachNode,
    ReadSiteNode,
    SourceModule,
    WrittenValue,
)
from bamboo.codemap.panda import sql, wakers
from bamboo.codemap.panda.recognizers import sqlwrite

PACKAGE = "pkg"
REL = f"{PACKAGE}/proxy.py"

PROXY = """\
from pkg.helper import get_sql_IN_bind_variables


class TaskSpec:
    @classmethod
    def commands(cls):
        return {"kill": {"doing": "aborting"}, "finish": {"doing": "finishing"}}


class DBProxy:
    def send(self, task_id, command):
        self.cur.execute("INSERT INTO ATLAS_DEFT.QUEUE (TASK, CMD, OWNER) VALUES (:t, :c, :o)", {":t": task_id})

    def finish_achieved(self):
        for task_id in self.achieved():
            self.send(task_id, "finish")  # FINISH

    def kill_one(self, task_id):
        self.send(task_id, "kill")  # KILL

    def commanded(self):
        held = TaskSpec.commands()
        names, held_map = get_sql_IN_bind_variables(held.keys(), prefix=":cmd")
        var_map = {":owner": "DEFT", **held_map}
        self.cur.execute(f"SELECT TASK FROM ATLAS_DEFT.QUEUE WHERE OWNER=:owner AND CMD IN ({names})", var_map)


class CommandProxy:
    def external(self):
        held = TaskSpec.commands()
        external = [c for c in held if c not in internal_commands()]
        names, held_map = get_sql_IN_bind_variables(external, prefix=":cmd")
        var_map = {":owner": "DEFT", **held_map}
        self.cur.execute(f"SELECT TASK FROM ATLAS_DEFT.QUEUE WHERE OWNER=:owner AND CMD IN ({names})", var_map)

    def handed(self, internal):
        held = TaskSpec.commands()
        names, held_map = get_sql_IN_bind_variables([c for c in held if c not in internal], prefix=":cmd")
        self.cur.execute(f"SELECT TASK FROM ATLAS_DEFT.QUEUE WHERE CMD IN ({names})", held_map)

    def caller(self):
        self.handed(["kill"])


def internal_commands():
    return ["kill"]
"""

HELPER = """\
def get_sql_IN_bind_variables(values, prefix):
    return "", {}
"""


def _line(marker: str) -> int:
    (found,) = [
        number
        for number, text in enumerate(PROXY.splitlines(), 1)
        if text.rstrip().endswith(f"# {marker}")
    ]
    return found


def _def_line(name: str) -> int:
    (found,) = [
        number
        for number, text in enumerate(PROXY.splitlines(), 1)
        if text.strip().startswith(f"def {name}(")
    ]
    return found


def _corpus(tmp_path: Path):
    root = tmp_path / PACKAGE
    root.mkdir(parents=True, exist_ok=True)
    (root / "__init__.py").write_text("")
    modules = []
    for name, text in {"proxy": PROXY, "helper": HELPER}.items():
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
    return modules, sqlwrite.corpus_calls(modules, {PACKAGE: root})


def _function(modules, name: str):
    (module,) = [one for one in modules if one.rel_path == REL]
    return next(
        node
        for node in ast.walk(module.tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def test_an_in_lists_map_spread_into_a_bind_map_holds_the_dicts_keys(
    tmp_path: Path,
) -> None:
    """getTasksToExecCommand_JEDI's ``{":comm_owner": "DEFT", **comm_var_map}`` of ``commandStatusMap.keys()``."""
    modules, calls = _corpus(tmp_path)
    func = _function(modules, "commanded")
    with sql.across_calls(calls):
        (bind,) = sql.bound_values(func, "var_map", ":cmd__in")
        held = values.handed(values.resolver(), calls)(bind.value, func, REL)
    assert sorted(one.value for one in held) == ["finish", "kill"]


def test_an_in_list_made_by_a_comprehension_holds_the_values_its_tests_keep(tmp_path: Path) -> None:
    """getTasksToExecCommand_JEDI's ``[c for c in commandStatusMap if c not in JediTaskSpec.internal_commands()]``:
    the dict's keys the test keeps -- read where the test's values are the function's own, and not where a caller
    hands them (each caller's list would keep other keys)."""
    modules, calls = _corpus(tmp_path)
    found = {}
    for name, varmap in (("external", "var_map"), ("handed", "held_map")):
        func = _function(modules, name)
        with sql.across_calls(calls):
            (bind,) = sql.bound_values(func, varmap, ":cmd__in")
            found[name] = sorted(one.value for one in values.handed(values.resolver(), calls)(bind.value, func, REL))
    assert found == {"external": ["finish"], "handed": []}


QUEUE = ReadSiteNode(
    name=ReadSiteNode.make_name("m", f"{REL}::commanded", 25),
    map_id="m",
    derived_from="v",
    owner=f"{REL}::commanded",
    selects=["QUEUE.OWNER=DEFT"],
    condition_trees=[
        ConditionReading(
            entry="whole",
            tree=ConditionNode(
                node="all",
                items=[
                    ConditionNode(
                        node="equal",
                        text="OWNER=:owner",
                        subject="QUEUE.OWNER",
                        op="=",
                        values=[
                            ConditionValue(kind="bind", text=":owner", constant="DEFT")
                        ],
                        settled=True,
                    ),
                    ConditionNode(
                        node="equal",
                        text="CMD IN (:cmd__in)",
                        subject="QUEUE.CMD",
                        op="IN",
                        values=[
                            ConditionValue(kind="bind", text=":cmd__in", constant=one)
                            for one in ("kill", "finish", "pause")
                        ],
                        settled=True,
                    ),
                ],
            ),
        )
    ],
    anchor=Anchor(package=PACKAGE, file=REL, line_start=25),
)
ACHIEVED = ReadSiteNode(
    name=ReadSiteNode.make_name("m", f"{REL}::achieved", 40),
    map_id="m",
    derived_from="v",
    owner=f"{REL}::achieved",
    selects=["TaskSpec.status=running"],
    anchor=Anchor(package=PACKAGE, file=REL, line_start=40),
)


def _fragment() -> MapFragment:
    send = ReachNode(
        name=f"m:{REL}::DBProxy.send",
        map_id="m",
        derived_from="v",
        owner=f"{REL}::DBProxy.send",
        named_as=f"{REL}::send",
        cls=f"{REL}::DBProxy",
        function="send",
        anchor=Anchor(package=PACKAGE, file=REL, line_start=_def_line("send")),
        writes=[
            DbWrite(
                lines=[_def_line("send") + 1],
                table="QUEUE",
                kind="insert",
                columns=[
                    WrittenValue(column="TASK", path="task_id"),
                    WrittenValue(column="CMD", path="command"),
                    WrittenValue(column="OWNER", value="'DEFT'"),
                ],
            )
        ],
    )
    # The function's other statement keyed by task_id carries it to the rows finish_achieved's loop takes.
    carried = KeyFrom(
        owner=f"{REL}::send",
        statement=_def_line("send") + 1,
        table="JEDI_Tasks",
        term="jediTaskID=:jediTaskID",
        reads=True,
        sources=[
            KeySource(kind=KEY_PARAMETER, name="task_id"),
            KeySource(
                kind=KEY_ROWS,
                name=ACHIEVED.name,
                via=[f"{REL}:{_line('FINISH') - 1}", f"{REL}:{_line('FINISH')}"],
            ),
        ],
    )
    return MapFragment(
        map_id="m",
        derived_from="v",
        read_sites=[QUEUE.model_copy(deep=True), ACHIEVED.model_copy(deep=True)],
        reach_graph=[send],
        key_froms=[carried],
    )


def test_an_insert_giving_a_column_the_value_on_a_route_wakes_the_rows_it_makes(
    tmp_path: Path,
) -> None:
    """sendCommandTaskPanda's INSERT: ``finish`` on the route from the loop over getAchievedTasks_JEDI's rows."""
    _modules, calls = _corpus(tmp_path)
    fragment = _fragment()
    assert wakers.attach(fragment, calls) == 1
    (queue,) = [site for site in fragment.read_sites if site.name == QUEUE.name]
    found = {(w.value, w.kind, w.stated): (w.routes, w.keyed_by) for w in queue.wakers}
    assert found == {
        ("finish", "insert", True): ([[f"{REL}:{_line('FINISH')}"]], [ACHIEVED.name]),
        ("kill", "insert", True): ([[f"{REL}:{_line('KILL')}"]], []),
    }
    # The owner the statement asks of itself is no other column; no route hands pause.
    assert {w.subject for w in queue.wakers} == {"QUEUE.CMD"}


def test_without_the_corpuss_calls_no_insert_is_read(tmp_path: Path) -> None:
    fragment = _fragment()
    assert wakers.attach(fragment) == 0


def _poll(name: str, line: int, keys: list[str]) -> ReadSiteNode:
    """listJobOutputReport's 2401: unlocked or with an expired lock, older than the grace; a label kept out where a
    caller hands one."""
    clock = {"base": "naive_utcnow", "needs": []}
    return ReadSiteNode(
        name=ReadSiteNode.make_name("m", f"{REL}::{name}", line),
        map_id="m",
        derived_from="v",
        owner=f"{REL}::{name}",
        tables=["Report"],
        keys=keys,
        conditions=["lockedBy IS NULL OR lockedTime<:lockedTime@1", "timeStamp<:timeStamp@2"],
        condition_trees=[
            ConditionReading(
                entry="lockedBy IS NULL OR lockedTime<:lockedTime@1",
                tree=ConditionNode(
                    node="any",
                    items=[
                        ConditionNode(node="null", text="lockedBy IS NULL", subject="Report.lockedBy", settled=True),
                        ConditionNode(
                            node="compare",
                            text="lockedTime<:lockedTime",
                            subject="Report.lockedTime",
                            op="<",
                            values=[
                                ConditionValue(
                                    kind="bind", text=":lockedTime", scale_seconds=-60.0, **clock | {"needs": ["poll.time_limit"]}
                                )
                            ],
                        ),
                    ],
                ),
            ),
            ConditionReading(
                entry="timeStamp<:timeStamp@2",
                tree=ConditionNode(
                    node="compare",
                    text="timeStamp<:timeStamp",
                    subject="Report.timeStamp",
                    op="<",
                    values=[
                        ConditionValue(
                            kind="bind", text=":timeStamp", scale_seconds=-1.0, **clock | {"needs": ["poll.grace_period"]}
                        )
                    ],
                ),
            ),
            ConditionReading(
                entry="prodSourceLabel NOT IN (:al___in@3)",
                tree=ConditionNode(
                    node="unequal",
                    text="prodSourceLabel NOT IN (:al___in)",
                    subject="Report.prodSourceLabel",
                    op="NOT IN",
                    values=[ConditionValue(kind="bind", text=":al___in", constant="user")],
                    settled=True,
                ),
            ),
        ],
        anchor=Anchor(package=PACKAGE, file=REL, line_start=line),
    )


def _report(function: str, columns: list[WrittenValue]) -> ReachNode:
    return ReachNode(
        name=f"m:{REL}::DBProxy.{function}",
        map_id="m",
        derived_from="v",
        owner=f"{REL}::DBProxy.{function}",
        named_as=f"{REL}::{function}",
        cls=f"{REL}::DBProxy",
        function=function,
        anchor=Anchor(package=PACKAGE, file=REL, line_start=_def_line(function)),
        writes=[DbWrite(lines=[_def_line(function) + 1], table="Report", kind="insert", columns=columns)],
    )


def test_an_insert_whose_new_row_a_polls_terms_take_is_born_into_it(tmp_path: Path) -> None:
    """updateWorkers' INSERT into Job_Output_Report: lockedBy left NULL (no default), timeStamp stamped with the
    clock the poll's bind counts back from -- taken once the grace has passed --, the label a parameter the row's own
    value decides where a caller adds ``NOT IN``.  An INSERT locking its row is not taken (a NULL lockedTime is not
    older than anything), and one stamped by another clock is not read; a lookup by a bound key has none."""
    _modules, calls = _corpus(tmp_path)
    stamped = [
        WrittenValue(column="PandaID", path="task_id"),
        WrittenValue(column="jobStatus", value="'failed'"),
        WrittenValue(column="prodSourceLabel", path="command"),
        WrittenValue(column="timeStamp", value="naive_utcnow()"),
    ]
    fragment = MapFragment(
        map_id="m",
        derived_from="v",
        read_sites=[_poll("poll", 30, []), _poll("lookup", 31, ["Report.PandaID"])],
        reach_graph=[
            _report("send", stamped),
            _report("kill_one", [*stamped, WrittenValue(column="lockedBy", value="'me'")]),
            _report("finish_achieved", [*stamped[:3], WrittenValue(column="timeStamp", value="CURRENT_DATE")]),
        ],
    )
    assert wakers.attach(fragment, calls, {"report": {"jobstatus": "'waiting'"}}) == 1
    poll, lookup = fragment.read_sites
    assert lookup.wakers == []
    (born,) = poll.wakers
    assert (born.kind, born.owner, born.subject, born.stated) == ("born", f"{REL}::send", "Report", True)
    assert born.term == "lockedBy IS NULL AND timeStamp<:timeStamp"
    assert born.routes == [[f"{REL}:{_line('FINISH')}"], [f"{REL}:{_line('KILL')}"]]
    assert [(one.term, one.holds, one.value, one.scale_seconds, one.needs, one.added) for one in born.born] == [
        ("lockedBy IS NULL", "default", "NULL", None, [], False),
        ("timeStamp<:timeStamp", "after", "naive_utcnow", -1.0, ["poll.grace_period"], False),
        ("prodSourceLabel NOT IN (:al___in)", "open", "command", None, [], True),
    ]
