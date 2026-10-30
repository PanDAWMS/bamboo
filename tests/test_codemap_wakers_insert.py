"""A row born with a value a statement asks wakes it too -- ``codemap.panda.wakers``' inserts, and the two readings
that settle the value TaskCommando's statement asks: an ``IN`` list's map spread into a bind map, a dict's keys."""

from __future__ import annotations

import ast
from pathlib import Path

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
from bamboo.codemap.panda import pathcond, sql, values, wakers
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
