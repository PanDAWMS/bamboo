"""The writes a row a statement selects waits on besides the mover -- ``codemap.panda.wakers``."""

from __future__ import annotations

from bamboo.codemap.models import (
    KEY_ROWS,
    OFF_ROW_NO_WRITE,
    Anchor,
    Branch,
    ConditionNode,
    ConditionReading,
    ConditionValue,
    JunctionNode,
    KeyFrom,
    KeySource,
    MapFragment,
    OffRow,
    ReadSiteNode,
)
from bamboo.codemap.panda import wakers

MODULE = "pkg/proxy.py"


def _term(
    text: str, subject: str, *values: str, node: str = "equal", settled: bool = True
) -> ConditionNode:
    return ConditionNode(
        node=node,
        text=text,
        subject=subject,
        op="IN" if len(values) > 1 else "=",
        values=[
            ConditionValue(kind="bind", text=":b", constant=value) for value in values
        ],
        settled=settled,
    )


def _site(line: int, selects: list[str], tree: ConditionNode) -> ReadSiteNode:
    owner = f"{MODULE}::f{line}"
    return ReadSiteNode(
        name=ReadSiteNode.make_name("m", owner, line),
        map_id="m",
        derived_from="v",
        owner=owner,
        selects=selects,
        condition_trees=[ConditionReading(entry="whole", tree=tree)],
        anchor=Anchor(package="pkg", file=MODULE, line_start=line),
    )


def _writer(owner: str, subject: str, *branches: Branch) -> JunctionNode:
    return JunctionNode(
        name=f"m:{owner}:{subject}",
        map_id="m",
        derived_from="v",
        subject=subject,
        owner=f"{MODULE}::{owner}",
        branches=list(branches),
    )


def _keyed(owner: str, line: int, subject: str, *sites: ReadSiteNode) -> KeyFrom:
    return KeyFrom(
        owner=f"{MODULE}::{owner}",
        statement=line + 1,
        table="JEDI_Datasets",
        term="jediTaskID=:jediTaskID",
        writes={subject: [line]},
        sources=[KeySource(kind=KEY_ROWS, name=site.name) for site in sites],
    )


# getDatasetsToFeedContents_JEDI's 130: a task at defined with a dataset defined or toupdate, or one at running /
# ready / defined with a mutable dataset.
FEED = _site(
    130,
    [
        "JediTaskSpec.status=defined",
        "JediTaskSpec.status=ready",
        "JediTaskSpec.status=running",
        "JediDatasetSpec.status=toupdate",
    ],
    ConditionNode(
        node="any",
        items=[
            ConditionNode(
                node="all",
                items=[
                    _term("tabT.status=:ts_defined", "JediTaskSpec.status", "defined"),
                    _term(
                        "tabD.status IN (:ds)",
                        "JediDatasetSpec.status",
                        "defined",
                        "toupdate",
                    ),
                ],
            ),
            ConditionNode(
                node="all",
                items=[
                    _term(
                        "tabT.status IN (:ts)",
                        "JediTaskSpec.status",
                        "running",
                        "ready",
                        "defined",
                    ),
                    _term("tabD.state=:mutable", "JediDatasetSpec.state", "mutable"),
                    _term(
                        "tabD.lockedBy IS NULL",
                        "JediDatasetSpec.lockedBy",
                        node="null",
                        settled=False,
                    ),
                    _term(
                        "tabD.status<>:removed",
                        "JediDatasetSpec.status",
                        "removed",
                        node="unequal",
                    ),
                ],
            ),
        ],
    ),
)
# restartTasksForContentsUpdate_JEDI's 1830: the tasks at defined whose ready datasets 1844 sets to toupdate.
RESTART = _site(
    1830,
    ["JediTaskSpec.status=defined"],
    _term("tabT.status=:ts1", "JediTaskSpec.status", "defined"),
)


def _fragment(*extra: JunctionNode, keys: list[KeyFrom] | None = None) -> MapFragment:
    return MapFragment(
        map_id="m",
        derived_from="v",
        read_sites=[FEED.model_copy(deep=True), RESTART.model_copy(deep=True)],
        junctions=[
            _writer(
                "restart",
                "JediDatasetSpec.status",
                Branch(outcome="toupdate", line=1844),
            ),
            _writer(
                "retry", "JediDatasetSpec.status", Branch(outcome="toupdate", line=5230)
            ),
            _writer(
                "set_state", "JediDatasetSpec.state", Branch(outcome="mutable", line=70)
            ),
            *extra,
        ],
        key_froms=keys
        if keys is not None
        else [_keyed("restart", 1844, "JediDatasetSpec.status", RESTART)],
    )


def _at(
    fragment: MapFragment, line: int, pair: str
) -> list[tuple[str, int, str, bool, list[str]]]:
    (site,) = [s for s in fragment.read_sites if s.anchor.line_start == line]
    return [
        (w.owner.split("::")[1], w.line, w.value, w.stated, w.keyed_at)
        for w in site.wakers
        if pair in w.pairs
    ]


def test_a_write_of_a_value_a_term_asks_wakes_the_rows_at_the_values_its_conjunction_admits() -> (
    None
):
    fragment = _fragment()
    assert wakers.attach(fragment) == 1
    # A task at defined waits on its datasets: restart's 1844 is handed the tasks 1830 selects at defined, retry's
    # write is not; the mutable branch admits defined too.
    assert _at(fragment, 130, "JediTaskSpec.status=defined") == [
        ("set_state", 70, "mutable", True, []),
        ("restart", 1844, "toupdate", True, ["JediTaskSpec.status=defined"]),
        ("retry", 5230, "toupdate", True, []),
    ]
    # At ready only the mutable branch is open: the dataset's status is not asked of it.
    assert _at(fragment, 130, "JediTaskSpec.status=ready") == [
        ("set_state", 70, "mutable", True, [])
    ]
    # A dataset at toupdate is asked the task's status too, which nothing here writes, and the state.
    assert {
        w.term
        for w in fragment.read_sites[0].wakers
        if "JediDatasetSpec.status=toupdate" in w.pairs
    } == {"tabD.state=:mutable"}


def test_a_run_time_writer_its_frames_do_not_rule_out_is_kept_unstated_and_one_off_its_row_is_not() -> (
    None
):
    fragment = _fragment(
        _writer(
            "update_status",
            "JediDatasetSpec.status",
            Branch(outcome="runtime(datasetStatus)", line=300, tier=2),
            Branch(
                outcome="runtime(f'merge_{s}')",
                line=310,
                tier=2,
                value_frames=["merge_{}"],
            ),
        ),
        _writer(
            "ask",
            "JediDatasetSpec.status",
            Branch(
                outcome="toupdate",
                line=400,
                off_row=OffRow(why=OFF_ROW_NO_WRITE, functions=["ask"]),
            ),
        ),
    )
    wakers.attach(fragment)
    found = {
        (owner, line, stated)
        for owner, line, value, stated, _ in _at(
            fragment, 130, "JediTaskSpec.status=defined"
        )
    }
    assert ("update_status", 300, False) in found
    assert not {line for _, line, _ in found} & {310, 400}


def test_the_statements_own_rows_wake_nothing_it_refused() -> None:
    # The mover's write on the rows 130 itself took: keyed by 130, so it is said, but at no pair.
    fragment = _fragment(keys=[_keyed("restart", 1844, "JediDatasetSpec.status", FEED)])
    wakers.attach(fragment)
    (site,) = [s for s in fragment.read_sites if s.anchor.line_start == 130]
    (restart,) = [w for w in site.wakers if w.line == 1844]
    assert restart.keyed_by == [FEED.name] and restart.keyed_at == []


def test_unsettled_unequal_and_null_terms_ask_no_write() -> None:
    fragment = _fragment(
        _writer("unlock", "JediDatasetSpec.lockedBy", Branch(outcome="None", line=90)),
        _writer("remove", "JediDatasetSpec.status", Branch(outcome="removed", line=95)),
    )
    wakers.attach(fragment)
    lines = {w.line for s in fragment.read_sites for w in s.wakers}
    assert not lines & {90, 95}
    # The statement that asks only its own subject has none.
    assert [s.wakers for s in fragment.read_sites if s.anchor.line_start == 1830] == [
        []
    ]
