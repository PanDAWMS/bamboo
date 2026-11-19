"""Field tests -- the plan answering the querier's rows asks (stage 2 of the rows ask).

The asks are in the shapes the querier writes them (querier-ra db6d1c8, ``--rows-asks``); the dump's tables
are test_fieldtest_rows's, and its read sites carry the fields the plan reads (``anchor``, ``keys``,
``condition_trees``), as the map writes them.
"""

from __future__ import annotations

import datetime

from bamboo.scripts.panda.fieldtest import fetch as FE
from bamboo.scripts.panda.fieldtest import rows as RB
from bamboo.scripts.panda.fieldtest import rowsask as RA
from tests.test_fieldtest_rows import (
    AS_OF,
    DUMP,
    JOB_Q,
    PID,
    SPELLINGS,
    TASK_Q,
    TID,
    T,
    _job_panda,
    _task_panda,
)

TABLES = FE.Tables.of(DUMP)
SITE = "pandaserver/taskbuffer/db_proxy_mods/task_standalone_module.py:1490"
#: the read site of getPandaIDsWithTask_JEDI's statement (x13M110m), cut to what the plan reads
READ_SITE = {
    "anchor": {"file": SITE.partition(":")[0], "line_start": 1490},
    "keys": ["JobSpec.jediTaskID"],
    "condition_trees": [
        {
            "entry": "modificationTime>(CURRENT_DATE-30)",
            "tree": {
                "node": "any",
                "items": [
                    {
                        "node": "all",
                        "items": [
                            {
                                "node": "compare",
                                "items": [],
                                "subject": "JobSpec.modificationTime",
                                "op": ">",
                                "values": [
                                    {"kind": "clock", "offset_seconds": -2592000.0}
                                ],
                            }
                        ],
                    }
                ],
            },
        }
    ],
}
SITES = RA.Sites({**DUMP, "read_sites": [READ_SITE]})
AT = ["2026-09-28 07:21:55,714", "2026-09-28 07:47:47,414"]


def _line(ask: dict) -> dict:
    return {"rows": ask, "step": "X", "why": "the rows file lacks it"}


def _plan(ask: dict) -> RA.Plan:
    return RA.plan(_line(ask), TABLES, SITES)


def _statement(tables: list[str], binds: dict) -> dict:
    return {
        "kind": "statement",
        "method": "getPandaIDsWithTask_JEDI",
        "site": [SITE],
        "tables": tables,
        "binds": binds,
        "at": AT,
    }


def test_a_dataset_of_a_task_is_named_by_both_identity_attributes():
    assert TABLES.identity["JediDatasetSpec"] == ("datasetID", "jediTaskID")
    assert "JediDatasetSpec" not in TABLES.key


def test_a_collection_the_rows_answer_holds_is_read_by_the_call_reading_the_row():
    got = _plan(
        {
            "kind": "collection",
            "row": {"entity": "JobSpec", "key": {"PandaID": 7339032371}},
            "attr": "Files",
            "entity": None,
        }
    )
    part, request = FE.KEY_REQUEST["JobSpec"]
    assert (got.why, got.histories) == ("", None)
    assert [(c.part, c.request) for c in got.calls] == [(part, request(7339032371))]


def test_a_collection_of_a_named_class_is_read_where_its_tables_name_the_rows_key():
    got = _plan(
        {
            "kind": "collection",
            "row": {"entity": "JediTaskSpec", "key": {"jediTaskID": 52757505}},
            "attr": "datasets",
            "entity": "JediDatasetSpec",
        }
    )
    assert [(c.part, c.request) for c in got.calls] == [
        (
            "db_read",
            {"tables": ["JEDI_Datasets"], "where": [["jediTaskID", "=", 52757505]]},
        )
    ]


def test_a_collection_no_rule_reads_says_why():
    unnamed = _plan(
        {
            "kind": "collection",
            "row": {"entity": "JediTaskSpec", "key": {"jediTaskID": 1}},
            "attr": "x",
            "entity": None,
        }
    )
    assert not unnamed.calls and "names no element class" in unnamed.why
    other_key = _plan(
        {
            "kind": "collection",
            "row": {"entity": "JobSpec", "key": {"PandaID": 1}},
            "attr": "x",
            "entity": "JediDatasetSpec",
        }
    )
    assert not other_key.calls and "not by ['PandaID']" in other_key.why


#: S1's ask for the rows of a class its statement joins to the row (querier-nr 6ee4df4)
S1_ASK = {
    "kind": "collection",
    "row": {"entity": "JediTaskSpec", "key": {"jediTaskID": TID}},
    "attr": None,
    "entity": "JediDatasetSpec",
    "where": {"jediTaskID": TID},
    "via": [{"from": "JediTaskSpec.jediTaskID", "to": "JediDatasetSpec.jediTaskID"}],
}


def test_a_joined_class_is_read_where_the_joins_to_columns_hold_the_rows_values():
    via = [{"from": "JediTaskSpec.jediTaskID", "to": "JediDatasetSpec.masterID"}]
    got = _plan({**S1_ASK, "where": {"masterID": 9}, "via": via})
    assert [(c.part, c.request) for c in got.calls] == [
        ("db_read", {"tables": ["JEDI_Datasets"], "where": [["masterID", "=", 9]]})
    ]


def test_a_joined_ask_whose_via_does_not_name_its_where_says_why():
    no_via = _plan({k: v for k, v in S1_ASK.items() if k != "via"})
    assert (
        not no_via.calls
        and "does not join JediTaskSpec to JediDatasetSpec" in no_via.why
    )
    via = [{"from": "JobSpec.jediTaskID", "to": "JediDatasetSpec.jediTaskID"}]
    other_class = _plan({**S1_ASK, "via": via})
    assert not other_class.calls and "does not join" in other_class.why


def test_a_joined_class_the_rows_answer_holds_is_read_by_the_call_reading_the_row():
    got = _plan(
        {
            "kind": "collection",
            "row": {"entity": "JobSpec", "key": {"PandaID": PID}},
            "attr": None,
            "entity": "FileSpec",
            "where": {"PandaID": PID},
            "via": [{"from": "JobSpec.PandaID", "to": "FileSpec.PandaID"}],
        }
    )
    part, request = FE.KEY_REQUEST["JobSpec"]
    assert [(c.part, c.request) for c in got.calls] == [(part, request(PID))]


def test_a_statement_reads_its_tables_by_the_read_sites_column_with_the_calls_values():
    got = _plan(
        _statement(
            ["jobsDefined4", "jobsActive4"],
            {"jediTaskID": [52757505, 52757505, "unknown"], "onlyActive": [True] * 3},
        )
    )
    where = [["jediTaskID", "in", [52757505]]]
    assert (got.why, got.histories) == ("", "JobSpec")
    assert [(c.role, c.request) for c in got.calls] == [
        (
            "job_rows",
            {
                "tables": ["jobsDefined4", "jobsActive4"],
                "where": where,
                "columns": FE.JOB_ROW_COLUMNS,
            },
        ),
        (
            "job_counts",
            {
                "tables": ["jobsDefined4", "jobsActive4"],
                "where": where,
                "group_by": ["jobStatus"],
                "aggregates": [["count", "*"]],
            },
        ),
    ]


def test_a_statements_archive_read_is_bounded_by_the_read_sites_clock_term_before_the_first_pass():
    got = _plan(_statement(["jobsActive4", "jobsArchived"], {"jediTaskID": [7, 7]}))
    first = datetime.datetime(2026, 9, 28, 7, 21, 55)
    bound = first - datetime.timedelta(days=30) - FE.CLOCK_SLACK
    arch = [c.request for c in got.calls if c.request["tables"] == ["jobsArchived"]]
    assert len(arch) == 2
    assert all(
        r["where"] == [["jediTaskID", "in", [7]], FE.since("modificationTime", bound)]
        for r in arch
    )


def test_a_statement_no_rule_reads_says_why():
    nowhere = _plan(
        {**_statement(["jobsActive4"], {"jediTaskID": [1]}), "site": ["x.py:1"]}
    )
    assert not nowhere.calls and "no read site" in nowhere.why
    no_value = _plan(
        _statement(["jobsActive4"], {"jediTaskID": ["unknown"], "onlyActive": [True]})
    )
    assert not no_value.calls and "gives a value of JobSpec.jediTaskID" in no_value.why
    other_tables = _plan(_statement(["JEDI_Tasks"], {"jediTaskID": [1]}))
    assert not other_tables.calls and "are not all JobSpec's" in other_tables.why


def test_an_sql_ask_is_listed_for_a_hand():
    ask = {
        "kind": "sql",
        "name": "getJobs@2404 held 'activated' at the move, by the order's items before the key",
        "sql": "WITH ...",
        "binds": {},
        "question": {"part": "H held at"},
    }
    got = _plan(ask)
    assert (
        not got.calls and "listed for a hand" in got.why and "hand-r<n>.json" in got.why
    )


# ------------------------------------------------------------------ answers, laid out


def _doc(*answers: dict) -> dict:
    return {"answers": list(answers)}


async def test_a_statements_answer_lays_out_as_the_tasks_fetch_lays_out_its_jobs():
    fetched = RB.build(
        await FE.fetch(TASK_Q, TABLES, _task_panda().parts()), SPELLINGS, TABLES
    )
    panda = _task_panda()
    line = _line(
        _statement(
            ["jobsActive4", "jobsArchived4", "jobsDefined4", "jobsArchived"],
            {"jediTaskID": [TID]},
        )
    )
    got = await RA.answer(line, RA.plan(line, TABLES, SITES), panda.parts(), TABLES)
    assert [c[0] for c in panda.calls] == ["db_read", "job_history"]
    sections, left_out = RB.answered(
        [_doc(got)], "JediTaskSpec", str(TID), SPELLINGS, TABLES
    )
    assert left_out == []
    for name in ("job_rows", "job_statuslog", "counts"):
        assert sections[name] == fetched[name], name


async def test_the_key_jobs_files_answer_lays_out_as_the_jobs_fetch_lays_them_out():
    fetched = RB.build(
        await FE.fetch(JOB_Q, TABLES, _job_panda().parts()), SPELLINGS, TABLES
    )
    line = _line(
        {
            "kind": "collection",
            "row": {"entity": "JobSpec", "key": {"PandaID": PID}},
            "attr": "Files",
            "entity": None,
        }
    )
    got = await RA.answer(
        line, RA.plan(line, TABLES, SITES), _job_panda().parts(), TABLES
    )
    sections, left_out = RB.answered(
        [_doc(got)], "JobSpec", str(PID), SPELLINGS, TABLES
    )
    assert left_out == [] and sections == {
        k: fetched[k] for k in ("files", "collections")
    }
    other, why = RB.answered([_doc(got)], "JobSpec", "1", SPELLINGS, TABLES)
    assert (
        other == {}
        and "no section of the rows file holds the rows of JobSpec.Files" in why[0]
    )


#: the spellings with a dataset's columns, as the pin's JediDatasetSpec.attributes spells them
DS_SPELLINGS = {
    **SPELLINGS,
    "JediDatasetSpec": {
        "datasetid": "datasetID",
        "jeditaskid": "jediTaskID",
        "type": "type",
    },
}


def _db_read(tables: dict[str, list[dict]]):
    async def db_read(queries):
        answer = {
            "status": "done",
            "truncated": False,
            "tables": [{"table": t, "rows": r} for t, r in tables.items()],
        }
        return {"success": True, "data": {"as_of": T(AS_OF), "answers": [answer]}}

    return db_read


async def _joined_answer(tables: dict[str, list[dict]]) -> dict:
    line = _line(S1_ASK)
    p = RA.plan(line, TABLES, SITES)
    return await RA.answer(line, p, FE.Parts(db_read=_db_read(tables)), TABLES)


async def test_a_joined_classs_rows_are_a_linked_entry_per_table_as_the_class_spells_them():
    row = {"datasetid": 11, "jeditaskid": TID, "type": "input", "nfiles": 3}
    got = await _joined_answer({"JEDI_Datasets": [row]})
    sections, left_out = RB.answered(
        [_doc(got)], "JediTaskSpec", str(TID), DS_SPELLINGS, TABLES
    )
    assert left_out == []
    assert sections["linked"] == [
        {
            "class": "JediDatasetSpec",
            "via": S1_ASK["via"][0],
            "asked": [TID],
            "table": "JEDI_Datasets",
            "source": f"R as of 2026-10-13 10:00:00, where {{'jediTaskID': {TID}}}",
            "as_of": "2026-10-13 10:00:00",
            "clock": RB.DATABASE,
            "rows": [{"datasetID": 11, "jediTaskID": TID, "type": "input"}],
        }
    ]


async def test_a_joined_table_with_no_row_is_an_entry_too_so_the_querier_does_not_ask_again():
    got = await _joined_answer({"JEDI_Datasets": []})
    sections, _ = RB.answered(
        [_doc(got)], "JediTaskSpec", str(TID), DS_SPELLINGS, TABLES
    )
    assert [(e["table"], e["rows"]) for e in sections["linked"]] == [
        ("JEDI_Datasets", [])
    ]


async def test_a_joined_answer_no_entry_can_hold_says_why():
    steps = (await _joined_answer({"JEDI_Datasets": []}))["steps"]
    two = {**S1_ASK, "via": S1_ASK["via"] * 2}
    assert RB.joined(two, steps, DS_SPELLINGS)[1].startswith(
        "a linked entry names one pair"
    )
    assert RB.joined(S1_ASK, steps, SPELLINGS) == (
        [],
        "no column list of JediDatasetSpec was read from the pin",
    )


def test_the_rows_steps_linked_entries_replace_the_fetchs_of_the_same_pair_values_and_table():
    a = {
        "class": "X",
        "via": {"from": "A.k", "to": "X.k"},
        "asked": [1],
        "table": "T",
        "rows": [1],
    }
    b, c = {**a, "rows": [2]}, {**a, "asked": [2], "rows": [3]}
    assert RB.lay_linked([a, c], [b]) == [b, c]


async def test_an_ask_no_rule_reads_asks_nothing_and_is_left_out_with_its_why():
    panda = _task_panda()
    line = _line({"kind": "sql", "name": "w", "sql": "WITH ...", "binds": {}})
    got = await RA.answer(line, RA.plan(line, TABLES, SITES), panda.parts(), TABLES)
    assert panda.calls == [] and got["steps"] == []
    sections, left_out = RB.answered([_doc(got)], "JobSpec", "1", SPELLINGS, TABLES)
    assert sections == {} and left_out == [
        "sql w: a window over the history no call reads: listed for a hand (hand-r<n>.json in the records directory)"
    ]


def test_a_round_of_answers_is_one_past_the_last(tmp_path):
    assert RA.next_round(tmp_path) == 1
    (tmp_path / "rows-r1.json").write_text("{}")
    (tmp_path / "rows-r3.json").write_text("{}")
    (tmp_path / "records-r7.jsonl").write_text("")
    assert RA.next_round(tmp_path) == 4


#: design H's asks: a job's task and a task's parent, each joined by its link's pair
TASK_OF_JOB = {
    "kind": "collection",
    "row": {"entity": "JobSpec", "key": {"PandaID": PID}},
    "attr": None,
    "entity": "JediTaskSpec",
    "where": {"jediTaskID": TID},
    "via": [{"from": "JobSpec.jediTaskID", "to": "JediTaskSpec.jediTaskID"}],
}
PARENT = 52757000
PARENT_OF_TASK = {
    "kind": "collection",
    "row": {"entity": "JediTaskSpec", "key": {"jediTaskID": TID}},
    "attr": None,
    "entity": "JediTaskSpec",
    "where": {"jediTaskID": PARENT},
    "via": [{"from": "JediTaskSpec.parent_tid", "to": "JediTaskSpec.jediTaskID"}],
}


def test_a_joined_class_whose_histories_a_call_reads_has_them_follow_its_rows():
    for ask, value in ((TASK_OF_JOB, TID), (PARENT_OF_TASK, PARENT)):
        got = _plan(ask)
        assert [(c.part, c.request) for c in got.calls] == [
            (
                "db_read",
                {"tables": ["JEDI_Tasks"], "where": [["jediTaskID", "=", value]]},
            )
        ]
        assert (got.histories, got.rows_role) == ("JediTaskSpec", "collection")
    # no call reads a dataset's history
    datasets = _plan(S1_ASK)
    assert (datasets.histories, datasets.rows_role) == (None, "")


def test_an_ask_naming_its_columns_reads_them_with_those_it_selects_by_and_those_its_histories_need():
    got = _plan({**PARENT_OF_TASK, "columns": ["status", "jediTaskID"]})
    assert got.calls[0].request["columns"] == ["status", "jediTaskID", "creationDate"]
    datasets = _plan({**S1_ASK, "columns": ["masterID", "type"]})
    assert datasets.calls[0].request["columns"] == ["masterID", "type", "jediTaskID"]
    assert "columns" not in _plan(S1_ASK).calls[0].request


async def test_a_tasks_history_is_asked_one_task_per_call():
    panda = _task_panda()
    await FE.HISTORY["JediTaskSpec"]([TID, PARENT], None, FE.Record({}), panda.parts())
    assert panda.calls == [("task_history", TID), ("task_history", PARENT)]


async def test_a_parents_row_is_a_linked_entry_and_its_history_is_in_statuslogs_by_its_key():
    panda = _task_panda()
    panda.tables["JEDI_Tasks"] = [
        {"jeditaskid": PARENT, "status": "done", "creationdate": "2026-09-20 00:00:00"}
    ]
    line = _line(PARENT_OF_TASK)
    got = await RA.answer(line, RA.plan(line, TABLES, SITES), panda.parts(), TABLES)
    assert [c[0] for c in panda.calls] == ["db_read", "task_history"]
    assert panda.calls[1] == ("task_history", PARENT)
    sections, left_out = RB.answered(
        [_doc(got)], "JediTaskSpec", str(TID), SPELLINGS, TABLES
    )
    assert left_out == []
    (entry,) = sections["linked"]
    assert (entry["class"], entry["asked"]) == ("JediTaskSpec", [PARENT])
    assert [r["jediTaskID"] for r in entry["rows"]] == [PARENT]
    log = sections["statuslogs"]["JediTaskSpec"]
    assert log["key"] == "JediTaskSpec.jediTaskID"
    assert list(log["by_key"]) == [str(PARENT)]
    parent = log["by_key"][str(PARENT)]
    assert [s["status"] for s in parent["states"]] == ["running", "finished"]
    assert parent["as_of"] == "2026-10-13 10:00:00"


def test_the_rows_steps_statuslogs_are_laid_over_the_fetchs_by_class_and_key():
    key = "JediTaskSpec.jediTaskID"
    fetched = {"JediTaskSpec": {"key": key, "by_key": {"1": {"states": ["a"]}}}}
    laid = {
        "JediTaskSpec": {
            "key": key,
            "by_key": {"1": {"states": ["b"]}, "2": {"states": ["c"]}},
        },
        "JobSpec": {"key": "JobSpec.PandaID", "by_key": {"3": {"states": []}}},
    }
    assert RB.lay_statuslogs(fetched, laid) == {
        "JediTaskSpec": {
            "key": key,
            "by_key": {"1": {"states": ["b"]}, "2": {"states": ["c"]}},
        },
        "JobSpec": {"key": "JobSpec.PandaID", "by_key": {"3": {"states": []}}},
    }
    # the fetch's is not changed in place
    assert fetched["JediTaskSpec"]["by_key"] == {"1": {"states": ["a"]}}


# ------------------------------------------------------------------ a task's datasets through task/get_dataset_rows


def _dataset_rows(rows: list[dict], truncated: bool = False):
    """task/get_dataset_rows (panda-server 80134dc2): the columns as JediDatasetSpec spells them."""

    async def dataset_rows(task_id):
        return {
            "success": True,
            "message": "",
            "data": {
                "rows": rows,
                "truncated": truncated,
                "max_rows": 1000,
                "as_of": T(AS_OF),
            },
        }

    return dataset_rows


def test_by_the_endpoint_a_tasks_datasets_by_its_key_are_read_by_task_get_dataset_rows():
    line = _line(S1_ASK)
    got = RA.plan(line, TABLES, SITES, "endpoint")
    assert [(c.part, c.role, c.request) for c in got.calls] == [
        (FE.TASK_DATASET_ROWS, "collection", {"task_id": TID})
    ]
    assert got.histories is None and not got.why
    # db_read by default, as before
    assert [c.part for c in RA.plan(line, TABLES, SITES).calls] == ["db_read"]
    # rows selected by another column stay db_read's
    via = [{"from": "JediTaskSpec.jediTaskID", "to": "JediDatasetSpec.masterID"}]
    other = RA.plan(
        _line({**S1_ASK, "where": {"masterID": 9}, "via": via}), TABLES, SITES, "endpoint"
    )
    assert [c.part for c in other.calls] == ["db_read"]


async def _endpoint_answer(rows: list[dict], truncated: bool = False) -> dict:
    line = _line(S1_ASK)
    p = RA.plan(line, TABLES, SITES, "endpoint")
    parts = FE.Parts(dataset_rows=_dataset_rows(rows, truncated))
    return await RA.answer(line, p, parts, TABLES)


async def test_by_the_endpoint_a_tasks_datasets_lay_out_as_rs():
    by_r = await _joined_answer(
        {"JEDI_Datasets": [{"datasetid": 11, "jeditaskid": TID, "type": "input"}]}
    )
    by_e = await _endpoint_answer([{"datasetID": 11, "jediTaskID": TID, "type": "input"}])
    assert [s["part"] for s in by_e["steps"]] == [FE.TASK_DATASET_ROWS]
    (r_entry,) = RB.answered([_doc(by_r)], "JediTaskSpec", str(TID), DS_SPELLINGS, TABLES)[0]["linked"]
    (e_entry,) = RB.answered([_doc(by_e)], "JediTaskSpec", str(TID), DS_SPELLINGS, TABLES)[0]["linked"]
    assert e_entry["source"] == f"task/get_dataset_rows as of 2026-10-13 10:00:00, where {{'jediTaskID': {TID}}}"
    assert {k: v for k, v in e_entry.items() if k != "source"} == {
        k: v for k, v in r_entry.items() if k != "source"
    }
    assert e_entry["table"] == "JEDI_Datasets"


async def test_by_the_endpoint_a_task_with_no_dataset_is_an_entry_too():
    sections, _ = RB.answered(
        [_doc(await _endpoint_answer([]))], "JediTaskSpec", str(TID), DS_SPELLINGS, TABLES
    )
    assert [(e["table"], e["rows"]) for e in sections["linked"]] == [("JEDI_Datasets", [])]


async def test_by_the_endpoint_a_cut_answer_is_left_out_and_said():
    got = await _endpoint_answer([{"datasetID": 11, "jediTaskID": TID}], truncated=True)
    sections, left_out = RB.answered([_doc(got)], "JediTaskSpec", str(TID), DS_SPELLINGS, TABLES)
    assert "linked" not in sections and left_out == ["collection JediDatasetSpec: truncated"]


# ------------------------------------------------------------------ another row by its key (kind "row")


def _row_ask(cls: str, key: dict) -> dict:
    return {"kind": "row", "row": {"entity": cls, "key": key}}


def test_a_row_ask_reads_the_row_by_the_call_a_fetch_reads_its_key_by_then_its_history():
    for cls, key in (("JobSpec", {"PandaID": str(PID)}), ("JediTaskSpec", {"jediTaskID": TID})):
        got = _plan(_row_ask(cls, key))
        part, request = FE.KEY_REQUEST[cls]
        assert (got.why, got.histories, got.rows_role) == ("", cls, "row")
        assert [(c.part, c.request) for c in got.calls] == [(part, request(int(*key.values())))]


def test_a_row_ask_by_a_column_other_than_the_classs_key_says_why():
    for cls, key in (("JobSpec", {"jediTaskID": TID}), ("JediDatasetSpec", {"datasetID": 11})):
        got = _plan(_row_ask(cls, key))
        assert got.calls == [] and got.why == f"no call reads a {cls} by {sorted(key)}"


async def test_another_jobs_row_lays_out_in_rows_by_key_as_the_jobs_fetch_lays_out_its_own():
    fetching = _job_panda()
    fetched = RB.build(await FE.fetch(JOB_Q, TABLES, fetching.parts()), SPELLINGS, TABLES)
    panda = _job_panda()
    line = _line(_row_ask("JobSpec", {"PandaID": PID}))
    got = await RA.answer(line, RA.plan(line, TABLES, SITES), panda.parts(), TABLES)
    assert panda.calls == fetching.calls  # the row, then its history from its creation
    sections, left_out = RB.answered([_doc(got)], "JediTaskSpec", str(TID), SPELLINGS, TABLES)
    assert left_out == [] and set(sections) == {"rows_by_key", "statuslogs"}
    entry = sections["rows_by_key"]["JobSpec"][str(PID)]
    own = fetched["rows"][RB.ENTITY["JobSpec"]]
    assert entry["row"] == {c: v["value"] for c, v in own.items()}
    assert {k: entry[k] for k in ("source", "as_of", "clock")} == {
        k: v for k, v in own["PandaID"].items() if k != "value"
    }
    assert entry["files"] == [{c: v["value"] for c, v in f.items()} for f in fetched["files"]]
    log = sections["statuslogs"]["JobSpec"]["by_key"][str(PID)]
    assert log["states"] == fetched["statuslog"]["states"]


async def test_another_job_with_no_row_asks_no_history_and_is_left_out_with_its_why():
    panda = _job_panda()
    line = _line(_row_ask("JobSpec", {"PandaID": 1}))
    got = await RA.answer(line, RA.plan(line, TABLES, SITES), panda.parts(), TABLES)
    assert panda.calls == [("jobs", [1])]
    sections, left_out = RB.answered([_doc(got)], "JediTaskSpec", str(TID), SPELLINGS, TABLES)
    assert "rows_by_key" not in sections
    assert left_out == [
        "row JobSpec PandaID=1: no JobSpec row for PandaID 1: in no live table, nor in the archive's last 90 days"
    ]
