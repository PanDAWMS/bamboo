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


def test_an_sql_ask_is_read_by_its_db_read_form_where_its_terms_are_structured():
    form = {
        "table": "ATLAS_PANDA.jobsActive4",
        "where": [
            ["jobStatus", "=", "activated"],
            ["computingSite", "=", "BOINC_MCORE"],
        ],
        "group_by": ["gshare", "currentpriority"],
        "least": "pandaid",
    }
    ask = {
        "kind": "sql",
        "name": "getJobs@2404 groups now",
        "sql": "SELECT ...",
        "binds": {},
    }
    got = _plan({**ask, "db_read": form})
    assert [c.request for c in got.calls] == [
        {
            "tables": ["jobsActive4"],
            "where": form["where"],
            "group_by": ["gshare", "currentpriority"],
            "aggregates": [["count", "*"], ["min", "pandaid"]],
        }
    ]
    text = _plan({**ask, "db_read": {**form, "where": "jobStatus='activated'"}})
    assert not text.calls and "SQL text" in text.why
    window = _plan(ask)
    assert not window.calls and "listed for a hand" in window.why


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
    sections, left_out = RB.answered([_doc(got)], "JediTaskSpec", str(TID), SPELLINGS)
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
    sections, left_out = RB.answered([_doc(got)], "JobSpec", str(PID), SPELLINGS)
    assert left_out == [] and sections == {
        k: fetched[k] for k in ("files", "collections")
    }
    other, why = RB.answered([_doc(got)], "JobSpec", "1", SPELLINGS)
    assert (
        other == {}
        and "no section of the rows file holds the rows of JobSpec.Files" in why[0]
    )


async def test_an_sql_answer_names_rs_aggregates_as_l3s_sql_names_them():
    async def db_read(queries):
        rows = [
            {
                "gshare": "Express",
                "currentpriority": 900,
                "count(*)": 3,
                "min(pandaid)": 11,
            }
        ]
        answer = {
            "status": "done",
            "truncated": False,
            "tables": [{"table": "jobsActive4", "rows": rows}],
        }
        return {"success": True, "data": {"as_of": T(AS_OF), "answers": [answer]}}

    form = {
        "table": "ATLAS_PANDA.jobsActive4",
        "where": [["jobStatus", "=", "activated"]],
        "group_by": ["gshare", "currentpriority"],
        "least": "pandaid",
    }
    line = _line(
        {
            "kind": "sql",
            "name": "getJobs@2404 groups now",
            "sql": "SELECT ...",
            "binds": {},
            "db_read": form,
        }
    )
    got = await RA.answer(
        line, RA.plan(line, TABLES, SITES), FE.Parts(db_read=db_read), TABLES
    )
    sections, _ = RB.answered([_doc(got)], "JobSpec", "1", SPELLINGS)
    answer = sections["answers"]["getJobs@2404 groups now"]
    assert answer["rows"] == [
        {"gshare": "Express", "currentpriority": 900, "n": 3, "min_pandaid": 11}
    ]
    assert answer["clock"] == RB.DATABASE and answer["as_of"] == "2026-10-13 10:00:00"


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
        [_doc(got)], "JediTaskSpec", str(TID), DS_SPELLINGS
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
    sections, _ = RB.answered([_doc(got)], "JediTaskSpec", str(TID), DS_SPELLINGS)
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
    sections, left_out = RB.answered([_doc(got)], "JobSpec", "1", SPELLINGS)
    assert sections == {} and left_out == [
        "sql w: no db_read form (a window question): listed for a hand"
    ]


def test_a_round_of_answers_is_one_past_the_last(tmp_path):
    assert RA.next_round(tmp_path) == 1
    (tmp_path / "rows-r1.json").write_text("{}")
    (tmp_path / "rows-r3.json").write_text("{}")
    (tmp_path / "records-r7.jsonl").write_text("")
    assert RA.next_round(tmp_path) == 4
