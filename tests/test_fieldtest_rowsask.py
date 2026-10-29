"""Field tests -- the plan answering the querier's rows asks (stage 2 of the rows ask).

The asks are in the shapes the querier writes them (querier-ra db6d1c8, ``--rows-asks``); the dump's tables
are test_fieldtest_rows's, and its read sites carry the fields the plan reads (``anchor``, ``keys``,
``condition_trees``), as the map writes them.
"""

from __future__ import annotations

import datetime

from bamboo.scripts.panda.fieldtest import fetch as FE
from bamboo.scripts.panda.fieldtest import rowsask as RA
from tests.test_fieldtest_rows import DUMP

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
